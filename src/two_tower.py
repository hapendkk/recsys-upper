from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import MODEL_DIR
from .evaluate import histories_by_user
from .metrics import evaluate_recommendation_list


@dataclass
class ItemFeatures:
    brand_to_idx: dict[str, int]
    cat_to_idx: dict[str, int]
    item_brand_idx: np.ndarray
    item_cat_idx: np.ndarray
    item_numeric: np.ndarray


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_item_features(items: pd.DataFrame, text_dim: int = 64) -> ItemFeatures:
    item_frame = items.sort_values("item_idx").reset_index(drop=True)
    brand_values = set(item_frame["brand"].fillna("Unknown").astype(str))
    brand_values.add("Unknown")
    cat_values = set(item_frame["primary_category"].fillna("Unknown").astype(str))
    cat_values.add("Unknown")
    brands = sorted(brand_values)
    categories = sorted(cat_values)
    brand_to_idx = {brand: idx for idx, brand in enumerate(brands)}
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    brand_idx = item_frame["brand"].fillna("Unknown").astype(str).map(brand_to_idx).fillna(0).to_numpy(dtype=np.int64)
    cat_idx = item_frame["primary_category"].fillna("Unknown").astype(str).map(cat_to_idx).fillna(0).to_numpy(dtype=np.int64)

    avg_rating = item_frame["average_rating"].map(lambda x: _safe_float(x, 0.0)).to_numpy(dtype=np.float32) / 5.0
    rating_number = np.log1p(item_frame["rating_number"].map(lambda x: _safe_float(x, 0.0)).to_numpy(dtype=np.float32)) / 10.0
    price = item_frame["price"].map(lambda x: _safe_float(x, np.nan)).to_numpy(dtype=np.float32)
    has_price = np.isfinite(price).astype(np.float32)
    log_price = np.where(np.isfinite(price), np.log1p(np.maximum(price, 0.0)) / 10.0, 0.0).astype(np.float32)
    numeric = np.column_stack([avg_rating, rating_number, has_price, log_price]).astype(np.float32)

    corpus = item_frame["text_blob"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    ).fit_transform(corpus)
    n_components = int(min(text_dim, max(tfidf.shape[1] - 1, 0), max(len(item_frame) - 1, 0)))
    if n_components > 0:
        text_features = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(tfidf).astype(np.float32)
        norms = np.linalg.norm(text_features, axis=1, keepdims=True)
        text_features = text_features / np.maximum(norms, 1e-6)
    else:
        text_features = np.zeros((len(item_frame), 0), dtype=np.float32)
    numeric = np.column_stack([numeric, text_features]).astype(np.float32)

    return ItemFeatures(
        brand_to_idx=brand_to_idx,
        cat_to_idx=cat_to_idx,
        item_brand_idx=brand_idx,
        item_cat_idx=cat_idx,
        item_numeric=numeric,
    )


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_brands: int,
        n_categories: int,
        user_emb_dim: int = 32,
        item_emb_dim: int = 32,
        brand_emb_dim: int = 16,
        cat_emb_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 64,
        numeric_dim: int = 4,
        use_user_id_emb: bool = True,
        normalize_embeddings: bool = False,
        embedding_init_std: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.numeric_dim = int(numeric_dim)
        self.use_user_id_emb = bool(use_user_id_emb)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.item_numeric: torch.Tensor | None = None
        self.user_id_emb = nn.Embedding(n_users, user_emb_dim) if self.use_user_id_emb else None
        self.item_id_emb = nn.Embedding(n_items, item_emb_dim)
        self.brand_emb = nn.Embedding(n_brands, brand_emb_dim)
        self.cat_emb = nn.Embedding(n_categories, cat_emb_dim)
        for emb in [self.user_id_emb, self.item_id_emb, self.brand_emb, self.cat_emb]:
            if emb is not None:
                nn.init.normal_(emb.weight, mean=0.0, std=float(embedding_init_std))
        user_in_dim = (user_emb_dim if self.use_user_id_emb else 0) + item_emb_dim + self.numeric_dim
        self.user_tower = nn.Sequential(
            nn.Linear(user_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_emb_dim + brand_emb_dim + cat_emb_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_items(
        self,
        item_ids: torch.Tensor,
        brand_ids: torch.Tensor,
        cat_ids: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.item_id_emb(item_ids), self.brand_emb(brand_ids), self.cat_emb(cat_ids), numeric], dim=1)
        out = self.item_tower(x)
        return F.normalize(out, dim=-1) if self.normalize_embeddings else out

    def encode_users(self, user_ids: torch.Tensor, history_item_ids: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        hist_emb = self.item_id_emb(history_item_ids)
        mask = history_mask.unsqueeze(-1).to(hist_emb.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        hist_mean = (hist_emb * mask).sum(dim=1) / denom
        if self.item_numeric is not None:
            item_numeric = self.item_numeric.to(device=history_item_ids.device, dtype=hist_emb.dtype)
            hist_numeric = item_numeric[history_item_ids]
            hist_numeric_mean = (hist_numeric * mask).sum(dim=1) / denom
        else:
            hist_numeric_mean = torch.zeros(
                user_ids.shape[0], self.numeric_dim, device=user_ids.device, dtype=hist_emb.dtype
            )
        parts = []
        if self.use_user_id_emb and self.user_id_emb is not None:
            parts.append(self.user_id_emb(user_ids))
        parts.extend([hist_mean, hist_numeric_mean])
        user_vec = torch.cat(parts, dim=1)
        out = self.user_tower(user_vec)
        return F.normalize(out, dim=-1) if self.normalize_embeddings else out

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def _pad_histories(histories: list[list[int]], max_history: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((len(histories), max_history), dtype=np.int64)
    mask = np.zeros((len(histories), max_history), dtype=np.float32)
    for i, hist in enumerate(histories):
        hist = [int(x) for x in hist if int(x) >= 0]
        if len(hist) > max_history:
            hist = hist[-max_history:]
        if hist:
            padded[i, -len(hist) :] = hist
            mask[i, -len(hist) :] = 1.0
    return padded, mask


def build_training_examples(interactions: pd.DataFrame, max_history: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")
    user_ids: list[int] = []
    target_ids: list[int] = []
    histories: list[list[int]] = []
    for user_idx, group in df.groupby("user_idx", sort=False):
        item_seq = group["item_idx"].astype(int).tolist()
        for pos in range(1, len(item_seq)):
            user_ids.append(int(user_idx))
            target_ids.append(int(item_seq[pos]))
            histories.append(item_seq[max(0, pos - max_history) : pos])
    hist_ids, hist_mask = _pad_histories(histories, max_history)
    return (
        np.asarray(user_ids, dtype=np.int64),
        np.asarray(target_ids, dtype=np.int64),
        hist_ids,
        hist_mask,
    )


def build_eval_examples(interactions: pd.DataFrame, target_split: str, history_splits: Sequence[str], max_history: int = 50) -> tuple[np.ndarray, list[list[int]]]:
    targets = interactions[interactions["split"] == target_split]
    histories = histories_by_user(interactions, history_splits)
    user_ids: list[int] = []
    hist_lists: list[list[int]] = []
    for row in targets.sort_values(["user_idx", "timestamp"])[["user_idx"]].itertuples(index=False):
        user_idx = int(row.user_idx)
        user_ids.append(user_idx)
        hist_lists.append([item for item, _ in histories.get(user_idx, [])])
    hist_ids, hist_mask = _pad_histories(hist_lists, max_history)
    return np.asarray(user_ids, dtype=np.int64), hist_ids, hist_mask, hist_lists


def train_two_tower(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
    user_emb_dim: int = 32,
    item_emb_dim: int = 32,
    hidden_dim: int = 128,
    out_dim: int = 64,
    max_history: int = 50,
    n_random_negatives: int = 256,
    max_examples: int | None = None,
    seed: int = 42,
    device: str = "cpu",
    train_splits: Sequence[str] = ("train",),
    use_user_id_emb: bool = True,
    normalize_embeddings: bool = False,
    embedding_init_std: float = 0.02,
    dropout: float = 0.0,
) -> tuple[TwoTowerModel, ItemFeatures, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    features = build_item_features(items)
    train = interactions[interactions["split"].isin(list(train_splits))]
    user_ids, target_ids, hist_ids, hist_mask = build_training_examples(train, max_history=max_history)
    if max_examples is not None and len(user_ids) > max_examples:
        idx = np.random.choice(len(user_ids), size=max_examples, replace=False)
        idx.sort()
        user_ids, target_ids, hist_ids, hist_mask = user_ids[idx], target_ids[idx], hist_ids[idx], hist_mask[idx]

    n_users = int(users["user_idx"].max()) + 1 if len(users) else int(train["user_idx"].max()) + 1
    n_items = int(items["item_idx"].max()) + 1
    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        n_brands=len(features.brand_to_idx),
        n_categories=len(features.cat_to_idx),
        user_emb_dim=user_emb_dim,
        item_emb_dim=item_emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        numeric_dim=features.item_numeric.shape[1],
        use_user_id_emb=use_user_id_emb,
        normalize_embeddings=normalize_embeddings,
        embedding_init_std=embedding_init_std,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history = {"train_loss": []}

    item_brand_t = torch.as_tensor(np.array(features.item_brand_idx, copy=True), dtype=torch.long, device=device)
    item_cat_t = torch.as_tensor(np.array(features.item_cat_idx, copy=True), dtype=torch.long, device=device)
    item_numeric_t = torch.as_tensor(np.array(features.item_numeric, copy=True), dtype=torch.float32, device=device)
    model.item_numeric = item_numeric_t

    n_examples = len(user_ids)
    for epoch in range(epochs):
        order = rng.permutation(n_examples)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n_examples, batch_size):
            batch_idx = order[start : start + batch_size]
            b_user = torch.as_tensor(user_ids[batch_idx], dtype=torch.long, device=device)
            b_target = torch.as_tensor(target_ids[batch_idx], dtype=torch.long, device=device)
            b_hist = torch.as_tensor(hist_ids[batch_idx], dtype=torch.long, device=device)
            b_mask = torch.as_tensor(hist_mask[batch_idx], dtype=torch.float32, device=device)

            user_emb = model.encode_users(b_user, b_hist, b_mask)
            neg_ids = torch.randint(0, n_items, (n_random_negatives,), device=device)
            cand_ids = torch.cat([b_target, neg_ids], dim=0)
            cand_brand = item_brand_t[cand_ids]
            cand_cat = item_cat_t[cand_ids]
            cand_numeric = item_numeric_t[cand_ids]
            cand_emb = model.encode_items(cand_ids, cand_brand, cand_cat, cand_numeric)
            logits = user_emb @ cand_emb.T * model.scale()
            labels = torch.arange(len(batch_idx), device=device)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        avg_loss = epoch_loss / max(steps, 1)
        history["train_loss"].append(avg_loss)
        print(f"epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    config = {
        "n_users": n_users,
        "n_items": n_items,
        "n_brands": len(features.brand_to_idx),
        "n_categories": len(features.cat_to_idx),
        "user_emb_dim": user_emb_dim,
        "item_emb_dim": item_emb_dim,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "numeric_dim": int(features.item_numeric.shape[1]),
        "max_history": max_history,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_random_negatives": n_random_negatives,
        "train_examples": int(n_examples),
        "train_splits": list(train_splits),
        "use_user_id_emb": bool(use_user_id_emb),
        "normalize_embeddings": bool(normalize_embeddings),
        "embedding_init_std": float(embedding_init_std),
        "dropout": float(dropout),
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config,
        "item_brand_idx": features.item_brand_idx,
        "item_cat_idx": features.item_cat_idx,
        "item_numeric": features.item_numeric,
        "train_history": history,
    }
    return model, features, checkpoint


def save_two_tower(checkpoint: dict[str, Any], path: Path | None = None) -> Path:
    path = path or (MODEL_DIR / "two_tower.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class TwoTowerRecommender:
    name = "two_tower"

    def __init__(self, checkpoint_path: Path | str | None = None, device: str = "cpu", batch_size: int = 256):
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (MODEL_DIR / "two_tower.pt")
        self._emb_path = self.checkpoint_path.with_name(self.checkpoint_path.stem + "_item_emb.npy")
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.config = checkpoint["config"]
        model_kwargs = {k: self.config[k] for k in ["n_users", "n_items", "n_brands", "n_categories", "user_emb_dim", "item_emb_dim", "hidden_dim", "out_dim", "numeric_dim"]}
        model_kwargs.update(
            {
                "use_user_id_emb": self.config.get("use_user_id_emb", True),
                "normalize_embeddings": self.config.get("normalize_embeddings", False),
                "embedding_init_std": self.config.get("embedding_init_std", 0.02),
                "dropout": self.config.get("dropout", 0.0),
            }
        )
        self.model = TwoTowerModel(**model_kwargs).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.n_items = int(self.config["n_items"])
        self.max_history = int(self.config.get("max_history", 50))
        self.item_brand = torch.as_tensor(checkpoint["item_brand_idx"], dtype=torch.long, device=device)
        self.item_cat = torch.as_tensor(checkpoint["item_cat_idx"], dtype=torch.long, device=device)
        self.item_numeric = torch.as_tensor(np.array(checkpoint["item_numeric"], copy=True), dtype=torch.float32, device=device)
        self.model.item_numeric = self.item_numeric
        self._item_embeddings: np.ndarray | None = None

    @property
    def item_embeddings(self) -> np.ndarray:
        if self._item_embeddings is None:
            emb_path = self._emb_path
            if emb_path.exists():
                self._item_embeddings = np.load(emb_path)
            else:
                self._item_embeddings = self._compute_item_embeddings()
                np.save(emb_path, self._item_embeddings)
        return self._item_embeddings

    @torch.no_grad()
    def _compute_item_embeddings(self) -> np.ndarray:
        all_emb = []
        for start in range(0, self.n_items, self.batch_size):
            end = min(start + self.batch_size, self.n_items)
            ids = torch.arange(start, end, device=self.device, dtype=torch.long)
            emb = self.model.encode_items(ids, self.item_brand[ids], self.item_cat[ids], self.item_numeric[ids])
            all_emb.append(emb.cpu().numpy().astype(np.float32))
        return np.vstack(all_emb)

    @torch.no_grad()
    def encode_users(self, user_ids: np.ndarray, histories: list[list[int]]) -> np.ndarray:
        hist_ids, hist_mask = _pad_histories(histories, self.max_history)
        out = []
        for start in range(0, len(user_ids), self.batch_size):
            end = min(start + self.batch_size, len(user_ids))
            users_t = torch.as_tensor(user_ids[start:end], dtype=torch.long, device=self.device)
            hist_t = torch.as_tensor(hist_ids[start:end], dtype=torch.long, device=self.device)
            mask_t = torch.as_tensor(hist_mask[start:end], dtype=torch.float32, device=self.device)
            out.append(self.model.encode_users(users_t, hist_t, mask_t).cpu().numpy().astype(np.float32))
        return np.vstack(out)

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        raise NotImplementedError("Use recommend/recommend_many for TwoTower inference")

    def _scores_for_histories(self, user_ids: np.ndarray, histories: list[list[int]]) -> np.ndarray:
        user_emb = self.encode_users(user_ids, histories)
        item_emb = self.item_embeddings
        scores = user_emb @ item_emb.T
        return scores.astype(np.float32)

    def recommend_many(self, requests: list[tuple[int, list[tuple[int, float]], int]], k: int = 10) -> list[np.ndarray]:
        user_ids = np.asarray([req[0] for req in requests], dtype=np.int64)
        histories = [[item for item, _ in req[1]] for req in requests]
        result: list[np.ndarray] = []
        for start in range(0, len(requests), self.batch_size):
            end = min(start + self.batch_size, len(requests))
            batch_users = user_ids[start:end]
            batch_histories = histories[start:end]
            user_emb = self.encode_users(batch_users, batch_histories)
            item_emb = self.item_embeddings
            scores = (user_emb @ item_emb.T).astype(np.float32)
            for local_idx, (_, history_pairs, _) in enumerate(requests[start:end]):
                exclude = {int(item) for item, _ in history_pairs}
                row = scores[local_idx].copy()
                if exclude:
                    row[list(exclude)] = -np.inf
                k_eff = min(k, self.n_items)
                part = np.argpartition(-row, kth=k_eff - 1)[:k_eff]
                part = part[np.argsort(-row[part], kind="mergesort")]
                result.append(part.astype(np.int32))
        return result

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        requests = [(int(user_idx), [(int(i), 1.0) for i in history], -1)]
        recs = self.recommend_many(requests, k=k)[0]
        if exclude:
            extra = set(exclude) - set(int(i) for i in history)
            if extra:
                user_emb = self.encode_users(np.array([user_idx], dtype=np.int64), [list(history)])
                scores = (user_emb @ self.item_embeddings.T).astype(np.float32)[0]
                scores[list(extra)] = -np.inf
                scores[list(history)] = -np.inf
                k_eff = min(k, self.n_items)
                part = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
                part = part[np.argsort(-scores[part], kind="mergesort")]
                return part.astype(np.int32)
        return recs


def save_item_embeddings(recommender: TwoTowerRecommender, path: Path | None = None) -> Path:
    path = path or (MODEL_DIR / "two_tower_item_emb.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, recommender.item_embeddings)
    return path
