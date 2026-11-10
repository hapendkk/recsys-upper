from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_processed
from src.predict import (
    item_similar_frame,
    load_baseline_bundle,
    load_two_tower_recommender,
    recommendations_frame,
    recommend_for_user,
)

st.set_page_config(page_title="RecSys Beauty", layout="wide")
st.title("RecSys Beauty — Amazon Reviews 2023")
st.caption("Content-based + Item-KNN + Hybrid + Two-Tower recommendation demo")


@st.cache_data(show_spinner=False)
def get_processed():
    return load_processed()


@st.cache_resource(show_spinner=False)
def get_baselines():
    return load_baseline_bundle()


@st.cache_resource(show_spinner=False)
def get_two_tower():
    return load_two_tower_recommender()


interactions, items, users = get_processed()
baselines = get_baselines()
two_tower = get_two_tower()

methods = {
    "Popularity": "popularity",
    "Recent items": "recent",
    "Same brand": "same_brand",
    "Same category": "same_category",
    "Content TF-IDF": "content_tfidf",
    "Item KNN": "item_knn",
    "Hybrid": "hybrid",
    "Two-Tower": "two_tower",
}

with st.sidebar:
    st.header("Controls")
    method_label = st.selectbox("Method", list(methods.keys()))
    method = methods[method_label]
    k = st.slider("Top-K", min_value=5, max_value=25, value=10, step=5)
    history_mode = st.selectbox("History", ["all observed", "train only", "train+valid"], index=0)
    history_splits = {
        "all observed": None,
        "train only": ("train",),
        "train+valid": ("train", "valid"),
    }[history_mode]

    user_options = users.merge(
        interactions.groupby("user_idx").size().rename("interactions").reset_index(),
        on="user_idx",
        how="left",
    ).sort_values("interactions", ascending=False)
    user_options["label"] = (
        user_options["user_id"].astype(str)
        + " ("
        + user_options["interactions"].fillna(0).astype(int).astype(str)
        + " interactions)"
    )
    label_to_idx = dict(zip(user_options["label"], user_options["user_idx"]))
    user_pick = st.selectbox("User", user_options["label"].tolist())
    user_idx = int(label_to_idx[user_pick])

    st.divider()
    st.header("Item seed")
    seed_item_id = st.text_input("Item ASIN / item_id", value="")
    seed_item_idx = None
    if seed_item_id:
        match = items.loc[items["item_id"] == seed_item_id, "item_idx"]
        if len(match):
            seed_item_idx = int(match.iloc[0])
        else:
            st.warning("Item not found in processed catalog")

st.subheader("Dataset")
prep_path = ROOT / "reports" / "data_prep_report.json"
if prep_path.exists():
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    cols = st.columns(4)
    cols[0].metric("Interactions", f"{prep.get('processed_rows', 0):,}")
    cols[1].metric("Users", f"{prep.get('processed_users', 0):,}")
    cols[2].metric("Items", f"{prep.get('processed_items', 0):,}")
    cols[3].metric("Positive rating", f">= {prep.get('min_positive_rating', 'n/a')}")
else:
    st.info("Run `python -m src.data_prep` to generate dataset report.")

if method in {"two_tower", "two_tower_hybrid"} and two_tower is None:
    st.warning("Two-Tower checkpoint not found. Run `python -m src.train_two_tower` first.")
    st.stop()

try:
    rec_items, scores = recommend_for_user(
        method=method,
        user_idx=user_idx,
        interactions=interactions,
        items=items,
        users=users,
        k=k,
        history_splits=history_splits,
        baseline_bundle=baselines,
        two_tower=two_tower,
    )
    rec_df = recommendations_frame(rec_items, scores, items)
except Exception as exc:
    st.error(f"Recommendation failed: {exc}")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Recommendations for {user_pick}")
    if rec_df.empty:
        st.info("No recommendations returned. Try another user/history mode.")
    else:
        show_df = rec_df[["rank", "title", "brand", "category", "price", "avg_rating", "score"]].copy()
        show_df["score"] = show_df["score"].round(4)
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        for _, row in rec_df.head(5).iterrows():
            with st.expander(f"#{int(row['rank'])} {row['title']}"):
                st.write(row["title"])
                st.write(f"Brand: {row['brand']} | Category: {row['category']} | Price: {row['price']}")
                if row.get("image_url"):
                    st.image(row["image_url"], width=180)

with right:
    st.subheader("User history")
    hist_items, hist_weights = [], []
    subset = interactions[interactions["user_idx"] == user_idx]
    if history_splits is not None:
        subset = subset[subset["split"].isin(list(history_splits))]
    subset = subset.sort_values(["timestamp", "item_idx"], kind="mergesort").tail(20)
    hist_lookup = items.set_index("item_idx")
    for row in subset.itertuples(index=False):
        item = hist_lookup.loc[int(row.item_idx)]
        st.markdown(f"- {item['title']} — rating {row.rating}")

    st.subheader("Offline metrics")
    metric_frames = []
    for path in [ROOT / "reports" / "baseline_metrics.csv", ROOT / "reports" / "two_tower_metrics.csv"]:
        if path.exists():
            metric_frames.append(pd.read_csv(path))
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        if "eval_split" in metrics.columns:
            test_metrics = metrics[metrics["eval_split"] == "test"].copy()
        else:
            test_metrics = metrics.copy()
        if "recall@10" in test_metrics.columns and not test_metrics.empty:
            test_metrics = test_metrics.sort_values("recall@10", ascending=False)
            cols = [c for c in ["method", "recall@10", "ndcg@10", "hit@10", "coverage", "novelty"] if c in test_metrics.columns]
            st.dataframe(test_metrics[cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(metrics, use_container_width=True)
    else:
        st.info("No metrics yet. Run baseline/two-tower training scripts.")

if seed_item_idx is not None:
    st.divider()
    st.subheader("Similar items to seed")
    seed_row = items.loc[items["item_idx"] == seed_item_idx].iloc[0]
    st.write(f"Seed: {seed_row['title']} ({seed_row['brand']})")
    for sim_method in ["content_tfidf", "item_knn", "hybrid"]:
        sim_df = item_similar_frame(seed_item_idx, sim_method, items, baseline_bundle=baselines, k=min(k, 10))
        st.markdown(f"**{sim_method}**")
        st.dataframe(
            sim_df[["rank", "title", "brand", "category", "score"]].head(10),
            use_container_width=True,
            hide_index=True,
        )
