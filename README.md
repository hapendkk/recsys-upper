# RecSys Upper — Amazon Beauty

Рекомендательная система для e-commerce на датасете **Amazon Reviews 2023 / All_Beauty** (McAuley Lab).

## Что внутри

- `src/data_prep.py` — загрузка и подготовка `rating_only` + metadata, support-фильтрация, temporal split train/valid/test.
- `src/recommenders.py` — классические методы: popularity, recent, same brand/category, content-based TF-IDF, item-item KNN, hybrid.
- `src/two_tower.py` + `src/train_two_tower.py` — modern Two-Tower модель (user tower + item tower с текстовыми SVD-признаками).
- `src/evaluate.py`, `src/metrics.py` — Recall@K, HitRate@K, NDCG@K, MRR, coverage, novelty.
- `app/streamlit_app.py` — Streamlit UI для сравнения рекомендаций.
- `reports/` — метрики и отчётные материалы.
- `REPORT.md` — полный отчёт: метрики, cold start, проблемы подходов, продакшен-риски и бизнес/ML-метрики.

## Быстрый старт

```bash
python -m pip install -r requirements.txt
python -m src.data_prep
python -m src.train_baselines
python -m src.train_two_tower --epochs 30 --batch-size 128 --n-random-negatives 256 --eval-max-users 4000 --output two_tower.pt --train-splits train --no-user-id-emb --normalize-embeddings --embedding-init-std 0.02 --dropout 0.1
python -m src.cold_start_analysis --methods content_tfidf item_knn hybrid two_tower --k 10
python -m src.tune_hybrid --max-users 1000 --step 0.2 --k 10
python -m src.significance --max-users 2000 --k 10 --bootstraps 2000
python -m pytest -q
streamlit run app/streamlit_app.py
```

Датасет скачивается автоматически с Hugging Face в `data/raw/` (~250 MB), обработанные parquet — в `data/processed/`.

## Итоговые offline-метрики (test, Recall@10)

| method | Recall@10 | NDCG@10 | coverage | novelty |
|---|---:|---:|---:|---:|
| popularity | 0.0105 | 0.0061 | 0.0017 | 6.909 |
| recent items | 0.0030 | 0.0011 | 0.0014 | 8.997 |
| same_brand | 0.0580 | 0.0373 | 0.3795 | 7.760 |
| same_category | 0.0096 | 0.0049 | 0.0184 | 7.244 |
| content_tfidf | 0.0832 | 0.0549 | 0.8416 | 9.071 |
| item_knn | 0.0245 | 0.0154 | 0.3902 | 8.947 |
| hybrid | **0.0913** | **0.0574** | 0.8110 | 8.799 |
| two_tower | 0.0269 | 0.0157 | 0.4614 | 8.745 |

Вывод: на этом разреженном срезе лучший вариант — `hybrid` (`content 0.65 / item-KNN 0.25 / popularity 0.10`). Two-Tower реализован и обучен, но на отзывах Amazon Beauty показывает слабый full-catalog recall; подробности и продакшен-выводы — в `REPORT.md`.

## Фильтрация данных

Amazon Beauty 2023 — 0-core и очень разреженный: у большинства пользователей только один отзыв. Строгое k-core схлопывает данные, поэтому используется support-фильтр:

- positive interaction: `rating >= 3`
- у пользователя минимум 3 взаимодействия
- у товара минимум 5 взаимодействий

Получается ~19.5k взаимодействий, ~5.7k пользователей, ~7.6k товаров.
