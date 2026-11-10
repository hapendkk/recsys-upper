# RecSys Upper — отчёт по рекомендательной системе

## 1. Концепт и выбранный сервис

Сервис: **e-commerce, категория Beauty (Amazon Reviews 2023 / All_Beauty, McAuley Lab)**.

Бизнес-цель: повысить персонализацию каталога и CX/конверсию за счёт блоков «Вам также может быть интересно», «С этим товаром покупают», «Новинки», «Популярное».

Ключевая сложность датасета: Amazon Reviews 2023 — это **0-core** данные. У большинства пользователей только один отзыв, поэтому классическое k-core очень быстро схлопывает выборку. После проверки я использовал **одноразовый support-фильтр** (не итеративный k-core):

- positive interaction: `rating >= 3`;
- у пользователя минимум 3 взаимодействия;
- у товара минимум 5 взаимодействий;
- дубликаты `user-item` оставляем по последнему timestamp.

Итог: **19 492 взаимодействия, 5 738 пользователей, 7 599 товаров**. Split — **per-user temporal holdout** (не глобальный temporal split): у каждого пользователя с >=3 взаимодействиями последний товар — test, предпоследний — valid, остальные — train.

## 2. Что реализовано

### Классические методы

- `popularity` — топ по числу взаимодействий;
- `recent items` — глобальная новизна по последнему timestamp (не «недавно просмотренные» пользователя: в данных нет view-событий);
- `same_brand` / `same_category` — эвристики по атрибутам; `same_category` использует `category_path`, а при вырождении категории (`All Beauty`/`Premium Beauty`) — pseudo-category по ключевым словам title/description/features;
- `content_tfidf` — content-based: TF-IDF по `title + category_path + brand + description + features`, пользовательский профиль = weighted mean TF-IDF истории;
- `item_knn` — item-based CF: cosine similarity по user-item матрице, top-100 соседей на товар;
- `hybrid` — weighted fusion content + item-KNN + popularity. Веса подобраны на valid: **content 0.65 / item-KNN 0.25 / popularity 0.10**.

Формула hybrid:

```python
score(item) = 0.65 * norm(content_tfidf(item)) \
            + 0.25 * norm(item_knn(item)) \
            + 0.10 * norm(popularity(item))
```

### Современная модель

Two-Tower recommender:

- **user tower**: user ID embedding + mean item ID embedding истории + mean item numeric/text-SVD признаков истории;
- **item tower**: item ID embedding + brand/category embeddings + numeric features + TF-IDF → TruncatedSVD(64) текстовый профиль;
- обучение: sampled softmax, in-batch targets + random negatives, AdamW, cosine-style dot-product scoring;
- inference: dot-product user embedding × all item embeddings.

```text
user tower:  user_id_emb + mean(item_id_emb(history)) + mean(item_numeric/text_svd(history))
item tower:  item_id_emb + brand_emb + category_emb + numeric + text_svd
score:       <user_emb, item_emb>  ->  top-k по всему каталогу
```

Также проверялся вариант `two_tower + rerank` эвристиками; он не улучшил результат, потому что чистый Two-Tower слишком слабый на этом датасете.

### Ресёрч современных подходов

- **YouTube recommendation (Two-Tower/DNN retrieval)** — классический пример отдельного candidate generation и ranking; Two-Tower естественно ложится на задачу «user → item» и масштабируется через embeddings/ANN.
- **Pinterest PinSage** — graph-based recommendation для визуального каталога; показал, что для item-centric каталогов важны side information и neighbourhood-агрегация, но GNN сложнее и тяжелее для CPU-демо.
- **Industrial two-tower / embedding retrieval** — во многих продакшен-системах retrieval делается через dense embeddings + approximate nearest neighbors (FAISS/HNSW), а точный ranking отдают более дешёвым/объяснимым моделям.

Выбор Two-Tower: это компромисс между «модностью», реализуемостью на CPU и объяснимой двухстадийной продакшен-схемой. GNN/LLM оставлены как направления для расширения при появлении более плотных графов/текстов.

## 3. Offline-метрики

Метрики на одном held-out target на пользователя. `Recall@10 = HitRate@10`, т.к. target один.

```python
hit@k      = 1[target in top_k]
recall@k   = hit@k                      # один релевантный target
mrr@k      = 1 / rank                   # если rank <= k, иначе 0
ndcg@k     = 1 / log2(rank + 1)         # gain=1 для единственного target
```

### Valid (`history=train`)

| method | Recall@10 | NDCG@10 | Hit@10 | MRR@10 | coverage | novelty |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.0138 | 0.0085 | 0.0138 | 0.0069 | 0.0017 | 6.928 |
| recent items | 0.0003 | 0.0001 | 0.0003 | 0.0001 | 0.0014 | 8.863 |
| same_brand | 0.0560 | 0.0359 | 0.0560 | 0.0295 | 0.3321 | 7.542 |
| same_category | 0.0114 | 0.0062 | 0.0114 | 0.0047 | 0.0183 | 7.252 |
| content_tfidf | 0.0706 | 0.0475 | 0.0706 | 0.0403 | 0.8712 | 9.092 |
| item_knn | 0.0296 | 0.0173 | 0.0296 | 0.0136 | 0.3477 | 8.974 |
| hybrid | **0.0859** | **0.0557** | **0.0859** | **0.0463** | 0.8172 | 8.694 |
| two_tower | 0.0341 | 0.0200 | 0.0341 | 0.0159 | 0.4824 | 8.708 |

### Test (`history=train+valid`)

| method | Recall@10 | NDCG@10 | Hit@10 | MRR@10 | coverage | novelty |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.0105 | 0.0061 | 0.0105 | 0.0048 | 0.0017 | 6.909 |
| recent items | 0.0030 | 0.0011 | 0.0030 | 0.0005 | 0.0014 | 8.997 |
| same_brand | 0.0580 | 0.0373 | 0.0580 | 0.0308 | 0.3795 | 7.760 |
| same_category | 0.0096 | 0.0049 | 0.0096 | 0.0035 | 0.0184 | 7.244 |
| content_tfidf | 0.0832 | 0.0549 | 0.0832 | 0.0462 | 0.8416 | 9.071 |
| item_knn | 0.0245 | 0.0154 | 0.0245 | 0.0126 | 0.3902 | 8.947 |
| hybrid | **0.0913** | **0.0574** | **0.0913** | **0.0470** | 0.8110 | 8.799 |
| two_tower | 0.0269 | 0.0157 | 0.0269 | 0.0124 | 0.4614 | 8.745 |

**Вывод по качеству:** на этом разреженном срезе лучший offline-результат даёт `hybrid`, в основном за счёт content-based сигнала. `item_knn` заметно слабее — item-item граф слишком разрежен. `two_tower` v2 после исправлений архитектуры даёт Recall@10 = 0.0269 — уже не провал, но всё ещё ~в 3.4 раза ниже hybrid: sampled-softmax и короткая история не компенсируют разреженность.

## 4. Cold start и бизнес-метрики

### Recall@10 по размеру истории пользователя (test)

| method | history <=2 | 3–5 | 6–10 | 11+ |
|---|---:|---:|---:|---:|
| content_tfidf | 0.0919 | 0.0796 | 0.0515 | 0.0375 |
| item_knn | 0.0258 | 0.0280 | 0.0043 | 0.0188 |
| hybrid | **0.0987** | **0.0930** | 0.0515 | 0.0438 |
| two_tower | 0.0209 | 0.0370 | 0.0300 | 0.0438 |

Интерпретация: это срез **short history**, а не полный cold start с 0–1 событием: в test-оценке минимальная история = 2 (`train+valid` при `min_user_interactions=3`). Гибрид лучше всего работает в short-history сегменте, потому что content/brand/category не требуют длинной истории. Two-Tower v2 относительно лучше при `11+` (0.0438, почти как hybrid), но всё ещё слаб на `<=2` (0.0209). При длинной истории качество content/hybrid падает — вероятно, из-за того, что старые интересы пользователя уже «исчерпаны», а held-out target относится к новой потребности.

### Recall@10 по популярности целевого товара (test)

| method | target popularity <=2 | 3–5 | 6–10 | 11+ |
|---|---:|---:|---:|---:|
| content_tfidf | 0.0923 | 0.0668 | 0.0796 | 0.0690 |
| item_knn | 0.0108 | 0.0327 | 0.0524 | **0.0443** |
| hybrid | 0.0923 | 0.0764 | 0.0958 | **0.1232** |
| two_tower | 0.0108 | 0.0396 | 0.0452 | 0.0788 |

Гибрид лучше всего предсказывает популярные товары; для редких товаров content-based всё ещё держится лучше item-KNN. Two-Tower v2 показывает второй результат на популярных целях (`11+`: 0.0788), но почти не помогает на редких (`<=2`: 0.0108). Это ожидаемо: item-KNN и Two-Tower нужны co-occurrence/плотные сигналы, которых у редких товаров мало.

### Каталожные ML-метрики и бизнес-прокси

- **Coverage** — ML/каталожная метрика: доля каталога, которую система способна рекомендовать. У `popularity` coverage = 0.0017 — почти всегда один и тот же топ, плохо для длинного хвоста.
- **Novelty** — ML/каталожная метрика: средний `-log((count+1)/(N+items))` рекомендованных товаров. После smoothing у `content_tfidf` novelty ≈ 9.07, у `hybrid` ≈ 8.80.
- **Cold-start сегменты** выше показывают, что для новых пользователей нужен content/attribute fallback, а не чистый CF.
- Coverage/novelty **не являются прямыми бизнес-метриками**. Бизнес-метрики оцениваются только через прокси: `Recall@10/HitRate@10` → CTR/add-to-cart, coverage/novelty/long-tail share → разнообразие и incremental revenue, а не выручка напрямую.

## 5. Проблемы подходов и варианты решений

### 5.1 Холодный старт пользователя

**Проблема.** У нового пользователя нет истории, CF и Two-Tower почти не работают.

**Решение.** Content/attribute fallback: первые рекомендации строить по онбординг-опросу (категории, бренд, бюджет, цель), popularity для совсем пустого профиля, gradual personalization по мере событий.

**Недостаток решения.** Онбординг увеличивает friction и может поднять ранний churn; popularity без personalization быстро приедается.

**Рассуждение.** В наших данных короткая история (1–2 события) — самый большой сегмент test (2057/3342), и именно там hybrid показывает лучший Recall@10 (0.0987). Значит, для бизнеса важнее сделать качественный cold-start блок, чем гнаться за сложной нейросетью. Лучшие практики: минимальный опрос + сразу полезная выдача + плавный переход к персонализации.

### 5.2 Холодный старт товара / long-tail

**Проблема.** У новых и редких товаров мало взаимодействий, item-KNN и collaborative-модели их недооценивают.

**Решение.** Content-based features (title/description/brand/SVD-text) и attribute boosts; для совсем новых товаров — exploration slot в выдаче.

**Недостаток решения.** Exploration может показывать нерелевантные товары и снижать краткосрочный CTR.

**Рассуждение.** В cold-start по популярности товара hybrid всё равно лучше item-KNN на редких товарах (0.092 vs 0.011 для popularity 1–2), потому что опирается на контент. В проде стоит держать отдельный «exploration budget», но измерять его по incremental conversion, а не по CTR.

### 5.3 Разреженность и слабые повторные пользователи

**Проблема.** Amazon Beauty 0-core: по всем рейтингам 589 163 из 631 986 пользователей имеют ровно один отзыв (93.2%); среди positive `rating>=3` — 472 151 из 503 701 (93.7%). После фильтра остаётся всего 5.7k пользователей и 19.5k взаимодействий; статистика добавлена в `reports/data_prep_report.json`.

**Решение.** Использовать все сигналы (view/click/cart/purchase), а не только reviews; понижать порог support-фильтра для content-моделей; использовать side information.

**Недостаток решения.** Больше сигналов = больше шума и дороже сбор/хранение; слишком мягкий фильтр вернёт холодный старт обратно.

**Рассуждение.** Низкие абсолютные метрики (Recall@10 ≈ 0.09) — не обязательно провал модели, а свойство задачи: один held-out отзыв предсказать очень сложно. В отчёте бизнесу нужно сравнивать с popularity/baseline и смотреть на uplift, а не на абсолютные значения.

### 5.4 Popularity bias и фильтр-пузырь

**Проблема.** popularity/same_brand/same_category и даже hybrid с popularity-компонентой усиливают уже популярные бренды/категории.

**Решение.** Ограничить долю popularity, добавить novelty/diversity rerank, brand/category caps, MMR/intent-aware diversification.

**Недостаток решения.** Слишком сильная диверсификация может снизить релевантность и конверсию.

**Рассуждение.** В моём hybrid weights content 0.65 / item 0.25 / popularity 0.10 уже сдвинуты к контенту; coverage 0.81 и novelty ≈ 8.80 лучше, чем у чистого popularity (coverage 0.0017). Для продакшена я бы добавил MMR/diversity rerank и A/B-тестировал влияние на long-tail revenue.

### 5.5 Этические/контентные риски

**Проблема.** «С этим товаром покупают» и brand/category blocks могут усиливать нежелательные стереотипы (например, навязывать гендерные/возрастные категории) или рекомендовать сомнительные товары.

**Решение.** Фильтрация запрещённых категорий/брендов, fairness metrics по демографическим прокси, human review для sensitive categories, explainability («почему показано»).

**Недостаток решения.** Жёсткие фильтры могут ухудшить релевантность; fairness-метрики трудно интерпретировать без разметки.

**Рассуждение.** Для Beauty-каталога риск ниже, чем для job/credit recsys, но фильтры по запрещённому контенту и логирование объяснений обязательны. В UI я показываю brand/category/score, что уже даёт базовую explainability.

### 5.6 Two-Tower: сильный скачок после исправлений, но всё ещё ниже hybrid

**Проблема.** Первая версия Two-Tower обучалась (loss падал), но на полном каталоге Recall@10 ≈ 0.004 — в ~20 раз хуже hybrid.

**Решение.** Для v2 добавлены: нормализация user/item embeddings, меньшая инициализация ID-эмбеддингов (`std=0.02`), dropout 0.1, отключение `user_id_emb` (пользовательские ID переобучались), batch size 128 и 30 эпох. Test Recall@10 вырос до 0.0269, coverage до 0.461, novelty 8.745.

**Недостатки решения.** Улучшение всё ещё не делает Two-Tower лучшим методом: на 19k позитивных взаимодействий content-сигнал доминирует, а sampled-softmax с 256 негативами остаётся грубым приближением full-catalog ranking. Валидация по эпохам и early stopping не реализованы — эпохи выбраны по loss, а не по Recall@10.

**Рассуждение.** Для текущего среза Two-Tower v2 можно использовать как дополнительный candidate generator/rerank feature, но не как единственный retrieval. При появлении view/cart/purchase событий и более плотных повторных пользователей модель стоит переобучить и сравнить повторно.

### 5.7 Подбор весов hybrid и статистическая значимость

Grid-search весов hybrid (`src/tune_hybrid.py`, valid sample 1000 пользователей, step 0.2) показал, что текущие веса `content=0.65 / item-KNN=0.25 / popularity=0.10` остаются лучшими по Recall@10/NDCG@10: recall 0.090, ndcg 0.060. Ближайший конкурент `0.60/0.40/0.00` даёт recall 0.089, но ниже coverage/novelty-контроль popularity-компоненты. Полные результаты: `reports/hybrid_weight_search.csv`.

Bootstrap/McNemar сравнение `hybrid` vs `content_tfidf` на test sample 2000 пользователей (`src/significance.py`): hit-rate hybrid 0.0935 vs content 0.0845, разница +0.009, 95% bootstrap CI [0.0035, 0.015], McNemar p = 0.0058 (28 пользователей выиграл только hybrid, 10 — только content). Разница небольшая, но статистически значимая; практический эффект ограничен разреженностью данных.

## 6. Продакшен-ограничения

### 6.1 Latency при росте каталога

**Проблема.** `item_knn`/`hybrid` требуют предрасчёта item-item similarity и TF-IDF; на 7.6k товаров это терпимо, на миллионах — риск p95/p99 latency.

**Решение.** ANN (FAISS/HNSW), инвертированные индексы, кэширование top-N, двухстадийный retrieve→rerank.

**Недостаток решения.** ANN добавляет recall-loss и инфраструктурную сложность; кэш устаревает при обновлении каталога.

**Рассуждение.** Для текущего среза latency некритична; в проде нужен бюджет p95/p99 и fallback на popularity/content при деградации.

### 6.2 Cold catalog updates и data drift

**Проблема.** Новый товар не имеет co-occurrence; вкус/сезонность Beauty меняются, popularity/item-KNN устаревают.

**Решение.** Content fallback для новых товаров, exploration slot, инкрементальное обновление popularity/similarity по расписанию и event stream.

**Недостаток решения.** Exploration может снижать краткосрочный CTR; частое переобучение дороже и шумнее.

**Рассуждение.** Нужен balance между freshness и стабильностью; обновления должны быть за feature flag с откатом.

### 6.3 Качество входных данных

**Проблема.** Reviews ≠ purchases; `rating>=3` включает нейтральные отзывы; metadata неполная (price/category).

**Решение.** Использовать явные события `view/click/cart/purchase/return`, разные веса по типам событий, side information и missing-value fallbacks.

**Недостаток решения.** Больше сигналов = больше шума, дороже сбор/хранение и сложнее causal interpretation.

**Рассуждение.** Для бизнеса важнее обучать на целевом событии (purchase/ATC), а reviews использовать как auxiliary signal.

### 6.4 Объяснимость, модерация и compliance

**Проблема.** «С этим товаром покупают» и brand/category blocks могут усиливать нежелательные паттерны или показывать сомнительный контент.

**Решение.** Explainability («почему показано»), фильтры запрещённых категорий/брендов, fairness metrics, human review для sensitive categories.

**Недостаток решения.** Жёсткие фильтры могут ухудшить релевантность; fairness-метрики трудно интерпретировать без разметки.

**Рассуждение.** Для Beauty риск ниже, чем для job/credit, но фильтры и логирование объяснений обязательны.

### 6.5 A/B и откат

**Проблема.** Любая рекомендация может ухудшить business metrics или безопасностью контента.

**Решение.** Feature flag, A/B против popularity/content baseline, guardrail metrics (complaints, return rate, long-tail share), мгновенный rollback.

**Недостаток решения.** A/B требует трафика и времени; guardrails могут конфликтовать с краткосрочным CTR.

**Рассуждение.** Без A/B и отката нельзя доказать бизнес-эффект и безопасно выкатывать изменения.

## 7. Мониторинг в проде: ML- и бизнес-метрики

### ML-метрики и alert thresholds

- Offline: `Recall@K`, `HitRate@K`, `NDCG@K`, `MRR` на per-user temporal holdout.
- Ranking quality: `nDCG` с graded relevance (rating 3/4/5), `MAP`, `MRR`.
- Catalog health: coverage, novelty, intra-list diversity (category entropy), concentration top-10 items.
- Cold-start slices: метрики по длине истории и популярности item.
- Model health: drift входных фич, calibration score→CTR, latency p50/p95/p99.
- Пример alert thresholds: `Recall@10` drop > 10% week-over-week, `coverage` < 0.5, p95 latency > 300ms, error rate > 0.5%, novelty/content-filter violations spike.

### Dashboard

- Funnel: impressions → CTR → add-to-cart → purchase → revenue per session.
- Quality: offline recall/NDCG на последнем holdout, coverage/novelty/diversity, cold-start slices.
- Operations: latency p50/p95/p99, error rate, cache hit rate, freshness lag, queue depth retraining.
- Safety: complaints/moderation flags, brand safety incidents, sensitive-category impressions.

### Бизнес-метрики

- CTR recommendation block, add-to-cart rate, purchase rate, revenue per session.
- Conversion uplift vs control/popularity baseline.
- Long-tail share recommended/purchased — защита от popularity bias.
- Repeat purchase / retention cohorts для пользователей, видевших рекомендации.
- Return rate и customer satisfaction — guardrail против «кликабельных, но плохих» рекомендаций.
- Guardrail: complaints/moderation flags, brand safety incidents.

### A/B test design

- Hypothesis: hybrid увеличивает CTR/ATC/purchase vs popularity/content baseline без роста return rate и moderation flags.
- Primary metric: purchase rate или revenue per session; secondary: CTR, ATC, long-tail share, retention.
- Guardrails: return rate, complaints, latency p95, coverage/novelty, sensitive-category violations.
- Design: random user split, одинаковый catalog/UI, minimum detectable effect по baseline conversion, sequential monitoring и rollback при guardrail breach.

## 8. Практический вывод

Для текущего датасета и сроков проекта **рекомендуемый продакшен-вариант — hybrid content+item-KNN+popularity** с сильным content fallback для cold start. Two-Tower v2 реализован и обучен (Recall@10 = 0.0269), но на разреженных отзывах всё ещё уступает hybrid; его стоит использовать как дополнительный сигнал/кандидат при появлении более плотных поведенческих данных.

## 8.1 Доработки по списку проблем (статус)

Сделано в коде/отчёте: тесты (`pytest` = 16 passed), исправлена novelty-формула со smoothing, `same_category` переведён на pseudo-category по `category_path`/тексту (категории в metadata вырождены до `All Beauty`/`Premium Beauty`), cold-start корзины переименованы в `<=2` и помечены как short-history, `recent` переименован в `recent items`, убран текст задания из корня, README/REPORT команды приведены к `--eval-max-users 4000`, добавлены блоки кода, раздел ресёрча, формат прода/мониторинга и статистика single-review в `data_prep_report.json`.

Осталось для полной доработки: создать git remote и ссылку, добавить скриншоты/графики в `reports/`, добавить валидацию по эпохам/early stopping для Two-Tower и при необходимости дообучить финальную версию на `train+valid` после offline-выбора. Grid-search весов hybrid и bootstrap/McNemar значимость добавлены (`src/tune_hybrid.py`, `src/significance.py`); результаты в `reports/hybrid_weight_search.*` и `reports/significance_hybrid_vs_content.*`.

## 9. Сложности и решения

- **Датасет слишком разреженный.** Строгое k-core схлопывало Amazon Beauty до сотен строк. Решение: support-фильтр `rating>=3`, user ≥3, item ≥5. Недостаток: остаётся мало повторных пользователей, поэтому collaborative и neural-модели слабее content-based.
- **Metadata неполная.** У ~79% товаров нет цены, у части слабые category/brand. Решение: fallback в `Unknown`, опора на title/description/features/text-SVD. Недостаток: часть сигналов теряется, price в UI часто пустой.
- **Two-Tower v1 провалился, v2 исправлен.** Причины v1: `user_id_emb` переобучался, ID-эмбеддинги имели слишком большую инициализацию, embeddings не нормализовались. Решение: v2 без `user_id_emb`, `std=0.02`, dropout 0.1, нормализация, batch 128/30 эпох — Recall@10 вырос до 0.0269. Недостаток: всё ещё ниже hybrid; валидация по эпохам не реализована.
- **Индексация/тензоры в Two-Tower.** Были ошибки brand/category index и non-writable NumPy arrays. Решение: уникальные словари категорий, `.copy()/contiguous` перед `torch.as_tensor`, кэш item embeddings. Недостаток: добавляет инженерной сложности.
- **Честность оценки.** Baseline обучаются только на `train`, valid/test используются как held-out; history для test = `train+valid`. Недостаток: absolute Recall@10 всё равно низкий из-за одного target на пользователя и разреженности.

## 10. Воспроизведение

```bash
python -m pip install -r requirements.txt
python -m src.data_prep
python -m src.train_baselines --k 10
python -m src.train_two_tower --epochs 30 --batch-size 128 --n-random-negatives 256 --eval-max-users 4000 --output two_tower_v2.pt --train-splits train --no-user-id-emb --normalize-embeddings --embedding-init-std 0.02 --dropout 0.1
python -m src.cold_start_analysis --methods content_tfidf item_knn hybrid two_tower --k 10
python -m pytest -q
streamlit run app/streamlit_app.py
```

Ссылка на репозиторий: указывается в сдаваемом отчёте ЦОР (проект локальный, git remote не создан).

## 11. Чек-лист соответствия требованиям

| Требование | Статус | Где закрыто |
|---|---|---|
| E-commerce/медиа сервис и персонализация | ✅ | Amazon Beauty, e-commerce каталог |
| Минимум 2 классических подхода | ✅ | `content_tfidf`, `item_knn`, `hybrid` |
| 1–2 простых/эвристических baseline | ✅ | `popularity`, `recent items`, `same_brand`, `same_category` |
| Тесты | ✅ | `tests/`: metrics, data_prep split/filter, recommenders, predict; `pytest -q` = 16 passed |
| Одна современная/«модная» модель | ✅ | Two-Tower: обучение, inference, UI, notebook |
| Проблемы/ограничения подходов + решения + недостатки + рассуждение | ✅ | Раздел 5: cold start user/item, sparsity, popularity bias, ethics, Two-Tower limits |
| Проблемы/ограничения в продакшене | ✅ | Раздел 6: latency, cold catalog updates, drift, scalability, data quality, moderation, rollback |
| ML-метрики | ✅ | Recall@K, HitRate@K, NDCG@K, MRR, coverage, novelty |
| Бизнес-метрики/мониторинг | ✅ | Раздел 7: CTR→purchase proxy, conversion uplift, long-tail share, retention, guardrails |
| Ход работы / результаты / метрики / сложности | ✅ | Разделы 1–4, 8–10 |
| Блоки кода | ✅ | `src/`, `app/`, `notebooks/`, команды воспроизведения |
| Скриншоты | ⚠️ | Нужно добавить в сдаваемый отчёт ЦОР: Streamlit UI, notebook output, графики метрик |
| Ссылка на репозиторий | ⚠️ | Обязательно для ЦОР; git-репозиторий не инициализирован, remote-ссылка отсутствует |
| `_requirements.txt` с текстом задания | ✅ | Убран из корня проекта; исходные требования сохранены вне кода |
