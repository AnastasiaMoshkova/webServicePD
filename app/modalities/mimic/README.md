# Модуль «Мимика» (`app/modalities/mimic/`)

Диагностика болезни Паркинсона (БП) по мимике лица: 3 пробы (улыбка, подъём бровей,
нахмуривание бровей) → геометрические+динамические признаки лэндмарков MediaPipe →
три бинарных классификатора → позднее слияние (late fusion). 

## Пайплайн 

Пациент выполняет четыре записи подряд: нейтраль, затем три мимические пробы. Каждая проба
записывается видео, сервис находит на нём лицо и считает, как меняются расстояния между
ключевыми точками лица относительно нейтрального выражения того же человека.

| Шаг | Упражнение (как в UI) | Что делает сервис |
|---|---|---|
| 0 | **Нейтральное выражение**, 10 сек | Эталон: средние расстояния между точками. Без него остальные пробы не считаются. |
| 1 | **Улыбка с усилием**, 10 раз | Модель `smile_geom_dyn` → P(БП) по улыбке |
| 2 | **Поднятие бровей**, 10 раз | Модель `brow_geom_dyn` → P(БП) по бровям (самый информативный сигнал в ВКР) |
| 3 | **Нахмуривание бровей**, 10 раз | Модель `frown_dyn` → P(БП) по нахмуриванию |
| 4 | **Финальный вердикт** | Мета-модель `fusion_meta` объединяет три вероятности в одну P(БП) |

Порог решения 0.5: P(БП) ≥ 0.5 — вердикт «PD», иначе «Healthy». Уверенность в ответе:
высокая при max(P) ≥ 0.75, средняя при ≥ 0.60, иначе низкая.

В интерфейсе есть и другие упражнения (моргание, эмоции), но модели на них не обучены,
сервис их не обрабатывает.

**Точность (по ВКР, ROC-AUC, оценка на уровне пациента):** ансамбль трёх проб ≈ 0.93
(Average Fusion 0.929, Stacking 0.937). Лучшая отдельная проба — брови (AUC ≈ 0.91).
Выборка: 150 пациентов (123 PD / 27 здоровых).

**Внешняя проверка на новых пациентах:** на 10 пациентах, которых модель не видела при
обучении, бинарный вердикт верен в 8 случаях из 10. 


## Где что лежит (в этом репозитории)

```
app/modalities/mimic/
├── README.md                      ← этот файл
├── service.py                     ← MimicService — вся бизнес-логика
├── processing/
│   ├── feature_extraction.py      ← FeatureExtraction — лэндмарки → попарные расстояния
│   ├── raw_data_processing.py     ← MediaPipe-обработка видео, сборка feature-строки
│   ├── ml_pipeline.py             ← CorrelationFilter (кастомный sklearn-трансформер)
│   └── shap_explain.py            ← локальная интерпретация (SHAP) предсказания
└── models/
    ├── smile_geom_dyn.joblib      ← LightGBM, БП vs КГ по улыбке
    ├── brow_geom_dyn.joblib       ← XGBoost, БП vs КГ по бровям
    ├── frown_dyn.joblib           ← RandomForest, БП vs КГ по нахмуриванию
    └── fusion_meta.joblib         ← LogisticRegression, мета-модель (3 пробы → 1 вердикт)

app/routers/mimic.py               ← тонкий роутер, дёргает методы service
templates/mimic.html               ← страница модуля (extends "base.html")
static/js/mimic.js                 ← запись видео с камеры, WebSocket-превью, отрисовка графиков
```

**Стадийные модели (`*_early.joblib`, `*_3class.joblib` — КГ/ранняя БП/поздняя БП) сюда
не перенесены.** В текущей версии модуль определяет только бинарный вердикт (БП/КГ) через
`fusion_meta`. Если понадобится добавить определение стадии — модели и обучающий код есть в
приватной папке автора (вне этого репозитория). По доступу обращаться к автору модуля.


## Архитектура — поток данных одного запроса

```
браузер (static/js/mimic.js)
   │  запись видео по шагам: нейтраль → улыбка → брови → нахмуривание
   ▼
app/routers/mimic.py
   GET  /mimic                — страница + session_id cookie
   POST /mimic/upload_video   — сохранить ролик на диск (режим "upload")
   WS   /mimic/ws             — запись с камеры + live-превью с маской лица
   POST /mimic/process        — обработать сохранённый ролик одного упражнения
   POST /mimic/fusion         — финальный вердикт по всем трём пробам
   │
   ▼
app/modalities/mimic/service.py → MimicService
   process_video(session_id, file_path, exercise_type)
     ├─ exercise_type == "neutral"              → process_neutral(...)
     └─ exercise_type in (smile/brows_up/frown)  → process_exercise(...)
   process_fusion(proba_smile, proba_brow, proba_frown)
   process_all_fusion(session_id)   — берёт закэшированные пробы сессии и зовёт process_fusion
   │
   ▼
processing/raw_data_processing.collect_pair_history(file_path, exercise_type)
   MediaPipe FaceMesh по кадрам → processing/feature_extraction.FeatureExtraction
   → {"i-j": [расстояние по кадрам]} для нужных анатомических групп точек
   │
   ▼
нормализация на pair_means из нейтрального видео → агрегаты (min/max/mean/median/std +
скорость/ускорение) → feature-строка → pipeline из .joblib (CorrelationFilter → Scaler →
модель) → proba_pd
   │
   ▼
processing/shap_explain.explain_prediction — топ-5 признаков, которые повлияли на решение
(человекочитаемые подписи анатомических точек)
```

## Изоляция между пользователями (session_id)

`MimicService` создаётся **один раз на всё приложение** (`service = MimicService()` в
`app/routers/mimic.py`), поэтому всё состояние одного прохождения теста (эталон нейтрального
лица, кэш вероятностей трёх проб до финального fusion) **не хранится в полях сервиса** —
оно лежит в файлах внутри `static/recordings_face/<session_id>/`:

- `neutral_ref.json` — средние по парам точек из нейтрального видео (`pair_means`), пишется в
  `process_neutral`, читается в `process_exercise`.
- `proba_cache.json` — накопленные `proba_pd` по улыбке/бровям/нахмуриванию, пишется в
  `_save_proba` после каждой пробы, читается в `process_all_fusion`.

`session_id` выдаётся через cookie `core/session.py` — без него два одновременных пользователя переписывали бы один и тот же файл.

## Модели — формат `.joblib`

Каждый файл (кроме `fusion_meta.joblib`) — словарь:
```python
{
    "pipeline": sklearn.Pipeline,   # CorrelationFilter → StandardScaler → классификатор
    "feature_columns": [...],       # имена и ПОРЯДОК признаков — важно
    "feature_stats": {...},         # healthy/pd mean±std по признаку — для SHAP-подписей
}
```
`fusion_meta.joblib` — просто `LogisticRegression` с 3 входами: `["smile", "brow", "frown"]`.

`CorrelationFilter` (в `processing/ml_pipeline.py`) — кастомный sklearn-трансформер.
`joblib.dump` запоминает его модульный путь (`app.modalities.mimic.processing.ml_pipeline.CorrelationFilter`)
прямо в бинарнике. **Если этот файл переместить/переименовать — старые `.joblib` перестанут
загружаться** (`ModuleNotFoundError`/`AttributeError`). При любом рефакторинге структуры модуля
модели нужно переэкспортировать заново с новым путём импорта, не просто скопировать файлы.
Подробнее — в `MIGRATION.md`, раздел «Модели (`.joblib`) — важный момент».






