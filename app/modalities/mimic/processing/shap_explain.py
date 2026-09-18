"""
Локальная интерпретация модели (SHAP) для классификаторов PD.
Использует TreeExplainer — точный и быстрый для LGBM / XGB / RF.
"""
import numpy as np
import pandas as pd
import shap

from app.modalities.mimic.processing.ml_pipeline import CorrelationFilter  # noqa: F401

_explainer_cache = {}

# Краткие названия ключевых точек для подписей
_PT = {
    39: "верхн. губа лев.", 40: "угол губы лев.", 46: "бровь лев. внешн.",
    50: "верхн. щека лев.", 52: "бровь лев. медиал.", 53: "бровь лев. внутр.",
    55: "нач. брови лев.",  61: "угол рта лев.",    63: "бровь лев. центр",
    65: "надбровье лев.",   66: "надбровье лев.",   70: "бровь лев. латер.",
    92: "нижн. губа лев.", 105: "арка брови лев.", 107: "арка брови лев. внешн.",
   123: "нижн. щека лев.", 165: "средн. щека лев.", 186: "верхн. губа лев.",
   187: "верхн. губа лев. бок.", 203: "у носа лев.", 205: "носогубн. складка лев.",
   206: "щека лев.",       207: "щека лев. внутр.", 216: "латер. щека лев.",
   269: "верхн. губа прав.", 270: "угол губы прав.", 276: "бровь прав. внешн.",
   280: "верхн. щека прав.", 282: "бровь прав. медиал.", 283: "бровь прав. внутр.",
   285: "нач. брови прав.", 291: "угол рта прав.",  293: "бровь прав. центр",
   295: "надбровье прав.",  296: "надбровье прав.", 300: "бровь прав. латер.",
   322: "нижн. губа прав.", 334: "арка брови прав.", 336: "арка брови прав. внешн.",
   352: "нижн. щека прав.", 391: "средн. щека прав.", 410: "верхн. губа прав.",
   411: "верхн. губа прав. бок.", 423: "у носа прав.", 425: "носогубн. складка прав.",
   426: "щека прав.",       427: "щека прав. внутр.", 436: "латер. щека прав.",
}

# Группы точек → мышца (из feature_extraction.py)
_MUSCLE_GROUPS = {
    "Большая скуловая (лев.)": [61, 186, 216, 207, 187, 123],
    "Большая скуловая (прав.)": [308, 410, 436, 427, 411, 352],
    "Малая скуловая (лев.)": [50, 203, 165, 39],
    "Малая скуловая (прав.)": [280, 423, 391, 269],
    "Леватор верхней губы (лев.)": [50, 205, 206, 92, 40],
    "Леватор верхней губы (прав.)": [280, 425, 426, 322, 270],
    "Корругатор (лев.)": [55, 65, 52, 53, 46],
    "Корругатор (прав.)": [285, 295, 282, 283, 276],
    "Фронталис (лев.)": [107, 66, 105, 63, 70],
    "Фронталис (прав.)": [336, 296, 334, 293, 300],
}

# Строим lookup: "id1-id2" → название мышцы
_PAIR_TO_MUSCLE = {}
for _muscle, _ids in _MUSCLE_GROUPS.items():
    from itertools import combinations as _comb
    for _a, _b in _comb(_ids, 2):
        _PAIR_TO_MUSCLE[f"{_a}-{_b}"] = _muscle

_AGG_LABELS = {
    "min_norm":    "мин. расстояние",
    "max_norm":    "макс. расстояние",
    "mean_norm":   "ср. расстояние",
    "median_norm": "медиана расст.",
    "std_norm":    "вариация расст.",
    "max_vel_norm":  "макс. скорость",
    "mean_vel_norm": "ср. скорость",
    "std_vel_norm":  "вариация скорости",
    "max_acc_norm":  "макс. ускорение",
    "std_acc_norm":  "вариация ускорения",
}


def _parse_col(col):
    """dist_426_270_mean_vel_norm -> ('426-270', 'mean_vel_norm')"""
    parts = col[len("dist_"):].split("_")
    id_parts = []
    for i, p in enumerate(parts):
        if p.isdigit():
            id_parts.append(p)
        else:
            return "-".join(id_parts), "_".join(parts[i:])
    return "-".join(id_parts), ""


def feature_label(col):
    pair_key, agg = _parse_col(col)
    muscle = _PAIR_TO_MUSCLE.get(pair_key)
    agg_label = _AGG_LABELS.get(agg, agg)
    try:
        id1, id2 = (int(x) for x in pair_key.split("-"))
        p1 = _PT.get(id1, f"т.{id1}")
        p2 = _PT.get(id2, f"т.{id2}")
        pts = f"{p1} → {p2}"
    except (ValueError, IndexError):
        pts = pair_key
    if muscle:
        return f"{col}\n({pts} | {muscle} | {agg_label})"
    return f"{col}\n({pts} | {agg_label})"


def _get_explainer(pipeline, exercise_type):
    """Создаёт и кэширует TreeExplainer на дереве модели."""
    if exercise_type not in _explainer_cache:
        model     = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        keep_mask = pipeline.named_steps["corr"].keep_mask_
        _explainer_cache[exercise_type] = {
            "explainer": explainer,
            "keep_mask": keep_mask,
        }
    return _explainer_cache[exercise_type]


def explain_prediction(pipeline, feature_columns, feature_row_df,
                       exercise_type="smile", top_k=5, feature_stats=None):
    """
    Возвращает топ-k признаков, толкающих предсказание к PD и к Healthy.
    TreeExplainer точный и быстрый — не требует фонового датасета.
    """
    cache     = _get_explainer(pipeline, exercise_type)
    explainer = cache["explainer"]
    keep_mask = cache["keep_mask"]

    # Прогоняем через CorrelationFilter + StandardScaler вручную
    X_corr   = pipeline.named_steps["corr"].transform(feature_row_df)
    X_scaled = pipeline.named_steps["scaler"].transform(X_corr)

    shap_values = explainer.shap_values(X_scaled)

    # Обрабатываем все форматы SHAP-вывода:
    # - list [cls0, cls1]         → старый SHAP для RF
    # - ndarray (n, feats, 2)     → новый SHAP для RF (берём класс 1)
    # - ndarray (n, feats)        → LGBM / XGB
    sv = np.asarray(shap_values)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_row = np.asarray(shap_values[1]).ravel()
    elif sv.ndim == 3:
        shap_row = sv[0, :, 1]
    else:
        shap_row = sv.ravel()

    # Расширяем обратно до полного числа признаков (удалённые CorrelationFilter → 0)
    full_shap = np.zeros(len(feature_columns))
    full_shap[keep_mask] = shap_row

    values = feature_row_df.iloc[0].values
    stats  = feature_stats or {}
    items  = []
    for i, col in enumerate(feature_columns):
        entry = {
            "feature":       col,
            "feature_label": feature_label(col),
            "shap":          float(full_shap[i]),
            "value":         float(values[i]),
        }
        if col in stats:
            entry.update(stats[col])
        items.append(entry)

    pushing_pd      = sorted([x for x in items if x["shap"] > 0],
                              key=lambda x: -x["shap"])[:top_k]
    pushing_healthy = sorted([x for x in items if x["shap"] < 0],
                              key=lambda x:  x["shap"])[:top_k]

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev_arr     = np.asarray(ev).ravel()
        base_value = float(ev_arr[1]) if len(ev_arr) >= 2 else float(ev_arr[0])
    else:
        base_value = float(ev)

    return {
        "base_value":      base_value,
        "pushing_pd":      pushing_pd,
        "pushing_healthy": pushing_healthy,
    }
