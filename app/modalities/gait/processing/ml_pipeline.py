import numpy as np
from typing import Dict, Any

MODEL_FEATURE_NAMES = [
    'cadence', 
    'sample_entropy', 
    'freeze_index_mean',
    'step_time_cv', 
    'X_std', 
    'X_rms', 
    'X_jerk_std'
]

def classify_features(features: Dict[str, Any], clf_model: Any, clf_scaler: Any) -> Dict[str, Any]:
    safe_defaults = {
        'cadence': 0.0, 'sample_entropy': 0.0, 'freeze_index_mean': 0.0,
        'step_time_cv': 0.0, 'X_std': 0.0, 'X_rms': 0.0, 'X_jerk_std': 0.0
    }
    feature_vec = []

    for name in MODEL_FEATURE_NAMES:
        val = features.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = safe_defaults[name]
        feature_vec.append(float(val))

    feature_vec = np.array(feature_vec, dtype=np.float64).reshape(1, -1)
    scaled = clf_scaler.transform(feature_vec)
    prediction = int(clf_model.predict(scaled)[0])
    probs = clf_model.predict_proba(scaled)[0]
    return {"prediction": prediction, "probability": float(probs[prediction])}

def check_feature_norm(features: Dict[str, float], bounds: Dict[str, Dict]) -> Dict[str, Dict]:
    norm_check = {}
    for feature, value in features.items():
        if feature not in bounds:
            norm_check[feature] = {
                "value": value, "status": "unknown",
                "in_norm": None, "message": "Границы не определены"
            }
            continue
        b = bounds[feature]
        lower = b["lower_bound"]
        upper = b["upper_bound"]
        norm_range = upper - lower
        buffer = norm_range * 0.2 if norm_range > 0 else abs(upper) * 0.1
        trans_lower_start = lower - buffer
        trans_upper_end = upper + buffer

        if value is None or (isinstance(value, float) and np.isnan(value)):
            norm_check[feature] = {
                "value": None, "status": "unknown",
                "in_norm": None, "message": "Не рассчитано",
                "lower_bound": lower, "upper_bound": upper,
                "trans_lower_start": trans_lower_start,
                "trans_upper_end": trans_upper_end,
            }
        elif lower <= value <= upper:
            norm_check[feature] = {
                "value": value, "status": "normal",
                "in_norm": True,
                "lower_bound": lower, "upper_bound": upper,
                "trans_lower_start": trans_lower_start,
                "trans_upper_end": trans_upper_end,
                "message": "В норме ✓"
            }
        elif trans_lower_start <= value < lower or upper < value <= trans_upper_end:
            norm_check[feature] = {
                "value": value, "status": "transition",
                "in_norm": False,
                "lower_bound": lower, "upper_bound": upper,
                "trans_lower_start": trans_lower_start,
                "trans_upper_end": trans_upper_end,
                "message": "Переходная зона ⚠"
            }
        else:
            dev = (f"Ниже на {lower - value:.4f}" if value < lower else f"Выше на {value - upper:.4f}")
            norm_check[feature] = {
                "value": value, "status": "abnormal",
                "in_norm": False,
                "lower_bound": lower, "upper_bound": upper,
                "trans_lower_start": trans_lower_start,
                "trans_upper_end": trans_upper_end,
                "message": f"Отклонение: {dev}"
            }
    return norm_check

def to_python_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_python_types(obj.tolist())
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj