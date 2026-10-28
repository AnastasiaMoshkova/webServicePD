"""
Модели модуля «Мимика»: сборка по configs/mimic.yaml.

Базовая модель пробы — sklearn Pipeline: CorrelationFilter → StandardScaler → estimator.
Мета-модель позднего слияния — классификатор на вероятностях PD трёх проб.
Обученные модели хранятся в joblib (см. model.weights_paths); сервис их только загружает,
а models/train.py собирает их здесь же — одной функцией с одними настройками.
"""
import importlib
from pathlib import Path

import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.modalities.mimic.processing.ml_pipeline import CorrelationFilter
from app.modalities.mimic.processing.mimic_features import PAIRS, col_name

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mimic.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROBES = ("smile", "brow", "frown")
FUSION_INPUTS = ["smile", "brow", "frown"]   # порядок столбцов входа мета-модели


def load_config(path=CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def weights_path(model_cfg: dict, name: str) -> Path:
    p = Path(model_cfg["weights_paths"][name])
    return p if p.is_absolute() else PROJECT_ROOT / p


def feature_columns(model_cfg: dict, probe: str) -> list:
    """Признаки пробы: все пары PAIRS[probe] × агрегаты из конфига, по алфавиту (как при обучении)."""
    aggs = model_cfg["probes"][probe]["aggregates"]
    return sorted(col_name(p, a) for p in PAIRS[probe] for a in aggs)


def _estimator(dotted: str, params: dict):
    module, cls = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)(**params)


def build_probe_model(model_cfg: dict, probe: str, y, seed: int) -> Pipeline:
    """
    Необученный пайплайн пробы. y — метки обучающей выборки
    (нужны для scale_pos_weight: auto у XGBoost).
    """
    pc = model_cfg["probes"][probe]
    params = dict(pc.get("params", {}), random_state=seed)
    if pc.get("scale_pos_weight") == "auto":
        params["scale_pos_weight"] = float((y == 0).sum()) / float((y == 1).sum())
    return Pipeline([
        ("corr", CorrelationFilter(threshold=model_cfg["correlation_threshold"])),
        ("scaler", StandardScaler()),
        ("model", _estimator(pc["estimator"], params)),
    ])


def build_fusion_model(model_cfg: dict, seed: int):
    fc = model_cfg["fusion"]
    return _estimator(fc["estimator"], dict(fc.get("params", {}), random_state=seed))
