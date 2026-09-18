import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from app.modalities.base_service import BaseModalityService
from app.modalities.mimic.processing import raw_data_processing as rdp
from app.modalities.mimic.processing.ml_pipeline import CorrelationFilter  # noqa: F401 (needed for pickle)
from app.modalities.mimic.processing.shap_explain import explain_prediction

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "models"

# Пары точек, выводимые на график "дистанция от времени" для каждого упражнения.
TOP_PAIRS = {
    "smile":    ["308-411", "308-427", "61-187", "61-207"],
    "brows_up": ["285-168", "336-168", "107-168", "55-168", "296-10", "66-10"],
    "frown":    ["285-6", "55-6", "52-168", "282-168", "295-4"],
}


class MimicService(BaseModalityService):
    """
    Обработка мимических проб (улыбка / подъём бровей / нахмуривание бровей):
    расчёт признаков по видео, классификация и позднее слияние (late fusion).
    """

    def __init__(self, local_dir: str = "static/recordings_face/") -> None:
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

        self.model_paths = {
            "smile":  _MODELS_DIR / "smile_geom_dyn.joblib",
            "brow":   _MODELS_DIR / "brow_geom_dyn.joblib",
            "frown":  _MODELS_DIR / "frown_dyn.joblib",
            "fusion": _MODELS_DIR / "fusion_meta.joblib",
        }
        self._models_cache: Dict[str, Any] = {}

    # ── модели ────────────────────────────────────────────────────────────

    def _get_model(self, name: str) -> Any:
        if name not in self._models_cache:
            if name == "fusion":
                self._models_cache["fusion"] = joblib.load(self.model_paths["fusion"])
            else:
                blob = joblib.load(self.model_paths[name])
                self._models_cache[name] = {
                    "pipeline": blob["pipeline"],
                    "feature_columns": blob["feature_columns"],
                    "feature_stats": blob.get("feature_stats", {}),
                }
        return self._models_cache[name]

    # ── изоляция состояния между пользователями ──────────────────────────
    # Файлы состояния (эталон нейтрального лица, кэш вероятностей) кладутся
    # в подпапку по session_id — иначе два одновременных пользователя
    # затирали бы данные друг друга (общий файл на всё приложение).

    def _session_dir(self, session_id: str) -> str:
        d = os.path.join(self.local_dir, session_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _neutral_ref_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "neutral_ref.json")

    def _proba_cache_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "proba_cache.json")

    # ── нейтральное выражение (эталон нормализации) ─────────────────────

    def process_neutral(self, session_id: str, file_path: str) -> Dict[str, Any]:
        history, fps, best_frame_b64, best_landmarks, quality = \
            rdp.collect_pair_history(file_path, "neutral_all")
        if not history:
            return {"status": "error", "message": "Лицо не распознано на видео."}

        pair_means = {k: float(np.mean(v)) for k, v in history.items() if v}
        n_frames   = len(next(iter(history.values())))

        # Новая нейтраль в этой сессии — сбрасываем кэш вероятностей предыдущей
        proba_cache_path = self._proba_cache_path(session_id)
        if os.path.exists(proba_cache_path):
            os.remove(proba_cache_path)
        with open(self._neutral_ref_path(session_id), "w") as f:
            json.dump({"pair_means": pair_means, "n_frames": n_frames, "fps": fps}, f)

        return {
            "status": "success",
            "kind": "neutral_ready",
            "n_frames": n_frames,
            "fps": fps,
            "pairs_detected": len(pair_means),
            "best_frame": best_frame_b64,
            "best_landmarks": best_landmarks,
            "quality": rdp.quality_level(quality, n_frames),
        }

    # ── одна мимическая проба ────────────────────────────────────────────

    def process_exercise(self, session_id: str, file_path: str, exercise_type: str) -> Dict[str, Any]:
        """exercise_type: 'smile' | 'brows_up' | 'frown'."""
        model_name = {"smile": "smile", "brows_up": "brow", "frown": "frown"}.get(exercise_type)
        if model_name is None:
            return {
                "status": "error",
                "message": f"Упражнение '{exercise_type}' не поддерживается. "
                           f"Доступно: smile, brows_up, frown.",
            }

        neutral_ref_path = self._neutral_ref_path(session_id)
        if not os.path.exists(neutral_ref_path):
            return {"status": "error", "message": "Сначала обработайте нейтральное выражение."}

        with open(neutral_ref_path) as f:
            ref = json.load(f)
        pair_means = ref["pair_means"]

        history, fps, best_frame_b64, best_landmarks, quality = \
            rdp.collect_pair_history(file_path, exercise_type)
        if not history:
            return {"status": "error", "message": "Лицо не распознано на видео."}

        model_blob      = self._get_model(model_name)
        pipe            = model_blob["pipeline"]
        feature_columns = model_blob["feature_columns"]
        feat_stats      = model_blob.get("feature_stats", {})

        normalized_series = {}
        for pair_key, vals in history.items():
            ref_mean = pair_means.get(pair_key)
            if ref_mean and ref_mean > 0:
                normalized_series[pair_key] = (np.array(vals) / ref_mean).tolist()

        feature_row, missing = rdp.build_feature_row(normalized_series, feature_columns)
        feat_df = pd.DataFrame([feature_row], columns=feature_columns)

        proba         = pipe.predict_proba(feat_df)[0]
        pred          = int(pipe.predict(feat_df)[0])
        proba_healthy = float(proba[0])
        proba_pd      = float(proba[1])
        max_proba     = max(proba_pd, proba_healthy)
        confidence    = "high" if max_proba >= 0.75 else "moderate" if max_proba >= 0.60 else "low"

        n_frames = len(next(iter(history.values())))
        times    = (np.arange(n_frames) / fps).tolist()

        chart_series = {}
        for pk in TOP_PAIRS.get(exercise_type, []):
            if pk in normalized_series:
                chart_series[f"dist_{pk.replace('-', '_')}"] = \
                    rdp.rolling_median(normalized_series[pk], window=5)

        try:
            explanation = explain_prediction(pipe, feature_columns, feat_df,
                                             exercise_type=model_name, top_k=5,
                                             feature_stats=feat_stats)
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            explanation = None

        self._save_proba(session_id, model_name, proba_pd)

        return {
            "status": "success",
            "kind": f"{exercise_type}_prediction",
            "prediction": {
                "label": "PD" if pred == 1 else "Healthy",
                "label_id": pred,
                "proba_pd": proba_pd,
                "proba_healthy": proba_healthy,
                "max_proba": max_proba,
                "confidence_level": confidence,
            },
            "chart": {"times": times, "series": chart_series},
            "best_frame": best_frame_b64,
            "best_landmarks": best_landmarks,
            "n_frames": n_frames,
            "fps": fps,
            "missing_features": missing,
            "quality": rdp.quality_level(quality, n_frames),
            "explanation": explanation,
        }

    # ── позднее слияние (late fusion) ────────────────────────────────────

    def _save_proba(self, session_id: str, exercise_name: str, proba_pd: float) -> None:
        proba_cache_path = self._proba_cache_path(session_id)
        cache = {}
        if os.path.exists(proba_cache_path):
            with open(proba_cache_path) as f:
                cache = json.load(f)
        cache[exercise_name] = proba_pd
        with open(proba_cache_path, "w") as f:
            json.dump(cache, f)

    def process_fusion(self, proba_smile: float, proba_brow: float, proba_frown: float) -> Dict[str, Any]:
        meta   = self._get_model("fusion")
        X_meta = pd.DataFrame([[proba_smile, proba_brow, proba_frown]],
                               columns=["smile", "brow", "frown"])
        proba_pd      = float(meta.predict_proba(X_meta)[0, 1])
        proba_healthy = 1.0 - proba_pd
        pred          = int(meta.predict(X_meta)[0])
        max_proba     = max(proba_pd, proba_healthy)
        confidence    = "high" if max_proba >= 0.75 else "moderate" if max_proba >= 0.60 else "low"
        return {
            "status": "success",
            "kind": "fusion_prediction",
            "prediction": {
                "label": "PD" if pred == 1 else "Healthy",
                "label_id": pred,
                "proba_pd": proba_pd,
                "proba_healthy": proba_healthy,
                "max_proba": max_proba,
                "confidence_level": confidence,
            },
            "components": {
                "smile": proba_smile,
                "brow":  proba_brow,
                "frown": proba_frown,
            },
        }

    def process_all_fusion(self, session_id: str) -> Dict[str, Any]:
        proba_cache_path = self._proba_cache_path(session_id)
        if not os.path.exists(proba_cache_path):
            return {"status": "error",
                    "message": "Нет данных упражнений. Сначала обработайте улыбку, брови и нахмуривание."}
        with open(proba_cache_path) as f:
            cache = json.load(f)
        missing = [e for e in ("smile", "brow", "frown") if e not in cache]
        if missing:
            names = {"smile": "улыбку", "brow": "брови", "frown": "нахмуривание"}
            missing_ru = ", ".join(names[e] for e in missing)
            return {"status": "error",
                    "message": f"Не хватает результатов: {missing_ru}. Обработайте все три упражнения."}
        return self.process_fusion(cache["smile"], cache["brow"], cache["frown"])

    # ── единая точка входа для одного упражнения ─────────────────────────

    def process_video(self, session_id: str, file_path: str, exercise_type: str = "smile") -> Dict[str, Any]:
        if exercise_type == "neutral":
            return self.process_neutral(session_id, file_path)
        if exercise_type in ("smile", "brows_up", "frown"):
            return self.process_exercise(session_id, file_path, exercise_type)
        return {
            "status": "error",
            "message": f"Упражнение '{exercise_type}' не поддерживается. "
                       f"Доступно: neutral, smile, brows_up, frown.",
        }

    # ── интерфейс BaseModalityService ────────────────────────────────────
    # Флоу мимики (нейтраль → 3 пробы → fusion) не укладывается в один метод
    # predict_binary — используются process_video/process_fusion выше
    # (тот же подход, что и у HandTrackingService: свои методы поверх базовых).

    def upload(self, file_name: str, content: bytes) -> Dict[str, Any]:
        raise NotImplementedError("upload is not implemented — см. process_video")

    def insert_in_db(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("insert_in_db is not implemented")
