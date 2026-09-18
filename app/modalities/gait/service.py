import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from app.modalities.base_service import BaseModalityService
from app.modalities.gait.processing.feature_extraction import FEATURE_UNITS, compute_all_segment_features
from app.modalities.gait.processing.ml_pipeline import MODEL_FEATURE_NAMES, check_feature_norm, classify_features, to_python_types
from app.modalities.gait.processing.raw_data_processing import (
    CFG,
    STEP_SIZE_DET,
    STEP_SIZE_SEG,
    WIN_SIZE_DET,
    WIN_SIZE_SEG,
    create_windows,
    extract_precise_timestamps,
    find_true_movement_boundaries,
    resample_and_prep,
    smooth_predictions,
)

logger = logging.getLogger(__name__)
_MODELS_DIR = Path(__file__).resolve().parent / "models"


class AttentionLSTM(nn.Module):
    """Architecture matching detector.pth and segmenter.pth state_dicts."""

    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        scores = self.attention(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(out * weights, dim=1)
        return self.fc(context)


class GaitService(BaseModalityService):
    """Complete gait pipeline used by the web router."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        self.classifier_scaler = None
        self.detector: Optional[AttentionLSTM] = None
        self.detector_scaler = None
        self.segmenter: Optional[AttentionLSTM] = None
        self.segmenter_scaler = None
        self.feature_bounds: Dict[str, Any] = {}
        self._session_cache: Dict[str, pd.DataFrame] = {}
        self._load_error: Optional[str] = None
        self.load_artifacts()

    @property
    def ready(self) -> bool:
        return all(
            x is not None
            for x in (
                self.classifier,
                self.classifier_scaler,
                self.detector,
                self.detector_scaler,
                self.segmenter,
                self.segmenter_scaler,
            )
        ) and not self._load_error

    def load_artifacts(self) -> None:
        try:
            self.classifier = joblib.load(self.models_dir / "classifier.pkl")
            self.classifier_scaler = joblib.load(self.models_dir / "classifier_scaler.pkl")
            self.detector_scaler = joblib.load(self.models_dir / "detector_scaler.pkl")
            self.segmenter_scaler = joblib.load(self.models_dir / "segmenter_scaler.pkl")

            self.detector = AttentionLSTM().to(self.device)
            det_state = torch.load(self.models_dir / "detector.pth", map_location=self.device, weights_only=True)
            self.detector.load_state_dict(det_state)
            self.detector.eval()

            self.segmenter = AttentionLSTM().to(self.device)
            seg_state = torch.load(self.models_dir / "segmenter.pth", map_location=self.device, weights_only=True)
            self.segmenter.load_state_dict(seg_state)
            self.segmenter.eval()

            with open(self.models_dir / "feature_bounds.json", "r", encoding="utf-8") as f:
                self.feature_bounds = json.load(f)
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load gait artifacts")

    def models_status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "device": str(self.device),
            "error": self._load_error,
            "artifacts": {
                "classifier": self.classifier is not None,
                "classifier_scaler": self.classifier_scaler is not None,
                "detector": self.detector is not None,
                "detector_scaler": self.detector_scaler is not None,
                "segmenter": self.segmenter is not None,
                "segmenter_scaler": self.segmenter_scaler is not None,
                "feature_bounds": bool(self.feature_bounds),
            },
        }

    @staticmethod
    def _read_csv(content: bytes, sensor: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать CSV {sensor}: {exc}") from exc
        if df.empty:
            raise ValueError(f"CSV {sensor} пуст.")
        return df

    @staticmethod
    def _time_axis(df: pd.DataFrame, sensor_name: str) -> np.ndarray:
        # Prefer the raw sensor timestamp. For phone exports it is nanoseconds.
        if "time" in df.columns:
            raw = df["time"]
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().sum() >= max(2, len(df) // 2):
                med = float(np.nanmedian(np.abs(numeric.to_numpy(dtype=float))))
                if med > 1e11:
                    # Important for gyro/accelerometer phone exports: time is Unix-like ns.
                    dt_index = pd.to_datetime(numeric, unit="ns", errors="coerce")
                    vals = dt_index.astype("int64", copy=False).to_numpy(dtype=np.float64) / 1e9
                    vals[dt_index.isna().to_numpy()] = np.nan
                    return vals
                return numeric.to_numpy(dtype=float)
            parsed = pd.to_datetime(raw, errors="coerce")
            if parsed.notna().sum() >= 2:
                vals = parsed.astype("int64", copy=False).to_numpy(dtype=np.float64) / 1e9
                vals[parsed.isna().to_numpy()] = np.nan
                return vals

        if "seconds_elapsed" in df.columns:
            vals = pd.to_numeric(df["seconds_elapsed"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 2:
                return vals
        raise ValueError(f"В {sensor_name} не найден корректный столбец time или seconds_elapsed.")

    @staticmethod
    def _sensor_frame(df: pd.DataFrame, prefix: str, sensor_name: str) -> pd.DataFrame:
        out = pd.DataFrame({"time_abs": GaitService._time_axis(df, sensor_name)})
        for axis in ("x", "y", "z"):
            candidates = [f"{prefix}_{axis}", axis]
            col = next((c for c in candidates if c in df.columns), None)
            if col is None:
                raise ValueError(f"В {sensor_name} отсутствует столбец оси {axis}.")
            out[f"{prefix}_{axis}"] = pd.to_numeric(df[col], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_abs"])
        out = out.sort_values("time_abs").drop_duplicates("time_abs")
        if len(out) < 20:
            raise ValueError(f"В {sensor_name} недостаточно валидных отсчётов.")
        return out

    def _merge_sensors(self, acc_content: bytes, gyro_content: bytes) -> pd.DataFrame:
        acc = self._sensor_frame(self._read_csv(acc_content, "акселерометра"), "acc", "акселерометре")
        gyr = self._sensor_frame(self._read_csv(gyro_content, "гироскопа"), "gyr", "гироскопе")

        start = max(float(acc.time_abs.iloc[0]), float(gyr.time_abs.iloc[0]))
        end = min(float(acc.time_abs.iloc[-1]), float(gyr.time_abs.iloc[-1]))
        if end <= start:
            # Some exports reset seconds_elapsed independently. Align starts only in that case.
            acc = acc.copy(); gyr = gyr.copy()
            acc["time_abs"] -= float(acc.time_abs.iloc[0])
            gyr["time_abs"] -= float(gyr.time_abs.iloc[0])
            start = 0.0
            end = min(float(acc.time_abs.iloc[-1]), float(gyr.time_abs.iloc[-1]))
        if end - start < 2.5:
            raise ValueError("Пересекающаяся запись акселерометра и гироскопа слишком короткая.")

        acc = acc[(acc.time_abs >= start) & (acc.time_abs <= end)].copy()
        gyr = gyr[(gyr.time_abs >= start) & (gyr.time_abs <= end)].copy()
        t = acc["time_abs"].to_numpy(dtype=float)
        merged = acc.rename(columns={"time_abs": "time_sec"}).copy()
        for axis in ("x", "y", "z"):
            merged[f"gyr_{axis}"] = np.interp(t, gyr.time_abs.to_numpy(dtype=float), gyr[f"gyr_{axis}"].to_numpy(dtype=float))
        merged["time_sec"] = merged["time_sec"] - start
        return merged[["time_sec", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]]

    def _predict_windows(self, model: AttentionLSTM, scaler: Any, matrix: np.ndarray, time_arr: np.ndarray, win: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        scaled = scaler.transform(matrix).astype(np.float32)
        windows, centers = create_windows(scaled, time_arr, win, step)
        if len(windows) == 0:
            return np.empty((0,), dtype=int), centers
        preds = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                xb = torch.from_numpy(windows[i:i + batch_size]).to(self.device)
                logits = model(xb)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        return np.asarray(preds, dtype=int), centers

    @staticmethod
    def _movement_fallback(df_res: pd.DataFrame) -> Tuple[float, float]:
        t = df_res["time_sec"].to_numpy(dtype=float)
        mag = df_res["acc_mag"].to_numpy(dtype=float)
        activity = np.abs(mag - pd.Series(mag).rolling(101, center=True, min_periods=1).median().to_numpy())
        threshold = max(float(np.quantile(activity, 0.60)), 0.05 * float(np.std(mag)))
        active = np.where(activity >= threshold)[0]
        if active.size >= 2:
            return float(t[active[0]]), float(t[active[-1]])
        return float(t[0]), float(t[-1])

    @staticmethod
    def _pointwise_segmentation(time_arr: np.ndarray, centers: np.ndarray, preds: np.ndarray, start: float, end: float) -> np.ndarray:
        out = np.zeros(len(time_arr), dtype=int)
        if len(centers) == 0:
            return out
        mask = (time_arr >= start) & (time_arr <= end)
        selected = np.where(mask)[0]
        if selected.size == 0:
            return out
        pos = np.searchsorted(centers, time_arr[selected])
        pos = np.clip(pos, 0, len(centers) - 1)
        prev = np.clip(pos - 1, 0, len(centers) - 1)
        choose_prev = np.abs(centers[prev] - time_arr[selected]) < np.abs(centers[pos] - time_arr[selected])
        nearest = np.where(choose_prev, prev, pos)
        out[selected] = preds[nearest]
        return out

    @staticmethod
    def _validate_points(points: Dict[str, Any], duration: float) -> Dict[str, float]:
        keys = ["T1", "T2", "T3", "T4", "T5"]
        vals = []
        for key in keys:
            value = points.get(key)
            if value is None:
                raise ValueError("Не все точки T1–T5 заданы. Укажите пять точек и повторите перерасчёт.")
            value = float(value)
            if not np.isfinite(value) or value < 0 or value > duration:
                raise ValueError(f"Точка {key} выходит за границы записи.")
            vals.append(value)
        if any(b <= a for a, b in zip(vals, vals[1:])):
            raise ValueError("Точки должны идти строго по времени: T1 < T2 < T3 < T4 < T5.")
        return dict(zip(keys, vals))

    @staticmethod
    def _average_features(features_1: Dict[str, Any], features_2: Dict[str, Any], turn_time: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        keys = set(features_1) | set(features_2)
        for key in keys:
            if key == "turn_time_sec":
                continue
            vals = []
            for source in (features_1, features_2):
                v = source.get(key)
                if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                    vals.append(float(v))
            result[key] = float(np.mean(vals)) if vals else np.nan
        result["turn_time_sec"] = float(turn_time)
        return result

    def _calculate_result(self, df_res: pd.DataFrame, points: Dict[str, Any], auto_points_complete: bool) -> Dict[str, Any]:
        duration = float(df_res["time_sec"].iloc[-1])
        p = self._validate_points(points, duration)
        t = df_res["time_sec"]
        s1 = df_res[(t >= p["T1"]) & (t <= p["T2"])].copy()
        s2 = df_res[(t >= p["T3"]) & (t <= p["T4"])].copy()
        if len(s1) < 30 or len(s2) < 30:
            raise ValueError("Один из сегментов ходьбы слишком короткий для расчёта признаков.")

        f1 = compute_all_segment_features(s1)
        f2 = compute_all_segment_features(s2)
        turn_time = ((p["T3"] - p["T2"]) + (p["T5"] - p["T4"])) / 2.0
        features = self._average_features(f1, f2, turn_time)
        classification = classify_features(features, self.classifier, self.classifier_scaler)
        model_feature_values = {name: features.get(name) for name in MODEL_FEATURE_NAMES}
        norm = check_feature_norm(model_feature_values, self.feature_bounds)

        def segment_payload(seg: pd.DataFrame, start: float, end: float) -> Dict[str, Any]:
            return {
                "start_time": float(start),
                "end_time": float(end),
                "time_data": seg["time_sec"].to_numpy(),
                "signal_data": seg["acc_mag"].to_numpy(),
            }

        result = {
            **classification,
            "points": p,
            "auto_points_complete": bool(auto_points_complete),
            "segments": {
                "segment_1": segment_payload(s1, p["T1"], p["T2"]),
                "segment_2": segment_payload(s2, p["T3"], p["T4"]),
            },
            "full_signal": {
                "time": df_res["time_sec"].to_numpy(),
                "magnitude": df_res["acc_mag"].to_numpy(),
            },
            "feature_norm": norm,
            "all_features": features,
            "feature_units": FEATURE_UNITS,
        }
        return to_python_types(result)

    def process_files(self, session_id: str, acc_content: bytes, gyro_content: bytes) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"Модели походки не готовы: {self._load_error or 'не все артефакты загружены'}")

        merged = self._merge_sensors(acc_content, gyro_content)
        matrix, time_arr, df_res = resample_and_prep(merged)
        self._session_cache[session_id] = df_res

        det_pred, t_det = self._predict_windows(self.detector, self.detector_scaler, matrix, time_arr, WIN_SIZE_DET, STEP_SIZE_DET)
        det_pred = smooth_predictions(det_pred, 5)
        movement = find_true_movement_boundaries(det_pred, min_duration_sec=4.5)
        if movement is not None and len(t_det):
            half = CFG["win_sec_det"] / 2.0
            move_start = max(float(time_arr[0]), float(t_det[movement[0]] - half))
            move_end = min(float(time_arr[-1]), float(t_det[movement[1] - 1] + half))
        else:
            move_start, move_end = self._movement_fallback(df_res)

        seg_pred, t_seg = self._predict_windows(self.segmenter, self.segmenter_scaler, matrix, time_arr, WIN_SIZE_SEG, STEP_SIZE_SEG)
        seg_pred = smooth_predictions(seg_pred, 5)
        point_pred = self._pointwise_segmentation(time_arr, t_seg, seg_pred, move_start, move_end)
        points = extract_precise_timestamps(time_arr, point_pred, move_start, move_end)
        auto_complete = all(points[k] is not None for k in ("T1", "T2", "T3", "T4", "T5"))

        if not auto_complete:
            # Keep the automatically found boundaries visible and make the result editable.
            # Interior missing points are filled only as an explicit provisional fallback.
            span = max(move_end - move_start, 0.1)
            fallback = {
                "T1": move_start,
                "T2": move_start + span * 0.38,
                "T3": move_start + span * 0.50,
                "T4": move_start + span * 0.88,
                "T5": move_end,
            }
            for key, value in fallback.items():
                if points.get(key) is None:
                    points[key] = value

        result = self._calculate_result(df_res, points, auto_points_complete=auto_complete)
        result["needs_manual_review"] = not auto_complete
        return result

    def recalculate(self, session_id: str, points: Dict[str, Any]) -> Dict[str, Any]:
        df_res = self._session_cache.get(session_id)
        if df_res is None:
            raise ValueError("Данные текущей сессии не найдены. Загрузите CSV-файлы заново.")
        result = self._calculate_result(df_res, points, auto_points_complete=False)
        result["needs_manual_review"] = False
        return result
