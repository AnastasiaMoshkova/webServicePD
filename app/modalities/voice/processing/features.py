from __future__ import annotations

import logging
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call

logger = logging.getLogger(__name__)


def _safe(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def extract_voice_features(wav_path: str) -> dict:
    sound = parselmouth.Sound(wav_path)

    features = {
        "jitter": {"value": None, "unit": "%"},
        "shimmer": {"value": None, "unit": "%"},
        "hnr": {"value": None, "unit": "dB"},
    }

    pitch_floor, pitch_ceiling = 75, 500
    period_floor, period_ceiling = 0.0001, 0.02
    max_period_factor = 1.3
    max_amplitude_factor = 1.6

    try:
        pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
        pulses = call([sound, pitch], "To PointProcess (cc)")

        jitter_local = call(
            pulses, "Get jitter (local)", 0, 0,
            period_floor, period_ceiling, max_period_factor,
        )
        j = _safe(jitter_local)
        if j is not None:
            features["jitter"]["value"] = j * 100.0

        shimmer_local = call(
            [sound, pulses], "Get shimmer (local)", 0, 0,
            period_floor, period_ceiling, max_period_factor, max_amplitude_factor,
        )
        s = _safe(shimmer_local)
        if s is not None:
            features["shimmer"]["value"] = s * 100.0
    except Exception as e:
        logger.warning(f"Jitter/Shimmer extraction failed: {e}")

    try:
        hnr_obj = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.02, 4.5)
        hnr_val = _safe(call(hnr_obj, "Get mean", 0, 0))
        if hnr_val is not None and 0 < hnr_val < 100:
            features["hnr"]["value"] = hnr_val
    except Exception as e:
        logger.warning(f"HNR extraction failed: {e}")

    logger.info(
        "Features: jitter=%s, shimmer=%s, hnr=%s",
        features["jitter"]["value"],
        features["shimmer"]["value"],
        features["hnr"]["value"],
    )
    return features