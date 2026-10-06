from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import librosa
import soundfile as sf

from app.modalities.base_service import BaseModalityService

logger = logging.getLogger(__name__)


class VoiceService(BaseModalityService):
    def __init__(self, config_path: str = "configs/voice.yaml"):
        super().__init__()
        self.config_path = config_path
        self._config = None
        self._classifier = None
        logger.info(f"VoiceService initialized with config: {config_path}")

    @property
    def config(self) -> dict:
        if self._config is None:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config not found: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Config loaded from {self.config_path}")
        return self._config

    @property
    def classifier(self):
        if self._classifier is None:
            from app.modalities.voice.processing.inference import VoiceClassifier
            mc = self.config["model"]
            weights_paths = mc.get("weights_paths", [])
            abs_paths = []
            project_root = Path(__file__).parent.parent.parent.parent
            for p in weights_paths:
                if not os.path.isabs(p):
                    p = str(project_root / p)
                abs_paths.append(p)
                logger.info(f"Model path: {p}")
            for p in abs_paths:
                if not os.path.exists(p):
                    logger.warning(f"Model checkpoint not found: {p}")
            self._classifier = VoiceClassifier(
                weights_paths=abs_paths,
                device=mc.get("device"),
                threshold=mc.get("threshold", 0.5),
                model_config=mc.get("architecture"),
            )
        return self._classifier

    def process(
        self,
        audio_path: str,
        patient_id: str,
        exercise: str,
        results_dir: str = "results/voice",
        spec_web_dir: str = "static/results/voice",
        spec_url_prefix: str = "/static/results/voice",
    ) -> dict[str, Any]:
        from app.modalities.voice.processing.features import extract_voice_features
        from app.modalities.voice.processing.three_channel import ThreeChannelConfig, generate_three_channel
        from app.modalities.voice.processing.debug_log import log_event
        import sys

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(spec_web_dir, exist_ok=True)

        logger.info(f"Processing audio: {audio_path} (exercise={exercise})")

        log_event(
            "process_start",
            audio_path=os.path.abspath(audio_path),
            patient_id=patient_id,
            exercise=exercise,
            python=sys.executable,
            cwd=os.getcwd(),
        )

        y_full, sr_full = librosa.load(audio_path, sr=16000, mono=True)
        if y_full is None or len(y_full) == 0:
            raise ValueError("Empty or unreadable audio")
        duration_sec = float(len(y_full) / sr_full)
        logger.info(f"Loaded: {duration_sec:.2f}s @ {sr_full} Hz")

        log_event(
            "preprocess_done",
            sample_rate=int(sr_full),
            voice_duration=duration_sec,
            waveform_len=len(y_full),
            waveform_mean=float(y_full.mean()),
            waveform_std=float(y_full.std()),
            waveform_min=float(y_full.min()),
            waveform_max=float(y_full.max()),
        )

        cleaned_path = os.path.join(os.path.dirname(audio_path), "cleaned.wav")
        sf.write(cleaned_path, y_full, sr_full)

        try:
            features = extract_voice_features(cleaned_path)
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features = {
                "jitter": {"value": None, "unit": "%"},
                "shimmer": {"value": None, "unit": "%"},
                "hnr": {"value": None, "unit": "dB"},
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spec_filename = f"{patient_id}_{timestamp}_spec.png"
        spec_path = os.path.join(spec_web_dir, spec_filename)
        model_input = generate_three_channel(audio_path, spec_path, ThreeChannelConfig())
        log_event(
            "three_channel_done",
            shape=list(model_input.shape),
            dtype=str(model_input.dtype),
            mean_rgb=[float(model_input[..., i].mean()) for i in range(3)],
            min_rgb=[int(model_input[..., i].min()) for i in range(3)],
            max_rgb=[int(model_input[..., i].max()) for i in range(3)],
            spec_path=spec_path,
        )

        prediction = self.classifier.predict_from_array(model_input)
        log_event(
            "prediction_done",
            probability=prediction["probability"],
            class_id=prediction["class"],
            label=prediction["label"],
            fold_probabilities=prediction["fold_probabilities"],
            ensemble_size=prediction["ensemble_size"],
            threshold=prediction["threshold"],
        )

        model_trained_for = ["a"]
        exercise_supported = exercise in model_trained_for

        spec_url = f"{spec_url_prefix}/{spec_filename}"

        y = y_full
        max_points = 2000
        if len(y) > max_points:
            window = len(y) // max_points
            trimmed = y[:window * max_points]
            waveform_display = trimmed.reshape(max_points, window).mean(axis=1).astype(float).tolist()
        else:
            waveform_display = y.astype(float).tolist()

        payload = {
            "patient_id": patient_id,
            "exercise": exercise,
            "exercise_supported_by_model": exercise_supported,
            "model_trained_for": model_trained_for,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "audio": {
                "sample_rate": int(sr_full),
                "voice_duration_sec": duration_sec,
                "voice_start_sec": 0.0,
                "voice_end_sec": duration_sec,
                "waveform_downsampled": waveform_display,
            },
            "spectrogram_url": spec_url,
            "spectrogram_path": spec_path,
            "model_input": "three_channel_logmel_chroma_mfcc",
            "features": features,
            "prediction": prediction,
        }

        result_filename = f"{patient_id}_{timestamp}.json"
        result_path = os.path.join(results_dir, result_filename)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Result saved: {result_path}")

        payload["saved_to"] = result_path
        return payload

    def upload(self, file_name: str, content: bytes):
        raise NotImplementedError("upload не реализован — см. /voice/upload")

    def insert_in_db(self, payload: dict):
        raise NotImplementedError("insert_in_db не реализован")