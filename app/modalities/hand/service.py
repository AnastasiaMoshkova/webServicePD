import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from envyaml import EnvYAML

from app.modalities.base_service import BaseModalityService
from app.modalities.hand.processing.automarking import AutoMarking
from app.modalities.hand.processing.raw_data_processing import PreProcessing
from app.modalities.hand.processing.feature_extraction import FE

logger = logging.getLogger(__name__)


class HandTrackingService(BaseModalityService):
    """
    Состояние записи (recording/out/output_file/...) хранится не на самом
    сервисе (он один на всё приложение), а в словаре по session_id — иначе
    два одновременных пользователя писали бы видео в один и тот же файл.
    """

    def __init__(self, local_dir: str = "static/recordings/") -> None:
        self.base_dir = local_dir
        self._fe_config: Optional[Dict[str, Any]] = None
        self._sessions: Dict[str, Dict[str, Any]] = {}

        self.hand_data_processing = PreProcessing()
        self.automarker = AutoMarking()
        self.feature_extraction = FE(self._get_fe_config())

    def _get_fe_config(self) -> Dict[str, Any]:
        if self._fe_config is None:
            cfg_dir = Path(__file__).resolve().parent / "configs"
            self._fe_config = dict(EnvYAML(cfg_dir / "feature.yaml"))
        return self._fe_config

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "recording": False,
                "countdown_time": None,
                "start_time": None,
                "out": None,
                "output_file": None,
                "local_dir": os.path.join(self.base_dir, session_id),
            }
        return self._sessions[session_id]

    def reset_local_dir(self, session_id: str) -> None:
        s = self.get_session(session_id)
        shutil.rmtree(s["local_dir"], ignore_errors=True)
        os.makedirs(s["local_dir"], exist_ok=True)

    def start_record(self, session_id: str, duration: int = 20) -> None:
        s = self.get_session(session_id)
        if s["recording"]:
            return
        self.reset_local_dir(session_id)
        s["output_file"] = os.path.join(s["local_dir"], "test_video.mp4")
        s["recording"] = True
        s["out"] = None
        s["countdown_time"] = duration
        s["start_time"] = datetime.now()

    def stop_record(self, session_id: str, clear_output: bool = True) -> Optional[str]:
        s = self.get_session(session_id)
        s["recording"] = False
        if s["out"] is not None:
            s["out"].release()
            s["out"] = None
        saved_file = s["output_file"]
        if clear_output:
            s["output_file"] = None
        s["countdown_time"] = None
        s["start_time"] = None
        return saved_file

    def init_writer(self, session_id: str, frame_shape) -> None:
        s = self.get_session(session_id)
        if s["out"] is not None:
            return
        if not s["output_file"]:
            s["output_file"] = os.path.join(s["local_dir"], "test_video.mp4")
        h_cam, w_cam = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        s["out"] = cv2.VideoWriter(s["output_file"], fourcc, 20.0, (w_cam, h_cam))

    def upload(self, session_id: str, file_name: str, content: bytes) -> Dict[str, Any]:
        rel_path = (file_name or "").strip().lstrip("/")
        if not rel_path:
            raise ValueError("Invalid path")

        self.reset_local_dir(session_id)
        s = self.get_session(session_id)
        local_path = os.path.join(s["local_dir"], "test_video.mp4")
        with open(local_path, "wb") as f:
            f.write(content)

        logger.info("Saved locally: %s", local_path)
        return {"status": "success", "uploaded": rel_path}

    def raw_data_processing(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        s = self.get_session(session_id)
        patient_id = payload["patientId"]
        exercise = payload["exercise"]
        confidence = payload["confidence"]
        hand = payload["hand"]
        logger.info(
            "Получены данные: patientId=%s, exercise=%s, confidence=%s",
            patient_id,
            exercise,
            confidence,
        )

        fps = self.hand_data_processing.processing(s["local_dir"])
        max_p, min_p, max_a, min_a, values, frames = self.automarker.processing(
            s["local_dir"], exercise, fps, hand
        )
        features, features_norm = self.feature_extraction.processing(
            os.path.join(s["local_dir"], "auto_algoritm_MP"), exercise
        )
        timestamps = np.array(frames) / fps
        result = {
            "values": values.tolist() if hasattr(values, "tolist") else values,
            "frames": frames.tolist() if hasattr(frames, "tolist") else frames,
            "times": timestamps.tolist() if hasattr(timestamps, "tolist") else timestamps,
            "max_X": max_p.tolist() if hasattr(max_p, "tolist") else max_p,
            "min_X": min_p.tolist() if hasattr(min_p, "tolist") else min_p,
            "max_Y": max_a.tolist() if hasattr(max_a, "tolist") else max_a,
            "min_Y": min_a.tolist() if hasattr(min_a, "tolist") else min_a,
            "features_norm": features_norm,
            "features": features,
        }
        self.reset_local_dir(session_id)
        return result

    def predict_binary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("predict_binary is not implemented")

    def insert_in_db(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("insert_in_db is not implemented")
