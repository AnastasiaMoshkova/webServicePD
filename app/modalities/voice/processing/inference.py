from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.modalities.voice.models.resnet18_light import ResNet18LightModel

logger = logging.getLogger(__name__)

IMG_SIZE = 224


def _as_list(paths: str | Sequence[str]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return [str(p) for p in paths]


class VoiceClassifier:
    def __init__(
        self,
        weights_paths: str | Sequence[str] | None = None,
        weights_path: str | Sequence[str] | None = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        model_config: Optional[dict[str, Any]] = None,
    ):
        raw_paths = weights_paths if weights_paths is not None else weights_path
        if raw_paths is None:
            raise ValueError("weights_paths must be provided")

        self.weights_paths = _as_list(raw_paths)
        if not self.weights_paths:
            raise ValueError("weights_paths must contain at least one checkpoint")

        for p in self.weights_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Voice model checkpoint not found: {p}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = float(threshold)

        cfg = {
            "dropout": 0.3,
            "unfreeze_from": "layer4",
            "gradual_unfreeze": False,
            "init_mode": "custom",
            "pretrained_backbone": False,
        }
        if model_config:
            cfg.update(model_config)
        cfg["gradual_unfreeze"] = False

        self.models = []
        for idx, checkpoint_path in enumerate(self.weights_paths, start=1):
            logger.info("Building ResNet18-light fold %s on %s...", idx, self.device)
            model = ResNet18LightModel(num_classes=1, **cfg).to(self.device)
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("Fold %s missing keys: %s", idx, missing)
            unexpected_non_adv = [k for k in unexpected if not k.startswith("_adv_heads.")]
            if unexpected_non_adv:
                logger.warning("Fold %s unexpected keys: %s", idx, unexpected_non_adv)

            if missing:
                raise RuntimeError(f"Missing keys in checkpoint: {missing}")
            if unexpected_non_adv:
                raise RuntimeError(f"Unexpected keys: {unexpected_non_adv}")

            model.eval()
            self.models.append(model)

        logger.info(
            "Voice ensemble ready: %s fold(s), threshold=%.3f",
            len(self.models), self.threshold,
        )

    def _array_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 image, got {arr.shape}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)
        if tensor.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            tensor = F.interpolate(
                tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
            )
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_from_array(self, arr: np.ndarray) -> dict:
        x = self._array_to_tensor(arr)
        fold_probs = []
        for model in self.models:
            logit = model(x).flatten()
            fold_probs.append(float(torch.sigmoid(logit).item()))
        from app.modalities.voice.processing.debug_log import log_event

        log_event(
            "ensemble_predict",
            n_models=len(self.models),
            fold_probs=fold_probs,
            weights_paths=self.weights_paths,
            device=str(self.device),
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
        )

        prob = float(np.mean(fold_probs))
        cls = 1 if prob > self.threshold else 0
        return {
            "probability": prob,
            "class": int(cls),
            "label": "PD" if cls == 1 else "HC",
            "label_ru": "болен" if cls == 1 else "здоров",
            "threshold": self.threshold,
            "fold_probabilities": fold_probs,
            "ensemble_size": len(self.models),
            "model": "resnet18_light_three_channel_ensemble",
        }

    @torch.no_grad()
    def predict_from_png(self, png_path: str) -> dict:
        img = Image.open(png_path).convert("RGB")
        return self.predict_from_array(np.asarray(img, dtype=np.uint8))