from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import librosa
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreeChannelConfig:
    sample_rate: int = 16000
    skip_sec: float = 0.5
    duration_sec: float = 1.5
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 40
    n_chroma: int = 12
    img_size: int = 224


def _extract_segment(y, sr, cfg):
    length = int(sr * cfg.duration_sec)
    skip = int(sr * cfg.skip_sec)
    if len(y) == 0:
        raise ValueError("Empty audio for three-channel input")
    start = skip if len(y) > skip else 0
    seg = y[start:start + length]
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)), mode="constant")
    return seg.astype(np.float32)


def _compute_features(
        y: np.ndarray, sr: int, cfg: ThreeChannelConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_chroma=cfg.n_chroma,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    ).astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    ).astype(np.float32)

    return mel_db, chroma, mfcc


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _resize_channel(arr: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(arr).resize((size, size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def generate_three_channel(
        audio_path: str,
        output_png_path: str | None = None,
        cfg: ThreeChannelConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ThreeChannelConfig()
    y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    if y is None or len(y) == 0:
        raise ValueError(...)

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    seg = _extract_segment(y, sr, cfg)
    mel_db, chroma, mfcc = _compute_features(seg, sr, cfg)

    channels = [
        _resize_channel(_normalize_to_uint8(mel_db), cfg.img_size),
        _resize_channel(_normalize_to_uint8(chroma), cfg.img_size),
        _resize_channel(_normalize_to_uint8(mfcc), cfg.img_size),
    ]
    img_arr = np.stack(channels, axis=-1)

    if output_png_path:
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        Image.fromarray(img_arr, mode="RGB").save(output_png_path)
        logger.info("Three-channel image saved: %s", output_png_path)

    return img_arr