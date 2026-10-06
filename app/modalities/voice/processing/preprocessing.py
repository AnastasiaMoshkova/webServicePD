from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    resample_rate: int = 48000
    hop_length: int = 512
    silence_threshold_db: int = 20
    lowcut: int = 80
    highcut: int = 8000
    filter_order: int = 4
    min_voice_duration_sec: float = 0.3


def _find_ffmpeg() -> Optional[str]:
    p = shutil.which("ffmpeg")
    if p:
        return p
    for cand in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ):
        if os.path.isfile(cand):
            return cand
    return None


def _ensure_wav(input_path: str, tmp_dir: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg not found in PATH. Не могу декодировать {ext} файл. "
            f"Установите ffmpeg: https://ffmpeg.org/download.html "
            f"(или запускайте через Docker — там ffmpeg уже есть)."
        )

    from pydub import AudioSegment

    os.makedirs(tmp_dir, exist_ok=True)
    wav_path = os.path.join(
        tmp_dir, os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    )

    AudioSegment.converter = ffmpeg
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"ffmpeg failed to decode {input_path}: {e}")
    logger.info(f"Converted {ext} → wav: {wav_path}")
    return wav_path


def _butter_bandpass(y: np.ndarray, sr: int, cfg: AudioConfig) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(cfg.filter_order, [cfg.lowcut, cfg.highcut],
                 btype="band", fs=sr, output="sos")
    return sosfilt(sos, y)


def _resample(y: np.ndarray, orig_sr: int, cfg: AudioConfig) -> tuple[np.ndarray, int]:
    if orig_sr == cfg.resample_rate:
        return y, orig_sr
    return librosa.resample(y, orig_sr=orig_sr, target_sr=cfg.resample_rate), cfg.resample_rate


def _find_longest_voice(y: np.ndarray, sr: int, cfg: AudioConfig) -> Optional[dict]:
    intervals = librosa.effects.split(y, top_db=cfg.silence_threshold_db, hop_length=cfg.hop_length)
    if len(intervals) == 0:
        return None
    durations = [(end - start) / sr for start, end in intervals]
    max_idx = int(np.argmax(durations))
    start_s, end_s = intervals[max_idx]
    return {
        "start_sample": int(start_s),
        "end_sample": int(end_s),
        "start_time": start_s / sr,
        "end_time": end_s / sr,
        "duration": float(durations[max_idx]),
    }


def preprocess_audio(
    input_path: str,
    output_wav_path: str,
    cfg: Optional[AudioConfig] = None,
    tmp_dir: str = "recordings/voice_tmp",
) -> dict:
    cfg = cfg or AudioConfig()

    wav_path = _ensure_wav(input_path, tmp_dir)

    y, orig_sr = librosa.load(wav_path, sr=None, mono=True)
    if y is None or len(y) == 0:
        raise ValueError("Empty or unreadable audio")
    logger.info(f"Loaded: {len(y) / orig_sr:.2f}s @ {orig_sr} Hz")

    y, sr = _resample(y, orig_sr, cfg)

    y = _butter_bandpass(y, sr, cfg)

    voice = _find_longest_voice(y, sr, cfg)
    if voice is None:
        raise ValueError("No voiced intervals detected in recording")
    if voice["duration"] < cfg.min_voice_duration_sec:
        raise ValueError(
            f"Longest voice interval too short: {voice['duration']:.2f}s "
            f"(< {cfg.min_voice_duration_sec}s)"
        )

    y = y[voice["start_sample"]:voice["end_sample"]]
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    y = y / (peak + 1e-8)

    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, y, sr)
    logger.info(f"Saved cleaned: {output_wav_path} ({voice['duration']:.2f}s)")

    return {
        "cleaned_path": output_wav_path,
        "sample_rate": int(sr),
        "voice_duration": voice["duration"],
        "voice_start_sec": voice["start_time"],
        "voice_end_sec": voice["end_time"],
        "waveform": y.astype(np.float32),
    }