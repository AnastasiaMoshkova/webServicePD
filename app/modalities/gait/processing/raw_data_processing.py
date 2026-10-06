import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt, detrend
from scipy.ndimage import median_filter
from typing import Dict, Tuple, Optional

CFG = {
    "target_fs": 100.0,
    "win_sec_det": 2.5,
    "step_sec_det": 0.5,
    "win_sec_seg": 1.5,
    "step_sec_seg": 0.2,
    "acc_cutoff": 20.0,
    "gyr_cutoff": 15.0,
    "walk_distance_meters": 10.0,
}
FS = CFG["target_fs"]
WIN_SIZE_DET = int(FS * CFG["win_sec_det"])
STEP_SIZE_DET = int(FS * CFG["step_sec_det"])
WIN_SIZE_SEG = int(FS * CFG["win_sec_seg"])
STEP_SIZE_SEG = int(FS * CFG["step_sec_seg"])


def _safe_filtfilt(b, a, data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if data.size <= padlen:
        return data.copy()
    return filtfilt(b, a, data)


def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    cutoff = min(float(cutoff), nyq * 0.99)
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return _safe_filtfilt(b, a, data)


def butter_highpass(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype="high", analog=False, output="sos")
    if data.size <= 3 * (2 * sos.shape[0] + 1):
        return data.copy()
    return sosfiltfilt(sos, data)


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float = FS, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    signal = np.asarray(signal, dtype=float)
    if signal.size < 15:
        return signal.copy()
    highcut = min(float(highcut), nyq * 0.99)
    if lowcut <= 0:
        b, a = butter(order, highcut / nyq, btype="low")
    else:
        b, a = butter(order, [float(lowcut) / nyq, highcut / nyq], btype="band")
    return _safe_filtfilt(b, a, signal)


def min_max_normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal
    x_min, x_max = np.nanmin(signal), np.nanmax(signal)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max - x_min < 1e-10:
        return np.zeros_like(signal)
    return (signal - x_min) / (x_max - x_min)


def calculate_imu_angles(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc_x = df["acc_x"].to_numpy(dtype=float)
    acc_y = df["acc_y"].to_numpy(dtype=float)
    acc_z = df["acc_z"].to_numpy(dtype=float)
    pitch = np.arctan2(acc_x, np.sqrt(acc_y ** 2 + acc_z ** 2))
    roll = np.arctan2(acc_y, acc_z)
    dt = 1.0 / CFG["target_fs"]
    yaw = np.cumsum(df["gyr_z"].to_numpy(dtype=float) * dt) if "gyr_z" in df.columns else np.zeros_like(pitch)
    return yaw, pitch, roll


def resample_and_prep(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Resample a synchronized ACC+GYR dataframe to the model frequency and build 5 DL channels."""
    required = ["time_sec", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {', '.join(missing)}")

    work = df[required].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_sec"])
    work = work.sort_values("time_sec").drop_duplicates("time_sec")
    for c in required[1:]:
        work[c] = pd.to_numeric(work[c], errors="coerce").interpolate(limit_direction="both").fillna(0.0)

    if len(work) < 20:
        raise ValueError("Слишком короткая запись: требуется не менее 20 отсчётов.")

    t_old = work["time_sec"].to_numpy(dtype=float)
    t_old = t_old - t_old[0]
    duration = float(t_old[-1])
    if duration <= 0:
        raise ValueError("Некорректная временная шкала сигнала.")

    dt = 1.0 / CFG["target_fs"]
    t_new = np.arange(0.0, duration + dt * 0.5, dt, dtype=float)
    resampled = {"time_sec": t_new}
    for c in required[1:]:
        resampled[c] = np.interp(t_new, t_old, work[c].to_numpy(dtype=float))

    df_res = pd.DataFrame(resampled)
    df_res["ori_yaw"], _, _ = calculate_imu_angles(df_res)

    for ax in ("x", "y", "z"):
        df_res[f"acc_{ax}"] = butter_lowpass(df_res[f"acc_{ax}"].to_numpy(), CFG["acc_cutoff"], FS)
        df_res[f"gyr_{ax}"] = butter_lowpass(df_res[f"gyr_{ax}"].to_numpy(), CFG["gyr_cutoff"], FS)

    df_res["ori_yaw"] = butter_lowpass(df_res["ori_yaw"].to_numpy(), CFG["gyr_cutoff"], FS)
    try:
        df_res["ori_yaw"] = butter_highpass(df_res["ori_yaw"].to_numpy(), cutoff=0.1, fs=FS)
    except ValueError:
        df_res["ori_yaw"] = detrend(df_res["ori_yaw"].to_numpy())

    df_res["acc_mag"] = np.sqrt(df_res["acc_x"] ** 2 + df_res["acc_y"] ** 2 + df_res["acc_z"] ** 2)
    df_res["gyr_mag"] = np.sqrt(df_res["gyr_x"] ** 2 + df_res["gyr_y"] ** 2 + df_res["gyr_z"] ** 2)

    for c in ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "ori_yaw", "acc_mag", "gyr_mag"):
        grad = np.gradient(df_res[c].to_numpy(dtype=float), dt)
        df_res[f"{c}_d1"] = butter_lowpass(grad, cutoff=3.0, fs=FS)

    model_features = ["acc_mag", "gyr_mag", "ori_yaw", "acc_mag_d1", "gyr_mag_d1"]
    model_matrix = np.nan_to_num(df_res[model_features].to_numpy(dtype=np.float32))
    return model_matrix, df_res["time_sec"].to_numpy(dtype=float), df_res


def resample_only(df: pd.DataFrame) -> pd.DataFrame:
    required = ["time_sec", "acc_x", "acc_y", "acc_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {', '.join(missing)}")
    work = df[required].copy().sort_values("time_sec").drop_duplicates("time_sec")
    t_old = work["time_sec"].to_numpy(dtype=float)
    t_old = t_old - t_old[0]
    dt = 1.0 / FS
    t_new = np.arange(0.0, t_old[-1] + dt * 0.5, dt)
    out = {"time_sec": t_new}
    for c in required[1:]:
        vals = pd.to_numeric(work[c], errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy()
        out[c] = np.interp(t_new, t_old, vals)
    df_res = pd.DataFrame(out)
    df_res["acc_mag"] = np.sqrt(df_res["acc_x"] ** 2 + df_res["acc_y"] ** 2 + df_res["acc_z"] ** 2)
    return df_res


def create_windows(data_mat: np.ndarray, time_array: np.ndarray, win_size: int, step_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(data_mat) < win_size:
        return np.empty((0, win_size, data_mat.shape[1]), dtype=np.float32), np.empty((0,), dtype=float)
    X, T = [], []
    for i in range(0, len(data_mat) - win_size + 1, step_size):
        X.append(data_mat[i:i + win_size])
        T.append(time_array[i + win_size // 2])
    return np.asarray(X, dtype=np.float32), np.asarray(T, dtype=float)


def smooth_predictions(preds: np.ndarray, k: int = 5) -> np.ndarray:
    preds = np.asarray(preds, dtype=int)
    if preds.size == 0:
        return preds
    k = max(1, min(int(k), preds.size if preds.size % 2 == 1 else max(1, preds.size - 1)))
    return median_filter(preds, size=k).astype(int)


def find_true_movement_boundaries(det_pred: np.ndarray, min_duration_sec: float = 1.5) -> Optional[Tuple[int, int]]:
    min_windows = max(1, int(np.ceil(min_duration_sec / CFG["step_sec_det"])))
    padded = np.concatenate(([0], np.clip(det_pred, 0, 1), [0]))
    diff = np.diff(padded)
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    valid = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_windows]
    return max(valid, key=lambda x: x[1] - x[0]) if valid else None


def extract_precise_timestamps(
    time_signals: np.ndarray,
    point_seg_pred: np.ndarray,
    movement_start: float,
    movement_end: float,
) -> Dict[str, Optional[float]]:
    """Extract the expected walk -> turn -> walk -> turn boundaries from pointwise classes (walk=0, turn=1)."""
    timestamps = {"T1": float(movement_start), "T2": None, "T3": None, "T4": None, "T5": float(movement_end)}
    mask = (time_signals >= movement_start) & (time_signals <= movement_end)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return timestamps

    start_i, end_i = int(idx[0]), int(idx[-1])
    transitions = []
    for i in range(start_i + 1, end_i + 1):
        if point_seg_pred[i] != point_seg_pred[i - 1]:
            transitions.append((float(time_signals[i]), int(point_seg_pred[i])))

    for t, cls in transitions:
        if cls == 1 and timestamps["T2"] is None:
            timestamps["T2"] = t
        elif cls == 0 and timestamps["T2"] is not None and timestamps["T3"] is None:
            timestamps["T3"] = t
        elif cls == 1 and timestamps["T3"] is not None and timestamps["T4"] is None:
            timestamps["T4"] = t
            break
    return timestamps
