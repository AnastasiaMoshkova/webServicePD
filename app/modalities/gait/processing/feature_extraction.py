import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from scipy.interpolate import interp1d
from typing import Dict, Any

from app.modalities.gait.processing.raw_data_processing import (
    bandpass_filter, min_max_normalize, FS, CFG
)

FEATURE_UNITS = {
    'step_time_mean': 'с', 'step_time_std': 'с', 'step_time_cv': 'у.е.',
    'cadence': 'шаги/мин', 'step_count_exact': 'шт.', 'segment_duration': 'с',
    'speed_m_s': 'м/с', 'turn_time_sec': 'с', 'freeze_index_mean': 'у.е.',
    'sample_entropy': 'у.е.', 'X_rms': 'м/с²', 'X_std': 'м/с²',
    'X_jerk_std': 'м/с³', 'X_dominant_freq': 'Гц', 'Y_rms': 'м/с²',
    'Y_std': 'м/с²', 'Y_jerk_std': 'м/с³', 'Y_dominant_freq': 'Гц',
    'Z_rms': 'м/с²', 'Z_std': 'м/с²', 'Z_jerk_std': 'м/с³',
    'Z_dominant_freq': 'Гц', 'magnitude_rms': 'м/с²', 'magnitude_std': 'м/с²',
    'magnitude_jerk_std': 'м/с³', 'magnitude_dominant_freq': 'Гц',
    'amplitude_mean': 'м/с²', 'amplitude_std': 'м/с²',
    'heel_strike_peak_mean': 'м/с²', 'heel_strike_peak_std': 'м/с²',
    'step_time_asi': '%', 'amplitude_asi': '%', 'cycle_symmetry_index': 'у.е.',
    'morphology_skewness': 'у.е.', 'morphology_kurtosis': 'у.е.'
}

FEATURE_NAMES_RU = {
    'step_time_mean': 'Среднее время шага',
    'step_time_std': 'Станд. откл. времени шага',
    'step_time_cv': 'Вариация времени шага (CV)',
    'cadence': 'Каденс (частота шагов)',
    'step_count_exact': 'Количество шагов',
    'segment_duration': 'Длительность сегмента',
    'speed_m_s': 'Скорость ходьбы',
    'turn_time_sec': 'Время поворота',
    'freeze_index_mean': 'Индекс застывания',
    'sample_entropy': 'Энтропия выборки',
    'X_rms': 'RMS (ось X)',
    'X_std': 'Станд. откл. (ось X)',
    'X_jerk_std': 'Рывок (ось X)',
    'X_dominant_freq': 'Доминантная частота (ось X)',
    'Y_rms': 'RMS (ось Y)',
    'Y_std': 'Станд. откл. (ось Y)',
    'Y_jerk_std': 'Рывок (ось Y)',
    'Y_dominant_freq': 'Доминантная частота (ось Y)',
    'Z_rms': 'RMS (ось Z)',
    'Z_std': 'Станд. откл. (ось Z)',
    'Z_jerk_std': 'Рывок (ось Z)',
    'Z_dominant_freq': 'Доминантная частота (ось Z)',
    'magnitude_rms': 'RMS (модуль ускорения)',
    'magnitude_std': 'Станд. откл. (модуль ускорения)',
    'magnitude_jerk_std': 'Рывок (модуль ускорения)',
    'magnitude_dominant_freq': 'Доминантная частота (модуль ускорения)',
    'amplitude_mean': 'Средняя амплитуда пиков',
    'amplitude_std': 'Станд. откл. амплитуды пиков',
    'heel_strike_peak_mean': 'Амплитуда касания пятки сред.',
    'heel_strike_peak_std': 'Амплитуда касания пятки станд.',
    'step_time_asi': 'Индекс асимметрии времени шага',
    'amplitude_asi': 'Индекс асимметрии амплитуды',
    'cycle_symmetry_index': 'Индекс симметрии цикла',
    'morphology_skewness': 'Асимметрия цикла ходьбы',
    'morphology_kurtosis': 'Эксцесс цикла ходьбы'
}

def estimate_step_distance(signal: np.ndarray) -> int:
    signal = signal - np.mean(signal)
    corr = correlate(signal, signal, mode='full')
    corr = corr[len(corr) // 2:]
    min_lag, max_lag = int(FS * 0.3), int(FS * 1.2)
    corr_range = corr[min_lag:max_lag]
    if len(corr_range) == 0:
        return int(FS * 0.5)
    return int(np.argmax(corr_range) + min_lag)

def detect_peaks_advanced(signal: np.ndarray):
    step_dist = estimate_step_distance(signal)
    height = np.percentile(signal, 70)
    peaks, props = find_peaks(
        signal,
        distance=int(step_dist * 0.45),
        height=height,
        prominence=np.std(signal) * 0.3,
    )
    return peaks, props

def calculate_freeze_index_from_array(signal: np.ndarray, fs: float) -> float:
    signal = np.array(signal, dtype=float)
    signal_centered = signal - np.mean(signal)
    yf = np.abs(rfft(signal_centered))
    xf = rfftfreq(len(signal_centered), 1 / fs)
    power_spectrum = yf ** 2
    locomotor_mask = (xf >= 0.5) & (xf <= 3.0)
    freeze_mask = (xf >= 3.0) & (xf <= 8.0)
    power_loco = np.sum(power_spectrum[locomotor_mask])
    power_freeze = np.sum(power_spectrum[freeze_mask])
    fi_value = float(power_freeze / power_loco) if power_loco > 0 else 0.0
    return fi_value

def calculate_dominant_freq(signal: np.ndarray, fs: float) -> float:
    signal = np.array(signal, dtype=float)
    signal_centered = signal - np.mean(signal)
    yf = np.abs(rfft(signal_centered))
    xf = rfftfreq(len(signal_centered), 1 / fs)
    mask = (xf >= 0.5) & (xf <= 7.0)
    xf_f, yf_f = xf[mask], yf[mask]
    if len(yf_f) == 0:
        return np.nan
    return float(xf_f[np.argmax(yf_f)])

def compute_morphology_kurtosis(signal: np.ndarray) -> float:
    signal = np.array(signal, dtype=float)
    if len(signal) < 4:
        return np.nan
    return float(scipy_kurtosis(signal, fisher=True))

def compute_morphology_skewness(signal: np.ndarray) -> float:
    signal = np.array(signal, dtype=float)
    if len(signal) < 4:
        return np.nan
    return float(scipy_skew(signal))

def compute_axis_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    signal = np.array(signal, dtype=float)
    jerk = np.diff(signal) * fs
    return {
        'rms': float(np.sqrt(np.mean(signal ** 2))),
        'std': float(np.std(signal)),
        'jerk_std': float(np.std(jerk)) if len(jerk) > 0 else np.nan,
        'dominant_freq': calculate_dominant_freq(signal, fs),
    }

def calculate_step_time_asi(step_times: np.ndarray) -> float:
    if len(step_times) < 4:
        return np.nan
    side_a = step_times[0::2]
    side_b = step_times[1::2]
    mean_a = np.mean(side_a)
    mean_b = np.mean(side_b)
    if (mean_a + mean_b) == 0:
        return np.nan
    asi = (2 * abs(mean_a - mean_b) / (mean_a + mean_b)) * 100
    return float(asi)

def calculate_amplitude_asi(amplitudes: np.ndarray) -> float:
    if len(amplitudes) < 4:
        return np.nan
    side_a = amplitudes[0::2]
    side_b = amplitudes[1::2]
    mean_a = np.mean(side_a)
    mean_b = np.mean(side_b)
    if (mean_a + mean_b) == 0:
        return np.nan
    asi = (2 * abs(mean_a - mean_b) / (mean_a + mean_b)) * 100
    return float(asi)

def calculate_cycle_symmetry_index(mean_cycle_shape: np.ndarray) -> float:
    half = len(mean_cycle_shape) // 2
    first_half = mean_cycle_shape[:half]
    second_half = mean_cycle_shape[half:2*half]
    if np.std(first_half) > 0 and np.std(second_half) > 0:
        corr = np.corrcoef(first_half, second_half)[0, 1]
        return float(corr)
    return np.nan

def sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    signal = np.array(signal, dtype=float)
    N = len(signal)
    if N < 10:
        return np.nan
    std_sig = np.std(signal)
    if std_sig == 0:
        return np.nan
    r_val = r * std_sig
    def _phi(m_val):
        x = np.array([signal[i:i + m_val] for i in range(N - m_val + 1)])
        C = np.sum([np.sum(np.max(np.abs(x - x_i), axis=1) <= r_val) - 1 for x_i in x])
        return C / (N - m_val + 1)
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return np.nan
    return float(-np.log(phi_m1 / phi_m))

def compute_all_segment_features(segment_df: pd.DataFrame) -> Dict[str, Any]:
    features = {}
    raw_x = segment_df['acc_x'].values if 'acc_x' in segment_df.columns else segment_df['acc_mag'].values
    filt_x = bandpass_filter(raw_x, lowcut=0.5, highcut=7.0, fs=FS, order=2)
    norm_x = min_max_normalize(filt_x)

    peaks, props = detect_peaks_advanced(norm_x)
    n_steps = len(peaks)

    if n_steps >= 2:
        step_times = np.diff(peaks) / FS
        step_times = step_times[step_times > 0]
        if len(step_times) > 0:
            step_time_mean = float(np.mean(step_times))
            step_time_std = float(np.std(step_times))
            step_time_cv = step_time_std / step_time_mean if step_time_mean > 0 else np.nan
            cadence = 60.0 / step_time_mean if step_time_mean > 0 else np.nan
            step_time_asi = calculate_step_time_asi(step_times)
        else:
            step_time_mean = step_time_std = step_time_cv = cadence = step_time_asi = np.nan
    else:
        step_time_mean = step_time_std = step_time_cv = cadence = step_time_asi = np.nan

    if n_steps >= 2 and 'peak_heights' in props:
        amplitudes = props['peak_heights']
        amplitude_mean = float(np.mean(amplitudes))
        amplitude_std = float(np.std(amplitudes))
        amplitude_asi = calculate_amplitude_asi(amplitudes)
        heel_strike_peak_mean = float(np.mean(amplitudes))
        heel_strike_peak_std = float(np.std(amplitudes))
    else:
        amplitude_mean = amplitude_std = amplitude_asi = heel_strike_peak_mean = heel_strike_peak_std = np.nan

    raw_mag = segment_df['acc_mag'].values if 'acc_mag' in segment_df.columns else raw_x
    filt_mag = bandpass_filter(raw_mag, lowcut=0.5, highcut=7.0, fs=FS, order=2)
    norm_mag = min_max_normalize(filt_mag)

    if 'time_sec' in segment_df.columns and len(segment_df) >= 2:
        segment_duration = float(segment_df['time_sec'].iloc[-1] - segment_df['time_sec'].iloc[0])
    else:
        segment_duration = float(len(segment_df) / FS)

    speed_m_s = CFG["walk_distance_meters"] / segment_duration if segment_duration > 0 else np.nan

    features.update({
        'step_time_mean': step_time_mean,
        'step_time_std': step_time_std,
        'step_time_cv': step_time_cv,
        'step_time_asi': step_time_asi,
        'cadence': cadence,
        'step_count_exact': n_steps,
        'segment_duration': segment_duration,
        'speed_m_s': speed_m_s,
        'sample_entropy': sample_entropy(norm_mag),
        'amplitude_mean': amplitude_mean,
        'amplitude_std': amplitude_std,
        'amplitude_asi': amplitude_asi,
        'heel_strike_peak_mean': heel_strike_peak_mean,
        'heel_strike_peak_std': heel_strike_peak_std,
    })

    features['freeze_index_mean'] = calculate_freeze_index_from_array(raw_mag, FS)

    axis_map = {'X': 'acc_x', 'Y': 'acc_y', 'Z': 'acc_z', 'magnitude': 'acc_mag'}
    for prefix, col in axis_map.items():
        raw_sig = segment_df[col].values if col in segment_df.columns else raw_x
        filt_sig = bandpass_filter(raw_sig, lowcut=0.5, highcut=7.0, fs=FS, order=2)
        norm_sig = min_max_normalize(filt_sig)
        ax_feats = compute_axis_features(norm_sig, FS)
        features[f'{prefix}_rms'] = ax_feats['rms']
        features[f'{prefix}_std'] = ax_feats['std']
        features[f'{prefix}_jerk_std'] = ax_feats['jerk_std']
        features[f'{prefix}_dominant_freq'] = ax_feats['dominant_freq']

    features['morphology_kurtosis'] = compute_morphology_kurtosis(norm_mag)
    features['morphology_skewness'] = compute_morphology_skewness(norm_mag)
    
    normalized_cycles = []
    if len(peaks) >= 3:
        # Peaks are indices inside the current segment. Using absolute time_sec here
        # would shift cycles whenever the segment starts after t=0.
        for i in range(len(peaks) - 2):
            start_idx, end_idx = int(peaks[i]), int(peaks[i + 2])
            cycle_data = norm_mag[start_idx:end_idx + 1]
            if len(cycle_data) > 10:
                f_interp = interp1d(
                    np.linspace(0, 1, len(cycle_data)),
                    cycle_data,
                    kind='cubic',
                    fill_value="extrapolate",
                )
                normalized_cycles.append(f_interp(np.linspace(0, 1, 100)))
    
    if normalized_cycles:
        mean_cycle_shape = np.mean(normalized_cycles, axis=0)
        features['cycle_symmetry_index'] = calculate_cycle_symmetry_index(mean_cycle_shape)
    else:
        features['cycle_symmetry_index'] = np.nan

    features['turn_time_sec'] = np.nan
    return features