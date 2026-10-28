"""
Признаки мимики по ключевым точкам MediaPipe FaceMesh.

Вход — массив [n_кадров, n_точек, 3] по точкам ALL_POINTS (кадры без лица = NaN)
для видео пробы и для видео нейтрального выражения.

  1. Расстояние между парами точек пробы, нормированное на межзрачковое (точки 467/472).
  2. Нормировка на нейтраль: улыбка — среднее по нейтрали; брови и нахмуривание — медиана
     нейтрали после удаления морганий (по EAR).
  3. Активный фрагмент упражнения выделяется автоматически (auto_window_probe).
  4. Агрегаты ряда: min/max/mean/median/std; скорость и ускорение — после сглаживания
     Savitzky–Golay (5, 2), СКО по знаковым разностям; углы пар.
"""
import numpy as np
from scipy.signal import savgol_filter

# ── пары точек, на которых обучены модели (порядок i_j как в именах колонок) ──
PAIRS = {
    'smile': ['50-39', '50-40', '50-92', '50-165', '50-203', '50-205', '50-206', '61-123', '61-186', '61-187',
              '61-207', '61-216', '92-40', '165-39', '186-123', '186-187', '186-207', '186-216', '187-123',
              '203-39', '203-165', '205-40', '205-92', '205-206', '206-40', '206-92', '207-123', '207-187',
              '216-123', '216-187', '216-207', '280-269', '280-270', '280-322', '280-391', '280-423', '280-425',
              '280-426', '308-352', '308-410', '308-411', '308-427', '308-436', '322-270', '391-269', '410-352',
              '410-411', '410-427', '410-436', '411-352', '423-269', '423-391', '425-270', '425-322', '425-426',
              '426-270', '426-322', '427-352', '427-411', '436-352', '436-411', '436-427'],
    'brow': ['46-107', '52-107', '53-107', '55-168', '63-4', '66-4', '66-10', '105-4', '105-10', '107-4',
             '107-10', '107-168', '276-336', '282-336', '283-336', '285-168', '293-4', '296-4', '296-10', '334-4',
             '334-10', '336-4', '336-10', '336-168'],
    'frown': ['46-276', '52-4', '52-151', '52-168', '52-282', '53-283', '55-4', '55-6', '55-8', '55-10', '55-285',
              '65-4', '65-8', '65-109', '65-295', '282-4', '282-151', '282-168', '285-4', '285-6', '285-8',
              '285-10', '295-4', '295-8', '295-338'],
}
EYE_POINTS = (467, 472)
EAR_POINTS = [159, 145, 158, 153, 33, 133, 386, 374, 385, 380, 362, 263]

ALL_POINTS = sorted({int(x) for v in PAIRS.values() for p in v for x in p.split('-')}
                    | {467, 468, 472, 473} | set(EAR_POINTS))   # 467/472 — нормировка, 468/473 — центры радужек
PIDX = {p: k for k, p in enumerate(ALL_POINTS)}

AGG_GEOM = ['min_norm', 'max_norm', 'mean_norm', 'median_norm', 'std_norm']
AGG_DYN = ['max_vel_norm', 'mean_vel_norm', 'std_vel_norm', 'max_acc_norm', 'std_acc_norm']
AGG_ANGLE = ['mean_angle', 'std_angle', 'range_angle']


def split_pair(p):
    i, j = p.split('-')
    return int(i), int(j)


def col_name(pair, agg):
    return f"dist_{pair.replace('-', '_')}_{agg}"


# ── ряды по кадрам ───────────────────────────────────────────────────────────
def valid_frames(lm):
    return ~np.isnan(lm[:, :, 0]).any(axis=1)


def point(lm, idx):
    return lm[:, PIDX[idx], :]


def eye_distance(lm):
    a, b = EYE_POINTS
    d = np.linalg.norm(point(lm, a) - point(lm, b), axis=1)
    return np.where(d > 0, d, 1.0)


def pair_series(lm, pair):
    """Нормированное на межзрачковое 3D-расстояние пары по кадрам."""
    i, j = split_pair(pair)
    return np.linalg.norm(point(lm, i) - point(lm, j), axis=1) / eye_distance(lm)


def pair_angle(lm, pair):
    i, j = split_pair(pair)
    d = point(lm, j) - point(lm, i)
    return np.arctan2(d[:, 1], d[:, 0])


def ear(lm):
    def dd(a, b):
        return np.linalg.norm(point(lm, a)[:, :2] - point(lm, b)[:, :2], axis=1)
    left = (dd(159, 145) + dd(158, 153)) / (2 * dd(33, 133))
    right = (dd(386, 374) + dd(385, 380)) / (2 * dd(362, 263))
    return (left + right) / 2


def blink_mask(e, min_abs_drop=0.08, min_rel_drop=0.20, window=31, margin=2, min_len=2):
    """Кадры моргания: резкое падение EAR относительно скользящей медианы (с запасом margin)."""
    base = np.asarray(__import__('pandas').Series(e).rolling(window, center=True, min_periods=1).median())
    drop = base - e
    raw = (drop >= min_abs_drop) & (drop / base >= min_rel_drop)
    out = np.zeros(len(e), bool)
    start = None
    for k, b in enumerate(raw):
        if b and start is None:
            start = k
        if (not b or k == len(raw) - 1) and start is not None:
            end = k - 1 if not b else k
            if end - start + 1 >= min_len:
                out[max(0, start - margin):min(len(e) - 1, end + margin) + 1] = True
            start = None
    return out


def interpolate_masked(x, mask):
    if not mask.any() or mask.all():
        return x
    idx = np.arange(len(x))
    y = x.copy()
    y[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return y


# ── нейтраль ─────────────────────────────────────────────────────────────────
# способ расчёта эталона по нейтрали: (статистика, удалять ли моргания)
NEUTRAL_RULES = {'smile': ('mean', False), 'brow': ('median', True), 'frown': ('median', True)}


def neutral_ref(lm_neu, probe):
    """Эталонное значение каждой пары пробы по видео нейтрального выражения."""
    stat, blink_clean = NEUTRAL_RULES[probe]
    lm = lm_neu[valid_frames(lm_neu)]
    mask = blink_mask(ear(lm)) if blink_clean else None
    ref = {}
    for p in PAIRS[probe]:
        s = pair_series(lm, p)
        if mask is not None:
            s = interpolate_masked(s, mask)
        ref[p] = float(np.median(s) if stat == 'median' else np.mean(s))
    return ref


# ── агрегаты ─────────────────────────────────────────────────────────────────
def aggregates(s):
    """Агрегаты ряда; скорость и ускорение — по сглаженному Savitzky–Golay (5, 2) ряду."""
    a = {'min_norm': np.min(s), 'max_norm': np.max(s), 'mean_norm': np.mean(s),
         'median_norm': np.median(s), 'std_norm': np.std(s)}
    sm = savgol_filter(s, 5, 2) if len(s) >= 5 else s
    vel = np.diff(sm)
    acc = np.diff(vel) if len(vel) > 1 else np.array([0.0])
    a.update(max_vel_norm=np.max(np.abs(vel)), mean_vel_norm=np.mean(np.abs(vel)), std_vel_norm=np.std(vel),
             max_acc_norm=np.max(np.abs(acc)), std_acc_norm=np.std(acc))
    return a


def angle_aggregates(ang):
    return {'mean_angle': np.mean(ang), 'std_angle': np.std(ang), 'range_angle': np.ptp(ang)}


def probe_features(lm_ex, lm_neu, probe, window=None):
    """
    Признаки одной пробы: {колонка: значение} для всех пар PAIRS[probe].
    window — (start, end) по номеру кадра включительно (auto_window_probe); None — всё видео.
    """
    ref = neutral_ref(lm_neu, probe)
    frames = np.arange(len(lm_ex))
    keep = valid_frames(lm_ex)
    if window is not None:
        keep &= (frames >= window[0]) & (frames <= window[1])
    lm = lm_ex[keep]
    row = {'num_frames': int(len(lm))}
    for p in PAIRS[probe]:
        s = pair_series(lm, p) / ref[p]
        for k, v in aggregates(s).items():
            row[col_name(p, k)] = float(v)
        for k, v in angle_aggregates(pair_angle(lm, p)).items():
            row[col_name(p, k)] = float(v)
    return row


# ── автообрезка активного фрагмента ──────────────────────────────────────────
def activity(lm_ex, lm_neu, probe, smooth=21):
    """
    Сигнал активности по кадрам: медиана по парам пробы |расстояние/нейтраль − 1|,
    сглаженная скользящим средним. Кадры без лица — NaN.
    """
    ref = neutral_ref(lm_neu, probe)
    dev = np.stack([pair_series(lm_ex, p) / ref[p] - 1 for p in PAIRS[probe]])
    a = np.median(np.abs(dev), axis=0)
    ok = ~np.isnan(a)
    k = np.ones(smooth)
    num = np.convolve(np.where(ok, a, 0.0), k, 'same')
    den = np.convolve(ok.astype(float), k, 'same')
    out = np.where(ok, num / np.maximum(den, 1), np.nan)
    return out


def auto_window(lm_ex, lm_neu, probe, alpha=0.3, pad=10, smooth=21):
    """
    Автоматическое окно упражнения (start, end) по номеру кадра включительно:
    от первого до последнего кадра, где активность выше
    p5 + alpha·(p95 − p5), с запасом pad кадров. Без ручных параметров на видео.
    """
    a = activity(lm_ex, lm_neu, probe, smooth)
    ok = ~np.isnan(a)
    if ok.sum() < 3 * smooth:
        return None
    lo, hi = np.percentile(a[ok], [5, 95])
    idx = np.where(ok & (a > lo + alpha * (hi - lo)))[0]
    if len(idx) == 0:
        return None
    return max(0, int(idx[0]) - pad), min(len(a) - 1, int(idx[-1]) + pad)


# Параметры автообрезки по пробам (одни на все видео)
AUTO_PARAMS = {'smile': dict(alpha=0.3, pad=0), 'frown': dict(alpha=0.3, pad=0), 'brow': dict(alpha=0.2, pad=0)}


def auto_window_probe(lm_ex, lm_neu, probe):
    """Окно активного фрагмента пробы с параметрами AUTO_PARAMS."""
    return auto_window(lm_ex, lm_neu, probe, **AUTO_PARAMS[probe])
