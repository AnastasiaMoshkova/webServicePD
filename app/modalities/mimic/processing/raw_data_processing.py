import base64
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np

from app.modalities.mimic.processing.feature_extraction import FeatureExtraction

mp_face_mesh = mp.solutions.face_mesh


def parse_col(col):
    """
    dist_426_270_mean_vel_norm -> ('426-270', 'mean_vel_norm')
    Парсит имя колонки: два целых ID → пара, остаток → агрегат.
    """
    parts = col[len("dist_"):].split("_")
    id_parts = []
    for i, p in enumerate(parts):
        if p.isdigit():
            id_parts.append(p)
        else:
            return "-".join(id_parts), "_".join(parts[i:])
    return "-".join(id_parts), ""


def compute_aggregates(series):
    """
    10 агрегатов нормированного временного ряда (по кадрам).
    Минимум 2 кадра для скорости, 3 — для ускорения.
    """
    arr = np.asarray(series, dtype=float)
    aggs = {
        "min_norm":    float(np.min(arr)),
        "max_norm":    float(np.max(arr)),
        "mean_norm":   float(np.mean(arr)),
        "median_norm": float(np.median(arr)),
        "std_norm":    float(np.std(arr)),
    }
    if len(arr) >= 2:
        vel = np.abs(np.diff(arr))
        aggs["max_vel_norm"]  = float(np.max(vel))
        aggs["mean_vel_norm"] = float(np.mean(vel))
        aggs["std_vel_norm"]  = float(np.std(vel))
    else:
        aggs["max_vel_norm"] = aggs["mean_vel_norm"] = aggs["std_vel_norm"] = 0.0
    if len(arr) >= 3:
        vel = np.abs(np.diff(arr))
        acc = np.abs(np.diff(vel))
        aggs["max_acc_norm"] = float(np.max(acc))
        aggs["std_acc_norm"] = float(np.std(acc))
    else:
        aggs["max_acc_norm"] = aggs["std_acc_norm"] = 0.0
    return aggs


def build_feature_row(normalized_series, feature_columns):
    """
    Строит строку признаков для модели из нормированных временных рядов.
    Возвращает (feature_row_dict, missing_columns).
    """
    pair_aggs = {pk: compute_aggregates(s) for pk, s in normalized_series.items()}
    feature_row = {}
    missing = []
    for col in feature_columns:
        pair_key, agg = parse_col(col)
        agg_dict = pair_aggs.get(pair_key)
        if agg_dict is None:
            missing.append(col)
            feature_row[col] = 0.0
        else:
            feature_row[col] = agg_dict.get(agg, 0.0)
    return feature_row, missing


def quality_level(quality, n_face_frames):
    rate  = quality.get("detection_rate", 0.0)
    total = quality.get("total_frames", 0)
    if n_face_frames < 60:
        level   = "too_short"
        message = f"Слишком короткая запись ({n_face_frames} кадров с лицом). Запиши не меньше 5 секунд."
    elif rate < 0.70:
        level   = "bad"
        message = f"Лицо распознано лишь в {rate * 100:.0f}% кадров. Рекомендуется пересъёмка."
    elif rate < 0.90:
        level   = "warn"
        message = f"Лицо распознано в {rate * 100:.0f}% кадров. Результат можно улучшить пересъёмкой."
    else:
        level   = "ok"
        message = f"Качество записи хорошее ({rate * 100:.0f}% кадров с лицом)."
    return {
        "level": level, "message": message,
        "detection_rate": rate, "total_frames": total, "face_frames": n_face_frames,
    }


def im_2_b64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def rolling_median(arr, window=3):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr.tolist()
    pad    = window // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    out    = np.array([np.median(padded[i:i + window]) for i in range(len(arr))])
    return out.tolist()


def collect_pair_history(file_path, exercise_type):
    """
    Прогоняет видео через FaceMesh.
    Возвращает:
      history: {pair_key: [eye_normalized_distance_per_frame]}
      fps, best_frame_b64, best_landmarks, quality
    """
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    extractor = FeatureExtraction()
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    history       = defaultdict(list)
    max_global_avg = -1.0
    best_frame_b64 = None
    best_landmarks = []
    total_frames   = 0
    face_frames    = 0

    while cap.isOpened():
        ok, image = cap.read()
        if not ok:
            break
        total_frames += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            lm    = results.multi_face_landmarks[0].landmark
            pairs = extractor.extract_features(lm, exercise_type)
            if pairs:
                face_frames += 1
                total = 0.0
                for k, v in pairs.items():
                    history[k].append(v)
                    total += v
                avg = total / len(pairs)
                if avg > max_global_avg:
                    max_global_avg  = avg
                    best_frame_b64  = im_2_b64(image)
                    h, w, _         = image.shape
                    best_landmarks  = [[p.x * w, p.y * h] for p in lm]

    cap.release()
    face_mesh.close()
    quality = {
        "total_frames":   total_frames,
        "face_frames":    face_frames,
        "detection_rate": (face_frames / total_frames) if total_frames else 0.0,
    }
    return history, fps, best_frame_b64, best_landmarks, quality
