"""
Переобучение моделей модуля «Мимика» на новых данных (по шаблону train_template.py).

Модели уже выбраны и настроены; скрипт переобучает ТЕ ЖЕ модели с ТЕМИ ЖЕ настройками
и кладёт веса туда, откуда их читает сервис.

Что берётся из веб-сервиса (ничего не копируется):
  * трекинг лица         -> processing/raw_data_processing.collect_landmarks
  * признаки             -> processing/mimic_features: auto_window_probe + probe_features
                            (те же вызовы, что в MimicService.process_exercise)
  * модели               -> models/__init__.py: build_probe_model, build_fusion_model, feature_columns
  * пути, порог, параметры моделей и обучения -> configs/mimic.yaml (его же читает сервис)

Данные: <папка>/<PatientN|HealthyN>/r<k>/face/<код>[_m<N>]_<id>.mp4,
коды видео — training.videos (нейтраль p1, брови p3, улыбка p5, нахмуривание p11).
Записи с другими именами файлов не загружаются — их нужно переименовать.
Метки — из PD_CSV / HEALTHY_CSV (столбец patient_id); исключённые записи — EXCLUDE_CSV.

Запуск из корня проекта:
    python -m app.modalities.mimic.models.train

Выход: веса по путям model.weights_paths; training_report.json рядом с весами;
       _backup/<дата_время>/ — прежние веса и отчёт. После обучения перезапустите сервис.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from app.modalities.mimic import models as mimic_models
from app.modalities.mimic.processing import mimic_features as mf
from app.modalities.mimic.processing.raw_data_processing import collect_landmarks

logger = logging.getLogger("train")

# --------------------------------------------------------------------------- #
#  Пути к данным (от корня проекта; можно переопределить переменными окружения,
#  несколько папок в MIMIC_DATA_DIRS — через os.pathsep: ":" в Linux/macOS, ";" в Windows)
# --------------------------------------------------------------------------- #
MODALITY = "mimic"
CONFIG_PATH = "app/modalities/mimic/configs/mimic.yaml"     # тот же конфиг, что читает сервис
DATA_DIRS = os.environ.get(
    "MIMIC_DATA_DIRS", os.pathsep.join(["data/mimic/PD face", "data/mimic/HEALTHY face"])
).split(os.pathsep)
PD_CSV = os.environ.get("MIMIC_PD_CSV", "data/mimic/PD.csv")
HEALTHY_CSV = os.environ.get("MIMIC_HEALTHY_CSV", "data/mimic/Healthy.csv")
EXCLUDE_CSV = os.environ.get("MIMIC_EXCLUDE_CSV", "data/mimic/exclude.csv")   # patient_id, retake, probe, reason
CSV_ID_COLUMN = "patient_id"
# Ключевые точки видео кэшируются (ключ — путь, размер и время изменения файла),
# чтобы повторный запуск не гонял MediaPipe заново. Пустая строка — без кэша.
LANDMARK_CACHE = os.environ.get("MIMIC_LANDMARK_CACHE", "results/mimic/train_landmarks")
N_WORKERS = int(os.environ.get("MIMIC_WORKERS", max(1, (os.cpu_count() or 2) - 1)))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROBES = mimic_models.PROBES


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def norm_id(x) -> str:
    """'Patient 7 ', 'patient7', 7.0 -> одинаковый вид для сравнения."""
    s = re.sub(r"\s+", "", str(x)).lower()
    return re.sub(r"\.0$", "", s)


# ── трекинг (в отдельных процессах) ──────────────────────────────────────────
def _track(args) -> tuple[str, str | None]:
    """Ключевые точки одного видео через collect_landmarks сервиса → npy в кэше."""
    video, probe, out = args
    if os.path.exists(out):
        return video, None
    try:
        lm, *_ = collect_landmarks(video, probe=probe)
        if not mf.valid_frames(lm).any():
            return video, "лицо не найдено ни в одном кадре"
        tmp = out + ".tmp.npy"
        np.save(tmp, lm)
        os.replace(tmp, out)
        return video, None
    except Exception as e:  # битый файл не останавливает загрузку
        return video, f"{type(e).__name__}: {e}"


def _cache_file(cache_dir: Path, video: str) -> str:
    st = os.stat(video)
    key = hashlib.md5(f"{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
    return str(cache_dir / f"{key}.npy")


# =========================================================================== #
#  1. load_data — загрузка датасета с диска
# =========================================================================== #
def load_data(
    data_dirs: list[str | Path],
    pd_csv: str | Path | None = None,
    healthy_csv: str | Path | None = None,
) -> dict:
    """
    Возвращает {проба: DataFrame}; строка = запись (пациент × ретейк),
    столбцы: patient_id, retake, label (1 = PD, 0 = Healthy), признаки модели этой пробы
    (ровно те, что MimicService подаёт модели), num_frames, window_start, window_end.
    """
    cfg = mimic_models.load_config()
    training_cfg, model_cfg = cfg["training"], cfg["model"]

    # ── метки ──
    labels, canon = {}, {}
    for path, lab in ((pd_csv, 1), (healthy_csv, 0)):
        ids = pd.read_csv(_abs(path), dtype=str)[CSV_ID_COLUMN].dropna()
        for raw in ids:
            k = norm_id(raw)
            if k in labels and labels[k] != lab:
                raise ValueError(f"Пациент {raw} есть и в списке PD, и в списке здоровых")
            labels[k], canon[k] = lab, raw.strip()
    logger.info("Метки: PD %d, Healthy %d", sum(v == 1 for v in labels.values()),
                sum(v == 0 for v in labels.values()))

    # ── обход диска ──
    codes = training_cfg["videos"]
    retakes = set(training_cfg["retakes"])
    exclude = {p: set() for p in PROBES}
    if EXCLUDE_CSV and _abs(EXCLUDE_CSV).exists():
        for _, x in pd.read_csv(_abs(EXCLUDE_CSV), dtype=str).iterrows():
            exclude[x["probe"].strip()].add((norm_id(x["patient_id"]), x["retake"].strip()))
    records, not_listed, other_retakes = [], set(), []
    for d in data_dirs:
        for pdir in sorted(Path(d).iterdir()):
            if not pdir.is_dir():
                continue
            k = norm_id(pdir.name)
            if k not in labels:
                not_listed.add(pdir.name)
                continue
            for rdir in sorted(pdir.iterdir()):
                if not rdir.is_dir() or not re.fullmatch(r"r\d+", rdir.name):
                    continue
                if rdir.name not in retakes:
                    other_retakes.append(f"{pdir.name}/{rdir.name}")
                    continue
                face = rdir / "face"
                files = sorted(face.iterdir()) if face.is_dir() else []
                vids = {}
                for role, code in codes.items():
                    pat = re.compile(rf"^{code}(_m\d+)?_{re.escape(pdir.name)}\.mp4$", re.IGNORECASE)
                    hit = [f for f in files if pat.match(f.name)]
                    vids[role] = str(hit[0]) if hit else None
                records.append({"key": k, "patient_id": canon[k], "retake": rdir.name,
                                "label": labels[k], **vids})
    if not_listed:
        logger.warning("Папки пациентов, которых нет ни в одном CSV (пропущены): %s", sorted(not_listed))
    if other_retakes:
        logger.warning("Ретейки вне training.retakes (пропущены): %s", other_retakes)
    found = {r["key"] for r in records}
    missing_patients = sorted(canon[k] for k in labels if k not in found)
    if missing_patients:
        logger.warning("Пациенты из CSV, которых нет на диске: %s", missing_patients)

    # ── трекинг всех нужных видео ──
    cache_dir = _abs(LANDMARK_CACHE) if LANDMARK_CACHE else _abs("results/mimic/_tmp_landmarks")
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for r in records:
        for role in ("neutral",) + PROBES:
            if r[role]:
                jobs[r[role]] = (r[role], None if role == "neutral" else role, _cache_file(cache_dir, r[role]))
    todo = [j for j in jobs.values() if not os.path.exists(j[2])]
    logger.info("Видео: %d, нужно отследить: %d (процессов: %d)", len(jobs), len(todo), N_WORKERS)
    broken = {}
    with Pool(N_WORKERS) as pool:
        for i, (video, err) in enumerate(pool.imap_unordered(_track, todo), 1):
            if err:
                broken[video] = err
            if i % 50 == 0 or i == len(todo):
                logger.info("  трекинг %d/%d", i, len(todo))
    if broken:
        logger.warning("Битые/нечитаемые видео (пропущены): %s", broken)

    # ── признаки: те же вызовы, что в MimicService.process_exercise ──
    dataset, no_video = {}, []
    for probe in PROBES:
        cols = mimic_models.feature_columns(model_cfg, probe)
        rows = []
        for r in records:
            if (r["key"], r["retake"]) in exclude[probe]:
                continue
            vn, ve = r["neutral"], r[probe]
            if not vn or not ve:
                no_video.append(f"{r['patient_id']}/{r['retake']}/{probe}")
                continue
            if vn in broken or ve in broken:
                continue
            lm_neu, lm = np.load(jobs[vn][2]), np.load(jobs[ve][2])
            window = mf.auto_window_probe(lm, lm_neu, probe)
            f = mf.probe_features(lm, lm_neu, probe, window)
            rows.append({"patient_id": r["patient_id"], "retake": r["retake"], "label": r["label"],
                         "num_frames": f["num_frames"],
                         "window_start": window[0] if window else None,
                         "window_end": window[1] if window else None,
                         **{c: f[c] for c in cols}})
        df = pd.DataFrame(rows)
        dataset[probe] = df
        logger.info("%-6s записей %d (PD %d, Healthy %d), пациентов %d (PD %d, Healthy %d)", probe, len(df),
                    (df.label == 1).sum(), (df.label == 0).sum(), df.patient_id.nunique(),
                    df[df.label == 1].patient_id.nunique(), df[df.label == 0].patient_id.nunique())
    unnamed = sorted({f"{r['patient_id']}/{r['retake']}" for r in records
                      if not any(r[role] for role in ("neutral",) + PROBES)})
    if unnamed:
        logger.warning("Папки без видео с именами вида <код>_<id>.mp4 — переименуйте файлы: %s", unnamed)
    no_video = [x for x in no_video if "/".join(x.split("/")[:2]) not in set(unnamed)]
    if no_video:
        logger.warning("Нет нужного видео (запись пропущена): %s", no_video)
    if not LANDMARK_CACHE:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return dataset


# =========================================================================== #
#  2. train_model — обучение, сохранение модели, расчёт и сохранение метрик
# =========================================================================== #
def fit_and_evaluate(dataset: dict, service_cfg: dict) -> tuple[dict, Any, dict, dict]:
    """
    Обучение без записи на диск: OOF-метрики и модели, обученные на всех данных.
    Возвращает (модели проб, мета-модель, метрики, сводка по данным).
    """
    mc, tc = service_cfg["model"], service_cfg["training"]
    seed, n_folds, thr = int(tc["seed"]), int(tc["n_folds"]), float(mc["threshold"])
    models, metrics, info, oof_pat, labels = {}, {}, {}, {}, {}
    for probe in PROBES:
        df = dataset[probe].reset_index(drop=True)
        cols = mimic_models.feature_columns(mc, probe)
        X, y, g = df[cols], df["label"].to_numpy(), df["patient_id"].to_numpy()
        oof = np.full(len(df), np.nan)
        for tr, te in StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y, g):
            m = mimic_models.build_probe_model(mc, probe, y[tr], seed)
            m.fit(X.iloc[tr], y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        pat = pd.DataFrame({"id": g, "p": oof, "y": y}).groupby("id").agg(p=("p", "mean"), y=("y", "first"))
        oof_pat[probe] = pat["p"]
        labels.update(pat["y"].to_dict())
        metrics[f"{probe} (записи)"] = compute_metrics(y, oof, thr)
        metrics[f"{probe} (пациенты)"] = compute_metrics(pat["y"].to_numpy(), pat["p"].to_numpy(), thr)

        final = mimic_models.build_probe_model(mc, probe, y, seed)
        final.fit(X, y)
        models[probe] = {"pipeline": final, "feature_columns": cols, "feature_stats": _feature_stats(X, y)}
        info[probe] = {"records": int(len(df)), "patients": int(df.patient_id.nunique()),
                       "pd_records": int((y == 1).sum()), "healthy_records": int((y == 0).sum())}

    # мета-модель: OOF-вероятности пациентов, общих для трёх проб
    common = sorted(set.intersection(*[set(oof_pat[p].index) for p in PROBES]))
    X_meta = pd.DataFrame({p: oof_pat[p].loc[common].to_numpy() for p in mimic_models.FUSION_INPUTS})
    y_meta = np.array([labels[i] for i in common])
    fus_oof = cross_val_predict(mimic_models.build_fusion_model(mc, seed), X_meta, y_meta,
                                cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed),
                                method="predict_proba")[:, 1]
    metrics["fusion (пациенты)"] = compute_metrics(y_meta, fus_oof, thr)
    meta = mimic_models.build_fusion_model(mc, seed).fit(X_meta, y_meta)
    info["fusion"] = {"patients": len(common), "pd_patients": int(y_meta.sum()),
                      "healthy_patients": int((y_meta == 0).sum())}
    return models, meta, metrics, info


def train_model(dataset: dict, service_cfg: dict) -> dict:
    mc, tc = service_cfg["model"], service_cfg["training"]
    models, meta, metrics, info = fit_and_evaluate(dataset, service_cfg)

    paths = {name: mimic_models.weights_path(mc, name) for name in list(PROBES) + ["fusion"]}
    wdir = paths["smile"].parent
    report_path = wdir / "training_report.json"

    # бэкап прежних весов и отчёта
    prev_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    backup = wdir / "_backup" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup.mkdir(parents=True, exist_ok=True)
    for p in list(paths.values()) + [report_path]:
        if p.exists():
            shutil.copy2(p, backup / p.name)
    logger.info("Прежние веса и отчёт: %s", backup)

    for probe in PROBES:
        joblib.dump(models[probe], paths[probe])
    joblib.dump(meta, paths["fusion"])

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "data": info,
        "n_folds": int(tc["n_folds"]),
        "training": tc,
        "threshold": float(mc["threshold"]),
        "weights_paths": {k: str(v.relative_to(PROJECT_ROOT)) if v.is_relative_to(PROJECT_ROOT) else str(v)
                          for k, v in paths.items()},
        "fusion_coef": dict(zip(mimic_models.FUSION_INPUTS, np.round(meta.coef_[0], 4).tolist())),
        "metrics": metrics,
        "versions": _versions(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json), encoding="utf-8")
    logger.info("Отчёт: %s", report_path)
    print_comparison(metrics, prev_report["metrics"] if prev_report else None,
                     prev_report.get("date") if prev_report else None)
    return metrics


def print_comparison(new: dict, old: dict | None, old_date: str | None = None) -> None:
    """Метрики новой модели рядом с прошлыми: «стало (было, Δ)»."""
    keys = [("roc_auc", "AUC"), ("balanced_accuracy", "BAcc"), ("sensitivity", "Sens"), ("specificity", "Spec")]
    print(f"\nМетрики (OOF): стало (было, Δ). Прошлый отчёт: {old_date or 'нет'}")
    for name, m in new.items():
        cells = []
        for k, lab in keys:
            o = old.get(name, {}).get(k) if old else None
            cells.append(f"{lab} {m[k]:.3f}" + (f" ({o:.3f}, {m[k] - o:+.3f})" if o is not None else ""))
        print(f"  {name:20s} " + "   ".join(cells))


def _feature_stats(X: pd.DataFrame, y: np.ndarray) -> dict:
    return {c: {"healthy_mean": float(X.loc[y == 0, c].mean()), "healthy_std": float(X.loc[y == 0, c].std()),
                "pd_mean": float(X.loc[y == 1, c].mean()), "pd_std": float(X.loc[y == 1, c].std())}
            for c in X.columns}


def _versions() -> dict:
    import importlib
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "mediapipe", "scipy"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    return str(o)


# =========================================================================== #
#  3. compute_metrics — расчёт метрик
# =========================================================================== #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Метрики бинарной классификации; класс PD (1), если вероятность > threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    nan = float("nan")
    sens = tp / (tp + fn) if tp + fn else nan
    spec = tn / (tn + fp) if tn + fp else nan
    two_classes = len(np.unique(y_true)) == 2
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if two_classes else nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if two_classes else nan,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(PROJECT_ROOT / CONFIG_PATH, encoding="utf-8") as f:
        service_cfg = yaml.safe_load(f)

    dataset = load_data(DATA_DIRS, PD_CSV, HEALTHY_CSV)
    train_model(dataset, service_cfg)
    print("Готово. Убедитесь, что результаты совпадают с прошлой моделью (метрики выше и\n"
          "прежний training_report.json в _backup/), затем перезапустите сервис.")
