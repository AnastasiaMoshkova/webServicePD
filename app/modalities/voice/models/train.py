from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
import yaml

from app.modalities.voice.processing.three_channel import (
    ThreeChannelConfig,
    generate_three_channel,
)
from app.modalities.voice.models.resnet18_light import ResNet18LightModel

logger = logging.getLogger("train.voice")

MODALITY = "voice"
CONFIG_PATH = f"app/modalities/{MODALITY}/configs/{MODALITY}.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[4]

AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".ogg", ".webm", ".flac")


def _norm_id(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _sex_code(v: Any) -> int:
    # M -> 1, всё остальное -> 0
    return 1 if (str(v) if v is not None else "") == "M" else 0


def _find_audio(flat_dir: Path, filename: Any) -> Path | None:
    if filename is None or (isinstance(filename, float) and np.isnan(filename)):
        return None
    stem = Path(str(filename)).stem
    for ext in AUDIO_EXTS:
        for cand in (flat_dir / f"{stem}{ext}", flat_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


class ResNet18LightWithAdv(ResNet18LightModel):
    # доп. головы для инвариантности (напр. пол); в сервисе игнорируются
    def __init__(self, *args, adv_targets: list[str] | None = None,
                 adv_hidden: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self._adv_heads = nn.ModuleDict()
        for attr in (adv_targets or []):
            self._adv_heads[attr] = nn.Sequential(
                nn.Linear(512, adv_hidden), nn.ReLU(inplace=True),
                nn.Linear(adv_hidden, 2),
            )

    def adversarial_logits(self, features):
        return {k: h(features) for k, h in self._adv_heads.items()}

    def adversarial_logits_detached(self, features):
        f = features.detach()
        return {k: h(f) for k, h in self._adv_heads.items()}


def _build_model(arch_cfg: dict, adv_targets: list[str]) -> ResNet18LightWithAdv:
    return ResNet18LightWithAdv(
        num_classes=1,
        dropout=arch_cfg.get("dropout", 0.3),
        unfreeze_from=arch_cfg.get("unfreeze_from", "layer4"),
        gradual_unfreeze=False,
        init_mode=arch_cfg.get("init_mode", "custom"),
        pretrained_backbone=arch_cfg.get("pretrained_backbone", True),
        adv_targets=adv_targets,
    )


class _ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        arr = self.images[i]
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


class _MetaImageDataset(Dataset):
    def __init__(self, images, labels, sex_codes):
        self.images, self.labels, self.sex = images, labels, sex_codes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        arr = self.images[i]
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        meta = {"sex": torch.tensor(int(self.sex[i]), dtype=torch.long)}
        return x, torch.tensor(self.labels[i], dtype=torch.float32), meta


def load_data(
    data_dirs=None,
    pd_csv=None,
    healthy_csv=None,
    data_cfg: dict | None = None,
) -> dict[str, Any]:
    # data_dirs/pd_csv/healthy_csv не юзаются — оставил для совместимости с шаблоном
    if data_cfg is None:
        raise ValueError("data_cfg is required")

    excel_path = PROJECT_ROOT / data_cfg["excel_path"]
    sheet = data_cfg.get("excel_sheet") or 0
    flat_dir = PROJECT_ROOT / data_cfg["flat_audio_dir"]

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    if not flat_dir.exists():
        raise FileNotFoundError(f"Flat audio dir not found: {flat_dir}")

    df = pd.read_excel(excel_path, sheet_name=sheet)
    need = {"filename", "label", "person_id", "sex"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Excel missing columns: {sorted(miss)}")

    def _flt(d, col, val):
        if val is None:
            return d
        if col not in d.columns:
            raise ValueError(f"No column {col}")
        return d[d[col].astype(str) == str(val)]

    df = _flt(df, "name_dataset", data_cfg.get("filter_name_dataset"))
    df = _flt(df, "device", data_cfg.get("filter_device"))
    df = _flt(df, "exercise", data_cfg.get("filter_exercise"))

    label_map = {"HC": 0, "PD": 1}
    df = df[df["label"].astype(str).str.upper().isin(label_map)]
    df = df.dropna(subset=["filename", "person_id"])
    df = df.drop_duplicates(subset=["filename"], keep="first")

    cfg3 = ThreeChannelConfig(
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        skip_sec=float(data_cfg.get("segment_skip_sec", 0.5)),
        duration_sec=float(data_cfg.get("segment_duration_sec", 1.5)),
        n_fft=int(data_cfg.get("n_fft", 2048)),
        hop_length=int(data_cfg.get("hop_length", 512)),
        n_mels=int(data_cfg.get("n_mels", 128)),
        n_mfcc=int(data_cfg.get("n_mfcc", 40)),
        n_chroma=int(data_cfg.get("n_chroma", 12)),
        img_size=int(data_cfg.get("img_size", 224)),
    )

    images, labels, pids, sex_codes, paths = [], [], [], [], []
    missing, bad = [], []

    for row in df.itertuples(index=False):
        audio = _find_audio(flat_dir, getattr(row, "filename"))
        if audio is None:
            missing.append(str(getattr(row, "filename")))
            continue
        try:
            img = generate_three_channel(str(audio), None, cfg3)
        except Exception as e:
            logger.warning("three_channel failed %s: %s", audio, e)
            bad.append(str(audio))
            continue
        if img.shape != (cfg3.img_size, cfg3.img_size, 3):
            bad.append(f"{audio} {img.shape}")
            continue

        images.append(img)
        labels.append(label_map[str(getattr(row, "label")).upper()])
        pids.append(_norm_id(getattr(row, "person_id")))
        sex_codes.append(_sex_code(getattr(row, "sex")))
        paths.append(str(audio))

    if missing:
        logger.warning("audio not found: %d, first: %s", len(missing), missing[:20])
    if bad:
        logger.warning("bad files: %d, first: %s", len(bad), bad[:20])
    if not images:
        raise RuntimeError("No valid audio")

    X = np.stack(images, axis=0).astype(np.uint8)
    y = np.array(labels, dtype=np.int64)
    sex = np.array(sex_codes, dtype=np.int64)

    n_pd = int(y.sum())
    n_hc = int(len(y) - n_pd)
    logger.info(
        "data: %d rec, %d ppl | PD %d rec/%d ppl, HC %d rec/%d ppl | M=%d F=%d",
        len(y), len(set(pids)),
        n_pd, len({p for p, l in zip(pids, y) if l == 1}),
        n_hc, len({p for p, l in zip(pids, y) if l == 0}),
        int(sex.sum()), int((sex == 0).sum()),
    )
    return {"images": X, "labels": y, "patient_ids": pids,
            "sex_codes": sex, "paths": paths}


def _train_one_fold(X_tr, y_tr, sex_tr, X_va, y_va,
                    arch_cfg, train_cfg, device):
    inv_cfg = train_cfg.get("invariance") or {}
    adv_targets = list(inv_cfg.get("targets") or []) if inv_cfg.get("enabled") else []
    inv_lambdas = {k: float(v) for k, v in (inv_cfg.get("lambdas") or {}).items()}
    inv_start = int(inv_cfg.get("start_epoch", 10))
    disc_lr = float(inv_cfg.get("disc_lr", 1e-5))

    model = _build_model(arch_cfg, adv_targets).to(device)

    train_loader = DataLoader(
        _MetaImageDataset(X_tr, y_tr, sex_tr),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        _ImageDataset(X_va, y_va),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)),
    )

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        logger.warning("fold degenerate: pos=%d neg=%d -> pw=1.0", n_pos, n_neg)
        pw = torch.tensor([1.0], device=device)
    else:
        pw = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    disc_opt = torch.optim.Adam(model._adv_heads.parameters(), lr=disc_lr) if adv_targets else None

    sch = train_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=sch.get("mode", "max"),
        factor=float(sch.get("factor", 0.5)),
        patience=int(sch.get("patience", 5)),
    )

    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    es = train_cfg["early_stopping"]
    es_pat = int(es.get("patience", 50))
    es_max = es.get("mode", "max") == "max"
    epochs = int(train_cfg["epochs"])

    best_score = -np.inf if es_max else np.inf
    best_state, bad, history = None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels, meta in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            opt.zero_grad()

            use_disc = bool(inv_lambdas) and meta is not None
            use_ent = use_disc and epoch >= inv_start

            if use_disc:
                feats = model.backbone(inputs)
                out = model.classifier(feats)

                if disc_opt is not None:
                    d_logits = model.adversarial_logits_detached(feats)
                    terms = []
                    for attr, dl in d_logits.items():
                        if inv_lambdas.get(attr, 0.0) <= 0:
                            continue
                        y_attr = meta[attr].to(device)
                        terms.append(F.cross_entropy(dl, y_attr.long()))
                    if terms:
                        d_loss = torch.stack(terms).sum()
                        disc_opt.zero_grad()
                        d_loss.backward()
                        disc_opt.step()

                adv_logits = model.adversarial_logits(feats) if use_ent else {}
            else:
                out = model(inputs)
                adv_logits = {}

            loss = criterion(out, labels)
            for attr, lg in adv_logits.items():
                lam = inv_lambdas.get(attr, 0.0)
                if lam <= 0:
                    continue
                p = F.softmax(lg, dim=1)
                ent = -(p * (p + 1e-8).log()).sum(dim=1)
                loss = loss - lam * ent.mean()

            loss.backward()
            if disc_opt is not None:
                for p in model._adv_heads.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                probs.append(torch.sigmoid(model(xb).flatten()).cpu().numpy())
                trues.append(yb.numpy())
        probs = np.concatenate(probs) if probs else np.array([])
        trues = np.concatenate(trues) if trues else np.array([])

        if len(np.unique(trues)) > 1:
            pred = (probs > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(trues, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else float("nan")
            spec = tn / (tn + fp) if (tn + fp) else float("nan")
            val_ba = float(np.nanmean([sens, spec]))
        else:
            val_ba = float("nan")

        scheduler.step(0.0 if np.isnan(val_ba) else val_ba)
        history.append({"epoch": epoch, "val_balanced_accuracy": val_ba})

        better = (not np.isnan(val_ba)) and (
            (es_max and val_ba > best_score) or ((not es_max) and val_ba < best_score)
        )
        if better:
            best_score = val_ba
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= es_pat:
                logger.info("early stop ep %d (best %.4f)", epoch, best_score)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            probs.append(torch.sigmoid(model(xb).flatten()).cpu().numpy())
    oof = np.concatenate(probs) if probs else np.array([])
    return model, oof, {"best_val_balanced_accuracy": float(best_score), "history": history}


def _backup_existing(weights_paths, report_path):
    existing = [p for p in weights_paths if p.exists()]
    if not existing and (report_path is None or not report_path.exists()):
        return None
    base = existing[0].parent if existing else report_path.parent
    bdir = base / "_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        # имя с папкой, чтоб фолды не перезаписывали друг друга
        shutil.copy2(p, bdir / f"{p.parent.name}_{p.name}")
    if report_path is not None and report_path.exists():
        shutil.copy2(report_path, bdir / report_path.name)
    logger.info("backup -> %s", bdir)
    return bdir


def train_model(dataset, service_cfg):
    train_cfg = service_cfg.get("training") or {}
    if not train_cfg:
        raise ValueError("no 'training' section in config")

    model_cfg = service_cfg["model"]
    arch_cfg = model_cfg.get("architecture", {})
    threshold = float(model_cfg.get("threshold", 0.5))

    weights_paths = [PROJECT_ROOT / p for p in model_cfg["weights_paths"]]
    n_folds = int(train_cfg.get("n_folds", len(weights_paths)))
    if n_folds != len(weights_paths):
        raise ValueError(f"n_folds={n_folds} != weights_paths={len(weights_paths)}")

    seed = int(train_cfg.get("seed", 42))
    device = torch.device(
        model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("device %s | folds %d | seed %d", device, n_folds, seed)

    X = dataset["images"]
    y = dataset["labels"]
    groups = np.array(dataset["patient_ids"])
    sex = np.asarray(dataset["sex_codes"], dtype=np.int64)

    strata = y * 2 + sex

    report_path = weights_paths[0].parent / "training_report.json"
    _backup_existing(weights_paths, report_path)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_prob = np.full(len(y), np.nan, dtype=np.float64)
    fold_info = []

    for fold, (tr, va) in enumerate(sgkf.split(X, strata, groups), start=1):
        logger.info("fold %d/%d: train=%d val=%d", fold, n_folds, len(tr), len(va))
        model, oof, info = _train_one_fold(
            X[tr], y[tr], sex[tr], X[va], y[va],
            arch_cfg, train_cfg, device,
        )
        oof_prob[va] = oof

        out = weights_paths[fold - 1]
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        fold_info.append({"fold": fold, "weights": str(out), **info})
        logger.info("saved fold %d -> %s", fold, out)

    mask = ~np.isnan(oof_prob)
    if (~mask).any():
        logger.warning("%d records without oof", int((~mask).sum()))
    metrics = compute_metrics(y[mask], oof_prob[mask], threshold)

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "modality": MODALITY,
        "n_records": int(len(y)),
        "n_persons": int(len(set(groups.tolist()))),
        "n_folds": n_folds,
        "threshold": threshold,
        "training": train_cfg,
        "architecture": arch_cfg,
        "weights_paths": [str(p) for p in weights_paths],
        "fold_info": fold_info,
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("report saved: %s", report_path)

    _print_comparison(metrics, weights_paths[0].parent)
    return metrics


def _print_comparison(new_metrics, weights_dir):
    bdirs = sorted([p for p in (weights_dir / "_backup").glob("*") if p.is_dir()]) \
        if (weights_dir / "_backup").exists() else []
    if not bdirs:
        logger.info("no previous report in _backup/")
        return
    prev_path = bdirs[-1] / "training_report.json"
    if not prev_path.exists():
        return
    try:
        with open(prev_path, encoding="utf-8") as f:
            prev = json.load(f)
        prev_m = prev.get("metrics", {})
    except Exception as e:
        logger.warning("cant read prev report: %s", e)
        return

    keys = ["balanced_accuracy", "roc_auc", "pr_auc", "f1", "sensitivity", "specificity"]
    logger.info("=== metrics: prev vs new ===")
    for k in keys:
        p = prev_m.get(k, float("nan"))
        n = new_metrics.get(k, float("nan"))
        logger.info("  %-18s prev=%.4f new=%.4f", k, p, n)


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def _d(a, b):
        return float(a) / float(b) if b else float("nan")

    sens = _d(tp, tp + fn)
    spec = _d(tn, tn + fp)
    bal = float(np.nanmean([sens, spec])) if not (np.isnan(sens) and np.isnan(spec)) else float("nan")

    try:
        roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        pr = float("nan")

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "sensitivity": sens, "specificity": spec,
        "balanced_accuracy": bal,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc, "pr_auc": pr,
        "threshold": float(threshold), "n": int(len(y_true)),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    ap = argparse.ArgumentParser(description="voice retrain")
    ap.add_argument("--config", "-c", default=CONFIG_PATH,
                    help="yaml для обучения (по умолч. voice.yaml)")
    args = ap.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    logger.info("config: %s", cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        service_cfg = yaml.safe_load(f)

    if "training" not in service_cfg:
        raise ValueError(f"{cfg_path}: no 'training' section")
    if "data" not in service_cfg:
        raise ValueError(f"{cfg_path}: no 'data' section")

    ds = load_data(data_cfg=service_cfg["data"])
    train_model(ds, service_cfg)
    print("done. проверь метрики и _backup/, потом перезапусти сервис")