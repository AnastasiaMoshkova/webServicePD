"""Реестр версий «набор признаков + скалер + классификатор» с горячей перезагрузкой.

Перезапуск не нужен. ``refresh()`` сравнивает mtime/размер YAML и файлов активной
версии (скалер, классификатор, bounds) с запомненными и перезагружает бандл,
только если что-то изменилось (обычный случай — несколько ``stat``). Поэтому:

  * сменить версию        -> поменять ``active_version`` в YAML;
  * обновить модель       -> положить новые .pkl (или заменить старые);
  * добавить версию       -> добавить блок в ``versions``.

"""
from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    version: str
    description: str
    feature_names: List[str]
    scaler: Any
    classifier: Any
    feature_bounds: Dict[str, Any]
    missing_value: float


class ModelRegistry:
    def __init__(self, config_path: Path, known_features: Optional[Iterable[str]] = None) -> None:
        """
        config_path    — путь к YAML с версиями.
        known_features — допустимые имена признаков; если задано, опечатки в
                         конфиге ловятся при загрузке, а не на первом запросе.
        Конструктор файлы не читает: первая загрузка происходит в refresh().
        """
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent
        self.known_features = set(known_features) if known_features is not None else None

        self.bundle: Optional[ModelBundle] = None  # последняя рабочая версия
        self.error: Optional[str] = None           # ошибка последней попытки обновления
        self.available_versions: List[str] = []

        self._lock = threading.Lock()
        self._watched: List[Path] = [self.config_path]
        self._sig: Optional[Tuple] = None

    def refresh(self) -> ModelBundle:
        """Актуализирует бандл (если файлы менялись) и возвращает рабочую версию.

        Бросает RuntimeError, только если рабочей версии ещё не было ни разу.
        """
        with self._lock:
            if self._sig is None or self._signature(self._watched) != self._sig:
                self._reload()
            if self.bundle is None:
                raise RuntimeError(f"Модель классификатора не загружена: {self.error}")
            return self.bundle

    def load(self, version: str) -> ModelBundle:
        """Загружает и проверяет произвольную версию (для CLI-проверки), состояние не меняет."""
        return self._build(self._read_cfg(), str(version))

    def _reload(self) -> None:
        sig: Optional[Tuple] = None
        self._watched = [self.config_path]
        try:
            cfg = self._read_cfg()
            self.available_versions = [str(v) for v in cfg["versions"]]
            version = str(cfg.get("active_version") or "")
            if not version:
                raise ValueError("не задан active_version")
            self._watched = [self.config_path, *self._paths(cfg, version)]
            sig = self._signature(self._watched)
            new_bundle = self._build(cfg, version)
        except Exception as exc:
            msg = str(exc)
            if msg != self.error:
                logger.warning(
                    "Не удалось обновить модели походки (%s). %s", msg,
                    "Работаю на предыдущей версии." if self.bundle else "Рабочей версии пока нет.",
                )
            self.error = msg
        else:
            if self.bundle is None or new_bundle.version != self.bundle.version:
                logger.info("Модели походки: активная версия признаков '%s' (%d признаков)",
                            new_bundle.version, len(new_bundle.feature_names))
            self.bundle, self.error = new_bundle, None
        finally:
            self._sig = sig if sig is not None else self._signature(self._watched)

    @staticmethod
    def _signature(paths: List[Path]) -> Tuple:
        out = []
        for p in paths:
            try:
                st = p.stat()
                out.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                out.append((str(p), None, None))
        return tuple(out)

    def _read_cfg(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg.get("versions"), dict) or not cfg["versions"]:
            raise ValueError(f"{self.config_path.name}: не задан раздел 'versions'")
        return cfg

    def _resolve(self, rel: str) -> Path:
        path = Path(str(rel))
        return path if path.is_absolute() else self.base_dir / path

    def _version_cfg(self, cfg: Dict[str, Any], version: str) -> Dict[str, Any]:
        versions = {str(k): v for k, v in cfg["versions"].items()}
        if version not in versions:
            raise ValueError(
                f"версия '{version}' не найдена в {self.config_path.name} (доступны: {', '.join(versions)})"
            )
        return versions[version] or {}

    def _paths(self, cfg: Dict[str, Any], version: str) -> List[Path]:
        """Файлы, за изменением которых следим (без проверки существования)."""
        vcfg = self._version_cfg(cfg, version)
        rels = [vcfg.get("scaler"), vcfg.get("classifier"), vcfg.get("bounds", cfg.get("bounds"))]
        return [self._resolve(r) for r in rels if r]

    def _build(self, cfg: Dict[str, Any], version: str) -> ModelBundle:
        vcfg = self._version_cfg(cfg, version)
        try:
            features = [str(x) for x in (vcfg.get("features") or [])]
            if not features:
                raise ValueError("пустой список features")
            if len(set(features)) != len(features):
                raise ValueError("в features есть дубликаты")
            if self.known_features is not None:
                unknown = [x for x in features if x not in self.known_features]
                if unknown:
                    raise ValueError(f"неизвестные признаки (опечатка?): {unknown}")
            for key in ("scaler", "classifier"):
                if not vcfg.get(key):
                    raise ValueError(f"не задан путь '{key}'")

            scaler = joblib.load(self._existing(vcfg["scaler"]))
            classifier = joblib.load(self._existing(vcfg["classifier"]))
            self._check_shape("scaler", scaler, features)
            self._check_shape("classifier", classifier, features)

            bounds_rel = vcfg.get("bounds", cfg.get("bounds"))
            bounds: Dict[str, Any] = {}
            if bounds_rel:
                with open(self._existing(bounds_rel), "r", encoding="utf-8") as f:
                    bounds = json.load(f)
        except Exception as exc:
            raise ValueError(f"версия '{version}': {exc}") from exc

        return ModelBundle(
            version=version,
            description=str(vcfg.get("description", "")),
            feature_names=features,
            scaler=scaler,
            classifier=classifier,
            feature_bounds=bounds,
            missing_value=float(cfg.get("missing_value", 0.0)),
        )

    def _existing(self, rel: str) -> Path:
        path = self._resolve(rel)
        if not path.is_file():
            raise FileNotFoundError(f"файл не найден: {path}")
        return path

    @staticmethod
    def _check_shape(role: str, model: Any, features: List[str]) -> None:
        """Ловит рассинхрон конфига и файлов моделей при загрузке."""
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(features):
            raise ValueError(f"{role} обучен на {n_in} признаках, а в конфиге их {len(features)}")
        names_in = getattr(model, "feature_names_in_", None)
        if names_in is not None and list(names_in) != features:
            raise ValueError(
                f"{role}: порядок/имена признаков при обучении {list(names_in)} "
                f"не совпадают с конфигом {features}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.modalities.gait.model_registry <model_versions.yaml>")
    reg = ModelRegistry(Path(sys.argv[1]))
    cfg = reg._read_cfg()
    active = str(cfg.get("active_version"))
    failed = False
    for v in (str(x) for x in cfg["versions"]):
        try:
            b = reg.load(v)
            mark = " (активная)" if v == active else ""
            print(f"OK   {v}{mark}: {len(b.feature_names)} признаков — {b.description}")
        except Exception as exc:
            failed = True
            print(f"FAIL {v}: {exc}")
    if active not in {str(x) for x in cfg["versions"]}:
        failed = True
        print(f"FAIL active_version '{active}' отсутствует в versions")
    sys.exit(1 if failed else 0)
