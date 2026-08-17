"""Загрузка конфигурации и разрешение путей проекта.

Единственная точка, знающая о расположении файлов. Ни один другой модуль
не содержит захардкоженных путей.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


def project_root() -> Path:
    """Корень проекта.

    Приоритет: переменная окружения AGROFORECAST_ROOT, иначе — родитель
    каталога, содержащего этот файл (src/utils/config.py -> ../../).
    """
    env = os.environ.get("AGROFORECAST_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Обёртка над config.yaml с разрешением путей."""

    raw: Dict[str, Any]
    root: Path

    # ---------------------------------------------------------------- доступ
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Абсолютный путь по ключу из секции paths."""
        value = self.raw["paths"][key]
        if isinstance(value, list):
            raise TypeError(
                f"paths.{key} — список; используйте Config.first_existing_path()"
            )
        return self.resolve(value)

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def first_existing_path(self, key: str) -> Path:
        """Первый существующий путь из списка paths.<key>.

        Raises:
            FileNotFoundError: если ни один вариант не найден (перечисляет все).
        """
        candidates: List[str] = self.raw["paths"][key]
        if isinstance(candidates, str):
            candidates = [candidates]
        resolved = [self.resolve(c) for c in candidates]
        for p in resolved:
            if p.exists():
                return p
        listing = "\n  ".join(str(p) for p in resolved)
        raise FileNotFoundError(
            f"Ни один из файлов paths.{key} не найден:\n  {listing}"
        )

    def ensure_dirs(self) -> None:
        """Создаёт все выходные каталоги, объявленные в конфиге."""
        for key in (
            "results",
            "raw_processed",
            "features_dir",
            "models_results",
            "predictions",
            "reports",
            "models_dir",
            "logs",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Читает configs/config.yaml (или указанный файл)."""
    root = project_root()
    cfg_path = Path(path) if path else root / "configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)
