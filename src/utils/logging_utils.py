"""Единая настройка логирования.

Логи одновременно идут в stdout и в файл logs/agroforecast.log.
Каждый скрипт вызывает setup_logging() один раз в начале работы.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"


def setup_logging(log_file: Path | None = None, level: str = "INFO") -> None:
    """Настраивает корневой логгер (идемпотентно)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Сторонние библиотеки не должны заглушать наш вывод.
    for noisy in ("matplotlib", "PIL", "numexpr", "fiona", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class StageTimer:
    """Контекстный менеджер для замера и логирования этапа."""

    def __init__(self, logger: logging.Logger, stage: str) -> None:
        self.logger = logger
        self.stage = stage
        self._t0 = 0.0

    def __enter__(self) -> "StageTimer":
        import time

        self._t0 = time.perf_counter()
        self.logger.info("[НАЧАЛО] %s", self.stage)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import time

        dt = time.perf_counter() - self._t0
        if exc_type is None:
            self.logger.info("[ГОТОВО] %s (%.1f c)", self.stage, dt)
        else:
            self.logger.error("[ОШИБКА] %s (%.1f c): %s: %s", self.stage, dt, exc_type.__name__, exc)
        return False  # исключения не подавляются
