#!/usr/bin/env python3
"""Полный прогон пайплайна AgroForecast.

    python scripts/run_all.py                  # все шаги
    python scripts/run_all.py --from 3         # начиная с шага 3
    python scripts/run_all.py --skip-era5      # без пересчёта ERA5 (шаг 1)

Каждый шаг — отдельный процесс, чтобы падение одного не оставляло
пайплайн в неопределённом состоянии. Коды возврата протоколируются.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger, setup_logging  # noqa: E402

STEPS = [
    (1, "01_extract_era5.py", "ERA5 → помесячные средние по субъектам"),
    (2, "02_extract_rosstat.py", "Извлечение панелей Росстата"),
    (3, "03_build_dataset.py", "Признаки, датасет, QC"),
    (4, "04_train_validate.py", "Временная валидация и выбор модели"),
    (5, "06_report.py", "Итоговый отчёт"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Полный прогон пайплайна AgroForecast")
    p.add_argument("--from", dest="start", type=int, default=1, help="номер первого шага")
    p.add_argument("--skip-era5", action="store_true", help="пропустить шаг 1")
    p.add_argument("--predict-year", type=int, default=None,
                   help="после обучения сделать прогноз на указанный год")
    return p.parse_args()


def run(script: str, args: list[str] | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + (args or [])
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("run_all")

    for number, script, title in STEPS:
        if number < args.start:
            logger.info("Шаг %d пропущен (--from %d): %s", number, args.start, title)
            continue
        if number == 1 and args.skip_era5:
            logger.info("Шаг 1 пропущен (--skip-era5): %s", title)
            continue
        logger.info("=" * 78)
        logger.info("ШАГ %d: %s (%s)", number, title, script)
        logger.info("=" * 78)
        code = run(script)
        if code != 0:
            logger.error("Шаг %d завершился с кодом %d — пайплайн остановлен", number, code)
            return code

    if args.predict_year is not None:
        logger.info("Прогноз на %d год", args.predict_year)
        code = run("05_predict.py", ["--year", str(args.predict_year)])
        if code != 0:
            logger.error("Прогноз завершился с кодом %d", code)
            return code

    logger.info("Пайплайн выполнен полностью.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
