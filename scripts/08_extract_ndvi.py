#!/usr/bin/env python3
"""Шаг 8. Приём выгрузки NDVI из Google Earth Engine.

Порядок действий описан в docs/NEXT_STEPS.md. Кратко:
    1. загрузить data/boundaries/gee/ru_regions_shapefile.zip как ассет GEE;
    2. выполнить gee/export_modis_ndvi.js — получить CSV на Google Drive;
    3. скачать их в data/NDVI/;
    4. запустить этот скрипт.

Вывод:
    results/raw_processed/ndvi_region_composite.csv   — панель для MIDAS
    results/raw_processed/ndvi_ingest_journal.csv     — журнал фильтрации
    results/raw_processed/ndvi_coverage.csv           — полнота сезона

Запуск:
    python scripts/08_extract_ndvi.py
    python scripts/08_extract_ndvi.py --dir data/NDVI --min-pixels 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ndvi import SENSORS, coverage_report, load_gee_exports, season_period_range
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Приём выгрузки NDVI из Earth Engine")
    p.add_argument("--dir", default="data/NDVI", help="каталог с CSV-файлами GEE")
    p.add_argument("--sensor", choices=list(SENSORS), default="modis",
                   help="сенсор: modis (MOD13Q1, 16 дней) или viirs (VNP13A1, 8 дней)")
    p.add_argument("--pattern", default=None,
                   help="маска имён файлов (по умолчанию — по сенсору)")
    p.add_argument("--min-pixels", type=int, default=100,
                   help="минимум валидных пикселей пашни на регион и композит")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("08_extract_ndvi")

    directory = cfg.resolve(args.dir)
    out_dir = cfg.path("raw_processed")

    if not directory.exists():
        logger.error(
            "Каталог %s не найден. Создайте его и положите туда CSV, "
            "выгруженные скриптом gee/export_modis_ndvi.js", directory,
        )
        return 2

    with StageTimer(logger, f"Чтение выгрузки GEE из {directory}"):
        panel, journal = load_gee_exports(
            directory, args.sensor, args.pattern, args.min_pixels
        )

    months = cfg["features"]["season_months"]
    periods = season_period_range(min(months), max(months), args.sensor)
    logger.info("Сезон %s → композиты %s № %s (K = %d)",
                months, SENSORS[args.sensor]["label"], periods, len(periods))

    # Полнота считается по периодам, которые реально присутствуют в выгрузке:
    # крайние композиты сезона могут отсутствовать у конкретного продукта,
    # и требовать их — значит объявить неполными все годы разом.
    available = sorted(set(panel["period"]) & set(periods))
    missing = sorted(set(periods) - set(available))
    if missing:
        logger.warning(
            "Композиты сезона, отсутствующие в выгрузке: %s — полнота считается "
            "по фактически доступным %d периодам", missing, len(available),
        )
    coverage = coverage_report(panel, available)

    tag = args.sensor
    panel.to_csv(out_dir / f"{tag}_region_composite.csv", index=False, encoding="utf-8-sig")
    journal.to_csv(out_dir / f"{tag}_ingest_journal.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_dir / f"{tag}_coverage.csv", index=False, encoding="utf-8-sig")
    # Совместимое имя для источника ndvi в scripts/07_midas_nowcast.py
    if args.sensor == "modis":
        panel.to_csv(out_dir / "ndvi_region_composite.csv",
                     index=False, encoding="utf-8-sig")

    incomplete = coverage[~coverage["полный_сезон"]]
    if len(incomplete):
        logger.warning(
            "Пар (регион, год) с неполным сезоном: %d — они будут исключены "
            "при сборке тензора (см. results/raw_processed/ndvi_coverage.csv)",
            len(incomplete),
        )

    logger.info("Готово. Дальше: python scripts/07_midas_nowcast.py --source ndvi")
    if args.sensor == "viirs":
        logger.info("Для гипотезы H4: python scripts/09_harmonize_sensors.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
