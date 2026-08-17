#!/usr/bin/env python3
"""Шаг 1. ERA5-Land → помесячные средние по субъектам РФ.

Читает NetCDF (сотни МБ), агрегирует по полигонам ru.json и сохраняет
компактный CSV, с которым работают все последующие шаги.

Вывод:
    results/raw_processed/era5_region_month.csv
    results/raw_processed/era5_structure.json      — фактическая структура файла
    results/raw_processed/era5_cells_per_region.csv — контроль покрытия сеткой

Запуск:
    python scripts/01_extract_era5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.era5 import extract_region_monthly, inspect_era5, validate_structure
from src.data.regions import load_boundaries, verify_boundary_disjointness
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("01_extract_era5")

    out_dir = cfg.path("raw_processed")

    with StageTimer(logger, "Загрузка границ субъектов"):
        boundaries = load_boundaries(cfg.path("boundaries_geojson"))
        checks = verify_boundary_disjointness(boundaries, cfg.path("boundaries_geojson"))
        checks.to_csv(out_dir / "boundary_checks.csv", index=False, encoding="utf-8-sig")
        if (checks["status"] != "ok").any():
            logger.error(
                "Полигоны «область без округа» пересекаются с округами:\n%s",
                checks.to_string(index=False),
            )
            return 2
        logger.info("Проверка непересечения полигонов пройдена (%d пар)", len(checks))

    nc_path = cfg.first_existing_path("era5_files")

    with StageTimer(logger, f"Проверка структуры {nc_path.name}"):
        structure = inspect_era5(nc_path)
        issues = validate_structure(structure, cfg["era5"]["expected"])
        with open(out_dir / "era5_structure.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "file": str(nc_path),
                    "variables": structure.variables,
                    "raw_units": structure.raw_units,
                    "n_time": structure.n_time,
                    "time_min": str(structure.time_min),
                    "time_max": str(structure.time_max),
                    "n_lat": structure.n_lat,
                    "n_lon": structure.n_lon,
                    "lat_range": [structure.lat_min, structure.lat_max],
                    "lon_range": [structure.lon_min, structure.lon_max],
                    "lat_step": structure.lat_step,
                    "lon_step": structure.lon_step,
                    "spec_issues": issues,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    with StageTimer(logger, "Агрегирование ERA5 по субъектам"):
        monthly = extract_region_monthly(nc_path, boundaries, cfg["era5"]["variables"])

    cell_stats = monthly.attrs.get("cell_stats")
    if cell_stats is not None:
        cell_stats.to_csv(
            out_dir / "era5_cells_per_region.csv", index=False, encoding="utf-8-sig"
        )
        thin = cell_stats[cell_stats["n_cells"] < int(cfg["era5"]["min_cells_per_region"])]
        for _, row in thin.iterrows():
            logger.warning(
                "Регион покрыт менее чем %s ячейками ERA5: %s (%d)",
                cfg["era5"]["min_cells_per_region"],
                row["region"],
                row["n_cells"],
            )

    out_csv = out_dir / "era5_region_month.csv"
    monthly.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("Сохранено: %s (%d строк)", out_csv, len(monthly))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
