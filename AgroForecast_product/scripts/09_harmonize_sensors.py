#!/usr/bin/env python3
"""Шаг 9. Межсенсорная гармонизация MODIS ↔ VIIRS (гипотеза H4).

Вход (оба обязательны):
    results/raw_processed/modis_region_composite.csv   ← 08_extract_ndvi.py --sensor modis
    results/raw_processed/viirs_region_composite.csv   ← 08_extract_ndvi.py --sensor viirs

Что делает:
    1. согласует сетки композитов (16 дней MODIS = два периода VIIRS по 8);
    2. сравнивает три спецификации переходной функции по вневыборочной ошибке
       на удерживаемых годах перекрытия;
    3. оценивает выбранную спецификацию на всех годах перекрытия;
    4. строит гармонизированную панель VIIRS в «MODIS-эквиваленте».

Вывод:
    results/models/harmonization/calibration_comparison.csv
    results/models/harmonization/calibration_by_region.csv
    results/models/harmonization/overlap_summary.csv
    results/raw_processed/viirs_harmonized_composite.csv

Запуск:
    python scripts/09_harmonize_sensors.py
    python scripts/09_harmonize_sensors.py --holdout 3 --column ndvi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.harmonization import (
    align_sensors,
    compare_modes,
    evaluate_calibration,
    fit_calibration,
)
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging

MIN_OVERLAP_SEASONS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Межсенсорная гармонизация NDVI (H4)")
    p.add_argument("--column", default="ndvi", choices=["ndvi", "evi"])
    p.add_argument("--holdout", type=int, default=3,
                   help="сколько последних лет перекрытия удержать для проверки")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("09_harmonize")

    raw = cfg.path("raw_processed")
    out_dir = cfg.path("models_results") / "harmonization"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "modis": raw / "modis_region_composite.csv",
        "viirs": raw / "viirs_region_composite.csv",
    }
    for sensor, path in paths.items():
        if not path.exists():
            logger.error(
                "Не найден %s. Выполните: python scripts/08_extract_ndvi.py --sensor %s",
                path, sensor,
            )
            return 2

    modis = pd.read_csv(paths["modis"])
    viirs = pd.read_csv(paths["viirs"])

    with StageTimer(logger, "Согласование сеток композитов"):
        aligned = align_sensors(modis, viirs, value_columns=("ndvi", "evi"))

    overlap_years = sorted(aligned["year"].unique())
    logger.info("Годы перекрытия: %s (всего %d сезонов)", overlap_years, len(overlap_years))

    summary = pd.DataFrame([{
        "сезонов_перекрытия": len(overlap_years),
        "первый_год": overlap_years[0],
        "последний_год": overlap_years[-1],
        "парных_наблюдений": len(aligned),
        "субъектов": aligned["region"].nunique(),
        "H4_проверяема": len(overlap_years) >= MIN_OVERLAP_SEASONS,
    }])
    summary.to_csv(out_dir / "overlap_summary.csv", index=False, encoding="utf-8-sig")

    if len(overlap_years) < MIN_OVERLAP_SEASONS:
        logger.error(
            "Сезонов перекрытия %d < %d — калибровать не на чем. Гипотезу H4 "
            "следует перенести в раздел «направления развития».",
            len(overlap_years), MIN_OVERLAP_SEASONS,
        )
        return 3

    with StageTimer(logger, "Сравнение спецификаций переходной функции"):
        comparison = compare_modes(aligned, args.column, args.holdout)
        comparison.to_csv(
            out_dir / "calibration_comparison.csv", index=False, encoding="utf-8-sig"
        )
        best_mode = str(comparison.iloc[0]["спецификация"])
        logger.info("Выбрана спецификация «%s» — по вневыборочной ошибке", best_mode)

    with StageTimer(logger, f"Финальная калибровка ({best_mode})"):
        calibration = fit_calibration(aligned, args.column, best_mode)
        evaluation = evaluate_calibration(
            aligned, calibration, overlap_years[-args.holdout:]
        )
        pd.DataFrame([{
            "спецификация": best_mode, "показатель": args.column,
            "a": calibration.intercept, "b": calibration.slope,
            "R2": calibration.r2, "RMSE": calibration.rmse,
            "n": calibration.n_obs, **evaluation,
        }]).to_csv(out_dir / "calibration_final.csv", index=False, encoding="utf-8-sig")

        if calibration.per_region:
            pd.DataFrame([
                {"region": r, "a": a, "b": b}
                for r, (a, b) in sorted(calibration.per_region.items())
            ]).to_csv(
                out_dir / "calibration_by_region.csv", index=False, encoding="utf-8-sig"
            )

    with StageTimer(logger, "Построение гармонизированной панели VIIRS"):
        harmonised = viirs.copy()
        harmonised[args.column] = calibration.apply(
            harmonised[args.column].to_numpy(dtype=float),
            harmonised["region"].to_numpy(),
        )
        harmonised["sensor"] = "viirs_harmonised"
        harmonised.to_csv(
            raw / "viirs_harmonized_composite.csv", index=False, encoding="utf-8-sig"
        )

    logger.info(
        "Готово. H4 проверяема на %d сезонах перекрытия. Дальше — сравнить "
        "точность модели на сыром и гармонизированном VIIRS.",
        len(overlap_years),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
