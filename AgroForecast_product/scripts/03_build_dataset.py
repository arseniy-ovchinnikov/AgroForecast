#!/usr/bin/env python3
"""Шаг 3. Сопоставление регионов, признаки, финальный датасет и QC.

Вход:
    results/raw_processed/era5_region_month.csv   (шаг 1)
    results/raw_processed/rosstat_panel.csv       (шаг 2)

Вывод:
    results/raw_processed/region_mapping.csv
    results/features/training_dataset.csv         — финальный датасет
    results/features/feature_leakage_table.csv    — реестр признаков и утечек
    results/features/qc_report.csv
    results/features/qc_filter_journal.csv
    results/features/feature_dictionary.csv

Запуск:
    python scripts/03_build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.regions import build_region_mapping, load_boundaries
from src.features.agro import add_fertilizer_features, add_lag_features, add_structural_features
from src.features.climate import build_climate_features
from src.features.leakage import FORBIDDEN_COLUMNS, feature_set, leakage_table
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging
from src.validation.qc import apply_hard_filters, run_qc


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("03_build_dataset")

    raw_dir = cfg.path("raw_processed")
    feat_dir = cfg.path("features_dir")
    target = cfg["project"]["target"]

    era5_csv = raw_dir / "era5_region_month.csv"
    rosstat_csv = raw_dir / "rosstat_panel.csv"
    if not rosstat_csv.exists():
        logger.error("Не найден %s — сначала выполните scripts/02_extract_rosstat.py", rosstat_csv)
        return 2
    if not era5_csv.exists():
        logger.error("Не найден %s — сначала выполните scripts/01_extract_era5.py", era5_csv)
        return 2

    panel = pd.read_csv(rosstat_csv)
    monthly = pd.read_csv(era5_csv)

    # --- Сопоставление регионов ----------------------------------------------
    with StageTimer(logger, "Сопоставление регионов Росстат ↔ ERA5"):
        boundaries = load_boundaries(cfg.path("boundaries_geojson"))
        mapping = build_region_mapping(panel["region"].unique(), boundaries)
        mapping.to_csv(raw_dir / "region_mapping.csv", index=False, encoding="utf-8-sig")

        unmatched = mapping[mapping["status"] == "no_boundary"]["rosstat_region"].tolist()
        if unmatched:
            logger.warning(
                "Регионы без климатических данных (исключаются из обучения): %s", unmatched
            )
        matched = set(mapping.loc[mapping["status"] == "matched", "rosstat_region"])
        panel = panel[panel["region"].isin(matched)].copy()

    # --- Климатические признаки ----------------------------------------------
    with StageTimer(logger, "Климатические признаки"):
        climate = build_climate_features(
            monthly,
            season_months=cfg["features"]["season_months"],
            monthly_months=cfg["features"]["monthly_months"],
            warm_threshold_c=cfg["features"]["warm_month_threshold_c"],
            cold_threshold_c=cfg["features"]["cold_month_threshold_c"],
        )

    # --- Аграрные признаки ----------------------------------------------------
    with StageTimer(logger, "Аграрные признаки (лаги, структура, удобрения)"):
        panel = add_lag_features(
            panel,
            yield_column=target,
            yield_lags=cfg["features"]["yield_lags"],
            sown_area_lags=cfg["features"]["sown_area_lags"],
            rolling_windows=cfg["features"]["yield_rolling_windows"],
        )
        panel = add_structural_features(panel)
        panel = add_fertilizer_features(panel)

    # --- Объединение ----------------------------------------------------------
    with StageTimer(logger, "Сборка обучающего датасета"):
        dataset = panel.merge(climate, on=["region", "year"], how="inner")
        logger.info(
            "После объединения с климатом: %d строк (%d регионов, %d–%d)",
            len(dataset), dataset["region"].nunique(),
            dataset["year"].min(), dataset["year"].max(),
        )

        feature_columns = feature_set("climate_history_agro", list(dataset.columns))

        # Сначала — жёсткие фильтры с журналом (каждая исключённая строка учтена),
        # затем — QC уже по итоговому датасету, чтобы отчёт описывал то, что
        # реально пойдёт в обучение.
        dataset, journal = apply_hard_filters(
            dataset,
            target=target,
            expected_season_months=len(cfg["features"]["season_months"]),
        )
        journal.to_csv(feat_dir / "qc_filter_journal.csv", index=False, encoding="utf-8-sig")

        qc_report = run_qc(
            dataset,
            target=target,
            feature_columns=feature_columns,
            expected_season_months=len(cfg["features"]["season_months"]),
        )
        qc_report.to_csv(feat_dir / "qc_report.csv", index=False, encoding="utf-8-sig")

    # --- Словарь признаков и таблица утечек -----------------------------------
    leakage_table().to_csv(
        feat_dir / "feature_leakage_table.csv", index=False, encoding="utf-8-sig"
    )

    dictionary = pd.DataFrame(
        {
            "column": dataset.columns,
            "dtype": [str(dataset[c].dtype) for c in dataset.columns],
            "n_missing": [int(dataset[c].isna().sum()) for c in dataset.columns],
            "share_missing": [round(float(dataset[c].isna().mean()), 4) for c in dataset.columns],
            "role": [
                "target" if c == target
                else "identifier" if c in ("region", "year")
                else "excluded (leakage)" if c in FORBIDDEN_COLUMNS
                else "feature" if c in feature_columns
                else "service"
                for c in dataset.columns
            ],
        }
    )
    dictionary.to_csv(feat_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")

    out_csv = feat_dir / "training_dataset.csv"
    dataset.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(
        "Финальный датасет: %s — %d строк × %d колонок; регионов %d; годы %d–%d; признаков %d",
        out_csv, len(dataset), dataset.shape[1], dataset["region"].nunique(),
        dataset["year"].min(), dataset["year"].max(), len(feature_columns),
    )

    n_err = int((qc_report["status"] == "ERROR").sum())
    if n_err:
        logger.error("QC выявил %d критических проблем — см. results/features/qc_report.csv", n_err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
