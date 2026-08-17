#!/usr/bin/env python3
"""Шаг 2. Извлечение годовых панелей Росстата по субъектам РФ.

Источники:
    * «Регионы России», раздел 13 — основной, 2000–2024;
    * бюллетени ГМЦ Росстата 2025 — расширение на 2025 год (если файлы есть);
    * витрина ЕМИСС — только перекрёстная проверка урожайности.

Вывод:
    results/raw_processed/rosstat_panel.csv
    results/raw_processed/rosstat_yield_crosscheck.csv
    results/raw_processed/rosstat_coverage.csv

Запуск:
    python scripts/02_extract_rosstat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rosstat import (
    cross_check_yield,
    load_regions_of_russia,
    parse_emiss_yield,
    parse_gmc_regional_sheet,
)
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("02_extract_rosstat")

    out_dir = cfg.path("raw_processed")
    missing_tokens = set(cfg["rosstat"]["missing_tokens"])

    with StageTimer(logger, "«Регионы России», раздел 13"):
        panel = load_regions_of_russia(
            cfg.path("regions_of_russia_xlsx"),
            cfg["rosstat"]["sheets"],
            missing_tokens,
        )

    # --- Расширение на 2025 год из оперативных бюллетеней ГМЦ ----------------
    with StageTimer(logger, "Бюллетени ГМЦ Росстата за 2025 год (опционально)"):
        extra_frames = []
        gmc_specs = [
            ("gmc_val_2025_xlsx", "120(1000001)", "yield_c_ha", 2025),
            ("gmc_posev_2025_xlsx", "101(1000001)", "sown_area_grain_kha", 2025),
        ]
        for path_key, sheet, column, year in gmc_specs:
            try:
                path = cfg.path(path_key)
            except KeyError:
                logger.info("В конфиге нет пути %s — пропуск", path_key)
                continue
            if not path.exists():
                logger.info("Файл не найден, пропуск: %s", path)
                continue
            try:
                frame = parse_gmc_regional_sheet(path, sheet, column, year, 5, missing_tokens)
                extra_frames.append(frame)
            except Exception as exc:  # noqa: BLE001 — причина логируется, шаг не критичен
                logger.warning("Не удалось разобрать %s:%s — %s: %s",
                               path.name, sheet, type(exc).__name__, exc)

        if extra_frames:
            extra = extra_frames[0]
            for frame in extra_frames[1:]:
                extra = extra.merge(frame, on=["region", "year"], how="outer")
            before = len(panel)
            panel = pd.concat([panel, extra], ignore_index=True)
            panel = (
                panel.groupby(["region", "year"], as_index=False)
                .first()
                .sort_values(["region", "year"])
                .reset_index(drop=True)
            )
            logger.info(
                "Панель расширена данными 2025 г.: %d → %d строк, годы %d–%d",
                before, len(panel), panel["year"].min(), panel["year"].max(),
            )

    # --- Перекрёстная проверка урожайности -----------------------------------
    with StageTimer(logger, "Перекрёстная проверка урожайности с витриной ЕМИСС"):
        try:
            emiss = parse_emiss_yield(
                cfg.path("emiss_yield_xls"),
                category=cfg["project"]["farm_category"],
                crop=cfg["project"]["crop"],
                missing_tokens=missing_tokens,
            )
            check = cross_check_yield(panel, emiss)
            check.to_csv(
                out_dir / "rosstat_yield_crosscheck.csv", index=False, encoding="utf-8-sig"
            )
            n_bad = int((check["flag"] == "MISMATCH").sum())
            if n_bad:
                worst = check.sort_values("rel_diff", ascending=False).head(10)
                logger.warning(
                    "Расхождения между 13.11 и ЕМИСС (>5%%): %d наблюдений. Худшие:\n%s",
                    n_bad,
                    worst[["region", "year", "yield_c_ha", "yield_c_ha_emiss", "rel_diff"]]
                    .to_string(index=False),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Перекрёстная проверка пропущена — %s: %s", type(exc).__name__, exc
            )

    # --- Покрытие -------------------------------------------------------------
    coverage = (
        panel.groupby("region")
        .agg(
            years=("year", "nunique"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            yield_obs=("yield_c_ha", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .sort_values("yield_obs")
    )
    coverage.to_csv(out_dir / "rosstat_coverage.csv", index=False, encoding="utf-8-sig")

    out_csv = out_dir / "rosstat_panel.csv"
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(
        "Сохранено: %s — %d строк, %d регионов, %d–%d",
        out_csv, len(panel), panel["region"].nunique(),
        panel["year"].min(), panel["year"].max(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
