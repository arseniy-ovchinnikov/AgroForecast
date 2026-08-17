#!/usr/bin/env python3
"""Шаг 5. Прогноз урожайности по сохранённой модели.

Прогноз возможен только для тех (регион, год), для которых В ДАТАСЕТЕ есть
полный набор признаков модели — в первую очередь полный вегетационный сезон
ERA5. Если данных не хватает, скрипт сообщает об этом и НЕ достраивает
недостающие значения.

Примеры:
    # Проверочный прогноз на 2024 год (обучение строго до 2024)
    python scripts/05_predict.py --year 2024

    # Прогноз на 2025 год по всем доступным регионам
    python scripts/05_predict.py --year 2025

    # Один регион
    python scripts/05_predict.py --year 2024 --region "Республика Хакасия"

Вывод:
    results/predictions/prediction_<год>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Прогноз урожайности зерновых (ц/га)")
    p.add_argument("--year", type=int, required=True, help="прогнозируемый год")
    p.add_argument("--region", type=str, default=None, help="один субъект РФ (по умолчанию — все)")
    p.add_argument(
        "--retrain-until",
        type=int,
        default=None,
        help="переобучить модель на годах < указанного (по умолчанию — < --year)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("05_predict")

    target = cfg["project"]["target"]
    meta_path = cfg.path("models_dir") / "model_metadata.json"
    if not meta_path.exists():
        logger.error("Не найдены метаданные модели (%s) — выполните scripts/04_train_validate.py",
                     meta_path)
        return 2
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    dataset = pd.read_csv(cfg.path("features_dir") / "training_dataset.csv")
    features = meta["features"]

    target_rows = dataset[dataset["year"] == args.year].copy()
    if args.region:
        target_rows = target_rows[target_rows["region"] == args.region]

    if target_rows.empty:
        logger.error(
            "В датасете нет строк для года %d%s. Возможные причины: "
            "нет полного вегетационного сезона ERA5 или нет статистики за нужные лаги. "
            "Доступные годы: %d–%d",
            args.year,
            f" и региона «{args.region}»" if args.region else "",
            int(dataset["year"].min()),
            int(dataset["year"].max()),
        )
        return 3

    # Проверка полноты признаков — без молчаливого заполнения
    missing_report = (
        target_rows[features].isna().mean().sort_values(ascending=False).head(10)
    )
    fully_missing = [f for f in features if target_rows[f].isna().all()]
    if fully_missing:
        logger.error(
            "Для года %d полностью отсутствуют признаки: %s. Прогноз не выполняется.",
            args.year, fully_missing,
        )
        return 4
    logger.info("Доля пропусков в признаках (топ-10):\n%s", missing_report.to_string())

    cutoff = args.retrain_until if args.retrain_until is not None else args.year
    train = dataset[
        (dataset["year"] >= cfg["validation"]["train_start_year"])
        & (dataset["year"] < cutoff)
        & (dataset[target].notna())
    ]
    if train["year"].nunique() < cfg["validation"]["min_train_years"]:
        logger.error(
            "Недостаточно лет для обучения (%d < %d)",
            train["year"].nunique(), cfg["validation"]["min_train_years"],
        )
        return 5

    with StageTimer(logger, f"Обучение {meta['algorithm']} на {int(train['year'].min())}–{cutoff - 1}"):
        model = build_model(meta["algorithm"], cfg["models"], features, use_region=True)
        model.fit(train, target)

    preds = model.predict(target_rows)
    out = pd.DataFrame(
        {
            "region": target_rows["region"].to_numpy(),
            "year": args.year,
            "predicted_yield_c_ha": preds.round(3),
        }
    )
    if target in target_rows.columns and target_rows[target].notna().any():
        out["actual_yield_c_ha"] = target_rows[target].to_numpy()
        out["error_c_ha"] = (out["predicted_yield_c_ha"] - out["actual_yield_c_ha"]).round(3)
        out["abs_error_c_ha"] = out["error_c_ha"].abs()

    out = out.sort_values("region").reset_index(drop=True)
    out_path = cfg.path("predictions") / f"prediction_{args.year}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Прогноз сохранён: %s (%d регионов)", out_path, len(out))

    if "abs_error_c_ha" in out.columns:
        logger.info(
            "Сверка с фактом за %d: MAE=%.2f ц/га, Bias=%+.2f ц/га",
            args.year, out["abs_error_c_ha"].mean(), out["error_c_ha"].mean(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
