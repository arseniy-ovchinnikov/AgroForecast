#!/usr/bin/env python3
"""Шаг 4. Временная валидация, сравнение моделей, выбор и сохранение лучшей.

Вывод:
    results/models/model_comparison.csv
    results/models/predictions_all.csv
    results/models/metrics_by_year.csv
    results/models/metrics_by_region.csv
    results/models/feature_importance.csv
    results/models/khakassia_by_year.csv
    results/models/khakassia_metrics.csv
    models/final_model.cbm  (или .joblib)
    models/model_metadata.json

Запуск:
    python scripts/04_train_validate.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.leakage import assert_no_forbidden_features, feature_set
from src.models.experiment import (
    focus_region_report,
    per_region_metrics,
    per_year_metrics,
    run_experiments,
    select_best,
)
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging
from src.validation.temporal import expanding_window_splits

ALGORITHMS = ["catboost", "random_forest", "hist_gradient_boosting"]


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("04_train_validate")

    target = cfg["project"]["target"]
    out_dir = cfg.path("models_results")
    models_dir = cfg.path("models_dir")

    dataset_path = cfg.path("features_dir") / "training_dataset.csv"
    if not dataset_path.exists():
        logger.error("Не найден %s — сначала выполните scripts/03_build_dataset.py", dataset_path)
        return 2
    dataset = pd.read_csv(dataset_path).sort_values(["region", "year"]).reset_index(drop=True)

    with StageTimer(logger, "Построение временных разбиений"):
        splits = expanding_window_splits(
            dataset,
            test_years=cfg["validation"]["test_years"],
            train_start_year=cfg["validation"]["train_start_year"],
            min_train_years=cfg["validation"]["min_train_years"],
        )

    with StageTimer(logger, "Сравнение моделей на временной валидации"):
        predictions, comparison = run_experiments(
            dataset=dataset,
            splits=splits,
            target=target,
            feature_sets=cfg["models"]["feature_sets"],
            algorithms=ALGORITHMS,
            model_params=cfg["models"],
        )
    predictions.to_csv(out_dir / "predictions_all.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    best = select_best(comparison, criterion="rmse")

    with StageTimer(logger, "Разрезы метрик лучшей модели"):
        per_year_metrics(predictions, best["model"], best["feature_set"]).to_csv(
            out_dir / "metrics_by_year.csv", index=False, encoding="utf-8-sig"
        )
        per_region_metrics(predictions, best["model"], best["feature_set"]).to_csv(
            out_dir / "metrics_by_region.csv", index=False, encoding="utf-8-sig"
        )

        focus = cfg["validation"]["focus_region"]
        khak_table, khak_metrics = focus_region_report(
            predictions, focus, best["model"], best["feature_set"]
        )
        khak_table.to_csv(out_dir / "khakassia_by_year.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([khak_metrics]).to_csv(
            out_dir / "khakassia_metrics.csv", index=False, encoding="utf-8-sig"
        )

    # --- Финальная модель: обучение на всех доступных годах -------------------
    with StageTimer(logger, "Обучение финальной модели на полном ряде"):
        features = feature_set(best["feature_set"], list(dataset.columns))
        assert_no_forbidden_features(features)
        final_train = dataset[dataset["year"] >= cfg["validation"]["train_start_year"]]
        model = build_model(best["model"], cfg["models"], features, use_region=True)
        model.fit(final_train, target)

        importance = model.feature_importance()
        importance.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        logger.info("Топ-15 признаков:\n%s", importance.head(15).to_string(index=False))

        suffix = ".cbm" if best["model"] == "catboost" else ".joblib"
        model_path = models_dir / f"final_model{suffix}"
        model.save(model_path)

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": target,
            "target_unit": cfg["project"]["target_unit"],
            "crop": cfg["project"]["crop"],
            "farm_category": cfg["project"]["farm_category"],
            "algorithm": best["model"],
            "feature_set": best["feature_set"],
            "features": features,
            "categorical_features": ["region"],
            "model_params": dict(cfg["models"][best["model"]]),
            "train_years": [int(final_train["year"].min()), int(final_train["year"].max())],
            "train_rows": int(len(final_train)),
            "n_regions": int(final_train["region"].nunique()),
            "season_months": list(cfg["features"]["season_months"]),
            "validation": {
                "scheme": "expanding window (rolling origin), без перемешивания",
                "test_years": [int(s.test_year) for s in splits],
                "metrics": comparison[
                    (comparison["model"] == best["model"])
                    & (comparison["feature_set"] == best["feature_set"])
                ].to_dict(orient="records"),
                "focus_region": cfg["validation"]["focus_region"],
                "focus_region_metrics": khak_metrics,
            },
            "model_file": model_path.name,
            "dataset_file": str(dataset_path.relative_to(cfg.root)),
        }
        with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
        logger.info("Метаданные сохранены: %s", models_dir / "model_metadata.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
