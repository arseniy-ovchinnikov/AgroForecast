"""Сравнение моделей на временной валидации.

Прогоняет матрицу «набор признаков × алгоритм» по расширяющемуся окну,
собирает прогнозы и метрики, включая отдельный разрез по фокусному региону.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features.leakage import assert_no_forbidden_features, feature_set
from ..utils.logging_utils import get_logger
from ..validation.temporal import (
    Split,
    baseline_previous_year,
    baseline_regional_mean,
    metrics_by_group,
    regression_metrics,
)
from .registry import build_model

logger = get_logger(__name__)


def run_experiments(
    dataset: pd.DataFrame,
    splits: Sequence[Split],
    target: str,
    feature_sets: Sequence[str],
    algorithms: Sequence[str],
    model_params: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Обучает все комбинации и собирает прогнозы.

    Returns:
        (predictions, comparison)
        predictions: [model, feature_set, region, year, y_true, y_pred, error, abs_error]
        comparison:  агрегированные метрики по каждой комбинации.
    """
    columns = list(dataset.columns)
    all_predictions: List[pd.DataFrame] = []

    # --- Эталонные (небучаемые) модели --------------------------------------
    for split in splits:
        test = dataset.iloc[split.test_index]
        for base_name, values in (
            ("baseline_regional_mean", baseline_regional_mean(dataset, split, target)),
            ("baseline_previous_year", baseline_previous_year(dataset, split)),
        ):
            all_predictions.append(
                pd.DataFrame(
                    {
                        "model": base_name,
                        "feature_set": "baseline",
                        "region": test["region"].to_numpy(),
                        "year": test["year"].to_numpy(),
                        "y_true": test[target].to_numpy(dtype=float),
                        "y_pred": values,
                    }
                )
            )

    # --- Обучаемые модели ----------------------------------------------------
    for fs_name in feature_sets:
        features = feature_set(fs_name, columns)
        assert_no_forbidden_features(features)
        logger.info("Набор признаков «%s»: %d признаков", fs_name, len(features))

        for algo in algorithms:
            for split in splits:
                train = dataset.iloc[split.train_index]
                test = dataset.iloc[split.test_index]
                model = build_model(algo, model_params, features, use_region=True)
                model.fit(train, target)
                y_pred = model.predict(test)
                all_predictions.append(
                    pd.DataFrame(
                        {
                            "model": algo,
                            "feature_set": fs_name,
                            "region": test["region"].to_numpy(),
                            "year": test["year"].to_numpy(),
                            "y_true": test[target].to_numpy(dtype=float),
                            "y_pred": y_pred,
                        }
                    )
                )
            logger.info("  обучено: %s × %s (%d разбиений)", algo, fs_name, len(splits))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["error"].abs()

    rows: List[Dict[str, object]] = []
    for (model_name, fs_name), sub in predictions.groupby(["model", "feature_set"]):
        m = regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        m.update({"model": model_name, "feature_set": fs_name})
        rows.append(m)
    comparison = (
        pd.DataFrame(rows)[["model", "feature_set", "n", "mae", "rmse", "r2", "bias", "mape"]]
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    logger.info("Сравнение моделей:\n%s", comparison.to_string(index=False))
    return predictions, comparison


def select_best(comparison: pd.DataFrame, criterion: str = "rmse") -> Dict[str, str]:
    """Выбирает лучшую обучаемую комбинацию (эталоны исключаются)."""
    trainable = comparison[~comparison["model"].str.startswith("baseline_")]
    if trainable.empty:
        raise ValueError("Нет обучаемых моделей для выбора")
    best = trainable.sort_values(criterion).iloc[0]
    logger.info(
        "Лучшая комбинация по %s: %s / %s (RMSE=%.3f, MAE=%.3f, R²=%.3f)",
        criterion,
        best["model"],
        best["feature_set"],
        best["rmse"],
        best["mae"],
        best["r2"],
    )
    return {"model": str(best["model"]), "feature_set": str(best["feature_set"])}


def focus_region_report(
    predictions: pd.DataFrame,
    region: str,
    model: str,
    feature_set_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Погодовой разрез и метрики по одному региону (напр., Республика Хакасия)."""
    sub = predictions[
        (predictions["region"] == region)
        & (predictions["model"] == model)
        & (predictions["feature_set"] == feature_set_name)
    ].sort_values("year")

    if sub.empty:
        logger.warning("Нет прогнозов для региона «%s» (%s/%s)", region, model, feature_set_name)
        return sub, regression_metrics(np.array([]), np.array([]))

    table = sub[["year", "y_true", "y_pred", "error", "abs_error"]].rename(
        columns={
            "year": "год",
            "y_true": "факт_ц_га",
            "y_pred": "прогноз_ц_га",
            "error": "ошибка_ц_га",
            "abs_error": "абс_ошибка_ц_га",
        }
    )
    table["отн_ошибка_%"] = (table["ошибка_ц_га"] / table["факт_ц_га"] * 100).round(2)
    metrics = regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
    logger.info(
        "%s (%s/%s): MAE=%.2f RMSE=%.2f R²=%.3f Bias=%+.2f",
        region,
        model,
        feature_set_name,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        metrics["bias"],
    )
    return table.reset_index(drop=True), metrics


def per_year_metrics(
    predictions: pd.DataFrame,
    model: str,
    feature_set_name: str,
) -> pd.DataFrame:
    sub = predictions[
        (predictions["model"] == model) & (predictions["feature_set"] == feature_set_name)
    ]
    return metrics_by_group(sub, "year")


def per_region_metrics(
    predictions: pd.DataFrame,
    model: str,
    feature_set_name: str,
) -> pd.DataFrame:
    sub = predictions[
        (predictions["model"] == model) & (predictions["feature_set"] == feature_set_name)
    ]
    return metrics_by_group(sub, "region").sort_values("mae", ascending=False)
