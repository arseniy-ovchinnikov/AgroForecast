"""Временная (rolling-origin) валидация и метрики.

Случайное разбиение train_test_split здесь принципиально не используется:
данные — панель «регион × год», и любое перемешивание позволяет модели
«подсмотреть» будущее того же региона.

Схема: расширяющееся окно.
    обучение [train_start … Y-1]  →  тест Y
для каждого Y из validation.test_years.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    """Одно разбиение: обучение до test_year, тест — test_year."""

    test_year: int
    train_index: np.ndarray
    test_index: np.ndarray

    def describe(self, years: pd.Series) -> str:
        tr = years.iloc[self.train_index]
        return (
            f"тест {self.test_year}: обучение {int(tr.min())}–{int(tr.max())} "
            f"({len(self.train_index)} строк), тест {len(self.test_index)} строк"
        )


def expanding_window_splits(
    dataset: pd.DataFrame,
    test_years: Sequence[int],
    train_start_year: int,
    min_train_years: int = 8,
) -> List[Split]:
    """Строит разбиения с расширяющимся окном обучения."""
    years = dataset["year"].to_numpy()
    splits: List[Split] = []
    for test_year in sorted(test_years):
        train_mask = (years >= train_start_year) & (years < test_year)
        test_mask = years == test_year
        n_train_years = len(np.unique(years[train_mask]))
        if n_train_years < min_train_years:
            logger.warning(
                "Год %d пропущен: только %d лет обучения (нужно ≥ %d)",
                test_year,
                n_train_years,
                min_train_years,
            )
            continue
        if not test_mask.any():
            logger.warning("Год %d пропущен: в датасете нет наблюдений", test_year)
            continue
        splits.append(
            Split(
                test_year=test_year,
                train_index=np.flatnonzero(train_mask),
                test_index=np.flatnonzero(test_mask),
            )
        )
    if not splits:
        raise ValueError("Не построено ни одного разбиения — проверьте validation.* в конфиге")
    for s in splits:
        logger.info("Разбиение — %s", s.describe(dataset["year"]))
    return splits


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R², Bias, MAPE и число наблюдений.

    R² считается относительно среднего фактических значений выборки
    (коэффициент детерминации, а не квадрат корреляции).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]

    if y_true.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan, "mape": np.nan}

    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(err) / np.where(y_true != 0, y_true, np.nan)) * 100)

    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "bias": float(np.mean(err)),
        "mape": mape,
    }


def metrics_by_group(
    predictions: pd.DataFrame,
    group: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Метрики в разрезе указанной колонки (год, регион, модель)."""
    rows: List[Dict[str, object]] = []
    for key, sub in predictions.groupby(group, sort=True):
        m = regression_metrics(sub[y_true_col].to_numpy(), sub[y_pred_col].to_numpy())
        m[group] = key
        rows.append(m)
    cols = [group, "n", "mae", "rmse", "r2", "bias", "mape"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Базовые модели-эталоны
# ---------------------------------------------------------------------------
def baseline_previous_year(dataset: pd.DataFrame, split: Split) -> np.ndarray:
    """Наивный прогноз: урожайность прошлого года (yield_lag_1)."""
    return dataset.iloc[split.test_index]["yield_lag_1"].to_numpy(dtype=float)


def baseline_regional_mean(
    dataset: pd.DataFrame,
    split: Split,
    target: str = "yield_c_ha",
) -> np.ndarray:
    """Эталон: среднее историческое значение региона по обучающей выборке.

    Если регион отсутствует в обучении, используется общее среднее — это
    поведение фиксируется явно, а не приводит к NaN.
    """
    train = dataset.iloc[split.train_index]
    test = dataset.iloc[split.test_index]
    region_mean = train.groupby("region")[target].mean()
    global_mean = float(train[target].mean())
    return test["region"].map(region_mean).fillna(global_mean).to_numpy(dtype=float)
