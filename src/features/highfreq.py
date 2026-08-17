"""Построение высокочастотной панели для MIDAS.

Модуль намеренно не знает, какой сенсор дал данные. На вход подаётся длинная
таблица «регион × год × период × переменные», на выходе — тензор
(N, V, K), согласованный по строкам с низкочастотным датасетом.

Благодаря этому один и тот же код обслуживает:
  * месячный ERA5-Land — периоды 4…9 (апрель–сентябрь), K = 6;
  * 16-дневные композиты MODIS MOD13Q1 — периоды 1…K по номеру композита;
  * любой другой внутрисезонный источник.

Замена источника — это замена аргументов ``variables`` и ``period_values``,
а не переписывание модели.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HighFreqPanel:
    """Тензор высокочастотных регрессоров и его метаданные."""

    X: np.ndarray                 # (N, V, K)
    variables: List[str]
    periods: List[int]            # длины K, в хронологическом порядке
    index: pd.DataFrame           # (N, 2): region, year — соответствие строкам X

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def K(self) -> int:
        return self.X.shape[2]

    def truncate(self, k: int) -> "HighFreqPanel":
        """Оставляет только первые k периодов сезона — отсечка наукастинга.

        Горизонт прогноза задаётся именно здесь: модель на отсечке k видит
        периоды 0…k−1 и ничего позже. Это единственное место, где
        обеспечивается соблюдение информационного множества.
        """
        if not 1 <= k <= self.K:
            raise ValueError(f"Отсечка k = {k} вне диапазона 1…{self.K}")
        return HighFreqPanel(
            X=self.X[:, :, :k].copy(),
            variables=list(self.variables),
            periods=list(self.periods[:k]),
            index=self.index.copy(),
        )

    def seasonal_mean(self) -> np.ndarray:
        """Среднее по периодам: (N, V, 1). Это и есть эталон B4.

        Тот же информационный набор, что у MIDAS, но свёрнутый в одно число
        на переменную. Сравнение MIDAS против этого объекта изолирует ровно
        один эффект — отказ от временнóго агрегирования.
        """
        return self.X.mean(axis=2, keepdims=True)


def build_high_freq_panel(
    long_df: pd.DataFrame,
    index_df: pd.DataFrame,
    variables: Sequence[str],
    period_values: Sequence[int],
    period_col: str = "month",
    region_col: str = "region",
    year_col: str = "year",
) -> Tuple[HighFreqPanel, pd.DataFrame]:
    """Собирает тензор (N, V, K), согласованный со строками index_df.

    Args:
        long_df: длинная таблица с колонками region, year, <period_col> и
            переменными из ``variables``.
        index_df: низкочастотный датасет; строки, для которых нет ПОЛНОГО
            набора периодов, исключаются и попадают в журнал.
        period_values: периоды в хронологическом порядке (например, 4…9).

    Returns:
        (панель, журнал исключённых строк). Молчаливой потери наблюдений нет.
    """
    variables = list(variables)
    period_values = list(period_values)
    missing_cols = set(variables) - set(long_df.columns)
    if missing_cols:
        raise KeyError(f"В высокочастотной таблице нет колонок: {sorted(missing_cols)}")

    sub = long_df[long_df[period_col].isin(period_values)].copy()
    pivot = sub.pivot_table(
        index=[region_col, year_col], columns=period_col, values=variables, aggfunc="first"
    )

    keys = list(zip(index_df[region_col], index_df[year_col]))
    K, V = len(period_values), len(variables)
    X = np.full((len(keys), V, K), np.nan, dtype=float)

    for v, var in enumerate(variables):
        for j, period in enumerate(period_values):
            col = (var, period)
            if col not in pivot.columns:
                logger.warning("Нет данных для %s, период %s — столбец останется пустым",
                               var, period)
                continue
            series = pivot[col]
            X[:, v, j] = series.reindex(keys).to_numpy(dtype=float)

    complete = np.isfinite(X).all(axis=(1, 2))
    dropped = index_df.loc[~complete, [region_col, year_col]].copy()
    if len(dropped):
        dropped["reason"] = "неполный набор высокочастотных периодов"
        logger.warning(
            "Исключено строк без полного набора периодов: %d (годы %s)",
            len(dropped),
            sorted(dropped[year_col].unique().tolist()),
        )

    panel = HighFreqPanel(
        X=X[complete],
        variables=variables,
        periods=period_values,
        index=index_df.loc[complete, [region_col, year_col]].reset_index(drop=True),
    )
    logger.info(
        "Высокочастотная панель: %d наблюдений × %d переменных × %d периодов "
        "(периоды %s)",
        panel.n_obs, V, K, period_values,
    )
    return panel, dropped


def standardize(
    X_train: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Стандартизация по каждой паре (переменная, период) статистиками ОБУЧЕНИЯ.

    Переменные ERA5 различаются по масштабу на три порядка (МДж/м² против
    м³/м³), из-за чего задача НМНК плохо обусловлена. Статистики берутся
    только из обучающей выборки — иначе возникнет утечка.
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return tuple((A - mu) / sd for A in (X_train, *others))
