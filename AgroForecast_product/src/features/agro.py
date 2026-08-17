"""Аграрные признаки: лаги урожайности, площади, удобрения.

Ключевое правило: лаг строится СТРОГО по календарному году внутри региона.
Пропуск в предыдущем году даёт NaN — никакого «протягивания» последнего
известного значения. Модели (CatBoost, HistGradientBoosting) работают с NaN
нативно; RandomForest требует импутации, которая выполняется явно и только
внутри обучающего пайплайна.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _lag_by_year(
    panel: pd.DataFrame,
    column: str,
    lag: int,
    out_column: str,
) -> pd.Series:
    """Значение column за год (t - lag) для того же региона.

    Реализовано через merge по (region, year+lag), а не через shift(), чтобы
    пропущенные годы в панели не «сдвигали» ряд и не создавали ложных лагов.
    """
    src = panel[["region", "year", column]].copy()
    src["year"] = src["year"] + lag
    src = src.rename(columns={column: out_column})
    merged = panel[["region", "year"]].merge(src, on=["region", "year"], how="left")
    return merged[out_column]


def add_lag_features(
    panel: pd.DataFrame,
    yield_column: str = "yield_c_ha",
    yield_lags: Sequence[int] = (1, 2, 3),
    sown_area_column: str = "sown_area_grain_kha",
    sown_area_lags: Sequence[int] = (1,),
    rolling_windows: Sequence[int] = (3, 5),
) -> pd.DataFrame:
    """Добавляет лаги и скользящие средние по предыдущим годам.

    Скользящее среднее считается ТОЛЬКО по годам строго до текущего,
    поэтому утечки будущей информации не возникает.
    """
    out = panel.sort_values(["region", "year"]).reset_index(drop=True).copy()

    for lag in yield_lags:
        out[f"yield_lag_{lag}"] = _lag_by_year(out, yield_column, lag, f"yield_lag_{lag}")

    for lag in sown_area_lags:
        if sown_area_column in out.columns:
            out[f"sown_area_lag_{lag}"] = _lag_by_year(
                out, sown_area_column, lag, f"sown_area_lag_{lag}"
            )

    # Скользящие средние по предыдущим годам (shift(1) исключает текущий год).
    for window in rolling_windows:
        col = f"yield_roll_mean_{window}"
        out[col] = (
            out.groupby("region")[yield_column]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
        )

    # Отклонение прошлогодней урожайности от долгосрочной нормы региона:
    # характеризует, был ли предыдущий год аномальным.
    if "yield_roll_mean_5" in out.columns:
        out["yield_lag1_vs_norm"] = out["yield_lag_1"] - out["yield_roll_mean_5"]

    created = [c for c in out.columns if c not in panel.columns]
    logger.info("Аграрные лаги: добавлено признаков %d: %s", len(created), created)
    return out


def add_fertilizer_features(
    panel: pd.DataFrame,
    mineral_column: str = "fert_mineral_kg_ha",
    organic_column: str = "fert_organic_t_ha",
) -> pd.DataFrame:
    """Признаки по удобрениям.

    Внесение удобрений за год t известно только по годовым итогам (публикация
    в марте t+1), поэтому как признак для прогноза года t используется
    ЛАГ за год t-1. Текущий год в модель не подаётся (см. leakage.py).
    """
    out = panel.copy()
    for col, name in ((mineral_column, "fert_mineral_lag_1"), (organic_column, "fert_organic_lag_1")):
        if col in out.columns:
            out[name] = _lag_by_year(out, col, 1, name)
    created = [c for c in out.columns if c not in panel.columns]
    logger.info("Признаки удобрений: %s", created)
    return out


def add_structural_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Структурные признаки, известные до уборки текущего года.

    *   ``grain_share_lag_1`` — доля зерновых в посевной площади предыдущего
        года: устойчивая характеристика специализации региона.
    *   ``sown_area_change_1`` — относительное изменение посевной площади
        зерновых между t-2 и t-1.

    Посевная площадь ТЕКУЩЕГО года в базовый набор не включается: окончательные
    итоги публикуются позже начала сезона (см. leakage.py).
    """
    out = panel.copy()
    if {"sown_area_grain_kha", "sown_area_total_kha"}.issubset(out.columns):
        grain_lag = _lag_by_year(out, "sown_area_grain_kha", 1, "_g1")
        total_lag = _lag_by_year(out, "sown_area_total_kha", 1, "_t1")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["grain_share_lag_1"] = np.where(
                total_lag > 0, grain_lag / total_lag, np.nan
            )
    if "sown_area_grain_kha" in out.columns:
        g1 = _lag_by_year(out, "sown_area_grain_kha", 1, "_g1")
        g2 = _lag_by_year(out, "sown_area_grain_kha", 2, "_g2")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["sown_area_change_1"] = np.where(g2 > 0, g1 / g2 - 1.0, np.nan)

    created = [c for c in out.columns if c not in panel.columns]
    logger.info("Структурные признаки: %s", created)
    return out
