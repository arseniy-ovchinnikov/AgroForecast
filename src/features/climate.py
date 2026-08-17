"""Климатические признаки: помесячные ряды ERA5 → «регион × год».

Принципы:

*   Признаки строятся только по месяцам вегетационного сезона (по умолчанию
    апрель–сентябрь) — именно они физически определяют урожай зерновых.
*   Никакого «взрыва признаков»: помесячные значения + сезонные агрегаты +
    небольшой набор физически осмысленных производных.
*   Год-строка считается пригодной только при ПОЛНОМ наборе месяцев сезона;
    неполные годы помечаются, а не «дозаполняются».
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Колонка ERA5 -> (короткий префикс, набор сезонных агрегатов)
CLIMATE_COLUMNS: Dict[str, Dict[str, object]] = {
    "t2m_c": {"prefix": "t2m", "aggs": ["mean", "min", "max"]},
    "swvl1_m3m3": {"prefix": "swvl1", "aggs": ["mean", "min", "max"]},
    "tp_mm": {"prefix": "tp", "aggs": ["sum"]},
    "ssrd_mj_m2": {"prefix": "ssrd", "aggs": ["sum"]},
    "pev_mm": {"prefix": "pev", "aggs": ["sum"]},
}

MONTH_ABBR = {4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep",
              1: "jan", 2: "feb", 3: "mar", 10: "oct", 11: "nov", 12: "dec"}


def build_climate_features(
    monthly: pd.DataFrame,
    season_months: Sequence[int],
    monthly_months: Sequence[int],
    warm_threshold_c: float = 15.0,
    cold_threshold_c: float = 5.0,
) -> pd.DataFrame:
    """Преобразует помесячные ERA5 в годовые климатические признаки.

    Args:
        monthly: DataFrame[region, year, month, t2m_c, swvl1_m3m3, tp_mm,
            ssrd_mj_m2, pev_mm].
        season_months: месяцы для сезонных агрегатов.
        monthly_months: месяцы, выносимые отдельными колонками.

    Returns:
        DataFrame[region, year, <климатические признаки>, season_months_available].
    """
    required = {"region", "year", "month"} | set(CLIMATE_COLUMNS)
    missing = required - set(monthly.columns)
    if missing:
        raise KeyError(f"В помесячных данных ERA5 отсутствуют колонки: {sorted(missing)}")

    season = monthly[monthly["month"].isin(list(season_months))].copy()

    # --- Сезонные агрегаты ---------------------------------------------------
    agg_spec: Dict[str, List[str]] = {}
    for col, spec in CLIMATE_COLUMNS.items():
        agg_spec[col] = list(spec["aggs"])
    seasonal = season.groupby(["region", "year"]).agg(agg_spec)
    seasonal.columns = [
        f"{CLIMATE_COLUMNS[col]['prefix']}_season_{how}" for col, how in seasonal.columns
    ]
    seasonal = seasonal.reset_index()

    # Контроль полноты сезона: сколько месяцев реально присутствует
    counts = (
        season.groupby(["region", "year"])["month"].nunique().rename("season_months_available")
    ).reset_index()
    seasonal = seasonal.merge(counts, on=["region", "year"], how="left")

    # --- Помесячные значения -------------------------------------------------
    wide_parts: List[pd.DataFrame] = []
    for month in monthly_months:
        sub = monthly[monthly["month"] == month]
        if sub.empty:
            logger.warning("В ERA5 нет месяца %d — помесячные признаки не построены", month)
            continue
        renamed = sub[["region", "year"] + list(CLIMATE_COLUMNS)].rename(
            columns={
                col: f"{spec['prefix']}_{MONTH_ABBR[month]}"
                for col, spec in CLIMATE_COLUMNS.items()
            }
        )
        wide_parts.append(renamed)

    features = seasonal
    for part in wide_parts:
        features = features.merge(part, on=["region", "year"], how="left")

    # --- Производные (физически обоснованные) --------------------------------
    # Водный баланс сезона: осадки минус потенциальное испарение.
    features["water_balance_season_mm"] = (
        features["tp_season_sum"] - features["pev_season_sum"]
    )
    # Индекс сухости (аридности): PET / P. Ограничен сверху, чтобы редкие
    # регионы с почти нулевыми осадками не порождали выбросы.
    with np.errstate(divide="ignore", invalid="ignore"):
        aridity = features["pev_season_sum"] / features["tp_season_sum"].replace(0, np.nan)
    features["aridity_index_season"] = aridity.clip(upper=20.0)

    # Число месяцев сезона с температурой выше/ниже порога.
    temp_flags = season.copy()
    temp_flags["warm"] = (temp_flags["t2m_c"] > warm_threshold_c).astype(int)
    temp_flags["cold"] = (temp_flags["t2m_c"] < cold_threshold_c).astype(int)
    flags = (
        temp_flags.groupby(["region", "year"])[["warm", "cold"]]
        .sum()
        .rename(
            columns={
                "warm": f"n_months_above_{int(warm_threshold_c)}c",
                "cold": f"n_months_below_{int(cold_threshold_c)}c",
            }
        )
        .reset_index()
    )
    features = features.merge(flags, on=["region", "year"], how="left")

    # Сумма активных температур (приближение по месячным средним):
    # Σ по месяцам сезона (T_месяц − 10 °C)⁺ × число дней месяца.
    gdd = season.copy()
    days = gdd.apply(
        lambda r: pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1).days_in_month,
        axis=1,
    )
    gdd["gdd"] = np.clip(gdd["t2m_c"] - 10.0, 0, None) * days
    gdd_sum = gdd.groupby(["region", "year"])["gdd"].sum().rename("gdd_base10_season").reset_index()
    features = features.merge(gdd_sum, on=["region", "year"], how="left")

    features = features.sort_values(["region", "year"]).reset_index(drop=True)
    n_incomplete = int((features["season_months_available"] < len(season_months)).sum())
    logger.info(
        "Климатические признаки: %d строк, %d регионов, %d–%d, признаков: %d "
        "(строк с неполным сезоном: %d)",
        len(features),
        features["region"].nunique(),
        features["year"].min(),
        features["year"].max(),
        features.shape[1] - 2,
        n_incomplete,
    )
    return features


def climate_feature_columns(features: pd.DataFrame) -> List[str]:
    """Список названий климатических признаков (без служебных колонок)."""
    service = {"region", "year", "season_months_available"}
    return [c for c in features.columns if c not in service]
