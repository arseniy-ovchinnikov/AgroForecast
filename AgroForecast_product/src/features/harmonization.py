"""Межсенсорная гармонизация NDVI: MODIS ↔ VIIRS (гипотеза H4).

Задача
------
MOD13Q1 (Terra, 250 м, шаг 16 дней) и VNP13A1 (Suomi NPP, 500 м, шаг 8 дней)
измеряют один и тот же физический индекс, но различаются спектральными
каналами, пространственным разрешением, временем пролёта и алгоритмом
композитинга. Прямая подстановка одного вместо другого смещает ряд и
обесценивает модель, обученную на архиве.

Гармонизация — это оценка переходной функции на годах, где оба сенсора
работали одновременно, и последующий пересчёт VIIRS в «MODIS-эквивалент».

Согласование сеток
------------------
Сетки композитов вложены детерминированно: 16-дневный период MODIS m
покрывает дни года (m−1)·16+1 … m·16, то есть ровно два 8-дневных периода
VIIRS — 2m−1 и 2m. Поэтому сопоставление не требует интерполяции по датам:
VIIRS усредняется по паре периодов. Это точное соответствие, а не сближение
по ближайшей дате.

Спецификации переходной функции
-------------------------------
``global``      одна пара (a, b) на всю страну: modis ≈ a + b·viirs;
``per_region``  своя пара на каждый субъект — учитывает различия ландшафта
                и доли пашни, но требует достаточного числа наблюдений;
``ratio``       только масштаб (b), без сдвига — жёсткое ограничение,
                полезное как проверка устойчивости.

Выбор спецификации не постулируется, а решается сравнением вневыборочной
ошибки на удерживаемых годах перекрытия.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

MODIS_STEP = 16
VIIRS_STEP = 8
MIN_OBS_PER_REGION = 30


def viirs_periods_for_modis(modis_period: int) -> Tuple[int, int]:
    """Пара 8-дневных периодов VIIRS, покрывающих 16-дневный период MODIS."""
    return 2 * modis_period - 1, 2 * modis_period


def align_sensors(
    modis: pd.DataFrame,
    viirs: pd.DataFrame,
    value_columns: Sequence[str] = ("ndvi", "evi"),
) -> pd.DataFrame:
    """Сводит два сенсора в одну таблицу по (регион, год, период MODIS).

    VIIRS усредняется по паре своих периодов, попадающих в окно MODIS.

    Returns:
        DataFrame[region, year, period, <col>_modis, <col>_viirs, n_viirs_periods]
    """
    value_columns = list(value_columns)
    v = viirs.copy()
    v["modis_period"] = ((v["period"] + 1) // 2).astype(int)

    agg = {c: "mean" for c in value_columns}
    agg["period"] = "nunique"
    v_agg = (
        v.groupby(["region", "year", "modis_period"], as_index=False)
        .agg(agg)
        .rename(columns={"modis_period": "period", "period": "n_viirs_periods"})
    )

    merged = modis[["region", "year", "period"] + value_columns].merge(
        v_agg, on=["region", "year", "period"], how="inner",
        suffixes=("_modis", "_viirs"),
    )

    # Крайние окна сезона могут быть покрыты лишь одним периодом VIIRS вместо
    # двух: 16-дневное окно MODIS выходит за границу выгрузки. Такие пары
    # сравнивают разные интервалы времени и смещают калибровку, поэтому
    # исключаются явно, с записью в лог.
    partial = merged["n_viirs_periods"] < 2
    if partial.any():
        logger.warning(
            "Исключено %d пар с неполным окном (один период VIIRS вместо двух); "
            "затронуты периоды MODIS: %s",
            int(partial.sum()),
            sorted(merged.loc[partial, "period"].unique().tolist()),
        )
        merged = merged[~partial]

    if merged.empty:
        raise ValueError(
            "Нет пересечения MODIS и VIIRS по (регион, год, период). "
            "Проверьте, что обе выгрузки охватывают общие годы"
        )
    logger.info(
        "Согласование сенсоров: %d парных наблюдений, %d субъектов, годы %d–%d",
        len(merged), merged["region"].nunique(),
        merged["year"].min(), merged["year"].max(),
    )
    return merged


@dataclass
class Calibration:
    """Переходная функция VIIRS → MODIS-эквивалент."""

    mode: str
    column: str
    intercept: float
    slope: float
    per_region: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    n_obs: int = 0
    r2: float = np.nan
    rmse: float = np.nan
    train_years: Tuple[int, int] = (0, 0)

    def apply(self, values: np.ndarray, regions: Optional[np.ndarray] = None) -> np.ndarray:
        """Пересчитывает значения VIIRS в MODIS-эквивалент."""
        values = np.asarray(values, dtype=float)
        if self.mode != "per_region" or regions is None:
            return self.intercept + self.slope * values
        out = np.empty_like(values)
        for i, (v, r) in enumerate(zip(values, regions)):
            a, b = self.per_region.get(str(r), (self.intercept, self.slope))
            out[i] = a + b * v
        return out

    def summary(self) -> str:
        return (
            f"{self.column} [{self.mode}]: a = {self.intercept:+.4f}, b = {self.slope:.4f}, "
            f"R² = {self.r2:.4f}, RMSE = {self.rmse:.4f}, n = {self.n_obs}, "
            f"годы {self.train_years[0]}–{self.train_years[1]}"
        )


def _ols(x: np.ndarray, y: np.ndarray, with_intercept: bool = True) -> Tuple[float, float]:
    if with_intercept:
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(coef[0]), float(coef[1])
    coef, *_ = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)
    return 0.0, float(coef[0])


def fit_calibration(
    aligned: pd.DataFrame,
    column: str = "ndvi",
    mode: str = "global",
    train_years: Optional[Sequence[int]] = None,
) -> Calibration:
    """Оценивает переходную функцию на годах перекрытия.

    Args:
        mode: 'global' | 'per_region' | 'ratio'.
        train_years: годы для оценивания; остальные остаются для проверки.
    """
    df = aligned if train_years is None else aligned[aligned["year"].isin(list(train_years))]
    if df.empty:
        raise ValueError("Обучающая выборка калибровки пуста")

    x = df[f"{column}_viirs"].to_numpy(dtype=float)
    y = df[f"{column}_modis"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if mode == "ratio":
        a, b = _ols(x, y, with_intercept=False)
    else:
        a, b = _ols(x, y, with_intercept=True)

    per_region: Dict[str, Tuple[float, float]] = {}
    if mode == "per_region":
        thin: List[str] = []
        for region, sub in df.groupby("region"):
            xr = sub[f"{column}_viirs"].to_numpy(dtype=float)
            yr = sub[f"{column}_modis"].to_numpy(dtype=float)
            m = np.isfinite(xr) & np.isfinite(yr)
            if m.sum() < MIN_OBS_PER_REGION:
                thin.append(str(region))
                continue
            per_region[str(region)] = _ols(xr[m], yr[m], with_intercept=True)
        if thin:
            logger.warning(
                "Субъектов с числом наблюдений менее %d — %d; для них применяется "
                "общая калибровка: %s", MIN_OBS_PER_REGION, len(thin), thin[:10],
            )

    cal = Calibration(
        mode=mode, column=column, intercept=a, slope=b, per_region=per_region,
        n_obs=int(x.size), train_years=(int(df["year"].min()), int(df["year"].max())),
    )
    pred = cal.apply(x, df.loc[ok, "region"].to_numpy() if mode == "per_region" else None)
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    cal.r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    cal.rmse = float(np.sqrt(np.mean(resid ** 2)))
    logger.info("Калибровка — %s", cal.summary())
    return cal


def evaluate_calibration(
    aligned: pd.DataFrame,
    calibration: Calibration,
    test_years: Sequence[int],
) -> Dict[str, float]:
    """Вневыборочная проверка калибровки на удерживаемых годах.

    Сравниваются три величины: расхождение сенсоров без коррекции,
    после коррекции и доля устранённого смещения.
    """
    test = aligned[aligned["year"].isin(list(test_years))]
    if test.empty:
        raise ValueError(f"Нет наблюдений в тестовых годах {list(test_years)}")

    col = calibration.column
    x = test[f"{col}_viirs"].to_numpy(dtype=float)
    y = test[f"{col}_modis"].to_numpy(dtype=float)
    regions = test["region"].to_numpy()

    raw_bias = float(np.mean(x - y))
    raw_rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    adj = calibration.apply(x, regions)
    adj_bias = float(np.mean(adj - y))
    adj_rmse = float(np.sqrt(np.mean((adj - y) ** 2)))

    result = {
        "n_test": int(len(test)),
        "смещение_без_коррекции": round(raw_bias, 5),
        "смещение_после": round(adj_bias, 5),
        "RMSE_без_коррекции": round(raw_rmse, 5),
        "RMSE_после": round(adj_rmse, 5),
        "устранено_смещения_%": round(100 * (1 - abs(adj_bias) / abs(raw_bias)), 2)
        if abs(raw_bias) > 1e-9 else np.nan,
        "снижение_RMSE_%": round(100 * (1 - adj_rmse / raw_rmse), 2)
        if raw_rmse > 1e-9 else np.nan,
    }
    logger.info(
        "Проверка калибровки на годах %s: смещение %.4f → %.4f, RMSE %.4f → %.4f",
        list(test_years), raw_bias, adj_bias, raw_rmse, adj_rmse,
    )
    return result


def compare_modes(
    aligned: pd.DataFrame,
    column: str = "ndvi",
    holdout_years: int = 3,
) -> pd.DataFrame:
    """Сравнивает спецификации переходной функции по вневыборочной ошибке.

    Последние ``holdout_years`` лет перекрытия удерживаются для проверки —
    выбор спецификации делается по данным, а не по предпочтению.
    """
    years = sorted(aligned["year"].unique())
    if len(years) <= holdout_years:
        raise ValueError(
            f"Лет перекрытия ({len(years)}) недостаточно при holdout_years={holdout_years}"
        )
    train, test = years[:-holdout_years], years[-holdout_years:]
    logger.info("Калибровка: обучение %s, проверка %s", train, test)

    rows = []
    for mode in ("global", "per_region", "ratio"):
        cal = fit_calibration(aligned, column, mode, train)
        ev = evaluate_calibration(aligned, cal, test)
        rows.append({
            "спецификация": mode, "a": round(cal.intercept, 5), "b": round(cal.slope, 5),
            "R2_обучение": round(cal.r2, 4), **ev,
        })
    out = pd.DataFrame(rows).sort_values("RMSE_после").reset_index(drop=True)
    logger.info("Сравнение спецификаций:\n%s", out.to_string(index=False))
    return out
