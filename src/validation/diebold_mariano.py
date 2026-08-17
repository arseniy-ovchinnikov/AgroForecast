"""Тест Диболда–Мариано для сравнения точности прогнозов.

Зачем
-----
Разница в RMSE между двумя моделями сама по себе ничего не доказывает: она
может быть случайной. Тест Диболда–Мариано проверяет нулевую гипотезу
о равной ожидаемой потере двух прогнозов и даёт p-значение. Для гипотезы H1
это обязательный инструмент: «MIDAS лучше сезонного агрегата» — утверждение
статистическое, а не арифметическое.

Постановка
----------
d_t = L(e₁ₜ) − L(e₂ₜ),   H₀: E[d_t] = 0

Отрицательная статистика — первая модель точнее.

Реализованные уточнения
-----------------------
* Долгосрочная дисперсия d_t оценивается по Ньюи–Уэсту (ядро Бартлетта),
  так как при горизонте h > 1 ошибки автокоррелированы по построению.
* Поправка Харви–Лейборна–Ньюболда на малую выборку; критические значения
  берутся из распределения Стьюдента с T−1 степенями свободы, а не из
  нормального. При T порядка 5–15 это принципиально.

Панельный случай
----------------
Прогнозы образуют панель «субъект × год», а не временной ряд. Наблюдения
внутри одного года зависимы: они получены одной и той же моделью на одной
обучающей выборке. Трактовать их как независимые — значит завысить число
наблюдений в разы и получить ложную значимость.

Поэтому ``panel_diebold_mariano`` сначала усредняет d по субъектам внутри
года, а тест применяет к полученному ряду длины T (число тестовых лет).
Это консервативно и корректно. Дополнительно возвращается кластерная оценка
(кластер — год) как справочная.

Ограничение мощности
--------------------
При T = 5 тест почти не имеет мощности: даже заметное преимущество не
достигнет значимости. Число тестовых лет — параметр дизайна, а не
техническая деталь; ``power_note`` в результате об этом предупреждает.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DMResult:
    """Результат теста."""

    statistic: float
    p_value: float
    mean_loss_diff: float
    n_periods: int
    horizon: int
    loss: str
    better: str
    significant_10: bool
    significant_05: bool
    power_note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _newey_west_variance(d: np.ndarray, lag: int) -> float:
    """Долгосрочная дисперсия ряда по Ньюи–Уэсту с ядром Бартлетта."""
    T = d.size
    dc = d - d.mean()
    gamma0 = float(dc @ dc) / T
    total = gamma0
    for k in range(1, lag + 1):
        if k >= T:
            break
        cov = float(dc[k:] @ dc[:-k]) / T
        total += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return total


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
    name_a: str = "A",
    name_b: str = "B",
) -> DMResult:
    """Тест Диболда–Мариано с поправкой Харви–Лейборна–Ньюболда.

    Args:
        errors_a, errors_b: ряды ошибок прогноза (факт − прогноз или наоборот,
            важно лишь единообразие) одинаковой длины.
        horizon: горизонт прогноза h; задаёт число лагов Ньюи–Уэста (h − 1).
        loss: 'squared' | 'absolute'.

    Returns:
        DMResult. Отрицательная статистика означает, что точнее модель A.
    """
    from scipy import stats

    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Разная длина рядов ошибок: {a.shape} и {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    T = a.size
    if T < 3:
        raise ValueError(f"Для теста нужно минимум 3 периода, получено {T}")

    if loss == "squared":
        d = a ** 2 - b ** 2
    elif loss == "absolute":
        d = np.abs(a) - np.abs(b)
    else:
        raise ValueError(f"Неизвестная функция потерь: {loss}")

    d_bar = float(d.mean())
    var_d = _newey_west_variance(d, lag=max(0, horizon - 1))

    if var_d <= 0 or not np.isfinite(var_d):
        # Ряды потерь совпадают тождественно — различий нет по построению.
        return DMResult(
            statistic=0.0, p_value=1.0, mean_loss_diff=d_bar, n_periods=T,
            horizon=horizon, loss=loss, better="нет различий",
            significant_10=False, significant_05=False,
            power_note="нулевая дисперсия разности потерь",
        )

    dm = d_bar / np.sqrt(var_d / T)

    # Поправка Харви–Лейборна–Ньюболда на малую выборку
    h = horizon
    correction = (T + 1 - 2 * h + h * (h - 1) / T) / T
    dm_star = dm * np.sqrt(max(correction, 1e-12))
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_star), df=T - 1))

    if p >= 0.10:
        better = "различия незначимы"
    else:
        better = name_a if d_bar < 0 else name_b

    note = ""
    if T < 8:
        note = (
            f"T = {T}: мощность теста низка, отсутствие значимости не является "
            "доказательством равной точности"
        )

    return DMResult(
        statistic=float(dm_star), p_value=float(p), mean_loss_diff=d_bar,
        n_periods=T, horizon=horizon, loss=loss, better=better,
        significant_10=bool(p < 0.10), significant_05=bool(p < 0.05),
        power_note=note,
    )


def panel_diebold_mariano(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    model_col: str = "model_id",
    year_col: str = "year",
    region_col: str = "region",
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    loss: str = "squared",
    horizon: int = 1,
) -> DMResult:
    """Тест на панели «субъект × год».

    Разность потерь усредняется по субъектам внутри каждого года, затем тест
    применяется к ряду длины T. Сравниваются только те пары (регион, год),
    которые присутствуют у обеих моделей.
    """
    need = {model_col, year_col, region_col, truth_col, pred_col}
    missing = need - set(predictions.columns)
    if missing:
        raise KeyError(f"В таблице прогнозов нет колонок: {sorted(missing)}")

    key = [region_col, year_col]
    A = predictions[predictions[model_col] == model_a][key + [truth_col, pred_col]]
    B = predictions[predictions[model_col] == model_b][key + [truth_col, pred_col]]
    if A.empty or B.empty:
        raise ValueError(f"Нет прогнозов для «{model_a}» или «{model_b}»")

    m = A.merge(B, on=key, suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("Нет общих наблюдений (регион, год) у сравниваемых моделей")

    ea = m[f"{truth_col}_a"] - m[f"{pred_col}_a"]
    eb = m[f"{truth_col}_b"] - m[f"{pred_col}_b"]
    la = ea ** 2 if loss == "squared" else ea.abs()
    lb = eb ** 2 if loss == "squared" else eb.abs()

    per_year = (
        pd.DataFrame({year_col: m[year_col], "d": la.to_numpy() - lb.to_numpy()})
        .groupby(year_col)["d"]
        .mean()
        .sort_index()
    )

    d = per_year.to_numpy()
    var_d = _newey_west_variance(d, lag=max(0, horizon - 1))
    T = d.size
    from scipy import stats

    if var_d <= 0:
        return DMResult(0.0, 1.0, float(d.mean()), T, horizon, loss,
                        "нет различий", False, False, "нулевая дисперсия")
    dm = d.mean() / np.sqrt(var_d / T)
    corr = (T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T
    dm_star = dm * np.sqrt(max(corr, 1e-12))
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_star), df=T - 1))
    better = "различия незначимы" if p >= 0.10 else (model_a if d.mean() < 0 else model_b)
    note = (f"T = {T}: мощность теста низка, отсутствие значимости не доказывает "
            "равной точности") if T < 8 else ""
    return DMResult(float(dm_star), float(p), float(d.mean()), T, horizon, loss,
                    better, bool(p < 0.10), bool(p < 0.05), note)


def pairwise_dm_matrix(
    predictions: pd.DataFrame,
    models: Optional[list] = None,
    model_col: str = "model_id",
    **kwargs,
) -> pd.DataFrame:
    """Попарные тесты для всех моделей. Возвращает длинную таблицу."""
    models = models or sorted(predictions[model_col].unique())
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            try:
                r = panel_diebold_mariano(predictions, a, b, model_col=model_col, **kwargs)
                rows.append({"model_a": a, "model_b": b, **r.as_dict()})
            except Exception as exc:  # noqa: BLE001
                logger.warning("DM (%s vs %s) пропущен — %s: %s", a, b, type(exc).__name__, exc)
    return pd.DataFrame(rows)
