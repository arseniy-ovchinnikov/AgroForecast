"""Промышленный контур наукастинга валового сбора зерна.

Целевая величина работы — **валовой сбор** G (тыс. т), а не урожайность.
Прогнозируется он не напрямую, а через тождество

    G_it = Y_it · ρ_it · S_it / 10,

где Y — урожайность (ц/га убранной площади), S — посевная площадь (тыс. га),
ρ = H/S — доля посевов, дошедшая до уборки. Делитель 10 переводит
ц/га × тыс. га в тыс. т.

Почему декомпозиция, а не прямая регрессия
------------------------------------------
Разложение внутрирегиональной дисперсии log G на панели 2000–2024
(1 837 наблюдений, 78 субъектов) даёт: посевная площадь — 48,6 %,
урожайность — 39,9 %, коэффициент уборки — 11,5 %. Спутниковый индекс
информативен об урожайности, то есть охватывает менее половины
изменчивости целевой величины. Прямая регрессия G на NDVI это скрывает;
декомпозиция делает вклад каждого канала измеримым и позволяет брать
площадь из оперативной статистики, где она известна точно.

Источники компонент на момент отсечки h
---------------------------------------
Y  — MIDAS на внутрисезонных композитах NDVI плюс исторические лаги;
S  — оперативные сведения о ходе сева (ф. 4-СХ, публикуются с июня);
ρ  — среднее по трём предыдущим годам региона: ряд слабо прогнозируем
     (корреляция с собственным лагом 0,31), поэтому уровень региона —
     лучшее, что можно утверждать без данных о гибели посевов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Переводной множитель тождества G = Y·ρ·S/10 (ц/га × тыс. га → тыс. т)
UNIT_FACTOR = 10.0


def derive_components(panel: pd.DataFrame) -> pd.DataFrame:
    """Восстанавливает убранную площадь и коэффициент уборки из панели Росстата.

    H = 10·G/Y — убранная площадь, тыс. га; ρ = H/S.
    Значения ρ вне (0,2; 1,3) отбрасываются как артефакты округления
    показателей, публикуемых с разной точностью.
    """
    out = panel.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["harvested_kha"] = UNIT_FACTOR * out["gross_harvest_kt"] / out["yield_c_ha"]
        out["rho"] = out["harvested_kha"] / out["sown_area_grain_kha"]
    bad = ~out["rho"].between(0.2, 1.3)
    n_bad = int((bad & out["rho"].notna()).sum())
    out.loc[bad, "rho"] = np.nan
    if n_bad:
        logger.info("Коэффициент уборки: отброшено %d значений вне (0,2; 1,3)", n_bad)
    return out


def add_rho_forecast(panel: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Прогноз ρ — среднее по предыдущим годам региона (без текущего)."""
    out = panel.sort_values(["region", "year"]).copy()
    out["rho_hat"] = (
        out.groupby("region")["rho"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    # Регионы без истории получают общее среднее — решение фиксируется явно.
    global_mean = float(out["rho"].mean())
    n_fill = int(out["rho_hat"].isna().sum())
    out["rho_hat"] = out["rho_hat"].fillna(global_mean)
    if n_fill:
        logger.info(
            "Коэффициент уборки: %d строк без истории заполнены общим средним %.3f",
            n_fill, global_mean,
        )
    return out


@dataclass
class HarvestNowcast:
    """Результат наукастинга валового сбора."""

    predictions: pd.DataFrame
    metrics_yield: Dict[str, float]
    metrics_harvest: Dict[str, float]
    metrics_national: Dict[str, float]
    area_source: str
    cutoff_label: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


def combine_to_harvest(
    yield_pred: np.ndarray,
    rho_hat: np.ndarray,
    sown_area: np.ndarray,
) -> np.ndarray:
    """Собирает прогноз валового сбора из компонент."""
    return (
        np.asarray(yield_pred, dtype=float)
        * np.asarray(rho_hat, dtype=float)
        * np.asarray(sown_area, dtype=float)
        / UNIT_FACTOR
    )


def national_aggregate(df: pd.DataFrame, value_col: str, year_col: str = "year") -> pd.Series:
    """Сумма по стране за год — величина, релевантная продовольственной безопасности."""
    return df.groupby(year_col)[value_col].sum()


def shortfall_events(
    panel: pd.DataFrame,
    value_col: str = "gross_harvest_kt",
    window: int = 5,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Бинарное событие «существенный недобор» для системы раннего предупреждения.

    Событие в регионе i в году t: валовой сбор ниже среднего за ``window``
    предыдущих лет более чем на ``threshold`` (доля).

    Порог задан относительно собственной нормы региона, а не абсолютным
    значением: абсолютные пороги самообеспеченности определены для страны
    в целом и к отдельному субъекту неприменимы.
    """
    out = panel.sort_values(["region", "year"]).copy()
    out["norm"] = (
        out.groupby("region")[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    )
    out["shortfall"] = (out[value_col] < (1 - threshold) * out["norm"]).astype("float")
    out.loc[out["norm"].isna(), "shortfall"] = np.nan
    n = int(np.nansum(out["shortfall"].to_numpy()))
    total = int(out["shortfall"].notna().sum())
    logger.info(
        "События «недобор более %.0f %% от нормы за %d лет»: %d из %d наблюдений (%.1f %%)",
        threshold * 100, window, n, total, 100 * n / total if total else 0,
    )
    return out


def roc_curve_manual(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """ROC-кривая и AUC без внешних зависимостей.

    Возвращает (FPR, TPR, AUC). Score — «сила сигнала тревоги»: чем больше,
    тем сильнее ожидаемый недобор.
    """
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    if y_true.size == 0 or y_true.sum() == 0 or y_true.sum() == y_true.size:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate([[0.0], tps / tps[-1]])
    fpr = np.concatenate([[0.0], fps / fps[-1]])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return fpr, tpr, float(trapz(tpr, fpr))


def warning_quality(
    y_true: np.ndarray, score: np.ndarray, alarm_rate: float = 0.20
) -> Dict[str, float]:
    """Качество раннего предупреждения при заданной доле тревог.

    Порог выбирается так, чтобы система подавала тревогу в ``alarm_rate``
    доле случаев — это операционное ограничение: проверять можно ограниченное
    число регионов.
    """
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    if y_true.size == 0:
        return {}

    _, _, auc = roc_curve_manual(y_true, score)
    cutoff = float(np.quantile(score, 1 - alarm_rate))
    alarm = score >= cutoff
    tp = float(((alarm) & (y_true == 1)).sum())
    fp = float(((alarm) & (y_true == 0)).sum())
    fn = float(((~alarm) & (y_true == 1)).sum())
    tn = float(((~alarm) & (y_true == 0)).sum())
    return {
        "AUC": round(auc, 4),
        "доля_тревог": round(float(alarm.mean()), 4),
        "полнота_TPR": round(tp / (tp + fn), 4) if tp + fn else float("nan"),
        "точность_PPV": round(tp / (tp + fp), 4) if tp + fp else float("nan"),
        "ложные_тревоги_FPR": round(fp / (fp + tn), 4) if fp + tn else float("nan"),
        "событий": int(y_true.sum()),
        "наблюдений": int(y_true.size),
    }
