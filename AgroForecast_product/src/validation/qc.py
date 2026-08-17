"""Автоматический контроль качества обучающего датасета.

QC не исправляет данные и не удаляет строки молча: он формирует отчёт
с явными статусами. Строки, не прошедшие обязательные проверки, отбираются
функцией ``apply_hard_filters`` с логированием количества и причины.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Физически допустимые диапазоны (урожайность зерновых в РФ, ц/га).
PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "yield_c_ha": (0.5, 80.0),
    "t2m_season_mean": (-15.0, 30.0),
    "tp_season_sum": (0.0, 2000.0),
    "swvl1_season_mean": (0.0, 1.0),
    "sown_area_lag_1": (0.0, 10000.0),
    "fert_mineral_lag_1": (0.0, 500.0),
}


def run_qc(
    dataset: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
    expected_season_months: int,
) -> pd.DataFrame:
    """Полный набор проверок. Возвращает таблицу [check, status, detail]."""
    checks: List[Dict[str, object]] = []

    def add(name: str, ok: bool, detail: str, severity: str = "error") -> None:
        checks.append(
            {
                "check": name,
                "status": "ok" if ok else severity.upper(),
                "detail": detail,
            }
        )

    # 1. Ключ панели уникален
    dup = int(dataset.duplicated(subset=["region", "year"]).sum())
    add("Уникальность (region, year)", dup == 0, f"дубликатов: {dup}")

    # 2. Целевая переменная заполнена
    n_missing_target = int(dataset[target].isna().sum())
    add(
        "Заполненность целевой переменной",
        n_missing_target == 0,
        f"пропусков {target}: {n_missing_target}",
    )

    # 3. Диапазоны значений
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in dataset.columns:
            continue
        series = dataset[col].dropna()
        out_of_range = int(((series < lo) | (series > hi)).sum())
        add(
            f"Диапазон {col} [{lo}; {hi}]",
            out_of_range == 0,
            f"вне диапазона: {out_of_range} из {len(series)}",
            severity="warning",
        )

    # 4. Полнота климатического сезона
    if "season_months_available" in dataset.columns:
        incomplete = int(
            (dataset["season_months_available"] < expected_season_months).sum()
        )
        add(
            "Полнота вегетационного сезона ERA5",
            incomplete == 0,
            f"строк с неполным сезоном: {incomplete} (ожидалось месяцев: {expected_season_months})",
        )

    # 5. Доля пропусков по признакам
    for col in feature_columns:
        if col not in dataset.columns:
            add(f"Наличие признака {col}", False, "колонка отсутствует")
            continue
        share = float(dataset[col].isna().mean())
        add(
            f"Пропуски в {col}",
            share <= 0.30,
            f"{share:.1%}",
            severity="warning",
        )

    # 6. Покрытие по годам
    per_year = dataset.groupby("year")["region"].nunique()
    add(
        "Покрытие регионов по годам",
        bool((per_year >= 30).all()),
        f"мин. регионов в году: {int(per_year.min())} ({int(per_year.idxmin())}), "
        f"макс.: {int(per_year.max())}",
    )

    # 7. Отсутствие константных признаков
    constant = [
        c for c in feature_columns
        if c in dataset.columns and dataset[c].dropna().nunique() <= 1
    ]
    add(
        "Отсутствие константных признаков",
        not constant,
        f"константные: {constant}" if constant else "нет",
        severity="warning",
    )

    # 8. Отсутствие бесконечных значений
    numeric = dataset.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).sum())
    add("Отсутствие inf", n_inf == 0, f"бесконечных значений: {n_inf}")

    report = pd.DataFrame(checks)
    n_err = int((report["status"] == "ERROR").sum())
    n_warn = int((report["status"] == "WARNING").sum())
    logger.info("QC: проверок %d, ошибок %d, предупреждений %d", len(report), n_err, n_warn)
    for _, row in report[report["status"] != "ok"].iterrows():
        logger.warning("QC %s: %s — %s", row["status"], row["check"], row["detail"])
    return report


def apply_hard_filters(
    dataset: pd.DataFrame,
    target: str,
    expected_season_months: int,
    require_full_season: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Отбирает строки, пригодные для обучения. Возвращает (данные, журнал).

    Журнал фиксирует, сколько строк и по какой причине исключено — молчаливого
    удаления не происходит.
    """
    log: List[Dict[str, object]] = []
    df = dataset.copy()
    n0 = len(df)

    mask = df[target].notna()
    log.append(
        {
            "filter": f"{target} не пропущен",
            "removed": int((~mask).sum()),
            "remaining": int(mask.sum()),
        }
    )
    df = df[mask]

    lo, hi = PLAUSIBLE_RANGES[target]
    mask = df[target].between(lo, hi)
    log.append(
        {
            "filter": f"{target} в диапазоне [{lo}; {hi}]",
            "removed": int((~mask).sum()),
            "remaining": int(mask.sum()),
        }
    )
    df = df[mask]

    if require_full_season and "season_months_available" in df.columns:
        mask = df["season_months_available"] >= expected_season_months
        log.append(
            {
                "filter": f"полный сезон ERA5 ({expected_season_months} мес.)",
                "removed": int((~mask).sum()),
                "remaining": int(mask.sum()),
            }
        )
        df = df[mask]

    journal = pd.DataFrame(log)
    logger.info(
        "Фильтрация датасета: было %d строк, стало %d (исключено %d)",
        n0,
        len(df),
        n0 - len(df),
    )
    for _, row in journal.iterrows():
        if row["removed"]:
            logger.info("  — %s: исключено %d", row["filter"], row["removed"])
    return df.reset_index(drop=True), journal
