"""Реестр признаков и контроль утечки данных (data leakage).

Модель прогнозирует урожайность зерновых года *t*. Признак допустим только
если его значение становится известным РАНЬШЕ, чем официальная урожайность
года *t*, и при этом не является функцией фактического урожая года *t*.

Горизонт прогноза, принятый в проекте
-------------------------------------
«Конец сезона»: прогноз формируется после завершения вегетации (начало
октября года *t*), но до публикации официальной статистики Росстата
(предварительные данные — 3-я декада декабря *t*, окончательные — март *t+1*).
Это обосновано паспортами показателей ЕМИСС (поле «Представляется»).

Если требуется более ранний, внутрисезонный прогноз (например, на 1 июля),
достаточно сократить ``features.season_months`` в configs/config.yaml —
код автоматически перестроит климатические признаки, а реестр ниже
пересчитает даты доступности.

Источники дат публикации
------------------------
Паспорта показателей ЕМИСС (лист «Паспорт» в выгрузках Росстата), поле
«Периодичность и характеристика временного ряда» → «Представляется»:
  * урожайность, посевные площади (годовые итоги) — 3-я декада февраля,
    1-я декада марта, 3-я декада декабря, 3-я декада марта;
  * внесение удобрений — 6 марта года *t+1*;
  * оперативные сведения о сборе урожая — на 8–12-й рабочий день после
    отчётного месяца.
ERA5-Land (Copernicus C3S): месячные средние публикуются примерно на 6-й день
месяца, следующего за отчётным.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Реестр признаков
# ---------------------------------------------------------------------------
# reference_year: к какому году относится ЗНАЧЕНИЕ признака относительно
#                 прогнозируемого года t.
# available: когда значение становится известным.
# allowed:   допустим ли признак для прогноза урожайности года t.
FEATURE_REGISTRY: List[Dict[str, object]] = [
    # --- Целевая переменная --------------------------------------------------
    {
        "feature": "yield_c_ha",
        "role": "target",
        "source": "Росстат, «Регионы России», табл. 13.11",
        "reference_year": "t",
        "availability": "декабрь t (предварит.) — март t+1 (окончат.)",
        "allowed_for_prediction": False,
        "reason": "Целевая переменная; не является признаком",
    },
    # --- Климат ERA5 ---------------------------------------------------------
    {
        "feature": "climate_season_*",
        "role": "feature",
        "source": "ERA5-Land, месячные средние",
        "reference_year": "t (апрель–сентябрь)",
        "availability": "~6-е число месяца, следующего за отчётным → сезон закрыт к 6 октября t",
        "allowed_for_prediction": True,
        "reason": "Метеоданные независимы от факта уборки и доступны до публикации урожайности",
    },
    {
        "feature": "climate_monthly_*",
        "role": "feature",
        "source": "ERA5-Land, месячные средние",
        "reference_year": "t (по месяцам)",
        "availability": "~6-е число следующего месяца",
        "allowed_for_prediction": True,
        "reason": "См. выше; помесячная детализация того же источника",
    },
    # --- Исторические лаги урожайности --------------------------------------
    {
        "feature": "yield_lag_1",
        "role": "feature",
        "source": "Росстат, табл. 13.11",
        "reference_year": "t-1",
        "availability": "март t",
        "allowed_for_prediction": True,
        "reason": "Окончательные данные за t-1 опубликованы до конца сезона t",
    },
    {
        "feature": "yield_lag_2",
        "role": "feature",
        "source": "Росстат, табл. 13.11",
        "reference_year": "t-2",
        "availability": "март t-1",
        "allowed_for_prediction": True,
        "reason": "Опубликовано задолго до прогнозируемого года",
    },
    {
        "feature": "yield_lag_3",
        "role": "feature",
        "source": "Росстат, табл. 13.11",
        "reference_year": "t-3",
        "availability": "март t-2",
        "allowed_for_prediction": True,
        "reason": "Опубликовано задолго до прогнозируемого года",
    },
    {
        "feature": "yield_roll_mean_3",
        "role": "feature",
        "source": "Росстат, табл. 13.11 (производный)",
        "reference_year": "t-3 … t-1",
        "availability": "март t",
        "allowed_for_prediction": True,
        "reason": "Считается по годам строго до t (shift(1) перед rolling)",
    },
    {
        "feature": "yield_roll_mean_5",
        "role": "feature",
        "source": "Росстат, табл. 13.11 (производный)",
        "reference_year": "t-5 … t-1",
        "availability": "март t",
        "allowed_for_prediction": True,
        "reason": "Считается по годам строго до t",
    },
    {
        "feature": "yield_lag1_vs_norm",
        "role": "feature",
        "source": "производный от 13.11",
        "reference_year": "t-5 … t-1",
        "availability": "март t",
        "allowed_for_prediction": True,
        "reason": "Только прошлые годы",
    },
    # --- Площади -------------------------------------------------------------
    {
        "feature": "sown_area_grain_kha",
        "role": "excluded",
        "source": "Росстат, табл. 13.5",
        "reference_year": "t",
        "availability": "декабрь t — февраль t+1 (окончательные итоги)",
        "allowed_for_prediction": False,
        "reason": (
            "Окончательные итоги по посевной площади года t публикуются после "
            "завершения уборки; включение создаёт риск утечки. Используется лаг"
        ),
    },
    {
        "feature": "sown_area_lag_1",
        "role": "feature",
        "source": "Росстат, табл. 13.5",
        "reference_year": "t-1",
        "availability": "февраль t",
        "allowed_for_prediction": True,
        "reason": "Окончательные итоги за t-1 доступны до начала сезона t",
    },
    {
        "feature": "grain_share_lag_1",
        "role": "feature",
        "source": "Росстат, табл. 13.4 и 13.5 (производный)",
        "reference_year": "t-1",
        "availability": "февраль t",
        "allowed_for_prediction": True,
        "reason": "Структура посевов прошлого года; устойчивая характеристика региона",
    },
    {
        "feature": "sown_area_change_1",
        "role": "feature",
        "source": "Росстат, табл. 13.5 (производный)",
        "reference_year": "t-2 → t-1",
        "availability": "февраль t",
        "allowed_for_prediction": True,
        "reason": "Только прошлые годы",
    },
    {
        "feature": "sown_area_total_kha",
        "role": "excluded",
        "source": "Росстат, табл. 13.4",
        "reference_year": "t",
        "availability": "декабрь t — февраль t+1",
        "allowed_for_prediction": False,
        "reason": "Тот же аргумент, что и для посевной площади зерновых года t",
    },
    # --- Валовой сбор и убранная площадь ------------------------------------
    {
        "feature": "gross_harvest_kt",
        "role": "excluded",
        "source": "Росстат, табл. 13.10",
        "reference_year": "t",
        "availability": "декабрь t — март t+1",
        "allowed_for_prediction": False,
        "reason": (
            "ПРЯМАЯ УТЕЧКА: урожайность = валовой сбор / убранная площадь. "
            "Признак известен только после уборки года t"
        ),
    },
    {
        "feature": "harvested_area",
        "role": "excluded",
        "source": "Росстат, оперативные формы 2-фермер / 29-СХ",
        "reference_year": "t",
        "availability": "по ходу и после уборки t",
        "allowed_for_prediction": False,
        "reason": "Знаменатель целевой переменной; прямая утечка",
    },
    {
        "feature": "namolocheno_s_1ga",
        "role": "excluded",
        "source": "Росстат, «Намолочено с 1 га в сельхозорганизациях»",
        "reference_year": "t (оперативно, помесячно)",
        "availability": "на 8–12-й рабочий день после отчётного месяца",
        "allowed_for_prediction": False,
        "reason": (
            "Оперативная урожайность по ходу уборки — фактически та же целевая "
            "величина; кроме того, ряд доступен только с 2023 г."
        ),
    },
    # --- Удобрения -----------------------------------------------------------
    {
        "feature": "fert_mineral_kg_ha",
        "role": "excluded",
        "source": "Росстат, табл. 13.21.1",
        "reference_year": "t",
        "availability": "6 марта t+1",
        "allowed_for_prediction": False,
        "reason": "Годовой итог за t публикуется уже после уборки; используется лаг",
    },
    {
        "feature": "fert_mineral_lag_1",
        "role": "feature",
        "source": "Росстат, табл. 13.21.1",
        "reference_year": "t-1",
        "availability": "6 марта t",
        "allowed_for_prediction": True,
        "reason": "Уровень интенсификации прошлого года; известен до сезона t",
    },
    {
        "feature": "fert_organic_lag_1",
        "role": "feature",
        "source": "Росстат, табл. 13.21.2",
        "reference_year": "t-1",
        "availability": "6 марта t",
        "allowed_for_prediction": True,
        "reason": "Аналогично минеральным удобрениям",
    },
    {
        "feature": "fert_organic_t_ha",
        "role": "excluded",
        "source": "Росстат, табл. 13.21.2",
        "reference_year": "t",
        "availability": "6 марта t+1",
        "allowed_for_prediction": False,
        "reason": "Годовой итог публикуется после уборки",
    },
    # --- Служебные -----------------------------------------------------------
    {
        "feature": "region",
        "role": "identifier",
        "source": "—",
        "reference_year": "—",
        "availability": "всегда",
        "allowed_for_prediction": True,
        "reason": "Категориальный идентификатор субъекта (CatBoost: cat_feature)",
    },
    {
        "feature": "year",
        "role": "identifier",
        "source": "—",
        "reference_year": "t",
        "availability": "всегда",
        "allowed_for_prediction": False,
        "reason": (
            "В обучении не используется как признак: при временной валидации "
            "тестовый год лежит вне диапазона обучения и модель не сможет "
            "экстраполировать по этой оси"
        ),
    },
]


def leakage_table() -> pd.DataFrame:
    """Реестр признаков в виде таблицы (для отчёта и артефактов)."""
    df = pd.DataFrame(FEATURE_REGISTRY)
    return df[
        [
            "feature",
            "role",
            "source",
            "reference_year",
            "availability",
            "allowed_for_prediction",
            "reason",
        ]
    ]


# ---------------------------------------------------------------------------
# Наборы признаков для сравнения моделей
# ---------------------------------------------------------------------------
def _match(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    return [c for c in columns if any(c.startswith(p) for p in prefixes)]


CLIMATE_PREFIXES = ("t2m_", "swvl1_", "tp_", "ssrd_", "pev_", "water_balance_",
                    "aridity_", "n_months_", "gdd_")
HISTORY_PREFIXES = ("yield_lag_", "yield_roll_mean_", "yield_lag1_vs_norm")
AGRO_PREFIXES = ("sown_area_lag_", "grain_share_lag_", "sown_area_change_",
                 "fert_mineral_lag_", "fert_organic_lag_")

FORBIDDEN_COLUMNS = frozenset(
    {
        "yield_c_ha",
        "gross_harvest_kt",
        "sown_area_grain_kha",
        "sown_area_total_kha",
        "fert_mineral_kg_ha",
        "fert_organic_t_ha",
        "year",
        "season_months_available",
    }
)


def feature_set(name: str, columns: Sequence[str]) -> List[str]:
    """Возвращает список колонок для указанного набора признаков.

    Наборы:
        baseline_historical    — только исторические лаги урожайности;
        climate_only           — только климат;
        climate_plus_history   — климат + история урожайности;
        climate_history_agro   — климат + история + агростатистика (площади,
                                 удобрения) — полный допустимый набор.
    """
    cols = [c for c in columns if c not in FORBIDDEN_COLUMNS and c != "region"]
    if name == "baseline_historical":
        selected = _match(cols, HISTORY_PREFIXES)
    elif name == "climate_only":
        selected = _match(cols, CLIMATE_PREFIXES)
    elif name == "climate_plus_history":
        selected = _match(cols, CLIMATE_PREFIXES + HISTORY_PREFIXES)
    elif name == "climate_history_agro":
        selected = _match(cols, CLIMATE_PREFIXES + HISTORY_PREFIXES + AGRO_PREFIXES)
    else:
        raise ValueError(f"Неизвестный набор признаков: {name}")

    if not selected:
        raise ValueError(f"Набор признаков «{name}» пуст — проверьте колонки датасета")
    return sorted(selected)


def assert_no_forbidden_features(features: Sequence[str]) -> None:
    """Жёсткая проверка: среди признаков нет запрещённых колонок.

    Raises:
        ValueError: при обнаружении утечки.
    """
    bad = sorted(set(features) & FORBIDDEN_COLUMNS)
    if bad:
        raise ValueError(
            "Обнаружена утечка данных: в наборе признаков присутствуют "
            f"запрещённые колонки {bad}. См. src/features/leakage.py"
        )
