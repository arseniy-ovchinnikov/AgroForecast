"""Проверка теста Диболда–Мариано на данных с известным ответом."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.diebold_mariano import (  # noqa: E402
    diebold_mariano,
    pairwise_dm_matrix,
    panel_diebold_mariano,
)


def test_identical_errors_no_difference() -> None:
    e = np.random.default_rng(0).normal(0, 1, 30)
    r = diebold_mariano(e, e.copy())
    assert r.p_value == 1.0 and not r.significant_10
    print(f"  ok: одинаковые прогнозы → p = {r.p_value:.3f}, «{r.better}»")


def test_antisymmetry() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 40), rng.normal(0, 2, 40)
    r1 = diebold_mariano(a, b)
    r2 = diebold_mariano(b, a)
    assert abs(r1.statistic + r2.statistic) < 1e-10
    assert abs(r1.p_value - r2.p_value) < 1e-12
    print(f"  ok: статистика антисимметрична ({r1.statistic:.3f} / {r2.statistic:.3f})")


def test_detects_clear_superiority() -> None:
    """Модель A вдвое точнее на длинном ряде — тест обязан это увидеть."""
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1.0, 120)
    b = rng.normal(0, 2.0, 120)
    r = diebold_mariano(a, b, name_a="точная", name_b="грубая")
    assert r.significant_05 and r.better == "точная", r
    print(f"  ok: явное преимущество обнаружено, DM = {r.statistic:.2f}, "
          f"p = {r.p_value:.2e}")


def test_no_false_positive_under_null() -> None:
    """При равной точности доля ложных срабатываний близка к номиналу 10 %."""
    rng = np.random.default_rng(3)
    hits = 0
    trials = 400
    for _ in range(trials):
        a = rng.normal(0, 1, 40)
        b = rng.normal(0, 1, 40)
        if diebold_mariano(a, b).significant_10:
            hits += 1
    rate = hits / trials
    assert 0.04 < rate < 0.18, f"уровень ошибки I рода {rate:.1%}"
    print(f"  ok: под нулевой гипотезой отвергается в {rate:.1%} случаев "
          f"(номинал 10 %)")


def test_small_sample_correction_is_conservative() -> None:
    """Поправка ХЛН должна давать более широкие p, чем нормальное приближение."""
    from scipy import stats
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1.0, 6), rng.normal(0, 1.6, 6)
    r = diebold_mariano(a, b)
    p_normal = 2 * (1 - stats.norm.cdf(abs(r.statistic)))
    assert r.p_value > p_normal
    assert "мощность" in r.power_note
    print(f"  ok: T = 6 → p(t) = {r.p_value:.3f} > p(норм.) = {p_normal:.3f}, "
          f"выдано предупреждение о мощности")


def _panel(n_reg=70, years=range(2010, 2025), seed=5):
    """Панель: модель «midas» точнее «seasonal» на 20 % по СКО ошибки."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_reg):
            truth = 20 + rng.normal(0, 6)
            rows.append({"region": f"R{i}", "year": y, "model_id": "midas",
                         "y_true": truth, "y_pred": truth + rng.normal(0, 4.0)})
            rows.append({"region": f"R{i}", "year": y, "model_id": "seasonal",
                         "y_true": truth, "y_pred": truth + rng.normal(0, 5.0)})
    return pd.DataFrame(rows)


def test_panel_aggregates_by_year() -> None:
    df = _panel()
    r = panel_diebold_mariano(df, "midas", "seasonal")
    assert r.n_periods == 15, f"должно быть 15 периодов, получено {r.n_periods}"
    assert r.better == "midas" and r.significant_05
    print(f"  ok: панель свёрнута к T = {r.n_periods} годам, "
          f"DM = {r.statistic:.2f}, p = {r.p_value:.2e}, лучше — {r.better}")


def test_panel_does_not_inflate_n() -> None:
    """Ключевая защита: T — это число ЛЕТ, а не число наблюдений панели."""
    df = _panel(n_reg=70, years=range(2020, 2025))
    r = panel_diebold_mariano(df, "midas", "seasonal")
    assert r.n_periods == 5, f"T = {r.n_periods}, а не 5 — панель была развёрнута!"
    assert len(df) == 2 * 70 * 5
    assert "мощность" in r.power_note
    print(f"  ok: 700 наблюдений панели → T = {r.n_periods}; "
          f"ложная значимость невозможна, выдано предупреждение")


def test_pairwise_matrix() -> None:
    df = _panel(n_reg=40, years=range(2012, 2025))
    extra = df[df.model_id == "seasonal"].copy()
    extra["model_id"] = "naive"
    extra["y_pred"] = extra["y_true"] + np.random.default_rng(9).normal(0, 7, len(extra))
    m = pairwise_dm_matrix(pd.concat([df, extra], ignore_index=True))
    assert len(m) == 3 and set(m.columns) >= {"model_a", "model_b", "p_value", "better"}
    print(f"  ok: попарная матрица из {len(m)} сравнений\n"
          + m[["model_a", "model_b", "statistic", "p_value", "better"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Диболд–Мариано — {len(tests)} проверок\n")
    for t in tests:
        print(f"* {t.__name__}")
        t()
    print("\nВсе проверки пройдены.")
