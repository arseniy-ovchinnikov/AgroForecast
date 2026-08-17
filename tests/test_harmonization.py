"""Проверка межсенсорной гармонизации на данных с известным ответом."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.harmonization import (  # noqa: E402
    align_sensors,
    compare_modes,
    evaluate_calibration,
    fit_calibration,
    viirs_periods_for_modis,
)

A_TRUE, B_TRUE = -0.030, 1.080   # истинная переходная функция VIIRS → MODIS


def test_period_mapping() -> None:
    """16-дневный период MODIS покрывает ровно два 8-дневных периода VIIRS."""
    for m in (1, 6, 12, 18, 23):
        p1, p2 = viirs_periods_for_modis(m)
        assert p2 == p1 + 1
        # дни года должны совпадать по границам
        assert (m - 1) * 16 + 1 == (p1 - 1) * 8 + 1
        assert m * 16 == p2 * 8
    print("  ok: сетки композитов вложены точно, без интерполяции по датам")


def _panels(n_reg=60, years=range(2012, 2026), noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    modis, viirs = [], []
    for y in years:
        for i in range(n_reg):
            base = rng.uniform(0.2, 0.5)
            for m in range(6, 19):                      # периоды MODIS сезона
                phen = np.exp(-((m - 12) ** 2) / 18.0)
                true = base + 0.35 * phen
                modis.append({"region": f"R{i:02d}", "year": y, "period": m,
                              "ndvi": true + rng.normal(0, noise),
                              "evi": 0.62 * true + rng.normal(0, noise)})
                # VIIRS = обратное преобразование MODIS плюс собственный шум
                v = (true - A_TRUE) / B_TRUE
                for p in viirs_periods_for_modis(m):
                    viirs.append({"region": f"R{i:02d}", "year": y, "period": p,
                                  "ndvi": v + rng.normal(0, noise),
                                  "evi": 0.62 * v + rng.normal(0, noise)})
    return pd.DataFrame(modis), pd.DataFrame(viirs)


def test_align_produces_paired_panel() -> None:
    m, v = _panels(n_reg=10, years=range(2012, 2015))
    a = align_sensors(m, v)
    assert len(a) == 10 * 3 * 13, len(a)
    assert (a["n_viirs_periods"] == 2).all(), "не все окна получили по два периода VIIRS"
    assert {"ndvi_modis", "ndvi_viirs", "evi_modis", "evi_viirs"} <= set(a.columns)
    print(f"  ok: {len(a)} парных наблюдений, каждое окно MODIS = 2 периода VIIRS")


def test_recovers_true_transfer_function() -> None:
    m, v = _panels()
    a = align_sensors(m, v)
    cal = fit_calibration(a, "ndvi", "global")
    assert abs(cal.intercept - A_TRUE) < 0.01, cal.intercept
    assert abs(cal.slope - B_TRUE) < 0.02, cal.slope
    assert cal.r2 > 0.98
    print(f"  ok: a = {cal.intercept:+.4f} (истина {A_TRUE:+.3f}), "
          f"b = {cal.slope:.4f} (истина {B_TRUE:.3f}), R² = {cal.r2:.4f}")


def test_calibration_removes_bias_out_of_sample() -> None:
    m, v = _panels()
    a = align_sensors(m, v)
    years = sorted(a["year"].unique())
    cal = fit_calibration(a, "ndvi", "global", years[:-3])
    ev = evaluate_calibration(a, cal, years[-3:])
    # Главный критерий — устранение систематического смещения. Снижение RMSE
    # ограничено снизу собственным шумом сенсоров и не может стремиться к 100 %.
    assert ev["устранено_смещения_%"] > 90, ev
    assert ev["снижение_RMSE_%"] > 25, ev
    print(f"  ok: смещение {ev['смещение_без_коррекции']:+.4f} → "
          f"{ev['смещение_после']:+.4f} (устранено {ev['устранено_смещения_%']:.1f} %), "
          f"RMSE снижен на {ev['снижение_RMSE_%']:.1f} %")


def test_per_region_wins_when_regions_differ() -> None:
    """Если переход зависит от региона, порегиональная схема обязана победить."""
    rng = np.random.default_rng(3)
    modis, viirs = [], []
    for i in range(40):
        b_i = 1.0 + 0.25 * rng.standard_normal()      # свой наклон у региона
        for y in range(2012, 2026):
            for mp in range(6, 19):
                true = rng.uniform(0.2, 0.7)
                modis.append({"region": f"R{i:02d}", "year": y, "period": mp,
                              "ndvi": true, "evi": true * 0.6})
                vv = true / b_i
                for p in viirs_periods_for_modis(mp):
                    viirs.append({"region": f"R{i:02d}", "year": y, "period": p,
                                  "ndvi": vv + rng.normal(0, 0.005),
                                  "evi": vv * 0.6})
    a = align_sensors(pd.DataFrame(modis), pd.DataFrame(viirs))
    cmp = compare_modes(a, "ndvi", holdout_years=3)
    best = cmp.iloc[0]["спецификация"]
    assert best == "per_region", cmp
    print(f"  ok: выбрана спецификация «{best}» — сравнение по вневыборочной ошибке\n"
          + cmp[["спецификация", "RMSE_после", "смещение_после"]].to_string(index=False))


def test_compare_modes_requires_enough_years() -> None:
    m, v = _panels(n_reg=5, years=range(2012, 2015))
    a = align_sensors(m, v)
    try:
        compare_modes(a, "ndvi", holdout_years=3)
    except ValueError as exc:
        assert "перекрытия" in str(exc)
        print("  ok: при нехватке лет перекрытия — явная ошибка, а не молчаливый результат")
        return
    raise AssertionError("нехватка лет не была обнаружена")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Гармонизация сенсоров — {len(tests)} проверок\n")
    for t in tests:
        print(f"* {t.__name__}")
        t()
    print("\nВсе проверки пройдены.")
