#!/usr/bin/env python3
"""Шаг 10. Наукастинг ВАЛОВОГО СБОРА зерна — промышленный контур.

Это основной расчёт работы. В отличие от шага 7, который прогнозирует
урожайность, здесь целевая величина — валовой сбор в тыс. т, собираемый
по тождеству

    G = Y · ρ · S / 10

с прогнозом урожайности Y моделью MIDAS на внутрисезонных спутниковых
композитах, оперативной посевной площадью S и коэффициентом уборки ρ,
оценённым по истории региона.

Что считается
-------------
1. Вневыборочный наукастинг G на сетке отсечек сезона (расширяющееся окно).
2. Метрики по регионам и по стране в целом.
3. Информационное опережение: на сколько раньше официальной публикации
   Росстата получается оценка.
4. Раннее предупреждение: качество обнаружения существенного недобора.

Вывод
-----
    results/harvest/nowcast_by_horizon.csv     метрики по отсечкам
    results/harvest/predictions.csv            прогнозы «регион × год × отсечка»
    results/harvest/national.csv               свод по стране
    results/harvest/early_warning.csv          качество раннего предупреждения
    results/harvest/component_errors.csv       вклад компонент в ошибку
    models/harvest_model.json                  параметры финальной модели

Запуск
------
    python scripts/10_harvest_nowcast.py --source viirs
    python scripts/10_harvest_nowcast.py --source viirs --area-source lag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.highfreq import build_high_freq_panel, standardize
from src.models.midas import PanelMIDAS
from src.models.production import (
    UNIT_FACTOR,
    add_rho_forecast,
    combine_to_harvest,
    derive_components,
    shortfall_events,
    warning_quality,
)
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging
from src.validation.diebold_mariano import panel_diebold_mariano
from src.validation.temporal import regression_metrics

LF_CONTROLS = ["yield_lag_1", "yield_roll_mean_3", "yield_roll_mean_5"]

SOURCES = {
    "viirs": ("viirs_region_composite.csv", ["ndvi", "evi"], "VIIRS VNP13A1, 8 дней"),
    "modis": ("modis_region_composite.csv", ["ndvi", "evi"], "MODIS MOD13Q1, 16 дней"),
    "era5": ("era5_region_month.csv",
             ["t2m_c", "swvl1_m3m3", "ssrd_mj_m2", "tp_mm", "pev_mm"],
             "ERA5-Land, месяц"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Наукастинг валового сбора зерна")
    p.add_argument("--source", choices=list(SOURCES), default="viirs")
    p.add_argument("--train-start", type=int, default=None)
    p.add_argument("--test-years", type=str, default=None)
    p.add_argument("--starts", type=int, default=6)
    p.add_argument("--scheme", default="beta", choices=["beta", "almon", "umidas"])
    p.add_argument("--area-source", choices=["operational", "lag"], default="operational",
                   help="посевная площадь: оперативная за год t или лаг за t-1")
    p.add_argument("--alarm-rate", type=float, default=0.20,
                   help="доля регионов, по которым подаётся тревога")
    return p.parse_args()


def _years(spec: Optional[str], default: List[int]) -> List[int]:
    if not spec:
        return default
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("10_harvest")

    out_dir = cfg.path("results") / "harvest"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname, hf_vars, source_label = SOURCES[args.source]
    hf_path = cfg.path("raw_processed") / fname
    ds_path = cfg.path("features_dir") / "training_dataset.csv"
    for p in (hf_path, ds_path):
        if not p.exists():
            logger.error("Не найден %s", p)
            return 2

    # ---------------------------------------------------------------- данные
    dataset = pd.read_csv(ds_path)
    long_hf = pd.read_csv(hf_path)
    if "month" in long_hf.columns and "period" not in long_hf.columns:
        long_hf = long_hf.rename(columns={"month": "period"})

    panel = derive_components(dataset)
    panel = add_rho_forecast(panel)
    need = ["gross_harvest_kt", "yield_c_ha", "sown_area_grain_kha"] + LF_CONTROLS
    before = len(panel)
    panel = panel.dropna(subset=need).reset_index(drop=True)
    logger.info("Панель: %d → %d строк после требования полноты компонент",
                before, len(panel))

    with StageTimer(logger, f"Высокочастотная панель ({source_label})"):
        periods = sorted(long_hf["period"].unique())
        hf, dropped = build_high_freq_panel(long_hf, panel, hf_vars, periods,
                                            period_col="period")
        aligned = (
            panel.merge(hf.index.assign(_k=1), on=["region", "year"], how="inner")
            .drop(columns="_k").reset_index(drop=True)
        )
        assert len(aligned) == hf.n_obs, "рассогласование панели и тензора"

    y_all = aligned["yield_c_ha"].to_numpy(float)
    g_all = aligned["gross_harvest_kt"].to_numpy(float)
    reg = aligned["region"].to_numpy()
    yr = aligned["year"].to_numpy()

    train_start = args.train_start or int(yr.min())
    default_test = [y for y in sorted(set(yr)) if y >= train_start + 5]
    test_years = _years(args.test_years, default_test)

    trend = (yr - train_start).astype(float).reshape(-1, 1)
    Z_all = np.hstack([aligned[LF_CONTROLS].to_numpy(float), trend])

    area_col = "sown_area_grain_kha" if args.area_source == "operational" else "sown_area_lag_1"
    if area_col not in aligned.columns or aligned[area_col].isna().all():
        logger.error("Нет колонки посевной площади %s", area_col)
        return 3
    area_all = aligned[area_col].to_numpy(float)
    rho_all = aligned["rho_hat"].to_numpy(float)

    logger.info(
        "Наукастинг валового сбора: %d наблюдений, %d субъектов, %d–%d; "
        "обучение с %d, тест %d–%d (T = %d); площадь — %s",
        len(aligned), aligned["region"].nunique(), yr.min(), yr.max(),
        train_start, min(test_years), max(test_years), len(test_years),
        "оперативная (ф. 4-СХ)" if args.area_source == "operational" else "лаг t−1",
    )

    # ------------------------------------------------------- по отсечкам
    records: List[Dict[str, object]] = []
    with StageTimer(logger, "Оценивание по отсечкам сезона"):
        for k in range(1, hf.K + 1):
            cut = hf.truncate(k)
            for ty in test_years:
                tr = (yr >= train_start) & (yr < ty)
                te = yr == ty
                if tr.sum() == 0 or te.sum() == 0 or len(np.unique(yr[tr])) < 5:
                    continue
                Xtr, Xte = standardize(cut.X[tr], cut.X[te])

                scheme = args.scheme if k >= 4 else "umidas"
                model = PanelMIDAS(scheme=scheme, n_starts=args.starts, seed=42)
                model.fit(Xtr, y_all[tr], reg[tr], Z_all[tr])
                y_hat = model.predict(Xte, reg[te], Z_all[te])

                # эталон: то же, но регрессоры усреднены по сезону
                base = PanelMIDAS(scheme="umidas", n_starts=1, seed=42)
                base.fit(Xtr.mean(axis=2, keepdims=True), y_all[tr], reg[tr], Z_all[tr])
                y_base = base.predict(Xte.mean(axis=2, keepdims=True), reg[te], Z_all[te])

                g_hat = combine_to_harvest(y_hat, rho_all[te], area_all[te])
                g_base = combine_to_harvest(y_base, rho_all[te], area_all[te])
                # «идеальная урожайность»: сколько ошибки даёт сама декомпозиция
                g_ideal = combine_to_harvest(y_all[te], rho_all[te], area_all[te])

                for i, idx in enumerate(np.flatnonzero(te)):
                    records.append({
                        "horizon_k": k, "region": reg[idx], "year": int(ty),
                        "yield_true": y_all[idx], "yield_pred": float(y_hat[i]),
                        "yield_base": float(y_base[i]),
                        "harvest_true": g_all[idx], "harvest_pred": float(g_hat[i]),
                        "harvest_base": float(g_base[i]),
                        "harvest_ideal": float(g_ideal[i]),
                        "sown_area": area_all[idx], "rho_hat": rho_all[idx],
                    })
            if k % 5 == 0 or k == hf.K:
                logger.info("  отсечка %d / %d", k, hf.K)

    pred = pd.DataFrame(records)
    pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- метрики
    rows = []
    for k, sub in pred.groupby("horizon_k"):
        my = regression_metrics(sub.yield_true, sub.yield_pred)
        mg = regression_metrics(sub.harvest_true, sub.harvest_pred)
        mb = regression_metrics(sub.harvest_true, sub.harvest_base)
        mi = regression_metrics(sub.harvest_true, sub.harvest_ideal)
        nat = sub.groupby("year")[["harvest_true", "harvest_pred"]].sum()
        mn = regression_metrics(nat.harvest_true, nat.harvest_pred)
        rows.append({
            "horizon_k": k, "n": mg["n"],
            "урожайность_RMSE_ц_га": round(my["rmse"], 3),
            "сбор_MAE_тыс_т": round(mg["mae"], 1),
            "сбор_RMSE_тыс_т": round(mg["rmse"], 1),
            "сбор_R2": round(mg["r2"], 4),
            "сбор_MAPE_%": round(mg["mape"], 2),
            "эталон_RMSE_тыс_т": round(mb["rmse"], 1),
            "выигрыш_vs_эталон_%": round(100 * (1 - mg["rmse"] / mb["rmse"]), 2),
            "предел_декомпозиции_RMSE": round(mi["rmse"], 1),
            "страна_MAPE_%": round(mn["mape"], 2),
            "страна_Bias_тыс_т": round(mn["bias"], 1),
        })
    metrics = pd.DataFrame(rows).sort_values("horizon_k")
    metrics.to_csv(out_dir / "nowcast_by_horizon.csv", index=False, encoding="utf-8-sig")
    logger.info("Метрики по отсечкам:\n%s", metrics.to_string(index=False))

    best_k = int(metrics.sort_values("сбор_RMSE_тыс_т").iloc[0]["horizon_k"])
    logger.info("Лучшая отсечка по RMSE валового сбора: k = %d", best_k)

    # ------------------------------------------------------- свод по стране
    nat_rows = []
    for k, sub in pred.groupby("horizon_k"):
        n = sub.groupby("year")[["harvest_true", "harvest_pred", "harvest_base"]].sum()
        n["ошибка_%"] = 100 * (n.harvest_pred / n.harvest_true - 1)
        n["horizon_k"] = k
        nat_rows.append(n.reset_index())
    national = pd.concat(nat_rows, ignore_index=True)
    national.to_csv(out_dir / "national.csv", index=False, encoding="utf-8-sig")
    logger.info(
        "Страна, отсечка k=%d:\n%s", best_k,
        national[national.horizon_k == best_k][
            ["year", "harvest_true", "harvest_pred", "ошибка_%"]
        ].round(1).to_string(index=False),
    )

    # -------------------------------------------- раннее предупреждение
    ev = shortfall_events(panel)[["region", "year", "shortfall", "norm"]]
    warn_rows = []
    for k, sub in pred.groupby("horizon_k"):
        m = sub.merge(ev, on=["region", "year"], how="left").dropna(subset=["shortfall"])
        if m.empty:
            continue
        # сигнал тревоги — относительный дефицит к норме региона
        score = 1.0 - m.harvest_pred / m.norm
        q = warning_quality(m.shortfall.to_numpy(), score.to_numpy(), args.alarm_rate)
        if q:
            q["horizon_k"] = k
            warn_rows.append(q)
    warning = pd.DataFrame(warn_rows)
    if not warning.empty:
        warning = warning[["horizon_k", "AUC", "полнота_TPR", "точность_PPV",
                           "ложные_тревоги_FPR", "доля_тревог", "событий", "наблюдений"]]
        warning.to_csv(out_dir / "early_warning.csv", index=False, encoding="utf-8-sig")
        logger.info("Раннее предупреждение:\n%s", warning.to_string(index=False))

    # ---------------------------------------------- вклад компонент в ошибку
    b = pred[pred.horizon_k == best_k]
    comp = pd.DataFrame([{
        "компонента": "полная модель",
        "RMSE_тыс_т": round(regression_metrics(b.harvest_true, b.harvest_pred)["rmse"], 1),
    }, {
        "компонента": "идеальная урожайность (предел декомпозиции)",
        "RMSE_тыс_т": round(regression_metrics(b.harvest_true, b.harvest_ideal)["rmse"], 1),
    }, {
        "компонента": "эталон: сезонное среднее вместо MIDAS",
        "RMSE_тыс_т": round(regression_metrics(b.harvest_true, b.harvest_base)["rmse"], 1),
    }])
    comp.to_csv(out_dir / "component_errors.csv", index=False, encoding="utf-8-sig")
    logger.info("Вклад компонент (k=%d):\n%s", best_k, comp.to_string(index=False))

    # -------------------------------------------------- тест значимости
    dm_rows = []
    for k, sub in pred.groupby("horizon_k"):
        a = sub[["region", "year", "harvest_true", "harvest_pred"]].rename(
            columns={"harvest_true": "y_true", "harvest_pred": "y_pred"})
        a["model_id"] = "MIDAS"
        c = sub[["region", "year", "harvest_true", "harvest_base"]].rename(
            columns={"harvest_true": "y_true", "harvest_base": "y_pred"})
        c["model_id"] = "seasonal"
        try:
            r = panel_diebold_mariano(pd.concat([a, c], ignore_index=True), "MIDAS", "seasonal")
            dm_rows.append({"horizon_k": k, **r.as_dict()})
        except Exception as exc:  # noqa: BLE001
            logger.warning("DM для k=%d пропущен: %s", k, exc)
    if dm_rows:
        dm = pd.DataFrame(dm_rows)
        dm.to_csv(out_dir / "dm_tests.csv", index=False, encoding="utf-8-sig")
        sig = dm[dm.p_value < 0.10]
        logger.info("Значимых отсечек (p<0,10): %d из %d", len(sig), len(dm))
        if len(sig):
            logger.info("\n%s", sig[["horizon_k", "statistic", "p_value", "better"]]
                        .round(4).to_string(index=False))

    # ------------------------------------------------------- метаданные
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "gross_harvest_kt",
        "target_unit": "тыс. т",
        "identity": "G = Y * rho * S / 10",
        "source": args.source,
        "source_label": source_label,
        "hf_variables": hf_vars,
        "n_periods": hf.K,
        "lf_controls": LF_CONTROLS + ["linear_trend"],
        "scheme": args.scheme,
        "area_source": args.area_source,
        "train_start": train_start,
        "test_years": test_years,
        "best_horizon_k": best_k,
        "n_obs": int(len(aligned)),
        "n_regions": int(aligned["region"].nunique()),
        "metrics_best": metrics[metrics.horizon_k == best_k].to_dict("records"),
    }
    with open(cfg.path("models_dir") / "harvest_model.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    logger.info("Готово. Артефакты: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
