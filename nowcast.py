#!/usr/bin/env python3
"""AgroForecast — оперативный наукастинг валового сбора зерна.

Единая точка входа для практического использования. Принимает дату отсечки
и выдаёт прогноз валового сбора по субъектам РФ на текущий сезон, используя
спутниковые композиты, доступные на эту дату.

Примеры
-------
    # Прогноз на 15 июля 2025 г. по всем регионам
    python nowcast.py --date 2025-07-15

    # Один регион
    python nowcast.py --date 2025-07-15 --region "Республика Хакасия"

    # Сводка по стране с оценкой опережения официальной публикации
    python nowcast.py --date 2025-07-15 --national

    # Список доступных регионов и лет
    python nowcast.py --list

Что выдаётся
------------
    predicted_harvest_kt   прогноз валового сбора, тыс. т
    predicted_yield_c_ha   прогноз урожайности, ц/га
    lower_kt / upper_kt    интервал по исторической ошибке модели
    shortfall_risk         сигнал риска существенного недобора
    lead_days              опережение предварительной публикации Росстата
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.ndvi import SENSORS, composite_index
from src.features.highfreq import build_high_freq_panel, standardize
from src.models.midas import PanelMIDAS
from src.models.production import (
    add_rho_forecast,
    combine_to_harvest,
    derive_components,
    shortfall_events,
)
from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.validation.temporal import regression_metrics

LF_CONTROLS = ["yield_lag_1", "yield_roll_mean_3", "yield_roll_mean_5"]

# Предварительные данные Росстата по валовому сбору публикуются в 3-й декаде
# декабря отчётного года (паспорт показателя ЕМИСС, поле «Представляется»).
ROSSTAT_PRELIMINARY_MONTH_DAY = (12, 20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Оперативный наукастинг валового сбора зерна по субъектам РФ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--date", type=str, default=None,
                   help="дата отсечки в формате ГГГГ-ММ-ДД (по умолчанию — сегодня)")
    p.add_argument("--region", type=str, default=None, help="один субъект РФ")
    p.add_argument("--national", action="store_true", help="сводка по стране")
    p.add_argument("--list", action="store_true", help="показать доступные регионы и годы")
    p.add_argument("--source", choices=["viirs", "modis"], default="viirs")
    p.add_argument("--scheme", default="beta", choices=["beta", "almon", "umidas"])
    p.add_argument("--starts", type=int, default=6)
    p.add_argument("--out", type=str, default=None, help="куда сохранить CSV")
    return p.parse_args()


def _load(cfg, source: str, logger):
    ds = pd.read_csv(cfg.path("features_dir") / "training_dataset.csv")
    hf_path = cfg.path("raw_processed") / f"{source}_region_composite.csv"
    if not hf_path.exists():
        raise FileNotFoundError(
            f"Не найдена спутниковая панель {hf_path}.\n"
            f"Выполните: python scripts/08_extract_ndvi.py --sensor {source}"
        )
    hf = pd.read_csv(hf_path)
    panel = add_rho_forecast(derive_components(ds))
    return panel, hf


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", "WARNING")
    logger = get_logger("nowcast")

    try:
        panel, hf_long = _load(cfg, args.source, logger)
    except FileNotFoundError as exc:
        print(f"ОШИБКА: {exc}")
        return 2

    if args.list:
        yrs = sorted(hf_long["year"].unique())
        regs = sorted(hf_long["region"].unique())
        print(f"Источник: {SENSORS[args.source]['label']}")
        print(f"Годы со спутниковыми данными: {yrs[0]}–{yrs[-1]}")
        print(f"Субъектов: {len(regs)}\n")
        for r in regs:
            print(" ", r)
        return 0

    cutoff = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    target_year = cutoff.year
    step = SENSORS[args.source]["composite_days"]
    cutoff_period = int((cutoff.timetuple().tm_yday - 1) // step + 1)

    season = sorted(hf_long.loc[hf_long["year"] == target_year, "period"].unique())
    if not season:
        available = sorted(hf_long["year"].unique())
        print(f"ОШИБКА: нет спутниковых данных за {target_year} год.")
        print(f"Доступные годы: {available[0]}–{available[-1]}")
        return 3

    usable = [p for p in season if p <= cutoff_period]
    if len(usable) < 4:
        print(f"ОШИБКА: на дату {cutoff} доступно лишь {len(usable)} композитов "
              f"(нужно не менее 4). Сезон начинается в апреле.")
        return 4

    # Периоды, общие для целевого года и истории обучения
    hist = hf_long[hf_long["year"] < target_year]
    common = sorted(set(usable) & set(hist["period"].unique()))
    hf_use = hf_long[hf_long["period"].isin(common)]

    need = ["yield_c_ha"] + LF_CONTROLS
    train_panel = panel[(panel["year"] < target_year)].dropna(subset=need)
    target_panel = panel[panel["year"] == target_year].dropna(subset=LF_CONTROLS)
    if target_panel.empty:
        print(f"ОШИБКА: для {target_year} года нет исторических лагов урожайности.")
        return 5

    both = pd.concat([train_panel, target_panel], ignore_index=True)
    tensor, _ = build_high_freq_panel(
        hf_use, both, SENSORS[args.source] and ["ndvi", "evi"], common, period_col="period"
    )
    aligned = (both.merge(tensor.index.assign(_k=1), on=["region", "year"], how="inner")
               .drop(columns="_k").reset_index(drop=True))

    yr = aligned["year"].to_numpy()
    is_train = yr < target_year
    is_target = yr == target_year
    if is_target.sum() == 0:
        print(f"ОШИБКА: ни один субъект не имеет полного набора композитов "
              f"на {cutoff} за {target_year} год.")
        return 6

    train_start = int(yr[is_train].min())
    trend = (yr - train_start).astype(float).reshape(-1, 1)
    Z = np.hstack([aligned[LF_CONTROLS].to_numpy(float), trend])
    y = aligned["yield_c_ha"].to_numpy(float)
    reg = aligned["region"].to_numpy()

    Xtr, Xte = standardize(tensor.X[is_train], tensor.X[is_target])
    scheme = args.scheme if tensor.K >= 4 else "umidas"
    model = PanelMIDAS(scheme=scheme, n_starts=args.starts, seed=42)
    model.fit(Xtr, y[is_train], reg[is_train], Z[is_train])
    y_hat = model.predict(Xte, reg[is_target], Z[is_target])

    tgt = aligned[is_target].reset_index(drop=True)
    area = tgt["sown_area_grain_kha"].fillna(tgt["sown_area_lag_1"]).to_numpy(float)
    rho = tgt["rho_hat"].to_numpy(float)
    g_hat = combine_to_harvest(y_hat, rho, area)

    # Интервал по исторической ошибке модели на обучающей выборке
    y_in = model.predict(Xtr, reg[is_train], Z[is_train])
    resid_rel = np.std((y_in - y[is_train]) / np.clip(y[is_train], 1e-6, None))
    lower, upper = g_hat * (1 - 1.64 * resid_rel), g_hat * (1 + 1.64 * resid_rel)

    # Риск существенного недобора относительно нормы региона
    ev = shortfall_events(panel)[["region", "year", "norm"]]
    norms = ev[ev.year == target_year].set_index("region")["norm"]
    norm_v = tgt["region"].map(norms).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = 1.0 - g_hat / norm_v

    pub = date(target_year, *ROSSTAT_PRELIMINARY_MONTH_DAY)
    lead = (pub - cutoff).days

    out = pd.DataFrame({
        "region": tgt["region"],
        "year": target_year,
        "cutoff_date": cutoff.isoformat(),
        "composites_used": len(common),
        "predicted_yield_c_ha": np.round(y_hat, 2),
        "sown_area_kha": np.round(area, 1),
        "harvest_ratio": np.round(rho, 3),
        "predicted_harvest_kt": np.round(g_hat, 1),
        "lower_kt": np.round(lower, 1),
        "upper_kt": np.round(upper, 1),
        "shortfall_risk": np.round(risk, 3),
        "lead_days": lead,
    })
    if "gross_harvest_kt" in tgt.columns and tgt["gross_harvest_kt"].notna().any():
        out["actual_harvest_kt"] = tgt["gross_harvest_kt"].round(1)
        out["error_%"] = (100 * (out.predicted_harvest_kt / out.actual_harvest_kt - 1)).round(1)

    if args.region:
        out = out[out.region == args.region]
        if out.empty:
            print(f"ОШИБКА: регион «{args.region}» не найден или без данных на эту дату.")
            return 7

    out = out.sort_values("predicted_harvest_kt", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------- вывод
    print("=" * 78)
    print("AgroForecast — наукастинг валового сбора зерна")
    print("=" * 78)
    print(f"Дата отсечки:      {cutoff}")
    print(f"Прогнозируемый год: {target_year}")
    print(f"Источник:          {SENSORS[args.source]['label']}")
    print(f"Композитов учтено: {len(common)} (периоды {common[0]}–{common[-1]})")
    print(f"Обучение:          {train_start}–{target_year - 1}, {int(is_train.sum())} наблюдений")
    print(f"Субъектов в прогнозе: {len(out)}")
    print(f"Опережение предварительной публикации Росстата: {lead} дней")
    print()

    if args.national or not args.region:
        total = float(out.predicted_harvest_kt.sum())
        lo, hi = float(out.lower_kt.sum()), float(out.upper_kt.sum())
        print(f"ИТОГО ПО ВЫБОРКЕ: {total:,.0f} тыс. т  (интервал {lo:,.0f} … {hi:,.0f})"
              .replace(",", " "))
        if "actual_harvest_kt" in out.columns:
            fact = float(out.actual_harvest_kt.sum())
            print(f"Факт:             {fact:,.0f} тыс. т, ошибка {100*(total/fact-1):+.1f} %"
                  .replace(",", " "))
        print()

    if args.national:
        risky = out.nlargest(10, "shortfall_risk")[
            ["region", "predicted_harvest_kt", "shortfall_risk"]]
        print("Наибольший риск недобора относительно нормы региона:")
        print(risky.to_string(index=False))
    else:
        cols = ["region", "predicted_yield_c_ha", "predicted_harvest_kt",
                "lower_kt", "upper_kt", "shortfall_risk"]
        if "error_%" in out.columns:
            cols += ["actual_harvest_kt", "error_%"]
        print(out[cols].head(30).to_string(index=False))
        if len(out) > 30:
            print(f"... ещё {len(out) - 30} субъектов")

    dst = Path(args.out) if args.out else cfg.path("predictions") / (
        f"nowcast_{target_year}_{cutoff.isoformat()}.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\nСохранено: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
