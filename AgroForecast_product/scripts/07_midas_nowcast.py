#!/usr/bin/env python3
"""Шаг 7. Многогоризонтный наукастинг: MIDAS против сезонного агрегата.

Проверяемые гипотезы
--------------------
H1  MIDAS на внутрисезонных индексах точнее вневыборочно, чем те же данные,
    усреднённые по сезону, и чем авторегрессионные эталоны.
H2  Прирост точности нелинеен по моменту отсечки сезона; существует точка
    насыщения, после которой дополнительные периоды почти ничего не дают.
H3  Точность деградирует в поздних сезонах (проверяется по динамике ошибки
    во времени; для спутниковых данных — признак орбитального дрейфа).

Дизайн сравнения
----------------
Ключевой эталон — ``seasonal_within`` (B4): та же оценочная процедура, те же
переменные, то же информационное множество, но регрессоры усреднены по
периодам. Технически это U-MIDAS при K = 1 на средних, то есть буквально
тот же код. Поэтому разница между MIDAS и B4 изолирует РОВНО ОДИН эффект —
отказ от временнóго агрегирования, — а не различие в реализации, наборе
переменных или обработке фиксированных эффектов.

Информационное множество на отсечке k строго ограничено периодами 0…k−1:
усечение выполняется в ``HighFreqPanel.truncate`` до любого оценивания.

Запуск
------
    python scripts/07_midas_nowcast.py
    python scripts/07_midas_nowcast.py --bootstrap 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.highfreq import build_high_freq_panel, standardize
from src.models.midas import PanelMIDAS, bootstrap_weights
from src.utils.config import load_config
from src.utils.logging_utils import StageTimer, get_logger, setup_logging
from src.validation.diebold_mariano import pairwise_dm_matrix
from src.validation.temporal import regression_metrics

# Источники высокочастотных регрессоров. Модель одна и та же; меняется только
# таблица и список переменных — ради этого и написан src/features/highfreq.py.
SOURCES = {
    "era5": {
        "file": "era5_region_month.csv",
        "variables": ["t2m_c", "swvl1_m3m3", "ssrd_mj_m2", "tp_mm", "pev_mm"],
        "period_col": "month",
        "label": "ERA5-Land, месячные средние",
    },
    "ndvi": {
        "file": "ndvi_region_composite.csv",
        "variables": ["ndvi", "evi"],
        "period_col": "period",
        "label": "MODIS MOD13Q1, 16-дневные композиты",
    },
    "viirs": {
        "file": "viirs_region_composite.csv",
        "variables": ["ndvi", "evi"],
        "period_col": "period",
        "label": "VIIRS VNP13A1, 8-дневные композиты",
    },
    "both": {
        "file": None,
        "variables": ["ndvi", "evi", "t2m_c", "swvl1_m3m3", "tp_mm"],
        "period_col": "period",
        "label": "NDVI + ERA5 совместно",
    },
}
LF_CONTROLS = ["yield_lag_1", "yield_roll_mean_3", "yield_roll_mean_5"]
# Линейный тренд включается ВО ВСЕ модели, включая эталоны: урожайность в РФ
# росла с 15,6 ц/га в 2000 г. до 27,9 в 2024 г., и без учёта тренда любая
# модель систематически недооценивает поздние годы. Год известен заранее,
# поэтому утечки нет; в отличие от древесных моделей, линейная спецификация
# способна экстраполировать тренд за пределы обучающей выборки.
TREND_COL = "trend"
SCHEMES = ["beta", "almon", "umidas"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Многогоризонтный наукастинг MIDAS")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="число итераций кластерного бутстрэпа весов (0 — пропустить)")
    p.add_argument("--starts", type=int, default=6, help="число стартов оптимизатора")
    p.add_argument("--no-trend", action="store_true",
                   help="не включать линейный тренд в контроли (для проверки его вклада)")
    p.add_argument("--source", choices=list(SOURCES), default="era5",
                   help="источник высокочастотных регрессоров")
    p.add_argument("--train-start", type=int, default=None,
                   help="первый год обучения (по умолчанию — из конфига)")
    p.add_argument("--test-years", type=str, default=None,
                   help="тестовые годы через запятую или диапазон вида 2017-2025")
    return p.parse_args()


def _parse_years(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def _load_source(cfg, source: str, logger):
    """Возвращает (длинная таблица, список переменных, колонка периода)."""
    spec = SOURCES[source]
    raw_dir = cfg.path("raw_processed")
    if source == "both":
        era5 = pd.read_csv(raw_dir / SOURCES["era5"]["file"])
        ndvi = pd.read_csv(raw_dir / SOURCES["ndvi"]["file"])
        # ERA5 приводится к сетке композитов: месяц → композиты этого месяца.
        era5 = era5.rename(columns={"month": "_month"})
        bridge = pd.DataFrame({"period": range(1, 24)})
        bridge["_month"] = [
            pd.Timestamp("2001-01-01").dayofyear and
            (pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=16 * (p - 1))).month
            for p in bridge["period"]
        ]
        era5 = era5.merge(bridge, on="_month", how="inner").drop(columns="_month")
        long = ndvi.merge(era5, on=["region", "year", "period"], how="inner")
    else:
        path = raw_dir / spec["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден {path}. Для source={source} сначала выполните "
                + ("scripts/08_extract_ndvi.py" if source in ("ndvi", "viirs")
                   else "scripts/01_extract_era5.py")
            )
        long = pd.read_csv(path)
        if spec["period_col"] != "period":
            long = long.rename(columns={spec["period_col"]: "period"})
    logger.info("Источник «%s»: %s, %d строк", source, spec["label"], len(long))
    return long, list(spec["variables"])


def _fit_predict(
    scheme: str,
    Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, Ztr: np.ndarray,
    Xte: np.ndarray, gte: np.ndarray, Zte: np.ndarray,
    starts: int,
) -> np.ndarray:
    model = PanelMIDAS(scheme=scheme, n_starts=starts, seed=42)
    model.fit(Xtr, ytr, gtr, Ztr)
    return model.predict(Xte, gte, Zte)


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("07_midas_nowcast")

    out_dir = cfg.path("models_results") / "midas"
    out_dir = out_dir if args.source == "era5" else out_dir.parent / f"midas_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = cfg["project"]["target"]

    mid_cfg = cfg.get("midas", {}) or {}
    test_years = _parse_years(args.test_years) or list(
        mid_cfg.get("test_years", range(2013, 2026))
    )
    train_start = args.train_start or int(mid_cfg.get("train_start_year", 2005))
    months = list(cfg["features"]["season_months"])

    # ------------------------------------------------------------ загрузка
    ds_path = cfg.path("features_dir") / "training_dataset.csv"
    if not ds_path.exists():
        logger.error("Не найден %s — выполните scripts/run_all.py", ds_path)
        return 2
    try:
        long_hf, hf_variables = _load_source(cfg, args.source, logger)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    dataset = pd.read_csv(ds_path)

    needed = [target, "region", "year"] + LF_CONTROLS
    before = len(dataset)
    dataset = dataset.dropna(subset=needed).reset_index(drop=True)
    logger.info(
        "Датасет: %d → %d строк после требования полноты лагов %s "
        "(лаг-5 недоступен для первых лет ряда)",
        before, len(dataset), LF_CONTROLS,
    )

    with StageTimer(logger, "Сборка высокочастотной панели"):
        available = sorted(long_hf["period"].unique().tolist())
        if args.source == "era5":
            periods = [p for p in months if p in available]
        else:
            periods = available
        logger.info("Периоды сезона: %s (K = %d)", periods, len(periods))
        panel, dropped = build_high_freq_panel(
            long_hf, dataset, hf_variables, periods, period_col="period"
        )
        if len(dropped):
            dropped.to_csv(out_dir / "dropped_rows.csv", index=False, encoding="utf-8-sig")
        aligned = dataset.merge(panel.index.assign(_keep=1), on=["region", "year"], how="inner")
        aligned = aligned.drop(columns="_keep").reset_index(drop=True)
        assert len(aligned) == panel.n_obs, "рассогласование панели и датасета"

    y_all = aligned[target].to_numpy(dtype=float)
    g_all = aligned["region"].to_numpy()
    yr_all = aligned["year"].to_numpy()

    control_names = list(LF_CONTROLS)
    Z_all = aligned[LF_CONTROLS].to_numpy(dtype=float)
    if not args.no_trend:
        trend = (yr_all - train_start).astype(float).reshape(-1, 1)
        Z_all = np.hstack([Z_all, trend])
        control_names.append(TREND_COL)
    logger.info("Низкочастотные контроли: %s", control_names)

    logger.info(
        "Панель для MIDAS: %d наблюдений, %d субъектов, годы %d–%d; "
        "тестовые годы %s (T = %d)",
        len(aligned), aligned["region"].nunique(), yr_all.min(), yr_all.max(),
        f"{min(test_years)}–{max(test_years)}", len(test_years),
    )

    # ------------------------------------------------- прогон по горизонтам
    records: List[Dict[str, object]] = []
    weight_rows: List[Dict[str, object]] = []

    with StageTimer(logger, "Оценивание по всем отсечкам сезона"):
        for k in range(1, panel.K + 1):
            cut = panel.truncate(k)
            label = f"периоды 1–{k} из {panel.K}"
            logger.info("Отсечка k = %d (%s), периодов в модели: %d", k, label, k)

            for test_year in test_years:
                tr = (yr_all >= train_start) & (yr_all < test_year)
                te = yr_all == test_year
                if tr.sum() == 0 or te.sum() == 0:
                    continue
                if len(np.unique(yr_all[tr])) < 5:
                    continue

                Xtr_raw, Xte_raw = cut.X[tr], cut.X[te]
                Xtr, Xte = standardize(Xtr_raw, Xte_raw)
                ytr, yte = y_all[tr], y_all[te]
                gtr, gte = g_all[tr], g_all[te]
                Ztr, Zte = Z_all[tr], Z_all[te]

                preds: Dict[str, np.ndarray] = {}

                # MIDAS во всех весовых схемах. При k < 4 параметрические схемы
                # неидентифицируемы (см. PanelMIDAS) и осознанно пропускаются —
                # соответствующие ячейки в таблицах просто отсутствуют.
                for scheme in SCHEMES:
                    try:
                        preds[f"midas_{scheme}"] = _fit_predict(
                            scheme, Xtr, ytr, gtr, Ztr, Xte, gte, Zte, args.starts
                        )
                    except ValueError as exc:
                        if "неидентифицируем" in str(exc):
                            if test_year == test_years[0]:
                                logger.info("  k=%d: схема «%s» пропущена — %s",
                                            k, scheme, exc)
                            continue
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("MIDAS %s, k=%d, год %d — %s: %s",
                                       scheme, k, test_year, type(exc).__name__, exc)

                # B4: тот же код, но регрессоры усреднены по периодам
                Atr, Ate = Xtr.mean(axis=2, keepdims=True), Xte.mean(axis=2, keepdims=True)
                preds["seasonal_within"] = _fit_predict(
                    "umidas", Atr, ytr, gtr, Ztr, Ate, gte, Zte, 1
                )

                # Только авторегрессия: климата нет вообще
                empty_tr = np.zeros((tr.sum(), 0, 1))
                empty_te = np.zeros((te.sum(), 0, 1))
                preds["ar_only"] = _fit_predict(
                    "umidas", empty_tr, ytr, gtr, Ztr, empty_te, gte, Zte, 1
                )

                # Наивный: урожайность прошлого года
                preds["naive_prev_year"] = aligned.loc[te, "yield_lag_1"].to_numpy(dtype=float)

                for model_id, yhat in preds.items():
                    for reg, truth, pred in zip(g_all[te], yte, yhat):
                        records.append({
                            "horizon_k": k, "horizon_label": label, "model_id": model_id,
                            "region": reg, "year": int(test_year),
                            "y_true": float(truth), "y_pred": float(pred),
                        })

            # Веса на полной выборке (для интерпретации формы отклика)
            if k == panel.K:
                Xstd, = standardize(cut.X)
                for scheme in ("beta", "almon"):
                    try:
                        fit = PanelMIDAS(scheme=scheme, n_starts=args.starts, seed=42).fit(
                            Xstd, y_all, g_all, Z_all,
                            hf_names=hf_variables, lf_names=control_names,
                        )
                        for var, w in fit.weights().items():
                            for j, (period, wj) in enumerate(zip(cut.periods, w)):
                                weight_rows.append({
                                    "scheme": scheme, "variable": var, "period": period,
                                    "lag_index": j, "weight": float(wj),
                                    "beta": fit.beta()[var],
                                })
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Оценка весов (%s) не удалась: %s", scheme, exc)

    predictions = pd.DataFrame(records)
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions.to_csv(out_dir / "nowcast_predictions.csv", index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------- метрики
    metric_rows = []
    for (k, label, model_id), sub in predictions.groupby(
        ["horizon_k", "horizon_label", "model_id"]
    ):
        m = regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        m.update({"horizon_k": k, "horizon_label": label, "model_id": model_id})
        metric_rows.append(m)
    metrics = pd.DataFrame(metric_rows)[
        ["horizon_k", "horizon_label", "model_id", "n", "mae", "rmse", "r2", "bias", "mape"]
    ].sort_values(["horizon_k", "rmse"]).reset_index(drop=True)
    metrics.to_csv(out_dir / "nowcast_metrics.csv", index=False, encoding="utf-8-sig")
    logger.info("Метрики по отсечкам:\n%s", metrics.to_string(index=False))

    # ------------------------------------------- H2: кривая насыщения
    best_per_k = (
        metrics[metrics.model_id.str.startswith("midas_")]
        .sort_values("rmse").groupby("horizon_k").first().reset_index()
    )
    base_k = metrics[metrics.model_id == "ar_only"].groupby("horizon_k")["rmse"].first()
    best_per_k["ss_vs_ar_%"] = [
        round(100 * (1 - r / base_k.loc[k]), 2)
        for k, r in zip(best_per_k.horizon_k, best_per_k.rmse)
    ]
    best_per_k["прирост_к_пред_отсечке_%"] = (
        best_per_k["ss_vs_ar_%"].diff().round(2)
    )
    best_per_k.to_csv(out_dir / "h2_saturation_curve.csv", index=False, encoding="utf-8-sig")
    logger.info("H2 — кривая насыщения:\n%s", best_per_k[
        ["horizon_k", "horizon_label", "model_id", "rmse", "ss_vs_ar_%",
         "прирост_к_пред_отсечке_%"]].to_string(index=False))

    # --------------------------------------------- H1: тесты Диболда–Мариано
    with StageTimer(logger, "Тесты Диболда–Мариано"):
        dm_frames = []
        for k, sub in predictions.groupby("horizon_k"):
            m = pairwise_dm_matrix(sub)
            m.insert(0, "horizon_k", k)
            dm_frames.append(m)
        dm = pd.concat(dm_frames, ignore_index=True)
        dm.to_csv(out_dir / "dm_tests.csv", index=False, encoding="utf-8-sig")

        key = dm[
            (dm.model_a.str.startswith("midas_")) & (dm.model_b == "seasonal_within")
            | (dm.model_b.str.startswith("midas_")) & (dm.model_a == "seasonal_within")
        ]
        logger.info("H1 — MIDAS против сезонного агрегата:\n%s", key[
            ["horizon_k", "model_a", "model_b", "statistic", "p_value", "better"]
        ].round(4).to_string(index=False))

    # ------------------------------------------------ H3: динамика ошибки
    # ВАЖНО. Рост абсолютной ошибки сам по себе НЕ доказывает деградацию
    # источника данных: уровень урожайности растёт, и MAE растёт вместе с ним
    # чисто механически. Поэтому тренд проверяется и для относительной ошибки
    # (|e| / факт), и — главное — для ВСЕХ моделей, включая эталоны. Только
    # РАЗНИЦА трендов между спутниковой и несенсорной моделью может служить
    # свидетельством в пользу H3.
    from scipy import stats

    full = predictions[predictions.horizon_k == panel.K].copy()
    full["rel_error"] = full["abs_error"] / full["y_true"].abs().clip(lower=1e-6)
    by_year = (
        full.groupby(["model_id", "year"])[["abs_error", "rel_error"]].mean().reset_index()
    )
    trend_rows = []
    for model_id, sub in by_year.groupby("model_id"):
        x = sub["year"].to_numpy(dtype=float)
        row: Dict[str, object] = {"model_id": model_id, "n_years": len(x)}
        for col, tag in (("abs_error", "MAE"), ("rel_error", "отн_ошибка")):
            z = sub[col].to_numpy(dtype=float)
            slope = float(np.polyfit(x, z, 1)[0])
            r = float(np.corrcoef(x, z)[0, 1])
            n = len(x)
            t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
            row[f"наклон_{tag}_в_год"] = round(slope, 5)
            row[f"корреляция_{tag}"] = round(r, 4)
            row[f"p_{tag}"] = round(float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))), 4)
        trend_rows.append(row)
    trend = pd.DataFrame(trend_rows).sort_values("model_id")
    trend.to_csv(out_dir / "h3_error_trend.csv", index=False, encoding="utf-8-sig")
    logger.info("H3 — тренд ошибки по годам:\n%s", trend.to_string(index=False))
    by_year.to_csv(out_dir / "h3_error_by_year.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------- веса и бутстрэп
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(
            out_dir / "midas_weights.csv", index=False, encoding="utf-8-sig"
        )

    if args.bootstrap > 0:
        with StageTimer(logger, f"Кластерный бутстрэп весов ({args.bootstrap} итераций)"):
            Xstd, = standardize(panel.X)
            ci = bootstrap_weights(
                Xstd, y_all, g_all, Z_all, scheme="beta",
                n_boot=args.bootstrap, seed=42, n_starts=3,
            )
            rows = []
            for v, var in enumerate(hf_variables):
                for j, period in enumerate(panel.periods):
                    rows.append({
                        "variable": var, "period": period, "lag_index": j,
                        "lower_5%": float(ci["lower"][v, j]),
                        "median": float(ci["median"][v, j]),
                        "upper_95%": float(ci["upper"][v, j]),
                    })
            pd.DataFrame(rows).to_csv(
                out_dir / "midas_weights_bootstrap.csv", index=False, encoding="utf-8-sig"
            )
            logger.info("Бутстрэп: успешных итераций %d из %d", ci["n_ok"], args.bootstrap)

    logger.info("Артефакты MIDAS сохранены в %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
