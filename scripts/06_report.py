#!/usr/bin/env python3
"""Шаг 6. Сборка итогового отчёта results/reports/final_report.md.

Отчёт формируется ТОЛЬКО из артефактов предыдущих шагов: ни одна цифра
не вписывается вручную. Если артефакт отсутствует, соответствующий раздел
явно помечается как недоступный.

Запуск:
    python scripts/06_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _render_markdown(view: pd.DataFrame) -> str:
    """Markdown-таблица без внешних зависимостей.

    pandas.DataFrame.to_markdown требует пакет tabulate; отчёт не должен
    падать из-за отсутствия косметической зависимости.
    """
    columns = [str(c) for c in view.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in view.to_numpy()]
    widths = [
        max(len(columns[i]), *(len(r[i]) for r in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def md_table(df: Optional[pd.DataFrame], max_rows: int = 30, floatfmt: int = 3) -> str:
    if df is None or df.empty:
        return "_Артефакт отсутствует._\n"
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=["float", "float64"]).columns:
        view[col] = view[col].round(floatfmt)
    try:
        return view.to_markdown(index=False) + "\n"
    except ImportError:
        return _render_markdown(view)


def skill_scores(comparison: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Относительное улучшение RMSE над эталонами (skill score).

    На панели «регион × год» бóльшая часть дисперсии — межрегиональная,
    поэтому сводный R² завышает реальную полезность модели. Skill score
    показывает выигрыш относительно конкретного эталона:
        SS = 1 − RMSE_модели / RMSE_эталона.
    """
    if comparison is None or comparison.empty:
        return None
    ref = comparison.set_index(["model", "feature_set"])["rmse"]

    def get(model: str, fs: str) -> Optional[float]:
        try:
            return float(ref.loc[(model, fs)])
        except KeyError:
            return None

    rmse_mean = get("baseline_regional_mean", "baseline")
    rmse_prev = get("baseline_previous_year", "baseline")

    rows = []
    for _, r in comparison.iterrows():
        if str(r["model"]).startswith("baseline_"):
            continue
        row = {
            "model": r["model"],
            "feature_set": r["feature_set"],
            "rmse": round(float(r["rmse"]), 3),
        }
        if rmse_mean:
            row["SS_vs_среднее_региона_%"] = round(100 * (1 - r["rmse"] / rmse_mean), 1)
        if rmse_prev:
            row["SS_vs_прошлый_год_%"] = round(100 * (1 - r["rmse"] / rmse_prev), 1)
        # Вклад климата: тот же алгоритм на history-only против текущего набора
        base = get(str(r["model"]), "baseline_historical")
        if base:
            row["SS_vs_только_история_%"] = round(100 * (1 - r["rmse"] / base), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("06_report")

    raw = cfg.path("raw_processed")
    feat = cfg.path("features_dir")
    mres = cfg.path("models_results")
    models_dir = cfg.path("models_dir")

    dataset = read_csv(feat / "training_dataset.csv")
    comparison = read_csv(mres / "model_comparison.csv")
    by_year = read_csv(mres / "metrics_by_year.csv")
    by_region = read_csv(mres / "metrics_by_region.csv")
    khak = read_csv(mres / "khakassia_by_year.csv")
    khak_m = read_csv(mres / "khakassia_metrics.csv")
    importance = read_csv(mres / "feature_importance.csv")
    leakage = read_csv(feat / "feature_leakage_table.csv")
    mapping = read_csv(raw / "region_mapping.csv")
    qc = read_csv(feat / "qc_report.csv")
    journal = read_csv(feat / "qc_filter_journal.csv")
    dictionary = read_csv(feat / "feature_dictionary.csv")
    coverage = read_csv(raw / "rosstat_coverage.csv")
    crosscheck = read_csv(raw / "rosstat_yield_crosscheck.csv")

    meta = {}
    meta_path = models_dir / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    era5_struct = {}
    era5_path = raw / "era5_structure.json"
    if era5_path.exists():
        era5_struct = json.loads(era5_path.read_text(encoding="utf-8"))

    L = []
    A = L.append

    A("# AgroForecast — прогноз урожайности зерновых и зернобобовых культур по субъектам РФ\n")
    A(f"_Отчёт сформирован автоматически: {datetime.now():%Y-%m-%d %H:%M}_\n")
    A("> Все числовые значения в отчёте получены из артефактов пайплайна "
      "(`results/`), ручной ввод отсутствует.\n")

    # 1
    A("## 1. Источники данных\n")
    A("| Источник | Файл | Показатели | Период | Роль |")
    A("|---|---|---|---|---|")
    A("| Росстат, «Регионы России. Социально-экономические показатели», разд. 13 | "
      "`Раздел 13 - Сельское хозяйство.xlsx` | урожайность зерновых (13.11), посевные площади "
      "зерновых (13.5) и всех культур (13.4), валовой сбор зерна (13.10), внесение минеральных "
      "(13.21.1) и органических (13.21.2) удобрений | 2000–2024 | основной |")
    A("| Росстат, ГМЦ, оперативные бюллетени | `Val1_2025_v2.xlsx`, `Posev_2025_v2.xlsx` | "
      "урожайность и посевные площади зерновых по субъектам | 2025 | расширение панели |")
    A("| ERA5-Land (Copernicus C3S) | `data_stream_moda.nc` | t2m, swvl1, ssrd, tp, pev | "
      f"{era5_struct.get('time_min', '—')} … {era5_struct.get('time_max', '—')} | климат |")
    A("| Границы субъектов (simplemaps) | `data/boundaries/ru.json` | 83 полигона | — | геопривязка |")
    A("| Витрина ЕМИСС | `Урожайность ... (в расчёте на убранную площадь).xls` | урожайность | "
      "2010–2024 | перекрёстная проверка |")
    A("")
    if era5_struct:
        A("**Фактическая структура ERA5** (проверена программно):\n")
        A("```json")
        A(json.dumps(
            {k: era5_struct[k] for k in
             ("variables", "raw_units", "n_time", "time_min", "time_max",
              "n_lat", "n_lon", "lat_range", "lon_range", "lat_step", "lon_step")
             if k in era5_struct},
            ensure_ascii=False, indent=2))
        A("```\n")

    # 2-4
    A("## 2. Период, регионы, объём выборки\n")
    if dataset is not None:
        A(f"* Годы в обучающем датасете: **{int(dataset['year'].min())}–{int(dataset['year'].max())}**")
        A(f"* Субъектов РФ: **{dataset['region'].nunique()}**")
        A(f"* Наблюдений (регион × год): **{len(dataset)}**")
        A(f"* Колонок в датасете: **{dataset.shape[1]}**")
        A("")
        A("Наблюдений по годам:\n")
        per_year = dataset.groupby("year").agg(
            регионов=("region", "nunique"),
            наблюдений=("region", "size"),
            средняя_урожайность=("yield_c_ha", "mean"),
        ).reset_index()
        A(md_table(per_year, max_rows=40, floatfmt=2))
    else:
        A("_Датасет не сформирован._\n")

    A("### Сопоставление регионов Росстат ↔ ERA5\n")
    if mapping is not None:
        A("Сводка по статусам:\n")
        A(md_table(mapping["status"].value_counts().rename_axis("status")
                   .reset_index(name="n")))
        A("\nРегионы, исключённые из обучения:\n")
        A(md_table(mapping[mapping["status"] != "matched"], max_rows=20))
        A("\nПолная таблица: `results/raw_processed/region_mapping.csv`\n")

    # 5
    A("## 3. Целевая переменная\n")
    A(f"* Показатель: **{cfg['project']['crop']}**, {cfg['project']['farm_category']}")
    A(f"* Переменная: `{cfg['project']['target']}`, единица — **{cfg['project']['target_unit']}** "
      "(в расчёте на убранную площадь, в весе после доработки)")
    A("* Источник: Росстат, «Регионы России», табл. 13.11")
    A("* Строки «Российская Федерация», федеральные округа и агрегаты "
      "«область с автономными округами» исключены программно\n")
    if crosscheck is not None and not crosscheck.empty:
        n_bad = int((crosscheck["flag"] == "MISMATCH").sum())
        A(f"**Перекрёстная проверка с витриной ЕМИСС:** сопоставлено "
          f"{len(crosscheck)} наблюдений, расхождений более 5 % — **{n_bad}**.\n")
        if n_bad:
            A(md_table(crosscheck.sort_values("rel_diff", ascending=False)
                       .head(10)[["region", "year", "yield_c_ha", "yield_c_ha_emiss", "rel_diff"]]))

    # 6
    A("## 4. Признаки\n")
    if meta.get("features"):
        A(f"Итоговый набор признаков (`{meta.get('feature_set')}`), всего **{len(meta['features'])}**:\n")
        A("```")
        A("\n".join(meta["features"]))
        A("```\n")
    if dictionary is not None:
        A("Словарь колонок датасета (роль и доля пропусков): "
          "`results/features/feature_dictionary.csv`\n")

    # 7
    A("## 5. Контроль утечки данных (data leakage)\n")
    A("Горизонт прогноза: **конец вегетационного сезона** (начало октября года *t*) — "
      "раньше публикации официальной статистики Росстата (предварительные данные — "
      "3-я декада декабря *t*, окончательные — март *t+1*).\n")
    A(md_table(leakage, max_rows=40))
    A("\nЗапрещённые к использованию колонки жёстко проверяются функцией "
      "`assert_no_forbidden_features` перед каждым обучением.\n")

    # 8
    A("## 6. Контроль качества датасета\n")
    if qc is not None:
        n_err = int((qc["status"] == "ERROR").sum())
        n_warn = int((qc["status"] == "WARNING").sum())
        A(f"Проверок: {len(qc)}; ошибок: **{n_err}**; предупреждений: **{n_warn}**.\n")
        problems = qc[qc["status"] != "ok"]
        A(md_table(problems if not problems.empty else qc.head(10), max_rows=30))
    if journal is not None:
        A("\nЖурнал фильтрации строк (молчаливого удаления нет):\n")
        A(md_table(journal))

    # 9
    A("## 7. Методика моделирования\n")
    A("* Валидация: **расширяющееся окно (rolling origin)** — обучение на годах "
      f"[{cfg['validation']['train_start_year']} … Y−1], тест на году Y. "
      "Случайное разбиение не применяется.")
    A(f"* Тестовые годы: {cfg['validation']['test_years']}")
    A("* Метрики: MAE, RMSE, R², Bias, MAPE.")
    A("* Сравниваются четыре набора признаков "
      "(`baseline_historical`, `climate_only`, `climate_plus_history`, `climate_history_agro`) "
      "и три алгоритма (CatBoost, RandomForest, HistGradientBoosting), "
      "плюс два эталона: среднее по региону и урожайность прошлого года.\n")

    # 10
    A("## 8. Результаты сравнения моделей\n")
    A(md_table(comparison, max_rows=40))
    if meta:
        A(f"\n**Выбранная модель:** `{meta.get('algorithm')}` на наборе "
          f"`{meta.get('feature_set')}`.\n")

    A("### Skill score — выигрыш над эталонами\n")
    A("На панели «регион × год» бóльшая часть дисперсии приходится на различия "
      "**между** субъектами, поэтому сводный R² завышает реальную полезность "
      "модели. Ниже — относительное сокращение RMSE к каждому эталону: "
      "`SS = 1 − RMSE_модели / RMSE_эталона`.\n")
    A(md_table(skill_scores(comparison), max_rows=40))

    A("### Метрики по годам (лучшая модель)\n")
    A(md_table(by_year, max_rows=20))

    A("### Регионы с наибольшей ошибкой\n")
    A(md_table(by_region, max_rows=15))

    A("### Важность признаков (топ-25)\n")
    A(md_table(importance, max_rows=25))

    # 11
    focus = cfg["validation"]["focus_region"]
    A(f"## 9. Результаты по региону: {focus}\n")
    A(md_table(khak, max_rows=30, floatfmt=2))
    A("")
    if khak_m is not None and not khak_m.empty:
        A("Агрегированные метрики:\n")
        A(md_table(khak_m, floatfmt=3))

    # 12
    A("## 10. Ограничения\n")
    A("1. **Крым и г. Севастополь исключены** — в `data/boundaries/ru.json` "
      "(83 полигона, границы до 2014 г.) для них нет геометрии, поэтому "
      "климатические признаки построить нельзя.")
    A("2. **г. Санкт-Петербург** отсутствует в статистике урожайности как "
      "самостоятельный субъект: Росстат включает его в Ленинградскую область "
      "(сноска 4 к табл. 13.11). Полигон ERA5 при этом отдельный — "
      "климат Ленинградской области считается без города.")
    A("3. **Автономные округа.** ХМАО, ЯНАО и НАО ведутся как самостоятельные "
      "субъекты; Тюменской и Архангельской областям сопоставлены строки "
      "«без автономного округа». Непересечение полигонов проверено программно "
      "(`results/raw_processed/boundary_checks.csv`).")
    A("4. **Помесячные удобрения и оперативные сводки не используются**: "
      "региональные ряды по удобрениям в отдельных выгрузках ЕМИСС доступны "
      "только на уровне РФ, а оперативные показатели («Намолочено с 1 га», "
      "«Убрано площадей») начинаются с 2023 г. и создают прямую утечку.")
    A("5. **Агрегирование климата** — простое среднее по всем ячейкам ERA5 "
      "внутри полигона, без взвешивания по площади пашни. Для крупных "
      "северных субъектов это смещает климат в сторону незасеваемых территорий.")
    A("6. Модель обучена на панели «регион × год» и не предназначена для "
      "экстраполяции на субъекты, отсутствовавшие в обучении.")
    A("7. **R² по отдельному региону следует трактовать осторожно.** На "
      f"{len(cfg['validation']['test_years'])} тестовых годах внутрирегиональная "
      "дисперсия факта мала, поэтому знаменатель R² неустойчив и значение "
      "может оказаться отрицательным даже при небольшой MAE. Для регионального "
      "разреза приоритетны MAE, RMSE и Bias.")
    A("8. Урожайность в табл. 13.11 приводится **в весе после доработки**, "
      "тогда как витрина ЕМИСС даёт несколько иной показатель; расхождения "
      "зафиксированы в `results/raw_processed/rosstat_yield_crosscheck.csv` "
      "и не устранялись — используется один последовательный источник.\n")

    # 13
    A("## 11. Возможность прогноза на 2025 и 2026 годы\n")
    if era5_struct:
        A(f"* ERA5 в проекте покрывает **{era5_struct.get('time_min', '—')} … "
          f"{era5_struct.get('time_max', '—')}**.")
    season = cfg["features"]["season_months"]
    A(f"* Модели требуется полный вегетационный сезон, месяцы {season}.")
    if dataset is not None:
        avail = sorted(dataset["year"].unique())
        A(f"* Годы с полным набором признаков в датасете: **{int(min(avail))}–{int(max(avail))}**.")
        for year in (2025, 2026):
            ok = year in set(avail)
            A(f"* Прогноз на {year} год: "
              f"{'**возможен** — все признаки доступны' if ok else '**невозможен** — нет полного набора признаков (климат и/или лаги)'}.")
    A("\nЗапуск прогноза: `python scripts/05_predict.py --year <год>`; "
      "результат — `results/predictions/prediction_<год>.csv`.\n")

    # 14
    A("## 12. Артефакты\n")
    A("| Что | Путь |")
    A("|---|---|")
    A("| Обучающий датасет | `results/features/training_dataset.csv` |")
    A("| Таблица утечек | `results/features/feature_leakage_table.csv` |")
    A("| Отчёт QC | `results/features/qc_report.csv` |")
    A("| Сопоставление регионов | `results/raw_processed/region_mapping.csv` |")
    A("| ERA5 по регионам (помесячно) | `results/raw_processed/era5_region_month.csv` |")
    A("| Панель Росстата | `results/raw_processed/rosstat_panel.csv` |")
    A("| Сравнение моделей | `results/models/model_comparison.csv` |")
    A("| Важность признаков | `results/models/feature_importance.csv` |")
    A("| Все прогнозы валидации | `results/models/predictions_all.csv` |")
    A(f"| Финальная модель | `models/{meta.get('model_file', 'final_model.cbm')}` |")
    A("| Метаданные модели | `models/model_metadata.json` |")
    A("| Прогнозы | `results/predictions/` |")
    A("")

    out_path = cfg.path("reports") / "final_report.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    logger.info("Отчёт сохранён: %s (%d строк)", out_path, len(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
