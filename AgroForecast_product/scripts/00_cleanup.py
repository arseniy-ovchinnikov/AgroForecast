#!/usr/bin/env python3
"""Шаг 0. Очистка проекта: удаление прежней реализации и служебного мусора.

Скрипт приводит C:\\AgroForecast к чистому виду: остаются только первичные
данные, новый код, конфигурация и документация.

ГАРАНТИИ БЕЗОПАСНОСТИ
    1. Перед любым удалением проверяется комплектность первичных данных
       (ERA5, ru.json, книги Росстата). Если чего-то нет — скрипт отказывается
       работать и ничего не трогает.
    2. Ни один путь внутри data/ не может попасть в план удаления: это
       проверяется отдельным утверждением непосредственно перед выполнением.
    3. Удаляется только то, что перечислено в явных списках ниже. Незнакомые
       файлы не удаляются, а выводятся в отчёт как «оставлено».
    4. По умолчанию выполняется сухой прогон: печатается план, диск не
       изменяется.

ИСПОЛЬЗОВАНИЕ
    python scripts/00_cleanup.py                       # показать план
    python scripts/00_cleanup.py --apply               # выполнить очистку
    python scripts/00_cleanup.py --apply --archive     # не удалять, а перенести
                                                       # в archive_old_pipeline/
    python scripts/00_cleanup.py --apply --consolidate-sources
                                                       # скопировать оставшиеся
                                                       # книги Росстата в
                                                       # data/Rosstat/_source/
    python scripts/00_cleanup.py --apply --include-external
                                                       # + удалить дубликаты
                                                       # проекта на рабочем столе
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Что удалять (явные списки; всё остальное остаётся нетронутым)
# ---------------------------------------------------------------------------

# Файлы в корне проекта
ROOT_FILES = [
    "AgroForecast_new.zip",       # архив поставки
    "install_agroforecast.py",    # установщик
]

# Каталоги в корне проекта
ROOT_DIRS = [
    "resuts",                     # каталог с опечаткой из прежней версии
]

# Скрипты прежней реализации
OLD_SCRIPTS = [
    "analyze_errors.py",
    "build_era5_features.py",
    "check_era5_results.py",
    "check_era5.py",
    "eda.py",
    "extract_yield_2007.py",
    "extract_yield.py",
    "feature_importance.py",
    "find_yield_years.py",
    "inspect_rosstat.py",
    "inspect_yield.py",
    "merge_dataset.py",
    "prepare_features.py",
    "prepare_yield.py",
    "process_era5.py",
    "test_catboost.py",
    "train_models.py",
    "tune_models.py",
    "00_archive_old_pipeline.py",  # заменён этим скриптом
]

# Файлы прежних результатов в корне results/
OLD_RESULT_FILES = [
    "era5_features.csv",
    "era5_regions_monthly.csv",
    "era5_regions_monthly.parquet",
    "model_dataset.csv",
    "rosstat_yield_clean.csv",
    "rosstat_yield.csv",
    "training_dataset.csv",
]

# Каталоги прежних результатов
OLD_RESULT_DIRS = [
    "catboost",
    "eda",
    "feature_importance",
    "tuning",
]

# Прежние артефакты, лежащие ВНУТРИ каталогов нового пайплайна
OLD_FILES_IN_NEW_DIRS = [
    "results/models/model_predictions.csv",
    "results/models/model_results_by_year.csv",
    "results/models/model_results_summary.csv",
]

# Каталоги нового пайплайна внутри results/ — не удаляются никогда
NEW_RESULT_DIRS = {"raw_processed", "features", "models", "predictions", "reports"}

# Внешние дубликаты проекта (удаляются только с --include-external)
EXTERNAL_DUPLICATES = [
    r"C:\Users\User\Desktop\AgroForecast",
    r"C:\Users\User\Desktop\агро\пайтон",
]

# Каталог с исходными книгами Росстата (для --consolidate-sources)
SOURCE_FOLDER = r"C:\Users\User\Desktop\агро"

# Что НЕ копировать при консолидации источников
CONSOLIDATE_SKIP_SUFFIXES = {".nc"}
CONSOLIDATE_SKIP_NAMES = {
    "Федеральный закон от 29.12.2012 N 273-ФЗ Об образовании в Российской Федерации.rtf",
    "Постановление Правительства РФ от 19.10.2023 № 1738 "
    "(Правила выявления детей, проявивших выдающиеся способности).docx",
}

# ---------------------------------------------------------------------------
# Проверка комплектности первичных данных
# ---------------------------------------------------------------------------
REQUIRED_ROSSTAT = [
    "Раздел 13 - Сельское хозяйство.xlsx",
    "Val1_2025_v2.xlsx",
    "Posev_2025_v2.xlsx",
    "Урожайность сельскохозяйственных культур (в расчёте на убранную площадь).xls",
]
MIN_ERA5_BYTES = 100 * 1024 * 1024
EXPECTED_POLYGONS = 83


def verify_primary_data(cfg, logger) -> List[str]:
    """Проверяет наличие и целостность первичных данных.

    Returns:
        Список проблем. Пустой список — данные комплектны.
    """
    problems: List[str] = []

    # ERA5
    try:
        nc = cfg.first_existing_path("era5_files")
        size = nc.stat().st_size
        if size < MIN_ERA5_BYTES:
            problems.append(f"файл ERA5 подозрительно мал: {nc} ({size / 1e6:.1f} МБ)")
        else:
            logger.info("ERA5: %s (%.0f МБ) — на месте", nc.name, size / 1e6)
    except FileNotFoundError as exc:
        problems.append(str(exc))

    # Границы
    geo = cfg.path("boundaries_geojson")
    if not geo.exists():
        problems.append(f"не найден файл границ: {geo}")
    else:
        try:
            data = json.loads(geo.read_text(encoding="utf-8"))
            n = len(data.get("features", []))
            if n != EXPECTED_POLYGONS:
                problems.append(f"{geo.name}: полигонов {n}, ожидалось {EXPECTED_POLYGONS}")
            else:
                logger.info("Границы: %s — %d полигонов, на месте", geo.name, n)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{geo.name} не читается: {type(exc).__name__}: {exc}")

    # Книги Росстата
    rosstat_dir = cfg.path("rosstat_dir")
    for name in REQUIRED_ROSSTAT:
        path = rosstat_dir / name
        if not path.exists():
            problems.append(f"не найдена книга Росстата: {path}")
    if not problems:
        logger.info("Книги Росстата: все %d файла на месте", len(REQUIRED_ROSSTAT))

    return problems


# ---------------------------------------------------------------------------
# План
# ---------------------------------------------------------------------------
def build_plan(root: Path) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []

    def add(path: Path, reason: str) -> None:
        if path.exists():
            plan.append(
                {
                    "путь": str(path.relative_to(root)) if root in path.parents or path == root
                    else str(path),
                    "тип": "каталог" if path.is_dir() else "файл",
                    "размер_МБ": round(
                        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 3
                    )
                    if path.is_dir()
                    else round(path.stat().st_size / 1e6, 3),
                    "причина": reason,
                    "_abs": path,
                }
            )

    for name in ROOT_FILES:
        add(root / name, "служебный файл поставки")
    for name in ROOT_DIRS:
        add(root / name, "каталог прежней версии")
    for name in OLD_SCRIPTS:
        add(root / "scripts" / name, "скрипт прежнего пайплайна")
    for name in OLD_RESULT_FILES:
        add(root / "results" / name, "результат прежнего пайплайна")
    for name in OLD_RESULT_DIRS:
        add(root / "results" / name, "результаты прежнего пайплайна")
    for rel in OLD_FILES_IN_NEW_DIRS:
        add(root / rel, "прежний артефакт в каталоге нового пайплайна")

    # Кэш байт-кода Python: пересоздаётся автоматически, в репозитории не нужен.
    data_dir = (root / "data").resolve()
    for cache in sorted(root.rglob("__pycache__")):
        if data_dir in cache.resolve().parents:
            continue
        add(cache, "кэш байт-кода Python")

    # Неопознанное в results/ — только сообщаем, не удаляем
    return plan


def report_unknown(root: Path, plan: List[Dict[str, object]], logger) -> None:
    planned = {row["_abs"] for row in plan}
    results = root / "results"
    if not results.exists():
        return
    for entry in sorted(results.iterdir()):
        if entry.name in NEW_RESULT_DIRS or entry in planned:
            continue
        logger.warning(
            "Не опознано как старый артефакт — ОСТАВЛЕНО: results/%s "
            "(при необходимости удалите вручную)",
            entry.name,
        )


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Очистка проекта AgroForecast")
    p.add_argument("--apply", action="store_true", help="выполнить (иначе — только план)")
    p.add_argument("--archive", action="store_true",
                   help="переносить в archive_old_pipeline/ вместо удаления")
    p.add_argument("--consolidate-sources", action="store_true",
                   help=f"скопировать оставшиеся книги из «{SOURCE_FOLDER}» в data/Rosstat/_source/")
    p.add_argument("--include-external", action="store_true",
                   help="также удалить дубликаты проекта вне C:\\AgroForecast")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_dirs()
    setup_logging(cfg.path("logs") / "agroforecast.log", cfg["logging"]["level"])
    logger = get_logger("00_cleanup")
    root = cfg.root

    # --- Шлюз безопасности ---------------------------------------------------
    logger.info("Проверка комплектности первичных данных…")
    problems = verify_primary_data(cfg, logger)
    if problems:
        logger.error("ОТКАЗ: первичные данные неполны, очистка не выполняется.")
        for item in problems:
            logger.error("  • %s", item)
        logger.error("Восстановите данные в data/ и повторите запуск.")
        return 2
    logger.info("Первичные данные комплектны — очистка разрешена.")

    # --- Консолидация источников --------------------------------------------
    if args.consolidate_sources:
        src_dir = Path(SOURCE_FOLDER)
        dst_dir = cfg.path("rosstat_dir") / "_source"
        if not src_dir.exists():
            logger.warning("Каталог источников не найден, пропуск: %s", src_dir)
        else:
            copied = 0
            for item in sorted(src_dir.iterdir()):
                if item.is_dir():
                    continue
                if item.suffix.lower() in CONSOLIDATE_SKIP_SUFFIXES:
                    logger.info("Пропуск (дубликат ERA5): %s", item.name)
                    continue
                if item.name in CONSOLIDATE_SKIP_NAMES:
                    logger.info("Пропуск (не относится к теме): %s", item.name)
                    continue
                target = dst_dir / item.name
                if target.exists():
                    continue
                if args.apply:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                copied += 1
            verb = "скопировано" if args.apply else "будет скопировано"
            logger.info("Консолидация источников: %s файлов — %d → %s",
                        verb, copied, dst_dir)

    # --- План удаления -------------------------------------------------------
    plan = build_plan(root)
    if args.include_external:
        for raw in EXTERNAL_DUPLICATES:
            path = Path(raw)
            if path.exists():
                plan.append(
                    {
                        "путь": str(path),
                        "тип": "каталог" if path.is_dir() else "файл",
                        "размер_МБ": round(
                            sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 3
                        ) if path.is_dir() else round(path.stat().st_size / 1e6, 3),
                        "причина": "внешний дубликат проекта",
                        "_abs": path,
                    }
                )

    report_unknown(root, plan, logger)

    if not plan:
        logger.info("Удалять нечего — проект уже чистый.")
        return 0

    table = pd.DataFrame(plan).drop(columns=["_abs"])
    total = table["размер_МБ"].sum()
    logger.info("План (%d объектов, %.1f МБ):\n%s", len(plan), total,
                table.to_string(index=False))

    if not args.apply:
        logger.info("Сухой прогон. Для выполнения добавьте --apply "
                    "(или --apply --archive, чтобы перенести, а не удалить).")
        return 0

    # --- Жёсткая проверка: data/ не должен фигурировать ----------------------
    data_root = cfg.path("data").resolve()
    for row in plan:
        target = Path(row["_abs"]).resolve()
        if target == data_root or data_root in target.parents:
            logger.error("ОТКАЗ: в плане присутствует путь внутри data/: %s", target)
            return 3

    # --- Выполнение -----------------------------------------------------------
    archive_dir = None
    if args.archive:
        archive_dir = root / "archive_old_pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir.mkdir(parents=True, exist_ok=True)

    done, failed = 0, 0
    for row in plan:
        target = Path(row["_abs"])
        try:
            if archive_dir is not None:
                dst = archive_dir / target.name
                counter = 1
                while dst.exists():
                    dst = archive_dir / f"{target.stem}_{counter}{target.suffix}"
                    counter += 1
                shutil.move(str(target), str(dst))
            elif target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            done += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Не удалось обработать %s — %s: %s", target, type(exc).__name__, exc)
            failed += 1

    verb = "перенесено" if args.archive else "удалено"
    logger.info("Готово: %s объектов — %d, освобождено ~%.1f МБ, ошибок — %d",
                verb, done, total, failed)
    logger.info("Каталог data/ не затронут.")

    if failed:
        return 1

    logger.info("Дальше: python scripts/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
