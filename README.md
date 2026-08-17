# AgroForecast

**Спутниковый наукастинг валового сбора зерновых по субъектам РФ.**
Модель смешанных частот (MIDAS) на внутрисезонных композитах VIIRS как
инструмент опережающего мониторинга продовольственной безопасности.

| Показатель | Значение |
|---|---|
| RMSE валового сбора | **434 тыс. т** (R² 0,973) |
| Ошибка по стране | **6,8 %** |
| AUC раннего предупреждения | **0,79** с конца июля |
| Опережение публикации Росстата | **≈ 4 месяца** |

```bash
python nowcast.py --date 2025-07-20 --national
```

**Полный паспорт продукта со всеми числами и структурой — [`PROJECT.md`](PROJECT.md).**
Итоговый отчёт — `results/reports/FINAL_REPORT.md`, интерактивная сводка —
`results/reports/dashboard.html`.

---

## Быстрый старт

```bash
cd C:\AgroForecast
python -m pip install -r requirements.txt

python scripts/00_cleanup.py              # план очистки (диск не меняется)
python scripts/00_cleanup.py --apply      # удалить прежнюю реализацию

python scripts/run_all.py                 # полный прогон
python scripts/05_predict.py --year 2024  # проверочный прогноз

python scripts/07_midas_nowcast.py --bootstrap 200   # MIDAS-наукастинг
python tests/test_midas.py                # 7 проверок MIDAS
python tests/test_diebold_mariano.py      # 8 проверок теста ДМ
python tests/test_harmonization.py        # 6 проверок гармонизации
```

`00_cleanup.py` перед любым удалением проверяет комплектность первичных
данных (ERA5, `ru.json`, книги Росстата) и отказывается работать, если
чего-то не хватает. Каталог `data/` не может попасть в план удаления.

Итоговый отчёт: `results/reports/final_report.md`.

## Структура

```
C:\AgroForecast\
├── configs/config.yaml          единственное место с путями и параметрами
├── data/                        ПЕРВИЧНЫЕ ДАННЫЕ — не изменяются пайплайном
│   ├── ERA5/                    data_stream_moda.nc
│   ├── Rosstat/                 книги Excel Росстата
│   └── boundaries/              ru.json (83 полигона)
├── src/
│   ├── data/       regions.py, rosstat.py, era5.py, ndvi.py
│   ├── features/   climate.py, agro.py, leakage.py, highfreq.py, harmonization.py
│   ├── models/     registry.py, experiment.py, midas.py
│   ├── validation/ qc.py, temporal.py, diebold_mariano.py
│   └── utils/      config.py, logging_utils.py
├── scripts/
│   ├── 00_cleanup.py            удаление прежней реализации (с проверками)
│   ├── 01_extract_era5.py       ERA5 → помесячные средние по субъектам
│   ├── 02_extract_rosstat.py    панели Росстата
│   ├── 03_build_dataset.py      признаки, датасет, QC
│   ├── 04_train_validate.py     временная валидация, выбор модели
│   ├── 05_predict.py            прогноз на заданный год
│   ├── 06_report.py             итоговый отчёт
│   ├── 07_midas_nowcast.py      MIDAS: многогоризонтный наукастинг, H1–H3
│   ├── 08_extract_ndvi.py       приём выгрузки NDVI (MODIS / VIIRS)
│   ├── 09_harmonize_sensors.py  межсенсорная гармонизация MODIS↔VIIRS (H4)
│   └── run_all.py               полный прогон
├── gee/                         скрипты для Google Earth Engine
├── tests/                       модульные тесты (MIDAS, Диболд–Мариано)
├── results/
│   ├── raw_processed/  промежуточные таблицы, region_mapping.csv
│   ├── features/       training_dataset.csv, таблица утечек, QC
│   ├── models/         сравнение моделей, метрики, важность признаков
│   ├── predictions/    прогнозы
│   └── reports/        final_report.md
├── models/             final_model.cbm | .joblib, model_metadata.json
├── logs/               agroforecast.log
├── docs/               DATA_INVENTORY.md, HANDOFF.md
└── archive_old_pipeline/   прежние скрипты и результаты (не используются)
```

## Данные

| Роль | Источник |
|---|---|
| Целевая переменная | Росстат, «Регионы России», разд. 13, табл. **13.11** — урожайность зерновых и зернобобовых (в весе после доработки), хозяйства всех категорий, **2000–2024** |
| Расширение на 2025 | ГМЦ Росстата, `Val1_2025_v2.xlsx`, лист `120(1000001)` |
| Посевные площади | табл. 13.5 (зерновые) и 13.4 (все культуры) |
| Валовой сбор | табл. 13.10 — **только для контроля, в признаки не входит** |
| Удобрения | табл. 13.21.1 (минеральные) и 13.21.2 (органические) |
| Климат | ERA5-Land, месячные средние: `t2m`, `swvl1`, `ssrd`, `tp`, `pev` |
| Геометрия | `data/boundaries/ru.json`, 83 субъекта |

Подробный аудит всех найденных файлов — `docs/DATA_INVENTORY.md`.

## Контроль утечки данных

Модель прогнозирует урожайность года *t* на горизонте **конца вегетационного
сезона** (начало октября *t*) — раньше, чем Росстат публикует официальную
статистику (декабрь *t* предварительно, март *t+1* окончательно).

Запрещены к использованию и жёстко блокируются функцией
`assert_no_forbidden_features`:

* `gross_harvest_kt` — валовой сбор года *t* (прямая утечка: урожайность =
  сбор / убранная площадь);
* убранная площадь и оперативные сводки («Намолочено с 1 га», «Убрано площадей»);
* `sown_area_grain_kha`, `sown_area_total_kha` — окончательные итоги по
  посевам года *t* публикуются после уборки;
* `fert_mineral_kg_ha`, `fert_organic_t_ha` за год *t* — публикуются 6 марта *t+1*;
* `year` как числовой признак.

Полный реестр с датами доступности: `src/features/leakage.py` →
`results/features/feature_leakage_table.csv`.

## Валидация

Только **временная**, расширяющимся окном (rolling origin); случайное
`train_test_split` не применяется:

```
обучение 2001–2019 → тест 2020
обучение 2001–2020 → тест 2021
обучение 2001–2021 → тест 2022
обучение 2001–2022 → тест 2023
обучение 2001–2023 → тест 2024
```

Метрики: MAE, RMSE, R², Bias, MAPE — в целом, по годам, по регионам и
отдельно по **Республике Хакасия** (`validation.focus_region`).

Сравниваются 4 набора признаков (`baseline_historical`, `climate_only`,
`climate_plus_history`, `climate_history_agro`) × 3 алгоритма (CatBoost,
RandomForest, HistGradientBoosting) плюс два эталона: среднее по региону
и урожайность прошлого года.

## Настройка

Все параметры — в `configs/config.yaml`. Захардкоженных путей и регионов в
коде нет. Часто меняемое:

| Параметр | Смысл |
|---|---|
| `features.season_months` | месяцы вегетационного сезона; сократите (напр. `[4,5,6,7]`) для внутрисезонного прогноза |
| `validation.test_years` | тестовые годы временной валидации |
| `validation.focus_region` | регион для детального разбора |
| `models.*` | гиперпараметры алгоритмов |

## Требования

Python ≥ 3.10. Зависимости — `requirements.txt`.
Для шага 1 нужен доступ к файлу ERA5 (≈ 700 МБ) и ~4 ГБ оперативной памяти.

## MIDAS-наукастинг

`scripts/07_midas_nowcast.py` проверяет гипотезы H1–H3 на многогоризонтном
дизайне: для каждой отсечки сезона k = 1…K сравниваются MIDAS (весовые схемы
`beta`, `almon`, `umidas`), эталон на **сезонно усреднённых** тех же данных,
чистая авторегрессия и наивный прогноз.

Ключ дизайна: эталон `seasonal_within` — это тот же код при K = 1 на средних,
поэтому сравнение изолирует ровно один эффект — отказ от временнóго
агрегирования, а не различие реализаций.

Модуль `src/features/highfreq.py` не знает, какой сенсор дал данные: замена
месячного ERA5 на 16-дневные композиты MODIS сводится к смене аргументов
`variables` и `period_values`.

Результаты на данных проекта — `docs/MIDAS_RESULTS.md`.
Связь репозитория с темой научной работы — `docs/RESEARCH_PLAN.md`.
Пошаговый план дальнейших действий — `docs/NEXT_STEPS.md`.

### Переключение источника

```bash
python scripts/07_midas_nowcast.py --source era5   # месячный ERA5, K = 6
python scripts/07_midas_nowcast.py --source ndvi   # 16-дневный MODIS, K = 13
python scripts/07_midas_nowcast.py --source both   # совместная спецификация
```

Готовое к загрузке в Earth Engine: `data/boundaries/gee/ru_regions_shapefile.zip`
(те же 83 полигона, что использованы для ERA5) и `gee/export_modis_ndvi.js`.

## Лицензия

| Что | Лицензия | Файл |
|---|---|---|
| Исходный код (`src/`, `scripts/`, `gee/`, `tests/`, `configs/`, `nowcast.py`) | MIT | [`LICENSE`](LICENSE) |
| Производные данные и отчёты (`results/`) | CC BY 4.0 | [`LICENSE-DATA`](LICENSE-DATA) |
| Первичные данные | режим издателей, см. таблицу | [`LICENSE-DATA`](LICENSE-DATA) |

Автор не является правообладателем первичных данных. Спутниковые продукты
предоставлены NASA LP DAAC, климатический реанализ — Copernicus Climate
Change Service, статистика — Росстатом.

Contains modified Copernicus Climate Change Service information 2026.
Neither the European Commission nor ECMWF is responsible for any use that
may be made of the Copernicus information or data it contains.

### Цитирование

При использовании кода или результатов:

> Овчинников А. AgroForecast: конвейер внутрисезонной оценки валового
> сбора зерновых по субъектам Российской Федерации. 2026.
> URL: https://github.com/arseniy-ovchinnikov/AgroForecast
