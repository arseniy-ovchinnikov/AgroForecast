/**
 * AgroForecast — выгрузка VIIRS NDVI/EVI по субъектам РФ.
 *
 * Продукт:  NASA/VIIRS/002/VNP13A1 (Suomi NPP, 500 м)
 *           Внимание: несмотря на название «16-Day», коллекция обновляется
 *           каждые 8 дней (скользящее 16-дневное окно). За сезон
 *           апрель–сентябрь это ~23 композита против ~12 у MOD13Q1.
 * Период:   2012-01-17 … настоящее время.
 * Маска:    ESA WorldCover v200, класс 40 «Cropland» — та же, что для MODIS.
 * Границы:  ассет ru_regions (тот же файл, что для ERA5 и MODIS).
 *
 * Зачем нужен: годы 2012–2025 перекрываются с MOD13Q1, и на этом перекрытии
 * оценивается межсенсорная кросс-калибровка (гипотеза H4).
 *
 * Запуск — как у export_modis_ndvi.js: подставить REGIONS, Run, Tasks.
 */

var REGIONS = ee.FeatureCollection('projects/ВАШ_ПРОЕКТ/assets/ru_regions_grain');

var YEAR_START   = 2012;   // первый полный год VNP13A1
var YEAR_END     = 2025;
var MONTH_START  = 4;
var MONTH_END    = 9;
var SCALE        = 500;    // native VNP13A1
var DRIVE_FOLDER = 'AgroForecast_NDVI';

var CROPLAND = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40);

// pixel_reliability: 0 = Excellent, 1 = Good. Оставляем 0 и 1 —
// смысловой аналог фильтра SummaryQA <= 1 у MOD13Q1, чтобы строгость
// отбора у двух сенсоров была сопоставимой.
function maskQuality(img) {
  var rel = img.select('pixel_reliability');
  // ВНИМАНИЕ: в отличие от MOD13Q1, коллекция NASA/VIIRS/002/VNP13A1
  // в Earth Engine отдаёт индексы УЖЕ в диапазоне [-1; 1]. Повторное
  // умножение на 0,0001 даёт значения порядка 6e-5 — формально «непустые»,
  // но физически бессмысленные. Масштаб здесь не применяется.
  return img.select(['NDVI', 'EVI'])
            .updateMask(rel.lte(1))
            .updateMask(CROPLAND)
            .copyProperties(img, ['system:time_start']);
}

for (var year = YEAR_START; year <= YEAR_END; year++) {
  var start = ee.Date.fromYMD(year, MONTH_START, 1);
  var end   = ee.Date.fromYMD(year, MONTH_END, 1).advance(1, 'month');

  var perComposite = ee.ImageCollection('NASA/VIIRS/002/VNP13A1')
      .filterDate(start, end)
      .map(maskQuality)
      .map(function (img) {
        var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
        var stats = img.reduceRegions({
          collection: REGIONS,
          // sharedInputs: TRUE — см. пояснение в export_modis_ndvi.js.
          reducer: ee.Reducer.mean().combine({
            reducer2: ee.Reducer.count(), sharedInputs: true
          }),
          scale: SCALE,
          // См. комментарий в export_modis_ndvi.js: без crs синусоидальная
          // проекция продукта роняет задачу на восточных субъектах.
          crs: 'EPSG:4326',
          tileScale: 4
        });
        return stats.map(function (f) {
          return ee.Feature(null, {
            reg_id:   f.get('reg_id'),
            part:     f.get('part'),
            date:     date,
            ndvi:     f.get('NDVI_mean'),
            evi:      f.get('EVI_mean'),
            n_pixels: f.get('NDVI_count')
          });
        });
      }).flatten();

  Export.table.toDrive({
    collection:  perComposite,
    description: 'viirs_' + year,
    folder:      DRIVE_FOLDER,
    fileNamePrefix: 'viirs_' + year,
    fileFormat:  'CSV',
    selectors:   ['reg_id', 'part', 'date', 'ndvi', 'evi', 'n_pixels']
  });
}

print('Создано задач экспорта VIIRS:', YEAR_END - YEAR_START + 1);
print('Перекрытие с MOD13Q1:', YEAR_START + '–' + YEAR_END,
      '=', YEAR_END - YEAR_START + 1, 'сезонов');
