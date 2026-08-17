/**
 * AgroForecast — выгрузка 16-дневных NDVI/EVI по субъектам РФ.
 *
 * Продукт:  MODIS/061/MOD13Q1 (Terra, 250 м, 16 дней)
 * Маска:    ESA WorldCover v200, класс 40 «Cropland»
 * Границы:  ru_regions (83 субъекта, атрибут reg_id) — тот же файл, что
 *           использован для агрегирования ERA5, поэтому геометрия
 *           климатического и спутникового каналов идентична.
 *
 * Результат: по одному CSV на год в папке Google Drive «AgroForecast_NDVI»
 *            с колонками reg_id, date, ndvi, evi, n_pixels.
 *
 * ── КАК ЗАПУСТИТЬ ──────────────────────────────────────────────────────────
 * 1. Assets → NEW → Shape files → загрузить ru_regions_grain.zip,
 *    Asset ID назвать ru_regions_grain (76 субъектов с урожайностью,
 *    разрезанные на 343 части ради устойчивости проекции).
 * 2. Подставить свой путь в REGIONS ниже (кнопка «Import» рядом с ассетом
 *    подставит его автоматически).
 * 3. Run → вкладка Tasks → запустить все задачи (по одной на год).
 * ───────────────────────────────────────────────────────────────────────────
 */

// ============ ПАРАМЕТРЫ ====================================================
var REGIONS = ee.FeatureCollection('projects/ВАШ_ПРОЕКТ/assets/ru_regions_grain');

var YEAR_START   = 2000;   // MOD13Q1 начинается 18.02.2000
var YEAR_END     = 2025;
var MONTH_START  = 4;      // апрель — начало вегетационного сезона
var MONTH_END    = 9;      // сентябрь
var SCALE        = 250;    // native MOD13Q1
var DRIVE_FOLDER = 'AgroForecast_NDVI';

// Маска пашни. Применяется одна и та же ко всем годам — ограничение,
// которое следует оговорить в тексте работы.
var CROPLAND = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40);

// ============ КАЧЕСТВО ПИКСЕЛЕЙ ============================================
// SummaryQA: 0 = good data, 1 = marginal, 2 = snow/ice, 3 = cloudy.
// Оставляем 0 и 1 — стандартная практика для агромониторинга; более строгий
// фильтр на севере оставляет слишком мало наблюдений.
function maskQuality(img) {
  var qa = img.select('SummaryQA');
  var good = qa.lte(1);
  return img.select(['NDVI', 'EVI'])
            .multiply(0.0001)            // масштаб MOD13Q1
            .updateMask(good)
            .updateMask(CROPLAND)
            .copyProperties(img, ['system:time_start']);
}

// ============ ВЫГРУЗКА ПО ГОДАМ ============================================
for (var year = YEAR_START; year <= YEAR_END; year++) {
  var start = ee.Date.fromYMD(year, MONTH_START, 1);
  var end   = ee.Date.fromYMD(year, MONTH_END, 1).advance(1, 'month');

  var collection = ee.ImageCollection('MODIS/061/MOD13Q1')
      .filterDate(start, end)
      .map(maskQuality);

  // Для каждого композита — среднее по пашне внутри каждого субъекта
  var perComposite = collection.map(function (img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var stats = img.reduceRegions({
      collection: REGIONS,
      // sharedInputs: TRUE — обязательно. При false комбинированный редуктор
      // получает два РАЗНЫХ входа (среднее берётся от NDVI, счёт от EVI),
      // и выходные свойства называются просто mean/count, без префикса
      // канала: обращение к NDVI_mean возвращает null, а выгрузка выходит
      // пустой. При true редуктор с одним входом применяется к каждому
      // каналу отдельно и даёт NDVI_mean, NDVI_count, EVI_mean, EVI_count.
      reducer: ee.Reducer.mean().combine({
        reducer2: ee.Reducer.count(), sharedInputs: true
      }),
      scale: SCALE,
      // КРИТИЧНО. Без crs расчёт идёт в синусоидальной проекции продукта
      // (SR-ORG:6974), и границы крупных восточных субъектов не переводятся
      // в неё: угол их прямоугольной рамки выходит за область определения
      // проекции, GEE падает с «Unable to transform edge». Явное указание
      // EPSG:4326 переносит редукцию в географические координаты и снимает
      // проблему полностью.
      crs: 'EPSG:4326',
      tileScale: 4
    });
    return stats.map(function (f) {
      return ee.Feature(null, {
        reg_id:    f.get('reg_id'),
        part:      f.get('part'),
        date:      date,
        ndvi:      f.get('NDVI_mean'),
        evi:       f.get('EVI_mean'),
        n_pixels:  f.get('NDVI_count')
      });
    });
  }).flatten();

  Export.table.toDrive({
    collection:  perComposite,
    description: 'ndvi_' + year,
    folder:      DRIVE_FOLDER,
    fileNamePrefix: 'ndvi_' + year,
    fileFormat:  'CSV',
    selectors:   ['reg_id', 'part', 'date', 'ndvi', 'evi', 'n_pixels']
  });
}

print('Создано задач экспорта:', YEAR_END - YEAR_START + 1);
print('Проверка границ — субъектов в ассете:', REGIONS.size());
