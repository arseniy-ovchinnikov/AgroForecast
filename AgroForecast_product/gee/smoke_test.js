/**
 * AgroForecast — быстрая проверка ПЕРЕД запуском 40 задач экспорта.
 *
 * Считает статистику для одного композита по трём регионам и печатает
 * СПИСОК СВОЙСТВ полученного объекта. Занимает секунды и показывает,
 * совпадают ли имена свойств с теми, что забирает скрипт выгрузки.
 *
 * Именно этой проверки не хватило в прошлый раз: задачи отработали успешно,
 * но выгрузка вышла пустой, потому что имена свойств не совпали.
 *
 * ЧТО ДОЛЖНО НАПЕЧАТАТЬСЯ:
 *   свойства объекта: [..., EVI_count, EVI_mean, NDVI_count, NDVI_mean, ...]
 * Если вместо NDVI_mean видно просто mean — редуктор собран неверно.
 */

var REGIONS = ee.FeatureCollection('projects/ВАШ_ПРОЕКТ/assets/ru_regions_grain');

var CROPLAND = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40);

var img = ee.ImageCollection('MODIS/061/MOD13Q1')
    .filterDate('2020-07-01', '2020-07-31')
    .first();

var masked = img.select(['NDVI', 'EVI'])
    .multiply(0.0001)
    .updateMask(img.select('SummaryQA').lte(1))
    .updateMask(CROPLAND);

var sample = REGIONS.filter(ee.Filter.inList('reg_id', ['RUKK', 'RUBEL', 'RUALT']));

var stats = masked.reduceRegions({
  collection: sample,
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.count(), sharedInputs: true
  }),
  scale: 250,
  crs: 'EPSG:4326',
  tileScale: 4
});

print('регионов в выборке:', sample.size());
print('дата композита:', ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'));
print('СВОЙСТВА первого объекта:', ee.Feature(stats.first()).propertyNames());
print('первые три объекта:', stats.limit(3));

// Явная проверка: значения не должны быть null
var f = ee.Feature(stats.first());
print('NDVI_mean  =', f.get('NDVI_mean'));
print('NDVI_count =', f.get('NDVI_count'));
print('EVI_mean   =', f.get('EVI_mean'));
print('reg_id     =', f.get('reg_id'), '| part =', f.get('part'));
print('---');
print('ПРОВЕРКА 1 — имена свойств: в списке должно быть NDVI_mean, а не просто mean.');
print('ПРОВЕРКА 2 — масштаб: NDVI_mean должен быть около 0,3–0,8 для пашни в июле.');
print('   ~5000      → масштаб не применён (убрать .multiply не нужно, он уже есть);');
print('   ~0,00005   → масштаб применён дважды (для этой коллекции его быть не должно);');
print('   null       → редуктор собран неверно.');
print('Запускать массовую выгрузку можно ТОЛЬКО если обе проверки пройдены.');
