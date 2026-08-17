/**
 * AgroForecast — проверка каталога перед основной выгрузкой.
 *
 * Отвечает на три вопроса, от которых зависит объём работы:
 *   1. до какой даты доступен архив MOD13Q1 (Terra);
 *   2. доступны ли VIIRS-продукты на NOAA-20/21 (замена уходящему SNPP);
 *   3. каков период перекрытия MODIS и VIIRS.
 *
 * ПОРОГ РЕШЕНИЯ: если перекрытие короче трёх сезонов, гипотеза о межсенсорной
 * гармонизации (H4) непроверяема и переносится в «направления развития».
 *
 * Запуск: вставить в code.earthengine.google.com, Run, читать консоль.
 */

function describe(id, label) {
  var col = ee.ImageCollection(id);
  var dates = col.aggregate_array('system:time_start')
                 .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); });
  print(label + ' [' + id + ']');
  print('   снимков:', col.size());
  print('   первый:', dates.reduce(ee.Reducer.min()));
  print('   последний:', dates.reduce(ee.Reducer.max()));
  print('   каналы:', ee.Image(col.first()).bandNames());
}

// --- 1. MODIS Terra, основной архив ----------------------------------------
describe('MODIS/061/MOD13Q1', '1. MODIS Terra 250 м / 16 дней');

// --- 2. MODIS Aqua (работает дольше Terra) ---------------------------------
describe('MODIS/061/MYD13Q1', '2. MODIS Aqua 250 м / 16 дней');

// --- 3. VIIRS на Suomi NPP (поставка прекращается) -------------------------
describe('NASA/VIIRS/002/VNP13A1', '3. VIIRS SNPP 500 м / 16 дней');

// --- 4. VIIRS на NOAA-20 — вероятный преемник ------------------------------
// Если строка ниже выдаёт ошибку, коллекции в каталоге GEE ещё нет:
// это сам по себе результат, который нужно зафиксировать в работе.
describe('NOAA/VIIRS/001/VJ113A1', '4. VIIRS NOAA-20 (JPSS-1)');

// --- 5. Маска пашни ---------------------------------------------------------
var wc = ee.ImageCollection('ESA/WorldCover/v200').first();
print('5. ESA WorldCover v200 — каналы:', wc.bandNames());
print('   класс 40 = Cropland');

print('ЧТО ЗАПИСАТЬ: последние даты MOD13Q1 и VJ113A1, дата начала VJ113A1,');
print('число полных сезонов их перекрытия.');
