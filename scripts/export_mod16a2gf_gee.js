// Run in the Google Earth Engine Code Editor, then start the Tasks manually.
// Export raw ET, PET and ET_QC with original masks/codes; NO 0.1 scaling here.
// ET/PET are period TOTALS, not daily rates. Keep the 5/6-day final period.
// MOD16 PET is not automatically interchangeable with AWY reference ET0.
// https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF
var datasetId = 'MODIS/061/MOD16A2GF';
var years = [2019, 2020, 2021, 2022];
// Tangnaihai headwaters plus a buffer. Do not use this rectangle as a watershed.
var region = ee.Geometry.Rectangle([95.4, 32.0, 104.0, 37.0], null, false);
var collection = ee.ImageCollection(datasetId).filterBounds(region);
var projection = collection.first().select('ET').projection().getInfo();
var inventory = [];

years.forEach(function(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var yearly = collection.filterDate(start, end).sort('system:time_start');
  var dates = yearly.aggregate_array('system:time_start').getInfo();
  // Stop instead of silently exporting incomplete or duplicate years.
  if (dates.length !== 46) {
    throw new Error('Expected 46 periods for ' + year + ', got ' + dates.length);
  }
  dates.forEach(function(milliseconds, index) {
    var expected = Date.UTC(year, 0, 1) + index * 8 * 86400000;
    if (milliseconds !== expected) {
      throw new Error('Unexpected period date for ' + year + ', index ' + index);
    }
    var periodDays = Math.min(8, (Date.UTC(year + 1, 0, 1) - milliseconds) / 86400000);
    inventory.push(ee.Feature(null, {
      dataset: datasetId,
      year: year,
      period_start: new Date(milliseconds).toISOString().slice(0, 10),
      days: periodDays,
      exported_value: 'raw; ET and PET require scale 0.1 ONCE locally',
      nodata: -32768,
      band_prefix: new Date(milliseconds).toISOString().slice(0, 10).replace(/-/g, '')
    }));
  });
  var stack = yearly.map(function(image) {
    return image.select(['ET', 'PET', 'ET_QC']).toInt16()
      .set('system:index', image.date().format('YYYYMMdd'));
  }).toBands();
  // All bands use Int16 so raw ET/PET and QC export together without type errors.
  // -32768 is distinct from the documented valid raw minimum -32767.
  Export.image.toDrive({
    image: stack.clip(region).unmask(-32768, false),
    description: 'MOD16A2GF061_raw_' + year,
    folder: 'EcologyHydro_MOD16A2GF061',
    fileNamePrefix: 'MOD16A2GF061_raw_' + year,
    region: region,
    crs: projection.crs,
    crsTransform: projection.transform,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -32768}
  });
});
Export.table.toDrive({
  collection: ee.FeatureCollection(inventory),
  description: 'MOD16A2GF061_period_inventory',
  folder: 'EcologyHydro_MOD16A2GF061',
  fileNamePrefix: 'MOD16A2GF061_period_inventory',
  fileFormat: 'CSV',
  selectors: ['dataset', 'year', 'period_start', 'days', 'exported_value', 'nodata', 'band_prefix']
});
print('Export grid (native MODIS sinusoidal)', projection);
print('Four annual 138-band TIFF tasks plus one period inventory CSV task.');
Map.centerObject(region, 6);
Map.addLayer(region, {}, 'Export extent (not the watershed)');
