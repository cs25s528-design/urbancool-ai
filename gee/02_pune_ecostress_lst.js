// UrbanCool AI - ECOSTRESS LST export for Pune
//
// Free Earth Engine workflow:
// 1. Open https://code.earthengine.google.com/
// 2. Paste this script.
// 3. Update START_DATE / END_DATE if needed.
// 4. Run and export the table to Google Drive.
//
// Output columns are intended to be joined later by lon/lat/grid/date.

var START_DATE = '2023-03-01';
var END_DATE = '2023-06-30';
var EXPORT_DESCRIPTION = 'pune_ecostress_lst_samples_2023_hotseason';
var EXPORT_FOLDER = 'UrbanCool_AI';
var EXPORT_SCALE_M = 70;

// Pune bounding box. Replace with a stricter PMC ward boundary asset if available.
var pune = ee.Geometry.Rectangle([73.65, 18.35, 74.10, 18.75], null, false);

Map.centerObject(pune, 10);
Map.addLayer(pune, {color: 'red'}, 'Pune ROI', false);

// The ECOSTRESS L2T LSTE collection contains instantaneous land-surface
// temperature observations. The LST band is represented in Kelvin in Earth
// Engine for this collection, so Celsius = Kelvin - 273.15.
var ecostress = ee.ImageCollection('NASA/ECOSTRESS/L2T_LSTE')
  .filterBounds(pune)
  .filterDate(START_DATE, END_DATE)
  .select(['LST', 'QC']);

function maskGoodPixels(image) {
  var lst = image.select('LST');
  var qc = image.select('QC');

  // Conservative mask: keep physically plausible LST values and remove
  // missing/zero pixels. QC encodings can vary by product version, so this
  // avoids over-filtering while still removing obvious bad data.
  var lstC = lst.subtract(273.15).rename('ecostress_lst_celsius');
  var valid = lstC.gt(5).and(lstC.lt(65)).and(qc.mask());

  return lstC
    .updateMask(valid)
    .copyProperties(image, ['system:time_start'])
    .set('date', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'));
}

var lstImages = ecostress.map(maskGoodPixels);

print('ECOSTRESS image count', lstImages.size());

var medianLst = lstImages.median().clip(pune);
Map.addLayer(
  medianLst,
  {min: 20, max: 50, palette: ['2c7bb6', 'abd9e9', 'ffffbf', 'fdae61', 'd7191c']},
  'Median ECOSTRESS LST C'
);

// Export a regular sample grid. For exact joins with the ML table, replace
// this with the same point FeatureCollection used by the Landsat/Sentinel
// export if you have it uploaded as an Earth Engine asset.
var samples = medianLst.sample({
  region: pune,
  scale: EXPORT_SCALE_M,
  geometries: true,
  tileScale: 4
});

var samplesWithCoords = samples.map(function (feature) {
  var coords = feature.geometry().coordinates();
  return feature.set({
    lon: coords.get(0),
    lat: coords.get(1),
    start_date: START_DATE,
    end_date: END_DATE
  });
});

Export.table.toDrive({
  collection: samplesWithCoords,
  description: EXPORT_DESCRIPTION,
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV'
});

