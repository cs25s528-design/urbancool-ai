// ╔══════════════════════════════════════════════════════════════╗
// ║  01_pune_city_landsat_sentinel_weather_population_export.js  ║
// ║  UrbanCool AI — Pune Urban Heat Data Pipeline  v3.1           ║
// ║  Google Earth Engine — paste at code.earthengine.google.com  ║
// ║                                                                ║
// ║  v3.1 CHANGELOG (Fix 1 of 3 — Error Code 3 memory fix):       ║
// ║    • ML_SAMPLES  50000 → 20000  (lighter sampler load)        ║
// ║    • TILE_SCALE  added to PARAMS (default 4)                  ║
// ║    • tileScale: PARAMS.TILE_SCALE added to BOTH .sample()     ║
// ║      calls (ml_samples + hot_samples)                         ║
// ║    Remaining causes (fixes 2-3, not yet applied):              ║
// ║    • .resample('bilinear') on ml_weather/ml_pop/ml_terrain    ║
// ║      forces full-res caching of those source layers           ║
// ║    • focal_mean(500m) for TPI_500m at 30m is expensive         ║
// ╚══════════════════════════════════════════════════════════════╝
//
//  WHAT THIS SCRIPT PRODUCES
//  ─────────────────────────
//  PRIMARY → Pune_ML_Dataset_CSV_<year>.csv   (Export.table → CSV)
//             One row per sampled 100 m pixel, ready for XGBoost training.
//
//  SECONDARY → GeoTIFFs for Landsat / LST / Terrain  (controlled by PARAMS)
//
//  ML FEATURE STAGES (ablation study design)
//  ──────────────────────────────────────────
//  ┌─ MODEL A — Baseline (satellite + weather + population + terrain) ─────┐
//  │  Spectral (Landsat 30 m) : NDVI_L, NDWI_L, MNDWI_L, NDBI_L,         │
//  │                             EVI_L, SAVI_L, NBI_L                       │
//  │  Albedo (Liang 2001)     : albedo                                       │
//  │  LST                     : lst_celsius, emissivity                      │
//  │  Spectral (Sentinel 10 m): NDVI_S2, NDWI_S2, MNDWI_S2, NDBI_S2,      │
//  │                             EVI_S2, SAVI_S2, NBI_S2                     │
//  │  ERA5 Monthly weather    : air_temp_C, humidity_pct,                    │
//  │                             wind_speed, rainfall_mm                      │
//  │  Population              : pop_density, GHSL_Pop_2020, GPW_PopDensity   │
//  │  Built-up                : BuiltUp_m2                                   │
//  │  Terrain                 : Elevation_m, Slope_deg, Aspect_deg, TPI_500m │
//  │  LULC label              : LULC_ESA  (ESA WorldCover 11 classes)        │
//  └─────────────────────────────────────────────────────────────────────────┘
//  ┌─ MODEL B — Vulnerability-aware (+4 cols, added here in GEE) ─────────────┐
//  │  solar_rad_W_m2   ERA5-Land Daily downwelling solar flux                 │
//  │  ntl_radiance     VIIRS monthly nighttime lights  (nW/cm²/sr)            │
//  │  children_ratio   WorldPop Age/Sex 2020  (0–14 yr fraction)              │
//  │  elderly_ratio    WorldPop Age/Sex 2020  (60+ yr fraction)               │
//  └──────────────────────────────────────────────────────────────────────────┘
//  ┌─ MODEL C — Urban morphology (+6 cols, added in Python via OSMnx) ────────┐
//  │  road_density, building_density, impervious_ratio                        │
//  │  dist_water_m, dist_park_m, dist_road_m                                  │
//  │  → see  src/data/02_add_osm_features.py                                  │
//  └──────────────────────────────────────────────────────────────────────────┘
//
//  Python pipeline (run after this export):
//    src/data/01_schema_align_and_albedo.py  — albedo fallback + null checks
//    src/data/02_add_osm_features.py         — OSM morphology via OSMnx
//    src/data/03_add_ward_join.py            — PMC 173-ward spatial join
// ══════════════════════════════════════════════════════════════════════════════


// ────────────────────────────────────────────────────────────
//  0.  USER PARAMETERS  ← edit before running
// ────────────────────────────────────────────────────────────

var PARAMS = {
    START_DATE: '2023-01-01',   // inclusive
    END_DATE: '2024-01-01',   // exclusive

    // Cloud thresholds
    S2_CLOUD_PROB: 20,   // Sentinel-2 s2cloudless probability (0-100)
    L_MAX_CLOUD: 20,   // Landsat scene-level CLOUD_COVER %

    // Google Drive output folder
    DRIVE_FOLDER: 'GEE_Pune_Export',

    // Output CRS — UTM Zone 43N covers all of Pune
    CRS: 'EPSG:32643',

    // ── ML CSV settings (primary output) ─────────────────────
    ML_SAMPLES: 40000,  // ↓ from 50000 — lighter sampler load (scale back up once fixes 2-3 land)
    ML_SCALE: 100,    // metres — native resolution of pop/GHSL layers
    ML_SEED: 42,     // fixed seed for reproducibility
    TILE_SCALE: 4,      // NEW — subdivides per-tile compute; try 8 if Error 3 still fires

    // ── Optional GeoTIFF exports (secondary output) ──────────
    //  Set false to skip and reduce Tasks panel clutter.
    EXPORT_TIFF: {
        landsat: true,   // 3 seasonal composites (dry/monsoon/annual)
        sentinel: false,  // large files — enable when needed
        lst: true,   // LST seasonal + annual
        weather: false,  // ERA5 raster — usually not needed alongside CSV
        population: false,  // population rasters
        terrain: true    // DEM + slope + aspect + TPI
    }
};


// ────────────────────────────────────────────────────────────
//  1.  STUDY AREA — Pune Municipal Corporation + 2 km buffer
// ────────────────────────────────────────────────────────────

var india_l2 = ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'));
var PMC = india_l2.filter(ee.Filter.eq('ADM2_NAME', 'Pune'));

// 2 km buffer around PMC boundary covers peri-urban fringe
var AOI = PMC.geometry().buffer(2000);

Map.centerObject(AOI, 11);
Map.addLayer(AOI, { color: 'FF0000', fillColor: '00000000' }, '01. Pune AOI');
print('Pune AOI area (km²):', AOI.area().divide(1e6));


// ────────────────────────────────────────────────────────────
//  2.  HELPER FUNCTIONS
// ────────────────────────────────────────────────────────────

// 2a. Landsat Collection 2 QA_PIXEL bitmask cloud mask
function maskLandsatSR(img) {
    var qa = img.select('QA_PIXEL');
    var mask = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
        .and(qa.bitwiseAnd(1 << 2).eq(0))   // cirrus
        .and(qa.bitwiseAnd(1 << 3).eq(0))   // cloud
        .and(qa.bitwiseAnd(1 << 4).eq(0));  // cloud shadow
    return img.updateMask(mask);
}

// 2b. Landsat C2 L2 scale factors  →  reflectance 0-1, temperature K
function applyLandsatScale(img) {
    var opt = img.select('SR_B.').multiply(0.0000275).add(-0.2);  // DN → SR
    var thm = img.select('ST_B.*').multiply(0.00341802).add(149.0); // DN → K
    return img.addBands(opt, null, true).addBands(thm, null, true);
}

// 2c. Sentinel-2 cloud mask: s2cloudless probability + SCL band
//     Called in the join-mapped function (see Section 4)
function maskS2(img) {
    var prob = ee.Image(img.get('cloud_prob')).select('probability');
    var scl = img.select('SCL');
    // SCL: 3=shadow, 8=cloud med, 9=cloud high, 10=thin cirrus, 11=snow
    var good = prob.lt(PARAMS.S2_CLOUD_PROB)
        .and(scl.neq(3)).and(scl.neq(8))
        .and(scl.neq(9)).and(scl.neq(10));
    return img.updateMask(good)
        .divide(10000)  // DN → SR reflectance 0-1
        .copyProperties(img, ['system:time_start']);
}

// 2d. Spectral indices — works for both Landsat and Sentinel-2
//     sensor: 'L8L9' or 'S2'
function addIndices(img, sensor) {
    var R, G, NIR, SWIR1, SWIR2;
    if (sensor === 'L8L9') {
        R = img.select('SR_B4'); G = img.select('SR_B3');
        NIR = img.select('SR_B5'); SWIR1 = img.select('SR_B6');
        SWIR2 = img.select('SR_B7');
    } else {
        R = img.select('B4'); G = img.select('B3');
        NIR = img.select('B8'); SWIR1 = img.select('B11');
        SWIR2 = img.select('B12');
    }
    var L = 0.5;
    return img.addBands([
        NIR.subtract(R).divide(NIR.add(R)).rename('NDVI'),
        G.subtract(NIR).divide(G.add(NIR)).rename('NDWI'),
        G.subtract(SWIR1).divide(G.add(SWIR1)).rename('MNDWI'),
        SWIR1.subtract(NIR).divide(SWIR1.add(NIR)).rename('NDBI'),
        NIR.subtract(R).multiply(2.5)
            .divide(NIR.add(R.multiply(6)).subtract(SWIR1.multiply(7.5)).add(1))
            .rename('EVI'),
        NIR.subtract(R).multiply(1 + L)
            .divide(NIR.add(R).add(L)).rename('SAVI'),
        R.multiply(SWIR1).divide(NIR).rename('NBI')
    ]);
}

// 2e. Broadband albedo — Liang (2001) formula for Landsat 8/9 SR
//     Bands must already be in reflectance (0-1) after applyLandsatScale
function computeAlbedo(img) {
    return img.select('SR_B2').multiply(0.356)
        .add(img.select('SR_B4').multiply(0.130))
        .add(img.select('SR_B5').multiply(0.373))
        .add(img.select('SR_B6').multiply(0.085))
        .add(img.select('SR_B7').multiply(0.072))
        .subtract(0.0018)
        .clamp(0, 1)
        .rename('albedo');
}

// 2f. Emissivity-corrected LST from Landsat Band 10 (Kelvin)
//     Formula: T_s = T_B / (1 + A * T_B * ln(ε))
//     A = λ/ρ = 10.895 μm / 14380 μm·K = 7.576e-4 K⁻¹
//     All arithmetic uses ee.Image operations to avoid type errors.
function computeLST(img) {
    var T_B = img.select('ST_B10');                   // K after scaling
    var ndvi = img.select('SR_B5').subtract(img.select('SR_B4'))
        .divide(img.select('SR_B5').add(img.select('SR_B4')));
    var pv = ndvi.max(0.2).min(0.5)                  // clamp → [0.2, 0.5]
        .subtract(0.2).divide(0.3).pow(2);  // fractional veg cover
    var em = pv.multiply(0.004).add(0.986);          // Sobrino emissivity
    var A = 0.000796;                               // λ/ρ [K⁻¹] for L8/L9 B10
    var T_s = T_B.divide(
        T_B.multiply(A).multiply(em.log()).add(1)
    ).subtract(273.15);                     // K → °C
    return img.addBands(T_s.rename('lst_celsius'))
        .addBands(em.rename('emissivity'));
}

// 2g. GeoTIFF export helper
function exportTiff(image, name, scale) {
    Export.image.toDrive({
        image: image.clip(AOI),
        description: name,
        folder: PARAMS.DRIVE_FOLDER,
        region: AOI,
        scale: scale,
        crs: PARAMS.CRS,
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF',
        formatOptions: { cloudOptimized: true }
    });
}


// ────────────────────────────────────────────────────────────
//  3.  LANDSAT 8 + 9  —  C2 T1 L2 Surface Reflectance
// ────────────────────────────────────────────────────────────

var prepL = function (col) {
    return col.filterBounds(AOI)
        .filterDate(PARAMS.START_DATE, PARAMS.END_DATE)
        .filter(ee.Filter.lt('CLOUD_COVER', PARAMS.L_MAX_CLOUD))
        .map(maskLandsatSR)
        .map(applyLandsatScale);
};

var L8 = prepL(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'));
var L9 = prepL(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'));
var landsat = L8.merge(L9).sort('system:time_start');
print('Landsat L8 scenes  :', L8.size());
print('Landsat L9 scenes  :', L9.size());

// Seasonal composites (Indian seasons)
var l_annual = landsat.median();
var l_hot = landsat.filter(ee.Filter.calendarRange(3, 5, 'month')).median();  // MAM hot/dry
var l_monsoon = landsat.filter(ee.Filter.calendarRange(6, 9, 'month')).median();  // JJAS monsoon
var l_winter = landsat.filter(ee.Filter.calendarRange(11, 2, 'month')).median();  // NDJ winter

// Albedo on annual composite
var landsat_albedo = computeAlbedo(l_annual);
Map.addLayer(l_annual.clip(AOI),
    { bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3, gamma: 1.4 }, '02. Landsat Annual RGB');
Map.addLayer(landsat_albedo.clip(AOI),
    { min: 0.05, max: 0.35, palette: ['black', 'gray', 'white'] }, '03. Albedo (Landsat)');

if (PARAMS.EXPORT_TIFF.landsat) {
    var l_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];
    exportTiff(l_annual.select(l_bands), 'Pune_Landsat_Annual_' + PARAMS.START_DATE.slice(0, 4), 30);
    exportTiff(l_hot.select(l_bands), 'Pune_Landsat_Hot_' + PARAMS.START_DATE.slice(0, 4), 30);
    exportTiff(l_monsoon.select(l_bands), 'Pune_Landsat_Monsoon_' + PARAMS.START_DATE.slice(0, 4), 30);
    print('✅ Landsat GeoTIFF exports queued (3)');
}


// ────────────────────────────────────────────────────────────
//  4.  SENTINEL-2  —  Harmonized L2A + s2cloudless
// ────────────────────────────────────────────────────────────

var s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(AOI).filterDate(PARAMS.START_DATE, PARAMS.END_DATE)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50));
var s2_cp = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(AOI).filterDate(PARAMS.START_DATE, PARAMS.END_DATE);

// Join cloud probability onto SR collection
var s2_joined = ee.ImageCollection(
    ee.Join.saveFirst('cloud_prob').apply({
        primary: s2_sr,
        secondary: s2_cp,
        condition: ee.Filter.equals({ leftField: 'system:index', rightField: 'system:index' })
    })
).map(maskS2);

var s2_annual = s2_joined.median();
var s2_hot = s2_joined.filter(ee.Filter.calendarRange(3, 5, 'month')).median();
var s2_monsoon = s2_joined.filter(ee.Filter.calendarRange(6, 9, 'month')).median();
print('Sentinel-2 scenes  :', s2_joined.size());

Map.addLayer(s2_annual.clip(AOI),
    { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3, gamma: 1.4 }, '04. Sentinel-2 Annual RGB');
Map.addLayer(s2_annual.clip(AOI),
    { bands: ['B8', 'B4', 'B3'], min: 0, max: 0.5 }, '05. Sentinel-2 FCC');

if (PARAMS.EXPORT_TIFF.sentinel) {
    var s2b = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
    exportTiff(s2_annual.select(s2b), 'Pune_S2_Annual_' + PARAMS.START_DATE.slice(0, 4), 10);
    print('✅ Sentinel-2 GeoTIFF queued (1)');
}


// ────────────────────────────────────────────────────────────
//  5.  SPECTRAL INDICES (Landsat + Sentinel-2)
// ────────────────────────────────────────────────────────────

var l_idx = addIndices(l_annual, 'L8L9');
var s2_idx = addIndices(s2_annual, 'S2');

Map.addLayer(l_idx.select('NDVI').clip(AOI),
    { min: -0.1, max: 0.8, palette: ['#8B4513', '#F5F5DC', '#2E8B57'] }, '06. NDVI (Landsat)');
Map.addLayer(l_idx.select('NDBI').clip(AOI),
    { min: -0.5, max: 0.5, palette: ['#006400', 'white', '#4B0082'] }, '07. NDBI (Landsat)');
Map.addLayer(l_idx.select('MNDWI').clip(AOI),
    { min: -0.5, max: 0.5, palette: ['#8B0000', 'white', '#00008B'] }, '08. MNDWI (Landsat)');


// ────────────────────────────────────────────────────────────
//  6.  LAND SURFACE TEMPERATURE  —  Emissivity-corrected LST
// ────────────────────────────────────────────────────────────

var lst_stack = landsat.map(computeLST).select(['lst_celsius', 'emissivity']);
var lst_annual = lst_stack.median();
var lst_hot = landsat.filter(ee.Filter.calendarRange(3, 5, 'month'))
    .map(computeLST).median().select(['lst_celsius', 'emissivity']);
var lst_monsoon = landsat.filter(ee.Filter.calendarRange(6, 9, 'month'))
    .map(computeLST).median().select(['lst_celsius', 'emissivity']);

Map.addLayer(lst_annual.select('lst_celsius').clip(AOI),
    { min: 20, max: 48, palette: ['#00BFFF', '#7FFF00', '#FFD700', '#FF8C00', '#FF0000'] },
    '09. LST Annual (°C)');
Map.addLayer(lst_hot.select('lst_celsius').clip(AOI),
    { min: 28, max: 52, palette: ['#00BFFF', '#7FFF00', '#FFD700', '#FF8C00', '#FF0000'] },
    '10. LST Hot Season (MAM, °C)');

if (PARAMS.EXPORT_TIFF.lst) {
    exportTiff(lst_annual, 'Pune_LST_Annual_' + PARAMS.START_DATE.slice(0, 4), 30);
    exportTiff(lst_hot, 'Pune_LST_HotSeason_' + PARAMS.START_DATE.slice(0, 4), 30);
    exportTiff(lst_monsoon, 'Pune_LST_Monsoon_' + PARAMS.START_DATE.slice(0, 4), 30);
    print('✅ LST GeoTIFF exports queued (3)');
}


// ────────────────────────────────────────────────────────────
//  7.  ERA5-LAND MONTHLY  —  Temperature, Precipitation, Wind, RH
//      Column names match final Python schema (no renaming needed)
// ────────────────────────────────────────────────────────────

var era5_monthly = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    .filterBounds(AOI)
    .filterDate(PARAMS.START_DATE, PARAMS.END_DATE);

// air_temp_C: mean, max, min over year (K → °C)
var air_temp_C = era5_monthly.select('temperature_2m')
    .mean().subtract(273.15).rename('air_temp_C');
var air_temp_C_max = era5_monthly.select('temperature_2m')
    .max().subtract(273.15).rename('air_temp_C_max');
var air_temp_C_min = era5_monthly.select('temperature_2m')
    .min().subtract(273.15).rename('air_temp_C_min');

// rainfall_mm: annual total (m/month → mm/year)
var rainfall_mm = era5_monthly.select('total_precipitation_sum')
    .sum().multiply(1000).rename('rainfall_mm');

// wind_speed: annual mean of monthly scalar wind
var wind_speed = era5_monthly.map(function (img) {
    var u = img.select('u_component_of_wind_10m');
    var v = img.select('v_component_of_wind_10m');
    return u.pow(2).add(v.pow(2)).sqrt().copyProperties(img, ['system:time_start']);
}).mean().rename('wind_speed');

// humidity_pct: relative humidity via Magnus approximation (annual mean)
var humidity_pct = era5_monthly.map(function (img) {
    var t = img.select('temperature_2m').subtract(273.15);
    var td = img.select('dewpoint_temperature_2m').subtract(273.15);
    return td.multiply(17.625).divide(td.add(243.04)).exp()
        .divide(t.multiply(17.625).divide(t.add(243.04)).exp())
        .multiply(100)
        .copyProperties(img, ['system:time_start']);
}).mean().rename('humidity_pct');

Map.addLayer(air_temp_C.clip(AOI),
    { min: 22, max: 34, palette: ['#00BFFF', '#ADFF2F', '#FFD700', '#FF4500'] },
    '11. ERA5 Mean Temp (°C)');
Map.addLayer(rainfall_mm.clip(AOI),
    { min: 400, max: 1000, palette: ['#F5F5F5', '#87CEEB', '#1E90FF', '#00008B'] },
    '12. ERA5 Annual Rainfall (mm)');

print('ERA5 monthly images:', era5_monthly.size());
if (PARAMS.EXPORT_TIFF.weather) {
    var wx_stack = air_temp_C.addBands(air_temp_C_max).addBands(air_temp_C_min)
        .addBands(rainfall_mm).addBands(wind_speed).addBands(humidity_pct);
    exportTiff(wx_stack, 'Pune_ERA5_Weather_' + PARAMS.START_DATE.slice(0, 4), 1000);
    print('✅ ERA5 weather GeoTIFF queued (1)');
}


// ────────────────────────────────────────────────────────────
//  8.  ERA5-LAND DAILY  —  Solar Radiation  [MODEL B NEW]
//      band: surface_solar_radiation_downwards_sum  [J/m²/day]
//      → divide by 86400 s/day to get mean daily flux [W/m²]
// ────────────────────────────────────────────────────────────

var era5_daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterBounds(AOI)
    .filterDate(PARAMS.START_DATE, PARAMS.END_DATE)
    .select('surface_solar_radiation_downwards_sum');

var solar_rad_W_m2 = era5_daily.mean()
    .divide(86400)   // J/m²/day → W/m²
    .rename('solar_rad_W_m2');

// Summer solar vs monsoon (important driver for LST in Pune)
var solar_hot = era5_daily.filter(ee.Filter.calendarRange(3, 5, 'month'))
    .mean().divide(86400).rename('solar_hot_W_m2');
var solar_monsoon = era5_daily.filter(ee.Filter.calendarRange(6, 9, 'month'))
    .mean().divide(86400).rename('solar_monsoon_W_m2');

Map.addLayer(solar_rad_W_m2.clip(AOI),
    { min: 180, max: 280, palette: ['#4169E1', '#F0E68C', '#FF8C00', '#FF0000'] },
    '13. Solar Radiation Annual Mean (W/m²)');

print('ERA5 Daily solar images:', era5_daily.size());
print('Solar rad range approx [W/m²]:',
    solar_rad_W_m2.reduceRegion({
        reducer: ee.Reducer.minMax(), geometry: AOI, scale: 5000
    }));


// ────────────────────────────────────────────────────────────
//  9.  VIIRS NIGHTTIME LIGHTS  [MODEL B NEW]
//      NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
//      band: avg_rad  [nanoWatts/cm²/sr]
//      Use VCMSLCFG (stray-light corrected) for better urban quality
// ────────────────────────────────────────────────────────────

var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filterBounds(AOI)
    .filterDate(PARAMS.START_DATE, PARAMS.END_DATE)
    .select('avg_rad');

// median composite — robust to cloud contamination in monthly data
var ntl_radiance = viirs.median().rename('ntl_radiance');

// High NTL = dense human activity, commerce, traffic → anthropogenic heat proxy
Map.addLayer(ntl_radiance.clip(AOI),
    { min: 0, max: 80, palette: ['#000033', '#003399', '#FFD700', '#FFFFFF'] },
    '14. VIIRS Nighttime Lights (nW/cm²/sr)');

print('VIIRS monthly images:', viirs.size());
print('NTL stats (AOI):',
    ntl_radiance.reduceRegion({
        reducer: ee.Reducer.percentile([25, 50, 75, 95]), geometry: AOI, scale: 500
    }));


// ────────────────────────────────────────────────────────────
//  10. POPULATION  —  WorldPop + GHSL + GPW + Age/Sex
// ────────────────────────────────────────────────────────────

// 10a. WorldPop 2020 constrained — 100 m (primary pop density feature)
var pop_density = ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.eq('country', 'IND'))
    .filter(ee.Filter.eq('year', 2020))
    .first()
    .rename('pop_density');

// 10b. GHSL Population 2020 — 100 m
var GHSL_Pop_2020 = ee.ImageCollection('JRC/GHSL/P2023A/GHS_POP')
    .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
    .first().select('population_count').rename('GHSL_Pop_2020');

// 10c. GPW v4.11 — 1 km population density (UN-adjusted)
var GPW_PopDensity = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density')
    .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
    .first().select('population_density').rename('GPW_PopDensity');

// 10d. WorldPop Age/Sex 2020  [MODEL B NEW]
//      Bands: M_0, M_1, M_5, M_10 … M_80  (male, 5-year groups)
//             F_0, F_1, F_5, F_10 … F_80  (female, 5-year groups)
//      Unit: population count per 100 m cell
var ageSex = ee.ImageCollection('WorldPop/GP/100m/pop_age_sex')
    .filter(ee.Filter.eq('country', 'IND'))
    .filter(ee.Filter.eq('year', 2020))
    .first();

// Children 0-14: M_0, M_1, M_5, M_10 + F_0, F_1, F_5, F_10
var children_count = ageSex.select(
    ['M_0', 'M_1', 'M_5', 'M_10', 'F_0', 'F_1', 'F_5', 'F_10']
).reduce(ee.Reducer.sum()).rename('children_count');

// Elderly 60+: M_60 through M_80, F_60 through F_80
var elderly_count = ageSex.select(
    ['M_60', 'M_65', 'M_70', 'M_75', 'M_80', 'F_60', 'F_65', 'F_70', 'F_75', 'F_80']
).reduce(ee.Reducer.sum()).rename('elderly_count');

// Total from age/sex bands (avoid divide-by-zero with small epsilon)
var total_age = ageSex.reduce(ee.Reducer.sum()).rename('total_age');

var children_ratio = children_count.divide(total_age.add(1e-6))
    .clamp(0, 1).rename('children_ratio');
var elderly_ratio = elderly_count.divide(total_age.add(1e-6))
    .clamp(0, 1).rename('elderly_ratio');

Map.addLayer(pop_density.clip(AOI),
    { min: 0, max: 500, palette: ['white', '#FFEDA0', '#FEB24C', '#F03B20', '#BD0026'] },
    '15. WorldPop 2020 (pop/100m cell)');
Map.addLayer(elderly_ratio.clip(AOI),
    { min: 0, max: 0.25, palette: ['white', '#FECC5C', '#FD8D3C', '#E31A1C'] },
    '16. Elderly Ratio (60+)');
Map.addLayer(children_ratio.clip(AOI),
    { min: 0, max: 0.35, palette: ['white', '#74C476', '#31A354', '#006D2C'] },
    '17. Children Ratio (0-14)');

print('WorldPop age/sex band count:', ageSex.bandNames().size());
print('WorldPop total pop (AOI):',
    pop_density.reduceRegion({ reducer: ee.Reducer.sum(), geometry: AOI, scale: 100 }));

if (PARAMS.EXPORT_TIFF.population) {
    var pop_stack = pop_density.addBands(GHSL_Pop_2020).addBands(GPW_PopDensity)
        .addBands(children_ratio).addBands(elderly_ratio);
    exportTiff(pop_stack, 'Pune_Population_Stack_2020', 100);
    print('✅ Population GeoTIFF queued (1)');
}


// ────────────────────────────────────────────────────────────
//  11. TERRAIN  —  SRTM DEM + Slope + Aspect + TPI
// ────────────────────────────────────────────────────────────

var srtm = ee.Image('USGS/SRTMGL1_003');
var Elevation_m = srtm.rename('Elevation_m');
var Slope_deg = ee.Terrain.slope(srtm).rename('Slope_deg');
var Aspect_deg = ee.Terrain.aspect(srtm).rename('Aspect_deg');
// TPI: local elevation minus 500 m neighbourhood mean
var TPI_500m = srtm.subtract(srtm.focal_mean(500, 'circle', 'meters'))
    .rename('TPI_500m');
var Hillshade = ee.Terrain.hillshade(srtm, 315, 45).rename('Hillshade');

Map.addLayer(Elevation_m.clip(AOI),
    { min: 500, max: 1100, palette: ['#3CB371', '#F0E68C', '#A0522D', '#FFFFFF'] }, '18. DEM (m)');
Map.addLayer(Slope_deg.clip(AOI),
    { min: 0, max: 45, palette: ['white', '#F5DEB3', '#D2691E', '#8B0000'] }, '19. Slope (°)');

if (PARAMS.EXPORT_TIFF.terrain) {
    var terrain_stack = Elevation_m.addBands(Slope_deg).addBands(Aspect_deg)
        .addBands(TPI_500m).addBands(Hillshade);
    exportTiff(terrain_stack, 'Pune_SRTM_Terrain', 30);
    print('✅ Terrain GeoTIFF queued (1)');
}


// ────────────────────────────────────────────────────────────
//  12. GHSL BUILT-UP  —  Built surface area m² per 100 m cell
//      Band: built_surface  (P2023A — residential + non-residential)
// ────────────────────────────────────────────────────────────

var BuiltUp_m2 = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S')
    .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
    .first().select('built_surface').rename('BuiltUp_m2');

var BuiltUp_2015 = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S')
    .filter(ee.Filter.date('2015-01-01', '2016-01-01'))
    .first().select('built_surface').rename('BuiltUp_m2_2015');

var builtup_change = BuiltUp_m2.subtract(BuiltUp_2015).rename('BuiltUp_change_2015_2020');

Map.addLayer(BuiltUp_m2.clip(AOI),
    { min: 0, max: 8000, palette: ['white', '#D3D3D3', '#808080', '#2F4F4F'] },
    '20. GHSL Built-Up 2020 (m²/pixel)');
Map.addLayer(builtup_change.clip(AOI),
    { min: -500, max: 2000, palette: ['blue', 'white', 'red'] },
    '21. Built-Up Change 2015-2020 (m²)');


// ────────────────────────────────────────────────────────────
//  13. ESA WORLDCOVER 2021  —  LULC label (11 classes)
//      10: Tree    20: Shrub    30: Grassland  40: Cropland
//      50: Built   60: Bare     70: Snow/Ice   80: Water
//      90: Wetland 95: Mangrove 100: Moss/Lichen
// ────────────────────────────────────────────────────────────

var LULC_ESA = ee.ImageCollection('ESA/WorldCover/v200')
    .filterBounds(AOI)
    .first()
    .select('Map')
    .rename('LULC_ESA');

Map.addLayer(LULC_ESA.clip(AOI),
    {
        min: 10, max: 100, palette: ['006400', 'FFBB22', 'FFFF4C', 'F096FF',
            'FA0000', 'B4B4B4', 'F0F0F0', '0064C8',
            '0096A0', '00CF75', 'FAE6A0']
    },
    '22. ESA WorldCover 2021');


// ════════════════════════════════════════════════════════════
//  14. ML CSV EXPORT  ★ PRIMARY OUTPUT ★
//      Export.table.toDrive → fileFormat: 'CSV'
//      One row per sampled 100 m pixel
//      Column names match final Python schema (no renaming needed)
// ════════════════════════════════════════════════════════════

// ──────────────────────────────────────────────────────────
//  14A. MODEL A FEATURES  (baseline — satellite + weather + pop + terrain)
// ──────────────────────────────────────────────────────────

// TARGET: lst_celsius, emissivity  (from annual LST stack)
var ml_lst = lst_annual.select(['lst_celsius', 'emissivity']);

// Landsat SR bands (reflectance 0-1)
var ml_l_bands = l_annual
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    .rename(['L_Blue', 'L_Green', 'L_Red', 'L_NIR', 'L_SWIR1', 'L_SWIR2']);

// Landsat indices (7 features)
var ml_l_idx = addIndices(l_annual, 'L8L9')
    .select(['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'EVI', 'SAVI', 'NBI'])
    .rename(['NDVI_L', 'NDWI_L', 'MNDWI_L', 'NDBI_L', 'EVI_L', 'SAVI_L', 'NBI_L']);

// Albedo from Landsat (Liang 2001)
var ml_albedo = computeAlbedo(l_annual);

// Sentinel-2 SR bands (reflectance 0-1, already divided by 10000 in Section 4)
var ml_s2_bands = s2_annual
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    .rename(['S2_Blue', 'S2_Green', 'S2_Red', 'S2_NIR', 'S2_SWIR1', 'S2_SWIR2']);

// Sentinel-2 indices (7 features, 10 m → resampled to ML_SCALE in sample())
var ml_s2_idx = addIndices(s2_annual, 'S2')
    .select(['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'EVI', 'SAVI', 'NBI'])
    .rename(['NDVI_S2', 'NDWI_S2', 'MNDWI_S2', 'NDBI_S2', 'EVI_S2', 'SAVI_S2', 'NBI_S2']);

// ERA5 Monthly weather (already computed globally above, re-use)
var ml_weather = air_temp_C
    .addBands(air_temp_C_max)
    .addBands(air_temp_C_min)
    .addBands(rainfall_mm)
    .addBands(wind_speed)
    .addBands(humidity_pct)
    .resample('bilinear');

// Population
var ml_pop = pop_density
    .addBands(GHSL_Pop_2020)
    .addBands(GPW_PopDensity.resample('bilinear'));

// Terrain (already 30 m, bilinear upsample to 100 m)
var ml_terrain = Elevation_m
    .addBands(Slope_deg)
    .addBands(Aspect_deg)
    .addBands(TPI_500m)
    .resample('bilinear');

// ──────────────────────────────────────────────────────────
//  14B. MODEL B FEATURES  (+ solar + NTL + age/sex)
// ──────────────────────────────────────────────────────────

// Solar radiation already computed in Section 8 → reuse
// ntl_radiance already computed in Section 9 → reuse
// children_ratio, elderly_ratio already computed in Section 10 → reuse

var ml_model_b = solar_rad_W_m2
    .addBands(ntl_radiance)
    .addBands(children_ratio.resample('bilinear'))
    .addBands(elderly_ratio.resample('bilinear'));

// ──────────────────────────────────────────────────────────
//  14C. BUILT-UP + LULC (shared across models)
// ──────────────────────────────────────────────────────────

var ml_builtup = BuiltUp_m2;   // 'BuiltUp_m2' — already named correctly

// ──────────────────────────────────────────────────────────
//  14D. MASTER FEATURE STACK  (Model A + Model B + labels)
//       Band order = column order in the exported CSV
//       MODEL C (OSM features) will be appended in Python.
// ──────────────────────────────────────────────────────────

var masterStack = ml_lst          // ★ target: lst_celsius, emissivity
    // ─ Model A: spectral ─
    .addBands(ml_l_bands)          // L_Blue … L_SWIR2         (6)
    .addBands(ml_l_idx)            // NDVI_L … NBI_L           (7)
    .addBands(ml_albedo)           // albedo                   (1)
    .addBands(ml_s2_bands)         // S2_Blue … S2_SWIR2       (6)
    .addBands(ml_s2_idx)           // NDVI_S2 … NBI_S2         (7)
    // ─ Model A: weather ─
    .addBands(ml_weather)          // air_temp_C × 3, rainfall, wind, humidity  (6)
    // ─ Model A: population ─
    .addBands(ml_pop)              // pop_density, GHSL_Pop_2020, GPW_PopDensity (3)
    // ─ Model A: built-up & terrain ─
    .addBands(ml_builtup)          // BuiltUp_m2                (1)
    .addBands(ml_terrain)          // Elevation_m … TPI_500m    (4)
    // ─ Model A: LULC label ─
    .addBands(LULC_ESA)            // LULC_ESA                  (1) int
    // ─ Model B: solar + NTL + demographics ─
    .addBands(ml_model_b)          // solar_rad_W_m2, ntl_radiance, children_ratio, elderly_ratio (4)
    .clip(AOI);

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('ML master stack band count:', masterStack.bandNames().size());
print('ML master stack bands:', masterStack.bandNames());

// ──────────────────────────────────────────────────────────
//  14E. NULL-SAFE SAMPLING
//       dropNulls:true skips pixels where ANY band is masked.
//       Critical columns are further enforced by the notNull filter
//       applied to the FeatureCollection after sampling.
//
//       v3.1 FIX 1: tileScale: PARAMS.TILE_SCALE added — subdivides
//       each compute tile so the ~50-band masterStack expression
//       graph fits under the per-worker memory ceiling (Error 3).
// ──────────────────────────────────────────────────────────

var ml_samples = masterStack.sample({
    region: AOI,
    scale: PARAMS.ML_SCALE,       // 100 m
    numPixels: PARAMS.ML_SAMPLES,     // 20000
    seed: PARAMS.ML_SEED,        // 42
    tileScale: PARAMS.TILE_SCALE,     // NEW — 4 (try 8/16 if Error 3 persists)
    geometries: true,
    dropNulls: true
});

// Hard null-check on the five most critical columns
// (belt-and-suspenders — dropNulls already removes most)
var REQUIRED_COLS = ['lst_celsius', 'NDVI_L', 'NDBI_L', 'air_temp_C',
    'solar_rad_W_m2', 'ntl_radiance'];
var ml_clean = ml_samples.filter(ee.Filter.notNull(REQUIRED_COLS));

// ──────────────────────────────────────────────────────────
//  14F. ADD COORDINATE COLUMNS + grid_id
//       Flatten geometry into explicit lon/lat columns so CSV
//       opens cleanly in pandas without parsing GeoJSON.
//       grid_id = 'lat4_lon4' e.g. '18.5234_73.8812'
// ──────────────────────────────────────────────────────────

var ml_csv = ml_clean.map(function (feat) {
    var coords = feat.geometry().coordinates();
    var lon = coords.get(0);
    var lat = coords.get(1);
    // Rounded to ~11 m precision (4 decimal places)
    var lon4 = ee.Number(lon).multiply(10000).round().divide(10000);
    var lat4 = ee.Number(lat).multiply(10000).round().divide(10000);
    var gid = lat4.format('%.4f').cat('_').cat(lon4.format('%.4f'));
    return feat
        .set('longitude', lon)
        .set('latitude', lat)
        .set('grid_id', gid)
        .set('year', PARAMS.START_DATE.slice(0, 4))
        .setGeometry(null);  // drop GeoJSON geometry column from CSV
});

// ──────────────────────────────────────────────────────────
//  14G. EXPORT AS CSV  ← this is the ML training file
// ──────────────────────────────────────────────────────────

Export.table.toDrive({
    collection: ml_csv,
    description: 'Pune_ML_Dataset_CSV_' + PARAMS.START_DATE.slice(0, 4),
    folder: PARAMS.DRIVE_FOLDER,
    fileFormat: 'CSV'    // ← table export, NOT image/GeoTIFF
});

// ─ Optional: Hot-season CSV (Model training for peak heat) ─
var ml_hot_stack = landsat.filter(ee.Filter.calendarRange(3, 5, 'month'))
    .map(computeLST).median().select(['lst_celsius', 'emissivity'])
    .addBands(addIndices(l_hot, 'L8L9').select(['NDVI', 'NDBI', 'MNDWI', 'EVI'])
        .rename(['NDVI_L', 'NDBI_L', 'MNDWI_L', 'EVI_L']))
    .addBands(computeAlbedo(l_hot))
    .addBands(solar_hot)
    .addBands(ntl_radiance)
    .addBands(air_temp_C).addBands(wind_speed).addBands(humidity_pct)
    .addBands(pop_density).addBands(BuiltUp_m2)
    .addBands(Elevation_m).addBands(Slope_deg)
    .addBands(LULC_ESA)
    .clip(AOI);

// v3.1 FIX 1: tileScale: PARAMS.TILE_SCALE added here too (same rationale as 14E)
var hot_samples = ml_hot_stack.sample({
    region: AOI, scale: PARAMS.ML_SCALE, numPixels: ee.Number(PARAMS.ML_SAMPLES).divide(2).int(),
    seed: PARAMS.ML_SEED + 1, tileScale: PARAMS.TILE_SCALE, geometries: true, dropNulls: true
}).filter(ee.Filter.notNull(['lst_celsius', 'NDVI_L', 'solar_hot_W_m2']))
    .map(function (f) {
        var c = f.geometry().coordinates();
        return f.set('longitude', c.get(0)).set('latitude', c.get(1))
            .set('season', 'hot_MAM').setGeometry(null);
    });

Export.table.toDrive({
    collection: hot_samples,
    description: 'Pune_ML_HotSeason_CSV_' + PARAMS.START_DATE.slice(0, 4),
    folder: PARAMS.DRIVE_FOLDER,
    fileFormat: 'CSV'
});

print('✅ ML Annual CSV  export queued → Pune_ML_Dataset_CSV_' + PARAMS.START_DATE.slice(0, 4) + '.csv');
print('✅ ML Hot-Season  export queued → Pune_ML_HotSeason_CSV_' + PARAMS.START_DATE.slice(0, 4) + '.csv');


// ════════════════════════════════════════════════════════════
//  15. SUMMARY PRINT
// ════════════════════════════════════════════════════════════

print('');
print('╔════════════════════════════════════════╗');
print('  URBANCOOL AI — PUNE PIPELINE v3.1     ');
print('╚════════════════════════════════════════╝');
print('Study period  :', PARAMS.START_DATE, '→', PARAMS.END_DATE);
print('Drive folder  :', PARAMS.DRIVE_FOLDER);
print('Landsat scenes:', landsat.size());
print('Sentinel scene:', s2_joined.size());
print('VIIRS monthly :', viirs.size());
print('ERA5 daily    :', era5_daily.size());
print('');
print('ML_SAMPLES    :', PARAMS.ML_SAMPLES, ' | TILE_SCALE:', PARAMS.TILE_SCALE);
print('');
print('★ PRIMARY CSV EXPORTS (check Tasks panel):');
print('  Pune_ML_Dataset_CSV_<year>.csv');
print('  ├── Cols: grid_id, longitude, latitude, year');
print('  ├── TARGET : lst_celsius, emissivity');
print('  ├── MODEL A: L_Blue…L_SWIR2 (6)');
print('  │            NDVI_L…NBI_L (7)');
print('  │            albedo (1)');
print('  │            S2_Blue…S2_SWIR2 (6)');
print('  │            NDVI_S2…NBI_S2 (7)');
print('  │            air_temp_C, air_temp_C_max, air_temp_C_min (3)');
print('  │            rainfall_mm, wind_speed, humidity_pct (3)');
print('  │            pop_density, GHSL_Pop_2020, GPW_PopDensity (3)');
print('  │            BuiltUp_m2 (1)');
print('  │            Elevation_m, Slope_deg, Aspect_deg, TPI_500m (4)');
print('  │            LULC_ESA (1)');
print('  └── MODEL B: solar_rad_W_m2, ntl_radiance (2)');
print('               children_ratio, elderly_ratio (2)');
print('  Total GEE cols: ~50 + 4 metadata = ~54 columns');
print('  → Python adds MODEL C: road/building/park/water (OSMnx)');
print('');
print('  Pune_ML_HotSeason_CSV_<year>.csv');
print('  └── Lean hot-season subset for peak LST training');
print('');
print('★ SECONDARY GeoTIFF EXPORTS:');
if (PARAMS.EXPORT_TIFF.landsat) print('  • Pune_Landsat_Annual/Hot/Monsoon (30 m)');
if (PARAMS.EXPORT_TIFF.lst) print('  • Pune_LST_Annual/HotSeason/Monsoon (30 m)');
if (PARAMS.EXPORT_TIFF.terrain) print('  • Pune_SRTM_Terrain (30 m)');
if (PARAMS.EXPORT_TIFF.population) print('  • Pune_Population_Stack_2020 (100 m)');
print('');
print('Next steps after download:');
print('  python src/data/01_schema_align_and_albedo.py');
print('  python src/data/02_add_osm_features.py');
print('  python src/data/03_add_ward_join.py');
print('  python src/models/train_lst_model.py --model A');
print('  python src/models/train_lst_model.py --model B');
print('  python src/models/train_lst_model.py --model C');
print('╔════════════════════════════════════════╗');
print('  Rows: ~', PARAMS.ML_SAMPLES, ' | Scale: ', PARAMS.ML_SCALE, ' m ');
print('╚════════════════════════════════════════╝');


// ════════════════════════════════════════════════════════════
//  MAP LAYER LEGEND
//  ─────────────────────────────────────────────────────────
//  01. Pune AOI              red outline
//  02. Landsat Annual RGB    true colour
//  03. Albedo (Landsat)      Liang 2001  [black → white]
//  04. Sentinel-2 Annual RGB true colour
//  05. Sentinel-2 FCC        NIR-R-G
//  06. NDVI (Landsat)        brown → green
//  07. NDBI (Landsat)        green → purple (built-up)
//  08. MNDWI (Landsat)       red → blue (water)
//  09. LST Annual (°C)       blue → red
//  10. LST Hot Season (°C)   blue → red
//  11. ERA5 Mean Temp (°C)   blue → orange
//  12. ERA5 Rainfall (mm)    white → dark blue
//  13. Solar Rad (W/m²)      blue → red  [NEW]
//  14. VIIRS NTL (nW/cm²/sr) dark → white  [NEW]
//  15. WorldPop 2020         white → dark red
//  16. Elderly Ratio (60+)   white → red  [NEW]
//  17. Children Ratio (0-14) white → green  [NEW]
//  18. DEM (m)               green → white
//  19. Slope (°)             white → dark red
//  20. GHSL Built-Up 2020    white → dark (m²/pixel)
//  21. Built-Up Change       blue (loss) → red (gain)
//  22. ESA WorldCover 2021   standard 11-class palette
// ════════════════════════════════════════════════════════════