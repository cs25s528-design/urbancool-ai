"use strict";

const INTERVENTION_META = {
  trees: {
    label: "Tree Cover",
    color: "#1f8a5b",
    bgColor: "#e8f7ef",
    co2PerDegC: 12.5,
    energySavePctPerDegC: 3.2,
    timeToEffect: "3-5 yrs",
    svg: `
      <rect x="17" y="28" width="6" height="10" rx="2" fill="#8B5E3C"/>
      <ellipse cx="20" cy="22" rx="9.5" ry="7.5" fill="#2da65a"/>
      <ellipse cx="13" cy="18" rx="6.5" ry="5.5" fill="#1a7a42"/>
      <ellipse cx="27" cy="18" rx="6.5" ry="5.5" fill="#1a7a42"/>
      <ellipse cx="20" cy="13" rx="8.5" ry="6.5" fill="#27c064"/>
      <ellipse cx="15" cy="14" rx="4" ry="3.5" fill="#2dd670" opacity="0.45"/>
    `,
  },
  cool_roofs: {
    label: "Cool Roofs",
    color: "#087f8c",
    bgColor: "#e0f5f7",
    co2PerDegC: 8.2,
    energySavePctPerDegC: 7.5,
    timeToEffect: "Immediate",
    svg: `
      <polygon points="20,5 37,20 3,20" fill="#c5ecef"/>
      <polygon points="20,5 37,20 3,20" fill="rgba(255,255,255,0.32)"/>
      <rect x="6" y="20" width="28" height="16" rx="1" fill="#087f8c"/>
      <rect x="15" y="24" width="10" height="12" rx="1.5" fill="#065f6a"/>
      <ellipse cx="29" cy="11" rx="3.2" ry="2" fill="rgba(255,255,255,0.65)" transform="rotate(-30 29 11)"/>
      <line x1="6" y1="20" x2="34" y2="20" stroke="rgba(255,255,255,0.4)" stroke-width="1"/>
    `,
  },
  reflective_pavement: {
    label: "Reflective Pavement",
    color: "#4a5965",
    bgColor: "#ecf0f3",
    co2PerDegC: 3.8,
    energySavePctPerDegC: 2.1,
    timeToEffect: "1-2 yrs",
    svg: `
      <rect x="4" y="14" width="32" height="14" rx="3" fill="#8a9eac"/>
      <rect x="4" y="14" width="32" height="5" rx="3" fill="rgba(255,255,255,0.22)"/>
      <rect x="7" y="19.5" width="6" height="2.5" rx="1" fill="white" opacity="0.8"/>
      <rect x="17" y="19.5" width="6" height="2.5" rx="1" fill="white" opacity="0.8"/>
      <rect x="27" y="19.5" width="6" height="2.5" rx="1" fill="white" opacity="0.8"/>
      <ellipse cx="20" cy="8" rx="6.5" ry="4" fill="#ffe87a" opacity="0.55"/>
      <line x1="20" y1="4" x2="20" y2="12" stroke="#ffd700" stroke-width="1.5" opacity="0.35"/>
    `,
  },
  blue_green: {
    label: "Blue-Green Infra.",
    color: "#277da1",
    bgColor: "#e0eef5",
    co2PerDegC: 10.8,
    energySavePctPerDegC: 4.8,
    timeToEffect: "2-3 yrs",
    svg: `
      <path d="M20 5 C20 5 9 19 9 26 a11 11 0 0 0 22 0 C31 19 20 5 20 5z" fill="#277da1"/>
      <path d="M20 12 C20 12 13 22 13 26 a7 7 0 0 0 14 0 C27 22 20 12 20 12z" fill="#4aa3c8"/>
      <path d="M12 24 C14 16 27 18 28 24 C24 31 12 24z" fill="#43aa8b" opacity="0.88"/>
      <circle cx="20" cy="24" r="2" fill="rgba(255,255,255,0.35)"/>
    `,
  },
  combined: {
    label: "Combined Package",
    color: "#a05c00",
    bgColor: "#fef8ec",
    co2PerDegC: 26.4,
    energySavePctPerDegC: 11.5,
    timeToEffect: "3-7 yrs",
    svg: `
      <polygon points="20,3 23.5,13 34.5,13 26,19.5 29,30 20,24 11,30 14,19.5 5.5,13 16.5,13" fill="#f9a620"/>
      <polygon points="20,7 22.8,14.5 30.5,14.5 24.5,18.5 26.8,26 20,22 13.2,26 15.5,18.5 9.5,14.5 17.2,14.5" fill="#ffd166"/>
    `,
  },
};

const state = {
  map: null,
  selectedMarker: null,
  heatLayer: null,
  leafletReady: false,
  fallbackBounds: null,
  fallbackMarker: null,
  latestHeat: null,
  latestWeather: null,
  latestDrivers: null,
  latestScenarios: null,
  latestFuture: null,
};

const el = (id) => document.getElementById(id);
const fmt = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";
const PREDICTION_YEAR = 2030;

function currentDateLabel() {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date());
}

function heatColor(lst) {
  if (lst >= 42) return "#c1121f";
  if (lst >= 38) return "#f3722c";
  if (lst >= 34) return "#f9c74f";
  if (lst >= 30) return "#43aa8b";
  return "#277da1";
}

function calcHeatIndex(tempC, rhPct) {
  if (!Number.isFinite(tempC) || !Number.isFinite(rhPct)) return null;
  const tempF = tempC * 9 / 5 + 32;
  if (tempF < 80) return tempC;
  const rh = rhPct;
  const hi =
    -42.379 + 2.04901523 * tempF + 10.14333127 * rh
    - 0.22475541 * tempF * rh - 0.00683783 * tempF * tempF
    - 0.05481717 * rh * rh + 0.00122874 * tempF * tempF * rh
    + 0.00085282 * tempF * rh * rh - 0.00000199 * tempF * tempF * rh * rh;
  return (hi - 32) * 5 / 9;
}

function labelIntervention(name) {
  return INTERVENTION_META[name]?.label ?? name;
}

function riskText(risk) {
  return String(risk || "unknown").replace("_", " ");
}

const DRIVER_LABELS = {
  NDVI_L: ["Low tree cover", "Less shade and evapotranspiration makes surfaces hotter."],
  EVI_L: ["Weak vegetation health", "Sparse or stressed vegetation reduces natural cooling."],
  SAVI_L: ["Less green cover near soil", "Open dry soil and low vegetation can raise local heat."],
  NDBI_L: ["Dense built-up area", "Concrete, roofs, and roads store heat during the day."],
  MNDWI_L: ["Less nearby water", "Water bodies and moist areas usually cool the surroundings."],
  albedo: ["Dark surface materials", "Darker roofs and pavements absorb more sunlight."],
  solar_rad_W_m2: ["Strong sunlight", "High solar radiation increases surface heating."],
  air_temp_C: ["Hot weather", "Higher air temperature adds to heat stress."],
  air_temp_C_max: ["Very hot daytime weather", "High daily peaks increase heat exposure."],
  air_temp_C_min: ["Warm nights", "Night heat can keep surfaces from cooling down."],
  humidity_pct: ["High humidity", "Humidity makes the body cool itself less effectively."],
  wind_speed: ["Low air movement", "Weak wind reduces natural cooling and ventilation."],
  rainfall_mm: ["Low recent rainfall", "Dry ground and dry vegetation heat up faster."],
  Elevation_m: ["Local terrain", "Terrain affects air flow, shade, and heat accumulation."],
  Slope_deg: ["Slope and terrain shape", "Land shape changes sunlight exposure and drainage."],
  TPI_500m: ["Built-up land form", "Local land shape can trap heat in dense areas."],
  pop_density: ["Crowded urban activity", "More people and activity often means more waste heat."],
  ntl_radiance: ["Night activity and lighting", "Bright night-time areas often mark dense urban use."],
  road_density: ["More roads", "Road surfaces absorb heat and reduce green cover."],
  building_density: ["More buildings", "Dense buildings store heat and limit air movement."],
  impervious_ratio: ["More paved surface", "Hard surfaces stop soil moisture cooling."],
  dist_road_m: ["Road exposure", "Nearby roads add hot pavement and traffic heat."],
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function loadCss(href) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`link[href="${href}"]`)) {
      resolve();
      return;
    }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    link.onload = resolve;
    link.onerror = reject;
    document.head.appendChild(link);
  });
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = reject;
    document.body.appendChild(script);
  });
}

async function loadLeaflet() {
  if (window.L) return true;
  const timeout = new Promise((resolve) => {
    window.setTimeout(() => resolve(false), 3500);
  });
  const load = Promise.all([
    loadCss("https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"),
    loadScript("https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"),
  ]).then(() => Boolean(window.L)).catch(() => false);
  return Promise.race([load, timeout]);
}

function createInterventionDivIcon(type) {
  const meta = INTERVENTION_META[type] ?? INTERVENTION_META.combined;
  const html = `
    <div class="uc-pin" style="--pin-color:${meta.color};--pin-bg:${meta.bgColor};">
      <div class="uc-pin__head">
        <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" width="32" height="32" aria-hidden="true">${meta.svg}</svg>
      </div>
      <div class="uc-pin__tip"></div>
    </div>`;
  return L.divIcon({
    html,
    className: "uc-intervention-icon",
    iconSize: [48, 62],
    iconAnchor: [24, 62],
    popupAnchor: [0, -64],
  });
}

function setRecommendationMarker(lat, lon, type, risk, deltaC) {
  if (!state.leafletReady || !state.map) {
    setFallbackRecommendationMarker(lat, lon, type, risk, deltaC);
    return;
  }
  if (state.selectedMarker) {
    state.selectedMarker.remove();
    state.selectedMarker = null;
  }

  const meta = INTERVENTION_META[type] ?? INTERVENTION_META.combined;
  const co2 = ((deltaC || 0) * (meta.co2PerDegC || 5)).toFixed(1);
  const energy = ((deltaC || 0) * (meta.energySavePctPerDegC || 3)).toFixed(1);
  const riskText = (risk || "unknown").replace("_", " ");

  const popupHtml = `
    <div class="uc-popup-inner">
      <div class="uc-popup-title" style="color:${meta.color}">${meta.label}</div>
      <p class="uc-popup-sub">Recommended cooling strategy</p>
      <dl class="uc-popup-dl">
        <div><dt>Heat risk</dt><dd><span class="risk-tag risk-${risk || "unknown"}">${riskText}</span></dd></div>
        <div><dt>Cooling</dt><dd>-${fmt(deltaC)} °C</dd></div>
        <div><dt>CO2 offset</dt><dd>~${co2} t/yr</dd></div>
        <div><dt>Energy saved</dt><dd>~${energy}%</dd></div>
        <div><dt>Time to effect</dt><dd>${meta.timeToEffect}</dd></div>
      </dl>
    </div>`;

  state.selectedMarker = L.marker([lat, lon], {
    icon: createInterventionDivIcon(type),
  })
    .bindPopup(L.popup({ maxWidth: 250, className: "uc-popup" }).setContent(popupHtml))
    .addTo(state.map)
    .openPopup();

  state.map.panTo([lat, lon], { animate: true, duration: 0.4 });
}

function initMap() {
  if (!window.L) {
    initFallbackMap();
    return;
  }

  state.leafletReady = true;
  state.map = L.map("map", { zoomControl: false }).setView([18.5204, 73.8567], 11);
  L.control.zoom({ position: "bottomright" }).addTo(state.map);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);

  state.heatLayer = L.layerGroup().addTo(state.map);
  state.map.on("click", (evt) => {
    setInputs(evt.latlng.lat, evt.latlng.lng);
    analyze();
  });
}

function initFallbackMap() {
  state.leafletReady = false;
  const map = el("map");
  map.classList.add("fallback-map");
  map.innerHTML = `
    <div class="fallback-map__base"></div>
    <div class="fallback-map__title">
      <strong>Pune heat layer</strong>
      <span>Local map fallback</span>
    </div>
    <div id="fallbackPoints" class="fallback-points"></div>
  `;
  el("mapFallback").hidden = true;
  map.addEventListener("click", (evt) => {
    if (!state.fallbackBounds) return;
    const rect = map.getBoundingClientRect();
    const x = (evt.clientX - rect.left) / rect.width;
    const y = (evt.clientY - rect.top) / rect.height;
    const lon = state.fallbackBounds.minLon + x * (state.fallbackBounds.maxLon - state.fallbackBounds.minLon);
    const lat = state.fallbackBounds.maxLat - y * (state.fallbackBounds.maxLat - state.fallbackBounds.minLat);
    setInputs(lat, lon);
    analyze();
  });
}

function computeBounds(points) {
  const lats = points.map((p) => Number(p.lat)).filter(Number.isFinite);
  const lons = points.map((p) => Number(p.lon)).filter(Number.isFinite);
  if (!lats.length || !lons.length) return null;
  const padLat = Math.max((Math.max(...lats) - Math.min(...lats)) * 0.04, 0.01);
  const padLon = Math.max((Math.max(...lons) - Math.min(...lons)) * 0.04, 0.01);
  return {
    minLat: Math.min(...lats) - padLat,
    maxLat: Math.max(...lats) + padLat,
    minLon: Math.min(...lons) - padLon,
    maxLon: Math.max(...lons) + padLon,
  };
}

function fallbackPosition(lat, lon) {
  if (!state.fallbackBounds) return { left: 50, top: 50 };
  const b = state.fallbackBounds;
  const left = ((lon - b.minLon) / Math.max(b.maxLon - b.minLon, 0.000001)) * 100;
  const top = ((b.maxLat - lat) / Math.max(b.maxLat - b.minLat, 0.000001)) * 100;
  return {
    left: Math.max(0, Math.min(100, left)),
    top: Math.max(0, Math.min(100, top)),
  };
}

function renderFallbackPoints(points) {
  if (state.leafletReady) return;
  const holder = document.getElementById("fallbackPoints");
  if (!holder) return;
  state.fallbackBounds = computeBounds(points);
  holder.innerHTML = points.map((p) => {
    const pos = fallbackPosition(Number(p.lat), Number(p.lon));
    const color = heatColor(Number(p.lst_celsius));
    return `<button class="fallback-dot" type="button" style="left:${pos.left}%;top:${pos.top}%;background:${color}" data-lat="${p.lat}" data-lon="${p.lon}" title="Surface heat ${fmt(p.lst_celsius)} °C"></button>`;
  }).join("");
  holder.querySelectorAll(".fallback-dot").forEach((dot) => {
    dot.addEventListener("click", (evt) => {
      evt.stopPropagation();
      setInputs(Number(dot.dataset.lat), Number(dot.dataset.lon));
      analyze();
    });
  });
}

function setFallbackRecommendationMarker(lat, lon, type, risk, deltaC) {
  const holder = document.getElementById("fallbackPoints");
  if (!holder || !state.fallbackBounds) return;
  if (state.fallbackMarker) {
    state.fallbackMarker.remove();
  }
  const meta = INTERVENTION_META[type] ?? INTERVENTION_META.combined;
  const pos = fallbackPosition(lat, lon);
  const marker = document.createElement("div");
  marker.className = "fallback-recommendation";
  marker.style.left = `${pos.left}%`;
  marker.style.top = `${pos.top}%`;
  marker.style.setProperty("--pin-color", meta.color);
  marker.style.setProperty("--pin-bg", meta.bgColor);
  marker.innerHTML = `
    <div class="uc-pin">
      <div class="uc-pin__head">
        <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" width="32" height="32" aria-hidden="true">${meta.svg}</svg>
      </div>
      <div class="uc-pin__tip"></div>
    </div>
    <div class="fallback-callout">
      <strong>${meta.label}</strong>
      <span>${(risk || "unknown").replace("_", " ")} · -${fmt(deltaC)} °C</span>
    </div>`;
  holder.appendChild(marker);
  state.fallbackMarker = marker;
}

function setInputs(lat, lon) {
  el("latInput").value = Number(lat).toFixed(6);
  el("lonInput").value = Number(lon).toFixed(6);
}

async function loadSamples() {
  try {
    const data = await api("/v1/heat/samples?limit=1100");
    el("rowCount").textContent = data.total_rows.toLocaleString();
    el("meanLst").textContent = `${fmt(data.lst_mean)} °C`;
    el("lstRange").textContent = `${fmt(data.lst_min)}-${fmt(data.lst_max)} °C`;

    renderFallbackPoints(data.points);
    if (!state.leafletReady || !state.map || !state.heatLayer) return;
    state.heatLayer.clearLayers();

    data.points.forEach((p) => {
    const marker = L.circleMarker([p.lat, p.lon], {
        radius: 4,
        stroke: false,
        fillColor: heatColor(p.lst_celsius),
        fillOpacity: 0.62,
      });
      marker.bindTooltip(
        `Surface heat ${fmt(p.lst_celsius)} °C${p.NDVI_L != null ? ` · green cover ${fmt(p.NDVI_L, 2)}` : ""}`,
        { sticky: true, className: "uc-tooltip" }
      );
      marker.on("click", () => {
        setInputs(p.lat, p.lon);
        analyze();
      });
      marker.addTo(state.heatLayer);
    });
  } catch (err) {
    console.error(err);
    el("rowCount").textContent = "unavailable";
    el("lstRange").textContent = "--";
  }
}

function renderHeat(data) {
  state.latestHeat = data;
  el("predLst").textContent = `${fmt(data.pred_lst_C)} °C`;
  const risk = data.risk_level || "unknown";
  const riskEl = el("riskLevel");
  riskEl.textContent = riskText(risk);
  riskEl.className = `risk-badge risk-${risk}`;

  const hi = calcHeatIndex(data.air_temp_C, data.humidity_pct);
  el("heatIndex").textContent = hi !== null ? `${fmt(hi)} °C` : "--";
  el("hiLabel").textContent = hi !== null && hi > (data.air_temp_C ?? 0) + 1.5 ? "Feels Hotter" : "Feels Like";
  el("weatherMeta").textContent = `${fmt(data.humidity_pct, 0)}% humidity · ${fmt(data.wind_speed)} m/s wind`;

  el("airTemp").textContent = `${fmt(data.air_temp_C)} °C`;
  el("obsLst").textContent = `area heat ${fmt(data.observed_lst_C)} °C`;

  const grid = data.nearest_grid || {};
  el("nearestGrid").textContent = `${fmt(grid.lat, 5)}, ${fmt(grid.lon, 5)}`;
  el("wardName").textContent = data.ward_name || "Unknown";
  el("wardId").textContent = data.ward_id ? `#${data.ward_id}` : "-";

  const sideRisk = el("sideRisk");
  sideRisk.textContent = riskText(risk);
  sideRisk.className = `risk-badge risk-${risk}`;
  renderSummary();
}

function renderWeather(weather, heatFallback) {
  if (!weather || weather.status !== "ok") {
    state.latestWeather = null;
    el("weatherCondition").textContent = "Live weather unavailable";
    el("cloudCover").textContent = "--";
    el("pressure").textContent = "--";
    el("rainNow").textContent = "--";
    el("weatherSource").textContent = "Weather update unavailable";
    renderSummary();
    return;
  }

  state.latestWeather = weather;
  const temp = Number(weather.temperature_C);
  const humidity = Number(weather.humidity_pct);
  const wind = Number(weather.wind_speed_m_s);
  const feels = Number(weather.feels_like_C);
  const heatIndex = Number.isFinite(feels) ? feels : calcHeatIndex(temp, humidity);

  el("airTemp").textContent = `${fmt(temp)} °C`;
  el("heatIndex").textContent = heatIndex !== null ? `${fmt(heatIndex)} °C` : "--";
  el("hiLabel").textContent = heatIndex !== null && heatIndex > temp + 1.5 ? "Feels Hotter" : "Feels Like";
  el("weatherMeta").textContent = `${fmt(humidity, 0)}% humidity · ${fmt(wind)} m/s wind`;
  el("weatherCondition").textContent = weather.condition || "Current weather";
  el("cloudCover").textContent = `${fmt(weather.cloud_cover_pct, 0)}%`;
  el("pressure").textContent = `${fmt(weather.pressure_hpa, 0)} hPa`;
  el("rainNow").textContent = `${fmt(weather.precipitation_mm, 1)} mm`;
  el("weatherSource").textContent = weather.time ? `Updated ${weather.time}` : "Weather updated";

  if (heatFallback) {
    el("obsLst").textContent = `area heat ${fmt(heatFallback.observed_lst_C)} °C`;
  }
  renderSummary();
}

function renderDrivers(data) {
  state.latestDrivers = data;
  const drivers = data.drivers || [];
  const max = Math.max(...drivers.map((d) => Math.abs(d.importance || 0)), 0.001);
  el("driversList").innerHTML = drivers.map((d) => {
    const pct = Math.max(4, Math.round((Math.abs(d.importance || 0) / max) * 100));
    const [label, note] = DRIVER_LABELS[d.feature] || [d.feature, "This local condition is linked with higher heat."];
    return `
      <div class="driver-row">
        <span><strong>${label}</strong><small>${note}</small></span>
        <div class="driver-track"><div class="driver-fill" style="width:${pct}%"></div></div>
        <strong>${Math.round(pct)}%</strong>
      </div>`;
  }).join("");
  renderSummary();
}

function renderInterventionCards(data) {
  state.latestScenarios = data;
  const rows = Object.values(data.scenarios || {})
    .sort((a, b) => (b.delta_C || 0) - (a.delta_C || 0));

  const best = data.best || rows[0];
  el("bestDelta").textContent = best ? `-${fmt(best.delta_C)} °C` : "--";
  el("bestName").textContent = best ? labelIntervention(best.intervention) : "--";
  el("scenarioLatLon").textContent = `${fmt(data.lat, 5)}, ${fmt(data.lon, 5)}`;

  const maxDelta = Math.max(...rows.map((r) => r.delta_C || 0), 0.1);
  el("interventionCards").innerHTML = rows.map((row, idx) => {
    const meta = INTERVENTION_META[row.intervention] ?? INTERVENTION_META.combined;
    const pct = Math.max(4, Math.round(((row.delta_C || 0) / maxDelta) * 100));
    const co2 = ((row.delta_C || 0) * (meta.co2PerDegC || 5)).toFixed(1);
    const energy = ((row.delta_C || 0) * (meta.energySavePctPerDegC || 3)).toFixed(1);
    return `
      <div class="ic" style="--ic-color:${meta.color};--ic-bg:${meta.bgColor};" data-intervention="${row.intervention}">
        <div class="ic-icon">
          <svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" width="32" height="32" aria-hidden="true">${meta.svg}</svg>
        </div>
        <div class="ic-body">
          <div class="ic-top">
            <span class="ic-label">${meta.label}</span>
            ${idx === 0 ? '<span class="ic-best-badge">Best</span>' : ""}
          </div>
          <div class="ic-bar-track"><div class="ic-bar-fill" style="width:${pct}%"></div></div>
          <div class="ic-sub">
            <span>-${fmt(row.delta_C)} °C</span>
            <span>${meta.timeToEffect}</span>
            <span>~${co2} t CO2</span>
            <span>${energy}% energy</span>
          </div>
        </div>
        <div class="ic-delta">${fmt(row.delta_C)}<span class="ic-unit">°C</span></div>
      </div>`;
  }).join("");
  renderSummary();
}

function renderFuture(data) {
  state.latestFuture = data;
  renderSummary();
}

function topCauseLabels(limit = 2) {
  const drivers = state.latestDrivers?.drivers || [];
  return drivers.slice(0, limit).map((d) => {
    const [label] = DRIVER_LABELS[d.feature] || [d.feature];
    return label.toLowerCase();
  });
}

function renderSummary() {
  const heat = state.latestHeat;
  if (!heat) return;

  const weather = state.latestWeather;
  const scenarios = state.latestScenarios;
  const future = state.latestFuture;
  const best = scenarios?.best;
  const place = heat.ward_name || "selected area";
  const risk = riskText(heat.risk_level);
  const currentTemp = Number.isFinite(Number(weather?.temperature_C))
    ? Number(weather.temperature_C)
    : Number(heat.air_temp_C);
  const feels = Number.isFinite(Number(weather?.feels_like_C))
    ? Number(weather.feels_like_C)
    : null;
  const futureHeat = Number.isFinite(Number(future?.no_action_lst_C))
    ? Number(future.no_action_lst_C)
    : Number(heat.pred_lst_C);
  const causes = topCauseLabels(2);
  const causeText = causes.length ? causes.join(" and ") : "local built-up and weather conditions";
  const action = best ? labelIntervention(best.intervention).toLowerCase() : "a cooling intervention";
  const reduction = Number.isFinite(Number(best?.delta_C)) ? Math.max(0, Number(best.delta_C)) : null;

  el("summaryPlace").textContent = place;
  el("summaryCurrentDate").textContent = currentDateLabel();
  el("summaryCurrentTemp").textContent = `${fmt(currentTemp)} °C`;
  el("summaryPredictionYear").textContent = String(PREDICTION_YEAR);
  el("summaryFutureTemp").textContent = `${fmt(futureHeat)} °C by ${PREDICTION_YEAR} if no cooling action`;
  el("summaryAction").textContent = best ? labelIntervention(best.intervention) : "--";
  el("summaryUpdated").textContent = `today: ${currentDateLabel()} · prediction: ${PREDICTION_YEAR}`;

  const feelsText = feels !== null ? `, feeling like ${fmt(feels)} °C` : "";
  const reductionText = reduction !== null
    ? ` The strongest cooling option is ${action}, with an estimated reduction of about ${fmt(reduction)} °C.`
    : "";
  el("areaSummaryText").textContent =
    `${place} is currently around ${fmt(currentTemp)} °C${feelsText}. ` +
    `The local surface heat is estimated near ${fmt(heat.pred_lst_C)} °C, giving a ${risk} heat risk. ` +
    `By ${PREDICTION_YEAR}, if no cooling action is taken, the heat outlook stays around ${fmt(futureHeat)} °C. ` +
    `The main reasons appear to be ${causeText}.${reductionText}`;

  const badges = [
    ["Weather", weather?.condition || "saved local weather"],
    ["Risk", risk],
    ["Main Cause", causes[0] || "urban heat conditions"],
    ["Cooling", reduction !== null ? `-${fmt(reduction)} °C` : "--"],
  ];
  el("summaryBadges").innerHTML = badges.map(([k, v]) => `
    <span class="summary-badge"><strong>${k}</strong>${v}</span>
  `).join("");
}

async function analyze() {
  const lat = Number(el("latInput").value);
  const lon = Number(el("lonInput").value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

  el("analyzeBtn").disabled = true;
  el("analyzeBtn").innerHTML = "<span aria-hidden='true'>↻</span> Analyzing...";

  try {
    const weatherPromise = api(`/v1/weather/current?lat=${lat}&lon=${lon}`)
      .catch(() => null);

    const [heat, drivers, future, scenarios] = await Promise.all([
      api(`/v1/heat/current?lat=${lat}&lon=${lon}`),
      api(`/v1/heat/drivers?lat=${lat}&lon=${lon}&n=6`),
      api(`/v1/heat/future?lat=${lat}&lon=${lon}&horizon=2030`),
      api("/v1/scenario/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat,
          lon,
          interventions: ["trees", "cool_roofs", "blue_green", "combined", "reflective_pavement"],
        }),
      }),
    ]);

    renderHeat(heat);
    renderDrivers(drivers);
    renderInterventionCards(scenarios);
    renderFuture(future);
    weatherPromise.then((weather) => renderWeather(weather, heat));

    const grid = heat.nearest_grid || {};
    const pinLat = Number.isFinite(grid.lat) ? grid.lat : lat;
    const pinLon = Number.isFinite(grid.lon) ? grid.lon : lon;
    const recommended = scenarios.best?.intervention ?? "combined";
    const deltaC = scenarios.best?.delta_C ?? 0;
    setRecommendationMarker(pinLat, pinLon, recommended, heat.risk_level, deltaC);

  } catch (err) {
    console.error(err);
    el("weatherCondition").textContent = "Unable to update";
  } finally {
    el("analyzeBtn").disabled = false;
    el("analyzeBtn").innerHTML = "<span aria-hidden='true'>↻</span> Analyze Location";
  }
}

function bindControls() {
  el("analyzeBtn").addEventListener("click", analyze);
  el("locateBtn").addEventListener("click", () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition((pos) => {
      setInputs(pos.coords.latitude, pos.coords.longitude);
      analyze();
    });
  });
}

async function boot() {
  const leafletReady = await loadLeaflet();
  if (!leafletReady) {
    console.warn("Leaflet CDN unavailable; using local fallback map.");
  }
  initMap();
  bindControls();
  await loadSamples();
  await analyze();
}

boot();
