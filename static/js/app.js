/**
 * Asteroid Trajectory & Earth Risk Analysis - Main Application Controller
 */

// Application State
const AppState = {
  currentTab: "overview",
  catalogPage: 1,
  catalogLimit: 25,
  catalogSortBy: "moid",
  catalogOrder: "asc",
  catalogFilters: {
    search: "",
    orbitClass: "ALL",
    phaOnly: null,
    sentryOnly: false,
    cadOnly: false
  },
  selectedAsteroid: null,
  orbitVisualizer: null,
  edaLoaded: false,
  modelsLoaded: false
};

// Preset configurations for the custom risk prediction form
const PRESETS = {
  apophis: {
    name: "99942 Apophis (Critical PHA)",
    H: 19.09,
    diameter: 0.37,
    a: 0.9224,
    e: 0.1912,
    i: 3.33,
    moid: 0.0003,
    orbit_class: "ATE"
  },
  bennu: {
    name: "101955 Bennu (Sentry Monitored)",
    H: 20.61,
    diameter: 0.49,
    a: 1.1264,
    e: 0.2037,
    i: 6.03,
    moid: 0.0033,
    orbit_class: "APO"
  },
  tunguska: {
    name: "Tunguska-Class Bolide (~50m)",
    H: 24.5,
    diameter: 0.05,
    a: 1.05,
    e: 0.28,
    i: 4.5,
    moid: 0.012,
    orbit_class: "APO"
  },
  eros: {
    name: "433 Eros (Harmless Amor Giant)",
    H: 10.4,
    diameter: 16.8,
    a: 1.458,
    e: 0.2229,
    i: 10.83,
    moid: 0.149,
    orbit_class: "AMO"
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initSummaryMetrics();
  initCatalogTable();
  init3DVisualizer();
  initPredictionForm();
  initDataRefresh();
  initDetailModal();
});

/* -------------------------------------------------------------
   Navigation
   ------------------------------------------------------------- */
function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab-item");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetPaneId = tab.getAttribute("data-tab");
      switchTab(targetPaneId);
    });
  });
}

function switchTab(tabId) {
  AppState.currentTab = tabId;

  // Update nav tabs active style
  document.querySelectorAll(".nav-tab-item").forEach((t) => {
    t.classList.toggle("active", t.getAttribute("data-tab") === tabId);
  });

  // Show selected pane
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `tab-${tabId}`);
  });

  // Lazy load chart data
  if (tabId === "eda" && !AppState.edaLoaded) {
    loadEdaData();
  } else if (tabId === "models" && !AppState.modelsLoaded) {
    loadModelsData();
  } else if (tabId === "orbit" && AppState.orbitVisualizer) {
    setTimeout(() => AppState.orbitVisualizer.handleResize(), 50);
  }
}

/* -------------------------------------------------------------
   Summary Metrics & Critical Alerts
   ------------------------------------------------------------- */
async function initSummaryMetrics() {
  try {
    const res = await fetch("/api/metrics/summary");
    if (!res.ok) throw new Error("Failed to fetch summary metrics");
    const data = await res.json();

    // Populate metric cards
    document.getElementById("stat-total-monitored").innerText = data.total_monitored.toLocaleString();
    document.getElementById("stat-pha-count").innerText = `${data.pha_count.toLocaleString()} (${data.pha_percentage}%)`;
    document.getElementById("stat-sentry-count").innerText = data.sentry_monitored_count.toLocaleString();
    document.getElementById("stat-cad-count").innerText = data.upcoming_close_approaches.toLocaleString();
    document.getElementById("stat-avg-velocity").innerText = `${data.avg_encounter_velocity_kms} km/s`;
    document.getElementById("stat-min-moid").innerText = `${data.minimum_moid_au} AU`;

    // Render alert cards
    renderAlertCards(data.alert_objects || []);
  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

function renderAlertCards(alerts) {
  const container = document.getElementById("alerts-carousel");
  if (!container) return;

  if (alerts.length === 0) {
    container.innerHTML = `<div class="text-muted" style="padding: 1rem;">No immediate critical alerts recorded in current window.</div>`;
    return;
  }

  container.innerHTML = alerts
    .slice(0, 6)
    .map((item) => {
      const isSentry = item.is_sentry;
      const badgeClass = isSentry ? "badge-danger" : "badge-warning";
      const badgeLabel = isSentry ? "NASA SENTRY MONITORED" : "CLOSE APPROACH";

      const subDetail = isSentry
        ? `<span>Palermo: <b>${item.sentry_ps_max !== null ? item.sentry_ps_max.toFixed(2) : 'N/A'}</b></span>
           <span>Torino: <b>${item.sentry_ts_max}</b></span>`
        : `<span>Flyby: <b>${item.cad_date.split(' ')[0] || 'Upcoming'}</b></span>
           <span>Dist: <b>${item.cad_dist_ld !== null ? item.cad_dist_ld + ' LD' : 'N/A'}</b></span>`;

      return `
        <div class="alert-card ${isSentry ? 'sentry' : 'cad'}" onclick="openAsteroidDetail('${item.spkid}')">
          <div class="alert-card-top">
            <span class="alert-card-name">${item.full_name}</span>
            <span class="badge ${badgeClass}">${badgeLabel}</span>
          </div>
          <div class="alert-card-details">
            <span>MOID: <b>${item.moid} AU</b></span>
            <span>Diam: <b>${(item.diameter_km * 1000).toFixed(0)} m</b></span>
            ${subDetail}
          </div>
        </div>
      `;
    })
    .join("");
}

/* -------------------------------------------------------------
   Interactive Catalog Table
   ------------------------------------------------------------- */
function initCatalogTable() {
  loadCatalogData();

  // Search input debounce
  const searchInput = document.getElementById("catalog-search-input");
  let debounceTimer;
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        AppState.catalogFilters.search = e.target.value.trim();
        AppState.catalogPage = 1;
        loadCatalogData();
      }, 300);
    });
  }

  // Orbit Class Select
  const classSelect = document.getElementById("catalog-class-select");
  if (classSelect) {
    classSelect.addEventListener("change", (e) => {
      AppState.catalogFilters.orbitClass = e.target.value;
      AppState.catalogPage = 1;
      loadCatalogData();
    });
  }

  // Filter Buttons
  const btnPhaOnly = document.getElementById("filter-pha-only");
  if (btnPhaOnly) {
    btnPhaOnly.addEventListener("click", () => {
      AppState.catalogFilters.phaOnly = AppState.catalogFilters.phaOnly === true ? null : true;
      btnPhaOnly.classList.toggle("active", AppState.catalogFilters.phaOnly === true);
      AppState.catalogPage = 1;
      loadCatalogData();
    });
  }

  const btnSentryOnly = document.getElementById("filter-sentry-only");
  if (btnSentryOnly) {
    btnSentryOnly.addEventListener("click", () => {
      AppState.catalogFilters.sentryOnly = !AppState.catalogFilters.sentryOnly;
      btnSentryOnly.classList.toggle("active", AppState.catalogFilters.sentryOnly);
      AppState.catalogPage = 1;
      loadCatalogData();
    });
  }

  const btnCadOnly = document.getElementById("filter-cad-only");
  if (btnCadOnly) {
    btnCadOnly.addEventListener("click", () => {
      AppState.catalogFilters.cadOnly = !AppState.catalogFilters.cadOnly;
      btnCadOnly.classList.toggle("active", AppState.catalogFilters.cadOnly);
      AppState.catalogPage = 1;
      loadCatalogData();
    });
  }

  // Pagination Buttons
  const prevBtn = document.getElementById("catalog-prev-btn");
  const nextBtn = document.getElementById("catalog-next-btn");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (AppState.catalogPage > 1) {
        AppState.catalogPage--;
        loadCatalogData();
      }
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      AppState.catalogPage++;
      loadCatalogData();
    });
  }

  // Table Column Header Sorts
  document.querySelectorAll(".data-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const field = th.getAttribute("data-sort");
      if (AppState.catalogSortBy === field) {
        AppState.catalogOrder = AppState.catalogOrder === "asc" ? "desc" : "asc";
      } else {
        AppState.catalogSortBy = field;
        AppState.catalogOrder = "asc";
      }
      loadCatalogData();
    });
  });
}

async function loadCatalogData() {
  const tbody = document.getElementById("catalog-table-body");
  if (!tbody) return;

  tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 2rem; color: #94a3b8;"><i class="fa-solid fa-spinner fa-spin"></i> Querying asteroid database...</td></tr>`;

  try {
    const params = new URLSearchParams({
      page: AppState.catalogPage,
      limit: AppState.catalogLimit,
      sort_by: AppState.catalogSortBy,
      order: AppState.catalogOrder
    });

    if (AppState.catalogFilters.search) params.append("search", AppState.catalogFilters.search);
    if (AppState.catalogFilters.orbitClass !== "ALL") params.append("orbit_class", AppState.catalogFilters.orbitClass);
    if (AppState.catalogFilters.phaOnly !== null) params.append("is_pha", AppState.catalogFilters.phaOnly);
    if (AppState.catalogFilters.sentryOnly) params.append("sentry_only", "true");
    if (AppState.catalogFilters.cadOnly) params.append("cad_only", "true");

    const res = await fetch(`/api/asteroids?${params.toString()}`);
    if (!res.ok) throw new Error("Failed to load catalog");
    const data = await res.json();

    renderCatalogTable(data.records || []);
    updatePaginationUI(data);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #ef4444; padding: 2rem;">Error querying asteroid catalog: ${err.message}</td></tr>`;
  }
}

function renderCatalogTable(records) {
  const tbody = document.getElementById("catalog-table-body");
  if (!tbody) return;

  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 2rem; color: #64748b;">No asteroids matched your search filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = records
    .map((r) => {
      // Risk Badges
      let riskBadges = "";
      if (r.is_hazardous) {
        riskBadges += `<span class="badge badge-danger">PHA</span> `;
      } else {
        riskBadges += `<span class="badge badge-info">Nominal</span> `;
      }

      if (r.is_sentry) {
        riskBadges += `<span class="badge badge-warning" title="NASA Sentry Monitored">SENTRY</span> `;
      }
      if (r.has_cad) {
        riskBadges += `<span class="badge badge-success" title="Close Approach: ${r.cad_date}">CAD</span> `;
      }

      const diameterStr = r.diameter_km >= 1.0
        ? `${r.diameter_km.toFixed(2)} km`
        : `${(r.diameter_km * 1000).toFixed(0)} m`;

      return `
        <tr>
          <td class="asteroid-name-cell">${r.full_name}</td>
          <td><span class="badge badge-info">${r.class}</span></td>
          <td class="mono-cell">${r.H.toFixed(2)}</td>
          <td class="mono-cell">${diameterStr}</td>
          <td class="mono-cell ${r.moid <= 0.05 ? 'text-danger' : ''}">${r.moid.toFixed(4)} AU <span style="font-size: 0.7rem; color: #64748b;">(${r.moid_ld} LD)</span></td>
          <td class="mono-cell">${r.relative_velocity_kms} km/s</td>
          <td>${riskBadges}</td>
          <td>
            <div class="table-actions-cell">
              <button class="btn-table-action" onclick="openAsteroidDetail('${r.spkid}')" title="Inspect Full Parameters & ML Risk">
                <i class="fa-solid fa-magnifying-glass-chart"></i> Inspect
              </button>
              <button class="btn-table-action orbit-btn" onclick="plotAsteroidOrbit('${r.spkid}')" title="Plot 3D Trajectory">
                <i class="fa-solid fa-circle-nodes"></i> Orbit
              </button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

function updatePaginationUI(data) {
  const pageInfo = document.getElementById("catalog-pagination-info");
  const prevBtn = document.getElementById("catalog-prev-btn");
  const nextBtn = document.getElementById("catalog-next-btn");

  if (pageInfo) {
    pageInfo.innerText = `Page ${data.page} of ${data.total_pages} (${data.total.toLocaleString()} objects)`;
  }
  if (prevBtn) prevBtn.disabled = data.page <= 1;
  if (nextBtn) nextBtn.disabled = data.page >= data.total_pages;
}

/* -------------------------------------------------------------
   3D Keplerian Orbit Visualizer Integration
   ------------------------------------------------------------- */
function init3DVisualizer() {
  const container = document.getElementById("three-orbit-canvas");
  if (container && window.OrbitVisualizer3D) {
    AppState.orbitVisualizer = new OrbitVisualizer3D("three-orbit-canvas");
    // Load default Apophis trajectory
    loadOrbitTrajectory({ identifier: "99942", days_window: 180, step_days: 2 });
  }

  // Target object selector change
  const selectTarget = document.getElementById("orbit-target-select");
  if (selectTarget) {
    selectTarget.addEventListener("change", (e) => {
      const val = e.target.value;
      loadOrbitTrajectory({ identifier: val, days_window: 180, step_days: 2 });
    });
  }

  // Time window buttons
  document.querySelectorAll(".orbit-window-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".orbit-window-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const days = parseInt(btn.getAttribute("data-days"), 10);
      const sel = document.getElementById("orbit-target-select");
      const id = sel ? sel.value : "99942";
      loadOrbitTrajectory({ identifier: id, days_window: days, step_days: Math.max(1, Math.round(days / 90)) });
    });
  });
}

async function loadOrbitTrajectory(payload) {
  try {
    const res = await fetch("/api/orbit/propagate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Propagation failed");
    const data = await res.json();

    if (AppState.orbitVisualizer) {
      AppState.orbitVisualizer.loadTrajectory(data);
    }

    // Update orbit sidebar details
    const nameEl = document.getElementById("orbit-asteroid-name");
    if (nameEl) nameEl.innerText = data.asteroid_name || "Target Asteroid";

    const aEl = document.getElementById("orb-param-a");
    if (aEl) aEl.innerText = `${data.parameters.semi_major_axis_au} AU`;

    const eEl = document.getElementById("orb-param-e");
    if (eEl) eEl.innerText = data.parameters.eccentricity;

    const iEl = document.getElementById("orb-param-i");
    if (iEl) iEl.innerText = `${data.parameters.inclination_deg}°`;

    const qEl = document.getElementById("orb-param-q");
    if (qEl) qEl.innerText = `${data.parameters.perihelion_distance_au} AU`;

    const perEl = document.getElementById("orb-param-per");
    if (perEl) perEl.innerText = `${data.parameters.orbital_period_yr} yr`;

    const minEvt = data.closest_approach_in_window;
    const minTextEl = document.getElementById("orb-min-encounter");
    if (minTextEl && minEvt) {
      minTextEl.innerText = `${minEvt.date} (${minEvt.min_distance_au} AU / ${minEvt.min_distance_ld} LD)`;
      minTextEl.className = minEvt.min_distance_au <= 0.05 ? "tel-val text-danger" : "tel-val text-success";
    }
  } catch (err) {
    console.error("Error loading orbit trajectory:", err);
  }
}

function plotAsteroidOrbit(identifier) {
  switchTab("orbit");
  loadOrbitTrajectory({ identifier: String(identifier), days_window: 180, step_days: 2 });
}

/* -------------------------------------------------------------
   Planetary Defense Risk Predictor
   ------------------------------------------------------------- */
function initPredictionForm() {
  // Preset Chips
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const presetKey = chip.getAttribute("data-preset");
      if (PRESETS[presetKey]) {
        fillPredictionForm(PRESETS[presetKey]);
      }
    });
  });

  // Form Submit
  const form = document.getElementById("custom-prediction-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await runCustomPrediction();
    });
  }

  // Pre-fill with Apophis preset initially
  fillPredictionForm(PRESETS.apophis);
}

function fillPredictionForm(data) {
  document.getElementById("pred-h").value = data.H;
  document.getElementById("pred-diameter").value = data.diameter || "";
  document.getElementById("pred-a").value = data.a;
  document.getElementById("pred-e").value = data.e;
  document.getElementById("pred-i").value = data.i;
  document.getElementById("pred-moid").value = data.moid;
  document.getElementById("pred-class").value = data.orbit_class || "APO";
}

async function runCustomPrediction() {
  const btn = document.getElementById("btn-run-prediction");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Machine Learning Inference...`;
  }

  try {
    const payload = {
      H: parseFloat(document.getElementById("pred-h").value),
      diameter: document.getElementById("pred-diameter").value ? parseFloat(document.getElementById("pred-diameter").value) : null,
      a: parseFloat(document.getElementById("pred-a").value),
      e: parseFloat(document.getElementById("pred-e").value),
      i: parseFloat(document.getElementById("pred-i").value),
      moid: parseFloat(document.getElementById("pred-moid").value),
      orbit_class: document.getElementById("pred-class").value
    };

    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Inference failed");
    const data = await res.json();

    renderPredictionResults(data);
  } catch (err) {
    alert("Prediction error: " + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-brain"></i> Run Risk Inference`;
    }
  }
}

function renderPredictionResults(data) {
  const container = document.getElementById("prediction-results-area");
  if (!container) return;

  const pred = data.model_prediction;
  const auth = data.authoritative_status;

  const riskBadgeClass = pred.risk_badge === "critical" ? "badge-danger" : pred.risk_badge === "elevated" ? "badge-warning" : "badge-success";

  // Explanations list HTML
  const expHtml = data.explanations
    .map((exp) => {
      const isUp = exp.impact === "INCREASES_RISK";
      const icon = isUp ? "fa-arrow-trend-up text-danger" : "fa-shield-halved text-success";
      const impactClass = isUp ? "text-danger" : "text-success";
      return `
        <div class="explanation-card">
          <div class="explanation-top">
            <span style="font-weight: 600; font-size: 0.85rem; color: #fff;">
              <i class="fa-solid ${icon}"></i> ${exp.label} (${exp.value})
            </span>
            <span class="${impactClass}" style="font-size: 0.75rem; font-weight: 600;">${exp.impact.replace('_', ' ')}</span>
          </div>
          <div class="explanation-desc">${exp.description}</div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    <!-- Top Result Banner -->
    <div class="risk-score-banner" style="border-left: 4px solid ${pred.risk_color};">
      <div>
        <div style="font-size: 0.775rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Machine Learning Risk Assessment</div>
        <div class="risk-score-value" style="color: ${pred.risk_color};">${pred.hazard_percentage}%</div>
        <div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500; margin-top: 0.25rem;">
          Classification: <span class="badge ${riskBadgeClass}">${pred.risk_level}</span>
        </div>
      </div>
      <div style="text-align: right;">
        <span class="badge badge-info">${pred.uncertainty_level}</span>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem; max-width: 220px;">${pred.uncertainty_text}</div>
      </div>
    </div>

    <!-- Authoritative NASA Check -->
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1rem;">
      <div style="font-size: 0.8rem; font-weight: 600; color: #38bdf8; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <i class="fa-solid fa-satellite-dish"></i> Authoritative NASA/JPL Criteria Comparison
      </div>
      <div style="font-size: 0.8rem; color: #cbd5e1;">
        Status: <b>${auth.nasa_pha_criteria_met ? 'QUALIFIES AS AUTHORITATIVE PHA' : 'DOES NOT QUALIFY AS AUTHORITATIVE PHA'}</b>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
          • MOID <= 0.05 AU: <b>${auth.moid_threshold_met ? 'YES' : 'NO'}</b><br>
          • Magnitude H <= 22.0 (~140m): <b>${auth.magnitude_threshold_met ? 'YES' : 'NO'}</b>
        </div>
      </div>
    </div>

    <!-- Plain Language Summary -->
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 1rem;">
      <div style="font-size: 0.8rem; font-weight: 600; color: #818cf8; margin-bottom: 0.35rem;">
        <i class="fa-solid fa-file-waveform"></i> Plain-Language Analysis
      </div>
      <div style="font-size: 0.825rem; color: #e2e8f0; line-height: 1.5;">${data.plain_language_summary}</div>
    </div>

    <!-- Local Feature Explanations -->
    <div>
      <div style="font-size: 0.825rem; font-weight: 600; color: #fff; margin-bottom: 0.5rem;">
        <i class="fa-solid fa-list-check"></i> Local Feature Attribution Factors:
      </div>
      <div class="explanations-list">${expHtml}</div>
    </div>
  `;
}

/* -------------------------------------------------------------
   Exploratory Data Analysis (EDA) Charts
   ------------------------------------------------------------- */
async function loadEdaData() {
  try {
    const res = await fetch("/api/eda/stats");
    if (!res.ok) throw new Error("Failed to fetch EDA stats");
    const data = await res.json();
    AppState.edaLoaded = true;

    if (window.ChartsManager) {
      ChartsManager.renderMoidVsDiameter("chart-moid-diameter-scatter", data.scatter_sample);
      ChartsManager.renderOrbitClassDonut("chart-orbit-classes-donut", data.orbit_classes);
      ChartsManager.renderHistogram("chart-h-distribution", data.histograms.H, "Absolute Magnitude (H, mag)", "#6366f1");
      ChartsManager.renderHistogram("chart-vel-distribution", data.histograms.relative_velocity_kms, "Encounter Velocity (km/s)", "#38bdf8");
    }
  } catch (err) {
    console.error("Error loading EDA data:", err);
  }
}

/* -------------------------------------------------------------
   Model Benchmarks & Curves
   ------------------------------------------------------------- */
async function loadModelsData() {
  try {
    const [compRes, featRes] = await Promise.all([
      fetch("/api/models/comparison"),
      fetch("/api/models/feature-importance")
    ]);
    if (!compRes.ok || !featRes.ok) throw new Error("Failed to load model artifacts");
    const compData = await compRes.json();
    const featData = await featRes.json();
    AppState.modelsLoaded = true;

    // Render benchmark table
    renderBenchmarkTable(compData.models, compData.champion_model);

    if (window.ChartsManager) {
      ChartsManager.renderROCCurves("chart-roc-curves", compData.models);
      ChartsManager.renderPRCurves("chart-pr-curves", compData.models);
      ChartsManager.renderCalibration("chart-calibration-curves", compData.models);
      ChartsManager.renderFeatureImportance("chart-feature-importance", featData);
    }
  } catch (err) {
    console.error("Error loading model comparison data:", err);
  }
}

function renderBenchmarkTable(models, championKey) {
  const tbody = document.getElementById("models-benchmark-tbody");
  if (!tbody) return;

  tbody.innerHTML = Object.entries(models)
    .map(([key, m]) => {
      const isChamp = key === championKey;
      const champBadge = isChamp ? `<span class="badge badge-success">CHAMPION</span>` : "";
      return `
        <tr class="${isChamp ? 'champion-row' : ''}">
          <td><b>${m.name}</b> ${champBadge}</td>
          <td class="mono-cell">${(m.metrics.accuracy * 100).toFixed(2)}%</td>
          <td class="mono-cell">${(m.metrics.precision * 100).toFixed(2)}%</td>
          <td class="mono-cell">${(m.metrics.recall * 100).toFixed(2)}%</td>
          <td class="mono-cell">${m.metrics.f1_score.toFixed(4)}</td>
          <td class="mono-cell text-primary"><b>${m.metrics.roc_auc.toFixed(4)}</b></td>
          <td class="mono-cell text-accent"><b>${m.metrics.pr_auc.toFixed(4)}</b></td>
          <td class="mono-cell">${m.metrics.brier_score.toFixed(4)}</td>
          <td class="mono-cell">TP:${m.confusion_matrix.tp} FP:${m.confusion_matrix.fp} FN:${m.confusion_matrix.fn}</td>
        </tr>
      `;
    })
    .join("");
}

/* -------------------------------------------------------------
   Asteroid Detail Modal / Drawer
   ------------------------------------------------------------- */
function initDetailModal() {
  const modal = document.getElementById("asteroid-detail-modal");
  const closeBtn = document.getElementById("btn-close-modal");
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => modal.classList.remove("active"));
  }
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("active");
    });
  }
}

async function openAsteroidDetail(identifier) {
  const modal = document.getElementById("asteroid-detail-modal");
  const modalBody = document.getElementById("modal-detail-body");
  const modalTitle = document.getElementById("modal-asteroid-title");

  if (!modal || !modalBody) return;
  modal.classList.add("active");
  modalBody.innerHTML = `<div style="text-align: center; padding: 3rem; color: #94a3b8;"><i class="fa-solid fa-spinner fa-spin fa-2x"></i><br><br>Retrieving complete physical, orbital, and risk telemetry...</div>`;

  try {
    const res = await fetch(`/api/asteroids/${identifier}`);
    if (!res.ok) throw new Error("Asteroid not found");
    const data = await res.json();

    const info = data.asteroid_info;
    const ml = data.ml_analysis;
    const pred = ml.model_prediction;

    if (modalTitle) modalTitle.innerText = `${info.full_name} (${info.class})`;

    const isSentry = info.is_sentry_monitored === 1;
    const hasCad = info.has_cad_approach === 1;

    modalBody.innerHTML = `
      <!-- Actions Banner -->
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
        <div style="display: flex; gap: 0.5rem; align-items: center;">
          <span class="badge ${info.is_hazardous ? 'badge-danger' : 'badge-info'}">${info.is_hazardous ? 'AUTHORITATIVE PHA' : 'NOMINAL NEA'}</span>
          ${isSentry ? '<span class="badge badge-warning">SENTRY IMPACT MONITORING</span>' : ''}
          ${hasCad ? '<span class="badge badge-success">CAD FLYBY</span>' : ''}
        </div>
        <button class="btn-primary-gradient" style="width: auto; padding: 0.5rem 1rem; margin: 0;" onclick="plotAsteroidOrbit('${info.spkid}'); document.getElementById('asteroid-detail-modal').classList.remove('active');">
          <i class="fa-solid fa-circle-nodes"></i> Plot in 3D Orbit Visualizer
        </button>
      </div>

      <!-- ML Risk Score Card -->
      <div class="risk-score-banner" style="border-left: 4px solid ${pred.risk_color};">
        <div>
          <div style="font-size: 0.775rem; color: #94a3b8; text-transform: uppercase;">Machine Learning Risk Model</div>
          <div class="risk-score-value" style="color: ${pred.risk_color};">${pred.hazard_percentage}%</div>
          <div style="font-size: 0.85rem; color: #cbd5e1;">Classified: <b>${pred.risk_level}</b></div>
        </div>
        <div style="text-align: right;">
          <span class="badge badge-info">${pred.uncertainty_level}</span>
          <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem; max-width: 250px;">${pred.uncertainty_text}</div>
        </div>
      </div>

      <!-- Sentry Impact Risk Data if present -->
      ${isSentry ? `
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 1.25rem;">
          <div style="font-weight: 600; color: #f87171; font-size: 0.9rem; margin-bottom: 0.5rem;">
            <i class="fa-solid fa-satellite-dish"></i> NASA JPL Sentry Impact Risk Evaluation
          </div>
          <div class="param-grid-4col">
            <div class="param-card"><div class="param-title">Palermo Scale (Max)</div><div class="param-val">${info.sentry_ps_max !== null ? info.sentry_ps_max : 'N/A'}</div></div>
            <div class="param-card"><div class="param-title">Torino Scale</div><div class="param-val">${info.sentry_ts_max}</div></div>
            <div class="param-card"><div class="param-title">Impact Probability</div><div class="param-val">${info.sentry_ip ? info.sentry_ip.toExponential(2) : '0'}</div></div>
            <div class="param-card"><div class="param-title">Potential Impact Years</div><div class="param-val">${info.sentry_range || 'N/A'}</div></div>
          </div>
        </div>
      ` : ''}

      <!-- Close Approach Data if present -->
      ${hasCad ? `
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 1.25rem;">
          <div style="font-weight: 600; color: #fbbf24; font-size: 0.9rem; margin-bottom: 0.5rem;">
            <i class="fa-solid fa-bullseye"></i> NASA JPL Close Approach Geometry
          </div>
          <div class="param-grid-4col">
            <div class="param-card"><div class="param-title">Approach Date</div><div class="param-val">${info.cad_next_date}</div></div>
            <div class="param-card"><div class="param-title">Nominal Distance (AU)</div><div class="param-val">${info.cad_dist_au} AU</div></div>
            <div class="param-card"><div class="param-title">Distance (Lunar Distances)</div><div class="param-val text-warning">${info.cad_dist_ld} LD</div></div>
            <div class="param-card"><div class="param-title">Relative Velocity</div><div class="param-val">${info.cad_v_rel} km/s</div></div>
          </div>
        </div>
      ` : ''}

      <!-- Physical & Orbital Parameters Grid -->
      <div>
        <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.6rem; color: #fff;">
          <i class="fa-solid fa-atom"></i> Physical & Orbital Parameters
        </div>
        <div class="param-grid-4col">
          <div class="param-card"><div class="param-title">Earth MOID</div><div class="param-val ${info.moid <= 0.05 ? 'text-danger' : ''}">${info.moid} AU</div></div>
          <div class="param-card"><div class="param-title">Absolute Magnitude (H)</div><div class="param-val">${info.H} mag</div></div>
          <div class="param-card"><div class="param-title">Estimated Diameter</div><div class="param-val">${info.diameter_estimated_km >= 1 ? info.diameter_estimated_km.toFixed(2) + ' km' : (info.diameter_estimated_km * 1000).toFixed(0) + ' m'}</div></div>
          <div class="param-card"><div class="param-title">Orbit Class</div><div class="param-val">${info.class}</div></div>
          <div class="param-card"><div class="param-title">Semi-Major Axis (a)</div><div class="param-val">${info.a} AU</div></div>
          <div class="param-card"><div class="param-title">Eccentricity (e)</div><div class="param-val">${info.e}</div></div>
          <div class="param-card"><div class="param-title">Inclination (i)</div><div class="param-val">${info.i}°</div></div>
          <div class="param-card"><div class="param-title">Perihelion (q)</div><div class="param-val">${info.q} AU</div></div>
          <div class="param-card"><div class="param-title">Aphelion (Q)</div><div class="param-val">${info.ad} AU</div></div>
          <div class="param-card"><div class="param-title">Period (per_y)</div><div class="param-val">${info.per_y ? info.per_y.toFixed(2) : 'N/A'} yr</div></div>
          <div class="param-card"><div class="param-title">Rel Velocity (Proxy)</div><div class="param-val">${info.relative_velocity_kms} km/s</div></div>
          <div class="param-card"><div class="param-title">Kinetic Energy (Est)</div><div class="param-val">${info.kinetic_energy_mt ? Number(info.kinetic_energy_mt).toLocaleString(undefined, {maximumFractionDigits: 1}) : '0'} MT</div></div>
        </div>
      </div>

      <!-- Plain Language & Scientific Limitations -->
      <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1.25rem;">
        <div style="font-weight: 600; color: #818cf8; font-size: 0.85rem; margin-bottom: 0.4rem;">Analysis & Uncertainty Statement</div>
        <div style="font-size: 0.825rem; color: #cbd5e1; line-height: 1.6;">${ml.plain_language_summary}</div>
      </div>
    `;
  } catch (err) {
    modalBody.innerHTML = `<div style="color: #ef4444; padding: 2rem; text-align: center;">Error loading asteroid details: ${err.message}</div>`;
  }
}

/* -------------------------------------------------------------
   Data Refresh Control
   ------------------------------------------------------------- */
function initDataRefresh() {
  const btn = document.getElementById("btn-refresh-data");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.classList.add("spinning");
    btn.disabled = true;

    try {
      const res = await fetch("/api/refresh-data", { method: "POST" });
      const data = await res.json();
      alert(data.message || "Live data sync triggered! Models and catalog updating in background.");
      // Reload overview metrics after 3 seconds
      setTimeout(() => {
        initSummaryMetrics();
        loadCatalogData();
      }, 3000);
    } catch (err) {
      alert("Refresh request error: " + err.message);
    } finally {
      setTimeout(() => {
        btn.classList.remove("spinning");
        btn.disabled = false;
      }, 1500);
    }
  });
}

// Global exposure
window.openAsteroidDetail = openAsteroidDetail;
window.plotAsteroidOrbit = plotAsteroidOrbit;
