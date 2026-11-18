/**
 * Asteroid Trajectory & Earth Risk Analysis - Scientific Charting Suite
 * Built on Plotly.js with dark SaaS styling
 */

const darkPlotLayout = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "rgba(11, 17, 32, 0.4)",
  font: {
    family: "Inter, sans-serif",
    color: "#94a3b8",
    size: 11
  },
  margin: { l: 50, r: 25, t: 30, b: 45 },
  xaxis: {
    gridcolor: "rgba(255, 255, 255, 0.06)",
    zerolinecolor: "rgba(255, 255, 255, 0.12)",
    tickfont: { color: "#94a3b8" }
  },
  yaxis: {
    gridcolor: "rgba(255, 255, 255, 0.06)",
    zerolinecolor: "rgba(255, 255, 255, 0.12)",
    tickfont: { color: "#94a3b8" }
  },
  legend: {
    font: { color: "#cbd5e1" },
    bgcolor: "rgba(17, 26, 46, 0.8)",
    bordercolor: "rgba(255, 255, 255, 0.08)",
    borderwidth: 1
  }
};

const plotlyConfig = {
  responsive: true,
  displayModeBar: false
};

const ChartsManager = {
  /**
   * Render MOID vs Diameter scatter plot (Core PHA classification space)
   */
  renderMoidVsDiameter(containerId, scatterData) {
    if (!document.getElementById(containerId) || !scatterData) return;

    const phaX = [];
    const phaY = [];
    const phaNames = [];

    const nonPhaX = [];
    const nonPhaY = [];
    const nonPhaNames = [];

    scatterData.forEach((item) => {
      if (item.is_hazardous === 1) {
        phaX.push(item.moid);
        phaY.push(item.diameter_estimated_km);
        phaNames.push(`${item.full_name}<br>MOID: ${item.moid} AU<br>Diam: ${item.diameter_estimated_km} km`);
      } else {
        nonPhaX.push(item.moid);
        nonPhaY.push(item.diameter_estimated_km);
        nonPhaNames.push(`${item.full_name}<br>MOID: ${item.moid} AU<br>Diam: ${item.diameter_estimated_km} km`);
      }
    });

    const tracePha = {
      x: phaX,
      y: phaY,
      text: phaNames,
      mode: "markers",
      type: "scatter",
      name: "Potentially Hazardous (PHA)",
      marker: {
        color: "#ef4444",
        size: 6,
        opacity: 0.8
      }
    };

    const traceNonPha = {
      x: nonPhaX,
      y: nonPhaY,
      text: nonPhaNames,
      mode: "markers",
      type: "scatter",
      name: "Nominal NEA",
      marker: {
        color: "#38bdf8",
        size: 5,
        opacity: 0.5
      }
    };

    const layout = {
      ...darkPlotLayout,
      xaxis: {
        ...darkPlotLayout.xaxis,
        title: "Earth MOID (Astronomical Units, AU)",
        range: [0, 0.5]
      },
      yaxis: {
        ...darkPlotLayout.yaxis,
        title: "Estimated Diameter (km, log scale)",
        type: "log"
      },
      shapes: [
        // 0.05 AU critical MOID boundary line
        {
          type: "line",
          x0: 0.05,
          x1: 0.05,
          y0: 0.001,
          y1: 50,
          line: {
            color: "#f59e0b",
            width: 1.5,
            dash: "dash"
          }
        },
        // 140 meter (0.14 km) critical diameter line
        {
          type: "line",
          x0: 0,
          x1: 0.5,
          y0: 0.14,
          y1: 0.14,
          line: {
            color: "#f59e0b",
            width: 1.5,
            dash: "dash"
          }
        }
      ],
      annotations: [
        {
          x: 0.05,
          y: Math.log10(15),
          text: "MOID <= 0.05 AU",
          showarrow: true,
          arrowhead: 2,
          arrowcolor: "#f59e0b",
          font: { color: "#f59e0b", size: 10 }
        },
        {
          x: 0.35,
          y: Math.log10(0.14),
          text: "D >= 140m (H <= 22)",
          showarrow: true,
          arrowhead: 2,
          arrowcolor: "#f59e0b",
          font: { color: "#f59e0b", size: 10 }
        }
      ]
    };

    Plotly.newPlot(containerId, [traceNonPha, tracePha], layout, plotlyConfig);
  },

  /**
   * Render Orbit Classes Donut Chart
   */
  renderOrbitClassDonut(containerId, classCounts) {
    if (!document.getElementById(containerId) || !classCounts) return;

    const labels = Object.keys(classCounts);
    const values = Object.values(classCounts);

    const data = [{
      labels: labels,
      values: values,
      type: "pie",
      hole: 0.6,
      marker: {
        colors: ["#6366f1", "#38bdf8", "#ec4899", "#f59e0b", "#10b981"]
      },
      textinfo: "label+percent",
      hoverinfo: "label+value+percent"
    }];

    const layout = {
      ...darkPlotLayout,
      margin: { l: 20, r: 20, t: 20, b: 20 },
      showlegend: true
    };

    Plotly.newPlot(containerId, data, layout, plotlyConfig);
  },

  /**
   * Render Histogram (e.g. H Magnitude or Velocity)
   */
  renderHistogram(containerId, histObj, xTitle, barColor = "#6366f1") {
    if (!document.getElementById(containerId) || !histObj) return;

    const data = [{
      x: histObj.bin_labels,
      y: histObj.counts,
      type: "bar",
      marker: {
        color: barColor,
        opacity: 0.85
      }
    }];

    const layout = {
      ...darkPlotLayout,
      xaxis: {
        ...darkPlotLayout.xaxis,
        title: xTitle,
        tickangle: -35
      },
      yaxis: {
        ...darkPlotLayout.yaxis,
        title: "Asteroid Count"
      }
    };

    Plotly.newPlot(containerId, data, layout, plotlyConfig);
  },

  /**
   * Render Multi-Model ROC Curves
   */
  renderROCCurves(containerId, models) {
    if (!document.getElementById(containerId) || !models) return;

    const colors = {
      Logistic_Regression: "#94a3b8",
      Random_Forest: "#10b981",
      Gradient_Boosting: "#38bdf8",
      Deep_Learning_MLP: "#a855f7"
    };

    const traces = [];

    // Diagonal random baseline
    traces.push({
      x: [0, 1],
      y: [0, 1],
      mode: "lines",
      name: "Random Chance (AUC = 0.50)",
      line: { color: "rgba(255, 255, 255, 0.2)", dash: "dash" }
    });

    for (const [key, model] of Object.entries(models)) {
      if (!model.roc_curve) continue;
      const aucVal = model.metrics.roc_auc.toFixed(4);
      traces.push({
        x: model.roc_curve.fpr,
        y: model.roc_curve.tpr,
        mode: "lines",
        name: `${model.name} (AUC: ${aucVal})`,
        line: {
          color: colors[key] || "#6366f1",
          width: key === "Gradient_Boosting" ? 3 : 1.8
        }
      });
    }

    const layout = {
      ...darkPlotLayout,
      xaxis: { ...darkPlotLayout.xaxis, title: "False Positive Rate (1 - Specificity)", range: [0, 1] },
      yaxis: { ...darkPlotLayout.yaxis, title: "True Positive Rate (Recall / Sensitivity)", range: [0, 1.02] }
    };

    Plotly.newPlot(containerId, traces, layout, plotlyConfig);
  },

  /**
   * Render Multi-Model Precision-Recall Curves
   */
  renderPRCurves(containerId, models) {
    if (!document.getElementById(containerId) || !models) return;

    const colors = {
      Logistic_Regression: "#94a3b8",
      Random_Forest: "#10b981",
      Gradient_Boosting: "#38bdf8",
      Deep_Learning_MLP: "#a855f7"
    };

    const traces = [];

    for (const [key, model] of Object.entries(models)) {
      if (!model.pr_curve) continue;
      const prAuc = model.metrics.pr_auc.toFixed(4);
      traces.push({
        x: model.pr_curve.recall,
        y: model.pr_curve.precision,
        mode: "lines",
        name: `${model.name} (PR-AUC: ${prAuc})`,
        line: {
          color: colors[key] || "#6366f1",
          width: key === "Gradient_Boosting" ? 3 : 1.8
        }
      });
    }

    const layout = {
      ...darkPlotLayout,
      xaxis: { ...darkPlotLayout.xaxis, title: "Recall (Hazardous Detection Rate)", range: [0, 1] },
      yaxis: { ...darkPlotLayout.yaxis, title: "Precision (Positive Predictive Value)", range: [0.7, 1.02] }
    };

    Plotly.newPlot(containerId, traces, layout, plotlyConfig);
  },

  /**
   * Render Calibration Curves (Reliability Diagram)
   */
  renderCalibration(containerId, models) {
    if (!document.getElementById(containerId) || !models) return;

    const colors = {
      Logistic_Regression: "#94a3b8",
      Random_Forest: "#10b981",
      Gradient_Boosting: "#38bdf8",
      Deep_Learning_MLP: "#a855f7"
    };

    const traces = [];

    // Perfectly calibrated diagonal
    traces.push({
      x: [0, 1],
      y: [0, 1],
      mode: "lines",
      name: "Perfect Calibration",
      line: { color: "rgba(255, 255, 255, 0.25)", dash: "dot" }
    });

    for (const [key, model] of Object.entries(models)) {
      if (!model.calibration_curve) continue;
      traces.push({
        x: model.calibration_curve.mean_predicted_value,
        y: model.calibration_curve.fraction_of_positives,
        mode: "lines+markers",
        name: model.name,
        line: { color: colors[key] || "#6366f1", width: 2 },
        marker: { size: 6 }
      });
    }

    const layout = {
      ...darkPlotLayout,
      xaxis: { ...darkPlotLayout.xaxis, title: "Mean Predicted Probability", range: [0, 1] },
      yaxis: { ...darkPlotLayout.yaxis, title: "Observed Fraction of Positives", range: [0, 1] }
    };

    Plotly.newPlot(containerId, traces, layout, plotlyConfig);
  },

  /**
   * Render Global Feature Importances
   */
  renderFeatureImportance(containerId, impData) {
    if (!document.getElementById(containerId) || !impData || !impData.importances) return;

    const topItems = impData.importances.slice(0, 10).reverse();
    const yFeatures = topItems.map((item) => item.feature);
    const xWeights = topItems.map((item) => item.importance);
    const hoverTexts = topItems.map((item) => {
      const desc = impData.feature_descriptions[item.feature] || item.feature;
      return `${item.feature}: ${item.importance}<br>${desc}`;
    });

    const data = [{
      y: yFeatures,
      x: xWeights,
      text: hoverTexts,
      type: "bar",
      orientation: "h",
      marker: {
        color: xWeights.map((w) => (w > 0.1 ? "#6366f1" : "#38bdf8")),
        opacity: 0.9
      }
    }];

    const layout = {
      ...darkPlotLayout,
      margin: { l: 150, r: 25, t: 20, b: 40 },
      xaxis: { ...darkPlotLayout.xaxis, title: "Normalized Feature Importance Weight" },
      yaxis: { ...darkPlotLayout.yaxis, automargin: true }
    };

    Plotly.newPlot(containerId, data, layout, plotlyConfig);
  }
};

window.ChartsManager = ChartsManager;
