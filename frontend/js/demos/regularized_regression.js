(function () {
  // ── Constants ───────────────────────────────────────────────────────────
  const C = {
    train:        '#4682b4',
    val:          '#e07b39',
    best:         '#2ca02c',
    coarseOpacity: 0.35,
    coarseSize:    7,
    fineSize:      5,
  };

  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  // ── State ───────────────────────────────────────────────────────────────
  const state = {
    mode:          'manual',      // 'manual' | 'systematic'
    grid:          'coarse',      // 'coarse' | 'fine'
    visitedIdx:    new Set(),     // alpha indices already plotted as dots
    currentIdx:    0,             // current slider position (0–10)
    initData:      null,
    fineData:      null,
    fineLoading:   false,
    chartsReady:   { pearson: false, ridgeErr: false, lassoErr: false,
                     ridgeBar: false, lassoBar: false,
                     ridgePath: false, lassoPaths: false },
  };

  // ── DOM refs ────────────────────────────────────────────────────────────
  const alphaSlider    = document.getElementById('rr-alpha');
  const alphaVal       = document.getElementById('rr-alpha-val');
  const ridgeRmse      = document.getElementById('rr-ridge-rmse');
  const lassoRmse      = document.getElementById('rr-lasso-rmse');
  const gridHint       = document.getElementById('rr-grid-hint');
  const systematicBtn  = document.getElementById('rr-systematic-btn');
  const panelGrid      = document.getElementById('rr-panel-grid');
  const coarseBtn      = document.getElementById('rr-coarse-btn');
  const fineBtn        = document.getElementById('rr-fine-btn');
  const testBtn        = document.getElementById('rr-test-btn');
  const panelTest      = document.getElementById('rr-panel-test');
  const ridgeTestStat  = document.getElementById('rr-ridge-test-stat');
  const lassoTestStat  = document.getElementById('rr-lasso-test-stat');

  // ── Helpers ─────────────────────────────────────────────────────────────
  function idxToAlpha(idx) {
    return (idx / 10).toFixed(2);
  }

  function rmseHtml(train, val) {
    return `Train RMSE: <span style="color:${C.train}"><strong>${train.toFixed(2)}</strong></span>
            &nbsp;·&nbsp;
            Val RMSE: <span style="color:${C.val}"><strong>${val.toFixed(2)}</strong></span>`;
  }

  // ── Layout builders ─────────────────────────────────────────────────────
  function errorLayout(title) {
    return {
      title: { text: title, font: { size: 13 } },
      xaxis: { title: 'α', range: [-0.005, 0.105] },
      yaxis: { title: 'RMSE' },
      margin: { t: 36, r: 12, b: 44, l: 52 },
      legend: { orientation: 'h', y: -0.2 },
      height: 240,
      shapes: [],
    };
  }

  function coefBarLayout(title, featureNames) {
    return {
      title: { text: title, font: { size: 12 } },
      xaxis: { title: 'Coefficient', zeroline: true },
      yaxis: { tickfont: { size: 10 } },
      margin: { t: 32, r: 12, b: 36, l: 38 },
      height: 200,
    };
  }

  function coefPathLayout(title) {
    return {
      title: { text: title, font: { size: 12 } },
      xaxis: { title: 'α' },
      yaxis: { title: 'Coefficient', zeroline: true, zerolinecolor: '#999' },
      margin: { t: 32, r: 12, b: 44, l: 52 },
      legend: { font: { size: 9 }, x: 1, xanchor: 'right', y: 1 },
      height: 260,
      shapes: [],
    };
  }

  // ── Best-α vline shape ──────────────────────────────────────────────────
  function bestAlphaShape(alpha) {
    return {
      type: 'line',
      x0: alpha, x1: alpha,
      y0: 0, y1: 1, yref: 'paper',
      line: { color: C.best, dash: 'dash', width: 1.5 },
    };
  }

  // ── Chart initialization ─────────────────────────────────────────────────

  function renderPearsonChart() {
    const d = state.initData;
    // Sort features by |r| descending for display
    const order = d.feature_names
      .map((_, i) => i)
      .sort((a, b) => Math.abs(d.pearson_r[b]) - Math.abs(d.pearson_r[a]));

    const names  = order.map(i => d.feature_names[i]);
    const rs     = order.map(i => d.pearson_r[i]);
    const colors = rs.map(r => r >= 0 ? C.train : C.val);

    Plotly.newPlot('rr-pearson-chart', [{
      type: 'bar', orientation: 'h',
      x: rs, y: names,
      marker: { color: colors },
      hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
    }], {
      xaxis: { title: 'Pearson r', range: [-1, 1], zeroline: true, zerolinecolor: '#999' },
      yaxis: { tickfont: { size: 10 } },
      margin: { t: 16, r: 12, b: 44, l: 46 },
      height: 240,
      showlegend: false,
    }, PLOTLY_CFG);

    state.chartsReady.pearson = true;
  }

  function initErrorCharts() {
    const emptyTraces = () => [
      { x: [], y: [], mode: 'markers', name: 'train',
        marker: { color: C.train, size: 10 }, hovertemplate: 'α=%{x:.2f}<br>RMSE=%{y:.2f}<extra>train</extra>' },
      { x: [], y: [], mode: 'markers', name: 'val',
        marker: { color: C.val,   size: 10 }, hovertemplate: 'α=%{x:.2f}<br>RMSE=%{y:.2f}<extra>val</extra>' },
    ];
    Plotly.newPlot('rr-ridge-error', emptyTraces(), errorLayout('Ridge (L2) — validation error'), PLOTLY_CFG);
    Plotly.newPlot('rr-lasso-error', emptyTraces(), errorLayout('Lasso (L1) — validation error'), PLOTLY_CFG);
    state.chartsReady.ridgeErr = true;
    state.chartsReady.lassoErr = true;
  }

  function initCoefBarCharts() {
    const d = state.initData;
    const coefBarTraces = (coefs, featureNames) => [{
      type: 'bar', orientation: 'h',
      x: coefs, y: featureNames,
      marker: { color: coefs.map(v => v >= 0 ? C.train : C.val) },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    }];

    Plotly.newPlot('rr-ridge-coef-bar',
      coefBarTraces(d.ridge.coef_paths[0], d.feature_names),
      coefBarLayout('Ridge coefficients at current α', d.feature_names),
      PLOTLY_CFG);
    Plotly.newPlot('rr-lasso-coef-bar',
      coefBarTraces(d.lasso.coef_paths[0], d.feature_names),
      coefBarLayout('Lasso coefficients at current α', d.feature_names),
      PLOTLY_CFG);

    state.chartsReady.ridgeBar = true;
    state.chartsReady.lassoBar = true;
  }

  // ── Manual mode updates ─────────────────────────────────────────────────

  function addManualDot(idx) {
    if (state.visitedIdx.has(idx)) return;
    state.visitedIdx.add(idx);

    const d    = state.initData;
    const a    = d.alphas[idx];
    const rTr  = d.ridge.train_rmse[idx];
    const rVa  = d.ridge.val_rmse[idx];
    const lTr  = d.lasso.train_rmse[idx];
    const lVa  = d.lasso.val_rmse[idx];

    Plotly.extendTraces('rr-ridge-error', { x: [[a], [a]], y: [[rTr], [rVa]] }, [0, 1]);
    Plotly.extendTraces('rr-lasso-error', { x: [[a], [a]], y: [[lTr], [lVa]] }, [0, 1]);

    if (state.visitedIdx.size >= 3) {
      gridHint.classList.remove('hidden');
    }
  }

  function updateCoefBars(idx) {
    const d = state.initData;
    const rCoefs = d.ridge.coef_paths[idx];
    const lCoefs = d.lasso.coef_paths[idx];

    const update = (id, coefs) => Plotly.react(id, [{
      type: 'bar', orientation: 'h',
      x: coefs, y: d.feature_names,
      marker: { color: coefs.map(v => v >= 0 ? C.train : C.val) },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    }], coefBarLayout('', d.feature_names), PLOTLY_CFG);

    update('rr-ridge-coef-bar', rCoefs);
    update('rr-lasso-coef-bar', lCoefs);
  }

  function updateRmseText(idx) {
    const d = state.initData;
    ridgeRmse.innerHTML = rmseHtml(d.ridge.train_rmse[idx], d.ridge.val_rmse[idx]);
    lassoRmse.innerHTML = rmseHtml(d.lasso.train_rmse[idx], d.lasso.val_rmse[idx]);
  }

  // ── Systematic mode ─────────────────────────────────────────────────────

  function enterSystematicMode() {
    state.mode = 'systematic';
    alphaSlider.disabled = true;
    systematicBtn.disabled = true;
    systematicBtn.textContent = 'Grid search active';

    renderCoarseGridLines();
    panelGrid.classList.remove('hidden');
    renderCoefPaths('coarse');

    // Update coef bars to show best-α coefficients
    const d = state.initData;
    updateCoefBarsAtBest(d, d.ridge.best_alpha_idx, d.lasso.best_alpha_idx);
    ridgeRmse.innerHTML = rmseHtml(d.ridge.train_rmse[d.ridge.best_alpha_idx],
                                   d.ridge.val_rmse[d.ridge.best_alpha_idx]);
    lassoRmse.innerHTML = rmseHtml(d.lasso.train_rmse[d.lasso.best_alpha_idx],
                                   d.lasso.val_rmse[d.lasso.best_alpha_idx]);
  }

  function updateCoefBarsAtBest(data, ridgeBestIdx, lassoBestIdx) {
    const rCoefs = data.ridge.coef_paths[ridgeBestIdx];
    const lCoefs = data.lasso.coef_paths[lassoBestIdx];
    const names  = state.initData.feature_names;

    const mkTrace = (coefs) => [{
      type: 'bar', orientation: 'h',
      x: coefs, y: names,
      marker: { color: coefs.map(v => v >= 0 ? C.train : C.val) },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    }];

    Plotly.react('rr-ridge-coef-bar', mkTrace(rCoefs),
      coefBarLayout(`Ridge coefs at best α = ${data.alphas[ridgeBestIdx].toFixed(3)}`, names), PLOTLY_CFG);
    Plotly.react('rr-lasso-coef-bar', mkTrace(lCoefs),
      coefBarLayout(`Lasso coefs at best α = ${data.alphas[lassoBestIdx].toFixed(3)}`, names), PLOTLY_CFG);
  }

  function coarseGridTraces(data) {
    const a    = data.alphas;
    const ridge = data.ridge;
    const lasso = data.lasso;
    return {
      ridgeErr: [
        { x: a, y: ridge.train_rmse, mode: 'lines+markers', name: 'train',
          line: { color: C.train }, marker: { color: C.train, size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train</extra>' },
        { x: a, y: ridge.val_rmse,   mode: 'lines+markers', name: 'val',
          line: { color: C.val },   marker: { color: C.val,   size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val</extra>' },
      ],
      lassoErr: [
        { x: a, y: lasso.train_rmse, mode: 'lines+markers', name: 'train',
          line: { color: C.train }, marker: { color: C.train, size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train</extra>' },
        { x: a, y: lasso.val_rmse,   mode: 'lines+markers', name: 'val',
          line: { color: C.val },   marker: { color: C.val,   size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val</extra>' },
      ],
    };
  }

  function renderCoarseGridLines() {
    const d = state.initData;
    const traces = coarseGridTraces(d);

    const ridgeBestA = d.alphas[d.ridge.best_alpha_idx];
    const lassoBestA = d.alphas[d.lasso.best_alpha_idx];

    const ridgeLayout = Object.assign({}, errorLayout('Ridge (L2) — error curve'), {
      shapes: [bestAlphaShape(ridgeBestA)],
      annotations: [{ x: ridgeBestA, y: d.ridge.best_val_rmse, xref: 'x', yref: 'y',
        text: `best α=${ridgeBestA.toFixed(3)}<br>val=${d.ridge.best_val_rmse.toFixed(2)}`,
        showarrow: true, arrowhead: 2, ax: 40, ay: -30, font: { size: 10 } }],
    });
    const lassoLayout = Object.assign({}, errorLayout('Lasso (L1) — error curve'), {
      shapes: [bestAlphaShape(lassoBestA)],
      annotations: [{ x: lassoBestA, y: d.lasso.best_val_rmse, xref: 'x', yref: 'y',
        text: `best α=${lassoBestA.toFixed(3)}<br>val=${d.lasso.best_val_rmse.toFixed(2)}`,
        showarrow: true, arrowhead: 2, ax: 40, ay: -30, font: { size: 10 } }],
    });

    Plotly.react('rr-ridge-error', traces.ridgeErr, ridgeLayout, PLOTLY_CFG);
    Plotly.react('rr-lasso-error', traces.lassoErr, lassoLayout, PLOTLY_CFG);
  }

  // ── Fine grid ────────────────────────────────────────────────────────────

  async function loadFine() {
    if (state.fineData) return;
    if (state.fineLoading) return;
    state.fineLoading = true;
    fineBtn.textContent = 'Loading…';
    fineBtn.disabled = true;
    try {
      const res = await fetch('/api/regularized_regression/fine');
      state.fineData = await res.json();
      renderFineGrid();
    } finally {
      fineBtn.textContent = 'Fine (101 steps)';
      fineBtn.disabled = false;
    }
  }

  function fineGridTraces(data) {
    const a     = data.alphas;
    const ridge = data.ridge;
    const lasso = data.lasso;
    return {
      ridgeErr: [
        { x: a, y: ridge.train_rmse, mode: 'lines+markers', name: 'train (fine)',
          line: { color: C.train, width: 2 }, marker: { color: C.train, size: C.fineSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train (fine)</extra>' },
        { x: a, y: ridge.val_rmse, mode: 'lines+markers', name: 'val (fine)',
          line: { color: C.val, width: 2 }, marker: { color: C.val, size: C.fineSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val (fine)</extra>' },
      ],
      lassoErr: [
        { x: a, y: lasso.train_rmse, mode: 'lines+markers', name: 'train (fine)',
          line: { color: C.train, width: 2 }, marker: { color: C.train, size: C.fineSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train (fine)</extra>' },
        { x: a, y: lasso.val_rmse, mode: 'lines+markers', name: 'val (fine)',
          line: { color: C.val, width: 2 }, marker: { color: C.val, size: C.fineSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val (fine)</extra>' },
      ],
    };
  }

  function renderFineGrid() {
    const fd = state.fineData;
    const cd = state.initData;

    const cTraces = coarseGridTraces(cd);
    const fTraces = fineGridTraces(fd);

    // Dim coarse traces; render both sets simultaneously
    const dimTrace = (t) => Object.assign({}, t, {
      opacity: C.coarseOpacity,
      marker: Object.assign({}, t.marker, { size: C.coarseSize }),
      showlegend: false,
    });

    const ridgeBestA = fd.alphas[fd.ridge.best_alpha_idx];
    const lassoBestA = fd.alphas[fd.lasso.best_alpha_idx];

    const ridgeLayout = Object.assign({}, errorLayout('Ridge (L2) — error curve'), {
      shapes: [bestAlphaShape(ridgeBestA)],
      annotations: [{ x: ridgeBestA, y: fd.ridge.best_val_rmse, xref: 'x', yref: 'y',
        text: `best α=${ridgeBestA.toFixed(3)}<br>val=${fd.ridge.best_val_rmse.toFixed(2)}`,
        showarrow: true, arrowhead: 2, ax: 40, ay: -30, font: { size: 10 } }],
    });
    const lassoLayout = Object.assign({}, errorLayout('Lasso (L1) — error curve'), {
      shapes: [bestAlphaShape(lassoBestA)],
      annotations: [{ x: lassoBestA, y: fd.lasso.best_val_rmse, xref: 'x', yref: 'y',
        text: `best α=${lassoBestA.toFixed(3)}<br>val=${fd.lasso.best_val_rmse.toFixed(2)}`,
        showarrow: true, arrowhead: 2, ax: 40, ay: -30, font: { size: 10 } }],
    });

    Plotly.react('rr-ridge-error',
      [...cTraces.ridgeErr.map(dimTrace), ...fTraces.ridgeErr],
      ridgeLayout, PLOTLY_CFG);
    Plotly.react('rr-lasso-error',
      [...cTraces.lassoErr.map(dimTrace), ...fTraces.lassoErr],
      lassoLayout, PLOTLY_CFG);

    renderCoefPaths('fine');
    updateCoefBarsAtBest(fd, fd.ridge.best_alpha_idx, fd.lasso.best_alpha_idx);

    ridgeRmse.innerHTML = rmseHtml(fd.ridge.train_rmse[fd.ridge.best_alpha_idx],
                                    fd.ridge.val_rmse[fd.ridge.best_alpha_idx]);
    lassoRmse.innerHTML = rmseHtml(fd.lasso.train_rmse[fd.lasso.best_alpha_idx],
                                    fd.lasso.val_rmse[fd.lasso.best_alpha_idx]);
  }

  function revertToCoarseGrid() {
    const d = state.initData;
    renderCoarseGridLines();
    renderCoefPaths('coarse');
    updateCoefBarsAtBest(d, d.ridge.best_alpha_idx, d.lasso.best_alpha_idx);
    ridgeRmse.innerHTML = rmseHtml(d.ridge.train_rmse[d.ridge.best_alpha_idx],
                                    d.ridge.val_rmse[d.ridge.best_alpha_idx]);
    lassoRmse.innerHTML = rmseHtml(d.lasso.train_rmse[d.lasso.best_alpha_idx],
                                    d.lasso.val_rmse[d.lasso.best_alpha_idx]);
  }

  // ── Coefficient path charts ──────────────────────────────────────────────

  function renderCoefPaths(resolution) {
    const data = resolution === 'fine' ? state.fineData : state.initData;
    const alphas = data.alphas;
    const names  = state.initData.feature_names;

    const ridgeBestA = alphas[data.ridge.best_alpha_idx];
    const lassoBestA = alphas[data.lasso.best_alpha_idx];

    const mkTraces = (coefPaths) =>
      names.map((name, j) => ({
        x: alphas,
        y: coefPaths.map(row => row[j]),
        mode: 'lines',
        name: name,
        line: { width: 1.5 },
        hovertemplate: `${name}<br>α=%{x:.3f}<br>coef=%{y:.4f}<extra></extra>`,
      }));

    const ridgeLayout = Object.assign({}, coefPathLayout('Ridge coefficient paths'), {
      shapes: [bestAlphaShape(ridgeBestA)],
    });
    const lassoLayout = Object.assign({}, coefPathLayout('Lasso coefficient paths'), {
      shapes: [bestAlphaShape(lassoBestA)],
    });

    if (!state.chartsReady.ridgePath) {
      Plotly.newPlot('rr-ridge-coef-path', mkTraces(data.ridge.coef_paths), ridgeLayout, PLOTLY_CFG);
      Plotly.newPlot('rr-lasso-coef-path', mkTraces(data.lasso.coef_paths), lassoLayout, PLOTLY_CFG);
      state.chartsReady.ridgePath  = true;
      state.chartsReady.lassoPaths = true;
    } else {
      Plotly.react('rr-ridge-coef-path', mkTraces(data.ridge.coef_paths), ridgeLayout, PLOTLY_CFG);
      Plotly.react('rr-lasso-coef-path', mkTraces(data.lasso.coef_paths), lassoLayout, PLOTLY_CFG);
    }
  }

  // ── Test evaluation reveal ───────────────────────────────────────────────

  function revealTestPanel() {
    const isFineCurrent = state.grid === 'fine' && state.fineData;
    const data = isFineCurrent ? state.fineData : state.initData;

    const ridgeBestA = data.alphas[data.ridge.best_alpha_idx];
    const lassoBestA = data.alphas[data.lasso.best_alpha_idx];

    ridgeTestStat.innerHTML =
      `Best α: <strong>${ridgeBestA.toFixed(3)}</strong><br>` +
      `Val RMSE: ${data.ridge.best_val_rmse.toFixed(2)}<br>` +
      `Test RMSE: <strong>${data.ridge.test_rmse.toFixed(2)}</strong>`;

    lassoTestStat.innerHTML =
      `Best α: <strong>${lassoBestA.toFixed(3)}</strong><br>` +
      `Val RMSE: ${data.lasso.best_val_rmse.toFixed(2)}<br>` +
      `Test RMSE: <strong>${data.lasso.test_rmse.toFixed(2)}</strong>`;

    panelTest.classList.remove('hidden');
    testBtn.disabled = true;
  }

  // ── Event listeners ──────────────────────────────────────────────────────

  alphaSlider.addEventListener('input', () => {
    const idx = parseInt(alphaSlider.value, 10);
    state.currentIdx = idx;
    alphaVal.textContent = idxToAlpha(idx);
  });

  alphaSlider.addEventListener('change', () => {
    if (state.mode !== 'manual') return;
    const idx = parseInt(alphaSlider.value, 10);
    addManualDot(idx);
    updateCoefBars(idx);
    updateRmseText(idx);
  });

  systematicBtn.addEventListener('click', () => {
    enterSystematicMode();
  });

  coarseBtn.addEventListener('click', () => {
    if (state.grid === 'coarse') return;
    state.grid = 'coarse';
    coarseBtn.classList.add('active');
    fineBtn.classList.remove('active');
    revertToCoarseGrid();
  });

  fineBtn.addEventListener('click', async () => {
    if (state.grid === 'fine') return;
    state.grid = 'fine';
    fineBtn.classList.add('active');
    coarseBtn.classList.remove('active');
    await loadFine();
  });

  testBtn.addEventListener('click', () => {
    revealTestPanel();
  });

  // ── Bootstrap ────────────────────────────────────────────────────────────

  let bootstrapped = false;

  async function bootstrap() {
    if (bootstrapped) return;
    bootstrapped = true;

    const res = await fetch('/api/regularized_regression/init');
    state.initData = await res.json();

    renderPearsonChart();
    initErrorCharts();
    initCoefBarCharts();

    // Show OLS baseline immediately (α=0, index 0)
    updateRmseText(0);
  }

  window._regularizedRegressionBootstrap = bootstrap;
})();
