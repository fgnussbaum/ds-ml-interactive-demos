(function () {
  // ── Constants ───────────────────────────────────────────────────────────
  const C = {
    train:         '#4682b4',
    val:           '#e07b39',
    best:          '#2ca02c',
    coarseOpacity: 0.35,
    coarseSize:    7,
    fineSize:      5,
  };

  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  // ── State ───────────────────────────────────────────────────────────────
  const state = {
    mode:         'manual',   // 'manual' | 'systematic'
    grid:         'coarse',   // 'coarse' | 'fine'
    visitedIdx:   new Set(),  // alpha indices already plotted as dots
    currentIdx:   0,
    initData:     null,
    fineData:     null,
    fineLoading:  false,
    yLim:         null,       // [ymin, ymax] shared across both error charts
    pearsonOrder: null,       // feature indices sorted by |r| desc (for bonus panel)
    pathsReady:   false,
  };

  // ── DOM refs ────────────────────────────────────────────────────────────
  const alphaSlider   = document.getElementById('rr-alpha');
  const alphaVal      = document.getElementById('rr-alpha-val');
  const ridgeRmse     = document.getElementById('rr-ridge-rmse');
  const lassoRmse     = document.getElementById('rr-lasso-rmse');
  const gridHint      = document.getElementById('rr-grid-hint');
  const systematicBtn = document.getElementById('rr-systematic-btn');
  const panelGrid     = document.getElementById('rr-panel-grid');
  const coarseBtn     = document.getElementById('rr-coarse-btn');
  const fineBtn       = document.getElementById('rr-fine-btn');
  const testBtn       = document.getElementById('rr-test-btn');
  const panelTest     = document.getElementById('rr-panel-test');
  const ridgeTestStat = document.getElementById('rr-ridge-test-stat');
  const lassoTestStat = document.getElementById('rr-lasso-test-stat');
  const bonusBtn      = document.getElementById('rr-bonus-btn');
  const panelBonus    = document.getElementById('rr-panel-bonus');

  // ── Helpers ─────────────────────────────────────────────────────────────
  function idxToAlpha(idx) {
    return (idx / 10).toFixed(2);
  }

  function rmseHtml(train, val) {
    return `Train RMSE: <span style="color:${C.train}"><strong>${train.toFixed(2)}</strong></span>
            &nbsp;·&nbsp;
            Val RMSE: <span style="color:${C.val}"><strong>${val.toFixed(2)}</strong></span>`;
  }

  // Compute shared y-axis range from all train/val RMSE values in init data.
  // Padding keeps the OLS baseline point from touching the top edge.
  function computeYLim(data) {
    const all = [
      ...data.ridge.train_rmse, ...data.ridge.val_rmse,
      ...data.lasso.train_rmse, ...data.lasso.val_rmse,
    ];
    const pad = (Math.max(...all) - Math.min(...all)) * 0.08;
    return [Math.min(...all) - pad, Math.max(...all) + pad];
  }

  // ── Layout builders ─────────────────────────────────────────────────────
  function errorLayout(title, extra) {
    return Object.assign({
      title:  { text: title, font: { size: 13 } },
      xaxis:  { title: 'α', range: [-0.005, 0.105] },
      yaxis:  { title: 'RMSE', range: state.yLim },
      margin: { t: 36, r: 12, b: 44, l: 52 },
      legend: { orientation: 'h', y: -0.2 },
      height: 240,
      shapes: [],
    }, extra || {});
  }

  function coefPathLayout(title) {
    return {
      title:  { text: title, font: { size: 12 } },
      xaxis:  { title: 'α' },
      yaxis:  { title: 'Coefficient', zeroline: true, zerolinecolor: '#999' },
      margin: { t: 32, r: 12, b: 44, l: 52 },
      legend: { font: { size: 9 }, x: 1, xanchor: 'right', y: 1 },
      height: 260,
      shapes: [],
    };
  }

  function bonusBarLayout(title, xRange) {
    return {
      title:  { text: title, font: { size: 11 } },
      xaxis:  { zeroline: true, zerolinecolor: '#bbb', range: xRange },
      yaxis:  { tickfont: { size: 9 }, automargin: true },
      margin: { t: 28, r: 8, b: 28, l: 8 },
      height: 240,
      showlegend: false,
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

  // ── Pearson chart ────────────────────────────────────────────────────────
  function renderPearsonChart() {
    const d = state.initData;
    const order = d.feature_names
      .map((_, i) => i)
      .sort((a, b) => Math.abs(d.pearson_r[b]) - Math.abs(d.pearson_r[a]));
    state.pearsonOrder = order;

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
  }

  // ── Error charts ─────────────────────────────────────────────────────────
  function initErrorCharts() {
    const d   = state.initData;
    const a0  = d.alphas[0];

    // Seed with α=0 dot immediately (OLS baseline visible before any interaction)
    const mkTraces = (tr0, va0) => [
      { x: [a0], y: [tr0], mode: 'markers', name: 'train',
        marker: { color: C.train, size: 10 },
        hovertemplate: 'α=%{x:.2f}<br>RMSE=%{y:.2f}<extra>train</extra>' },
      { x: [a0], y: [va0], mode: 'markers', name: 'val',
        marker: { color: C.val,   size: 10 },
        hovertemplate: 'α=%{x:.2f}<br>RMSE=%{y:.2f}<extra>val</extra>' },
    ];

    Plotly.newPlot('rr-ridge-error',
      mkTraces(d.ridge.train_rmse[0], d.ridge.val_rmse[0]),
      errorLayout('Ridge (L2) — validation error'), PLOTLY_CFG);
    Plotly.newPlot('rr-lasso-error',
      mkTraces(d.lasso.train_rmse[0], d.lasso.val_rmse[0]),
      errorLayout('Lasso (L1) — validation error'), PLOTLY_CFG);

    state.visitedIdx.add(0);
  }

  // ── Manual mode updates ─────────────────────────────────────────────────
  function addManualDot(idx) {
    if (state.visitedIdx.has(idx)) return;
    state.visitedIdx.add(idx);

    const d = state.initData;
    const a = d.alphas[idx];

    Plotly.extendTraces('rr-ridge-error',
      { x: [[a], [a]], y: [[d.ridge.train_rmse[idx]], [d.ridge.val_rmse[idx]]] }, [0, 1]);
    Plotly.extendTraces('rr-lasso-error',
      { x: [[a], [a]], y: [[d.lasso.train_rmse[idx]], [d.lasso.val_rmse[idx]]] }, [0, 1]);

    if (state.visitedIdx.size >= 3) {
      gridHint.classList.remove('hidden');
    }
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

    testBtn.disabled = true;   // only unlocked once fine grid is active
    renderCoarseGridLines();
    panelGrid.classList.remove('hidden');
    requestAnimationFrame(() => renderCoefPaths('coarse'));

    const d = state.initData;
    ridgeRmse.innerHTML = rmseHtml(
      d.ridge.train_rmse[d.ridge.best_alpha_idx],
      d.ridge.val_rmse[d.ridge.best_alpha_idx]);
    lassoRmse.innerHTML = rmseHtml(
      d.lasso.train_rmse[d.lasso.best_alpha_idx],
      d.lasso.val_rmse[d.lasso.best_alpha_idx]);
  }

  // ── Coarse grid error lines ──────────────────────────────────────────────
  function coarseGridTraces(data) {
    const a = data.alphas;
    return {
      ridgeErr: [
        { x: a, y: data.ridge.train_rmse, mode: 'lines+markers', name: 'train',
          line: { color: C.train }, marker: { color: C.train, size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train</extra>' },
        { x: a, y: data.ridge.val_rmse, mode: 'lines+markers', name: 'val',
          line: { color: C.val }, marker: { color: C.val, size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val</extra>' },
      ],
      lassoErr: [
        { x: a, y: data.lasso.train_rmse, mode: 'lines+markers', name: 'train',
          line: { color: C.train }, marker: { color: C.train, size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train</extra>' },
        { x: a, y: data.lasso.val_rmse, mode: 'lines+markers', name: 'val',
          line: { color: C.val }, marker: { color: C.val, size: C.coarseSize },
          hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val</extra>' },
      ],
    };
  }

  function annotationAt(alpha, rmse, label) {
    return { x: alpha, y: rmse, xref: 'x', yref: 'y',
      text: label, showarrow: true, arrowhead: 2,
      ax: 40, ay: -30, font: { size: 10 } };
  }

  function renderCoarseGridLines() {
    const d       = state.initData;
    const traces  = coarseGridTraces(d);
    const rBestA  = d.alphas[d.ridge.best_alpha_idx];
    const lBestA  = d.alphas[d.lasso.best_alpha_idx];

    Plotly.react('rr-ridge-error', traces.ridgeErr,
      errorLayout('Ridge (L2) — error curve', {
        shapes:      [bestAlphaShape(rBestA)],
        annotations: [annotationAt(rBestA, d.ridge.best_val_rmse,
                        `best α=${rBestA.toFixed(3)}<br>val=${d.ridge.best_val_rmse.toFixed(2)}`)],
      }), PLOTLY_CFG);
    Plotly.react('rr-lasso-error', traces.lassoErr,
      errorLayout('Lasso (L1) — error curve', {
        shapes:      [bestAlphaShape(lBestA)],
        annotations: [annotationAt(lBestA, d.lasso.best_val_rmse,
                        `best α=${lBestA.toFixed(3)}<br>val=${d.lasso.best_val_rmse.toFixed(2)}`)],
      }), PLOTLY_CFG);
  }

  // ── Fine grid ────────────────────────────────────────────────────────────
  async function loadFine() {
    if (state.fineLoading) return;  // in-flight, ignore duplicate click
    if (!state.fineData) {
      state.fineLoading = true;
      fineBtn.textContent = 'Loading…';
      fineBtn.disabled = true;
      try {
        const res = await fetch('/api/regularized_regression/fine');
        state.fineData = await res.json();
      } finally {
        state.fineLoading = false;
        fineBtn.textContent = 'Fine (101 steps)';
        fineBtn.disabled = false;
      }
    }
    renderFineGrid();  // always render, whether fresh or cached
  }

  function renderFineGrid() {
    const fd = state.fineData;
    const cd = state.initData;
    const a  = fd.alphas;

    const dim = (t) => Object.assign({}, t, {
      opacity: C.coarseOpacity,
      marker: Object.assign({}, t.marker, { size: C.coarseSize }),
      showlegend: false,
    });

    const fineTraces = (model) => [
      { x: a, y: model.train_rmse, mode: 'lines+markers', name: 'train (fine)',
        line: { color: C.train, width: 2 }, marker: { color: C.train, size: C.fineSize },
        hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>train (fine)</extra>' },
      { x: a, y: model.val_rmse, mode: 'lines+markers', name: 'val (fine)',
        line: { color: C.val, width: 2 }, marker: { color: C.val, size: C.fineSize },
        hovertemplate: 'α=%{x:.3f}<br>RMSE=%{y:.2f}<extra>val (fine)</extra>' },
    ];

    const cTraces = coarseGridTraces(cd);
    const rBestA  = fd.alphas[fd.ridge.best_alpha_idx];
    const lBestA  = fd.alphas[fd.lasso.best_alpha_idx];

    Plotly.react('rr-ridge-error',
      [...cTraces.ridgeErr.map(dim), ...fineTraces(fd.ridge)],
      errorLayout('Ridge (L2) — error curve', {
        shapes:      [bestAlphaShape(rBestA)],
        annotations: [annotationAt(rBestA, fd.ridge.best_val_rmse,
                        `best α=${rBestA.toFixed(3)}<br>val=${fd.ridge.best_val_rmse.toFixed(2)}`)],
      }), PLOTLY_CFG);
    Plotly.react('rr-lasso-error',
      [...cTraces.lassoErr.map(dim), ...fineTraces(fd.lasso)],
      errorLayout('Lasso (L1) — error curve', {
        shapes:      [bestAlphaShape(lBestA)],
        annotations: [annotationAt(lBestA, fd.lasso.best_val_rmse,
                        `best α=${lBestA.toFixed(3)}<br>val=${fd.lasso.best_val_rmse.toFixed(2)}`)],
      }), PLOTLY_CFG);

    renderCoefPaths('fine');
    testBtn.disabled = false;  // fine grid loaded — test reveal now available

    ridgeRmse.innerHTML = rmseHtml(
      fd.ridge.train_rmse[fd.ridge.best_alpha_idx],
      fd.ridge.val_rmse[fd.ridge.best_alpha_idx]);
    lassoRmse.innerHTML = rmseHtml(
      fd.lasso.train_rmse[fd.lasso.best_alpha_idx],
      fd.lasso.val_rmse[fd.lasso.best_alpha_idx]);
  }

  function revertToCoarseGrid() {
    const d = state.initData;
    testBtn.disabled = true;   // lock test reveal until back on fine grid
    renderCoarseGridLines();
    renderCoefPaths('coarse');
    ridgeRmse.innerHTML = rmseHtml(
      d.ridge.train_rmse[d.ridge.best_alpha_idx],
      d.ridge.val_rmse[d.ridge.best_alpha_idx]);
    lassoRmse.innerHTML = rmseHtml(
      d.lasso.train_rmse[d.lasso.best_alpha_idx],
      d.lasso.val_rmse[d.lasso.best_alpha_idx]);
  }

  // ── Coefficient path charts ──────────────────────────────────────────────
  function renderCoefPaths(resolution) {
    const data   = resolution === 'fine' ? state.fineData : state.initData;
    const alphas = data.alphas;
    const names  = state.initData.feature_names;

    const mkTraces = (coefPaths) =>
      names.map((name, j) => ({
        x: alphas,
        y: coefPaths.map(row => row[j]),
        mode: 'lines', name,
        line: { width: 1.5 },
        hovertemplate: `${name}<br>α=%{x:.3f}<br>coef=%{y:.4f}<extra></extra>`,
      }));

    const rLayout = Object.assign({}, coefPathLayout('Ridge coefficient paths'),
      { shapes: [bestAlphaShape(alphas[data.ridge.best_alpha_idx])] });
    const lLayout = Object.assign({}, coefPathLayout('Lasso coefficient paths'),
      { shapes: [bestAlphaShape(alphas[data.lasso.best_alpha_idx])] });

    if (!state.pathsReady) {
      Plotly.newPlot('rr-ridge-coef-path', mkTraces(data.ridge.coef_paths), rLayout, PLOTLY_CFG);
      Plotly.newPlot('rr-lasso-coef-path', mkTraces(data.lasso.coef_paths), lLayout, PLOTLY_CFG);
      state.pathsReady = true;
    } else {
      Plotly.react('rr-ridge-coef-path', mkTraces(data.ridge.coef_paths), rLayout, PLOTLY_CFG);
      Plotly.react('rr-lasso-coef-path', mkTraces(data.lasso.coef_paths), lLayout, PLOTLY_CFG);
    }
  }

  // ── Test evaluation reveal ───────────────────────────────────────────────
  function revealTestPanel() {
    // Always use fine-grid best alphas — testBtn is only enabled in fine mode.
    const data   = state.fineData;
    const rBestA = data.alphas[data.ridge.best_alpha_idx];
    const lBestA = data.alphas[data.lasso.best_alpha_idx];

    ridgeTestStat.innerHTML =
      `Best α: <strong>${rBestA.toFixed(3)}</strong><br>` +
      `Val RMSE: ${data.ridge.best_val_rmse.toFixed(2)}<br>` +
      `Test RMSE: <strong>${data.ridge.test_rmse.toFixed(2)}</strong>`;
    lassoTestStat.innerHTML =
      `Best α: <strong>${lBestA.toFixed(3)}</strong><br>` +
      `Val RMSE: ${data.lasso.best_val_rmse.toFixed(2)}<br>` +
      `Test RMSE: <strong>${data.lasso.test_rmse.toFixed(2)}</strong>`;

    panelTest.classList.remove('hidden');
    testBtn.disabled = true;
    // Lock the toggle — best-α is now committed, switching grids would be misleading.
    coarseBtn.disabled = true;
    fineBtn.disabled = true;
    bonusBtn.classList.remove('hidden');
  }

  // ── Bonus comparison panel ───────────────────────────────────────────────
  function renderBonusPanel() {
    const data  = (state.grid === 'fine' && state.fineData) ? state.fineData : state.initData;
    const d     = state.initData;
    const order = state.pearsonOrder;  // sorted by |pearson_r| desc

    const sortedNames = order.map(i => d.feature_names[i]);
    const sortedR     = order.map(i => d.pearson_r[i]);
    const rCoefs      = order.map(i => data.ridge.coef_paths[data.ridge.best_alpha_idx][i]);
    const lCoefs      = order.map(i => data.lasso.coef_paths[data.lasso.best_alpha_idx][i]);

    const barTrace = (xs, ys) => [{
      type: 'bar', orientation: 'h',
      x: xs, y: ys,
      marker: { color: xs.map(v => v >= 0 ? C.train : C.val) },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    }];

    const rBestA     = data.alphas[data.ridge.best_alpha_idx];
    const lBestA     = data.alphas[data.lasso.best_alpha_idx];
    const maxAbsCoef = Math.max(...rCoefs.map(Math.abs), ...lCoefs.map(Math.abs));
    const coefRange  = [-(maxAbsCoef * 1.15), maxAbsCoef * 1.15];

    // Reveal first so the browser computes grid column widths before Plotly
    // measures the containers; requestAnimationFrame defers until after layout.
    panelBonus.classList.remove('hidden');
    bonusBtn.disabled = true;

    requestAnimationFrame(() => {
      Plotly.newPlot('rr-bonus-ridge',
        barTrace(rCoefs, sortedNames),
        bonusBarLayout(`Ridge coefs at α=${rBestA.toFixed(3)}`, coefRange), PLOTLY_CFG);
      Plotly.newPlot('rr-bonus-pearson',
        barTrace(sortedR, sortedNames),
        bonusBarLayout('Pearson r', [-1, 1]), PLOTLY_CFG);
      Plotly.newPlot('rr-bonus-lasso',
        barTrace(lCoefs, sortedNames),
        bonusBarLayout(`Lasso coefs at α=${lBestA.toFixed(3)}`, coefRange), PLOTLY_CFG);
    });
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
    updateRmseText(idx);
  });

  systematicBtn.addEventListener('click', enterSystematicMode);

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

  testBtn.addEventListener('click', revealTestPanel);

  bonusBtn.addEventListener('click', renderBonusPanel);

  // ── Bootstrap ────────────────────────────────────────────────────────────
  let bootstrapped = false;

  async function bootstrap() {
    if (bootstrapped) return;
    bootstrapped = true;

    const res = await fetch('/api/regularized_regression/init');
    state.initData = await res.json();
    state.yLim = computeYLim(state.initData);

    renderPearsonChart();
    initErrorCharts();      // seeds α=0 dot
    updateRmseText(0);      // show OLS RMSE immediately
  }

  window._regularizedRegressionBootstrap = bootstrap;
})();
