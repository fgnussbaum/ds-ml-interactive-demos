(function () {
  // ── Constants ─────────────────────────────────────────────────────────────
  const SPECIES_COLORS = ['#e84040', '#4682b4', '#2ca02c'];
  const SPECIES_NAMES  = ['Adelie', 'Chinstrap', 'Gentoo'];
  const FEATURE_LABELS = {
    bill_depth_mm:    'Bill depth (mm)',
    bill_length_mm:   'Bill length (mm)',
    flipper_length_mm:'Flipper length (mm)',
    body_mass_g:      'Body mass (g)',
  };
  const C_TRAIN  = '#4682b4';
  const C_TEST   = '#e07b39';
  const C_MARKER = '#bbb';
  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  // ── State ─────────────────────────────────────────────────────────────────
  const state = {
    maxDepth:       3,
    minSamplesLeaf: 5,
    featureX:       'bill_depth_mm',
    featureY:       'flipper_length_mm',
    animDepth:      3,
    data:           null,
    rfData:         null,
    rfActive:       false,
  };

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const depthSlider   = document.getElementById('dt-depth');
  const depthVal      = document.getElementById('dt-depth-val');
  const mslSlider     = document.getElementById('dt-msl');
  const mslVal        = document.getElementById('dt-msl-val');
  const featXSel      = document.getElementById('dt-feat-x');
  const featYSel      = document.getElementById('dt-feat-y');
  const animSlider    = document.getElementById('dt-anim');
  const animVal       = document.getElementById('dt-anim-val');
  const treeImg       = document.getElementById('dt-tree-img');
  const boundaryImg   = document.getElementById('dt-boundary-img');
  const rootCaption   = document.getElementById('dt-root-caption');
  const accText       = document.getElementById('dt-acc-text');
  const loadingEl     = document.getElementById('dt-loading');
  const rfBtn         = document.getElementById('dt-rf-btn');
  const rfContent     = document.getElementById('dt-rf-content');
  const rfDtImg       = document.getElementById('dt-rf-dt-img');
  const rfImg         = document.getElementById('dt-rf-img');
  const rfDtAcc       = document.getElementById('dt-rf-dt-acc');
  const rfAccEl       = document.getElementById('dt-rf-acc');
  const rfDepthLabel  = document.getElementById('dt-rf-depth-label');
  const rfLoading     = document.getElementById('dt-rf-loading');

  // ── Chart init guards ─────────────────────────────────────────────────────
  const ready = { scatter: false, importance: false, acc: false };

  // ── Event listeners ───────────────────────────────────────────────────────
  depthSlider.addEventListener('input', () => {
    state.maxDepth = +depthSlider.value;
    depthVal.textContent = state.maxDepth;
    state.animDepth = state.maxDepth;
    fetchAndRender();
  });

  let mslTimer = null;
  mslSlider.addEventListener('input', () => {
    state.minSamplesLeaf = +mslSlider.value;
    mslVal.textContent = state.minSamplesLeaf;
    clearTimeout(mslTimer);
    mslTimer = setTimeout(fetchAndRender, 300);
  });

  featXSel.addEventListener('change', () => {
    state.featureX = featXSel.value;
    state.rfData = null;
    fetchAndRender();
  });

  featYSel.addEventListener('change', () => {
    state.featureY = featYSel.value;
    state.rfData = null;
    fetchAndRender();
  });

  animSlider.addEventListener('input', () => {
    state.animDepth = +animSlider.value;
    animVal.textContent = state.animDepth;
    if (state.data) renderDepthView();
  });

  rfBtn.addEventListener('click', () => {
    state.rfActive = !state.rfActive;
    rfBtn.classList.toggle('active', state.rfActive);
    rfContent.style.display = state.rfActive ? '' : 'none';
    if (state.rfActive) {
      if (!state.rfData) fetchRF();
      else renderRF();
    }
  });

  // ── Data fetching ─────────────────────────────────────────────────────────
  async function fetchAndRender() {
    loadingEl.style.display = '';
    state.rfData = null;

    const res = await fetch('/api/decision_tree/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        max_depth:        state.maxDepth,
        min_samples_leaf: state.minSamplesLeaf,
        feature_x:        state.featureX,
        feature_y:        state.featureY,
      }),
    });
    state.data = await res.json();
    loadingEl.style.display = 'none';

    animSlider.max   = state.maxDepth;
    animSlider.value = state.maxDepth;
    animVal.textContent = state.maxDepth;
    state.animDepth  = state.maxDepth;

    renderAll();
    if (state.rfActive) fetchRF();
  }

  async function fetchRF() {
    rfLoading.style.display = '';
    rfImg.style.display = 'none';

    const res = await fetch('/api/decision_tree/train-rf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature_x: state.featureX, feature_y: state.featureY }),
    });
    state.rfData = await res.json();
    rfLoading.style.display = 'none';
    rfImg.style.display = '';
    renderRF();
  }

  // ── Render ────────────────────────────────────────────────────────────────
  function renderAll() {
    renderScatter();
    renderDepthView();
    renderImportance();
    renderAccChart();
    rootCaption.textContent = state.data.root_caption;
  }

  function renderDepthView() {
    const d = state.data.depths[state.animDepth - 1];
    treeImg.src     = d.tree_src;
    boundaryImg.src = `data:image/png;base64,${d.boundary_png}`;
    accText.innerHTML =
      `Train: <strong>${pct(d.train_acc)}</strong>` +
      `&ensp;&middot;&ensp;Test: <strong>${pct(d.test_acc)}</strong>`;
    updateAccMarker();
  }

  function renderScatter() {
    const d = state.data;
    const traces = SPECIES_NAMES.map((name, i) => {
      const xs = [], ys = [];
      d.scatter_species.forEach((s, j) => {
        if (s === i) { xs.push(d.scatter_x[j]); ys.push(d.scatter_y[j]); }
      });
      return {
        x: xs, y: ys, mode: 'markers', name,
        marker: { color: SPECIES_COLORS[i], size: 6 },
        type: 'scatter',
      };
    });
    const layout = {
      margin: { l: 60, r: 16, t: 12, b: 52 },
      height: 270,
      xaxis: { title: FEATURE_LABELS[state.featureX] },
      yaxis: { title: FEATURE_LABELS[state.featureY] },
      legend: { orientation: 'h', y: -0.22, x: 0 },
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    };
    if (!ready.scatter) {
      Plotly.newPlot('dt-scatter', traces, layout, PLOTLY_CFG);
      ready.scatter = true;
    } else {
      Plotly.react('dt-scatter', traces, layout);
    }
  }

  function renderImportance() {
    const imp = state.data.feature_importances;
    const entries = Object.entries(imp).sort(([, a], [, b]) => b - a);
    const traces = [{
      x: entries.map(([, v]) => v),
      y: entries.map(([k]) => FEATURE_LABELS[k]),
      type: 'bar', orientation: 'h',
      marker: { color: '#4a9eff' },
    }];
    const layout = {
      margin: { l: 140, r: 16, t: 4, b: 36 },
      height: 150,
      xaxis: { title: 'Importance', range: [0, 1] },
      yaxis: { autorange: 'reversed' },
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    };
    if (!ready.importance) {
      Plotly.newPlot('dt-importance-chart', traces, layout, PLOTLY_CFG);
      ready.importance = true;
    } else {
      Plotly.react('dt-importance-chart', traces, layout);
    }
  }

  function renderAccChart() {
    const depths   = state.data.depths.map(r => r.depth);
    const trainAcc = state.data.depths.map(r => r.train_acc);
    const testAcc  = state.data.depths.map(r => r.test_acc);
    const yMax = 1.02;
    const traces = [
      {
        x: depths, y: trainAcc, mode: 'lines+markers', name: 'Train accuracy',
        line: { color: C_TRAIN, width: 2 }, marker: { color: C_TRAIN, size: 6 },
        type: 'scatter',
      },
      {
        x: depths, y: testAcc, mode: 'lines+markers', name: 'Test accuracy',
        line: { color: C_TEST, width: 2 }, marker: { color: C_TEST, size: 6 },
        type: 'scatter',
      },
      {
        x: [state.animDepth, state.animDepth], y: [0, yMax],
        mode: 'lines', line: { color: C_MARKER, width: 1.5, dash: 'dash' },
        type: 'scatter', showlegend: false,
      },
    ];
    const layout = {
      margin: { l: 56, r: 16, t: 12, b: 52 },
      height: 220,
      xaxis: { title: 'Tree depth', tickvals: depths, dtick: 1 },
      yaxis: { title: 'Accuracy', range: [0.4, yMax] },
      legend: { orientation: 'h', y: -0.26, x: 0 },
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    };
    if (!ready.acc) {
      Plotly.newPlot('dt-acc-chart', traces, layout, PLOTLY_CFG);
      ready.acc = true;
    } else {
      Plotly.react('dt-acc-chart', traces, layout);
    }
  }

  function updateAccMarker() {
    if (!ready.acc) return;
    Plotly.restyle('dt-acc-chart',
      { x: [[state.animDepth, state.animDepth]] }, [2]);
  }

  function renderRF() {
    if (!state.rfData || !state.data) return;
    const dt = state.data.depths[state.maxDepth - 1];
    rfDtImg.src = `data:image/png;base64,${dt.boundary_png}`;
    rfDtAcc.innerHTML =
      `Train: <strong>${pct(dt.train_acc)}</strong>` +
      `&ensp;&middot;&ensp;Test: <strong>${pct(dt.test_acc)}</strong>`;
    rfImg.src = `data:image/png;base64,${state.rfData.boundary_png}`;
    rfAccEl.innerHTML =
      `Train: <strong>${pct(state.rfData.train_acc)}</strong>` +
      `&ensp;&middot;&ensp;Test: <strong>${pct(state.rfData.test_acc)}</strong>`;
    rfDepthLabel.textContent = state.maxDepth;
  }

  // ── Utility ───────────────────────────────────────────────────────────────
  function pct(v) { return (v * 100).toFixed(1) + '%'; }

  // ── Bootstrap ─────────────────────────────────────────────────────────────
  let bootstrapped = false;

  function bootstrap() {
    if (!bootstrapped) {
      bootstrapped = true;
      fetchAndRender();
    }
  }

  window._decisionTreeBootstrap = bootstrap;
})();
