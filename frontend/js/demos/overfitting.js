(function () {
  // ── Constants ─────────────────────────────────────────────────────────────
  const C = {
    train:     '#4682b4',
    test:      '#e07b39',
    true_fn:   '#999',
    fit:       '#cc2222',
    rmse_train:'#4682b4',
    rmse_test: '#e07b39',
    marker:    '#bbb',
  };

  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  // ── State ─────────────────────────────────────────────────────────────────
  const state = {
    scenario: 'cubic',
    nTrain: 25,
    degree: 3,
    showTest: false,
    data: null,
  };

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const scenarioSel   = document.getElementById('of-scenario');
  const nSlider       = document.getElementById('of-n');
  const nVal          = document.getElementById('of-n-val');
  const degreeSlider  = document.getElementById('of-degree');
  const degreeVal     = document.getElementById('of-degree-val');
  const testToggle    = document.getElementById('of-test-toggle');
  const nudge         = document.getElementById('of-nudge');
  const nudgeDegree   = document.getElementById('of-nudge-degree');
  const mainRmse      = document.getElementById('of-main-rmse');
  const placeholder   = document.getElementById('of-panel-best-placeholder');
  const bestTitle     = document.getElementById('of-panel-best-title');

  // ── Chart init guards ─────────────────────────────────────────────────────
  const ready = { main: false, rmse: false, underfit: false, best: false, overfit: false };

  // ── Event listeners ───────────────────────────────────────────────────────
  scenarioSel.addEventListener('change', () => {
    state.scenario = scenarioSel.value;
    state.showTest = false;
    syncToggleUI();
    fetchAndRender();
  });

  let nDebounce = null;
  nSlider.addEventListener('input', () => {
    state.nTrain = +nSlider.value;
    nVal.textContent = state.nTrain;
    clearTimeout(nDebounce);
    nDebounce = setTimeout(fetchAndRender, 150);
  });

  degreeSlider.addEventListener('input', () => {
    state.degree = +degreeSlider.value;
    degreeVal.textContent = state.degree;
    nudgeDegree.textContent = state.degree;
    if (state.data) {
      renderMainChart();
      updateMainRmse();
      updateRmseMarker();
    }
  });

  testToggle.addEventListener('click', () => {
    state.showTest = !state.showTest;
    syncToggleUI();
    if (state.data) renderAll();
  });

  // ── UI sync ───────────────────────────────────────────────────────────────
  function syncToggleUI() {
    testToggle.textContent = state.showTest ? 'Hide test data' : 'Show test data';
    testToggle.classList.toggle('of-active', state.showTest);
    nudge.style.display = state.showTest ? 'none' : '';
    placeholder.style.display = state.showTest ? 'none' : '';
    bestTitle.textContent = state.showTest && state.data
      ? `Best fit — degree ${state.data.best_degree}`
      : 'Best fit';
  }

  // ── Data ──────────────────────────────────────────────────────────────────
  async function fetchAndRender() {
    const res = await fetch('/api/overfitting/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: state.scenario, n_train: state.nTrain }),
    });
    state.data = await res.json();
    syncToggleUI();
    renderAll();
  }

  // ── Y-axis range (fixed per dataset so charts don't jump while sliding) ───
  function yRange() {
    const d = state.data;
    const all = [...d.train_y, ...d.test_y, ...d.true_curve_y];
    const mn = Math.min(...all);
    const mx = Math.max(...all);
    const pad = (mx - mn) * 0.18;
    return [mn - pad, mx + pad];
  }

  // ── Shared trace builders ─────────────────────────────────────────────────
  function scatterTraces(d, deg, small) {
    const fit = d.fits[deg - 1];
    const sz = small ? 5 : 7;
    return [
      {
        x: d.train_x, y: d.train_y, mode: 'markers', name: 'Train',
        marker: { color: C.train, size: sz }, type: 'scatter',
        showlegend: !small,
      },
      {
        x: d.test_x, y: d.test_y, mode: 'markers', name: 'Test',
        marker: { color: C.test, size: sz, symbol: 'diamond' }, type: 'scatter',
        visible: state.showTest,
        showlegend: !small,
      },
      {
        x: d.curve_x, y: d.true_curve_y, mode: 'lines', name: 'True function',
        line: { color: C.true_fn, width: small ? 1.5 : 2, dash: 'dot' }, type: 'scatter',
        showlegend: !small,
      },
      {
        x: d.curve_x, y: fit.curve_y, mode: 'lines', name: `Degree ${deg}`,
        line: { color: C.fit, width: small ? 2 : 2.5 }, type: 'scatter',
        showlegend: !small,
      },
    ];
  }

  // ── Main chart ────────────────────────────────────────────────────────────
  function mainLayout() {
    return {
      margin: { l: 48, r: 16, t: 12, b: 44 },
      height: 360,
      showlegend: true,
      legend: { orientation: 'h', y: -0.14, x: 0 },
      xaxis: { title: 'x', zeroline: false },
      yaxis: { title: 'y', zeroline: false, range: yRange() },
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
    };
  }

  function renderMainChart() {
    const traces = scatterTraces(state.data, state.degree, false);
    if (!ready.main) {
      Plotly.newPlot('of-main-chart', traces, mainLayout(), PLOTLY_CFG);
      ready.main = true;
    } else {
      Plotly.react('of-main-chart', traces, mainLayout());
    }
  }

  function updateMainRmse() {
    const fit = state.data.fits[state.degree - 1];
    let html = `Train RMSE: <strong>${fit.train_rmse.toFixed(3)}</strong>`;
    if (state.showTest) {
      html += `&ensp;&middot;&ensp;Test RMSE: <strong>${fit.test_rmse.toFixed(3)}</strong>`;
    }
    mainRmse.innerHTML = html;
  }

  // ── RMSE chart ────────────────────────────────────────────────────────────
  function rmseTraces() {
    const d = state.data;
    const degrees = d.fits.map(f => f.degree);
    const yMax = Math.max(...d.fits.map(f => f.test_rmse), ...d.fits.map(f => f.train_rmse)) * 1.15;
    return [
      {
        x: degrees, y: d.fits.map(f => f.train_rmse),
        mode: 'lines+markers', name: 'Train RMSE',
        line: { color: C.rmse_train, width: 2 },
        marker: { color: C.rmse_train, size: 6 }, type: 'scatter',
      },
      {
        x: degrees, y: d.fits.map(f => f.test_rmse),
        mode: 'lines+markers', name: 'Test RMSE',
        line: { color: C.rmse_test, width: 2 },
        marker: { color: C.rmse_test, size: 6 }, type: 'scatter',
        visible: state.showTest,
      },
      {
        x: [state.degree, state.degree], y: [0, yMax],
        mode: 'lines', name: 'Selected degree',
        line: { color: C.marker, width: 1.5, dash: 'dash' }, type: 'scatter',
        showlegend: false,
      },
    ];
  }

  function rmseLayout() {
    return {
      margin: { l: 56, r: 16, t: 16, b: 52 },
      height: 240,
      showlegend: true,
      legend: { orientation: 'h', y: -0.24, x: 0 },
      xaxis: { title: 'Polynomial degree', tickvals: [1,2,3,4,5,6,7,8,9] },
      yaxis: { title: 'RMSE', rangemode: 'tozero' },
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
    };
  }

  function renderRmseChart() {
    const traces = rmseTraces();
    if (!ready.rmse) {
      Plotly.newPlot('of-rmse-chart', traces, rmseLayout(), PLOTLY_CFG);
      ready.rmse = true;
    } else {
      Plotly.react('of-rmse-chart', traces, rmseLayout());
    }
  }

  function updateRmseMarker() {
    if (!ready.rmse) return;
    const d = state.data;
    const yMax = Math.max(...d.fits.map(f => f.test_rmse), ...d.fits.map(f => f.train_rmse)) * 1.15;
    Plotly.restyle('of-rmse-chart', { x: [[state.degree, state.degree]], y: [[0, yMax]] }, [2]);
  }

  // ── Comparison panels ─────────────────────────────────────────────────────
  const PANELS = [
    { divId: 'of-panel-underfit', rmseId: 'of-panel-underfit-rmse', readyKey: 'underfit', deg: () => 1 },
    { divId: 'of-panel-best',     rmseId: 'of-panel-best-rmse',     readyKey: 'best',     deg: () => state.data.best_degree },
    { divId: 'of-panel-overfit',  rmseId: 'of-panel-overfit-rmse',  readyKey: 'overfit',  deg: () => 9 },
  ];

  function panelLayout() {
    return {
      margin: { l: 36, r: 8, t: 8, b: 28 },
      height: 200,
      showlegend: false,
      xaxis: { zeroline: false, showticklabels: true },
      yaxis: { zeroline: false, range: yRange() },
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
    };
  }

  function renderPanels() {
    for (const p of PANELS) {
      const deg = p.deg();
      const traces = scatterTraces(state.data, deg, true);
      const layout = panelLayout();
      if (!ready[p.readyKey]) {
        Plotly.newPlot(p.divId, traces, layout, PLOTLY_CFG);
        ready[p.readyKey] = true;
      } else {
        Plotly.react(p.divId, traces, layout);
      }
      const fit = state.data.fits[deg - 1];
      let html = `Train RMSE: <strong>${fit.train_rmse.toFixed(3)}</strong>`;
      if (state.showTest) {
        html += `&ensp;&middot;&ensp;Test RMSE: <strong>${fit.test_rmse.toFixed(3)}</strong>`;
      }
      document.getElementById(p.rmseId).innerHTML = html;
    }
    bestTitle.textContent = state.showTest
      ? `Best fit — degree ${state.data.best_degree}`
      : 'Best fit';
  }

  // ── Full render ───────────────────────────────────────────────────────────
  function renderAll() {
    renderMainChart();
    updateMainRmse();
    renderPanels();
    renderRmseChart();
  }

  // ── Init (only when demo section is first shown) ──────────────────────────
  let bootstrapped = false;

  function bootstrap() {
    if (!bootstrapped) {
      bootstrapped = true;
      fetchAndRender();
    }
  }

  // Expose so main.js can call it on nav activation
  window._overfittingBootstrap = bootstrap;
})();
