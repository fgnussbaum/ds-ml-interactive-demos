(function () {
  // ── Constants ─────────────────────────────────────────────────────────────
  const C_CURVE   = '#888888';
  const C_LINE    = '#e84040';
  const C_OLS     = '#2ca02c';
  const C_ARROW   = '#e07b39';
  const C_DATA    = '#4a9eff';
  const C_PATH    = '#1a1a8c';
  const ANIM_MS   = 120;
  const PLOTLY_CFG = { responsive: true, displayModeBar: false };

  // ── State ─────────────────────────────────────────────────────────────────
  const state = {
    theta:      0,
    data:       null,
    trajectory: null,
    gdStep:     0,      // index of the next step to consume from trajectory
    animTimer:  null,
    isRunning:  false,
    ready:      { loss: false, scatter: false, contour: false },
    xMin: 0, xMax: 0, yMin: 0, yMax: 0,
    lossXRange: null,   // fixed axis bounds for the loss plot, computed once from data at init
    lossYRange: null,
    ghostTheta: null,   // θ from the step before the last GD update; null when user set θ manually
  };

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const thetaSlider = document.getElementById('gd-theta-slider');
  const mseVal      = document.getElementById('gd-mse-val');
  const gradVal     = document.getElementById('gd-grad-val');
  const runBtn      = document.getElementById('gd-run-btn');
  const stepBtn     = document.getElementById('gd-step-btn');
  const resetBtn    = document.getElementById('gd-reset-btn');
  const stepVal     = document.getElementById('gd-step-val');

  // ── Event listeners ───────────────────────────────────────────────────────
  thetaSlider.addEventListener('input', () => {
    if (state.isRunning) return;
    state.theta = parseFloat(thetaSlider.value);
    clearTrajectory();
    renderLoss(state.theta);
    renderScatter(state.theta);
    updateStats(state.theta);
  });

  // LR change invalidates any loaded trajectory so the next run uses the new rate
  document.querySelectorAll('input[name="gd-lr"]').forEach(r => {
    r.addEventListener('change', () => { if (!state.isRunning) clearTrajectory(); });
  });

  runBtn.addEventListener('click', async () => {
    if (state.isRunning) { stopAnimation(); return; }
    if (!state.trajectory || state.gdStep >= state.trajectory.length) {
      await loadTrajectory(currentPreset());
    }
    if (state.trajectory) startAnimation();
  });

  stepBtn.addEventListener('click', async () => {
    if (state.isRunning) return;
    if (!state.trajectory || state.gdStep >= state.trajectory.length) {
      await loadTrajectory(currentPreset());
    }
    if (state.trajectory && state.gdStep < state.trajectory.length) advanceStep();
  });

  resetBtn.addEventListener('click', resetAll);

  // ── API calls ─────────────────────────────────────────────────────────────
  async function init() {
    const res = await fetch('/api/gradient_descent/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    state.data = await res.json();
    const d = state.data;

    state.theta       = d.theta_start;
    thetaSlider.min   = d.theta_range[0];
    thetaSlider.max   = d.theta_range[d.theta_range.length - 1];
    thetaSlider.step  = (d.theta_range[d.theta_range.length - 1] - d.theta_range[0]) / 400;
    thetaSlider.value = d.theta_start;

    state.xMin = Math.min(...d.x_data);
    state.xMax = Math.max(...d.x_data);
    state.yMin = Math.min(...d.y_data);
    state.yMax = Math.max(...d.y_data);

    state.lossXRange = [d.theta_range[0], d.theta_range[d.theta_range.length - 1]];
    const lcMin = Math.min(...d.loss_curve);
    const lcMax = Math.max(...d.loss_curve);
    const lcPad = (lcMax - lcMin) * 0.06;
    state.lossYRange = [lcMin - lcPad, lcMax + lcPad];

    renderLoss(state.theta);
    renderScatter(state.theta);
    renderContour([]);
    updateStats(state.theta);
  }

  async function loadTrajectory(preset) {
    const res = await fetch('/api/gradient_descent/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lr_preset: preset, theta_start: state.theta }),
    });
    const result = await res.json();
    state.trajectory = result.trajectory;
    // trajectory[0] is the starting position (= current theta, already displayed).
    // Start at index 1 so the first advanceStep() shows actual movement.
    state.gdStep = 1;
    if (state.ready.contour && state.trajectory.length > 0) {
      const s0 = state.trajectory[0];
      // Seed the path with just the starting dot; update the start marker
      Plotly.restyle('gd-contour', {
        x: [[s0.intercept_2d]], y: [[s0.slope_2d]], mode: ['markers'],
      }, [1]);
      Plotly.restyle('gd-contour', { x: [[s0.intercept_2d]], y: [[s0.slope_2d]] }, [3]);
    }
  }

  // ── Animation ─────────────────────────────────────────────────────────────
  function advanceStep() {
    state.ghostTheta = state.theta;
    const s = state.trajectory[state.gdStep];
    state.theta       = s.theta;
    thetaSlider.value = s.theta;
    renderLoss(s.theta);
    renderScatter(s.theta);
    updateContourPath(state.gdStep);
    stepVal.textContent = state.gdStep;   // gdStep starts at 1 = number of updates applied
    mseVal.textContent  = s.loss_1d.toFixed(5);
    gradVal.textContent = s.gradient.toFixed(4);
    state.gdStep++;
  }

  function startAnimation() {
    state.isRunning      = true;
    runBtn.innerHTML     = '&#9632; Stop GD';
    runBtn.classList.add('gd-running');
    scheduleNext();
  }

  function stopAnimation() {
    if (state.animTimer) { clearTimeout(state.animTimer); state.animTimer = null; }
    state.isRunning      = false;
    runBtn.innerHTML     = '&#9654; Run GD';
    runBtn.classList.remove('gd-running');
  }

  function scheduleNext() {
    if (!state.isRunning) return;
    if (!state.trajectory || state.gdStep >= state.trajectory.length) {
      stopAnimation(); return;
    }
    advanceStep();
    state.animTimer = setTimeout(scheduleNext, ANIM_MS);
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  function renderLoss(theta) {
    const d    = state.data;
    const mse  = clientMSE(theta);
    const grad = clientGrad(theta);

    // Arrow in the -gradient direction; length proportional to |grad| (may extend past axis)
    const annotations = [];

    // Ghost annotation: previous GD position (only after a step, not after manual θ set)
    if (state.ghostTheta !== null) {
      const gMSE  = clientMSE(state.ghostTheta);
      const gGrad = clientGrad(state.ghostTheta);
      if (Math.abs(gGrad) > 0.001) {
        const gdTheta = -gGrad;
        annotations.push({
          x: state.ghostTheta + gdTheta, y: gMSE + gGrad * gdTheta,
          ax: state.ghostTheta,          ay: gMSE,
          xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
          showarrow: false,
          arrowhead: 3, arrowsize: 1.2, arrowwidth: 2.5,
          arrowcolor: 'rgba(224,123,57,0.30)',
          text: '',
        });
      }
    }

    if (Math.abs(grad) > 0.001) {
      const dTheta = -grad;              // θ-displacement = gradient magnitude (proportional)
      const dMSE   = grad * dTheta;      // tangent approximation (always ≤ 0)
      annotations.push({
        x: theta + dTheta, y: mse + dMSE,
        ax: theta,         ay: mse,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        showarrow: true,
        arrowhead: 3, arrowsize: 1.2, arrowwidth: 2.5,
        arrowcolor: C_ARROW,
        text: '',
      });
    }

    const olsMSE = clientMSE(d.theta_ols);
    const traces = [
      {
        x: d.theta_range, y: d.loss_curve,
        mode: 'lines', type: 'scatter',
        line: { color: C_CURVE, width: 2 },
        name: 'MSE', showlegend: false,
        hovertemplate: 'θ=%{x:.3f}<br>MSE=%{y:.5f}<extra></extra>',
      },
      {
        x: [d.theta_ols], y: [olsMSE],
        mode: 'markers', type: 'scatter',
        marker: { color: C_OLS, size: 13, symbol: 'star', line: { color: '#fff', width: 1 } },
        name: 'OLS minimum',
        hovertemplate: 'OLS: θ=%{x:.4f}<br>MSE=%{y:.5f}<extra></extra>',
      },
    ];

    // Ghost marker: previous θ at 30 % opacity (only after a GD step)
    if (state.ghostTheta !== null) {
      traces.push({
        x: [state.ghostTheta], y: [clientMSE(state.ghostTheta)],
        mode: 'markers', type: 'scatter',
        marker: { color: 'rgba(232,64,64,0.30)', size: 10, symbol: 'circle' },
        name: 'prev θ', showlegend: false,
        hoverinfo: 'skip',
      });
    }

    traces.push({
      x: [theta], y: [mse],
      mode: 'markers', type: 'scatter',
      marker: { color: C_LINE, size: 10, symbol: 'circle' },
      name: 'current θ', showlegend: false,
      hovertemplate: 'θ=%{x:.3f}<br>MSE=%{y:.5f}<extra></extra>',
    });

    const { xrange: lxr, yrange: lyr } = state.ready.loss ? getCurrentRanges('gd-loss') : {};
    const layout = {
      margin: { l: 60, r: 20, t: 16, b: 52 },
      height: 280,
      xaxis: { title: 'θ (slope)', range: lxr ?? state.lossXRange },
      yaxis: { title: 'MSE', range: lyr ?? state.lossYRange },
      legend: { orientation: 'h', y: -0.24, x: 0 },
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      annotations,
    };

    if (!state.ready.loss) {
      Plotly.newPlot('gd-loss', traces, layout, PLOTLY_CFG);
      state.ready.loss = true;
      // Click anywhere on the loss curve to place θ (only when GD is not running)
      document.getElementById('gd-loss').on('plotly_click', (ev) => {
        if (state.isRunning || !ev.points.length) return;
        const clicked = ev.points[0].x;
        const lo = parseFloat(thetaSlider.min), hi = parseFloat(thetaSlider.max);
        const clamped = Math.max(lo, Math.min(hi, clicked));
        state.theta       = clamped;
        thetaSlider.value = clamped;
        clearTrajectory();
        renderLoss(clamped);
        renderScatter(clamped);
        updateStats(clamped);
      });
    } else {
      Plotly.react('gd-loss', traces, layout);
    }
  }

  function renderScatter(theta) {
    const d     = state.data;
    const lineX = [state.xMin, state.xMax];
    const lineY = [
      theta * state.xMin + d.intercept_ols,
      theta * state.xMax + d.intercept_ols,
    ];
    const olsY = [
      d.theta_ols * state.xMin + d.intercept_ols,
      d.theta_ols * state.xMax + d.intercept_ols,
    ];

    const traces = [
      {
        x: d.x_data, y: d.y_data,
        mode: 'markers', type: 'scatter',
        marker: { color: C_DATA, size: 5, opacity: 0.6 },
        name: 'tips data', showlegend: false,
        hovertemplate: 'x_std=%{x:.2f}<br>tip_rate=%{y:.3f}<extra></extra>',
      },
      {
        x: lineX, y: lineY,
        mode: 'lines', type: 'scatter',
        line: { color: C_LINE, width: 2 },
        name: 'current fit', showlegend: false,
      },
      {
        x: lineX, y: olsY,
        mode: 'lines', type: 'scatter',
        line: { color: C_OLS, width: 1.5, dash: 'dot' },
        name: 'OLS line', showlegend: false,
      },
    ];

    const pad  = 0.06;
    const { xrange: sxr, yrange: syr } = state.ready.scatter ? getCurrentRanges('gd-scatter') : {};
    const layout = {
      margin: { l: 60, r: 20, t: 16, b: 52 },
      height: 280,
      xaxis: { title: 'total_bill (standardised)', ...(sxr ? { range: sxr } : {}) },
      yaxis: { title: 'tip rate', range: syr ?? [state.yMin - pad, state.yMax + pad] },
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    };

    if (!state.ready.scatter) {
      Plotly.newPlot('gd-scatter', traces, layout, PLOTLY_CFG);
      state.ready.scatter = true;
    } else {
      Plotly.react('gd-scatter', traces, layout);
    }
  }

  function renderContour(path2d) {
    const d = state.data;

    const traces = [
      // contour surface (trace 0)
      {
        type: 'contour',
        x: d.b_range,
        y: d.slope_range,
        z: d.contour_z,
        colorscale: 'YlOrRd',
        reversescale: true,
        showscale: true,
        colorbar: { title: { text: 'MSE' }, thickness: 14, len: 0.8 },
        contours: { showlabels: true, labelfont: { size: 9 } },
        hovertemplate: 'b=%{x:.3f}<br>θ=%{y:.3f}<br>MSE=%{z:.5f}<extra></extra>',
        name: 'MSE',
      },
      // GD path (trace 1 — updated incrementally during animation)
      {
        x: path2d.map(s => s.intercept_2d),
        y: path2d.map(s => s.slope_2d),
        mode: path2d.length > 1 ? 'lines+markers' : 'markers',
        type: 'scatter',
        line:   { color: C_PATH, width: 2 },
        marker: { color: C_PATH, size: 5 },
        name: 'GD path', showlegend: false,
      },
      // OLS minimum (trace 2)
      {
        x: [d.intercept_ols], y: [d.theta_ols],
        mode: 'markers', type: 'scatter',
        marker: { symbol: 'star', size: 16, color: C_OLS, line: { color: '#fff', width: 1 } },
        name: 'OLS min',
        hovertemplate: 'OLS: b=%{x:.4f}<br>θ=%{y:.4f}<extra></extra>',
      },
      // start marker (trace 3 — updated by loadTrajectory to reflect actual run origin)
      {
        x: [d.b_start_2d], y: [d.theta_start],
        mode: 'markers+text', type: 'scatter',
        marker: { symbol: 'x-thin', size: 12, color: C_LINE, line: { color: C_LINE, width: 2.5 } },
        text: ['start'], textposition: 'top right', textfont: { size: 10, color: C_LINE },
        name: 'start', showlegend: false,
      },
    ];

    const layout = {
      margin: { l: 64, r: 80, t: 16, b: 52 },
      height: 360,
      xaxis: { title: 'intercept (b)' },
      yaxis: { title: 'slope (θ)' },
      legend: { orientation: 'h', y: -0.18, x: 0 },
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    };

    if (!state.ready.contour) {
      Plotly.newPlot('gd-contour', traces, layout, PLOTLY_CFG);
      state.ready.contour = true;
    } else {
      Plotly.react('gd-contour', traces, layout);
    }
  }

  function updateContourPath(step) {
    if (!state.ready.contour || !state.trajectory) return;
    const path = state.trajectory.slice(0, step + 1);
    Plotly.restyle('gd-contour', {
      x: [path.map(s => s.intercept_2d)],
      y: [path.map(s => s.slope_2d)],
      mode: [path.length > 1 ? 'lines+markers' : 'markers'],
    }, [1]);
  }

  // ── Client-side math (avoids round-trips for slider updates) ──────────────
  function clientMSE(theta) {
    const { x_data, y_data, intercept_ols } = state.data;
    let sum = 0;
    for (let i = 0; i < x_data.length; i++) {
      const r = theta * x_data[i] + intercept_ols - y_data[i];
      sum += r * r;
    }
    return sum / x_data.length;
  }

  function clientGrad(theta) {
    const { x_data, y_data, intercept_ols } = state.data;
    let sum = 0;
    for (let i = 0; i < x_data.length; i++) {
      const r = theta * x_data[i] + intercept_ols - y_data[i];
      sum += r * x_data[i];
    }
    return 2.0 * sum / x_data.length;
  }

  // ── UI helpers ─────────────────────────────────────────────────────────────
  // Returns the user's current zoom ranges for a Plotly div, or {} if not yet zoomed
  function getCurrentRanges(divId) {
    const div = document.getElementById(divId);
    if (!div || !div._fullLayout) return {};
    const xl = div._fullLayout.xaxis;
    const yl = div._fullLayout.yaxis;
    return {
      xrange: xl.autorange ? undefined : xl.range.slice(),
      yrange: yl.autorange ? undefined : yl.range.slice(),
    };
  }

  function currentPreset() {
    return document.querySelector('input[name="gd-lr"]:checked')?.value || 'good';
  }

  function clearTrajectory() {
    state.trajectory = null;
    state.gdStep     = 0;
    state.ghostTheta = null;
    if (state.ready.contour) {
      Plotly.restyle('gd-contour', { x: [[]], y: [[]], mode: ['markers'] }, [1]);
    }
  }

  function updateStats(theta) {
    mseVal.textContent  = clientMSE(theta).toFixed(5);
    gradVal.textContent = clientGrad(theta).toFixed(4);
    stepVal.textContent = '—';
  }

  function resetAll() {
    stopAnimation();
    clearTrajectory();
    if (state.data) {
      state.theta       = state.data.theta_start;
      thetaSlider.value = state.data.theta_start;
      renderLoss(state.theta);
      renderScatter(state.theta);
      renderContour([]);   // full re-render resets the start marker to server default
      updateStats(state.theta);
    }
  }

  // ── Bootstrap ─────────────────────────────────────────────────────────────
  let bootstrapped = false;

  function bootstrap() {
    if (!bootstrapped) {
      bootstrapped = true;
      init();
    }
  }

  window._gradientDescentBootstrap = bootstrap;
})();
