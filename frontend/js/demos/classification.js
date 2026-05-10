(() => {
  // ── Constants ───────────────────────────────────────────────────────────────
  const C_POS    = '#ff7f0e';   // class 1 (square) — matplotlib orange
  const C_NEG    = '#1f77b4';   // class 0 (circle) — matplotlib blue
  const C_MISS   = '#d62728';   // misclassification cross — matplotlib red
  const C_POS_BG = 'rgba(255,127,14,0.12)';
  const C_NEG_BG = 'rgba(31,119,180,0.12)';
  const N_BINS   = 50;
  const THRESH_STEPS = 200;     // precomputed threshold resolution

  // ── State ───────────────────────────────────────────────────────────────────
  const state = {
    probs:     [],
    labels:    [],
    threshold: 0.5,
    bins:      [],   // [{prob, label, binIdx, stackPos}]
    curves:    null, // precomputed {thresholds, accuracy, precision, recall}
    chart:     null, // Plotly chart reference
  };

  // ── Canvas setup ────────────────────────────────────────────────────────────
  const canvas  = document.getElementById('cl-canvas');
  const ctx     = canvas.getContext('2d');

  function resizeCanvas() {
    const wrap = canvas.parentElement;
    canvas.width  = wrap.clientWidth || 800;
    canvas.height = 140;
  }

  // ── Binning ─────────────────────────────────────────────────────────────────
  function buildBins(probs, labels) {
    const stackCount = new Array(N_BINS).fill(0);
    return probs.map((p, i) => {
      const binIdx   = Math.min(Math.floor(p * N_BINS), N_BINS - 1);
      const stackPos = stackCount[binIdx]++;
      return { prob: p, label: labels[i], binIdx, stackPos };
    });
  }

  // ── Metrics ─────────────────────────────────────────────────────────────────
  function computeMetrics(probs, labels, threshold) {
    let TP = 0, FP = 0, TN = 0, FN = 0;
    for (let i = 0; i < probs.length; i++) {
      const pred = probs[i] >= threshold ? 1 : 0;
      if (pred === 1 && labels[i] === 1) TP++;
      else if (pred === 1 && labels[i] === 0) FP++;
      else if (pred === 0 && labels[i] === 0) TN++;
      else FN++;
    }
    const total = probs.length;
    const acc   = total > 0 ? (TP + TN) / total : 0;
    const prec  = (TP + FP) > 0 ? TP / (TP + FP) : 0;
    const rec   = (TP + FN) > 0 ? TP / (TP + FN) : 0;
    return { TP, FP, TN, FN, acc, prec, rec };
  }

  function precomputeCurves(probs, labels) {
    const thresholds = [], accuracy = [], precision = [], recall = [];
    for (let i = 0; i <= THRESH_STEPS; i++) {
      const t = i / THRESH_STEPS;
      const m = computeMetrics(probs, labels, t);
      thresholds.push(t);
      accuracy.push(m.acc);
      precision.push(m.prec);
      recall.push(m.rec);
    }
    return { thresholds, accuracy, precision, recall };
  }

  // ── Drawing ──────────────────────────────────────────────────────────────────
  const DOT_R = 6;
  const PAD_L = 24;
  const PAD_R = 24;
  const PAD_TOP = 10;

  function probToX(p) {
    return PAD_L + p * (canvas.width - PAD_L - PAD_R);
  }

  function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const tx = probToX(state.threshold);

    // Background regions
    ctx.fillStyle = C_NEG_BG;
    ctx.fillRect(PAD_L, 0, tx - PAD_L, H);
    ctx.fillStyle = C_POS_BG;
    ctx.fillRect(tx, 0, W - PAD_R - tx, H);

    // Axis line
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_L, H - 18);
    ctx.lineTo(W - PAD_R, H - 18);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#888';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    for (let v = 0; v <= 10; v++) {
      const x = probToX(v / 10);
      ctx.fillText((v / 10).toFixed(1), x, H - 4);
      ctx.strokeStyle = '#eee';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, H - 20);
      ctx.lineTo(x, H - 14);
      ctx.stroke();
    }

    // "← circle" and "square →" region labels
    ctx.font = '11px system-ui';
    ctx.fillStyle = C_NEG;
    ctx.textAlign = 'left';
    if (tx > PAD_L + 50) ctx.fillText('← predict ○', PAD_L + 4, 27);
    ctx.fillStyle = C_POS;
    ctx.textAlign = 'right';
    if (tx < W - PAD_R - 50) ctx.fillText('predict ■ →', W - PAD_R - 4, 27);

    // Dots
    const maxStack = Math.max(...state.bins.map(b => b.stackPos), 0);
    const dotArea  = H - 32;   // available height above axis
    const step     = Math.min(DOT_R * 2 + 2, dotArea / Math.max(maxStack + 1, 1));

    for (const { prob, label, binIdx, stackPos } of state.bins) {
      const x = probToX((binIdx + 0.5) / N_BINS);
      const y = H - 22 - stackPos * step - DOT_R;

      const predicted = prob >= state.threshold ? 1 : 0;
      const isMiss    = predicted !== label;
      const color     = label === 1 ? C_POS : C_NEG;

      if (label === 1) {
        // Square
        ctx.fillStyle   = color;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.rect(x - DOT_R, y - DOT_R, DOT_R * 2, DOT_R * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        // Circle
        ctx.fillStyle   = color;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.arc(x, y, DOT_R, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }

      if (isMiss) {
        drawCross(x, y, DOT_R - 1);
      }
    }

    // Threshold line
    ctx.strokeStyle = '#333';
    ctx.lineWidth   = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(tx, 0);
    ctx.lineTo(tx, H - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // Threshold handle triangle
    ctx.fillStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(tx - 7, H - 20);
    ctx.lineTo(tx + 7, H - 20);
    ctx.lineTo(tx, H - 10);
    ctx.closePath();
    ctx.fill();

    // Threshold label — top, right of line when space allows, else left
    ctx.fillStyle = '#333';
    ctx.font      = 'bold 12px system-ui';
    const TEXT_W  = 60;  // approx pixel width of "t = 0.00"
    if (tx + 6 + TEXT_W <= W - PAD_R) {
      ctx.textAlign = 'left';
      ctx.fillText(`t = ${state.threshold.toFixed(2)}`, tx + 6, 13);
    } else {
      ctx.textAlign = 'right';
      ctx.fillText(`t = ${state.threshold.toFixed(2)}`, Math.max(PAD_L + TEXT_W, tx - 6), 13);
    }
  }

  function drawCross(x, y, r) {
    ctx.strokeStyle = C_MISS;
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
    ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
    ctx.stroke();
  }

  // ── Metrics chart (Plotly) ────────────────────────────────────────────────
  function initChart() {
    const { thresholds, accuracy, precision, recall } = state.curves;
    const t = state.threshold;

    const traces = [
      { x: thresholds, y: accuracy,  name: 'Accuracy',  line: { color: '#111',    width: 2 } },
      { x: thresholds, y: precision, name: 'Precision', line: { color: C_POS,     width: 2 } },
      { x: thresholds, y: recall,    name: 'Recall',    line: { color: C_NEG,     width: 2 } },
    ];

    const layout = {
      height: 220,
      margin: { t: 10, r: 20, b: 40, l: 44 },
      xaxis: { title: 'Threshold', range: [0, 1], tickformat: '.1f' },
      yaxis: { range: [0, 1.05], tickformat: '.2f' },
      legend: { orientation: 'h', y: -0.28 },
      shapes: [thresholdShape(t)],
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor:  '#fafafa',
    };

    Plotly.newPlot('cl-metrics-chart', traces, layout, { displayModeBar: false, responsive: true });
    state.chart = true;
  }

  function thresholdShape(t) {
    return {
      type: 'line', x0: t, x1: t, y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: { color: '#333', width: 2, dash: 'dot' },
    };
  }

  function updateChartThreshold(t) {
    Plotly.relayout('cl-metrics-chart', { shapes: [thresholdShape(t)] });
  }

  // ── DOM update ────────────────────────────────────────────────────────────
  function updateDOM(metrics) {
    const { TP, FP, TN, FN, acc, prec, rec } = metrics;
    const P    = TP + FN;
    const N    = TN + FP;
    const Pp   = TP + FP;
    const Np   = TN + FN;
    const total = TP + FP + TN + FN;

    const map = { TP, FP, TN, FN, P, N, PP: Pp, NP: Np, TOTAL: total };

    // Update all [data-metric] elements (confusion matrix cells + formula vars)
    document.querySelectorAll('[data-metric]').forEach(el => {
      const key = el.dataset.metric;
      const val = map[key];
      if (val === undefined) return;
      const countEl = el.querySelector('.count') || el.querySelector('.cl-cell-count');
      if (countEl) countEl.textContent = val;
    });

    // Scalar metric values
    document.getElementById('cl-acc-val').textContent  = acc  > 0 || TP + TN > 0 ? acc.toFixed(2)  : '—';
    document.getElementById('cl-prec-val').textContent = (TP + FP) > 0 ? prec.toFixed(2) : '—';
    document.getElementById('cl-rec-val').textContent  = (TP + FN) > 0 ? rec.toFixed(2)  : '—';

    // Flowchart SVG counts
    setText('fl-total',  total);
    setText('fl-p',      P);
    setText('fl-n',      N);
    setText('fl-tp',     TP);
    setText('fl-fn',     FN);
    setText('fl-fp',     FP);
    setText('fl-tn',     TN);
    setText('fl-pp',     Pp);
    setText('fl-np',     Np);
    setText('fl-total2', total);
  }

  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // ── Threshold interaction ────────────────────────────────────────────────
  let dragging = false;

  function thresholdFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const x = clientX - rect.left;
    const t = (x - PAD_L) / (canvas.width - PAD_L - PAD_R);
    return Math.max(0, Math.min(1, t));
  }

  canvas.addEventListener('mousedown', e => { dragging = true; onMove(e); });
  canvas.addEventListener('mousemove', e => { if (dragging) onMove(e); });
  window.addEventListener('mouseup',   () => { dragging = false; });
  canvas.addEventListener('touchstart', e => { dragging = true; onMove(e); e.preventDefault(); }, { passive: false });
  canvas.addEventListener('touchmove',  e => { if (dragging) { onMove(e); e.preventDefault(); } }, { passive: false });
  window.addEventListener('touchend',   () => { dragging = false; });

  function onMove(e) {
    state.threshold = thresholdFromEvent(e);
    const m = computeMetrics(state.probs, state.labels, state.threshold);
    draw();
    updateDOM(m);
    if (state.chart) updateChartThreshold(state.threshold);
  }

  // ── Tooltip ──────────────────────────────────────────────────────────────
  const tooltip = (() => {
    const el = document.createElement('div');
    el.id = 'cl-tooltip';
    el.style.cssText = 'position:fixed;background:#333;color:#fff;padding:4px 8px;border-radius:4px;font-size:12px;pointer-events:none;display:none;z-index:1000;max-width:220px;line-height:1.4';
    document.body.appendChild(el);
    return el;
  })();

  function attachTooltips(root) {
    root.addEventListener('mouseover', e => {
      const target = e.target.closest('[data-tooltip]');
      if (!target) return;
      tooltip.textContent = target.dataset.tooltip;
      tooltip.style.display = 'block';
    });
    root.addEventListener('mousemove', e => {
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top  = (e.clientY - 28) + 'px';
    });
    root.addEventListener('mouseout', e => {
      if (!e.target.closest('[data-tooltip]')) return;
      tooltip.style.display = 'none';
    });
  }

  // ── Dataset loading ──────────────────────────────────────────────────────
  async function loadDataset(name) {
    const res  = await fetch(`/api/classification/dataset/${name}`);
    const data = await res.json();
    state.probs  = data.probs;
    state.labels = data.labels;
    state.bins   = buildBins(state.probs, state.labels);
    state.curves = precomputeCurves(state.probs, state.labels);
    state.threshold = 0.5;

    const m = computeMetrics(state.probs, state.labels, state.threshold);
    draw();
    updateDOM(m);

    if (!state.chart) {
      initChart();
    } else {
      const { thresholds, accuracy, precision, recall } = state.curves;
      Plotly.react('cl-metrics-chart',
        [
          { x: thresholds, y: accuracy,  name: 'Accuracy',  line: { color: '#111',    width: 2 } },
          { x: thresholds, y: precision, name: 'Precision', line: { color: C_POS,     width: 2 } },
          { x: thresholds, y: recall,    name: 'Recall',    line: { color: C_NEG,     width: 2 } },
        ],
        {
          height: 220,
          margin: { t: 10, r: 20, b: 40, l: 44 },
          xaxis: { title: 'Threshold', range: [0, 1], tickformat: '.1f' },
          yaxis: { range: [0, 1.05], tickformat: '.2f' },
          legend: { orientation: 'h', y: -0.28 },
          shapes: [thresholdShape(state.threshold)],
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor:  '#fafafa',
        }
      );
    }
    updateChartThreshold(state.threshold);
  }

  // ── Bootstrap ────────────────────────────────────────────────────────────
  let _booted = false;

  function bootstrap() {
    resizeCanvas();
    if (!_booted) {
      _booted = true;
      const section = document.getElementById('demo-classification');
      attachTooltips(section);
      document.getElementById('cl-dataset').addEventListener('change', e => {
        loadDataset(e.target.value);
      });
      loadDataset('unseparated');
    } else {
      draw();
    }
  }

  window.addEventListener('resize', () => {
    if (!document.getElementById('demo-classification').classList.contains('active')) return;
    resizeCanvas();
    draw();
  });

  window._classificationBootstrap = bootstrap;
})();
