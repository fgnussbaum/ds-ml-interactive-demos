(function () {
  const canvas = document.getElementById('regression-canvas');
  const ctx = canvas.getContext('2d');
  const POINT_RADIUS = 8;
  const DRAG_THRESHOLD_SQ = 20 * 20;

  // matplotlib default color cycle
  const COLOR_LS = '#1f77b4';  // C0 blue  — OLS
  const COLOR_RR = '#ff7f0e';  // C1 orange — Robust / LAD
  const COLOR_PT = '#555';
  const COLOR_PT_DRAG = '#999';

  const SEED_POINTS = [
    [100, 320], [160, 295], [230, 278], [310, 255],
    [385, 235], [450, 215], [525, 198], [600, 182], [675, 162],
    [350, 425],  // outlier
  ];

  const state = {
    points: [],
    dragIndex: -1,
    lines: { ls: null, rr: null },
  };

  // ── Events ────────────────────────────────────────────────────────────────

  canvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    const { x, y } = canvasPos(e);
    state.points.push([x, y]);
    refitAndDraw();
  });

  canvas.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    const { x, y } = canvasPos(e);
    state.dragIndex = nearestPoint(x, y);
  });

  canvas.addEventListener('mousemove', e => {
    if (state.dragIndex === -1) return;
    const { x, y } = canvasPos(e);
    state.points[state.dragIndex] = [x, y];
    refitAndDraw();
  });

  canvas.addEventListener('mouseup', () => { state.dragIndex = -1; });
  canvas.addEventListener('mouseleave', () => { state.dragIndex = -1; });

  document.getElementById('show-ls').addEventListener('change', refitAndDraw);
  document.getElementById('show-rr').addEventListener('change', refitAndDraw);
  document.getElementById('show-devs').addEventListener('change', draw);

  document.getElementById('seed-btn').addEventListener('click', () => {
    state.points = SEED_POINTS.map(p => [...p]);
    refitAndDraw();
  });

  document.getElementById('clear-btn').addEventListener('click', () => {
    state.points = [];
    state.lines = { ls: null, rr: null };
    draw();
  });

  // ── Helpers ───────────────────────────────────────────────────────────────

  function canvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  function nearestPoint(x, y) {
    let best = -1, bestDist = DRAG_THRESHOLD_SQ;
    state.points.forEach(([px, py], i) => {
      const d = (px - x) ** 2 + (py - y) ** 2;
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  }

  // ── API ───────────────────────────────────────────────────────────────────

  async function refitAndDraw() {
    if (state.points.length < 2) {
      state.lines = { ls: null, rr: null };
      draw();
      return;
    }
    const res = await fetch('/api/regression/fit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: state.points,
        show_ls: document.getElementById('show-ls').checked,
        show_rr: document.getElementById('show-rr').checked,
      }),
    });
    state.lines = await res.json();
    draw();
  }

  // ── Drawing ───────────────────────────────────────────────────────────────

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (document.getElementById('show-devs').checked) drawResiduals();
    drawLine(state.lines.ls, COLOR_LS);
    drawLine(state.lines.rr, COLOR_RR);
    drawPoints();
  }

  function drawLine(line, color) {
    if (!line) return;
    const { slope: m, intercept: b } = line;
    ctx.beginPath();
    ctx.moveTo(0, b);
    ctx.lineTo(canvas.width, canvas.width * m + b);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.stroke();
  }

  function drawResiduals() {
    [
      { line: state.lines.ls, color: COLOR_LS },
      { line: state.lines.rr, color: COLOR_RR },
    ].forEach(({ line, color }) => {
      if (!line) return;
      const { slope: m, intercept: b } = line;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      state.points.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, m * x + b);
        ctx.stroke();
      });
      ctx.setLineDash([]);
    });
  }

  function drawPoints() {
    state.points.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = i === state.dragIndex ? COLOR_PT_DRAG : COLOR_PT;
      ctx.fill();
    });
  }

  // ── KaTeX formulas ────────────────────────────────────────────────────────

  function renderFormulas() {
    const opts = { throwOnError: false };
    katex.render(String.raw`\min_{\beta} \sum_{i} r_i^2`, document.getElementById('formula-ols'), { ...opts, displayMode: true });
    katex.render(String.raw`\min_{\beta} \sum_{i} |r_i|`,  document.getElementById('formula-lad'), { ...opts, displayMode: true });
    katex.render(String.raw`r_i = y_i - \hat{y}_i`,        document.getElementById('formula-residual'), opts);
  }

  // ── Loss function comparison plot ─────────────────────────────────────────

  function renderLossPlot() {
    const xs = [];
    for (let x = -2.5; x <= 2.5; x += 0.05) xs.push(Math.round(x * 100) / 100);
    Plotly.newPlot('regression-loss-plot', [
      {
        x: xs, y: xs.map(x => x * x),
        mode: 'lines', name: 'L2 (squared)',
        line: { color: COLOR_LS, width: 2.5 },
      },
      {
        x: xs, y: xs.map(x => Math.abs(x)),
        mode: 'lines', name: 'L1 (absolute)',
        line: { color: COLOR_RR, width: 2.5 },
      },
    ], {
      margin: { t: 10, b: 45, l: 55, r: 10 },
      height: 220,
      xaxis: { title: 'residual rᵢ', zeroline: true, zerolinecolor: '#ccc' },
      yaxis: { title: 'loss', zeroline: true, zerolinecolor: '#ccc' },
      legend: { orientation: 'h', x: 0, y: 1.15 },
      plot_bgcolor: '#fff',
      paper_bgcolor: '#fff',
    }, { responsive: true, displayModeBar: false });
  }

  renderFormulas();
  renderLossPlot();
})();
