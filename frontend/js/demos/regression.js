(function () {
  const canvas = document.getElementById('regression-canvas');
  const ctx = canvas.getContext('2d');
  const POINT_RADIUS = 8;
  const DRAG_THRESHOLD_SQ = 20 * 20;

  const state = {
    points: [],      // [[x, y], ...]
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
    drawLine(state.lines.ls, '#cc0000');
    drawLine(state.lines.rr, '#9a7000');
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
      { line: state.lines.ls, color: '#cc0000' },
      { line: state.lines.rr, color: '#9a7000' },
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
      ctx.fillStyle = i === state.dragIndex ? '#1a6fa0' : '#4682b4';
      ctx.fill();
    });
  }
})();
