function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = resolve;
    el.onerror = reject;
    document.head.appendChild(el);
  });
}

(async () => {
  // 1. Inject partials in order so demo scripts find their DOM elements
  const content = document.getElementById('content');
  const partials = ['regression', 'overfitting', 'classification', 'decision_tree', 'gradient_descent'];
  for (const name of partials) {
    const html = await fetch(`/partials/${name}.html`).then(r => r.text());
    content.insertAdjacentHTML('beforeend', html);
  }

  // 2. Load demo scripts in order (each IIFE runs after its section is in the DOM)
  for (const src of [
    '/js/demos/regression.js',
    '/js/demos/overfitting.js',
    '/js/demos/classification.js',
    '/js/demos/decision_tree.js',
    '/js/demos/gradient_descent.js',
  ]) {
    await _loadScript(src);
  }

  // 3. Wire nav routing
  document.querySelectorAll('#sidebar a[data-demo]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const demo = e.currentTarget.dataset.demo;
      document.querySelectorAll('#sidebar a').forEach(a => a.classList.remove('active'));
      document.querySelectorAll('.demo-section').forEach(s => s.classList.remove('active'));
      e.currentTarget.classList.add('active');
      document.getElementById('start-page').style.display = 'none';
      document.getElementById('demo-' + demo)?.classList.add('active');
      if (demo === 'overfitting')        window._overfittingBootstrap?.();
      if (demo === 'classification')     window._classificationBootstrap?.();
      if (demo === 'decision_tree')      window._decisionTreeBootstrap?.();
      if (demo === 'gradient_descent')   window._gradientDescentBootstrap?.();
    });
  });
})().catch(err => console.error('[main] failed to initialise:', err));
