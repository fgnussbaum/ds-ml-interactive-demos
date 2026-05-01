document.querySelectorAll('#sidebar a[data-demo]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const demo = e.currentTarget.dataset.demo;
    document.querySelectorAll('#sidebar a').forEach(a => a.classList.remove('active'));
    document.querySelectorAll('.demo-section').forEach(s => s.classList.remove('active'));
    e.currentTarget.classList.add('active');
    document.getElementById('demo-' + demo)?.classList.add('active');
    if (demo === 'overfitting') window._overfittingBootstrap?.();
  });
});
