// Resolve repository links for both GitHub Pages and an anonymous mirror.
(() => {
  const url = new URL(window.location.href);
  const mirror = url.pathname.match(/^\/w\/([^/]+)(?:\/|$)/);
  let repository;
  let benchmark;
  if (mirror) {
    repository = `/r/${mirror[1]}/`;
    benchmark = `${repository}benchmark/`;
  } else if (url.hostname.endsWith('.github.io')) {
    const owner = url.hostname.slice(0, -'.github.io'.length);
    const name = url.pathname.split('/').filter(Boolean)[0];
    repository = `https://github.com/${owner}/${name}`;
    benchmark = `${repository}/tree/main/benchmark`;
  }
  if (repository) {
    document.querySelectorAll('[data-repository-link]').forEach(link => {
      link.href = link.dataset.repositoryLink === 'benchmark' ? benchmark : repository;
    });
  }
})();
