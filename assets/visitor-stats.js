(function () {
  'use strict';

  var root = document.querySelector('[data-visitor-stats]');
  if (!root || root.dataset.initialized) return;
  root.dataset.initialized = 'true';

  var isProduction = window.location.origin === root.dataset.origin;
  var isLoopback = function (hostname) {
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
  };

  async function loadCounts() {
    if (!root.dataset.endpoint) return;
    var endpoint;
    try {
      endpoint = new URL(root.dataset.endpoint);
    } catch (_) { return; }

    // Preview traffic must never reach the production counter.
    var isLocal = root.dataset.allowLocal === 'true' &&
      isLoopback(window.location.hostname) && isLoopback(endpoint.hostname);
    if (!isProduction && !isLocal) return;
    if (endpoint.protocol !== 'https:' && !(isLocal && endpoint.protocol === 'http:')) return;

    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 8000);
    try {
      var response = await fetch(endpoint.href.replace(/\/$/, '') + '/visit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: root.dataset.path }),
        credentials: 'omit',
        cache: 'no-store',
        signal: controller.signal
      });
      if (!response.ok) return;
      var counts = await response.json();
      if (!Number.isSafeInteger(counts.siteViews) || counts.siteViews < 0 ||
          !Number.isSafeInteger(counts.pageViews) || counts.pageViews < 0) return;

      root.querySelector('[data-site-views]').textContent = counts.siteViews.toLocaleString();
      root.querySelector('[data-site-views-label]').hidden = false;
      var article = document.querySelector('[data-article-views]');
      if (article) {
        article.querySelector('[data-page-views]').textContent = counts.pageViews.toLocaleString();
        article.hidden = false;
      }
    } catch (_) {
      // Counts are optional: keep the article readable if the service is unavailable.
    } finally {
      clearTimeout(timeout);
    }
  }

  function loadMap() {
    // The map has its own tracker. Never load it on localhost or a preview domain.
    if (!isProduction || !root.dataset.mapScript) return;
    var url;
    try { url = new URL(root.dataset.mapScript); } catch (_) { return; }
    if (url.protocol !== 'https:' || url.hostname !== 'mapmyvisitors.com' ||
        url.pathname !== '/map.js' || !url.searchParams.get('d')) return;

    var map = root.querySelector('[data-visitor-map]');
    var script = document.createElement('script');
    script.id = 'mapmyvisitors'; // Required by the provider's embed script.
    script.src = url.href;
    script.async = true;
    script.onerror = function () { map.hidden = true; };
    map.hidden = false;
    root.querySelector('[data-map-embed]').appendChild(script);
  }

  loadCounts();
  loadMap();
})();
