(function () {
  'use strict';

  var root = document.querySelector('[data-visitor-stats]');
  if (!root || root.dataset.initialized) return;
  root.dataset.initialized = 'true';

  var isProduction = window.location.origin === root.dataset.origin;
  var isLoopback = function (hostname) {
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
  };

  async function visitorId() {
    function loadOrCreate() {
      try {
        var key = 'dianyo.visitor-id.v1';
        var id = localStorage.getItem(key);
        if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id || '')) {
          id = crypto.randomUUID();
          localStorage.setItem(key, id);
        }
        return id;
      } catch (_) {
        // If storage is blocked, read counts without creating a fresh identity
        // on every reload. Never fall back to fingerprinting or IP identity.
        return null;
      }
    }
    // Coordinate the first visit across tabs before either creates a new ID.
    try {
      return navigator.locks ? await navigator.locks.request('dianyo.visitor-id.v1', loadOrCreate) : loadOrCreate();
    } catch (_) { return null; }
  }

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

    var id = await visitorId();
    var base = endpoint.href.replace(/\/$/, '');
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 8000);
    try {
      var response = await fetch(id ? base + '/visit' : base + '/counts?path=' + encodeURIComponent(root.dataset.path), {
        method: id ? 'POST' : 'GET',
        headers: id ? { 'Content-Type': 'application/json' } : {},
        body: id ? JSON.stringify({ path: root.dataset.path, visitorId: id }) : undefined,
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
    // Explicit local preview mode includes the real map; arbitrary preview
    // domains still never load either tracker.
    var localPreview = root.dataset.allowLocal === 'true' && isLoopback(window.location.hostname);
    if ((!isProduction && !localPreview) || !root.dataset.mapScript) return;
    var url;
    try { url = new URL(root.dataset.mapScript); } catch (_) { return; }
    if (url.protocol !== 'https:' || url.hostname !== 'mapmyvisitors.com' ||
        url.pathname !== '/map.js' || !url.searchParams.get('d')) return;

    var map = root.querySelector('[data-visitor-map]');
    var embed = root.querySelector('[data-map-embed]');
    var status = root.querySelector('[data-map-status]');
    // Show geography only. The provider's pageview total has its own counting
    // policy and would contradict our once-per-day numeric counters.
    url.searchParams.set('t', 'n');
    var observer = new MutationObserver(function () {
      if (embed.querySelector('.mapmyvisitors-map svg')) {
        clearTimeout(timeout);
        observer.disconnect();
        status.hidden = true;
      }
    });
    function unavailable() {
      clearTimeout(timeout);
      observer.disconnect();
      embed.hidden = true;
      status.textContent = 'Visitor map is unavailable right now.';
      status.hidden = false;
    }
    var timeout = setTimeout(unavailable, 12000);
    observer.observe(embed, { childList: true, subtree: true });
    var script = document.createElement('script');
    script.id = 'mapmyvisitors'; // Required by the provider's embed script.
    script.src = url.href;
    script.async = true;
    script.onerror = unavailable;
    map.hidden = false;
    embed.appendChild(script);
  }

  loadCounts();
  loadMap();
})();
