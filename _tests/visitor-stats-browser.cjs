// Build with a test-only endpoint before running; see _workers/visits/README.md.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { createRequire } = require('node:module');
const workerRequire = createRequire(path.resolve(__dirname, '../_workers/visits/package.json'));
const { chromium } = workerRequire('playwright');
const { Miniflare, convertV4MiniflareOptions } = workerRequire('miniflare');
const site = process.env.VISITOR_SITE_DIR || '/tmp/dianyo-visitors-test';
const origin = 'https://dianyo.github.io';
const endpoint = 'https://counter.test';
const localOrigin = 'http://localhost:4000';
const localEndpoint = 'http://127.0.0.1:8787';

(async () => {
  const mf = new Miniflare(convertV4MiniflareOptions({
    name: 'browser-test-counter', modules: true,
    scriptPath: path.resolve(__dirname, '../_workers/visits/dist/index.js'),
    compatibilityDate: '2026-09-16', bindings: { ALLOWED_ORIGINS: origin + ',' + localOrigin },
    durableObjects: { COUNTERS: { className: 'VisitCounter', useSQLite: true } },
    ratelimits: { VISIT_LIMITER: { namespace_id: '1001', simple: { limit: 120, period: 60 } } }
  }));
  let browser;
  try {
    browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
    const context = await browser.newContext();
    let unavailable = false;
    let blockMap = false;
    let emptyMap = false;
    let allowLocal = false;
    const visits = [];
    let mapRequests = 0;
    const errors = [];
    const handleRoute = async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.origin === endpoint || url.origin === localEndpoint) {
        if (request.method() === 'POST') visits.push(JSON.parse(request.postData()).path);
        if (unavailable) return route.fulfill({ status: 503, headers: { 'Access-Control-Allow-Origin': origin }, body: '{}' });
        const response = await mf.dispatchFetch(url.href, {
          method: request.method(), headers: await request.allHeaders(), body: request.postData() || undefined
        });
        return route.fulfill({ status: response.status, headers: Object.fromEntries(response.headers), body: await response.text() });
      }
      if (url.hostname === 'mapmyvisitors.com') {
        mapRequests++;
        if (blockMap) return route.abort();
        // Never register automated test traffic with the real map service.
        assert.equal(url.searchParams.get('t'), 'n', 'Hide the provider’s separate pageview total');
        return route.fulfill({ contentType: 'text/javascript', body: emptyMap ? '' : 'document.getElementById("mapmyvisitors").insertAdjacentHTML("afterend", "<div class=mapmyvisitors-map><svg width=300 height=150 aria-label=fixture></svg></div>");' });
      }
      if (url.origin === origin || url.origin === 'http://localhost:4000') {
        const filePath = decodeURIComponent(url.pathname) + (url.pathname.endsWith('/') ? 'index.html' : '');
        const file = path.join(site, filePath);
        const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.jpg': 'image/jpeg', '.png': 'image/png', '.woff2': 'font/woff2' };
        try {
          let body = await fs.readFile(file);
          if (allowLocal && url.origin === localOrigin && path.extname(file) === '.html') {
            body = body.toString().replace('data-allow-local="false"', 'data-allow-local="true"').replace(endpoint, localEndpoint);
          }
          return route.fulfill({ body, contentType: mime[path.extname(file)] || 'application/octet-stream' });
        } catch (_) { return route.fulfill({ status: 404, body: '' }); }
      }
      return route.abort();
    };
    await context.route('**/*', handleRoute);
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(origin + '/');
    await page.locator('[data-site-views-label]').waitFor({ state: 'visible' });
    assert.equal(await page.locator('[data-site-views]').textContent(), '1');
    assert.equal(await page.locator('[data-article-views]').count(), 0);
    assert.equal(await page.locator('#mapmyvisitors').count(), 1);
    assert.match(await page.locator('#mapmyvisitors').getAttribute('src'), /Q42jl5SLxXZzG4B3VEHOMQSr2BvKMMW1hcmNCsHdnFs/);
    await page.locator('[data-map-status]').waitFor({ state: 'hidden' });
    const initialId = await page.evaluate(() => localStorage.getItem('dianyo.visitor-id.v1'));
    assert.ok(initialId);

    await page.goto(origin + '/understanding-kallisto/?source=test#em-lab');
    await page.locator('[data-article-views]').waitFor({ state: 'visible' });
    assert.equal(await page.locator('[data-page-views]').textContent(), '1');
    assert.equal(await page.locator('[data-site-views]').textContent(), '1');
    assert.deepEqual(visits, ['/', '/understanding-kallisto/']);
    await page.reload();
    await page.locator('[data-article-views]').waitFor({ state: 'visible' });
    assert.equal(await page.locator('[data-page-views]').textContent(), '1');
    assert.equal(await page.locator('[data-site-views]').textContent(), '1');
    assert.equal(await page.evaluate(() => localStorage.getItem('dianyo.visitor-id.v1')), initialId);
    // Loading the script twice must not count or embed twice.
    await page.addScriptTag({ path: path.resolve(__dirname, '../assets/visitor-stats.js') });
    assert.equal(visits.length, 3);
    assert.equal(await page.locator('#mapmyvisitors').count(), 1);

    await page.setViewportSize({ width: 375, height: 812 });
    await page.locator('footer').scrollIntoViewIfNeeded();
    const map = await page.locator('[data-visitor-map]').boundingBox();
    assert.ok(map.x >= 0 && map.x + map.width <= 375, 'Map fits mobile viewport');
    await page.screenshot({ path: '/tmp/dianyo-visitors-mobile-test.png' });

    const beforeLocal = visits.length;
    const mapsBeforeLocal = mapRequests;
    await page.goto('http://localhost:4000/understanding-kallisto/');
    assert.equal(visits.length, beforeLocal, 'Local previews do not record visits');
    assert.equal(mapRequests, mapsBeforeLocal, 'Local previews do not load the map tracker');
    assert.equal(await page.locator('[data-article-views]').isHidden(), true);
    assert.equal(await page.locator('[data-visitor-map]').isHidden(), true);
    await page.goto(origin + '/404.html');
    assert.equal(visits.length, beforeLocal, '404s do not record visits');
    assert.equal(mapRequests, mapsBeforeLocal);

    allowLocal = true;
    await page.goto(localOrigin + '/understanding-kallisto/');
    await page.locator('[data-article-views]').waitFor({ state: 'visible' });
    await page.locator('[data-map-status]').waitFor({ state: 'hidden' });
    assert.equal(await page.locator('.mapmyvisitors-map svg').isVisible(), true, 'Opted-in localhost preview shows the map');
    assert.equal(await page.locator('[data-site-views]').textContent(), '2');
    await page.reload();
    await page.locator('[data-article-views]').waitFor({ state: 'visible' });
    assert.equal(await page.locator('[data-site-views]').textContent(), '2');

    const newBrowser = await browser.newContext();
    await newBrowser.route('**/*', handleRoute);
    const tabs = await Promise.all([newBrowser.newPage(), newBrowser.newPage()]);
    await Promise.all(tabs.map(async tab => {
      await tab.goto(origin + '/understanding-kallisto/');
      await tab.locator('[data-article-views]').waitFor({ state: 'visible' });
      assert.equal(await tab.locator('[data-site-views]').textContent(), '3');
      assert.equal(await tab.locator('[data-page-views]').textContent(), '3');
    }));
    assert.deepEqual(await Promise.all(tabs.map(tab => tab.evaluate(() => localStorage.getItem('dianyo.visitor-id.v1')))), [
      await tabs[0].evaluate(() => localStorage.getItem('dianyo.visitor-id.v1')),
      await tabs[0].evaluate(() => localStorage.getItem('dianyo.visitor-id.v1'))
    ], 'Simultaneous first tabs share one browser ID');
    await newBrowser.close();

    const noStorage = await browser.newContext();
    await noStorage.route('**/*', handleRoute);
    await noStorage.addInitScript(() => {
      Storage.prototype.getItem = Storage.prototype.setItem = () => { throw new Error('Storage blocked'); };
    });
    const blocked = await noStorage.newPage();
    const beforeBlocked = visits.length;
    await blocked.goto(origin + '/understanding-kallisto/');
    await blocked.locator('[data-article-views]').waitFor({ state: 'visible' });
    await blocked.reload();
    await blocked.locator('[data-article-views]').waitFor({ state: 'visible' });
    assert.equal(await blocked.locator('[data-site-views]').textContent(), '3');
    assert.equal(visits.length, beforeBlocked, 'Blocked storage reads totals without recording refreshes');
    await noStorage.close();

    unavailable = true;
    blockMap = true;
    await page.goto(origin + '/understanding-kallisto/');
    await page.waitForLoadState('networkidle');
    assert.equal(await page.locator('[data-article-views]').isHidden(), true);
    assert.equal(await page.locator('[data-site-views-label]').isHidden(), true);
    assert.equal(await page.locator('[data-map-embed]').isHidden(), true);
    assert.equal(await page.locator('[data-map-status]').textContent(), 'Visitor map is unavailable right now.');
    assert.equal(await page.locator('article h1').isVisible(), true);

    blockMap = false;
    emptyMap = true;
    await page.goto(origin + '/');
    await page.waitForFunction(() => document.querySelector('[data-map-status]').textContent.includes('unavailable'), null, { timeout: 15000 });
    assert.equal(await page.locator('[data-map-embed]').isHidden(), true, 'A script that loads but never renders gets a useful fallback');
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: daily counts, simultaneous tabs, blocked storage, localhost map, mobile layout, canonical paths, preview isolation, 404s, and service failures.');
  } finally {
    if (browser) await browser.close();
    await mf.dispose();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
