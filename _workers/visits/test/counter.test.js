import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { randomUUID } from 'node:crypto';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

const origin = 'https://dianyo.github.io';
const visitorId = randomUUID();
const options = {
  name: 'dianyo-visitor-counts',
  modules: true,
  scriptPath: new URL('../dist/index.js', import.meta.url).pathname,
  compatibilityDate: '2026-09-16',
  bindings: { ALLOWED_ORIGINS: origin },
  durableObjects: { COUNTERS: { className: 'VisitCounter', useSQLite: true } },
  ratelimits: { VISIT_LIMITER: { namespace_id: '1001', simple: { limit: 120, period: 60 } } }
};

function runtime(config = options) {
  return new Miniflare(convertV4MiniflareOptions(config));
}

function visit(mf, path = '/first/', overrides = {}) {
  return mf.dispatchFetch('https://counter.test/visit', {
    method: 'POST',
    headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, visitorId }),
    ...overrides
  });
}

async function read(mf, path) {
  const response = await mf.dispatchFetch('https://counter.test/counts?path=' + encodeURIComponent(path), {
    headers: { Origin: origin }
  });
  assert.equal(response.status, 200);
  return response.json();
}

test('refreshes and concurrent tabs count once; different browsers and articles count independently', async t => {
  const mf = runtime();
  t.after(() => mf.dispose());
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 0, pageViews: 0 });
  const first = await visit(mf);
  assert.equal(first.status, 200);
  assert.equal(first.headers.get('cache-control'), 'no-store');
  assert.equal(first.headers.get('access-control-allow-origin'), origin);
  assert.deepEqual(await first.json(), { siteViews: 1, pageViews: 1 });
  assert.deepEqual(await (await visit(mf, '/second/')).json(), { siteViews: 1, pageViews: 1 });

  const responses = await Promise.all(Array.from({ length: 40 }, () => visit(mf)));
  const counts = await Promise.all(responses.map(async response => {
    assert.equal(response.status, 200);
    return response.json();
  }));
  assert.ok(counts.every(count => count.siteViews === 1 && count.pageViews === 1), 'Same browser is deduplicated atomically');
  const unique = await Promise.all(Array.from({ length: 40 }, () => visit(mf, '/first/', {
    body: JSON.stringify({ path: '/first/', visitorId: randomUUID() })
  })));
  const uniqueCounts = await Promise.all(unique.map(response => response.json()));
  assert.equal(new Set(uniqueCounts.map(count => count.siteViews)).size, 40, 'No lost increments for distinct browsers');
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 41, pageViews: 41 });
  assert.deepEqual(await read(mf, '/second/'), { siteViews: 41, pageViews: 1 });
  assert.deepEqual(await read(mf, '/unvisited/'), { siteViews: 41, pageViews: 0 });
});

test('eligibility expires after 24 hours independently for site and article, without sliding on refresh', async t => {
  const mf = runtime({ ...options, unsafeInspectDurableObjects: true });
  t.after(() => mf.dispose());
  const started = Date.now();
  assert.equal((await visit(mf)).status, 200);
  const storage = await mf.unsafeGetDurableObjectStorage(options.name, 'VisitCounter', { name: 'dianyo.github.io' });
  const initial = await storage.exec('SELECT * FROM recent_visits ORDER BY scope');
  assert.equal(initial.length, 2);
  for (const row of initial) {
    assert.match(row.visitor, /^[0-9a-f]{64}$/, 'Only the digest is retained');
    assert.ok(row.expires_at >= started + 86400000 && row.expires_at <= Date.now() + 86400000);
  }
  assert.deepEqual(await (await visit(mf)).json(), { siteViews: 1, pageViews: 1 });
  assert.deepEqual(await storage.exec('SELECT * FROM recent_visits ORDER BY scope'), initial, 'Refresh does not extend the window');

  // Move only the local test records past their boundary, without exposing a
  // clock override or reset route in the public API.
  await storage.exec('UPDATE recent_visits SET expires_at = ? WHERE scope = ?', Date.now() - 1, 'site');
  assert.deepEqual(await (await visit(mf)).json(), { siteViews: 2, pageViews: 1 });
  await storage.exec('UPDATE recent_visits SET expires_at = ? WHERE scope = ?', Date.now() - 1, 'page:/first/');
  assert.deepEqual(await (await visit(mf)).json(), { siteViews: 2, pageViews: 2 });

  await storage.exec('INSERT INTO recent_visits VALUES (?, ?, ?)', 'expired-browser', 'page:/old/', Date.now() - 1);
  assert.deepEqual(await (await visit(mf)).json(), { siteViews: 2, pageViews: 2 });
  assert.equal((await storage.exec('SELECT * FROM recent_visits')).length, 2, 'Expired tracking records are pruned');
});

test('origin, method, payload, and path validation do not change totals', async t => {
  const mf = runtime();
  t.after(() => mf.dispose());
  const preflight = await mf.dispatchFetch('https://counter.test/visit', {
    method: 'OPTIONS', headers: { Origin: origin }
  });
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('access-control-allow-origin'), origin);
  assert.equal((await visit(mf, '/first/', { headers: { Origin: 'https://other.test' } })).status, 403);
  assert.equal((await visit(mf, '/first/', { headers: {} })).status, 403);
  assert.equal((await visit(mf, '/first/', { method: 'GET', body: undefined })).status, 405);
  assert.equal((await visit(mf, '/first/', { headers: { Origin: origin, 'Content-Type': 'text/plain' } })).status, 415);
  assert.equal((await visit(mf, '/first/', { body: '{' })).status, 400);
  assert.equal((await visit(mf, '/first/', { body: 'null' })).status, 400);
  assert.equal((await visit(mf, '/first/', { body: ' '.repeat(2049) })).status, 413);
  for (const id of [null, '', 'not-a-uuid', 42]) {
    assert.equal((await visit(mf, '/first/', { body: JSON.stringify({ path: '/first/', visitorId: id }) })).status, 400);
  }
  assert.deepEqual(await (await visit(mf, '/first/', { body: JSON.stringify({ path: '/first/' }) })).json(), { siteViews: 0, pageViews: 0 }, 'Old clients are read-only');
  for (const path of [null, 1, 'https://other.test/', '//other.test/', '/x?y=1', '/x#part', '/a/../b', '/a/%2e%2e/b', '/has space/', '/a\\b', '/' + 'a'.repeat(512)]) {
    assert.equal((await visit(mf, path)).status, 400, String(path));
  }
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 0, pageViews: 0 });
});

test('the rate limiter rejects a burst without incrementing the persistent counter', async t => {
  const mf = runtime({ ...options,
    ratelimits: { VISIT_LIMITER: { namespace_id: '1001', simple: { limit: 2, period: 60 } } }
  });
  t.after(() => mf.dispose());
  assert.equal((await visit(mf)).status, 200);
  assert.equal((await visit(mf)).status, 200);
  const limited = await visit(mf);
  assert.equal(limited.status, 429);
  assert.equal(limited.headers.get('retry-after'), '60');
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 1, pageViews: 1 });
});

test('counts survive a runtime restart with the same persistent storage', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'visitor-counts-'));
  let mf;
  try {
    const persistent = { ...options, resourcePersistencePath: directory };
    mf = runtime(persistent);
    assert.equal((await visit(mf)).status, 200);
    await mf.dispose();
    mf = runtime(persistent);
    assert.deepEqual(await read(mf, '/first/'), { siteViews: 1, pageViews: 1 });
    assert.deepEqual(await (await visit(mf)).json(), { siteViews: 1, pageViews: 1 }, 'A restart preserves daily deduplication too');
  } finally {
    if (mf) await mf.dispose();
    await rm(directory, { recursive: true, force: true });
  }
});
