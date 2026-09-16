import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

const origin = 'https://dianyo.github.io';
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
    body: JSON.stringify({ path }),
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

test('site and independent article counts are immediately readable under concurrent visits', async t => {
  const mf = runtime();
  t.after(() => mf.dispose());
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 0, pageViews: 0 });
  const first = await visit(mf);
  assert.equal(first.status, 200);
  assert.equal(first.headers.get('cache-control'), 'no-store');
  assert.equal(first.headers.get('access-control-allow-origin'), origin);
  assert.deepEqual(await first.json(), { siteViews: 1, pageViews: 1 });
  assert.deepEqual(await (await visit(mf, '/second/')).json(), { siteViews: 2, pageViews: 1 });

  const responses = await Promise.all(Array.from({ length: 40 }, () => visit(mf)));
  const counts = await Promise.all(responses.map(async response => {
    assert.equal(response.status, 200);
    return response.json();
  }));
  assert.equal(new Set(counts.map(count => count.siteViews)).size, 40, 'No lost increments');
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 42, pageViews: 41 });
  assert.deepEqual(await read(mf, '/second/'), { siteViews: 42, pageViews: 1 });
  assert.deepEqual(await read(mf, '/unvisited/'), { siteViews: 42, pageViews: 0 });
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
  assert.deepEqual(await read(mf, '/first/'), { siteViews: 2, pageViews: 2 });
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
    assert.deepEqual(await (await visit(mf)).json(), { siteViews: 2, pageViews: 2 });
  } finally {
    if (mf) await mf.dispose();
    await rm(directory, { recursive: true, force: true });
  }
});
