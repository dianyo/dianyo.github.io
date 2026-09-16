import { DurableObject } from 'cloudflare:workers';

const DAY_MS = 24 * 60 * 60 * 1000;
const VISITOR_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

// The browser sends Jekyll's canonical page.url, never a query string or hash.
function validPath(path) {
  if (typeof path !== 'string' || path.length > 512 || !path.startsWith('/') ||
      /[\s\\?#\u0000-\u001f\u007f]/.test(path) || path.includes('//')) return false;
  try {
    return new URL(path, 'https://site.invalid').pathname === path;
  } catch (_) { return false; }
}

function reply(data, status, origin, extra = {}) {
  return new Response(data === null ? null : JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
      'Vary': 'Origin',
      ...(origin ? { 'Access-Control-Allow-Origin': origin } : {}),
      ...extra
    }
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== '/visit' && url.pathname !== '/counts') {
      return reply({ error: 'Not found' }, 404);
    }

    const origin = request.headers.get('Origin');
    const allowed = (env.ALLOWED_ORIGINS || '').split(',').map(value => value.trim());
    if (!origin || !allowed.includes(origin)) return reply({ error: 'Origin not allowed' }, 403);

    if (request.method === 'OPTIONS') {
      return reply(null, 204, origin, {
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '86400'
      });
    }

    const record = url.pathname === '/visit';
    const method = record ? 'POST' : 'GET';
    if (request.method !== method) return reply({ error: 'Method not allowed' }, 405, origin, { Allow: method });

    let path;
    let visitorId;
    if (record) {
      if (request.headers.get('Content-Type')?.split(';')[0].trim() !== 'application/json') {
        return reply({ error: 'Expected application/json' }, 415, origin);
      }
      // Bound request memory even when Content-Length is absent or incorrect.
      const reader = request.body?.getReader();
      if (!reader) return reply({ error: 'Missing body' }, 400, origin);
      let size = 0;
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > 2048) {
          await reader.cancel();
          return reply({ error: 'Body too large' }, 413, origin);
        }
        chunks.push(value);
      }
      try {
        const data = JSON.parse(await new Blob(chunks).text());
        path = data?.path;
        visitorId = data?.visitorId;
      } catch (_) { return reply({ error: 'Invalid JSON' }, 400, origin); }
    } else {
      path = url.searchParams.get('path');
    }
    if (!validPath(path)) return reply({ error: 'Invalid canonical path' }, 400, origin);
    if (visitorId !== undefined && (typeof visitorId !== 'string' || !VISITOR_ID.test(visitorId))) {
      return reply({ error: 'Invalid visitor ID' }, 400, origin);
    }

    if (record) {
      // Anonymous blog: a generous IP limit bounds obvious bursts. Shared networks
      // share this allowance; it is abuse reduction, not visitor identification.
      const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
      const { success } = await env.VISIT_LIMITER.limit({ key: 'visits:' + ip });
      if (!success) return reply({ error: 'Too many requests' }, 429, origin, { 'Retry-After': '60' });
    }

    try {
      // Keep this stable across deployments: changing the name starts new totals.
      const counter = env.COUNTERS.getByName('dianyo.github.io');
      let counts;
      if (record && visitorId) {
        // Retain only a digest with an expiry, never the raw browser identifier.
        const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(visitorId));
        const visitor = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
        counts = await counter.record(path, visitor);
      } else {
        // Older cached scripts without an ID remain read-only after deployment.
        counts = await counter.read(path);
      }
      return reply(counts, 200, origin);
    } catch (_) {
      return reply({ error: 'Counter unavailable' }, 503, origin);
    }
  }
};

export class VisitCounter extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    ctx.storage.sql.exec(`
      CREATE TABLE IF NOT EXISTS recent_visits (
        visitor TEXT NOT NULL,
        scope TEXT NOT NULL,
        expires_at INTEGER NOT NULL,
        PRIMARY KEY (visitor, scope)
      );
      CREATE INDEX IF NOT EXISTS recent_visits_expiry ON recent_visits(expires_at);
    `);
  }

  async read(path) {
    const values = await this.ctx.storage.get(['site', 'page:' + path]);
    return { siteViews: values.get('site') || 0, pageViews: values.get('page:' + path) || 0 };
  }

  async record(path, visitor) {
    // Keep the existing counter keys and totals. The new expiring records only
    // control whether a browser is eligible to increment each counter again.
    return this.ctx.storage.transaction(async storage => {
      const now = Date.now();
      const sql = this.ctx.storage.sql;
      const pageKey = 'page:' + path;
      const values = await storage.get(['site', pageKey]);
      sql.exec('DELETE FROM recent_visits WHERE expires_at <= ?', now);
      const eligible = scope => sql.exec(
        'INSERT INTO recent_visits (visitor, scope, expires_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING RETURNING expires_at',
        visitor, scope, now + DAY_MS
      ).toArray().length;
      const siteViews = (values.get('site') || 0) + eligible('site');
      const pageViews = (values.get(pageKey) || 0) + eligible(pageKey);
      await storage.put({ site: siteViews, [pageKey]: pageViews });
      await this.scheduleCleanup();
      return { siteViews, pageViews };
    });
  }

  async scheduleCleanup() {
    const next = this.ctx.storage.sql.exec('SELECT MIN(expires_at) AS expiry FROM recent_visits').one().expiry;
    if (next !== null && await this.ctx.storage.getAlarm() !== next) {
      await this.ctx.storage.setAlarm(next);
    }
  }

  async alarm() {
    this.ctx.storage.sql.exec('DELETE FROM recent_visits WHERE expires_at <= ?', Date.now());
    await this.scheduleCleanup();
  }
}
