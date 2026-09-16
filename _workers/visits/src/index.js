import { DurableObject } from 'cloudflare:workers';

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
      } catch (_) { return reply({ error: 'Invalid JSON' }, 400, origin); }
    } else {
      path = url.searchParams.get('path');
    }
    if (!validPath(path)) return reply({ error: 'Invalid canonical path' }, 400, origin);

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
      const counts = record ? await counter.record(path) : await counter.read(path);
      return reply(counts, 200, origin);
    } catch (_) {
      return reply({ error: 'Counter unavailable' }, 503, origin);
    }
  }
};

export class VisitCounter extends DurableObject {
  async read(path) {
    const values = await this.ctx.storage.get(['site', 'page:' + path]);
    return { siteViews: values.get('site') || 0, pageViews: values.get('page:' + path) || 0 };
  }

  async record(path) {
    // One transaction keeps site and article totals together under concurrency.
    return this.ctx.storage.transaction(async storage => {
      const pageKey = 'page:' + path;
      const values = await storage.get(['site', pageKey]);
      const siteViews = (values.get('site') || 0) + 1;
      const pageViews = (values.get(pageKey) || 0) + 1;
      await storage.put({ site: siteViews, [pageKey]: pageViews });
      return { siteViews, pageViews };
    });
  }
}
