# Site views, article views, and visitor map

The blog stays on Jekyll / GitHub Pages. A Cloudflare Worker records page views
in one SQLite-backed Durable Object. MapMyVisitors supplies the footer map using
the public widget URL in `_config.yml`.

Deployed counter endpoint:
<https://dianyo-visitor-counts.dianyo-visitor-counts.workers.dev>

## What gets counted

- One successful `POST /visit` per document load, including reloads. These are
  page views, not unique people or homepage-link clicks.
- Every tracked page contributes to the site total. Each article has its own
  total keyed by Jekyll's canonical path, including `baseurl`. Query strings and
  section anchors do not create additional article counters.
- No historical visits are imported. Renaming an article's permalink creates a
  new counter for it.
- The response includes the just-saved site and page totals with `Cache-Control:
  no-store`. An already-open page does not poll for other readers' later visits.
- No JavaScript, blocked requests, or a failed service means no recorded view and
  no displayed count. A failed counter does not interfere with the map or article.
- The map tracks independently through MapMyVisitors. Its totals and update timing
  can differ from Cloudflare's. Only the Cloudflare numeric counters bypass an
  analytics provider's display cache.
- Localhost, other preview origins, and the 404 page do not contact production
  trackers. The optional local override only allows a loopback counter endpoint;
  the MapMyVisitors script stays disabled locally.

## Cloudflare setup and deployment

Requires Node.js 22 or newer and a Cloudflare account with Workers access.

```sh
cd _workers/visits
npm ci
npm exec wrangler login
npm test
npm run deploy
```

Copy the deployed HTTPS Worker URL (without `/visit`) to
`visitor_stats.endpoint` in `_config.yml`, then publish the Jekyll changes through
the normal GitHub Pages workflow. The Worker and the site deploy separately.
No Cloudflare token belongs in `_config.yml`, browser JavaScript, or this repository.

Production accepts only `https://dianyo.github.io`, configured in both
`visitor_stats.origin` and `wrangler.jsonc`'s `ALLOWED_ORIGINS`. Update both if the
site moves to a custom domain. The public counter API is:

```text
POST /visit                  JSON body: {"path":"/article/"}
GET  /counts?path=/article/   Read without recording another view
                             Response: {"siteViews":123,"pageViews":45}
```

Both routes require an allowed `Origin` header. Paths must be canonical absolute
paths, not full URLs. The endpoint rejects malformed bodies and limits their size.
There is no public reset or delete endpoint.

The Worker applies a generous limit of 120 writes per minute per IP at each
Cloudflare location. Shared networks share that allowance. This and the origin
check reduce casual misuse; they cannot prove a request came from a human, and
non-browser clients can forge origins. Do not treat the counters as audited
analytics. The app stores only totals and paths, not IP addresses, cookies,
visitor IDs, or location histories. Cloudflare uses the IP transiently as the
rate-limit key; MapMyVisitors handles its own location data.

`ratelimits.namespace_id` must be unused by other rate-limit bindings in this
Cloudflare account unless sharing limits is intentional. This project uses `1001`
for production and `1002` for the local environment.

### Preserving data through CI/CD

The counts live outside the generated `_site` files. Ordinary Jekyll rebuilds and
Worker deployments do not reset them. Preserve all of these identifiers:

- Worker name: `dianyo-visitor-counts`
- Durable Object class and binding: `VisitCounter` / `COUNTERS`
- Instance name in source: `dianyo.github.io`
- Existing migration history in `wrangler.jsonc`

Do not delete the Worker/namespace or introduce a new instance name to redeploy.
The initial `v1` migration provisions storage; it does not clear it each time.

For unattended CI, supply `CLOUDFLARE_API_TOKEN` as a GitHub Actions secret and
`CLOUDFLARE_ACCOUNT_ID` through the workflow environment, then run `npm ci`,
`npm test`, and `npm run deploy` from this directory. Restrict deployment to the
intended production branch; do not expose credentials to untrusted PR jobs.

## MapMyVisitors

The supplied public embed is already configured. Its script is loaded
asynchronously into the footer with the required `id="mapmyvisitors"`.
To replace it, create a widget for the site's public URL at
<https://mapmyvisitors.com/add>, then copy the full script `src` into
`visitor_stats.map_script_url`, using `https://` instead of `//`.
The integration accepts the provider's `https://mapmyvisitors.com/map.js?...`
format. An empty URL disables the map. A blank counter endpoint disables counters.

## Local preview

Run the isolated local Worker:

```sh
cd _workers/visits
npm run dev
```

In a second terminal, from the repository root:

```sh
cat > /tmp/visitor-local.yml <<'EOF'
visitor_stats:
  endpoint: "http://127.0.0.1:8787"
  allow_local: true
EOF
jekyll serve --config _config.yml,/tmp/visitor-local.yml --host 127.0.0.1
```

Wrangler stores local counts under `.wrangler`, which is ignored by Git. Local
storage and the production Durable Object are separate. Do not use `--remote`
for this preview.

## Verification

`npm test` builds the Worker and exercises the actual local Cloudflare runtime:
independent page counts, concurrent increments, immediate reads, origin and payload
validation, rate limiting, and persistence across a runtime restart.

The browser check uses that same runtime and intercepts MapMyVisitors requests
with a fixture so automated tests never add map visits. From the repository root:

```sh
cat > /tmp/dianyo-visitors-test.yml <<'EOF'
visitor_stats:
  endpoint: "https://counter.test"
EOF
jekyll build --config _config.yml,/tmp/dianyo-visitors-test.yml --destination /tmp/dianyo-visitors-test
node _tests/visitor-stats-browser.cjs
```

Google Chrome is used by default. To use Playwright's bundled Chromium instead:

```sh
cd _workers/visits
npm exec playwright install chromium
cd ../..
PLAYWRIGHT_CHANNEL=chromium node _tests/visitor-stats-browser.cjs
```

The browser checks verify article and site counters, canonical paths, duplicate
script protection, mobile layout, preview isolation, 404 exclusion, and graceful
failure when services are blocked. `VISITOR_SITE_DIR` overrides the build directory.
All worker sources, dependencies, and tests remain outside Jekyll output because
their directories begin with underscores.

References:
- <https://developers.cloudflare.com/durable-objects/examples/build-a-counter/>
- <https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/>
- <https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/>
- <https://mapmyvisitors.com/>
