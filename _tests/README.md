# Kallisto article checks

The post uses the existing Jekyll post layout. Its JavaScript and CSS load only on
`/understanding-kallisto/`; no global layout, stylesheet, or configuration changes
are needed. The article remains readable without JavaScript. This is a teaching
model, with deliberate simplifications documented in the post.

Run the dependency-free numerical tests with Node.js:

```sh
node --test _tests/kallisto-model.test.cjs
```

Build and serve with the repository's existing Jekyll environment:

```sh
jekyll build --destination /tmp/kallisto-site
python3 -m http.server 4000 --bind 127.0.0.1 --directory /tmp/kallisto-site
```

In another terminal, run the browser checks with Playwright resolvable by Node.js
and Google Chrome installed:

```sh
node _tests/kallisto-browser.cjs
```

If Playwright is installed outside this repository, set `NODE_PATH` to its parent
`node_modules` directory. `PLAYWRIGHT_CHANNEL` can select another installed browser
channel. `KALLISTO_PREVIEW_URL` overrides `http://127.0.0.1:4000`. Screenshots are
written to the operating system's temporary directory.

The checks cover independent hand calculations, count conservation, likelihood
factorization and monotonicity, nonidentifiability, bootstrap reproducibility,
invalid and empty input, the full read-to-EM workflow, mobile overflow, keyboard
operation, and isolation from the homepage. `_tests` is excluded from Jekyll output
by Jekyll's normal underscore-directory handling.
