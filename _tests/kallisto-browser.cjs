// Run against a Jekyll build served at KALLISTO_PREVIEW_URL (default localhost:4000).
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const path = require('node:path');
const os = require('node:os');
const base = process.env.KALLISTO_PREVIEW_URL || 'http://127.0.0.1:4000';
const errors = [];

(async () => {
  const browser = await chromium.launch({headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome'});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => {
      if (response.url().startsWith(base + '/assets/kallisto/') && response.status() >= 400) errors.push(response.url() + ': ' + response.status());
    });
    await page.goto(base + '/understanding-kallisto/');
    await page.waitForFunction(() => window.katex && document.querySelector('[data-math-rendered]'));
    assert.equal(await page.locator('[data-math-error]').count(), 0, 'No LaTeX parse errors');
    assert.equal(await page.locator('.k-equation[data-math-rendered]').count(), 9, 'All display equations render');
    assert.equal(await page.locator('.k-math:not([data-math-rendered])').count(), 0, 'All inline equations render');
    assert.ok(await page.locator('.katex-mathml').count() > 40, 'Equations include accessible MathML');
    await page.evaluate(() => document.fonts.ready);
    await page.locator('.k-equation').nth(2).screenshot({path: path.join(os.tmpdir(), 'kallisto-likelihood-latex.png')});
    await page.locator('.k-equation').nth(3).screenshot({path: path.join(os.tmpdir(), 'kallisto-em-latex.png')});
    assert.equal(await page.locator('.k-paper-figure img').count(), 3, 'Original paper figures are embedded');
    for (const figure of await page.locator('.k-paper-figure').all()) {
      await figure.scrollIntoViewIfNeeded();
      await figure.locator('img').evaluate(img => img.decode());
      assert.equal(await figure.locator('img').evaluate(img => img.naturalWidth === Number(img.getAttribute('width'))), true, 'Original image loads at its recorded resolution');
      assert.equal(await figure.locator('a').first().getAttribute('href'), await figure.locator('img').getAttribute('src'), 'Image opens at full size');
    }
    await page.locator('#gpu-runtime').screenshot({path: path.join(os.tmpdir(), 'kallisto-gpu-runtime.png')});
    const content = id => page.locator('#' + id).innerText();
    assert.match(await content('k-read-result'), /\{T1, T3\}/);
    assert.equal(await page.locator('#k-kmers button').count(), 5);
    await page.locator('#k-kmers button').nth(2).click();
    assert.match(await content('k-intersection'), /\{T1, T3\}/);
    await page.locator('#sequence-lab').screenshot({path: path.join(os.tmpdir(), 'kallisto-sequence.png')});
    await page.getByRole('button', {name: 'One substitution', exact: true}).click();
    assert.match(await content('k-read-result'), /\{T1\}/);
    await page.locator('#k-size').fill('7');
    assert.match(await content('k-read-result'), /Unassigned/);
    await page.locator('#k-size').fill('3');
    await page.getByRole('button', {name: 'No matches', exact: true}).click();
    await page.locator('#k-add-read').click();
    assert.match(await content('k-sample-totals'), /260 assigned \+ 10 unassigned/);
    await page.locator('#k-read').fill('AC');
    assert.match(await content('k-read-result'), /less than k/);
    await page.locator('#k-read').fill('<script>');
    assert.equal(await page.locator('#k-read').getAttribute('aria-invalid'), 'true');
    assert.equal(await page.locator('#k-add-read').isDisabled(), true);
    await page.locator('#k-read').fill('act gacg');
    assert.match(await content('k-read-result'), /\{T1\}/);
    await page.locator('#k-use-sample').click();
    assert.match(await content('k-em-status'), /Loaded 260/);
    assert.equal(await page.locator('#k-em-preset').inputValue(), 'custom');
    await page.locator('#k-em-run').click();
    assert.match(await content('k-em-status'), /Stopped/);

    await page.locator('#k-em-preset').selectOption('worked');
    await page.locator('#k-em-step').click();
    assert.match(await content('k-e-totals'), /145/);
    assert.match(await content('k-e-totals'), /55/);
    await page.locator('#k-em-step').click();
    assert.match(await content('k-em-estimates'), /0.72500/);
    await page.locator('#k-em-step').click();
    await page.locator('#k-em-step').click();
    assert.match(await content('k-em-estimates'), /0.82625/);
    await page.locator('#em-lab').screenshot({path: path.join(os.tmpdir(), 'kallisto-em.png')});
    await page.locator('#k-em-run').click();
    assert.match(await content('k-em-estimates'), /0.90909/);
    await page.locator('#k-bootstrap').click();
    await page.waitForFunction(() => document.querySelector('#k-bootstrap-status').textContent.startsWith('30 resamples'));
    assert.equal(await page.locator('#k-bootstrap-output circle').count(), 90);
    await page.locator('#bootstrap-lab').screenshot({path: path.join(os.tmpdir(), 'kallisto-bootstrap.png')});

    await page.locator('#k-em-preset').selectOption('slow');
    await page.locator('#k-em-run').click();
    assert.match(await content('k-em-status'), /not converged/);
    await page.locator('#k-em-preset').selectOption('ambiguous');
    await page.locator('#k-em-run').click();
    assert.match(await content('k-em-estimates'), /0.50000/);
    const equalLL = await content('k-em-likelihood');
    await page.locator('#k-em-favor').click();
    await page.locator('#k-em-run').click();
    assert.match(await content('k-em-estimates'), /0.88889/);
    assert.equal((await content('k-em-likelihood')).split('. Change')[0], equalLL.split('. Change')[0]);
    await page.locator('#em-lab details summary').click();
    await page.locator('[data-length="0"]').fill('0');
    await page.locator('[data-length="0"]').press('Tab');
    assert.equal(await page.locator('[data-length="0"]').inputValue(), '100');
    for (const input of await page.locator('[data-ec]').all()) { await input.fill('0'); await input.press('Tab'); }
    assert.match(await content('k-em-status'), /No observations/);
    assert.equal(await page.locator('#k-em-run').isDisabled(), true);
    assert.doesNotMatch(await content('k-em-estimates'), /NaN|Infinity/);
    await page.locator('#k-clear-reads').click();
    assert.match(await content('k-sample-totals'), /0 reads/);
    assert.equal(await page.locator('#k-use-sample').isDisabled(), true);

    await page.reload();
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({path: path.join(os.tmpdir(), 'kallisto-desktop.png')});
    assert.equal(await page.locator('h1').count(), 2); // Existing site title + post title.
    const brokenAnchors = await page.locator('a[href^="#"]').evaluateAll(links => links.map(a => a.getAttribute('href')).filter(href => !document.getElementById(href.slice(1))));
    assert.deepEqual(brokenAnchors, []);
    const duplicateIDs = await page.locator('[id]').evaluateAll(nodes => nodes.map(n => n.id).filter((id, i, ids) => ids.indexOf(id) !== i));
    assert.deepEqual(duplicateIDs, []);
    await page.setViewportSize({width: 375, height: 812});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'No horizontal page overflow on mobile');
    await page.locator('#gpu-benchmark').screenshot({path: path.join(os.tmpdir(), 'kallisto-mobile-benchmark.png')});
    await page.locator('.k-equation').nth(3).screenshot({path: path.join(os.tmpdir(), 'kallisto-mobile-latex.png')});
    await page.locator('.k-equation').nth(0).focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('.k-equation').scrollLeft > 0);
    await page.locator('#sequence-lab').screenshot({path: path.join(os.tmpdir(), 'kallisto-mobile-sequence.png')});
    await page.locator('#k-next').focus();
    await page.keyboard.press('Enter');
    assert.match(await content('k-step-label'), /Window 2/);
    await page.locator('#k-em-step').click();
    await page.locator('#em-lab').screenshot({path: path.join(os.tmpdir(), 'kallisto-mobile-em.png')});
    await page.goto(base + '/');
    assert.equal(await page.locator('a.read-more[href="/understanding-kallisto/"]').count(), 1);
    assert.equal(await page.locator('.k-lab').count(), 0, 'Homepage excerpt must not include widgets');
    assert.equal(await page.locator('script[src*="kallisto"]').count(), 0, 'Scripts only load on the article');
    assert.equal(await page.locator('link[href*="katex"]').count(), 0, 'Math assets only load on the article');
    const noJS = await browser.newPage({javaScriptEnabled: false});
    await noJS.goto(base + '/understanding-kallisto/');
    assert.match(await noJS.locator('noscript').innerText(), /JavaScript is disabled/);
    assert.match(await noJS.locator('body').innerText(), /0.82625/);
    assert.deepEqual(errors, [], 'No browser runtime errors');
    console.log('Browser checks passed: LaTeX rendering and MathML, local math assets, original paper images, all four labs, custom sample transfer, exact EM steps, degeneracy, slow convergence, invalid/empty states, keyboard operation, mobile overflow, no-JS content, and homepage isolation.');
    console.log('Screenshots saved to ' + os.tmpdir());
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
