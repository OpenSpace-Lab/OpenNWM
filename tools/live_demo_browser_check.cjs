const { chromium } = require('/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/browser/node_modules/playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = '/file_system/nas/algorithm/dujun.nie/opennwm_live_demo_20261029';
const report = { startedAt: new Date().toISOString(), url: process.argv[2] || 'http://127.0.0.1:8766/', checks: [], errors: [], failedRequests: [], apiResponses: [] };
const check = (name, condition) => { assert.ok(condition, name); report.checks.push(name); };

(async () => {
  const proxyUrl = process.env.HTTPS_PROXY || process.env.HTTP_PROXY;
  let proxy;
  if (proxyUrl) {
    const parsed = new URL(proxyUrl);
    proxy = { server: parsed.origin, bypass: '127.0.0.1,localhost' };
    if (parsed.username) proxy.username = decodeURIComponent(parsed.username);
    if (parsed.password) proxy.password = decodeURIComponent(parsed.password);
  }
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'], proxy });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, ignoreHTTPSErrors: true });
  page.on('pageerror', error => report.errors.push(error.message));
  page.on('requestfailed', request => report.failedRequests.push({ url: request.url(), error: request.failure()?.errorText }));
  page.on('response', async response => {
    if (!response.url().includes('/api/')) return;
    report.apiResponses.push({ url: response.url(), status: response.status() });
    if (response.url().endsWith('/api/predict') && response.request().method() === 'POST') {
      report.submitted = response.request().postDataJSON();
      report.admitted = await response.json();
      console.log(JSON.stringify({ phase: 'submitted', ...report.admitted, points: report.submitted.points.length }));
    }
  });
  try {
    await page.goto(report.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.locator('#playground-scenes [data-scene]').first().waitFor({ timeout: 60000 });
    await page.locator('#playground').scrollIntoViewIfNeeded();
    await page.waitForFunction(() => document.querySelector('#service-status').classList.contains('connected'), null, { timeout: 60000 });
    check('public service connected in browser', true);
    await page.locator('[data-scene="garden-recon"]').click();
    const canvas = page.locator('#trajectory-canvas');
    await canvas.scrollIntoViewIfNeeded();
    const box = await canvas.boundingBox();
    const route = [[0, 0], [-.015, .12], [-.07, .32], [-.18, .55], [-.28, .78], [-.23, 1.02], [-.08, 1.18]];
    const screen = ([x, y]) => [box.x + (270 + x * 222) / 540 * box.width, box.y + (385 - y * 220) / 470 * box.height];
    await page.mouse.move(...screen(route[0]));
    await page.mouse.down();
    for (const point of route.slice(1)) await page.mouse.move(...screen(point), { steps: 7 });
    await page.mouse.up();
    check('drawn path clears active presets', await page.locator('[data-path].active').count() === 0);
    check('generation is enabled for drawing', await page.locator('#generate-path').isEnabled());
    const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/predict') && response.request().method() === 'POST', { timeout: 30000 });
    await page.locator('#generate-path').click();
    const admission = await responsePromise;
    check('real public POST accepted', admission.status() === 202 && admission.url().startsWith('https://'));
    const accepted = await admission.json();
    report.admitted = accepted;
    report.submitted = admission.request().postDataJSON();
    check('mouse drawing contains nonpreset points', report.submitted.points.length > 20);
    const deadline = Date.now() + 240000;
    while (Date.now() < deadline) {
      const mode = await page.locator('#prediction-mode').textContent();
      const status = await page.locator('#playground-status').textContent();
      if (mode === 'LIVE MODEL OUTPUT') break;
      if (status.includes('could not finish')) throw new Error(status);
      await page.waitForTimeout(1000);
    }
    check('completed output is explicitly live', await page.locator('#prediction-mode').textContent() === 'LIVE MODEL OUTPUT');
    await page.locator('#playground-video').scrollIntoViewIfNeeded();
    await page.waitForFunction(() => {
      const video = document.querySelector('#playground-video');
      return video.readyState >= 3 && !video.paused && video.currentTime > 0.05;
    }, null, { timeout: 60000 });
    const videoState = () => page.locator('#playground-video').evaluate(video => ({ src: video.currentSrc, time: video.currentTime, paused: video.paused, width: video.videoWidth, height: video.videoHeight, duration: video.duration, controls: video.controls }));
    report.videoBefore = await videoState();
    await page.waitForTimeout(700);
    report.videoAfter = await videoState();
    check('returned job video plays in page', report.videoAfter.src.includes(accepted.jobId) && !report.videoAfter.paused);
    check('video playback time advances', Math.abs(report.videoAfter.time - report.videoBefore.time) > .1);
    check('no playback buttons', !report.videoAfter.controls);
    const manifestUrl = new URL(`/api/media/${accepted.jobId}/manifest.json`, admission.url()).href;
    report.manifest = await page.evaluate(async url => {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Manifest unavailable');
      return response.json();
    }, manifestUrl);
    check('saved model request exactly matches drawn points', JSON.stringify(report.manifest.points) === JSON.stringify(report.submitted.points));
    check('real OpenNWM DDPM 50-step manifest', report.manifest.diffusionSteps === 50 && report.manifest.protocol.includes('DDPM') && report.manifest.seed === report.submitted.seed);
    check('no page JavaScript errors', report.errors.length === 0);
    await page.locator('#playground').screenshot({ path: path.join(root, 'browser-custom-drawing.png') });
    report.completedAt = new Date().toISOString();
    console.log(JSON.stringify({ phase: 'completed', jobId: accepted.jobId, checks: report.checks, video: report.videoAfter, elapsedModelSeconds: report.manifest.elapsedBatchSeconds }, null, 2));
  } catch (error) {
    report.failure = error.stack;
    await page.screenshot({ path: path.join(root, 'browser-custom-failure.png') }).catch(() => {});
    throw error;
  } finally {
    fs.writeFileSync(path.join(root, 'browser-custom-drawing-report.json'), JSON.stringify(report, null, 2) + '\n');
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
