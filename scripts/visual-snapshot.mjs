/**
 * Screenshot every Lab page into a directory, for before/after comparison.
 * Run with: npm run visual:snap -- <outdir> [page ...]
 *
 *   npm run dev &                          # the pages have to be served
 *   npm run visual:snap -- .visual/before
 *   ...make a CSS change...
 *   npm run visual:snap -- .visual/after
 *   npm run visual:diff -- .visual/before .visual/after
 *
 * Uses a Chromium already on the machine (the Playwright cache, or Chrome
 * itself) rather than adding a dependency — this is a local check, not a CI
 * gate, and the no-extra-deps constraint applies to tooling too.
 *
 * Limitation worth knowing: this captures a fixed 1440x2400 viewport, not the
 * whole scroll height. Anything below that fold is not compared, so a change
 * confined to the bottom of a long widget will read as clean. Raise it for
 * those, e.g. VISUAL_VIEWPORT=1440,4000.
 *
 * Overridable via env: VISUAL_BASE_URL, VISUAL_VIEWPORT, VISUAL_SETTLE_MS,
 * CHROME_PATH.
 */
import { existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { homedir } from 'node:os';
import { join } from 'node:path';

const BASE = process.env.VISUAL_BASE_URL ?? 'http://localhost:4321';
const VIEWPORT = process.env.VISUAL_VIEWPORT ?? '1440,2400';
/** Chromium renders on a virtual clock; enough budget for the widgets to compute and draw. */
const SETTLE_MS = Number(process.env.VISUAL_SETTLE_MS ?? 9000);

/** Depth-limited search for a Chromium executable under `dir`. */
function findBinary(dir, depth = 4) {
  if (depth < 0 || !existsSync(dir)) return null;
  let entries;
  try { entries = readdirSync(dir); } catch { return null; }
  const NAMES = ['chrome-headless-shell', 'Chromium', 'chrome'];
  for (const name of NAMES) {
    const p = join(dir, name);
    if (existsSync(p) && statSync(p).isFile()) return p;
  }
  for (const e of entries) {
    const p = join(dir, e);
    let st;
    try { st = statSync(p); } catch { continue; }
    if (st.isDirectory()) {
      const hit = findBinary(p, depth - 1);
      if (hit) return hit;
    }
  }
  return null;
}

function resolveBrowser() {
  if (process.env.CHROME_PATH) {
    if (!existsSync(process.env.CHROME_PATH)) {
      console.error(`CHROME_PATH points at ${process.env.CHROME_PATH}, which does not exist.`);
      process.exit(2);
    }
    return process.env.CHROME_PATH;
  }

  // Playwright's browser cache, newest build first
  for (const cache of [join(homedir(), 'Library/Caches/ms-playwright'), join(homedir(), '.cache/ms-playwright')]) {
    if (!existsSync(cache)) continue;
    const builds = readdirSync(cache)
      .filter((d) => d.startsWith('chromium'))
      .sort((a, b) => Number(b.split('-').pop()) - Number(a.split('-').pop()));
    for (const b of builds) {
      const hit = findBinary(join(cache, b));
      if (hit) return hit;
    }
  }

  for (const p of [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome',
  ]) if (existsSync(p)) return p;

  return null;
}

/** Route slug -> URL. `index` is the Lab landing page. */
function labPages() {
  return readdirSync('src/pages/lab')
    .filter((f) => f.endsWith('.astro'))
    .map((f) => f.replace(/\.astro$/, ''))
    .sort();
}

const [outDir, ...only] = process.argv.slice(2);
if (!outDir) {
  console.error('usage: npm run visual:snap -- <outdir> [page ...]');
  process.exit(2);
}

const browser = resolveBrowser();
if (!browser) {
  console.error(
    'No Chromium found. Set CHROME_PATH=/path/to/chrome, install Google Chrome,\n' +
    'or run `npx playwright install chromium` to populate the Playwright cache.'
  );
  process.exit(2);
}

try {
  execFileSync('curl', ['-sf', '-o', '/dev/null', `${BASE}/lab/`]);
} catch {
  console.error(`Nothing serving ${BASE}/lab/ — start the dev server first (npm run dev).`);
  process.exit(2);
}

mkdirSync(outDir, { recursive: true });
const pages = only.length ? only : labPages();
let failed = 0;

for (const page of pages) {
  const url = page === 'index' ? `${BASE}/lab/` : `${BASE}/lab/${page}/`;
  try {
    execFileSync(browser, [
      '--headless', '--disable-gpu', '--hide-scrollbars',
      `--window-size=${VIEWPORT}`,
      `--virtual-time-budget=${SETTLE_MS}`,
      `--screenshot=${join(outDir, `${page}.png`)}`,
      url,
    ], { stdio: 'ignore' });
    process.stdout.write('.');
  } catch {
    console.error(`\nfailed: ${page}`);
    failed++;
  }
}

console.log(`\n${pages.length - failed}/${pages.length} pages -> ${outDir}`);
console.log(`browser: ${browser}`);
process.exit(failed ? 1 : 0);
