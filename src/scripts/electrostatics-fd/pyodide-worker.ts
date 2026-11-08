/// <reference lib="webworker" />
/**
 * Web Worker: Pyodide, the electrostatics_fd wheel, and one calculation at a time.
 *
 * CLASSIC worker on purpose. Pyodide's CDN build is loaded through
 * importScripts(), which exists only in a classic worker, so THIS FILE CARRIES NO
 * RUNTIME ES IMPORT OR EXPORT — either one makes Vite emit a module chunk, and
 * importScripts then throws at runtime.
 *
 * It does carry a TYPE-ONLY export at the bottom, and that is deliberate and
 * load-bearing: without it TypeScript treats this file as a global script, and
 * its top-level `pyodide`, `post` and `init` then collide with the identically
 * named globals in src/scripts/bem/pyodide-worker.ts — six `astro check` errors
 * across two files, three of them reported against BEM, which this file never
 * touched. `export type` is erased before bundling, so the emitted chunk stays a
 * plain IIFE. src/scripts/rcwa/pyodide-worker.ts has relied on exactly this
 * since it shipped (its `export interface RcwaParams`); BEM is the one that
 * does not, which is why BEM is where the collision shows up. Do not remove it.
 *
 * The Python side (`electrostatics_fd.web.run`) never raises and never returns
 * anything but a dict, so almost everything here is about getting that dict
 * across the boundary without lying about it: see `_efd_jsonable` below.
 */

// @ts-ignore — Pyodide loaded from CDN
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js');

/**
 * Two cheap sizes, timed once at startup. The cost model's coefficients were
 * measured on a desktop CPython and WASM runs the same work about 3.5x slower on
 * solves, so an uncalibrated estimate understates a browser by enough to make
 * the warning useless. 61 and 151 nodes cost well under a second between them.
 */
const CALIBRATION_SIZES = [61, 151];

/**
 * Bootstrap. `_efd_wheel_url` is set as a global rather than interpolated into
 * this source: the page resolves it from Astro's BASE_URL and it has no business
 * being spliced into Python text.
 *
 * `_efd_jsonable` is the one liberty taken with the library's output. json.dumps
 * would write a bare `NaN` token, which JSON.parse rejects, and Pyodide's own
 * toJs() maps Python None to `undefined` rather than null. Both would cost the
 * page the distinction it needs most: a non-finite result must arrive as null and
 * plot as a gap, never as a zero and never as a silently dropped point.
 */
const BOOTSTRAP = `
import json, math
import micropip

await micropip.install(_efd_wheel_url, deps=False)
from electrostatics_fd import web


def _efd_jsonable(o):
    """Every non-finite float to None, recursively. Nothing else is touched."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _efd_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_efd_jsonable(v) for v in o]
    return o


def _efd_call(kind, spec_json):
    spec = json.loads(spec_json) if spec_json else {}
    return json.dumps(_efd_jsonable(web.run(kind, spec)), allow_nan=False)
`;

interface InitMessage {
  type: 'init';
  /** Absolute URL of the wheel, resolved page-side. The worker never invents one. */
  wheelUrl: string;
}

interface RunMessage {
  type: 'run';
  id: number;
  kind: string;
  spec?: Record<string, unknown>;
}

type InboundMessage = InitMessage | RunMessage;

let pyodide: any = null;
/** The `_efd_call(kind, spec_json) -> json` proxy, held for the worker's life. */
let call: any = null;

function post(msg: any) {
  self.postMessage(msg);
}

/**
 * The line of a Pyodide error worth putting in front of a reader. A PythonError
 * carries the whole traceback in `message`; its last line is the exception
 * itself, and the full text still goes to the worker's console for a bug report.
 */
function readableError(e: any): string {
  const full = e && e.message ? String(e.message) : String(e);
  console.error('[electrostatics-fd worker]', full);
  const lines = full.split('\n').map(s => s.trim()).filter(Boolean);
  return lines.length ? lines[lines.length - 1] : 'unknown worker error';
}

async function init(wheelUrl: string) {
  try {
    post({ type: 'init-progress', stage: 'Downloading the Pyodide runtime (~6 MB)…', pct: 0.05 });
    // @ts-ignore
    pyodide = await loadPyodide();

    post({ type: 'init-progress', stage: 'Loading numpy and scipy (~28 MB)…', pct: 0.3 });
    await pyodide.loadPackage(['numpy', 'scipy', 'micropip']);

    post({ type: 'init-progress', stage: 'Installing electrostatics-fd (125 kB)…', pct: 0.75 });
    pyodide.globals.set('_efd_wheel_url', wheelUrl);
    await pyodide.runPythonAsync(BOOTSTRAP);
    call = pyodide.globals.get('_efd_call');

    post({ type: 'init-progress', stage: 'Timing this browser…', pct: 0.9 });
    const calibration = JSON.parse(call('calibrate', JSON.stringify({ sizes: CALIBRATION_SIZES })));

    post({ type: 'init-progress', stage: 'Ready — first load cached for next time', pct: 1.0 });
    post({ type: 'init-done', calibration });
  } catch (e: any) {
    post({ type: 'init-error', error: readableError(e) });
  }
}

/**
 * One calculation, synchronously.
 *
 * Calling the Python function directly rather than through runPythonAsync is not
 * only cheaper (no code string to re-parse per request): runPythonAsync awaits
 * before it executes, so two overlapping requests — and this page overlaps by
 * construction — could swap their arguments through the globals. Nothing is given
 * up by blocking, because Python here is one thread either way.
 */
function runKind(id: number, kind: string, spec?: Record<string, unknown>) {
  if (!call) {
    post({ type: 'error', id, reason: 'Pyodide is still loading; init has not finished' });
    return;
  }
  try {
    post({ type: 'result', id, payload: JSON.parse(call(kind, JSON.stringify(spec ?? {}))) });
  } catch (e: any) {
    // web.run() is contracted never to raise, so anything caught here is a fault
    // in the boundary itself — a payload json could not represent, a broken
    // install — not a spec the user got wrong. Those come back as ok: false.
    post({ type: 'error', id, reason: readableError(e), unexpected: true });
  }
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data as InboundMessage;
  if (msg.type === 'init') {
    await init(msg.wheelUrl);
    return;
  }
  if (msg.type === 'run') {
    runKind(msg.id, msg.kind, msg.spec);
    return;
  }
  // Answer even a message this worker does not know, as long as it carried an id:
  // staying silent would leave the bridge's promise waiting for its timeout.
  const id = typeof e.data?.id === 'number' ? e.data.id : -1;
  post({ type: 'error', id, reason: `unknown worker message type: ${String(e.data?.type)}` });
};

// Type-only, erased at compile time: see the note at the top of this file. It
// makes the module boundary explicit instead of leaving it to the presence of an
// `export` somewhere, and lets bridge.ts `import type` the wire format rather
// than keeping a hand-synced second copy of it.
export type { InitMessage, RunMessage, InboundMessage };
