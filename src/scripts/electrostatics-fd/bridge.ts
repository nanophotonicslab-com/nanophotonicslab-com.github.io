/**
 * Promise-based bridge to the electrostatics-FD Pyodide worker.
 *
 * Deliberately not shaped like src/scripts/bem/pyodide-bridge.ts, which keeps one
 * pendingResolve/pendingReject pair that every method reassigns. This page
 * overlaps requests by construction — a cost estimate fires on every control
 * change while a solve may still be running, and the schematic re-requests its
 * geometry — and the first overlap there resolves the wrong promise. So every
 * request carries an id, the worker echoes it back, and a Map routes the reply.
 * Each entry owns a timeout too: a worker that posts nothing at all (a thrown
 * importScripts, an OOM kill) would otherwise leave its promise unsettled for the
 * life of the page.
 *
 * The wire format is split in two. The page -> worker half is imported as a TYPE
 * from pyodide-worker.ts, so a field renamed on either side is a compile error
 * rather than a request that silently does nothing; the import is type-only and
 * erased, so nothing here pulls the worker in as a module. The worker -> page
 * replies declared below have no counterpart over there — the worker posts them
 * as object literals — so these interfaces are the only description of that
 * half, and a change to one has to be made in both files.
 */

import type { InitMessage, RunMessage } from './pyodide-worker';
import { WHEEL_VERSION } from './wheel-version';

/** Everything `web.run` dispatches on, minus the three the bridge drives itself. */
export type EfdKind =
  | 'geometry'
  | 'material'
  | 'near_field' | 'eigenmodes' | 'bands' | 'sheet_dispersion'
  | 'linear_spectrum' | 'eels_spectrum' | 'ldos' | 'nonlinear_spectrum';

/**
 * Any payload from the Python boundary.
 *
 * `ok: false` is DATA, not a failure: a refused spec arrives here with a `reason`
 * the page is expected to show. `unexpected` marks the one case that is a bug
 * report rather than a user error. The index signature is what lets a page read
 * a payload field the bridge has no interface for yet.
 */
export interface EfdResult {
  ok: boolean;
  kind: string;
  schema: number;
  notes?: string[];
  reason?: string;
  unexpected?: boolean;
  [key: string]: unknown;
}

/**
 * What a spec will cost, from the spec alone. Only meaningful when `ok` is true —
 * a spec the library cannot even build comes back as `{ok: false, reason}` with
 * none of these fields set.
 */
export interface EfdCost extends EfdResult {
  n_nodes: number; n_q: number; n_energies: number;
  fast_path: boolean; fast_path_obstruction: string | null;
  n_full_solves: number; n_q_units: number;
  estimated_seconds: number; calibrated: boolean; cost_model: string;
  warn: boolean; refuse: boolean; is_bound: boolean;
}

// ── payload shapes ──────────────────────────────────────────────────────────
//
// Two conventions run through all of them. Grids (energies, q, node positions)
// are `number[]`: they are constructed, not computed, and are always finite.
// Computed values are `(number | null)[]`, because the library maps every
// non-finite float to null at the JSON boundary and a null must be drawn as a
// GAP — plotted as zero it would be a lie about the physics. 2D arrays are
// row-major rows: `a[i][j]` is `a[q][energy]` unless said otherwise.

/**
 * A complex array: Python always splits it into re/im.
 *
 * Both halves are `(number | null)[]`, not `number[]`: the Python side builds
 * them through the same `_fl` that maps every non-finite value to null, so a
 * component CAN arrive as null — the divergent null-space eigenvalue is the case
 * that manages it. Typing it `number[]` let callers write arithmetic that
 * silently treats a gap as zero, which is the one thing the null convention
 * exists to prevent.
 */
export interface EfdComplex {
  re: (number | null)[];
  im: (number | null)[];
}

/** One layer of the substrate stack. */
export interface EfdLayer {
  eps_par: number;
  d_nm: number;
}

/** One ribbon's parameters and its discretisation. */
export interface EfdGeometryRibbon {
  W_nm: number;
  x0_nm: number;
  z0_nm: number;
  EF_eV: number;
  gamma_eV: number;
  beta: number;
  /** Node count on this ribbon. */
  L: number;
  /** The step in theta, 1/(L - 1) — dimensionless, whatever the library's key says. */
  h_nm: number;
  /** Node positions, x0_nm + W_nm * theta. A ribbon IS this line of nodes at z0_nm, never a slab. */
  x_nodes_nm: number[];
  /** The dimensionless coordinate across the ribbon: linspace(-0.5, 0.5, L). */
  theta: number[];
}

/** `geometry` — structurally the GeometryPayload the schematic takes. */
export interface EfdGeometry extends EfdResult {
  n_ribbons: number;
  n_nodes_total: number;
  dx_nm: number;
  eps_eff: number;
  ribbons: EfdGeometryRibbon[];
  layers: EfdLayer[];
  notes: string[];
}

/** `eigenmodes` — the geometric eigenvalues at one q, bright modes by default. */
export interface EfdEigenmodes extends EfdResult {
  q_nm_inv: number;
  bright_only: boolean;
  /** Index of each returned mode in the full spectrum, so hidden modes stay countable. */
  mode_index: number[];
  eta: EfdComplex;
  zeta: EfdComplex;
  bright: boolean[];
  /** Closed-form Drude energy per returned mode; null where there is no real root. */
  E_drude_eV: (number | null)[];
  /** One row per MODE, over every node of the system concatenated. */
  profiles: number[][];
  /** One row per RIBBON. Stitching a profile back onto these is the page's job. */
  x_nodes_nm: number[][];
  n_modes_total: number;
  notes: string[];
}

/** `bands` — band energies over a q grid. Rows are `[q][band]`, not `[q][energy]`. */
export interface EfdBands extends EfdResult {
  qs_nm_inv: number[];
  n_bands: number;
  /** null where this band has no resonance inside the energy window at that q.
   *
   *  NOT only a failed root find, which is what this used to say. Bands are
   *  tracked across q, so a slot is one branch throughout and goes null
   *  wherever that branch's root falls outside [E_min, E_max] -- the ribbon's
   *  longitudinal branch does exactly that as q -> 0. A trailing run of nulls
   *  also means "fewer branches than n_bands were available". */
  E_eV: (number | null)[][];
  fwhm_eV: (number | null)[][];
  E_drude_eV: (number | null)[][];
  eta_re: (number | null)[][];
  eta_im: (number | null)[][];
  weight: (number | null)[][];
  bright: boolean[][];
  notes: string[];
}

/** `sheet_dispersion` — the extended sheet over (Q, E). No ribbon involved. */
export interface EfdSheetDispersion extends EfdResult {
  Qs_nm_inv: number[];
  energies_eV: number[];
  im_r: (number | null)[][];
  sigma_re: (number | null)[][];
  sigma_im: (number | null)[][];
  /** Present only with drift and a grid straddling Q = 0. */
  asymmetry_Q_nm_inv?: number[];
  asymmetry?: (number | null)[][];
  notes: string[];
}

/** The material panel beside a spectrum. `kind` is 'graphene' or 'metal'. */
export interface EfdMaterial {
  kind: string;
  sigma_nm_fs: EfdComplex;
  /** Graphene only, below. A metal sheet carries none of it. */
  EF_eV?: number;
  abs_EF_eV?: number;
  T_K?: number;
  /** 2|E_F|, where the page draws the Pauli-edge marker. */
  pauli_edge_eV?: number;
  sigma_drude_nm_fs?: EfdComplex;
  note?: string;
}

/** `linear_spectrum` — polarizability per unit length over (q, E). */
export interface EfdLinearSpectrum extends EfdResult {
  qs_nm_inv: number[];
  energies_eV: number[];
  alpha_re: (number | null)[][];
  alpha_im: (number | null)[][];
  /** Im(alpha): peak POSITIONS are what it is for, the absolute scale is convention. */
  extinction_arb: (number | null)[][];
  /** Extinction cross section, (4 pi omega/c) Im alpha, per unit length [nm]. */
  sigma_ext_nm?: (number | null)[][];
  /** The same divided by the total ribbon width: dimensionless. */
  sigma_ext_per_area?: (number | null)[][];
  /** EELS only. -Im of the induced charge against the electron's own
   *  potential: a loss, positive by construction, on an arbitrary scale. */
  loss_arb?: (number | null)[][];
  material: EfdMaterial;
  /** False when drift or a stack forced a full solve per (q, E) point. */
  fast_path: boolean;
  /** Present only with a q grid straddling zero. */
  asymmetry_q_nm_inv?: number[];
  asymmetry?: (number | null)[][];
  notes: string[];
}

/** `eels_spectrum` — the same spectrum under a swift electron's own potential. */
export interface EfdEelsSpectrum extends EfdLinearSpectrum {
  gamma_lorentz: number;
  /** Impact parameter. Aloof and parallel only; a touching trajectory is refused. */
  b_nm: number;
  v_over_c: number;
}

/** `ldos` — the (q, E) map, its peak tracks, and the ±q asymmetry. */
export interface EfdLdos extends EfdResult {
  qs_nm_inv: number[];
  energies_eV: number[];
  quantity: string;
  map: (number | null)[][];
  /** `[q][peak]`: the tracked peak positions and their weights. */
  peak_E_eV: (number | null)[][];
  peak_weight: (number | null)[][];
  fast_path: boolean;
  asymmetry_q_nm_inv?: number[];
  asymmetry?: (number | null)[][];
  /**
   * g(E), one per energy, present with a q grid straddling zero. The library
   * builds it float by float rather than through its own null mapping, but the
   * worker sanitises every non-finite float on the way out, so a value that did
   * not converge arrives here as null like everywhere else — not as NaN.
   */
  dissymmetry?: (number | null)[];
  notes: string[];
}

/** `nonlinear_spectrum` — SHG and THG at one q. Each order is screened at its own harmonic. */
export interface EfdNonlinearSpectrum extends EfdResult {
  q_nm_inv: number;
  energies_eV: number[];
  cascade: boolean;
  /** Present only when that order was asked for in spec.orders. */
  alpha_shg?: EfdComplex;
  alpha_thg?: EfdComplex;
  notes: string[];
}

/** What `init()` resolves with: this runtime's measured speed and the scaled cost model. */
export interface EfdCalibration extends EfdResult {
  n_nodes: number[];
  eigen_seconds: number[];
  solve_seconds: number[];
  per_q_scale: number;
  per_solve_scale: number;
  /** Inferred from per_solve_scale when the per-cell probe could not be timed. */
  per_cell_scale: number;
  /** Only from a wheel whose calibrate() times the per-cell term. */
  cell_probe_seconds?: number;
  model: Record<string, number | string | boolean>;
}

/** Where a preset's value came from. The page is expected to show it. */
export type EfdProvenanceTag = 'paper' | 'matlab' | 'assumed';

/** How strong the reproduction claim is. 'reference-default' matches no published figure. */
export type EfdPresetClaim = 'confirmed' | 'attributed' | 'reference-default';

export interface EfdPreset {
  id: string;
  label: string;
  citation: string;
  claim: EfdPresetClaim;
  kind: EfdKind;
  blurb: string;
  spec: Record<string, unknown>;
  /** Tag per spec field. A field missing from here was not tagged at all. */
  provenance: Record<string, EfdProvenanceTag>;
  notes: string[];
  /** Measured on a desktop CPython, so a floor for a browser — ask costEstimate for the real number. */
  estimated_seconds_desktop: number;
}

/** What `presets()` resolves with. */
export interface EfdPresets extends EfdResult {
  presets: EfdPreset[];
}

// ── the wire format, worker -> page ─────────────────────────────────────────

interface InitProgressReply { type: 'init-progress'; stage: string; pct: number }
interface InitDoneReply { type: 'init-done'; calibration: EfdResult }
interface InitErrorReply { type: 'init-error'; error: string }
interface ResultReply { type: 'result'; id: number; payload: EfdResult }
interface ErrorReply { type: 'error'; id: number; reason: string; unexpected?: boolean }

type WorkerReply =
  | InitProgressReply | InitDoneReply | InitErrorReply | ResultReply | ErrorReply;

/** A rejection out of the worker. `unexpected` means a boundary bug, not a bad spec. */
export class EfdWorkerError extends Error {
  readonly unexpected: boolean;

  constructor(reason: string, unexpected = false) {
    super(reason);
    this.name = 'EfdWorkerError';
    this.unexpected = unexpected;
  }
}

interface Pending {
  resolve: (value: EfdResult) => void;
  reject: (reason: Error) => void;
  kind: string;
  timer: ReturnType<typeof setTimeout>;
}

/** The wheel, with its own content hash as a cache buster.
 *
 *  `public/sw.js` caches `.whl` CACHE-FIRST keyed on the URL, and the
 *  filename carries only the package version -- which does not change when the
 *  Python inside it does. Without the hash, a browser that had loaded this page
 *  once kept running whatever wheel it first saw, through any number of
 *  rebuilds, and reported physics that the library on disk no longer computed.
 *  `scripts/build_efd_wheel.py` stamps `wheel-version.ts` on every build. */
const WHEEL_PATH =
  `wasm/electrostatics_fd-0.1.0-py3-none-any.whl?v=${WHEEL_VERSION}`;

/** Long enough for a map the cost model allows (its own ceiling is 120 s of work). */
const DEFAULT_TIMEOUT_MS = 180_000;

/**
 * Pyodide, numpy and scipy are ~34 MB cold. That is 23 s on a fast link and
 * minutes on a bad one, and a spurious failure costs more than a long wait, so
 * the constructor's timeoutMs — which is about calculations — does not apply
 * here. A page that wants to give up on the load sooner calls terminate().
 */
const INIT_TIMEOUT_MS = 600_000;

/** Reserved id for init: there is only ever one, so the worker need not echo it. */
const INIT_ID = 0;

export class EfdBridge {
  private worker: Worker | null = null;
  private readonly pending = new Map<number, Pending>();
  private nextId = INIT_ID + 1;
  private initPromise: Promise<EfdResult> | null = null;
  private progress: (stage: string, pct: number) => void = () => {};
  private readonly timeoutMs: number;
  private initialised = false;

  constructor(opts?: { timeoutMs?: number }) {
    this.timeoutMs = opts?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  /** True once Pyodide, the wheel and the calibration are all in place. */
  get ready(): boolean {
    return this.initialised;
  }

  onInitProgress(cb: (stage: string, pct: number) => void): void {
    this.progress = cb;
  }

  /** Resolves with the calibration result. Idempotent: one Pyodide per bridge. */
  init(): Promise<EfdResult> {
    if (this.initPromise) return this.initPromise;

    // The URL argument has to stay a static literal: Vite's worker plugin only
    // rewrites one of those, and a computed path ships this .ts URL to production
    // where it 404s.
    this.worker = new Worker(new URL('./pyodide-worker.ts', import.meta.url), { type: 'classic' });
    this.worker.onmessage = (e: MessageEvent) => this.handle(e.data as WorkerReply);
    this.worker.onerror = (e: ErrorEvent) => {
      // A worker that dies (a thrown importScripts, say) posts nothing further,
      // so everything in flight has to be failed from here.
      this.failAll(new EfdWorkerError(`worker failed: ${e.message || 'unknown error'}`, true));
    };

    // Resolved page-side, NOT from self.location.origin in the worker: that
    // discards any Astro base path and only happens to work on an apex domain.
    // BASE_URL already ends in '/'.
    const wheelUrl = new URL(`${import.meta.env.BASE_URL}${WHEEL_PATH}`, window.location.href).href;

    const message: InitMessage = { type: 'init', wheelUrl };
    this.initPromise = this.track(INIT_ID, 'init', INIT_TIMEOUT_MS, worker => {
      worker.postMessage(message);
    });
    return this.initPromise;
  }

  /**
   * One calculation. Resolves on `ok: false` as well: a refused spec is data the
   * page displays, not an exception. Rejects only on transport failure, timeout,
   * a dead worker, or a Python fault the worker could not turn into a payload.
   */
  run(kind: EfdKind, spec: Record<string, unknown>): Promise<EfdResult> {
    return this.send(kind, spec);
  }

  /** Price a spec without running it. `for_kind` is how the dispatcher knows what to price. */
  costEstimate(spec: Record<string, unknown>, forKind: EfdKind): Promise<EfdCost> {
    return this.send('cost_estimate', { ...spec, for_kind: forKind }) as Promise<EfdCost>;
  }

  /** The published parameter sets, each with its provenance and its claim. */
  presets(): Promise<EfdResult> {
    return this.send('presets', {});
  }

  terminate(): void {
    this.failAll(new EfdWorkerError('the electrostatics-fd worker was terminated'));
    this.worker?.terminate();
    this.worker = null;
    this.initialised = false;
    // Dropped rather than kept, so a page that terminates on hide can init again.
    this.initPromise = null;
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private send(kind: string, spec: Record<string, unknown>): Promise<EfdResult> {
    if (!this.initPromise) {
      return Promise.reject(new EfdWorkerError('EfdBridge.init() has not been called'));
    }
    // A control change during the cold load queues behind init instead of
    // throwing — the difference between a disabled button and a lost click.
    return this.initPromise.then(() => {
      const id = this.nextId++;
      const message: RunMessage = { type: 'run', id, kind, spec };
      return this.track(id, kind, this.timeoutMs, worker => {
        worker.postMessage(message);
      });
    });
  }

  private track(
    id: number,
    kind: string,
    timeoutMs: number,
    send: (worker: Worker) => void,
  ): Promise<EfdResult> {
    const worker = this.worker;
    if (!worker) return Promise.reject(new EfdWorkerError('the bridge has been terminated'));

    return new Promise<EfdResult>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new EfdWorkerError(`${kind} timed out after ${Math.round(timeoutMs / 1000)} s`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, kind, timer });
      try {
        send(worker);
      } catch (e) {
        // A spec structured-clone cannot copy (a function, a DOM node) would
        // otherwise sit in the map until its timeout.
        this.fail(id, e instanceof Error ? e : new EfdWorkerError(String(e)));
      }
    });
  }

  private handle(msg: WorkerReply): void {
    switch (msg.type) {
      case 'init-progress':
        this.progress(msg.stage, msg.pct);
        break;
      case 'init-done':
        this.initialised = true;
        this.settle(INIT_ID, msg.calibration);
        break;
      case 'init-error':
        this.fail(INIT_ID, new EfdWorkerError(msg.error, true));
        break;
      case 'result':
        this.settle(msg.id, msg.payload);
        break;
      case 'error':
        this.fail(msg.id, new EfdWorkerError(msg.reason, msg.unexpected ?? false));
        break;
    }
  }

  private settle(id: number, payload: EfdResult): void {
    const entry = this.take(id);
    entry?.resolve(payload);
  }

  private fail(id: number, error: Error): void {
    const entry = this.take(id);
    entry?.reject(error);
  }

  private take(id: number): Pending | undefined {
    const entry = this.pending.get(id);
    if (!entry) {
      // Already timed out, or terminated under it. Dropping the reply is right;
      // dropping it silently would hide a protocol mismatch.
      console.warn(`[electrostatics-fd] reply for request ${id} has no pending entry`);
      return undefined;
    }
    clearTimeout(entry.timer);
    this.pending.delete(id);
    return entry;
  }

  private failAll(error: Error): void {
    for (const entry of this.pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(error);
    }
    this.pending.clear();
  }
}
