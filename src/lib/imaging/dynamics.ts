/**
 * Particle dynamics. All lengths are nanometres in the sample plane; the
 * conversion to pixels happens only at render time.
 *
 * DeepTrack2 ships no motion models of its own — an audit of the whole package
 * finds no class or function encoding a law of movement. What it provides is a
 * *mechanism*: `SequentialProperty` / `Feature.to_sequential(position=rule)`
 * evaluated inside a `dt.Sequence`, where the user supplies an update rule that
 * receives `previous_value` (and, if wanted, the whole `previous_values`
 * history). Its own tutorials demonstrate exactly one motion model through that
 * mechanism — free Brownian motion, `previous + np.random.standard_normal()`
 * (tutorials DTAT331 §4, DTGS106 §6, DTAT311 §4.2).
 *
 * Every model below is therefore written as a per-frame update rule of the same
 * shape, so each one round-trips into DeepTrack2's own machinery with nothing
 * but numpy alongside it — see modules/diffusion-tracking/export-python.ts.
 */
import type { Rng } from './rng';

export type MotionKind = 'brownian' | 'directed' | 'confined' | 'network';

/** Parameters of the motion model, in kernel units (nm, s). */
export interface MotionParams {
  kind: MotionKind;
  /** Per-axis step standard deviation for one frame, in nm. */
  sigmaNm: number;
  /** Directed motion: speed in nm per frame, and its direction in radians. */
  driftNm?: number;
  driftAngle?: number;
  /** Confined motion: side of the reflecting corral, in nm. */
  corralNm?: number;
  /** Network motion: mesh spacing in nm, and the per-crossing hop probability. */
  meshNm?: number;
  hopProb?: number;
}

export interface Emitter {
  /** Sample-plane position in nm. */
  x: number;
  y: number;
  /** Photons emitted per frame, before detection. */
  photons: number;
  /** Centre of this particle's corral (confined motion only), in nm. */
  cx?: number;
  cy?: number;
}

/** Per-axis step standard deviation in nm for free 2D diffusion. */
export function stepSigmaNm(dUm2PerS: number, dtS: number): number {
  // D in um^2/s, dt in s -> sigma in um; x1000 for nm
  return 1000 * Math.sqrt(2 * dUm2PerS * dtS);
}

/** Positive modulo — periodic boundaries wrap rather than reflect or clamp. */
function wrap(v: number, period: number): number {
  const m = v % period;
  return m < 0 ? m + period : m;
}

/**
 * Fold a coordinate back into [lo, hi] by specular reflection, repeating as
 * many times as needed (a triangle wave), so even a step much larger than the
 * interval lands correctly rather than sticking to a wall.
 */
export function reflect(v: number, lo: number, hi: number): number {
  const span = hi - lo;
  if (!(span > 0)) return lo;
  let t = (v - lo) % (2 * span);
  if (t < 0) t += 2 * span;
  return lo + (t <= span ? t : 2 * span - t);
}

/**
 * Uniformly scatter `n` emitters over a square field of `fieldNm`.
 * Each particle also remembers where it started, which is the centre of its
 * corral when the motion model confines it.
 */
export function seedEmitters(n: number, fieldNm: number, photons: number, r: Rng): Emitter[] {
  const out: Emitter[] = [];
  for (let i = 0; i < n; i++) {
    const x = r.range(0, fieldNm);
    const y = r.range(0, fieldNm);
    out.push({ x, y, photons, cx: x, cy: y });
  }
  return out;
}

/**
 * One axis of one particle, advanced by a single frame.
 *
 * Split out per axis because every model here acts independently on x and y:
 * the drift is a fixed vector, the corral is a square, and the meshwork is a
 * square grid, so the two axes never couple.
 */
function stepAxis(
  pos: number, centre: number, drift: number, p: MotionParams, r: Rng,
): number {
  const proposed = pos + drift + r.normal(0, p.sigmaNm);

  switch (p.kind) {
    case 'brownian':
    case 'directed':
      return proposed;

    case 'confined': {
      // A corral fixed around where the particle started: hard reflecting walls.
      const half = (p.corralNm ?? 0) / 2;
      if (!(half > 0)) return proposed;
      return reflect(proposed, centre - half, centre + half);
    }

    case 'network': {
      // Meshwork ("fence") model: a periodic grid of compartments. A step that
      // would cross a mesh line succeeds only with probability hopProb, and is
      // otherwise reflected back into the compartment it started in. Free
      // diffusion inside a compartment, rare hops between them — which is what
      // makes the apparent diffusion slow down at long lag times.
      const mesh = p.meshNm ?? 0;
      const hop = p.hopProb ?? 0;
      if (!(mesh > 0)) return proposed;
      const cell = Math.floor(pos / mesh);
      if (Math.floor(proposed / mesh) === cell) return proposed;
      if (r.uniform() < hop) return proposed;
      return reflect(proposed, cell * mesh, (cell + 1) * mesh);
    }

    default:
      throw new Error(`unsupported motion: ${p.kind as string}`);
  }
}

/**
 * Advance every emitter by one frame, in place.
 *
 * Boundaries of the field of view are periodic in x and y: a particle leaving
 * one edge re-enters at the opposite one. This is visible in the trajectory
 * trails, so it is stated in the module's notes.
 *
 * A confined particle never reaches those edges — its corral holds it — so it
 * is left unwrapped, which also keeps its corral walls fixed in space. The
 * meshwork does wrap: its spacing is snapped upstream so a whole number of
 * compartments spans the field, and wrapping by the field is then wrapping by a
 * whole number of cells, leaving the grid seamless.
 */
export function step(emitters: Emitter[], p: MotionParams, fieldNm: number, r: Rng): void {
  const dx = p.kind === 'directed' ? (p.driftNm ?? 0) * Math.cos(p.driftAngle ?? 0) : 0;
  const dy = p.kind === 'directed' ? (p.driftNm ?? 0) * Math.sin(p.driftAngle ?? 0) : 0;
  const wraps = p.kind !== 'confined';
  for (const e of emitters) {
    const nx = stepAxis(e.x, e.cx ?? e.x, dx, p, r);
    const ny = stepAxis(e.y, e.cy ?? e.y, dy, p, r);
    e.x = wraps ? wrap(nx, fieldNm) : nx;
    e.y = wraps ? wrap(ny, fieldNm) : ny;
  }
}

/**
 * Unwrapped displacement between two positions on a periodic field: the
 * shortest signed separation. Needed for ground-truth MSD, where a wrap must
 * not be mistaken for a field-sized jump.
 */
export function minimumImage(d: number, period: number): number {
  const h = period / 2;
  let v = d % period;
  if (v > h) v -= period;
  if (v < -h) v += period;
  return v;
}
