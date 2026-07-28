/**
 * Particle dynamics. All lengths are nanometres in the sample plane; the
 * conversion to pixels happens only at render time.
 *
 * v1 implements free 2D Brownian motion. Drift and flow are later modules —
 * `step()` dispatches on the motion kind so adding them is a new branch plus a
 * new choice in the spec, not a new call site.
 */
import type { Rng } from './rng';

export type MotionKind = 'brownian';

export interface Emitter {
  /** Sample-plane position in nm. */
  x: number;
  y: number;
  /** Photons emitted per frame, before detection. */
  photons: number;
}

/** Per-axis step standard deviation in nm for free 2D diffusion. */
export function stepSigmaNm(dUm2PerS: number, dtS: number): number {
  // D in um^2/s, dt in s -> sigma in um; x1000 for nm
  return 1000 * Math.sqrt(2 * dUm2PerS * dtS);
}

/** Uniformly scatter `n` emitters over a square field of `fieldNm`. */
export function seedEmitters(n: number, fieldNm: number, photons: number, r: Rng): Emitter[] {
  const out: Emitter[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ x: r.range(0, fieldNm), y: r.range(0, fieldNm), photons });
  }
  return out;
}

/** Positive modulo — periodic boundaries wrap rather than reflect or clamp. */
function wrap(v: number, period: number): number {
  const m = v % period;
  return m < 0 ? m + period : m;
}

/**
 * Advance every emitter by one frame, in place.
 *
 * Boundaries are periodic in x and y: a particle leaving one edge re-enters at
 * the opposite one. This is visible in the trajectory trails, so it is stated
 * in the module's notes.
 */
export function step(
  emitters: Emitter[], kind: MotionKind, sigmaNm: number, fieldNm: number, r: Rng,
): void {
  if (kind !== 'brownian') throw new Error(`unsupported motion: ${kind}`);
  for (const e of emitters) {
    e.x = wrap(e.x + r.normal(0, sigmaNm), fieldNm);
    e.y = wrap(e.y + r.normal(0, sigmaNm), fieldNm);
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
