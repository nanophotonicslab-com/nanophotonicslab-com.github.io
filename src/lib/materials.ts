import { MATERIALS } from '../data/optical-constants';

export { MATERIALS };

import type { Complex } from './complex';
import { cubicSpline, type CubicSplineFn } from './cubic-spline';

/** Alias kept for existing imports — the canonical type lives in complex.ts. */
export type ComplexDielectric = Complex;

export interface DrudeParams {
  epsB: number;
  wpEv: number;
  gammaEv: number;
}

export const PHOTON_HC_EV_NM = 1239.841984;

/**
 * Shared optical-material helpers for all Lab widgets.
 *
 * Material data are stored as refractive-index samples in the same
 * refractiveindex.info-compatible format used by the Mie page:
 * `[[wavelength_um, n, k], ...]`.  Widgets should resolve materials through
 * this module instead of duplicating interpolation, Drude, or epsilon
 * conversion code.  That keeps built-in tables, uploaded data, and database
 * results numerically consistent across sphere, cylinder, and nanoparticle
 * tools.
 */

/**
 * Cached spline pair for one table, built on first use and reused afterwards.
 *
 * Keyed by the table object itself, so the built-in materials build their
 * splines once for the lifetime of the page while an uploaded or database
 * table is rebuilt only when the user actually picks a different one.
 */
interface EpsilonSplines {
  re: CubicSplineFn;
  im: CubicSplineFn;
  evMin: number;
  evMax: number;
  /** Endpoint (n, k) in wavelength order, for values outside the table. */
  loNK: [number, number];
  hiNK: [number, number];
}

const SPLINE_CACHE = new WeakMap<number[][], EpsilonSplines>();

function buildEpsilonSplines(data: number[][]): EpsilonSplines {
  // The table arrives as [[λ_µm, n, k], …] sorted by wavelength, so energy
  // runs the other way; the spline wants a strictly increasing abscissa.
  const ev: number[] = [];
  const epsRe: number[] = [];
  const epsIm: number[] = [];
  for (let i = data.length - 1; i >= 0; i--) {
    const [lamUm, n, k] = data[i];
    const e = PHOTON_HC_EV_NM / (lamUm * 1000);
    // Duplicate wavelengths would make the spline singular. Keep the first.
    if (ev.length > 0 && e <= ev[ev.length - 1]) continue;
    ev.push(e);
    epsRe.push(n * n - k * k);
    epsIm.push(2 * n * k);
  }
  const last = data.length - 1;
  return {
    re: cubicSpline(ev, epsRe),
    im: cubicSpline(ev, epsIm),
    evMin: ev[0],
    evMax: ev[ev.length - 1],
    loNK: [data[0][1], data[0][2]],
    hiNK: [data[last][1], data[last][2]],
  };
}

/**
 * Interpolate tabulated (n, k) optical constants at a wavelength.
 *
 * The input table must be sorted by wavelength in micrometers. Interpolation
 * is a cubic spline on the complex permittivity ε = (n + ik)² against photon
 * energy — the same quantity, the same abscissa and the same not-a-knot
 * spline the Pyodide BEM solver applies to its own copy of the tables. Two
 * codes reading one published table therefore see one material, which is what
 * makes a BEM-against-Mie comparison in the browser a statement about the
 * methods rather than about their interpolation.
 *
 * This was piecewise-linear until the interband edge of gold showed what that
 * costs: the Johnson and Christy table is sampled every 20 nm there while
 * dn/dλ changes by a factor of three across a single node at 471 nm, and the
 * resulting corner is plainly visible in any spectrum computed through it.
 *
 * Values outside the tabulated interval are clamped to the nearest endpoint,
 * matching the historical Mie-page behavior.
 */
export function interpolateNK(data: number[][], lambdaNm: number): [number, number] {
  const last = data.length - 1;
  if (last <= 0) return [data[0][1], data[0][2]];

  let splines = SPLINE_CACHE.get(data);
  if (!splines) {
    splines = buildEpsilonSplines(data);
    SPLINE_CACHE.set(data, splines);
  }

  const ev = PHOTON_HC_EV_NM / lambdaNm;
  if (ev <= splines.evMin) return splines.hiNK;
  if (ev >= splines.evMax) return splines.loNK;

  // A spline through a table whose absorption falls to zero can undershoot
  // into ε'' < 0, which would be gain rather than loss. Hold it at zero: a
  // transparent material stays transparent instead of becoming unphysical.
  return epsilonToNK(splines.re(ev), Math.max(0, splines.im(ev)));
}

export function nkToEpsilon(n: number, k: number): ComplexDielectric {
  return { re: n * n - k * k, im: 2 * n * k };
}

export function epsilonToNK(epsRe: number, epsIm: number): [number, number] {
  const epsAbs = Math.hypot(epsRe, epsIm);
  const n = Math.sqrt(Math.max(0, (epsAbs + epsRe) / 2));
  const k = Math.sqrt(Math.max(0, (epsAbs - epsRe) / 2));
  return [n, Math.abs(k)];
}

export function nkFromMaterial(materialId: string, lambdaNm: number): [number, number] {
  const material = MATERIALS[materialId];
  if (!material) {
    throw new Error(`Unknown material "${materialId}".`);
  }
  return interpolateNK(material.data, lambdaNm);
}

export function epsilonFromNKData(data: number[][], lambdaNm: number): ComplexDielectric {
  const [n, k] = interpolateNK(data, lambdaNm);
  return nkToEpsilon(n, k);
}

export function epsilonFromMaterial(materialId: string, lambdaNm: number): ComplexDielectric {
  const [n, k] = nkFromMaterial(materialId, lambdaNm);
  return nkToEpsilon(n, k);
}

export function drudeEpsilon(lambdaNm: number, params: DrudeParams): ComplexDielectric {
  const energyEv = PHOTON_HC_EV_NM / lambdaNm;
  const denRe = energyEv * energyEv;
  const denIm = energyEv * params.gammaEv;
  const denAbs2 = denRe * denRe + denIm * denIm;
  const termRe = params.wpEv * params.wpEv * denRe / denAbs2;
  const termIm = -params.wpEv * params.wpEv * denIm / denAbs2;
  return { re: params.epsB - termRe, im: -termIm };
}

export function drudeNK(lambdaNm: number, params: DrudeParams): [number, number] {
  const eps = drudeEpsilon(lambdaNm, params);
  return epsilonToNK(eps.re, eps.im);
}
