import { MATERIALS } from '../data/optical-constants';

export { MATERIALS };

export interface ComplexDielectric {
  re: number;
  im: number;
}

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
 * Interpolate tabulated (n, k) optical constants at a wavelength.
 *
 * The input table must be sorted by wavelength in micrometers. Values outside
 * the tabulated interval are clamped to the nearest endpoint, matching the
 * historical Mie-page behavior.
 */
export function interpolateNK(data: number[][], lambdaNm: number): [number, number] {
  const lam = lambdaNm / 1000;
  if (lam <= data[0][0]) return [data[0][1], data[0][2]];
  const last = data.length - 1;
  if (lam >= data[last][0]) return [data[last][1], data[last][2]];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (data[mid][0] <= lam) lo = mid;
    else hi = mid;
  }
  const t = (lam - data[lo][0]) / (data[hi][0] - data[lo][0]);
  return [
    data[lo][1] + t * (data[hi][1] - data[lo][1]),
    data[lo][2] + t * (data[hi][2] - data[lo][2]),
  ];
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
