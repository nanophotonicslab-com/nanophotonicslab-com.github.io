/**
 * Surface-plasmon dispersion for planar layered media.
 *
 * The pure kernel behind the Heterostructures "Surface plasmons" module: the
 * lossless Drude permittivity, the transverse decay constant, the reflection
 * ratio that Economou writes as (1-R)/(1+R), and the bracketing root-finder
 * that turns a residual into a dispersion branch.
 *
 * Everything here is expressed in the normalized variables of
 *   E. N. Economou, "Surface Plasmons in Thin Films", Phys. Rev. 182, 539 (1969):
 *     u = omega / omega_p        q = k c / omega_p        d~ = d omega_p / c
 * so a branch is a curve u(q) and no dimensional constant enters.
 *
 * Extracted from the page script so it can be unit-tested; the page imports it.
 */

export interface PlasmonLayer {
  /** High-frequency permittivity. A pure dielectric has wpEv = 0. */
  epsInf: number;
  /** Plasma energy in eV. Zero for a dielectric. */
  wpEv: number;
}

/** Lossless Drude permittivity, eps(E) = epsInf - (wp/E)^2. */
export function epsilonLossless(layer: PlasmonLayer, energyEv: number): number {
  if (layer.wpEv <= 0) return layer.epsInf;
  if (energyEv <= 0) return Number.NaN;
  return layer.epsInf - (layer.wpEv * layer.wpEv) / (energyEv * energyEv);
}

/**
 * Transverse decay constant kappa = sqrt(q^2 - eps u^2), normalized to
 * omega_p/c. Returns null where the field is oscillatory rather than
 * evanescent (kappa^2 <= 0), which is where no bound mode exists.
 */
export function layerKappa(q: number, u: number, layer: PlasmonLayer, wp: number): number | null {
  const epsilon = epsilonLossless(layer, u * wp);
  const value = q * q - epsilon * u * u;
  if (!Number.isFinite(value) || value <= 0) return null;
  return Math.sqrt(value);
}

/**
 * The ratio x = (1 - R)/(1 + R) with R = -(kappa/eps)_metal / (kappa/eps)_dielectric,
 * i.e. the left-hand side of Economou's equations (3.17) and (3.23a).
 *
 * Returning this ratio rather than R itself is what keeps the film residuals
 * free of the pole at eps_metal -> 0, which would otherwise put a spurious root
 * at the bulk-plasmon frequency and swallow the lower branch.
 */
export function xRatioForPair(
  q: number,
  u: number,
  metal: PlasmonLayer,
  dielectric: PlasmonLayer,
  wp: number,
): number | null {
  const energyEv = u * wp;
  const epsMetal = epsilonLossless(metal, energyEv);
  const epsDielectric = epsilonLossless(dielectric, energyEv);
  const kappaMetal = layerKappa(q, u, metal, wp);
  const kappaDielectric = layerKappa(q, u, dielectric, wp);
  if (
    kappaMetal === null
    || kappaDielectric === null
    || !Number.isFinite(epsMetal)
    || !Number.isFinite(epsDielectric)
    || Math.abs(epsMetal) < 1e-10
    || Math.abs(epsDielectric) < 1e-10
  ) {
    return null;
  }
  const reflection = -((kappaMetal / epsMetal) / (kappaDielectric / epsDielectric));
  const denominator = 1 + reflection;
  if (!Number.isFinite(reflection) || Math.abs(denominator) < 1e-10) return null;
  return (1 - reflection) / denominator;
}

/** Hyperbolic cotangent, guarded at the origin. */
export function coth(value: number): number {
  const tanh = Math.tanh(value);
  if (Math.abs(tanh) < 1e-12) return value < 0 ? -1e12 : 1e12;
  return 1 / tanh;
}

/**
 * Residual of Economou's equation (3.23a) for a metal film of normalized
 * thickness `kpThickness` between two identical dielectric half-spaces (DMD).
 * The two signs select the two field symmetries; each has one root per q.
 */
export function dmdFilmResidual(
  q: number,
  u: number,
  metal: PlasmonLayer,
  dielectric: PlasmonLayer,
  wp: number,
  kpThickness: number,
  sign: 1 | -1,
): number | null {
  const x = xRatioForPair(q, u, metal, dielectric, wp);
  const km = layerKappa(q, u, metal, wp);
  if (x === null || km === null) return null;
  return x - sign * Math.exp(-km * kpThickness);
}

/**
 * Exact single-interface surface-plasmon frequency for a lossless Drude metal
 * (epsInf = 1) against a dielectric of permittivity `epsD`, in normalized
 * units. This is the d -> infinity limit of both film branches.
 *
 * From kappa_m/eps_m + kappa_d/eps_d = 0 with eps_m = 1 - 1/u^2:
 *     eps_d u^4 - u^2 [eps_d + q^2 (1 + eps_d)] + q^2 = 0,
 * whose lower root is the bound mode.
 */
export function singleInterfaceU(q: number, epsD: number): number {
  const b = epsD + q * q * (1 + epsD);
  const disc = b * b - 4 * epsD * q * q;
  if (disc < 0) return Number.NaN;
  return Math.sqrt((b - Math.sqrt(disc)) / (2 * epsD));
}

/**
 * Find every root of `residual` on [uMin, uMax] by scanning for sign changes
 * and bisecting each bracket. `residual` may return null where it is undefined
 * (no bound mode); those samples break the bracket rather than poisoning it.
 *
 * Roots closer together than 1e-4 are merged, so a curve that grazes zero does
 * not report a doublet.
 */
export function solveScalarRoots(
  residual: (u: number) => number | null,
  uMin: number,
  uMax: number,
  samples = 260,
): number[] {
  if (!Number.isFinite(uMin) || !Number.isFinite(uMax) || uMax <= uMin) return [];
  const roots: number[] = [];
  let previousU = uMin;
  let previousValue = residual(previousU);

  for (let i = 1; i <= samples; i++) {
    const u = uMin + (uMax - uMin) * i / samples;
    const value = residual(u);
    if (value === null || !Number.isFinite(value)) {
      previousU = u;
      previousValue = null;
      continue;
    }
    if (Math.abs(value) < 1e-7) {
      roots.push(u);
    } else if (previousValue !== null && Number.isFinite(previousValue) && previousValue * value < 0) {
      let lo = previousU;
      let hi = u;
      let flo = previousValue;
      for (let j = 0; j < 44; j++) {
        const mid = 0.5 * (lo + hi);
        const fmid = residual(mid);
        if (fmid === null || !Number.isFinite(fmid)) break;
        if (Math.abs(fmid) < 1e-9) {
          lo = mid;
          hi = mid;
          break;
        }
        if (flo * fmid <= 0) {
          hi = mid;
        } else {
          lo = mid;
          flo = fmid;
        }
      }
      roots.push(0.5 * (lo + hi));
    }
    previousU = u;
    previousValue = value;
  }

  return roots
    .sort((a, b) => a - b)
    .filter((root, index, sorted) => index === 0 || Math.abs(root - sorted[index - 1]) > 1e-4);
}
