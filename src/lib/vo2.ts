// Vanadium dioxide (VO₂) optical model — ported from the gsp_lib VO₂ library
// (Dias, Rodríguez Echarri, Rasmussen, García de Abajo & Cox, Light: Sci. Appl. 15, 218 (2026)).
//
// Each pure phase is a Drude–Lorentz dielectric; the mixed phase is combined with the
// Bruggeman effective-medium approximation. The parameter tables are a compact
// parameterization CALIBRATED to the qualitative spectra of the measured Verleur et al.
// (Phys. Rev. 172, 788, 1968) constants (not distributed with the paper), so VO₂ results are
// faithful in structure/trend but not last-digit values. Energies are in eV.

import { type Complex, cadd, csub, cmul, cscale, csqrt } from './complex';

/** Alias kept for existing imports — the canonical type lives in complex.ts. */
export type Cpx = Complex;

// ── pure-phase oscillator tables: (E0 [eV], strength S, γ [eV]) ──
const EPS_INF_INS = 4.0;
const OSC_INS: [number, number, number][] = [
  [0.060, 5.0, 0.015],   // IR phonon → large static permittivity below ~0.1 eV
  [1.020, 1.6, 0.45],    // semiconductor absorption onset (d∥ → π*)
  [1.900, 1.2, 0.65],    // interband
  [2.850, 2.0, 0.90],    // interband
  [3.900, 1.5, 1.00],    // UV interband
];
const EPS_INF_MET = 3.5;
const DRUDE_MET = { Ep: 3.30, Etau: 0.55 };   // plasma energy and damping [eV]
const OSC_MET: [number, number, number][] = [
  [1.500, 2.2, 1.20],    // interband
  [3.000, 6.0, 1.30],    // interband toward the visible
];

// eps(E) = eps_inf + Σ_k S_k E0_k² / (E0_k² − E² − i E γ_k)  [ − E_p²/(E(E + i E_τ)) for the metal ]
function lorentzSum(E: number, epsInf: number, osc: [number, number, number][], drude?: { Ep: number; Etau: number }): Cpx {
  let re = epsInf, im = 0;
  for (const [E0, S, g] of osc) {
    const dr = E0 * E0 - E * E, di = E * g;          // denominator D = dr − i·di
    const den = dr * dr + di * di;
    const c = S * E0 * E0 / den;                     // S E0² (dr + i·di) / |D|²
    re += c * dr; im += c * di;
  }
  if (drude) {
    const den = E * E * (E * E + drude.Etau * drude.Etau);
    re += -drude.Ep * drude.Ep * E * E / den;
    im += drude.Ep * drude.Ep * E * drude.Etau / den;
  }
  return { re, im };
}

/** Insulating (monoclinic, semiconducting) VO₂ permittivity at energy E [eV]. */
export const epsInsulating = (E: number): Cpx => lorentzSum(E, EPS_INF_INS, OSC_INS);
/** Metallic (rutile) VO₂ permittivity at energy E [eV]: Drude + interband Lorentz. */
export const epsMetallic = (E: number): Cpx => lorentzSum(E, EPS_INF_MET, OSC_MET, DRUDE_MET);

// ── complex helpers: canonical implementations in complex.ts (csqrt = principal branch) ──

/**
 * Bruggeman effective-medium permittivity of a mix of insulating (B) and metallic (A) VO₂
 * at metallic fraction fm ∈ [0,1] (Eq. 18). Returns the passive (Im ≥ 0) root of
 * 2 eps² − b·eps − A·B = 0,  b = A(3fm − 1) + B(2 − 3fm).
 */
export function bruggeman(epsI: Cpx, epsM: Cpx, fm: number): Cpx {
  const A = epsM, B = epsI;
  const b = cadd(cscale(A, 3 * fm - 1), cscale(B, 2 - 3 * fm));
  const disc = csqrt(cadd(cmul(b, b), cscale(cmul(A, B), 8)));
  const r1 = cscale(cadd(b, disc), 0.25), r2 = cscale(csub(b, disc), 0.25);
  return r1.im >= 0 ? r1 : r2;
}

/** Mixed-phase VO₂ permittivity at energy E [eV] and metallic fraction fm ∈ [0,1]. */
export const epsVO2 = (E: number, fm: number): Cpx => bruggeman(epsInsulating(E), epsMetallic(E), fm);

// ── phase-change driving law: temperature ⇄ metallic fraction (Eq. S21) ──
export const VO2_TM = 341.0;   // transition midpoint [K]
export const VO2_DT = 2.0;     // transition width [K]

/** Metallic fraction vs temperature T [K] (Eq. S21): fm = 1/(1 + exp(−(T − T_M)/ΔT)). */
export const fmFromTemperature = (T: number): number => 1 / (1 + Math.exp(-(T - VO2_TM) / VO2_DT));

/** Invert the sigmoid: temperature [K] giving metallic fraction fm ∈ (0,1). */
export const temperatureFromFm = (fm: number): number => {
  const f = Math.min(1 - 1e-9, Math.max(1e-9, fm));
  return VO2_TM - VO2_DT * Math.log(1 / f - 1);
};
