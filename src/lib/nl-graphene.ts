// Nonlinear plasmonic polarizabilities of doped graphene nanostructures.
//
// Faithful TypeScript port of the `nlgraphene` reference (analytical model of
// J. D. Cox, R. Yu & F. J. García de Abajo, "Analytical description of the
// nonlinear plasmonic response in nanographene", Phys. Rev. B 96, 045442 (2017)).
// Extended-graphene Drude conductivities (Eqs. 8, 10) feed an electrostatic
// eigenmode expansion; the lowest dipole mode gives closed-form linear (α¹¹),
// SHG (α²²), Kerr (α³¹) and THG (α³³) polarizabilities (Eqs. 29), set by the
// dimensionless Table-I numbers per geometry. Validated to <4×10⁻⁶ against the
// library. Energies in eV, lengths in nm. Absolute magnitudes are in the model's
// internal Gaussian (eV-nm-fs) units; spectral shapes and relative magnitudes
// across a size/doping series are physical (the paper plots arbitrary units).

export type NlOrder = '11' | '22' | '31' | '33';
export type NlGeometryId = 'ribbon' | 'triangle' | 'disk';

interface Cpx { re: number; im: number; }

const HBAR = 0.6582119569;   // ħ [eV·fs]
const E2 = 1.439964;          // e²/(4πε₀) [eV·nm]
const VF = 1.0;               // v_F [nm/fs]
const E3 = E2 ** 1.5, E4 = E2 ** 2, PI = Math.PI;

const c = (re: number, im = 0): Cpx => ({ re, im });
const cAdd = (a: Cpx, b: Cpx): Cpx => ({ re: a.re + b.re, im: a.im + b.im });
const cSub = (a: Cpx, b: Cpx): Cpx => ({ re: a.re - b.re, im: a.im - b.im });
const cMul = (a: Cpx, b: Cpx): Cpx => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const cDiv = (a: Cpx, b: Cpx): Cpx => { const d = b.re * b.re + b.im * b.im; return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d }; };
const cScale = (a: Cpx, s: number): Cpx => ({ re: a.re * s, im: a.im * s });
const cAbs2 = (a: Cpx): number => a.re * a.re + a.im * a.im;
const Ic = c(0, 1);

export interface NlGeometry {
  id: NlGeometryId; label: string; dLabel: string;
  eta1: number; zeta1x: number; xt22A: number; xt22B: number; xt22C: number; xt31: number; xt33: number;
  perLength: boolean; hasShg: boolean;
}

export const NL_GEOMETRIES: NlGeometry[] = [
  { id: 'ribbon', label: 'Ribbon', dLabel: 'width', eta1: -0.06873, zeta1x: 0.9428, xt22A: 0, xt22B: 0, xt22C: 0, xt31: 1.031, xt33: -0.9415, perLength: true, hasShg: false },
  { id: 'triangle', label: 'Triangle', dLabel: 'side', eta1: -0.08780, zeta1x: 0.5437, xt22A: 0.3192, xt22B: -0.3742, xt22C: -0.7490, xt31: 0.2816, xt33: 0.2608, perLength: false, hasShg: true },
  { id: 'disk', label: 'Disk', dLabel: 'diameter', eta1: -0.07310, zeta1x: 0.8510, xt22A: 0, xt22B: 0, xt22C: 0, xt31: 0.7728, xt33: 0.7334, perLength: false, hasShg: false },
];
export const getNlGeometry = (id: string): NlGeometry => NL_GEOMETRIES.find(g => g.id === id) ?? NL_GEOMETRIES[1];

// ── extended-graphene conductivities (Eqs. 8, 10) ──
const sigma11 = (E: number, EF: number, g: number) => cDiv(cScale(Ic, E2 * EF / (PI * HBAR)), c(E, g));   // i e²E_F/[π ħ(E+iγ)]
const PRE2 = E3 * VF * VF / (4 * PI * HBAR * HBAR);
const pre3 = (EF: number) => E4 * VF * VF / (4 * PI * HBAR * HBAR * EF);

function sigma22(which: 'A' | 'B' | 'C', E: number, g: number, doping: number): Cpx {
  const w = E / HBAR, ti = g / HBAR;
  const pref0 = cDiv(cScale(Ic, PRE2), cMul(c(2 * w, ti), c(w, ti)));
  let s: number, bracket: Cpx;
  if (which === 'A') { s = -doping; bracket = cAdd(cDiv(c(3), c(w, ti)), cDiv(c(4), c(2 * w, ti))); }
  else if (which === 'B') { s = -doping; bracket = cSub(cAdd(cDiv(c(-1), c(w, ti)), cDiv(c(4), c(2 * w, ti))), c(4 / w)); }
  else { s = doping; bracket = cSub(cAdd(cDiv(c(0.5), c(w, ti)), cDiv(c(2), c(2 * w, ti))), c(2 / w)); }
  return cMul(cScale(pref0, s), bracket);
}
const sigma31 = (E: number, EF: number, g: number) => { const w = E / HBAR, ti = g / HBAR; return cDiv(cScale(Ic, -3 * pre3(EF)), cMul(c(2 * w, ti), c(w * w + ti * ti))); };
const sigma33 = (E: number, EF: number, g: number) => { const w = E / HBAR, ti = g / HBAR; return cDiv(cScale(Ic, 3 * pre3(EF)), cMul(cMul(c(3 * w, ti), c(2 * w, ti)), c(w, ti))); };

const dPow = (geo: NlGeometry, p: number) => geo.perLength ? p - 1 : p;

// ── 2D sheet conductivity σ²ᴰ(ω) [nm/fs] per material ──
export const grapheneSigma2d = (E: number, EF: number, gamma: number): Cpx => sigma11(E, EF, gamma);   // Eq. 8 (Drude)
export function metalSigma2d(E: number, epsRe: number, epsIm: number, tNm: number): Cpx {
  const w = E / HBAR;   // thin-film mapping: σ²ᴰ = i (ωt/4π)(1 − ε)
  return cScale(cMul(Ic, cSub(c(1), c(epsRe, epsIm))), w * tNm / (4 * PI));
}

// Material side: σ²ᴰ(E) provider + the graphene (E_F, γ) used by the nonlinear
// conductivities. For metal/custom σ the nonlinear orders carry graphene's nonlinear
// σ (illustrative); only the linear α¹¹ and the resonance η(ω) are material-exact.
export interface NlMaterial { sigma2d: (E: number) => Cpx; nlEF: number; nlGamma: number; doping: number; }

// Complex polarizability of one order at photon energy E (eV). Eqs. (29a–d).
export function nlAlphaAt(order: NlOrder, geo: NlGeometry, D: number, epsBar: number, mat: NlMaterial, E: number): Cpx {
  const w = E / HBAR;
  const s2d = mat.sigma2d(E);
  const etaE = cScale(cMul(Ic, s2d), 1 / (w * D * epsBar));   // η = iσ²ᴰ/(ωDε̄)  (Eq. 16)
  const en = cSub(c(1), cScale(etaE, 1 / geo.eta1));          // 1 − η/η_j
  if (order === '11') return cDiv(cScale(cMul(cScale(Ic, D ** dPow(geo, 2) / w), s2d), geo.zeta1x ** 2), en);
  if (order === '22') {
    if (!geo.hasShg) return c(0, 0);
    const num = cAdd(cAdd(cScale(sigma22('A', E, mat.nlGamma, mat.doping), geo.xt22A), cScale(sigma22('B', E, mat.nlGamma, mat.doping), geo.xt22B)), cScale(sigma22('C', E, mat.nlGamma, mat.doping), geo.xt22C));
    return cDiv(cMul(cScale(Ic, -(D ** dPow(geo, 1)) / (2 * w)), num), cMul(en, en));
  }
  if (order === '31') return cDiv(cScale(cMul(cScale(Ic, D ** dPow(geo, 2) / w), sigma31(E, mat.nlEF, mat.nlGamma)), 3 * geo.xt31), cMul(cMul(en, en), c(cAbs2(en))));
  return cDiv(cScale(cMul(cScale(Ic, D ** dPow(geo, 2) / (3 * w)), sigma33(E, mat.nlEF, mat.nlGamma)), geo.xt33), cMul(cMul(en, en), en));
}

// Graphene closed-form plasmon energy ℏω_p = √(e²E_F/(π ε̄ D|η₁|)).
export const nlResonanceEnergyGraphene = (geoId: NlGeometryId, EF: number, D: number, epsBar: number) => Math.sqrt(E2 * EF / (PI * epsBar * D * Math.abs(getNlGeometry(geoId).eta1)));

// Generic plasmon energy: where Re[η(E)] crosses η₁ (smallest |Im η|). NaN if none.
export function nlResonance(geoId: NlGeometryId, D: number, epsBar: number, sigma2d: (E: number) => Cpx, Elo = 0.05, Ehi = 8, n = 4000): number {
  const eta1 = getNlGeometry(geoId).eta1;
  let prevF = NaN, prevE = NaN, best = NaN, bestImAbs = Infinity;
  for (let i = 0; i < n; i++) {
    const E = Elo + (Ehi - Elo) * i / (n - 1), w = E / HBAR, s = sigma2d(E);
    const etaE = cScale(cMul(Ic, s), 1 / (w * D * epsBar));
    const f = etaE.re - eta1;
    if (i > 0 && Number.isFinite(prevF) && prevF * f <= 0) { const Ec = prevE + (E - prevE) * (-prevF) / (f - prevF); const ia = Math.abs(etaE.im); if (ia < bestImAbs) { bestImAbs = ia; best = Ec; } }
    prevF = f; prevE = E;
  }
  return best;
}

// Spectrum of one order over an energy grid → Re/Im arrays.
export function nlSpectrum(order: NlOrder, geoId: NlGeometryId, D: number, epsBar: number, mat: NlMaterial, Eev: Float64Array): { re: Float64Array; im: Float64Array } {
  const geo = getNlGeometry(geoId);
  const n = Eev.length, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { const a = nlAlphaAt(order, geo, D, epsBar, mat, Eev[i]); re[i] = a.re; im[i] = a.im; }
  return { re, im };
}
