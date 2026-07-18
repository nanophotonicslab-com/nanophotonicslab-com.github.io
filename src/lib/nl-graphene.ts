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

import { grapheneMuT, mikhailovS3THG, mikhailovS3Kerr, KB_EV } from './graphene-conductivity';

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
const cLog = (a: Cpx): Cpx => ({ re: 0.5 * Math.log(a.re * a.re + a.im * a.im), im: Math.atan2(a.im, a.re) });
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

// Third-order interband enhancement R(ω) = σ³_full/σ³_intra: full-RPA upgrade for the Drude
// σ³³ (THG) and σ³¹ (Kerr). The Mikhailov σ³ (PRB 93, 085403) lives in the shared module.
export const grapheneR_THG = (E: number, EF: number, gamma: number): Cpx => {
  const { full, intra } = mikhailovS3THG(E, EF, gamma);
  return cDiv(full, intra);
};
export const grapheneR_Kerr = (E: number, EF: number, gamma: number): Cpx => {
  const { full, intra } = mikhailovS3Kerr(E, EF, gamma);
  return cDiv(full, intra);
};

const dPow = (geo: NlGeometry, p: number) => geo.perLength ? p - 1 : p;

// ── 2D sheet conductivity σ²ᴰ(ω) [nm/fs] per material ──
export const grapheneSigma2d = (E: number, EF: number, gamma: number): Cpx => sigma11(E, EF, gamma);   // Eq. 8 (Drude / BTE)
// Local-RPA interband linear conductivity (Hanson form, T=0): the reactive part below the
// 2E_F onset red-shifts the plasmon resonances — strongly for higher modes, barely for the dipole.
//   σ_inter = i (e²/4πħ) ln[(2E_F − (E+iγ)) / (2E_F + (E+iγ))]   [nm/fs]
export function grapheneInterband(E: number, EF: number, gamma: number): Cpx {
  return cScale(cMul(Ic, cLog(cDiv(c(2 * EF - E, -gamma), c(2 * EF + E, gamma)))), E2 / (4 * PI * HBAR));
}
// Finite-T local-RPA: thermal μ(T) and σ₀ come from the shared graphene-conductivity module.
const SIGMA0_NMFS = E2 / (4 * HBAR);           // σ₀ = e²/4ħ  [nm/fs]
// Local-RPA linear conductivity = Drude (intraband) + interband; optional electron temperature
// T [K] (default 0 → T-limit closed forms). Finite T uses the thermal μ and a smeared interband.
export function grapheneSigma2dRPA(E: number, EF: number, gamma: number, T = 0): Cpx {
  if (T < 1) return cAdd(grapheneSigma2d(E, EF, gamma), grapheneInterband(E, EF, gamma));
  const KT = KB_EV * T, mu = grapheneMuT(EF, T), x = E - 2 * mu;
  const intra = cDiv(cScale(Ic, E2 * mu / (PI * HBAR)), c(E, gamma));                                  // iE²μ(T)/[πħ(E+iγ)]
  // Interband step smeared by thermal AND collisional broadening in quadrature:
  // W → γ as T→0 recovers the zero-T (γ-broadened) branch continuously; the pure
  // 2kT form is discontinuous across the T switch whenever γ ≫ kT.
  const W = Math.hypot(2 * KT, gamma);
  const inter = cScale(c(0.5 + Math.atan(x / W) / PI,
    Math.log((x * x + W * W) / ((E + 2 * mu) * (E + 2 * mu))) / (2 * PI)), SIGMA0_NMFS);
  return cAdd(intra, inter);
}
export function metalSigma2d(E: number, epsRe: number, epsIm: number, tNm: number): Cpx {
  const w = E / HBAR;   // thin-film mapping: σ²ᴰ = i (ωt/4π)(1 − ε)
  return cScale(cMul(Ic, cSub(c(1), c(epsRe, epsIm))), w * tNm / (4 * PI));
}

// Material side: σ²ᴰ(E) provider + the graphene (E_F, γ) used by the nonlinear
// conductivities. For metal/custom σ the nonlinear orders carry graphene's nonlinear
// σ (illustrative); only the linear α¹¹ and the resonance η(ω) are material-exact.
// sigma3Boost (optional) multiplies the Drude third-order σ by the full-RPA interband ratio
// (Mikhailov): for graphene local-RPA, '33' → R_THG, upgrading Drude THG to full RPA.
export interface NlMaterial { sigma2d: (E: number) => Cpx; nlEF: number; nlGamma: number; doping: number; sigma3Boost?: (order: NlOrder, E: number) => Cpx; }
const matSigma31 = (mat: NlMaterial, E: number): Cpx => { const s = sigma31(E, mat.nlEF, mat.nlGamma); return mat.sigma3Boost ? cMul(s, mat.sigma3Boost('31', E)) : s; };
const matSigma33 = (mat: NlMaterial, E: number): Cpx => { const s = sigma33(E, mat.nlEF, mat.nlGamma); return mat.sigma3Boost ? cMul(s, mat.sigma3Boost('33', E)) : s; };

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
  if (order === '31') return cDiv(cScale(cMul(cScale(Ic, D ** dPow(geo, 2) / w), matSigma31(mat, E)), 3 * geo.xt31), cMul(cMul(en, en), c(cAbs2(en))));
  return cDiv(cScale(cMul(cScale(Ic, D ** dPow(geo, 2) / (3 * w)), matSigma33(mat, E)), geo.xt33), cMul(cMul(en, en), en));
}

// Graphene sheet conductivity for a chosen response order (Material view):
//   11 → linear σ¹¹ (Eq. 8); 22 → SHG σ²² = σ²²ᴬ+σ²²ᴮ+σ²²ᶜ; 31 → Kerr σ³¹; 33 → THG σ³³ (Eq. 10). [nm/fs]
export function grapheneSigmaOrder(order: NlOrder, E: number, EF: number, gamma: number, doping = 1): Cpx {
  if (order === '11') return grapheneSigma2d(E, EF, gamma);
  if (order === '22') return cAdd(cAdd(sigma22('A', E, gamma, doping), sigma22('B', E, gamma, doping)), sigma22('C', E, gamma, doping));
  if (order === '31') return sigma31(E, EF, gamma);
  return sigma33(E, EF, gamma);
}

// ── Higher-order modes (j ≤ 3) — full multimode polarizabilities, Eqs. (27), all geometries ──
// SM Table S2 gives the eigenvalues (η_j) and mode dipoles (ζ_{j,x}); Tables S4/S5 give the
// third-order tensor ξ³_{jklm}, tabulated for the ribbon, triangle and disk (notes/03 + crops/si).
// Summing j = 1..nmodes adds the secondary peaks at the higher-mode plasmons η₂, η₃ (paper Fig. 2).
// The linear α¹¹ is exact for every geometry; the nonlinear orders apply the SI ξ³ tensor directly.
export const RIBBON_ETA_J = [-0.0687, -0.0184, -0.0107];      // η₁, η₂, η₃ (Table S2)
export const RIBBON_ZETA_J = [0.943, 0.239, 0.137];           // ζ_{1x}, ζ_{2x}, ζ_{3x}
const RIBBON_XI3: number[][][][] = [
  [[[1.30448, -0.397492, -0.053434], [-0.397492, 0.996661, -0.387999], [-0.053434, -0.387999, 0.999412]],
   [[-0.397492, 0.996661, -0.387999], [0.996661, -0.060738, 0.553514], [-0.387999, 0.553514, -0.006479]],
   [[-0.053434, -0.387999, 0.999412], [-0.387999, 0.553514, -0.006479], [0.999412, -0.006479, -0.031209]]],
  [[[-0.397492, 0.996661, -0.387999], [0.996661, -0.060738, 0.553514], [-0.387999, 0.553514, -0.006479]],
   [[0.996661, -0.060738, 0.553514], [-0.060738, 1.44575, -0.040959], [0.553514, -0.040959, 0.962558]],
   [[-0.387999, 0.553514, -0.006479], [0.553514, -0.040959, 0.962558], [-0.006479, 0.962558, -0.035376]]],
  [[[-0.053434, -0.387999, 0.999412], [-0.387999, 0.553514, -0.006479], [0.999412, -0.006479, -0.031209]],
   [[-0.387999, 0.553514, -0.006479], [0.553514, -0.040959, 0.962558], [-0.006479, 0.962558, -0.035376]],
   [[0.999412, -0.006479, -0.031209], [-0.006479, 0.962558, -0.035376], [-0.031209, -0.035376, 1.46822]]],
];
const TRIANGLE_XI3: number[][][][] = [
  [[[3.22255, -0.333728, -0.322105], [-0.333728, 2.33422, -0.74661], [-0.322105, -0.74661, 2.3374]],
   [[-0.333728, 2.00088, -1.12261], [2.00088, -1.11085, 1.44667], [-1.12261, 1.44667, -0.709045]],
   [[-0.322105, -1.12261, 1.70401], [-1.12261, 1.25855, -0.948622], [1.70401, -0.948622, 0.758705]]],
  [[[-0.333728, 2.00088, -1.12261], [2.00088, -1.11085, 1.44667], [-1.12261, 1.44667, -0.709045]],
   [[2.33422, -1.11085, 1.25855], [-1.11085, 3.95477, -1.29436], [1.25855, -1.29436, 3.25688]],
   [[-0.74661, 1.44667, -0.948622], [1.44667, -1.29436, 2.84265], [-0.948622, 2.84265, -1.31038]]],
  [[[-0.322105, -1.12261, 1.70401], [-1.12261, 1.25855, -0.948622], [1.70401, -0.948622, 0.758705]],
   [[-0.74661, 1.44667, -0.948622], [1.44667, -1.29436, 2.84265], [-0.948622, 2.84265, -1.31038]],
   [[2.3374, -0.709045, 0.758705], [-0.709045, 3.25688, -1.31038], [0.758705, -1.31038, 3.85673]]],
];
const DISK_XI3: number[][][][] = [
  [[[1.47341, -0.302721, -0.0725778], [-0.302721, 1.34588, -0.357085], [-0.0725778, -0.357085, 1.38273]],
   [[-0.302721, 0.966806, -0.316373], [0.966806, -0.194493, 0.75022], [-0.316373, 0.75022, -0.280829]],
   [[-0.0725778, -0.316373, 0.963752], [-0.316373, 0.549654, -0.0566473], [0.963752, -0.0566473, -0.0253987]]],
  [[[-0.302721, 0.966806, -0.316373], [0.966806, -0.194493, 0.75022], [-0.316373, 0.75022, -0.280829]],
   [[1.34588, -0.194493, 0.549654], [-0.194493, 2.22244, -0.3961], [0.549654, -0.3961, 1.74503]],
   [[-0.357085, 0.75022, -0.0566473], [0.75022, -0.3961, 1.52344], [-0.0566473, 1.52344, -0.68151]]],
  [[[-0.0725778, -0.316373, 0.963752], [-0.316373, 0.549654, -0.0566473], [0.963752, -0.0566473, -0.0253987]],
   [[-0.357085, 0.75022, -0.0566473], [0.75022, -0.3961, 1.52344], [-0.0566473, 1.52344, -0.68151]],
   [[1.38273, -0.280829, -0.0253987], [-0.280829, 1.74503, -0.68151], [-0.0253987, -0.68151, 2.87907]]],
];
const MODE_ETA: Record<NlGeometryId, number[]> = { ribbon: RIBBON_ETA_J, triangle: [-0.0878, -0.0299, -0.0223], disk: [-0.0731, -0.0162, -0.00980] };
const MODE_ZETA: Record<NlGeometryId, number[]> = { ribbon: RIBBON_ZETA_J, triangle: [0.544, 0.207, 0.207], disk: [0.851, 0.168, 0.103] };
const MODE_XI3: Record<NlGeometryId, number[][][][]> = { ribbon: RIBBON_XI3, triangle: TRIANGLE_XI3, disk: DISK_XI3 };

// Multimode polarizability for one order, summing nmodes dipole-active modes (Eqs. 27).
// SHG (22) keeps the single-mode form (the multimode ξ² is the triangle Table S3, not embedded here).
export function geometryAlphaMulti(geoId: NlGeometryId, order: NlOrder, D: number, epsBar: number, mat: NlMaterial, E: number, nmodes: number): Cpx {
  const geo = getNlGeometry(geoId);
  if (order === '22') return nlAlphaAt(order, geo, D, epsBar, mat, E);
  const ETA = MODE_ETA[geoId], Z = MODE_ZETA[geoId], XI = MODE_XI3[geoId];
  const w = E / HBAR, n = Math.max(1, Math.min(3, Math.round(nmodes))), Dp = D ** dPow(geo, 2);
  const etaAt = (Ex: number): Cpx => cScale(cMul(Ic, mat.sigma2d(Ex)), 1 / ((Ex / HBAR) * D * epsBar));   // η(ω)=iσ²ᴰ/(ωDε̄)
  const g = (eta: Cpx, j: number): Cpx => cSub(c(1), cDiv(eta, c(ETA[j])));                                // 1 − η/η_j
  const eW = etaAt(E);
  if (order === '11') {
    let tot = c(0, 0);
    for (let j = 0; j < n; j++) tot = cAdd(tot, cDiv(c(Z[j] * Z[j]), g(eW, j)));
    return cMul(cScale(cMul(Ic, mat.sigma2d(E)), Dp / w), tot);                                            // (iD^p/ω) σ¹¹ Σ ζ²/(1−η/η_j)
  }
  if (order === '33') {
    const e3 = etaAt(3 * E), s33 = matSigma33(mat, E), gw: Cpx[] = [], g3: Cpx[] = [];
    for (let j = 0; j < n; j++) { gw.push(g(eW, j)); g3.push(g(e3, j)); }
    let tot = c(0, 0);
    for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) for (let l = 0; l < n; l++) for (let m = 0; m < n; m++)
      tot = cAdd(tot, cDiv(c(XI[j][k][l][m] * Z[j] * Z[k] * Z[l] * Z[m]), cMul(cMul(cMul(g3[j], gw[k]), gw[l]), gw[m])));
    return cMul(cScale(cMul(Ic, s33), Dp / (3 * w)), tot);
  }
  // order 31 (Kerr): numerator (ξ[j,m,l,k] + 2ξ[j,k,l,m]); the last denominator factor uses η(−ω)
  const em = etaAt(-E), s31 = matSigma31(mat, E), gw: Cpx[] = [], gm: Cpx[] = [];
  for (let j = 0; j < n; j++) { gw.push(g(eW, j)); gm.push(g(em, j)); }
  let tot = c(0, 0);
  for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) for (let l = 0; l < n; l++) for (let m = 0; m < n; m++)
    tot = cAdd(tot, cDiv(c((XI[j][m][l][k] + 2 * XI[j][k][l][m]) * Z[j] * Z[k] * Z[l] * Z[m]), cMul(cMul(cMul(gw[j], gw[k]), gw[l]), gm[m])));
  return cMul(cScale(cMul(Ic, s31), Dp / w), tot);
}
// ribbon-specific wrapper (kept for the existing test imports)
export const ribbonAlphaMulti = (order: NlOrder, D: number, epsBar: number, mat: NlMaterial, E: number, nmodes: number): Cpx =>
  geometryAlphaMulti('ribbon', order, D, epsBar, mat, E, nmodes);

// Multimode spectrum over an energy grid → Re/Im arrays.
export function geometrySpectrumMulti(geoId: NlGeometryId, order: NlOrder, D: number, epsBar: number, mat: NlMaterial, Eev: Float64Array, nmodes: number): { re: Float64Array; im: Float64Array } {
  const n = Eev.length, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { const a = geometryAlphaMulti(geoId, order, D, epsBar, mat, Eev[i], nmodes); re[i] = a.re; im[i] = a.im; }
  return { re, im };
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
