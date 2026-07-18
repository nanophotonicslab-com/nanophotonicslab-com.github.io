/**
 * Ultrafast pulse toolbox — material dispersion from Sellmeier equations and
 * Gaussian-pulse propagation bookkeeping.
 *
 *   β₂ = (λ³/2πc²)·d²n/dλ²                       (GVD, s²/m)
 *   β₃ = −(λ⁴/4π²c³)·(3·d²n/dλ² + λ·d³n/dλ³)     (TOD, s³/m)
 *   τ_out = τ_in·√(1 + (4ln2·GDD/τ_in²)²)         (transform-limited Gaussian)
 *
 * Sellmeier n²(λ) = 1 + Σ Bᵢλ²/(λ² − Cᵢ), λ in µm, Cᵢ in µm². Derivatives are
 * taken with 5-point central stencils on the (smooth, analytic) Sellmeier
 * curve — no tabulated-data noise.
 */

export interface SellmeierMaterial {
  label: string;
  ref: string;
  B: number[];
  C: number[];           // µm²
  rangeUm: [number, number];
}

export const SELLMEIER: Record<string, SellmeierMaterial> = {
  'fused-silica': {
    label: 'Fused silica', ref: 'Malitson, JOSA 55, 1205 (1965)',
    B: [0.6961663, 0.4079426, 0.8974794],
    C: [0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2],
    rangeUm: [0.21, 3.7],
  },
  bk7: {
    label: 'N-BK7', ref: 'SCHOTT catalog',
    B: [1.03961212, 0.231792344, 1.01046945],
    C: [0.00600069867, 0.0200179144, 103.560653],
    rangeUm: [0.3, 2.5],
  },
  sapphire: {
    label: 'Sapphire (o)', ref: 'Malitson & Dodge, JOSA 62, 1405 (1972)',
    B: [1.4313493, 0.65054713, 5.3414021],
    C: [0.0726631 ** 2, 0.1193242 ** 2, 18.028251 ** 2],
    rangeUm: [0.2, 5.0],
  },
  caf2: {
    label: 'CaF₂', ref: 'Malitson, Appl. Opt. 2, 1103 (1963)',
    B: [0.5675888, 0.4710914, 3.8484723],
    C: [0.050263605 ** 2, 0.1003909 ** 2, 34.649040 ** 2],
    rangeUm: [0.23, 9.7],
  },
  sf10: {
    label: 'SF10', ref: 'SCHOTT catalog',
    B: [1.62153902, 0.256287842, 1.64447552],
    C: [0.0122241457, 0.0595736775, 147.468793],
    rangeUm: [0.38, 2.5],
  },
  yag: {
    label: 'YAG', ref: 'Zelmon et al., Appl. Opt. 37, 4933 (1998)',
    B: [2.28200, 3.27644],
    C: [0.01185, 282.734],
    rangeUm: [0.4, 5.0],
  },
};

const C_UM_FS = 0.299792458;   // speed of light, µm/fs

export function sellmeierN(mat: SellmeierMaterial, lambdaUm: number): number {
  let n2 = 1;
  for (let i = 0; i < mat.B.length; i++) {
    n2 += (mat.B[i] * lambdaUm * lambdaUm) / (lambdaUm * lambdaUm - mat.C[i]);
  }
  return Math.sqrt(n2);
}

/** d²n/dλ² and d³n/dλ³ (per µm², per µm³) via 5-point central stencils. */
function nDerivatives(mat: SellmeierMaterial, lambdaUm: number): { d2: number; d3: number } {
  const h = 5e-4; // 0.5 nm — Sellmeier is analytic, truncation dominates
  const f = (x: number) => sellmeierN(mat, x);
  const fm2 = f(lambdaUm - 2 * h), fm1 = f(lambdaUm - h), f0 = f(lambdaUm),
    fp1 = f(lambdaUm + h), fp2 = f(lambdaUm + 2 * h);
  const d2 = (-fm2 + 16 * fm1 - 30 * f0 + 16 * fp1 - fp2) / (12 * h * h);
  const d3 = (-fm2 + 2 * fm1 - 2 * fp1 + fp2) / (2 * h * h * h);
  return { d2, d3 };
}

/** Group index n_g = n − λ·dn/dλ. */
export function groupIndex(mat: SellmeierMaterial, lambdaNm: number): number {
  const lam = lambdaNm / 1000, h = 5e-4;
  const d1 = (sellmeierN(mat, lam - 2 * h) - 8 * sellmeierN(mat, lam - h)
    + 8 * sellmeierN(mat, lam + h) - sellmeierN(mat, lam + 2 * h)) / (12 * h);
  return sellmeierN(mat, lam) - lam * d1;
}

/** GVD β₂ in fs²/mm. */
export function gvdFs2PerMm(mat: SellmeierMaterial, lambdaNm: number): number {
  const lam = lambdaNm / 1000; // µm
  const { d2 } = nDerivatives(mat, lam);
  // β₂ = λ³/(2πc²)·n″ with λ in µm, c in µm/fs, n″ in µm⁻² → fs²/µm; ×1000 → fs²/mm
  return (lam ** 3 / (2 * Math.PI * C_UM_FS * C_UM_FS)) * d2 * 1000;
}

/** TOD β₃ in fs³/mm. */
export function todFs3PerMm(mat: SellmeierMaterial, lambdaNm: number): number {
  const lam = lambdaNm / 1000;
  const { d2, d3 } = nDerivatives(mat, lam);
  return (-(lam ** 4) / (4 * Math.PI * Math.PI * C_UM_FS ** 3)) * (3 * d2 + lam * d3) * 1000;
}

export type PulseShape = 'gaussian' | 'sech2';
export const TBP: Record<PulseShape, number> = { gaussian: 0.441, sech2: 0.315 };
/** Autocorrelation FWHM = deconvolution factor × pulse FWHM. */
export const AC_FACTOR: Record<PulseShape, number> = { gaussian: Math.SQRT2, sech2: 1.543 };

/** FWHM after GDD (fs²) for a transform-limited Gaussian input of FWHM τ₀ (fs). */
export function broadenedFwhmFs(tau0Fs: number, gddFs2: number): number {
  const x = (4 * Math.LN2 * gddFs2) / (tau0Fs * tau0Fs);
  return tau0Fs * Math.sqrt(1 + x * x);
}

/** Transform-limited spectral FWHM in nm at center λ (nm). */
export function transformLimitedBandwidthNm(tauFs: number, lambdaNm: number, shape: PulseShape): number {
  const dNuPerFs = TBP[shape] / tauFs;               // 1/fs
  return (lambdaNm * lambdaNm * dNuPerFs) / (C_UM_FS * 1e3); // Δλ = λ²Δν/c, c in nm/fs
}
