/**
 * Shared Mie scattering engine — single sphere, single wavelength.
 * Bohren-Huffman algorithm with logarithmic derivatives (downward recurrence).
 */

export interface MieResult {
  csca: number;   // scattering cross-section (nm²)
  cabs: number;   // absorption cross-section (nm²)
  cext: number;   // extinction cross-section (nm²)
}

export interface MieCoeffs {
  aRe: number; aIm: number;
  bRe: number; bIm: number;
}

/**
 * Compute Mie cross-sections for a homogeneous sphere at a single wavelength.
 *
 * @param nPart  Real part of particle refractive index
 * @param kPart  Imaginary part of particle refractive index (absorption)
 * @param nHost  Host medium refractive index (real)
 * @param radiusNm  Particle radius in nm
 * @param lambdaNm  Wavelength in nm
 * @returns Cross-sections in nm²
 */
export function mieAt(
  nPart: number, kPart: number, nHost: number,
  radiusNm: number, lambdaNm: number,
): MieResult {
  const R = radiusNm * 1e-9;
  const lm = lambdaNm * 1e-9;
  const k0 = 2 * Math.PI * nHost / lm;
  const x = k0 * R;
  if (x < 1e-12) return { csca: 0, cabs: 0, cext: 0 };

  const mRe = nPart / nHost, mIm = kPart / nHost;
  const nmax = Math.max(Math.ceil(x + 4 * Math.cbrt(x) + 2), 3);
  const mAbs = Math.sqrt(mRe * mRe + mIm * mIm);
  const nmx = Math.max(nmax, Math.ceil(mAbs * x)) + 16;

  const mxRe = mRe * x, mxIm = mIm * x;
  const mxMag2 = mxRe * mxRe + mxIm * mxIm;
  const mMag2 = mRe * mRe + mIm * mIm;

  // Downward recurrence for D_n(mx)
  const DRe = new Float64Array(nmx + 2);
  const DIm = new Float64Array(nmx + 2);
  let dRe = 0, dIm = 0;
  for (let n = nmx; n >= 1; n--) {
    const novRe = n * mxRe / mxMag2, novIm = -n * mxIm / mxMag2;
    const tRe = dRe + novRe, tIm = dIm + novIm;
    const tM2 = tRe * tRe + tIm * tIm;
    dRe = novRe - tRe / tM2;
    dIm = novIm + tIm / tM2;
    if (n - 1 <= nmax) { DRe[n - 1] = dRe; DIm[n - 1] = dIm; }
  }

  // Upward recurrence for psi_n(x) and xi_n(x)
  const psiX = new Float64Array(nmax + 2);
  const xiRe = new Float64Array(nmax + 2);
  const xiIm = new Float64Array(nmax + 2);
  const sinX = Math.sin(x), cosX = Math.cos(x);
  psiX[0] = sinX; psiX[1] = sinX / x - cosX;
  for (let n = 1; n < nmax; n++) psiX[n + 1] = ((2 * n + 1) / x) * psiX[n] - psiX[n - 1];
  xiRe[0] = sinX; xiIm[0] = -cosX;
  xiRe[1] = sinX / x - cosX; xiIm[1] = -cosX / x - sinX;
  for (let n = 1; n < nmax; n++) {
    const f = (2 * n + 1) / x;
    xiRe[n + 1] = f * xiRe[n] - xiRe[n - 1];
    xiIm[n + 1] = f * xiIm[n] - xiIm[n - 1];
  }

  let sumSca = 0, sumExt = 0;
  for (let n = 1; n <= nmax; n++) {
    const DomRe = (DRe[n] * mRe + DIm[n] * mIm) / mMag2;
    const DomIm = (DIm[n] * mRe - DRe[n] * mIm) / mMag2;
    const mDRe = mRe * DRe[n] - mIm * DIm[n];
    const mDIm = mRe * DIm[n] + mIm * DRe[n];
    const nx = n / x;
    const ARe = DomRe + nx, AIm = DomIm;
    const BRe = mDRe + nx, BIm = mDIm;

    const anNRe = ARe * psiX[n] - psiX[n - 1];
    const anNIm = AIm * psiX[n];
    const anDRe = (ARe * xiRe[n] - AIm * xiIm[n]) - xiRe[n - 1];
    const anDIm = (ARe * xiIm[n] + AIm * xiRe[n]) - xiIm[n - 1];
    const bnNRe = BRe * psiX[n] - psiX[n - 1];
    const bnNIm = BIm * psiX[n];
    const bnDRe = (BRe * xiRe[n] - BIm * xiIm[n]) - xiRe[n - 1];
    const bnDIm = (BRe * xiIm[n] + BIm * xiRe[n]) - xiIm[n - 1];

    const adM2 = anDRe * anDRe + anDIm * anDIm;
    const aRe = (anNRe * anDRe + anNIm * anDIm) / adM2;
    const aIm = (anNIm * anDRe - anNRe * anDIm) / adM2;
    const bdM2 = bnDRe * bnDRe + bnDIm * bnDIm;
    const bRe = (bnNRe * bnDRe + bnNIm * bnDIm) / bdM2;
    const bIm = (bnNIm * bnDRe - bnNRe * bnDIm) / bdM2;

    const w = 2 * n + 1;
    sumSca += w * (aRe * aRe + aIm * aIm + bRe * bRe + bIm * bIm);
    sumExt += w * (aRe + bRe);
  }

  const pf = (2 * Math.PI / (k0 * k0)) * 1e18; // m² → nm²
  const csca = pf * sumSca;
  const cext = pf * sumExt;
  return { csca, cext, cabs: Math.max(0, cext - csca) };
}

/**
 * Per-order complex Mie coefficients {a_ℓ, b_ℓ} for a homogeneous sphere.
 *
 * ADDED (dipole extension, see the Spheres_dipoles package). Mirrors the
 * validated `mieAt` Bohren-Huffman algorithm but (i) returns the per-order
 * complex reflection coefficients instead of accumulating cross-sections, and
 * (ii) lets the caller force the multipole truncation with `lmaxForce` (the
 * dipole decay-rate figures fix ℓ_max = 20).
 *
 * The returned a_ℓ, b_ℓ are the standard Mie coefficients for non-magnetic
 * media (μ₁ = μ₂), where a_ℓ is the electric (TM) and b_ℓ the magnetic (TE)
 * reflection coefficient. Index ℓ = 1..lmax maps to array[ℓ-1].
 *
 * NOTE ON ACCURACY: this routine keeps the original UPWARD recurrence for
 * ψ_ℓ(x), adequate for cross-sections (dominated by ℓ ≲ x) but it loses
 * precision for ℓ ≫ x. The dipole decay-rate code (dipole-decay.ts) therefore
 * uses its own DOWNWARD-recurrence spherical-Bessel core for the high-ℓ terms
 * and cross-checks against this function in the convergent regime.
 *
 * @returns Array of {aRe,aIm,bRe,bIm}, one entry per order ℓ = 1..lmax.
 */
export function mieCoefficients(
  nPart: number, kPart: number, nHost: number,
  radiusNm: number, lambdaNm: number, lmaxForce?: number,
): MieCoeffs[] {
  const R = radiusNm * 1e-9;
  const lm = lambdaNm * 1e-9;
  const k0 = 2 * Math.PI * nHost / lm;
  const x = k0 * R;
  if (x < 1e-12) return [];

  const mRe = nPart / nHost, mIm = kPart / nHost;
  const nmaxAuto = Math.max(Math.ceil(x + 4 * Math.cbrt(x) + 2), 3);
  const nmax = lmaxForce && lmaxForce > 0 ? lmaxForce : nmaxAuto;
  const mAbs = Math.sqrt(mRe * mRe + mIm * mIm);
  const nmx = Math.max(nmax, Math.ceil(mAbs * x)) + 16;

  const mxRe = mRe * x, mxIm = mIm * x;
  const mxMag2 = mxRe * mxRe + mxIm * mxIm;
  const mMag2 = mRe * mRe + mIm * mIm;

  // Downward recurrence for the logarithmic derivative D_n(mx)
  const DRe = new Float64Array(nmx + 2);
  const DIm = new Float64Array(nmx + 2);
  let dRe = 0, dIm = 0;
  for (let n = nmx; n >= 1; n--) {
    const novRe = n * mxRe / mxMag2, novIm = -n * mxIm / mxMag2;
    const tRe = dRe + novRe, tIm = dIm + novIm;
    const tM2 = tRe * tRe + tIm * tIm;
    dRe = novRe - tRe / tM2;
    dIm = novIm + tIm / tM2;
    if (n - 1 <= nmax) { DRe[n - 1] = dRe; DIm[n - 1] = dIm; }
  }

  // Upward recurrence for psi_n(x) and xi_n(x)
  const psiX = new Float64Array(nmax + 2);
  const xiRe = new Float64Array(nmax + 2);
  const xiIm = new Float64Array(nmax + 2);
  const sinX = Math.sin(x), cosX = Math.cos(x);
  psiX[0] = sinX; psiX[1] = sinX / x - cosX;
  for (let n = 1; n < nmax; n++) psiX[n + 1] = ((2 * n + 1) / x) * psiX[n] - psiX[n - 1];
  xiRe[0] = sinX; xiIm[0] = -cosX;
  xiRe[1] = sinX / x - cosX; xiIm[1] = -cosX / x - sinX;
  for (let n = 1; n < nmax; n++) {
    const f = (2 * n + 1) / x;
    xiRe[n + 1] = f * xiRe[n] - xiRe[n - 1];
    xiIm[n + 1] = f * xiIm[n] - xiIm[n - 1];
  }

  const out: MieCoeffs[] = [];
  for (let n = 1; n <= nmax; n++) {
    const DomRe = (DRe[n] * mRe + DIm[n] * mIm) / mMag2;
    const DomIm = (DIm[n] * mRe - DRe[n] * mIm) / mMag2;
    const mDRe = mRe * DRe[n] - mIm * DIm[n];
    const mDIm = mRe * DIm[n] + mIm * DRe[n];
    const nx = n / x;
    const ARe = DomRe + nx, AIm = DomIm;
    const BRe = mDRe + nx, BIm = mDIm;

    const anNRe = ARe * psiX[n] - psiX[n - 1];
    const anNIm = AIm * psiX[n];
    const anDRe = (ARe * xiRe[n] - AIm * xiIm[n]) - xiRe[n - 1];
    const anDIm = (ARe * xiIm[n] + AIm * xiRe[n]) - xiIm[n - 1];
    const bnNRe = BRe * psiX[n] - psiX[n - 1];
    const bnNIm = BIm * psiX[n];
    const bnDRe = (BRe * xiRe[n] - BIm * xiIm[n]) - xiRe[n - 1];
    const bnDIm = (BRe * xiIm[n] + BIm * xiRe[n]) - xiIm[n - 1];

    const adM2 = anDRe * anDRe + anDIm * anDIm;
    const aRe = (anNRe * anDRe + anNIm * anDIm) / adM2;
    const aIm = (anNIm * anDRe - anNRe * anDIm) / adM2;
    const bdM2 = bnDRe * bnDRe + bnDIm * bnDIm;
    const bRe = (bnNRe * bnDRe + bnNIm * bnDIm) / bdM2;
    const bIm = (bnNIm * bnDRe - bnNRe * bnDIm) / bdM2;

    out.push({ aRe, aIm, bRe, bIm });
  }
  return out;
}

/**
 * Interpolate tabulated (n, k) optical constants at a given wavelength.
 * Data format: [[wavelength_μm, n, k], ...] sorted by wavelength.
 *
 * @param data  Tabulated optical constants (wavelength in μm)
 * @param lambdaNm  Query wavelength in nm
 * @returns [n, k] at the query wavelength
 */
export { interpolateNK } from './materials';

export const HC_EV_NM = 1239.841984;

// ============================================================================
// Spectrum-level engine (extracted verbatim from mie-scattering.astro).
// Bohren-Huffman with logarithmic derivatives; BHCOAT for coated spheres;
// generalized recursive BHCOAT for N-layer particles; far-field angular
// patterns and near-field |E|² maps from the per-order coefficients.
// ============================================================================

const MAX_N = 100;

// Scratch arrays for single-sphere only — core-shell uses bhcoat.
const _psiX = new Float64Array(MAX_N + 2);
const _xiRe = new Float64Array(MAX_N + 2);
const _xiIm = new Float64Array(MAX_N + 2);
const _DRe = new Float64Array(MAX_N + 2);
const _DIm = new Float64Array(MAX_N + 2);

/** Cross-section spectra over a wavelength range, with E/M and multipole decompositions. */
export interface MieSpectrum {
  lambda: Float64Array; energy: Float64Array;
  csca: Float64Array; cext: Float64Array; cabs: Float64Array; qsca: Float64Array;
  csca_e: Float64Array; csca_m: Float64Array;
  cext_e: Float64Array; cext_m: Float64Array;
  csca_mp: [Float64Array, Float64Array, Float64Array, Float64Array];
  cext_mp: [Float64Array, Float64Array, Float64Array, Float64Array];
}

/** One layer of a (multi-)shell particle: dispersion callback + outer radius. */
export interface ShellLayer {
  getDielectric: (lambdaNm: number) => [number, number];
  radiusNm: number;
}

export function computeMie(
  getDielectric: (lambdaNm: number) => [number, number],
  radiusNm: number, nHost: number, lambdaMin: number, lambdaMax: number,
  numPoints = 500,
): MieSpectrum {
  const R = radiusNm * 1e-9;
  const lambda = new Float64Array(numPoints);
  const csca = new Float64Array(numPoints);
  const cext = new Float64Array(numPoints);
  const cabs = new Float64Array(numPoints);
  const qsca = new Float64Array(numPoints);
  const energy = new Float64Array(numPoints);
  const csca_e = new Float64Array(numPoints), csca_m = new Float64Array(numPoints);
  const cext_e = new Float64Array(numPoints), cext_m = new Float64Array(numPoints);
  const csca_mp: [Float64Array, Float64Array, Float64Array, Float64Array] = [new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints)];
  const cext_mp: [Float64Array, Float64Array, Float64Array, Float64Array] = [new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints)];

  for (let i = 0; i < numPoints; i++)
    lambda[i] = lambdaMin + (lambdaMax - lambdaMin) * i / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    const [nP, kP] = getDielectric(lambda[i]);
    const mRe = nP / nHost, mIm = kP / nHost;

    const lm = lambda[i] * 1e-9;
    const k0 = 2 * Math.PI * nHost / lm;
    const x = k0 * R;
    if (x < 1e-12) { csca[i] = 0; cext[i] = 0; cabs[i] = 0; continue; }

    const nmax = Math.max(Math.ceil(x + 4 * Math.cbrt(x) + 2), 3);
    const mAbs = Math.sqrt(mRe * mRe + mIm * mIm);
    const nmx = Math.max(nmax, Math.ceil(mAbs * x)) + 16;

    // Complex mx = m * x
    const mxRe = mRe * x, mxIm = mIm * x;
    const mxMag2 = mxRe * mxRe + mxIm * mxIm;
    const mMag2 = mRe * mRe + mIm * mIm;

    // Downward recurrence for D_n(mx) = psi'_n(mx)/psi_n(mx)
    let dRe = 0, dIm = 0;
    for (let n = nmx; n >= 1; n--) {
      const novRe = n * mxRe / mxMag2, novIm = -n * mxIm / mxMag2;
      const tRe = dRe + novRe, tIm = dIm + novIm;
      const tM2 = tRe * tRe + tIm * tIm;
      dRe = novRe - tRe / tM2;
      dIm = novIm + tIm / tM2;
      if (n - 1 <= nmax) { _DRe[n - 1] = dRe; _DIm[n - 1] = dIm; }
    }

    // Upward recurrence for psi_n(x) and xi_n(x) at real x
    const sinX = Math.sin(x), cosX = Math.cos(x);
    _psiX[0] = sinX; _psiX[1] = sinX / x - cosX;
    for (let n = 1; n < nmax; n++) _psiX[n + 1] = ((2 * n + 1) / x) * _psiX[n] - _psiX[n - 1];
    _xiRe[0] = sinX; _xiIm[0] = -cosX;
    _xiRe[1] = sinX / x - cosX; _xiIm[1] = -cosX / x - sinX;
    for (let n = 1; n < nmax; n++) {
      const f = (2 * n + 1) / x;
      _xiRe[n + 1] = f * _xiRe[n] - _xiRe[n - 1];
      _xiIm[n + 1] = f * _xiIm[n] - _xiIm[n - 1];
    }

    let sumSca = 0, sumExt = 0, sSE = 0, sSM = 0, sEE = 0, sEM = 0;
    const sSMP = [0, 0, 0, 0], sEMP = [0, 0, 0, 0];
    for (let n = 1; n <= nmax; n++) {
      // D_n(mx)/m  (complex / complex)
      const DomRe = (_DRe[n] * mRe + _DIm[n] * mIm) / mMag2;
      const DomIm = (_DIm[n] * mRe - _DRe[n] * mIm) / mMag2;
      // m * D_n(mx)
      const mDRe = mRe * _DRe[n] - mIm * _DIm[n];
      const mDIm = mRe * _DIm[n] + mIm * _DRe[n];

      const nx = n / x;

      // A = D_n/m + n/x (for a_n), B = m*D_n + n/x (for b_n)
      const ARe = DomRe + nx, AIm = DomIm;
      const BRe = mDRe + nx, BIm = mDIm;

      // a_n = (A * psi_n - psi_{n-1}) / (A * xi_n - xi_{n-1})
      const anNRe = ARe * _psiX[n] - _psiX[n - 1];
      const anNIm = AIm * _psiX[n];
      const anDRe = (ARe * _xiRe[n] - AIm * _xiIm[n]) - _xiRe[n - 1];
      const anDIm = (ARe * _xiIm[n] + AIm * _xiRe[n]) - _xiIm[n - 1];

      // b_n = (B * psi_n - psi_{n-1}) / (B * xi_n - xi_{n-1})
      const bnNRe = BRe * _psiX[n] - _psiX[n - 1];
      const bnNIm = BIm * _psiX[n];
      const bnDRe = (BRe * _xiRe[n] - BIm * _xiIm[n]) - _xiRe[n - 1];
      const bnDIm = (BRe * _xiIm[n] + BIm * _xiRe[n]) - _xiIm[n - 1];

      // Complex division a = N/D
      const adM2 = anDRe * anDRe + anDIm * anDIm;
      const aRe = (anNRe * anDRe + anNIm * anDIm) / adM2;
      const aIm = (anNIm * anDRe - anNRe * anDIm) / adM2;
      const bdM2 = bnDRe * bnDRe + bnDIm * bnDIm;
      const bRe = (bnNRe * bnDRe + bnNIm * bnDIm) / bdM2;
      const bIm = (bnNIm * bnDRe - bnNRe * bnDIm) / bdM2;

      const w = 2 * n + 1;
      const sa = w * (aRe * aRe + aIm * aIm), sb = w * (bRe * bRe + bIm * bIm);
      const ea = w * aRe, eb = w * bRe;
      sumSca += sa + sb; sumExt += ea + eb;
      sSE += sa; sSM += sb; sEE += ea; sEM += eb;
      const mi = n <= 3 ? n - 1 : 3;
      sSMP[mi] += sa + sb; sEMP[mi] += ea + eb;
    }

    const pf = (2 * Math.PI / (k0 * k0)) * 1e18;
    csca[i] = pf * sumSca; cext[i] = pf * sumExt;
    cabs[i] = Math.max(0, cext[i] - csca[i]);
    csca_e[i] = pf * sSE; csca_m[i] = pf * sSM;
    cext_e[i] = pf * sEE; cext_m[i] = pf * sEM;
    for (let k = 0; k < 4; k++) { csca_mp[k][i] = pf * sSMP[k]; cext_mp[k][i] = pf * sEMP[k]; }
  }

  const geo = Math.PI * radiusNm * radiusNm;
  for (let i = 0; i < numPoints; i++) { qsca[i] = csca[i] / geo; energy[i] = HC_EV_NM / lambda[i]; }
  return { lambda, energy, csca, cext, cabs, qsca, csca_e, csca_m, cext_e, cext_m, csca_mp, cext_mp };
}

// === Core-Shell Mie (BHCOAT — Bohren-Huffman upward recurrence) ===
//
// Direct translation of the BHCOAT Fortran/MATLAB algorithm.
// Validated against miecoat.dat reference (max |err| < 5e-6).

/** Complex sin(z) */
export function csin(re: number, im: number): [number, number] {
  return [Math.sin(re) * Math.cosh(im), Math.cos(re) * Math.sinh(im)];
}
/** Complex cos(z) */
export function ccos(re: number, im: number): [number, number] {
  return [Math.cos(re) * Math.cosh(im), -Math.sin(re) * Math.sinh(im)];
}
/** Complex cot(z) = cos(z)/sin(z) */
export function ccot(re: number, im: number): [number, number] {
  const [sR, sI] = csin(re, im);
  const [cR, cI] = ccos(re, im);
  const sM = sR * sR + sI * sI;
  return [(cR * sR + cI * sI) / sM, (cI * sR - cR * sI) / sM];
}
/** Complex n/z */
export function cdivN(n: number, zRe: number, zIm: number): [number, number] {
  const m = zRe * zRe + zIm * zIm;
  return [n * zRe / m, -n * zIm / m];
}
/** Complex a * b */
export function cmul(aR: number, aI: number, bR: number, bI: number): [number, number] {
  return [aR * bR - aI * bI, aR * bI + aI * bR];
}
/** Complex a / b */
export function cdiv(aR: number, aI: number, bR: number, bI: number): [number, number] {
  const m = bR * bR + bI * bI;
  return [(aR * bR + aI * bI) / m, (aI * bR - aR * bI) / m];
}

/** Total + decomposed efficiencies at a single size parameter. */
export interface MieDecomp {
  qs: number; qe: number;
  qs_e: number; qs_m: number; qe_e: number; qe_m: number;
  qs_mp: [number, number, number, number]; qe_mp: [number, number, number, number];
}

/**
 * BHCOAT: efficiencies for a coated sphere (core + 1 shell).
 * XX = core size parameter, YY = outer size parameter (both real, in host medium).
 * RF1 = n_core/n_host (complex), RF2 = n_shell/n_host (complex).
 * Returns MieDecomp with total + decomposed efficiencies.
 */
export function bhcoat(XX: number, YY: number,
  RF1Re: number, RF1Im: number, RF2Re: number, RF2Im: number): MieDecomp {
  const DEL = 1e-8;
  const X = XX, Y = YY;
  // REFREL = RF2 / RF1
  const [REFRe, REFIm] = cdiv(RF2Re, RF2Im, RF1Re, RF1Im);
  // Complex size parameters
  const X1Re = RF1Re * X, X1Im = RF1Im * X;
  const X2Re = RF2Re * X, X2Im = RF2Im * X;
  const Y2Re = RF2Re * Y, Y2Im = RF2Im * Y;

  const NSTOP = Math.floor(Y + 4 * Math.pow(Y, 1 / 3) + 2);
  if (NSTOP < 1) {
    return { qs: 0, qe: 0, qs_e: 0, qs_m: 0, qe_e: 0, qe_m: 0,
      qs_mp: [0, 0, 0, 0], qe_mp: [0, 0, 0, 0] };
  }

  // Initial logarithmic derivatives D_0 = cot(z)
  let [D0X1Re, D0X1Im] = ccot(X1Re, X1Im);
  let [D0X2Re, D0X2Im] = ccot(X2Re, X2Im);
  let [D0Y2Re, D0Y2Im] = ccot(Y2Re, Y2Im);

  // Riccati-Bessel at Y (real)
  let PSI0Y = Math.cos(Y), PSI1Y = Math.sin(Y);
  let CHI0Y = -Math.sin(Y), CHI1Y = Math.cos(Y);
  let XI1YRe = PSI1Y, XI1YIm = -CHI1Y;

  // χ at complex X2, Y2
  let [sY2R, sY2I] = csin(Y2Re, Y2Im); let [cY2R, cY2I] = ccos(Y2Re, Y2Im);
  let CHI0Y2Re = -sY2R, CHI0Y2Im = -sY2I, CHI1Y2Re = cY2R, CHI1Y2Im = cY2I;
  let [sX2R, sX2I] = csin(X2Re, X2Im); let [cX2R, cX2I] = ccos(X2Re, X2Im);
  let CHI0X2Re = -sX2R, CHI0X2Im = -sX2I, CHI1X2Re = cX2R, CHI1X2Im = cX2I;

  let QSCA = 0, QEXT = 0, IFLAG = 0;
  let QSE = 0, QSM = 0, QEE = 0, QEM = 0;
  const QSMP: [number, number, number, number] = [0, 0, 0, 0], QEMP: [number, number, number, number] = [0, 0, 0, 0];
  let BRACKRe = 0, BRACKIm = 0, CRACKRe = 0, CRACKIm = 0;
  let AMESS1 = 0, AMESS2 = 0, AMESS3 = 0, AMESS4 = 0;

  const X2M = X2Re * X2Re + X2Im * X2Im;
  const Y2M = Y2Re * Y2Re + Y2Im * Y2Im;

  for (let N = 1; N <= NSTOP; N++) {
    const RN = N;
    // ψ_n, χ_n at Y (real, upward)
    const PSIY = (2 * RN - 1) * PSI1Y / Y - PSI0Y;
    const CHIY = (2 * RN - 1) * CHI1Y / Y - CHI0Y;
    const XIYRe = PSIY, XIYIm = -CHIY;

    // D_n(Y2) upward: D1 = 1/(n/z - D0) - n/z
    const [novY2Re, novY2Im] = cdivN(RN, Y2Re, Y2Im);
    const aY2Re = novY2Re - D0Y2Re, aY2Im = novY2Im - D0Y2Im;
    const [invAY2Re, invAY2Im] = cdiv(1, 0, aY2Re, aY2Im);
    const D1Y2Re = invAY2Re - novY2Re, D1Y2Im = invAY2Im - novY2Im;

    if (IFLAG === 0) {
      // D_n(X1)
      const [novX1Re, novX1Im] = cdivN(RN, X1Re, X1Im);
      const [invAX1Re, invAX1Im] = cdiv(1, 0, novX1Re - D0X1Re, novX1Im - D0X1Im);
      const D1X1Re = invAX1Re - novX1Re, D1X1Im = invAX1Im - novX1Im;
      // D_n(X2)
      const [novX2Re, novX2Im] = cdivN(RN, X2Re, X2Im);
      const [invAX2Re, invAX2Im] = cdiv(1, 0, novX2Re - D0X2Re, novX2Im - D0X2Im);
      const D1X2Re = invAX2Re - novX2Re, D1X2Im = invAX2Im - novX2Im;

      // χ_n at X2, Y2 (upward)
      const fac = 2 * RN - 1;
      const [fX2Re, fX2Im] = cdivN(fac, X2Re, X2Im);
      const [CHIX2Re, CHIX2Im] = [fX2Re * CHI1X2Re - fX2Im * CHI1X2Im - CHI0X2Re, fX2Re * CHI1X2Im + fX2Im * CHI1X2Re - CHI0X2Im];
      const [fY2Re, fY2Im] = cdivN(fac, Y2Re, Y2Im);
      const [CHIY2Re, CHIY2Im] = [fY2Re * CHI1Y2Re - fY2Im * CHI1Y2Im - CHI0Y2Re, fY2Re * CHI1Y2Im + fY2Im * CHI1Y2Re - CHI0Y2Im];

      // χ'_n = χ_{n-1} - n·χ_n/z
      const [ncX2Re, ncX2Im] = cdivN(RN, X2Re, X2Im);
      const CHIPX2Re = CHI1X2Re - (ncX2Re * CHIX2Re - ncX2Im * CHIX2Im);
      const CHIPX2Im = CHI1X2Im - (ncX2Re * CHIX2Im + ncX2Im * CHIX2Re);
      const [ncY2Re, ncY2Im] = cdivN(RN, Y2Re, Y2Im);
      const CHIPY2Re = CHI1Y2Re - (ncY2Re * CHIY2Re - ncY2Im * CHIY2Im);
      const CHIPY2Im = CHI1Y2Im - (ncY2Re * CHIY2Im + ncY2Im * CHIY2Re);

      // ANCAP = (REF·D1X1 - D1X2) / ((REF·D1X1·χX2 - χ'X2) · (χX2·D1X2 - χ'X2))
      const [rD1Re, rD1Im] = cmul(REFRe, REFIm, D1X1Re, D1X1Im);
      const [rDcRe, rDcIm] = cmul(rD1Re, rD1Im, CHIX2Re, CHIX2Im);
      const [cDRe, cDIm] = cmul(CHIX2Re, CHIX2Im, D1X2Re, D1X2Im);
      const den1Re = rDcRe - CHIPX2Re, den1Im = rDcIm - CHIPX2Im;
      const den2Re = cDRe - CHIPX2Re, den2Im = cDIm - CHIPX2Im;
      const [ddRe, ddIm] = cmul(den1Re, den1Im, den2Re, den2Im);
      const [ANCAPRe, ANCAPIm] = cdiv(rD1Re - D1X2Re, rD1Im - D1X2Im, ddRe, ddIm);

      // BRACK = ANCAP · (χY2·D1Y2 - χ'Y2)
      const [cdyRe, cdyIm] = cmul(CHIY2Re, CHIY2Im, D1Y2Re, D1Y2Im);
      const brArgRe = cdyRe - CHIPY2Re, brArgIm = cdyIm - CHIPY2Im;
      [BRACKRe, BRACKIm] = cmul(ANCAPRe, ANCAPIm, brArgRe, brArgIm);

      // BNCAP = (REF·D1X2 - D1X1) / ((REF·χ'X2 - D1X1·χX2) · (χX2·D1X2 - χ'X2))
      const [rD2Re, rD2Im] = cmul(REFRe, REFIm, D1X2Re, D1X2Im);
      const [rCpRe, rCpIm] = cmul(REFRe, REFIm, CHIPX2Re, CHIPX2Im);
      const [d1cRe, d1cIm] = cmul(D1X1Re, D1X1Im, CHIX2Re, CHIX2Im);
      const den1bRe = rCpRe - d1cRe, den1bIm = rCpIm - d1cIm;
      const [ddbRe, ddbIm] = cmul(den1bRe, den1bIm, den2Re, den2Im);
      const [BNCAPRe, BNCAPIm] = cdiv(rD2Re - D1X1Re, rD2Im - D1X1Im, ddbRe, ddbIm);
      [CRACKRe, CRACKIm] = cmul(BNCAPRe, BNCAPIm, brArgRe, brArgIm);

      // Convergence test magnitudes
      const [bc1R, bc1I] = cmul(BRACKRe, BRACKIm, CHIPY2Re, CHIPY2Im);
      AMESS1 = Math.sqrt(bc1R * bc1R + bc1I * bc1I);
      const [bc2R, bc2I] = cmul(BRACKRe, BRACKIm, CHIY2Re, CHIY2Im);
      AMESS2 = Math.sqrt(bc2R * bc2R + bc2I * bc2I);
      const [cc1R, cc1I] = cmul(CRACKRe, CRACKIm, CHIPY2Re, CHIPY2Im);
      AMESS3 = Math.sqrt(cc1R * cc1R + cc1I * cc1I);
      const [cc2R, cc2I] = cmul(CRACKRe, CRACKIm, CHIY2Re, CHIY2Im);
      AMESS4 = Math.sqrt(cc2R * cc2R + cc2I * cc2I);

      // Update χ recurrences
      CHI0X2Re = CHI1X2Re; CHI0X2Im = CHI1X2Im; CHI1X2Re = CHIX2Re; CHI1X2Im = CHIX2Im;
      CHI0Y2Re = CHI1Y2Re; CHI0Y2Im = CHI1Y2Im; CHI1Y2Re = CHIY2Re; CHI1Y2Im = CHIY2Im;
      D0X1Re = D1X1Re; D0X1Im = D1X1Im;
      D0X2Re = D1X2Re; D0X2Im = D1X2Im;
    }

    // Convergence check
    const D1Y2abs = Math.sqrt(D1Y2Re * D1Y2Re + D1Y2Im * D1Y2Im);
    if (AMESS1 < DEL * D1Y2abs && AMESS2 < DEL && AMESS3 < DEL * D1Y2abs && AMESS4 < DEL) {
      BRACKRe = 0; BRACKIm = 0; CRACKRe = 0; CRACKIm = 0; IFLAG = 1;
    } else { IFLAG = 0; }

    D0Y2Re = D1Y2Re; D0Y2Im = D1Y2Im;

    // Effective derivatives DNBAR, GNBAR
    let DNBARRe: number, DNBARIm: number, GNBARRe: number, GNBARIm: number;
    if (IFLAG === 0) {
      // CHIPY2 = CHI_{n-1}(Y2) - n·CHI_n(Y2)/Y2  (using updated recurrence values)
      const [ncRe, ncIm] = cdivN(RN, Y2Re, Y2Im);
      const CHIPY2Re = CHI0Y2Re - (ncRe * CHI1Y2Re - ncIm * CHI1Y2Im);
      const CHIPY2Im = CHI0Y2Im - (ncRe * CHI1Y2Im + ncIm * CHI1Y2Re);
      // DNBAR = (D1Y2 - BRACK·χ'Y2) / (1 - BRACK·χY2)
      const [bcpRe, bcpIm] = cmul(BRACKRe, BRACKIm, CHIPY2Re, CHIPY2Im);
      const [bcyRe, bcyIm] = cmul(BRACKRe, BRACKIm, CHI1Y2Re, CHI1Y2Im);
      [DNBARRe, DNBARIm] = cdiv(D1Y2Re - bcpRe, D1Y2Im - bcpIm, 1 - bcyRe, -bcyIm);
      // GNBAR = (D1Y2 - CRACK·χ'Y2) / (1 - CRACK·χY2)
      const [ccpRe, ccpIm] = cmul(CRACKRe, CRACKIm, CHIPY2Re, CHIPY2Im);
      const [ccyRe, ccyIm] = cmul(CRACKRe, CRACKIm, CHI1Y2Re, CHI1Y2Im);
      [GNBARRe, GNBARIm] = cdiv(D1Y2Re - ccpRe, D1Y2Im - ccpIm, 1 - ccyRe, -ccyIm);
    } else {
      DNBARRe = D1Y2Re; DNBARIm = D1Y2Im;
      GNBARRe = D1Y2Re; GNBARIm = D1Y2Im;
    }

    // a_n = (DNBAR/RF2 + n/Y)·ψ(Y) - ψ_{n-1}(Y)  /  (DNBAR/RF2 + n/Y)·ξ(Y) - ξ_{n-1}(Y)
    const [dbrRe, dbrIm] = cdiv(DNBARRe, DNBARIm, RF2Re, RF2Im);
    const ARe = dbrRe + RN / Y, AIm = dbrIm;
    const anNRe = ARe * PSIY - PSI1Y, anNIm = AIm * PSIY;
    const anDRe = (ARe * XIYRe - AIm * XIYIm) - XI1YRe;
    const anDIm = (ARe * XIYIm + AIm * XIYRe) - XI1YIm;
    const [aRe, aIm] = cdiv(anNRe, anNIm, anDRe, anDIm);

    // b_n = (RF2·GNBAR + n/Y)·ψ(Y) - ψ_{n-1}(Y)  /  (RF2·GNBAR + n/Y)·ξ(Y) - ξ_{n-1}(Y)
    const [rgRe, rgIm] = cmul(RF2Re, RF2Im, GNBARRe, GNBARIm);
    const BRe = rgRe + RN / Y, BIm = rgIm;
    const bnNRe = BRe * PSIY - PSI1Y, bnNIm = BIm * PSIY;
    const bnDRe = (BRe * XIYRe - BIm * XIYIm) - XI1YRe;
    const bnDIm = (BRe * XIYIm + BIm * XIYRe) - XI1YIm;
    const [bRe, bIm] = cdiv(bnNRe, bnNIm, bnDRe, bnDIm);

    const w = 2 * RN + 1;
    const sa = w * (aRe * aRe + aIm * aIm), sb = w * (bRe * bRe + bIm * bIm);
    const ea = w * aRe, eb = w * bRe;
    QSCA += sa + sb; QEXT += ea + eb;
    QSE += sa; QSM += sb; QEE += ea; QEM += eb;
    const mi = N <= 3 ? N - 1 : 3;
    QSMP[mi] += sa + sb; QEMP[mi] += ea + eb;

    PSI0Y = PSI1Y; PSI1Y = PSIY;
    CHI0Y = CHI1Y; CHI1Y = CHIY;
    XI1YRe = PSI1Y; XI1YIm = -CHI1Y;
  }
  const f = 2 / (Y * Y);
  return { qs: f * QSCA, qe: f * QEXT, qs_e: f * QSE, qs_m: f * QSM, qe_e: f * QEE, qe_m: f * QEM,
    qs_mp: [f * QSMP[0], f * QSMP[1], f * QSMP[2], f * QSMP[3]],
    qe_mp: [f * QEMP[0], f * QEMP[1], f * QEMP[2], f * QEMP[3]] };
}

/**
 * Multi-shell Mie scattering — generalized BHCOAT.
 * Recursive upward-recurrence matching from core outward, shell by shell.
 * layers[0] = core, layers[1..N-1] = shells (inside-out).
 * For N=2 delegates to the validated bhcoat fast path.
 */
export function computeMieMultiShell(
  layers: ShellLayer[],
  nHost: number, lambdaMin: number, lambdaMax: number,
  numPoints = 500,
): MieSpectrum {
  const N = layers.length;
  const outerRadiusNm = layers[N - 1].radiusNm;
  const lambda = new Float64Array(numPoints);
  const csca = new Float64Array(numPoints);
  const cext = new Float64Array(numPoints);
  const cabs = new Float64Array(numPoints);
  const qsca = new Float64Array(numPoints);
  const energy = new Float64Array(numPoints);
  const csca_e = new Float64Array(numPoints), csca_m = new Float64Array(numPoints);
  const cext_e = new Float64Array(numPoints), cext_m = new Float64Array(numPoints);
  const csca_mp: [Float64Array, Float64Array, Float64Array, Float64Array] = [new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints)];
  const cext_mp: [Float64Array, Float64Array, Float64Array, Float64Array] = [new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints), new Float64Array(numPoints)];

  for (let i = 0; i < numPoints; i++)
    lambda[i] = lambdaMin + (lambdaMax - lambdaMin) * i / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    const lm = lambda[i] * 1e-9;
    const k0 = 2 * Math.PI * nHost / lm;
    const Y = k0 * outerRadiusNm * 1e-9;
    if (Y < 1e-12) { csca[i] = 0; cext[i] = 0; cabs[i] = 0; continue; }

    // Refractive indices relative to host
    const mRe: number[] = [], mIm: number[] = [];
    for (let j = 0; j < N; j++) {
      const [nj, kj] = layers[j].getDielectric(lambda[i]);
      mRe.push(nj / nHost); mIm.push(kj / nHost);
    }
    const radii = layers.map(l => l.radiusNm * 1e-9);

    // N=2 fast path: use validated bhcoat
    if (N === 2) {
      const XX = k0 * radii[0];
      const bh = bhcoat(XX, Y, mRe[0], mIm[0], mRe[1], mIm[1]);
      const geo = Math.PI * outerRadiusNm * outerRadiusNm;
      csca[i] = bh.qs * geo; cext[i] = bh.qe * geo;
      cabs[i] = Math.max(0, cext[i] - csca[i]);
      csca_e[i] = bh.qs_e * geo; csca_m[i] = bh.qs_m * geo;
      cext_e[i] = bh.qe_e * geo; cext_m[i] = bh.qe_m * geo;
      for (let k = 0; k < 4; k++) { csca_mp[k][i] = bh.qs_mp[k] * geo; cext_mp[k][i] = bh.qe_mp[k] * geo; }
      continue;
    }

    // --- General N-layer recursive BHCOAT ---
    const NSTOP = Math.floor(Y + 4 * Math.pow(Y, 1 / 3) + 2);

    // Size parameters: S[j] = k0 * r[j] (in host medium)
    const S = radii.map(r => k0 * r);

    // Per-shell recurrence state: D_prev and chi_{n-1}, chi_{n-2}
    // at inner boundary (m[j]*S[j-1]) and outer boundary (m[j]*S[j])
    interface ShellState {
      zInRe: number; zInIm: number; zOutRe: number; zOutIm: number;
      D0InRe: number; D0InIm: number; D0OutRe: number; D0OutIm: number;
      chi0InRe: number; chi0InIm: number; chi1InRe: number; chi1InIm: number;
      chi0OutRe: number; chi0OutIm: number; chi1OutRe: number; chi1OutIm: number;
    }
    // Core D recurrence
    const zCRe = mRe[0] * S[0], zCIm = mIm[0] * S[0];
    let [D0CRe, D0CIm] = ccot(zCRe, zCIm);

    // Shell states
    const ss: ShellState[] = [];
    for (let j = 1; j < N; j++) {
      const zIR = mRe[j] * S[j-1], zII = mIm[j] * S[j-1];
      const zOR = mRe[j] * S[j],   zOI = mIm[j] * S[j];
      const [d0IR, d0II] = ccot(zIR, zII);
      const [d0OR, d0OI] = ccot(zOR, zOI);
      const [sIR, sII] = csin(zIR, zII); const [cIR, cII] = ccos(zIR, zII);
      const [sOR, sOI] = csin(zOR, zOI); const [cOR, cOI] = ccos(zOR, zOI);
      ss.push({
        zInRe: zIR, zInIm: zII, zOutRe: zOR, zOutIm: zOI,
        D0InRe: d0IR, D0InIm: d0II, D0OutRe: d0OR, D0OutIm: d0OI,
        chi0InRe: -sIR, chi0InIm: -sII, chi1InRe: cIR, chi1InIm: cII,
        chi0OutRe: -sOR, chi0OutIm: -sOI, chi1OutRe: cOR, chi1OutIm: cOI,
      });
    }

    // Outer real Bessel recurrence
    let PSI0Y = Math.cos(Y), PSI1Y = Math.sin(Y);
    let CHI0Y = -Math.sin(Y), CHI1Y = Math.cos(Y);
    let XI1YRe = PSI1Y, XI1YIm = -CHI1Y;

    let QSCA = 0, QEXT = 0, QSE = 0, QSM = 0, QEE = 0, QEM = 0;
    const QSMP = [0, 0, 0, 0], QEMP = [0, 0, 0, 0];
    for (let n = 1; n <= NSTOP; n++) {
      const RN = n;
      const PSIY = (2 * RN - 1) * PSI1Y / Y - PSI0Y;
      const CHIY = (2 * RN - 1) * CHI1Y / Y - CHI0Y;
      const XIYRe = PSIY, XIYIm = -CHIY;

      // Core D_n upward
      const [novCRe, novCIm] = cdivN(RN, zCRe, zCIm);
      const [invCRe, invCIm] = cdiv(1, 0, novCRe - D0CRe, novCIm - D0CIm);
      const DCRe = invCRe - novCRe, DCIm = invCIm - novCIm;

      // Initialize effective D from core
      let DNBARRe = DCRe, DNBARIm = DCIm;
      let GNBARRe = DCRe, GNBARIm = DCIm;

      // Process each shell (inside-out)
      for (let j = 0; j < ss.length; j++) {
        const s = ss[j];
        const jLayer = j + 1;

        // D_n at inner boundary
        const [novIRe, novIIm] = cdivN(RN, s.zInRe, s.zInIm);
        const [invIRe, invIIm] = cdiv(1, 0, novIRe - s.D0InRe, novIIm - s.D0InIm);
        const D1IRe = invIRe - novIRe, D1IIm = invIIm - novIIm;

        // D_n at outer boundary
        const [novORe, novOIm] = cdivN(RN, s.zOutRe, s.zOutIm);
        const [invORe, invOIm] = cdiv(1, 0, novORe - s.D0OutRe, novOIm - s.D0OutIm);
        const D1ORe = invORe - novORe, D1OIm = invOIm - novOIm;

        // χ_n at inner boundary (upward)
        const fac = 2 * RN - 1;
        const [fIRe, fIIm] = cdivN(fac, s.zInRe, s.zInIm);
        const CHIIRe = fIRe * s.chi1InRe - fIIm * s.chi1InIm - s.chi0InRe;
        const CHIIIm = fIRe * s.chi1InIm + fIIm * s.chi1InRe - s.chi0InIm;
        // χ'_n at inner
        const [ncIRe, ncIIm] = cdivN(RN, s.zInRe, s.zInIm);
        const CHIPIRe = s.chi1InRe - (ncIRe * CHIIRe - ncIIm * CHIIIm);
        const CHIPIIm = s.chi1InIm - (ncIRe * CHIIIm + ncIIm * CHIIRe);

        // χ_n at outer boundary
        const [fORe, fOIm] = cdivN(fac, s.zOutRe, s.zOutIm);
        const CHIORe = fORe * s.chi1OutRe - fOIm * s.chi1OutIm - s.chi0OutRe;
        const CHIOIm = fORe * s.chi1OutIm + fOIm * s.chi1OutRe - s.chi0OutIm;
        // χ'_n at outer
        const [ncORe, ncOIm] = cdivN(RN, s.zOutRe, s.zOutIm);
        const CHIPORe = s.chi1OutRe - (ncORe * CHIORe - ncOIm * CHIOIm);
        const CHIPOIm = s.chi1OutIm - (ncORe * CHIOIm + ncOIm * CHIORe);

        // REFREL = m[jLayer] / m[jLayer-1]
        const [REFRe, REFIm] = cdiv(mRe[jLayer], mIm[jLayer], mRe[jLayer - 1], mIm[jLayer - 1]);

        // --- TM: ANCAP uses DNBAR as inner D ---
        const [rDRe, rDIm] = cmul(REFRe, REFIm, DNBARRe, DNBARIm);
        const [rDcRe, rDcIm] = cmul(rDRe, rDIm, CHIIRe, CHIIIm);
        const [cDaRe, cDaIm] = cmul(CHIIRe, CHIIIm, D1IRe, D1IIm);
        const den1aRe = rDcRe - CHIPIRe, den1aIm = rDcIm - CHIPIIm;
        const den2Re = cDaRe - CHIPIRe, den2Im = cDaIm - CHIPIIm;
        const [ddaRe, ddaIm] = cmul(den1aRe, den1aIm, den2Re, den2Im);
        const [ANCRe, ANCIm] = cdiv(rDRe - D1IRe, rDIm - D1IIm, ddaRe, ddaIm);
        // BRACK
        const [cdoRe, cdoIm] = cmul(CHIORe, CHIOIm, D1ORe, D1OIm);
        const brRe = cdoRe - CHIPORe, brIm = cdoIm - CHIPOIm;
        const [BRKRe, BRKIm] = cmul(ANCRe, ANCIm, brRe, brIm);

        // --- TE: BNCAP uses GNBAR as inner D ---
        const [rD2Re, rD2Im] = cmul(REFRe, REFIm, D1IRe, D1IIm);
        const [rCpRe, rCpIm] = cmul(REFRe, REFIm, CHIPIRe, CHIPIIm);
        const [gCRe, gCIm] = cmul(GNBARRe, GNBARIm, CHIIRe, CHIIIm);
        const den1bRe = rCpRe - gCRe, den1bIm = rCpIm - gCIm;
        const [ddbRe, ddbIm] = cmul(den1bRe, den1bIm, den2Re, den2Im);
        const [BNCRe, BNCIm] = cdiv(rD2Re - GNBARRe, rD2Im - GNBARIm, ddbRe, ddbIm);
        const [CRKRe, CRKIm] = cmul(BNCRe, BNCIm, brRe, brIm);

        // Propagate DNBAR, GNBAR to outer boundary
        const [bcpRe, bcpIm] = cmul(BRKRe, BRKIm, CHIPORe, CHIPOIm);
        const [bcyRe, bcyIm] = cmul(BRKRe, BRKIm, CHIORe, CHIOIm);
        [DNBARRe, DNBARIm] = cdiv(D1ORe - bcpRe, D1OIm - bcpIm, 1 - bcyRe, -bcyIm);
        const [ccpRe, ccpIm] = cmul(CRKRe, CRKIm, CHIPORe, CHIPOIm);
        const [ccyRe, ccyIm] = cmul(CRKRe, CRKIm, CHIORe, CHIOIm);
        [GNBARRe, GNBARIm] = cdiv(D1ORe - ccpRe, D1OIm - ccpIm, 1 - ccyRe, -ccyIm);

        // Update recurrence state
        s.D0InRe = D1IRe; s.D0InIm = D1IIm;
        s.D0OutRe = D1ORe; s.D0OutIm = D1OIm;
        s.chi0InRe = s.chi1InRe; s.chi0InIm = s.chi1InIm;
        s.chi1InRe = CHIIRe; s.chi1InIm = CHIIIm;
        s.chi0OutRe = s.chi1OutRe; s.chi0OutIm = s.chi1OutIm;
        s.chi1OutRe = CHIORe; s.chi1OutIm = CHIOIm;
      }
      D0CRe = DCRe; D0CIm = DCIm;

      // a_n, b_n from final DNBAR/GNBAR and outermost m
      const [dbrRe, dbrIm] = cdiv(DNBARRe, DNBARIm, mRe[N - 1], mIm[N - 1]);
      const ARe = dbrRe + RN / Y, AIm = dbrIm;
      const anNRe = ARe * PSIY - PSI1Y, anNIm = AIm * PSIY;
      const anDRe = (ARe * XIYRe - AIm * XIYIm) - XI1YRe;
      const anDIm = (ARe * XIYIm + AIm * XIYRe) - XI1YIm;
      const [aRe, aIm] = cdiv(anNRe, anNIm, anDRe, anDIm);

      const [rgRe, rgIm] = cmul(mRe[N - 1], mIm[N - 1], GNBARRe, GNBARIm);
      const BRe = rgRe + RN / Y, BIm = rgIm;
      const bnNRe = BRe * PSIY - PSI1Y, bnNIm = BIm * PSIY;
      const bnDRe = (BRe * XIYRe - BIm * XIYIm) - XI1YRe;
      const bnDIm = (BRe * XIYIm + BIm * XIYRe) - XI1YIm;
      const [bRe, bIm] = cdiv(bnNRe, bnNIm, bnDRe, bnDIm);

      const w = 2 * RN + 1;
      const sa = w * (aRe * aRe + aIm * aIm), sb = w * (bRe * bRe + bIm * bIm);
      const ea = w * aRe, eb = w * bRe;
      QSCA += sa + sb; QEXT += ea + eb;
      QSE += sa; QSM += sb; QEE += ea; QEM += eb;
      const mi = n <= 3 ? n - 1 : 3;
      QSMP[mi] += sa + sb; QEMP[mi] += ea + eb;

      PSI0Y = PSI1Y; PSI1Y = PSIY;
      CHI0Y = CHI1Y; CHI1Y = CHIY;
      XI1YRe = PSI1Y; XI1YIm = -CHI1Y;
    }

    const ff = 2 / (Y * Y);
    const geo = Math.PI * outerRadiusNm * outerRadiusNm;
    csca[i] = ff * QSCA * geo; cext[i] = ff * QEXT * geo;
    cabs[i] = Math.max(0, cext[i] - csca[i]);
    csca_e[i] = ff * QSE * geo; csca_m[i] = ff * QSM * geo;
    cext_e[i] = ff * QEE * geo; cext_m[i] = ff * QEM * geo;
    for (let k = 0; k < 4; k++) { csca_mp[k][i] = ff * QSMP[k] * geo; cext_mp[k][i] = ff * QEMP[k] * geo; }
  }

  const geo = Math.PI * outerRadiusNm * outerRadiusNm;
  for (let i = 0; i < numPoints; i++) { qsca[i] = csca[i] / geo; energy[i] = HC_EV_NM / lambda[i]; }
  return { lambda, energy, csca, cext, cabs, qsca, csca_e, csca_m, cext_e, cext_m, csca_mp, cext_mp };
}

// === Mie Coefficients (single wavelength, for angular patterns) ===

/** Per-order coefficient arrays at a single wavelength (a/b scattered; c/d internal). */
export interface SphereCoeffs {
  aRe: Float64Array; aIm: Float64Array;
  bRe: Float64Array; bIm: Float64Array;
  cRe?: Float64Array; cIm?: Float64Array;
  dRe?: Float64Array; dIm?: Float64Array;
  mRe?: number; mIm?: number;
  nmax: number; x: number;
}

export function computeSphereCoeffs(
  getDiel: (l: number) => [number, number], radiusNm: number, nHost: number, lambdaNm: number,
): SphereCoeffs {
  const [nP, kP] = getDiel(lambdaNm);
  const mRe = nP / nHost, mIm = kP / nHost;
  const R = radiusNm * 1e-9, lm = lambdaNm * 1e-9;
  const k0 = 2 * Math.PI * nHost / lm, x = k0 * R;
  if (x < 1e-12) return { aRe: new Float64Array(0), aIm: new Float64Array(0), bRe: new Float64Array(0), bIm: new Float64Array(0), nmax: 0, x };
  const nmax = Math.max(Math.ceil(x + 4 * Math.cbrt(x) + 2), 3);
  const mAbs = Math.sqrt(mRe * mRe + mIm * mIm);
  const nmx = Math.max(nmax, Math.ceil(mAbs * x)) + 16;
  const mxRe = mRe * x, mxIm = mIm * x, mxMag2 = mxRe * mxRe + mxIm * mxIm, mMag2 = mRe * mRe + mIm * mIm;
  let dRe = 0, dIm = 0;
  for (let n = nmx; n >= 1; n--) {
    const novRe = n * mxRe / mxMag2, novIm = -n * mxIm / mxMag2;
    const tRe = dRe + novRe, tIm = dIm + novIm, tM2 = tRe * tRe + tIm * tIm;
    dRe = novRe - tRe / tM2; dIm = novIm + tIm / tM2;
    if (n - 1 <= nmax) { _DRe[n - 1] = dRe; _DIm[n - 1] = dIm; }
  }
  const sinX = Math.sin(x), cosX = Math.cos(x);
  _psiX[0] = sinX; _psiX[1] = sinX / x - cosX;
  for (let n = 1; n < nmax; n++) _psiX[n + 1] = ((2 * n + 1) / x) * _psiX[n] - _psiX[n - 1];
  _xiRe[0] = sinX; _xiIm[0] = -cosX; _xiRe[1] = sinX / x - cosX; _xiIm[1] = -cosX / x - sinX;
  for (let n = 1; n < nmax; n++) { const f = (2 * n + 1) / x; _xiRe[n + 1] = f * _xiRe[n] - _xiRe[n - 1]; _xiIm[n + 1] = f * _xiIm[n] - _xiIm[n - 1]; }
  // ψ_n(mx) for complex mx — needed for internal field coefficients
  const [mxSinRe, mxSinIm] = csin(mxRe, mxIm);
  const [mxCosRe, mxCosIm] = ccos(mxRe, mxIm);
  const _psiMxRe = new Float64Array(nmax + 2), _psiMxIm = new Float64Array(nmax + 2);
  _psiMxRe[0] = mxSinRe; _psiMxIm[0] = mxSinIm;
  const [sovRe, sovIm] = cdiv(mxSinRe, mxSinIm, mxRe, mxIm);
  _psiMxRe[1] = sovRe - mxCosRe; _psiMxIm[1] = sovIm - mxCosIm;
  for (let n = 1; n < nmax + 1; n++) {
    const [fRe, fIm] = cdivN(2 * n + 1, mxRe, mxIm);
    _psiMxRe[n + 1] = fRe * _psiMxRe[n] - fIm * _psiMxIm[n] - _psiMxRe[n - 1];
    _psiMxIm[n + 1] = fRe * _psiMxIm[n] + fIm * _psiMxRe[n] - _psiMxIm[n - 1];
  }
  const aR = new Float64Array(nmax), aI = new Float64Array(nmax), bR = new Float64Array(nmax), bI = new Float64Array(nmax);
  const cR = new Float64Array(nmax), cI = new Float64Array(nmax), dR2 = new Float64Array(nmax), dI2 = new Float64Array(nmax);
  for (let n = 1; n <= nmax; n++) {
    const DomRe = (_DRe[n]*mRe+_DIm[n]*mIm)/mMag2, DomIm = (_DIm[n]*mRe-_DRe[n]*mIm)/mMag2;
    const mDRe = mRe*_DRe[n]-mIm*_DIm[n], mDIm = mRe*_DIm[n]+mIm*_DRe[n];
    const nx = n/x, ARe = DomRe+nx, AIm = DomIm, BRe = mDRe+nx, BIm = mDIm;
    const anNRe = ARe*_psiX[n]-_psiX[n-1], anNIm = AIm*_psiX[n];
    const anDRe = (ARe*_xiRe[n]-AIm*_xiIm[n])-_xiRe[n-1], anDIm = (ARe*_xiIm[n]+AIm*_xiRe[n])-_xiIm[n-1];
    const bnNRe = BRe*_psiX[n]-_psiX[n-1], bnNIm = BIm*_psiX[n];
    const bnDRe = (BRe*_xiRe[n]-BIm*_xiIm[n])-_xiRe[n-1], bnDIm = (BRe*_xiIm[n]+BIm*_xiRe[n])-_xiIm[n-1];
    const adM2 = anDRe*anDRe+anDIm*anDIm; aR[n-1] = (anNRe*anDRe+anNIm*anDIm)/adM2; aI[n-1] = (anNIm*anDRe-anNRe*anDIm)/adM2;
    const bdM2 = bnDRe*bnDRe+bnDIm*bnDIm; bR[n-1] = (bnNRe*bnDRe+bnNIm*bnDIm)/bdM2; bI[n-1] = (bnNIm*bnDRe-bnNRe*bnDIm)/bdM2;
    // Internal field coefficients (B&H 4.53):
    // c_n = -i / (m * ψ_n(mx) * anD), d_n = -m*i / (ψ_n(mx) * bnD)
    const [mpRe, mpIm] = cmul(mRe, mIm, _psiMxRe[n], _psiMxIm[n]);
    const [dcRe, dcIm] = cmul(mpRe, mpIm, anDRe, anDIm);
    const dcM2 = dcRe * dcRe + dcIm * dcIm;
    cR[n - 1] = -dcIm / dcM2; cI[n - 1] = -dcRe / dcM2;
    const [ddRe, ddIm] = cmul(_psiMxRe[n], _psiMxIm[n], bnDRe, bnDIm);
    const ddM2 = ddRe * ddRe + ddIm * ddIm;
    dR2[n - 1] = (mIm * ddRe - mRe * ddIm) / ddM2;
    dI2[n - 1] = (-mRe * ddRe - mIm * ddIm) / ddM2;
  }
  return { aRe: aR, aIm: aI, bRe: bR, bIm: bI, cRe: cR, cIm: cI, dRe: dR2, dIm: dI2, mRe, mIm, nmax, x };
}

export function computeShellCoeffsAtLambda(layers: ShellLayer[], nHost: number, lambdaNm: number): SphereCoeffs {
  const NL = layers.length, outerR = layers[NL-1].radiusNm;
  const lm = lambdaNm*1e-9, k0 = 2*Math.PI*nHost/lm, Y = k0*outerR*1e-9;
  if (Y < 1e-12) return { aRe: new Float64Array(0), aIm: new Float64Array(0), bRe: new Float64Array(0), bIm: new Float64Array(0), nmax: 0, x: Y };
  const mR: number[] = [], mI: number[] = [];
  for (let j = 0; j < NL; j++) { const [nj, kj] = layers[j].getDielectric(lambdaNm); mR.push(nj/nHost); mI.push(kj/nHost); }
  const S = layers.map(l => k0*l.radiusNm*1e-9);
  const NSTOP = Math.floor(Y + 4*Math.pow(Y, 1/3) + 2);
  const zCRe = mR[0]*S[0], zCIm = mI[0]*S[0];
  let [D0CRe, D0CIm] = ccot(zCRe, zCIm);
  interface SS { zInRe:number;zInIm:number;zOutRe:number;zOutIm:number;D0InRe:number;D0InIm:number;D0OutRe:number;D0OutIm:number;chi0InRe:number;chi0InIm:number;chi1InRe:number;chi1InIm:number;chi0OutRe:number;chi0OutIm:number;chi1OutRe:number;chi1OutIm:number; }
  const ss: SS[] = [];
  for (let j = 1; j < NL; j++) {
    const zIR=mR[j]*S[j-1],zII=mI[j]*S[j-1],zOR=mR[j]*S[j],zOI=mI[j]*S[j];
    const [d0IR,d0II]=ccot(zIR,zII),[d0OR,d0OI]=ccot(zOR,zOI);
    const [sIR,sII]=csin(zIR,zII),[cIR,cII]=ccos(zIR,zII),[sOR,sOI]=csin(zOR,zOI),[cOR,cOI]=ccos(zOR,zOI);
    ss.push({zInRe:zIR,zInIm:zII,zOutRe:zOR,zOutIm:zOI,D0InRe:d0IR,D0InIm:d0II,D0OutRe:d0OR,D0OutIm:d0OI,chi0InRe:-sIR,chi0InIm:-sII,chi1InRe:cIR,chi1InIm:cII,chi0OutRe:-sOR,chi0OutIm:-sOI,chi1OutRe:cOR,chi1OutIm:cOI});
  }
  let PSI0Y=Math.cos(Y),PSI1Y=Math.sin(Y),CHI0Y=-Math.sin(Y),CHI1Y=Math.cos(Y),XI1YRe=PSI1Y,XI1YIm=-CHI1Y;
  const aRA = new Float64Array(NSTOP), aIA = new Float64Array(NSTOP), bRA = new Float64Array(NSTOP), bIA = new Float64Array(NSTOP);
  for (let n = 1; n <= NSTOP; n++) {
    const RN=n, PSIY=(2*RN-1)*PSI1Y/Y-PSI0Y, CHIY=(2*RN-1)*CHI1Y/Y-CHI0Y, XIYRe=PSIY, XIYIm=-CHIY;
    const [novCRe,novCIm]=cdivN(RN,zCRe,zCIm),[invCRe,invCIm]=cdiv(1,0,novCRe-D0CRe,novCIm-D0CIm);
    const DCRe=invCRe-novCRe,DCIm=invCIm-novCIm;
    let DNBARRe=DCRe,DNBARIm=DCIm,GNBARRe=DCRe,GNBARIm=DCIm;
    for (let j = 0; j < ss.length; j++) {
      const s=ss[j],jL=j+1;
      const [novIRe,novIIm]=cdivN(RN,s.zInRe,s.zInIm),[invIRe,invIIm]=cdiv(1,0,novIRe-s.D0InRe,novIIm-s.D0InIm);
      const D1IRe=invIRe-novIRe,D1IIm=invIIm-novIIm;
      const [novORe,novOIm]=cdivN(RN,s.zOutRe,s.zOutIm),[invORe,invOIm]=cdiv(1,0,novORe-s.D0OutRe,novOIm-s.D0OutIm);
      const D1ORe=invORe-novORe,D1OIm=invOIm-novOIm;
      const fac=2*RN-1;
      const [fIRe,fIIm]=cdivN(fac,s.zInRe,s.zInIm);
      const CHIIRe=fIRe*s.chi1InRe-fIIm*s.chi1InIm-s.chi0InRe,CHIIIm=fIRe*s.chi1InIm+fIIm*s.chi1InRe-s.chi0InIm;
      const [ncIRe,ncIIm]=cdivN(RN,s.zInRe,s.zInIm);
      const CHIPIRe=s.chi1InRe-(ncIRe*CHIIRe-ncIIm*CHIIIm),CHIPIIm=s.chi1InIm-(ncIRe*CHIIIm+ncIIm*CHIIRe);
      const [fORe,fOIm]=cdivN(fac,s.zOutRe,s.zOutIm);
      const CHIORe=fORe*s.chi1OutRe-fOIm*s.chi1OutIm-s.chi0OutRe,CHIOIm=fORe*s.chi1OutIm+fOIm*s.chi1OutRe-s.chi0OutIm;
      const [ncORe,ncOIm]=cdivN(RN,s.zOutRe,s.zOutIm);
      const CHIPORe=s.chi1OutRe-(ncORe*CHIORe-ncOIm*CHIOIm),CHIPOIm=s.chi1OutIm-(ncORe*CHIOIm+ncOIm*CHIORe);
      const [REFRe,REFIm]=cdiv(mR[jL],mI[jL],mR[jL-1],mI[jL-1]);
      const [rDRe,rDIm]=cmul(REFRe,REFIm,DNBARRe,DNBARIm);
      const [rDcRe,rDcIm]=cmul(rDRe,rDIm,CHIIRe,CHIIIm);
      const [cDaRe,cDaIm]=cmul(CHIIRe,CHIIIm,D1IRe,D1IIm);
      const den2Re=cDaRe-CHIPIRe,den2Im=cDaIm-CHIPIIm;
      const [ddaRe,ddaIm]=cmul(rDcRe-CHIPIRe,rDcIm-CHIPIIm,den2Re,den2Im);
      const [ANCRe,ANCIm]=cdiv(rDRe-D1IRe,rDIm-D1IIm,ddaRe,ddaIm);
      const [cdoRe,cdoIm]=cmul(CHIORe,CHIOIm,D1ORe,D1OIm);
      const brRe=cdoRe-CHIPORe,brIm=cdoIm-CHIPOIm;
      const [BRKRe,BRKIm]=cmul(ANCRe,ANCIm,brRe,brIm);
      const [rD2Re,rD2Im]=cmul(REFRe,REFIm,D1IRe,D1IIm);
      const [rCpRe,rCpIm]=cmul(REFRe,REFIm,CHIPIRe,CHIPIIm);
      const [gCRe,gCIm]=cmul(GNBARRe,GNBARIm,CHIIRe,CHIIIm);
      const [ddbRe,ddbIm]=cmul(rCpRe-gCRe,rCpIm-gCIm,den2Re,den2Im);
      const [BNCRe,BNCIm]=cdiv(rD2Re-GNBARRe,rD2Im-GNBARIm,ddbRe,ddbIm);
      const [CRKRe,CRKIm]=cmul(BNCRe,BNCIm,brRe,brIm);
      const [bcpRe,bcpIm]=cmul(BRKRe,BRKIm,CHIPORe,CHIPOIm),[bcyRe,bcyIm]=cmul(BRKRe,BRKIm,CHIORe,CHIOIm);
      [DNBARRe,DNBARIm]=cdiv(D1ORe-bcpRe,D1OIm-bcpIm,1-bcyRe,-bcyIm);
      const [ccpRe,ccpIm]=cmul(CRKRe,CRKIm,CHIPORe,CHIPOIm),[ccyRe,ccyIm]=cmul(CRKRe,CRKIm,CHIORe,CHIOIm);
      [GNBARRe,GNBARIm]=cdiv(D1ORe-ccpRe,D1OIm-ccpIm,1-ccyRe,-ccyIm);
      s.D0InRe=D1IRe;s.D0InIm=D1IIm;s.D0OutRe=D1ORe;s.D0OutIm=D1OIm;
      s.chi0InRe=s.chi1InRe;s.chi0InIm=s.chi1InIm;s.chi1InRe=CHIIRe;s.chi1InIm=CHIIIm;
      s.chi0OutRe=s.chi1OutRe;s.chi0OutIm=s.chi1OutIm;s.chi1OutRe=CHIORe;s.chi1OutIm=CHIOIm;
    }
    D0CRe=DCRe;D0CIm=DCIm;
    const [dbrRe,dbrIm]=cdiv(DNBARRe,DNBARIm,mR[NL-1],mI[NL-1]);
    const ARe=dbrRe+RN/Y,AIm=dbrIm;
    const anNRe=ARe*PSIY-PSI1Y,anNIm=AIm*PSIY;
    const anDRe=(ARe*XIYRe-AIm*XIYIm)-XI1YRe,anDIm=(ARe*XIYIm+AIm*XIYRe)-XI1YIm;
    [aRA[n-1],aIA[n-1]]=cdiv(anNRe,anNIm,anDRe,anDIm);
    const [rgRe,rgIm]=cmul(mR[NL-1],mI[NL-1],GNBARRe,GNBARIm);
    const BRe=rgRe+RN/Y,BIm=rgIm;
    const bnNRe=BRe*PSIY-PSI1Y,bnNIm=BIm*PSIY;
    const bnDRe=(BRe*XIYRe-BIm*XIYIm)-XI1YRe,bnDIm=(BRe*XIYIm+BIm*XIYRe)-XI1YIm;
    [bRA[n-1],bIA[n-1]]=cdiv(bnNRe,bnNIm,bnDRe,bnDIm);
    PSI0Y=PSI1Y;PSI1Y=PSIY;CHI0Y=CHI1Y;CHI1Y=CHIY;XI1YRe=PSI1Y;XI1YIm=-CHI1Y;
  }
  return { aRe:aRA, aIm:aIA, bRe:bRA, bIm:bIA, nmax:NSTOP, x:Y };
}

// === Angular Pattern (far-field) ===

export interface AngularPattern { theta: Float64Array; iPerp: Float64Array; iPar: Float64Array; }

export function computeAngularPattern(coeffs: SphereCoeffs, nTheta = 361): AngularPattern {
  const theta = new Float64Array(nTheta), iPerp = new Float64Array(nTheta), iPar = new Float64Array(nTheta);
  for (let t = 0; t < nTheta; t++) {
    theta[t] = Math.PI * t / (nTheta - 1);
    const mu = Math.cos(theta[t]);
    let piPrev = 0, piCur = 1;
    let s1Re = 0, s1Im = 0, s2Re = 0, s2Im = 0;
    for (let n = 1; n <= coeffs.nmax; n++) {
      const tau = n * mu * piCur - (n + 1) * piPrev;
      const w = (2 * n + 1) / (n * (n + 1));
      s1Re += w * (coeffs.aRe[n-1] * piCur + coeffs.bRe[n-1] * tau);
      s1Im += w * (coeffs.aIm[n-1] * piCur + coeffs.bIm[n-1] * tau);
      s2Re += w * (coeffs.aRe[n-1] * tau + coeffs.bRe[n-1] * piCur);
      s2Im += w * (coeffs.aIm[n-1] * tau + coeffs.bIm[n-1] * piCur);
      const piNext = ((2 * n + 1) / n) * mu * piCur - ((n + 1) / n) * piPrev;
      piPrev = piCur; piCur = piNext;
    }
    iPerp[t] = s1Re * s1Re + s1Im * s1Im;
    iPar[t] = s2Re * s2Re + s2Im * s2Im;
  }
  return { theta, iPerp, iPar };
}

// === Near field |E|² on the x-z plane ===

export interface NearFieldResult { grid: Float64Array; gridSize: number; extent: number; minVal: number; maxVal: number; }

export function computeNearField(coeffs: SphereCoeffs, k: number, radiusNm: number, gridSize: number, extentNm: number): NearFieldResult {
  const R = radiusNm * 1e-9;
  const hasInternal = !!(coeffs.cRe && coeffs.dRe && coeffs.mRe !== undefined);
  const grid = new Float64Array(gridSize * gridSize);
  let minVal = Infinity, maxVal = 0;
  for (let iz = 0; iz < gridSize; iz++) {
    const z = (-extentNm + iz * 2 * extentNm / (gridSize - 1)) * 1e-9;
    for (let ix = 0; ix < gridSize; ix++) {
      const x = (-extentNm + ix * 2 * extentNm / (gridSize - 1)) * 1e-9;
      let r = Math.sqrt(x * x + z * z);
      const inside = r < R;

      if (inside && !hasInternal) { grid[iz * gridSize + ix] = -1; continue; }

      if (inside) {
        // --- Internal field (B&H 4.40) ---
        if (r < R * 0.005) r = R * 0.005; // regularize near origin
        const costh = z / r, sinth = Math.sqrt(1 - costh * costh);
        // ρ₁ = m·k·r (complex)
        const r1Re = coeffs.mRe! * k * r, r1Im = coeffs.mIm! * k * r;
        const r1M2 = r1Re * r1Re + r1Im * r1Im;
        // Complex sin/cos(ρ₁)
        const chI = Math.cosh(r1Im), shI = Math.sinh(r1Im);
        const sinRRe = Math.sin(r1Re) * chI, sinRIm = Math.cos(r1Re) * shI;
        const cosRRe = Math.cos(r1Re) * chI, cosRIm = -Math.sin(r1Re) * shI;
        // ψ₀ = sin(ρ₁), ψ₁ = sin(ρ₁)/ρ₁ - cos(ρ₁)
        let ppRe = sinRRe, ppIm = sinRIm;
        const sovRe = (sinRRe * r1Re + sinRIm * r1Im) / r1M2;
        const sovIm = (sinRIm * r1Re - sinRRe * r1Im) / r1M2;
        let pcRe = sovRe - cosRRe, pcIm = sovIm - cosRIm;
        let piPr = 0, piCu = 1;
        let ErRe = 0, ErIm = 0, EtRe = 0, EtIm = 0;
        for (let n = 1; n <= coeffs.nmax; n++) {
          const psiRe = pcRe, psiIm = pcIm;
          // ψ'_n = ψ_{n-1} - n/ρ₁ · ψ_n
          const novRe = n * r1Re / r1M2, novIm = -n * r1Im / r1M2;
          const psiDRe = ppRe - (novRe * psiRe - novIm * psiIm);
          const psiDIm = ppIm - (novRe * psiIm + novIm * psiRe);
          const tau = n * costh * piCu - (n + 1) * piPr;
          const Pn1 = sinth * piCu;
          const ww = (2 * n + 1) / (n * (n + 1));
          let eRe = 0, eIm = 0;
          switch (n % 4) { case 0: eRe = ww; break; case 1: eIm = ww; break; case 2: eRe = -ww; break; case 3: eIm = -ww; break; }
          // E_n·(-i)·d_n for N term: -i·E_n = (eIm, -eRe)
          const eniRe = eIm, eniIm = -eRe;
          const eidRe = eniRe * coeffs.dRe![n-1] - eniIm * coeffs.dIm![n-1];
          const eidIm = eniRe * coeffs.dIm![n-1] + eniIm * coeffs.dRe![n-1];
          // E_n·c_n for M term
          const ecRe = eRe * coeffs.cRe![n-1] - eIm * coeffs.cIm![n-1];
          const ecIm = eRe * coeffs.cIm![n-1] + eIm * coeffs.cRe![n-1];
          // E_r += E_n·(-i·d_n)·n(n+1)·Pn1 · ψ_n/ρ₁²
          const nn1 = n * (n + 1);
          const r1SqRe = r1Re * r1Re - r1Im * r1Im, r1SqIm = 2 * r1Re * r1Im;
          const r1SqM2 = r1SqRe * r1SqRe + r1SqIm * r1SqIm;
          const psrRe = (psiRe * r1SqRe + psiIm * r1SqIm) / r1SqM2;
          const psrIm = (psiIm * r1SqRe - psiRe * r1SqIm) / r1SqM2;
          const fRR = nn1 * Pn1;
          ErRe += fRR * (eidRe * psrRe - eidIm * psrIm);
          ErIm += fRR * (eidRe * psrIm + eidIm * psrRe);
          // E_θ += [E_n·c_n·π_n·ψ_n + E_n·(-i·d_n)·τ_n·ψ'_n] / ρ₁
          const invR1Re = r1Re / r1M2, invR1Im = -r1Im / r1M2;
          // c_n term: ec·ψ_n / ρ₁ · π_n
          const ecpRe = ecRe * psiRe - ecIm * psiIm, ecpIm = ecRe * psiIm + ecIm * psiRe;
          const t1Re = (ecpRe * invR1Re - ecpIm * invR1Im) * piCu;
          const t1Im = (ecpRe * invR1Im + ecpIm * invR1Re) * piCu;
          EtRe += t1Re; EtIm += t1Im;
          // d_n term: eid·ψ'_n / ρ₁ · τ_n
          const edpRe = eidRe * psiDRe - eidIm * psiDIm, edpIm = eidRe * psiDIm + eidIm * psiDRe;
          const t2Re = (edpRe * invR1Re - edpIm * invR1Im) * tau;
          const t2Im = (edpRe * invR1Im + edpIm * invR1Re) * tau;
          EtRe += t2Re; EtIm += t2Im;
          // Recurrences: ψ_{n+1} = (2n+1)/ρ₁ · ψ_n - ψ_{n-1}
          const bfRe = (2 * n + 1) * r1Re / r1M2, bfIm = -(2 * n + 1) * r1Im / r1M2;
          const pNRe = bfRe * pcRe - bfIm * pcIm - ppRe;
          const pNIm = bfRe * pcIm + bfIm * pcRe - ppIm;
          ppRe = pcRe; ppIm = pcIm; pcRe = pNRe; pcIm = pNIm;
          const piN = ((2 * n + 1) / n) * costh * piCu - ((n + 1) / n) * piPr;
          piPr = piCu; piCu = piN;
        }
        const ExRe = ErRe * sinth + EtRe * costh, ExIm = ErIm * sinth + EtIm * costh;
        const EzRe = ErRe * costh - EtRe * sinth, EzIm = ErIm * costh - EtIm * sinth;
        const Esq = ExRe * ExRe + ExIm * ExIm + EzRe * EzRe + EzIm * EzIm;
        grid[iz * gridSize + ix] = Esq;
        if (Esq > maxVal) maxVal = Esq;
        if (Esq > 0 && Esq < minVal) minVal = Esq;
      } else {
        // --- External field (incident + scattered) ---
        const rho = k * r;
        const costh = z / r, sinth = Math.sqrt(1 - costh * costh);
        const phase = k * z;
        let ErRe = sinth * Math.cos(phase), ErIm = sinth * Math.sin(phase);
        let EtRe = costh * Math.cos(phase), EtIm = costh * Math.sin(phase);
        const sinR = Math.sin(rho), cosR = Math.cos(rho);
        let pP = sinR, pC = sinR / rho - cosR;
        let cP = cosR, cC = cosR / rho + sinR;
        let piPr = 0, piCu = 1;
        for (let n = 1; n <= coeffs.nmax; n++) {
          const xiRe = pC, xiIm = -cC;
          const psiD = pP - (n / rho) * pC, chiD = cP - (n / rho) * cC;
          const xiDRe = psiD, xiDIm = -chiD;
          const tau = n * costh * piCu - (n + 1) * piPr;
          const Pn1 = sinth * piCu;
          const ww = (2 * n + 1) / (n * (n + 1));
          let eRe = 0, eIm = 0;
          switch (n % 4) { case 0: eRe = ww; break; case 1: eIm = ww; break; case 2: eRe = -ww; break; case 3: eIm = -ww; break; }
          const eiRe = -eIm, eiIm = eRe;
          const earRe = eiRe * coeffs.aRe[n-1] - eiIm * coeffs.aIm[n-1];
          const earIm = eiRe * coeffs.aIm[n-1] + eiIm * coeffs.aRe[n-1];
          const nn1 = n * (n + 1), fR = nn1 * Pn1 / (rho * rho);
          ErRe += fR * (earRe * xiRe - earIm * xiIm);
          ErIm += fR * (earRe * xiIm + earIm * xiRe);
          const tF = tau / rho;
          EtRe += tF * (earRe * xiDRe - earIm * xiDIm);
          EtIm += tF * (earRe * xiDIm + earIm * xiDRe);
          const ebRe = eRe * coeffs.bRe[n-1] - eIm * coeffs.bIm[n-1];
          const ebIm = eRe * coeffs.bIm[n-1] + eIm * coeffs.bRe[n-1];
          const pF = piCu / rho;
          EtRe -= pF * (ebRe * xiRe - ebIm * xiIm);
          EtIm -= pF * (ebRe * xiIm + ebIm * xiRe);
          const bf = (2 * n + 1) / rho;
          const pN = bf * pC - pP; pP = pC; pC = pN;
          const cN = bf * cC - cP; cP = cC; cC = cN;
          const piN = ((2 * n + 1) / n) * costh * piCu - ((n + 1) / n) * piPr;
          piPr = piCu; piCu = piN;
        }
        const ExRe = ErRe * sinth + EtRe * costh, ExIm = ErIm * sinth + EtIm * costh;
        const EzRe = ErRe * costh - EtRe * sinth, EzIm = ErIm * costh - EtIm * sinth;
        const Esq = ExRe * ExRe + ExIm * ExIm + EzRe * EzRe + EzIm * EzIm;
        grid[iz * gridSize + ix] = Esq;
        if (Esq > maxVal) maxVal = Esq;
        if (Esq > 0 && Esq < minVal) minVal = Esq;
      }
    }
  }
  return { grid, gridSize, extent: extentNm, minVal, maxVal };
}
