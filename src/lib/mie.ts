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
 * Interpolate tabulated (n, k) optical constants at a given wavelength.
 * Data format: [[wavelength_μm, n, k], ...] sorted by wavelength.
 *
 * @param data  Tabulated optical constants (wavelength in μm)
 * @param lambdaNm  Query wavelength in nm
 * @returns [n, k] at the query wavelength
 */
export { interpolateNK } from './materials';

export const HC_EV_NM = 1239.841984;
