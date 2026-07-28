/**
 * Emitters -> photon image.
 *
 * Each emitter contributes a photon-conserving normalised 2D Gaussian: its
 * integral over the plane equals its photon budget, so the total photon count
 * does not depend on sigma_psf or on the pixel size. Rather than sampling the
 * Gaussian at pixel centres (which loses photons whenever the PSF is narrow
 * relative to a pixel), each pixel receives the *integral* of the Gaussian
 * over its area, which is exact and separable in x and y.
 */
import type { Emitter } from './dynamics';

/**
 * Abramowitz & Stegun 7.1.26 rational approximation, |error| < 1.5e-7 — three
 * orders of magnitude tighter than the 0.5% photon-conservation requirement.
 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return sign * y;
}

/** Integral of a unit-area 1D Gaussian over the pixel spanning [k, k+1]. */
function pixelWeight(k: number, centre: number, sigma: number): number {
  const s = Math.SQRT2 * sigma;
  return 0.5 * (erf((k + 1 - centre) / s) - erf((k - centre) / s));
}

export interface RenderOptions {
  /** Field of view, in pixels (square). */
  field: number;
  /** Camera pixel size referred to the sample plane, in nm. */
  pixelNm: number;
  /** PSF standard deviation in the sample plane, in nm. */
  sigmaPsfNm: number;
}

/**
 * Render emitters into a photon image (photons per pixel, no noise).
 *
 * Only the +/-4 sigma neighbourhood of each emitter is evaluated: a full-field
 * pass per emitter is O(N * field^2) and misses the performance target. The
 * truncation discards ~0.013% of a Gaussian's volume, well inside the 0.5%
 * conservation tolerance.
 *
 * Emitters near an edge are clipped by the field of view, exactly as a real
 * camera crops a sample that extends beyond it.
 */
export function renderFrame(
  emitters: Emitter[], opt: RenderOptions, out?: Float32Array,
): Float32Array {
  const { field, pixelNm, sigmaPsfNm } = opt;
  let img: Float32Array;
  if (out && out.length === field * field) { img = out; img.fill(0); }
  else img = new Float32Array(field * field);
  const sigmaPx = sigmaPsfNm / pixelNm;
  const reach = 4 * sigmaPx;
  // Column weights are reused across the rows of one emitter.
  const wx: number[] = [];
  for (const e of emitters) {
    const cx = e.x / pixelNm;
    const cy = e.y / pixelNm;
    const i0 = Math.max(0, Math.floor(cx - reach) - 1);
    const i1 = Math.min(field - 1, Math.ceil(cx + reach) + 1);
    const j0 = Math.max(0, Math.floor(cy - reach) - 1);
    const j1 = Math.min(field - 1, Math.ceil(cy + reach) + 1);
    if (i1 < i0 || j1 < j0) continue;
    wx.length = 0;
    for (let i = i0; i <= i1; i++) wx.push(pixelWeight(i, cx, sigmaPx));
    for (let j = j0; j <= j1; j++) {
      const wy = pixelWeight(j, cy, sigmaPx);
      if (wy === 0) continue;
      const row = j * field;
      for (let i = i0; i <= i1; i++) {
        img[row + i] += e.photons * wx[i - i0] * wy;
      }
    }
  }
  return img;
}

/**
 * Radially averaged profile of an image about a centre, in pixel-wide bins.
 * Used by the PSF cross-section plot and by the DeepTrack2 parity check.
 */
export function radialProfile(
  img: Float32Array, field: number, cx: number, cy: number, nBins: number,
): { r: Float64Array; mean: Float64Array; counts: Float64Array } {
  const sum = new Float64Array(nBins);
  const count = new Float64Array(nBins);
  for (let j = 0; j < field; j++) {
    for (let i = 0; i < field; i++) {
      const dx = i + 0.5 - cx;
      const dy = j + 0.5 - cy;
      const b = Math.round(Math.sqrt(dx * dx + dy * dy));
      if (b < nBins) { sum[b] += img[j * field + i]; count[b]++; }
    }
  }
  const r = new Float64Array(nBins);
  const mean = new Float64Array(nBins);
  for (let b = 0; b < nBins; b++) {
    r[b] = b;
    mean[b] = count[b] > 0 ? sum[b] / count[b] : 0;
  }
  return { r, mean, counts: count };
}

/**
 * Standard deviation of a noise-free spot, in pixels, measured from its second
 * moment about (cx, cy) and corrected for the camera's own blur.
 *
 * A pixel reports the integral of the PSF over its area, so the recorded spot
 * is the PSF convolved with a 1-pixel box. Variances add, giving
 * sigma_recorded^2 = sigma_psf^2 + 1/12 (the variance of a unit box), and the
 * 1/12 is removed here so the result is the PSF width itself.
 *
 * Moments are used rather than interpolating to half maximum because 1-pixel
 * radial bins report the *mean over an annulus*, which overestimates the width
 * of a narrow spot by several percent.
 */
export function measureSigmaPx(img: Float32Array, field: number, cx: number, cy: number): number {
  let w = 0, m2 = 0;
  for (let j = 0; j < field; j++) {
    for (let i = 0; i < field; i++) {
      const v = img[j * field + i];
      if (v <= 0) continue;
      const dx = i + 0.5 - cx;
      const dy = j + 0.5 - cy;
      w += v;
      m2 += v * (dx * dx + dy * dy);
    }
  }
  if (!(w > 0)) return NaN;
  // m2/w is the sum of both axes' variances for a radially symmetric spot
  const perAxis = m2 / w / 2 - 1 / 12;
  return perAxis > 0 ? Math.sqrt(perAxis) : NaN;
}

/**
 * FWHM of a noise-free spot in pixels, from its second moment.
 * See `measureSigmaPx` for why moments rather than a half-maximum crossing.
 */
export function measureFwhmPx(img: Float32Array, field: number, cx: number, cy: number): number {
  return FWHM_PER_SIGMA_LOCAL * measureSigmaPx(img, field, cx, cy);
}

/**
 * FWHM in pixels of a spot centred on the pixel (i0, j0), measured along the
 * row through its centre by linear interpolation to half maximum.
 *
 * Three estimators exist in this module for three different jobs:
 *  - `measureSigmaPx` (second moments) is exact for this module's Gaussian PSF
 *    and is what the kernel's own width test uses;
 *  - it is meaningless for an Airy-like pupil PSF, whose rings decay as r^-3 so
 *    that the second moment grows with the field of view;
 *  - radial binning would fix that but averages over annuli, which biases a
 *    narrow spot more than a wide one — precisely the two things being compared.
 *
 * A cut through the centre row samples the profile at exact integer radii with
 * no annulus averaging, so the same systematic applies to both spots and the
 * comparison is meaningful. This is what the DeepTrack2 parity check uses.
 */
export function measureFwhmCut(img: Float32Array, field: number, i0: number, j0: number): number {
  const row = j0 * field;
  const peak = img[row + i0];
  if (!(peak > 0)) return NaN;
  const half = peak / 2;
  // walk outwards in both directions and average the two half-widths
  const halfWidth = (dir: 1 | -1): number => {
    let prev = peak;
    for (let k = 1; ; k++) {
      const i = i0 + dir * k;
      if (i < 0 || i >= field) return NaN;
      const v = img[row + i];
      if (v <= half) return k - 1 + (prev - half) / (prev - v);
      prev = v;
    }
  };
  const a = halfWidth(1), b = halfWidth(-1);
  if (Number.isNaN(a) && Number.isNaN(b)) return NaN;
  if (Number.isNaN(a)) return 2 * b;
  if (Number.isNaN(b)) return 2 * a;
  return a + b;
}

/** Local copy of sqrt(8 ln 2) to keep this module free of import cycles. */
const FWHM_PER_SIGMA_LOCAL = 2.3548200450309493;
