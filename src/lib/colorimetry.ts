/**
 * CIE 1931 colorimetry — Wyman-Sloan-Shirley analytic color-matching functions
 * with illuminant E and XYZ → sRGB (D65) conversion.
 */

function _gauss(lam: number, mu: number, s1: number, s2: number): number {
  const t = lam - mu;
  return t < 0
    ? Math.exp(-0.5 * (t / s1) * (t / s1))
    : Math.exp(-0.5 * (t / s2) * (t / s2));
}

// KNOWN LIMITATION (verified 2026-07-18): the analytic Wyman fit underestimates
// x̄ beyond ~680 nm (x̄(700) ≈ 0.0057 vs CIE 0.0114), so deep-red spectra render
// orange-shifted. Accurate through ~650 nm; a table-based x̄ tail would fix it.
export function xBar(l: number): number {
  return 0.362 * _gauss(l, 442.0, 16.0, 26.7)
       + 1.056 * _gauss(l, 599.8, 37.9, 31.0)
       - 0.065 * _gauss(l, 501.1, 20.4, 26.2);
}

export function yBar(l: number): number {
  return 0.821 * _gauss(l, 568.8, 46.9, 40.5)
       + 0.286 * _gauss(l, 530.9, 16.3, 31.1);
}

export function zBar(l: number): number {
  return 1.217 * _gauss(l, 437.0, 11.8, 36.0)
       + 0.681 * _gauss(l, 459.0, 26.0, 13.8);
}

/**
 * Integrate a reflectance/scattering spectrum R(λ) under illuminant E
 * and return an sRGB hex color string.
 *
 * @param lams  Wavelengths in nm
 * @param R     Spectral values (arbitrary units; will be normalized)
 * @returns Hex color string, e.g. '#c8a040'
 */
export function spectrumToHex(lams: Float64Array, R: Float64Array): string {
  let X = 0, Y = 0, Z = 0, N = 0;
  for (let i = 0; i < lams.length; i++) {
    const l = lams[i];
    if (l < 380 || l > 780) continue;
    const dl = i > 0 ? (lams[i] - lams[i - 1]) : 1;
    const xb = xBar(l), yb = yBar(l), zb = zBar(l);
    X += R[i] * xb * dl;
    Y += R[i] * yb * dl;
    Z += R[i] * zb * dl;
    N += yb * dl;
  }
  if (N <= 0) return '#222222';
  X /= N; Y /= N; Z /= N;

  // XYZ → linear sRGB (D65)
  let r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  let g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  let b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;

  // Desaturate out-of-gamut
  const minC = Math.min(r, g, b, 0);
  if (minC < 0) { r -= minC; g -= minC; b -= minC; }
  const maxC = Math.max(r, g, b, 1e-9);
  r /= maxC; g /= maxC; b /= maxC;

  // sRGB gamma
  const gamma = (c: number) =>
    c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  const toHex = (c: number) =>
    Math.max(0, Math.min(255, Math.round(gamma(c) * 255))).toString(16).padStart(2, '0');
  return '#' + toHex(r) + toHex(g) + toHex(b);
}
