/**
 * Steady-state photothermal heating of a nanoparticle under CW Gaussian
 * illumination — same model as the Photothermal tool:
 *   I = 2P / (π w₀²)  (peak irradiance of a Gaussian beam)
 *   ΔT(R) = P_abs / (4π κ R),  P_abs = σ_abs · I
 * Uniform infinite host, temperature-independent κ.
 */

/** Peak irradiance of a Gaussian beam, in W/m². */
export function peakIrradianceWm2(powerMw: number, waistUm: number): number {
  const P = powerMw * 1e-3;
  const w0 = waistUm * 1e-6;
  return (2 * P) / (Math.PI * w0 * w0);
}

/**
 * Steady-state surface temperature rise, in K.
 * @param sigmaAbsNm2  absorption cross-section (nm²)
 * @param irradianceWm2  illumination irradiance (W/m²)
 * @param kappaWmK  host thermal conductivity (W/m·K)
 * @param radiusNm  particle radius (nm)
 */
export function steadyStateDeltaT(
  sigmaAbsNm2: number, irradianceWm2: number, kappaWmK: number, radiusNm: number,
): number {
  if (radiusNm <= 0 || kappaWmK <= 0) return 0;
  const pAbsW = sigmaAbsNm2 * 1e-18 * irradianceWm2;
  return pAbsW / (4 * Math.PI * kappaWmK * radiusNm * 1e-9);
}

/** ΔT(λ) for a whole σ_abs spectrum channel. */
export function deltaTSpectrum(
  sigmaAbsNm2: Float64Array, irradianceWm2: number, kappaWmK: number, radiusNm: number,
): Float64Array {
  return Float64Array.from(sigmaAbsNm2, (s) => steadyStateDeltaT(s, irradianceWm2, kappaWmK, radiusNm));
}
