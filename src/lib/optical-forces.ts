/**
 * Optical forces on a sphere — radiation pressure from exact Mie theory and
 * optical-tweezers trap parameters in the Rayleigh (dipole) regime.
 *
 * Radiation pressure (any size, exact):
 *   g = <cosθ> from the Mie coefficients (Bohren & Huffman §4.5):
 *     g·Qsca·x²/4 = Σₙ n(n+2)/(n+1)·Re(aₙaₙ₊₁* + bₙbₙ₊₁*)
 *                 + Σₙ (2n+1)/(n(n+1))·Re(aₙbₙ*)
 *   σ_pr = σ_ext − g·σ_sca,   F = n_host·σ_pr·I/c  (plane wave / beam centre)
 *
 * Rayleigh trap (a ≪ λ), Gaussian beam of waist w₀ and power P:
 *   CM = (m² − 1)/(m² + 2),  β = (2π n_host a³ / c)·Re(CM)
 *   U(r) = −β·I(r);  I₀ = 2P/πw₀²;  z_R = π w₀² n_host / λ
 *   κ_r = 4βI₀/w₀²,  κ_z = 2βI₀/z_R²,  trap depth U₀ = βI₀
 * Validity: dipole approximation — trust it for 2a ≲ λ/(5 n_host).
 */
import type { SphereCoeffs } from './mie';
import { peakIrradianceWm2 } from './photothermal';

export const C_M_S = 2.99792458e8;
export const KB_J_K = 1.380649e-23;

/** Asymmetry parameter g = <cosθ> from per-order Mie coefficients. */
export function asymmetryParameter(co: SphereCoeffs): number {
  const { aRe, aIm, bRe, bIm, nmax, x } = co;
  if (nmax < 1 || x <= 0) return 0;
  let s = 0, qs = 0;
  for (let n = 1; n <= nmax; n++) {
    const i = n - 1;
    qs += (2 * n + 1) * (aRe[i] * aRe[i] + aIm[i] * aIm[i] + bRe[i] * bRe[i] + bIm[i] * bIm[i]);
    // Re(aₙ bₙ*) term
    s += ((2 * n + 1) / (n * (n + 1))) * (aRe[i] * bRe[i] + aIm[i] * bIm[i]);
    if (n < nmax) {
      const j = n; // index of order n+1
      // Re(aₙ aₙ₊₁* + bₙ bₙ₊₁*)
      s += (n * (n + 2) / (n + 1)) *
        (aRe[i] * aRe[j] + aIm[i] * aIm[j] + bRe[i] * bRe[j] + bIm[i] * bIm[j]);
    }
  }
  if (qs <= 0) return 0;
  // g = (4/x²)·s / Qsca with Qsca = (2/x²)·qs → the x² cancels, leaving 2s/qs.
  return (2 * s) / qs;
}

/** Radiation-pressure cross-section σ_pr = σ_ext − g·σ_sca (same units as inputs). */
export function pressureCrossSection(cext: number, csca: number, g: number): number {
  return cext - g * csca;
}

/** Radiation force in N on a sphere at intensity I (W/m²), σ_pr in nm². */
export function radiationForceN(sigmaPrNm2: number, irradianceWm2: number, nHost: number): number {
  return (nHost * sigmaPrNm2 * 1e-18 * irradianceWm2) / C_M_S;
}

export interface RayleighTrapParams {
  /** Particle refractive index (real part; absorption ignored for the gradient force). */
  nParticle: number;
  radiusNm: number;
  nHost: number;
  lambdaNm: number;
  powerMw: number;
  waistUm: number;
  temperatureK?: number;
}

export interface RayleighTrapResult {
  /** Clausius–Mossotti factor Re[(m²−1)/(m²+2)]. Negative → particle is repelled. */
  cmFactor: number;
  /** Transverse and axial stiffness, pN/µm. */
  kappaRPnPerUm: number;
  kappaZPnPerUm: number;
  /** Trap depth in J and in units of k_B·T. */
  trapDepthJ: number;
  trapDepthKbT: number;
  /** Peak irradiance at the focus, W/m². */
  irradianceWm2: number;
  /** Rayleigh range, µm. */
  rayleighRangeUm: number;
  /** True when the dipole approximation is trustworthy (2a ≲ λ/(5·n_host)). */
  dipoleValid: boolean;
}

export function rayleighTrap(p: RayleighTrapParams): RayleighTrapResult {
  const T = p.temperatureK ?? 293;
  const m = p.nParticle / p.nHost;
  const cm = (m * m - 1) / (m * m + 2);
  const a = p.radiusNm * 1e-9;
  const beta = (2 * Math.PI * p.nHost * a * a * a / C_M_S) * cm; // U = −β·I
  const I0 = peakIrradianceWm2(p.powerMw, p.waistUm);
  const w0 = p.waistUm * 1e-6;
  const zR = (Math.PI * w0 * w0 * p.nHost) / (p.lambdaNm * 1e-9);
  const kappaR = (4 * beta * I0) / (w0 * w0);   // N/m
  const kappaZ = (2 * beta * I0) / (zR * zR);   // N/m
  const U0 = beta * I0;                          // J
  return {
    cmFactor: cm,
    kappaRPnPerUm: kappaR * 1e6,   // N/m → pN/µm (1 N/m = 1e12 pN / 1e6 µm)
    kappaZPnPerUm: kappaZ * 1e6,
    trapDepthJ: U0,
    trapDepthKbT: U0 / (KB_J_K * T),
    irradianceWm2: I0,
    rayleighRangeUm: zR * 1e6,
    dipoleValid: 2 * p.radiusNm <= p.lambdaNm / (5 * p.nHost),
  };
}
