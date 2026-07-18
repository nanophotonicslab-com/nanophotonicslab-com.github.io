import { describe, it, expect } from 'vitest';
import { peakIrradianceWm2, steadyStateDeltaT, deltaTSpectrum } from './photothermal';
import { peakAnalysis } from './spectrum';

describe('photothermal', () => {
  it('peak irradiance of a Gaussian beam: I = 2P/πw₀²', () => {
    // 1 mW into w₀ = 1 µm → 2·1e-3 / (π·1e-12) W/m²
    expect(peakIrradianceWm2(1, 1)).toBeCloseTo(2e-3 / (Math.PI * 1e-12), -3);
  });

  it('ΔT = σ_abs·I / (4πκR) in K', () => {
    const I = peakIrradianceWm2(1, 1);
    // σ_abs = 1e4 nm², water κ = 0.606, R = 50 nm
    const dT = steadyStateDeltaT(1e4, I, 0.606, 50);
    const expected = (1e4 * 1e-18 * I) / (4 * Math.PI * 0.606 * 50e-9);
    expect(dT).toBeCloseTo(expected, 6);
    expect(dT).toBeGreaterThan(1); // physically: several K for a strong absorber
  });

  it('guards degenerate inputs and maps whole spectra', () => {
    expect(steadyStateDeltaT(1e4, 1e8, 0.606, 0)).toBe(0);
    expect(steadyStateDeltaT(1e4, 1e8, 0, 50)).toBe(0);
    const dts = deltaTSpectrum(Float64Array.from([0, 1e4, 2e4]), 1e8, 0.606, 50);
    expect(dts[0]).toBe(0);
    expect(dts[2]).toBeCloseTo(2 * dts[1], 9);
  });
});

describe('peakAnalysis', () => {
  it('recovers position, height and FWHM of a Lorentzian', () => {
    const n = 801, x0 = 600, gamma = 40; // FWHM = γ
    const x = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = 400 + (400 * i) / (n - 1);
      const d = x[i] - x0;
      y[i] = 1 / (1 + (2 * d / gamma) ** 2);
    }
    const p = peakAnalysis(x, y)!;
    expect(p.x0).toBeCloseTo(x0, 0);
    expect(p.y0).toBeCloseTo(1, 3);
    expect(p.fwhm).toBeCloseTo(gamma, 0);
    expect(p.q).toBeCloseTo(x0 / gamma, 1);
  });

  it('returns null for empty/flat input and NaN fwhm when a flank leaves the grid', () => {
    expect(peakAnalysis(new Float64Array(0), new Float64Array(0))).toBeNull();
    expect(peakAnalysis(Float64Array.from([1, 2, 3]), Float64Array.from([0, 0, 0]))).toBeNull();
    const x = Float64Array.from([1, 2, 3, 4]);
    const y = Float64Array.from([1, 0.9, 0.8, 0.7]); // peak at the edge, no left crossing
    const p = peakAnalysis(x, y)!;
    expect(Number.isNaN(p.fwhm)).toBe(true);
  });
});
