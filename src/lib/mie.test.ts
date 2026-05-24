import { describe, it, expect } from 'vitest';
import { mieAt, interpolateNK, HC_EV_NM } from './mie';

describe('mieAt', () => {
  it('returns zero cross-sections for zero-size particle', () => {
    const r = mieAt(1.5, 0, 1, 0, 550);
    expect(r.csca).toBe(0);
    expect(r.cext).toBe(0);
    expect(r.cabs).toBe(0);
  });

  it('satisfies Cext = Csca + Cabs', () => {
    const r = mieAt(1.5, 0.1, 1, 50, 550);
    expect(r.cext).toBeCloseTo(r.csca + r.cabs, 6);
  });

  it('produces non-negative cross-sections', () => {
    const r = mieAt(3.5, 0.5, 1.33, 100, 400);
    expect(r.csca).toBeGreaterThanOrEqual(0);
    expect(r.cext).toBeGreaterThanOrEqual(0);
    expect(r.cabs).toBeGreaterThanOrEqual(0);
  });

  it('Rayleigh limit: Csca scales as r^6 for small dielectric sphere', () => {
    // For x << 1, Csca ~ x^4 ~ r^4 at fixed wavelength, but the
    // full Rayleigh formula gives Csca = (128 pi^5 / 3) * (a^6/lam^4) * |K|^2
    // where K = (eps-1)/(eps+2). So Csca ∝ r^6.
    const r1 = mieAt(1.5, 0, 1, 5, 1000);
    const r2 = mieAt(1.5, 0, 1, 10, 1000);
    // Csca(10nm) / Csca(5nm) should be (10/5)^6 = 64
    const ratio = r2.csca / r1.csca;
    expect(ratio).toBeCloseTo(64, 0);
  });

  it('Rayleigh limit: matches analytical formula for small dielectric sphere', () => {
    const n = 1.5;
    const radius = 5; // nm
    const lambda = 1000; // nm
    const eps = n * n;
    const K = (eps - 1) / (eps + 2);
    const a = radius * 1e-9;
    const lam = lambda * 1e-9;
    const csca_analytic = (128 * Math.PI ** 5 / 3) * (a ** 6 / lam ** 4) * K * K;
    const csca_analytic_nm2 = csca_analytic * 1e18;

    const r = mieAt(n, 0, 1, radius, lambda);
    expect(r.csca).toBeCloseTo(csca_analytic_nm2, 3);
  });

  it('no absorption for purely real refractive index', () => {
    const r = mieAt(2.0, 0, 1, 50, 550);
    expect(r.cabs).toBeCloseTo(0, 6);
  });

  it('absorbing particle has Cabs > 0', () => {
    const r = mieAt(1.5, 0.5, 1, 50, 550);
    expect(r.cabs).toBeGreaterThan(0);
  });

  it('Cext increases with particle size (resonance regime)', () => {
    const r1 = mieAt(1.5, 0, 1, 20, 550);
    const r2 = mieAt(1.5, 0, 1, 100, 550);
    expect(r2.cext).toBeGreaterThan(r1.cext);
  });

  it('host medium affects cross-sections', () => {
    const r1 = mieAt(1.5, 0, 1.0, 50, 550);
    const r2 = mieAt(1.5, 0, 1.33, 50, 550);
    expect(r1.csca).not.toBeCloseTo(r2.csca, 2);
  });
});

describe('interpolateNK', () => {
  const data = [
    [0.3, 1.4, 0.01],
    [0.5, 1.5, 0.02],
    [0.7, 1.6, 0.03],
    [0.9, 1.7, 0.04],
  ];

  it('returns first entry for wavelength below range', () => {
    const [n, k] = interpolateNK(data, 200);
    expect(n).toBe(1.4);
    expect(k).toBe(0.01);
  });

  it('returns last entry for wavelength above range', () => {
    const [n, k] = interpolateNK(data, 1500);
    expect(n).toBe(1.7);
    expect(k).toBe(0.04);
  });

  it('returns exact value at a data point', () => {
    const [n, k] = interpolateNK(data, 500);
    expect(n).toBeCloseTo(1.5, 10);
    expect(k).toBeCloseTo(0.02, 10);
  });

  it('interpolates linearly between data points', () => {
    const [n, k] = interpolateNK(data, 600); // midpoint of 500-700nm
    expect(n).toBeCloseTo(1.55, 10);
    expect(k).toBeCloseTo(0.025, 10);
  });
});

describe('HC_EV_NM', () => {
  it('has the correct value for hc in eV·nm', () => {
    expect(HC_EV_NM).toBeCloseTo(1239.84, 1);
  });
});
