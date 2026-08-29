import { describe, it, expect } from 'vitest';
import { mieAt, interpolateNK, HC_EV_NM } from './mie';
import { MATERIALS } from './materials';

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

  it('interpolates with a cubic spline on eps(eV), matching SciPy', () => {
    // Not the linear midpoint 1.55: the spline runs on eps against photon
    // energy, so that this and the Pyodide BEM solver read one table the same
    // way. Reference from scipy.interpolate.CubicSpline (bc_type default).
    const [n, k] = interpolateNK(data, 600);
    expect(n).toBeCloseTo(1.547313778528, 9);
    expect(k).toBeCloseTo(0.024679868124, 9);
  });

  it('leaves no corner at the gold interband edge', () => {
    // The Johnson and Christy table samples gold every ~20 nm while dn/dlam
    // changes by a factor of three across the node at 471.4 nm. Under linear
    // interpolation that node was a corner, large enough to show up as a
    // visible kink in a computed spectrum; the spline must not reintroduce it.
    const nAt = (lam: number) => interpolateNK(MATERIALS.au.data, lam)[0];
    const h = 0.01;
    const left = (nAt(471.4) - nAt(471.4 - h)) / h;
    const right = (nAt(471.4 + h) - nAt(471.4)) / h;
    expect(Math.abs(right - left)).toBeLessThan(1e-3);
  });

  it('agrees with the BEM solver tables for gold', () => {
    // (n, k) of gold from a cubic spline on eps(eV) through the J&C table,
    // computed with SciPy exactly as nplab_bem.material.tabulated does. Both
    // codes must return the same material for the BEM-against-Mie benchmark
    // to measure the method rather than the interpolation.
    const expected: [number, number, number][] = [
      [400, 1.4694496116, 1.9537048915],
      [450, 1.3828728275, 1.9162845312],
      [500, 0.9679225585, 1.8538793132],
      [530, 0.5329061763, 2.2073158458],
      [600, 0.2442870409, 3.0797864901],
      [700, 0.1296335204, 4.0635335800],
      [800, 0.1555117706, 4.9069893038],
    ];
    for (const [lam, n, k] of expected) {
      const [gotN, gotK] = interpolateNK(MATERIALS.au.data, lam);
      expect(gotN).toBeCloseTo(n, 6);
      expect(gotK).toBeCloseTo(k, 6);
    }
  });

  it('never returns a negative k', () => {
    // A spline through a table whose absorption falls to zero can undershoot;
    // gain instead of loss would be unphysical.
    const transparent = [
      [0.4, 1.5, 0.5],
      [0.5, 1.5, 0.05],
      [0.6, 1.5, 0.0],
      [0.7, 1.5, 0.0],
      [0.8, 1.5, 0.0],
    ];
    for (let lam = 400; lam <= 800; lam += 2) {
      expect(interpolateNK(transparent, lam)[1]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('HC_EV_NM', () => {
  it('has the correct value for hc in eV·nm', () => {
    expect(HC_EV_NM).toBeCloseTo(1239.84, 1);
  });
});
