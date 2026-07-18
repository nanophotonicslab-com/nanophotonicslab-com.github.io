/**
 * Regression tests for the 2026-07-18 adversarial physics audit.
 * Each test encodes a bug found by numerical verification agents so it can
 * never silently return.
 */
import { describe, it, expect } from 'vitest';
import { mieAt, computeMie, computeSphereCoeffs, computeNearField } from './mie';
import { decayRatesAt } from './dipole-decay';
import { computeElectronSphereSpectrum } from './electron-sphere';
import { fitMieRadius } from './fit';

const diel = (n: number, k = 0) => (_l: number): [number, number] => [n, k];

describe('near-field internal coefficients (c/d swap, audit M1)', () => {
  it('quasistatic interior |E|² matches |3εh/(εp+2εh)|²', () => {
    const nP = 1.5, kP = 0.1, nH = 1.33, lam = 550, R = 5;
    const co = computeSphereCoeffs(diel(nP, kP), R, nH, lam);
    const k = (2 * Math.PI * nH) / (lam * 1e-9);
    const nf = computeNearField(co, k, R, 41, 2 * R);
    // expected uniform interior field: |3εh/(εp+2εh)|², εp complex
    const eH = nH * nH;
    const epR = nP * nP - kP * kP, epI = 2 * nP * kP;
    const dR = epR + 2 * eH, dI = epI;
    const expected = (9 * eH * eH) / (dR * dR + dI * dI);
    // average pixels well inside the sphere (r < 0.6R)
    let sum = 0, cnt = 0;
    for (let iz = 0; iz < 41; iz++) {
      for (let ix = 0; ix < 41; ix++) {
        const x = -2 * R + (ix * 4 * R) / 40, z = -2 * R + (iz * 4 * R) / 40;
        if (Math.hypot(x, z) < 0.6 * R) { sum += nf.grid[iz * 41 + ix]; cnt++; }
      }
    }
    expect(cnt).toBeGreaterThan(4);
    expect(Math.abs(sum / cnt - expected) / expected).toBeLessThan(0.05);
  });

  it('no spurious hotspots near the sphere centre (ψ Miller, audit M2)', () => {
    const co = computeSphereCoeffs(diel(1.5, 0.1), 60, 1.33, 550);
    const k = (2 * Math.PI * 1.33) / 550e-9;
    for (const grid of [150, 151]) { // even and odd (odd hits the clamped centre)
      const nf = computeNearField(co, k, 60, grid, 180);
      let mx = 0;
      for (const v of nf.grid) {
        expect(Number.isFinite(v)).toBe(true);
        if (v > mx) mx = v;
      }
      expect(mx).toBeLessThan(100); // was 2e5–6.5e17 before the fix
    }
  });
});

describe('large size parameters (scratch overflow, audit M3)', () => {
  it('computeMie stays finite and matches mieAt beyond x ≈ 91', () => {
    const r = 8000, lam0 = 540, lam1 = 560;
    const spec = computeMie(diel(1.5, 0.01), r, 1.0, lam0, lam1, 5);
    for (let i = 0; i < 5; i++) {
      expect(Number.isFinite(spec.csca[i])).toBe(true);
      const ref = mieAt(1.5, 0.01, 1.0, r, spec.lambda[i]);
      expect(Math.abs(spec.csca[i] - ref.csca) / ref.csca).toBeLessThan(1e-9);
    }
    const co = computeSphereCoeffs(diel(1.5, 0.01), r, 1.0, 550);
    expect(Number.isFinite(co.aRe[co.nmax - 1])).toBe(true);
  });
});

describe('dipole-decay high-lmax NaN wall (audit E1)', () => {
  it('stays finite and converged at lmax 120 near a large sphere', () => {
    const eps = { re: -15, im: 1 };
    const r100 = decayRatesAt(600, eps, 1.33, 100, 104, 100);
    const r120 = decayRatesAt(600, eps, 1.33, 100, 104, 120);
    for (const v of Object.values(r120)) expect(Number.isFinite(v)).toBe(true);
    expect(r120.gammaPar).toBeGreaterThan(0);
    // series has plateaued: raising lmax past the old NaN wall changes nothing
    expect(Math.abs(r120.gammaPar - r100.gammaPar) / r100.gammaPar).toBeLessThan(1e-2);
  });
});

describe('electron-sphere stability (audits E3, E5)', () => {
  it('high multipoles at low energy no longer blow up (sphJ Miller)', () => {
    const spec = computeElectronSphereSpectrum(
      diel(0.4, 2.6),
      { radiusNm: 50, beta: 0.5867, impactNm: 60, lmax: 14, qCutNm: 0.71, nz: 24 },
      0.8, 1.6, 15,
    );
    for (let i = 0; i < spec.energy.length; i++) {
      expect(Number.isFinite(spec.eelsTotal[i])).toBe(true);
      expect(spec.eelsTotal[i]).toBeGreaterThanOrEqual(0);  // aloof invariant
      expect(Math.abs(spec.eelsTotal[i])).toBeLessThan(1e3); // was ~1e10 garbage
    }
  });

  it('survives the exact Cherenkov threshold εβ² = 1', () => {
    // n = 2, β = 0.5 → εβ² = 1 exactly; the whole spectrum used to turn NaN
    const spec = computeElectronSphereSpectrum(
      diel(2, 0),
      { radiusNm: 50, beta: 0.5, impactNm: 20, lmax: 6, qCutNm: 0.71, nz: 24 },
      1, 3, 9,
    );
    for (let i = 0; i < spec.energy.length; i++) {
      expect(Number.isFinite(spec.eelsTotal[i])).toBe(true);
      expect(Number.isFinite(spec.clTotal[i])).toBe(true);
    }
  });
});

describe('fit multi-start (audit M4)', () => {
  it('escapes the sharp-resonance local minimum (n=4, r=52, wide bounds)', () => {
    const N = 120;
    const x = new Float64Array(N), y = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      x[i] = 420 + (900 - 420) * i / (N - 1);
      y[i] = mieAt(4.0, 0, 1.33, 52, x[i]).cext;
    }
    const fit = fitMieRadius(diel(4.0), 1.33, x, y, 'cext', 20, 500);
    expect(Math.abs(fit.radiusNm - 52)).toBeLessThan(1);
    expect(fit.r2).toBeGreaterThan(0.999);
  });
});
