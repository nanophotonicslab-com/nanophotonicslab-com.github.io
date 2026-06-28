// Physics-invariant tests for the GSP library — ported from gsp_lib/tests/test_gsp.py.
// They check relationships (limits, symmetries, polarization identities), not hard numbers.
import { describe, it, expect } from 'vitest';
import {
  spCondition, gspCondition, gspAngleDeg, radiativeChannels, wavelengthRatioForAngle,
  dipoleDistribution, fourierModes, farFieldVector, farFieldAmplitudePhi0,
  sphericalComponents, stokes, dipoleOrientation, xPolarized, yPolarized, zPolarized, oriented,
  eVFromWavelengthNm, omegaFromEV, cabs, C_NM_S, type Vec3c,
} from './smith-purcell';

const rad = (deg: number) => deg * Math.PI / 180;
const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

// ── geometry / GSP ──
describe('GSP condition (Eq. 6)', () => {
  it('reduces to traditional SP at ℓ=0', () => {
    for (const n of [-2, -1, 1, 2, 3]) for (const aL of [0.05, 0.09, 0.25])
      expect(gspCondition(n, 0, 51, 0.1, aL)).toBeCloseTo(spCondition(n, 0.1, aL), 12);
  });
  it('Fig. 2a radiative modes: N=51, a/λ₀=0.09, β=0.1, n=1 → ℓ=1..9 radiate, 0 and 10 do not', () => {
    const s = (ell: number) => gspCondition(1, ell, 51, 0.1, 0.09);
    expect(Math.abs(s(0)) <= 1).toBe(false);
    expect(Math.abs(s(10)) <= 1).toBe(false);
    for (let ell = 1; ell <= 9; ell++) expect(Math.abs(s(ell)) <= 1).toBe(true);
  });
  it('angle is finite and near-normal for mode 5', () => {
    const a = gspAngleDeg(1, 5, 51, 0.1, 0.09);
    expect(Number.isFinite(a)).toBe(true);
    expect(Math.abs(Math.sin(rad(a)))).toBeLessThan(0.05);
  });
  it('wavelength_ratio_for_angle inverts Eq. (6)', () => {
    const N = 51, beta = 0.1, aL0 = 0.09, n = 1, ell = 4;
    for (const thetaDeg of [-30, -5, 15, 40]) {
      const u = wavelengthRatioForAngle(thetaDeg, n, ell, N, beta, aL0);
      const s = gspCondition(n, ell, N, beta, aL0 / u);
      expect(s).toBeCloseTo(Math.sin(rad(thetaDeg)), 9);
    }
  });
  it('all enumerated channels lie within the unit circle and match their angle', () => {
    for (const ch of radiativeChannels(51, 0.1, 0.09)) {
      expect(Math.abs(ch.sin)).toBeLessThanOrEqual(1);
      expect(Math.sin(rad(ch.thetaDeg))).toBeCloseTo(ch.sin, 9);
    }
  });
});

// ── distributions ──
describe('Engineered dipole distribution (Eqs. 7–8)', () => {
  it('DFT content: |p̃₀|=1, |p̃_ξ|=|p̃_{N−ξ}|=A/2, others ~0', () => {
    const N = 51, xi = 7, A = 0.8;
    const pl = fourierModes(dipoleDistribution(N, xi, A)).map(cabs);
    expect(pl[0]).toBeCloseTo(1.0, 9);
    expect(pl[xi]).toBeCloseTo(A / 2, 9);
    expect(pl[N - xi]).toBeCloseTo(A / 2, 9);
    pl.forEach((v, i) => { if (i !== 0 && i !== xi && i !== N - xi) expect(v).toBeLessThan(1e-9); });
  });
  it('mean magnitude equals the baseline p₀=1', () => {
    const p = dipoleDistribution(101, 9, 1.0);
    expect(p.reduce((a, b) => a + b, 0) / p.length).toBeCloseTo(1.0, 9);
  });
});

// ── far field ──
describe('Vector far field (Eq. 2)', () => {
  it('φ=0 vector magnitude equals the φ=0 amplitude helper', () => {
    const th = linspace(-Math.PI / 2, Math.PI / 2, 51);
    const vec = xPolarized(dipoleDistribution(51, 7));
    for (const t of th) {
      const f = farFieldVector(t, 0, vec, 0.09, 0.1);
      const mag = Math.sqrt(cabs(f[0]) ** 2 + cabs(f[1]) ** 2 + cabs(f[2]) ** 2);
      expect(mag).toBeCloseTo(farFieldAmplitudePhi0(t, vec, 0.09, 0.1), 10);
    }
  });
  it('peak amplitude of an x-pol steered array tracks cos(θ_target)/2 (Fig. 2d)', () => {
    const N = 51, beta = 0.1, aL0 = 0.09, th = linspace(-Math.PI / 2, Math.PI / 2, 4001);
    for (const xi of [3, 5, 7]) {
      const vec = xPolarized(dipoleDistribution(N, xi, 1.0));
      const peak = Math.max(...th.map(t => farFieldAmplitudePhi0(t, vec, aL0, beta) / N));
      const thT = rad(gspAngleDeg(1, xi, N, beta, aL0));
      expect(peak).toBeCloseTo(Math.cos(thT) / 2, 1);   // atol ~0.03
    }
  });
  it('|f| is invariant under a global phase on all dipoles', () => {
    const th = linspace(-Math.PI / 2, Math.PI / 2, 41);
    const mag = dipoleDistribution(51, 5);
    const a = th.map(t => farFieldAmplitudePhi0(t, xPolarized(mag), 0.09, 0.1));
    const phase = { re: Math.cos(1.234), im: Math.sin(1.234) };
    const vecPhased: Vec3c[] = mag.map(m => [
      { re: m * phase.re, im: m * phase.im }, { re: 0, im: 0 }, { re: 0, im: 0 }]);
    th.forEach((t, i) => expect(farFieldAmplitudePhi0(t, vecPhased, 0.09, 0.1)).toBeCloseTo(a[i], 10));
  });
});

// ── polarization ──
describe('Polarization (Stokes) identities', () => {
  const mag = Array(51).fill(1);
  it('px, py, pz all radiate', () => {
    for (const builder of [xPolarized, yPolarized, zPolarized]) {
      const f = farFieldVector(rad(35), rad(40), builder(mag), 0.09, 0.1);
      expect(cabs(f[0]) ** 2 + cabs(f[1]) ** 2 + cabs(f[2]) ** 2).toBeGreaterThan(0);
    }
  });
  it('φ=0 plane: x/z dipoles are pure p (f_φ=0), y dipoles pure s (f_θ=0)', () => {
    const th = rad(40);
    let { fTheta, fPhi } = sphericalComponents(th, 0, farFieldVector(th, 0, xPolarized(mag), 0.09, 0.1));
    expect(cabs(fPhi)).toBeLessThan(1e-9);
    ({ fTheta, fPhi } = sphericalComponents(th, 0, farFieldVector(th, 0, zPolarized(mag), 0.09, 0.1)));
    expect(cabs(fPhi)).toBeLessThan(1e-9);
    ({ fTheta, fPhi } = sphericalComponents(th, 0, farFieldVector(th, 0, yPolarized(mag), 0.09, 0.1)));
    expect(cabs(fTheta)).toBeLessThan(1e-9);
  });
  it('a real dipole orientation is linearly polarized (S3=0) everywhere', () => {
    const vec = oriented(mag, [{ re: 1, im: 0 }, { re: 0, im: 0 }, { re: 2, im: 0 }]);
    for (const th of [20, 50, 70].map(rad)) for (const ph of [0, 30, 90, 150].map(rad)) {
      const { fTheta, fPhi } = sphericalComponents(th, ph, farFieldVector(th, ph, vec, 0.09, 0.1));
      expect(Math.abs(stokes(fTheta, fPhi)[3])).toBeLessThan(1e-9);
    }
  });
  it('the electron elliptical dipole (i x + z) is circular off-plane, linear in-plane', () => {
    const vec = oriented(mag, [{ re: 0, im: 1 }, { re: 0, im: 0 }, { re: 1, im: 0 }]);
    let { fTheta, fPhi } = sphericalComponents(rad(30), 0, farFieldVector(rad(30), 0, vec, 0.09, 0.1));
    expect(Math.abs(stokes(fTheta, fPhi)[3])).toBeLessThan(1e-9);
    ({ fTheta, fPhi } = sphericalComponents(rad(30), rad(45), farFieldVector(rad(30), rad(45), vec, 0.09, 0.1)));
    const [S0, , , S3] = stokes(fTheta, fPhi);
    expect(Math.abs(S3 / S0)).toBeGreaterThan(0.2);
  });
  it('a coherent dipole field is fully polarized: S1²+S2²+S3² = S0²', () => {
    const vec = oriented(mag, [{ re: 0, im: 1 }, { re: 0.3, im: 0 }, { re: 1, im: 0 }]);
    for (const th of [25, 55].map(rad)) for (const ph of [10, 70, 130].map(rad)) {
      const { fTheta, fPhi } = sphericalComponents(th, ph, farFieldVector(th, ph, vec, 0.09, 0.1));
      const [S0, S1, S2, S3] = stokes(fTheta, fPhi);
      expect(S1 * S1 + S2 * S2 + S3 * S3).toBeCloseTo(S0 * S0, 6);
    }
  });
  it('electron dipole orientation has quadrature phase (Eq. 9): x imaginary, z real, no y', () => {
    const g = 1 / Math.sqrt(1 - 0.1 ** 2);
    const om = omegaFromEV(eVFromWavelengthNm(5000));
    const d = dipoleOrientation(om, 10.0, 0.1 * C_NM_S, g, false);
    expect(cabs(d[1])).toBe(0);
    expect(Math.abs(d[0].re)).toBeLessThan(1e-12 * cabs(d[0]));
    expect(Math.abs(d[2].im)).toBeLessThan(1e-12 * cabs(d[2]));
  });
});
