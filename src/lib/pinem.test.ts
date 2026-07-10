// Physics-invariant tests for the PINEM library — checks Bessel values, the
// Feist et al. (2015) reference numbers (120 keV / 800 nm), and closure/limits.
import { describe, it, expect } from 'vitest';
import {
  besselJ, besselJ0toN, sidebandProbabilities, pinemSpectrum,
  beamPhotonEV, beamPeriodS, beamVelocity, beamGamma, talbotDistanceM,
  betaToEnergyEV, energyEVToBeta, comovingDeltaK, C_LIGHT, type Beam,
} from './pinem';

const BEAM: Beam = { energyEV: 120e3, wavelengthNm: 800 };

describe('Bessel functions of the first kind', () => {
  it('matches known J_n(1) values', () => {
    expect(besselJ(0, 1)).toBeCloseTo(0.7651976866, 8);
    expect(besselJ(1, 1)).toBeCloseTo(0.4400505857, 8);
    expect(besselJ(2, 1)).toBeCloseTo(0.1149034849, 8);
    expect(besselJ(3, 1)).toBeCloseTo(0.0195633540, 8);
  });
  it('matches known J_n(10) values (large argument)', () => {
    expect(besselJ(0, 10)).toBeCloseTo(-0.2459357645, 7);
    expect(besselJ(1, 10)).toBeCloseTo(0.0434727462, 7);
    expect(besselJ(5, 10)).toBeCloseTo(-0.2340615282, 7);
  });
  it('parity: J_{-n} = (−1)^n J_n', () => {
    expect(besselJ(-1, 3.4)).toBeCloseTo(-besselJ(1, 3.4), 10);
    expect(besselJ(-2, 3.4)).toBeCloseTo(besselJ(2, 3.4), 10);
  });
  it('J_0(0)=1 and J_n(0)=0 for n≥1', () => {
    const J = besselJ0toN(0, 5);
    expect(J[0]).toBe(1);
    for (let n = 1; n <= 5; n++) expect(J[n]).toBe(0);
  });
});

describe('Sideband probabilities P_N = J_N(2|g|)²', () => {
  it('closure Σ_N P_N → 1 for a range of |g|', () => {
    for (const g of [0.5, 1, 2, 3, 5, 5.7]) {
      const { P } = sidebandProbabilities(g, 60);
      const sum = P.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 6);
    }
  });
  it('|g|=0 → all population in N=0', () => {
    const { N, P } = sidebandProbabilities(0, 10);
    const i0 = [...N].indexOf(0);
    expect(P[i0]).toBeCloseTo(1, 12);
    expect(P.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
  it('symmetric in ±N', () => {
    const { N, P } = sidebandProbabilities(3.3, 20);
    const idx = (n: number) => [...N].indexOf(n);
    for (const n of [1, 2, 5, 9]) expect(P[idx(n)]).toBeCloseTo(P[idx(-n)], 12);
  });
});

describe('Beam kinematics (Feist 120 keV / 800 nm)', () => {
  it('photon energy ħω = 1.55 eV (sideband spacing)', () => {
    expect(beamPhotonEV(BEAM)).toBeCloseTo(1.5498, 3);
  });
  it('optical period T = 2.669 fs', () => {
    expect(beamPeriodS(BEAM) * 1e15).toBeCloseTo(2.669, 2);
  });
  it('velocity v/c = 0.5867 and γ = 1.2349', () => {
    expect(beamVelocity(BEAM) / C_LIGHT).toBeCloseTo(0.5867, 3);
    expect(beamGamma(BEAM)).toBeCloseTo(1.2349, 3);
  });
  it('Talbot distance z_T = 200.6 mm', () => {
    expect(talbotDistanceM(BEAM) * 1e3).toBeCloseTo(200.6, 0);
  });
  it('β ↔ energy round-trips', () => {
    for (const E of [30e3, 120e3, 200e3]) expect(betaToEnergyEV(energyEVToBeta(E))).toBeCloseTo(E, 2);
  });
});

describe('EELS spectrum', () => {
  it('peaks sit at N·ħω and the spectrum is peak-normalized', () => {
    const E = new Float64Array(4001); for (let i = 0; i < 4001; i++) E[i] = -20 + (40 * i) / 4000;
    const S = pinemSpectrum(3, E, 0.25, beamPhotonEV(BEAM), 40, true);
    let mx = 0; for (const v of S) if (v > mx) mx = v;
    expect(mx).toBeCloseTo(1, 6);
    // a peak near +ħω exists (a local maximum within ±0.1 eV of the spacing)
    const spacing = beamPhotonEV(BEAM);
    let near = 0; for (let i = 0; i < E.length; i++) if (Math.abs(E[i] - spacing) < 0.05) near = Math.max(near, S[i]);
    expect(near).toBeGreaterThan(0.3);
  });
});

describe('Co-moving dispersion', () => {
  it('exact and Talbot ΔK agree to <0.1% up to |N|=30', () => {
    const N = new Int32Array(2 * 30 + 1); for (let i = 0; i < N.length; i++) N[i] = i - 30;
    const ex = comovingDeltaK(N, BEAM, 'exact');
    const tb = comovingDeltaK(N, BEAM, 'talbot');
    for (let i = 0; i < N.length; i++) {
      if (Math.abs(N[i]) < 2) continue;   // near N=0 both ≈ 0
      const rel = Math.abs(ex[i] - tb[i]) / Math.abs(tb[i]);
      expect(rel).toBeLessThan(1e-3);
    }
  });
  it('Talbot ΔK is quadratic and negative: ΔK_N = −2πN²/z_T', () => {
    const N = new Int32Array([0, 1, 2, 3]);
    const tb = comovingDeltaK(N, BEAM, 'talbot');
    const zT = talbotDistanceM(BEAM);
    for (let i = 0; i < N.length; i++) expect(tb[i]).toBeCloseTo((-2 * Math.PI * N[i] * N[i]) / zT, 6);
  });
});
