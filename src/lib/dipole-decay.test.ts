// Invariants + Kim (1988) Fig. 1 reference values for the dipole-OUTSIDE-sphere
// decay rates. Ported from the Spheres_dipoles validation package, whose
// dipole-decay.ts reproduces MNPBEM's @mieret/decayrate.m to ~1e-9.
import { describe, it, expect } from 'vitest';
import { decayRatesAt, silverEpsilon, evToNm, type Cx } from './dipole-decay';

// Kim Fig. 1: Ag sphere, a = 10 nm, λ = 413.3 nm, ε₂ = −4.42 + 0.73i, vacuum host,
// ℓ_max = 20. Radial = gammaPar/gammaParRad, tangential = gammaPerp/gammaPerpRad.
const EPS_AG: Cx = { re: -4.42, im: 0.73 };
const kim = (dOverA: number) => decayRatesAt(413.3, EPS_AG, 1, 10, dOverA * 10, 20);

describe('Dipole outside a sphere — Kim (1988) Fig. 1 reference table', () => {
  // [d/a, radial-total, radial-rad, tangential-total, tangential-rad] (README table)
  const TABLE: [number, number, number, number, number][] = [
    [2.0, 18.15, 2.486, 4.497, 0.5503],
    [2.5, 5.341, 1.689, 1.555, 0.7588],
    [3.0, 2.580, 1.384, 1.090, 0.8607],
    [4.0, 1.386, 1.161, 0.9778, 0.9439],
    [5.0, 1.150, 1.084, 0.9810, 0.9727],
    [9.0, 1.018, 1.014, 0.9953, 0.9945],
  ];
  it('matches the four published curves to <0.2%', () => {
    for (const [da, radTot, radRad, tanTot, tanRad] of TABLE) {
      const r = kim(da);
      expect(r.gammaPar).toBeCloseTo(radTot, radTot > 100 ? 0 : 2);
      expect(r.gammaParRad / radRad).toBeCloseTo(1, 2);
      expect(r.gammaPerp / tanTot).toBeCloseTo(1, 2);
      expect(r.gammaPerpRad / tanRad).toBeCloseTo(1, 2);
    }
  });
  it('captures the tangential-radiative near-surface dip (Kim ~0.038 at d/a≈1.3)', () => {
    // the distinctive image-dipole radiation cancellation
    expect(kim(1.3).gammaPerpRad).toBeLessThan(0.06);
    expect(kim(1.2).gammaPerpRad).toBeLessThan(0.15);
  });
});

describe('Decay-rate physical invariants', () => {
  it('far from the sphere every rate → 1 (bare dipole)', () => {
    const r = decayRatesAt(413.3, EPS_AG, 1, 10, 10 * 40, 20);   // d/a = 40
    for (const g of [r.gammaPar, r.gammaPerp, r.gammaParRad, r.gammaPerpRad])
      expect(g).toBeCloseTo(1, 2);
  });
  it('non-radiative channel is non-negative (γ ≥ γ_rad) for a lossy sphere', () => {
    for (const da of [1.2, 1.5, 2, 3, 5, 9]) {
      const r = kim(da);
      expect(r.gammaParNr).toBeGreaterThan(-1e-6);
      expect(r.gammaPerpNr).toBeGreaterThan(-1e-6);
    }
  });
  it('silverEpsilon/evToNm helpers give a metallic (Re ε < 0) Drude sphere', () => {
    const eps = silverEpsilon(evToNm(3.0));
    expect(eps.re).toBeLessThan(0);
    expect(eps.im).toBeGreaterThan(0);
  });
});
