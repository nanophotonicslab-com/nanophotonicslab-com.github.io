// Invariants for the dipole-INSIDE-sphere decay rates (Kim 1988 Figs. 2, 4 & 5).
// Ported from the Spheres_dipoles validation package.
import { describe, it, expect } from 'vitest';
import { insideDecay, insideCoeffs, csqrt, type Cx } from './dipole-inside';

describe('Dipole in a vacuum cavity — Kim (1988) Fig. 2', () => {
  // Cavity: dipole in a vacuum cavity (ε₂ = 1) surrounded by a dielectric ε₁ = 2.16;
  // a = 400 nm, λ = 413.3 nm, ℓ_max = 30. Purely radiative (total = radiative).
  const VAC: Cx = { re: 1, im: 0 };
  const OUTER: Cx = { re: 2.16, im: 0 };
  const cav = (dOverA: number) => insideDecay(413.3, VAC, 400, dOverA * 400, 30, undefined, OUTER);

  it('matches the published Fig. 2 reference table (radial γ_⊥ and tangential γ_∥)', () => {
    // [d/a, radial, tangential] from the README reference-values table
    const TABLE: [number, number, number][] = [
      [0.20, 1.269, 1.118],
      [0.35, 1.045, 0.906],
      [0.50, 0.939, 1.012],
      [0.70, 1.062, 1.079],
      [0.85, 1.346, 1.007],
      [0.95, 1.751, 1.198],
      [0.99, 2.019, 1.393],
    ];
    for (const [da, radial, tangential] of TABLE) {
      const r = cav(da);
      expect(r.perpTot).toBeCloseTo(radial, 2);
      expect(r.parTot).toBeCloseTo(tangential, 2);
    }
  });
  it('centre value = 1 + Re(E₁) ≈ 1.447, both orientations equal', () => {
    const r = insideDecay(413.3, VAC, 400, 1e-3, 30, undefined, OUTER);
    expect(r.perpTot).toBeCloseTo(1.447, 2);
    expect(r.perpTot).toBeCloseTo(r.parTot, 3);
  });
  it('radial dips below 1 near mid-radius then climbs above 2 at the wall', () => {
    expect(cav(0.5).perpTot).toBeLessThan(1);       // ≈ 0.939
    expect(cav(0.99).perpTot).toBeGreaterThan(1.9); // ≈ 2.019
  });
});

describe('Dipole inside a sphere — Kim (1988) Figs. 4 & 5', () => {
  // Fig. 4: transparent sphere ε₂ = 2.16, a = 400 nm, λ = 413.3 nm, ℓ_max = 30.
  const EPS_T: Cx = { re: 2.16, im: 0 };
  it("Chew's identity: transparent total = radiative (no absorption)", () => {
    for (const dOverA of [0.2, 0.5, 0.8, 0.95]) {
      const r = insideDecay(413.3, EPS_T, 400, dOverA * 400, 30);
      expect(r.perpTot).toBeCloseTo(r.perpRad, 6);
      expect(r.parTot).toBeCloseTo(r.parRad, 6);
      expect(r.perpNr).toBeCloseTo(0, 6);
      expect(r.parNr).toBeCloseTo(0, 6);
    }
  });
  it('transparent centre value is 1 + Re(E₁) for both orientations', () => {
    const eps1: Cx = { re: 1, im: 0 };
    const k0 = 2 * Math.PI / 413.3;
    const rho1 = { re: k0 * 400, im: 0 };
    const rho2 = { re: csqrt(EPS_T).re * k0 * 400, im: csqrt(EPS_T).im * k0 * 400 };
    const { E } = insideCoeffs(eps1, EPS_T, rho1, rho2, 30);
    const centre = 1 + E[1].re;   // Kim's closed form at d=0 (only ℓ=1 survives)
    // insideDecay divides by y₂ = k₂d, so evaluate the limit at a tiny offset
    const r = insideDecay(413.3, EPS_T, 400, 0.001 * 400, 30);
    expect(r.perpTot).toBeCloseTo(centre, 3);
    expect(r.parTot).toBeCloseTo(centre, 3);
  });

  // Fig. 5: dissipative Ag sphere ε₂ = −4.42 + 0.73i, a = 100 nm, ℓ_max = 10.
  const EPS_D: Cx = { re: -4.42, im: 0.73 };
  it('dissipative sphere: total ≥ 1, radiative > 0, non-radiative dominates', () => {
    const f = 6.88e-8;
    for (const dOverA of [0.3, 0.6, 0.9]) {
      const r = insideDecay(413.3, EPS_D, 100, dOverA * 100, 10, f);
      expect(r.perpTot).toBeGreaterThan(1);
      expect(r.perpRad).toBeGreaterThan(0);
      expect(r.perpRad).toBeLessThan(r.perpNr);   // non-radiative dominates
    }
  });
});
