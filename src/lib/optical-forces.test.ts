import { describe, it, expect } from 'vitest';
import { computeSphereCoeffs, computeAngularPattern, mieAt } from './mie';
import {
  asymmetryParameter, pressureCrossSection, radiationForceN, rayleighTrap, C_M_S,
} from './optical-forces';

const diel = (n: number, k = 0) => (_l: number): [number, number] => [n, k];

/** Independent g: ∫I(θ)cosθ sinθ dθ / ∫I(θ) sinθ dθ over the unpolarized pattern. */
function gFromPattern(nP: number, kP: number, radiusNm: number, nHost: number, lambdaNm: number): number {
  const co = computeSphereCoeffs(diel(nP, kP), radiusNm, nHost, lambdaNm);
  const pat = computeAngularPattern(co, 721);
  let num = 0, den = 0;
  for (let i = 0; i < pat.theta.length - 1; i++) {
    const dth = pat.theta[i + 1] - pat.theta[i];
    const I = (pat.iPerp[i] + pat.iPar[i]) / 2;
    num += I * Math.cos(pat.theta[i]) * Math.sin(pat.theta[i]) * dth;
    den += I * Math.sin(pat.theta[i]) * dth;
  }
  return num / den;
}

describe('asymmetryParameter', () => {
  it('matches the angular-pattern integral (independent formulation)', () => {
    for (const [r, lam] of [[60, 550], [150, 550], [40, 800]] as const) {
      const co = computeSphereCoeffs(diel(1.5, 0.05), r, 1.33, lam);
      const gCoeff = asymmetryParameter(co);
      const gInt = gFromPattern(1.5, 0.05, r, 1.33, lam);
      expect(Math.abs(gCoeff - gInt)).toBeLessThan(2e-3);
    }
  });

  it('vanishes in the Rayleigh limit and grows forward for large spheres', () => {
    const tiny = asymmetryParameter(computeSphereCoeffs(diel(1.5), 5, 1.33, 800));
    const large = asymmetryParameter(computeSphereCoeffs(diel(1.5), 300, 1.33, 550));
    expect(Math.abs(tiny)).toBeLessThan(0.01);
    expect(large).toBeGreaterThan(0.5);
    expect(large).toBeLessThan(1);
  });
});

describe('radiation pressure', () => {
  it('σ_pr is bounded by σ_abs and σ_ext', () => {
    const r = 80, lam = 550, nH = 1.33;
    const { csca, cext, cabs } = mieAt(1.5, 0.2, nH, r, lam);
    const g = asymmetryParameter(computeSphereCoeffs(diel(1.5, 0.2), r, nH, lam));
    const spr = pressureCrossSection(cext, csca, g);
    expect(spr).toBeLessThanOrEqual(cext);
    expect(spr).toBeGreaterThanOrEqual(cabs);
  });

  it('reduces to σ_ext in the Rayleigh limit', () => {
    const { csca, cext } = mieAt(1.5, 0, 1.33, 5, 800);
    const g = asymmetryParameter(computeSphereCoeffs(diel(1.5), 5, 1.33, 800));
    expect(Math.abs(pressureCrossSection(cext, csca, g) - cext) / cext).toBeLessThan(1e-3);
  });

  it('force has the right scale: ~fN–pN for typical tweezers numbers', () => {
    // 100 nm polystyrene-like sphere, 10 mW in ~µm² → I ~ 1e10 W/m²
    const { csca, cext } = mieAt(1.57, 0, 1.33, 100, 1064);
    const g = asymmetryParameter(computeSphereCoeffs(diel(1.57), 100, 1.33, 1064));
    const F = radiationForceN(pressureCrossSection(cext, csca, g), 1e10, 1.33);
    expect(F).toBeGreaterThan(1e-16);
    expect(F).toBeLessThan(1e-10);
    // dimensional cross-check: F = n σ I / c exactly
    expect(F).toBeCloseTo((1.33 * pressureCrossSection(cext, csca, g) * 1e-18 * 1e10) / C_M_S, 25);
  });
});

describe('rayleighTrap', () => {
  const base = { nParticle: 1.57, radiusNm: 60, nHost: 1.33, lambdaNm: 1064, powerMw: 50, waistUm: 0.6 };

  it('traps a high-index particle with sensible stiffness and depth', () => {
    const t = rayleighTrap(base);
    expect(t.cmFactor).toBeGreaterThan(0);
    expect(t.trapDepthKbT).toBeGreaterThan(0);
    expect(t.kappaRPnPerUm).toBeGreaterThan(0.1);
    expect(t.kappaRPnPerUm).toBeLessThan(1e4);
    // transverse confinement is stiffer than axial for a weakly focused beam
    expect(t.kappaRPnPerUm).toBeGreaterThan(t.kappaZPnPerUm);
    expect(t.dipoleValid).toBe(true);
  });

  it('repels a low-index particle (bubble): negative CM factor and depth', () => {
    const t = rayleighTrap({ ...base, nParticle: 1.0 });
    expect(t.cmFactor).toBeLessThan(0);
    expect(t.trapDepthKbT).toBeLessThan(0);
  });

  it('scales linearly with power and cubically with radius', () => {
    const t1 = rayleighTrap(base);
    const t2 = rayleighTrap({ ...base, powerMw: 100 });
    const t3 = rayleighTrap({ ...base, radiusNm: 120 });
    expect(t2.kappaRPnPerUm / t1.kappaRPnPerUm).toBeCloseTo(2, 6);
    expect(t3.trapDepthKbT / t1.trapDepthKbT).toBeCloseTo(8, 6);
  });

  it('flags the dipole approximation when the particle is too large', () => {
    expect(rayleighTrap({ ...base, radiusNm: 300 }).dipoleValid).toBe(false);
  });
});
