// Invariants for the VO₂ optical model (ported from gsp_lib/materials/vo2.py).
import { describe, it, expect } from 'vitest';
import { epsInsulating, epsMetallic, bruggeman, epsVO2, fmFromTemperature, temperatureFromFm, VO2_TM, VO2_DT } from './vo2';

describe('VO₂ phase-change driving law (Eq. S21)', () => {
  it('fm(T_M) = 1/2 and is a symmetric sigmoid', () => {
    expect(fmFromTemperature(VO2_TM)).toBeCloseTo(0.5, 12);
    expect(fmFromTemperature(VO2_TM + VO2_DT)).toBeCloseTo(1 / (1 + Math.exp(-1)), 12);
    expect(fmFromTemperature(VO2_TM + 3) + fmFromTemperature(VO2_TM - 3)).toBeCloseTo(1, 12);
  });
  it('temperatureFromFm inverts fmFromTemperature', () => {
    for (const T of [330, 339, 341, 343, 350]) expect(temperatureFromFm(fmFromTemperature(T))).toBeCloseTo(T, 9);
  });
});

describe('Bruggeman effective medium (Eq. 18)', () => {
  const Es = [0.2, 0.5, 1.0, 2.0, 3.5];
  it('reduces to the pure insulating phase at fm=0 and metallic phase at fm=1', () => {
    for (const E of Es) {
      const i = epsInsulating(E), m = epsMetallic(E);
      const b0 = bruggeman(i, m, 0), b1 = bruggeman(i, m, 1);
      expect(b0.re).toBeCloseTo(i.re, 9); expect(b0.im).toBeCloseTo(i.im, 9);
      expect(b1.re).toBeCloseTo(m.re, 9); expect(b1.im).toBeCloseTo(m.im, 9);
    }
  });
  it('mixed-phase permittivity is passive (Im ε ≥ 0)', () => {
    for (const E of Es) for (const fm of [0.1, 0.3, 0.5, 0.7, 0.9]) expect(epsVO2(E, fm).im).toBeGreaterThanOrEqual(-1e-9);
  });
  it('the metal develops a Drude (negative-Re) response in the IR as fm grows', () => {
    // at 0.3 eV the metallic phase is Drude-like (Re ε < 0); the insulator is positive
    expect(epsMetallic(0.3).re).toBeLessThan(0);
    expect(epsInsulating(0.3).re).toBeGreaterThan(0);
    expect(epsVO2(0.3, 1).re).toBeLessThan(epsVO2(0.3, 0).re);
  });
});
