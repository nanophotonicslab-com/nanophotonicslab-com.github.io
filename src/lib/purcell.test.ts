import { describe, it, expect } from 'vitest';
import { purcellAt, modifiedQuantumYield, modifiedLifetime } from './purcell';
import { silverEpsilon } from './dipole-decay';

const GLASS = { re: 2.25, im: 0 };

describe('purcellAt', () => {
  it('approaches free space far from a small sphere', () => {
    // NOTE: the multipole series needs lmax ≳ k·r' to converge — at gap 1 µm
    // and λ = 650 nm, k·r' ≈ 16, so lmax = 40 has ample margin.
    for (const o of ['radial', 'tangential', 'isotropic'] as const) {
      const f = purcellAt(650, GLASS, 1, 20, 1000, o, 40);
      expect(Math.abs(f.fTot - 1)).toBeLessThan(0.05);
      expect(Math.abs(f.fRad - 1)).toBeLessThan(0.05);
    }
  });

  it('isotropic is the (radial + 2·tangential)/3 average', () => {
    const rad = purcellAt(650, silverEpsilon(650), 1.33, 50, 10, 'radial');
    const tan = purcellAt(650, silverEpsilon(650), 1.33, 50, 10, 'tangential');
    const iso = purcellAt(650, silverEpsilon(650), 1.33, 50, 10, 'isotropic');
    expect(iso.fTot).toBeCloseTo((rad.fTot + 2 * tan.fTot) / 3, 10);
    expect(iso.fRad).toBeCloseTo((rad.fRad + 2 * tan.fRad) / 3, 10);
  });

  it('the non-radiative channel diverges ~1/d³ approaching a metal surface', () => {
    const near = purcellAt(650, silverEpsilon(650), 1.33, 50, 1, 'radial');
    const far = purcellAt(650, silverEpsilon(650), 1.33, 50, 50, 'radial');
    expect(near.fTot).toBeGreaterThan(10);
    expect(near.fNr).toBeGreaterThan(0);
    expect(near.fNr).toBeGreaterThan(50 * far.fNr);
  });

  it('a lossless dielectric sphere has negligible non-radiative channel', () => {
    const f = purcellAt(650, GLASS, 1, 60, 10, 'isotropic');
    expect(Math.abs(f.fNr)).toBeLessThan(1e-6 * Math.max(1, f.fTot));
  });
});

describe('emitter bookkeeping', () => {
  const FREE = { fTot: 1, fRad: 1, fNr: 0 };

  it('free space leaves q₀ and τ₀ unchanged', () => {
    expect(modifiedQuantumYield(FREE, 0.7)).toBeCloseTo(0.7, 12);
    expect(modifiedLifetime(FREE, 0.7, 12)).toBeCloseTo(12, 12);
  });

  it('a perfect emitter (q₀=1) has q′ = fRad/fTot and τ′ = τ₀/fTot', () => {
    const f = { fTot: 8, fRad: 5, fNr: 3 };
    expect(modifiedQuantumYield(f, 1)).toBeCloseTo(5 / 8, 12);
    expect(modifiedLifetime(f, 1, 10)).toBeCloseTo(10 / 8, 12);
  });

  it('Purcell enhancement can raise the quantum yield of a poor emitter', () => {
    // q0 = 0.1 emitter with a good antenna (fRad 20 of fTot 25)
    const q = modifiedQuantumYield({ fTot: 25, fRad: 20, fNr: 5 }, 0.1);
    expect(q).toBeGreaterThan(0.1);
  });
});
