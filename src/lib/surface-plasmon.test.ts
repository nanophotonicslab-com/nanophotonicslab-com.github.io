import { describe, it, expect } from 'vitest';
import {
  epsilonLossless,
  layerKappa,
  xRatioForPair,
  coth,
  dmdFilmResidual,
  singleInterfaceU,
  solveScalarRoots,
  type PlasmonLayer,
} from './surface-plasmon';

// Economou's figure 4 geometry, as carried by the tool's preset: a Drude metal
// film of hbar*wp = 9 eV between two vacuum half-spaces, 25 nm thick, which is
// kp*d = 1.140 in normalized units.
const WP = 9;
const HBARC_EV_NM = 197.3269804;
const METAL: PlasmonLayer = { epsInf: 1, wpEv: WP };
const VACUUM: PlasmonLayer = { epsInf: 1, wpEv: 0 };
const GLASS: PlasmonLayer = { epsInf: 2.25, wpEv: 0 };
const KPD = 25 * (WP / HBARC_EV_NM);

/**
 * Both branches of the DMD film at a given q, from the two field symmetries.
 * The sign of the exponential in equation (3.23a) selects the branch: the minus
 * sign carries the upper branch (Economou's I, above wp/sqrt2 over most of its
 * range) and the plus sign the lower one (his II, approaching wp/sqrt2 from
 * below). Both collapse onto the single-interface mode as the film thickens.
 */
function dmdBranches(q: number, kpThickness = KPD): { upper: number; lower: number } {
  const root = (sign: 1 | -1) => {
    const r = solveScalarRoots(
      u => dmdFilmResidual(q, u, METAL, VACUUM, WP, kpThickness, sign),
      1e-5, q * (1 - 1e-6), 1500,
    );
    expect(r.length).toBeGreaterThan(0);
    return sign > 0 ? r[r.length - 1] : r[0];
  };
  return { upper: root(-1), lower: root(1) };
}

describe('lossless Drude permittivity', () => {
  it('is epsInf for a dielectric at any energy', () => {
    expect(epsilonLossless(VACUUM, 1)).toBe(1);
    expect(epsilonLossless(GLASS, 5)).toBe(2.25);
  });

  it('crosses zero at the plasma energy and is negative below it', () => {
    expect(epsilonLossless(METAL, WP)).toBeCloseTo(0, 12);
    expect(epsilonLossless(METAL, WP / 2)).toBeCloseTo(-3, 12);
    expect(epsilonLossless(METAL, 2 * WP)).toBeCloseTo(0.75, 12);
  });
});

describe('transverse decay constant', () => {
  it('is null where the field is oscillatory rather than evanescent', () => {
    // Inside the light line of vacuum (q < u) there is no bound solution.
    expect(layerKappa(0.3, 0.9, VACUUM, WP)).toBeNull();
  });

  it('reduces to sqrt(q^2 - eps u^2) where it exists', () => {
    const q = 2, u = 0.5;
    expect(layerKappa(q, u, VACUUM, WP)).toBeCloseTo(Math.sqrt(4 - 0.25), 12);
    expect(layerKappa(q, u, GLASS, WP)).toBeCloseTo(Math.sqrt(4 - 2.25 * 0.25), 12);
  });

  it('grows without bound as q does, so modes localize at large wavevector', () => {
    const small = layerKappa(1, 0.5, VACUUM, WP)!;
    const large = layerKappa(10, 0.5, VACUUM, WP)!;
    expect(large).toBeGreaterThan(small);
    expect(large / 10).toBeCloseTo(1, 2);
  });
});

describe('coth', () => {
  it('matches 1/tanh away from the origin and is guarded at it', () => {
    expect(coth(1.3)).toBeCloseTo(1 / Math.tanh(1.3), 12);
    expect(coth(-2)).toBeCloseTo(1 / Math.tanh(-2), 12);
    expect(Number.isFinite(coth(0))).toBe(true);
    expect(coth(0)).toBeGreaterThan(0);
    expect(coth(-0.0)).toBeGreaterThan(0);
  });
});

describe('root finder', () => {
  it('finds every simple root of an analytic function', () => {
    const roots = solveScalarRoots(u => Math.sin(Math.PI * u), 0.5, 3.5, 400);
    expect(roots).toHaveLength(3);
    [1, 2, 3].forEach((expected, i) => expect(roots[i]).toBeCloseTo(expected, 9));
  });

  it('brackets to high precision, not just to the sampling step', () => {
    const roots = solveScalarRoots(u => u * u - 2, 0, 3, 12);
    expect(roots).toHaveLength(1);
    expect(roots[0]).toBeCloseTo(Math.SQRT2, 10);
  });

  it('does not invent a root across a gap where the residual is undefined', () => {
    // Jumps -1 -> +1 through an undefined window: no sign-change bracket exists.
    const roots = solveScalarRoots(u => (u < 1 ? -1 : u > 2 ? 1 : null), 0, 3, 300);
    expect(roots).toHaveLength(0);
  });

  it('merges duplicates and returns roots in ascending order', () => {
    const roots = solveScalarRoots(u => (u - 1) * (u - 1) * (u - 2), 0, 3, 500);
    expect(roots.every((r, i) => i === 0 || r > roots[i - 1])).toBe(true);
    expect(roots.filter(r => Math.abs(r - 1) < 1e-3).length).toBeLessThanOrEqual(1);
  });

  it('returns nothing for an inverted or degenerate interval', () => {
    expect(solveScalarRoots(u => u, 2, 1)).toEqual([]);
    expect(solveScalarRoots(u => u, 1, 1)).toEqual([]);
    expect(solveScalarRoots(u => u, Number.NaN, 1)).toEqual([]);
  });
});

describe('single interface (exact closed form)', () => {
  it('satisfies the dispersion relation kappa_m/eps_m + kappa_d/eps_d = 0', () => {
    for (const q of [0.5, 1, 2, 5]) {
      const u = singleInterfaceU(q, 1);
      const km = layerKappa(q, u, METAL, WP)!;
      const kd = layerKappa(q, u, VACUUM, WP)!;
      const epsM = epsilonLossless(METAL, u * WP);
      expect(km / epsM + kd / 1).toBeCloseTo(0, 9);
    }
  });

  it('approaches wp/sqrt(1 + eps_d) from below as q grows', () => {
    for (const [epsD, limit] of [[1, Math.SQRT1_2], [2.25, 1 / Math.sqrt(3.25)]] as const) {
      const far = singleInterfaceU(400, epsD);
      expect(far).toBeLessThan(limit);
      expect(far).toBeCloseTo(limit, 4);
      expect(singleInterfaceU(2, epsD)).toBeLessThan(far);
    }
  });

  it('stays below the light line of the dielectric', () => {
    for (const q of [0.2, 1, 3, 10]) expect(singleInterfaceU(q, 1)).toBeLessThan(q);
  });
});

describe('Economou figure 4 — Drude film between vacuum half-spaces', () => {
  it('yields two distinct bound branches per wavevector, both below the light line', () => {
    const { upper, lower } = dmdBranches(1.5);
    expect(upper).toBeGreaterThan(lower);
    expect(upper).toBeLessThan(1.5);
    expect(lower).toBeGreaterThan(0);
  });

  it('brackets the single-interface mode once both branches have formed', () => {
    for (const q of [1.2, 1.5, 2.0, 2.5]) {
      const { upper, lower } = dmdBranches(q);
      expect(upper).toBeGreaterThan(Math.SQRT1_2);
      expect(lower).toBeLessThan(Math.SQRT1_2);
    }
  });

  it('keeps the upper branch below wp/sqrt2 at small q, where it leaves the light line', () => {
    expect(dmdBranches(0.8).upper).toBeLessThan(Math.SQRT1_2);
  });

  it('reproduces the peak of branch I reported in the article', () => {
    // The article quotes a maximum of u = 0.7208 near q = 1.4 for kp*d = 1.140.
    let best = 0, argmax = 0;
    for (let q = 1.0; q <= 2.0; q += 0.005) {
      const u = dmdBranches(q).upper;
      if (u > best) { best = u; argmax = q; }
    }
    expect(best).toBeCloseTo(0.7208, 3);
    expect(argmax).toBeGreaterThan(1.3);
    expect(argmax).toBeLessThan(1.6);
  });

  it('collapses onto the single-interface mode as the film thickens', () => {
    const q = 2;
    const exact = singleInterfaceU(q, 1);
    for (const [kpd, tol] of [[5, 2e-3], [12, 1e-6], [25, 1e-9]] as const) {
      const { upper, lower } = dmdBranches(q, kpd);
      expect(Math.abs(upper - exact)).toBeLessThan(tol);
      expect(Math.abs(lower - exact)).toBeLessThan(tol);
    }
  });

  it('splits the two branches further apart as the film thins', () => {
    const q = 2;
    const gap = (kpd: number) => {
      const { upper, lower } = dmdBranches(q, kpd);
      return upper - lower;
    };
    expect(gap(0.5)).toBeGreaterThan(gap(1.14));
    expect(gap(1.14)).toBeGreaterThan(gap(3));
  });

  it('agrees with an independent solution of equation (3.23a)', () => {
    // Independent reference: solve (1-R)/(1+R) = ±exp(-K_m d) directly, with
    // R written out from scratch rather than through xRatioForPair.
    const reference = (q: number, sign: 1 | -1): number => {
      const residual = (u: number): number | null => {
        const epsM = 1 - 1 / (u * u);
        const kmSq = q * q - epsM * u * u;
        const kdSq = q * q - u * u;
        if (kmSq <= 0 || kdSq <= 0 || Math.abs(epsM) < 1e-10) return null;
        const km = Math.sqrt(kmSq);
        const R = -(km / epsM) / Math.sqrt(kdSq);
        if (Math.abs(1 + R) < 1e-10) return null;
        return (1 - R) / (1 + R) - sign * Math.exp(-km * KPD);
      };
      const r = solveScalarRoots(residual, 1e-5, q * (1 - 1e-6), 1500);
      return sign > 0 ? r[r.length - 1] : r[0];
    };
    for (const q of [0.8, 1.2, 1.8, 2.4, 3.0]) {
      const { upper, lower } = dmdBranches(q);
      expect(upper).toBeCloseTo(reference(q, -1), 9);
      expect(lower).toBeCloseTo(reference(q, 1), 9);
    }
  });
});

describe('reflection ratio', () => {
  it('is null where either medium carries an oscillatory field', () => {
    expect(xRatioForPair(0.3, 0.9, METAL, VACUUM, WP)).toBeNull();
  });

  it('is null at the bulk plasma frequency, where eps_metal vanishes', () => {
    // Guarding this pole is what keeps a spurious root off the branch.
    expect(xRatioForPair(2, 1, METAL, VACUUM, WP)).toBeNull();
  });

  it('vanishes at the single-interface mode, which is the thick-film limit', () => {
    const q = 2;
    const u = singleInterfaceU(q, 1);
    // There kappa_m/eps_m = -kappa_d/eps_d exactly, so R = 1 and x = 0 --- which
    // is what equation (3.23a) demands, since exp(-K_m d) -> 0 as d -> infinity.
    expect(xRatioForPair(q, u, METAL, VACUUM, WP)!).toBeCloseTo(0, 12);
  });
});
