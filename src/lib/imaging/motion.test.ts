import { describe, it, expect } from 'vitest';
import {
  reflect, msdCurve, fitMsd, fitAlpha, simulateTruth, truthTrajectories, motionParams,
  type SimParams,
} from './index';

/**
 * Each motion model has a signature in the mean-square displacement, and that
 * signature is what these tests assert — not the implementation.
 *
 *   free Brownian   MSD = 4 D tau                      alpha = 1
 *   directed        MSD = 4 D tau + v^2 tau^2          alpha -> 2
 *   confined        MSD -> L^2 / 3   (2D, side L)      alpha -> 0
 *   meshwork        free early, slowed late            alpha < 1
 *
 * All statistics are ensemble averages over many particles, so the tolerances
 * are set by sampling error rather than by anything in the code.
 */

const base: SimParams = {
  N: 200, D: 0.5, motion: 'brownian', photons: 1200, modality: 'fluorescence',
  NA: 1.4, lambda: 520, pixel: 65, field: 256, background: 8, readNoise: 1.6,
  frames: 120, dt: 20, seed: 11,
};

/** Ground-truth MSD curve and its fits, without going through the microscope. */
function truthMsd(p: SimParams) {
  const truth = simulateTruth(p);
  const curve = msdCurve(truthTrajectories(truth), p.dt / 1000, p.frames);
  return { curve, fit: fitMsd(curve, 5), alpha: fitAlpha(curve), truth };
}

describe('reflect', () => {
  it('leaves interior points alone', () => {
    expect(reflect(5, 0, 10)).toBe(5);
  });

  it('mirrors a single overshoot at either wall', () => {
    expect(reflect(12, 0, 10)).toBeCloseTo(8, 12);
    expect(reflect(-3, 0, 10)).toBeCloseTo(3, 12);
  });

  it('folds repeatedly for a step much larger than the interval', () => {
    // 34 with span 10: 34 -> 26 -> 14 -> 6 by successive mirroring
    expect(reflect(34, 0, 10)).toBeCloseTo(6, 12);
    expect(reflect(-34, 0, 10)).toBeCloseTo(6, 12);
  });

  it('always lands inside the interval, for any input', () => {
    for (const v of [-1e4, -7.3, 0, 3.2, 10, 1e4, 99.9]) {
      const out = reflect(v, -2, 7);
      expect(out).toBeGreaterThanOrEqual(-2);
      expect(out).toBeLessThanOrEqual(7);
    }
  });

  it('degenerates safely on a zero-width interval', () => {
    expect(reflect(5, 3, 3)).toBe(3);
  });
});

describe('free Brownian motion', () => {
  it('gives MSD = 4 D tau and alpha = 1', () => {
    const { fit, alpha } = truthMsd({ ...base, motion: 'brownian' });
    expect(Math.abs(fit.D - base.D) / base.D).toBeLessThan(0.05);
    expect(Math.abs(alpha.alpha - 1)).toBeLessThan(0.05);
  });
});

describe('directed motion', () => {
  it('adds the ballistic term v^2 tau^2 to the diffusive one', () => {
    const v = 3; // um/s
    const p: SimParams = { ...base, motion: 'directed', driftV: v, driftAngle: 30 };
    const { curve } = truthMsd(p);
    // MSD(tau) = 4 D tau + v^2 tau^2, with v in nm/s
    const vNm = v * 1000;
    for (let i = 0; i < curve.tau.length; i++) {
      const tau = curve.tau[i];
      const expected = 4 * base.D * 1e6 * tau + vNm * vNm * tau * tau;
      expect(Math.abs(curve.msd[i] - expected) / expected).toBeLessThan(0.1);
    }
  });

  it('is superdiffusive, approaching alpha = 2 when drift dominates', () => {
    const slow = truthMsd({ ...base, motion: 'directed', driftV: 0.5, driftAngle: 0 });
    const fast = truthMsd({ ...base, motion: 'directed', driftV: 20, driftAngle: 0 });
    expect(slow.alpha.alpha).toBeGreaterThan(1);
    expect(fast.alpha.alpha).toBeGreaterThan(1.8);
    expect(fast.alpha.alpha).toBeLessThan(2.05);
    expect(fast.alpha.alpha).toBeGreaterThan(slow.alpha.alpha);
  });

  it('moves the ensemble centre of mass at the set speed and heading', () => {
    const p: SimParams = { ...base, N: 400, motion: 'directed', driftV: 2, driftAngle: 45, frames: 60 };
    const trajs = truthTrajectories(simulateTruth(p));
    const last = trajs[0].x.length - 1;
    const meanDx = trajs.reduce((a, t) => a + (t.x[last] - t.x[0]), 0) / trajs.length;
    const meanDy = trajs.reduce((a, t) => a + (t.y[last] - t.y[0]), 0) / trajs.length;
    const elapsed = last * (p.dt / 1000);
    const expected = 2 * 1000 * elapsed * Math.cos(Math.PI / 4);
    expect(Math.abs(meanDx - expected) / expected).toBeLessThan(0.1);
    expect(Math.abs(meanDy - expected) / expected).toBeLessThan(0.1);
  });
});

describe('confined diffusion', () => {
  // For a 1D interval of length L with reflecting walls the equilibrium
  // mean-square displacement saturates at L^2/6; in 2D the two axes add, giving
  // L^2/3.
  it('saturates at L^2/3 for a square corral', () => {
    const L = 400; // nm
    const p: SimParams = { ...base, N: 400, motion: 'confined', corralNm: L, frames: 200, dt: 50 };
    const { curve } = truthMsd(p);
    const plateau = curve.msd[curve.msd.length - 1];
    expect(Math.abs(plateau - (L * L) / 3) / ((L * L) / 3)).toBeLessThan(0.1);
  });

  it('is subdiffusive: alpha well below 1', () => {
    const { alpha } = truthMsd({
      ...base, N: 300, motion: 'confined', corralNm: 400, frames: 200, dt: 50,
    });
    expect(alpha.alpha).toBeLessThan(0.5);
    expect(alpha.alpha).toBeGreaterThan(-0.05);
  });

  it('keeps every particle inside its own corral for the whole movie', () => {
    const L = 300;
    const p: SimParams = { ...base, N: 50, motion: 'confined', corralNm: L, frames: 150 };
    const truth = simulateTruth(p);
    for (let i = 0; i < truth.N; i++) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let t = 0; t < truth.frames; t++) {
        const x = truth.x[t * truth.N + i], y = truth.y[t * truth.N + i];
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
      // never wider than the corral (plus a hair for floating point)
      expect(maxX - minX).toBeLessThanOrEqual(L + 1e-6);
      expect(maxY - minY).toBeLessThanOrEqual(L + 1e-6);
    }
  });

  it('recovers free diffusion when the corral is much larger than the walk', () => {
    const small = truthMsd({ ...base, motion: 'confined', corralNm: 300 });
    const huge = truthMsd({ ...base, motion: 'confined', corralNm: 1e5 });
    expect(Math.abs(huge.fit.D - base.D) / base.D).toBeLessThan(0.06);
    expect(small.fit.D).toBeLessThan(huge.fit.D);
  });
});

describe('meshwork (interaction with an underlying network)', () => {
  it('reduces the long-time diffusion coefficient as hops get rarer', () => {
    const mk = (hopProb: number) => truthMsd({
      ...base, N: 300, motion: 'network', meshNm: 800, hopProb, frames: 200, dt: 40,
    });
    const open = mk(1);      // every crossing succeeds -> free diffusion
    const leaky = mk(0.1);
    const tight = mk(0.005);
    // the apparent D falls monotonically as the fences get harder to cross
    expect(open.fit.D).toBeGreaterThan(leaky.fit.D);
    expect(leaky.fit.D).toBeGreaterThan(tight.fit.D);
    // with every crossing allowed the meshwork must be indistinguishable from
    // free Brownian motion
    expect(Math.abs(open.fit.D - base.D) / base.D).toBeLessThan(0.08);
    expect(Math.abs(open.alpha.alpha - 1)).toBeLessThan(0.06);
  });

  it('is subdiffusive when the fences are hard to cross', () => {
    const { alpha } = truthMsd({
      ...base, N: 300, motion: 'network', meshNm: 600, hopProb: 0.002, frames: 200, dt: 40,
    });
    expect(alpha.alpha).toBeLessThan(0.85);
  });

  it('confines a sealed meshwork to one compartment', () => {
    // hopProb = 0 makes every compartment a corral of side meshNm
    const p: SimParams = {
      ...base, N: 200, motion: 'network', meshNm: 500, hopProb: 0, frames: 250, dt: 50,
    };
    const mesh = motionParams(p).meshNm!;
    const { curve } = truthMsd(p);
    const plateau = curve.msd[curve.msd.length - 1];
    expect(Math.abs(plateau - (mesh * mesh) / 3) / ((mesh * mesh) / 3)).toBeLessThan(0.12);
  });

  it('snaps the mesh so a whole number of compartments spans the field', () => {
    const p: SimParams = { ...base, motion: 'network', meshNm: 1000, hopProb: 0.1 };
    const fieldNm = p.field * p.pixel; // 16640 nm
    const mesh = motionParams(p).meshNm!;
    const cells = fieldNm / mesh;
    expect(Math.abs(cells - Math.round(cells))).toBeLessThan(1e-9);
    // and stays close to what was asked for
    expect(Math.abs(mesh - 1000) / 1000).toBeLessThan(0.1);
  });
});

describe('model dispatch', () => {
  it('converts drift from um/s into nm per frame', () => {
    const mp = motionParams({ ...base, motion: 'directed', driftV: 2, driftAngle: 90 });
    expect(mp.driftNm).toBeCloseTo(2 * 1000 * 0.02, 10);
    expect(mp.driftAngle).toBeCloseTo(Math.PI / 2, 12);
  });

  it('gives every model the same diffusive step', () => {
    const expected = 1000 * Math.sqrt(2 * base.D * 0.02);
    for (const motion of ['brownian', 'directed', 'confined', 'network'] as const) {
      expect(motionParams({ ...base, motion }).sigmaNm).toBeCloseTo(expected, 9);
    }
  });

  it('rejects an unknown motion model rather than silently drifting', () => {
    const p = { ...base, motion: 'teleport' as never };
    expect(() => simulateTruth(p)).toThrow(/unsupported motion/);
  });
});
