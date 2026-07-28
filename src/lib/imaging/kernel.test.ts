import { describe, it, expect } from 'vitest';
import {
  Rng, gaussianPSF, FWHM_PER_SIGMA, stepSigmaNm, renderFrame, measureFwhmPx,
  detect, localize, msdCurve, fitMsd, thompsonSigma,
  simulateTruth, renderDetected, analyzeSequence, truthTrajectories,
  type SimParams,
} from './index';

// Acceptance criteria from the implementation brief, §10 "Kernel — automated".

const baseParams: SimParams = {
  N: 20, D: 0.5, motion: 'brownian', photons: 1200, modality: 'fluorescence',
  NA: 1.4, lambda: 520, pixel: 65, field: 256, background: 8, readNoise: 1.6,
  frames: 100, dt: 20, seed: 42,
};

// Determinism does not depend on the field size, so these use a smaller frame:
// byte-comparing 256x256 frames repeatedly cost seconds for no extra coverage.
const smallParams: SimParams = { ...baseParams, field: 64, frames: 20 };

describe('1. determinism', () => {
  it('gives byte-identical frames for the same seed and parameters', () => {
    const t1 = simulateTruth(smallParams);
    const t2 = simulateTruth(smallParams);
    expect(Array.from(t1.x)).toEqual(Array.from(t2.x));
    const a = renderDetected(smallParams, t1, 7);
    const b = renderDetected(smallParams, t2, 7);
    expect(new Uint8Array(a.buffer)).toEqual(new Uint8Array(b.buffer));
  });

  it('gives a different movie for a different seed', () => {
    const a = simulateTruth(baseParams);
    const b = simulateTruth({ ...baseParams, seed: 43 });
    expect(Array.from(a.x)).not.toEqual(Array.from(b.x));
  });

  it('renders any frame independently of the order frames are requested', () => {
    const truth = simulateTruth(smallParams);
    const forward = [0, 1, 2].map(t => renderDetected(smallParams, truth, t));
    const backward = [2, 1, 0].map(t => renderDetected(smallParams, truth, t)).reverse();
    forward.forEach((f, i) => expect(Array.from(f)).toEqual(Array.from(backward[i])));
  });
});

describe('2. photon conservation', () => {
  // one emitter, centred, no noise and no background: the rendered image must
  // contain the emitted photons regardless of PSF width or pixel size
  for (const sigmaPx of [0.5, 1, 2, 4, 8]) {
    it(`conserves photons at sigma = ${sigmaPx} px`, () => {
      const field = 128, pixelNm = 65, photons = 5000;
      const centre = (field / 2) * pixelNm;
      const img = renderFrame(
        [{ x: centre, y: centre, photons }],
        { field, pixelNm, sigmaPsfNm: sigmaPx * pixelNm },
      );
      let sum = 0;
      for (const v of img) sum += v;
      expect(Math.abs(sum - photons) / photons).toBeLessThan(0.005);
    });
  }
});

describe('3. PSF width', () => {
  for (const [lambda, NA] of [[520, 1.4], [640, 1.2], [488, 0.9]] as const) {
    it(`renders FWHM = 2.3548*0.21*lambda/NA/pixel for lambda=${lambda}, NA=${NA}`, () => {
      const field = 128, pixelNm = 40;
      const psf = gaussianPSF(lambda, NA);
      const expected = psf.fwhmNm / pixelNm;
      // put the emitter at a pixel centre so the radial bins are well populated
      const centrePx = field / 2 + 0.5;
      const centre = centrePx * pixelNm;
      const img = renderFrame(
        [{ x: centre, y: centre, photons: 1e6 }],
        { field, pixelNm, sigmaPsfNm: psf.sigmaNm },
      );
      const measured = measureFwhmPx(img, field, centrePx, centrePx);
      expect(Math.abs(measured - expected) / expected).toBeLessThan(0.02);
    });
  }

  it('matches the analytic sigma-to-FWHM relation', () => {
    const psf = gaussianPSF(520, 1.4);
    expect(psf.sigmaNm).toBeCloseTo((0.21 * 520) / 1.4, 10);
    expect(psf.fwhmNm / psf.sigmaNm).toBeCloseTo(FWHM_PER_SIGMA, 10);
  });
});

describe('4. diffusion statistics', () => {
  it('gives per-axis displacement variance 2*D*dt over 1e4 steps', () => {
    const D = 0.5, dtS = 0.02;
    const sigma = stepSigmaNm(D, dtS); // nm
    const r = new Rng(7);
    const n = 20000;
    let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) {
      const d = r.normal(0, sigma);
      s += d; s2 += d * d;
    }
    const varNm2 = s2 / n - (s / n) ** 2;
    const expectedNm2 = 2 * D * dtS * 1e6; // um^2 -> nm^2
    expect(Math.abs(varNm2 - expectedNm2) / expectedNm2).toBeLessThan(0.03);
  });

  it('walks the trajectory with the same statistics through the periodic field', () => {
    const p: SimParams = { ...baseParams, N: 200, frames: 60, seed: 3 };
    const truth = simulateTruth(p);
    const trajs = truthTrajectories(truth);
    let s2 = 0, n = 0;
    for (const t of trajs) {
      for (let i = 1; i < t.x.length; i++) {
        s2 += (t.x[i] - t.x[i - 1]) ** 2;
        n++;
      }
    }
    const expected = 2 * p.D * (p.dt / 1000) * 1e6;
    expect(Math.abs(s2 / n - expected) / expected).toBeLessThan(0.03);
  });
});

describe('5. shot noise', () => {
  // exercises both branches of the Poisson sampler: Knuth below 50, normal
  // approximation at and above it
  for (const lambda of [5, 50, 500, 5000]) {
    it(`gives var/mean = 1 for a flat field at lambda = ${lambda}`, () => {
      const n = 40000;
      const flat = new Float32Array(n).fill(lambda);
      const out = detect(flat, { background: 0, readNoise: 0 }, new Rng(11));
      let s = 0, s2 = 0;
      for (const v of out) { s += v; s2 += v * v; }
      const mean = s / n;
      const variance = s2 / n - mean * mean;
      expect(Math.abs(mean - lambda) / lambda).toBeLessThan(0.02);
      expect(Math.abs(variance / mean - 1)).toBeLessThan(0.02);
    });
  }
});

describe('6. MSD round trip', () => {
  // With 100 frames x 20 particles the MSD fit is unbiased but carries about
  // 4.6% scatter per realization (measured over 25 seeds), so "within 5%" is a
  // statement about the estimator, not about any single movie. It is asserted
  // the only way it can hold: on the ensemble mean, plus a per-seed bound set
  // at roughly three standard deviations.
  it('recovers the input D from the ground truth, unbiased within 5%', () => {
    const errors: number[] = [];
    for (let seed = 1; seed <= 25; seed++) {
      const p: SimParams = { ...baseParams, frames: 100, N: 20, seed };
      const curve = msdCurve(truthTrajectories(simulateTruth(p)), p.dt / 1000, p.frames);
      errors.push((fitMsd(curve, 5).D - p.D) / p.D);
    }
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    for (const e of errors) expect(Math.abs(e)).toBeLessThan(0.15);
  });

  it('recovers D from the localized movie within 15%', () => {
    // through the full pipeline — render, detect, localize, link, fit — the
    // recovered D also carries localization noise, hence the looser bound
    const p: SimParams = { ...baseParams, N: 12, frames: 60, photons: 3000, seed: 9 };
    const truth = simulateTruth(p);
    const a = analyzeSequence(p, truth);
    expect(a.meanDetections).toBeGreaterThan(p.N * 0.8);
    expect(Math.abs(a.fit.D - p.D) / p.D).toBeLessThan(0.15);
  }, 30000);
});

// The performance targets in section 6 are acceptance criterion 11, which the
// brief lists as manual: they are verified in the browser on the real page,
// where the numbers mean something. Asserting wall-clock here only produces
// flaky failures on a loaded CI machine.

describe('localizer', () => {
  it('finds an isolated emitter to well below one pixel', () => {
    const field = 64, pixelNm = 65;
    const psf = gaussianPSF(520, 1.4);
    const x = 30.3 * pixelNm, y = 21.8 * pixelNm;
    const img = renderFrame([{ x, y, photons: 20000 }], { field, pixelNm, sigmaPsfNm: psf.sigmaNm });
    const noisy = detect(img, { background: 5, readNoise: 1.6 }, new Rng(2));
    const dets = localize(noisy, { field, sigmaPx: psf.sigmaNm / pixelNm });
    expect(dets.length).toBe(1);
    // the centroid reports the same continuous pixel coordinate the renderer
    // consumes, so it should land on the emitter itself
    expect(Math.hypot(dets[0].xPx - 30.3, dets[0].yPx - 21.8)).toBeLessThan(0.3);
  });
});

describe('Thompson localization precision', () => {
  it('follows the shot-noise limit s/sqrt(N) when background is negligible', () => {
    const s = 78, a = 65;
    const sigma = thompsonSigma(s, a, 10000, 0);
    expect(sigma).toBeCloseTo(Math.sqrt((s * s + (a * a) / 12) / 10000), 6);
  });

  it('degrades as photons fall', () => {
    const dim = thompsonSigma(78, 65, 200, 3);
    const bright = thompsonSigma(78, 65, 20000, 3);
    expect(dim).toBeGreaterThan(bright);
  });
});
