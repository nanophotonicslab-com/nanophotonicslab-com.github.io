import { describe, it, expect } from 'vitest';
import { diffusionTracking, derive } from './spec';
import { computeQuick } from './compute';
import { defaultValues, evaluateEnvelope, type EnvelopeLevel } from '../../../solver-spec';
import { gaussianPSF, stepSigmaNm, thompsonSigma } from '../../index';

/**
 * Acceptance criterion 7 — each envelope predicate flips state exactly at its
 * stated threshold. Also checks the derived quantities the readouts show, since
 * the predicates are expressed in terms of them.
 */

const defaults = defaultValues(diffusionTracking);

/** Evaluate the envelope for a parameter set, as the page does. */
function envelope(overrides: Record<string, number | string> = {}) {
  const p = { ...defaults, ...overrides };
  const q = computeQuick(p);
  const rows = evaluateEnvelope(diffusionTracking.envelope, { p, o: q.observables });
  const byId: Record<string, { level: EnvelopeLevel; message: string }> = {};
  for (const r of rows) byId[r.id] = { level: r.level, message: r.message };
  return byId;
}

describe('derived quantities', () => {
  it('matches the formulas the brief specifies', () => {
    const d = derive(defaults);
    const lambda = 520, NA = 1.4, pixel = 65, D = 0.5, dt = 20, photons = 1200;
    expect(d.stepPx).toBeCloseTo((1000 * Math.sqrt(2 * D * (dt / 1000))) / pixel, 10);
    expect(d.fwhmPx).toBeCloseTo((2.3548200450309493 * 0.21 * lambda) / NA / pixel, 10);
    expect(d.nyquistNm).toBeCloseTo(lambda / (4 * NA), 10);
    // peak of a photon-conserving Gaussian, and the peak-SNR built from it
    const sigmaPx = gaussianPSF(lambda, NA).sigmaNm / pixel;
    const peak = photons / (2 * Math.PI * sigmaPx * sigmaPx);
    expect(d.peak).toBeCloseTo(peak, 6);
    expect(d.snr).toBeCloseTo(peak / Math.sqrt(peak + 8 + 1.6 ** 2), 6);
    expect(d.sigmaLocNm).toBeCloseTo(
      thompsonSigma(gaussianPSF(lambda, NA).sigmaNm, pixel, photons, Math.sqrt(8 + 1.6 ** 2)), 10,
    );
  });

  it('scales the step with sqrt(D dt), as diffusion requires', () => {
    const a = derive({ ...defaults, D: 1, dt: 10 }).stepPx;
    const b = derive({ ...defaults, D: 4, dt: 10 }).stepPx;
    expect(b / a).toBeCloseTo(2, 6);
    expect(stepSigmaNm(1, 0.01)).toBeCloseTo(1000 * Math.sqrt(0.02), 8);
  });
});

describe('Nyquist sampling check', () => {
  // flips at pixel = lambda / (4 NA); at the defaults that is 92.857 nm
  const limit = 520 / (4 * 1.4);

  it('is ok exactly at the limit and warns just past it', () => {
    expect(envelope({ pixel: Math.floor(limit) }).nyquist.level).toBe('ok');
    expect(envelope({ pixel: Math.ceil(limit) }).nyquist.level).toBe('warn');
  });

  it('names the fix when it warns', () => {
    expect(envelope({ pixel: 160 }).nyquist.message).toMatch(/smaller pixel|lower NA/);
  });

  it('moves with wavelength and NA, not just pixel size', () => {
    // a longer wavelength relaxes the requirement
    expect(envelope({ pixel: 100, lambda: 700 }).nyquist.level).toBe('ok');
    expect(envelope({ pixel: 100, lambda: 500 }).nyquist.level).toBe('warn');
  });
});

describe('motion blur check', () => {
  // flips at stepPx = 0.5 * fwhmPx
  it('flips between the two sides of half the PSF width', () => {
    const d = derive(defaults);
    const half = 0.5 * d.fwhmPx;
    // dt that puts the step just below / just above half the FWHM
    const dtFor = (stepPx: number) => {
      const sigmaNm = stepPx * 65;
      return 1000 * (sigmaNm / 1000) ** 2 / (2 * 0.5);
    };
    expect(envelope({ dt: dtFor(half * 0.98) }).blur.level).toBe('ok');
    expect(envelope({ dt: dtFor(half * 1.02) }).blur.level).toBe('warn');
  });

  it('warns at the declared defaults, where the step already exceeds half the FWHM', () => {
    // documented deviation from the brief: its "Textbook, everything green" preset
    // is not achievable at its own defaults (D = 0.5, dt = 20 ms, pixel = 65 nm)
    const d = derive(defaults);
    expect(d.stepPx).toBeGreaterThan(0.5 * d.fwhmPx);
    expect(envelope().blur.level).toBe('warn');
  });

  it('is ok once Δt drops to 8 ms', () => {
    expect(envelope({ dt: 8 }).blur.level).toBe('ok');
  });
});

describe('linking density check', () => {
  // density = N pi (fwhm/2)^2 / field^2; warn at 0.05, fail at 0.15
  const densityFor = (N: number, field: number) => {
    const fwhm = (2.3548200450309493 * 0.21 * 520) / 1.4 / 65;
    return (N * Math.PI * (fwhm / 2) ** 2) / (field * field);
  };

  it('flips to warn at 0.05 and to fail at 0.15', () => {
    // solve for the N that lands either side of each threshold at field 128
    const perParticle = densityFor(1, 128);
    const nWarn = 0.05 / perParticle;
    const nFail = 0.15 / perParticle;
    expect(envelope({ field: 128, N: Math.floor(nWarn) }).density.level).toBe('ok');
    expect(envelope({ field: 128, N: Math.ceil(nWarn) }).density.level).toBe('warn');
    expect(envelope({ field: 128, N: Math.floor(nFail) }).density.level).toBe('warn');
    expect(envelope({ field: 128, N: Math.ceil(nFail) }).density.level).toBe('fail');
  });

  it('cannot be tripped at a 256 px field within the allowed N, as the brief assumed', () => {
    // recorded deviation: the brief's "Crowded field (N=200)" preset does not
    // trip this check at the default field, so that preset also halves the field
    expect(densityFor(200, 256)).toBeLessThan(0.05);
    expect(densityFor(500, 256)).toBeLessThan(0.05);
    expect(envelope({ N: 200, field: 256 }).density.level).toBe('ok');
  });

  it('is tripped by the Crowded field preset as shipped', () => {
    const crowded = diffusionTracking.scenarios.find(s => s.label === 'Crowded field')!;
    expect(envelope(crowded.set).density.level).toBe('warn');
  });
});

describe('photon budget check', () => {
  it('warns once sigma_loc exceeds one pixel', () => {
    // a very dim, high-background run pushes the precision past a pixel
    const bad = envelope({ photons: 10, background: 500, pixel: 20 });
    expect(bad.photons.level).toBe('warn');
    expect(bad.photons.message).toMatch(/raise the intensity|lower the background/);
  });

  it('warns below 100 photons even when the precision looks fine', () => {
    expect(envelope({ photons: 99 }).photons.level).toBe('warn');
    expect(envelope({ photons: 100 }).photons.level).toBe('ok');
  });
});

describe('shipped presets', () => {
  const byLabel = (l: string) => diffusionTracking.scenarios.find(s => s.label === l)!;

  it('trips exactly the check each preset claims', () => {
    expect(envelope(byLabel('Fast diffusion').set).blur.level).toBe('warn');
    expect(envelope(byLabel('Fast diffusion').set).nyquist.level).toBe('ok');
    expect(envelope(byLabel('Fast diffusion').set).density.level).toBe('ok');

    const nyq = envelope(byLabel('Nyquist-limited pixels').set);
    expect(nyq.nyquist.level).toBe('warn');
    expect(nyq.blur.level).toBe('ok');
    expect(nyq.density.level).toBe('ok');
    expect(nyq.photons.level).toBe('ok');

    const crowded = envelope(byLabel('Crowded field').set);
    expect(crowded.density.level).toBe('warn');
  });

  it('keeps every preset inside the declared parameter domains', () => {
    for (const sc of diffusionTracking.scenarios) {
      for (const [key, value] of Object.entries(sc.set)) {
        const p = diffusionTracking.params.find(q => q.key === key);
        expect(p, `preset "${sc.label}" sets unknown parameter ${key}`).toBeDefined();
        if (typeof value === 'number' && p!.min !== undefined && p!.max !== undefined) {
          expect(value).toBeGreaterThanOrEqual(p!.min);
          expect(value).toBeLessThanOrEqual(p!.max);
        }
        if (p!.choices) expect(p!.choices).toContain(value);
      }
    }
  });
});

describe('spec integrity', () => {
  it('gives every parameter a group that exists, and every group parameters', () => {
    const groupIds = new Set(diffusionTracking.groups.map(g => g.id));
    for (const p of diffusionTracking.params) expect(groupIds.has(p.group)).toBe(true);
    for (const g of diffusionTracking.groups) {
      expect(diffusionTracking.params.some(p => p.group === g.id)).toBe(true);
    }
  });

  it('documents every parameter and observable', () => {
    for (const p of diffusionTracking.params) expect(p.help, `${p.key} has no help`).toBeTruthy();
    for (const o of diffusionTracking.observables) expect(o.help, `${o.key} has no help`).toBeTruthy();
  });

  it('produces a value for every declared observable except the fitted one', () => {
    const q = computeQuick(defaults);
    for (const o of diffusionTracking.observables) {
      if (o.key === 'dFit') continue; // only available after the movie is analysed
      expect(Number.isFinite(q.observables[o.key]), `${o.key} is not finite`).toBe(true);
    }
    expect(q.observables.dFit).toBeNaN();
  });
});
