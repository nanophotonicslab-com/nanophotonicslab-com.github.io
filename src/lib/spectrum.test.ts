/**
 * Tests for the shared Spectrum type: each adapter is exercised against real
 * solver output, and the node-editor spec's minimal vertical slice
 * (Material → PlasmonSolver → ColorSwatch) is wired end-to-end through it.
 */
import { describe, it, expect } from 'vitest';
import { computeMie } from './mie';
import { computePlasmonSpectrum } from './plasmonic-nanoparticles';
import { computeElectronSphereSpectrum } from './electron-sphere';
import { spectrumToHex } from './colorimetry';
import { spectrumFromMie, spectrumFromPlasmon, spectrumFromElectron, type Spectrum } from './spectrum';

function expectWellFormed(s: Spectrum) {
  expect(s.x.length).toBeGreaterThan(0);
  for (const [name, ch] of Object.entries(s.series)) {
    expect(ch.length, `channel ${name}`).toBe(s.x.length);
  }
}

describe('spectrum adapters', () => {
  it('adapts computeMie output without copying', () => {
    const r = computeMie((_l) => [1.5, 0.1], 60, 1.33, 400, 800, 25);
    const s = spectrumFromMie(r);
    expect(s.axis).toBe('lambdaNm');
    expectWellFormed(s);
    expect(s.x).toBe(r.lambda);          // reference, not copy
    expect(s.series.csca).toBe(r.csca);
  });

  it('adapts computePlasmonSpectrum output', () => {
    const r = computePlasmonSpectrum({
      shape: 'rod', materialId: 'au', lengthNm: 50, aspectRatio: 4,
      hostIndex: 1.33, lambdaMinNm: 400, lambdaMaxNm: 900, points: 60,
    });
    const s = spectrumFromPlasmon(r);
    expect(s.axis).toBe('lambdaNm');
    expectWellFormed(s);
    expect(s.series.sigmaExtNm2).toBe(r.sigmaExtNm2);
  });

  it('adapts computeElectronSphereSpectrum output', () => {
    const r = computeElectronSphereSpectrum(
      (_l) => [1.5, 0.1],
      { radiusNm: 50, beta: 0.5, impactNm: 60, lmax: 4, qCutNm: 0.71, nz: 16 },
      1.5, 4.5, 30,
    );
    const s = spectrumFromElectron(r);
    expect(s.axis).toBe('energyEv');
    expectWellFormed(s);
    expect(s.x).toBe(r.energy);
  });
});

describe('minimal vertical slice: Material → PlasmonSolver → ColorSwatch', () => {
  it('flows a Spectrum into the colorimetry sink and yields a colour', () => {
    const r = computePlasmonSpectrum({
      shape: 'rod', materialId: 'au', lengthNm: 50, aspectRatio: 4,
      hostIndex: 1.33, lambdaMinNm: 380, lambdaMaxNm: 780, points: 120,
    });
    const s = spectrumFromPlasmon(r);
    // sigma → bounded [0,1] response, the one-line adapter the spec calls for
    const sca = s.series.sigmaScaNm2;
    const peak = Math.max(...sca);
    const R = Float64Array.from(sca, (v) => v / peak);
    const hex = spectrumToHex(s.x, R);
    expect(hex).toMatch(/^#[0-9a-f]{6}$/i);
  });
});
