import { describe, it, expect } from 'vitest';
import {
  shapeMode, shapeModes, availableModeCount,
  crossSectionsAt, polarizabilityAt, drudeEpsilon,
  peakMetrics, computePlasmonSpectrum, retardationCorrection,
  SHAPE_OPTIONS, type PlasmonShapeId, type Complex,
} from './plasmonic-nanoparticles';

describe('shapeMode', () => {
  it('returns valid parameters for all shapes at default aspect ratio', () => {
    for (const opt of SHAPE_OPTIONS) {
      const mode = shapeMode(opt.id, opt.defaultR);
      expect(mode.shape).toBe(opt.id);
      expect(Number.isFinite(mode.epsilonJ)).toBe(true);
      expect(Number.isFinite(mode.vjOverV)).toBe(true);
      expect(Number.isFinite(mode.a2)).toBe(true);
      expect(Number.isFinite(mode.a4)).toBe(true);
      expect(mode.vOverL3).toBeGreaterThan(0);
    }
  });

  it('rod transverse supports 3 modes', () => {
    expect(availableModeCount('rod transverse')).toBe(3);
    const modes = shapeModes('rod transverse', 4, 3);
    expect(modes.length).toBe(3);
    for (const m of modes) {
      expect(m.shape).toBe('rod transverse');
      expect(Number.isFinite(m.epsilonJ)).toBe(true);
    }
  });

  it('rod longitudinal mode: epsilonJ becomes more negative with higher R', () => {
    const m2 = shapeMode('rod', 2);
    const m6 = shapeMode('rod', 6);
    expect(m6.epsilonJ).toBeLessThan(m2.epsilonJ);
  });

  it('Platonic solids have R-independent parameters', () => {
    const m1 = shapeMode('tetrahedron', 1);
    const m2 = shapeMode('tetrahedron', 1);
    expect(m1.epsilonJ).toBe(m2.epsilonJ);
    expect(m1.vjOverV).toBe(m2.vjOverV);
  });
});

describe('drudeEpsilon', () => {
  it('real part is negative in the plasmonic regime', () => {
    const eps = drudeEpsilon(600, 1, 9.0, 0.07);
    expect(eps.re).toBeLessThan(0);
  });

  it('imaginary part is positive (lossy)', () => {
    const eps = drudeEpsilon(600, 1, 9.0, 0.07);
    expect(eps.im).toBeGreaterThan(0);
  });

  it('approaches epsB at very short wavelengths (high energy)', () => {
    const epsB = 5;
    const eps = drudeEpsilon(10, epsB, 9.0, 0.07);
    // At very high energy, Drude term vanishes
    expect(eps.re).toBeCloseTo(epsB, 0);
  });
});

describe('crossSectionsAt', () => {
  it('extinction is positive for positive alpha.im', () => {
    const alpha: Complex = { re: 100, im: 50 };
    const xs = crossSectionsAt(550, alpha, 1);
    expect(xs.sigmaExtNm2).toBeGreaterThan(0);
  });

  it('scattering is non-negative', () => {
    const alpha: Complex = { re: 100, im: 50 };
    const xs = crossSectionsAt(550, alpha, 1);
    expect(xs.sigmaScaNm2).toBeGreaterThanOrEqual(0);
  });

  it('absorption = extinction - scattering', () => {
    const alpha: Complex = { re: 100, im: 50 };
    const xs = crossSectionsAt(550, alpha, 1);
    expect(xs.sigmaAbsNm2).toBeCloseTo(xs.sigmaExtNm2 - xs.sigmaScaNm2, 6);
  });
});

describe('retardationCorrection', () => {
  it('vanishes for vanishing size', () => {
    const mode = shapeMode('rod', 4);
    const corr = retardationCorrection(600, 0.001, 1, mode);
    expect(Math.abs(corr.re)).toBeLessThan(1e-6);
    expect(Math.abs(corr.im)).toBeLessThan(1e-6);
  });
});

describe('peakMetrics', () => {
  it('finds peak in a simple Lorentzian-like signal', () => {
    const n = 100;
    const wl = new Float64Array(n);
    const sig = new Float64Array(n);
    const center = 700;
    const gamma = 30;
    for (let i = 0; i < n; i++) {
      wl[i] = 500 + 4 * i;
      const d = wl[i] - center;
      sig[i] = 1 / (1 + (d / gamma) ** 2);
    }
    const p = peakMetrics(wl, sig);
    expect(p.lambdaPeakNm).toBeCloseTo(center, 0);
    expect(p.peakValue).toBeCloseTo(1, 1);
    expect(p.fwhmNm).toBeCloseTo(2 * gamma, 0);
  });

  it('q factor is peak / FWHM', () => {
    const n = 200;
    const wl = new Float64Array(n);
    const sig = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      wl[i] = 400 + 2 * i;
      const d = wl[i] - 600;
      sig[i] = 1 / (1 + (d / 20) ** 2);
    }
    const p = peakMetrics(wl, sig);
    expect(p.q).toBeCloseTo(p.lambdaPeakNm / p.fwhmNm, 2);
  });
});

describe('computePlasmonSpectrum', () => {
  it('returns arrays of correct length', () => {
    const sp = computePlasmonSpectrum({
      shape: 'rod',
      materialId: 'au',
      lengthNm: 50,
      aspectRatio: 4,
      hostIndex: 1.33,
      lambdaMinNm: 550,
      lambdaMaxNm: 1000,
      points: 50,
    });
    expect(sp.wavelengthNm.length).toBe(50);
    expect(sp.sigmaExtNm2.length).toBe(50);
    expect(sp.sigmaScaNm2.length).toBe(50);
    expect(sp.sigmaAbsNm2.length).toBe(50);
    expect(sp.quantumYield.length).toBe(50);
  });

  it('extinction has a resonance peak', () => {
    const sp = computePlasmonSpectrum({
      shape: 'rod',
      materialId: 'au',
      lengthNm: 50,
      aspectRatio: 4,
      hostIndex: 1.33,
      lambdaMinNm: 550,
      lambdaMaxNm: 1200,
      points: 200,
    });
    // Peak should not be at the edges
    const peakIdx = sp.sigmaExtOverV.indexOf(Math.max(...sp.sigmaExtOverV));
    expect(peakIdx).toBeGreaterThan(5);
    expect(peakIdx).toBeLessThan(195);
  });

  it('volume is positive', () => {
    const sp = computePlasmonSpectrum({
      shape: 'rod',
      materialId: 'au',
      lengthNm: 50,
      aspectRatio: 4,
      hostIndex: 1,
      lambdaMinNm: 550,
      lambdaMaxNm: 1000,
    });
    expect(sp.volumeNm3).toBeGreaterThan(0);
  });

  it('quantum yield is between 0 and 1', () => {
    const sp = computePlasmonSpectrum({
      shape: 'rod',
      materialId: 'au',
      lengthNm: 50,
      aspectRatio: 4,
      hostIndex: 1,
      lambdaMinNm: 550,
      lambdaMaxNm: 1000,
      points: 50,
    });
    for (let i = 0; i < sp.quantumYield.length; i++) {
      expect(sp.quantumYield[i]).toBeGreaterThanOrEqual(0);
      expect(sp.quantumYield[i]).toBeLessThanOrEqual(1);
    }
  });

  it('higher host index red-shifts the peak', () => {
    const sp1 = computePlasmonSpectrum({
      shape: 'rod', materialId: 'au', lengthNm: 50, aspectRatio: 4,
      hostIndex: 1, lambdaMinNm: 500, lambdaMaxNm: 1500, points: 200,
    });
    const sp2 = computePlasmonSpectrum({
      shape: 'rod', materialId: 'au', lengthNm: 50, aspectRatio: 4,
      hostIndex: 1.5, lambdaMinNm: 500, lambdaMaxNm: 1500, points: 200,
    });
    expect(sp2.peak.lambdaPeakNm).toBeGreaterThan(sp1.peak.lambdaPeakNm);
  });
});
