import { describe, it, expect } from 'vitest';
import { beampropCN, type BpmParams } from './bpm';

function defaultParams(overrides: Partial<BpmParams> = {}): BpmParams {
  return {
    z: 50,
    dz: 0.1,
    nd: 1.5,
    lambda: 1.0,
    Nx: 128,
    w: 5,
    xa: 40,
    xb: 5,
    nCladding: 1.5,
    nCore: 1.505,
    delta: 10,
    ...overrides,
  };
}

describe('beampropCN', () => {
  it('returns correct output dimensions', () => {
    const p = defaultParams();
    const r = beampropCN(p);
    expect(r.Nx).toBe(p.Nx);
    const expectedNz = Math.floor(Math.round(p.z / p.dz) / p.delta) + 1;
    expect(r.Nz).toBe(expectedNz);
    expect(r.field.length).toBeGreaterThanOrEqual(r.Nz * r.Nx);
  });

  it('initial row is a Gaussian profile', () => {
    const p = defaultParams();
    const r = beampropCN(p);
    // Row 0 should peak at the center (x=0)
    const mid = Math.floor(r.Nx / 2);
    const centerVal = r.field[mid];
    expect(centerVal).toBeGreaterThan(0);
    // Edges should be near zero
    expect(r.field[0]).toBeLessThan(centerVal * 0.01);
    expect(r.field[r.Nx - 1]).toBeLessThan(centerVal * 0.01);
  });

  it('maxVal is positive', () => {
    const r = beampropCN(defaultParams());
    expect(r.maxVal).toBeGreaterThan(0);
  });

  it('field values are non-negative (intensity)', () => {
    const r = beampropCN(defaultParams());
    for (let i = 0; i < r.Nz * r.Nx; i++) {
      expect(r.field[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('guided mode preserves power (approximately)', () => {
    const p = defaultParams();
    const r = beampropCN(p);
    const dx = p.xa / (p.Nx - 1);

    // Compute power at first and last rows
    let power0 = 0;
    for (let j = 0; j < r.Nx; j++) power0 += r.field[j] * dx;
    let powerLast = 0;
    const lastRow = (r.Nz - 1) * r.Nx;
    for (let j = 0; j < r.Nx; j++) powerLast += r.field[lastRow + j] * dx;

    // For a guided mode in a step-index waveguide, power should be roughly conserved
    // Allow 20% tolerance (some radiation loss is expected)
    expect(powerLast / power0).toBeGreaterThan(0.8);
    expect(powerLast / power0).toBeLessThan(1.2);
  });

  it('free-space propagation: beam spreads', () => {
    // Set nCore = nCladding (no waveguide) — beam should diffract and spread
    const p = defaultParams({ nCore: 1.5, nCladding: 1.5, z: 100 });
    const r = beampropCN(p);
    const dx = p.xa / (p.Nx - 1);

    // Compute RMS width at first and last rows
    function rmsWidth(rowOffset: number): number {
      let sum = 0, sumX2 = 0;
      for (let j = 0; j < r.Nx; j++) {
        const x = -p.xa / 2 + p.xa * j / (p.Nx - 1);
        const v = r.field[rowOffset + j];
        sum += v * dx;
        sumX2 += v * x * x * dx;
      }
      return Math.sqrt(sumX2 / sum);
    }

    const w0 = rmsWidth(0);
    const wEnd = rmsWidth((r.Nz - 1) * r.Nx);
    expect(wEnd).toBeGreaterThan(w0 * 1.1);
  });

  it('absorbing BC reduces power at edges', () => {
    const pNoBC = defaultParams({ nCore: 1.5, nCladding: 1.5, bcType: 'none', z: 100 });
    const pABC = defaultParams({ nCore: 1.5, nCladding: 1.5, bcType: 'absorbing', z: 100 });
    const rNoBC = beampropCN(pNoBC);
    const rABC = beampropCN(pABC);

    // With absorbing BC, total power at end should be less (absorber eats escaping light)
    const dx = pNoBC.xa / (pNoBC.Nx - 1);
    let powerNoBCEnd = 0, powerABCEnd = 0;
    const lastNoBC = (rNoBC.Nz - 1) * rNoBC.Nx;
    const lastABC = (rABC.Nz - 1) * rABC.Nx;
    for (let j = 0; j < rNoBC.Nx; j++) powerNoBCEnd += rNoBC.field[lastNoBC + j] * dx;
    for (let j = 0; j < rABC.Nx; j++) powerABCEnd += rABC.field[lastABC + j] * dx;

    expect(powerABCEnd).toBeLessThan(powerNoBCEnd);
  });

  it('beam offset shifts the peak position', () => {
    const offset = 5;
    const r = beampropCN(defaultParams({ beamOffset: offset }));
    // Initial row: peak should be near x = offset
    let maxIdx = 0, maxVal = 0;
    for (let j = 0; j < r.Nx; j++) {
      if (r.field[j] > maxVal) { maxVal = r.field[j]; maxIdx = j; }
    }
    const peakX = r.xMin + (r.xMax - r.xMin) * maxIdx / (r.Nx - 1);
    expect(Math.abs(peakX - offset)).toBeLessThan(1);
  });
});
