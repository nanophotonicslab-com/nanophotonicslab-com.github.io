import { describe, it, expect } from 'vitest';
import { mieAt } from './mie';
import { fitMieRadius, parseSpectrumCsv } from './fit';

const GOLDLIKE = (_l: number): [number, number] => [0.4, 2.6]; // crude constant metal
const DIEL = (_l: number): [number, number] => [1.6, 0.05];

/** Synthetic "measurement": A·σ_ext(λ; rTrue) with multiplicative noise. */
function synth(getNK: (l: number) => [number, number], rTrue: number, scale: number, noise: number) {
  const N = 120;
  const x = new Float64Array(N), y = new Float64Array(N);
  let seed = 42;
  const rand = () => { seed = (seed * 1103515245 + 12345) % 2 ** 31; return seed / 2 ** 31; };
  for (let i = 0; i < N; i++) {
    x[i] = 420 + (900 - 420) * i / (N - 1);
    const [n, k] = getNK(x[i]);
    y[i] = scale * mieAt(n, k, 1.33, rTrue, x[i]).cext * (1 + noise * (2 * rand() - 1));
  }
  return { x, y };
}

describe('fitMieRadius', () => {
  it('recovers radius and scale from noisy synthetic data', () => {
    const { x, y } = synth(DIEL, 73, 2.5, 0.02);
    const fit = fitMieRadius(DIEL, 1.33, x, y, 'cext', 20, 200);
    expect(Math.abs(fit.radiusNm - 73)).toBeLessThan(2);
    expect(Math.abs(fit.scale - 2.5) / 2.5).toBeLessThan(0.05);
    expect(fit.r2).toBeGreaterThan(0.98);
    expect(fit.model.length).toBe(x.length);
  });

  it('works on a metal-like particle and the csca channel', () => {
    const N = 100;
    const x = new Float64Array(N), y = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      x[i] = 400 + (800 - 400) * i / (N - 1);
      const [n, k] = GOLDLIKE(x[i]);
      y[i] = 0.8 * mieAt(n, k, 1.33, 45, x[i]).csca;
    }
    const fit = fitMieRadius(GOLDLIKE, 1.33, x, y, 'csca', 10, 150);
    expect(Math.abs(fit.radiusNm - 45)).toBeLessThan(1);
    expect(fit.r2).toBeGreaterThan(0.999);
  });

  it('respects the radius bounds', () => {
    const { x, y } = synth(DIEL, 73, 1, 0);
    const fit = fitMieRadius(DIEL, 1.33, x, y, 'cext', 100, 200);
    expect(fit.radiusNm).toBeGreaterThanOrEqual(100);
    expect(fit.radiusNm).toBeLessThanOrEqual(200);
  });

  it('handles degenerate flat data without blowing up', () => {
    const x = Float64Array.from({ length: 20 }, (_, i) => 500 + 10 * i);
    const y = new Float64Array(20); // all zeros
    const fit = fitMieRadius(DIEL, 1.33, x, y, 'cext', 20, 100);
    expect(Number.isFinite(fit.radiusNm)).toBe(true);
    expect(fit.scale).toBe(0);
  });
});

describe('parseSpectrumCsv', () => {
  it('parses comma/tab data, skips headers, sorts, and converts µm', () => {
    const text = 'wavelength,extinction\n# comment\n0.700,0.5\n500,1.2\n600\t2.4\nnot,numbers\n'
      + Array.from({ length: 6 }, (_, i) => `${800 + i * 10},1.${i}`).join('\n');
    const r = parseSpectrumCsv(text);
    expect('x' in r).toBe(true);
    if ('x' in r) {
      expect(r.x.length).toBe(9);
      expect(r.x[0]).toBe(500);
      expect(r.x[2]).toBe(700);     // 0.700 µm → 700 nm, sorted into place
      expect(r.y[1]).toBe(2.4);     // tab-separated row
    }
  });

  it('rejects files without enough numeric rows', () => {
    const r = parseSpectrumCsv('a,b\n1nm,2\n');
    expect('error' in r).toBe(true);
  });
});
