import { describe, it, expect } from 'vitest';
import { cubicSpline } from './cubic-spline';

describe('cubicSpline', () => {
  it('passes through every tabulated point', () => {
    const x = [0, 1, 2, 3, 4, 5];
    const y = [2, -1, 4, 0.5, 3, -2];
    const s = cubicSpline(x, y);
    for (let i = 0; i < x.length; i++) expect(s(x[i])).toBeCloseTo(y[i], 12);
  });

  it('matches SciPy CubicSpline (not-a-knot) on a five-point zigzag', () => {
    // Reference values from scipy.interpolate.CubicSpline with its default
    // bc_type='not-a-knot', the same configuration the BEM solver uses.
    const s = cubicSpline([0, 1, 2, 3, 4], [0, 1, 0, 1, 0]);
    const expected: [number, number][] = [
      [0.5, 1.125],
      [1.5, 0.375],
      [2.5, 0.375],
      [3.5, 1.125],
    ];
    for (const [t, v] of expected) expect(s(t)).toBeCloseTo(v, 10);
  });

  it('matches SciPy in the first and last intervals', () => {
    // The not-a-knot conditions only bite in the two end intervals, and an
    // error there decays towards the middle of the table: a check taken well
    // inside will not see it. Unevenly spaced on purpose.
    const x = [0, 0.4, 1.5, 2.1, 4.0, 4.3];
    const y = [1, -2, 0.5, 3, -1, 2];
    const s = cubicSpline(x, y);
    const expected: [number, number][] = [
      [0.1, -0.043827083890],
      [0.25, -1.229108961322],
      [4.1, -0.240718065488],
      [4.25, 1.343426042573],
    ];
    for (const [t, v] of expected) expect(s(t)).toBeCloseTo(v, 10);
  });

  it('is the unique cubic through four points', () => {
    // With four nodes the two not-a-knot conditions leave a single cubic.
    const x = [1.3776, 1.7712, 2.4797, 4.1328];
    const y = [2.8884, 2.5591, 2.2496, 1.9599];
    const s = cubicSpline(x, y);
    // Lagrange interpolation through the same four points.
    const lagrange = (t: number) =>
      x.reduce((acc, xi, i) => {
        let term = y[i];
        for (let j = 0; j < x.length; j++) if (j !== i) term *= (t - x[j]) / (xi - x[j]);
        return acc + term;
      }, 0);
    for (const t of [1.5, 2.0664, 3.0, 3.9]) expect(s(t)).toBeCloseTo(lagrange(t), 9);
  });

  it('reproduces a cubic exactly', () => {
    const f = (t: number) => 2 * t ** 3 - 3 * t ** 2 + t - 5;
    const x = [-2, -1, 0, 1, 2, 3];
    const s = cubicSpline(x, x.map(f));
    for (const t of [-1.7, -0.3, 0.42, 1.9, 2.75]) expect(s(t)).toBeCloseTo(f(t), 9);
  });

  it('fits a parabola through three points', () => {
    const s = cubicSpline([0, 1, 3], [1, 2, 0]);
    expect(s(0.5)).toBeCloseTo(1.666666666667, 10);
    expect(s(2)).toBeCloseTo(1.666666666667, 10);
  });

  it('falls back to a straight line and a constant on short tables', () => {
    const line = cubicSpline([1, 3], [10, 20]);
    expect(line(2)).toBeCloseTo(15, 12);
    const point = cubicSpline([7], [42]);
    expect(point(0)).toBe(42);
  });

  it('holds the end values outside the tabulated interval', () => {
    const s = cubicSpline([0, 1, 2, 3], [5, 6, 7, 8]);
    expect(s(-10)).toBe(5);
    expect(s(99)).toBe(8);
  });

  it('is smooth where linear interpolation would put a corner', () => {
    // A table whose slope changes abruptly at one node — the shape of the
    // gold interband edge. Linear interpolation leaves a kink there; a C²
    // spline must not.
    const x = [0, 1, 2, 3, 4, 5];
    const y = [0, 0.1, 0.2, 1.6, 3.0, 4.4];
    const s = cubicSpline(x, y);
    const h = 1e-4;
    const left = (s(2) - s(2 - h)) / h;
    const right = (s(2 + h) - s(2)) / h;
    expect(right - left).toBeLessThan(1e-3);
  });
});
