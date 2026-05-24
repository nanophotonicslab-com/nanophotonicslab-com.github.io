import { describe, it, expect } from 'vitest';
import {
  Cx, besselJ, besselY, besselH1, besselJ_d, besselH1_d,
  besselI, besselK, det4, solve4, giveM,
  type C,
} from './cylinder';

// Helper: check complex values within tolerance
function expectC(actual: C, expected: C, tol = 1e-8) {
  expect(actual[0]).toBeCloseTo(expected[0], -Math.log10(tol));
  expect(actual[1]).toBeCloseTo(expected[1], -Math.log10(tol));
}

describe('complex arithmetic (Cx)', () => {
  it('add', () => expectC(Cx.add([1, 2], [3, 4]), [4, 6]));
  it('sub', () => expectC(Cx.sub([3, 4], [1, 2]), [2, 2]));
  it('mul', () => expectC(Cx.mul([1, 2], [3, 4]), [-5, 10]));
  it('div', () => expectC(Cx.div([1, 2], [3, 4]), [0.44, 0.08]));
  it('abs', () => expect(Cx.abs([3, 4])).toBeCloseTo(5, 10));
  it('sqrt of real', () => expectC(Cx.sqrt([4, 0]), [2, 0]));
  it('sqrt of negative', () => expectC(Cx.sqrt([-1, 0]), [0, 1]));
  it('exp(0) = 1', () => expectC(Cx.exp([0, 0]), [1, 0]));
  it('exp(i*pi) = -1', () => expectC(Cx.exp([0, Math.PI]), [-1, 0], 1e-12));
  it('log(exp(z)) = z', () => {
    const z: C = [1.5, 0.7];
    expectC(Cx.log(Cx.exp(z)), z, 1e-12);
  });
});

describe('Bessel J', () => {
  it('J_0(0) = 1', () => expectC(besselJ(0, [0, 0]), [1, 0]));
  it('J_1(0) = 0', () => expectC(besselJ(1, [0, 0]), [0, 0]));
  it('J_0 at first zero (2.4048)', () => {
    const j = besselJ(0, [2.4048255577, 0]);
    expect(Math.abs(j[0])).toBeLessThan(1e-6);
  });
  it('J_1 at first zero (3.8317)', () => {
    const j = besselJ(1, [3.8317059702, 0]);
    expect(Math.abs(j[0])).toBeLessThan(1e-6);
  });
  it('J_{-m} = (-1)^m J_m for real argument', () => {
    const z: C = [2.5, 0];
    expectC(besselJ(-2, z), besselJ(2, z));
    const jNeg3 = besselJ(-3, z);
    const j3 = besselJ(3, z);
    expectC(jNeg3, [-j3[0], -j3[1]], 1e-10);
  });
  it('J_0(1) matches known value', () => {
    // J_0(1) = 0.7651976865579666
    const j = besselJ(0, [1, 0]);
    expect(j[0]).toBeCloseTo(0.7651976866, 8);
  });
});

describe('Bessel Y', () => {
  it('Y_0 at first zero (0.8936)', () => {
    const y = besselY(0, [0.8935769663, 0]);
    expect(Math.abs(y[0])).toBeLessThan(1e-5);
  });
  it('Wronskian J_m Y_m\' - J_m\' Y_m = 2/(pi*z) for real z', () => {
    const z: C = [3.0, 0];
    const J0 = besselJ(0, z);
    const Y0 = besselY(0, z);
    const J0d = besselJ_d(0, z);
    const Y0d: C = [
      // Y_0'(z) = -Y_1(z)
      -besselY(1, z)[0],
      -besselY(1, z)[1],
    ];
    const W = J0[0] * Y0d[0] - J0d[0] * Y0[0]; // real parts only for real z
    expect(W).toBeCloseTo(2 / (Math.PI * z[0]), 6);
  });
});

describe('Modified Bessel I and K', () => {
  it('I_0(0) = 1', () => expectC(besselI(0, [0, 0]), [1, 0]));
  it('I_0(1) matches known value', () => {
    // I_0(1) = 1.2660658777520084
    const i0 = besselI(0, [1, 0]);
    expect(i0[0]).toBeCloseTo(1.2660658778, 6);
  });
  it('K_0(1) matches known value', () => {
    // K_0(1) = 0.4210244382
    const k0 = besselK(0, [1, 0]);
    expect(k0[0]).toBeCloseTo(0.4210244382, 5);
  });
  it('I_{-m} = I_m for integer m', () => {
    expectC(besselI(-2, [1.5, 0]), besselI(2, [1.5, 0]));
  });
});

describe('det4', () => {
  it('identity matrix has determinant 1', () => {
    const I4 = [
      [[1, 0], [0, 0], [0, 0], [0, 0]],
      [[0, 0], [1, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [1, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0], [1, 0]],
    ] as C[][];
    expectC(det4(I4), [1, 0]);
  });

  it('scalar multiple of identity has det = s^4', () => {
    const s = 3;
    const M = [
      [[s, 0], [0, 0], [0, 0], [0, 0]],
      [[0, 0], [s, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [s, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0], [s, 0]],
    ] as C[][];
    expectC(det4(M), [81, 0], 1e-6);
  });
});

describe('solve4', () => {
  it('solves identity system', () => {
    const I4 = [
      [[1, 0], [0, 0], [0, 0], [0, 0]],
      [[0, 0], [1, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [1, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0], [1, 0]],
    ] as C[][];
    const rhs: C[] = [[1, 0], [2, 0], [3, 0], [4, 0]];
    const x = solve4(I4, rhs);
    expectC(x[0], [1, 0]);
    expectC(x[1], [2, 0]);
    expectC(x[2], [3, 0]);
    expectC(x[3], [4, 0]);
  });

  it('solves a known system with complex entries', () => {
    // 2x = [4, 2i] => x = [2, i]
    const M = [
      [[2, 0], [0, 0], [0, 0], [0, 0]],
      [[0, 0], [2, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [2, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0], [2, 0]],
    ] as C[][];
    const rhs: C[] = [[4, 0], [0, 2], [6, 0], [0, 4]];
    const x = solve4(M, rhs);
    expectC(x[0], [2, 0]);
    expectC(x[1], [0, 1]);
    expectC(x[2], [3, 0]);
    expectC(x[3], [0, 2]);
  });
});

describe('giveM', () => {
  it('returns a 4x4 matrix', () => {
    const M = giveM(0.1, 0.5, [4, 0], [1, 0], 1);
    expect(M.length).toBe(4);
    for (const row of M) expect(row.length).toBe(4);
  });

  it('determinant is real for lossless dielectrics in guided window', () => {
    // For real eps1 > eps_h > 0 and q in the guided window,
    // det(M) should be purely real (or very nearly so).
    const eps1: C = [12, 0]; // e.g. silicon-like
    const eps_h: C = [1, 0]; // vacuum
    const ka = 0.5;
    const qa = 1.2; // inside guided window
    const M = giveM(qa, ka, eps1, eps_h, 0);
    const d = det4(M);
    // Imaginary part should be negligible compared to real part
    expect(Math.abs(d[1])).toBeLessThan(Math.abs(d[0]) * 1e-10 + 1e-15);
  });
});
