/**
 * Quick numerical test for src/lib/cylinder.ts.
 * Run with: npx tsx scripts/test_cylinder.ts
 */

import {
  Cx,
  besselJ,
  besselY,
  besselH1,
  giveM,
  det4,
  getDispersion,
  HC_EV_NM,
} from '../src/lib/cylinder.ts';
import type { C } from '../src/lib/cylinder.ts';

function fmt(c: C): string {
  const sign = c[1] >= 0 ? '+' : '-';
  return `(${c[0].toExponential(10)}) ${sign} (${Math.abs(c[1]).toExponential(10)})i`;
}

function test(name: string, actual: C, expected: C, tol = 1e-8): void {
  const err = Math.hypot(actual[0] - expected[0], actual[1] - expected[1]);
  const rel = err / (Math.hypot(expected[0], expected[1]) + 1e-30);
  const ok = rel < tol;
  const mark = ok ? '✓' : '✗';
  console.log(`${mark} ${name.padEnd(28)}  got=${fmt(actual)}  expected=${fmt(expected)}  rel=${rel.toExponential(2)}`);
  if (!ok) process.exitCode = 1;
}

console.log('== Bessel J/Y/H scalar tests (vs SciPy) ==');

test('J_0(1+0i)', besselJ(0, [1, 0]),       [7.6519768656e-01, 0.0e+00]);
test('Y_0(1+0i)', besselY(0, [1, 0]),       [8.8256964216e-02, 0.0e+00]);
test('H_0(1+0i)', besselH1(0, [1, 0]),      [7.6519768656e-01, 8.8256964216e-02]);

test('J_0(0.5+0.3i)', besselJ(0, [0.5, 0.3]), [9.5901068765e-01, -7.3498364867e-02]);
test('Y_0(0.5+0.3i)', besselY(0, [0.5, 0.3]), [-3.4840690030e-01, 4.0691542161e-01]);
test('H_0(0.5+0.3i)', besselH1(0, [0.5, 0.3]), [5.5209526604e-01, -4.2190526517e-01]);

test('J_0(3+0i)',  besselJ(0, [3, 0]),  [-2.6005195490e-01, 0]);
test('Y_0(3+0i)',  besselY(0, [3, 0]),  [3.7685001001e-01, 0]);

test('J_0(5+2i)',  besselJ(0, [5, 2]),  [-4.2973398507e-01, 1.1922394411e+00]);
test('Y_0(5+2i)',  besselY(0, [5, 2]),  [-1.2272416036e+00, -3.9976830942e-01]);
test('H_0(5+2i)',  besselH1(0, [5, 2]), [-2.9965675656e-02, -3.5002162493e-02]);

test('J_1(1+0i)',  besselJ(1, [1, 0]),  [4.4005058574e-01, 0]);
test('Y_1(1+0i)',  besselY(1, [1, 0]),  [-7.8121282130e-01, 0]);
test('J_1(0.5+0.3i)', besselJ(1, [0.5, 0.3]), [2.5046714293e-01, 1.3770042616e-01]);
test('Y_1(0.5+0.3i)', besselY(1, [0.5, 0.3]), [-1.1679826771e+00, 5.5246691495e-01]);
test('J_1(5+2i)',  besselJ(1, [5, 2]),  [-1.2287085813e+00, -2.8070443605e-01]);

test('J_2(1+0i)',  besselJ(2, [1, 0]),  [1.1490348493e-01, 0]);
test('Y_2(1+0i)',  besselY(2, [1, 0]),  [-1.6506826068e+00, 0]);
test('J_2(5+2i)',  besselJ(2, [5, 2]),  [-3.2676482413e-02, -1.1195570286e+00]);
test('Y_2(5+2i)',  besselY(2, [5, 2]),  [1.1696203817e+00, -5.3152918707e-02]);

test('J_3(1+0i)',  besselJ(3, [1, 0]),  [1.9563353983e-02, 0]);
test('Y_3(1+0i)',  besselY(3, [1, 0]),  [-5.8215176060e+00, 0]);
test('J_3(5+2i)',  besselJ(3, [5, 2]),  [8.9732975795e-01, -4.8238965746e-01]);

console.log('\n== det(M) tests: Álvaro example a=15 nm, ε₁=10+0.1i, ε_h=1 ==');

const a_nm = 15.0;
const eps_h: C = [1, 0];
const eps1: C = [10, 0.1];

function detSample(w_eV: number, q_nm: number, m: number): C {
  const k0 = w_eV / HC_EV_NM;
  const ka = k0 * a_nm;
  const qa = q_nm * a_nm;
  const M = giveM(qa, ka, eps1, eps_h, m);
  return det4(M);
}

// Pick a few points from SciPy output — inner guided region at ω=10 eV, m=1.
// qa=1.24 → Re(D) ≈ -0.0259;  qa=1.31 → Re(D) ≈ +0.0278 (sign change)
const w = 10.0;
const k0 = w / HC_EV_NM;
const ka = k0 * a_nm;
console.log(`ω=${w} eV  k₀=${k0.toFixed(6)} nm⁻¹  ka=${ka.toFixed(6)}`);

for (const m of [0, 1, 2]) {
  console.log(`  m=${m}`);
  for (const qa of [0.96, 1.10, 1.24, 1.38, 1.55, 1.72, 2.00, 2.30]) {
    const q = qa / a_nm;
    const d = detSample(w, q, m);
    console.log(`    qa=${qa.toFixed(2)}  Re(D)=${d[0].toExponential(4)}  Im(D)=${d[1].toExponential(4)}`);
  }
}

console.log('\n== Dispersion sample: m=1 at ω=10 eV ==');
const dispRes = getDispersion({
  energies_eV: [10.0],
  q_nm: Array.from({ length: 500 }, (_, i) => 0.001 + (0.2 - 0.001) * i / 499),
  a_nm: 15.0,
  eps_h: [1, 0],
  eps1_of_w: () => [10, 0.1],
  m: 1,
});
console.log(`q_light = ${dispRes.q_light_nm[0].toFixed(6)}  q_mat = ${dispRes.q_mat_nm[0].toFixed(6)}`);
console.log(`roots = ${dispRes.roots_nm[0].map(r => r.toFixed(6)).join(', ')}`);
