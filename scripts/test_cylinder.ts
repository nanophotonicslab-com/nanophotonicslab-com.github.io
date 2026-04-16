/**
 * Quick numerical test for src/lib/cylinder.ts.
 * Run with: npx tsx scripts/test_cylinder.ts
 */

import {
  Cx,
  besselJ,
  besselY,
  besselH1,
  besselI,
  besselK,
  giveM,
  det4,
  solve4,
  getRTCoefs,
  getDispersion,
  eelsParallel,
  eelsPerpendicular,
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

console.log('\n== Modified Bessel I_m, K_m (real arg) ==');
function testReal(name: string, got: number, expected: number, tol = 1e-8): void {
  const err = Math.abs(got - expected);
  const rel = err / (Math.abs(expected) + 1e-30);
  const ok = rel < tol;
  console.log(`${ok ? '✓' : '✗'} ${name.padEnd(18)} got=${got.toExponential(10)}  expected=${expected.toExponential(10)}  rel=${rel.toExponential(2)}`);
  if (!ok) process.exitCode = 1;
}
testReal('I_0(1.0)', besselI(0, [1, 0])[0], 1.266065877752e+0);
testReal('I_1(1.0)', besselI(1, [1, 0])[0], 5.651591039925e-1);
testReal('I_2(2.5)', besselI(2, [2.5, 0])[0], 1.276466147819e+0);
testReal('I_3(5.0)', besselI(3, [5, 0])[0], 1.033115016915e+1);
testReal('K_0(1.0)', besselK(0, [1, 0])[0], 4.210244382407e-1);
testReal('K_1(1.0)', besselK(1, [1, 0])[0], 6.019072301972e-1);
testReal('K_2(2.5)', besselK(2, [2.5, 0])[0], 1.214602062786e-1);
testReal('K_3(5.0)', besselK(3, [5, 0])[0], 8.291768415231e-3);

console.log('\n== RT coefs at a=15 nm, ε₁=10+0.1i, ε_h=1, ω=10 eV, qa=1.5, m=1 ==');
{
  const ka = (10 / HC_EV_NM) * 15;
  const rtOut = getRTCoefs(1.5, ka, [10, 0.1], [1, 0], 1, 'outside');
  test('outside t_pp', rtOut.t_pp, [-4.2158119385e-01, 8.6614818253e+00]);
  test('outside t_ss', rtOut.t_ss, [-6.9879634059e-01, 9.2106790755e+00]);
  const rtIn = getRTCoefs(1.5, ka, [10, 0.1], [1, 0], 1, 'inside');
  test('inside r_pp',  rtIn.r_pp,  [-6.6941298271e-01, -4.0092682467e+00]);
  test('inside r_ss',  rtIn.r_ss,  [-9.3577659051e-01, -1.7185711700e-01]);
}

console.log('\n== EELS parallel (outside b=25, inside b=5), v=0.3c, Max=3 ==');
{
  const wList = [2.0, 5.0, 10.0, 15.0];
  const scipyOutside = [3.67049486e-07, 8.49551035e-08, 1.23937761e-08, 2.04847547e-09];
  const scipyInside  = [-1.00549854e-05, -2.35830336e-06, -6.68651479e-07, -2.89108500e-07];
  const tsOutside = eelsParallel({ a_nm: 15, b_nm: 25, eps_h: [1, 0], eps1_of_w: () => [10, 0.1], w_eV: wList, vFrac: 0.3, maxOrder: 3 });
  const tsInside  = eelsParallel({ a_nm: 15, b_nm: 5,  eps_h: [1, 0], eps1_of_w: () => [10, 0.1], w_eV: wList, vFrac: 0.3, maxOrder: 3 });
  for (let i = 0; i < wList.length; i++) {
    testReal(`outside ω=${wList[i]} eV`, tsOutside[i], scipyOutside[i], 1e-6);
  }
  for (let i = 0; i < wList.length; i++) {
    testReal(`inside ω=${wList[i]} eV`, tsInside[i], scipyInside[i], 1e-6);
  }
}

console.log('\n== EELS perpendicular: Ag Drude (ωp=9.17, γ=0.021), a=15, b=20, v=0.2c, Max=10 ==');
{
  const wp = 9.17, gamma = 0.021, eps_b = 1.0;
  const eps1_ag = (w: number): C => {
    // ε = 1 − ωp²/(ω(ω+iγ))
    const denom: C = [w * w, w * gamma];
    const den2 = denom[0] * denom[0] + denom[1] * denom[1];
    const inv: C = [denom[0] / den2, -denom[1] / den2];
    const wp2 = wp * wp;
    return [eps_b - wp2 * inv[0], -wp2 * inv[1]];
  };
  const wList = [2.0, 3.5, 4.0, 4.5, 6.0, 10.0];
  const scipyPerp = [5.08058809e-06, 7.95620314e-06, 8.36043116e-06, 2.18310885e-05, 2.16667056e-04, 7.97349695e-08];
  const qz_nm: number[] = [];
  for (let i = 0; i < 80; i++) qz_nm.push(1e-4 + (1.0 - 1e-4) * i / 79);
  const ts = eelsPerpendicular({ a_nm: 15, b_nm: 20, eps_h: [1, 0], eps1_of_w: eps1_ag, w_eV: wList, vFrac: 0.2, maxOrder: 10, qz_nm });
  for (let i = 0; i < wList.length; i++) {
    testReal(`⊥ ω=${wList[i]} eV`, ts[i], scipyPerp[i], 1e-5);
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
