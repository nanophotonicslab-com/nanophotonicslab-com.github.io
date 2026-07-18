// Shared graphene-conductivity physics, used by BOTH the Materials/Graphene calculator
// (materials.astro) and the nonlinear-plasmonics engine (nl-graphene.ts).
//
// Everything here is unit-agnostic: it works with energies in eV and returns either a
// dimensionless quantity (the chemical potential is in eV; the Mikhailov S⁽³⁾ tensor is
// dimensionless). Each caller maps the result into its own unit convention (σ₀/a.u. for
// the Materials module, eV-nm-fs/Gaussian for the plasmonics lab).

export interface Cx { re: number; im: number; }

const c = (re: number, im = 0): Cx => ({ re, im });
const cAdd = (a: Cx, b: Cx): Cx => ({ re: a.re + b.re, im: a.im + b.im });
const cSub = (a: Cx, b: Cx): Cx => ({ re: a.re - b.re, im: a.im - b.im });
const cMul = (a: Cx, b: Cx): Cx => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const cDiv = (a: Cx, b: Cx): Cx => { const d = b.re * b.re + b.im * b.im; return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d }; };
const cScale = (a: Cx, s: number): Cx => ({ re: a.re * s, im: a.im * s });
const cLog = (a: Cx): Cx => ({ re: 0.5 * Math.log(a.re * a.re + a.im * a.im), im: Math.atan2(a.im, a.re) });
const I = c(0, 1);
const ONE = c(1);
const sum = (...xs: Cx[]) => xs.reduce(cAdd, c(0));
const prod = (...xs: Cx[]) => xs.reduce(cMul);
const neg = (z: Cx) => cScale(z, -1);
const sq = (z: Cx) => cMul(z, z);
const cb = (z: Cx) => cMul(cMul(z, z), z);
const qd = (z: Cx) => sq(sq(z));
const iv = (...f: Cx[]) => cDiv(ONE, prod(...f));        // 1 / (f₁·f₂·…)
const tm = (num: Cx, ...f: Cx[]) => cDiv(num, prod(...f)); // num / (f₁·f₂·…)
const lg1 = (z: Cx) => cLog(cAdd(z, ONE));               // log(1 + z)
const eqC = (a: Cx, b: Cx) => a.re === b.re && a.im === b.im;

export const KB_EV = 8.617333262e-5;   // Boltzmann constant [eV/K]

// Thermal Drude-weight energy of doped graphene: 2kT·ln(2cosh(E_F/2kT))
// = E_F + 2kT·ln(1+e^(−E_F/kT)). NOTE this is NOT the chemical potential μ(T)
// (which *decreases* with T at fixed density); it is the effective energy in
// the intraband weight with μ approximated by E_F. Using it as the interband
// Pauli threshold overestimates the onset at low doping (E_F ≲ 2k_BT) — fine
// for the E_F ≥ 0.2 eV regimes the tools expose (error < 3% at 300 K).
export const grapheneMuT = (EF_ev: number, T_K: number): number => {
  const KT = KB_EV * Math.max(T_K, 1e-9);
  return Math.abs(EF_ev) + 2 * KT * Math.log(1 + Math.exp(-Math.abs(EF_ev) / KT));
};

// ── Mikhailov, PRB 93, 085403 (2016): full third-order tensor S⁽³⁾_xxxx ──
// Port of graphene_sigma3.m for the all-x case (indices a=b=c=d ⇒ Dabcd=3, every δ-product
// = 1). f_int are the Appendix-D integrals; the distinct-argument branch covers THG (ω,ω,ω),
// the coincident branches (a=c etc.) are reached by the Kerr permutations of (ω,ω,−ω).
const fInt = (n: number, a: Cx, b: Cx, cc: Cx): Cx => {
  const a2 = sq(a), b2 = sq(b), c2 = sq(cc), a3 = cb(a), b3 = cb(b), c3 = cb(cc);
  const ab = cMul(a, b), ac = cMul(a, cc), bc = cMul(b, cc), abc = cMul(ab, cc);
  const amb = cSub(a, b), amc = cSub(a, cc), bmc = cSub(b, cc);
  const la = lg1(a), lb = lg1(b), lc = lg1(cc);
  if (n === 1) {
    if (eqC(a, b) && !eqC(a, cc)) return sum(  // a = b ≠ c
      iv(a2, cc), tm(cSub(cc, cScale(a, 2)), a2, sq(amc), cAdd(a, ONE)), tm(cSub(cScale(cc, 4), a), cc, cb(amc), cAdd(cc, ONE)),
      neg(tm(ONE, c(2), sq(amc), sq(cAdd(a, ONE)))), neg(iv(sq(amc), sq(cAdd(cc, ONE)))),
      neg(cMul(tm(sum(cScale(a3, -3), cScale(cMul(a2, cc), 12), cScale(cMul(a, c2), -8), cScale(c3, 2)), a3, qd(amc)), la)),
      neg(cMul(tm(cSub(cScale(a, 2), cScale(cc, 5)), cc, qd(amc)), lc)));
    if (eqC(a, cc) && !eqC(a, b)) return sum(  // a = c ≠ b
      iv(a2, b), tm(ONE, a, sq(amb), cAdd(a, ONE)), neg(iv(cb(amb), cAdd(b, ONE))),
      tm(cSub(cScale(a, 2), b), c(2), a, sq(amb), sq(cAdd(a, ONE))), tm(c(2), c(3), amb, cb(cAdd(a, ONE))),
      neg(cMul(tm(sum(neg(a3), cScale(cMul(a2, b), 2), cScale(cMul(a, b2), -3), b3), a3, qd(amb)), la)),
      neg(cMul(tm(sum(a2, cScale(ab, -3), cScale(b2, 3)), b2, qd(amb)), lb)));
    if (eqC(b, cc) && !eqC(a, b)) return sum(  // b = c ≠ a
      iv(a, b2), tm(cSub(a, cScale(b, 4)), b, cb(amb), cAdd(b, ONE)), tm(cSub(cScale(b, 4), a), c(2), b, sq(amb), sq(cAdd(b, ONE))),
      neg(iv(amb, cb(cAdd(b, ONE)))), neg(cMul(tm(sum(neg(a2), cScale(ab, -3), b2), a2, qd(amb)), la)),
      neg(cMul(tm(sum(a3, cScale(cMul(a2, b), -3), cScale(cMul(a, b2), 2), cScale(b3, 3)), b3, qd(amb)), lb)));
    if (eqC(a, b) && eqC(a, cc)) return sum(  // a = b = c
      iv(a3), tm(ONE, a3, cAdd(a, ONE)), neg(tm(c(3), c(4), qd(cAdd(a, ONE)))), neg(tm(ONE, c(3), a, cb(cAdd(a, ONE)))), neg(cMul(tm(c(2), qd(a)), la)));
    return sum(  // distinct
      iv(a, b, cc), tm(sum(neg(ab), cScale(ac, 2), cScale(bc, 3), cScale(c2, -4)), cc, sq(cSub(cc, a)), sq(bmc), cAdd(cc, ONE)),
      neg(iv(amb, sq(bmc), cAdd(b, ONE))), neg(iv(amc, bmc, sq(cAdd(cc, ONE)))),
      neg(cMul(tm(sum(neg(a3), cScale(cMul(a, ac), -2), cScale(abc, 3), cMul(ac, cc), neg(cMul(bc, cc))), a2, sq(amb), cb(amc)), la)),
      neg(cMul(tm(sum(cScale(ab, -2), ac, cScale(b2, 3), neg(bc)), b2, sq(amb), sq(bmc)), lb)),
      neg(cMul(tm(sum(a2, ab, cScale(ac, -4), cScale(bc, -3), cScale(c2, 5)), cc, cb(amc), sq(cSub(cc, b))), lc)));
  }
  if (n === 2) {
    if (eqC(a, b) && !eqC(a, cc)) return sum(  // a = b ≠ c
      tm(ONE, c(2), a, amc, sq(cAdd(a, ONE))), tm(cSub(cScale(a, 2), cc), a2, sq(amc), cAdd(a, ONE)),
      neg(cMul(tm(sum(cScale(a2, 3), cScale(ac, -3), c2), a3, cb(amc)), la)), cMul(tm(ONE, cc, cb(amc)), lc));
    if (eqC(a, cc) && !eqC(a, b)) return sum(  // a = c ≠ b
      neg(tm(ONE, a, sq(amb), cAdd(a, ONE))), neg(tm(ONE, b, sq(amb), cAdd(b, ONE))),
      neg(cMul(tm(cSub(b, cScale(a, 3)), a2, cb(amb)), la)), neg(cMul(tm(cSub(cScale(b, 3), a), b2, cb(amb)), lb)));
    if (eqC(b, cc) && !eqC(a, b)) return sum(  // b = c ≠ a
      tm(cSub(cScale(b, 2), a), b2, sq(amb), cAdd(b, ONE)), neg(tm(ONE, c(2), b, amb, sq(cAdd(b, ONE)))),
      neg(cMul(tm(ONE, a, cb(amb)), la)), neg(cMul(tm(sum(neg(a2), cScale(ab, 3), cScale(b2, -3)), b3, cb(amb)), lb)));
    if (eqC(a, b) && eqC(a, cc)) return sum(  // a = b = c
      neg(tm(ONE, a3, cAdd(a, ONE))), neg(tm(ONE, c(2), a2, sq(cAdd(a, ONE)))), neg(tm(ONE, c(3), a, cb(cAdd(a, ONE)))), cMul(tm(ONE, qd(a)), la));
    return sum(  // distinct
      iv(b, amb, bmc, cAdd(b, ONE)), neg(cMul(iv(a, sq(amb), amc), la)),
      neg(cMul(tm(sum(cScale(ab, 2), neg(ac), cScale(b2, -3), cScale(bc, 2)), b2, sq(amb), sq(bmc)), lb)),
      neg(cMul(iv(cc, cSub(cc, a), sq(bmc)), lc)));
  }
  if (n === 3) {
    if (eqC(a, b)) return sum(neg(cMul(tm(c(2), a3), la)), iv(a2), tm(ONE, a2, cAdd(a, ONE)));
    return sum(iv(a, b), cMul(tm(ONE, a2, amb), la), neg(cMul(tm(ONE, b2, amb), lb)));
  }
  // n === 4 (n === 5 is handled by fInt5 below, never routed here)
  if (eqC(a, b)) return sum(neg(tm(ONE, c(2), a, sq(cAdd(a, ONE)))), neg(tm(ONE, a2, cAdd(a, ONE))), cMul(tm(ONE, a3), la));
  return sum(neg(iv(b, amb, cAdd(b, ONE))), cMul(tm(ONE, a, sq(amb)), la), cMul(tm(cSub(a, cScale(b, 2)), b2, sq(amb)), lb));
};

// n === 5 needs its own logs of (1∓·)/(1±·); kept separate for clarity.
const fInt5 = (a: Cx, b: Cx): Cx => {
  const expr = (z: Cx) => sum(cDiv(c(4), sq(z)), tm(c(2), sq(z), cSub(ONE, sq(z))), cMul(cDiv(c(3), cb(z)), cLog(cDiv(cSub(ONE, z), cAdd(ONE, z)))));
  if (eqC(a, b)) return expr(a);                 // a = b
  if (eqC(a, neg(b))) return neg(expr(a));       // a = −b
  return sum(cDiv(c(4), cMul(a, b)),             // distinct
    neg(cMul(tm(cScale(b, 2), sq(a), cSub(a, b), cAdd(a, b)), cLog(cDiv(cSub(ONE, a), cAdd(ONE, a))))),
    cMul(tm(cScale(a, 2), sq(b), cSub(a, b), cAdd(a, b)), cLog(cDiv(cSub(ONE, b), cAdd(ONE, b)))));
};

// Unsymmetrized all-x third-order tensor for a frequency triple, expressed through its
// dimensionless arguments O1=(ω₁+iγ)/2E_F, O12=(ω₁+ω₂+iγ)/2E_F, O123=(ω₁+ω₂+ω₃+iγ)/2E_F.
// Returns { full: S_3_0+S_2_1+S_1_2+S_0_3, intra: S_3_0 }.
function sigma3Unsym(O1: Cx, O12: Cx, O123: Cx): { full: Cx; intra: Cx } {
  const S30 = cDiv(cScale(I, 3 / 8), prod(O123, O12, O1));
  let S21 = cMul(cScale(I, -1 / 4), sum(
    cDiv(cAdd(ONE, cScale(O123, 0.75)), prod(O1, O12, sq(cAdd(ONE, O123)))),
    cDiv(c(0.25), prod(O1, cAdd(ONE, O12), cAdd(ONE, O123))),
    cDiv(c(0.25), prod(O1, cAdd(ONE, O12), sq(cAdd(ONE, O123)))),
    cScale(fInt(1, O1, O12, O123), -0.25), cScale(fInt(2, O1, O12, O123), 0.25)));
  S21 = cAdd(S21, cMul(cScale(I, 1 / 4), sum(
    cDiv(cSub(ONE, cScale(O123, 0.75)), prod(O1, O12, sq(cSub(ONE, O123)))),
    cDiv(c(0.25), prod(neg(O1), cSub(ONE, O12), cSub(ONE, O123))),
    cDiv(c(0.25), prod(neg(O1), cSub(ONE, O12), sq(cSub(ONE, O123)))),
    cScale(fInt(1, neg(O1), neg(O12), neg(O123)), -0.25), cScale(fInt(2, neg(O1), neg(O12), neg(O123)), 0.25))));
  let S12 = cMul(cScale(I, 1 / 4), sum(
    cMul(cDiv(c(0.75), prod(sq(O1), O123, O12)), cLog(cAdd(ONE, O1))),
    neg(cDiv(c(0.75), prod(O123, O12, O1))),
    cDiv(c(0.25), prod(O123, O1, cAdd(ONE, O12))),
    cScale(cDiv(fInt(3, O1, O12, ONE), O123), -0.25), cScale(cDiv(fInt(4, O1, O12, ONE), O123), 0.25)));
  S12 = cAdd(S12, cMul(cScale(I, -1 / 4), sum(
    cMul(cDiv(c(0.75), prod(sq(O1), O123, O12)), cLog(cSub(ONE, O1))),
    neg(cDiv(c(0.75), prod(neg(O123), O12, O1))),
    cDiv(c(0.25), prod(O123, O1, cSub(ONE, O12))),
    cScale(cDiv(fInt(3, neg(O1), neg(O12), ONE), O123), 0.25), cScale(cDiv(fInt(4, neg(O1), neg(O12), ONE), O123), -0.25))));
  const S03 = cMul(cDiv(cScale(I, 3 / 32), O12), fInt5(O1, O123));
  return { full: sum(S30, S21, S12, S03), intra: S30 };
}

const Osub = (re: number, EF_ev: number, gamma_ev: number): Cx => c(re / (2 * Math.abs(EF_ev)), gamma_ev / (2 * Math.abs(EF_ev)));

// Full third-harmonic (THG, ω,ω,ω): the 6 permutations coincide. E, E_F, γ in eV.
export function mikhailovS3THG(E_ev: number, EF_ev: number, gamma_ev: number): { full: Cx; intra: Cx } {
  return sigma3Unsym(Osub(E_ev, EF_ev, gamma_ev), Osub(2 * E_ev, EF_ev, gamma_ev), Osub(3 * E_ev, EF_ev, gamma_ev));
}

// Kerr / self-phase modulation (ω,ω,−ω): symmetrized average of the 3 distinct frequency
// orderings — these hit the coincident-argument f_int branches (O1 = O123). E, E_F, γ in eV.
export function mikhailovS3Kerr(E_ev: number, EF_ev: number, gamma_ev: number): { full: Cx; intra: Cx } {
  const Ow = Osub(E_ev, EF_ev, gamma_ev), O0 = Osub(0, EF_ev, gamma_ev), Onw = Osub(-E_ev, EF_ev, gamma_ev), O2w = Osub(2 * E_ev, EF_ev, gamma_ev);
  const p1 = sigma3Unsym(Ow, O2w, Ow);   // (ω, ω, −ω)
  const p2 = sigma3Unsym(Ow, O0, Ow);    // (ω, −ω, ω)
  const p3 = sigma3Unsym(Onw, O0, Ow);   // (−ω, ω, ω)
  return { full: cScale(sum(p1.full, p2.full, p3.full), 1 / 3), intra: cScale(sum(p1.intra, p2.intra, p3.intra), 1 / 3) };
}
