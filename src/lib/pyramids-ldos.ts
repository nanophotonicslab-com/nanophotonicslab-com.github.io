// Layered-media LDOS engine — a faithful TypeScript port of the electric /
// magnetic / magnetoelectric LDOS in PyRAMIDS
// (github.com/AMOLFResonantNanophotonics/PyRAMIDS, Pal & Koenderink), which in
// turn follows Amos & Barnes, Phys. Rev. B 55, 7249 (1997).
//
// Everything is in "user-centric" coordinates: the first interface sits at
// z = 0 and finite layers of thickness d1..d_{m-1} stack upward, between a
// semi-infinite substrate n0 (below, z<0) and superstrate n_m (above).
//
//   k0     = 2*pi / lambda_vac
//   nstack = [n0, n1, ..., n_{m}]      (complex; ends are semi-infinite)
//   dstack = [d1, ..., d_{m-1}]        (finite-layer thicknesses)
//
// LDOS is returned normalized to the vacuum LDOS (so free space → 1).
// Validated against PyRAMIDS in pyramids-ldos.test.ts.

export type Cx = { re: number; im: number };

export const cx = (re: number, im = 0): Cx => ({ re, im });
const add = (a: Cx, b: Cx): Cx => ({ re: a.re + b.re, im: a.im + b.im });
const sub = (a: Cx, b: Cx): Cx => ({ re: a.re - b.re, im: a.im - b.im });
const mul = (a: Cx, b: Cx): Cx => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const scale = (a: Cx, s: number): Cx => ({ re: a.re * s, im: a.im * s });
const neg = (a: Cx): Cx => ({ re: -a.re, im: -a.im });
const conj = (a: Cx): Cx => ({ re: a.re, im: -a.im });
const abs2 = (a: Cx): number => a.re * a.re + a.im * a.im;
function div(a: Cx, b: Cx): Cx {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
}
function inv(b: Cx): Cx {
  const d = b.re * b.re + b.im * b.im;
  return { re: b.re / d, im: -b.im / d };
}
function cexp(a: Cx): Cx {
  const e = Math.exp(a.re);
  return { re: e * Math.cos(a.im), im: e * Math.sin(a.im) };
}
// principal-branch complex sqrt (matches numpy)
function csqrt(a: Cx): Cx {
  const r = Math.hypot(a.re, a.im);
  if (r === 0) return { re: 0, im: 0 };
  const re = Math.sqrt((r + a.re) / 2);
  let im = Math.sqrt((r - a.re) / 2);
  if (a.im < 0) im = -im;
  return { re, im };
}

// ── S-matrix building blocks (Lifeng Li, JOSA A 13, 1024) ──

const I = cx(0, 1);

// perpendicular wavevector kz = sqrt(n^2 k0^2 - kpar^2), principal branch
function kz(n: Cx, k0: number, kpar: Cx): Cx {
  const nk = scale(mul(n, n), k0 * k0);
  return add(cx(1e-23), csqrt(sub(nk, mul(kpar, kpar))));
}

type S4 = [Cx, Cx, Cx, Cx]; // s11, s12, s21, s22

// Redheffer star product of two scalar S-matrices
function dotstar(A: S4, B: S4): S4 {
  const [a11, a12, a21, a22] = A;
  const [b11, b12, b21, b22] = B;
  const denom = inv(sub(cx(1), mul(a12, b21)));
  return [
    mul(mul(b11, a11), denom),
    add(b12, mul(mul(mul(b11, a12), b22), denom)),
    add(a21, mul(mul(mul(a22, a11), b21), denom)),
    mul(mul(a22, b22), denom),
  ];
}

// interface t-matrix (s: sorp=true; p: sorp=false)
function layerTsp(n1: Cx, n2: Cx, k0: number, kpar: Cx, sorp: boolean): S4 {
  const f2 = sorp ? cx(1) : inv(mul(n2, n2));
  const f1 = sorp ? cx(1) : inv(mul(n1, n1));
  const kz2 = mul(kz(n2, k0, kpar), f2);
  const kz1 = mul(kz(n1, k0, kpar), f1);
  const x = scale(div(kz1, kz2), 0.5);
  const half = cx(0.5);
  return [add(half, x), sub(half, x), sub(half, x), add(half, x)];
}

function interfacesSp(n1: Cx, n2: Cx, k0: number, kpar: Cx, sorp: boolean): S4 {
  const [t11, t12, t21, t22] = layerTsp(n1, n2, k0, kpar, sorp);
  const it22 = inv(t22);
  return [sub(t11, mul(t12, mul(t21, it22))), mul(t12, it22), neg(mul(t21, it22)), it22];
}

function layerSsp(n1: Cx, n2: Cx, k0: number, kpar: Cx, d: number, sorp: boolean): S4 {
  const kzz = kz(n1, k0, kpar);
  const ezd = cexp(mul(I, scale(kzz, d)));
  const [s11, s12, s21, s22] = interfacesSp(n1, n2, k0, kpar, sorp);
  return [mul(s11, ezd), s12, mul(s21, mul(ezd, ezd)), mul(s22, ezd)];
}

export type ND = { n: Cx; d: number };

// recursive full-stack S-matrix; n2/n3 are the semi-infinite ends,
// ndlist the finite layers (counting away from n2 towards n3)
function recurSsp(k0: number, kpar: Cx, n2: Cx, n3: Cx, ndlist: ND[], sorp: boolean): S4 {
  let S: S4 = [cx(1), cx(0), cx(0), cx(1)];
  const nlist: Cx[] = [n2, ...ndlist.map((x) => x.n), n3];
  const dlist: number[] = [0, ...ndlist.map((x) => x.d)];
  for (let m = 0; m < dlist.length; m++) {
    S = dotstar(S, layerSsp(nlist[m], nlist[m + 1], k0, kpar, dlist[m], sorp));
  }
  return S;
}

// reflection coefficient only (input side n2)
function rS(k0: number, kpar: Cx, n2: Cx, n3: Cx, ndlist: ND[]): Cx {
  return recurSsp(k0, kpar, n2, n3, ndlist, true)[2]; // S21
}
function rP(k0: number, kpar: Cx, n2: Cx, n3: Cx, ndlist: ND[]): Cx {
  return neg(recurSsp(k0, kpar, n2, n3, ndlist, false)[2]); // -S21 (recast H→E)
}
// reflection + transmission (matching PyRAMIDS rt_s / rt_p)
function rtSFull(k0: number, kpar: Cx, n2: Cx, n3: Cx, ndlist: ND[]): { r: Cx; t: Cx } {
  const S = recurSsp(k0, kpar, n2, n3, ndlist, true);
  return { r: S[2], t: S[0] };
}
function rtPFull(k0: number, kpar: Cx, n2: Cx, n3: Cx, ndlist: ND[]): { r: Cx; t: Cx } {
  const S = recurSsp(k0, kpar, n2, n3, ndlist, false);
  return { r: neg(S[2]), t: mul(S[0], div(n2, n3)) }; // recast H→E
}

// ── Amos & Barnes integrand: 5 channels [E_par, E_perp, M_par, M_perp, C] ──
// u is the dimensionless parallel momentum (kpar = nslab*k0*u).
function amosIntegrand(
  u: Cx, k0: number, z: number, dslab: number, nslab: Cx,
  n2: Cx, n3: Cx, nd2: ND[], nd3: ND[],
): [Cx, Cx, Cx, Cx, Cx] {
  // l = -i sqrt(1 - u^2)
  const l = mul(neg(I), csqrt(sub(cx(1), mul(u, u))));
  const kpar = scale(mul(nslab, u), k0);
  const twoNk = 2 * k0;
  const eb1 = cexp(scale(mul(mul(nslab, l), cx(-z * twoNk)), 1)); // exp(-2 nslab k0 l z)
  const eb2 = cexp(scale(mul(mul(nslab, l), cx(-(dslab - z) * twoNk)), 1));
  const eb12 = mul(eb1, eb2);

  let r12p = rP(k0, kpar, nslab, n2, nd2);
  let r13p = rP(k0, kpar, nslab, n3, nd3);
  let r12s = rS(k0, kpar, nslab, n2, nd2);
  let r13s = rS(k0, kpar, nslab, n3, nd3);

  const u2 = mul(u, u);
  const u3ol = div(mul(u, u2), l);
  const uol = div(u, l);
  const one = cx(1);

  // resonant denominators
  const dp = sub(one, mul(eb12, mul(r12p, r13p)));
  const ds = sub(one, mul(eb12, mul(r12s, r13s)));

  const zPperp = mul(u3ol, div(mul(sub(one, mul(eb1, r12p)), sub(one, mul(eb2, r13p))), dp));
  const zPpar = mul(uol, add(
    div(mul(add(one, mul(eb1, r12s)), add(one, mul(eb2, r13s))), ds),
    mul(sub(one, u2), div(mul(add(one, mul(eb1, r12p)), add(one, mul(eb2, r13p))), dp)),
  ));
  const zC = mul(u, sub(
    div(mul(sub(one, mul(eb1, r12s)), add(one, mul(eb2, r13s))), ds),
    div(mul(add(one, mul(eb1, r12p)), sub(one, mul(eb2, r13p))), dp),
  ));

  // magnetic: s<->p and every reflection coeff gains a minus sign
  const nr12p = neg(r12s), nr12s = neg(r12p), nr13p = neg(r13s), nr13s = neg(r13p);
  const dpm = sub(one, mul(eb12, mul(nr12p, nr13p)));
  const dsm = sub(one, mul(eb12, mul(nr12s, nr13s)));
  const zMperp = mul(u3ol, div(mul(sub(one, mul(eb1, nr12p)), sub(one, mul(eb2, nr13p))), dpm));
  const zMpar = mul(uol, add(
    div(mul(add(one, mul(eb1, nr12s)), add(one, mul(eb2, nr13s))), dsm),
    mul(sub(one, u2), div(mul(add(one, mul(eb1, nr12p)), add(one, mul(eb2, nr13p))), dpm)),
  ));

  return [scale(zPpar, 0.75), scale(zPperp, 1.5), scale(zMpar, 0.75), scale(zMperp, 1.5), scale(zC, 0.75)];
}

// Paulus contour (Phys. Rev. E 62, 5797): a flattened ellipse from 0 to kmax
// that bypasses guided-mode poles just above the real axis.
function trajectory(t: number, kmin: number, kmax: number): { x: Cx; v: Cx } {
  const ct = Math.cos(t), st = Math.sin(t);
  return {
    x: { re: kmax * (0.5 + 0.5 * ct), im: -kmax * kmin * st },
    v: { re: -0.5 * st * kmax, im: -kmin * ct * kmax },
  };
}

// adaptive Simpson for a vector-valued real integrand (shared nodes, max-error driven)
function adaptiveSimpsonVec(
  f: (x: number) => number[], a: number, b: number, tol: number, n: number, maxDepth = 40,
): number[] {
  const fa = f(a), fb = f(b), m0 = (a + b) / 2, fm = f(m0);
  const whole = simp(fa, fm, fb, a, b);
  const rec = (a: number, b: number, fa: number[], fm: number[], fb: number[], whole: number[], tol: number, depth: number): number[] => {
    const lm = (a + b) / 2, ml = (a + lm) / 2, mr = (lm + b) / 2;
    const fml = f(ml), fmr = f(mr);
    const left = simp(fa, fml, fm, a, lm), right = simp(fm, fmr, fb, lm, b);
    const out = new Array(n).fill(0);
    let err = 0;
    for (let i = 0; i < n; i++) {
      const lr = left[i] + right[i];
      err = Math.max(err, Math.abs(lr - whole[i]));
      out[i] = lr + (lr - whole[i]) / 15;
    }
    if (depth <= 0 || err < 15 * tol) return out;
    const l = rec(a, lm, fa, fml, fm, left, tol / 2, depth - 1);
    const r = rec(lm, b, fm, fmr, fb, right, tol / 2, depth - 1);
    return l.map((v, i) => v + r[i]);
  };
  function simp(fa: number[], fm: number[], fb: number[], a: number, b: number): number[] {
    const h6 = (b - a) / 6;
    return fa.map((_, i) => h6 * (fa[i] + 4 * fm[i] + fb[i]));
  }
  return rec(a, b, fa, fm, fb, whole, tol, maxDepth);
}

// LDOS in a single slab (5 channels), integrating the Amos–Barnes integrand
// along the Paulus contour [0,pi] then along the real axis [kmax,∞).
function ldosInLayer(
  k0: number, z: number, dslab: number, nslab: Cx, n2: Cx, n3: Cx, nd2: ND[], nd3: ND[],
): number[] {
  const kmin = 0.01;
  let kmax = 2 * Math.max(
    Math.abs(n2.re), Math.abs(n3.re),
    ...nd2.map((x) => Math.abs(x.n.re)), ...nd3.map((x) => Math.abs(x.n.re)),
  );
  if (!Number.isFinite(kmax) || kmax < 2) kmax = 2;

  // contour part: integrand = imag(-v * F(x(t)))
  const contour = (t: number): number[] => {
    const { x, v } = trajectory(t, kmin, kmax);
    const F = amosIntegrand(x, k0, z, dslab, nslab, n2, n3, nd2, nd3);
    return F.map((Fi) => {
      const p = mul(neg(v), Fi);
      return p.im;
    });
  };
  const zt1 = adaptiveSimpsonVec(contour, 0, Math.PI, 1e-6, 5);

  // tail [kmax,∞): u = kmax + s/(1-s), du = ds/(1-s)^2; integrand = imag(F(u))
  const tail = (s: number): number[] => {
    const om = 1 - s;
    const u = kmax + s / om;
    const jac = 1 / (om * om);
    const F = amosIntegrand(cx(u), k0, z, dslab, nslab, n2, n3, nd2, nd3);
    return F.map((Fi) => Fi.im * jac);
  };
  const zt2 = adaptiveSimpsonVec(tail, 0, 1 - 1e-9, 1e-6, 5);

  return zt1.map((v, i) => nslab.re * (v + zt2[i]));
}

// ── user-centric → slab-centric coordinate wrapper ──

export type LDOSResult = {
  E_par: number[]; E_perp: number[]; M_par: number[]; M_perp: number[]; C: number[];
};

function pinpointDomain(z: number, dstack: number[]): number {
  // domain 0 = substrate half-space (z<0); domain numLayers+1 = superstrate.
  // A point exactly on an interface is assigned to the UPPER domain, matching
  // PyRAMIDS' pinpointdomain tie-break (inclusive ascending loop, last match wins).
  if (z < 0) return 0;
  let acc = 0;
  for (let i = 0; i < dstack.length; i++) {
    acc += dstack[i];
    if (z < acc) return i + 1;
  }
  return dstack.length + 1;
}

// returns slab-centric params for a point at user coordinate z in domain m
function provideCoordinates(m: number, z: number, nstack: Cx[], dstack: number[]) {
  const n2 = nstack[0], n3 = nstack[nstack.length - 1];
  const interior = nstack.slice(1, -1);
  const stacksize = dstack.reduce((a, b) => a + b, 0);
  const numDomains = dstack.length + 2;
  const overhang = Math.max(-z, z - stacksize, 0) || 1; // finite fictitious slab for half-spaces
  const zBoundary = (i: number) => dstack.slice(0, i).reduce((a, b) => a + b, 0);

  if (m === 0) {
    // substrate half-space (z<0)
    const dslab = 2 * overhang;
    return { zz: dslab + z, dslab, nslab: n2, n2, n3,
      nd2: [] as ND[], nd3: interior.map((n, i) => ({ n, d: dstack[i] })) };
  }
  if (m === numDomains - 1) {
    // superstrate half-space (z>stack)
    const dslab = 2 * overhang;
    const rev = interior.map((n, i) => ({ n, d: dstack[i] })).reverse();
    return { zz: z - stacksize, dslab, nslab: n3, n2, n3, nd2: rev, nd3: [] as ND[] };
  }
  // finite layer m (1-based interior index m-1)
  const dslab = dstack[m - 1];
  const nslab = interior[m - 1];
  const zz = z - zBoundary(m - 1);
  const nd2 = interior.slice(0, m - 1).map((n, i) => ({ n, d: dstack[i] })).reverse();
  const nd3 = interior.slice(m).map((n, i) => ({ n, d: dstack[m + i] }));
  return { zz, dslab, nslab, n2, n3, nd2, nd3 };
}

function realPositive(n: Cx): boolean {
  return Math.abs(n.im) < 1e-9 && n.re > 0;
}

/** Canonical LDOS channels (normalized to vacuum) at each z in `zlist`. */
export function ldos(k0: number, zlist: number[], nstack: Cx[], dstack: number[]): LDOSResult {
  const out: LDOSResult = { E_par: [], E_perp: [], M_par: [], M_perp: [], C: [] };
  for (const z of zlist) {
    const m = pinpointDomain(z, dstack);
    const c = provideCoordinates(m, z, nstack, dstack);
    if (!realPositive(c.nslab)) {
      // LDOS is undefined for an absorbing source layer — emit 0 (as PyRAMIDS does)
      out.E_par.push(0); out.E_perp.push(0); out.M_par.push(0); out.M_perp.push(0); out.C.push(0);
      continue;
    }
    const ld = ldosInLayer(k0, c.zz, c.dslab, c.nslab, c.n2, c.n3, c.nd2, c.nd3);
    out.E_par.push(ld[0]); out.E_perp.push(ld[1]); out.M_par.push(ld[2]); out.M_perp.push(ld[3]); out.C.push(ld[4]);
  }
  return out;
}

/** LDOS projected onto arbitrary electric (pu) and magnetic (mu) dipole vectors. */
export function ldosAtAnyPandM(pu: Cx[], mu: Cx[], k0: number, zlist: number[], nstack: Cx[], dstack: number[]): number[] {
  const r = ldos(k0, zlist, nstack, dstack);
  const norm = pu.reduce((a, p) => a + abs2(p), 0) + mu.reduce((a, mm) => a + abs2(mm), 0);
  return zlist.map((_, i) => {
    let v = (abs2(pu[0]) + abs2(pu[1])) * r.E_par[i] + abs2(pu[2]) * r.E_perp[i];
    v += (abs2(mu[0]) + abs2(mu[1])) * r.M_par[i] + abs2(mu[2]) * r.M_perp[i];
    // 2*rhoC * Im(conj(p_y) m_x - p_x conj(m_y))
    const t = sub(mul(conj(pu[1]), mu[0]), mul(pu[0], conj(mu[1])));
    v += 2 * r.C[i] * t.im;
    return v / norm;
  });
}

// ══════════════════════════════════════════════════════════════════════════
// Far-field radiation pattern (PyRAMIDS Core_Radiationpattern PEup / PEdown)
// Asymptotic dyadic Green function; Novotny & Hecht (1st ed.) Eq. 10.32/10.36
// with full multiple-reflection denominators. Power normalized to the total
// power of the same dipole in free space (prefactor 3/(8π)·nslab).
// ══════════════════════════════════════════════════════════════════════════

// far field into the UPPER half-space n3 (θ measured from +z, cosθ > 0)
function peUp(
  pu: Cx[], mu: Cx[], th: number, ph: number, k0: number,
  h: number, dslab: number, nslab: Cx, n2: Cx, n3: Cx, nd2: ND[], nd3: ND[],
): number {
  const kpar = cx(k0 * n3.re * Math.sin(th));
  const kzs = kz(nslab, k0, kpar), kz3 = kz(n3, k0, kpar);
  const ed = cexp(mul(I, scale(kzs, dslab)));
  const eh = cexp(mul(I, scale(kzs, h)));
  const P1 = div(ed, eh), P2 = mul(ed, eh), P3 = ed;
  const one = cx(1);
  const { r: rp3, t: tp3 } = rtPFull(k0, kpar, nslab, n3, nd3);
  const { r: rs3, t: ts3 } = rtSFull(k0, kpar, nslab, n3, nd3);
  const rp2 = rP(k0, kpar, nslab, n2, nd2);
  const rs2 = rS(k0, kpar, nslab, n2, nd2);
  const P3sq = mul(P3, P3);
  const prep = div(tp3, sub(one, mul(mul(rp3, rp2), P3sq)));
  const pres = div(ts3, sub(one, mul(mul(rs3, rs2), P3sq)));
  const gtpp = mul(prep, sub(P1, mul(P2, rp2)));
  const gtpm = mul(prep, add(P1, mul(P2, rp2)));
  const gtsp = mul(pres, add(P1, mul(P2, rs2)));
  const gtsm = mul(pres, sub(P1, mul(P2, rs2)));
  const sq = csqrt(div(n3, nslab));
  const ratio = div(kz3, kzs);
  const nr = div(n3, nslab);
  const st = Math.sin(th), ct = Math.cos(th), sp = Math.sin(ph), cp = Math.cos(ph);
  // E from electric dipole
  const EsP = mul(mul(mul(sq, ratio), gtsp), add(scale(pu[0], -sp), scale(pu[1], cp)));
  const EpP = mul(mul(sq, nr), add(
    scale(mul(mul(ratio, gtpp), pu[2]), -st),
    scale(mul(gtpm, add(scale(pu[0], cp), scale(pu[1], sp))), ct)));
  // E from magnetic dipole (s↔p swap + sign, see PyRAMIDS)
  const EpM = neg(mul(mul(mul(sq, ratio), gtpp), add(scale(mu[0], sp), scale(mu[1], cp))));
  const EsM = neg(mul(mul(sq, nr), add(
    scale(mul(mul(ratio, gtsp), mu[2]), -st),
    scale(mul(gtsm, sub(scale(mu[0], cp), scale(mu[1], sp))), ct))));
  const Es = add(EsP, EsM), Ep = add(EpP, EpM);
  return (3 / (8 * Math.PI)) * nslab.re * (abs2(Es) + abs2(Ep));
}

// far field into the LOWER half-space n2 (cosθ ≤ 0; sinθ ≥ 0)
function peDown(
  pu: Cx[], mu: Cx[], th: number, ph: number, k0: number,
  h: number, dslab: number, nslab: Cx, n2: Cx, n3: Cx, nd2: ND[], nd3: ND[],
): number {
  const kpar = cx(k0 * n2.re * Math.sin(th));
  const kzs = kz(nslab, k0, kpar), kz2 = kz(n2, k0, kpar);
  const ed = cexp(mul(I, scale(kzs, dslab)));
  const eh = cexp(mul(I, scale(kzs, h)));
  const P3 = ed, P4 = div(mul(ed, ed), eh), P5 = eh;
  const one = cx(1);
  const { r: rp2, t: tp2 } = rtPFull(k0, kpar, nslab, n2, nd2);
  const { r: rs2, t: ts2 } = rtSFull(k0, kpar, nslab, n2, nd2);
  const rp3 = rP(k0, kpar, nslab, n3, nd3);
  const rs3 = rS(k0, kpar, nslab, n3, nd3);
  const P3sq = mul(P3, P3);
  const prep = div(tp2, sub(one, mul(mul(rp3, rp2), P3sq)));
  const pres = div(ts2, sub(one, mul(mul(rs3, rs2), P3sq)));
  const gtpp = mul(prep, sub(P5, mul(P4, rp3)));
  const gtpm = mul(prep, add(P5, mul(P4, rp3)));
  const gtsp = mul(pres, add(P5, mul(P4, rs3)));
  const gtsm = mul(pres, sub(P5, mul(P4, rs3)));
  const sq = csqrt(div(n2, nslab));
  const ratio = div(kz2, kzs);
  const nr = div(n2, nslab);
  const st = Math.sin(th), ct = Math.cos(th), sp = Math.sin(ph), cp = Math.cos(ph);
  const EsP = mul(mul(mul(sq, ratio), gtsp), add(scale(pu[0], sp), scale(pu[1], cp)));
  const EpP = mul(mul(sq, nr), add(
    scale(mul(mul(ratio, gtpp), pu[2]), -st),
    scale(mul(gtpm, sub(scale(pu[0], cp), scale(pu[1], sp))), ct)));
  const EpM = neg(mul(mul(mul(sq, ratio), gtpp), add(scale(mu[0], -sp), scale(mu[1], cp))));
  const EsM = neg(mul(mul(sq, nr), add(
    scale(mul(mul(ratio, gtsp), mu[2]), -st),
    scale(mul(gtsm, add(scale(mu[0], cp), scale(mu[1], sp))), ct))));
  const Es = add(EsP, EsM), Ep = add(EpP, EpM);
  return (3 / (8 * Math.PI)) * nslab.re * (abs2(Es) + abs2(Ep));
}

/**
 * Angle-resolved far-field power of a dipole (pu, mu) at user-centric height z.
 * θ ∈ [0, π] measured from +z: cosθ > 0 radiates into the superstrate (top),
 * cosθ ≤ 0 into the substrate. Absorbing half-spaces (and an absorbing source
 * layer) contribute zero. Power per solid angle, normalized to the free-space
 * total power of the same dipole.
 */
export function radiationPattern(
  k0: number, z: number, pu: Cx[], mu: Cx[],
  thetaList: number[], phiList: number[], nstack: Cx[], dstack: number[],
): number[] {
  const m = pinpointDomain(z, dstack);
  const c = provideCoordinates(m, z, nstack, dstack);
  const out = new Array(thetaList.length).fill(0);
  if (!realPositive(c.nslab)) return out;
  const upOk = realPositive(c.n3), downOk = realPositive(c.n2);
  for (let i = 0; i < thetaList.length; i++) {
    const th = thetaList[i], ph = phiList[i];
    if (Math.cos(th) > 0) {
      if (upOk) out[i] = peUp(pu, mu, th, ph, k0, c.zz, c.dslab, c.nslab, c.n2, c.n3, c.nd2, c.nd3);
    } else if (downOk) {
      out[i] = peDown(pu, mu, th, ph, k0, c.zz, c.dslab, c.nslab, c.n2, c.n3, c.nd2, c.nd3);
    }
  }
  return out;
}

// ══════════════════════════════════════════════════════════════════════════
// Plane-wave response (PyRAMIDS Use_Planewaves / S-matrix solver)
// Incidence from the FIRST stack entry (nstack[0], must be transparent).
// ══════════════════════════════════════════════════════════════════════════

/** Complex Fresnel amplitudes of the full stack at one (k0, k∥). */
export function fresnelAmplitudes(
  k0: number, kpar: number, nstack: Cx[], dstack: number[],
): { rs: Cx; ts: Cx; rp: Cx; tp: Cx } {
  const nin = nstack[0], nout = nstack[nstack.length - 1];
  const nd: ND[] = nstack.slice(1, -1).map((n, i) => ({ n, d: dstack[i] }));
  const kp = cx(kpar);
  const s = rtSFull(k0, kp, nin, nout, nd);
  const p = rtPFull(k0, kp, nin, nout, nd);
  return { rs: s.r, ts: s.t, rp: p.r, tp: p.t };
}

export type RTA = { R: number; T: number; A: number };

/** Intensity reflectance / transmittance / absorptance for s and p. */
export function intensityRT(
  k0: number, kpar: number, nstack: Cx[], dstack: number[],
): { s: RTA; p: RTA } {
  const nin = nstack[0], nout = nstack[nstack.length - 1];
  const { rs, ts, rp, tp } = fresnelAmplitudes(k0, kpar, nstack, dstack);
  const kp = cx(kpar);
  const proj = div(kz(nout, k0, kp), kz(nin, k0, kp)).re;
  const Rs = abs2(rs), Rp = abs2(rp);
  const Ts = abs2(ts) * proj, Tp = abs2(tp) * proj;
  return {
    s: { R: Rs, T: Ts, A: 1 - Rs - Ts },
    p: { R: Rp, T: Tp, A: 1 - Rp - Tp },
  };
}

// up/down amplitudes in target layer nlay (0 = input side, N+1 = output side)
function udcoefT(k0: number, kpar: Cx, nin: Cx, nout: Cx, nd: ND[], nlay: number, sorp: boolean): { u: Cx; d: Cx } {
  const S = recurSsp(k0, kpar, nin, nout, nd, sorp);
  const u0 = cx(1), d0 = S[2];
  const numlay = nd.length;
  if (nlay === 0) return { u: u0, d: d0 };
  if (nlay > numlay) return { u: S[0], d: cx(0) };
  const n3 = nd[nlay - 1].n;
  const partial = recurSsp(k0, kpar, nin, n3, nd.slice(0, nlay - 1), sorp);
  const d = div(sub(d0, mul(partial[2], u0)), partial[3]);
  const u = add(mul(partial[0], u0), mul(partial[1], d));
  return { u, d };
}

// E-field u/d coefficient triples in layer nlay: s → [Ey,Hx,Hz], p → [Ex,Ez,Hy]
function esFromUd(k0: number, kpar: Cx, nin: Cx, nout: Cx, nd: ND[], nlay: number): { u: Cx[]; d: Cx[] } {
  const { u: Euy, d: Edy } = udcoefT(k0, kpar, nin, nout, nd, nlay, true);
  const nlist = [nin, ...nd.map(x => x.n), nout];
  const nslab = nlist[nlay];
  const kzs = kz(nslab, k0, kpar);
  const iNin = inv(nin);
  return {
    u: [Euy, neg(mul(mul(Euy, scale(kzs, 1 / k0)), iNin)), mul(mul(Euy, scale(kpar, 1 / k0)), iNin)],
    d: [Edy, mul(mul(Edy, scale(kzs, 1 / k0)), iNin), mul(mul(Edy, scale(kpar, 1 / k0)), iNin)],
  };
}
function epFromUd(k0: number, kpar: Cx, nin: Cx, nout: Cx, nd: ND[], nlay: number): { u: Cx[]; d: Cx[] } {
  const { u: Huy, d: Hdy } = udcoefT(k0, kpar, nin, nout, nd, nlay, false);
  const nlist = [nin, ...nd.map(x => x.n), nout];
  const nslab = nlist[nlay];
  const kzs = kz(nslab, k0, kpar);
  const fac = mul(inv(mul(nslab, nslab)), scale(nin, 1 / k0)); // nin/(k0·nslab²)
  return {
    u: [mul(mul(Huy, kzs), fac), neg(mul(mul(Huy, kpar), fac)), Huy],
    d: [neg(mul(mul(Hdy, kzs), fac)), neg(mul(mul(Hdy, kpar), fac)), Hdy],
  };
}

export type PlaneWaveFields = {
  z: number[];
  /** s-pol: |Ey|(z); p-pol: sqrt(|Ex|²+|Ez|²)(z) — field magnitudes for unit incident E */
  Es: number[]; Ep: number[];
  /** local absorption density (per unit length) for s / p */
  As: number[]; Ap: number[];
};

/** Local field magnitudes and absorption density along z (x = y = 0). */
export function planeWaveFieldsAtZ(
  k0: number, kpar: number, nstack: Cx[], dstack: number[], zlist: number[],
): PlaneWaveFields {
  const nin = nstack[0], nout = nstack[nstack.length - 1];
  const nd: ND[] = nstack.slice(1, -1).map((n, i) => ({ n, d: dstack[i] }));
  const kp = cx(kpar);
  const kzin = kz(nin, k0, kp);
  const stacksize = dstack.reduce((a, b) => a + b, 0);
  const zBoundary = (i: number) => dstack.slice(0, i).reduce((a, b) => a + b, 0);
  const out: PlaneWaveFields = { z: zlist.slice(), Es: [], Ep: [], As: [], Ap: [] };
  // cache u/d per domain
  const cacheS = new Map<number, { u: Cx[]; d: Cx[] }>();
  const cacheP = new Map<number, { u: Cx[]; d: Cx[] }>();
  const nlist = [nin, ...nd.map(x => x.n), nout];
  for (const z of zlist) {
    const m = pinpointDomain(z, dstack);
    if (!cacheS.has(m)) { cacheS.set(m, esFromUd(k0, kp, nin, nout, nd, m)); cacheP.set(m, epFromUd(k0, kp, nin, nout, nd, m)); }
    const s = cacheS.get(m)!, p = cacheP.get(m)!;
    const nslab = nlist[m];
    const kzs = kz(nslab, k0, kp);
    // local coordinate: input halfspace uses the raw user z (interface at 0)
    const zz = m === 0 ? z : (m > nd.length ? z - stacksize : z - zBoundary(m - 1));
    const up = cexp(mul(I, scale(kzs, zz)));
    const dn = inv(up);
    const Ey = add(mul(s.u[0], up), mul(s.d[0], dn));
    const Ex = add(mul(p.u[0], up), mul(p.d[0], dn));
    const Ez = add(mul(p.u[1], up), mul(p.d[1], dn));
    const prefac = mul(nslab, nslab).im * (k0 * div(cx(k0), kzin).re);
    out.Es.push(Math.sqrt(abs2(Ey)));
    out.Ep.push(Math.sqrt(abs2(Ex) + abs2(Ez)));
    out.As.push(prefac * abs2(Ey));
    out.Ap.push(prefac * (abs2(Ex) + abs2(Ez)));
  }
  return out;
}

/**
 * Absorbed power fraction per finite layer (+ the output half-space as the
 * last entry) for s and p, for unit incident intensity.
 */
export function perLayerAbsorption(
  k0: number, kpar: number, nstack: Cx[], dstack: number[],
): { s: number[]; p: number[] } {
  const nin = nstack[0], nout = nstack[nstack.length - 1];
  const nd: ND[] = nstack.slice(1, -1).map((n, i) => ({ n, d: dstack[i] }));
  const kp = cx(kpar);
  const kzin = kz(nin, k0, kp);
  const absOne = (u: Cx, d: Cx, kzs: Cx, dslab: number): number => {
    const kprime = kzs.re, kpprime = kzs.im + 1e-23;
    const au = Math.sqrt(abs2(u)), ad = Math.sqrt(abs2(d));
    if (dslab > 0) {
      const e2 = Math.exp(2 * kpprime * dslab);
      const delta = Math.atan2(u.im, u.re) - Math.atan2(d.im, d.re);
      const a = (au * au * (1 - 1 / e2) + ad * ad * (e2 - 1)) / (2 * kpprime);
      const b = au * ad * (Math.sin(2 * kprime * dslab + delta) - Math.sin(delta)) / kprime;
      return a + b;
    }
    return (au * au) / (2 * kpprime);
  };
  const layers: ND[] = [...nd, { n: nout, d: -1 }];
  const s: number[] = [], p: number[] = [];
  layers.forEach((lay, m) => {
    const kzs = kz(lay.n, k0, kp);
    const prefac = mul(lay.n, lay.n).im * (k0 * div(cx(k0), kzin).re);
    const es = esFromUd(k0, kp, nin, nout, nd, m + 1);
    s.push(prefac * absOne(es.u[0], es.d[0], kzs, lay.d));
    const ep = epFromUd(k0, kp, nin, nout, nd, m + 1);
    p.push(prefac * (absOne(ep.u[0], ep.d[0], kzs, lay.d) + absOne(ep.u[1], ep.d[1], kzs, lay.d)));
  });
  return { s, p };
}

/**
 * k∥-resolved LDOS integrand (before integration), for modal / dispersion maps.
 * Returns data[channel][zIndex][kparIndex]; channels [E_par,E_perp,M_par,M_perp,C].
 * kparList is in units of k0.
 */
export function ldosIntegrandTrace(
  k0: number, kparList: number[], zList: number[], nstack: Cx[], dstack: number[], guideVisible = 1,
): number[][][] {
  const out: number[][][] = Array.from({ length: 5 }, () =>
    Array.from({ length: zList.length }, () => new Array(kparList.length).fill(0)));
  zList.forEach((z, zi) => {
    const m = pinpointDomain(z, dstack);
    const c = provideCoordinates(m, z, nstack, dstack);
    if (!realPositive(c.nslab)) return;
    kparList.forEach((kp, ki) => {
      // u = kpar/nslab - i*1e-4*guideVisible ; result scaled by Re(nslab)
      const u = { re: kp / c.nslab.re, im: -1e-4 * guideVisible };
      const F = amosIntegrand(u, k0, c.zz, c.dslab, c.nslab, c.n2, c.n3, c.nd2, c.nd3);
      for (let ch = 0; ch < 5; ch++) out[ch][zi][ki] = c.nslab.re * F[ch].im;
    });
  });
  return out;
}
