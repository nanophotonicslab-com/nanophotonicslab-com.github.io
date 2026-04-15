/**
 * Infinite cylinder electromagnetic scattering.
 *
 * Ported from Álvaro's MATLAB `cylinders.m` (classdef). Computes the 4×4
 * tangential-field matching matrix M(qa, ka, ε₁, ε_h, m) and finds the
 * guided-mode dispersion branches q_α(ω) for a given angular order m from
 * det(M) = 0 inside the guided window k_h < q < k_mat.
 *
 * Units: nm and eV externally, SI internally. ℏc ≈ 197.3269804 eV·nm.
 *
 * References:
 *   - Bohren & Huffman, "Absorption and Scattering of Light by Small Particles"
 *     (Wiley, 1983), §8, infinite cylinder expansion.
 *   - Abramowitz & Stegun §9.1.11 for integer-order Y_m series.
 *   - Álvaro's paper 047 for the matching matrix M (labels RT_s, RT_p).
 */

// ============================================================================
// Complex arithmetic (tuple [re, im])
// ============================================================================

export type C = [number, number];

export const Cx = {
  zero:  (): C => [0, 0],
  one:   (): C => [1, 0],
  i:     (): C => [0, 1],
  re:    (x: number): C => [x, 0],
  add:   (a: C, b: C): C => [a[0] + b[0], a[1] + b[1]],
  sub:   (a: C, b: C): C => [a[0] - b[0], a[1] - b[1]],
  neg:   (a: C): C        => [-a[0], -a[1]],
  mul:   (a: C, b: C): C => [a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]],
  scale: (a: C, s: number): C => [a[0]*s, a[1]*s],
  div:   (a: C, b: C): C => {
    const d = b[0]*b[0] + b[1]*b[1];
    return [(a[0]*b[0] + a[1]*b[1]) / d, (a[1]*b[0] - a[0]*b[1]) / d];
  },
  invReal: (a: C): C => {
    const d = a[0]*a[0] + a[1]*a[1];
    return [a[0]/d, -a[1]/d];
  },
  abs2:  (a: C): number => a[0]*a[0] + a[1]*a[1],
  abs:   (a: C): number => Math.hypot(a[0], a[1]),
  arg:   (a: C): number => Math.atan2(a[1], a[0]),
  sqrt:  (a: C): C => {
    // Principal branch.
    if (a[0] === 0 && a[1] === 0) return [0, 0];
    const r = Math.hypot(a[0], a[1]);
    const re = Math.sqrt((r + a[0]) / 2);
    const im = (a[1] >= 0 ? 1 : -1) * Math.sqrt((r - a[0]) / 2);
    return [re, im];
  },
  log:   (a: C): C => [Math.log(Math.hypot(a[0], a[1])), Math.atan2(a[1], a[0])],
  exp:   (a: C): C => {
    const e = Math.exp(a[0]);
    return [e * Math.cos(a[1]), e * Math.sin(a[1])];
  },
};

// ============================================================================
// Bessel functions of integer order, complex argument
// ============================================================================

const EULER_GAMMA = 0.5772156649015329;

/**
 * J_m(z) for integer m ≥ 0 and complex z, via power series
 * J_m(z) = Σ_{k=0}^∞ (-1)^k / (k!(m+k)!) · (z/2)^{m+2k}.
 * Accurate for |z| ≲ 25 with double precision; cylinder sizes of interest give
 * arguments well below that.
 */
export function besselJ(m: number, z: C): C {
  if (m < 0) {
    const y = besselJ(-m, z);
    return m % 2 === 0 ? y : Cx.neg(y);
  }
  const half: C = [z[0] / 2, z[1] / 2];
  const halfSq = Cx.mul(half, half);
  // First term: (z/2)^m / m!
  let term: C = [1, 0];
  for (let i = 0; i < m; i++) term = Cx.mul(term, half);
  let mFact = 1;
  for (let i = 2; i <= m; i++) mFact *= i;
  term = Cx.scale(term, 1 / mFact);
  let sum: C = [term[0], term[1]];
  for (let k = 1; k < 120; k++) {
    term = Cx.mul(term, halfSq);
    term = Cx.scale(term, -1 / (k * (m + k)));
    sum = Cx.add(sum, term);
    if (Cx.abs2(term) < 1e-32 * (Cx.abs2(sum) + 1e-300)) break;
  }
  return sum;
}

/**
 * Y_m(z) for integer m ≥ 0 and complex z, via A&S 9.1.11.
 * Y_m(z) = -(1/π) Σ_{k=0}^{m-1} (m-k-1)!/k! · (z/2)^{2k-m}
 *        + (2/π) (ln(z/2) + γ) J_m(z)
 *        - (1/π) Σ_{k=0}^∞ (-1)^k (h_k + h_{m+k}) / (k!(m+k)!) · (z/2)^{m+2k}
 * where h_0 = 0, h_n = 1 + 1/2 + … + 1/n.
 */
export function besselY(m: number, z: C): C {
  if (m < 0) {
    const y = besselY(-m, z);
    return m % 2 === 0 ? y : Cx.neg(y);
  }
  const half: C = [z[0] / 2, z[1] / 2];
  const halfSq = Cx.mul(half, half);
  const logHalf = Cx.log(half);
  const logPlusGamma: C = [logHalf[0] + EULER_GAMMA, logHalf[1]];

  // Term A: -(1/π) Σ_{k=0}^{m-1} (m-k-1)!/k! · (z/2)^{2k-m}
  let termA: C = [0, 0];
  if (m > 0) {
    // (z/2)^{-m}
    let pw: C = [1, 0];
    for (let i = 0; i < m; i++) pw = Cx.div(pw, half);
    let facMk = 1; // (m-1)! for k=0
    for (let i = 2; i <= m - 1; i++) facMk *= i;
    let kFact = 1;
    for (let k = 0; k < m; k++) {
      const coef = facMk / kFact;
      termA = Cx.add(termA, Cx.scale(pw, coef));
      // Next: k → k+1, pw *= halfSq, facMk /= (m-k-1), kFact *= (k+1)
      pw = Cx.mul(pw, halfSq);
      if (m - k - 1 > 0) facMk /= (m - k - 1);
      kFact *= (k + 1);
    }
    termA = Cx.scale(termA, -1 / Math.PI);
  }

  // Term B: (2/π) (log(z/2) + γ) J_m(z)
  const Jm = besselJ(m, z);
  const termB = Cx.scale(Cx.mul(logPlusGamma, Jm), 2 / Math.PI);

  // Term C: -(1/π) Σ_{k=0}^∞ (-1)^k (h_k + h_{m+k}) / (k!(m+k)!) · (z/2)^{m+2k}
  let termC: C = [0, 0];
  // Initial (z/2)^m / (0! m!)
  let pw3: C = [1, 0];
  for (let i = 0; i < m; i++) pw3 = Cx.mul(pw3, half);
  let kFact3 = 1;
  let mkFact3 = 1;
  for (let i = 2; i <= m; i++) mkFact3 *= i;
  let hK = 0;        // h_0 = 0
  let hMK = 0;       // h_m = 1 + 1/2 + ... + 1/m
  for (let i = 1; i <= m; i++) hMK += 1 / i;
  for (let k = 0; k < 200; k++) {
    const coef = (hK + hMK) / (kFact3 * mkFact3);
    const sign = k % 2 === 0 ? 1 : -1;
    const contrib = Cx.scale(pw3, sign * coef);
    termC = Cx.add(termC, contrib);
    if (Cx.abs2(contrib) < 1e-32 * (Cx.abs2(termC) + 1e-300) && k > 3) break;
    // k → k+1
    pw3 = Cx.mul(pw3, halfSq);
    kFact3 *= (k + 1);
    mkFact3 *= (m + k + 1);
    hK += 1 / (k + 1);
    hMK += 1 / (m + k + 1);
  }
  termC = Cx.scale(termC, -1 / Math.PI);

  return Cx.add(Cx.add(termA, termB), termC);
}

/** Hankel function of the first kind, H_m^(1)(z) = J_m(z) + i Y_m(z). */
export function besselH1(m: number, z: C): C {
  const J = besselJ(m, z);
  const Y = besselY(m, z);
  return [J[0] - Y[1], J[1] + Y[0]];
}

/** J_m'(z) = (m/z) J_m(z) - J_{m+1}(z). */
export function besselJ_d(m: number, z: C): C {
  const Jm = besselJ(m, z);
  const Jm1 = besselJ(m + 1, z);
  const mOverZ = Cx.div([m, 0], z);
  return Cx.sub(Cx.mul(mOverZ, Jm), Jm1);
}

/** H_m^(1)'(z) = (m/z) H_m^(1)(z) - H_{m+1}^(1)(z). */
export function besselH1_d(m: number, z: C): C {
  const Hm = besselH1(m, z);
  const Hm1 = besselH1(m + 1, z);
  const mOverZ = Cx.div([m, 0], z);
  return Cx.sub(Cx.mul(mOverZ, Hm), Hm1);
}

// ============================================================================
// 4×4 matching matrix (after Álvaro's give_M)
// ============================================================================

export type Mat4 = C[][];

/**
 * 4×4 EM matching matrix at the cylinder surface.
 * Inputs: qa = q·a (real), ka = (ω/c)·a (real),
 *         eps1 cylinder dielectric, eps_h host dielectric, m angular order.
 * The guided-mode condition is det(M) = 0.
 */
export function giveM(qa: number, ka: number, eps1: C, eps_h: C, m: number): Mat4 {
  const eps1_over_h = Cx.div(eps1, eps_h);
  const chi = Cx.sqrt(eps1_over_h);
  const k1a = Cx.scale(Cx.sqrt(eps1), ka);
  const kha = Cx.scale(Cx.sqrt(eps_h), ka);

  // Q1a = sqrt(k1a² − qa²) with Re ≥ 0 convention.
  let Q1a = Cx.sqrt(Cx.sub(Cx.mul(k1a, k1a), [qa * qa, 0]));
  if (Q1a[0] < 0) Q1a = Cx.neg(Q1a);
  let Qha = Cx.sqrt(Cx.sub(Cx.mul(kha, kha), [qa * qa, 0]));
  if (Qha[0] < 0) Qha = Cx.neg(Qha);

  const Jma = besselJ(m, Q1a);
  const Hmh = besselH1(m, Qha);
  const Jmap = besselJ_d(m, Q1a);
  const Hmhp = besselH1_d(m, Qha);

  // Block helpers
  const Q1aOverK1a = Cx.div(Q1a, k1a);
  const QhaOverKha = Cx.div(Qha, kha);
  const chi_Q1aOverK1a = Cx.mul(chi, Q1aOverK1a);

  const mqa: C = [m * qa, 0];
  const mqaOverK1aQ1a = Cx.div(mqa, Cx.mul(k1a, Q1a));
  const mqaOverKhaQha = Cx.div(mqa, Cx.mul(kha, Qha));
  const chi_mqaOverK1aQ1a = Cx.mul(chi, mqaOverK1aQ1a);

  const M: Mat4 = [
    [Cx.zero(), Cx.zero(), Cx.zero(), Cx.zero()],
    [Cx.zero(), Cx.zero(), Cx.zero(), Cx.zero()],
    [Cx.zero(), Cx.zero(), Cx.zero(), Cx.zero()],
    [Cx.zero(), Cx.zero(), Cx.zero(), Cx.zero()],
  ];

  M[0][0] = Cx.mul(chi_Q1aOverK1a, Jma);
  M[0][1] = Cx.neg(Cx.mul(QhaOverKha, Hmh));
  M[1][0] = Jmap;
  M[1][1] = Cx.neg(Hmhp);
  M[1][2] = Cx.mul(mqaOverK1aQ1a, Jma);
  M[1][3] = Cx.neg(Cx.mul(mqaOverKhaQha, Hmh));
  M[2][2] = Cx.mul(Q1aOverK1a, Jma);
  M[2][3] = M[0][1];
  M[3][0] = Cx.mul(chi_mqaOverK1aQ1a, Jma);
  M[3][1] = Cx.neg(Cx.mul(mqaOverKhaQha, Hmh));
  M[3][2] = Cx.mul(chi, Jmap);
  M[3][3] = M[1][1];

  return M;
}

// ============================================================================
// 4×4 complex determinant (Laplace expansion)
// ============================================================================

function det3(a: C[][]): C {
  // Cofactor expansion along first row.
  const t1 = Cx.mul(a[0][0], Cx.sub(Cx.mul(a[1][1], a[2][2]), Cx.mul(a[1][2], a[2][1])));
  const t2 = Cx.mul(a[0][1], Cx.sub(Cx.mul(a[1][0], a[2][2]), Cx.mul(a[1][2], a[2][0])));
  const t3 = Cx.mul(a[0][2], Cx.sub(Cx.mul(a[1][0], a[2][1]), Cx.mul(a[1][1], a[2][0])));
  return Cx.add(Cx.sub(t1, t2), t3);
}

export function det4(M: Mat4): C {
  // Laplace expansion along first row.
  const minor = (i: number, j: number): C[][] => {
    const out: C[][] = [];
    for (let r = 0; r < 4; r++) {
      if (r === i) continue;
      const row: C[] = [];
      for (let c = 0; c < 4; c++) {
        if (c === j) continue;
        row.push(M[r][c]);
      }
      out.push(row);
    }
    return out;
  };
  const c0 = Cx.mul(M[0][0], det3(minor(0, 0)));
  const c1 = Cx.mul(M[0][1], det3(minor(0, 1)));
  const c2 = Cx.mul(M[0][2], det3(minor(0, 2)));
  const c3 = Cx.mul(M[0][3], det3(minor(0, 3)));
  return Cx.sub(Cx.add(Cx.sub(c0, c1), c2), c3);
}

// ============================================================================
// Dispersion root finding
// ============================================================================

export const HC_EV_NM = 197.3269804;

export interface DispersionOpts {
  /** Energies to scan, eV. */
  energies_eV: number[];
  /** q-axis samples to scan (1/nm), must be sorted ascending. */
  q_nm: number[];
  /** Cylinder radius a, nm. */
  a_nm: number;
  /** Host medium dielectric. */
  eps_h: C;
  /** Cylinder dielectric function: ω (eV) → ε. */
  eps1_of_w: (w_eV: number) => C;
  /** Angular order m (integer ≥ 0). */
  m: number;
  /** Exclude a fractional margin from both light and material lines. */
  tol_edge?: number;
}

export interface DispersionResult {
  energies_eV: number[];
  /** For each energy, the list of q roots (nm⁻¹). */
  roots_nm: number[][];
  /** Light line in the host, per energy: q_light = √(Re εₕ) · k₀. */
  q_light_nm: number[];
  /** Material line in the cylinder, per energy: q_mat = √(Re ε₁) · k₀. */
  q_mat_nm: number[];
}

/**
 * Scan det(M) along the q axis for each energy and return sign changes of
 * Re(det M) in the guided window (k_light < q < k_mat), refined by linear
 * interpolation on the sampling grid.
 *
 * For a lossless cylinder det M is purely real in the guided window and the
 * sign-change detection gives true dispersion roots. For lossy materials, the
 * roots become resonances (local minima of |det M|); this routine still
 * returns the sign changes of Re(det M), which is usually close enough for
 * visualization but not quantitatively exact.
 */
export function getDispersion(opts: DispersionOpts): DispersionResult {
  const { energies_eV, q_nm, a_nm, eps_h, eps1_of_w, m } = opts;
  const tol_edge = opts.tol_edge ?? 1e-3;

  const roots_nm: number[][] = [];
  const q_light_nm: number[] = [];
  const q_mat_nm: number[] = [];

  for (const w_eV of energies_eV) {
    const eps1 = eps1_of_w(w_eV);
    const k0_nm = w_eV / HC_EV_NM;
    const ka = k0_nm * a_nm;
    const q_light = Math.sqrt(Math.max(0, eps_h[0])) * k0_nm;
    const q_mat = Math.sqrt(Math.max(0, eps1[0])) * k0_nm; // Re(ε₁)

    q_light_nm.push(q_light);
    q_mat_nm.push(q_mat);

    if (q_mat <= q_light) {
      roots_nm.push([]);
      continue;
    }

    const q_low = q_light * (1 + tol_edge);
    const q_high = q_mat * (1 - tol_edge);

    // Sample det M on q_nm restricted to [q_low, q_high]
    const q_used: number[] = [];
    const D_real: number[] = [];
    for (const q of q_nm) {
      if (q < q_low || q > q_high) continue;
      q_used.push(q);
      const qa = q * a_nm;
      const M = giveM(qa, ka, eps1, eps_h, m);
      const d = det4(M);
      D_real.push(d[0]);
    }

    // Sign changes with linear interp
    const roots: number[] = [];
    for (let i = 1; i < q_used.length; i++) {
      const y0 = D_real[i - 1];
      const y1 = D_real[i];
      if (y0 === 0) { roots.push(q_used[i - 1]); continue; }
      if (y0 * y1 < 0) {
        const t = y0 / (y0 - y1);
        roots.push(q_used[i - 1] + t * (q_used[i] - q_used[i - 1]));
      }
    }
    roots_nm.push(roots);
  }

  return { energies_eV, roots_nm, q_light_nm, q_mat_nm };
}
