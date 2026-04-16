/**
 * Beam Propagation Method — Crank-Nicolson solver for the paraxial wave equation.
 *
 * Ported from the Jena Computational Photonics seminar (beamprop_CN.m).
 * All spatial coordinates in micrometres.
 */

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface BpmParams {
  z: number;         // propagation distance [um]
  dz: number;        // step size in z
  nd: number;        // characteristic refractive index
  lambda: number;    // wavelength [um]
  Nx: number;        // number of x-intervals
  w: number;         // gaussian waist [um]
  xa: number;        // calculation window width [um]
  xb: number;        // waveguide width [um]
  nCladding: number; // cladding refractive index
  nCore: number;     // core refractive index
  delta: number;     // sub-sampling: store every delta-th step
}

export interface BpmResult {
  field: Float64Array;  // |v|² flattened row-major (Nz rows × Nx cols)
  Nx: number;
  Nz: number;
  xMin: number;
  xMax: number;
  zMax: number;
  maxVal: number;
  xb: number;           // waveguide half-width for overlay
}

/* ------------------------------------------------------------------ */
/*  Helper: Gaussian beam                                              */
/* ------------------------------------------------------------------ */

function makeGauss(xa: number, Nx: number, w: number) {
  const x = new Float64Array(Nx);
  const re = new Float64Array(Nx);
  const im = new Float64Array(Nx);   // zero-initialised
  for (let i = 0; i < Nx; i++) {
    x[i] = -xa / 2 + (xa * i) / (Nx - 1);
    re[i] = Math.exp(-(x[i] * x[i]) / (w * w));
  }
  return { re, im, x };
}

/* ------------------------------------------------------------------ */
/*  Helper: step-index waveguide                                       */
/* ------------------------------------------------------------------ */

function makeWaveguide(xa: number, xb: number, Nx: number, nClad: number, nCore: number) {
  const x = new Float64Array(Nx);
  const n = new Float64Array(Nx);
  const halfXb = xb / 2;
  for (let i = 0; i < Nx; i++) {
    x[i] = -xa / 2 + (xa * i) / (Nx - 1);
    n[i] = (x[i] >= -halfXb && x[i] <= halfXb) ? nCore : nClad;
  }
  return { n, x };
}

/* ------------------------------------------------------------------ */
/*  Complex tridiagonal solver (Thomas algorithm)                      */
/*  Solves  A·x = d  where A is tridiagonal.                          */
/*  lower / diag / upper / d are complex (paired re/im arrays).       */
/*  Modifies diag and d in place; result in d.                         */
/* ------------------------------------------------------------------ */

function thomasSolve(
  N: number,
  lRe: Float64Array, lIm: Float64Array,   // sub-diagonal  [0..N-2]
  dRe: Float64Array, dIm: Float64Array,   // main diagonal [0..N-1]
  uRe: Float64Array, uIm: Float64Array,   // super-diagonal[0..N-2]
  rRe: Float64Array, rIm: Float64Array,   // rhs → solution [0..N-1]
) {
  // Forward sweep
  for (let i = 1; i < N; i++) {
    // m = lower[i-1] / diag[i-1]
    const dr = dRe[i - 1], di = dIm[i - 1];
    const den = dr * dr + di * di;
    const lr = lRe[i - 1], li = lIm[i - 1];
    const mRe = (lr * dr + li * di) / den;
    const mIm = (li * dr - lr * di) / den;

    // diag[i] -= m * upper[i-1]
    const ur = uRe[i - 1], ui = uIm[i - 1];
    dRe[i] -= mRe * ur - mIm * ui;
    dIm[i] -= mRe * ui + mIm * ur;

    // rhs[i] -= m * rhs[i-1]
    const rrPrev = rRe[i - 1], riPrev = rIm[i - 1];
    rRe[i] -= mRe * rrPrev - mIm * riPrev;
    rIm[i] -= mRe * riPrev + mIm * rrPrev;
  }

  // Back substitution
  {
    const dr = dRe[N - 1], di = dIm[N - 1];
    const den = dr * dr + di * di;
    const rr = rRe[N - 1], ri = rIm[N - 1];
    rRe[N - 1] = (rr * dr + ri * di) / den;
    rIm[N - 1] = (ri * dr - rr * di) / den;
  }
  for (let i = N - 2; i >= 0; i--) {
    // rhs[i] = (rhs[i] - upper[i] * rhs[i+1]) / diag[i]
    const ur = uRe[i], ui = uIm[i];
    const xr = rRe[i + 1], xi = rIm[i + 1];
    const nr = rRe[i] - (ur * xr - ui * xi);
    const ni = rIm[i] - (ur * xi + ui * xr);
    const dr = dRe[i], di = dIm[i];
    const den = dr * dr + di * di;
    rRe[i] = (nr * dr + ni * di) / den;
    rIm[i] = (ni * dr - nr * di) / den;
  }
}

/* ------------------------------------------------------------------ */
/*  Crank-Nicolson BPM                                                 */
/* ------------------------------------------------------------------ */

export function beampropCN(p: BpmParams): BpmResult {
  const { z, dz, nd, lambda, Nx, w, xa, xb, nCladding, nCore, delta } = p;

  const { re: vRe, im: vIm, x } = makeGauss(xa, Nx, w);
  const { n } = makeWaveguide(xa, xb, Nx, nCladding, nCore);

  const dx = xa / (Nx - 1);
  const k_m = (2 * Math.PI / lambda) * nd;

  // Pre-compute W[j] = i * (k[j]^2 - k_m^2) * dz / (2*k_m)   (pure imaginary × real)
  // and a = i * dz / (2*k_m*dx^2)                               (pure imaginary scalar)
  const aIm = dz / (2 * k_m * dx * dx);  // a = i * aIm  ⇒  aRe = 0

  // W[j] is pure imaginary:  W_im[j] = (k[j]^2 - k_m^2) * dz / (2*k_m)
  const Wim = new Float64Array(Nx);
  const k_m2 = k_m * k_m;
  for (let j = 0; j < Nx; j++) {
    const kj = (2 * Math.PI / lambda) * n[j];
    Wim[j] = (kj * kj - k_m2) * dz / (2 * k_m);
  }

  // Build A and B diagonals (constant across z-steps):
  //   B_main[j] = 1 + W[j]/2 - a  =  (1)  + i*(Wim[j]/2 - aIm)
  //   B_off     =     a/2          =  0    + i*(aIm/2)
  //   A_main[j] = 1 - W[j]/2 + a  =  (1)  + i*(-Wim[j]/2 + aIm)
  //   A_off     =    -a/2          =  0    + i*(-aIm/2)

  const AmRe = new Float64Array(Nx);
  const AmIm = new Float64Array(Nx);
  const BmRe = new Float64Array(Nx);
  const BmIm = new Float64Array(Nx);
  for (let j = 0; j < Nx; j++) {
    AmRe[j] = 1;
    AmIm[j] = -Wim[j] / 2 + aIm;
    BmRe[j] = 1;
    BmIm[j] = Wim[j] / 2 - aIm;
  }
  const AoffRe = 0;
  const AoffIm = -aIm / 2;
  const BoffRe = 0;
  const BoffIm = aIm / 2;

  // Output storage
  const totalSteps = Math.round(z / dz);
  const Nz = Math.floor(totalSteps / delta) + 1;
  const field = new Float64Array(Nz * Nx);

  // Store initial |v|²
  for (let j = 0; j < Nx; j++) {
    field[j] = vRe[j] * vRe[j] + vIm[j] * vIm[j];
  }

  // Temporary arrays for Thomas solver (re-used each step)
  const tDiagRe = new Float64Array(Nx);
  const tDiagIm = new Float64Array(Nx);
  const tLowRe = new Float64Array(Nx - 1);
  const tLowIm = new Float64Array(Nx - 1);
  const tUpRe = new Float64Array(Nx - 1);
  const tUpIm = new Float64Array(Nx - 1);
  const rhsRe = new Float64Array(Nx);
  const rhsIm = new Float64Array(Nx);

  let outRow = 1;

  for (let step = 1; step <= totalSteps; step++) {
    // RHS = B · v   (tridiagonal matvec)
    rhsRe[0] = BmRe[0] * vRe[0] - BmIm[0] * vIm[0]
             + BoffRe * vRe[1] - BoffIm * vIm[1];
    rhsIm[0] = BmRe[0] * vIm[0] + BmIm[0] * vRe[0]
             + BoffRe * vIm[1] + BoffIm * vRe[1];
    for (let j = 1; j < Nx - 1; j++) {
      rhsRe[j] = BoffRe * vRe[j - 1] - BoffIm * vIm[j - 1]
               + BmRe[j] * vRe[j] - BmIm[j] * vIm[j]
               + BoffRe * vRe[j + 1] - BoffIm * vIm[j + 1];
      rhsIm[j] = BoffRe * vIm[j - 1] + BoffIm * vRe[j - 1]
               + BmRe[j] * vIm[j] + BmIm[j] * vRe[j]
               + BoffRe * vIm[j + 1] + BoffIm * vRe[j + 1];
    }
    const last = Nx - 1;
    rhsRe[last] = BoffRe * vRe[last - 1] - BoffIm * vIm[last - 1]
                + BmRe[last] * vRe[last] - BmIm[last] * vIm[last];
    rhsIm[last] = BoffRe * vIm[last - 1] + BoffIm * vRe[last - 1]
                + BmRe[last] * vIm[last] + BmIm[last] * vRe[last];

    // Copy A diagonals (Thomas modifies them in place)
    for (let j = 0; j < Nx; j++) { tDiagRe[j] = AmRe[j]; tDiagIm[j] = AmIm[j]; }
    for (let j = 0; j < Nx - 1; j++) {
      tLowRe[j] = AoffRe; tLowIm[j] = AoffIm;
      tUpRe[j] = AoffRe;  tUpIm[j] = AoffIm;
    }

    // Solve A · v_next = rhs
    thomasSolve(Nx, tLowRe, tLowIm, tDiagRe, tDiagIm, tUpRe, tUpIm, rhsRe, rhsIm);

    // Update v
    for (let j = 0; j < Nx; j++) { vRe[j] = rhsRe[j]; vIm[j] = rhsIm[j]; }

    // Store every delta-th step
    if (step % delta === 0 && outRow < Nz) {
      const off = outRow * Nx;
      for (let j = 0; j < Nx; j++) {
        field[off + j] = vRe[j] * vRe[j] + vIm[j] * vIm[j];
      }
      outRow++;
    }
  }

  // Find max
  let maxVal = 0;
  for (let i = 0; i < field.length; i++) {
    if (field[i] > maxVal) maxVal = field[i];
  }

  return {
    field,
    Nx,
    Nz: outRow,   // actual number of rows stored
    xMin: x[0],
    xMax: x[Nx - 1],
    zMax: z,
    maxVal,
    xb,
  };
}
