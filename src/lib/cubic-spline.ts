/**
 * Natural-cubic-spline interpolation with not-a-knot end conditions.
 *
 * This is the interpolation scheme of SciPy's `CubicSpline` and MATLAB's
 * `spline` in their default configuration, which is what the Pyodide BEM
 * solver uses for its own tabulated permittivities. Matching it here is what
 * lets the BEM and the TypeScript kernels read the same published table and
 * get the same material, so a browser-side comparison of the two measures the
 * method rather than the interpolation.
 *
 * The spline is C² and reproduces a cubic exactly. Piecewise-linear
 * interpolation, which this replaces, is only C⁰: it puts a corner at every
 * tabulated point, and where a table is sparse next to a fast-moving feature
 * — the interband edge of gold at 471 nm, sampled every 20 nm — that corner
 * is large enough to be visible in a spectrum computed from it.
 */

export interface CubicSplineFn {
  /** Evaluate the spline. Outside [x[0], x[n-1]] the end value is held. */
  (x: number): number;
}

/**
 * Solve a tridiagonal system by the Thomas algorithm.
 *
 * `sub`, `diag` and `sup` are the three bands; `sub[0]` and `sup[n-1]` are
 * unused. The arrays are consumed in place, so callers must pass copies they
 * do not need afterwards.
 */
function solveTridiagonal(
  sub: Float64Array,
  diag: Float64Array,
  sup: Float64Array,
  rhs: Float64Array,
): Float64Array {
  const n = diag.length;
  for (let i = 1; i < n; i++) {
    const w = sub[i] / diag[i - 1];
    diag[i] -= w * sup[i - 1];
    rhs[i] -= w * rhs[i - 1];
  }
  const out = new Float64Array(n);
  out[n - 1] = rhs[n - 1] / diag[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    out[i] = (rhs[i] - sup[i] * out[i + 1]) / diag[i];
  }
  return out;
}

/**
 * Build an interpolating cubic spline through (x, y).
 *
 * `x` must be strictly increasing. Fewer than four points cannot carry two
 * distinct not-a-knot conditions, so the natural lower-order interpolant is
 * used instead: a parabola through three points, a straight line through two,
 * a constant through one. That is also what SciPy does.
 */
export function cubicSpline(x: readonly number[], y: readonly number[]): CubicSplineFn {
  const n = x.length;
  if (n !== y.length) throw new Error('cubicSpline: x and y must have the same length.');
  if (n === 0) throw new Error('cubicSpline: needs at least one point.');

  if (n === 1) {
    const v = y[0];
    return () => v;
  }
  if (n === 2) {
    const slope = (y[1] - y[0]) / (x[1] - x[0]);
    return (t: number) => {
      if (t <= x[0]) return y[0];
      if (t >= x[1]) return y[1];
      return y[0] + slope * (t - x[0]);
    };
  }

  const h = new Float64Array(n - 1);
  const slope = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = x[i + 1] - x[i];
    slope[i] = (y[i + 1] - y[i]) / h[i];
  }

  // Solve for the first derivative at each node. Writing the system in the
  // derivatives rather than the second derivatives keeps it genuinely
  // tridiagonal under not-a-knot: the end conditions land inside the band
  // instead of reaching one column past it, which the moment form does not,
  // and which degenerates to a zero pivot on a uniformly spaced table.
  let m: Float64Array;
  if (n === 3) {
    // The two not-a-knot conditions coincide here, leaving a single parabola
    // through the three points.
    const c2 = (slope[1] - slope[0]) / (h[0] + h[1]);
    m = Float64Array.from([
      slope[0] - c2 * h[0],
      slope[0] + c2 * h[0],
      slope[1] + c2 * h[1],
    ]);
  } else {
    const sub = new Float64Array(n);
    const diag = new Float64Array(n);
    const sup = new Float64Array(n);
    const rhs = new Float64Array(n);

    for (let i = 1; i < n - 1; i++) {
      sub[i] = h[i];
      diag[i] = 2 * (h[i - 1] + h[i]);
      sup[i] = h[i - 1];
      rhs[i] = 3 * (h[i] * slope[i - 1] + h[i - 1] * slope[i]);
    }

    // Not-a-knot: the third derivative is continuous across x[1] and x[n-2].
    const d0 = x[2] - x[0];
    diag[0] = h[1];
    sup[0] = d0;
    rhs[0] = ((h[0] + 2 * d0) * h[1] * slope[0] + h[0] * h[0] * slope[1]) / d0;

    const dN = x[n - 1] - x[n - 3];
    diag[n - 1] = h[n - 3];
    sub[n - 1] = dN;
    rhs[n - 1] =
      (h[n - 2] * h[n - 2] * slope[n - 3] +
        (2 * dN + h[n - 2]) * h[n - 3] * slope[n - 2]) /
      dN;

    m = solveTridiagonal(sub, diag, sup, rhs);
  }

  return (t: number): number => {
    if (t <= x[0]) return y[0];
    if (t >= x[n - 1]) return y[n - 1];
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (x[mid] <= t) lo = mid;
      else hi = mid;
    }
    // Cubic Hermite on [x[lo], x[hi]] from the node values and derivatives.
    const hL = h[lo];
    const dt = t - x[lo];
    const c2 = (3 * slope[lo] - 2 * m[lo] - m[hi]) / hL;
    const c3 = (m[lo] + m[hi] - 2 * slope[lo]) / (hL * hL);
    return y[lo] + dt * (m[lo] + dt * (c2 + dt * c3));
  };
}
