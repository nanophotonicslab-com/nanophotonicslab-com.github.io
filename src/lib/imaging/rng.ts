/**
 * Seeded random number generation for the Imaging kernel.
 *
 * Determinism is a hard requirement: the same seed and parameters must give
 * byte-identical frames across reloads and machines, because permalinks,
 * reproducibility and the acceptance tests all depend on it. `Math.random()`
 * is therefore forbidden anywhere in this kernel — everything draws from a
 * xoshiro128** instance created from the `seed` parameter.
 */

/** xoshiro128** — small, fast, well-distributed 32-bit generator. */
export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;
  // Box-Muller produces normals in pairs; the spare is kept for the next call.
  private spare: number | null = null;

  constructor(seed: number) {
    // SplitMix32 expands the single seed into the four words of state, so that
    // neighbouring seeds (42, 43) give unrelated streams.
    let z = (seed | 0) >>> 0;
    const next = (): number => {
      z = (z + 0x9e3779b9) >>> 0;
      let t = z;
      t = Math.imul(t ^ (t >>> 16), 0x21f0aaad) >>> 0;
      t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
      return (t ^ (t >>> 15)) >>> 0;
    };
    this.s0 = next(); this.s1 = next(); this.s2 = next(); this.s3 = next();
    // an all-zero state is a fixed point of xoshiro — avoid it
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  /** Raw 32-bit unsigned draw. */
  nextUint32(): number {
    const rotl = (x: number, k: number) => ((x << k) | (x >>> (32 - k))) >>> 0;
    const result = Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7) >>> 0, 9) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 = (this.s2 ^ this.s0) >>> 0;
    this.s3 = (this.s3 ^ this.s1) >>> 0;
    this.s1 = (this.s1 ^ this.s2) >>> 0;
    this.s0 = (this.s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform in [0, 1). */
  uniform(): number {
    // 2^-32; the top bits of xoshiro** are the well-mixed ones
    return this.nextUint32() * 2.3283064365386963e-10;
  }

  /** Uniform in [lo, hi). */
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.uniform();
  }

  /** Standard normal via the polar (Marsaglia) form of Box-Muller. */
  normal(mean = 0, sd = 1): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return mean + sd * v;
    }
    let u: number, v: number, s: number;
    do {
      u = this.uniform() * 2 - 1;
      v = this.uniform() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * f;
    return mean + sd * (u * f);
  }

  /**
   * Poisson sample with mean `lambda`.
   *
   * Knuth's multiplicative method is exact but O(lambda), which stalls at the
   * photon counts this module reaches (~1e3 per pixel and above). For
   * lambda >= 50 the normal approximation N(lambda, sqrt(lambda)) is used
   * instead, rounded and clamped at zero: at that mean the relative error of
   * the approximation is well below the shot noise it represents.
   */
  poisson(lambda: number): number {
    if (!(lambda > 0)) return 0;
    if (lambda < 50) {
      const L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= this.uniform();
      } while (p > L);
      return k - 1;
    }
    const v = Math.round(this.normal(lambda, Math.sqrt(lambda)));
    return v < 0 ? 0 : v;
  }
}

/** Convenience factory, so callers read as `rng(seed)`. */
export function rng(seed: number): Rng {
  return new Rng(seed);
}

/**
 * Exact Poisson sampler for a *fixed* mean, by inverse transform of a
 * precomputed CDF: one uniform and a binary search per draw, instead of the
 * ~lambda+1 uniforms Knuth's method needs.
 *
 * Most pixels of a frame see exactly the same mean (the uniform background),
 * so amortising the CDF over them is the difference between a 100-frame
 * analysis taking seconds and taking well under a second. The distribution is
 * identical to `Rng.poisson` — only the number of random draws differs.
 */
export class PoissonTable {
  private cdf: Float64Array;
  readonly lambda: number;

  constructor(lambda: number) {
    this.lambda = lambda;
    // grow until the tail left over is negligible, with a hard cap well above
    // lambda + 10 sqrt(lambda)
    const cap = Math.max(16, Math.ceil(lambda + 12 * Math.sqrt(lambda) + 12));
    const cdf = new Float64Array(cap);
    let term = Math.exp(-lambda); // P(0)
    let acc = term;
    cdf[0] = acc;
    for (let k = 1; k < cap; k++) {
      term *= lambda / k;
      acc += term;
      cdf[k] = acc;
    }
    this.cdf = cdf;
  }

  /** Draw one sample using a single uniform from `r`. */
  sample(r: Rng): number {
    const u = r.uniform();
    const cdf = this.cdf;
    let lo = 0, hi = cdf.length - 1;
    if (u >= cdf[hi]) return hi; // beyond the tabulated tail (probability ~1e-12)
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (u < cdf[mid]) hi = mid; else lo = mid + 1;
    }
    return lo;
  }
}
