/**
 * Camera model: background, quantum efficiency, shot noise, read noise.
 *
 * The output is in photoelectrons and carries no display stretch — the viewer
 * applies a percentile stretch for display only, and the exports contain these
 * raw values.
 */
import { PoissonTable, type Rng } from './rng';

export interface DetectorOptions {
  /** Uniform background photons per pixel per frame. */
  background: number;
  /** Gaussian read-noise standard deviation, in electrons. */
  readNoise: number;
  /** Quantum efficiency (advanced; 1.0 keeps photons and electrons equal). */
  qe?: number;
}

/**
 * Apply the detection chain to a noiseless photon image:
 *   1. add the uniform background
 *   2. convert to electrons through the quantum efficiency, with Poisson shot noise
 *   3. add Gaussian read noise
 */
export function detect(
  photons: Float32Array, opt: DetectorOptions, r: Rng, into?: Float32Array,
): Float32Array {
  const qe = opt.qe ?? 1.0;
  const out = into && into.length === photons.length ? into : new Float32Array(photons.length);
  // Away from the emitters the signal is exactly zero, so those pixels all
  // share the background mean and can be drawn from one tabulated CDF.
  const bgMean = opt.background * qe;
  const bgTable = bgMean > 0 && bgMean < 50 ? new PoissonTable(bgMean) : null;
  for (let i = 0; i < photons.length; i++) {
    const s = photons[i];
    let e: number;
    if (s === 0 && bgTable) {
      e = bgTable.sample(r);
    } else {
      e = r.poisson((s + opt.background) * qe);
    }
    if (opt.readNoise > 0) e += r.normal(0, opt.readNoise);
    out[i] = e;
  }
  return out;
}

/**
 * Percentile stretch limits for display. Returns the values at the given
 * lower/upper percentiles, computed on a subsample for speed on large fields.
 */
let stretchBuf: Float64Array | null = null;

export function stretchLimits(img: Float32Array, loPct = 0.1, hiPct = 99.9): [number, number] {
  // A strided subsample into a reused typed array, sorted numerically. Building
  // a boxed JS array of every pixel and sorting it with a comparator dominated
  // the per-frame cost, and percentiles do not need every pixel.
  const target = 8192;
  const stride = Math.max(1, Math.floor(img.length / target));
  const n = Math.ceil(img.length / stride);
  if (!stretchBuf || stretchBuf.length !== n) stretchBuf = new Float64Array(n);
  const buf = stretchBuf;
  for (let i = 0, k = 0; k < n; i += stride, k++) buf[k] = img[i];
  buf.sort();
  const at = (p: number) => buf[Math.min(n - 1, Math.max(0, Math.round((p / 100) * (n - 1))))];
  const lo = at(loPct);
  let hi = at(hiPct);
  if (!(hi > lo)) hi = lo + 1;
  return [lo, hi];
}
