/**
 * Least-squares fit of a Mie sphere model to measured spectra.
 *
 * Model: y(λ) ≈ A·σ_channel(λ; r) — the amplitude A absorbs concentration and
 * instrument units, so the shape carries the radius information. For a given
 * radius the optimal A is analytic (least squares through the origin),
 * reducing the fit to a 1D search over radius: coarse grid, then
 * golden-section refinement.
 */
import { mieAt } from './mie';

export type MieChannel = 'cext' | 'csca' | 'cabs';

export interface MieFitResult {
  radiusNm: number;
  scale: number;
  rms: number;
  r2: number;
  /** Model curve A·σ(λᵢ; r_fit) evaluated on the data grid. */
  model: Float64Array;
}

/** Optimal amplitude A minimizing Σ(A·m − y)², and the resulting SSE. */
function scaleAndSse(model: Float64Array, y: Float64Array): { A: number; sse: number } {
  let mm = 0, my = 0;
  for (let i = 0; i < y.length; i++) { mm += model[i] * model[i]; my += model[i] * y[i]; }
  const A = mm > 0 ? my / mm : 0;
  let sse = 0;
  for (let i = 0; i < y.length; i++) { const d = A * model[i] - y[i]; sse += d * d; }
  return { A, sse };
}

export function fitMieRadius(
  getNK: (lambdaNm: number) => [number, number],
  nHost: number,
  xNm: Float64Array, yData: Float64Array,
  channel: MieChannel,
  rMinNm: number, rMaxNm: number,
  coarseSteps = 32,
): MieFitResult {
  const n = Math.min(xNm.length, yData.length);
  // Cache n,k per data wavelength — shared across all radius evaluations.
  const nks = Array.from({ length: n }, (_, i) => getNK(xNm[i]));

  const modelAt = (r: number): Float64Array => {
    const m = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const [np, kp] = nks[i];
      m[i] = mieAt(np, kp, nHost, r, xNm[i])[channel];
    }
    return m;
  };
  const sseAt = (r: number) => scaleAndSse(modelAt(r), yData).sse;

  // Coarse grid
  const cells: { r: number; sse: number }[] = [];
  for (let s = 0; s < coarseSteps; s++) {
    const r = rMinNm + ((rMaxNm - rMinNm) * s) / (coarseSteps - 1);
    cells.push({ r, sse: sseAt(r) });
  }
  cells.sort((p, q) => p.sse - q.sse);
  const step = (rMaxNm - rMinNm) / (coarseSteps - 1);

  // Refine from up to 3 well-separated coarse minima: sharp-resonance
  // (high-index) particles can put a spurious local minimum inside a single
  // golden window (physics audit 2026-07-18).
  const seeds: number[] = [];
  for (const cell of cells) {
    if (seeds.every((r) => Math.abs(r - cell.r) > 1.5 * step)) seeds.push(cell.r);
    if (seeds.length === 3) break;
  }
  const golden = (seed: number): { r: number; sse: number } => {
    let lo = Math.max(rMinNm, seed - step), hi = Math.min(rMaxNm, seed + step);
    const phi = (Math.sqrt(5) - 1) / 2;
    let c = hi - phi * (hi - lo), d = lo + phi * (hi - lo);
    let fc = sseAt(c), fd = sseAt(d);
    for (let it = 0; it < 40 && hi - lo > 1e-3; it++) {
      if (fc < fd) { hi = d; d = c; fd = fc; c = hi - phi * (hi - lo); fc = sseAt(c); }
      else { lo = c; c = d; fc = fd; d = lo + phi * (hi - lo); fd = sseAt(d); }
    }
    const r = (lo + hi) / 2;
    return { r, sse: sseAt(r) };
  };
  let best = golden(seeds[0]);
  for (let i = 1; i < seeds.length; i++) {
    const cand = golden(seeds[i]);
    if (cand.sse < best.sse) best = cand;
  }
  const rFit = best.r;

  const model = modelAt(rFit);
  const { A, sse } = scaleAndSse(model, yData);
  for (let i = 0; i < n; i++) model[i] *= A;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += yData[i];
  mean /= n;
  let ssTot = 0;
  for (let i = 0; i < n; i++) { const d2 = yData[i] - mean; ssTot += d2 * d2; }
  return {
    radiusNm: rFit,
    scale: A,
    rms: Math.sqrt(sse / n),
    r2: ssTot > 0 ? 1 - sse / ssTot : 0,
    model,
  };
}

/**
 * Parse a two-column spectrum CSV/TSV: wavelength, value. Skips non-numeric
 * rows (headers, comments). Wavelengths below 10 are treated as µm → nm.
 */
export function parseSpectrumCsv(text: string): { x: Float64Array; y: Float64Array } | { error: string } {
  const xs: number[] = [], ys: number[] = [];
  for (const line of text.split(/\r?\n/)) {
    const parts = line.split(/[,;\t]+|\s{2,}/).map((p) => parseFloat(p.trim()));
    if (parts.length >= 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1])) {
      xs.push(parts[0] < 10 ? parts[0] * 1000 : parts[0]);
      ys.push(parts[1]);
    }
  }
  if (xs.length < 8) return { error: 'need at least 8 numeric (wavelength, value) rows' };
  // sort by wavelength
  const idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  return {
    x: Float64Array.from(idx, (i) => xs[i]),
    y: Float64Array.from(idx, (i) => ys[i]),
  };
}
