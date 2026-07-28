/**
 * Binds the imaging kernel to the IMG1 spec.
 *
 * Split in two, because the page has two very different latency budgets: a
 * parameter change must update the preview and every derived readout in well
 * under 100 ms (`computeQuick`), while the movie-wide analysis — render,
 * localize and link every frame — is allowed up to two seconds and reports
 * progress (`computeFull`).
 */
import {
  analyzeSequence, gaussianPSF, renderFrame, simulateTruth, thompsonSigma,
  type Analysis, type SimParams, type Truth,
} from '../../index';
import { derive, type Derived } from './spec';

export type Values = Record<string, number | string>;

/** Translate spec parameter values into kernel simulation parameters. */
export function toSimParams(p: Values): SimParams {
  return {
    N: Number(p.N),
    D: Number(p.D),
    motion: 'brownian',
    photons: Number(p.photons),
    modality: 'fluorescence',
    NA: Number(p.NA),
    lambda: Number(p.lambda),
    pixel: Number(p.pixel),
    field: Number(p.field),
    background: Number(p.background),
    readNoise: Number(p.readNoise),
    frames: Number(p.frames),
    dt: Number(p.dt),
    seed: Number(p.seed),
    qe: p.qe === undefined ? 1 : Number(p.qe),
  };
}

export interface XY {
  x: Float64Array;
  y: Float64Array;
}

export interface QuickResult {
  derived: Derived;
  /** Values for the readout row and the envelope context. */
  observables: Record<string, number>;
  /** PSF cross-section through the centre of a single rendered emitter. */
  psfCut: XY;
  /** Thompson precision against photon budget, for the precision plot. */
  precision: XY;
}

/**
 * Everything that follows from the parameters without simulating the movie.
 * `dFit` is left NaN here — it can only come from the analysis.
 */
export function computeQuick(p: Values): QuickResult {
  const d = derive(p);
  const pixel = Number(p.pixel);
  const lambda = Number(p.lambda);
  const NA = Number(p.NA);
  const background = Number(p.background);
  const readNoise = Number(p.readNoise);

  // ── PSF cross-section: render one emitter on a small field and cut the row
  const cut = Math.max(24, Math.ceil(8 * d.fwhmPx) | 1);
  const centrePx = cut / 2;
  const psf = gaussianPSF(lambda, NA);
  const img = renderFrame(
    [{ x: centrePx * pixel, y: centrePx * pixel, photons: Number(p.photons) }],
    { field: cut, pixelNm: pixel, sigmaPsfNm: psf.sigmaNm },
  );
  const row = Math.floor(centrePx);
  const cx = new Float64Array(cut);
  const cy = new Float64Array(cut);
  for (let i = 0; i < cut; i++) {
    cx[i] = i + 0.5 - centrePx;
    cy[i] = img[row * cut + i];
  }

  // ── localization precision against photon budget (log sweep)
  const NP = 80;
  const px = new Float64Array(NP);
  const py = new Float64Array(NP);
  const bStd = Math.sqrt(background + readNoise * readNoise);
  for (let i = 0; i < NP; i++) {
    const n = 10 ** (1 + (4 * i) / (NP - 1)); // 10 .. 1e5 photons
    px[i] = n;
    py[i] = thompsonSigma(psf.sigmaNm, pixel, n, bStd);
  }

  return {
    derived: d,
    observables: {
      stepPx: d.stepPx,
      fwhmPx: d.fwhmPx,
      snr: d.snr,
      sigmaLoc: d.sigmaLocNm,
      dFit: NaN,
      // extra context the envelope predicates read
      nyquistNm: d.nyquistNm,
      density: d.density,
      sigmaLocNm: d.sigmaLocNm,
    },
    psfCut: { x: cx, y: cy },
    precision: { x: px, y: py },
  };
}

export interface FullResult {
  truth: Truth;
  analysis: Analysis;
  /** MSD of the localized tracks, and of the ground truth as a reference. */
  msd: { tau: Float64Array; localized: Float64Array; truth: Float64Array };
  /** Straight-line fit of the localized MSD, for the in-plot annotation. */
  fitLine: { tau: Float64Array; msd: Float64Array };
  dFit: number;
  sigmaLocFitNm: number;
}

/** Simulate the whole movie, localize it, link it and fit the MSD. */
export function computeFull(p: Values, onProgress?: (f: number) => void): FullResult {
  const sim = toSimParams(p);
  const truth = simulateTruth(sim);
  const analysis = analyzeSequence(sim, truth, onProgress);
  const { msd, msdTruth, fit } = analysis;

  // the fitted line over the lags actually used
  const nFit = Math.min(5, msd.tau.length);
  const tauLine = new Float64Array(nFit);
  const msdLine = new Float64Array(nFit);
  for (let i = 0; i < nFit; i++) {
    tauLine[i] = msd.tau[i];
    msdLine[i] = 4 * fit.D * 1e6 * msd.tau[i] + fit.intercept;
  }

  return {
    truth,
    analysis,
    msd: { tau: msd.tau, localized: msd.msd, truth: msdTruth.msd },
    fitLine: { tau: tauLine, msd: msdLine },
    dFit: fit.D,
    sigmaLocFitNm: fit.sigmaLocNm,
  };
}
