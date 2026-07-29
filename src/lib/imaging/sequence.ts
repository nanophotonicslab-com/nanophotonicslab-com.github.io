/**
 * Sequence driver: parameters -> trajectory -> frames -> tracks -> MSD.
 *
 * The trajectory is walked once and kept (it is small: frames x N positions),
 * while frames are rendered on demand from a *per-frame* noise stream keyed on
 * (seed, frame). That keeps scrubbing instant without ever holding the whole
 * movie in memory — at the maximum settings (2000 frames of 512x512) the movie
 * would be about 2 GB — and it keeps every frame independently reproducible.
 */
import { Rng } from './rng';
import {
  seedEmitters, step, stepSigmaNm, minimumImage,
  type Emitter, type MotionKind, type MotionParams,
} from './dynamics';
import { gaussianPSF } from './psf';
import { renderFrame } from './render';
import { detect } from './detector';
import { localize, linkTracks, type Detection } from './localize';
import {
  msdCurve, fitMsd, fitAlpha, maxLag,
  type Trajectory, type MsdCurve, type MsdFit, type AlphaFit,
} from './msd';

export interface SimParams {
  N: number;
  /** Diffusion coefficient, um^2/s. */
  D: number;
  motion: MotionKind;
  /** Directed motion: speed in um/s and heading in degrees. */
  driftV?: number;
  driftAngle?: number;
  /** Confined motion: corral side in nm. */
  corralNm?: number;
  /** Network motion: mesh spacing in nm and per-crossing hop probability. */
  meshNm?: number;
  hopProb?: number;
  /** Photons emitted per particle per frame. */
  photons: number;
  modality: 'fluorescence';
  NA: number;
  /** Emission wavelength, nm. */
  lambda: number;
  /** Pixel size referred to the sample plane, nm. */
  pixel: number;
  /** Square field of view, pixels. */
  field: number;
  /** Background photons per pixel per frame. */
  background: number;
  /** Read noise standard deviation, electrons. */
  readNoise: number;
  frames: number;
  /** Frame interval, ms. */
  dt: number;
  seed: number;
  /** Quantum efficiency (advanced). */
  qe?: number;
}

export interface Truth {
  /** Positions in nm, laid out frame-major: index = frame * N + particle. */
  x: Float64Array;
  y: Float64Array;
  N: number;
  frames: number;
  /** Field size in nm, for periodic-boundary bookkeeping. */
  fieldNm: number;
}

/** Distinct, well-separated noise stream for each frame. */
function frameSeed(seed: number, frame: number): number {
  // mix so that neighbouring frames and neighbouring seeds do not correlate
  let h = (seed ^ 0x9e3779b9) >>> 0;
  h = (Math.imul(h ^ (frame + 1), 0x85ebca6b)) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  return Math.imul(h, 0xc2b2ae35) >>> 0;
}

/**
 * Resolve the spec's motion parameters into kernel units (nm per frame).
 *
 * The meshwork spacing is snapped so that a whole number of compartments spans
 * the field. The mesh is anchored at the field origin and positions wrap at the
 * field edge, so without snapping the compartment pattern would have a seam
 * across the wrap; the adjustment is at most half a cell and keeps the grid
 * exact. `effectiveMeshNm` reports what was actually used.
 */
export function motionParams(p: SimParams): MotionParams & { effectiveMeshNm?: number } {
  const fieldNm = p.field * p.pixel;
  const dtS = p.dt / 1000;
  const base = { kind: p.motion, sigmaNm: stepSigmaNm(p.D, dtS) };
  if (p.motion === 'directed') {
    return {
      ...base,
      // um/s -> nm per frame
      driftNm: (p.driftV ?? 0) * 1000 * dtS,
      driftAngle: ((p.driftAngle ?? 0) * Math.PI) / 180,
    };
  }
  if (p.motion === 'confined') {
    return { ...base, corralNm: Math.min(p.corralNm ?? fieldNm, fieldNm) };
  }
  if (p.motion === 'network') {
    const requested = Math.max(1, Math.min(p.meshNm ?? fieldNm, fieldNm));
    const cells = Math.max(1, Math.round(fieldNm / requested));
    const meshNm = fieldNm / cells;
    return { ...base, meshNm, hopProb: p.hopProb ?? 0, effectiveMeshNm: meshNm };
  }
  return base;
}

/** Walk the ground-truth trajectory. This is the exact answer, by construction. */
export function simulateTruth(p: SimParams): Truth {
  const fieldNm = p.field * p.pixel;
  const r = new Rng(p.seed);
  const emitters: Emitter[] = seedEmitters(p.N, fieldNm, p.photons, r);
  const mp = motionParams(p);

  if (mp.kind === 'confined') {
    // Pull each corral fully inside the field, so a particle seeded near an
    // edge does not spend the movie half outside the frame.
    const half = (mp.corralNm ?? 0) / 2;
    for (const e of emitters) {
      e.cx = Math.min(Math.max(e.x, half), fieldNm - half);
      e.cy = Math.min(Math.max(e.y, half), fieldNm - half);
      e.x = Math.min(Math.max(e.x, e.cx - half), e.cx + half);
      e.y = Math.min(Math.max(e.y, e.cy - half), e.cy + half);
    }
  }

  const x = new Float64Array(p.frames * p.N);
  const y = new Float64Array(p.frames * p.N);
  for (let t = 0; t < p.frames; t++) {
    if (t > 0) step(emitters, mp, fieldNm, r);
    for (let i = 0; i < p.N; i++) {
      x[t * p.N + i] = emitters[i].x;
      y[t * p.N + i] = emitters[i].y;
    }
  }
  return { x, y, N: p.N, frames: p.frames, fieldNm };
}

/** Noiseless photon image of one frame. */
export function renderPhotons(
  p: SimParams, truth: Truth, t: number, out?: Float32Array,
): Float32Array {
  const psf = gaussianPSF(p.lambda, p.NA);
  const emitters: Emitter[] = [];
  for (let i = 0; i < truth.N; i++) {
    emitters.push({ x: truth.x[t * truth.N + i], y: truth.y[t * truth.N + i], photons: p.photons });
  }
  return renderFrame(emitters, { field: p.field, pixelNm: p.pixel, sigmaPsfNm: psf.sigmaNm }, out);
}

/**
 * Fully detected frame, in photoelectrons. Reproducible from (seed, t) alone,
 * which is what makes scrubbing and the exports independent of frame order.
 * `scratch` lets a caller looping over frames reuse its buffers.
 */
export function renderDetected(
  p: SimParams, truth: Truth, t: number,
  scratch?: { photons?: Float32Array; out?: Float32Array },
): Float32Array {
  const photons = renderPhotons(p, truth, t, scratch?.photons);
  const r = new Rng(frameSeed(p.seed, t));
  return detect(
    photons, { background: p.background, readNoise: p.readNoise, qe: p.qe }, r, scratch?.out,
  );
}

export interface Analysis {
  /** Detections per frame. */
  perFrame: Detection[][];
  /** Localized, linked trajectories. */
  tracks: ReturnType<typeof linkTracks>;
  /** MSD of the localized tracks (the honest number). */
  msd: MsdCurve;
  fit: MsdFit;
  /** Anomalous exponent of the localized MSD — how the models differ. */
  alpha: AlphaFit;
  /** MSD of the ground truth, as a dashed reference. */
  msdTruth: MsdCurve;
  fitTruth: MsdFit;
  alphaTruth: AlphaFit;
  /** Mean detections per frame — a crowding diagnostic. */
  meanDetections: number;
}

/** Ground-truth trajectories, unwrapped through the periodic boundary. */
export function truthTrajectories(truth: Truth): Trajectory[] {
  const out: Trajectory[] = [];
  for (let i = 0; i < truth.N; i++) {
    const x: number[] = [], y: number[] = [], frame: number[] = [];
    let ux = truth.x[i], uy = truth.y[i];
    x.push(ux); y.push(uy); frame.push(0);
    for (let t = 1; t < truth.frames; t++) {
      const dx = minimumImage(truth.x[t * truth.N + i] - truth.x[(t - 1) * truth.N + i], truth.fieldNm);
      const dy = minimumImage(truth.y[t * truth.N + i] - truth.y[(t - 1) * truth.N + i], truth.fieldNm);
      ux += dx; uy += dy;
      x.push(ux); y.push(uy); frame.push(t);
    }
    out.push({ x, y, frame });
  }
  return out;
}

/**
 * Run the localizer over every frame, link the detections and fit the MSD.
 * `onProgress` is called with a 0..1 fraction so the UI can show a bar beyond
 * the 500 ms mark.
 */
export function analyzeSequence(
  p: SimParams, truth: Truth, onProgress?: (f: number) => void,
): Analysis {
  const psf = gaussianPSF(p.lambda, p.NA);
  const sigmaPx = psf.sigmaNm / p.pixel;
  const perFrame: Detection[][] = [];
  // reuse the two frame buffers across the whole pass rather than allocating
  // ~50 MB of short-lived typed arrays
  const scratch = {
    photons: new Float32Array(p.field * p.field),
    out: new Float32Array(p.field * p.field),
  };
  for (let t = 0; t < p.frames; t++) {
    const img = renderDetected(p, truth, t, scratch);
    perFrame.push(localize(img, { field: p.field, sigmaPx }));
    if (onProgress && (t & 7) === 0) onProgress(t / p.frames);
  }
  // Link within a generous radius: three per-frame diffusive steps plus the
  // drift, since directed motion displaces every particle the same way and
  // would otherwise fall outside a purely diffusive search radius.
  const mp = motionParams(p);
  const stepPx = mp.sigmaNm / p.pixel;
  const driftPx = (mp.driftNm ?? 0) / p.pixel;
  const maxDist = Math.max(3 * stepPx + driftPx, 2.3548200450309493 * sigmaPx);
  const tracks = linkTracks(perFrame, maxDist);

  const trajs: Trajectory[] = tracks
    .filter(tr => tr.points.length >= 2)
    .map(tr => ({
      x: tr.points.map(q => q.xPx * p.pixel),
      y: tr.points.map(q => q.yPx * p.pixel),
      frame: tr.points.map(q => q.frame),
    }));

  const dtS = p.dt / 1000;
  const msd = msdCurve(trajs, dtS, p.frames);
  const msdTruth = msdCurve(truthTrajectories(truth), dtS, p.frames);
  const nFit = Math.min(5, maxLag(p.frames));
  const total = perFrame.reduce((a, d) => a + d.length, 0);
  onProgress?.(1);
  return {
    perFrame,
    tracks,
    msd,
    fit: fitMsd(msd, nFit),
    alpha: fitAlpha(msd),
    msdTruth,
    fitTruth: fitMsd(msdTruth, nFit),
    alphaTruth: fitAlpha(msdTruth),
    meanDetections: total / Math.max(1, p.frames),
  };
}
