/**
 * Imaging kernel — pure physics for the Imaging lab.
 *
 * Nothing in this folder touches the DOM, reads component state or calls
 * `Math.random()`. Every Imaging module reuses it, so keep it that way: if a
 * function needs the page, it belongs in the module, not here.
 */
export { Rng, rng } from './rng';
export { gaussianPSF, FWHM_PER_SIGMA, type PsfModel } from './psf';
export {
  seedEmitters, step, stepSigmaNm, minimumImage, reflect,
  type Emitter, type MotionKind, type MotionParams,
} from './dynamics';
export {
  renderFrame, radialProfile, measureFwhmPx, measureSigmaPx, measureFwhmCut,
  type RenderOptions,
} from './render';
export { detect, stretchLimits, type DetectorOptions } from './detector';
export { localize, linkTracks, type Detection, type Track, type LocalizeOptions } from './localize';
export {
  msdCurve, fitMsd, fitAlpha, maxLag, thompsonSigma,
  type Trajectory, type MsdCurve, type MsdFit, type AlphaFit,
} from './msd';
export {
  simulateTruth, renderPhotons, renderDetected, analyzeSequence, truthTrajectories,
  motionParams,
  type SimParams, type Truth, type Analysis,
} from './sequence';
