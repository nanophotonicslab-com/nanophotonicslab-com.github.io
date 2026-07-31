/**
 * IMG1 — Diffusion and Tracking.
 *
 * Brownian particles imaged through a simulated fluorescence microscope, with
 * exact ground truth. Ground truth is an *input*, not a fit: the user places
 * every particle, so every position is known exactly, and the localizer's
 * answer can be compared against it.
 */
import { gaussianPSF, stepSigmaNm, thompsonSigma } from '../../index';
import { fmt, type Solver } from '../../../solver-spec';

/**
 * Quantities the readouts, the envelope and the plots all need. Derived from
 * the parameters alone — no simulation required, so it is cheap enough to run
 * on every keystroke.
 */
export interface Derived {
  /** Per-axis per-frame diffusive step, in pixels. */
  stepPx: number;
  /** Drift per frame, in pixels (zero unless the motion is directed). */
  driftPx: number;
  /**
   * Per-frame displacement scale including drift, in pixels. This is what
   * smears a particle within one exposure, so it is what the motion-blur check
   * compares against the PSF width.
   */
  blurPx: number;
  /** Confinement length of the active model in pixels, or NaN if unconfined. */
  confinePx: number;
  /** PSF standard deviation, nm and pixels. */
  sigmaPsfNm: number;
  sigmaPsfPx: number;
  /** PSF full width at half maximum, in pixels. */
  fwhmPx: number;
  /** Peak pixel signal, in photons. */
  peak: number;
  /** Peak-signal-to-noise ratio (defined in the readout tooltip). */
  snr: number;
  /** Thompson localization precision, in nm. */
  sigmaLocNm: number;
  /** Nyquist pixel limit lambda/(4 NA), in nm. */
  nyquistNm: number;
  /** Particles per PSF area. */
  density: number;
}

export function derive(p: Record<string, number | string>): Derived {
  const D = Number(p.D), dt = Number(p.dt), pixel = Number(p.pixel);
  const lambda = Number(p.lambda), NA = Number(p.NA), field = Number(p.field);
  const photons = Number(p.photons), background = Number(p.background);
  const readNoise = Number(p.readNoise), N = Number(p.N);

  const psf = gaussianPSF(lambda, NA);
  const sigmaPsfPx = psf.sigmaNm / pixel;
  const fwhmPx = psf.fwhmNm / pixel;
  const stepPx = stepSigmaNm(D, dt / 1000) / pixel;
  const motion = String(p.motion ?? 'brownian');
  // um/s over one frame -> nm -> px
  const driftPx = motion === 'directed'
    ? (Number(p.driftV ?? 0) * 1000 * (dt / 1000)) / pixel
    : 0;
  const blurPx = Math.hypot(stepPx, driftPx);
  const confineNm = motion === 'confined'
    ? Number(p.corralNm ?? NaN)
    : motion === 'network' ? Number(p.meshNm ?? NaN) : NaN;
  const confinePx = confineNm / pixel;
  // peak of a photon-conserving 2D Gaussian: N / (2 pi sigma^2), per pixel
  const peak = photons / (2 * Math.PI * sigmaPsfPx * sigmaPsfPx);
  const snr = peak / Math.sqrt(peak + background + readNoise * readNoise);
  // background standard deviation per pixel, in photons, as Thompson defines it
  const bStd = Math.sqrt(background + readNoise * readNoise);
  const sigmaLocNm = thompsonSigma(psf.sigmaNm, pixel, photons, bStd);
  const density = (N * Math.PI * (fwhmPx / 2) ** 2) / (field * field);

  return {
    stepPx, driftPx, blurPx, confinePx,
    sigmaPsfNm: psf.sigmaNm, sigmaPsfPx, fwhmPx, peak, snr, sigmaLocNm,
    nyquistNm: lambda / (4 * NA), density,
  };
}

export const diffusionTracking: Solver = {
  meta: {
    lab: 'imaging',
    id: 'diffusion-tracking',
    code: 'IMG1',
    title: 'Diffusion and Tracking',
    blurb: 'Brownian particles imaged through a fluorescence microscope — with ground truth.',
    status: 'Experimental',
    version: '0.2.0',
    updated: '2026-07-29',
  },

  docs: {
    model: 'Per-frame Gaussian steps under a choice of motion model — free Brownian, directed '
      + '(drift + diffusion), confined in a reflecting corral, or hopping between the '
      + 'compartments of an underlying meshwork. Emitters are rendered with a Gaussian PSF '
      + '(sigma = 0.21 lambda / NA), each contributing a photon-conserving spot, then Poisson '
      + 'shot noise and Gaussian read noise. A deliberately naive localizer (threshold, local '
      + 'maxima, weighted centroid) recovers the tracks, and their mean-square displacement '
      + 'gives back both the diffusion coefficient and the anomalous exponent α.',
    assumptions: [
      'Overdamped 2D motion, isotropic D',
      'No interparticle interaction or hydrodynamics',
      'Thin sample; all emitters in the focal plane',
      'Periodic boundaries: a particle leaving one edge re-enters at the opposite one '
        + '(a confined particle never reaches them)',
      'Corral and meshwork walls are hard and reflect specularly; the meshwork is a regular '
        + 'square grid with a single hop probability, not a disordered network',
    ],
    validity: 'Dilute fields; per-frame step below the PSF width; photon budget above ~100 per frame.',
    limitations: [
      'No axial motion or defocus (a later module)',
      'No photobleaching or blinking (a later module)',
      'The motion models are the ones expressible through DeepTrack2’s own sequential-property '
        + 'mechanism. DeepTrack2 ships no diffusion models itself — it provides '
        + 'Feature.to_sequential(position=rule) and demonstrates free Brownian motion — so each '
        + 'model here is a per-frame update rule and is exported as one.',
      'Continuous-time trapping and fractional Brownian motion are not included: both need the '
        + 'whole trajectory history rather than one step at a time',
      'Gaussian PSF approximation, not pupil-based — no aberrations. Its FWHM is 3.9% narrower '
        + 'than an ideal Airy pattern at every NA, and 7–12% narrower than DeepTrack2’s numerically '
        + 'sampled pupil PSF; see the validation note below.',
      'No motion-blur correction in the MSD fit (Berglund’s −4/3 D dt term is not applied)',
    ],
    references: [
      'B. Midtvedt, S. Helgadottir, A. Argun, J. Pineda, D. Midtvedt & G. Volpe, '
        + 'Quantitative digital microscopy with deep learning, Appl. Phys. Rev. 8, 011310 (2021), '
        + 'doi:10.1063/5.0034891 — the paper DeepTrack2 asks to be cited by.',
      'DeepTrack2 (DeepTrackAI), MIT licence, https://github.com/DeepTrackAI/DeepTrack2 — the '
        + 'reference implementation this module was validated against and exports scripts for. '
        + 'No DeepTrack2 code is vendored here; the forward model is reimplemented in TypeScript.',
      'R. E. Thompson, D. R. Larson & W. W. Webb, Precise nanometer localization analysis for '
        + 'individual fluorescent probes, Biophys. J. 82, 2775 (2002), doi:10.1016/S0006-3495(02)75618-X',
      'B. Zhang, J. Zerubia & J.-C. Olivo-Marin, Gaussian approximations of fluorescence microscope '
        + 'point-spread function models, Appl. Opt. 46, 1819 (2007), doi:10.1364/AO.46.001819',
    ],
    notes: 'Ground truth is exact — it is an input, not a fit. The dashed green markers are where '
      + 'the particles really are; the localizer never sees them. Boundaries are periodic, which is '
      + 'visible in the trajectory trails.',
  },

  structure: { kind: 'widefield', sample: 'point-emitters', detector: 'camera' },

  groups: [
    { id: 'particle', label: 'Particle & dynamics' },
    { id: 'optics', label: 'Optics' },
    { id: 'detector', label: 'Detector' },
    { id: 'sequence', label: 'Sequence' },
  ],

  params: [
    // ── particle & dynamics ──
    {
      key: 'N', symbol: 'N', group: 'particle', label: 'N particles',
      default: 20, min: 1, max: 500, step: 1, integer: true,
      help: 'Number of independently diffusing emitters.',
    },
    {
      key: 'D', symbol: 'D', unit: 'µm²/s', group: 'particle', label: 'D',
      default: 0.5, min: 0.001, max: 50, scale: 'log',
      help: 'Diffusion coefficient. It sets the per-frame step, sqrt(2 D Δt) per axis.',
    },
    {
      key: 'motion', group: 'particle', label: 'motion',
      default: 'brownian',
      choices: [
        { value: 'brownian', label: 'Brownian (free)' },
        { value: 'directed', label: 'Directed (drift + diffusion)' },
        { value: 'confined', label: 'Confined (corral)' },
        { value: 'network', label: 'Meshwork (hopping between compartments)' },
      ],
      help: 'How each particle moves. Free Brownian motion is the reference; directed adds a '
        + 'constant velocity; confined traps the particle in a corral; meshwork lets it diffuse '
        + 'freely inside a compartment and only rarely cross into the next one. Watch the '
        + 'anomalous exponent α change as you switch.',
    },
    {
      key: 'driftV', symbol: 'v', unit: 'µm/s', group: 'particle', label: 'speed',
      default: 2, min: 0, max: 200, step: 0.1,
      showIf: v => v.motion === 'directed',
      help: 'Constant velocity added on top of diffusion — flow, or active transport along a filament.',
    },
    {
      key: 'driftAngle', symbol: 'θ', unit: '°', group: 'particle', label: 'direction',
      default: 30, min: 0, max: 360, step: 1,
      showIf: v => v.motion === 'directed',
      help: 'Heading of the drift, measured anticlockwise from the +x axis.',
    },
    {
      key: 'corralNm', symbol: 'L', unit: 'nm', group: 'particle', label: 'corral size',
      default: 400, min: 20, max: 20000, scale: 'log',
      showIf: v => v.motion === 'confined',
      help: 'Side of the square corral each particle is trapped in. The MSD stops growing and '
        + 'flattens at L²/3, which is how confinement is recognised in real data.',
    },
    {
      key: 'meshNm', symbol: 'L', unit: 'nm', group: 'particle', label: 'mesh size',
      default: 800, min: 50, max: 20000, scale: 'log',
      showIf: v => v.motion === 'network',
      help: 'Spacing of the underlying network — the compartment size set by a membrane skeleton '
        + 'or a crowded matrix. Rounded slightly so a whole number of compartments spans the field.',
    },
    {
      key: 'hopProb', symbol: 'p', group: 'particle', label: 'hop probability',
      default: 0.05, min: 0, max: 1, scale: 'log',
      showIf: v => v.motion === 'network',
      help: 'Chance that a particle meeting a mesh boundary crosses it instead of bouncing back. '
        + 'p = 1 is free diffusion, p = 0 seals every compartment; in between the particle looks '
        + 'free at short lag times and slower at long ones.',
    },
    {
      key: 'photons', symbol: 'I', unit: 'photons/frame', group: 'particle', label: 'intensity',
      default: 1200, min: 10, max: 100000, scale: 'log',
      help: 'Total photons emitted per particle per frame, before detection.',
    },

    // ── optics ──
    {
      key: 'modality', group: 'optics', label: 'modality',
      choices: ['fluorescence'], default: 'fluorescence',
      help: 'Incoherent fluorescence imaging. Brightfield, darkfield, iSCAT and holography arrive in a later module.',
    },
    {
      key: 'NA', symbol: 'NA', group: 'optics', label: 'NA',
      default: 1.40, min: 0.1, max: 1.7, step: 0.01,
      help: 'Objective numerical aperture. It sets the PSF width, and with it the resolution.',
    },
    {
      key: 'lambda', symbol: 'λ', unit: 'nm', group: 'optics', label: 'λ_em',
      default: 520, min: 350, max: 900, step: 1, presets: [488, 520, 561, 640],
      help: 'Emission wavelength of the fluorophore.',
    },

    // ── detector ──
    {
      key: 'pixel', symbol: 'a', unit: 'nm', group: 'detector', label: 'pixel size',
      default: 65, min: 10, max: 500, step: 1,
      help: 'Camera pixel size referred to the sample plane — the physical size one pixel covers on the sample.',
    },
    {
      key: 'field', unit: 'px', group: 'detector', label: 'field',
      default: 256, choices: [128, 256, 512],
      help: 'Square field of view, in pixels.',
    },
    {
      key: 'background', unit: 'photons/px', group: 'detector', label: 'background',
      default: 8, min: 0, max: 1000, step: 1,
      help: 'Uniform background photons per pixel per frame — out-of-focus fluorescence and stray light.',
    },
    {
      key: 'readNoise', unit: 'e⁻', group: 'detector', label: 'read noise',
      default: 1.6, min: 0, max: 50, step: 0.1,
      help: 'Gaussian read-noise standard deviation added by the camera electronics.',
    },
    {
      key: 'qe', group: 'detector', label: 'quantum efficiency',
      default: 1.0, min: 0.05, max: 1, step: 0.01, advanced: true,
      help: 'Fraction of incident photons converted to photoelectrons.',
    },

    // ── sequence ──
    {
      key: 'frames', group: 'sequence', label: 'frames',
      default: 100, min: 1, max: 2000, step: 1, integer: true,
      help: 'Number of frames in the movie.',
    },
    {
      key: 'dt', symbol: 'Δt', unit: 'ms', group: 'sequence', label: 'Δt',
      default: 20, min: 0.1, max: 1000, scale: 'log',
      help: 'Frame interval — the time between consecutive exposures.',
    },
    {
      key: 'seed', group: 'sequence', label: 'seed',
      default: 42, min: 0, max: 999999, step: 1, integer: true,
      help: 'Random seed. The same seed reproduces the run exactly, on any machine.',
    },
  ],

  observables: [
    {
      key: 'stepPx', label: 'step per frame', unit: 'px',
      help: 'Typical per-axis displacement between frames. Compare it with the PSF width: once a '
        + 'particle moves a good fraction of its own image, tracking becomes unreliable.',
    },
    {
      key: 'fwhmPx', label: 'PSF FWHM', unit: 'px',
      help: 'Width of a single point emitter’s image, at half its peak height.',
    },
    {
      key: 'snr', label: 'SNR',
      help: 'Peak-pixel signal-to-noise ratio: the brightest pixel’s signal divided by the total '
        + 'noise in it, sqrt(peak + background + readNoise²). Other definitions of SNR exist — this '
        + 'is the one shown here.',
    },
    {
      key: 'sigmaLoc', label: 'σ localization', unit: 'nm',
      help: 'How precisely one particle’s centre can be found, from the Thompson–Larson–Webb '
        + 'estimate for the current photon count, pixel size and background.',
    },
    {
      key: 'dFit', label: 'D recovered', unit: 'µm²/s',
      help: 'Diffusion coefficient recovered from the simulated movie: the slope of the localized '
        + 'tracks’ mean-square displacement over its first five lags, divided by four. Compare it '
        + 'with the D you set — confinement and a meshwork both push it below.',
    },
    {
      key: 'alpha', label: 'α exponent',
      help: 'Anomalous exponent from a log–log fit of MSD against lag time, MSD ∝ τ^α. It is what '
        + 'separates the motion models: α = 1 free Brownian, α < 1 subdiffusive (a corral or a '
        + 'meshwork holding the particle back), α → 2 directed transport.',
      digits: 2,
    },
  ],

  plots: [
    {
      id: 'msd', title: 'Mean-square displacement',
      xLabel: 'lag τ', xUnit: 's', yLabel: 'MSD (nm²)',
    },
    {
      id: 'psfCut', title: 'PSF cross-section',
      xLabel: 'x', xUnit: 'px', yLabel: 'photons / px',
    },
    {
      id: 'precision', title: 'Localization precision',
      xLabel: 'photons / frame', yLabel: 'σ (nm)',
    },
  ],

  envelope: [
    {
      id: 'nyquist', label: 'Nyquist sampling',
      evaluate: ({ p, o }) => {
        const pixel = Number(p.pixel);
        const limit = o.nyquistNm;
        return pixel <= limit
          ? { level: 'ok', message: `pixel ${fmt(pixel, 3)} nm ≤ λ/4NA = ${fmt(limit, 3)} nm — the PSF is well sampled` }
          : {
            level: 'warn',
            message: `pixel ${fmt(pixel, 3)} nm exceeds λ/4NA = ${fmt(limit, 3)} nm — the PSF is `
              + 'undersampled and localization degrades; use a smaller pixel or a lower NA',
          };
      },
    },
    {
      id: 'blur', label: 'Motion blur',
      evaluate: ({ o }) => {
        const half = 0.5 * o.fwhmPx;
        // includes the drift, since directed motion smears a particle within
        // the exposure just as diffusion does
        const moved = o.blurPx;
        const what = o.driftPx > 0 ? 'step + drift' : 'step';
        return moved < half
          ? { level: 'ok', message: `${what} ${fmt(moved, 2)} px < ½ FWHM = ${fmt(half, 2)} px` }
          : {
            level: 'warn',
            message: `${what} ${fmt(moved, 2)} px approaches the PSF width (½ FWHM = ${fmt(half, 2)} px) — `
              + 'particles smear and linking becomes unreliable; lower D, the speed, or Δt',
          };
      },
    },
    {
      id: 'confinement', label: 'Confinement scale',
      evaluate: ({ p, o }) => {
        const motion = String(p.motion);
        if (motion !== 'confined' && motion !== 'network') {
          return {
            level: 'ok',
            message: motion === 'directed'
              ? 'directed motion — no confinement length to resolve'
              : 'free diffusion — no confinement length to resolve',
          };
        }
        const what = motion === 'confined' ? 'corral' : 'mesh';
        const L = o.confinePx * Number(p.pixel); // back to nm
        const sigmaLoc = o.sigmaLocNm;
        // To see confinement at all the compartment must be bigger than the
        // localization blur, and the particle must have room to diffuse inside
        // it before hitting a wall.
        if (!(L > 0)) return { level: 'ok', message: '' };
        if (L < 2 * sigmaLoc) {
          return {
            level: 'fail',
            message: `${what} ${fmt(L, 3)} nm is below twice the localization precision `
              + `(${fmt(2 * sigmaLoc, 3)} nm) — the confinement cannot be measured; enlarge it `
              + 'or raise the photon budget',
          };
        }
        const stepNm = o.stepPx * Number(p.pixel);
        if (L < 2 * stepNm) {
          return {
            level: 'warn',
            message: `${what} ${fmt(L, 3)} nm is only ${fmt(L / stepNm, 2)}× the per-frame step — `
              + 'the particle crosses it every frame and looks like a static blob; enlarge it or lower Δt',
          };
        }
        return {
          level: 'ok',
          message: `${what} ${fmt(L, 3)} nm spans ${fmt(L / stepNm, 2)} steps and `
            + `${fmt(L / sigmaLoc, 2)}× the localization precision`,
        };
      },
    },
    {
      id: 'density', label: 'Linking density',
      evaluate: ({ o }) => {
        const d = o.density;
        const msg = `${fmt(d, 2)} particles per PSF area`;
        if (d < 0.05) return { level: 'ok', message: msg };
        if (d < 0.15) {
          return {
            level: 'warn',
            message: `${msg} — trajectories may swap; lower N or Δt`,
          };
        }
        return {
          level: 'fail',
          message: `${msg} — the field is crowded, detections merge and trajectories will swap; lower N`,
        };
      },
    },
    {
      id: 'photons', label: 'Photon budget',
      evaluate: ({ p, o }) => {
        const photons = Number(p.photons);
        const pixel = Number(p.pixel);
        if (photons >= 100 && o.sigmaLocNm < pixel) {
          return { level: 'ok', message: `σ_loc = ${fmt(o.sigmaLocNm, 3)} nm, below one pixel` };
        }
        return {
          level: 'warn',
          message: `σ_loc = ${fmt(o.sigmaLocNm, 3)} nm exceeds one pixel (${fmt(pixel, 3)} nm) — `
            + 'localization is not meaningful at this photon count; raise the intensity or lower the background',
        };
      },
    },
  ],

  // Each preset resets every parameter to its default and then applies these
  // overrides, so a preset always means the same thing regardless of what was
  // clicked before it.
  scenarios: [
    {
      label: 'Textbook',
      set: {},
      note: 'All defaults. Note that the default Δt = 20 ms already puts the per-frame step '
        + '(2.18 px) above half the PSF width (1.41 px), so the motion-blur check warns; '
        + 'drop Δt to 8 ms or below for an all-green run.',
    },
    {
      label: 'Single QD, low SNR',
      set: { N: 1, photons: 150, background: 20 },
      note: 'One dim emitter: localization precision, not crowding, is the limit.',
    },
    {
      label: 'Crowded field',
      set: { N: 200, field: 128 },
      breaksEnvelope: true,
      note: 'Deliberately trips the linking-density check. N alone is not enough at a '
        + '256 px field — 200 particles fill only 0.019 of it — so the field is halved too.',
    },
    {
      label: 'Fast diffusion',
      set: { D: 10 },
      breaksEnvelope: true,
      note: 'Deliberately trips the motion-blur check: the step grows to ~9.7 px per frame.',
    },
    {
      label: 'Nyquist-limited pixels',
      set: { pixel: 160, dt: 5 },
      breaksEnvelope: true,
      note: 'Deliberately trips the Nyquist check, and only that one: Δt is lowered so the '
        + 'coarse pixels do not also trip motion blur.',
    },
    {
      label: 'Directed transport',
      set: { motion: 'directed', driftV: 2, driftAngle: 30, D: 0.1, dt: 20, N: 15 },
      note: 'Drift plus diffusion: the MSD picks up a v²τ² term and α climbs towards 2.',
    },
    {
      label: 'Confined in corrals',
      set: { motion: 'confined', corralNm: 400, D: 0.5, dt: 20, frames: 200, N: 25 },
      note: 'Each particle trapped in a 400 nm corral: the MSD flattens at L²/3 and α falls well '
        + 'below 1.',
    },
    {
      label: 'Meshwork hopping',
      set: { motion: 'network', meshNm: 800, hopProb: 0.02, D: 0.5, dt: 20, frames: 200, N: 25 },
      note: 'Interaction with an underlying network: free inside a compartment, rarely crossing — '
        + 'free at short lags, subdiffusive at long ones.',
    },
  ],
};
