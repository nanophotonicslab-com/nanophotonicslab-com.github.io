/**
 * Mean-square displacement and the diffusion-coefficient fit.
 *
 * For free 2D diffusion observed with finite localization precision,
 *
 *   MSD(tau) = 4 D tau + 4 sigma_loc^2
 *
 * so the slope gives D and the intercept gives the localization precision.
 * Motion-blur correction (MSD = 4 D tau + 4 sigma^2 - (4/3) D dt, Berglund) is
 * NOT applied in v1; this is stated in the module's limitations.
 */

export interface Trajectory {
  /** Positions in nm, one entry per observed frame. */
  x: number[];
  y: number[];
  /** Frame index of each entry (may skip frames for localized tracks). */
  frame: number[];
}

export interface MsdCurve {
  /** Lag time in seconds. */
  tau: Float64Array;
  /** Ensemble- and time-averaged MSD in nm^2. */
  msd: Float64Array;
  /** Number of displacement samples behind each point. */
  counts: Float64Array;
}

/** Largest lag to evaluate: min(frames/4, 25), per the brief. */
export function maxLag(frames: number): number {
  return Math.max(1, Math.min(Math.floor(frames / 4), 25));
}

/**
 * Ensemble- and time-averaged 2D MSD over lags 1..maxLag.
 * Trajectories may be of different lengths; every available pair contributes.
 */
export function msdCurve(trajs: Trajectory[], dtS: number, frames: number): MsdCurve {
  const L = maxLag(frames);
  const sum = new Float64Array(L);
  const counts = new Float64Array(L);
  for (const t of trajs) {
    const n = t.frame.length;
    // index positions by frame so gaps in a localized track are handled
    const byFrame = new Map<number, number>();
    for (let i = 0; i < n; i++) byFrame.set(t.frame[i], i);
    for (let i = 0; i < n; i++) {
      for (let lag = 1; lag <= L; lag++) {
        const j = byFrame.get(t.frame[i] + lag);
        if (j === undefined) continue;
        const dx = t.x[j] - t.x[i];
        const dy = t.y[j] - t.y[i];
        sum[lag - 1] += dx * dx + dy * dy;
        counts[lag - 1]++;
      }
    }
  }
  const tau = new Float64Array(L);
  const msd = new Float64Array(L);
  for (let l = 0; l < L; l++) {
    tau[l] = (l + 1) * dtS;
    msd[l] = counts[l] > 0 ? sum[l] / counts[l] : NaN;
  }
  return { tau, msd, counts };
}

export interface MsdFit {
  /** Diffusion coefficient in um^2/s. */
  D: number;
  /** Intercept in nm^2 (= 4 sigma_loc^2). */
  intercept: number;
  /** Localization precision implied by the intercept, in nm (NaN if negative). */
  sigmaLocNm: number;
  /** Number of lags used. */
  nPoints: number;
}

/**
 * Straight-line least squares over the first `nLags` points of the MSD curve.
 * MSD is in nm^2 and tau in s, so the slope is nm^2/s; D = slope/4 converted
 * to um^2/s by dividing by 1e6.
 */
export function fitMsd(curve: MsdCurve, nLags = 5): MsdFit {
  const n = Math.min(nLags, curve.tau.length);
  let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
  for (let i = 0; i < n; i++) {
    const x = curve.tau[i], y = curve.msd[i];
    if (!Number.isFinite(y)) continue;
    sx += x; sy += y; sxx += x * x; sxy += x * y; m++;
  }
  if (m < 2) return { D: NaN, intercept: NaN, sigmaLocNm: NaN, nPoints: m };
  const denom = m * sxx - sx * sx;
  const slope = (m * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / m;
  return {
    D: slope / 4 / 1e6,
    intercept,
    sigmaLocNm: intercept > 0 ? Math.sqrt(intercept / 4) : NaN,
    nPoints: m,
  };
}

/**
 * Thompson, Larson & Webb (Biophys. J. 82, 2775 (2002)) localization
 * precision, the standard single-molecule estimate:
 *
 *   sigma^2 = (s^2 + a^2/12)/N + 8 pi s^4 b^2 / (a^2 N^2)
 *
 * @param sNm PSF standard deviation in nm
 * @param aNm pixel size in nm
 * @param N photons detected per frame
 * @param bPhotons background standard deviation per pixel, in photons
 */
export function thompsonSigma(sNm: number, aNm: number, N: number, bPhotons: number): number {
  if (!(N > 0)) return NaN;
  const s2 = sNm * sNm;
  const a2 = aNm * aNm;
  const term1 = (s2 + a2 / 12) / N;
  const term2 = (8 * Math.PI * s2 * s2 * bPhotons * bPhotons) / (a2 * N * N);
  return Math.sqrt(term1 + term2);
}
