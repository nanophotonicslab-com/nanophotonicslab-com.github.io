/**
 * A deliberately naive single-molecule localizer.
 *
 * It exists so that "D recovered" is measured from the simulated movie rather
 * than read back from the input — without it the readout would be circular.
 * It is naive on purpose: threshold, local maxima, centroid. Its failure under
 * crowding is exactly what the linking-density envelope check warns about, so
 * strengthening it here would hide the lesson.
 */

export interface Detection {
  /** Refined centre, in pixels. */
  xPx: number;
  yPx: number;
  /** Background-subtracted intensity summed over the fitting window. */
  intensity: number;
}

/** Median of an already-populated scratch buffer (sorted in place). */
function medianOf(buf: Float64Array): number {
  buf.sort();
  const m = buf.length >> 1;
  return buf.length % 2 ? buf[m] : 0.5 * (buf[m - 1] + buf[m]);
}

/**
 * Robust background level and noise scale: the median, and the median absolute
 * deviation scaled to a standard deviation for normal data
 * (sigma ~= 1.4826 * MAD). Both are insensitive to the bright spots themselves.
 *
 * Estimated from a strided subsample (~16k pixels). The estimate only sets a
 * detection threshold, so subsampling costs nothing statistically, and it keeps
 * a 100-frame analysis inside its performance budget — a full sort of every
 * pixel twice per frame dominated the runtime.
 */
let statsBuf: Float64Array | null = null;

function robustStats(values: Float32Array): { med: number; sigma: number } {
  // 4096 samples put the median's own uncertainty near 1.5%, far below what a
  // detection threshold cares about, and keep the two sorts off the hot path.
  const target = 4096;
  const stride = Math.max(1, Math.floor(values.length / target));
  const n = Math.ceil(values.length / stride);
  if (!statsBuf || statsBuf.length !== n) statsBuf = new Float64Array(n);
  const buf = statsBuf;
  for (let i = 0, k = 0; k < n; i += stride, k++) buf[k] = values[i];
  const med = medianOf(buf);
  for (let i = 0, k = 0; k < n; i += stride, k++) buf[k] = Math.abs(values[i] - med);
  return { med, sigma: 1.4826 * medianOf(buf) };
}

export interface LocalizeOptions {
  field: number;
  /** PSF standard deviation in pixels — sets the window and the separation. */
  sigmaPx: number;
  /** Threshold in robust standard deviations above the median. */
  k?: number;
}

/**
 * Detect and localize emitters in one frame.
 *
 * 1. threshold at median + k * MAD
 * 2. keep local maxima in a 3x3 neighbourhood, enforcing a minimum separation
 *    of one FWHM (brightest wins)
 * 3. refine to sub-pixel with a background-subtracted, intensity-weighted
 *    centroid over a +/-2 sigma window
 */
export function localize(img: Float32Array, opt: LocalizeOptions): Detection[] {
  const { field, sigmaPx } = opt;
  const k = opt.k ?? 5;
  const { med, sigma: sigmaNoise } = robustStats(img);
  const threshold = med + k * (sigmaNoise > 0 ? sigmaNoise : 1e-9);

  // candidate local maxima
  const cands: { i: number; j: number; v: number }[] = [];
  for (let j = 1; j < field - 1; j++) {
    for (let i = 1; i < field - 1; i++) {
      const v = img[j * field + i];
      if (v < threshold) continue;
      let isMax = true;
      for (let dj = -1; dj <= 1 && isMax; dj++) {
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dj === 0) continue;
          if (img[(j + dj) * field + (i + di)] > v) { isMax = false; break; }
        }
      }
      if (isMax) cands.push({ i, j, v });
    }
  }

  // enforce a minimum separation of one FWHM, brightest first
  cands.sort((a, b) => b.v - a.v);
  const fwhm = 2.3548200450309493 * sigmaPx;
  const minSep2 = fwhm * fwhm;
  const kept: typeof cands = [];
  for (const c of cands) {
    let ok = true;
    for (const p of kept) {
      const dx = c.i - p.i, dy = c.j - p.j;
      if (dx * dx + dy * dy < minSep2) { ok = false; break; }
    }
    if (ok) kept.push(c);
  }

  // sub-pixel refinement by weighted centroid over +/-2 sigma
  const w = Math.max(1, Math.round(2 * sigmaPx));
  const out: Detection[] = [];
  for (const c of kept) {
    let sw = 0, sx = 0, sy = 0;
    for (let dj = -w; dj <= w; dj++) {
      const j = c.j + dj;
      if (j < 0 || j >= field) continue;
      for (let di = -w; di <= w; di++) {
        const i = c.i + di;
        if (i < 0 || i >= field) continue;
        const v = img[j * field + i] - med;
        if (v <= 0) continue;
        sw += v;
        sx += v * (i + 0.5);
        sy += v * (j + 0.5);
      }
    }
    if (sw > 0) out.push({ xPx: sx / sw, yPx: sy / sw, intensity: sw });
  }
  return out;
}

export interface Track {
  id: number;
  /** frame index -> detection index within that frame, or -1 when missing */
  points: { frame: number; xPx: number; yPx: number }[];
}

/**
 * Nearest-neighbour frame-to-frame linking with a maximum search radius.
 *
 * Like the localizer this is the simplest thing that works, and it is what the
 * linking-density check is warning about: at high density the nearest
 * neighbour is often the wrong particle and tracks swap.
 */
export function linkTracks(
  perFrame: Detection[][], maxDistPx: number,
): Track[] {
  const tracks: Track[] = [];
  let active: { track: Track; x: number; y: number }[] = [];
  let nextId = 0;
  perFrame.forEach((dets, frame) => {
    const used = new Set<number>();
    const stillActive: typeof active = [];
    for (const a of active) {
      let best = -1, bestD2 = maxDistPx * maxDistPx;
      dets.forEach((d, di) => {
        if (used.has(di)) return;
        const dx = d.xPx - a.x, dy = d.yPx - a.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) { bestD2 = d2; best = di; }
      });
      if (best >= 0) {
        used.add(best);
        const d = dets[best];
        a.track.points.push({ frame, xPx: d.xPx, yPx: d.yPx });
        stillActive.push({ track: a.track, x: d.xPx, y: d.yPx });
      }
    }
    dets.forEach((d, di) => {
      if (used.has(di)) return;
      const t: Track = { id: nextId++, points: [{ frame, xPx: d.xPx, yPx: d.yPx }] };
      tracks.push(t);
      stillActive.push({ track: t, x: d.xPx, y: d.yPx });
    });
    active = stillActive;
  });
  return tracks;
}
