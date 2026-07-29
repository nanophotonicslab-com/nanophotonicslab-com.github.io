/**
 * Emits a runnable DeepTrack2 script reproducing the current experiment.
 *
 * This is the strategic half of the module: the browser removes the barrier to
 * entry, and the export makes sure it is not also the ceiling. The script must
 * run unmodified against a current `deeptrack`, so the API used here was
 * checked against the installed package rather than taken on trust:
 *
 *  - `Fluorescence(NA, wavelength, resolution, magnification,
 *     refractive_index_medium, output_region, upscale)` — verified signature.
 *  - `PointParticle(position=..., intensity=...)`, `^` to replicate,
 *    `dt.Sequence(..., sequence_length=...)`.
 *  - The sequential-property API differs across releases, and the two forms do
 *    not overlap: deeptrack 2.0.1 (PyPI) has only `deeptrack.sequences.Sequential`,
 *    while the development branch adds `feature.to_sequential(...)`, stops
 *    exporting `Sequential` and leaves it defective. The brief assumed
 *    `to_sequential`, which raises AttributeError on the released package, so the
 *    script resolves whichever the installed build actually has.
 *  - The update rule's `previous_values` EXCLUDES the immediately preceding value
 *    (it is `previous()[:step - 1]`) and is empty on the first call, which the
 *    confined model has to allow for when it reads a particle's starting point.
 *  - `dt.Poisson(snr=...)` exists but is NOT a photon-count model: it rescales
 *    the image so the peak above background matches a target SNR, then samples.
 *    This module works in physical photons, so the script applies shot noise and
 *    read noise explicitly with numpy, and normalises DeepTrack's arbitrary
 *    intensity scale to a photon budget first (DeepTrack's `intensity` is not a
 *    photon count — a unit-intensity emitter integrates to ~0.095, not 1).
 *
 * The optics — the part being cross-checked — stays DeepTrack's.
 */
import type { Values } from './compute';

export interface ExportOptions {
  /** Permalink reproducing this exact state in the browser. */
  permalink: string;
  version: string;
}

/** Number formatting that always yields a valid Python literal. */
function py(v: number): string {
  if (!Number.isFinite(v)) return 'float("nan")';
  return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(12)));
}

/** Immersion index implied by the NA, so the emitted script stays physical. */
export function mediumIndexFor(NA: number): number {
  if (NA > 1.35) return 1.518; // oil
  if (NA > 1.0) return 1.33; // water
  return 1.0; // dry
}

/**
 * The motion model, written as a DeepTrack2 sequential update rule.
 *
 * DeepTrack2 has no diffusion models of its own; it has
 * `Feature.to_sequential(position=rule)` (and the deprecated `dt.Sequential`),
 * where `rule` receives `previous_value` and returns the next position. Every
 * model in this module is one such rule, so the exported script stays inside
 * DeepTrack2's own machinery with nothing but numpy beside it.
 */
function motionBlock(p: Values, fieldPx: number, pixelNm: number, dtS: number): string {
  const motion = String(p.motion ?? 'brownian');

  if (motion === 'directed') {
    const v = Number(p.driftV ?? 0);
    const angle = Number(p.driftAngle ?? 0);
    return `# Directed motion: a constant velocity on top of the diffusive step.
DRIFT_V = ${py(v)}      # um/s
DRIFT_DEG = ${py(angle)}
drift_px = DRIFT_V * 1000 * DT / PIXEL   # per frame, in pixels
drift = drift_px * np.array([
    np.cos(np.deg2rad(DRIFT_DEG)), np.sin(np.deg2rad(DRIFT_DEG)),
])

def update_position(previous_value):
    step = rng.normal(0, sigma_step_px, 2) + drift
    return (previous_value + step) % FIELD`;
  }

  if (motion === 'confined') {
    const L = Number(p.corralNm ?? 0) / pixelNm;
    return `# Confined diffusion: each particle is trapped in a square corral with
# reflecting walls, centred where it started. The MSD flattens at L^2/3.
#
# The corral centre is the particle's own starting point, which the rule reads
# from the property's history — DeepTrack2 hands the update rule everything it
# has stored, so no per-particle bookkeeping is needed on our side.
#
# Note DeepTrack2's convention: previous_values is previous()[:step - 1], i.e.
# it EXCLUDES the immediately preceding value and is empty on the first update.
# On that first call previous_value is itself the starting point.
CORRAL = ${py(Number(L.toFixed(6)))}   # px

def reflect(v, lo, hi):
    """Fold a coordinate back into [lo, hi], repeating for large steps."""
    span = hi - lo
    if span <= 0:
        return lo
    t = (v - lo) % (2 * span)
    return lo + (t if t <= span else 2 * span - t)

def update_position(previous_value, previous_values):
    start = previous_values[0] if len(previous_values) else previous_value
    centre = np.asarray(start, dtype=float)
    proposed = previous_value + rng.normal(0, sigma_step_px, 2)
    half = CORRAL / 2
    return np.array([
        reflect(proposed[i], centre[i] - half, centre[i] + half) for i in range(2)
    ])`;
  }

  if (motion === 'network') {
    // mirror the kernel's snapping so a whole number of compartments spans the field
    const requested = Math.max(1, Math.min(Number(p.meshNm ?? 0), fieldPx * pixelNm));
    const cells = Math.max(1, Math.round((fieldPx * pixelNm) / requested));
    const meshPx = fieldPx / cells;
    const hop = Number(p.hopProb ?? 0);
    return `# Meshwork ("fence") model: the particle diffuses freely inside a
# compartment of an underlying network and crosses into the next one only with
# probability HOP; otherwise it bounces off the boundary. Free at short lag
# times, subdiffusive at long ones.
MESH = ${py(Number(meshPx.toFixed(6)))}   # px, snapped so a whole number spans the field
HOP = ${py(hop)}

def reflect(v, lo, hi):
    span = hi - lo
    if span <= 0:
        return lo
    t = (v - lo) % (2 * span)
    return lo + (t if t <= span else 2 * span - t)

def update_position(previous_value):
    proposed = previous_value + rng.normal(0, sigma_step_px, 2)
    out = np.empty(2)
    for i in range(2):
        cell = np.floor(previous_value[i] / MESH)
        if np.floor(proposed[i] / MESH) == cell or rng.random() < HOP:
            out[i] = proposed[i]
        else:
            out[i] = reflect(proposed[i], cell * MESH, (cell + 1) * MESH)
    return out % FIELD`;
  }

  return `# Free Brownian motion — the model DeepTrack2's own tutorials demonstrate
# (DTAT331 section 4, DTGS106 section 6), wrapped periodically into the field.

def update_position(previous_value):
    return (previous_value + rng.normal(0, sigma_step_px, 2)) % FIELD`;
}

export function toPythonScript(p: Values, opt: ExportOptions): string {
  const N = Number(p.N);
  const D = Number(p.D);
  const dtMs = Number(p.dt);
  const pixel = Number(p.pixel);
  const NA = Number(p.NA);
  const lambda = Number(p.lambda);
  const field = Number(p.field);
  const photons = Number(p.photons);
  const background = Number(p.background);
  const readNoise = Number(p.readNoise);
  const frames = Number(p.frames);
  const seed = Number(p.seed);
  const qe = p.qe === undefined ? 1 : Number(p.qe);
  const nMedium = mediumIndexFor(NA);

  return `# Generated by NanophotonicsLab — Imaging / Diffusion and Tracking (IMG1) v${opt.version}
# Permalink: ${opt.permalink}
#
# Reproduces the browser run in DeepTrack2. The optics are DeepTrack's; the
# detector chain is applied explicitly in photon units, because dt.Poisson
# targets a signal-to-noise ratio rather than a photon count (see below).
#
#   pip install deeptrack
import numpy as np
import deeptrack as dt

# ── parameters ───────────────────────────────────────────────────────────────
N        = ${py(N)}            # particles
D        = ${py(D)}            # um^2/s
DT       = ${py(dtMs / 1000)}  # s (frame interval)
PIXEL    = ${py(pixel)}        # nm at the sample
NA       = ${py(NA)}
LAMBDA   = ${py(lambda)}       # nm (emission)
N_MEDIUM = ${py(nMedium)}      # immersion index implied by the NA
FIELD    = ${py(field)}        # px
PHOTONS  = ${py(photons)}      # photons per particle per frame
BACKGND  = ${py(background)}   # photons per pixel per frame
READNOISE= ${py(readNoise)}    # electrons rms
QE       = ${py(qe)}
FRAMES   = ${py(frames)}
SEED     = ${py(seed)}

rng = np.random.default_rng(SEED)

# per-axis diffusive step, in pixels
sigma_step_px = np.sqrt(2 * D * DT) * 1000 / PIXEL

# ── motion model ─────────────────────────────────────────────────────────────
${motionBlock(p, field, pixel, dtMs / 1000)}

particle = dt.PointParticle(
    position=lambda: rng.uniform(0, FIELD, 2),
    intensity=1.0,               # arbitrary units; renormalised to photons below
)

def make_sequential(feature, **rules):
    """Attach per-frame update rules, across DeepTrack2 API versions.

    The sequential-property API moved between releases and the two forms do not
    overlap, so neither alone is portable:
      - deeptrack 2.0.1 (PyPI) has the free function deeptrack.sequences.Sequential
        and no Feature.to_sequential;
      - the development branch adds Feature.to_sequential, stops exporting
        Sequential (dt.Sequential raises AttributeError) and leaves the old
        function defective.
    Prefer the method, fall back to the free function by explicit import.
    """
    if hasattr(feature, "to_sequential"):
        return feature.to_sequential(**rules)
    try:
        from deeptrack.sequences import Sequential
    except ImportError as exc:  # pragma: no cover - depends on the installed version
        raise RuntimeError(
            "This deeptrack build exposes neither Feature.to_sequential nor "
            "deeptrack.sequences.Sequential; cannot attach a motion model."
        ) from exc
    return Sequential(feature, **rules)


walker = make_sequential(particle, position=update_position)

optics = dt.Fluorescence(
    NA=NA,
    wavelength=LAMBDA * 1e-9,
    resolution=PIXEL * 1e-9,
    magnification=1,
    refractive_index_medium=N_MEDIUM,
    output_region=(0, 0, FIELD, FIELD),
    # Note on PSF width: DeepTrack samples the pupil on the output grid, and at
    # this pixel size that disc is only ~10 samples in radius, so its pixelated
    # edge widens the PSF by a few percent. Raising \`upscale\` converges it, but
    # upscale > 1 combined with dt.Sequence overflows DeepTrack's FFT size table,
    # so the default is kept here. The browser module's own PSF is the Gaussian
    # approximation sigma = 0.21 lambda / NA, about 4% narrower than an ideal
    # Airy pattern; see the module's validation note.
)

pipeline = optics(walker ^ N)
movie = dt.Sequence(pipeline, sequence_length=FRAMES)

# ── photon normalisation ─────────────────────────────────────────────────────
# DeepTrack's \`intensity\` is not a photon count: a unit-intensity emitter
# integrates to ~0.095 here, and the constant depends on NA, wavelength and
# pixel size. Calibrate it once on a single centred emitter so that each
# particle really does emit PHOTONS photons per frame.
_calib_particle = dt.PointParticle(
    position=(FIELD / 2 + 0.5, FIELD / 2 + 0.5), intensity=1.0,
)
_calib = np.asarray(optics(_calib_particle)()).astype(float)
if _calib.ndim == 3:
    _calib = _calib[..., 0]
PHOTON_SCALE = PHOTONS / _calib.sum()

# ── run ──────────────────────────────────────────────────────────────────────
frames = movie.update()()
frames = [np.asarray(f).astype(float) for f in frames]
frames = [f[..., 0] if f.ndim == 3 else f for f in frames]

electrons = []
for signal in frames:
    photons = signal * PHOTON_SCALE + BACKGND
    # shot noise on the detected photoelectrons, then Gaussian read noise
    e = rng.poisson(np.clip(photons * QE, 0, None)).astype(float)
    if READNOISE > 0:
        e += rng.normal(0, READNOISE, e.shape)
    electrons.append(e)
movie_e = np.stack(electrons)          # (FRAMES, FIELD, FIELD), photoelectrons

print("movie:", movie_e.shape, "mean", movie_e.mean().round(2))

# An equivalent using DeepTrack's own noise convention, for comparison — note it
# fixes a target SNR instead of a photon budget, so absolute counts differ:
#   pipeline = optics(walker ^ N) >> dt.Background(offset=BACKGND) \\
#              >> dt.Poisson(snr=${py(Math.max(1, Math.round(Math.sqrt(photons))))}, background=BACKGND)

# ── quick look ───────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    plt.imshow(movie_e[0], cmap="gray")
    plt.title("frame 0 — photoelectrons")
    plt.colorbar()
    plt.show()
except ImportError:
    pass
`;
}

/** Ground-truth CSV: exact positions, because they are an input, not a fit. */
export function groundTruthCsv(
  truth: { x: Float64Array; y: Float64Array; N: number; frames: number },
  pixelNm: number, photons: number,
): string {
  const rows = ['frame,particle_id,x_nm,y_nm,x_px,y_px,photons'];
  for (let t = 0; t < truth.frames; t++) {
    for (let i = 0; i < truth.N; i++) {
      const x = truth.x[t * truth.N + i];
      const y = truth.y[t * truth.N + i];
      rows.push(
        `${t},${i},${x.toFixed(3)},${y.toFixed(3)},`
        + `${(x / pixelNm).toFixed(4)},${(y / pixelNm).toFixed(4)},${photons}`,
      );
    }
  }
  return rows.join('\n');
}

/** Localized detections, with the linked track id where one was assigned. */
export function tracksCsv(
  tracks: { id: number; points: { frame: number; xPx: number; yPx: number }[] }[],
  pixelNm: number,
): string {
  const rows = ['frame,track_id,x_nm,y_nm,x_px,y_px'];
  const flat: { frame: number; id: number; xPx: number; yPx: number }[] = [];
  for (const tr of tracks) {
    for (const q of tr.points) flat.push({ frame: q.frame, id: tr.id, xPx: q.xPx, yPx: q.yPx });
  }
  flat.sort((a, b) => a.frame - b.frame || a.id - b.id);
  for (const r of flat) {
    rows.push(
      `${r.frame},${r.id},${(r.xPx * pixelNm).toFixed(3)},${(r.yPx * pixelNm).toFixed(3)},`
      + `${r.xPx.toFixed(4)},${r.yPx.toFixed(4)}`,
    );
  }
  return rows.join('\n');
}
