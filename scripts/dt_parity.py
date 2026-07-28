"""Render one in-focus emitter with DeepTrack2's Fluorescence optics and
measure its width, for the IMG1 parity fixture.

Second moments are used (not a half-max crossing) so the number is directly
comparable to the TypeScript kernel's measureSigmaPx, including the 1/12
pixel-box correction.
"""
import json
import logging
import warnings

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import deeptrack as dt

FIELD = 64
PIXEL_NM = 65.0
# The immersion index must exceed the NA to be physical, so each case names its
# own medium: oil for the high-NA objective, water, then a dry objective.
CASES = [
    {"NA": 1.40, "lambda_nm": 520.0, "n_medium": 1.518},
    {"NA": 1.20, "lambda_nm": 640.0, "n_medium": 1.33},
    {"NA": 0.90, "lambda_nm": 488.0, "n_medium": 1.00},
]
PHOTONS = 1000.0
# DeepTrack samples the pupil on the output grid; at 65 nm pixels that disc is
# only ~10 samples in radius and its pixelated edge broadens the PSF. upscale=4
# is where the width has converged (8 changes it by <0.5%).
UPSCALE = 4
# Half-maximum width of an ideal Airy intensity pattern, in units of lambda/NA:
# the first half-max of (2 J1(v)/v)^2 is at v = 1.61633, giving 2*1.61633/(2 pi).
AIRY_FWHM_COEFF = 0.51446


def moments(img, cx, cy):
    """Total weight, and the per-axis sigma in pixels after removing the
    variance of the 1-pixel box (1/12), matching the TS estimator."""
    ny, nx = img.shape
    xs = np.arange(nx) + 0.5 - cx
    ys = np.arange(ny) + 0.5 - cy
    w = np.clip(img, 0, None)
    tot = w.sum()
    r2 = (xs[None, :] ** 2 + ys[:, None] ** 2)
    m2 = (w * r2).sum() / tot
    per_axis = m2 / 2 - 1 / 12
    return tot, np.sqrt(per_axis) if per_axis > 0 else float("nan")


def fwhm_cut(img, i0, j0):
    """FWHM in pixels along the row through the spot's centre pixel.

    Must match src/lib/imaging/render.ts `measureFwhmCut` exactly. A cut samples
    the profile at exact integer radii, so unlike radial binning it applies the
    same systematic to a narrow and a wide spot — which is what makes the
    Gaussian-versus-pupil comparison meaningful. Second moments are unusable
    here: DeepTrack's PSF is Airy-like and its rings decay as r^-3, so its
    second moment grows with the field of view.
    """
    peak = img[j0, i0]
    half = peak / 2

    def half_width(direction):
        prev = peak
        k = 1
        while 0 <= i0 + direction * k < img.shape[1]:
            v = img[j0, i0 + direction * k]
            if v <= half:
                return k - 1 + (prev - half) / (prev - v)
            prev = v
            k += 1
        return float("nan")

    return half_width(1) + half_width(-1)


def radial_profile(img, cx, cy, nbins):
    ny, nx = img.shape
    xs = np.arange(nx) + 0.5 - cx
    ys = np.arange(ny) + 0.5 - cy
    r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    b = np.round(r).astype(int)
    out = np.zeros(nbins)
    for k in range(nbins):
        m = b == k
        out[k] = img[m].mean() if m.any() else 0.0
    return out


out = {
    "field": FIELD, "pixel_nm": PIXEL_NM, "photons": PHOTONS,
    "upscale": UPSCALE, "airy_fwhm_coeff": AIRY_FWHM_COEFF,
    "generated_by": "scratchpad/dt_parity.py against deeptrack from PyPI",
    "cases": [],
}

for case in CASES:
    NA, lam, nmed = case["NA"], case["lambda_nm"], case["n_medium"]
    # centre the emitter on a pixel centre so the radial bins are populated
    centre = FIELD / 2 + 0.5
    particle = dt.PointParticle(position=(centre, centre), intensity=PHOTONS)
    optics = dt.Fluorescence(
        NA=NA,
        wavelength=lam * 1e-9,
        resolution=PIXEL_NM * 1e-9,
        magnification=1,
        refractive_index_medium=nmed,
        output_region=(0, 0, FIELD, FIELD),
        upscale=UPSCALE,
    )
    img = np.asarray(optics(particle)()).astype(float)
    if img.ndim == 3:
        img = img[..., 0]

    total, sigma_px = moments(img, centre, centre)
    # the emitter sits at the centre of pixel (FIELD//2, FIELD//2)
    fwhm = fwhm_cut(img, FIELD // 2, FIELD // 2)
    prof = radial_profile(img, centre, centre, 20)
    # normalised profile, so the fixture compares shape rather than DeepTrack's
    # internal amplitude convention
    prof_norm = prof / prof[0] if prof[0] > 0 else prof
    sigma_gauss_px = 0.21 * lam / NA / PIXEL_NM
    fwhm_gauss = 2.3548200450309493 * sigma_gauss_px
    fwhm_airy = AIRY_FWHM_COEFF * lam / NA / PIXEL_NM
    out["cases"].append({
        "NA": NA,
        "lambda_nm": lam,
        "n_medium": nmed,
        "total": float(total),
        "peak": float(img.max()),
        "sigma_moment_px": float(sigma_px),
        "fwhm_px": float(fwhm),
        "fwhm_gaussian_px": float(fwhm_gauss),
        "fwhm_ideal_airy_px": float(fwhm_airy),
        "radial_profile_normalized": [float(v) for v in prof_norm],
    })
    print(
        f"NA={NA} lam={lam} n={nmed}: fwhm_dt={fwhm:.4f}px "
        f"gauss={fwhm_gauss:.4f}px ({fwhm / fwhm_gauss:.3f}x) "
        f"airy={fwhm_airy:.4f}px ({fwhm / fwhm_airy:.3f}x) "
        f"gauss/airy={fwhm_gauss / fwhm_airy:.3f}"
    )

with open("dt_parity.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote dt_parity.json")
