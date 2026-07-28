/**
 * Point-spread function models.
 *
 * v1 uses the Gaussian approximation to the paraxial widefield PSF. The
 * pupil-based implementation (P(kx,ky) = circ(k < NA*k0)*exp(i*sum c_j Z_j),
 * PSF = |FFT{P}|^2) belongs to the later PSF & aberrations module; the
 * interface below is shaped so it can be added without touching callers.
 */

export interface PsfModel {
  /** Standard deviation of the PSF in the sample plane, in nanometres. */
  sigmaNm: number;
  /** Full width at half maximum, in nanometres. */
  fwhmNm: number;
}

/** sqrt(8 ln 2) — the Gaussian sigma-to-FWHM factor. */
export const FWHM_PER_SIGMA = 2.3548200450309493;

/**
 * Gaussian approximation to the widefield emission PSF
 * (Zhang, Zerubia & Olivo-Marin, Appl. Opt. 46, 1819 (2007)):
 *
 *   sigma = 0.21 * lambda / NA
 *
 * @param lambdaNm emission wavelength in nm
 * @param NA objective numerical aperture
 */
export function gaussianPSF(lambdaNm: number, NA: number): PsfModel {
  const sigmaNm = (0.21 * lambdaNm) / NA;
  return { sigmaNm, fwhmNm: FWHM_PER_SIGMA * sigmaNm };
}
