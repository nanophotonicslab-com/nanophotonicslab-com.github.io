/**
 * Shared Spectrum type — the backbone edge of the simulation-pipeline design
 * (docs/node-editor-spec.md). Every spectrum-shaped solver output is morally
 * "x grid + named Float64Array channels"; this is that shape, plus adapters
 * for the pure lib solvers that emit one. Adapters do not copy — channels
 * reference the source arrays.
 *
 * Not adapted here: cylinder.getDispersion (a dispersion relation, not a
 * spectrum) and the Pyodide workers (BEM/RCWA — async, outside the lib).
 */
import type { MieSpectrum } from './mie';
import type { PlasmonSpectrum } from './plasmonic-nanoparticles';
import type { ElectronSpectrum } from './electron-sphere';

export type SpectrumAxis = 'lambdaNm' | 'energyEv';

export interface Spectrum {
  x: Float64Array;
  axis: SpectrumAxis;
  series: Record<string, Float64Array>;
}

/** Plane-wave Mie cross-section spectra (src/lib/mie.ts computeMie / computeMieMultiShell). */
export function spectrumFromMie(r: MieSpectrum): Spectrum {
  return {
    x: r.lambda,
    axis: 'lambdaNm',
    series: {
      csca: r.csca, cext: r.cext, cabs: r.cabs, qsca: r.qsca,
      csca_e: r.csca_e, csca_m: r.csca_m, cext_e: r.cext_e, cext_m: r.cext_m,
    },
  };
}

/** Plasmonic-nanoparticle cross-section spectra (computePlasmonSpectrum). */
export function spectrumFromPlasmon(r: PlasmonSpectrum): Spectrum {
  return {
    x: r.wavelengthNm,
    axis: 'lambdaNm',
    series: {
      sigmaExtNm2: r.sigmaExtNm2, sigmaScaNm2: r.sigmaScaNm2, sigmaAbsNm2: r.sigmaAbsNm2,
      quantumYield: r.quantumYield,
    },
  };
}

/** Peak metrics of one spectrum channel: position, height, FWHM, quality factor. */
export interface PeakAnalysis {
  x0: number; y0: number;
  /** Full width at half maximum; NaN when a half-max crossing falls outside the grid. */
  fwhm: number;
  /** x0 / fwhm — dimensionless Q for either λ or energy axes. */
  q: number;
}

export function peakAnalysis(x: Float64Array, y: Float64Array): PeakAnalysis | null {
  if (x.length < 3) return null;
  let iMax = 0;
  for (let i = 1; i < y.length; i++) if (y[i] > y[iMax]) iMax = i;
  const y0 = y[iMax];
  if (!(y0 > 0)) return null;
  const half = y0 / 2;
  const crossing = (from: number, step: -1 | 1): number => {
    for (let i = from; i + step >= 0 && i + step < y.length; i += step) {
      const a = y[i], b = y[i + step];
      if ((a - half) * (b - half) <= 0 && a !== b) {
        const t = (half - a) / (b - a);
        return x[i] + t * (x[i + step] - x[i]);
      }
    }
    return NaN;
  };
  const xl = crossing(iMax, -1), xr = crossing(iMax, 1);
  const fwhm = Math.abs(xr - xl);
  return { x0: x[iMax], y0, fwhm, q: fwhm > 0 ? x[iMax] / fwhm : NaN };
}

/** EELS/CL spectra of a swift electron near a sphere (computeElectronSphereSpectrum). */
export function spectrumFromElectron(r: ElectronSpectrum): Spectrum {
  return {
    x: r.energy,
    axis: 'energyEv',
    series: {
      eelsTotal: r.eelsTotal, eelsSurface: r.eelsSurface, eelsBulk: r.eelsBulk,
      eelsBegrenzung: r.eelsBegrenzung, clTotal: r.clTotal,
    },
  };
}
