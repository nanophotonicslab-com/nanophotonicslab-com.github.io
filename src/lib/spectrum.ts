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
