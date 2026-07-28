/**
 * Imaging lab manifest.
 *
 * The lab is one page with a module bar, the same shape Heterostructures uses:
 * every module is a panel on `/lab/imaging/`, selected by the bar and by the
 * URL hash. This file is the single place a module is registered — adding one
 * means a spec, a compute binding, and one entry in `modules` below (plus its
 * implementation in `modules/index.ts`).
 */
import type { Solver } from '../solver-spec';
import { diffusionTracking } from './modules/diffusion-tracking/spec';

export interface LabModule {
  /** URL hash fragment and element-id segment. */
  slug: string;
  spec: Solver;
  /** One line under the module name in the bar. */
  tagline: string;
}

/** A module named on the roadmap but not built: shown disabled in the bar. */
export interface PlannedModule {
  slug: string;
  name: string;
  tagline: string;
}

export interface LabManifest {
  id: string;
  title: string;
  blurb: string;
  modules: LabModule[];
  planned: PlannedModule[];
}

export const imagingLab: LabManifest = {
  id: 'imaging',
  title: 'Imaging',
  blurb: 'Simulated microscopy with exact ground truth — the forward model behind '
    + 'quantitative digital microscopy, in the browser.',
  modules: [
    {
      slug: 'diffusion-tracking',
      spec: diffusionTracking,
      tagline: 'Brownian particles, localization & MSD',
    },
  ],
  planned: [
    {
      slug: 'psf-aberrations',
      name: 'PSF & aberrations',
      tagline: 'Pupil-based PSF, Zernike aberrations & defocus',
    },
    {
      slug: 'modality',
      name: 'Modality comparison',
      tagline: 'Brightfield, darkfield, iSCAT & holography',
    },
    {
      slug: 'photophysics',
      name: 'Photophysics',
      tagline: 'Photobleaching, blinking & saturation',
    },
  ],
};

export function findModule(slug: string): LabModule | undefined {
  return imagingLab.modules.find(m => m.slug === slug);
}
