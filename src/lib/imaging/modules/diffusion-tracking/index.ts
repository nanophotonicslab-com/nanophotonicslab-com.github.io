/**
 * IMG1 as a ModuleImpl — the only thing the generic module page knows about it.
 */
import { fmt } from '../../../solver-spec';
import type { ModuleImpl, PlotData, Values } from '../types';
import { diffusionTracking } from './spec';
import { computeFull, computeQuick, toSimParams } from './compute';
import { groundTruthCsv, toPythonScript, tracksCsv } from './export-python';

function psfPlot(q: ReturnType<typeof computeQuick>): PlotData {
  return {
    series: [{ x: q.psfCut.x, y: q.psfCut.y, color: 'primary' }],
    markers: [{ x: q.derived.fwhmPx / 2, label: `FWHM ${fmt(q.derived.fwhmPx, 3)} px` }],
    note: `σ = ${fmt(q.derived.sigmaPsfNm, 3)} nm`,
    csv: () => 'x_px,photons\n' + Array.from(q.psfCut.x)
      .map((x, i) => `${x},${q.psfCut.y[i]}`).join('\n'),
  };
}

function precisionPlot(q: ReturnType<typeof computeQuick>, values: Values): PlotData {
  // log x, because the photon budget spans four decades
  const logx = Float64Array.from(q.precision.x, v => Math.log10(v));
  return {
    series: [{ x: logx, y: q.precision.y, color: 'primary' }],
    markers: [{
      x: Math.log10(Number(values.photons)),
      y: q.derived.sigmaLocNm,
      label: `you are here · ${fmt(q.derived.sigmaLocNm, 3)} nm`,
    }],
    xLabel: 'log₁₀ photons / frame',
    note: `σ_loc = ${fmt(q.derived.sigmaLocNm, 3)} nm`,
    csv: () => 'photons_per_frame,sigma_loc_nm\n' + Array.from(q.precision.x)
      .map((x, i) => `${x},${q.precision.y[i]}`).join('\n'),
  };
}

export const diffusionTrackingModule: ModuleImpl = {
  spec: diffusionTracking,

  sim: values => toSimParams(values),

  quick(values) {
    const q = computeQuick(values);
    return {
      observables: q.observables,
      plots: {
        // the MSD needs the movie, so it is empty until the deferred pass lands
        msd: { series: [], note: 'analysing…' },
        psfCut: psfPlot(q),
        precision: precisionPlot(q, values),
      },
    };
  },

  full(values, onProgress) {
    const f = computeFull(values, onProgress);
    const q = computeQuick(values);
    return {
      truth: f.truth,
      analysis: f.analysis,
      observables: { ...q.observables, dFit: f.dFit, alpha: f.alpha },
      plots: {
        msd: {
          series: [
            { x: f.msd.tau, y: f.msd.localized, color: 'primary', label: 'localized' },
            { x: f.msd.tau, y: f.msd.truth, color: 'truth', dash: [4, 3], label: 'ground truth' },
            { x: f.fitLine.tau, y: f.fitLine.msd, color: 'fit', dash: [2, 2], label: 'fit' },
          ],
          note: `D_fit = ${fmt(f.dFit, 3)} µm²/s (set ${fmt(Number(values.D), 3)}) · α = ${fmt(f.alpha, 2)}`,
          csv: () => 'tau_s,msd_localized_nm2,msd_truth_nm2\n' + Array.from(f.msd.tau)
            .map((t, i) => `${t},${f.msd.localized[i]},${f.msd.truth[i]}`).join('\n'),
        },
        psfCut: psfPlot(q),
        precision: precisionPlot(q, values),
      },
    };
  },

  exports: {
    stem: 'img1',
    python: (values, opt) => toPythonScript(values, opt),
    groundTruth: (truth, values) => groundTruthCsv(truth, Number(values.pixel), Number(values.photons)),
    tracks: (analysis, values) => tracksCsv(analysis.tracks, Number(values.pixel)),
  },
};
