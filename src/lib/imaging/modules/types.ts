/**
 * The contract an Imaging module implements.
 *
 * The module page is a single generic route, so everything module-specific has
 * to arrive through this interface: the spec (which generates the markup), a
 * cheap per-keystroke compute, an expensive deferred compute, and the exports.
 * Anything the page can do for every module — the frame viewer, the overlays,
 * the axes, the download plumbing — stays in the page.
 *
 * If a new module cannot be expressed as one of these, that is the signal to
 * extend this interface rather than to special-case the page.
 */
import type { Solver } from '../../solver-spec';
import type { Analysis, SimParams, Truth } from '../index';

export type Values = Record<string, number | string>;

/** Semantic series colours, mapped to design tokens by the page. */
export type SeriesColor = 'primary' | 'truth' | 'fit' | 'muted';

export interface PlotSeries {
  x: Float64Array | number[];
  y: Float64Array | number[];
  color: SeriesColor;
  dash?: number[];
  label?: string;
}

export interface PlotMarker {
  x: number;
  y?: number;
  label: string;
}

export interface PlotData {
  series: PlotSeries[];
  markers?: PlotMarker[];
  /** Axis captions; the spec supplies defaults, these may override per state. */
  xLabel?: string;
  yLabel?: string;
  /** Short annotation shown in the plot header. */
  note?: string;
  /** Rows for the CSV export, header first. */
  csv?: () => string;
}

export interface QuickResult {
  /** Values for the readout row and the envelope context. */
  observables: Record<string, number>;
  /** Keyed by the plot ids declared in the spec. */
  plots: Record<string, PlotData>;
}

export interface FullResult extends QuickResult {
  truth: Truth;
  analysis: Analysis;
}

export interface ModuleImpl {
  spec: Solver;
  /** Kernel simulation parameters for the frame viewer. */
  sim(values: Values): SimParams;
  /** Cheap enough to run on every parameter change. */
  quick(values: Values): QuickResult;
  /** The movie-wide pass: debounced, reports progress, may take seconds. */
  full(values: Values, onProgress?: (f: number) => void): FullResult;
  exports: {
    /** A runnable script reproducing the run in the reference library. */
    python(values: Values, opt: { permalink: string; version: string }): string;
    /** Filename stem for downloads. */
    stem: string;
    groundTruth(truth: Truth, values: Values): string;
    tracks(analysis: Analysis, values: Values): string;
  };
}
