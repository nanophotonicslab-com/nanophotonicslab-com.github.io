/**
 * Electrostatics-FD — page controller.
 *
 * Owns the DOM: reads the controls into a solver spec, keeps the geometry
 * schematic and the cost estimate live, runs a calculation on demand, and draws
 * whatever comes back.
 *
 * The page is organised the way the physics is. A SYSTEM (nanoribbons; tubes
 * and disks later) fixes the kernel. An EXCITATION decides what can be asked of
 * it: a cross section needs a plane wave, an LDOS needs a dipole, a loss
 * spectrum needs an electron. Modes and bands are properties of the structure
 * rather than of the drive, so they are offered under every excitation. That
 * three-way split — system, excitation, observable — is what the module bar,
 * the excitation bar and the observable tabs map onto, one each.
 *
 * Two rules shape the rest.
 *
 * 1. NOTHING HEAVY RUNS WITHOUT A CLICK. The schematic and the cost estimate
 *    update as you type, because both are cheap — `geometry` builds no matrices
 *    and `cost_estimate` runs no physics at all. The calculation itself waits
 *    for Run. A drifted or substrate-coupled map is a full solve per (q, E)
 *    cell rather than one decomposition per wave vector, so an auto-run on
 *    keystroke would lock the tab.
 *
 * 2. EVERY `notes` ENTRY IS DISPLAYED. The Python boundary returns an array of
 *    them on every successful payload: accuracy caveats, quadrature auto-tuning
 *    reports, and library warnings it captured rather than swallowed. Dropping
 *    them would turn a hedged number into a confident one.
 */

import { EfdBridge } from './bridge';
import type { EfdComplex, EfdKind, EfdResult } from './bridge';
import {
  drawDivergingHeatmap,
  drawHeatmap,
  drawModeProfiles,
  drawRibbonSchematic,
  drawSpectrum,
} from './plots';
import { seriesColor, seriesLegendHtml } from './plots';
import type { GeometryPayload, MapOverlays, SchematicSource, Series } from './plots';
import { onThemeChange } from '../../lib/chart-theme';
// Three copies of hc already exist in this repo with two different values.
// Import the shared one rather than adding a fourth.
import { HC_EV_NM } from '../../lib/mie';
import { betaFromKeV, energyFromBeta } from '../../lib/transition-radiation';

// --------------------------------------------------------------------------
// DOM helpers
// --------------------------------------------------------------------------

const $ = <T extends HTMLElement = HTMLElement>(id: string): T | null =>
  document.getElementById(id) as T | null;

const num = (id: string, fallback: number): number => {
  const el = $<HTMLInputElement>(id);
  if (!el) return fallback;
  const v = Number.parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
};

const int = (id: string, fallback: number): number => Math.round(num(id, fallback));

const str = (id: string, fallback: string): string => {
  const el = $<HTMLInputElement | HTMLSelectElement>(id);
  return el && el.value !== '' ? el.value : fallback;
};

const setText = (id: string, text: string): void => {
  const el = $(id);
  if (el) el.textContent = text;
};

/** As `setText`, for the handful of titles that carry a subscript.
 *
 *  Only ever called with constants defined in this file -- nothing a user
 *  typed, nothing out of a payload, reaches it. */
const setHtml = (id: string, html: string): void => {
  const el = $(id);
  if (el) el.innerHTML = html;
};

/** The colour key under a chart. A three-band plot without one is unreadable. */
const setLegend = (id: string, series: Series[] | null): void => {
  const el = $(id);
  if (!el) return;
  el.innerHTML = series ? seriesLegendHtml(series) : '';
  // The sentence layout is opt-in and belongs to one caller, so it is dropped
  // here rather than left for every other legend to remember to remove.
  el.classList.remove('chart-legend--note');
};

/** Seconds as something a person reads at a glance. */
function fmtSeconds(s: number): string {
  if (!Number.isFinite(s)) return '—';
  if (s < 0.1) return '<0.1 s';
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - 60 * m)} s`;
}

/** The UI offers this many structures; the solver itself has no such limit. */
const MAX_RIBBONS = 5;

/** How far below the plotted energy range the band root-find is allowed to go.
 *
 *  The library's own default, and low enough that what ends a band is the
 *  physics rather than the window: below `gamma` the mode is damped out and
 *  `drude_band_energy` has no real root, which for any sane graphene gamma
 *  happens well above this. Bands drawn below the frame are clipped by the
 *  plot, not dropped, so searching down here only costs a few root finds. */
const BAND_SEARCH_FLOOR_EV = 1e-3;

/** The smallest energy the model can be EVALUATED at.
 *
 *  `eta = i sigma / (omega W eps)` diverges as `1/omega`, so E = 0 is a genuine
 *  singularity of the physics rather than an awkward axis value, and the
 *  library refuses it outright.
 *
 *  But zero is a perfectly reasonable thing to want on the AXIS -- the
 *  longitudinal branch ends exactly there -- so the two are not the same
 *  number. Typing 0 into the energy range used to go straight through to the
 *  solver and come back as `ValueError: need 0 < E_min < E_max, got (0.0,
 *  1.0)`: a library contract quoted at a reader who had asked for something
 *  entirely sensible. The axis keeps what they typed; only what is handed to
 *  the solver is clamped here. */
const MIN_SOLVABLE_EV = 1e-4;

/** `energyGrid()` with the lower end made solvable. Everything that reaches
 *  the library goes through this; everything that draws an axis does not. */
function solvedEnergyGrid(): { min: number; max: number; n: number } {
  const g = energyGrid();
  return { min: Math.max(g.min, MIN_SOLVABLE_EV), max: g.max, n: g.n };
}

// --------------------------------------------------------------------------
// Excitations and observables
// --------------------------------------------------------------------------

type Excitation = 'plane' | 'dipole' | 'electron';
type Observable =
  | 'spectrum'
  | 'nonlinear'
  | 'ldos'
  | 'eels'
  | 'nearfield'
  | 'modes'
  | 'dispersion'
  | 'material';

interface ObservableDef {
  kind: EfdKind;
  /** Which excitation this observable needs, or 'all' for a structure property. */
  needs: Excitation | 'all';
  mainTitle: string;
  auxTitle: string;
  headlineUnit: string;
  /** Sweeps a +-q grid rather than a single wave vector. */
  qGrid: boolean;
}

const OBSERVABLES: Record<Observable, ObservableDef> = {
  spectrum: {
    kind: 'linear_spectrum',
    needs: 'plane',
    mainTitle: 'Extinction cross section, σ = (4πω/c) Im α',
    auxTitle: 'Sheet conductivity σ(ω)  [nm/fs]',
    headlineUnit: 'peak energy, eV',
    qGrid: false,
  },
  nonlinear: {
    kind: 'nonlinear_spectrum',
    needs: 'plane',
    mainTitle: 'Nonlinear polarizability |α| vs fundamental energy',
    auxTitle: 'Sheet conductivity σ(ω)  [nm/fs]',
    headlineUnit: 'peak energy, eV',
    qGrid: false,
  },
  ldos: {
    kind: 'ldos',
    needs: 'dipole',
    mainTitle: 'LDOS over (q, ω) — arbitrary units, log scale',
    auxTitle: '±q asymmetry — symmetric scale, zero is neutral',
    headlineUnit: 'strongest peak, eV',
    qGrid: true,
  },
  eels: {
    kind: 'eels_spectrum',
    needs: 'electron',
    mainTitle: 'Loss spectrum — aloof swift electron',
    auxTitle: 'Sheet conductivity σ(ω)  [nm/fs]',
    headlineUnit: 'loss peak, eV',
    // A q grid, because the loss is defined per wave vector and the spectrum
    // is its integral: Gamma(w) = Int dk Gamma(k, w), paper 430's Methods.
    // Both are drawn, the map and the integral, through #efd-eels-pick.
    qGrid: true,
  },
  modes: {
    kind: 'eigenmodes',
    needs: 'all',
    mainTitle: 'Geometric eigenvalues η_j and their Drude band energies',
    auxTitle: 'Mode profiles across the ribbon',
    headlineUnit: 'η₁, geometric',
    qGrid: false,
  },
  dispersion: {
    kind: 'bands',
    needs: 'all',
    mainTitle: 'Ribbon bands — energy vs guided wave vector',
    // No aux panel: an unpatterned sheet is not what this view is about.
    auxTitle: '',
    headlineUnit: 'band 1 at q max, eV',
    qGrid: true,
  },
  nearfield: {
    kind: 'near_field',
    needs: 'all',
    mainTitle: '|E| of the induced field over (x, z) — arbitrary scale, log',
    auxTitle: 'Re E_x — signed, symmetric scale about zero',
    headlineUnit: 'strongest |E|, arb.',
    qGrid: false,
  },
  material: {
    kind: 'material',
    needs: 'all',
    mainTitle: 'Sheet conductivity σ(ω)  [nm/fs]',
    auxTitle: 'η(ω) = iσ/ωWε̄',
    headlineUnit: 'μ at this T and β, eV',
    qGrid: false,
  },
};

/** What each excitation shows when you first switch to it. */
const DEFAULT_OBSERVABLE: Record<Excitation, Observable> = {
  plane: 'spectrum',
  dipole: 'ldos',
  electron: 'eels',
};

let excitation: Excitation = 'plane';
let observable: Observable = 'spectrum';

/** Which near-field component the single map shows, and on what scale. */
type FieldKey = 'map' | 'ex_re' | 'ex_im' | 'ey_re' | 'ey_im' | 'ez_re' | 'ez_im';
/** Set as HTML, not text: Unicode has a subscript x but no subscript y or z,
 *  so the only way to typeset all three alike is markup. These are constants
 *  defined here, never anything a user typed. */
const FIELD_LABEL: Record<FieldKey, string> = {
  map: '|E|',
  ex_re: 'Re E<sub>x</sub>',
  ex_im: 'Im E<sub>x</sub>',
  ey_re: 'Re E<sub>y</sub>',
  ey_im: 'Im E<sub>y</sub>',
  ez_re: 'Re E<sub>z</sub>',
  ez_im: 'Im E<sub>z</sub>',
};
let nfComponent: FieldKey = 'map';

/** Which form of the extinction cross section the spectrum shows.
 *
 *  Both are absolute. `nm` is sigma per unit length of ribbon, which is what
 *  the cross section physically is for a structure invariant along y; `area`
 *  divides by the total width to give the dimensionless efficiency, which is
 *  the form the plasmonic-nanoparticles lab plots and the one that compares
 *  across different widths. Neither is arbitrary. */
type XsUnit = 'area' | 'nm';
let xsUnit: XsUnit = 'area';

/** Whether the near-field map draws the drive over the field.
 *
 *  The electron's trajectory and the dipole's marker both come out of
 *  `source_marker`, which the payload already carries, so this costs a redraw
 *  and never a solve. The ribbons stay drawn either way: they are the
 *  structure, not the source, and a map without them cannot be read at all. */
let nfShowSource = true;

/** Does the LDOS map draw the peak tracks it found in itself?
 *
 *  A display switch over data the payload already carries, so it costs a
 *  redraw and never a solve. On by default: they came free with the run that
 *  drew the map, and they are what the map says about itself. */
let ldosShowPeaks = true;

/** Band curves to lay over the LDOS map, or null when the box is unticked.
 *
 *  Filled by the closure, which is the only place that can reach the solver;
 *  `drawResult` is module level and just draws whatever is here. */
let ldosBandTracks: { xs: number[]; ys: (number | null)[]; color?: string; dash?: number[] }[]
  | null = null;

/** The one colour on the LDOS map that is not the colormap.
 *
 *  The map's own peak tracks are white and have already spent solid, dashed
 *  and dotted on themselves, so a fourth white line would read as a fourth
 *  peak. Magenta is nowhere in viridis, which is why it survives being drawn
 *  over any part of it. */
const BAND_OVERLAY_COLOR = '#ff5fbf';

/** What the band overlay is, in a sentence, or null when it is not drawn.
 *
 *  It travels with `ldosBandTracks` because every redraw clears the legend,
 *  and a line whose provenance is only in the message that announced it loses
 *  that provenance at the next theme change. */
let ldosBandNote: string | null = null;

/** The spec these bands were solved from, so the page can tell when they stop
 *  describing the structure on screen.
 *
 *  The overlay outlives a Run -- it is a display switch, and the map under it
 *  is recomputed without asking. Widen a ribbon, Run, and the old bands would
 *  sit over the new map looking like agreement, now with a caption asserting
 *  what they are. A stale line is worse than no line, and a stale labelled one
 *  is worse again, so the overlay leaves the moment its spec does. */
let ldosBandKey: string | null = null;

/** Which face of the loss the EELS view shows.
 *
 *  `Gamma(q, w)` is what the solver produces and `Gamma(w) = Int dq Gamma(q, w)`
 *  is what an experiment without momentum resolution measures, so both belong
 *  on the page and neither is a convenience derived from the other. One run
 *  gives both. */
/** Log or linear on the nonlinear |alpha| axis. Log by default: the orders
 *  span decades, and on a linear axis the third harmonic is a flat line along
 *  the bottom of the frame. */
let nlLog = true;

type EelsView = 'spectrum' | 'map';
let eelsView: EelsView = 'spectrum';

/** Which of the nonlinear polarizabilities the chart shows. */
type NlPick = 'all' | 'lin' | 'shg' | 'thg';
let nlPick: NlPick = 'all';

/** The orders a nonlinear payload actually carries, in drawing order. */
const NL_ORDERS: {
  key: Exclude<NlPick, 'all'>;
  field: string;
  tab: string;
  label: string;
}[] = [
  // The linear one is here because the sweep solves it at every energy anyway
  // -- each harmonic is driven by it -- so the view can show what the
  // harmonics are being compared against without a second run.
  {
    key: 'lin',
    field: 'alpha_linear',
    tab: '\u03b1\u207d\u00b9\u02bc\u00b9\u207e',
    label: '|\u03b1\u207d\u00b9\u02bc\u00b9\u207e|  linear',
  },
  {
    key: 'shg',
    field: 'alpha_shg',
    tab: '\u03b1\u207d\u00b2\u02bc\u00b2\u207e',
    label: '|\u03b1\u207d\u00b2\u02bc\u00b2\u207e|  second harmonic',
  },
  {
    key: 'thg',
    field: 'alpha_thg',
    tab: '\u03b1\u207d\u00b3\u02bc\u00b3\u207e',
    label: '|\u03b1\u207d\u00b3\u02bc\u00b3\u207e|  third harmonic',
  },
];

/** Rebuild the "Show" menu from what the run produced.
 *
 *  Built from the payload rather than from the orders select, because the two
 *  can disagree: the select says what the NEXT run will compute, the payload
 *  says what the one on screen did. Offering a curve that was never computed
 *  would be a lie about the result. */
function renderNlPick(res: EfdResult | null): void {
  const host = $('efd-nl-pick');
  if (!host) return;
  // Two sources, each right at its own moment. With a result in hand the
  // PAYLOAD decides, because it says what was actually computed. With none --
  // landing on the view, or after changing the orders -- the SELECT decides,
  // because it says what the next run will compute. Building only from the
  // payload left the menu empty until the first run finished: the element was
  // there, in the header, four pixels wide and holding nothing, which is
  // indistinguishable from not having been built at all.
  // With a result in hand the PAYLOAD decides, because it says what was
  // actually computed. With none -- landing on the view before the first run
  // -- every order is offered, because every run computes every order.
  const have = res ? NL_ORDERS.filter(o => res[o.field]) : NL_ORDERS;
  const opts: { key: NlPick; label: string }[] = have.map(o => ({
    key: o.key,
    label: o.tab,
  }));
  if (opts.length > 1) opts.unshift({ key: 'all', label: 'All' });
  if (!opts.some(o => o.key === nlPick)) nlPick = opts[0]?.key ?? 'all';
  host.innerHTML = opts
    .map(
      o =>
        `<button class="plot-tab${o.key === nlPick ? ' active' : ''}" type="button" ` +
        `data-nl="${o.key}" role="radio" aria-checked="${o.key === nlPick}">${o.label}</button>`,
    )
    .join('');
}
let nfLog = true;

const availableHere = (o: Observable): boolean => {
  const needs = OBSERVABLES[o].needs;
  return needs === 'all' || needs === excitation;
};

// --------------------------------------------------------------------------
// Spec assembly
// --------------------------------------------------------------------------

function ribbonSpec(j: number): Record<string, number> {
  return {
    W_nm: num(`efd-w-${j}`, 10),
    x0_nm: num(`efd-x0-${j}`, 0),
    z0_nm: num(`efd-z0-${j}`, 0),
    EF_eV: num(`efd-ef-${j}`, 0.5),
    gamma_eV: num(`efd-gamma-${j}`, 0.02),
    beta: num(`efd-beta-${j}`, 0),
    T_K: num(`efd-temp-${j}`, 0),
    L: int(`efd-l-${j}`, 121),
  };
}

/** Does any ribbon on the page carry drift? */
function driftingNow(): boolean {
  const n = Math.min(MAX_RIBBONS, Math.max(1, int('efd-nribbons', 1)));
  return Array.from({ length: n }, (_, j) => num(`efd-beta-${j}`, 0)).some(b => b !== 0);
}

/** `spec` with drift switched off everywhere, and whether that changed anything.
 *
 *  Both places beta lives: each ribbon's own, which is what `_build_system`
 *  reads, and the flat `material` block the Level 1 sheet map stands on. A
 *  copy either way -- the caller still needs the real spec to key its cache
 *  and to run every other view. */
function withoutDrift(
  spec: Record<string, unknown>,
): { spec: Record<string, unknown>; drifting: boolean } {
  const ribbons = (spec.ribbons as Record<string, number>[] | undefined) ?? [];
  const mat = (spec.material as Record<string, number> | undefined) ?? undefined;
  const drifting = ribbons.some(r => r.beta !== 0) || (mat ? mat.beta !== 0 : false);
  if (!drifting) return { spec, drifting: false };
  return {
    spec: {
      ...spec,
      ribbons: ribbons.map(r => ({ ...r, beta: 0 })),
      ...(mat ? { material: { ...mat, beta: 0 } } : {}),
    },
    drifting: true,
  };
}

function sourceSpec(): Record<string, unknown> {
  if (excitation === 'dipole') {
    const axis = str('efd-dip-p', 'x');
    const p: [number, number, number] =
      axis === 'y' ? [0, 1, 0] : axis === 'z' ? [0, 0, 1] : [1, 0, 0];
    return { kind: 'dipole', r0_nm: [num('efd-dip-x', 0), 0, num('efd-dip-z', 5)], p };
  }
  if (excitation === 'electron') {
    return { kind: 'electron', v_over_c: num('efd-vc', 0.2), b_nm: num('efd-b', 5) };
  }
  return { kind: 'plane_wave' };
}

/** The spec for the current observable, in the shape `web.run` expects. */
/** Every number input that is paired with a range slider: width and Fermi
 *  level on each ribbon, plus the near-field map's energy up in the toolbar. */
const SLIDER_PAIRS: readonly string[] = [
  ...Array.from({ length: MAX_RIBBONS }, (_, j) => [`efd-w-${j}`, `efd-ef-${j}`]).flat(),
  'efd-nf-energy',
];

/** The near-field map's energy slider spans the energy grid already set, and
 *  the wavelength beside it follows the energy. One slider, not two: eV is what
 *  the solver takes, nm is the reading. */
function syncNfEnergy(): void {
  const E = num('efd-nf-energy', 0.58);
  const s = $<HTMLInputElement>('efd-nf-energy-slider');
  if (s) {
    // Widened to include the current value: a slider whose thumb cannot reach
    // it would snap the energy on the first drag.
    const g = energyGrid();
    s.min = String(Math.min(g.min, E));
    s.max = String(Math.max(g.max, E));
    // AFTER the bounds, never before. The pair handler writes `s.value = n.value`
    // against the OLD max, and the browser clamps it there and then: type 2.5
    // into a 0.1-1.2 slider and the thumb stuck at 1.2 while the box, the
    // lambda readout and the spec all said 2.5 -- and the next nudge of the
    // thumb wrote 1.2 back over the energy.
    s.value = String(E);
  }
  // The wavelength box is an entry, not a readout, so it is only written
  // here -- never while it has focus, or it would rewrite what is being typed
  // under the cursor.
  const lam = $<HTMLInputElement>('efd-nf-lambda');
  if (lam && document.activeElement !== lam) {
    lam.value = E > 0 ? String(Number((HC_EV_NM / E).toPrecision(5))) : '';
  }
}

/** Keep the electron's kinetic energy showing the speed that is set.
 *
 *  v/c is the parameter; K is the reading, except while it is being typed. */
function syncElectronSpeed(): void {
  const kev = $<HTMLInputElement>('efd-kev');
  if (!kev || document.activeElement === kev) return;
  const beta = num('efd-vc', 0.2);
  kev.value =
    beta > 0 && beta < 1 ? String(Number(energyFromBeta(beta).toPrecision(5))) : '';
}

/** Show the node spacing each ribbon's grid actually has.
 *
 *  `solver.RibbonSystem` lays nodes at `x = x0 + W * linspace(-0.5, 0.5, L)`,
 *  so the spacing is `W/(L-1)` exactly -- not `W/L`. It is the number that
 *  decides whether the edge singularity is resolved, which the node count on
 *  its own never said. */
function updateNodeSpacing(): void {
  for (let j = 0; j < MAX_RIBBONS; j += 1) {
    const out = $(`efd-dx-${j}`);
    if (!out) continue;
    const W = num(`efd-w-${j}`, 0);
    const L = int(`efd-l-${j}`, 0);
    out.textContent = W > 0 && L > 1 ? (W / (L - 1)).toPrecision(3) : '\u2014';
  }
}

/** The eV endpoints as last entered IN eV.
 *
 *  A round trip eV -> nm -> eV through the displayed, rounded numbers does not
 *  return what was typed: 0.1 eV shows as 12398 nm and comes back 0.1000034,
 *  which both looks sloppy in the box and makes the spec differ from the one
 *  that was run, lighting the stale flag for a change that never happened. So
 *  the eV pair is remembered along with the exact strings written into the nm
 *  boxes; while those strings are untouched, the original numbers are used. */
let evRange: { min: number; max: number; nmMin: string; nmMax: string } | null = null;

const fmtEv = (v: number): string => String(Number(v.toPrecision(4)));
const fmtNm = (v: number): string => String(Number(v.toFixed(v >= 100 ? 0 : 1)));

/** The energy grid in eV, whatever unit the two boxes are showing.
 *
 *  THE one conversion point: everything downstream -- the spec, the cost
 *  estimate, the near-field slider's span -- speaks eV, because the library
 *  does. In nm the two ends swap, since the smallest wavelength is the largest
 *  energy. */
function energyGrid(): { min: number; max: number; n: number } {
  const n = int('efd-e-n', 160);
  if (xUnit === 'energy') {
    return { min: num('efd-e-min', 0.1), max: num('efd-e-max', 1.2), n };
  }
  const loS = $<HTMLInputElement>('efd-e-min')?.value ?? '';
  const hiS = $<HTMLInputElement>('efd-e-max')?.value ?? '';
  if (evRange && loS === evRange.nmMin && hiS === evRange.nmMax) {
    return { min: evRange.min, max: evRange.max, n };
  }
  const a = Number(loS);
  const b = Number(hiS);
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return {
    min: hi > 0 ? HC_EV_NM / hi : 0.1,
    max: lo > 0 ? HC_EV_NM / lo : 1.2,
    n,
  };
}

/** The per-ribbon summary under the geometry drawing.
 *
 *  Built from the geometry payload, which `refreshLive` fetches on every
 *  control change, so the table follows what is typed rather than what was
 *  last run. Width, node count and spacing come straight from the solver's own
 *  grid -- `x_nodes_nm.length` is what it actually built, not what was asked
 *  for -- and the material columns from the same ribbons it built them for. */
function renderRibbonTable(geo: GeometryPayload, epsEff?: number): void {
  const host = $<HTMLTableElement>('efd-ribbon-table');
  if (!host) return;
  const ribs = (geo.ribbons ?? []) as unknown as {
    W_nm: number;
    L: number;
    EF_eV: number;
    gamma_eV: number;
    beta: number;
    T_K: number;
  }[];
  const num3 = (v: number | undefined): string =>
    v === undefined || !Number.isFinite(v) ? '\u2014' : String(Number(v.toPrecision(3)));
  const head =
    '<thead><tr><th>Ribbon</th><th>W (nm)</th><th>Points</th><th>\u0394x (nm)</th>' +
    '<th>E_F (eV)</th><th>\u03b3 (eV)</th><th>\u03b2</th><th>T (K)</th></tr></thead>';
  const body = ribs
    .map((r, j) => {
      // W/(L-1), the spacing the solver actually laid down: its nodes are
      // x0 + W * linspace(-0.5, 0.5, L).
      const dx = r.L > 1 ? r.W_nm / (r.L - 1) : Number.NaN;
      return (
        `<tr><td>${j + 1}</td><td>${num3(r.W_nm)}</td><td>${r.L}</td>` +
        `<td>${num3(dx)}</td><td>${num3(r.EF_eV)}</td><td>${num3(r.gamma_eV)}</td>` +
        `<td>${num3(r.beta)}</td><td>${num3(r.T_K)}</td></tr>`
      );
    })
    .join('');
  host.innerHTML = head + `<tbody>${body}</tbody>`;
  const total = ribs.reduce((a, r) => a + (r.L ?? 0), 0);
  setText(
    'efd-table-caption',
    `${ribs.length} ribbon${ribs.length === 1 ? '' : 's'} \u00b7 ${total} nodes` +
      (epsEff === undefined ? '' : ` \u00b7 \u03b5_eff = ${Number(epsEff.toPrecision(4))}`),
  );
}

function readSpec(obs: Observable): Record<string, unknown> {
  const n = Math.min(MAX_RIBBONS, Math.max(1, int('efd-nribbons', 1)));
  const ribbons = Array.from({ length: n }, (_, j) => ribbonSpec(j));

  const spec: Record<string, unknown> = {
    ribbons,
    eps_a: num('efd-eps-a', 1),
    eps_b: num('efd-eps-b', 1),
    // Temperature belongs to each ribbon now. A top-level one is still sent
    // because the extended-sheet map (Level 1) has no ribbon to read it from:
    // there it is ribbon 1's, which is the sheet that map stands for. The
    // library also uses it as the default for any ribbon that carries none.
    T_K: num('efd-temp-0', 0),
    energies_eV: solvedEnergyGrid(),
  };

  // A substrate epsilon of 0 means "none". The spec's `substrate` is the stack
  // BELOW the ribbons' own medium; the boundary prepends the ambient layer and
  // enforces the library's layers[0] convention, so nothing here has to know it.
  //
  // The finite layers come first and the substrate is the semi-infinite
  // terminator, whose thickness the library ignores. The ambient layer is NOT
  // sent: `_layers` prepends it with d_nm = 0, which is also what keeps the
  // standoff unambiguous -- it lives in each ribbon's z0 and nowhere else.
  const subEps = num('efd-sub-eps', 0);
  if (subEps > 0) {
    const layers: Record<string, number>[] = [];
    document.querySelectorAll<HTMLElement>('.efd-layer-row').forEach(row => {
      const field = (key: string): number => {
        const el = row.querySelector<HTMLInputElement>(`[data-key="${key}"]`);
        return el ? Number(el.value) : 0;
      };
      const epsPar = field('eps_par');
      layers.push({ d_nm: field('d_nm'), eps_par: epsPar, eps_perp: field('eps_perp') || epsPar });
    });
    layers.push({ eps_par: subEps });
    spec.substrate = layers;
    // eps_b and the stack are two descriptions of the SAME interface. eps_b
    // enters only through eps_eff = (eps_a + eps_b)/2, which scales every eta,
    // while the stack screens that interface again through r(Q) inside Vbar --
    // so sending both screens it twice, and the library does not complain. The
    // stack wins, and the page hides the eps-below box while it is in force.
    spec.eps_b = spec.eps_a;
  }

  if (OBSERVABLES[obs].qGrid) {
    spec.qs = {
      min: num('efd-q-min', 0.02),
      max: num('efd-q-max', 0.5),
      n_pos: int('efd-q-npos', 16),
    };
  } else {
    spec.q_nm_inv = num('efd-q', 0);
  }

  spec.material = {
    EF_eV: num('efd-ef-0', 0.5),
    gamma_eV: num('efd-gamma-0', 0.02),
    beta: num('efd-beta-0', 0),
    T_K: num('efd-temp-0', 0),
  };

  if (obs === 'modes') spec.n_modes = int('efd-nmodes', 4);
  if (obs === 'dispersion') {
    spec.n_bands = int('efd-nbands', 3);
    // The TOP of the search is the energy grid, so the bands you get are the
    // ones inside the LDOS map next door and the two plots can be laid against
    // each other -- which is why this was tied to the grid in the first place.
    //
    // The BOTTOM is not. Tying it to the grid floor made the search stop where
    // the FRAME stops, and the longitudinal branch -- whose energy goes to zero
    // as q -> 0 -- was cut off in mid-air above the axis: a curve that visibly
    // broke a third of the way down the picture. The plot already clips to
    // `yRange`, so a point below the frame is drawn and clipped rather than
    // dropped; searching lower therefore costs nothing and lets that branch run
    // off the bottom edge the way it actually behaves. What stops it now is the
    // physics -- below `gamma` the mode is damped out and has no resonance at
    // all -- instead of an axis limit.
    const g = solvedEnergyGrid();
    spec.E_min_eV = Math.max(MIN_SOLVABLE_EV, Math.min(g.min, BAND_SEARCH_FLOOR_EV));
    spec.E_max_eV = g.max;
  }
  if (obs === 'ldos') {
    spec.n_peaks = int('efd-npeaks', 3);
    spec.quantity = 'ldos';
  }
  if (obs === 'nonlinear') {
    // Always both. Which one is DRAWN is the menu beside the chart title, and
    // a parameter that only decided what the menu could offer was the same
    // choice asked twice -- once before the run and once after. The cost model
    // already priced both: _work_counts charges 3 solves per energy for this
    // kind whatever `orders` says.
    spec.orders = ['shg', 'thg'];
    spec.cascade = true;
  }
  if (obs === 'nearfield') {
    spec.E_ev = num('efd-nf-energy', 0.58);
    spec.nx = int('efd-nf-nx', 140);
    spec.nz = int('efd-nf-nz', 84);
  }

  // Modes, bands and the material response are eta-free or source-free
  // properties and take no drive; everything else is driven by the excitation.
  if (obs !== 'modes' && obs !== 'dispersion' && obs !== 'material') {
    spec.source = sourceSpec();
  }

  return spec;
}

// --------------------------------------------------------------------------
// Notes and errors
// --------------------------------------------------------------------------

function showNotes(notes: string[] | undefined): void {
  const block = $('efd-notes-block');
  const list = $<HTMLUListElement>('efd-notes');
  if (!block || !list) return;
  list.innerHTML = '';
  if (!notes || notes.length === 0) {
    block.hidden = true;
    return;
  }
  for (const n of notes) {
    const li = document.createElement('li');
    li.textContent = n;
    // The boundary prefixes captured Python warnings with their class name, and
    // writes its own hard caveats in capitals. Both deserve the warning colour.
    if (/Warning:|NOT |CLAMPED|arbitrary units/i.test(n)) li.classList.add('efd-note--warn');
    list.appendChild(li);
  }
  block.hidden = false;
}

function showError(message: string | null): void {
  let box = $('efd-error');
  if (!box) {
    const anchor = $('efd-notes-block');
    const panel = anchor?.parentElement;
    if (!panel) return;
    box = document.createElement('div');
    box.id = 'efd-error';
    box.className = 'efd-error';
    panel.insertBefore(box, anchor);
  }
  if (message === null) {
    box.hidden = true;
    box.textContent = '';
  } else {
    box.hidden = false;
    box.textContent = message;
  }
}

// --------------------------------------------------------------------------
// Drawing
// --------------------------------------------------------------------------

const row = (rows: (number | null)[][], i: number): (number | null)[] => rows[i] ?? [];

/**
 * What goes ON a near-field map: the ribbons, and where the source is.
 *
 * A field map without the structure drawn on it is a picture of colour — the
 * question being asked is always how the field sits relative to the sheet and
 * to whatever is driving it.
 */
function nearFieldOverlays(res: EfdResult): MapOverlays {
  const ribbons = (res.ribbons ?? []) as {
    x_min_nm: number;
    x_max_nm: number;
    z0_nm: number;
  }[];
  const segments: NonNullable<MapOverlays['segments']> = ribbons.map(r => ({
    x0: r.x_min_nm,
    y0: r.z0_nm,
    x1: r.x_max_nm,
    y1: r.z0_nm,
    kind: 'structure' as const,
  }));
  const markers: NonNullable<MapOverlays['markers']> = [];

  // Unticked, the source simply is not looked up, so neither branch below
  // adds anything and the ribbons are all that is drawn.
  const src = (nfShowSource ? res.source_marker : undefined) as
    | {
        kind?: string;
        x_nm?: number;
        z_nm?: number;
        p?: number[];
        b_nm?: number;
        v_over_c?: number;
      }
    | undefined;
  if (src?.kind === 'dipole' && src.x_nm !== undefined && src.z_nm !== undefined) {
    const p = src.p ?? [1, 0, 0];
    const [pxv, pyv, pzv] = [Math.abs(p[0] ?? 0), Math.abs(p[1] ?? 0), Math.abs(p[2] ?? 0)];
    const axis = pxv >= pzv && pxv >= pyv ? 'x' : pzv >= pyv ? 'z' : 'y';
    markers.push({ x: src.x_nm, y: src.z_nm, label: 'dipole', axis });
  } else if (src?.kind === 'electron' && src.b_nm !== undefined) {
    // The electron runs across the ribbon at its impact height, so the whole
    // width of the map is its path; the arrowhead gives it a direction.
    const xs = (res.xs_nm as number[]) ?? [];
    if (xs.length > 1) {
      segments.push({
        x0: xs[0],
        y0: src.b_nm,
        x1: xs[xs.length - 1],
        y1: src.b_nm,
        kind: 'trajectory' as const,
      });
    }
  }
  return { segments, markers };
}

/** One near-field component into one canvas. |E| is positive, the rest signed. */
function drawNearFieldInto(
  canvas: HTMLCanvasElement,
  res: EfdResult,
  key: FieldKey,
  log: boolean,
): void {
  const common = {
    xs: res.xs_nm as number[],
    ys: res.zs_nm as number[],
    z: res[key] as (number | null)[][],
    xLabel: 'x  (nm)',
    yLabel: 'z  (nm)',
    overlays: nearFieldOverlays(res),
  };
  // A signed component on a positive colormap would hide its sign, which is
  // the whole structure of a dipolar mode; |E| on a diverging one would waste
  // half the scale.
  if (key === 'map') drawHeatmap(canvas, { ...common, log });
  else drawDivergingHeatmap(canvas, common);
}

/** Which unit the horizontal axis of the LINE views is drawn in. Energy is
 *  native: the library speaks eV only and refuses a non-positive energy.
 *
 *  Only the line views get this. The maps carry energy vertically on a grid the
 *  drawing code assumes uniform, and a grid uniform in eV is NOT uniform in
 *  wavelength, so converting them would draw the rows in the wrong places. The
 *  toggle hides itself there instead. */
type XUnit = 'energy' | 'lambda';
let xUnit: XUnit = 'energy';

/** The observables whose main chart actually has energy on the horizontal
 *  axis. The rest either plot against q or are maps. */
const ENERGY_X: ReadonlySet<Observable> = new Set<Observable>([
  'spectrum',
  'nonlinear',
  'eels',
  'material',
]);

interface AxisView {
  xs: number[];
  series: Series[];
  markers: { x: number; label: string }[];
  xLabel: string;
}

/** Reverse the DATA, never the axis. `drawSpectrum` maps x with an increasing
 *  function, so handing it a descending `xs` would mirror every curve instead
 *  of flipping the axis. x, every y and every marker turn over together. */
function energyAxis(
  Es: number[],
  series: Series[],
  markers: { x: number; label: string }[] = [],
): AxisView {
  if (xUnit === 'energy') return { xs: Es, series, markers, xLabel: 'Energy  (eV)' };
  const n = Es.length;
  const xs = Array.from({ length: n }, (_, i) => HC_EV_NM / Es[n - 1 - i]);
  return {
    xs,
    series: series.map(se => ({ ...se, ys: se.ys.slice().reverse() })),
    markers: markers.map(m => ({ ...m, x: HC_EV_NM / m.x })),
    xLabel: 'Wavelength  (nm)',
  };
}

/** Which observable the canvases currently hold.
 *
 *  Switching views does not recompute, so without this the previous view's
 *  curves sit under the new view's title, and the aux panel stays open over a
 *  plot that answers a different question. */
let lastDrawnObs: Observable | null = null;

/** Wipe a canvas back to nothing. */
function blankCanvas(c: HTMLCanvasElement): void {
  const ctx = c.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, c.width, c.height);
}

/** One ribbon's slice of the material payload. */
type MaterialRibbon = {
  index: number;
  W_nm: number;
  EF_eV: number;
  gamma_eV: number;
  beta: number;
  T_K: number;
  material: {
    sigma_nm_fs: EfdComplex;
    sigma_drude_nm_fs?: EfdComplex;
    pauli_edge_eV?: number;
  };
  eta_perp: EfdComplex;
  eta_par: EfdComplex;
  sigma_perp_nm_fs?: EfdComplex;
  sigma_par_nm_fs?: EfdComplex;
};

/** "Ribbon 2", "Ribbons 1-3", "Ribbons 1, 3" -- who a shared curve stands for. */
function ribbonTag(idx: number[]): string {
  const first = idx[0] ?? 0;
  const last = idx[idx.length - 1] ?? first;
  if (idx.length === 1) return `Ribbon ${first + 1}`;
  const run = idx.every((v, i) => i === 0 || v === (idx[i - 1] ?? -2) + 1);
  return run
    ? `Ribbons ${first + 1}\u2013${last + 1}`
    : `Ribbons ${idx.map(i => i + 1).join(', ')}`;
}

/** Collapse ribbons whose curve would be identical, keeping who is in each.
 *
 *  Five ribbons with one material drawn five times is four invisible
 *  duplicates and a legend five times too long; the tag says which ribbons the
 *  single curve stands for, so nothing is hidden by the collapse. */
function groupRibbons(
  ribs: MaterialRibbon[],
  keyOf: (r: MaterialRibbon) => string,
): { idx: number[]; r: MaterialRibbon }[] {
  const out: { key: string; idx: number[]; r: MaterialRibbon }[] = [];
  ribs.forEach((r, i) => {
    const key = keyOf(r);
    const hit = out.find(o => o.key === key);
    if (hit) hit.idx.push(i);
    else out.push({ key, idx: [i], r });
  });
  return out;
}

function drawResult(
  obs: Observable,
  res: EfdResult,
  mainCanvas: HTMLCanvasElement,
  auxCanvas: HTMLCanvasElement,
): void {
  const auxBlock = $('efd-chart-aux-block');
  const def = OBSERVABLES[obs];
  lastDrawnObs = obs;
  setText('efd-chart-main-title', def.mainTitle);
  setText('efd-chart-aux-title', def.auxTitle);
  if (auxBlock) auxBlock.hidden = false;

  if (obs === 'eels' && eelsView === 'map') {
    const qs = (res.qs_nm_inv as number[]) ?? [0];
    drawHeatmap(mainCanvas, {
      xs: qs,
      ys: res.energies_eV as number[],
      z: res.loss_arb as (number | null)[][],
      xLabel: 'q  (nm⁻¹)',
      yLabel: 'Energy  (eV)',
      log: true,
    });
    setLegend('efd-chart-main-legend', null);
    setLegend('efd-chart-aux-legend', null);
    if (auxBlock) auxBlock.hidden = true;
    setText('efd-chart-main-title', 'Γ(q, ω) — loss per unit wave vector');
    // The headline stays the peak of the INTEGRATED spectrum: it is the same
    // number whichever face is on screen, and a map has no single peak energy.
    const integ = (res.loss_integrated as (number | null)[][] | undefined)?.[0];
    const Es = res.energies_eV as number[];
    if (integ) {
      let best = -Infinity;
      let at = Number.NaN;
      integ.forEach((v, i) => {
        if (v !== null && Number.isFinite(v) && v > best) {
          best = v;
          at = Es[i] ?? Number.NaN;
        }
      });
      setText('efd-out-headline', Number.isFinite(at) ? at.toFixed(4) : '—');
    }
    return;
  }

  if (obs === 'spectrum' || obs === 'eels' || obs === 'nonlinear') {
    const Es = res.energies_eV as number[];
    const series: Series[] = [];

    if (obs === 'nonlinear') {
      renderNlPick(res);
      // A null component is a gap, not a zero. Math.hypot(null, x) coerces to 0
      // and would draw a confident point where the solver reported none.
      const mag = (c: EfdComplex): (number | null)[] =>
        c.re.map((r, i) => {
          const im = c.im[i];
          return r === null || im === null || im === undefined ? null : Math.hypot(r, im);
        });
      // ALL: one modulus per order, each its own colour, so the three can be
      // told apart at a glance. ONE order: its real and imaginary parts, which
      // is what you look at once you have stopped comparing orders -- same
      // colour, solid and dashed, because they are two halves of one quantity
      // and colouring them differently would say they are two quantities. The
      // colour is the one that order carries in the ALL view, so switching
      // between the two does not relabel anything.
      NL_ORDERS.forEach((o, i) => {
        const a = res[o.field] as EfdComplex | undefined;
        if (!a) return;
        if (nlPick === 'all') {
          series.push({ label: o.label, ys: mag(a), color: seriesColor(i) });
        } else if (nlPick === o.key) {
          series.push({ label: `Re ${o.tab}`, ys: a.re, color: seriesColor(i) });
          series.push({ label: `Im ${o.tab}`, ys: a.im, color: seriesColor(i), dashed: true });
        }
      });
    } else {
      // The cross section view plots a real cross section: sigma per area,
      // (4 pi omega/c) Im alpha divided by the total width, which the library
      // computes because the prefactor and alpha's charge convention belong
      // together. EELS has no such thing -- it is a loss probability -- and
      // keeps the raw array.
      // Three different quantities share this branch, and each view has to
      // read its own: a cross section, a loss probability, and -- for a
      // payload older than either key -- the bare Im alpha. The EELS line is
      // the q INTEGRAL, one curve, not one curve per q: the map face is where
      // the wave-vector resolution is shown.
      const eelsLine = obs === 'eels'
        ? ((res.loss_integrated as (number | null)[][] | undefined)
            ?? (res.loss_arb as (number | null)[][] | undefined))
        : undefined;
      const xsRows = xsUnit === 'nm'
        ? (res.sigma_ext_nm as (number | null)[][] | undefined)
        : (res.sigma_ext_per_area as (number | null)[][] | undefined);
      const ext = eelsLine
        ?? xsRows
        ?? (res.extinction_arb as (number | null)[][]);
      const qs = eelsLine ? [0] : ((res.qs_nm_inv as number[]) ?? [0]);
      // Under an electron this is a LOSS spectrum, not a polarizability: the
      // same array, but labelling it Im α said the wrong thing about what the
      // electron measures.
      const name = obs === 'eels'
        ? 'Loss Γ(ω)'
        : xsUnit === 'nm'
          ? 'σ_ext  (nm)'
          : 'σ_ext / area';
      for (let i = 0; i < ext.length; i += 1) {
        series.push({
          label: qs.length > 1 ? `q = ${qs[i]?.toFixed(3)} nm⁻¹` : name,
          ys: row(ext, i),
        });
      }
    }

    const markers: { x: number; label: string }[] = [];
    const material = res.material as { pauli_edge_eV?: number } | undefined;
    if (material?.pauli_edge_eV && material.pauli_edge_eV <= (Es[Es.length - 1] ?? 0)) {
      markers.push({ x: material.pauli_edge_eV, label: '2|E_F|' });
    }

    const ax = energyAxis(Es, series, markers);
    drawSpectrum(mainCanvas, {
      xs: ax.xs,
      series: ax.series,
      xLabel: ax.xLabel,
      yLabel:
        obs === 'nonlinear'
          // alpha^(1,1) is a polarizability per unit length, nm^2. The harmonic
          // orders are not: alpha^(2,2) multiplies E and alpha^(3,3) multiplies
          // E^2, so they carry different dimensions and a shared axis cannot
          // name one unit for all three. It says so rather than inventing one.
          // And no modulus bars on a single order: that view draws Re and Im.
          ? (nlPick === 'all'
              ? '|α|  (mixed orders, arb.)'
              : nlPick === 'lin'
                ? 'α⁽¹ʼ¹⁾  (nm²)'
                : `${NL_ORDERS.find(o => o.key === nlPick)?.tab ?? 'α'}  (arb.)`)
          : obs === 'eels'
            ? 'Loss probability  (arb.)'
            : xsUnit === 'nm'
              ? 'σ_ext per unit length  (nm)'
              : 'σ_ext / area  (dimensionless)',
      // Re and Im go negative, and a log axis drops every non-positive
      // sample as unplottable -- half the curve would simply vanish. So the
      // single-order view is linear whatever the switch says, and the switch
      // is hidden there.
      logY: obs === 'nonlinear' && nlPick === 'all' && nlLog,
      markers: ax.markers,
    });
    setLegend('efd-chart-main-legend', series);

    // The conductivity used to be repeated here under every spectrum. It has
    // its own view now, which is where it belongs: one plot, one place.
    if (auxBlock) auxBlock.hidden = true;

    const first = series[0];
    if (first) {
      let best = -Infinity;
      let at = Number.NaN;
      first.ys.forEach((v, i) => {
        if (v !== null && Number.isFinite(v) && v > best) {
          best = v;
          at = Es[i] ?? Number.NaN;
        }
      });
      setText('efd-out-headline', Number.isFinite(at) ? at.toFixed(4) : '—');
    }
    return;
  }

  if (obs === 'material') {
    const Es = res.energies_eV as number[];

    // Every ribbon has its own E_F, gamma, beta, T and width, so the payload
    // carries one entry each. The fallback is for a payload from a wheel built
    // before `ribbons` existed: ribbon 1 alone, which is what it holds.
    const ribs: MaterialRibbon[] = (res.ribbons as MaterialRibbon[] | undefined) ?? [
      {
        index: 0,
        W_nm: (res.W_nm as number) ?? 0,
        EF_eV: 0,
        gamma_eV: 0,
        beta: 0,
        T_K: 0,
        material: res.material as MaterialRibbon['material'],
        eta_perp: res.eta_perp as EfdComplex,
        eta_par: res.eta_par as EfdComplex,
        sigma_perp_nm_fs: res.sigma_perp_nm_fs as EfdComplex | undefined,
        sigma_par_nm_fs: res.sigma_par_nm_fs as EfdComplex | undefined,
      },
    ];

    // sigma depends on E_F, gamma, beta and T; eta divides by W as well, so two
    // ribbons of the same material but different widths share a sigma curve and
    // not an eta curve.
    const sigGroups = groupRibbons(
      ribs,
      r => `${r.EF_eV}|${r.gamma_eV}|${r.beta}|${r.T_K}`,
    );
    const etaGroups = groupRibbons(
      ribs,
      r => `${r.EF_eV}|${r.gamma_eV}|${r.beta}|${r.T_K}|${r.W_nm}`,
    );
    const many = ribs.length > 1;
    const lastE = Es[Es.length - 1] ?? 0;

    // One Pauli edge per distinct |E_F|, and with several of them each marker
    // has to say whose it is.
    const markers: { x: number; label: string }[] = [];
    const seenEdge = new Set<string>();
    for (const g of sigGroups) {
      const edge = g.r.material.pauli_edge_eV;
      if (!edge || edge > lastE) continue;
      const key = edge.toFixed(6);
      if (seenEdge.has(key)) continue;
      seenEdge.add(key);
      markers.push({ x: edge, label: many ? `2|E_F| ${ribbonTag(g.idx)}` : '2|E_F|' });
    }

    const sigma: Series[] = [];
    for (const g of sigGroups) {
      const p = many ? `${ribbonTag(g.idx)} \u00b7 ` : '';
      sigma.push({ label: `${p}Re \u03c3`, ys: g.r.material.sigma_nm_fs.re });
      sigma.push({ label: `${p}Im \u03c3`, ys: g.r.material.sigma_nm_fs.im, dashed: true });
      // The Drude-only trace is the comparison that matters -- it is the sigma
      // that eta itself carries -- but it makes the legend half as long again,
      // so it rides along only while there is one material to compare against.
      if (sigGroups.length === 1 && g.r.material.sigma_drude_nm_fs) {
        sigma.push({
          label: `${p}Re \u03c3, Drude only`,
          ys: g.r.material.sigma_drude_nm_fs.re,
          dashed: true,
        });
      }
      // With drift the conductivity is a tensor, and the two components are not
      // interchangeable: the longitudinal one is what a plasmon condition wants.
      if (g.r.sigma_par_nm_fs) {
        sigma.push({ label: `${p}Re \u03c3\u2225 (longitudinal)`, ys: g.r.sigma_par_nm_fs.re });
      }
      if (g.r.sigma_perp_nm_fs) {
        sigma.push({
          label: `${p}Re \u03c3\u22a5 (transverse)`,
          ys: g.r.sigma_perp_nm_fs.re,
          dashed: true,
        });
      }
    }

    const axSigma = energyAxis(Es, sigma, markers);
    drawSpectrum(mainCanvas, {
      xs: axSigma.xs,
      series: axSigma.series,
      xLabel: axSigma.xLabel,
      yLabel: '\u03c3  (nm/fs)',
      markers: axSigma.markers,
    });
    setLegend('efd-chart-main-legend', sigma);

    const etaSeries: Series[] = [];
    for (const g of etaGroups) {
      const p = many ? `${ribbonTag(g.idx)} \u00b7 ` : '';
      etaSeries.push({ label: `${p}Re \u03b7\u2225`, ys: g.r.eta_par.re });
      etaSeries.push({ label: `${p}Im \u03b7\u2225`, ys: g.r.eta_par.im, dashed: true });
      if (g.r.sigma_par_nm_fs) {
        etaSeries.push({ label: `${p}Re \u03b7\u22a5`, ys: g.r.eta_perp.re });
      }
    }
    const axEta = energyAxis(Es, etaSeries, markers);
    drawSpectrum(auxCanvas, {
      xs: axEta.xs,
      series: axEta.series,
      xLabel: axEta.xLabel,
      yLabel: '\u03b7  (dimensionless)',
      markers: axEta.markers,
    });
    setLegend('efd-chart-aux-legend', etaSeries);

    const mu = res.mu_eV as number | undefined;
    setText('efd-out-headline', typeof mu === 'number' ? mu.toFixed(4) : '\u2014');
    return;
  }

  if (obs === 'nearfield') {
    const xs = res.xs_nm as number[];
    const zs = res.zs_nm as number[];
    drawNearFieldInto(mainCanvas, res, nfComponent, nfLog);
    setLegend('efd-chart-main-legend', null);
    setLegend('efd-chart-aux-legend', null);
    setHtml(
      'efd-chart-main-title',
      `${FIELD_LABEL[nfComponent]} of the induced field over (x, z) — arbitrary scale`,
    );
    // One map by default. The other components are a click away in the dialog,
    // which is where comparing them belongs.
    if (auxBlock) auxBlock.hidden = true;
    const map = res.map as (number | null)[][];
    let peak = 0;
    for (const col of map) {
      for (const v of col) if (v !== null && Number.isFinite(v) && v > peak) peak = v;
    }
    setText('efd-out-headline', peak > 0 ? peak.toPrecision(3) : '—');
    void xs;
    void zs;
    return;
  }

  if (obs === 'modes') {
    const eta = res.eta as EfdComplex;
    const drude = res.E_drude_eV as (number | null)[];
    const labels = eta.re.map((_, i) => `mode ${i + 1}`);
    // Two different quantities on one log axis, which the axis label and legend
    // have to be explicit about: eta_j is dimensionless and purely geometric,
    // the band energy is eV and assumes a single Drude ribbon.
    drawSpectrum(mainCanvas, {
      xs: eta.re.map((_, i) => i + 1),
      series: [
        { label: 'ħω_j  (eV, Drude closed form)', ys: drude },
        {
          label: '−η_j  (dimensionless, geometric)',
          ys: eta.re.map(v => (v === null ? null : -v)),
          dashed: true,
        },
      ],
      xLabel: 'Bright mode index',
      yLabel: 'see legend — two different units',
      logY: true,
    });
    setLegend('efd-chart-main-legend', [
      { label: 'ħω_j  (eV, Drude closed form)', ys: [] },
      { label: '−η_j  (dimensionless, geometric)', ys: [], dashed: true },
    ]);
    setLegend('efd-chart-aux-legend', labels.map(l => ({ label: l, ys: [] })));
    drawModeProfiles(auxCanvas, {
      xNodes: res.x_nodes_nm as number[][],
      profiles: res.profiles as number[][][],
      labels,
    });
    const eta1 = eta.re[0];
    setText('efd-out-headline', typeof eta1 === 'number' ? eta1.toFixed(6) : '—');
    return;
  }

  if (obs === 'dispersion') {
    const qs = res.qs_nm_inv as number[];
    const E = res.E_eV as (number | null)[][];
    // Pinned to the grid, not fitted to the bands, so this reads against the
    // LDOS map of the same structure over the same window.
    const eGrid = energyGrid();
    const nb = Math.max(...E.map(r => r.length));
    const series: Series[] = [];
    for (let b = 0; b < nb; b += 1) {
      series.push({ label: `band ${b + 1}`, ys: E.map(r => r[b] ?? null) });
    }
    drawSpectrum(mainCanvas, {
      xs: qs,
      series,
      xLabel: 'q  (nm⁻¹)',
      yLabel: 'Energy  (eV)',
      yRange: [eGrid.min, eGrid.max],
    });
    setLegend('efd-chart-main-legend', series);
    setLegend('efd-chart-aux-legend', null);
    // The extended-sheet map used to ride along under these bands. It is a
    // different object -- an unpatterned sheet, not the ribbons this page
    // solves -- so it is gone, and with it the second calculation that fed it.
    if (auxBlock) auxBlock.hidden = true;
    const last = E[E.length - 1];
    setText('efd-out-headline', last && last[0] != null ? last[0].toFixed(4) : '—');
    return;
  }

  // ldos
  const qs = res.qs_nm_inv as number[];
  const Es = res.energies_eV as number[];
  const map = res.map as (number | null)[][];
  const peakE = res.peak_E_eV as (number | null)[][];
  const tracks: { xs: number[]; ys: (number | null)[] }[] = [];
  const nPeaks = ldosShowPeaks ? Math.max(0, ...peakE.map(r => r.length)) : 0;
  for (let p = 0; p < nPeaks; p += 1) {
    tracks.push({ xs: qs, ys: peakE.map(r => r[p] ?? null) });
  }
  drawHeatmap(mainCanvas, {
    xs: qs,
    ys: Es,
    z: map,
    xLabel: 'q  (nm⁻¹)',
    yLabel: 'Energy  (eV)',
    log: true,
    // The peak tracks this map found, and -- when asked for -- the bands Level
    // 2 solves for independently. Agreement between them is the point: one is
    // read off a map, the other is a root-find, and neither knows about the
    // other.
    tracks: ldosBandTracks ? [...tracks, ...ldosBandTracks] : tracks,
  });
  // The map carries no legend of its own -- the colorbar is the legend -- so
  // these lines appear only when something has been laid over it, and name what
  // that something is. Two curve families can be on at once and they are two
  // different calculations, so each gets its own line. Under drift the second
  // of them is the sentence that keeps a beta = 0 reference from being read as
  // the drifting answer, which is the whole reason the overlay is allowed there.
  const caption: Series[] = [];
  if (nPeaks > 0) {
    caption.push({
      label: 'LDOS peak tracks — the ridges this map resolves in itself, followed '
        + 'across q, so a curve stays on one branch until that branch ends. A break '
        + 'is a branch ending, not a gap in the data.',
      ys: [],
      // The lines themselves are white, which no legend swatch can be: this one
      // sits on the page, not on the colormap. The body colour reads in both
      // themes and its only job is to not be the other family's magenta.
      color: 'var(--text-soft)',
    });
  }
  if (ldosBandTracks && ldosBandNote) {
    caption.push({ label: ldosBandNote, ys: [], color: BAND_OVERLAY_COLOR });
  }
  setLegend('efd-chart-main-legend', caption.length ? caption : null);
  if (caption.length) $('efd-chart-main-legend')?.classList.add('chart-legend--note');
  setLegend('efd-chart-aux-legend', null);

  const asym = res.asymmetry as (number | null)[][] | undefined;
  const asymQ = res.asymmetry_q_nm_inv as number[] | undefined;
  if (asym && asymQ) {
    drawDivergingHeatmap(auxCanvas, {
      xs: asymQ,
      ys: Es,
      z: asym,
      xLabel: '|q|  (nm⁻¹)',
      yLabel: 'Energy  (eV)',
    });
  } else if (auxBlock) {
    auxBlock.hidden = true;
  }

  let best = -Infinity;
  let at = Number.NaN;
  const weights = res.peak_weight as (number | null)[][];
  weights.forEach((r, i) => {
    r.forEach((w, p) => {
      if (w !== null && Number.isFinite(w) && w > best) {
        best = w;
        at = peakE[i]?.[p] ?? Number.NaN;
      }
    });
  });
  setText('efd-out-headline', Number.isFinite(at) ? at.toFixed(4) : '—');
}

// --------------------------------------------------------------------------
// Controller
// --------------------------------------------------------------------------

export function initElectrostaticsFd(): void {
  const runBtnEl = $<HTMLButtonElement>('efd-run');
  const schematicEl = $<HTMLCanvasElement>('efd-schematic');
  const mainEl = $<HTMLCanvasElement>('efd-chart-main');
  const auxEl = $<HTMLCanvasElement>('efd-chart-aux');
  if (!runBtnEl || !schematicEl || !mainEl || !auxEl) return;
  // Re-bound with non-nullable declared types. The guard above narrows these,
  // but `run()` below is a hoisted function declaration, so TypeScript will not
  // carry the narrowing into it — it cannot prove the guard ran first.
  const runBtn: HTMLButtonElement = runBtnEl;
  const schematic: HTMLCanvasElement = schematicEl;
  const mainCanvas: HTMLCanvasElement = mainEl;
  const auxCanvas: HTMLCanvasElement = auxEl;

  const bridge = new EfdBridge();
  let lastResult: EfdResult | null = null;

  /** Every result this spec has produced, one per view.
   *
   *  A Run computes the view you are on and then, quietly, the other linear
   *  views, so moving between them is navigation rather than another wait.
   *  Each entry remembers the spec it came from: change a parameter and the
   *  keys stop matching, which is what turns the notice back on and keeps a
   *  stale curve from being presented as current. */
  const resultCache = new Map<Observable, { key: string; res: EfdResult }>();
  const specKeyFor = (o: Observable): string => `${o}|${JSON.stringify(readSpec(o))}`;

  /** The cached result for `o`, but only if it is still the current spec's. */
  const cachedFor = (o: Observable): EfdResult | null => {
    const hit = resultCache.get(o);
    return hit && hit.key === specKeyFor(o) ? hit.res : null;
  };

  const setNeedsRun = (on: boolean): void => {
    const el = $('efd-needs-run');
    if (el) el.hidden = !on;
  };

  /** The views one Run gets ready. Not the near field, the LDOS or EELS: those
   *  are a different drive or a map, and computing them uninvited would turn
   *  one click into a minute. Not the nonlinear spectrum either -- it is a
   *  cascade of its own and costs more than the four together. */
  const LINEAR_FAMILY: readonly Observable[] = ['spectrum', 'modes', 'dispersion', 'material'];

  /** How long a view may cost before it is left for an explicit Run. */
  const PREFETCH_BUDGET_S = 10;
  /** The spec that produced what is currently drawn, or null before any run. */
  let lastRunKey: string | null = null;
  const specKey = (): string => `${observable}|${JSON.stringify(readSpec(observable))}`;
  let lastGeometry: GeometryPayload | null = null;
  let lastSource: SchematicSource | undefined;
  let busy = false;

  // ---- visibility ---------------------------------------------------------

  function applyVisibility(): void {
    const n = Math.min(MAX_RIBBONS, Math.max(1, int('efd-nribbons', 1)));
    document.querySelectorAll<HTMLElement>('.efd-ribbon-group').forEach(g => {
      g.hidden = Number(g.dataset.ribbon ?? '0') >= n;
    });

    // Cheap, and both have to follow other inputs, so they ride on the one
    // function every control change already calls.
    updateNodeSpacing();
    syncNfEnergy();
    syncElectronSpeed();

    // With a stack in force the medium below IS the stack, so the eps-below box
    // would be a second, conflicting description of it (see readSpec).
    const stacked = num('efd-sub-eps', 0) > 0;
    const stackBlock = $('efd-stack');
    if (stackBlock) stackBlock.hidden = !stacked;
    const epsBField = $('efd-eps-b-field');
    if (epsBField) epsBField.hidden = stacked;
    // Unconditionally: a row left behind while the stack was hidden would
    // reappear contradicting the count beside it, which is what Reset did.
    renderLayerRows();

    // The axis toggle belongs only to the views that put energy on x.
    const axisOverlay = $('efd-axis-overlay');
    // ... and not on the EELS map either, whose horizontal axis is q.
    const mapFace = observable === 'eels' && eelsView === 'map';
    if (axisOverlay) axisOverlay.hidden = !ENERGY_X.has(observable) || mapFace;

    // The nonlinear menu is a control, not a readout of the last run, so it
    // has to be populated the moment the view is on screen -- and again when
    // the orders change, since that changes what it may offer.
    if (observable === 'nonlinear') renderNlPick(cachedFor('nonlinear'));
    // The scale switch belongs to the moduli only, so its visibility turns on
    // the PICK as well as the observable -- which `.efd-obs-only` alone cannot
    // express.
    const nlScale = $('efd-nl-scale-wrap');
    if (nlScale) nlScale.hidden = !(observable === 'nonlinear' && nlPick === 'all');

    // The near-field map is at ONE energy, set beside the map. A range and a
    // point count are not parameters of it.
    const needsGrid = observable !== 'nearfield';
    document.querySelectorAll<HTMLElement>('.efd-e-grid').forEach(el => {
      el.hidden = !needsGrid;
    });

    for (const exc of ['plane', 'dipole', 'electron'] as Excitation[]) {
      document.querySelectorAll<HTMLElement>(`.efd-${exc}-only`).forEach(el => {
        el.hidden = exc !== excitation;
      });
    }

    const grid = OBSERVABLES[observable].qGrid;
    document.querySelectorAll<HTMLElement>('.efd-q-grid').forEach(el => { el.hidden = !grid; });
    document.querySelectorAll<HTMLElement>('.efd-q-scalar').forEach(el => { el.hidden = grid; });

    document.querySelectorAll<HTMLElement>('.efd-obs-only').forEach(el => {
      el.hidden = el.dataset.obsFor !== observable;
    });

    // What the overlay will be, said before the click rather than after it.
    // With drift the bands are solved at beta = 0 -- see the tick handler --
    // so the hover has to say which of the two you are about to get.
    const bandsBox = $<HTMLInputElement>('efd-ldos-bands');
    // Costs a readSpec, so only while there is an overlay to invalidate.
    if (ldosBandTracks && bandsBox) {
      const now = `dispersion|${JSON.stringify(withoutDrift(readSpec('dispersion')).spec)}`;
      if (now !== ldosBandKey) {
        ldosBandTracks = null;
        ldosBandNote = null;
        ldosBandKey = null;
        bandsBox.checked = false;
        redrawCurrent();
      }
    }
    const bandsLabel = bandsBox?.closest<HTMLElement>('label');
    if (bandsLabel) {
      bandsLabel.title = driftingNow()
        ? 'Lay the Level 2 band energies over the map. Drift makes eta a tensor '
          + '(eta_perp ≠ eta_par), so a drifting band structure is not a Level 2 '
          + 'spectrum at all — these are solved at β = 0 and drawn as the '
          + 'no-drift reference, labelled as such. What drift did is the gap '
          + 'between them and the ridges beneath.'
        : 'Lay the Level 2 band energies over the map. They are a separate '
          + 'calculation that knows nothing about it, so agreement is a check.';
    }

    // An observable that needs a different excitation is removed, not greyed
    // out: offering "LDOS" under a plane wave would be offering something that
    // cannot exist.
    document.querySelectorAll<HTMLButtonElement>('#efd-observable-tabs .plot-tab').forEach(b => {
      const obs = b.dataset.obs as Observable | undefined;
      if (!obs) return;
      b.hidden = !availableHere(obs);
      const active = obs === observable;
      b.classList.toggle('active', active);
      b.setAttribute('aria-checked', active ? 'true' : 'false');
    });

    // The Source heading has nothing under it for a structure observable.
    const srcTitle = document.querySelector<HTMLElement>('.efd-src-title');
    if (srcTitle) srcTitle.hidden = observable === 'modes' || observable === 'dispersion';
  }

  /** Is what `lastResult` holds drawable as `o`?
   *
   *  THE one condition. Three places re-draw without recomputing -- a view
   *  switch, the axis-unit toggle and a theme change -- and all three have to
   *  agree, because drawResult's branches read payload-specific keys through
   *  unchecked casts. Handing a cross-section payload to the material branch
   *  is not a blank chart: it throws at `g.r.eta_par.re`, AFTER drawResult has
   *  already reopened the aux block and set lastDrawnObs, which wedges the view
   *  until the next successful run. Two of the three used to skip the check. */
  const resultIsDrawableAs = (o: Observable): boolean =>
    !!lastResult && lastResult.ok === true && lastDrawnObs === o;

  /** Re-draw the current view from the payload already in hand, or do nothing.
   *  For the callers that must never blank the chart as a side effect. */
  const redrawCurrent = (): void => {
    if (resultIsDrawableAs(observable)) {
      drawResult(observable, lastResult as EfdResult, mainCanvas, auxCanvas);
    }
  };

  /** Put on screen whatever belongs to `o` -- which is often nothing.
   *
   *  Switching views does not recompute. Leaving the previous view's curves
   *  under the new view's title invites reading them as the answer to a
   *  question they never answered, and it is why the aux panel sat open and
   *  captioned "Sheet conductivity" over an empty canvas on first load. */
  function showResultFor(o: Observable): void {
    // What a Run left ready for this view, if the settings have not moved
    // since. This is what makes going back and forth free.
    const cached = cachedFor(o);
    if (cached) {
      lastResult = cached;
      drawResult(o, cached, mainCanvas, auxCanvas);
      showNotes(cached.notes as string[] | undefined);
      lastRunKey = specKeyFor(o);
      setStale(false);
      setNeedsRun(false);
      return;
    }
    if (resultIsDrawableAs(o)) {
      drawResult(o, lastResult as EfdResult, mainCanvas, auxCanvas);
      setNeedsRun(false);
      return;
    }
    blankCanvas(mainCanvas);
    blankCanvas(auxCanvas);
    setLegend('efd-chart-main-legend', null);
    setLegend('efd-chart-aux-legend', null);
    const auxBlock = $('efd-chart-aux-block');
    if (auxBlock) auxBlock.hidden = true;
    showNotes(undefined);
    setText('efd-out-headline', '\u2014');
    setNeedsRun(true);
  }

  function selectObservable(o: Observable): void {
    observable = o;
    // Mirror into the hidden field the permalink tracks. Assigning `.value`
    // fires nothing, so the event is explicit -- that is what keeps the URL in
    // the address bar current as you move between panels.
    setMirror('efd-view', o);
    setText('efd-chart-main-title', OBSERVABLES[o].mainTitle);
    setText('efd-chart-aux-title', OBSERVABLES[o].auxTitle);
    setText('efd-out-headline-unit', OBSERVABLES[o].headlineUnit);
    applyVisibility();
    showResultFor(o);
    showError(null);
    void refreshLive();
  }

  function selectExcitation(e: Excitation): void {
    excitation = e;
    setMirror('efd-src', e);
    document.querySelectorAll<HTMLButtonElement>('#efd-excitation-tabs .plot-tab').forEach(b => {
      const active = b.dataset.excitation === e;
      b.classList.toggle('active', active);
      b.setAttribute('aria-checked', active ? 'true' : 'false');
    });
    // Keep a structure observable across a change of drive — switching from a
    // plane wave to a dipole while reading the bands should not throw you back
    // to a cross section — but a drive-specific one has to move.
    if (!availableHere(observable)) {
      selectObservable(DEFAULT_OBSERVABLE[e]);
      return;
    }
    applyVisibility();
    showError(null);
    void refreshLive();
  }

  /** Write a value into a hidden permalink mirror and announce it.
   *
   *  Assigning `.value` dispatches nothing, so the event is explicit: without
   *  it the permalink's own listener never hears the change and the address
   *  bar goes stale. The restore listeners below return early when the value
   *  already matches, so the announcement cannot bounce back into this call. */
  function setMirror(id: string, value: string): void {
    const el = $<HTMLInputElement>(id);
    if (!el || el.value === value) return;
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  // The other direction: a shared link writes these fields and dispatches
  // `input`, which is the only way a pair of buttons can be restored from a
  // URL. Guarded against the echo of our own setMirror above.
  $<HTMLInputElement>('efd-src')?.addEventListener('input', ev => {
    const e = (ev.target as HTMLInputElement).value as Excitation;
    if (e && e !== excitation && e in DEFAULT_OBSERVABLE) selectExcitation(e);
  });
  $<HTMLInputElement>('efd-view')?.addEventListener('input', ev => {
    const o = (ev.target as HTMLInputElement).value as Observable;
    if (o && o !== observable && o in OBSERVABLES && availableHere(o)) selectObservable(o);
  });

  document.querySelectorAll<HTMLButtonElement>('#efd-excitation-tabs .plot-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const e = btn.dataset.excitation as Excitation | undefined;
      if (e) selectExcitation(e);
    });
  });

  document.querySelectorAll<HTMLButtonElement>('#efd-observable-tabs .plot-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const o = btn.dataset.obs as Observable | undefined;
      if (o && availableHere(o)) selectObservable(o);
    });
  });

  // ---- the cheap, live half ----------------------------------------------

  let liveTimer: number | undefined;

  async function refreshLive(): Promise<void> {
    if (!bridge.ready) return;
    const spec = readSpec(observable);

    const geo = await bridge.run('geometry', spec).catch(() => null);
    if (geo && geo.ok) {
      lastGeometry = geo as unknown as GeometryPayload;
      const src = spec.source as
        | { kind?: string; r0_nm?: number[]; p?: number[]; b_nm?: number }
        | undefined;
      lastSource = src?.kind
        ? {
            kind: src.kind as SchematicSource['kind'],
            p: src.p as [number, number, number] | undefined,
            r0_nm: src.r0_nm as [number, number, number] | undefined,
            b_nm: src.b_nm,
          }
        : { kind: 'plane_wave' };
      drawRibbonSchematic(schematic, lastGeometry, lastSource);
      renderRibbonTable(lastGeometry, geo.eps_eff as number | undefined);
    } else if (geo && !geo.ok) {
      setText('efd-table-caption', 'Ribbons');
      showError(geo.reason ?? 'The geometry is not usable.');
      return;
    }

    const cost = await bridge.costEstimate(spec, OBSERVABLES[observable].kind).catch(() => null);
    if (cost && cost.ok) {
      // With a substrate the library reports a floor rather than an estimate:
      // the cost model counts solves and cells, and a screened rebuild is a
      // quadrature it does not count. Showing a bare number there would promise
      // something measured 4x to 16x optimistic, so show the direction instead.
      const floor = cost.estimate_is_floor === true;
      setText(
        'efd-out-cost',
        (floor ? '\u2265 ' : '') + fmtSeconds(cost.estimated_seconds),
      );
      setText('efd-out-route', cost.fast_path ? 'eigen fast path' : 'full solve per point');
      // Short, because the button now sits inline beside the heading. The
      // refusal still needs its reason, which moves to the tooltip: the
      // estimate readout beside the geometry carries the number itself.
      runBtn.textContent = cost.refuse
        ? 'Too large'
        : cost.warn
          ? `Run (${floor ? '≥' : '~'}${fmtSeconds(cost.estimated_seconds)})`
          : 'Run';
      runBtn.title = cost.refuse
        ? 'This would take about ' +
          fmtSeconds(cost.estimated_seconds) +
          ' — coarsen the energy or q grid, use fewer nodes, or drop the substrate.'
        : 'Run the calculation';
      runBtn.disabled = busy || cost.refuse;

      // "What is drawn is not what is dialled in." The label cannot carry this
      // -- refreshLive rewrites it on every keystroke -- so it rides on a class.
      // Nothing is stale before the first run, and a refused spec is not stale
      // either, since there is nothing to re-run.
      setStale(!cost.refuse && lastRunKey !== null && lastRunKey !== specKey());

      // Warn before the click, not after. The cost estimate does not check q,
      // so these would only fail on Run — and both reasons are real ones.
      const scalarQ = !OBSERVABLES[observable].qGrid;
      const qZero = scalarQ && num('efd-q', 0) === 0;
      if (num('efd-sub-eps', 0) > 0 && qZero) {
        showError(
          'A substrate at q = 0 will be refused: the reflected Coulomb integral is ' +
            'logarithmically divergent there. Use a small finite q — the physics is ' +
            'smooth in q, so qW of order 0.01 already gives the q → 0 answer.',
        );
      } else if (excitation === 'dipole' && qZero) {
        showError(
          'A point dipole has no q = 0 Fourier component, so this needs a finite q. ' +
            'The plane wave is exactly the q = 0 source if that is what you want.',
        );
      } else {
        showError(
          cost.fast_path_obstruction && !cost.fast_path
            ? `No fast path: ${cost.fast_path_obstruction}`
            : null,
        );
      }
    }
  }

  function scheduleLive(): void {
    // PresetStrip applies selects immediately and inputs on the next animation
    // frame, so a debounce shorter than a frame would read a half-applied preset.
    window.clearTimeout(liveTimer);
    liveTimer = window.setTimeout(() => void refreshLive(), 250);
  }

  /** Render the finite layers between the sheet and the substrate.
   *
   *  The rows are injected, so they carry no Astro scope attribute and their
   *  classes exist as `:global()` in the page. They are also invisible to the
   *  one-shot `.controls input` wiring, which is why the container below gets
   *  its own delegated listener. Existing values survive a count change:
   *  stepping 2 -> 3 must not reset the two layers already dialled in. */
  function renderLayerRows(): void {
    const host = $('efd-layer-editor');
    if (!host) return;
    const want = Math.max(0, Math.min(8, int('efd-nlayers', 0)));
    const have = host.querySelectorAll('.efd-layer-row').length;
    if (want === have) return;

    const kept: { d: string; par: string; perp: string }[] = [];
    host.querySelectorAll<HTMLElement>('.efd-layer-row').forEach(row => {
      const v = (k: string) =>
        row.querySelector<HTMLInputElement>(`[data-key="${k}"]`)?.value ?? '';
      kept.push({ d: v('d_nm'), par: v('eps_par'), perp: v('eps_perp') });
    });

    const rows: string[] = [];
    for (let i = 0; i < want; i += 1) {
      const k = kept[i] ?? { d: '10', par: '4', perp: '4' };
      // `value=` is also `defaultValue`, which the permalink compares against
      // to decide whether a field is worth putting in a link. Re-rendering a
      // row with the kept numbers would make every layer look untouched, so
      // the attribute stays at the factory setting and the kept number is
      // assigned as a property below.
      rows.push(
        `<div class="efd-layer-row" data-efd-layer="${i}">` +
          `<span class="efd-layer-chip">Layer ${i + 1}</span>` +
          '<div class="efd-layer-fields">' +
          `<label><span>d (nm)</span><input type="number" class="param-number" id="efd-lay-${i}-d" data-key="d_nm" value="${k.d}" step="1" min="0.1"></label>` +
          `<label><span>&epsilon;&#8741;</span><input type="number" class="param-number" id="efd-lay-${i}-par" data-key="eps_par" value="${k.par}" step="0.1" min="1"></label>` +
          `<label><span>&epsilon;&#8869;</span><input type="number" class="param-number" id="efd-lay-${i}-perp" data-key="eps_perp" value="${k.perp}" step="0.1" min="1"></label>` +
          '</div></div>',
      );
    }
    host.innerHTML = rows.join('');
    // Assigned, not written into the markup: see the note above.
    host.querySelectorAll<HTMLElement>('.efd-layer-row').forEach((row, i) => {
      const k = kept[i];
      if (!k) return;
      const put = (key: string, v: string) => {
        const el = row.querySelector<HTMLInputElement>(`[data-key="${key}"]`);
        if (el && v !== '') el.value = v;
      };
      put('d_nm', k.d);
      put('eps_par', k.par);
      put('eps_perp', k.perp);
    });
  }

  // Injected inputs never saw the one-shot wiring below, so the container
  // listens for them. `change`, not `input`: a keystroke-by-keystroke re-cost
  // of a screened stack is exactly the thing worth not doing.
  $('efd-layer-editor')?.addEventListener('change', () => scheduleLive());

  // ---- the expensive half, on demand --------------------------------------

  /** The house loading rim: a spectrum bar that fills across the card's top
   *  edge. It animates on the compositor, so it keeps moving even while the
   *  worker is busy — which is exactly when it is needed. */
  const resultCard = mainCanvas.closest('.chart-panel');
  const setComputing = (on: boolean) => resultCard?.classList.toggle('is-computing', on);
  const setStale = (on: boolean) => runBtn.classList.toggle('is-stale', on);

  async function run(): Promise<void> {
    if (busy || !bridge.ready) return;
    busy = true;
    runBtn.disabled = true;
    setComputing(true);
    const label = runBtn.textContent;
    runBtn.textContent = 'Solving…';
    showError(null);

    try {
      const spec = readSpec(observable);
      const key = `${observable}|${JSON.stringify(spec)}`;
      const res = await bridge.run(OBSERVABLES[observable].kind, spec);
      if (!res.ok) {
        // One case is worth rescuing rather than reporting: Level 2 bands cannot
        // be computed with drift, because eta becomes a tensor and a single
        // eigenvalue spectrum has no resonance condition to invert. Level 1 — the
        // extended sheet — has no such problem and is closed-form, and it is the
        // rung that isolates the drift conductivity from everything else.
        if (observable === 'dispersion') {
          const sheet = await bridge.run('sheet_dispersion', spec).catch(() => null);
          if (sheet && sheet.ok) {
            lastResult = sheet;
            setText('efd-chart-main-title', 'Extended sheet: Im r(Q, ω) — Level 1');
            drawHeatmap(mainCanvas, {
              xs: sheet.Qs_nm_inv as number[],
              ys: sheet.energies_eV as number[],
              z: sheet.im_r as (number | null)[][],
              xLabel: 'Q  (nm⁻¹)',
              yLabel: 'Energy  (eV)',
              log: true,
            });
            const asym = sheet.asymmetry as (number | null)[][] | undefined;
            const asymQ = sheet.asymmetry_Q_nm_inv as number[] | undefined;
            const auxBlock = $('efd-chart-aux-block');
            if (asym && asymQ) {
              setText('efd-chart-aux-title', '±Q asymmetry — symmetric scale, zero is neutral');
              drawDivergingHeatmap(auxCanvas, {
                xs: asymQ,
                ys: sheet.energies_eV as number[],
                z: asym,
                xLabel: '|Q|  (nm⁻¹)',
                yLabel: 'Energy  (eV)',
              });
              if (auxBlock) auxBlock.hidden = false;
            } else if (auxBlock) {
              auxBlock.hidden = true;
            }
            showNotes([
              `Ribbon bands are unavailable here: ${res.reason ?? 'the solver refused them.'}`,
              ...(sheet.notes ?? []),
            ]);
            showError(null);
            // This map was drawn by hand from a `sheet_dispersion` payload, not
            // through drawResult, and drawResult('dispersion', sheet) would
            // throw -- the sheet carries Qs_nm_inv/im_r, not E_eV. So nothing
            // may re-draw it: null says "no cached redraw is valid", which is
            // honest, where naming an observable would invite exactly the throw
            // the guard above exists to stop. The run itself did succeed, so
            // the stale flag has to be cleared or the button would claim this
            // result was never computed.
            lastDrawnObs = null;
            lastRunKey = key;
            setStale(false);
            return;
          }
        }
        showNotes(undefined);
        showError(res.reason ?? 'The solver refused this configuration.');
        // The canvases still hold the last successful run. Leaving them
        // unlabelled beside an error invites reading them as the answer to the
        // settings now on screen, which they are not.
        setText('efd-chart-main-title', 'Previous result — the settings now set were refused');
        lastResult = res;
        return;
      }

      lastResult = res;
      resultCache.set(observable, { key, res });
      drawResult(observable, res, mainCanvas, auxCanvas);
      setNeedsRun(false);
      // What is on screen now came from THIS spec. Anything else dialled in
      // afterwards makes the drawing stale, which refreshLive then shows.
      lastRunKey = key;
      setStale(false);
      showNotes(res.notes as string[] | undefined);
    } catch (err) {
      showError(`The solver did not answer: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      busy = false;
      setComputing(false);
      runBtn.disabled = false;
      runBtn.textContent = label ?? 'Run';
    }
  }

  /** After the view you asked for, get the other linear views ready.
   *
   *  Each is priced first and skipped if it would cost real time, so one click
   *  cannot quietly become a minute; a view that was skipped simply asks for
   *  its own Run, which is what it did before any of this existed. Anything
   *  that fails is dropped rather than reported: this was never requested, so
   *  it has no business raising an error over the result you did ask for. */
  async function prefetchFamily(): Promise<void> {
    for (const o of LINEAR_FAMILY) {
      if (busy) return; // a new Run took over; it will prefetch in its turn
      if (o === observable || !availableHere(o) || cachedFor(o)) continue;
      const spec = readSpec(o);
      const key = `${o}|${JSON.stringify(spec)}`;
      const cost = await bridge.costEstimate(spec, OBSERVABLES[o].kind).catch(() => null);
      if (!cost || !cost.ok || cost.refuse) continue;
      if (Number(cost.estimated_seconds) > PREFETCH_BUDGET_S) continue;
      if (busy) return;
      const res = await bridge.run(OBSERVABLES[o].kind, spec).catch(() => null);
      if (res && res.ok) resultCache.set(o, { key, res });
    }
  }

  runBtn.addEventListener('click', () => {
    void run().then(() => prefetchFamily());
  });

  // ---- control wiring -----------------------------------------------------

  // Number/slider pairs, the house pattern: the slider drives the number and
  // vice versa, and either one triggers a refresh.
  for (const base of SLIDER_PAIRS) {
    const n = $<HTMLInputElement>(base);
    const s = $<HTMLInputElement>(`${base}-slider`);
    if (!n || !s) continue;
    s.addEventListener('input', () => {
      n.value = s.value;
      scheduleLive();
    });
    // initPermalinks listens for `input` ON THE NUMBER BOX, and assigning
    // `.value` in script fires nothing -- so a value only ever reached by
    // dragging was never in the shared URL. One synthetic event when the drag
    // ends is enough; during the drag `scheduleLive` already keeps the cost
    // estimate and the stale flag in step.
    s.addEventListener('change', () => {
      n.value = s.value;
      n.dispatchEvent(new Event('input', { bubbles: true }));
    });
    n.addEventListener('input', () => {
      s.value = n.value;
      scheduleLive();
    });
  }

  // The near-field energy moved into the chart toolbar, which is outside
  // `.controls` -- without it in this selector the map's own energy would be
  // the one control that never re-costed or re-flagged the Run button.
  //
  // `.efd-mirror` is deliberately NOT in this selector. A mirror is a second
  // face of a parameter -- the wavelength beside the map's energy, the beam
  // energy beside v/c -- and the spec reads only the parameter. Wiring a
  // mirror here as well makes the two fight: this listener runs first, the
  // sync it triggers rewrites the mirror from the value that is about to
  // change, and the mirror's own handler then reads back what was already
  // there. Typing 800 nm came out as the energy it already had. Each mirror
  // has its own handler that converts and dispatches on the real parameter,
  // so everything downstream still happens, exactly once.
  document.querySelectorAll<HTMLElement>(
    '.controls input:not(.efd-mirror), .controls select, ' +
      '.efd-nf-energy-box input:not(.efd-mirror)',
  ).forEach(el => {
    const onChange = () => {
      applyVisibility();
      scheduleLive();
    };
    el.addEventListener('input', onChange);
    el.addEventListener('change', onChange);
  });

  // A preset carries "<excitation>:<observable>" — the scene its figure shows.
  const hint = $<HTMLInputElement>('efd-scene-hint');
  if (hint) {
    hint.addEventListener('input', () => {
      // Every preset writes its range in eV, so the block has to be in eV to
      // read them -- without converting, since the numbers are already right.
      setXUnit('energy', false);
      const [e, o] = hint.value.split(':') as [Excitation, Observable];
      if (e && e !== excitation) selectExcitation(e);
      if (o && OBSERVABLES[o] && availableHere(o) && o !== observable) selectObservable(o);
    });
  }

  $<HTMLButtonElement>('efd-reset')?.addEventListener('click', () => {
    // Re-reading the document's own defaults is more honest than keeping a
    // second copy of them in here that can drift out of step.
    document
      .querySelectorAll<HTMLInputElement>('.controls input, .efd-nf-energy-box input')
      .forEach(el => {
        if (el.defaultValue !== '') el.value = el.defaultValue;
      });
    document.querySelectorAll<HTMLSelectElement>('.controls select').forEach(el => {
      const def = [...el.options].find(o => o.defaultSelected) ?? el.options[0];
      if (def) el.value = def.value;
    });
    for (const base of SLIDER_PAIRS) {
      const n = $<HTMLInputElement>(base);
      const s = $<HTMLInputElement>(`${base}-slider`);
      if (n && s) s.value = n.value;
    }
    selectExcitation('plane');
    showNotes(undefined);
    showError(null);
  });

  // ---- the near-field view: component, scale, and all of them at once -----

  const redrawNearField = (): void => {
    if (observable !== 'nearfield' || !lastResult || !lastResult.ok) return;
    drawNearFieldInto(mainCanvas, lastResult, nfComponent, nfLog);
    setHtml(
      'efd-chart-main-title',
      `${FIELD_LABEL[nfComponent]} of the induced field over (x, z) — arbitrary scale`,
    );
  };

  $<HTMLSelectElement>('efd-nf-comp')?.addEventListener('change', ev => {
    nfComponent = (ev.target as HTMLSelectElement).value as FieldKey;
    // A scale toggle only means anything for the positive magnitude; the signed
    // components are always on the diverging scale, symmetric about zero.
    const scale = $('efd-nf-scale');
    if (scale) scale.hidden = nfComponent !== 'map';
    redrawNearField();
  });

  /** Put the whole range block into `u` -- heading, both numbers, their step
   *  and bounds, the pills, and the drawn axis.
   *
   *  `convert` is false when the caller has already written the numbers in the
   *  target unit: a permalink restore, or a preset, which always writes eV.
   *  Nothing re-solves; `energyGrid()` hands the library the same eV either
   *  way, so the Run button must not go stale over a change of units. */
  function setXUnit(u: XUnit, convert = true): void {
    const lo = $<HTMLInputElement>('efd-e-min');
    const hi = $<HTMLInputElement>('efd-e-max');
    const nm = u === 'lambda';
    let converted = false;

    if (convert && u !== xUnit && lo && hi) {
      const g = energyGrid();
      if (nm) {
        // The ends swap: the smallest wavelength is the largest energy.
        const nmMin = fmtNm(HC_EV_NM / g.max);
        const nmMax = fmtNm(HC_EV_NM / g.min);
        evRange = { min: g.min, max: g.max, nmMin, nmMax };
        lo.value = nmMin;
        hi.value = nmMax;
      } else {
        lo.value = fmtEv(g.min);
        hi.value = fmtEv(g.max);
        evRange = null;
      }
      converted = true;
    }
    // Before the dispatch below: the handlers it wakes call energyGrid(), which
    // reads this to decide what the two boxes mean.
    xUnit = u;
    if (converted && lo && hi) {
      // Assigning `.value` fires nothing, and the permalink listens for
      // `input` on these two boxes -- without this the shared URL kept the
      // numbers from before the conversion while `u` said the other unit, so
      // the link restored 600-2000 nm as eV.
      for (const el of [lo, hi]) el.dispatchEvent(new Event('input', { bubbles: true }));
    }

    for (const [el, end] of [[lo, 'minimum'], [hi, 'maximum']] as const) {
      if (!el) continue;
      el.step = nm ? '10' : '0.05';
      el.min = nm ? '1' : '0.001';
      el.setAttribute(
        'aria-label',
        `${nm ? 'Wavelength' : 'Energy'} range ${end} (${nm ? 'nm' : 'eV'})`,
      );
    }
    setText('efd-range-label', nm ? 'Wavelength (nm)' : 'Energy (eV)');
    document.querySelectorAll<HTMLButtonElement>('#efd-axis-tabs .plot-tab').forEach(b => {
      const active = (b.dataset.axis === 'lambda') === nm;
      b.classList.toggle('active', active);
      b.setAttribute('aria-checked', active ? 'true' : 'false');
    });

    // So a shared link comes back in the unit it was shared in. Re-entrant by
    // one pass: the listener below calls setXUnit(u, false), which finds the
    // value already set and dispatches nothing.
    const store = $<HTMLInputElement>('efd-xunit');
    if (store && store.value !== u) {
      store.value = u;
      store.dispatchEvent(new Event('input', { bubbles: true }));
    }
    syncNfEnergy();
  }

  document.querySelectorAll<HTMLButtonElement>('#efd-axis-tabs .plot-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      setXUnit(btn.dataset.axis === 'lambda' ? 'lambda' : 'energy');
      redrawCurrent();
      void refreshLive();
    });
  });

  /* v/c <-> K, the same two-faced pair as the near field's energy and
     wavelength, and the same rule: only ONE of them is the parameter. v/c is
     what `sourceSpec` sends, so the keV box is excluded from the generic
     wiring and converts through here, while syncElectronSpeed writes it back
     from v/c. Both use the house relativistic pair in lib/transition-radiation
     rather than a second copy of gamma. */
  $<HTMLInputElement>('efd-kev')?.addEventListener('input', ev => {
    const keV = Number((ev.target as HTMLInputElement).value);
    const vc = $<HTMLInputElement>('efd-vc');
    if (!vc || !(keV > 0)) return;
    const beta = betaFromKeV(keV);
    if (!Number.isFinite(beta) || beta <= 0 || beta >= 1) return;
    vc.value = String(Number(beta.toPrecision(5)));
    vc.dispatchEvent(new Event('input', { bubbles: true }));
  });

  /* lambda -> E, the other direction. Setting `.value` in script fires
     nothing, so writing the energy box here cannot loop back; the synthetic
     `input` is what the cost estimate, the stale flag and the slider listen
     for, exactly as if the energy had been typed. */
  $<HTMLInputElement>('efd-nf-lambda')?.addEventListener('input', ev => {
    const nm = Number((ev.target as HTMLInputElement).value);
    const en = $<HTMLInputElement>('efd-nf-energy');
    if (!en || !(nm > 0)) return;
    en.value = String(Number((HC_EV_NM / nm).toPrecision(5)));
    en.dispatchEvent(new Event('input', { bubbles: true }));
  });

  // A permalink restore writes the unit here and dispatches; the numbers that
  // follow are already in it, so nothing is converted.
  $<HTMLInputElement>('efd-xunit')?.addEventListener('input', ev => {
    const v = (ev.target as HTMLInputElement).value;
    setXUnit(v === 'lambda' ? 'lambda' : 'energy', false);
  });

  // Display only: both arrays come out of the same run.
  $('efd-xs-unit')?.addEventListener('click', ev => {
    const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-xs]');
    if (!b) return;
    xsUnit = b.dataset.xs === 'nm' ? 'nm' : 'area';
    document.querySelectorAll<HTMLButtonElement>('#efd-xs-unit .plot-tab').forEach(x => {
      const on = x === b;
      x.classList.toggle('active', on);
      x.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    redrawCurrent();
  });

  // Display only: the overlay is built from the payload already in hand.
  $<HTMLInputElement>('efd-nf-source')?.addEventListener('change', ev => {
    nfShowSource = (ev.target as HTMLInputElement).checked;
    redrawNearField();
  });

  /** Tick box: the map's own peak tracks.
   *
   *  Display only -- `peak_E_eV` is in the payload either way, and the headline
   *  readout below the chart keeps reading it whether or not the curves are
   *  drawn, because that number is what the map FOUND and not what is on
   *  screen. Turning them off is for looking at the colormap unobstructed, or
   *  at the Level 2 bands alone. */
  $<HTMLInputElement>('efd-ldos-peaks')?.addEventListener('change', ev => {
    ldosShowPeaks = (ev.target as HTMLInputElement).checked;
    redrawCurrent();
  });

  /** The beta = 0 bands, when the page's own beta is not zero.
   *
   *  Keyed on the de-drifted spec, and kept apart from `resultCache` because it
   *  is NOT the dispersion view's result: `cachedFor('dispersion')` compares
   *  against the spec as typed, so a beta = 0 payload filed there would be a
   *  curve that answers a question nobody asked, waiting to be shown under the
   *  Dispersion tab as if it answered theirs. */
  let bandRefCache: { key: string; res: EfdResult } | null = null;

  /** Tick box: lay Level 2's bands over the Level 3 map.
   *
   *  The bands are a separate calculation, so the first tick may have to run
   *  one -- but usually it does not, because a Run already prefetched
   *  `dispersion` into the cache for exactly these parameters. Unticking keeps
   *  the result cached, so ticking again is free.
   *
   *  WITH DRIFT the bands are solved at beta = 0 and said to be. Level 2 needs
   *  eta to be one number multiplying a fixed geometric operator; drift makes
   *  it a tensor (eta_perp != eta_par), so a drifting band structure is not a
   *  Level 2 spectrum at all and `web.bands` refuses the spec -- rightly. The
   *  no-drift bands still exist, and over a drifting map they are the thing
   *  worth seeing: drift is the distance between them and the ridges beneath.
   *  Refusing to draw them left the tick box dead for the one setting where
   *  the comparison has something to say. */
  $<HTMLInputElement>('efd-ldos-bands')?.addEventListener('change', async ev => {
    const on = (ev.target as HTMLInputElement).checked;
    if (!on) {
      ldosBandTracks = null;
      ldosBandNote = null;
      ldosBandKey = null;
      redrawCurrent();
      return;
    }
    const asTyped = readSpec('dispersion');
    const { spec, drifting } = withoutDrift(asTyped);
    const key = `dispersion|${JSON.stringify(spec)}`;
    let bands = drifting
      ? (bandRefCache && bandRefCache.key === key ? bandRefCache.res : null)
      : cachedFor('dispersion');
    // Why a refusal was refused, in the solver's own words. `ok: false` is data
    // and arrives with a `reason` that says which setting cannot work and what
    // to do instead; this handler used to drop it and print "could not be
    // solved for these settings" -- the one sentence that fits every cause and
    // identifies none, after a wait long enough to look like the tick itself
    // had failed. The Run button has surfaced `reason` all along; this is the
    // same payload reaching the same box.
    let why: string | null = null;
    if (!bands) {
      showError('Solving the bands to overlay\u2026');
      const got = await bridge
        .run('bands', spec)
        .catch(err => {
          why = `The solver did not answer: ${err instanceof Error ? err.message : String(err)}`;
          return null;
        });
      showError(null);
      if (got && got.ok) {
        if (drifting) bandRefCache = { key, res: got };
        else resultCache.set('dispersion', { key, res: got });
        bands = got;
      } else if (got) {
        why = got.reason ?? null;
      }
    }
    if (!bands) {
      showError(
        why ?? 'The bands could not be solved for these settings, so there is nothing to overlay.',
      );
      (ev.target as HTMLInputElement).checked = false;
      return;
    }
    const bq = bands.qs_nm_inv as number[];
    const bE = bands.E_eV as (number | null)[][];
    const nb = Math.max(0, ...bE.map(r => r.length));
    ldosBandTracks = Array.from({ length: nb }, (_, b) => ({
      xs: bq,
      ys: bE.map(r => r[b] ?? null),
      // One hue for the whole family and one dash for all of it: these are not
      // the next few peak tracks, they are a different calculation, and the
      // map's own tracks have already used solid, dashed and dotted.
      color: BAND_OVERLAY_COLOR,
      dash: [] as number[],
    }));
    ldosBandKey = key;
    ldosBandNote = drifting
      ? 'Level 2 bands, solved at \u03b2 = 0 \u2014 the no-drift reference. With drift '
        + '\u03b7 is a tensor (\u03b7\u22a5 \u2260 \u03b7\u2225), so the drifting bands are '
        + 'not a Level 2 spectrum: what drift did is the gap between these lines and '
        + 'the ridges beneath them.'
      : 'Level 2 bands \u2014 a root-find that never saw this map, so where they land '
        + 'on its ridges is a check on both.';
    redrawCurrent();
  });

  // Display only: both faces come out of the same run.
  $('efd-eels-pick')?.addEventListener('click', ev => {
    const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-eels]');
    if (!b) return;
    eelsView = b.dataset.eels === 'map' ? 'map' : 'spectrum';
    document.querySelectorAll<HTMLButtonElement>('#efd-eels-pick .plot-tab').forEach(x => {
      const on = x === b;
      x.classList.toggle('active', on);
      x.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    applyVisibility();
    redrawCurrent();
  });

  // Display only: the orders are already computed and in hand.
  $('efd-nl-pick')?.addEventListener('click', ev => {
    const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-nl]');
    if (!b) return;
    nlPick = (b.dataset.nl as NlPick | undefined) ?? 'all';
    // The scale switch appears and disappears with the pick.
    applyVisibility();
    redrawCurrent();
  });

  // Display only: the data is in hand and only the axis changes.
  $('efd-nl-scale')?.addEventListener('click', ev => {
    const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-nlscale]');
    if (!b) return;
    nlLog = b.dataset.nlscale !== 'linear';
    document.querySelectorAll<HTMLButtonElement>('#efd-nl-scale .plot-tab').forEach(x => {
      const on = x === b;
      x.classList.toggle('active', on);
      x.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    redrawCurrent();
  });

  document.querySelectorAll<HTMLButtonElement>('#efd-nf-scale .plot-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      nfLog = btn.dataset.scale !== 'linear';
      document.querySelectorAll<HTMLButtonElement>('#efd-nf-scale .plot-tab').forEach(b => {
        const active = b === btn;
        b.classList.toggle('active', active);
        b.setAttribute('aria-checked', active ? 'true' : 'false');
      });
      redrawNearField();
    });
  });

  const fieldDialog = $<HTMLDialogElement>('efd-field-dialog');
  let fieldDialogLog = true;
  const FIELD_TILES: { id: string; key: FieldKey }[] = [
    { id: 'efd-f-ex-re', key: 'ex_re' },
    { id: 'efd-f-ey-re', key: 'ey_re' },
    { id: 'efd-f-ez-re', key: 'ez_re' },
    { id: 'efd-f-ex-im', key: 'ex_im' },
    { id: 'efd-f-ey-im', key: 'ey_im' },
    { id: 'efd-f-ez-im', key: 'ez_im' },
    { id: 'efd-f-mag', key: 'map' },
  ];

  const drawFieldTiles = (tries = 20): void => {
    if (!lastResult || !lastResult.ok) return;
    const first = $<HTMLCanvasElement>('efd-f-mag');
    // Same reason as the geometry dialog: a canvas in a closed <dialog> has no
    // size, and a timer rather than rAF because rAF does not fire in a page the
    // browser is not compositing.
    if (!first || first.clientWidth === 0) {
      if (tries > 0) window.setTimeout(() => drawFieldTiles(tries - 1), 16);
      return;
    }
    for (const t of FIELD_TILES) {
      const c = $<HTMLCanvasElement>(t.id);
      if (c) drawNearFieldInto(c, lastResult, t.key, t.key === 'map' && fieldDialogLog);
    }
    setText(
      'efd-field-meta',
      `${(lastResult.E_ev as number)?.toFixed(3)} eV · q = ${(lastResult.q_nm_inv as number)?.toFixed(3)} nm⁻¹`,
    );
  };

  $<HTMLButtonElement>('efd-nf-all')?.addEventListener('click', () => {
    if (!fieldDialog) return;
    if (!lastResult || !lastResult.ok || observable !== 'nearfield') {
      showError('Run the near field first — there is no field to show yet.');
      return;
    }
    fieldDialog.showModal();
    drawFieldTiles();
  });
  $<HTMLButtonElement>('efd-field-log')?.addEventListener('click', () => {
    fieldDialogLog = !fieldDialogLog;
    drawFieldTiles();
  });
  $<HTMLButtonElement>('efd-field-dialog-close')?.addEventListener('click', () => fieldDialog?.close());
  fieldDialog?.addEventListener('click', ev => {
    if (ev.target === fieldDialog) fieldDialog.close();
  });

  // ---- the enlarged geometry ----------------------------------------------

  const dialog = $<HTMLDialogElement>('efd-schematic-dialog');
  const large = $<HTMLCanvasElement>('efd-schematic-large');
  const drawLarge = (): void => {
    if (large && lastGeometry) drawRibbonSchematic(large, lastGeometry, lastSource);
  };
  /**
   * Draw once the canvas actually has a size.
   *
   * A canvas inside a closed `<dialog>` has clientWidth 0, and the drawing
   * functions correctly refuse to draw into nothing. One requestAnimationFrame
   * after showModal() is not reliably enough for the dialog to be laid out —
   * it opened blank — so this waits for a non-zero width instead of guessing a
   * number of frames.
   *
   * It retries on a timer rather than on requestAnimationFrame: rAF is tied to
   * painting and does not fire at all in a page the browser is not compositing
   * (a background tab, or an embedded pane that is hidden), which would leave
   * the dialog blank in exactly the cases hardest to notice.
   */
  const drawLargeWhenSized = (tries = 20): void => {
    if (!large) return;
    if (large.clientWidth > 0 && large.clientHeight > 0) {
      drawLarge();
      return;
    }
    if (tries > 0) window.setTimeout(() => drawLargeWhenSized(tries - 1), 16);
  };
  $<HTMLButtonElement>('efd-enlarge')?.addEventListener('click', () => {
    if (!dialog) return;
    dialog.showModal();
    drawLargeWhenSized();
  });
  $<HTMLButtonElement>('efd-schematic-dialog-close')?.addEventListener('click', () => dialog?.close());
  dialog?.addEventListener('click', ev => {
    // Clicking the backdrop closes it: the event target is the dialog itself
    // only when the click missed the panel.
    if (ev.target === dialog) dialog.close();
  });
  window.addEventListener('resize', () => {
    if (dialog?.open) drawLarge();
  });

  // ---- exports ------------------------------------------------------------

  $<HTMLButtonElement>('efd-dl-png')?.addEventListener('click', () => {
    const a = document.createElement('a');
    a.download = `electrostatics-fd-${observable}.png`;
    a.href = mainCanvas.toDataURL('image/png');
    a.click();
  });

  $<HTMLButtonElement>('efd-dl-json')?.addEventListener('click', () => {
    if (!lastResult) return;
    const payload = { excitation, observable, spec: readSpec(observable), result: lastResult };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.download = `electrostatics-fd-${observable}.json`;
    a.href = URL.createObjectURL(blob);
    a.click();
    URL.revokeObjectURL(a.href);
  });

  $<HTMLButtonElement>('efd-dl-csv')?.addEventListener('click', () => {
    if (!lastResult || !lastResult.ok) return;
    const lines: string[] = [];
    // The notes go into the CSV header as comments. A number exported without
    // its caveat is the same number without its caveat.
    for (const n of lastResult.notes ?? []) lines.push(`# ${n.replace(/\r?\n/g, ' ')}`);

    const res = lastResult;
    if (observable === 'ldos' || observable === 'dispersion') {
      const xs = (res.qs_nm_inv as number[]) ?? [];
      const ys = (res.energies_eV as number[]) ?? [];
      const z = (res.map as (number | null)[][]) ?? null;
      if (z) {
        lines.push(['q_nm_inv', ...ys.map(v => `E_${v.toFixed(4)}`)].join(','));
        xs.forEach((q, i) => lines.push([q, ...(z[i] ?? []).map(v => (v ?? ''))].join(',')));
      } else {
        const E = (res.E_eV as (number | null)[][]) ?? [];
        lines.push(['q_nm_inv', ...(E[0]?.map((_, b) => `band_${b + 1}_eV`) ?? [])].join(','));
        xs.forEach((q, i) => lines.push([q, ...(E[i] ?? []).map(v => (v ?? ''))].join(',')));
      }
    } else {
      const Es = (res.energies_eV as number[]) ?? [];
      // The column that is exported is the one that was plotted. Hard-coding
      // `extinction_arb` meant the EELS export silently wrote a header and no
      // rows the moment that view started returning a loss instead.
      const col: [string, (number | null)[][]] | null =
        res.loss_arb
          ? ['loss_arb', res.loss_arb as (number | null)[][]]
          : res.sigma_ext_per_area
            ? ['sigma_ext_per_area', res.sigma_ext_per_area as (number | null)[][]]
            : res.extinction_arb
              ? ['Im_alpha', res.extinction_arb as (number | null)[][]]
              : null;
      const shg = res.alpha_shg as EfdComplex | undefined;
      lines.push(col ? `E_eV,${col[0]}` : 'E_eV,Re_alpha,Im_alpha');
      Es.forEach((E, i) => {
        if (col) lines.push([E, col[1][0]?.[i] ?? ''].join(','));
        else if (shg) lines.push([E, shg.re[i] ?? '', shg.im[i] ?? ''].join(','));
      });
    }

    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.download = `electrostatics-fd-${observable}.csv`;
    a.href = URL.createObjectURL(blob);
    a.click();
    URL.revokeObjectURL(a.href);
  });

  $<HTMLButtonElement>('efd-geo-png')?.addEventListener('click', () => {
    const a = document.createElement('a');
    a.download = 'electrostatics-fd-geometry.png';
    a.href = schematic.toDataURL('image/png');
    a.click();
  });

  $<HTMLButtonElement>('efd-geo-csv')?.addEventListener('click', () => {
    if (!lastGeometry) return;
    // The discretisation itself, which is what the drawing is a picture of.
    const lines = ['ribbon,node,x_nm,z_nm,theta'];
    lastGeometry.ribbons.forEach((r, j) => {
      r.x_nodes_nm.forEach((x, i) => {
        lines.push([j + 1, i, x, r.z0_nm, r.theta[i] ?? ''].join(','));
      });
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.download = 'electrostatics-fd-geometry.csv';
    a.href = URL.createObjectURL(blob);
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ---- theme ---------------------------------------------------------------

  onThemeChange(() => {
    if (lastGeometry) drawRibbonSchematic(schematic, lastGeometry, lastSource);
    if (dialog?.open) drawLarge();
    if (fieldDialog?.open) drawFieldTiles();
    redrawCurrent();
  });

  // ---- boot ----------------------------------------------------------------

  const boot = $('efd-boot');
  const bootFill = $('efd-boot-fill');
  const bootText = $('efd-boot-text');

  bridge.onInitProgress((stage, pct) => {
    if (bootFill) bootFill.style.width = `${Math.round(pct * 100)}%`;
    if (bootText) bootText.textContent = stage;
  });

  runBtn.disabled = true;
  applyVisibility();
  // Nothing has been computed yet, and an empty chart on first load is the
  // very case the notice is for.
  setNeedsRun(true);
  setText('efd-out-headline-unit', OBSERVABLES[observable].headlineUnit);

  bridge
    .init()
    .then(cal => {
      if (boot) boot.hidden = true;
      runBtn.disabled = false;
      // Say plainly whether the cost estimate is this browser's number or a
      // desktop default, because the two differ by about 3x on solves.
      const model = (cal as { model?: { source?: string } }).model?.source;
      if (model) {
        const stat = $('efd-out-cost')?.closest('.stat');
        if (stat) (stat as HTMLElement).title = model;
      }
      return refreshLive();
    })
    .catch((err: unknown) => {
      if (bootText) {
        bootText.textContent =
          `The Python solver failed to start: ${err instanceof Error ? err.message : String(err)}`;
      }
      if (bootFill) bootFill.style.width = '100%';
      showError(
        'The solver could not start, so nothing on this page will compute. ' +
          'It needs to download Pyodide (~25 MB, cached afterwards) from a CDN.',
      );
    });
}
