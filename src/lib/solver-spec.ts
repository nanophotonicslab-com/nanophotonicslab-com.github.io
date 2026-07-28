/**
 * Declarative solver specification.
 *
 * A tool page is *generated* from one of these: the eleven regions the lab
 * already uses by convention (identity, model block, schematic, grouped
 * parameters, presets, plots, readouts, derived box, notes, live envelope,
 * export) become fields of a declaration. Adding a solver should cost a spec
 * and a kernel, not a page.
 *
 * This module is pure and is imported both at build time (to emit the controls)
 * and on the client (to evaluate the envelope and read parameters back), so it
 * must not touch the DOM.
 */

// ─── parameters ──────────────────────────────────────────────────────────────

export interface ParamGroup {
  id: string;
  label: string;
  /** Start collapsed. Groups render in declaration order. */
  collapsed?: boolean;
}

export interface Param {
  key: string;
  label: string;
  group: string;
  /** One-line meaning for the ⓘ tooltip — the physics, not the formula. */
  help?: string;
  /** Symbol shown next to the label, e.g. 'λ'. */
  symbol?: string;
  /** Unit shown in the label, e.g. 'nm'. */
  unit?: string;
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
  /** Slider mapping; 'log' for quantities spanning decades. */
  scale?: 'linear' | 'log';
  integer?: boolean;
  /** One-click values, rendered as small buttons under the field. */
  presets?: number[];
  /** Presence of `choices` makes this a select rather than a number field. */
  choices?: (string | number)[];
  /** Hidden behind the advanced toggle. */
  advanced?: boolean;
}

export function isChoice(p: Param): boolean {
  return Array.isArray(p.choices) && p.choices.length > 0;
}

/** True when the parameter should get a slider alongside its number field. */
export function hasSlider(p: Param): boolean {
  return !isChoice(p) && typeof p.min === 'number' && typeof p.max === 'number';
}

// ─── outputs ─────────────────────────────────────────────────────────────────

export interface Observable {
  key: string;
  label: string;
  unit?: string;
  /** Meaning for the ⓘ tooltip. Define ambiguous quantities (e.g. SNR) here. */
  help: string;
  /** Significant digits for display. */
  digits?: number;
}

export interface PlotSpec {
  id: string;
  title: string;
  xLabel: string;
  xUnit?: string;
  yLabel?: string;
  /** Short annotation drawn under the title, e.g. a fit result. */
  note?: string;
}

// ─── validity envelope ───────────────────────────────────────────────────────

export type EnvelopeLevel = 'ok' | 'warn' | 'fail';

export interface EnvelopeContext {
  /** Current parameter values, keyed as declared. */
  p: Record<string, number | string>;
  /** Current observable values, keyed as declared. */
  o: Record<string, number>;
}

export interface EnvelopeResult {
  level: EnvelopeLevel;
  /** One line that names the fix, not just the problem. */
  message: string;
}

export interface EnvelopeCheck {
  id: string;
  label: string;
  evaluate: (ctx: EnvelopeContext) => EnvelopeResult;
}

/** Evaluate every check. Never throws: a broken predicate must not block the run. */
export function evaluateEnvelope(
  checks: EnvelopeCheck[], ctx: EnvelopeContext,
): (EnvelopeResult & { id: string; label: string })[] {
  return checks.map(c => {
    try {
      return { id: c.id, label: c.label, ...c.evaluate(ctx) };
    } catch {
      return { id: c.id, label: c.label, level: 'ok' as EnvelopeLevel, message: '' };
    }
  });
}

// ─── presets ─────────────────────────────────────────────────────────────────

export interface Scenario {
  label: string;
  /** Parameter values to apply; omitted keys keep their current value. */
  set: Record<string, number | string>;
  /** Why this preset exists — several deliberately trip a check. */
  note?: string;
}

// ─── the solver ──────────────────────────────────────────────────────────────

export interface SolverMeta {
  lab: string;
  id: string;
  /** Short code shown as a badge, e.g. 'IMG1'. */
  code: string;
  title: string;
  blurb: string;
  status: 'Stable' | 'Beta' | 'Experimental';
  version: string;
  updated: string;
}

export interface SolverDocs {
  model: string;
  assumptions: string[];
  validity: string;
  limitations: string[];
  references: string[];
  notes?: string;
}

/** Drives the auto-drawn setup schematic — never hand-draw it per module. */
export interface SolverStructure {
  kind: string;
  sample: string;
  detector: string;
}

export interface Solver {
  meta: SolverMeta;
  docs: SolverDocs;
  structure?: SolverStructure;
  groups: ParamGroup[];
  params: Param[];
  observables: Observable[];
  plots: PlotSpec[];
  envelope: EnvelopeCheck[];
  scenarios: Scenario[];
}

// ─── helpers over a spec ─────────────────────────────────────────────────────

/** Default parameter values, keyed as declared. */
export function defaultValues(spec: Solver): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const p of spec.params) out[p.key] = p.default;
  return out;
}

/** Parameters of one group, in declaration order. */
export function groupParams(spec: Solver, groupId: string): Param[] {
  return spec.params.filter(p => p.group === groupId);
}

/** Clamp and round a value to a parameter's declared domain. */
export function coerce(p: Param, raw: number | string): number | string {
  if (isChoice(p)) {
    const asNum = typeof p.choices![0] === 'number' ? Number(raw) : String(raw);
    return p.choices!.includes(asNum as never) ? asNum : p.default;
  }
  let v = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isFinite(v)) v = Number(p.default);
  if (typeof p.min === 'number') v = Math.max(p.min, v);
  if (typeof p.max === 'number') v = Math.min(p.max, v);
  if (p.integer) v = Math.round(v);
  return v;
}

/** Resolution of a logarithmic slider, in steps. */
export const LOG_SLIDER_STEPS = 1000;

/** True when the parameter's slider is logarithmic. */
export function isLogSlider(p: Param): boolean {
  return p.scale === 'log' && hasSlider(p) && (p.min as number) > 0;
}

/** Map a value to its slider position (identity for linear sliders). */
export function toSliderPos(p: Param, value: number): number {
  if (!isLogSlider(p)) return value;
  const lo = Math.log(p.min as number), hi = Math.log(p.max as number);
  const v = Math.min(Math.max(value, p.min as number), p.max as number);
  return Math.round((LOG_SLIDER_STEPS * (Math.log(v) - lo)) / (hi - lo));
}

/** Map a slider position back to a value (identity for linear sliders). */
export function fromSliderPos(p: Param, pos: number): number {
  if (!isLogSlider(p)) return pos;
  const lo = Math.log(p.min as number), hi = Math.log(p.max as number);
  const v = Math.exp(lo + ((hi - lo) * pos) / LOG_SLIDER_STEPS);
  // round to a readable number of significant digits for the number field
  return Number(v.toPrecision(4));
}

/** Label including symbol and unit, e.g. 'λ_em (nm)'. */
export function paramLabel(p: Param): string {
  const base = p.symbol && p.symbol !== p.label ? `${p.label} ${p.symbol}` : p.label;
  return p.unit ? `${base} (${p.unit})` : base;
}

/**
 * Compact fixed-significance formatting shared by readouts and annotations.
 * Falls back to an em dash for values that are not finite.
 */
export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(0, digits - 1));
  const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(a)));
  return v.toFixed(Math.min(decimals, 6));
}

// ─── permalinks ──────────────────────────────────────────────────────────────

/**
 * Encode the parameter set into a compact URL hash fragment. Only values that
 * differ from their declared default are stored, so links stay short and a
 * later change of default does not silently change an old link's meaning for
 * parameters it explicitly pinned.
 */
export function encodeState(spec: Solver, values: Record<string, number | string>): string {
  const parts: string[] = [];
  for (const p of spec.params) {
    const v = values[p.key];
    if (v === undefined || v === p.default) continue;
    parts.push(`${encodeURIComponent(p.key)}=${encodeURIComponent(String(v))}`);
  }
  return parts.join('&');
}

/** Decode a hash fragment, coercing every value into its declared domain. */
export function decodeState(spec: Solver, hash: string): Record<string, number | string> {
  const out = defaultValues(spec);
  const clean = hash.replace(/^#/, '');
  if (!clean) return out;
  const byKey = new Map(spec.params.map(p => [p.key, p]));
  for (const kv of clean.split('&')) {
    const i = kv.indexOf('=');
    if (i < 0) continue;
    const key = decodeURIComponent(kv.slice(0, i));
    const p = byKey.get(key);
    if (!p) continue;
    out[key] = coerce(p, decodeURIComponent(kv.slice(i + 1)));
  }
  return out;
}
