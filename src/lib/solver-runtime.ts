/**
 * Client-side glue between a solver spec and the DOM the spec renderer emitted.
 *
 * It owns exactly the cross-cutting behaviour every solver needs — reading and
 * writing parameters, keeping number fields and sliders in step, applying
 * scenarios, restoring and updating the permalink, filling readouts, and
 * painting the live envelope — so that a module's own script is left with just
 * its physics and its plots.
 */
import {
  coerce, decodeState, defaultValues, encodeState, evaluateEnvelope, fmt, fromSliderPos,
  isChoice, isVisible, toSliderPos, type EnvelopeContext, type Param, type Solver,
} from './solver-spec';

export type Values = Record<string, number | string>;

export interface SolverBinding {
  /** Current values, coerced into their declared domains. */
  values(): Values;
  /** Write values into the controls. Fires the change callback unless silent. */
  setValues(next: Values, silent?: boolean): void;
  /** Fill the readout row from a map of observable values. */
  setObservables(o: Record<string, number>): void;
  /** Evaluate and paint the envelope rows. */
  setEnvelope(ctx: EnvelopeContext): void;
  /** Absolute URL encoding the current state. */
  permalink(): string;
}

export interface BindOptions {
  prefix?: string;
  outPrefix?: string;
  envPrefix?: string;
  /**
   * Selector for this solver's scenario buttons. Several solvers can share a
   * page (a lab with modules), so the default would otherwise wire every
   * module's presets to every module's binding.
   */
  scenarioSelector?: string;
  /** Called (debounced) whenever any parameter changes. */
  onChange: (values: Values) => void;
  /** Debounce in ms; 0 disables. */
  debounceMs?: number;
}

export function bindSolver(spec: Solver, opts: BindOptions): SolverBinding {
  const prefix = opts.prefix ?? 'sp-';
  const outPrefix = opts.outPrefix ?? 'out-';
  const envPrefix = opts.envPrefix ?? 'env-';
  const debounceMs = opts.debounceMs ?? 120;

  const byKey = new Map<string, Param>(spec.params.map(p => [p.key, p]));
  const el = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLSelectElement | null;
  const field = (key: string) => el(`${prefix}${key}`);
  const slider = (key: string) => el(`${prefix}${key}-slider`) as HTMLInputElement | null;

  function values(): Values {
    const out: Values = {};
    for (const p of spec.params) {
      const f = field(p.key);
      out[p.key] = f ? coerce(p, f.value) : p.default;
    }
    return out;
  }

  /**
   * Show or hide the controls whose `showIf` depends on the current values —
   * for parameters belonging to one branch of a model choice. Hidden controls
   * keep their value and still reach `compute`, so what the form shows never
   * changes the physics.
   */
  function applyVisibility(current: Values): void {
    for (const p of spec.params) {
      if (!p.showIf) continue;
      const row = document.querySelector<HTMLElement>(`[data-param-row="${prefix}${p.key}"]`);
      if (row) row.hidden = !isVisible(p, current);
    }
  }

  let timer: number | undefined;
  function emit(): void {
    if (debounceMs <= 0) { opts.onChange(values()); return; }
    if (timer !== undefined) clearTimeout(timer);
    timer = window.setTimeout(() => { timer = undefined; opts.onChange(values()); }, debounceMs);
  }

  /** Reflect a value into its number field, slider and preset buttons. */
  function paint(p: Param, v: number | string): void {
    const f = field(p.key);
    if (f) f.value = String(v);
    const s = slider(p.key);
    if (s && typeof v === 'number') s.value = String(toSliderPos(p, v));
    document.querySelectorAll(`[data-param-preset="${prefix}${p.key}"]`).forEach(b => {
      b.classList.toggle('active', Number((b as HTMLElement).dataset.val) === Number(v));
    });
  }

  function setValues(next: Values, silent = false): void {
    for (const [key, raw] of Object.entries(next)) {
      const p = byKey.get(key);
      if (!p) continue;
      paint(p, coerce(p, raw));
    }
    syncUrl();
    if (!silent) emit();
  }

  function syncUrl(): void {
    const current = values();
    applyVisibility(current);
    const qs = encodeState(spec, current);
    const url = new URL(window.location.href);
    url.search = qs;
    history.replaceState(null, '', url.toString());
  }

  // ── wire the controls ──────────────────────────────────────────────────────
  for (const p of spec.params) {
    const f = field(p.key);
    if (f) {
      const evt = isChoice(p) ? 'change' : 'input';
      f.addEventListener(evt, () => {
        // do not coerce mid-typing, or "0.0" collapses before the user finishes
        const s = slider(p.key);
        const raw = Number(f.value);
        if (s && Number.isFinite(raw)) s.value = String(toSliderPos(p, raw));
        document.querySelectorAll(`[data-param-preset="${prefix}${p.key}"]`).forEach(b => {
          b.classList.toggle('active', Number((b as HTMLElement).dataset.val) === raw);
        });
        syncUrl();
        emit();
      });
      // normalise into the declared domain when focus leaves
      f.addEventListener('change', () => { paint(p, coerce(p, f.value)); syncUrl(); });
    }
    const s = slider(p.key);
    if (s) {
      s.addEventListener('input', () => {
        const v = coerce(p, fromSliderPos(p, Number(s.value)));
        const ff = field(p.key);
        if (ff) ff.value = String(v);
        document.querySelectorAll(`[data-param-preset="${prefix}${p.key}"]`).forEach(b => {
          b.classList.toggle('active', Number((b as HTMLElement).dataset.val) === Number(v));
        });
        syncUrl();
        emit();
      });
    }
  }

  // Value chips are keyed by the prefixed parameter id, so a page hosting more
  // than one solver (a lab with modules) does not cross-wire them.
  document.querySelectorAll<HTMLElement>(`[data-param-preset^="${prefix}"]`).forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.paramPreset!.slice(prefix.length);
      if (!byKey.has(key)) return;
      setValues({ [key]: Number(btn.dataset.val) });
    });
  });

  // Scenarios; the strip is rendered by the page from spec.scenarios. Scoped
  // for the same reason as the value chips above.
  const scenarioButtons = document.querySelectorAll<HTMLElement>(
    opts.scenarioSelector ?? '[data-scenario]',
  );
  scenarioButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const sc = spec.scenarios[Number(btn.dataset.scenario)];
      if (!sc) return;
      scenarioButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Reset to the declared defaults before applying the scenario's overrides.
      // Without the reset a preset inherits whatever the previous one changed, so
      // "the crowded-field preset" would silently mean something different
      // depending on what was clicked before it — and a preset whose whole point
      // is to trip one specific validity check would trip a different one.
      setValues({ ...defaultValues(spec), ...sc.set });
    });
  });

  // ── outputs ───────────────────────────────────────────────────────────────
  function setObservables(o: Record<string, number>): void {
    for (const ob of spec.observables) {
      const target = document.getElementById(`${outPrefix}${ob.key}`);
      if (target) target.textContent = fmt(o[ob.key], ob.digits ?? 3);
    }
  }

  function setEnvelope(ctx: EnvelopeContext): void {
    for (const r of evaluateEnvelope(spec.envelope, ctx)) {
      const row = document.getElementById(`${envPrefix}${r.id}`);
      if (row) row.dataset.level = r.level;
      const msg = document.getElementById(`${envPrefix}${r.id}-msg`);
      if (msg) msg.textContent = r.message;
    }
  }

  function permalink(): string {
    const url = new URL(window.location.href);
    url.search = encodeState(spec, values());
    return url.toString();
  }

  // ── restore from the URL, then run once ───────────────────────────────────
  const initial = decodeState(spec, window.location.search.replace(/^\?/, ''));
  setValues(initial, true);

  const api = { values, setValues, setObservables, setEnvelope, permalink };

  // The first run is deferred by a microtask, not called inline: a caller
  // naturally writes `const binding = bindSolver(...)` and then uses `binding`
  // inside its onChange, which is still uninitialised while this constructor
  // runs. Deferring lets that assignment complete first.
  queueMicrotask(() => opts.onChange(values()));

  return api;
}
