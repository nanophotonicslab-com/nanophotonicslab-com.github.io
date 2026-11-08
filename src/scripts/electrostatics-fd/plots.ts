/**
 * Canvas renderers for the electrostatics-FD widget: spectra, (q, E) maps, the
 * ribbon schematic and mode profiles.
 *
 * Every entry point is idempotent — a theme change just re-calls it — and
 * re-reads its chrome colours from the global.css tokens on each draw, so dark
 * mode needs no extra wiring.
 *
 * Non-finite samples cross the Python boundary as `null`. A null is a gap in
 * the physics, not a zero, so every polyline here breaks its subpath and every
 * map cell paints a distinct "no data" colour instead of the bottom of the
 * scale. Plotting one as zero would be a lie about the result.
 *
 * Frame geometry and number formatting come from src/lib/chart-lite.ts so these
 * charts line up with the rest of the lab; see the note on drawSpectrum for why
 * the series stroking itself lives here.
 *
 * Legends are HTML, as in every other lab widget: hand seriesLegendHtml() to a
 * `.chart-legend` container.
 */

import { LITE_PAD, fmtLite, legendHtml } from '../../lib/chart-lite';
import { chartTheme } from '../../lib/chart-theme';
import { viridis } from '../bem/colormap';

/* ===== Theme access ===== */

/** One CSS custom property, resolved against the live theme. */
export function cssVar(name: string, fallback = '#888'): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

/* Spectral tokens from global.css, in an order that keeps adjacent series
 * distinguishable for the common two- and three-curve cases (Re/Im, s/p). */
const SERIES_TOKENS = [
  '--indigo', '--rose', '--emerald', '--amber', '--cyan', '--violet', '--orange', '--blue',
];

/** Palette colour for series `i`, wrapping once the tokens run out. */
export function seriesColor(i: number): string {
  return cssVar(SERIES_TOKENS[((i % SERIES_TOKENS.length) + SERIES_TOKENS.length) % SERIES_TOKENS.length]);
}

/**
 * CSS colour string to 8-bit RGB, for the ImageData path — createImageData
 * takes numbers, not the `var(--…)` strings the rest of the file can use.
 */
function rgbOf(css: string, fallback: [number, number, number]): [number, number, number] {
  const hex = /^#([0-9a-f]{3,8})$/i.exec(css.trim());
  if (hex) {
    const d = hex[1];
    if (d.length === 3) {
      return [parseInt(d[0] + d[0], 16), parseInt(d[1] + d[1], 16), parseInt(d[2] + d[2], 16)];
    }
    if (d.length >= 6) {
      return [parseInt(d.slice(0, 2), 16), parseInt(d.slice(2, 4), 16), parseInt(d.slice(4, 6), 16)];
    }
    return fallback;
  }
  const fn = /rgba?\(([^)]+)\)/i.exec(css);
  if (fn) {
    const p = fn[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    if (p.length >= 3 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2])) {
      return [p[0], p[1], p[2]];
    }
  }
  return fallback;
}

/* --text-muted is a neutral grey in both themes, so it reads as "absent" next
 * to every colormap below — none of which contains a grey. */
const NO_DATA_FALLBACK: [number, number, number] = [107, 114, 128];

/* ===== Canvas plumbing ===== */

interface Surface { ctx: CanvasRenderingContext2D; w: number; h: number; dpr: number }

/** Size the backing store for the display and hand back CSS-pixel dimensions. */
function surface(canvas: HTMLCanvasElement): Surface | null {
  const r = canvas.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return null;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  // setTransform rather than scale(): these functions are re-entered on every
  // theme change, and scale() would compound if the width assignment above ever
  // stopped resetting the matrix for us.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.font = `11px ${cssVar('--font', 'Inter, sans-serif')}`;
  return { ctx, w: r.width, h: r.height, dpr };
}

function message(ctx: CanvasRenderingContext2D, w: number, h: number, text: string): void {
  ctx.fillStyle = chartTheme().textMuted;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2);
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/* chart-lite keeps its tick helper private and this file may not edit it, so
 * the 1-2-5 ladder is repeated here (same rungs as the heterostructures page). */
function niceTicks(min: number, max: number, n = 5): number[] {
  const span = max - min;
  if (!(span > 0)) return [min];
  const mag = 10 ** Math.floor(Math.log10(span / n));
  const norm = span / n / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-2; v += step) {
    out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  return out;
}

/** Decade ticks in log10 space, thinned so a many-decade axis stays readable. */
function decadeTicks(lo: number, hi: number): number[] {
  const step = Math.max(1, Math.ceil((hi - lo) / 6));
  const out: number[] = [];
  for (let e = Math.ceil(lo - 1e-9); e <= hi + 1e-9; e += step) out.push(e);
  return out.length > 0 ? out : [lo, hi];
}

/** Widen a degenerate range so no axis transform divides by zero. */
function padRange(lo: number, hi: number): [number, number] {
  if (hi > lo) return [lo, hi];
  const d = Math.max(Math.abs(lo), 1) * 0.05;
  return [lo - d, hi + d];
}

/* Looped rather than Math.min(...xs): a dense energy sweep is long enough to
 * overflow the argument list. */
function extent(xs: number[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of xs) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? padRange(lo, hi) : [0, 1];
}

/**
 * For each of `nPix` pixel centres spanning [lo, hi], the index of the nearest
 * entry of `nodes` — the per-pixel cell lookup the (q, E) maps sample through.
 *
 * Nearest in DATA space, not a proportional index: see the note in renderMap
 * for the grid that makes the difference.
 */
function nearestIndices(nodes: number[], lo: number, hi: number, nPix: number): Int32Array {
  const out = new Int32Array(nPix);
  const n = nodes.length;
  if (n < 2) return out;
  let ascending = true;
  for (let i = 1; i < n; i++) {
    if (!(nodes[i] > nodes[i - 1])) { ascending = false; break; }
  }
  const step = nPix > 1 ? (hi - lo) / (nPix - 1) : 0;
  let i0 = 0;
  for (let p = 0; p < nPix; p++) {
    const v = lo + step * p;
    if (ascending) {
      // Both the nodes and v increase, so the nearest index never moves back.
      while (i0 + 1 < n && Math.abs(nodes[i0 + 1] - v) <= Math.abs(nodes[i0] - v)) i0++;
      out[p] = i0;
    } else {
      let best = 0, bd = Infinity;
      for (let i = 0; i < n; i++) {
        const d = Math.abs(nodes[i] - v);
        if (d < bd) { bd = d; best = i; }
      }
      out[p] = best;
    }
  }
  return out;
}

/** Nearest 1/2/5 round number at or below `v` — for the schematic scale bar. */
function niceLength(v: number): number {
  if (!(v > 0)) return 1;
  const mag = 10 ** Math.floor(Math.log10(v));
  const n = v / mag;
  return (n < 2 ? 1 : n < 5 ? 2 : 5) * mag;
}

interface Box { l: number; r: number; t: number; b: number }
interface Frame { px: (v: number) => number; py: (v: number) => number; pw: number; ph: number }

/** Grid, ticks and axis titles; returns the data-to-pixel transform. */
function axes(
  ctx: CanvasRenderingContext2D, w: number, h: number, box: Box,
  xLo: number, xHi: number, yLo: number, yHi: number,
  xLabel: string, yLabel: string,
  xTicks: number[], yTicks: number[],
  fmtX: (v: number) => string, fmtY: (v: number) => string,
  grid = true,
): Frame {
  const t = chartTheme();
  const pw = w - box.l - box.r;
  const ph = h - box.t - box.b;
  const sx = pw / (xHi - xLo);
  const sy = ph / (yHi - yLo);
  const px = (v: number) => box.l + (v - xLo) * sx;
  const py = (v: number) => box.t + ph - (v - yLo) * sy;

  ctx.strokeStyle = t.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(box.l, box.t, pw, ph);

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of yTicks) {
    const Y = py(v);
    if (Y < box.t - 1 || Y > box.t + ph + 1) continue;
    if (grid) {
      ctx.strokeStyle = t.grid;
      ctx.beginPath();
      ctx.moveTo(box.l, Y);
      ctx.lineTo(box.l + pw, Y);
      ctx.stroke();
    }
    ctx.fillStyle = t.textMuted;
    ctx.fillText(fmtY(v), box.l - 6, Y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const v of xTicks) {
    const X = px(v);
    if (X < box.l - 1 || X > box.l + pw + 1) continue;
    ctx.fillStyle = t.textMuted;
    ctx.fillText(fmtX(v), X, box.t + ph + 6);
  }

  ctx.fillStyle = t.textSoft;
  ctx.fillText(xLabel, box.l + pw / 2, h - 14);
  ctx.save();
  ctx.translate(14, box.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  return { px, py, pw, ph };
}

/* ===== Spectra ===== */

export interface Series { label: string; ys: (number | null)[]; color?: string; dashed?: boolean }

export interface SpectrumData {
  xs: number[];
  series: Series[];
  xLabel: string;
  yLabel: string;
  logY?: boolean;
  /** Pin the vertical axis instead of fitting it to the data.
   *
   *  For a chart that is meant to be read against another one -- the band
   *  diagram against the LDOS map over the same (q, E) window -- fitting to
   *  the data is exactly wrong: the two end up with different axes and the
   *  comparison they exist for cannot be made. */
  yRange?: [number, number];
  /** `_x` in a label is a subscript: "2|E_F|" draws E with a small F. */
  markers?: { x: number; label: string }[];
}

/* LITE_PAD plus room for the rotated y-axis title and the marker captions,
 * neither of which chart-lite draws. attachSpectrumHover() reads the same
 * constant, so its pointer-to-index arithmetic stays in step with the frame. */
const SPEC_PAD: Box = { l: LITE_PAD.l + 16, r: LITE_PAD.r + 8, t: LITE_PAD.t + 12, b: LITE_PAD.b + 6 };

/* A spectrum carries no hover index of its own, so it lives beside the canvas:
 * drawSpectrum stays a pure function of SpectrumData, and a page that never
 * calls attachSpectrumHover never gets a crosshair. */
const hoverAt = new WeakMap<HTMLCanvasElement, number>();

/** At most this many decades below the peak, so one near-zero sample cannot flatten the rest. */
const LOG_DECADES = 8;

const NO_Y = new Float64Array(0);

/**
 * Legend markup for a `.chart-legend` container, with the same colour
 * resolution drawSpectrum uses. Built through chart-lite's own legendHtml so
 * the markup has a single definition (the `y` it wants is unused there).
 */
export function seriesLegendHtml(series: Series[]): string {
  return legendHtml(series.map((s, i) => ({
    y: NO_Y, label: s.label, color: s.color ?? seriesColor(i),
  })));
}

/**
 * Line chart for the spectrum/dispersion panels.
 *
 * chart-lite's drawLite is not reused directly: LiteChart carries its y data as
 * Float64Array, which cannot hold a `null` — and a NaN handed to lineTo is
 * silently dropped by the canvas, so the stroke jumps straight across the gap.
 * drawLite also seeds its y-range at zero, has no log axis and no markers, and
 * exposes no transform to draw them over. The frame geometry (SPEC_PAD) and the
 * tick formatter (fmtLite) are taken from it so the result is pixel-compatible
 * with the lab's other charts; only the stroking loop is local.
 */
/** Split a label on `_`: "2|E_F|" -> [ "2|E", sub "F", "|" ].
 *
 *  A subscript runs from the underscore to the end of the following run of
 *  letters and digits, so "2|E_F|" subscripts only the F. */
function subscriptRuns(text: string): { s: string; sub: boolean }[] {
  const runs: { s: string; sub: boolean }[] = [];
  let i = 0;
  while (i < text.length) {
    const u = text.indexOf('_', i);
    if (u < 0) {
      runs.push({ s: text.slice(i), sub: false });
      break;
    }
    if (u > i) runs.push({ s: text.slice(i, u), sub: false });
    let j = u + 1;
    while (j < text.length && /[A-Za-z0-9]/.test(text.charAt(j))) j += 1;
    if (j > u + 1) runs.push({ s: text.slice(u + 1, j), sub: true });
    else runs.push({ s: '_', sub: false }); // a lone underscore is just itself
    i = j;
  }
  return runs;
}

/** The font the subscript runs are drawn in: the current one, three quarters
 *  the size, never below 8px. */
function subscriptFont(font: string): { font: string; drop: number } {
  const m = /(\d+(?:\.\d+)?)px/.exec(font);
  const px = m ? Number(m[1]) : 11;
  const small = Math.max(8, Math.round(px * 0.75));
  return { font: font.replace(/(\d+(?:\.\d+)?)px/, `${small}px`), drop: Math.round(px * 0.22) };
}

/** Width of a label once its `_x` runs are shrunk. */
function subscriptWidth(ctx: CanvasRenderingContext2D, text: string): number {
  const base = ctx.font;
  const { font: small } = subscriptFont(base);
  let w = 0;
  for (const r of subscriptRuns(text)) {
    ctx.font = r.sub ? small : base;
    w += ctx.measureText(r.s).width;
  }
  ctx.font = base;
  return w;
}

/** Draw a label with real subscripts, at `textBaseline: 'top'`.
 *
 *  Canvas has no rich text, and Unicode cannot help: its Latin subscript block
 *  has no F, no y and no z. The only way to typeset "2|E_F|" is to place the
 *  run after the underscore in a smaller font, lowered. `align` positions the
 *  whole label, since ctx.textAlign cannot span runs drawn one at a time. */
function subscriptText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  align: 'left' | 'right',
): void {
  const base = ctx.font;
  const { font: small, drop } = subscriptFont(base);
  const runs = subscriptRuns(text);
  const widths = runs.map(r => {
    ctx.font = r.sub ? small : base;
    return ctx.measureText(r.s).width;
  });
  const prevAlign = ctx.textAlign;
  ctx.textAlign = 'left';
  let cx = align === 'right' ? x - widths.reduce((a, b) => a + b, 0) : x;
  runs.forEach((r, k) => {
    ctx.font = r.sub ? small : base;
    ctx.fillText(r.s, cx, r.sub ? y + drop : y);
    cx += widths[k] ?? 0;
  });
  ctx.font = base;
  ctx.textAlign = prevAlign;
}

export function drawSpectrum(canvas: HTMLCanvasElement, d: SpectrumData): void {
  const s = surface(canvas);
  if (!s) return;
  const { ctx, w, h } = s;
  const t = chartTheme();
  const box = SPEC_PAD;
  if (w - box.l - box.r < 30 || h - box.t - box.b < 30) return;

  const n = d.xs.length;
  const log = d.logY === true;
  // Under a log axis a non-positive sample is as unplottable as a null. The
  // `undefined` case covers a series shorter than xs, which is also a gap.
  const usable = (v: number | null | undefined): v is number =>
    v !== null && v !== undefined && Number.isFinite(v) && (!log || v > 0);

  let yMin = Infinity, yMax = -Infinity;
  for (const se of d.series) {
    for (let i = 0; i < n; i++) {
      const v = se.ys[i];
      if (!usable(v)) continue;
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (n === 0 || d.series.length === 0 || !Number.isFinite(yMin)) {
    message(ctx, w, h, 'no finite data');
    return;
  }

  const [xLo, xHi] = extent(d.xs);
  let yLo: number, yHi: number, yTicks: number[], fmtY: (v: number) => string;
  const ty = log ? (v: number) => Math.log10(v) : (v: number) => v;
  if (log) {
    const hiE = Math.log10(yMax);
    const loE = Math.max(Math.log10(yMin), hiE - LOG_DECADES);
    yLo = Math.floor(loE);
    yHi = Math.ceil(hiE);
    if (yHi <= yLo) yHi = yLo + 1;
    yTicks = decadeTicks(yLo, yHi);
    fmtY = (v) => fmtLite(10 ** v);
  } else if (d.yRange && d.yRange[1] > d.yRange[0]) {
    [yLo, yHi] = d.yRange;
    yTicks = niceTicks(yLo, yHi, 4);
    fmtY = fmtLite;
  } else {
    const pad = (yMax - yMin) * 0.05;
    [yLo, yHi] = padRange(yMin - pad, yMax + pad);
    yTicks = niceTicks(yLo, yHi, 4);
    fmtY = fmtLite;
  }

  const fr = axes(ctx, w, h, box, xLo, xHi, yLo, yHi, d.xLabel, d.yLabel,
    niceTicks(xLo, xHi, 6), yTicks, fmtLite, fmtY);

  ctx.save();
  ctx.beginPath();
  ctx.rect(box.l, box.t, fr.pw, fr.ph);
  ctx.clip();

  // Markers first: a labelled rule (the Pauli edge at 2|E_F|) belongs behind
  // the data it explains.
  for (const m of d.markers ?? []) {
    if (!Number.isFinite(m.x) || m.x < xLo || m.x > xHi) continue;
    const X = fr.px(m.x);
    ctx.strokeStyle = t.textMuted;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(X, box.t);
    ctx.lineTo(X, box.t + fr.ph);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = t.textSoft;
    ctx.textBaseline = 'top';
    // BESIDE the rule, never centred on it: centred, the caption straddled its
    // own dashed line and the two read as one smudge. To the right by default,
    // flipped left when the rule is close enough to the right edge that the
    // caption would leave the frame.
    const w = subscriptWidth(ctx, m.label);
    const right = X + 5 + w <= box.l + fr.pw - 2;
    subscriptText(ctx, m.label, right ? X + 5 : X - 5, box.t + 3, right ? 'left' : 'right');
  }

  d.series.forEach((se, si) => {
    ctx.strokeStyle = se.color ?? seriesColor(si);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.setLineDash(se.dashed === true ? [5, 4] : []);
    ctx.beginPath();
    let run = 0;
    for (let i = 0; i < n; i++) {
      const v = se.ys[i];
      if (!usable(v)) { run = 0; continue; }          // gap: next point starts a new subpath
      const X = fr.px(d.xs[i]);
      const Y = fr.py(ty(v));
      if (run === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      run++;
      // A lone sample between two gaps has no segment to draw, and dropping it
      // would hide real data — mark it.
      if (run === 1 && !usable(se.ys[i + 1])) ctx.fillRect(X - 1.5, Y - 1.5, 3, 3);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  });

  const hi = hoverAt.get(canvas) ?? -1;
  if (hi >= 0 && hi < n) {
    const X = fr.px(d.xs[hi]);
    ctx.strokeStyle = t.textMuted;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(X, box.t);
    ctx.lineTo(X, box.t + fr.ph);
    ctx.stroke();
    ctx.setLineDash([]);
    d.series.forEach((se, si) => {
      const v = se.ys[hi];
      if (!usable(v)) return;
      ctx.fillStyle = se.color ?? seriesColor(si);
      ctx.beginPath();
      ctx.arc(X, fr.py(ty(v)), 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
  ctx.restore();
}

/**
 * Crosshair and `.chart-tooltip` readout for a spectrum canvas.
 *
 * chart-lite's attachLiteHover cannot be reused: it redraws through drawLite.
 * The pointer arithmetic and tooltip placement mirror it, and a `null` sample
 * is reported as an em dash rather than as a number that was never computed.
 */
export function attachSpectrumHover(
  canvas: HTMLCanvasElement, tip: HTMLElement, get: () => SpectrumData | undefined,
): void {
  canvas.addEventListener('pointermove', (ev) => {
    const d = get();
    if (!d || d.xs.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const span = rect.width - SPEC_PAD.l - SPEC_PAD.r;
    if (span <= 0) return;
    const rel = (ev.clientX - rect.left - SPEC_PAD.l) / span;
    const i = Math.max(0, Math.min(d.xs.length - 1, Math.round(rel * (d.xs.length - 1))));
    hoverAt.set(canvas, i);
    drawSpectrum(canvas, d);
    tip.classList.add('visible');
    tip.style.left = `${Math.min(ev.clientX - rect.left + 10, Math.max(0, rect.width - 150))}px`;
    tip.style.top = '6px';
    tip.innerHTML = `${fmtLite(d.xs[i])}<br>` + d.series.map((se) => {
      const v = se.ys[i];
      return `${se.label}: ${v === null || !Number.isFinite(v) ? '—' : fmtLite(v)}`;
    }).join('<br>');
  });
  canvas.addEventListener('pointerleave', () => {
    hoverAt.delete(canvas);
    tip.classList.remove('visible');
    const d = get();
    if (d) drawSpectrum(canvas, d);
  });
}

/* ===== (q, E) maps ===== */

/**
 * Things drawn ON the map in data coordinates: the structure, and where the
 * source is. A field map without the structure on it is a picture of colour —
 * the whole question being asked is how the field sits relative to the sheet.
 */
export interface MapOverlays {
  /** Line segments. `structure` is the ribbon itself; `trajectory` an electron path. */
  segments?: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    kind?: 'structure' | 'trajectory';
  }[];
  /** A point source, e.g. a dipole. `axis` draws its orientation. */
  markers?: { x: number; y: number; label?: string; axis?: 'x' | 'y' | 'z' }[];
}

export interface MapData {
  xs: number[];
  ys: number[];
  z: (number | null)[][];
  xLabel: string;
  yLabel: string;
  log?: boolean;
  /** Curves over the map. `color` and `dash` default to the house style below
   *  -- white, separated by dash pattern -- and are set only when a track is a
   *  DIFFERENT KIND of thing from the rest, not merely the next one along. The
   *  LDOS does both at once: its own peak tracks, read off the map, and the
   *  band energies Level 2 root-finds without ever seeing it. Dash alone
   *  cannot say which is which, because the peaks have already spent it. */
  tracks?: { xs: number[]; ys: (number | null)[]; color?: string; dash?: number[] }[];
  overlays?: MapOverlays;
}

type Cmap = (t: number) => [number, number, number];

function viridisRgb(t: number): [number, number, number] {
  const [r, g, b] = viridis(clamp01(t));
  return [r * 255, g * 255, b * 255];
}

/* The signed ramp is fixed rather than themed: the same number has to read as
 * the same colour in both themes, and the neutral midpoint has to stay the
 * perceptual centre of the ramp. Stops match the heterostructures k-map so the
 * site's two signed maps are directly comparable. */
function divergingRgb(t: number): [number, number, number] {
  const u = clamp01(t);
  if (u < 0.5) {
    const f = u / 0.5;
    return [37 + 210 * f, 99 + 148 * f, 235 + 12 * f];
  }
  const f = (u - 0.5) / 0.5;
  return [247 - 27 * f, 247 - 209 * f, 247 - 209 * f];
}

const LUT_N = 256;

interface MapStyle {
  cmap: Cmap;
  /** Map [-max|z|, +max|z|] so zero is always the midpoint. */
  symmetric: boolean;
}

/**
 * Draw the structure and the source over a field map.
 *
 * Black on a white halo, deliberately: these have to read on top of viridis at
 * its darkest AND at its brightest, and no single flat colour does. The halo is
 * stroked first and the black over it, which is the cheapest way to get an
 * outline that survives any colormap underneath.
 */
function drawMapOverlays(
  ctx: CanvasRenderingContext2D,
  fr: { px: (v: number) => number; py: (v: number) => number; pw: number; ph: number },
  box: { l: number; t: number },
  ov: MapOverlays,
): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(box.l, box.t, fr.pw, fr.ph);
  ctx.clip();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const seg of ov.segments ?? []) {
    const x0 = fr.px(seg.x0);
    const y0 = fr.py(seg.y0);
    const x1 = fr.px(seg.x1);
    const y1 = fr.py(seg.y1);
    const trajectory = seg.kind === 'trajectory';
    for (const pass of [0, 1]) {
      ctx.beginPath();
      ctx.setLineDash(trajectory ? [7, 5] : []);
      ctx.lineWidth = pass === 0 ? (trajectory ? 4.5 : 6) : trajectory ? 1.8 : 3;
      ctx.strokeStyle = pass === 0 ? 'rgba(255,255,255,0.9)' : '#000';
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (trajectory) {
      // An arrowhead, because a trajectory has a direction and a bare dashed
      // line does not say which way the electron goes.
      const a = Math.atan2(y1 - y0, x1 - x0);
      const h = 7;
      for (const pass of [0, 1]) {
        ctx.beginPath();
        ctx.lineWidth = pass === 0 ? 4 : 1.8;
        ctx.strokeStyle = pass === 0 ? 'rgba(255,255,255,0.9)' : '#000';
        ctx.moveTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4));
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4));
        ctx.stroke();
      }
    }
  }

  for (const m of ov.markers ?? []) {
    const x = fr.px(m.x);
    const y = fr.py(m.y);
    ctx.beginPath();
    ctx.arc(x, y, 5.5, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, 2 * Math.PI);
    ctx.fillStyle = '#000';
    ctx.fill();
    // The orientation, drawn the same way the geometry schematic draws it:
    // an arrow along the moment for x and z, the circle-and-dot for y, which
    // points out of this section. A white casing keeps it readable on top of
    // a dark patch of the map.
    if (m.axis) {
      const r = 10;
      ctx.lineCap = 'round';
      for (const [col, wide] of [['rgba(255,255,255,0.92)', 4], ['#000', 1.6]] as const) {
        ctx.strokeStyle = col;
        ctx.lineWidth = wide;
        if (m.axis === 'y') {
          ctx.beginPath();
          ctx.arc(x, y, 7.5, 0, 2 * Math.PI);
          ctx.stroke();
        } else {
          const dx = m.axis === 'x' ? r : 0;
          const dy = m.axis === 'z' ? r : 0;
          arrow(ctx, x, y, x + dx, y + dy, 4);
          arrow(ctx, x, y, x - dx, y - dy, 4);
        }
      }
      ctx.lineWidth = 1;
    }
    if (m.label) {
      ctx.font = '600 10px ' + getComputedStyle(document.body).fontFamily;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255,255,255,0.92)';
      ctx.strokeText(m.label, x + 8, y - 5);
      ctx.fillStyle = '#000';
      ctx.fillText(m.label, x + 8, y - 5);
    }
  }
  ctx.restore();
}

function renderMap(canvas: HTMLCanvasElement, d: MapData, style: MapStyle): void {
  const s = surface(canvas);
  if (!s) return;
  const { ctx, w, h, dpr } = s;
  const t = chartTheme();
  const box: Box = { l: 60, r: 84, t: 16, b: 42 };
  const pw = w - box.l - box.r;
  const ph = h - box.t - box.b;
  if (pw < 40 || ph < 40) return;

  const nx = d.xs.length, ny = d.ys.length;
  if (nx === 0 || ny === 0 || d.z.length === 0) { message(ctx, w, h, 'no data'); return; }

  // A log colour scale is meaningless for signed data, so it is ignored there.
  const log = !style.symmetric && d.log === true;

  // Flatten once. The pixel loop below runs over the whole plot rectangle at
  // device resolution, and must not re-walk nested arrays or re-test for null.
  const vals = new Float64Array(nx * ny);
  const ok = new Uint8Array(nx * ny);
  let any = false;
  for (let i = 0; i < nx; i++) {
    const row: (number | null)[] | undefined = d.z[i];
    if (!row) continue;
    for (let j = 0; j < ny; j++) {
      const v = row[j];
      if (v === null || !Number.isFinite(v)) continue;
      // Zero and negative values have no log; they are absent from the scale,
      // not at the bottom of it.
      if (log && !(v > 0)) continue;
      vals[i * ny + j] = log ? Math.log10(v) : v;
      ok[i * ny + j] = 1;
      any = true;
    }
  }
  if (!any) { message(ctx, w, h, 'no finite data'); return; }

  let lo: number, hi: number;
  if (style.symmetric) {
    let m = 0;
    for (let k = 0; k < vals.length; k++) if (ok[k] === 1) m = Math.max(m, Math.abs(vals[k]));
    lo = -m;
    hi = m;
  } else {
    lo = Infinity;
    hi = -Infinity;
    for (let k = 0; k < vals.length; k++) {
      if (ok[k] !== 1) continue;
      if (vals[k] < lo) lo = vals[k];
      if (vals[k] > hi) hi = vals[k];
    }
    // Same guard drawSpectrum uses on a log y axis: one near-zero cell would
    // otherwise stretch the ramp over so many decades that everything else
    // collapses into a single colour. An LDOS map is exactly that shape.
    if (log) lo = Math.max(lo, hi - LOG_DECADES);
  }
  // A flat map — including the all-zero asymmetry of a perfectly reciprocal
  // system — maps to the midpoint of the ramp rather than to an edge.
  const inv = hi > lo ? 1 / (hi - lo) : 0;

  const lut = new Uint8Array(LUT_N * 3);
  for (let k = 0; k < LUT_N; k++) {
    const [r, g, b] = style.cmap(k / (LUT_N - 1));
    lut[k * 3] = r;
    lut[k * 3 + 1] = g;
    lut[k * 3 + 2] = b;
  }
  const [nr, ng, nb] = rgbOf(cssVar('--text-muted'), NO_DATA_FALLBACK);

  const [xLo, xHi] = extent(d.xs);
  const [yLo, yHi] = extent(d.ys);

  const imgW = Math.max(1, Math.round(pw * dpr));
  const imgH = Math.max(1, Math.round(ph * dpr));
  const img = ctx.createImageData(imgW, imgH);
  // Each pixel takes the node NEAREST IT IN DATA SPACE. Spreading the cells
  // evenly over the index range instead would only be right for a uniform
  // grid, and the +-q grid these maps run on is not one: symmetric_q_grid
  // mirrors a positive half that starts at q_min, so the gap across zero is
  // wider than the step (0.6 against 0.013 for q in [0.3, 0.5], 16 per side).
  // Evenly spaced columns would then paint cells at q values that were never
  // solved, and the peak tracks below — drawn through the real axis transform —
  // would sit beside the ridges they are tracking rather than on them.
  const iOf = nearestIndices(d.xs, xLo, xHi, imgW);
  const jOf = nearestIndices(d.ys, yLo, yHi, imgH);
  // Nearest-neighbour, never interpolated: a smoothed map would blend the
  // no-data colour into its finite neighbours and invent values at the edge of
  // a gap.
  for (let py = 0; py < imgH; py++) {
    const j = jOf[imgH - 1 - py];                        // canvas top = ys[last]
    for (let px = 0; px < imgW; px++) {
      const k = iOf[px] * ny + j;
      const off = (py * imgW + px) * 4;
      if (ok[k] === 1) {
        const ti = Math.round(clamp01(inv === 0 ? 0.5 : (vals[k] - lo) * inv) * (LUT_N - 1)) * 3;
        img.data[off] = lut[ti];
        img.data[off + 1] = lut[ti + 1];
        img.data[off + 2] = lut[ti + 2];
      } else {
        img.data[off] = nr;
        img.data[off + 1] = ng;
        img.data[off + 2] = nb;
      }
      img.data[off + 3] = 255;
    }
  }
  ctx.putImageData(img, Math.round(box.l * dpr), Math.round(box.t * dpr));

  const fr = axes(ctx, w, h, box, xLo, xHi, yLo, yHi, d.xLabel, d.yLabel,
    niceTicks(xLo, xHi, 6), niceTicks(yLo, yHi, 5), fmtLite, fmtLite, false);

  if (d.overlays) drawMapOverlays(ctx, fr, box, d.overlays);

  if (d.tracks && d.tracks.length > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.l, box.t, fr.pw, fr.ph);
    ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    // Light over a dark colormap, as on the BPM and k-map panels. Dash pattern,
    // not colour, separates tracks of the SAME kind, so the map keeps the only
    // hue -- until a track says otherwise, which is how a second family gets
    // told apart from the first.
    d.tracks.forEach((tr, ti) => {
      ctx.strokeStyle = tr.color ?? 'rgba(255,255,255,0.85)';
      ctx.setLineDash(tr.dash ?? (ti === 0 ? [] : ti === 1 ? [5, 3] : [2, 3]));
      ctx.beginPath();
      let run = 0;
      const m = Math.min(tr.xs.length, tr.ys.length);
      const live = (y: number | null | undefined, x: number) =>
        y !== null && y !== undefined && Number.isFinite(y) && Number.isFinite(x);
      for (let i = 0; i < m; i++) {
        const v = tr.ys[i];
        if (!live(v, tr.xs[i])) { run = 0; continue; }
        const X = fr.px(tr.xs[i]);
        const Y = fr.py(v as number);
        if (run === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        run++;
        // A lone sample between two gaps has no segment to draw, so without
        // this the whole band disappears from the overlay -- no line, no gap,
        // no dot. drawSpectrum has always marked these; this renderer did not,
        // and a tracked band structure makes the case ordinary, because a
        // branch that leaves the energy window leaves holes around whatever
        // of it survives inside.
        if (run === 1 && !live(tr.ys[i + 1], tr.xs[i + 1])) {
          ctx.fillStyle = ctx.strokeStyle as string;
          ctx.fillRect(X - 1.5, Y - 1.5, 3, 3);
        }
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.restore();
  }

  /* ---- colorbar ---- */
  const cbW = 12;
  const cbX = w - box.r + 14;
  const cbT = box.t;
  const cbH = Math.round(Math.max(20, fr.ph - 26));     // room for the no-data swatch below
  for (let i = 0; i < cbH; i++) {
    const [r, g, b] = style.cmap(1 - i / Math.max(1, cbH - 1));
    ctx.fillStyle = `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
    ctx.fillRect(cbX, cbT + i, cbW, 1);
  }
  ctx.strokeStyle = t.textMuted;
  ctx.lineWidth = 0.5;
  ctx.strokeRect(cbX, cbT, cbW, cbH);

  ctx.fillStyle = t.textSoft;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  // Labels come from the lo/hi actually drawn; under a log scale they are the
  // physical values, so the numbers on the bar match the quantity, not its log.
  const nLab = 5;
  for (let i = 0; i < nLab; i++) {
    const f = i / (nLab - 1);
    const v = hi > lo ? lo + f * (hi - lo) : lo;
    ctx.fillText(fmtLite(log ? 10 ** v : v), cbX + cbW + 4, cbT + (1 - f) * cbH);
  }

  ctx.fillStyle = cssVar('--text-muted');
  ctx.fillRect(cbX, cbT + cbH + 10, cbW, 10);
  ctx.strokeRect(cbX, cbT + cbH + 10, cbW, 10);
  ctx.fillStyle = t.textSoft;
  ctx.fillText('n/d', cbX + cbW + 4, cbT + cbH + 15);
}

/** False-colour (q, E) map for positive-definite data. */
export function drawHeatmap(canvas: HTMLCanvasElement, d: MapData): void {
  renderMap(canvas, d, { cmap: viridisRgb, symmetric: false });
}

/**
 * False-colour (q, E) map for signed data — the +-q asymmetry.
 *
 * The scale is symmetric about zero by construction. Centring it on the data
 * range instead would paint a perfectly reciprocal system (asymmetry
 * identically zero, or zero to round-off) as strongly asymmetric, which is the
 * exact error this map exists to rule out.
 */
export function drawDivergingHeatmap(canvas: HTMLCanvasElement, d: MapData): void {
  renderMap(canvas, d, { cmap: divergingRgb, symmetric: true });
}

/* ===== Ribbon schematic ===== */

export interface GeometryPayload {
  n_ribbons: number;
  n_nodes_total: number;
  dx_nm: number;
  eps_eff: number;
  ribbons: {
    W_nm: number; x0_nm: number; z0_nm: number; EF_eV: number;
    gamma_eV: number; beta: number; L: number; h_nm: number;
    x_nodes_nm: number[]; theta: number[];
  }[];
  layers: { eps_par: number; d_nm: number }[];
}

export interface SchematicSource {
  kind: 'plane_wave' | 'dipole' | 'electron';
  r0_nm?: [number, number, number];
  /** The dipole moment's direction. Drawn, so a y dipole does not look like
   *  an x one -- the three choices behave completely differently. */
  p?: [number, number, number];
  b_nm?: number;
}

function arrow(
  ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, head = 5,
): void {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4));
  ctx.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4));
  ctx.closePath();
  ctx.fill();
}

/**
 * x-z section of the ribbon stack.
 *
 * A ribbon is two-dimensional: it is drawn as the line of discretisation nodes
 * that represents it, at its own z0, and never as a rectangle, slab or filled
 * band — a drawn thickness would assert a third dimension the model does not
 * have. Both axes carry the same nm per pixel for the same reason: a stretched
 * z turns a 1 nm standoff into a slab.
 */
export function drawRibbonSchematic(
  canvas: HTMLCanvasElement, g: GeometryPayload, source?: SchematicSource,
): void {
  const s = surface(canvas);
  if (!s) return;
  const { ctx, w, h } = s;
  const t = chartTheme();
  const box: Box = { l: 56, r: 18, t: 16, b: 40 };
  const pw = w - box.l - box.r;
  const ph = h - box.t - box.b;
  if (pw < 60 || ph < 60) return;

  let xLo = Infinity, xHi = -Infinity, zLo = Infinity, zHi = -Infinity;
  for (const r of g.ribbons) {
    for (const xn of r.x_nodes_nm) {
      if (!Number.isFinite(xn)) continue;
      if (xn < xLo) xLo = xn;
      if (xn > xHi) xHi = xn;
    }
    if (Number.isFinite(r.z0_nm)) {
      zLo = Math.min(zLo, r.z0_nm);
      zHi = Math.max(zHi, r.z0_nm);
    }
  }
  if (!Number.isFinite(xLo) || !Number.isFinite(zLo)) { message(ctx, w, h, 'no geometry'); return; }

  let eHeight: number | null = null;
  if (source && source.kind === 'electron') eHeight = source.b_nm ?? source.r0_nm?.[2] ?? 0;
  if (eHeight !== null && Number.isFinite(eHeight)) {
    zLo = Math.min(zLo, eHeight);
    zHi = Math.max(zHi, eHeight);
  }
  if (source?.kind === 'dipole' && source.r0_nm) {
    // r0 is (x, y, z); y is the translationally invariant direction and has no
    // place in an x-z section.
    const [dx, , dz] = source.r0_nm;
    if (Number.isFinite(dx)) { xLo = Math.min(xLo, dx); xHi = Math.max(xHi, dx); }
    if (Number.isFinite(dz)) { zLo = Math.min(zLo, dz); zHi = Math.max(zHi, dz); }
  }

  const spanX = Math.max(xHi - xLo, 1e-9);
  const subD = g.layers.reduce((a, l) => a + (Number.isFinite(l.d_nm) && l.d_nm > 0 ? l.d_nm : 0), 0);
  // Show enough substrate for a half-space to read as one. Widening the shown
  // RANGE is honest; stretching the z scale to fit it would not be.
  if (g.layers.length > 0) {
    const shown = Math.max(subD, 0.25 * spanX);
    zLo = Math.min(zLo, -shown);
    // Headroom ABOVE the sheet, not merely up to it. `Math.max(zHi, 0)` pinned
    // the ribbon to the frame's top edge, which is also where the width label,
    // the source caption and the incidence arrows all have to go -- so with a
    // substrate every one of them landed on the sheet and on each other. Half
    // the shown depth clears them and still reads as far more medium below
    // than above, which is the thing the picture has to say.
    zHi = Math.max(zHi, 0.5 * shown);
  }
  if (zHi - zLo < 0.2 * spanX) {
    const mid = (zHi + zLo) / 2;
    zLo = mid - 0.1 * spanX;
    zHi = mid + 0.1 * spanX;
  }
  const mx = 0.06 * spanX;
  const mz = 0.06 * (zHi - zLo);
  xLo -= mx; xHi += mx; zLo -= mz; zHi += mz;

  // One nm-per-pixel for both axes, then grow the slack axis's RANGE until that
  // single scale fills the box exactly. Doing it this way keeps the drawing and
  // the ticks axes() lays down below on one and the same mapping — a frame
  // stretched to the data range would print x and z labels that do not describe
  // where anything was actually drawn.
  const sc = Math.min(pw / (xHi - xLo), ph / (zHi - zLo));
  const xPad = (pw / sc - (xHi - xLo)) / 2;
  const zPad = (ph / sc - (zHi - zLo)) / 2;
  xLo -= xPad; xHi += xPad; zLo -= zPad; zHi += zPad;
  const px = (x: number) => box.l + (x - xLo) * sc;
  const pz = (z: number) => box.t + ph - (z - zLo) * sc;
  const top = box.t, bot = box.t + ph, left = box.l, right = box.l + pw;

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, pw, ph);
  ctx.clip();

  /* ---- substrate stack, as a half-space below z = 0 ----
     layers[0] is NOT a substrate layer. By the solver's convention it is the
     half-space the ribbons themselves sit in (its eps_par must equal eps_a and
     its d_nm must be zero), so the stack to shade below z = 0 is layers[1:].
     Drawing index 0 here shaded the ambient medium as if it were glass. */
  const substrate = g.layers.slice(1);
  let zTop = 0;
  substrate.forEach((l, i) => {
    const last = i === substrate.length - 1;
    const thick = Number.isFinite(l.d_nm) && l.d_nm > 0 ? l.d_nm : null;
    const yTop = Math.max(top, pz(zTop));
    const yBot = last || thick === null ? bot : Math.min(bot, pz(zTop - thick));
    if (yBot > yTop) {
      ctx.fillStyle = i % 2 === 0 ? t.grid : cssVar('--bg-soft');
      ctx.fillRect(left, yTop, pw, yBot - yTop);
      ctx.strokeStyle = t.textMuted;
      ctx.lineWidth = 0.75;
      ctx.beginPath();
      ctx.moveTo(left, yTop);
      ctx.lineTo(right, yTop);
      ctx.stroke();
      if (yBot - yTop > 13) {
        ctx.fillStyle = t.textSoft;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(`ε∥ = ${fmtLite(l.eps_par)}`, left + 5, yTop + 3);
      }
    }
    if (thick !== null) zTop -= thick;
  });

  /* ---- ribbons: a line of nodes, no thickness ---- */
  const nodeR = Math.max(0.9, Math.min(2.6, 0.45 * g.dx_nm * sc));
  g.ribbons.forEach((r, i) => {
    const col = seriesColor(i);
    const Y = pz(r.z0_nm);
    ctx.fillStyle = col;
    ctx.beginPath();
    for (const xn of r.x_nodes_nm) {
      if (!Number.isFinite(xn)) continue;
      const X = px(xn);
      ctx.moveTo(X + nodeR, Y);                        // moveTo per node: no connecting stroke
      ctx.arc(X, Y, nodeR, 0, 2 * Math.PI);
    }
    ctx.fill();

    if (g.ribbons.length <= 4 && r.x_nodes_nm.length > 0) {
      ctx.fillStyle = col;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`W = ${fmtLite(r.W_nm)} nm`, px(r.x0_nm), Y - nodeR - 4);
    }

    // Drift is a bias along the ribbon, not a frequency shift; the arrow only
    // states its direction.
    if (r.beta !== 0 && Number.isFinite(r.beta)) {
      const len = Math.min(28, Math.max(12, 0.3 * r.W_nm * sc));
      const dir = r.beta > 0 ? 1 : -1;
      const yArr = Y + nodeR + 9;
      ctx.strokeStyle = t.textSoft;
      ctx.fillStyle = t.textSoft;
      ctx.lineWidth = 1;
      arrow(ctx, px(r.x0_nm) - dir * len / 2, yArr, px(r.x0_nm) + dir * len / 2, yArr, 4);
      if (i === 0) {
        ctx.textAlign = dir > 0 ? 'left' : 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText('drift', px(r.x0_nm) + dir * (len / 2 + 4), yArr);
      }
    }
  });

  /* ---- source ---- */
  if (source) {
    ctx.strokeStyle = t.text;
    ctx.fillStyle = t.text;
    ctx.lineWidth = 1.2;
    if (source.kind === 'plane_wave') {
      for (let i = 0; i < 3; i++) {
        const X = left + pw * (0.25 + 0.25 * i);
        arrow(ctx, X, top + 18, X, top + 38);
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('plane wave  k ↓  E ∥ x', left + 5, top + 3);
    } else if (source.kind === 'electron' && eHeight !== null) {
      // Aloof and parallel: a straight trajectory at constant height, never
      // crossing a ribbon.
      const Y = pz(eHeight);
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(left, Y);
      ctx.lineTo(right - 14, Y);
      ctx.stroke();
      ctx.setLineDash([]);
      arrow(ctx, right - 16, Y, right - 4, Y, 5);
      ctx.textAlign = 'left';
      // Above the trajectory when there is room for it, below when there is
      // not. The electron usually runs at the very top of the shown range --
      // it is the highest thing in the picture -- so "above" clipped it
      // against the frame and the label simply vanished.
      const above = Y - 4 - 11 >= top;
      ctx.textBaseline = above ? 'bottom' : 'top';
      ctx.fillText(`e⁻  b = ${fmtLite(eHeight)} nm`, left + 5, Y + (above ? -4 : 4));
    } else if (source.kind === 'dipole' && source.r0_nm) {
      const X = px(source.r0_nm[0]);
      const Y = pz(source.r0_nm[2]);
      // Two-headed along the moment's own axis. y is the translationally
      // invariant direction, so a y dipole points out of this section and is
      // drawn as the circle-and-dot that convention reserves for it rather
      // than as an arrow it cannot have here.
      const p = source.p ?? [1, 0, 0];
      const ax = Math.abs(p[0]) >= Math.abs(p[2]) && Math.abs(p[0]) >= Math.abs(p[1])
        ? 'x'
        : Math.abs(p[2]) >= Math.abs(p[1])
          ? 'z'
          : 'y';
      if (ax === 'y') {
        ctx.beginPath();
        ctx.arc(X, Y, 7, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(X, Y, 1.8, 0, 2 * Math.PI);
        ctx.fill();
      } else if (ax === 'x') {
        arrow(ctx, X, Y, X - 11, Y, 4);
        arrow(ctx, X, Y, X + 11, Y, 4);
      } else {
        arrow(ctx, X, Y, X, Y - 11, 4);
        arrow(ctx, X, Y, X, Y + 11, 4);
      }
      ctx.beginPath();
      ctx.arc(X, Y, 2, 0, 2 * Math.PI);
      ctx.fill();
      // Clear of whatever was just drawn, not 6px from the centre: the x
      // arrow reaches 11px and the y circle 7, so a fixed small offset put
      // the word straight through the symbol it names. Flipped to the left
      // when the right side has no room for it.
      ctx.textBaseline = 'middle';
      const lead = ax === 'x' ? 15 : ax === 'y' ? 12 : 8;
      const wide = ctx.measureText('dipole').width;
      const toRight = X + lead + wide <= right - 2;
      ctx.textAlign = toRight ? 'left' : 'right';
      ctx.fillText('dipole', X + (toRight ? lead : -lead), Y);
    }
  }

  /* ---- scale bar ---- */
  const barNm = niceLength(0.3 * (xHi - xLo));
  const barPx = barNm * sc;
  if (barPx > 16 && barPx < pw - 20) {
    const bx = right - 12 - barPx;
    const by = bot - 12;
    ctx.strokeStyle = t.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx + barPx, by);
    ctx.moveTo(bx, by - 3);
    ctx.lineTo(bx, by + 3);
    ctx.moveTo(bx + barPx, by - 3);
    ctx.lineTo(bx + barPx, by + 3);
    ctx.stroke();
    ctx.fillStyle = t.text;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${fmtLite(barNm)} nm`, bx + barPx / 2, by - 5);
  }
  ctx.restore();

  axes(ctx, w, h, box, xLo, xHi, zLo, zHi, 'x (nm)', 'z (nm)',
    niceTicks(xLo, xHi, 6), niceTicks(zLo, zHi, 5), fmtLite, fmtLite, false);

  // The "N ribbons / M nodes / dx / eps_eff" line used to be printed here, at
  // the top of the frame, where it crowded the source caption and the width
  // label. It is four numbers, not a picture, so it belongs with the other
  // numbers: the stats row under this card carries it now.
}

/* ===== Mode profiles ===== */

/**
 * Eigenmode charge profiles, one line per mode over its own ribbon's nodes.
 *
 * The y range is symmetric about zero and a zero rule is drawn, so an even
 * profile (bright: non-zero dipole overlap) and an odd one (dark: the lobes
 * cancel) are told apart by shape rather than by the legend.
 */
/**
 * Eigenmode profiles across the structure.
 *
 * `profiles[m][j]` is mode m on ribbon j, and `xNodes[j]` is that ribbon's own
 * x grid: one polyline per (mode, ribbon). The indices are NOT interchangeable.
 * An earlier version took a flat profile per mode and reached for `xNodes[m]`,
 * which quietly plotted the second mode against the second RIBBON's
 * coordinates as soon as there was more than one ribbon.
 *
 * Each ribbon is stroked separately rather than as one polyline, so the gap
 * between two ribbons stays a gap instead of a line across empty space.
 */
export function drawModeProfiles(canvas: HTMLCanvasElement, d: {
  xNodes: number[][]; profiles: number[][][]; labels: string[];
}): void {
  const s = surface(canvas);
  if (!s) return;
  const { ctx, w, h } = s;
  const t = chartTheme();
  const box = SPEC_PAD;
  if (w - box.l - box.r < 30 || h - box.t - box.b < 30) return;
  if (d.profiles.length === 0) { message(ctx, w, h, 'no modes'); return; }

  let xLo = Infinity, xHi = -Infinity, m = 0;
  for (const mode of d.profiles) {
    for (let j = 0; j < mode.length; j++) {
      const xn = d.xNodes[j];
      const pf = mode[j];
      if (xn === undefined || pf === undefined) continue;
      const n = Math.min(xn.length, pf.length);
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(xn[i]) || !Number.isFinite(pf[i])) continue;
        if (xn[i] < xLo) xLo = xn[i];
        if (xn[i] > xHi) xHi = xn[i];
        m = Math.max(m, Math.abs(pf[i]));
      }
    }
  }
  if (!Number.isFinite(xLo)) { message(ctx, w, h, 'no finite data'); return; }

  const [axLo, axHi] = padRange(xLo, xHi);
  const yAbs = m > 0 ? m * 1.1 : 1;
  const fr = axes(ctx, w, h, box, axLo, axHi, -yAbs, yAbs, 'x (nm)', 'surface charge (arb.)',
    niceTicks(axLo, axHi, 6), niceTicks(-yAbs, yAbs, 4), fmtLite, fmtLite);

  ctx.save();
  ctx.beginPath();
  ctx.rect(box.l, box.t, fr.pw, fr.ph);
  ctx.clip();

  ctx.strokeStyle = t.textMuted;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(box.l, fr.py(0));
  ctx.lineTo(box.l + fr.pw, fr.py(0));
  ctx.stroke();

  for (let k = 0; k < d.profiles.length; k++) {
    const mode = d.profiles[k];
    if (mode === undefined) continue;
    ctx.strokeStyle = seriesColor(k);
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    for (let j = 0; j < mode.length; j++) {
      const xn = d.xNodes[j];
      const pf = mode[j];
      if (xn === undefined || pf === undefined) continue;
      ctx.beginPath();
      let run = 0;
      const n = Math.min(xn.length, pf.length);
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(xn[i]) || !Number.isFinite(pf[i])) { run = 0; continue; }
        const X = fr.px(xn[i]);
        const Y = fr.py(pf[i]);
        if (run === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        run++;
      }
      ctx.stroke();
    }
  }
  ctx.restore();

  // Mode profiles have no HTML legend of their own, and the plot corners are
  // empty for a profile that is centred on its ribbon.
  if (d.labels.length > 0) {
    const lh = 13;
    const bw = 76;
    const rows = Math.min(d.labels.length, d.profiles.length, Math.floor((fr.ph - 16) / lh));
    const bx = box.l + fr.pw - bw - 6;
    const by = box.t + 6;
    if (rows > 0) {
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = t.bg;
      ctx.fillRect(bx, by, bw, rows * lh + 4);
      ctx.globalAlpha = 1;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      for (let k = 0; k < rows; k++) {
        const y = by + 2 + lh * k + lh / 2;
        ctx.fillStyle = seriesColor(k);
        ctx.fillRect(bx + 4, y - 1.5, 12, 3);
        ctx.fillStyle = t.textSoft;
        ctx.fillText(d.labels[k], bx + 20, y);
      }
    }
  }
}
