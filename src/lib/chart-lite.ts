/**
 * Minimal shared line-chart renderer for tool pages (canvas 2D, DPR-aware,
 * linear axes with negative-y support, dashed series, crosshair + HTML tooltip).
 * View-layer utility — physics stays in the compute libs.
 */
import { chartTheme } from './chart-theme';

export interface LiteSeries { y: Float64Array; label: string; color: string; dash?: number[]; }
export interface LiteChart {
  x: Float64Array;
  series: LiteSeries[];
  xLabel: string;
  xUnit: string;
  hoverI: number;
}

export function fmtLite(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e4 || a < 1e-2) return v.toExponential(1).replace('e+', 'e');
  return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : v.toFixed(2);
}

function niceTicks(min: number, max: number, n = 5): number[] {
  const span = max - min || 1;
  const mag = 10 ** Math.floor(Math.log10(span / n));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 1) || mag * 10;
  const out: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) out.push(t);
  return out;
}

export const LITE_PAD = { l: 46, r: 10, t: 8, b: 30 };

export function drawLite(canvas: HTMLCanvasElement, st: LiteChart): void {
  const t = chartTheme();
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (W === 0) return;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const PAD = LITE_PAD;
  const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
  const xmin = st.x[0], xmax = st.x[st.x.length - 1];
  let ymin = 0, ymax = 0;
  for (const s of st.series) for (const v of s.y) { if (v > ymax) ymax = v; if (v < ymin) ymin = v; }
  if (ymax <= ymin) ymax = ymin + 1;
  const tx = (v: number) => PAD.l + ((v - xmin) / (xmax - xmin)) * pw;
  const ty = (v: number) => PAD.t + ph - ((v - ymin) / (ymax - ymin)) * ph;
  ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
  ctx.strokeStyle = t.grid; ctx.fillStyle = t.textMuted; ctx.lineWidth = 1;
  for (const yt of niceTicks(ymin, ymax, 4)) {
    ctx.beginPath(); ctx.moveTo(PAD.l, ty(yt)); ctx.lineTo(W - PAD.r, ty(yt)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(fmtLite(yt), PAD.l - 5, ty(yt));
  }
  for (const xt of niceTicks(xmin, xmax, 6)) {
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(fmtLite(xt), tx(xt), PAD.t + ph + 6);
  }
  ctx.fillStyle = t.textSoft; ctx.textAlign = 'center';
  ctx.fillText(st.xLabel, PAD.l + pw / 2, H - 12);
  for (const s of st.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    for (let i = 0; i < st.x.length; i++) {
      i === 0 ? ctx.moveTo(tx(st.x[i]), ty(s.y[i])) : ctx.lineTo(tx(st.x[i]), ty(s.y[i]));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (st.hoverI >= 0) {
    const hx = tx(st.x[st.hoverI]);
    ctx.strokeStyle = t.textMuted; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(hx, PAD.t); ctx.lineTo(hx, PAD.t + ph); ctx.stroke();
    ctx.setLineDash([]);
    for (const s of st.series) {
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(hx, ty(s.y[st.hoverI]), 3, 0, 2 * Math.PI); ctx.fill();
    }
  }
}

export function legendHtml(series: LiteSeries[]): string {
  return series.map((s) => `<span><i style="--c:${s.color}"></i>${s.label}</span>`).join('');
}

export function attachLiteHover(
  canvas: HTMLCanvasElement, tip: HTMLElement,
  get: () => LiteChart | undefined,
): void {
  canvas.addEventListener('pointermove', (ev) => {
    const st = get();
    if (!st) return;
    const rect = canvas.getBoundingClientRect();
    const rel = (ev.clientX - rect.left - LITE_PAD.l) / (rect.width - LITE_PAD.l - LITE_PAD.r);
    st.hoverI = Math.max(0, Math.min(st.x.length - 1, Math.round(rel * (st.x.length - 1))));
    drawLite(canvas, st);
    tip.style.display = 'block';
    tip.style.left = Math.min(ev.clientX - rect.left + 10, rect.width - 150) + 'px';
    tip.style.top = '6px';
    tip.innerHTML = `${fmtLite(st.x[st.hoverI])} ${st.xUnit}<br>`
      + st.series.map((s) => `${s.label}: ${fmtLite(s.y[st.hoverI])}`).join('<br>');
  });
  canvas.addEventListener('pointerleave', () => {
    const st = get();
    if (!st) return;
    st.hoverI = -1; tip.style.display = 'none';
    drawLite(canvas, st);
  });
}
