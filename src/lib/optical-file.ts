/**
 * Reading optical constants out of a user-supplied text file.
 *
 * Every Lab tool that accepts an uploaded material needs the same three steps:
 * split the text into numeric rows, guess how to read the columns, and convert
 * them into the canonical table. That logic had been copied into four pages
 * (mie-scattering, photothermal, plasmonic-nanoparticles, materials) and had
 * already drifted — materials.astro uses different option values for the very
 * same choices. This module is the single copy; it is deliberately free of DOM
 * so it can be unit-tested like the physics kernels.
 *
 * The canonical table is the one `MATERIALS[x].data` uses and the only shape
 * `interpolateNK` accepts: `[[λ_µm, n, k], …]` sorted ascending by wavelength.
 */
import { epsilonToNK, PHOTON_HC_EV_NM } from './materials';

/** How to read the first column. */
export type ImportAxisMode = 'wavelength-nm' | 'wavelength-um' | 'energy-ev';
/** How to read the second and third columns. */
export type ImportValueMode = 'nk' | 'epsilon';

export type RawOpticalRow = [number, number, number];

export interface ParsedOpticalFile {
  rows: RawOpticalRow[];
  /** Rows that were not three finite numbers, and were dropped. */
  skipped: number;
}

export interface ConvertedOpticalFile {
  /** `[[λ_µm, n, k], …]`, ascending — ready for interpolateNK. */
  data: number[][];
  lambdaMinNm: number;
  lambdaMaxNm: number;
  nMin: number; nMax: number;
  kMin: number; kMax: number;
}

/**
 * Split a delimited text file into numeric rows.
 *
 * Comment lines (`#`, `%`) and blanks are dropped, the delimiter is sniffed as
 * tab, then comma, then any whitespace, and a non-numeric first row is treated
 * as a header. Nothing is interpreted here: the caller decides what the columns
 * mean, because a file of three numbers is genuinely ambiguous between
 * (λ, n, k) and (E, ε′, ε″).
 */
export function parseOpticalFile(text: string): ParsedOpticalFile | { error: string } {
  const lines = text.split(/\r?\n/).map(l => l.trim())
    .filter(l => l && !l.startsWith('#') && !l.startsWith('%'));
  if (lines.length < 2) return { error: 'File must have at least 2 data rows' };

  let delim = /\t/;
  if (lines[0].split(delim).length < 3) delim = /,/;
  if (lines[0].split(delim).length < 3) delim = /\s+/;

  const startIdx = Number.isNaN(parseFloat(lines[0].split(delim)[0])) ? 1 : 0;

  const rows: RawOpticalRow[] = [];
  let skipped = 0;
  for (let i = startIdx; i < lines.length; i++) {
    const parts = lines[i].split(delim).map(Number);
    if (parts.length < 3 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1]) || !Number.isFinite(parts[2])) {
      skipped++;
      continue;
    }
    rows.push([parts[0], parts[1], parts[2]]);
  }
  if (rows.length < 2) return { error: 'Could not parse at least 2 data rows with 3 numeric columns' };
  return { rows, skipped };
}

/** Convert raw rows to the canonical `[λ_µm, n, k]` table under a chosen reading. */
export function convertOpticalRows(
  rows: RawOpticalRow[], axisMode: ImportAxisMode, valueMode: ImportValueMode,
): ConvertedOpticalFile | { error: string } {
  const data: number[][] = [];
  let nMin = Infinity, nMax = -Infinity, kMin = Infinity, kMax = -Infinity;
  for (const [x, a, b] of rows) {
    if (x <= 0) continue;
    const lambdaUm = axisMode === 'energy-ev' ? PHOTON_HC_EV_NM / x / 1000
      : axisMode === 'wavelength-nm' ? x / 1000
        : x;
    const [n, k] = valueMode === 'epsilon' ? epsilonToNK(a, b) : [a, Math.abs(b)];
    if (!Number.isFinite(lambdaUm) || !Number.isFinite(n) || !Number.isFinite(k) || n < 0) continue;
    data.push([lambdaUm, n, k]);
    nMin = Math.min(nMin, n); nMax = Math.max(nMax, n);
    kMin = Math.min(kMin, k); kMax = Math.max(kMax, k);
  }
  if (data.length < 2) return { error: 'The selected settings do not produce at least 2 valid optical rows' };
  data.sort((p, q) => p[0] - q[0]);
  return {
    data,
    lambdaMinNm: data[0][0] * 1000,
    lambdaMaxNm: data[data.length - 1][0] * 1000,
    nMin, nMax, kMin, kMax,
  };
}

/**
 * Guess how the columns should be read, to preselect the confirmation dialog.
 *
 * A negative second column, or values far above what a refractive index reaches,
 * mean the file is a permittivity; a first column beyond 50 is nanometres rather
 * than micrometres, and a small first column alongside a permittivity is far more
 * likely to be photon energy than micrometres.
 */
export function inferImportModes(rows: RawOpticalRow[]): { axis: ImportAxisMode; values: ImportValueMode } {
  const xs = rows.map(r => r[0]).filter(Number.isFinite);
  const ys = rows.flatMap(r => [r[1], r[2]]).filter(Number.isFinite);
  const xMax = Math.max(...xs);
  const yMaxAbs = Math.max(...ys.map(Math.abs));
  const hasNegativeReal = rows.some(r => r[1] < 0);
  const values: ImportValueMode = hasNegativeReal || yMaxAbs > 25 ? 'epsilon' : 'nk';
  let axis: ImportAxisMode = 'wavelength-um';
  if (xMax > 50) axis = 'wavelength-nm';
  else if (values === 'epsilon' || xMax > 8) axis = 'energy-ev';
  return { axis, values };
}

/** Evenly spaced sample of at most `maxRows` entries, endpoints included. */
export function sampleRows<T>(rows: T[], maxRows: number): T[] {
  if (rows.length <= maxRows) return rows.slice();
  const out: T[] = [];
  for (let i = 0; i < maxRows; i++) out.push(rows[Math.round(i * (rows.length - 1) / (maxRows - 1))]);
  return out;
}

/** Compact fixed/exponential formatting for the preview table. */
export function formatCompact(v: number): string {
  if (!Number.isFinite(v)) return '';
  const av = Math.abs(v);
  if (av >= 1000 || (av > 0 && av < 0.01)) return v.toExponential(3);
  return v.toPrecision(5);
}
