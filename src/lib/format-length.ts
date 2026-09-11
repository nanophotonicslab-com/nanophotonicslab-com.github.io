/**
 * Display formatting for a physical length: always one decimal place, always at
 * least two significant figures, in whichever unit makes both true.
 *
 * The unit is chosen so the displayed value lands in [1, 1000). Below 1, a
 * single decimal leaves one significant figure or none (0.3 mm, 0.0 mm); from
 * 1000 up, the next unit reads better. With one decimal, any value >= 1 carries
 * at least two significant figures, so the range and the precision rule are the
 * same constraint.
 *
 * The unit is chosen from the *raw* length — the largest unit in which it is at
 * least 1 — so no precision is thrown away: 960 mm stays 960.0 mm rather than
 * rounding up to 1.0 m. Only when rounding to one decimal would print 1000.0
 * does the length move up a unit: 999.96 mm is 1.0 m, and 999.96 nm is 1.0 µm.
 *
 * Pure and DOM-free, like the other src/lib modules, so it is unit-tested.
 */

export interface FormattedLength {
  /** The number, with exactly one decimal place — or '—' when there is no length to show. */
  value: string;
  /** 'pm' | 'nm' | 'µm' | 'mm' | 'm', or '' alongside '—'. */
  unit: string;
}

/** Largest first. */
const LENGTH_UNITS: readonly (readonly [string, number])[] = [
  ['m', 1],
  ['mm', 1e-3],
  ['µm', 1e-6],
  ['nm', 1e-9],
  ['pm', 1e-12],
];

const round1 = (v: number) => Math.round(v * 10) / 10;

export function formatLength(meters: number): FormattedLength {
  if (!Number.isFinite(meters) || meters <= 0) return { value: '—', unit: '' };
  // The natural unit: the largest in which the raw length is at least 1. Choosing
  // on the raw value, not a value already rounded in a coarser unit, is what
  // keeps 960 mm from printing as 1.0 m.
  let i = LENGTH_UNITS.findIndex(([, scale]) => meters / scale >= 1);
  // Below 1 pm, which no length on this site reaches: stay in pm. This is the
  // one case where a single decimal can no longer carry two significant figures.
  if (i === -1) i = LENGTH_UNITS.length - 1;
  let v = round1(meters / LENGTH_UNITS[i][1]);
  // Rounding can carry 999.96 up to 1000.0; that length belongs one unit up.
  if (v >= 1000 && i > 0) {
    i -= 1;
    v = round1(meters / LENGTH_UNITS[i][1]);
  }
  return { value: v.toFixed(1), unit: LENGTH_UNITS[i][0] };
}

/** The same, as one string: "200.6 mm". */
export function formatLengthText(meters: number): string {
  const { value, unit } = formatLength(meters);
  return unit ? `${value} ${unit}` : value;
}
