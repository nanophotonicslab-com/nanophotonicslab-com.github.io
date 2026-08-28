import { describe, it, expect } from 'vitest';
import {
  parseOpticalFile, convertOpticalRows, inferImportModes, sampleRows, formatCompact,
  type RawOpticalRow,
} from './optical-file';
import { PHOTON_HC_EV_NM } from './materials';

const ok = <T>(r: T | { error: string }): T => {
  if (r && typeof r === 'object' && 'error' in (r as object)) throw new Error((r as { error: string }).error);
  return r as T;
};

describe('parseOpticalFile — delimiters, headers and junk', () => {
  it('reads comma, tab and whitespace delimited files alike', () => {
    const want = [[0.4, 1.5, 0.1], [0.5, 1.6, 0.2]];
    for (const text of [
      '0.4,1.5,0.1\n0.5,1.6,0.2',
      '0.4\t1.5\t0.1\n0.5\t1.6\t0.2',
      '0.4 1.5 0.1\n0.5   1.6 0.2',
    ]) {
      expect(ok(parseOpticalFile(text)).rows).toEqual(want);
    }
  });

  it('skips a non-numeric header row', () => {
    const r = ok(parseOpticalFile('lambda,n,k\n0.4,1.5,0.1\n0.5,1.6,0.2'));
    expect(r.rows).toHaveLength(2);
    expect(r.rows[0][0]).toBe(0.4);
  });

  it('drops comments and blank lines without counting them as skipped', () => {
    const r = ok(parseOpticalFile('# comment\n\n% another\n0.4,1.5,0.1\n0.5,1.6,0.2'));
    expect(r.rows).toHaveLength(2);
    expect(r.skipped).toBe(0);
  });

  it('counts malformed rows as skipped rather than failing', () => {
    const r = ok(parseOpticalFile('0.4,1.5,0.1\nbroken,row,here\n0.5,1.6,0.2'));
    expect(r.rows).toHaveLength(2);
    expect(r.skipped).toBe(1);
  });

  it('errors when fewer than two usable rows survive', () => {
    expect(parseOpticalFile('0.4,1.5,0.1')).toHaveProperty('error');
    expect(parseOpticalFile('')).toHaveProperty('error');
    expect(parseOpticalFile('a,b,c\n1,2')).toHaveProperty('error');
  });
});

describe('convertOpticalRows — the canonical [λ_µm, n, k] table', () => {
  const rows: RawOpticalRow[] = [[500, 1.5, 0.1], [600, 1.6, 0.2]];

  it('converts nm to µm', () => {
    const r = ok(convertOpticalRows(rows, 'wavelength-nm', 'nk'));
    expect(r.data[0][0]).toBeCloseTo(0.5, 12);
    expect(r.lambdaMinNm).toBeCloseTo(500, 9);
    expect(r.lambdaMaxNm).toBeCloseTo(600, 9);
  });

  it('leaves µm untouched', () => {
    const r = ok(convertOpticalRows([[0.5, 1.5, 0.1], [0.6, 1.6, 0.2]], 'wavelength-um', 'nk'));
    expect(r.data[0][0]).toBeCloseTo(0.5, 12);
  });

  it('maps photon energy to wavelength and re-sorts ascending', () => {
    // energy and wavelength run opposite ways, so the 3 eV row (shorter λ) must
    // end up FIRST once the table is sorted ascending in wavelength
    const r = ok(convertOpticalRows([[3, 1.5, 0.1], [2, 1.6, 0.2]], 'energy-ev', 'nk'));
    expect(r.data[0][0]).toBeCloseTo(PHOTON_HC_EV_NM / 3 / 1000, 9);
    expect(r.data[1][0]).toBeCloseTo(PHOTON_HC_EV_NM / 2 / 1000, 9);
    expect(r.data[0][0]).toBeLessThan(r.data[1][0]);
    // and the n,k values must travel with their own row through the re-sort
    expect(r.data[0][1]).toBeCloseTo(1.5, 12);
    expect(r.data[1][1]).toBeCloseTo(1.6, 12);
  });

  it('converts a permittivity pair to (n, k)', () => {
    // eps = -4 + 0i is a lossless metal below the plasma frequency: n = 0, k = 2
    const r = ok(convertOpticalRows([[500, -4, 0], [600, -4, 0]], 'wavelength-nm', 'epsilon'));
    expect(r.data[0][1]).toBeCloseTo(0, 9);
    expect(r.data[0][2]).toBeCloseTo(2, 9);
  });

  it('takes |k| so a negative-convention file still yields absorption', () => {
    const r = ok(convertOpticalRows([[500, 1.5, -0.3], [600, 1.6, -0.4]], 'wavelength-nm', 'nk'));
    expect(r.data[0][2]).toBeCloseTo(0.3, 12);
  });

  it('drops non-positive abscissae instead of producing infinities', () => {
    const r = ok(convertOpticalRows([[0, 1.5, 0.1], [500, 1.5, 0.1], [600, 1.6, 0.2]], 'wavelength-nm', 'nk'));
    expect(r.data).toHaveLength(2);
    expect(r.data.every(row => Number.isFinite(row[0]))).toBe(true);
  });

  it('errors when the chosen reading leaves fewer than two rows', () => {
    expect(convertOpticalRows([[-1, 1, 0], [0, 1, 0]], 'wavelength-nm', 'nk')).toHaveProperty('error');
  });

  it('reports the n and k extrema it saw', () => {
    const r = ok(convertOpticalRows([[500, 1.5, 0.1], [600, 2.5, 0.4]], 'wavelength-nm', 'nk'));
    expect(r.nMin).toBeCloseTo(1.5, 12); expect(r.nMax).toBeCloseTo(2.5, 12);
    expect(r.kMin).toBeCloseTo(0.1, 12); expect(r.kMax).toBeCloseTo(0.4, 12);
  });
});

describe('inferImportModes — preselecting the dialog', () => {
  it('reads a large first column as nanometres', () => {
    expect(inferImportModes([[400, 1.5, 0.1], [800, 1.6, 0.2]]).axis).toBe('wavelength-nm');
  });

  it('reads a small first column with index-like values as micrometres', () => {
    const m = inferImportModes([[0.4, 1.5, 0.1], [0.8, 1.6, 0.2]]);
    expect(m.axis).toBe('wavelength-um');
    expect(m.values).toBe('nk');
  });

  it('treats a negative real part as a permittivity, on an energy axis', () => {
    const m = inferImportModes([[2, -4, 0.5], [3, -2, 0.6]]);
    expect(m.values).toBe('epsilon');
    expect(m.axis).toBe('energy-ev');
  });

  it('treats implausibly large values as a permittivity', () => {
    expect(inferImportModes([[2, 40, 5], [3, 45, 6]]).values).toBe('epsilon');
  });
});

describe('sampleRows and formatCompact', () => {
  it('returns everything when under the cap, keeping order', () => {
    expect(sampleRows([1, 2, 3], 8)).toEqual([1, 2, 3]);
  });

  it('samples down to the cap and keeps both endpoints', () => {
    const s = sampleRows(Array.from({ length: 100 }, (_, i) => i), 8);
    expect(s).toHaveLength(8);
    expect(s[0]).toBe(0);
    expect(s[s.length - 1]).toBe(99);
  });

  it('switches to exponential only outside the readable range', () => {
    expect(formatCompact(1.5)).toBe('1.5000');
    expect(formatCompact(12345)).toContain('e+');
    expect(formatCompact(0.0001)).toContain('e-');
    expect(formatCompact(NaN)).toBe('');
  });
});
