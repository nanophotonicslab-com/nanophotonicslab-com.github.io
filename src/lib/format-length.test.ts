import { describe, it, expect } from 'vitest';
import { formatLength, formatLengthText } from './format-length';

describe('formatLength — unit choice', () => {
  it('keeps a Talbot distance of 200.6 mm in millimetres', () => {
    expect(formatLength(0.2006)).toEqual({ value: '200.6', unit: 'mm' });
  });

  it('switches to µm below a millimetre instead of printing 0.0 mm', () => {
    expect(formatLength(34.2e-6)).toEqual({ value: '34.2', unit: 'µm' });
  });

  it('switches to nm below a micrometre', () => {
    expect(formatLength(97e-9)).toEqual({ value: '97.0', unit: 'nm' });
  });

  it('uses metres from 1 m up', () => {
    expect(formatLength(1.234)).toEqual({ value: '1.2', unit: 'm' });
  });

  it('pads a whole number to one decimal', () => {
    expect(formatLength(5e-3)).toEqual({ value: '5.0', unit: 'mm' });
  });

  it('picks the unit on the rounded value, never printing 1000.0', () => {
    expect(formatLengthText(0.99996)).toBe('1.0 m');       // not "1000.0 mm"
    expect(formatLengthText(999.96e-9)).toBe('1.0 µm');    // not "1000.0 nm"
    expect(formatLengthText(0.99994)).toBe('999.9 mm');    // just below: stays down
  });

  it('shows a dash, not a number, for a length that is not a length', () => {
    for (const v of [0, -1, NaN, Infinity, -Infinity]) {
      expect(formatLength(v)).toEqual({ value: '—', unit: '' });
      expect(formatLengthText(v)).toBe('—');
    }
  });
});

describe('formatLength — the display rule, across twelve decades', () => {
  // Every length from 1 pm to 1 km, on a fine logarithmic grid with awkward mantissas.
  const samples: number[] = [];
  for (let e = -12; e <= 3; e += 0.013) samples.push(10 ** e);

  it('always prints exactly one decimal place', () => {
    for (const m of samples) expect(formatLength(m).value).toMatch(/^\d+\.\d$/);
  });

  it('always carries at least two significant figures', () => {
    for (const m of samples) {
      const { value } = formatLength(m);
      // with one decimal, >= 2 significant figures is exactly value >= 1.0
      expect(parseFloat(value)).toBeGreaterThanOrEqual(1);
    }
  });

  it('stays below 1000 in every unit except the largest', () => {
    for (const m of samples) {
      const { value, unit } = formatLength(m);
      if (unit !== 'm') expect(parseFloat(value)).toBeLessThan(1000);
    }
  });

  it('never drifts from the true length by more than the last printed digit', () => {
    const scale: Record<string, number> = { m: 1, mm: 1e-3, 'µm': 1e-6, nm: 1e-9, pm: 1e-12 };
    for (const m of samples) {
      const { value, unit } = formatLength(m);
      expect(Math.abs(parseFloat(value) * scale[unit] - m)).toBeLessThanOrEqual(0.05 * scale[unit] * (1 + 1e-9));
    }
  });
});
