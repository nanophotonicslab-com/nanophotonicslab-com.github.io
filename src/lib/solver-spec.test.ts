import { describe, it, expect } from 'vitest';
import {
  coerce, decodeState, defaultValues, encodeState, evaluateEnvelope, fmt,
  fromSliderPos, groupParams, isChoice, hasSlider, isLogSlider, toSliderPos,
  LOG_SLIDER_STEPS, type Solver,
} from './solver-spec';

const spec: Solver = {
  meta: {
    lab: 'test', id: 'demo', code: 'T1', title: 'Demo', blurb: '',
    status: 'Beta', version: '0.1.0', updated: '2026-07-28',
  },
  docs: { model: '', assumptions: [], validity: '', limitations: [], references: [] },
  groups: [{ id: 'a', label: 'A' }, { id: 'b', label: 'B' }],
  params: [
    { key: 'n', label: 'n', group: 'a', default: 20, min: 1, max: 500, integer: true },
    { key: 'd', label: 'D', group: 'a', default: 0.5, min: 0.001, max: 50, scale: 'log' },
    { key: 'mode', label: 'mode', group: 'b', default: 'brownian', choices: ['brownian', 'drift'] },
    { key: 'field', label: 'field', group: 'b', default: 256, choices: [128, 256, 512] },
  ],
  observables: [{ key: 'x', label: 'X', help: '' }],
  plots: [],
  envelope: [
    {
      id: 'nyquist', label: 'Nyquist',
      evaluate: ({ p }) => Number(p.n) <= 100
        ? { level: 'ok', message: 'fine' }
        : { level: 'warn', message: 'too many' },
    },
    { id: 'boom', label: 'Broken', evaluate: () => { throw new Error('bug'); } },
  ],
  scenarios: [{ label: 'S', set: { n: 3 } }],
};

describe('parameter classification', () => {
  it('separates choices from numeric fields', () => {
    expect(isChoice(spec.params[0])).toBe(false);
    expect(isChoice(spec.params[2])).toBe(true);
    expect(hasSlider(spec.params[0])).toBe(true);
    expect(hasSlider(spec.params[2])).toBe(false);
    expect(isLogSlider(spec.params[1])).toBe(true);
    expect(isLogSlider(spec.params[0])).toBe(false);
  });

  it('groups parameters in declaration order', () => {
    expect(groupParams(spec, 'a').map(p => p.key)).toEqual(['n', 'd']);
    expect(groupParams(spec, 'b').map(p => p.key)).toEqual(['mode', 'field']);
  });

  it('collects defaults', () => {
    expect(defaultValues(spec)).toEqual({ n: 20, d: 0.5, mode: 'brownian', field: 256 });
  });
});

describe('coercion', () => {
  it('clamps to the declared domain and rounds integers', () => {
    expect(coerce(spec.params[0], 1e6)).toBe(500);
    expect(coerce(spec.params[0], -5)).toBe(1);
    expect(coerce(spec.params[0], 3.7)).toBe(4);
  });

  it('falls back to the default for junk', () => {
    expect(coerce(spec.params[0], 'nonsense')).toBe(20);
  });

  it('keeps choices inside their list, preserving numeric type', () => {
    expect(coerce(spec.params[2], 'drift')).toBe('drift');
    expect(coerce(spec.params[2], 'teleport')).toBe('brownian');
    expect(coerce(spec.params[3], '512')).toBe(512);
    expect(coerce(spec.params[3], 999)).toBe(256);
  });
});

describe('logarithmic sliders', () => {
  it('maps the endpoints and round-trips interior values', () => {
    const p = spec.params[1];
    expect(toSliderPos(p, 0.001)).toBe(0);
    expect(toSliderPos(p, 50)).toBe(LOG_SLIDER_STEPS);
    // the track is quantized: 4.7 decades over 1000 steps is ~1.1% per step, so
    // a round trip is accurate to half a step, not to an absolute tolerance
    for (const v of [0.002, 0.05, 0.5, 5, 40]) {
      const back = fromSliderPos(p, toSliderPos(p, v));
      expect(Math.abs(back - v) / v).toBeLessThan(0.01);
    }
  });

  it('is the identity for linear sliders', () => {
    expect(toSliderPos(spec.params[0], 42)).toBe(42);
    expect(fromSliderPos(spec.params[0], 42)).toBe(42);
  });

  it('puts the geometric midpoint at the middle of the track', () => {
    const p = spec.params[1];
    const mid = fromSliderPos(p, LOG_SLIDER_STEPS / 2);
    expect(mid).toBeCloseTo(Math.sqrt(0.001 * 50), 3);
  });
});

describe('permalinks', () => {
  it('omits defaults so links stay short', () => {
    expect(encodeState(spec, defaultValues(spec))).toBe('');
    expect(encodeState(spec, { ...defaultValues(spec), n: 7 })).toBe('n=7');
  });

  it('round-trips a full state', () => {
    const state = { n: 33, d: 12.5, mode: 'drift', field: 512 };
    const back = decodeState(spec, encodeState(spec, state));
    expect(back).toEqual(state);
  });

  it('ignores unknown keys and coerces hostile values', () => {
    const back = decodeState(spec, 'n=99999&bogus=1&mode=teleport');
    expect(back.n).toBe(500);
    expect(back.mode).toBe('brownian');
    expect('bogus' in back).toBe(false);
  });

  it('accepts a leading question mark or hash', () => {
    expect(decodeState(spec, '#n=5').n).toBe(5);
  });
});

describe('envelope evaluation', () => {
  it('reports each check and never throws on a broken predicate', () => {
    const rows = evaluateEnvelope(spec.envelope, { p: { n: 20 }, o: {} });
    expect(rows.map(r => r.level)).toEqual(['ok', 'ok']);
    expect(rows[0].message).toBe('fine');
  });

  it('flips level when the condition is crossed', () => {
    const rows = evaluateEnvelope(spec.envelope, { p: { n: 101 }, o: {} });
    expect(rows[0].level).toBe('warn');
  });
});

describe('formatting', () => {
  it('shows an em dash for non-finite values', () => {
    expect(fmt(NaN)).toBe('—');
    expect(fmt(Infinity)).toBe('—');
  });

  it('keeps three significant digits by default', () => {
    expect(fmt(0.5)).toBe('0.500');
    expect(fmt(12.345)).toBe('12.3');
    expect(fmt(1234)).toBe('1234');
    expect(fmt(0)).toBe('0');
  });

  it('switches to exponential outside the readable range', () => {
    expect(fmt(1e-6)).toBe('1.00e-6');
    expect(fmt(5e6)).toBe('5.00e+6');
  });
});
