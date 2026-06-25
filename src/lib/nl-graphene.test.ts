// Validation of the nonlinear-graphene-plasmonics engine against the original
// Python reference package (C:\Users\alvar\Downloads\NLPlasmonics, 44 pytest cases).
// Reference numbers were generated with nlgraphene.polarizability at fixed points
// and transcribed here. Energies in eV, lengths in nm, gamma = 50 meV (the package
// default), eps_bar = 1.
import { describe, it, expect } from 'vitest';
import {
  NL_GEOMETRIES, getNlGeometry, grapheneSigma2d, nlAlphaAt,
  nlResonanceEnergyGraphene, nlSpectrum,
  type NlOrder, type NlGeometryId, type NlMaterial,
} from './nl-graphene';

const GAMMA = 0.05, EPS = 1.0;
const graphene = (EF: number, gamma = GAMMA): NlMaterial =>
  ({ sigma2d: (E) => grapheneSigma2d(E, EF, gamma), nlEF: EF, nlGamma: gamma, doping: 1 });

// Table I of Cox, Yu & García de Abajo, PRB 96, 045442 (2017) — independent transcription.
const TABLE_I: Record<NlGeometryId, Record<string, number>> = {
  ribbon:   { eta1: -0.06873, zeta1x: 0.9428, xt22A: 0, xt22B: 0, xt22C: 0, xt31: 1.031, xt33: -0.9415 },
  triangle: { eta1: -0.08780, zeta1x: 0.5437, xt22A: 0.3192, xt22B: -0.3742, xt22C: -0.7490, xt31: 0.2816, xt33: 0.2608 },
  disk:     { eta1: -0.07310, zeta1x: 0.8510, xt22A: 0, xt22B: 0, xt22C: 0, xt31: 0.7728, xt33: 0.7334 },
};

describe('Table I geometry parameters (test_geometries.py)', () => {
  for (const id of Object.keys(TABLE_I) as NlGeometryId[]) {
    it(`${id} matches the paper`, () => {
      const geo = getNlGeometry(id) as unknown as Record<string, number>;
      for (const [k, v] of Object.entries(TABLE_I[id])) expect(geo[k]).toBeCloseTo(v, 6);
    });
  }
  it('all eta1 are negative', () => {
    for (const g of NL_GEOMETRIES) expect(g.eta1).toBeLessThan(0);
  });
});

describe('Plasmon resonance energies (test_known_resonance_energies)', () => {
  const cases: [NlGeometryId, number, number, number][] = [
    ['ribbon', 10, 1, 0.817], ['ribbon', 50, 1, 0.365],
    ['disk', 10, 1, 0.792], ['disk', 10, 2, 1.120], ['triangle', 10, 1, 0.723],
  ];
  for (const [id, D, EF, expected] of cases) {
    it(`${id} D=${D} EF=${EF} -> ${expected} eV`, () => {
      expect(nlResonanceEnergyGraphene(id, EF, D, EPS)).toBeCloseTo(expected, 2);
    });
  }
  it('scales as 1/sqrt(D) and sqrt(EF) (test_resonance_scaling_laws)', () => {
    const r = nlResonanceEnergyGraphene('ribbon', 1, 10, EPS) / nlResonanceEnergyGraphene('ribbon', 1, 40, EPS);
    expect(r).toBeCloseTo(2.0, 6);
    const s = nlResonanceEnergyGraphene('ribbon', 4, 20, EPS) / nlResonanceEnergyGraphene('ribbon', 1, 20, EPS);
    expect(s).toBeCloseTo(2.0, 6);
  });
});

describe('SHG selection rule (test_centrosymmetric_no_shg / test_triangle_has_shg)', () => {
  const Es = Array.from({ length: 40 }, (_, i) => 0.3 + (0.9 * i) / 39);
  it('ribbon and disk: alpha22 == 0', () => {
    for (const id of ['ribbon', 'disk'] as NlGeometryId[]) {
      const geo = getNlGeometry(id);
      for (const E of Es) {
        const a = nlAlphaAt('22', geo, 10, EPS, graphene(1), E);
        expect(Math.hypot(a.re, a.im)).toBe(0);
      }
    }
  });
  it('triangle: alpha22 != 0', () => {
    const geo = getNlGeometry('triangle');
    const mx = Math.max(...Es.map(E => { const a = nlAlphaAt('22', geo, 10, EPS, graphene(1), E); return Math.hypot(a.re, a.im); }));
    expect(mx).toBeGreaterThan(0);
  });
});

// Exact port comparison: nlAlphaAt vs nlgraphene.polarizability at fixed points.
describe('Polarizability port matches Python reference (Eqs. 29a-d)', () => {
  // geo, EF, D, E, order -> [Re, Im] from nlgraphene.polarizability
  const REF: [NlGeometryId, number, number, number, NlOrder, number, number][] = [
    ['ribbon', 1, 20, 0.40, '11', 46.3628708, 5.346089181],
    ['ribbon', 1, 20, 0.40, '31', 169.4447129, 74.83690577],
    ['ribbon', 1, 20, 0.40, '33', -6.029868035, -3.208957203],
    ['ribbon', 1, 20, 0.70, '11', -49.57069294, 11.08226867],
    ['ribbon', 1, 20, 0.70, '31', 2282.948673, -789.5768523],
    ['ribbon', 1, 20, 0.70, '33', 22.41494337, -14.58114059],
    ['triangle', 1, 20, 0.50, '11', 800.2436462, 1815.091755],
    ['triangle', 1, 20, 0.50, '22', 605.0498791, -712.7107406],
    ['triangle', 1, 20, 0.50, '31', -3184892.626, 2581375.273],
    ['triangle', 1, 20, 0.50, '33', -13889.53794, -6557.840354],
    ['triangle', 1, 20, 0.30, '11', 314.4851344, 27.58285331],
    ['triangle', 1, 20, 0.30, '22', -23.50412353, -2.697226132],
    ['triangle', 1, 20, 0.30, '31', 315.6795958, 141.7789488],
    ['triangle', 1, 20, 0.30, '33', 20.31672956, 9.917288824],
    ['disk', 1, 10, 0.80, '11', -243.5669348, 750.8412964],
    ['disk', 1, 10, 0.80, '31', -2454365.644, -2156824.56],
    ['disk', 1, 10, 0.80, '33', 6404.059364, -3988.158292],
    ['disk', 2, 10, 1.10, '11', 588.9513168, 735.3762654],
    ['disk', 2, 10, 1.10, '31', -210300.6904, 708972.9067],
    ['disk', 2, 10, 1.10, '33', -1392.8217, 592.2780692],
  ];
  for (const [id, EF, D, E, order, re, im] of REF) {
    it(`${id} alpha${order} @ E=${E}, EF=${EF}, D=${D}`, () => {
      const a = nlAlphaAt(order, getNlGeometry(id), D, EPS, graphene(EF), E);
      const rel = Math.max(Math.abs(re), Math.abs(im), 1e-9);
      expect(Math.abs(a.re - re) / rel).toBeLessThan(1e-5);
      expect(Math.abs(a.im - im) / rel).toBeLessThan(1e-5);
    });
  }
});

describe('Nonlinear peaks track the fundamental plasmon (test_nonlinear_peaks_track_fundamental)', () => {
  it('|alpha31| and |alpha33| peak at the linear resonance (disk D=10)', () => {
    const id: NlGeometryId = 'disk', EF = 1, D = 10;
    const Ep = nlResonanceEnergyGraphene(id, EF, D, EPS);
    const N = 8000, E = new Float64Array(N);
    for (let i = 0; i < N; i++) E[i] = 0.3 + (1.1 * i) / (N - 1);
    for (const order of ['31', '33'] as NlOrder[]) {
      const { re, im } = nlSpectrum(order, id, D, EPS, graphene(EF), E);
      let best = 0, bestv = -1;
      for (let i = 0; i < N; i++) { const v = Math.hypot(re[i], im[i]); if (v > bestv) { bestv = v; best = i; } }
      expect(E[best]).toBeCloseTo(Ep, 1);
    }
  });
});
