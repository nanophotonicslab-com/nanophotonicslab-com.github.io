// Validation of the nonlinear-graphene-plasmonics engine against the original
// Python reference package (C:\Users\alvar\Downloads\NLPlasmonics, 44 pytest cases).
// Reference numbers were generated with nlgraphene.polarizability at fixed points
// and transcribed here. Energies in eV, lengths in nm, gamma = 50 meV (the package
// default), eps_bar = 1.
import { describe, it, expect } from 'vitest';
import {
  NL_GEOMETRIES, getNlGeometry, grapheneSigma2d, grapheneSigma2dRPA, nlAlphaAt,
  nlResonanceEnergyGraphene, nlSpectrum, ribbonAlphaMulti, geometryAlphaMulti, grapheneR_THG, grapheneR_Kerr,
  type NlOrder, type NlGeometryId, type NlMaterial,
} from './nl-graphene';

const GAMMA = 0.05, EPS = 1.0;
const graphene = (EF: number, gamma = GAMMA): NlMaterial =>
  ({ sigma2d: (E) => grapheneSigma2d(E, EF, gamma), nlEF: EF, nlGamma: gamma, doping: 1 });
const grapheneRpa = (EF: number, gamma = GAMMA): NlMaterial =>
  ({ sigma2d: (E) => grapheneSigma2dRPA(E, EF, gamma), nlEF: EF, nlGamma: gamma, doping: 1 });
const close = (a: number, b: number) => expect(Math.abs(a - b) / Math.max(Math.abs(b), 1e-6)).toBeLessThan(2e-4);

describe('Local-RPA linear conductivity (sigma11_localRPA, EF=1, γ=0.05)', () => {
  const REF: [number, number, number][] = [
    [0.3, 0.38531553, 2.205886], [0.6, 0.10561252, 1.0449113],
    [1.0, 0.046329893, 0.50356172], [1.8, 0.055677151, -0.1207676],
  ];
  for (const [E, re, im] of REF) it(`σ²ᴰ_RPA(E=${E})`, () => { const s = grapheneSigma2dRPA(E, 1, 0.05); close(s.re, re); close(s.im, im); });
});

// Higher-order ribbon modes — multimode Eqs. (27), validated against nlgraphene.ribbon_modes.
describe('Multimode ribbon polarizability (ribbon D=40, EF=1, γ=0.05)', () => {
  // nmodes, E, order -> [Re, Im] from ribbon_modes.alphaXX_multi (model="drude")
  const REF: [number, number, NlOrder, number, number][] = [
    [1, 0.3, '11', 204.497, 39.943], [1, 0.3, '31', 2368.55, 1737.66], [1, 0.3, '33', 86.4605, 81.5194],
    [1, 0.6, '11', -82.3989, 12.7946], [1, 0.6, '33', -23.4345, 9.00254],
    [2, 0.3, '11', 206.461, 39.9983], [2, 0.3, '31', 2272.52, 1677.02], [2, 0.3, '33', 65.7355, 65.8706],
    [3, 0.3, '11', 206.812, 40.0037], [3, 0.3, '31', 2265.64, 1673.05], [3, 0.3, '33', 65.751, 66.9793],
    [3, 0.6, '11', -77.9815, 13.2642], [3, 0.6, '33', -24.9602, 7.95574],
  ];
  for (const [nm, E, order, re, im] of REF) {
    it(`α${order} nmodes=${nm} @E=${E}`, () => {
      const a = ribbonAlphaMulti(order, 40, EPS, graphene(1), E, nm);
      close(a.re, re); close(a.im, im);
    });
  }
  it('multimode with local-RPA conductivity (nmodes=3, E=0.3)', () => {
    const a11 = ribbonAlphaMulti('11', 40, EPS, grapheneRpa(1), 0.3, 3);
    const a33 = ribbonAlphaMulti('33', 40, EPS, grapheneRpa(1), 0.3, 3);
    close(a11.re, 210.449); close(a11.im, 44.6753);
    close(a33.re, 68.2054); close(a33.im, 87.6375);
  });
});

// Mikhailov full-RPA THG interband ratio R_THG = σ³_full/σ³_intra (PRB 93, 085403),
// validated against the external library via nlgraphene.rpa_bridge.interband_ratio_thg (EF=1, γ=0.05).
describe('Mikhailov full-RPA THG ratio R_THG(ω)', () => {
  const REF: [number, number, number][] = [
    [0.30, 1.268580, 0.031152], [0.40, 1.564807, 0.056950], [0.55, 2.689539, 0.218816],
    [0.63, 4.071679, 1.087147], [0.66667, 11.316215, -4.695218], [0.70, 5.150808, 10.443756],
    [0.79, -1.292310, 7.503531], [0.91, -4.847604, 5.767292], [1.00, -9.325938, 1.038836],
  ];
  for (const [E, re, im] of REF) it(`R_THG(${E})`, () => { const r = grapheneR_THG(E, 1, 0.05); close(r.re, re); close(r.im, im); });
});

// Mikhailov full-RPA Kerr interband ratio R_Kerr = σ³(ω,ω,−ω)_full/σ³_intra (PRB 93, 085403),
// validated against the external library via nlgraphene.rpa_bridge.interband_ratio_kerr (EF=1, γ=0.05).
// Exercises the coincident-argument f_int branches (O₁=O₁₂₃ in the symmetrized permutations).
describe('Mikhailov full-RPA Kerr ratio R_Kerr(ω)', () => {
  const REF: [number, number, number][] = [
    [0.30, 1.047129, 0.008262], [0.40, 1.088723, 0.011945], [0.55, 1.189083, 0.019724],
    [0.63, 1.271221, 0.025974], [0.66667, 1.318992, 0.029707], [0.70, 1.369803, 0.033802],
    [0.79, 1.559392, 0.050675], [0.91, 2.097773, 0.122559], [1.00, 3.443385, 1.358400],
  ];
  for (const [E, re, im] of REF) it(`R_Kerr(${E})`, () => { const r = grapheneR_Kerr(E, 1, 0.05); close(r.re, re); close(r.im, im); });
});

// Full-RPA THG: Drude σ³³ × R_THG via sigma3Boost; validated vs ribbon_modes.alpha33_multi(model="fullrpa").
describe('Full-RPA multimode THG (ribbon D=40, EF=1, γ=0.05, nmodes=3)', () => {
  const fullrpa: NlMaterial = {
    sigma2d: (E) => grapheneSigma2dRPA(E, 1, 0.05), nlEF: 1, nlGamma: 0.05, doping: 1,
    sigma3Boost: (order, E) => order === '33' ? grapheneR_THG(E, 1, 0.05) : { re: 1, im: 0 },
  };
  const REF: [number, number, number][] = [
    [0.30, 83.7939, 113.3], [0.40, 1880.38, -11518.4], [0.60, -77.1313, 8.93315], [0.66667, -179.948, 37.491],
  ];
  for (const [E, re, im] of REF) it(`α33_fullrpa @E=${E}`, () => {
    const a = geometryAlphaMulti('ribbon', '33', 40, EPS, fullrpa, E, 3);
    close(a.re, re); close(a.im, im);
  });
});

// Triangle & disk higher modes — Table S2 (η_j, ζ_j) + SI S4/S5 ξ³ tensors (notes/03_geometries_table.md).
describe('Multimode triangle / disk polarizability (D=20, EF=1, γ=0.05)', () => {
  // geo, order, E, nmodes -> [Re, Im] from the multimode Eqs. (27) with the SI tensors
  const REF: [NlGeometryId, NlOrder, number, number, number, number][] = [
    ['triangle', '11', 0.4, 1, 516.828, 102.32],   ['triangle', '11', 0.7, 3, -189.03, 39.8797],
    ['triangle', '31', 0.4, 1, 7021.08, 4571.84],  ['triangle', '31', 0.4, 3, 6793.5, 4439.32],
    ['triangle', '33', 0.4, 1, 182.877, 160.976],  ['triangle', '33', 0.4, 3, 160.887, 140.672],
    ['disk', '11', 0.4, 1, 850.488, 110.804],      ['disk', '11', 0.4, 3, 855.504, 110.878],
    ['disk', '31', 0.7, 3, 22841.9, -6558.43],     ['disk', '33', 0.4, 2, 84.9221, 157.647],
  ];
  for (const [geo, order, E, nm, re, im] of REF) {
    it(`${geo} α${order} nmodes=${nm} @E=${E}`, () => {
      const a = geometryAlphaMulti(geo, order, 20, EPS, graphene(1), E, nm);
      close(a.re, re); close(a.im, im);
    });
  }
});

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
