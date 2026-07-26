import { describe, it, expect } from 'vitest';
import {
  ldos, ldosIntegrandTrace, radiationPattern, intensityRT, perLayerAbsorption,
  planeWaveFieldsAtZ, cx, type Cx,
} from './pyramids-ldos';
import fixture from './pyramids-ldos.fixture.json';

// The fixture is generated directly from PyRAMIDS
// (libraries/PyRAMIDS via scratchpad/gen_fixture.py). These tests prove the
// TypeScript port reproduces PyRAMIDS' LDOS to ~1e-3 relative.

const toCx = (pairs: number[][]): Cx[] => pairs.map(([re, im]) => cx(re, im));
const k0 = fixture.k0;

type Case = {
  nstack: number[][]; dstack: number[]; z: number[];
  E_par: number[]; E_perp: number[]; M_par: number[]; M_perp: number[]; C: number[];
};

function closeArr(got: number[], want: number[], atol = 3e-3, rtol = 3e-3) {
  expect(got.length).toBe(want.length);
  got.forEach((g, i) => {
    const tol = atol + rtol * Math.abs(want[i]);
    expect(Math.abs(g - want[i]), `index ${i}: got ${g}, want ${want[i]}`).toBeLessThan(tol);
  });
}

describe('PyRAMIDS LDOS port vs reference fixture', () => {
  for (const [name, c] of Object.entries(fixture.cases as Record<string, Case>)) {
    it(`reproduces LDOS channels for "${name}"`, () => {
      const r = ldos(k0, c.z, toCx(c.nstack), c.dstack);
      closeArr(r.E_par, c.E_par);
      closeArr(r.E_perp, c.E_perp);
      closeArr(r.M_par, c.M_par);
      closeArr(r.M_perp, c.M_perp);
      closeArr(r.C, c.C);
    });
  }

  it('free space gives LDOS = 1 in every channel', () => {
    const r = ldos(k0, [0.2, 0.6], [cx(1), cx(1)], []);
    for (const v of [...r.E_par, ...r.E_perp, ...r.M_par, ...r.M_perp]) {
      expect(Math.abs(v - 1)).toBeLessThan(3e-3);
    }
  });

  it('reproduces the k∥-resolved integrand trace (slab waveguide)', () => {
    const t = fixture.integrand_slab_wg;
    const got = ldosIntegrandTrace(k0, t.kpar, t.z, toCx([[1.5, 0], [3.5, 0], [1.0, 0]]), [0.1]);
    // t.data shape [5, Nz, Nk]
    for (let ch = 0; ch < 5; ch++) {
      for (let zi = 0; zi < t.z.length; zi++) {
        closeArr(got[ch][zi], (t.data as number[][][])[ch][zi], 5e-3, 5e-3);
      }
    }
  });
});

type RadCase = {
  nstack: number[][]; dstack: number[]; z: number;
  pu: number[][]; mu: number[][]; theta: number[]; phi: number[]; P: number[];
};

describe('PyRAMIDS far-field radiation pattern port', () => {
  for (const [name, c] of Object.entries(fixture.radiation as Record<string, RadCase>)) {
    it(`reproduces the radiation pattern for "${name}"`, () => {
      const got = radiationPattern(k0, c.z, toCx(c.pu), toCx(c.mu), c.theta, c.phi, toCx(c.nstack), c.dstack);
      closeArr(got, c.P, 1e-5, 3e-3);
    });
  }
});

type RtaCase = { nstack: number[][]; dstack: number[]; cases: { q: number; s: { R: number; T: number; A: number }; p: { R: number; T: number; A: number } }[] };

describe('PyRAMIDS plane-wave (S-matrix solver) port', () => {
  for (const name of ['lossy_rta', 'bragg_rta'] as const) {
    const c = fixture.planewave[name] as RtaCase;
    it(`reproduces R/T/A for "${name}"`, () => {
      for (const cs of c.cases) {
        const got = intensityRT(k0, cs.q * k0, toCx(c.nstack), c.dstack);
        for (const pol of ['s', 'p'] as const) {
          for (const key of ['R', 'T', 'A'] as const) {
            expect(Math.abs(got[pol][key] - cs[pol][key]), `${name} q=${cs.q} ${pol}.${key}`).toBeLessThan(3e-3);
          }
        }
      }
    });
  }

  it('reproduces per-layer absorption (lossy stack)', () => {
    const c = fixture.planewave.lossy_perlayer;
    const got = perLayerAbsorption(k0, c.q * k0, toCx(c.nstack), c.dstack);
    closeArr(got.s, c.s, 1e-4, 3e-3);
    closeArr(got.p, c.p, 1e-4, 3e-3);
  });

  it('per-layer absorption sums to total absorptance A (energy conservation)', () => {
    const c = fixture.planewave.lossy_perlayer;
    const got = perLayerAbsorption(k0, c.q * k0, toCx(c.nstack), c.dstack);
    const rta = intensityRT(k0, c.q * k0, toCx(c.nstack), c.dstack);
    const sumS = got.s.reduce((a, b) => a + b, 0);
    const sumP = got.p.reduce((a, b) => a + b, 0);
    expect(Math.abs(sumS - rta.s.A)).toBeLessThan(3e-3);
    expect(Math.abs(sumP - rta.p.A)).toBeLessThan(3e-3);
  });

  it('reproduces local fields and absorption density', () => {
    const c = fixture.planewave.lossy_fields;
    const got = planeWaveFieldsAtZ(k0, c.q * k0, toCx(c.nstack), c.dstack, c.z);
    closeArr(got.Es, c.Es, 1e-4, 3e-3);
    closeArr(got.Ep, c.Ep, 1e-4, 3e-3);
    closeArr(got.As, c.As, 1e-4, 3e-3);
    closeArr(got.Ap, c.Ap, 1e-4, 3e-3);
  });
});
