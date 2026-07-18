/**
 * Equivalence and invariant tests for the spectrum-level Mie engine
 * extracted from mie-scattering.astro into src/lib/mie.ts.
 *
 * The engine was moved verbatim; these tests pin it against the
 * independently-validated single-wavelength routines (mieAt,
 * mieCoefficients) and against physical invariants, so any regression
 * introduced by the extraction (or later refactors) is caught.
 */
import { describe, it, expect } from 'vitest';
import {
  mieAt, mieCoefficients, HC_EV_NM,
  computeMie, computeMieMultiShell, bhcoat,
  computeSphereCoeffs, computeShellCoeffsAtLambda,
  computeAngularPattern, computeNearField,
  type ShellLayer,
} from './mie';

const N_P = 1.5, K_P = 0.1;          // lossy dielectric particle
const N_HOST = 1.33;
const R_NM = 60;
const constDiel = (_l: number): [number, number] => [N_P, K_P];

const relDiff = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-300);

describe('computeMie (spectrum) vs mieAt (single wavelength)', () => {
  it('matches the validated single-λ engine at every grid point', () => {
    const res = computeMie(constDiel, R_NM, N_HOST, 400, 800, 5);
    for (let i = 0; i < res.lambda.length; i++) {
      const ref = mieAt(N_P, K_P, N_HOST, R_NM, res.lambda[i]);
      expect(relDiff(res.csca[i], ref.csca)).toBeLessThan(1e-10);
      expect(relDiff(res.cext[i], ref.cext)).toBeLessThan(1e-10);
      expect(Math.abs(res.cabs[i] - ref.cabs)).toBeLessThan(1e-10 * ref.cext);
    }
  });

  it('satisfies decomposition invariants (E+M, multipole sums, qsca, energy)', () => {
    const res = computeMie(constDiel, R_NM, N_HOST, 400, 800, 7);
    const geo = Math.PI * R_NM * R_NM;
    for (let i = 0; i < res.lambda.length; i++) {
      expect(relDiff(res.csca_e[i] + res.csca_m[i], res.csca[i])).toBeLessThan(1e-12);
      expect(relDiff(res.cext_e[i] + res.cext_m[i], res.cext[i])).toBeLessThan(1e-12);
      const mpSca = res.csca_mp[0][i] + res.csca_mp[1][i] + res.csca_mp[2][i] + res.csca_mp[3][i];
      const mpExt = res.cext_mp[0][i] + res.cext_mp[1][i] + res.cext_mp[2][i] + res.cext_mp[3][i];
      expect(relDiff(mpSca, res.csca[i])).toBeLessThan(1e-12);
      expect(relDiff(mpExt, res.cext[i])).toBeLessThan(1e-12);
      expect(relDiff(res.qsca[i], res.csca[i] / geo)).toBeLessThan(1e-12);
      expect(relDiff(res.energy[i], HC_EV_NM / res.lambda[i])).toBeLessThan(1e-12);
    }
  });
});

describe('multilayer paths degenerate to the homogeneous sphere', () => {
  const layer = (r: number): ShellLayer => ({ getDielectric: constDiel, radiusNm: r });

  it('computeMieMultiShell with a single layer matches computeMie', () => {
    const one = computeMieMultiShell([layer(R_NM)], N_HOST, 400, 800, 5);
    const ref = computeMie(constDiel, R_NM, N_HOST, 400, 800, 5);
    for (let i = 0; i < ref.lambda.length; i++) {
      expect(relDiff(one.csca[i], ref.csca[i])).toBeLessThan(1e-6);
      expect(relDiff(one.cext[i], ref.cext[i])).toBeLessThan(1e-6);
    }
  });

  it('bhcoat with identical core and shell matches mieAt', () => {
    const lambdaNm = 500, rCore = 25, rOuter = 50, nHost = 1.0;
    const k0 = 2 * Math.PI * nHost / (lambdaNm * 1e-9);
    const XX = k0 * rCore * 1e-9, YY = k0 * rOuter * 1e-9;
    const mRe = N_P / nHost, mIm = K_P / nHost;
    const bh = bhcoat(XX, YY, mRe, mIm, mRe, mIm);
    const geo = Math.PI * rOuter * rOuter;
    const ref = mieAt(N_P, K_P, nHost, rOuter, lambdaNm);
    expect(relDiff(bh.qs * geo, ref.csca)).toBeLessThan(1e-6);
    expect(relDiff(bh.qe * geo, ref.cext)).toBeLessThan(1e-6);
  });

  it('computeMieMultiShell with 3 identical layers matches computeMie', () => {
    const layers = [layer(20), layer(40), layer(R_NM)];
    const three = computeMieMultiShell(layers, N_HOST, 400, 800, 5);
    const ref = computeMie(constDiel, R_NM, N_HOST, 400, 800, 5);
    for (let i = 0; i < ref.lambda.length; i++) {
      expect(relDiff(three.csca[i], ref.csca[i])).toBeLessThan(1e-5);
      expect(relDiff(three.cext[i], ref.cext[i])).toBeLessThan(1e-5);
    }
  });
});

describe('per-order coefficients', () => {
  it('computeSphereCoeffs a/b match mieCoefficients order by order', () => {
    const lambdaNm = 550;
    const co = computeSphereCoeffs(constDiel, R_NM, N_HOST, lambdaNm);
    const ref = mieCoefficients(N_P, K_P, N_HOST, R_NM, lambdaNm);
    expect(co.nmax).toBe(ref.length);
    for (let n = 0; n < ref.length; n++) {
      expect(Math.abs(co.aRe[n] - ref[n].aRe)).toBeLessThan(1e-12);
      expect(Math.abs(co.aIm[n] - ref[n].aIm)).toBeLessThan(1e-12);
      expect(Math.abs(co.bRe[n] - ref[n].bRe)).toBeLessThan(1e-12);
      expect(Math.abs(co.bIm[n] - ref[n].bIm)).toBeLessThan(1e-12);
    }
  });

  it('computeShellCoeffsAtLambda with 2 identical layers matches the homogeneous coefficients', () => {
    const lambdaNm = 550;
    const layers: ShellLayer[] = [
      { getDielectric: constDiel, radiusNm: 30 },
      { getDielectric: constDiel, radiusNm: R_NM },
    ];
    const sh = computeShellCoeffsAtLambda(layers, N_HOST, lambdaNm);
    const ref = computeSphereCoeffs(constDiel, R_NM, N_HOST, lambdaNm);
    const scale = Math.max(...Array.from(ref.aRe, Math.abs), ...Array.from(ref.bRe, Math.abs));
    const nCompare = Math.min(sh.nmax, ref.nmax);
    expect(nCompare).toBeGreaterThan(2);
    for (let n = 0; n < nCompare; n++) {
      expect(Math.abs(sh.aRe[n] - ref.aRe[n])).toBeLessThan(1e-6 * scale);
      expect(Math.abs(sh.aIm[n] - ref.aIm[n])).toBeLessThan(1e-6 * scale);
      expect(Math.abs(sh.bRe[n] - ref.bRe[n])).toBeLessThan(1e-6 * scale);
      expect(Math.abs(sh.bIm[n] - ref.bIm[n])).toBeLessThan(1e-6 * scale);
    }
  });
});

describe('computeAngularPattern', () => {
  it('reproduces the Rayleigh limit for a tiny sphere', () => {
    const co = computeSphereCoeffs((_l) => [1.5, 0], 5, 1.0, 600);
    const p = computeAngularPattern(co, 181);
    const i0 = p.iPerp[0], i90 = p.iPerp[90], i180 = p.iPerp[180];
    // iPerp is isotropic in the Rayleigh limit; iPar ∝ cos²θ vanishes at 90°
    expect(relDiff(i0, i90)).toBeLessThan(0.01);
    expect(relDiff(i0, i180)).toBeLessThan(0.01);
    expect(p.iPar[90] / p.iPar[0]).toBeLessThan(1e-3);
  });

  it('is forward-dominant for a wavelength-scale sphere', () => {
    const co = computeSphereCoeffs((_l) => [1.5, 0], 200, 1.0, 500);
    const p = computeAngularPattern(co);
    expect(p.iPerp[0]).toBeGreaterThan(10 * p.iPerp[p.iPerp.length - 1]);
  });
});

describe('computeNearField', () => {
  const lambdaNm = 550, nHost = 1.33;
  const k = 2 * Math.PI * nHost / (lambdaNm * 1e-9); // 1/m, matches the page's usage
  const grid = 31;

  it('fills internal + external field when internal coefficients are present', () => {
    const co = computeSphereCoeffs(constDiel, R_NM, nHost, lambdaNm);
    const nf = computeNearField(co, k, R_NM, grid, 2.5 * R_NM);
    expect(nf.gridSize).toBe(grid);
    expect(nf.extent).toBe(2.5 * R_NM);
    expect(nf.maxVal).toBeGreaterThan(0);
    expect(nf.minVal).toBeGreaterThan(0);
    for (const v of nf.grid) { expect(Number.isFinite(v)).toBe(true); expect(v).toBeGreaterThanOrEqual(0); }
  });

  it('masks the interior with -1 when internal coefficients are absent', () => {
    const layers: ShellLayer[] = [
      { getDielectric: constDiel, radiusNm: 30 },
      { getDielectric: constDiel, radiusNm: R_NM },
    ];
    const co = computeShellCoeffsAtLambda(layers, nHost, lambdaNm);
    const nf = computeNearField(co, k, R_NM, grid, 2.5 * R_NM);
    const centre = Math.floor(grid / 2);
    expect(nf.grid[centre * grid + centre]).toBe(-1);
    expect(nf.maxVal).toBeGreaterThan(0);
  });
});
