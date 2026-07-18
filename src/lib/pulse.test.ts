import { describe, it, expect } from 'vitest';
import {
  SELLMEIER, sellmeierN, groupIndex, gvdFs2PerMm, todFs3PerMm,
  broadenedFwhmFs, transformLimitedBandwidthNm, AC_FACTOR,
} from './pulse';

const FS = SELLMEIER['fused-silica'], BK7 = SELLMEIER.bk7, SAPPH = SELLMEIER.sapphire;

describe('Sellmeier refractive index (literature values)', () => {
  it('fused silica: n(588 nm) ≈ 1.4585, n(1064 nm) ≈ 1.4496', () => {
    expect(sellmeierN(FS, 0.588)).toBeCloseTo(1.4585, 3);
    expect(sellmeierN(FS, 1.064)).toBeCloseTo(1.4496, 3);
  });
  it('N-BK7: n(587.6 nm) ≈ 1.5168 (the catalog n_d)', () => {
    expect(sellmeierN(BK7, 0.5876)).toBeCloseTo(1.5168, 3);
  });
  it('group index of fused silica at 800 nm ≈ 1.467', () => {
    expect(groupIndex(FS, 800)).toBeGreaterThan(1.460);
    expect(groupIndex(FS, 800)).toBeLessThan(1.475);
  });
});

describe('GVD/TOD (literature values)', () => {
  it('fused silica at 800 nm: β₂ ≈ +36.2 fs²/mm, β₃ ≈ +27.5 fs³/mm', () => {
    expect(gvdFs2PerMm(FS, 800)).toBeCloseTo(36.2, 0);
    expect(todFs3PerMm(FS, 800)).toBeGreaterThan(20);
    expect(todFs3PerMm(FS, 800)).toBeLessThan(35);
  });
  it('N-BK7 at 800 nm: β₂ ≈ +44.6 fs²/mm; sapphire ≈ +58 fs²/mm', () => {
    expect(gvdFs2PerMm(BK7, 800)).toBeCloseTo(44.6, 0);
    expect(gvdFs2PerMm(SAPPH, 800)).toBeCloseTo(58, -1);
  });
  it('fused silica zero-dispersion wavelength ≈ 1.27 µm', () => {
    // bisection for the β₂ sign change
    let lo = 1000, hi = 1600;
    expect(gvdFs2PerMm(FS, lo)).toBeGreaterThan(0);
    expect(gvdFs2PerMm(FS, hi)).toBeLessThan(0);
    for (let i = 0; i < 40; i++) {
      const mid = (lo + hi) / 2;
      if (gvdFs2PerMm(FS, mid) > 0) lo = mid; else hi = mid;
    }
    expect((lo + hi) / 2).toBeGreaterThan(1240);
    expect((lo + hi) / 2).toBeLessThan(1300);
  });
});

describe('pulse bookkeeping', () => {
  it('zero GDD leaves the pulse unchanged; broadening follows the Gaussian law', () => {
    expect(broadenedFwhmFs(10, 0)).toBeCloseTo(10, 12);
    // 10 fs through 1 mm fused silica at 800 nm: x = 4ln2·36.2/100 ≈ 1.004 → τ ≈ 14.2 fs
    const tau = broadenedFwhmFs(10, gvdFs2PerMm(FS, 800));
    expect(tau).toBeGreaterThan(13.5);
    expect(tau).toBeLessThan(14.9);
  });
  it('a longer pulse is barely affected by the same GDD', () => {
    const ratio = broadenedFwhmFs(100, 36.2) / 100;
    expect(ratio).toBeLessThan(1.01);
  });
  it('transform-limited bandwidth: 10 fs Gaussian at 800 nm ≈ 94 nm', () => {
    expect(transformLimitedBandwidthNm(10, 800, 'gaussian')).toBeCloseTo(94.1, 0);
  });
  it('autocorrelation deconvolution factors', () => {
    expect(AC_FACTOR.gaussian).toBeCloseTo(1.414, 3);
    expect(AC_FACTOR.sech2).toBeCloseTo(1.543, 3);
  });
});
