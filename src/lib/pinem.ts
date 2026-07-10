// PINEM physics — TypeScript port of the reference `pinem.py` library, following
// A. Feist, K. E. Echternkamp, J. Schauss, S. V. Yalunin, S. Schäfer & C. Ropers,
// "Quantum coherent optical phase modulation in an ultrafast transmission electron
// microscope", Nature 521, 200–203 (2015).
//
// The optical near field acts on the electron via the unitary S = exp(g* a − g a†),
// giving sideband amplitudes c_N = (g/|g|)^N J_N(2|g|) and populations P_N = J_N(2|g|)².
// This module implements the three computations of the reference: sideband
// probabilities, the Lorentzian-broadened EELS spectrum, and the co-moving-frame
// propagation of the charge density into an attosecond pulse train.
//
// Units: electron kinetic energy in eV, optical wavelength in nm; SI internally.

// ── physical constants (CODATA, identical to pinem.py) ──
export const C_LIGHT = 299_792_458.0;        // m/s
export const HBAR = 1.054_571_817e-34;       // J·s
export const QE = 1.602_176_634e-19;         // C
export const ME = 9.109_383_7015e-31;        // kg

// ── Bessel functions of the first kind, integer order ──
// J_0..J_nmax at x, via downward (Miller) recurrence with the closure
// normalization 1 = J_0 + 2(J_2 + J_4 + …). Robust for the needed range
// (x up to ~40, order up to ~60). J_{-n} = (−1)^n J_n.
export function besselJ0toN(x: number, nmax: number): Float64Array {
  const out = new Float64Array(nmax + 1);
  if (x === 0) { out[0] = 1; return out; }
  const ax = Math.abs(x), tox = 2 / ax;
  // downward recurrence must start well above both the order and the argument
  const start = Math.max(nmax, Math.ceil(ax));
  let m = 2 * (Math.floor((start + Math.floor(Math.sqrt(40 * (start + 1)))) / 2) + 6);
  if (m < nmax + 2) m = nmax + 2;
  const BIG = 1e10, SMALL = 1e-10;
  let bjp = 0, bj = 1, sum = 0, jsum = false;
  for (let j = m; j > 0; j--) {
    const bjm = j * tox * bj - bjp;
    bjp = bj;   // J_j
    bj = bjm;   // J_{j-1}
    if (Math.abs(bj) > BIG) {
      bj *= SMALL; bjp *= SMALL; sum *= SMALL;
      for (let k = 0; k <= nmax; k++) out[k] *= SMALL;
    }
    if (jsum) sum += bj;
    jsum = !jsum;
    if (j <= nmax) out[j] = bjp;
  }
  out[0] = bj;                 // J_0 after the final step
  sum = 2 * sum - bj;          // closure: J_0 + 2(J_2+J_4+…) = 1
  for (let k = 0; k <= nmax; k++) out[k] /= sum;
  if (x < 0) for (let k = 1; k <= nmax; k += 2) out[k] = -out[k];
  return out;
}

/** Single-order Bessel J_n(x) (n may be negative). */
export function besselJ(n: number, x: number): number {
  const an = Math.abs(n);
  const v = besselJ0toN(x, an)[an];
  return n < 0 && (an % 2 === 1) ? -v : v;
}

// ── electron beam + optical drive ──
export interface Beam { energyEV: number; wavelengthNm: number; }
export const beamOmega = (b: Beam) => (2 * Math.PI * C_LIGHT) / (b.wavelengthNm * 1e-9);   // rad/s
export const beamPeriodS = (b: Beam) => (2 * Math.PI) / beamOmega(b);                       // s
export const beamPhotonEV = (b: Beam) => (HBAR * beamOmega(b)) / QE;                        // eV (sideband spacing)
export const beamGamma = (b: Beam) => 1 + (b.energyEV * QE) / (ME * C_LIGHT * C_LIGHT);
export const beamVelocity = (b: Beam) => { const g = beamGamma(b); return C_LIGHT * Math.sqrt(1 - 1 / (g * g)); };
/** Electron Talbot distance z_T = 4π γ³ m v³ / (ħ ω²) [m]. */
export function talbotDistanceM(b: Beam): number {
  const g = beamGamma(b), v = beamVelocity(b), w = beamOmega(b);
  return (4 * Math.PI * g * g * g * ME * v * v * v) / (HBAR * w * w);
}
/** β = v/c ↔ kinetic energy [eV]. */
export const betaToEnergyEV = (beta: number) => { const g = 1 / Math.sqrt(1 - beta * beta); return (g - 1) * ME * C_LIGHT * C_LIGHT / QE; };
export const energyEVToBeta = (E: number) => { const g = 1 + (E * QE) / (ME * C_LIGHT * C_LIGHT); return Math.sqrt(Math.max(0, 1 - 1 / (g * g))); };
export const wavelengthNmFromEV = (eV: number) => 1239.841984 / eV;
export const eVFromWavelengthNm = (nm: number) => 1239.841984 / nm;

// ── 1. sideband probabilities ──
export interface Sidebands { N: Int32Array; P: Float64Array; }
/** P_N = J_N(2|g|)² for N = −nmax..nmax. Σ_N P_N → 1 (Bessel closure). */
export function sidebandProbabilities(absg: number, nmax = 30): Sidebands {
  const J = besselJ0toN(2 * Math.abs(absg), nmax);
  const N = new Int32Array(2 * nmax + 1), P = new Float64Array(2 * nmax + 1);
  for (let i = 0; i < 2 * nmax + 1; i++) { const n = i - nmax; N[i] = n; const j = J[Math.abs(n)]; P[i] = j * j; }
  return { N, P };
}

/** Complex amplitudes c_N = e^{iN·argG} J_N(2|g|) for N = −nmax..nmax. */
export function sidebandAmplitudes(absg: number, argG: number, nmax = 30): { N: Int32Array; re: Float64Array; im: Float64Array } {
  const J = besselJ0toN(2 * Math.abs(absg), nmax);
  const N = new Int32Array(2 * nmax + 1), re = new Float64Array(2 * nmax + 1), im = new Float64Array(2 * nmax + 1);
  for (let i = 0; i < 2 * nmax + 1; i++) {
    const n = i - nmax; N[i] = n;
    let jn = J[Math.abs(n)]; if (n < 0 && (Math.abs(n) % 2 === 1)) jn = -jn;
    const ph = n * argG; re[i] = jn * Math.cos(ph); im[i] = jn * Math.sin(ph);
  }
  return { N, re, im };
}

// ── 2. Lorentzian-broadened EELS spectrum ──
/** Area-normalized Lorentzian, FWHM = fwhm, centre E0. */
export function lorentzian(E: number, E0: number, fwhm: number): number {
  const g = 0.5 * fwhm, d = E - E0;
  return (g / Math.PI) / (d * d + g * g);
}
/** S(E) = Σ_N P_N · L(E − N·ħω ; fwhm). Peak-normalized when `normalize`. */
export function pinemSpectrum(absg: number, energyGridEV: Float64Array, fwhmEV: number, photonEV: number, nmax = 30, normalize = true): Float64Array {
  const { N, P } = sidebandProbabilities(absg, nmax);
  const S = new Float64Array(energyGridEV.length);
  for (let i = 0; i < energyGridEV.length; i++) {
    const E = energyGridEV[i]; let s = 0;
    for (let k = 0; k < N.length; k++) { if (P[k] < 1e-12) continue; s += P[k] * lorentzian(E, N[k] * photonEV, fwhmEV); }
    S[i] = s;
  }
  if (normalize) { let mx = 0; for (const v of S) if (v > mx) mx = v; if (mx > 0) for (let i = 0; i < S.length; i++) S[i] /= mx; }
  return S;
}

// ── 3. co-moving-frame propagation ──
export type DispMethod = 'exact' | 'talbot';
/** Per-sideband propagation wavenumber ΔK_N [1/m]. */
export function comovingDeltaK(N: Int32Array, b: Beam, method: DispMethod): Float64Array {
  const dK = new Float64Array(N.length);
  const w = beamOmega(b);
  if (method === 'talbot') {
    const zT = talbotDistanceM(b);
    for (let i = 0; i < N.length; i++) dK[i] = (-2 * Math.PI * N[i] * N[i]) / zT;
  } else {
    const E0 = b.energyEV * QE, mc2 = ME * C_LIGHT * C_LIGHT, v = beamVelocity(b);
    const k0 = Math.sqrt((E0 + mc2) ** 2 - mc2 * mc2) / (HBAR * C_LIGHT);
    for (let i = 0; i < N.length; i++) {
      const EN = E0 + N[i] * HBAR * w;
      const kN = Math.sqrt((EN + mc2) ** 2 - mc2 * mc2) / (HBAR * C_LIGHT);
      dK[i] = kN - k0 - (N[i] * w) / v;
    }
  }
  return dK;
}

export interface DensityField { rho: Float64Array; nz: number; ntau: number; max: number; } // rho row-major [iz*ntau + it]
/** ρ(z,τ) = envelope(τ)·|Σ_N c_N e^{−iNωτ} e^{iΔK_N z}|², optionally peak-normalized. */
export function chargeDensity(
  absg: number, argG: number, zArr: Float64Array, tauArr: Float64Array, b: Beam,
  nmax = 30, method: DispMethod = 'exact', coherenceLengthM: number | null = null, normalize = true,
): DensityField {
  const full = sidebandAmplitudes(absg, argG, nmax);
  const dKfull = comovingDeltaK(full.N, b, method);
  // Prune negligible sidebands (|c_N| below 1e-4 → P_N < 1e-8): only a handful of
  // orders carry weight for typical |g|, so this is the main speed-up.
  const keep: number[] = [];
  for (let k = 0; k < full.N.length; k++) if (Math.hypot(full.re[k], full.im[k]) > 1e-4) keep.push(k);
  const nN = keep.length;
  const N = new Int32Array(nN), re = new Float64Array(nN), im = new Float64Array(nN), dK = new Float64Array(nN);
  for (let j = 0; j < nN; j++) { const k = keep[j]; N[j] = full.N[k]; re[j] = full.re[k]; im[j] = full.im[k]; dK[j] = dKfull[k]; }
  const w = beamOmega(b), nz = zArr.length, ntau = tauArr.length;
  const rho = new Float64Array(nz * ntau);
  let mx = 0;
  let sigmaTau = 0;   // envelope σ_τ from longitudinal coherence FWHM
  if (coherenceLengthM != null && coherenceLengthM > 0) {
    const sigmaZ = coherenceLengthM / (2 * Math.sqrt(2 * Math.log(2)));
    sigmaTau = sigmaZ / beamVelocity(b);
  }
  // Precompute the two phase factors so the O(nz·ntau·nN) core has no transcendentals:
  //   ct[k,it] = c_N · e^{−iNωτ}      (nN·ntau trig)
  //   zp[iz,k] = e^{iΔK_N z}          (nz·nN trig)
  const ctRe = new Float64Array(nN * ntau), ctIm = new Float64Array(nN * ntau);
  for (let k = 0; k < nN; k++) {
    const cr = re[k], ci = im[k], nw = N[k] * w, base = k * ntau;
    for (let it = 0; it < ntau; it++) { const a = -nw * tauArr[it], ca = Math.cos(a), sa = Math.sin(a); ctRe[base + it] = cr * ca - ci * sa; ctIm[base + it] = cr * sa + ci * ca; }
  }
  const zpRe = new Float64Array(nz * nN), zpIm = new Float64Array(nz * nN);
  for (let iz = 0; iz < nz; iz++) { const z = zArr[iz], base = iz * nN; for (let k = 0; k < nN; k++) { const a = dK[k] * z; zpRe[base + k] = Math.cos(a); zpIm[base + k] = Math.sin(a); } }
  for (let iz = 0; iz < nz; iz++) {
    const zoff = iz * nN;
    for (let it = 0; it < ntau; it++) {
      let sr = 0, si = 0;
      for (let k = 0; k < nN; k++) { const zr = zpRe[zoff + k], zi = zpIm[zoff + k], tr = ctRe[k * ntau + it], ti = ctIm[k * ntau + it]; sr += zr * tr - zi * ti; si += zr * ti + zi * tr; }
      let val = sr * sr + si * si;
      if (sigmaTau > 0) { const tau = tauArr[it]; val *= Math.exp(-(tau * tau) / (2 * sigmaTau * sigmaTau)); }
      rho[iz * ntau + it] = val;
      if (val > mx) mx = val;
    }
  }
  if (normalize && mx > 0) for (let i = 0; i < rho.length; i++) rho[i] /= mx;
  return { rho, nz, ntau, max: normalize ? 1 : mx };
}

/** 1D density profile ρ(τ) at a single distance z (peak-normalized). */
export function densityProfileAtZ(absg: number, argG: number, z: number, tauArr: Float64Array, b: Beam, nmax = 30, method: DispMethod = 'exact', coherenceLengthM: number | null = null): Float64Array {
  const f = chargeDensity(absg, argG, new Float64Array([z]), tauArr, b, nmax, method, coherenceLengthM, true);
  return f.rho;
}

// ── attosecond-pulse analysis ──
export interface FocusInfo {
  zFocusM: number;        // distance of maximum temporal bunching
  peakContrast: number;   // max density there (arb.)
  fwhmS: number;          // temporal FWHM of the central spike at the focus [s]
  spikeTauS: number;      // τ of the central spike [s]
}
/**
 * Locate the temporal focus (max bunching) by scanning z, then measure the
 * central-spike temporal FWHM and position there. Uses fine τ sampling over one
 * optical period around τ = 0.
 */
export function analyzeFocus(absg: number, argG: number, b: Beam, zMaxM: number, nmax = 30, method: DispMethod = 'exact'): FocusInfo {
  const T = beamPeriodS(b);
  const nTau = 601, nZ = 240;
  const tau = new Float64Array(nTau);
  for (let i = 0; i < nTau; i++) tau[i] = (-0.75 * T) + (1.5 * T * i) / (nTau - 1);
  const zArr = new Float64Array(nZ);
  for (let i = 0; i < nZ; i++) zArr[i] = (zMaxM * i) / (nZ - 1);
  // single (unnormalized) ρ(z,τ) evaluation over the scan grid — one optimized pass
  const { rho } = chargeDensity(absg, argG, zArr, tau, b, nmax, method, null, false);
  // best z = maximum bunching contrast (peak/mean over τ), skipping z=0 (flat)
  let bestZ = 0, bestContrast = -1, bestIz = 1;
  for (let iz = 1; iz < nZ; iz++) {
    const off = iz * nTau; let pk = 0, sum = 0;
    for (let it = 0; it < nTau; it++) { const v = rho[off + it]; if (v > pk) pk = v; sum += v; }
    const mean = sum / nTau, contrast = mean > 0 ? pk / mean : 0;
    if (contrast > bestContrast) { bestContrast = contrast; bestZ = zArr[iz]; bestIz = iz; }
  }
  // central-spike FWHM at the focus
  const off = bestIz * nTau; let imax = 0, vmax = 0;
  for (let i = 0; i < nTau; i++) { const v = rho[off + i]; if (v > vmax) { vmax = v; imax = i; } }
  const half = vmax / 2;
  let tl = tau[0], tr = tau[nTau - 1];
  for (let i = imax; i > 0; i--) { const a = rho[off + i], p = rho[off + i - 1]; if (a >= half && p < half) { const f = (half - p) / (a - p); tl = tau[i - 1] + f * (tau[i] - tau[i - 1]); break; } }
  for (let i = imax; i < nTau - 1; i++) { const a = rho[off + i], p = rho[off + i + 1]; if (a >= half && p < half) { const f = (half - a) / (p - a); tr = tau[i] + f * (tau[i + 1] - tau[i]); break; } }
  return { zFocusM: bestZ, peakContrast: bestContrast, fwhmS: Math.max(0, tr - tl), spikeTauS: tau[imax] };
}
