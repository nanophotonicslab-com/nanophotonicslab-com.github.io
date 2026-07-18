// Generalized Smith–Purcell (GSP) physics — TypeScript port of the `gsp_lib` reference
// package (Dias, Rodríguez Echarri, Rasmussen, García de Abajo & Cox, Light: Sci. Appl.
// 15, 218 (2026)). Energies in eV, lengths in nm, angles in radians unless a name ends in
// "Deg". Covers the core needed by the Smith–Purcell lab module: the GSP condition (Eq. 6),
// engineered dipole distributions (Eqs. 7–8), the vector far field (Eq. 2) with full
// polarization analysis (Stokes), and the swift-electron elliptical dipole (Eq. 9).

import { type Complex, complex, cadd, csub, cmul, cscale, cconj, cabs2, cabs } from './complex';

/** Aliases kept for existing imports — the canonical type lives in complex.ts. */
export type Cpx = Complex;
export type Vec3c = [Cpx, Cpx, Cpx];

const PI = Math.PI;
export const c = complex;
export { cabs };

// ── constants / unit helpers (gsp_lib.constants) ──
export const HBAR_EVS = 6.582119569e-16;          // ħ [eV·s]
export const C_NM_S = 2.99792458e17;              // c [nm/s]
export const EV_TO_INV_NM = 1 / (HBAR_EVS * C_NM_S);  // 1 eV → k [1/nm]
export const wavelengthNmFromEV = (E_eV: number): number => 2 * PI / (E_eV * EV_TO_INV_NM);
export const eVFromWavelengthNm = (lam_nm: number): number => 2 * PI / (lam_nm * EV_TO_INV_NM);
export const omegaFromEV = (E_eV: number): number => E_eV / HBAR_EVS;

// ── geometry: SP / GSP conditions (gsp_lib.geometry, Eqs. 3, 6) ──
export const spCondition = (n: number, beta: number, aOverLambda: number): number =>
  1 / beta - n / aOverLambda;

export const gspCondition = (n: number, ell: number, N: number, beta: number, aOverLambda: number): number =>
  1 / beta - (n - ell / N) / aOverLambda;

export const gspAngleDeg = (n: number, ell: number, N: number, beta: number, aOverLambda: number): number => {
  const s = gspCondition(n, ell, N, beta, aOverLambda);
  return Math.abs(s) <= 1 ? Math.asin(s) * 180 / PI : NaN;
};

export interface Channel { n: number; ell: number; sin: number; thetaDeg: number; }
export function radiativeChannels(N: number, beta: number, aOverLambda: number,
  nRange: number[] = [-3, -2, -1, 0, 1, 2, 3], ellRange?: number[]): Channel[] {
  const ells = ellRange ?? Array.from({ length: N }, (_, i) => i);
  const out: Channel[] = [];
  for (const n of nRange) for (const ell of ells) {
    const s = gspCondition(n, ell, N, beta, aOverLambda);
    if (Math.abs(s) <= 1) out.push({ n, ell, sin: s, thetaDeg: Math.asin(s) * 180 / PI });
  }
  return out;
}

// Invert Eq. (6) → λ/λ₀ such that channel (n,ℓ) emits at θ (spectral plots, Fig. 3a/S2a).
export const wavelengthRatioForAngle = (thetaDeg: number, n: number, ell: number, N: number,
  beta: number, aOverLambda0: number): number =>
  aOverLambda0 * (1 / beta - Math.sin(thetaDeg * PI / 180)) / (n - ell / N);

// ── engineered dipole distributions (gsp_lib.distributions, Eqs. 7–8) ──
export const dipoleDistribution = (N: number, xi: number, A = 1.0): number[] =>
  Array.from({ length: N }, (_, j) => 1 + A * Math.sin(2 * PI * xi * j / N));

// DFT mode amplitudes p̃_ℓ/p₀ = (1/N) Σ_j p_j e^{−2πijℓ/N}  (Eq. 7).
export function fourierModes(p0norm: number[]): Cpx[] {
  const N = p0norm.length, out: Cpx[] = [];
  for (let l = 0; l < N; l++) {
    let re = 0, im = 0;
    for (let j = 0; j < N; j++) { const ph = -2 * PI * j * l / N; re += p0norm[j] * Math.cos(ph); im += p0norm[j] * Math.sin(ph); }
    out.push(c(re / N, im / N));
  }
  return out;
}

// ── modified Bessel functions K₀, K₁ (Abramowitz & Stegun 9.8) for the electron field ──
function besselI0(x: number): number {
  const t = x / 3.75, t2 = t * t;
  if (Math.abs(x) < 3.75)
    return 1 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
  const ax = Math.abs(x), y = 3.75 / ax;
  return (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
}
function besselI1(x: number): number {
  const ax = Math.abs(x);
  let ans: number;
  if (ax < 3.75) {
    const t = x / 3.75, t2 = t * t;
    ans = ax * (0.5 + t2 * (0.87890594 + t2 * (0.51498869 + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))));
  } else {
    const y = 3.75 / ax;
    ans = (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967 + y * (-0.02895312 + y * (0.01787654 + y * -0.00420059))))))));
  }
  return x < 0 ? -ans : ans;
}
export function besselK0(x: number): number {
  if (x <= 0) return NaN;
  if (x <= 2) {
    const t2 = (x / 2) * (x / 2);
    return -Math.log(x / 2) * besselI0(x) + (-0.57721566 + t2 * (0.42278420 + t2 * (0.23069756 + t2 * (0.03488590 + t2 * (0.00262698 + t2 * (0.00010750 + t2 * 0.00000740))))));
  }
  const y = 2 / x;
  return (Math.exp(-x) / Math.sqrt(x)) * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))));
}
export function besselK1(x: number): number {
  if (x <= 0) return NaN;
  if (x <= 2) {
    const t2 = (x / 2) * (x / 2);
    return (Math.log(x / 2) * besselI1(x)) + (1 / x) * (1 + t2 * (0.15443144 + t2 * (-0.67278579 + t2 * (-0.18156897 + t2 * (-0.01919402 + t2 * (-0.00110404 + t2 * -0.00004686))))));
  }
  const y = 2 / x;
  return (Math.exp(-x) / Math.sqrt(x)) * (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268 + y * (-0.00780353 + y * (0.00325614 + y * -0.00068245))))));
}

// ── swift-electron external field & elliptical dipole (gsp_lib.electron, Eq. 9) ──
// E_ext ∝ (i/γ) K₀(ωb/vγ) x̂ + K₁(ωb/vγ) ẑ. The factor i makes the dipole elliptical.
export function externalFieldOnArray(omega: number, b: number, v: number, gamma: number): { Ex: Cpx; Ez: Cpx } {
  const arg = omega * b / (v * gamma);
  const pref = 2 * omega / (v * v * gamma);   // electron charge = 1 (overall scale)
  return { Ex: c(0, (pref / gamma) * besselK0(arg)), Ez: c(pref * besselK1(arg), 0) };
}
export const fieldMagnitudeX = (omega: number, b: number, v: number, gamma: number): number =>
  cabs(externalFieldOnArray(omega, b, v, gamma).Ex);

export function dipoleOrientation(omega: number, b: number, v: number, gamma: number, normalize = true): Vec3c {
  const { Ex, Ez } = externalFieldOnArray(omega, b, v, gamma);
  if (!normalize) return [Ex, c(0), Ez];
  const nrm = Math.sqrt(cabs2(Ex) + cabs2(Ez));
  return [cscale(Ex, 1 / nrm), c(0), cscale(Ez, 1 / nrm)];
}

// ── dipole-vector builders (gsp_lib.farfield) ──
export const xPolarized = (mag: number[]): Vec3c[] => mag.map(m => [c(m), c(0), c(0)]);
export const yPolarized = (mag: number[]): Vec3c[] => mag.map(m => [c(0), c(m), c(0)]);
export const zPolarized = (mag: number[]): Vec3c[] => mag.map(m => [c(0), c(0), c(m)]);
// Common complex orientation `dir` scaled by per-element magnitudes (elliptical dipoles).
export const oriented = (mag: number[], dir: Vec3c): Vec3c[] =>
  mag.map(m => [cscale(dir[0], m), cscale(dir[1], m), cscale(dir[2], m)]);

// ── vector far field & polarization (gsp_lib.farfield, Eq. 2) ──
// f(θ,φ) = k²(1 − r̂⊗r̂)·Σ_j p_j exp[i k a j (1/β − sinθ cosφ)].
export function farFieldVector(theta: number, phi: number, p0vec: Vec3c[], aOverLambda: number, beta: number, k = 1.0): Vec3c {
  const ka = 2 * PI * aOverLambda;
  const arg = ka * (1 / beta - Math.sin(theta) * Math.cos(phi));
  let Sx = c(0), Sy = c(0), Sz = c(0);
  for (let j = 0; j < p0vec.length; j++) {
    const ph = arg * j, e = c(Math.cos(ph), Math.sin(ph));
    Sx = cadd(Sx, cmul(e, p0vec[j][0])); Sy = cadd(Sy, cmul(e, p0vec[j][1])); Sz = cadd(Sz, cmul(e, p0vec[j][2]));
  }
  const rx = Math.sin(theta) * Math.cos(phi), ry = Math.sin(theta) * Math.sin(phi), rz = Math.cos(theta);
  const rdotS = cadd(cadd(cscale(Sx, rx), cscale(Sy, ry)), cscale(Sz, rz));
  const k2 = k * k;
  return [cscale(csub(Sx, cscale(rdotS, rx)), k2), cscale(csub(Sy, cscale(rdotS, ry)), k2), cscale(csub(Sz, cscale(rdotS, rz)), k2)];
}

export const farFieldAmplitudePhi0 = (theta: number, p0vec: Vec3c[], aOverLambda: number, beta: number, k = 1.0): number => {
  const f = farFieldVector(theta, 0, p0vec, aOverLambda, beta, k);
  return Math.sqrt(cabs2(f[0]) + cabs2(f[1]) + cabs2(f[2]));
};

// Decompose Cartesian far field into (f_θ, f_φ) = (p-pol, s-pol) spherical components.
export function sphericalComponents(theta: number, phi: number, f: Vec3c): { fTheta: Cpx; fPhi: Cpx } {
  const thx = Math.cos(theta) * Math.cos(phi), thy = Math.cos(theta) * Math.sin(phi), thz = -Math.sin(theta);
  const phx = -Math.sin(phi), phy = Math.cos(phi);
  const fTheta = cadd(cadd(cscale(f[0], thx), cscale(f[1], thy)), cscale(f[2], thz));
  const fPhi = cadd(cscale(f[0], phx), cscale(f[1], phy));   // φ̂ has no z component
  return { fTheta, fPhi };
}

// Stokes parameters in the (θ̂, φ̂) basis: S1 = p−s linear, S2 = ±45°, S3 = circular.
export function stokes(fTheta: Cpx, fPhi: Cpx): [number, number, number, number] {
  const S0 = cabs2(fTheta) + cabs2(fPhi);
  const S1 = cabs2(fTheta) - cabs2(fPhi);
  const ab = cmul(fTheta, cconj(fPhi));
  return [S0, S1, 2 * ab.re, 2 * ab.im];
}

// Polarization ellipse: orientation ψ, ellipticity χ (sign = handedness), degree of pol.
export function polarizationEllipse(fTheta: Cpx, fPhi: Cpx): { psi: number; chi: number; dop: number } {
  const [S0, S1, S2, S3] = stokes(fTheta, fPhi);
  const d = S0 === 0 ? 1 : S0;
  return {
    psi: 0.5 * Math.atan2(S2, S1),
    chi: 0.5 * Math.asin(Math.max(-1, Math.min(1, S3 / d))),
    dop: Math.sqrt(S1 * S1 + S2 * S2 + S3 * S3) / d,
  };
}

// Parametric trace (u,v) = (Re[f_θ e^{−iωt}], Re[f_φ e^{−iωt}]) of the real polarization ellipse.
export function ellipseTrace(fTheta: Cpx, fPhi: Cpx, n = 80): { u: number[]; v: number[] } {
  const u: number[] = [], v: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = 2 * PI * i / (n - 1), cs = Math.cos(t), sn = -Math.sin(t);   // e^{−it}
    u.push(fTheta.re * cs - fTheta.im * sn);
    v.push(fPhi.re * cs - fPhi.im * sn);
  }
  return { u, v };
}
