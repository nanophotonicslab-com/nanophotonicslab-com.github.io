/**
 * Transition radiation (TR) in cathodoluminescence — single interface and
 * free-standing film.
 *
 * TypeScript port of the validated Python reference `trlib` (the TR_CL
 * project): `trlib/single_interface.py`, `trlib/thin_film.py`,
 * `trlib/resonances.py`, `trlib/kinematics.py`, with the conventions of
 * `TR_CL/docs/THEORY.md`. The port is formula-for-formula; nothing is
 * re-derived here.
 *
 * References
 *   - Ritchie & Eldridge, Phys. Rev. 126, 1935 (1962)        — free-standing film, Eqs. (14), (15), (24)
 *   - García de Abajo, Rev. Mod. Phys. 82, 209 (2010)        — single interface
 *   - Brenny, PhD thesis, UvA/AMOLF (2016), §3.3–3.5         — Eqs. (3.8)–(3.10), (3.23), (3.24)
 *   - Ebel et al., Adv. Funct. Mater. 36, e76737 (2026)      — Eqs. (1)–(3), (S1)–(S2)
 *   - Ginzburg & Frank (1946)                                — perfect-conductor limit
 *
 * Conventions (THEORY.md — these must not be changed, the fixture depends on them)
 *   - time dependence exp(−iωt); absorbing media have Im ε > 0;
 *   - z-components of wave vectors take the branch Im q_z ≥ 0 (`sqrtImPos`);
 *   - θ is measured from the film/surface normal;
 *   - `'backward'` = emission towards the electron source (signed velocity b = −β),
 *     `'forward'` = emission downstream (b = +β);
 *   - probabilities are photons per electron, per nm of wavelength (and per sr
 *     where the name says so).
 *
 * Units: wavelength nm, film thickness nm, electron kinetic energy keV, angles rad,
 * photon energy eV. The field formulas of the single interface are evaluated in
 * Gaussian (CGS) units, exactly as in the references and in `trlib`.
 *
 * Numerics worth keeping
 *   - the film's γ and Δ are both multiplied by exp(itσ), so only exp(2itσ)
 *     (modulus ≤ 1 because Im σ ≥ 0) ever appears — a thick metal film would
 *     otherwise overflow;
 *   - the single-interface amplitude uses the closed form for a vacuum emission
 *     half-space (`trlib._amplitude_vacuum_side`): one complex square root, and
 *     the wavenumber cancels analytically;
 *   - the thickness-independent film coefficients are evaluated once per (θ, λ)
 *     and reused across thicknesses, in flat typed arrays with no allocation in
 *     the inner loop (the UI sweeps ~150 × 200 × 96 points).
 *
 * Complex numbers follow the hot-loop conventions documented in `complex.ts`:
 * the tuple type `C = [re, im]` of `cylinder.ts` at the API surface, and bare
 * scalar (re, im) pairs inside the kernels, as in `mie.ts`.
 */

import { type C } from './cylinder';

/** Re-exported so callers need only this module: the tuple complex of cylinder.ts. */
export type { C };

// ── constants (CODATA 2018, the same values as trlib/constants.py) ──

/** Fine-structure constant. */
const ALPHA = 7.2973525693e-3;
/** Electron rest energy [keV]. */
const ME_C2_KEV = 510.99895;
/** Speed of light [m/s]. */
const C_SI = 299792458.0;
/** Speed of light [cm/s] (Gaussian units). */
const C_CGS = 2.99792458e10;
/** Elementary charge [statC]. */
const E_CGS = 4.803204712570263e-10;
/** Reduced Planck constant [erg s]. */
const HBAR_CGS = 1.054571817e-27;
/** Reduced Planck constant [eV s]. */
const HBAR_EV_S = 6.582119569e-16;
/** Planck constant [eV s]. */
const H_EV_S = 4.135667696e-15;
/**
 * hc [eV nm]. The full CODATA value, not the 1239.841984 used elsewhere in
 * src/lib — trlib carries all digits and the per-eV fixture values follow it;
 * the truncated constant would cost 2.7e-13 of relative parity.
 */
export const HC_EV_NM = 1239.8419843320026;

const PI = Math.PI;
const TWO_PI = 2 * Math.PI;

// ── types ──

/** Which hemisphere the light is collected in. */
export type TrSide = 'backward' | 'forward';

/** Spectral density the probability is expressed per. */
export type SpectralDensity = 'nm' | 'eV' | 'omega';

/** The three Ritchie/Ebel film terms: entrance interface, film interior, exit interface. */
export type FilmTerm = 'B1' | 'B2' | 'B3';

export const ALL_FILM_TERMS: readonly FilmTerm[] = ['B1', 'B2', 'B3'];

/** Permittivity: one complex value, one per wavelength, or a function of λ [nm]. */
export type Permittivity = C | readonly C[] | ((lambdaNm: number) => C);

export interface FilmTermOptions {
  /** Subset of (B1, B2, B3) to keep — the control calculations of Ebel Fig. S1. */
  terms?: readonly FilmTerm[];
  /** +1 (physical) or −1 to flip the sign of the B3 term (Ebel Fig. S1b). */
  b3Sign?: number;
}

export interface AngleIntegrationOptions {
  /** Lower edge of the collected polar band [rad]; default 0. */
  thetaMinRad?: number;
  /** Upper edge of the collected polar band [rad]; default π/2 (full hemisphere). */
  thetaMaxRad?: number;
  /** Gauss–Legendre order in θ; default 96 (the trlib default). */
  nTheta?: number;
}

export interface InterfaceSpectrumOptions extends AngleIntegrationOptions {
  wavelengthNm: ArrayLike<number>;
  eps: Permittivity;
  energyKeV: number;
  side?: TrSide;
  per?: SpectralDensity;
}

export interface FilmSpectrumOptions extends AngleIntegrationOptions, FilmTermOptions {
  wavelengthNm: ArrayLike<number>;
  eps: Permittivity;
  thicknessNm: number;
  energyKeV: number;
  side?: TrSide;
  /** 'nm' (default) or 'eV'; 'omega' is not defined for the film formulas. */
  per?: 'nm' | 'eV';
}

export interface ThicknessMapOptions extends Omit<FilmSpectrumOptions, 'thicknessNm'> {
  thicknessNm: ArrayLike<number>;
}

// ── kinematics (trlib/kinematics.py) ──

/** Lorentz factor γ = 1 + E_kin/(m_e c²). */
export const gammaLorentz = (energyKeV: number): number => 1 + energyKeV / ME_C2_KEV;

/** Electron speed in units of c. */
export function betaFromKeV(energyKeV: number): number {
  const g = gammaLorentz(energyKeV);
  return Math.sqrt(1 - 1 / (g * g));
}

/** Kinetic energy [keV] of an electron with β = v/c. */
export const energyFromBeta = (beta: number): number =>
  ME_C2_KEV * (1 / Math.sqrt(1 - beta * beta) - 1);

/** Photon energy [eV] from free-space wavelength [nm]. */
export const wavelengthToEv = (lambdaNm: number): number => HC_EV_NM / lambdaNm;

/** Free-space wavelength [nm] from photon energy [eV]. */
export const evToWavelength = (energyEv: number): number => HC_EV_NM / energyEv;

/** Angular frequency [rad/s] from free-space wavelength [nm]. */
export const wavelengthToOmega = (lambdaNm: number): number => TWO_PI * C_SI / (lambdaNm * 1e-9);

/** P_λ = P_ω |dω/dλ| — spectral density per rad/s → per nm. */
export const perOmegaToPerNm = (pOmega: number, lambdaNm: number): number =>
  pOmega * TWO_PI * C_SI / ((lambdaNm * 1e-9) * (lambdaNm * 1e-9)) * 1e-9;

/** Spectral density per rad/s → per eV. */
export const perOmegaToPerEv = (pOmega: number): number => pOmega / HBAR_EV_S;

/** P_E = P_λ λ²/(hc) — spectral density per nm → per eV. */
export const perNmToPerEv = (pNm: number, lambdaNm: number): number =>
  pNm * lambdaNm * lambdaNm / HC_EV_NM;

function fromPerOmega(pOmega: number, lambdaNm: number, per: SpectralDensity): number {
  if (per === 'omega') return pOmega;
  if (per === 'nm') return perOmegaToPerNm(pOmega, lambdaNm);
  return perOmegaToPerEv(pOmega);
}

// ── complex helpers ──

/**
 * Square root on the branch Im √z ≥ 0 (and Re ≥ 0 when Im √z = 0) — the
 * physical branch for z-components of wave vectors with exp(−iωt)
 * (`trlib._common.sqrt_im_pos`; `csqrtUpper` in complex.ts is the same choice
 * for the struct-style complex type).
 */
export function sqrtImPos(z: C): C {
  const r = Math.hypot(z[0], z[1]);
  const im = Math.sqrt(Math.max(0, 0.5 * (r - z[0])));
  const re = Math.sqrt(Math.max(0, 0.5 * (r + z[0])));
  return z[1] < 0 ? [-re, im] : [re, im];
}

/** σ = √(ε − 1 + μ²) = q_z/k inside the film (Im σ ≥ 0). */
export const filmSigma = (eps: C, mu: number): C => sqrtImPos([eps[0] - 1 + mu * mu, eps[1]]);

/**
 * |re + i·im| for the kernels. `Math.hypot` is variadic and roughly an order of
 * magnitude slower in V8, and it is called once per (θ, λ) cell; the direct
 * form is exact here and falls back to `Math.hypot` in the only cases where it
 * would not be (over/underflow of the square).
 */
function fastAbs(re: number, im: number): number {
  const m = re * re + im * im;
  return m > 0 && m < Infinity ? Math.sqrt(m) : Math.hypot(re, im);
}

// ── Gauss–Legendre quadrature ──
//
// Re-implemented here rather than imported: the only other copy in the repo is
// module-local to electron-sphere.ts and stops Newton one step early (its weight
// uses P'_n at the previous iterate). These nodes agree with numpy's
// `polynomial.legendre.leggauss` — what trlib integrates with — to ~1e-15, so
// the angle-integrated spectra match the Python reference to round-off.

const GL_CACHE = new Map<number, { x: Float64Array; w: Float64Array }>();

/** Nodes and weights of the n-point Gauss–Legendre rule on [−1, 1] (cached per n). */
function gaussLegendreUnit(n: number): { x: Float64Array; w: Float64Array } {
  const hit = GL_CACHE.get(n);
  if (hit) return hit;
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  const half = (n + 1) >> 1;
  for (let i = 0; i < half; i++) {
    // Chebyshev-like starting guess for the i-th root, then Newton on P_n.
    let z = Math.cos(PI * (i + 0.75) / (n + 0.5));
    for (let iter = 0; iter < 100; iter++) {
      let p1 = 1;
      let p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      const dz = p1 * (z * z - 1) / (n * (z * p1 - p2));
      z -= dz;
      if (Math.abs(dz) < 2e-16) break;
    }
    // P'_n at the converged node, so the weight is as accurate as the node.
    let p1 = 1;
    let p2 = 0;
    for (let j = 1; j <= n; j++) {
      const p3 = p2;
      p2 = p1;
      p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
    }
    const pp = n * (z * p1 - p2) / (z * z - 1);
    const wi = 2 / ((1 - z * z) * pp * pp);
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = wi;
    w[n - 1 - i] = wi;
  }
  const rule = { x, w };
  GL_CACHE.set(n, rule);
  return rule;
}

/** Nodes and weights of the n-point Gauss–Legendre rule on [a, b]. */
export function gaussLegendre(a: number, b: number, n: number): { x: Float64Array; w: Float64Array } {
  const base = gaussLegendreUnit(n);
  const s = 0.5 * (b - a);
  const m = 0.5 * (b + a);
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = s * base.x[i] + m;
    w[i] = s * base.w[i];
  }
  return { x, w };
}

// ── permittivity plumbing ──

/** Interleaved [re₀, im₀, re₁, im₁, …] permittivity, one entry per wavelength. */
function resolveEps(eps: Permittivity, wavelengthNm: ArrayLike<number>): Float64Array {
  const n = wavelengthNm.length;
  const out = new Float64Array(2 * n);
  if (typeof eps === 'function') {
    for (let i = 0; i < n; i++) {
      const e = eps(wavelengthNm[i]);
      out[2 * i] = e[0];
      out[2 * i + 1] = e[1];
    }
    return out;
  }
  // A single [re, im] tuple starts with a number; a per-wavelength list with a tuple.
  if (typeof (eps as readonly unknown[])[0] === 'number') {
    const e = eps as C;
    for (let i = 0; i < n; i++) {
      out[2 * i] = e[0];
      out[2 * i + 1] = e[1];
    }
    return out;
  }
  const list = eps as readonly C[];
  if (list.length !== n) {
    throw new Error(`eps has ${list.length} entries but ${n} wavelengths were given`);
  }
  for (let i = 0; i < n; i++) {
    out[2 * i] = list[i][0];
    out[2 * i + 1] = list[i][1];
  }
  return out;
}

/** Signed velocity b of the Ritchie/Ebel formulas: −β backward, +β forward. */
const signedBeta = (beta: number, side: TrSide): number => (side === 'backward' ? -beta : beta);

// ============================================================================
// 1. Single interface (García de Abajo RMP 2010; Brenny Eqs. 3.8–3.10, 3.23–3.24)
// ============================================================================

/**
 * Closed form of f = i k cosθ D μ_j for a vacuum emission half-space
 * (`trlib._amplitude_vacuum_side`), written into `out[0..1]`.
 *
 * With s = sinθ, c = cosθ, σ = √(ε − s²) (Im ≥ 0) and a = s² + 1/β², the
 * wavenumber cancels and
 *
 *   backward (vacuum above, material below, Eq. 3.9):
 *     f = −(2 e s c/c₀) / (σ + ε|c|) · [ (σ − 1/β)/(a − ε) − (σ − ε/β)/(a − 1) ]
 *   forward (material above, vacuum below, Eq. 3.10):
 *     f = −(2 e s c/c₀) / (σ + ε|c|) · [ (σ + ε/β)/(a − 1) − (σ + 1/β)/(a − ε) ]
 *
 * `eps` is in both cases the permittivity of the *material* half-space.
 */
function interfaceAmplitudeKernel(
  sinT: number, cosT: number, invBeta: number, er: number, ei: number,
  backward: boolean, out: Float64Array, off: number,
): void {
  const ac = Math.abs(cosT);
  // σ = √(ε − sin²θ), Im σ ≥ 0
  const zr = er - sinT * sinT;
  const rr = fastAbs(zr, ei);
  const si = Math.sqrt(Math.max(0, 0.5 * (rr - zr)));
  const sr0 = Math.sqrt(Math.max(0, 0.5 * (rr + zr)));
  const sr = ei < 0 ? -sr0 : sr0;

  const a = sinT * sinT + invBeta * invBeta;
  // (a − ε) and (a − 1)
  const dEr = a - er;
  const dEi = -ei;
  const dE = dEr * dEr + dEi * dEi;
  const d1 = a - 1;

  let brR: number;
  let brI: number;
  if (backward) {
    // (σ − 1/β)/(a − ε)
    const n1r = sr - invBeta;
    const n1i = si;
    const t1r = (n1r * dEr + n1i * dEi) / dE;
    const t1i = (n1i * dEr - n1r * dEi) / dE;
    // (σ − ε/β)/(a − 1)
    const t2r = (sr - er * invBeta) / d1;
    const t2i = (si - ei * invBeta) / d1;
    brR = t1r - t2r;
    brI = t1i - t2i;
  } else {
    // (σ + ε/β)/(a − 1)
    const t1r = (sr + er * invBeta) / d1;
    const t1i = (si + ei * invBeta) / d1;
    // (σ + 1/β)/(a − ε)
    const n2r = sr + invBeta;
    const n2i = si;
    const t2r = (n2r * dEr + n2i * dEi) / dE;
    const t2i = (n2i * dEr - n2r * dEi) / dE;
    brR = t1r - t2r;
    brI = t1i - t2i;
  }

  const pref = (-2 * E_CGS / C_CGS) * sinT * cosT;
  // f = pref · br / (σ + ε|c|)
  const qr = sr + er * ac;
  const qi = si + ei * ac;
  const q = qr * qr + qi * qi;
  out[off] = pref * (brR * qr + brI * qi) / q;
  out[off + 1] = pref * (brI * qr - brR * qi) / q;
}

/**
 * Complex far-field amplitude f (Gaussian units) of the single interface, such
 * that E_θ = f e^{ikr}/r (Brenny Eq. 3.23). The field is purely p-polarized.
 *
 * Deliberately has no wavelength argument: in the vacuum-side closed form the
 * wavenumber cancels analytically, so f depends on λ only through ε — which the
 * caller supplies. `eps` is the permittivity of the material half-space.
 */
export function interfaceAmplitude(
  thetaRad: number, energyKeV: number, eps: C, side: TrSide = 'backward',
): C {
  const out = new Float64Array(2);
  interfaceAmplitudeKernel(
    Math.sin(thetaRad), Math.cos(thetaRad), 1 / betaFromKeV(energyKeV),
    eps[0], eps[1], side === 'backward', out, 0,
  );
  return [out[0], out[1]];
}

/**
 * Differential emission probability of the single interface,
 * dΓ/dΩdω = |f|²/(4π²ħk) (Brenny Eq. 3.24), converted to the requested
 * spectral density — photons per electron per sr per nm by default.
 */
export function interfaceSrPerNm(
  thetaRad: number, lambdaNm: number, energyKeV: number, eps: C,
  side: TrSide = 'backward', per: SpectralDensity = 'nm',
): number {
  const f = interfaceAmplitude(thetaRad, energyKeV, eps, side);
  const k = TWO_PI / (lambdaNm * 1e-7);                 // 1/cm
  const pOmega = (f[0] * f[0] + f[1] * f[1]) / (4 * PI * PI * HBAR_CGS * k);
  return fromPerOmega(pOmega, lambdaNm, per);
}

/**
 * Angle-integrated single-interface emission, 2π ∫ (dΓ/dΩ) sinθ dθ over
 * [thetaMin, thetaMax] — photons per electron per nm by default.
 * Returns one value per wavelength.
 */
export function interfaceSpectrum(opts: InterfaceSpectrumOptions): Float64Array {
  const {
    wavelengthNm, eps, energyKeV, side = 'backward', per = 'nm',
    thetaMinRad = 0, thetaMaxRad = PI / 2, nTheta = 96,
  } = opts;
  const nL = wavelengthNm.length;
  const out = new Float64Array(nL);
  if (nL === 0) return out;

  const e = resolveEps(eps, wavelengthNm);
  const { x: th, w: wt } = gaussLegendre(thetaMinRad, thetaMaxRad, nTheta);
  const invBeta = 1 / betaFromKeV(energyKeV);
  const backward = side === 'backward';
  const amp = new Float64Array(2);

  // Angle factors do not depend on λ.
  const sinT = new Float64Array(nTheta);
  const cosT = new Float64Array(nTheta);
  const solid = new Float64Array(nTheta);
  for (let i = 0; i < nTheta; i++) {
    sinT[i] = Math.sin(th[i]);
    cosT[i] = Math.cos(th[i]);
    solid[i] = TWO_PI * wt[i] * sinT[i];
  }

  for (let l = 0; l < nL; l++) {
    const lam = wavelengthNm[l];
    const er = e[2 * l];
    const ei = e[2 * l + 1];
    const k = TWO_PI / (lam * 1e-7);
    const norm = 1 / (4 * PI * PI * HBAR_CGS * k);
    let acc = 0;
    for (let i = 0; i < nTheta; i++) {
      interfaceAmplitudeKernel(sinT[i], cosT[i], invBeta, er, ei, backward, amp, 0);
      acc += solid[i] * (amp[0] * amp[0] + amp[1] * amp[1]) * norm;
    }
    out[l] = fromPerOmega(acc, lam, per);
  }
  return out;
}

/**
 * Ginzburg–Frank perfect-conductor limit,
 * dN/dωdΩ = α β² sin²θ / (π² ω (1 − β²cos²θ)²).
 */
export function perfectConductorSrPerNm(
  thetaRad: number, lambdaNm: number, energyKeV: number, per: SpectralDensity = 'nm',
): number {
  const b = betaFromKeV(energyKeV);
  const omega = (TWO_PI / (lambdaNm * 1e-7)) * C_CGS;
  const s = Math.sin(thetaRad);
  const c = Math.cos(thetaRad);
  const den = 1 - b * b * c * c;
  const pOmega = ALPHA * b * b * s * s / (PI * PI * omega * den * den);
  return fromPerOmega(pOmega, lambdaNm, per);
}

// ============================================================================
// 2. Free-standing film (Ritchie & Eldridge 1962 Eq. 14/15; Ebel Eqs. 1–2, S1–S2)
// ============================================================================
//
// I(θ,λ,d,β) = αβ²/(π²λ) · μ²(1−μ²) · |γ/Δ|²,  μ = cosθ, σ = √(ε−1+μ²), t = 2πd/λ
//
//   γ = B₁e^{−itσ} + B₂e^{itσ} − B₃e^{it/β},   Δ = (με−σ)²e^{itσ} − (με+σ)²e^{−itσ}
//
// Both are multiplied by e^{itσ} here, giving
//
//   γ' = c₁ + c₂e^{2itσ} − c₃ e^{itσ}e^{−it/b},   Δ' = P e^{2itσ} − M
//
// with P = (με−σ)², M = (με+σ)² and the signed velocity b (= −β backward).
// |e^{2itσ}| ≤ 1, so thick lossy films underflow to the single-interface limit
// instead of overflowing.

/** Doubles stored per (θ, λ) cell of the coefficient cache. */
const PART_STRIDE = 12;

/**
 * Thickness-independent film coefficients for one (μ, ε): writes
 * [c₁, c₂, c₃, σ, P, M] (re, im each) into `out` at `off`.
 */
function filmPartsKernel(
  mu: number, b: number, er: number, ei: number,
  keep1: boolean, keep2: boolean, keep3: boolean, b3Sign: number,
  out: Float64Array, off: number,
): void {
  // σ = √(ε − 1 + μ²), Im σ ≥ 0
  const zr = er - 1 + mu * mu;
  const rr = fastAbs(zr, ei);
  const si = Math.sqrt(Math.max(0, 0.5 * (rr - zr)));
  const sr0 = Math.sqrt(Math.max(0, 0.5 * (rr + zr)));
  const sr = ei < 0 ? -sr0 : sr0;

  const om = 1 - mu * mu * b * b;          // 1 − μ²b², real
  const mer = mu * er;
  const mei = mu * ei;
  const t1r = mer + sr;                    // με + σ
  const t1i = mei + si;
  const t2r = mer - sr;                    // με − σ
  const t2i = mei - si;

  // B₁ = [ (bσ + ε)/(1 − μ²b²) − 1/(1 − bσ) ] (με + σ)
  const ur = (b * sr + er) / om;
  const ui = (b * si + ei) / om;
  const w1r = 1 - b * sr;
  const w1i = -b * si;
  const w1 = w1r * w1r + w1i * w1i;
  const q1r = ur - w1r / w1;
  const q1i = ui + w1i / w1;
  const B1r = q1r * t1r - q1i * t1i;
  const B1i = q1r * t1i + q1i * t1r;

  // B₂ = [ (bσ − ε)/(1 − μ²b²) + 1/(1 + bσ) ] (με − σ)
  const vr = (b * sr - er) / om;
  const vi = (b * si - ei) / om;
  const w2r = 1 + b * sr;
  const w2i = b * si;
  const w2 = w2r * w2r + w2i * w2i;
  const q2r = vr + w2r / w2;
  const q2i = vi - w2i / w2;
  const B2r = q2r * t2r - q2i * t2i;
  const B2i = q2r * t2i + q2i * t2r;

  // B₃ = 2σ [ ε/(1 − bμ) − (1 + bεμ)/(1 − b²σ²) ]
  const obm = 1 / (1 - b * mu);
  const y1r = er * obm;
  const y1i = ei * obm;
  const nr = 1 + b * mu * er;
  const ni = b * mu * ei;
  const s2r = sr * sr - si * si;           // σ²
  const s2i = 2 * sr * si;
  const ddr = 1 - b * b * s2r;
  const ddi = -b * b * s2i;
  const dd = ddr * ddr + ddi * ddi;
  const y2r = (nr * ddr + ni * ddi) / dd;
  const y2i = (ni * ddr - nr * ddi) / dd;
  const gr = y1r - y2r;
  const gi = y1i - y2i;
  const B3r = 2 * (sr * gr - si * gi);
  const B3i = 2 * (sr * gi + si * gr);

  // Dropped terms are stored as zero, so the inner loop never branches.
  out[off] = keep1 ? B1r : 0;
  out[off + 1] = keep1 ? B1i : 0;
  out[off + 2] = keep2 ? B2r : 0;
  out[off + 3] = keep2 ? B2i : 0;
  out[off + 4] = keep3 ? b3Sign * B3r : 0;
  out[off + 5] = keep3 ? b3Sign * B3i : 0;
  out[off + 6] = sr;
  out[off + 7] = si;
  out[off + 8] = t2r * t2r - t2i * t2i;    // P = (με − σ)²
  out[off + 9] = 2 * t2r * t2i;
  out[off + 10] = t1r * t1r - t1i * t1i;   // M = (με + σ)²
  out[off + 11] = 2 * t1r * t1i;
}

function termFlags(terms: readonly FilmTerm[] | undefined): [boolean, boolean, boolean] {
  if (!terms) return [true, true, true];
  return [terms.includes('B1'), terms.includes('B2'), terms.includes('B3')];
}

/**
 * Complex ratio γ/Δ of the film (dimensionless), with both numerator and
 * denominator carrying the stabilising factor e^{itσ}.
 */
export function filmGammaOverDelta(
  thetaRad: number, lambdaNm: number, thicknessNm: number, energyKeV: number, eps: C,
  side: TrSide = 'backward', opts: FilmTermOptions = {},
): C {
  const [k1, k2, k3] = termFlags(opts.terms);
  const b = signedBeta(betaFromKeV(energyKeV), side);
  const p = new Float64Array(PART_STRIDE);
  filmPartsKernel(Math.cos(thetaRad), b, eps[0], eps[1], k1, k2, k3, opts.b3Sign ?? 1, p, 0);
  const t = TWO_PI * thicknessNm / lambdaNm;
  const sr = p[6];
  const si = p[7];
  const mag = Math.exp(-t * si);
  const e1r = mag * Math.cos(t * sr);
  const e1i = mag * Math.sin(t * sr);
  const e2r = e1r * e1r - e1i * e1i;
  const e2i = 2 * e1r * e1i;
  const phR = Math.cos(-t / b);
  const phI = Math.sin(-t / b);
  const epr = e1r * phR - e1i * phI;
  const epi = e1r * phI + e1i * phR;
  const gr = p[0] + (p[2] * e2r - p[3] * e2i) - (p[4] * epr - p[5] * epi);
  const gi = p[1] + (p[2] * e2i + p[3] * e2r) - (p[4] * epi + p[5] * epr);
  const dr = p[8] * e2r - p[9] * e2i - p[10];
  const di = p[8] * e2i + p[9] * e2r - p[11];
  const d2 = dr * dr + di * di;
  return [(gr * dr + gi * di) / d2, (gi * dr - gr * di) / d2];
}

/**
 * Photons per electron per sr per nm from a free-standing film
 * (Ritchie Eq. 14 / Ebel Eq. 1). `per: 'eV'` converts to photons per eV.
 */
export function filmSrPerNm(
  thetaRad: number, lambdaNm: number, thicknessNm: number, energyKeV: number, eps: C,
  side: TrSide = 'backward', opts: FilmTermOptions & { per?: 'nm' | 'eV' } = {},
): number {
  const beta = betaFromKeV(energyKeV);
  const mu = Math.cos(thetaRad);
  const r = filmGammaOverDelta(thetaRad, lambdaNm, thicknessNm, energyKeV, eps, side, opts);
  const I = ALPHA * beta * beta / (PI * PI * lambdaNm) * mu * mu * (1 - mu * mu)
    * (r[0] * r[0] + r[1] * r[1]);
  return opts.per === 'eV' ? perNmToPerEv(I, lambdaNm) : I;
}

/**
 * Complex p-polarized film amplitude a_θ with |a_θ|² = photons/(electron sr nm).
 * The global phase is arbitrary (Ritchie only gives |γ/Δ|²) but the relative
 * phase between angles at fixed λ is meaningful; the sign of μ√(1−μ²) is kept
 * so the field is odd under θ → −θ, like a radially polarized dipole field.
 */
export function filmAmplitude(
  thetaRad: number, lambdaNm: number, thicknessNm: number, energyKeV: number, eps: C,
  side: TrSide = 'backward', opts: FilmTermOptions = {},
): C {
  const beta = betaFromKeV(energyKeV);
  const pref = Math.sqrt(ALPHA * beta * beta / (PI * PI * lambdaNm))
    * Math.cos(thetaRad) * Math.sin(thetaRad);
  const r = filmGammaOverDelta(thetaRad, lambdaNm, thicknessNm, energyKeV, eps, side, opts);
  return [pref * r[0], pref * r[1]];
}

/**
 * Cached thickness-independent film coefficients on a (λ, θ) grid, plus the
 * angular quadrature weight folded together with the αβ²μ²(1−μ²)/(π²λ)
 * prefactor. Built once, reused for every thickness.
 *
 * Layout is λ-major: cell (l, i) lives at ((l·nTheta) + i)·12, so the inner
 * loop over θ at fixed λ walks contiguous memory.
 */
interface FilmCoefficients {
  nTheta: number;
  nLambda: number;
  parts: Float64Array;
  weight: Float64Array;
  /** 2π/λ [1/nm], so that t = d · k0[l]. */
  k0: Float64Array;
  /** Signed velocity. */
  b: number;
}

function buildFilmCoefficients(
  wavelengthNm: ArrayLike<number>, eps: Float64Array, energyKeV: number, side: TrSide,
  thetaMinRad: number, thetaMaxRad: number, nTheta: number, opts: FilmTermOptions,
): FilmCoefficients {
  const nL = wavelengthNm.length;
  const beta = betaFromKeV(energyKeV);
  const b = signedBeta(beta, side);
  const [k1, k2, k3] = termFlags(opts.terms);
  const b3Sign = opts.b3Sign ?? 1;
  const { x: th, w: wt } = gaussLegendre(thetaMinRad, thetaMaxRad, nTheta);

  const parts = new Float64Array(nL * nTheta * PART_STRIDE);
  const weight = new Float64Array(nL * nTheta);
  const k0 = new Float64Array(nL);
  const mu = new Float64Array(nTheta);
  const ang = new Float64Array(nTheta);
  for (let i = 0; i < nTheta; i++) {
    mu[i] = Math.cos(th[i]);
    // 2π w_θ sinθ · μ²(1 − μ²)
    ang[i] = TWO_PI * wt[i] * Math.sin(th[i]) * mu[i] * mu[i] * (1 - mu[i] * mu[i]);
  }
  for (let l = 0; l < nL; l++) {
    const lam = wavelengthNm[l];
    k0[l] = TWO_PI / lam;
    const pref = ALPHA * beta * beta / (PI * PI * lam);
    const er = eps[2 * l];
    const ei = eps[2 * l + 1];
    const row = l * nTheta;
    for (let i = 0; i < nTheta; i++) {
      filmPartsKernel(mu[i], b, er, ei, k1, k2, k3, b3Sign, parts, (row + i) * PART_STRIDE);
      weight[row + i] = ang[i] * pref;
    }
  }
  return { nTheta, nLambda: nL, parts, weight, k0, b };
}

/**
 * Angle-integrated spectra for a list of thicknesses, written into `out`
 * (row-major in thickness).
 *
 * Loop order is λ → θ → d so that the twelve cached coefficients of a (θ, λ)
 * cell are read into registers once and reused for every thickness, and the
 * thickness phase e^{−it/b} — which does not depend on θ — is tabulated once
 * per wavelength. The θ sum is accumulated in a column buffer and scattered to
 * `out` at the end of each wavelength; the summation order over θ is unchanged.
 */
function integrateFilm(co: FilmCoefficients, thicknessNm: ArrayLike<number>, out: Float64Array): void {
  const { nTheta, nLambda, parts, weight, k0, b } = co;
  const nD = thicknessNm.length;
  const tPhase = new Float64Array(nD);
  const phRe = new Float64Array(nD);
  const phIm = new Float64Array(nD);
  const col = new Float64Array(nD);

  for (let l = 0; l < nLambda; l++) {
    const k = k0[l];
    for (let j = 0; j < nD; j++) {
      const t = thicknessNm[j] * k;
      tPhase[j] = t;
      phRe[j] = Math.cos(-t / b);
      phIm[j] = Math.sin(-t / b);
    }
    col.fill(0);
    const row = l * nTheta;
    for (let i = 0; i < nTheta; i++) {
      const off = (row + i) * PART_STRIDE;
      const c1r = parts[off];
      const c1i = parts[off + 1];
      const c2r = parts[off + 2];
      const c2i = parts[off + 3];
      const c3r = parts[off + 4];
      const c3i = parts[off + 5];
      const sr = parts[off + 6];
      const si = parts[off + 7];
      const Pr = parts[off + 8];
      const Pi = parts[off + 9];
      const Mr = parts[off + 10];
      const Mi = parts[off + 11];
      const w = weight[row + i];
      for (let j = 0; j < nD; j++) {
        const t = tPhase[j];
        // e^{itσ} (modulus e^{−t Im σ} ≤ 1) and its square
        const mag = Math.exp(-t * si);
        const ang = t * sr;
        const e1r = mag * Math.cos(ang);
        const e1i = mag * Math.sin(ang);
        const e2r = e1r * e1r - e1i * e1i;
        const e2i = 2 * e1r * e1i;
        // γ' = c₁ + c₂e^{2itσ} − c₃ e^{itσ}e^{−it/b}
        const pr = phRe[j];
        const pi = phIm[j];
        const epr = e1r * pr - e1i * pi;
        const epi = e1r * pi + e1i * pr;
        const gr = c1r + (c2r * e2r - c2i * e2i) - (c3r * epr - c3i * epi);
        const gi = c1i + (c2r * e2i + c2i * e2r) - (c3r * epi + c3i * epr);
        // Δ' = P e^{2itσ} − M
        const dr = Pr * e2r - Pi * e2i - Mr;
        const di = Pr * e2i + Pi * e2r - Mi;
        col[j] += w * ((gr * gr + gi * gi) / (dr * dr + di * di));
      }
    }
    for (let j = 0; j < nD; j++) out[j * nLambda + l] = col[j];
  }
}

/**
 * Angle-integrated film emission, 2π ∫ I sinθ dθ over [thetaMin, thetaMax] —
 * photons per electron per nm (or per eV). One value per wavelength.
 */
export function filmSpectrum(opts: FilmSpectrumOptions): Float64Array {
  const {
    wavelengthNm, eps, thicknessNm, energyKeV, side = 'backward', per = 'nm',
    thetaMinRad = 0, thetaMaxRad = PI / 2, nTheta = 96,
  } = opts;
  const nL = wavelengthNm.length;
  const out = new Float64Array(nL);
  if (nL === 0) return out;
  const co = buildFilmCoefficients(
    wavelengthNm, resolveEps(eps, wavelengthNm), energyKeV, side,
    thetaMinRad, thetaMaxRad, nTheta, opts,
  );
  integrateFilm(co, [thicknessNm], out);
  if (per === 'eV') {
    for (let l = 0; l < nL; l++) out[l] = perNmToPerEv(out[l], wavelengthNm[l]);
  }
  return out;
}

/**
 * Angle-integrated film spectra on a thickness × wavelength grid.
 * Row-major in thickness: `grid[di * wavelengthNm.length + li]`.
 *
 * The (θ, λ) coefficients are built once and reused for every thickness, which
 * is what makes the UI's ~150 × 200 × 96 map interactive.
 */
export function thicknessMap(opts: ThicknessMapOptions): Float64Array {
  const {
    wavelengthNm, eps, thicknessNm, energyKeV, side = 'backward', per = 'nm',
    thetaMinRad = 0, thetaMaxRad = PI / 2, nTheta = 96,
  } = opts;
  const nL = wavelengthNm.length;
  const nD = thicknessNm.length;
  const out = new Float64Array(nD * nL);
  if (nL === 0 || nD === 0) return out;
  const co = buildFilmCoefficients(
    wavelengthNm, resolveEps(eps, wavelengthNm), energyKeV, side,
    thetaMinRad, thetaMaxRad, nTheta, opts,
  );
  integrateFilm(co, thicknessNm, out);
  if (per === 'eV') {
    for (let di = 0; di < nD; di++) {
      for (let l = 0; l < nL; l++) {
        out[di * nL + l] = perNmToPerEv(out[di * nL + l], wavelengthNm[l]);
      }
    }
  }
  return out;
}

// ── film limits ──

/**
 * Ritchie's thick-foil limit (d → ∞), photons/(electron sr nm):
 *
 *   I_∞ = αβ²/(π²λ) · μ²(1−μ²)/(1−β²μ²)² · |(ε−1)(1−β²∓βσ) / ((σ+με)(1∓βσ))|²
 *
 * (upper signs forward, lower signs backward). For `side: 'backward'` this is
 * the single-interface TR of an electron entering the material — a cross-check
 * of two independent formalisms.
 */
export function thickFilmLimitSrPerNm(
  thetaRad: number, lambdaNm: number, energyKeV: number, eps: C, side: TrSide = 'backward',
): number {
  const beta = betaFromKeV(energyKeV);
  const b = signedBeta(beta, side);
  const mu = Math.cos(thetaRad);
  const s = filmSigma(eps, mu);
  const er = eps[0];
  const ei = eps[1];
  // numerator (ε − 1)(1 − β² − bσ)
  const ar = er - 1;
  const ai = ei;
  const cr = 1 - beta * beta - b * s[0];
  const ci = -b * s[1];
  const nr = ar * cr - ai * ci;
  const ni = ar * ci + ai * cr;
  // denominator (σ + με)(1 − bσ)(1 − β²μ²)
  const p1r = s[0] + mu * er;
  const p1i = s[1] + mu * ei;
  const p2r = 1 - b * s[0];
  const p2i = -b * s[1];
  const scale = 1 - beta * beta * mu * mu;
  const dr = (p1r * p2r - p1i * p2i) * scale;
  const di = (p1r * p2i + p1i * p2r) * scale;
  const amp2 = (nr * nr + ni * ni) / (dr * dr + di * di);
  return ALPHA * beta * beta / (PI * PI * lambdaNm) * mu * mu * (1 - mu * mu) * amp2;
}

/**
 * Ritchie's small-thickness expansion (Eq. 24), photons/(electron sr nm), valid
 * for ωa/v ≪ 1 and |ε|^{1/2} ωa/c ≪ 1:
 *
 *   n = α t²/(4π²λ) · μ²(1−μ²) · |1 − εμβ − β²|²/(1−β²μ²)²
 *       · |ε−1|² / |με − i(μ²ε + σ²) t/2|²,   t = 2πd/λ
 */
export function thinFilmLimitSrPerNm(
  thetaRad: number, lambdaNm: number, thicknessNm: number, energyKeV: number, eps: C,
): number {
  const beta = betaFromKeV(energyKeV);
  const mu = Math.cos(thetaRad);
  const s = filmSigma(eps, mu);
  const er = eps[0];
  const ei = eps[1];
  const t = TWO_PI * thicknessNm / lambdaNm;
  const pref = ALPHA * t * t / (4 * PI * PI * lambdaNm);
  // |1 − εμβ − β²|²
  const ur = 1 - er * mu * beta - beta * beta;
  const ui = -ei * mu * beta;
  // |ε − 1|²
  const vr = er - 1;
  const vi = ei;
  // με − i(μ²ε + σ²) t/2
  const s2r = s[0] * s[0] - s[1] * s[1];
  const s2i = 2 * s[0] * s[1];
  const wr = mu * er + 0.5 * t * (mu * mu * ei + s2i);
  const wi = mu * ei - 0.5 * t * (mu * mu * er + s2r);
  const dmu = 1 - beta * beta * mu * mu;
  return pref * mu * mu * (1 - mu * mu) * (ur * ur + ui * ui) / (dmu * dmu)
    * (vr * vr + vi * vi) / (wr * wr + wi * wi);
}

// ============================================================================
// 3. Resonances, Cherenkov and escape cone (trlib/resonances.py; Ebel Eq. 3)
// ============================================================================

/**
 * Interface-interference phase φ = t Re σ_θ + t/β with σ_θ = √(ε − sin²θ) and
 * t = 2πd/λ. Constructive interference at φ = 2π(m − ½) (Ebel Eqs. 3, S5–S6).
 */
export function interferencePhase(
  lambdaNm: number, thicknessNm: number, energyKeV: number, eps: C, thetaRad = 0,
): number {
  const sinT = Math.sin(thetaRad);
  const s = sqrtImPos([eps[0] - sinT * sinT, eps[1]]);
  const t = TWO_PI * thicknessNm / lambdaNm;
  return t * s[0] + t / betaFromKeV(energyKeV);
}

/**
 * Film thickness [nm] at which interference order m is resonant at this
 * wavelength: d_m = (m − ½) λ β / (1 + β Re σ_θ) (Ebel Eq. 3, solved for d —
 * the white overlays of Ebel Fig. 1d / S1).
 *
 * Takes the full complex ε, because the reference takes Re of the square root,
 * not the square root of Re ε.
 */
export function resonanceThicknessNm(
  lambdaNm: number, energyKeV: number, eps: C, m = 1, thetaRad = Math.PI / 3,
): number {
  const beta = betaFromKeV(energyKeV);
  const sinT = Math.sin(thetaRad);
  const s = sqrtImPos([eps[0] - sinT * sinT, eps[1]]);
  return (m - 0.5) * lambdaNm * beta / (1 + beta * s[0]);
}

/** Resonance photon energy [eV] of order m for a non-dispersive ε (Ebel Eq. 3). */
export function resonanceEnergyEv(
  thicknessNm: number, energyKeV: number, eps: C, m = 1, thetaRad = Math.PI / 3,
): number {
  const beta = betaFromKeV(energyKeV);
  const sinT = Math.sin(thetaRad);
  const s = sqrtImPos([eps[0] - sinT * sinT, eps[1]]);
  return H_EV_S * (beta * C_SI) / (thicknessNm * 1e-9) * (m - 0.5) / (1 + beta * s[0]);
}

/** Electron energy [keV] above which v > c/n, with n = √(Re ε). */
export const cherenkovThresholdKeV = (epsRe: number): number => energyFromBeta(1 / Math.sqrt(epsRe));

/** Cherenkov angle inside the medium [rad], cos θ_C = 1/(nβ); NaN below threshold. */
export function cherenkovAngleRad(energyKeV: number, epsRe: number): number {
  const x = 1 / (Math.sqrt(epsRe) * betaFromKeV(energyKeV));
  return x <= 1 ? Math.acos(Math.max(-1, Math.min(1, x))) : NaN;
}

/** Critical angle for total internal reflection at the film/vacuum interface [rad]. */
export const tirAngleRad = (epsRe: number): number => Math.asin(1 / Math.sqrt(epsRe));

/** True if Cherenkov light can leave a planar film (θ_C < θ_TIR). */
export const cherenkovEscapes = (energyKeV: number, epsRe: number): boolean =>
  cherenkovAngleRad(energyKeV, epsRe) < tirAngleRad(epsRe);

/**
 * Brenny's boundary angle θ = arcsin √(Re ε), defined where 0 < Re ε < 1, which
 * separates backward emission from coupling into the medium (thesis Figs. 3.6, 3.11).
 */
export function lightLineAngleRad(epsRe: number): number {
  return epsRe > 0 && epsRe < 1 ? Math.asin(Math.sqrt(epsRe)) : NaN;
}
