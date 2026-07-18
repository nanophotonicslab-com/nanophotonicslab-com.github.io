/**
 * Decay rates of a point electric dipole OUTSIDE a metallic nanosphere.
 * =====================================================================
 *
 * Extends the NPLab Mie library (mie.ts / materials.ts) with the dipole
 * decay-rate formalism of Kim, Leung & George, Surf. Sci. 195, 1 (1988) and
 * Sections 2.6-2.7 of G. Carella's MSc thesis — "Table 2.3: Definitions of
 * decay rates for a point dipole near metallic nanospheres". Ported from the
 * Spheres_dipoles validation package (validated against MNPBEM to ~1e-9).
 *
 * Conventions (matching the thesis / Kim et al.):
 *   - Time convention e^{-iωt}; lossy media have Im(ε) > 0, Im(m) > 0.
 *   - h_ℓ ≡ h_ℓ^{(1)} = j_ℓ + i·y_ℓ  (outgoing spherical Hankel).
 *   - a_ℓ = electric (TM) Mie reflection coeff, b_ℓ = magnetic (TE); Eq. (2.26).
 *   - "parallel / ∥"  = radial dipole p ∥ ẑ  (points at the sphere centre).
 *   - "perpendicular / ⊥" = tangential dipole p ∥ x̂,ŷ.
 *   - d  ≡ r'  = radial distance of the dipole from the sphere CENTRE.
 *              d/R = 1 sits on the surface; d/R = 1.02 ≈ 1 nm out for R = 50 nm.
 *   - All γ are dimensionless (power ÷ P₀); a bare dipole gives γ = 1.
 *
 * Numerical note: unlike the cross-section path in mie.ts (which uses upward
 * recurrence for ψ_ℓ, fine for ℓ ≲ x), the near-surface decay rates need many
 * multipoles at small argument (thesis truncates at ℓ_max = 20). We therefore
 * evaluate j_ℓ by DOWNWARD (Miller) recurrence and y_ℓ upward, which is stable
 * for ℓ ≫ x.
 */

import { epsilonToNK, drudeEpsilon, type DrudeParams } from './materials';

// ---------------------------------------------------------------------------
// Complex helpers — canonical implementations in complex.ts
// ---------------------------------------------------------------------------
import { type Complex, complex as cx, cadd, csub, cmul, cdiv, cscale, cabs2, csqrtUpper } from './complex';

/** Aliases kept for existing imports — the canonical type lives in complex.ts. */
export type Cx = Complex;
export const csqrt = csqrtUpper;

// ---------------------------------------------------------------------------
// Real spherical Bessel functions j_ℓ(x), y_ℓ(x) for ℓ = 0..lmax, real x > 0.
//   y_ℓ : upward recurrence  (stable — y is the growing solution)
//   j_ℓ : downward Miller recurrence, normalised to j_0 = sin x / x
// ---------------------------------------------------------------------------
export function sphBesselReal(x: number, lmax: number): { j: Float64Array; y: Float64Array } {
  const j = new Float64Array(lmax + 1);
  const y = new Float64Array(lmax + 1);

  // y_ℓ upward
  y[0] = -Math.cos(x) / x;
  if (lmax >= 1) y[1] = -Math.cos(x) / (x * x) - Math.sin(x) / x;
  for (let l = 1; l < lmax; l++) y[l + 1] = ((2 * l + 1) / x) * y[l] - y[l - 1];

  // j_ℓ downward (Miller). Start well above lmax and above x.
  const start = lmax + 16 + Math.ceil(Math.sqrt(40 * x));
  let jp1 = 0.0;          // j_{start+1}
  let jl = 1e-300;        // j_{start}  (arbitrary small seed)
  const tmp = new Float64Array(lmax + 1);
  for (let l = start; l >= 1; l--) {
    const jm1 = ((2 * l + 1) / x) * jl - jp1;   // j_{l-1}
    jp1 = jl; jl = jm1;
    if (l - 1 <= lmax) tmp[l - 1] = jl;
    // guard against overflow during the (unnormalised) downward sweep
    if (Math.abs(jl) > 1e250) {
      const s = 1e-250;
      jl *= s; jp1 *= s;
      for (let k = 0; k <= lmax; k++) tmp[k] *= s;
    }
  }
  const norm = (Math.sin(x) / x) / tmp[0];      // scale so j_0 matches analytic value
  for (let l = 0; l <= lmax; l++) j[l] = tmp[l] * norm;
  return { j, y };
}

// ---------------------------------------------------------------------------
// Complex logarithmic derivative D_ℓ(z) = ψ_ℓ'(z)/ψ_ℓ(z), ℓ = 0..lmax.
// Downward recurrence (same idea as mie.ts, generalised to return all orders):
//   D_{ℓ-1} = ℓ/z − 1/(D_ℓ + ℓ/z)
// ---------------------------------------------------------------------------
function logDerivComplex(z: Cx, lmax: number): Cx[] {
  const nstart = lmax + 16 + Math.ceil(Math.sqrt(40 * Math.hypot(z.re, z.im)));
  const D: Cx[] = new Array(lmax + 1);
  let d = cx(0, 0); // D_{nstart} ≈ 0
  for (let l = nstart; l >= 1; l--) {
    const loz = cdiv(cx(l, 0), z);              // ℓ/z
    d = csub(loz, cdiv(cx(1, 0), cadd(d, loz))); // D_{ℓ-1}
    if (l - 1 <= lmax) D[l - 1] = d;
  }
  return D;
}

// ---------------------------------------------------------------------------
// Complex Mie reflection coefficients a_ℓ, b_ℓ (thesis Eq. 2.26 == BH 4.88),
// robust to high ℓ. Non-magnetic media (μ₁ = μ₂). Index ℓ = 1..lmax → [ℓ-1].
//   x  = k₂ R      (size parameter, real, host lossless)
//   m  = n₁ / n₂   (relative refractive index, complex)
// ---------------------------------------------------------------------------
export function mieAB(nSphere: Cx, nHost: number, x: number, lmax: number): { a: Cx[]; b: Cx[] } {
  const m = cdiv(nSphere, cx(nHost, 0));
  const mx = cscale(m, x);
  const D = logDerivComplex(mx, lmax);

  // Real Riccati-Bessel at the host size parameter x = k₂R
  const { j, y } = sphBesselReal(x, lmax);
  const psi = (l: number) => x * j[l];
  const xi = (l: number): Cx => ({ re: x * j[l], im: x * y[l] }); // ξ_ℓ = x h_ℓ

  const a: Cx[] = [], b: Cx[] = [];
  for (let l = 1; l <= lmax; l++) {
    const nx = cx(l / x, 0);
    // A = D_ℓ/m + ℓ/x   ;   B = m·D_ℓ + ℓ/x
    const A = cadd(cdiv(D[l], m), nx);
    const B = cadd(cmul(m, D[l]), nx);

    const psiL = cx(psi(l), 0), psiL1 = cx(psi(l - 1), 0);
    const xiL = xi(l), xiL1 = xi(l - 1);

    // a_ℓ = (A ψ_ℓ − ψ_{ℓ-1}) / (A ξ_ℓ − ξ_{ℓ-1})
    a.push(cdiv(csub(cmul(A, psiL), psiL1), csub(cmul(A, xiL), xiL1)));
    // b_ℓ = (B ψ_ℓ − ψ_{ℓ-1}) / (B ξ_ℓ − ξ_{ℓ-1})
    b.push(cdiv(csub(cmul(B, psiL), psiL1), csub(cmul(B, xiL), xiL1)));
  }
  return { a, b };
}

// ---------------------------------------------------------------------------
// Decay rates (Table 2.3). One dipole position r' (= d), one frequency.
// ---------------------------------------------------------------------------
export interface DecayRates {
  gammaParRad: number;  // γ_∥^rad   (2.71)
  gammaPerpRad: number; // γ_⊥^rad   (2.72)
  gammaPar: number;     // γ_∥       (2.75)
  gammaPerp: number;    // γ_⊥       (2.76)
  gammaParNr: number;   // γ_∥^nr = γ_∥ − γ_∥^rad
  gammaPerpNr: number;  // γ_⊥^nr = γ_⊥ − γ_⊥^rad
}

/**
 * @param lambdaNm    emission wavelength in nm (ω = 2πc/λ)
 * @param epsSphere   complex permittivity of the sphere ε₁(ω)
 * @param nHost       host refractive index (real, lossless) — vacuum → 1
 * @param radiusNm    sphere radius R (nm)
 * @param rPrimeNm    dipole radial distance from centre r' = d (nm), r' ≥ R
 * @param lmax        multipole truncation (thesis: 20)
 */
export function decayRatesAt(
  lambdaNm: number,
  epsSphere: Cx,
  nHost: number,
  radiusNm: number,
  rPrimeNm: number,
  lmax: number,
): DecayRates {
  const nSphere = (() => { const [n, k] = epsilonToNK(epsSphere.re, epsSphere.im); return cx(n, k); })();

  const k2 = 2 * Math.PI * nHost / lambdaNm;     // host wavenumber (nm⁻¹)
  const x = k2 * radiusNm;                        // size parameter at R
  const u = k2 * rPrimeNm;                        // argument at the dipole r'

  const { a, b } = mieAB(nSphere, nHost, x, lmax);

  // Host spherical functions at the dipole position u = k₂ r'
  const { j, y } = sphBesselReal(u, lmax);
  const h = (l: number): Cx => ({ re: j[l], im: y[l] });          // h_ℓ(u)
  // Riccati derivatives at u:  ψ_ℓ'(u) = u j_{ℓ-1} − ℓ j_ℓ ;  ξ_ℓ'(u) = u h_{ℓ-1} − ℓ h_ℓ
  const psiP = (l: number) => u * j[l - 1] - l * j[l];
  const xiP = (l: number): Cx => ({ re: u * j[l - 1] - l * j[l], im: u * y[l - 1] - l * y[l] });

  let gParRad = 0, gPerpRad = 0;
  let gParTotSum = 0;   // Re Σ ℓ(ℓ+1)(2ℓ+1) a_ℓ [h_ℓ/u]²
  let gPerpTotSum = 0;  // Re Σ (2ℓ+1) ( a_ℓ [ξ'_ℓ/u]² + b_ℓ h_ℓ² )

  for (let l = 1; l <= lmax; l++) {
    const al = a[l - 1], bl = b[l - 1];
    const wPar = l * (l + 1) * (2 * l + 1);
    const wPerp = 2 * l + 1;

    // --- radiative rates (moduli squared) ---
    // γ_∥^rad (2.71): (3/2) Σ ℓ(ℓ+1)(2ℓ+1) |[ j_ℓ − a_ℓ h_ℓ ]/u|²
    const parNum = csub(cx(j[l], 0), cmul(al, h(l)));      // j_ℓ − a_ℓ h_ℓ
    gParRad += wPar * cabs2(parNum) / (u * u);

    // γ_⊥^rad (2.72): (3/4) Σ (2ℓ+1) { |j_ℓ − b_ℓ h_ℓ|² + |[ψ'_ℓ − a_ℓ ξ'_ℓ]/u|² }
    const perpB = csub(cx(j[l], 0), cmul(bl, h(l)));       // j_ℓ − b_ℓ h_ℓ
    const perpA = csub(cx(psiP(l), 0), cmul(al, xiP(l)));  // ψ'_ℓ − a_ℓ ξ'_ℓ
    gPerpRad += wPerp * (cabs2(perpB) + cabs2(perpA) / (u * u));

    // --- total rates (complex squares inside Re{…}) ---
    const hOverU = cscale(h(l), 1 / u);                    // h_ℓ/u
    const hSq = cmul(hOverU, hOverU);                      // (h_ℓ/u)²
    // γ_∥ (2.75)
    gParTotSum += wPar * cmul(al, hSq).re;
    // γ_⊥ (2.76): a_ℓ (ξ'_ℓ/u)² + b_ℓ h_ℓ²   (magnetic term has NO 1/u — matches
    // MNPBEM/Kim et al. 1988; the printed thesis Eq. 2.76 carries a stray 1/k₂r'
    // on the magnetic term.)
    const xiPU = cscale(xiP(l), 1 / u);
    const xiPSq = cmul(xiPU, xiPU);
    const hSqNoU = cmul(h(l), h(l));                       // h_ℓ²  (no /u)
    gPerpTotSum += wPerp * (cmul(al, xiPSq).re + cmul(bl, hSqNoU).re);
  }

  const gammaParRad = 1.5 * gParRad;
  const gammaPerpRad = 0.75 * gPerpRad;
  const gammaPar = 1 - 1.5 * gParTotSum;
  const gammaPerp = 1 - 0.75 * gPerpTotSum;

  return {
    gammaParRad,
    gammaPerpRad,
    gammaPar,
    gammaPerp,
    gammaParNr: gammaPar - gammaParRad,
    gammaPerpNr: gammaPerp - gammaPerpRad,
  };
}

// ---------------------------------------------------------------------------
// Convenience: silver Drude sphere as used for thesis Figs. 2.7 / 2.8.
// ε₁(ω) = ε_b − ω_p²/[ω(ω + iγ_D)] with the Table 1.1 silver parameters.
// ---------------------------------------------------------------------------
export const SILVER_DRUDE: DrudeParams = { epsB: 4.0, wpEv: 9.17, gammaEv: 0.021 };
export const HC_EV_NM = 1239.841984;
export const evToNm = (eV: number): number => HC_EV_NM / eV;

/** ε₁(ω) for the silver Drude sphere at a given wavelength (nm). */
export function silverEpsilon(lambdaNm: number): Cx {
  const e = drudeEpsilon(lambdaNm, SILVER_DRUDE);
  return { re: e.re, im: e.im };
}
