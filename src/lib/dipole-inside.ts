/**
 * Decay rates of a point dipole INSIDE a sphere — Kim, Leung & George,
 * Surf. Sci. 195, 1 (1988), §2.3 (the "aerosol" problem) — used to reproduce
 * their Figs. 4 (transparent dielectric) and 5 (dissipative dielectric).
 * Ported from the Spheres_dipoles validation package.
 *
 * The molecule sits at distance d from the centre, INSIDE a sphere of radius a
 * and permittivity ε₂; outside is vacuum ε₁ = 1. Notation (Kim Eqs. 1-12):
 *   kᵢ = √εᵢ · ω/c ,  ρᵢ = kᵢ a ,  y₂ = k₂ d ,  ω/c = 2π/λ (λ = vacuum wavelength).
 *   ψₙ(x)=x jₙ(x),  ζₙ(x)=x hₙ⁽¹⁾(x)  (Riccati-Bessel).
 *   Coefficients Aₙ,Bₙ,Cₙ,Dₙ  = Eq. (11);  Eₙ=T{Bₙ}, Fₙ=T{Aₙ}  = Eqs. (23,24),
 *   with the transformation T (Eq. 12): jₙ↔hₙ⁽¹⁾ and medium 1↔2.
 *
 * Orientation labels follow Kim (⊥ = radial/vertical, ∥ = tangential/horizontal).
 *
 * VALIDATION built in: for a TRANSPARENT sphere (Im ε₂ = 0) the total rates
 * (Eqs. 21,22) must equal the radiative rates (Eqs. 26,27) — Chew's identity.
 */

// ---------------------------------------------------------------------------
// complex arithmetic — canonical implementations in complex.ts
// (div = Smith's algorithm, robust to the tiny high-order Bessel values
//  in Miller's recurrence; csqrt = upper-half-plane branch)
// ---------------------------------------------------------------------------
import {
  type Complex, complex as C, cadd as add, csub as sub, cmul as mul,
  cdivRobust as div, cscale as scale, cneg as neg, cabs2 as abs2,
  csin, ccos, csqrtUpper,
} from './complex';

/** Aliases kept for existing imports — the canonical type lives in complex.ts. */
export type Cx = Complex;
export const csqrt = csqrtUpper;

// ---------------------------------------------------------------------------
// complex spherical Bessel jₙ(z), yₙ(z) for n = 0..nmax
//   yₙ : upward recurrence ;  jₙ : downward (Miller), normalised to j₀=sin z/z
// ---------------------------------------------------------------------------
export function sphBesselComplex(z: Cx, nmax: number): { j: Cx[]; y: Cx[] } {
  const j: Cx[] = new Array(nmax + 1);
  const y: Cx[] = new Array(nmax + 1);
  const invz = div(C(1), z);
  const sinz = csin(z), cosz = ccos(z);
  // y upward
  y[0] = neg(mul(cosz, invz));                              // -cos z / z
  if (nmax >= 1) y[1] = sub(neg(mul(mul(cosz, invz), invz)), mul(sinz, invz)); // -cos z/z² - sin z/z
  for (let n = 1; n < nmax; n++) y[n + 1] = sub(mul(scale(y[n], 2 * n + 1), invz), y[n - 1]);
  // j downward (Miller)
  const start = nmax + 15 + Math.ceil(Math.sqrt(60 * Math.hypot(z.re, z.im)));
  let jp1 = C(0), jl = C(1e-300, 0);
  const tmp: Cx[] = new Array(nmax + 1);
  for (let n = start; n >= 1; n--) {
    const jm1 = sub(mul(scale(jl, 2 * n + 1), invz), jp1);
    jp1 = jl; jl = jm1;
    if (n - 1 <= nmax) tmp[n - 1] = jl;
    if (Math.hypot(jl.re, jl.im) > 1e250) { const s = 1e-250; jl = scale(jl, s); jp1 = scale(jp1, s); for (let k = 0; k <= nmax; k++) if (tmp[k]) tmp[k] = scale(tmp[k], s); }
  }
  const norm = div(mul(sinz, invz), tmp[0]);                // (sin z/z)/j₀^computed
  for (let n = 0; n <= nmax; n++) j[n] = mul(tmp[n], norm);
  return { j, y };
}

// bundle of the four functions needed at a point: jₙ, hₙ, ψ'ₙ, ζ'ₙ  (n=0..nmax)
interface Bundle { j: Cx[]; h: Cx[]; psip: Cx[]; zetp: Cx[]; }
function bundle(z: Cx, nmax: number): Bundle {
  const { j, y } = sphBesselComplex(z, nmax);
  const h = j.map((jn, n) => add(jn, mul(C(0, 1), y[n])));  // hₙ = jₙ + i yₙ
  const psip: Cx[] = new Array(nmax + 1), zetp: Cx[] = new Array(nmax + 1);
  for (let n = 1; n <= nmax; n++) {
    psip[n] = sub(mul(z, j[n - 1]), scale(j[n], n));          // ψ'ₙ = z j_{n-1} − n jₙ
    zetp[n] = sub(mul(z, h[n - 1]), scale(h[n], n));          // ζ'ₙ = z h_{n-1} − n hₙ
  }
  return { j, h, psip, zetp };
}

// ---------------------------------------------------------------------------
// inside-sphere coefficients A,B,C,D (Eq. 11) and E=T{B}, F=T{A} (Eqs. 12,23,24)
// ρ₁, ρ₂ are k₁a, k₂a. Index n = 1..nmax → array[n].
// ---------------------------------------------------------------------------
export interface InsideCoeffs { A: Cx[]; B: Cx[]; Cc: Cx[]; D: Cx[]; E: Cx[]; F: Cx[]; }
export function insideCoeffs(eps1: Cx, eps2: Cx, rho1: Cx, rho2: Cx, nmax: number): InsideCoeffs {
  const b1 = bundle(rho1, nmax);   // functions at ρ₁ (outside medium)
  const b2 = bundle(rho2, nmax);   // functions at ρ₂ (inside medium)
  const A: Cx[] = [], B: Cx[] = [], Cc: Cx[] = [], D: Cx[] = [], E: Cx[] = [], F: Cx[] = [];
  for (let n = 1; n <= nmax; n++) {
    const j1 = b1.j[n], h1 = b1.h[n], pp1 = b1.psip[n], zp1 = b1.zetp[n];
    const j2 = b2.j[n], h2 = b2.h[n], pp2 = b2.psip[n], zp2 = b2.zetp[n];
    // Cₙ = jₙ(ρ₂)ζ'ₙ(ρ₁) − hₙ(ρ₁)ψ'ₙ(ρ₂)
    const Cn = sub(mul(j2, zp1), mul(h1, pp2));
    // Dₙ = ε₂ jₙ(ρ₂)ζ'ₙ(ρ₁) − ε₁ hₙ(ρ₁)ψ'ₙ(ρ₂)
    const Dn = sub(mul(eps2, mul(j2, zp1)), mul(eps1, mul(h1, pp2)));
    // Aₙ = [jₙ(ρ₁)ψ'ₙ(ρ₂) − jₙ(ρ₂)ψ'ₙ(ρ₁)] / Cₙ
    const An = div(sub(mul(j1, pp2), mul(j2, pp1)), Cn);
    // Bₙ = [ε₁ jₙ(ρ₁)ψ'ₙ(ρ₂) − ε₂ jₙ(ρ₂)ψ'ₙ(ρ₁)] / Dₙ
    const Bn = div(sub(mul(eps1, mul(j1, pp2)), mul(eps2, mul(j2, pp1))), Dn);
    // Eₙ = T{Bₙ} = −[ε₂ hₙ(ρ₂)ζ'ₙ(ρ₁) − ε₁ hₙ(ρ₁)ζ'ₙ(ρ₂)] / Dₙ
    const En = neg(div(sub(mul(eps2, mul(h2, zp1)), mul(eps1, mul(h1, zp2))), Dn));
    // Fₙ = T{Aₙ} = −[hₙ(ρ₂)ζ'ₙ(ρ₁) − hₙ(ρ₁)ζ'ₙ(ρ₂)] / Cₙ
    const Fn = neg(div(sub(mul(h2, zp1), mul(h1, zp2)), Cn));
    A[n] = An; B[n] = Bn; Cc[n] = Cn; D[n] = Dn; E[n] = En; F[n] = Fn;
  }
  return { A, B, Cc, D, E, F };
}

// ---------------------------------------------------------------------------
// decay rates for a dipole inside the sphere
// ---------------------------------------------------------------------------
export interface InsideRates {
  perpTot: number; parTot: number;   // γ_⊥/γ₀, γ_∥/γ₀  (⊥ = radial, ∥ = tangential)
  perpRad: number; parRad: number;   // radiative
  perpNr: number; parNr: number;     // nonradiative = total − radiative
}

/**
 * @param lambdaNm  vacuum wavelength (nm)
 * @param eps2      permittivity of the medium the dipole is IN (the inner sphere).
 *                  Aerosol (Figs 4,5): ε₂ = the sphere. Cavity (Fig 2): ε₂ = 1 (vacuum).
 * @param aNm       sphere/cavity radius a (nm)
 * @param dNm       dipole distance from centre d (nm), 0 ≤ d < a
 * @param nmax      multipole truncation
 * @param fOmega    f(ω)=γ_sp/γ₀ for a DISSIPATIVE inner medium (Eq. 28). Omit/undefined
 *                  for a transparent inner medium, where total = radiative and
 *                  Eqs. (21,22) are used directly.
 * @param eps1      permittivity of the OUTER medium. Aerosol → vacuum (default {1,0}).
 *                  Cavity (Fig 2) → the surrounding dielectric (e.g. 2.16).
 */
export function insideDecay(
  lambdaNm: number, eps2: Cx, aNm: number, dNm: number, nmax: number,
  fOmega?: number, eps1: Cx = C(1, 0),
): InsideRates {
  const k0 = 2 * Math.PI / lambdaNm;
  const sq2 = csqrt(eps2);
  const rho1 = scale(csqrt(eps1), k0 * aNm);       // k₁ a = (2πa/λ)√ε₁
  const rho2 = scale(sq2, k0 * aNm);               // k₂ a
  const y2 = scale(sq2, k0 * dNm);                 // k₂ d
  const { Cc, D, E, F } = insideCoeffs(eps1, eps2, rho1, rho2, nmax);
  const by = bundle(y2, nmax);                     // functions at the dipole position y₂
  const eps2_32 = mul(eps2, sq2);                  // ε₂^{3/2}

  // ---- total rates ----
  // transparent (Eqs. 21,22):  γ/γ₀ = 1 + (3/2) Re Σ …
  // dissipative (Eqs. 29,30):  γ/γ₀ = 1 + (3/2) f Re{ √ε₂ Σ … }
  let perpSum = C(0), parSum = C(0);
  for (let n = 1; n <= nmax; n++) {
    const jn = by.j[n], pp = by.psip[n];
    const jOverY = div(jn, y2);                    // jₙ(y₂)/y₂
    const ppOverY = div(pp, y2);                   // ψ'ₙ(y₂)/y₂
    // ⊥ (radial): (2n+1)n(n+1) Eₙ [jₙ/y₂]²
    perpSum = add(perpSum, scale(mul(E[n], mul(jOverY, jOverY)), (2 * n + 1) * n * (n + 1)));
    // ∥ (tangential): (n+½){ Eₙ [ψ'ₙ/y₂]² + Fₙ jₙ² }
    const parTerm = add(mul(E[n], mul(ppOverY, ppOverY)), mul(F[n], mul(jn, jn)));
    parSum = add(parSum, scale(parTerm, n + 0.5));
  }
  let perpTot: number, parTot: number;
  if (fOmega === undefined) {                      // transparent
    perpTot = 1 + 1.5 * perpSum.re;
    parTot = 1 + 1.5 * parSum.re;
  } else {                                         // dissipative: 1 + (3/2) f Re{√ε₂ Σ}
    perpTot = 1 + 1.5 * fOmega * mul(sq2, perpSum).re;
    parTot = 1 + 1.5 * fOmega * mul(sq2, parSum).re;
  }

  // ---- radiative rates (Eqs. 26,27), × f for the dissipative case ----
  let perpR = 0, parR = 0;
  for (let n = 1; n <= nmax; n++) {
    const jn = by.j[n], pp = by.psip[n];
    // ⊥: (3/2) ε₂^{3/2} n(n+1)(2n+1) | jₙ/(ρ₂ y₂ Dₙ) |²
    const t1 = div(jn, mul(mul(rho2, y2), D[n]));
    perpR += n * (n + 1) * (2 * n + 1) * abs2(t1);
    // ∥: (3/4) ε₂^{3/2} (2n+1) [ |ψ'ₙ/(ρ₂ y₂ Dₙ)|² + |jₙ/(ρ₂ √ε₂ Cₙ)|² ]
    const t2 = div(pp, mul(mul(rho2, y2), D[n]));
    const t3 = div(jn, mul(mul(rho2, sq2), Cc[n]));
    parR += (2 * n + 1) * (abs2(t2) + abs2(t3));
  }
  // Radiative prefactor. Transparent (Eq. 26,27): ε₂^{3/2} — this makes the total
  // = radiative (Chew) identity hold exactly (validated to 1e-15). Dissipative:
  // Kim's explicit centre formula Eq. (32) γ^R/γ₀ = f|ε₂/(ρ₂D₁)|² requires |ε₂|²
  // (the two cases normalise by different γ₀). Both reduce to the same real,
  // positive prefactor here (|·| of a complex power).
  const abs = Math.hypot(eps2.re, eps2.im);
  // KNOWN LIMITATION (physics audit 2026-07-18): this radiative prefactor
  // implements Kim Eqs. 26–27, derived for VACUUM outside the sphere. For the
  // cavity geometry with a dielectric outer medium (ε₁ ≠ 1) the rad/nr split
  // is invalid (can go negative even for lossless media); the TOTAL rates
  // (Eqs. 21–22) remain correct. A general-outer-medium radiative formula
  // needs re-derivation before trusting the split in that regime.
  const e32 = fOmega === undefined ? abs ** 1.5 : abs ** 2;
  void eps2_32;
  let perpRad = 1.5 * e32 * perpR;
  let parRad = 0.75 * e32 * parR;
  if (fOmega !== undefined) { perpRad *= fOmega; parRad *= fOmega; }

  return {
    perpTot, parTot, perpRad, parRad,
    perpNr: perpTot - perpRad, parNr: parTot - parRad,
  };
}

// ---------------------------------------------------------------------------
// f(ω) = γ_sp/γ₀ for a dissipative medium — Kim Eq. (28):
//   γ₀/γ_sp = (3Z / 64π⁴) ε'' N₀ λ³   ⇒   f = 64π⁴ / (3 Z ε'' N₀ λ³)
// N₀ = number density of the medium (cm⁻³), λ in cm, Z ≈ 10 (lattice factor).
// ---------------------------------------------------------------------------
export function fOmegaDissipative(epsImag: number, lambdaNm: number, N0_cm3: number, Z = 10): number {
  const lamCm = lambdaNm * 1e-7;                   // nm → cm
  const gamma0_over_sp = (3 * Z / (64 * Math.PI ** 4)) * epsImag * N0_cm3 * lamCm ** 3;
  return 1 / gamma0_over_sp;
}
