import { besselK, Cx, type C } from './cylinder';

const E_CHARGE = 1.602176634e-19;
const EPS0 = 8.8541878128e-12;
const MU0 = 1.25663706212e-6;
const C0 = 299792458.0;
const HBAR_JS = 1.054571817e-34;
const HBAR_EVS = 6.582119569e-16;
const Z0 = Math.sqrt(MU0 / EPS0);
const HC_EV_NM = 1239.841984;

const LOG_SQRT_2PI = 0.9189385332046727;
const LANCZOS = [
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export interface ElectronSphereParams {
  radiusNm: number;
  beta: number;
  impactNm: number;
  lmax: number;
  qCutNm: number;
  nz: number;
  phi0?: number;
}

export interface ElectronSpectrum {
  energy: Float64Array;
  eelsTotal: Float64Array;
  eelsSurface: Float64Array;
  eelsBulk: Float64Array;
  eelsBegrenzung: Float64Array;
  eelsElectric: Float64Array;
  eelsMagnetic: Float64Array;
  clTotal: Float64Array;
  clElectric: Float64Array;
  clMagnetic: Float64Array;
  eelsMultipoles: Float64Array[];
  clMultipoles: Float64Array[];
}

interface SpectrumPoint {
  eelsTotal: number;
  eelsSurface: number;
  eelsBulk: number;
  eelsBegrenzung: number;
  eelsElectric: number;
  eelsMagnetic: number;
  clTotal: number;
  clElectric: number;
  clMagnetic: number;
  eelsByL: number[];
  clByL: number[];
}

interface TCoeffs {
  TE11: C; TE12: C; TE21: C; TE22: C;
  TM11: C; TM12: C; TM21: C; TM22: C;
}

interface PathInts {
  yPlus: C;
  yMinus: C;
  fPlus: C;
  fMinus: C;
}

interface Quadrature {
  ze: number;
  z: Float64Array;
  w: Float64Array;
}

function c(re: number, im = 0): C { return [re, im]; }
function cadd(a: C, b: C): C { return Cx.add(a, b); }
function csub(a: C, b: C): C { return Cx.sub(a, b); }
function cmul(a: C, b: C): C { return Cx.mul(a, b); }
function cdiv(a: C, b: C): C { return Cx.div(a, b); }
function cscale(a: C, s: number): C { return Cx.scale(a, s); }
function cneg(a: C): C { return [-a[0], -a[1]]; }
function cconj(a: C): C { return [a[0], -a[1]]; }
function creal(a: C): number { return a[0]; }
function cabs2(a: C): number { return Cx.abs2(a); }
function csqrt(a: C): C { return Cx.sqrt(a); }
function clog(a: C): C { return Cx.log(a); }
function cexp(a: C): C { return Cx.exp(a); }

function cInv(a: C): C {
  const d = a[0] * a[0] + a[1] * a[1];
  return [a[0] / d, -a[1] / d];
}

function cpowI(n: number): C {
  const m = ((n % 4) + 4) % 4;
  if (m === 0) return [1, 0];
  if (m === 1) return [0, 1];
  if (m === 2) return [-1, 0];
  return [0, -1];
}

function csin(z: C): C {
  return [Math.sin(z[0]) * Math.cosh(z[1]), Math.cos(z[0]) * Math.sinh(z[1])];
}

function ccos(z: C): C {
  return [Math.cos(z[0]) * Math.cosh(z[1]), -Math.sin(z[0]) * Math.sinh(z[1])];
}

function logGamma(z: number): number {
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  let x = 0.99999999999980993;
  const t0 = z - 1;
  for (let i = 0; i < LANCZOS.length; i++) x += LANCZOS[i] / (t0 + i + 1);
  const t = t0 + LANCZOS.length - 0.5;
  return LOG_SQRT_2PI + (t0 + 0.5) * Math.log(t) - t + Math.log(x);
}

function doubleFactorialOddLog(m: number): number {
  if (m === 0) return 0;
  return logGamma(2 * m + 1) - m * Math.log(2) - logGamma(m + 1);
}

function gegenbauer(n: number, alpha: number, x: number): number {
  if (n < 0) return 0;
  if (n === 0) return 1;
  let c0 = 1;
  let c1 = 2 * alpha * x;
  if (n === 1) return c1;
  for (let k = 2; k <= n; k++) {
    const ck = (2 * (k + alpha - 1) * x * c1 - (k + 2 * alpha - 2) * c0) / k;
    c0 = c1;
    c1 = ck;
  }
  return c1;
}

function c_lm(ell: number, m: number): number {
  const value = (ell - m) * (ell + m + 1);
  return value <= 0 ? 0 : 0.5 * Math.sqrt(value);
}

function mLm(ell: number, m: number, beta: number, gamma0: number): C {
  if (Math.abs(m) > ell) return [0, 0];
  if (m < 0) {
    const mp = -m;
    const v = mLm(ell, mp, beta, gamma0);
    return mp % 2 === 0 ? v : cneg(v);
  }
  let logPref = 0.5 * (
    Math.log((2 * ell + 1) / Math.PI)
    + logGamma(ell - m + 1)
    - logGamma(ell + m + 1)
  );
  logPref += doubleFactorialOddLog(m);
  if (m) logPref -= m * Math.log(beta * gamma0);
  const g = gegenbauer(ell - m, m + 0.5, 1 / beta);
  return cscale(cpowI(ell + m), Math.exp(logPref) * g);
}

function nLm(ell: number, m: number, beta: number, gamma0: number): C {
  return csub(
    cscale(mLm(ell, m + 1, beta, gamma0), c_lm(ell, m)),
    cscale(mLm(ell, m - 1, beta, gamma0), c_lm(ell, -m)),
  );
}

function associatedLegendrePositive(ell: number, m: number, xRaw: number): number {
  const x = Math.max(-1, Math.min(1, xRaw));
  if (m < 0 || m > ell) return 0;
  let pmm = 1;
  if (m > 0) {
    const somx2 = Math.sqrt(Math.max(0, (1 - x) * (1 + x)));
    let fact = 1;
    for (let i = 1; i <= m; i++) {
      pmm *= -fact * somx2;
      fact += 2;
    }
  }
  if (ell === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (ell === m + 1) return pmmp1;
  let pll = 0;
  for (let l = m + 2; l <= ell; l++) {
    pll = ((2 * l - 1) * x * pmmp1 - (l + m - 1) * pmm) / (l - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

function sphericalY(ell: number, m: number, theta: number): C {
  if (Math.abs(m) > ell) return [0, 0];
  if (m < 0) {
    const y = sphericalY(ell, -m, theta);
    return (-m) % 2 === 0 ? y : cneg(y);
  }
  const x = Math.cos(theta);
  const p = associatedLegendrePositive(ell, m, x);
  const norm = Math.sqrt((2 * ell + 1) / (4 * Math.PI) * Math.exp(logGamma(ell - m + 1) - logGamma(ell + m + 1)));
  return [norm * p, 0];
}

function sphJ(ell: number, z: C): C {
  if (ell === 0) return cdiv(csin(z), z);
  const sinz = csin(z);
  const cosz = ccos(z);
  const j0 = cdiv(sinz, z);
  const j1 = csub(cdiv(sinz, cmul(z, z)), cdiv(cosz, z));
  if (ell === 1) return j1;
  const zAbs = Math.hypot(z[0], z[1]);
  if (ell <= zAbs) {
    // upward recurrence is stable while ℓ ≲ |z|
    let jm1 = j0;
    let j = j1;
    for (let n = 1; n < ell; n++) {
      const jp1 = csub(cscale(cdiv(j, z), 2 * n + 1), jm1);
      jm1 = j;
      j = jp1;
    }
    return j;
  }
  // ℓ > |z|: upward recurrence amplifies rounding as ~(2ℓ−1)!!/|z|ℓ and blew
  // up to ±1e24 at low energies (physics audit 2026-07-18). Downward Miller
  // recurrence, normalized to j₀ = sin z / z.
  const start = ell + 16 + Math.ceil(zAbs);
  let jp: C = [0, 0];
  let jc: C = [1e-30, 0];
  let out: C | null = null;
  for (let n = start; n >= 1; n--) {
    const jm = csub(cscale(cdiv(jc, z), 2 * n + 1), jp);
    jp = jc;
    jc = jm;
    if (n - 1 === ell) out = jm;
    const mag = Math.abs(jc[0]) + Math.abs(jc[1]);
    if (mag > 1e120) {
      jp = cscale(jp, 1e-120);
      jc = cscale(jc, 1e-120);
      if (out) out = cscale(out, 1e-120);
    }
  }
  return cmul(out!, cdiv(j0, jc));
}

function sphY(ell: number, z: C): C {
  const sinz = csin(z);
  const cosz = ccos(z);
  if (ell === 0) return cneg(cdiv(cosz, z));
  let ym1 = cneg(cdiv(cosz, z));
  let y = csub(cneg(cdiv(cosz, cmul(z, z))), cdiv(sinz, z));
  if (ell === 1) return y;
  for (let n = 1; n < ell; n++) {
    const yp1 = csub(cscale(cdiv(y, z), 2 * n + 1), ym1);
    ym1 = y;
    y = yp1;
  }
  return y;
}

function sphH1(ell: number, z: C): C {
  const j = sphJ(ell, z);
  const y = sphY(ell, z);
  return [j[0] - y[1], j[1] + y[0]];
}

function sphJPrime(ell: number, z: C): C {
  if (ell === 0) return cneg(sphJ(1, z));
  return csub(sphJ(ell - 1, z), cscale(cdiv(sphJ(ell, z), z), ell + 1));
}

function sphH1Prime(ell: number, z: C): C {
  if (ell === 0) return cneg(sphH1(1, z));
  return csub(sphH1(ell - 1, z), cscale(cdiv(sphH1(ell, z), z), ell + 1));
}

function riccatiJPrime(ell: number, x: C): C {
  return cadd(sphJ(ell, x), cmul(x, sphJPrime(ell, x)));
}

function riccatiHPrime(ell: number, x: C): C {
  return cadd(sphH1(ell, x), cmul(x, sphH1Prime(ell, x)));
}

function mieTCoefficients(ell: number, eps: C, k0: number, radiusM: number): TCoeffs {
  const x0 = c(k0 * radiusM);
  const sqrtEps = csqrt(eps);
  const x = cscale(sqrtEps, k0 * radiusM);

  const j0 = sphJ(ell, x0);
  const j = sphJ(ell, x);
  const h0 = sphH1(ell, x0);
  const h = sphH1(ell, x);

  const psi0p = riccatiJPrime(ell, x0);
  const psip = riccatiJPrime(ell, x);
  const xi0p = riccatiHPrime(ell, x0);
  const xip = riccatiHPrime(ell, x);

  const denomE = csub(cmul(h0, psip), cmul(cmul(eps, xi0p), j));
  const denomM = csub(cmul(h0, psip), cmul(xi0p, j));
  const negIOverX0 = c(0, -1 / (k0 * radiusM));

  return {
    TE22: cdiv(csub(cmul(cmul(eps, j), psi0p), cmul(psip, j0)), denomE),
    TM22: cdiv(csub(cmul(j, psi0p), cmul(psip, j0)), denomM),
    TE21: cdiv(cscale(cmul(c(0, -1 / (k0 * radiusM)), sqrtEps), 1), denomE),
    TM21: cdiv(negIOverX0, denomM),
    TE11: cdiv(csub(cmul(cmul(eps, xi0p), h), cmul(h0, xip)), denomE),
    TM11: cdiv(csub(cmul(xi0p, h), cmul(h0, xip)), denomM),
    TE12: cdiv(negIOverX0, denomE),
    TM12: cdiv(cdiv(negIOverX0, sqrtEps), denomM),
  };
}

function gaussLegendre(n: number): { x: Float64Array; w: Float64Array } {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  const m = Math.floor((n + 1) / 2);
  const eps = 1e-14;
  for (let i = 0; i < m; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
    let z1 = 0;
    let pp = 0;
    while (Math.abs(z - z1) > eps) {
      let p1 = 1;
      let p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1);
      z1 = z;
      z = z1 - p1 / pp;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    const wi = 2 / ((1 - z * z) * pp * pp);
    w[i] = wi;
    w[n - 1 - i] = wi;
  }
  return { x, w };
}

function pathQuadrature(bM: number, radiusM: number, nz: number): Quadrature {
  if (bM >= radiusM) return { ze: 0, z: new Float64Array(0), w: new Float64Array(0) };
  const ze = Math.sqrt(radiusM * radiusM - bM * bM);
  const base = gaussLegendre(Math.max(8, Math.round(nz)));
  const z = new Float64Array(base.x.length);
  const w = new Float64Array(base.w.length);
  for (let i = 0; i < base.x.length; i++) {
    z[i] = ze * base.x[i];
    w[i] = ze * base.w[i];
  }
  return { ze, z, w };
}

function fieldFSum(ell: number, m: number, k: C, bM: number, z: number, r: number, theta: number, kind: 'j' | 'h'): C {
  const kr = cscale(k, r);
  const f = kind === 'j' ? sphJ(ell, kr) : sphH1(ell, kr);
  const fp = kind === 'j' ? sphJPrime(ell, kr) : sphH1Prime(ell, kr);
  let total: C = [0, 0];

  for (const sign of [1, -1] as const) {
    const cOuter = c_lm(ell, sign * m);
    if (cOuter === 0) continue;
    const y1 = sphericalY(ell, m + sign, theta);
    const y2 = sphericalY(ell, m + 2 * sign, theta);
    const y0 = sphericalY(ell, m, theta);
    const term1 = cscale(cmul(k, fp), bM * bM / r);
    const innerY = csub(cscale(y2, c_lm(ell, sign * m + 1)), cscale(y0, cOuter));
    const term2 = cscale(cmul(f, innerY), sign * z * bM / (r * r));
    const term3 = cscale(cmul(f, y1), 1 + sign * m);
    const bracket = cadd(cadd(cmul(term1, y1), term2), term3);
    total = cadd(total, cscale(bracket, -sign * cOuter));
  }

  return total;
}

function pathIntegrals(
  ell: number,
  m: number,
  k: C,
  omega: number,
  params: Required<ElectronSphereParams>,
  bM: number,
  quad: Quadrature,
  kind: 'j' | 'h',
): PathInts {
  if (quad.z.length === 0) return { yPlus: [0, 0], yMinus: [0, 0], fPlus: [0, 0], fMinus: [0, 0] };

  let yPlus: C = [0, 0];
  let yMinus: C = [0, 0];
  let fPlus: C = [0, 0];
  let fMinus: C = [0, 0];

  for (let i = 0; i < quad.z.length; i++) {
    const z = quad.z[i];
    const weight = quad.w[i];
    const r = Math.sqrt(bM * bM + z * z);
    const theta = Math.acos(Math.max(-1, Math.min(1, z / r)));
    const phase = omega * z / (params.beta * C0);
    const phasePlus = c(Math.cos(phase), Math.sin(phase));
    const phaseMinus = c(Math.cos(phase), -Math.sin(phase));
    const kr = cscale(k, r);
    const f = kind === 'j' ? sphJ(ell, kr) : sphH1(ell, kr);
    const y = sphericalY(ell, m, theta);
    const fsum = fieldFSum(ell, m, k, bM, z, r, theta, kind);

    yPlus = cadd(yPlus, cscale(cmul(cmul(phasePlus, f), y), weight));
    yMinus = cadd(yMinus, cscale(cmul(cmul(phaseMinus, f), y), weight));
    fPlus = cadd(fPlus, cscale(cmul(phasePlus, fsum), weight));
    fMinus = cadd(fMinus, cscale(cmul(phaseMinus, fsum), weight));
  }

  return { yPlus, yMinus, fPlus, fMinus };
}

function gammaMedium(eps: C, beta: number): C {
  const arg = csub([1, 0], cscale(eps, beta * beta));
  // Exactly at the Cherenkov threshold (εβ² = 1) 1/√0 turned the whole
  // spectrum NaN — regularize the branch point (physics audit 2026-07-18).
  if (Math.hypot(arg[0], arg[1]) < 1e-12) return cInv(csqrt([1e-12, 1e-12]));
  return cInv(csqrt(arg));
}

function point(
  energyEv: number,
  eps: C,
  paramsInput: ElectronSphereParams,
): SpectrumPoint {
  const params: Required<ElectronSphereParams> = {
    phi0: 0,
    ...paramsInput,
    lmax: Math.max(1, Math.min(18, Math.round(paramsInput.lmax))),
    nz: Math.max(8, Math.min(256, Math.round(paramsInput.nz))),
    beta: Math.max(0.02, Math.min(0.95, paramsInput.beta)),
    impactNm: Math.max(0.1, paramsInput.impactNm),
    qCutNm: Math.max(1e-4, paramsInput.qCutNm),
  };
  const omega = energyEv * E_CHARGE / HBAR_JS;
  const k0 = omega / C0;
  const sqrtEps = csqrt(eps);
  const k = cscale(sqrtEps, k0);
  const gamma = gammaMedium(eps, params.beta);
  const gamma0 = 1 / Math.sqrt(1 - params.beta * params.beta);
  const radiusM = params.radiusNm * 1e-9;
  const bM = params.impactNm * 1e-9;
  const quad = pathQuadrature(bM, radiusM, params.nz);

  const prefEel = E_CHARGE / (Math.PI * HBAR_JS * omega);
  const prefCommon = c(0, k0 * k0 * E_CHARGE / (EPS0 * omega));
  const kArg = omega * bM / (params.beta * C0 * gamma0);

  let surfSum: C = [0, 0];
  let surfESum: C = [0, 0];
  let surfMSum: C = [0, 0];
  let begrSum: C = [0, 0];
  let begrESum: C = [0, 0];
  let begrMSum: C = [0, 0];
  let clSum = 0;
  let clElectric = 0;
  let clMagnetic = 0;
  const eelsByLComplex: C[] = Array.from({ length: params.lmax }, () => [0, 0] as C);
  const clByL = new Array(params.lmax).fill(0);

  for (let ell = 1; ell <= params.lmax; ell++) {
    const T = mieTCoefficients(ell, eps, k0, radiusM);
    const sqrtL = Math.sqrt(ell * (ell + 1));

    for (let m = -ell; m <= ell; m++) {
      const M = mLm(ell, m, params.beta, gamma0);
      const N = nLm(ell, m, params.beta, gamma0);
      const Km = Math.max(0, besselK(Math.abs(m), [Math.max(kArg, 1e-9), 0])[0]);
      const ih = pathIntegrals(ell, m, c(k0), omega, params, bM, quad, 'h');
      const ij = pathIntegrals(ell, m, k, omega, params, bM, quad, 'j');
      const iair = pathIntegrals(ell, m, c(k0), omega, params, bM, quad, 'j');
      const phaseM = cexp(c(0, -m * params.phi0));

      const b0II = cscale(
        cmul(prefCommon, cmul(phaseM, csub(cscale(M, Km), cmul(c(0, k0), ih.yPlus)))),
        -m / sqrtL,
      );
      const a0II = cscale(
        cmul(prefCommon, cmul(phaseM, csub(cscale(N, Km / (params.beta * gamma0)), cmul(c(0, 1 / bM), ih.fPlus)))),
        1 / sqrtL,
      );
      const b0I = cscale(cmul(prefCommon, cmul(phaseM, cmul(c(0, 1), cmul(k, ij.yPlus)))), -m / sqrtL);
      const a0I = cscale(cmul(prefCommon, cmul(phaseM, cmul(c(0, 1), cscale(ij.fPlus, 1 / bM)))), 1 / sqrtL);
      const b0IAir = cscale(cmul(prefCommon, cmul(phaseM, cmul(c(0, k0), iair.yPlus))), -m / sqrtL);
      const a0IAir = cscale(cmul(prefCommon, cmul(phaseM, cmul(c(0, 1), cscale(iair.fPlus, 1 / bM)))), 1 / sqrtL);

      const aI = cadd(cmul(T.TE11, a0I), cmul(T.TE21, a0II));
      const bI = cadd(cmul(T.TM11, b0I), cmul(T.TM21, b0II));
      const aII = csub(cadd(cmul(T.TE12, a0I), cmul(T.TE22, a0II)), a0IAir);
      const bII = csub(cadd(cmul(T.TM12, b0I), cmul(T.TM22, b0II)), b0IAir);

      const expCancel = cexp(c(0, m * params.phi0));
      const surfM = cscale(
        cmul(cmul(bII, expCancel), csub(cmul(c(0, -Km / k0), cconj(M)), ih.yMinus)),
        m / sqrtL,
      );
      const surfE = cscale(
        cmul(cmul(aII, expCancel), csub(cmul(c(0, -Km / (k0 * params.beta * gamma0)), cconj(N)), cscale(ih.fMinus, 1 / (k0 * bM)))),
        -1 / sqrtL,
      );
      surfMSum = cadd(surfMSum, surfM);
      surfESum = cadd(surfESum, surfE);
      surfSum = cadd(surfSum, cadd(surfM, surfE));

      const begrM = cscale(cmul(expCancel, csub(cmul(bI, ij.yMinus), cmul(b0II, iair.yMinus))), m / sqrtL);
      const begrE = cscale(
        cmul(expCancel, csub(cdiv(cmul(aI, ij.fMinus), cscale(k, bM)), cscale(cmul(a0II, iair.fMinus), 1 / (k0 * bM)))),
        -1 / sqrtL,
      );
      begrMSum = cadd(begrMSum, begrM);
      begrESum = cadd(begrESum, begrE);
      begrSum = cadd(begrSum, cadd(begrM, begrE));
      eelsByLComplex[ell - 1] = cadd(eelsByLComplex[ell - 1], cadd(cadd(surfM, surfE), cadd(begrM, begrE)));

      const clE = cabs2(aII) / (Math.PI * HBAR_JS * omega * Z0 * k0 * k0);
      const clM = cabs2(bII) / (Math.PI * HBAR_JS * omega * Z0 * k0 * k0);
      clElectric += clE;
      clMagnetic += clM;
      clSum += clE + clM;
      clByL[ell - 1] += clE + clM;
    }
  }

  let bulk = 0;
  if (quad.ze > 0) {
    const term = cmul(cInv(cmul(cmul(gamma, gamma), eps)), clog(cadd(cmul(cscale(gamma, params.qCutNm * 1e9 * params.beta * C0 / omega), cscale(gamma, params.qCutNm * 1e9 * params.beta * C0 / omega)), [1, 0])));
    bulk = (
      -E_CHARGE * E_CHARGE * quad.ze
      / (2 * Math.PI * Math.PI * EPS0 * HBAR_JS * Math.pow(params.beta * C0, 2))
      * term[1]
    );
  }

  const surface = prefEel * creal(surfSum);
  const surfaceE = prefEel * creal(surfESum);
  const surfaceM = prefEel * creal(surfMSum);
  const begr = prefEel * creal(begrSum);
  const begrE = prefEel * creal(begrESum);
  const begrM = prefEel * creal(begrMSum);
  const toPerEv = 1 / HBAR_EVS;

  return {
    eelsTotal: (surface + begr + bulk) * toPerEv,
    eelsSurface: surface * toPerEv,
    eelsBulk: bulk * toPerEv,
    eelsBegrenzung: begr * toPerEv,
    eelsElectric: (surfaceE + begrE) * toPerEv,
    eelsMagnetic: (surfaceM + begrM) * toPerEv,
    clTotal: clSum * toPerEv,
    clElectric: clElectric * toPerEv,
    clMagnetic: clMagnetic * toPerEv,
    eelsByL: eelsByLComplex.map(v => prefEel * creal(v) * toPerEv),
    clByL: clByL.map(v => v * toPerEv),
  };
}

export function computeElectronSphereSpectrum(
  getDielectric: (lambdaNm: number) => [number, number],
  params: ElectronSphereParams,
  energyMinEv = 1.5,
  energyMaxEv = 5.5,
  points = 140,
): ElectronSpectrum {
  const n = Math.max(24, Math.min(320, Math.round(points)));
  const energy = new Float64Array(n);
  const eelsTotal = new Float64Array(n);
  const eelsSurface = new Float64Array(n);
  const eelsBulk = new Float64Array(n);
  const eelsBegrenzung = new Float64Array(n);
  const eelsElectric = new Float64Array(n);
  const eelsMagnetic = new Float64Array(n);
  const clTotal = new Float64Array(n);
  const clElectric = new Float64Array(n);
  const clMagnetic = new Float64Array(n);
  const lmax = Math.max(1, Math.min(18, Math.round(params.lmax)));
  const eelsMultipoles = Array.from({ length: lmax }, () => new Float64Array(n));
  const clMultipoles = Array.from({ length: lmax }, () => new Float64Array(n));
  const eMin = Math.max(0.05, Math.min(energyMinEv, energyMaxEv - 0.01));
  const eMax = Math.max(eMin + 0.01, energyMaxEv);

  for (let i = 0; i < n; i++) {
    const e = eMin + (eMax - eMin) * i / (n - 1);
    energy[i] = e;
    const lambdaNm = HC_EV_NM / e;
    const [nr, ki] = getDielectric(lambdaNm);
    const eps = c(nr * nr - ki * ki, 2 * nr * ki);
    const p = point(e, eps, params);
    eelsTotal[i] = p.eelsTotal;
    eelsSurface[i] = p.eelsSurface;
    eelsBulk[i] = p.eelsBulk;
    eelsBegrenzung[i] = p.eelsBegrenzung;
    eelsElectric[i] = p.eelsElectric;
    eelsMagnetic[i] = p.eelsMagnetic;
    clTotal[i] = p.clTotal;
    clElectric[i] = p.clElectric;
    clMagnetic[i] = p.clMagnetic;
    for (let ell = 0; ell < lmax; ell++) eelsMultipoles[ell][i] = p.eelsByL[ell] ?? 0;
    for (let ell = 0; ell < lmax; ell++) clMultipoles[ell][i] = p.clByL[ell] ?? 0;
  }

  return {
    energy,
    eelsTotal,
    eelsSurface,
    eelsBulk,
    eelsBegrenzung,
    eelsElectric,
    eelsMagnetic,
    clTotal,
    clElectric,
    clMagnetic,
    eelsMultipoles,
    clMultipoles,
  };
}

export function wavelengthFromEnergyEv(energyEv: number): number {
  return HC_EV_NM / energyEv;
}
