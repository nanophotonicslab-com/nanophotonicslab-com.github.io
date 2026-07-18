/**
 * Canonical complex-number type and arithmetic for the lab libraries.
 *
 * One struct shape `{ re, im }` shared by every struct-style module
 * (materials, plasmonic-nanoparticles, vo2, smith-purcell, dipole-decay,
 * dipole-inside). Two other calling conventions remain by design — both are
 * performance-motivated hot-loop styles local to their solvers:
 *   - cylinder.ts / electron-sphere.ts: tuple `[re, im]` (`C` + `Cx` namespace)
 *   - mie.ts: scalar-pair arguments `(re, im, …)` returning tuples
 *
 * The two square roots are deliberately distinct branch choices — do not merge:
 *   - csqrt:      principal branch, Im(√z) carries the sign of Im(z)
 *   - csqrtUpper: upper-half-plane root (Im ≥ 0), the physical
 *                 refractive-index branch used by the dipole/Kim models
 * Likewise cdiv (direct formula) vs cdivRobust (Smith's algorithm) trade
 * speed against over/underflow safety in Miller-recurrence Bessel code.
 */
export interface Complex { re: number; im: number; }

export const complex = (re: number, im = 0): Complex => ({ re, im });

export const cadd = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });
export const csub = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });
export const cmul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
export const cscale = (a: Complex, s: number): Complex => ({ re: a.re * s, im: a.im * s });
export const cneg = (a: Complex): Complex => ({ re: -a.re, im: -a.im });
export const cconj = (a: Complex): Complex => ({ re: a.re, im: -a.im });
export const cabs2 = (a: Complex): number => a.re * a.re + a.im * a.im;
export const cabs = (a: Complex): number => Math.hypot(a.re, a.im);

export const cinv = (a: Complex): Complex => {
  const d = a.re * a.re + a.im * a.im;
  return { re: a.re / d, im: -a.im / d };
};

/** a / b — direct formula; fine when |b| is far from over/underflow. */
export const cdiv = (a: Complex, b: Complex): Complex => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};

/**
 * a / b via Smith's algorithm — robust to over/underflow (avoids squaring |b|,
 * which underflows to 0 for the tiny high-order Bessel values in Miller's
 * recurrence).
 */
export const cdivRobust = (a: Complex, b: Complex): Complex => {
  if (Math.abs(b.re) >= Math.abs(b.im)) {
    const r = b.im / b.re, d = b.re + b.im * r;
    return { re: (a.re + a.im * r) / d, im: (a.im - a.re * r) / d };
  }
  const r = b.re / b.im, d = b.re * r + b.im;
  return { re: (a.re * r + a.im) / d, im: (a.im * r - a.re) / d };
};

export const csin = (z: Complex): Complex => ({ re: Math.sin(z.re) * Math.cosh(z.im), im: Math.cos(z.re) * Math.sinh(z.im) });
export const ccos = (z: Complex): Complex => ({ re: Math.cos(z.re) * Math.cosh(z.im), im: -Math.sin(z.re) * Math.sinh(z.im) });

/** Principal square root: Im(√z) carries the sign of Im(z). */
export function csqrt(z: Complex): Complex {
  const r = Math.hypot(z.re, z.im), re = Math.sqrt(Math.max(0, (r + z.re) / 2));
  let im = Math.sqrt(Math.max(0, (r - z.re) / 2));
  if (z.im < 0) im = -im;
  return { re, im };
}

/** Upper-half-plane square root (Im ≥ 0) — the physical refractive-index branch. */
export function csqrtUpper(z: Complex): Complex {
  const r = Math.hypot(z.re, z.im);
  let re = Math.sqrt(Math.max(0, (r + z.re) / 2));
  let im = Math.sqrt(Math.max(0, (r - z.re) / 2));
  if (z.im < 0) im = -im;
  if (im < 0) { re = -re; im = -im; }
  return { re, im };
}
