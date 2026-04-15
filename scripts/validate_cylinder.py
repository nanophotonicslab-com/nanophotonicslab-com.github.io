"""
Reference values for the cylinder library — computed with SciPy and
printed as TypeScript-shaped test vectors. Run this script and compare
numerically against the TS implementation via `scripts/run_cylinder_tests.mjs`
(or by eye).
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, yv, hankel1

HC_EV_NM = 197.3269804


def besselj_d(m: int, z: complex) -> complex:
    return (m / z) * jv(m, z) - jv(m + 1, z)


def besselh_d(m: int, z: complex) -> complex:
    return (m / z) * hankel1(m, z) - hankel1(m + 1, z)


def give_M(qa: float, ka: float, eps1: complex, eps_h: complex, m: int):
    chi = np.sqrt(eps1 / eps_h)
    k1a = ka * np.sqrt(eps1)
    kha = ka * np.sqrt(eps_h)

    def _pick(z):
        s = np.sqrt(z)
        return -s if s.real < 0 else s

    Q1a = _pick(k1a * k1a - qa * qa)
    Qha = _pick(kha * kha - qa * qa)

    Jma = jv(m, Q1a)
    Hmh = hankel1(m, Qha)
    Jmap = besselj_d(m, Q1a)
    Hmhp = besselh_d(m, Qha)

    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = chi * Q1a / k1a * Jma
    M[0, 1] = -Qha / kha * Hmh
    M[1, 0] = Jmap
    M[1, 1] = -Hmhp
    M[1, 2] = m * qa / (k1a * Q1a) * Jma
    M[1, 3] = -m * qa / (kha * Qha) * Hmh
    M[2, 2] = Q1a / k1a * Jma
    M[2, 3] = M[0, 1]
    M[3, 0] = chi * m * qa / (k1a * Q1a) * Jma
    M[3, 1] = -m * qa / (kha * Qha) * Hmh
    M[3, 2] = chi * Jmap
    M[3, 3] = M[1, 1]
    return M


def detM(qa, ka, eps1, eps_h, m):
    return np.linalg.det(give_M(qa, ka, eps1, eps_h, m))


def fmt(c: complex) -> str:
    return f"({c.real: .10e}) + ({c.imag: .10e})i"


def main() -> None:
    print("== Bessel scalar tests ==")
    for m in [0, 1, 2, 3]:
        for z in [1 + 0j, 0.5 + 0.3j, 3.0 + 0j, 5.0 + 2.0j]:
            print(f"J_{m}({z})    = {fmt(jv(m, z))}")
            print(f"Y_{m}({z})    = {fmt(yv(m, z))}")
            print(f"H_{m}^(1)({z}) = {fmt(hankel1(m, z))}")
    print()

    # Álvaro's canonical example: dielectric cylinder with small regularization
    print("== det(M) tests — Álvaro's example: a=15 nm, ε_h=1, ε₁=10+0.1i ==")
    a_nm = 15.0
    eps_h = 1.0 + 0j
    eps1 = 10.0 + 0.1j
    for w_eV in [5.0, 10.0]:
        k0 = w_eV / HC_EV_NM
        ka = k0 * a_nm
        q_light = np.sqrt(eps_h.real) * k0
        q_mat = np.sqrt(eps1.real) * k0
        print(f"\nω = {w_eV} eV  k₀ = {k0:.6f} nm⁻¹  ka = {ka:.6f}")
        print(f"q_light = {q_light:.6f} nm⁻¹   q_mat = {q_mat:.6f} nm⁻¹")
        for m in [0, 1, 2]:
            print(f"  m = {m}  (Re det(M) along q)")
            Nq = 25
            for i in range(Nq):
                q = q_light * 1.001 + (q_mat * 0.999 - q_light * 1.001) * i / (Nq - 1)
                qa = q * a_nm
                d = detM(qa, ka, eps1, eps_h, m)
                print(f"    q={q:.6f}  qa={qa:.4f}  Re(D)={d.real:+.4e}  Im(D)={d.imag:+.4e}")


if __name__ == "__main__":
    main()
