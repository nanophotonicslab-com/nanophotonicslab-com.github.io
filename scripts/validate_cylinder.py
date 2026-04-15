"""
Reference values for the cylinder library — computed with SciPy and
printed as TypeScript-shaped test vectors. Run this script and compare
numerically against the TS implementation via `scripts/run_cylinder_tests.mjs`
(or by eye).
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, yv, hankel1, iv, kv

HC_EV_NM = 197.3269804


def besselj_d(m: int, z: complex) -> complex:
    return (m / z) * jv(m, z) - jv(m + 1, z)


def besselh_d(m: int, z: complex) -> complex:
    return (m / z) * hankel1(m, z) - hankel1(m + 1, z)


def give_RT(qa: float, ka: float, eps1: complex, eps_h: complex, m: int, opt: str):
    """Return (r, t) structs matching Álvaro's convention for one m, one opt."""
    M = give_M(qa, ka, eps1, eps_h, m)
    chi = np.sqrt(eps1 / eps_h)
    k1a = ka * np.sqrt(eps1)
    kha = ka * np.sqrt(eps_h)

    def _pick(z):
        s = np.sqrt(z)
        return -s if s.real < 0 else s

    Q1a = _pick(k1a * k1a - qa * qa)
    Qha = _pick(kha * kha - qa * qa)

    if opt == "inside":
        Hma = hankel1(m, Q1a)
        Hmp = besselh_d(m, Q1a)
        v_s = np.array([
            -chi * Q1a / k1a * Hma,
            -Hmp,
            0,
            -chi * m * qa / (k1a * Q1a) * Hma,
        ], dtype=complex)
        v_p = np.array([
            0,
            -m * qa / (k1a * Q1a) * Hma,
            -Q1a / k1a * Hma,
            -chi * Hmp,
        ], dtype=complex)
    else:
        Jmh = jv(m, Qha)
        Jmhp = besselj_d(m, Qha)
        v_s = np.array([
            Qha / kha * Jmh,
            Jmhp,
            0,
            m * qa / (kha * Qha) * Jmh,
        ], dtype=complex)
        v_p = np.array([
            0,
            m * qa / (kha * Qha) * Jmh,
            Qha / kha * Jmh,
            Jmhp,
        ], dtype=complex)

    RT_s = np.linalg.solve(M, v_s)
    RT_p = np.linalg.solve(M, v_p)
    return {
        "r_ss": RT_s[0], "t_ss": RT_s[1],
        "r_ps": RT_s[2], "t_ps": RT_s[3],
        "r_sp": RT_p[0], "t_sp": RT_p[1],
        "r_pp": RT_p[2], "t_pp": RT_p[3],
    }


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


def eels_parallel(a_nm, b_nm, eps_h, eps1, w_eV_list, v_frac, max_order):
    AU_NM = 0.05291772083
    AU_EV = 27.2113834
    C_AU = 137.03599971
    NM_AU = 1 / AU_NM
    a = a_nm * NM_AU
    b = b_nm * NM_AU
    v = v_frac * C_AU
    outside = b_nm > a_nm
    out = []
    for we in w_eV_list:
        W = we / AU_EV
        qz = W / v
        k = W / C_AU
        eps_med = eps_h if outside else eps1
        g = 1 / np.sqrt(1 - v_frac**2 * eps_med)
        arg = W * b / (v * g)
        total = 0 + 0j
        for m in range(-max_order, max_order + 1):
            am = abs(m)
            sign_m = 1 if m % 2 == 0 else -1
            if outside:
                coefs = give_RT(qz * a, k * a, eps1, eps_h, m, "outside")
                Km = kv(am, arg)
                total += sign_m * coefs["t_pp"].real * Km * Km
            else:
                coefs = give_RT(qz * a, k * a, eps1, eps_h, m, "inside")
                Im = iv(am, arg)
                total += -sign_m * Im * Im * (coefs["r_pp"] / eps1).real
        if outside:
            eels_au = 4 / (np.pi**2 * v**2 * g**2) * total / eps_h
        else:
            eels_au = 1 / (v**2 * g**2) * total
        out.append((eels_au * NM_AU / AU_EV).real)
    return out


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

    print("\n== I_m / K_m reference values (real arg) ==")
    for m in [0, 1, 2, 3]:
        for x in [0.5, 1.0, 2.5, 5.0]:
            print(f"I_{m}({x}) = {iv(m, x):.12e}   K_{m}({x}) = {kv(m, x):.12e}")

    print("\n== RT coefs: a=15 nm, ε₁=10+0.1i, ε_h=1, ω=10 eV, qa=1.5, ka=0.76, m=1 ==")
    ka = 10.0 / HC_EV_NM * 15.0
    rt = give_RT(qa=1.5, ka=ka, eps1=10 + 0.1j, eps_h=1 + 0j, m=1, opt="outside")
    print(f"  outside: t_pp = {fmt(rt['t_pp'])}")
    print(f"           t_ss = {fmt(rt['t_ss'])}")
    rt2 = give_RT(qa=1.5, ka=ka, eps1=10 + 0.1j, eps_h=1 + 0j, m=1, opt="inside")
    print(f"  inside:  r_pp = {fmt(rt2['r_pp'])}")
    print(f"           r_ss = {fmt(rt2['r_ss'])}")

    print("\n== EELS parallel, outside: a=15 nm, b=25 nm, ε₁=10+0.1i, ε_h=1, v=0.3c, Max=3 ==")
    w_list = [2.0, 5.0, 10.0, 15.0]
    eels = eels_parallel(15, 25, 1 + 0j, 10 + 0.1j, w_list, 0.3, 3)
    for w, v in zip(w_list, eels):
        print(f"  ω = {w:6.2f} eV → EELS = {v:.8e}")

    print("\n== EELS parallel, inside: a=15 nm, b=5 nm, same ε, v=0.3c, Max=3 ==")
    eels = eels_parallel(15, 5, 1 + 0j, 10 + 0.1j, w_list, 0.3, 3)
    for w, v in zip(w_list, eels):
        print(f"  ω = {w:6.2f} eV → EELS = {v:.8e}")


if __name__ == "__main__":
    main()
