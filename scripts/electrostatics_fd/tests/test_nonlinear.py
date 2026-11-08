"""Tests for ``nonlinear.py`` (paper 406, nonlinear orders).

Two families of tests live here.

``test_ref_*`` / white-box tests
    Run against a self-contained reference harness defined in this file.  The harness
    is a TEST DOUBLE: its Coulomb kernel and conductivities are deliberately simple
    stand-ins and make no claim to be graphene physics.  What is faithful about it is
    the *structure* the assertions depend on -- the kernel is even in the node
    separation, the FD operators commute with node reversal, and the stub second-order
    conductivities are exactly odd in ``doping_sign``.  Every assertion in this family
    is therefore a statement that holds for ANY admissible kernel (a parity law, an
    exact power law in the driving potential, a derivative identity, or a white-box
    check of the block assembly), never an invented reference number.

``test_real_*``
    Run the same physics checks through the real sibling modules.  They ``skip`` while
    those modules do not exist yet.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import nonlinear as nl  # noqa: E402


# ======================================================================================
# reference harness (test double -- NOT physics)
# ======================================================================================
_KERNEL_REG = 0.01  # smooths the log singularity on the scale of ~0.1 in theta units
_ETA_SCALE = 6.0e-4  # stub coupling strength: spectral radius of N ~ 0.3, cond(I-N) ~ 1.4


def _ref_kernel(t: np.ndarray, ztil: float) -> np.ndarray:
    """Smoothed log kernel, ``-0.5*log(t^2 + ztil^2 + reg)``.

    Stands in for ``operators.coulomb_V_entries``.  The only property the tests rely
    on is that it is a function of ``t**2``, hence even in the node separation -- which
    the real quasistatic kernel also is.
    """
    return -0.5 * np.log(np.asarray(t, dtype=float) ** 2 + float(ztil) ** 2 + _KERNEL_REG) + 0j


def _tridiag(L: int, h: float) -> np.ndarray:
    """Second-difference operator ``tridiag(1,-2,1)/h**2`` with zero-ghost-flux ends.

    Boundary rows are ``[-1, 1]/h**2`` and ``[1, -1]/h**2``, matching
    ``operators._build_D_parts`` (vanishing normal current at the ribbon edge).  Like
    the real operator it is symmetric and commutes with node reversal, which is the
    property the parity tests rely on.
    """
    off = np.ones(L - 1)
    diag = np.full(L, -2.0)
    diag[0] = diag[-1] = -1.0
    return (np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)).astype(
        np.complex128
    ) / h**2


class _StubOperators:
    """Stand-in for ``operators``; signatures match the module contract exactly."""

    @staticmethod
    def build_D_geom(q: float, W: float, h: float, L: int, f: Any = None) -> np.ndarray:
        return _tridiag(L, h) - (q * W) ** 2 * np.eye(L, dtype=np.complex128)

    @staticmethod
    def build_D(
        q: float,
        W: float,
        h: float,
        L: int,
        eta_perp: complex,
        eta_par: complex,
        f: Any = None,
    ) -> np.ndarray:
        return eta_par * _tridiag(L, h) - eta_perp * (q * W) ** 2 * np.eye(
            L, dtype=np.complex128
        )

    @staticmethod
    def build_V(
        q: float,
        W_j: float,
        W_jp: float,
        x_j: float,
        x_jp: float,
        z_j: float,
        z_jp: float,
        theta_j: np.ndarray,
        theta_jp: np.ndarray,
        h_jp: float,
    ) -> np.ndarray:
        # Normalised separations exactly as the geometry block of the contract defines.
        # q is ignored: this harness is used only at q = 0.
        t = (
            x_j - x_jp + W_j * np.asarray(theta_j)[:, None] - W_jp * np.asarray(theta_jp)[None, :]
        ) / W_jp
        return h_jp * _ref_kernel(t, (z_j - z_jp) / W_jp)


class _StubConductivity:
    """Stand-in for the nonlinear coefficients of ``conductivity``.

    Faithful properties, and the only ones the tests use:
      * ``sigma_shg_1d_q0`` is exactly ODD in ``doping_sign``;
      * ``sigma_cascade_abcd``'s ``A`` is exactly ODD in ``doping_sign`` and differs
        between ``(s1, s2) = (1, 2)`` and ``(2, 1)``;
      * ``sigma_thg_local`` does not depend on ``doping_sign`` at all;
      * ``_S2`` / ``_D`` are present with the same meaning as in ``conductivity``, so
        the harness exercises ``nonlinear._cascade_A``'s default reference path rather
        than its fallback.
    The numerical values are arbitrary.
    """

    #: arbitrary; chosen so the cascade is an O(1) correction in this harness
    _S2 = 0.045 + 0.0j

    @staticmethod
    def _D(s: float, omega: float, gam_fs: float) -> complex:
        """Harmonic denominator ``D_s = 1/(s omega + 1j gamma/hbar)``."""
        return 1.0 / (s * omega + 1j * gam_fs)

    @staticmethod
    def sigma_shg_1d_q0(E_ev: float, gamma_ev: float, doping_sign: float = +1) -> complex:
        return doping_sign * 0.37 / (E_ev + 1j * gamma_ev) ** 2

    @staticmethod
    def sigma_thg_local(E_ev: float, EF_ev: float, gamma_ev: float) -> complex:
        return 0.11 / (E_ev + 1j * gamma_ev) ** 3

    @staticmethod
    def sigma_cascade_abcd(
        E_ev: float, gamma_ev: float, s1: int, s2: int, doping_sign: float = +1
    ) -> Tuple[complex, complex, complex, complex]:
        # 0.004 is chosen only so that the cascade is an O(1) correction to the local
        # third-order term in this harness, keeping the regression band below tight.
        base = doping_sign * 0.004 / (E_ev + 1j * gamma_ev) ** 3
        A = base * (s1 + 0.5j * s2)
        B = base * (s2 - 0.25j * s1)
        C = base * 0.5 * (s1 + s2)
        D = base * 0.125 * s1 * s2
        return A, B, C, D


class _CountingOperators:
    """Wraps ``_StubOperators`` and counts operator/kernel builds.

    Used to pin the hot-path cost structure deterministically, without timing.
    """

    def __init__(self) -> None:
        self.calls = {"build_V": 0, "build_D": 0, "build_D_geom": 0}

    def build_V(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self.calls["build_V"] += 1
        return _StubOperators.build_V(*args, **kwargs)

    def build_D(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self.calls["build_D"] += 1
        return _StubOperators.build_D(*args, **kwargs)

    def build_D_geom(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self.calls["build_D_geom"] += 1
        return _StubOperators.build_D_geom(*args, **kwargs)


@dataclass
class _RefRibbon:
    """Mirrors the fields of ``solver.Ribbon`` that ``nonlinear.py`` reads."""

    W_nm: float
    x0_nm: float = 0.0
    z0_nm: float = 0.0
    EF_ev: float = 0.5
    gamma_ev: float = 0.02
    beta: float = 0.0
    L: Optional[int] = 81


class _RefSystem:
    """Duck-typed stand-in for ``solver.RibbonSystem``."""

    def __init__(self, ribbons: Sequence[_RefRibbon], eta_scale: float = _ETA_SCALE):
        self.ribbons: List[_RefRibbon] = list(ribbons)
        self.dx_nm = 0.25
        self.layers = None
        self.operators = _StubOperators
        self.conductivity = _StubConductivity
        self.eta_scale = float(eta_scale)

    def eta_pair(self, ribbon_index: int, q: float, E_ev: float) -> Tuple[complex, complex]:
        gamma = self.ribbons[ribbon_index].gamma_ev
        eta = self.eta_scale * 1j / (E_ev + 1j * gamma)
        return eta, eta


# --------------------------------------------------------------------------------------
# trial potentials and reductions
# --------------------------------------------------------------------------------------
_PHI_LEN = 10.0  # nm, fixes the shape of the trial potential


def _phi_profile(x: np.ndarray, amp: float = 1.0) -> np.ndarray:
    """A globally ODD trial potential: ``phi(-x) == -phi(x)`` exactly.

    A pure plane-wave potential ``-E0*x`` would make ``E_x`` uniform and every
    nonlinear source identically zero, which would make the symmetry tests vacuous.
    The cubic term keeps ``dE_x/dx`` finite.
    """
    u = np.asarray(x, dtype=float) / _PHI_LEN
    return (-amp * (u + 0.4 * u**3)).astype(np.complex128)


def _x_nodes(system: Any, j: int) -> np.ndarray:
    theta, _h, _L = nl._grid(system, j)
    r = system.ribbons[j]
    return r.x0_nm + r.W_nm * theta


def _odd_phi(system: Any, amp: float = 1.0) -> List[np.ndarray]:
    """``_phi_profile`` sampled on every ribbon, so the result is odd system-wide."""
    return [_phi_profile(_x_nodes(system, j), amp) for j in range(len(system.ribbons))]


def _cancellation_scale(system: Any, rho: Sequence[np.ndarray]) -> float:
    """``sum_j W_j^2 * trapz(|theta*rho_j|, theta)``: the size alpha would have
    without the parity cancellation.  Comparing against this is a sharper statement
    of "exactly zero" than comparing against ``|alpha^(1,1)|``."""
    total = 0.0
    for j in range(len(system.ribbons)):
        theta, _h, _L = nl._grid(system, j)
        total += system.ribbons[j].W_nm ** 2 * float(
            np.trapz(np.abs(theta * np.asarray(rho[j])), theta)
        )
    return total


_E_EV = 0.20


# ======================================================================================
# parity laws: the sharpest available test of the source assembly
# ======================================================================================
def test_ref_single_centred_ribbon_has_no_shg():
    """A single ribbon centred at x = 0 is centrosymmetric, so alpha^(2,2) vanishes.

    Also asserts the source and the solved density are NOT small, so the cancellation
    is a genuine parity cancellation in the moment integral rather than a vacuous zero.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=0.0, L=81)])
    phi = _odd_phi(system)
    res = nl.solve_shg(system, 0.0, _E_EV, phi)

    src_scale = float(np.max(np.abs(res["rho_nl"][0])))
    rho_scale = float(np.max(np.abs(res["rho"][0])))
    assert src_scale > 1e-6, "SHG source is numerically zero: test would be vacuous"
    assert rho_scale > 1e-6, "solved rho^(2,2) is numerically zero: test would be vacuous"

    assert abs(res["alpha"]) < 1e-10 * _cancellation_scale(system, res["rho"])


def test_ref_single_centred_ribbon_shg_source_is_even():
    """The parity law behind the test above: rho_NL^(2,2) is EVEN about the centre.

    ``E_x`` is even, ``dE_x`` and ``div_term`` are odd, so ``div_term*dE_x`` and
    ``E_x*d(dE_x)/dx`` are both even; the odd moment ``trapz(theta*rho)`` then vanishes.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=0.0, L=81)])
    src = nl.shg_source_rho(system, 0, 0.0, _E_EV, _odd_phi(system)[0])
    # Compare only away from the boundary, where the one-sided stencils are not
    # reversal-symmetric by construction.
    interior = src[3:-3]
    assert np.max(np.abs(interior - interior[::-1])) < 1e-9 * np.max(np.abs(interior))


def test_ref_symmetric_pair_has_no_shg():
    """Two identical ribbons placed symmetrically about the origin are also
    centrosymmetric, so alpha^(2,2) vanishes."""
    system = _RefSystem(
        [
            _RefRibbon(W_nm=8.0, x0_nm=-7.0, L=61),
            _RefRibbon(W_nm=8.0, x0_nm=+7.0, L=61),
        ]
    )
    res = nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))

    assert float(np.max(np.abs(res["rho_nl"][0]))) > 1e-6
    assert abs(res["alpha"]) < 1e-10 * _cancellation_scale(system, res["rho"])


def test_ref_unequal_widths_give_shg():
    """Two ribbons of DIFFERENT width break inversion symmetry: alpha^(2,2) != 0."""
    system = _RefSystem(
        [
            _RefRibbon(W_nm=6.0, x0_nm=-8.0, L=61),
            _RefRibbon(W_nm=10.0, x0_nm=+8.0, L=61),
        ]
    )
    res = nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))
    scale = _cancellation_scale(system, res["rho"])
    assert abs(res["alpha"]) > 0.0
    # Not a near-cancellation either: the asymmetry survives at the percent level.
    assert abs(res["alpha"]) > 1e-3 * scale


def test_ref_offcentre_single_ribbon_gives_shg():
    """A single ribbon displaced off the origin is not centrosymmetric with respect to
    the (odd) driving potential, so alpha^(2,2) != 0.  Guards against an assembly that
    returns zero for every geometry."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    res = nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))
    assert abs(res["alpha"]) > 1e-3 * _cancellation_scale(system, res["rho"])


# ======================================================================================
# doping-sign law
# ======================================================================================
def test_ref_doping_sign_flips_alpha22_exactly():
    """Flipping ``doping_sign`` flips alpha^(2,2) exactly (rtol 1e-10).

    ``rho_NL^(2,2)`` is strictly linear in ``sigma_shg_1d_q0``, which carries the
    explicit doping sign, and the block operator ``I - N`` does not depend on it.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    phi = _odd_phi(system)
    plus = nl.solve_shg(system, 0.0, _E_EV, phi, doping_sign=+1.0)
    minus = nl.solve_shg(system, 0.0, _E_EV, phi, doping_sign=-1.0)

    assert abs(plus["alpha"]) > 0.0
    assert plus["alpha"] == pytest.approx(-minus["alpha"], rel=1e-10)
    # stronger: the whole density flips node by node
    np.testing.assert_allclose(plus["rho"][0], -minus["rho"][0], rtol=1e-10, atol=0.0)


def test_ref_doping_sign_leaves_alpha33_unchanged():
    """alpha^(3,3) is EVEN in ``doping_sign``, with and without the cascade.

    The local third-order term carries no doping sign.  The cascade term is linear in
    ``rho_shg`` (one factor of the sign) times the second-order coefficient ``A``
    (a second factor), so it is even.  This test is what forces ``nonlinear.py`` to
    pass the ribbon's doping sign into ``sigma_cascade_abcd`` rather than leaving it
    at the default ``+1`` -- see the CONTRACT DEVIATION note in ``thg_source_rho``.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    phi = _odd_phi(system)
    for cascade in (False, True):
        plus = nl.solve_thg(system, 0.0, _E_EV, phi, None, cascade, doping_sign=+1.0)
        minus = nl.solve_thg(system, 0.0, _E_EV, phi, None, cascade, doping_sign=-1.0)
        assert abs(plus["alpha"]) > 0.0
        assert plus["alpha"] == pytest.approx(minus["alpha"], rel=1e-10), (
            f"alpha^(3,3) changed sign under a doping flip (cascade={cascade})"
        )


def test_ref_cascade_term_is_even_in_doping_sign():
    """Isolate the claim above: the cascade-only part of the source is even in the
    doping sign, i.e. it really is a product of two second-order coefficients."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    phi = _odd_phi(system)
    out = {}
    for sign in (+1.0, -1.0):
        full = nl.solve_thg(system, 0.0, _E_EV, phi, None, True, doping_sign=sign)
        local = nl.solve_thg(system, 0.0, _E_EV, phi, None, False, doping_sign=sign)
        out[sign] = full["rho_nl"][0] - local["rho_nl"][0]
    assert np.max(np.abs(out[+1.0])) > 1e-9
    np.testing.assert_allclose(out[+1.0], out[-1.0], rtol=1e-10, atol=0.0)


# ======================================================================================
# third harmonic
# ======================================================================================
def test_ref_thg_nonzero_for_single_centred_ribbon():
    """THG is an odd order: no symmetry obstruction, so alpha^(3,3) != 0 even for the
    centrosymmetric single ribbon that gives exactly zero SHG."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=0.0, L=81)])
    phi = _odd_phi(system)
    thg = nl.solve_thg(system, 0.0, _E_EV, phi)
    shg = nl.solve_shg(system, 0.0, _E_EV, phi)

    assert abs(thg["alpha"]) > 1e-3 * _cancellation_scale(system, thg["rho"])
    assert abs(shg["alpha"]) < 1e-10 * _cancellation_scale(system, shg["rho"])


def test_ref_cascade_changes_alpha33():
    """``cascade=False`` changes alpha^(3,3) by a finite amount at a strongly coupled
    geometry (two unequal ribbons with a 1 nm gap).

    Measured relative change in this harness: |d alpha33| / |alpha33_local| = 0.2476.
    The band below is a regression guard on the reference harness -- it is NOT a
    physical reference value, since the harness kernel and conductivities are stubs
    and the cascade/local ratio is fixed by their arbitrary relative magnitudes.
    """
    system = _RefSystem(
        [
            _RefRibbon(W_nm=6.0, x0_nm=-3.5, L=61),
            _RefRibbon(W_nm=10.0, x0_nm=+6.5, L=61),
        ]
    )
    phi = _odd_phi(system)
    with_c = nl.solve_thg(system, 0.0, _E_EV, phi, None, True)
    without_c = nl.solve_thg(system, 0.0, _E_EV, phi, None, False)

    rel = abs(with_c["alpha"] - without_c["alpha"]) / abs(without_c["alpha"])
    assert rel > 1e-6, "cascade made no difference to alpha^(3,3)"
    assert 0.23 < rel < 0.27, f"cascade relative contribution moved to {rel:.6f}"


def test_ref_explicit_rho_shg_matches_internal_cascade():
    """Passing ``rho_shg`` explicitly reproduces the internally computed cascade."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    phi = _odd_phi(system)
    shg = nl.solve_shg(system, 0.0, _E_EV, phi)
    auto = nl.solve_thg(system, 0.0, _E_EV, phi, None, True)
    given = nl.solve_thg(system, 0.0, _E_EV, phi, shg["rho"], True)
    assert given["alpha"] == pytest.approx(auto["alpha"], rel=1e-12)


# ======================================================================================
# exact power laws in the driving potential
# ======================================================================================
def test_ref_shg_source_is_exactly_quadratic_in_phi():
    """rho_NL^(2,2) is bilinear in the fundamental potential: doubling phi quadruples it."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    one = nl.shg_source_rho(system, 0, 0.0, _E_EV, _odd_phi(system, 1.0)[0])
    two = nl.shg_source_rho(system, 0, 0.0, _E_EV, _odd_phi(system, 2.0)[0])
    np.testing.assert_allclose(two, 4.0 * one, rtol=1e-12, atol=0.0)


def test_ref_thg_local_source_is_exactly_cubic_in_phi():
    """The local rho_NL^(3,3) is trilinear: doubling phi multiplies it by eight."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    one = nl.thg_source_rho(system, 0, 0.0, _E_EV, _odd_phi(system, 1.0)[0], None)
    two = nl.thg_source_rho(system, 0, 0.0, _E_EV, _odd_phi(system, 2.0)[0], None)
    np.testing.assert_allclose(two, 8.0 * one, rtol=1e-12, atol=0.0)


# ======================================================================================
# derivative identities: the source brackets are the stated continuum expressions
# ======================================================================================
def _sine_setup() -> Tuple[_RefSystem, np.ndarray, np.ndarray, float, float]:
    """A single wide ribbon driven by ``phi = A sin(k x)``, ``k = 2 pi / W``.

    Analytic derivatives let the discretised brackets be checked against closed forms.
    """
    W, L, amp = 20.0, 401, 1.0
    system = _RefSystem([_RefRibbon(W_nm=W, x0_nm=0.0, L=L)])
    x = _x_nodes(system, 0)
    k = 2.0 * np.pi / W
    phi = (amp * np.sin(k * x)).astype(np.complex128)
    return system, x, phi, k, amp


def test_ref_shg_bracket_is_d_dx_of_Ex_times_dEx():
    """The SHG bracket discretises ``d/dx ( E_x * dE_x/dx )``.

    With ``phi = A sin(kx)``: ``E_x = -A k cos``, so
    ``d/dx(E_x dE_x/dx) = (phi'')^2 + phi' phi''' = -A^2 k^4 cos(2kx)``.
    """
    system, x, phi, k, amp = _sine_setup()
    src = nl.shg_source_rho(system, 0, 0.0, _E_EV, phi)
    pref = (-1j / (2.0 * nl._omega(_E_EV))) * _StubConductivity.sigma_shg_1d_q0(
        _E_EV, system.ribbons[0].gamma_ev, doping_sign=+1.0
    )
    bracket = src / pref
    analytic = -(amp**2) * k**4 * np.cos(2.0 * k * x)

    scale = amp**2 * k**4
    err = np.max(np.abs(bracket[3:-3] - analytic[3:-3])) / scale
    assert err < 1e-3, f"SHG bracket deviates from d/dx(E_x dE_x/dx) by {err:.3e}"


def test_ref_thg_local_bracket_is_d_dx_of_Ex_cubed():
    """``(div_term + 2*dE_x) * E_x**2`` discretises ``d(E_x**3)/dx``.

    ``div_term`` and ``dE_x`` are two discretisations of the same ``dE_x/dx``, so the
    stated weighting sums to ``3 E_x**2 dE_x/dx``.  With ``phi = A sin(kx)`` that is
    ``3 A^3 k^4 sin(kx) cos(kx)^2``.
    """
    system, x, phi, k, amp = _sine_setup()
    src = nl.thg_source_rho(system, 0, 0.0, _E_EV, phi, None)
    pref = (-1j / (3.0 * nl._omega(_E_EV))) * _StubConductivity.sigma_thg_local(
        _E_EV, system.ribbons[0].EF_ev, system.ribbons[0].gamma_ev
    )
    bracket = src / pref
    analytic = 3.0 * amp**3 * k**4 * np.sin(k * x) * np.cos(k * x) ** 2

    scale = amp**3 * k**4
    err = np.max(np.abs(bracket[3:-3] - analytic[3:-3])) / scale
    assert err < 1e-3, f"THG local bracket deviates from d(E_x^3)/dx by {err:.3e}"


# ======================================================================================
# white-box check of the block assembly
# ======================================================================================
@pytest.mark.parametrize(("harmonic", "which"), [(2, "shg"), (3, "thg")])
def test_ref_block_matrix_is_D_times_V_in_that_order(harmonic: int, which: str):
    """``N[j,jp] = (W_jp/W_j) * D_j(s) @ V(j, jp, s*q)`` -- operators in the OPPOSITE
    order to the linear problem, with ``D`` carrying the OBSERVATION ribbon's eta.

    Reassembles ``N`` by hand from the stub operators at finite q (so ``D`` and ``V``
    do not commute and the ordering is observable) and checks the solved density.
    """
    q = 0.013
    system = _RefSystem(
        [
            _RefRibbon(W_nm=6.0, x0_nm=-8.0, L=41, gamma_ev=0.02, EF_ev=0.4),
            _RefRibbon(W_nm=10.0, x0_nm=+8.0, L=51, gamma_ev=0.05, EF_ev=0.7),
        ]
    )
    phi = _odd_phi(system)
    res = (
        nl.solve_shg(system, q, _E_EV, phi)
        if which == "shg"
        else nl.solve_thg(system, q, _E_EV, phi, None, False)
    )

    sizes = [nl._grid(system, j)[2] for j in range(2)]
    offs = np.concatenate(([0], np.cumsum(sizes)))
    A = np.eye(int(offs[-1]), dtype=np.complex128)
    for j in range(2):
        theta_j, h_j, L_j = nl._grid(system, j)
        W_j = system.ribbons[j].W_nm
        eta_p, eta_l = system.eta_pair(j, harmonic * q, harmonic * _E_EV)
        D_j = _StubOperators.build_D(harmonic * q, W_j, h_j, L_j, eta_p, eta_l)
        for jp in range(2):
            theta_jp, h_jp, _ = nl._grid(system, jp)
            r_jp = system.ribbons[jp]
            V = _StubOperators.build_V(
                harmonic * q,
                W_j,
                r_jp.W_nm,
                system.ribbons[j].x0_nm,
                r_jp.x0_nm,
                system.ribbons[j].z0_nm,
                r_jp.z0_nm,
                theta_j,
                theta_jp,
                h_jp,
            )
            A[offs[j] : offs[j + 1], offs[jp] : offs[jp + 1]] -= (
                r_jp.W_nm / W_j
            ) * (D_j @ V)

    expected = np.linalg.solve(A, np.concatenate(res["rho_nl"]))
    got = np.concatenate(res["rho"])
    assert np.max(np.abs(got)) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


def test_ref_polarizability_matches_trapz_definition():
    """``alpha = sum_j W_j^2 trapz(theta*rho_j, theta)``, evaluated independently."""
    system = _RefSystem(
        [_RefRibbon(W_nm=6.0, x0_nm=-8.0, L=41), _RefRibbon(W_nm=10.0, x0_nm=8.0, L=51)]
    )
    res = nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))
    expected = 0.0 + 0.0j
    for j in range(2):
        theta, _h, _L = nl._grid(system, j)
        expected += system.ribbons[j].W_nm ** 2 * np.trapz(theta * res["rho"][j], theta)
    assert res["alpha"] == pytest.approx(expected, rel=1e-13)


# ======================================================================================
# hot path: the div_term stencil, and the cost structure of a solve
# ======================================================================================
@pytest.mark.parametrize("q", [0.0, 0.021])
def test_ref_div_term_stencil_matches_dense_build_D_geom(q: float):
    """The O(L) stencil used for ``div_term`` reproduces ``-(D_geom @ phi)/W**2``.

    ``nonlinear.py`` evaluates the matvec from the closed form the contract pins for
    ``build_D_geom`` (``tridiag/h**2 - (qW)**2 I``) instead of materialising the dense
    matrix per source evaluation.  This is the equivalence that licenses that.

    ``atol`` is tied to the magnitude of the result because the dense product and the
    stencil sum their terms in different orders; interior nodes where ``div_term``
    passes through zero carry no meaningful relative tolerance.
    """
    system = _RefSystem([_RefRibbon(W_nm=14.0, x0_nm=2.0, L=57)])
    phi = _odd_phi(system)[0]
    theta, h, L = nl._grid(system, 0)
    W = system.ribbons[0].W_nm
    dense = -(_StubOperators.build_D_geom(q, W, h, L) @ phi) / W**2
    np.testing.assert_allclose(
        nl._div_term(system, 0, q, phi),
        dense,
        rtol=1e-11,
        atol=1e-12 * float(np.max(np.abs(dense))),
    )


@pytest.mark.parametrize("q", [0.0, 0.021])
def test_real_div_term_stencil_matches_operators_build_D_geom(q: float):
    """Same equivalence against the REAL ``operators.build_D_geom``.

    This is the test that matters: ``operators`` implements the boundary rows as
    ``[-1, 1]/h**2`` (vanishing normal current / zero ghost flux at the ribbon edge),
    NOT the naive ``[-2, 1]/h**2``.  The stencil in ``nonlinear._div_term`` was
    initially written with the naive row and was wrong by ``phi[0]/h**2`` at each edge
    node; this test caught it.  Keep it passing whenever ``operators`` changes its
    boundary convention.
    """
    ops = pytest.importorskip("operators", reason="operators.py not written yet")
    system = _RefSystem([_RefRibbon(W_nm=14.0, x0_nm=2.0, L=57)])
    phi = _odd_phi(system)[0]
    theta, h, L = nl._grid(system, 0)
    W = system.ribbons[0].W_nm
    dense = -(np.asarray(ops.build_D_geom(q, W, h, L)) @ phi) / W**2
    np.testing.assert_allclose(
        nl._div_term(system, 0, q, phi),
        dense,
        rtol=1e-11,
        atol=1e-12 * float(np.max(np.abs(dense))),
    )


def test_ref_source_assembly_builds_no_matrices():
    """Assembling a nonlinear source must not build ANY dense operator.

    The first implementation called ``build_D_geom`` inside the source assembly, which
    made the dominant cost an O(L^2) matrix build per evaluation (510 ms at 3201 nodes
    versus 0.4 ms once removed).  This pins it at zero.
    """
    ops = _CountingOperators()
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=3.0, L=81)])
    system.operators = ops
    phi = _odd_phi(system)[0]

    nl.shg_source_rho(system, 0, 0.0, _E_EV, phi)
    nl.thg_source_rho(system, 0, 0.0, _E_EV, phi, phi)
    assert ops.calls == {"build_V": 0, "build_D": 0, "build_D_geom": 0}


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("shg", {"build_V": 4, "build_D": 2, "build_D_geom": 0}),
        ("thg_local", {"build_V": 4, "build_D": 2, "build_D_geom": 0}),
        ("thg_cascade", {"build_V": 8, "build_D": 4, "build_D_geom": 0}),
    ],
)
def test_ref_solve_builds_each_kernel_block_exactly_once(call: str, expected: dict):
    """Pin the kernel-build count of a solve on a 2-ribbon system.

    ``nrib**2`` ``build_V`` calls and ``nrib`` ``build_D`` calls per harmonic, and no
    more: the self-block potential reuses the diagonal ``V`` from the solve rather than
    rebuilding it.  That reuse matters because ``V`` may come from
    ``substrate.build_Vbar``, which runs an NQ-point quadrature per entry.
    """
    ops = _CountingOperators()
    system = _RefSystem(
        [_RefRibbon(W_nm=6.0, x0_nm=-3.5, L=41), _RefRibbon(W_nm=10.0, x0_nm=6.5, L=51)]
    )
    system.operators = ops
    phi = _odd_phi(system)

    if call == "shg":
        nl.solve_shg(system, 0.0, _E_EV, phi)
    elif call == "thg_local":
        nl.solve_thg(system, 0.0, _E_EV, phi, None, False)
    else:
        nl.solve_thg(system, 0.0, _E_EV, phi, None, True)
    assert ops.calls == expected


# ======================================================================================
# plumbing / contract guards
# ======================================================================================
def test_ref_grid_reconstruction_matches_geometry_block():
    """``L = round(W/dx)+1`` with ``h = 1/(L-1)`` and nodes spanning [-1/2, 1/2]."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, L=None)])
    system.dx_nm = 0.5
    theta, h, L = nl._grid(system, 0)
    assert L == 41
    assert h == pytest.approx(1.0 / 40.0, rel=0, abs=1e-15)
    assert theta[0] == pytest.approx(-0.5) and theta[-1] == pytest.approx(0.5)
    assert np.max(np.abs(np.diff(theta) - h)) < 1e-14


def test_ref_wrong_length_potential_is_rejected():
    system = _RefSystem([_RefRibbon(W_nm=20.0, L=81)])
    with pytest.raises(ValueError):
        nl.shg_source_rho(system, 0, 0.0, _E_EV, np.zeros(80, dtype=np.complex128))


def test_ref_substrate_layers_without_hook_warn():
    """Substrate screening is absent from the nonlinear blocks unless the system
    supplies a ``nonlinear_V_block`` hook; that must not pass silently."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, L=41)])
    system.layers = [object()]
    with pytest.warns(RuntimeWarning, match="substrate"):
        nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))


def test_hbar_fallback_matches_units_module():
    """The local hbar fallback must not drift from ``units.HBAR_EV_FS``."""
    units = pytest.importorskip("units", reason="units.py not written yet")
    assert nl.HBAR_EV_FS_FALLBACK == pytest.approx(units.HBAR_EV_FS, rel=1e-15)


def _reference_cascade_A(
    cond: Any, E_ev: float, gamma_ev: float, s1: int, s2: int
) -> complex:
    """``S_s1_s2 = S2 * D_sum * ( D_s2**2 + 2*D_sum*(D_s1 + D_s2) )``.

    The paper-406 MATLAB form quoted in the task block, built from the harmonic
    denominators ``conductivity`` exposes.
    """
    omega = nl._omega(E_ev)
    gam_fs = gamma_ev / nl._hbar()
    d1 = cond._D(float(s1), omega, gam_fs)
    d2 = cond._D(float(s2), omega, gam_fs)
    dsum = cond._D(float(s1 + s2), omega, gam_fs)
    return complex(cond._S2 * dsum * (d2**2 + 2.0 * dsum * (d1 + d2)))


@pytest.mark.parametrize(("E_ev", "gamma_ev"), [(1.3, 0.025), (0.7, 0.01), (2.1, 0.05)])
def test_reference_cascade_A_generalises_the_q0_collapse_not_the_tensor_A(
    E_ev: float, gamma_ev: float
):
    """The paper-406 ``S_s1_s2`` and ``sigma_cascade_abcd``'s A are different objects.

    This test replaces an earlier ``xfail(strict=True)`` that asserted
    ``sigma_cascade_abcd(...)[0] == S_s1_s2`` and was filed as a suspected
    transcription error upstream.  The premise was wrong: setting ``s1 = s2 = 1``
    (the degenerate case, which IS second-harmonic generation) identifies each
    expression uniquely, and both reduce exactly -- but to DIFFERENT things.

    * ``-S_1_1 == sigma_shg_1d_q0``, the q = 0 one-dimensional COLLAPSE
      ``A + B + 2C``.  So the MATLAB ``S_s1_s2`` is that scalar, generalised to
      mixed harmonics -- which is exactly what a 1-D q = 0 nonlinear source
      needs, and why ``nonlinear._cascade_A`` defaults to it.
    * ``sigma_cascade_abcd(..., 1, 1)[0] == sigma_shg_abc[0]``, the TENSOR
      STRUCTURE A of Eq. (12a), generalised to mixed harmonics.

    Both identities hold to machine precision, so neither module is wrong and
    nothing upstream needs changing.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")

    collapse = cond.sigma_shg_1d_q0(E_ev, gamma_ev, +1)
    s_ref_11 = -_reference_cascade_A(cond, E_ev, gamma_ev, 1, 1)
    assert s_ref_11 == pytest.approx(collapse, rel=1e-14), (
        "the MATLAB reference cascade expression at s1 = s2 = 1 must equal the "
        f"q = 0 1-D collapse A+B+2C; got {s_ref_11!r} vs {collapse!r}"
    )

    tensor_a = cond.sigma_shg_abc(E_ev, gamma_ev, +1)[0]
    cascade_a_11 = cond.sigma_cascade_abcd(E_ev, gamma_ev, 1, 1, +1)[0]
    assert cascade_a_11 == pytest.approx(tensor_a, rel=1e-14), (
        "sigma_cascade_abcd's A at s1 = s2 = 1 must equal the tensor structure A "
        f"of sigma_shg_abc; got {cascade_a_11!r} vs {tensor_a!r}"
    )

    # ... and the two are genuinely different numbers, which is the whole point.
    assert abs(cascade_a_11 - s_ref_11) > 0.1 * abs(s_ref_11)


@pytest.mark.parametrize(("s1", "s2"), [(1, 2), (2, 1)])
def test_cascade_A_discrepancy_is_the_D_s2_squared_term(s1: int, s2: int):
    """Pin the disagreement exactly: ``A_cond + A_ref == -2 S2 D_sum D_s2**2``.

    ``conductivity``'s A is ``-S2 D_sum (3 D_s2^2 + 2 D_sum (D_s1+D_s2))`` at
    ``doping_sign = +1``; the reference is ``+S2 D_sum (D_s2^2 + 2 D_sum (...))``.  The
    ``2 D_sum (...)`` parts cancel in the sum and only ``-2 S2 D_sum D_s2^2`` survives.
    If agent A changes that coefficient, this test reports it immediately.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    E, gamma = _E_EV, 0.02
    a_cond = cond.sigma_cascade_abcd(E, gamma, s1, s2, +1)[0]
    a_ref = _reference_cascade_A(cond, E, gamma, s1, s2)

    omega = nl._omega(E)
    gam_fs = gamma / nl._hbar()
    d_s2 = cond._D(float(s2), omega, gam_fs)
    d_sum = cond._D(float(s1 + s2), omega, gam_fs)
    expected = complex(-2.0 * cond._S2 * d_sum * d_s2**2)

    assert abs(a_cond - a_ref) > 1e-6 * abs(a_ref), "the two forms now agree"
    assert a_cond + a_ref == pytest.approx(expected, rel=1e-10)


def test_cascade_A_is_odd_in_doping_sign_in_both_forms():
    """``nonlinear._cascade_A`` is exactly odd in the doping sign in BOTH forms.

    That is what makes ``alpha^(3,3)`` even under a doping flip, once the cascade source
    picks up its own factor of the sign from ``rho^(2,2)``.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    for form in (nl.CASCADE_A_REFERENCE, nl.CASCADE_A_CONDUCTIVITY):
        plus = nl._cascade_A(cond, _E_EV, 0.02, 1, 2, +1.0, form)
        minus = nl._cascade_A(cond, _E_EV, 0.02, 1, 2, -1.0, form)
        assert abs(plus) > 0.0
        assert plus == pytest.approx(-minus, rel=1e-12), f"form={form}"


def test_cascade_A_form_selects_the_documented_expression():
    """Each ``cascade_A_form`` value returns the expression its docstring names."""
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    E, gamma = _E_EV, 0.02
    ref = nl._cascade_A(cond, E, gamma, 1, 2, +1.0, nl.CASCADE_A_REFERENCE)
    con = nl._cascade_A(cond, E, gamma, 1, 2, +1.0, nl.CASCADE_A_CONDUCTIVITY)
    # _cascade_A folds in -doping_sign, so the reference branch returns -S_1_2.
    assert ref == pytest.approx(-_reference_cascade_A(cond, E, gamma, 1, 2), rel=1e-12)
    assert con == pytest.approx(
        cond.sigma_cascade_abcd(E, gamma, 1, 2, +1)[0], rel=1e-12
    )
    with pytest.raises(ValueError, match="cascade_A_form"):
        nl._cascade_A(cond, E, gamma, 1, 2, +1.0, "nonsense")


def test_cascade_A_falls_back_with_a_warning_when_S2_is_absent():
    """Without ``_S2``/``_D`` the reference form cannot be built; that must not pass
    silently, because the fallback uses the disagreeing coefficient."""

    class _NoPrivates:
        @staticmethod
        def sigma_cascade_abcd(
            E_ev: float, gamma_ev: float, s1: int, s2: int, doping_sign: float = +1
        ) -> Tuple[complex, complex, complex, complex]:
            return (1.0 + 0j, 0j, 0j, 0j)

    with pytest.warns(RuntimeWarning, match="reference cascade A"):
        got = nl._cascade_A(_NoPrivates, _E_EV, 0.02, 1, 2, +1.0, nl.CASCADE_A_REFERENCE)
    assert got == pytest.approx(1.0 + 0j)


def test_ref_cascade_A_form_changes_alpha33():
    """The two cascade-A forms give measurably different alpha^(3,3).

    Confirms the discrepancy is not cosmetic: it moves the observable.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, x0_nm=13.0, L=81)])
    phi = _odd_phi(system)
    a_ref = nl.solve_thg(
        system, 0.0, _E_EV, phi, None, True, cascade_A_form=nl.CASCADE_A_REFERENCE
    )["alpha"]
    a_con = nl.solve_thg(
        system, 0.0, _E_EV, phi, None, True, cascade_A_form=nl.CASCADE_A_CONDUCTIVITY
    )["alpha"]
    assert abs(a_ref - a_con) > 1e-6 * abs(a_ref)


def test_cascade_A_differs_between_1_2_and_2_1():
    """Structural consequence of the reference forms: the two orderings differ only in
    the squared denominator (``D_2**2`` vs ``D_1**2``), so ``A12 != A21``.  This is
    assertable without knowing ``S2`` or ``D_s``, and it is what makes the two cascade
    terms in ``thg_source_rho`` independent."""
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    A12 = cond.sigma_cascade_abcd(_E_EV, 0.02, 1, 2)[0]
    A21 = cond.sigma_cascade_abcd(_E_EV, 0.02, 2, 1)[0]
    assert np.isfinite(A12) and np.isfinite(A21)
    assert abs(A12 - A21) > 1e-12 * max(abs(A12), abs(A21))


# ======================================================================================
# the same physics through the real sibling modules
# ======================================================================================
def _real_modules() -> Tuple[Any, Any]:
    for name in ("units", "conductivity", "operators", "solver", "sources"):
        pytest.importorskip(name, reason=f"{name}.py not written yet")
    import solver  # noqa: PLC0415
    import sources  # noqa: PLC0415

    return solver, sources


def _real_system(solver: Any, specs: Sequence[dict]) -> Any:
    ribbons = [solver.Ribbon(**spec) for spec in specs]
    return solver.RibbonSystem(ribbons, dx_nm=0.25)


def _real_linear(system: Any, sources: Any, E_ev: float) -> Tuple[dict, complex]:
    """Drive with a q = 0 plane wave and return ``(linear result, alpha^(1,1))``."""
    phis = [
        sources.phi_ext_plane_wave(_x_nodes(system, j))
        for j in range(len(system.ribbons))
    ]
    try:
        lin = system.solve_linear(0.0, E_ev, phis)
    except (TypeError, ValueError):
        # fall back in case solve_linear wants one stacked vector
        lin = system.solve_linear(0.0, E_ev, np.concatenate(phis))
    return lin, nl.polarizability_from_rho(system, lin["rho"])


def test_real_grid_matches_the_grid_the_linear_solve_used():
    """``nonlinear._grid`` must land on the SAME grid the linear solve used.

    ``solver.RibbonSystem`` builds its own node counts (31 nodes for W = 6 nm at
    dx = 0.25), which do NOT match the ``round(W/dx)+1`` reconstruction the geometry
    block specifies (25).  ``_grid`` therefore reads the solver's own node array and
    only falls back to the reconstruction when it cannot find one.  If that discovery
    ever breaks, the nonlinear source would be assembled on a different grid from the
    potential driving it, so pin the invariant explicitly rather than relying on a
    length check to trip.
    """
    solver, sources = _real_modules()
    system = _real_system(
        solver, [dict(W_nm=6.0, x0_nm=-3.5), dict(W_nm=10.0, x0_nm=6.5)]
    )
    lin, _a11 = _real_linear(system, sources, _E_EV)
    reconstructed = []
    for j in range(len(system.ribbons)):
        theta, h, L = nl._grid(system, j)
        assert L == len(np.asarray(lin["phi"][j]).ravel()), f"ribbon {j}: phi length"
        assert L == len(np.asarray(lin["rho"][j]).ravel()), f"ribbon {j}: rho length"
        assert h == pytest.approx(1.0 / (L - 1), rel=0, abs=1e-15)
        assert theta[0] == pytest.approx(-0.5) and theta[-1] == pytest.approx(0.5)
        reconstructed.append(int(round(system.ribbons[j].W_nm / system.dx_nm)) + 1)
    # Documents that the fallback is genuinely not equivalent for this solver.
    assert reconstructed != [nl._grid(system, j)[2] for j in range(len(system.ribbons))]


def test_real_single_centred_ribbon_has_no_shg():
    solver, sources = _real_modules()
    system = _real_system(solver, [dict(W_nm=20.0, x0_nm=0.0, EF_ev=0.5, gamma_ev=0.02)])
    lin, a11 = _real_linear(system, sources, _E_EV)
    shg = nl.solve_shg(system, 0.0, _E_EV, lin)
    assert float(np.max(np.abs(shg["rho_nl"][0]))) > 0.0
    assert abs(a11) > 0.0
    assert abs(shg["alpha"]) < 1e-10 * abs(a11)


def test_real_symmetric_pair_has_no_shg():
    solver, sources = _real_modules()
    system = _real_system(
        solver,
        [dict(W_nm=8.0, x0_nm=-7.0, EF_ev=0.5), dict(W_nm=8.0, x0_nm=+7.0, EF_ev=0.5)],
    )
    lin, a11 = _real_linear(system, sources, _E_EV)
    shg = nl.solve_shg(system, 0.0, _E_EV, lin)
    assert abs(shg["alpha"]) < 1e-10 * abs(a11)


def test_real_unequal_widths_give_shg():
    solver, sources = _real_modules()
    system = _real_system(
        solver,
        [dict(W_nm=6.0, x0_nm=-8.0, EF_ev=0.5), dict(W_nm=10.0, x0_nm=+8.0, EF_ev=0.5)],
    )
    lin, _a11 = _real_linear(system, sources, _E_EV)
    shg = nl.solve_shg(system, 0.0, _E_EV, lin)
    assert abs(shg["alpha"]) > 0.0


def test_real_doping_sign_flips_alpha22_only():
    """alpha^(2,2) flips exactly; alpha^(1,1) and alpha^(3,3) are untouched."""
    solver, sources = _real_modules()
    system = _real_system(solver, [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)])
    lin, a11 = _real_linear(system, sources, _E_EV)

    shg_p = nl.solve_shg(system, 0.0, _E_EV, lin, doping_sign=+1.0)
    shg_m = nl.solve_shg(system, 0.0, _E_EV, lin, doping_sign=-1.0)
    assert abs(shg_p["alpha"]) > 0.0
    assert shg_p["alpha"] == pytest.approx(-shg_m["alpha"], rel=1e-10)

    thg_p = nl.solve_thg(system, 0.0, _E_EV, lin, None, True, doping_sign=+1.0)
    thg_m = nl.solve_thg(system, 0.0, _E_EV, lin, None, True, doping_sign=-1.0)
    assert thg_p["alpha"] == pytest.approx(thg_m["alpha"], rel=1e-10)

    # alpha^(1,1) cannot depend on the doping sign: it never enters the linear problem.
    _lin2, a11_again = _real_linear(system, sources, _E_EV)
    assert a11_again == pytest.approx(a11, rel=1e-12)


def test_real_thg_nonzero_for_single_ribbon():
    solver, sources = _real_modules()
    system = _real_system(solver, [dict(W_nm=20.0, x0_nm=0.0, EF_ev=0.5)])
    lin, a11 = _real_linear(system, sources, _E_EV)
    thg = nl.solve_thg(system, 0.0, _E_EV, lin)
    assert abs(thg["alpha"]) > 1e-12 * abs(a11)


def test_real_cascade_changes_alpha33():
    solver, sources = _real_modules()
    system = _real_system(
        solver,
        [dict(W_nm=6.0, x0_nm=-3.5, EF_ev=0.5), dict(W_nm=10.0, x0_nm=+6.5, EF_ev=0.5)],
    )
    lin, _a11 = _real_linear(system, sources, _E_EV)
    with_c = nl.solve_thg(system, 0.0, _E_EV, lin, None, True)
    without_c = nl.solve_thg(system, 0.0, _E_EV, lin, None, False)
    rel = abs(with_c["alpha"] - without_c["alpha"]) / abs(without_c["alpha"])
    assert rel > 1e-9, "cascade made no difference to alpha^(3,3)"


# ======================================================================================
# tensor contractions: which collapse each nonlinear order consumes
# ======================================================================================
# These are the checks that the misunderstanding which produced the old, false cascade
# test ("cascaded C must reduce to degenerate SHG C") did NOT also produce a wrong
# contraction inside nonlinear.py.  It did not: the two orders consume two DIFFERENT
# contractions of the same four coefficients, and each consumes the right one.
#
#   Eq. (12), second harmonic:  A + B + 2C      -- Eq. (12c)'s C multiplies grad(E.E),
#                                                  which is 2 E_x d_x E_x in 1D
#   Eq. (15), cascade:          A + B + C + D   -- Eq. (15c)'s C multiplies
#                                                  sum_j E_j grad(E_j) (one 1-D term),
#                                                  and Eq. (15d)'s D multiplies
#                                                  grad(E_s1.E_s2), whose two 1-D terms
#                                                  come from the two ORDERINGS
_CONTRACTION_GRID = [(0.20, 0.02), (0.70, 0.01), (1.30, 0.05)]


@pytest.mark.parametrize(("E_ev", "gamma_ev"), _CONTRACTION_GRID)
@pytest.mark.parametrize("ds", [+1, -1])
def test_shg_source_coefficient_is_the_eq12_contraction(
    E_ev: float, gamma_ev: float, ds: int
):
    """``shg_source_rho``'s coefficient is ``A + B + 2C`` of Eq. (12), not ``A + B + C``.

    The factor 2 on ``C`` is the whole content of "Eq. (12c)'s C multiplies grad(E.E)":
    in one dimension that structure is ``2 E_x d_x E_x`` while ``A``'s and ``B``'s give
    ``E_x d_x E_x`` once.  Asserting that the neighbouring weights are NOT what the
    module uses is the part that would catch a mis-contraction.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    a, b, c = cond.sigma_shg_abc(E_ev, gamma_ev, ds)
    used = cond.sigma_shg_1d_q0(E_ev, gamma_ev, ds)

    assert used == pytest.approx(a + b + 2.0 * c, rel=1e-13), (
        "the SHG source coefficient is not the Eq. (12) contraction A+B+2C"
    )
    for weight in (1.0, 3.0):
        assert abs(used - (a + b + weight * c)) > 1e-6 * abs(used), (
            f"A+B+{weight}C is numerically indistinguishable from A+B+2C, so this "
            "test cannot discriminate the contraction"
        )


@pytest.mark.parametrize(("E_ev", "gamma_ev"), _CONTRACTION_GRID)
@pytest.mark.parametrize(("s1", "s2"), [(1, 2), (2, 1), (1, 1), (2, 3)])
@pytest.mark.parametrize("ds", [+1.0, -1.0])
def test_cascade_A_is_the_full_abcd_contraction(
    E_ev: float, gamma_ev: float, s1: int, s2: int, ds: float
):
    """``_cascade_A`` (reference form) is exactly ``A + B + C + D`` of Eq. (15).

    This is the identity that settles what ``nonlinear.py`` contracts in the cascade.
    The paper-406 MATLAB coefficient ``-ds S2 D_sum (D_s2^2 + 2 D_sum (D_s1 + D_s2))``
    is not an independent expression: it is the sum of the four Eq. (15) coefficients,
    each with weight 1.  Weight 1 on ``C`` because Eq. (15c)'s structure
    ``sum_j E_j grad(E_j)`` gives a single 1-D monomial; weight 1 on ``D`` because the
    two 1-D terms of ``grad(E_s1 . E_s2)`` are supplied by the two orderings
    ``thg_source_rho`` sums over, not by a factor of 2 here.

    Also asserts that the bare ``sigma_cascade_abcd(...)[0]`` a literal reading of the
    contract would use is a measurably different number, so the test discriminates.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    quad = cond.sigma_cascade_abcd(E_ev, gamma_ev, s1, s2, ds)
    used = nl._cascade_A(cond, E_ev, gamma_ev, s1, s2, ds, nl.CASCADE_A_REFERENCE)

    assert used == pytest.approx(sum(quad), rel=1e-13), (
        "the cascade coefficient is not the Eq. (15) contraction A+B+C+D"
    )
    assert abs(used - quad[0]) > 1e-3 * abs(used), (
        "A alone is indistinguishable from A+B+C+D here"
    )


@pytest.mark.parametrize(("E_ev", "gamma_ev"), _CONTRACTION_GRID)
@pytest.mark.parametrize("ds", [+1.0, -1.0])
def test_cascade_A_degenerate_is_the_shg_collapse(
    E_ev: float, gamma_ev: float, ds: float
):
    """At ``s1 = s2 = 1`` the cascade coefficient equals ``sigma_shg_1d_q0``.

    The degenerate case IS second-harmonic generation, so this is what fixes the
    cascade's bookkeeping uniquely.  In particular it forbids assigning the full
    ``grad(E_s1.E_s2)`` structure (a factor 2 on ``D``) to each ordering: ``A+B+C+2D``
    does not reduce to the Eq. (12) collapse, and the second assertion checks that the
    two candidates are far enough apart for the first one to decide between them.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    collapse = cond.sigma_shg_1d_q0(E_ev, gamma_ev, ds)
    used = nl._cascade_A(cond, E_ev, gamma_ev, 1, 1, ds, nl.CASCADE_A_REFERENCE)
    assert used == pytest.approx(collapse, rel=1e-13)

    quad = cond.sigma_cascade_abcd(E_ev, gamma_ev, 1, 1, ds)
    doubled_D = sum(quad) + quad[3]
    assert abs(doubled_D - collapse) > 1e-3 * abs(collapse), (
        "A+B+C+2D also reduces to the SHG collapse, so this test cannot rule it out"
    )


@pytest.mark.parametrize(("s1", "s2"), [(1, 2), (1, 3), (2, 3)])
def test_cascade_D_is_symmetric_under_harmonic_swap(s1: int, s2: int):
    """``sigma_D`` is symmetric in ``(s1, s2)``; ``sigma_A`` is not.

    ``D = +ds 2 S2 D_sum^2 (D_s1 + D_s2)`` depends on the pair only through ``D_sum``
    and ``D_s1 + D_s2``.  That symmetry is why "one ``D`` per ordering" reproduces both
    1-D terms of ``grad(E_s1 . E_s2)``, and it is the structural premise of the two
    tests above.
    """
    cond = pytest.importorskip("conductivity", reason="conductivity.py not written yet")
    fwd = cond.sigma_cascade_abcd(_E_EV, 0.02, s1, s2, +1)
    rev = cond.sigma_cascade_abcd(_E_EV, 0.02, s2, s1, +1)
    assert fwd[3] == pytest.approx(rev[3], rel=1e-13), "sigma_D is not symmetric"
    assert abs(fwd[0] - rev[0]) > 1e-6 * abs(fwd[0]), "sigma_A is symmetric too"


def test_ref_cascade_brackets_are_the_two_orderings_unsymmetrised():
    """The cascade source is ``A12 d/dx(E_1 d_x E_2) + A21 d/dx(E_2 d_x E_1)``.

    Fits the two coefficients out of the assembled source by least squares, against two
    1-D monomials made linearly independent by driving the fundamental and the second
    harmonic with different spatial profiles.  A swap of ``A12``/``A21``, or a
    symmetrised ``grad(E.E)`` form that puts ``(A12+A21)/2`` on both, moves the fitted
    pair and fails -- which is non-vacuous only because ``A12 != A21``, asserted here.
    """
    W, L = 20.0, 401
    k = 2.0 * np.pi / W
    system = _RefSystem([_RefRibbon(W_nm=W, x0_nm=0.0, L=L)])
    x = _x_nodes(system, 0)
    phi = np.sin(k * x).astype(np.complex128)
    phi_2w = (0.7 * np.sin(2.0 * k * x) + 0.3 * np.cos(k * x)).astype(np.complex128)

    full = nl.thg_source_rho(system, 0, 0.0, _E_EV, phi, phi_2w)
    local = nl.thg_source_rho(system, 0, 0.0, _E_EV, phi, None)
    pref = -1j / (3.0 * nl._omega(_E_EV))
    observed = (full - local) / pref

    E_1, dE_1, div_1, dx = nl._fields(system, 0, 0.0, phi)
    E_2, dE_2, div_2, _dx2 = nl._fields(system, 0, 0.0, phi_2w)
    m12 = div_1 * dE_2 + E_1 * np.gradient(dE_2, dx)
    m21 = div_2 * dE_1 + E_2 * np.gradient(dE_1, dx)

    gam = system.ribbons[0].gamma_ev
    a12 = nl._cascade_A(_StubConductivity, _E_EV, gam, 1, 2, +1.0, nl.CASCADE_A_REFERENCE)
    a21 = nl._cascade_A(_StubConductivity, _E_EV, gam, 2, 1, +1.0, nl.CASCADE_A_REFERENCE)
    assert abs(a12 - a21) > 1e-3 * abs(a12), "A12 == A21: a swap would be undetectable"

    basis = np.stack([m12, m21], axis=1)
    coeff, *_rest = np.linalg.lstsq(basis, observed, rcond=None)
    residual = float(np.max(np.abs(basis @ coeff - observed)))
    assert residual < 1e-10 * float(np.max(np.abs(observed))), (
        f"the cascade source is not spanned by the two orderings (residual {residual:.3e})"
    )
    assert coeff[0] == pytest.approx(a12, rel=1e-8), "coefficient of d/dx(E_1 d_x E_2)"
    assert coeff[1] == pytest.approx(a21, rel=1e-8), "coefficient of d/dx(E_2 d_x E_1)"


# ======================================================================================
# substrate screening of the nonlinear orders, and the per-order frequency
# ======================================================================================
# ``nonlinear.py`` gets substrate screening only through the
# ``system.nonlinear_V_block(j, jp, q, E_ev)`` hook, and the hook must be called at the
# frequency of EACH ORDER: 2 omega for the second-harmonic blocks, 3 omega for the
# third, 2 omega again for the cascade's phi_2w inside a third-order solve.  Because
# r(Q, omega) is frequency dependent only through the sheet term of
# ``substrate.reflection_coefficient``, a stack of plain dielectrics has an EXACTLY
# frequency-independent r and cannot demonstrate that plumbing at all -- those tests
# would pass with the fundamental hard-wired everywhere.  The plumbing tests below
# therefore use a DISPERSIVE stack (a conducting sheet), and the dielectric stack is
# kept as the control that must show no difference.
#
# q != 0 throughout: ``RibbonSystem.nonlinear_V_block`` raises at q = 0 with a stack
# present (the reflected Coulomb integral is logarithmically divergent there), and
# ``substrate.build_Vbar`` warns when its quadrature step cannot resolve |q|.  z0_nm is
# finite so that the zero-standoff accuracy warning does not fire either.
_Q_SUB = 0.05          # 1/nm; q*W = 1 for the 20 nm ribbons used here
_Z0_SUB = 1.0          # nm standoff, convention A


def _substrate_modules() -> Tuple[Any, Any, Any]:
    """``(solver, sources, substrate)``, skipping while any of them is unwritten."""
    solver, sources = _real_modules()
    substrate = pytest.importorskip("substrate", reason="substrate.py not written yet")
    return solver, sources, substrate


def _transparent_stack(substrate: Any) -> List[Any]:
    """A stack with r == 0 everywhere; ``build_Vbar`` short-circuits it to exact zeros."""
    return [substrate.Layer(1.0), substrate.Layer(1.0)]


def _dielectric_stack(substrate: Any) -> List[Any]:
    """A real but NON-dispersive stack: r is exactly frequency independent."""
    return [substrate.Layer(1.0), substrate.Layer(4.0)]


def _dispersive_stack(substrate: Any) -> List[Any]:
    """A stack whose r genuinely depends on omega, via a conducting sheet.

    ``sigma_sheet`` sits at the TOP interface of its layer, so it must NOT be put on
    ``layers[0]`` (there is no interface above the first layer and it would be ignored,
    making every frequency test below vacuous).
    """
    return [substrate.Layer(1.0), substrate.Layer(2.0, sigma_sheet=0.05 + 0.0j)]


def _sub_system(solver: Any, layers: Optional[Sequence[Any]], specs: Sequence[dict]) -> Any:
    ribbons = [solver.Ribbon(z0_nm=_Z0_SUB, **spec) for spec in specs]
    return solver.RibbonSystem(ribbons, layers=layers, dx_nm=0.25)


def _sub_linear(system: Any, sources: Any, E_ev: float = _E_EV) -> dict:
    phis = [
        sources.phi_ext_plane_wave(_x_nodes(system, j))
        for j in range(len(system.ribbons))
    ]
    return system.solve_linear(_Q_SUB, E_ev, phis)


class _HookProxy:
    """A ``RibbonSystem`` with its ``nonlinear_V_block`` intercepted.

    Delegates every other attribute to the wrapped system, so ``nonlinear.py`` sees a
    real solver.  ``log`` records ``(j, jp, q, E_ev)`` of each hook call; ``remap_E``
    rewrites the energy the wrapped hook is given, which is how the "screened at the
    fundamental instead" comparison is constructed.

    The four positional parameters matter: ``nonlinear._hook_takes_energy`` inspects the
    signature, and a three-argument hook would be routed down the warning path.
    """

    def __init__(
        self,
        system: Any,
        remap_E: Any = None,
        log: Optional[List[Tuple[int, int, float, Optional[float]]]] = None,
    ) -> None:
        self._system = system
        self._remap_E = remap_E
        self._log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._system, name)

    def nonlinear_V_block(
        self, j: int, jp: int, q: float, E_ev: Optional[float] = None
    ) -> np.ndarray:
        if self._log is not None:
            self._log.append(
                (int(j), int(jp), float(q), None if E_ev is None else float(E_ev))
            )
        used = E_ev if self._remap_E is None or E_ev is None else self._remap_E(float(E_ev))
        return self._system.nonlinear_V_block(j, jp, q, used)


def _alpha_scale(system: Any, rho: Sequence[np.ndarray]) -> float:
    """``_cancellation_scale`` for a real system (same definition)."""
    return _cancellation_scale(system, rho)


# --------------------------------------------------------------------------------------
# the hook is consulted at each order's OWN frequency
# --------------------------------------------------------------------------------------
def test_real_hook_is_called_at_each_harmonic_frequency():
    """SHG asks for ``(2q, 2E)``; THG with cascade asks for ``(2q, 2E)`` AND ``(3q, 3E)``.

    The most direct statement of the per-order frequency rule: record what the hook is
    actually handed.  The fundamental ``(q, E)`` must never appear -- asking for it is
    precisely the silent error, since the blocks keep their shape and the solve still
    converges.
    """
    solver, sources, substrate = _substrate_modules()
    system = _sub_system(
        solver, _dispersive_stack(substrate), [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)]
    )
    lin = _sub_linear(system, sources)

    log: List[Tuple[int, int, float, Optional[float]]] = []
    nl.solve_shg(_HookProxy(system, log=log), _Q_SUB, _E_EV, lin)
    asked = {(round(q, 12), None if e is None else round(e, 12)) for _j, _jp, q, e in log}
    assert asked == {(round(2.0 * _Q_SUB, 12), round(2.0 * _E_EV, 12))}, asked

    log = []
    nl.solve_thg(_HookProxy(system, log=log), _Q_SUB, _E_EV, lin, None, True)
    asked = {(round(q, 12), None if e is None else round(e, 12)) for _j, _jp, q, e in log}
    assert asked == {
        (round(2.0 * _Q_SUB, 12), round(2.0 * _E_EV, 12)),
        (round(3.0 * _Q_SUB, 12), round(3.0 * _E_EV, 12)),
    }, asked
    assert all(e is not None for _j, _jp, _q, e in log), "a block was built without E_ev"


def test_real_cascade_phi_2w_is_screened_at_2E_inside_a_third_order_solve():
    """``phi_2w`` is a second-harmonic quantity, so its blocks are built at ``(2q, 2E)``.

    Exercises the branch where ``rho_shg`` is passed in explicitly, which is the only
    path that builds ``phi_2w`` outside a second-harmonic solve and so the only one
    that could plausibly reach for the third-harmonic energy it is nested inside.
    """
    solver, sources, substrate = _substrate_modules()
    system = _sub_system(
        solver, _dispersive_stack(substrate), [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)]
    )
    lin = _sub_linear(system, sources)
    shg = nl.solve_shg(system, _Q_SUB, _E_EV, lin)

    log: List[Tuple[int, int, float, Optional[float]]] = []
    nl.solve_thg(_HookProxy(system, log=log), _Q_SUB, _E_EV, lin, shg["rho"], True)
    energies = {round(e, 12) for _j, _jp, _q, e in log if e is not None}
    assert round(2.0 * _E_EV, 12) in energies, energies
    assert round(3.0 * _E_EV, 12) in energies, energies
    assert round(_E_EV, 12) not in energies, "a block was screened at the fundamental"


@pytest.mark.parametrize("which", ["shg", "thg"])
def test_real_dispersive_stack_detects_the_wrong_screening_frequency(which: str):
    """Evaluating the substrate block at the fundamental gives a DIFFERENT answer.

    This is the test that proves the ``2 omega`` / ``3 omega`` really is used rather
    than merely documented: the same solve is run twice through the same hook, once
    correctly and once with the energy rewritten to the fundamental, and the two must
    disagree.  Measured: 23% in alpha^(2,2) for this stack.
    """
    solver, sources, substrate = _substrate_modules()
    system = _sub_system(
        solver, _dispersive_stack(substrate), [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)]
    )
    lin = _sub_linear(system, sources)
    wrong = _HookProxy(system, remap_E=lambda _e: _E_EV)

    if which == "shg":
        right_a = nl.solve_shg(system, _Q_SUB, _E_EV, lin)["alpha"]
        wrong_a = nl.solve_shg(wrong, _Q_SUB, _E_EV, lin)["alpha"]
    else:
        right_a = nl.solve_thg(system, _Q_SUB, _E_EV, lin, None, True)["alpha"]
        wrong_a = nl.solve_thg(wrong, _Q_SUB, _E_EV, lin, None, True)["alpha"]

    assert abs(right_a) > 0.0
    rel = abs(right_a - wrong_a) / abs(right_a)
    assert rel > 1e-3, (
        f"screening the harmonics at the fundamental changed alpha by only {rel:.3e}: "
        "either the harmonic energy is not reaching the hook, or this stack is not "
        "dispersive enough for the test to discriminate"
    )


@pytest.mark.parametrize("which", ["shg", "thg"])
def test_real_dielectric_stack_is_blind_to_the_screening_frequency(which: str):
    """The control for the test above: a NON-dispersive stack must not notice.

    ``substrate.reflection_coefficient`` uses the energy in exactly one place, the sheet
    term, so a stack of plain dielectrics has an exactly frequency-independent r.  The
    correct and the wrong-frequency runs must therefore agree BIT FOR BIT.  If this ever
    fails while the previous test passes, the difference seen there is not coming from
    r's frequency dependence and the conclusion drawn from it is void.
    """
    solver, sources, substrate = _substrate_modules()
    system = _sub_system(
        solver, _dielectric_stack(substrate), [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)]
    )
    lin = _sub_linear(system, sources)
    wrong = _HookProxy(system, remap_E=lambda _e: _E_EV)

    if which == "shg":
        right_a = nl.solve_shg(system, _Q_SUB, _E_EV, lin)["alpha"]
        wrong_a = nl.solve_shg(wrong, _Q_SUB, _E_EV, lin)["alpha"]
    else:
        right_a = nl.solve_thg(system, _Q_SUB, _E_EV, lin, None, True)["alpha"]
        wrong_a = nl.solve_thg(wrong, _Q_SUB, _E_EV, lin, None, True)["alpha"]
    assert abs(right_a) > 0.0
    assert right_a == wrong_a


# --------------------------------------------------------------------------------------
# the screening reaches the observable, and a transparent stack changes nothing
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("stack", ["dielectric", "dispersive"])
@pytest.mark.parametrize("which", ["shg", "thg"])
def test_real_substrate_changes_the_nonlinear_polarizability(stack: str, which: str):
    """A real substrate must MOVE alpha^(2,2) and alpha^(3,3), not just the linear
    response.  Without the hook being consumed this test fails: the nonlinear blocks
    would be the bare kernel and the two answers would coincide."""
    solver, sources, substrate = _substrate_modules()
    layers = (
        _dielectric_stack(substrate) if stack == "dielectric" else _dispersive_stack(substrate)
    )
    specs = [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)]
    bare = _sub_system(solver, None, specs)
    screened = _sub_system(solver, layers, specs)

    solve = (
        (lambda sy, lin: nl.solve_shg(sy, _Q_SUB, _E_EV, lin))
        if which == "shg"
        else (lambda sy, lin: nl.solve_thg(sy, _Q_SUB, _E_EV, lin, None, True))
    )
    a_bare = solve(bare, _sub_linear(bare, sources))["alpha"]
    a_scr = solve(screened, _sub_linear(screened, sources))["alpha"]

    assert abs(a_bare) > 0.0
    rel = abs(a_scr - a_bare) / abs(a_bare)
    assert rel > 1e-2, f"substrate screening did not reach alpha ({rel:.3e})"


@pytest.mark.parametrize("which", ["shg", "thg"])
def test_real_transparent_stack_is_bit_for_bit_unchanged(which: str):
    """A transparent stack (r == 0 everywhere) must change NOTHING.

    ``substrate.build_Vbar`` short-circuits r == 0 to exact zeros and
    ``RibbonSystem.nonlinear_V_block`` then returns the system's own direct block, so
    adding the substrate path must not perturb a single bit.  Anything weaker than
    equality here would hide a normalisation slip in the hook.
    """
    solver, sources, substrate = _substrate_modules()
    specs = [dict(W_nm=20.0, x0_nm=13.0, EF_ev=0.5)]
    bare = _sub_system(solver, None, specs)
    clear = _sub_system(solver, _transparent_stack(substrate), specs)

    solve = (
        (lambda sy, lin: nl.solve_shg(sy, _Q_SUB, _E_EV, lin))
        if which == "shg"
        else (lambda sy, lin: nl.solve_thg(sy, _Q_SUB, _E_EV, lin, None, True))
    )
    res_bare = solve(bare, _sub_linear(bare, sources))
    res_clear = solve(clear, _sub_linear(clear, sources))

    assert abs(res_bare["alpha"]) > 0.0
    assert res_clear["alpha"] == res_bare["alpha"]
    np.testing.assert_array_equal(res_clear["rho"][0], res_bare["rho"][0])


# --------------------------------------------------------------------------------------
# inversion symmetry, with and without a substrate
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("stack", ["none", "dielectric", "dispersive"])
@pytest.mark.parametrize(
    ("label", "specs"),
    [
        ("centred single", [dict(W_nm=20.0, x0_nm=0.0, EF_ev=0.5)]),
        (
            "centred symmetric pair",
            [dict(W_nm=8.0, x0_nm=-7.0, EF_ev=0.5), dict(W_nm=8.0, x0_nm=+7.0, EF_ev=0.5)],
        ),
    ],
)
def test_real_centrosymmetric_geometry_has_no_shg_even_over_a_substrate(
    stack: str, label: str, specs: Sequence[dict]
):
    """A centrosymmetric geometry gives EXACTLY zero alpha^(2,2), substrate or not.

    A substrate breaks inversion symmetry in z but leaves it intact in x, and the
    x-polarised channel is the only one this 1-D collapse carries -- so the zero must
    survive.  That makes this a simultaneous check on the hook contract and on the
    frequency plumbing: the screened block is a function of the node SEPARATION only,
    so it commutes with node reversal and preserves the parity of the source; a hook
    that returned something asymmetric, or that was handed a different frequency on
    different blocks of the same solve, would break the cancellation.
    """
    solver, sources, substrate = _substrate_modules()
    layers = {
        "none": None,
        "dielectric": _dielectric_stack(substrate),
        "dispersive": _dispersive_stack(substrate),
    }[stack]
    system = _sub_system(solver, layers, specs)
    res = nl.solve_shg(system, _Q_SUB, _E_EV, _sub_linear(system, sources))

    scale = _alpha_scale(system, res["rho"])
    assert float(np.max(np.abs(res["rho_nl"][0]))) > 0.0, "vacuous: source is zero"
    assert scale > 0.0, "vacuous: solved density is zero"
    assert abs(res["alpha"]) < 1e-10 * scale, (
        f"{label} over a {stack} substrate is centrosymmetric in x but gave "
        f"alpha^(2,2) = {res['alpha']!r} against a cancellation scale of {scale:.3e}"
    )


@pytest.mark.parametrize("stack", ["dielectric", "dispersive"])
def test_real_asymmetric_pair_over_a_substrate_does_give_shg(stack: str):
    """The companion to the test above: the zero is a symmetry statement, not a bug.

    Two ribbons of different width over the same substrate break inversion symmetry in
    x, and alpha^(2,2) then survives at order unity relative to the cancellation scale.
    """
    solver, sources, substrate = _substrate_modules()
    layers = (
        _dielectric_stack(substrate) if stack == "dielectric" else _dispersive_stack(substrate)
    )
    system = _sub_system(
        solver,
        layers,
        [dict(W_nm=6.0, x0_nm=-8.0, EF_ev=0.5), dict(W_nm=10.0, x0_nm=+8.0, EF_ev=0.5)],
    )
    res = nl.solve_shg(system, _Q_SUB, _E_EV, _sub_linear(system, sources))
    assert abs(res["alpha"]) > 1e-3 * _alpha_scale(system, res["rho"])


# --------------------------------------------------------------------------------------
# the warning contract around the hook
# --------------------------------------------------------------------------------------
def test_ref_three_argument_hook_with_layers_warns():
    """A hook that cannot be told its harmonic frequency is not a usable hook.

    With layers present, a ``(j, jp, q)`` hook would have to screen every order at one
    unspecified frequency.  That is the silent error the per-order plumbing exists to
    prevent, so it warns rather than passing quietly.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, L=41)])
    system.layers = [object()]
    system.nonlinear_V_block = lambda j, jp, q: nl._V_block(_RefSystem(system.ribbons), j, jp, q)
    with pytest.warns(RuntimeWarning, match="harmonic frequency"):
        nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))


def test_ref_four_argument_hook_with_layers_does_not_warn():
    """...and a hook that DOES accept the energy is used silently.

    The warning must not survive the fix: once the hook is genuinely in use at its own
    frequency there is nothing to warn about, and a warning that always fires is a
    warning nobody reads.  Also records the energies, so this doubles as a check that
    the no-warning path is the path that receives ``2*E_ev``.
    """
    seen: List[float] = []

    def hook(j: int, jp: int, q: float, E_ev: Optional[float] = None) -> np.ndarray:
        seen.append(-1.0 if E_ev is None else float(E_ev))
        return nl._V_block(_RefSystem(system.ribbons), j, jp, q)

    system = _RefSystem([_RefRibbon(W_nm=20.0, L=41)])
    system.layers = [object()]
    system.nonlinear_V_block = hook
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))
    assert seen and set(seen) == {2.0 * _E_EV}


def test_ref_three_argument_hook_without_layers_is_silent():
    """A layer-free system may still supply the older three-argument hook."""
    system = _RefSystem([_RefRibbon(W_nm=20.0, L=41)])
    inner = _RefSystem(system.ribbons)
    system.nonlinear_V_block = lambda j, jp, q: nl._V_block(inner, j, jp, q)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = nl.solve_shg(system, 0.0, _E_EV, _odd_phi(system))
    assert np.max(np.abs(res["rho"][0])) > 0.0


def test_hook_accepts_energy_probe_answers_the_cross_module_contract():
    """``_hook_accepts_energy(system)`` is the probe the solver side pins by name.

    ``tests/test_integration.py`` asserts ``nonlinear._hook_accepts_energy(system) is
    True`` for a real ``RibbonSystem``, so the name and the SYSTEM-level argument are a
    contract, not an internal detail.  It must be False both for a system with no hook
    and for one whose hook predates the energy argument, because those are exactly the
    two cases ``_V_block`` has to warn about.
    """
    system = _RefSystem([_RefRibbon(W_nm=20.0, L=41)])
    assert nl._hook_accepts_energy(system) is False

    system.nonlinear_V_block = lambda j, jp, q: np.zeros((41, 41), dtype=np.complex128)
    assert nl._hook_accepts_energy(system) is False

    system.nonlinear_V_block = lambda j, jp, q, E_ev=None: np.zeros(
        (41, 41), dtype=np.complex128
    )
    assert nl._hook_accepts_energy(system) is True


def test_real_solver_hook_is_detected_as_energy_aware():
    """The same probe against the real ``solver.RibbonSystem``."""
    solver, _sources = _real_modules()
    system = _real_system(solver, [dict(W_nm=20.0, x0_nm=0.0)])
    assert nl._hook_accepts_energy(system) is True
    assert nl._hook_takes_energy(system.nonlinear_V_block) is True
