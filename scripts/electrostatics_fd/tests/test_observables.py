"""Tests for ``observables.py``: the induced field's quadrature and its absolute scale.

The subject of this file is the 2026-11-03 fix in ``observables.induced_field_q``:
the ribbon-node integral now uses UNIFORM CELL WEIGHTS (full weight ``dx`` on
every node, both ends included), the quadrature ``operators.build_V`` applies to
the same spatial sum, instead of ``np.trapz``, which halves the end nodes.

Four independent checks of the absolute normalisation, in descending order of
decisiveness:

1. :func:`test_v_gradient_consistency_x` / ``_z`` -- the field is the exact
   negative gradient of the potential ``operators.build_V`` builds from the same
   rho.  This is the test that would have caught the trapz bug: it reads 1.000000
   with cell weights and 0.955-class ratios with trapezoidal ones.
2. :func:`test_energy_conservation_dipole_vs_ohmic_loss` -- the power a dipole
   source delivers equals ``Re(sigma)|E|^2`` integrated over the ribbon, with no
   fitted constant anywhere.
3. :func:`test_q_to_zero_reduces_to_line_charge_field` -- the textbook
   ``E = 2 lambda / r`` limit.
4. :func:`test_kernel_structure_against_own_quadrature` -- the K_1/K_0 structure
   and the three components' relative signs against a Coulomb kernel built here
   from the 3D ``1/|r-r'|`` integral, not from the library's formula.

plus :func:`test_convergence_order_is_second_order_in_dx`, which fails if anyone
reinstates ``np.trapz``.

Units: eV-nm-fs Gaussian throughout (see ``units.py``).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k0, k1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conductivity  # noqa: E402
import observables  # noqa: E402
import operators  # noqa: E402
import solver  # noqa: E402
import sources  # noqa: E402
import units  # noqa: E402


# ===========================================================================
# helpers
# ===========================================================================
def _nodes(W_nm: float, L: int, x0_nm: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Ribbon node grid exactly as ``solver`` builds it: ``(x, h, dx)``."""
    th = np.linspace(-0.5, 0.5, int(L))
    h = float(th[1] - th[0])
    return x0_nm + float(W_nm) * th, h, h * float(W_nm)


def _phi_from_V(
    q: float,
    W_nm: float,
    theta: np.ndarray,
    h: float,
    rho: np.ndarray,
    x_obs: float,
    z_obs: float,
) -> complex:
    """Physical Coulomb potential at ``(x_obs, z_obs)`` via ``operators.build_V``.

    ``build_V`` returns ``2 h_jp K_0``-type cell integrals in the normalised
    coordinate of ribbon ``j'``, so ``V @ rho`` is the physical line integral
    ``int dx' 2 K_0(|q| s) rho(x')`` DIVIDED by ``W_jp`` (see the continuity
    derivation in ``solver.py``: ``phi_ind = 2 sum_jp h_jp sum_l' K_0 W_jp rho``).
    Multiplying by ``W_jp`` here recovers the physical potential, with no other
    adjustment -- which is the whole point of the test.

    The observation point is a single row placed by ``W_j = 1``, ``x_j = x_obs``,
    ``theta_j = [0]``, so ``z_obs != 0`` puts the block on ``build_V``'s plain
    ``2 h K_0`` branch -- the midpoint rule, i.e. literally the uniform cell
    weight whose consistency is under test.
    """
    V = operators.build_V(
        q, 1.0, float(W_nm), float(x_obs), 0.0, float(z_obs), 0.0,
        np.array([0.0]), theta, h,
    )
    return float(W_nm) * complex((V @ rho)[0])


def _d5(f, t: float, step: float) -> complex:
    """Five-point central first derivative of a complex-valued ``f`` at ``t``."""
    return (
        -f(t + 2.0 * step) + 8.0 * f(t + step) - 8.0 * f(t - step) + f(t - 2.0 * step)
    ) / (12.0 * step)


def _E_trapz(q: float, x: np.ndarray, rho: np.ndarray, r0: tuple[float, float]):
    """The SUPERSEDED trapezoidal form, written out here as the regression target.

    Kept local on purpose: ``observables`` offers no switch back to it, so the
    only way a future edit can resurrect it is by editing the module, and these
    tests then fail.
    """
    x0, z0 = r0
    d = x0 - x
    s = np.sqrt(d * d + z0 * z0)
    qa = abs(q)
    return np.array([
        2.0 * qa * np.trapz(rho * d / s * k1(qa * s), x),
        -2.0j * q * np.trapz(rho * k0(qa * s), x),
        2.0 * qa * np.trapz(rho * (z0 / s) * k1(qa * s), x),
    ])


def _eigenmode_rho(q: float, W_nm: float, L: int) -> tuple[np.ndarray, np.ndarray]:
    """``(x_nodes, rho)`` of the fundamental geometric eigenmode of one ribbon.

    This is the physically relevant profile for the quadrature question: the
    induced charge density of a plasmon diverges at the ribbon edges, so the end
    nodes carry a large share of the integral and halving their weight is not a
    small perturbation.  Measured edge/centre amplitude ratio ~16 at L = 301.
    """
    rib = solver.Ribbon(W_nm=W_nm, L=L)
    sys_ = solver.RibbonSystem(rib)
    _, u, _ = sys_.eigenmodes(q)
    D = operators.build_D_geom(q, W_nm, sys_.h[0], L)
    return sys_.x_nodes[0], (D @ u[:, 0]) / W_nm


# ===========================================================================
# 1. THE DECISIVE TEST: E is the exact negative gradient of V's potential
# ===========================================================================
@pytest.mark.parametrize("L", [151, 301])
def test_v_gradient_consistency_x(L: int) -> None:
    """``Ex = -d/dx0 (V rho)`` to finite-difference accuracy -- ratio 1, not 0.955.

    Both sides integrate over the same nodes.  ``build_V`` gives every node the
    full cell weight; with ``np.trapz`` the two end nodes were halved and this
    ratio moved by percent, which is exactly the bug this file exists to pin.
    """
    q, W = 0.05, 100.0
    x_obs, z_obs = 23.0, 7.0
    x, h, dx = _nodes(W, L)
    rho = _eigenmode_rho(q, W, L)[1]
    theta = np.linspace(-0.5, 0.5, L)

    ex = observables.induced_field_q(q, x, rho, (x_obs, z_obs))[0]
    ref = -_d5(lambda t: _phi_from_V(q, W, theta, h, rho, t, z_obs), x_obs, 1e-3)
    trapz = _E_trapz(q, x, rho, (x_obs, z_obs))[0]
    print(
        f"\nL={L} Ex: cell/Vgrad = {abs(ex / ref):.10f}   "
        f"trapz/Vgrad = {abs(trapz / ref):.6f}"
    )
    assert ex == pytest.approx(ref, rel=1e-9)
    # and the superseded form is genuinely, measurably off -- not a coin flip
    assert abs(abs(trapz / ref) - 1.0) > 1e-3


@pytest.mark.parametrize("L", [151, 301])
def test_v_gradient_consistency_z(L: int) -> None:
    """``Ez = -d/dz0 (V rho)``, same configuration, same accuracy."""
    q, W = 0.05, 100.0
    x_obs, z_obs = 23.0, 7.0
    x, h, _ = _nodes(W, L)
    rho = _eigenmode_rho(q, W, L)[1]
    theta = np.linspace(-0.5, 0.5, L)

    ez = observables.induced_field_q(q, x, rho, (x_obs, z_obs))[2]
    ref = -_d5(lambda t: _phi_from_V(q, W, theta, h, rho, x_obs, t), z_obs, 1e-3)
    trapz = _E_trapz(q, x, rho, (x_obs, z_obs))[2]
    print(
        f"\nL={L} Ez: cell/Vgrad = {abs(ez / ref):.10f}   "
        f"trapz/Vgrad = {abs(trapz / ref):.6f}"
    )
    assert ez == pytest.approx(ref, rel=1e-9)
    assert abs(abs(trapz / ref) - 1.0) > 1e-3


def test_v_gradient_consistency_y() -> None:
    """``Ey = -i q phi``: the y component is the same potential times ``-i q``.

    No finite difference is involved -- in the ``e^{i q y}`` representation
    ``d/dy -> i q`` exactly -- so this one pins the cell weight to roundoff.
    """
    q, W, L = 0.05, 100.0, 301
    x_obs, z_obs = 23.0, 7.0
    x, h, _ = _nodes(W, L)
    rho = _eigenmode_rho(q, W, L)[1]
    theta = np.linspace(-0.5, 0.5, L)

    ey = observables.induced_field_q(q, x, rho, (x_obs, z_obs))[1]
    ref = -1.0j * q * _phi_from_V(q, W, theta, h, rho, x_obs, z_obs)
    trapz = _E_trapz(q, x, rho, (x_obs, z_obs))[1]
    print(f"\nEy: cell/ref = {abs(ey / ref):.12f}   trapz/ref = {abs(trapz / ref):.6f}")
    assert ey == pytest.approx(ref, rel=1e-12)
    assert abs(abs(trapz / ref) - 1.0) > 1e-3


def test_cell_weight_matches_build_V_exactly_for_a_point_source() -> None:
    """One source node: ``phi`` from ``build_V`` is ``2 dx K_0``, so ``dx`` is the weight.

    Strips the test down to a single cell, where the uniform weight is visible as
    a number rather than inferred from a ratio.
    """
    q, W, L = 0.04, 80.0, 41
    x, h, dx = _nodes(W, L)
    z_obs, x_obs = 5.0, 31.0
    rho = np.zeros(L, dtype=np.complex128)
    rho[7] = 1.0 + 0.0j

    s = math.hypot(x_obs - x[7], z_obs)
    phi_expected = 2.0 * dx * k0(abs(q) * s)
    phi_V = _phi_from_V(q, W, np.linspace(-0.5, 0.5, L), h, rho, x_obs, z_obs)
    assert phi_V == pytest.approx(phi_expected, rel=1e-12)

    ex, ey, ez = observables.induced_field_q(q, x, rho, (x_obs, z_obs))
    assert ex == pytest.approx(
        2.0 * abs(q) * dx * (x_obs - x[7]) / s * k1(abs(q) * s), rel=1e-13
    )
    assert ey == pytest.approx(-2.0j * q * dx * k0(abs(q) * s), rel=1e-13)
    assert ez == pytest.approx(
        2.0 * abs(q) * dx * (z_obs / s) * k1(abs(q) * s), rel=1e-13
    )


# ===========================================================================
# 2. CONVERGENCE ORDER -- the test that trips on a reinstated trapz
# ===========================================================================
#: Observation point and wave vector of the convergence study.
_CONV_Q, _CONV_W = 0.05, 100.0
_CONV_XOBS, _CONV_ZOBS = 23.0, 7.0


def _conv_rho(x: np.ndarray) -> np.ndarray:
    """A smooth profile that is deliberately NON-ZERO at the ribbon edges.

    With rho vanishing at both ends the two quadratures coincide identically and
    the question has no content; the physical induced density is largest there,
    so non-zero ends are also the realistic case.
    """
    t = (np.asarray(x, dtype=np.float64) - 0.0) / _CONV_W
    return np.cos(2.2 * np.pi * t) + 0.5 * np.sin(1.3 * np.pi * t) + 0.3


def _conv_kernel_x(x: float) -> float:
    d = _CONV_XOBS - x
    s = math.hypot(d, _CONV_ZOBS)
    return 2.0 * abs(_CONV_Q) * d / s * k1(abs(_CONV_Q) * s)


def _conv_exact(a: float, b: float) -> float:
    """``int_a^b rho(x') kernel(x') dx'`` to machine precision."""
    return quad(
        lambda t: _conv_rho(np.array(t))[()] * _conv_kernel_x(t),
        a, b, epsabs=1e-14, epsrel=1e-13, limit=400,
    )[0]


def _measured_order(errs: list[float]) -> float:
    """Mean log2 error ratio over successive dx halvings."""
    return float(
        np.mean([math.log2(errs[i - 1] / errs[i]) for i in range(1, len(errs))])
    )


def test_convergence_order_is_second_order_in_dx() -> None:
    """Uniform cell weights converge at second order; ``np.trapz`` at first.

    The reference is the exact integral over the UNION OF THE NODE CELLS,
    ``[x_0 - dx/2, x_{L-1} + dx/2]``.  That is not a free choice: ``build_V``'s
    exact branches integrate the Coulomb kernel over a cell of width ``h``
    CENTRED on each node, so the charge distribution the whole finite-difference
    scheme represents occupies exactly that interval, and it is the continuum
    object the discrete field must converge to if the field is to stay the
    gradient of the discrete potential.

    (Against the narrower fixed interval ``[-W/2, W/2]`` the roles swap --
    trapezoidal is second order there and cell weights first -- which is the
    honest statement of the trade-off.  It is the wrong interval for this scheme:
    picking it would mean the field converges to the gradient of a potential
    ``build_V`` does not build, which is precisely the inconsistency measured in
    ``test_v_gradient_consistency_x``.  Consistency, not the convergence
    exponent, is what decides, and the exponent then follows.)
    """
    errs_cell: list[float] = []
    errs_trapz: list[float] = []
    Ls = [51, 101, 201, 401, 801]
    for L in Ls:
        x, _, dx = _nodes(_CONV_W, L)
        rho = _conv_rho(x).astype(np.complex128)
        ref = _conv_exact(x[0] - 0.5 * dx, x[-1] + 0.5 * dx)
        cell = observables.induced_field_q(
            _CONV_Q, x, rho, (_CONV_XOBS, _CONV_ZOBS)
        )[0].real
        trapz = _E_trapz(_CONV_Q, x, rho, (_CONV_XOBS, _CONV_ZOBS))[0].real
        errs_cell.append(abs(cell - ref))
        errs_trapz.append(abs(trapz - ref))

    order_cell = _measured_order(errs_cell)
    order_trapz = _measured_order(errs_trapz)
    print("\n  L     dx      |cell - exact|   |trapz - exact|")
    for L, ec, et in zip(Ls, errs_cell, errs_trapz):
        print(f"{L:5d} {_CONV_W / (L - 1):8.5f}   {ec:.4e}       {et:.4e}")
    print(f"  measured order: cell {order_cell:.3f}   trapz {order_trapz:.3f}")

    assert order_cell > 1.5, (
        f"uniform cell weights should converge at second order in dx; measured "
        f"{order_cell:.3f}. An order near 1 means the trapezoidal node weighting "
        f"has been reinstated in observables.induced_field_q (it measures "
        f"{order_trapz:.3f} here)."
    )
    assert order_trapz < 1.3, (
        f"the superseded trapezoidal form is supposed to be FIRST order against "
        f"this reference; measured {order_trapz:.3f}. If this fails the reference "
        f"integral, not the library, has changed."
    )
    # the two are not the same rule: at the coarsest grid they differ by >10x
    assert errs_trapz[0] / errs_cell[0] > 10.0


def test_trapz_error_on_a_plasmon_mode_decays_only_as_sqrt_dx() -> None:
    """The recorded sqrt(dx) behaviour of the superseded form, on a real mode.

    The induced density of a ribbon plasmon diverges at the edges like
    ``(W/2 - |x|)^{-1/2}``, so the end-node amplitude grows as ``dx^{-1/2}`` and
    the half-weight that ``np.trapz`` gives it leaves an error ``~ dx^{1/2}``.
    Measured against the cell-weighted field, which the gradient test above has
    already tied to ``build_V``.
    """
    q, W = 0.05, 100.0
    r0 = (200.0, 10.0)
    Ls = [151, 301, 601, 1201]
    rel: list[float] = []
    for L in Ls:
        x, _, _ = _nodes(W, L)
        rho = _eigenmode_rho(q, W, L)[1]
        cell = observables.induced_field_q(q, x, rho, r0)[0]
        trapz = _E_trapz(q, x, rho, r0)[0]
        rel.append(abs(trapz - cell) / abs(cell))
    order = _measured_order(rel)
    print("\n  L      |trapz - cell| / |cell|")
    for L, r in zip(Ls, rel):
        print(f"{L:5d}   {r:.5f}")
    print(f"  measured decay order in dx: {order:.3f} (sqrt(dx) would be 0.5)")
    assert 0.3 < order < 0.8, (
        f"expected the trapezoidal error on an edge-divergent mode to decay as "
        f"~sqrt(dx) (order 0.5); measured {order:.3f}"
    )
    assert rel[0] > 0.03, (
        f"the trapezoidal error at L = 151 is only {rel[0]:.4f}; it was measured "
        "at several percent and this test has lost its grip on the geometry"
    )


def test_trapz_vs_cell_ratio_at_L301_is_on_the_record() -> None:
    """Records the absolute size of the fixed bug at L = 301, for the log.

    Printed, and asserted only loosely: the exact percentage depends on where the
    observation point sits, so the assertion is that it is a percent-scale
    effect and that the cell-weighted value is the larger one (trapz is LOW,
    because it discards half of the two largest contributions).
    """
    q, W, L = 0.05, 100.0, 301
    x, _, _ = _nodes(W, L)
    rho = _eigenmode_rho(q, W, L)[1]
    print("\n  r0 (nm)         trapz/cell  Ex        Ez")
    worst = 0.0
    for r0 in ((200.0, 10.0), (60.0, 0.5), (0.0, 2.0), (51.0, 0.2)):
        cell = np.asarray(observables.induced_field_q(q, x, rho, r0))
        trapz = _E_trapz(q, x, rho, r0)
        rx = abs(trapz[0]) / abs(cell[0])
        rz = abs(trapz[2]) / abs(cell[2])
        print(f"  {str(r0):15s} {rx:.6f}  {rz:.6f}")
        worst = max(worst, abs(rx - 1.0), abs(rz - 1.0))
    assert worst > 0.03, (
        f"the trapezoidal/cell discrepancy at L = 301 is at most {worst:.4f}; it "
        "was measured in the 4%-30% range over these observation points"
    )


# ===========================================================================
# 3. q -> 0: the textbook 2D line-charge field
# ===========================================================================
def test_q_to_zero_reduces_to_line_charge_field() -> None:
    """``E -> 2 lambda / r`` as ``q -> 0``, with ``lambda = int rho dx``.

    ``K_1(u) -> 1/u``, so ``2|q| (d/s) K_1(|q| s) -> 2 d / s^2``: the in-plane
    and out-of-plane components become the field of a single line charge of
    strength ``lambda`` at the ribbon's charge centroid, magnitude ``2 lambda/r``
    and direction along ``r``.  ``Ey ~ -2 i q lambda ln(1/|q|)`` instead vanishes
    only logarithmically, which is why it is excluded here.

    Two limits are in play and both have to be taken, or the test measures the
    wrong thing: ``|q| r -> 0`` for ``K_1 -> 1/u``, and ``W/r -> 0`` for the
    distribution to look like one line charge at its centroid.  The second one
    floors the agreement at ``(sigma_rho/r)^2`` -- 7e-5 at r = 1.1 um, which is
    where a first attempt at this test stalled -- so ``r`` is taken out to
    100 um and the q sequence scaled to match.
    """
    W, L = 40.0, 401
    x, _, dx = _nodes(W, L)
    rho = (np.exp(-((x / 9.0) ** 2)) + 0.25) + 0.0j
    lam = dx * np.sum(rho)                      # same cell weights as the field
    xc = float((dx * np.sum(x * rho) / lam).real)   # charge centroid
    r0 = (6.0e4, 8.0e4)                         # r = 100 um >> W = 40 nm

    rows = []
    qs = (1e-6, 1e-7, 1e-8)
    for q in qs:
        ex, _, ez = observables.induced_field_q(q, x, rho, r0)
        d, z0 = r0[0] - xc, r0[1]
        r2 = d * d + z0 * z0
        ex_ref, ez_ref = 2.0 * lam * d / r2, 2.0 * lam * z0 / r2
        rows.append((q, ex / ex_ref, ez / ez_ref))
        print(f"\n  q={q:g}: Ex/ref = {abs(ex / ex_ref):.8f}  "
              f"Ez/ref = {abs(ez / ez_ref):.8f}  |E| = {abs(complex(ex, 0)):.4e}")
    # The leading correction to K_1(u) ~ 1/u is O(u^2 ln u), so the error must
    # shrink roughly as q^2 ln(1/q) -- not merely be small.
    err = [max(abs(a - 1.0), abs(b - 1.0)) for _, a, b in rows]
    print("  errors:", [f"{e:.3e}" for e in err])
    assert err[0] > err[1] > err[2]
    assert err[-1] < 3e-5
    assert err[-2] / err[-1] > 20.0, (
        f"the q -> 0 error is not shrinking like q^2 ln(1/q): "
        f"{err[-2]:.3e} -> {err[-1]:.3e} for a tenfold drop in q"
    )

    # magnitude 2 lambda / r and direction along r, at the smallest q
    ex, _, ez = observables.induced_field_q(qs[-1], x, rho, r0)
    r = math.hypot(r0[0] - xc, r0[1])
    assert math.hypot(abs(ex), abs(ez)) == pytest.approx(2.0 * lam.real / r, rel=1e-4)
    assert (ez / ex).real == pytest.approx(r0[1] / (r0[0] - xc), rel=1e-4)

    # The same limit without the single-centroid truncation: node by node,
    # E -> 2 sum_l' dx rho_l' (r0 - r_l')/|r0 - r_l'|^2.  Nothing is expanded
    # here, so this must hold far more tightly, and it does at a near-ribbon
    # observation point where the centroid form would be useless.
    r_near = (70.0, 15.0)
    ex, _, ez = observables.induced_field_q(1e-8, x, rho, r_near)
    d, z0 = r_near[0] - x, r_near[1]
    s2 = d * d + z0 * z0
    # residual is the O((q s)^2 ln(q s)) tail of K_1, ~7e-12 at q = 1e-8
    assert ex == pytest.approx(2.0 * dx * np.sum(rho * d / s2), rel=1e-10)
    assert ez == pytest.approx(2.0 * dx * np.sum(rho * z0 / s2), rel=1e-10)


# ===========================================================================
# 4. the kernel structure, against a Coulomb integral written here
# ===========================================================================
def test_y_fourier_transform_of_the_coulomb_kernel_is_2K0() -> None:
    """``1/|r-r'| -> 2 K_0(|q| s)``: the one assumption behind the whole module.

    Checked on the INVERSE transform, which is the numerically well-behaved
    direction: ``K_0(q s)`` decays like ``exp(-q s)/sqrt(q s)``, so

        (1/2 pi) int dq e^{-i q y} 2 K_0(|q| s)
            = (2/pi) int_0^inf K_0(q s) cos(q y) dq
            == 1 / sqrt(s^2 + y^2)

    converges exponentially, whereas the forward integral
    ``int dy e^{i q y}/sqrt(s^2+y^2)`` decays only as ``1/|y|`` and no
    general-purpose quadrature reaches it.  The ``1/(2 pi)`` appearing here is
    the measure of paper 456 Eq. (21) -- the one the module docstring says
    belongs in the caller's q integral, not in ``induced_field_q``.
    """
    for s, y in ((7.0, 3.0), (3.0, 0.0), (40.0, 95.0), (0.5, 12.0)):
        val = (2.0 / math.pi) * quad(
            lambda t: k0(t * s) * math.cos(t * y),
            0.0, 60.0 / s, epsabs=1e-14, epsrel=1e-12, limit=800,
        )[0]
        ref = 1.0 / math.hypot(s, y)
        print(f"\n  s={s} y={y}: quad = {val:.12e}  1/|r| = {ref:.12e}  "
              f"ratio {val / ref:.12f}")
        assert val == pytest.approx(ref, rel=1e-8)


def test_kernel_structure_against_own_quadrature() -> None:
    """K_1 for x/z, K_0 for y, and the three relative signs, from first principles.

    The reference potential is built here from the module's own stated kernel
    ``2 K_0`` (verified independently just above) with uniform cell weights, and
    differentiated numerically.  No library formula is reused: this pins that
    ``Ex, Ez`` carry ``K_1`` with a ``+`` sign along ``(r0 - r')``, that ``Ey``
    carries ``K_0`` with a ``-i q`` prefactor, and that nothing has swapped.
    """
    q, W, L = 0.06, 60.0, 401
    x, _, dx = _nodes(W, L)
    rho = (0.7 + 0.25j) * np.cos(0.11 * x) + 0.4 * np.sin(0.05 * x)
    x_obs, z_obs = 17.0, 9.0

    def phi(xo: float, zo: float) -> complex:
        s = np.sqrt((xo - x) ** 2 + zo * zo)
        return complex(dx * np.sum(rho * 2.0 * k0(abs(q) * s)))

    ex_ref = -_d5(lambda t: phi(t, z_obs), x_obs, 1e-3)
    ez_ref = -_d5(lambda t: phi(x_obs, t), z_obs, 1e-3)
    ey_ref = -1.0j * q * phi(x_obs, z_obs)

    ex, ey, ez = observables.induced_field_q(q, x, rho, (x_obs, z_obs))
    print(f"\n  Ex {ex:.10e} vs {ex_ref:.10e}\n  Ey {ey:.10e} vs {ey_ref:.10e}"
          f"\n  Ez {ez:.10e} vs {ez_ref:.10e}")
    assert ex == pytest.approx(ex_ref, rel=1e-9)
    assert ey == pytest.approx(ey_ref, rel=1e-13)
    assert ez == pytest.approx(ez_ref, rel=1e-9)

    # structural signs, checked on a single positive point charge
    one = np.zeros(L)
    one[L // 2] = 1.0
    for dxs, dzs in ((+30.0, +5.0), (-30.0, +5.0), (+30.0, -5.0)):
        xo, zo = x[L // 2] + dxs, dzs
        a, b, c = observables.induced_field_q(q, x, one, (xo, zo))
        assert np.sign(a.real) == np.sign(dxs)   # Ex points away from the charge
        assert np.sign(c.real) == np.sign(dzs)   # Ez likewise
        assert a.imag == 0.0 and c.imag == 0.0   # real rho -> real Ex, Ez
        assert b.real == 0.0 and b.imag < 0.0    # Ey = -2 i q (positive) K0


# ===========================================================================
# 5. ABSOLUTE ENERGY CONSERVATION: dipole power in == Ohmic loss out
# ===========================================================================
#: One configuration of the energy-conservation check: (q, E_ev, W, L, r0, p).
_ENERGY_CASES = [
    (0.05, 0.30, 100.0, 301, (13.0, 20.0), (1.0, 0.0, 0.0)),
    (0.05, 0.30, 100.0, 301, (13.0, 20.0), (0.0, 0.0, 1.0)),
    (-0.04, 0.45, 100.0, 201, (-70.0, 8.0), (1.0, 1.0j, 0.0)),
    (0.12, 0.60, 60.0, 251, (5.0, 4.0), (0.3, -0.2j, 0.7)),
    (0.02, 0.25, 150.0, 181, (300.0, 50.0), (1.0, 0.0, 1.0)),
    (0.08, 0.50, 80.0, 401, (0.0, 1.5), (0.0, 1.0, 0.0)),
]


@pytest.mark.parametrize("q, E_ev, W, L, r0, p", _ENERGY_CASES)
def test_energy_conservation_dipole_vs_ohmic_loss(
    q: float, E_ev: float, W: float, L: int, r0, p
) -> None:
    """``(omega/2) Im(conj(p) . E_ind(r0))`` equals the Ohmic loss in the ribbon.

    The absolute check with no adjustable constant anywhere.  Derivation, in the
    library's own discrete variables (``eps_a = eps_b = 1`` so that
    ``rho_lib = rho_phys``):

    * the ribbon dissipates, per unit length in ``y``,
      ``P = (1/2)[Re sigma_perp int |Ex|^2 dx + Re sigma_par int |Ey|^2 dx]``.
      With ``Ex`` on the faces and ``Ey = -i q phi`` on the nodes, the two
      integrals are exactly the quadratic forms of ``operators.build_D``'s own
      two pieces: ``int|Ex|^2 dx = -(h/W) phi^dag A phi`` with
      ``A = build_D(.., 1, 0)``, and ``int|Ey|^2 dx = -(h/W) phi^dag B phi`` with
      ``B = build_D(.., 0, 1)``.  Nothing is approximated: these ARE the
      operator's pieces.
    * the power the dipole delivers is ``(omega/2) Im(conj(p) . E_ind(r0))``,
      with ``E_ind`` from :func:`observables.induced_field_q`.

    The two agree only if the cell weight in ``induced_field_q`` matches the one
    in ``build_V`` (through which ``phi`` was solved) AND the overall
    normalisation ``2 K_0`` / ``2|q| K_1`` is right -- i.e. this is the test that
    the absolute scale, not just the shape, is correct.  The trapezoidal form
    misses by ~1%.
    """
    rib = solver.Ribbon(W_nm=W, x0_nm=0.0, z0_nm=0.0, EF_ev=0.5, gamma_ev=0.02, L=L)
    sys_ = solver.RibbonSystem(rib, eps_a=1.0, eps_b=1.0)
    x, h = sys_.x_nodes[0], sys_.h[0]

    phi_ext = sources.phi_ext_dipole(q, x, 0.0, (r0[0], 0.0, r0[1]), p, eps_a=1.0)
    sol = sys_.solve_linear(q, E_ev, phi_ext)
    phi, rho = sol["phi"][0], sol["rho"][0]

    # --- out: Ohmic dissipation, from build_D's own quadratic forms ---
    A = operators.build_D(q, W, h, L, 1.0, 0.0)
    B = operators.build_D(q, W, h, L, 0.0, 1.0)
    sigma = conductivity.graphene_drude(E_ev, rib.EF_ev, rib.gamma_ev)
    int_Ex2 = -(h / W) * float(np.vdot(phi, A @ phi).real)
    int_Ey2 = -(h / W) * float(np.vdot(phi, B @ phi).real)
    assert int_Ex2 > 0.0 and int_Ey2 > 0.0
    p_ohmic = 0.5 * sigma.real * (int_Ex2 + int_Ey2)

    # --- in: work done on the dipole by the induced field ---
    omega = units.omega_of(E_ev)
    pv = np.asarray(p, dtype=np.complex128)
    e_ind = np.asarray(observables.induced_field_q(q, x, rho, r0))
    p_dipole = 0.5 * omega * float(np.vdot(pv, e_ind).imag)

    e_trapz = _E_trapz(q, x, rho, r0)
    p_trapz = 0.5 * omega * float(np.vdot(pv, e_trapz).imag)

    print(
        f"\n  q={q} E={E_ev} L={L} p={p}\n"
        f"    P_ohmic  = {p_ohmic:.12e}\n"
        f"    P_dipole = {p_dipole:.12e}   ratio {p_dipole / p_ohmic:.12f}\n"
        f"    (trapz)  = {p_trapz:.12e}   ratio {p_trapz / p_ohmic:.12f}"
    )
    assert p_ohmic > 0.0
    assert p_dipole == pytest.approx(p_ohmic, rel=1e-10)
    # The superseded quadrature breaks the balance.  By how much depends on the
    # geometry -- 1.0% with the dipole far from a 100 nm ribbon, 5.5e-5 with it
    # 4 nm above a 60 nm one -- so the assertion is only that it misses by orders
    # of magnitude more than the cell-weighted form's 1e-10.
    assert abs(p_trapz / p_ohmic - 1.0) > 1e-8


def test_energy_conservation_also_closes_through_reciprocity() -> None:
    """``int phi_ext* rho dx = -(conj(p) . E_ind)``: the same cell weight both ways.

    Reciprocity of the Coulomb kernel, independent of any solve: the dipole's
    potential on the nodes (``sources.phi_ext_dipole``) paired with rho under the
    uniform cell measure must reproduce minus the induced field at the dipole.
    It isolates the quadrature from the energy balance above -- with ``np.trapz``
    on one side only, this fails.
    """
    q, W, L = 0.05, 100.0, 301
    r0, p = (13.0, 20.0), (0.4, 0.3j, -0.9)
    x, _, dx = _nodes(W, L)
    rho = _eigenmode_rho(q, W, L)[1]

    phi_ext = sources.phi_ext_dipole(q, x, 0.0, (r0[0], 0.0, r0[1]), p, eps_a=1.0)
    lhs = dx * np.vdot(phi_ext, rho)
    pv = np.asarray(p, dtype=np.complex128)
    rhs = -np.vdot(pv, np.asarray(observables.induced_field_q(q, x, rho, r0)))
    trapz_lhs = np.trapz(np.conj(phi_ext) * rho, x)
    print(f"\n  lhs = {lhs:.12e}\n  rhs = {rhs:.12e}\n  ratio {lhs / rhs:.12f}"
          f"\n  (trapz lhs ratio {trapz_lhs / rhs:.6f})")
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert abs(abs(trapz_lhs / rhs) - 1.0) > 1e-3


# ===========================================================================
# 6. ldos_q: sign, formula, and the absolute scale it inherits
# ===========================================================================
def test_ldos_q_is_positive_and_equals_the_dissipated_power() -> None:
    """``D(q) > 0`` for a passive ribbon, and ``(omega/2) 2 pi^2 |p|^2 D = P_ohmic``.

    ``ldos_q`` is ``Im(conj(p) . E_q)/(2 pi^2 |p|^2)``, so the energy balance
    above fixes its absolute value too; this states that explicitly, which is the
    claim the removed "uncertain by -1/(2 pi)" caveat used to deny.
    """
    q, E_ev, W, L = 0.05, 0.30, 100.0, 301
    r0, p = (13.0, 20.0), (1.0, 0.0, 0.0)
    rib = solver.Ribbon(W_nm=W, EF_ev=0.5, gamma_ev=0.02, L=L)
    sys_ = solver.RibbonSystem(rib, eps_a=1.0, eps_b=1.0)
    x, h = sys_.x_nodes[0], sys_.h[0]
    phi_ext = sources.phi_ext_dipole(q, x, 0.0, (r0[0], 0.0, r0[1]), p, eps_a=1.0)
    sol = sys_.solve_linear(q, E_ev, phi_ext)

    d = observables.ldos_q(q, x, sol["rho"][0], r0, p)
    assert d > 0.0, f"LDOS of a passive ribbon must be positive, got {d!r}"

    A = operators.build_D(q, W, h, L, 1.0, 0.0)
    B = operators.build_D(q, W, h, L, 0.0, 1.0)
    sigma = conductivity.graphene_drude(E_ev, rib.EF_ev, rib.gamma_ev)
    phi = sol["phi"][0]
    p_ohmic = 0.5 * sigma.real * (
        -(h / W) * float(np.vdot(phi, A @ phi).real)
        - (h / W) * float(np.vdot(phi, B @ phi).real)
    )
    p2 = float(np.vdot(np.asarray(p, dtype=complex), np.asarray(p, dtype=complex)).real)
    recovered = 0.5 * units.omega_of(E_ev) * 2.0 * np.pi**2 * p2 * d
    print(f"\n  D(q) = {d:.8e}   P_ohmic = {p_ohmic:.8e}   "
          f"ratio {recovered / p_ohmic:.12f}")
    assert recovered == pytest.approx(p_ohmic, rel=1e-10)


def test_ldos_q_forwards_dx_nm_and_matches_the_field() -> None:
    """``ldos_q`` is the documented reduction of ``induced_field_q``, dx_nm included."""
    q, r0 = 0.08, (1.5, 5.0)
    x = np.linspace(-5.0, 5.0, 301)
    rho = (1.0 + 0.4j) * np.cos(0.3 * x)
    for kw in ({}, {"dx_nm": 0.25}):
        e = np.asarray(observables.induced_field_q(q, x, rho, r0, **kw))
        for p in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1j, 0.0), (0.3, -0.2j, 0.7)):
            pv = np.asarray(p, dtype=complex)
            ref = np.vdot(pv, e).imag / (2.0 * np.pi**2 * float(np.vdot(pv, pv).real))
            assert observables.ldos_q(q, x, rho, r0, p, **kw) == pytest.approx(
                ref, rel=1e-13
            )


# ===========================================================================
# 7. the cell-width plumbing: linearity, blocks, dx_nm, and its validation
# ===========================================================================
def test_field_is_exactly_linear_in_rho() -> None:
    """Linearity, which ``dispersion._field_map`` relies on to build ``G`` per node."""
    q, r0 = 0.07, (4.0, 3.0)
    x = np.linspace(-20.0, 20.0, 201)
    rng = np.random.default_rng(20261103)
    a = rng.standard_normal(201) + 1j * rng.standard_normal(201)
    b = rng.standard_normal(201) + 1j * rng.standard_normal(201)
    ea = np.asarray(observables.induced_field_q(q, x, a, r0))
    eb = np.asarray(observables.induced_field_q(q, x, b, r0))
    both = np.asarray(observables.induced_field_q(q, x, 2.5 * a - 1.5j * b, r0))
    assert np.allclose(both, 2.5 * ea - 1.5j * eb, rtol=1e-14, atol=0.0)
    # and the basis-vector route, which is what _field_map actually does
    G = np.empty((3, x.size), dtype=np.complex128)
    e = np.zeros(x.size)
    for l in range(x.size):
        e[l] = 1.0
        G[:, l] = observables.induced_field_q(q, x, e, r0)
        e[l] = 0.0
    assert np.allclose(G @ a, ea, rtol=1e-13, atol=0.0)


def test_per_block_cell_widths_are_used_independently() -> None:
    """Two ribbons of different width each get their own ``dx = h_j W_j``."""
    q, r0 = 0.05, (10.0, 6.0)
    xa, _, dxa = _nodes(40.0, 81, x0_nm=-60.0)
    xb, _, dxb = _nodes(100.0, 51, x0_nm=+60.0)
    assert not math.isclose(dxa, dxb)
    ra = np.ones(81) * (0.5 + 0.1j)
    rb = np.ones(51) * (0.2 - 0.3j)
    both = np.asarray(observables.induced_field_q(q, [xa, xb], [ra, rb], r0))
    sep = np.asarray(observables.induced_field_q(q, xa, ra, r0)) + np.asarray(
        observables.induced_field_q(q, xb, rb, r0)
    )
    assert np.allclose(both, sep, rtol=1e-14, atol=0.0)
    # explicit dx_nm, one per block, reproduces the inferred widths
    exp = np.asarray(
        observables.induced_field_q(q, [xa, xb], [ra, rb], r0, dx_nm=[dxa, dxb])
    )
    assert np.allclose(exp, both, rtol=1e-15, atol=0.0)
    # a single scalar applies to both and is NOT the same answer
    one = np.asarray(
        observables.induced_field_q(q, [xa, xb], [ra, rb], r0, dx_nm=dxa)
    )
    assert not np.allclose(one, both, rtol=1e-6, atol=0.0)


def test_dx_nm_scales_the_field_linearly() -> None:
    """The cell width is a plain prefactor, so doubling it doubles the field."""
    q, r0 = 0.05, (10.0, 6.0)
    x, _, dx = _nodes(50.0, 101)
    rho = np.cos(0.2 * x) + 0.3j
    base = np.asarray(observables.induced_field_q(q, x, rho, r0, dx_nm=dx))
    twice = np.asarray(observables.induced_field_q(q, x, rho, r0, dx_nm=2.0 * dx))
    assert np.allclose(twice, 2.0 * base, rtol=1e-14, atol=0.0)
    inferred = np.asarray(observables.induced_field_q(q, x, rho, r0))
    assert np.allclose(inferred, base, rtol=1e-15, atol=0.0)


def test_single_node_block_needs_an_explicit_dx() -> None:
    """A bare point charge has no inferable spacing; ``dx_nm=1.0`` makes it a unit charge."""
    q, r0 = 0.05, (12.0, 4.0)
    with pytest.raises(ValueError, match="cannot be inferred"):
        observables.induced_field_q(q, np.array([0.0]), np.array([1.0]), r0)
    ex, ey, ez = observables.induced_field_q(
        q, np.array([0.0]), np.array([1.0]), r0, dx_nm=1.0
    )
    s = math.hypot(r0[0], r0[1])
    assert ex == pytest.approx(2.0 * q * r0[0] / s * k1(q * s), rel=1e-13)
    assert ey == pytest.approx(-2.0j * q * k0(q * s), rel=1e-13)
    assert ez == pytest.approx(2.0 * q * r0[1] / s * k1(q * s), rel=1e-13)


def test_non_uniform_grid_is_rejected_not_silently_mis_weighted() -> None:
    """A uniform cell weight is wrong on a non-uniform grid, so it raises."""
    q, r0 = 0.05, (12.0, 4.0)
    x = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    rho = np.ones(5)
    with pytest.raises(ValueError, match="uniform grid"):
        observables.induced_field_q(q, x, rho, r0)
    # ... but an explicit dx_nm is the caller taking responsibility
    out = observables.induced_field_q(q, x, rho, r0, dx_nm=1.25)
    assert np.all(np.isfinite(np.asarray(out)))
    # decreasing grids are rejected too (the spacing sign would flip the field)
    with pytest.raises(ValueError, match="uniform grid"):
        observables.induced_field_q(q, x[::-1], rho, r0)


def test_dx_nm_validation() -> None:
    q, r0 = 0.05, (12.0, 4.0)
    x, _, _ = _nodes(50.0, 51)
    rho = np.ones(51)
    with pytest.raises(ValueError, match="must be positive"):
        observables.induced_field_q(q, x, rho, r0, dx_nm=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        observables.induced_field_q(q, x, rho, r0, dx_nm=-1.0)
    with pytest.raises(ValueError, match="ribbon block"):
        observables.induced_field_q(q, x, rho, r0, dx_nm=[1.0, 2.0])


def test_parities_in_q_survive_the_fix() -> None:
    """Ex, Ez even in q; Ey odd.  Unchanged by the quadrature, and pinned again here."""
    q, r0 = 0.09, (0.5, 3.0)
    x = np.linspace(-5.0, 5.0, 201)
    rho = (0.3 + 0.2j) * np.sin(x)
    plus = observables.induced_field_q(+q, x, rho, r0)
    minus = observables.induced_field_q(-q, x, rho, r0)
    assert minus[0] == pytest.approx(plus[0], rel=1e-15, abs=1e-18)
    assert minus[2] == pytest.approx(plus[2], rel=1e-15, abs=1e-18)
    assert minus[1] == pytest.approx(-plus[1], rel=1e-15, abs=1e-18)


def test_singular_observation_point_still_raises() -> None:
    x = np.linspace(-1.0, 1.0, 11)
    with pytest.raises(ValueError, match="coincides"):
        observables.induced_field_q(0.1, x, np.ones_like(x), (0.0, 0.0))


# ===========================================================================
# 8. dissymmetry: the q-grid trapz is the RIGHT rule and was left alone
# ===========================================================================
def test_dissymmetry_still_uses_trapz_on_the_q_grid() -> None:
    """The q quadrature is trapezoidal, as it should be: a Gaussian pins the value.

    Guards against someone "fixing" ``dissymmetry`` the way ``induced_field_q``
    was fixed.  A uniform-cell rule over the q grid would change the three
    integrals by ``dq/2 * (D(q_min) + D(q_max))`` each; with a spectrum that does
    NOT vanish at the grid ends, that is a measurable difference, and the
    trapezoidal value is the one that matches the analytic answer.
    """
    # D = exp(-q^2/w) (1 + c q), truncated where it is still clearly non-zero
    w, c = 1.2, 0.5
    qs = np.linspace(-1.0, 1.0, 20001)
    d = np.exp(-(qs**2) / w) * (1.0 + c * qs)
    assert d[0] > 0.1 and d[-1] > 0.1, "grid must not vanish at the ends"

    got = observables.dissymmetry(qs, d)
    # analytic on the SAME finite interval, to machine precision
    f = lambda t: math.exp(-(t**2) / w) * (1.0 + c * t)  # noqa: E731
    i_pos = quad(f, 0.0, 1.0, epsabs=1e-14)[0]
    i_neg = quad(f, -1.0, 0.0, epsabs=1e-14)[0]
    expect = 2.0 * (i_pos - i_neg) / (i_pos + i_neg)

    # the uniform-cell alternative, for contrast
    dq = float(qs[1] - qs[0])
    pos, neg = qs >= 0.0, qs <= 0.0
    cell = 2.0 * (dq * d[pos].sum() - dq * d[neg].sum()) / (dq * d.sum())
    print(f"\n  trapz g      = {got:.10f}\n  analytic     = {expect:.10f}"
          f"\n  uniform-cell = {cell:.10f}")
    assert got == pytest.approx(expect, rel=1e-8)
    assert abs(cell - expect) > 1e-6, (
        "the two rules are indistinguishable on this grid, so the test cannot "
        "tell whether dissymmetry still uses trapz"
    )


def test_dissymmetry_contract_unchanged() -> None:
    """The documented edge cases, re-pinned here so this file stands on its own."""
    qs = np.linspace(-1.0, 1.0, 2001)
    even = np.exp(-(qs**2) / 0.08)
    assert observables.dissymmetry(qs, even) == pytest.approx(0.0, abs=1e-14)
    assert observables.dissymmetry(qs, np.where(qs > 0, even, 0.0)) == pytest.approx(
        2.0, rel=1e-12
    )
    # a grid with no node at q = 0 still splits consistently
    qs2 = np.linspace(-1.0, 1.0, 2000)
    assert 0.0 not in qs2
    assert observables.dissymmetry(qs2, np.exp(-(qs2**2) / 0.08)) == pytest.approx(
        0.0, abs=1e-12
    )
    with pytest.raises(ValueError, match="straddle"):
        observables.dissymmetry(np.linspace(0.1, 1.0, 11), np.ones(11))
    with pytest.raises(ValueError, match="strictly increasing"):
        observables.dissymmetry(np.array([-1.0, 0.0, 0.0, 1.0]), np.ones(4))
    with pytest.raises(ValueError, match="vanishes"):
        observables.dissymmetry(np.array([-1.0, 0.0, 1.0]), np.zeros(3))


# ===========================================================================
# 9. the docstring must not still claim the normalisation is unresolved
# ===========================================================================
def test_module_docstring_records_the_resolution_not_the_old_concern() -> None:
    """A stale "UNRESOLVED" note is worse than none; this keeps it gone."""
    doc = observables.__doc__ or ""
    assert "UNRESOLVED" not in doc
    assert "RESOLVED" in doc
    assert "NOT uncertain by -1/(2 pi)" in doc
    assert "Eq. (21)" in doc          # where the 1/(2 pi) does belong
    assert "UNIFORM CELL WEIGHTS" in doc
    # the exact phrases of the superseded concern block, which must be gone
    for stale in (
        "NORMALISATION DISCREPANCY",
        "cannot both be",
        "absolute-normalisation caveat",
        "Integrals are trapezoidal",
    ):
        assert stale not in doc, f"stale concern text still in the module docstring: {stale!r}"
    for fn in (observables.induced_field_q, observables.ldos_q):
        fdoc = fn.__doc__ or ""
        assert "unresolved" not in fdoc.lower()
        assert "caveat" not in fdoc.lower()
        assert "trapezoidal" not in fdoc or "replaced" in fdoc
