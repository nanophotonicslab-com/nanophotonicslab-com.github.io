"""Tests for ``electrostatics_fd.operators``.

Every assertion is against an independent reference: ``scipy.special.modstruve``
for the Struve series, ``scipy.integrate.quad`` for the exact Coulomb branches,
the closed-form Neumann-Laplacian spectrum for the differential operator, and a
literal transcription of the specified stencils for ``build_D``.
"""

from __future__ import annotations

import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval
from scipy.integrate import IntegrationWarning, quad
from scipy.special import k0, kve, modstruve

try:  # the high-precision reference for the Struve combination; optional
    import mpmath as mp
except ImportError:  # pragma: no cover - exercised only without mpmath
    mp = None

# The package has no __init__.py (owned by no agent), so make the module
# directory importable without touching any file outside this one.
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import operators as ops  # noqa: E402

H149 = 1.0 / 149.0  # the cell width named in the module specification


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _quad_k0_cell(big_q: float, h: float, theta: float) -> float:
    """``2 * int_{theta-h/2}^{theta+h/2} K0(Q|t|) dt`` by adaptive quadrature."""
    a, b = theta - 0.5 * h, theta + 0.5 * h

    def kernel(t: float) -> float:
        return 2.0 * float(k0(big_q * abs(t)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        if a < 0.0 < b:  # split the logarithmic singularity at the origin
            lo = quad(kernel, a, 0.0, epsabs=1e-16, epsrel=1e-13)[0]
            hi = quad(kernel, 0.0, b, epsabs=1e-16, epsrel=1e-13)[0]
            return lo + hi
        return quad(kernel, a, b, epsabs=1e-16, epsrel=1e-13)[0]


def _quad_log_cell(h: float, theta: float, ztil: float) -> float:
    """``int_{theta-h/2}^{theta+h/2} -2*log(sqrt(t**2 + ztil**2)) dt``."""
    a, b = theta - 0.5 * h, theta + 0.5 * h

    def kernel(t: float) -> float:
        return -np.log(t * t + ztil * ztil)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        if a < 0.0 < b and ztil == 0.0:
            lo = quad(kernel, a, 0.0, epsabs=1e-16, epsrel=1e-13)[0]
            hi = quad(kernel, 0.0, b, epsabs=1e-16, epsrel=1e-13)[0]
            return lo + hi
        return quad(kernel, a, b, epsabs=1e-16, epsrel=1e-13)[0]


@lru_cache(maxsize=None)
def _fc_reference(x: float) -> float:
    """``Fc(x) = (2/pi)*int_x^inf K0(t) dt`` from an independent high-precision
    evaluation.

    With mpmath: the identity ``Fc = 1 - x*(K0*L_{-1} + K1*L_0)`` evaluated at
    ``45 + 0.44*x`` decimal digits, which is enough working precision for the
    ``1 - (1 - 1e-0.44x)`` cancellation to leave 45 correct digits.  This was
    cross-checked against ``mp.quad`` of ``int_x^inf K0`` (agreement to all 20
    printed digits at x = 3, and to 10 digits at x = 200), so it is a genuine
    independent reference and not a restatement of the module's own algorithm:
    the module never evaluates ``L_n`` at ``x > 4``.

    Without mpmath: 80-node Gauss-Laguerre on
    ``Fc = (2/pi)*exp(-x)*int_0^inf exp(-s)*kve(0, x+s) ds``, which was
    measured at <=3e-14 relative for ``x >= 1`` and ~1e-10 at ``x = 0.5``.
    """
    if mp is not None:
        with mp.workdps(int(0.44 * float(x)) + 45):
            xx = mp.mpf(float(x))
            f = xx * (
                mp.besselk(0, xx) * mp.struvel(-1, xx)
                + mp.besselk(1, xx) * mp.struvel(0, xx)
            )
            return float(1 - f)
    s, w = np.polynomial.laguerre.laggauss(80)
    return float((2.0 / np.pi) * np.exp(-float(x)) * np.sum(w * kve(0, float(x) + s)))


def _cell_reference(big_q: float, h: float, theta: float):
    """``2*int_{theta-h/2}^{theta+h/2} K0(Q|t|) dt`` in high precision.

    Uses the mpmath ``Fc`` identity rather than ``mp.quad``: the quadrature
    form needs ~300 working digits at these arguments and takes ~100 s for a
    handful of points, while the identity is exact and instant.  The identity
    itself is cross-checked against ``mp.quad`` by
    :func:`test_cell_reference_agrees_with_quadrature`, so the chain back to a
    genuinely independent evaluation is unbroken.
    """
    a, b = abs(theta) - 0.5 * h, abs(theta) + 0.5 * h
    lo, hi = min(abs(a), abs(b)), max(abs(a), abs(b))
    straddle = (theta - 0.5 * h) < 0.0 < (theta + 0.5 * h)
    with mp.workdps(int(0.44 * big_q * hi) + 45):
        q = mp.mpf(big_q)

        def fc(x):
            xx = q * mp.mpf(x)
            if xx == 0:
                return mp.mpf(1)
            return 1 - xx * (
                mp.besselk(0, xx) * mp.struvel(-1, xx)
                + mp.besselk(1, xx) * mp.struvel(0, xx)
            )

        if straddle:
            return +((mp.pi / q) * (2 - fc(lo) - fc(hi)))
        return +((mp.pi / q) * (fc(lo) - fc(hi)))


def _cheb_panel(x, i):
    """One Chebyshev panel of ``Fc`` evaluated on its own, for overlap tests."""
    x = np.asarray(x, dtype=np.float64)
    t0, dt = ops._FC_CHEB_MAP[i]
    s = np.polynomial.chebyshev.chebval((1.0 / x - t0) / dt, ops._FC_CHEB_C[i])
    return np.sqrt(2.0 / (np.pi * x)) * np.exp(-x) * s


def _asym_branch(x):
    """The asymptotic branch of ``Fc`` evaluated on its own, for overlap tests."""
    x = np.asarray(x, float)
    return (
        np.sqrt(2.0 / (np.pi * x))
        * np.exp(-x)
        * polyval(1.0 / x, ops._FC_ASYM)
    )


def _one_minus_f_branch(x):
    """The ``1 - F`` branch of ``Fc`` evaluated on its own, for overlap tests."""
    return 1.0 - ops._f_direct(np.asarray(x, float))


def _ref_build_D(q, W, h, L, eta_perp, eta_par, f=None):
    """Literal transcription of the specified ``set_D`` stencils (loops on purpose)."""
    big_q = q * W
    D = np.zeros((L, L), dtype=np.complex128)
    if f is None:
        for l in range(1, L - 1):
            D[l, l - 1] = eta_perp / h**2
            D[l, l] = -2.0 * eta_perp / h**2 - eta_par * big_q**2
            D[l, l + 1] = eta_perp / h**2
        D[0, 0] = -eta_perp / h**2 - eta_par * big_q**2
        D[0, 1] = eta_perp / h**2
        D[L - 1, L - 1] = -eta_perp / h**2 - eta_par * big_q**2
        D[L - 1, L - 2] = eta_perp / h**2
    else:
        for l in range(1, L - 1):
            D[l, l - 1] = (f[l - 1] + f[l]) / (2.0 * h**2) * eta_perp
            D[l, l] = (
                -(f[l - 1] + 2.0 * f[l] + f[l + 1]) / (2.0 * h**2) * eta_perp
                - eta_par * big_q**2 * f[l]
            )
            D[l, l + 1] = (f[l + 1] + f[l]) / (2.0 * h**2) * eta_perp
        # Boundary rows: the zero-ghost-flux extension documented in operators.py
        # (the specification gives interior rows only for the f-case).
        D[0, 0] = -(f[0] + f[1]) / (2.0 * h**2) * eta_perp - eta_par * big_q**2 * f[0]
        D[0, 1] = (f[0] + f[1]) / (2.0 * h**2) * eta_perp
        D[L - 1, L - 1] = (
            -(f[L - 1] + f[L - 2]) / (2.0 * h**2) * eta_perp
            - eta_par * big_q**2 * f[L - 1]
        )
        D[L - 1, L - 2] = (f[L - 1] + f[L - 2]) / (2.0 * h**2) * eta_perp
    return D


# ===========================================================================
# struve_L
# ===========================================================================


@pytest.mark.parametrize("order", [0, -1])
@pytest.mark.parametrize("x", [0.0, 1e-8, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0])
def test_struve_L_matches_scipy_modstruve(order, x):
    """30-term series reproduces scipy's L_n to double precision for x <~ 4."""
    got = float(ops.struve_L(order, x))
    ref = float(modstruve(order, x))
    assert got == pytest.approx(ref, rel=1e-14, abs=1e-300)


def test_struve_L_known_series_leading_terms():
    """Hand-checked leading coefficients: L_0 ~ 2x/pi, L_-1(0) = 2/pi."""
    assert float(ops.struve_L(-1, 0.0)) == pytest.approx(2.0 / np.pi, rel=1e-15)
    x = 1e-5
    # L_0(x) = (2/pi)*(x + x^3/9 + ...)
    assert float(ops.struve_L(0, x)) == pytest.approx(
        (2.0 / np.pi) * (x + x**3 / 9.0), rel=1e-14
    )
    # L_-1(x) = (2/pi)*(1 + x^2/3 + ...)
    assert float(ops.struve_L(-1, x)) == pytest.approx(
        (2.0 / np.pi) * (1.0 + x**2 / 3.0), rel=1e-14
    )


def test_struve_L_vectorises_and_keeps_shape():
    x = np.linspace(0.0, 3.0, 17).reshape(17, 1)
    out = ops.struve_L(0, x)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, modstruve(0, x), rtol=1e-14, atol=1e-300)
    assert np.asarray(ops.struve_L(0, 1.0)).shape == ()


def test_struve_L_moderate_x_still_good_to_1e9():
    """Accuracy holds to ~1e-9 out to x = 8, the top of the useful range."""
    x = np.linspace(4.0, 8.0, 25)
    for order in (0, -1):
        rel = np.abs(ops.struve_L(order, x) - modstruve(order, x)) / np.abs(
            modstruve(order, x)
        )
        assert rel.max() < 1e-9, f"order {order}: max rel {rel.max():.3e}"


def test_bare_30_term_series_still_degrades_which_is_why_the_branch_exists():
    """Pin the truncation failure of the *bare series*, not of ``struve_L``.

    This test used to assert that ``struve_L`` itself degrades (45% at x = 30).
    It no longer does -- that was the defect, and the large-argument branch is
    the fix.  What is pinned here is the *reason the branch has to exist*: the
    30-term polynomial of the reference MATLAB, evaluated on its own, is still
    useless past ~10.  If that ever stopped being true the branch could be
    dropped; it has not.
    """
    bare_30 = polyval(30.0, ops._STRUVE_P_SCALED[0])
    assert float(bare_30) < 0.6 * float(modstruve(0, 30.0))
    bare_20 = polyval(20.0, ops._STRUVE_P_SCALED[0])
    rel20 = abs(float(bare_20) - float(modstruve(0, 20.0))) / float(
        modstruve(0, 20.0)
    )
    assert rel20 > 1e-3, f"series unexpectedly accurate at x=20 (rel {rel20:.3e})"


def test_struve_L_small_argument_is_bit_for_bit_the_unchanged_series():
    """The large-argument branch must not perturb the validated small-x region.

    ``|x| <= _STRUVE_SERIES_MAX`` must still be the bare 30-term polynomial,
    exactly -- not to 1e-15, bit for bit -- so that every number this package
    has ever been validated against is untouched.
    """
    assert ops._STRUVE_SERIES_MAX == 4.0
    x = np.concatenate(
        [
            np.linspace(-4.0, 4.0, 401),
            np.array([0.0, 1e-300, 1e-8, 4.0, -4.0, np.nextafter(4.0, 0.0)]),
        ]
    )
    for order in (0, -1):
        got = ops.struve_L(order, x)
        bare = polyval(x, ops._STRUVE_P_SCALED[order])
        assert np.array_equal(got, bare), (
            f"order {order}: struve_L differs from the bare series inside the "
            f"series range by up to {np.max(np.abs(got - bare)):.3e}"
        )


def test_struve_L_large_argument_branch_is_accurate_everywhere():
    """Past the series range ``struve_L`` is accurate, not merely finite.

    Reference: ``mpmath.struvel`` when available, else ``scipy`` (which is what
    the branch delegates to, so the scipy leg of this test only checks the
    plumbing -- the mpmath leg is the one that validates scipy itself, and it
    is the leg that runs in this environment).
    """
    xs = np.concatenate([
        np.linspace(4.0, 30.0, 53),   # includes x ~ 14, scipy's own worst point
        np.array([50.0, 100.0, 200.0, 400.0, 700.0, 710.0]),
    ])
    worst, at, which = 0.0, 0.0, 0
    for order in (0, -1):
        for x in xs:
            got = float(ops.struve_L(order, x))
            if mp is not None:
                with mp.workdps(40):
                    ref = float(mp.struvel(order, x))
            else:
                ref = float(modstruve(order, x))
            rel = abs(got - ref) / abs(ref)
            if rel > worst:
                worst, at, which = rel, x, order
    print(f"\nstruve_L large-argument branch: worst rel err {worst:.3e} at "
          f"x = {at:.4g}, n = {which} "
          f"(reference: {'mpmath' if mp is not None else 'scipy'})")
    # Measured 7.4e-13 at x ~ 14, where scipy switches its own algorithm, and
    # 3e-16 or better everywhere else.  The threshold is set just above that
    # measured worst case; it is scipy's accuracy, not this module's, and it
    # does not reach the Coulomb kernel (Fc never calls modstruve).
    assert worst < 2e-12, f"large-argument branch is off by {worst:.3e}"


def test_struve_L_parity_extends_the_branch_to_negative_arguments():
    """``L_0`` odd, ``L_{-1}`` even -- the series has that parity, so the
    large-argument branch must keep it."""
    x = np.array([5.0, 12.0, 60.0, 300.0])
    np.testing.assert_allclose(
        ops.struve_L(0, -x), -ops.struve_L(0, x), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        ops.struve_L(-1, -x), ops.struve_L(-1, x), rtol=0.0, atol=0.0
    )


def test_struve_L_overflows_rather_than_lying_past_the_double_range():
    """``L_n`` genuinely exceeds float64 past ~711; ``inf`` is the honest answer.

    The old series returned a finite, wrong number there.  ``NaN`` would be
    unacceptable (it poisons a whole block); ``inf`` is not reached by any
    kernel path, because branch 3 goes through ``Fc``.
    """
    for order in (0, -1):
        assert np.isfinite(ops.struve_L(order, 710.0))
        big = ops.struve_L(order, np.array([720.0, 1e3, 1e4]))
        assert np.all(np.isinf(big)) and np.all(big > 0.0)
        assert not np.any(np.isnan(big))


@pytest.mark.parametrize("bad", [1, -2, 2, 5])
def test_struve_L_rejects_unsupported_order(bad):
    with pytest.raises(ValueError, match="n in"):
        ops.struve_L(bad, 1.0)


# ===========================================================================
# F(x) = (2/pi)*int_0^x K0 and Fc = 1 - F: the large-argument branch
# ===========================================================================
_DECADES = [(0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 200.0)]


def test_Fc_matches_high_precision_reference():
    """``Fc`` to 1e-12 relative over 0.01 <= x <= 200, measured per decade.

    ``Fc`` is the function that carries the physics: branch 3 differences it,
    so its *relative* accuracy at large argument is exactly what decides
    whether two coplanar ribbons decouple.  Worst relative error per decade is
    printed, because the four evaluation regions have genuinely different
    accuracy and hiding that behind one number would be dishonest.
    """
    worst_all = 0.0
    lines = []
    for lo, hi in _DECADES:
        xs = np.geomspace(lo, hi, 13)
        worst, at = 0.0, lo
        for x in xs:
            got = float(ops._struve_Fc(float(x)))
            ref = _fc_reference(float(x))
            rel = abs(got - ref) / abs(ref)
            if rel > worst:
                worst, at = rel, x
        lines.append(f"  [{lo:>6g}, {hi:>5g}]  worst rel {worst:.2e} at x = {at:.4g}")
        worst_all = max(worst_all, worst)
    print("\nFc relative error per decade (reference: "
          f"{'mpmath' if mp is not None else 'Gauss-Laguerre quadrature'}):")
    print("\n".join(lines))
    # Measured worst is 3.9e-14, in the ``1 - F`` region near x = 3.9 where
    # that subtraction has thrown away 2.1 digits; everywhere above 4 it is
    # 3.2e-16 or better.  The requirement was 1e-10 relative out to x = 200.
    tol = 1e-12 if mp is not None else 1e-9
    assert worst_all < tol, f"Fc worst relative error {worst_all:.3e}"


def test_F_matches_high_precision_reference():
    """``F`` itself to 1e-10 relative out to x = 200 (the stated requirement).

    ``F -> 1``, so this is the easy half; it is asserted anyway because the
    old implementation failed it outright (45% at x = 30) and because a sign
    error in the ``1 - Fc`` plumbing would show up here and nowhere else.
    """
    worst = 0.0
    for lo, hi in _DECADES:
        for x in np.geomspace(lo, hi, 13):
            got = float(ops._struve_F(float(x)))
            ref = 1.0 - _fc_reference(float(x))
            worst = max(worst, abs(got - ref) / abs(ref))
    print(f"\nF worst relative error over [0.01, 200]: {worst:.3e}")
    assert worst < 1e-12, f"F worst relative error {worst:.3e}"


def test_F_and_Fc_are_exactly_complementary():
    """``F + Fc == 1`` to one ulp everywhere, which is why the two difference
    forms of branch 3 cannot leave a step in the assembled ``V``."""
    x = np.concatenate([np.geomspace(1e-6, 1e4, 4000), np.array([0.0])])
    s = ops._struve_F(x) + ops._struve_Fc(x)
    worst = float(np.max(np.abs(s - 1.0)))
    print(f"\nmax |F + Fc - 1| over [0, 1e4]: {worst:.3e}")
    assert worst <= 2.3e-16


@pytest.mark.parametrize(
    "label, x_window, branch_lo, branch_hi",
    [
        ("series -> Chebyshev panel 1 at x = 4",
         (3.6, 4.4), _one_minus_f_branch, lambda x: _cheb_panel(x, 0)),
        ("Chebyshev panel 1 -> panel 2 at x = 8",
         (7.5, 8.5), lambda x: _cheb_panel(x, 0), lambda x: _cheb_panel(x, 1)),
        ("Chebyshev panel 2 -> asymptotic at x = 48",
         (44.0, 52.0), lambda x: _cheb_panel(x, 1), _asym_branch),
    ],
)
def test_branches_agree_across_every_switch_over(label, x_window, branch_lo, branch_hi):
    """Each pair of neighbouring branches agrees to <=1e-11 in an overlap window.

    The switch-over is seamless, not merely small: both representations are
    evaluated *at the same arguments* on both sides of the boundary (each
    Chebyshev panel is its own analytic continuation a little way outside its
    interval), so what is measured is the branch disagreement and not the
    slope of ``Fc``.  A step here would put a kink in the assembled ``V``.
    """
    x = np.linspace(*x_window, 41)
    lo = np.asarray(branch_lo(x), dtype=np.float64)
    hi = np.asarray(branch_hi(x), dtype=np.float64)
    rel = np.max(np.abs(hi - lo) / np.abs(lo))
    print(f"\n{label}: worst relative branch disagreement {rel:.3e} "
          f"over x in [{x_window[0]}, {x_window[1]}]")
    assert rel < 1e-11, f"{label}: branches disagree by {rel:.3e}"


def test_Fc_switch_over_leaves_no_step_in_the_assembled_function():
    """``Fc`` as actually dispatched is smooth across 4, 8 and 48.

    Three-point second differences straddling each boundary, scaled by the
    local value, must be no larger than the same quantity computed well inside
    a single branch -- i.e. the dispatch adds no curvature of its own.
    """
    for edge in (4.0, 8.0, 48.0):
        d = 1e-5 * edge
        at_edge = ops._struve_Fc(np.array([edge - d, edge, edge + d]))
        inside = ops._struve_Fc(np.array([0.7 * edge - d, 0.7 * edge, 0.7 * edge + d]))
        kink = abs(at_edge[0] - 2 * at_edge[1] + at_edge[2]) / at_edge[1]
        smooth = abs(inside[0] - 2 * inside[1] + inside[2]) / inside[1]
        print(f"\n  edge {edge:5g}: scaled second difference {kink:.3e} "
              f"(smooth reference {smooth:.3e})")
        assert kink < 10.0 * smooth + 1e-14, (
            f"Fc has a kink at the x = {edge} switch-over: second difference "
            f"{kink:.3e} against {smooth:.3e} inside a branch"
        )


def test_no_overflow_or_nan_anywhere_up_to_1e4():
    """``0 < x <= 1e4``: nothing overflows, nothing is NaN, bounds respected.

    This is the range branch 3 can actually be asked for -- ``Q*|theta|`` for a
    far-separated coplanar pair -- and it is where the old form produced
    ``inf*0 -> NaN`` (``K0`` underflows to 0 while ``L_n`` is already ``inf``
    past x ~ 746).
    """
    x = np.unique(np.concatenate([
        np.geomspace(1e-12, 1e4, 20001),
        np.linspace(1e-9, 1e4, 20001),
        np.array([4.0, 8.0, 48.0, 700.0, 710.0, 745.0, 746.0, 1e4]),
        np.nextafter([4.0, 8.0, 48.0], 0.0),
        np.nextafter([4.0, 8.0, 48.0], np.inf),
    ]))
    f = ops._struve_F(x)
    fc = ops._struve_Fc(x)
    assert np.all(np.isfinite(f)), "F is not finite somewhere in (0, 1e4]"
    assert np.all(np.isfinite(fc)), "Fc is not finite somewhere in (0, 1e4]"
    assert np.all(f >= 0.0) and np.all(f <= 1.0), "F left [0, 1]"
    assert np.all(fc >= 0.0) and np.all(fc <= 1.0), "Fc left [0, 1]"
    # monotone: F non-decreasing, Fc non-increasing (both to one ulp)
    assert np.min(np.diff(f)) >= -1e-16, "F is not monotone increasing"
    assert np.max(np.diff(fc)) <= 1e-16, "Fc is not monotone decreasing"
    print(f"\nsweep over {x.size} points in (0, 1e4]: F in "
          f"[{f.min():.3e}, {f.max():.17g}], Fc in [{fc.min():.3e}, {fc.max():.17g}]")
    # and the kernel built on top of them
    for big_q in (0.3, 1.0, 7.0):
        v = ops.coulomb_V_entries(big_q, H149, np.geomspace(1e-9, 1.5e3, 5000), 0.0)
        assert np.all(np.isfinite(v)) and np.all(v >= 0.0)


def test_branch3_is_accurate_at_large_argument():
    """Branch 3 against a high-precision cell integral at ``Q*|theta|`` >> 1.

    The old series gave 45% at argument 30 and nonsense beyond; the whole point
    of the fix is that the exponentially small cell integral is now right in
    relative terms.
    """
    if mp is None:
        pytest.skip("needs mpmath for a reference at these magnitudes")
    worst = 0.0
    rows = []
    for big_q, theta in ((1.0, 10.0), (1.0, 50.0), (2.0, 100.0), (5.0, 10.0),
                         (5.0, 50.0), (0.5, 120.0), (5.0, 120.0)):
        got = float(ops.coulomb_V_entries(big_q, H149, theta, 0.0))
        ref = _cell_reference(big_q, H149, theta)
        rel = abs(mp.mpf(got) - ref) / abs(ref)
        rows.append(f"  Q={big_q:4g} theta={theta:7g} (arg {big_q * theta:7g}): "
                    f"V={got:.6e}  rel {float(rel):.2e}")
        worst = max(worst, float(rel))
    print()
    print("branch 3 against the high-precision cell integral at large argument:")
    print("\n".join(rows))
    assert worst < 1e-10, f"branch 3 worst relative error {worst:.3e}"


def test_cell_reference_agrees_with_quadrature():
    """Tie the fast mpmath cell reference back to actual quadrature.

    One moderate argument, where ``mp.quad`` is affordable: this is what makes
    :func:`test_branch3_is_accurate_at_large_argument` a test against an
    independent evaluation rather than against the same identity the module
    uses (the module never evaluates ``L_n`` past x = 4, so even the identity
    is independent code -- this closes the loop anyway).
    """
    if mp is None:
        pytest.skip("needs mpmath")
    for big_q, theta in ((1.0, 3.0), (2.0, 0.0), (0.5, 20.0)):
        a, b = theta - 0.5 * H149, theta + 0.5 * H149
        with mp.workdps(60):
            if a < 0 < b:
                ref = 2 * (
                    mp.quad(lambda t: mp.besselk(0, mp.mpf(big_q) * abs(t)), [a, 0])
                    + mp.quad(lambda t: mp.besselk(0, mp.mpf(big_q) * abs(t)), [0, b])
                )
            else:
                ref = 2 * mp.quad(
                    lambda t: mp.besselk(0, mp.mpf(big_q) * t), [mp.mpf(a), mp.mpf(b)]
                )
        got = _cell_reference(big_q, H149, theta)
        rel = float(abs(got - ref) / abs(ref))
        print(f"  Q={big_q} theta={theta}: identity vs quadrature rel {rel:.3e}")
        assert rel < 1e-25


def test_coplanar_far_ribbons_cross_block_tracks_K0():
    """THE test this module could not pass: coplanar pairs now decouple.

    Two coplanar ribbons of width ``W`` whose centres are ``d`` apart couple
    through branch 3 at argument ``q*d``; the largest cross-block entry is the
    exact cell integral at the nearest node pair, i.e. ``2*h*K0(q*(d - W))`` up
    to the ``(Q*h)**2/24 = 3.1e-6`` midpoint difference between a cell integral
    and a cell midpoint.

    Measured, at ``q*d`` from 5 to 60: the old 30-term series flattened at an
    absolute 9.07e-15 -- it was already 7.5x too large at ``q*d = 30`` and
    4.7e9 times too large at ``q*d = 50``, where the true coupling is 1.9e-24 --
    while the new branch tracks ``K0`` over 20 decades.
    """
    W, L, q = 10.0, 60, 0.05
    theta = np.linspace(-0.5, 0.5, L)
    h = 1.0 / (L - 1)
    big_q = q * W

    qds = np.arange(5.0, 60.1, 5.0)
    mags, refs = [], []
    for qd in qds:
        d = qd / q
        block = ops.build_V(q, W, W, d, 0.0, 0.0, 0.0, theta, theta, h)
        mags.append(float(np.max(np.abs(block))))
        refs.append(2.0 * h * float(k0(qd - big_q)))
    mags = np.array(mags)
    refs = np.array(refs)

    print("\ncoplanar cross block vs 2*h*K0(q*(d-W)):")
    for qd, m, r in zip(qds, mags, refs):
        print(f"  q*d={qd:5g}  max|V|={m:.6e}  ref={r:.6e}  ratio={m / r:.9f}")

    # it tracks K0 -- not merely "is small"
    np.testing.assert_allclose(mags, refs, rtol=1e-4)
    # and it decays exponentially rather than flattening at a noise floor
    assert mags[-1] < 1e-27, (
        f"cross block at q*d = 60 is {mags[-1]:.3e}; a noise floor, not K0"
    )
    assert mags[0] / mags[-1] > 1e24, "cross block spans too few decades to be K0"
    slope = np.diff(np.log(mags)) / np.diff(qds)
    assert np.all(slope < -0.9), f"decay is not exponential in q*d: {slope}"
    # the whole block, not just its largest entry, is below the old noise floor
    far = ops.build_V(q, W, W, 50.0 / q, 0.0, 0.0, 0.0, theta, theta, h)
    assert float(np.max(np.abs(far))) < 1e-23


def test_branch3_tracks_2hK0_into_the_denormals_and_vanishes_with_it():
    """The large-argument limit is double precision itself, not the algorithm.

    Branch 3 keeps agreeing with ``2*h*K0(arg)`` -- which at ``Q*h << 1`` is
    what the cell integral tends to -- all the way into the denormal range, and
    reaches exactly ``0`` at the same argument ``2*h*K0`` does.  So the place
    where ``V`` becomes 0 is where its true value drops below the smallest
    denormal (4.9e-324), not where an expansion gives up.  Pinned because it is
    the only remaining "wrong answer" in the whole range and it is the right
    wrong answer.
    """
    rows = []
    for arg in (300.0, 600.0, 700.0, 710.0, 720.0, 730.0, 740.0, 760.0,
                1000.0, 1e4):
        got = float(ops.coulomb_V_entries(1.0, H149, arg, 0.0))
        ref = 2.0 * H149 * float(k0(arg))
        rows.append(f"  arg={arg:8g}  V={got:.4e}  2*h*K0={ref:.4e}")
        assert np.isfinite(got) and got >= 0.0
        if ref == 0.0:
            assert got == 0.0, f"V = {got:.3e} where 2*h*K0 has underflowed"
        else:
            assert got == pytest.approx(ref, rel=1e-6), (
                f"arg {arg}: V = {got:.6e} against 2*h*K0 = {ref:.6e}"
            )
    print("\nbranch 3 into the denormals:")
    print("\n".join(rows))


def test_remaining_small_argument_floor_is_bounded_and_documented():
    """The one accuracy floor left: ``Q*|theta| < 4`` with ``Q*h`` tiny.

    Below the switch at 4 the difference is still formed from two ``O(1)``
    numbers, so ``V`` keeps an absolute floor of a few ulps of ``pi/Q`` -- the
    floor the old implementation had everywhere.  It is harmless (nothing is
    exponentially small there, so nothing decouples wrongly) but it is real, so
    it is measured here rather than left for someone to rediscover.  Measured
    worst, at ``h = 1/149``: 3.3e-11 at ``Q = 0.1``, argument 4.
    """
    if mp is None:
        pytest.skip("needs mpmath")
    worst = {}
    for big_q in (0.1, 0.5, 1.0, 2.0, 5.0):
        w, at = 0.0, 0.0
        for theta in np.geomspace(1e-3, 4.0 / big_q, 25):
            got = float(ops.coulomb_V_entries(big_q, H149, float(theta), 0.0))
            rel = float(abs(mp.mpf(got) - _cell_reference(big_q, H149, float(theta)))
                        / abs(_cell_reference(big_q, H149, float(theta))))
            if rel > w:
                w, at = rel, theta
        worst[big_q] = (w, at)
    print("\nremaining floor below argument 4 (h = 1/149):")
    for big_q, (w, at) in worst.items():
        print(f"  Q={big_q:5g}: worst rel {w:.2e} at theta={at:.4g} "
              f"(argument {big_q * at:.3g})")
    assert max(w for w, _ in worst.values()) < 1e-10


def test_builders_are_silent_for_every_coplanar_geometry():
    """``build_V``/``build_V_toeplitz`` must not warn now that branch 3 is good.

    The ``RuntimeWarning`` about ``STRUVE_TRUSTED_MAX_ARG`` was removed rather
    than re-scoped: branch 3 is accurate at every coplanar separation, so no
    range remains untrusted to re-scope it to.
    """
    assert not hasattr(ops, "_warn_struve_range"), (
        "the Struve range warning is still installed; it now fires only for "
        "cases that are fine"
    )
    W, L = 10.0, 40
    theta = np.linspace(-0.5, 0.5, L)
    h = 1.0 / (L - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for q, x_off in ((0.5 / W, 0.0), (0.5, 100.0), (0.5, 1.0e4),
                         (50.0, 1.0e4), (0.0, 100.0)):
            ops.build_V(q, W, W, x_off, 0.0, 0.0, 0.0, theta, theta, h)
            ops.build_V_toeplitz(q, W, x_off, 0.0, theta, h)
            ops.build_V(q, W, W, x_off, 0.0, 5.0, 0.0, theta, theta, h)


def test_struve_trusted_max_arg_is_kept_as_documentation():
    """The constant survives at 8.0 -- it is referenced by sibling modules.

    Its meaning changed: it is the ceiling of the *bare 30-term series*, not a
    trust ceiling on the Coulomb block, and it is no longer the switch-over
    point (``_STRUVE_SERIES_MAX = 4`` is).  It stays in the module but out of
    ``__all__``, exactly as before.
    """
    assert ops.STRUVE_TRUSTED_MAX_ARG == 8.0
    assert "STRUVE_TRUSTED_MAX_ARG" not in ops.__all__
    assert ops._STRUVE_SERIES_MAX <= ops.STRUVE_TRUSTED_MAX_ARG


# ===========================================================================
# Coulomb branch 3 -- spec test (a)
# ===========================================================================


@pytest.mark.parametrize("big_q", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "theta", [0.0, H149 / 2.0, H149, 0.05, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, -0.37]
)
def test_branch3_matches_quadrature(big_q, theta):
    """Spec test (a): branch 3 is the exact cell integral of 2*K0(Q|t|).

    rtol 1e-8 for Q in {0.1, 0.5, 1, 2} at h = 1/149.
    """
    got = float(ops.coulomb_V_entries(big_q, H149, theta, 0.0))
    ref = _quad_k0_cell(big_q, H149, theta)
    assert got == pytest.approx(ref, rel=1e-8)


def test_branch3_row_telescopes_to_full_span_integral():
    """Contiguous cells tile exactly, so a full matrix row sums to one integral.

    sum_l' V(theta_l - theta_l') = 2*int over [theta_l-1/2-h/2, theta_l+1/2+h/2]
    of K0(Q|t|) dt. A genuine whole-row check of branch 3.
    """
    n = 101
    h = 1.0 / (n - 1)
    theta = np.linspace(-0.5, 0.5, n)
    big_q = 1.7
    sep = theta[:, None] - theta[None, :]
    rows = ops.coulomb_V_entries(big_q, h, sep, 0.0).sum(axis=1)
    for l in (0, 1, n // 3, n // 2, n - 1):
        span_lo = theta[l] - 0.5 - 0.5 * h
        span_hi = theta[l] + 0.5 + 0.5 * h
        mid = 0.5 * (span_lo + span_hi)
        ref = _quad_k0_cell(big_q, span_hi - span_lo, mid)
        assert float(rows[l]) == pytest.approx(ref, rel=1e-11)


def test_branch3_handles_zero_endpoint():
    """F(0) = 0: theta = +-h/2 puts a cell endpoint exactly on the pole."""
    for theta in (H149 / 2.0, -H149 / 2.0):
        val = float(ops.coulomb_V_entries(1.0, H149, theta, 0.0))
        assert np.isfinite(val)
        assert val == pytest.approx(_quad_k0_cell(1.0, H149, theta), rel=1e-8)
    # the published primitive P(u) (renamed from _struve_F when the one-argument
    # F(x) = (2/pi)*int_0^x K0 was introduced) and F itself both vanish at 0
    assert float(np.asarray(ops._struve_primitive(1.0, 0.0))) == 0.0
    assert float(np.asarray(ops._struve_F(0.0))) == 0.0
    assert float(np.asarray(ops._struve_Fc(0.0))) == 1.0


def test_branch3_is_even_in_theta_and_in_Q():
    theta = np.array([0.013, 0.1, 0.4, 1.3])
    plus = ops.coulomb_V_entries(1.3, H149, theta, 0.0)
    minus = ops.coulomb_V_entries(1.3, H149, -theta, 0.0)
    np.testing.assert_allclose(plus, minus, rtol=1e-14)
    # the kernel depends on |q| only (matters for +-q dissymmetry sweeps)
    np.testing.assert_allclose(
        ops.coulomb_V_entries(-1.3, H149, theta, 0.0), plus, rtol=0.0, atol=0.0
    )


# ===========================================================================
# Coulomb branch 2 -- spec test (b)
# ===========================================================================


@pytest.mark.parametrize("ztil", [0.0, 0.01, 0.3, 1.0, -0.7])
@pytest.mark.parametrize("theta", [0.0, H149 / 2.0, 0.1, 0.5, 1.0, 2.0, -0.4])
def test_branch2_matches_quadrature(ztil, theta):
    """Spec test (b): branch 2 is the exact cell integral of -2*log(R) at Q = 0."""
    got = float(ops.coulomb_V_entries(0.0, H149, theta, ztil))
    ref = _quad_log_cell(H149, theta, ztil)
    # atol covers theta ~ 1 where the primitive crosses zero and the relative
    # error of the reference quadrature itself blows up.
    assert got == pytest.approx(ref, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("ztil", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("theta", [0.1, 0.3, 0.5, 0.9, 1.4])
def test_branch2_primitive_differentiates_to_the_integrand(ztil, theta):
    """Spec test (b), unambiguous form: dV/dtheta = g(b) - g(a).

    With g(t) = -2*log(sqrt(t**2+ztil**2)) and a, b = theta +- h/2, both limits
    shift with theta, so the derivative of the branch-2 primitive must be the
    integrand difference across the cell.
    """
    eps = 1e-5
    v_p = float(ops.coulomb_V_entries(0.0, H149, theta + eps, ztil))
    v_m = float(ops.coulomb_V_entries(0.0, H149, theta - eps, ztil))
    numeric = (v_p - v_m) / (2.0 * eps)

    def g(t: float) -> float:
        return -np.log(t * t + ztil * ztil)

    exact = g(theta + 0.5 * H149) - g(theta - 0.5 * H149)
    assert numeric == pytest.approx(exact, rel=1e-6, abs=1e-12)


def test_branch2_ztil_zero_is_the_limit_of_the_general_form():
    """The ztil -> 0 special case agrees with the general formula, error ~ ztil."""
    theta = np.array([0.0, H149 / 2.0, 0.2, 0.8])
    at_zero = ops.coulomb_V_entries(0.0, H149, theta, 0.0)
    prev = None
    for ztil in (1e-6, 1e-8, 1e-10):
        near = ops.coulomb_V_entries(0.0, H149, theta, ztil)
        err = float(np.max(np.abs(near - at_zero)))
        if prev is not None:
            assert err < prev, "branch 2 does not converge as ztil -> 0"
        prev = err
    np.testing.assert_allclose(
        ops.coulomb_V_entries(0.0, H149, theta, 1e-10), at_zero, rtol=1e-7, atol=1e-12
    )


def test_branch2_centre_cell_closed_form():
    """theta = 0, ztil = 0 reduces to 2*h*(1 - log(h/2)) -- hand-integrated."""
    h = H149
    got = float(ops.coulomb_V_entries(0.0, h, 0.0, 0.0))
    assert got == pytest.approx(2.0 * h * (1.0 - np.log(h / 2.0)), rel=1e-14)


# ===========================================================================
# Coulomb branch 1 -- spec test (c) and midpoint convergence
# ===========================================================================


@pytest.mark.parametrize("big_q", [0.5, 1.0, 2.0])
def test_branch1_small_ztil_approaches_branch3(big_q):
    """Spec test (c): branch 1 at ztil -> 0 approaches branch 3, rtol 1e-4.

    The residual is NOT the ztil offset (that dies like ztil**2); it is the
    O(h**2) midpoint-rule error of branch 1 against the exact branch 3. At
    h = 1/149 that error exceeds 1e-4 for |theta| < ~0.1, where K0'' is large,
    so the comparison is made over |theta| >= 0.2. The h-convergence of the
    remaining gap is pinned by the next test.
    """
    theta = np.linspace(0.2, 1.0, 40)
    exact = ops.coulomb_V_entries(big_q, H149, theta, 0.0)
    for ztil in (1e-5, 1e-7):
        midpoint = ops.coulomb_V_entries(big_q, H149, theta, ztil)
        rel = np.max(np.abs(midpoint - exact) / np.abs(exact))
        assert rel < 1e-4, f"Q={big_q} ztil={ztil}: max rel {rel:.3e}"


def test_branch1_midpoint_error_is_second_order_in_h():
    """The branch1/branch3 gap is the midpoint error: it quarters as h halves."""
    theta, big_q = 0.3, 1.0
    errs = []
    for n in (150, 300, 600, 1200):
        h = 1.0 / (n - 1)
        exact = float(ops.coulomb_V_entries(big_q, h, theta, 0.0))
        midpoint = float(ops.coulomb_V_entries(big_q, h, theta, 1e-9))
        errs.append(abs(midpoint - exact) / abs(exact))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert r == pytest.approx(4.0, rel=0.05), f"ratios {ratios}"


def test_branch1_is_the_plain_midpoint_rule():
    """Branch 1 is literally 2*h*K0(Q*sqrt(theta^2+ztil^2))."""
    theta = np.array([0.0, 0.1, 0.6, -1.2])
    big_q, ztil, h = 1.4, 0.45, H149
    got = ops.coulomb_V_entries(big_q, h, theta, ztil)
    ref = 2.0 * h * k0(big_q * np.sqrt(theta**2 + ztil**2))
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)


def test_branch1_error_in_ztil_is_quadratic():
    """At fixed theta, branch 1 is smooth in ztil with error ~ ztil**2."""
    theta = np.array([0.1, 0.5, 1.0])
    base = 2.0 * H149 * k0(1.0 * np.abs(theta))  # branch 1 evaluated at ztil = 0
    errs = []
    for ztil in (1e-3, 1e-4, 1e-5):
        near = ops.coulomb_V_entries(1.0, H149, theta, ztil)
        errs.append(float(np.max(np.abs(near - base) / np.abs(base))))
    for i in range(len(errs) - 1):
        assert errs[i] / errs[i + 1] == pytest.approx(100.0, rel=0.1), f"{errs}"


# ===========================================================================
# coulomb_V_entries dispatch and shape contract
# ===========================================================================


def test_dispatch_selects_the_specified_branch():
    theta = np.array([0.0, 0.21, -0.9])
    h = H149
    np.testing.assert_allclose(
        ops.coulomb_V_entries(0.0, h, theta, 0.4),
        ops._v_branch2_q0(h, theta, 0.4),
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        ops.coulomb_V_entries(0.0, h, theta, 0.0),
        ops._v_branch2_q0(h, theta, 0.0),
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        ops.coulomb_V_entries(1.1, h, theta, 0.0),
        ops._v_branch3_struve(1.1, h, theta),
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        ops.coulomb_V_entries(1.1, h, theta, 0.4),
        ops._v_branch1_midpoint(1.1, h, theta, 0.4),
        rtol=0.0, atol=0.0,
    )


@pytest.mark.parametrize("shape", [(), (5,), (3, 4), (2, 3, 4)])
def test_coulomb_V_entries_preserves_shape(shape):
    theta = np.full(shape, 0.3) if shape else np.float64(0.3)
    for big_q, ztil in ((0.0, 0.0), (0.0, 0.5), (1.0, 0.0), (1.0, 0.5)):
        out = ops.coulomb_V_entries(big_q, H149, theta, ztil)
        assert isinstance(out, np.ndarray)
        assert out.shape == shape
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))


# ===========================================================================
# build_V
# ===========================================================================


def _grid(n):
    return np.linspace(-0.5, 0.5, n), 1.0 / (n - 1)


def test_build_V_entries_match_the_normalised_separation_formula():
    n_j, n_jp = 7, 11
    th_j, _ = _grid(n_j)
    th_jp, h_jp = _grid(n_jp)
    q, w_j, w_jp = 0.08, 30.0, 45.0
    x_j, x_jp, z_j, z_jp = 12.0, -7.0, 4.0, 1.0

    got = ops.build_V(q, w_j, w_jp, x_j, x_jp, z_j, z_jp, th_j, th_jp, h_jp)
    assert got.shape == (n_j, n_jp)
    assert got.dtype == np.complex128

    ztil = (z_j - z_jp) / w_jp
    big_q = q * w_jp
    ref = np.empty((n_j, n_jp))
    for l in range(n_j):
        for lp in range(n_jp):
            sep = (x_j - x_jp + w_j * th_j[l] - w_jp * th_jp[lp]) / w_jp
            ref[l, lp] = ops.coulomb_V_entries(big_q, h_jp, sep, ztil)
    np.testing.assert_allclose(got.real, ref, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got.imag, 0.0, rtol=0.0, atol=0.0)


def test_build_V_single_ribbon_is_symmetric_and_positive_definite():
    """The q-space Coulomb kernel 2*K0 is positive definite; the exact-cell
    discretisation must inherit that (it is the Gram matrix of the cells)."""
    n = 60
    theta, h = _grid(n)
    for big_q in (0.5, 2.0, 5.0):
        q = big_q / 50.0
        v = ops.build_V(q, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, theta, theta, h)
        np.testing.assert_allclose(v, v.T, rtol=0.0, atol=0.0)
        eig = np.linalg.eigvalsh(v.real)
        assert eig.min() > 0.0, f"Q={big_q}: min eig {eig.min():.3e}"


def test_build_V_z_offset_uses_branch1_and_decays_with_height():
    n = 20
    theta, h = _grid(n)
    prev = None
    for z in (1.0, 5.0, 20.0, 60.0):
        v = ops.build_V(0.04, 50.0, 50.0, 0.0, 0.0, z, 0.0, theta, theta, h)
        nrm = float(np.max(np.abs(v)))
        if prev is not None:
            assert nrm < prev, "Coulomb coupling must decay with height"
        prev = nrm


# ===========================================================================
# build_V_toeplitz
# ===========================================================================


@pytest.mark.parametrize(
    "z_off, big_q",
    [
        (0.0, 2.0),    # branch 3 (exact Struve kernel)
        (0.0, 0.0),    # branch 2 (logarithmic kernel)
        (18.0, 2.0),   # branch 1 (midpoint, out-of-plane)
        (18.0, 0.0),   # branch 2 with a height offset
    ],
)
def test_build_V_toeplitz_matches_build_V_single_ribbon(z_off, big_q):
    """Spec requirement: entry-for-entry equality at rtol 1e-12."""
    n = 64
    theta, h = _grid(n)
    w = 50.0
    q = big_q / w
    dense = ops.build_V(q, w, w, 0.0, 0.0, z_off, 0.0, theta, theta, h)
    fast = ops.build_V_toeplitz(q, w, 0.0, z_off, theta, h)
    assert fast.shape == dense.shape
    assert fast.dtype == np.complex128
    np.testing.assert_allclose(fast, dense, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "x_off, z_off, big_q",
    [
        (37.0, 0.0, 2.0),    # lateral offset, branch 3, NOT symmetric
        (37.0, 0.0, 0.0),    # lateral offset at q = 0
        (-23.0, 11.0, 1.3),  # both offsets, branch 1
    ],
)
def test_build_V_toeplitz_matches_build_V_offset_ribbons(x_off, z_off, big_q):
    """Beyond the spec: equal-width ribbons at an offset are still exactly Toeplitz.

    Tolerance note. With x_off != 0 the dense and Toeplitz paths reach the same
    mathematical separation through different arithmetic, so their inputs differ
    in the last bit. Branch 3 computes ``pi*(F(b) - F(a))`` with F = O(1) and the
    difference O(h*K0), so a last-bit input change moves the answer by a few ulps
    of O(1) -- an ~1e-15 ABSOLUTE floor, independent of the entry's size. The
    assertion is therefore anchored on atol, with rtol well inside it.
    """
    n = 64
    theta, h = _grid(n)
    w = 50.0
    q = big_q / w
    dense = ops.build_V(q, w, w, x_off, 0.0, z_off, 0.0, theta, theta, h)
    fast = ops.build_V_toeplitz(q, w, x_off, z_off, theta, h)
    assert fast.shape == dense.shape
    np.testing.assert_allclose(fast, dense, rtol=1e-11, atol=5e-15)
    # The first row and column share build_V's own grouping, so they are exact.
    np.testing.assert_array_equal(fast[:, 0], dense[:, 0])
    np.testing.assert_array_equal(fast[0, :], dense[0, :])


def test_build_V_toeplitz_symmetry_follows_the_offset():
    """Symmetric only when x_off == 0; the general toeplitz(c, r) form is needed."""
    n = 32
    theta, h = _grid(n)
    centred = ops.build_V_toeplitz(0.04, 50.0, 0.0, 0.0, theta, h)
    np.testing.assert_allclose(centred, centred.T, rtol=0.0, atol=0.0)
    shifted = ops.build_V_toeplitz(0.04, 50.0, 37.0, 0.0, theta, h)
    asym = float(np.max(np.abs(shifted - shifted.T)))
    assert asym > 1e-6, f"off-centre block should not be symmetric (asym {asym:.2e})"


def test_build_V_toeplitz_rejects_non_uniform_grid():
    n = 16
    theta, h = _grid(n)
    bad = theta.copy()
    bad[5] += 0.3 * h
    with pytest.raises(ValueError, match="uniform grid"):
        ops.build_V_toeplitz(0.04, 50.0, 0.0, 0.0, bad, h)
    with pytest.raises(ValueError, match="uniform grid"):
        ops.build_V_toeplitz(0.04, 50.0, 0.0, 0.0, theta, 2.0 * h)
    with pytest.raises(ValueError, match="uniform grid"):
        ops.build_V_toeplitz(0.04, 50.0, 0.0, 0.0, theta[::-1], h)
    with pytest.raises(ValueError, match="at least 2 nodes"):
        ops.build_V_toeplitz(0.04, 50.0, 0.0, 0.0, np.array([0.0]), h)


@pytest.mark.parametrize("z_off, label", [(0.0, "branch3"), (15.0, "branch1")])
def test_build_V_toeplitz_is_at_least_5x_faster_at_L800(z_off, label):
    """Performance requirement: >= 5x speedup over the dense build at L = 800."""
    n = 800
    theta, h = _grid(n)
    w, q = 50.0, 2.0 / 50.0

    def dense():
        return ops.build_V(q, w, w, 0.0, 0.0, z_off, 0.0, theta, theta, h)

    def fast():
        return ops.build_V_toeplitz(q, w, 0.0, z_off, theta, h)

    dense(), fast()  # warm up scipy/numpy dispatch out of the timing

    def best_of_3(fn):
        out = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return min(out)

    t_dense = best_of_3(dense)
    t_fast = best_of_3(fast)
    speedup = t_dense / t_fast
    # atol is the branch-3 cancellation floor; see the offset-ribbon test.
    np.testing.assert_allclose(fast(), dense(), rtol=1e-11, atol=5e-15)
    assert speedup >= 5.0, (
        f"{label}: build_V_toeplitz only {speedup:.2f}x faster at L={n} "
        f"(dense {t_dense * 1e3:.2f} ms, toeplitz {t_fast * 1e3:.2f} ms)"
    )


# ===========================================================================
# build_D / build_D_geom
# ===========================================================================


def test_build_D_matches_the_specified_stencils():
    n = 9
    h = 1.0 / (n - 1)
    q, w = 0.07, 40.0
    e_perp, e_par = 1.3 - 0.4j, 0.6 + 2.1j
    got = ops.build_D(q, w, h, n, e_perp, e_par)
    ref = _ref_build_D(q, w, h, n, e_perp, e_par)
    assert got.shape == (n, n)
    assert got.dtype == np.complex128
    np.testing.assert_allclose(got, ref, rtol=1e-15, atol=0.0)


def test_build_D_with_filling_matches_the_symmetric_form():
    n = 11
    h = 1.0 / (n - 1)
    q, w = 0.07, 40.0
    e_perp, e_par = 1.3 - 0.4j, 0.6 + 2.1j
    f = np.array([0.0, 0.0, 0.3, 1.0, 1.0, 1.0, 1.0, 0.8, 0.2, 0.0, 0.0])
    got = ops.build_D(q, w, h, n, e_perp, e_par, f=f)
    ref = _ref_build_D(q, w, h, n, e_perp, e_par, f=f)
    np.testing.assert_allclose(got, ref, rtol=1e-15, atol=0.0)


def test_build_D_isotropic_equals_scaled_build_D_geom_exactly():
    """Spec test: build_D(eta, eta) == eta * build_D_geom, exactly."""
    n = 17
    h = 1.0 / (n - 1)
    q, w = 0.05, 60.0
    f = np.linspace(0.0, 1.0, n) ** 2
    for eta in (1.0 + 0.0j, 2.5 - 0.75j, -0.3 + 4.0j):
        for filling in (None, f):
            got = ops.build_D(q, w, h, n, eta, eta, f=filling)
            ref = eta * ops.build_D_geom(q, w, h, n, f=filling)
            np.testing.assert_array_equal(got, ref)


def test_build_D_geom_is_build_D_with_unit_etas():
    n = 13
    h = 1.0 / (n - 1)
    np.testing.assert_array_equal(
        ops.build_D_geom(0.06, 35.0, h, n),
        ops.build_D(0.06, 35.0, h, n, 1.0, 1.0),
    )


def test_build_D_geom_is_tridiag_over_h2_minus_Q2_identity():
    n = 12
    h = 1.0 / (n - 1)
    q, w = 0.09, 25.0
    big_q = q * w
    got = ops.build_D_geom(q, w, h, n)
    tri = np.zeros((n, n))
    for l in range(1, n - 1):
        tri[l, l - 1], tri[l, l], tri[l, l + 1] = 1.0, -2.0, 1.0
    tri[0, 0], tri[0, 1] = -1.0, 1.0
    tri[n - 1, n - 1], tri[n - 1, n - 2] = -1.0, 1.0
    np.testing.assert_allclose(
        got, tri / h**2 - big_q**2 * np.eye(n), rtol=1e-15, atol=1e-18
    )


@pytest.mark.parametrize("n", [2, 3, 8, 150])
def test_build_D_rows_sum_to_zero_at_q0(n):
    """Charge conservation / vanishing normal current. Spec tolerance 1e-12."""
    h = 1.0 / (n - 1)
    d = ops.build_D(0.0, 40.0, h, n, 1.7 - 0.9j, 0.4 + 0.2j)
    sums = np.abs(d.sum(axis=1))
    assert sums.max() < 1e-12, f"L={n}: max |row sum| {sums.max():.3e}"


@pytest.mark.parametrize("n", [5, 40])
def test_build_D_rows_sum_to_zero_at_q0_with_filling(n):
    h = 1.0 / (n - 1)
    rng = np.random.default_rng(20261101)
    f = rng.uniform(0.0, 1.0, n)
    d = ops.build_D(0.0, 40.0, h, n, 1.7 - 0.9j, 0.4 + 0.2j, f=f)
    sums = np.abs(d.sum(axis=1))
    assert sums.max() < 1e-12, f"L={n}: max |row sum| {sums.max():.3e}"


def test_build_D_row_sums_are_minus_eta_par_Q2_f():
    """At finite q every row sums to -eta_par*(qW)**2*f[l]."""
    n = 24
    h = 1.0 / (n - 1)
    q, w = 0.11, 30.0
    e_par = 0.4 + 0.2j
    f = np.linspace(0.1, 1.0, n)
    d = ops.build_D(q, w, h, n, 1.7 - 0.9j, e_par, f=f)
    np.testing.assert_allclose(
        d.sum(axis=1), -e_par * (q * w) ** 2 * f, rtol=1e-12, atol=1e-14
    )


def test_build_D_filling_of_ones_equals_no_filling():
    n = 19
    h = 1.0 / (n - 1)
    np.testing.assert_array_equal(
        ops.build_D(0.05, 45.0, h, n, 1.1 + 0.2j, 0.9 - 0.3j, f=np.ones(n)),
        ops.build_D(0.05, 45.0, h, n, 1.1 + 0.2j, 0.9 - 0.3j, f=None),
    )


def test_build_D_is_symmetric():
    """D is a symmetric (not Hermitian) operator for any eta and any f."""
    n = 15
    h = 1.0 / (n - 1)
    f = np.abs(np.sin(np.linspace(0.0, 3.0, n)))
    d = ops.build_D(0.07, 33.0, h, n, 1.2 - 0.5j, 0.3 + 0.8j, f=f)
    np.testing.assert_allclose(d, d.T, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("n", [5, 9, 40])
def test_build_D_geom_spectrum_is_the_neumann_laplacian(n):
    """Closed-form reference: eig = -(4/h**2)*sin(k*pi/(2L))**2 - (qW)**2.

    This is the discrete Neumann (vanishing-flux) Laplacian spectrum, so it
    validates the boundary rows against an independent analytic result, and in
    the L -> inf limit reproduces the continuum eigenvalues -(k*pi)**2 on
    theta in [-1/2, 1/2].
    """
    h = 1.0 / (n - 1)
    q, w = 0.08, 27.0
    big_q = q * w
    got = np.sort(np.linalg.eigvalsh(ops.build_D_geom(q, w, h, n).real))
    k = np.arange(n)
    ref = np.sort(-(4.0 / h**2) * np.sin(k * np.pi / (2.0 * n)) ** 2 - big_q**2)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-10)


def test_build_D_geom_lowest_modes_approach_continuum_neumann():
    """Physics check: -(k*pi)**2 on a unit interval as L grows (q = 0)."""
    n = 2000
    h = 1.0 / (n - 1)
    eig = np.sort(np.linalg.eigvalsh(ops.build_D_geom(0.0, 10.0, h, n).real))[::-1]
    for k in range(1, 5):
        assert -eig[k] == pytest.approx((k * np.pi) ** 2, rel=2e-3)


def test_build_D_rejects_bad_arguments():
    with pytest.raises(ValueError, match="at least 2 nodes"):
        ops.build_D(0.05, 40.0, 0.1, 1, 1.0, 1.0)
    with pytest.raises(ValueError, match="length"):
        ops.build_D(0.05, 40.0, 0.1, 6, 1.0, 1.0, f=np.ones(5))
    with pytest.raises(ValueError, match="h must be non-zero"):
        ops.build_D(0.05, 40.0, 0.0, 6, 1.0, 1.0)


# ===========================================================================
# module surface
# ===========================================================================


def test_public_surface_matches_the_contract():
    expected = {
        "struve_L",
        "coulomb_V_entries",
        "build_V",
        "build_V_toeplitz",
        "build_D",
        "build_D_geom",
    }
    assert set(ops.__all__) == expected
    for name in expected:
        fn = getattr(ops, name)
        assert callable(fn)
        assert fn.__doc__, f"{name} has no docstring"
