"""Finite-difference operators for quasistatic graphene-nanoribbon simulation.

This module builds the two matrix blocks of the quasistatic (electrostatic)
boundary-element/finite-difference scheme for zero-thickness ribbons that are
infinite along ``y`` and carry a guided wave vector ``q``:

* the **Coulomb operator** ``V`` -- paper 406 SI, reference implementation
  ``set_V`` inside ``GNR_phi_LDOS_anisotropic.m``;
* the **anisotropic differential operator** ``D`` -- paper 456 Eq. (19),
  reference implementation ``set_D``.

Geometry / normalisation (see the package README):
ribbon ``j`` has width ``W_j`` [nm], centre ``x_j`` [nm] and height ``z_j``
[nm].  The normalised coordinate is ``theta in [-1/2, 1/2]`` on ``L_j`` uniform
nodes with spacing ``h_j = 1/(L_j - 1)`` and ``x = x_j + W_j*theta``.  Ribbon
pairs enter only through

    theta_jjp_llp = (x_j - x_jp + W_j*theta_j_l - W_jp*theta_jp_lp) / W_jp
    ztil_jjp      = (z_j - z_jp) / W_jp

and the dimensionless wave vector ``Q = q * W_jp``.

MODIFIED STRUVE FUNCTIONS: THE FOUR-REGION EVALUATION
-----------------------------------------------------
The coplanar Coulomb branch (branch 3) is the exact cell integral of ``2*K0``,
which the reference MATLAB writes through the modified Struve functions as
``x*(K0(x)*L_{-1}(x) + K1(x)*L_0(x))`` at ``x = Q*|theta +- h/2|``.  For a
coplanar ribbon *pair* that argument is ``q*d`` with ``d`` the centre-to-centre
separation, so it must be allowed to run to tens or hundreds -- far past the
30-term Taylor series of the reference MATLAB, whose measured relative error is
7e-8 at 10, 2e-4 at 15 and 45% at 30.

Two separate things are needed, and they are not the same thing:

1. ``struve_L`` itself uses the 30-term series for ``|x| <= 4``, where it is
   exact to <1e-15, and ``scipy.special.modstruve`` above that.  scipy's
   implementation was measured against ``mpmath.struvel`` at <=7.4e-13
   relative all the way to ``x = 710``, where ``L_n`` overflows double
   precision (it then returns ``+inf``, as the true function does).  That
   worst case sits at ``x ~ 14``, where scipy switches its own algorithm; it
   is 3e-16 or better over most of the range, and the series-to-scipy
   switch-over at 4 is seamless (the two agree to 4.4e-16 there).  No
   hand-rolled asymptotic series is needed for ``L_n`` itself.

   Note that this 7.4e-13 does *not* leak into the Coulomb kernel: ``Fc``
   below is built from the series (``x <= 4``) and from Chebyshev/asymptotic
   expansions of its own (``x > 4``), and never calls ``modstruve``.

2. The *combination* the kernel needs,

       F(x) = x*(K0(x)*L_{-1}(x) + K1(x)*L_0(x)) = (2/pi)*int_0^x K0(t) dt,

   cannot be evaluated from ``L_n`` and ``K_n`` separately at large ``x``: the
   factors go like ``exp(+x)`` and ``exp(-x)``, so the product is formed from a
   huge number and a tiny one and loses everything to overflow/underflow --
   ``K0`` underflows to 0 while ``L_n`` is already ``inf`` past ``x ~ 746``,
   giving ``NaN``.  Worse, ``F -> 1`` exponentially, so what branch 3 actually
   wants -- the *difference* ``F(b) - F(a)`` -- is a cancellation between two
   numbers that both round to 1.

   The module therefore evaluates the complementary function

       Fc(x) = 1 - F(x) = (2/pi)*int_x^inf K0(t) dt
             = x*( K0(x)*(I_1(x) - L_{-1}(x)) + K1(x)*(I_0(x) - L_0(x)) ),

   the second form following from the Wronskian ``I_0*K_1 + I_1*K_0 = 1/x``;
   it exhibits the cancellation of the growing part of ``L`` against ``K``
   explicitly.  ``Fc`` is evaluated in the exponentially scaled form

       Fc(x) = sqrt(2/(pi*x)) * exp(-x) * S(x),     S(x) -> sqrt(2/pi),

   with the two representations of ``S`` given by ``_FC_CHEB_C`` (Chebyshev in
   ``1/x``) and ``_FC_ASYM`` (asymptotic in ``1/x``).  Branch 3 then uses
   ``F(b) - F(a) = Fc(a) - Fc(b)`` whenever the cell does not straddle the
   origin, which is exact in *relative* terms however small the coupling is.

Measured worst relative error of ``F``/``Fc`` against ``mpmath`` (reference
``1 - x*(K0*L_{-1} + K1*L_0)`` evaluated at ``45 + 0.44*x`` decimal digits,
itself cross-checked against high-precision quadrature of ``int_x^inf K0``):

    region of Fc              x range        Fc rel. err.   F rel. err.
    30-term series, 1 - F     0 < x <= 4     <= 3.9e-14     <= 6.3e-16
    Chebyshev panel 1 in 1/x  4 < x <= 8     <= 3.1e-16     <= 6e-17
    Chebyshev panel 2 in 1/x  8 < x <= 48    <= 3.1e-16     <= 6e-17
    asymptotic in 1/x         48 < x <= 700  <= 3.2e-16     <= 1 ulp

(worst case per region on a 300/120/150/80-point geometric grid; the 3.9e-14
sits at x = 3.89, where ``Fc = 7.2e-3`` and ``1 - F`` has thrown away 2.1
digits)

The one ``1 - F`` region loses digits to that subtraction and nowhere else,
which is why it stops at 4: past there ``Fc < 6.5e-3`` and falling
exponentially, and a Chebyshev fit of ``S`` is what keeps *relative* accuracy.
A coplanar pair therefore now decouples -- the branch-3 cross block tracks
``2*h*K0(q*d)`` bit for bit down into the denormals (6.3e-308 at ``q*d = 700``,
1.3e-316 at 720, and both reach exactly 0 at 740, where the true value is below
the smallest denormal 4.9e-324) instead of flattening at the old 9.07e-15
absolute noise floor.  Nothing in ``0 < x <= 1e4`` overflows or returns
``NaN``.

ONE FLOOR REMAINS, unchanged from before and now the only one.  For
``Q*|theta| < 4`` the difference branch 3 forms is still a difference of two
``O(1)`` numbers (``Fc`` is ``O(1)`` there, and no exponential cancellation is
possible because there is nothing exponentially small to resolve), so ``V``
keeps an absolute floor of a few ulps of ``pi/Q``.  Measured worst relative
error of branch 3 at ``h = 1/149``: 3.3e-11 at ``Q = 0.1`` (argument 4, where
``Q*h = 6.7e-4`` makes the cell integral 11 orders below the primitives),
3.1e-12 at ``Q = 1``, and <=3.4e-12 at every argument above 4 out to 600.
Extending the Chebyshev panels down to ``x ~ 1`` would remove this one too; it
was left alone because it is bounded, documented, pre-existing (the old form
had exactly this floor at *every* argument) and harmless -- at those arguments
``V`` is not exponentially small, so nothing decouples wrongly.

``tests/test_operators.py`` pins all of the above against ``mpmath`` (with a
quadrature fallback when mpmath is absent), the continuity of every
switch-over, the coplanar decoupling, and cross-checks branch 3 against branch
1 (midpoint rule at small ``ztil``) as required by the module specification.

Dtype policy: the scalar kernel evaluators (``struve_L``,
``coulomb_V_entries``) return real ``float64`` because the kernel is manifestly
real; the four matrix builders return ``complex128`` so that the blocks drop
straight into the complex system matrix assembled by ``solver.RibbonSystem``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.polynomial import polyval
from scipy.linalg import toeplitz
from scipy.special import k0, k1, modstruve

__all__ = [
    "struve_L",
    "coulomb_V_entries",
    "build_V",
    "build_V_toeplitz",
    "build_D",
    "build_D_geom",
]
# NOTE: STRUVE_TRUSTED_MAX_ARG is a public module attribute but is deliberately
# kept OUT of __all__, which tests/test_operators.py pins to exactly the six
# names of the module contract.

Scalar = Union[float, complex, np.number]
ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Modified Struve functions L_0 and L_{-1}
# ---------------------------------------------------------------------------
# Coefficient lists exactly as supplied in the module specification (the ``p``
# arrays of the reference MATLAB).  They are indexed from the CONSTANT term
# upward, and the MATLAB then evaluates ``polyval(flip(2*p/pi), x)``, i.e.
#
#     StruveL(n, x) = (2/pi) * sum_k p[k] * x**k
#
# These are the Taylor coefficients of the modified Struve function
#     L_v(x) = sum_k (x/2)**(2k+v+1) / (Gamma(k+3/2) * Gamma(k+v+3/2)),
# verified term by term against ``scipy.special.modstruve``.
_STRUVE_P = {
    0: (
        0.0, 1.0, 0.0, 1 / 9, 0.0, 1 / 225, 0.0, 1 / 11025, 0.0, 1 / 893025,
        0.0, 1 / 108056025, 0.0, 1 / 18261468225, 0.0, 1 / 4108830350625,
        0.0, 1 / 1187451971330625, 0.0, 1 / 428670161650355625,
        0.0, 1 / 189043541287806830625,
        0.0, 1 / 100004033341249813400625,
        0.0, 1 / 62502520838281133375390625,
        0.0, 1 / 45564337691106946230659765625,
        0.0, 1 / 38319607998220941779984862890625,
    ),
    -1: (
        1.0, 0.0, 1 / 3, 0.0, 1 / 45, 0.0, 1 / 1575, 0.0, 1 / 99225,
        0.0, 1 / 9823275, 0.0, 1 / 1404728325, 0.0, 1 / 273922023375,
        0.0, 1 / 69850115960625, 0.0, 1 / 22561587455281875,
        0.0, 1 / 9002073394657468125, 0.0, 1 / 4348001449619557104375,
        0.0, 1 / 2500100833531245335015625,
        0.0, 1 / 1687568062633590601135546875,
        0.0, 1 / 1321365793042101440689133203125,
        0.0, 1 / 1187907847944849195179530749609375,
    ),
}

# Pre-scaled, ascending-order coefficients -- mirrors the MATLAB ``2*p/pi``.
_STRUVE_P_SCALED = {
    n: (2.0 / np.pi) * np.asarray(p, dtype=np.float64)
    for n, p in _STRUVE_P.items()
}


#: Largest ``|x|`` at which the 30-term Taylor series is used.  It is exact to
#: <1e-15 up to 4 (measured against ``mpmath.struvel``), 4e-13 at 6 and 4e-10
#: at 8, so the switch-over sits at 4: that is the largest argument at which
#: series and ``scipy.special.modstruve`` agree to full double precision, which
#: makes the switch-over seamless rather than merely small (measured worst
#: disagreement over ``[3, 4]``: 3.9e-16 for ``L_0``, 4.4e-16 for ``L_{-1}``).
_STRUVE_SERIES_MAX = 4.0


def struve_L(n: int, x: ArrayLike) -> np.ndarray:
    """Modified Struve function ``L_n(x)``, accurate for every ``x``.

    Source: reference MATLAB ``GNR_phi_LDOS_anisotropic.m`` (paper 406 SI),
    which tabulates the coefficient arrays ``p`` and evaluates
    ``StruveL(n, x) = (2/pi) * sum_k p[k] * x**k``.  Equivalent to the
    Abramowitz & Stegun 12.2.1 series for ``L_v``.  That 30-term series is used
    for ``|x| <= 4`` (bit for bit the reference MATLAB); past 4 it decays into
    nonsense (45% at 30) and ``scipy.special.modstruve`` is used instead, which
    was measured against ``mpmath.struvel`` at <=7.4e-13 relative out to
    ``x = 710``.  The two branches agree to <=4.4e-16 at the switch-over.

    The parity of the series is used to extend the large-argument branch to
    negative arguments: ``L_0`` is odd and ``L_{-1}`` is even, because the
    ``p`` arrays carry only odd / only even powers.

    Parameters
    ----------
    n
        Order, restricted to ``{-1, 0}`` (the only orders tabulated).
    x
        Real argument, any shape.

    Returns
    -------
    ndarray
        ``L_n(x)``, real ``float64``, same shape as ``x`` (0-d for scalar
        input).  ``+-inf`` past ``|x| ~ 711``, where ``L_n`` genuinely overflows
        double precision; see :func:`_struve_Fc` for the kernel combination
        that stays finite everywhere.  Measured worst relative error against
        ``mpmath.struvel`` over ``0 < x <= 710``: 7.4e-13, at ``x ~ 14`` where
        scipy changes algorithm; 3e-16 or better elsewhere.
    """
    key = int(n)
    if key not in _STRUVE_P_SCALED:
        raise ValueError(
            f"struve_L supports only n in {{-1, 0}} (30-term series); got n={n!r}"
        )
    xa = np.asarray(x, dtype=np.float64)
    flat = np.atleast_1d(xa)
    out = np.asarray(polyval(flat, _STRUVE_P_SCALED[key]), dtype=np.float64)
    ax = np.abs(flat)
    big = ax > _STRUVE_SERIES_MAX
    if big.any():
        vals = np.asarray(modstruve(key, ax[big]), dtype=np.float64)
        if key == 0:  # L_0 is odd; L_{-1} is even, so needs no sign
            vals = np.copysign(vals, flat[big])
        out = np.array(out, dtype=np.float64, copy=True)
        out[big] = vals
    return np.asarray(out, dtype=np.float64).reshape(xa.shape)


# ---------------------------------------------------------------------------
# The kernel combination F(x) = (2/pi)*int_0^x K0 and its complement Fc = 1 - F
# ---------------------------------------------------------------------------
#: Top of the one ``1 - F`` region of ``Fc``.  It coincides with
#: ``_STRUVE_SERIES_MAX``: below 4 the 30-term series is exact to <1e-16 and
#: ``Fc >= 6.5e-3``, so forming ``Fc = 1 - F`` costs at most ~2.2 decimal
#: digits; above 4 ``Fc`` decays exponentially and that subtraction would lose
#: everything, so the Chebyshev representation of ``S`` takes over.
_FC_DIRECT_MAX = _STRUVE_SERIES_MAX

#: Panel edges of the Chebyshev representation of
#: ``S(x) = Fc(x)*sqrt(pi*x/2)*exp(x)``: ``[4, 8]`` and ``[8, 48]``, with the
#: asymptotic series above 48.  Two panels rather than one because ``S(1/x)``
#: is singular at ``1/x = 0`` (the asymptotic series diverges) and at
#: ``1/x = inf`` (the branch point of ``int_x^inf K0`` at the origin), so a
#: single panel reaching from 4 to 48 would need ~50 terms where two need 33.
_FC_CHEB_EDGES = (4.0, 8.0, 48.0)
_FC_CHEB_MAX = _FC_CHEB_EDGES[-1]

# Chebyshev coefficients of ``S`` in the variable ``t = 1/x``, mapped onto
# ``[-1, 1]`` by ``u = (1/x - T0)/DT`` per panel.  Generated from the mpmath
# reference of ``Fc`` at 60 digits by the exact 40-node Chebyshev transform and
# truncated where the coefficients fall below 1e-18.  The retained 15 and 18
# terms reproduce ``S`` to 1.5e-16 and ``Fc`` to 3.3e-16 relative over their
# panels (measured on 401 points per panel; the tests re-measure this, see
# ``tests/test_operators.py::test_Fc_matches_high_precision_reference``).
_FC_CHEB_C = (
    # panel [4, 8]
    np.array([
        0.9082974956234201,
        -0.024960942676084553,
        0.0007387339902535993,
        -2.9975504476068714e-05,
        1.471306516950306e-06,
        -8.237544590917746e-08,
        5.0872512328319534e-09,
        -3.392145085795843e-10,
        2.4066687205256408e-11,
        -1.7977647690517254e-12,
        1.402816460432367e-13,
        -1.136531625444679e-14,
        9.514609399659055e-16,
        -8.198797126479277e-17,
        7.249085898490822e-18,
    ]),
    # panel [8, 48]
    np.array([
        0.9598423645326372,
        -0.026641573116113038,
        0.0008668368660399151,
        -4.1254394876558286e-05,
        2.49952624938183e-06,
        -1.8006312604375782e-07,
        1.4809942900316694e-08,
        -1.3541439475384415e-09,
        1.3509417909831991e-10,
        -1.4504453440988167e-11,
        1.658514719600091e-12,
        -2.0032759024177474e-13,
        2.5393778157700702e-14,
        -3.360213989513408e-15,
        4.621111405172876e-16,
        -6.580535536423102e-17,
        9.672744405025289e-18,
        -1.4636744854758292e-18,
    ]),
)
#: ``(T0, DT)`` of each panel: the midpoint and half-width of ``t = 1/x``.
_FC_CHEB_MAP = tuple(
    (0.5 * (1.0 / hi + 1.0 / lo), 0.5 * (1.0 / lo - 1.0 / hi))
    for lo, hi in zip(_FC_CHEB_EDGES[:-1], _FC_CHEB_EDGES[1:])
)


def _fc_asym_coeffs(n_terms: int) -> np.ndarray:
    """Ascending coefficients ``b_k`` of ``S(x) ~ sum_k b_k x**-k``.

    Derivation (the expansion is *not* guessed: it is fixed by ``Fc' = -K0``).
    Write ``Fc = sqrt(2/(pi*x))*exp(-x)*S`` and
    ``K0 = sqrt(pi/(2*x))*exp(-x)*A`` with the standard Hankel expansion
    ``A = sum_k a_k x**-k``, ``a_0 = 1``, ``a_k = a_{k-1}*(-(2k-1)**2)/(8k)``.
    Then ``Fc'(x) = -(2/pi)*K0(x)`` becomes ``S - S/(2x) ... `` -- explicitly,

        S' - S/(2*x) - S = -A   ==>   b_n = a_n - (n - 1/2)*b_{n-1},  b_0 = 1,

    which also reproduces the independent route through
    ``I_0 - L_0 ~ (2/pi)*sum (-1)**k (2k-1)!!**2 x**-(2k+1)`` and
    ``I_1 - L_{-1} ~ -(2/pi)*sum (-1)**k (2k-1)!!(2k+1)!! x**-(2k+2)``
    (DLMF 11.6.3) inserted into the Wronskian form of ``Fc``: both give
    ``S = 1 - 5/(8x) + ...``.
    """
    a = np.empty(n_terms, dtype=np.float64)
    b = np.empty(n_terms, dtype=np.float64)
    a[0] = 1.0
    b[0] = 1.0
    for k in range(1, n_terms):
        a[k] = a[k - 1] * (-((2 * k - 1) ** 2)) / (8.0 * k)
        b[k] = a[k] - (k - 0.5) * b[k - 1]
    return b


#: 20 terms: measured 4.6e-17 relative at ``x = 48`` and better above, while
#: the series is still far from its optimal-truncation turning point there
#: (14 terms give 4.2e-14 at 48, 24 terms give no further gain).
_FC_ASYM = _fc_asym_coeffs(20)


def _f_direct(x: np.ndarray) -> np.ndarray:
    """``F(x) = x*(K0*L_{-1} + K1*L_0)`` for ``0 <= x <= _FC_DIRECT_MAX``.

    The 30-term series of the reference MATLAB, which is exact to <1e-16 over
    this range -- so ``V`` in the historically validated small-argument range is
    bit for bit what the reference implementation gives.
    """
    is_zero = x == 0.0
    xs = np.where(is_zero, 1.0, x)  # keep K0/K1 away from their pole
    val = xs * (k0(xs) * struve_L(-1, xs) + k1(xs) * struve_L(0, xs))
    return np.where(is_zero, 0.0, val)


def _fc_tail(x: np.ndarray) -> np.ndarray:
    """``Fc(x) = sqrt(2/(pi*x))*exp(-x)*S(x)`` for ``x > _FC_DIRECT_MAX``.

    ``S`` comes from the Chebyshev panels below ``_FC_CHEB_MAX`` and from the
    asymptotic series above it.
    """
    t = 1.0 / x
    s = np.empty(x.shape, dtype=np.float64)
    done = np.zeros(x.shape, dtype=bool)
    for (lo, hi), (t0, dt), coeffs in zip(
        zip(_FC_CHEB_EDGES[:-1], _FC_CHEB_EDGES[1:]), _FC_CHEB_MAP, _FC_CHEB_C
    ):
        panel = (x > lo) & (x <= hi)
        if panel.any():
            s[panel] = chebval((t[panel] - t0) / dt, coeffs)
            done |= panel
    asym = ~done
    if asym.any():
        s[asym] = polyval(t[asym], _FC_ASYM)
    # exp(-x) underflows to 0.0 past x ~ 746; that is the correct double value
    # (the true Fc is then below the smallest denormal) and raises no warning.
    return np.sqrt(2.0 / (np.pi * x)) * np.exp(-x) * s


def _struve_F(x: ArrayLike) -> np.ndarray:
    """``F(x) = (2/pi)*int_0^x K0(t) dt = x*(K0*L_{-1} + K1*L_0)``, ``x >= 0``.

    Four-region evaluation; see the module docstring for the measured accuracy.
    ``F(0) = 0``, ``F`` increases monotonically to 1, and ``F`` is exactly
    ``1.0`` in double precision past ``x ~ 746``.
    """
    xa = np.asarray(x, dtype=np.float64)
    flat = np.atleast_1d(xa)
    small = flat <= _FC_DIRECT_MAX
    # The all / none fast paths matter: they keep a whole-block evaluation free
    # of the fancy-index copies, which is most of the cost at L ~ 800.
    if small.all():
        out = _f_direct(flat)
    elif not small.any():
        out = 1.0 - _fc_tail(flat)
    else:
        out = np.empty(flat.shape, dtype=np.float64)
        out[small] = _f_direct(flat[small])
        big = ~small
        out[big] = 1.0 - _fc_tail(flat[big])
    return np.asarray(out, dtype=np.float64).reshape(xa.shape)


def _struve_Fc(x: ArrayLike) -> np.ndarray:
    """``Fc(x) = 1 - F(x) = (2/pi)*int_x^inf K0(t) dt``, ``x >= 0``.

    Complementary to :func:`_struve_F` by construction in every region, so
    ``F + Fc == 1`` to one ulp and the difference forms used by branch 3 cannot
    disagree by more than a rounding error.  ``Fc(0) = 1``.
    """
    xa = np.asarray(x, dtype=np.float64)
    flat = np.atleast_1d(xa)
    small = flat <= _FC_DIRECT_MAX
    if small.all():
        out = 1.0 - _f_direct(flat)
    elif not small.any():
        out = _fc_tail(flat)
    else:
        out = np.empty(flat.shape, dtype=np.float64)
        out[small] = 1.0 - _f_direct(flat[small])
        big = ~small
        out[big] = _fc_tail(flat[big])
    return np.asarray(out, dtype=np.float64).reshape(xa.shape)


# ---------------------------------------------------------------------------
# Coulomb operator -- the three branches
# ---------------------------------------------------------------------------
# All three evaluate (up to the branch's own approximation)
#
#     V(theta) = 2 * int_{theta-h/2}^{theta+h/2} K0(Q*sqrt(t^2 + ztil^2)) dt
#
# i.e. the q-Fourier transform of the Coulomb kernel integrated over one
# finite-difference cell of normalised width h.  At Q = 0 the kernel 2*K0
# is replaced by its renormalised logarithmic limit -2*log(R).


def _xlogabs(x: np.ndarray) -> np.ndarray:
    """``x * log|x|`` with the ``x -> 0`` limit taken as 0."""
    ax = np.abs(x)
    is_zero = ax == 0.0
    safe = np.where(is_zero, 1.0, ax)
    return np.where(is_zero, 0.0, x * np.log(safe))


def _v_branch1_midpoint(Q: float, h: float, theta: np.ndarray, ztil: float) -> np.ndarray:
    """Branch 1 (``Q != 0`` and ``ztil != 0``): midpoint rule on the cell.

    ``V = 2*h*K0(Q*sqrt(theta**2 + ztil**2))`` -- paper 406 SI, ``set_V``.

    Second-order accurate in ``h``; it is the only branch used when the two
    ribbons sit in different planes.  Returns ``inf`` at ``theta = ztil = 0``
    (the kernel's own logarithmic singularity), which cannot be reached through
    :func:`coulomb_V_entries` because ``ztil == 0`` dispatches to branch 3.
    """
    r = np.sqrt(theta * theta + ztil * ztil)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * h * k0(Q * r)


def _v_branch2_q0(h: float, theta: np.ndarray, ztil: float) -> np.ndarray:
    """Branch 2 (``Q == 0``, any ``ztil``): exact cell integral of ``-2*log(R)``.

    With ``a = theta - h/2`` and ``b = theta + h/2`` (paper 406 SI, ``set_V``)::

        V = 2*( h + a*log(sqrt(a**2 + ztil**2)) + ztil*arctan(a/ztil)
                  - b*log(sqrt(b**2 + ztil**2)) - ztil*arctan(b/ztil) )

    This is the exact antiderivative of ``-log(t**2 + ztil**2)``.  For
    ``ztil == 0`` the ``ztil*arctan(./ztil) -> 0`` limit is taken and
    ``log(sqrt(a**2)) = log|a|``, with the ``x*log|x| -> 0`` limit at ``a == 0``
    or ``b == 0``.
    """
    a = theta - 0.5 * h
    b = theta + 0.5 * h
    if ztil == 0.0:
        return 2.0 * (h + _xlogabs(a) - _xlogabs(b))
    # a == 0 or b == 0 needs no special case here: 0*log|ztil| == 0 and
    # arctan(0/ztil) == 0.
    return 2.0 * (
        h
        + a * np.log(np.sqrt(a * a + ztil * ztil))
        + ztil * np.arctan(a / ztil)
        - b * np.log(np.sqrt(b * b + ztil * ztil))
        - ztil * np.arctan(b / ztil)
    )


def _struve_primitive(Q: float, u: ArrayLike) -> np.ndarray:
    """Primitive ``P(u) = u*(K0(Q|u|)*L_{-1}(Q|u|) + K1(Q|u|)*L_0(Q|u|))``.

    Paper 406 SI, ``set_V``.  Equals ``(2/pi) * int_0^u K0(Q|t|) dt``, i.e.
    ``sign(u)*F(|Q*u|)/|Q|``, so it is odd in ``u``, even in ``Q`` and
    ``P(0) = 0``.  Kept because it *is* the published form of the primitive;
    :func:`_v_branch3_struve` does not use it, because differencing it is the
    cancellation this module now avoids.
    """
    ua = np.asarray(u, dtype=np.float64)
    qf = abs(float(Q))
    return np.sign(ua) * _struve_F(qf * np.abs(ua)) / qf


#: Argument below which branch 3 differences ``F`` rather than ``Fc``.  Both
#: forms are algebraically identical; the one whose two operands are *smaller*
#: has the smaller absolute rounding error, and the operands cross at
#: ``F = Fc = 1/2`` at ``x = 0.3652``.  The threshold is set at 1 rather than
#: there because 1 is also where ``Fc`` starts to decay exponentially, so it is
#: the natural boundary between "no cancellation" and "cancellation that
#: matters", and because the two choices differ only by a factor 2 in an
#: absolute floor of order one ulp.
_V_FDIFF_MAX = 1.0


def _v_branch3_struve(Q: float, h: float, theta: ArrayLike) -> np.ndarray:
    """Branch 3 (``Q != 0``, ``ztil == 0``): exact cell integral via Struve L.

    ``V = pi*(P(b) - P(a))`` with ``a = theta - h/2``, ``b = theta + h/2``
    (paper 406 SI, ``set_V``).  This is exactly ``2*int_a^b K0(Q|t|) dt``,
    using ``int_0^x K0(t) dt = (pi*x/2)*(K0(x)*L_{-1}(x) + K1(x)*L_0(x))``.

    NO CANCELLATION AT LARGE ARGUMENT (the floor documented here before is
    gone where it mattered).  Differencing the published primitive directly is
    catastrophic at large ``Q*|theta|``: ``P(b)`` and ``P(a)`` both tend to
    ``pi/(2*Q)`` while their difference is ``O(h*K0(Q*theta))``, so the old
    form had an absolute floor of a few ulps of ``pi/Q`` -- measured 9.07e-15
    for a coplanar pair at ``W = 10``, ``L = 60``, ``q*d = 50``, where the true
    coupling is 1.92e-24, i.e. too large by 4.7e9.  That is why such a pair
    failed to decouple.  With ``F + Fc == 1`` the same difference is written,
    for a cell that does not straddle the origin, as

        V = (pi/Q) * ( Fc(Q*min(|a|,|b|)) - Fc(Q*max(|a|,|b|)) ),

    a difference of two exponentially small numbers, which is accurate in
    relative terms at every separation.  A cell that *does* straddle the
    origin keeps the ``F`` form, where the two contributions add instead of
    cancelling::

        V = (pi/Q) * ( F(Q*|a|) + F(Q*|b|) ),

    and so does any cell whose largest argument is below ``_V_FDIFF_MAX``,
    where ``F`` is the smaller of the two operands.  The three forms agree to a
    rounding error by construction, so the assembled ``V`` has no step.

    What remains: for ``Q*|theta| < 4``, ``Fc`` is itself formed as ``1 - F``
    (see the module docstring), so the difference above is still a difference
    of two ``O(1)`` numbers and ``V`` keeps the old absolute floor of a few
    ulps of ``pi/Q`` -- up to 3.3e-11 relative at ``Q = 0.1``, ``h = 1/149``.
    It is harmless, because at those arguments ``V`` is not exponentially
    small; ``tests/test_operators.py`` measures it rather than hiding it.

    Written through ``min``/``max`` of ``|a|, |b|``, the result is *exactly*
    even in ``theta`` and in ``Q``, as the kernel is.
    """
    th = np.asarray(theta, dtype=np.float64)
    shape = th.shape
    flat = np.atleast_1d(th).ravel()
    qf = abs(float(Q))
    hf = float(h)

    a = flat - 0.5 * hf
    b = flat + 0.5 * hf
    lo = qf * np.minimum(np.abs(a), np.abs(b))
    hi = qf * np.maximum(np.abs(a), np.abs(b))

    # The straddling and the near forms are the same expression up to the sign
    # of the F(lo) term -- F(hi) + F(lo) versus F(hi) - F(lo) -- so they share
    # one evaluation path, which keeps a whole self block (every cell near, the
    # diagonal straddling) off the fancy-index path.
    straddle = (a < 0.0) & (b > 0.0)
    f_form = straddle | (hi <= _V_FDIFF_MAX)

    def _f_branch(m_lo, m_hi, m_straddle):
        return _struve_F(m_hi) + np.where(m_straddle, 1.0, -1.0) * _struve_F(m_lo)

    if f_form.all():
        out = _f_branch(lo, hi, straddle)
    elif not f_form.any():
        out = _struve_Fc(lo) - _struve_Fc(hi)
    else:
        out = np.empty(flat.shape, dtype=np.float64)
        out[f_form] = _f_branch(lo[f_form], hi[f_form], straddle[f_form])
        c_form = ~f_form
        out[c_form] = _struve_Fc(lo[c_form]) - _struve_Fc(hi[c_form])

    return (np.pi / qf * np.asarray(out, dtype=np.float64)).reshape(shape)


def coulomb_V_entries(
    Q: float, h: float, theta: ArrayLike, ztil: float
) -> np.ndarray:
    """Coulomb-operator entries for one cell row, dispatching the three branches.

    Source: paper 406 SI; reference implementation ``set_V`` inside
    ``GNR_phi_LDOS_anisotropic.m``.

    Parameters
    ----------
    Q
        Dimensionless wave vector ``q * W_jp``.  The kernel is even in ``Q``
        (it depends on ``|q|``), so ``|Q|`` is used; this keeps ``q < 0`` legs
        of a dissymmetry sweep from producing NaNs.
    h
        Normalised cell width ``h_jp = 1/(L_jp - 1)``.
    theta
        Normalised separation array, any shape (see module docstring).
    ztil
        Normalised out-of-plane offset ``(z_j - z_jp)/W_jp``, scalar.

    Returns
    -------
    ndarray
        Real ``float64``, same shape as ``theta``.

    Notes
    -----
    Branch selection, exactly as specified:
    ``Q == 0`` -> branch 2 (exact, logarithmic kernel);
    ``Q != 0 and ztil == 0`` -> branch 3 (exact, modified Struve);
    ``Q != 0 and ztil != 0`` -> branch 1 (midpoint rule).
    """
    th = np.asarray(theta, dtype=np.float64)
    hf = float(h)
    zf = float(ztil)
    # The kernel depends on |Q| only; see the parameter note above.
    qf = abs(float(Q))

    if qf == 0.0:
        out = _v_branch2_q0(hf, th, zf)
    elif zf == 0.0:
        out = _v_branch3_struve(qf, hf, th)
    else:
        out = _v_branch1_midpoint(qf, hf, th, zf)
    # numpy collapses 0-d results to scalars; keep the declared ndarray return.
    return np.asarray(out, dtype=np.float64)


#: Largest modified-Struve argument at which the *bare 30-term series* of the
#: reference MATLAB is good to ~1e-9 (measured: <1e-15 at x <= 4, 4e-13 at 6,
#: 4e-10 at 8, 7e-8 at 10, 2e-4 at 15, 45% at 30).
#:
#: HISTORY, because the meaning of this constant changed.  It used to be a
#: *trust ceiling on the Coulomb block*: ``build_V`` and ``build_V_toeplitz``
#: raised a ``RuntimeWarning`` whenever a coplanar pair pushed branch 3 past
#: it, because the block was then numerical noise.  There is no such ceiling
#: any more -- branch 3 is accurate at every separation (see the module
#: docstring), so both builders are silent for every geometry and the warning
#: was removed rather than re-scoped: nothing untrusted remains to re-scope it
#: to.  The constant is kept, at the same value, because it still states where
#: the series alone stops being usable, and ``dispersion.py``, ``solver.py``
#: and ``tests/test_integration.py`` refer to it.  ``struve_L`` itself leaves
#: the series earlier, at ``_STRUVE_SERIES_MAX = 4``, which is where the switch
#: to ``scipy.special.modstruve`` is seamless to <=3e-16.
STRUVE_TRUSTED_MAX_ARG = 8.0


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
    """Dense Coulomb block ``V_{j j'}`` between ribbons ``j`` (rows) and ``j'``.

    Source: paper 406 SI, ``set_V``.  Fully vectorised: the ``(L_j, L_jp)``
    separation array is formed by broadcasting and
    :func:`coulomb_V_entries` is called exactly once.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].
    W_j, W_jp
        Ribbon widths [nm].  ``W_jp`` sets the normalisation.
    x_j, x_jp, z_j, z_jp
        Ribbon centres and heights [nm].
    theta_j, theta_jp
        Normalised node coordinates in ``[-1/2, 1/2]``, lengths ``L_j``/``L_jp``.
    h_jp
        Normalised cell width of ribbon ``j'``, ``1/(L_jp - 1)``.

    Returns
    -------
    ndarray
        ``(L_j, L_jp)`` ``complex128`` block.
    """
    th_j = np.asarray(theta_j, dtype=np.float64).reshape(-1, 1)
    th_jp = np.asarray(theta_jp, dtype=np.float64).reshape(1, -1)
    w_jp = float(W_jp)
    if w_jp == 0.0:
        raise ValueError("W_jp must be non-zero; it sets the normalisation.")

    sep = (float(x_j) - float(x_jp) + float(W_j) * th_j - w_jp * th_jp) / w_jp
    ztil = (float(z_j) - float(z_jp)) / w_jp
    Q = float(q) * w_jp
    return coulomb_V_entries(Q, float(h_jp), sep, ztil).astype(np.complex128)


def build_V_toeplitz(
    q: float,
    W: float,
    x_off: float,
    z_off: float,
    theta: np.ndarray,
    h: float,
) -> np.ndarray:
    """Fast Coulomb block for two ribbons of equal width on a common grid.

    Source: paper 406 SI, ``set_V``, exploiting that for ``W_j == W_jp == W``
    and a shared uniform grid the separation collapses to

        theta_llp = x_off/W + (l - l')*h,

    so the block is Toeplitz and only ``2L-1`` distinct kernel values are
    needed instead of ``L**2``.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].
    W
        Common ribbon width [nm].
    x_off, z_off
        Centre and height offsets ``x_j - x_jp`` and ``z_j - z_jp`` [nm].
    theta
        Shared uniform node array in ``[-1/2, 1/2]``, length ``L``, increasing.
    h
        Node spacing, which must match ``diff(theta)``.

    Returns
    -------
    ndarray
        ``(L, L)`` ``complex128`` block, equal to
        ``build_V(q, W, W, x_off, 0, z_off, 0, theta, theta, h)`` -- exactly
        for ``x_off == 0``, and otherwise to a few ulps (the two paths reach
        the same mathematical separation through different arithmetic, and
        branch 3 no longer has the ~1e-15 absolute cancellation floor that used
        to limit this agreement).

        Neither builder warns for any geometry: branch 3 is now accurate at
        every coplanar separation, so the ``RuntimeWarning`` that used to guard
        ``STRUVE_TRUSTED_MAX_ARG`` was removed.

    Raises
    ------
    ValueError
        If ``theta`` has fewer than two nodes, or is not an increasing uniform
        grid of spacing ``h`` -- the precondition for the Toeplitz structure.

    Notes
    -----
    CONTRACT DEVIATION (flagged, not silently fixed).  The specification says a
    symmetric Toeplitz is correct only for identical ribbons at ``ztil == 0``
    and to "raise otherwise".  Raising would make the ``x_off``/``z_off``
    arguments unusable, so this implementation instead assembles the *general*
    Toeplitz form ``toeplitz(first_column, first_row)``.  That is exact for
    every ``x_off`` and ``z_off``: the kernel is even in the separation, so the
    block is symmetric when ``x_off == 0`` and genuinely non-symmetric
    otherwise, and passing both the first column and the first row covers both
    cases.  Only the real structural precondition -- a shared uniform grid --
    is enforced by raising.  ``tests/test_operators.py`` verifies equality with
    :func:`build_V` for non-zero ``x_off`` and ``z_off``.
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    n = th.size
    if n < 2:
        raise ValueError(f"build_V_toeplitz needs at least 2 nodes; got L={n}.")

    hf = float(h)
    spacing = np.diff(th)
    if hf <= 0.0 or not np.allclose(spacing, hf, rtol=1e-10, atol=1e-14):
        raise ValueError(
            "build_V_toeplitz requires an increasing uniform grid with "
            f"spacing h={hf!r}; got diff(theta) in "
            f"[{spacing.min()!r}, {spacing.max()!r}]. Use build_V instead."
        )

    wf = float(W)
    if wf == 0.0:
        raise ValueError("W must be non-zero; it sets the normalisation.")

    # Separations for l - l' = -(L-1) ... (L-1): 2L-1 distinct values, kernel
    # evaluated once.  Grouped exactly as build_V groups it -- using theta
    # itself rather than (l-l')*h -- so that the first row and first column
    # reproduce the dense build bit for bit.
    x0 = float(x_off)
    th0 = th[0]
    col_sep = (x0 + wf * th - wf * th0) / wf   # l - l' = 0, 1, ..., L-1
    row_sep = (x0 + wf * th0 - wf * th) / wf   # l - l' = 0, -1, ..., -(L-1)
    sep = np.concatenate((row_sep[:0:-1], col_sep))
    Q = float(q) * wf
    ztil = float(z_off) / wf
    vals = coulomb_V_entries(Q, hf, sep, ztil)

    first_col = vals[n - 1:]          # l - l' = 0, 1, ..., L-1
    first_row = vals[n - 1::-1]       # l - l' = 0, -1, ..., -(L-1)
    return toeplitz(first_col, first_row).astype(np.complex128)


# ---------------------------------------------------------------------------
# Differential operator
# ---------------------------------------------------------------------------


def _build_D_parts(
    q: float, W: float, h: float, L: int, f: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``D`` into its ``eta_perp`` matrix and its ``eta_par`` diagonal.

    Returns ``(A, b_diag)`` such that
    ``D = eta_perp*A + eta_par*diag(b_diag)``.

    ``A`` is written in flux form: with the face weight between nodes ``k`` and
    ``k+1`` given by ``w[k] = (f[k] + f[k+1]) / (2*h**2)``,

        A[l, l-1] = w[l-1],   A[l, l+1] = w[l],
        A[l, l]   = -(w[l-1] + w[l]),   with w[-1] = w[L-1] = 0.

    The ghost weights vanishing at the two ends *is* the vanishing-normal-
    current boundary condition, and reproduces both specified forms: with
    ``f = None`` (all ones) the interior rows become
    ``[1, -2, 1]/h**2`` and the boundary rows ``[-1, 1]/h**2``; with ``f``
    given the interior rows become the paper-406 symmetric form
    ``[(f[l-1]+f[l]), -(f[l-1]+2f[l]+f[l+1]), (f[l]+f[l+1])]/(2h**2)``.

    The specification gives no boundary rows for the inhomogeneous case; the
    zero-ghost-flux extension used here is the one that keeps every row summing
    to ``-eta_par*Q**2*f[l]`` (charge conservation) for arbitrary ``f``.
    """
    n = int(L)
    if n < 2:
        raise ValueError(f"build_D needs at least 2 nodes; got L={n}.")
    hf = float(h)
    if hf == 0.0:
        raise ValueError("h must be non-zero.")

    if f is None:
        fv = np.ones(n, dtype=np.float64)
    else:
        fv = np.asarray(f, dtype=np.float64).ravel()
        if fv.size != n:
            raise ValueError(
                f"filling function f has length {fv.size}, expected L={n}."
            )

    # Face weights (length L-1). For f = None this is exactly 1/h**2.
    w = (fv[:-1] + fv[1:]) / (2.0 * hf * hf)

    A = np.zeros((n, n), dtype=np.complex128)
    off = np.arange(n - 1)
    A[off, off + 1] = w
    A[off + 1, off] = w
    w_minus = np.concatenate(([0.0], w))   # w[l-1], zero ghost flux at l = 0
    w_plus = np.concatenate((w, [0.0]))    # w[l],   zero ghost flux at l = L-1
    diag = np.arange(n)
    A[diag, diag] = -(w_minus + w_plus)

    big_q = float(q) * float(W)
    b_diag = -(big_q * big_q) * fv
    return A, b_diag


def build_D(
    q: float,
    W: float,
    h: float,
    L: int,
    eta_perp: Scalar,
    eta_par: Scalar,
    f: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Anisotropic differential operator ``D`` on one ribbon.

    Source: paper 456 Eq. (19); reference implementation ``set_D``.  With
    ``Q = q*W`` and ``h`` the normalised node spacing, interior rows
    ``1 <= l <= L-2`` read

        D[l, l-1] =  eta_perp/h**2
        D[l, l]   = -2*eta_perp/h**2 - eta_par*Q**2
        D[l, l+1] =  eta_perp/h**2

    and the two boundary rows enforce vanishing normal current at the ribbon
    edges

        D[0, 0]     = -eta_perp/h**2 - eta_par*Q**2,  D[0, 1]     = eta_perp/h**2
        D[L-1, L-1] = -eta_perp/h**2 - eta_par*Q**2,  D[L-1, L-2] = eta_perp/h**2

    Parameters
    ----------
    q
        Guided wave vector [1/nm].
    W
        Ribbon width [nm].
    h
        Normalised node spacing ``1/(L-1)``.
    L
        Number of nodes.
    eta_perp, eta_par
        Complex dimensionless conductivity ratios across / along ``y``.
    f
        Optional filling function of length ``L`` (1 inside the ribbon, 0
        outside) for inhomogeneous doping.  ``None`` means all ones.  When
        given, the paper-406 symmetric form is used (see
        :func:`_build_D_parts`).

    Returns
    -------
    ndarray
        ``(L, L)`` ``complex128`` operator.

    Notes
    -----
    When ``eta_perp == eta_par`` the operator is exactly ``eta * D_geom``, so
    that case is computed as a scaled copy of :func:`build_D_geom`.  This is
    not just an optimisation: it makes the isotropic identity hold bit for bit
    rather than to within a rounding error.
    """
    A, b_diag = _build_D_parts(q, W, h, L, f)
    e_perp = complex(eta_perp)
    e_par = complex(eta_par)
    diag = np.arange(int(L))

    if e_perp == e_par:
        A[diag, diag] += b_diag
        A *= e_perp
        return A

    A *= e_perp
    A[diag, diag] += e_par * b_diag
    return A


def build_D_geom(
    q: float,
    W: float,
    h: float,
    L: int,
    f: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Eta-free (geometric) differential operator: ``tridiag/h**2 - (q*W)**2 I``.

    Source: paper 456 Eq. (19) with ``eta_perp = eta_par = 1``; reference
    implementation ``set_D``.  Used by the isotropic eigenmode fast path, where
    the generalised problem ``D_geom v = zeta V^{-1} v`` is eta-independent.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].
    W
        Ribbon width [nm].
    h
        Normalised node spacing ``1/(L-1)``.
    L
        Number of nodes.
    f
        Optional filling function of length ``L``; ``None`` means all ones.

    Returns
    -------
    ndarray
        ``(L, L)`` ``complex128`` operator, equal to
        ``build_D(q, W, h, L, 1.0, 1.0, f)``.
    """
    A, b_diag = _build_D_parts(q, W, h, L, f)
    diag = np.arange(int(L))
    A[diag, diag] += b_diag
    return A
