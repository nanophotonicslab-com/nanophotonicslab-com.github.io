"""Layered-substrate electrostatic response for the quasistatic ribbon FD solver.

Two public pieces of physics live here:

1. ``reflection_coefficient`` -- the quasistatic (electrostatic) reflection
   coefficient r(Q, omega) of an arbitrary stack of uniaxially anisotropic
   layers, each optionally carrying a conducting sheet at its top interface.
   Built by the bottom-up recursion of the nanodisk manuscript, SI section S1B.

2. ``build_Vbar`` -- the reflected ("image") Coulomb block that adds to the
   direct block produced by ``operators.build_V``, obtained by numerically
   inverting the 2D Fourier representation of the reflected potential.

UNITS. eV-nm-fs Gaussian throughout (see ``units.py``):
Q and q in 1/nm, omega in 1/fs, 2D sheet conductivity in nm/fs, lengths in nm.

=============================================================================
CONVENTION HAZARD -- READ BEFORE USING ``build_Vbar`` WITH A NON-ZERO d_1.
=============================================================================
The standoff between the ribbon plane and the first substrate interface can be
expressed in two places, and the formulas as specified encode it in BOTH:

  * ``reflection_coefficient`` multiplies r[1,2] by exp(-2*Qhat_1*d_1), with
    d_1 = ``layers[0].d_nm``;
  * ``build_Vbar`` multiplies the integrand by exp(-Q*(z_j + z_jp)), the image
    combination for two charges at heights z_j, z_jp above the interface.

Put the standoff in EXACTLY ONE of them, never both, or it is double counted:

  convention A (recommended): ``layers[0].d_nm = 0.0`` and z_j, z_jp measured
      from the first interface, i.e. z = standoff;
  convention B: ``layers[0].d_nm = standoff`` and z_j = z_jp = 0.

For an isotropic layer 1 (Qhat_1 = Q), the two are algebraically identical and
``tests/test_substrate.py::test_standoff_conventions_A_and_B_agree`` asserts it.
=============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Layer",
    "reflection_coefficient",
    "build_Vbar",
    "default_Qmax",
    "HBAR_EV_FS",
]

# --------------------------------------------------------------------------
# units.py is written by another agent and may not exist yet; fall back to the
# documented literal so this module is importable stand-alone. The fallback
# value is cross-checked against units.py in
# tests/test_substrate.py::test_hbar_matches_units_module when that module
# becomes importable.
# --------------------------------------------------------------------------
try:  # pragma: no cover - exercised only once units.py lands
    from .units import HBAR_EV_FS  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 - ImportError or relative-import-without-package
    try:  # pragma: no cover
        from units import HBAR_EV_FS  # type: ignore[import-not-found,no-redef]
    except Exception:  # noqa: BLE001
        HBAR_EV_FS = 0.6582119569  # hbar [eV fs]

_C = np.complex128
ArrayLike = Union[float, complex, np.ndarray]


# ==========================================================================
# Layer description
# ==========================================================================
@dataclass
class Layer:
    """One uniaxial layer of the substrate stack.

    Attributes
    ----------
    eps_par
        In-plane permittivity eps_parallel (dimensionless, complex allowed).
    eps_perp
        Out-of-plane permittivity. ``None`` means isotropic, i.e.
        eps_perp = eps_par.
    d_nm
        Layer thickness [nm]. Ignored for the LAST layer of a stack, which is
        always treated as the semi-infinite terminating half-space. For the
        FIRST layer this is the ribbon-plane standoff -- see the
        CONVENTION HAZARD note in the module docstring.
    sigma_sheet
        2D conducting sheet [nm/fs, Gaussian] sitting at the TOP interface of
        this layer. ``layers[0].sigma_sheet`` therefore never enters the
        recursion: there is no interface above the first layer.

    Notes
    -----
    Nanodisk manuscript, SI section S1B (layered quasistatic response).
    """

    eps_par: complex = 1.0
    eps_perp: Optional[complex] = None
    d_nm: float = 0.0
    sigma_sheet: complex = 0.0

    @property
    def eps_perp_eff(self) -> complex:
        """Out-of-plane permittivity, resolving ``eps_perp=None`` to isotropic."""
        return self.eps_par if self.eps_perp is None else self.eps_perp

    @property
    def eps_geo(self) -> complex:
        """sqrt(eps_par * eps_perp): the combination entering kappa (SI S1B).

        BRANCH.  The square root is taken on the *average-argument* branch,

            eps_geo = sqrt(|eps_par eps_perp|) exp(i (arg eps_par + arg eps_perp)/2)

        and NOT on numpy's principal branch.  The two agree whenever
        ``|arg eps_par + arg eps_perp| <= pi`` and differ by a sign exactly when
        the product's argument wraps past pi -- which is the generic case for a
        metal, where Re eps < 0.

        This matters, and the principal branch is simply wrong there.  For an
        ISOTROPIC layer eps_geo must reduce to eps_par identically, since kappa
        is the quasistatic Fresnel coefficient
        ``(eps_1 - eps_2)/(eps_1 + eps_2)``.  With ``np.sqrt`` and
        ``eps_par = -20 + 1j`` (a Drude metal below its plasma frequency, i.e.
        the whole point of a plasmonic substrate), ``sqrt(eps_par**2)`` returns
        ``+20 - 1j``: the sign of the permittivity is flipped, r comes out
        -0.9050 + 0.0045j instead of -1.1050 - 0.0055j, and the ``|r| > 1``
        surface-plasmon enhancement disappears.  The isotropic case is short
        circuited below so that it is not merely correct but bit-exact.
        """
        ep = _C(self.eps_par)
        ez = _C(self.eps_perp_eff)
        if ez == ep:
            # Isotropic: sqrt(eps**2) == eps exactly, on any sane branch.
            return complex(ep)
        if ep == 0.0 or ez == 0.0:
            return 0.0 + 0.0j
        mag = float(np.sqrt(np.abs(ep) * np.abs(ez)))
        phase = 0.5 * (float(np.angle(ep)) + float(np.angle(ez)))
        return complex(mag * np.exp(1j * phase))

    @property
    def aniso_factor(self) -> complex:
        """sqrt(eps_par / eps_perp): the anisotropic decay rescaling of Q (SI S1B)."""
        return complex(np.sqrt(_C(self.eps_par) / _C(self.eps_perp_eff)))


# ==========================================================================
# Reflection coefficient
# ==========================================================================
def reflection_coefficient(
    Q_inv_nm: ArrayLike,
    E_ev: ArrayLike,
    layers: Sequence[Layer],
    eps1: complex = 1.0,
) -> Union[complex, np.ndarray]:
    """Quasistatic reflection coefficient r(Q, omega) of a layered stack.

    Implements the bottom-up recursion of the nanodisk manuscript, SI section
    S1B. Layers are indexed l = 1..L from the TOP; layer 1 is the half-space
    containing the ribbons (usually vacuum) and layer L is the semi-infinite
    terminating medium. Starting from r[L, L+1] = 0 and working upwards,

        Qhat_l    = sqrt(eps_par_l / eps_perp_l) * Q
        kappa_l   = (eg_l - eg_{l+1}) / (eg_l + eg_{l+1}),
                      eg_l = sqrt(eps_par_l * eps_perp_l)
        sigt_l    = 4*pi*sigma_sheet_{l,l+1} / (1j*omega*(eg_l + eg_{l+1}))
        xi_l      = sigt_l * Q * (r[l+1, l+2] + 1)
        r[l, l+1] = exp(-2*Qhat_l*d_l) * (kappa_l + r[l+1,l+2] + xi_l)
                                       / (1 + r[l+1,l+2]*kappa_l - xi_l)

    and r(Q, omega) = r[1, 2].

    UNITS AUDIT of the sheet term. The source manuscript is SI, where

        sigt_SI = sigma_SI / (1j * omega * eps0 * (eg_l + eg_{l+1})).

    In the Gaussian eV-nm-fs convention used by this package 1/(4*pi*eps0) -> 1,
    i.e. 1/eps0 -> 4*pi, so

        sigt_G  = 4*pi*sigma / (1j * omega * (eg_l + eg_{l+1})).

    Dimensions close: [sigma] = nm/fs and [omega] = 1/fs give [sigt] = nm, so
    xi = sigt*Q is dimensionless with Q in 1/nm. The 4*pi is confirmed
    independently by the pole of r for a single interface carrying a sheet,
    which lands on the textbook graphene plasmon dispersion
    Q* = 1j*omega*(eps1 + eps2) / (4*pi*sigma)
    (Jablan et al., PRB 80, 245435 (2009), eq. 4, Gaussian form); see
    tests/test_substrate.py::test_single_interface_sheet_pole_is_textbook_dispersion.

    Parameters
    ----------
    Q_inv_nm
        In-plane wavevector magnitude Q = sqrt(Qx^2 + q^2) [1/nm]. Array-valued
        input is broadcast against ``E_ev`` by the usual numpy rules, so sweep
        a grid with e.g. ``Q[:, None]`` and ``E[None, :]``.
    E_ev
        Photon energy [eV]; omega = E_ev / HBAR_EV_FS [1/fs]. Only used by the
        sheet term, so E_ev = 0 is accepted for a purely dielectric stack.
    layers
        Stack from top to bottom. ``layers[0]`` is the medium containing the
        ribbons. A stack of fewer than two layers has no interface and returns 0.
    eps1
        Permittivity of the ribbon-containing half-space. REDUNDANT with
        ``layers[0].eps_par``, which is what the recursion actually uses (the
        mandated single-interface test fixes that reading). Retained for
        signature compatibility. An explicitly supplied value (anything other
        than the 1.0 default) that disagrees with ``layers[0].eps_par`` raises
        a ``UserWarning`` rather than silently taking one of the two.

    Returns
    -------
    complex or ndarray
        r(Q, omega). Scalar inputs give a scalar; otherwise the broadcast shape.

    Notes
    -----
    Sign convention: a perfect mirror (eps_2 -> infinity) gives r -> -1, as
    stated explicitly in the manuscript.

    Branch cuts: the principal square root is used for both sqrt(eps_par*eps_perp)
    and sqrt(eps_par/eps_perp). For passive media in this package's
    exp(-1j*omega*t) convention (Im eps > 0) that is the physical branch,
    Re(Qhat) > 0, so exp(-2*Qhat*d) decays. A stack that violates this (e.g. a
    hand-built gain medium, or an anisotropic layer whose eps_par and eps_perp
    have widely separated phases) is reported through a ``UserWarning``; the
    value is still returned as the formula gives it.
    """
    layers = list(layers)
    if len(layers) == 0:
        raise ValueError("layers must contain at least one Layer")

    Q = np.asarray(Q_inv_nm, dtype=np.float64)
    if np.any(Q < 0.0):
        raise ValueError("Q_inv_nm must be non-negative (it is |Q|)")

    # eps1 bookkeeping: flag, never silently resolve, a contradiction. The
    # default eps1=1.0 is treated as "not supplied" so that an ordinary stack
    # whose layer 1 is not vacuum does not warn on every call.
    if complex(eps1) != 1.0 and not np.isclose(
        complex(eps1), complex(layers[0].eps_par), rtol=1e-12, atol=0.0
    ):
        warnings.warn(
            f"reflection_coefficient: eps1={eps1!r} disagrees with "
            f"layers[0].eps_par={layers[0].eps_par!r}; the recursion uses "
            f"layers[0], eps1 is ignored.",
            UserWarning,
            stacklevel=2,
        )

    if len(layers) == 1:
        # No interface: nothing to reflect from.
        out = np.zeros(np.broadcast(Q, np.asarray(E_ev)).shape, dtype=_C)
        return complex(out[()]) if out.ndim == 0 else out

    needs_omega = any(complex(lay.sigma_sheet) != 0.0 for lay in layers[1:])
    if needs_omega:
        omega = np.asarray(E_ev, dtype=np.float64) / HBAR_EV_FS  # [1/fs]
        if np.any(omega == 0.0):
            raise ValueError(
                "E_ev = 0 with a non-zero sigma_sheet: the sheet term "
                "4*pi*sigma/(1j*omega*(...)) diverges at omega = 0."
            )
    else:
        omega = None

    # r[L, L+1] = 0, broadcast to the output shape.
    shape = np.broadcast(Q, np.asarray(E_ev, dtype=np.float64)).shape
    r = np.zeros(shape, dtype=_C)

    # Upward recursion over interfaces. 0-indexed i labels the interface
    # between layers[i] (upper) and layers[i+1] (lower), i.e. the 1-indexed
    # interface (l, l+1) with l = i+1.
    for i in range(len(layers) - 2, -1, -1):
        upper, lower = layers[i], layers[i + 1]
        eg_u = _C(upper.eps_geo)
        eg_l = _C(lower.eps_geo)
        denom_eg = eg_u + eg_l
        if denom_eg == 0.0:
            raise ValueError(
                f"sqrt(eps_par*eps_perp) sums to zero across interface {i}; "
                "kappa and the sheet term are both undefined there."
            )
        kappa = (eg_u - eg_l) / denom_eg

        Qhat = _C(upper.aniso_factor) * Q
        if np.any(np.real(Qhat) < 0.0) and float(upper.d_nm) != 0.0:
            warnings.warn(
                f"layer {i}: Re(sqrt(eps_par/eps_perp)) < 0, so "
                "exp(-2*Qhat*d) grows with thickness. Check the branch of "
                "your permittivities.",
                UserWarning,
                stacklevel=2,
            )

        # Sheet at the top interface of the LOWER layer == this interface.
        sigma = _C(lower.sigma_sheet)
        if sigma != 0.0:
            # SI:       sigt = sigma / (1j*omega*eps0*denom_eg)
            # Gaussian: sigt = 4*pi*sigma / (1j*omega*denom_eg)      [nm]
            sigt = 4.0 * np.pi * sigma / (1j * omega * denom_eg)
            xi = sigt * Q * (r + 1.0)
        else:
            xi = np.zeros_like(r)

        decay = np.exp(-2.0 * Qhat * float(upper.d_nm))
        r = decay * (kappa + r + xi) / (1.0 + r * kappa - xi)

    return complex(r[()]) if r.ndim == 0 else r


# ==========================================================================
# Reflected Coulomb block
# ==========================================================================
def default_Qmax(q: float, z_sum_nm: float, dx_max_nm: float) -> float:
    """Default upper limit of the Qx quadrature for ``build_Vbar``.

    As specified in the task physics block:

        Qmax = max(50/max(z_j+z_jp, 0.05), 50*|q|, 200/max(|dx|_max, 0.1))

    CONCERN (implemented as given, not altered). The third term runs the
    "wrong" way: it makes Qmax *smaller* for larger separations. That is
    harmless -- and in fact helpful for resolution -- because ``build_Vbar``
    passes the maximum |dx| over the whole node pair matrix, which is of order
    the ribbon width, so the first two terms dominate in practice. It does
    mean that for a single very small |dx| the formula inflates Qmax to 2000
    while NQ stays fixed, leaving the integrand's peak near Qx ~ q
    unresolved. ``build_Vbar`` therefore emits a ``UserWarning`` when the
    resulting step dQ = Qmax/(NQ-1) fails to resolve min(|q|, 1/|dx|_max),
    instead of silently returning a wrong number.

    Parameters
    ----------
    q
        Guided wavevector along the ribbon axis [1/nm].
    z_sum_nm
        z_j + z_jp [nm], the image-pair height entering exp(-Q*(z_j+z_jp)).
    dx_max_nm
        Largest physical node separation |dx| [nm] in the block being built.

    Returns
    -------
    float
        Qmax [1/nm].
    """
    return float(
        max(
            50.0 / max(float(z_sum_nm), 0.05),
            50.0 * abs(float(q)),
            200.0 / max(abs(float(dx_max_nm)), 0.1),
        )
    )


def _eval_r(r_of_Q: Callable[[np.ndarray], ArrayLike], Q: np.ndarray) -> np.ndarray:
    """Evaluate ``r_of_Q`` on the whole Q grid, falling back to a scalar loop."""
    try:
        r = np.asarray(r_of_Q(Q), dtype=_C)
        if r.shape == ():
            r = np.broadcast_to(r, Q.shape)
        elif r.shape != Q.shape:
            r = np.broadcast_to(r, Q.shape)
        return np.ascontiguousarray(r, dtype=_C)
    except Exception:  # noqa: BLE001 - any non-vectorised callable
        warnings.warn(
            "build_Vbar: r_of_Q is not vectorised over Q; falling back to a "
            "Python-level loop. Supply a numpy-aware callable for speed.",
            UserWarning,
            stacklevel=3,
        )
        return np.array([_C(r_of_Q(float(qq))) for qq in Q], dtype=_C)


def build_Vbar(
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
    r_of_Q: Callable[[np.ndarray], ArrayLike],
    Qmax: Optional[float] = None,
    NQ: int = 4000,
) -> np.ndarray:
    """Reflected (image) Coulomb block between ribbons j and j'.

    The reflected potential in the 2D Fourier plane carries r(Q, omega) and the
    image height combination exp(-Q*(z_j + z_jp)) in place of the direct
    block's exp(-Q*|z_j - z_jp|) (nanodisk manuscript, SI section S1B):

        Vbar(dx) = 2 * integral_0^inf dQx [r(Q)/Q] * exp(-Q*(z_j + z_jp))
                                          * cos(Qx*dx),     Q = sqrt(Qx^2 + q^2)

    with dx = x_j + W_j*theta_j_l - x_jp - W_jp*theta_jp_lp the PHYSICAL node
    separation [nm]. The returned entries are multiplied by ``h_jp`` so that
    the source cell is integrated by the midpoint rule, matching branch 1 of
    ``operators.build_V`` exactly; the two blocks then multiply the same charge
    vector.

    Validation identity (used by the test file). For a constant r = r0,

        Vbar(dx) = r0 * 2 * h_jp * K0(|q| * sqrt(dx^2 + (z_j + z_jp)^2))

    which follows from
        2*int_0^inf cos(Qx*dx)*exp(-Q*z)/Q dQx
          = int dy exp(-1j*q*y)/sqrt(dx^2 + y^2 + z^2)
          = 2*K0(|q|*sqrt(dx^2 + z^2))
    (Gradshteyn & Ryzhik 3.961.2 / the Lipschitz-Hankel integral). At z = 0
    this reduces to the mandated r0*2*h_jp*K0(|q||dx|).

    Parameters
    ----------
    q
        Guided wavevector [1/nm]. Must be non-zero: see Raises.
    W_j, W_jp
        Ribbon widths [nm].
    x_j, x_jp
        Ribbon centres [nm].
    z_j, z_jp
        Ribbon heights above the first substrate interface [nm]. Their SUM
        enters, through the image factor exp(-Q*(z_j + z_jp)); it must be
        non-negative. See the CONVENTION HAZARD note in the module docstring
        before combining a non-zero z with a non-zero ``layers[0].d_nm``.
    theta_j, theta_jp
        Normalised node coordinates in [-1/2, 1/2], lengths L_j and L_jp.
    h_jp
        Normalised source cell width 1/(L_jp - 1).
    r_of_Q
        Callable Q [1/nm] -> complex reflection coefficient, ideally vectorised
        over a numpy array. Typically
        ``lambda Q: reflection_coefficient(Q, E_ev, layers)``.
    Qmax
        Quadrature cutoff [1/nm]; ``None`` uses :func:`default_Qmax`.
    NQ
        Number of uniform quadrature nodes on [0, Qmax]. The quadrature is
        ``numpy.trapz`` on that uniform grid, applied through an equivalent
        weight vector so the sum can be done as a matrix product.

    Returns
    -------
    ndarray
        Complex ``(L_j, L_jp)`` block.

    Raises
    ------
    ValueError
        If ``q == 0``. The integrand then behaves as r(0)/Qx near Qx = 0 and
        the integral is logarithmically divergent -- exactly the divergence of
        the direct kernel 2*K0(|q||dx|) as q -> 0. The divergence is an additive
        constant that cancels in the response of a charge-neutral ribbon, so
        the solver must approach q = 0 with a small finite q rather than
        evaluate it here.
        Also if ``z_j + z_jp < 0``, where exp(-Q*(z_j+z_jp)) grows and the
        integral diverges.

    Notes
    -----
    Convergence degrades as the standoff z_j + z_jp -> 0: the integrand then
    decays only as 1/Q and the truncated oscillatory tail leaves a relative
    error of order 1/(Qmax*|dx|). Callers should keep d_1 (equivalently the
    standoff) small but FINITE. At z_j + z_jp >= 1 nm the exponential cuts the
    integrand off well inside Qmax and the trapezoid rule is accurate to
    ~1e-12; at z_j + z_jp = 0 it is accurate to ~1e-3 at best with NQ = 4000.

    Implementation. The l, l' coupling is factorised exactly rather than
    materialising an (L_j, L_jp, NQ) array:

        cos(Qx*dx_{l,l'}) = Re[ exp(1j*Qx*D0) * exp(1j*Qx*a_l)
                                             * conj(exp(1j*Qx*b_l')) ]

    with a_l, b_l' the node coordinates centred on their own ribbons and D0 the
    centre-to-centre offset, so the quadrature sum becomes two complex matrix
    products of shape (L_j, NQ) @ (NQ, L_jp). Centring keeps the trig arguments
    bounded by Qmax*W/2 instead of Qmax*|x|, which preserves accuracy for
    ribbons far from the origin. The Qx grid is processed in blocks to bound
    peak memory independently of L and NQ.
    """
    q = float(q)
    if q == 0.0:
        raise ValueError(
            "build_Vbar: q = 0 makes the reflected kernel logarithmically "
            "divergent (integrand ~ r(0)/Qx as Qx -> 0). Use a small finite q; "
            "see the Raises section of the docstring."
        )

    NQ = int(NQ)
    if NQ < 2:
        raise ValueError(f"NQ must be at least 2, got {NQ}")

    th_j = np.asarray(theta_j, dtype=np.float64).ravel()
    th_jp = np.asarray(theta_jp, dtype=np.float64).ravel()
    if th_j.size == 0 or th_jp.size == 0:
        raise ValueError("theta_j and theta_jp must be non-empty")

    z_sum = float(z_j) + float(z_jp)
    if z_sum < 0.0:
        raise ValueError(
            f"z_j + z_jp = {z_sum} < 0: exp(-Q*(z_j+z_jp)) grows and the "
            "reflected integral diverges. Ribbons must sit above the interface."
        )

    # Physical node positions [nm] and the node separation matrix dx = A - B.
    A = float(x_j) + float(W_j) * th_j      # (L_j,)
    B = float(x_jp) + float(W_jp) * th_jp   # (L_jp,)

    # Centre each ribbon on itself; D0 carries the centre-to-centre offset.
    A0 = 0.5 * (float(A.min()) + float(A.max()))
    B0 = 0.5 * (float(B.min()) + float(B.max()))
    a = A - A0
    b = B - B0
    D0 = A0 - B0
    dx_max = abs(D0) + float(np.abs(a).max()) + float(np.abs(b).max())

    if Qmax is None:
        Qmax = default_Qmax(q, z_sum, dx_max)
    Qmax = float(Qmax)
    if Qmax <= 0.0:
        raise ValueError(f"Qmax must be positive, got {Qmax}")

    # Resolution diagnostic. The integrand r(Q)/Q*exp(-Q*z_sum) has a feature
    # of width ~|q| near Qx = 0 (where Q = sqrt(Qx^2+q^2) turns over); missing
    # it is by far the dominant error source and is what the inflated Qmax of
    # default_Qmax causes at small |dx|_max. Measured at finite standoff:
    # dQ <= |q|/4 gives ~1e-12 relative, dQ ~ |q| gives ~1e-3, dQ ~ 5|q| gives
    # tens of percent. The threshold is 0.3 so that dQ = |q|/4 exactly (which
    # the default Qmax hits at NQ = 4000) does not trip it.
    #
    # There is deliberately NO criterion on the cos(Qx*dx) oscillation: with a
    # decaying envelope the trapezoid rule stays accurate to ~1e-17 absolute
    # even at dQ*|dx|_max > 2, because the aliased spectrum of the analytic
    # envelope is negligible there. Entries whose true value is itself
    # exponentially small (K0 of a large argument) carry a large RELATIVE but
    # a negligible ABSOLUTE error.
    dQ = Qmax / (NQ - 1)

    Qx = np.linspace(0.0, Qmax, NQ)
    Q = np.sqrt(Qx * Qx + q * q)            # >= |q| > 0, so r/Q is finite
    r_vals = _eval_r(r_of_Q, Q)

    L_j = th_j.size
    L_jp = th_jp.size

    # A transparent stack (r == 0 on the whole grid, e.g. layers of vacuum)
    # has an identically zero integrand. Short-circuit it: the quadrature
    # would return exact zeros anyway, and there is no resolution to warn
    # about. This is the common no-substrate path through solver.RibbonSystem.
    if not np.any(r_vals):
        return np.zeros((L_j, L_jp), dtype=_C)

    if dQ > 0.3 * abs(q):
        warnings.warn(
            f"build_Vbar: quadrature step dQ={dQ:.3g} does not resolve the "
            f"integrand feature of width |q|={abs(q):.3g} near Qx=0 "
            f"(Qmax={Qmax:.3g}, NQ={NQ}). Increase NQ or reduce Qmax.",
            UserWarning,
            stacklevel=2,
        )
    if z_sum > 0.0 and Qmax * z_sum < 20.0:
        warnings.warn(
            f"build_Vbar: Qmax*(z_j+z_jp)={Qmax * z_sum:.3g} < 20, so the "
            "exp(-Q*(z_j+z_jp)) tail is truncated while still significant. "
            "Increase Qmax (default_Qmax gives 50/(z_j+z_jp)).",
            UserWarning,
            stacklevel=2,
        )

    # Integrand envelope f(Qx) = r(Q)/Q * exp(-Q*(z_j+z_jp)).
    f = r_vals / Q
    if z_sum != 0.0:
        f = f * np.exp(-Q * z_sum)

    # Trapezoid weights on the uniform grid: identical to numpy.trapz(y, Qx),
    # asserted in tests/test_substrate.py::test_trapz_weight_vector_matches_numpy.
    w = np.full(NQ, dQ, dtype=np.float64)
    w[0] = 0.5 * dQ
    w[-1] = 0.5 * dQ
    g = f * w

    acc = np.zeros((L_j, L_jp), dtype=_C)

    # Block over Qx so peak memory is ~4*(L_j+L_jp)*nb complex128 regardless
    # of L and NQ. 64 MB budget.
    bytes_per_col = 16 * 4 * (L_j + L_jp)
    nb = int(max(256, min(NQ, (64 << 20) // max(bytes_per_col, 1))))

    for s in range(0, NQ, nb):
        e = min(s + nb, NQ)
        Qb = Qx[s:e]
        gb = g[s:e]
        Eu = np.exp(1j * np.outer(a, Qb))        # (L_j, nb)
        Fv = np.exp(1j * np.outer(b, Qb))        # (L_jp, nb)
        zb = np.exp(1j * (D0 * Qb))              # (nb,)
        # sum_k g_k*cos(Qx_k*dx) = 0.5*sum_k [ g z Eu conj(Fv) + conj(...) ]
        acc += 0.5 * ((Eu * (gb * zb)) @ Fv.conj().T)
        acc += 0.5 * ((Eu.conj() * (gb * zb.conj())) @ Fv.T)

    return 2.0 * float(h_jp) * acc
