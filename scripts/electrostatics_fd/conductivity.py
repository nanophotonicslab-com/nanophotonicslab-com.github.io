"""Sheet conductivities of doped graphene and thin metal films  [nm/fs].

Everything here is local in the sense of the quasistatic solver: a function of
photon energy E [eV] (and, for the drift-biased case, of the guided wave vector
Q [1/nm]), returning a 2D conductivity in the Gaussian eV-nm-fs units of
:mod:`units`.  Sources:

* "paper 406" = J. D. Cox, R. Yu & F. J. Garcia de Abajo, *Analytical
  description of the nonlinear plasmonic response in nanographene*,
  Phys. Rev. B 96, 045442 (2017) -- linear Drude Eq. (8), second-order
  Eqs. (11)-(12), third-order Eqs. (14)-(15), cascaded Eqs. (15a)-(15d).
* "paper 456" -- drift-biased anisotropic nonlocal conductivity, Eqs. (12a)-(12b)
  and the thermal Drude weight quoted after Eq. (11).
* "paper 430" -- thin-film mapping of a local dielectric function onto a sheet
  conductivity.
* G. W. Hanson, J. Appl. Phys. 103, 064302 (2008) -- T = 0 interband term.

FLAGGED CONCERNS (items 1 and 4 are RESOLVED, see below; 2, 3 and 5 stand.
Numbering is kept so that cross-references elsewhere still point correctly.)
-------------------------------------------------------------------
1. RESOLVED 2026-11-03 -- the two candidate interband prefactors were never
   alternatives.  The *log* prefactor is ``e^2/(4 pi hbar) = sigma0/pi``, and
   the *absorptive part* therefore saturates at the universal
   ``sigma0 = e^2/(4 hbar)``: the pi comes from the branch cut of the logarithm
   (``log`` jumps by ``i pi`` across E = 2 E_F), not from a convention choice.
   The T = 0 branch of :func:`graphene_interband_T0` used to carry
   ``1j * sigma0 * log(...)``, i.e. one factor pi too much; it now carries
   ``1j * (sigma0/pi) * log(...)``, which agrees with Hanson, with the existing
   NPLab port in ``src/lib/graphene-conductivity.ts``, and with the finite-T
   branch of :func:`graphene_local_rpa` (whose step height ``sigma0`` was
   already right and is unchanged).  ``SIGMA0_NM_FS`` itself is unchanged -- it
   is the correct universal saturation value.  :func:`graphene_local_rpa` is
   consequently continuous across its T_K = 1 switch to O(gamma/(E + 2 mu)):
   at E = 1.2 eV, E_F = 0.5 eV, gamma = 0.02 eV the interband parts differ by
   1.9e-3 relative and the full conductivity by 2.9e-3.
2. RESOLVED 2026-11-03 (MATLAB parity campaign) -- the old note called the
   finite-T closed form an approximation that "departs from the numerical
   carrier-number solution by about 0.5 % at 300 K".  That was a
   misdiagnosis: the two routes answer *different questions*.
   :func:`drift_chemical_potential_numeric` is a faithful port of the
   reference's own drift chemical potential, whose root condition is
   ``n(mu, beta, T) = n(E_F, 0, T)`` -- both sides at the *same* T, so the
   thermal shift cancels identically and it returns the pure drift shift.
   Measured: it gives exactly 0.50000000 eV at E_F = 0.5 eV, beta = 0 for
   every T from 50 K to 1500 K.  The closed form carries the thermal shift
   *as well*, so the gap is that shift, not an error -- it grows 5.8e-4,
   5.2e-3, 5.6e-2 at T = 100, 300, 1000 K exactly as a thermal shift must.
   Use the numeric route for the drift shift alone, the closed form for both.
   The ``xi`` constant was also corrected in the same pass: it is
   ``pi^2/3``, fixed by the exact Sommerfeld limit
   ``mu = E_F [1 - (pi^2/6)(k_B T/E_F)^2 + ...]``.  Against an independent
   carrier-conservation oracle (exact Dirac-cone Fermi integral, root-solved)
   ``pi^2/3`` is accurate to 2.4e-7 at E_F = 0.5 eV, T = 100 K, where the old
   ``2 log(4)^2`` gave 8.2e-5, and the alternative reading ``(log 16)^2``
   -- which is 2x the old one -- gave 6.5e-4.  That reading differs by an
   operator precedence; its small-T coefficient is 3.84 against the exact
   1.64.
3. RESOLVED 2026-11-03 (MATLAB parity campaign) -- hole doping.  Every
   formula in this module now takes ``|E_F|`` (see :func:`_absEF`), and
   ``thermal_drude_weight`` returns ``|mu|`` at T = 0, which is also the
   T -> 0+ limit of its own ``2 k_B T log[2 cosh(mu/2 k_B T)]``, so that
   branch is no longer discontinuous.  Before this, :func:`graphene_drude`
   and :func:`graphene_interband_T0` were *exactly odd* in E_F -- measured
   ``max|sigma(-E_F) + sigma(+E_F)| = 1.1e-16`` -- so a caller passing
   E_F < 0 got ``Re sigma_inter < 0`` at every point: negative absorption,
   i.e. optical gain from passive doped graphene, and a Drude term with both
   parts flipped, which kills the plasmon.  The Dirac cone is exactly
   electron-hole symmetric, so this was a defect, not a convention.
   The reference tree agrees and shows where the sanitisation belongs: its
   low-level entry point takes signed E_F and is simply never handed a
   negative one, because every one of its 25 call sites in the physics
   drivers passes ``abs(EF(j))``.  This library
   exposes the low-level functions *as* its public API, with no driver in
   between, so the ``abs`` has to live here.  Second order is unaffected: it
   carries an explicit ``doping_sign``, which is the reference's
   ``-sign(EF)``, correctly separated from the magnitude.
4. RESOLVED 2026-11-03 -- and it was never a code defect.  The old note claimed
   that :func:`sigma_cascade_abcd` at ``s1 = s2 = 1`` fails to reproduce the
   degenerate ``sigma_C`` of :func:`sigma_shg_abc`.  It does not fail: the two C
   coefficients multiply *different tensor structures*, so comparing them
   one-to-one asserts a false identity.  Eq. (12c)'s C multiplies
   ``grad(E.E)``, while Eq. (15c)'s C multiplies ``sum_j E_j grad(E_j)``, which
   in the degenerate limit is ``(1/2) grad(E.E)`` by the product rule -- and
   Eq. (15)'s fourth coefficient ``sigma_D``, which the old comparison ignored
   entirely, multiplies ``grad(E.E)`` directly.  The true reduction is
   ``sigma_C^(2,2) = (1/2) [sigma_C^(2,1,1) + sigma_D^(2,1,1)]``; see
   :func:`sigma_cascade_abcd`.  Both equations are transcribed faithfully and
   nothing in the library's numerics changed.
5. The 0/0 of the drift-biased conductivity at ``alpha + beta -> 0`` is
   removable in closed form, so no series fallback is used; see
   :func:`graphene_drift_sigma`.
6. SCOPE, not a defect (found 2026-11-03 by the MATLAB parity campaign):
   :func:`sigma_thg_local` and :func:`sigma_kerr` are the *intraband*
   (Drude/BTE) third order of paper 406 only.  The reference tree's own
   third-order routine computes the full Mikhailov third order,
   including the interband contribution, and the ratio
   ``|intraband / complete|`` over its own parameter range runs from 4.0e-4 to
   7.6e+01.  So wherever the interband third order dominates -- which it does
   once the photon energy approaches or exceeds ``2 E_F / 3`` -- these two
   functions can be orders of magnitude off, in either direction.  They are
   correct transcriptions of the intraband formulas (verified to 1.6e-15
   against its sigma_33 and 1.4e-15 against its sigma_31);
   the limitation is which physics they contain, not how it is coded.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq

try:  # normal case: the directory is imported as a package
    from .units import (
        E2_EV_NM,
        HBAR_EV_FS,
        KB_EV_K,
        VF_NM_FS,
        _unwrap,
        omega_of,
    )
except ImportError:  # flat layout: the package directory itself is on sys.path
    from units import (  # type: ignore[no-redef]
        E2_EV_NM,
        HBAR_EV_FS,
        KB_EV_K,
        VF_NM_FS,
        _unwrap,
        omega_of,
    )

__all__ = [
    "SIGMA0_NM_FS",
    "graphene_drude",
    "graphene_interband_T0",
    "graphene_local_rpa",
    "thermal_drude_weight",
    "drift_chemical_potential",
    "drift_chemical_potential_numeric",
    "graphene_drift_sigma",
    "metal_thin_film",
    "sigma_shg_abc",
    "sigma_shg_1d_q0",
    "sigma_thg_local",
    "sigma_kerr",
    "sigma_cascade_abcd",
]

ArrayLike = Union[float, complex, np.ndarray]
CxOut = Union[complex, np.ndarray]

#: Universal interband conductivity scale sigma0 = e^2/(4 hbar)  [nm/fs].
SIGMA0_NM_FS: float = E2_EV_NM / (4.0 * HBAR_EV_FS)

#: Second-order prefactor S2 = 1j e^3 v_F^2 / (4 pi hbar^2) of paper 406 Eq. (11).
#: The e^2**1.5 power is the one used by the existing NPLab port: absolute
#: magnitudes are model-internal, spectral shapes and ratios are what matter.
_S2 = 1j * E2_EV_NM ** 1.5 * VF_NM_FS ** 2 / (4.0 * np.pi * HBAR_EV_FS ** 2)

#: Third-order prefactor e^4 v_F^2 / (4 pi hbar^2) of paper 406 Eq. (14), still
#: missing its 1/E_F (doping enters the third-order response).
_S3_NO_EF = E2_EV_NM ** 2 * VF_NM_FS ** 2 / (4.0 * np.pi * HBAR_EV_FS ** 2)


def _c(x: ArrayLike) -> np.ndarray:
    """View an input as a complex128 array (0-d for scalars)."""
    return np.asarray(x, dtype=np.complex128)


def _absEF(x: ArrayLike) -> np.ndarray:
    """``|E_F|`` as a real array.

    Graphene's Dirac-cone response is even in the Fermi level: electron and
    hole doping are related by the exact electron-hole symmetry of the cone,
    so every formula in this module takes ``|E_F|``.  See module docstring
    item 3 for why this is applied here rather than left to the caller.
    """
    return np.abs(np.asarray(x, dtype=np.float64))


def _D(s: ArrayLike, omega: np.ndarray, gam_fs: np.ndarray) -> np.ndarray:
    """Harmonic denominator ``D_s = 1 / (s omega + 1j gamma/hbar)``  [fs].

    Building block of the nonlinear conductivities of paper 406, Eqs. (11)-(15).
    """
    return 1.0 / (s * omega + 1j * gam_fs)


# --------------------------------------------------------------------------- #
# Linear conductivities
# --------------------------------------------------------------------------- #
def graphene_drude(E_ev: ArrayLike, EF_ev: ArrayLike, gamma_ev: ArrayLike) -> CxOut:
    """Intraband (Drude) sheet conductivity of extended doped graphene  [nm/fs].

    ``sigma(E) = 1j e^2 |E_F| / [pi hbar (E + 1j gamma)]``
    -- paper 406 (Cox, Yu & Garcia de Abajo, PRB 96, 045442 (2017)), Eq. (8).

    Parameters are broadcast against each other, so any of them may be an array.
    """
    E = _c(E_ev)
    return _unwrap(1j * E2_EV_NM * _absEF(EF_ev) / (np.pi * HBAR_EV_FS * (E + 1j * _c(gamma_ev))))


def graphene_interband_T0(E_ev: ArrayLike, EF_ev: ArrayLike, gamma_ev: ArrayLike) -> CxOut:
    """Interband sheet conductivity of doped graphene at T = 0  [nm/fs].

    ``sigma_inter = 1j (sigma0/pi) log[(2 |E_F| - (E + 1j gamma))
                                       / (2 |E_F| + (E + 1j gamma))]``
    with ``sigma0 = e^2/(4 hbar)`` -- Hanson form, J. Appl. Phys. 103, 064302
    (2008), whose log prefactor is ``e^2/(4 pi hbar) = sigma0/pi``.

    The pi in the prefactor and the pi-free universal saturation value are the
    same pi, not two conventions: above the Pauli threshold E > 2 E_F the
    logarithm's branch cut contributes ``-1j pi`` to its imaginary part, so the
    absorptive part saturates at exactly ``sigma0 = e^2/(4 hbar)``, the
    universal interband conductivity.  Below threshold the term is purely
    reactive (negative imaginary part, which red-shifts the plasmons).

    This matches the finite-temperature interband branch of
    :func:`graphene_local_rpa`, whose step height is also ``sigma0``, to
    O(gamma/(E + 2 E_F)) -- the two terms that the arctan/log pair of that
    branch drops.  Module docstring item 1 records the history: before
    2026-11-03 this function carried ``1j sigma0 log(...)``, a factor pi too
    much, which made :func:`graphene_local_rpa` jump by pi across its T_K = 1
    switch.
    """
    w = _c(E_ev) + 1j * _c(gamma_ev)
    two_EF = 2.0 * _absEF(EF_ev)
    return _unwrap(1j * (SIGMA0_NM_FS / np.pi) * np.log((two_EF - w) / (two_EF + w)))


def thermal_drude_weight(mu_ev: ArrayLike, T_K: float) -> Union[float, np.ndarray]:
    """Thermal Drude weight ``mu_D = mu + 2 k_B T log(1 + exp(-mu/k_B T))``  [eV].

    Paper 456, text following Eq. (11).  Equivalently
    ``mu_D = 2 k_B T log[2 cosh(mu / 2 k_B T)]``; it is the effective energy of
    the intraband (Drude) weight at electron temperature T, *not* the chemical
    potential (which decreases with T at fixed carrier density).

    The ``log(1 + exp(-x))`` is evaluated as ``logaddexp(0, -x)``, so large
    negative ``mu/k_B T`` cannot overflow.  At T = 0 the function returns
    ``|mu|``, which is the T -> 0+ limit of the formula itself, so the branch
    is continuous (module docstring, item 3).
    """
    mu = np.asarray(mu_ev, dtype=np.float64)
    T = float(T_K)
    if T <= 0.0:
        return _unwrap(np.abs(mu))
    kT = KB_EV_K * T
    return _unwrap(mu + 2.0 * kT * np.logaddexp(0.0, -mu / kT))


def graphene_local_rpa(
    E_ev: ArrayLike, EF_ev: ArrayLike, gamma_ev: ArrayLike, T_K: float = 0.0
) -> CxOut:
    """Local-RPA linear sheet conductivity, intraband + interband  [nm/fs].

    T = 0 (``T_K < 1``): :func:`graphene_drude` + :func:`graphene_interband_T0`
    (paper 406 Eq. (8) plus the Hanson interband term).

    T >= 1 K: the Drude weight is the thermal one, ``E_F ->
    thermal_drude_weight(E_F, T)`` (paper 456 after Eq. (11)), and the interband
    step is smeared by thermal and collisional broadening in quadrature,
    ``W = hypot(2 k_B T, gamma)``::

        x     = E - 2 mu
        inter = sigma0 [ 1/2 + arctan(x/W)/pi
                         + 1j log((x^2 + W^2)/(E + 2 mu)^2) / (2 pi) ]

    ``W -> gamma`` as ``T -> 0`` keeps the smeared step continuous in gamma.

    Both branches saturate at the universal ``sigma0`` and the switch at
    T_K = 1 is continuous to O(gamma/(E + 2 mu)) -- the two terms the arctan/log
    pair drops relative to the closed-form logarithm.  Measured at E = 1.2 eV,
    E_F = 0.5 eV, gamma = 0.02 eV: warm/cold = 0.9971990 - 0.0006476j for the
    full conductivity, 1/(1.0018535 + 0.0014443j) for the interband part alone.
    (Before
    2026-11-03 the T = 0 branch carried a spurious factor pi; module docstring
    item 1.)  ``T_K`` must be a scalar; the finite-T branch assumes real photon
    energies (it takes ``arctan`` of ``x/W``).
    """
    T = float(T_K)
    if T < 1.0:
        return _unwrap(
            _c(graphene_drude(E_ev, EF_ev, gamma_ev))
            + _c(graphene_interband_T0(E_ev, EF_ev, gamma_ev))
        )
    mu = np.asarray(thermal_drude_weight(EF_ev, T), dtype=np.float64)
    E_real = np.asarray(E_ev, dtype=np.float64)
    gam = np.asarray(gamma_ev, dtype=np.float64)
    intra = _c(graphene_drude(E_ev, mu, gamma_ev))
    W = np.hypot(2.0 * KB_EV_K * T, gam)
    x = E_real - 2.0 * mu
    inter = SIGMA0_NM_FS * (
        0.5
        + np.arctan(x / W) / np.pi
        + 1j * np.log((x ** 2 + W ** 2) / (E_real + 2.0 * mu) ** 2) / (2.0 * np.pi)
    )
    return _unwrap(intra + inter)


# --------------------------------------------------------------------------- #
# Drift bias: chemical potential and anisotropic nonlocal conductivity
# --------------------------------------------------------------------------- #
def drift_chemical_potential(EF_ev: float, beta: float, T_K: float = 0.0) -> float:
    """Chemical potential of drift-biased graphene, closed form  [eV].

    Carrier-number conservation for a Fermi-Dirac distribution skewed by a drift
    velocity ``beta = v_drift / v_F`` (GRARPA notes)::

        T = 0:  mu = |E_F| (1 - beta^2)^(3/4)
        T > 0:  mu = |E_F| (1 - beta^2)^(3/4) [sqrt(1 + xi^2) - xi]^(1/2)
                xi = (pi^2/3) [k_B T / (|E_F| (1 - beta^2)^(3/4))]^2

    The ``xi`` coefficient is fixed by the exact Sommerfeld limit of
    Dirac-cone carrier conservation, ``mu = E_F [1 - (pi^2/6)(k_B T/E_F)^2]``:
    expanding the bracket above for small xi gives ``mu ~ mu0 (1 - xi/2)``, so
    the coefficient must be ``pi^2/3``.  Verified against an independent
    numerical solve of the exact Fermi integral (module docstring, item 2).

    This returns the chemical potential including *both* the drift shift and
    the thermal shift.  :func:`drift_chemical_potential_numeric` returns the
    drift shift alone -- it is not a more accurate version of this function
    (module docstring, item 2).

    Raises:
        ValueError: if ``|beta| >= 1`` (super-Fermi drift is unphysical here).
    """
    b = float(beta)
    if not abs(b) < 1.0:
        raise ValueError(f"|beta| must be < 1 (got beta={b!r})")
    mu0 = abs(float(EF_ev)) * (1.0 - b ** 2) ** 0.75
    T = float(T_K)
    if T <= 0.0 or mu0 == 0.0:
        return mu0
    xi = (np.pi ** 2 / 3.0) * (KB_EV_K * T / mu0) ** 2
    return float(mu0 * np.sqrt(np.sqrt(1.0 + xi ** 2) - xi))


def _drift_carrier_number(
    mu_ev: float,
    beta: float,
    T_K: float,
    n_E: int = 2001,
    n_phi: int = 241,
    E_lo: float = -5.0,
    E_hi: float = 2.0,
) -> float:
    """Carrier-number integral of the skewed Fermi-Dirac distribution [arb.].

    Port of the MATLAB carrier-conservation integrand behind
    :func:`drift_chemical_potential_numeric`::

        f(E, mu, beta, phi) = |E| / (exp[(E - mu - |E| beta cos phi)/(k_B T)] + 1)
        n(mu, beta)         = int_0^2pi dphi int_{-5 eV}^{2 eV} dE  f

    The ``|E|`` weight is the Dirac-cone density of states; the energy window is
    the one hard-coded in the MATLAB source, so ``|E_F|`` must sit well inside
    it.  Both integrals are Simpson quadratures on a fixed product grid, so the
    discretisation error largely cancels in the root-find (which compares two
    values of this same function).
    """
    kT = KB_EV_K * float(T_K)
    E = np.linspace(E_lo, E_hi, int(n_E))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    absE = np.abs(E)
    arg = (E[None, :] - float(mu_ev) - absE[None, :] * float(beta) * np.cos(phi)[:, None]) / kT
    # clip keeps exp() from overflowing where the occupation is 0 or 1 anyway
    occ = 1.0 / (np.exp(np.clip(arg, -700.0, 700.0)) + 1.0)
    return float(simpson(simpson(absE[None, :] * occ, x=E, axis=1), x=phi))


def drift_chemical_potential_numeric(
    EF_ev: float,
    beta: float,
    T_K: float = 300.0,
    n_E: int = 2001,
    n_phi: int = 241,
    max_expand: int = 40,
) -> float:
    """Chemical potential of drift-biased graphene by root-finding  [eV].

    Solves ``n(mu, beta) = n(E_F, 0)`` with ``n`` from
    :func:`_drift_carrier_number` (the MATLAB carrier-conservation integral),
    using :func:`scipy.optimize.brentq` on a bracket grown around the closed-form
    guess :func:`drift_chemical_potential`.

    REFERENCE: E_F = 0.5 eV, beta = 0.2, T = 300 K -> 0.484789 eV, against the
    0.4848 eV recorded in the MATLAB docstring.  Grid-converged: 1001x121 and
    8001x721 nodes agree to 8 decimals.

    Raises:
        ValueError: if ``|beta| >= 1``.
        RuntimeError: if no sign change is bracketed (``|E_F|`` outside the
            hard-coded [-5, 2] eV window, or T so small that ``n`` is flat).
    """
    b = float(beta)
    if not abs(b) < 1.0:
        raise ValueError(f"|beta| must be < 1 (got beta={b!r})")
    T = float(T_K)
    if T <= 0.0:
        raise ValueError("T_K must be > 0 for the numerical route; use the closed form at T=0")
    EF = float(EF_ev)
    target = _drift_carrier_number(EF, 0.0, T, n_E, n_phi)

    def residual(mu: float) -> float:
        return _drift_carrier_number(mu, b, T, n_E, n_phi) - target

    mu0 = drift_chemical_potential(EF, b, 0.0)
    delta = max(0.05, 0.25 * abs(EF))
    for _ in range(int(max_expand)):
        lo, hi = mu0 - delta, mu0 + delta
        if residual(lo) < 0.0 < residual(hi):
            return float(brentq(residual, lo, hi, xtol=1e-12, rtol=1e-14))
        delta *= 2.0
    raise RuntimeError(
        f"could not bracket mu for EF={EF!r}, beta={b!r}, T_K={T!r} "
        f"after {max_expand} expansions"
    )


def _drift_brackets_literal(
    alpha: np.ndarray, beta: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """The two square brackets of paper 456 Eqs. (12a)-(12b), literally.

    ``bracket_perp = (1 + alpha beta)/sqrt(1 - beta^2) - sqrt(1 - alpha^2)``
    ``bracket_par  = (1 + alpha beta)/sqrt(1 - alpha^2) - sqrt(1 - beta^2)``

    Kept for verification only: it is 0/0-prone near ``alpha + beta = 0`` (both
    brackets vanish there, and the prefactor ``2 sigma_w/(alpha+beta)^2``
    diverges).  :func:`graphene_drift_sigma` evaluates the algebraically
    identical cancellation-free form instead and the test module asserts the two
    agree where the literal one is well conditioned.
    """
    a = _c(alpha)
    b = _c(beta)
    sa = np.sqrt(1.0 - a ** 2)
    sb = np.sqrt(1.0 - b ** 2)
    num = 1.0 + a * b
    return num / sb - sa, num / sa - sb


def graphene_drift_sigma(
    E_ev: ArrayLike,
    EF_ev: float,
    gamma_ev: ArrayLike,
    Q_inv_nm: ArrayLike,
    beta: float,
    T_K: float = 0.0,
    mu_ev: Optional[float] = None,
) -> Tuple[CxOut, CxOut]:
    """Drift-biased anisotropic nonlocal graphene conductivity  [nm/fs].

    Paper 456, Eqs. (12a)-(12b) (verified against the reference's drift
    intraband conductivity), with

    * ``omega_c = E + 1j gamma``                      [eV]
    * ``alpha   = v_F Q hbar / omega_c``              (dimensionless)
    * ``beta    = v_drift / v_F``                     (given directly)
    * ``mu      = drift_chemical_potential(E_F, beta, T)``
    * ``sigma_w = 1j e^2 mu_D(mu, T) / (pi hbar omega_c)``
    * ``fct     = 2 sigma_w / (alpha + beta)^2``

      ``sigma_perp = fct [ (1 + alpha beta)/sqrt(1-beta^2) - sqrt(1-alpha^2) ]``
      ``sigma_par  = fct [ (1 + alpha beta)/sqrt(1-alpha^2) - sqrt(1-beta^2) ]``

    Returns ``(sigma_perp, sigma_par)``: across the ribbon (x) first, along the
    drift direction (y) second.

    NUMERICAL NOTE -- the ``(alpha + beta)^2`` denominator is a *removable*
    singularity, and it is removed exactly rather than by a series.  With
    ``sa = sqrt(1-alpha^2)``, ``sb = sqrt(1-beta^2)`` (principal complex
    branches),

        (1 + alpha beta)^2 - sa^2 sb^2 = (alpha + beta)^2

    identically, so each bracket equals ``(alpha+beta)^2 / {s [(1 + alpha beta)
    + sa sb]}`` with ``s = sb`` (perp) or ``s = sa`` (par), and

        sigma_perp = 2 sigma_w / {sb [(1 + alpha beta) + sa sb]}
        sigma_par  = 2 sigma_w / {sa [(1 + alpha beta) + sa sb]}

    which is finite and accurate at ``alpha + beta = 0``, needs no threshold,
    and is bit-for-bit equivalent to the literal expression away from it.  This
    replaces the fourth-order series fallback that the module brief suggested;
    note in passing that the brief's shortcut of "return sigma_w when
    |alpha+beta| < 1e-8" is only right when alpha is *also* small -- the true
    limit at ``beta = -alpha`` is ``sigma_w / (1 - alpha^2)^(3/2)``, which the
    closed form above reproduces.  Singular only at ``alpha = -beta = +-1``.

    ``mu_ev`` optionally overrides the closed-form drift chemical potential
    (e.g. with :func:`drift_chemical_potential_numeric`, see module docstring
    item 2).  ``beta``, ``T_K`` and ``EF_ev`` are scalars; ``E_ev``, ``gamma_ev``
    and ``Q_inv_nm`` broadcast.
    """
    b = float(beta)
    if not abs(b) < 1.0:
        raise ValueError(f"|beta| must be < 1 (got beta={b!r})")
    omega_c = _c(E_ev) + 1j * _c(gamma_ev)
    alpha = VF_NM_FS * _c(Q_inv_nm) * HBAR_EV_FS / omega_c
    mu = drift_chemical_potential(EF_ev, b, T_K) if mu_ev is None else float(mu_ev)
    sigma_w = 1j * E2_EV_NM * float(thermal_drude_weight(mu, T_K)) / (np.pi * HBAR_EV_FS * omega_c)

    sa = np.sqrt(1.0 - alpha ** 2)
    sb = np.sqrt(1.0 - np.complex128(b) ** 2)
    den = (1.0 + alpha * b) + sa * sb
    sigma_perp = 2.0 * sigma_w / (sb * den)
    sigma_par = 2.0 * sigma_w / (sa * den)
    return _unwrap(sigma_perp), _unwrap(sigma_par)


# --------------------------------------------------------------------------- #
# Thin metal film
# --------------------------------------------------------------------------- #
def metal_thin_film(E_ev: ArrayLike, eps: ArrayLike, t_nm: float) -> CxOut:
    """Sheet conductivity of a thin film of local dielectric function eps  [nm/fs].

    ``sigma2d = 1j omega t (1 - eps) / (4 pi)``, ``omega = E / hbar``
    -- paper 430 thin-film mapping.  A free-electron
    metal (Re eps << 0) gives Im sigma2d > 0, the same inductive sign as
    :func:`graphene_drude`.
    """
    omega = _c(omega_of(E_ev))
    return _unwrap(1j * omega * float(t_nm) * (1.0 - _c(eps)) / (4.0 * np.pi))


# --------------------------------------------------------------------------- #
# Nonlinear conductivities
# --------------------------------------------------------------------------- #
def sigma_shg_abc(
    E_ev: ArrayLike, gamma_ev: ArrayLike, doping_sign: int = +1
) -> Tuple[CxOut, CxOut, CxOut]:
    """Second-harmonic conductivity tensor structures A, B, C  [nm/fs units].

    Paper 406, Eqs. (11)-(12), with ``D_s = 1/(s omega + 1j gamma/hbar)`` and
    ``S2 = 1j e^3 v_F^2/(4 pi hbar^2)``::

        sigma_A = -doping_sign S2 D_2 D_1 ( 3 D_1 + 4 D_2 )
        sigma_B = -doping_sign S2 D_2 D_1 ( - D_1 + 4 D_2 - 4/omega )
        sigma_C = -doping_sign S2 D_2 D_1 ( - D_1/2 - 2 D_2 + 2/omega )

    ``doping_sign = +1`` for electron doping (the paper's upper sign), -1 for
    holes; the whole second-order response is odd in it.  Singular at E = 0
    (the ``1/omega`` terms).  Doping *level* E_F does not enter the second-order
    Dirac-cone response.
    """
    omega = _c(omega_of(E_ev))
    gam_fs = _c(gamma_ev) / HBAR_EV_FS
    d1 = _D(1.0, omega, gam_fs)
    d2 = _D(2.0, omega, gam_fs)
    pref = -float(doping_sign) * _S2 * d2 * d1
    sigma_a = pref * (3.0 * d1 + 4.0 * d2)
    sigma_b = pref * (-d1 + 4.0 * d2 - 4.0 / omega)
    sigma_c = pref * (-0.5 * d1 - 2.0 * d2 + 2.0 / omega)
    return _unwrap(sigma_a), _unwrap(sigma_b), _unwrap(sigma_c)


def sigma_shg_1d_q0(E_ev: ArrayLike, gamma_ev: ArrayLike, doping_sign: int = +1) -> CxOut:
    """q = 0 collapse of the 1D second-harmonic conductivity  [nm/fs units].

    ``sigma = sigma_A + sigma_B + 2 sigma_C`` of :func:`sigma_shg_abc`, which
    reduces analytically to

        ``-doping_sign S2 D_2 D_1 (D_1 + 4 D_2)``

    (paper 406, Eqs. (11)-(12): the three tensor structures collapse onto one in
    one dimension at q = 0).  Both forms are asserted equal to rtol 1e-12 in the
    test module; this function evaluates the reduced one.
    """
    omega = _c(omega_of(E_ev))
    gam_fs = _c(gamma_ev) / HBAR_EV_FS
    d1 = _D(1.0, omega, gam_fs)
    d2 = _D(2.0, omega, gam_fs)
    return _unwrap(-float(doping_sign) * _S2 * d2 * d1 * (d1 + 4.0 * d2))


def sigma_thg_local(E_ev: ArrayLike, EF_ev: ArrayLike, gamma_ev: ArrayLike) -> CxOut:
    """Local third-harmonic conductivity of extended graphene  [nm/fs units].

    ``sigma_THG = 3j e^4 v_F^2 / (4 pi hbar^2 |E_F|) D_3 D_2 D_1``
    -- paper 406, Eqs. (14)-(15) (Drude/BTE third order, ``sigma^{33}``).
    """
    omega = _c(omega_of(E_ev))
    gam_fs = _c(gamma_ev) / HBAR_EV_FS
    d1 = _D(1.0, omega, gam_fs)
    d2 = _D(2.0, omega, gam_fs)
    d3 = _D(3.0, omega, gam_fs)
    return _unwrap(3j * _S3_NO_EF / _absEF(EF_ev) * d3 * d2 * d1)


def sigma_kerr(E_ev: ArrayLike, EF_ev: ArrayLike, gamma_ev: ArrayLike) -> CxOut:
    """Local Kerr / self-phase-modulation conductivity  [nm/fs units].

    ``sigma_Kerr = -3j e^4 v_F^2 / (4 pi hbar^2 |E_F|) D_2 / (omega^2 + (gamma/hbar)^2)``
    -- paper 406, Eqs. (14)-(15) (``sigma^{31}``, the (omega, omega, -omega)
    permutation sum).
    """
    omega = _c(omega_of(E_ev))
    gam_fs = _c(gamma_ev) / HBAR_EV_FS
    d2 = _D(2.0, omega, gam_fs)
    return _unwrap(-3j * _S3_NO_EF / _absEF(EF_ev) * d2 / (omega ** 2 + gam_fs ** 2))


def sigma_cascade_abcd(
    E_ev: ArrayLike,
    gamma_ev: ArrayLike,
    s1: float,
    s2: float,
    doping_sign: int = +1,
) -> Tuple[CxOut, CxOut, CxOut, CxOut]:
    """Cascaded second-order coefficients A, B, C, D  [nm/fs units].

    Paper 406, Eqs. (15a)-(15d), for a pair of harmonic indices ``(s1, s2)``
    with ``D_sum = D_(s1+s2)``::

        A = -doping_sign S2 D_sum ( 3 D_s2^2 + 2 D_sum (D_s1 + D_s2) )
        B = -doping_sign S2 D_sum ( -4 D_s1/(s2 omega) - D_s2^2
                                    + 2 D_sum (D_s1 + D_s2) )
        C = -doping_sign S2 D_sum (  4 D_s1/(s2 omega) - D_s2^2 )
        D = +doping_sign 2 S2 D_sum^2 ( D_s1 + D_s2 )

    Note the sign of D is opposite to A, B, C, as given.  ``s2`` must be nonzero
    (the ``1/(s2 omega)`` terms); E = 0 is singular.

    CONSISTENCY AT ``s1 = s2 = 1`` (the degenerate case IS second-harmonic
    generation, so the coefficients must reduce to :func:`sigma_shg_abc`).  They
    do, but NOT coefficient-by-coefficient, because Eq. (12)'s and Eq. (15)'s
    third coefficients multiply *different* tensor structures:

    * Eq. (12c)'s ``sigma_C`` multiplies ``grad(E.E)``;
    * Eq. (15c)'s ``sigma_C`` multiplies ``sum_j E_j grad(E_j)``, which is
      ``(1/2) grad(E.E)`` in the degenerate limit by the product rule;
    * Eq. (15d)'s ``sigma_D`` multiplies ``grad(E_s1 . E_s2)`` directly.

    So the reduction is::

        sigma_A^(2,2) = sigma_A^(2,1,1)                           (exact, 1e-16)
        sigma_B^(2,2) = sigma_B^(2,1,1)                           (exact, 1e-16)
        sigma_C^(2,2) = (1/2) [ sigma_C^(2,1,1) + sigma_D^(2,1,1) ]
        sigma_A + sigma_B + sigma_C + sigma_D  at s1 = s2 = 1  ==
            :func:`sigma_shg_1d_q0`                              (exact, 1e-16)

    all four verified to ~6e-16 over a 24-point (E, gamma, doping-sign) grid.
    Comparing ``C`` against ``sigma_C`` alone, which this docstring used to call
    an unresolved transcription error in Eq. (15c), asserts a false identity:
    both published equations are correct and both are transcribed faithfully
    here.  ``sigma_cascade_abcd`` deliberately does NOT special-case s1 = s2.
    The reconstructed candidate C that the old note recorded is *wrong* off the
    diagonal (it diverges from ``(C + D)/2`` by up to 1.5e2) and has been
    removed to stop it being resurrected.

    The 1D q = 0 collapse per ordering takes the FULL ``sigma_D`` in each
    ordering (in 1D, ``grad(E_a.E_b) -> E_ax d_x E_bx + E_bx d_x E_ax`` feeds
    both), giving the paper-406 MATLAB coefficient

        ``S_s1_s2 = A + B + C + D = -ds S2 D_sum (D_s2^2 + 2 D_sum (D_s1 + D_s2))``

    which is what ``nonlinear._cascade_A`` consumes.  Note that ``A + B + 2C``
    is *not* the cascade's collapse -- that form is the degenerate Eq. (12)
    contraction, where the third structure is ``grad(E.E)`` and contributes
    twice in 1D.

    At the level of the physical current, Eq. (15) at s1 = s2 = 1 equals exactly
    ``2 x`` Eq. (12): the 2 is two-slot multiplicity (one source term for
    n = 2, s = 2 against two orderings for the cascade), and it lives in the
    term count, not in any coefficient.
    """
    omega = _c(omega_of(E_ev))
    gam_fs = _c(gamma_ev) / HBAR_EV_FS
    d_s1 = _D(float(s1), omega, gam_fs)
    d_s2 = _D(float(s2), omega, gam_fs)
    d_sum = _D(float(s1) + float(s2), omega, gam_fs)
    sign = float(doping_sign)
    pref = -sign * _S2 * d_sum
    cross = 4.0 * d_s1 / (float(s2) * omega)
    two_sum = 2.0 * d_sum * (d_s1 + d_s2)
    coeff_a = pref * (3.0 * d_s2 ** 2 + two_sum)
    coeff_b = pref * (-cross - d_s2 ** 2 + two_sum)
    coeff_c = pref * (cross - d_s2 ** 2)
    coeff_d = sign * 2.0 * _S2 * d_sum ** 2 * (d_s1 + d_s2)
    return _unwrap(coeff_a), _unwrap(coeff_b), _unwrap(coeff_c), _unwrap(coeff_d)
