"""Dispersion relations of graphene sheets and ribbons, at three levels of cost.

Notes section 11 specifies three rungs, each a validation step in its own right
and each cheaper than the next:

**Level 1 -- extended sheet.**  No ribbon: the quasistatic reflection
coefficient of a bare conducting sheet between two half spaces.  ``Im r`` over
the ``(Q, E)`` plane traces the sheet plasmon, and with drift (``beta != 0``) the
``+Q`` and ``-Q`` branches split.  Closed form, fully vectorised over the whole
grid in one call, and it isolates the drift conductivity from every other moving
part.

**Level 2 -- ribbon eigenvalue bands.**  One eigendecomposition of the
material-free operator ``V @ D_geom`` per ``q`` (``solver.RibbonSystem.eigenmodes``),
reused across every frequency; bands come from solving ``Re eta(E) = eta_jq``,
linewidths from ``Im eta``.  A purely dielectric substrate is handled by
``screened=True``, which swaps in ``RibbonSystem.eigenmodes_screened``.  This is
a ``beta = 0`` method -- see the warning in :func:`ribbon_bands`.

**Level 3 -- LDOS map.**  The induced field / loss probability over the
``(q, E)`` plane from the full self-consistent solve, peak-tracked to extract the
bands.  The only level that carries mode WEIGHT as well as position, which is
what shows that drift both shifts and quenches bands; the ``+-q`` asymmetry map
is the headline figure.

Source equations
----------------
* Level 1 sheet reflection, Gaussian eV-nm-fs units::

      chi(Q, omega) = 4 pi i sigma_par(Q, omega) |Q| / omega
      r(Q, omega)   = (eps_a - eps_b + chi) / (eps_a + eps_b + chi)

  -- quasistatic limit of the Fresnel coefficient of a conducting sheet;
  reduces to ``1/(1 - i omega / (2 pi sigma |Q|))`` at ``eps_a = eps_b = 1``,
  which is the kernel used by exploratory sheet-dispersion scripts, and to the
  quasistatic Fresnel
  ``(eps_a - eps_b)/(eps_a + eps_b)`` of :func:`substrate.reflection_coefficient`
  at ``sigma = 0``.  Its pole is ``eps_a + eps_b + 4 pi i sigma Q / omega = 0``,
  i.e. ``sigma = i eps_bar omega / (2 pi |Q|)`` with
  ``eps_bar = (eps_a + eps_b)/2`` -- the Gaussian 2D plasmon condition.
* Level 1 long-wavelength (local Drude) limit, derived in this module's units
  and verified to machine precision by ``tests/test_dispersion.py``::

      hbar omega = sqrt( 4 e^2 E_F |Q| / (eps_a + eps_b) )
                 = sqrt( 2 e^2 E_F |Q| / eps_bar )

  (:func:`sheet_plasmon_energy`).  Note the 4, not 2, in the numerator.
* Longitudinal conductivity: paper 456, Eqs. (12a)-(12b) via
  ``conductivity.graphene_drift_sigma``, SECOND returned component
  (``sigma_par``).  See the component note below.
* Level 2 resonance condition: ``eta(E) = eta_jq`` with
  ``eta(E) = i sigma(E) / (omega W eps_bar)`` (``solver.RibbonSystem.eta_pair``)
  and ``eta_jq`` the geometric eigenvalues of ``V @ D_geom``
  (``solver.RibbonSystem.eigenmodes``).  For local Drude graphene this inverts
  in closed form::

      hbar omega_jq = sqrt( -e^2 E_F / (pi eps_bar eta_jq W) - gamma^2 )

  (:func:`drude_band_energy`; the ``gamma^2`` term is the shift of the
  ``Re eta`` root and vanishes in the quoted lossless closed form).
* Level 2 linewidth: ``FWHM = 2 |Im eta(E*)| / |d Re eta / dE|_{E*}``, from
  expanding the resonant denominator ``1 - eta(E) lambda_j = -lambda_j
  (eta(E) - eta_j)`` about the root.
* Level 3 LDOS: ``observables.ldos_q``, i.e. paper 456's
  ``D(q) = Im(conj(p) . E_q) / (2 pi^2 |p|^2)`` with ``E_q`` from
  ``observables.induced_field_q``.

WHICH CONDUCTIVITY COMPONENT (settled)
--------------------------------------
Level 1 needs the LONGITUDINAL conductivity, i.e. the SECOND component returned
by ``conductivity.graphene_drift_sigma`` (``sigma_par``).
Expanding paper 456's conductivity at ``beta = 0`` and small ``alpha`` gives
``sigma/sigma_0 - 1 -> (1/4) alpha^2`` for the first component and
``(3/4) alpha^2`` for the second, which is the textbook semiclassical 2D Dirac
transverse / longitudinal split.  Taking the FIRST component instead --
transverse -- moves the ``Im r`` peak by 0.36 % at ``Q = 0.1``
(0.535300 -> 0.537248 eV) and by 1.10 % at ``Q = 0.3``
(0.933386 -> 0.943643 eV), at ``E_F = 1``, ``gamma = 0.1``: small enough to go
unnoticed in an exploratory script, which is why it is pinned.  Both numbers
are remeasured by ``tests/test_dispersion.py``.

WHY THE ABSOLUTE POLE ENERGIES ARE NOT UNIVERSAL (measured here)
----------------------------------------------------------------
A kernel written as ``1 - 1i*W/(2*pi*sigma*|Q|)`` with ``W`` in eV, ``Q`` in
1/nm and ``sigma`` in ATOMIC units mixes three unit systems: the dimensionally
correct combination is
``omega_au/(sigma_au Q_au)``, and ``W/(sigma_au Q_nm)`` is larger than it by
exactly ``au_eV * au_nm = e^2 = 1.439964`` eV nm, so such a kernel's plasmon
sits a factor ``sqrt(e^2) = 1.19998`` too low in energy.  In the LOCAL-DRUDE
limit that factor is exact: that kernel resolves to
``omega^2 = 2 E_F |Q|`` against the correct ``(hbar omega)^2 = 2 e^2 E_F |Q|``
of :func:`sheet_plasmon_energy`, so the ratio is ``sqrt(e^2)`` identically
(pinned to 1e-14 in the test file).  With the nonlocal conductivity the two
poles sit at different energies and therefore at different ``alpha``, so the
ratio is only approximately ``sqrt(e^2)``: that kernel's ``Im r`` peak is at
0.447882 eV at ``Q = 0.1`` where this module gives 0.537248 (ratio 1.1995), and
at 0.791449 eV at ``Q = 0.3`` where this module gives 0.943643 (ratio 1.1923).
This module works throughout in the library's eV-nm-fs units, in which the
kernel is dimensionally consistent as written; it does NOT reproduce those
absolute pole energies, by design.

RANGE LIMIT ON THE LEVEL 1 <-> LEVEL 2 CROSS-CHECK
--------------------------------------------------
:func:`extended_sheet_eta` gives the eta at which the Level 2 resonance
condition is algebraically identical to the Level 1 pole -- that identity holds
to machine precision and is what pins the two levels to one convention.  The
finite-ribbon approach to it is slow: the uniform-potential eigenmode of a
ribbon of width ``W`` converges to ``-1/(2 pi |q| W)`` from above, measured at
``L = 201`` as ratio 1.968 at ``qW = 1``, 1.515 at 2, 1.298 at 4 and 1.221 at 8
(about ``1 + 0.6/sqrt(qW)`` once ``qW >= 4``), because the inverse-square-root
charge pile-up at the ribbon edges is a genuine finite-width effect.  Reaching
1 % would need ``qW ~ 3600``, far past the
``operators.STRUVE_TRUSTED_MAX_ARG = 8`` ceiling of the coplanar Coulomb block.
So the numerical wide-ribbon limit is a trend test, not a tolerance test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.signal import find_peaks

try:  # normal case: the directory is imported as a package
    from . import conductivity, observables, operators, sources
    from .solver import RibbonSystem, bright_mode_mask
    from .units import E2_EV_NM, _unwrap, omega_of
except ImportError:  # flat layout: the package directory itself is on sys.path
    import conductivity  # type: ignore[no-redef]
    import observables  # type: ignore[no-redef]
    import operators  # type: ignore[no-redef]
    import sources  # type: ignore[no-redef]
    from solver import RibbonSystem, bright_mode_mask  # type: ignore[no-redef]
    from units import E2_EV_NM, _unwrap, omega_of  # type: ignore[no-redef]

__all__ = [
    "sheet_sigma_longitudinal",
    "sheet_response",
    "sheet_reflection_sigma",
    "sheet_reflection",
    "sheet_plasmon_energy",
    "extended_sheet_eta",
    "drude_band_energy",
    "BandStructure",
    "ribbon_bands",
    "SCREEN_REFERENCE_EV",
    "ldos_map",
    "track_peaks",
    "follow_peaks",
    "symmetric_q_grid",
    "pm_asymmetry",
]

ArrayLike = Union[float, complex, np.ndarray]
CxOut = Union[complex, np.ndarray]
FOut = Union[float, np.ndarray]


def _c(x: ArrayLike) -> NDArray[np.complex128]:
    """View an input as a complex128 array (0-d for scalars)."""
    return np.asarray(x, dtype=np.complex128)


def _f(x: ArrayLike) -> NDArray[np.float64]:
    """View an input as a float64 array (0-d for scalars)."""
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Level 1 -- extended sheet
# --------------------------------------------------------------------------- #
def sheet_sigma_longitudinal(
    Q_inv_nm: ArrayLike,
    E_ev: ArrayLike,
    EF_ev: float,
    gamma_ev: ArrayLike,
    beta: float = 0.0,
    *,
    T_K: float = 0.0,
    mu_ev: Optional[float] = None,
    local: bool = False,
) -> CxOut:
    """Longitudinal (``Q``-parallel) sheet conductivity of graphene  [nm/fs].

    The second component of ``conductivity.graphene_drift_sigma`` (paper 456
    Eqs. 12a-12b), which is the longitudinal one -- see the module docstring.
    ``local=True`` substitutes the local Drude conductivity
    ``conductivity.graphene_drude``, which is independent of ``Q``; it is what
    the long-wavelength closed form :func:`sheet_plasmon_energy` inverts, and it
    is also the conductivity that ``solver.RibbonSystem.eta_pair`` uses at
    ``beta = 0``, so Level 1 / Level 2 comparisons want ``local=True``.

    Parameters
    ----------
    Q_inv_nm
        Guided wave vector [1/nm], SIGNED.  The drift response is not even in
        ``Q``; that is the non-reciprocity.  Broadcast against ``E_ev``.
    E_ev
        Photon energy [eV], strictly positive.  Broadcast against ``Q_inv_nm``.
    EF_ev
        Fermi energy [eV].  Scalar (the drift chemical potential is a scalar
        root find).
    gamma_ev
        Inelastic broadening [eV]; broadcast.
    beta
        Drift velocity in units of ``v_F``; scalar, ``|beta| < 1``.
    T_K
        Temperature [K], for the drift chemical potential / Drude weight.
    mu_ev
        Optional override of the closed-form drift chemical potential, e.g. from
        ``conductivity.drift_chemical_potential_numeric`` (the closed form is
        ~0.5 % off at 300 K).  Ignored when ``local=True``.
    local
        Use the local Drude conductivity instead of the nonlocal drift tensor.
        ``beta``, ``T_K`` and ``mu_ev`` are then unused.

    Returns
    -------
    complex or ndarray
        Broadcast shape of ``Q_inv_nm`` and ``E_ev``.
    """
    if local:
        sig = _c(conductivity.graphene_drude(E_ev, EF_ev, gamma_ev))
        # broadcast against Q even though the local conductivity does not use it
        return _unwrap(sig + np.zeros(np.shape(_f(Q_inv_nm)), dtype=np.complex128))
    _, sigma_par = conductivity.graphene_drift_sigma(
        E_ev, EF_ev, gamma_ev, Q_inv_nm, beta, T_K=T_K, mu_ev=mu_ev
    )
    return sigma_par


def sheet_response(
    sigma_nm_fs: ArrayLike,
    Q_inv_nm: ArrayLike,
    E_ev: ArrayLike,
) -> CxOut:
    """Dimensionless sheet response ``chi = 4 pi i sigma |Q| / omega``.

    The single combination in which a zero-thickness conducting sheet enters the
    quasistatic boundary condition: the sheet behaves as an extra permittivity
    ``chi`` added to ``eps_a + eps_b``.  ``omega = E / hbar``, so with ``sigma``
    in nm/fs and ``Q`` in 1/nm this is dimensionless.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _unwrap(
            4.0 * np.pi * 1j * _c(sigma_nm_fs) * np.abs(_f(Q_inv_nm)) / _c(omega_of(E_ev))
        )


def sheet_reflection_sigma(
    sigma_nm_fs: ArrayLike,
    Q_inv_nm: ArrayLike,
    E_ev: ArrayLike,
    eps_a: complex = 1.0,
    eps_b: complex = 1.0,
) -> CxOut:
    """Quasistatic reflection coefficient of a sheet of given conductivity.

        ``r = (eps_a - eps_b + chi) / (eps_a + eps_b + chi)``,
        ``chi = 4 pi i sigma |Q| / omega``

    At ``sigma = 0`` this is the quasistatic Fresnel coefficient
    ``(eps_a - eps_b)/(eps_a + eps_b)`` of ``substrate.reflection_coefficient``;
    at ``eps_a = eps_b = 1`` it is ``1/(1 - i omega/(2 pi sigma |Q|))``, the
    kernel of the reference MATLAB.  The pole is at
    ``sigma = i eps_bar omega/(2 pi |Q|)``.

    Separated from :func:`sheet_reflection` so that the pole condition can be
    probed with an arbitrary (e.g. analytically constructed) conductivity.

    Parameters
    ----------
    sigma_nm_fs
        Longitudinal sheet conductivity [nm/fs]; broadcast.
    Q_inv_nm, E_ev
        Wave vector [1/nm] and photon energy [eV]; broadcast.
    eps_a, eps_b
        Permittivities of the half spaces above and below the sheet.

    Returns
    -------
    complex or ndarray
        A complex infinity at an exact pole (the lossless case), not an
        exception.  numpy's complex division gives ``-inf + nan j`` there,
        because the numerator's imaginary part is also zero; ``abs(r)`` is
        ``inf`` either way, so test the magnitude and not the parts.
    """
    chi = _c(sheet_response(sigma_nm_fs, Q_inv_nm, E_ev))
    num = _c(eps_a) - _c(eps_b) + chi
    den = _c(eps_a) + _c(eps_b) + chi
    with np.errstate(divide="ignore", invalid="ignore"):
        return _unwrap(num / den)


def sheet_reflection(
    Q_inv_nm: ArrayLike,
    E_ev: ArrayLike,
    EF_ev: float,
    gamma_ev: ArrayLike,
    beta: float = 0.0,
    eps_a: complex = 1.0,
    eps_b: complex = 1.0,
    *,
    T_K: float = 0.0,
    mu_ev: Optional[float] = None,
    local: bool = False,
) -> CxOut:
    """Level 1: quasistatic reflection coefficient of a drift-biased graphene sheet.

    ``Im r`` over the ``(Q, E)`` plane traces the sheet plasmon.  Closed form and
    fully vectorised: pass ``Q_inv_nm`` and ``E_ev`` already broadcast against
    each other (e.g. ``Q[:, None]`` and ``E[None, :]``) and the whole map is
    computed in one call, with no Python loop.

    With ``beta != 0`` the ``+Q`` and ``-Q`` branches split; feed the result to
    :func:`pm_asymmetry` for the asymmetry map.  At ``beta = 0`` the result is
    even in ``Q`` bit for bit (the conductivity depends on ``Q`` only through
    ``alpha^2`` and the kernel only through ``|Q|``), so the asymmetry is exactly
    zero rather than merely small.

    Parameters
    ----------
    Q_inv_nm, E_ev, EF_ev, gamma_ev, beta, T_K, mu_ev, local
        As in :func:`sheet_sigma_longitudinal`.
    eps_a, eps_b
        Permittivities of the half spaces above and below the sheet.

    Returns
    -------
    complex or ndarray
        Broadcast shape of ``Q_inv_nm`` and ``E_ev``.
    """
    sigma = sheet_sigma_longitudinal(
        Q_inv_nm, E_ev, EF_ev, gamma_ev, beta, T_K=T_K, mu_ev=mu_ev, local=local
    )
    return sheet_reflection_sigma(sigma, Q_inv_nm, E_ev, eps_a, eps_b)


def sheet_plasmon_energy(
    Q_inv_nm: ArrayLike,
    EF_ev: ArrayLike,
    *,
    eps_a: complex = 1.0,
    eps_b: complex = 1.0,
) -> FOut:
    """Textbook long-wavelength 2D plasmon energy of a local-Drude sheet  [eV].

        ``hbar omega = sqrt( 4 e^2 E_F |Q| / (eps_a + eps_b) )``

    Derivation in this module's units: the pole of
    :func:`sheet_reflection_sigma` is ``eps_a + eps_b + 4 pi i sigma Q/omega = 0``;
    substituting the lossless Drude ``sigma = i e^2 E_F/(pi hbar^2 omega)`` gives
    ``omega^2 = 4 e^2 E_F |Q| / (hbar^2 (eps_a + eps_b))``.  The prefactor is 4,
    not 2 -- with ``eps_bar = (eps_a + eps_b)/2`` it reads
    ``(hbar omega)^2 = 2 e^2 E_F |Q| / eps_bar``.

    Verified against :func:`sheet_reflection_sigma` with the lossless
    ``conductivity.graphene_drude``: the denominator
    ``eps_a + eps_b + chi`` vanishes to at most 3e-15 in absolute value at this
    energy over ``Q`` in [0.01, 0.5], ``E_F`` in [0.2, 1] and
    ``(eps_a, eps_b)`` in {(1,1), (1,4), (2,2)} (``tests/test_dispersion.py``).

    Parameters
    ----------
    Q_inv_nm
        Wave vector [1/nm]; only ``|Q|`` enters.  Broadcast against ``EF_ev``.
    EF_ev
        Fermi energy [eV]; broadcast.
    eps_a, eps_b
        Permittivities of the half spaces.  Real parts only are meaningful here
        (the closed form is the lossless limit).

    Returns
    -------
    float or ndarray
    """
    eps_sum = float(np.real(_c(eps_a) + _c(eps_b)))
    return _unwrap(
        np.sqrt(4.0 * E2_EV_NM * _f(EF_ev) * np.abs(_f(Q_inv_nm)) / eps_sum)
    )


def extended_sheet_eta(Q_inv_nm: ArrayLike, W_nm: ArrayLike) -> FOut:
    """The Level 2 ``eta`` whose resonance coincides with the Level 1 sheet pole.

        ``eta_sheet = -1 / (2 pi |Q| W)``

    An extended sheet is the ``q``-uniform limit of a ribbon: the mode with
    constant potential across the width has ``D_geom phi = -(qW)^2 phi``, and the
    ``y``-Fourier Coulomb operator applied to a uniform charge on an infinite
    sheet is ``int 2 K_0(|q| W |u|) du = 2 pi / (|q| W)``, so
    ``lambda = -2 pi |q| W`` and ``eta = 1/lambda``.  Substituting into the
    Level 2 condition ``i sigma/(omega W eps_bar) = eta`` gives
    ``sigma = i eps_bar omega/(2 pi |Q|)``, which is EXACTLY the Level 1 pole
    condition -- the two levels share one convention, and this function is the
    bridge used to test it.

    ``W_nm`` cancels out of the resulting resonance frequency; it appears here
    only because Level 2's ``eta`` is defined with a ``1/W``.

    The finite-ribbon uniform mode approaches this value only as
    ``1 + ~0.6/sqrt(qW)`` for ``qW >= 4`` -- see the module docstring for the
    measured ratios and for why this is a trend test, not a tolerance test.

    Parameters
    ----------
    Q_inv_nm
        Wave vector [1/nm]; only ``|Q|`` enters.  Broadcast against ``W_nm``.
    W_nm
        Ribbon width used in the Level 2 ``eta`` normalisation [nm].

    Returns
    -------
    float or ndarray
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _unwrap(-1.0 / (2.0 * np.pi * np.abs(_f(Q_inv_nm)) * _f(W_nm)))


# --------------------------------------------------------------------------- #
# Level 2 -- ribbon eigenvalue bands
# --------------------------------------------------------------------------- #
def drude_band_energy(
    eta_j: ArrayLike,
    W_nm: ArrayLike,
    EF_ev: ArrayLike,
    *,
    eps_bar: complex = 1.0,
    gamma_ev: ArrayLike = 0.0,
) -> FOut:
    """Closed-form Level 2 band energy for local Drude graphene  [eV].

        ``hbar omega_jq = sqrt( -e^2 E_F / (pi eps_bar eta_jq W) - gamma^2 )``

    Obtained by inverting ``eta(E) = i sigma/(omega W eps_bar)`` with
    ``sigma = i e^2 E_F/(pi hbar (E + i gamma))`` and ``omega = E/hbar``, which
    gives ``Re eta(E) = -e^2 E_F / (pi W eps_bar (E^2 + gamma^2))``.  With
    ``gamma = 0`` (the default) this is the lossless closed form quoted in the
    notes; it returns ``hbar omega_1 = 0.5775 eV`` for ``E_F = 0.5 eV``,
    ``W = 10 nm``, ``eta_1 = -0.0687``.

    ``NaN`` is returned where the argument of the square root is negative
    (``|eta_j|`` too large for any real root, i.e. ``gamma`` exceeds the lossless
    resonance energy) or where ``eta_j >= 0``.

    Parameters
    ----------
    eta_j
        Geometric eigenvalue(s) from ``solver.RibbonSystem.eigenmodes``
        (negative).  Broadcast against the other arguments.
    W_nm
        Ribbon width [nm].
    EF_ev
        Fermi energy [eV].
    eps_bar
        ``(eps_a + eps_b)/2`` of the ribbon system; real part is used.
    gamma_ev
        Inelastic broadening [eV].  ``0`` gives the lossless closed form; a
        nonzero value gives the root of ``Re eta(E) = eta_j``.

    Returns
    -------
    float or ndarray
    """
    eta = _f(np.real(eta_j))
    eb = float(np.real(_c(eps_bar)))
    with np.errstate(divide="ignore", invalid="ignore"):
        e0sq = -E2_EV_NM * _f(EF_ev) / (np.pi * eb * eta * _f(W_nm))
        arg = e0sq - _f(gamma_ev) ** 2
        # eta_j >= 0 has no resonance at all; the explicit mask also catches
        # eta_j == -0.0, which would otherwise divide to +inf and survive the
        # arg > 0 test.
        ok = (arg > 0.0) & (eta < 0.0)
        out = np.where(ok, np.sqrt(np.where(ok, arg, 1.0)), np.nan)
    return _unwrap(np.asarray(out, dtype=np.float64))


@dataclass
class BandStructure:
    """Result of :func:`ribbon_bands`: Level 2 bands over a ``q`` grid.

    All arrays are ``(nq, n_bands)`` except ``qs`` which is ``(nq,)``.  Bands are
    ordered per ``q`` by DESCENDING ``|eta|``, i.e. ASCENDING energy.

    A MODE WITH NO ROOT OF ``Re eta(E) = eta_j`` INSIDE ``[E_min, E_max]`` IS
    DROPPED, not reported: the retained bands are contiguous from index 0 and
    the unused tail is ``NaN`` (never the nearest grid point, and never a root
    clamped to a window edge).  The band index is therefore NOT the eigenmode
    index -- ``mode_index`` records which eigenmode each band came from, and the
    two differ whenever a mode was skipped or ``bright_only`` filtered one out.
    Band 0 is the fundamental only when nothing lower-energy was retained; see
    the ``bright_only`` note in :func:`ribbon_bands` for the one mode that
    routinely gets in front of it.

    Attributes
    ----------
    qs
        Wave-vector grid [1/nm].
    eta
        Geometric eigenvalue of each band (complex128; real in practice).
    zeta
        Dipole overlap ``zeta_jx = int theta rho_j`` of each band.
    weight
        ``|zeta|^2`` divided by the largest ``|zeta|^2`` over the RETAINED bands
        at that ``q``.  Dark modes land near zero.
    bright
        ``solver.bright_mode_mask`` applied to the full spectrum, restricted to
        the retained bands.
    E_ev
        Band energy [eV], the root of ``Re eta(E) = eta_j``.
    fwhm_ev
        Full width at half maximum [eV], ``2 |Im eta| / |d Re eta / dE|``.
    E_drude_ev
        The closed-form :func:`drude_band_energy` for the same ``eta``, for
        reference; equals ``E_ev`` to the root-finder tolerance for local Drude
        graphene and diverges from it for any other conductivity model.
    mode_index
        Index of the eigenmode (in ``eigenmodes``' descending-``|eta|`` order)
        that each band came from, or ``-1`` in a ``NaN``-padded slot.  Makes the
        skipping above visible instead of implicit.
    """

    qs: NDArray[np.float64]
    eta: NDArray[np.complex128]
    zeta: NDArray[np.complex128]
    weight: NDArray[np.float64]
    bright: NDArray[np.bool_]
    E_ev: NDArray[np.float64]
    fwhm_ev: NDArray[np.float64]
    E_drude_ev: NDArray[np.float64]
    mode_index: NDArray[np.int64]


#: Reference energy at which :func:`ribbon_bands` evaluates
#: ``RibbonSystem.eigenmodes_screened``.  Its value is irrelevant: that method
#: is rejected here unless the stack is non-dispersive, in which case the
#: returned spectrum is bit-for-bit independent of the energy (verified in
#: ``tests/test_dispersion.py``).
SCREEN_REFERENCE_EV: float = 1.0

#: An eigenvalue below this fraction of the largest is treated as unresolved,
#: and its eigenvector is never used as a tracking reference (see the ordering
#: note in :func:`ribbon_bands`).  Machine epsilon is 2.2e-16 and the smallest
#: legitimately-resolved eigenvalue seen on a real grid is 6.3e-4 of the
#: largest, so anything in between separates the two cases comfortably.
_NULL_EIGENVALUE: float = 1e-12

#: How well an eigenvector must match the one a band held at the previous |q|
#: before the band will accept it.  The two populations are cleanly separated,
#: measured rather than guessed:
#:
#: * the SAME branch one step later scores 0.9996 to 1.0000, across a whole
#:   sweep and every band;
#: * two DIFFERENT eigenvectors of the same operator score at most 0.4032
#:   (``A = V @ D_geom`` is not symmetric, so they are not orthogonal and this
#:   is not zero);
#: * the wrong match that caused the spike -- band 0 reaching for a resolved
#:   mode at ``q = 0``, where its own eigenvector is undefined -- scored
#:   0.2157, while the other bands still scored 0.99.
#:
#: so anything between 0.41 and 0.99 separates them and 0.5 is comfortable at
#: both ends.  A band that finds nothing this good is left EMPTY for that
#: ``|q|`` rather than handed whatever was nearest: a one-point gap in a curve
#: is honest, a silently wrong branch is not.  It keeps its reference, so it
#: picks its own branch up again at the next ``|q|``.
_MIN_OVERLAP: float = 0.5

#: Two eigenvalues this close, relative to the largest, are treated as ONE
#: degenerate block.  Identical ribbons produce exactly degenerate spectra --
#: measured at 1e-12 relative and tighter for the high-order modes, which are
#: localised and barely feel their neighbours -- while distinct branches of a
#: single ribbon are separated by 1e-1.  Nothing lives in between.
_DEGENERATE_ETA: float = 1e-8

#: How much of a band's previous eigenvector may fail to lie in a degenerate
#: block's SPAN before the band is judged not to belong to it.  Measured on
#: five identical ribbons at the step where the plain overlap test broke down:
#: the three bands whose branch really is in the block gave 1.167e-02 each,
#: and the one that is elsewhere gave 9.280e-01 -- two orders of magnitude
#: apart, against 0.37 vs 0.98 for the overlap, which is far less separated
#: and, worse, not a projection at all (see :func:`_follow_modes`).
_SPAN_RESIDUAL: float = 0.1


def _band_obstruction(system: RibbonSystem, screened: bool) -> Optional[str]:
    """Why :func:`ribbon_bands` cannot run on ``system``, or ``None``.

    Level 2 needs a single scalar ``eta(E)`` multiplying one frequency-
    independent geometric operator.  ``RibbonSystem.fast_path_obstruction``
    tests exactly that, but it also rejects every ``layers`` stack, whereas
    :meth:`RibbonSystem.eigenmodes_screened` handles a NON-DISPERSIVE one
    exactly (``r(Q)`` is then frequency independent, so ``(V + Vbar) @ D_geom``
    is as geometric as the free-standing operator).  So the substrate clause is
    re-derived here instead of being inherited.
    """
    if any(rib.beta != 0.0 for rib in system.ribbons):
        return (
            f"beta != 0 for {[r.beta for r in system.ribbons]}: with drift eta "
            "becomes a tensor (eta_perp != eta_par), so M is not eta(E) * A and a "
            "single geometric spectrum has no resonance condition to invert"
        )
    ref = system.ribbons[0]
    for i, rib in enumerate(system.ribbons[1:], start=1):
        if (rib.W_nm, rib.EF_ev, rib.gamma_ev) != (ref.W_nm, ref.EF_ev, ref.gamma_ev):
            return (
                "every ribbon must share one (W_nm, EF_ev, gamma_ev) so that a "
                f"single scalar eta(E) multiplies A; ribbon {i} has "
                f"{(rib.W_nm, rib.EF_ev, rib.gamma_ev)} vs "
                f"{(ref.W_nm, ref.EF_ev, ref.gamma_ev)}"
            )
    if system.layers and not screened:
        return (
            "the system carries a substrate stack but screened=False, so the "
            "purely geometric eigenmodes() would ignore it; pass screened=True "
            "to diagonalise (V + Vbar) @ D_geom instead"
        )
    if system.layers and system._stack_is_dispersive():
        return (
            "the substrate stack is dispersive (a layer carries sigma_sheet), so "
            "r(Q, omega) and hence the screened operator depend on the frequency "
            "and there is no single geometric spectrum to invert"
        )
    return None


def _follow_modes(
    prev: Optional[NDArray[np.complex128]],
    has_prev: NDArray[np.bool_],
    modes: NDArray[np.complex128],
    cand: NDArray[np.int64],
    nb: int,
    eta: Optional[NDArray[np.complex128]] = None,
) -> NDArray[np.int64]:
    """Match each band slot to the eigenvector it held at the previous ``q``.

    Returns an array of length ``nb`` holding the eigenmode index for each band,
    or ``-1`` where there was no candidate left to give it.

    ``A = V @ D_geom`` is not symmetric, so its eigenvectors are not orthogonal
    and ``|<a|b>|`` is a similarity rather than a projection.  That is why the
    assignment is greedy over the WHOLE overlap matrix -- best pair first, each
    band and each eigenvector used once -- instead of an independent argmax per
    band, which can hand the same eigenvector to two bands where two branches
    nearly touch.  Slots with no history are filled afterwards, in the
    descending-``|eta|`` order ``eigenmodes`` returns.
    """
    sel = np.full(nb, -1, dtype=np.int64)
    if cand.size == 0:
        return sel
    rows = np.flatnonzero(has_prev[:nb]) if prev is not None else np.empty(0, dtype=np.int64)
    taken: set = set()
    if rows.size:
        P = prev[:, rows]
        P = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-300)
        C = modes[:, cand]
        C = C / np.maximum(np.linalg.norm(C, axis=0, keepdims=True), 1e-300)
        ov = np.abs(P.conj().T @ C)
        ov = np.where(np.isfinite(ov), ov, -1.0)
        ncol = ov.shape[1]
        for flat in np.argsort(ov, axis=None)[::-1]:
            r, c = divmod(int(flat), ncol)
            # Sorted descending, so once the best remaining pair is below the
            # floor nothing further can qualify.
            if ov[r, c] < _MIN_OVERLAP:
                break
            b = int(rows[r])
            if sel[b] >= 0 or c in taken:
                continue
            sel[b] = int(cand[c])
            taken.add(c)
            if len(taken) == min(rows.size, ncol):
                break
    # DEGENERATE BLOCKS, where eigenvector continuity is not merely hard but
    # ILL-POSED. LAPACK returns an arbitrary basis for a repeated eigenvalue,
    # so a band's overlap with any single member can be as low as 1/sqrt(m) --
    # 0.707, 0.577, 0.500, 0.447 for m = 2 to 5 -- and the floor then rejects a
    # branch that did continue, leaving holes scattered through the curve at
    # whichever q the arithmetic chose. `_MIN_OVERLAP` is 0.5, which is exactly
    # 1/sqrt(4); a four-fold block lands on the threshold and is decided by the
    # last bit.
    #
    # This is the ordinary case, not a corner: `_band_obstruction` requires
    # every ribbon to share one (W, EF, gamma), so a multi-ribbon system here
    # is always an array of IDENTICAL ribbons, whose spectrum is degenerate by
    # construction. Measured on five of them: band 1's best overlap was 0.5000,
    # the floor itself, and 5 of 52 points were dropped.
    #
    # The well-defined object is the BLOCK, not its vectors. The right question
    # is whether the band's previous eigenvector lies in the block's SPAN, which
    # no basis choice inside the block can change -- answered by least squares,
    # not by overlap.
    #
    # Summing |<p|c>|^2 over the block does NOT answer it. These eigenvectors
    # are not orthonormal (A = V @ D_geom is not symmetric), so that sum is not
    # a projection and is not bounded by 1: measured on the step where this
    # mattered it came out 1.65, 1.08 and 1.12 for three bands that do belong
    # to the block and 0.37 for one that does not. The separation is poor and
    # the quantity means nothing. The least-squares residual on the same step
    # gives 1.167e-02 for all three against 9.280e-01 for the fourth.
    #
    # Members of a block share one eta and therefore one energy, so WHICH member
    # a band is handed cannot change the picture.
    if rows.size and eta is not None and cand.size:
        ec = np.real(np.asarray(eta)[cand])
        scale = float(np.max(np.abs(ec))) or 1.0
        ncol = ec.size
        for r, b in enumerate(rows):
            if sel[int(b)] >= 0:
                continue
            free = np.asarray([c for c in range(ncol) if c not in taken], dtype=int)
            if not free.size:
                continue
            c_best = int(free[int(np.argmax(ov[r, free]))])
            block = free[np.abs(ec[free] - ec[c_best]) <= _DEGENERATE_ETA * scale]
            if block.size < 2:
                continue
            p = P[:, r]
            B = C[:, block]
            try:
                x, *_ = np.linalg.lstsq(B, p, rcond=None)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                continue
            denom = float(np.linalg.norm(p)) or 1.0
            if float(np.linalg.norm(p - B @ x)) / denom <= _SPAN_RESIDUAL:
                sel[int(b)] = int(cand[c_best])
                taken.add(c_best)

    # Slots that have NEVER held a mode -- the first |q|, or a band that
    # outran the supply of candidates -- take what is left, highest |eta|
    # first. A band WITH a history is deliberately not filled this way: having
    # failed the overlap floor above, it has no branch here, and giving it the
    # nearest spare is exactly how band 0 came to report a high-order mode's
    # energy at q = 0 and draw a spike out of the top of the picture.
    spare = [int(m) for c, m in enumerate(cand) if c not in taken]
    for b in range(nb):
        if sel[b] < 0 and not bool(has_prev[b]) and spare:
            sel[b] = spare.pop(0)
    return sel


def ribbon_bands(
    system: RibbonSystem,
    qs: ArrayLike,
    *,
    n_bands: int = 4,
    bright_only: bool = False,
    screened: bool = False,
    track: bool = False,
    E_min: float = 1e-3,
    E_max: float = 10.0,
    rel_step: float = 1e-4,
) -> BandStructure:
    """Level 2: ribbon plasmon bands from the material-free eigenvalues.

    One eigendecomposition of ``V @ D_geom`` per ``q``
    (``system.eigenmodes(q)``), reused for every band and every frequency -- the
    operator is explicitly ``eta``-free, so nothing is rebuilt per frequency.
    Bands follow from a scalar root find of ``Re eta(E) = eta_jq`` per band per
    ``q``, seeded by the local-Drude closed form :func:`drude_band_energy`, and
    linewidths from ``Im eta`` at the root.

    THIS IS A ``beta = 0`` METHOD.  With drift, ``eta`` becomes a tensor
    (``eta_perp != eta_par``), the ``eta``-free factorisation ``M = eta(E) A``
    that makes a single geometric spectrum meaningful no longer holds, and a band
    computed this way would simply be wrong.  The same applies to ribbons with
    different ``(W, EF, gamma)`` (no single scalar ``eta``) and to a DISPERSIVE
    substrate stack (one whose layers carry ``sigma_sheet``, so that ``r(Q,
    omega)`` and hence the screened operator depend on the frequency).  All
    three are rejected with ``ValueError`` rather than silently approximated;
    use Level 3 (:func:`ldos_map` + :func:`track_peaks`) in those cases.

    A PURELY DIELECTRIC stack IS supported, via ``screened=True``.  Its ``r(Q)``
    is exactly frequency independent, so ``(V + Vbar) @ D_geom``
    (``system.eigenmodes_screened``) is as geometric as the free-standing
    operator and the resonance condition is unchanged.  The spectrum is
    evaluated once per ``q`` at :data:`SCREEN_REFERENCE_EV`; that choice is
    immaterial, and ``tests/test_dispersion.py`` pins the energy independence
    bit-for-bit.  Screening makes every ``eta_j`` more negative (``r < 0`` for
    ``eps_substrate > 1``, so the image charge weakens the self interaction) and
    therefore red-shifts every band.

    Parameters
    ----------
    system
        ``solver.RibbonSystem``.
    qs
        Wave-vector grid [1/nm].  Use a small nonzero ``qW`` (e.g. ``1e-4/W``)
        for the ``q -> 0`` limit; ``eigenmodes`` is immune to the ``K_0``
        log divergence there because the physical modes are charge neutral.
        With ``screened=True`` the stack's reflected integral IS divergent at
        ``q = 0`` and ``solver`` raises, so keep ``q`` finite.
    n_bands
        Number of bands to return per ``q``.
    bright_only
        Keep only the dipole-active modes (``solver.bright_mode_mask``).  The
        default ``False`` keeps the dark modes too, so that e.g. the dark
        ``eta = -0.0289`` ribbon mode appears in the band structure with
        negligible ``weight`` instead of being silently dropped.

        IT ALSO KEEPS THE CHARGE-NON-NEUTRAL MODE, and that one can outrank the
        fundamental.  With the zero-flux ``D_geom`` a mode of non-zero mean
        potential necessarily carries non-zero total charge
        (``int rho dtheta = -(qW)^2 int phi dtheta / W``), and the near-uniform
        one has ``|eta| ~ 1/(qW)^2``: it is pushed out of ``[E_min, E_max]``
        only while ``qW`` is tiny (``eta = -4.6e6`` at ``qW = 1e-4``, i.e.
        ``E ~ 1e-4 eV``), whereas at ``qW = 0.1`` it already sits at
        ``eta = -13.0``, ``E = 0.037 eV`` and takes band 0 ahead of the
        fundamental at 0.575 eV.  It is dark (``|zeta| ~ 1e-13``), so
        ``bright_only=True`` removes it and gives a band ordering that is stable
        in ``q``; that is what the Level 2 / Level 3 cross-check uses.

        THAT MODE IS PHYSICAL, not a mesh artefact, and dropping it is a
        presentation choice rather than a correction.  The decisive evidence is
        that LEVEL 3 FINDS IT TOO: :func:`ldos_map` + :func:`track_peaks` build
        an induced-field map and peak-track it without ever diagonalising the
        geometric operator, and at ``q = 0.1``, ``L = 61``,
        ``r0 = (3, 0, 6)`` nm they return ``0.2713, 0.5767, 0.8894`` eV against
        this function's ``0.2708, 0.5764, 0.8893`` -- the branch agrees to
        2.1e-3 relative under a method that cannot inherit a discretisation
        artefact of the eigenproblem.

        THE SOURCE MUST BE OFF THE RIBBON AXIS for that check.  With the on-axis
        ``r0 = (0, 0, 6)`` that the other Level 3 tests use, ``track_peaks``
        returns only ``0.5767`` and ``1.1096`` -- the bright modes -- and this
        branch does not appear at all, because an on-axis dipole does not drive
        it.  Any ``x0 >= 1`` nm brings it out (0.27139 at 1 nm, 0.27134 from
        3 nm on, flat in the energy grid), so a reader reproducing the number
        with the house ``r0`` and finding nothing has not refuted it.

        Supporting it, measured at ``W = 10`` nm: ``eta`` converges in ``L`` (``-3.9558, -3.9817, -3.9975, -4.0027``
        at ``L = 61, 121, 301, 601``, ``q = 0.02``), its charge is ALL net
        charge (``h sum(rho) / h sum|rho| = -1.000`` to 14 digits, against
        ``1e-14`` for the DIPOLE-ACTIVE modes -- the higher dark ones are not
        exactly neutral, keeping ``-5.3e-4, -6.8e-5, -2.1e-5`` at modes 2, 4, 6,
        which does NOT shrink with the mesh, ``-5.287e-4`` at ``L = 121`` against
        ``-5.272e-4`` at ``L = 301``, and so is a property of the operator at
        finite ``q`` rather than a discretisation error; mode 0 is still the only
        one whose charge is ENTIRELY net charge, by three orders of magnitude),
        and ``1/[eta (qW)^2]`` is linear in
        ``ln(2/qW)`` with slope 2.00 -- the ``2 K_0`` kernel's logarithm.
        Inverting the lossless Drude ``eta`` then gives
        ``E ~ q sqrt(ln(1/qW))``, which is the one-dimensional wire plasmon of
        the ribbon: the longitudinal branch, dark to a transverse field by
        symmetry, going to zero energy as ``q -> 0``.  It is far below the light
        line there (0.023 eV against 0.99 eV at ``q = 0.005`` / nm), so the
        quasistatic treatment of it is sound.  Keep it with ``track=True`` if
        the band structure should show it; use ``bright_only=True`` if the plot
        is only about what a transverse field can drive.
    screened
        Diagonalise ``(V + Vbar) @ D_geom`` instead of ``V @ D_geom``, i.e. take
        the substrate into account (see above).  Requires ``system.layers`` and
        a non-dispersive stack; with ``layers=None`` it is identical to the
        default.
    track
        FOLLOW each band across the ``q`` grid instead of re-ranking the modes
        at every point.  USE THIS FOR A DISPERSION PLOT.

        The default (``False``) is the documented one-``q`` contract: band ``k``
        is the ``k``-th mode that produced a root, so the kept bands are
        contiguous from index 0.  Swept over ``q`` that contract draws
        discontinuities that are not in the physics, because the membership of
        the list changes underneath the index.  Two things change it: a branch
        whose root leaves ``[E_min, E_max]``, after which every later band
        slides down one slot, and two branches exchanging rank in ``|eta|``.
        The first is routine -- the longitudinal branch described under
        ``bright_only`` leaves the window at small ``q``.  For ``W = 10`` nm,
        ``E_min = 0.15`` eV on a 20-point grid over ``0.02 <= |q| <= 0.5`` it is
        gone from the two smallest ``|q|`` (0.020 and 0.045, lowest live point
        0.071), and the slide that follows puts a 0.37 eV step into a band whose
        true steps are 0.01 eV.  (The
        second was not observed for a single ribbon anywhere inside the
        ``qW <~ 6`` the Coulomb cross block is trustworthy over; ``track`` covers
        it because nothing guarantees it stays that way.)

        With ``track=True`` band ``k`` is one eigenvector followed from one
        ``q`` to the next by overlap (:func:`_follow_modes`), the bands are the
        ``n_bands`` leading eigenmodes rather than the first ``n_bands`` to find
        a root, and a band with no root at some ``q`` reports ``NaN`` THERE and
        keeps its slot -- so a branch that leaves the window leaves a gap in its
        own curve instead of shifting every curve above it.  ``mode_index``
        then says which eigenmode each band is at each ``q``, and it is no
        longer constant along a band once two branches cross.

        The operator depends on ``q`` only through ``|q|``, so a ``+-q`` grid
        matches across ``q = 0`` exactly and the two halves cannot disagree.
    E_min, E_max
        Search window for the root find [eV].  Roots outside it are reported as
        ``NaN``, and a mode whose seed lies outside is skipped rather than
        clamped to the boundary.
    rel_step
        Relative step for the central difference of ``Re eta`` used by the
        linewidth.

    Returns
    -------
    BandStructure

    Raises
    ------
    ValueError
        If ``system`` has drift, ribbons that do not share one
        ``(W_nm, EF_ev, gamma_ev)``, a substrate with ``screened=False``, or a
        dispersive substrate; or if ``n_bands < 1`` or the energy window is
        empty.
    """
    reason = _band_obstruction(system, bool(screened))
    if reason is not None:
        raise ValueError(
            "ribbon_bands is a beta == 0, single-eta, frequency-independent-operator "
            "method (see its docstring): " + reason
            + " -- use ldos_map() + track_peaks() instead"
        )
    nb = int(n_bands)
    if nb < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if not 0.0 < float(E_min) < float(E_max):
        raise ValueError(f"need 0 < E_min < E_max, got ({E_min}, {E_max})")

    q_arr = np.atleast_1d(_f(qs)).copy()
    nq = q_arr.size
    rib0 = system.ribbons[0]
    eps_bar = system.eps_eff

    eta_out = np.full((nq, nb), np.nan, dtype=np.complex128)
    zeta_out = np.full((nq, nb), np.nan, dtype=np.complex128)
    weight_out = np.full((nq, nb), np.nan, dtype=np.float64)
    bright_out = np.zeros((nq, nb), dtype=bool)
    E_out = np.full((nq, nb), np.nan, dtype=np.float64)
    fwhm_out = np.full((nq, nb), np.nan, dtype=np.float64)
    Ed_out = np.full((nq, nb), np.nan, dtype=np.float64)
    midx_out = np.full((nq, nb), -1, dtype=np.int64)

    def eta_of_E(q: float, E: float) -> complex:
        # beta == 0 is guaranteed above, so eta_perp == eta_par and the pair is
        # in fact independent of q; the signature keeps q explicit anyway.
        return system.eta_pair(0, q, E)[0]

    prev_vecs: Optional[NDArray[np.complex128]] = None
    has_prev = np.zeros(nb, dtype=bool)
    #: The eta each band last held, used only to tell which branch is the one
    #: running away to |eta| = infinity when a null eigenvalue turns up.
    prev_eta = np.zeros(nb, dtype=np.float64)

    # THE ORDER THE TRACKED SWEEP WALKS THE GRID, and it is not the grid's own.
    #
    # `_band_obstruction` has guaranteed `beta == 0`, so the operator depends on
    # q only through |q| and the spectra at +q and -q are bit-identical. Walk
    # the distinct |q| instead, SMALLEST FIRST, and write each result to both
    # signs: half the eigendecompositions, and the two halves of the picture are
    # identical by construction rather than by the tracker happening to make the
    # same choices twice.
    #
    # `q = 0` IS LEFT OUT OF THE CHAIN and done last, seeded from the nearest
    # nonzero |q| without ever becoming the reference for what follows. There
    # the near-uniform mode's eigenvalue is 2.4e-16 of the largest -- machine
    # epsilon is 2.2e-16 -- so its eigenvector is numerically UNDEFINED and
    # LAPACK may return any vector in that direction. Desktop OpenBLAS happens
    # to return the uniform mode; the browser's BLAS returned something else,
    # and because the chain ran straight through q = 0 every band above it was
    # then matched against garbage: band 0 came out of q = 0 holding a
    # high-order mode and kept it for the whole positive half. It reproduced
    # nowhere in CPython, which is exactly what a platform-dependent null space
    # looks like. One grid point short of zero the same eigenvalue is 6.3e-4 of
    # the largest, twelve orders of magnitude above the noise, so everything
    # except the exact zero is well conditioned.
    if track:
        absq = np.abs(q_arr)
        chain = [float(u) for u in np.unique(absq) if u > 0.0]
        tail = [0.0] if np.any(absq == 0.0) else []
        order = chain + tail
    else:
        order = [float(q_arr[iq]) for iq in range(nq)]

    for step, q in enumerate(order):
        if screened:
            # ONE eigendecomposition; the energy is immaterial for a
            # non-dispersive stack, which _band_obstruction has guaranteed.
            eta_g, modes_g, zeta_g = system.eigenmodes_screened(q, SCREEN_REFERENCE_EV)
        else:
            eta_g, modes_g, zeta_g = system.eigenmodes(q)  # ONE eigendecomposition
        mask_bright = bright_mode_mask(zeta_g)
        idx = np.arange(eta_g.size)
        if bright_only:
            idx = idx[mask_bright]
        # Every row this q's result belongs to: both signs when tracking.
        rows = (
            np.flatnonzero(np.abs(q_arr) == q) if track else np.array([step], dtype=np.int64)
        )

        if track:
            # Band k is an eigenVECTOR followed from the last |q|, so the slot
            # survives a root leaving the window and a crossing in |eta|.
            # Band k is an eigenVECTOR followed from the last |q|, so the slot
            # survives a root leaving the window and a crossing in |eta|.
            pool = idx.astype(np.int64)
            null_pool = np.empty(0, dtype=np.int64)

            if q == 0.0 and pool.size:
                # STRUCTURAL, and taken BEFORE any filtering. At q = 0, D_geom
                # is the plain second difference under its zero-flux closure
                # and its null vector is the uniform potential, so
                # A = V @ D_geom HAS an exactly zero eigenvalue -- one of them,
                # always, as a fact about the operator rather than about the
                # arithmetic. The mode carrying it is the one with the largest
                # |eta|, safe by a wide margin: the physical branches sit at
                # |eta| ~ 0.0075 to 0.07 while this one is 1e13.
                #
                # TAKEN FIRST BECAUSE ITS SIGN IS MEANINGLESS. `eta = 1/lambda`
                # with lambda at machine zero, so the sign is whatever the
                # rounding left behind: at L = 121 this machine returns
                # eta = -2.2e12 and at L = 61 it returns +4.0e13, for the same
                # physics. The `eta < 0` test below -- correct for every real
                # branch, since a positive eta has no resonance -- therefore
                # threw the null mode away about half the time, and whatever
                # was then picked as "largest |eta|" was the FUNDAMENTAL, which
                # got reported at zero energy while its own band went empty.
                # Filter for a physical sign only after the unphysical mode has
                # been set aside.
                # ONE PER RIBBON, not one in total. D_geom is BLOCK
                # diagonal -- one block per ribbon -- so each ribbon brings its
                # own uniform-potential null vector and an N-ribbon system has
                # N zero eigenvalues, N longitudinal branches and N endpoints.
                # Taking only the largest rescued one of them and left the rest
                # to the sign test below, which is a coin flip: measured at
                # q = 0, five identical ribbons gave etas
                # +2.10e12 +1.69e12 -1.40e12 -1.14e12 +7.19e11, three of which
                # fail `eta < 0`. The q = 0 row of a four-ribbon run read
                # "null null null 0.000".
                #
                # `ribbon_bands` makes this the ONLY multi-ribbon case it ever
                # sees: `_band_obstruction` requires every ribbon to share one
                # (W, EF, gamma), so a multi-ribbon system here is always an
                # array of IDENTICAL ribbons.
                lam_p = np.abs(1.0 / np.real(eta_g[pool]))
                is_null = lam_p <= _NULL_EIGENVALUE * float(np.max(lam_p))
                if not np.any(is_null):
                    # The operator says there is at least one; if rounding put
                    # none under the tolerance, take the single smallest.
                    is_null = lam_p == float(np.min(lam_p))
                null_pool = pool[is_null].astype(np.int64)
                pool = pool[~is_null]

            elif pool.size:
                # Away from zero there is no structural guarantee, so the guard
                # stays a tolerance -- a backstop against an accidental
                # near-degeneracy, not the definition of anything. It runs HERE,
                # before the sign filter, and not after it: a mode the sign test
                # has already discarded can never reach a backstop placed below
                # it, which was exactly the mistake that let the null mode
                # vanish in the first place.
                lam_p = np.abs(1.0 / np.real(eta_g[pool]))
                is_null = lam_p <= _NULL_EIGENVALUE * float(np.max(lam_p))
                null_pool = pool[is_null].astype(np.int64)
                pool = pool[~is_null]

            ok = np.asarray(
                [
                    bool(np.isfinite(np.real(eta_g[m])) and np.real(eta_g[m]) < 0.0)
                    for m in pool
                ],
                dtype=bool,
            )
            cand = pool[ok].astype(np.int64)
            sel_t = _follow_modes(prev_vecs, has_prev, modes_g, cand, nb, eta_g)

            # THE LONGITUDINAL BRANCH'S EXACT ENDPOINT.
            #
            # A null eigenvalue is not a failure to resolve something; at q = 0
            # it is exact. D_geom is d^2_theta - (qW)^2, so at q = 0 it is the
            # plain second difference under its zero-flux closure, whose null
            # vector is the uniform potential -- A = V @ D_geom then has an
            # EXACTLY zero eigenvalue, eta = 1/lambda is infinite, and
            # Re eta(E) = -e^2 E_F / (pi W eps (E^2 + gamma^2)) can only meet
            # it at E = 0. So the branch ends at the origin, exactly, for any
            # gamma, and none of that needs the eigenVECTOR -- which is just as
            # well, because that is the one thing here that is undefined.
            #
            # Which band it belongs to is settled by elimination rather than by
            # matching: it is the slot left empty whose last eta was largest in
            # magnitude, i.e. the branch already on its way to infinity. Without
            # this the curve simply stopped at the last grid point before zero,
            # a visible gap at the middle of the picture where the physics is
            # at its least ambiguous.
            zero_bands: set = set()
            if null_pool.size:
                # Every empty slot that was heading for infinity gets one, the
                # largest |eta| first, until the endpoints run out. With one
                # ribbon that is the single branch it always was; with N it is
                # all N of them, which is what the picture needs.
                empty = sorted(
                    (b for b in range(nb) if sel_t[b] < 0 and bool(has_prev[b])),
                    key=lambda b: -abs(float(prev_eta[b])),
                )
                for b_end, m_end in zip(empty, [int(x) for x in null_pool]):
                    sel_t[b_end] = m_end
                    zero_bands.add(b_end)
            if prev_vecs is None:
                prev_vecs = np.zeros((modes_g.shape[0], nb), dtype=np.complex128)
            # An eigenvector is only worth remembering if its eigenvalue is
            # resolved. This is the same guard as leaving q = 0 out of the
            # chain, applied per mode, so no future near-degeneracy can poison
            # the reference either.
            lam = np.abs(1.0 / np.real(eta_g[idx])) if idx.size else np.zeros(0)
            lam_max = float(np.max(lam)) if lam.size else 0.0
            resolved = (
                (lambda m: abs(1.0 / float(np.real(eta_g[m]))) > _NULL_EIGENVALUE * lam_max)
                if lam_max > 0.0
                else (lambda m: False)
            )
            for b in range(nb):
                m = int(sel_t[b])
                if m < 0:
                    continue
                if resolved(m):
                    prev_vecs[:, b] = modes_g[:, m]
                    prev_eta[b] = float(np.real(eta_g[m]))
                    has_prev[b] = True
                ej = float(np.real(eta_g[m]))
                if b in zero_bands:
                    # The exact endpoint above. No root find: there is nothing
                    # to solve, and `drude_band_energy` would return NaN here
                    # because the lossless energy has fallen below gamma --
                    # true, and the reason the band has no WIDTH at q = 0, but
                    # not a reason to leave its last point off the picture.
                    e0 = 0.0
                    root = 0.0
                else:
                    e0 = float(
                        drude_band_energy(
                            ej, rib0.W_nm, rib0.EF_ev, eps_bar=eps_bar, gamma_ev=rib0.gamma_ev
                        )
                    )
                    root = _solve_eta_root(eta_of_E, q, ej, e0, float(E_min), float(E_max))
                # The band keeps its slot either way; only the ENERGY is NaN
                # where this branch has no resonance inside the window. Written
                # to BOTH signs of this |q| at once -- the operator cannot tell
                # them apart, so neither may the picture.
                eta_out[rows, b] = eta_g[m]
                zeta_out[rows, b] = zeta_g[m]
                bright_out[rows, b] = bool(mask_bright[m])
                midx_out[rows, b] = m
                Ed_out[rows, b] = e0
                if root is not None:
                    E_out[rows, b] = root
                    # No linewidth at the endpoint: a zero-energy mode has no
                    # resonance to be wide, and the central difference would
                    # divide by zero anyway.
                    if b not in zero_bands:
                        fwhm_out[rows, b] = _linewidth(eta_of_E, q, root, float(rel_step))
            live = sel_t >= 0
            if live.any():
                z2 = np.abs(zeta_out[rows[0], live]) ** 2
                zmax = float(np.max(z2))
                for r in rows:
                    weight_out[r, live] = z2 / zmax if zmax > 0.0 else 0.0
            continue

        # eigenmodes already returns descending |eta|; keep that order
        picked: list[int] = []
        roots: list[float] = []
        widths: list[float] = []
        closed: list[float] = []
        for m in idx:
            ej = float(np.real(eta_g[m]))
            if not np.isfinite(ej) or ej >= 0.0:
                continue
            e0 = float(
                drude_band_energy(
                    ej, rib0.W_nm, rib0.EF_ev, eps_bar=eps_bar, gamma_ev=rib0.gamma_ev
                )
            )
            root = _solve_eta_root(eta_of_E, q, ej, e0, float(E_min), float(E_max))
            if root is None:
                continue
            fw = _linewidth(eta_of_E, q, root, float(rel_step))
            picked.append(int(m))
            roots.append(root)
            widths.append(fw)
            closed.append(e0)
            if len(picked) == nb:
                break
        if picked:
            iq = int(rows[0])
            k = len(picked)
            sel = np.asarray(picked, dtype=int)
            eta_out[iq, :k] = eta_g[sel]
            zeta_out[iq, :k] = zeta_g[sel]
            bright_out[iq, :k] = mask_bright[sel]
            E_out[iq, :k] = roots
            fwhm_out[iq, :k] = widths
            Ed_out[iq, :k] = closed
            midx_out[iq, :k] = sel
            z2 = np.abs(zeta_g[sel]) ** 2
            zmax = float(np.max(z2))
            weight_out[iq, :k] = z2 / zmax if zmax > 0.0 else 0.0

    return BandStructure(
        qs=q_arr,
        eta=eta_out,
        zeta=zeta_out,
        weight=weight_out,
        bright=bright_out,
        E_ev=E_out,
        fwhm_ev=fwhm_out,
        E_drude_ev=Ed_out,
        mode_index=midx_out,
    )


def _solve_eta_root(
    eta_of_E: Any,
    q: float,
    eta_j: float,
    E_seed: float,
    E_min: float,
    E_max: float,
) -> Optional[float]:
    """Root of ``Re eta(E) = eta_j`` in ``[E_min, E_max]``, or ``None``.

    ``Re eta`` is negative and monotonically increasing towards zero for the
    local Drude model, so a single sign change is expected; the seed from
    :func:`drude_band_energy` gives a tight bracket, and the full window is the
    fallback for any other conductivity model.  Returns ``None`` -- never a
    nearest grid point -- when no bracket can be found.
    """

    def g(E: float) -> float:
        return float(np.real(eta_of_E(q, E))) - eta_j

    brackets: list[tuple[float, float]] = []
    if np.isfinite(E_seed) and E_seed > 0.0:
        lo = max(E_min, 0.5 * E_seed)
        hi = min(E_max, 2.0 * E_seed)
        if lo < hi:
            brackets.append((lo, hi))
    brackets.append((E_min, E_max))
    for lo, hi in brackets:
        try:
            glo, ghi = g(lo), g(hi)
        except (ZeroDivisionError, FloatingPointError, ValueError):
            continue
        if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0.0:
            continue
        if glo == 0.0:
            return lo
        if ghi == 0.0:
            return hi
        return float(brentq(g, lo, hi, xtol=1e-12, rtol=8.9e-16))
    return None


def _linewidth(eta_of_E: Any, q: float, E_star: float, rel_step: float) -> float:
    """``FWHM = 2 |Im eta(E*)| / |d Re eta/dE|_{E*}``  [eV], ``NaN`` if degenerate.

    From ``1 - eta(E) lambda_j = -lambda_j (eta(E) - eta_j)``: expanding about
    the root gives a Lorentzian in ``E`` with half width ``|Im eta| / |Re eta'|``.
    """
    dE = max(abs(rel_step) * E_star, 1e-9)
    try:
        hi = float(np.real(eta_of_E(q, E_star + dE)))
        lo = float(np.real(eta_of_E(q, max(E_star - dE, 1e-12))))
        im = float(np.imag(eta_of_E(q, E_star)))
    except (ZeroDivisionError, FloatingPointError, ValueError):
        return float("nan")
    slope = (hi - lo) / (2.0 * dE)
    if not np.isfinite(slope) or slope == 0.0:
        return float("nan")
    return float(2.0 * abs(im) / abs(slope))


# --------------------------------------------------------------------------- #
# Level 3 -- LDOS map
# --------------------------------------------------------------------------- #
def _ldos_from_field(E_vec: NDArray[np.complex128], p: NDArray[np.complex128]) -> float:
    """``D(q) = Im(conj(p) . E) / (2 pi^2 |p|^2)`` -- paper 456, as in ``observables``.

    Duplicated from ``observables.ldos_q`` only so that the frequency loop can
    reuse a precomputed linear field map instead of re-evaluating the Bessel
    kernels at every frequency; ``tests/test_dispersion.py`` pins the two against
    each other to 1e-12.
    """
    p2 = float(np.vdot(p, p).real)
    return float(np.vdot(p, E_vec).imag) / (2.0 * np.pi**2 * p2)


def _field_map(system: RibbonSystem, q: float, r0_nm: Sequence[float]) -> NDArray[np.complex128]:
    """``(3, N)`` matrix ``G`` with ``E_q = G @ rho_flat``, for one ``q``.

    ``observables.induced_field_q`` is exactly linear in ``rho`` (it is a sum of
    quadratures of ``rho`` against frequency-independent Bessel kernels), so one
    call per node basis vector builds the whole map; verified to 2e-16 against
    a direct call on a random ``rho``.  Costs ``N`` calls of ``O(N)`` once per
    ``q`` instead of one ``O(N)`` call per frequency, which is the win whenever
    the energy grid is longer than ``N``.
    """
    G = np.empty((3, system.N), dtype=np.complex128)
    basis = [np.zeros(n, dtype=np.float64) for n in system.L]
    for j, sl in enumerate(system.slices):
        for l in range(system.L[j]):
            basis[j][l] = 1.0
            G[:, sl.start + l] = observables.induced_field_q(
                q, system.x_nodes, basis, r0_nm
            )
            basis[j][l] = 0.0
    return G


def ldos_map(
    system: RibbonSystem,
    qs: ArrayLike,
    Es: ArrayLike,
    r0_nm: Sequence[float],
    p: Sequence[complex] = (1.0, 0.0, 0.0),
    *,
    quantity: str = "ldos",
    fast_path: bool = True,
) -> NDArray[np.float64]:
    """Level 3: LDOS (or ``|E_q|``) over the ``(q, E)`` plane from the full solve.

    A point dipole at ``r0_nm`` drives the system
    (``sources.phi_ext_dipole``), the self-consistent charge is solved for, and
    ``observables`` turns it into the ``q``-resolved loss probability density
    ``D(q, E) = Im(conj(p) . E_q)/(2 pi^2 |p|^2)`` (paper 456).  This is how
    paper 456 extracts its dispersions, and -- unlike Level 2 -- it shows mode
    WEIGHT as well as position, which matters because drift both shifts and
    quenches bands.  Feed the result to :func:`track_peaks` for the bands and to
    :func:`pm_asymmetry` for the non-reciprocity map.

    COST.  Two routes:

    * ``fast_path=True`` and no obstruction (``beta = 0``, no substrate, one
      common ``(W, EF, gamma)``): ONE eigendecomposition and one eigenvector
      inversion per ``q``, both ``O(N^3)``, then ``O(N^2)`` per frequency --
      the ``eta``-free operator is never rebuilt.  Measured at ``L = 121``,
      ``nE = 601``: ~0.11 s per ``q`` against ~0.72 s for the full route, a 6.6x
      speed-up, agreeing with it to 5e-12 relative.  The speed-up grows with
      ``nE/N`` and is smaller than the naive ``O(N^3)`` vs ``O(N^2)`` estimate
      because the full route's per-frequency cost is dominated by a BLAS-3
      ``V @ D`` product and a LAPACK solve, both of which run near peak.
    * otherwise: one ``O(N^3)`` solve per ``(q, E)``.  Required for drift, which
      is exactly the interesting case, so keep the grids modest.

    Either route evaluates the Bessel kernels of the induced field only once per
    ``q`` when ``len(Es) > system.N`` (see :func:`_field_map`).

    LIMITATION INHERITED FROM ``observables``.  ``observables.induced_field_q``
    places every source node at ``z = 0`` and measures ``r0``'s height from that
    plane, so a stack of ribbons at different ``z0_nm`` is not represented.  A
    ``RuntimeWarning`` fires if any ribbon has ``z0_nm != 0``.

    Parameters
    ----------
    system
        ``solver.RibbonSystem``.
    qs
        Wave-vector grid [1/nm], SIGNED and nonzero (``phi_ext_dipole`` has no
        ``q = 0`` component).  Include both signs for :func:`pm_asymmetry`.
    Es
        Photon energies [eV], strictly positive.
    r0_nm
        Dipole position ``(x0, y0, z0)`` or ``(x0, z0)`` [nm]; must not coincide
        with a node.
    p
        Dipole moment ``(px, py, pz)``; may be complex.  The default ``x``
        dipole couples to the bright (dipole-active) ribbon modes.
    quantity
        ``"ldos"`` for the loss probability density (default) or ``"absE"`` for
        the Euclidean norm ``|E_q|`` of the induced field vector.
    fast_path
        Allow the eigen-reuse route.  ``False`` forces a full solve per
        ``(q, E)``; the two agree to ~1e-11 relative at ``L = 201`` and the
        usual reason to pass it is a cross-check.  The eigen route expands the
        source in the eigenbasis, so it needs ``A`` to be diagonalisable; a
        defective ``A`` would make ``numpy.linalg.solve`` raise, and
        ``fast_path=False`` is then the fallback.  It is also ignored (silently,
        since it cannot change the answer) whenever
        ``system.fast_path_obstruction()`` is not ``None``.

    Returns
    -------
    ndarray of float64
        Shape ``(len(qs), len(Es))``.

    Raises
    ------
    ValueError
        For an unknown ``quantity``, a zero dipole, or ``q = 0`` in ``qs``.
    """
    if quantity not in ("ldos", "absE"):
        raise ValueError(f"quantity must be 'ldos' or 'absE', got {quantity!r}")
    q_arr = np.atleast_1d(_f(qs))
    E_arr = np.atleast_1d(_f(Es))
    if np.any(q_arr == 0.0):
        raise ValueError(
            "q = 0 has no point-dipole Fourier component (K_0/K_1 diverge); "
            "use a small nonzero q instead"
        )
    pv = _c(p).ravel()
    if pv.size != 3:
        raise ValueError(f"p must have 3 components (px, py, pz), got {pv.size}")
    if float(np.vdot(pv, pv).real) == 0.0:
        raise ValueError("p must be a non-zero dipole moment")
    if any(rib.z0_nm != 0.0 for rib in system.ribbons):
        warnings.warn(
            "observables.induced_field_q places all source nodes at z = 0, so the "
            "LDOS of ribbons at different z0_nm is not represented; the map will "
            "treat every ribbon as coplanar with the dipole's reference plane",
            RuntimeWarning,
            stacklevel=2,
        )

    use_eigen = bool(fast_path) and system.fast_path_obstruction() is None
    use_G = E_arr.size > system.N
    out = np.empty((q_arr.size, E_arr.size), dtype=np.float64)

    for iq in range(q_arr.size):
        q = float(q_arr[iq])
        rhs = [
            sources.phi_ext_dipole(
                q, system.x_nodes[j], rib.z0_nm, r0_nm, pv, system.eps_a
            )
            for j, rib in enumerate(system.ribbons)
        ]
        rhs_flat = np.concatenate(rhs).astype(np.complex128)
        G = _field_map(system, q, r0_nm) if use_G else None

        if use_eigen:
            eta_g, modes, _zeta = system.eigenmodes(q)
            modes = np.ascontiguousarray(modes, dtype=np.complex128)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(np.isfinite(eta_g), 1.0 / _c(eta_g), 0.0)
            lam = np.where(np.isfinite(lam), lam, 0.0)
            coef0 = np.linalg.solve(modes, rhs_flat)
            # rho = eta * D_geom phi / W, block by block; fold into one matrix
            R = np.empty_like(modes)
            for j, rib in enumerate(system.ribbons):
                sl = system.slices[j]
                Dg = np.asarray(
                    operators.build_D_geom(q, rib.W_nm, system.h[j], system.L[j]),
                    dtype=np.complex128,
                )
                R[sl, :] = (Dg @ modes[sl, :]) / rib.W_nm
            for iE in range(E_arr.size):
                E = float(E_arr[iE])
                eta = system.eta_pair(0, q, E)[0]
                rho_flat = eta * (R @ (coef0 / (1.0 - eta * lam)))
                out[iq, iE] = _reduce_field(system, q, rho_flat, r0_nm, pv, G, quantity)
        else:
            for iE in range(E_arr.size):
                E = float(E_arr[iE])
                sol = system.solve_linear(q, E, rhs)
                out[iq, iE] = _reduce_field(
                    system, q, sol["rho_flat"], r0_nm, pv, G, quantity
                )
    return out


def _reduce_field(
    system: RibbonSystem,
    q: float,
    rho_flat: NDArray[np.complex128],
    r0_nm: Sequence[float],
    pv: NDArray[np.complex128],
    G: Optional[NDArray[np.complex128]],
    quantity: str,
) -> float:
    """One map entry: induced field at ``r0``, reduced to LDOS or ``|E_q|``."""
    if G is not None:
        E_vec = G @ rho_flat
    else:
        rho_list = [rho_flat[sl] for sl in system.slices]
        E_vec = np.asarray(
            observables.induced_field_q(q, system.x_nodes, rho_list, r0_nm),
            dtype=np.complex128,
        )
    if quantity == "absE":
        return float(np.sqrt(np.vdot(E_vec, E_vec).real))
    return _ldos_from_field(E_vec, pv)


def track_peaks(
    Es: ArrayLike,
    values: ArrayLike,
    *,
    n_peaks: int = 3,
    rel_prominence: float = 1e-3,
    sort: str = "weight",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Locate band frequencies and weights in a map, honestly.

    Finds interior local maxima along the LAST axis of ``values`` (the energy
    axis, as :func:`ldos_map` returns), refines each to sub-grid accuracy by a
    three-point parabolic fit, and returns the refined positions and heights.

    NO PEAK MEANS ``NaN``, never the nearest grid point.  A maximum that sits on
    the first or last energy sample is not a resolved peak and is dropped, as is
    anything whose prominence falls below ``rel_prominence`` times the row's
    peak-to-peak range.  A row with fewer than ``n_peaks`` resolved peaks is
    padded with ``NaN``, and a row that is all ``NaN`` or constant gives all
    ``NaN``.

    Parameters
    ----------
    Es
        Energy grid [eV], strictly increasing.  Parabolic refinement is applied
        only on a uniform grid; on a non-uniform one the grid node is returned.
    values
        Map, shape ``(..., len(Es))``.  Real.  Peaks are MAXIMA, so hand it a
        quantity whose sign convention makes resonances positive.
    n_peaks
        Number of peaks to return per row.
    rel_prominence
        Prominence floor relative to each row's peak-to-peak range.
    sort
        ``"weight"`` (default) orders the returned peaks by descending height,
        ``"energy"`` by ascending energy.

    Returns
    -------
    (E_peaks, weights)
        Both ``(..., n_peaks)`` float64, ``NaN``-padded.

    Raises
    ------
    ValueError
        For a bad ``sort``, a non-increasing ``Es``, ``n_peaks < 1``, or a
        trailing axis of ``values`` that does not match ``len(Es)``.
    """
    if sort not in ("weight", "energy"):
        raise ValueError(f"sort must be 'weight' or 'energy', got {sort!r}")
    npk = int(n_peaks)
    if npk < 1:
        raise ValueError(f"n_peaks must be >= 1, got {n_peaks}")
    E = _f(Es).ravel()
    if E.size < 3:
        raise ValueError("need at least 3 energy samples to resolve an interior peak")
    if np.any(np.diff(E) <= 0.0):
        raise ValueError("Es must be strictly increasing")
    vals = _f(values)
    if vals.shape[-1] != E.size:
        raise ValueError(
            f"values last axis {vals.shape[-1]} != len(Es) {E.size}"
        )
    dE = np.diff(E)
    uniform = bool(np.allclose(dE, dE[0], rtol=1e-9, atol=0.0))

    flat = vals.reshape(-1, E.size)
    pos = np.full((flat.shape[0], npk), np.nan, dtype=np.float64)
    wgt = np.full((flat.shape[0], npk), np.nan, dtype=np.float64)

    for i in range(flat.shape[0]):
        y = flat[i]
        if not np.all(np.isfinite(y)):
            continue
        span = float(np.max(y) - np.min(y))
        if span <= 0.0:
            continue
        idx, _props = find_peaks(y, prominence=max(float(rel_prominence) * span, 0.0))
        idx = idx[(idx > 0) & (idx < E.size - 1)]
        if idx.size == 0:
            continue
        if uniform:
            ym, y0, yp = y[idx - 1], y[idx], y[idx + 1]
            curv = ym - 2.0 * y0 + yp
            delta = np.where(curv != 0.0, 0.5 * (ym - yp) / np.where(curv != 0.0, curv, 1.0), 0.0)
            delta = np.clip(delta, -0.5, 0.5)
            e_ref = E[idx] + delta * dE[0]
            h_ref = y0 - 0.25 * (ym - yp) * delta
        else:
            e_ref = E[idx]
            h_ref = y[idx]
        order = np.argsort(-h_ref, kind="stable") if sort == "weight" else np.argsort(e_ref, kind="stable")
        take = order[:npk]
        pos[i, : take.size] = e_ref[take]
        wgt[i, : take.size] = h_ref[take]

    shape = vals.shape[:-1] + (npk,)
    return pos.reshape(shape), wgt.reshape(shape)


#: How many times the sweep's OWN median branch speed a peak may move between
#: adjacent columns and still be the same branch.
#:
#: Not an energy, because an energy does not separate the two cases.  Measured on
#: Level 2 bands -- which ``ribbon_bands(track=True)`` follows by eigenvector
#: overlap and which therefore never swap slots -- a real branch reaches
#: 4.68 eV per unit q (W = 30 nm, E_F = 0.9 eV, 80 columns), while the slot swaps
#: this function exists to prevent sat at 5.7 eV per unit q on the page.  A
#: factor 1.2 is not a threshold.
#:
#: Calibrating on the sweep's own accepted steps instead gives a wide plateau.
#: Over eight maps (beta 0 / 0.26 / 0.40, one to five ribbons, W 6-30 nm,
#: E_F 0.3-0.9 eV, 32 and 80 columns) the worst single step is 0.0583 eV for
#: EVERY value from 2 to 30, and jumps to 1.0514 eV at 50 -- where a stranger
#: finally gets through.  6 sits inside that plateau with a factor 5 to the
#: failure and a factor 3 to the tight edge.
_PEAK_GATE_SPEED: float = 6.0

#: Floor under that gate, in energy-grid cells, for the columns before a sweep
#: has enough accepted steps to have a median at all.
#:
#: Same eight maps: 2 cells keeps only 532 of 811 resolved peaks and 4 cells
#: keeps 710 -- the gate is rejecting real branches.  From 12 cells the answer
#: stops moving (808 kept at 12, 808 at 25), so 12 is the start of that plateau
#: and not a number chosen to make one case look good.
_PEAK_GATE_FLOOR_CELLS: float = 12.0

#: Consecutive columns a track may be absent before its slot is free for a
#: different branch.
#:
#: Worth little -- 807 peaks kept at 1 against 809 at 0 and 805 at 3 -- but it is
#: what turns the failure this function was written for into an honest GAP: at
#: beta = 0.26 slot 2's branch ends mid-sweep, and without a delay the slot is
#: refilled in the very next column by a peak 1.05 eV away, drawing a vertical
#: line across three quarters of the panel.  One blank column breaks the stroke,
#: so the eye reads two segments instead of one impossible curve.
_PEAK_MAX_MISS: int = 1


def follow_peaks(
    qs: ArrayLike,
    Es: ArrayLike,
    values: ArrayLike,
    *,
    n_peaks: int = 3,
    rel_prominence: float = 1e-3,
    sort: str = "weight",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Peak tracks across a sweep, each slot staying on ONE branch.

    :func:`track_peaks` does not track.  It finds the peaks of each row on its
    own and sorts that row by descending height, so slot ``k`` means "the k-th
    tallest peak IN THIS COLUMN".  The instant two branches swap height -- which
    happens wherever a map has a bright region -- slot ``k`` swaps with it, and
    the curve drawn through it steps vertically between two branches that were
    never connected.  Measured on the page's own panel (W = 12 nm, beta = 0.26,
    32 columns) slot 0 stepped 0.371 eV in one column, 26% of the plotted
    window, with a typical real step of 0.05 eV.

    This walks the sweep instead.  Each column's peaks are assigned to the slots
    they are nearest to in energy -- greedily over the whole cost matrix, as
    :func:`ribbon_bands`' own tracker does with eigenvector overlaps -- so the
    question stops being "which is tallest here" and becomes "which branch is
    this".  On the same panel the worst step falls to 0.048 eV.

    NOTHING IS INVENTED.  A slot with no candidate inside the gate gets ``NaN``
    for that column, exactly as :func:`track_peaks` would; a branch that ends
    ends, and the gap is drawn as a gap.  The ONE thing this adds beyond
    assignment is the refusal to let a slot whose branch has ended be refilled
    by a stranger in the next column (see :data:`_PEAK_MAX_MISS`), which is what
    a continuity test cannot catch by itself: a perfectly continuous-looking
    curve made of two different branches is worse than a broken one.

    WHAT THIS CANNOT DO, measured rather than assumed.  At an ANTICROSSING --
    two branches approaching to a gap comparable with their own linewidth and
    separating again -- which slot comes out on which branch is not determined,
    and no amount of care in the gate fixes it.  On synthetic pairs of Gaussian
    ridges of width 0.045 eV, three trackers built independently for this (a
    nearest-energy walk, a global ridge-grouping pass, and an optimal
    assignment using ``scipy.optimize.linear_sum_assignment``) disagreed with
    this one and with each other once the gap fell to 0.08 eV, and the number
    of points any of them could draw collapsed from 82 to 41 -- because below
    that the two maxima stop resolving as two peaks at all.  Above it, at a gap
    of 0.20 eV, all four agreed exactly and none made an error.  So the honest
    statement is: where the map resolves two ridges, the slot follows one of
    them; where it does not, the labelling is a convention and this one follows
    the ridge it is already on.

    AND WHY THE GATE IS ONLY A DEATH TEST.  It would be natural to let the gate
    decide identity as well -- reject a candidate that is too far and the slot
    keeps its branch.  It cannot: an absolute ``|dE|`` does not separate the two
    populations, because correct links reach 0.1016 eV while wrong ones start at
    0.0199 eV, and expressing the same thing in grid cells only moves the
    numbers (9.1 against 2.0).  Identity here is settled by the ASSIGNMENT --
    which pairing of the whole column is cheapest overall -- and the gate does
    the one job the assignment cannot, which is to notice that a branch has
    stopped existing.

    Parameters
    ----------
    qs
        Sweep coordinate, one per row of ``values``.  Any order the caller
        likes; the walk follows the array as given, and the gate divides by the
        actual spacing, so a non-uniform grid is fine.
    Es
        Energy grid [eV], strictly increasing.  As :func:`track_peaks`.
    values
        Map, shape ``(len(qs), len(Es))``.  Real, resonances positive.
    n_peaks
        How many branches to follow.  Slots are a budget: where more branches
        are resolved than there are slots, the strongest survive.
    rel_prominence
        Passed to :func:`track_peaks` for the per-column peak finding.
    sort
        What slot order MEANS once the tracking is done.  ``"weight"``
        (default) orders slots by total weight over the whole sweep, descending
        -- the sweep-wide reading of :func:`track_peaks`' own default, and the
        reason it is still the default here.  ``"energy"`` orders by mean
        energy, ascending.  Either way the order is decided ONCE for the whole
        sweep, which is the entire point: a per-column order is the bug.

    Returns
    -------
    (E_peaks, weights)
        Both ``(len(qs), n_peaks)`` float64, ``NaN``-padded.

    Raises
    ------
    ValueError
        For a bad ``sort``, ``n_peaks < 1``, a ``values`` whose shape does not
        match ``(len(qs), len(Es))``, or anything :func:`track_peaks` rejects.
    """
    if sort not in ("weight", "energy"):
        raise ValueError(f"sort must be 'weight' or 'energy', got {sort!r}")
    npk = int(n_peaks)
    if npk < 1:
        raise ValueError(f"n_peaks must be >= 1, got {n_peaks}")
    q = _f(qs).ravel()
    E = _f(Es).ravel()
    vals = _f(values)
    if vals.ndim != 2 or vals.shape[0] != q.size:
        raise ValueError(
            f"values must be (len(qs), len(Es)) = ({q.size}, {E.size}), got {vals.shape}"
        )

    # EVERY resolved peak, not the top `npk`. A tracker that can only choose
    # among the tallest three is still choosing by height, which is the bug.
    # The pool is what the energy grid can resolve at all -- an interior maximum
    # needs a sample either side -- so it is bounded by the map, not by a guess.
    pool = max(1, (E.size - 1) // 2)
    cand_E, cand_W = track_peaks(
        E, vals, n_peaks=pool, rel_prominence=rel_prominence, sort="weight"
    )

    n = q.size
    outE = np.full((n, npk), np.nan, dtype=np.float64)
    outW = np.full((n, npk), np.nan, dtype=np.float64)
    last_E = np.full(npk, np.nan, dtype=np.float64)
    last_q = np.full(npk, np.nan, dtype=np.float64)
    miss = np.zeros(npk, dtype=np.int64)
    dE_cell = float(np.min(np.diff(E))) if E.size > 1 else 0.0
    floor = _PEAK_GATE_FLOOR_CELLS * dE_cell
    speeds: List[float] = []

    for i in range(n):
        keep = np.isfinite(cand_E[i])
        ce = cand_E[i][keep]
        cw = cand_W[i][keep]
        if ce.size == 0:
            miss += 1
            last_E[miss > _PEAK_MAX_MISS] = np.nan
            continue

        live = [k for k in range(npk) if np.isfinite(last_E[k])]
        taken_c: set = set()
        taken_s: set = set()
        if live:
            cost = np.abs(ce[None, :] - last_E[live][:, None])
            # Greedy over the WHOLE matrix, cheapest pair first: the same rule
            # _follow_modes uses, and for the same reason -- a per-slot argmin
            # lets whichever slot is considered first take a peak that belonged
            # to another.
            flat = np.argsort(cost, axis=None, kind="stable")
            speed = float(np.median(speeds)) if len(speeds) >= 3 else float("nan")
            for idx in flat:
                r, c = int(idx // ce.size), int(idx % ce.size)
                k = live[r]
                if k in taken_s or c in taken_c:
                    continue
                dq = abs(q[i] - last_q[k])
                lim = floor
                if np.isfinite(speed) and dq > 0.0:
                    lim = max(lim, _PEAK_GATE_SPEED * speed * dq)
                if float(cost[r, c]) > lim:
                    # Sorted ascending, so every pair after this one is worse
                    # for THIS slot but may still be fine for another.
                    continue
                taken_s.add(k)
                taken_c.add(c)
                outE[i, k] = ce[c]
                outW[i, k] = cw[c]

        # Spares seed only slots that are genuinely free: never used, or empty
        # long enough that the branch they held has ended.
        spare = sorted((c for c in range(ce.size) if c not in taken_c),
                       key=lambda c: -float(cw[c]))
        free = [k for k in range(npk) if k not in taken_s and not np.isfinite(last_E[k])]
        for k, c in zip(free, spare):
            outE[i, k] = ce[c]
            outW[i, k] = cw[c]

        for k in range(npk):
            if np.isfinite(outE[i, k]):
                dq = abs(q[i] - last_q[k])
                if np.isfinite(last_E[k]) and dq > 0.0:
                    speeds.append(abs(float(outE[i, k]) - float(last_E[k])) / dq)
                last_E[k], last_q[k], miss[k] = outE[i, k], q[i], 0
            else:
                miss[k] += 1
                if miss[k] > _PEAK_MAX_MISS:
                    last_E[k] = np.nan

    # One ordering for the whole sweep. An all-NaN slot sorts last either way,
    # so empty slots stay at the end where a reader expects the padding.
    with np.errstate(invalid="ignore"):
        if sort == "weight":
            key = np.array([
                -np.nansum(outW[:, k]) if np.any(np.isfinite(outW[:, k])) else np.inf
                for k in range(npk)
            ])
        else:
            key = np.array([
                np.nanmean(outE[:, k]) if np.any(np.isfinite(outE[:, k])) else np.inf
                for k in range(npk)
            ])
    order = np.argsort(key, kind="stable")
    return outE[:, order], outW[:, order]



# --------------------------------------------------------------------------- #
# Non-reciprocity
# --------------------------------------------------------------------------- #
def symmetric_q_grid(
    q_max: float,
    n_pos: int,
    *,
    include_zero: bool = False,
    q_min: Optional[float] = None,
) -> NDArray[np.float64]:
    """A ``q`` grid that is BIT-EXACTLY antisymmetric, for :func:`pm_asymmetry`.

    ``np.linspace(-a, a, 2k+1)`` is NOT mirror symmetric in floating point --
    ``linspace(-0.3, 0.3, 13)`` has ``q + q[::-1]`` as large as 5.6e-17 -- and
    that is enough to break the exact ``+-q`` cancellation at ``beta = 0``: the
    conductivity and the Coulomb kernel then see two slightly different ``|q|``
    and the Level 1 asymmetry comes out at ~4e-14 instead of 0.  This builds the
    positive half once and mirrors it, so ``q == -q[::-1]`` holds exactly and the
    ``beta = 0`` asymmetry is identically zero.

    Parameters
    ----------
    q_max
        Largest ``|q|`` [1/nm]; must be positive.
    n_pos
        Number of strictly positive samples (``>= 1``).  The result has
        ``2 n_pos`` entries, or ``2 n_pos + 1`` with ``include_zero``.
    include_zero
        Insert ``q = 0``.  :func:`pm_asymmetry` accepts it and it contributes an
        exactly zero row, but :func:`ldos_map` rejects it, so the default leaves
        it out.
    q_min
        Smallest positive sample [eV^0 / nm^-1]; defaults to ``q_max / n_pos``,
        which makes the positive half a uniform grid
        ``q_max * (1, 2, ..., n_pos) / n_pos``.

    Returns
    -------
    ndarray of float64
        Strictly increasing, with ``q == -q[::-1]`` bit for bit.
    """
    qm = float(q_max)
    npos = int(n_pos)
    if not qm > 0.0:
        raise ValueError(f"q_max must be positive, got {q_max}")
    if npos < 1:
        raise ValueError(f"n_pos must be >= 1, got {n_pos}")
    if q_min is None:
        pos = qm * (np.arange(1, npos + 1, dtype=np.float64) / float(npos))
    else:
        lo = float(q_min)
        if not 0.0 < lo <= qm:
            raise ValueError(f"need 0 < q_min <= q_max, got ({q_min}, {q_max})")
        pos = np.linspace(lo, qm, npos, dtype=np.float64) if npos > 1 else np.array([qm])
    if np.any(np.diff(pos) <= 0.0) or not np.all(pos > 0.0):
        raise ValueError("the requested grid is not strictly positive and increasing")
    parts = [-pos[::-1]]
    if include_zero:
        parts.append(np.zeros(1, dtype=np.float64))
    parts.append(pos)
    return np.concatenate(parts)


def pm_asymmetry(
    qs: ArrayLike,
    values: ArrayLike,
    *,
    normalise: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``+-q`` asymmetry of any map whose FIRST axis is ``q``.

        ``A(q) = M(+q) - M(-q)``,
        ``A_norm(q) = 2 [M(+q) - M(-q)] / [M(+q) + M(-q)]``

    The physics payoff of the drift formalism.  It is EXACTLY zero at
    ``beta = 0`` -- not merely small -- for both Level 1 and Level 3: the
    undrifted conductivity depends on ``q`` only through ``alpha^2``, the Coulomb
    kernel and ``D_geom`` only through ``|q|`` and ``(qW)^2``, and an ``x``- or
    ``z``-oriented dipole source only through ``|q|``, so the two half planes are
    bit-for-bit identical.  (A ``y``-oriented dipole is odd in ``q`` through
    ``phi_ext``; see the note in ``sources.phi_ext_dipole``.)  It is clearly
    nonzero at ``beta = 0.3``.

    THE GRID MUST BE BIT-EXACTLY MIRROR SYMMETRIC for that cancellation to be
    exact, and ``np.linspace(-a, a, odd)`` is NOT:
    ``linspace(-0.3, 0.3, 13)`` leaves ``q + q[::-1]`` as large as 5.6e-17, and
    the resulting mismatch in ``|q|`` shows up as a ~4e-14 Level 1 asymmetry at
    ``beta = 0``.  The validation below uses a tolerance, because it is testing
    the caller's INTENT; use :func:`symmetric_q_grid` when the exact zero
    matters.

    Parameters
    ----------
    qs
        Wave vectors [1/nm], strictly increasing and antisymmetric about zero
        (``qs == -qs[::-1]``), so that every ``+q`` has its ``-q`` partner.  A
        ``q = 0`` sample is allowed and contributes an exactly zero row.  Checked
        to ``1e-12 * max|q|``; see the note above on exactness.
    values
        Map with ``q`` along axis 0 and any trailing shape, e.g. the
        ``(nq, nE)`` output of :func:`ldos_map` or ``Im`` of
        :func:`sheet_reflection`.
    normalise
        Divide by the mean of the two branches (times 2), giving the
        dimensionless relative asymmetry of paper 456's dissymmetry ratio.
        Entries where the two branches sum to zero become ``NaN``.

    Returns
    -------
    (q_pos, asym)
        ``q_pos`` the strictly positive wave vectors, ``asym`` the asymmetry
        with the same trailing shape as ``values``.

    Raises
    ------
    ValueError
        If ``qs`` is not strictly increasing or not antisymmetric about zero, or
        if its length does not match axis 0 of ``values``.
    """
    q = _f(qs).ravel()
    vals = np.asarray(values)
    if q.size != vals.shape[0]:
        raise ValueError(f"len(qs) {q.size} != values.shape[0] {vals.shape[0]}")
    if q.size < 2:
        raise ValueError("need at least 2 wave vectors")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("qs must be strictly increasing")
    scale = float(np.max(np.abs(q)))
    if not np.allclose(q, -q[::-1], rtol=0.0, atol=1e-12 * max(scale, 1.0)):
        raise ValueError(
            "qs must be antisymmetric about zero (qs == -qs[::-1]) so that every "
            "+q has a -q partner; got a grid that is not mirror symmetric"
        )
    # Every strictly increasing, mirror-symmetric grid of length >= 2 has a
    # positive entry, so no separate "nothing positive" guard is needed here.
    mask = q > 0.0
    plus = vals[mask]
    minus = vals[::-1][mask]
    asym = plus - minus
    if normalise:
        tot = plus + minus
        with np.errstate(divide="ignore", invalid="ignore"):
            asym = np.where(tot != 0.0, 2.0 * asym / np.where(tot != 0.0, tot, 1.0), np.nan)
    return q[mask], np.asarray(asym)
