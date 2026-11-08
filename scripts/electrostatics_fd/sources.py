"""External (driving) potentials for the quasistatic finite-difference ribbon solver.

Every excitation in this library enters the linear problem through one object only:
the external-potential vector ``phi_ext`` sampled on the ribbon nodes.  The three
sources below (plane wave, point dipole, swift electron) therefore share a single
job -- return the ``y``-Fourier component ``phi_ext_q`` evaluated at the physical
node coordinates -- and differ in nothing else.  Downstream modules
(``solver.RibbonSystem.solve_linear``, ``observables``) never branch on which
excitation produced the vector.

Conventions
-----------
* Units are the eV-nm-fs Gaussian set defined in :mod:`units`.  Lengths in nm,
  angular frequency in 1/fs, guided wave vector ``q`` in 1/nm.
* The ``y`` dependence is Fourier transformed with the ``exp(+i q y)`` kernel, so a
  source centred at ``y0`` carries ``exp(-i q y0)``.
* All three functions return ``complex128`` (the plane wave is real-valued but is
  returned complex anyway, so the assembled right-hand side keeps one dtype and
  ``numpy`` never silently casts the solve down to float).

Normalisation asymmetry across the three sources (by construction, not a bug)
-----------------------------------------------------------------------------
:func:`phi_ext_plane_wave` and :func:`phi_ext_dipole` are evaluated at a fixed
``omega`` and transformed in ``y`` only.  :func:`phi_ext_electron` is a DOUBLE
transform, over ``t`` and over ``y``, because the electron supplies its own
frequency content through the trajectory: ``e phi_q,omega`` therefore carries one
extra factor of time (``eV nm fs``, which is the correct dimension for that
transform pair, not an unpinned convention).  Absolute amplitudes from the three
sources are consequently not on a common footing, and any future code that mixes
them, or normalises one against another, must insert that factor explicitly.

Source equations are quoted per function.  Where the project physics spec supplied
a formula without a paper equation number, that is said explicitly rather than
invented.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import k0, k1

# numpy >= 2 renamed trapz -> trapezoid; pick whichever this interpreter has.
# Same shim as nonlinear.py and observables.py, and for the same reason: the
# browser runs this under Pyodide's numpy 2, where the old name warns and the
# warning surfaces to the user as a result caveat.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _substrate() -> Any:
    """The sibling ``substrate`` module, however this file was loaded.

    The reflected sources need ``default_Qmax`` and ``_eval_r`` from it, but a
    plain ``from . import substrate`` fails under the flat import style the
    per-module tests use (conftest puts the package DIRECTORY on sys.path, so
    there is no parent package). Same shape as ``nonlinear._import_sibling``.
    """
    package = __package__ or ""
    for dotted in ([f"{package}.substrate"] if package else []) + [
        "substrate",
        "electrostatics_fd.substrate",
    ]:
        try:
            return importlib.import_module(dotted)
        except ImportError:
            continue
    raise ImportError("cannot import the substrate module from sources.py")

try:  # normal case: the directory is imported as a package
    from .units import C_NM_FS, E2_EV_NM, omega_of
except ImportError:  # flat layout (no __init__.py yet): plain module import
    from units import C_NM_FS, E2_EV_NM, omega_of

__all__ = [
    "phi_ext_dipole_reflected",
    "phi_ext_electron_reflected",
    "phi_ext_plane_wave",
    "phi_ext_dipole",
    "phi_ext_electron",
    "e_ext_electron",
    "electron_trajectory_is_valid",
    "lorentz_gamma",
    "electron_decay_constant",
]

# A dipole closer than this to a node makes K1(Q s)/s diverge: reject instead of
# returning inf and poisoning the solve.
_ZERO_SEP_TOL_NM: float = 1.0e-12

# Trajectory/plane coincidence tolerance, fixed by the spec ("within 1e-9").
_PLANE_TOL_NM: float = 1.0e-9


# ---------------------------------------------------------------------------
# kinematic helpers (shared by the swift-electron source and its tests)
# ---------------------------------------------------------------------------
def lorentz_gamma(v_over_c: ArrayLike) -> float | NDArray[np.float64]:
    """Lorentz factor ``gammaL = 1 / sqrt(1 - (v/c)^2)``.

    Source: project physics spec, "Swift electron, aloof" block (definition of
    ``gammaL``; no paper equation number supplied).

    Parameters
    ----------
    v_over_c
        Electron speed in units of ``c``.  Must lie strictly inside ``(0, 1)``:
        ``v = 0`` makes the ``exp(i omega x / v)`` phase and the ``1/v`` prefactor
        singular, and ``|v/c| >= 1`` makes ``gammaL`` non-real.

    Returns
    -------
    float or ndarray
        ``gammaL``, scalar for scalar input.
    """
    beta = np.asarray(v_over_c, dtype=float)
    if np.any(beta <= 0.0) or np.any(beta >= 1.0):
        raise ValueError(
            f"v_over_c must lie strictly inside (0, 1); got {v_over_c!r}."
        )
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    return float(gamma) if gamma.ndim == 0 else gamma


def electron_decay_constant(
    q: ArrayLike, E_ev: ArrayLike, v_over_c: ArrayLike
) -> float | NDArray[np.float64]:
    """Transverse decay constant ``Lam = sqrt(q^2 + (omega / (v gammaL))^2)`` [1/nm].

    Source: project physics spec, "Swift electron, aloof" block (definition of
    ``Lam``; no paper equation number supplied).  This is the standard aloof-EELS
    evanescent decay rate: the ``y``-Fourier, ``x``-real-space quasistatic potential
    obeys ``(d^2/dz^2 - Lam^2) phi = -4 pi rho`` once ``k_x`` has been fixed to
    ``omega / v`` by the trajectory.

    Parameters
    ----------
    q
        Guided wave vector along the ribbon axis [1/nm].
    E_ev
        Photon (loss) energy [eV]; ``omega = E_ev / HBAR_EV_FS``.
    v_over_c
        Electron speed in units of ``c``.

    Returns
    -------
    float or ndarray
        ``Lam``, scalar for scalar input.
    """
    beta = np.asarray(v_over_c, dtype=float)
    gamma = np.asarray(lorentz_gamma(beta), dtype=float)
    v_nm_fs = beta * C_NM_FS
    omega = np.asarray(omega_of(E_ev), dtype=float)
    lam = np.sqrt(
        np.asarray(q, dtype=float) ** 2 + (omega / (v_nm_fs * gamma)) ** 2
    )
    return float(lam) if lam.ndim == 0 else lam


# ---------------------------------------------------------------------------
# 1. plane wave
# ---------------------------------------------------------------------------
def phi_ext_plane_wave(x_nm: ArrayLike, E0: complex = 1.0) -> NDArray[np.complex128]:
    """Normal-incidence plane wave polarised across the ribbons, ``q = 0`` only.

    ``phi_ext(x) = -E0 * x``

    Source: project physics spec, "Plane wave, normal incidence" block (long-
    wavelength/quasistatic limit; no paper equation number supplied).

    The potential of a uniform in-plane field ``E0 x_hat``.  Valid at ``q = 0``
    only: at finite ``q`` the incident field is no longer uniform along ``y`` and
    this expression is not its Fourier component.  ``q`` is not an argument of this
    function, so the caller is responsible for only using it in the ``q = 0`` sweep
    row -- flagged here because nothing in the signature can enforce it.

    Parameters
    ----------
    x_nm
        PHYSICAL node coordinates [nm], i.e. ``x_j + W_j * theta``, not the
        normalised ``theta``.  The ribbon offset therefore enters the result: a
        ribbon centred away from the origin sees a potential offset, which is the
        physically correct gauge for a multi-ribbon array.
    E0
        Field amplitude [eV/(e nm) in the Gaussian eV-nm convention].

    Returns
    -------
    ndarray of complex128
        Same shape as ``x_nm``.
    """
    x = np.asarray(x_nm, dtype=float)
    return np.asarray(-np.asarray(E0, dtype=np.complex128) * x, dtype=np.complex128)


# ---------------------------------------------------------------------------
# 2. point dipole
# ---------------------------------------------------------------------------
def phi_ext_dipole(
    q: float,
    x_nm: ArrayLike,
    z_nm: ArrayLike,
    r0_nm: Sequence[float],
    p: Sequence[complex],
    eps_a: complex = 1.0,
) -> NDArray[np.complex128]:
    """``y``-Fourier potential of a point dipole (paper 456 Eq. 20; ref. ``set_phiext``).

    With ``Qa = |q|``, ``dx = x - x0``, ``dz = z - z0``, ``s = sqrt(dx^2 + dz^2)``::

        phi_ext_q(x) = 2 * ( Qa*(px*dx + pz*dz)*K1(Qa*s)/s
                             - 1j*q*py*K0(Qa*s) ) * exp(-1j*q*y0) / eps_a

    Source: paper 456, Eq. (20), as transcribed in the project physics spec.

    Note on the formula as given (implemented verbatim, not "fixed"): the in-plane
    radial term uses ``Qa = |q|`` while the ``py`` term uses the SIGNED ``q``.  That
    asymmetry is deliberate in the source -- ``K0`` and ``K1`` take ``|q|`` because
    they are even in ``q``, whereas ``py`` couples through ``d/dy -> i q``, which is
    odd -- but it does mean ``phi_ext_q`` for a ``y``-oriented dipole flips sign with
    ``q``, so any caller that assumes ``phi(-q) = conj(phi(q))`` must verify that
    assumption per dipole orientation.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].  Must be non-zero: ``K0`` and ``K1`` diverge
        logarithmically / as ``1/x`` at zero argument, so the ``q = 0`` Fourier
        component of a point dipole does not exist in this representation.
    x_nm, z_nm
        Physical node coordinates [nm]; broadcast against each other.  ``z_nm`` is
        normally the scalar ribbon height ``z_j``.
    r0_nm
        Dipole position ``(x0, y0, z0)`` [nm].
    p
        Dipole moment ``(px, py, pz)``; may be complex (e.g. a circularly polarised
        emitter), which is why the result is complex even at ``y0 = 0``.
    eps_a
        Permittivity of the medium holding the dipole.

    Returns
    -------
    ndarray of complex128
        Broadcast shape of ``x_nm`` and ``z_nm``.

    Raises
    ------
    ValueError
        If ``q == 0``, if the dipole sits on a node (``s -> 0``), if ``r0_nm`` or
        ``p`` is not length 3, or if ``eps_a == 0``.
    """
    q_val = float(q)
    if q_val == 0.0:
        raise ValueError(
            "the dipole source requires q != 0: K0(|q| s) and K1(|q| s) diverge as "
            "q -> 0, so the q = 0 Fourier component of a point dipole is not "
            "defined in this representation. Use phi_ext_plane_wave for q = 0."
        )
    q_abs = abs(q_val)

    r0 = np.asarray(r0_nm, dtype=float)
    if r0.shape != (3,):
        raise ValueError(f"r0_nm must be (x0, y0, z0); got shape {r0.shape}.")
    p_vec = np.asarray(p, dtype=np.complex128)
    if p_vec.shape != (3,):
        raise ValueError(f"p must be (px, py, pz); got shape {p_vec.shape}.")

    eps = np.asarray(eps_a, dtype=np.complex128)
    if np.any(eps == 0.0):
        raise ValueError("eps_a must be non-zero.")

    dx, dz = np.broadcast_arrays(
        np.asarray(x_nm, dtype=float) - r0[0],
        np.asarray(z_nm, dtype=float) - r0[2],
    )
    s = np.sqrt(dx * dx + dz * dz)
    if np.any(s < _ZERO_SEP_TOL_NM):
        raise ValueError(
            f"the dipole at r0 = {tuple(r0)} nm sits on (or within "
            f"{_ZERO_SEP_TOL_NM:g} nm of) a ribbon node: K1(Qa*s)/s diverges as "
            "s -> 0. Offset the dipole from the node grid, or move it off the "
            "ribbon plane."
        )

    arg = q_abs * s
    radial = q_abs * (p_vec[0] * dx + p_vec[2] * dz) * k1(arg) / s
    axial = 1j * q_val * p_vec[1] * k0(arg)
    phi = 2.0 * (radial - axial) * np.exp(-1j * q_val * r0[1]) / eps
    return np.asarray(phi, dtype=np.complex128)


# ---------------------------------------------------------------------------
# 3. swift electron
# ---------------------------------------------------------------------------
def electron_trajectory_is_valid(
    b_nm: float,
    z_plane_nm: ArrayLike,
    layers: Sequence[Any] | None = None,
) -> tuple[bool, str]:
    """Check that a swift-electron trajectory is aloof (precondition, enforced).

    Source: project physics spec, "Swift electron, aloof" PRECONDITION block.

    Rejects two configurations:

    1. ``|b - z_plane| <= 1e-9`` nm -- the trajectory lies in the ribbon plane, so
       the ``exp(-Lam |z - b|)`` source is evaluated at its own kink and the
       quasistatic aloof potential no longer describes the coupling (a penetrating
       trajectory needs a different, non-aloof treatment).
    2. ``layers`` given and ``b < z_plane`` -- the electron would pass below the
       ribbons, i.e. inside the substrate stack, which the aloof source does not
       describe.

    With ``layers=None`` (free-standing ribbon) either side of the plane is allowed
    and only case 1 is rejected.  ``layers=[]`` is treated the same as ``None``:
    an empty stack is physically free-standing.

    Parameters
    ----------
    b_nm
        Impact parameter: height of the straight trajectory [nm].
    z_plane_nm
        Ribbon-plane height(s) [nm].  An array is accepted so a multi-ribbon system
        can be validated in one call; the first offending plane is named.
    layers
        Substrate stack (``substrate.Layer`` sequence) or ``None`` for free-standing.

    Returns
    -------
    (bool, str)
        ``(True, reason)`` if aloof; ``(False, reason)`` with a human-readable
        explanation otherwise.
    """
    b = float(b_nm)
    z = np.atleast_1d(np.asarray(z_plane_nm, dtype=float))

    intersects = np.abs(z - b) <= _PLANE_TOL_NM
    if np.any(intersects):
        z_bad = float(z[intersects][0])
        return False, (
            f"trajectory at b = {b:.6g} nm intersects the ribbon plane at "
            f"z = {z_bad:.6g} nm (|b - z| <= {_PLANE_TOL_NM:g} nm); the aloof "
            "quasistatic source is only defined off the plane."
        )

    if layers:
        below = b < z
        if np.any(below):
            z_bad = float(z[below][0])
            return False, (
                f"trajectory at b = {b:.6g} nm passes below the ribbon plane at "
                f"z = {z_bad:.6g} nm; with a layered substrate the electron would "
                "enter the stack, which the aloof source does not describe."
            )

    return True, (
        f"aloof trajectory: b = {b:.6g} nm, closest ribbon plane at "
        f"z = {float(z[np.argmin(np.abs(z - b))]):.6g} nm."
    )


def phi_ext_electron(
    q: ArrayLike,
    x_nm: ArrayLike,
    z_nm: ArrayLike,
    E_ev: ArrayLike,
    v_over_c: float,
    b_nm: float,
    layers: Sequence[Any] | None = None,
) -> NDArray[np.complex128]:
    """``y``-Fourier potential of an aloof swift electron crossing the ribbon.

    The electron moves along ``+x`` (across the ribbon) at constant height ``b``,
    parallel to the ribbon plane.  With ``omega = E_ev / HBAR_EV_FS``,
    ``v = v_over_c * C_NM_FS``, ``gammaL = 1/sqrt(1 - (v/c)^2)`` and
    ``Lam = sqrt(q^2 + (omega/(v gammaL))^2)``::

        phi_ext_q(x, z) = -(2 pi e^2 / (v Lam))
                          * exp(1j omega x / v) * exp(-Lam |z - b|)

    Source equations
    ----------------
    Exact ``(t, y)`` double Fourier transform of the uniform-motion
    Lienard-Wiechert scalar potential
    ``phi = -e gammaL / sqrt(gammaL^2 (x - v t)^2 + y^2 + (z - b)^2)``, using
    ``int dw exp(-i k w) / sqrt(w^2 + R^2) = 2 K0(|k| |R|)`` and
    ``int dy exp(-i q y) K0(kappa sqrt(y^2 + d^2)) = pi exp(-|d| Lam) / Lam``,
    ``Lam = sqrt(q^2 + kappa^2)``, ``kappa = omega / (gammaL v)``.  The intermediate
    (``t``-transformed, ``y``-real) form
    ``phi_omega = -(2 e / v) exp(i omega x / v) K0(omega R / (gammaL v))`` is the
    published aloof quasistatic potential of Garcia de Abajo,
    Rev. Mod. Phys. 82, 209 (2010), Sec. II, whose closed form likewise carries NO
    ``gammaL`` prefactor -- ``gammaL`` appears only inside the ``K0`` argument, i.e.
    only inside ``Lam``.  Geometry is paper 430's ``Gamma^(1)`` geometry (sheet at
    ``z = const``, width along ``x``, invariant along ``y`` with ``q`` conjugate to
    ``y``, electron at ``(v t, 0, b)`` along ``+x`` across the width); there
    ``g = Lam W``.

    The prefactor is settled, with ``gammaL^0``
    ------------------------------------------
    An earlier version of this docstring flagged the ``1/gammaL`` of the project
    spec as a CONCERN and implemented it anyway.  That concern is now resolved in
    favour of dropping it, and the code does.  The explicit ``gammaL`` of the
    boosted potential cancels exactly against the ``1/gammaL`` of the
    ``s -> gammaL s`` rescaling needed to reach the isotropic ``2 K0`` form (the
    remaining ``1/v`` is the transit-time Jacobian, which is physical and stays);
    the opposite integration ordering, where no ``1/gammaL`` Jacobian arises at all,
    gives the same answer.  A fully numerical ``(t, y)`` double quadrature that uses
    no analytic Fourier identity anywhere converges monotonically to this form, and
    a high-accuracy sweep over 54 configurations
    (``v/c`` in {0.1, 0.2, 0.5} x ``E`` in {0.5, 1.5} eV x ``q`` in
    {0.01, 0.05, 0.3} 1/nm x ``b`` in {0.5, 2, 10} nm) gives
    quadrature/formula = 1.000000000000 with spread 7.8e-16, while the spec's
    ``1/gammaL`` form is too small by exactly ``gammaL``.

    Paper 430's ``gammaL^2`` is NOT a competing potential prefactor: it is the
    prefactor of the longitudinal external FIELD, via the exact uniform-motion
    identity ``E_x = -gammaL^-2 d(phi)/dx`` (``A_x = (v/c) phi`` cancels ``v^2/c^2``
    of the gradient).  See :func:`e_ext_electron`.  The decisive argument is
    internal to that paper: its ``y``-weight ``q/g`` carries no ``gammaL`` at all and
    ``E_y = -d(phi^ext)/dy`` is exact (``A_y = 0``), so ``phi^ext`` can carry no
    ``gammaL`` prefactor -- otherwise the paper's weights would have to read
    ``omega/(v gammaL^3)`` and ``q/gammaL``.

    Note on the prefactor: the charge is ``-e``, and the ``E2_EV_NM`` factor carries
    ``e^2`` in the Gaussian eV-nm convention so that this quantity times a charge
    density gives an energy.  Dimensionally ``e^2/(v Lam)`` is ``eV nm fs``, which is
    the correct dimension for a double (``t`` and ``y``) transform; the only free
    convention is the ``2 pi``, and it is the right one for the transform pair this
    module declares (no ``1/2pi`` on either forward transform), reproduced by the
    quadrature along with everything else.  The real caveat is the cross-source one
    in the module header: this function is a ``(t, y)`` double transform while
    :func:`phi_ext_dipole` and :func:`phi_ext_plane_wave` are single ``y``
    transforms at fixed ``omega``, so absolute amplitudes are not comparable across
    the three sources.

    Remaining open item (recorded, not a defect of this function): ``solver.py``
    drives with a scalar ``phi_ext``, whose gradient carries ``gammaL^0`` on every
    component, whereas paper 430 needs ``gammaL^-2`` on the ``x`` component only.
    No scalar potential can supply both, so the scalar drive over-weights the
    across-the-ribbon channel by ``gammaL^2`` relative to the along-the-axis one
    (1.0% in amplitude / 2.0% in a rate at ``v/c = 0.1``; 33% / 78% at
    ``v/c = 0.5``).  Eigenvalues, eigenvectors and resonance positions come from the
    geometry operator alone and are untouched.  Fix at the rate level the way paper
    430 does -- weight the ``x``-overlap integral by ``gammaL^-2`` -- or drive with
    the vector field from :func:`e_ext_electron`.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].
    x_nm, z_nm
        Physical node coordinates [nm]; broadcast against each other.  ``z_nm`` is
        normally the scalar ribbon height ``z_j``.
    E_ev
        Loss energy [eV]; normally scalar (one frequency per solve), but broadcast
        is supported.
    v_over_c
        Electron speed in units of ``c``; must lie strictly inside ``(0, 1)``.
    b_nm
        Impact parameter (trajectory height) [nm].
    layers
        Optional substrate stack, forwarded to :func:`electron_trajectory_is_valid`.
        Keyword-defaulted, so the contract call signature is unchanged; pass the
        stack whenever one is present, or a trajectory under the substrate will be
        accepted as merely "free-standing, other side".

    Returns
    -------
    ndarray of complex128
        Broadcast shape of ``x_nm``, ``z_nm`` and ``E_ev``.

    Raises
    ------
    ValueError
        If the trajectory is not aloof, if ``v_over_c`` is outside ``(0, 1)``, or if
        ``Lam == 0`` (``q = 0`` and ``E_ev = 0``), where the prefactor diverges.
    """
    z = np.asarray(z_nm, dtype=float)
    ok, reason = electron_trajectory_is_valid(b_nm, z, layers=layers)
    if not ok:
        raise ValueError(f"invalid swift-electron trajectory: {reason}")

    beta = np.asarray(v_over_c, dtype=float)
    # lorentz_gamma is still called, inside electron_decay_constant: gammaL
    # survives in Lam and nowhere else. It is deliberately absent here.
    v_nm_fs = beta * C_NM_FS
    omega = np.asarray(omega_of(E_ev), dtype=float)
    lam = np.asarray(electron_decay_constant(q, E_ev, beta), dtype=float)
    if np.any(lam == 0.0):
        raise ValueError(
            "Lam = sqrt(q^2 + (omega/(v gammaL))^2) == 0 (q = 0 and E_ev = 0): the "
            "1/Lam prefactor diverges. The static, q = 0 limit of a swift-electron "
            "source is not normalisable here."
        )

    x = np.asarray(x_nm, dtype=float)
    prefactor = -(2.0 * np.pi * E2_EV_NM) / (v_nm_fs * lam)
    phase = np.exp(1j * omega * x / v_nm_fs)
    decay = np.exp(-lam * np.abs(z - float(b_nm)))
    return np.asarray(prefactor * phase * decay, dtype=np.complex128)


def e_ext_electron(
    q: ArrayLike,
    x_nm: ArrayLike,
    z_nm: ArrayLike,
    E_ev: ArrayLike,
    v_over_c: float,
    b_nm: float,
    layers: Sequence[Any] | None = None,
) -> tuple[
    NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]
]:
    """``(q, omega)``-domain external FIELD components of an aloof swift electron.

    For uniform motion along ``+x`` the three components are exact multiples of the
    scalar potential returned by :func:`phi_ext_electron`::

        E_x,q = -1j * (omega / v) * gammaL**-2   * phi_ext_q
        E_y,q = -1j * q                          * phi_ext_q
        E_z,q = +Lam * sign(z - b)               * phi_ext_q

    Source equations
    ----------------
    ``E = -grad(phi) - (1/c) dA/dt`` with the uniform-motion Lienard-Wiechert
    four-potential ``A = (v/c) phi xhat`` (so ``A_y = A_z = 0``) and the exact
    identity ``d(phi)/dt = -v d(phi)/dx``, which holds because ``phi`` depends on
    ``x`` and ``t`` only through ``x - v t``:

    * ``E_x = -d(phi)/dx - (v/c^2) d(phi)/dt = -(1 - v^2/c^2) d(phi)/dx
      = -gammaL^-2 d(phi)/dx``.  The ``x`` dependence of ``phi_ext_q`` is
      ``exp(i omega x / v)``, i.e. ``d/dx -> i omega / v``, giving the first line
      above.  This -- not any property of ``phi^ext`` -- is where paper 430's
      ``gammaL^2`` lives.
    * ``A_y = 0``, so ``E_y = -d(phi)/dy`` exactly, and in the ``y``-Fourier
      representation ``d/dy -> i q``.  No ``gammaL``, which is why paper 430's
      ``y``-weight ``q/g`` is ``gammaL``-free.
    * ``A_z = 0``, so ``E_z = -d(phi)/dz``, and ``d/dz exp(-Lam |z - b|)
      = -Lam sign(z - b) exp(-Lam |z - b|)``.

    Validation: paper 430's two in-plane ``Gamma^(1)`` weights, with lengths scaled
    by the ribbon width ``W`` so that ``g = Lam W`` and the paper's ``q`` slot is the
    DIMENSIONLESS ``k_parallel W``, are ``omega W / (v gammaL^2 g)`` and ``q/g``.
    Their ratio equals ``|E_x/E_y|`` here, verified to 12 digits for
    ``v/c`` in {0.1, 0.2, 0.5} and ``k_parallel`` in {0.02, 0.2} 1/nm.  Feeding the
    PHYSICAL ``k_parallel`` into the paper's dimensionless ``q`` slot instead
    misstates that ratio by a flat factor ``W``; both forms are pinned in
    ``test_sources.py``.

    The ``gammaL^-2`` is a retardation/magnetic effect: it comes entirely from
    ``A_x``.  A strictly non-retarded treatment (``A = 0``) would give
    ``E_x = -d(phi)/dx`` with ``gammaL^0``.  Paper 430, and this function, use the
    exact relativistic external field with a quasistatic sheet response -- the
    standard Garcia de Abajo aloof-EELS construction (Rev. Mod. Phys. 82, 209
    (2010), Sec. II), a legitimate hybrid.  ``phi^ext`` itself is exact either way;
    only its gradient differs.

    Parameters
    ----------
    q
        Guided wave vector along the ribbon axis [1/nm], PHYSICAL (not scaled by
        ``W``).  Enters ``E_y`` signed, so ``E_y`` is odd in ``q``.
    x_nm, z_nm
        Physical node coordinates [nm]; broadcast against each other.  ``z_nm`` is
        normally the scalar ribbon height ``z_j``.
    E_ev
        Loss energy [eV]; normally scalar, but broadcast is supported.
    v_over_c
        Electron speed in units of ``c``; must lie strictly inside ``(0, 1)``.
    b_nm
        Impact parameter (trajectory height) [nm].
    layers
        Optional substrate stack, forwarded to :func:`electron_trajectory_is_valid`
        through :func:`phi_ext_electron`.

    Returns
    -------
    (ndarray, ndarray, ndarray) of complex128
        ``(E_x, E_y, E_z)``, each of the broadcast shape of ``x_nm``, ``z_nm``
        and ``E_ev``.

    Notes
    -----
    ``sign(z - b)`` makes ``E_z`` DISCONTINUOUS across the trajectory plane: it
    flips sign at ``z = b`` while ``|E_z|`` is continuous, which is the usual kink
    of an evanescent half-space solution at the source plane (the normal field of a
    sheet of charge).  The aloof precondition enforced by
    :func:`electron_trajectory_is_valid` guarantees ``z != b``, so the kink is never
    evaluated; ``E_x`` and ``E_y`` are continuous there.

    Raises
    ------
    ValueError
        Whatever :func:`phi_ext_electron` raises: non-aloof trajectory,
        ``v_over_c`` outside ``(0, 1)``, or ``Lam == 0``.
    """
    phi = phi_ext_electron(q, x_nm, z_nm, E_ev, v_over_c, b_nm, layers=layers)

    beta = np.asarray(v_over_c, dtype=float)
    gamma = np.asarray(lorentz_gamma(beta), dtype=float)
    v_nm_fs = beta * C_NM_FS
    omega = np.asarray(omega_of(E_ev), dtype=float)
    lam = np.asarray(electron_decay_constant(q, E_ev, beta), dtype=float)
    q_arr = np.asarray(q, dtype=float)
    z = np.asarray(z_nm, dtype=float)

    e_x = -1j * (omega / v_nm_fs) / (gamma * gamma) * phi
    e_y = -1j * q_arr * phi
    e_z = lam * np.sign(z - float(b_nm)) * phi

    return (
        np.asarray(e_x, dtype=np.complex128),
        np.asarray(e_y, dtype=np.complex128),
        np.asarray(e_z, dtype=np.complex128),
    )


# ======================================================================================
# Reflected ("image") source potentials above a layered substrate
# ======================================================================================
#
# With a stack present the source's own field reaches the sheet twice: directly,
# and after reflection off the stack.  Screening the RESPONSE through
# ``substrate.build_Vbar`` while driving with the free-space source is wrong by
# O(r) in the drive, which is the same order as the screening itself.
#
# THE CONSTRUCTION.  Write the direct potential in the 2D Fourier plane as an
# amplitude times its own vertical decay.  The identity the whole module rests on
# is the Fourier representation of the direct kernel,
#
#     2 K0(|q| sqrt(dx^2 + dz^2)) = integral dQx exp(i Qx dx) exp(-Q |dz|) / Q,
#     Q = sqrt(Qx^2 + q^2),
#
# so the direct monopole kernel has amplitude 1/Q and vertical factor
# exp(-Q|z - z_src|).  The image term is the SAME amplitude with the separation
# folded about the interface and weighted by the stack's reflection coefficient:
#
#     G_refl(dx; z, z_src) = 2 integral_0^inf dQx [r(Q)/Q] exp(-Q(z + z_src)) cos(Qx dx)
#
# which is exactly the integrand of ``substrate.build_Vbar``, as it must be --
# the same reflected kernel acts on the induced charge and on the source.
#
# It has to be written on the AMPLITUDE, not on the direct potential:
# r * exp(-Q(z+z_src)) * phi_direct would carry the vertical decay twice.


def phi_ext_dipole_reflected(
    q: float,
    x_nm: ArrayLike,
    z_nm: ArrayLike,
    r0_nm: Sequence[float],
    p: Sequence[complex],
    r_of_Q: Callable[[np.ndarray], ArrayLike],
    eps_a: complex = 1.0,
    Qmax: Optional[float] = None,
    NQ: int = 4000,
) -> NDArray[np.complex128]:
    """Reflected part of a point dipole's potential above a layered stack.

    The direct potential is ``(p . grad_r0)`` acting on the monopole kernel
    ``2 K0(|q| s) exp(-i q y0) / eps_a`` (see :func:`phi_ext_dipole`), so the
    reflected part is the same derivative acting on ``G_refl`` above.  Taking the
    three derivatives under the integral gives three quadratures on one grid::

        d/dx0 : + 2 int dQx (r/Q) exp(-Q(z+z0)) * Qx * sin(Qx*dx)
        d/dz0 : - 2 int dQx  r    exp(-Q(z+z0)) *      cos(Qx*dx)
        d/dy0 : the -i q factor on exp(-i q y0)

    with ``dx = x - x0``.  The ``d/dz0`` sign is the one worth checking: the
    image exponential falls with ``z + z0``, so differentiating it brings down
    ``-Q`` where the direct term's ``exp(-Q|z - z0|)`` brings down
    ``+Q sign(z - z0)``.  That single sign is what turns the in-plane moment
    over in the image and leaves the normal one alone.

    PERFECT-MIRROR LIMIT, which is the test that pins all three signs at once.
    For ``r = -1`` this reduces exactly to the direct potential of an image
    dipole ``(-px, -py, +pz)`` at ``z0 -> -z0`` -- the textbook grounded-plane
    image -- and ``tests/test_sources.py`` asserts it to machine precision.

    Parameters
    ----------
    q
        Guided wave vector [1/nm], non-zero: at ``q = 0`` the reflected integral
        is logarithmically divergent, exactly as in ``substrate.build_Vbar``.
    x_nm, z_nm
        Observation points [nm], broadcast against each other; ``z_nm`` is
        normally the scalar ribbon height, measured from the first interface.
    r0_nm
        Dipole position ``(x0, y0, z0)`` [nm], with ``z0 > 0`` above the stack.
    p
        Dipole moment ``(px, py, pz)``, possibly complex.
    r_of_Q
        ``r(Q) -> complex`` for the stack, e.g.
        ``lambda Q: substrate.reflection_coefficient(Q, E_ev, layers, eps_a)``.
        Injected rather than built here, so this function knows nothing about
        how the stack is described -- the same seam ``build_Vbar`` uses.
    eps_a
        Permittivity of the medium holding the dipole.
    Qmax, NQ
        Quadrature controls, defaulting to ``substrate.default_Qmax`` on the
        image separation and a 4000-point uniform trapezoid rule.

    Returns
    -------
    ndarray of complex128
        Broadcast shape of ``x_nm`` and ``z_nm``.

    Raises
    ------
    ValueError
        If ``q == 0``, if ``z + z0 < 0`` (the image exponential would grow), or
        if ``NQ < 2``.
    """
    _sub = _substrate()
    q = float(q)
    if q == 0.0:
        raise ValueError(
            "phi_ext_dipole_reflected: q = 0 makes the reflected integral "
            "logarithmically divergent (integrand ~ r(0)/Qx as Qx -> 0). "
            "Use a small finite q."
        )
    NQ = int(NQ)
    if NQ < 2:
        raise ValueError(f"NQ must be at least 2, got {NQ}")

    x = np.asarray(x_nm, dtype=np.float64)
    z = np.asarray(z_nm, dtype=np.float64)
    x, z = np.broadcast_arrays(x, z)

    x0, y0, z0 = (float(v) for v in r0_nm)
    px, py, pz = (complex(v) for v in p)

    z_sum = z + z0
    if np.any(z_sum < 0.0):
        raise ValueError(
            "phi_ext_dipole_reflected: z + z0 < 0 somewhere; the image "
            "exponential exp(-Q(z+z0)) grows and the integral diverges. "
            "Both the dipole and the sheet must sit above the interface."
        )

    dx = x - x0
    if Qmax is None:
        Qmax = _sub.default_Qmax(q, float(np.min(z_sum)), float(np.abs(dx).max()))
    Qmax = float(Qmax)
    if Qmax <= 0.0:
        raise ValueError(f"Qmax must be positive, got {Qmax}")

    Qx = np.linspace(0.0, Qmax, NQ)
    Q = np.sqrt(Qx * Qx + q * q)
    r = np.asarray(_sub._eval_r(r_of_Q, Q), dtype=np.complex128)

    # (points, NQ) integrand pieces. The grids a page uses are small (a few
    # hundred nodes), so the dense form is clearer than blocking it.
    env = r[None, :] * np.exp(-Q[None, :] * z_sum.reshape(-1, 1))   # r e^{-Q(z+z0)}
    arg = Qx[None, :] * dx.reshape(-1, 1)

    def two_int(f: np.ndarray) -> np.ndarray:
        """The ``2 * integral_0^inf dQx`` the three derivatives share."""
        return 2.0 * _TRAPZ(f, Qx, axis=1)

    d_dx0 = two_int(env / Q[None, :] * Qx[None, :] * np.sin(arg))
    d_dz0 = -two_int(env * np.cos(arg))
    g_refl = two_int(env / Q[None, :] * np.cos(arg))

    phi = (px * d_dx0 + pz * d_dz0 - 1j * q * py * g_refl) * np.exp(-1j * q * y0) / eps_a
    return np.asarray(phi.reshape(x.shape), dtype=np.complex128)


def phi_ext_electron_reflected(
    q: ArrayLike,
    x_nm: ArrayLike,
    z_nm: ArrayLike,
    E_ev: ArrayLike,
    v_over_c: float,
    b_nm: float,
    r_of_Q: Callable[[np.ndarray], ArrayLike],
    layers: "Sequence[Any] | None" = None,
) -> NDArray[np.complex128]:
    """Reflected part of a swift electron's potential above a layered stack.

    No quadrature is needed here, and that is a property of the source rather
    than a shortcut: the aloof electron's potential carries the single in-plane
    Fourier component ``Qx = omega/v`` (the ``exp(i omega x / v)`` phase), so the
    stack sees one momentum::

        Q    = sqrt((omega/v)^2 + q^2)
        phi_refl(x, z) = r(Q, omega) * phi_dir(x, z = 0) * exp(-Q z)

    i.e. the incident potential's boundary value at the interface, reflected and
    propagated back up quasistatically.

    THE ONE APPROXIMATION, stated rather than buried.  ``phi_dir`` decays as
    ``exp(-Lam|z - b|)`` with ``Lam = sqrt(q^2 + (omega/(gammaL v))^2)``, which
    carries the Lorentz factor, while the reflected term propagates with the
    electrostatic ``Q``, which does not.  ``Lam`` is the vertical decay of a
    RETARDED field; ``Q`` is an in-plane momentum, and ``r(Q, omega)`` is a
    property of the stack at that momentum, so each is evaluated with the right
    quantity.  The two coincide as ``gammaL -> 1``, so the construction is exact
    in the non-retarded limit.

    HOW BIG IS IT, measured rather than estimated.  Against a perfect mirror,
    where the answer must be the image charge at ``-b``, the relative error is
    ``|expm1(-(Q - Lam) z)|`` -- not approximately but to a ratio of 1.000 at
    every speed tested, so the ENTIRE discrepancy is this one term and nothing
    else hides inside it.  At ``q = 0.05 /nm``, ``E = 0.6 eV``, ``z = 2 nm``
    that is 1.8e-04 at ``v = 0.5 c``, falling to 1.5e-05 at ``v = 0.005 c``.  It
    is far smaller than ``gammaL - 1`` (0.15 at ``0.5 c``) because ``gammaL``
    enters ``Lam`` only inside the square root, alongside ``q^2``.
    ``tests/test_sources.py`` pins both the mirror limit and that scaling.

    Note that the comparison is degenerate at ``z = 0``: there the source and
    its image are equidistant and both sides collapse to the same expression,
    so a mirror test evaluated on the interface passes identically and proves
    nothing.  Test it at a finite height.

    Parameters mirror :func:`phi_ext_electron`, plus ``r_of_Q`` as in
    :func:`phi_ext_dipole_reflected`.
    """
    _sub = _substrate()
    z = np.asarray(z_nm, dtype=np.float64)
    ok, reason = electron_trajectory_is_valid(b_nm, z, layers=layers)
    if not ok:
        raise ValueError(f"invalid swift-electron trajectory: {reason}")

    beta = np.asarray(v_over_c, dtype=np.float64)
    v_nm_fs = beta * C_NM_FS
    omega = np.asarray(omega_of(E_ev), dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)

    # The in-plane momentum the stack actually sees. No gammaL: this is a
    # momentum, not the decay constant of a retarded field.
    Qx = omega / v_nm_fs
    Q = np.sqrt(Qx * Qx + qa * qa)
    if np.any(Q == 0.0):
        raise ValueError(
            "phi_ext_electron_reflected: Q = 0 (q = 0 and E_ev = 0); the "
            "reflected potential is not defined there."
        )

    # The incident boundary value at the interface, from the direct source
    # itself rather than a second copy of its formula.
    phi_at_interface = phi_ext_electron(
        qa, x_nm, np.zeros_like(z), E_ev, v_over_c, b_nm, layers=None
    )
    r = np.asarray(_sub._eval_r(r_of_Q, np.atleast_1d(Q)), dtype=np.complex128)
    r = r.reshape(np.shape(Q)) if np.shape(Q) else r.reshape(())

    return np.asarray(
        phi_at_interface * r * np.exp(-Q * np.abs(z)), dtype=np.complex128
    )
