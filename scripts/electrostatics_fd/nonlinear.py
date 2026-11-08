"""Nonlinear (second- and third-harmonic) quasistatic response of graphene nanoribbons.

Reference: paper 406, "nonlinear orders" block, and its q = 0 polarizability
block as implemented in the reference code.

Perturbation expansion
----------------------
Quantities carry a pair ``(n, s)``: ``n`` is the order in the external field and ``s``
the harmonic index.  At order ``(n, s)`` the nonlinear current acts as a *source* and
the system is solved for the CHARGE DENSITY, with the two operators applied in the
OPPOSITE order to the linear problem (paper 406, nonlinear-order block)::

    N[block j, block jp] = (W_jp / W_j) * D_j(s) @ V(j, jp, s*q)
    rho^(n,s)            = solve(I - N, rho_NL^(n,s))

``D_j(s)`` is built from ribbon ``j``'s own eta evaluated at harmonic ``s``,

    eta_j^(s) = 1j * sigma_j(s*omega) / (s*omega*W_j*eps_eff),

obtained here by calling ``system.eta_pair(j, s*q, s*E_ev)``; ``V`` is evaluated at
wave vector ``s*q``.

.. note::
   The eta in ``N`` belongs to the OBSERVATION ribbon ``j``, not the source ribbon
   ``jp``.  This asymmetry with respect to the linear problem is present in the
   reference implementation and is deliberate -- see the module contract.

Validity
--------
The source assemblies below are the **one-dimensional q = 0 collapse**: only
derivatives along x appear.  The ``q`` argument is threaded through to ``V`` and ``D``
(so the block solve is correct at finite q), but the source terms themselves drop the
``i*q*phi`` y-derivative contributions.  Results at ``q != 0`` are therefore not
complete; this mirrors the reference implementation, which is written at q = 0.

Dependency injection
--------------------
Sibling modules (``units``, ``operators``, ``conductivity``) are imported lazily, so
this module can be imported before they exist.  A ``system`` object may additionally
override any of the following hooks (all optional, all namespaced so they cannot
collide with a solver's own internals):

``system.operators``          module-like, supplying ``build_V``/``build_D``/``build_D_geom``
``system.conductivity``       module-like, supplying the nonlinear sigma coefficients
``system.nonlinear_grid(j)``  -> ``(theta, h, L)`` for ribbon ``j``
``system.nonlinear_V_block(j, jp, q, E_ev=None)`` -> ``(L_j, L_jp)`` array
``system.nonlinear_D(j, q, E_ev, s)``    -> ``(L_j, L_j)`` array
``system.nonlinear_D_geom(j, q)``        -> ``(L_j, L_j)`` array

``solver.RibbonSystem`` implements ``nonlinear_grid`` and ``nonlinear_V_block``,
so a real system hands over its own node arrays rather than having them guessed,
and a layered system screens its harmonic blocks.  ``nonlinear_V_block`` is the
only way to get substrate screening (the ``substrate.build_Vbar`` kernel) into
the nonlinear blocks; without a usable hook the bare ``operators.build_V`` kernel
is used and a ``RuntimeWarning`` is raised when the system carries layers.

PER-ORDER FREQUENCY.  The hook is called with the wave vector AND photon energy
OF THE BLOCK -- both already multiplied by that block's harmonic index, never the
fundamental.  ``r(Q, omega)`` is frequency dependent, so the ``s``-th harmonic
block must be screened at ``s*omega``: the second-harmonic blocks at ``2 omega``,
the third-harmonic blocks at ``3 omega``, and the cascade's second-harmonic
self-potential at ``2 omega`` even though it is assembled inside a third-order
solve.  Screening a harmonic at the fundamental is a silent error -- the blocks
keep their shape and the solve still converges -- so every call site below names
the harmonic index it scales by, and ``tests/test_nonlinear.py`` pins the
energies the hook actually receives.  The nonlinear SOURCES
(:func:`shg_source_rho`, :func:`thg_source_rho`) build no Coulomb block at all and
so have no frequency to get wrong; the screening enters only through the block
solve and the induced self-potential.

A hook that accepts only ``(j, jp, q)`` cannot be told which harmonic it is
building and is therefore treated as unusable when the system carries layers: it
is still called (so a layer-free system keeps working), but the same
``RuntimeWarning`` is raised, because screening every harmonic at one unspecified
frequency is exactly the silent error above.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "shg_source_rho",
    "thg_source_rho",
    "solve_shg",
    "solve_thg",
    "polarizability_from_rho",
    "HBAR_EV_FS_FALLBACK",
    "CASCADE_A_REFERENCE",
    "CASCADE_A_CONDUCTIVITY",
]

_HERE = Path(__file__).resolve().parent

#: Fallback for ``units.HBAR_EV_FS``, used only if ``units`` cannot be imported.
#: Must stay identical to ``units.HBAR_EV_FS``; ``test_nonlinear.py`` asserts this.
HBAR_EV_FS_FALLBACK: float = 0.6582119569

# numpy >= 2 renamed trapz -> trapezoid; pick whichever this interpreter has.
_TRAPZ: Callable[..., Any] = getattr(np, "trapezoid", None) or np.trapz

ArrayLike = Union[np.ndarray, Sequence[complex]]
PhiLike = Union[Mapping[str, Any], Sequence[ArrayLike], np.ndarray]


# --------------------------------------------------------------------------------------
# lazy sibling imports
# --------------------------------------------------------------------------------------
def _import_sibling(name: str) -> Any:
    """Import a sibling module of this package, however this file was loaded.

    Tries the enclosing package first, then a bare top-level name (how the test
    suite puts the package directory on ``sys.path``), then ``electrostatics_fd.<name>``,
    and finally re-tries after prepending this file's own directory to ``sys.path``.
    """
    candidates: List[str] = []
    package = __package__ or ""
    if package:
        candidates.append(f"{package}.{name}")
    candidates.append(name)
    candidates.append(f"electrostatics_fd.{name}")

    last: Optional[BaseException] = None
    for dotted in candidates:
        try:
            return importlib.import_module(dotted)
        except ImportError as exc:  # module genuinely absent, or absent sub-dependency
            last = exc

    here = str(_HERE)
    if here not in sys.path:
        sys.path.insert(0, here)
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last = exc

    raise ImportError(
        f"nonlinear.py could not import sibling module {name!r}; tried {candidates!r}"
    ) from last


def _operators(system: Any) -> Any:
    """Return the operator provider: ``system.operators`` if set, else ``operators``."""
    injected = getattr(system, "operators", None)
    if injected is not None:
        return injected
    return _import_sibling("operators")


def _conductivity(system: Any) -> Any:
    """Return the conductivity provider: ``system.conductivity`` if set, else the module."""
    injected = getattr(system, "conductivity", None)
    if injected is not None:
        return injected
    return _import_sibling("conductivity")


def _omega(E_ev: float) -> float:
    """Angular frequency ``omega = E / hbar`` in 1/fs (units block of the contract)."""
    try:
        units = _import_sibling("units")
    except ImportError:
        return float(E_ev) / HBAR_EV_FS_FALLBACK
    return float(units.omega_of(float(E_ev)))


def _hbar() -> float:
    """``hbar`` in eV fs, from ``units`` when available."""
    try:
        units = _import_sibling("units")
    except ImportError:
        return HBAR_EV_FS_FALLBACK
    return float(units.HBAR_EV_FS)


#: ``cascade_A_form`` values; see :func:`_cascade_A`.
CASCADE_A_REFERENCE = "reference"
CASCADE_A_CONDUCTIVITY = "conductivity"


def _cascade_A(
    cond: Any,
    E_ev: float,
    gamma_ev: float,
    s1: int,
    s2: int,
    sign: float,
    form: str = CASCADE_A_REFERENCE,
) -> complex:
    """Cascaded second-order coefficient of the 1-D q = 0 source, harmonics ``(s1, s2)``.

    RESOLVED.  There is no discrepancy here and nothing upstream to fix: the two forms
    below are a SCALAR and a TENSOR COMPONENT, and a 1-D q = 0 source needs the scalar.

    The paper-406 reference implementation uses, with
    ``D_sum = D_(s1+s2)``::

        S_s1_s2 = S2 * D_sum * ( D_s2**2 + 2*D_sum*(D_s1 + D_s2) )

    and ``-doping_sign * S_s1_s2`` -- what ``CASCADE_A_REFERENCE`` returns -- is
    exactly the FULL Eq. (15) contraction summed over its four tensor structures::

        -doping_sign * S_s1_s2 == A + B + C + D   of sigma_cascade_abcd(E, g, s1, s2)

    (verified to <= 6e-16 over a 24-point (E, gamma, doping-sign, (s1,s2)) grid; pinned
    by ``test_cascade_A_is_the_full_abcd_contraction``).  The coefficient on each
    structure is 1 because of what the structures ARE in one dimension -- Eq. (15c)'s
    ``C`` multiplies ``sum_j E_j grad(E_j)``, which collapses to a single
    ``E_s1 d_x E_s2``, and Eq. (15d)'s ``D`` multiplies ``grad(E_s1 . E_s2)``, whose
    two 1-D terms are supplied by the two ORDERINGS this module sums over in
    :func:`thg_source_rho` rather than by a factor of 2 on one of them.  The degenerate
    case fixes that bookkeeping uniquely: ``A + B + C + D`` at ``s1 = s2 = 1`` equals
    ``sigma_shg_1d_q0`` (pinned by ``test_cascade_A_degenerate_is_the_shg_collapse``),
    which an extra factor on ``D`` would break.

    ``conductivity.sigma_cascade_abcd(...)[0]`` alone is Eq. (15a)'s ``A``, i.e. one
    tensor structure of four; ``sigma_cascade_abcd(E, g, 1, 1)[0] == sigma_shg_abc[0]``
    identifies it as the mixed-harmonic generalisation of the tensor structure ``A`` of
    Eq. (12a).  Inserting that single component into a scalar 1-D source is the error
    that an earlier reading of the contract made, so ``form=CASCADE_A_REFERENCE`` is
    the default on physical grounds, not for want of a decision.
    ``form=CASCADE_A_CONDUCTIVITY`` selects the bare ``A`` anyway for comparison, and
    ``test_cascade_A_discrepancy_is_the_D_s2_squared_term`` keeps the algebraic
    relation between the two pinned.

    The ``-doping_sign`` prefactor is taken from ``conductivity`` in BOTH forms.  The
    bare reference expression carries no doping sign, but the cascade source is already
    linear in ``rho^(2,2)``, which carries one factor of it, and the contract requires
    ``alpha^(3,3)`` to be UNCHANGED under a doping flip.  An ``A`` that is even in the
    doping sign would make ``alpha^(3,3)`` flip, so the MATLAB reference must fold the
    sign in elsewhere.
    """
    if form == CASCADE_A_CONDUCTIVITY:
        return cond.sigma_cascade_abcd(
            float(E_ev), float(gamma_ev), s1, s2, float(sign)
        )[0]
    if form != CASCADE_A_REFERENCE:
        raise ValueError(
            f"cascade_A_form must be {CASCADE_A_REFERENCE!r} or "
            f"{CASCADE_A_CONDUCTIVITY!r}, got {form!r}"
        )

    S2 = getattr(cond, "_S2", None)
    D_of = getattr(cond, "_D", None)
    if S2 is None or not callable(D_of):
        warnings.warn(
            "nonlinear.py cannot build the reference cascade A: the conductivity "
            "provider exposes no _S2/_D, so sigma_cascade_abcd is used instead. That "
            "returns Eq. (15a)'s tensor component A alone, not the full A+B+C+D "
            "contraction a 1-D q = 0 source needs.",
            RuntimeWarning,
            stacklevel=3,
        )
        return cond.sigma_cascade_abcd(
            float(E_ev), float(gamma_ev), s1, s2, float(sign)
        )[0]

    omega = _omega(E_ev)
    gam_fs = float(gamma_ev) / _hbar()
    d_s1 = D_of(float(s1), omega, gam_fs)
    d_s2 = D_of(float(s2), omega, gam_fs)
    d_sum = D_of(float(s1 + s2), omega, gam_fs)
    return complex(
        -float(sign) * S2 * d_sum * (d_s2**2 + 2.0 * d_sum * (d_s1 + d_s2))
    )


# --------------------------------------------------------------------------------------
# grid / operator plumbing
# --------------------------------------------------------------------------------------
def _theta_from_L(L: int) -> Tuple[np.ndarray, float]:
    """Uniform normalised nodes on ``[-1/2, 1/2]`` with spacing ``h = 1/(L-1)``."""
    if L < 3:
        raise ValueError(f"ribbon needs at least 3 nodes for the FD operators, got L={L}")
    return np.linspace(-0.5, 0.5, L), 1.0 / (L - 1)


def _looks_like_theta(arr: Any) -> bool:
    """True if ``arr`` is a plausible normalised node array on [-1/2, 1/2]."""
    try:
        a = np.asarray(arr, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    return (
        a.ndim == 1
        and a.size >= 3
        and abs(a[0] + 0.5) < 1e-9
        and abs(a[-1] - 0.5) < 1e-9
        and bool(np.all(np.diff(a) > 0.0))
    )


def _grid(system: Any, j: int) -> Tuple[np.ndarray, float, int]:
    """Normalised grid ``(theta, h, L)`` of ribbon ``j``.

    Prefers an explicit ``system.nonlinear_grid(j)`` hook, then a ``system.thetas``
    sequence if it validates as normalised nodes, and otherwise reconstructs the grid
    from the geometry block of the contract: ``L = ribbon.L`` when given, else
    ``round(W/dx_nm) + 1``, with ``h = 1/(L-1)``.
    """
    hook = getattr(system, "nonlinear_grid", None)
    if callable(hook):
        theta, h, L = hook(j)
        return np.asarray(theta, dtype=float).ravel(), float(h), int(L)

    for attr in ("thetas", "theta_nodes", "theta"):
        seq = getattr(system, attr, None)
        if seq is None or callable(seq):
            continue
        try:
            candidate = seq[j]
        except (TypeError, IndexError, KeyError):
            continue
        if _looks_like_theta(candidate):
            theta = np.asarray(candidate, dtype=float).ravel()
            return theta, 1.0 / (theta.size - 1), int(theta.size)

    ribbon = system.ribbons[j]
    L = getattr(ribbon, "L", None)
    if L is None:
        dx = float(getattr(system, "dx_nm", 0.1))
        L = int(round(float(ribbon.W_nm) / dx)) + 1
    L = max(3, int(L))
    theta, h = _theta_from_L(L)
    return theta, h, L


def _hook_takes_energy(hook: Any) -> bool:
    """True if a ``nonlinear_V_block`` hook accepts a fourth (energy) argument.

    Keeps the older three-argument hook signature working while letting a
    substrate-aware system receive the harmonic energy it needs.
    """
    try:
        params = inspect.signature(hook).parameters
    except (TypeError, ValueError):  # builtins / C callables
        return False
    if any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    ):
        return True
    positional = [
        p
        for p in params.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 4


def _hook_accepts_energy(system: Any) -> bool:
    """True if ``system`` can be asked for a Coulomb block at a GIVEN photon energy.

    The system-level form of :func:`_hook_takes_energy`: it answers "will this system's
    nonlinear blocks be screened at their own harmonic frequency?", which is the
    contract between this module and ``solver.RibbonSystem`` and is pinned from the
    solver side by
    ``tests/test_integration.py::test_nonlinear_V_block_signature_is_the_one_nonlinear_py_probes_for``.

    False when the system supplies no ``nonlinear_V_block`` hook at all, or supplies one
    that takes only ``(j, jp, q)``: in both cases a layered system cannot be screened
    per order and :func:`_V_block` raises the ``RuntimeWarning``.
    """
    hook = getattr(system, "nonlinear_V_block", None)
    return bool(callable(hook) and _hook_takes_energy(hook))


def _V_block(
    system: Any, j: int, jp: int, q: float, E_ev: Optional[float] = None
) -> np.ndarray:
    """Coulomb block ``V(j, jp, q)``, shape ``(L_j, L_jp)`` (operators.build_V contract).

    ``q`` and ``E_ev`` are the wave vector and photon energy OF THIS BLOCK, i.e.
    already multiplied by that block's harmonic index ``s`` by the caller -- see the
    PER-ORDER FREQUENCY paragraph of the module docstring.  The energy is needed only
    by a substrate-aware ``system.nonlinear_V_block`` hook, whose reflection
    coefficient is frequency dependent.

    A hook that takes only ``(j, jp, q)`` is still called, so a layer-free system that
    supplies the older three-argument signature keeps working unchanged.  But such a
    hook cannot be told which harmonic it is building, so with layers present it is
    reported as unusable by the same ``RuntimeWarning`` the no-hook case raises: an
    unscreened block and a block screened at an unspecified frequency are both wrong,
    and the second is harder to notice.
    """
    hook = getattr(system, "nonlinear_V_block", None)
    if callable(hook):
        if E_ev is not None and _hook_takes_energy(hook):
            # The hook is genuinely in use AND is being told its own harmonic
            # frequency: this is the screened path, so no warning.
            return np.asarray(hook(j, jp, q, float(E_ev)), dtype=np.complex128)
        if getattr(system, "layers", None):
            warnings.warn(
                "nonlinear.py cannot screen this nonlinear block at its own harmonic "
                "frequency: the system carries substrate layers but its "
                "nonlinear_V_block hook does not accept the photon energy "
                "(signature (j, jp, q, E_ev=None) is required), so the substrate "
                "reflection coefficient r(Q, omega) would be evaluated at an "
                "unspecified frequency.",
                RuntimeWarning,
                stacklevel=3,
            )
        return np.asarray(hook(j, jp, q), dtype=np.complex128)

    if getattr(system, "layers", None):
        warnings.warn(
            "nonlinear.py is using the bare operators.build_V kernel: the system carries "
            "substrate layers but exposes no nonlinear_V_block hook, so substrate "
            "screening is absent from the nonlinear blocks.",
            RuntimeWarning,
            stacklevel=3,
        )

    ops = _operators(system)
    r_j = system.ribbons[j]
    r_jp = system.ribbons[jp]
    theta_j, _h_j, _L_j = _grid(system, j)
    theta_jp, h_jp, _L_jp = _grid(system, jp)
    V = ops.build_V(
        q,
        float(r_j.W_nm),
        float(r_jp.W_nm),
        float(r_j.x0_nm),
        float(r_jp.x0_nm),
        float(r_j.z0_nm),
        float(r_jp.z0_nm),
        theta_j,
        theta_jp,
        h_jp,
    )
    return np.asarray(V, dtype=np.complex128)


def _D_matrix(system: Any, j: int, q: float, E_ev: float, s: int) -> np.ndarray:
    """``D_j(s)``: ribbon ``j``'s FD operator with eta taken at harmonic ``s``.

    eta_j^(s) = 1j*sigma_j(s*omega)/(s*omega*W_j*eps_eff), obtained from
    ``system.eta_pair(j, s*q, s*E_ev)`` (paper 406, nonlinear-order block).
    """
    hook = getattr(system, "nonlinear_D", None)
    if callable(hook):
        return np.asarray(hook(j, q, E_ev, s), dtype=np.complex128)

    ops = _operators(system)
    ribbon = system.ribbons[j]
    _theta, h, L = _grid(system, j)
    eta_perp, eta_par = system.eta_pair(j, s * q, s * E_ev)
    D = ops.build_D(s * q, float(ribbon.W_nm), h, L, eta_perp, eta_par)
    return np.asarray(D, dtype=np.complex128)


def _D_geom_matrix(system: Any, j: int, q: float) -> np.ndarray:
    """Eta-free geometric operator ``tridiag/h^2 - (qW)^2 I`` for ribbon ``j``.

    Built at the wave vector it is applied at (``q`` for a fundamental-frequency
    potential, ``2q`` for the second-harmonic potential).  The source block of the
    contract writes a single symbol ``D_geom_j`` for both; at the q = 0 regime the
    formulas are stated for, the two coincide exactly.
    """
    hook = getattr(system, "nonlinear_D_geom", None)
    if callable(hook):
        return np.asarray(hook(j, q), dtype=np.complex128)

    ops = _operators(system)
    ribbon = system.ribbons[j]
    _theta, h, L = _grid(system, j)
    Dg = ops.build_D_geom(q, float(ribbon.W_nm), h, L)
    return np.asarray(Dg, dtype=np.complex128)


def _div_term(system: Any, j: int, q: float, phi: np.ndarray) -> np.ndarray:
    """``-(D_geom_j @ phi) / W_j**2``, i.e. ``d(E_x)/dx``, evaluated as a 3-point stencil.

    ``operators.build_D_geom`` is pinned by the contract to ``tridiag/h**2 - (qW)**2 I``
    (with ``f = None``, which is how this module calls it), so this matvec is computed
    in O(L) from that closed form rather than materialising the dense ``(L, L)`` matrix
    on every source evaluation.  Materialising it dominated
    the source assembly: a 3201-node SHG source took 510 ms with the dense build and
    0.4 ms without, for an identical result.

    ``test_ref_div_term_matches_build_D_geom`` asserts this equals
    ``build_D_geom @ phi`` for the real ``operators`` module, so a different boundary
    convention there fails loudly instead of silently diverging.  A
    ``system.nonlinear_D_geom`` hook, when present, takes precedence and is used as a
    dense matrix.
    """
    if callable(getattr(system, "nonlinear_D_geom", None)):
        W = float(system.ribbons[j].W_nm)
        return -(_D_geom_matrix(system, j, q) @ phi) / W**2

    _theta, h, L = _grid(system, j)
    W = float(system.ribbons[j].W_nm)
    if phi.size != L:
        raise ValueError(
            f"ribbon {j}: potential has {phi.size} nodes but the grid has L={L}"
        )
    lap = np.empty_like(phi)
    lap[1:-1] = phi[:-2] - 2.0 * phi[1:-1] + phi[2:]
    # Boundary rows are [-1, 1]/h**2 and [1, -1]/h**2, not [-2, 1]: the vanishing
    # normal current at the ribbon edge (zero ghost flux) that operators._build_D_parts
    # implements.  A naive [-2, 1] row is wrong by phi[0]/h**2 at the edge node and was
    # caught by test_real_div_term_stencil_matches_operators_build_D_geom.
    lap[0] = phi[1] - phi[0]
    lap[-1] = phi[-2] - phi[-1]
    lap /= h**2
    lap -= (q * W) ** 2 * phi
    return -lap / W**2


def _doping_sign(ribbon: Any) -> float:
    """``sign(EF)`` for a ribbon, as the source block specifies.

    NOTE: an undoped ribbon (``EF == 0``) gives ``0.0``, hence identically zero
    second-order response.  That is implemented as given and is also the physically
    expected statement -- the doping-induced second-order conductivity of graphene
    vanishes at charge neutrality.
    """
    return float(np.sign(float(ribbon.EF_ev)))


# --------------------------------------------------------------------------------------
# field helpers
# --------------------------------------------------------------------------------------
def _fields(
    system: Any, j: int, q: float, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(E_x, dE_x, div_term, dx)`` on ribbon ``j`` for a nodal potential.

    Source block of paper 406::

        dx       = h_j * W_j
        E_x      = -gradient(phi, dx)
        dE_x     =  gradient(E_x, dx)
        div_term = -(D_geom_j @ phi) / W_j**2        # == d(E_x)/dx, smoother

    ``div_term`` and ``dE_x`` are two discretisations of the same continuum quantity
    ``dE_x/dx``; they are kept separate because the reference implementation does.
    """
    ribbon = system.ribbons[j]
    _theta, h, L = _grid(system, j)
    W = float(ribbon.W_nm)
    phi = np.asarray(phi, dtype=np.complex128).ravel()
    if phi.size != L:
        raise ValueError(
            f"ribbon {j}: potential has {phi.size} nodes but the grid has L={L}"
        )
    dx = h * W
    E_x = -np.gradient(phi, dx)
    dE_x = np.gradient(E_x, dx)
    return E_x, dE_x, _div_term(system, j, q, phi), dx


def _as_phi_list(system: Any, phi: PhiLike) -> List[np.ndarray]:
    """Normalise a potential argument into one complex array per ribbon.

    Accepts the dict returned by ``solver.RibbonSystem.solve_linear`` (uses its
    ``'phi'`` entry), a sequence of per-ribbon arrays, or a bare array for a
    single-ribbon system.
    """
    n = len(system.ribbons)
    if isinstance(phi, Mapping):
        if "phi" not in phi:
            raise KeyError("potential mapping has no 'phi' key")
        phi = phi["phi"]

    arr = None
    if isinstance(phi, np.ndarray) and phi.ndim == 1:
        arr = phi
    elif not isinstance(phi, (list, tuple)) and np.ndim(phi) == 1:
        arr = np.asarray(phi)
    if arr is not None:
        if n != 1:
            raise ValueError(
                f"a single flat potential array was given but the system has {n} ribbons"
            )
        return [np.asarray(arr, dtype=np.complex128).ravel()]

    out = [np.asarray(p, dtype=np.complex128).ravel() for p in phi]
    if len(out) != n:
        raise ValueError(f"got {len(out)} potentials for {n} ribbons")
    return out


# --------------------------------------------------------------------------------------
# block solve
# --------------------------------------------------------------------------------------
def _solve_harmonic(
    system: Any,
    q: float,
    E_ev: float,
    s: int,
    rho_nl: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Solve ``rho^(n,s) = solve(I - N, rho_NL)`` with ``N = (W_jp/W_j) D_j(s) V(j,jp,s q)``.

    Paper 406, nonlinear-order block.  Fully vectorised: one dense solve over the
    stacked node vector, no per-node Python loops.

    Returns ``(rho_per_ribbon, V_diagonal_blocks)``; the diagonal blocks are handed
    back so :func:`_self_potential` can reuse them instead of rebuilding ``V(j,j,s q)``
    -- which matters when ``V`` comes from ``substrate.build_Vbar`` and carries an
    NQ-point quadrature per entry.

    Every Coulomb block of this harmonic is requested at ``(s*q, s*E_ev)``, NOT at the
    fundamental: see the PER-ORDER FREQUENCY paragraph of the module docstring.
    """
    ribbons = system.ribbons
    nrib = len(ribbons)
    sizes = [_grid(system, j)[2] for j in range(nrib)]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    ntot = int(offsets[-1])

    # This harmonic's own wave vector and photon energy.  The substrate reflection
    # coefficient is frequency dependent, so s*E_ev -- not E_ev -- is what reaches
    # system.nonlinear_V_block for every block of this order.
    q_block = float(s) * float(q)
    E_block = float(s) * float(E_ev)

    A = np.eye(ntot, dtype=np.complex128)
    v_diag: List[Optional[np.ndarray]] = [None] * nrib
    for j in range(nrib):
        D_j = _D_matrix(system, j, q, E_ev, s)
        W_j = float(ribbons[j].W_nm)
        rows = slice(int(offsets[j]), int(offsets[j + 1]))
        for jp in range(nrib):
            W_jp = float(ribbons[jp].W_nm)
            cols = slice(int(offsets[jp]), int(offsets[jp + 1]))
            V = _V_block(system, j, jp, q_block, E_block)
            if j == jp:
                v_diag[j] = V
            A[rows, cols] -= (W_jp / W_j) * (D_j @ V)

    b = np.concatenate([np.asarray(r, dtype=np.complex128).ravel() for r in rho_nl])
    if b.size != ntot:
        raise ValueError(f"source vector has {b.size} entries, expected {ntot}")
    x = np.linalg.solve(A, b)
    rho = [x[int(offsets[j]) : int(offsets[j + 1])] for j in range(nrib)]
    return rho, [v for v in v_diag if v is not None]


def _self_potential(
    system: Any,
    rho: Sequence[np.ndarray],
    q: float,
    v_diag: Optional[Sequence[np.ndarray]] = None,
    E_ev: Optional[float] = None,
) -> List[np.ndarray]:
    """Self-block induced potential ``phi_j = W_j * V(j, j, q) @ rho_j``.

    This is the quantity the THG cascade block of paper 406 calls ``phi_2w`` (with
    ``q -> 2q``).  Note it uses only the DIAGONAL block: inter-ribbon cascade coupling
    is absent from the reference expression and therefore absent here.

    ``v_diag`` supplies already-built ``V(j, j, q)`` blocks to avoid a second O(L^2)
    kernel build; they must have been built at the SAME ``(q, E_ev)`` pair, which is
    why every caller passes the diagonal blocks of the solve of the same harmonic.
    ``E_ev`` is the photon energy of this block -- already scaled by the harmonic
    index, so ``2*E_ev`` for the cascade's ``phi_2w`` even though it is assembled
    inside a third-order solve -- and reaches a substrate-aware
    ``system.nonlinear_V_block`` hook; it is unused when ``v_diag`` is supplied.
    """
    out: List[np.ndarray] = []
    for j in range(len(system.ribbons)):
        V = (
            v_diag[j]
            if v_diag is not None and j < len(v_diag)
            else _V_block(system, j, j, q, E_ev)
        )
        rho_j = np.asarray(rho[j], dtype=np.complex128).ravel()
        out.append(float(system.ribbons[j].W_nm) * (V @ rho_j))
    return out


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def polarizability_from_rho(system: Any, rho: Sequence[ArrayLike]) -> complex:
    """Polarizability at any order: ``alpha = sum_j W_j^2 * trapz(theta * rho_j, theta)``.

    Paper 406, "Polarizability at any order".
    """
    total = 0.0 + 0.0j
    for j in range(len(system.ribbons)):
        theta, _h, _L = _grid(system, j)
        rho_j = np.asarray(rho[j], dtype=np.complex128).ravel()
        total += float(system.ribbons[j].W_nm) ** 2 * _TRAPZ(theta * rho_j, theta)
    return complex(total)


def shg_source_rho(
    system: Any,
    j: int,
    q: float,
    E_ev: float,
    phi: ArrayLike,
    *,
    doping_sign: Optional[float] = None,
) -> np.ndarray:
    """Second-harmonic nonlinear charge source on ribbon ``j`` (paper 406, SHG source).

    At q = 0, from the converged linear nodal potential ``phi``::

        fct    = sigma_shg_1d_q0(E_ev, gamma_j, doping_sign=sign(EF_j))
        rho_NL = -1j/(2*omega) * fct * ( div_term*dE_x + E_x*gradient(dE_x, dx) )

    The bracket is the discretisation of ``d/dx ( E_x * dE_x/dx )``.

    TENSOR STRUCTURE.  ``sigma_shg_1d_q0`` is the Eq. (12) contraction
    ``sigma_A + sigma_B + 2 sigma_C`` of :func:`conductivity.sigma_shg_abc`; the factor
    2 on ``C`` is there because Eq. (12c)'s ``C`` multiplies ``grad(E.E)``, which is
    ``2 E_x d_x E_x`` in one dimension, while ``A`` and ``B`` multiply structures that
    give ``E_x d_x E_x`` once.  All three therefore collapse onto the single monomial
    ``E_x d_x E_x``, whose divergence is the bracket above.  This is NOT the cascade's
    contraction -- Eq. (15c)'s ``C`` multiplies a different structure and carries
    coefficient 1 there; see :func:`_cascade_A`.  Pinned by
    ``test_shg_source_coefficient_is_the_eq12_contraction``.

    Parameters
    ----------
    phi
        Nodal linear potential on ribbon ``j`` (length ``L_j``), on the physical x grid.
    doping_sign
        Optional override of ``sign(EF_j)``, passed straight to the second-order
        conductivity.  ``None`` (default) uses ``sign(EF_j)`` as specified.  The
        override exists so the ``alpha^(2,2)`` sign law can be tested without
        perturbing the linear and third-order conductivities, which depend on ``EF``
        itself rather than on its sign.

    Returns
    -------
    ndarray
        Complex source density, length ``L_j``.
    """
    ribbon = system.ribbons[j]
    E_x, dE_x, div_term, dx = _fields(system, j, q, np.asarray(phi))
    sign = _doping_sign(ribbon) if doping_sign is None else float(doping_sign)
    fct = _conductivity(system).sigma_shg_1d_q0(
        float(E_ev), float(ribbon.gamma_ev), doping_sign=sign
    )
    bracket = div_term * dE_x + E_x * np.gradient(dE_x, dx)
    return (-1j / (2.0 * _omega(E_ev))) * fct * bracket


def thg_source_rho(
    system: Any,
    j: int,
    q: float,
    E_ev: float,
    phi: ArrayLike,
    phi_2w: Optional[ArrayLike] = None,
    *,
    doping_sign: Optional[float] = None,
    cascade_A_form: str = CASCADE_A_REFERENCE,
) -> np.ndarray:
    """Third-harmonic nonlinear charge source on ribbon ``j`` (paper 406, THG block).

    Local part::

        S3       = sigma_thg_local(E_ev, EF_j, gamma_j)
        d_j33_0  = S3 * ( div_term + 2*dE_x ) * E_x**2
        rho_NL_0 = -1j/(3*omega) * d_j33_0

    Since ``div_term`` and ``dE_x`` are both discretisations of ``dE_x/dx``, the
    weighted combination is ``3 * E_x**2 * dE_x/dx = d(E_x**3)/dx``; the split
    weighting is the reference implementation's and is kept verbatim.

    Cascade part (mixing of the fundamental with its own second harmonic), active when
    ``phi_2w`` is given::

        A12 = _cascade_A(..., 1, 2, doping_sign)   # see _cascade_A for the discrepancy
        A21 = _cascade_A(..., 2, 1, doping_sign)
        d_j33 = d_j33_0 + A12*( div_term*dE_2 + E_x*gradient(dE_2, dx) )
                        + A21*( div_2*dE_x    + E_2*gradient(dE_x, dx) )

    CONTRACT DEVIATION 1 (flagged deliberately): the source block writes
    ``sigma_cascade_abcd(E, gamma, 1, 2)[0]``, i.e. the default ``doping_sign=+1``.
    The ribbon's doping sign is passed instead.  The cascade term is linear in
    ``rho_shg``, which already carries one factor of ``doping_sign``; a second-order
    coefficient ``A`` that is odd in ``doping_sign`` makes the product even, which is
    what the contract's own test requirement ("flipping doping_sign leaves
    alpha^(3,3) unchanged") demands.  Using the literal default would make
    ``alpha^(3,3)`` flip sign.  See ``test_ref_cascade_term_is_even_in_doping_sign``.

    CONTRACT DEVIATION 2: ``A12``/``A21`` come from :func:`_cascade_A`, which by
    default evaluates the paper-406 MATLAB reference expression -- the FULL Eq. (15)
    contraction ``A + B + C + D`` -- rather than ``sigma_cascade_abcd(...)[0]``, which
    is Eq. (15a)'s tensor component ``A`` alone.  The two are different objects, not
    two readings of one; see :func:`_cascade_A`.  Pass
    ``cascade_A_form="conductivity"`` to use the bare component for comparison.

    TENSOR STRUCTURE OF THE TWO CASCADE TERMS.  The brackets are
    ``d/dx (E_x d_x E_2)`` for the ``(s1, s2) = (1, 2)`` ordering and
    ``d/dx (E_2 d_x E_x)`` for ``(2, 1)``: one 1-D monomial per ordering, each carrying
    its own ordering's full contraction.  That is what supplies both terms of
    Eq. (15d)'s ``grad(E_s1 . E_s2)``; ``D`` is symmetric under ``s1 <-> s2``, so
    assigning it once per ordering is what makes the degenerate limit reduce to
    ``sigma_shg_1d_q0``.

    Parameters
    ----------
    phi_2w
        Nodal second-harmonic potential on ribbon ``j``, ``W_j * V(j,j,2q) @ rho_shg_j``.
        ``None`` disables the cascade and returns the local part only.
    cascade_A_form
        ``"reference"`` (default) or ``"conductivity"``; see :func:`_cascade_A`.
    """
    ribbon = system.ribbons[j]
    E_x, dE_x, div_term, dx = _fields(system, j, q, np.asarray(phi))
    cond = _conductivity(system)
    sign = _doping_sign(ribbon) if doping_sign is None else float(doping_sign)

    S3 = cond.sigma_thg_local(float(E_ev), float(ribbon.EF_ev), float(ribbon.gamma_ev))
    d_j33 = S3 * (div_term + 2.0 * dE_x) * E_x**2

    if phi_2w is not None:
        E_2, dE_2, div_2, _dx2 = _fields(system, j, 2.0 * q, np.asarray(phi_2w))
        gam = float(ribbon.gamma_ev)
        A12 = _cascade_A(cond, float(E_ev), gam, 1, 2, sign, cascade_A_form)
        A21 = _cascade_A(cond, float(E_ev), gam, 2, 1, sign, cascade_A_form)
        d_j33 = (
            d_j33
            + A12 * (div_term * dE_2 + E_x * np.gradient(dE_2, dx))
            + A21 * (div_2 * dE_x + E_2 * np.gradient(dE_x, dx))
        )

    return (-1j / (3.0 * _omega(E_ev))) * d_j33


def solve_shg(
    system: Any,
    q: float,
    E_ev: float,
    phi_lin: PhiLike,
    *,
    doping_sign: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve the second-order, second-harmonic problem ``(n, s) = (2, 2)``.

    Assembles ``rho_NL^(2,2)`` from the converged linear potential (see
    :func:`shg_source_rho`) and solves ``rho^(2,2) = solve(I - N, rho_NL)`` with
    ``N = (W_jp/W_j) D_j(2) V(j, jp, 2q)`` (paper 406, nonlinear-order block).

    Returns
    -------
    dict
        ``'rho'``    list of per-ribbon ``rho^(2,2)`` arrays
        ``'rho_nl'`` list of per-ribbon nonlinear sources
        ``'phi'``    list of per-ribbon SELF-block induced potentials at 2 omega,
                     ``W_j * V(j,j,2q) @ rho_j`` -- the ``phi_2w`` of the cascade block
        ``'alpha'``  ``alpha^(2,2)``
        ``'harmonic'`` 2
    """
    phis = _as_phi_list(system, phi_lin)
    rho_nl = [
        shg_source_rho(system, j, q, E_ev, phis[j], doping_sign=doping_sign)
        for j in range(len(system.ribbons))
    ]
    rho, v_diag = _solve_harmonic(system, q, E_ev, 2, rho_nl)
    return {
        "rho": rho,
        "rho_nl": rho_nl,
        # Second-harmonic blocks: screened at (2q, 2*E_ev), the frequency of THIS
        # order, not the fundamental.  v_diag carries exactly those blocks.
        "phi": _self_potential(system, rho, 2.0 * q, v_diag, 2.0 * float(E_ev)),
        "alpha": polarizability_from_rho(system, rho),
        "harmonic": 2,
    }


def solve_thg(
    system: Any,
    q: float,
    E_ev: float,
    phi_lin: PhiLike,
    rho_shg: Optional[Sequence[ArrayLike]] = None,
    cascade: bool = True,
    *,
    doping_sign: Optional[float] = None,
    cascade_A_form: str = CASCADE_A_REFERENCE,
) -> Dict[str, Any]:
    """Solve the third-order, third-harmonic problem ``(n, s) = (3, 3)``.

    Local part plus, when ``cascade`` is true, the second-order mixing of the
    fundamental with its own second harmonic (paper 406, THG blocks; see
    :func:`thg_source_rho`).  The block solve uses ``D_j(3)`` and ``V(j, jp, 3q)``.

    Parameters
    ----------
    rho_shg
        Per-ribbon ``rho^(2,2)``.  Ignored when ``cascade`` is false; computed
        internally via :func:`solve_shg` when ``cascade`` is true and this is ``None``.
    cascade
        Include the cascaded contribution.
    cascade_A_form
        ``"reference"`` (default) or ``"conductivity"``; see :func:`_cascade_A` for the
        discrepancy between the two.

    Returns
    -------
    dict
        ``'rho'``, ``'rho_nl'``, ``'phi'`` (self-block potential at 3 omega),
        ``'alpha'`` = ``alpha^(3,3)``, ``'harmonic'`` = 3, ``'cascade'``,
        and ``'rho_shg'`` / ``'phi_2w'`` (``None`` when the cascade is off).
    """
    phis = _as_phi_list(system, phi_lin)

    phi_2w: Optional[List[np.ndarray]] = None
    rho_shg_list: Optional[List[np.ndarray]] = None
    if cascade:
        if rho_shg is None:
            shg = solve_shg(system, q, E_ev, phis, doping_sign=doping_sign)
            rho_shg_list = list(shg["rho"])
            phi_2w = list(shg["phi"])
        else:
            rho_shg_list = [
                np.asarray(r, dtype=np.complex128).ravel() for r in rho_shg
            ]
            # phi_2w is a SECOND-harmonic potential even though it is built inside a
            # third-order solve, so its blocks are screened at (2q, 2*E_ev).
            phi_2w = _self_potential(
                system, rho_shg_list, 2.0 * q, None, 2.0 * float(E_ev)
            )

    rho_nl = [
        thg_source_rho(
            system,
            j,
            q,
            E_ev,
            phis[j],
            None if phi_2w is None else phi_2w[j],
            doping_sign=doping_sign,
            cascade_A_form=cascade_A_form,
        )
        for j in range(len(system.ribbons))
    ]
    rho, v_diag = _solve_harmonic(system, q, E_ev, 3, rho_nl)
    return {
        "rho": rho,
        "rho_nl": rho_nl,
        # Third-harmonic blocks: screened at (3q, 3*E_ev).
        "phi": _self_potential(system, rho, 3.0 * q, v_diag, 3.0 * float(E_ev)),
        "alpha": polarizability_from_rho(system, rho),
        "harmonic": 3,
        "cascade": bool(cascade),
        "rho_shg": rho_shg_list,
        "phi_2w": phi_2w,
    }
