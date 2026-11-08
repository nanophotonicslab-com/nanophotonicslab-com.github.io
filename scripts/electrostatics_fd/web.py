"""JSON-in / JSON-out driver for the NPLab Electrostatics-FD Toolkit page.

The Toolkit page runs *this* library in the browser through Pyodide, so there is
no second implementation to keep in parity with the first.  What it needs that
the library proper does not provide is a boundary: plain dicts and lists in,
plain dicts and lists out, every complex array split into ``re``/``im``, every
numpy scalar demoted to a Python float, and an explicit, enforced ceiling on how
much work a single call may do.  That boundary is this module, and it lives in
the library rather than in the worker's TypeScript so that it is testable with
pytest like everything else here.

Design rules, in the order they matter:

1. **Nothing is silently truncated.**  Every call returns a ``notes`` list, and
   anything this module clamps, skips or routes differently says so there in a
   sentence a user can read.  A grid that was quietly halved to keep the page
   responsive would make the plot a lie.
2. **Cost before compute.**  :func:`cost_estimate` answers "how big is this
   going to be?" from the spec alone, so the page can warn before a click
   freezes a tab.  These calculations get demanding and the demand is not where
   it first looks -- a single solve is small, the sweep is what costs.
3. **The fast path is used when it is valid and never faked.**  ``sweep`` and
   ``ldos_map`` raise rather than fall back when drift, a substrate, or unequal
   ribbons invalidate the isotropic eigen decomposition.  This module catches
   that, retries honestly on the slow path, and records the downgrade in
   ``notes`` along with what it cost.
4. **Units are the library's: eV, nm, fs, Gaussian.**  Spec keys carry their
   unit in the name (``W_nm``, ``EF_eV``) so a caller cannot guess wrong.

Entry points, all ``spec -> dict``, dispatched by :func:`run`:

``geometry``          node coordinates for the schematic
``eigenmodes``        Level 2 at one q: eta, zeta, bright/dark, mode profiles
``bands``             Level 2 over a q grid
``sheet_dispersion``  Level 1: extended-sheet reflection over (Q, E), +-Q split
``linear_spectrum``   polarizability / extinction vs energy
``eels_spectrum``     swift-electron loss spectrum
``ldos``              Level 3: LDOS map over (q, E), peak tracks, +-q asymmetry
``nonlinear_spectrum``  SHG and THG polarizability spectra

See ``notes/electrostatics-fd-notes.html`` for the formalism and its section
numbers, and its section 15 for what is validated against the reference
MATLAB and what is not.
Two things this module will not let you ask for, because the library would be
answering a question it has not been checked on: an absolute LDOS is returned in
arbitrary units and labelled as such (the ``2 pi^2 |p|^2`` normalisation has no
external confirmation), and a layered calculation is flagged in ``notes`` as
screening the response while driving with the free-space source (notes 8.4).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import dispersion as _disp
from . import nonlinear as _nl
from . import observables as _obs
from . import sources as _src
from .conductivity import (
    drift_chemical_potential,
    graphene_drift_sigma,
    graphene_drude,
    graphene_local_rpa,
    metal_thin_film,
    thermal_drude_weight,
)
from .solver import Ribbon, RibbonSystem
from .units import C_NM_FS, HBAR_EV_FS

#: ``np.trapz`` is deprecated in numpy 2.0 and ONLY the browser sees it --
#: Pyodide ships 2.0.2 while this desktop has 1.x -- so the bare call warns in
#: the one place the warning is visible to a user, because the JSON boundary
#: surfaces library warnings as notes. observables, sources and nonlinear all
#: carry this same line; web.py was the one that did not.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz
from .substrate import Layer, reflection_coefficient

__all__ = [
    "SCHEMA_VERSION",
    "MAX_NODES",
    "MAX_SPECTRUM_POINTS",
    "MAX_MAP_CELLS",
    "MAX_SLOW_SOLVES",
    "MAX_ESTIMATED_SECONDS",
    "WebSpecError",
    "run",
    "calibrate",
    "cost_estimate",
    "presets",
    "geometry",
    "material",
    "near_field",
    "eigenmodes",
    "bands",
    "sheet_dispersion",
    "linear_spectrum",
    "eels_spectrum",
    "ldos",
    "nonlinear_spectrum",
]

SCHEMA_VERSION = 1

# Ceilings.  Chosen from the measured cost of the library rather than from
# taste.  2048 nodes is where a single solve reaches about a second natively,
# and it is deliberately high enough to reach the reference MATLAB's own
# grid for the paper-406 two-ribbon case (L = [1600, 400] = 2000 nodes), so
# that preset can be run at its published resolution rather than only at a
# coarsened one.  MAX_ESTIMATED_SECONDS below is what stops that becoming a
# frozen tab.
MAX_NODES = 2048
MAX_SPECTRUM_POINTS = 800
MAX_MAP_CELLS = 60_000
# On the slow path every (q, E) cell is a full assemble-and-solve.
#
# A BACKSTOP, NOT A SECOND CEILING.  MAX_ESTIMATED_SECONDS is the real limit --
# its own comment says why, that 300 solves is harmless at 150 nodes and five
# minutes at 2000 -- and this tally was being applied on top of it, in AND.  So
# a run the time model priced in seconds was refused anyway for having too many
# cheap solves: measured 2026-11-06, an EELS grid of 5120 cells estimated 13.2 s
# and RAN IN 8.6 s, and the page would not start it. That is not a safety
# ceiling, it is a locked door.
#
# It still applies where the time model is known to under-report, which is the
# screened case and only that: with a substrate every rebuild is an O(N*NQ)
# quadrature the model does not count, measured 4x to 16x low (see
# cost_estimate's `estimate_is_floor`). Unscreened the same model is good to
# 1.3x, so there the seconds govern.
MAX_SLOW_SOLVES = 4_000


class WebSpecError(ValueError):
    """A spec the page sent is unusable, with a message meant for the user."""


# --------------------------------------------------------------------------- #
# JSON boundary helpers
# --------------------------------------------------------------------------- #


def _f(x: Any) -> float:
    """A JSON-safe float.  numpy scalars and 0-d arrays included."""
    return float(np.asarray(x).reshape(()).item())


def _fl(a: Any) -> List[float]:
    """A JSON-safe list of floats, with non-finite values mapped to ``None``."""
    arr = np.asarray(a, dtype=np.float64).ravel()
    if np.all(np.isfinite(arr)):
        return [float(v) for v in arr]
    # NaN and +-inf are not JSON values.  null survives the bridge and plots as
    # a gap, which is what a non-converged band or a missing peak should look
    # like -- never as a zero.
    return [None if not np.isfinite(v) else float(v) for v in arr]  # type: ignore[list-item]


def _cx(a: Any) -> Dict[str, List[float]]:
    """A complex array as ``{"re": [...], "im": [...]}``."""
    arr = np.asarray(a, dtype=np.complex128).ravel()
    return {"re": _fl(arr.real), "im": _fl(arr.imag)}


def _rows(a: Any) -> List[List[float]]:
    """A 2D real array as a list of rows."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    return [_fl(row) for row in arr]


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #


def _get(spec: Dict[str, Any], key: str, default: Any = None) -> Any:
    """``spec[key]`` tolerating the JS side sending ``null`` for "unset"."""
    v = spec.get(key, default)
    return default if v is None else v


def _grid(block: Any, *, name: str, cap: int) -> np.ndarray:
    """A 1D grid from either an explicit list or a ``{min, max, n}`` block."""
    if isinstance(block, dict):
        lo, hi = _f(block.get("min", 0.0)), _f(block.get("max", 1.0))
        n = int(block.get("n", 101))
        if n < 1:
            raise WebSpecError(f"{name}: n must be at least 1 (got {n})")
        if n > cap:
            raise WebSpecError(
                f"{name}: {n} points exceeds the {cap}-point ceiling. "
                f"Narrow the range or ask for fewer points."
            )
        if hi < lo:
            lo, hi = hi, lo
        return np.linspace(lo, hi, n) if n > 1 else np.array([lo])
    arr = np.asarray(block, dtype=np.float64).ravel()
    if arr.size == 0:
        raise WebSpecError(f"{name}: empty grid")
    if arr.size > cap:
        raise WebSpecError(f"{name}: {arr.size} points exceeds the {cap}-point ceiling")
    return arr


def _energies(spec: Dict[str, Any], *, cap: int = MAX_SPECTRUM_POINTS) -> np.ndarray:
    block = _get(spec, "energies_eV", _get(spec, "energies", {"min": 0.05, "max": 1.5, "n": 240}))
    Es = _grid(block, name="energies_eV", cap=cap)
    if np.any(Es <= 0.0):
        raise WebSpecError("energies_eV: every energy must be > 0 (sigma diverges at omega = 0)")
    return Es


def _qs(spec: Dict[str, Any], *, cap: int = MAX_SPECTRUM_POINTS) -> np.ndarray:
    block = _get(spec, "qs_nm_inv", _get(spec, "qs"))
    if block is None:
        return np.array([_f(_get(spec, "q_nm_inv", 0.0))])
    if isinstance(block, dict) and "n_pos" in block:
        # A +-q grid, which is the one the non-reciprocity map needs.
        q_lo = block.get("min")
        want_zero = bool(block.get("include_zero", False))
        if q_lo is not None and _f(q_lo) <= 0.0:
            # "q from 0 to 0.5" is a perfectly ordinary request, and q = 0 is a
            # perfectly ordinary point: it is the NORMAL-INCIDENCE case, where
            # the ribbon eigenvalues are the published -0.0687, -0.0289, -0.0184
            # and the bands are the plane-wave resonances the cross section
            # shows.  `symmetric_q_grid`'s `q_min` is the smallest POSITIVE
            # sample, though, so zero is not a legal value for it -- zero is
            # asked for with `include_zero`.  Passing it straight through gave
            # "need 0 < q_min <= q_max, got (0.0, 0.5)", a refusal from the grid
            # builder that never reached the physics and told the reader nothing
            # about what to do.  Read it as what it plainly means instead.
            want_zero = True
            q_lo = None
        qs = _disp.symmetric_q_grid(
            _f(block.get("max", 0.3)),
            int(block["n_pos"]),
            include_zero=want_zero,
            q_min=None if q_lo is None else _f(q_lo),
        )
        if qs.size > cap:
            raise WebSpecError(f"qs: {qs.size} points exceeds the {cap}-point ceiling")
        return qs
    return _grid(block, name="qs_nm_inv", cap=cap)


def _one_layer(d: Dict[str, Any], where: str) -> Layer:
    try:
        return Layer(
            eps_par=complex(d.get("eps_par", 1.0)),
            eps_perp=None if d.get("eps_perp") is None else complex(d["eps_perp"]),
            d_nm=_f(d.get("d_nm", 0.0)),
            sigma_sheet=complex(d.get("sigma_sheet", 0.0)),
        )
    except (TypeError, ValueError) as exc:
        raise WebSpecError(f"{where} is not a valid layer: {exc}") from exc


def _layers(spec: Dict[str, Any], eps_a: complex) -> Optional[List[Layer]]:
    """Build the library's layer stack from the page's simpler description.

    The library indexes layers from the top with ``layers[0]`` being the
    half-space the ribbons sit in, so ``layers[0].eps_par`` must equal ``eps_a``
    and ``layers[0].d_nm`` must be zero (the standoff belongs in the ribbon's
    ``z0_nm``; putting it in both double-counts the source-to-image round trip).
    Two ways to get that wrong, so the page does not get the chance: it sends
    ``substrate`` -- the stack strictly BELOW the ribbons' own medium -- and the
    ambient layer is prepended here.  ``layers`` remains as a full-stack escape
    hatch for anything that needs the raw convention.
    """
    below = _get(spec, "substrate")
    full = _get(spec, "layers")
    if below and full:
        raise WebSpecError(
            "Pass either 'substrate' (the stack below the ribbons, ambient added for you) "
            "or 'layers' (the full stack, layers[0] = the ribbons' own medium), not both."
        )
    if full:
        return [_one_layer(d, f"layers[{i}]") for i, d in enumerate(full)]
    if not below:
        return None
    return [Layer(eps_par=eps_a, d_nm=0.0)] + [
        _one_layer(d, f"substrate[{i}]") for i, d in enumerate(below)
    ]


def _build_system(spec: Dict[str, Any]) -> RibbonSystem:
    raw = _get(spec, "ribbons")
    if not raw:
        raise WebSpecError("ribbons: at least one ribbon is required")

    dx_nm = _f(_get(spec, "dx_nm", 0.1))
    if dx_nm <= 0.0:
        raise WebSpecError(f"dx_nm must be > 0 (got {dx_nm})")

    # Temperature is a property of each sheet, not of the surroundings, so it
    # is read per ribbon.  A spec-level ``T_K`` is still honoured as the default
    # for every ribbon that does not carry its own -- older specs, the preset
    # strip and the permalink all wrote it there.
    spec_T = _f(_get(spec, "T_K", 0.0))

    ribbons: List[Ribbon] = []
    for i, d in enumerate(raw):
        W = _f(d.get("W_nm", 10.0))
        if W <= 0.0:
            raise WebSpecError(f"ribbons[{i}].W_nm must be > 0 (got {W})")
        L = d.get("L")
        ribbons.append(
            Ribbon(
                W_nm=W,
                x0_nm=_f(d.get("x0_nm", 0.0)),
                z0_nm=_f(d.get("z0_nm", 0.0)),
                EF_ev=_f(d.get("EF_eV", d.get("EF_ev", 0.5))),
                gamma_ev=_f(d.get("gamma_eV", d.get("gamma_ev", 0.02))),
                beta=_f(d.get("beta", 0.0)),
                T_K=_f(d.get("T_K", spec_T)),
                L=None if L is None else int(L),
            )
        )

    total = sum(r.L if r.L is not None else max(3, int(round(r.W_nm / dx_nm)) | 1) for r in ribbons)
    if total > MAX_NODES:
        raise WebSpecError(
            f"{total} total nodes exceeds the {MAX_NODES}-node ceiling. "
            f"Raise dx_nm (currently {dx_nm} nm) or set L per ribbon. "
            f"Solve time scales as the cube of the node count."
        )

    eps_a = complex(_get(spec, "eps_a", 1.0))
    return RibbonSystem(
        ribbons,
        eps_a=eps_a,
        eps_b=complex(_get(spec, "eps_b", 1.0)),
        layers=_layers(spec, eps_a),
        dx_nm=dx_nm,
    )


def _layer_note(system: RibbonSystem, notes: List[str]) -> None:
    """Say the honest things about every layered run (notes 7.1, 8.4)."""
    if not system.layers:
        return
    notes.append(
        "Substrate present: both halves are screened through the same r(Q, omega) -- "
        "the response through Vbar, and the source itself, which the stack reflects "
        "before it reaches the sheet. The layered path still has no reference "
        "implementation to have been checked against; its limits are analytic "
        "(a perfect mirror reproduces the textbook image dipole and image charge)."
    )
    if any(r.z0_nm == 0.0 for r in system.ribbons):
        notes.append(
            "A ribbon sits at z0_nm = 0, in the plane of its own image. The reflected "
            "Coulomb integrand then decays only as 1/Q and the quadrature bottoms out "
            "around 7e-3 relative however finely it is refined. Lift the ribbon."
        )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def _tune_vbar(
    system: RibbonSystem, qs: np.ndarray, spec: Dict[str, Any], notes: List[str]
) -> None:
    """Pick the substrate quadrature so it resolves what it is integrating.

    ``build_Vbar`` integrates ``r(Q)/Q * exp(-Q(z_j+z_j')) * cos(Qx dx)`` on a
    uniform grid, and the library's default ``Qmax`` carries a ``200/|dx|`` term
    that its own docstring flags as running the wrong way: it inflates ``Qmax``
    for *small* separations while ``NQ`` stays fixed, leaving the feature of
    width ``|q|`` near ``Qx = 0`` unresolved.  Raising ``Qmax`` cannot help
    resolve an oscillation -- only ``NQ`` can -- so this picks ``Qmax`` from the
    truncation requirement alone (the exponential dies at ``Q ~ 1/z_sum``, and
    50 e-folds is ample) and then sets ``NQ`` from the resolution requirement.

    Both are documented public tunables, both are in the cache key since the
    stale-``Vbar`` fix, and ``vbar_auto: false`` in the spec turns this off.
    Anything it clamps is reported in ``notes``.
    """
    if not system.layers or not _get(spec, "vbar_auto", True):
        return
    if spec.get("vbar_Qmax") is not None or spec.get("vbar_NQ") is not None:
        system.vbar_Qmax = (
            None if spec.get("vbar_Qmax") is None else _f(spec["vbar_Qmax"])
        )
        if spec.get("vbar_NQ") is not None:
            system.vbar_NQ = int(spec["vbar_NQ"])
        notes.append("Substrate quadrature taken from the spec, not auto-tuned.")
        return

    z_sum = 2.0 * min(abs(_f(r.z0_nm)) for r in system.ribbons)
    q_abs = np.abs(np.asarray(qs, dtype=float))
    q_ref = float(np.min(q_abs[q_abs > 0.0])) if np.any(q_abs > 0.0) else 0.0
    # The widest node separation sets the cosine's frequency.
    dx_max = max(
        float(np.max(x) - np.min(x)) for x in system.x_nodes
    ) or 1.0
    span = max(
        float(np.max([np.max(x) for x in system.x_nodes]))
        - float(np.min([np.min(x) for x in system.x_nodes])),
        dx_max,
    )

    Qmax = max(50.0 / max(z_sum, 0.05), 50.0 * float(np.max(q_abs)))
    feature = min(q_ref if q_ref else np.inf, 1.0 / max(span, 0.1))
    want = int(np.ceil(8.0 * Qmax / feature)) + 1
    NQ = int(min(max(want, 4000), 60_000))

    system.vbar_Qmax = Qmax
    system.vbar_NQ = NQ
    notes.append(
        "Substrate quadrature auto-tuned: Qmax = %.3g 1/nm (truncation: the image "
        "exponential dies at Q ~ %.3g), NQ = %d (resolution: the narrowest feature is "
        "%.3g 1/nm)." % (Qmax, 1.0 / max(z_sum, 0.05), NQ, feature)
    )
    if want > NQ:
        notes.append(
            "That NQ is CLAMPED: resolving this q at this standoff wants NQ = %d. The "
            "screened blocks are correspondingly less converged -- raise q, raise the "
            "standoff, or pass vbar_NQ explicitly and accept the wait." % want
        )


def _guard_q(system: RibbonSystem, qs: np.ndarray) -> None:
    """Refuse ``q = 0`` with a stack, before the library has to.

    ``build_Vbar``'s integrand goes as ``r(Q)/Q`` with ``Q = sqrt(Qx^2 + q^2)``,
    so at ``q = 0`` the reflected Coulomb integral is logarithmically divergent
    and the library raises rather than returning a number that depends on where
    the quadrature was cut off.  Catching it here turns a traceback into a
    sentence that says what to do instead.
    """
    if system.layers and np.any(qs == 0.0):
        raise WebSpecError(
            "q = 0 with a substrate: the reflected Coulomb integral is logarithmically "
            "divergent there, so its value would be set by where the quadrature stops. "
            "Use a small finite q (the physics is smooth in q, so q W of order 0.01 is "
            "already the q -> 0 answer), or remove the substrate and set eps_a/eps_b "
            "for a uniform environment."
        )


def _source_fn(
    spec: Dict[str, Any], system: RibbonSystem, qs_hint: Any = 1.0
) -> Callable[[float, float], Any]:
    """Return ``fn(q, E_ev) -> per-ribbon list of phi_ext arrays``.

    ``qs_hint`` is the q grid the caller will sweep, used only to refuse a source
    that is undefined on it before any solving starts.

    WITH A SUBSTRATE THE SOURCE IS DIRECT PLUS REFLECTED.  The stack reflects
    the source's own field before it reaches the sheet, and that term is the
    same order as the screening of the response: dropping it while keeping
    ``build_Vbar`` is wrong by O(r) in the drive.  Both halves use the same
    ``r(Q, omega)``, so the reflected source and the reflected Coulomb block
    cannot disagree about the stack.
    """
    src = _get(spec, "source", {"kind": "plane_wave"})
    kind = str(src.get("kind", "plane_wave"))
    xs = system.x_nodes

    if kind == "plane_wave":
        E0 = complex(src.get("E0", 1.0))
        phi = [_src.phi_ext_plane_wave(x, E0) for x in xs]
        return lambda q, E: phi

    layers = system.layers
    eps_a = system.eps_a

    def _r_of_Q(E_ev: float) -> Callable[[Any], Any]:
        """``r(Q)`` for this stack at one energy, for the reflected source."""
        return lambda Q: reflection_coefficient(Q, E_ev, layers, eps_a)

    if kind == "dipole":
        # The dipole's q = 0 Fourier component does not exist: K0(|q|s) and
        # K1(|q|s) both diverge as q -> 0, because a point dipole's potential is
        # not integrable along the invariant direction. The library raises, which
        # is right, but it reaches the page as an exception rather than as advice.
        if np.any(np.asarray(qs_hint, dtype=float) == 0.0):
            raise WebSpecError(
                "A point dipole has no q = 0 Fourier component: K0(|q|s) and K1(|q|s) "
                "both diverge as q -> 0. Set a finite q (0.01-0.1 nm^-1 is a sensible "
                "range for a 10 nm ribbon), or drive with a plane wave, which is "
                "exactly the q = 0 case."
            )
        r0 = [_f(v) for v in src.get("r0_nm", (0.0, 0.0, 5.0))]
        p = [complex(v) for v in src.get("p", (1.0, 0.0, 0.0))]
        zs = [np.full_like(np.asarray(x, dtype=float), r.z0_nm) for x, r in zip(xs, system.ribbons)]

        def _dipole(q: float, E: float) -> Any:
            direct = [_src.phi_ext_dipole(q, x, z, r0, p, eps_a) for x, z in zip(xs, zs)]
            if not layers:
                return direct
            rq = _r_of_Q(E)
            return [
                d + _src.phi_ext_dipole_reflected(
                    q, x, z, r0, p, rq, eps_a,
                    Qmax=system.vbar_Qmax, NQ=system.vbar_NQ,
                )
                for d, x, z in zip(direct, xs, zs)
            ]

        return _dipole

    if kind == "electron":
        v_over_c = _f(src.get("v_over_c", 0.2))
        b_nm = _f(src.get("b_nm", 5.0))
        zs = [np.full_like(np.asarray(x, dtype=float), r.z0_nm) for x, r in zip(xs, system.ribbons)]

        def _electron(q: float, E: float) -> Any:
            direct = [
                _src.phi_ext_electron(q, x, z, E, v_over_c, b_nm, layers)
                for x, z in zip(xs, zs)
            ]
            if not layers:
                return direct
            rq = _r_of_Q(E)
            return [
                d + _src.phi_ext_electron_reflected(
                    q, x, z, E, v_over_c, b_nm, rq, layers
                )
                for d, x, z in zip(direct, xs, zs)
            ]

        return _electron

    raise WebSpecError(
        f"source.kind {kind!r} is not one of 'plane_wave', 'dipole', 'electron'"
    )


# --------------------------------------------------------------------------- #
# Material response, for the conductivity plot
# --------------------------------------------------------------------------- #


def _extinction(
    system: RibbonSystem, Es: np.ndarray, alpha: np.ndarray
) -> Dict[str, Any]:
    """Extinction cross section of the structure, from its polarizability.

        sigma_ext = (4 pi omega / c) Im alpha                     [nm, per unit length]
        sigma_ext / A = that divided by sum_j W_j                 [dimensionless]

    The prefactor is the house one, the same ``(4 pi omega/c) Im alpha^(1,1)``
    the plasmonic-nanoparticles lab prints under its own spectrum, and the
    per-area form is how that lab reports it -- for a structure that is
    infinite along y both the cross section and the area are per unit length,
    so the ratio is the efficiency and carries no units.

    Only the absorptive term.  In 3D that lab adds a scattering term
    ``(8 pi/3)(omega/c)^4 |alpha|^2``; here it is dropped, for two reasons.  It
    has a different form for a line dipole than for a point one, so the 3D
    expression would simply be wrong, and in the quasistatic regime these
    ribbons live in -- ribbon widths of nanometres against free-space
    wavelengths of micrometres -- it is smaller than the absorptive term by
    orders of magnitude.  The name ``sigma_ext`` is kept because extinction is
    what it is to the accuracy of the model.

    ALPHA'S CONVENTION, which is the whole scale of the answer:
    ``solver.polarizability`` returns the dipole of ``rho`` in the library's
    charge convention, ``rho_phys / eps_eff``, so the physical polarizability
    is ``eps_eff`` times it.  In vacuum eps_eff = 1 and the two coincide, which
    is the case every reference figure here is in; with a substrate they do
    not, and the factor is applied.
    """
    omega = np.asarray(Es, dtype=np.float64) / HBAR_EV_FS            # [1/fs]
    eps_eff = float(np.real(system.eps_eff))
    width = float(sum(r.W_nm for r in system.ribbons))
    pref = 4.0 * np.pi * (omega / C_NM_FS) * eps_eff                 # [1/nm]
    sig = pref[None, :] * np.imag(np.atleast_2d(alpha))              # [nm]
    out: Dict[str, Any] = {"sigma_ext_nm": _rows(sig)}
    if width > 0.0:
        out["sigma_ext_per_area"] = _rows(sig / width)
        out["total_width_nm"] = width
    return out


def _sigma(
    spec: Dict[str, Any], Es: np.ndarray, rib: Optional[Ribbon] = None
) -> Dict[str, Any]:
    """Sheet conductivity over the energy grid, for the material panel.

    With ``rib`` given, E_F, gamma and T come from that ribbon rather than from
    the spec-level ``material`` block -- every ribbon carries its own, and the
    panel shows all of them.  The ``kind`` (graphene or thin metal) stays a
    property of the spec: it is the model, not a per-ribbon knob.
    """
    mat = _get(spec, "material", {})
    kind = str(mat.get("kind", "graphene"))
    if kind == "metal":
        eps = complex(mat.get("eps_re", -10.0), mat.get("eps_im", 1.0))
        return {"kind": "metal", "sigma_nm_fs": _cx(metal_thin_film(Es, eps, _f(mat.get("t_nm", 1.0))))}

    if rib is not None:
        EF, gam, T_K = _f(rib.EF_ev), _f(rib.gamma_ev), _f(rib.T_K)
    else:
        EF = _f(mat.get("EF_eV", 0.5))
        gam = _f(mat.get("gamma_eV", 0.02))
        T_K = _f(_get(spec, "T_K", mat.get("T_K", 0.0)))
    # Two curves, and they are deliberately two different conductivities.
    #
    # `total` is the full local-RPA sigma, intraband + interband, which is what
    # the sheet actually does.  `drude` is the intraband part ALONE, which is
    # what `solver.eta_pair` carries and therefore what every resonance on this
    # page is computed from -- so the panel has to show it, or eta and sigma
    # disagree with nothing on screen explaining why.
    #
    # Both are warmed the same way eta is: the Drude weight |E_F| becomes
    # mu_D(mu(T), T).  Before 2026-11-05 this branch dropped the Drude trace
    # entirely at T > 0, which both hid the comparison and made the trace blink
    # out between 0 and 1 K, where graphene_local_rpa still takes its T = 0
    # branch.  There is no branch here now: at T = 0 the two calls below return
    # |E_F| exactly, so the cold case is unchanged.
    mu = drift_chemical_potential(EF, 0.0, T_K)
    weight = float(thermal_drude_weight(mu, T_K))
    drude = graphene_drude(Es, weight, gam)
    total = graphene_local_rpa(Es, EF, gam, T_K=T_K)
    out: Dict[str, Any] = {
        "kind": "graphene",
        "EF_eV": EF,
        "abs_EF_eV": abs(EF),
        "T_K": T_K,
        "sigma_nm_fs": _cx(total),
        "mu_drude_eV": weight,
        "pauli_edge_eV": 2.0 * abs(EF),
        "sigma_drude_nm_fs": _cx(drude),
    }
    if EF < 0.0:
        out["note"] = (
            "Hole doping: the Dirac cone is electron-hole symmetric, so the response is the "
            "same as at |E_F|. Signed formulas would give Re sigma < 0 here, which would be "
            "gain from a passive sheet."
        )
    return out


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def geometry(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Node coordinates for the schematic.

    Ribbons are two-dimensional: the page draws each one as a line of dots at
    its own ``z0_nm``, never as a slab with thickness.
    """
    system = _build_system(spec)
    notes: List[str] = []
    _layer_note(system, notes)
    return {
        "schema": SCHEMA_VERSION,
        "n_ribbons": len(system.ribbons),
        "n_nodes_total": int(system.N),
        "dx_nm": _f(system.dx_nm),
        "eps_eff": _cx(np.array([system.eps_eff]))["re"][0],
        "ribbons": [
            {
                "W_nm": _f(r.W_nm),
                "x0_nm": _f(r.x0_nm),
                "z0_nm": _f(r.z0_nm),
                "EF_eV": _f(r.EF_ev),
                "gamma_eV": _f(r.gamma_ev),
                "beta": _f(r.beta),
                "T_K": _f(r.T_K),
                "L": int(np.size(x)),
                "h_nm": _f(h),
                "x_nodes_nm": _fl(x),
                "theta": _fl(th),
            }
            for r, x, th, h in zip(system.ribbons, system.x_nodes, system.theta, system.h)
        ],
        "layers": [
            {"eps_par": _cx(np.array([l.eps_par]))["re"][0], "d_nm": _f(l.d_nm)}
            for l in (system.layers or [])
        ],
        "notes": notes,
    }


def material(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The material response on its own: sigma(omega) and the dimensionless eta.

    No solve, no geometry beyond each ribbon's own width -- this is the input
    to every other calculation here, shown by itself. It is worth its own view
    because the whole formalism turns on one number: every resonance of every
    structure sits where ``eta(omega)`` meets a geometric eigenvalue
    ``eta_j``, so reading eta next to the modes tells you where the peaks will
    be before any of them is computed.

    Every ribbon carries its own E_F, gamma, beta, T and width, so ``ribbons``
    holds one entry each; the top-level keys repeat ribbon 1.
    """
    system = _build_system(spec)
    Es = _energies(spec)
    q = _f(_get(spec, "q_nm_inv", 0.0))
    notes: List[str] = []

    # One entry per ribbon.  E_F, gamma, beta, T and W are all per ribbon, so a
    # single curve taken from ribbon 1 was a picture of one ribbon standing in
    # for a structure that may hold five different ones.
    per: List[Dict[str, Any]] = []
    for j, rib in enumerate(system.ribbons):
        eta_p = np.empty(Es.size, dtype=np.complex128)
        eta_l = np.empty(Es.size, dtype=np.complex128)
        for i, E in enumerate(Es):
            eta_p[i], eta_l[i] = system.eta_pair(j, q, _f(E))
        entry: Dict[str, Any] = {
            "index": j,
            "W_nm": _f(rib.W_nm),
            "EF_eV": _f(rib.EF_ev),
            "gamma_eV": _f(rib.gamma_ev),
            "beta": _f(rib.beta),
            "T_K": _f(rib.T_K),
            "material": _sigma(spec, Es, rib),
            "eta_perp": _cx(eta_p),
            "eta_par": _cx(eta_l),
            "mu_eV": _f(drift_chemical_potential(rib.EF_ev, rib.beta, rib.T_K)),
        }
        if rib.beta:
            sp, sl = graphene_drift_sigma(
                Es, rib.EF_ev, rib.gamma_ev, max(abs(q), 1e-6), rib.beta, rib.T_K
            )
            entry["sigma_perp_nm_fs"] = _cx(np.asarray(sp))
            entry["sigma_par_nm_fs"] = _cx(np.asarray(sl))
        per.append(entry)

    r0 = system.ribbons[0]
    mat = per[0]["material"]
    beta = _f(r0.beta)

    out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "energies_eV": _fl(Es),
        "q_nm_inv": q,
        # The top-level keys are ribbon 1, kept so an older reader of this
        # payload still finds what it expects; `ribbons` is the whole picture.
        "material": mat,
        "eta_perp": per[0]["eta_perp"],
        "eta_par": per[0]["eta_par"],
        "W_nm": _f(r0.W_nm),
        "eps_eff": _f(np.real(system.eps_eff)),
        "mu_eV": per[0]["mu_eV"],
        "ribbons": per,
        "notes": notes
        + [
            "eta = i sigma / (omega W eps_eff), one curve per ribbon at this q. A "
            "resonance sits wherever Re eta(omega) meets a geometric eigenvalue "
            "eta_j, which is why the Modes view and this one are read together.",
            "eta carries the INTRABAND (Drude) conductivity only, warmed by each "
            "ribbon's own T. The sigma plotted above it also carries the interband "
            "term, so the two panels are not the same sigma above the Pauli edge.",
        ],
    }

    if beta:
        # With drift eta is a tensor, and which component a given condition wants
        # is a question the reference MATLAB gets wrong; the longitudinal one is
        # component 1 (settled 2026-11-02).
        out["sigma_perp_nm_fs"] = per[0]["sigma_perp_nm_fs"]
        out["sigma_par_nm_fs"] = per[0]["sigma_par_nm_fs"]
        out["notes"].append(
            "Drift splits sigma into a transverse and a LONGITUDINAL component, and "
            "eta with it. Drift is a skewed Fermi-Dirac distribution solved through "
            "Boltzmann, never a Galilean frame shift: it survives q -> 0 and breaks "
            "the +-q symmetry."
        )
        if q == 0.0:
            out["notes"].append(
                "The tensor is shown at |q| = 1e-6 nm^-1 rather than exactly zero, "
                "which is its q -> 0 limit to the digits plotted."
            )
    return out


def near_field(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The induced near field on an (x, z) grid, at one energy and one q.

    One solve, then the field evaluated at every grid point. The field diverges
    on the sheet itself, so points that land on a node come back non-finite and
    therefore as ``null`` -- a gap in the map rather than a hot spot, which is
    the honest rendering of a singularity.
    """
    system = _build_system(spec)
    q = _f(_get(spec, "q_nm_inv", 0.0))
    E_ev = _f(_get(spec, "E_ev", _get(spec, "energy_eV", 0.5)))
    if E_ev <= 0.0:
        raise WebSpecError("E_ev must be > 0 for a near-field map")

    notes: List[str] = []
    _layer_note(system, notes)
    _tune_vbar(system, np.array([q]), spec, notes)
    if system.layers and q == 0.0:
        raise WebSpecError(
            "q = 0 with a substrate: the reflected Coulomb integral is logarithmically "
            "divergent there. Use a small finite q."
        )

    nx = int(_get(spec, "nx", 160))
    nz = int(_get(spec, "nz", 96))
    if nx < 2 or nz < 2:
        raise WebSpecError("nx and nz must each be at least 2")
    if nx * nz > MAX_MAP_CELLS:
        raise WebSpecError(
            f"{nx} x {nz} = {nx * nz} field points exceeds the {MAX_MAP_CELLS}-point "
            f"ceiling. Coarsen the map."
        )

    # Default window: the structures plus a margin, and a symmetric z range, so
    # the map shows the field falling off on both sides of the sheet.
    xs_all = np.concatenate([np.asarray(x, dtype=float) for x in system.x_nodes])
    span = float(xs_all.max() - xs_all.min()) or 1.0
    pad = _f(_get(spec, "pad_nm", 0.6 * span))
    x_lo = _f(_get(spec, "x_min_nm", float(xs_all.min()) - pad))
    x_hi = _f(_get(spec, "x_max_nm", float(xs_all.max()) + pad))
    z_half = _f(_get(spec, "z_half_nm", 0.75 * span))
    xs = np.linspace(x_lo, x_hi, nx)
    zs = np.linspace(-z_half, z_half, nz)

    fn = _source_fn(spec, system, np.array([q]))
    lin = system.solve_linear(q, E_ev, fn(q, E_ev))
    rho = lin["rho"]
    dx = [float(h) * float(r.W_nm) for h, r in zip(system.h, system.ribbons)]

    # The Coulomb kernel carries 2|q| K_1(|q| s), which is 0 * inf at exactly
    # q = 0. The field is smooth in q, and |q| = 1e-6 nm^-1 reproduces the
    # q -> 0 limit to four digits (checked against 1e-3), so the limit is taken
    # numerically rather than refused.
    q_eval = q if q != 0.0 else 1e-6

    # Every component, not just the magnitude: the phase structure is the point
    # of looking at a near field, and Re E_x either side of a ribbon is what
    # makes a dipolar mode legible. Six real arrays rather than three complex
    # ones, because the JSON boundary splits complex into re/im anyway.
    comp = {k: np.empty((nx, nz), dtype=np.float64)
            for k in ("abs", "ex_re", "ex_im", "ey_re", "ey_im", "ez_re", "ez_im")}
    with np.errstate(all="ignore"):
        for i, x0 in enumerate(xs):
            for j, z0 in enumerate(zs):
                Ex, Ey, Ez = _obs.induced_field_q(q_eval, system.x_nodes, rho, (x0, z0), dx_nm=dx)
                comp["abs"][i, j] = float(np.sqrt(abs(Ex) ** 2 + abs(Ey) ** 2 + abs(Ez) ** 2))
                comp["ex_re"][i, j] = float(np.real(Ex))
                comp["ex_im"][i, j] = float(np.imag(Ex))
                comp["ey_re"][i, j] = float(np.real(Ey))
                comp["ey_im"][i, j] = float(np.imag(Ey))
                comp["ez_re"][i, j] = float(np.real(Ez))
                comp["ez_im"][i, j] = float(np.imag(Ez))

    src = _get(spec, "source", {}) or {}
    src_kind = str(src.get("kind", "plane_wave"))
    source_marker: Dict[str, Any] = {"kind": src_kind}
    if src_kind == "dipole":
        r0 = [_f(v) for v in src.get("r0_nm", (0.0, 0.0, 5.0))]
        source_marker["x_nm"] = r0[0]
        source_marker["z_nm"] = r0[2] if len(r0) > 2 else r0[1]
        # The moment's direction, so the map can draw the dipole it actually
        # solved rather than a generic dot: x, y and z behave nothing alike.
        p = [_f(v) for v in src.get("p", (1.0, 0.0, 0.0))]
        source_marker["p"] = p + [0.0] * (3 - len(p)) if len(p) < 3 else p[:3]
    elif src_kind == "electron":
        source_marker["b_nm"] = _f(src.get("b_nm", 5.0))
        source_marker["v_over_c"] = _f(src.get("v_over_c", 0.2))

    return {
        "schema": SCHEMA_VERSION,
        "xs_nm": _fl(xs),
        "zs_nm": _fl(zs),
        "map": _rows(comp["abs"]),
        "ex_re": _rows(comp["ex_re"]),
        "ex_im": _rows(comp["ex_im"]),
        "ey_re": _rows(comp["ey_re"]),
        "ey_im": _rows(comp["ey_im"]),
        "ez_re": _rows(comp["ez_re"]),
        "ez_im": _rows(comp["ez_im"]),
        "q_nm_inv": q,
        "E_ev": E_ev,
        "source_marker": source_marker,
        "ribbons": [
            {
                "x_nodes_nm": _fl(x),
                "z0_nm": _f(r.z0_nm),
                "x_min_nm": _f(np.min(x)),
                "x_max_nm": _f(np.max(x)),
            }
            for x, r in zip(system.x_nodes, system.ribbons)
        ],
        "notes": notes,
    }


def eigenmodes(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Level 2 at a single q: geometric eigenvalues, overlaps, mode profiles.

    These eigenvalues are purely geometric -- ``V @ D_geom`` with the material
    stripped out -- so they carry no information about sigma's normalisation.
    The bands come from them through ``eta(omega) = eta_j``.
    """
    system = _build_system(spec)
    q = _f(_get(spec, "q_nm_inv", 0.0))
    n_modes = int(_get(spec, "n_modes", 6))
    notes: List[str] = []
    _layer_note(system, notes)

    _guard_cost("eigenmodes", spec, system, np.array([q]), np.array([1.0]))
    if system.layers:
        _guard_q(system, np.array([q]))
        _tune_vbar(system, np.array([q]), spec, notes)
        E_ev = _f(_get(spec, "E_ev", 0.5))
        eta, phi, zeta = system.eigenmodes_screened(q, E_ev)
        notes.append(f"Screened eigenmodes at E = {E_ev} eV: eta_j depends on energy with a stack.")
    else:
        eta, phi, zeta = system.eigenmodes(q)

    from .solver import bright_mode_mask

    eta_all = np.asarray(eta)
    zeta_all = np.asarray(zeta)
    phi_all = np.asarray(phi)
    bright_all = np.asarray(bright_mode_mask(zeta_all))

    # DEFAULT TO BRIGHT MODES, and not for tidiness.  ``eigenmodes`` returns
    # modes in order of descending |eta|, and the operator has a null space --
    # the charge-neutrality mode, with lambda numerically zero and therefore
    # eta = 1/lambda enormous (-2.2e12 at q = 0, -0.89 at q = 0.05).  It is an
    # artefact of the operator, not a resonance, and it sorts FIRST.  A page that
    # took mode 0 as the fundamental would print -2.2e12 as its headline number.
    # The dipole overlap zeta is what distinguishes them: the null mode and the
    # antisymmetric dark modes have |zeta| at the 1e-8..1e-16 level against 0.945
    # for the fundamental.  The frozen acceptance eigenvalues (-0.0687, -0.0184,
    # -0.0107) are the BRIGHT ones, by that same filter.
    bright_only = bool(_get(spec, "bright_only", True))
    idx = np.flatnonzero(bright_all) if bright_only else np.arange(eta_all.size)
    if idx.size == 0:
        raise WebSpecError(
            "No bright mode found: every eigenmode has a vanishing dipole overlap, so "
            "none of them couples to a plane wave. Pass bright_only=false to see the "
            "dark spectrum anyway."
        )
    n_modes = max(1, min(n_modes, int(idx.size)))
    sel = idx[:n_modes]

    W0 = _f(system.ribbons[0].W_nm)
    EF0 = _f(system.ribbons[0].EF_ev)
    E_drude = _disp.drude_band_energy(eta_all[sel], W0, EF0, eps_bar=system.eps_eff)

    n_hidden = int(eta_all.size - idx.size)
    if bright_only and n_hidden:
        notes.append(
            "%d dark modes hidden: |zeta| below the dipole floor, so they do not couple "
            "to a plane wave. One of them is the charge-neutrality null mode, whose eta "
            "diverges (%.3g here) because its eigenvalue is numerically zero -- an "
            "artefact of the operator's null space, not a resonance. Pass "
            "bright_only=false to see them." % (n_hidden, float(np.real(eta_all[0])))
        )

    return {
        "schema": SCHEMA_VERSION,
        "q_nm_inv": q,
        "bright_only": bright_only,
        "mode_index": [int(i) for i in sel],
        "eta": _cx(eta_all[sel]),
        "zeta": _cx(zeta_all[sel]),
        "bright": [bool(b) for b in bright_all[sel]],
        "E_drude_eV": _fl(E_drude),
        # Split PER RIBBON, so profiles[m][j] lines up with x_nodes_nm[j].
        # Flat, the array is the whole system's N nodes while x_nodes_nm is a
        # list per ribbon, and a caller cannot tell which x belongs to which
        # value: the page was plotting one mode against another ribbon's
        # coordinates and nothing said so.
        "profiles": [
            [_fl(np.real(phi_all[sl, int(m)])) for sl in system.slices] for m in sel
        ],
        "x_nodes_nm": [_fl(x) for x in system.x_nodes],
        "n_modes_total": int(eta_all.size),
        "notes": notes
        + [
            "eta_j is geometric: the same numbers come out under any conductivity "
            "convention, so they cannot be used to pin one.",
            "E_drude_eV assumes a single Drude ribbon of width W = %g nm at E_F = %g eV; "
            "it is the closed form, not a solve." % (W0, EF0),
        ],
    }


def _band_notes(bs: Any, notes: List[str]) -> None:
    """Explain the two things a tracked band plot shows that a reader may not expect.

    One: each curve is ONE branch over the whole sweep, so where a branch has
    no resonance its curve breaks instead of every curve above it sliding down
    a slot.  Two: which branch broke, below which ``|q|``, AND WHICH OF THE TWO
    REASONS applies -- because the one that routinely breaks is the ribbon's
    longitudinal (wire) branch, whose energy goes to zero as ``q -> 0``, and it
    can run out for either reason.  Usually it simply leaves the bottom of the
    plotted window.  But widen the window all the way down and it still stops,
    at a smaller ``|q|``: once its energy falls below ``gamma`` the mode is
    damped out and there is no resonance left to find at all.  Saying "below
    the window" in that second case would be a plain falsehood, and the reader
    would widen the window again and get nothing.
    """
    E = np.atleast_2d(np.asarray(bs.E_ev, dtype=np.float64))
    if E.size == 0:
        return
    notes.append(
        "Bands are followed across q by eigenvector overlap, so each curve is one "
        "branch throughout. A branch with no resonance inside the energy window "
        "breaks there rather than shifting the bands above it."
    )
    qs = np.abs(np.asarray(bs.qs, dtype=np.float64))
    zeta = np.abs(np.atleast_2d(np.asarray(bs.zeta)))
    Ed = np.atleast_2d(np.asarray(bs.E_drude_ev, dtype=np.float64))
    for b in range(E.shape[1]):
        col = np.isfinite(E[:, b])
        if col.all() or not col.any():
            continue
        live, dead = qs[col], qs[~col]
        lo, hi = float(np.min(live)), float(np.max(live))

        # WHICH END ran out.  The first version of this assumed every gap was
        # at small |q|, because that is where the longitudinal branch leaves,
        # and then described a band whose gap is at LARGE |q| -- its energy
        # climbing out of the top of the window -- as stopping at "|q| = 0" and
        # being "under the bottom".  Four wrong statements in one sentence, all
        # from one unexamined assumption.
        at_low = bool(np.any(dead < lo))
        at_high = bool(np.any(dead > hi))
        if at_low and not at_high:
            edge, past, where = lo, (~col) & (qs < lo), f"below |q| = {lo:.3g} / nm"
        elif at_high and not at_low:
            edge, past, where = hi, (~col) & (qs > hi), f"above |q| = {hi:.3g} / nm"
        elif at_low and at_high:
            edge, past, where = lo, ~col, (
                f"outside {lo:.3g} <= |q| <= {hi:.3g} / nm"
            )
        else:
            # Holes strictly inside the live range: nothing tidy to say about
            # an end, so say where without pretending to a direction.
            edge, past, where = lo, ~col, "at some q inside the swept range"

        # WHY, read off the data rather than guessed from a threshold.
        # ``E_drude_ev`` is the closed-form lossless root and
        # ``drude_band_energy`` returns NaN exactly when there is none, which
        # for a branch of negative eta means the lossless energy has dropped
        # below gamma: no resonance at all, and widening the window gains
        # nothing.  When it IS finite the resonance exists and the window is
        # what cut it off -- and then its value says which side, which is the
        # part that used to be assumed.
        damped = bool(np.all(~np.isfinite(Ed[past, b]))) if past.any() else False
        if damped:
            why = (
                "there its energy has fallen under gamma, so the mode is damped "
                "out and there is no resonance left to plot -- widening the "
                "energy window does not bring it back"
            )
        else:
            finite_past = Ed[past, b][np.isfinite(Ed[past, b])]
            above = bool(finite_past.size and np.median(finite_past) > float(np.nanmax(E[col, b])))
            why = (
                "there it is above the top of the energy window"
                if above
                else "there it is under the bottom of the energy window"
            )

        # WHICH BRANCH.  Dark is NOT the same as longitudinal: the even-parity
        # ribbon modes are dark too, and calling one of those "the wire branch"
        # was the other half of the wrong sentence.  The longitudinal branch is
        # the one that runs to zero energy at q -> 0, so require that as well:
        # it is dark, its gap is at the small-|q| end, and its energy there is
        # the lowest it reaches.
        # RELATIVE, not a bare 1e-6. A dark mode's |zeta| is not a small
        # physical number, it is rounding: at q = 0 the normalisation divides
        # by sqrt(|u^T D u|), which is identically zero for the uniform
        # potential, so what is stored is sqrt(machine-epsilon) scale and it
        # grows with the ribbon. Measured for the longitudinal mode at q = 0:
        # 9.3e-8 at W = 10 nm L = 121, 1.26e-7 at L = 301, and 1.095e-6 at
        # W = 100 nm L = 301 -- over the bare threshold, so moving the node
        # count from 301 to 601 (2.0e-7) flipped this sentence's claim about
        # the physics, on one machine, with no BLAS change needed at all.
        # Against the brightest band in the same row it is 1e-7 either way.
        bright_scale = float(np.nanmax(zeta[col, :])) if zeta.size else 0.0
        dark = (
            bool(np.all(zeta[col, b] < 1e-6 * bright_scale))
            if zeta.shape[1] > b and bright_scale > 0.0
            else False
        )
        falls_to_zero = at_low and not at_high and float(
            E[col, b][np.argmin(live)]
        ) <= float(np.nanmin(E[col, b])) + 1e-12
        what = (
            "the longitudinal (wire) branch -- dark to a transverse field, and its "
            "energy goes to zero as q -> 0"
            if dark and falls_to_zero
            else ("dark to a transverse field" if dark else "a transverse ribbon band")
        )
        notes.append(
            f"Band {b + 1} is {what}. Its curve runs over "
            f"{lo:.3g} <= |q| <= {hi:.3g} / nm and breaks {where}, where "
            f"E = {float(E[col, b][np.argmin(np.abs(live - edge))]):.3g} eV: {why}."
        )


def bands(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Level 2 over a q grid: band energies, widths, brightness."""
    system = _build_system(spec)
    qs = _qs(spec)
    n_bands = int(_get(spec, "n_bands", 4))
    notes: List[str] = []
    _layer_note(system, notes)

    # A positive bottom to the search window, said in words rather than as the
    # library's own contract.  ``ribbon_bands`` raises "need 0 < E_min < E_max,
    # got (0.0, 1.0)" -- true, and useless to a reader who typed 0 into an
    # energy range, which is a perfectly reasonable thing to want: the
    # longitudinal branch ends exactly at zero, so the AXIS may well start
    # there.  What cannot start there is the model.
    E_lo = _f(_get(spec, "E_min_eV", 1e-3))
    E_hi = _f(_get(spec, "E_max_eV", 10.0))
    if not 0.0 < E_lo < E_hi:
        raise WebSpecError(
            f"The band search needs an energy window with a strictly positive bottom, "
            f"and got {E_lo:g} to {E_hi:g} eV. eta = i sigma / (omega W eps_eff) "
            "diverges as 1/omega, so E = 0 is a singularity of the model rather than "
            "merely a small number. Set the range minimum above zero -- 1e-4 eV is "
            "already far below any graphene plasmon. The plotted axis is free to "
            "start at zero; it is only what gets solved that cannot."
        )

    screened = bool(system.layers)
    if any(r.beta for r in system.ribbons):
        raise WebSpecError(
            "Level 2 bands need beta = 0. With drift, eta becomes a tensor "
            "(eta_perp != eta_par), so M is no longer eta(E) times one geometric "
            "operator and a single eigenvalue spectrum has no resonance condition to "
            "invert. Use the LDOS map (Level 3), which reads the bands off the peaks "
            "and shows their weight as well -- which matters, because drift both "
            "shifts and quenches them."
        )
    _guard_q(system, qs)
    _tune_vbar(system, qs, spec, notes)
    _guard_cost("bands", spec, system, qs, np.array([1.0]))
    bs = _disp.ribbon_bands(
        system,
        qs,
        n_bands=n_bands,
        bright_only=bool(_get(spec, "bright_only", False)),
        screened=screened,
        # Always, because this entry point exists to draw a curve over q.  The
        # library default re-ranks the modes at every point, which is right for
        # one q and draws false steps across a sweep: see ribbon_bands' `track`.
        track=True,
        E_min=_f(_get(spec, "E_min_eV", 1e-3)),
        E_max=_f(_get(spec, "E_max_eV", 10.0)),
    )
    _band_notes(bs, notes)
    if screened:
        notes.append("Bands solved with the screened eigenproblem, root-found in energy per q.")

    return {
        "schema": SCHEMA_VERSION,
        "qs_nm_inv": _fl(bs.qs),
        "n_bands": int(np.asarray(bs.E_ev).shape[1]) if np.asarray(bs.E_ev).ndim == 2 else 1,
        "E_eV": _rows(bs.E_ev),
        "fwhm_eV": _rows(bs.fwhm_ev),
        "E_drude_eV": _rows(bs.E_drude_ev),
        "eta_re": _rows(np.real(bs.eta)),
        "eta_im": _rows(np.imag(bs.eta)),
        "weight": _rows(bs.weight),
        "bright": [[bool(b) for b in row] for row in np.atleast_2d(np.asarray(bs.bright))],
        "notes": notes,
    }


def sheet_dispersion(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Level 1: the extended-sheet reflection over (Q, E).  No ribbon at all.

    The cheapest rung, and the one that isolates the drift conductivity from
    every other moving part.  With drift the +-Q branches split, which is the
    whole point of the map.
    """
    mat = _get(spec, "material", {})
    EF = _f(mat.get("EF_eV", 0.5))
    gam = _f(mat.get("gamma_eV", 0.02))
    beta = _f(mat.get("beta", _get(spec, "beta", 0.0)))
    T_K = _f(_get(spec, "T_K", mat.get("T_K", 0.0)))
    eps_a = complex(_get(spec, "eps_a", 1.0))
    eps_b = complex(_get(spec, "eps_b", 1.0))

    Qs = _qs(spec)
    Es = _energies(spec)
    cells = int(Qs.size * Es.size)
    if cells > MAX_MAP_CELLS:
        raise WebSpecError(
            f"{Qs.size} x {Es.size} = {cells} cells exceeds the {MAX_MAP_CELLS}-cell ceiling"
        )

    if np.any(Qs == 0.0):
        raise WebSpecError(
            "Q = 0 is not a point of the extended-sheet map: r(Q) carries 1/|Q| and the "
            "sheet plasmon has no q = 0 limit. Start the grid at a small finite Q."
        )
    QQ, EE = np.meshgrid(Qs, Es, indexing="ij")
    r = _disp.sheet_reflection(QQ, EE, EF, gam, beta, eps_a, eps_b, T_K=T_K)
    sig = _disp.sheet_sigma_longitudinal(QQ, EE, EF, gam, beta, T_K=T_K)

    notes = [
        "Level 1 uses the LONGITUDINAL conductivity (component 1 of the drift tensor). "
        "The reference script passes the transverse one, which shifts the pole 1-3%."
    ]
    if beta:
        notes.append(
            "Drift is a skewed Fermi-Dirac distribution solved through Boltzmann, not a "
            "Galilean frame shift: it survives Q -> 0 and breaks +-Q symmetry."
        )
        notes.append("mu(drift) = %.6f eV at this beta and T." % drift_chemical_potential(EF, beta, T_K))

    out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "Qs_nm_inv": _fl(Qs),
        "energies_eV": _fl(Es),
        "im_r": _rows(np.imag(r)),
        "sigma_re": _rows(np.real(sig)),
        "sigma_im": _rows(np.imag(sig)),
        "notes": notes,
    }
    if beta and np.any(Qs < 0) and np.any(Qs > 0):
        qsym, asym = _disp.pm_asymmetry(Qs, np.imag(r), normalise=True)
        out["asymmetry_Q_nm_inv"] = _fl(qsym)
        out["asymmetry"] = _rows(asym)
    return out


def linear_spectrum(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Polarizability per unit length vs energy, at one or more q."""
    system = _build_system(spec)
    Es = _energies(spec)
    qs = _qs(spec)
    notes: List[str] = []
    _layer_note(system, notes)

    _guard_q(system, qs)
    _tune_vbar(system, qs, spec, notes)
    _guard_cost("linear_spectrum", spec, system, qs, Es)
    fn = _source_fn(spec, system, qs)
    obstruction = system.fast_path_obstruction()
    fast = obstruction is None
    if not fast:
        n_solves = int(qs.size * Es.size)
        # Screened only, to match cost_estimate: a spec the button allowed must
        # not then be refused by the solver it allowed it for.
        if system.layers and n_solves > MAX_SLOW_SOLVES:
            raise WebSpecError(
                f"{n_solves} full solves needed ({obstruction}) against a "
                f"{MAX_SLOW_SOLVES}-solve ceiling, and with a substrate the time "
                f"estimate runs 4x to 16x low. Reduce the energy or q grid, or drop "
                f"the substrate."
            )
        notes.append(
            f"Fast path unavailable ({obstruction}); every (q, E) point is a full solve."
        )

    alpha = system.sweep(qs, Es, fn, fast_path=fast)

    out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "qs_nm_inv": _fl(qs),
        "energies_eV": _fl(Es),
        "alpha_re": _rows(np.real(alpha)),
        "alpha_im": _rows(np.imag(alpha)),
        "extinction_arb": _rows(np.imag(alpha)),
        "material": _sigma(spec, Es, system.ribbons[0]),
        "fast_path": bool(fast),
        "notes": notes,
    }
    out.update(_extinction(system, Es, alpha))
    if qs.size > 1 and np.any(qs < 0) and np.any(qs > 0):
        qsym, asym = _disp.pm_asymmetry(qs, np.imag(alpha), normalise=True)
        out["asymmetry_q_nm_inv"] = _fl(qsym)
        out["asymmetry"] = _rows(asym)
    return out


def eels_spectrum(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Swift-electron loss spectrum: aloof, parallel trajectories only.

    A zero-thickness sheet has no interior, so there is no bulk-loss channel and
    no begrenzung term, and the external field diverges where a trajectory meets
    a node.  The library refuses a trajectory that touches the plane or enters a
    layer; this surfaces that refusal as a message rather than a traceback.
    """
    system = _build_system(spec)
    src = dict(_get(spec, "source", {}))
    src.setdefault("kind", "electron")
    if src["kind"] != "electron":
        raise WebSpecError("eels_spectrum needs source.kind == 'electron'")
    spec = dict(spec, source=src)

    Es = _energies(spec)
    qs = _qs(spec)
    notes: List[str] = []
    _layer_note(system, notes)

    b_nm = _f(src.get("b_nm", 5.0))
    v_over_c = _f(src.get("v_over_c", 0.2))
    ok, reason = _src.electron_trajectory_is_valid(
        b_nm, np.array([r.z0_nm for r in system.ribbons]), layers=system.layers
    )
    if not ok:
        raise WebSpecError(f"Trajectory is not aloof: {reason}")

    # RECIPROCITY, NOT THE DIPOLE.  The loss is the work the electron does
    # against the field it induces, which by reciprocity is the overlap of the
    # induced charge with the potential that created it:
    #
    #     Gamma(q, E)  ~  -Im sum_j W_j h_j sum_i rho_j(x_i) conj(phi_ext_j(x_i))
    #
    # That is positive definite for a passive sheet, and it peaks on the
    # resonance.  What this function reported until 2026-11-05 was
    # ``linear_spectrum``'s alpha = Im sum_j int x rho_j dx -- the induced
    # DIPOLE under the electron's potential.  Under a plane wave the two
    # coincide, because there phi_ext = -E0 x makes the overlap the dipole; a
    # user reported negative loss on a bare ribbon, and measured on a 10 nm
    # ribbon the old quantity swung from +2.88 to -2.77 across the resonance
    # and was negative over 54 % of the grid -- a dispersive lineshape, the
    # signature of the wrong phase reference.  The same case now runs
    # 0.0017 to 0.179, nowhere negative, peaking at 0.570 eV.
    _guard_q(system, qs)
    _tune_vbar(system, qs, spec, notes)
    _guard_cost("eels_spectrum", spec, system, qs, Es)
    fn = _source_fn(spec, system, qs)

    loss = np.empty((qs.size, Es.size), dtype=np.float64)
    for iq, q in enumerate(qs):
        for iE, E in enumerate(Es):
            phis = fn(float(q), float(E))
            if not isinstance(phis, (list, tuple)):
                phis = [phis]
            sol = system.solve_linear(float(q), float(E), phis)
            acc = 0.0 + 0.0j
            for j, rho_j in enumerate(sol["rho"]):
                acc += (
                    system.ribbons[j].W_nm
                    * system.h[j]
                    * np.sum(np.asarray(rho_j) * np.conj(np.asarray(phis[j])))
                )
            loss[iq, iE] = -float(np.imag(acc))

    # Gamma(omega) = int dk_par Gamma(k_par, omega) -- paper 430's Methods,
    # the line under Eq. (G1gena). A plain dk measure, no 1/2pi: that factor
    # already sits in the (e / 2 pi^2 hbar omega) prefactor of Eq. (G1genb).
    # Over a one-point q grid there is nothing to integrate, so the spectrum
    # is that single cut and says so in a note.
    if qs.size > 1:
        integrated = _TRAPZ(loss, qs, axis=0)
        # The integrand piles up at small |q| -- the electron's potential
        # carries K_0(Lam b) with Lam -> |q| there -- so a q grid that starts
        # too high integrates the tail and misses the plasmon entirely. On a
        # 10 nm ribbon at E_F = 0.5 eV, q_min = 0.02 nm^-1 puts the peak of
        # Gamma(omega) on the LOWEST energy sampled; q_min = 0.004 moves it to
        # 0.580 eV, which is the plasmon the single-q spectrum also finds. That
        # is not a small shift, so it is reported rather than left to be found.
        # Halve the resolution and see whether the answer moves. Measured on a
        # 10 nm ribbon at E_F = 0.5 eV, q in [0.004, 0.3]: 20 points put the
        # peak of Gamma(omega) at 0.300 eV, 40 points at 0.580 eV -- the
        # plasmon the single-q spectrum also finds. A peak that moves by that
        # much with the grid is not a result, and nothing on the page would
        # have said so. Costs nothing: no new solves, just the same rows.
        # The thing a reader takes off this curve is WHERE it peaks, so that is
        # what is tested, not the magnitude: on the case above the magnitude
        # drift stayed under 8 % while the peak moved 0.300 -> 0.580 eV.
        if qs.size >= 5:
            coarse = _TRAPZ(loss[::2], qs[::2], axis=0)
            e_full = float(Es[int(np.argmax(integrated))])
            e_half = float(Es[int(np.argmax(coarse))])
            step = float(Es[1] - Es[0]) if Es.size > 1 else 0.0
            if abs(e_full - e_half) > 2.0 * step:
                notes.append(
                    "The q grid is too coarse to locate this peak: halving its "
                    "resolution moves Gamma(omega) from %.3f to %.3f eV. Add q points "
                    "until the two agree." % (e_full, e_half)
                )

        # Resolution is not the only way to get this wrong, and the other way
        # is NOT detected here. q min truncates the integral as well: on the
        # case above, q in [0.02, 0.3] with 20 points reports a converged peak
        # at 0.320 eV, and the same grid from 0.004 reports 0.580 eV, the
        # plasmon. Three cheap tests for that were tried and all three fired on
        # converged grids -- the integrand being largest at the innermost |q|
        # is normal, not evidence of truncation, because that peak is
        # integrable. Rather than ship a fourth guess, the caveat is stated
        # plainly and the reader is told what to vary.
        notes.append(
            "q min truncates this integral as well as resolves it: the loss grows "
            "towards q = 0. Measured on a 10 nm ribbon at E_F = 0.5 eV, q from 0.02 "
            "nm^-1 puts the peak at 0.320 eV and q from 0.004 puts it at 0.580 eV, "
            "the plasmon. Lower q min until Gamma(omega) stops moving; nothing here "
            "detects it for you."
        )
    else:
        integrated = loss[0]
        notes.append(
            "Gamma(omega) is a single q cut, not an integral: the q grid has one "
            "point. Sweep q to integrate over it as the definition asks."
        )

    res: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "qs_nm_inv": _fl(qs),
        "energies_eV": _fl(Es),
        "loss_arb": _rows(loss),
        "loss_integrated": _rows(np.atleast_2d(integrated)),
        "material": _sigma(spec, Es, system.ribbons[0]),
        "gamma_lorentz": _f(_src.lorentz_gamma(v_over_c)),
        "b_nm": b_nm,
        "v_over_c": v_over_c,
        "notes": notes
        + [
            "The external potential carries gamma_L^0; gamma_L enters only through Lam. "
            "The longitudinal FIELD carries gamma_L^-2, via A_x = (v/c) phi (notes 8.3).",
            "Aloof and parallel only -- there is no penetrating-trajectory formalism here.",
            "Loss is -Im of the overlap of the induced charge with the electron's own "
            "potential: the work the electron does against the field it induces, which "
            "is what paper 430 Eq. (G1gena) integrates along the trajectory. Positive "
            "by construction.",
            "This is get_EELS of the project's own reference, "
            "sanity-checks/efd-matlab-parity/matlab-ref/Joel/GNR_phi_EELS.m (its "
            "Eq. 46), which the parity README lists as having no library counterpart. "
            "Algebraically its val is exactly -(1/2pi^2) Int rho conj(phi_ext) dx, the "
            "overlap computed here: the reference's own phi_ext is the same "
            "-2pi/(v Lam) exp(i w x/v) exp(-Lam|z-b|) form, already matched to this "
            "library at 1.14e-13. So the SHAPE is the reference's, exactly.",
            "ABSOLUTE SCALE still open. The constant 1/(2 pi^2) is the reference's, "
            "but its phi_ext is atomic-unit (e^2 = 1) and non-relativistic while this "
            "library carries e^2 and gamma_L in Lam, and rho here is rho_phys/eps_eff "
            "-- so the two differ by a unit factor that has to be measured in the "
            "parity harness rather than read off. Four constants have been guessed on "
            "this page already and all four were wrong; this one is left to the "
            "harness. Ratios and peak positions are unaffected.",
        ],
    }
    return res


def ldos(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Level 3: the LDOS map over (q, E), with peak tracks and +-q asymmetry.

    This is how paper 456 extracts its dispersions, and the only level that
    shows mode *weight* as well as position -- which matters, because drift both
    shifts and quenches the bands.
    """
    system = _build_system(spec)
    qs = _qs(spec)
    Es = _energies(spec)
    src = _get(spec, "source", {})
    r0 = [_f(v) for v in src.get("r0_nm", (0.0, 0.0, 5.0))]
    p = [complex(v) for v in src.get("p", (1.0, 0.0, 0.0))]
    quantity = str(_get(spec, "quantity", "ldos"))

    cells = int(qs.size * Es.size)
    if cells > MAX_MAP_CELLS:
        raise WebSpecError(
            f"{qs.size} x {Es.size} = {cells} cells exceeds the {MAX_MAP_CELLS}-cell ceiling"
        )

    notes: List[str] = []
    _layer_note(system, notes)
    if np.any(np.asarray(qs, dtype=float) == 0.0):
        raise WebSpecError(
            "The LDOS map is dipole-driven and a point dipole has no q = 0 Fourier "
            "component, so q = 0 cannot be a column of the map. Start the q grid at a "
            "small finite value."
        )
    _guard_q(system, qs)
    _tune_vbar(system, qs, spec, notes)
    _guard_cost("ldos", spec, system, qs, Es)
    obstruction = system.fast_path_obstruction()
    fast = obstruction is None
    if not fast:
        # Screened only, to match cost_estimate; see MAX_SLOW_SOLVES.
        if system.layers and cells > MAX_SLOW_SOLVES:
            raise WebSpecError(
                f"{cells} full solves needed ({obstruction}) against a "
                f"{MAX_SLOW_SOLVES}-solve ceiling, and with a substrate the time "
                f"estimate runs 4x to 16x low. Shrink the map, or compute bands "
                f"(Level 2) instead, which costs one eigenproblem per q."
            )
        notes.append(f"Fast path unavailable ({obstruction}); every cell is a full solve.")

    D = _disp.ldos_map(system, qs, Es, r0, p, quantity=quantity, fast_path=fast)

    # FOLLOWED across q, not ranked within each column.  This loop used to call
    # `track_peaks` once per row, which sorts that row by descending height --
    # so slot k meant "the k-th tallest peak HERE" and swapped branches wherever
    # two of them swapped height.  The curve drawn through slot 0 stepped
    # 0.371 eV in a single column on the page's own default panel, a quarter of
    # the plotted window, against a typical real step of 0.05 eV.  Feeding the
    # whole map to `follow_peaks` instead takes that to 0.048 eV.
    n_peaks = int(_get(spec, "n_peaks", 3))
    pe, pw = _disp.follow_peaks(qs, Es, np.asarray(D), n_peaks=n_peaks)
    peak_E: List[List[float]] = [_fl(row) for row in pe]
    peak_w: List[List[float]] = [_fl(row) for row in pw]

    out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "qs_nm_inv": _fl(qs),
        "energies_eV": _fl(Es),
        "quantity": quantity,
        "map": _rows(D),
        "peak_E_eV": peak_E,
        "peak_weight": peak_w,
        "fast_path": bool(fast),
        "notes": notes
        + [
            "The LDOS is in ARBITRARY UNITS. Its 2 pi^2 |p|^2 normalisation has no "
            "counterpart in any reference code, although the induced field it multiplies "
            "agrees with one to 7e-14. Read positions and relative weights, not absolutes.",
            "Every source node is taken at z = 0, so ribbons at different z0_nm are not "
            "represented in the map.",
        ],
    }
    if qs.size > 1 and np.any(qs < 0) and np.any(qs > 0):
        qsym, asym = _disp.pm_asymmetry(qs, D, normalise=True)
        out["asymmetry_q_nm_inv"] = _fl(qsym)
        out["asymmetry"] = _rows(asym)
        out["dissymmetry"] = [_f(_obs.dissymmetry(qs, D[:, k])) for k in range(Es.size)]
        out["notes"].append(
            "dissymmetry g(E) is exactly zero at beta = 0 by symmetry; a nonzero value "
            "there would be a numerical artefact, not physics."
        )
    return out


def nonlinear_spectrum(spec: Dict[str, Any]) -> Dict[str, Any]:
    """SHG and THG polarizability spectra at one q.

    Each order is screened at its *own* harmonic.  The cascade carries
    A+B+C+D with D weighted once (notes 9).
    """
    system = _build_system(spec)
    Es = _energies(spec, cap=240)
    q = _f(_get(spec, "q_nm_inv", 0.0))
    orders = list(_get(spec, "orders", ["shg", "thg"]))
    cascade = bool(_get(spec, "cascade", True))
    notes: List[str] = []
    _layer_note(system, notes)

    if any(r.beta for r in system.ribbons):
        raise WebSpecError(
            "Nonlinear and drift have never been combined: the SHG and THG source currents "
            "assume an isotropic sigma. Set beta = 0, or compute the linear response."
        )

    _guard_q(system, np.array([q]))
    _tune_vbar(system, np.array([q]), spec, notes)
    _guard_cost("nonlinear_spectrum", spec, system, np.array([q]), Es)
    fn = _source_fn(spec, system, np.array([q]))
    # The linear polarizability at the fundamental is already in hand at every
    # energy -- every harmonic is driven by that same solution -- so returning
    # it costs nothing and gives the view its alpha^(1,1).
    a1: List[complex] = []
    a2: List[complex] = []
    a3: List[complex] = []
    for E in Es:
        lin = system.solve_linear(q, _f(E), fn(q, _f(E)))
        a1.append(complex(system.dipole_from_rho(lin["rho"])))
        if "shg" in orders:
            a2.append(complex(_nl.solve_shg(system, q, _f(E), lin["phi"])["alpha"]))
        if "thg" in orders:
            a3.append(
                complex(_nl.solve_thg(system, q, _f(E), lin["phi"], cascade=cascade)["alpha"])
            )

    out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "q_nm_inv": q,
        "energies_eV": _fl(Es),
        "cascade": cascade,
        "notes": notes
        + [
            "sigma^(3,3) and the Kerr term here are the INTRABAND third order only; against "
            "a full-RPA sigma3 the ratio runs from 4e-4 to 76 depending on where you look.",
            "At q = 0 in one dimension the three second-order tensor structures collapse to "
            "A + B + 2C; at finite q all three are needed.",
        ],
    }
    out["alpha_linear"] = _cx(np.asarray(a1))
    if a2:
        out["alpha_shg"] = _cx(np.asarray(a2))
    if a3:
        out["alpha_thg"] = _cx(np.asarray(a3))
    return out


# --------------------------------------------------------------------------- #
# Cost and dispatch
# --------------------------------------------------------------------------- #


#: Cost model, ``scale * (c2 N^2 + c3 N^3)`` seconds for each of the two units of
#: work this library spends time in.  The ``N^2`` term is the Coulomb kernel --
#: one special-function evaluation per node pair -- and it dominates at the sizes
#: a browser will use; ``N^3`` is the linear algebra and takes over above
#: ``N ~ 700``.
#:
#: The coefficients below are MEASURED, not guessed.  An earlier version of this
#: module carried guessed ones and under-reported a 10 s calculation as 0.5 s,
#: which defeats the entire point of warning before a click.  They come from
#: timing this library on native CPython 3.11 at N = 121, 301, 365, 601, 730 and
#: 801; the fit reproduces the measured per-solve cost at N = 150, 365 and 730 to
#: better than 20 %, which is all an order-of-magnitude warning needs.
#:
#: ``scale`` exists because the browser is NOT this machine: Pyodide's WASM numpy
#: runs the same work slower by a factor that depends on the device and the
#: build.  :func:`calibrate` measures one cheap point and sets ``scale`` from the
#: ratio, which corrects the LEVEL while keeping the SHAPE (the c3/c2 ratio, a
#: property of the BLAS, not of the host) from the native fit.  Refitting both
#: coefficients from two cheap browser points instead would give a worse
#: extrapolation to the large N that actually hurts.
_COST: Dict[str, Any] = {
    "per_q_c2": 1.10e-6,
    "per_q_c3": 5.02e-9,
    "per_q_scale": 1.0,
    "per_solve_c2": 9.14e-8,
    "per_solve_c3": 8.89e-11,
    "per_solve_scale": 1.0,
    # Per (q, E) cell on top of the per-q and per-solve work: evaluating the
    # source, the eigen expansion and the induced field over the nodes, all O(N).
    # It is small per cell and it DOMINATES a map -- the LDOS default is 5120
    # cells, and leaving this term out under-priced it eightfold (0.8 s estimated
    # against 6.4 s measured), which is the one module most able to lock a tab.
    "per_cell_c1": 9.0e-6,
    "per_cell_scale": 1.0,
    "source": "measured on native CPython, NOT on this runtime",
    "calibrated": False,
}

#: Refuse anything the cost model puts beyond this.  Counts alone are not
#: enough: 300 full solves is harmless at 150 nodes and five minutes at 2000, so
#: the ceiling that matters is a time, not a tally.
MAX_ESTIMATED_SECONDS = 120.0


def _fit_two_point(n1: int, t1: float, n2: int, t2: float) -> tuple:
    """Solve ``t = c2 n^2 + c3 n^3`` through two measurements.

    Returns ``(c2, c3)``, both clamped at zero: timing noise can easily make the
    implied c3 negative, and a negative coefficient would turn the estimate into
    a decreasing function of N, which is worse than crude.  Kept available for
    anyone who wants a full refit, though :func:`calibrate` prefers scaling.
    """
    a11, a12 = float(n1) ** 2, float(n1) ** 3
    a21, a22 = float(n2) ** 2, float(n2) ** 3
    det = a11 * a22 - a12 * a21
    if det == 0.0:
        return (t1 / a11 if a11 else 0.0, 0.0)
    c2 = (t1 * a22 - a12 * t2) / det
    c3 = (a11 * t2 - t1 * a21) / det
    return (max(c2, 0.0), max(c3, 0.0))


def _unit_cost(prefix: str, N: int) -> float:
    """Seconds for one unit of ``prefix`` work at ``N`` nodes."""
    if prefix == "per_cell":
        return _COST["per_cell_scale"] * _COST["per_cell_c1"] * N
    shape = _COST[prefix + "_c2"] * N ** 2 + _COST[prefix + "_c3"] * N ** 3
    return _COST[prefix + "_scale"] * shape


def calibrate(sizes: Sequence[int] = (121, 301)) -> Dict[str, Any]:
    """Measure this runtime's speed and scale the cost model to it.

    Times one eigenmode decomposition (the per-q unit) and one linear solve (the
    per-point unit) at each size, then sets ``per_q_scale`` and
    ``per_solve_scale`` to the mean ratio of measured to predicted.  The worker
    calls this once at startup; it costs about 0.3 s natively and rather more in
    a browser, which is precisely the information being bought.
    """
    import time

    q = 0.05
    ns: List[int] = []
    t_eig: List[float] = []
    t_solve: List[float] = []
    for L in sizes:
        system = RibbonSystem([Ribbon(W_nm=10.0, L=int(L))])
        phi = [_src.phi_ext_plane_wave(x) for x in system.x_nodes]
        t0 = time.perf_counter()
        system.eigenmodes(q)
        t_eig.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        system.solve_linear(q, 0.5, phi)
        t_solve.append(time.perf_counter() - t0)
        ns.append(int(system.N))

    def _scale(prefix: str, times: List[float]) -> float:
        # Ratio against the UNSCALED shape, so calibrating twice is idempotent.
        ratios = []
        for N, t in zip(ns, times):
            shape = _COST[prefix + "_c2"] * N ** 2 + _COST[prefix + "_c3"] * N ** 3
            if shape > 0.0 and t > 0.0:
                ratios.append(t / shape)
        return float(np.mean(ratios)) if ratios else 1.0

    _COST["per_q_scale"] = _scale("per_q", t_eig)
    _COST["per_solve_scale"] = _scale("per_solve", t_solve)

    # The per-cell term is MEASURED, not inferred from the solve scale. Inferring
    # it was wrong by an order of magnitude in a browser: WASM's penalty on the
    # O(N^3) LAPACK solve is not its penalty on O(N) numpy over the nodes, and
    # using the solve's factor for both priced the LDOS default at 84 s when it
    # ran in 8.3 s. A ten-fold over-estimate on the one module most able to lock a
    # tab is not a safe error -- it just stops anyone opening it.
    #
    # Measured as a SLOPE, not as a total minus an estimate. Two maps over the
    # same two wave vectors, one with few energies and one with many: the per-q
    # work is identical in both, so the difference divided by the extra cells is
    # the per-cell cost with the eigendecompositions cancelled exactly. Taking a
    # single small map and subtracting a *predicted* per-q part does not work --
    # the prediction is a cold-start number and the probe's second call is warm,
    # so the subtraction swallows the whole measurement and leaves zero.
    t_cell = 0.0
    cell_N = 0
    try:
        small = RibbonSystem([Ribbon(W_nm=10.0, L=61)])
        q_small = np.array([0.05, 0.1])
        r0, p_vec = (0.0, 0.0, 5.0), (1.0, 0.0, 0.0)
        n_lo, n_hi = 8, 48

        def _map(n: int) -> float:
            Es_probe = np.linspace(0.3, 0.9, n)
            t0 = time.perf_counter()
            _disp.ldos_map(small, q_small, Es_probe, r0, p_vec, fast_path=True)
            return time.perf_counter() - t0

        _map(n_lo)                      # warm the kernel caches first
        t_lo = _map(n_lo)
        t_hi = _map(n_hi)
        t_cell = t_hi
        cell_N = int(small.N)
        extra_cells = int(q_small.size * (n_hi - n_lo))
        measured_per_cell = max(t_hi - t_lo, 0.0) / extra_cells
        shape = _COST["per_cell_c1"] * cell_N
        if shape > 0.0 and measured_per_cell > 0.0:
            _COST["per_cell_scale"] = float(measured_per_cell / shape)
    except Exception:  # noqa: BLE001 - a calibration that fails must not stop the page
        # Fall back to the solve factor, which is at least the right order of
        # magnitude, and say so in `source` below.
        _COST["per_cell_scale"] = _COST["per_solve_scale"]
    _COST["source"] = "calibrated on this runtime at N = %s%s" % (
        ", ".join(map(str, ns)),
        "" if cell_N else " (per-cell term inferred, not measured)",
    )
    _COST["calibrated"] = True
    return {
        "schema": SCHEMA_VERSION,
        "ok": True,
        "n_nodes": ns,
        "eigen_seconds": [_f(t) for t in t_eig],
        "solve_seconds": [_f(t) for t in t_solve],
        "per_q_scale": _f(_COST["per_q_scale"]),
        "per_solve_scale": _f(_COST["per_solve_scale"]),
        "per_cell_scale": _f(_COST["per_cell_scale"]),
        "cell_probe_seconds": _f(t_cell),
        "model": {k: (_f(v) if isinstance(v, float) else v) for k, v in _COST.items()},
    }


def _work_counts(
    kind: str, spec: Dict[str, Any], system: RibbonSystem, qs: np.ndarray, Es: np.ndarray
) -> tuple:
    """``(n_full_solves, n_q_units, n_cells)`` for one calculation.  No timing."""
    fast = system.fast_path_obstruction() is None
    screened = bool(system.layers)
    if kind in ("geometry", "sheet_dispersion", "material"):
        # A closed-form conductivity on a grid: no linear algebra at all.
        return (0, 0, 0)
    if kind == "near_field":
        # One solve, then the kernel evaluated at every point of the map.
        nx = int(_get(spec, "nx", 160))
        nz = int(_get(spec, "nz", 96))
        return (1, 0, nx * nz)
    if kind == "eigenmodes":
        return (0, 1, 0)
    if kind == "bands":
        # One eigenproblem per q unscreened.  Screened, eta depends on energy, so
        # each band is a root-find whose every evaluation rebuilds Vbar -- call it
        # a dozen evaluations per band, and say in the result that it is a bound.
        n_bands = int(_get(spec, "n_bands", 4))
        return (0, int(qs.size) * (1 if not screened else 12 * max(1, n_bands)), 0)
    if kind == "nonlinear_spectrum":
        # Per energy: the linear solve, then the SHG and THG blocks.
        return (int(Es.size) * 3, 0, int(Es.size))
    if kind == "eels_spectrum":
        # No fast path here: the loss is an overlap with a source that moves
        # with the energy, so there is nothing to hoist out of the loop and
        # every (q, E) is a full solve. Pricing it as linear_spectrum's eigen
        # route would under-report it by the whole point of that route.
        return (int(qs.size * Es.size), 0, int(qs.size * Es.size))
    # linear_spectrum, ldos
    cells = int(qs.size * Es.size)
    return (0 if fast else cells, int(qs.size) if fast else 0, cells)


def _estimate_seconds(N: int, n_solves: int, n_q_units: int, n_cells: int = 0) -> float:
    return (
        n_solves * _unit_cost("per_solve", N)
        + n_q_units * _unit_cost("per_q", N)
        + n_cells * _unit_cost("per_cell", N)
    )


def _guard_cost(
    kind: str, spec: Dict[str, Any], system: RibbonSystem, qs: np.ndarray, Es: np.ndarray
) -> float:
    """Refuse a calculation the cost model puts past the time ceiling.

    Returns the estimate in seconds so the caller can report it.  This is the
    ceiling that actually protects the tab: ``MAX_SLOW_SOLVES`` counts solves and
    cannot tell 300 cheap ones from 300 expensive ones.
    """
    n_solves, n_q_units, n_cells = _work_counts(kind, spec, system, qs, Es)
    seconds = _estimate_seconds(int(system.N), n_solves, n_q_units, n_cells)
    if system.layers:
        # The model does not count the substrate quadratures (see
        # cost_estimate's `estimate_is_floor`), and measurement puts the true
        # cost 4x to 16x higher.  The ceiling exists to stop a frozen tab, so it
        # has to act on the pessimistic end of what was measured rather than on
        # a figure known to be low.  This is a stopgap for a missing model term,
        # and it is deliberately a bound and not a prediction.
        seconds = seconds * 16.0
    if seconds > MAX_ESTIMATED_SECONDS:
        raise WebSpecError(
            "This would take about %.0f s (%d nodes, %d full solves, %d per-q "
            "decompositions) against a %.0f s ceiling. Coarsen the grid, use fewer "
            "nodes, or -- if drift or a substrate is blocking the fast path -- "
            "drop it. The estimate is %s."
            % (
                seconds,
                int(system.N),
                n_solves,
                n_q_units,
                MAX_ESTIMATED_SECONDS,
                "calibrated on this runtime"
                if _COST["calibrated"]
                else "from a desktop default and may understate a browser",
            )
        )
    return seconds


def cost_estimate(spec: Dict[str, Any], kind: str = "linear_spectrum") -> Dict[str, Any]:
    """What this spec will cost, from the spec alone.  Runs no physics.

    The page calls this on every control change so it can warn *before* a click
    rather than after a frozen tab.  Accuracy is an order of magnitude, which is
    all a warning needs; ``calibrated`` says whether the price came from this
    runtime or from a desktop default.
    """
    try:
        system = _build_system(spec)
    except WebSpecError as exc:
        return {"schema": SCHEMA_VERSION, "ok": False, "reason": str(exc)}

    N = int(system.N)
    Es = _energies(spec) if kind != "geometry" else np.array([1.0])
    qs = _qs(spec)
    obstruction = system.fast_path_obstruction()
    screened = bool(system.layers)

    n_solves, n_q_units, n_cells = _work_counts(kind, spec, system, qs, Es)
    seconds = _estimate_seconds(N, n_solves, n_q_units, n_cells)
    return {
        "schema": SCHEMA_VERSION,
        "ok": True,
        "kind": kind,
        "n_nodes": N,
        "n_q": int(qs.size),
        "n_energies": int(Es.size),
        "fast_path": obstruction is None,
        "fast_path_obstruction": obstruction,
        # A SUBSTRATE MAKES THIS A FLOOR, NOT AN ESTIMATE.  Every screened block
        # rebuild runs O(N*NQ) quadratures -- build_Vbar, and the reflected
        # source of notes 8.4 for a dipole -- and the model counts solves, per-q
        # decompositions and cells, none of which is a quadrature.  Measured
        # native, one ribbon, LDOS: 48 cells 0.12 s predicted against 1.40 s
        # actual, 1280 cells 3.3 s against 54 s.  Unscreened the same model is
        # good to 1.3x, so the substrate is the whole error.  No coefficient is
        # offered here on purpose: the ratio runs 3.7x to 16x over those grids
        # and one number cannot carry that spread, so the honest report is the
        # direction of the error, which the page can show as ">=".
        "estimate_is_floor": screened,
        "floor_reason": (
            "a substrate adds O(N*NQ) quadratures per screened rebuild that the "
            "cost model does not count; measured 4x to 16x the figure above"
            if screened
            else None
        ),
        "n_full_solves": n_solves,
        "n_q_units": n_q_units,
        "n_cells": n_cells,
        "estimated_seconds": _f(seconds),
        "calibrated": bool(_COST["calibrated"]),
        "cost_model": str(_COST["source"]),
        "warn": bool(seconds > 3.0),
        "refuse": bool(
            seconds > MAX_ESTIMATED_SECONDS
            # Only where the seconds cannot be trusted -- see MAX_SLOW_SOLVES.
            or (screened and n_solves > MAX_SLOW_SOLVES)
            or qs.size * Es.size > MAX_MAP_CELLS
        ),
        "is_bound": bool(kind == "bands" and screened),
    }


_KINDS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "geometry": geometry,
    "material": material,
    "near_field": near_field,
    "eigenmodes": eigenmodes,
    "bands": bands,
    "sheet_dispersion": sheet_dispersion,
    "linear_spectrum": linear_spectrum,
    "eels_spectrum": eels_spectrum,
    "ldos": ldos,
    "nonlinear_spectrum": nonlinear_spectrum,
}


def run(kind: str, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch one calculation.  Never raises: errors come back as data.

    The worker has no good way to distinguish a user's impossible request from a
    library bug across the Pyodide boundary, so every failure is returned with
    ``ok: False`` and a message, and only a genuinely unexpected exception
    carries its type for a bug report.

    ``presets``, ``cost_estimate`` and ``calibrate`` dispatch here too, so the
    worker has exactly one entry point and no special cases.  ``cost_estimate``
    reads which calculation to price from ``spec["for_kind"]``.
    """
    kind = str(kind)
    if kind == "presets":
        return dict(presets(), ok=True, kind=kind)
    if kind == "calibrate":
        out = calibrate(**({"sizes": spec["sizes"]} if spec and spec.get("sizes") else {}))
        out["kind"] = kind
        return out
    if kind == "cost_estimate":
        spec = dict(spec or {})
        for_kind = str(spec.pop("for_kind", "linear_spectrum"))
        out = cost_estimate(spec, for_kind)
        out["kind"] = kind
        return out

    fn = _KINDS.get(kind)
    if fn is None:
        return {
            "schema": SCHEMA_VERSION,
            "ok": False,
            "reason": "unknown calculation %r; expected one of %s"
            % (kind, ", ".join(sorted(list(_KINDS) + ["presets", "cost_estimate", "calibrate"]))),
        }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn(dict(spec or {}))
        # The library warns rather than lying -- an unresolved quadrature, a zero
        # standoff, a map that cannot represent stacked ribbons. Dropping those at
        # the JSON boundary would turn a hedged number into a confident one.
        seen = set()
        for w in caught:
            msg = " ".join(str(w.message).split())
            if msg not in seen:
                seen.add(msg)
                out.setdefault("notes", []).append("%s: %s" % (w.category.__name__, msg))
    except WebSpecError as exc:
        return {"schema": SCHEMA_VERSION, "ok": False, "kind": kind, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak a traceback
        return {
            "schema": SCHEMA_VERSION,
            "ok": False,
            "kind": kind,
            "reason": "%s: %s" % (type(exc).__name__, exc),
            "unexpected": True,
        }
    out["ok"] = True
    out["kind"] = kind
    return out


def presets() -> Dict[str, Any]:
    """The published parameter sets, with a provenance tag on every value.

    Three tags, because two would hide the distinction that matters:

    ``paper``
        Independently confirmed in the publication itself -- its text, its
        caption or its figure axes.
    ``matlab``
        Present in the reference MATLAB's own parameter block, but not checked
        against the publication from here.
    ``assumed``
        Chosen by this project. Not in either source.

    Each preset also carries a ``claim``, which is the honest strength of the
    reproduction:

    ``confirmed``
        The parameters are corroborated by the publication.
    ``attributed``
        The only evidence tying these parameters to that figure is a comment in
        the reference MATLAB. The publication was not available to check.
    ``reference-default``
        The reference's own working default. It matches no published figure and
        must not be presented as one.

    The page is expected to show the tag and the claim. A preset that presented
    a guessed parameter as a published reproduction would be making a claim
    nobody checked, which is the single failure this structure exists to stop.

    Timings are seconds measured on a desktop CPython; a browser is slower, so
    treat them as a floor and ask :func:`cost_estimate` for the real number.
    """
    return {"schema": SCHEMA_VERSION, "presets": _PRESETS}


_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "single-ribbon-w10",
        "label": "Single 10 nm ribbon",
        "citation": "reference default (parity harness case C3_single_W10)",
        "claim": "reference-default",
        "kind": "linear_spectrum",
        "blurb": "The cheapest thing here, and the one to start from: one ribbon, "
                 "one spectrum, under a second.",
        "spec": {
            "ribbons": [{"W_nm": 10.0, "x0_nm": 0.0, "EF_eV": 0.5, "gamma_eV": 0.0263, "L": 150}],
            "q_nm_inv": 0.0,
            "energies_eV": {"min": 0.30, "max": 1.20, "n": 180},
            "source": {"kind": "plane_wave"},
            "material": {"EF_eV": 0.5, "gamma_eV": 0.0263},
        },
        "provenance": {
            "W_nm": "matlab", "EF_eV": "matlab", "gamma_eV": "matlab", "L": "matlab",
            "energies_eV": "matlab", "q_nm_inv": "matlab", "eps": "assumed",
        },
        "notes": [
            "gamma = 0.0263 eV is derivable, not arbitrary: it is the Drude rate at a "
            "mobility of 500 cm^2/V/s and E_F = 0.5 eV.",
            "Vacuum on both sides. The reference's polarizability driver has no "
            "dielectric environment at all, so eps = 1 is forced by its code.",
        ],
        "estimated_seconds_desktop": 0.1,
    },
    {
        "id": "yu-2017-fig5c",
        "label": "Two coplanar ribbons, hybridised",
        "citation": "Yu, Cox, Garcia de Abajo et al., ACS Photonics 4, 3106 (2017), Fig. 5c",
        "claim": "attributed",
        "kind": "linear_spectrum",
        "blurb": "A 51 nm and a 22 nm ribbon 3 nm apart. The narrow one's mode "
                 "hybridises with the wide one's, which is the point of the figure.",
        "spec": {
            "ribbons": [
                {"W_nm": 51.0, "x0_nm": 0.0, "z0_nm": 0.0, "EF_eV": 1.0, "gamma_eV": 0.02, "L": 255},
                {"W_nm": 22.0, "x0_nm": 39.5, "z0_nm": 0.0, "EF_eV": 1.0, "gamma_eV": 0.02, "L": 110},
            ],
            "q_nm_inv": 0.0,
            "energies_eV": {"min": 0.45, "max": 0.65, "n": 120},
            "source": {"kind": "plane_wave"},
            "material": {"EF_eV": 1.0, "gamma_eV": 0.02},
        },
        "provenance": {
            "W_nm": "matlab", "x0_nm": "matlab", "z0_nm": "matlab", "EF_eV": "matlab",
            "gamma_eV": "matlab", "L": "matlab", "energies_eV": "matlab",
            "q_nm_inv": "matlab", "eps": "assumed", "T_K": "matlab",
        },
        "notes": [
            "ATTRIBUTED, NOT CONFIRMED. The only evidence tying these parameters to "
            "Yu Fig. 5c is a one-line comment in the reference MATLAB; that paper is "
            "not available here to check against. Read this as 'the parameters the "
            "reference attributes to that figure'.",
            "L = [255, 110] is dx = 0.2 nm, the resolution three of the reference's "
            "own drivers use for this case. Its polarizability driver uses dx = 0.1 nm, "
            "L = [510, 220], which is four times the work. Peak POSITIONS are "
            "insensitive to the choice; absolute magnitudes move a few percent.",
            "The gap is 3.0 nm, from the edges at 25.5 and 28.5 nm. It is not a "
            "variable in the reference -- it follows from W and x0.",
            "Ribbon order is not cosmetic: phi_ext carries the absolute offset, so "
            "swapping which ribbon sits at the origin is not a translation. The "
            "reference leaves x0 = [39.5, 0] commented beside this one, unresolved.",
            "The reference's sigma here is a zero-temperature Drude expression with no "
            "temperature parameter at all. Raising T departs from the figure.",
        ],
        "reference": {"L": [510, 220], "dx_nm": 0.1, "n_energies": 300},
        "estimated_seconds_desktop": 2.1,
    },
    {
        "id": "paper406-fig1b",
        "label": "Two ribbons, second harmonic",
        "citation": "Nanoscale 15, 3150 (2023), Fig. 1b",
        "claim": "confirmed",
        "kind": "nonlinear_spectrum",
        "blurb": "A 160 nm and a 40 nm ribbon 2.5 nm apart, driven to second order. "
                 "The asymmetric pair breaks the centrosymmetry that would forbid SHG.",
        "spec": {
            "ribbons": [
                {"W_nm": 160.0, "x0_nm": 0.0, "z0_nm": 0.0, "EF_eV": 0.4, "gamma_eV": 0.01, "L": 400},
                {"W_nm": 40.0, "x0_nm": 102.5, "z0_nm": 0.0, "EF_eV": 0.4, "gamma_eV": 0.01, "L": 250},
            ],
            "q_nm_inv": 0.0,
            "energies_eV": {"min": 0.05, "max": 0.30, "n": 60},
            "orders": ["shg"],
            "source": {"kind": "plane_wave"},
            "material": {"EF_eV": 0.4, "gamma_eV": 0.01},
        },
        "provenance": {
            "W_nm": "paper", "x0_nm": "matlab", "z0_nm": "matlab", "EF_eV": "paper",
            "gamma_eV": "matlab", "L": "matlab", "energies_eV": "matlab",
            "q_nm_inv": "matlab", "eps": "paper", "orders": "matlab",
        },
        "notes": [
            "W = [160, 40] nm at E_Fj = 0.4 eV and eps = 1 are confirmed in paper 406's "
            "own text, not just in the MATLAB.",
            "ONE CURVE OF A MULTI-CURVE FIGURE. The published figure sweeps several "
            "ribbon separations d; the reference MATLAB pins exactly one, d = 2.5 nm "
            "(edges at 80 and 82.5 nm). The other separations are nowhere in the tree.",
            "L = [400, 250] is the reference's own commented alternative, and it is not "
            "uniform: dx = 0.4 nm on the wide ribbon and 0.16 nm on the narrow one. Its "
            "primary block is L = [1600, 400], 2000 nodes, which this tool will run but "
            "which costs minutes rather than seconds.",
            "This preset asks for the second order only. The published figure has two "
            "panels -- absorption and |chi^(2,2)| -- and the reference's own switch is "
            "set to compute up to the second harmonic.",
        ],
        "reference": {"L": [1600, 400], "dx_nm": 0.1, "n_energies": 300, "separations_nm": "several; only 2.5 is in the reference code"},
        "estimated_seconds_desktop": 9.4,
    },
    {
        "id": "paper456-drift-ldos",
        "label": "Drift-biased ribbon, LDOS map",
        "citation": "Nat. Commun. 17, 1114 (2026)",
        "claim": "attributed",
        "kind": "ldos",
        "blurb": "A 10 nm ribbon carrying a drift current, mapped over (q, energy). "
                 "The +-q asymmetry is the whole point: drift both shifts the bands "
                 "and quenches them.",
        "spec": {
            "ribbons": [
                {"W_nm": 10.0, "x0_nm": 5.0, "z0_nm": 0.0, "EF_eV": 0.5,
                 "gamma_eV": 0.0263, "beta": 0.2573, "L": 150}
            ],
            "qs": {"min": 0.02, "max": 0.5, "n_pos": 16},
            "energies_eV": {"min": 0.10, "max": 1.00, "n": 40},
            "source": {"kind": "dipole", "r0_nm": [5.0, 0.0, 5.0], "p": [1.0, 0.0, 0.0]},
            "quantity": "ldos",
            "n_peaks": 3,
            "material": {"EF_eV": 0.5, "gamma_eV": 0.0263, "beta": 0.2573},
        },
        "provenance": {
            "W_nm": "matlab", "x0_nm": "matlab", "L": "matlab", "EF_eV": "matlab",
            "gamma_eV": "matlab", "beta": "matlab", "source": "matlab",
            "qs": "assumed", "energies_eV": "assumed", "eps": "assumed",
        },
        "notes": [
            "gamma = 0.0263 eV and beta = 0.2573 are the values the paper's own "
            "production drivers use, with a header comment deriving gamma from a "
            "mobility of 500 cm^2/V/s at E_F = 0.5 eV. A second script in the "
            "reference tree uses the round numbers 0.02 and 0.2 instead and plots "
            "log|E_x| rather than an LDOS -- it is not the production run.",
            "THE GRID IS COARSENED. The production run is 300 q by 300 energies. With "
            "drift there is no fast path -- eta is a tensor, so the operator depends on "
            "frequency and every cell is a full solve -- which makes 90000 cells "
            "unaffordable in a browser. 32 by 40 shows the same physics at a fortieth "
            "of the cost.",
            "eps = 1 is an assumption. The production driver's own eps_eff is "
            "unknowable from the reference tree: the function that holds it is missing, "
            "and only its call sites survive.",
            "Drift is a skewed Fermi-Dirac distribution solved through Boltzmann, never "
            "a Galilean frame shift. It survives q -> 0, makes eta a tensor, and puts "
            "the non-reciprocity in (alpha+beta)^2 and the alpha*beta cross term.",
            "The LDOS is in arbitrary units -- its 2 pi^2 |p|^2 normalisation has no "
            "counterpart in any reference code. Read positions and relative weights.",
        ],
        "reference": {"n_q": 300, "n_energies": 300, "q_max_nm_inv": 0.5,
                      "energy_range_eV": [0.01, 1.0]},
        "estimated_seconds_desktop": 3.0,
    },
    {
        "id": "paper456-no-drift",
        "label": "The same ribbon, no drift",
        "citation": "Nat. Commun. 17, 1114 (2026), comparison case",
        "claim": "reference-default",
        "kind": "ldos",
        "blurb": "The drift preset with beta = 0, for the comparison the figure is "
                 "built on. Reciprocal, and much faster -- without drift the fast "
                 "path is available.",
        "spec": {
            "ribbons": [
                {"W_nm": 10.0, "x0_nm": 5.0, "z0_nm": 0.0, "EF_eV": 0.5,
                 "gamma_eV": 0.0263, "beta": 0.0, "L": 150}
            ],
            "qs": {"min": 0.02, "max": 0.5, "n_pos": 24},
            "energies_eV": {"min": 0.10, "max": 1.00, "n": 80},
            "source": {"kind": "dipole", "r0_nm": [5.0, 0.0, 5.0], "p": [1.0, 0.0, 0.0]},
            "quantity": "ldos",
            "n_peaks": 3,
            "material": {"EF_eV": 0.5, "gamma_eV": 0.0263},
        },
        "provenance": {
            "W_nm": "matlab", "x0_nm": "matlab", "L": "matlab", "EF_eV": "matlab",
            "gamma_eV": "matlab", "beta": "assumed", "source": "matlab",
            "qs": "assumed", "energies_eV": "assumed", "eps": "assumed",
        },
        "notes": [
            "beta = 0 is ASSUMED for this comparison. It never appears as an assigned "
            "value beside E_F = 0.5 in the reference tree -- only as part of a saved "
            "data filename. The drifted values that do appear are 0.2, 0.2573, 0.3 and "
            "0.5, so treat 'beta = 0 vs 0.2573' as this project's pairing.",
            "The dissymmetry must come out exactly zero here, by symmetry. A nonzero "
            "value would be a numerical artefact, not physics -- it is worth checking.",
            "Without drift the isotropic eigen decomposition applies: one decomposition "
            "per q instead of a solve per cell, which is why this map can afford a "
            "finer grid than the drifted one.",
        ],
        "estimated_seconds_desktop": 4.7,
    },
]
