"""Performance budget for the quasistatic ribbon finite-difference solver.

Every test here ASSERTS a timing or a speed-up; none merely reports one.  Each
timing is the best of three runs of ``time.perf_counter`` (best-of, not mean,
because the quantity of interest is the machine's capability, and scheduler
noise only ever adds time).  Every failure message carries the measured
numbers, so a regression report needs nothing but the pytest output.

The whole module carries ``pytest.mark.perf`` so it can be deselected on a slow
or shared machine:

    py -3.11 -m pytest scripts/electrostatics_fd/tests -q -m "not perf"

MARKER REGISTRATION.  The ``perf`` marker is registered in
``scripts/electrostatics_fd/conftest.py`` (``pytest_configure``), which pytest
loads for any test path inside the package regardless of rootdir resolution.
An earlier ``pytest.ini`` alongside it did the same job only when the path given
on the command line happened to lie inside the package; it has been deleted in
favour of the conftest.

TWO CONFIGURATIONS, DELIBERATELY.  ``RibbonSystem.use_toeplitz`` now defaults to
``True`` -- partly BECAUSE of this module: with the dense O(L^2)-entry Coulomb
build a first solve at L = 300 cost ~115 ms against the mandated 50 ms budget,
and ``test_toeplitz_assembly_pays_for_itself_in_a_solve`` pinned the ~14x gap
that justified the change.  Every test below still sets ``use_toeplitz``
EXPLICITLY rather than relying on the default, so these numbers describe the two
assemblies and not whichever one happens to be current: the absolute budget is
measured with the Toeplitz assembly (and separately with the operator cache
warm), while the cubic-scaling window is measured on the dense end-to-end solve,
whose protocol is the only one that reproduces it (see that test's docstring).

Run with:
    cd <repo root> && py -3.11 -m pytest scripts/electrostatics_fd/tests/test_performance.py -q
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import time
import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import harness (see tests/test_acceptance.py for the rationale: conductivity
# uses a package-relative import, so the package's parent must be on sys.path).
# ---------------------------------------------------------------------------
_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_PKG_DIR), str(_PKG_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PKG_NAME = _PKG_DIR.name
_IMPORT_ERRORS: dict[str, str] = {}


def _load(name: str) -> Optional[Any]:
    """Import one sibling module, recording (not raising) an import failure."""
    for candidate in (f"{_PKG_NAME}.{name}", name):
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # noqa: BLE001 - any import-time failure
            _IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
    return None


operators = _load("operators")
substrate = _load("substrate")
solver = _load("solver")


def _why(name: str) -> str:
    return _IMPORT_ERRORS.get(name, "module present")


pytestmark = pytest.mark.perf

needs_operators = pytest.mark.xfail(
    operators is None,
    reason=(
        "operators.py (agent B) not importable: "
        + _why("operators")
        + ". Expected: build_V(q, W_j, W_jp, x_j, x_jp, z_j, z_jp, theta_j, "
        "theta_jp, h_jp) and build_V_toeplitz(q, W, x_off, z_off, theta, h)."
    ),
)
needs_substrate = pytest.mark.xfail(
    substrate is None,
    reason=(
        "substrate.py (agent C) not importable: "
        + _why("substrate")
        + ". Expected: Layer, reflection_coefficient(Q, E_ev, layers) and "
        "build_Vbar(..., r_of_Q, Qmax=None, NQ=4000)."
    ),
)
needs_solver = pytest.mark.xfail(
    solver is None,
    reason=(
        "solver.py (agent E) not importable: "
        + _why("solver")
        + ". Expected: Ribbon, RibbonSystem with solve_linear(q, E_ev, phi_ext) "
        "and sweep(qs, Es, phi_ext_fn, fast_path=True)."
    ),
)

# ---------------------------------------------------------------------------
# Common geometry
# ---------------------------------------------------------------------------
W_NM = 20.0
Q_INV_NM = 0.05
E_EV = 0.30

#: Budgets straight from the performance specification.
TOEPLITZ_MIN_SPEEDUP = 5.0
FAST_PATH_MIN_SPEEDUP = 8.0
FAST_PATH_RTOL = 1e-8
SOLVE_BUDGET_S = 50e-3
VBAR_BUDGET_S = 2.0
SCALING_WINDOW = (4.0, 16.0)


def _best_of(fn: Callable[[], Any], repeats: int = 3) -> Tuple[float, Any]:
    """Best-of-``repeats`` wall time of ``fn``, with its last return value.

    Minimum rather than mean: OS scheduling, turbo clocks and BLAS thread
    start-up only ever add time, so the fastest run is the closest estimate of
    the work actually required.
    """
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _grid(L: int) -> Tuple[np.ndarray, float]:
    """Normalised node array and spacing for ``L`` nodes."""
    theta = np.linspace(-0.5, 0.5, L)
    return theta, 1.0 / (L - 1)


def _ramp(L: int) -> np.ndarray:
    """Dipolar drive phi_ext = x on the nodes of one ribbon [nm]."""
    return (W_NM * np.linspace(-0.5, 0.5, L)).astype(np.complex128)


def _system(L: int, use_toeplitz: bool = False):
    """One isotropic, undrifted, substrate-free ribbon with ``L`` nodes."""
    system = solver.RibbonSystem([solver.Ribbon(W_nm=W_NM, L=L)])
    system.use_toeplitz = use_toeplitz
    return system


def _cold_solve_time(L: int, use_toeplitz: bool, repeats: int = 3) -> float:
    """Best-of time for a FIRST solve: operator assembly included.

    A fresh ``RibbonSystem`` per repeat, because ``_V_full`` is cached per q and
    a second solve at the same q would skip the Coulomb build entirely.  System
    construction itself stays outside the timed region.
    """
    best = float("inf")
    for _ in range(repeats):
        system = _system(L, use_toeplitz)
        phi_ext = [_ramp(L)]
        t0 = time.perf_counter()
        system.solve_linear(Q_INV_NM, E_EV, phi_ext)
        best = min(best, time.perf_counter() - t0)
    return best


# ==========================================================================
# 1. Toeplitz Coulomb assembly
# ==========================================================================
@needs_operators
def test_toeplitz_V_build_is_at_least_5x_faster_at_L800() -> None:
    """build_V_toeplitz >= 5x faster than build_V at L = 800, same matrix.

    The Toeplitz path evaluates the Coulomb kernel on the 2L-1 distinct node
    separations instead of all L^2 entries, so at L = 800 it does 1599 kernel
    evaluations rather than 640000.  Measured speed-up on the development
    machine: ~143x, i.e. the 5x budget has a very large margin and a failure
    here means the fast path stopped being taken, not that the machine is slow.

    "Identical results" is asserted as agreement to rtol 1e-10, not bit for
    bit: both paths evaluate the same closed-form kernel, but on arrays of
    different shape, so the floating-point operations are reassociated.
    Measured worst entrywise relative difference: ~2.1e-12.
    """
    L = 800
    theta, h = _grid(L)

    t_naive, V_naive = _best_of(
        lambda: operators.build_V(
            Q_INV_NM, W_NM, W_NM, 0.0, 0.0, 0.0, 0.0, theta, theta, h
        )
    )
    t_toep, V_toep = _best_of(
        lambda: operators.build_V_toeplitz(Q_INV_NM, W_NM, 0.0, 0.0, theta, h)
    )

    speedup = t_naive / t_toep
    assert speedup >= TOEPLITZ_MIN_SPEEDUP, (
        f"build_V_toeplitz is only {speedup:.2f}x faster than build_V at "
        f"L = {L} (naive {t_naive * 1e3:.2f} ms, Toeplitz {t_toep * 1e3:.3f} ms); "
        f"budget is {TOEPLITZ_MIN_SPEEDUP:.0f}x"
    )

    scale = float(np.max(np.abs(V_naive)))
    worst_abs = float(np.max(np.abs(V_naive - V_toep)))
    nz = np.abs(V_naive) > 0.0
    worst_rel = float(np.max(np.abs(V_naive[nz] - V_toep[nz]) / np.abs(V_naive[nz])))
    assert worst_rel < 1e-10, (
        f"the two Coulomb builds disagree at L = {L}: worst entrywise relative "
        f"difference {worst_rel:.3e}, worst absolute {worst_abs:.3e} against a "
        f"matrix scale of {scale:.3e} (speed-up measured was {speedup:.1f}x)"
    )


@needs_operators
@pytest.mark.parametrize("x_off, z_off", [(0.0, 0.0), (30.0, 0.0), (30.0, 5.0)])
def test_toeplitz_V_build_stays_faster_off_diagonal(x_off: float, z_off: float) -> None:
    """The Toeplitz win survives the off-diagonal (two-ribbon) blocks.

    An inter-ribbon block is Toeplitz but not symmetric, and a non-zero z
    offset switches the kernel to the cheaper midpoint branch, which narrows
    the gap.  Measured at L = 400: 98x at (30, 0) and 16x at (30, 5); the 5x
    budget must hold in every case or the solver's multi-ribbon assembly loses
    the fast path.
    """
    L = 400
    theta, h = _grid(L)
    t_naive, V_naive = _best_of(
        lambda: operators.build_V(
            Q_INV_NM, W_NM, W_NM, x_off, 0.0, z_off, 0.0, theta, theta, h
        )
    )
    t_toep, V_toep = _best_of(
        lambda: operators.build_V_toeplitz(Q_INV_NM, W_NM, x_off, z_off, theta, h)
    )
    speedup = t_naive / t_toep
    assert speedup >= TOEPLITZ_MIN_SPEEDUP, (
        f"at x_off = {x_off}, z_off = {z_off}, L = {L} the Toeplitz build is "
        f"only {speedup:.2f}x faster (naive {t_naive * 1e3:.2f} ms, Toeplitz "
        f"{t_toep * 1e3:.3f} ms); budget {TOEPLITZ_MIN_SPEEDUP:.0f}x"
    )
    nz = np.abs(V_naive) > 0.0
    worst_rel = float(np.max(np.abs(V_naive[nz] - V_toep[nz]) / np.abs(V_naive[nz])))
    assert worst_rel < 1e-10, (
        f"off-diagonal block mismatch at x_off = {x_off}, z_off = {z_off}: "
        f"worst entrywise relative difference {worst_rel:.3e}"
    )


# ==========================================================================
# 2. Isotropic eigenmode fast path over a frequency sweep
# ==========================================================================
@needs_solver
def test_eigen_fast_path_is_at_least_8x_faster_over_200_frequencies() -> None:
    """Fast-path sweep >= 8x faster than per-frequency solves, agreeing to 1e-8.

    At L = 300 over 200 frequencies the fast path eigendecomposes the geometric
    operator once per q and then evaluates
    ``phi(E) = sum_m v_m (u_m . phi_ext) / (1 - eta(E) lambda_m)``, replacing
    200 O(L^3) factorisations by one plus 200 O(L^2) applications.  Measured:
    ~45 ms against ~2.1 s, a 48x speed-up, agreeing to 3.5e-11.

    Both numbers matter.  A fast path that is quick but wrong is the failure
    mode this test exists for, so the accuracy assertion comes first.
    """
    L, n_freq = 300, 200
    system = _system(L)
    Es = np.linspace(0.15, 0.50, n_freq)
    drive = _ramp(L)

    def phi_ext_fn(q: float, E_ev: float) -> np.ndarray:
        return drive

    obstruction = system.fast_path_obstruction()
    assert obstruction is None, (
        "the reference system should admit the isotropic fast path, but "
        f"fast_path_obstruction() reports: {obstruction}"
    )

    t_fast, A_fast = _best_of(
        lambda: system.sweep([Q_INV_NM], Es, phi_ext_fn, fast_path=True)
    )
    t_direct, A_direct = _best_of(
        lambda: system.sweep([Q_INV_NM], Es, phi_ext_fn, fast_path=False)
    )

    worst_rel = float(np.max(np.abs(A_fast - A_direct) / np.abs(A_direct)))
    assert worst_rel < FAST_PATH_RTOL, (
        f"the fast path disagrees with the direct solves by {worst_rel:.3e} "
        f"(budget rtol {FAST_PATH_RTOL:.0e}) over {n_freq} frequencies at "
        f"L = {L}; timings were fast {t_fast * 1e3:.1f} ms, direct "
        f"{t_direct * 1e3:.1f} ms"
    )

    speedup = t_direct / t_fast
    assert speedup >= FAST_PATH_MIN_SPEEDUP, (
        f"the eigen fast path is only {speedup:.2f}x faster than {n_freq} "
        f"direct solves at L = {L} (fast {t_fast * 1e3:.1f} ms, direct "
        f"{t_direct * 1e3:.1f} ms); budget is {FAST_PATH_MIN_SPEEDUP:.0f}x"
    )


# ==========================================================================
# 3. Single-solve budget
# ==========================================================================
@needs_solver
def test_single_linear_solve_at_L300_is_under_50ms() -> None:
    """One linear solve at L = 300 finishes inside 50 ms.

    Asserted for both configurations in which "a single solve" is a meaningful
    unit of work:

    * a complete first solve with the Toeplitz Coulomb assembly enabled
      (``use_toeplitz = True``): measured ~8 ms;
    * a solve with the operator cache already warm, which is what every
      frequency after the first in a sweep costs: measured ~6 ms.

    Both must fit the budget.  The remaining case -- a first solve on the DENSE
    entry-by-entry Coulomb assembly (``use_toeplitz = False``) -- does NOT, at
    ~115 ms; that is pinned separately by
    ``test_toeplitz_assembly_pays_for_itself_in_a_solve`` rather than hidden
    here, and it is why ``use_toeplitz`` now defaults to True.
    """
    L = 300
    t_cold_fast = _cold_solve_time(L, use_toeplitz=True)
    assert t_cold_fast < SOLVE_BUDGET_S, (
        f"a first solve at L = {L} with the Toeplitz assembly took "
        f"{t_cold_fast * 1e3:.2f} ms, over the {SOLVE_BUDGET_S * 1e3:.0f} ms budget"
    )

    system = _system(L)
    phi_ext = [_ramp(L)]
    system.solve_linear(Q_INV_NM, E_EV, phi_ext)  # populate the operator cache
    t_warm, result = _best_of(
        lambda: system.solve_linear(Q_INV_NM, E_EV, phi_ext)
    )
    assert t_warm < SOLVE_BUDGET_S, (
        f"a cached-operator solve at L = {L} took {t_warm * 1e3:.2f} ms, over "
        f"the {SOLVE_BUDGET_S * 1e3:.0f} ms budget"
    )

    # A timing is only meaningful if the call did the work.
    rho = np.asarray(result["rho"][0])
    assert rho.shape == (L,), f"solve returned rho of shape {rho.shape}, expected ({L},)"
    assert np.all(np.isfinite(rho)), "solve returned a non-finite charge density"


@needs_solver
def test_toeplitz_assembly_pays_for_itself_in_a_solve() -> None:
    """The Toeplitz assembly cuts a first solve at L = 300 by at least 5x.

    With ``use_toeplitz = False`` the first solve at a new q builds the Coulomb
    matrix entry by entry: measured ~115 ms at L = 300, against ~8 ms with the
    Toeplitz path, i.e. the dense assembly alone puts a cold solve over the
    50 ms budget of the test above.

    Asserted as a ratio, not as two absolute times, so the result does not
    depend on the machine's clock speed.  The two configurations must also
    agree on the answer to 1e-9 -- a faster assembly that changed the physics
    would be worthless.  This test is the measurement that moved
    ``RibbonSystem.use_toeplitz`` to a ``True`` default at integration; it keeps
    the gap pinned so the decision can be revisited on evidence.
    """
    L = 300
    t_naive = _cold_solve_time(L, use_toeplitz=False)
    t_toep = _cold_solve_time(L, use_toeplitz=True)
    speedup = t_naive / t_toep
    assert speedup >= TOEPLITZ_MIN_SPEEDUP, (
        f"enabling use_toeplitz speeds a first solve at L = {L} by only "
        f"{speedup:.2f}x (naive {t_naive * 1e3:.2f} ms, Toeplitz "
        f"{t_toep * 1e3:.2f} ms); expected at least "
        f"{TOEPLITZ_MIN_SPEEDUP:.0f}x from the kernel-evaluation count"
    )

    rho_naive = np.asarray(
        _system(L, False).solve_linear(Q_INV_NM, E_EV, [_ramp(L)])["rho"][0]
    )
    rho_toep = np.asarray(
        _system(L, True).solve_linear(Q_INV_NM, E_EV, [_ramp(L)])["rho"][0]
    )
    worst = float(np.max(np.abs(rho_naive - rho_toep)) / np.max(np.abs(rho_naive)))
    assert worst < 1e-9, (
        f"the two Coulomb assemblies give different charge densities at "
        f"L = {L}: worst relative difference {worst:.3e}"
    )


# ==========================================================================
# 4. Reflected Coulomb block budget
# ==========================================================================
@needs_substrate
def test_build_Vbar_at_L150_NQ4000_is_under_2s() -> None:
    """build_Vbar at L = 150, NQ = 4000 finishes inside 2 s.

    The implementation factorises the node-pair coupling into two complex
    matrix products of shape (L, NQ) @ (NQ, L) instead of materialising an
    (L, L, NQ) array, so the cost is ~2 L^2 NQ complex multiply-adds.  Measured
    with a real two-layer stack and ``reflection_coefficient`` as the callable:
    ~128 ms, a 15x margin on the budget.
    """
    L = 150
    theta, h = _grid(L)
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]

    def r_of_Q(Q):
        return substrate.reflection_coefficient(Q, E_EV, layers)

    with warnings.catch_warnings():
        # The quadrature-resolution diagnostic is about accuracy, not speed.
        warnings.simplefilter("ignore")
        t_vbar, Vbar = _best_of(
            lambda: substrate.build_Vbar(
                Q_INV_NM, W_NM, W_NM, 0.0, 0.0, 1.0, 1.0,
                theta, theta, h, r_of_Q, NQ=4000,
            )
        )

    assert t_vbar < VBAR_BUDGET_S, (
        f"build_Vbar at L = {L}, NQ = 4000 took {t_vbar * 1e3:.1f} ms, over the "
        f"{VBAR_BUDGET_S:.0f} s budget"
    )
    assert Vbar.shape == (L, L), (
        f"build_Vbar returned shape {Vbar.shape}, expected ({L}, {L}); the "
        f"{t_vbar * 1e3:.1f} ms timing is not measuring the right work"
    )
    assert np.all(np.isfinite(Vbar)), "build_Vbar returned non-finite entries"


# ==========================================================================
# 5. Cubic-scaling sanity
# ==========================================================================
@needs_solver
def test_solve_time_scales_roughly_cubically() -> None:
    """t(L = 400)/t(L = 200) for a full solve lies in [4, 16].

    The specification predicts ~8 and deliberately leaves the window wide,
    because at these sizes BLAS threading, cache effects and the O(L^2)-entry
    Coulomb assembly all compete with the O(L^3) factorisation.

    Measured on the default configuration (a complete first solve, assembly
    included), 15 trials: min 5.75, median 7.41, max 8.12 -- never outside the
    window.  That configuration is chosen BECAUSE it is the reproducible one:
    the same experiment on cached operators spans 3.2 to 7.2 and leaves the
    window about a quarter of the time, since at N = 200-400 the pure linear
    algebra is fast enough that timer and thread-pool noise dominate.
    """
    lo, hi = SCALING_WINDOW
    t200 = _cold_solve_time(200, use_toeplitz=False)
    t400 = _cold_solve_time(400, use_toeplitz=False)
    ratio = t400 / t200
    assert lo <= ratio <= hi, (
        f"solve time ratio t(400)/t(200) = {ratio:.2f} is outside the "
        f"[{lo:.0f}, {hi:.0f}] window (t(200) = {t200 * 1e3:.2f} ms, "
        f"t(400) = {t400 * 1e3:.2f} ms); ~8 is expected"
    )


@needs_solver
def test_solve_time_grows_monotonically_with_L() -> None:
    """Solve time increases with every step in L, over 100 -> 800.

    A cheap guard against the scaling test passing on noise: if two of these
    sizes come out equal in cost, something is being skipped or cached across
    systems and the ratio in the test above means nothing.
    """
    sizes = (100, 200, 400, 800)
    times = {L: _cold_solve_time(L, use_toeplitz=False) for L in sizes}
    pretty = ", ".join(f"L={L}: {times[L] * 1e3:.2f} ms" for L in sizes)
    for small, large in zip(sizes, sizes[1:]):
        assert times[large] > times[small], (
            f"solve at L = {large} was not slower than at L = {small} ({pretty})"
        )


# ==========================================================================
# Report
# ==========================================================================
@pytest.mark.xfail(
    bool(_IMPORT_ERRORS),
    reason=(
        "performance suite could not import: "
        + "; ".join(f"{k} ({v})" for k, v in sorted(_IMPORT_ERRORS.items()))
        if _IMPORT_ERRORS
        else "all sibling modules importable"
    ),
)
def test_every_timed_module_is_importable() -> None:
    """Surfaces any module the performance suite could not time.

    Not a timing check.  It passes once the package is complete and until then
    xfails with the exact import errors in its reason string, so a budget that
    was never measured is distinguishable from one that was met.
    """
    assert not _IMPORT_ERRORS, (
        "performance suite could not import: "
        + "; ".join(f"{k} ({v})" for k, v in sorted(_IMPORT_ERRORS.items()))
    )
