"""Standalone timing bench for the quasistatic ribbon finite-difference solver.

Run from the repository root:

    py -3.11 scripts/electrostatics_fd/bench.py

Prints four tables and nothing else -- no plots, no files, no dependencies
beyond numpy and scipy (both reached only through the package's own modules):

1. Coulomb matrix assembly, ``operators.build_V`` against
   ``operators.build_V_toeplitz``, at L in {200, 400, 800}.
2. A single linear solve at L in {100, 200, 400, 800}, in the three regimes
   that differ in practice: a first solve with the default naive assembly, a
   first solve with the Toeplitz assembly, and a solve on cached operators.
3. A 100-frequency sweep with and without the isotropic eigenmode fast path.
4. The reflected (image) Coulomb block ``substrate.build_Vbar`` at L = 150,
   NQ = 4000.

Every timing is the best of ``REPEATS`` runs of ``time.perf_counter``: the
minimum, not the mean, because scheduler and clock noise only ever add time.
The budgets asserted in ``tests/test_performance.py`` are the same quantities;
this script is the human-readable view of them.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings
from typing import Any, Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import the package by name.  ``conductivity`` uses a package-relative import,
# so the package's PARENT directory has to be on sys.path; running this file as
# a script only puts the package directory itself there.
# ---------------------------------------------------------------------------
_PKG_DIR = pathlib.Path(__file__).resolve().parent
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))

_PKG = _PKG_DIR.name
operators = __import__(f"{_PKG}.operators", fromlist=["operators"])
substrate = __import__(f"{_PKG}.substrate", fromlist=["substrate"])
solver = __import__(f"{_PKG}.solver", fromlist=["solver"])

# ---------------------------------------------------------------------------
# Bench configuration
# ---------------------------------------------------------------------------
REPEATS = 3
W_NM = 20.0
Q_INV_NM = 0.05
E_EV = 0.30

V_SIZES = (200, 400, 800)
SOLVE_SIZES = (100, 200, 400, 800)
SWEEP_L = 300
SWEEP_NFREQ = 100
VBAR_L = 150
VBAR_NQ = 4000


def best_of(fn: Callable[[], Any], repeats: int = REPEATS) -> Tuple[float, Any]:
    """Best-of-``repeats`` wall time of ``fn`` [s], with its last return value."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def grid(L: int) -> Tuple[np.ndarray, float]:
    """Normalised node array on [-1/2, 1/2] and its spacing h = 1/(L-1)."""
    theta = np.linspace(-0.5, 0.5, L)
    return theta, 1.0 / (L - 1)


def ramp(L: int) -> np.ndarray:
    """Dipolar drive phi_ext = x on one ribbon's nodes [nm]."""
    return (W_NM * np.linspace(-0.5, 0.5, L)).astype(np.complex128)


def make_system(L: int, use_toeplitz: bool):
    """One isotropic, undrifted, substrate-free ribbon with ``L`` nodes."""
    system = solver.RibbonSystem([solver.Ribbon(W_nm=W_NM, L=L)])
    system.use_toeplitz = use_toeplitz
    return system


def rule(width: int = 78) -> None:
    """Print a horizontal rule."""
    print(f"{'-' * width}")


def heading(title: str) -> None:
    """Print a section heading."""
    print(f"\n{title}")
    rule()


# ---------------------------------------------------------------------------
# 1. Coulomb assembly
# ---------------------------------------------------------------------------
def bench_v_build() -> None:
    """Time the naive and Toeplitz Coulomb assemblies at several node counts."""
    heading(
        f"1. Coulomb assembly V (q*W = {Q_INV_NM * W_NM:.2f}, one ribbon, "
        f"z offset 0)"
    )
    print(f"{'L':>6} {'build_V [ms]':>15} {'build_V_toeplitz [ms]':>23}"
          f" {'speed-up':>10} {'max rel diff':>14}")
    for L in V_SIZES:
        theta, h = grid(L)
        t_naive, v_naive = best_of(
            lambda: operators.build_V(
                Q_INV_NM, W_NM, W_NM, 0.0, 0.0, 0.0, 0.0, theta, theta, h
            )
        )
        t_toep, v_toep = best_of(
            lambda: operators.build_V_toeplitz(Q_INV_NM, W_NM, 0.0, 0.0, theta, h)
        )
        nz = np.abs(v_naive) > 0.0
        rel = float(np.max(np.abs(v_naive[nz] - v_toep[nz]) / np.abs(v_naive[nz])))
        print(f"{L:>6} {t_naive * 1e3:>15.2f} {t_toep * 1e3:>23.3f}"
              f" {t_naive / t_toep:>9.1f}x {rel:>14.2e}")


# ---------------------------------------------------------------------------
# 2. Single solve
# ---------------------------------------------------------------------------
def cold_solve(L: int, use_toeplitz: bool) -> float:
    """Best-of time for a FIRST solve at ``L``: operator assembly included.

    A fresh system per repeat, because the Coulomb matrix is cached per q and a
    second solve at the same q would skip the assembly entirely.  Construction
    stays outside the timed region.
    """
    best = float("inf")
    for _ in range(REPEATS):
        system = make_system(L, use_toeplitz)
        phi_ext = [ramp(L)]
        t0 = time.perf_counter()
        system.solve_linear(Q_INV_NM, E_EV, phi_ext)
        best = min(best, time.perf_counter() - t0)
    return best


def warm_solve(L: int) -> float:
    """Best-of time for a solve on already-assembled operators at ``L``."""
    system = make_system(L, False)
    phi_ext = [ramp(L)]
    system.solve_linear(Q_INV_NM, E_EV, phi_ext)
    t, _ = best_of(lambda: system.solve_linear(Q_INV_NM, E_EV, phi_ext))
    return t


def bench_solve() -> None:
    """Time one linear solve in the three regimes that differ in practice."""
    heading(
        f"2. Single linear solve (one ribbon, q = {Q_INV_NM:.2f} /nm, "
        f"E = {E_EV:.2f} eV)"
    )
    print(f"{'L':>6} {'first, naive V [ms]':>21} {'first, Toeplitz V [ms]':>24}"
          f" {'cached V [ms]':>15} {'t/t(L/2)':>10}")
    previous = None
    for L in SOLVE_SIZES:
        t_naive = cold_solve(L, False)
        t_toep = cold_solve(L, True)
        t_warm = warm_solve(L)
        ratio = f"{t_naive / previous:.2f}" if previous is not None else "-"
        print(f"{L:>6} {t_naive * 1e3:>21.2f} {t_toep * 1e3:>24.2f}"
              f" {t_warm * 1e3:>15.2f} {ratio:>10}")
        previous = t_naive
    print("\n  note: RibbonSystem.use_toeplitz now defaults to True, so the"
          " SECOND column is what a\n  caller gets today; the first is the"
          " dense entry-by-entry assembly, kept for\n  comparison (the two agree"
          " on rho to ~1e-12). The last column is the\n  per-frequency cost"
          " inside a sweep, once V is cached.")


# ---------------------------------------------------------------------------
# 3. Frequency sweep
# ---------------------------------------------------------------------------
def bench_sweep() -> None:
    """Time a frequency sweep with and without the isotropic eigen fast path."""
    heading(f"3. {SWEEP_NFREQ}-frequency sweep at L = {SWEEP_L} "
            f"(0.15 - 0.50 eV, single q)")
    system = make_system(SWEEP_L, False)
    energies = np.linspace(0.15, 0.50, SWEEP_NFREQ)
    drive = ramp(SWEEP_L)

    def phi_ext_fn(q: float, E_ev: float) -> np.ndarray:
        return drive

    obstruction = system.fast_path_obstruction()
    if obstruction is not None:
        print(f"  fast path unavailable: {obstruction}")
        return

    t_fast, a_fast = best_of(
        lambda: system.sweep([Q_INV_NM], energies, phi_ext_fn, fast_path=True)
    )
    t_direct, a_direct = best_of(
        lambda: system.sweep([Q_INV_NM], energies, phi_ext_fn, fast_path=False)
    )
    rel = float(np.max(np.abs(a_fast - a_direct) / np.abs(a_direct)))

    print(f"{'path':>22} {'total [ms]':>12} {'per frequency [ms]':>20}")
    print(f"{'eigen fast path':>22} {t_fast * 1e3:>12.1f}"
          f" {t_fast * 1e3 / SWEEP_NFREQ:>20.3f}")
    print(f"{'direct solve per E':>22} {t_direct * 1e3:>12.1f}"
          f" {t_direct * 1e3 / SWEEP_NFREQ:>20.3f}")
    print(f"\n  speed-up {t_direct / t_fast:.1f}x, max relative difference {rel:.2e}")


# ---------------------------------------------------------------------------
# 4. Reflected Coulomb block
# ---------------------------------------------------------------------------
def bench_vbar() -> None:
    """Time the reflected Coulomb block for a two-layer substrate."""
    heading(f"4. Reflected Coulomb block Vbar at L = {VBAR_L}, NQ = {VBAR_NQ}")
    theta, h = grid(VBAR_L)
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]

    def r_of_Q(Q):
        return substrate.reflection_coefficient(Q, E_EV, layers)

    with warnings.catch_warnings():
        # The quadrature-resolution diagnostic is about accuracy, not speed.
        warnings.simplefilter("ignore")
        t_vbar, vbar = best_of(
            lambda: substrate.build_Vbar(
                Q_INV_NM, W_NM, W_NM, 0.0, 0.0, 1.0, 1.0,
                theta, theta, h, r_of_Q, NQ=VBAR_NQ,
            )
        )
    entries = vbar.size
    print(f"{'stack':>22} {'time [ms]':>12} {'entries':>10} {'us/entry':>10}")
    print(f"{'vacuum / eps = 4':>22} {t_vbar * 1e3:>12.1f} {entries:>10}"
          f" {t_vbar * 1e6 / entries:>10.2f}")
    print(f"\n  image standoff z_j + z_jp = 2.0 nm, "
          f"max |Vbar| = {np.max(np.abs(vbar)):.3e}")


# ---------------------------------------------------------------------------
def main() -> int:
    """Run every bench section in order."""
    print("electrostatics_fd timing bench")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, "
          f"best of {REPEATS} runs, times are wall clock")
    bench_v_build()
    bench_solve()
    bench_sweep()
    bench_vbar()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
