"""Integration-level tests: the seams BETWEEN the modules, not their interiors.

Each per-module test file verifies its own module against the physics.  Nothing
there can catch a module pair that disagrees about a convention, a contract
signature that drifted, a constant duplicated in two places, or a hook one
module calls and the other never implemented.  That is what this file is for.

Every test here asserts a number or an exact identity.  Where a quantity is
genuinely unresolved it is pinned as an algebraic relation between the two
candidate expressions, so that changing either one reports immediately rather
than silently shifting a published figure.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
import warnings

import numpy as np
import pytest
from scipy.special import k0

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_PKG_DIR.parent), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PKG_NAME = _PKG_DIR.name

pkg = importlib.import_module(_PKG_NAME)
conductivity = importlib.import_module(f"{_PKG_NAME}.conductivity")
nonlinear = importlib.import_module(f"{_PKG_NAME}.nonlinear")
observables = importlib.import_module(f"{_PKG_NAME}.observables")
operators = importlib.import_module(f"{_PKG_NAME}.operators")
solver = importlib.import_module(f"{_PKG_NAME}.solver")
sources = importlib.import_module(f"{_PKG_NAME}.sources")
substrate = importlib.import_module(f"{_PKG_NAME}.substrate")
units = importlib.import_module(f"{_PKG_NAME}.units")

E_EV = 0.6
Q_INV_NM = 0.05


# ==========================================================================
# 1. package surface
# ==========================================================================
#: (module attribute, name) pairs of every symbol the module-contract block
#: publishes.  A rename or a dropped export fails here, loudly, in one place.
CONTRACT = {
    "units": (
        "HBAR_EV_FS E2_EV_NM C_NM_FS VF_NM_FS KB_EV_K AU_NM AU_EV "
        "SIGMA_AU_TO_NM_FS omega_of sigma_au_to_nm_fs sigma_nm_fs_to_au"
    ),
    "conductivity": (
        "graphene_drude graphene_interband_T0 graphene_local_rpa "
        "thermal_drude_weight drift_chemical_potential "
        "drift_chemical_potential_numeric graphene_drift_sigma metal_thin_film "
        "sigma_shg_abc sigma_shg_1d_q0 sigma_thg_local sigma_kerr "
        "sigma_cascade_abcd"
    ),
    "operators": (
        "struve_L coulomb_V_entries build_V build_V_toeplitz build_D build_D_geom"
    ),
    "substrate": "Layer reflection_coefficient build_Vbar",
    "sources": (
        "phi_ext_plane_wave phi_ext_dipole phi_ext_electron "
        "electron_trajectory_is_valid"
    ),
    "solver": "Ribbon RibbonSystem",
    "observables": "induced_field_q ldos_q dissymmetry",
    "nonlinear": "shg_source_rho solve_shg solve_thg",
    # dispersion.py: the Level 1 / Level 2 / Level 3 stack, added to the
    # contract 2026-11-03 when the module landed.
    "dispersion": (
        "sheet_response sheet_reflection sheet_plasmon_energy "
        "extended_sheet_eta ribbon_bands ldos_map pm_asymmetry"
    ),
}


@pytest.mark.parametrize("mod_name", sorted(CONTRACT))
def test_every_contract_symbol_exists_and_is_documented(mod_name: str) -> None:
    """Every name in the module-contract block resolves and has a docstring."""
    mod = importlib.import_module(f"{_PKG_NAME}.{mod_name}")
    for name in CONTRACT[mod_name].split():
        assert hasattr(mod, name), f"{mod_name}.{name} is missing"
        obj = getattr(mod, name)
        if callable(obj):
            assert obj.__doc__, f"{mod_name}.{name} has no docstring"


def test_package_reexports_the_whole_contract() -> None:
    """``import electrostatics_fd`` alone reaches every contract symbol."""
    missing = [
        f"{m}.{n}"
        for m, names in CONTRACT.items()
        for n in names.split()
        if not hasattr(pkg, n)
    ]
    assert not missing, f"not re-exported from the package root: {missing}"
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"__all__ advertises missing name {name}"


def test_package_reexports_every_submodule_all() -> None:
    """``__init__``'s "whole public surface is re-exported here" claim, enforced.

    The package docstring says every public name is reachable from the root.
    That was false when ``dispersion.py`` landed -- the module was not imported
    at all, so none of its 14 public names resolved -- and false for
    ``sources.e_ext_electron`` from the moment it was added.  Trusting a
    docstring is what let both through, so this test walks each sub-module's
    own ``__all__`` instead of a hand-maintained list.

    ``substrate.HBAR_EV_FS`` is reached through ``units``, which defines the
    same constant; the two are asserted bit-identical in
    ``tests/test_substrate.py::test_hbar_matches_units_module``.
    """
    missing: list[str] = []
    for mod_name in pkg.__all__:
        mod = getattr(pkg, mod_name)
        if not isinstance(mod, type(pkg)):
            continue  # a re-exported name, not a sub-module
        for name in getattr(mod, "__all__", ()):
            if not hasattr(pkg, name):
                missing.append(f"{mod_name}.{name}")
    assert not missing, (
        "the package docstring claims the whole public surface is re-exported, "
        f"but these are not reachable from the root: {missing}"
    )


CONTRACT_SIGNATURES = {
    (solver.RibbonSystem.__init__): "self ribbons eps_a eps_b layers dx_nm",
    (solver.RibbonSystem.eta_pair): "self ribbon_index q E_ev",
    (solver.RibbonSystem.build_M): "self q E_ev",
    (solver.RibbonSystem.solve_linear): "self q E_ev phi_ext",
    (solver.RibbonSystem.eigenmodes): "self q",
    (solver.RibbonSystem.polarizability): "self q E_ev phi_ext",
    (operators.build_V): (
        "q W_j W_jp x_j x_jp z_j z_jp theta_j theta_jp h_jp"
    ),
    (operators.build_V_toeplitz): "q W x_off z_off theta h",
    (operators.build_D): "q W h L eta_perp eta_par f",
    (operators.build_D_geom): "q W h L f",
    (substrate.reflection_coefficient): "Q_inv_nm E_ev layers eps1",
    (substrate.build_Vbar): (
        "q W_j W_jp x_j x_jp z_j z_jp theta_j theta_jp h_jp r_of_Q Qmax NQ"
    ),
    (observables.induced_field_q): "q x_nodes rho r0_nm",
    (observables.ldos_q): "q x_nodes rho r0_nm p",
    (observables.dissymmetry): "qs Dq",
    (sources.phi_ext_plane_wave): "x_nm E0",
    (sources.phi_ext_dipole): "q x_nm z_nm r0_nm p eps_a",
}


@pytest.mark.parametrize("fn, expected", list(CONTRACT_SIGNATURES.items()))
def test_contract_positional_parameters_are_in_order(fn, expected: str) -> None:
    """Positional parameter NAMES and ORDER match the contract exactly.

    Keyword-only extras added by a module are allowed (several modules added
    them deliberately), but a renamed or reordered positional argument would
    break every caller silently and is rejected here.
    """
    want = expected.split()
    got = [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert got[: len(want)] == want, f"{fn.__qualname__}: {got} vs contract {want}"


def test_ribbon_dataclass_fields_and_defaults() -> None:
    """``Ribbon`` field order and defaults are the contract's."""
    import dataclasses

    fields = [(f.name, f.default) for f in dataclasses.fields(solver.Ribbon)]
    assert fields == [
        ("W_nm", dataclasses.MISSING),
        ("x0_nm", 0.0),
        ("z0_nm", 0.0),
        ("EF_ev", 0.5),
        ("gamma_ev", 0.02),
        ("beta", 0.0),
        ("L", None),
        # Appended 2026-11-05, with a default, so every positional construction
        # that existed before still means what it did. Temperature belongs to
        # the sheet and not to the surroundings: two ribbons on one substrate
        # can sit at different electron temperatures.
        ("T_K", 0.0),
    ]


def test_layer_dataclass_fields_and_defaults() -> None:
    """``substrate.Layer`` field order and defaults are the contract's."""
    import dataclasses

    fields = [(f.name, f.default) for f in dataclasses.fields(substrate.Layer)]
    assert fields == [
        ("eps_par", 1.0),
        ("eps_perp", None),
        ("d_nm", 0.0),
        ("sigma_sheet", 0.0),
    ]


# ==========================================================================
# 2. constants duplicated across modules
# ==========================================================================
def test_hbar_is_one_number_everywhere() -> None:
    """Three modules carry their own hbar fallback; all must be units.HBAR_EV_FS."""
    assert substrate.HBAR_EV_FS == units.HBAR_EV_FS
    assert nonlinear.HBAR_EV_FS_FALLBACK == units.HBAR_EV_FS
    assert units.HBAR_EV_FS == 0.6582119569


def test_derived_constants_match_their_definitions() -> None:
    """The documented algebraic relations between the unit constants hold."""
    assert units.SIGMA_AU_TO_NM_FS == pytest.approx(
        units.E2_EV_NM / units.HBAR_EV_FS, rel=1e-15
    )
    assert units.VF_NM_FS == pytest.approx(units.C_NM_FS / 300.0, rel=1e-15)
    # e^2/(4 hbar), the universal interband conductivity in nm/fs
    assert conductivity.SIGMA0_NM_FS == pytest.approx(
        units.E2_EV_NM / (4.0 * units.HBAR_EV_FS), rel=1e-15
    )
    assert units.omega_of(1.0) == pytest.approx(1.0 / units.HBAR_EV_FS, rel=1e-15)


# ==========================================================================
# 3. the cascade-A question, resolved
# ==========================================================================
def _s_ref(cond, E_ev: float, gamma_ev: float, s1: int, s2: int) -> complex:
    """Paper-406 MATLAB ``S_s1_s2 = S2 D_sum (D_s2^2 + 2 D_sum (D_s1 + D_s2))``."""
    omega = units.omega_of(E_ev)
    gam_fs = gamma_ev / units.HBAR_EV_FS
    d1 = cond._D(float(s1), omega, gam_fs)
    d2 = cond._D(float(s2), omega, gam_fs)
    dsum = cond._D(float(s1 + s2), omega, gam_fs)
    return complex(cond._S2 * dsum * (d2**2 + 2.0 * dsum * (d1 + d2)))


@pytest.mark.parametrize("E_ev, gamma_ev", [(1.3, 0.025), (0.7, 0.01), (2.1, 0.05)])
def test_cascade_coefficients_reduce_correctly_at_s1_s2_equal_one(
    E_ev: float, gamma_ev: float
) -> None:
    """``s1 = s2 = 1`` IS second-harmonic generation, and identifies each expression.

    Two exact reductions, both to machine precision, and they go to DIFFERENT
    places -- which is the resolution of what both ``conductivity.py`` and
    ``nonlinear.py`` reported as a suspected upstream transcription error:

    * ``sigma_cascade_abcd(...)[0]`` -> the tensor structure ``A`` of Eq. (12a);
    * the MATLAB ``S_1_1`` -> the q = 0 1-D collapse ``A + B + 2C``.

    ``nonlinear.py`` is the 1-D q = 0 collapse, so the second is the coefficient
    it needs and its ``cascade_A_form='reference'`` default is correct.
    """
    a_tensor = conductivity.sigma_shg_abc(E_ev, gamma_ev, +1)[0]
    a_cascade = conductivity.sigma_cascade_abcd(E_ev, gamma_ev, 1, 1, +1)[0]
    assert a_cascade == pytest.approx(a_tensor, rel=1e-14)

    collapse = conductivity.sigma_shg_1d_q0(E_ev, gamma_ev, +1)
    assert -_s_ref(conductivity, E_ev, gamma_ev, 1, 1) == pytest.approx(
        collapse, rel=1e-14
    )

    b_cascade = conductivity.sigma_cascade_abcd(E_ev, gamma_ev, 1, 1, +1)[1]
    assert b_cascade == pytest.approx(
        conductivity.sigma_shg_abc(E_ev, gamma_ev, +1)[1], rel=1e-14
    )


@pytest.mark.parametrize("E_ev, gamma_ev", [(1.3, 0.025), (0.7, 0.01)])
def test_cascade_C_fails_the_shg_reduction_that_A_and_B_pass(
    E_ev: float, gamma_ev: float
) -> None:
    """UNRESOLVED, pinned: Eq. (15c) does not reduce to Eq. (12c) at s1 = s2 = 1.

    A and B reduce exactly (test above), so C must too; it does not.  The gap is
    O(1), not round-off: the measured ratio is 5.99 - 0.19i at E = 1.3 eV.  Kept
    AS GIVEN per the project rule against guessing physics, and pinned here as
    an exact algebraic statement so that a future correction to either Eq. (15c)
    or Eq. (12c) reports immediately instead of moving a figure quietly.

    The candidate reading of (15c) that WOULD reduce correctly, recorded but not
    implemented, is
    ``C = -ds S2 D_sum (2 D_s1/(s2 omega) - D_s2^2/2 - 2 D_sum D_s2)``.
    """
    c_cascade = conductivity.sigma_cascade_abcd(E_ev, gamma_ev, 1, 1, +1)[2]
    c_tensor = conductivity.sigma_shg_abc(E_ev, gamma_ev, +1)[2]
    assert abs(c_cascade - c_tensor) > 0.5 * abs(c_tensor), (
        "Eq. (15c) now reduces to Eq. (12c): the discrepancy this test pins has "
        "been resolved, so promote it to an equality assertion."
    )

    omega = units.omega_of(E_ev)
    gam_fs = gamma_ev / units.HBAR_EV_FS
    d1 = conductivity._D(1.0, omega, gam_fs)
    d2 = conductivity._D(2.0, omega, gam_fs)
    pref = -conductivity._S2 * d2 * d1
    # exactly the two brackets, so the gap is attributable term by term
    assert c_cascade == pytest.approx(pref * (4.0 / omega - d1), rel=1e-13)
    assert c_tensor == pytest.approx(
        pref * (2.0 / omega - 0.5 * d1 - 2.0 * d2), rel=1e-13
    )

    candidate = -conductivity._S2 * d2 * (
        2.0 * d1 / omega - 0.5 * d1**2 - 2.0 * d2 * d1
    )
    assert candidate == pytest.approx(c_tensor, rel=1e-13), (
        "the recorded candidate form for Eq. (15c) must at least reproduce "
        "Eq. (12c) in the degenerate limit"
    )


# ==========================================================================
# 4. solver <-> nonlinear handoff
# ==========================================================================
def _system(**kw):
    return solver.RibbonSystem([solver.Ribbon(W_nm=6.0, EF_ev=0.5)], dx_nm=0.25, **kw)


def test_nonlinear_uses_the_same_grid_the_linear_solve_used() -> None:
    """``nonlinear`` must assemble its source on the solver's own nodes.

    Without the ``nonlinear_grid`` hook, ``nonlinear._grid`` falls back to
    ``round(W/dx) + 1`` while ``RibbonSystem`` uses ``max(31, round(W/dx))``:
    25 nodes against 31 for W = 6 nm at dx = 0.25 nm.  The source would then be
    built on a different grid from the potential driving it.
    """
    system = _system()
    theta, h, L = system.nonlinear_grid(0)
    assert (theta, h, L) == (system.theta[0], system.h[0], system.L[0])

    nl_theta, nl_h, nl_L = nonlinear._grid(system, 0)
    np.testing.assert_array_equal(nl_theta, system.theta[0])
    assert nl_h == system.h[0]
    assert nl_L == system.L[0]

    phi = -system.x_nodes[0].astype(np.complex128)
    sol = system.solve_linear(0.0, E_EV, phi)
    assert nl_L == len(sol["phi"][0])

    # the fallback really is a different grid, so the hook is load bearing
    fallback_L = int(round(6.0 / 0.25)) + 1
    assert fallback_L != nl_L


def test_nonlinear_V_block_hook_matches_operators_without_a_substrate() -> None:
    """With no stack the hook is the bare Coulomb block of ``operators``.

    Bit-exactly on the dense path; on the default Toeplitz path to the ~1e-15
    absolute cancellation floor documented in ``operators.py`` (the two reach
    the same mathematical separation through different arithmetic, so even at
    ``x_off = 0`` they are not bit-identical).
    """
    q = 2.0 * Q_INV_NM
    dense = _system()
    dense.use_toeplitz = False
    want = operators.build_V(
        abs(q), 6.0, 6.0, 0.0, 0.0, 0.0, 0.0,
        dense.theta[0], dense.theta[0], dense.h[0],
    )
    np.testing.assert_allclose(
        dense.nonlinear_V_block(0, 0, q, 2.0 * E_EV), want, rtol=0.0, atol=0.0
    )

    got = _system().nonlinear_V_block(0, 0, q, 2.0 * E_EV)
    assert float(np.max(np.abs(got - want))) < 1e-13
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_nonlinear_V_block_hook_adds_substrate_screening() -> None:
    """A layered system screens its HARMONIC blocks, not just the linear one.

    This closes the gap ``nonlinear.py`` reported: with no hook it used the bare
    Coulomb kernel and merely warned.  The hook also has to take the harmonic
    ENERGY, because the reflection coefficient is frequency dependent -- so the
    2-omega block of a dispersive stack must differ from its 1-omega block.
    """
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=2.0)], layers=layers, dx_nm=0.5
    )
    system.vbar_Qmax, system.vbar_NQ = 30.0, 6000
    q = 2.0 * Q_INV_NM

    bare = operators.build_V(
        abs(q), 6.0, 6.0, 0.0, 0.0, 2.0, 2.0,
        system.theta[0], system.theta[0], system.h[0],
    )
    screened = system.nonlinear_V_block(0, 0, q, 2.0 * E_EV)
    gap = float(np.max(np.abs(screened - bare)))
    assert gap > 1e-4 * float(np.max(np.abs(bare))), (
        f"substrate screening changed the harmonic block by only {gap:.3e}"
    )
    # r = -0.6 < 0 for eps = 1 -> 4, so an image charge of opposite sign reduces
    # the self interaction.
    assert float(np.max(np.abs(screened))) < float(np.max(np.abs(bare)))

    with pytest.raises(ValueError, match="needs E_ev"):
        system.nonlinear_V_block(0, 0, q, None)


def test_nonlinear_V_block_hook_is_frequency_dependent_for_a_dispersive_stack() -> None:
    """A sheet-bearing stack gives different blocks at omega and 2 omega."""
    layers = [
        substrate.Layer(eps_par=1.0, d_nm=0.0),
        substrate.Layer(eps_par=4.0, sigma_sheet=0.5 + 0.2j),
    ]
    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=2.0)], layers=layers, dx_nm=0.5
    )
    system.vbar_Qmax, system.vbar_NQ = 30.0, 6000
    q = 2.0 * Q_INV_NM
    at_w = system.nonlinear_V_block(0, 0, q, E_EV)
    at_2w = system.nonlinear_V_block(0, 0, q, 2.0 * E_EV)
    rel = float(np.max(np.abs(at_2w - at_w))) / float(np.max(np.abs(at_w)))
    assert rel > 1e-3, (
        f"a dispersive stack must screen omega and 2 omega differently; rel {rel:.2e}"
    )


def test_layered_shg_runs_screened_and_without_the_bare_kernel_warning() -> None:
    """``solve_shg`` on a layered system is screened and silent.

    Before the hook existed this path raised a ``RuntimeWarning`` ("substrate
    screening is absent from the nonlinear blocks") and returned an unscreened
    number.
    """
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=2.0)], layers=layers, dx_nm=0.5
    )
    system.vbar_Qmax, system.vbar_NQ = 30.0, 6000
    phi = -system.x_nodes[0].astype(np.complex128)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = nonlinear.solve_shg(system, Q_INV_NM, E_EV, [phi])

    bare = solver.RibbonSystem([solver.Ribbon(W_nm=6.0, z0_nm=2.0)], dx_nm=0.5)
    out_bare = nonlinear.solve_shg(bare, Q_INV_NM, E_EV, [phi])
    assert out["alpha"] != out_bare["alpha"]
    assert np.isfinite(abs(out["alpha"]))


# ==========================================================================
# 5. cross-module convention guards
# ==========================================================================
def test_eps_a_contradicting_layers_zero_is_rejected() -> None:
    """The ambient medium is specified twice; disagreement must raise, not warn.

    ``eta`` takes it from ``eps_eff``, the reflection recursion from
    ``layers[0]``.  ``substrate.reflection_coefficient`` used to emit a
    ``UserWarning`` once per call for this, which a sweep would bury.
    """
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    with pytest.raises(ValueError, match="contradicts layers"):
        solver.RibbonSystem([solver.Ribbon(W_nm=6.0)], eps_a=2.5, layers=layers)

    # consistent inputs construct, and no warning survives to the caller
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ok = solver.RibbonSystem(
            [solver.Ribbon(W_nm=6.0, z0_nm=1.0)],
            eps_a=1.0,
            eps_b=1.0,
            layers=layers,
        )
    assert ok.eps_eff == 1.0


def test_double_counted_standoff_is_rejected() -> None:
    """``layers[0].d_nm`` and a non-zero ribbon ``z0_nm`` square the standoff."""
    layers = [substrate.Layer(eps_par=1.0, d_nm=3.0), substrate.Layer(eps_par=4.0)]
    with pytest.raises(ValueError, match="double counted"):
        solver.RibbonSystem([solver.Ribbon(W_nm=6.0, z0_nm=3.0)], layers=layers)

    # convention B (standoff only in d_nm) is still allowed
    solver.RibbonSystem([solver.Ribbon(W_nm=6.0, z0_nm=0.0)], layers=layers)
    # convention A (standoff only in z0_nm) is the recommended one
    solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=3.0)],
        layers=[substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)],
    )


def test_q_zero_with_a_substrate_raises_where_the_user_can_see_it() -> None:
    """The divergent q = 0 reflected integral must fail in the solver's own words."""
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=2.0)], layers=layers, dx_nm=0.5
    )
    phi = -system.x_nodes[0].astype(np.complex128)
    with pytest.raises(ValueError, match="logarithmically divergent"):
        system.solve_linear(0.0, E_EV, phi)

    # and a small finite q is finite
    system.vbar_Qmax, system.vbar_NQ = 30.0, 6000
    out = system.solve_linear(1e-3, E_EV, phi)
    assert np.all(np.isfinite(out["phi_flat"]))


def test_toeplitz_is_the_default_and_does_not_change_the_answer() -> None:
    """The fast Coulomb path is on by default and agrees with the dense build."""
    fast = _system()
    assert fast.use_toeplitz is True
    slow = _system()
    slow.use_toeplitz = False

    phi = -fast.x_nodes[0].astype(np.complex128)
    rho_f = fast.solve_linear(Q_INV_NM, E_EV, phi)["rho"][0]
    rho_s = slow.solve_linear(Q_INV_NM, E_EV, phi)["rho"][0]
    rel = float(np.max(np.abs(rho_f - rho_s))) / float(np.max(np.abs(rho_s)))
    assert rel < 1e-10, f"Toeplitz and dense paths disagree on rho by {rel:.2e}"


# ==========================================================================
# 6. the Struve range guard
# ==========================================================================
def test_coplanar_kernel_is_accurate_and_silent_at_every_separation() -> None:
    """The coplanar (branch 3) kernel is right at every separation, and says nothing.

    INVERTED FROM A WARNING TEST (2026-11-03).  This used to assert that
    ``operators`` emits a ``RuntimeWarning`` matching 'modified-Struve' for a
    far coplanar pair, because the bare 30-term series behind branch 3 degraded
    without failing (7e-8 relative at argument 10, 45% at 30) and the block hit
    an ABSOLUTE cancellation floor of 9.068e-15 -- against a true coupling
    K_0(50) = 3.410e-23, i.e. wrong by 4.7e8.

    ``operators`` now forms branch 3 from the complementary function
    Fc = 1 - F, evaluated in four regions, so the cancellation is gone and
    there is nothing left to warn about: ``_warn_struve_range`` was deleted
    rather than re-scoped.  This test therefore asserts SILENCE everywhere and,
    more usefully, the ACCURACY that silence is now claiming -- a bare
    no-warning check would pass just as well against a builder that returned
    zeros.

    ``STRUVE_TRUSTED_MAX_ARG`` is deliberately still 8.0; its documented meaning
    changed from "trust ceiling of the Coulomb block" to "ceiling of the bare
    30-term series", which is why the constant is asserted but the warning is
    not.
    """
    L = 40
    theta = np.linspace(-0.5, 0.5, L)
    h = 1.0 / (L - 1)
    W = 10.0

    # inside the trusted range: silent, as it always was
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        operators.build_V(0.5 / W, W, W, 0.0, 0.0, 0.0, 0.0, theta, theta, h)
        operators.build_V_toeplitz(0.5 / W, W, 0.0, 0.0, theta, h)

    # far coplanar pair: now SILENT, and correct
    q, d = 0.5, 100.0  # q*d = 50
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        V = np.asarray(
            operators.build_V(q, W, W, d, 0.0, 0.0, 0.0, theta, theta, h)
        )
        Vt = np.asarray(operators.build_V_toeplitz(q, W, d, 0.0, theta, h))

    # It must track the exact coplanar kernel 2 h K_0(q|dx|) to the cell-integral
    # vs cell-midpoint difference (Q h)^2/24 and no worse.  Measured max rel
    # 7.004e-04 against (Q h)^2/24 = 6.849e-04 -- the quadrature difference
    # itself, not an error.  Before the fix this block was flat at 9.068e-15
    # ABSOLUTE, so the magnitude assertion below is the one that moved.
    dx = (d + W * theta[:, None]) - (0.0 + W * theta[None, :])
    ref = 2.0 * h * k0(q * np.abs(dx))
    rel = float(np.max(np.abs(V - ref) / np.abs(ref)))
    midpoint_gap = (q * W * h) ** 2 / 24.0
    print(f"\nfar coplanar cross block (q*d = {q*d:g}): |V| in "
          f"{np.abs(V).min():.4e} .. {np.abs(V).max():.4e}; "
          f"max rel vs 2 h K_0 = {rel:.4e}, (Q h)^2/24 = {midpoint_gap:.4e}")
    assert rel < 2.0 * midpoint_gap, (
        f"coplanar kernel off by {rel:.3e}, more than the {midpoint_gap:.3e} "
        "cell-integral/midpoint difference can explain"
    )
    # exponentially small, not parked on the old 9.07e-15 floor
    assert np.max(np.abs(V)) < 1e-20
    assert np.max(np.abs(V)) > 1e-30
    # and the Toeplitz fast path agrees there too
    assert np.max(np.abs(V - Vt)) / np.max(np.abs(V)) < 1e-10

    # the same pair offset in z takes the exact K_0 branch and stays silent
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        operators.build_V(
            0.5, W, W, 100.0, 0.0, 5.0, 0.0, theta, theta, h
        )

    assert operators.STRUVE_TRUSTED_MAX_ARG == 8.0


def test_far_coplanar_pair_decouples_through_the_solver() -> None:
    """The system the old Struve guard was written for now decouples instead.

    INVERTED FROM A WARNING TEST (2026-11-03).  This used to assert
    ``pytest.warns(RuntimeWarning, match='not trustworthy')`` around
    ``build_M``; that warning no longer exists and can never fire again, so the
    test asserts the physics it was standing in for.  Two coplanar ribbons 100
    nm apart at q = 0.5/nm are at q*d = 50, where the true Coulomb coupling is
    K_0(50) = 3.410e-23: ``build_M`` must come out block diagonal to that level.

    MEASURED: max |off-diagonal block| / max |diagonal block| = 2.907e-22
    (off-diagonal 3.727e-21, diagonal 1.282e+01).  Before the branch-3 fix the
    cross block was pinned at the 9.068e-15 absolute cancellation floor and the
    pair did not decouple at all.
    """
    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=10.0, x0_nm=-50.0), solver.Ribbon(W_nm=10.0, x0_nm=50.0)],
        dx_nm=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        M = np.asarray(system.build_M(0.5, E_EV))

    s0, s1 = system.slices[0], system.slices[1]
    off = max(float(np.max(np.abs(M[s0, s1]))), float(np.max(np.abs(M[s1, s0]))))
    dia = max(float(np.max(np.abs(M[s0, s0]))), float(np.max(np.abs(M[s1, s1]))))
    print(f"\nbuild_M(0.5, {E_EV}) at q*d = 50: max|off-block| = {off:.4e}, "
          f"max|diag block| = {dia:.4e}, ratio = {off/dia:.4e} "
          f"(K_0(50) = {float(k0(50.0)):.3e})")
    assert off / dia < 1e-18, (
        f"coplanar ribbons at q*d = 50 fail to decouple: off/diag = {off/dia:.3e}"
    )


# ==========================================================================
# 7. the LAYERED path, end to end  (substrate.build_Vbar through the solver)
# ==========================================================================
# Every solver-level substrate test before this section used
# ``[Layer(1.0), Layer(1.0)]``, i.e. r = 0, which exercises the plumbing and no
# physics at all.  The tests below drive a REFLECTING stack through
# ``build_M`` / ``solve_linear`` / ``eigenmodes_screened`` and pin the physics
# the reflected block is supposed to produce.
#
# Every one of them keeps the standoff FINITE.  ``substrate.build_Vbar``'s
# integrand decays only as 1/Q when z_j + z_jp = 0 and ``layers[0].d_nm = 0``,
# so its uniform trapezoid rule caps the accuracy at ~1e-3 there however large
# NQ is; 1 nm already gives ~1e-11.  That is an upstream quadrature limitation,
# measured and documented in ``substrate.py``.

#: geometry shared by the layered tests.  qW = 0.5 is inside the trusted range
#: of every kernel branch (``operators`` wants q*W <~ 6 for one ribbon).
L_W_NM = 10.0
L_Z_NM = 1.0
L_Q = 0.05
L_L = 61
L_EF = 0.5
L_GAMMA = 0.005

#: ``build_Vbar`` quadrature for the layered tests.  Qmax*z_sum = 400 >> 20 and
#: dQ = Qmax/(NQ-1) ~ 0.01 = |q|/5 < 0.3|q|, so neither of ``build_Vbar``'s two
#: resolution warnings fires and the block is converged (see its Notes).
L_QMAX = 200.0
L_NQ = 20000


def _vac_layers():
    return [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=1.0)]


def _layered(layers, z_nm: float = L_Z_NM, W_nm: float = L_W_NM):
    """One ribbon at a finite standoff over ``layers`` (convention A)."""
    system = solver.RibbonSystem(
        [
            solver.Ribbon(
                W_nm=W_nm, z0_nm=z_nm, EF_ev=L_EF, gamma_ev=L_GAMMA, L=L_L
            )
        ],
        layers=layers,
    )
    system.vbar_Qmax = L_QMAX
    system.vbar_NQ = L_NQ
    return system


def _bright_eta1(system, q: float = L_Q, E_ev: float = 0.6) -> float:
    """Most negative BRIGHT geometric eigenvalue, screened when layers are set."""
    if system.layers is None:
        eta, _, zeta = system.eigenmodes(q)
    else:
        eta, _, zeta = system.eigenmodes_screened(q, E_ev)
    mask = solver.bright_mode_mask(zeta)
    return float(np.real(eta[mask][0]))


def _mode_energy(eta: float, W_nm: float = L_W_NM, eps_eff: float = 1.0) -> float:
    """``E_m = sqrt(-e^2 E_F / (pi eta_m W eps_eff))`` (eigenmodes_screened)."""
    return float(np.sqrt(-units.E2_EV_NM * L_EF / (np.pi * eta * W_nm * eps_eff)))


def test_vacuum_stack_reproduces_the_free_standing_solve_bit_for_bit() -> None:
    """r -> 0 continuity, and it is EXACT, not approximate.

    ``substrate.build_Vbar`` short-circuits ``r == 0`` on the whole quadrature
    grid to exact zeros, and adding exact zeros to V changes no bit.  So the
    layered code path and the free-standing one must agree to the last bit --
    not to a tolerance.  Anything less would mean the layered path perturbs the
    answer with no substrate present, which would make every number in this
    section unattributable.
    """
    free = _layered(None)
    vac = _layered(_vac_layers())

    assert np.array_equal(vac._Vbar_full(L_Q, 0.6), np.zeros((vac.N, vac.N)))
    assert _bright_eta1(vac) == _bright_eta1(free)

    phi = -free.x_nodes[0].astype(np.complex128)
    sol_f = free.solve_linear(L_Q, 0.6, phi)
    sol_v = vac.solve_linear(L_Q, 0.6, phi)
    np.testing.assert_array_equal(sol_v["rho_flat"], sol_f["rho_flat"])
    np.testing.assert_array_equal(sol_v["phi_flat"], sol_f["phi_flat"])
    assert vac.polarizability(L_Q, 0.6, phi) == free.polarizability(L_Q, 0.6, phi)

    # ... and eigenmodes_screened with layers=None IS eigenmodes
    bare = _layered(None)
    e_a, v_a, z_a = bare.eigenmodes(L_Q)
    e_b, v_b, z_b = bare.eigenmodes_screened(L_Q, 0.6)
    np.testing.assert_array_equal(e_a, e_b)
    np.testing.assert_array_equal(v_a, v_b)
    np.testing.assert_array_equal(z_a, z_b)


def test_substrate_red_shifts_the_plasmon_monotonically_in_eps() -> None:
    """A dielectric substrate pushes every eta_m down and every E_m down with it.

    r < 0 for eps_substrate > 1, so the image charge has the opposite sign, it
    weakens the self interaction, and the mode needs a MORE negative eta --
    reached at a lower energy -- to resonate.  Measured eta_1 at W = 10 nm,
    z0 = 1 nm, qW = 0.5: -0.06996, -0.08376, -0.09953, -0.12017 at
    eps = 1, 2, 4, 11.7, i.e. E_1 = 0.5723, 0.5231, 0.4799, 0.4367 eV.
    """
    etas, energies = [], []
    for eps in (1.0, 1.5, 2.0, 4.0, 11.7):
        layers = [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=eps)]
        eta = _bright_eta1(_layered(layers))
        etas.append(eta)
        energies.append(_mode_energy(eta))

    assert all(b < a for a, b in zip(etas, etas[1:])), f"not monotone: {etas}"
    assert all(b < a for a, b in zip(energies, energies[1:])), energies

    assert etas[0] == pytest.approx(-0.0699604, abs=2e-6)
    assert etas[3] == pytest.approx(-0.0995316, abs=2e-6)
    assert energies[0] == pytest.approx(0.572347, abs=2e-5)
    assert energies[3] == pytest.approx(0.479850, abs=2e-5)

    # the eps = 1 end of the sweep is the free-standing answer exactly
    assert etas[0] == _bright_eta1(_layered(None))


def test_perfect_mirror_is_the_extreme_and_it_saturates() -> None:
    """eps -> infinity gives r -> -1, the largest shift a passive stack can make.

    The image-charge limit: at r = -1 the reflected kernel is minus the direct
    one evaluated at the image separation, and nothing further happens as eps
    grows.  Measured eta_1 -> -0.1389309, approached from above as eps climbs,
    with the remaining gap falling by ~1e-2 per two decades of eps: 1.91e-2 at
    eps = 100, 1.97e-4 at 1e4, 1.97e-6 at 1e6 (relative).  It must also bound
    the shift of every finite eps, which is the statement that r is monotone.
    """

    def eta_at(eps):
        return _bright_eta1(
            _layered([substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=eps)])
        )

    e100, e4, e6, e8 = eta_at(100.0), eta_at(1e4), eta_at(1e6), eta_at(1e8)

    assert e8 == pytest.approx(-0.1389309, abs=2e-6)
    # saturation: each two decades of eps closes the remaining gap ~100-fold
    gaps = [abs(e - e8) / abs(e8) for e in (e100, e4, e6)]
    assert gaps == pytest.approx([1.907e-2, 1.965e-4, 1.965e-6], rel=2e-2)
    assert all(b < 0.02 * a for a, b in zip(gaps, gaps[1:])), gaps

    # r really is -1 there, and the mirror bounds eps = 11.7
    r = substrate.reflection_coefficient(
        1.0 / L_W_NM, 0.6, [substrate.Layer(1.0), substrate.Layer(eps_par=1e8)]
    )
    assert complex(r) == pytest.approx(-1.0, abs=1e-7)
    assert e8 < eta_at(11.7) < _bright_eta1(_layered(None)), (
        "the mirror must shift further than any finite dielectric"
    )


def test_screening_dies_with_standoff_faster_than_the_image_exponential() -> None:
    """Lift the ribbon and the substrate stops mattering -- and it must vanish.

    The rigorous envelope is ``exp(-2 q z)``: ``build_Vbar``'s image factor is
    ``exp(-Q (z_j + z_jp)) = exp(-2 Q z)`` and the quadrature runs over
    ``Q = sqrt(Qx^2 + q^2) >= |q|``, so no part of the reflected block can decay
    more slowly than the q component.  Asserting that -- rather than "roughly
    exponentially in q z" -- is what the geometry actually implies, and the
    measured decay is strictly faster at every doubling because these blocks are
    dominated by Q ~ 1/W >> q, not by Q ~ q.

    Measured eta_1 shift against the free-standing value: -2.96e-2 at z = 1 nm,
    -1.27e-3 at 8 nm, -1.82e-7 at 64 nm, -3.9e-14 at 200 nm.
    """
    free = _bright_eta1(_layered(None))
    zs = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    shifts = []
    for z in zs:
        system = _layered(
            [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)], z_nm=z
        )
        system.vbar_Qmax = max(L_QMAX, 60.0 / z)
        shifts.append(_bright_eta1(system) - free)

    assert all(s < 0.0 for s in shifts), f"a substrate must red shift: {shifts}"
    assert all(abs(b) < abs(a) for a, b in zip(shifts, shifts[1:])), shifts
    assert shifts[0] == pytest.approx(-2.957e-2, rel=2e-3)
    assert shifts[3] == pytest.approx(-1.272e-3, rel=2e-3)
    assert shifts[6] == pytest.approx(-1.825e-7, rel=1e-2)

    for (z0, s0), (z1, s1) in zip(zip(zs, shifts), zip(zs[1:], shifts[1:])):
        envelope = float(np.exp(-2.0 * L_Q * (z1 - z0)))
        assert abs(s1) < envelope * abs(s0), (
            f"z {z0} -> {z1}: shift ratio {abs(s1 / s0):.3e} exceeds the "
            f"exp(-2 q dz) = {envelope:.3e} envelope that Q >= |q| guarantees"
        )

    # the limit itself: at 200 nm the substrate is gone to 1e-12
    far = _layered(
        [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)], z_nm=200.0
    )
    assert abs(_bright_eta1(far) - free) < 1e-12


def test_anisotropic_half_space_is_exactly_its_eps_geo_isotropic_twin() -> None:
    """The ``eps_perp`` path, which no solver-level test had ever reached.

    For a TERMINATING half-space the recursion sees the layer only through
    ``eps_geo = sqrt(eps_par eps_perp)``: ``aniso_factor`` enters solely via
    ``exp(-2 Qhat d)``, and d is ignored for the last layer.  So an hBN-like
    (6.7, 3.6) half-space must be BIT-IDENTICAL to an isotropic
    ``eps = sqrt(6.7*3.6) = 4.9112`` one.  That is an identity, not a tolerance,
    and it is the cleanest available check that the anisotropic branch is wired
    to the right place.

    A FINITE slab is different, because there ``exp(-2 Qhat_2 d_2)`` does carry
    ``sqrt(eps_par/eps_perp)`` -- so swapping eps_par and eps_perp changes the
    answer (measured -0.1040394 against -0.1037810 at d = 20 nm) while leaving
    eps_geo alone, and the slab converges to the half-space as d grows.
    """
    eps_geo = float(np.sqrt(6.7 * 3.6))
    iso = _bright_eta1(
        _layered([substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=eps_geo)])
    )
    ani = _bright_eta1(
        _layered(
            [
                substrate.Layer(1.0, d_nm=0.0),
                substrate.Layer(eps_par=6.7, eps_perp=3.6),
            ]
        )
    )
    assert ani == iso, (
        "an anisotropic half-space must be its eps_geo isotropic twin exactly; "
        f"got {ani!r} vs {iso!r}"
    )
    assert iso == pytest.approx(-0.1040710, abs=2e-6)

    # finite slab: the anisotropy is now visible, and in eps_par/eps_perp only
    def slab(ep, ez, d_nm):
        return _bright_eta1(
            _layered(
                [
                    substrate.Layer(1.0, d_nm=0.0),
                    substrate.Layer(eps_par=ep, eps_perp=ez, d_nm=d_nm),
                    substrate.Layer(eps_par=1.0),
                ]
            )
        )

    s_fwd = slab(6.7, 3.6, 20.0)
    s_rev = slab(3.6, 6.7, 20.0)
    assert s_fwd != s_rev, (
        "exp(-2*Qhat*d) carries sqrt(eps_par/eps_perp), so swapping the two at "
        "fixed eps_geo must change a finite slab"
    )
    assert s_fwd == pytest.approx(-0.1040394, abs=2e-6)
    assert s_rev == pytest.approx(-0.1037810, abs=2e-6)

    # and the slab becomes the half-space as it thickens
    rels = [abs(slab(6.7, 3.6, d) - iso) / abs(iso) for d in (5.0, 20.0, 50.0, 200.0)]
    assert all(b < a for a, b in zip(rels, rels[1:])), rels
    assert rels[0] == pytest.approx(1.85e-2, rel=0.05)
    assert rels[-1] < 1e-12


def test_screened_mode_energy_closed_form_matches_the_polarizability_peak() -> None:
    """``eigenmodes_screened`` and the layered ``polarizability`` are consistent.

    The eigen route and the solved route are different code paths through the
    same reflected block, so their agreement is the end-to-end check.  The
    closed form ``E_m = sqrt(-e^2 E_F/(pi eta_m W eps_eff))`` is the gamma = 0
    resonance, so the damped Im-peak must sit BELOW it by exactly the O(gamma^2)
    Lorentzian shift.  Measured relative gap -2.17e-4, -5.43e-5, -1.36e-5,
    -3.39e-6 at gamma = 20, 10, 5, 2.5 meV -- a clean -gamma^2/(8 E_m^2), the
    coefficient 1/8 holding to three digits over a factor 8 in gamma.  That LAW
    is asserted here rather than a bound, because a bound would hide a mode
    energy drifting with the stack.
    """
    layers = [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    for gamma in (0.02, 0.005):
        system = solver.RibbonSystem(
            [
                solver.Ribbon(
                    W_nm=L_W_NM, z0_nm=L_Z_NM, EF_ev=L_EF, gamma_ev=gamma, L=L_L
                )
            ],
            layers=layers,
        )
        system.vbar_Qmax, system.vbar_NQ = L_QMAX, L_NQ
        E_pred = _mode_energy(_bright_eta1(system))

        phi = -system.x_nodes[0].astype(np.complex128)
        Es = np.linspace(E_pred - 0.01, E_pred + 0.01, 2001)
        amp = np.array(
            [abs(system.polarizability(L_Q, float(E), phi).imag) for E in Es]
        )
        i = int(np.argmax(amp))
        assert 0 < i < amp.size - 1, "the peak must be inside the scanned window"
        # parabolic refinement of the sampled maximum
        frac = 0.5 * (amp[i - 1] - amp[i + 1]) / (
            amp[i - 1] - 2.0 * amp[i] + amp[i + 1]
        )
        E_peak = float(Es[i] + frac * (Es[1] - Es[0]))

        rel = (E_peak - E_pred) / E_pred
        law = -(gamma**2) / (8.0 * E_pred**2)
        assert rel < 0.0, "damping must red shift the Im-peak off the resonance"
        assert rel == pytest.approx(law, rel=0.05), (
            f"gamma = {gamma}: measured gap {rel:.3e} vs -gamma^2/(8 E^2) = {law:.3e}"
        )


def test_zero_standoff_warns_only_when_the_quadrature_is_actually_degraded() -> None:
    """The zero-standoff guard, both halves of it.

    It must fire for a reflecting stack at zero standoff -- where
    ``build_Vbar``'s 1/Q integrand caps the accuracy at ~1e-3 for any NQ -- and
    it must stay silent otherwise, or the ``[Layer(1), Layer(1)]`` plumbing
    tests and every convention-B system would drown in it.
    """

    def vbar_warnings(layers, z_nm):
        system = solver.RibbonSystem(
            [solver.Ribbon(W_nm=L_W_NM, z0_nm=z_nm, L=31)], layers=layers
        )
        system.vbar_Qmax, system.vbar_NQ = 100.0, 4000
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            system._Vbar_full(L_Q, 0.6)
        return [w for w in rec if "zero standoff" in str(w.message)]

    reflecting = [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    hit = vbar_warnings(reflecting, 0.0)
    assert len(hit) == 1 and hit[0].category is UserWarning
    assert "convention A" in str(hit[0].message)

    # a finite standoff cures the quadrature, so no warning
    assert not vbar_warnings(reflecting, 1.0)
    # a transparent stack runs no quadrature at all
    assert not vbar_warnings(_vac_layers(), 0.0)
    # convention B puts the standoff in exp(-2*Qhat_1*d_1) instead
    assert not vbar_warnings(
        [substrate.Layer(1.0, d_nm=1.0), substrate.Layer(eps_par=4.0)], 0.0
    )


def _pair_over_substrate(sep_nm: float):
    system = solver.RibbonSystem(
        [
            solver.Ribbon(W_nm=L_W_NM, x0_nm=-0.5 * sep_nm, z0_nm=L_Z_NM, L=31),
            solver.Ribbon(W_nm=L_W_NM, x0_nm=+0.5 * sep_nm, z0_nm=L_Z_NM, L=31),
        ],
        layers=[substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)],
    )
    system.vbar_Qmax, system.vbar_NQ = L_QMAX, L_NQ
    return system


def test_substrate_couples_two_ribbons_through_their_images() -> None:
    """The reflected CROSS block, which the single-ribbon tests cannot reach.

    Ribbon j' polarises the substrate and ribbon j feels that image, so Vbar is
    a full matrix, not block diagonal.  Measured at W = 10 nm, z0 = 1 nm,
    eps = 4, centre separation 20 nm: the cross block peaks at 37% of the
    diagonal one, and it decays with separation (3.63e-2, 8.52e-3, 7.83e-4 at
    20, 40, 80 nm) because the image interaction is still a Coulomb tail.

    Ribbons offset only in x take the modified-Struve branch of the DIRECT
    kernel, trusted to q*d <~ 4 (concern 6 in ``solver``); q*d = 1, 2, 4 here.
    """
    system = _pair_over_substrate(20.0)
    Vb = system._Vbar_full(L_Q, 0.6)
    diag = Vb[system.slices[0], system.slices[0]]
    cross = Vb[system.slices[0], system.slices[1]]
    assert float(np.max(np.abs(cross))) > 0.1 * float(np.max(np.abs(diag))), (
        "the substrate must couple the two ribbons; Vbar is not block diagonal"
    )
    # reciprocity: equal widths and grids, so the two cross blocks transpose
    np.testing.assert_allclose(
        cross, Vb[system.slices[1], system.slices[0]].T, rtol=1e-12, atol=0.0
    )

    peaks = []
    for sep in (20.0, 40.0, 80.0):
        s = _pair_over_substrate(sep)
        peaks.append(
            float(np.max(np.abs(s._Vbar_full(L_Q, 0.6)[s.slices[0], s.slices[1]])))
        )
    assert all(b < a for a, b in zip(peaks, peaks[1:])), peaks
    assert peaks[0] == pytest.approx(3.633e-2, rel=1e-2)

    # and the two-ribbon layered solve itself runs and is screened
    phi = [-system.x_nodes[j].astype(np.complex128) for j in range(2)]
    out = system.solve_linear(L_Q, 0.6, phi)
    assert np.all(np.isfinite(out["rho_flat"]))
    bare = solver.RibbonSystem(
        [
            solver.Ribbon(W_nm=L_W_NM, x0_nm=-10.0, z0_nm=L_Z_NM, L=31),
            solver.Ribbon(W_nm=L_W_NM, x0_nm=+10.0, z0_nm=L_Z_NM, L=31),
        ]
    )
    out_bare = bare.solve_linear(L_Q, 0.6, phi)
    rel = float(np.max(np.abs(out["rho_flat"] - out_bare["rho_flat"]))) / float(
        np.max(np.abs(out_bare["rho_flat"]))
    )
    assert rel > 1e-2, f"the layered two-ribbon solve is unscreened (rel {rel:.2e})"


# ==========================================================================
# 8. the Vbar cache: the premise the layered speed-up rests on
# ==========================================================================
# ``solver._Vbar_full`` collapses its cache key from (q, E) to (q, 0.0) when the
# stack carries no conducting sheet below the first layer.  If that premise were
# wrong for any stack the collapse accepts, the cache would return a Vbar built
# at the WRONG frequency and every layered harmonic would be silently wrong --
# far worse than being slow.  These tests check the premise at the source, then
# through the solver, then catch a mis-scoped collapse from the outside.

_DISPERSIVE_PROBES = [
    # (name, layers factory, expected _stack_is_dispersive)
    ("vacuum", _vac_layers, False),
    (
        "dielectric half-space",
        lambda: [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)],
        False,
    ),
    (
        "anisotropic slab on Si",
        lambda: [
            substrate.Layer(1.0, d_nm=0.0),
            substrate.Layer(eps_par=6.7, eps_perp=3.6, d_nm=5.0),
            substrate.Layer(eps_par=11.7),
        ],
        False,
    ),
    (
        "lossy dielectric",
        lambda: [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0 + 0.3j)],
        False,
    ),
    (
        "sheet on layers[0] only",
        lambda: [
            substrate.Layer(1.0, d_nm=0.0, sigma_sheet=0.7),
            substrate.Layer(eps_par=4.0),
        ],
        False,
    ),
    (
        "sheet at the top interface",
        lambda: [
            substrate.Layer(1.0, d_nm=0.0),
            substrate.Layer(eps_par=4.0, sigma_sheet=0.5 + 0.2j),
        ],
        True,
    ),
    (
        "buried sheet",
        lambda: [
            substrate.Layer(1.0, d_nm=0.0),
            substrate.Layer(eps_par=4.0, d_nm=5.0),
            substrate.Layer(eps_par=11.7, sigma_sheet=0.3),
        ],
        True,
    ),
]


@pytest.mark.parametrize(
    "name, make_layers, dispersive",
    _DISPERSIVE_PROBES,
    ids=[p[0] for p in _DISPERSIVE_PROBES],
)
def test_reflection_depends_on_energy_only_through_the_sheet_term(
    name: str, make_layers, dispersive: bool
) -> None:
    """``r(Q, E)`` is EXACTLY E-independent without a sheet below layers[0].

    This is the premise of the (q, E) -> (q, 0.0) cache collapse in
    ``solver._Vbar_full``, and it has to be an identity rather than an
    approximation for the collapse to be legitimate.
    ``substrate.reflection_coefficient`` touches ``E_ev`` in exactly one place,
    the sheet term ``4 pi sigma/(1j omega (eg_l + eg_l+1))``, gated on
    ``any(sigma_sheet != 0 for layers[1:])`` -- and ``layers[0].sigma_sheet`` is
    excluded on both sides, because there is no interface above layer 1.

    So r must agree bit for bit across nearly two decades of energy, and
    ``solver._stack_is_dispersive`` must classify the same stacks the same way.
    A disagreement here is exactly the mis-scoped cache this section guards.
    """
    layers = make_layers()
    Q = np.linspace(0.01, 80.0, 17)
    r_lo = substrate.reflection_coefficient(Q, 0.05, layers)
    r_hi = substrate.reflection_coefficient(Q, 3.0, layers)

    if dispersive:
        rel = float(np.max(np.abs(r_hi - r_lo))) / float(np.max(np.abs(r_lo)))
        assert rel > 1e-4, (
            f"{name}: a sheet must make r frequency dependent (rel {rel:.1e})"
        )
    else:
        np.testing.assert_array_equal(r_hi, r_lo)

    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=L_W_NM, z0_nm=L_Z_NM, L=31)], layers=layers
    )
    assert system._stack_is_dispersive() is dispersive, name


def test_vbar_cache_collapses_the_energy_only_for_a_non_dispersive_stack() -> None:
    """The cache-scope test: this is the one that catches the dangerous case.

    A purely dielectric stack must give the SAME Vbar at two energies (and the
    cache may therefore keep one entry under ``(q, 0.0)``).  A sheet-bearing
    stack must give DIFFERENT ones, under two separate ``(q, E)`` entries -- if
    the collapse were mis-scoped it would hand back the first energy's matrix,
    and this assertion is what notices.
    """
    q = L_Q
    diel = _layered([substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)])
    a = diel._Vbar_full(q, 0.3)
    b = diel._Vbar_full(q, 2.7)
    np.testing.assert_array_equal(a, b)
    # One entry, and its energy slot zeroed.  The key is _vbar_cache_key's wide
    # tuple (it also carries the layer contents, eps_a, the quadrature controls
    # and the geometry), so this asserts the COLLAPSE rather than the whole key:
    # slot 0 is |q| and slot 1 is the energy.
    assert len(diel._vbar_cache) == 1
    (only,) = diel._vbar_cache
    assert (only[0], only[1]) == (q, 0.0)

    # ... and it is a genuine physical identity, not just a cache hit: rebuild
    # from scratch at the second energy with the cache emptied.
    diel._vbar_cache.clear()
    np.testing.assert_array_equal(diel._Vbar_full(q, 2.7), a)

    disp = _layered(
        [
            substrate.Layer(1.0, d_nm=0.0),
            substrate.Layer(eps_par=4.0, sigma_sheet=0.5 + 0.2j),
        ]
    )
    c = disp._Vbar_full(q, 0.3)
    d = disp._Vbar_full(q, 2.7)
    rel = float(np.max(np.abs(d - c))) / float(np.max(np.abs(c)))
    assert rel > 1e-3, (
        f"a dispersive stack returned the same Vbar at 0.3 and 2.7 eV (rel "
        f"{rel:.2e}): the cache key has collapsed (q, E) where it must not"
    )
    assert sorted((k[0], k[1]) for k in disp._vbar_cache) == [(q, 0.3), (q, 2.7)]

    # the screened eigenvalues inherit both behaviours
    assert diel.eigenmodes_screened(q, 0.3)[0][0] == (
        diel.eigenmodes_screened(q, 2.7)[0][0]
    )
    assert disp.eigenmodes_screened(q, 0.3)[0][0] != (
        disp.eigenmodes_screened(q, 2.7)[0][0]
    )


def test_vbar_cache_is_bounded_so_a_sweep_cannot_leak() -> None:
    """``VBAR_CACHE_MAX`` caps the memo at 8 (N, N) matrices.

    Unbounded, a dispersive 10 x 200 (q, E) sweep at L = 150 would retain
    2000 * 150**2 * 16 B ~ 0.7 GB of reflected blocks.  Eviction is oldest
    first, which matches the sweep access pattern (inner loop over E).
    """
    assert solver.VBAR_CACHE_MAX == 8
    system = _layered(
        [
            substrate.Layer(1.0, d_nm=0.0),
            substrate.Layer(eps_par=4.0, sigma_sheet=0.5 + 0.2j),
        ]
    )
    # a cheap but still fully resolved quadrature: dQ = |q|/10, Qmax*z_sum = 40
    system.vbar_Qmax, system.vbar_NQ = 20.0, 4000
    energies = np.linspace(0.3, 1.2, 20)
    for E in energies:
        system._Vbar_full(L_Q, float(E))
    assert len(system._vbar_cache) == solver.VBAR_CACHE_MAX
    # oldest first: the last 8 energies survive, the first 12 are gone
    kept = sorted(k[1] for k in system._vbar_cache)
    assert kept == pytest.approx(list(energies[-8:]))


def test_vbar_cache_sees_every_input_it_depends_on() -> None:
    """A key blind to the layer stack silently returns the WRONG matrix.

    This is the companion to the energy-scope test above, for the axes that
    test did not cover.  ``layers``, ``eps_a``, ``vbar_Qmax`` and ``vbar_NQ``
    are plain public attributes (the last two are documented tunables), and
    ``Ribbon`` is a mutable dataclass whose geometry ``build_Vbar`` reads live.
    Keyed on ``(q, E)`` alone, retuning any of them returned the previous
    matrix bit-for-bit with no symptom: measured, swapping the substrate from
    eps = 4 to eps = 12 at fixed (q, E) handed back the eps = 4 block, a 29%
    error.  A dispersive dielectric can only be written by swapping ``Layer``
    objects, since ``Layer.eps_par`` is a constant, so this is the ordinary way
    to sweep a real substrate.

    Each axis is checked the same way: perturb it, and require the matrix to
    change AND to equal what a system built that way from scratch gives.
    """
    q = L_Q

    def fresh(eps=4.0, NQ=L_NQ, z_nm=L_Z_NM):
        s = _layered(
            [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=eps)], z_nm=z_nm
        )
        s.vbar_NQ = NQ          # _layered already set L_QMAX / L_NQ
        return s

    base = fresh()
    ref = base._Vbar_full(q, 0.6).copy()

    # 1. the layer stack
    base.layers = [substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=12.0)]
    got = base._Vbar_full(q, 0.6)
    rel = float(np.max(np.abs(got - ref))) / float(np.max(np.abs(ref)))
    assert rel > 1e-3, (
        f"swapping eps 4 -> 12 returned the old Vbar (rel {rel:.2e}): the cache "
        f"key does not see the layer stack"
    )
    np.testing.assert_array_equal(got, fresh(eps=12.0)._Vbar_full(q, 0.6))

    # 2. the quadrature controls -- a convergence check must not be a no-op
    b = fresh()
    b._Vbar_full(q, 0.6)
    b.vbar_NQ = 2 * L_NQ        # the realistic action: refine, not coarsen
    np.testing.assert_array_equal(
        b._Vbar_full(q, 0.6), fresh(NQ=2 * L_NQ)._Vbar_full(q, 0.6)
    )
    assert len(b._vbar_cache) == 2, "retuning NQ did not add a cache entry"

    # 3. the geometry read live out of the mutable Ribbon
    c = fresh()
    c._Vbar_full(q, 0.6)
    c.ribbons[0].z0_nm = 2.0 * L_Z_NM
    got_c = c._Vbar_full(q, 0.6)
    rel_c = float(np.max(np.abs(got_c - ref))) / float(np.max(np.abs(ref)))
    assert rel_c > 1e-3, f"doubling z0 returned the old Vbar (rel {rel_c:.2e})"
    np.testing.assert_array_equal(
        got_c, fresh(z_nm=2.0 * L_Z_NM)._Vbar_full(q, 0.6)
    )


def test_vbar_is_even_in_q_so_the_cache_keys_on_its_modulus() -> None:
    """``Vbar(-q) == Vbar(+q)`` bit-for-bit, which is why the key uses |q|.

    ``_Vbar_full`` passes ``abs(float(q))`` to ``build_Vbar`` and ``r_of_Q``
    sees only the quadrature's own ``Q``, so the evenness is exact rather than
    numerical.  ``observables.dissymmetry`` evaluates +-q, so keying on the
    modulus halves its quadrature work; the assertion is bit-equality because
    anything weaker would hide a sign leaking into the kernel.
    """
    q = L_Q
    a = _layered([substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)])
    plus = a._Vbar_full(+q, 0.6).copy()
    a._vbar_cache.clear()
    minus = a._Vbar_full(-q, 0.6).copy()
    np.testing.assert_array_equal(plus, minus)

    # and it is one cache entry, not two
    b = _layered([substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)])
    b._Vbar_full(+q, 0.6)
    b._Vbar_full(-q, 0.6)
    assert len(b._vbar_cache) == 1
    (only,) = b._vbar_cache
    assert only[0] == abs(q)


@pytest.mark.perf
def test_vbar_per_q_cache_speeds_up_a_layered_frequency_sweep() -> None:
    """Re-measured, not inherited: the per-q collapse is worth ~27x at 40 points.

    Measured here (W = 10 nm, z0 = 1 nm, eps = 4, NQ = 4000, 40 energies):
    1.50 s -> 0.055 s at L = 61 and 6.42 s -> 0.23 s at L = 150, i.e. 27x in
    both cases -- the quadrature is the whole cost and it now runs once per q
    instead of 40 times.  The asserted bound is 5x, well clear of timing noise,
    and the output must be BIT-IDENTICAL either way: a speed-up that changed an
    answer would be a bug, not an optimisation.
    """
    import time

    def run(collapse: bool):
        system = solver.RibbonSystem(
            [solver.Ribbon(W_nm=L_W_NM, z0_nm=L_Z_NM, L=61)],
            layers=[substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)],
        )
        system.vbar_NQ = 4000
        if not collapse:
            # force the per-(q, E) key without changing any physics
            system._stack_is_dispersive = lambda: True
        phi = -system.x_nodes[0].astype(np.complex128)
        t0 = time.perf_counter()
        out = np.array(
            [
                system.polarizability(L_Q, float(E), phi)
                for E in np.linspace(0.3, 1.2, 40)
            ]
        )
        return time.perf_counter() - t0, out

    t_slow, a_slow = run(False)
    t_fast, a_fast = run(True)
    np.testing.assert_array_equal(a_fast, a_slow)
    assert t_slow / t_fast >= 5.0, (
        f"per-q Vbar cache gave only {t_slow / t_fast:.1f}x "
        f"({t_slow:.2f} s -> {t_fast:.2f} s)"
    )


# ==========================================================================
# 9. the nonlinear_V_block contract, as nonlinear.py codes against it
# ==========================================================================
def test_nonlinear_V_block_signature_is_the_one_nonlinear_py_probes_for() -> None:
    """``(j, jp, q, E_ev=None)``, and ``nonlinear`` must detect the 4th argument.

    ``nonlinear.py`` probes the BOUND hook with ``_hook_takes_energy`` and, if
    the probe says no, calls it as ``hook(j, jp, q)`` and warns -- which with a
    stack present means the harmonics are screened at an unspecified frequency.
    So this signature is a contract between the two modules and is pinned here
    rather than left to the two docstrings to agree by hand.
    """
    params = list(
        inspect.signature(solver.RibbonSystem.nonlinear_V_block).parameters.values()
    )
    assert [p.name for p in params] == ["self", "j", "jp", "q", "E_ev"]
    assert params[-1].default is None
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params
    ), "nonlinear.py calls the hook positionally"
    assert nonlinear._hook_takes_energy(_system().nonlinear_V_block) is True

    # the probe is what decides between the screened and the warning path, so
    # a 3-argument hook must still be detected as the old signature
    assert nonlinear._hook_takes_energy(lambda j, jp, q: None) is False


@pytest.mark.parametrize("use_toeplitz", [False, True])
@pytest.mark.parametrize("transparent_stack", [False, True])
def test_nonlinear_V_block_is_a_bit_exact_no_op_without_a_substrate(
    use_toeplitz: bool, transparent_stack: bool
) -> None:
    """The hook must not perturb a free-standing system AT ALL.

    ``build_Vbar`` short-circuits ``r == 0`` to exact zeros, so both
    ``layers=None`` and a vacuum stack must return the system's own direct block
    bit for bit -- on either Coulomb path, and at any energy, since nothing
    frequency dependent is present to enter.  A tolerance here would hide the
    hook adding a small spurious screening to every free-standing nonlinear
    solve.
    """
    layers = _vac_layers() if transparent_stack else None
    system = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=1.0, EF_ev=0.5)], layers=layers, dx_nm=0.25
    )
    system.use_toeplitz = use_toeplitz
    q = 2.0 * Q_INV_NM

    direct = system._V_block(0, 0, q)
    for E in (E_EV, 2.0 * E_EV, 3.0 * E_EV):
        np.testing.assert_array_equal(system.nonlinear_V_block(0, 0, q, E), direct)

    # and on the dense path the direct block IS operators.build_V
    if not use_toeplitz:
        np.testing.assert_array_equal(
            direct,
            operators.build_V(
                abs(q), 6.0, 6.0, 0.0, 0.0, 1.0, 1.0,
                system.theta[0], system.theta[0], system.h[0],
            ),
        )

    # layers=None additionally tolerates E_ev=None; a stack never does
    if layers is None:
        np.testing.assert_array_equal(system.nonlinear_V_block(0, 0, q, None), direct)
    else:
        with pytest.raises(ValueError, match="needs E_ev"):
            system.nonlinear_V_block(0, 0, q, None)


def test_nonlinear_V_block_screens_each_harmonic_at_its_own_frequency() -> None:
    """The energy is genuinely plumbed through, and the cache does not eat it.

    For a DISPERSIVE stack the 1-, 2- and 3-omega blocks must all differ: that
    is the whole reason the hook takes an energy, and it is also the outside
    view of the cache-scope test above.  For a purely dielectric stack they must
    be bit-identical instead, because r is exactly frequency independent there
    -- so "the blocks differ" is the wrong universal assertion, and the two
    cases are pinned separately.
    """
    q = 2.0 * Q_INV_NM

    disp = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=2.0)],
        layers=[
            substrate.Layer(1.0, d_nm=0.0),
            substrate.Layer(eps_par=4.0, sigma_sheet=0.5 + 0.2j),
        ],
        dx_nm=0.5,
    )
    disp.vbar_Qmax, disp.vbar_NQ = 30.0, 6000
    blocks = [disp.nonlinear_V_block(0, 0, s * q, s * E_EV) for s in (1, 2, 3)]
    scale = float(np.max(np.abs(blocks[0])))
    for s, blk in zip((2, 3), blocks[1:]):
        rel = float(np.max(np.abs(blk - blocks[0]))) / scale
        assert rel > 1e-3, (
            f"harmonic s = {s} was screened like the fundamental (rel {rel:.2e}): "
            "either E_ev is ignored or the Vbar cache collapsed (q, E)"
        )

    # the energy alone must matter, at fixed q -- otherwise only q is plumbed
    same_q = [disp.nonlinear_V_block(0, 0, q, s * E_EV) for s in (1, 2)]
    assert float(np.max(np.abs(same_q[1] - same_q[0]))) / scale > 1e-3

    diel = solver.RibbonSystem(
        [solver.Ribbon(W_nm=6.0, z0_nm=2.0)],
        layers=[substrate.Layer(1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)],
        dx_nm=0.5,
    )
    diel.vbar_Qmax, diel.vbar_NQ = 30.0, 6000
    np.testing.assert_array_equal(
        diel.nonlinear_V_block(0, 0, q, 2.0 * E_EV),
        diel.nonlinear_V_block(0, 0, q, E_EV),
    )
