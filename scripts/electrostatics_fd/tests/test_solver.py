"""Tests for ``solver.py`` and ``observables.py``.

Reference values
----------------
* Bright eigenvalues ``eta = -0.0687, -0.0184, -0.0107`` and the interleaved
  dark mode at ``-0.0289``: NPLab ``nl-graphene.ts`` RIBBON_ETA_J, from Cox, Yu
  and Garcia de Abajo, Phys. Rev. B 96, 045442 (2017), Table S2 (which quotes
  -0.0689, -0.0289, -0.0184 for the first three eigenvalues *including* the
  dark one).
* Dipole overlaps ``zeta = 0.943, 0.239, 0.137`` from the same table; the
  ratios 0.2535 and 0.1453 are normalisation independent and are what is
  asserted.  The absolute scale of this implementation's normalisation is
  reported, never tuned.
* Dissymmetry definition: paper 456, Eq. (2).
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import k0, k1

# The package has no guaranteed import route (it is used both as
# ``electrostatics_fd.solver`` with scripts/ on the path and as a flat module
# directory), so make both work.
_PKG = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_PKG), str(_PKG.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conductivity  # noqa: E402
import observables  # noqa: E402
import operators  # noqa: E402
import solver  # noqa: E402
import units  # noqa: E402
from solver import Ribbon, RibbonSystem, bright_mode_mask  # noqa: E402

# Reference values -----------------------------------------------------------
ETA_REF = (-0.0687, -0.0184, -0.0107)
ETA_DARK_REF = -0.0289
ZETA_REF = (0.943, 0.239, 0.137)
ZETA_RATIO_REF = (0.239 / 0.943, 0.137 / 0.943)  # 0.25345, 0.14528

QW_SMALL = 1e-4  # q*W used for the q -> 0 limit


def _single(W=10.0, L=600, **kw) -> RibbonSystem:
    return RibbonSystem([Ribbon(W_nm=W, L=L, **kw)])


def _bright(sys_: RibbonSystem, qw: float = QW_SMALL):
    """(eta, zeta) of the dipole-active modes, and the full spectrum."""
    q = qw / sys_.ribbons[0].W_nm
    eta, _modes, zeta = sys_.eigenmodes(q)
    mask = bright_mode_mask(zeta)
    return np.real(eta[mask]), np.real(zeta[mask]), np.real(eta)


# ===========================================================================
# 1. ACCEPTANCE TARGET: bright eigenvalues, dark mode, convergence, zeta
# ===========================================================================
def test_eigenmodes_bright_eta_acceptance():
    """eta_1,2,3 of the bright (dipole-active) modes at q -> 0, L = 600."""
    eta_br, zeta_br, _full = _bright(_single(L=600))
    got = eta_br[:3]
    print(f"\nbright eta (L=600, qW={QW_SMALL:g}): {got}")
    print(f"bright zeta: {zeta_br[:3]}")
    for k, (g, ref) in enumerate(zip(got, ETA_REF), start=1):
        assert abs(g - ref) < 5e-4, (
            f"eta_{k} = {g:.6f}, expected {ref} +- 5e-4 (|diff| = {abs(g-ref):.2e}). "
            "If eta_1 is near -0.0222 then operators.build_D_geom is using a "
            "Dirichlet tridiagonal instead of the zero-flux (Neumann) boundary "
            "rows [-1, 1]/h**2 that the no-current edge condition requires."
        )


def test_eigenmodes_dark_mode_present():
    """The unfiltered spectrum contains the dark mode at -0.0289 (paper 406)."""
    eta_br, zeta_br, eta_full = _bright(_single(L=600))
    near = eta_full[np.abs(eta_full - ETA_DARK_REF) < 5e-4]
    print(f"\nspectrum entries near {ETA_DARK_REF}: {near}")
    assert near.size >= 1, (
        f"no eigenvalue within 5e-4 of the dark mode {ETA_DARK_REF}; "
        f"closest in the unfiltered spectrum: "
        f"{eta_full[np.argmin(np.abs(eta_full - ETA_DARK_REF))]:.6f}"
    )
    # and it must be classified dark, i.e. absent from the bright list
    assert np.min(np.abs(eta_br - ETA_DARK_REF)) > 5e-3, (
        "the -0.0289 mode was classified bright; the bright/dark filter is wrong"
    )
    # the bright list must also be interleaved with dark modes: between bright 1
    # and bright 2 there is at least one (dark) eigenvalue
    between = eta_full[(eta_full > ETA_REF[0]) & (eta_full < ETA_REF[1])]
    assert between.size >= 1
    assert np.all(np.abs(zeta_br[:3]) > 1e-2)  # the compared modes really are bright


def test_eigenmodes_convergence_in_L():
    """|eta_1(L=600) - eta_1(L=300)| < 1e-4."""
    e600 = _bright(_single(L=600))[0][0]
    e300 = _bright(_single(L=300))[0][0]
    diff = abs(e600 - e300)
    print(f"\neta_1: L=300 {e300:.8f}  L=600 {e600:.8f}  |diff| = {diff:.3e}")
    assert diff < 1e-4, f"eta_1 not converged: |diff| = {diff:.3e} >= 1e-4"


def test_zeta_ratios_and_absolute_scale(record_property):
    """Normalisation-independent zeta ratios, plus a report of the absolute scale.

    The ratios zeta_2/zeta_1 and zeta_3/zeta_1 are asserted against the
    reference 0.2535 and 0.1453.  The absolute scale factor relative to 0.943 is
    only REPORTED (print + recorded property): no constant is tuned to make it
    match.
    """
    _eta, zeta_br, _full = _bright(_single(L=600))
    z = np.abs(zeta_br[:3])
    r2, r3 = z[1] / z[0], z[2] / z[0]
    scale = z[0] / ZETA_REF[0]
    print(
        f"\nzeta (normalisation: {solver.ZETA_NORMALISATION})"
        f"\n  computed  : {z}"
        f"\n  reference : {ZETA_REF}"
        f"\n  ratios    : {r2:.5f}, {r3:.5f}  (reference "
        f"{ZETA_RATIO_REF[0]:.5f}, {ZETA_RATIO_REF[1]:.5f})"
        f"\n  ABSOLUTE SCALE FACTOR vs 0.943: {scale:.6f}"
    )
    record_property("zeta_absolute_scale_factor_vs_0p943", float(scale))
    record_property("zeta_1_computed", float(z[0]))
    record_property("zeta_normalisation", solver.ZETA_NORMALISATION)
    assert r2 == pytest.approx(ZETA_RATIO_REF[0], rel=5e-2)
    assert r3 == pytest.approx(ZETA_RATIO_REF[1], rel=5e-2)


def test_zeta_sign_and_normalisation_identity():
    """The returned modes really satisfy the documented normalisation.

    CONDITIONING SPLIT (2026-11-03), and it is a TIGHTENING, not a relaxation.
    This used to assert ``atol=1e-8`` on every mode at once.  Exactly one mode
    cannot meet that, and not for any reason to do with the solver: at small
    qW the zero-flux ``D_geom`` has a near-null uniform-potential eigenvector
    (eta = -4.65e6, |zeta| = 8.3e-11, i.e. dark and spurious -- see the
    ``eigenmodes`` docstring).  Its reduced potential is essentially constant at
    u ~ 9.98e3 with std 1.1e-6, and ``D_geom u`` is then a cancellation of
    O(1/h^2) terms down to O((qW)^2), so the quadratic form is amplified by
    1/(qW)^2 = 1e8.  MEASURED at L = 200: a ONE-ULP relative perturbation of
    that eigenvector moves ``nrm2`` by up to 9.3e-05 (median 1.5e-05), and
    evaluating the quadratic form of the SAME float64 vector in 40-digit
    arithmetic still gives nrm2 - 1 = +7.4e-06.  The quantity simply is not
    computable to 1e-8 in float64; the old blanket assertion was passing on
    luck and went red when ``operators``' branch-3 rearrangement moved V at the
    1e-15 level.

    So the normalisation is asserted at atol=1e-11 -- a THOUSAND times tighter
    than before -- on every mode whose quadratic form is actually conditioned
    to deliver it, and the spurious mode is held to the roundoff bound its own
    condition number implies.  The condition number separates the two cases by
    nine orders of magnitude with nothing in between (measured at L = 100/200/600:
    the spurious mode is 3.9e12/1.6e13/1.4e14, every other mode is at most
    3.9e3/1.6e4/1.4e5), so the 1e8 threshold is not a tuned constant.
    """
    sys_ = _single(L=200)
    q = QW_SMALL / sys_.ribbons[0].W_nm
    eta, modes, zeta = sys_.eigenmodes(q)
    Dg = np.asarray(operators.build_D_geom(q, 10.0, sys_.h[0], sys_.L[0]))
    u = modes / 10.0  # reduced potential phi/W
    nrm2 = -sys_.h[0] * np.einsum("im,ij,jm->m", u, Dg, u)

    # Condition number of the quadratic form, mode by mode:
    # |u|^T |D_geom| |u| / |u^T D_geom u|.  This is the standard bound on the
    # relative roundoff of the form, and it is computed here, not assumed.
    cond = (sys_.h[0] * np.einsum("im,ij,jm->m", np.abs(u), np.abs(Dg), np.abs(u))
            / np.abs(nrm2))
    well = cond < 1e8
    assert np.count_nonzero(~well) == 1, (
        f"expected exactly one ill-conditioned mode, got {np.count_nonzero(~well)}"
    )
    # the nine-decade gap the threshold sits in, asserted rather than trusted
    assert np.max(cond[well]) < 1e7 < 1e9 < np.min(cond[~well])

    assert np.allclose(np.real(nrm2[well]), 1.0, atol=1e-11), (
        f"normalisation violated: min {np.min(np.real(nrm2[well]))}, "
        f"max {np.max(np.real(nrm2[well]))}"
    )

    # The excluded mode is the documented spurious uniform-potential one: dark,
    # |eta| ~ 1/(qW)^2, and within the roundoff its conditioning allows.
    spur = np.flatnonzero(~well)[0]
    print(f"\nill-conditioned mode {spur}: cond = {cond[spur]:.3e}, "
          f"eta = {np.real(eta[spur]):.4e}, |zeta| = {abs(np.real(zeta[spur])):.3e}, "
          f"nrm2 - 1 = {np.real(nrm2[spur]) - 1.0:+.3e}, "
          f"eps*cond bound = {np.finfo(float).eps * cond[spur]:.3e}")
    assert abs(np.real(eta[spur])) > 1e5          # the 1/(qW)^2 mode, not a physical one
    assert abs(np.real(zeta[spur])) < 1e-9        # dark: filtered by bright_mode_mask
    assert not bright_mode_mask(zeta)[spur]
    assert abs(np.real(nrm2[spur]) - 1.0) <= np.finfo(float).eps * cond[spur]
    # zeta must be the theta-moment of rho = D_geom phi / W, same Riemann rule
    rho = (Dg @ modes) / 10.0
    zeta_direct = sys_.h[0] * (sys_.theta[0] @ rho)
    assert np.allclose(zeta, zeta_direct, atol=1e-12)
    # sign convention: bright modes come out with zeta >= 0
    assert np.all(np.real(zeta[bright_mode_mask(zeta)]) > 0.0)


# ===========================================================================
# 2. eta pair
# ===========================================================================
def test_eta_pair_drude_and_drift_agree_at_beta_zero():
    """The beta = 0 Drude shortcut and graphene_drift_sigma agree to rtol 1e-10.

    CONTRACT CORRECTION.  The contract asks for this agreement at beta = 0, full
    stop.  It holds only at q = 0: graphene_drift_sigma also carries the
    nonlocal factor alpha = v_F q hbar / (E + i gamma), and at beta = 0 it
    reduces to sigma_perp = 2 sigma_w/(1 + sqrt(1-alpha^2)), which equals the
    Drude sigma only when alpha = 0.  The finite-q difference is a physical
    O(alpha^2) correction, pinned quantitatively in the next test, not an
    inconsistency between the two routes.
    """
    sys_ = _single(L=31)
    for E in (0.3, 0.6, 1.2):
        ep, ea = sys_.eta_pair(0, 0.0, E)
        sp, sl = conductivity.graphene_drift_sigma(E, 0.5, 0.02, 0.0, 0.0)
        pref = 1j / (units.omega_of(E) * 10.0 * sys_.eps_eff)
        assert ep == pytest.approx(pref * sp, rel=1e-10)
        assert ea == pytest.approx(pref * sl, rel=1e-10)
        assert ep == ea  # isotropic at beta = 0


def test_drift_route_nonlocal_correction_at_beta_zero():
    """At beta = 0, q != 0 the drift route exceeds Drude by exactly alpha^2/4.

        sigma_perp/sigma_Drude = 2/(1 + sqrt(1-alpha^2)) = 1 + alpha^2/4 + O(alpha^4)
        sigma_par /sigma_Drude = 2/(sqrt(1-alpha^2)(1+sqrt(1-alpha^2)))
                               = 1 + 3 alpha^2/4 + O(alpha^4)
    """
    E, gamma = 0.6, 0.02
    for q in (1e-4, 0.05, 0.3):
        alpha = units.VF_NM_FS * q * units.HBAR_EV_FS / (E + 1j * gamma)
        drude = conductivity.graphene_drude(E, 0.5, gamma)
        sp, sl = conductivity.graphene_drift_sigma(E, 0.5, gamma, q, 0.0)
        got_perp = sp / drude - 1.0
        got_par = sl / drude - 1.0
        sa = np.sqrt(1.0 - alpha**2)
        exact_perp = 2.0 / (1.0 + sa) - 1.0
        exact_par = 2.0 / (sa * (1.0 + sa)) - 1.0
        print(f"\nq={q:g}: alpha={alpha:.4e}  perp excess {got_perp:.4e} "
              f"(exact {exact_perp:.4e}, alpha^2/4 = {alpha**2/4:.4e})  "
              f"par excess {got_par:.4e} (exact {exact_par:.4e})")
        # the exact closed form, to machine precision
        assert got_perp == pytest.approx(exact_perp, rel=1e-12)
        assert got_par == pytest.approx(exact_par, rel=1e-12)
        # and the leading order where alpha is small enough for it to hold
        if abs(alpha) < 0.1:
            assert got_perp == pytest.approx(alpha**2 / 4.0, rel=2e-2)
            assert got_par == pytest.approx(3.0 * alpha**2 / 4.0, rel=2e-2)


def test_eta_pair_matches_closed_form_and_is_dimensionless():
    """eta = i sigma / (omega W eps_eff) against an independent evaluation."""
    sys_ = RibbonSystem([Ribbon(W_nm=20.0, L=31, EF_ev=0.4, gamma_ev=0.01)],
                        eps_a=1.0, eps_b=3.9)
    E = 0.45
    ep, ea = sys_.eta_pair(0, 0.02, E)
    sigma = conductivity.graphene_drude(E, 0.4, 0.01)
    expect = 1j * sigma / (units.omega_of(E) * 20.0 * (1.0 + 3.9) / 2.0)
    assert ep == pytest.approx(expect, rel=1e-13)
    assert ea == pytest.approx(expect, rel=1e-13)
    # collisionless Drude gives a negative real eta: same sign as eta_m
    assert ep.real < 0.0


def test_eta_pair_drift_is_odd_in_q():
    """With beta != 0 the response must break q -> -q parity (signed q)."""
    sys_ = _single(L=31, beta=0.3)
    ep_p, ea_p = sys_.eta_pair(0, 0.2, 0.6)
    ep_m, ea_m = sys_.eta_pair(0, -0.2, 0.6)
    rel = abs(ep_p - ep_m) / abs(ep_p)
    print(f"\n|eta_perp(+q) - eta_perp(-q)|/|eta_perp| = {rel:.4e}")
    assert rel > 1e-3, (
        "eta is even in q at beta = 0.3; the drift asymmetry was lost "
        "(solver.SIGNED_Q_IN_DRIFT_SIGMA, or abs() inside graphene_drift_sigma)"
    )
    assert abs(ep_p - ea_p) / abs(ep_p) > 1e-6, "drift did not make eta anisotropic"


# ===========================================================================
# 3. assembly and linear solve
# ===========================================================================
def test_solve_linear_residual_and_rho_definition():
    """(1 - M) phi = phi_ext to machine precision, and rho = D phi / W."""
    sys_ = _single(W=10.0, L=120)
    q, E = 0.05, 0.6
    phi_ext = -sys_.x_nodes[0].astype(np.complex128)
    sol = sys_.solve_linear(q, E, phi_ext)
    M = sys_.build_M(q, E)
    res = (np.eye(sys_.N) - M) @ sol["phi_flat"] - phi_ext
    rel = np.linalg.norm(res) / np.linalg.norm(phi_ext)
    print(f"\nrelative residual of (1-M)phi = phi_ext: {rel:.3e}")
    assert rel < 1e-10
    ep, ea = sys_.eta_pair(0, q, E)
    D = np.asarray(operators.build_D(q, 10.0, sys_.h[0], sys_.L[0], ep, ea))
    assert np.allclose(sol["rho"][0], D @ sol["phi"][0] / 10.0, rtol=1e-12, atol=0)


def test_build_M_block_structure_two_ribbons():
    """M[j, jp] = V(j,jp) @ D(jp): the eta inside D belongs to the SOURCE ribbon.

    The structural identity is checked against an INDEPENDENT ``build_V``
    reference at rtol 1e-13, with ``use_toeplitz`` off so that both sides reach
    the separation through identical arithmetic.  ``use_toeplitz`` is on by
    default and takes the ``build_V_toeplitz`` path, which is the same matrix
    but, for a non-zero ``x_off``, only to the ~1e-15 ABSOLUTE cancellation
    floor of the branch-3 Struve difference documented in ``operators.py``; the
    off-diagonal blocks here have ``x_off = 16 nm``, and ``V @ D`` amplifies
    that floor by ``|D| ~ 1.4e2``.  The second half of the test pins the
    cross-path agreement at that floor instead of pretending it is 1e-13.
    """
    r0 = Ribbon(W_nm=10.0, x0_nm=-8.0, L=40, EF_ev=0.5)
    r1 = Ribbon(W_nm=10.0, x0_nm=+8.0, L=40, EF_ev=0.8)  # different eta
    sys_ = RibbonSystem([r0, r1])
    sys_.use_toeplitz = False
    q, E = 0.03, 0.5
    M = sys_.build_M(q, E)
    assert M.shape == (80, 80)
    for jp, rib in enumerate((r0, r1)):
        ep, ea = sys_.eta_pair(jp, q, E)
        D = np.asarray(operators.build_D(q, rib.W_nm, sys_.h[jp], sys_.L[jp], ep, ea))
        for j, rj in enumerate((r0, r1)):
            V = np.asarray(operators.build_V(
                abs(q), rj.W_nm, rib.W_nm, rj.x0_nm, rib.x0_nm, rj.z0_nm, rib.z0_nm,
                sys_.theta[j], sys_.theta[jp], sys_.h[jp]))
            assert np.allclose(M[sys_.slices[j], sys_.slices[jp]], V @ D,
                               rtol=1e-13, atol=0)
    # the two blocks really do carry different eta
    assert sys_.eta_pair(0, q, E)[0] != sys_.eta_pair(1, q, E)[0]

    # Same M from the default Toeplitz path, to the documented absolute floor.
    fast = RibbonSystem([r0, r1])
    assert fast.use_toeplitz is True, "the Toeplitz fast path is the default"
    M_fast = fast.build_M(q, E)
    gap = float(np.max(np.abs(M_fast - M)))
    scale = float(np.max(np.abs(M)))
    assert gap < 1e-11 * scale, (
        f"Toeplitz and dense Coulomb paths disagree by {gap:.3e} "
        f"(peak |M| = {scale:.3e}); the documented floor is ~1e-15 absolute in V"
    )


def test_stacked_far_ribbons_decouple():
    """Two identical ribbons 400 nm apart in z give exactly twice the single response.

    A vertical offset puts the cross blocks on Coulomb branch 1 (plain
    ``2 h K0``), which is accurate at any separation, so this is the clean test
    of the block assembly.  q*dz = 20 -> K0(20) ~ 5.7e-10.
    """
    one = RibbonSystem([Ribbon(W_nm=10.0, L=60)])
    two = RibbonSystem([Ribbon(W_nm=10.0, z0_nm=0.0, L=60),
                        Ribbon(W_nm=10.0, z0_nm=400.0, L=60)])
    q, E = 0.05, 0.6
    a1 = one.polarizability(q, E, -one.x_nodes[0])
    a2 = two.polarizability(q, E, [-two.x_nodes[0], -two.x_nodes[1]])
    print(f"\nalpha(1 ribbon)  = {a1:.10e}\nalpha(2 stacked) = {a2:.10e}"
          f"\nrelative deviation from 2x: {abs(a2 - 2*a1)/abs(2*a1):.3e}")
    assert a2 == pytest.approx(2.0 * a1, rel=1e-8)


def test_coplanar_far_ribbons_decouple():
    """Coplanar ribbons at large q*d must decouple: alpha(2) == 2 alpha(1).

    PROMOTED FROM A STRICT XFAIL (2026-11-03).  This was xfailed on the grounds
    that the coplanar Coulomb cross block takes branch 3 of ``operators``, whose
    bare 30-term modified-Struve series was accurate only to Q*|theta| ~ 4 and
    useless past ~15.  At q*d = 50 the cross block hit an ABSOLUTE cancellation
    floor of 9.068e-15 -- 4.7e9 times the true coupling K_0(50) ~ 1.9e-24 --
    and the pair did not decouple (~4e-4 relative in this observable).

    ``operators`` now evaluates the branch-3 primitive through the complementary
    function Fc = 1 - F in four regions (series, two Chebyshev panels, an
    asymptotic series), so the cross block is formed without the cancellation
    and tracks 2 h K_0(q d) over 24 decades.  The deviation from 2x measured
    here is 3.2e-13 against the rel=1e-9 this test always asked for, i.e. a
    ~3000x margin.  The assertion below is unchanged from the xfailed body;
    only the marker is gone.

    The residual floor that remains is confined to Q*|theta| < 4 (worst 3.3e-11
    relative), where V is not exponentially small and so cannot cause a wrong
    decoupling; it is pinned by
    ``tests/test_operators.py::test_remaining_small_argument_floor_is_bounded_and_documented``.
    """
    one = RibbonSystem([Ribbon(W_nm=10.0, L=60)])
    two = RibbonSystem([Ribbon(W_nm=10.0, x0_nm=0.0, L=60),
                        Ribbon(W_nm=10.0, x0_nm=1000.0, L=60)])
    q, E = 0.05, 0.6  # q*d = 50
    a1 = one.polarizability(q, E, -one.x_nodes[0])
    phi = [-(two.x_nodes[0] - 0.0), -(two.x_nodes[1] - 1000.0)]
    a2 = two.polarizability(q, E, phi)
    print(f"\nalpha(1 ribbon)  = {a1:.10e}\nalpha(2 coplanar, q*d = 50) = {a2:.10e}"
          f"\nrelative deviation from 2x: {abs(a2 - 2*a1)/abs(2*a1):.3e} (was ~4e-4)")
    assert a2 == pytest.approx(2.0 * a1, rel=1e-9)


def test_transparent_substrate_is_a_no_op():
    """layers = [vacuum, vacuum] has r = 0, so Vbar must vanish identically."""
    substrate = pytest.importorskip("substrate")
    q, E = 0.04, 0.55
    plain = _single(W=10.0, L=60)
    withsub = RibbonSystem([Ribbon(W_nm=10.0, L=60)],
                           layers=[substrate.Layer(1.0), substrate.Layer(1.0)])
    r = substrate.reflection_coefficient(abs(q), E, withsub.layers, eps1=1.0)
    assert abs(complex(r)) < 1e-14, f"vacuum/vacuum reflection is {r}, expected 0"
    phi_ext = -plain.x_nodes[0]
    a_plain = plain.polarizability(q, E, phi_ext)
    a_sub = withsub.polarizability(q, E, phi_ext)
    print(f"\nalpha without substrate {a_plain:.10e}\nalpha with r=0 stack   {a_sub:.10e}")
    assert a_sub == pytest.approx(a_plain, rel=1e-10)


def test_toeplitz_fast_path_matches_dense_V():
    """solver.use_toeplitz must not change the answer (operators contract)."""
    q, E = 0.05, 0.6
    base = RibbonSystem([Ribbon(W_nm=10.0, x0_nm=-7.0, L=50),
                         Ribbon(W_nm=10.0, x0_nm=+7.0, L=50)])
    fast = RibbonSystem([Ribbon(W_nm=10.0, x0_nm=-7.0, L=50),
                         Ribbon(W_nm=10.0, x0_nm=+7.0, L=50)])
    fast.use_toeplitz = True
    m0 = base.build_M(q, E)
    m1 = fast.build_M(q, E)
    err = np.max(np.abs(m0 - m1)) / np.max(np.abs(m0))
    print(f"\nmax relative |M_dense - M_toeplitz| = {err:.3e}")
    assert err < 1e-10


# ===========================================================================
# 4. resonance cross-check: eigenmode eta_1 predicts the polarizability peak
# ===========================================================================
def test_polarizability_peak_sits_at_eta_1():
    """The |alpha| peak must occur where Re eta(E) = eta_1 of the eigenproblem.

    This ties the eigenvalue machinery to the full self-consistent solve: the
    resonance of (1 - eta(E) lambda_1) is at eta(E) = 1/lambda_1 = eta_1.
    """
    sys_ = _single(W=10.0, L=200)
    eta_1 = _bright(sys_)[0][0]
    q = QW_SMALL / 10.0

    def f(E):
        return sys_.eta_pair(0, q, E)[0].real - eta_1

    E_res = brentq(f, 0.2, 1.5, xtol=1e-10)
    Es = np.linspace(E_res - 0.08, E_res + 0.08, 161)
    alpha = sys_.sweep([q], Es, lambda qq, EE: -sys_.x_nodes[0], fast_path=True)[0]
    E_peak = Es[np.argmax(np.abs(alpha))]
    print(f"\neta_1 = {eta_1:.6f} -> predicted resonance {E_res:.6f} eV, "
          f"sweep peak {E_peak:.6f} eV (grid step {Es[1]-Es[0]:.2e})")
    assert abs(E_peak - E_res) <= 2.0 * (Es[1] - Es[0])
    # and the peak must be a real resonance, not a monotone edge
    assert abs(alpha[np.argmax(np.abs(alpha))]) > 5.0 * abs(alpha[0])


# ===========================================================================
# 5. frequency-sweep fast path: correctness, speed, refusal
# ===========================================================================
def test_sweep_fast_and_direct_agree():
    """40-frequency sweep, W = 10 nm, EF = 0.5, gamma = 0.02, L = 200, q*W = 0.5."""
    sys_ = _single(W=10.0, L=200)
    q = 0.5 / 10.0
    Es = np.linspace(0.3, 1.1, 40)
    import sources
    phi0 = sources.phi_ext_dipole(q, sys_.x_nodes[0], 0.0, (2.0, 0.0, 6.0), (1.0, 0.0, 0.0))

    def src(qq, EE):
        return phi0

    fast = sys_.sweep([q], Es, src, fast_path=True)
    direct = sys_.sweep([q], Es, src, fast_path=False)
    rel = np.max(np.abs(fast - direct) / np.abs(direct))
    print(f"\nmax relative fast-vs-direct deviation over 40 frequencies: {rel:.3e}")
    assert rel < 1e-8, f"fast path deviates by {rel:.3e} (> 1e-8)"
    assert np.max(np.abs(direct)) > 0.0


def test_sweep_fast_path_speedup():
    """200-frequency sweep at L = 300: the fast path must be >= 8x faster."""
    sys_fast = _single(W=10.0, L=300)
    sys_slow = _single(W=10.0, L=300)
    q = 0.5 / 10.0
    Es = np.linspace(0.3, 1.1, 200)
    phi0 = -sys_fast.x_nodes[0].astype(np.complex128)

    def src(qq, EE):
        return phi0

    # warm up both (V cache, BLAS threads) so the comparison is of the solve
    sys_fast.sweep([q], Es[:2], src, fast_path=True)
    sys_slow.sweep([q], Es[:2], src, fast_path=False)

    t0 = time.perf_counter()
    a_fast = sys_fast.sweep([q], Es, src, fast_path=True)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    a_slow = sys_slow.sweep([q], Es, src, fast_path=False)
    t_direct = time.perf_counter() - t0

    speedup = t_direct / t_fast
    msg = (f"fast path {t_fast*1e3:.1f} ms, direct path {t_direct*1e3:.1f} ms, "
           f"speedup {speedup:.2f}x over {Es.size} frequencies at L=300")
    print("\n" + msg)
    assert np.max(np.abs(a_fast - a_slow) / np.abs(a_slow)) < 1e-8, (
        "fast path is fast but wrong: " + msg
    )
    assert speedup >= 8.0, msg


def test_sweep_fast_path_refuses_drift():
    """Documented behaviour: fast_path=True RAISES when it is not applicable."""
    drifted = _single(W=10.0, L=40, beta=0.3)
    with pytest.raises(ValueError, match="tensor"):
        drifted.sweep([0.05], [0.6], lambda q, E: -drifted.x_nodes[0], fast_path=True)
    # ... and the direct path still works for the same system
    out = drifted.sweep([0.05], [0.6], lambda q, E: -drifted.x_nodes[0], fast_path=False)
    assert np.isfinite(out).all() and out.shape == (1, 1)
    assert abs(out[0, 0]) > 0.0

    # mismatched ribbons and a substrate are refused for the same reason
    mixed = RibbonSystem([Ribbon(W_nm=10.0, L=40), Ribbon(W_nm=12.0, L=40)])
    with pytest.raises(ValueError, match="common"):
        mixed.sweep([0.05], [0.6], lambda q, E: np.zeros(mixed.N), fast_path=True)
    assert mixed.fast_path_obstruction() is not None
    assert _single(L=40).fast_path_obstruction() is None


def test_sweep_fast_path_two_identical_ribbons():
    """The fast path must also be exact for a multi-block (coupled) system."""
    def make():
        return RibbonSystem([Ribbon(W_nm=10.0, x0_nm=-7.0, L=90),
                             Ribbon(W_nm=10.0, x0_nm=+7.0, L=90)])
    a, b = make(), make()
    q = 0.3 / 10.0           # q*d = 0.42, well inside the Struve-accurate range
    Es = np.linspace(0.35, 1.0, 25)
    src = [-(a.x_nodes[0] + 7.0), -(a.x_nodes[1] - 7.0)]
    fast = a.sweep([q], Es, lambda qq, EE: src, fast_path=True)
    direct = b.sweep([q], Es, lambda qq, EE: src, fast_path=False)
    rel = np.max(np.abs(fast - direct) / np.abs(direct))
    print(f"\ntwo-ribbon fast-vs-direct max relative deviation: {rel:.3e}")
    assert rel < 1e-8
    # the two ribbons are genuinely coupled at this separation
    single = RibbonSystem([Ribbon(W_nm=10.0, L=90)])
    lone = single.sweep([q], Es, lambda qq, EE: -single.x_nodes[0], fast_path=True)
    assert np.max(np.abs(direct / 2.0 - lone) / np.abs(lone)) > 1e-3


def test_polarizability_matches_plane_wave_source_convention():
    """polarizability() is alpha/unit length for sources.phi_ext_plane_wave(x, E0=1).

    ``phi_ext_plane_wave`` returns ``-E0*x``, so the in-plane driving field is
    ``+E0``; with E0 = 1 the returned dipole moment per unit length IS the
    polarizability, and it must scale linearly with E0.
    """
    import sources
    sys_ = _single(W=10.0, L=60)
    q, E = 0.04, 0.58
    pw = sources.phi_ext_plane_wave(sys_.x_nodes[0], 1.0)
    assert np.allclose(pw, -sys_.x_nodes[0], rtol=0, atol=0)
    a1 = sys_.polarizability(q, E, pw)
    a3 = sys_.polarizability(q, E, sources.phi_ext_plane_wave(sys_.x_nodes[0], 3.0))
    print(f"\nalpha(E0=1) = {a1:.8e}, alpha(E0=3)/3 = {a3/3:.8e}")
    assert a3 == pytest.approx(3.0 * a1, rel=1e-12)
    assert abs(a1) > 0.0


def test_sweep_shape_and_single_arg_source():
    sys_ = _single(W=10.0, L=40)
    qs = [0.02, 0.05]
    Es = [0.5, 0.6, 0.7]
    out = sys_.sweep(qs, Es, lambda q: -sys_.x_nodes[0], fast_path=True)
    assert out.shape == (2, 3)
    ref = sys_.polarizability(qs[1], Es[2], -sys_.x_nodes[0])
    assert out[1, 2] == pytest.approx(ref, rel=1e-8)


# ===========================================================================
# 6. observables
# ===========================================================================
def test_induced_field_against_analytic_line_charge():
    """A narrow charge spike reproduces the analytic single-line-charge field.

    For rho concentrated at x' = xc with weight A = \\int rho dx',
        Ex = 2|q| A (x0-xc)/s K1(|q| s),  Ey = -2i q A K0(|q| s),  Ez = 2|q| A z0/s K1
    """
    q, z0, x0, xc = 0.07, 4.0, 1.0, 0.0
    x = np.linspace(-0.5, 0.5, 2001) + xc          # fine, narrow support
    amp = 3.0
    rho = np.exp(-(x - xc) ** 2 / (2 * 0.02**2))
    rho = amp * rho / np.trapz(rho, x)             # unit-weight -> amplitude amp
    ex, ey, ez = observables.induced_field_q(q, x, rho, (x0, z0))
    s = math.hypot(x0 - xc, z0)
    ex_ref = 2 * q * amp * (x0 - xc) / s * k1(q * s)
    ey_ref = -2j * q * amp * k0(q * s)
    ez_ref = 2 * q * amp * (z0 / s) * k1(q * s)
    print(f"\nEx {ex:.8e} vs {ex_ref:.8e}\nEy {ey:.8e} vs {ey_ref:.8e}"
          f"\nEz {ez:.8e} vs {ez_ref:.8e}")
    assert ex == pytest.approx(ex_ref, rel=2e-4)
    assert ey == pytest.approx(ey_ref, rel=2e-4)
    assert ez == pytest.approx(ez_ref, rel=2e-4)


def test_induced_field_parities_and_multiblock():
    """Ex, Ez are even in q; Ey is odd.  A list of blocks sums."""
    q, r0 = 0.09, (0.5, 3.0)
    x = np.linspace(-5, 5, 201)
    rho = (0.3 + 0.2j) * np.sin(x)
    p = observables.induced_field_q(+q, x, rho, r0)
    m = observables.induced_field_q(-q, x, rho, r0)
    assert m[0] == pytest.approx(p[0], rel=1e-15, abs=1e-18)
    assert m[2] == pytest.approx(p[2], rel=1e-15, abs=1e-18)
    assert m[1] == pytest.approx(-p[1], rel=1e-15, abs=1e-18)
    two = observables.induced_field_q(q, [x, x], [rho, rho], r0)
    for a, b in zip(two, p):
        assert a == pytest.approx(2.0 * b, rel=1e-13)


def test_induced_field_rejects_singular_point():
    x = np.linspace(-1.0, 1.0, 11)
    with pytest.raises(ValueError, match="coincides"):
        observables.induced_field_q(0.1, x, np.ones_like(x), (0.0, 0.0))


def test_ldos_q_formula():
    """ldos_q = Im(conj(p).E)/(2 pi^2 |p|^2), checked against a manual evaluation."""
    q, r0 = 0.08, (1.5, 5.0)
    x = np.linspace(-5, 5, 301)
    rho = (1.0 + 0.4j) * np.cos(0.3 * x)
    e = np.asarray(observables.induced_field_q(q, x, rho, r0))
    for p in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1j, 0.0), (0.3, -0.2j, 0.7)):
        pv = np.asarray(p, dtype=complex)
        ref = np.sum(np.conj(pv) * e).imag / (2 * np.pi**2 * np.sum(np.abs(pv) ** 2))
        assert observables.ldos_q(q, x, rho, r0, p) == pytest.approx(ref, rel=1e-13)
    with pytest.raises(ValueError):
        observables.ldos_q(q, x, rho, r0, (0.0, 0.0, 0.0))


def test_dissymmetry_analytic_cases():
    """g = 0 for an even spectrum, g = 2 for a one-sided one (paper 456 Eq. 2)."""
    qs = np.linspace(-1.0, 1.0, 2001)
    even = np.exp(-(qs**2) / 0.08)
    assert observables.dissymmetry(qs, even) == pytest.approx(0.0, abs=1e-14)
    one_sided = np.where(qs > 0, even, 0.0)
    assert observables.dissymmetry(qs, one_sided) == pytest.approx(2.0, rel=1e-12)
    # a known asymmetric case: D = exp(-q^2/w)*(1 + c q) on a symmetric grid.
    # int q exp(-q^2/w) over (0, inf) = w/2; total = sqrt(pi w); so
    # g = 2 * (2 * c * w/2) / sqrt(pi*w) = 2 c sqrt(w/pi) for a wide grid.
    w, c = 0.08, 0.5
    qs = np.linspace(-4.0, 4.0, 200001)
    d = np.exp(-(qs**2) / w) * (1.0 + c * qs)
    expect = 2.0 * c * math.sqrt(w / math.pi)
    got = observables.dissymmetry(qs, d)
    print(f"\ndissymmetry analytic: got {got:.8f}, expected {expect:.8f}")
    assert got == pytest.approx(expect, rel=1e-6)
    # grid without a node at q = 0 still splits consistently
    qs2 = np.linspace(-1.0, 1.0, 2000)
    assert 0.0 not in qs2
    assert observables.dissymmetry(qs2, np.exp(-(qs2**2) / 0.08)) == pytest.approx(
        0.0, abs=1e-12)
    with pytest.raises(ValueError, match="straddle"):
        observables.dissymmetry(np.linspace(0.1, 1.0, 11), np.ones(11))


def test_induced_field_absolute_normalisation_is_a_reference_convention():
    """The two reference forms differ by exactly -1/(2 pi) in ALL THREE components.

    REPOINTED FROM A STRICT XFAIL (2026-11-03).  This test used to assert that
    ``set_Eq_Riboons``' Ex equals the reference's inline ``set_Eind`` Ex,
    and xfailed with the reason that the two "cannot
    both be right for one rho normalisation", so absolute LDOS magnitudes were
    uncertain by -1/(2 pi).

    THAT IS RESOLVED.  ``observables.induced_field_q``'s ``set_Eq_Riboons`` form
    is correct ABSOLUTELY, not merely up to a constant: the energy-conservation
    identity ``(omega/2) Im(conj(p).E_ind) == (1/2) Re(sigma) int(|E|^2)``, with
    both sides built from ``operators.build_D``'s own quadratic forms, holds to
    ratio 1.000000000000 in six independent configurations, and
    ``observables.ldos_q``'s ``2 pi^2 |p|^2 D(q)`` reproduces the Ohmic power to
    rel 1e-10.  Neither identity would hold if the magnitude carried a spurious
    -1/(2 pi).  The factor is a CONVENTION ARTIFACT of the reference MATLAB: the
    1/(2 pi) belongs in the caller's own q quadrature (paper 456 Eq. (21)), where
    ``set_Eind``'s host script applies it, and it cancels in ``dissymmetry``.
    See the "NORMALISATION -- RESOLVED (2026-11-03)" block in observables.py.

    So the relation between the two reference routines is a known exact constant,
    and that is what is asserted here.  Both sides must use the SAME quadrature
    rule for the comparison to be meaningful: ``induced_field_q`` sums over
    uniform cells of width h (matching ``operators.build_V``, whose branches
    integrate the kernel over a cell of width h centred on each node), so the
    local reference below does too.  With matched rules the ratio is
    -1/(2 pi) to 0.0e+00 relative in every component; with ``np.trapz`` on one
    side only it is off by 5.2e-09 (Ex), 2.8e-08 (Ey), 1.6e-08 (Ez), which is
    the end-node half-weight difference and nothing physical.
    """
    q, r0 = 0.06, (1.0, 4.0)
    x = np.linspace(-5.0, 5.0, 401)
    rho = np.exp(-(x**2) / 2.0) + 0.0j
    impl = np.asarray(observables.induced_field_q(q, x, rho, r0))

    # the set_Eind form, on induced_field_q's own uniform-cell rule
    dx = r0[0] - x
    s = np.sqrt(dx**2 + r0[1] ** 2)
    cell = float(x[1] - x[0])
    set_eind = np.array([
        -q * cell * np.sum(rho * dx / s * k1(q * s)) / np.pi,
        +1j * q * cell * np.sum(rho * k0(q * s)) / np.pi,
        -q * cell * np.sum(rho * (r0[1] / s) * k1(q * s)) / np.pi,
    ])
    const = -1.0 / (2.0 * np.pi)
    for name, got, ref in zip("xyz", impl, set_eind):
        ratio = ref / got
        print(f"\nE{name}: set_Eind / set_Eq_Riboons = {ratio:.15f} "
              f"(= -1/(2 pi) = {const:.15f})")
        assert ratio.real == pytest.approx(const, rel=1e-12)
        assert abs(ratio.imag) < 1e-12 * abs(const)


def test_induced_field_normalisation_ratio_is_the_known_constant():
    """The constant above is exactly -1/(2 pi), to the last bit, nothing else.

    Companion to the test above, kept because it pins the scalar identity at the
    tightest tolerance the suite carries.  The ``np.trapz`` that used to stand in
    for ``induced_field_q``'s quadrature here is gone: ``induced_field_q`` sums
    uniform cells of width h (``operators.build_V``'s own weight), so the
    reference must too.  Matched, the ratio equals -1/(2 pi) to 0.0e+00
    relative; the old mismatched pair agreed only to 5.240e-09, which is why
    this test went red when the quadrature was corrected.  The rel=1e-12
    assertion is right and is deliberately NOT loosened -- it is informative
    precisely because both sides now use the same rule.
    """
    q, r0 = 0.06, (1.0, 4.0)
    x = np.linspace(-5.0, 5.0, 401)
    rho = np.exp(-(x**2) / 2.0) + 0.0j
    impl = np.asarray(observables.induced_field_q(q, x, rho, r0))
    dx = r0[0] - x
    s = np.sqrt(dx**2 + r0[1] ** 2)
    cell = float(x[1] - x[0])
    set_eind = -q * cell * np.sum(rho * dx / s * k1(q * s)) / np.pi
    ratio = set_eind / impl[0]
    print(f"\nset_Eind / set_Eq_Riboons = {ratio:.12f} (= -1/(2 pi) = {-1/(2*np.pi):.12f})")
    assert ratio.real == pytest.approx(-1.0 / (2.0 * np.pi), rel=1e-12)


# ===========================================================================
# 7. THE DECISIVE DRIFT TEST: dissymmetry vanishes at beta = 0, not at beta != 0
# ===========================================================================
def _dq_spectrum(beta: float, E_ev: float = 0.6, nq: int = 36):
    """Dq(q) for one ribbon driven by an in-plane dipole, both signs of q."""
    import sources
    sys_ = RibbonSystem([Ribbon(W_nm=10.0, L=31, beta=beta)], dx_nm=0.5)
    r0 = (2.0, 0.0, 6.0)      # off the sheet: no Coulomb singularity
    p = (1.0, 0.0, 0.0)       # in-plane, p_y = 0 -> phi_ext is even in q
    qpos = np.linspace(0.02, 0.60, nq)
    qs = np.concatenate([-qpos[::-1], qpos])
    dq = np.empty(qs.size)
    for i, q in enumerate(qs):
        phi_ext = sources.phi_ext_dipole(q, sys_.x_nodes[0], 0.0, r0, p)
        sol = sys_.solve_linear(q, E_ev, phi_ext)
        dq[i] = observables.ldos_q(q, sys_.x_nodes[0], sol["rho"][0], r0, p)
    return qs, dq


def test_dissymmetry_vanishes_without_drift():
    """beta = 0: Dq is even in q by symmetry, so g must vanish."""
    qs, dq = _dq_spectrum(beta=0.0)
    g = observables.dissymmetry(qs, dq)
    half = dq.size // 2
    asym = np.max(np.abs(dq[half:] - dq[:half][::-1])) / np.max(np.abs(dq))
    print(f"\nbeta=0: g = {g:.3e}, max relative q-parity violation of Dq = {asym:.3e}")
    assert abs(g) < 1e-10, f"undrifted dissymmetry {g:.3e} should vanish"


def test_dissymmetry_nonzero_with_drift():
    """beta = 0.3: the drift breaks q -> -q parity, so g must be clearly nonzero."""
    qs, dq = _dq_spectrum(beta=0.3)
    g = observables.dissymmetry(qs, dq)
    half = dq.size // 2
    asym = np.max(np.abs(dq[half:] - dq[:half][::-1])) / np.max(np.abs(dq))
    print(f"\nbeta=0.3: g = {g:.6e}, max relative q-parity violation = {asym:.3e}")
    assert abs(g) > 1e-3, (
        f"drifted dissymmetry {g:.3e} is not resolvable; with abs(q) in the drift "
        "conductivity it would be identically zero (see solver concern 2)"
    )


def test_dissymmetry_flips_with_drift_direction():
    """g is odd under beta -> -beta (reversing the drift reverses the handedness)."""
    g_p = observables.dissymmetry(*_dq_spectrum(beta=0.3))
    g_m = observables.dissymmetry(*_dq_spectrum(beta=-0.3))
    print(f"\ng(+0.3) = {g_p:.6e}, g(-0.3) = {g_m:.6e}")
    assert g_m == pytest.approx(-g_p, rel=1e-6)


# ===========================================================================
# Ribbon temperature (2026-11-05)
# ===========================================================================
# Until this date ``eta_pair`` took no temperature at all: the page's T box
# moved the sigma(omega) panel and left every resonance exactly where it was.
# T now rides on the ribbon, because it is a property of the sheet rather than
# of its surroundings, and these pin both halves of that: that T = 0 changes
# nothing, and that T > 0 changes the right thing by the right amount.


def test_eta_pair_at_T_zero_is_the_drude_eta() -> None:
    """The beta = 0 branch is still i sigma_Drude / (omega W eps_eff)."""
    sysm = RibbonSystem([Ribbon(W_nm=10.0, EF_ev=0.5, gamma_ev=0.02, L=31)])
    E = 0.6
    eta_perp, eta_par = sysm.eta_pair(0, 0.0, E)
    sigma = conductivity.graphene_drude(E, 0.5, 0.02)
    want = 1j * sigma / (units.omega_of(E) * 10.0 * sysm.eps_eff)
    assert eta_perp == pytest.approx(want, rel=1e-15)
    assert eta_par == pytest.approx(want, rel=1e-15)


def test_the_warm_drude_weight_collapses_to_EF_at_T_zero() -> None:
    """The identity the whole T = 0 reduction rests on, checked exactly.

    ``eta_pair`` at T > 0 replaces the Drude weight |E_F| with
    ``thermal_drude_weight(drift_chemical_potential(E_F, 0, T), T)``.  Both of
    those return |E_F| at T = 0 -- not to a tolerance, identically -- which is
    why setting no temperature cannot perturb a single published number here.
    """
    for EF in (0.05, 0.5, -0.8):
        mu = conductivity.drift_chemical_potential(EF, 0.0, 0.0)
        assert float(conductivity.thermal_drude_weight(mu, 0.0)) == abs(EF)


def test_ribbon_temperature_weakens_a_lightly_doped_sheet() -> None:
    """eta scales with the thermal Drude weight, and only with it.

    At E_F = 0.05 eV the chemical potential falls far enough by 300 K that the
    effect is percent-level rather than numerical noise.  The expected ratio is
    taken from conductivity.py's own closed forms, which test_conductivity.py
    checks against an exact Fermi-integral solve -- this test is about the
    plumbing from Ribbon.T_K to eta, not about the formula.
    """
    args = dict(W_nm=10.0, EF_ev=0.05, gamma_ev=0.02, L=31)
    cold = RibbonSystem([Ribbon(**args)])
    warm = RibbonSystem([Ribbon(T_K=300.0, **args)])
    E = 0.3
    ratio = abs(warm.eta_pair(0, 0.0, E)[0]) / abs(cold.eta_pair(0, 0.0, E)[0])

    mu = conductivity.drift_chemical_potential(0.05, 0.0, 300.0)
    expect = float(conductivity.thermal_drude_weight(mu, 300.0)) / 0.05
    assert ratio == pytest.approx(expect, rel=1e-12)
    assert 0.85 < ratio < 0.98, "300 K on a 0.05 eV sheet should be a percent-level cut"


def test_ribbon_temperature_reaches_the_drift_branch_too() -> None:
    """With drift, T goes through to graphene_drift_sigma rather than its default."""
    args = dict(W_nm=10.0, EF_ev=0.05, gamma_ev=0.02, beta=0.3, L=31)
    cold = RibbonSystem([Ribbon(**args)])
    warm = RibbonSystem([Ribbon(T_K=300.0, **args)])
    c = abs(cold.eta_pair(0, 0.1, 0.3)[1])
    w = abs(warm.eta_pair(0, 0.1, 0.3)[1])
    assert w < 0.98 * c


def test_ribbons_carry_independent_temperatures() -> None:
    """Two ribbons in one system, two temperatures, two different etas."""
    sysm = RibbonSystem(
        [
            Ribbon(W_nm=10.0, EF_ev=0.05, gamma_ev=0.02, L=31),
            Ribbon(W_nm=10.0, x0_nm=40.0, EF_ev=0.05, gamma_ev=0.02, L=31, T_K=300.0),
        ]
    )
    a = abs(sysm.eta_pair(0, 0.0, 0.3)[0])
    b = abs(sysm.eta_pair(1, 0.0, 0.3)[0])
    assert b < 0.98 * a


def test_the_fast_path_refuses_ribbons_that_differ_only_in_temperature() -> None:
    """sweep() evaluates eta ONCE per (q, E) and applies it to every block.

    So anything that makes two ribbons' eta differ has to appear in
    fast_path_obstruction, or the second ribbon is silently solved at the
    first one's parameters.  T_K was added to Ribbon on 2026-11-05 and was
    missing from that tuple; a review caught it.  The cost of being wrong here
    is invisible: no warning, no exception, just the wrong number.
    """
    args = dict(W_nm=10.0, EF_ev=0.05, gamma_ev=0.02, L=31)
    same = RibbonSystem([Ribbon(**args), Ribbon(x0_nm=40.0, **args)])
    assert same.fast_path_obstruction() is None

    warm = RibbonSystem([Ribbon(**args), Ribbon(x0_nm=40.0, T_K=300.0, **args)])
    reason = warm.fast_path_obstruction()
    assert reason is not None
    assert "T_K" in reason

    # Both ribbons at one temperature is still interchangeable.
    both = RibbonSystem(
        [Ribbon(T_K=300.0, **args), Ribbon(x0_nm=40.0, T_K=300.0, **args)]
    )
    assert both.fast_path_obstruction() is None


def test_the_slow_path_gives_each_ribbon_its_own_temperature() -> None:
    """The guard above is only worth having if the slow path is actually right."""
    args = dict(W_nm=10.0, EF_ev=0.05, gamma_ev=0.02, L=31)
    sysm = RibbonSystem([Ribbon(**args), Ribbon(x0_nm=40.0, T_K=300.0, **args)])
    cold = abs(sysm.eta_pair(0, 0.0, 0.3)[0])
    warm = abs(sysm.eta_pair(1, 0.0, 0.3)[0])
    assert warm < 0.98 * cold
