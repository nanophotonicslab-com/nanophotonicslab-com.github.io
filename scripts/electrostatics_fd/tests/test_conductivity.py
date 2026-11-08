"""Tests for ``electrostatics_fd.units`` and ``electrostatics_fd.conductivity``.

Three kinds of test live here:

* **Implementation** -- every public symbol of the two modules, checked against
  an independently written-out closed form and against hard-coded regression
  literals, plus shape/dtype/broadcast behaviour.
* **Validation** -- the reference numbers quoted in the module brief (the
  MATLAB drift chemical potential, the q = 0 collapse of the second-order
  tensor, the Q -> 0 drift limits, the removable 0/0 of Eqs. 12a-12b), and
  plus one *strict* xfail pinning the one place where the specified formulas
  are still mutually inconsistent (the finite-T drift chemical potential).  A
  strict xfail turns into a failure the moment somebody repairs the physics,
  which is the alert we want.  Two other strict xfails lived here until
  2026-11-03: the interband prefactor (a real factor-pi bug, now fixed in
  ``conductivity.py``) and the degenerate reduction of the cascaded C
  coefficient (never a bug at all -- the test asserted a false identity, and the
  real identity ``sigma_C^(2,2) = (C + D)/2`` is asserted here now).
* **Speed** -- the hot paths must vectorise: a single call over a large grid has
  to beat a Python loop over the same points by a wide margin.

Run with::

    cd <repo root>
    py -3.11 -m pytest scripts/electrostatics_fd/tests/test_conductivity.py -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# The package sits in <repo>/scripts/electrostatics_fd; make <repo>/scripts
# importable no matter where pytest is invoked from.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from electrostatics_fd import conductivity as cond  # noqa: E402
from electrostatics_fd import units as un  # noqa: E402

PI = np.pi
HBAR = un.HBAR_EV_FS
E2 = un.E2_EV_NM
VF = un.VF_NM_FS
KB = un.KB_EV_K
S2_REF = 1j * E2 ** 1.5 * VF ** 2 / (4.0 * PI * HBAR ** 2)
S3_REF = E2 ** 2 * VF ** 2 / (4.0 * PI * HBAR ** 2)
# xi of the finite-T closed-form chemical potential at E_F = 0.5 eV, beta = 0, 300 K
XI0 = (np.pi ** 2 / 3.0) * (KB * 300.0 / 0.5) ** 2


def _d(s: float, E: float, gamma: float) -> complex:
    """Reference harmonic denominator D_s = 1/(s omega + i gamma/hbar)."""
    return 1.0 / (s * E / HBAR + 1j * gamma / HBAR)


# --------------------------------------------------------------------------- #
# units.py
# --------------------------------------------------------------------------- #
def test_units_constants_have_the_specified_values():
    assert un.HBAR_EV_FS == 0.6582119569
    assert un.E2_EV_NM == 1.439964
    assert un.C_NM_FS == 299.792458
    assert un.KB_EV_K == 8.617333262e-5
    assert un.AU_NM == 0.05291772083
    assert un.AU_EV == 27.2113834
    assert un.VF_NM_FS == pytest.approx(0.9993081933333333, rel=1e-15)
    assert un.SIGMA_AU_TO_NM_FS == pytest.approx(un.E2_EV_NM / un.HBAR_EV_FS, rel=1e-15)
    assert un.SIGMA_AU_TO_NM_FS == pytest.approx(2.1876904314862955, rel=1e-12)


def test_units_omega_of_scalar_and_array():
    assert un.omega_of(1.0) == pytest.approx(1.5192674479961275, rel=1e-12)
    assert un.omega_of(0.0) == 0.0
    E = np.array([0.1, 0.5, 2.0])
    got = un.omega_of(E)
    assert isinstance(got, np.ndarray) and got.shape == (3,)
    np.testing.assert_allclose(got, E / HBAR, rtol=1e-15)


def test_units_sigma_conversions_roundtrip_and_scale():
    assert un.sigma_au_to_nm_fs(1.0) == pytest.approx(2.1876904314862955, rel=1e-12)
    assert un.sigma_nm_fs_to_au(un.SIGMA_AU_TO_NM_FS) == pytest.approx(1.0, rel=1e-15)
    z = np.array([1.0 + 2.0j, -0.5j, 3.0])
    back = un.sigma_nm_fs_to_au(un.sigma_au_to_nm_fs(z))
    assert back.dtype == np.complex128
    np.testing.assert_allclose(back, z, rtol=1e-15, atol=0.0)


# --------------------------------------------------------------------------- #
# Linear conductivities
# --------------------------------------------------------------------------- #
def test_drude_matches_eq8_and_regression_literal():
    # sigma = i e^2 E_F / [pi hbar (E + i gamma)]  (paper 406, Eq. 8)
    E, EF, gamma = 0.5, 0.5, 0.02
    want = 1j * E2 * EF / (PI * HBAR * (E + 1j * gamma))
    assert cond.graphene_drude(E, EF, gamma) == pytest.approx(want, rel=1e-15)
    # lossless limit: purely imaginary, positive (inductive free-carrier response)
    s0 = cond.graphene_drude(0.5, 0.5, 0.0)
    assert s0.real == 0.0
    assert s0.imag == pytest.approx(0.6963634922517707, rel=1e-12)


def test_drude_scales_as_one_over_E_plus_i_gamma():
    E = np.linspace(0.2, 3.0, 29)
    gamma = 0.02
    s = cond.graphene_drude(E, 0.4, gamma)
    assert s.dtype == np.complex128 and s.shape == E.shape
    prod = s * (E + 1j * gamma)  # must be E-independent
    np.testing.assert_allclose(prod, prod[0], rtol=1e-14)
    # linear in E_F
    np.testing.assert_allclose(
        cond.graphene_drude(E, 0.8, gamma), 2.0 * s, rtol=1e-14
    )


def test_interband_T0_is_lossless_below_the_pauli_threshold():
    s = cond.graphene_interband_T0(0.3, 0.5, 0.0)  # E < 2 E_F
    assert abs(s.real) < 1e-15
    assert s.imag < 0.0  # reactive, red-shifts the plasmons
    assert s.imag == pytest.approx(-0.1077690762516324, rel=1e-12, abs=0.0)


def test_interband_T0_log_prefactor_is_sigma0_over_pi():
    """The log prefactor is e^2/(4 pi hbar) = sigma0/pi, Hanson's form.

    SIGMA0_NM_FS itself stays the universal e^2/(4 hbar): it is the value the
    *absorptive* part saturates at, the extra pi coming from the branch cut of
    the logarithm, not from a convention choice (module docstring item 1).
    """
    assert cond.SIGMA0_NM_FS == pytest.approx(E2 / (4.0 * HBAR), rel=1e-15)
    E, EF, gamma = 1.7, 0.4, 0.03
    w = E + 1j * gamma
    want = 1j * (cond.SIGMA0_NM_FS / PI) * np.log((2.0 * EF - w) / (2.0 * EF + w))
    assert cond.graphene_interband_T0(E, EF, gamma) == pytest.approx(want, rel=1e-14)


def test_interband_T0_should_saturate_at_sigma0():
    s = cond.graphene_interband_T0(2.0, 0.5, 1e-7)
    assert s.real / cond.SIGMA0_NM_FS == pytest.approx(1.0, rel=1e-3)


def test_interband_T0_matches_the_finite_T_branch_in_the_cold_limit():
    """T=0 and T>=1K interband terms agree; they used to differ by exactly pi.

    The residual is the two O(gamma/(E + 2 mu)) terms that the finite-T
    arctan/log pair drops relative to the closed-form logarithm: 1.58e-3 real
    and 7.3e-6 imaginary at E = 1.2, E_F = 0.5, gamma = 0.02.
    """
    for E, EF, gamma, rel in (
        (0.3, 0.5, 1e-6, 1e-6),
        (1.2, 0.5, 0.02, 1e-4),
        (2.0, 0.5, 0.02, 1e-4),
        (0.9, 0.3, 0.01, 1e-4),
    ):
        T = 1.0
        s_t0 = cond.graphene_interband_T0(E, EF, gamma)
        mu = cond.thermal_drude_weight(EF, T)
        W = np.hypot(2.0 * KB * T, gamma)
        x = E - 2.0 * mu
        inter_T = cond.SIGMA0_NM_FS * (
            0.5
            + np.arctan(x / W) / PI
            + 1j * np.log((x ** 2 + W ** 2) / (E + 2.0 * mu) ** 2) / (2.0 * PI)
        )
        assert s_t0.imag / inter_T.imag == pytest.approx(1.0, rel=rel)
        assert s_t0 / inter_T == pytest.approx(1.0, rel=3e-3)


def test_local_rpa_T0_is_drude_plus_interband():
    E = np.linspace(0.2, 2.5, 17)
    EF, gamma = 0.5, 0.02
    want = cond.graphene_drude(E, EF, gamma) + cond.graphene_interband_T0(E, EF, gamma)
    np.testing.assert_allclose(cond.graphene_local_rpa(E, EF, gamma, 0.0), want, rtol=0.0, atol=0.0)
    # T_K below the 1 K switch takes the same branch
    np.testing.assert_allclose(cond.graphene_local_rpa(E, EF, gamma, 0.5), want, rtol=0.0, atol=0.0)


def test_local_rpa_finite_T_composition_and_saturation():
    E, EF, gamma, T = 1.2, 0.5, 0.02, 300.0
    mu = cond.thermal_drude_weight(EF, T)
    W = np.hypot(2.0 * KB * T, gamma)
    x = E - 2.0 * mu
    inter = cond.SIGMA0_NM_FS * (
        0.5
        + np.arctan(x / W) / PI
        + 1j * np.log((x ** 2 + W ** 2) / (E + 2.0 * mu) ** 2) / (2.0 * PI)
    )
    intra = 1j * E2 * mu / (PI * HBAR * (E + 1j * gamma))
    got = cond.graphene_local_rpa(E, EF, gamma, T)
    assert got == pytest.approx(intra + inter, rel=1e-14)
    # just above the smeared 2mu threshold the interband part is ~0.91 sigma0
    assert inter.real / cond.SIGMA0_NM_FS == pytest.approx(0.9139295683577147, rel=1e-9)
    # far above threshold it saturates at the universal sigma0
    far = cond.graphene_local_rpa(6.0, EF, gamma, T) - cond.graphene_drude(
        6.0, mu, gamma
    )
    assert far.real / cond.SIGMA0_NM_FS == pytest.approx(1.0, rel=5e-3)


def test_local_rpa_is_continuous_across_the_T_switch():
    """Both branches carry step height sigma0, so the T_K=1 switch is smooth.

    The residual is the two dropped O(gamma/(E + 2 mu)) terms of the finite-T
    arctan/log pair: warm/cold = 0.9971990 - 0.0006476j at this point, i.e.
    |warm/cold - 1| = 2.87e-3 against the rel = 1e-2 asserted below.  (On the
    interband part alone the cold/warm ratio is 1.0018535 + 0.0014443j.)
    """
    E, EF, gamma = 1.2, 0.5, 0.02
    cold = cond.graphene_local_rpa(E, EF, gamma, 0.999)
    warm = cond.graphene_local_rpa(E, EF, gamma, 1.001)
    assert warm == pytest.approx(cold, rel=1e-2)


def test_thermal_drude_weight_limits_and_overflow_guard():
    # T = 0 returns |mu|, which is the T -> 0+ limit of the formula itself,
    # so the branch is continuous for hole doping too (module docstring item 3)
    assert cond.thermal_drude_weight(0.5, 0.0) == 0.5
    assert cond.thermal_drude_weight(-0.5, 0.0) == 0.5
    assert cond.thermal_drude_weight(-0.5, 1e-6) == pytest.approx(0.5, rel=1e-12)
    # mu = 0: mu_D = 2 kT log 2
    assert cond.thermal_drude_weight(0.0, 300.0) == pytest.approx(
        2.0 * KB * 300.0 * np.log(2.0), rel=1e-14
    )
    # strongly negative mu/kT: no overflow, mu_D -> |mu|
    got = cond.thermal_drude_weight(-10.0, 300.0)
    assert np.isfinite(got) and got == pytest.approx(10.0, rel=1e-12)
    # degenerate doping: mu_D -> mu
    assert cond.thermal_drude_weight(0.5, 300.0) == pytest.approx(0.5, rel=1e-8)
    # monotonically increasing with T at fixed mu > 0
    temps = np.array([1.0, 100.0, 300.0, 1000.0, 3000.0])
    vals = np.array([cond.thermal_drude_weight(0.2, float(t)) for t in temps])
    assert np.all(np.diff(vals) > 0.0)
    # array input
    mus = np.array([-1.0, 0.0, 1.0])
    out = cond.thermal_drude_weight(mus, 300.0)
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    np.testing.assert_allclose(
        out, mus + 2.0 * KB * 300.0 * np.log1p(np.exp(-mus / (KB * 300.0))), rtol=1e-12
    )


# --------------------------------------------------------------------------- #
# Drift chemical potential
# --------------------------------------------------------------------------- #
def test_drift_chemical_potential_T0_closed_form():
    assert cond.drift_chemical_potential(0.5, 0.2, T_K=0) == pytest.approx(
        0.5 * (1.0 - 0.04) ** 0.75, rel=1e-12
    )
    assert cond.drift_chemical_potential(0.5, 0.0, T_K=0) == 0.5
    assert cond.drift_chemical_potential(0.5, 0.0, T_K=300.0) == pytest.approx(
        0.5 * np.sqrt(np.sqrt(1.0 + XI0 ** 2) - XI0), rel=1e-14
    )
    with pytest.raises(ValueError):
        cond.drift_chemical_potential(0.5, 1.0)
    with pytest.raises(ValueError):
        cond.drift_chemical_potential(0.5, -1.5)


def test_drift_chemical_potential_numeric_matches_matlab_reference():
    # MATLAB docstring: EF = 0.5 eV, beta = 0.2, T = 300 K -> mu = 0.4848 eV
    mu = cond.drift_chemical_potential_numeric(0.5, 0.2, 300.0)
    assert mu == pytest.approx(0.4848, abs=2e-3)
    assert mu == pytest.approx(0.4847890481691425, rel=1e-6)  # regression literal


def test_drift_chemical_potential_numeric_is_grid_converged():
    coarse = cond.drift_chemical_potential_numeric(0.5, 0.2, 300.0, n_E=1001, n_phi=121)
    fine = cond.drift_chemical_potential_numeric(0.5, 0.2, 300.0, n_E=4001, n_phi=481)
    assert fine == pytest.approx(coarse, rel=1e-7)


def test_drift_chemical_potential_numeric_recovers_EF_at_zero_drift():
    assert cond.drift_chemical_potential_numeric(0.5, 0.0, 300.0) == pytest.approx(
        0.5, rel=1e-9
    )


def test_drift_chemical_potential_numeric_is_monotone_in_beta():
    mus = [cond.drift_chemical_potential_numeric(0.5, b, 300.0) for b in (0.0, 0.2, 0.4)]
    assert mus[0] > mus[1] > mus[2]


def test_the_gap_between_the_two_drift_routes_is_the_thermal_shift():
    """The two routes answer different questions (module docstring item 2).

    This used to be recorded as a ~0.5 % *approximation error* in the closed
    form, with a strict-xfail asserting they "should match".  That premise was
    false: the numeric route ports a MATLAB root condition
    ``n(mu, beta, T) = n(E_F, 0, T)`` whose two sides are at the same T, so the
    thermal shift cancels identically and it returns the drift shift alone.
    The closed form carries both shifts.  The signature of a thermal shift,
    rather than an error in the drift part, is that the gap scales with T.
    """
    numeric_T = [cond.drift_chemical_potential_numeric(0.5, 0.0, T)
                 for T in (50.0, 300.0, 1500.0)]
    # no thermal shift at all in the numeric route: exactly E_F at every T
    for got in numeric_T:
        assert got == pytest.approx(0.5, abs=1e-8)

    # while the closed form shifts downward, monotonically and quadratically
    closed_T = [cond.drift_chemical_potential(0.5, 0.0, T)
                for T in (50.0, 300.0, 1500.0)]
    assert closed_T[0] > closed_T[1] > closed_T[2]

    # the beta != 0 gap therefore grows with T; it is not a fixed 0.5 % error
    gaps = []
    for T in (100.0, 300.0, 1000.0):
        c = cond.drift_chemical_potential(0.5, 0.2, T)
        nu = cond.drift_chemical_potential_numeric(0.5, 0.2, T)
        gaps.append(abs(c - nu) / abs(nu))
    assert gaps[0] < gaps[1] < gaps[2]
    assert gaps[0] < 1e-3 and gaps[2] > 1e-2

    # and as T -> 0 the two converge, because then there is no thermal shift
    assert cond.drift_chemical_potential(0.5, 0.2, 1.0) == pytest.approx(
        cond.drift_chemical_potential_numeric(0.5, 0.2, 1.0), rel=2e-3
    )


def test_finite_T_xi_has_the_exact_sommerfeld_coefficient():
    """``xi = (pi^2/3)(k_B T/mu0)^2`` is fixed by carrier conservation.

    Dirac-cone carrier conservation gives
    ``mu = E_F [1 - (pi^2/6)(k_B T/E_F)^2 + O(T^4)]``.  Expanding the library's
    bracket for small xi gives ``mu ~ mu0 (1 - xi/2)``, so the xi coefficient
    must be ``pi^2/3`` and no other constant will do.  The alternative
    reading ``(log 16)^2 = 7.687`` is 2.34x the required value -- an operator
    precedence away from ``2 log(4)^2``.
    """
    EF = 0.5
    # extract the coefficient numerically in the degenerate limit
    for T in (20.0, 40.0, 80.0):
        mu = cond.drift_chemical_potential(EF, 0.0, T)
        c = (1.0 - mu / EF) / (KB * T / EF) ** 2
        assert c == pytest.approx(np.pi ** 2 / 6.0, rel=2e-3)
    # the two rejected constants do not have this limit
    assert 2.0 * np.log(4.0) ** 2 / 2.0 != pytest.approx(np.pi ** 2 / 6.0, rel=0.1)
    assert np.log(16.0) ** 2 / 2.0 != pytest.approx(np.pi ** 2 / 6.0, rel=0.1)


def test_finite_T_closed_form_against_an_exact_carrier_conservation_oracle():
    """Independent oracle: root-solve the exact Dirac-cone Fermi integral.

    This is not the MATLAB reference -- it is the defining physical condition
    ``int_0^inf E [f(E-mu) - f(E+mu)] dE = E_F^2/2``, solved numerically.  The
    reference's own constant lands 3 decades worse here.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    def mu_exact(EF, T):
        kT = KB * T

        def f(x):
            return 1.0 / (np.exp(np.clip(x / kT, -700.0, 700.0)) + 1.0)

        def n(mu):
            val, _ = quad(lambda E: E * (f(E - mu) - f(E + mu)),
                          0.0, mu + 60.0 * kT + 5.0, limit=400)
            return val

        return brentq(lambda m: n(m) - EF ** 2 / 2.0, 1e-9, 2.0 * EF,
                      xtol=1e-13, rtol=1e-14)

    for EF, T, tol in ((0.5, 100.0, 1e-5), (0.5, 300.0, 1e-4), (1.0, 300.0, 1e-5)):
        assert cond.drift_chemical_potential(EF, 0.0, T) == pytest.approx(
            mu_exact(EF, T), rel=tol
        )


# --------------------------------------------------------------------------- #
# Drift-biased anisotropic nonlocal conductivity
# --------------------------------------------------------------------------- #
def _sigma_w(E: float, EF: float, gamma: float, beta: float, T_K: float = 0.0) -> complex:
    """Reference Drude weight sigma_w of paper 456 Eqs. 12a-12b."""
    mu = cond.drift_chemical_potential(EF, beta, T_K)
    mu_d = cond.thermal_drude_weight(mu, T_K)
    return 1j * E2 * mu_d / (PI * HBAR * (E + 1j * gamma))


def test_drift_sigma_reduces_to_drude_at_Q0_beta0():
    E, EF, gamma = 0.8, 0.45, 0.015
    s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, 0.0, 0.0)
    want = cond.graphene_drude(E, EF, gamma)
    assert s_perp == pytest.approx(want, rel=1e-10)
    assert s_par == pytest.approx(want, rel=1e-10)


def test_drift_sigma_Q0_limits_at_finite_beta():
    E, EF, gamma, beta = 0.5, 0.5, 0.02, 0.3
    s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, 0.0, beta)
    sw = _sigma_w(E, EF, gamma, beta)
    root = np.sqrt(1.0 - beta ** 2)
    assert s_par == pytest.approx((2.0 * sw / beta ** 2) * (1.0 - root), rel=1e-12)
    assert s_perp == pytest.approx((2.0 * sw / beta ** 2) * (1.0 / root - 1.0), rel=1e-12)
    # and NOT sigma_w: the drift bias renormalises the Q -> 0 weight
    assert abs(s_par / sw - 1.0) > 1e-2
    assert abs(s_perp / sw - 1.0) > 1e-2


def test_drift_sigma_matches_the_literal_eq12_brackets_away_from_the_singularity():
    """The implemented cancellation-free form equals Eqs. 12a-12b as written."""
    E = np.array([0.3, 0.8, 1.7])[:, None]
    Q = np.array([0.0, 0.05, 0.3, 1.0])[None, :]
    EF, gamma, beta = 0.5, 0.02, 0.25
    s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, Q, beta)
    alpha = VF * Q * HBAR / (E + 1j * gamma)
    b_perp, b_par = cond._drift_brackets_literal(alpha, beta)
    fct = 2.0 * _sigma_w(E, EF, gamma, beta) / (alpha + beta) ** 2
    np.testing.assert_allclose(s_perp, fct * b_perp, rtol=1e-11)
    np.testing.assert_allclose(s_par, fct * b_par, rtol=1e-11)


def test_drift_sigma_is_exact_where_the_literal_form_is_0_over_0():
    """alpha = -beta is a removable singularity; the limit is sigma_w/(1-b^2)^1.5."""
    E, EF, gamma, beta = 0.5, 0.5, 0.0, -0.2
    Q_sing = -beta * E / (VF * HBAR)  # makes alpha = +0.2 = -beta exactly
    s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, Q_sing, beta)
    sw = _sigma_w(E, EF, gamma, beta)
    limit = sw / (1.0 - beta ** 2) ** 1.5
    assert np.isfinite(s_perp) and np.isfinite(s_par)
    assert s_perp == pytest.approx(limit, rel=1e-12)
    assert s_par == pytest.approx(limit, rel=1e-12)
    # the literal Eqs. 12a-12b prefactor really does blow up there
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.complex128(-beta)
        b_perp, _ = cond._drift_brackets_literal(alpha, beta)
        literal = 2.0 * sw * b_perp / (alpha + beta) ** 2
    assert not np.isfinite(literal)


def test_drift_sigma_is_continuous_around_the_singular_point():
    E, EF, gamma, beta = 0.5, 0.5, 0.0, -0.2
    Q_sing = -beta * E / (VF * HBAR)
    s_perp0, s_par0 = cond.graphene_drift_sigma(E, EF, gamma, Q_sing, beta)
    sw = _sigma_w(E, EF, gamma, beta)
    for delta in (1e-3, 1e-6, 1e-9):
        for sign in (+1.0, -1.0):
            Q = Q_sing * (1.0 + sign * delta)
            s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, Q, beta)
            assert np.isfinite(s_perp) and np.isfinite(s_par)
            assert abs(s_perp - s_perp0) / abs(s_perp0) < 20.0 * delta
            assert abs(s_par - s_par0) / abs(s_par0) < 20.0 * delta
            assert abs(s_perp) < 100.0 * abs(sw)  # no blow-up
    # a dense sweep straight through the point stays finite and smooth
    Q = Q_sing * (1.0 + np.linspace(-1e-7, 1e-7, 4001))
    s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, Q, beta)
    assert np.all(np.isfinite(s_perp)) and np.all(np.isfinite(s_par))
    assert np.ptp(np.abs(s_perp)) / abs(s_perp0) < 1e-5
    # approaching along gamma instead of Q is equally well behaved
    for g in (1e-9, 1e-6, 1e-3):
        s_perp, s_par = cond.graphene_drift_sigma(E, EF, g, Q_sing, beta)
        assert np.isfinite(s_perp) and np.isfinite(s_par)


def test_drift_sigma_is_nonreciprocal_only_under_drift():
    E, EF, gamma, Q = 0.5, 0.5, 0.02, 0.05
    # beta = 0: even in Q
    fwd = cond.graphene_drift_sigma(E, EF, gamma, Q, 0.0)
    bwd = cond.graphene_drift_sigma(E, EF, gamma, -Q, 0.0)
    assert fwd[0] == pytest.approx(bwd[0], rel=1e-14)
    assert fwd[1] == pytest.approx(bwd[1], rel=1e-14)
    # beta != 0: sigma(Q) != sigma(-Q)
    fwd = cond.graphene_drift_sigma(E, EF, gamma, Q, 0.3)
    bwd = cond.graphene_drift_sigma(E, EF, gamma, -Q, 0.3)
    assert abs(fwd[0] - bwd[0]) / abs(fwd[0]) > 1e-3
    assert abs(fwd[1] - bwd[1]) / abs(fwd[1]) > 1e-3


def test_drift_sigma_anisotropy_and_overrides():
    E, EF, gamma, Q, beta = 0.6, 0.5, 0.02, 0.2, 0.4
    s_perp, s_par = cond.graphene_drift_sigma(E, EF, gamma, Q, beta)
    # the two components are genuinely different at finite Q and beta
    assert abs(s_perp - s_par) / abs(s_par) > 1e-3
    # sigma_par/sigma_perp = sqrt(1-alpha^2)/sqrt(1-beta^2) exactly
    alpha = VF * Q * HBAR / (E + 1j * gamma)
    ratio = np.sqrt(1.0 - np.complex128(beta) ** 2) / np.sqrt(1.0 - alpha ** 2)
    assert s_par / s_perp == pytest.approx(ratio, rel=1e-13)
    # mu_ev override bypasses the closed-form chemical potential
    mu_num = cond.drift_chemical_potential_numeric(EF, beta, 300.0)
    s_over = cond.graphene_drift_sigma(E, EF, gamma, Q, beta, T_K=300.0, mu_ev=mu_num)[0]
    s_closed = cond.graphene_drift_sigma(E, EF, gamma, Q, beta, T_K=300.0)[0]
    assert abs(s_over - s_closed) / abs(s_closed) > 1e-3
    assert s_over / s_closed == pytest.approx(
        cond.thermal_drude_weight(mu_num, 300.0)
        / cond.thermal_drude_weight(cond.drift_chemical_potential(EF, beta, 300.0), 300.0),
        rel=1e-13,
    )
    with pytest.raises(ValueError):
        cond.graphene_drift_sigma(E, EF, gamma, Q, 1.2)


# --------------------------------------------------------------------------- #
# Thin metal film
# --------------------------------------------------------------------------- #
def test_metal_thin_film_matches_the_thin_film_mapping():
    E, eps, t = 1.0, -10.0 + 1.0j, 20.0
    want = 1j * (E / HBAR) * t * (1.0 - eps) / (4.0 * PI)
    got = cond.metal_thin_film(E, eps, t)
    assert got == pytest.approx(want, rel=1e-15)
    assert got == pytest.approx(2.4179892422719274 + 26.597881664991203j, rel=1e-12)
    # vacuum film carries no sheet current; response is linear in thickness
    assert cond.metal_thin_film(E, 1.0, t) == 0.0
    assert cond.metal_thin_film(E, eps, 2.0 * t) == pytest.approx(2.0 * got, rel=1e-15)
    # free-electron metal: inductive, same sign as the graphene Drude term
    assert got.imag > 0.0
    assert cond.graphene_drude(E, 0.5, 0.02).imag > 0.0
    # array over energy
    Eg = np.linspace(0.2, 3.0, 11)
    arr = cond.metal_thin_film(Eg, eps, t)
    assert arr.dtype == np.complex128
    np.testing.assert_allclose(arr, 1j * (Eg / HBAR) * t * (1.0 - eps) / (4.0 * PI), rtol=1e-15)


# --------------------------------------------------------------------------- #
# Nonlinear conductivities
# --------------------------------------------------------------------------- #
def test_shg_abc_match_eqs_11_12():
    E, gamma = 1.3, 0.025
    d1, d2 = _d(1.0, E, gamma), _d(2.0, E, gamma)
    w = E / HBAR
    pref = -1.0 * S2_REF * d2 * d1
    a, b, c = cond.sigma_shg_abc(E, gamma, +1)
    assert a == pytest.approx(pref * (3.0 * d1 + 4.0 * d2), rel=1e-14)
    assert b == pytest.approx(pref * (-d1 + 4.0 * d2 - 4.0 / w), rel=1e-14)
    assert c == pytest.approx(pref * (-0.5 * d1 - 2.0 * d2 + 2.0 / w), rel=1e-14)
    # regression literals
    assert a == pytest.approx(-0.004545371038907847 - 0.10270839586477767j, rel=1e-10)
    assert b == pytest.approx(0.0017790048411603242 + 0.06166456571045563j, rel=1e-10)
    assert c == pytest.approx(9.851687223024744e-05 - 0.010295162843831444j, rel=1e-10)


def test_shg_is_odd_in_doping_sign():
    E, gamma = 0.9, 0.02
    plus = cond.sigma_shg_abc(E, gamma, +1)
    minus = cond.sigma_shg_abc(E, gamma, -1)
    for p, m in zip(plus, minus):
        assert m == pytest.approx(-p, rel=1e-15)
    assert cond.sigma_shg_1d_q0(E, gamma, -1) == pytest.approx(
        -cond.sigma_shg_1d_q0(E, gamma, +1), rel=1e-15
    )


def test_shg_1d_q0_collapse_identity_over_a_grid():
    """sigma_A + sigma_B + 2 sigma_C == -ds S2 D_2 D_1 (D_1 + 4 D_2), rtol 1e-12."""
    E = np.linspace(0.2, 3.0, 23)[:, None]
    gamma = np.array([0.0, 0.005, 0.02, 0.1])[None, :]
    for sign in (+1, -1):
        a, b, c = cond.sigma_shg_abc(E, gamma, sign)
        total = cond.sigma_shg_1d_q0(E, gamma, sign)
        np.testing.assert_allclose(a + b + 2.0 * c, total, rtol=1e-12)
        w = E / HBAR
        gf = gamma / HBAR
        d1 = 1.0 / (w + 1j * gf)
        d2 = 1.0 / (2.0 * w + 1j * gf)
        np.testing.assert_allclose(
            total, -float(sign) * S2_REF * d2 * d1 * (d1 + 4.0 * d2), rtol=1e-12
        )


def test_thg_and_kerr_match_eqs_14_15():
    E, EF, gamma = 1.3, 0.5, 0.025
    d1, d2, d3 = _d(1.0, E, gamma), _d(2.0, E, gamma), _d(3.0, E, gamma)
    w, gf = E / HBAR, gamma / HBAR
    thg = cond.sigma_thg_local(E, EF, gamma)
    kerr = cond.sigma_kerr(E, EF, gamma)
    assert thg == pytest.approx(3j * S3_REF / EF * d3 * d2 * d1, rel=1e-14)
    assert kerr == pytest.approx(-3j * S3_REF / EF * d2 / (w ** 2 + gf ** 2), rel=1e-14)
    # regression literals
    assert thg == pytest.approx(0.0017395340058374209 + 0.049322921371311126j, rel=1e-10)
    assert kerr == pytest.approx(-0.0014233614329444005 - 0.14802958902621763j, rel=1e-10)
    # both scale as 1/E_F
    assert cond.sigma_thg_local(E, 2.0 * EF, gamma) == pytest.approx(0.5 * thg, rel=1e-14)
    assert cond.sigma_kerr(E, 2.0 * EF, gamma) == pytest.approx(0.5 * kerr, rel=1e-14)
    # lossless limit: purely reactive, opposite signs
    thg0 = cond.sigma_thg_local(E, EF, 0.0)
    kerr0 = cond.sigma_kerr(E, EF, 0.0)
    assert abs(thg0.real) < 1e-18 and abs(kerr0.real) < 1e-18
    assert thg0.imag > 0.0 > kerr0.imag
    assert thg0.imag == pytest.approx(0.04936600830859749, rel=1e-10)
    assert kerr0.imag == pytest.approx(-0.1480980249257925, rel=1e-10)


def test_cascade_abcd_match_eqs_15a_15d():
    E, gamma, s1, s2 = 1.1, 0.02, 1.0, 2.0
    d1, d2 = _d(s1, E, gamma), _d(s2, E, gamma)
    dsum = _d(s1 + s2, E, gamma)
    w = E / HBAR
    a, b, c, d = cond.sigma_cascade_abcd(E, gamma, s1, s2, +1)
    pref = -1.0 * S2_REF * dsum
    assert a == pytest.approx(pref * (3.0 * d2 ** 2 + 2.0 * dsum * (d1 + d2)), rel=1e-14)
    assert b == pytest.approx(
        pref * (-4.0 * d1 / (s2 * w) - d2 ** 2 + 2.0 * dsum * (d1 + d2)), rel=1e-14
    )
    assert c == pytest.approx(pref * (4.0 * d1 / (s2 * w) - d2 ** 2), rel=1e-14)
    assert d == pytest.approx(2.0 * S2_REF * dsum ** 2 * (d1 + d2), rel=1e-14)
    # A, B, C are odd in doping_sign and so is D -- but D carries the opposite sign
    a2, b2, c2, d2c = cond.sigma_cascade_abcd(E, gamma, s1, s2, -1)
    for p, m in zip((a, b, c, d), (a2, b2, c2, d2c)):
        assert m == pytest.approx(-p, rel=1e-15)
    assert d / (2.0 * S2_REF * dsum ** 2 * (d1 + d2)) == pytest.approx(1.0, rel=1e-14)


#: (E [eV], gamma [eV], doping_sign) points for the degenerate-limit identities.
_DEGENERATE_POINTS = [
    (E, gamma, sign)
    for E in (0.4, 0.9, 1.3, 2.2)
    for gamma in (0.0, 0.005, 0.025)
    for sign in (+1, -1)
]


@pytest.mark.parametrize("E,gamma,sign", _DEGENERATE_POINTS)
def test_cascade_degenerate_reproduces_shg_A_and_B_one_to_one(E, gamma, sign):
    """A and B of Eq. 15 at s1=s2=1 ARE A and B of Eq. 12, coefficient for coefficient.

    Their two-ordering multiplicity sits in the term count of the cascade source
    sum, not in the coefficient, so these two reduce 1:1 (unlike C, which
    multiplies a different tensor structure -- see the test below).
    """
    a_s, b_s, _ = cond.sigma_shg_abc(E, gamma, sign)
    a_c, b_c, _, d_c = cond.sigma_cascade_abcd(E, gamma, 1.0, 1.0, sign)
    assert a_c == pytest.approx(a_s, rel=1e-12)
    assert b_c == pytest.approx(b_s, rel=1e-12)
    assert np.isfinite(d_c)


@pytest.mark.parametrize("E,gamma,sign", _DEGENERATE_POINTS)
def test_cascade_C_plus_D_halved_reproduces_degenerate_shg_C(E, gamma, sign):
    """sigma_C^(2,2) = (1/2)[sigma_C^(2,1,1) + sigma_D^(2,1,1)].

    Eq. 12c's C multiplies grad(E.E); Eq. 15c's C multiplies sum_j E_j grad(E_j),
    which is (1/2) grad(E.E) in the degenerate limit, and Eq. 15d's D multiplies
    grad(E.E) directly.  Comparing C against C alone -- which this test used to
    do under a strict xfail -- asserts a false identity; both equations are
    correct as published and as coded (module docstring item 4).
    """
    c_s = cond.sigma_shg_abc(E, gamma, sign)[2]
    _, _, c_c, d_c = cond.sigma_cascade_abcd(E, gamma, 1.0, 1.0, sign)
    assert c_s == pytest.approx(0.5 * (c_c + d_c), rel=1e-12)


@pytest.mark.parametrize("E,gamma,sign", _DEGENERATE_POINTS)
def test_cascade_A_plus_B_plus_C_plus_D_is_the_q0_collapse(E, gamma, sign):
    """A+B+C+D at s1=s2=1 == sigma_shg_1d_q0: the real per-ordering sum rule.

    In 1D at q = 0, grad(E_a.E_b) -> E_ax d_x E_bx + E_bx d_x E_ax feeds BOTH
    orderings, so each ordering's coefficient carries the FULL sigma_D.  That is
    the paper-406 MATLAB coefficient S_s1_s2 that nonlinear.py consumes.

    A+B+2C is NOT the cascade's collapse: that form belongs to the degenerate
    Eq. 12, whose third structure grad(E.E) contributes twice in 1D.  Applying
    it to the cascade (as this test used to, recording the mismatch as a defect)
    misses by a factor 2.667.
    """
    a, b, c, d = cond.sigma_cascade_abcd(E, gamma, 1.0, 1.0, sign)
    collapse = cond.sigma_shg_1d_q0(E, gamma, sign)
    assert a + b + c + d == pytest.approx(collapse, rel=1e-12)


@pytest.mark.parametrize("s1,s2", [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (1.0, 3.0),
                                   (3.0, 1.0), (2.0, 2.0), (2.0, 3.0)])
def test_cascade_per_ordering_collapse_matches_the_matlab_coefficient(s1, s2):
    """A+B+C+D == -ds S2 D_sum (D_s2^2 + 2 D_sum (D_s1 + D_s2)) for mixed harmonics."""
    E, gamma = 1.1, 0.02
    for sign in (+1, -1):
        a, b, c, d = cond.sigma_cascade_abcd(E, gamma, s1, s2, sign)
        d1, d2 = _d(s1, E, gamma), _d(s2, E, gamma)
        dsum = _d(s1 + s2, E, gamma)
        want = -float(sign) * S2_REF * dsum * (d2 ** 2 + 2.0 * dsum * (d1 + d2))
        assert a + b + c + d == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("E,gamma,sign", _DEGENERATE_POINTS)
def test_cascade_current_at_s1_s2_equal_one_is_twice_the_shg_current(E, gamma, sign):
    """Two-slot multiplicity at the level of the physical current.

    Eq. 15 writes two orderings, Eq. 12 one source term, so with E_1 = E_2 = E
    the cascade current is exactly 2x the SHG current.  Split per ordering, the
    single-slot kernel is

        K(a,b) = A E_a (div E_b) + B (E_a.grad) E_b
                 + C sum_j E_aj grad(E_bj) + (D/2) grad(E_a.E_b)

    and K(1,2) + K(2,1) is Eq. 15 identically while K(E,E) is Eq. 12 with no
    factor.  Checked here on an analytic curl-free field (the library's own
    quasistatic regime, E = -grad phi) with exact derivatives, using only the
    public coefficient functions -- no new API.
    """
    a_s, b_s, c_s = cond.sigma_shg_abc(E, gamma, sign)
    a_c, b_c, c_c, d_c = cond.sigma_cascade_abcd(E, gamma, 1.0, 1.0, sign)

    # phi = sin(k1 x) cosh(k2 y) + x^2 y ;  E = -grad phi, exactly curl-free.
    k1, k2 = 0.7, 0.4
    pts = [(0.3, -0.2), (1.1, 0.5), (-0.6, 0.9), (0.0, 0.0), (2.0, -1.3)]
    for x, y in pts:
        Ev = np.array([-(k1 * np.cos(k1 * x) * np.cosh(k2 * y) + 2.0 * x * y),
                       -(k2 * np.sin(k1 * x) * np.sinh(k2 * y) + x ** 2)])
        # dE[i, j] = d E_i / d x_j
        dE = np.array([
            [k1 ** 2 * np.sin(k1 * x) * np.cosh(k2 * y) - 2.0 * y,
             -(k1 * k2 * np.cos(k1 * x) * np.sinh(k2 * y) + 2.0 * x)],
            [-(k1 * k2 * np.cos(k1 * x) * np.sinh(k2 * y) + 2.0 * x),
             -k2 ** 2 * np.sin(k1 * x) * np.cosh(k2 * y)],
        ])
        div = dE[0, 0] + dE[1, 1]
        grad_EE = 2.0 * dE.T @ Ev           # grad(E.E) = 2 sum_j E_j grad(E_j)
        sum_EgradE = 0.5 * grad_EE          # the Eq. 15c structure, degenerate
        E_dot_grad_E = dE @ Ev              # (E.grad) E

        j12 = a_s * Ev * div + b_s * E_dot_grad_E + c_s * grad_EE
        kernel = (a_c * Ev * div + b_c * E_dot_grad_E
                  + c_c * sum_EgradE + 0.5 * d_c * grad_EE)
        j15 = 2.0 * kernel                  # K(1,1) + K(1,1), the two orderings
        np.testing.assert_allclose(j15, 2.0 * j12, rtol=1e-12)
        np.testing.assert_allclose(kernel, j12, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Vectorisation and speed
# --------------------------------------------------------------------------- #
def test_all_contract_symbols_are_exported():
    expected = [
        ("units", un, ["HBAR_EV_FS", "E2_EV_NM", "C_NM_FS", "VF_NM_FS", "KB_EV_K",
                       "AU_NM", "AU_EV", "SIGMA_AU_TO_NM_FS", "omega_of",
                       "sigma_au_to_nm_fs", "sigma_nm_fs_to_au"]),
        ("conductivity", cond, ["graphene_drude", "graphene_interband_T0",
                                "graphene_local_rpa", "thermal_drude_weight",
                                "drift_chemical_potential",
                                "drift_chemical_potential_numeric",
                                "graphene_drift_sigma", "metal_thin_film",
                                "sigma_shg_abc", "sigma_shg_1d_q0",
                                "sigma_thg_local", "sigma_kerr",
                                "sigma_cascade_abcd"]),
    ]
    for name, module, symbols in expected:
        missing = [s for s in symbols if not hasattr(module, s)]
        assert not missing, f"{name} is missing {missing}"


def test_array_returns_are_complex128_and_shape_preserving():
    E = np.linspace(0.2, 2.0, 7)
    EF, gamma = 0.5, 0.02
    for out in (
        cond.graphene_drude(E, EF, gamma),
        cond.graphene_interband_T0(E, EF, gamma),
        cond.graphene_local_rpa(E, EF, gamma, 0.0),
        cond.graphene_local_rpa(E, EF, gamma, 300.0),
        cond.metal_thin_film(E, -5.0 + 0.5j, 10.0),
        cond.sigma_shg_1d_q0(E, gamma),
        cond.sigma_thg_local(E, EF, gamma),
        cond.sigma_kerr(E, EF, gamma),
    ):
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.complex128
        assert out.shape == E.shape
        assert np.all(np.isfinite(out))
    for tup in (
        cond.sigma_shg_abc(E, gamma),
        cond.graphene_drift_sigma(E, EF, gamma, 0.1, 0.2),
        cond.sigma_cascade_abcd(E, gamma, 1.0, 2.0),
    ):
        for out in tup:
            assert isinstance(out, np.ndarray) and out.dtype == np.complex128
            assert out.shape == E.shape and np.all(np.isfinite(out))


def test_broadcasting_over_energy_and_wavevector_grids():
    E = np.linspace(0.3, 1.5, 9)[:, None]
    Q = np.linspace(0.0, 0.5, 6)[None, :]
    s_perp, s_par = cond.graphene_drift_sigma(E, 0.5, 0.02, Q, 0.25)
    assert s_perp.shape == (9, 6) and s_par.shape == (9, 6)
    # row/column slices agree with the scalar calls
    for i in (0, 4, 8):
        for j in (0, 3, 5):
            one = cond.graphene_drift_sigma(
                float(E[i, 0]), 0.5, 0.02, float(Q[0, j]), 0.25
            )
            assert s_perp[i, j] == pytest.approx(one[0], rel=1e-15)
            assert s_par[i, j] == pytest.approx(one[1], rel=1e-15)


@pytest.mark.parametrize(
    "fn",
    [
        lambda E: cond.graphene_local_rpa(E, 0.5, 0.02, 300.0),
        lambda E: cond.graphene_drift_sigma(E, 0.5, 0.02, 0.1, 0.2)[0],
        lambda E: cond.sigma_shg_1d_q0(E, 0.02),
        lambda E: cond.sigma_thg_local(E, 0.5, 0.02),
    ],
)
def test_vectorised_calls_beat_a_python_loop_by_at_least_5x(fn):
    """Speed: the hot paths must be real array ops, not hidden per-node loops."""
    n_vec, n_loop = 200_000, 2_000
    E_vec = np.linspace(0.2, 3.0, n_vec)
    E_loop = np.linspace(0.2, 3.0, n_loop)

    t0 = time.perf_counter()
    out = fn(E_vec)
    t_vec = (time.perf_counter() - t0) / n_vec

    t0 = time.perf_counter()
    for e in E_loop:
        fn(float(e))
    t_loop = (time.perf_counter() - t0) / n_loop

    assert np.all(np.isfinite(out))
    assert t_vec * 5.0 < t_loop, (
        f"vectorised {t_vec * 1e9:.1f} ns/point vs loop {t_loop * 1e9:.1f} ns/point"
    )
    # generous absolute ceiling: 200k energies in well under a second
    assert t_vec * n_vec < 1.0


def test_vectorised_results_equal_the_scalar_calls_exactly():
    E = np.linspace(0.2, 3.0, 257)
    EF, gamma = 0.5, 0.02
    vec = cond.graphene_local_rpa(E, EF, gamma, 300.0)
    for i in (0, 1, 63, 128, 256):
        # arctan/log have SIMD and scalar inner loops in numpy, so allow 1 ULP
        assert vec[i] == pytest.approx(
            cond.graphene_local_rpa(float(E[i]), EF, gamma, 300.0), rel=1e-15
        )
    vec_perp, vec_par = cond.graphene_drift_sigma(E, EF, gamma, 0.2, 0.3)
    for i in (0, 17, 200, 256):
        one = cond.graphene_drift_sigma(float(E[i]), EF, gamma, 0.2, 0.3)
        assert vec_perp[i] == one[0]
        assert vec_par[i] == one[1]


def test_numeric_chemical_potential_cost_is_bounded():
    """Speed: the root-find is the only expensive call in the module."""
    t0 = time.perf_counter()
    cond.drift_chemical_potential_numeric(0.5, 0.2, 300.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"drift_chemical_potential_numeric took {elapsed:.2f} s"


# --------------------------------------------------------------------------- #
# Electron-hole symmetry (module docstring item 3)
# --------------------------------------------------------------------------- #
def test_every_conductivity_is_even_in_the_fermi_level():
    """The Dirac cone is exactly electron-hole symmetric, so sigma(-E_F) = sigma(+E_F).

    Before 2026-11-03 the T = 0 linear and the third-order functions were
    exactly *odd* in E_F instead, which is the sign error documented as module
    docstring item 3.  Second order is excluded on purpose: it carries an
    explicit ``doping_sign``, which is the physical ``-sign(E_F)`` of the
    reference, correctly separated from the magnitude.
    """
    E = np.linspace(0.15, 3.0, 40)
    g, EF = 0.02, 0.5

    for name, f in (
        ("graphene_drude", lambda ef: cond.graphene_drude(E, ef, g)),
        ("graphene_interband_T0", lambda ef: cond.graphene_interband_T0(E, ef, g)),
        ("graphene_local_rpa T=0", lambda ef: cond.graphene_local_rpa(E, ef, g, 0.0)),
        ("graphene_local_rpa T=300", lambda ef: cond.graphene_local_rpa(E, ef, g, 300.0)),
        ("sigma_thg_local", lambda ef: cond.sigma_thg_local(E, ef, g)),
        ("sigma_kerr", lambda ef: cond.sigma_kerr(E, ef, g)),
    ):
        plus, minus = np.asarray(f(+EF)), np.asarray(f(-EF))
        scale = np.max(np.abs(plus))
        assert np.max(np.abs(plus - minus)) <= 1e-14 * scale, f"{name} is not even in E_F"

    # the drift conductivity, both components, at zero and finite drift
    for beta in (0.0, 0.3):
        for comp_p, comp_m in zip(
            cond.graphene_drift_sigma(0.6, +EF, g, 0.2, beta),
            cond.graphene_drift_sigma(0.6, -EF, g, 0.2, beta),
        ):
            assert comp_p == pytest.approx(comp_m, rel=1e-14)

    # and the chemical potential and Drude weight that feed them
    for T in (0.0, 300.0):
        assert cond.drift_chemical_potential(+EF, 0.3, T) == pytest.approx(
            cond.drift_chemical_potential(-EF, 0.3, T), rel=1e-14
        )
        assert cond.thermal_drude_weight(+EF, T) == pytest.approx(
            cond.thermal_drude_weight(-EF, T), rel=1e-14
        )


def test_hole_doping_does_not_produce_optical_gain():
    """Passive doped graphene absorbs: Re sigma >= 0 at both doping signs.

    This is the observable consequence of the old sign error -- with sigma odd
    in E_F, every hole-doped point had Re sigma_inter < 0, i.e. negative
    absorption, and the Drude term had both parts flipped, which destroys the
    plasmon rather than shifting it.
    """
    E = np.linspace(0.15, 3.0, 60)
    g = 0.02
    for EF in (+0.5, -0.5, +0.2, -0.2):
        for name, sig in (
            ("drude", cond.graphene_drude(E, EF, g)),
            ("interband", cond.graphene_interband_T0(E, EF, g)),
            ("rpa T=0", cond.graphene_local_rpa(E, EF, g, 0.0)),
            ("rpa T=300", cond.graphene_local_rpa(E, EF, g, 300.0)),
        ):
            assert np.all(np.real(np.asarray(sig)) >= -1e-15), (
                f"{name} gives optical gain at E_F = {EF}"
            )

    # the plasmon-carrying imaginary part of the Drude term keeps its sign too
    assert np.all(np.imag(np.asarray(cond.graphene_drude(E, -0.5, g))) > 0.0)
