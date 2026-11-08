"""Tests for sources.py -- the external-potential excitations and the electron field.

Every assertion here is against a number: either a reference value computed once
and baked in, an exact symmetry/linearity relation, a fitted slope compared to its
analytic value, or an independent quadrature.  Nothing asserts merely that a call
returns.

There are no xfails left in this file.  The swift-electron prefactor dispute (was:
``1/gammaL`` or not) is settled in favour of ``gammaL^0``, so
``test_electron_prefactor_has_no_spurious_lorentz_factor`` is a plain regression
test, and ``test_electron_matches_lienard_wiechert_quadrature`` rebuilds the
Lienard-Wiechert double transform from scratch here rather than trusting the
closed form.

Run with:
    cd <repo root> && py -3.11 -m pytest scripts/electrostatics_fd/tests/test_sources.py -q
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k0, k1

# The package has no __init__.py (another agent owns any such file), so pytest's
# rootdir insertion only puts tests/ on sys.path. Add the package directory and
# its parent so `import sources` and `import units` resolve either way.
_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(_PKG_DIR), str(_PKG_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import sources  # noqa: E402
from units import C_NM_FS, E2_EV_NM, HBAR_EV_FS, omega_of  # noqa: E402

# --------------------------------------------------------------------------
# Reference configuration, shared by the swift-electron tests.
# Reference values were produced by sources.py itself and cross-checked by hand
# against the closed forms written out below; they pin conventions (signs,
# factors of 2, eps_a placement, exp(-i q y0)) against future edits.
# --------------------------------------------------------------------------
Q_REF = 0.05          # 1/nm
E_REF = 1.0           # eV
VOC_REF = 0.5         # v/c
B_REF = 10.0          # nm, impact parameter
Z_PLANE = 0.0         # nm, ribbon plane

GAMMA_AT_0P2 = 1.0206207261596576
GAMMA_AT_0P5 = 1.1547005383792517
LAM_REF = 0.05076461054559902           # 1/nm
OMEGA_REF = 1.5192674479961275          # 1/fs
V_REF = 149.896229                      # nm/fs
PHI_E_AT_X0 = -0.7156683885258904       # phi_ext_electron(Q,0,0,E,VOC,B)
PHI_E_AT_X7P5 = -0.7136016654990023 - 0.05434984206320088j


# ==========================================================================
# cross-module contract check
# ==========================================================================
def test_units_omega_matches_hbar_division() -> None:
    """omega_of must be E/HBAR; the electron source's phase depends on it."""
    assert float(omega_of(E_REF)) == pytest.approx(E_REF / HBAR_EV_FS, rel=1e-15)
    assert float(omega_of(E_REF)) == pytest.approx(OMEGA_REF, rel=1e-14)
    assert VOC_REF * C_NM_FS == pytest.approx(V_REF, rel=1e-15)


# ==========================================================================
# 1. plane wave
# ==========================================================================
def test_plane_wave_is_exactly_linear_in_x() -> None:
    """phi = -E0 x has a vanishing second difference on a uniform grid."""
    width, centre = 50.0, 12.0
    x = centre + width * np.linspace(-0.5, 0.5, 401)
    phi = sources.phi_ext_plane_wave(x, E0=0.75)
    assert np.abs(np.diff(phi, 2)).max() < 1e-13


def test_plane_wave_values_and_sign() -> None:
    """Exact nodewise values, including the physical (not normalised) offset."""
    x = np.array([-2.0, 0.0, 3.5, 11.25])
    phi = sources.phi_ext_plane_wave(x, E0=2.0)
    np.testing.assert_allclose(phi.real, [4.0, 0.0, -7.0, -22.5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(phi.imag, 0.0, rtol=0, atol=0.0)
    assert phi.dtype == np.complex128


def test_plane_wave_is_linear_in_E0_and_complex_safe() -> None:
    """A complex amplitude scales the potential without touching the x profile."""
    x = np.linspace(-5.0, 5.0, 17)
    base = sources.phi_ext_plane_wave(x, E0=1.0)
    amp = 0.3 - 1.7j
    np.testing.assert_allclose(
        sources.phi_ext_plane_wave(x, E0=amp), amp * base, rtol=1e-15, atol=0.0
    )


def test_plane_wave_gradient_is_minus_E0() -> None:
    """d(phi)/dx == -E0 exactly: the defining property of a uniform field."""
    x = np.linspace(-20.0, 20.0, 801)
    phi = sources.phi_ext_plane_wave(x, E0=1.25)
    slope = np.diff(phi.real) / np.diff(x)
    np.testing.assert_allclose(slope, -1.25, rtol=1e-13, atol=0.0)


# ==========================================================================
# 2. point dipole (paper 456 Eq. 20)
# ==========================================================================
def test_dipole_px_is_odd_in_dx() -> None:
    """p = (1,0,0): phi ~ dx * K1(Qa s)/s with s even in dx, so phi is odd."""
    q, r0 = 0.4, (1.5, 0.0, 4.0)
    dx = np.array([0.25, 0.9, 2.0, 5.5, 13.0])
    plus = sources.phi_ext_dipole(q, r0[0] + dx, 0.0, r0, (1.0, 0.0, 0.0))
    minus = sources.phi_ext_dipole(q, r0[0] - dx, 0.0, r0, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(minus, -plus, rtol=1e-14, atol=0.0)
    # and it is genuinely non-zero, so the symmetry is not trivially satisfied
    assert np.abs(plus).min() > 1e-6


def test_dipole_pz_is_even_in_dx() -> None:
    """p = (0,0,1): phi ~ dz * K1(Qa s)/s, even in dx at fixed dz."""
    q, r0 = 0.4, (1.5, 0.0, 4.0)
    dx = np.array([0.25, 0.9, 2.0, 5.5, 13.0])
    plus = sources.phi_ext_dipole(q, r0[0] + dx, 0.0, r0, (0.0, 0.0, 1.0))
    minus = sources.phi_ext_dipole(q, r0[0] - dx, 0.0, r0, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(minus, plus, rtol=1e-14, atol=0.0)
    assert np.abs(plus).min() > 1e-6


def test_dipole_py_is_even_in_dx_and_purely_imaginary_at_y0_zero() -> None:
    """The py term is -1j q py K0(Qa s): even in dx, and i-times-real at y0 = 0."""
    q, r0 = 0.4, (1.5, 0.0, 4.0)
    dx = np.array([0.25, 2.0, 5.5])
    plus = sources.phi_ext_dipole(q, r0[0] + dx, 0.0, r0, (0.0, 1.0, 0.0))
    minus = sources.phi_ext_dipole(q, r0[0] - dx, 0.0, r0, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(minus, plus, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(plus.real, 0.0, rtol=0, atol=1e-17)
    assert np.abs(plus.imag).min() > 1e-6


def test_dipole_reference_values() -> None:
    """Hard reference numbers pinning the factor 2, eps_a and the sign of each term."""
    q, r0 = 0.4, (0.0, 0.0, 4.0)
    px = complex(sources.phi_ext_dipole(q, 2.0, 0.0, r0, (1.0, 0.0, 0.0)))
    pz = complex(sources.phi_ext_dipole(q, 2.0, 0.0, r0, (0.0, 0.0, 1.0)))
    py = complex(sources.phi_ext_dipole(q, 2.0, 0.0, r0, (0.0, 1.0, 0.0), eps_a=2.0))
    assert px == pytest.approx(0.06633212875409547, rel=1e-13)
    assert pz == pytest.approx(-0.13266425750819094, rel=1e-13)
    assert py == pytest.approx(-0.05919292040048847j, rel=1e-13)

    # Independent closed-form evaluation of Eq. (20) at the same point.
    dx, dz = 2.0, -4.0
    s = np.hypot(dx, dz)
    assert px == pytest.approx(2.0 * q * dx * k1(q * s) / s, rel=1e-14)
    assert pz == pytest.approx(2.0 * q * dz * k1(q * s) / s, rel=1e-14)
    assert py == pytest.approx(2.0 * (-1j * q * k0(q * s)) / 2.0, rel=1e-14)


def test_dipole_carries_exp_minus_i_q_y0() -> None:
    """Displacing the dipole along y multiplies phi by exp(-1j q y0), nothing else."""
    q, y0 = 0.4, 1.25
    at_zero = complex(
        sources.phi_ext_dipole(q, 2.0, 0.0, (0.0, 0.0, 4.0), (1.0, 0.0, 0.0))
    )
    shifted = complex(
        sources.phi_ext_dipole(q, 2.0, 0.0, (0.0, y0, 4.0), (1.0, 0.0, 0.0))
    )
    assert shifted / at_zero == pytest.approx(np.exp(-1j * q * y0), rel=1e-14)
    assert shifted == pytest.approx(
        0.05821191948766116 - 0.03180131655469556j, rel=1e-13
    )


def test_dipole_is_linear_in_p() -> None:
    """Eq. (20) is linear in p, so a complex dipole decomposes exactly."""
    q, r0 = 0.3, (0.5, 0.75, 3.0)
    x = np.linspace(-6.0, 6.0, 25)
    p = np.array([0.4 - 0.2j, 1.0j, -0.6])
    combined = sources.phi_ext_dipole(q, x, 0.0, r0, p, eps_a=2.5 - 0.1j)
    parts = sum(
        p[i] * sources.phi_ext_dipole(q, x, 0.0, r0, np.eye(3)[i], eps_a=2.5 - 0.1j)
        for i in range(3)
    )
    np.testing.assert_allclose(combined, parts, rtol=1e-13, atol=0.0)


def test_dipole_scales_as_one_over_eps_a() -> None:
    q, r0 = 0.3, (0.0, 0.0, 3.0)
    x = np.linspace(-4.0, 4.0, 9)
    one = sources.phi_ext_dipole(q, x, 0.0, r0, (1.0, 0.0, 1.0), eps_a=1.0)
    four = sources.phi_ext_dipole(q, x, 0.0, r0, (1.0, 0.0, 1.0), eps_a=4.0)
    np.testing.assert_allclose(four, one / 4.0, rtol=1e-14, atol=0.0)


def test_dipole_raises_at_q_zero() -> None:
    with pytest.raises(ValueError, match=r"requires q != 0"):
        sources.phi_ext_dipole(0.0, np.array([1.0]), 0.0, (0.0, 0.0, 2.0),
                               (1.0, 0.0, 0.0))


def test_dipole_raises_when_sitting_on_a_node() -> None:
    """s -> 0 makes K1(Qa s)/s diverge; the guard must fire, not return inf."""
    x = np.linspace(-2.0, 2.0, 5)          # contains x = 0.0 exactly
    with pytest.raises(ValueError, match=r"sits on"):
        sources.phi_ext_dipole(0.4, x, 0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_dipole_in_plane_but_off_node_is_finite() -> None:
    """In-plane is allowed as long as no node coincides with the dipole."""
    x = np.array([-1.0, 0.5, 2.0])
    phi = sources.phi_ext_dipole(0.4, x, 0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert np.all(np.isfinite(phi))
    assert phi.dtype == np.complex128


def test_dipole_rejects_malformed_vectors() -> None:
    with pytest.raises(ValueError, match=r"r0_nm must be"):
        sources.phi_ext_dipole(0.4, 1.0, 0.0, (0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"p must be"):
        sources.phi_ext_dipole(0.4, 1.0, 0.0, (0.0, 0.0, 2.0), (1.0, 0.0))


def test_dipole_broadcasts_x_against_z() -> None:
    # no node at dx = 0, where a px dipole potential vanishes identically
    x = np.array([-3.0, -1.5, -0.5, 0.75, 1.5, 2.25, 3.0])
    z = np.array([[0.0], [1.0]])
    phi = sources.phi_ext_dipole(0.4, x, z, (0.0, 0.0, 5.0), (1.0, 0.0, 0.0))
    assert phi.shape == (2, 7)
    assert phi.dtype == np.complex128
    # z = 1.0 is nearer the dipole at z0 = 5, and K1(Qa s)/s decreases with s,
    # so every node must be strictly stronger there
    assert np.all(np.abs(phi[1]) > np.abs(phi[0]))
    assert np.abs(phi).min() > 1e-6


# ==========================================================================
# 3. swift electron -- kinematics
# ==========================================================================
def test_lorentz_factor_at_v_over_c_0p2() -> None:
    """gammaL(0.2) = 1/sqrt(0.96) = 1.0206207261596576, as stated in the spec."""
    assert sources.lorentz_gamma(0.2) == pytest.approx(GAMMA_AT_0P2, rel=1e-15)
    assert sources.lorentz_gamma(0.2) == pytest.approx(
        1.0 / np.sqrt(1.0 - 0.2 ** 2), rel=1e-15
    )
    assert sources.lorentz_gamma(VOC_REF) == pytest.approx(GAMMA_AT_0P5, rel=1e-15)


def test_lorentz_factor_rejects_unphysical_speeds() -> None:
    for bad in (0.0, -0.3, 1.0, 1.4):
        with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
            sources.lorentz_gamma(bad)


def test_decay_constant_reference_and_limits() -> None:
    """Lam = sqrt(q^2 + (omega/(v gammaL))^2)."""
    lam = sources.electron_decay_constant(Q_REF, E_REF, VOC_REF)
    assert lam == pytest.approx(LAM_REF, rel=1e-14)
    closed = np.sqrt(
        Q_REF ** 2 + (OMEGA_REF / (V_REF * GAMMA_AT_0P5)) ** 2
    )
    assert lam == pytest.approx(closed, rel=1e-14)
    # q = 0 leaves the purely kinematic part
    assert sources.electron_decay_constant(0.0, E_REF, VOC_REF) == pytest.approx(
        OMEGA_REF / (V_REF * GAMMA_AT_0P5), rel=1e-14
    )
    # E = 0 leaves |q|
    assert sources.electron_decay_constant(Q_REF, 0.0, VOC_REF) == pytest.approx(
        Q_REF, rel=1e-15
    )


# ==========================================================================
# 3. swift electron -- the two physically meaningful dependences
# ==========================================================================
@pytest.mark.parametrize("side", ["above", "below"])
def test_electron_decays_as_exp_minus_lam_abs_dz(side: str) -> None:
    """Fit the log-slope of |phi| in z and compare to -/+ Lam, rtol 1e-9."""
    lam = sources.electron_decay_constant(Q_REF, E_REF, VOC_REF)
    if side == "above":
        z = np.linspace(B_REF + 0.5, B_REF + 40.0, 211)
        expected = -lam
    else:
        z = np.linspace(B_REF - 40.0, B_REF - 0.5, 211)
        expected = +lam
    phi = sources.phi_ext_electron(Q_REF, 3.0, z, E_REF, VOC_REF, B_REF)
    slope, _ = np.polyfit(z, np.log(np.abs(phi)), 1)
    assert slope == pytest.approx(expected, rel=1e-9)


def test_electron_phase_advances_as_omega_over_v() -> None:
    """angle(phi(x2)) - angle(phi(x1)) == omega (x2-x1)/v, modulo 2 pi, rtol 1e-9."""
    x1, x2 = 3.0, 10.5
    p1 = complex(sources.phi_ext_electron(Q_REF, x1, Z_PLANE, E_REF, VOC_REF, B_REF))
    p2 = complex(sources.phi_ext_electron(Q_REF, x2, Z_PLANE, E_REF, VOC_REF, B_REF))
    expected = OMEGA_REF * (x2 - x1) / V_REF
    assert expected == pytest.approx(0.07601596074822506, rel=1e-13)

    raw = np.angle(p2) - np.angle(p1)
    # reduce the mismatch into (-pi, pi] so the comparison is modulo 2 pi
    assert abs(np.angle(np.exp(1j * (raw - expected)))) < 1e-12
    # wrap-safe estimator of the same difference, compared at rtol 1e-9
    assert np.angle(p2 / p1) == pytest.approx(expected, rel=1e-9)


def test_electron_phase_advance_wraps_consistently_over_many_turns() -> None:
    """Same relation over a 1000 nm run (1.6 turns), so the modulo really bites."""
    x1, dx = 3.0, 1000.0
    p1 = complex(sources.phi_ext_electron(Q_REF, x1, Z_PLANE, E_REF, VOC_REF, B_REF))
    p2 = complex(
        sources.phi_ext_electron(Q_REF, x1 + dx, Z_PLANE, E_REF, VOC_REF, B_REF)
    )
    expected = OMEGA_REF * dx / V_REF
    assert expected == pytest.approx(10.135461433096674, rel=1e-13)
    assert expected / (2.0 * np.pi) > 1.5          # genuinely wraps
    raw = np.angle(p2) - np.angle(p1)
    assert abs(np.angle(np.exp(1j * (raw - expected)))) < 1e-12


def test_electron_amplitude_is_flat_in_x() -> None:
    """exp(1j omega x/v) is unimodular: |phi| must not depend on x at all."""
    x = np.linspace(-25.0, 25.0, 101)
    phi = sources.phi_ext_electron(Q_REF, x, Z_PLANE, E_REF, VOC_REF, B_REF)
    np.testing.assert_allclose(np.abs(phi), abs(PHI_E_AT_X0), rtol=1e-13, atol=0.0)


def test_electron_reference_values() -> None:
    """Hard reference numbers for the full expression, prefactor carrying gammaL^0."""
    at0 = complex(
        sources.phi_ext_electron(Q_REF, 0.0, Z_PLANE, E_REF, VOC_REF, B_REF)
    )
    assert at0 == pytest.approx(PHI_E_AT_X0, rel=1e-13)
    assert at0.imag == pytest.approx(0.0, abs=1e-18)

    at7p5 = complex(
        sources.phi_ext_electron(Q_REF, 7.5, Z_PLANE, E_REF, VOC_REF, B_REF)
    )
    assert at7p5 == pytest.approx(PHI_E_AT_X7P5, rel=1e-13)

    # closed form, written out independently of sources.py -- no gammaL in the
    # prefactor; gammaL survives only inside LAM_REF
    closed = (
        -(2.0 * np.pi * E2_EV_NM) / (V_REF * LAM_REF)
        * np.exp(1j * OMEGA_REF * 7.5 / V_REF)
        * np.exp(-LAM_REF * abs(Z_PLANE - B_REF))
    )
    assert at7p5 == pytest.approx(closed, rel=1e-12)


def test_electron_returns_complex128_and_broadcasts() -> None:
    x = np.linspace(-10.0, 10.0, 8)
    z = np.array([[0.0], [-2.0], [-5.0]])
    phi = sources.phi_ext_electron(Q_REF, x, z, E_REF, VOC_REF, B_REF)
    assert phi.shape == (3, 8)
    assert phi.dtype == np.complex128
    # further from the beam => weaker; z = 0 is nearest b = 10
    assert np.all(np.abs(phi[0]) > np.abs(phi[1]))
    assert np.all(np.abs(phi[1]) > np.abs(phi[2]))


def test_electron_raises_when_lam_is_zero() -> None:
    with pytest.raises(ValueError, match=r"Lam .* == 0"):
        sources.phi_ext_electron(0.0, 1.0, Z_PLANE, 0.0, VOC_REF, B_REF)


# ==========================================================================
# 3. swift electron -- the aloof precondition (three cases)
# ==========================================================================
def test_trajectory_valid_when_aloof() -> None:
    ok, reason = sources.electron_trajectory_is_valid(10.0, 0.0)
    assert ok is True
    assert "aloof" in reason
    ok_stack, reason_stack = sources.electron_trajectory_is_valid(
        10.0, 0.0, layers=[object()]
    )
    assert ok_stack is True
    assert "aloof" in reason_stack


def test_trajectory_rejected_when_it_intersects_the_ribbon_plane() -> None:
    """Case 1: b == z_plane within 1e-9, with or without a substrate."""
    ok, reason = sources.electron_trajectory_is_valid(3.0, 3.0)
    assert ok is False
    assert "intersects" in reason

    ok_near, reason_near = sources.electron_trajectory_is_valid(3.0 + 5e-10, 3.0)
    assert ok_near is False
    assert "intersects" in reason_near

    # just outside the tolerance it is accepted again
    ok_out, _ = sources.electron_trajectory_is_valid(3.0 + 5e-9, 3.0)
    assert ok_out is True


def test_trajectory_rejected_below_the_plane_when_layers_are_given() -> None:
    """Case 2: with a stack, b < z_plane means the electron enters the substrate."""
    ok, reason = sources.electron_trajectory_is_valid(-4.0, 0.0, layers=[object()])
    assert ok is False
    assert "below" in reason


def test_trajectory_allowed_below_the_plane_when_free_standing() -> None:
    """Case 3: layers=None (and the empty stack) allow either side."""
    ok_none, reason_none = sources.electron_trajectory_is_valid(-4.0, 0.0, layers=None)
    assert ok_none is True
    assert "aloof" in reason_none
    ok_empty, _ = sources.electron_trajectory_is_valid(-4.0, 0.0, layers=[])
    assert ok_empty is True


def test_trajectory_validation_scans_every_ribbon_plane() -> None:
    """An array of plane heights is checked as a whole; the offender is named."""
    ok, reason = sources.electron_trajectory_is_valid(5.0, [0.0, 2.0, 5.0])
    assert ok is False
    assert "intersects" in reason and "5" in reason
    ok_all, _ = sources.electron_trajectory_is_valid(5.0, [0.0, 2.0, 4.0])
    assert ok_all is True


def test_phi_ext_electron_enforces_the_precondition() -> None:
    """The potential must refuse a non-aloof trajectory, not silently evaluate."""
    with pytest.raises(ValueError, match=r"intersects the ribbon plane"):
        sources.phi_ext_electron(Q_REF, 1.0, 4.0, E_REF, VOC_REF, 4.0)

    with pytest.raises(ValueError, match=r"passes below the ribbon plane"):
        sources.phi_ext_electron(
            Q_REF, 1.0, 0.0, E_REF, VOC_REF, -6.0, layers=[object()]
        )

    # free-standing, below the plane: allowed, and symmetric with the mirror height
    below = sources.phi_ext_electron(Q_REF, 1.0, 0.0, E_REF, VOC_REF, -6.0)
    above = sources.phi_ext_electron(Q_REF, 1.0, 0.0, E_REF, VOC_REF, +6.0)
    assert complex(below) == pytest.approx(complex(above), rel=1e-14)

    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
        sources.phi_ext_electron(Q_REF, 1.0, Z_PLANE, E_REF, 1.0, B_REF)


# ==========================================================================
# 3. swift electron -- the prefactor carries gammaL^0 (settled 2026-11-03)
# ==========================================================================
def test_electron_prefactor_has_no_spurious_lorentz_factor() -> None:
    """phi^ext = -(2 pi e^2)/(v Lam) ...: gammaL lives only inside Lam.

    Was a strict xfail against the project spec's -(2 pi e^2)/(v gammaL Lam).  The
    dispute is settled in favour of the value asserted here: the explicit gammaL of
    the boosted Lienard-Wiechert numerator cancels the 1/gammaL of the s -> gammaL s
    rescaling, the opposite integration ordering (no such Jacobian) agrees, and the
    intermediate t-transformed form is the published aloof potential of Garcia de
    Abajo, Rev. Mod. Phys. 82, 209 (2010), Sec. II, which carries no gammaL
    prefactor either.  Paper 430's gammaL^2 is a FIELD factor -- see
    test_electron_field_ratio_matches_paper_430_in_plane_weights.
    """
    at0 = complex(
        sources.phi_ext_electron(Q_REF, 0.0, Z_PLANE, E_REF, VOC_REF, B_REF)
    )
    derived = (
        -(2.0 * np.pi * E2_EV_NM) / (V_REF * LAM_REF)
        * np.exp(-LAM_REF * abs(Z_PLANE - B_REF))
    )
    assert derived == pytest.approx(-0.7156683885258904, rel=1e-13)
    assert at0 == pytest.approx(derived, rel=1e-12)

    # and the discarded spec form is wrong by exactly gammaL, not by something else
    spec = derived / GAMMA_AT_0P5
    assert at0 / spec == pytest.approx(GAMMA_AT_0P5, rel=1e-13)


def test_electron_prefactor_scales_as_one_over_v_at_fixed_lam() -> None:
    """gammaL^0: at fixed (Lam, E, b) the amplitude depends on v only through 1/v.

    Lam = sqrt(q^2 + (omega/(v gammaL))^2) is held fixed while v is scaled, by
    re-solving for the q that keeps it there.  Any surviving gammaL^n in the
    prefactor would show up as a residual gammaL ratio on top of v1/v2; with
    n = 0 the amplitude ratio is exactly v1/v2.
    """
    lam_target = 0.08                        # 1/nm, above the kinematic floor below
    amplitudes: list[tuple[float, float]] = []
    for voc in (0.1, 0.2, 0.3, 0.5, 0.7):
        v = voc * C_NM_FS
        kin = OMEGA_REF / (v * sources.lorentz_gamma(voc))
        assert kin < lam_target             # otherwise no real q reaches the target
        q = float(np.sqrt(lam_target ** 2 - kin ** 2))
        assert sources.electron_decay_constant(q, E_REF, voc) == pytest.approx(
            lam_target, rel=1e-13
        )
        phi = complex(
            sources.phi_ext_electron(q, 0.0, Z_PLANE, E_REF, voc, B_REF)
        )
        amplitudes.append((v, abs(phi)))

    v0, a0 = amplitudes[0]
    for v, a in amplitudes[1:]:
        assert a / a0 == pytest.approx(v0 / v, rel=1e-13)

    # absolute check of the same statement, against the closed prefactor
    for v, a in amplitudes:
        expected = (
            (2.0 * np.pi * E2_EV_NM) / (v * lam_target)
            * np.exp(-lam_target * abs(Z_PLANE - B_REF))
        )
        assert a == pytest.approx(expected, rel=1e-13)


# ==========================================================================
# 3. swift electron -- Lienard-Wiechert quadrature, built here from scratch
# ==========================================================================
def _lw_phi_quadrature_y_first(
    q: float, x: float, z: float, E_ev: float, v_over_c: float, b: float
) -> complex:
    """e*phi_q,omega of a uniformly moving charge, y step analytic, s step by quad.

    Start from the exact uniform-motion Lienard-Wiechert scalar potential
    phi = -e gammaL / sqrt(gammaL^2 (x - v t)^2 + y^2 + (z - b)^2), transform in y
    with int dy exp(-i q y)/sqrt(A^2 + y^2) = 2 K0(|q| |A|) and then in t via
    s = x - v t (dt = ds/v):

        phi_q,omega = -(2 e gammaL / v) exp(i omega x / v)
                      * int ds exp(-i omega s / v) K0(|q| sqrt(gammaL^2 s^2 + d^2))

    The integrand is even in s, so the sine branch drops and only the cosine
    branch is integrated (QAWO weighting).  Note this ordering has NO 1/gammaL
    Jacobian -- the numerator's gammaL survives and the s integral supplies the
    compensating 1/gammaL -- which is why it is a genuine check of the prefactor
    rather than a restatement of it.
    """
    gamma = 1.0 / np.sqrt(1.0 - v_over_c ** 2)
    v = v_over_c * C_NM_FS
    omega = float(omega_of(E_ev))
    d = z - b
    q_abs = abs(q)

    def kernel(s: float) -> float:
        return 2.0 * k0(q_abs * np.sqrt(gamma * gamma * s * s + d * d))

    s_max = 60.0 / (q_abs * gamma)
    half, _ = quad(
        kernel, 0.0, s_max, weight="cos", wvar=omega / v,
        limit=800, epsabs=1e-15, epsrel=1e-13,
    )
    return -(E2_EV_NM * gamma / v) * np.exp(1j * omega * x / v) * 2.0 * half


def _lw_phi_quadrature_t_first(
    q: float, x: float, z: float, E_ev: float, v_over_c: float, b: float
) -> complex:
    """Same object, opposite ordering: t step analytic, y step by quad.

    The t step gives the published aloof form
    phi_omega = -(2 e / v) exp(i omega x / v) K0(kappa sqrt(y^2 + d^2)) with
    kappa = omega / (gammaL v) (Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010),
    Sec. II), and the remaining y transform is done numerically instead of with
    int dy exp(-i q y) K0(kappa sqrt(y^2 + d^2)) = pi exp(-|d| Lam)/Lam.
    """
    gamma = 1.0 / np.sqrt(1.0 - v_over_c ** 2)
    v = v_over_c * C_NM_FS
    omega = float(omega_of(E_ev))
    kappa = omega / (gamma * v)
    d = z - b

    def kernel(y: float) -> float:
        return k0(kappa * np.sqrt(y * y + d * d))

    y_max = 40.0 / kappa
    half, _ = quad(
        kernel, 0.0, y_max, weight="cos", wvar=abs(q),
        limit=2000, epsabs=1e-16, epsrel=1e-13,
    )
    return -(2.0 * E2_EV_NM / v) * np.exp(1j * omega * x / v) * 2.0 * half


_LW_CASES = [
    (0.1, 0.01, 0.5),
    (0.1, 0.30, 10.0),
    (0.2, 0.05, 2.0),
    (0.2, 0.30, 0.5),
    (0.5, 0.01, 10.0),
    (0.5, 0.05, 2.0),
    (0.5, 0.30, 0.5),
]


@pytest.mark.parametrize("v_over_c, q, b", _LW_CASES)
def test_electron_matches_lienard_wiechert_quadrature(
    v_over_c: float, q: float, b: float
) -> None:
    """phi_ext_electron == the double transform done by quadrature, rtol 1e-9."""
    x, z = 1.7, 0.0
    lib = complex(sources.phi_ext_electron(q, x, z, E_REF, v_over_c, b))
    num = _lw_phi_quadrature_y_first(q, x, z, E_REF, v_over_c, b)
    assert num.real == pytest.approx(lib.real, rel=1e-9)
    assert num.imag == pytest.approx(lib.imag, rel=1e-9)
    assert abs(lib) > 1e-6                   # not a trivially satisfied comparison


@pytest.mark.parametrize("v_over_c, q, b", [(0.1, 0.05, 2.0), (0.5, 0.05, 2.0)])
def test_electron_lienard_wiechert_both_orderings_agree(
    v_over_c: float, q: float, b: float
) -> None:
    """The two integration orderings agree with each other and with the library.

    They differ in where the gammaL Jacobians sit (the y-first ordering has none,
    the t-first ordering has 1/gammaL from the s -> gammaL s rescaling), so their
    agreement is what shows the cancellation is not an artefact of one
    substitution.
    """
    x, z = -2.25, 0.0
    lib = complex(sources.phi_ext_electron(q, x, z, E_REF, v_over_c, b))
    a = _lw_phi_quadrature_y_first(q, x, z, E_REF, v_over_c, b)
    c = _lw_phi_quadrature_t_first(q, x, z, E_REF, v_over_c, b)
    assert a == pytest.approx(c, rel=1e-9)
    assert a == pytest.approx(lib, rel=1e-9)


def test_electron_quadrature_reproduces_the_transit_phase_as_a_complex_ratio() -> None:
    """quadrature/library == 1 + 0j at several x, not merely |.| == 1.

    A prefactor error would show in the modulus; a wrong transform convention
    (sign of omega, or exp(-i omega x/v)) would show only in the argument.  Both
    are pinned here.  The library's own x dependence is checked against
    exp(i omega x / v) in the same breath.
    """
    at0 = complex(
        sources.phi_ext_electron(Q_REF, 0.0, Z_PLANE, E_REF, VOC_REF, B_REF)
    )
    for x in (-5.0, -1.0, 0.0, 2.0, 7.0):
        lib = complex(
            sources.phi_ext_electron(Q_REF, x, Z_PLANE, E_REF, VOC_REF, B_REF)
        )
        num = _lw_phi_quadrature_y_first(
            Q_REF, x, Z_PLANE, E_REF, VOC_REF, B_REF
        )
        ratio = num / lib
        assert ratio.real == pytest.approx(1.0, rel=1e-9)
        assert abs(ratio.imag) < 1e-9
        # transit phase, complex, against the closed form
        assert lib / at0 == pytest.approx(
            np.exp(1j * OMEGA_REF * x / V_REF), rel=1e-13
        )


# ==========================================================================
# 3. swift electron -- the external FIELD components (paper 430's weights)
# ==========================================================================
def test_e_ext_electron_components_are_the_exact_multiples_of_phi() -> None:
    """E_x = -i(omega/v)gammaL^-2 phi, E_y = -i q phi, E_z = +Lam sign(z-b) phi."""
    x = np.array([-3.0, 0.0, 4.5])
    phi = sources.phi_ext_electron(Q_REF, x, Z_PLANE, E_REF, VOC_REF, B_REF)
    ex, ey, ez = sources.e_ext_electron(Q_REF, x, Z_PLANE, E_REF, VOC_REF, B_REF)

    np.testing.assert_allclose(
        ex, -1j * (OMEGA_REF / V_REF) / GAMMA_AT_0P5 ** 2 * phi, rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(ey, -1j * Q_REF * phi, rtol=1e-14, atol=0.0)
    # z = 0 < b = 10, so sign(z - b) = -1
    np.testing.assert_allclose(ez, -LAM_REF * phi, rtol=1e-14, atol=0.0)
    for comp in (ex, ey, ez):
        assert comp.dtype == np.complex128
        assert comp.shape == x.shape
        assert np.abs(comp).min() > 1e-6


def test_e_ext_electron_ex_over_ey_is_the_closed_ratio() -> None:
    """Ex/Ey == +(omega/(v q)) gammaL^-2 exactly, for several (v, q).

    The ratio is POSITIVE and real: both components carry the same factor -1j
    (E_x = -1j(omega/v)gammaL^-2 phi, E_y = -1j q phi), so it cancels.  A ratio of
    -(omega/(v q))gammaL^-2 would require E_y = +1j q phi, i.e. the opposite
    y-transform convention from the one this module declares
    (phi(y) = int dq/2pi phi_q exp(+i q y), hence d/dy -> +i q and
    E_y = -dphi/dy = -i q phi).  The signs of the individual components are pinned
    independently of this test, against finite-difference gradients, in
    test_e_ext_electron_matches_finite_difference_gradients -- and the paper-430
    comparison constrains only |Ex/Ey|.
    """
    for voc in (0.1, 0.2, 0.5, 0.8):
        gamma = sources.lorentz_gamma(voc)
        v = voc * C_NM_FS
        for q in (0.005, 0.05, 0.4):
            ex, ey, _ = sources.e_ext_electron(
                q, 2.0, Z_PLANE, E_REF, voc, B_REF
            )
            expected = (OMEGA_REF / (v * q)) / gamma ** 2
            ratio = complex(ex) / complex(ey)
            assert ratio.real == pytest.approx(expected, rel=1e-13)
            assert abs(ratio.imag) < 1e-15 * abs(ratio.real)


@pytest.mark.parametrize("v_over_c", [0.1, 0.2, 0.5])
@pytest.mark.parametrize("k_par", [0.02, 0.2])
def test_electron_field_ratio_matches_paper_430_in_plane_weights(
    v_over_c: float, k_par: float
) -> None:
    """|Ex/Ey| == paper 430's x-weight / y-weight, 12 digits.

    Paper 430's Gamma^(1) bracket weights the two in-plane mode-overlap integrals
    by omega*W/(v gammaL^2 g) and q/g, with lengths scaled by the ribbon width W so
    that g = Lam*W -- and with the paper's q slot holding the DIMENSIONLESS
    k_parallel*W, not the physical k_parallel.  The ratio of those two weights is
    W-independent and must equal |Ex/Ey| here.  Reference values are the six
    measured in the 2026-11-03 derivation record.
    """
    W, b, E_ev = 50.0, 1.0, 1.0
    expected_ratio = {
        (0.1, 0.02): 2.508526704691,
        (0.1, 0.20): 0.250852670469,
        (0.2, 0.02): 1.216255371972,
        (0.2, 0.20): 0.121625537197,
        (0.5, 0.02): 0.380079803741,
        (0.5, 0.20): 0.038007980374,
    }[(v_over_c, k_par)]

    ex, ey, _ = sources.e_ext_electron(k_par, 0.0, Z_PLANE, E_ev, v_over_c, b)
    measured = abs(complex(ex) / complex(ey))

    gamma = sources.lorentz_gamma(v_over_c)
    v = v_over_c * C_NM_FS
    omega = float(omega_of(E_ev))
    g = sources.electron_decay_constant(k_par, E_ev, v_over_c) * W
    q_paper = k_par * W                      # dimensionless, as the paper defines it
    x_weight = omega * W / (v * gamma ** 2 * g)
    y_weight = q_paper / g

    assert measured == pytest.approx(x_weight / y_weight, rel=1e-12)
    assert measured == pytest.approx(expected_ratio, rel=1e-12)

    # g = Lam*W is paper 430's own g^2 = q^2 + (omega W/(v gammaL))^2
    assert g ** 2 == pytest.approx(
        q_paper ** 2 + (omega * W / (v * gamma)) ** 2, rel=1e-13
    )

    # THE UNIT SLIP, pinned so nobody reintroduces it: feeding the physical
    # k_parallel into the paper's dimensionless q slot inflates the ratio by W.
    slipped = x_weight / (k_par / g)
    assert slipped / (x_weight / y_weight) == pytest.approx(W, rel=1e-13)


@pytest.mark.parametrize(
    "v_over_c, q", [(0.1, 0.02), (0.2, 0.05), (0.5, 0.2), (0.8, 0.05)]
)
def test_e_ext_electron_matches_finite_difference_gradients(
    v_over_c: float, q: float
) -> None:
    """Independent check of signs and factors: FD gradients of phi, not the paper.

    E_x = -gammaL^-2 d(phi)/dx (the A_x contribution cancels v^2/c^2 of the
    gradient), E_y = -d(phi)/dy on the inverse-transform kernel exp(+i q y), and
    E_z = -d(phi)/dz.  Central differences at h = 1e-4 nm; the leading FD error is
    O(h^2 (omega/v)^2) ~ 1e-12 relative, well inside the tolerance.
    """
    x0, z0, b = 3.25, 0.0, 6.0
    h = 1.0e-4
    gamma = sources.lorentz_gamma(v_over_c)

    def phi(x: float, z: float) -> complex:
        return complex(
            sources.phi_ext_electron(q, x, z, E_REF, v_over_c, b)
        )

    ex, ey, ez = sources.e_ext_electron(q, x0, z0, E_REF, v_over_c, b)

    d_dx = (phi(x0 + h, z0) - phi(x0 - h, z0)) / (2.0 * h)
    assert complex(ex) == pytest.approx(-d_dx / gamma ** 2, rel=1e-9)

    # y lives in Fourier space; reconstruct the y dependence with the module's
    # declared inverse kernel exp(+i q y) and difference that.
    phi_q = phi(x0, z0)
    d_dy = (
        phi_q * np.exp(1j * q * h) - phi_q * np.exp(-1j * q * h)
    ) / (2.0 * h)
    assert complex(ey) == pytest.approx(-d_dy, rel=1e-9)

    d_dz = (phi(x0, z0 + h) - phi(x0, z0 - h)) / (2.0 * h)
    assert complex(ez) == pytest.approx(-d_dz, rel=1e-9)

    # a wrong gammaL power on E_x would be caught: the raw gradient differs
    assert complex(ex) != pytest.approx(-d_dx, rel=1e-3)


def test_e_ext_electron_ez_kinks_across_the_trajectory_plane() -> None:
    """|E_z| is continuous at z = b but E_z flips sign; E_x, E_y do not.

    The sign(z - b) factor is discontinuous at the trajectory plane.  The aloof
    precondition guarantees z != b, so the kink is never evaluated, but it is real
    and any caller differentiating through it must know.
    """
    b, eps = 6.0, 1.0e-6
    below = sources.e_ext_electron(Q_REF, 1.0, b - eps, E_REF, VOC_REF, b)
    above = sources.e_ext_electron(Q_REF, 1.0, b + eps, E_REF, VOC_REF, b)

    assert abs(complex(below[2])) == pytest.approx(abs(complex(above[2])), rel=1e-9)
    assert complex(below[2]) == pytest.approx(-complex(above[2]), rel=1e-9)
    assert complex(below[0]) == pytest.approx(complex(above[0]), rel=1e-9)
    assert complex(below[1]) == pytest.approx(complex(above[1]), rel=1e-9)
    assert abs(complex(below[2])) > 1e-6

    # at exactly z = b the aloof guard fires rather than sign(0) = 0 leaking out
    with pytest.raises(ValueError, match=r"intersects the ribbon plane"):
        sources.e_ext_electron(Q_REF, 1.0, b, E_REF, VOC_REF, b)


def test_e_ext_electron_broadcasts_and_inherits_the_precondition() -> None:
    x = np.linspace(-10.0, 10.0, 8)
    z = np.array([[0.0], [-2.0], [-5.0]])
    ex, ey, ez = sources.e_ext_electron(Q_REF, x, z, E_REF, VOC_REF, B_REF)
    for comp in (ex, ey, ez):
        assert comp.shape == (3, 8)
        assert comp.dtype == np.complex128
    # all three decay away from the beam, by the same exp(-Lam |z - b|)
    for comp in (ex, ey, ez):
        assert np.all(np.abs(comp[0]) > np.abs(comp[1]))
        assert np.all(np.abs(comp[1]) > np.abs(comp[2]))

    with pytest.raises(ValueError, match=r"passes below the ribbon plane"):
        sources.e_ext_electron(
            Q_REF, 1.0, 0.0, E_REF, VOC_REF, -6.0, layers=[object()]
        )
    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
        sources.e_ext_electron(Q_REF, 1.0, Z_PLANE, E_REF, 1.0, B_REF)
    with pytest.raises(ValueError, match=r"Lam .* == 0"):
        sources.e_ext_electron(0.0, 1.0, Z_PLANE, 0.0, VOC_REF, B_REF)


def test_e_ext_electron_is_exported() -> None:
    assert "e_ext_electron" in sources.__all__
    assert all(name in dir(sources) for name in sources.__all__)


# ======================================================================================
# Reflected ("image") sources above a layered substrate
# ======================================================================================
#
# These are the tests that pin the reflected source, and they are analytic: a
# perfect mirror has a known answer, so none of this rests on agreeing with
# another numerical routine.

_MIRROR = lambda Q: np.full(np.shape(Q), -1.0 + 0j)          # noqa: E731
_NOTHING = lambda Q: np.zeros(np.shape(Q), dtype=complex)    # noqa: E731


def _const_r(value: complex):
    return lambda Q: np.full(np.shape(Q), complex(value))


@pytest.mark.parametrize(
    "p", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.3, -0.7, 1.2)]
)
def test_reflected_dipole_mirror_limit_is_the_textbook_image_dipole(p):
    """r = -1 must reproduce the image dipole (-px, -py, +pz) at -z0, exactly.

    This pins all three derivative signs at once, and the normal component
    against the two in-plane ones: get d/dz0 wrong and pz flips, get d/dx0
    wrong and px does. It is an analytic answer, not another code's output.
    """
    q, z, r0 = 0.05, 0.0, (1.5, 0.7, 4.0)
    x = np.linspace(-8.0, 8.0, 9)
    refl = sources.phi_ext_dipole_reflected(q, x, z, r0, p, _MIRROR, NQ=20000, Qmax=400.0)
    image = sources.phi_ext_dipole(
        q, x, z, (r0[0], r0[1], -r0[2]), (-p[0], -p[1], p[2])
    )
    rel = np.max(np.abs(refl - image)) / np.max(np.abs(image))
    assert rel < 2e-7, f"mirror limit off by {rel:.3e} for p = {p}"


@pytest.mark.parametrize("eps2", [2.0, 4.0, 11.7])
def test_reflected_dipole_single_interface_scales_with_the_fresnel_factor(eps2):
    """A Q-independent r just scales the mirror image by -r."""
    q, z, r0 = 0.05, 0.0, (1.5, 0.7, 4.0)
    x = np.linspace(-8.0, 8.0, 9)
    r_const = (1.0 - eps2) / (1.0 + eps2)
    refl = sources.phi_ext_dipole_reflected(
        q, x, z, r0, (0.0, 0.0, 1.0), _const_r(r_const), NQ=20000, Qmax=400.0
    )
    expect = (-r_const) * sources.phi_ext_dipole(
        q, x, z, (r0[0], r0[1], -r0[2]), (0.0, 0.0, 1.0)
    )
    rel = np.max(np.abs(refl - expect)) / np.max(np.abs(expect))
    assert rel < 1e-8


def test_reflected_dipole_vanishes_with_no_stack():
    q, z, r0 = 0.05, 0.0, (1.5, 0.7, 4.0)
    x = np.linspace(-8.0, 8.0, 9)
    refl = sources.phi_ext_dipole_reflected(q, x, z, r0, (0.3, -0.7, 1.2), _NOTHING)
    assert np.allclose(refl, 0.0, atol=1e-15)


def test_reflected_dipole_refuses_q_zero_and_a_sunken_image():
    x = np.linspace(-4.0, 4.0, 5)
    with pytest.raises(ValueError, match="divergent"):
        sources.phi_ext_dipole_reflected(0.0, x, 0.0, (0, 0, 4.0), (0, 0, 1), _MIRROR)
    with pytest.raises(ValueError, match=r"z \+ z0"):
        sources.phi_ext_dipole_reflected(0.05, x, 0.0, (0, 0, -4.0), (0, 0, 1), _MIRROR)


def test_reflected_electron_mirror_limit_is_the_image_charge():
    """r = -1 must give the image charge at -b, up to one named approximation.

    EVALUATED AT z > 0 ON PURPOSE. At z = 0 the source and its image are
    equidistant, both sides collapse to the same expression, and the test
    passes identically while proving nothing -- which is exactly what happened
    the first time it was written.
    """
    q, E, b, z = 0.05, 0.6, 5.0, 2.0
    x = np.linspace(-6.0, 6.0, 7)
    for v_over_c in (0.5, 0.2, 0.05):
        refl = sources.phi_ext_electron_reflected(q, x, z, E, v_over_c, b, _MIRROR)
        image = -sources.phi_ext_electron(q, x, z, E, v_over_c, -b)
        rel = np.max(np.abs(refl - image)) / np.max(np.abs(image))
        assert rel < 1e-3, f"v/c = {v_over_c}: {rel:.3e}"


def test_the_electron_residual_is_exactly_the_gamma_factor_and_nothing_else():
    """The whole mirror-limit discrepancy is |expm1(-(Q - Lam) z)|.

    Not "of that order": to a ratio of 1.000. That is what makes the
    approximation in this function a known quantity rather than a hope -- if
    anything else were wrong, this ratio would not be one.
    """
    q, E, b, z = 0.05, 0.6, 5.0, 2.0
    x = np.linspace(-6.0, 6.0, 7)
    for v_over_c in (0.3, 0.1, 0.03):
        refl = sources.phi_ext_electron_reflected(q, x, z, E, v_over_c, b, _MIRROR)
        image = -sources.phi_ext_electron(q, x, z, E, v_over_c, -b)
        actual = float(np.max(np.abs(refl - image)) / np.max(np.abs(image)))

        lam = float(sources.electron_decay_constant(q, E, v_over_c))
        Q = float(np.hypot(omega_of(E) / (v_over_c * C_NM_FS), q))
        predicted = abs(np.expm1(-(Q - lam) * z))
        assert actual == pytest.approx(predicted, rel=5e-3), (
            f"v/c = {v_over_c}: residual {actual:.4e} is not the gammaL term "
            f"{predicted:.4e}; something else is wrong"
        )


def test_the_electron_mirror_test_is_degenerate_on_the_interface():
    """Guard the guard: show that z = 0 cannot detect anything.

    Kept so nobody "simplifies" the test above back onto the interface and
    congratulates themselves on an exact zero.
    """
    q, E, b = 0.05, 0.6, 5.0
    x = np.linspace(-6.0, 6.0, 7)
    refl = sources.phi_ext_electron_reflected(q, x, 0.0, E, 0.5, b, _MIRROR)
    image = -sources.phi_ext_electron(q, x, 0.0, E, 0.5, -b)
    assert np.max(np.abs(refl - image)) == 0.0


def test_reflected_electron_vanishes_with_no_stack():
    x = np.linspace(-6.0, 6.0, 7)
    refl = sources.phi_ext_electron_reflected(0.05, x, 2.0, 0.6, 0.2, 5.0, _NOTHING)
    assert np.allclose(refl, 0.0, atol=1e-18)
