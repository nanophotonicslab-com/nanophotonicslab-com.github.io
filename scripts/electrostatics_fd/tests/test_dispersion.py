"""Tests for ``dispersion.py`` -- the three-level dispersion stack.

This file was written against a module that had never been executed.  Every
number quoted in ``dispersion.py``'s docstrings is re-measured here rather than
trusted, and the two CROSS-LEVEL checks at the bottom are the real validation:
a very wide ribbon must approach the extended sheet (Level 1 vs Level 2) and the
peak-tracked LDOS bands must sit on the eigenvalue bands within the linewidth
(Level 2 vs Level 3).

Reference values
----------------
* Bright ribbon eigenvalues ``eta = -0.0687, -0.0184, -0.0107`` with an
  interleaved DARK mode at ``-0.0289``: Cox, Yu and Garcia de Abajo, Phys. Rev.
  B 96, 045442 (2017), Table S2.  Same anchors as
  ``tests/test_solver.py``.
* ``hbar omega_1 = 0.578 eV`` for ``E_F = 0.5 eV``, ``W = 10 nm``,
  ``eta_1 = -0.0687``, from the Drude closed form
  ``hbar omega = sqrt(-e^2 E_F / (pi eps_bar eta W))``.
* Level 1 long-wavelength limit ``(hbar omega)^2 = 4 e^2 E_F |Q| / (eps_a +
  eps_b)``, derived here from the pole condition rather than quoted.
* Level 2 linewidth closed form ``FWHM = gamma (E^2 + gamma^2) / E^2``, derived
  here from ``eta(E) = -A (E - i gamma) / (E (E^2 + gamma^2))``.
* An exploratory sheet-dispersion kernel in ATOMIC units, reproduced inline so
  that both of its differences from this module -- the transverse/longitudinal
  component choice, and the eV/nm/atomic-unit mix in its kernel -- are measured
  here rather than asserted.

NOT TESTED ABSOLUTELY, ON PURPOSE: the magnitude of the Level 3 map.
``observables.induced_field_q`` carries an unresolved overall constant and its
quadrature weights are being changed concurrently, so every Level 3 assertion
here is on a shape, a peak position, a ratio or a symmetry.  Where an absolute
value is needed it is computed from the library inside the test.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

# The package has no guaranteed import route (it is used both as
# ``electrostatics_fd.dispersion`` with scripts/ on the path and as a flat
# module directory), so make both work.
_PKG = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_PKG), str(_PKG.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conductivity  # noqa: E402
import dispersion  # noqa: E402
import observables  # noqa: E402
import operators  # noqa: E402
import solver  # noqa: E402
import sources  # noqa: E402
import substrate  # noqa: E402
from units import E2_EV_NM, HBAR_EV_FS, VF_NM_FS, omega_of  # noqa: E402

# Acceptance anchors (never tuned, never relaxed).
BRIGHT_ETA = (-0.0687, -0.0184, -0.0107)
DARK_ETA = -0.0289
EF_REF = 0.5
W_REF = 10.0
E1_REF = 0.578


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ribbon_system(L=61, **kw):
    """Single-ribbon system with the reference parameters."""
    params = dict(W_nm=W_REF, EF_ev=EF_REF, gamma_ev=0.02, L=L)
    params.update(kw)
    return solver.RibbonSystem(solver.Ribbon(**params))


def _imr_peak(f, lo, hi):
    """Location of the maximum of a scalar function on ``[lo, hi]``."""
    res = minimize_scalar(
        lambda w: -f(w), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11}
    )
    assert res.success
    return float(res.x)


def _matlab_sigma_au(EF_ev, gamma_ev, w_ev, Q_inv_nm, beta):
    """The drift intraband conductivity transcribed literally: ATOMIC units.

    Same constants as the .m file (they differ in the last digits from
    ``units.py``, which is part of what is being tested -- the .m file is the
    reference, not this package).
    """
    au_nm = 0.05291772083
    au_eV = 27.2113834
    vF = 137.03599971 / 300.0
    EF = EF_ev / au_eV
    gam = gamma_ev / au_eV
    w = np.asarray(w_ev, dtype=np.complex128) / au_eV
    vFQ = vF * Q_inv_nm * au_nm
    wg = w + 1j * gam
    alpha = vFQ / wg
    sigma_0 = 1j / np.pi * EF / wg
    sa = np.sqrt(1.0 - alpha**2)
    sb = np.sqrt(1.0 - np.complex128(beta) ** 2)
    den = (1.0 + alpha * beta) + sa * sb
    return 2.0 * sigma_0 / (sb * den), 2.0 * sigma_0 / (sa * den)


def _matlab_imr(w_ev, Q_inv_nm, component, EF_ev=1.0, gamma_ev=0.1):
    """``r = 1./(1-1i*W./(2*pi*sigma.*abs(Q)))`` of the .m file, units as written."""
    sig = _matlab_sigma_au(EF_ev, gamma_ev, w_ev, Q_inv_nm, 0.0)[component]
    return float(np.imag(1.0 / (1.0 - 1j * w_ev / (2.0 * np.pi * sig * abs(Q_inv_nm)))))


def _module_imr(w_ev, Q_inv_nm, component, EF_ev=1.0, gamma_ev=0.1):
    sig = conductivity.graphene_drift_sigma(w_ev, EF_ev, gamma_ev, Q_inv_nm, 0.0)[
        component
    ]
    return float(np.imag(dispersion.sheet_reflection_sigma(sig, Q_inv_nm, w_ev)))


#: The sweep the tracking defect was reported on: one 10 nm ribbon over a +-q
#: grid reaching down to |q| = 0.02 / nm, which is where the longitudinal
#: branch's root drops out of a window whose floor is 0.15 eV.
_DEFECT_QS = dispersion.symmetric_q_grid(0.5, 20, include_zero=False, q_min=0.02)
_DEFECT_KW = dict(n_bands=3, E_min=0.15, E_max=1.2)
_DEFECT_CACHE = {}


def _defect_bands(track):
    """``ribbon_bands`` over :data:`_DEFECT_QS`, memoised.

    Four tests read the same two sweeps and each one is 40 eigendecompositions
    at ``L = 121`` (~2 s), so they are computed once.  Nothing mutates a
    ``BandStructure``, so sharing it is safe.
    """
    key = bool(track)
    if key not in _DEFECT_CACHE:
        _DEFECT_CACHE[key] = dispersion.ribbon_bands(
            _ribbon_system(L=121), _DEFECT_QS, track=key, **_DEFECT_KW
        )
    return _DEFECT_CACHE[key]


def _max_step(col):
    """Largest jump in one band between adjacent ``q``, skipping the gaps."""
    d = np.abs(np.diff(np.asarray(col, dtype=np.float64)))
    d = d[np.isfinite(d)]
    assert d.size, "a band with no two adjacent finite samples says nothing"
    return float(np.max(d))


# =========================================================================== #
# Level 1 -- extended sheet
# =========================================================================== #
def test_pole_is_the_gaussian_2d_plasmon_condition():
    """At ``sigma = i eps_bar omega / (2 pi |Q|)`` the denominator vanishes exactly.

    This is the Gaussian 2D plasmon condition
    ``eps_1 + eps_2 + 4 pi i sigma Q / omega = 0``; the conductivity is
    constructed analytically so that nothing but the kernel is under test.
    """
    for Q in (0.013, 0.1, 0.77):
        for eps_a, eps_b in ((1.0, 1.0), (1.0, 4.0), (2.5, 2.5)):
            eps_bar = 0.5 * (eps_a + eps_b)
            for E in (0.21, 0.6, 1.3):
                sigma = 1j * eps_bar * omega_of(E) / (2.0 * np.pi * abs(Q))
                chi = dispersion.sheet_response(sigma, Q, E)
                assert abs(eps_a + eps_b + chi) < 1e-15 * (eps_a + eps_b)
                r = dispersion.sheet_reflection_sigma(sigma, Q, E, eps_a, eps_b)
                # the denominator is zero only to machine precision -- whether it
                # underflows to EXACTLY zero depends on the rounding of pi, so
                # the assertion is on the divergence, not on inf itself
                assert abs(r) > 1e12


def test_sheet_reflection_sigma_diverges_without_raising():
    """An exactly vanishing denominator gives a complex infinity, not an exception."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dispersion.sheet_reflection_sigma(0.0, 0.1, 0.5, 1.0, -1.0)
    assert abs(r) == np.inf


def test_pole_condition_is_eps1_plus_eps2_form():
    """``sheet_response`` really is ``4 pi i sigma |Q| / omega``, term by term."""
    sigma = 0.3 + 1.7j
    Q, E = -0.21, 0.44
    want = 4.0 * np.pi * 1j * sigma * abs(Q) / (E / HBAR_EV_FS)
    assert dispersion.sheet_response(sigma, Q, E) == pytest.approx(want, rel=1e-15)


def test_sheet_reflection_sigma_is_fresnel_at_zero_conductivity():
    for eps_a, eps_b in ((1.0, 4.0), (2.0, 2.0), (3.9, 1.0)):
        r = dispersion.sheet_reflection_sigma(0.0, 0.1, 0.5, eps_a, eps_b)
        assert r == pytest.approx((eps_a - eps_b) / (eps_a + eps_b), rel=1e-15)


def test_sheet_reflection_sigma_matches_the_matlab_kernel_form_at_eps_one():
    """``r = 1/(1 - i omega/(2 pi sigma |Q|))`` at ``eps_a = eps_b = 1``."""
    for Q in (0.05, -0.3):
        for E in (0.3, 0.9):
            sigma = conductivity.graphene_drude(E, 1.0, 0.1)
            want = 1.0 / (1.0 - 1j * omega_of(E) / (2.0 * np.pi * sigma * abs(Q)))
            got = dispersion.sheet_reflection_sigma(sigma, Q, E)
            assert got == pytest.approx(want, rel=1e-14)


def test_sheet_plasmon_energy_is_the_pole_of_the_local_drude_sheet():
    """The closed form sits on the pole to 3e-15 absolute in the denominator."""
    worst = 0.0
    for Q in (0.01, 0.1, 0.5):
        for EF in (0.2, 0.5, 1.0):
            for eps_a, eps_b in ((1.0, 1.0), (1.0, 4.0), (2.0, 2.0)):
                E = float(
                    dispersion.sheet_plasmon_energy(Q, EF, eps_a=eps_a, eps_b=eps_b)
                )
                sigma = conductivity.graphene_drude(E, EF, 0.0)
                den = eps_a + eps_b + dispersion.sheet_response(sigma, Q, E)
                worst = max(worst, abs(den))
    assert worst <= 3e-15, f"worst |denominator| = {worst:.3e}"


def test_sheet_plasmon_energy_prefactor_is_four_not_two():
    """Independent derivation: solve the pole numerically, compare to the form.

    The textbook sqrt(Q) 2D plasmon in this library's units is
    ``(hbar omega)^2 = 4 e^2 E_F |Q| / (eps_a + eps_b)``.  Here the pole is
    located by a root find on the library's own denominator with the lossless
    Drude conductivity, so the prefactor 4 is measured, not assumed.
    """
    for Q in (0.02, 0.2):
        for EF in (0.3, 0.9):
            for eps_a, eps_b in ((1.0, 1.0), (1.0, 4.0)):

                def den(E):
                    sig = conductivity.graphene_drude(E, EF, 0.0)
                    return float(
                        np.real(eps_a + eps_b + dispersion.sheet_response(sig, Q, E))
                    )

                root = brentq(den, 1e-4, 10.0, xtol=1e-14, rtol=8.9e-16)
                want = np.sqrt(4.0 * E2_EV_NM * EF * Q / (eps_a + eps_b))
                assert root == pytest.approx(want, rel=1e-12)
                assert float(
                    dispersion.sheet_plasmon_energy(Q, EF, eps_a=eps_a, eps_b=eps_b)
                ) == pytest.approx(root, rel=1e-12)
    # sqrt(Q) scaling
    Qs = np.array([0.01, 0.04, 0.09, 0.16])
    Es = dispersion.sheet_plasmon_energy(Qs, 0.5)
    assert np.allclose(Es / np.sqrt(Qs), Es[0] / np.sqrt(Qs[0]), rtol=1e-14)


def test_longitudinal_is_the_second_drift_component():
    """``sigma/sigma_w - 1 -> alpha^2/4`` (perp) and ``3 alpha^2/4`` (par).

    The textbook semiclassical 2D Dirac transverse/longitudinal split, and the
    reason Level 1 uses the SECOND component.
    """
    EF, E = 1.0, 1.0
    sigma_w = conductivity.graphene_drude(E, EF, 0.0)
    coef_perp, coef_par = [], []
    for Q in (1e-3, 5e-4, 2.5e-4):
        alpha = VF_NM_FS * Q * HBAR_EV_FS / E
        s_perp, s_par = conductivity.graphene_drift_sigma(E, EF, 0.0, Q, 0.0)
        coef_perp.append((s_perp / sigma_w - 1.0) / alpha**2)
        coef_par.append((s_par / sigma_w - 1.0) / alpha**2)
    assert np.real(coef_perp[-1]) == pytest.approx(0.25, rel=1e-6)
    assert np.real(coef_par[-1]) == pytest.approx(0.75, rel=1e-6)
    # and the module hands Level 1 the par component, unmodified
    got = dispersion.sheet_sigma_longitudinal(0.1, 0.5, 1.0, 0.02, 0.3)
    want = conductivity.graphene_drift_sigma(0.5, 1.0, 0.02, 0.1, 0.3)[1]
    assert got == want


def test_matlab_component_slip_shifts_the_pole_by_the_quoted_amounts():
    """The .m file's ``val(1)`` choice moves the ``Im r`` peak by 0.36 % / 1.10 %."""
    shifts = {}
    for Q in (0.1, 0.3):
        w_par = _imr_peak(lambda w: _module_imr(w, Q, 1), 0.2, 1.5)
        w_perp = _imr_peak(lambda w: _module_imr(w, Q, 0), 0.2, 1.5)
        shifts[Q] = 100.0 * (w_par - w_perp) / w_perp
    assert shifts[0.1] == pytest.approx(0.364, abs=0.01)
    assert shifts[0.3] == pytest.approx(1.099, abs=0.01)


def test_matlab_unit_mix_is_exactly_sqrt_e2_in_the_local_drude_limit():
    """The .m kernel's pole is ``sqrt(e^2)`` too low -- exactly, in the local limit.

    ``1 - 1i*W/(2*pi*sigma_au*abs(Q_nm))`` resolves to ``omega^2 = 2 E_F |Q|``
    with the lossless Drude conductivity, against the dimensionally consistent
    ``(hbar omega)^2 = 2 e^2 E_F |Q|``.
    """
    for Q in (0.05, 0.3):
        for EF in (0.5, 1.0):
            matlab_pole = np.sqrt(2.0 * EF * Q)  # mixed units, as the .m file has it
            ours = float(dispersion.sheet_plasmon_energy(Q, EF))
            assert ours / matlab_pole == pytest.approx(np.sqrt(E2_EV_NM), rel=1e-14)


def test_matlab_geometry_regression_with_the_corrected_component():
    """Reproduce the .m script's geometry (EF = 1, gamma = 0.1, beta = 0).

    Pinned with the LONGITUDINAL component, which the .m file gets wrong, and in
    the library's eV-nm-fs units, in which its kernel is dimensionally
    consistent.  The .m file's own numbers are measured alongside so the two
    slips stay visible and separated.
    """
    assert _imr_peak(lambda w: _module_imr(w, 0.1, 1), 0.2, 1.5) == pytest.approx(
        0.537248, abs=1e-4
    )
    assert _imr_peak(lambda w: _module_imr(w, 0.3, 1), 0.2, 1.5) == pytest.approx(
        0.943643, abs=1e-4
    )
    assert _imr_peak(lambda w: _matlab_imr(w, 0.1, 1), 0.1, 1.5) == pytest.approx(
        0.447882, abs=1e-4
    )
    assert _imr_peak(lambda w: _matlab_imr(w, 0.3, 1), 0.1, 1.5) == pytest.approx(
        0.791449, abs=1e-4
    )


def test_sheet_reflection_is_vectorised_over_the_whole_grid():
    Q = dispersion.symmetric_q_grid(0.3, 5)
    E = np.linspace(0.2, 1.1, 7)
    grid = dispersion.sheet_reflection(Q[:, None], E[None, :], 0.6, 0.03, 0.25)
    assert grid.shape == (Q.size, E.size)
    for i, q in enumerate(Q):
        for j, e in enumerate(E):
            one = dispersion.sheet_reflection(float(q), float(e), 0.6, 0.03, 0.25)
            assert grid[i, j] == one


def test_sheet_sigma_local_option_is_q_independent_and_broadcasts():
    Q = np.array([0.05, -0.2, 0.9])
    E = np.array([0.4])
    sig = dispersion.sheet_sigma_longitudinal(Q[:, None], E[None, :], 0.5, 0.02, local=True)
    assert sig.shape == (3, 1)
    assert np.all(sig == sig[0, 0])
    assert sig[0, 0] == conductivity.graphene_drude(0.4, 0.5, 0.02)
    # scalar in, scalar out
    assert np.ndim(dispersion.sheet_sigma_longitudinal(0.1, 0.4, 0.5, 0.02, local=True)) == 0


def test_sheet_reflection_passes_temperature_and_mu_override_through():
    """``T_K`` and ``mu_ev`` reach ``graphene_drift_sigma`` unchanged."""
    Q, E, EF, gam, beta = 0.1, 0.5, 0.5, 0.02, 0.2
    cold = dispersion.sheet_reflection(Q, E, EF, gam, beta)
    warm = dispersion.sheet_reflection(Q, E, EF, gam, beta, T_K=300.0)
    assert cold != warm
    mu_num = conductivity.drift_chemical_potential_numeric(EF, beta, 300.0)
    over = dispersion.sheet_reflection(Q, E, EF, gam, beta, T_K=300.0, mu_ev=mu_num)
    want_sigma = conductivity.graphene_drift_sigma(
        E, EF, gam, Q, beta, T_K=300.0, mu_ev=mu_num
    )[1]
    assert over == dispersion.sheet_reflection_sigma(want_sigma, Q, E)
    # local=True ignores both, as documented
    loc = dispersion.sheet_reflection(Q, E, EF, gam, beta, T_K=300.0, local=True)
    assert loc == dispersion.sheet_reflection(Q, E, EF, gam, 0.0, local=True)


def test_sheet_reflection_branches_split_only_with_drift():
    Q = dispersion.symmetric_q_grid(0.4, 8)
    E = np.linspace(0.2, 1.4, 61)
    r0 = np.imag(dispersion.sheet_reflection(Q[:, None], E[None, :], 1.0, 0.05, 0.0))
    r3 = np.imag(dispersion.sheet_reflection(Q[:, None], E[None, :], 1.0, 0.05, 0.3))
    assert np.array_equal(r0, r0[::-1])  # even in Q, bit for bit
    assert not np.allclose(r3, r3[::-1])
    # the drifted peak moves in opposite directions on the two branches
    up = E[np.argmax(r3[Q > 0][-1])]
    down = E[np.argmax(r3[Q < 0][0])]
    assert up != down


# =========================================================================== #
# Level 1 <-> Level 2 bridge
# =========================================================================== #
def test_extended_sheet_eta_is_algebraically_the_level1_pole():
    """``eta = -1/(2 pi |Q| W)`` <=> ``sigma = i eps_bar omega/(2 pi |Q|)``."""
    for Q in (0.007, 0.1, 1.3):
        for W in (1.0, 10.0, 137.0):
            for eps_bar in (1.0, 2.75):
                for E in (0.15, 0.8):
                    eta = float(dispersion.extended_sheet_eta(Q, W))
                    # the sigma that puts Level 2 on resonance at this eta
                    sigma = eta * omega_of(E) * W * eps_bar / 1j
                    den = 2.0 * eps_bar + dispersion.sheet_response(sigma, Q, E)
                    assert abs(den) < 1e-14 * eps_bar


def test_extended_sheet_eta_sign_and_scaling():
    assert float(dispersion.extended_sheet_eta(0.1, 10.0)) == pytest.approx(
        -1.0 / (2.0 * np.pi), rel=1e-15
    )
    assert dispersion.extended_sheet_eta(-0.1, 10.0) == dispersion.extended_sheet_eta(
        0.1, 10.0
    )
    assert float(dispersion.extended_sheet_eta(0.0, 10.0)) == -np.inf


@pytest.mark.parametrize("L", [201])
def test_wide_ribbon_eta_trends_to_the_sheet_value(L):
    """CROSS-LEVEL 1: the ribbon's uniform-potential mode approaches the sheet.

    A TREND test, not a tolerance test: the edge charge pile-up makes the
    approach go as ``1 + ~0.6/sqrt(qW)``, so 1 % would need ``qW ~ 3600``, far
    past ``operators.STRUVE_TRUSTED_MAX_ARG``.  What is asserted is that the
    ratio is above 1, falls monotonically, and has roughly halved its excess by
    ``qW = 8``.  See the module docstring of ``dispersion.py``.
    """
    ratios = []
    for qW in (1.0, 2.0, 4.0, 8.0):
        q = qW / W_REF
        sysm = _ribbon_system(L=L)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eta, modes, _zeta = sysm.eigenmodes(q)
        # the near-uniform-potential mode: largest |mean phi| / |phi|_max
        v = np.real(modes)
        score = np.abs(v.mean(axis=0)) / np.max(np.abs(v), axis=0)
        k = int(np.argmax(score))
        ratios.append(float(np.real(eta[k]) / dispersion.extended_sheet_eta(q, W_REF)))
    assert all(r > 1.0 for r in ratios), ratios
    assert all(b < a for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[0] == pytest.approx(1.968, abs=0.02)
    assert ratios[-1] == pytest.approx(1.221, abs=0.02)
    assert (ratios[-1] - 1.0) < 0.5 * (ratios[0] - 1.0)


# =========================================================================== #
# Level 2 -- ribbon eigenvalue bands
# =========================================================================== #
def test_drude_band_energy_reference_value():
    got = float(
        dispersion.drude_band_energy(BRIGHT_ETA[0], W_REF, EF_REF, gamma_ev=0.0)
    )
    assert got == pytest.approx(E1_REF, abs=1e-3)
    assert got == pytest.approx(0.5775738, abs=1e-6)


def test_drude_band_energy_inverts_the_eta_resonance_condition():
    """Closed form == numerical root of ``Re eta(E) = eta_j``, including gamma."""
    sysm = _ribbon_system(L=31)
    for eta_j in BRIGHT_ETA + (DARK_ETA,):
        for gamma in (0.0, 0.02, 0.1):
            rib = solver.Ribbon(W_nm=W_REF, EF_ev=EF_REF, gamma_ev=gamma, L=31)
            s = solver.RibbonSystem(rib)
            closed = float(
                dispersion.drude_band_energy(eta_j, W_REF, EF_REF, gamma_ev=gamma)
            )

            def g(E):
                return float(np.real(s.eta_pair(0, 0.01, E)[0])) - eta_j

            root = brentq(g, 1e-4, 10.0, xtol=1e-14, rtol=8.9e-16)
            assert closed == pytest.approx(root, rel=1e-10)
    assert sysm is not None


def test_drude_band_energy_is_nan_for_non_negative_eta():
    for bad in (0.0, -0.0, 1e-9, 0.5):
        assert np.isnan(float(dispersion.drude_band_energy(bad, W_REF, EF_REF)))
    # gamma above the lossless resonance also has no real root: eta_j = -1 puts
    # the lossless energy at 0.151 eV, so gamma = 0.5 eV overdamps it entirely
    assert float(
        dispersion.drude_band_energy(-1.0, W_REF, EF_REF, gamma_ev=0.0)
    ) == pytest.approx(0.1513861, abs=1e-6)
    assert np.isnan(
        float(dispersion.drude_band_energy(-1.0, W_REF, EF_REF, gamma_ev=0.5))
    )


def test_drude_band_energy_broadcasts():
    out = dispersion.drude_band_energy(np.array(BRIGHT_ETA), W_REF, EF_REF)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0.0)  # less negative eta -> higher energy


@pytest.mark.parametrize("L", [601])
def test_ribbon_bands_reproduces_the_acceptance_eigenvalues(L):
    """The three bright eta and the interleaved dark one, at the q -> 0 limit."""
    sysm = _ribbon_system(L=L)
    bs = dispersion.ribbon_bands(sysm, [1e-6], n_bands=6)
    eta = np.real(bs.eta[0])
    for want in BRIGHT_ETA:
        assert np.min(np.abs(eta - want)) < 4e-3 * abs(want), (want, eta)
    assert np.min(np.abs(eta - DARK_ETA)) < 4e-3 * abs(DARK_ETA)
    # the dark mode is present AND carries negligible weight
    k_dark = int(np.argmin(np.abs(eta - DARK_ETA)))
    assert not bool(bs.bright[0, k_dark])
    assert bs.weight[0, k_dark] < 1e-6
    k_bright = int(np.argmin(np.abs(eta - BRIGHT_ETA[0])))
    assert bool(bs.bright[0, k_bright])
    assert bs.weight[0, k_bright] == pytest.approx(1.0, abs=1e-12)
    # and its band energy is the reference fundamental
    assert bs.E_ev[0, k_bright] == pytest.approx(E1_REF, abs=4e-3)


def test_ribbon_bands_energy_equals_the_drude_closed_form():
    sysm = _ribbon_system(L=61)
    bs = dispersion.ribbon_bands(sysm, [1e-5, 0.02, 0.06], n_bands=4)
    ok = np.isfinite(bs.E_ev)
    assert ok.any()
    assert np.allclose(bs.E_ev[ok], bs.E_drude_ev[ok], rtol=1e-9)


def test_ribbon_bands_linewidth_matches_the_closed_form():
    """``FWHM = gamma (E^2 + gamma^2) / E^2`` for local Drude graphene.

    From ``eta(E) = -A (E - i gamma)/(E (E^2 + gamma^2))``:
    ``Re eta = -A/(E^2+gamma^2)``, ``Im eta = A gamma/(E (E^2+gamma^2))``,
    ``d Re eta/dE = 2 A E/(E^2+gamma^2)^2``.
    """
    for gamma in (0.005, 0.02, 0.08):
        sysm = _ribbon_system(L=41, gamma_ev=gamma)
        bs = dispersion.ribbon_bands(sysm, [1e-5], n_bands=3, bright_only=True)
        for E, fw in zip(bs.E_ev[0], bs.fwhm_ev[0]):
            if not np.isfinite(E):
                continue
            want = gamma * (E**2 + gamma**2) / E**2
            assert fw == pytest.approx(want, rel=2e-6), (gamma, E, fw, want)


def test_ribbon_bands_bright_only_gives_a_q_stable_ordering():
    """The charge-non-neutral mode can outrank the fundamental; bright_only drops it.

    Documents the behaviour the ``bright_only`` docstring now describes: with
    ``bright_only=False`` band 0 at ``qW = 0.1`` is the dark, charge-carrying
    near-uniform mode at ~0.037 eV, not the 0.575 eV fundamental.
    """
    sysm = _ribbon_system(L=121)
    qs = [1e-5, 0.01, 0.05]
    loose = dispersion.ribbon_bands(sysm, qs, n_bands=1)
    tight = dispersion.ribbon_bands(sysm, qs, n_bands=1, bright_only=True)
    # with every mode kept, band 0 jumps from 0.575 eV to a far lower energy
    assert loose.E_ev[0, 0] == pytest.approx(0.575, abs=0.01)
    assert loose.E_ev[1, 0] < 0.1
    assert not bool(loose.bright[1, 0])
    # bright_only is stable across q
    assert np.allclose(tight.E_ev[:, 0], 0.575, atol=0.01)
    assert np.all(tight.bright[:, 0])


def test_ribbon_bands_pads_with_nan_and_never_snaps_to_a_grid_point():
    sysm = _ribbon_system(L=31)
    # a window that contains no resonance at all
    bs = dispersion.ribbon_bands(sysm, [1e-5], n_bands=3, E_min=3.0, E_max=4.0)
    assert np.all(np.isnan(bs.E_ev))
    assert np.all(np.isnan(bs.fwhm_ev))
    assert not bs.bright.any()
    assert np.all(bs.mode_index == -1)
    # more bands requested than resolvable in a narrow window
    bs2 = dispersion.ribbon_bands(sysm, [1e-5], n_bands=5, E_min=0.5, E_max=0.65)
    assert np.isfinite(bs2.E_ev[0, 0])
    assert np.isnan(bs2.E_ev[0, 1])
    assert bs2.mode_index[0, 0] >= 0 and bs2.mode_index[0, 1] == -1
    # no returned root is ever clamped to a window edge
    finite = bs2.E_ev[np.isfinite(bs2.E_ev)]
    assert np.all(finite > 0.5) and np.all(finite < 0.65)


def test_ribbon_bands_mode_index_exposes_the_modes_that_were_skipped():
    """A mode with no root in the window is DROPPED, so band index != mode index."""
    sysm = _ribbon_system(L=121)
    full = dispersion.ribbon_bands(sysm, [1e-5], n_bands=4)
    # mode 0 at qW = 1e-4 is the charge-carrying near-uniform one, eta = -4.6e6,
    # whose resonance sits below E_min = 1e-3 eV; it is dropped, so the four
    # returned bands are eigenmodes 1..4 and mode_index says so
    assert full.mode_index[0].tolist() == [1, 2, 3, 4]
    eta0 = sysm.eigenmodes(1e-5)[0]
    assert abs(np.real(eta0[0])) > 1e5
    assert float(dispersion.drude_band_energy(eta0[0].real, W_REF, EF_REF)) < 1e-3
    # restrict the window to the first two bands and the later modes drop out
    narrow = dispersion.ribbon_bands(sysm, [1e-5], n_bands=4, E_min=0.4, E_max=1.0)
    kept = narrow.mode_index[0][narrow.mode_index[0] >= 0]
    assert kept.size >= 1
    assert np.all(np.isfinite(narrow.E_ev[0, : kept.size]))
    assert np.all(np.isnan(narrow.E_ev[0, kept.size :]))
    # bright_only renumbers too: the dark modes are gone from mode_index
    bright = dispersion.ribbon_bands(sysm, [1e-5], n_bands=3, bright_only=True)
    assert bright.mode_index[0].tolist() != [0, 1, 2]
    assert np.all(bright.bright[0])
    eta_all = sysm.eigenmodes(1e-5)[0]
    for band, m in enumerate(bright.mode_index[0]):
        assert bright.eta[0, band] == eta_all[m]


def test_ribbon_bands_rejects_the_cases_where_it_would_be_wrong():
    drift = solver.RibbonSystem(
        solver.Ribbon(W_nm=W_REF, EF_ev=EF_REF, gamma_ev=0.02, beta=0.3, L=31)
    )
    with pytest.raises(ValueError, match="beta"):
        dispersion.ribbon_bands(drift, [0.02])

    mixed = solver.RibbonSystem(
        [
            solver.Ribbon(W_nm=10.0, x0_nm=-8.0, EF_ev=EF_REF, L=31),
            solver.Ribbon(W_nm=6.0, x0_nm=8.0, EF_ev=EF_REF, L=31),
        ]
    )
    with pytest.raises(ValueError, match="W_nm"):
        dispersion.ribbon_bands(mixed, [0.02])

    layered = solver.RibbonSystem(
        solver.Ribbon(W_nm=W_REF, z0_nm=1.0, EF_ev=EF_REF, L=31),
        layers=[substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)],
    )
    with pytest.raises(ValueError, match="screened=True"):
        dispersion.ribbon_bands(layered, [0.05])

    dispersive = solver.RibbonSystem(
        solver.Ribbon(W_nm=W_REF, z0_nm=1.0, EF_ev=EF_REF, L=31),
        layers=[
            substrate.Layer(eps_par=1.0, d_nm=0.0),
            substrate.Layer(eps_par=4.0, sigma_sheet=1.0),
        ],
    )
    with pytest.raises(ValueError, match="dispersive"):
        dispersion.ribbon_bands(dispersive, [0.05], screened=True)

    good = _ribbon_system(L=31)
    with pytest.raises(ValueError, match="n_bands"):
        dispersion.ribbon_bands(good, [0.01], n_bands=0)
    with pytest.raises(ValueError, match="E_min"):
        dispersion.ribbon_bands(good, [0.01], E_min=2.0, E_max=1.0)


def test_ribbon_bands_screened_redshifts_and_is_energy_independent():
    """A purely dielectric stack: ``r`` is frequency independent, so eta is too."""
    layers = [substrate.Layer(eps_par=1.0, d_nm=0.0), substrate.Layer(eps_par=4.0)]
    sub = solver.RibbonSystem(
        solver.Ribbon(W_nm=W_REF, z0_nm=1.0, EF_ev=EF_REF, gamma_ev=0.02, L=61),
        layers=layers,
    )
    free = _ribbon_system(L=61, z0_nm=1.0)
    assert sub._stack_is_dispersive() is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta_a = sub.eigenmodes_screened(0.05, 0.3)[0]
        eta_b = sub.eigenmodes_screened(0.05, 1.7)[0]
        assert np.array_equal(eta_a, eta_b)  # bit for bit
        bs_sub = dispersion.ribbon_bands(sub, [0.05], n_bands=2, bright_only=True,
                                         screened=True)
    bs_free = dispersion.ribbon_bands(free, [0.05], n_bands=2, bright_only=True)
    assert np.all(np.real(bs_sub.eta[0, :2]) < np.real(bs_free.eta[0, :2]))
    assert np.all(bs_sub.E_ev[0, :2] < bs_free.E_ev[0, :2])
    # screened=True with no layers is identical to the default
    plain = dispersion.ribbon_bands(free, [0.05], n_bands=2, bright_only=True,
                                    screened=True)
    assert np.array_equal(plain.eta, bs_free.eta)


def test_ribbon_bands_result_shapes_and_scalar_q():
    sysm = _ribbon_system(L=31)
    bs = dispersion.ribbon_bands(sysm, 0.02, n_bands=3)
    assert bs.qs.shape == (1,)
    for arr in (bs.eta, bs.zeta, bs.weight, bs.bright, bs.E_ev, bs.fwhm_ev,
                bs.E_drude_ev, bs.mode_index):
        assert arr.shape == (1, 3)


# --------------------------------------------------------------------------- #
# track=True -- following one branch across the q grid
#
# The untracked contract is a ONE-Q contract: band k is the k-th mode that
# produced a root, so the kept bands are contiguous from index 0.  Swept over q
# that renumbers its own bands wherever a branch's root leaves [E_min, E_max],
# and every band above it slides down one slot -- a discontinuity in the plot
# that is not in the physics.  These four tests pin the defect, the fix, and
# that the untracked contract did not move.
# --------------------------------------------------------------------------- #
def test_tracking_removes_the_false_step_the_untracked_sweep_draws():
    """THE REPORTED DEFECT, on the case it was reported from.

    One ``W = 10`` nm ribbon, ``L = 121``, over ``_DEFECT_QS`` with the window
    floor at 0.15 eV.  Near ``|q| = 0.05`` / nm the longitudinal branch's root
    falls below the floor, the untracked run drops it, and every later band
    moves down one slot: band 0's largest step between adjacent ``q`` is
    0.366 eV, against the 0.054 eV the same band moves once it is followed --
    and its genuine steps here are ~0.02 eV, so the artefact is the largest
    feature in the curve.

    BOTH sides are measured.  Hard-coding only the tracked number would pass
    just as well if the untracked route had quietly been fixed to agree, and
    the point of the test is the gap between the two.
    """
    loose = _defect_bands(track=False)
    tight = _defect_bands(track=True)
    for band in range(3):
        step_u = _max_step(loose.E_ev[:, band])
        step_t = _max_step(tight.E_ev[:, band])
        assert step_u > 0.2, (band, step_u)
        assert 5.0 * step_t < step_u, (band, step_u, step_t)
    # the two sizes the defect was reported at, loosely
    assert _max_step(loose.E_ev[:, 0]) == pytest.approx(0.366, abs=0.01)
    assert _max_step(tight.E_ev[:, 0]) == pytest.approx(0.054, abs=0.01)


def test_tracking_keeps_each_band_on_one_eigenmode_across_the_sweep():
    """``track=True`` follows an eigenVECTOR; ``track=False`` follows a rank.

    For this system the three leading eigenmodes never exchange rank in
    ``|eta|``, so a followed band's ``mode_index`` is constant down its whole
    column.  (``ribbon_bands``' docstring is careful that this is a property of
    THIS sweep, not a guarantee -- the index is free to change where two
    branches genuinely cross.)
    """
    loose = _defect_bands(track=False)
    tight = _defect_bands(track=True)
    assert tight.mode_index[0].tolist() == [0, 1, 2]
    for band in range(3):
        col = tight.mode_index[:, band]
        assert np.all(col == col[0]), (band, col.tolist())
    # the untracked run does not: at the small-|q| end its slots renumber
    assert not all(
        bool(np.all(loose.mode_index[:, b] == loose.mode_index[0, b]))
        for b in range(3)
    )
    assert loose.mode_index[0].tolist() != loose.mode_index[
        int(np.argmin(np.abs(_DEFECT_QS)))
    ].tolist()
    # and the slot really did keep its branch, not just its number: band 0 is
    # the dark longitudinal one at every q, band 1 the bright fundamental
    assert not tight.bright[:, 0].any()
    assert np.max(np.abs(tight.zeta[:, 0])) < 1e-6
    assert tight.bright[:, 1].all()


def test_tracking_leaves_the_gap_instead_of_sliding_the_bands_down():
    """The gap IS the physics, and the bands around it are the same roots.

    At the smallest ``|q|`` of the sweep the longitudinal branch has no
    resonance inside the window, so tracked band 0 reports ``NaN`` THERE and
    keeps its slot.  Bands 1 and 2 then hold exactly -- bit for bit, same
    bracket, same root find -- the two energies the untracked run reports in
    ITS bands 0 and 1.  Same physics, one slot higher.
    """
    loose = _defect_bands(track=False)
    tight = _defect_bands(track=True)
    iq = int(np.argmin(np.abs(_DEFECT_QS)))
    assert np.isnan(tight.E_ev[iq, 0])
    assert np.all(np.isfinite(tight.E_ev[iq, 1:]))
    assert np.all(np.isfinite(loose.E_ev[iq, :]))
    assert tight.E_ev[iq, 1] == loose.E_ev[iq, 0]
    assert tight.E_ev[iq, 2] == loose.E_ev[iq, 1]
    assert tight.mode_index[iq, 1] == loose.mode_index[iq, 0]
    assert tight.mode_index[iq, 2] == loose.mode_index[iq, 1]
    # only the ENERGY is missing: the slot still reports its eigenvalue, and the
    # closed form says where the root went (0.073 eV, below the 0.15 eV floor)
    assert tight.mode_index[iq, 0] == 0
    assert np.isfinite(np.real(tight.eta[iq, 0]))
    assert np.isnan(tight.fwhm_ev[iq, 0])
    assert tight.E_drude_ev[iq, 0] < 0.15
    # the hole is confined to the small-|q| end, and only to band 0
    gap = ~np.isfinite(tight.E_ev[:, 0])
    assert gap.any() and not gap.all()
    assert np.max(np.abs(_DEFECT_QS)[gap]) < np.min(np.abs(_DEFECT_QS)[~gap])
    assert np.all(np.isfinite(tight.E_ev[:, 1:]))


def test_the_untracked_contract_is_unchanged_by_the_tracking_option():
    """REGRESSION GUARD.  ``track`` defaults to ``False`` and changes nothing there.

    The same renumbering
    ``test_ribbon_bands_mode_index_exposes_the_modes_that_were_skipped`` reads
    at one ``q``, over three: at ``qW = 1e-4`` the near-uniform mode's
    resonance sits below ``E_min`` and is dropped, so the four bands are
    eigenmodes 1..4; by ``qW = 0.2`` it is inside the window and takes band 0.
    """
    sysm = _ribbon_system(L=121)
    qs = [1e-5, 0.02, 0.06]
    default = dispersion.ribbon_bands(sysm, qs, n_bands=4)
    assert default.mode_index.tolist() == [[1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 3]]
    explicit = dispersion.ribbon_bands(sysm, qs, n_bands=4, track=False)
    assert np.array_equal(explicit.mode_index, default.mode_index)
    assert np.array_equal(explicit.E_ev, default.E_ev, equal_nan=True)
    assert np.array_equal(explicit.eta, default.eta, equal_nan=True)


def test_tracking_survives_the_undefined_eigenvector_at_q_zero():
    """A grid containing q = 0 must not hand band 0 to whatever LAPACK returns.

    At exactly q = 0 the near-uniform mode's eigenvalue is 2.4e-16 of the
    largest and machine epsilon is 2.2e-16, so its eigenvector is numerically
    UNDEFINED: any vector in that direction is a valid answer. Desktop
    OpenBLAS returns the uniform mode, the browser's BLAS returned something
    else, and while the tracking chain ran straight through q = 0 that was
    enough to give band 0 a high-order mode for the whole positive half -- a
    bug that reproduced nowhere in CPython. The sweep now walks |q| ascending
    with zero done last, so nothing is ever matched against that vector.

    Forced here by substituting a high-order mode for it, which is the worst
    thing LAPACK could plausibly hand back.
    """
    qs = dispersion.symmetric_q_grid(0.5, 8, include_zero=True)
    kw = dict(n_bands=3, track=True, E_min=1e-3, E_max=1.2)

    honest = dispersion.ribbon_bands(_ribbon_system(L=121), qs, **kw)

    hostile_sys = _ribbon_system(L=121)
    real = hostile_sys.eigenmodes

    def scrambled(q):
        eta, modes, zeta = real(q)
        if q == 0.0:
            modes = modes.copy()
            modes[:, 0] = modes[:, 6]
        return eta, modes, zeta

    hostile_sys.eigenmodes = scrambled
    got = dispersion.ribbon_bands(hostile_sys, qs, **kw)

    # EVERY energy is unchanged, q = 0 included -- not merely close. The two
    # runs differ only in a vector nothing looks at any more.
    np.testing.assert_array_equal(
        np.nan_to_num(got.E_ev, nan=-1.0), np.nan_to_num(honest.E_ev, nan=-1.0)
    )

    # The labelling is identical too, q = 0 included.
    assert np.array_equal(got.mode_index, honest.mode_index)

    z = int(np.flatnonzero(qs == 0.0)[0])
    off = [i for i in range(len(qs)) if i != z]
    assert set(got.mode_index[off, 0].tolist()) == {0}

    # AT zero that band sits at EXACTLY zero energy, in both runs. D_geom is
    # the plain second difference there, its null vector is the uniform
    # potential, so A has an exactly zero eigenvalue, eta is infinite and
    # Re eta(E) can only meet it at E = 0. The endpoint follows from the
    # eigenVALUE alone, which is why the substitution cannot move it -- and
    # that is the whole point: the one quantity here that is undefined is the
    # one the answer does not use.
    assert got.E_ev[z, 0] == 0.0
    assert honest.E_ev[z, 0] == 0.0
    # No width: a zero-energy mode has no resonance to be wide.
    assert not np.isfinite(got.fwhm_ev[z, 0])

    # It must NOT instead be given a resolved high-order mode. That is what
    # happened when an unmatched slot was filled from the spare pool, and
    # because a resolved mode has a real energy the curve shot from the bottom
    # of the frame to 1.18 eV and back in a single point. Nothing else in the
    # band may carry an energy above the branch's own range.
    assert np.nanmax(got.E_ev[:, 0]) == np.nanmax(honest.E_ev[:, 0])
    assert np.nanmax(got.E_ev[:, 0]) < np.nanmin(got.E_ev[:, 2])

    # The transverse bands are untouched there -- they match their own
    # eigenvectors at 0.99, so nothing about q = 0 troubles them.
    assert np.isfinite(got.E_ev[z, 1]) and np.isfinite(got.E_ev[z, 2])
    assert got.E_ev[z, 1] == honest.E_ev[z, 1]

    # And +-q agree exactly, which at beta = 0 they must: the operator depends
    # on q only through |q|, so the two halves are now one computation.
    E = got.E_ev
    np.testing.assert_array_equal(
        np.nan_to_num(E, nan=-1.0), np.nan_to_num(E[::-1], nan=-1.0)
    )


@pytest.mark.parametrize("nrib", [1, 2, 3, 4])
def test_every_ribbon_gets_its_own_q_zero_endpoint(nrib):
    """N ribbons have N null modes, N longitudinal branches and N endpoints.

    ``D_geom`` is BLOCK diagonal -- one block per ribbon -- so each ribbon
    brings its own uniform-potential null vector. Rescuing only the largest one
    left the rest to the ``eta < 0`` test, which at machine zero is a coin
    flip: five identical ribbons gave
    ``+2.10e12 +1.69e12 -1.40e12 -1.14e12 +7.19e11``, three of which fail it.
    The q = 0 row of a four-ribbon run read "null null null 0.000".

    ``_band_obstruction`` requires every ribbon to share one
    ``(W, EF, gamma)``, so an array of IDENTICAL ribbons is the only
    multi-ribbon case this function ever sees -- which makes this the ordinary
    case, not a corner.
    """
    pitch = 200.0
    rib = [
        solver.Ribbon(
            W_nm=W_REF,
            x0_nm=float(x),
            z0_nm=0.0,
            L=41,
            EF_ev=EF_REF,
            gamma_ev=0.02,
            beta=0.0,
        )
        for x in (np.arange(nrib) - (nrib - 1) / 2.0) * pitch
    ]
    sysm = solver.RibbonSystem(rib)

    eta0 = sysm.eigenmodes(0.0)[0]
    lam = np.abs(1.0 / np.real(eta0))
    n_null = int(np.sum(lam <= 1e-12 * np.max(lam)))
    assert n_null == nrib, (nrib, n_null)

    qs = dispersion.symmetric_q_grid(0.2, 4, include_zero=True)
    bs = dispersion.ribbon_bands(sysm, qs, n_bands=nrib, track=True,
                                 E_min=1e-4, E_max=1.2)
    z = int(np.flatnonzero(qs == 0.0)[0])

    # every band ends at exactly zero there, one per ribbon
    assert np.all(bs.E_ev[z] == 0.0), (nrib, bs.E_ev[z])
    # and they are N DISTINCT modes, not one mode claimed N times
    assert len(set(bs.mode_index[z].tolist())) == nrib, bs.mode_index[z]


@pytest.mark.parametrize("L", [61, 121, 301])
def test_the_q_zero_endpoint_survives_either_sign_of_the_null_eigenvalue(L):
    """The sign of a machine-zero eigenvalue is noise, and it really does flip.

    ``eta = 1/lambda`` with ``lambda`` at machine zero, so its sign is whatever
    the rounding left behind. On this machine, same physics, same code:
    ``L = 61`` gives ``eta = +4.0e13``, ``L = 121`` gives ``-2.2e12``,
    ``L = 301`` gives ``+3.0e13``.

    That matters because the candidate filter requires ``eta < 0`` -- right for
    every real branch, since a positive ``eta`` has no resonance -- so a
    positive null mode was discarded before anything looked at it, and the
    "largest |eta|" left over was then the FUNDAMENTAL. It got reported at zero
    energy while its own band went empty. Testing only at ``L = 121`` hid this
    completely; it is parametrised over L for exactly that reason.
    """
    sysm = _ribbon_system(L=L)
    qs = dispersion.symmetric_q_grid(0.5, 4, include_zero=True)
    bs = dispersion.ribbon_bands(sysm, qs, n_bands=3, track=True, E_min=1e-3, E_max=1.2)
    z = int(np.flatnonzero(qs == 0.0)[0])

    # whichever sign came out, the endpoint is band 0's and it is exactly zero
    assert bs.E_ev[z, 0] == 0.0, (L, bs.E_ev[z])
    assert bs.mode_index[z].tolist() == [0, 1, 2], (L, bs.mode_index[z])
    # and the fundamental is NOT the one sitting at zero
    assert bs.E_ev[z, 1] == pytest.approx(E1_REF, abs=6e-3), (L, bs.E_ev[z])
    assert np.all(np.isfinite(bs.E_ev[:, 1]))


@pytest.mark.parametrize("lam_rel", [None, 1e-11, 1e-9, 1e-7])
def test_the_q_zero_endpoint_does_not_depend_on_how_zero_the_solver_calls_zero(lam_rel):
    """``E = 0`` at ``q = 0`` whatever magnitude LAPACK gives the null eigenvalue.

    The endpoint is a fact about the operator: at ``q = 0``, ``D_geom`` is the
    plain second difference under its zero-flux closure, its null vector is the
    uniform potential, so ``A`` has an exactly zero eigenvalue and the branch
    ends at ``E = 0``. What no implementation guarantees is the MAGNITUDE it
    returns for that eigenvalue. This machine gives 2.4e-16 of the largest;
    1e-11 is just as numerically zero and another BLAS may well return it.

    Classifying the mode by a tolerance (``|lambda|/max < 1e-12``) therefore
    worked here and failed in the browser, where the endpoint was never
    assigned and band 1 stopped one grid point short of the origin while this
    machine drew it perfectly. It is identified structurally now -- the largest
    ``|eta|``, which the physical branches (0.0075 to 0.07) cannot approach --
    so this parametrisation must hold at every scale, with the eigenvector
    scrambled as well since that one really is undefined.
    """
    sysm = _ribbon_system(L=121)
    real = sysm.eigenmodes

    def returned(q):
        eta, modes, zeta = real(q)
        if q == 0.0 and lam_rel is not None:
            eta, modes = eta.copy(), modes.copy()
            lam_max = float(np.max(np.abs(1.0 / np.real(eta[1:]))))
            eta[0] = -1.0 / (lam_rel * lam_max)
            modes[:, 0] = modes[:, 6]
        return eta, modes, zeta

    sysm.eigenmodes = returned
    qs = dispersion.symmetric_q_grid(1.0, 8, include_zero=True)
    bs = dispersion.ribbon_bands(sysm, qs, n_bands=3, track=True, E_min=1e-3, E_max=1.2)

    z = int(np.flatnonzero(qs == 0.0)[0])
    assert bs.E_ev[z, 0] == 0.0, bs.E_ev[z]
    # and the transverse bands are untouched by any of it
    assert bs.E_ev[z, 1] == pytest.approx(0.5755, abs=5e-3)
    assert bs.E_ev[z, 2] == pytest.approx(0.8879, abs=5e-3)
    # the curve is CONTINUOUS through the origin: no gap on either side
    assert np.all(np.isfinite(bs.E_ev[:, 0]))


def test_follow_modes_never_gives_one_eigenvector_to_two_bands():
    """Greedy over the WHOLE overlap matrix: one eigenvector per slot, or ``-1``.

    ``A = V @ D_geom`` is not symmetric, so its eigenvectors are not orthogonal
    and ``|<a|b>|`` is a similarity, not a projection -- an independent argmax
    per band can hand the same eigenvector to two of them.  The history here is
    rigged to force exactly that: two slots start out holding the SAME
    eigenvector, so whatever the overlaps say, one of them has to be given
    something else.
    """
    sysm = _ribbon_system(L=61)
    _eta, modes, _zeta = sysm.eigenmodes(0.04)
    nb = 5
    every = np.arange(modes.shape[1], dtype=np.int64)
    full = np.ones(nb, dtype=bool)

    dup = modes[:, [2, 2, 5, 7, 9]].copy()
    sel = dispersion._follow_modes(dup, full, modes, every, nb)
    # One of the two slots holding eigenvector 2 gets it; the other is left
    # EMPTY rather than handed the nearest spare. Two different eigenvectors
    # overlap by at most 0.4032 here, well under `_MIN_OVERLAP`, so there is
    # genuinely nothing for the loser to be. Filling it anyway is what let a
    # band take a high-order mode at q = 0 and draw its energy as a spike.
    assert sel.tolist() == [-1, 2, 5, 7, 9], sel.tolist()
    assigned = sel[sel >= 0].tolist()
    assert len(set(assigned)) == len(assigned), sel.tolist()

    # a permuted history is followed exactly: every slot gets ITS eigenvector
    order = [3, 1, 0, 2, 4]
    perm = modes[:, order].copy()
    assert dispersion._follow_modes(perm, full, modes, every, nb).tolist() == order

    # fewer candidates than bands: the slots that cannot be filled report -1,
    # never a repeat of one that was
    three = np.array([0, 1, 2], dtype=np.int64)
    short = dispersion._follow_modes(modes[:, :nb].copy(), full, modes, three, nb)
    assert short.tolist() == [0, 1, 2, -1, -1]
    assert len(set(short[short >= 0].tolist())) == int(np.count_nonzero(short >= 0))
    # no candidates at all
    assert dispersion._follow_modes(
        modes[:, :nb].copy(), full, modes, np.empty(0, dtype=np.int64), nb
    ).tolist() == [-1] * nb
    # no history at all: the descending-|eta| order eigenmodes returns
    fresh = dispersion._follow_modes(None, np.zeros(nb, dtype=bool), modes, three, nb)
    assert fresh.tolist() == [0, 1, 2, -1, -1]
    # a non-finite history column cannot poison the assignment either: that one
    # slot reports -1 -- a band whose own reference is NaN has no branch to
    # follow -- and every other slot still gets exactly its own eigenvector
    nanned = modes[:, :nb].copy()
    nanned[:, 2] = np.nan
    bad = dispersion._follow_modes(nanned, full, modes, every, nb)
    assert bad.tolist() == [0, 1, -1, 3, 4], bad.tolist()


# --------------------------------------------------------------------------- #
# The physics claim in the docstrings of ribbon_bands (bright_only / track) and
# of solver.RibbonSystem.eigenmodes: the near-uniform |eta| ~ 1/(qW)^2 mode is a
# mode of the OPERATOR, not of the mesh.  Three independent legs, loose
# tolerances -- what is pinned is the claim, not the last digit.
# --------------------------------------------------------------------------- #
def test_the_near_uniform_mode_converges_in_L():
    """LEG 1: refine the mesh and its eigenvalue settles.  A mesh mode would not.

    Quoted at ``W = 10`` nm, ``q = 0.02``: ``-3.9558, -3.9817, -3.9975,
    -4.0027`` at ``L = 61, 121, 301, 601``.  Asserted as a trend -- the
    successive differences shrink and the two finest meshes agree to better
    than 0.2 % -- plus the quoted values to 5e-3 absolute.
    """
    etas = []
    for L in (61, 121, 301, 601):
        eta, _modes, zeta = _ribbon_system(L=L).eigenmodes(0.02)
        etas.append(float(np.real(eta[0])))
        assert abs(zeta[0]) < 1e-6, (L, zeta[0])  # dark at every L, so never fitted
    assert etas == pytest.approx([-3.9558, -3.9817, -3.9975, -4.0027], abs=5e-3)
    diffs = np.abs(np.diff(etas))
    assert np.all(np.diff(diffs) < 0.0), diffs
    assert abs(etas[-1] - etas[-2]) < 2e-3 * abs(etas[-1]), etas[-2:]


def test_the_near_uniform_mode_is_entirely_charge_carrying():
    """LEG 2: ALL of its charge is net charge; the bright modes carry none.

    With ``rho = D_geom phi / W``, the signed fraction
    ``h sum(rho) / h sum|rho|`` is -1.000 for this mode -- every node of the
    same sign -- against ``< 1e-10`` for the dipole-active modes, which the
    zero-flux ``D_geom`` makes charge neutral.  Two mesh densities, so the -1
    is not a property of one ``L``.
    """
    for L in (121, 301):
        sysm = _ribbon_system(L=L)
        q, h = 0.02, sysm.h[0]
        eta, modes, zeta = sysm.eigenmodes(q)
        Dg = operators.build_D_geom(q, W_REF, h, sysm.L[0])
        bright = solver.bright_mode_mask(zeta)

        def net_fraction(m):
            rho = Dg @ modes[:, m] / W_REF
            return float(np.real(h * np.sum(rho) / (h * np.sum(np.abs(rho)))))

        assert abs(np.real(eta[0])) > 1.0          # it IS the near-uniform mode
        assert net_fraction(0) == pytest.approx(-1.0, abs=1e-9)
        assert max(abs(net_fraction(int(m))) for m in np.flatnonzero(bright[:8])) < 1e-10
        # every other mode is at least three orders of magnitude from mode 0.
        # NOT 1e-10: the higher DARK modes leave a ~5e-4 residual that does not
        # shrink with L (-5.287e-4 at L = 121, -5.272e-4 at L = 301), so the
        # clean 1e-14 of the docstring is the bright modes' number, not every
        # neutral mode's.
        assert max(abs(net_fraction(m)) for m in range(1, 8)) < 1e-3


def test_the_near_uniform_eta_carries_the_log_of_the_coulomb_kernel():
    """LEG 3: ``1/[eta_0 (qW)^2]`` is linear in ``ln(2/qW)``, slope ~2.

    The logarithm of the coplanar ``2 K_0`` kernel, which is what makes the
    inverted lossless Drude energy run as ``q sqrt(ln(1/qW))`` -- the ribbon's
    one-dimensional wire plasmon.  Fitted over ``qW`` from 0.05 to 1 at
    ``L = 301``; the measured slope is -1.954, the sign being only ``eta < 0``.
    """
    sysm = _ribbon_system(L=301)
    qW = np.array([0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.7, 1.0])
    y = np.empty(qW.size)
    for i, s in enumerate(qW):
        eta0 = float(np.real(sysm.eigenmodes(float(s) / W_REF)[0][0]))
        y[i] = 1.0 / (eta0 * s**2)
    x = np.log(2.0 / qW)
    slope, intercept = np.polyfit(x, y, 1)
    assert abs(slope) == pytest.approx(2.0, rel=0.1)
    # and it really is a straight line over that range, not a fitted curve
    resid = float(np.max(np.abs(y - (slope * x + intercept))))
    assert resid < 0.03 * float(y.max() - y.min()), resid
    assert abs(float(np.corrcoef(x, y)[0, 1])) > 0.999


# =========================================================================== #
# Level 3 -- LDOS map
# =========================================================================== #
def test_field_map_is_exactly_the_linear_operator_of_induced_field_q():
    sysm = _ribbon_system(L=41)
    rng = np.random.default_rng(12345)
    rho = rng.standard_normal(sysm.N) + 1j * rng.standard_normal(sysm.N)
    r0 = (0.3, 0.0, 6.0)
    G = dispersion._field_map(sysm, 0.03, r0)
    assert G.shape == (3, sysm.N)
    direct = np.asarray(observables.induced_field_q(0.03, sysm.x_nodes, [rho], r0))
    assert np.max(np.abs(G @ rho - direct)) < 1e-14 * np.max(np.abs(direct))


def test_ldos_from_field_matches_observables_ldos_q():
    sysm = _ribbon_system(L=41)
    rng = np.random.default_rng(7)
    rho = rng.standard_normal(sysm.N) + 1j * rng.standard_normal(sysm.N)
    r0, p = (0.0, 0.0, 6.0), np.array([1.0, 0.3j, -0.5])
    G = dispersion._field_map(sysm, 0.03, r0)
    want = observables.ldos_q(0.03, sysm.x_nodes, [rho], r0, p)
    got = dispersion._ldos_from_field(G @ rho, p)
    assert got == pytest.approx(want, rel=1e-12)


def test_ldos_map_fast_and_full_routes_agree():
    sysm = _ribbon_system(L=61)
    Es = np.linspace(0.4, 1.3, 91)
    qs = np.array([0.02, 0.07])
    fast = dispersion.ldos_map(sysm, qs, Es, (0.0, 0.0, 6.0))
    full = dispersion.ldos_map(sysm, qs, Es, (0.0, 0.0, 6.0), fast_path=False)
    assert fast.shape == (2, Es.size)
    assert np.max(np.abs(fast - full)) < 1e-9 * np.max(np.abs(full))


def test_ldos_map_absE_is_the_field_norm_and_positive():
    sysm = _ribbon_system(L=41)
    Es = np.linspace(0.45, 0.75, 31)
    amp = dispersion.ldos_map(sysm, [0.03], Es, (0.0, 0.0, 6.0), quantity="absE")
    assert np.all(amp > 0.0)
    # the resonance shows in |E| at the same place as in the LDOS
    ldos = dispersion.ldos_map(sysm, [0.03], Es, (0.0, 0.0, 6.0))
    assert abs(Es[np.argmax(amp[0])] - Es[np.argmax(ldos[0])]) <= 2.0 * (Es[1] - Es[0])
    # and |E| really is the Euclidean norm of the induced field vector
    E_star = float(Es[np.argmax(amp[0])])
    rhs = [
        sources.phi_ext_dipole(
            0.03, sysm.x_nodes[0], 0.0, (0.0, 0.0, 6.0), (1.0, 0.0, 0.0), sysm.eps_a
        )
    ]
    sol = sysm.solve_linear(0.03, E_star, rhs)
    vec = np.asarray(
        observables.induced_field_q(0.03, sysm.x_nodes, sol["rho"], (0.0, 0.0, 6.0))
    )
    assert float(np.sqrt(np.vdot(vec, vec).real)) == pytest.approx(
        float(amp[0, np.argmax(amp[0])]), rel=1e-9
    )


def test_ldos_map_fast_path_flag_is_a_no_op_when_the_system_obstructs_it():
    """With drift there is no eigen route, so the flag cannot change the answer."""
    drift = solver.RibbonSystem(
        solver.Ribbon(W_nm=W_REF, EF_ev=EF_REF, gamma_ev=0.02, beta=0.3, L=41)
    )
    assert drift.fast_path_obstruction() is not None
    Es = np.linspace(0.45, 0.8, 25)
    a = dispersion.ldos_map(drift, [0.05], Es, (0.0, 0.0, 6.0), fast_path=True)
    b = dispersion.ldos_map(drift, [0.05], Es, (0.0, 0.0, 6.0), fast_path=False)
    assert np.array_equal(a, b)


def test_ldos_map_rejects_bad_input():
    sysm = _ribbon_system(L=31)
    Es = np.linspace(0.5, 0.6, 5)
    with pytest.raises(ValueError, match="quantity"):
        dispersion.ldos_map(sysm, [0.02], Es, (0.0, 0.0, 6.0), quantity="nope")
    with pytest.raises(ValueError, match="q = 0"):
        dispersion.ldos_map(sysm, [0.0, 0.02], Es, (0.0, 0.0, 6.0))
    with pytest.raises(ValueError, match="3 components"):
        dispersion.ldos_map(sysm, [0.02], Es, (0.0, 0.0, 6.0), p=(1.0, 0.0))
    with pytest.raises(ValueError, match="non-zero dipole"):
        dispersion.ldos_map(sysm, [0.02], Es, (0.0, 0.0, 6.0), p=(0.0, 0.0, 0.0))


def test_ldos_map_warns_for_out_of_plane_ribbons():
    sysm = _ribbon_system(L=31, z0_nm=2.0)
    with pytest.warns(RuntimeWarning, match="z = 0"):
        dispersion.ldos_map(sysm, [0.02], np.linspace(0.5, 0.6, 3), (0.0, 0.0, 8.0))


def test_ldos_map_bessel_kernel_reuse_does_not_change_the_answer():
    """``len(Es) > N`` switches on the precomputed field map; both must agree."""
    sysm = _ribbon_system(L=31)
    Es_long = np.linspace(0.5, 0.65, sysm.N + 10)
    Es_short = Es_long[: sysm.N - 5]
    long_map = dispersion.ldos_map(sysm, [0.03], Es_long, (0.0, 0.0, 6.0))
    short_map = dispersion.ldos_map(sysm, [0.03], Es_short, (0.0, 0.0, 6.0))
    assert np.max(np.abs(long_map[0, : Es_short.size] - short_map[0])) < 1e-12 * np.max(
        np.abs(short_map)
    )


# --------------------------------------------------------------------------- #
# track_peaks
# --------------------------------------------------------------------------- #
def test_track_peaks_refines_to_sub_grid_accuracy():
    E = np.linspace(0.0, 10.0, 201)
    for centre in (3.0, 3.0123, 6.77):
        y = np.exp(-((E - centre) ** 2) / (2 * 0.4**2))
        pos, wgt = dispersion.track_peaks(E, y[None, :], n_peaks=1)
        assert pos[0, 0] == pytest.approx(centre, abs=2e-3)
        assert wgt[0, 0] == pytest.approx(1.0, rel=5e-3)
        # strictly better than snapping to the nearest node
        node = E[np.argmin(np.abs(E - centre))]
        assert abs(pos[0, 0] - centre) <= abs(node - centre) + 1e-12


def test_track_peaks_returns_nan_rather_than_a_nearest_grid_point():
    E = np.linspace(0.2, 1.2, 51)
    monotone = np.linspace(0.0, 1.0, 51)          # no interior maximum
    flat = np.zeros(51)                           # constant
    nans = np.full(51, np.nan)
    vals = np.vstack([monotone, flat, nans, monotone[::-1]])
    pos, wgt = dispersion.track_peaks(E, vals, n_peaks=2)
    assert np.all(np.isnan(pos))
    assert np.all(np.isnan(wgt))


def test_track_peaks_pads_and_sorts():
    E = np.linspace(0.0, 10.0, 401)
    y = (
        1.0 * np.exp(-((E - 3.0) ** 2) / 0.08)
        + 0.4 * np.exp(-((E - 6.0) ** 2) / 0.08)
    )
    by_w, w_w = dispersion.track_peaks(E, y[None, :], n_peaks=4, sort="weight")
    assert by_w[0, 0] == pytest.approx(3.0, abs=1e-3)
    assert by_w[0, 1] == pytest.approx(6.0, abs=1e-3)
    assert np.all(np.isnan(by_w[0, 2:]))
    assert w_w[0, 0] > w_w[0, 1]
    by_e, _ = dispersion.track_peaks(E, y[None, :], n_peaks=2, sort="energy")
    assert by_e[0, 0] < by_e[0, 1]
    # a tiny ripple is rejected by the prominence floor
    y2 = y + 1e-5 * np.sin(200.0 * E)
    strict, _ = dispersion.track_peaks(E, y2[None, :], n_peaks=4, rel_prominence=1e-2)
    assert np.count_nonzero(np.isfinite(strict[0])) == 2


def test_track_peaks_non_uniform_grid_returns_the_node():
    E = np.array([0.0, 1.0, 1.5, 1.7, 3.0, 5.0])
    y = np.array([0.0, 0.5, 1.0, 0.6, 0.2, 0.0])
    pos, wgt = dispersion.track_peaks(E, y[None, :], n_peaks=1)
    assert pos[0, 0] == 1.5
    assert wgt[0, 0] == 1.0


def test_track_peaks_validation():
    E = np.linspace(0.0, 1.0, 5)
    y = np.zeros((2, 5))
    with pytest.raises(ValueError, match="sort"):
        dispersion.track_peaks(E, y, sort="bogus")
    with pytest.raises(ValueError, match="n_peaks"):
        dispersion.track_peaks(E, y, n_peaks=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        dispersion.track_peaks(E[::-1], y)
    with pytest.raises(ValueError, match="at least 3"):
        dispersion.track_peaks(E[:2], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="last axis"):
        dispersion.track_peaks(E, np.zeros((2, 4)))


def test_track_peaks_preserves_leading_shape():
    E = np.linspace(0.0, 10.0, 201)
    y = np.exp(-((E - 5.0) ** 2) / 0.5)
    stack = np.broadcast_to(y, (2, 3, E.size)).copy()
    pos, wgt = dispersion.track_peaks(E, stack, n_peaks=2)
    assert pos.shape == (2, 3, 2)
    assert wgt.shape == (2, 3, 2)
    assert np.allclose(pos[..., 0], 5.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# follow_peaks
# --------------------------------------------------------------------------- #
def _two_branches(qs, Es, e_lo, e_hi, h_lo, h_hi, width=0.06):
    """A map with two Gaussian ridges whose energies and heights are given."""
    out = np.empty((qs.size, Es.size))
    for i in range(qs.size):
        out[i] = (h_lo[i] * np.exp(-((Es - e_lo[i]) ** 2) / (2 * width ** 2))
                  + h_hi[i] * np.exp(-((Es - e_hi[i]) ** 2) / (2 * width ** 2)))
    return out


@pytest.mark.parametrize("n_q", [11, 32, 61])
def test_follow_peaks_survives_a_height_crossing_where_track_peaks_does_not(n_q):
    """THE bug: two branches swap which one is taller, and nothing else.

    Their energies never approach each other, so there is no ambiguity about
    which is which -- yet a per-column sort by height swaps the slots at the
    crossing and draws a vertical step between two curves 0.5 eV apart.
    """
    qs = np.linspace(-1.0, 1.0, n_q)
    Es = np.linspace(0.1, 1.5, 241)
    e_lo = 0.40 + 0.05 * qs
    e_hi = 0.95 + 0.05 * qs
    h_lo = np.linspace(0.2, 1.0, n_q)          # the lower branch brightens
    h_hi = np.linspace(1.0, 0.2, n_q)          # the upper one fades
    vals = _two_branches(qs, Es, e_lo, e_hi, h_lo, h_hi)

    old, _ = dispersion.track_peaks(Es, vals, n_peaks=2)
    new, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=2)

    # The old one steps across the gap between the branches at the crossing.
    old_step = np.nanmax(np.abs(np.diff(old, axis=0)))
    assert old_step > 0.4, "expected the per-column sort to swap branches"

    # The new one never moves a slot further than the branches themselves move.
    new_step = np.nanmax(np.abs(np.diff(new, axis=0)))
    assert new_step < 0.05, f"slots still jumping: {new_step:.4f} eV"

    # And each slot stays on ONE branch for the whole sweep.
    for k in range(2):
        col = new[:, k]
        live = np.isfinite(col)
        assert live.sum() == n_q
        near_lo = np.abs(col - e_lo) < 0.05
        near_hi = np.abs(col - e_hi) < 0.05
        assert bool(np.all(near_lo)) or bool(np.all(near_hi))


def test_follow_peaks_orders_slots_once_for_the_whole_sweep():
    """sort= decides slot MEANING globally; per-column ranking is the bug."""
    n_q = 25
    qs = np.linspace(-1.0, 1.0, n_q)
    Es = np.linspace(0.1, 1.5, 241)
    e_lo = np.full(n_q, 0.40)
    e_hi = np.full(n_q, 0.95)
    h_lo = np.linspace(0.2, 1.0, n_q)
    h_hi = np.linspace(1.0, 0.2, n_q)
    vals = _two_branches(qs, Es, e_lo, e_hi, h_lo, h_hi)

    by_e, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=2, sort="energy")
    assert np.allclose(by_e[:, 0], 0.40, atol=0.02)
    assert np.allclose(by_e[:, 1], 0.95, atol=0.02)

    # Equal total weight by construction, so "weight" order is stable rather
    # than meaningful here -- what matters is that it is ONE order, not 25.
    by_w, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=2, sort="weight")
    for k in range(2):
        col = by_w[:, k]
        assert np.allclose(col, col[0], atol=0.02)


def test_follow_peaks_breaks_a_branch_that_ends_instead_of_refilling_the_slot():
    """A slot whose branch ends must not be handed a stranger next column.

    This is the 1.05 eV vertical line on the page's beta = 0.26 panel: the third
    branch stopped and the slot was immediately refilled from far away, so the
    stroke carried on as if nothing had happened.
    """
    n_q = 31
    qs = np.linspace(-1.0, 1.0, n_q)
    Es = np.linspace(0.1, 1.5, 241)
    half = n_q // 2
    vals = np.zeros((n_q, Es.size))
    for i in range(n_q):
        vals[i] += 1.0 * np.exp(-((Es - 0.50) ** 2) / (2 * 0.06 ** 2))
        if i < half:                     # a branch that stops at the midpoint
            vals[i] += 0.6 * np.exp(-((Es - 0.80) ** 2) / (2 * 0.06 ** 2))
        if i > half:                     # a different one that starts after it
            vals[i] += 0.6 * np.exp(-((Es - 1.30) ** 2) / (2 * 0.06 ** 2))

    pos, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=3)
    steps = np.abs(np.diff(pos, axis=0))
    worst = float(np.nanmax(steps)) if np.any(np.isfinite(steps)) else 0.0
    assert worst < 0.1, f"a slot stepped {worst:.3f} eV between adjacent columns"

    # The two short branches are present somewhere, and separated by a gap.
    assert np.any(np.abs(pos - 0.80) < 0.05)
    assert np.any(np.abs(pos - 1.30) < 0.05)


def test_follow_peaks_invents_nothing():
    """Every point returned is a peak track_peaks resolved in that same column."""
    n_q = 21
    qs = np.linspace(-0.8, 0.8, n_q)
    Es = np.linspace(0.1, 1.5, 201)
    e_lo = 0.35 + 0.25 * np.abs(qs)
    e_hi = 0.95 + 0.25 * np.abs(qs)
    h_lo = 1.0 - 0.5 * np.abs(qs)
    h_hi = 0.4 + 0.5 * np.abs(qs)
    vals = _two_branches(qs, Es, e_lo, e_hi, h_lo, h_hi)

    pos, wgt = dispersion.follow_peaks(qs, Es, vals, n_peaks=3)
    pool = max(1, (Es.size - 1) // 2)
    cand, _ = dispersion.track_peaks(Es, vals, n_peaks=pool)
    for i in range(n_q):
        allowed = cand[i][np.isfinite(cand[i])]
        for v in pos[i][np.isfinite(pos[i])]:
            assert np.min(np.abs(allowed - v)) == 0.0, (
                f"column {i}: {v} is not one of the peaks found there"
            )
    # Weights travel with their energies, never recombined across columns.
    assert np.all(np.isfinite(wgt) == np.isfinite(pos))


def test_follow_peaks_handles_empty_columns_and_slot_budgets():
    n_q = 15
    qs = np.linspace(-1.0, 1.0, n_q)
    Es = np.linspace(0.1, 1.5, 201)
    vals = np.zeros((n_q, Es.size))
    for i in range(n_q):
        if i in (5, 6):
            continue                        # genuinely featureless columns
        for c, h in ((0.4, 1.0), (0.8, 0.7), (1.2, 0.4)):
            vals[i] += h * np.exp(-((Es - c) ** 2) / (2 * 0.05 ** 2))

    empty, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=3)
    assert np.all(np.isnan(empty[5])) and np.all(np.isnan(empty[6]))

    # Fewer slots than branches: the strongest survive, by TOTAL weight.
    few, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=2)
    assert np.allclose(np.nanmean(few[:, 0]), 0.4, atol=0.02)
    assert np.allclose(np.nanmean(few[:, 1]), 0.8, atol=0.02)

    # More slots than branches: padded, not duplicated.
    many, _ = dispersion.follow_peaks(qs, Es, vals, n_peaks=5)
    assert np.all(np.isnan(many[:, 3:]))
    for i in range(n_q):
        live = many[i][np.isfinite(many[i])]
        assert live.size == len(set(np.round(live, 9)))


def test_follow_peaks_validation():
    qs = np.linspace(-1.0, 1.0, 4)
    Es = np.linspace(0.1, 1.0, 5)
    vals = np.zeros((4, 5))
    with pytest.raises(ValueError, match="sort"):
        dispersion.follow_peaks(qs, Es, vals, sort="bogus")
    with pytest.raises(ValueError, match="n_peaks"):
        dispersion.follow_peaks(qs, Es, vals, n_peaks=0)
    with pytest.raises(ValueError, match=r"len\(qs\)"):
        dispersion.follow_peaks(qs, Es, np.zeros((3, 5)))
    with pytest.raises(ValueError, match=r"len\(qs\)"):
        dispersion.follow_peaks(qs, Es, np.zeros(5))
    with pytest.raises(ValueError, match="strictly increasing"):
        dispersion.follow_peaks(qs, Es[::-1], vals)


# =========================================================================== #
# Non-reciprocity
# =========================================================================== #
def test_symmetric_q_grid_is_bit_exact_where_linspace_is_not():
    bad = np.linspace(-0.3, 0.3, 13)
    assert not np.array_equal(bad, -bad[::-1])  # the trap this helper exists for
    assert np.max(np.abs(bad + bad[::-1])) > 0.0
    for n in (1, 4, 17):
        g = dispersion.symmetric_q_grid(0.3, n)
        assert g.size == 2 * n
        assert np.array_equal(g, -g[::-1])
        assert np.all(np.diff(g) > 0.0)
        g0 = dispersion.symmetric_q_grid(0.3, n, include_zero=True)
        assert g0.size == 2 * n + 1
        assert np.array_equal(g0, -g0[::-1])
        assert g0[n] == 0.0
    g = dispersion.symmetric_q_grid(0.4, 5, q_min=0.1)
    assert g[-5] == pytest.approx(0.1)
    assert np.array_equal(g, -g[::-1])
    with pytest.raises(ValueError, match="q_max"):
        dispersion.symmetric_q_grid(0.0, 3)
    with pytest.raises(ValueError, match="n_pos"):
        dispersion.symmetric_q_grid(0.3, 0)
    with pytest.raises(ValueError, match="q_min"):
        dispersion.symmetric_q_grid(0.3, 3, q_min=0.9)


def test_pm_asymmetry_is_exactly_zero_at_beta_zero_level1():
    Q = dispersion.symmetric_q_grid(0.3, 6, include_zero=True)
    E = np.linspace(0.2, 1.2, 51)
    r = np.imag(dispersion.sheet_reflection(Q[:, None], E[None, :], 1.0, 0.05, 0.0))
    q_pos, asym = dispersion.pm_asymmetry(Q, r)
    assert q_pos.size == 6
    assert np.count_nonzero(asym) == 0  # exactly, not approximately


def test_pm_asymmetry_is_exactly_zero_at_beta_zero_level3():
    sysm = _ribbon_system(L=41)
    qs = dispersion.symmetric_q_grid(0.08, 2)
    Es = np.linspace(0.45, 0.75, 21)
    M = dispersion.ldos_map(sysm, qs, Es, (0.0, 0.0, 6.0))
    _q, asym = dispersion.pm_asymmetry(qs, M)
    assert np.count_nonzero(asym) == 0
    # and the map itself is even in q, bit for bit
    assert np.array_equal(M, M[::-1])


def test_pm_asymmetry_is_clearly_nonzero_at_beta_03():
    # Level 1
    Q = dispersion.symmetric_q_grid(0.3, 8)
    E = np.linspace(0.2, 1.4, 81)
    r = np.imag(dispersion.sheet_reflection(Q[:, None], E[None, :], 1.0, 0.05, 0.3))
    _q, asym = dispersion.pm_asymmetry(Q, r)
    assert np.max(np.abs(asym)) > 0.05 * np.max(np.abs(r))
    # Level 3
    drift = solver.RibbonSystem(
        solver.Ribbon(W_nm=W_REF, EF_ev=EF_REF, gamma_ev=0.02, beta=0.3, L=41)
    )
    qs = dispersion.symmetric_q_grid(0.08, 2)
    Es = np.linspace(0.4, 0.9, 31)
    M = dispersion.ldos_map(drift, qs, Es, (0.0, 0.0, 6.0))
    _q3, asym3 = dispersion.pm_asymmetry(qs, M)
    assert np.max(np.abs(asym3)) > 0.05 * np.max(np.abs(M))
    # the drifted bands SHIFT: the +q and -q peaks sit at different energies
    pk, _w = dispersion.track_peaks(Es, M, n_peaks=1)
    assert pk[-1, 0] != pk[0, 0]


def test_pm_asymmetry_normalise_and_validation():
    q = dispersion.symmetric_q_grid(0.2, 2)
    vals = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    qp, a = dispersion.pm_asymmetry(q, vals, normalise=True)
    assert qp.tolist() == [0.1, 0.2]
    #  q = [-0.2, -0.1, 0.1, 0.2]: the +q rows are vals[2], vals[3] and their
    #  mirror partners vals[1], vals[0]
    assert a[0, 0] == pytest.approx(2.0 * (5.0 - 3.0) / (5.0 + 3.0))
    assert a[1, 0] == pytest.approx(2.0 * (9.0 - 1.0) / (9.0 + 1.0))
    assert np.isnan(a[0, 1]) and np.isnan(a[1, 1])  # 0 + 0 denominator
    raw_q, raw = dispersion.pm_asymmetry(q, vals)
    assert raw[0, 0] == 5.0 - 3.0
    assert raw[1, 0] == 9.0 - 1.0
    assert raw_q.tolist() == [0.1, 0.2]
    with pytest.raises(ValueError, match="values.shape"):
        dispersion.pm_asymmetry(q, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="strictly increasing"):
        dispersion.pm_asymmetry(q[::-1], vals)
    with pytest.raises(ValueError, match="antisymmetric"):
        dispersion.pm_asymmetry(np.array([-0.1, 0.0, 0.2, 0.3]), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="at least 2"):
        dispersion.pm_asymmetry(np.array([0.1]), np.zeros((1, 2)))
    # an all-negative grid cannot be mirror symmetric, so it is the antisymmetry
    # check that catches it -- there is no reachable "nothing positive" case
    with pytest.raises(ValueError, match="antisymmetric"):
        dispersion.pm_asymmetry(np.array([-0.2, -0.1]), np.zeros((2, 2)))


# =========================================================================== #
# CROSS-LEVEL 2: Level 2 bands vs Level 3 peak-tracked LDOS
# =========================================================================== #
def test_level3_peaks_sit_on_the_level2_bands_within_the_linewidth():
    """The real validation: two independent routes to the same bands.

    Level 2 inverts ``Re eta(E) = eta_jq`` on one eigendecomposition per ``q``;
    Level 3 solves the driven problem on a dense energy grid and peak-tracks the
    LDOS.  Nothing is tuned: the tolerance is a fraction of the Level 2
    linewidth, which is the only scale on which a peak position is meaningful.
    """
    sysm = _ribbon_system(L=61)
    qs = np.array([0.02, 0.05])
    Es = np.linspace(0.35, 1.35, 1001)
    bands = dispersion.ribbon_bands(sysm, qs, n_bands=3, bright_only=True)
    M = dispersion.ldos_map(sysm, qs, Es, (0.0, 0.0, 6.0))
    peaks, weights = dispersion.track_peaks(Es, M, n_peaks=3, sort="energy")

    # the two bands inside the energy window are found, in order
    for iq in range(qs.size):
        for band in range(2):
            E2 = bands.E_ev[iq, band]
            E3 = peaks[iq, band]
            fwhm = bands.fwhm_ev[iq, band]
            assert np.isfinite(E2) and np.isfinite(E3) and np.isfinite(fwhm)
            assert abs(E3 - E2) < 0.2 * fwhm, (iq, band, E2, E3, fwhm)
    # the third band lies above the window, so Level 3 honestly reports NaN
    assert np.all(bands.E_ev[:, 2] > Es[-1])
    assert np.all(np.isnan(peaks[:, 2]))
    # and the fundamental dominates the weight, as its zeta says it should
    assert np.all(weights[:, 0] > 2.0 * weights[:, 1])
    assert np.all(bands.weight[:, 0] > bands.weight[:, 1])


# =========================================================================== #
# Speed budgets (the user flagged speed twice).  Deselect with -m "not perf".
# =========================================================================== #
@pytest.mark.perf
def test_level1_is_vectorised_over_the_whole_grid():
    """A 240 x 400 (Q, E) map in one closed-form call, under 0.5 s."""
    Q = dispersion.symmetric_q_grid(0.5, 120)
    E = np.linspace(0.05, 2.0, 400)
    t0 = time.perf_counter()
    r = dispersion.sheet_reflection(Q[:, None], E[None, :], 1.0, 0.05, 0.3)
    dt = time.perf_counter() - t0
    assert r.shape == (240, 400)
    assert dt < 0.5, f"Level 1 map took {dt:.3f} s"


@pytest.mark.perf
def test_level2_reuses_one_eigendecomposition_per_q():
    """Cost must scale with nq, not with nq x n_bands x nE."""
    sysm = _ribbon_system(L=121)
    qs = np.linspace(0.005, 0.1, 12)
    t0 = time.perf_counter()
    bs = dispersion.ribbon_bands(sysm, qs, n_bands=4, bright_only=True)
    dt = time.perf_counter() - t0
    assert np.isfinite(bs.E_ev[:, 0]).all()
    assert dt < 3.0, f"Level 2 over 12 q took {dt:.3f} s"


@pytest.mark.perf
def test_level3_eigen_route_beats_the_full_solve():
    """The documented fast path: same answer, measurably cheaper.

    ``dispersion.ldos_map``'s docstring quotes ~6.6x at ``L = 121``,
    ``nE = 601``.  The assertion is deliberately loose (>= 2x) because the
    margin is BLAS- and machine-dependent; what must not regress is that the
    eigen route is cheaper at all and agrees to roundoff.
    """
    sysm = _ribbon_system(L=121)
    Es = np.linspace(0.3, 1.5, 601)
    qs = np.array([0.03])
    r0 = (0.0, 0.0, 6.0)
    dispersion.ldos_map(sysm, qs, Es[:3], r0)  # warm the V cache
    t0 = time.perf_counter()
    fast = dispersion.ldos_map(sysm, qs, Es, r0)
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    full = dispersion.ldos_map(sysm, qs, Es, r0, fast_path=False)
    t_full = time.perf_counter() - t0
    assert np.max(np.abs(fast - full)) < 1e-9 * np.max(np.abs(full))
    assert t_full > 2.0 * t_fast, f"fast {t_fast:.3f} s vs full {t_full:.3f} s"
