"""Tests for ``web.py``: the JSON boundary the Toolkit page talks to.

The subject of this file is a boundary, not a physical formula, so the tests are
about the properties a boundary has to have rather than about numbers:

1. **It really produces JSON.** ``json.dumps`` on every entry point's output,
   with no numpy scalar, no complex number and no NaN surviving. A numpy float64
   crossing the Pyodide bridge is a silent failure in the browser and nowhere
   else, so it is checked here.
2. **It never raises.** Every malformed spec comes back as ``ok: False`` with a
   message aimed at a user. A traceback crossing the bridge gives the page
   nothing to show.
3. **It does not swallow the library's warnings.** ``electrostatics_fd`` warns
   rather than lying -- an unresolved quadrature, a zero standoff, a map that
   cannot represent stacked ribbons -- and a boundary that dropped those would
   turn hedged numbers into confident ones. :func:`test_a_library_warning_reaches_the_notes`
   is the regression test for exactly that.
4. **It refuses before the library has to**, where refusing early gives a better
   message: ``q = 0`` with a substrate, a node count that would freeze the tab,
   drift combined with the nonlinear orders.
5. **It carries the physics facts the page must not misreport**: hole doping is
   identical to electron doping, the LDOS is in arbitrary units, and a layered
   run screens its source and its response through the same r(Q, omega).

:func:`test_auto_tuned_substrate_quadrature_silences_the_resolution_warning`
is worth reading before changing ``_tune_vbar``: the library's default ``Qmax``
carries a term its own docstring flags as running the wrong way, and the
auto-tune is what keeps a layered run from being quietly under-resolved.

Units: eV-nm-fs Gaussian throughout (see ``units.py``).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE the import style. Every other test module here uses the flat form
# (``import solver``), which conftest.py supports by putting the package
# directory itself on sys.path. ``web`` cannot be imported that way: it uses
# package-relative imports (``from . import dispersion as _disp``) precisely
# because it is the module that ships inside the browser wheel, where the flat
# style has no meaning. So it is imported as part of the package.
from electrostatics_fd import web  # noqa: E402

# ===========================================================================
# helpers
# ===========================================================================

BASE = {
    "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
    "energies_eV": {"min": 0.2, "max": 1.0, "n": 12},
}

LAYERED = {
    "ribbons": [{"W_nm": 10.0, "z0_nm": 1.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
    "substrate": [{"eps_par": 4.0}],
    # q = 0.01 deliberately: the library's default Qmax only fails to resolve the
    # integrand feature of width |q| once |q| drops to about 0.01, so a larger q
    # would make test_auto_tuned_substrate_quadrature_silences_the_resolution_warning
    # vacuous -- it would pass with the auto-tune removed.
    "q_nm_inv": 0.01,
    "energies_eV": {"min": 0.3, "max": 0.9, "n": 4},
}

ALL_KINDS = (
    ("geometry", BASE),
    ("eigenmodes", BASE),
    ("bands", dict(BASE, qs={"max": 0.2, "n_pos": 3}, n_bands=2)),
    ("sheet_dispersion", dict(BASE, qs={"min": 0.02, "max": 0.2, "n_pos": 4})),
    ("linear_spectrum", BASE),
    (
        "eels_spectrum",
        dict(BASE, source={"kind": "electron", "v_over_c": 0.2, "b_nm": 5.0}, q_nm_inv=0.05),
    ),
    (
        "ldos",
        dict(
            BASE,
            qs={"max": 0.2, "n_pos": 3},
            source={"kind": "dipole", "r0_nm": [0.0, 0.0, 5.0]},
        ),
    ),
    ("nonlinear_spectrum", dict(BASE, energies_eV={"min": 0.3, "max": 0.6, "n": 3})),
)


def _walk(obj):
    """Yield every leaf of a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


# ===========================================================================
# 1. the JSON boundary
# ===========================================================================


@pytest.mark.parametrize("kind,spec", ALL_KINDS, ids=[k for k, _ in ALL_KINDS])
def test_every_entry_point_returns_serialisable_json(kind, spec):
    out = web.run(kind, spec)
    assert out["ok"] is True, out.get("reason")
    # json.dumps is the real test: it rejects numpy scalars, complex and NaN.
    text = json.dumps(out, allow_nan=False)
    assert len(text) > 50


@pytest.mark.parametrize("kind,spec", ALL_KINDS, ids=[k for k, _ in ALL_KINDS])
def test_no_numpy_or_complex_type_escapes(kind, spec):
    out = web.run(kind, spec)
    for leaf in _walk(out):
        assert not isinstance(leaf, (np.generic, np.ndarray)), f"{kind}: numpy leaked: {leaf!r}"
        assert not isinstance(leaf, complex), f"{kind}: complex leaked: {leaf!r}"
        if isinstance(leaf, float):
            assert np.isfinite(leaf), f"{kind}: non-finite float leaked"


def test_non_finite_values_become_null_not_zero():
    """A missing band must plot as a gap, never as a band sitting at zero."""
    out = web._fl(np.array([1.0, np.nan, 3.0, np.inf, -np.inf]))
    assert out[0] == 1.0 and out[2] == 3.0
    assert out[1] is None and out[3] is None and out[4] is None


def test_complex_arrays_split_into_re_and_im():
    out = web._cx(np.array([1 + 2j, -3 - 4j]))
    assert out == {"re": [1.0, -3.0], "im": [2.0, -4.0]}


@pytest.mark.parametrize("kind,spec", ALL_KINDS, ids=[k for k, _ in ALL_KINDS])
def test_notes_are_always_a_list_of_strings(kind, spec):
    out = web.run(kind, spec)
    assert isinstance(out["notes"], list)
    for n in out["notes"]:
        assert isinstance(n, str) and n.strip()


# ===========================================================================
# 2. it never raises
# ===========================================================================


BAD_SPECS = [
    ("no ribbons", "linear_spectrum", {"ribbons": []}),
    ("negative width", "linear_spectrum", {"ribbons": [{"W_nm": -5.0}]}),
    ("zero dx", "linear_spectrum", dict(BASE, dx_nm=0.0)),
    ("zero energy", "linear_spectrum", dict(BASE, energies_eV={"min": 0.0, "max": 1.0, "n": 5})),
    ("too many nodes", "linear_spectrum", {"ribbons": [{"W_nm": 10.0, "L": 99999}]}),
    ("too many points", "linear_spectrum", dict(BASE, energies_eV={"min": 0.1, "max": 1.0, "n": 99999})),
    ("empty grid", "linear_spectrum", dict(BASE, energies_eV=[])),
    ("bad source", "linear_spectrum", dict(BASE, source={"kind": "neutrino"})),
    ("bad layer", "linear_spectrum", dict(BASE, substrate=[{"eps_par": "glass"}])),
    ("both stacks", "linear_spectrum", dict(LAYERED, layers=[{"eps_par": 1.0}])),
    ("eels without electron", "eels_spectrum", dict(BASE, source={"kind": "plane_wave"})),
    ("penetrating electron", "eels_spectrum", dict(BASE, source={"kind": "electron", "b_nm": 0.0})),
]


@pytest.mark.parametrize("label,kind,spec", BAD_SPECS, ids=[b[0] for b in BAD_SPECS])
def test_a_bad_spec_comes_back_as_data_with_a_usable_reason(label, kind, spec):
    out = web.run(kind, spec)
    assert out["ok"] is False, f"{label}: expected a refusal"
    assert out["kind"] == kind
    reason = out["reason"]
    assert isinstance(reason, str) and len(reason) > 20, f"{label}: reason too terse"
    # 'unexpected' marks a bug in the library rather than a bad request; none of
    # the specs above should reach that path.
    assert not out.get("unexpected"), f"{label}: leaked an unexpected exception: {reason}"
    json.dumps(out)


def test_an_unknown_calculation_lists_the_ones_that_exist():
    out = web.run("spectroscopy", BASE)
    assert out["ok"] is False
    for kind, _ in ALL_KINDS:
        assert kind in out["reason"]


def test_the_node_ceiling_says_what_to_do_about_it():
    out = web.run("linear_spectrum", {"ribbons": [{"W_nm": 10.0, "L": 5000}]})
    assert out["ok"] is False
    assert "dx_nm" in out["reason"] and "cube" in out["reason"]


# ===========================================================================
# 3. warnings are not swallowed
# ===========================================================================


def test_a_library_warning_reaches_the_notes():
    """A zero standoff is a real accuracy limit, and the page must be told."""
    spec = dict(LAYERED, ribbons=[{"W_nm": 10.0, "z0_nm": 0.0, "L": 61}])
    out = web.run("linear_spectrum", spec)
    assert out["ok"] is True, out.get("reason")
    joined = " ".join(out["notes"])
    assert "UserWarning" in joined, f"the library's warning was swallowed: {out['notes']}"
    assert "standoff" in joined.lower()


def test_auto_tuned_substrate_quadrature_silences_the_resolution_warning():
    """The auto-tune is an accuracy fix, not a cosmetic one.

    With the library's own default ``Qmax`` the step dQ fails to resolve the
    integrand feature of width ``|q|`` and ``build_Vbar`` warns.  The auto-tune
    picks ``Qmax`` from the truncation requirement alone, which resolves it at
    the default ``NQ``.  If this test ever fails in the "off" direction, the
    library's default has changed and ``_tune_vbar`` should be revisited.
    """
    off = web.run("linear_spectrum", dict(LAYERED, vbar_auto=False))
    on = web.run("linear_spectrum", LAYERED)
    assert off["ok"] and on["ok"]

    def resolution_warnings(res):
        return [n for n in res["notes"] if "does not resolve" in n]

    assert resolution_warnings(off), "expected the library's default to warn here"
    assert not resolution_warnings(on), f"auto-tune failed to resolve: {resolution_warnings(on)}"
    assert any("auto-tuned" in n for n in on["notes"])


def test_the_auto_tune_reports_a_clamped_NQ_rather_than_hiding_it():
    """A q so small that resolving it is unaffordable must say so."""
    spec = dict(LAYERED, q_nm_inv=2e-4, energies_eV={"min": 0.5, "max": 0.5, "n": 1})
    out = web.run("linear_spectrum", spec)
    assert out["ok"] is True, out.get("reason")
    assert any("CLAMPED" in n for n in out["notes"]), out["notes"]


# ===========================================================================
# 4. refusing early, where the message can be better
# ===========================================================================


def test_q_zero_with_a_substrate_is_refused_with_a_way_forward():
    out = web.run("linear_spectrum", dict(LAYERED, q_nm_inv=0.0))
    assert out["ok"] is False
    assert "divergent" in out["reason"]
    assert "finite q" in out["reason"]
    assert not out.get("unexpected")


def test_Q_zero_is_not_a_point_of_the_extended_sheet_map():
    out = web.run("sheet_dispersion", dict(BASE, qs_nm_inv=[0.0, 0.1]))
    assert out["ok"] is False
    assert "1/|Q|" in out["reason"]


def test_nonlinear_with_drift_is_refused_because_nobody_has_derived_it():
    spec = dict(BASE, ribbons=[{"W_nm": 10.0, "beta": 0.2, "L": 61}])
    out = web.run("nonlinear_spectrum", spec)
    assert out["ok"] is False
    assert "isotropic" in out["reason"]


def test_a_penetrating_trajectory_is_refused_not_approximated():
    out = web.run(
        "eels_spectrum", dict(BASE, source={"kind": "electron", "b_nm": 0.0, "v_over_c": 0.2})
    )
    assert out["ok"] is False
    assert "aloof" in out["reason"].lower()


# ===========================================================================
# 5. physics the page must not misreport
# ===========================================================================


def test_hole_doping_gives_the_same_spectrum_as_electron_doping():
    """The Dirac cone is electron-hole symmetric; the page must show that."""
    plus = web.run("linear_spectrum", dict(BASE, ribbons=[{"W_nm": 10.0, "EF_eV": +0.5, "L": 61}]))
    minus = web.run("linear_spectrum", dict(BASE, ribbons=[{"W_nm": 10.0, "EF_eV": -0.5, "L": 61}]))
    assert plus["ok"] and minus["ok"]
    assert plus["alpha_re"] == minus["alpha_re"]
    assert plus["alpha_im"] == minus["alpha_im"]


def test_hole_doping_is_labelled_rather_than_silently_folded():
    out = web.run(
        "linear_spectrum",
        dict(BASE, ribbons=[{"W_nm": 10.0, "EF_eV": -0.5, "L": 61}],
             material={"EF_eV": -0.5}),
    )
    assert out["ok"] is True
    assert "symmetric" in out["material"]["note"]
    assert out["material"]["abs_EF_eV"] == 0.5


def test_the_material_panel_puts_the_pauli_edge_at_twice_the_modulus():
    """And it reads the RIBBON, not a spec-level material block beside it.

    Until 2026-11-05 the sigma(omega) shown beneath a spectrum came from the
    top-level ``material`` block while the spectrum itself came from the
    ribbons, so a spec that disagreed with itself got a panel describing a
    sheet that was never solved.  The page always wrote the two consistently,
    so nothing was ever shown wrong, but the panel now follows ribbon 1 and
    cannot drift from it.  This test used to set only the block, which is how
    it caught the change.
    """
    rib = [{"W_nm": 10.0, "EF_eV": -0.4, "gamma_eV": 0.02, "L": 61}]
    out = web.run("linear_spectrum", dict(BASE, ribbons=rib))
    assert out["material"]["pauli_edge_eV"] == pytest.approx(0.8)

    # Disagree with it on purpose: the solved ribbon is what the panel shows.
    out = web.run("linear_spectrum", dict(BASE, ribbons=rib, material={"EF_eV": 0.9}))
    assert out["material"]["pauli_edge_eV"] == pytest.approx(0.8)


def test_the_ldos_says_it_is_arbitrary_units():
    """The 2 pi^2 |p|^2 normalisation has no external confirmation. Say so."""
    out = web.run("ldos", dict(BASE, qs={"max": 0.2, "n_pos": 3}))
    assert out["ok"] is True
    joined = " ".join(out["notes"]).upper()
    assert "ARBITRARY UNITS" in joined


def test_a_layered_run_says_both_halves_screen_through_the_same_r():
    """Was: "it drives with the free-space source". That is no longer true.

    The reflected source landed on 2026-11-05, so the response and the drive now
    screen through one r(Q, omega) and cannot disagree about the stack. The note
    still has to say the layered path has no external reference, because that
    part has not changed.
    """
    out = web.run("linear_spectrum", LAYERED)
    joined = " ".join(out["notes"])
    assert "same r(Q, omega)" in joined
    assert "reference implementation" in joined
    assert "free-space source" not in joined, "the stale caveat is still being shown"


def test_a_drifted_sheet_map_says_it_is_not_a_doppler_shift():
    out = web.run(
        "sheet_dispersion",
        dict(BASE, qs={"min": 0.02, "max": 0.2, "n_pos": 4}, material={"beta": 0.2}),
    )
    assert out["ok"] is True
    joined = " ".join(out["notes"])
    assert "Galilean" in joined
    assert "asymmetry" in out


def test_an_undrifted_sheet_map_has_no_asymmetry_block():
    out = web.run("sheet_dispersion", dict(BASE, qs={"min": 0.02, "max": 0.2, "n_pos": 4}))
    assert "asymmetry" not in out


def test_level_1_uses_the_longitudinal_conductivity_and_says_so():
    out = web.run("sheet_dispersion", dict(BASE, qs={"min": 0.02, "max": 0.2, "n_pos": 4}))
    assert "LONGITUDINAL" in " ".join(out["notes"])


# ===========================================================================
# 6. the system the page describes is the system the library builds
# ===========================================================================


def test_substrate_and_an_explicit_layer_stack_agree():
    """``substrate`` must be exactly ``layers`` with the ambient layer prepended."""
    simple = web.run("linear_spectrum", LAYERED)
    explicit = web.run(
        "linear_spectrum",
        {
            "ribbons": LAYERED["ribbons"],
            "layers": [{"eps_par": 1.0, "d_nm": 0.0}, {"eps_par": 4.0}],
            "q_nm_inv": LAYERED["q_nm_inv"],
            "energies_eV": LAYERED["energies_eV"],
        },
    )
    assert simple["ok"] and explicit["ok"], (simple.get("reason"), explicit.get("reason"))
    assert simple["alpha_re"] == explicit["alpha_re"]
    assert simple["alpha_im"] == explicit["alpha_im"]


def test_geometry_draws_a_line_of_dots_and_carries_no_thickness():
    """Ribbons are two-dimensional. One z per ribbon, and no thickness anywhere."""
    out = web.run(
        "geometry",
        {"ribbons": [{"W_nm": 10.0, "L": 21}, {"W_nm": 4.0, "x0_nm": 15.0, "L": 11}]},
    )
    assert out["ok"] is True
    assert out["n_ribbons"] == 2 and out["n_nodes_total"] == 32
    for r, L in zip(out["ribbons"], (21, 11)):
        assert r["L"] == L
        assert len(r["x_nodes_nm"]) == L
        assert len(r["theta"]) == L
        assert isinstance(r["z0_nm"], float)
        assert "t_nm" not in r and "thickness" not in r


def test_the_fundamental_eigenvalue_is_the_acceptance_value():
    """Ties the boundary to the frozen acceptance table (notes 14)."""
    out = web.run(
        "eigenmodes", dict(BASE, ribbons=[{"W_nm": 10.0, "L": 121}], q_nm_inv=0.0, n_modes=3)
    )
    assert out["ok"] is True
    for got, want in zip(out["eta"]["re"], (-0.0687, -0.0184, -0.0107)):
        assert got == pytest.approx(want, abs=5e-4)


def test_the_null_space_mode_is_hidden_rather_than_reported_as_the_fundamental():
    """The operator's null mode sorts first and its eta diverges.

    ``eigenmodes`` orders by descending |eta|, and the charge-neutrality mode has
    a numerically zero eigenvalue, so eta = 1/lambda comes out at -2e12. Taking
    mode 0 as the fundamental would put that on the page.
    """
    spec = dict(BASE, ribbons=[{"W_nm": 10.0, "L": 121}], q_nm_inv=0.0, n_modes=3)
    bright = web.run("eigenmodes", spec)
    every = web.run("eigenmodes", dict(spec, bright_only=False))
    assert bright["ok"] and every["ok"]

    assert abs(bright["eta"]["re"][0]) < 1.0, "a divergent eta reached the output"
    assert abs(every["eta"]["re"][0]) > 1e6, "expected the null mode to sort first"
    assert every["bright"][0] is False
    assert bright["mode_index"][0] != 0
    assert bright["n_modes_total"] > len(bright["eta"]["re"])
    assert any("null mode" in n for n in bright["notes"])


def test_hiding_the_dark_modes_is_reported_not_silent():
    out = web.run("eigenmodes", dict(BASE, ribbons=[{"W_nm": 10.0, "L": 121}], n_modes=3))
    assert out["bright_only"] is True
    assert any("dark modes hidden" in n for n in out["notes"])
    assert all(b for b in out["bright"])


def test_the_eigenvalues_are_flagged_as_carrying_no_convention_information():
    out = web.run("eigenmodes", BASE)
    assert any("geometric" in n for n in out["notes"])


def test_bands_return_one_row_per_q():
    qs = {"max": 0.2, "n_pos": 3}
    out = web.run("bands", dict(BASE, qs=qs, n_bands=2))
    assert out["ok"] is True
    assert len(out["qs_nm_inv"]) == len(out["E_eV"])
    assert all(len(row) == 2 for row in out["E_eV"])


def test_a_q_range_starting_at_zero_includes_zero_instead_of_being_refused():
    """``q`` from 0 is an ordinary request, and ``q = 0`` is an ordinary point.

    It is the NORMAL-INCIDENCE case -- the one the cross section solves -- so
    the bands there are the published plane-wave resonances. But
    ``symmetric_q_grid``'s ``q_min`` is the smallest POSITIVE sample, so a
    literal 0 was passed into it and came back as "need 0 < q_min <= q_max,
    got (0.0, 0.5)": a refusal from the grid builder that never reached the
    physics. A range starting at zero is now read as what it says.
    """
    out = web.run(
        "bands",
        dict(BASE, qs={"max": 0.5, "n_pos": 4, "min": 0.0}, n_bands=3,
             E_min_eV=1e-3, E_max_eV=1.2),
    )
    assert out["ok"] is True, out.get("reason")
    qs = out["qs_nm_inv"]
    assert 0.0 in qs
    # Still bit-for-bit antisymmetric, which is what pm_asymmetry rests on.
    assert qs == [-q for q in reversed(qs)]

    row = out["E_eV"][qs.index(0.0)]
    # The transverse ribbon modes are exactly the plane-wave resonances there.
    assert row[1] == pytest.approx(0.5755, abs=5e-3)
    assert row[2] == pytest.approx(0.8879, abs=5e-3)
    # And the longitudinal branch sits at exactly zero: its eta diverges as
    # 1/(qW)^2, so Re eta(E) can only meet it at E = 0. That is the endpoint of
    # the branch, not a missing value, so it is reported rather than left null
    # -- which is what closes the curve at the origin instead of stopping it a
    # grid point short.
    assert row[0] == 0.0


def test_a_band_that_leaves_the_TOP_of_the_window_is_described_as_such():
    """A gap is not always at small |q|, and dark is not the same as longitudinal.

    The first version of this note assumed every break was the longitudinal
    branch leaving the bottom at small |q|, because that is the one that
    usually breaks. Handed a band whose energy climbs out of the TOP at large
    |q|, it produced four wrong statements in one sentence: it called band 3
    "the longitudinal (wire) branch", said "its energy goes to zero as q -> 0",
    said the curve "stops at |q| = 0" -- which is exactly where that band is
    live -- and blamed "the bottom of the energy window".

    Band 3 of a 12 nm ribbon swept to |q| = 1 with the window capped at 1.2 eV
    is that case.
    """
    out = web.run(
        "bands",
        dict(
            BASE,
            ribbons=[dict(BASE["ribbons"][0], W_nm=12.0)],
            qs={"max": 1.0, "n_pos": 16, "min": 0.0},
            n_bands=3,
            E_min_eV=1e-3,
            E_max_eV=1.2,
        ),
    )
    assert out["ok"] is True, out.get("reason")
    said = [n for n in out["notes"] if n.startswith("Band 3")]
    assert said, out["notes"]
    note = said[0]
    assert "above the top of the energy window" in note, note
    assert "longitudinal" not in note, note
    assert "goes to zero" not in note, note
    # and it must not claim the break is at a |q| where the band is live
    assert "breaks above" in note, note


def test_an_energy_window_starting_at_zero_is_refused_in_words_not_in_contract():
    """E = 0 is a singularity of the model, and the refusal has to say so.

    ``eta = i sigma / (omega W eps_eff)`` diverges as ``1/omega``, so the band
    search cannot start at zero. ``ribbon_bands`` says that as "need
    0 < E_min < E_max, got (0.0, 1.0)" -- its own contract, quoted at a reader
    who typed 0 into an energy range, which is a reasonable thing to want since
    the longitudinal branch ends exactly there and the AXIS may start at zero.
    Nothing on the page is called ``E_min``.
    """
    out = web.run(
        "bands",
        dict(BASE, qs={"max": 0.5, "n_pos": 4}, n_bands=2, E_min_eV=0.0, E_max_eV=1.0),
    )
    assert out["ok"] is False
    why = out.get("reason") or ""
    assert "strictly positive bottom" in why, why
    assert "diverges as 1/omega" in why, why
    # it must tell the reader what to do, and that the axis is not the problem
    assert "1e-4" in why and "axis is free to start at zero" in why, why

    # and the same window with a positive floor is fine
    ok = web.run(
        "bands",
        dict(BASE, qs={"max": 0.5, "n_pos": 4}, n_bands=2, E_min_eV=1e-4, E_max_eV=1.0),
    )
    assert ok["ok"] is True, ok.get("reason")


def test_q_zero_is_still_refused_with_a_substrate_because_there_it_diverges():
    """The ONE case where q = 0 is genuinely ill-defined, kept refused.

    ``build_Vbar``'s integrand goes as ``r(Q)/Q``, so with a stack the
    reflected Coulomb integral really is logarithmically divergent at q = 0 and
    any number returned would be set by where the quadrature was cut off.
    Free-standing there is no such integral and the limit is clean, which is
    why the two cases differ.
    """
    sub = [{"d_nm": 0.0, "eps_par": 4.0, "eps_perp": 4.0}]
    out = web.run(
        "bands",
        dict(BASE, substrate=sub, qs={"max": 0.5, "n_pos": 4, "min": 0.0}, n_bands=2),
    )
    assert out["ok"] is False
    assert "q = 0 with a substrate" in (out.get("reason") or "")
    # ... and the same stack is fine once q is finite.
    ok = web.run(
        "bands",
        dict(BASE, substrate=sub, qs={"max": 0.5, "n_pos": 4, "min": 0.02}, n_bands=2),
    )
    assert ok["ok"] is True, ok.get("reason")


def test_the_band_notes_say_the_curves_are_tracked_and_which_branch_broke():
    """A gap in a band is now a gap, so the page has to say why there is one.

    ``bands`` passes ``track=True`` always: each curve is one branch over the
    whole sweep, and where that branch has no resonance inside the energy
    window its curve breaks instead of every curve above it sliding down a
    slot. A reader looking for the optical bands would otherwise just see a
    curve stop. So two notes are required -- one for the tracking, one naming
    the branch that routinely leaves (the longitudinal wire branch, dark to a
    transverse field, energy -> 0 as q -> 0) and the ``|q|`` below which it
    does.

    At the page's own defaults -- ``E_min = 0.1`` eV, ``q`` from 0.02 to 0.5
    over 16 positive samples -- that threshold is 0.052 / nm, and the note has
    to agree with the column it describes.
    """
    spec = dict(
        BASE,
        qs={"min": 0.02, "max": 0.5, "n_pos": 16},
        n_bands=3,
        E_min_eV=0.1,
        E_max_eV=1.2,
    )
    out = web.run("bands", spec)
    assert out["ok"] is True, out.get("reason")
    notes = out["notes"]
    assert any("followed across q by eigenvector overlap" in n for n in notes), notes
    assert any("breaks there rather than shifting the bands above it" in n
               for n in notes), notes
    named = [n for n in notes if "longitudinal (wire) branch" in n]
    assert len(named) == 1, notes
    assert "0.052" in named[0], named[0]
    assert "dark to a transverse field" in named[0]

    # and the note is not decoration: band 1 really does break, and exactly at
    # the |q| the sentence quotes. _fl sends a missing band across as null, so
    # the gap is None here and not a zero (test_non_finite_values_become_null).
    assert "Band 1 is" in named[0]
    col = [row[0] for row in out["E_eV"]]
    qs = [abs(q) for q in out["qs_nm_inv"]]
    missing = [q for q, v in zip(qs, col) if v is None]
    live = [q for q, v in zip(qs, col) if v is not None]
    assert missing and live
    assert max(missing) < min(live)
    assert min(live) == pytest.approx(0.052, abs=1e-6)
    # the bands above it have no holes at all -- that is the whole point
    for band in (1, 2):
        assert all(row[band] is not None for row in out["E_eV"])


# ===========================================================================
# 7. cost, which is the whole reason this boundary exists
# ===========================================================================


def test_cost_estimate_runs_no_physics():
    """It must be cheap enough to call on every slider move."""
    spec = dict(BASE, qs={"max": 0.3, "n_pos": 40}, energies_eV={"min": 0.1, "max": 1.2, "n": 200})
    t0 = time.perf_counter()
    est = web.cost_estimate(spec, "linear_spectrum")
    dt = time.perf_counter() - t0
    assert est["ok"] is True
    assert dt < 0.25, f"cost_estimate took {dt:.3f}s; it is supposed to count, not compute"
    # n_pos = 40 means 40 POSITIVE q, so symmetric_q_grid returns 80.
    assert est["n_q"] == 80 and est["n_energies"] == 200


def test_cost_estimate_refuses_what_run_would_refuse():
    """The button and the solver must agree, whichever way they agree.

    What changed on 2026-11-06 is WHERE the line sits, not that the two sides
    track each other. The solve tally used to refuse on its own, in AND with
    the time ceiling, so a run the cost model priced in seconds was blocked for
    having many cheap solves -- an EELS grid of 5120 cells estimated 13.2 s and
    ran in 8.6 s, and the page would not start it. The tally is now a backstop
    for the screened case only, where the model is measured 4x to 16x low.
    Unscreened, the seconds govern, which is what MAX_ESTIMATED_SECONDS' own
    comment always said.
    """
    drifted = {
        "ribbons": [{"W_nm": 10.0, "beta": 0.2, "L": 121}],
        "qs": {"max": 0.3, "n_pos": 60},
        "energies_eV": {"min": 0.1, "max": 1.2, "n": 200},
    }
    est = web.cost_estimate(drifted, "linear_spectrum")
    assert est["fast_path"] is False
    # 24000 full solves at 121 nodes: a real 62 s, and allowed, because 62 < 120.
    assert est["n_full_solves"] == 24000
    assert est["refuse"] is False

    # Over the time ceiling it is refused, and the solver agrees.
    too_big = dict(drifted, ribbons=[{"W_nm": 10.0, "beta": 0.2, "L": 600}])
    est_big = web.cost_estimate(too_big, "linear_spectrum")
    assert est_big["estimated_seconds"] > 120.0
    assert est_big["refuse"] is True
    out = web.run("linear_spectrum", too_big)
    assert out["ok"] is False and "ceiling" in out["reason"]

    # And the tally still bites where the seconds cannot be trusted.
    layered = dict(
        drifted,
        ribbons=[{"W_nm": 10.0, "z0_nm": 1.0, "L": 121}],
        substrate=[{"eps_par": 4.0}],
        q_nm_inv=0.05,
    )
    est_lay = web.cost_estimate(layered, "linear_spectrum")
    assert est_lay["estimate_is_floor"] is True
    assert est_lay["refuse"] is True
    out_lay = web.run("linear_spectrum", layered)
    assert out_lay["ok"] is False and "ceiling" in out_lay["reason"]


def test_cost_estimate_reports_the_fast_path_obstruction_verbatim():
    est = web.cost_estimate(LAYERED, "linear_spectrum")
    assert est["fast_path"] is False
    assert isinstance(est["fast_path_obstruction"], str)
    assert "substrate" in est["fast_path_obstruction"]


def test_cost_estimate_on_a_bad_spec_answers_rather_than_raising():
    est = web.cost_estimate({"ribbons": []}, "linear_spectrum")
    assert est["ok"] is False and "ribbons" in est["reason"]


def test_cost_grows_with_the_grid():
    small = web.cost_estimate(dict(BASE, qs={"max": 0.2, "n_pos": 2}), "bands")
    big = web.cost_estimate(dict(BASE, qs={"max": 0.2, "n_pos": 40}), "bands")
    assert big["estimated_seconds"] > small["estimated_seconds"]


def test_the_default_cost_model_admits_it_was_not_measured_here():
    """Pyodide is slower than native, so an uncalibrated estimate must say so."""
    before = dict(web._COST)
    try:
        web._COST.update(
            {"calibrated": False, "source": "default: measured on native CPython, NOT on this runtime"}
        )
        est = web.cost_estimate(BASE, "linear_spectrum")
        assert est["calibrated"] is False
        assert "NOT on this runtime" in est["cost_model"]
    finally:
        web._COST.clear()
        web._COST.update(before)


def test_calibrate_measures_this_runtime_and_says_so():
    before = dict(web._COST)
    try:
        res = web.calibrate(sizes=(31, 61))
        assert res["ok"] is True
        assert res["n_nodes"] == [31, 61]
        assert all(t > 0.0 for t in res["eigen_seconds"])
        assert all(t > 0.0 for t in res["solve_seconds"])
        assert web._COST["calibrated"] is True
        est = web.cost_estimate(BASE, "linear_spectrum")
        assert est["calibrated"] is True
        assert "calibrated on this runtime" in est["cost_model"]
        json.dumps(res)
    finally:
        web._COST.clear()
        web._COST.update(before)


def test_the_cost_model_never_decreases_with_node_count():
    """A negative fitted coefficient would make a bigger grid look cheaper."""
    for prefix in ("per_q", "per_solve"):
        assert web._COST[prefix + "_c2"] >= 0.0
        assert web._COST[prefix + "_c3"] >= 0.0
    costs = [web._unit_cost("per_q", n) for n in (31, 61, 121, 301, 601)]
    assert costs == sorted(costs)


def test_the_two_point_fit_clamps_a_negative_coefficient_to_zero():
    # Timing noise that implies a negative cubic term: a faster large measurement.
    c2, c3 = web._fit_two_point(61, 0.010, 121, 0.011)
    assert c2 >= 0.0 and c3 >= 0.0


# ===========================================================================
# 8. sources that do not exist on the requested grid
# ===========================================================================


def test_a_dipole_at_q_zero_is_refused_with_advice_not_a_traceback():
    """The page's default q is 0, so this is the first thing a user hits.

    A point dipole has no q = 0 Fourier component -- K0(|q|s) and K1(|q|s) both
    diverge -- and the library raises. Before this was caught at the boundary it
    reached the page as ``unexpected: true``, which tells a user nothing.
    """
    out = web.run("linear_spectrum", dict(BASE, source={"kind": "dipole", "r0_nm": [0, 0, 5]}))
    assert out["ok"] is False
    assert not out.get("unexpected"), "a user error leaked as an internal exception"
    assert "q = 0" in out["reason"]
    assert "plane wave" in out["reason"], "the refusal should name the q = 0 source that does exist"


def test_the_same_dipole_works_at_finite_q():
    out = web.run(
        "linear_spectrum",
        dict(BASE, q_nm_inv=0.05, source={"kind": "dipole", "r0_nm": [0, 0, 5]}),
    )
    assert out["ok"] is True, out.get("reason")


def test_the_ldos_map_refuses_a_q_grid_containing_zero():
    out = web.run(
        "ldos",
        dict(BASE, qs_nm_inv=[0.0, 0.1], source={"kind": "dipole", "r0_nm": [0, 0, 5]}),
    )
    assert out["ok"] is False
    assert not out.get("unexpected")
    assert "q = 0" in out["reason"]


# ===========================================================================
# 9. the per-cell cost term
# ===========================================================================


def test_cost_estimate_counts_cells_not_just_solves():
    """A map's cost lives in its cells, and leaving them out under-priced it 8x.

    The LDOS default is 32 q by 160 energies. On the fast path that is 32
    eigendecompositions and 5120 cells, and the cells dominate: pricing only the
    decompositions reported 0.8 s for a calculation that took 6.4 s.
    """
    spec = dict(
        BASE,
        qs={"min": 0.02, "max": 0.5, "n_pos": 16},
        energies_eV={"min": 0.1, "max": 1.2, "n": 160},
        source={"kind": "dipole", "r0_nm": [0, 0, 5]},
    )
    est = web.cost_estimate(spec, "ldos")
    assert est["ok"] is True
    assert est["n_cells"] == est["n_q"] * est["n_energies"]
    assert est["n_cells"] == 5120
    # The cell term must be the dominant one here, not a rounding correction.
    per_q_only = est["n_q_units"] * web._unit_cost("per_q", est["n_nodes"])
    assert est["estimated_seconds"] > 2.0 * per_q_only


def test_a_calculation_with_no_cells_prices_none():
    est = web.cost_estimate(BASE, "eigenmodes")
    assert est["n_cells"] == 0


def test_calibrate_measures_the_cell_term_rather_than_inferring_it():
    """Inferring it from the solve scale was wrong by 10x in a browser.

    WASM's penalty on an O(N^3) LAPACK solve is not its penalty on O(N) numpy
    over the nodes. Using the solve's factor for both priced the LDOS default at
    84 s for a run that took 8.3 s, and a ten-fold over-estimate on the one
    module most able to lock a tab just stops anyone opening it.
    """
    before = dict(web._COST)
    try:
        res = web.calibrate(sizes=(31, 61))
        assert res["ok"] is True
        assert "per_cell_scale" in res
        assert res["per_cell_scale"] > 0.0
        # Measured, not copied from the solve factor. They are independent
        # quantities and would only coincide by accident.
        assert res["per_cell_scale"] != res["per_solve_scale"]
        assert "inferred" not in res["model"]["source"]
        assert res["cell_probe_seconds"] > 0.0
        json.dumps(res)
    finally:
        web._COST.clear()
        web._COST.update(before)


def test_the_cell_probe_is_cheap_enough_to_run_at_startup():
    before = dict(web._COST)
    try:
        t0 = time.perf_counter()
        web.calibrate(sizes=(31, 61))
        dt = time.perf_counter() - t0
        # Natively this is well under a second; a browser is a few times slower,
        # and the page shows a progress bar while it happens.
        assert dt < 5.0, f"calibration took {dt:.2f}s, too slow for every page load"
    finally:
        web._COST.clear()
        web._COST.update(before)


# ===========================================================================
# 10. the material view — the input to everything else, shown by itself
# ===========================================================================


def test_material_needs_no_source_and_runs_under_any_excitation():
    """It is a property of the sheet, so it must not depend on how you drive it."""
    plain = web.run("material", BASE)
    assert plain["ok"] is True, plain.get("reason")
    # The same spec with each source attached must give the same conductivity.
    for src in (
        {"kind": "plane_wave"},
        {"kind": "dipole", "r0_nm": [0, 0, 5]},
        {"kind": "electron", "v_over_c": 0.2, "b_nm": 5},
    ):
        out = web.run("material", dict(BASE, source=src))
        assert out["ok"] is True, out.get("reason")
        assert out["material"]["sigma_nm_fs"] == plain["material"]["sigma_nm_fs"]


def test_material_runs_no_physics_and_is_priced_free():
    est = web.cost_estimate(BASE, "material")
    assert est["ok"] is True
    assert est["n_full_solves"] == 0 and est["n_q_units"] == 0 and est["n_cells"] == 0
    t0 = time.perf_counter()
    out = web.run("material", dict(BASE, energies_eV={"min": 0.1, "max": 1.2, "n": 400}))
    dt = time.perf_counter() - t0
    assert out["ok"] is True
    assert dt < 2.0, f"the material view took {dt:.2f}s; it builds no matrices"


def test_material_eta_is_the_solvers_own_eta_not_a_second_copy():
    """A re-derived eta here could drift from the one the solver actually uses."""
    import solver as solver_mod  # noqa: PLC0415 - the flat test-suite import style

    spec = dict(BASE, q_nm_inv=0.0)
    out = web.run("material", spec)
    assert out["ok"] is True
    system = solver_mod.RibbonSystem([solver_mod.Ribbon(W_nm=10.0, EF_ev=0.5, gamma_ev=0.02, L=61)])
    E = out["energies_eV"][3]
    perp, par = system.eta_pair(0, 0.0, E)
    assert out["eta_perp"]["re"][3] == pytest.approx(perp.real, rel=1e-12)
    assert out["eta_par"]["im"][3] == pytest.approx(par.imag, rel=1e-12)


def test_material_says_where_the_resonances_will_be():
    out = web.run("material", BASE)
    joined = " ".join(out["notes"])
    assert "eta_j" in joined and "resonance" in joined


def test_drift_material_splits_the_tensor_and_names_the_longitudinal_one():
    spec = dict(BASE, ribbons=[{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.0263, "beta": 0.3, "L": 61}])
    out = web.run("material", spec)
    assert out["ok"] is True
    assert "sigma_par_nm_fs" in out and "sigma_perp_nm_fs" in out
    joined = " ".join(out["notes"])
    assert "LONGITUDINAL" in joined
    assert "Galilean" in joined, "the drift note must rule out the frame shift"


def test_undrifted_material_has_no_tensor_split():
    out = web.run("material", BASE)
    assert "sigma_par_nm_fs" not in out and "sigma_perp_nm_fs" not in out


# ===========================================================================
# 11. the near field
# ===========================================================================

NEAR = dict(BASE, E_ev=0.5774, nx=40, nz=24)


def test_near_field_returns_a_grid_with_the_ribbons_marked():
    out = web.run("near_field", NEAR)
    assert out["ok"] is True, out.get("reason")
    assert len(out["xs_nm"]) == 40
    assert len(out["zs_nm"]) == 24
    assert len(out["map"]) == 40 and len(out["map"][0]) == 24
    # The ribbons come back so the map can be drawn with the structure on it.
    assert len(out["ribbons"]) == 1
    assert len(out["ribbons"][0]["x_nodes_nm"]) == 61  # BASE's L
    json.dumps(out)


def test_near_field_takes_the_q_to_zero_limit_numerically():
    """2|q| K_1(|q| s) is 0 x inf at exactly zero, and the plane wave is q = 0.

    This used to assert that the payload said so in a note as well.  The note
    was dropped on the user's instruction on 2026-11-05 -- see
    :func:`test_the_near_field_notes_do_not_repeat_the_caveats` -- so what is
    left here is the behaviour itself: the limit is taken, and taking it leaves
    a map of finite numbers rather than a map of holes.
    """
    out = web.run("near_field", NEAR)
    assert out["ok"] is True
    finite = [v for rowv in out["map"] for v in rowv if v is not None]
    assert finite, "the whole map came back as gaps"
    assert all(np.isfinite(v) for v in finite)


def test_near_field_at_q_zero_matches_a_small_finite_q():
    """If the limit were not converged, these would differ visibly."""
    a = web.run("near_field", NEAR)
    b = web.run("near_field", dict(NEAR, q_nm_inv=1e-4))
    assert a["ok"] and b["ok"]
    va = np.array([v for rowv in a["map"] for v in rowv if v is not None])
    vb = np.array([v for rowv in b["map"] for v in rowv if v is not None])
    assert va.size == vb.size
    assert np.allclose(va, vb, rtol=2e-3)


def test_the_near_field_notes_do_not_repeat_the_caveats():
    """Dropped on the user's instruction, 2026-11-05. Recorded, not forgotten.

    Three notes used to ride on every near-field payload: the q -> 0 limit, the
    induced-field/arbitrary-scale caveat, and the one about gaps where the grid
    meets a node.  Of those, the first two survive where the reader already
    looks -- the chart title reads "... of the induced field over (x, z) --
    arbitrary scale" -- and the behaviour of all three is pinned by the tests
    around this one.  The gaps caveat now has no home in the UI at all; it
    belongs in the notes page, which does not yet carry it.

    This test exists so that re-adding them is a decision rather than an
    accident.
    """
    out = web.run("near_field", NEAR)
    joined = " ".join(out["notes"])
    assert "arbitrary scale" not in joined
    assert "hot spots" not in joined
    assert "q -> 0 limit" not in joined


def test_near_field_refuses_a_map_too_large_to_draw():
    out = web.run("near_field", dict(NEAR, nx=400, nz=400))
    assert out["ok"] is False
    assert "ceiling" in out["reason"]
    assert not out.get("unexpected")


def test_near_field_refuses_a_non_positive_energy():
    out = web.run("near_field", dict(NEAR, E_ev=0.0))
    assert out["ok"] is False
    assert not out.get("unexpected")


def test_near_field_cost_counts_its_grid_points():
    est = web.cost_estimate(dict(NEAR, nx=80, nz=60), "near_field")
    assert est["ok"] is True
    assert est["n_full_solves"] == 1
    assert est["n_cells"] == 4800


# ===========================================================================
# 12. the near field's components, and what the page draws over them
# ===========================================================================


def test_near_field_returns_every_component():
    out = web.run("near_field", NEAR)
    assert out["ok"] is True, out.get("reason")
    for key in ("map", "ex_re", "ex_im", "ey_re", "ey_im", "ez_re", "ez_im"):
        assert key in out, f"{key} missing"
        assert len(out[key]) == 40 and len(out[key][0]) == 24


def test_the_magnitude_is_the_components_it_ships():
    """|E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2, cell by cell.

    Not a tautology across the boundary: the magnitude and the six component
    arrays are built in separate statements, so a transposed or mis-assigned
    component would show up here and nowhere else.
    """
    out = web.run("near_field", NEAR)
    assert out["ok"] is True
    for i in range(len(out["map"])):
        for j in range(len(out["map"][0])):
            mag = out["map"][i][j]
            parts = [out[k][i][j] for k in ("ex_re", "ex_im", "ey_re", "ey_im", "ez_re", "ez_im")]
            if mag is None or any(p is None for p in parts):
                continue
            assert mag == pytest.approx(float(np.hypot(np.hypot(*parts[:2]), np.hypot(
                np.hypot(*parts[2:4]), np.hypot(*parts[4:])))), rel=1e-12)


def test_near_field_reports_the_ribbon_outline_for_the_overlay():
    out = web.run(
        "near_field",
        dict(NEAR, ribbons=[{"W_nm": 10.0, "x0_nm": 0.0, "L": 31},
                            {"W_nm": 4.0, "x0_nm": 12.0, "z0_nm": 2.0, "L": 21}]),
    )
    assert out["ok"] is True
    assert len(out["ribbons"]) == 2
    a, b = out["ribbons"]
    # A ribbon of width W centred on x0 spans x0 +- W/2, and the map draws that
    # span as a segment at the ribbon's own height.
    assert a["x_min_nm"] == pytest.approx(-5.0)
    assert a["x_max_nm"] == pytest.approx(5.0)
    assert b["x_min_nm"] == pytest.approx(10.0)
    assert b["x_max_nm"] == pytest.approx(14.0)
    assert b["z0_nm"] == pytest.approx(2.0)


def test_near_field_reports_where_the_source_is():
    """The map has to show the dipole and the electron's path, not just the field."""
    plane = web.run("near_field", NEAR)
    assert plane["source_marker"]["kind"] == "plane_wave"

    dip = web.run(
        "near_field",
        dict(NEAR, q_nm_inv=0.05, source={"kind": "dipole", "r0_nm": [3.0, 0.0, 6.0]}),
    )
    assert dip["ok"] is True
    # `p` joined the marker on 2026-11-05 so the map can draw the dipole's
    # ORIENTATION: x, y and z behave nothing alike and used to share one dot.
    assert dip["source_marker"] == {
        "kind": "dipole",
        "x_nm": 3.0,
        "z_nm": 6.0,
        "p": [1.0, 0.0, 0.0],
    }
    zdip = web.run(
        "near_field",
        dict(
            NEAR,
            q_nm_inv=0.05,
            source={"kind": "dipole", "r0_nm": [3.0, 0.0, 6.0], "p": [0, 0, 1]},
        ),
    )
    assert zdip["ok"] is True
    assert zdip["source_marker"]["p"] == [0.0, 0.0, 1.0]

    el = web.run(
        "near_field",
        dict(NEAR, q_nm_inv=0.05, source={"kind": "electron", "v_over_c": 0.3, "b_nm": 7.0}),
    )
    assert el["ok"] is True
    assert el["source_marker"]["kind"] == "electron"
    assert el["source_marker"]["b_nm"] == pytest.approx(7.0)
    assert el["source_marker"]["v_over_c"] == pytest.approx(0.3)


def test_mode_profiles_are_split_per_ribbon_so_the_x_grids_line_up():
    """profiles[m][j] must match x_nodes_nm[j], ribbon by ribbon.

    Flat, the profile is the whole system's N nodes while x_nodes_nm is a list
    per ribbon, and nothing in the payload says which x belongs to which value.
    The page drew the second mode against the SECOND RIBBON's coordinates and
    reported no error at all.
    """
    out = web.run(
        "eigenmodes",
        dict(BASE, ribbons=[{"W_nm": 10.0, "x0_nm": 0.0, "L": 61},
                            {"W_nm": 22.0, "x0_nm": 39.5, "L": 41}],
             q_nm_inv=0.0, n_modes=3),
    )
    assert out["ok"] is True, out.get("reason")
    widths = [len(x) for x in out["x_nodes_nm"]]
    assert widths == [61, 41]
    for mode in out["profiles"]:
        assert [len(p) for p in mode] == widths
    # And the nodes really are each ribbon's own, not one shared grid.
    assert min(out["x_nodes_nm"][0]) == pytest.approx(-5.0)
    assert min(out["x_nodes_nm"][1]) == pytest.approx(28.5)


# ===========================================================================
# Per-ribbon material and per-ribbon temperature (2026-11-05)
# ===========================================================================
def test_material_carries_one_entry_per_ribbon() -> None:
    """E_F, gamma, beta, T and W are per ribbon, so the panel is too.

    Before this the material view showed ribbon 1's conductivity alone, which
    is a picture of one ribbon standing in for a structure that may hold five
    different ones.
    """
    spec = {
        "ribbons": [
            {"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 31},
            {"W_nm": 20.0, "x0_nm": 40.0, "EF_eV": 0.9, "gamma_eV": 0.03, "L": 31},
        ],
        "energies_eV": {"min": 0.2, "max": 1.0, "n": 8},
    }
    out = web.run("material", spec)
    assert out["ok"], out.get("reason")

    ribs = out["ribbons"]
    assert [r["index"] for r in ribs] == [0, 1]
    assert [r["EF_eV"] for r in ribs] == [0.5, 0.9]
    assert [r["W_nm"] for r in ribs] == [10.0, 20.0]
    # Two dopings, two conductivities -- not one curve repeated twice.
    assert ribs[0]["material"]["sigma_nm_fs"]["im"] != ribs[1]["material"]["sigma_nm_fs"]["im"]
    # eta carries W as well as sigma, so it must differ for a second reason.
    assert ribs[0]["eta_par"]["re"] != ribs[1]["eta_par"]["re"]
    # The top-level keys stay ribbon 1's, for a reader that predates `ribbons`.
    assert out["material"]["sigma_nm_fs"] == ribs[0]["material"]["sigma_nm_fs"]
    assert out["eta_par"] == ribs[0]["eta_par"]
    json.dumps(out)


def test_each_ribbon_takes_its_own_temperature() -> None:
    """T_K on a ribbon reaches that ribbon's conductivity and no other's."""
    spec = {
        "ribbons": [
            {"W_nm": 10.0, "EF_eV": 0.05, "gamma_eV": 0.02, "L": 31},
            {"W_nm": 10.0, "x0_nm": 40.0, "EF_eV": 0.05, "gamma_eV": 0.02, "L": 31, "T_K": 300.0},
        ],
        "energies_eV": {"min": 0.2, "max": 1.0, "n": 8},
    }
    out = web.run("material", spec)
    assert out["ok"], out.get("reason")
    ribs = out["ribbons"]
    assert ribs[0]["T_K"] == 0.0
    assert ribs[1]["T_K"] == 300.0
    assert abs(ribs[1]["eta_par"]["im"][4]) < abs(ribs[0]["eta_par"]["im"][4])


def test_a_spec_level_temperature_still_reaches_every_ribbon() -> None:
    """Older specs, the preset strip and the permalink all wrote T at the top."""
    spec = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.05, "gamma_eV": 0.02, "L": 31}],
        "T_K": 300.0,
        "energies_eV": {"min": 0.2, "max": 1.0, "n": 8},
    }
    out = web.run("material", spec)
    assert out["ok"], out.get("reason")
    assert out["ribbons"][0]["T_K"] == 300.0


def test_a_ribbon_temperature_moves_the_cross_section() -> None:
    """Temperature reaches the SOLVE, not only the material panel.

    This is the regression test for the state before 2026-11-05: ``eta_pair``
    ignored temperature entirely, so the page's T box moved the sigma(omega)
    curve under the spectrum and left the resonance in it exactly where it was.
    A weaker thermal Drude weight red-shifts the plasmon.
    """

    def peak(T_K: float) -> float:
        spec = {
            "ribbons": [{"W_nm": 10.0, "EF_eV": 0.05, "gamma_eV": 0.005, "L": 61, "T_K": T_K}],
            "energies_eV": {"min": 0.05, "max": 0.35, "n": 120},
            "q_nm_inv": 0.0,
            "source": {"kind": "plane_wave"},
        }
        out = web.run("linear_spectrum", spec)
        assert out["ok"], out.get("reason")
        Es = out["energies_eV"]
        row = out["extinction_arb"][0]
        best = max(range(len(row)), key=lambda i: (row[i] if row[i] is not None else -1e30))
        return Es[best]

    cold, warm = peak(0.0), peak(300.0)
    assert warm < cold, "300 K must red-shift the resonance of a 0.05 eV ribbon"
    assert 0.9 < warm / cold < 0.999


def test_the_spectrum_notes_no_longer_explain_extinction_arb() -> None:
    """The charge-convention paragraph is in the notes page, not under the plot."""
    spec = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 31}],
        "energies_eV": {"min": 0.3, "max": 0.9, "n": 8},
        "q_nm_inv": 0.0,
        "source": {"kind": "plane_wave"},
    }
    out = web.run("linear_spectrum", spec)
    assert out["ok"], out.get("reason")
    assert not any("extinction_arb is Im(alpha)" in n for n in out["notes"])


def test_the_material_panel_always_carries_the_sigma_that_eta_uses():
    """The Drude trace is not decoration: it is the sigma every resonance uses.

    `solver.eta_pair` carries the INTRABAND conductivity alone, while the
    panel's main curve is the full local RPA.  Above the Pauli edge those two
    do not even move the same way with temperature, so the panel has to show
    both or the two halves of the Material view contradict each other with
    nothing on screen to explain it.  Until 2026-11-05 the Drude trace was
    dropped entirely at T > 0 -- exactly where the contradiction appears.

    This pins the strong form: the trace IS eta's sigma, to the digits.
    """
    import numpy as _np
    from electrostatics_fd import units as _u

    for T_K in (0.0, 0.5, 300.0):
        spec = {
            "ribbons": [{"W_nm": 10.0, "EF_eV": 0.05, "gamma_eV": 0.02, "L": 31, "T_K": T_K}],
            "energies_eV": {"min": 0.2, "max": 0.2, "n": 1},
            "q_nm_inv": 0.0,
        }
        out = web.run("material", spec)
        assert out["ok"], out.get("reason")
        rib = out["ribbons"][0]

        drude = rib["material"].get("sigma_drude_nm_fs")
        assert drude is not None, f"the Drude trace vanished at T_K={T_K}"

        # eta = i sigma_Drude / (omega W eps_eff), with eps_eff = 1 in vacuum.
        sigma = complex(drude["re"][0], drude["im"][0])
        E = out["energies_eV"][0]
        want = 1j * sigma / (_u.omega_of(E) * rib["W_nm"] * out["eps_eff"])
        got = complex(rib["eta_par"]["re"][0], rib["eta_par"]["im"][0])
        assert got == pytest.approx(want, rel=1e-12), f"at T_K={T_K}"
        assert _np.isfinite(sigma.real) and _np.isfinite(sigma.imag)

    json.dumps(out)


def test_the_two_material_curves_are_not_the_same_sigma_above_the_pauli_edge():
    """And the panel says so. This is the fact the note has to keep matching."""
    spec = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.05, "gamma_eV": 0.02, "L": 31, "T_K": 300.0}],
        "energies_eV": {"min": 0.2, "max": 0.2, "n": 1},
        "q_nm_inv": 0.0,
    }
    out = web.run("material", spec)
    assert out["ok"], out.get("reason")
    mat = out["ribbons"][0]["material"]
    full = complex(mat["sigma_nm_fs"]["re"][0], mat["sigma_nm_fs"]["im"][0])
    drude = complex(mat["sigma_drude_nm_fs"]["re"][0], mat["sigma_drude_nm_fs"]["im"][0])
    # 0.2 eV against a Pauli edge of 0.1 eV: interband is most of the answer.
    assert abs(full - drude) > 0.5 * abs(drude)
    joined = " ".join(out["notes"])
    assert "INTRABAND" in joined


# ===========================================================================
# The extinction cross section (2026-11-05)
# ===========================================================================
def test_the_spectrum_carries_a_real_cross_section():
    """sigma = (4 pi omega / c) Im alpha, the house prefactor, per unit area.

    The same expression the plasmonic-nanoparticles lab prints under its own
    spectrum. alpha is the library's, rho_phys/eps_eff, so the physical dipole
    is eps_eff times it; in vacuum the two coincide. The absorptive term only:
    the 3D scattering term has a different form for a line dipole and is
    negligible for a nanometre ribbon against a micrometre wavelength.
    """
    import numpy as _np

    HBAR, C = 0.6582119569, 299.792458
    spec = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
        "energies_eV": {"min": 0.3, "max": 0.9, "n": 25},
        "q_nm_inv": 0.0,
        "source": {"kind": "plane_wave"},
    }
    out = web.run("linear_spectrum", spec)
    assert out["ok"], out.get("reason")
    assert out["total_width_nm"] == 10.0

    Es = _np.asarray(out["energies_eV"])
    alpha_im = _np.asarray(out["extinction_arb"][0], dtype=float)
    want = 4.0 * _np.pi * (Es / HBAR) / C * alpha_im          # eps_eff = 1 here
    got = _np.asarray(out["sigma_ext_nm"][0], dtype=float)
    assert got == pytest.approx(want, rel=1e-12)
    per_area = _np.asarray(out["sigma_ext_per_area"][0], dtype=float)
    assert per_area == pytest.approx(want / 10.0, rel=1e-12)

    # A resonant graphene ribbon extinguishes a sizeable fraction of its own
    # area -- order 0.1 to a few, never 1e-6 and never 1e6. This is the guard
    # against a prefactor that is right in form and wrong by a unit system.
    assert 0.05 < float(_np.nanmax(per_area)) < 50.0
    json.dumps(out)


def test_the_cross_section_scales_with_eps_eff_not_against_it():
    """alpha_lib = alpha_phys / eps_eff, so undoing it MULTIPLIES.

    Getting this backwards would be invisible in vacuum, where eps_eff = 1,
    and wrong by eps_eff squared everywhere else.
    """
    import numpy as _np

    base = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
        "energies_eV": {"min": 0.3, "max": 0.9, "n": 9},
        "q_nm_inv": 0.0,
        "source": {"kind": "plane_wave"},
    }
    out = web.run("linear_spectrum", dict(base, eps_a=1.0, eps_b=3.0))
    assert out["ok"], out.get("reason")
    eps_eff = 2.0  # (1 + 3) / 2
    Es = _np.asarray(out["energies_eV"])
    want = (
        4.0 * _np.pi * (Es / 0.6582119569) / 299.792458
        * eps_eff
        * _np.asarray(out["extinction_arb"][0], dtype=float)
    )
    assert _np.asarray(out["sigma_ext_nm"][0], dtype=float) == pytest.approx(want, rel=1e-12)


def test_the_cross_section_covers_every_q_row():
    """One row per q, like every other array in the payload."""
    spec = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
        "energies_eV": {"min": 0.3, "max": 0.9, "n": 6},
        "qs": {"min": 0.02, "max": 0.2, "n_pos": 2},
        "source": {"kind": "plane_wave"},
    }
    out = web.run("linear_spectrum", spec)
    assert out["ok"], out.get("reason")
    assert len(out["sigma_ext_nm"]) == len(out["extinction_arb"])
    assert len(out["sigma_ext_nm"][0]) == len(out["energies_eV"])


def test_the_fast_path_obstructions_read_as_english_not_as_symbols():
    """These strings go on the page, verbatim, under "No fast path: ...".

    They used to end "so A is not frequency independent", where A is an
    internal name for the geometry matrix V @ D_geom that nothing the reader
    can see ever defines. A user asked what A meant, which is the whole bug: a
    message written for whoever wrote the solver rather than for whoever is
    looking at the page. Each one now names its cause in words.
    """
    import re as _re

    from electrostatics_fd.solver import Ribbon as _R, RibbonSystem as _S
    from electrostatics_fd.substrate import Layer

    cases = [
        (_S([_R(W_nm=10.0, L=31, beta=0.3)]), "tensor"),
        (_S([_R(W_nm=10.0, L=31), _R(W_nm=12.0, x0_nm=40.0, L=31)]), "interchangeable"),
        (_S([_R(W_nm=10.0, z0_nm=1.0, L=31)], layers=[Layer(eps_par=1.0), Layer(eps_par=4.0)]), "substrate"),
    ]
    for system, word in cases:
        msg = system.fast_path_obstruction()
        assert msg is not None
        assert word in msg, msg
        # No bare one-letter matrix names. The lowercase article "a" is fine.
        assert not _re.search(r"[A-Z]", msg), msg
        assert len(msg.split()) >= 8, "a user-facing reason needs a sentence"


def test_the_electron_loss_is_a_loss_and_never_negative():
    """A loss probability below zero is not a small error, it is the wrong quantity.

    Until 2026-11-05 this view reported ``linear_spectrum``'s alpha -- the
    induced DIPOLE under the electron's potential. Under a plane wave that IS
    the right thing, because phi_ext = -E0 x makes the overlap the dipole; the
    electron's potential is not proportional to x, so the same array came out
    with the wrong phase and swung through zero at the resonance. A user saw
    negative loss on a bare ribbon.

    What it reports now is the reciprocity form: minus the imaginary part of
    the induced charge against the potential that created it, i.e. the work
    the electron does against the field it induces. Positive by construction.
    """
    import numpy as _np

    spec = {
        "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
        "energies_eV": {"min": 0.3, "max": 0.9, "n": 41},
        "q_nm_inv": 0.05,
        "source": {"kind": "electron", "v_over_c": 0.2, "b_nm": 5.0},
    }
    out = web.run("eels_spectrum", spec)
    assert out["ok"], out.get("reason")

    loss = _np.asarray(out["loss_arb"][0], dtype=float)
    assert _np.all(_np.isfinite(loss))
    assert _np.all(loss >= 0.0), f"negative loss: min {loss.min()}"

    # And it is a resonance, not a monotone background: the peak sits on the
    # ribbon plasmon, well inside the window, and stands over its own edges.
    Es = _np.asarray(out["energies_eV"])
    peak = int(_np.argmax(loss))
    assert 0 < peak < len(Es) - 1
    assert 0.5 < Es[peak] < 0.65
    # Over its own edges by a factor of 4 on this grid. Not more: the
    # electron's potential is broadband, so the low-energy side carries a real
    # off-resonance tail (0.045 against a 0.187 peak here) rather than falling
    # to nothing the way a plane-wave cross section does.
    assert loss[peak] > 3.0 * max(loss[0], loss[-1])

    # The dipole is gone from this payload: it was never the loss.
    assert "extinction_arb" not in out
    json.dumps(out)


def test_the_loss_grows_as_the_electron_comes_closer():
    """An aloof electron couples more strongly the nearer it passes."""
    import numpy as _np

    def peak(b_nm: float) -> float:
        spec = {
            "ribbons": [{"W_nm": 10.0, "EF_eV": 0.5, "gamma_eV": 0.02, "L": 61}],
            "energies_eV": {"min": 0.4, "max": 0.8, "n": 25},
            "q_nm_inv": 0.05,
            "source": {"kind": "electron", "v_over_c": 0.2, "b_nm": b_nm},
        }
        out = web.run("eels_spectrum", spec)
        assert out["ok"], out.get("reason")
        return float(_np.max(_np.asarray(out["loss_arb"][0], dtype=float)))

    assert peak(3.0) > peak(6.0) > peak(12.0)


@pytest.mark.parametrize(
    "W, EF, beta, n_rib, qmax, n_pos",
    [
        (12.0, 0.5, 0.26, 1, 1.0, 16),      # the panel in the bug report
        (12.0, 0.5, 0.00, 1, 1.0, 16),
        (12.0, 0.5, 0.00, 2, 1.0, 16),
        (30.0, 0.9, 0.00, 1, 1.0, 20),
        (6.0, 0.3, 0.00, 1, 0.5, 16),
        (8.0, 0.4, 0.40, 3, 1.0, 16),
    ],
)
def test_ldos_peak_tracks_are_continuous_on_a_real_map(W, EF, beta, n_rib, qmax, n_pos):
    """No slot may step a large fraction of the plotted window in one column.

    Parametrised over drift, ribbon count, width and Fermi level because the
    previous round of this bug was fixed against ONE parameter set and two
    separate defects hid in the two that were never varied.
    """
    E_lo, E_hi = 0.1, 1.5
    spec = {
        "ribbons": [
            {"W_nm": W, "x0_nm": 40.0 * j, "z0_nm": 0.0, "EF_eV": EF,
             "gamma_eV": 0.02, "beta": beta, "T_K": 0.0, "L": 81}
            for j in range(n_rib)
        ],
        "eps_a": 1.0, "eps_b": 1.0, "T_K": 0.0,
        "energies_eV": {"min": E_lo, "max": E_hi, "n": 120},
        "qs_nm_inv": {"min": 0.02, "max": qmax, "n_pos": n_pos},
        "source": {"kind": "dipole", "r0_nm": [0.0, 0.0, 5.0], "p": [1.0, 0.0, 0.0]},
        "material": {"EF_eV": EF, "gamma_eV": 0.02, "beta": beta, "T_K": 0.0},
        "n_peaks": 3,
    }
    res = web.ldos(spec)
    E = np.array([[np.nan if v is None else v for v in row]
                  for row in res["peak_E_eV"]], dtype=float)
    steps = np.abs(np.diff(E, axis=0))
    worst = float(np.nanmax(steps)) if np.any(np.isfinite(steps)) else 0.0
    window = E_hi - E_lo
    # Before tracking this reached 75% of the window on the first case here.
    assert worst < 0.15 * window, (
        f"a peak track stepped {worst:.4f} eV, {100 * worst / window:.1f}% of the "
        f"plotted window, between adjacent q columns"
    )
