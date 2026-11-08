"""Quasistatic (electrostatic) finite-difference simulation of graphene nanoribbons.

Reference implementation in the Gaussian eV-nm-fs unit system: energies in eV,
lengths in nm, times in fs, 2D conductivities in nm/fs.  See ``units`` for the
constants, ``conductivity`` for the sheet-conductivity models, and ``README.md``
for the module map, what is verified, and what is still unproven.

The whole public surface is re-exported here, so both import styles work::

    from electrostatics_fd import Ribbon, RibbonSystem
    from electrostatics_fd.solver import RibbonSystem

Every name in every sub-module's ``__all__`` is reachable from the package root,
and ``tests/test_integration.py::test_package_reexports_every_submodule_all``
enforces that rather than trusting this sentence.  ``substrate.HBAR_EV_FS`` is
reached through ``units``, which defines the same value (the two are asserted
bit-identical in ``tests/test_substrate.py``).

ONE CAVEAT, because a re-export cannot hide it.  The module-level tuning
constants -- ``solver.VBAR_CACHE_MAX``, ``solver.SIGNED_Q_IN_DRIFT_SIGMA``,
``solver.ZETA_NORMALISATION``, ``dispersion.SCREEN_REFERENCE_EV`` -- are bound
here by VALUE at import time.  Reading them from the package root is fine;
CHANGING one must be done on the sub-module (``solver.VBAR_CACHE_MAX = 4``),
because rebinding ``electrostatics_fd.VBAR_CACHE_MAX`` updates the alias and
nothing that reads it.

Sub-modules are imported eagerly: they are small, they all import cleanly, and
an ``ImportError`` at package import is far easier to diagnose than one raised
halfway through a sweep.
"""

from __future__ import annotations

from . import (
    conductivity,
    dispersion,
    nonlinear,
    observables,
    operators,
    solver,
    sources,
    substrate,
    units,
)
from .conductivity import (
    SIGMA0_NM_FS,
    drift_chemical_potential,
    drift_chemical_potential_numeric,
    graphene_drift_sigma,
    graphene_drude,
    graphene_interband_T0,
    graphene_local_rpa,
    metal_thin_film,
    sigma_cascade_abcd,
    sigma_kerr,
    sigma_shg_1d_q0,
    sigma_shg_abc,
    sigma_thg_local,
    thermal_drude_weight,
)
from .dispersion import (
    SCREEN_REFERENCE_EV,
    BandStructure,
    drude_band_energy,
    extended_sheet_eta,
    ldos_map,
    pm_asymmetry,
    ribbon_bands,
    sheet_plasmon_energy,
    sheet_reflection,
    sheet_reflection_sigma,
    sheet_response,
    sheet_sigma_longitudinal,
    symmetric_q_grid,
    follow_peaks,
    track_peaks,
)
from .nonlinear import (
    CASCADE_A_CONDUCTIVITY,
    CASCADE_A_REFERENCE,
    HBAR_EV_FS_FALLBACK,
    polarizability_from_rho,
    shg_source_rho,
    solve_shg,
    solve_thg,
    thg_source_rho,
)
from .observables import dissymmetry, induced_field_q, ldos_q
from .operators import (
    build_D,
    build_D_geom,
    build_V,
    build_V_toeplitz,
    coulomb_V_entries,
    struve_L,
)
from .solver import (
    SIGNED_Q_IN_DRIFT_SIGMA,
    VBAR_CACHE_MAX,
    ZETA_NORMALISATION,
    Ribbon,
    RibbonSystem,
    bright_mode_mask,
)
from .sources import (
    e_ext_electron,
    electron_decay_constant,
    electron_trajectory_is_valid,
    lorentz_gamma,
    phi_ext_dipole,
    phi_ext_dipole_reflected,
    phi_ext_electron,
    phi_ext_electron_reflected,
    phi_ext_plane_wave,
)
from .substrate import Layer, build_Vbar, default_Qmax, reflection_coefficient
from .units import (
    AU_EV,
    AU_NM,
    C_NM_FS,
    E2_EV_NM,
    HBAR_EV_FS,
    KB_EV_K,
    SIGMA_AU_TO_NM_FS,
    VF_NM_FS,
    omega_of,
    sigma_au_to_nm_fs,
    sigma_nm_fs_to_au,
)

__all__ = [
    # sub-modules
    "conductivity",
    "dispersion",
    "nonlinear",
    "observables",
    "operators",
    "solver",
    "sources",
    "substrate",
    "units",
    # units
    "HBAR_EV_FS",
    "E2_EV_NM",
    "C_NM_FS",
    "VF_NM_FS",
    "KB_EV_K",
    "AU_NM",
    "AU_EV",
    "SIGMA_AU_TO_NM_FS",
    "omega_of",
    "sigma_au_to_nm_fs",
    "sigma_nm_fs_to_au",
    # conductivity
    "SIGMA0_NM_FS",
    "graphene_drude",
    "graphene_interband_T0",
    "graphene_local_rpa",
    "thermal_drude_weight",
    "drift_chemical_potential",
    "drift_chemical_potential_numeric",
    "graphene_drift_sigma",
    "metal_thin_film",
    "sigma_shg_abc",
    "sigma_shg_1d_q0",
    "sigma_thg_local",
    "sigma_kerr",
    "sigma_cascade_abcd",
    # operators
    "struve_L",
    "coulomb_V_entries",
    "build_V",
    "build_V_toeplitz",
    "build_D",
    "build_D_geom",
    # substrate
    "Layer",
    "reflection_coefficient",
    "build_Vbar",
    "default_Qmax",
    # sources
    "phi_ext_plane_wave",
    "phi_ext_dipole",
    "phi_ext_dipole_reflected",
    "phi_ext_electron",
    "phi_ext_electron_reflected",
    "e_ext_electron",
    "electron_trajectory_is_valid",
    "lorentz_gamma",
    "electron_decay_constant",
    # solver / observables
    "Ribbon",
    "RibbonSystem",
    "bright_mode_mask",
    "SIGNED_Q_IN_DRIFT_SIGMA",
    "VBAR_CACHE_MAX",
    "ZETA_NORMALISATION",
    "induced_field_q",
    "ldos_q",
    "dissymmetry",
    # nonlinear
    "shg_source_rho",
    "thg_source_rho",
    "solve_shg",
    "solve_thg",
    "polarizability_from_rho",
    "CASCADE_A_REFERENCE",
    "CASCADE_A_CONDUCTIVITY",
    "HBAR_EV_FS_FALLBACK",
    # dispersion: the Level 1 (extended sheet) / Level 2 (ribbon bands) /
    # Level 3 (LDOS map) stack
    "sheet_sigma_longitudinal",
    "sheet_response",
    "sheet_reflection_sigma",
    "sheet_reflection",
    "sheet_plasmon_energy",
    "extended_sheet_eta",
    "drude_band_energy",
    "BandStructure",
    "ribbon_bands",
    "SCREEN_REFERENCE_EV",
    "ldos_map",
    "track_peaks",
    "follow_peaks",
    "symmetric_q_grid",
    "pm_asymmetry",
]
