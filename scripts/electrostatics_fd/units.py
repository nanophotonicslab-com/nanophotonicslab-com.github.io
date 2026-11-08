"""Unit system and conversions for the quasistatic finite-difference solver.

Gaussian eV-nm-fs convention used throughout the package:

* energies in eV, lengths in nm, times in fs,
* ``E2_EV_NM`` is e^2 in Gaussian units (i.e. e^2/(4*pi*eps0) in SI), an energy
  times a length,
* angular frequencies in 1/fs, ``omega = E / hbar``,
* 2D (sheet) conductivities in nm/fs.

With those choices a Drude sheet conductivity reads
``sigma = 1j * e^2 * E_F / (pi * hbar * (E + 1j * gamma))``  [nm/fs]
(Cox, Yu & Garcia de Abajo, Phys. Rev. B 96, 045442 (2017), Eq. 8), and the
atomic-unit conductivity of a 2D sheet maps onto nm/fs by the single factor
``SIGMA_AU_TO_NM_FS = e^2 / hbar``.
"""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
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
]

#: Reduced Planck constant [eV fs].
HBAR_EV_FS: float = 0.6582119569
#: Squared electron charge in Gaussian units, e^2 = e^2/(4 pi eps0)|_SI [eV nm].
E2_EV_NM: float = 1.439964
#: Speed of light [nm/fs].
C_NM_FS: float = 299.792458
#: Graphene Fermi velocity, c/300 [nm/fs].
VF_NM_FS: float = C_NM_FS / 300.0
#: Boltzmann constant [eV/K].
KB_EV_K: float = 8.617333262e-5
#: Bohr radius [nm].
AU_NM: float = 0.05291772083
#: Hartree energy [eV].
AU_EV: float = 27.2113834
#: Multiply an atomic-unit 2D conductivity by this to get nm/fs (= e^2/hbar).
SIGMA_AU_TO_NM_FS: float = E2_EV_NM / HBAR_EV_FS

ArrayLike = Union[float, complex, np.ndarray]


def _unwrap(arr: np.ndarray) -> Union[np.generic, np.ndarray]:
    """Return a numpy scalar for a 0-d result and the array otherwise.

    Keeps ``f(scalar) -> scalar`` / ``f(array) -> array`` behaviour without
    branching on the input type in every public function.  ``np.float64`` and
    ``np.complex128`` subclass ``float`` and ``complex``, so the scalar return
    satisfies the declared ``float`` / ``complex`` types.
    """
    return arr if arr.ndim else arr[()]


def omega_of(E_ev: ArrayLike) -> Union[float, np.ndarray]:
    """Angular frequency from photon energy, ``omega = E / hbar``  [1/fs].

    Definition of the eV-nm-fs unit system (see module docstring); used wherever
    the nonlinear conductivities of Cox, Yu & Garcia de Abajo, Phys. Rev. B 96,
    045442 (2017), Eqs. (10)-(15) need omega rather than E.
    """
    return _unwrap(np.asarray(E_ev) / HBAR_EV_FS)


def sigma_au_to_nm_fs(sigma_au: ArrayLike) -> Union[complex, np.ndarray]:
    """Convert a 2D conductivity from atomic units to nm/fs.

    A 2D conductivity has dimensions of a velocity, so the conversion is the
    atomic unit of velocity ``e^2/hbar`` expressed in nm/fs, i.e.
    ``SIGMA_AU_TO_NM_FS = E2_EV_NM / HBAR_EV_FS`` (= 2.187691... nm/fs).
    """
    return _unwrap(np.asarray(sigma_au, dtype=np.complex128) * SIGMA_AU_TO_NM_FS)


def sigma_nm_fs_to_au(sigma: ArrayLike) -> Union[complex, np.ndarray]:
    """Convert a 2D conductivity from nm/fs to atomic units.

    Inverse of :func:`sigma_au_to_nm_fs`.
    """
    return _unwrap(np.asarray(sigma, dtype=np.complex128) / SIGMA_AU_TO_NM_FS)
