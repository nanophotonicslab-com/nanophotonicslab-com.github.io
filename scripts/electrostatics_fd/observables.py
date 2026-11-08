"""Observables derived from the induced charge of a quasistatic ribbon system.

Induced field of a 1D charge profile at fixed guided wave vector q, the
electron-energy-loss / photonic local density of states at that q, and the
circular-dissymmetry ratio of a q-resolved loss spectrum.

Source equations
----------------
* induced field                 the reference's own ``set_Eq`` form, the
                                y-Fourier transform of the Coulomb field of a
                                line-charge distribution:
                                    1/|r-r'|  ->  2 K_0(|q| s)
                                equivalently paper 456 Eqs. (14), (17), (19)
                                and (21).
* ldos_q                        paper 456, LDOS from the induced field,
                                    rho_ind = Im(p* . E) / (2 pi^2 |p|^2)
* dissymmetry                   paper 456, Eq. (2)

=============================================================================
NORMALISATION -- RESOLVED (2026-11-03); THE ABSOLUTE SCALE IS TRUSTWORTHY
=============================================================================
The reference's ``set_Eq`` form, implemented by :func:`induced_field_q`,

    s        = sqrt((x0 - x')^2 + z0^2)
    E_q,x    = +2|q| \\int dx' rho_q(x') (x0 - x')/s K_1(|q| s)
    E_q,y    = -2i q  \\int dx' rho_q(x')                K_0(|q| s)
    E_q,z    = +2|q| \\int dx' rho_q(x') (z0/s)          K_1(|q| s)

is CORRECT to an absolute -- not merely relative -- normalisation.  It was
confirmed four independent ways:

1. it is the exact negative gradient ``-(d/dx0, i q, d/dz0)`` of the potential
   the already-validated Coulomb operator ``operators.build_V`` produces from
   the same rho (measured ratio 1.0000000002; reproduced here to ~4e-12 by
   ``test_v_gradient_consistency_x`` / ``_z`` in ``tests/test_observables.py``);
2. absolute energy conservation between a dipole source and the Ohmic loss in
   the ribbon closes to 1.000000 in six configurations (reproduced here by
   ``test_energy_conservation_dipole_vs_ohmic_loss``, ratio 1.0 - 4e-13);
3. it reduces to the textbook 2D line-charge field ``E = 2 lambda / r`` as
   ``q -> 0``;
4. it reproduces paper 456 Eqs. (14) + (17) + (19) + (21) exactly.

The factor -1/(2 pi) against the reference implementation's inline
``set_Eind`` is a CONVENTION ARTIFACT, not an error in
either routine.  ``set_Eind`` folds the Fourier-inversion measure ``dq/(2 pi)``
of paper 456 Eq. (21) into the per-q field AND flips the sign; every MATLAB
caller compensates for both (they form ``-imag(p . Eind)`` and then trapz over
a bare ``dq``), so the published pipeline lands on the right answer through a
per-q object that simply is not ``E_q``.  This library follows Eq. (1)/(21) and
keeps the ``set_Eq_Riboons`` form, so:

* ABSOLUTE LDOS magnitudes are NOT uncertain by -1/(2 pi).  Trust them.
* A caller that reconstructs a real-space quantity from these per-q objects
  must supply the inversion measure itself: paper 456 Eq. (21) integrates
  ``dq/(2 pi)`` (``int dq/(2 pi)`` over the signed q axis), and that 1/(2 pi)
  belongs there -- in the caller's q quadrature -- and nowhere inside this
  module.  :func:`dissymmetry` is a ratio of such q integrals, so the measure
  cancels out of it entirely.

Also settled: ``solver``'s ``rho = D phi / W`` already equals the physical
induced charge density divided by eps_eff (derivation in ``solver.py``), and
the Coulomb kernel ``2 K_0`` carries no eps_eff either, so
:func:`induced_field_q` applied to the library's rho IS ``E_q`` *including*
substrate/ambient screening.  No extra eps_eff belongs anywhere in this
module.

=============================================================================
QUADRATURE: UNIFORM CELL WEIGHTS, NOT TRAPEZOIDAL  (bug fixed 2026-11-03)
=============================================================================
The ribbon-node integral is a UNIFORM CELL sum, full weight ``dx`` on every
node including both end nodes,

    E_q,x = 2|q| dx sum_l' rho_l' (x0 - x_l')/s_l' K_1(|q| s_l'),
    dx    = h_j * W_j   (physical node spacing of ribbon j)

exactly as ``operators.build_V`` weights the *same* spatial sum: its branches
integrate ``2 K_0`` over a cell ``[theta - h/2, theta + h/2]`` centred on each
node, giving ``2 h K_0`` at the midpoint level, and ``solver`` likewise uses
``W_j h_j sum_l`` for the dipole moment.  The field returned here is therefore
the exact discrete gradient of the discrete potential the operator builds, and
the two can be compared without a fudge factor.

Until 2026-11-03 this module used ``np.trapz`` over the nodes, which gives the
two end nodes HALF weight and so is inconsistent with ``build_V``.  Because the
physical induced charge density diverges at a ribbon edge, the end nodes carry
a large share of the integral and the error was not small: measured on the
fundamental eigenmode of a 100 nm ribbon at L = 301 the trapezoidal field was
3.6% to 30% low depending on the observation point (6.3% at q = 0.05 nm^-1,
r0 = (200, 10) nm), and it converged only as sqrt(dx) instead of at second
order.  ``tests/test_observables.py`` pins both the gradient consistency and
the convergence order so the trapezoidal form cannot be reinstated silently.

The three trapezoidal calls in :func:`dissymmetry` are a DIFFERENT integral --
the Fourier-inversion quadrature over the q grid of paper 456 Eq. (21), where
the grid is the caller's and trapezoidal is the right rule.  They stay.
=============================================================================
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import k0, k1

__all__ = ["induced_field_q", "ldos_q", "dissymmetry"]

# numpy >= 2 renamed trapz -> trapezoid; pick whichever this interpreter has.
# Same shim as ``nonlinear.py``, and it is not cosmetic: the browser runs this
# library under Pyodide's numpy 2.0, where the bare ``np.trapz`` still works but
# emits a DeprecationWarning -- and the web boundary surfaces captured warnings
# to the user as result caveats, so a deprecation notice was being shown on the
# page beside the physics. It will stop working entirely on some later numpy.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

_ArrayLike = Sequence[float] | np.ndarray


def _as_ribbon_list(x_nodes, rho) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Accept either single arrays or per-ribbon lists of arrays."""
    if isinstance(x_nodes, (list, tuple)):
        xs = [np.asarray(x, dtype=np.float64).ravel() for x in x_nodes]
    else:
        arr = np.asarray(x_nodes, dtype=np.float64)
        xs = [arr.ravel()] if arr.ndim == 1 else [np.asarray(r).ravel() for r in arr]
    if isinstance(rho, (list, tuple)):
        rs = [np.asarray(r, dtype=np.complex128).ravel() for r in rho]
    else:
        arr = np.asarray(rho, dtype=np.complex128)
        rs = [arr.ravel()] if arr.ndim == 1 else [np.asarray(r).ravel() for r in arr]
    if len(xs) != len(rs):
        raise ValueError(
            f"x_nodes has {len(xs)} ribbon block(s) but rho has {len(rs)}"
        )
    for i, (x, r) in enumerate(zip(xs, rs)):
        if x.shape != r.shape:
            raise ValueError(
                f"block {i}: x_nodes shape {x.shape} != rho shape {r.shape}"
            )
    return xs, rs


def _cell_widths(
    xs: list[np.ndarray], dx_nm: float | _ArrayLike | None
) -> list[float]:
    """Physical cell width ``dx = h_j W_j`` of each ribbon block.

    ``None`` infers it from the node spacing, which is what ``solver`` hands
    over (``x_nodes[j] = x0_j + W_j * linspace(-1/2, 1/2, L_j)``, so
    ``diff(x) == h_j W_j`` exactly).  The grid must be uniform, because a
    uniform cell weight is only the right quadrature on a uniform grid; a
    non-uniform grid or a single-node block has to state ``dx_nm`` explicitly.
    """
    n = len(xs)
    if dx_nm is None:
        widths: list[float] = []
        for i, x in enumerate(xs):
            if x.size < 2:
                raise ValueError(
                    f"block {i} has {x.size} node(s), so the cell width cannot be "
                    "inferred from the node spacing; pass dx_nm explicitly"
                )
            d = np.diff(x)
            if not np.allclose(d, d[0], rtol=1e-9, atol=0.0) or d[0] <= 0.0:
                raise ValueError(
                    f"block {i}: x_nodes must be an increasing uniform grid for the "
                    f"uniform cell quadrature; got spacings in "
                    f"[{d.min()!r}, {d.max()!r}]. Pass dx_nm to override."
                )
            widths.append(float(d[0]))
        return widths

    arr = np.asarray(dx_nm, dtype=np.float64).ravel()
    if arr.size == 1:
        arr = np.repeat(arr, n)
    if arr.size != n:
        raise ValueError(
            f"dx_nm has {arr.size} entries but there are {n} ribbon block(s)"
        )
    if np.any(arr <= 0.0):
        raise ValueError(f"dx_nm must be positive, got {dx_nm!r}")
    return [float(v) for v in arr]


def _unpack_r0(r0_nm: _ArrayLike) -> tuple[float, float]:
    """Return (x0, z0) [nm] from a 2- or 3-component position.

    A 3-component r0 = (x0, y0, z0) is accepted for convenience; y0 is ignored
    because the field is evaluated at fixed guided wave vector q, where the
    y dependence is the factor exp(i q y0) that the caller supplies through
    phi_ext instead.
    """
    r0 = np.asarray(r0_nm, dtype=np.float64).ravel()
    if r0.size == 2:
        return float(r0[0]), float(r0[1])
    if r0.size == 3:
        return float(r0[0]), float(r0[2])
    raise ValueError(f"r0_nm must have 2 (x0, z0) or 3 (x0, y0, z0) entries, got {r0.size}")


def induced_field_q(
    q: float,
    x_nodes,
    rho,
    r0_nm: _ArrayLike,
    *,
    dx_nm: float | _ArrayLike | None = None,
) -> tuple[complex, complex, complex]:
    """Induced electric field at (x0, z0) from a 1D charge profile at wave vector q.

    Implements the reference's ``set_Eq`` form, whose absolute normalisation is
    resolved and correct (module docstring):

        s  = sqrt((x0 - x')^2 + z0^2)
        Ex = 2|q| \\int rho(x') (x0 - x')/s K_1(|q| s) dx'
        Ey = -2i q \\int rho(x')                K_0(|q| s) dx'
        Ez = 2|q| \\int rho(x')   z0/s          K_1(|q| s) dx'

    The node integral uses UNIFORM CELL WEIGHTS -- full weight ``dx`` on every
    node, both end nodes included -- which is the quadrature
    ``operators.build_V`` applies to the same spatial sum, so the returned field
    is the exact discrete gradient of the discrete potential that operator
    builds.  See the module docstring for the trapezoidal form this replaced and
    the size of the error it carried.

    This is exactly linear in ``rho``, which is what lets ``dispersion._field_map``
    build the whole ``(3, N)`` field map one basis vector at a time.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].  Ex and Ez depend on |q| only; Ey is odd in q.
    x_nodes
        Node positions [nm], a single array or a list of per-ribbon arrays.
        Each block must be an increasing uniform grid unless ``dx_nm`` is given.
    rho
        Induced charge density at those nodes (``solver`` convention,
        ``rho = D phi / W``), same layout as ``x_nodes``.  Contributions from
        several ribbons are summed.
    r0_nm
        Observation point (x0, z0) or (x0, y0, z0) [nm].
    dx_nm
        Keyword-only.  Cell width(s) [nm]: a scalar for every block or one per
        block.  ``None`` (the default, and the behaviour every existing caller
        gets) infers ``dx = h_j W_j`` from the node spacing.  Pass it for a
        single-node block -- a bare point charge of weight ``rho`` needs
        ``dx_nm=1.0`` -- or for a deliberately non-uniform grid.

    Returns
    -------
    (Ex, Ey, Ez)
        Complex field components at the observation point.

    Raises
    ------
    ValueError
        If the observation point coincides with a node (s = 0), where K_0/K_1
        diverge; or if a block's cell width can be neither inferred nor read
        from ``dx_nm``.
    """
    xs, rs = _as_ribbon_list(x_nodes, rho)
    cells = _cell_widths(xs, dx_nm)
    x0, z0 = _unpack_r0(r0_nm)
    aq = abs(float(q))
    ex = ey = ez = 0.0 + 0.0j
    for x, r, cell in zip(xs, rs, cells):
        # ``sep`` is the in-plane separation x0 - x'; ``cell`` is the quadrature
        # weight dx = h_j W_j.  Both are lengths in nm and are easy to confuse.
        sep = x0 - x
        s = np.sqrt(sep * sep + z0 * z0)
        if np.any(s == 0.0):
            raise ValueError(
                f"observation point (x0={x0}, z0={z0}) coincides with a source "
                "node; the Coulomb kernel diverges there"
            )
        qs = aq * s
        k0v = k0(qs)
        k1v = k1(qs)
        ex += 2.0 * aq * cell * np.sum(r * sep / s * k1v)
        ey += -2.0j * float(q) * cell * np.sum(r * k0v)
        ez += 2.0 * aq * cell * np.sum(r * (z0 / s) * k1v)
    return complex(ex), complex(ey), complex(ez)


def ldos_q(
    q: float,
    x_nodes,
    rho,
    r0_nm: _ArrayLike,
    p: _ArrayLike,
    *,
    dx_nm: float | _ArrayLike | None = None,
) -> float:
    """q-resolved local density of states / loss probability density (paper 456).

        D(q) = Im( conj(p) . E_q ) / (2 pi^2 |p|^2)

    with E_q from :func:`induced_field_q`.  The absolute scale is trustworthy:
    the -1/(2 pi) discrepancy against the reference ``set_Eind`` is a convention
    artifact of that routine, not an uncertainty here (module docstring).  Note
    that ``2 pi^2 |p|^2`` is the paper's per-q normalisation only; a caller
    rebuilding the real-space loss probability still owes the ``dq/(2 pi)``
    inversion measure of paper 456 Eq. (21).

    Since the quantity is ``Im(conj(p) . E_q)``, it is proportional to the Ohmic
    power the ribbon dissipates: ``(omega/2) Im(conj(p) . E_q)`` equals that
    power exactly (see ``test_energy_conservation_dipole_vs_ohmic_loss``), so
    ``D(q) > 0`` for a passive ribbon driven by a real dipole.

    Parameters
    ----------
    q
        Guided wave vector [1/nm].
    x_nodes, rho
        Node positions [nm] and induced charge density, as in
        :func:`induced_field_q`.
    r0_nm
        Dipole (observation) position (x0, z0) or (x0, y0, z0) [nm].
    p
        Dipole moment, 3 components (px, py, pz); may be complex (e.g. a
        circular dipole (1, i, 0)/sqrt(2)).
    dx_nm
        Keyword-only; forwarded to :func:`induced_field_q`.

    Returns
    -------
    float
        The (real) q-resolved LDOS contribution.
    """
    pv = np.asarray(p, dtype=np.complex128).ravel()
    if pv.size != 3:
        raise ValueError(f"p must have 3 components (px, py, pz), got {pv.size}")
    p2 = float(np.vdot(pv, pv).real)
    if p2 == 0.0:
        raise ValueError("p must be a non-zero dipole moment")
    e_vec = np.asarray(
        induced_field_q(q, x_nodes, rho, r0_nm, dx_nm=dx_nm), dtype=np.complex128
    )
    # np.vdot(pv, e_vec) == sum(conj(p) * E)
    return float(np.vdot(pv, e_vec).imag) / (2.0 * np.pi**2 * p2)


def dissymmetry(qs: _ArrayLike, Dq: _ArrayLike) -> float:
    """Circular dissymmetry of a q-resolved spectrum (paper 456, Eq. 2).

        g = 2 ( \\int_{q>0} D(q) dq - \\int_{q<0} D(q) dq ) / \\int D(q) dq

    Both half-integrals and the total are trapezoidal on the supplied grid --
    deliberately so, and unaffected by the cell-weight fix in
    :func:`induced_field_q`.  This is the Fourier-inversion quadrature over the
    caller's q grid (paper 456 Eq. 21), a genuine integral of a smooth spectrum
    between its stated end points, where trapezoidal is the correct rule; the
    ribbon-node sums of :func:`induced_field_q` are a different integral, over
    finite-difference cells, and take uniform cell weights instead.  The
    ``1/(2 pi)`` of Eq. (21) cancels out of this ratio, so none appears here.
    If
    q = 0 is not a grid node, a node is inserted there with the linearly
    interpolated D value, so that the two half-integrals still add up exactly to
    the total (trapezoidal integration is exact for that insertion).

    g vanishes identically for any spectrum that is even in q -- in particular
    for an undrifted (beta = 0) system driven by a source whose external
    potential is even in q -- and is nonzero only when the response itself
    breaks q -> -q parity.

    Parameters
    ----------
    qs
        Strictly increasing wave-vector grid [1/nm], spanning both signs.
    Dq
        Spectrum sampled on ``qs`` (real).

    Returns
    -------
    float
        The dissymmetry ratio.
    """
    q = np.asarray(qs, dtype=np.float64).ravel()
    d = np.asarray(Dq, dtype=np.float64).ravel()
    if q.shape != d.shape:
        raise ValueError(f"qs shape {q.shape} != Dq shape {d.shape}")
    if q.size < 3:
        raise ValueError("need at least 3 q points")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("qs must be strictly increasing")
    if not (q[0] < 0.0 < q[-1]):
        raise ValueError(f"qs must straddle q = 0, got [{q[0]}, {q[-1]}]")

    if not np.any(q == 0.0):
        d0 = float(np.interp(0.0, q, d))
        i = int(np.searchsorted(q, 0.0))
        q = np.insert(q, i, 0.0)
        d = np.insert(d, i, d0)

    pos = q >= 0.0
    neg = q <= 0.0
    i_pos = _TRAPZ(d[pos], q[pos])
    i_neg = _TRAPZ(d[neg], q[neg])
    i_tot = _TRAPZ(d, q)
    if i_tot == 0.0:
        raise ValueError(
            "total integral of Dq vanishes; the dissymmetry ratio is undefined"
        )
    return float(2.0 * (i_pos - i_neg) / i_tot)
