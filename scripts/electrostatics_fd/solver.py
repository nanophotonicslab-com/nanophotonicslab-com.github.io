"""Self-consistent quasistatic (electrostatic) solver for graphene nanoribbons.

Assembles the finite-difference block system for one or several zero-thickness
ribbons that are infinite along y and carry a guided wave vector q, solves the
linear response, extracts the geometric eigenmodes used for dispersion and for
the frequency-sweep fast path, and returns polarizabilities.

Source equations
----------------
* Eigenmode / eta formulation, ribbon RIBBON_ETA_J reference values:
  T. Christensen-style quasistatic eigenvalue formulation as used in
  Cox, Yu and Garcia de Abajo, Phys. Rev. B 96, 045442 (2017), Table S2
  (NPLab ``nl-graphene.ts`` RIBBON_ETA_J).
* Drift-induced anisotropic conductivity: paper 456.
* Coulomb and finite-difference operators: ``operators.py``.

Physical derivation of the assembled system
-------------------------------------------
Charge continuity with a local anisotropic sheet conductivity
sigma = diag(sigma_perp, sigma_par) (perp = across the ribbon, x; par = along
the ribbon, y), for a field e^{i q y - i omega t} and x = x_j + W_j theta:

    rho = div(j)/(-i omega) with j = -sigma grad(phi)
        => W rho = eps_eff [ eta_perp d^2/dtheta^2 - eta_par (qW)^2 ] phi
                 = eps_eff D phi
    with eta_perp = i sigma_perp / (omega W eps_eff)   (dimensionless)
         eta_par  = i sigma_par  / (omega W eps_eff)

and the y-Fourier transformed Coulomb law 1/|r-r'| -> 2 K_0(|q| s):

    phi_ind(theta_l) = (2/eps_eff) sum_{jp} h_jp sum_{l'} K_0(...) (W_jp rho_jp)
                     = sum_{jp} V(j,jp) D(jp) phi_jp = (M phi)_j

so the eps_eff factors cancel and the self-consistent system is exactly the one
the contract specifies,

    (1 - M) phi = phi_ext,     M[j,jp] = V(j,jp) @ D(jp),    rho_j = D(j) phi_j / W_j,

with ``V = 2 h K_0`` carrying no eps_eff (``operators.build_V`` takes no eps
argument) and eps_eff living inside eta.  Note the consequence for the sign of
rho: the contract's ``rho = D phi / W`` is the PHYSICAL induced charge density
divided by eps_eff.  For eps_a = eps_b = 1 they are identical.  Everything
downstream of rho (``observables``, ``polarizability``) inherits that 1/eps_eff.

CONTRACT DEVIATIONS AND CONCERNS -- read before trusting a number
-----------------------------------------------------------------
1. Mode ordering.  The contract says "sorted by DESCENDING eta (i.e. the least
   negative first, which is the fundamental dipole mode)".  Those two
   statements are incompatible: every eta is negative and eta -> 0^- for the
   high-order modes, so literal descending-eta order puts the grid-scale modes
   (eta ~ -1e-6) first and the fundamental dipole mode (eta_1 = -0.0687, the
   MOST negative of the physical modes) last.  The acceptance values
   eta_1 = -0.0687, eta_2 = -0.0184, eta_3 = -0.0107 are in descending |eta|.
   We therefore sort by DESCENDING |eta|, which puts the fundamental dipole mode
   first as the contract intends.  ``eigenmodes`` takes no ordering argument, so
   this is the single ordering returned.

2. Signed q in the drift conductivity.  The contract's eta recipe calls
   ``graphene_drift_sigma(E, EF, gamma, abs(q), beta)``.  We pass the SIGNED q.
   The drift tensor is ODD in q -- its non-reciprocity lives in (alpha+beta)^2
   and in the alpha*beta cross term, NOT in any frame shift
   omega -> omega - q v_drift -- so with ``abs(q)`` the whole response would
   be even in q for any beta, the
   dissymmetry of paper 456 Eq. (2) would vanish identically, and the drift
   implementation could not be tested at all.  ``SIGNED_Q_IN_DRIFT_SIGMA``
   documents and controls this; set it to False to reproduce the contract
   literally (and to watch the beta != 0 dissymmetry collapse to zero).

3. ``build_V`` is called with ``abs(q)``.  The Coulomb kernel depends on q only
   through K_0(|q| s), so this is an exact identity, not an approximation; it
   also protects against a negative argument reaching K_0 for q < 0.

4. Boundary closure of ``operators.build_D_geom``.  The acceptance eigenvalues
   are reproduced ONLY with the zero-flux (Neumann) closure of the second
   difference, i.e. first/last diagonal entries -1/h^2 rather than -2/h^2,
   which is the physical no-current condition at the ribbon edges.  A plain
   Dirichlet tridiagonal gives eta_1 = -0.0222 instead of -0.0687.  The
   acceptance test's failure message says so explicitly.

5. ``eigenmodes`` is purely geometric and therefore ignores any substrate: the
   substrate reflection coefficient is frequency dependent in general, so there
   is no single frequency-independent A that includes it.
   ``build_M``/``solve_linear`` do include it, and since integration so does
   :meth:`RibbonSystem.eigenmodes_screened`, which takes the energy as a second
   argument and diagonalises ``(V + Vbar)(q, E) @ D_geom(q)``.  For a stack with
   no conducting sheet r does not depend on E at all, so that spectrum is then
   as geometric as the free-standing one -- see item 12.

6. COPLANAR RIBBON PAIRS: NO USABLE-RANGE LIMIT.  RESOLVED 2026-11-08; this
   item used to say the opposite and was wrong for as long as it said it.

   It claimed that a coplanar multi-ribbon system was trustworthy only while
   ``q*d <~ 4``, because the cross block took a 30-term modified-Struve series
   that decays past there, and that beyond it the block was "numerical noise
   instead of the exponentially small true coupling" (~4e-4 relative at
   ``q*d = 50``), guarded by a ``RuntimeWarning`` at
   ``operators.STRUVE_TRUSTED_MAX_ARG``.

   None of that is true any more.  ``operators.coulomb_V_entries`` branch 3 is
   accurate at EVERY coplanar separation and the warning was deliberately
   removed -- ``operators.STRUVE_TRUSTED_MAX_ARG``'s own docstring records the
   change, and says the constant was kept only because it still marks where the
   bare series stops being usable.  Measured here against ``2 h K0(Q d)``, the
   exact coupling, at ``h = 0.25``:

       q*d =  4   5.5806e-03 vs 5.5798e-03   ratio 1.0001
       q*d = 20   2.8707e-10 vs 2.8706e-10   ratio 1.0000
       q*d = 40   4.1969e-19 vs 4.1964e-19   ratio 1.0001

   -- four digits at a value of 4e-19, which is the opposite of noise.

   KEPT AS A WARNING ABOUT DOCUMENTATION, not about physics.  Two files
   disagreed, this one was the stale half, and a reader hunting a dropped band
   at ``q*d = 40`` would have found this item, concluded the regime was
   untrustworthy, and stopped looking.  That nearly happened.  When a limit
   quoted here is about to excuse a result, measure it before believing it.

7. Substrate quadrature.  ``substrate.build_Vbar`` warns when its default
   ``(Qmax, NQ)`` does not resolve the integrand; ``vbar_Qmax`` and ``vbar_NQ``
   on the instance are passed straight through so that can be tuned without
   touching the contract's defaults (None, 4000).

ADDED AT INTEGRATION
--------------------
8. ``use_toeplitz`` now defaults to True.  ``operators.build_V_toeplitz`` builds
   the same block from 2L-1 instead of L**2 kernel evaluations -- bit-exact
   through the first row and column, and agreeing with the dense build to
   ~1.4e-15 absolute elsewhere, which propagates to ~1e-12 on rho.  It is 33x
   to 149x faster (bench.py) and is what brings a cold solve at L = 300 inside
   the mandated 50 ms budget (~8 ms against ~115 ms).  Set it to False for the
   dense assembly; ``tests/test_integration.py`` asserts the two agree.

9. Substrate inputs are validated at construction, not warned about per call:
   see :meth:`RibbonSystem._validate_substrate_consistency` for the ``eps_a``
   vs ``layers[0].eps_par`` contradiction and the standoff double count, and
   :meth:`_Vbar_full` for the divergent q = 0 case.

10. ``nonlinear_grid`` and ``nonlinear_V_block`` are published hooks that
    ``nonlinear.py`` looks for.  They hand that module this class's own node
    arrays (instead of letting it reconstruct a different L) and give it
    substrate-screened harmonic blocks.  ``nonlinear_V_block`` takes the
    harmonic ENERGY as well as the wave vector, because the reflection
    coefficient is frequency dependent.

11. ZERO STANDOFF WITH A REFLECTING STACK IS WARNED ABOUT, not silently
    returned.  ``substrate.build_Vbar``'s integrand decays only as 1/Q when
    ``z_j + z_jp = 0`` and ``layers[0].d_nm = 0``, so the truncated oscillatory
    tail of its uniform trapezoid rule caps the accuracy at ~1e-3 however large
    NQ is (measured: Qmax = 1e5 with NQ = 2e6, 500x the work, still only
    reaches ~5e-4).  Any finite standoff cures it -- 1 nm gives ~2e-11 -- so
    :meth:`_Vbar_full` emits a ``UserWarning`` naming both conventions when the
    standoff is zero everywhere AND the stack actually reflects.  This is an
    upstream quadrature limitation, measured and documented in ``substrate.py``,
    not something this module can fix.

12. VBAR IS CACHED PER q, NOT PER (q, E), FOR A NON-DISPERSIVE STACK.  The
    reflection recursion uses ``E_ev`` only through the sheet term
    ``4 pi sigma/(i omega (eg_l + eg_l+1))``, so a stack whose
    ``layers[1:]`` all have ``sigma_sheet == 0`` -- any purely dielectric
    substrate -- has an exactly frequency-independent r, hence an exactly
    frequency-independent Vbar.  ``_Vbar_full`` therefore collapses the cache
    key to ``(q, 0.0)`` in that case and the NQ-point quadrature runs once per
    q instead of once per frequency.  Measured on a 40-frequency layered sweep
    (W = 10 nm, z0 = 1 nm, eps = 4, NQ = 4000): 1.50 s -> 0.055 s at L = 61
    (27x) and 6.42 s -> 0.23 s at L = 150 (28x), with BIT-IDENTICAL output
    (``tests/test_integration.py`` re-measures and asserts both).  The win
    grows linearly with the number of frequencies, and the premise is exact,
    not approximate -- see :meth:`_stack_is_dispersive`.  ``layers[0]``'s own
    ``sigma_sheet`` is excluded because there is no interface above the first
    layer, so it never enters the recursion (``substrate.Layer``).
    The cache is also BOUNDED at ``VBAR_CACHE_MAX`` entries (it was unbounded,
    which cost 150**2 * 16 bytes per (q, E) point -- 0.7 GB over a
    10 x 200 sweep at L = 150).
    The key covers the LAYER STACK, ``eps_a``, the quadrature controls and the
    ribbon geometry too, not just (q, E) -- all of them are public and mutable,
    and a key blind to them silently returns a matrix built from other inputs
    (measured: eps 4 -> 12 at fixed (q, E) returned the eps = 4 matrix, a 29%
    error with no symptom).  It also keys on ``abs(q)``, since Vbar is
    bit-identically even in q, which halves the work of a +-q sweep.
    See :meth:`_vbar_cache_key`.

13. THE LAYERED PATH IS NOW VALIDATED AGAINST PHYSICS, not just plumbed.  Until
    the ``tests/test_integration.py`` layered section existed, every
    solver-level substrate test used ``[Layer(1.0), Layer(1.0)]``, i.e. r = 0,
    so nothing downstream of ``substrate.build_Vbar`` had ever been exercised
    with a reflecting stack.  The anchors now pinned there, all measured at
    W = 10 nm, z0 = 1 nm, qW = 0.5, L = 61:

      * r -> 0 continuity: a vacuum stack reproduces the free-standing
        eta_1 = -0.0699604474, rho and polarizability BIT FOR BIT (build_Vbar
        short-circuits r == 0 to exact zeros, and adding exact zeros to V
        changes no bit);
      * monotone red shift in eps_substrate -- eta_1 = -0.06996, -0.08376,
        -0.09953, -0.12017 at eps = 1, 2, 4, 11.7, i.e. E_1 = 0.5723, 0.5231,
        0.4799, 0.4367 eV;
      * the perfect-mirror limit SATURATES: eta_1 -> -0.1389309 as
        eps -> infinity (r -> -1), already to 2e-4 at eps = 100 and to 2e-7 at
        eps = 1e4, which is the image-charge limit and the largest shift any
        passive dielectric can produce;
      * screening dies with standoff, FASTER at every doubling than the
        rigorous exp(-2 q z) envelope (the image factor exp(-Q (z_j + z_jp))
        with Q >= |q|): the eta_1 shift falls from -2.96e-2 at z = 1 nm to
        -1.82e-7 at z = 64 nm and -3.9e-14 at z = 200 nm;
      * an anisotropic TERMINATING half-space is bit-for-bit its isotropic
        ``eps_geo = sqrt(eps_par eps_perp)`` twin -- kappa depends on the layer
        only through eps_geo, and ``aniso_factor`` enters solely through
        ``exp(-2 Qhat d)``, which is 1 for a half-space.  A FINITE anisotropic
        slab does feel the anisotropy (eta_1 = -0.1040394 against -0.1037810
        with eps_par and eps_perp swapped at d = 20 nm) and converges to the
        half-space as d grows: 1.8e-2, 3.0e-4, 1.3e-6, 1.3e-14 relative at
        d = 5, 20, 50, 200 nm.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

try:  # package context (if an __init__.py is added later)
    from . import conductivity, operators, units
except ImportError:  # flat-module context: the package root is on sys.path
    import conductivity
    import operators
    import units

__all__ = [
    "Ribbon",
    "RibbonSystem",
    "bright_mode_mask",
    "SIGNED_Q_IN_DRIFT_SIGMA",
    "VBAR_CACHE_MAX",
    "ZETA_NORMALISATION",
]

#: See concern 2 in the module docstring.  True = physical (signed q),
#: False = the literal contract text (abs(q), which kills the drift asymmetry).
SIGNED_Q_IN_DRIFT_SIGMA = True

#: Largest number of substrate blocks ``RibbonSystem`` keeps memoised.  Each is
#: an (N, N) complex matrix, so an unbounded cache over a (q, E) grid is a
#: memory leak in all but name (see item 12 in the module docstring).  Eviction
#: is oldest-first, which matches the sweep access pattern (inner loop over E,
#: outer over q).
VBAR_CACHE_MAX = 8

#: Human-readable name of the mode normalisation used by :meth:`eigenmodes`.
ZETA_NORMALISATION = "reduced-potential Dirichlet energy: -sum_j h_j u_j^T D_geom,j u_j = 1, u_j = phi_j / W_j"


@dataclass
class Ribbon:
    """One zero-thickness graphene ribbon.

    Attributes
    ----------
    W_nm
        Width [nm].
    x0_nm, z0_nm
        Centre position [nm]; the ribbon spans x0 +- W/2 in the plane z = z0.
    EF_ev
        Fermi energy [eV].
    gamma_ev
        Inelastic broadening [eV].
    beta
        Drift velocity in units of the Fermi velocity; 0 = undrifted.
    L
        Number of grid nodes; ``None`` means derive it from ``dx_nm`` as
        ``max(31, round(W/dx))``.
    T_K
        Electron temperature [K] of *this* ribbon's conductivity; 0 means
        T = 0.  Temperature belongs to the sheet, not to the surroundings:
        two ribbons on the same substrate can be held at different electron
        temperatures, and each one's Drude weight follows its own.
    """

    W_nm: float
    x0_nm: float = 0.0
    z0_nm: float = 0.0
    EF_ev: float = 0.5
    gamma_ev: float = 0.02
    beta: float = 0.0
    L: int | None = None
    T_K: float = 0.0


def bright_mode_mask(zeta: np.ndarray, rel_floor: float = 1e-2) -> np.ndarray:
    """Boolean mask selecting the dipole-active (bright) modes.

    A mode is bright when ``|zeta| > rel_floor * max|zeta|``.  For a single
    ribbon at q -> 0 with L = 600 the bright overlaps are 0.943, 0.239,
    0.137, ... while the dark ones stay below 1e-3, so the default 1e-2
    separates the two families with a factor-of-ten margin on both sides.

    Parameters
    ----------
    zeta
        Dipole overlaps from :meth:`RibbonSystem.eigenmodes`.
    rel_floor
        Threshold relative to the largest |zeta|.

    Returns
    -------
    ndarray of bool
    """
    z = np.abs(np.asarray(zeta))
    zmax = float(np.max(z)) if z.size else 0.0
    if zmax == 0.0:
        return np.zeros(z.shape, dtype=bool)
    return z > rel_floor * zmax


class RibbonSystem:
    """A set of coplanar-or-stacked ribbons solved self-consistently.

    Parameters
    ----------
    ribbons
        Sequence of :class:`Ribbon` (a single :class:`Ribbon` is also accepted).
    eps_a, eps_b
        Permittivities above and below the sheet; ``eps_eff = (eps_a+eps_b)/2``.
    layers
        Optional substrate stack (list of ``substrate.Layer``).  When given, the
        reflected Coulomb interaction ``substrate.build_Vbar`` is added to every
        block of M, and to every block :meth:`nonlinear_V_block` hands
        ``nonlinear.py``.  ``eigenmodes`` and the sweep fast path do not support
        it (concern 5 in the module docstring); :meth:`eigenmodes_screened`
        does.  Use convention A -- ``layers[0].d_nm = 0.0`` with the standoff in
        each ``Ribbon.z0_nm`` -- and keep that standoff FINITE (item 11).
    dx_nm
        Target node spacing used when a ribbon leaves ``L`` unset:
        ``L_j = max(31, round(W_j / dx_nm))``.
    """

    def __init__(
        self,
        ribbons: Iterable[Ribbon] | Ribbon,
        eps_a: complex = 1.0,
        eps_b: complex = 1.0,
        layers: Sequence | None = None,
        dx_nm: float = 0.1,
    ) -> None:
        if isinstance(ribbons, Ribbon):
            ribbons = [ribbons]
        self.ribbons: list[Ribbon] = list(ribbons)
        if not self.ribbons:
            raise ValueError("need at least one ribbon")
        self.eps_a = eps_a
        self.eps_b = eps_b
        self.eps_eff = 0.5 * (eps_a + eps_b)
        self.layers = layers
        self.dx_nm = float(dx_nm)
        #: when True, inter/intra-ribbon V of equal-width, equal-grid pairs is
        #: built with ``operators.build_V_toeplitz`` instead of ``build_V``.
        #: DEFAULT True (integration decision, concern 8 in the module docstring):
        #: the Toeplitz block is the same matrix (bit-exact for x_off == 0, and
        #: to ~1.4e-15 absolute otherwise) built from 2L-1 instead of L**2
        #: kernel evaluations, and it is what brings a cold solve at L = 300
        #: inside the 50 ms budget (~8 ms vs ~115 ms).
        self.use_toeplitz = True
        #: passed through to ``substrate.build_Vbar`` (see concern 7).
        self.vbar_Qmax: float | None = None
        self.vbar_NQ: int = 4000

        self.L: list[int] = []
        self.theta: list[np.ndarray] = []
        self.h: list[float] = []
        self.x_nodes: list[np.ndarray] = []
        for rib in self.ribbons:
            if not rib.W_nm > 0.0:
                raise ValueError(f"ribbon width must be positive, got {rib.W_nm}")
            n = int(rib.L) if rib.L is not None else max(31, int(round(rib.W_nm / self.dx_nm)))
            if n < 3:
                raise ValueError(f"need at least 3 nodes per ribbon, got {n}")
            th = np.linspace(-0.5, 0.5, n)
            self.L.append(n)
            self.theta.append(th)
            self.h.append(float(th[1] - th[0]))
            self.x_nodes.append(rib.x0_nm + rib.W_nm * th)

        self.slices: list[slice] = []
        start = 0
        for n in self.L:
            self.slices.append(slice(start, start + n))
            start += n
        self.N = start

        self._validate_substrate_consistency()

        # instance-level memos (not global state): keyed by float(q)
        self._v_cache: dict[tuple[float, bool], np.ndarray] = {}
        self._eig_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        #: keyed by :meth:`_vbar_cache_key`, which covers EVERY input
        #: ``build_Vbar`` and ``r_of_Q`` read -- |q|, the energy (dropped when
        #: the stack carries no conducting sheet, since r is then exactly
        #: frequency independent), the layer stack's contents, eps_a, the
        #: quadrature controls and the ribbon geometry.  build_Vbar costs an
        #: NQ-point quadrature per entry, so neither the per-block accessor
        #: used by nonlinear.py nor a frequency sweep may rebuild it.
        #: Bounded at VBAR_CACHE_MAX (N, N) matrices.
        self._vbar_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # cross-module input validation
    # ------------------------------------------------------------------
    def _validate_substrate_consistency(self) -> None:
        """Reject the two ways a ``layers`` stack can contradict the ribbon data.

        Both are integration hazards reported by ``substrate.py``, and both
        produce a silently wrong number rather than an error if left alone.

        1. ``eps_a`` vs ``layers[0].eps_par``.  The reflection recursion takes
           the medium containing the ribbons from ``layers[0]``, while ``eta``
           takes it from ``eps_eff = (eps_a + eps_b)/2``.  If the two disagree
           the solve mixes two different ambient media, so this raises instead
           of letting ``substrate.reflection_coefficient`` warn once per call.

        2. Standoff double counting.  ``reflection_coefficient`` already
           carries ``exp(-2 Qhat_1 d_1)`` with ``d_1 = layers[0].d_nm``, and
           ``build_Vbar`` carries ``exp(-Q (z_j + z_jp))``.  Both describe the
           same source-to-image round trip, so setting ``layers[0].d_nm`` AND a
           non-zero ribbon ``z0_nm`` squares the standoff.  Convention A
           (``layers[0].d_nm = 0``, standoff in ``z0_nm``) is the one this
           class uses.

        Raises
        ------
        ValueError
            On either contradiction, naming the fix.
        """
        if not self.layers:
            return
        first = self.layers[0]

        eps_first = complex(getattr(first, "eps_par", 1.0))
        if complex(self.eps_a) != eps_first:
            raise ValueError(
                f"eps_a={self.eps_a!r} contradicts layers[0].eps_par={eps_first!r}. "
                "The ribbon-containing medium is specified twice: eps_a feeds eta via "
                "eps_eff, layers[0] feeds the reflection recursion. Set them equal "
                "(use eps_a = eps_b = layers[0].eps_par for an immersed sheet)."
            )

        d1 = float(getattr(first, "d_nm", 0.0))
        z_nonzero = [r.z0_nm for r in self.ribbons if float(r.z0_nm) != 0.0]
        if d1 != 0.0 and z_nonzero:
            raise ValueError(
                f"standoff is double counted: layers[0].d_nm={d1!r} and ribbon "
                f"z0_nm={z_nonzero!r} are both non-zero. reflection_coefficient "
                "applies exp(-2*Qhat_1*d_1) and build_Vbar applies "
                "exp(-Q*(z_j+z_jp)) for the same source-to-image round trip, so "
                "using both gives exp(-4*Q*s). Use convention A: layers[0].d_nm = "
                "0.0 with the standoff in each Ribbon.z0_nm."
            )

    # ------------------------------------------------------------------
    # geometry / operators
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RibbonSystem({len(self.ribbons)} ribbon(s), N={self.N}, "
            f"eps_eff={self.eps_eff}, layers={'yes' if self.layers else 'no'})"
        )

    def eta_pair(self, ribbon_index: int, q: float, E_ev: float) -> tuple[complex, complex]:
        """Dimensionless (eta_perp, eta_par) of one ribbon at (q, E).

            eta_{perp,par} = i sigma_{perp,par} / (omega W eps_eff)

        (module docstring, continuity derivation).  For ``beta == 0`` the
        isotropic Drude conductivity ``conductivity.graphene_drude`` is used for
        both components, as the contract allows; otherwise
        ``conductivity.graphene_drift_sigma`` supplies the anisotropic pair.
        Both branches carry the ribbon's own ``T_K`` through the Drude weight,
        and both reduce exactly to the T = 0 expressions when it is zero.
        The signed q is passed to the drift conductivity -- see concern 2 in the
        module docstring.

        Parameters
        ----------
        ribbon_index
            Index into ``self.ribbons``.
        q
            Guided wave vector [1/nm] (signed).
        E_ev
            Photon energy [eV].

        Returns
        -------
        (eta_perp, eta_par)
            Complex, dimensionless.
        """
        rib = self.ribbons[ribbon_index]
        omega = units.omega_of(E_ev)
        if rib.beta == 0.0:
            # Temperature enters the intraband weight and nothing else on this
            # branch: the T = 0 Drude weight |E_F| becomes mu_D(mu(T), T).  The
            # interband term is still absent, as the contract says -- this is
            # the Drude eta, warmed, not a switch to the local-RPA sigma that
            # the material panel plots.  At T_K = 0 the two calls below return
            # |E_F| exactly (drift_chemical_potential with beta = 0 and T = 0
            # is |E_F|; thermal_drude_weight at T = 0 is |mu|), so this branch
            # is bit-for-bit the old one whenever no temperature is set.
            weight = rib.EF_ev
            if rib.T_K > 0.0:
                mu = conductivity.drift_chemical_potential(rib.EF_ev, 0.0, rib.T_K)
                weight = float(conductivity.thermal_drude_weight(mu, rib.T_K))
            sigma = conductivity.graphene_drude(E_ev, weight, rib.gamma_ev)
            sigma_perp = sigma_par = sigma
        else:
            q_arg = float(q) if SIGNED_Q_IN_DRIFT_SIGMA else abs(float(q))
            sigma_perp, sigma_par = conductivity.graphene_drift_sigma(
                E_ev, rib.EF_ev, rib.gamma_ev, q_arg, rib.beta, T_K=rib.T_K
            )
        pref = 1j / (omega * rib.W_nm * self.eps_eff)
        return complex(pref * sigma_perp), complex(pref * sigma_par)

    def _V_block(self, j: int, jp: int, q: float) -> np.ndarray:
        """Bare Coulomb block V(j, jp), shape (L_j, L_jp).

        ``abs(q)`` is passed to ``operators`` (concern 3).
        """
        rj, rp = self.ribbons[j], self.ribbons[jp]
        aq = abs(float(q))
        same_grid = (rj.W_nm == rp.W_nm) and (self.L[j] == self.L[jp])
        if self.use_toeplitz and same_grid:
            V = operators.build_V_toeplitz(
                aq, rp.W_nm, rj.x0_nm - rp.x0_nm, rj.z0_nm - rp.z0_nm,
                self.theta[jp], self.h[jp],
            )
        else:
            V = operators.build_V(
                aq, rj.W_nm, rp.W_nm, rj.x0_nm, rp.x0_nm, rj.z0_nm, rp.z0_nm,
                self.theta[j], self.theta[jp], self.h[jp],
            )
        V = np.asarray(V)
        if V.shape != (self.L[j], self.L[jp]):
            raise ValueError(
                f"operators.build_V returned shape {V.shape}, expected "
                f"{(self.L[j], self.L[jp])} for ribbon pair ({j}, {jp})"
            )
        return V

    # ------------------------------------------------------------------
    # accessors used by nonlinear.py (published hooks, see README)
    # ------------------------------------------------------------------
    def nonlinear_grid(self, j: int) -> tuple[np.ndarray, float, int]:
        """Normalised grid of ribbon ``j`` as ``(theta, h, L)``.

        The explicit grid accessor ``nonlinear.py`` looks for first.  Without it
        that module falls back to sniffing attribute names and then to
        reconstructing ``L`` from ``W/dx``, which does NOT agree with this
        class's ``max(31, round(W/dx))`` -- so the nonlinear source could be
        assembled on a different grid from the potential driving it.  This hook
        removes the guesswork.

        Parameters
        ----------
        j
            Ribbon index.

        Returns
        -------
        (theta, h, L)
            The very arrays :meth:`solve_linear` works on.
        """
        return self.theta[j], self.h[j], self.L[j]

    def nonlinear_V_block(
        self, j: int, jp: int, q: float, E_ev: float | None = None
    ) -> np.ndarray:
        """Direct PLUS substrate-image Coulomb block, for the nonlinear solves.

        Returns ``V[j, jp] + Vbar[j, jp]`` at the given ``(q, E_ev)``, shape
        ``(L_j, L_jp)``, in exactly the normalisation ``operators.build_V``
        returns -- ``substrate.build_Vbar`` already carries the same ``2 h_jp``
        midpoint weight as branch 1 of ``build_V``, so the two blocks multiply
        the same charge vector and this is a drop-in replacement for
        ``build_V``.  ``nonlinear.py`` calls it in place of ``build_V`` so that
        a layered system screens its harmonic blocks as well as its linear one;
        without it that module uses the bare kernel and only warns.

        THE ENERGY ARGUMENT IS THE WHOLE POINT, and it is easy to get wrong.
        ``r(Q, omega)`` is frequency dependent, so the second- and third-order
        blocks must be screened at ``2 omega`` and ``3 omega``, NOT at the
        fundamental.  That is why this is a hook taking ``(q, E_ev)`` rather
        than a matrix cached once per system: the caller passes the wave vector
        AND the energy OF THIS BLOCK, both already multiplied by the harmonic
        index ``s``.  Screening the harmonics at the fundamental instead is a
        silent error -- the blocks stay the right shape and the solve still
        converges.  ``tests/test_integration.py`` pins it by asserting that a
        sheet-bearing stack gives measurably different blocks at ``E_ev`` and
        ``2*E_ev``.

        NO-OP GUARANTEE.  With ``layers=None``, or with a transparent stack
        (``r == 0`` on the whole quadrature grid, e.g. ``[Layer(1), Layer(1)]``,
        which ``build_Vbar`` short-circuits to exact zeros), the return value is
        bit for bit the system's own direct block :meth:`_V_block` -- adding
        exact zeros changes no bit.  That block is ``operators.build_V``
        bit-exactly when ``use_toeplitz`` is False, and agrees with it to the
        ~1e-15 absolute floor documented in ``operators.py`` on the default
        Toeplitz path.  The hook deliberately honours ``use_toeplitz`` rather
        than always calling ``build_V``, so that the nonlinear blocks are the
        same matrices the linear solve of the same system used.

        Parameters
        ----------
        j, jp
            Row and column ribbon indices.
        q
            Wave vector of this block [1/nm] (already multiplied by the
            harmonic index by the caller).
        E_ev
            Photon energy of this block [eV], likewise already scaled.  ``None``
            returns the bare block; with a stack present that is the unscreened
            answer, so it raises instead of quietly dropping the substrate.

        Returns
        -------
        (L_j, L_jp) complex128 ndarray

        Raises
        ------
        ValueError
            If the system carries layers and ``E_ev`` is None, or if ``q == 0``
            with a stack present (the reflected integral diverges there).
        """
        V = self._V_block(j, jp, q)
        if not self.layers:
            return V
        if E_ev is None:
            raise ValueError(
                "nonlinear_V_block needs E_ev when the system carries substrate "
                "layers: the reflection coefficient is frequency dependent, so the "
                "screened block cannot be built from q alone."
            )
        Vb = self._Vbar_full(float(q), float(E_ev))
        return V + Vb[self.slices[j], self.slices[jp]]

    def _V_full(self, q: float) -> np.ndarray:
        """Bare Coulomb matrix for all blocks, cached per (q, use_toeplitz)."""
        key = (float(q), bool(self.use_toeplitz))
        cached = self._v_cache.get(key)
        if cached is not None:
            return cached
        V = np.zeros((self.N, self.N), dtype=np.complex128)
        for j in range(len(self.ribbons)):
            for jp in range(len(self.ribbons)):
                V[self.slices[j], self.slices[jp]] = self._V_block(j, jp, q)
        self._v_cache[key] = V
        return V

    def _stack_is_dispersive(self) -> bool:
        """True if ``r(Q, omega)`` of this stack actually depends on omega.

        ``substrate.reflection_coefficient`` uses ``E_ev`` in exactly one place,
        the sheet term ``4 pi sigma_sheet/(1j omega (eg_l + eg_l+1))``.  A stack
        of plain dielectric layers therefore has a reflection coefficient --
        and so a ``Vbar`` -- that is EXACTLY frequency independent, not merely
        slowly varying, which is what lets :meth:`_Vbar_full` build it once per
        q over a frequency sweep (item 12 in the module docstring).

        ``layers[0].sigma_sheet`` is deliberately excluded: there is no
        interface above the first layer, so it never enters the recursion (see
        ``substrate.Layer``) and a value set there must not force the slow path.

        Returns
        -------
        bool
            False (non-dispersive) also when no layers are set.
        """
        if not self.layers:
            return False
        return any(complex(getattr(lay, "sigma_sheet", 0.0)) != 0.0 for lay in self.layers[1:])

    def _vbar_cache_key(self, q: float, E_ev: float) -> tuple:
        """Cache key covering every input ``Vbar`` actually depends on.

        Keying on ``(q, E)`` alone is not enough.  ``layers``, ``eps_a``,
        ``vbar_Qmax`` and ``vbar_NQ`` are plain public attributes -- the last
        two are documented tunables (concern 7) -- and ``Ribbon`` is a mutable
        dataclass whose ``z0_nm``/``W_nm``/``x0_nm`` ``build_Vbar`` reads live
        on every call.  A key blind to those returns a matrix built from
        DIFFERENT inputs, bit-for-bit, with nothing to indicate it: swapping a
        substrate from eps = 4 to eps = 12 at fixed (q, E) was measured
        returning the eps = 4 matrix, a silent 29% error on the screened
        Coulomb block.  A dispersive dielectric can only be expressed by
        swapping ``Layer`` objects (``Layer.eps_par`` is a constant), so that
        is the normal way to sweep a real substrate, not an abuse.

        Two deliberate collapses, both exact rather than approximate:

        * ``abs(q)``.  ``build_Vbar`` is called with ``abs(float(q))`` and
          ``r_of_Q`` sees only the quadrature's own ``Q``, so ``Vbar`` is even
          in q -- verified BIT-IDENTICAL, not merely close.  A +-q sweep
          (``observables.dissymmetry``) therefore halves its quadrature work.
        * the energy, when :meth:`_stack_is_dispersive` is False.  See item 12
          in the module docstring.

        Returns
        -------
        tuple
            Hashable; identical across a frequency sweep over a fixed
            non-dispersive stack, which is what preserves the item-12 win.
        """
        layers = tuple(
            (
                complex(lay.eps_par),
                None if getattr(lay, "eps_perp", None) is None else complex(lay.eps_perp),
                float(lay.d_nm),
                complex(lay.sigma_sheet),
            )
            for lay in (self.layers or ())
        )
        geom = tuple(
            (float(r.W_nm), float(r.x0_nm), float(r.z0_nm), n, h)
            for r, n, h in zip(self.ribbons, self.L, self.h)
        )
        return (
            abs(float(q)),
            float(E_ev) if self._stack_is_dispersive() else 0.0,
            layers,
            complex(self.eps_a),
            None if self.vbar_Qmax is None else float(self.vbar_Qmax),
            int(self.vbar_NQ),
            geom,
        )

    def _warn_if_zero_standoff(self, q: float, r_of_Q: Callable) -> None:
        """Warn when a reflecting stack is being evaluated at zero standoff.

        ``substrate.build_Vbar``'s integrand carries ``exp(-Q*(z_j + z_jp))``
        and ``reflection_coefficient`` carries ``exp(-2*Qhat_1*d_1)``; with both
        exponents zero the integrand decays only as 1/Q and the uniform
        trapezoid rule truncates an oscillatory tail it cannot resolve.  The
        measured accuracy ceiling is then ~1e-3 regardless of NQ, against
        ~2e-11 at a 1 nm standoff.  That is an upstream quadrature limitation
        documented in ``substrate.build_Vbar``, so this method does not try to
        fix it -- it refuses to let it pass silently, which is the real hazard
        (nothing else in the pipeline looks wrong).

        No warning is issued for a transparent stack: ``build_Vbar``
        short-circuits ``r == 0`` to exact zeros, so there is no quadrature and
        nothing to be inaccurate about.  That keeps the ``[Layer(1), Layer(1)]``
        plumbing check silent.
        """
        if float(getattr(self.layers[0], "d_nm", 0.0)) != 0.0:
            return  # convention B: the standoff lives in r's exp(-2*Qhat_1*d_1)
        if min(float(r.z0_nm) for r in self.ribbons) != 0.0:
            return
        # Only a stack that actually reflects runs the quadrature at all.
        W_max = max(float(r.W_nm) for r in self.ribbons)
        probe = np.array([abs(float(q)), max(abs(float(q)), 10.0 / W_max)])
        try:
            r_probe = np.asarray(r_of_Q(probe), dtype=np.complex128)
        except Exception:  # noqa: BLE001 - a non-vectorised stack is not our problem here
            return
        if not np.any(r_probe):
            return
        warnings.warn(
            "RibbonSystem: zero standoff with a reflecting stack "
            f"(layers[0].d_nm = 0.0 and z0_nm = 0.0, |r| ~ {abs(complex(r_probe[0])):.3g}). "
            "substrate.build_Vbar's integrand then decays only as 1/Q and its "
            "uniform trapezoid rule caps the relative accuracy at ~1e-3 for any "
            "NQ, so the screened blocks are good to about three digits and no "
            "more. Put the standoff in exactly one place: Ribbon.z0_nm "
            "(convention A, recommended -- 1 nm already gives ~2e-11) or "
            "layers[0].d_nm (convention B).",
            UserWarning,
            stacklevel=3,
        )

    def _Vbar_full(self, q: float, E_ev: float) -> np.ndarray:
        """Substrate-reflected Coulomb matrix (zero when no layers are set).

        Cached under :meth:`_vbar_cache_key`, which keys on every input the
        matrix depends on -- ``|q|``, the energy, the layer contents,
        ``eps_a``, the quadrature controls and the ribbon geometry -- so
        retuning any of them is seen rather than silently ignored.  Two exact
        collapses keep the win: ``Vbar`` is bit-identically even in q, and a
        stack with no conducting sheet has an exactly frequency-independent r,
        so its NQ-point quadrature runs once per ``|q|`` over a whole
        frequency sweep (item 12 in the module docstring).  The cache holds at
        most ``VBAR_CACHE_MAX`` matrices.

        Raises ``ValueError`` at ``q == 0`` with a stack present:
        ``substrate.build_Vbar``'s integrand goes as ``r(0)/Q_x`` near zero and
        the integral is genuinely logarithmically divergent there (exactly as
        the direct kernel ``2 K_0(|q| |dx|)`` diverges as ``q -> 0``).  The
        divergence is an additive constant that cancels in the response of a
        charge-neutral ribbon, so a sweep should approach q = 0 with a small
        finite q rather than include the point.  The message is raised here
        rather than three frames deeper because a q-sweep that happens to
        contain 0.0 is the normal way to hit it.
        """
        if not self.layers:
            return np.zeros((0, 0))
        # Covers the layer stack, eps_a, the quadrature controls and the
        # geometry as well as (|q|, E): see _vbar_cache_key for why each one
        # has to be in there, and which two collapses are exact.
        key = self._vbar_cache_key(q, E_ev)
        cached = self._vbar_cache.get(key)
        if cached is not None:
            return cached
        if float(q) == 0.0:
            raise ValueError(
                "q == 0 with a substrate stack: the reflected Coulomb integral "
                "substrate.build_Vbar evaluates is logarithmically divergent at "
                "q = 0 (the integrand goes as r(0)/Qx). Approach the limit with a "
                "small finite q (e.g. 1e-4/W) or drop the q = 0 point from the "
                "sweep; the divergence is an additive constant that cancels for a "
                "charge-neutral ribbon."
            )
        try:  # lazy: solver must import without substrate.py present
            from . import substrate  # type: ignore
        except ImportError:
            import substrate  # type: ignore

        def r_of_Q(Q):
            return substrate.reflection_coefficient(Q, E_ev, self.layers, eps1=self.eps_a)

        self._warn_if_zero_standoff(float(q), r_of_Q)

        Vb = np.zeros((self.N, self.N), dtype=np.complex128)
        for j in range(len(self.ribbons)):
            rj = self.ribbons[j]
            for jp in range(len(self.ribbons)):
                rp = self.ribbons[jp]
                blk = np.asarray(
                    substrate.build_Vbar(
                        abs(float(q)), rj.W_nm, rp.W_nm, rj.x0_nm, rp.x0_nm,
                        rj.z0_nm, rp.z0_nm, self.theta[j], self.theta[jp], self.h[jp],
                        r_of_Q, Qmax=self.vbar_Qmax, NQ=self.vbar_NQ,
                    )
                )
                if blk.shape != (self.L[j], self.L[jp]):
                    raise ValueError(
                        f"substrate.build_Vbar returned shape {blk.shape}, expected "
                        f"{(self.L[j], self.L[jp])} for ribbon pair ({j}, {jp})"
                    )
                Vb[self.slices[j], self.slices[jp]] = blk
        while self._vbar_cache and len(self._vbar_cache) >= VBAR_CACHE_MAX:
            self._vbar_cache.pop(next(iter(self._vbar_cache)))
        self._vbar_cache[key] = Vb
        return Vb

    def _D_blocks(self, q: float, E_ev: float) -> list[np.ndarray]:
        """Per-ribbon response operators D(jp) = eta_perp d^2_theta - eta_par (qW)^2."""
        out = []
        for jp, rp in enumerate(self.ribbons):
            eta_perp, eta_par = self.eta_pair(jp, q, E_ev)
            D = np.asarray(
                operators.build_D(q, rp.W_nm, self.h[jp], self.L[jp], eta_perp, eta_par)
            )
            if D.shape != (self.L[jp], self.L[jp]):
                raise ValueError(
                    f"operators.build_D returned shape {D.shape}, expected "
                    f"{(self.L[jp], self.L[jp])} for ribbon {jp}"
                )
            out.append(D)
        return out

    def _D_geom_blocks(self, q: float) -> list[np.ndarray]:
        """Per-ribbon eta-free operators D_geom(jp) = d^2_theta - (qW)^2."""
        out = []
        for jp, rp in enumerate(self.ribbons):
            D = np.asarray(operators.build_D_geom(q, rp.W_nm, self.h[jp], self.L[jp]))
            if D.shape != (self.L[jp], self.L[jp]):
                raise ValueError(
                    f"operators.build_D_geom returned shape {D.shape}, expected "
                    f"{(self.L[jp], self.L[jp])} for ribbon {jp}"
                )
            out.append(D)
        return out

    # ------------------------------------------------------------------
    # linear response
    # ------------------------------------------------------------------
    def build_M(self, q: float, E_ev: float) -> np.ndarray:
        """Full block interaction matrix, ``M[j, jp] = (V + Vbar)(j, jp) @ D(jp)``.

        The eta pair inside D belongs to the SOURCE ribbon jp (contract), which
        is why D carries a block index at all.  Shape (N, N) with
        ``N = sum_j L_j``.

        Parameters
        ----------
        q
            Guided wave vector [1/nm].
        E_ev
            Photon energy [eV].

        Returns
        -------
        (N, N) complex128 ndarray
        """
        M, _ = self._build_M_and_D(q, E_ev)
        return M

    def _build_M_and_D(self, q: float, E_ev: float) -> tuple[np.ndarray, list[np.ndarray]]:
        V = self._V_full(q)
        if self.layers:
            V = V + self._Vbar_full(q, E_ev)
        D_blocks = self._D_blocks(q, E_ev)
        M = np.zeros((self.N, self.N), dtype=np.complex128)
        for j in range(len(self.ribbons)):
            for jp in range(len(self.ribbons)):
                M[self.slices[j], self.slices[jp]] = (
                    V[self.slices[j], self.slices[jp]] @ D_blocks[jp]
                )
        return M, D_blocks

    def _concat(self, phi_ext) -> np.ndarray:
        """Flatten a per-ribbon list (or a single array) into length-N."""
        if isinstance(phi_ext, (list, tuple)):
            parts = [np.asarray(p, dtype=np.complex128).ravel() for p in phi_ext]
            if len(parts) != len(self.ribbons):
                raise ValueError(
                    f"phi_ext has {len(parts)} blocks, expected {len(self.ribbons)}"
                )
            vec = np.concatenate(parts)
        else:
            vec = np.asarray(phi_ext, dtype=np.complex128).ravel()
        if vec.size != self.N:
            raise ValueError(f"phi_ext has {vec.size} entries, expected {self.N}")
        return vec

    def _split(self, vec: np.ndarray) -> list[np.ndarray]:
        return [vec[sl].copy() for sl in self.slices]

    def solve_linear(self, q: float, E_ev: float, phi_ext) -> dict:
        """Solve ``(1 - M) phi = phi_ext`` and return the potential and charge.

        Parameters
        ----------
        q
            Guided wave vector [1/nm].
        E_ev
            Photon energy [eV].
        phi_ext
            External potential on the nodes: a length-N array or a per-ribbon
            list of arrays.

        Returns
        -------
        dict
            ``'phi'`` and ``'rho'``: lists of per-ribbon arrays, with
            ``rho_j = D(j) phi_j / W_j`` (contract; = physical charge density
            divided by eps_eff, see the module docstring).  ``'phi_flat'`` and
            ``'rho_flat'`` hold the same data concatenated.
        """
        rhs = self._concat(phi_ext)
        M, D_blocks = self._build_M_and_D(q, E_ev)
        A = np.eye(self.N, dtype=np.complex128) - M
        phi = np.linalg.solve(A, rhs)
        phi_list = self._split(phi)
        rho_list = [
            D_blocks[j] @ phi_list[j] / self.ribbons[j].W_nm
            for j in range(len(self.ribbons))
        ]
        return {
            "phi": phi_list,
            "rho": rho_list,
            "phi_flat": phi,
            "rho_flat": np.concatenate(rho_list) if rho_list else np.zeros(0),
        }

    # ------------------------------------------------------------------
    # geometric eigenmodes
    # ------------------------------------------------------------------
    def _A_from_V(self, V: np.ndarray, q: float) -> np.ndarray:
        """A = V @ D_geom, block assembled, demoted to real when V is."""
        D_blocks = self._D_geom_blocks(q)
        A = np.zeros((self.N, self.N), dtype=np.complex128)
        for j in range(len(self.ribbons)):
            for jp in range(len(self.ribbons)):
                A[self.slices[j], self.slices[jp]] = (
                    V[self.slices[j], self.slices[jp]] @ D_blocks[jp]
                )
        if np.max(np.abs(A.imag)) == 0.0:
            return A.real
        return A

    def _geometric_A(self, q: float) -> np.ndarray:
        """A = V @ D_geom with the BARE Coulomb matrix; frequency independent."""
        return self._A_from_V(self._V_full(q), q)

    def _screened_A(self, q: float, E_ev: float) -> np.ndarray:
        """A = (V + Vbar)(q, E) @ D_geom: the substrate-screened geometric operator."""
        V = self._V_full(q)
        if self.layers:
            V = V + self._Vbar_full(q, E_ev)
        return self._A_from_V(V, q)

    def eigenmodes(self, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Geometric (eta-free) eigenmodes of ``A = V @ D_geom``.

        The resonance condition is ``eta_m A phi = phi``, so ``eta_m = 1/lambda_m``
        with lambda the eigenvalues of A.  Modes are returned in order of
        DESCENDING |eta| (concern 1 in the module docstring), which puts the
        fundamental dipole mode first.

        Normalisation (this is the convention question the contract asks about).
        A = V D_geom is not symmetric, but D_geom A = D_geom V D_geom is, so A is
        self-adjoint under the bilinear form ``<u,v> = -h u^T D_geom v`` (plain
        transpose, not conjugate -- that is the form that symmetrises A).  We
        normalise the REDUCED potential ``u_j = phi_j / W_j``:

            -sum_j h_j u_j^T D_geom,j u_j = 1

        and then

            zeta_m = sum_j h_j sum_l theta_{j,l} rho_{m,j,l},
            rho_{m,j} = D_geom,j phi_{m,j} / W_j

        with UNIFORM-h Riemann sums (not ``trapz``: the reference convention
        weights every node by h, and trapz's half-weighted end nodes give
        zeta_1 = 0.910 at L = 600 and do not even converge in L, whereas the
        Riemann sum gives 0.94337 against the reference 0.943).  The sign of
        each mode is fixed so that zeta >= 0 where zeta is resolvable, and
        otherwise so that the largest-magnitude potential component is positive.

        This convention reproduces the reference dipole overlaps absolutely, not
        just in ratio: 0.94337, 0.23882, 0.13656 against the quoted 0.943,
        0.239, 0.137 (scale factor 1.0004).  No constant was tuned.

        At small but nonzero qW the near-null uniform-potential eigenvector of
        the zero-flux D_geom gives one mode with |eta| ~ 1/(qW)^2.  It is dark
        (zeta = 0) and :func:`bright_mode_mask` filters it out, which is why it
        used to be described here as spurious.  IT IS NOT.  ``dispersion.ldos_map``
        + ``track_peaks`` -- which peak-track an induced-field map and never
        diagonalise this operator at all -- put it at 0.2713 eV where the
        eigenvalue route puts it at 0.2708 (q = 0.1, L = 61, W = 10 nm, dipole
        at r0 = (3, 0, 6) nm; it must be OFF the ribbon axis, since an on-axis
        dipole does not drive this branch and Level 3 then shows only the bright
        modes), and a discretisation artefact of the eigenproblem cannot survive
        a method that does not solve the eigenproblem.  Its eta also converges in
        L (-3.9558, -3.9817, -3.9975, -4.0027 at L = 61, 121, 301, 601 for
        W = 10 nm, q = 0.02, so it belongs to the operator and not to the mesh),
        all of its charge is net charge (h sum(rho) / h sum|rho| = -1.000,
        against 1e-14 for the dipole-active modes -- the higher DARK modes are
        not exactly neutral either, keeping -5.3e-4, -6.8e-5, -2.1e-5 at modes
        2, 4, 6, L-independent and therefore a property of the operator at
        finite q, but mode 0 still stands three orders of magnitude clear of
        them), and 1/[eta (qW)^2] runs linearly
        in ln(2/qW) with slope 2.00 -- the logarithm of the 2 K_0 kernel.  With
        the lossless Drude eta that inverts to E ~ q sqrt(ln(1/qW)): the
        ribbon's one-dimensional wire plasmon, longitudinal and therefore dark
        to a transverse field, and far below the light line (0.023 eV against
        0.99 eV at q = 0.005 / nm) so quasistatics holds for it.  See the
        ``bright_only`` and ``track`` notes in ``dispersion.ribbon_bands``.

        Parameters
        ----------
        q
            Guided wave vector [1/nm].  Use a small nonzero qW (e.g. 1e-4) for
            the q -> 0 limit; the physical eta are charge neutral and therefore
            immune to the log divergence of K_0 as q -> 0 (verified identical
            from qW = 1e-3 down to 1e-6).

        Returns
        -------
        (eta, modes, zeta)
            ``eta`` shape (N,), ``modes`` shape (N, N) with ``modes[:, m]`` the
            normalised potential of mode m (concatenated over ribbons), ``zeta``
            shape (N,) the dipole overlaps.

        See Also
        --------
        eigenmodes_screened : the same spectrum for a layered system.
        """
        return self._eigenmodes_from_A(self._geometric_A(q), q)

    def eigenmodes_screened(
        self, q: float, E_ev: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eigenmodes of the SUBSTRATE-SCREENED operator ``(V + Vbar) @ D_geom``.

        Identical in every convention to :meth:`eigenmodes` -- same ordering,
        same normalisation, same zeta and sign rules -- but with the reflected
        Coulomb block added, which is what makes the substrate visible in the
        mode spectrum rather than only in a solved response.  With
        ``layers=None`` it returns exactly what :meth:`eigenmodes` returns.

        WHY THIS TAKES AN ENERGY AND :meth:`eigenmodes` DOES NOT.  r(Q, omega)
        is frequency dependent whenever the stack carries a conducting sheet, so
        there is then no single frequency-independent A and the eigenvalues are
        a function of E as well as q (concern 5 in the module docstring).  For a
        purely dielectric stack r does not depend on omega at all, so the
        returned spectrum is exactly frequency independent and E_ev only selects
        which cached block to use.

        What it is for: the resonance condition is unchanged, ``Re eta(E) =
        eta_m``, so for undrifted Drude graphene the screened mode energies are

            E_m = sqrt( -e^2 E_F / (pi eta_m W eps_eff) )

        (``units.E2_EV_NM`` for e^2).  A substrate makes every eta_m MORE
        negative -- r < 0 for eps_substrate > 1, so the image charge is of
        opposite sign and weakens the self interaction -- and therefore
        red-shifts every mode.  Measured against the Im-``polarizability`` peak
        of the layered solve (W = 10 nm, z0 = 1 nm, qW = 0.5), the closed form
        is low by exactly ``gamma**2 / (8 E_m**2)`` in relative terms -- the
        O(gamma^2) damping shift of a Lorentzian peak, which is all that is
        left once the mode energy itself is right.  Measured 2.17e-4, 5.43e-5,
        1.36e-5, 3.39e-6 at gamma = 20, 10, 5, 2.5 meV: a clean gamma^2 with
        the coefficient 1/8 to three digits.  So at gamma = 5 meV the closed
        form is good to 1.4e-5, not to a fraction of a percent.
        ``tests/test_integration.py`` asserts the law, not just a bound.

        Parameters
        ----------
        q
            Guided wave vector [1/nm].  Must be non-zero when layers are set:
            the reflected integral is logarithmically divergent at q = 0 (see
            :meth:`_Vbar_full`).
        E_ev
            Photon energy [eV] at which r is evaluated.

        Returns
        -------
        (eta, modes, zeta)
            As :meth:`eigenmodes`.
        """
        return self._eigenmodes_from_A(self._screened_A(q, float(E_ev)), q)

    def _eigenmodes_from_A(
        self, A: np.ndarray, q: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared body of :meth:`eigenmodes` and :meth:`eigenmodes_screened`."""
        lam, vec = np.linalg.eig(A)
        if np.iscomplexobj(lam):
            scale = float(np.max(np.abs(lam))) or 1.0
            if float(np.max(np.abs(lam.imag))) <= 1e-8 * scale:
                # A is real and its spectrum is real here; discard the roundoff
                # imaginary parts.  (If a genuine complex pair ever appears this
                # branch is skipped and complex modes are returned as-is.)
                lam = np.ascontiguousarray(lam.real)
                vec = np.ascontiguousarray(vec.real)

        with np.errstate(divide="ignore", invalid="ignore"):
            eta = 1.0 / lam
        eta = np.where(np.isfinite(eta), eta, np.inf)

        order = np.argsort(-np.abs(eta), kind="stable")
        eta = eta[order]
        vec = vec[:, order]

        D_blocks = self._D_geom_blocks(q)
        # normalisation: -sum_j h_j u_j^T D_j u_j = 1 with u_j = phi_j / W_j
        nrm2 = np.zeros(self.N, dtype=vec.dtype)
        for j, rib in enumerate(self.ribbons):
            u = vec[self.slices[j], :] / rib.W_nm
            nrm2 = nrm2 - self.h[j] * np.einsum("im,ij,jm->m", u, D_blocks[j], u)
        nrm = np.sqrt(np.abs(nrm2))
        nrm = np.where(nrm > 0.0, nrm, 1.0)
        vec = vec / nrm

        zeta = np.zeros(self.N, dtype=np.result_type(vec.dtype, np.complex128))
        for j, rib in enumerate(self.ribbons):
            rho_j = (D_blocks[j] @ vec[self.slices[j], :]) / rib.W_nm
            zeta = zeta + self.h[j] * (self.theta[j] @ rho_j)
        if np.max(np.abs(zeta.imag)) == 0.0:
            zeta = np.ascontiguousarray(zeta.real)

        # sign convention
        zabs = np.abs(zeta)
        floor = 1e-8 * (float(np.max(zabs)) if zabs.size else 0.0)
        sgn = np.ones(self.N, dtype=vec.dtype)
        resolvable = zabs > floor
        sgn[resolvable] = np.sign(np.real(zeta[resolvable]))
        sgn[sgn == 0] = 1.0
        if np.any(~resolvable):
            idx = np.where(~resolvable)[0]
            big = np.argmax(np.abs(vec[:, idx]), axis=0)
            picked = vec[big, idx]
            s = np.sign(picked.real)
            s[s == 0] = 1.0
            sgn[idx] = s
        vec = vec * sgn
        zeta = zeta * sgn
        return eta, vec, zeta

    # ------------------------------------------------------------------
    # observables
    # ------------------------------------------------------------------
    def _dipole_weights(self) -> np.ndarray:
        """Row vector g with ``alpha = g . (D phi)`` for the dipole integral.

        ``alpha = sum_j \\int x rho_j dx = sum_j W_j h_j sum_l x_{j,l} rho_{j,l}``
        and ``rho_j = (D phi)_j / W_j``, so ``g_{j,l} = h_j x_{j,l}``.
        Uniform-h Riemann sum, consistent with zeta.
        """
        g = np.zeros(self.N)
        for j in range(len(self.ribbons)):
            g[self.slices[j]] = self.h[j] * self.x_nodes[j]
        return g

    def _dipole_from_rho(self, rho_list: Sequence[np.ndarray]) -> complex:
        total = 0.0 + 0.0j
        for j, rib in enumerate(self.ribbons):
            total += rib.W_nm * self.h[j] * np.sum(self.x_nodes[j] * rho_list[j])
        return complex(total)

    def polarizability(self, q: float, E_ev: float, phi_ext) -> complex:
        """Induced dipole moment per unit length, ``sum_j \\int x rho_j(x) dx``.

        With ``phi_ext = -E0 x`` (the q = 0 plane wave of
        ``sources.phi_ext_plane_wave``) the external in-plane field is E0 and
        this is the polarizability per unit length divided by E0; for E0 = 1 the
        two coincide, which is why no field amplitude is taken as an argument.
        Units: nm^2 x (charge convention of rho, i.e. physical/eps_eff).

        Parameters
        ----------
        q
            Guided wave vector [1/nm].
        E_ev
            Photon energy [eV].
        phi_ext
            External potential, length-N array or per-ribbon list.

        Returns
        -------
        complex
        """
        sol = self.solve_linear(q, E_ev, phi_ext)
        return self._dipole_from_rho(sol["rho"])

    def dipole_from_rho(self, rho) -> complex:
        """The dipole per unit length of a charge density already solved for.

        :meth:`polarizability` is this applied to a fresh solve.  A caller that
        already holds ``solve_linear(...)["rho"]`` uses this instead, rather
        than solving the same system a second time for a number it has -- the
        nonlinear sweep does exactly that at every energy, since every harmonic
        is driven by the linear solution.

        Parameters
        ----------
        rho
            Charge density, as ``solve_linear`` returns it.

        Returns
        -------
        complex
        """
        return self._dipole_from_rho(rho)

    # ------------------------------------------------------------------
    # frequency sweeps
    # ------------------------------------------------------------------
    def fast_path_obstruction(self) -> str | None:
        """Why the isotropic eigen fast path is unavailable, or ``None`` if it is.

        The fast path needs ``M(E) = eta(E) A`` with a single scalar eta and a
        frequency-independent A.  That requires (i) beta = 0 for every ribbon
        (otherwise eta_perp != eta_par and D is not eta times D_geom), (ii) one
        common (W, EF, gamma, T_K) so that every block shares the same eta --
        eta carries a 1/W, and since 2026-11-05 a thermal Drude weight -- and
        (iii) no substrate, whose reflection coefficient is frequency dependent.

        ``T_K`` joined that tuple with the field itself: :meth:`sweep` evaluates
        ``eta_pair(0, ...)`` ONCE per (q, E) and applies it to every block, so a
        ribbon differing only in temperature would silently be solved at ribbon
        1's temperature rather than taking the slow path.

        Returns
        -------
        str or None
            A human-readable reason, or None when the fast path is valid.
        """
        if any(rib.beta != 0.0 for rib in self.ribbons):
            betas = [rib.beta for rib in self.ribbons]
            return (
                f"drift makes the conductivity a tensor (beta = {betas}). The fast "
                "path needs one number per energy multiplying a fixed geometry "
                "matrix, and with drift the across- and along-ribbon responses "
                "differ, so there is no single number"
            )
        if self.layers:
            return (
                "a substrate makes the Coulomb interaction frequency dependent. The "
                "fast path diagonalises the geometry once and reuses it at every "
                "energy, which needs an interaction matrix that does not move with "
                "the frequency; r(Q, omega) moves"
            )
        ref = self.ribbons[0]
        for i, rib in enumerate(self.ribbons[1:], start=1):
            got = (rib.W_nm, rib.EF_ev, rib.gamma_ev, rib.T_K)
            want = (ref.W_nm, ref.EF_ev, ref.gamma_ev, ref.T_K)
            if got != want:
                # No tuple dump. It read "ribbon 1 has (22.0, 0.5, 0.02, 0.0)
                # against ribbon 1's (75.0, 0.5, 0.02, 0.0)" -- four unlabelled
                # numbers the reader has to match up by position, and the same
                # ribbon named twice, because `enumerate(..., start=1)` makes
                # i = 1 the SECOND ribbon while the reference is also called
                # ribbon 1. The sentence says what is wrong; the parameter
                # column is where you see which ribbon differs.
                return (
                    "the ribbons are not interchangeable, so no single eta(E) "
                    "describes all of them: the fast path needs one common "
                    "(W_nm, EF_ev, gamma_ev, T_K)"
                )
        return None

    def _eigen_factors(self, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lambda, right eigenvectors, inverse of the eigenvector matrix), memoised."""
        key = float(q)
        cached = self._eig_cache.get(key)
        if cached is not None:
            return cached
        A = self._geometric_A(q)
        lam, vec = np.linalg.eig(A)
        lam = np.asarray(lam, dtype=np.complex128)
        vec = np.asarray(vec, dtype=np.complex128)
        vinv = np.linalg.inv(vec)
        self._eig_cache[key] = (lam, vec, vinv)
        return lam, vec, vinv

    def sweep(
        self,
        qs,
        Es,
        phi_ext_fn: Callable,
        fast_path: bool = True,
    ) -> np.ndarray:
        """Polarizability over a (q, E) grid.

        ``fast_path=True`` uses the isotropic eigen decomposition: A is
        eigendecomposed once per q and then, with ``M = eta(E) A``,

            phi(E) = sum_m v_m (u_m . phi_ext) / (1 - eta(E) lambda_m)

        where ``u_m`` are the rows of the inverse of the right-eigenvector
        matrix (the left eigenvectors, biorthonormalised).  ``fast_path=False``
        assembles and solves the full system at every frequency.

        BEHAVIOUR WITH DRIFT: ``fast_path=True`` RAISES ``ValueError`` when the
        fast path is invalid (any beta != 0, a substrate, or ribbons with
        different (W, EF, gamma)) instead of silently falling back, so a drifted
        sweep cannot quietly return isotropic numbers.  Pass
        ``fast_path=False``.  See :meth:`fast_path_obstruction`.

        Parameters
        ----------
        qs
            Wave vectors [1/nm].
        Es
            Photon energies [eV].
        phi_ext_fn
            ``phi_ext_fn(q, E_ev) -> length-N array`` (a per-ribbon list is also
            accepted).  A single-argument ``phi_ext_fn(q)`` is accepted too, for
            frequency-independent sources.
        fast_path
            Use the eigen expansion rather than a solve per frequency.

        Returns
        -------
        (len(qs), len(Es)) complex128 ndarray
            ``out[iq, iE] = polarizability(qs[iq], Es[iE], phi_ext)``.
        """
        qs = np.atleast_1d(np.asarray(qs, dtype=np.float64))
        Es = np.atleast_1d(np.asarray(Es, dtype=np.float64))
        out = np.zeros((qs.size, Es.size), dtype=np.complex128)

        if not fast_path:
            for iq, q in enumerate(qs):
                for iE, E in enumerate(Es):
                    out[iq, iE] = self.polarizability(
                        q, E, _eval_phi_ext(phi_ext_fn, q, E)
                    )
            return out

        reason = self.fast_path_obstruction()
        if reason is not None:
            raise ValueError(reason + " -- call sweep(..., fast_path=False)")

        g = self._dipole_weights()  # geometric weights; q independent
        for iq, q in enumerate(qs):
            lam, vec, vinv = self._eigen_factors(q)
            D_blocks = self._D_geom_blocks(q)
            for iE, E in enumerate(Es):
                eta = self.eta_pair(0, q, E)[0]  # isotropic: eta_perp == eta_par
                rhs = self._concat(_eval_phi_ext(phi_ext_fn, q, E))
                coef = (vinv @ rhs) / (1.0 - eta * lam)
                phi = vec @ coef
                # rho = D phi / W with D = eta * D_geom  =>  alpha = eta * g . (D_geom phi)
                acc = 0.0 + 0.0j
                for j in range(len(self.ribbons)):
                    sl = self.slices[j]
                    acc += g[sl] @ (D_blocks[j] @ phi[sl])
                out[iq, iE] = eta * acc
        return out


def _eval_phi_ext(fn: Callable, q: float, E_ev: float):
    """Call a source callable as ``fn(q, E_ev)`` or ``fn(q)``."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # builtins / C callables
        n = 2
    if n >= 2:
        return fn(q, E_ev)
    return fn(q)
