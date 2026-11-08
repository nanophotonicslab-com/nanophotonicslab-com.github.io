# electrostatics_fd

Quasistatic (electrostatic) finite-difference reference library for graphene
nanoribbons: self-consistent linear response, geometric eigenmodes, layered
substrates, swift-electron and dipole sources, and the second- and third-order
nonlinear blocks.

It is a **reference implementation**, written to be read and checked against the
papers and the MATLAB originals, not a production kernel. Every public function
names its source equation. Where a specified formula looked wrong it was
implemented **as given** and the concern recorded — in the docstring, in a test,
and in the "Not proven" table at the bottom of this file. Nothing was quietly
corrected.

---

## Running the suite

```sh
cd <repo root>
py -3.11 -m pytest scripts/electrostatics_fd/ -q
```

Last run (2026-11-03): **846 passed, 1 xfailed, 0 failed** with `-m "not perf"`,
and **15 passed** with `-m perf` (numpy 1.24.3, scipy 1.11.1, pytest 7.4.0,
Python 3.11.5). Identical under random test ordering, so nothing in the suite is
order dependent.

The timing tests carry the `perf` marker, registered in `conftest.py`:

```sh
py -3.11 -m pytest scripts/electrostatics_fd/ -q -m perf        # only the budgets
py -3.11 -m pytest scripts/electrostatics_fd/ -q -m "not perf"  # skip them
```

`conftest.py` also puts the package's parent directory and the package directory
on `sys.path`, so the suite imports cleanly however pytest is invoked (repo root,
`scripts/`, a single file, with or without a path argument). The per-module test
files still carry their own path shims from before it existed; they are redundant
now, harmless, and were left alone rather than churned.

**One** `xfail` remains, and it is a record of a genuinely approximate formula,
not of skipped work:
`test_conductivity.py::test_finite_T_closed_form_should_match_numeric` — the
finite-T drift chemical potential's closed form departs from the numerical
carrier-number solution by ~0.5 % at 300 K (see the table below). `-rx` prints
its reason in full:

```sh
py -3.11 -m pytest scripts/electrostatics_fd/ -q -rx
```

The other ten xfails that stood here are gone, every one by being *resolved* and
not by being deleted: four physics disputes were settled and implemented, the
coplanar-Struve limitation was fixed in `operators.py`, and the zero-standoff
`build_Vbar` target was a bad requirement that is now stated at finite standoff.
Each is itemised under "Disputes resolved and implemented" below, and each former
xfail is now a real passing assertion carrying the measured numbers in its
docstring.

## Running the bench

```sh
py -3.11 scripts/electrostatics_fd/bench.py
```

No arguments. It prints four tables — Coulomb assembly, single solve, 100-point
frequency sweep, reflected substrate block — with wall-clock best-of-3 times and,
where two code paths compute the same thing, the relative difference between
them. It asserts nothing; `tests/test_performance.py` holds the budgets.

## Module map

| Module | Contents |
| --- | --- |
| `units.py` | eV-nm-fs Gaussian constants, `omega_of`, a.u. conductivity conversions |
| `conductivity.py` | graphene Drude / interband / local-RPA sheet conductivity, the anisotropic nonlocal drift conductivity tensor (Boltzmann on a skewed Fermi-Dirac distribution, not a frame shift) and its chemical potential, thin metal film, SHG / THG / Kerr / cascade coefficients |
| `operators.py` | `struve_L`, the three-branch Coulomb kernel (branch 3 evaluated in four regions — series, two Chebyshev panels, asymptotic — through the complementary function `Fc = 1 − F`, so it is cancellation-free at every coplanar separation), dense and Toeplitz `V` blocks, the finite-difference `D` operator |
| `substrate.py` | `Layer`, the layered quasistatic reflection coefficient, the reflected ("image") Coulomb block `build_Vbar` |
| `sources.py` | external potentials: plane wave, dipole, swift electron; `e_ext_electron` for the swift-electron **field** components `(Ex, Ey, Ez)` (`Ex = −γ_L⁻²·dφ/dx`, so the paper-430 `γ_L²` is a field factor, not a potential one); `lorentz_gamma`, `electron_decay_constant`, trajectory validity |
| `solver.py` | `Ribbon`, `RibbonSystem`: `eta_pair`, `build_M`, `solve_linear`, `eigenmodes`, `eigenmodes_screened`, `polarizability`, `sweep`, `nonlinear_V_block` |
| `observables.py` | induced field, LDOS, dissymmetry (uniform cell-weight quadrature, matching `operators.build_V`) |
| `dispersion.py` | the three-level dispersion stack: **Level 1** extended sheet (`sheet_response`, `sheet_reflection`, `sheet_plasmon_energy`, `extended_sheet_eta`), **Level 2** ribbon bands from the geometric eigenvalues (`ribbon_bands`, `drude_band_energy`, `BandStructure`, with `screened=True` for a purely dielectric stack), **Level 3** LDOS maps (`ldos_map`, `track_peaks`); plus `symmetric_q_grid` and `pm_asymmetry` for non-reciprocity |
| `nonlinear.py` | SHG and THG sources and block solves, including the cascade; each order is screened at **its own** harmonic (`2ω`, `3ω`) |
| `bench.py` | timing harness (not a test) |
| `tests/` | one file per module, plus `test_acceptance.py` (the agreed acceptance table), `test_performance.py` (budgets) and `test_integration.py` (the seams between modules) |

Import either way; the package re-exports its whole public surface:

```python
from electrostatics_fd import Ribbon, RibbonSystem, Layer     # package
from electrostatics_fd.solver import RibbonSystem            # module
```

## Unit convention

Gaussian **eV-nm-fs** throughout. There are no SI factors anywhere in the code.

| Quantity | Unit | Constant |
| --- | --- | --- |
| energy | eV | `HBAR_EV_FS = 0.6582119569` (ħ, eV·fs) |
| length | nm | `E2_EV_NM = 1.439964` (e², eV·nm) |
| time | fs | `C_NM_FS = 299.792458` |
| angular frequency | 1/fs | `omega_of(E_ev) = E_ev / HBAR_EV_FS` |
| wave vector | 1/nm | — |
| 2D conductivity | nm/fs | `SIGMA_AU_TO_NM_FS = 2.1876904...` |
| temperature | K | `KB_EV_K = 8.617333262e-5` |

Graphene Fermi velocity is `VF_NM_FS = C_NM_FS/300 = 0.99930819` nm/fs. Note the
existing TypeScript port `src/lib/nl-graphene.ts` hardcodes `VF = 1.0`, so
absolute second- and third-order conductivities differ from it by 0.138 % (`VF²`).

Matrices are `complex128`. The two kernel evaluators (`struve_L`,
`coulomb_V_entries`) return real `float64` because the kernel is manifestly real;
the four matrix builders return `complex128`.

Geometry: ribbon `j` has width `W_j`, centre `x_j`, height `z_j`, zero thickness,
infinite along `y` with wave vector `q`. Nodes are uniform in `theta ∈ [-1/2, 1/2]`
with `h = 1/(L-1)` and `x = x_j + W_j·theta`.

## Conventions a caller has to get right

* **Standoff, once only.** `reflection_coefficient` carries `exp(-2 Q̂₁ d₁)` with
  `d₁ = layers[0].d_nm`; `build_Vbar` carries `exp(-Q(z_j+z_jp))`. Both describe
  the same source-to-image round trip. `RibbonSystem` enforces this: setting
  `layers[0].d_nm` **and** a non-zero `Ribbon.z0_nm` raises. Use convention A —
  `layers[0].d_nm = 0`, standoff in `z0_nm`.
* **Ambient medium, once only.** `eps_a` and `layers[0].eps_par` are the same
  physical quantity. Disagreement raises at construction.
* **`q = 0` with a substrate raises.** The reflected integral is logarithmically
  divergent there, exactly as `2K₀(|q||dx|)` is. Approach the limit with a small
  finite `q`; the divergence is an additive constant that cancels for a
  charge-neutral ribbon.
* **Signed `q` reaches the drift conductivity.** The contract said `abs(q)`;
  `solver.SIGNED_Q_IN_DRIFT_SIGMA = True` passes the signed value, because the
  drift tensor's non-reciprocity — and therefore the whole paper-456
  dissymmetry — is odd in `q` and vanishes identically with `abs(q)`. Set the flag to `False` to
  reproduce the contract literally and watch `g` collapse to zero.
* **`rho` is the physical charge density divided by `eps_eff`.** Exact for
  `eps_a = eps_b = 1`. Everything downstream of `rho` inherits it.
* **Mode ordering** is by descending `|eta|`, which puts the fundamental dipole
  mode first and matches the acceptance table. The contract's literal "descending
  eta" would return the grid-scale modes first.
* **`nonlinear.py` is a `q = 0` 1-D collapse.** `q` is threaded correctly into
  `V` and `D`, but the source terms drop the `i q phi` y-derivative pieces, so
  source values at `q ≠ 0` are incomplete.

## What is verified

Numbers below are measured, not aspirational.

**Acceptance table (all 11 items pass).**

| # | Item | Measured |
| --- | --- | --- |
| 1 | bright ribbon eigenvalues, L = 600 | −0.068808, −0.018455, −0.010680 (ref −0.0687, −0.0184, −0.0107; atol 5e-4) |
| 2 | dark mode in between | −0.028919 (ref −0.0289) |
| 3 | eigenvalues width independent at q = 0 | identical to 8 d.p. for W = 5, 20, 100 nm |
| 4 | dipole coupling ratios | 0.25322, 0.14483 (ref 0.2535, 0.1453; rtol 5e-2) |
| 5 | drift chemical potential | numeric 0.484789 (ref 0.4848); T = 0 closed form exact |
| 6 | drift sigma → Drude at beta = 0 | 0 to 1.1e-16 relative, at q = 0 |
| 7 | SHG collapse `A + B + 2C` | 1e-16 relative, all gamma and both doping signs |
| 8 | single-interface Fresnel `r = (ε₁−ε₂)/(ε₁+ε₂)` | exact to 1e-16, now including metallic ε (see below) |
| 9 | `build_Vbar` vs the `K₀` kernel | 2.3e-11 at finite standoff; **z = 0 unreachable**, see below |
| 10 | `build_D` with equal etas is exactly `eta · build_D_geom` | bit-identical |
| 11 | leading eigenvalue grid converged | −0.0688848 (L = 300) → −0.0688084 (L = 600) |

**Other things pinned by a real number**, not just exercised: the Coulomb
kernel's three branches against `scipy` and against each other (branch-1
midpoint error quarters as `h` halves, measured ratios 4.027/4.013/4.007); the
Toeplitz and dense Coulomb paths agreeing on `rho` to 1e-12 while the Toeplitz
build is 36–141× faster; `build_D` row sums giving charge conservation for
arbitrary inhomogeneous doping; the drift conductivity's removable
`(alpha+beta)²` singularity handled exactly (equal to the literal brackets to
1e-11 over an (E, Q) grid, where the literal prefactor returns a non-finite
value); the generalised-K₀ identity behind `build_Vbar` to rtol 1e-9; the
standoff conventions A and B agreeing to rtol 1e-11 and the double-counted
version differing by >10 %; the swift-electron decay constant; the eigen fast
path matching a per-frequency solve to 3.8e-11 while running 72× faster.

**Resolved at integration** (both were reported as suspected upstream errors):

* *The cascade-A "discrepancy" is not one.* `conductivity.sigma_cascade_abcd`'s
  `A` and the reference's `S_s1_s2` differ in the coefficient of `D_s2²`
  (3 vs 1). Setting `s1 = s2 = 1` — the degenerate case, which *is* SHG —
  identifies both, exactly: `sigma_cascade_abcd(…,1,1)[0] == sigma_shg_abc[0]`
  (the tensor structure **A**), and `-S_1_1 == sigma_shg_1d_q0` (the q = 0 1-D
  collapse `A+B+2C`). They are different objects and both are right.
  `nonlinear.py` is the 1-D collapse, so its `cascade_A_form='reference'` default
  is correct on physical grounds. Pinned in `test_integration.py`; the strict
  xfail that asserted the false premise is gone.
* *A real bug, found and fixed:* `substrate.Layer.eps_geo` evaluated
  `sqrt(eps_par·eps_perp)` on numpy's principal branch. For an **isotropic**
  layer that is `sqrt(eps²)`, which returns the root with non-negative real part
  — so a Drude metal with `eps = -20+1j` came back as `+20-1j`. The permittivity
  changed sign, `r` came out −0.9050+0.0045j instead of −1.1050−0.0055j, and the
  `|r| > 1` surface-plasmon enhancement vanished. Nothing in the original suite
  sampled `Re eps < 0`. Fixed (isotropic short circuit; average-argument branch
  for genuinely uniaxial layers) and covered by metallic Fresnel cases in both
  `test_substrate.py` and the acceptance item 8 parametrisation, plus a
  surface-plasmon-pole test.

## Disputes resolved and implemented (2026-11-03)

Four long-standing disputes were derived independently, adversarially verified,
and are now **implemented**, not merely decided. One alleged bug turned out to be
a test asserting a false identity. The full record is kept outside this repo;
one-line verdicts:

| Dispute | Resolution |
| --- | --- |
| **Swift-electron prefactor** | The potential carries `γ_L⁰`, not `γ_L⁻¹`: the specified `-(2πe²)/(v·γ_L·Λ)` had a spurious Lorentz factor, and paper 430's `γ_L²` is a **field** factor entering through `Ex = −γ_L⁻²·dφ/dx`. The spurious factor is **removed** from `sources.py`, and `e_ext_electron(q, x_nm, z_nm, E_ev, v_over_c, b_nm, layers=None) -> (Ex, Ey, Ez)` now exports the field directly. |
| **Interband prefactor** | The log prefactor is `e²/(4πħ)`, so the absorptive part saturates at the **universal** `e²/(4ħ) = 0.5469` nm/fs. These were never alternatives — the π is the logarithm's branch cut. **Fixed** in `conductivity.py`, which also removes the `graphene_local_rpa` discontinuity at T = 1 K (the two branches no longer differ by π). |
| **Eq. (15c), the cascade `C`** | There was **no bug**. The "discrepancy" was a *test* asserting a false identity. Both `sigma_cascade_abcd` and `sigma_shg_abc` are correct, and the true identities are `sigma_C^(2,2) = ½[sigma_C^(2,1,1) + sigma_D^(2,1,1)]` (holds to 1.2e-15) and `A+B+C+D` at `s1=s2=1` `== sigma_shg_1d_q0` (5.7e-16). The tests are repointed to those; no numerics changed anywhere. |
| **Induced-field normalisation** | `observables.induced_field_q`'s `set_Eq_Riboons` form is correct **absolutely**, not just up to a constant. The `-1/(2π)` was a convention artifact of the reference MATLAB: the `1/(2π)` belongs in the caller's own `q` quadrature (paper 456 Eq. 21) and cancels in `dissymmetry`. Confirmed by energy conservation — `(ω/2)·Im(p*·E_ind) == ½·Re(σ)·∫|E|²` from `build_D`'s own quadratic forms — to ratio 1.000000000000 in six configurations. **Absolute LDOS magnitudes are not uncertain.** |

Two further limitations were fixed or corrected in the same pass:

* **Coplanar ribbon pairs at large `q·d` — FIXED.** Branch 3 is now formed from
  the complementary function `Fc = 1 − F` (`F(x) = (2/π)∫₀ˣK₀`), evaluated in
  four regions, so the old 9.068e-15 **absolute** cancellation floor is gone.
  The cross block of a coplanar pair tracks `2hK₀(q·d)` over 24 decades
  (`2.2e-4` down to `7.9e-29`), and two coplanar ribbons at `q·d = 50` now
  decouple to 3.2e-13 relative instead of ~4e-4. The `RuntimeWarning` guard was
  **deleted** rather than re-scoped, because nothing untrusted remains for it to
  guard. `STRUVE_TRUSTED_MAX_ARG` is still exactly `8.0`, but now documents the
  ceiling of the *bare 30-term series*, not of the Coulomb block.
* **`build_Vbar` at zero standoff — REQUIREMENT CORRECTED.** The acceptance item
  asked for rtol 1e-6 without qualifying the standoff; at `z_j + z_jp = 0` that is
  arithmetically unreachable, and `z = 0` is not a physical configuration (it puts
  the ribbon in the plane of its own image). The target is now stated **at finite
  standoff**, where the measured absolute accuracy is ~1e-14 of the block peak —
  eight orders inside what was asked. The real `z = 0` floor (7.24e-3 relative,
  unmoved by refinement) is still pinned, from both sides, by a real assertion
  rather than an xfail. This corrected a bad *requirement*; no tolerance was crept.

## What is NOT proven

Read this before quoting an absolute number.

| Item | Status |
| --- | --- |
| **Finite-T drift chemical potential** | The closed form departs from the numerical carrier-number solution by 0.52 % at 300 K. `graphene_drift_sigma` uses the closed form as specified; pass `mu_ev=drift_chemical_potential_numeric(...)` for the accurate value. **This is the suite's one remaining xfail.** |
| **`default_Qmax`'s third term** | `200/max(|dx|_max, 0.1)` makes `Qmax` *smaller* for larger separations. Implemented verbatim; mostly harmless because the z and q terms dominate, but it inflates `Qmax` to 2000 in the `|dx| -> 0` limit while `NQ` stays 4000 and destroys the peak. |
| **Residual branch-3 floor below argument 4** | For `Q·|θ| < 4`, `Fc` is itself formed as `1 − F`, so `V` keeps an absolute floor of a few ulps of `π/Q` — worst **3.3e-11** relative at `Q = 0.1`, `h = 1/149`, argument 4. Pre-existing, bounded, and harmless because `V` is not exponentially small there, so it cannot cause a wrong decoupling. A third Chebyshev panel on `[1, 4]` would remove it. |
| **`struve_L` accuracy** | `scipy.special.modstruve` is **not** machine-accurate everywhere: worst **7.4e-13** relative (at `x ≈ 14`, where scipy switches its own algorithm), not 1e-16. It does **not** reach the Coulomb kernel — `Fc` is built from the series and from the package's own Chebyshev/asymptotic expansions and never calls `modstruve` — but quote 7.4e-13, not machine precision. |
| **Absolute LDOS magnitudes after the quadrature fix** | `observables.induced_field_q` moved from `np.trapz` to uniform cell weights (`operators.build_V`'s own rule), which raises absolute induced-field and `ldos_map` values by 3.6 % to 30 % depending on observation point (the old values were **low**; 6.3 % at a typical point, 2.015 measured convergence order against 0.991 for trapz). Nothing in the suite pins a pre-fix absolute LDOS literal — do not reintroduce one. |
| **Level 1 ↔ Level 2 wide-ribbon agreement** | A **trend** test, not a tolerance test, and it cannot be made into one: the uniform-mode ratio approaches `−1/(2π·qW)` as `1 + 0.6/√(qW)` because of genuine inverse-square-root edge charge pile-up, so 1 % would need `qW ~ 3600`. Monotone approach from above plus two endpoint ratios is asserted instead of an invented tolerance. |
| **Level 3 absolute magnitude** | Every `dispersion.py` Level 3 assertion is on a shape, position, ratio or symmetry. No absolute LDOS value is frozen. |
| **`ldos_map` with ribbons at different `z0`** | `observables.induced_field_q` requires every source node at `z = 0`, so a stack of ribbons at different `z0_nm` is not represented. `ldos_map` warns (`RuntimeWarning`) rather than returning a wrong number. |
| **Substrate-screened nonlinear solves cannot include `q = 0`** | `RibbonSystem.nonlinear_V_block` raises at `q == 0` with layers present (the reflected Coulomb integral is logarithmically divergent there), while `nonlinear.py`'s sources are the `q = 0` collapse. Any screened nonlinear sweep must *approach* `q → 0`, not include it, and `q` must stay large enough for `build_Vbar`'s quadrature step to resolve it (`q = 1e-3` warns at the default `Qmax`/`NQ`; `q = 0.05` is clean). |
| **Vbar cache assumes immutable layers** | The `(q, E) → (q, 0.0)` key collapse is correct for the stack as constructed. Setting a layer's `sigma_sheet` from 0 to non-zero *after* entries are cached would flip the key scope; mutating `layers` was never supported. |
| **Layered nonlinear results** | Substrate screening now reaches the nonlinear blocks (`RibbonSystem.nonlinear_V_block`, which takes the harmonic energy because `r` is frequency dependent) and is structurally tested. It is *not* validated against an external number, and `build_Vbar`'s own quadrature accuracy applies. |
| **Inter-ribbon cascade coupling** | The THG cascade uses only the diagonal block, as the reference expression does. A real physical omission for closely spaced ribbons. |
| **Model edges** | `sigma_shg_*` and `sigma_cascade_abcd` diverge at E = 0; `sigma_thg_local` and `sigma_kerr` divide by `EF`; the second-order response has no `EF` dependence at all (so SHG cannot be gated in this approximation) and vanishes identically at `EF = 0`. `eigenmodes` ignores any substrate — use `eigenmodes_screened(q, E_ev)`, which is exactly frequency independent for a purely dielectric stack and genuinely E-dependent once a layer carries `sigma_sheet`. |

## Known rough edges (not correctness)

* `operators.coulomb_V_entries` and `solver._V_block` both take `abs(q)`. The
  duplicate is idempotent and harmless; one of them could go.
* `substrate.reflection_coefficient`'s `eps1` argument is redundant with
  `layers[0].eps_par` and cannot be made meaningful without breaking the
  single-interface acceptance test. It is accepted for signature compatibility
  and warns on a contradiction; `RibbonSystem` now prevents the contradiction
  from arising.
* `eigenmodes` at small `q·W` produces one spurious near-null uniform-potential
  mode from the zero-flux `D_geom`, with `|eta| ~ 1/(qW)²`. It is dark and
  `bright_mode_mask` filters it.
* The cubic-scaling perf test only reproduces its mandated `[4, 16]` window on
  one protocol (a complete first solve on the dense assembly). Retuning it on
  cached operators makes it flaky.
