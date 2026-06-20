# Spec — Node Graph for Simulation Pipelines (DRAFT for review)

Status: **proposal for the V2 gate** (needs Álvaro's OK before any build).
Author: working session 2026-06-20. Grounded in an audit of `src/lib/`.

## The idea, honestly scoped

A Blender-"Material Nodes"-style canvas where you wire a simulation as a graph:

```
[Material/Dispersion] ──► [Geometry] ──► [Solver] ──► [Spectrum] ──► [Plot]
                                                          └────────► [ColorSwatch]
```

**What it is NOT:** a universal graph that wires every solver into every other.
Most of the 13 tools are *different physics*, not pipeline stages — you don't
pipe Mie into RCWA. A cross-solver mega-graph would be decorative wiring with no
semantics. The defensible win is the **vertical pipeline** above: define a
material/dispersion and geometry once, feed it to a solver, post-process to a
spectrum/field/color. That pipeline already exists *latently* in the code.

## Why it's plausible (evidence from the code)

The math core is in better shape than expected — **composability ≈ 5/10**, and
the hardest conceptual seam is already built:

- **The material→solver socket already exists as an injected callback.** Three
  solvers take dispersion as a function, not baked-in:
  - `cylinder.getDispersion({ eps1_of_w: (w_eV)=>C, … })` (`src/lib/cylinder.ts:621,650`)
  - `electron-sphere.computeElectronSphereSpectrum(getDielectric: (λnm)=>[n,k], …)` (`src/lib/electron-sphere.ts:540`)
  - `plasmonic.computePlasmonSpectrum({ epsilonAt?: (λnm)=>Complex, … })` (`src/lib/plasmonic-nanoparticles.ts:50,370`)
- **A genuine shared dispersion layer:** `src/lib/materials.ts` + the
  `[[λ_µm, n, k]]` table format (`src/data/optical-constants.ts:5-9`) is used by
  mie, plasmonic, electron-sphere, photothermal.
- **A natural terminal sink:** `colorimetry.spectrumToHex(lams, R)`
  (`src/lib/colorimetry.ts:37`) already consumes a spectrum → color.
- **6 of the TS solvers are pure** (deterministic, no DOM/globals): mie,
  cylinder, electron-sphere, plasmonic, bpm, plus materials/colorimetry helpers.

## Why it's a real project, not a weekend (the obstacles)

1. **The flagship Mie compute is inline in the page, not in `lib/`.**
   `mie-scattering.astro` reimplements Mie/multilayer in ~4100 lines of inline
   script and never imports `src/lib/mie.ts`. The single most important node
   would have to be extracted from a 5k-line page first.
2. **Three incompatible complex-number types.** `C = [number,number]` tuple
   (cylinder, electron-sphere) vs `ComplexDielectric{re,im}` (materials) vs
   `Complex{re,im}` (plasmonic). Sockets can't connect without unification.
3. **No shared `Spectrum` / `Grid` / `Geometry` types.** Every solver emits a
   bespoke struct (`MieResult`, `PlasmonSpectrum`, `ElectronSpectrum`,
   `DispersionResult`, …) — all morally "x-grid + named `Float64Array` channels"
   but none unified. Each node would need an adapter today.
4. **Geometry is not first-class.** It's loose scalars (`radiusNm`,
   `lengthNm`+`aspectRatio`, BEM `subdivisions`) passed *beside* the solver, not
   produced by an upstream node. There is no geometry node to occupy the middle
   of the pipeline. (Only `bpm.nProfile` is a structured geometry array, and
   nothing produces it.)
5. **BEM & RCWA can't be graph nodes (yet).** Both are async Pyodide/Python
   workers parameterized by **scalar ε**, not a dispersion function
   (`src/scripts/bem`, `src/scripts/rcwa`). Non-pure, non-synchronous, can't
   ingest a material node without re-plumbing the worker protocol. They stay
   outside the first graph.
6. **State/render entanglement.** Orchestration lives in multi-thousand-line
   `.astro` scripts (and `bem/main.ts` fuses state + 3D viewport + canvas). The
   graph engine has to be lifted out of the view layer.

## Proposed socket types (derived from actual I/O)

1. **Material** — `(lambdaNm) => [n, k]` (or `ε{re,im}`; plus the `(w_eV)=>C`
   variant). Already the injected callback in 3 solvers. The canonical material
   node output.
2. **Spectrum** — `{ x: Float64Array /*λ or E*/, series: Record<string, Float64Array> }`.
   The backbone edge. Adapters map each solver's bespoke struct into this.
3. **Geometry** — typed struct (radius, length, aspectRatio, hostIndex,
   subdivisions…). Promote today's loose scalars into a real node output.
4. **Field/Map** — 2D `Float64Array` + extents/mesh (`bpm.field`, EELS
   q-resolved maps, BEM enhancement). Heavy-array output for visual nodes.

## Smallest viable vertical slice (build this first)

**Material → PlasmonSolver → {Plot, ColorSwatch}** — a 3-node graph using only
`materials.ts` → `plasmonic-nanoparticles.computePlasmonSpectrum` →
`colorimetry.spectrumToHex`.

Why this trio:
- All three are **already in `lib/`, pure, typed** — no page extraction, no
  worker, no Python.
- `computePlasmonSpectrum` **already accepts the Material socket**
  (`epsilonAt?: (λnm)=>Complex`) and **already emits a spectrum**
  (`wavelengthNm` + `sigmaScaNm2`).
- `spectrumToHex(lams, R)` needs only a one-line `sigmaScaNm2 → R` adapter.

Glue required: a tiny `Spectrum` wrapper type + the sigma→R adapter. That's it.
Avoid for the first slice: Mie (inline in page), BPM (no material socket),
BEM/RCWA (async Python).

## Decision points for Álvaro

1. **Go / no-go on the bounded pipeline graph** (not the universal version).
2. **Canvas library** vs hand-rolled: `svelte-flow`/`@xyflow`, `rete.js`,
   `litegraph.js`. Trade-off = bundle weight + a framework dependency in an
   otherwise vanilla-Astro app vs build/maintain our own pan/zoom/wire.
3. **Prerequisite refactors as their own PRs** before any canvas work:
   - unify complex types → one `Complex`
   - define shared `Spectrum` / `Grid` / `Geometry` types in `src/lib/`
   - extract Mie compute out of `mie-scattering.astro` into `src/lib/mie.ts`
     usage (the page currently bypasses the lib)
4. **Scope of v1 nodes:** start with the 3-node slice; add Mie + electron-sphere
   once types are unified; leave BEM/RCWA/BPM out until there's a reason.

## Rough effort (order-of-magnitude, not a commitment)

- Prereq type unification + Mie extraction: **medium** (touches several files,
  but mechanical and independently shippable / testable).
- Minimal graph engine (eval order + caching) over pure node fns: **small–medium**.
- Canvas UI + serialization: **medium–large**, dominated by the library choice.
- Full pipeline across all compatible solvers: **months**, do not commit upfront.

Recommendation: approve the **prereq refactors** (they're good hygiene
regardless — they also resolve audit items #3 and #6), then build the 3-node
slice behind a hidden route as a proof, and re-evaluate before investing in the
canvas.
