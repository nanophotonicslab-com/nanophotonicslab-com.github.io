# Node editor — decision brief

One page for the go/no-go. Full technical detail: [node-editor-spec.md](node-editor-spec.md).

## The proposal

A Blender-style node canvas where a simulation is wired as a bounded vertical
pipeline — **Material → Geometry → Solver → Spectrum → Plot/Color** — reusing
the solvers the site already has. Explicitly **not** a universal graph wiring
every solver into every other: most tools are different physics, not pipeline
stages, and a mega-graph would be decorative.

## What changed since the spec was drafted (all shipped, 2026-07-18)

The three prerequisite refactors the spec asked to approve separately are
**done and in production**, so the remaining decision is only about the canvas:

1. **Mie engine extracted** from the 5k-line page into `src/lib/mie.ts`
   with numerical-equivalence tests (`2e311ac`).
2. **One canonical `Complex` type** across the six struct-style physics
   modules (`c49eef1`).
3. **Shared `Spectrum` type + adapters** for the spectrum-emitting solvers
   (`76ef248`) — and the spec's minimal slice, *Material → PlasmonSolver →
   ColorSwatch*, already **runs end-to-end as an automated test**. The sockets
   connect today; no canvas exists to show them.

## The decision

**Go / no-go on building the canvas UI** for the bounded pipeline.

If go, two sub-decisions:

- **Canvas library vs hand-rolled.** `@xyflow`/`rete.js`/`litegraph.js` mean a
  framework dependency in a deliberately vanilla app; hand-rolling pan/zoom/wires
  is more work but keeps the zero-dependency policy.
- **v1 node scope.** Recommended: the proven 3-node slice behind a hidden
  route, then Mie and electron-sphere (their adapters exist), leaving
  BEM/RCWA/BPM out until there is a reason.

## Effort (order of magnitude)

| Piece | Effort | Status |
|---|---|---|
| Prereq refactors | medium | **done** |
| Graph engine (eval order + caching) | small–medium | pending go |
| Canvas UI + serialization | medium–large | pending go; dominated by library choice |
| Full pipeline across all compatible solvers | months | do not commit upfront |

## Recommendation

Build the 3-node slice behind a hidden route as a proof (small, the hard part
is already tested), re-evaluate before investing in the full canvas. The
library question can be deferred until the slice needs real interaction.

---

## Resolution (2026-07-18, decision interview with José Ramón)

**GO**, with the v0 reframed by the actual first-week use case:

- **v0 is a "material workbench", not the linear color slice.** The personal
  value is the shared material node: define a dispersion once (database /
  Drude / file) and fan it out to **Mie + plasmonic + electron-sphere
  side by side**. All three solvers already accept the material callback and
  all three have `Spectrum` adapters — the fan-out is the same proven sockets,
  wired 1→3 instead of 1→1. Color swatch comes along nearly for free.
- **No canvas, no library in v0.** Fixed topology behind a hidden route:
  parameter panels + live recompute. The canvas-library decision is deferred
  until the graph actually needs re-wiring by hand.
- **Starts now** — it takes the weekly project day. Mobile touch-chart work
  (issue #22) only pre-empts it if real-device testing surfaces breakage.
- **Álvaro's role: consultative.** The internal v0 proceeds without waiting;
  his input is sought on the public-facing scope and timing.
