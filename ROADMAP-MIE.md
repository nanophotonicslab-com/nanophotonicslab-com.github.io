# Mie Scattering Widget — Roadmap & Feature Plan

## Context

The Mie scattering calculator at `/lab/mie-scattering/` currently supports:
- Single sphere and multi-shell (up to 5 shells) with validated BHCOAT algorithm
- 5 built-in materials + custom epsilon + CSV file upload
- Interactive chart with zoom, crosshair, PNG/CSV export
- Collapsible layer sections, mobile-responsive layout

Target audience: **researchers** (PhDs, postdocs, experimentalists designing nanoparticles).
They want precision, speed, and export — if it's also intuitive and beautiful, even better.

Monetization (Pro mode): **undecided** — design with clean separation so it can be gated later.
Compute strategy: **browser-first**, backend only for Pro/heavy compute.

---

## Feature Roadmap

### Phase 1: Spectral Analysis (browser, free)
Enhance the existing spectral plot with decomposition and better data sources.
All data already computed — just expose it.

| Feature | What | Effort | Impact |
|---------|------|--------|--------|
| **Multi-curve overlay** | Show Csca + Cext + Cabs simultaneously | Low | High |
| **Multipole decomposition** | Separate curves for n=1 (dipolar), n=2 (quadrupolar), n=3, n≥4 | Low | Very high |
| **Electric / Magnetic split** | Separate a_n (electric) vs b_n (magnetic) contributions | Low | Very high |
| **refractiveindex.info** | Fetch materials from GitHub YAML database, searchable selector | Medium | High |

**Implementation notes:**
- Multi-curve requires refactoring `drawChart` to accept `PlotSeries[]` instead of single `yArr`
- Multipole/E-M: extend `MieResult` with per-order and per-mode arrays; accumulate in existing Mie loop
- refractiveindex.info: fetch from `raw.githubusercontent.com` (CORS OK), parse YAML with `js-yaml`
- UI: add "Decomposition" tabs (Total | E / M | Multipoles) below quantity selector

### Phase 2: Spatial Visualization (browser, candidate for Pro)
New visualizations beyond the spectral plot.

| Feature | What | Effort | Impact |
|---------|------|--------|--------|
| **Particle cross-section** | SVG diagram showing layers, radii, materials with colors | Low | Medium |
| **Far-field angular pattern** | Polar plot of scattering amplitude vs angle (from S1, S2) | Medium | High |
| **Near-field (analytical)** | 2D heatmap of |E|² in a cross-section plane, all from Mie coefficients | High | Very high |
| **Save/share config** | Encode all parameters in URL hash for sharing | Low | Medium |

**Implementation notes:**
- Far-field: S1(θ), S2(θ) from a_n, b_n using π_n, τ_n angular functions (Bohren & Huffman Ch. 4)
- Near-field: internal + external field expansion in vector spherical harmonics. For single sphere, analytical. For multi-shell, need field matching at each interface. Computationally intensive (grid of points × nmax terms) but feasible in browser for moderate sizes.
- Cross-section SVG: pure rendering, no physics computation

### Phase 3: Pro Compute (backend)
Heavy computation, gated behind Pro mode.

| Feature | What | Effort | Impact |
|---------|------|--------|--------|
| **BEM integration** | Link to existing BEM solver for arbitrary shapes | High | Very high |
| **Parameter sweeps** | Sweep radius/wavelength/material, generate 2D color maps | Medium | High |
| **Batch export** | Compute and export data for multiple configurations | Medium | Medium |

---

## Proposed UI Structure

### Current (Free) Mode
```
[Geometry: Sphere | Core-Shell]
[Parameters: material, radius, etc.]
[Quantity: Csca | Cext | Cabs | Qsca | All]
[Decomposition: Total | E / M | Multipoles]    ← NEW
[Chart: spectral plot with multi-curve support]
[Export: PNG | CSV]
```

### Pro Mode (future toggle)
```
[Everything in Free, plus:]
[Tabs below chart: Spectrum | Angular | Near-field]
[Angular: polar far-field plot]
[Near-field: 2D heatmap with colorbar]
[Particle diagram: SVG cross-section]
[Save/Share button]
```

---

## Implementation Order (recommended)

### Sprint 1: Multi-curve + Decomposition
**Files:** `src/pages/lab/mie-scattering.astro`

1. Refactor `drawChart` to accept `PlotSeries[]` with colors and labels
2. Add legend rendering for multi-curve
3. Update tooltip to show all visible curve values
4. Extend `MieResult` interface:
   - `csca_e, csca_m, cext_e, cext_m` (electric/magnetic)
   - `csca_mp[], cext_mp[]` (per-multipole, 4 entries: n=1,2,3,rest)
5. Modify `computeMie` inner loop: accumulate per-mode/per-order sums
6. Modify `computeMieMultiShell`/`bhcoat`: same
7. Add "All" quantity option and "Decomposition" tab group to HTML/JS
8. Update `getPlotData` to return series based on quantity + decomposition mode

**Estimated scope:** ~200 lines changed, 1 session.

### Sprint 2: refractiveindex.info integration
**Files:** `src/pages/lab/mie-scattering.astro`, possibly new `src/data/refractiveindex.ts`

1. Add `js-yaml` dependency
2. Build material catalog: fetch `catalog-nk.yml` from GitHub, parse into searchable list
3. Add searchable material selector UI (combobox with autocomplete)
4. On material selection: fetch specific YAML file, extract tabulated n,k data
5. Cache fetched materials in memory

**Estimated scope:** ~300 lines new code, 1 session.

### Sprint 3: Far-field angular pattern
**Files:** `src/pages/lab/mie-scattering.astro` (or new component)

1. Implement π_n(cosθ), τ_n(cosθ) angular functions
2. Compute S1(θ), S2(θ) from a_n, b_n
3. Render polar plot (canvas or SVG)
4. Add "Angular" tab below chart

**Estimated scope:** ~400 lines, 1 session.

### Sprint 4: Near-field heatmap
Larger project — may span multiple sessions.

### Sprint 5: Pro mode infrastructure
UI toggle, feature gating, backend integration with BEM.

---

## Verification

Each sprint should include:
- `npx astro build` — clean compilation
- Visual testing in browser (dev server)
- For physics features: validation against known results (e.g., Rayleigh limit for dipole dominance)
- No regression in existing functionality (sphere mode, core-shell mode)
