# @nanophotonicslab/optics

The physics engines behind [nanophotonicslab.com](https://nanophotonicslab.com/lab/),
as a typed, **zero-dependency** ES module library. Every solver is exercised by
the test suite of the site (260+ tests, many pinned to literature values).

```sh
npm install @nanophotonicslab/optics
```

```js
import { mie, purcell, pulse, opticalForces } from '@nanophotonicslab/optics';

// Mie cross-sections of a 60 nm sphere (n = 1.5) in water at 550 nm
const { csca, cabs, cext } = mie.mieAt(1.5, 0, 1.33, 60, 550);

// Optical-tweezers trap depth for 50 mW focused to w0 = 0.6 µm
const trap = opticalForces.rayleighTrap({
  nParticle: 1.57, radiusNm: 60, nHost: 1.33,
  lambdaNm: 1064, powerMw: 50, waistUm: 0.6,
});
console.log(trap.trapDepthKbT); // ≈ 15 kBT

// GVD of fused silica at 800 nm
pulse.gvdFs2PerMm(pulse.SELLMEIER['fused-silica'], 800); // ≈ 36.2 fs²/mm
```

Exports are **namespaced per module** (solvers legitimately share names like
`csqrt` or `interpolateNK`):

| Namespace | Physics |
|---|---|
| `mie` | Plane-wave Mie: single sphere, BHCOAT core–shell, N-layer multishell, per-order coefficients, far-field angular patterns, near-field maps |
| `plasmonic` | Plasmonic nanoparticle mode expansion (rods, disks, cages, …) with retardation correction |
| `electronSphere` | EELS / cathodoluminescence spectra of a swift electron near a sphere |
| `dipoleDecay`, `dipoleInside`, `purcell` | Exact dipole decay rates near/inside a sphere (Kim 1988) and Purcell/quantum-yield/lifetime bookkeeping |
| `opticalForces` | Radiation pressure from exact Mie (asymmetry parameter), Rayleigh-regime Gaussian trap |
| `photothermal` | Steady-state nanoparticle heating under CW Gaussian illumination |
| `pulse` | Sellmeier dispersion of common optical materials, GVD/TOD, pulse broadening, autocorrelation factors |
| `fit` | Least-squares Mie fits to measured spectra + tolerant CSV parsing |
| `spectrum` | Shared spectrum type, adapters, peak analysis (FWHM, Q) |
| `materials`, `colorimetry`, `complex`, `graph` | Tabulated optical constants, spectrum→sRGB, complex arithmetic, dependency-graph engine |
| `cylinder`, `smithPurcell`, `vo2`, `nlGraphene`, `grapheneConductivity` | Cylinder EELS/dispersion, Smith–Purcell far fields, VO₂ phase-change optics, graphene response |

Units are stated per function (typically nm, eV, fs; SI where it matters).
Citations to the underlying papers live in the doc comments.

## Cite

See `CITATION.cff` in the repository root (Zenodo DOI per release).

## Development

Source of truth lives in the site repository (`src/lib`); this package re-exports
it. `npm run build` bundles with esbuild and emits declarations with tsc.

MIT © Jose Ramon Martinez Saavedra
