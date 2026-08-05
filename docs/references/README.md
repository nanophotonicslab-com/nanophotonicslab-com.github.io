# Reference documentation

Provenance notes for the Lab modules. Third-party publications are **not** redistributed
here — they are cited by DOI and must be obtained from the publisher.

## Mie Scattering — dipole excitation

The decay-rate formulas implemented in [`src/lib/dipole-decay.ts`](../../src/lib/dipole-decay.ts)
(dipole outside a sphere) and [`src/lib/dipole-inside.ts`](../../src/lib/dipole-inside.ts)
(dipole inside a sphere) follow:

> Y. S. Kim, P. T. Leung & T. F. George, *"Classical decay rates for molecules in the
> presence of a spherical surface: A complete treatment,"* Surface Science **195**, 1–14
> (1988). [doi:10.1016/0039-6028(88)90776-5](https://doi.org/10.1016/0039-6028(88)90776-5)

Validation against that paper is encoded in the test suites rather than in a document:
`src/lib/dipole-decay.test.ts` holds the reference table read from its figure 1 and asserts
agreement to better than 0.2%, and the implementation is additionally cross-checked against
the MNPBEM toolbox `@mieret/decayrate.m` to ~1e-9.

## BEM solver

`public/wasm/nplab_bem-0.1.0-py3-none-any.whl` is an independent Python implementation of
the Galerkin boundary-element scheme with Raviart–Thomas basis functions formulated in:

> U. Hohenester, N. Reichelt & G. Unger, *"Nanophotonic resonance modes with the nanobem
> toolbox,"* Computer Physics Communications **276**, 108337 (2022).
> [doi:10.1016/j.cpc.2022.108337](https://doi.org/10.1016/j.cpc.2022.108337)

It was written from the formulation published in that paper and cross-checked against the
authors' MATLAB toolbox during development. No code from that toolbox is included or
derived from; it is GPL-2.0, whereas this project is MIT. Tabulated optical constants
shipped with the wheel (`material/*.dat`) are the Johnson & Christy (1972) measurements,
regenerated from this repository's own `src/data/optical-constants.ts`.
