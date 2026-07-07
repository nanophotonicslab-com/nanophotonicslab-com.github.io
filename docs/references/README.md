# Reference documentation

Source material for Lab modules, kept for provenance and validation.

## Mie Scattering — dipole excitation

- **`Kim et al. - Surf. Sci. 195, 1 (1988).pdf`** — Young Sik Kim, P. T. Leung & Thomas F. George, *"Classical decay rates for molecules in the presence of a spherical surface: A complete treatment,"* Surface Science **195**, 1–14 (1988). [doi:10.1016/0039-6028(88)90776-5](https://doi.org/10.1016/0039-6028(88)90776-5).
  The original source of the decay-rate formulas implemented in [`src/lib/dipole-decay.ts`](../../src/lib/dipole-decay.ts) (dipole outside a sphere, Fig. 1) and [`src/lib/dipole-inside.ts`](../../src/lib/dipole-inside.ts) (dipole inside a sphere, Figs. 4 & 5), and reproduced by the **Dipoles** excitation of the Mie Scattering Calculator.

- **`mie-dipole-validation-results.pptx`** — validation deck: the library's computed curves overlaid on the scanned Kim figures (Figs. 1, 4, 5), confirming the NPLab port reproduces the published results. Also cross-checked against the MNPBEM toolbox `@mieret/decayrate.m` to ~1e-9 (see the `src/lib/dipole-*.test.ts` suites).
