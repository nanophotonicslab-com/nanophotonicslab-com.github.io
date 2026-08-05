# NanophotonicsLab

Source of [nanophotonicslab.com](https://nanophotonicslab.com) — a free collection of
interactive nanophotonics tools that run entirely in your browser. No installation, no
account, no server: the site is static files, and every calculation happens on your machine.

If these tools support published work, please cite them — see [Citing](#citing).

## What is in it

Seventeen tools, grouped the way the site groups them.

### Calculators — closed-form, answer-as-you-type

| Tool | What it does |
| --- | --- |
| [Photon](https://nanophotonicslab.com/lab/photon/) | wavelength ↔ energy ↔ frequency ↔ wavenumber ↔ period, on an annotated spectrum |
| [Relativistic Electron](https://nanophotonicslab.com/lab/electron/) | electron kinematics, the evanescent field of a passing electron, Smith–Purcell, PINEM |
| [Laser Pulse](https://nanophotonicslab.com/lab/laser/) | nine sub-tools: transform limit, peak power, GDD, Gaussian beams, focusing, chirp, nonlinear conversion, brightness, LG/HG modes |
| [Ultrafast Pulse Toolbox](https://nanophotonicslab.com/lab/pulse/) | GVD/TOD from Sellmeier data, zero-dispersion wavelengths, broadening through glass, autocorrelation deconvolution |
| [Materials](https://nanophotonicslab.com/lab/materials/) | the optical-constants library every other tool reads: built-in tables, refractiveindex.info search, file import, Drude and graphene models |
| [Unit Converter](https://nanophotonicslab.com/lab/units/) | SI ↔ Gaussian/ESU ↔ atomic units for eleven quantities |

### Photonics Lab — analytical and semi-analytical models

| Tool | What it does |
| --- | --- |
| [Mie Scattering](https://nanophotonicslab.com/lab/mie-scattering/) | exact Mie series for spheres and up to five shells: cross-sections, multipoles, angular patterns, near field, EELS, dipole decay, thermal profile, perceived colour |
| [Cylinder Dispersion](https://nanophotonicslab.com/lab/cylinder/) | exact boundary matching for an infinite cylinder: guided modes, group velocity, scattering, EELS |
| [Plasmonic Nanoparticles](https://nanophotonicslab.com/lab/plasmonic-nanoparticles/) | fitted analytical polarizabilities for rods, disks, rings, cages, bipyramids, prisms and polyhedra; graphene nanostructures with nonlinear response |
| [Photothermal](https://nanophotonicslab.com/lab/photothermal/) | laser heating of a nanoparticle: absorption from Mie, steady-state temperature profile |
| [Optical Tweezers](https://nanophotonicslab.com/lab/tweezers/) | radiation pressure from exact Mie coefficients, Rayleigh-regime trap stiffness and depth |
| [Purcell Factor](https://nanophotonicslab.com/lab/purcell/) | emitter near a sphere: enhancement against quenching, modified lifetime and quantum yield |
| [Heterostructures](https://nanophotonicslab.com/lab/heterostructures/) | surface plasmons in layered media, transfer-matrix multilayers, LDOS and dipole emission in stratified layers |

### Toolkit — full-wave and numerical

| Tool | What it does |
| --- | --- |
| [BEM Solver](https://nanophotonicslab.com/lab/bem-solver/) | retarded Galerkin BEM with Raviart–Thomas basis functions, in Python via Pyodide |
| [RCWA](https://nanophotonicslab.com/lab/rcwa/) | rigorous coupled-wave analysis of 1D gratings, via `inkstone` in Pyodide |
| [Beam Propagation](https://nanophotonicslab.com/lab/bpm/) | scalar paraxial BPM with a Crank–Nicolson scheme |
| [Imaging](https://nanophotonicslab.com/lab/imaging/) | simulated fluorescence microscopy with exact ground truth: diffusion, tracking, MSD recovery |

Plus a [Methods Assistant](https://nanophotonicslab.com/lab/assistant/), semantic search over
computational-photonics notes, run locally in the browser.

Every tool carries a maturity badge (**Stable** / **Beta** / **Experimental**), an
information box stating its model, assumptions and limitations, and export to PNG, CSV and a
shareable permalink that encodes the whole parameter set in the URL — open one and you get
that calculation back exactly.

## Stack

- **Astro** (static) for routing, layouts and components
- **Vanilla TypeScript** in each tool's scoped `<script>` — no frontend framework
- **Canvas 2D** for all plotting — no charting library
- **KaTeX** for maths
- **Pyodide** (Python → WebAssembly) for the BEM and RCWA solvers, lazy-loaded on first use
- **three.js** for the BEM mesh viewport only
- **`@huggingface/transformers`** for in-browser semantic retrieval, lazy-loaded

No backend, no database, no accounts. Analytics are anonymous and cookie-free, and no user
input is ever transmitted.

## Running locally

Requires Node.js ≥ 22.12.

```sh
npm install
npm run dev       # dev server on http://localhost:4321/
npm run build     # static output in ./dist/
npm run preview   # serve ./dist/ locally
npm test          # Vitest
npm run check     # astro check (types)
```

## Layout

```
src/lib/          physics kernels — pure TypeScript, unit-tested
src/pages/lab/    one page per tool; interactive logic in a scoped script
src/data/         optical constants and material tables
src/components/   shared UI (ModelInfo, ToolMeta, ParamsDrawer, …)
public/wasm/      the BEM wheel served to Pyodide
public/data/      precomputed embeddings for the methods assistant
scripts/          offline Python pipeline (not deployed)
docs/             provenance notes
benchmarks/       validation notebooks
```

## Tests

`npm test` runs the Vitest suites over the physics kernels. They are not smoke tests: they
encode analytical limits (Rayleigh scaling, the bare-dipole limit far from a sphere, the
`d⁻³` divergence of quenching near a metal, unit LDOS in free space), conservation checks
(`Cext = Csca + Cabs`, multipole sums, power conservation under Crank–Nicolson stepping),
agreement with published reference tables, parity against PyRAMIDS and DeepTrack2, and a
regression test for every numerical defect found in past audits.

The two Pyodide solvers are not covered by this suite, which is TypeScript-only. The BEM
solver is instead validated end-to-end against exact Mie theory; RCWA rests on the upstream
validation of `inkstone`. Both are marked *Experimental* for that reason.

## Methods assistant — offline pipeline

The assistant runs semantic retrieval entirely in the browser, over precomputed embeddings
shipped as a static JSON asset in `public/data/`. The pipeline that rebuilds the corpus from
a PDF of notes lives in `scripts/`.

```sh
# 1. One-time Python env setup
cd scripts
uv venv .venv
source .venv/bin/activate
uv pip install marker-pdf sentence-transformers numpy

# 2. Convert PDF → structured Markdown (2–10 min; downloads models on first run)
marker_single /path/to/notes.pdf --output_format markdown \
    --disable_image_extraction --output_dir ./out

# 3. Parse markdown, chunk by section, embed, write JSON
python build_embeddings.py ./out/notes/notes.md ../public/data/notes.json
```

Edit the `source` metadata in `build_embeddings.py` to match the new corpus. The browser side
(`src/pages/lab/assistant.astro`) hardcodes the JSON path via
`fetch('/data/cpho16_embeddings.json')` — update it if you rename the file.

## Deployment

Pushed to `main`; GitHub Pages builds and deploys automatically.

## Citing

Please cite the article describing the toolkit, and the archived release for the exact
version you used. `CITATION.cff` and `.zenodo.json` carry the machine-readable metadata, and
each tool page has its own "cite this tool" entry with its version and last-updated date.

## Contributing and feedback

Issues and pull requests are welcome. There is also a *Feedback* button on every page for
anonymous messages, and <info@nanophotonicslab.com> for anything longer.

## Licence

MIT — see [LICENSE](LICENSE). Tabulated optical constants belong to their original authors
and are cited in `src/data/optical-constants.ts` and in each tool's information box.
