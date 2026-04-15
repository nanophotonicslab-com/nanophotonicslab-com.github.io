# nanophotonicslab.com

Source code for the [NanophotonicsLab](https://nanophotonicslab.com) website —
a static Astro site hosted on GitHub Pages, with an interactive Lab section of
in-browser physics widgets.

## Stack

- **Astro** (static mode) for routing, layouts, components
- **Vanilla TypeScript** in each widget's scoped `<script>` block — no frontend framework
- **Canvas 2D** for all plotting (no Chart.js / Plotly dependencies)
- **KaTeX** for LaTeX math rendering in the methods assistant
- **`@huggingface/transformers`** for in-browser semantic retrieval (lazy-loaded)
- **Python + `sentence-transformers` + `marker-pdf`** for the offline corpus
  pipeline that feeds the methods assistant

## Lab widgets

All live under `src/pages/lab/`:

| Route | What |
|---|---|
| `/lab/mie-scattering/` | Exact Mie theory for spheres and multi-shell geometries, with far-field angular patterns, near-field maps, multipole decomposition, refractiveindex.info integration |
| `/lab/photothermal/` | Laser-driven nanoparticle heating: cross-sections, albedo, steady-state ΔT, and apparent color under white-light illumination (CIE 1931 + sRGB) |
| `/lab/photon/` | Converter for photon quantities (λ, E, ν, ω, wavenumber, period) with live EM spectrum chart |
| `/lab/electron/` | Relativistic electron kinematics (v/c, K, de Broglie wavelength, γ) |
| `/lab/units/` | SI ↔ CGS/ESU ↔ atomic units converter for 11 physical quantities |
| `/lab/assistant/` | Methods assistant (hidden from the Lab index, preview) |

## Local development

```sh
npm install
npm run dev         # dev server at localhost:4321
npm run build       # production build → ./dist/
npm run preview     # serve ./dist/ locally
```

## Methods assistant — offline pipeline

The assistant runs semantic retrieval entirely in the browser, over a set of
pre-computed embeddings shipped as a static JSON asset in `public/data/`. The
pipeline to (re)build the corpus from a PDF of notes lives in `scripts/`.

```sh
# 1. One-time Python env setup
cd scripts
uv venv .venv
source .venv/bin/activate
uv pip install marker-pdf sentence-transformers numpy

# 2. Convert PDF → structured Markdown (2-10 min, downloads models first run)
marker_single /path/to/notes.pdf --output_format markdown \
    --disable_image_extraction --output_dir ./out

# 3. Parse markdown, chunk by section, embed, write JSON
python build_embeddings.py ./out/notes/notes.md ../public/data/notes.json
```

Edit the `source` metadata in `build_embeddings.py` to match the new corpus.
The browser side (`src/pages/lab/assistant.astro`) hardcodes the JSON path via
`fetch('/data/cpho16_embeddings.json')` — update it if you rename the file.

## Project layout

```
.
├── public/
│   ├── data/           # embeddings JSON for the assistant
│   └── favicon.svg
├── scripts/            # Python offline pipeline (not deployed)
│   └── build_embeddings.py
├── src/
│   ├── components/     # Navbar, Footer, LabToolsBar
│   ├── data/           # TypeScript data modules (optical constants, etc.)
│   ├── layouts/
│   │   └── Layout.astro
│   ├── pages/
│   │   ├── index.astro
│   │   ├── research.astro
│   │   ├── publications.astro
│   │   ├── team.astro
│   │   └── lab/
│   │       └── *.astro
│   └── styles/
│       └── global.css
└── astro.config.mjs
```

## Deployment

Pushed to `main`; GitHub Pages builds and deploys automatically.
