# CLAUDE.md — working agreement

Read this first, every session. The startup procedure is always the same.

## Startup (do this, in order, without exploring)

1. **Repo:** this folder — `nanophotonicslab-com.github.io`. Remote: `github.com/nanophotonicslab-com/nanophotonicslab-com.github.io`. Default branch `main`.
2. **Sync:** `git pull` on `main`. (The untracked `.claude/` folder is local harness config — ignore it, never commit it.)
3. **Dev server:** start the `dev` launch config → Astro on **http://localhost:4321/**. Verify it's up via the preview logs, then stop.
4. Report status in 2–3 lines (branch, pull result, server URL) and wait for the change request.

Don't re-discover any of the above each session — it's fixed.

## What this is

Static **Astro** site for nanophotonicslab.com (GitHub Pages), plus an interactive **Lab** of in-browser physics widgets. No frontend framework — vanilla TS in each widget's scoped `<script>`, Canvas 2D for plotting, KaTeX for math. Full stack/layout details in [README.md](README.md).

## Commands

```sh
npm run dev       # dev server at localhost:4321
npm run build     # production build -> ./dist/
npm run preview   # serve ./dist/
npm test          # vitest
npm run lint:css  # stylelint — blocks CI
npm run check     # astro check (types in .astro) — blocks CI
```

## Styling rules (these are enforced, not suggestions)

- **Shared widget chrome lives in `src/styles/lab.css`**, imported by the lab pages. Hero, layout, controls panel, presets, chart frame, stat strip, export bar, references — one definition each. Do **not** re-declare these in a widget's scoped `<style>`; that is exactly how the styles drifted apart before (`.stat-value` had 4 definitions, `.preset-btn` 7).
- Per-widget variation goes through the knobs `lab.css` documents at the top (`--sidebar-w`, `--hero-measure`, `--chart-aspect`, `--stats-cols`). Need a new axis? Add a knob there rather than a local copy.
- A local override is legitimate for **behaviour** the shared component can't know about — `cursor: crosshair` on a chart with a hover readout, `overflow`, `z-index`. Keep it to just those properties.
- Widget styling follows the dense "Gen-A" style (compact pill buttons, `0.6875rem` labels, `--text-muted`), the convention cylinder / mie-scattering / photothermal already used.
- **No hex colours.** Every colour has a token in `src/styles/global.css` — spectral palette, status (`--success` / `--warning` / `--danger` plus their `-soft` / `-ink` pairs), band inks, `--graph-canvas`. Stylelint fails the build on a raw hex; if the colour you need genuinely has no name, add the token.

## How we proceed (keep it cheap)

- Lab widgets live in `src/pages/lab/*.astro`. Components in `src/components/`, data modules in `src/data/`, shared layout `src/layouts/Layout.astro`.
- Make the change, verify it in the running preview (don't ask the user to check manually), then summarize.
- Commit / push only when asked. Pushing to `main` auto-deploys via GitHub Pages.
- Don't add dependencies or frameworks without asking — the no-framework, no-plotting-lib constraint is deliberate.

## Shortcuts

- **`ccc`** — ship everything to the live site, in one go, no further confirmation:
  1. Stage all current changes (`git add -A`).
  2. Commit with a concise message summarizing what changed this session.
  3. Make sure it lands on `main` — if currently on a feature branch, merge it into `main` (fast-forward when possible); if already on `main`, skip the merge.
  4. `git push origin main` → this triggers the GitHub Pages build and publishes to nanophotonicslab.com.
  Then report the commit hash and confirm the push succeeded. Treat `ccc` as standing authorization to push to `main` — don't ask again.
