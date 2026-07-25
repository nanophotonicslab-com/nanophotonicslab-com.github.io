/**
 * The point of this config is the design-token guard, not cosmetics.
 *
 * Widget styles must take colour from the tokens in src/styles/global.css
 * (`var(--primary)`, `var(--rose)`, …) rather than pasting a hex literal —
 * a hex literal is invisible to the dark-mode overrides and is how the palette
 * drifted apart across widgets in the first place.
 *
 * Rules that would fight the house style (dense single-line CSS rules, blank
 * lines used for grouping) are switched off deliberately: this repo's CSS is
 * written to be scanned vertically, and reformatting it would churn every file
 * for no functional gain.
 */
export default {
  extends: ['stylelint-config-standard'],

  overrides: [
    {
      // lets stylelint read the <style> blocks inside .astro components
      files: ['**/*.astro'],
      customSyntax: 'postcss-html',
    },
    {
      // global.css *is* the palette — it is the one place hex belongs
      files: ['src/styles/global.css'],
      rules: { 'color-no-hex': null },
    },
  ],

  rules: {
    /* Every hex that had an exact token equivalent is already converted. The
     * ~67 that remain are status colours (success / warning / danger surfaces)
     * the palette simply doesn't define yet — promoting them needs new tokens
     * in global.css, which is a design call. Kept at `warning` so the guard is
     * visible on new code without blocking the build on that backlog; flip to
     * `true` once the tokens exist. */
    'color-no-hex': [true, { severity: 'warning' }],

    // blank lines around at-rules are used for grouping here, not structure
    'at-rule-empty-line-before': null,

    // `:global(...)` is Astro's escape hatch out of scoped styles, not a typo
    'selector-pseudo-class-no-unknown': [true, { ignorePseudoClasses: ['global'] }],

    // -webkit-backdrop-filter and friends are still load-bearing in Safari
    'property-no-vendor-prefix': null,
    'value-no-vendor-prefix': null,
    'at-rule-no-vendor-prefix': null,
    'color-function-alias-notation': null,

    // house style: dense, one rule per line
    'declaration-block-single-line-max-declarations': null,
    'rule-empty-line-before': null,
    'comment-empty-line-before': null,
    'declaration-empty-line-before': null,
    'custom-property-empty-line-before': null,

    /* Pre-existing split declarations (e.g. `.param-number` declared once for
     * layout and again for the Firefox spinner reset). Worth merging, but not
     * worth blocking a build over. */
    'no-duplicate-selectors': [true, { severity: 'warning' }],

    // naming and specificity are the author's call, not the linter's
    'selector-class-pattern': null,
    'custom-property-pattern': null,
    'keyframes-name-pattern': null,
    'no-descending-specificity': null,

    // shorthand/notation preferences that would rewrite working code
    'declaration-block-no-redundant-longhand-properties': null,
    'alpha-value-notation': null,
    'color-function-notation': null,
    'media-feature-range-notation': null,
    'value-keyword-case': null,
    'shorthand-property-no-redundant-values': null,
  },

  ignoreFiles: ['dist/**', 'node_modules/**', 'public/**', '.astro/**', 'paper/**'],
};
