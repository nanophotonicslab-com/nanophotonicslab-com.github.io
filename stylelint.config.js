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
    /* Hard error: every colour in the codebase now has a name in global.css,
     * including the status and spectral-band scales. A hex literal here is
     * either a duplicate of an existing token or a colour that needs adding to
     * the palette — both are worth stopping for. */
    'color-no-hex': true,

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
