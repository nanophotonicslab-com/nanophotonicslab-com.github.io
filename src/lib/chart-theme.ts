/**
 * Read current theme colors from CSS variables for Canvas 2D charts.
 * Call at the start of each draw cycle — fast (one getComputedStyle).
 */
export interface ChartTheme {
  bg: string;
  text: string;
  textSoft: string;
  textMuted: string;
  grid: string;
  primary: string;
}

export function chartTheme(): ChartTheme {
  const s = getComputedStyle(document.documentElement);
  return {
    bg:        s.getPropertyValue('--bg').trim()        || '#FFFFFF',
    text:      s.getPropertyValue('--text').trim()      || '#1E1B4B',
    textSoft:  s.getPropertyValue('--text-soft').trim() || '#64648B',
    textMuted: s.getPropertyValue('--text-muted').trim()|| '#9CA3AF',
    grid:      s.getPropertyValue('--bg-muted').trim()  || '#F1F0FB',
    primary:   s.getPropertyValue('--primary').trim()   || '#6366F1',
  };
}

/** Listen for theme changes and trigger a callback. */
export function onThemeChange(cb: () => void): void {
  window.addEventListener('themechange', cb);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', cb);
}
