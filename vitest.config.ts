import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    // These suites are compute-bound physics, not I/O: several individual cases
    // legitimately run for seconds (Mie sweeps, electron-sphere spectra, a
    // 100-frame imaging analysis). Vitest's 5 s default made them fail purely
    // from load when the whole suite runs together.
    testTimeout: 30000,
  },
});
