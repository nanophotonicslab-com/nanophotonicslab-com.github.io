import { describe, it, expect } from 'vitest';
import {
  betaFromKeV, gammaLorentz, wavelengthToEv, evToWavelength,
  interfaceAmplitude, interfaceSrPerNm, interfaceSpectrum, perfectConductorSrPerNm,
  filmSrPerNm, filmSpectrum, filmAmplitude, filmGammaOverDelta, thicknessMap,
  thickFilmLimitSrPerNm, thinFilmLimitSrPerNm,
  interferencePhase, resonanceThicknessNm,
  cherenkovThresholdKeV, cherenkovAngleRad, tirAngleRad, cherenkovEscapes,
  type C, type TrSide,
} from './transition-radiation';
import fixture from './transition-radiation.fixture.json';

/**
 * Parity of the TypeScript port against trlib (the TR_CL project), the
 * validated Python implementation of Ritchie & Eldridge 1962, García de Abajo
 * RMP 2010, Brenny's thesis 2016 and Ebel et al. 2026.
 *
 * Every fixture case carries the permittivity trlib used, so these tests feed
 * the port those exact ε values: what is checked is the physics, not either
 * side's optical-constant tables.
 *
 * Tolerances. A faithful port with the same Gauss–Legendre rule reaches
 * round-off, but the fixture is not exact: it stores 12 significant digits, so
 * a *perfect* port could still differ by 5e-13, and the ε it hands the port is
 * itself rounded to 12 digits while trlib computed with the full value. Most
 * blocks therefore sit at 1e-12 or below and are asserted at `TOL` = 1e-9.
 *
 * Two kinds of block legitimately need more:
 *   - the per-angle *film* values (`FILM_TOL` = 1e-8). A thick, high-index,
 *     low-loss film has sharp Fabry–Pérot fringes, and near a fringe the
 *     12-digit ε rounding is amplified: Si, 100 nm, 10 keV, θ = 80° reaches
 *     2e-10, a hundred times the other film blocks. Nothing in the port is
 *     less accurate there — the denominator Δ′ is simply small.
 *   - the *physics* cross-checks, which compare two different formalisms or an
 *     asymptotic expansion against the exact result, and carry their own
 *     tolerance and comment at each use (1e-9 for Ritchie's thick foil vs
 *     García de Abajo's interface — trlib asserts 1e-8 for the same pair — and
 *     5 % for Ritchie's Eq. 24 thin-film expansion).
 *
 * The worst deviation actually seen per block is printed at the end of the run.
 */
const TOL = 1e-9;
const FILM_TOL = 1e-8;

const DEG = Math.PI / 180;
const worst = new Map<string, { rel: number; at: string }>();

/** Largest relative deviation of `got` from `want`, recorded under `block`. */
function maxRel(block: string, got: ArrayLike<number>, want: ArrayLike<number>, at = ''): number {
  expect(got.length, `${block}: length`).toBe(want.length);
  let m = 0;
  let where = at;
  for (let i = 0; i < want.length; i++) {
    expect(Number.isFinite(got[i]), `${block}${at}[${i}] is not finite`).toBe(true);
    const w = want[i];
    const r = w === 0 ? Math.abs(got[i]) : Math.abs(got[i] - w) / Math.abs(w);
    if (r > m) { m = r; where = `${at}[${i}]`; }
  }
  const prev = worst.get(block);
  if (!prev || m > prev.rel) worst.set(block, { rel: m, at: where });
  return m;
}

function expectClose(block: string, got: ArrayLike<number>, want: ArrayLike<number>, tol = TOL, at = ''): void {
  expect(maxRel(block, got, want, at), `${block}${at}: worst relative deviation`).toBeLessThan(tol);
}

const epsOf = (c: { epsRe: number[]; epsIm: number[] }): C[] =>
  c.epsRe.map((re, i): C => [re, c.epsIm[i]]);

type Case = {
  id: string; kind: 'interface' | 'film'; material: string; energyKeV: number;
  side: string; thicknessNm?: number;
  wavelengthNm: number[]; epsRe: number[]; epsIm: number[]; thetaDeg: number[];
  srPerNm: number[][]; perNm: number[]; perNmCone60: number[]; perEv: number[];
  srPerNmB1B3?: number[][];
};

const cases = fixture.cases as unknown as Case[];

// ── 1. kinematics ──────────────────────────────────────────────────────────

describe('electron kinematics', () => {
  it('reproduces beta(keV) from the fixture', () => {
    const got = fixture.kinematics.map(k => betaFromKeV(k.energyKeV));
    const want = fixture.kinematics.map(k => k.beta);
    expectClose('kinematics: beta', got, want, 1e-11);
  });

  it('gamma and beta are consistent, and the photon-energy conversions invert', () => {
    for (const { energyKeV } of fixture.kinematics) {
      const g = gammaLorentz(energyKeV);
      const b = betaFromKeV(energyKeV);
      expect(g * Math.sqrt(1 - b * b)).toBeCloseTo(1, 12);
    }
    expect(wavelengthToEv(evToWavelength(2.5))).toBeCloseTo(2.5, 12);
    expect(wavelengthToEv(400)).toBeCloseTo(3.0996049608, 9);
  });
});

// ── 2. single interface (García de Abajo / Brenny) ─────────────────────────

describe('single interface vs trlib', () => {
  for (const c of cases.filter(x => x.kind === 'interface')) {
    const eps = epsOf(c);
    const side = c.side as TrSide;

    it(`${c.id}: dΓ/dΩ per nm at 5 angles × 21 wavelengths`, () => {
      c.thetaDeg.forEach((td, ti) => {
        const got = c.wavelengthNm.map((lam, li) =>
          interfaceSrPerNm(td * DEG, lam, c.energyKeV, eps[li], side));
        expectClose('interface: srPerNm', got, c.srPerNm[ti], TOL, ` ${c.id} θ=${td}°`);
      });
    });

    it(`${c.id}: angle-integrated spectra (hemisphere, 60° cone, per eV)`, () => {
      const base = { wavelengthNm: c.wavelengthNm, eps, energyKeV: c.energyKeV, side };
      expectClose('interface: perNm',
        interfaceSpectrum(base), c.perNm, TOL, ` ${c.id}`);
      expectClose('interface: perNmCone60',
        interfaceSpectrum({ ...base, thetaMaxRad: 60 * DEG }), c.perNmCone60, TOL, ` ${c.id}`);
      expectClose('interface: perEv',
        interfaceSpectrum({ ...base, per: 'eV' }), c.perEv, TOL, ` ${c.id}`);
    });
  }

  it('the amplitude is wavelength-independent and reproduces |f|²/(4π²ħk)', () => {
    // The closed form cancels k, so only ε (and hence λ) enters; the stored
    // per-sr values at two wavelengths must follow from the same amplitude law.
    const c = cases[0];
    const eps = epsOf(c);
    const f = interfaceAmplitude(40 * DEG, c.energyKeV, eps[0], 'backward');
    expect(Number.isFinite(f[0]) && Number.isFinite(f[1])).toBe(true);
    const direct = interfaceSrPerNm(40 * DEG, c.wavelengthNm[0], c.energyKeV, eps[0]);
    const k = 2 * Math.PI / (c.wavelengthNm[0] * 1e-7);
    const pOmega = (f[0] * f[0] + f[1] * f[1]) / (4 * Math.PI ** 2 * 1.054571817e-27 * k);
    const perNm = pOmega * 2 * Math.PI * 299792458 / (c.wavelengthNm[0] * 1e-9) ** 2 * 1e-9;
    expect(Math.abs(direct / perNm - 1)).toBeLessThan(1e-14);
  });
});

// ── 3. Ginzburg–Frank perfect-conductor limit ──────────────────────────────

describe('Ginzburg–Frank perfect conductor', () => {
  it('reproduces the closed-form αβ²sin²θ / (π²ω(1−β²cos²θ)²)', () => {
    const pc = fixture.perfectConductor;
    pc.thetaDeg.forEach((td, ti) => {
      const got = pc.wavelengthNm.map(lam => perfectConductorSrPerNm(td * DEG, lam, pc.energyKeV));
      expectClose('perfect conductor: srPerNm', got, pc.srPerNm[ti], TOL, ` θ=${td}°`);
    });
  });

  it('a very large |ε| metal approaches the perfect conductor', () => {
    const lam = 600;
    const keV = 30;
    const pec: C = [-1e8, 1e6];
    for (const td of [20, 40, 60, 80]) {
      const metal = interfaceSrPerNm(td * DEG, lam, keV, pec);
      const gf = perfectConductorSrPerNm(td * DEG, lam, keV);
      expect(Math.abs(metal / gf - 1)).toBeLessThan(2e-3);
    }
  });
});

// ── 4. free-standing film (Ritchie & Eldridge / Ebel) ──────────────────────

describe('free-standing film vs trlib', () => {
  for (const c of cases.filter(x => x.kind === 'film')) {
    const eps = epsOf(c);
    const side = c.side as TrSide;
    const d = c.thicknessNm as number;

    it(`${c.id}: dΓ/dΩ per nm at 5 angles × 21 wavelengths`, () => {
      c.thetaDeg.forEach((td, ti) => {
        const got = c.wavelengthNm.map((lam, li) =>
          filmSrPerNm(td * DEG, lam, d, c.energyKeV, eps[li], side));
        expectClose('film: srPerNm', got, c.srPerNm[ti], FILM_TOL, ` ${c.id} θ=${td}°`);
      });
    });

    it(`${c.id}: B1+B3 interface-only control (Ebel Fig. S1)`, () => {
      c.thetaDeg.forEach((td, ti) => {
        const got = c.wavelengthNm.map((lam, li) =>
          filmSrPerNm(td * DEG, lam, d, c.energyKeV, eps[li], side, { terms: ['B1', 'B3'] }));
        expectClose('film: srPerNmB1B3', got, (c.srPerNmB1B3 as number[][])[ti], FILM_TOL, ` ${c.id} θ=${td}°`);
      });
    });

    it(`${c.id}: angle-integrated spectra (hemisphere, 60° cone, per eV)`, () => {
      const base = {
        wavelengthNm: c.wavelengthNm, eps, thicknessNm: d, energyKeV: c.energyKeV, side,
      };
      expectClose('film: perNm', filmSpectrum(base), c.perNm, TOL, ` ${c.id}`);
      expectClose('film: perNmCone60',
        filmSpectrum({ ...base, thetaMaxRad: 60 * DEG }), c.perNmCone60, TOL, ` ${c.id}`);
      expectClose('film: perEv',
        filmSpectrum({ ...base, per: 'eV' }), c.perEv, TOL, ` ${c.id}`);
    });

    it(`${c.id}: thicknessMap row equals the single-thickness spectrum`, () => {
      const base = { wavelengthNm: c.wavelengthNm, eps, energyKeV: c.energyKeV, side };
      const grid = thicknessMap({ ...base, thicknessNm: [d / 2, d, 2 * d] });
      const n = c.wavelengthNm.length;
      expectClose('film: thicknessMap', grid.subarray(n, 2 * n), c.perNm, TOL, ` ${c.id}`);
    });
  }
});

// ── 5. limits (fixture `limits`, Ag at 30 keV) ─────────────────────────────

describe('film limits', () => {
  const L = fixture.limits;
  const eps = epsOf(L);

  it('thick-foil limit matches trlib', () => {
    L.thetaDeg.forEach((td, ti) => {
      const got = L.wavelengthNm.map((lam, li) =>
        thickFilmLimitSrPerNm(td * DEG, lam, L.energyKeV, eps[li], 'backward'));
      expectClose('limits: thickFilm', got, L.thickFilmSrPerNm[ti], TOL, ` θ=${td}°`);
    });
  });

  it('single-interface reference matches trlib', () => {
    L.thetaDeg.forEach((td, ti) => {
      const got = L.wavelengthNm.map((lam, li) =>
        interfaceSrPerNm(td * DEG, lam, L.energyKeV, eps[li], 'backward'));
      expectClose('limits: interface', got, L.interfaceSrPerNm[ti], TOL, ` θ=${td}°`);
    });
  });

  it('thick foil = single interface — two independent formalisms', () => {
    // Ritchie's Born-approximation foil vs García de Abajo's exact interface:
    // the same number out of two formalisms that share no algebra. trlib
    // asserts 1e-8 for this pair; the port reaches ~7e-11 on the same ε.
    L.thetaDeg.forEach((td, ti) => {
      const a = L.wavelengthNm.map((lam, li) =>
        thickFilmLimitSrPerNm(td * DEG, lam, L.energyKeV, eps[li], 'backward'));
      const b = L.wavelengthNm.map((lam, li) =>
        interfaceSrPerNm(td * DEG, lam, L.energyKeV, eps[li], 'backward'));
      expectClose('cross-check: thick foil ↔ interface', a, b, 1e-9, ` θ=${td}°`);
      expectClose('cross-check: thick foil ↔ interface', a, L.interfaceSrPerNm[ti], 1e-9, ` θ=${td}°`);
    });
  });

  it('thin-film expansion matches trlib', () => {
    L.thetaDeg.forEach((td, ti) => {
      const got = L.wavelengthNm.map((lam, li) =>
        thinFilmLimitSrPerNm(td * DEG, lam, 2, L.energyKeV, eps[li]));
      expectClose('limits: thinFilm', got, L.thinFilmSrPerNm[ti], TOL, ` θ=${td}°`);
    });
  });

  it('the real 2 nm film matches trlib', () => {
    L.thetaDeg.forEach((td, ti) => {
      const got = L.wavelengthNm.map((lam, li) =>
        filmSrPerNm(td * DEG, lam, 2, L.energyKeV, eps[li], 'backward'));
      expectClose('limits: film at 2 nm', got, L.filmAt2nmSrPerNm[ti], TOL, ` θ=${td}°`);
    });
  });

  it('Ritchie Eq. 24 reproduces a real film inside its regime of validity', () => {
    // Physics, not parity. The expansion needs ωa/v ≪ 1 *and* |ε|^½ωa/c ≪ 1, so
    // it is checked where trlib checks it: a weak dielectric, half a nanometre.
    // (The fixture's Ag-at-2-nm pair above is only a parity check — plasmonic
    // Ag has Re ε from −4 to −41 over 400–900 nm and the expansion is 1.4× to
    // 250× off there, in Python exactly as here.)
    const eps3: C = [3, 1];
    for (const td of [5, 20, 40, 60, 80]) {
      const exact = filmSrPerNm(td * DEG, 600, 0.5, 30, eps3, 'forward');
      const approx = thinFilmLimitSrPerNm(td * DEG, 600, 0.5, 30, eps3);
      expect(Math.abs(approx / exact - 1), `Eq. 24 at θ=${td}°`).toBeLessThan(0.05);
    }
  });

  it('a thick real film converges onto the thick-foil limit', () => {
    for (const td of [20, 40, 60, 80]) {
      for (let li = 0; li < eps.length; li += 5) {
        const lam = L.wavelengthNm[li];
        const thick = filmSrPerNm(td * DEG, lam, 3000, L.energyKeV, eps[li], 'backward');
        const limit = thickFilmLimitSrPerNm(td * DEG, lam, L.energyKeV, eps[li], 'backward');
        expect(Math.abs(thick / limit - 1), `d=3000 nm θ=${td}° λ=${lam}`).toBeLessThan(1e-9);
      }
    }
  });
});

// ── 6. resonances (Ebel Eq. 3) ─────────────────────────────────────────────

describe('thin-film resonances', () => {
  const R = fixture.resonance;
  const eps = epsOf(R);
  const theta = R.thetaDeg * DEG;

  it('resonance thickness d_m(λ) for m = 1, 2, 3', () => {
    for (const m of [1, 2, 3] as const) {
      const got = R.wavelengthNm.map((lam, li) =>
        resonanceThicknessNm(lam, R.energyKeV, eps[li], m, theta));
      expectClose('resonance: thickness', got,
        (R.thicknessNmM as unknown as Record<string, number[]>)[String(m)], TOL, ` m=${m}`);
    }
  });

  it('interference phase of an 83 nm film', () => {
    const got = R.wavelengthNm.map((lam, li) =>
      interferencePhase(lam, 83, R.energyKeV, eps[li], theta));
    expectClose('resonance: phase', got, R.phaseAt83nm, TOL);
  });

  it('the resonance thickness is exactly where the phase hits 2π(m − ½)', () => {
    for (const m of [1, 2, 3] as const) {
      R.wavelengthNm.forEach((lam, li) => {
        const d = resonanceThicknessNm(lam, R.energyKeV, eps[li], m, theta);
        const phi = interferencePhase(lam, d, R.energyKeV, eps[li], theta);
        expect(Math.abs(phi / (2 * Math.PI * (m - 0.5)) - 1)).toBeLessThan(1e-12);
      });
    }
  });
});

// ── 7. Cherenkov and the escape cone ───────────────────────────────────────

describe('Cherenkov / TIR criteria', () => {
  // Two fixture entries: Si-like ε = 12.25 probed at 30 keV (trapped by TIR)
  // and a glass-like ε = 2.25 probed at 200 keV (escapes).
  type CherenkovCase = {
    epsRe: number; thresholdKeV: number; tirDeg: number;
    angleDeg30keV?: number; angleDeg200keV?: number;
    escapes30keV?: boolean; escapes200keV?: boolean;
  };
  const [si, glass] = fixture.cherenkov as unknown as CherenkovCase[];
  const entries = [
    { keV: 30, c: si, angleDeg: si.angleDeg30keV as number, escapes: si.escapes30keV as boolean },
    { keV: 200, c: glass, angleDeg: glass.angleDeg200keV as number, escapes: glass.escapes200keV as boolean },
  ];

  it('reproduces the thresholds, angles and escape verdicts', () => {
    for (const { keV, c, angleDeg, escapes } of entries) {
      const tag = ` ε=${c.epsRe}`;
      expectClose('cherenkov: threshold',
        [cherenkovThresholdKeV(c.epsRe)], [c.thresholdKeV], TOL, tag);
      expectClose('cherenkov: TIR angle', [tirAngleRad(c.epsRe) / DEG], [c.tirDeg], TOL, tag);
      expectClose('cherenkov: angle', [cherenkovAngleRad(keV, c.epsRe) / DEG], [angleDeg], TOL, tag);
      expect(cherenkovEscapes(keV, c.epsRe)).toBe(escapes);
    }
  });

  it('below threshold the angle is NaN and nothing escapes', () => {
    expect(Number.isNaN(cherenkovAngleRad(5, 12.25))).toBe(true);
    expect(cherenkovEscapes(5, 12.25)).toBe(false);
    expect(betaFromKeV(cherenkovThresholdKeV(12.25)) * Math.sqrt(12.25)).toBeCloseTo(1, 12);
  });
});

// ── 8. invariants that do not come from the fixture ────────────────────────

describe('physical invariants', () => {
  const vacuum: C = [1, 0];
  const ag: C = [fixture.limits.epsRe[10], fixture.limits.epsIm[10]];

  it('ε = 1 radiates nothing (interface and film, both sides)', () => {
    // Both amplitudes cancel identically at ε = 1 — B₁ = B₂ = B₃ = 0 for the
    // film, and the two bracket terms coincide for the interface — so what is
    // left is the round-off of that cancellation, ~1e-40 against a typical
    // 1e-9 signal. trlib asserts the same bound for the same reason.
    for (const side of ['backward', 'forward'] as TrSide[]) {
      for (const td of [5, 30, 60, 85]) {
        expect(interfaceSrPerNm(td * DEG, 500, 30, vacuum, side)).toBeLessThan(1e-30);
        for (const d of [1, 50, 1000]) {
          expect(filmSrPerNm(td * DEG, 500, d, 30, vacuum, side)).toBeLessThan(1e-30);
        }
      }
    }
  });

  it('a film of zero thickness radiates nothing', () => {
    for (const side of ['backward', 'forward'] as TrSide[]) {
      for (const td of [5, 30, 60, 85]) {
        for (const eps of [ag, [12.25, 0] as C, [2.1, 3.4] as C]) {
          expect(Math.abs(filmSrPerNm(td * DEG, 500, 0, 30, eps, side))).toBeLessThan(1e-28);
        }
      }
    }
  });

  it('emission is non-negative and finite everywhere, including θ = 0 and θ = π/2', () => {
    for (const td of [0, 1e-9, 5, 45, 89.999, 90]) {
      for (const eps of [ag, [12.25, 0] as C, [-40, 2] as C, [0.5, 0.01] as C]) {
        for (const side of ['backward', 'forward'] as TrSide[]) {
          const i = interfaceSrPerNm(td * DEG, 500, 30, eps, side);
          const f = filmSrPerNm(td * DEG, 500, 80, 30, eps, side);
          expect(Number.isFinite(i) && i >= 0, `interface θ=${td} ε=${eps}`).toBe(true);
          expect(Number.isFinite(f) && f >= 0, `film θ=${td} ε=${eps}`).toBe(true);
        }
      }
    }
  });

  it('no overflow for a thick lossy film (Ag, 5000 nm)', () => {
    for (const td of [5, 40, 80]) {
      for (let li = 0; li < fixture.limits.wavelengthNm.length; li++) {
        const eps: C = [fixture.limits.epsRe[li], fixture.limits.epsIm[li]];
        const v = filmSrPerNm(td * DEG, fixture.limits.wavelengthNm[li], 5000, 30, eps, 'backward');
        expect(Number.isFinite(v) && v >= 0).toBe(true);
      }
    }
    // the stabilised exponentials must still be finite for an absurd thickness
    const v = filmSrPerNm(40 * DEG, 500, 1e6, 30, [-40, 2], 'backward');
    expect(Number.isFinite(v) && v >= 0).toBe(true);
  });

  it('the forward side is the backward one with b → −b', () => {
    // γ, Δ and the thick-foil limit depend on the velocity only through the
    // signed b, so reversing the electron maps one side onto the other. The
    // film amplitude |a|² must equal the per-sr probability on each side.
    for (const td of [10, 35, 70]) {
      for (const eps of [ag, [12.25, 0] as C, [2.1, 3.4] as C]) {
        for (const side of ['backward', 'forward'] as TrSide[]) {
          const a = filmAmplitude(td * DEG, 620, 90, 30, eps, side);
          const p = filmSrPerNm(td * DEG, 620, 90, 30, eps, side);
          expect(Math.abs((a[0] * a[0] + a[1] * a[1]) / p - 1)).toBeLessThan(1e-12);
        }
        // d → ∞ on either side reproduces that side's thick-foil limit
        for (const side of ['backward', 'forward'] as TrSide[]) {
          const thick = filmSrPerNm(td * DEG, 620, 4000, 30, [-40, 2], side);
          const limit = thickFilmLimitSrPerNm(td * DEG, 620, 30, [-40, 2], side);
          expect(Math.abs(thick / limit - 1)).toBeLessThan(1e-9);
        }
      }
    }
  });

  it('dropping every term leaves nothing, and B1+B2+B3 is the default', () => {
    const eps: C = [-12, 1.1];
    expect(filmSrPerNm(40 * DEG, 600, 70, 30, eps, 'backward', { terms: [] })).toBe(0);
    const all = filmSrPerNm(40 * DEG, 600, 70, 30, eps, 'backward');
    const spelled = filmSrPerNm(40 * DEG, 600, 70, 30, eps, 'backward', { terms: ['B1', 'B2', 'B3'] });
    expect(spelled).toBe(all);
    // flipping the sign of B3 changes the interference, not the magnitudes
    const flipped = filmSrPerNm(40 * DEG, 600, 70, 30, eps, 'backward', { b3Sign: -1 });
    expect(flipped).not.toBe(all);
    expect(Number.isFinite(flipped) && flipped > 0).toBe(true);
  });

  it('γ decomposes exactly into B1 + B2 + B3', () => {
    const args = [40 * DEG, 550, 120, 30, [25, 3] as C, 'backward' as TrSide] as const;
    const full = filmGammaOverDelta(...args);
    const parts = (['B1', 'B2', 'B3'] as const).map(t => filmGammaOverDelta(...args, { terms: [t] }));
    for (const k of [0, 1]) {
      expect(Math.abs((parts[0][k] + parts[1][k] + parts[2][k]) / full[k] - 1)).toBeLessThan(1e-12);
    }
  });

  it('a non-relativistic ε = −1 foil gives Ritchie Eq. 20, 2μ²(1 − μ²)', () => {
    const keV = 1e-5;                       // β ≈ 6e-3
    const beta = betaFromKeV(keV);
    const norm = 7.2973525693e-3 * beta * beta / (Math.PI ** 2 * 500);
    for (const td of [5, 20, 40, 60, 80]) {
      const mu = Math.cos(td * DEG);
      const got = thickFilmLimitSrPerNm(td * DEG, 500, keV, [-1, 1e-9], 'forward') / norm;
      expect(Math.abs(got / (2 * mu * mu * (1 - mu * mu)) - 1), `θ=${td}°`).toBeLessThan(0.02);
    }
  });

  it('a scalar ε, a per-wavelength list and a function agree', () => {
    const wl = [400, 500, 600];
    const e: C = [-12, 1.1];
    const a = filmSpectrum({ wavelengthNm: wl, eps: e, thicknessNm: 60, energyKeV: 30 });
    const b = filmSpectrum({ wavelengthNm: wl, eps: [e, e, e], thicknessNm: 60, energyKeV: 30 });
    const c = filmSpectrum({ wavelengthNm: wl, eps: () => e, thicknessNm: 60, energyKeV: 30 });
    expectClose('eps plumbing', b, a, 1e-15);
    expectClose('eps plumbing', c, a, 1e-15);
    expect(() => filmSpectrum({ wavelengthNm: wl, eps: [e], thicknessNm: 60, energyKeV: 30 }))
      .toThrow(/1 entries but 3 wavelengths/);
  });
});

// ── 9. performance (a note, not an assertion) ──────────────────────────────

describe('performance', () => {
  it('computes an interactive spectrum and thickness map', () => {
    const nL = 400;
    const wl = Array.from({ length: nL }, (_, i) => 350 + i * (550 / (nL - 1)));
    const eps = wl.map((l): C => [-0.6 - 1.1e-4 * (l - 350) ** 2, 0.2 + 1e-3 * (l - 350)]);

    let t0 = performance.now();
    const spec = filmSpectrum({ wavelengthNm: wl, eps, thicknessNm: 80, energyKeV: 30 });
    const tSpec = performance.now() - t0;

    const nD = 150;
    const wl2 = Array.from({ length: 200 }, (_, i) => 350 + i * (550 / 199));
    const eps2 = wl2.map((l): C => [-0.6 - 1.1e-4 * (l - 350) ** 2, 0.2 + 1e-3 * (l - 350)]);
    const d = Array.from({ length: nD }, (_, i) => 1 + i * (300 / (nD - 1)));
    t0 = performance.now();
    const grid = thicknessMap({ wavelengthNm: wl2, eps: eps2, thicknessNm: d, energyKeV: 30 });
    const tMap = performance.now() - t0;

    expect(spec.length).toBe(nL);
    expect(grid.length).toBe(nD * wl2.length);
    expect(spec.every(Number.isFinite)).toBe(true);
    expect(grid.every(Number.isFinite)).toBe(true);
    // Rough timings, printed rather than asserted. The spectrum is 38 k inner
    // points and the map 2.9 M; the inner loop is three transcendentals
    // (exp, cos, sin) plus ~30 flops on flat Float64Arrays, with the
    // thickness-independent (θ, λ) coefficients built once and held in
    // registers across the whole thickness sweep. Measured on the dev machine
    // (Node 22, otherwise idle): spectrum ~17 ms, map ~450 ms, i.e. ~150 ns
    // per point — but that box is slow at transcendentals (Math.sin alone
    // ~150 ns), and under load the map reached 2–4 s, so expect a wide spread.
    // A further ~6× is available for a *uniformly* spaced thickness grid by
    // stepping e^{itσ} multiplicatively instead of calling exp/cos/sin per
    // point (measured: 49 ns/point), at the cost of ~1e-12 relative drift —
    // not done here, because every value this module returns is currently
    // bit-identical to the exact per-point evaluation.
    console.log(`  spectrum 400λ×96θ: ${tSpec.toFixed(1)} ms   `
      + `thickness map 150d×200λ×96θ: ${tMap.toFixed(1)} ms`);
  });
});

// ── worst observed deviation per block ─────────────────────────────────────

describe('parity summary', () => {
  it('reports the worst relative deviation of every block', () => {
    const rows = [...worst.entries()].sort((a, b) => b[1].rel - a[1].rel);
    console.log('\n  worst relative deviation vs trlib\n'
      + rows.map(([k, v]) => `    ${v.rel.toExponential(2).padStart(9)}  ${k}${v.at}`).join('\n'));
    expect(rows.length).toBeGreaterThan(0);
  });
});
