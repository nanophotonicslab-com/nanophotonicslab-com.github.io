/**
 * Purcell-factor layer over the dipole-decay engine (Kim 1988), aimed at
 * quantum emitters near a sphere: orientation handling, intrinsic-quantum-yield
 * bookkeeping, and lifetime modification.
 *
 * Rate naming follows this codebase's Kim mapping (see mie-scattering dipole
 * mode): gammaPar* = RADIAL dipole (⊥ to the surface), gammaPerp* = TANGENTIAL.
 * All factors are normalized to the free-space radiative rate γ₀.
 *
 * Emitter bookkeeping (Bharadwaj & Novotny convention): an emitter with
 * intrinsic quantum yield q₀ keeps its internal non-radiative channel, so
 *   γ_new = γ₀·(q₀·F_tot + 1 − q₀)   →   τ' = τ₀ / (q₀·F_tot + 1 − q₀)
 *   q' = q₀·F_rad / (q₀·F_tot + 1 − q₀)
 */
import { decayRatesAt, type Cx } from './dipole-decay';

export type DipoleOrientation = 'radial' | 'tangential' | 'isotropic';

export interface PurcellFactors {
  /** Total decay enhancement Γ/Γ₀ (the Purcell factor). */
  fTot: number;
  /** Radiative part Γ_rad/Γ₀. */
  fRad: number;
  /** Non-radiative (absorbed in the sphere) part. */
  fNr: number;
}

/**
 * Purcell factors for an emitter at gap `gapNm` from the SURFACE of a sphere.
 */
export function purcellAt(
  lambdaNm: number, epsSphere: Cx, nHost: number, radiusNm: number,
  gapNm: number, orientation: DipoleOrientation, lmax = 20,
): PurcellFactors {
  const r = decayRatesAt(lambdaNm, epsSphere, nHost, radiusNm, radiusNm + gapNm, lmax);
  const radial = { fTot: r.gammaPar, fRad: r.gammaParRad, fNr: r.gammaParNr };
  const tangential = { fTot: r.gammaPerp, fRad: r.gammaPerpRad, fNr: r.gammaPerpNr };
  if (orientation === 'radial') return radial;
  if (orientation === 'tangential') return tangential;
  return {
    fTot: (radial.fTot + 2 * tangential.fTot) / 3,
    fRad: (radial.fRad + 2 * tangential.fRad) / 3,
    fNr: (radial.fNr + 2 * tangential.fNr) / 3,
  };
}

/** Quantum yield of the coupled system given the emitter's intrinsic q₀. */
export function modifiedQuantumYield(f: PurcellFactors, q0: number): number {
  const den = q0 * f.fTot + (1 - q0);
  return den > 0 ? (q0 * f.fRad) / den : 0;
}

/** Excited-state lifetime of the coupled system, same units as tau0. */
export function modifiedLifetime(f: PurcellFactors, q0: number, tau0: number): number {
  const den = q0 * f.fTot + (1 - q0);
  return den > 0 ? tau0 / den : Infinity;
}
