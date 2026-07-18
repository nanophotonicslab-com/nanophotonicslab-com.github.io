/**
 * @nanophotonicslab/optics — the physics engines behind nanophotonicslab.com,
 * as a typed, dependency-free ES module library.
 *
 * Namespaced exports (modules share physical names like csqrt or interpolateNK,
 * so each solver keeps its own namespace):
 *
 *   import { mie, purcell, pulse } from '@nanophotonicslab/optics';
 *   const { csca } = mie.mieAt(1.5, 0, 1.33, 60, 550);
 */
export * as complex from '../../src/lib/complex';
export * as mie from '../../src/lib/mie';
export * as spectrum from '../../src/lib/spectrum';
export * as materials from '../../src/lib/materials';
export * as colorimetry from '../../src/lib/colorimetry';
export * as graph from '../../src/lib/graph';
export * as photothermal from '../../src/lib/photothermal';
export * as opticalForces from '../../src/lib/optical-forces';
export * as purcell from '../../src/lib/purcell';
export * as pulse from '../../src/lib/pulse';
export * as fit from '../../src/lib/fit';
export * as cylinder from '../../src/lib/cylinder';
export * as electronSphere from '../../src/lib/electron-sphere';
export * as dipoleDecay from '../../src/lib/dipole-decay';
export * as dipoleInside from '../../src/lib/dipole-inside';
export * as plasmonic from '../../src/lib/plasmonic-nanoparticles';
export * as smithPurcell from '../../src/lib/smith-purcell';
export * as vo2 from '../../src/lib/vo2';
export * as nlGraphene from '../../src/lib/nl-graphene';
export * as grapheneConductivity from '../../src/lib/graphene-conductivity';

export const VERSION = '0.1.0';
