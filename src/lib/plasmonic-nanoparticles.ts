import { MATERIALS, interpolateNK } from './materials';

export type PlasmonShapeId =
  | 'rod'
  | 'triangle'
  | 'cage'
  | 'ellipsoid'
  | 'bicone'
  | 'disk'
  | 'ring'
  | 'bipyramid'
  | 'squared rod'
  | 'cylinder'
  | 'tetrahedron'
  | 'octahedron'
  | 'decahedron'
  | 'rod transverse';

export interface ShapeOption {
  id: PlasmonShapeId;
  label: string;
  defaultMaterialId?: string;
  defaultLengthNm: number;
  defaultR: number;
  defaultLambdaMinNm?: number;
  defaultLambdaMaxNm?: number;
  minR: number;
  maxR: number;
}

export interface ShapeMode {
  epsilonJ: number;
  vjOverV: number;
  a2: number;
  a4: number;
  vOverL3: number;
  shape: PlasmonShapeId;
  modeIndex: number;
}

export interface Complex {
  re: number;
  im: number;
}

export interface PlasmonSpectrumOptions {
  shape: PlasmonShapeId;
  materialId: string;
  epsilonAt?: (lambdaNm: number) => Complex;
  lengthNm: number;
  aspectRatio: number;
  hostIndex: number;
  lambdaMinNm: number;
  lambdaMaxNm: number;
  points?: number;
  modeCount?: number;
}

export interface PlasmonPeakMetrics {
  lambdaPeakNm: number;
  peakValue: number;
  leftHalfNm: number;
  rightHalfNm: number;
  fwhmNm: number;
  q: number;
}

export interface PlasmonSpectrum {
  wavelengthNm: Float64Array;
  alpha: Complex[];
  sigmaExtNm2: Float64Array;
  sigmaScaNm2: Float64Array;
  sigmaAbsNm2: Float64Array;
  sigmaExtOverV: Float64Array;
  sigmaAbsOverV: Float64Array;
  sigmaScaOverV: Float64Array;
  quantumYield: Float64Array;
  volumeNm3: number;
  modes: ShapeMode[];
  peak: PlasmonPeakMetrics;
}

export interface ShapeParameterCurves {
  r: Float64Array;
  epsilonJ: Float64Array;
  vjOverV: Float64Array;
  a2: Float64Array;
  a4: Float64Array;
  vOverL3: Float64Array;
  vjOverL3: Float64Array;
}

export const PLASMONIC_MATERIAL_IDS = ['au', 'ag', 'cu', 'al'] as const;

export const SHAPE_OPTIONS: ShapeOption[] = [
  { id: 'rod', label: 'Horizontal rod', defaultMaterialId: 'au', defaultLengthNm: 50, defaultR: 4, defaultLambdaMinNm: 550, defaultLambdaMaxNm: 1250, minR: 1.2, maxR: 8 },
  { id: 'rod transverse', label: 'Vertical rod', defaultMaterialId: 'ag', defaultLengthNm: 50, defaultR: 4, defaultLambdaMinNm: 320, defaultLambdaMaxNm: 440, minR: 1.2, maxR: 8 },
  { id: 'triangle', label: 'Triangular prism', defaultLengthNm: 70, defaultR: 3.5, minR: 1.2, maxR: 8 },
  { id: 'cage', label: 'Nanocage', defaultLengthNm: 90, defaultR: 3.0, minR: 1.2, maxR: 8 },
  { id: 'ellipsoid', label: 'Ellipsoid', defaultLengthNm: 80, defaultR: 4.0, minR: 1.2, maxR: 8 },
  { id: 'bicone', label: 'Bicone', defaultLengthNm: 80, defaultR: 3.5, minR: 1.2, maxR: 8 },
  { id: 'disk', label: 'Disk', defaultLengthNm: 100, defaultR: 3.0, minR: 1.2, maxR: 8 },
  { id: 'ring', label: 'Ring', defaultLengthNm: 100, defaultR: 3.0, minR: 1.2, maxR: 8 },
  { id: 'bipyramid', label: 'Bipyramid', defaultMaterialId: 'au', defaultLengthNm: 200, defaultR: 5, defaultLambdaMinNm: 500, defaultLambdaMaxNm: 1500, minR: 1.2, maxR: 8 },
  { id: 'squared rod', label: 'Squared rod', defaultLengthNm: 80, defaultR: 4.0, minR: 1.2, maxR: 8 },
  { id: 'cylinder', label: 'Cylinder', defaultLengthNm: 80, defaultR: 4.0, minR: 1.2, maxR: 8 },
  { id: 'tetrahedron', label: 'Tetrahedron', defaultLengthNm: 80, defaultR: 1.0, minR: 1, maxR: 1 },
  { id: 'octahedron', label: 'Octahedron', defaultLengthNm: 80, defaultR: 1.0, minR: 1, maxR: 1 },
  { id: 'decahedron', label: 'Decahedron', defaultLengthNm: 80, defaultR: 1.0, minR: 1, maxR: 1 },
];

const PI = Math.PI;

function complex(re: number, im = 0): Complex {
  return { re, im };
}

function cAdd(a: Complex, b: Complex): Complex {
  return complex(a.re + b.re, a.im + b.im);
}

function cSub(a: Complex, b: Complex): Complex {
  return complex(a.re - b.re, a.im - b.im);
}

function cMul(a: Complex, b: Complex): Complex {
  return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

function cScale(a: Complex, s: number): Complex {
  return complex(a.re * s, a.im * s);
}

function cDiv(a: Complex, b: Complex): Complex {
  const d = b.re * b.re + b.im * b.im;
  return complex((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

function cInv(a: Complex): Complex {
  const d = a.re * a.re + a.im * a.im;
  return complex(a.re / d, -a.im / d);
}

function cAbs2(a: Complex): number {
  return a.re * a.re + a.im * a.im;
}

export function nkFromMaterial(materialId: string, lambdaNm: number): [number, number] {
  const material = MATERIALS[materialId];
  if (!material) {
    throw new Error(`Unknown material "${materialId}".`);
  }
  return interpolateNK(material.data, lambdaNm);
}

export function epsilonFromMaterial(materialId: string, lambdaNm: number): Complex {
  const [n, k] = nkFromMaterial(materialId, lambdaNm);
  return complex(n * n - k * k, 2 * n * k);
}

export function epsilonFromNKData(data: number[][], lambdaNm: number): Complex {
  const [n, k] = interpolateNK(data, lambdaNm);
  return complex(n * n - k * k, 2 * n * k);
}

export function drudeEpsilon(lambdaNm: number, epsB: number, wpEv: number, gammaEv: number): Complex {
  const energyEv = 1239.841984 / lambdaNm;
  const denominator = complex(energyEv * energyEv, energyEv * gammaEv);
  const term = cDiv(complex(wpEv * wpEv), denominator);
  return cSub(complex(epsB), term);
}

export function shapeMode(shape: PlasmonShapeId, aspectRatio = 1, modeIndex = 1): ShapeMode {
  const r = Math.max(1e-9, aspectRatio);
  let epsilonJ: number;
  let vjOverV: number;
  let a2: number;
  let a4: number;
  let vOverL3: number;

  if (shape === 'rod') {
    epsilonJ = -1.73 * r ** 1.45 - 0.296;
    vjOverV = 0.896;
    a2 = 6.92 / (1 - epsilonJ);
    a4 = -11 / r ** 2.49 - 0.0868;
    vOverL3 = PI * (3 * r - 1) / (12 * r ** 3);
  } else if (shape === 'triangle') {
    epsilonJ = -0.87 * r ** 1.12 - 4.33;
    vjOverV = -0.645 * r ** -1.24 + 0.678;
    a2 = 5.57 / (1 - epsilonJ);
    a4 = -6.83 / (1 - epsilonJ);
    vOverL3 = -0.00544 / r ** 2 + 0.433 / r;
  } else if (shape === 'cage') {
    epsilonJ = -0.0678 * r ** 2.02 - 3.42;
    vjOverV = -0.008 * r ** 2 + 0.103 * r + 0.316;
    a2 = -0.00405 * r ** 2.59 + 2.21;
    a4 = -13.9;
    vOverL3 = 8.04 / r ** 3 - 12 / r ** 2 + 6 / r - 0.00138;
  } else if (shape === 'ellipsoid') {
    epsilonJ = -0.871 - 1.35 * r ** 1.54;
    vjOverV = 0.994;
    a2 = 5.52 / (1 - epsilonJ);
    a4 = -9.75 / r ** 2.53;
    vOverL3 = PI / (6 * r ** 2);
  } else if (shape === 'bicone') {
    epsilonJ = -0.687 - 2.54 * r ** 1.5;
    vjOverV = 0.648 - 0.441 / r ** 0.687;
    a2 = 1.34 / (1 - epsilonJ);
    a4 = -1.04 / (1 - epsilonJ);
    vOverL3 = 0.262 / r ** 2;
  } else if (shape === 'disk') {
    epsilonJ = -0.479 - 1.36 * r ** 0.872;
    vjOverV = 0.944;
    a2 = 7.05 / (1 - epsilonJ);
    a4 = -10.9 / r ** 0.98;
    vOverL3 = PI * (4 + 3 * (r - 1) * (2 * r + PI - 2)) / (24 * r ** 3);
  } else if (shape === 'ring') {
    epsilonJ = 1.39 - 1.31 * r ** 1.73;
    vjOverV = 0.514 + 2.07 / r ** 2.67;
    a2 = 7.24 / (1 - epsilonJ);
    a4 = -19.1 / (1 - epsilonJ);
    vOverL3 = PI ** 2 * (r - 1) / (4 * r ** 3);
  } else if (shape === 'bipyramid') {
    epsilonJ = 1.43 - 4.52 * r ** 1.12;
    vjOverV = 1.96 - 1.73 / r ** 0.207;
    a2 = 2.89 / (1 - epsilonJ);
    a4 = -1.79 / (1 - epsilonJ);
    vOverL3 = 0.219 / r ** 2;
  } else if (shape === 'squared rod') {
    epsilonJ = -2.28 - 1.47 * r ** 1.49;
    vjOverV = 0.904 - 0.411 / r ** 2.26;
    a2 = -0.573 + 3.31 / r ** 0.747;
    a4 = 0.213 - 13.1 / r ** 1.97;
    vOverL3 = 1 / r ** 2;
  } else if (shape === 'cylinder') {
    epsilonJ = -1.59 - 1.96 * r ** 1.4;
    vjOverV = 0.883 - 0.149 / r ** 3.97;
    a2 = -1.05 + 3.02 / r ** 0.494;
    a4 = 0.0796 - 9.08 / r ** 2.08;
    vOverL3 = PI / (4 * r ** 2);
  } else if (shape === 'tetrahedron') {
    epsilonJ = -6.35;
    vjOverV = 0.352;
    a2 = 0.459;
    a4 = -0.416;
    vOverL3 = 1 / (6 * Math.SQRT2);
  } else if (shape === 'octahedron') {
    epsilonJ = -3.85;
    vjOverV = 0.395;
    a2 = 0.547;
    a4 = -0.918;
    vOverL3 = 1 / 6;
  } else if (shape === 'decahedron') {
    epsilonJ = -4.19;
    vjOverV = 0.576;
    a2 = 0.654;
    a4 = -1.05;
    vOverL3 = 0.123;
  } else if (shape === 'rod transverse') {
    vOverL3 = PI * (3 * r - 1) / (12 * r ** 3);
    if (modeIndex === 1) {
      epsilonJ = -1.75 + 3.19 / r ** 6.14;
      vjOverV = 0.0679 + 1.83 / r ** 2.1;
      a2 = 0.0148 + 3.69 / r ** 2.86;
      a4 = 0.0142 - 16.9 / r ** 3.58;
    } else if (modeIndex === 2) {
      epsilonJ = -0.978 - 0.661 / r ** 1.1;
      vjOverV = 0.891 - 2.28 / r ** 2.53;
      a2 = -21.7 + 22.7 / r ** 0.0232;
      a4 = 1.48 - 3.67 / r ** 0.458;
    } else if (modeIndex === 3) {
      epsilonJ = -1.57 + 0.0446 * r;
      vjOverV = -0.0346 + 0.0111 * r;
      a2 = -0.0117 + 0.773 / r ** 1.46;
      a4 = -0.256 + 0.0554 * r ** 0.758;
    } else {
      throw new Error('Rod transverse mode index must be 1, 2, or 3.');
    }
  } else {
    const exhaustive: never = shape;
    throw new Error(`Unknown shape "${exhaustive}".`);
  }

  return {
    epsilonJ,
    vjOverV,
    a2,
    a4,
    vOverL3,
    shape,
    modeIndex,
  };
}

export function availableModeCount(shape: PlasmonShapeId): number {
  return shape === 'rod transverse' ? 3 : 1;
}

export function shapeModes(
  shape: PlasmonShapeId,
  aspectRatio = 1,
  modeCount = availableModeCount(shape),
): ShapeMode[] {
  const count = Math.min(availableModeCount(shape), Math.max(1, Math.round(modeCount)));
  return Array.from({ length: count }, (_, i) => shapeMode(shape, aspectRatio, i + 1));
}

export function retardationCorrection(lambdaNm: number, lengthNm: number, epsilonHost: number, mode: ShapeMode): Complex {
  const s = Math.sqrt(epsilonHost) * lengthNm / lambdaNm;
  return complex(
    mode.a2 * s ** 2 + mode.a4 * s ** 4,
    (4 * PI ** 2 * mode.vjOverV * mode.vOverL3 / 3) * s ** 3,
  );
}

export function polarizabilityAt(
  lambdaNm: number,
  materialId: string,
  modes: ShapeMode[],
  epsilonHost: number,
  lengthNm: number,
  epsilonAt?: (lambdaNm: number) => Complex,
): Complex {
  const epsM = epsilonAt ? epsilonAt(lambdaNm) : epsilonFromMaterial(materialId, lambdaNm);
  let alpha = complex(0, 0);

  for (const mode of modes) {
    const vjNm3 = mode.vjOverV * mode.vOverL3 * lengthNm ** 3;
    const epsRatioMinusOne = cSub(cScale(epsM, 1 / epsilonHost), complex(1));
    const spectralTerm = cInv(epsRatioMinusOne);
    const geometryTerm = complex(1 / (mode.epsilonJ - 1));
    const aj = retardationCorrection(lambdaNm, lengthNm, epsilonHost, mode);
    const denominator = cSub(cSub(spectralTerm, geometryTerm), aj);
    alpha = cAdd(alpha, cScale(cInv(denominator), epsilonHost * vjNm3 / (4 * PI)));
  }

  return alpha;
}

export function crossSectionsAt(lambdaNm: number, alpha: Complex, epsilonHost: number): {
  sigmaExtNm2: number;
  sigmaScaNm2: number;
  sigmaAbsNm2: number;
} {
  const sigmaExtNm2 = 8 * PI ** 2 * alpha.im / (Math.sqrt(epsilonHost) * lambdaNm);
  const sigmaScaNm2 = 128 * PI ** 5 * cAbs2(alpha) / (3 * lambdaNm ** 4);
  return {
    sigmaExtNm2,
    sigmaScaNm2,
    sigmaAbsNm2: sigmaExtNm2 - sigmaScaNm2,
  };
}

export function quantumYieldAt(
  lambdaNm: number,
  materialId: string,
  mode: ShapeMode,
  epsilonHost: number,
  lengthNm: number,
  epsilonAt?: (lambdaNm: number) => Complex,
): number {
  const epsM = epsilonAt ? epsilonAt(lambdaNm) : epsilonFromMaterial(materialId, lambdaNm);
  const invLoss = cInv(cSub(complex(epsilonHost), epsM));
  const vjNm3 = mode.vjOverV * mode.vOverL3 * lengthNm ** 3;
  const factor = (3 * lambdaNm ** 3 * invLoss.im) / (4 * PI ** 2 * Math.sqrt(epsilonHost) * vjNm3);
  const y = 1 / (1 + factor);
  return Math.min(1, Math.max(0, y));
}

export function computePlasmonSpectrum(options: PlasmonSpectrumOptions): PlasmonSpectrum {
  const points = Math.max(16, Math.round(options.points ?? 360));
  const lambdaMin = Math.min(options.lambdaMinNm, options.lambdaMaxNm - 1);
  const lambdaMax = Math.max(options.lambdaMaxNm, lambdaMin + 1);
  const epsilonHost = options.hostIndex ** 2;
  const modes = shapeModes(options.shape, options.aspectRatio, options.modeCount ?? 1);
  const volumeNm3 = modes[0].vOverL3 * options.lengthNm ** 3;

  const wavelengthNm = new Float64Array(points);
  const sigmaExtNm2 = new Float64Array(points);
  const sigmaScaNm2 = new Float64Array(points);
  const sigmaAbsNm2 = new Float64Array(points);
  const sigmaExtOverV = new Float64Array(points);
  const sigmaAbsOverV = new Float64Array(points);
  const sigmaScaOverV = new Float64Array(points);
  const quantumYield = new Float64Array(points);
  const alpha: Complex[] = new Array(points);

  for (let i = 0; i < points; i++) {
    const lambda = lambdaMin + (lambdaMax - lambdaMin) * i / (points - 1);
    wavelengthNm[i] = lambda;
    const a = polarizabilityAt(lambda, options.materialId, modes, epsilonHost, options.lengthNm, options.epsilonAt);
    alpha[i] = a;
    const xs = crossSectionsAt(lambda, a, epsilonHost);
    sigmaExtNm2[i] = xs.sigmaExtNm2;
    sigmaScaNm2[i] = xs.sigmaScaNm2;
    sigmaAbsNm2[i] = xs.sigmaAbsNm2;
    sigmaExtOverV[i] = xs.sigmaExtNm2 / volumeNm3;
    sigmaAbsOverV[i] = xs.sigmaAbsNm2 / volumeNm3;
    sigmaScaOverV[i] = xs.sigmaScaNm2 / volumeNm3;
    quantumYield[i] = quantumYieldAt(lambda, options.materialId, modes[0], epsilonHost, options.lengthNm, options.epsilonAt);
  }

  return {
    wavelengthNm,
    alpha,
    sigmaExtNm2,
    sigmaScaNm2,
    sigmaAbsNm2,
    sigmaExtOverV,
    sigmaAbsOverV,
    sigmaScaOverV,
    quantumYield,
    volumeNm3,
    modes,
    peak: peakMetrics(wavelengthNm, sigmaExtOverV),
  };
}

export function peakMetrics(wavelengthNm: ArrayLike<number>, signal: ArrayLike<number>): PlasmonPeakMetrics {
  const n = Math.min(wavelengthNm.length, signal.length);
  let idx = 0;
  let peakValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i++) {
    const value = signal[i];
    if (Number.isFinite(value) && value > peakValue) {
      peakValue = value;
      idx = i;
    }
  }

  let lambdaPeakNm = wavelengthNm[idx] ?? Number.NaN;
  if (idx > 0 && idx < n - 1) {
    const y0 = signal[idx - 1];
    const y1 = signal[idx];
    const y2 = signal[idx + 1];
    const x0 = wavelengthNm[idx - 1];
    const x1 = wavelengthNm[idx];
    const x2 = wavelengthNm[idx + 1];
    const denom = y0 - 2 * y1 + y2;
    if ([y0, y1, y2, x0, x1, x2, denom].every(Number.isFinite) && denom < 0) {
      const delta = 0.5 * (y0 - y2) / denom;
      if (Math.abs(delta) <= 1) {
        lambdaPeakNm = x1 + delta * (x2 - x0) / 2;
        peakValue = y1 - 0.25 * (y0 - y2) * delta;
      }
    }
  }
  const half = peakValue / 2;
  let leftHalfNm = Number.NaN;
  let rightHalfNm = Number.NaN;

  for (let i = idx; i > 0; i--) {
    if (signal[i - 1] <= half && half <= signal[i]) {
      leftHalfNm = interpolateX(half, signal[i - 1], signal[i], wavelengthNm[i - 1], wavelengthNm[i]);
      break;
    }
  }
  for (let i = idx; i < n - 1; i++) {
    if (signal[i] >= half && half >= signal[i + 1]) {
      rightHalfNm = interpolateX(half, signal[i], signal[i + 1], wavelengthNm[i], wavelengthNm[i + 1]);
      break;
    }
  }

  const fwhmNm = Number.isFinite(leftHalfNm) && Number.isFinite(rightHalfNm) && rightHalfNm > leftHalfNm
    ? rightHalfNm - leftHalfNm
    : Number.NaN;

  return {
    lambdaPeakNm,
    peakValue,
    leftHalfNm,
    rightHalfNm,
    fwhmNm,
    q: Number.isFinite(fwhmNm) && fwhmNm > 0 ? lambdaPeakNm / fwhmNm : Number.NaN,
  };
}

function interpolateX(y: number, y0: number, y1: number, x0: number, x1: number): number {
  if (y1 === y0) return (x0 + x1) / 2;
  return x0 + (y - y0) * (x1 - x0) / (y1 - y0);
}

export function shapeParameterCurves(shape: PlasmonShapeId, rMin = 1.2, rMax = 8, points = 180, modeIndex = 1): ShapeParameterCurves {
  const n = Math.max(2, Math.round(points));
  const r = new Float64Array(n);
  const epsilonJ = new Float64Array(n);
  const vjOverV = new Float64Array(n);
  const a2 = new Float64Array(n);
  const a4 = new Float64Array(n);
  const vOverL3 = new Float64Array(n);
  const vjOverL3 = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const rv = rMin + (rMax - rMin) * i / (n - 1);
    const mode = shapeMode(shape, rv, modeIndex);
    r[i] = rv;
    epsilonJ[i] = mode.epsilonJ;
    vjOverV[i] = mode.vjOverV;
    a2[i] = mode.a2;
    a4[i] = mode.a4;
    vOverL3[i] = mode.vOverL3;
    vjOverL3[i] = mode.vjOverV * mode.vOverL3;
  }

  return { r, epsilonJ, vjOverV, a2, a4, vOverL3, vjOverL3 };
}

export function nearestIndex(values: ArrayLike<number>, target: number): number {
  let idx = 0;
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < values.length; i++) {
    const d = Math.abs(values[i] - target);
    if (d < best) {
      best = d;
      idx = i;
    }
  }
  return idx;
}

export function validityNotes(options: PlasmonSpectrumOptions): string[] {
  const notes: string[] = [];
  const availableModes = availableModeCount(options.shape);
  if (availableModes > 1) {
    notes.push(`This configuration uses all ${availableModes} fitted analytical modes available in the local paper data.`);
  } else if (options.shape === 'bipyramid') {
    notes.push('The local paper tables provide one fitted bipyramid mode. Fig. 2 of Yu, Liz-Marzan, and Garcia de Abajo, Chem. Soc. Rev. 46, 6710-6724 (2017), overlays L = 10 nm and L = 200 nm bipyramids at R = 5 rather than first/second fitted modes.');
  } else {
    notes.push('This geometry has one fitted analytical mode in the local paper data.');
  }
  if (
    options.shape === 'rod transverse'
    && options.materialId.toLowerCase() === 'ag'
    && Math.abs(options.lengthNm - 50) < 1e-6
    && Math.abs(options.aspectRatio - 4) < 1e-6
  ) {
    notes.push('This default is the Ag transverse nanorod L = 50 nm, R = 4 curve from Fig. S3c of the Supporting Information to the same 2017 Chem. Soc. Rev. paper.');
  }
  if (
    options.shape === 'bipyramid'
    && options.materialId.toLowerCase() === 'au'
    && Math.abs(options.lengthNm - 200) < 1e-6
    && Math.abs(options.aspectRatio - 5) < 1e-6
  ) {
    notes.push('This default is the L = 200 nm, R = 5 Au bipyramid trace from Fig. 2 of that paper.');
  }
  if (options.lengthNm < 10) {
    notes.push('Below about 10 nm, nonlocal response, surface scattering, and quantum confinement can matter.');
  }
  if (options.lengthNm > 250) {
    notes.push('Large particles may need higher-order modes beyond this compact dipolar model.');
  }
  if (options.shape === 'rod transverse' && (options.modeCount ?? 1) >= 3 && options.aspectRatio < 4) {
    notes.push('The third rod transverse fit was reported for aspect ratios R >= 4.');
  }
  if (options.lambdaMinNm < 300 || options.lambdaMaxNm > 1800) {
    notes.push('The optical-constant interpolation is sparse outside the visible/NIR range used by the lab data.');
  }
  if (options.hostIndex <= 0) {
    notes.push('The host refractive index must be positive.');
  }
  return notes;
}
