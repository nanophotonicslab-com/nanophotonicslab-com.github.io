/** Typed pub/sub state manager for the BEM solver app. */

export interface GeometryState {
  shape: 'sphere' | 'rod';
  radius: number;       // nm
  aspectRatio: number;   // for rod: total length / diameter
  subdivisions: number;  // 1 = 80 faces, 2 = 320 faces
}

export interface MaterialState {
  key: 'au' | 'ag' | 'custom';
  epsRe: number;   // for custom: Re(ε)
  epsIm: number;   // for custom: Im(ε)
}

export interface MediumState {
  epsMedium: number;  // ε_m (real)
}

export interface WavelengthState {
  min: number;     // nm
  max: number;     // nm
  npoints: number;
}

export interface SolverState {
  status: 'idle' | 'loading' | 'solving' | 'done' | 'error';
  progress: number;   // 0–1
  message: string;
}

export interface SpectrumData {
  wavelengths: number[];
  extinction: number[];
  scattering: number[];
  absorption: number[];
}

export interface MeshData {
  vertices: Float64Array;
  faces: Int32Array;
  nFaces: number;
  nVertices: number;
}

export interface ResultsState {
  spectra: SpectrumData | null;
  mesh: MeshData | null;
  enhancement: Float64Array | null;
  selectedWavelength: number | null;
  timing: number | null;  // seconds
}

export interface AppState {
  geometry: GeometryState;
  material: MaterialState;
  medium: MediumState;
  wavelength: WavelengthState;
  solver: SolverState;
  results: ResultsState;
}

export const DEFAULT_STATE: AppState = {
  geometry: { shape: 'sphere', radius: 25, aspectRatio: 3, subdivisions: 1 },
  material: { key: 'au', epsRe: -10, epsIm: 1.2 },
  medium: { epsMedium: 1.77 },  // water (n=1.33)
  wavelength: { min: 400, max: 800, npoints: 20 },
  solver: { status: 'idle', progress: 0, message: '' },
  results: { spectra: null, mesh: null, enhancement: null, selectedWavelength: null, timing: null },
};

type Listener<K extends keyof AppState> = (val: AppState[K]) => void;

export class StateManager {
  private state: AppState;
  private listeners = new Map<keyof AppState, Set<Listener<any>>>();

  constructor(initial: AppState = DEFAULT_STATE) {
    this.state = structuredClone(initial);
  }

  get<K extends keyof AppState>(key: K): AppState[K] {
    return this.state[key];
  }

  set<K extends keyof AppState>(key: K, value: Partial<AppState[K]>): void {
    this.state[key] = { ...this.state[key], ...value } as AppState[K];
    this.notify(key);
  }

  subscribe<K extends keyof AppState>(key: K, fn: Listener<K>): () => void {
    if (!this.listeners.has(key)) this.listeners.set(key, new Set());
    this.listeners.get(key)!.add(fn);
    return () => this.listeners.get(key)?.delete(fn);
  }

  private notify<K extends keyof AppState>(key: K): void {
    this.listeners.get(key)?.forEach(fn => fn(this.state[key]));
  }
}
