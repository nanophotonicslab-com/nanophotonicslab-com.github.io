/** Promise-based bridge to the Pyodide Web Worker. */

import type { SpectrumData, MeshData } from './state';

export interface SolveParams {
  shape: string;
  radius: number;
  aspectRatio: number;
  subdivisions: number;
  material: string;
  epsRe: number;
  epsIm: number;
  epsMedium: number;
  wavelengths: number[];
}

export interface SolveResult {
  spectra: SpectrumData;
  mesh: MeshData;
}

type InitProgressFn = (stage: string, pct: number) => void;
type SolveProgressFn = (index: number, total: number, wavelength: number) => void;

export class PyodideBridge {
  private worker: Worker | null = null;
  private onInitProgress: InitProgressFn = () => {};
  private onSolveProgress: SolveProgressFn = () => {};
  private pendingResolve: ((value: any) => void) | null = null;
  private pendingReject: ((reason: any) => void) | null = null;

  async init(onProgress: InitProgressFn): Promise<void> {
    this.onInitProgress = onProgress;

    // Create worker from the bundled script
    this.worker = new Worker(
      new URL('./pyodide-worker.ts', import.meta.url),
      { type: 'classic' }
    );

    this.worker.onmessage = (e: MessageEvent) => this.handleMessage(e.data);

    return new Promise<void>((resolve, reject) => {
      this.pendingResolve = resolve;
      this.pendingReject = reject;
      this.worker!.postMessage({ type: 'init' });
    });
  }

  async solveSpectrum(params: SolveParams, onProgress: SolveProgressFn): Promise<SolveResult> {
    if (!this.worker) throw new Error('Worker not initialized');
    this.onSolveProgress = onProgress;

    return new Promise<SolveResult>((resolve, reject) => {
      this.pendingResolve = resolve;
      this.pendingReject = reject;
      this.worker!.postMessage({ type: 'solve-spectrum', params });
    });
  }

  async surfaceFields(wavelength: number): Promise<Float64Array> {
    if (!this.worker) throw new Error('Worker not initialized');

    return new Promise<Float64Array>((resolve, reject) => {
      this.pendingResolve = resolve;
      this.pendingReject = reject;
      this.worker!.postMessage({ type: 'surface-fields', wavelength });
    });
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
  }

  private handleMessage(msg: any): void {
    switch (msg.type) {
      case 'init-progress':
        this.onInitProgress(msg.stage, msg.pct);
        break;
      case 'init-done':
        this.pendingResolve?.(undefined);
        this.pendingResolve = null;
        break;
      case 'init-error':
        this.pendingReject?.(new Error(msg.error));
        this.pendingReject = null;
        break;
      case 'solve-progress':
        this.onSolveProgress(msg.index, msg.total, msg.wavelength);
        break;
      case 'solve-result':
        this.pendingResolve?.(msg);
        this.pendingResolve = null;
        break;
      case 'surface-fields-result':
        this.pendingResolve?.(msg.enhancement);
        this.pendingResolve = null;
        break;
      case 'solve-error':
        this.pendingReject?.(new Error(msg.error));
        this.pendingReject = null;
        break;
    }
  }
}
