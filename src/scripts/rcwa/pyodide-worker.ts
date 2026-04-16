/**
 * Web Worker: loads Pyodide + inkstone, runs 1D RCWA sweeps.
 */

declare const self: DedicatedWorkerGlobalScope;

// @ts-ignore — Pyodide loaded from CDN
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js');

export interface RcwaParams {
  period: number;           // nm
  thickness: number;        // nm
  ridgeWidth: number;       // nm (fractional: 0 < ridgeWidth < period)
  epsRidge: [number, number];   // (Re, Im)
  epsGroove: [number, number];
  epsSub: [number, number];
  epsTop: [number, number];
  wavelengths: number[];    // nm
  theta: number;            // degrees
  polarization: 's' | 'p';
  numG: number;             // Fourier harmonics
}

let pyodide: any = null;

function post(msg: any) { self.postMessage(msg); }

async function init() {
  try {
    post({ type: 'init-progress', stage: 'Downloading Pyodide runtime (~6 MB)…', pct: 0.05 });
    // @ts-ignore
    pyodide = await loadPyodide();

    post({ type: 'init-progress', stage: 'Loading numpy (~8 MB)…', pct: 0.25 });
    await pyodide.loadPackage(['numpy']);

    post({ type: 'init-progress', stage: 'Loading scipy (~20 MB)…', pct: 0.45 });
    await pyodide.loadPackage(['scipy']);

    post({ type: 'init-progress', stage: 'Loading micropip…', pct: 0.75 });
    await pyodide.loadPackage(['micropip']);

    post({ type: 'init-progress', stage: 'Installing inkstone…', pct: 0.9 });
    await pyodide.runPythonAsync(`
import micropip
await micropip.install('inkstone')
import inkstone
    `);

    post({ type: 'init-progress', stage: 'Ready — first load cached for next time', pct: 1.0 });
    post({ type: 'init-done' });
  } catch (e: any) {
    post({ type: 'init-error', error: e.message || String(e) });
  }
}

async function solveSpectrum(params: RcwaParams) {
  try {
    pyodide.globals.set('_post_progress', (i: number, n: number, lam: number, R: number, T: number) => {
      post({ type: 'solve-progress', index: i, total: n, wavelength: lam, R, T });
    });

    // All lengths in µm (inkstone uses c=1, so frequency = 1/λ in same units)
    const periodUm = params.period / 1000;
    const thicknessUm = params.thickness / 1000;
    const ridgeWidthUm = params.ridgeWidth / 1000;
    const wavelengthsUm = params.wavelengths.map(w => w / 1000);

    const sAmp = params.polarization === 's' ? 1 : 0;
    const pAmp = params.polarization === 'p' ? 1 : 0;

    const script = `
from inkstone import Inkstone
import numpy as np

s = Inkstone()
s.lattice = ${periodUm}
s.num_g = ${params.numG}

s.AddMaterial(name='top',    epsilon=complex(${params.epsTop[0]}, ${params.epsTop[1]}))
s.AddMaterial(name='ridge',  epsilon=complex(${params.epsRidge[0]}, ${params.epsRidge[1]}))
s.AddMaterial(name='groove', epsilon=complex(${params.epsGroove[0]}, ${params.epsGroove[1]}))
s.AddMaterial(name='sub',    epsilon=complex(${params.epsSub[0]}, ${params.epsSub[1]}))

s.AddLayer(name='in', thickness=0, material_background='top')
s.AddLayer(name='grating', thickness=${thicknessUm}, material_background='groove')
s.AddPattern1D(layer='grating', pattern_name='ridge', material='ridge', width=${ridgeWidthUm}, center=0.0)
s.AddLayer(name='out', thickness=0, material_background='sub')

s.SetExcitation(theta=${params.theta}, phi=0, s_amplitude=${sAmp}, p_amplitude=${pAmp})

wavelengths_um = ${JSON.stringify(wavelengthsUm)}
R_list = []
T_list = []
for i, lam in enumerate(wavelengths_um):
    s.frequency = 1.0 / lam  # c=1 in inkstone
    inc, ref = s.GetPowerFlux('in')
    tra = s.GetPowerFlux('out')[0]
    R_val = float((-ref / inc).real) if abs(inc) > 1e-30 else 0.0
    T_val = float((tra / inc).real) if abs(inc) > 1e-30 else 0.0
    R_list.append(R_val)
    T_list.append(T_val)
    _post_progress(i + 1, len(wavelengths_um), float(lam * 1000), R_val, T_val)
    `;

    await pyodide.runPythonAsync(script);

    const R = pyodide.globals.get('R_list').toJs();
    const T = pyodide.globals.get('T_list').toJs();

    post({
      type: 'solve-result',
      wavelengths: params.wavelengths,
      R: Array.from(R) as number[],
      T: Array.from(T) as number[],
    });
  } catch (e: any) {
    post({ type: 'solve-error', error: e.message || String(e) });
  }
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  if (msg.type === 'init') await init();
  else if (msg.type === 'solve') await solveSpectrum(msg.params);
};
