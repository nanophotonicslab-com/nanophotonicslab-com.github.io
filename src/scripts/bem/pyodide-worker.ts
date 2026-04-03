/**
 * Web Worker: loads Pyodide, installs nanobem, runs BEM solves.
 *
 * Communication via postMessage with typed message protocol.
 * Python computation runs here, never blocking the main thread.
 */

declare const self: DedicatedWorkerGlobalScope;

// @ts-ignore — Pyodide loaded from CDN
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js');

interface SolveParams {
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

let pyodide: any = null;

function post(msg: any) {
  self.postMessage(msg);
}

async function init() {
  try {
    post({ type: 'init-progress', stage: 'Loading Pyodide...', pct: 0.1 });

    // @ts-ignore
    pyodide = await loadPyodide();

    post({ type: 'init-progress', stage: 'Installing packages...', pct: 0.3 });

    await pyodide.loadPackage(['numpy', 'scipy', 'micropip']);

    post({ type: 'init-progress', stage: 'Installing trimesh...', pct: 0.6 });

    await pyodide.runPythonAsync(`
import micropip
await micropip.install('trimesh', deps=False)
    `);

    post({ type: 'init-progress', stage: 'Installing nanobem...', pct: 0.8 });

    // Install nanobem from the wheel in public/wasm/
    const wheelUrl = new URL('/wasm/nanobem-0.1.0-py3-none-any.whl', self.location.origin).href;
    await pyodide.runPythonAsync(`
import micropip
await micropip.install('${wheelUrl}', deps=False)
    `);

    // Verify import works
    await pyodide.runPythonAsync(`
import nanobem as nb
print(f"nanobem loaded: {nb.sphere_mesh(radius=10, subdivisions=1).n_faces} faces")
    `);

    post({ type: 'init-progress', stage: 'Ready', pct: 1.0 });
    post({ type: 'init-done' });
  } catch (e: any) {
    post({ type: 'init-error', error: e.message || String(e) });
  }
}

async function solveSpectrum(params: SolveParams) {
  try {
    // Inject progress callback that also sends per-wavelength results
    pyodide.globals.set('_post_progress', (i: number, n: number, lam: number, ext: number, sca: number, abs: number) => {
      post({ type: 'solve-progress', index: i, total: n, wavelength: lam, ext, sca, abs });
    });

    // Build the Python solve script
    const materialLine = params.material === 'custom'
      ? `eps_p = complex(${params.epsRe}, ${params.epsIm})`
      : `eps_p = nb.${params.material === 'au' ? 'gold' : 'silver'}(lam)`;

    const meshLine = params.shape === 'sphere'
      ? `mesh = nb.sphere_mesh(radius=${params.radius}, subdivisions=${params.subdivisions})`
      : `mesh = nb.rod_mesh(radius=${params.radius}, aspect_ratio=${params.aspectRatio}, subdivisions=${params.subdivisions})`;

    const wavelengthsJson = JSON.stringify(params.wavelengths);

    const script = `
import nanobem as nb
import numpy as np

${meshLine}

wavelengths = ${wavelengthsJson}
eps_m = ${params.epsMedium}
c_ext = []
c_sca = []
c_abs = []

for i, lam in enumerate(wavelengths):
    ${params.material === 'custom' ? `eps_p = complex(${params.epsRe}, ${params.epsIm})` : `eps_p = nb.${params.material === 'au' ? 'gold' : 'silver'}(lam)`}
    result = nb.solve(mesh, eps_p, eps_medium=eps_m, wavelength=lam)
    c_ext.append(result.extinction)
    c_sca.append(result.scattering)
    c_abs.append(result.absorption)
    _post_progress(i + 1, len(wavelengths), float(lam), result.extinction, result.scattering, result.absorption)

# Export mesh data
verts = mesh.vertices.ravel().tolist()
faces = mesh.faces.ravel().tolist()
n_faces = mesh.n_faces
n_verts = mesh.n_vertices
    `;

    await pyodide.runPythonAsync(script);

    // Extract results from Python
    const c_ext = pyodide.globals.get('c_ext').toJs();
    const c_sca = pyodide.globals.get('c_sca').toJs();
    const c_abs = pyodide.globals.get('c_abs').toJs();
    const verts = pyodide.globals.get('verts').toJs();
    const faces = pyodide.globals.get('faces').toJs();
    const nFaces = pyodide.globals.get('n_faces');
    const nVerts = pyodide.globals.get('n_verts');

    post({
      type: 'solve-result',
      spectra: {
        wavelengths: params.wavelengths,
        extinction: Array.from(c_ext),
        scattering: Array.from(c_sca),
        absorption: Array.from(c_abs),
      },
      mesh: {
        vertices: new Float64Array(verts),
        faces: new Int32Array(faces),
        nFaces,
        nVertices: nVerts,
      },
    });
  } catch (e: any) {
    post({ type: 'solve-error', error: e.message || String(e) });
  }
}

async function surfaceFields(wavelength: number, params: SolveParams) {
  try {
    const materialLine = params.material === 'custom'
      ? `eps_p = complex(${params.epsRe}, ${params.epsIm})`
      : `eps_p = nb.${params.material === 'au' ? 'gold' : 'silver'}(${wavelength})`;

    const meshLine = params.shape === 'sphere'
      ? `mesh = nb.sphere_mesh(radius=${params.radius}, subdivisions=${params.subdivisions})`
      : `mesh = nb.rod_mesh(radius=${params.radius}, aspect_ratio=${params.aspectRatio}, subdivisions=${params.subdivisions})`;

    const script = `
import nanobem as nb
import numpy as np

${meshLine}
${materialLine}
eps_m = ${params.epsMedium}

result = nb.solve(mesh, eps_p, eps_medium=eps_m, wavelength=${wavelength})
_, enhancement = result.surface_fields()
enh_list = enhancement.tolist()
    `;

    await pyodide.runPythonAsync(script);
    const enh = pyodide.globals.get('enh_list').toJs();

    post({
      type: 'surface-fields-result',
      enhancement: new Float64Array(enh),
      wavelength,
    });
  } catch (e: any) {
    post({ type: 'solve-error', error: e.message || String(e) });
  }
}

async function generateMesh(params: SolveParams) {
  try {
    const meshLine = params.shape === 'sphere'
      ? `mesh = nb.sphere_mesh(radius=${params.radius}, subdivisions=${params.subdivisions})`
      : `mesh = nb.rod_mesh(radius=${params.radius}, aspect_ratio=${params.aspectRatio}, subdivisions=${params.subdivisions})`;

    await pyodide.runPythonAsync(`
import nanobem as nb
${meshLine}
verts = mesh.vertices.ravel().tolist()
faces = mesh.faces.ravel().tolist()
n_faces = mesh.n_faces
n_verts = mesh.n_vertices
    `);

    const verts = pyodide.globals.get('verts').toJs();
    const faces = pyodide.globals.get('faces').toJs();
    const nFaces = pyodide.globals.get('n_faces');
    const nVerts = pyodide.globals.get('n_verts');

    post({
      type: 'mesh-result',
      mesh: {
        vertices: new Float64Array(verts),
        faces: new Int32Array(faces),
        nFaces,
        nVertices: nVerts,
      },
    });
  } catch (e: any) {
    post({ type: 'solve-error', error: e.message || String(e) });
  }
}

// Message handler
let lastParams: SolveParams | null = null;

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      await init();
      break;
    case 'generate-mesh':
      lastParams = msg.params;
      await generateMesh(msg.params);
      break;
    case 'solve-spectrum':
      lastParams = msg.params;
      await solveSpectrum(msg.params);
      break;
    case 'surface-fields':
      if (lastParams) {
        await surfaceFields(msg.wavelength, lastParams);
      }
      break;
  }
};
