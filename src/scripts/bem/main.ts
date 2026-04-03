/** Entry point: wires state, worker, viewport, and controls together. */

import { StateManager, DEFAULT_STATE } from './state';
import type { AppState } from './state';
import { PyodideBridge, type SolveParams } from './pyodide-bridge';
import { BEMViewport } from './viewport';
import { drawColorbar } from './colormap';

export async function initBEMSolver() {
  const state = new StateManager(DEFAULT_STATE);
  const bridge = new PyodideBridge();
  let viewport: BEMViewport | null = null;

  // --- DOM references ---
  const el = (id: string) => document.getElementById(id)!;
  const progressBar = el('progress-fill') as HTMLElement;
  const progressText = el('progress-text') as HTMLElement;
  const statusText = el('status-text') as HTMLElement;
  const solveBtn = el('solve-btn') as HTMLButtonElement;
  const viewportContainer = el('viewport-container');
  const colorbarCanvas = el('colorbar') as HTMLCanvasElement;

  // --- Initialize 3D viewport ---
  viewport = new BEMViewport(viewportContainer);

  // --- State → UI bindings ---
  state.subscribe('solver', (s) => {
    progressBar.style.width = `${s.progress * 100}%`;
    progressText.textContent = s.message;
    solveBtn.disabled = s.status === 'loading' || s.status === 'solving';

    if (s.status === 'idle') statusText.textContent = 'Ready';
    else if (s.status === 'loading') statusText.textContent = s.message;
    else if (s.status === 'solving') statusText.textContent = s.message;
    else if (s.status === 'done') statusText.textContent = 'Done';
    else if (s.status === 'error') statusText.textContent = `Error: ${s.message}`;
  });

  state.subscribe('results', (r) => {
    if (r.mesh && viewport) {
      viewport.setMesh(r.mesh.vertices, r.mesh.faces, r.mesh.nFaces, r.mesh.nVertices);
    }
    if (r.enhancement && viewport) {
      const enh = r.enhancement;
      const min = Math.min(...Array.from(enh));
      const max = Math.max(...Array.from(enh));
      viewport.setFaceColors(enh);
      drawColorbar(colorbarCanvas, min, max, '|E|/|E₀|');
    }
    if (r.spectra) {
      renderSpectrum(r.spectra, r.selectedWavelength);
    }
    if (r.timing !== null) {
      const meshInfo = r.mesh ? `${r.mesh.nFaces} faces` : '';
      const nLam = r.spectra?.wavelengths.length ?? 0;
      statusText.textContent = `${meshInfo} | ${nLam}λ in ${r.timing.toFixed(1)}s`;
    }
  });

  // --- Read params from DOM ---
  function readParams(): SolveParams {
    const g = state.get('geometry');
    const m = state.get('material');
    const med = state.get('medium');
    const w = state.get('wavelength');

    const wavelengths: number[] = [];
    const step = (w.max - w.min) / Math.max(w.npoints - 1, 1);
    for (let i = 0; i < w.npoints; i++) {
      wavelengths.push(w.min + i * step);
    }

    return {
      shape: g.shape,
      radius: g.radius,
      aspectRatio: g.aspectRatio,
      subdivisions: g.subdivisions,
      material: m.key,
      epsRe: m.epsRe,
      epsIm: m.epsIm,
      epsMedium: med.epsMedium,
      wavelengths,
    };
  }

  // --- Bind DOM inputs → state ---
  function bindInput(id: string, stateKey: keyof AppState, field: string, transform = parseFloat) {
    const input = document.getElementById(id) as HTMLInputElement | null;
    if (!input) return;
    input.addEventListener('input', () => {
      const val = transform(input.value);
      if (!isNaN(val as number)) {
        state.set(stateKey, { [field]: val } as any);
      }
    });
    // Sync initial value
    state.subscribe(stateKey, (s: any) => {
      if (document.activeElement !== input) {
        input.value = String(s[field]);
      }
    });
  }

  function bindSelect(id: string, stateKey: keyof AppState, field: string, transform: (v: string) => any = (v) => v) {
    const sel = document.getElementById(id) as HTMLSelectElement | null;
    if (!sel) return;
    sel.addEventListener('change', () => {
      state.set(stateKey, { [field]: transform(sel.value) } as any);
    });
  }

  // Geometry
  bindSelect('shape-select', 'geometry', 'shape');
  bindInput('radius-input', 'geometry', 'radius');
  bindSelect('subdivisions-input', 'geometry', 'subdivisions', parseInt);

  // Material
  bindSelect('material-select', 'material', 'key');

  // Medium
  bindInput('eps-medium-input', 'medium', 'epsMedium');

  // Wavelength
  bindInput('wl-min-input', 'wavelength', 'min');
  bindInput('wl-max-input', 'wavelength', 'max');
  bindInput('wl-npoints-input', 'wavelength', 'npoints', parseInt);

  // --- Regenerate mesh when geometry changes ---
  let meshDebounce: ReturnType<typeof setTimeout> | null = null;
  state.subscribe('geometry', () => {
    if (meshDebounce) clearTimeout(meshDebounce);
    meshDebounce = setTimeout(async () => {
      if (state.get('solver').status === 'loading') return;
      try {
        const mesh = await bridge.generateMesh(readParams());
        state.set('results', { mesh, enhancement: null });
      } catch { /* ignore if worker not ready */ }
    }, 300);
  });

  // --- Solve button ---
  solveBtn.addEventListener('click', () => solve());

  // --- Medium presets ---
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const eps = parseFloat((btn as HTMLElement).dataset.eps || '1');
      state.set('medium', { epsMedium: eps });
      const input = document.getElementById('eps-medium-input') as HTMLInputElement;
      if (input) input.value = String(eps);
    });
  });

  // --- Wireframe toggle ---
  el('wireframe-btn')?.addEventListener('click', () => viewport?.toggleWireframe());
  el('reset-view-btn')?.addEventListener('click', () => viewport?.resetView());

  // --- Spectrum chart click → surface fields ---
  function onSpectrumClick(wavelength: number) {
    state.set('results', { selectedWavelength: wavelength });
    state.set('solver', { status: 'solving', progress: 0, message: `Computing fields at ${wavelength.toFixed(0)}nm...` });

    bridge.surfaceFields(wavelength).then((enh) => {
      state.set('results', { enhancement: enh });
      state.set('solver', { status: 'done', progress: 1, message: '' });
    }).catch((err) => {
      state.set('solver', { status: 'error', progress: 0, message: err.message });
    });
  }

  // --- Solve ---
  async function solve() {
    const params = readParams();
    const t0 = performance.now();

    state.set('solver', { status: 'solving', progress: 0, message: 'Solving...' });
    state.set('results', { spectra: null, enhancement: null, selectedWavelength: null, timing: null });

    try {
      const result = await bridge.solveSpectrum(params, (i, n, lam) => {
        state.set('solver', {
          progress: i / n,
          message: `Wavelength ${i}/${n} (${lam.toFixed(0)}nm)`,
        });
      });

      const dt = (performance.now() - t0) / 1000;
      state.set('results', {
        spectra: result.spectra,
        mesh: result.mesh,
        timing: dt,
      });
      state.set('solver', { status: 'done', progress: 1, message: '' });

      // Auto-show surface fields at peak wavelength
      const ext = result.spectra.extinction;
      const peakIdx = ext.indexOf(Math.max(...ext));
      const peakLam = result.spectra.wavelengths[peakIdx];
      onSpectrumClick(peakLam);
    } catch (err: any) {
      state.set('solver', { status: 'error', progress: 0, message: err.message });
    }
  }

  // --- Simple spectrum rendering (canvas) ---
  function renderSpectrum(spectra: AppState['results']['spectra'], selectedLam: number | null) {
    if (!spectra) return;
    const canvas = el('spectrum-canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    const w = canvas.width, h = canvas.height;
    const pad = { top: 20, right: 20, bottom: 40, left: 60 };

    ctx.clearRect(0, 0, w, h);

    const lams = spectra.wavelengths;
    const datasets = [
      { data: spectra.extinction, color: '#6366f1', label: 'C_ext' },
      { data: spectra.scattering, color: '#06b6d4', label: 'C_sca' },
      { data: spectra.absorption, color: '#f59e0b', label: 'C_abs' },
    ];

    const allVals = [...spectra.extinction, ...spectra.scattering, ...spectra.absorption];
    const yMin = 0;
    const yMax = Math.max(...allVals) * 1.1;
    const xMin = Math.min(...lams);
    const xMax = Math.max(...lams);

    const toX = (v: number) => pad.left + (v - xMin) / (xMax - xMin) * (w - pad.left - pad.right);
    const toY = (v: number) => h - pad.bottom - (v - yMin) / (yMax - yMin) * (h - pad.top - pad.bottom);

    // Axes
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#1e1b4b';
    ctx.font = '12px Inter, system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Wavelength (nm)', w / 2, h - 5);
    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Cross-section (nm²)', 0, 0);
    ctx.restore();

    // Tick labels
    ctx.font = '10px Inter, system-ui';
    ctx.textAlign = 'center';
    for (let l = Math.ceil(xMin / 100) * 100; l <= xMax; l += 100) {
      ctx.fillText(String(l), toX(l), h - pad.bottom + 15);
    }
    ctx.textAlign = 'right';
    const yStep = Math.pow(10, Math.floor(Math.log10(yMax)));
    for (let v = 0; v <= yMax; v += yStep) {
      ctx.fillText(v.toFixed(0), pad.left - 5, toY(v) + 4);
    }

    // Selected wavelength marker
    if (selectedLam !== null) {
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(toX(selectedLam), pad.top);
      ctx.lineTo(toX(selectedLam), h - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Plot lines
    for (const ds of datasets) {
      ctx.strokeStyle = ds.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < lams.length; i++) {
        const x = toX(lams[i]), y = toY(ds.data[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Dots
      ctx.fillStyle = ds.color;
      for (let i = 0; i < lams.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(lams[i]), toY(ds.data[i]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Legend
    ctx.font = '11px Inter, system-ui';
    let lx = pad.left + 10;
    for (const ds of datasets) {
      ctx.fillStyle = ds.color;
      ctx.fillRect(lx, pad.top + 2, 12, 3);
      ctx.fillStyle = '#1e1b4b';
      ctx.textAlign = 'left';
      ctx.fillText(ds.label, lx + 16, pad.top + 8);
      lx += 70;
    }

    // Click handler for wavelength selection
    canvas.onclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const cx = (ev.clientX - rect.left) * (w / rect.width);
      const lam = xMin + (cx - pad.left) / (w - pad.left - pad.right) * (xMax - xMin);
      if (lam >= xMin && lam <= xMax) {
        onSpectrumClick(Math.round(lam));
      }
    };
  }

  // --- Initialize Pyodide ---
  state.set('solver', { status: 'loading', progress: 0, message: 'Loading Pyodide...' });

  try {
    await bridge.init((stage, pct) => {
      state.set('solver', { status: 'loading', progress: pct, message: stage });
    });
    state.set('solver', { status: 'idle', progress: 1, message: '' });

    // Show initial mesh preview
    const params = readParams();
    const mesh = await bridge.generateMesh(params);
    state.set('results', { mesh });
  } catch (err: any) {
    state.set('solver', { status: 'error', progress: 0, message: `Failed to load: ${err.message}` });
  }
}
