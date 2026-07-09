// Lazy-loaded client for the refractiveindex.info GitHub database.
// No dependencies — custom parsers for the known YAML structure.

const BASE_URL = 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database';

export interface RIIMaterial {
  shelf: string;
  book: string;
  bookName: string;
  pageName: string;
  dataPath: string;
}

// --- Catalog parser (line-by-line state machine) ---

function parseCatalog(text: string): RIIMaterial[] {
  const materials: RIIMaterial[] = [];
  let shelf = '', book = '', bookName = '', pageName = '';
  let lastEntity: 'shelf' | 'book' | 'page' | '' = '';

  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t || t.startsWith('#')) continue;

    if (t.startsWith('- SHELF:')) {
      shelf = t.slice(9).trim();
      lastEntity = 'shelf';
    } else if (t.startsWith('- BOOK:')) {
      book = t.slice(8).trim();
      lastEntity = 'book';
    } else if (t.startsWith('- PAGE:')) {
      lastEntity = 'page';
    } else if (t.startsWith('name:')) {
      const val = t.slice(5).trim().replace(/^["']|["']$/g, '');
      if (lastEntity === 'book') bookName = val;
      else if (lastEntity === 'page') pageName = val;
    } else if (t.startsWith('data:') && lastEntity === 'page') {
      materials.push({ shelf, book, bookName, pageName, dataPath: t.slice(5).trim() });
      lastEntity = '';
    }
  }
  return materials;
}

// --- Material data parser ---
// Handles the refractiveindex.info DATA section: any mix of `tabulated nk|n|k`
// blocks and `formula N` dispersion entries. Formula entries are evaluated over
// their wavelength_range (n only). The result is a merged [λ_um, n, k] table;
// materials with only n (formula or tabulated n) get k = 0.

// Evaluate a refractiveindex.info dispersion formula (returns n). λ in micrometres.
function evalFormula(type: number, c: number[], wl: number): number {
  const wl2 = wl * wl;
  const pow = Math.pow;
  switch (type) {
    case 1: { // Sellmeier
      let n2 = 1 + (c[0] || 0);
      for (let i = 1; i + 1 < c.length; i += 2) n2 += (c[i] * wl2) / (wl2 - c[i + 1] * c[i + 1]);
      return Math.sqrt(n2);
    }
    case 2: { // Sellmeier-2 (denominator term already squared)
      let n2 = 1 + (c[0] || 0);
      for (let i = 1; i + 1 < c.length; i += 2) n2 += (c[i] * wl2) / (wl2 - c[i + 1]);
      return Math.sqrt(n2);
    }
    case 3: { // Polynomial
      let n2 = c[0] || 0;
      for (let i = 1; i + 1 < c.length; i += 2) n2 += c[i] * pow(wl, c[i + 1]);
      return Math.sqrt(n2);
    }
    case 4: { // RefractiveIndex.INFO
      let n2 = c[0] || 0;
      if (c.length > 4) n2 += (c[1] * pow(wl, c[2])) / (wl2 - pow(c[3], c[4]));
      if (c.length > 8) n2 += (c[5] * pow(wl, c[6])) / (wl2 - pow(c[7], c[8]));
      for (let i = 9; i + 1 < c.length; i += 2) n2 += c[i] * pow(wl, c[i + 1]);
      return Math.sqrt(n2);
    }
    case 5: { // Cauchy
      let n = c[0] || 0;
      for (let i = 1; i + 1 < c.length; i += 2) n += c[i] * pow(wl, c[i + 1]);
      return n;
    }
    case 6: { // Gases
      let n = 1 + (c[0] || 0);
      for (let i = 1; i + 1 < c.length; i += 2) n += c[i] / (c[i + 1] - pow(wl, -2));
      return n;
    }
    case 7: { // Herzberger
      const L = 1 / (wl2 - 0.028);
      let n = (c[0] || 0) + (c[1] || 0) * L + (c[2] || 0) * L * L;
      if (c.length > 3) n += c[3] * wl2;
      if (c.length > 4) n += c[4] * wl2 * wl2;
      if (c.length > 5) n += c[5] * wl2 * wl2 * wl2;
      return n;
    }
    case 8: { // Retro
      const r = (c[0] || 0) + (c[1] * wl2) / (wl2 - c[2]) + (c[3] || 0) * wl2;
      return Math.sqrt((1 + 2 * r) / (1 - r));
    }
    default:
      return NaN; // formula 9 (exotic) not supported
  }
}

function parseNKData(text: string): number[][] {
  const nSamples: [number, number][] = [];   // [λ_um, n]
  const kSamples: [number, number][] = [];    // [λ_um, k]
  const nums = (s: string) => s.trim().split(/\s+/).map(Number).filter(v => Number.isFinite(v));

  type Entry = { kind: 'nk' | 'n' | 'k' | 'formula'; fnum?: number; range?: [number, number]; coeffs?: number[]; inData?: boolean };
  let cur: Entry | null = null;
  const flushFormula = (e: Entry) => {
    if (!e.range || !e.coeffs || e.coeffs.length === 0) return;
    const [a, b] = e.range, N = 240;
    for (let s = 0; s < N; s++) {
      const wl = a + ((b - a) * s) / (N - 1);
      const n = evalFormula(e.fnum!, e.coeffs, wl);
      if (Number.isFinite(n) && n > 0) nSamples.push([wl, n]);
    }
  };

  const lines = text.split('\n');
  for (const line of lines) {
    const t = line.trim();
    const typeM = t.match(/^-\s*type:\s*(.+)$/);
    if (typeM) {
      if (cur && cur.kind === 'formula') flushFormula(cur);
      const ts = typeM[1].trim().replace(/^["']|["']$/g, '');
      if (ts.startsWith('formula')) cur = { kind: 'formula', fnum: parseInt(ts.split(/\s+/)[1] || '0', 10), coeffs: [] };
      else { const w = ts.split(/\s+/)[1]; cur = { kind: w === 'n' ? 'n' : w === 'k' ? 'k' : 'nk', inData: false }; }
      continue;
    }
    if (!cur) continue;
    if (!t) continue;   // skip blanks (block scalars have none until they end)

    if (cur.kind === 'formula') {
      const rm = t.match(/^wavelength_range:\s*(.+)$/);
      if (rm) { const p = nums(rm[1]); if (p.length >= 2) cur.range = [p[0], p[1]]; continue; }
      const cm = t.match(/^coefficients:\s*(.+)$/);
      if (cm) { cur.coeffs = nums(cm[1]); continue; }
      continue;
    }
    // tabulated blocks
    if (t === 'data: |' || t === 'data: |+' || t === 'data: |-') { cur.inData = true; continue; }
    if (cur.inData) {
      if (line[0] !== ' ' && line[0] !== '\t') { cur.inData = false; continue; }   // dedent ends the block
      const p = nums(t);
      if (p.length >= 2) {
        if (cur.kind === 'k') kSamples.push([p[0], Math.abs(p[1])]);
        else { nSamples.push([p[0], p[1]]); if (cur.kind === 'nk' && p.length >= 3) kSamples.push([p[0], Math.abs(p[2])]); }
      }
    }
  }
  if (cur && cur.kind === 'formula') flushFormula(cur);

  nSamples.sort((a, b) => a[0] - b[0]);
  kSamples.sort((a, b) => a[0] - b[0]);
  if (nSamples.length === 0) return [];
  const interpK = (wl: number): number => {
    if (kSamples.length === 0) return 0;
    if (wl <= kSamples[0][0]) return kSamples[0][1];
    const last = kSamples.length - 1;
    if (wl >= kSamples[last][0]) return kSamples[last][1];
    let lo = 0, hi = last;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (kSamples[m][0] <= wl) lo = m; else hi = m; }
    const f = (wl - kSamples[lo][0]) / (kSamples[hi][0] - kSamples[lo][0]);
    return kSamples[lo][1] + f * (kSamples[hi][1] - kSamples[lo][1]);
  };
  return nSamples.map(([wl, n]) => [wl, n, interpK(wl)]);
}

// --- Lazy loaders with caching ---

let catalogPromise: Promise<RIIMaterial[]> | null = null;
const nkCache = new Map<string, number[][]>();

export function loadCatalog(): Promise<RIIMaterial[]> {
  if (!catalogPromise) {
    catalogPromise = fetch(`${BASE_URL}/catalog-nk.yml`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.text(); })
      .then(parseCatalog);
  }
  return catalogPromise;
}

export async function fetchMaterialNK(dataPath: string): Promise<number[][]> {
  const cached = nkCache.get(dataPath);
  if (cached) return cached;
  const resp = await fetch(`${BASE_URL}/data/${dataPath}`);
  if (!resp.ok) throw new Error(`${resp.status}`);
  const data = parseNKData(await resp.text());
  if (data.length === 0) throw new Error('No tabulated nk data found');
  data.sort((a, b) => a[0] - b[0]);
  nkCache.set(dataPath, data);
  return data;
}

export function searchCatalog(catalog: RIIMaterial[], query: string, limit = 30): RIIMaterial[] {
  if (!query) return catalog.slice(0, limit);
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  return catalog.filter(m => {
    const hay = `${m.book} ${m.bookName} ${m.pageName}`.toLowerCase();
    return terms.every(t => hay.includes(t));
  }).slice(0, limit);
}
