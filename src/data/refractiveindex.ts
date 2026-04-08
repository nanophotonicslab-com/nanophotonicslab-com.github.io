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

// --- Material data parser (tabulated nk) ---

function parseNKData(text: string): number[][] {
  const data: number[][] = [];
  let inBlock = false;
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (t === 'data: |' || t === 'data: |+') { inBlock = true; continue; }
    if (inBlock) {
      if (t === '' && data.length > 0) continue;
      if (line[0] !== ' ' && line[0] !== '\t' && t !== '') break;
      const parts = t.split(/\s+/).map(Number);
      if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        data.push([parts[0], parts[1], parts.length >= 3 && !isNaN(parts[2]) ? Math.abs(parts[2]) : 0]);
      }
    }
  }
  return data;
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
