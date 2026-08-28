/**
 * The two ways a Lab tool lets a user bring in their own optical constants:
 * searching refractiveindex.info, and uploading a file.
 *
 * Both had been reimplemented per page — the database picker seven times, the
 * upload path four — and the copies have drifted in signature, in the strings
 * they show, and in the option values they encode. These helpers take element
 * references and own no markup, because each page lays its controls out
 * differently; the classes they rely on (.rii-result, .file-name, …) already
 * live in the global src/styles/lab.css, so no page-scoped CSS is involved and
 * innerHTML-injected rows style correctly without :global.
 *
 * Both paths converge on the one shape the rest of the toolkit understands:
 * `[[λ_µm, n, k], …]`, exactly what `interpolateNK` consumes.
 */
import { loadCatalog, searchCatalog, fetchMaterialNK, type RIIMaterial } from '../data/refractiveindex';

export interface RIISearchElements {
  /** Wrapper shown while the "search database" option is selected. */
  group: HTMLElement;
  input: HTMLInputElement;
  results: HTMLElement;
  /** Short status line: catalog size, loading, point count, errors. */
  status: HTMLElement;
  /** The page's provenance caption, set to the chosen entry's page name. */
  ref: HTMLElement;
}

export interface RIISearchHandle {
  /** Load the catalog if needed and render the current query's hits. */
  ensureCatalog(): Promise<void>;
  /**
   * Re-select an entry by its dataPath, without the user searching — this is
   * what makes a shared link reproduce the same material. Resolves false when
   * the entry is no longer in the catalog, so the caller can say so rather than
   * silently showing something else.
   */
  selectByPath(dataPath: string): Promise<boolean>;
  /** dataPath of the current selection, or null. This is the permalink key. */
  currentPath(): string | null;
  clear(): void;
}

/**
 * Wire a refractiveindex.info search box.
 *
 * `onLoaded` receives the canonical table and the entry it came from; the entry
 * carries the dataPath the caller must serialize for the permalink to survive.
 */
export function setupRIISearch(
  els: RIISearchElements,
  onLoaded: (data: number[][], material: RIIMaterial) => void,
): RIISearchHandle {
  let catalog: RIIMaterial[] | null = null;
  let selected: RIIMaterial | null = null;
  let debounce: ReturnType<typeof setTimeout> | undefined;

  /**
   * Catalog strings come from refractiveindex.info over the network and are
   * interpolated into innerHTML below, so they are escaped. The four page-local
   * copies of this picker do not escape; only workbench does. New shared code
   * should not carry that forward.
   */
  const esc = (s: string): string => s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  function showResults(q: string): void {
    if (!catalog) return;
    const hits = searchCatalog(catalog, q);
    els.results.innerHTML = hits.map((m, i) =>
      `<div class="rii-result" data-idx="${i}">`
      + `<span class="rii-result-book">${esc(m.bookName)}</span><br>`
      + `<span class="rii-result-page">${esc(m.pageName)}</span></div>`).join('');
    (els.results as unknown as { _hits: RIIMaterial[] })._hits = hits;
  }

  async function ensureCatalog(): Promise<void> {
    if (catalog) { showResults(els.input.value); return; }
    els.status.textContent = 'Loading catalog…';
    try {
      catalog = await loadCatalog();
      els.status.textContent = `${catalog.length} materials available`;
      showResults(els.input.value);
    } catch {
      els.status.textContent = 'Failed to load catalog';
    }
  }

  async function apply(m: RIIMaterial): Promise<boolean> {
    els.status.textContent = 'Loading…';
    els.results.innerHTML = '';
    try {
      const data = await fetchMaterialNK(m.dataPath);
      selected = m;
      els.ref.textContent = m.pageName;
      els.status.textContent = `${data.length} points loaded`;
      els.input.value = m.bookName;
      onLoaded(data, m);
      return true;
    } catch {
      els.status.textContent = 'Failed to load material';
      return false;
    }
  }

  els.input.addEventListener('input', () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => showResults(els.input.value), 150);
  });
  els.input.addEventListener('focus', () => { if (catalog) showResults(els.input.value); });

  els.results.addEventListener('click', (e) => {
    const el = (e.target as HTMLElement).closest('.rii-result') as HTMLElement | null;
    if (!el) return;
    const hits = (els.results as unknown as { _hits?: RIIMaterial[] })._hits;
    const m = hits?.[Number.parseInt(el.dataset.idx!, 10)];
    if (m) void apply(m);
  });

  return {
    ensureCatalog,
    async selectByPath(dataPath: string): Promise<boolean> {
      if (!catalog) {
        try { catalog = await loadCatalog(); } catch { els.status.textContent = 'Failed to load catalog'; return false; }
      }
      const m = catalog.find(e => e.dataPath === dataPath);
      if (!m) {
        els.status.textContent = 'Shared entry is not in the database';
        return false;
      }
      return apply(m);
    },
    currentPath: () => selected?.dataPath ?? null,
    clear() {
      selected = null;
      els.results.innerHTML = '';
      els.input.value = '';
      els.status.textContent = '';
    },
  };
}
