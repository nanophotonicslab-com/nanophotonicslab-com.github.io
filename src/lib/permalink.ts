/**
 * Lightweight URL-parameter ↔ form-field sync for shareable tool states.
 *
 * Usage in an Astro page <script>:
 *   import { initPermalinks } from '../../lib/permalink';
 *   initPermalinks({ material: 'material-select', r: 'r-num', nh: 'nh-num' });
 */

type ParamMap = Record<string, string>; // URL param name → DOM element ID

export function initPermalinks(paramMap: ParamMap): void {
  const url = new URL(window.location.href);
  const hasParams = [...url.searchParams.keys()].some(k => k in paramMap);

  if (hasParams) {
    // Partition into selects (apply first) and inputs
    const selects: [string, string][] = [];
    const inputs: [string, string][] = [];

    for (const [param, elementId] of Object.entries(paramMap)) {
      const value = url.searchParams.get(param);
      if (value === null) continue;
      const el = document.getElementById(elementId);
      if (!el) continue;
      if (el instanceof HTMLSelectElement) selects.push([elementId, value]);
      else inputs.push([elementId, value]);
    }

    for (const [id, value] of selects) {
      const el = document.getElementById(id) as HTMLSelectElement;
      el.value = value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    requestAnimationFrame(() => {
      for (const [id, value] of inputs) {
        const el = document.getElementById(id) as HTMLInputElement | null;
        if (!el) continue;
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
  }

  // Build reverse map: element ID → URL param name
  const idToParam: Record<string, string> = {};
  for (const [param, id] of Object.entries(paramMap)) {
    idToParam[id] = param;
  }

  // Update URL silently on any tracked field change
  for (const [param, elementId] of Object.entries(paramMap)) {
    const el = document.getElementById(elementId);
    if (!el) continue;
    const eventName = el instanceof HTMLSelectElement ? 'change' : 'input';
    el.addEventListener(eventName, () => {
      const u = new URL(window.location.href);
      u.searchParams.set(param, (el as HTMLInputElement | HTMLSelectElement).value);
      history.replaceState(null, '', u.toString());
    });
  }

  // Wire up any copy-link buttons on the page
  document.querySelectorAll('[data-copy-link]').forEach(btn => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(window.location.href).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      });
    });
  });
}
