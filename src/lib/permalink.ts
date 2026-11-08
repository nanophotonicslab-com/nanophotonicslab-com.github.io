/**
 * Lightweight URL-parameter ↔ form-field sync for shareable tool states.
 *
 * Usage in an Astro page <script>:
 *   import { initPermalinks } from '../../lib/permalink';
 *   initPermalinks({ material: 'material-select', r: 'r-num', nh: 'nh-num' });
 *
 * A parameter is written only while its field differs from the field's own
 * default, so a link carries the settings that were CHANGED and nothing else.
 * That is what keeps a long map readable: restoring is "load the page, then
 * apply these", and the page already loads at its defaults.
 */

type ParamMap = Record<string, string>; // URL param name → DOM element ID
type Field = HTMLInputElement | HTMLSelectElement;

/** The inputs whose state is `checked`, not `value`, or null for the rest.
 *
 *  Deliberately NOT a `el is HTMLInputElement` type guard: that would also
 *  tell the compiler that a field which is not a toggle cannot be an input,
 *  which narrowed every ordinary number box to `never`. */
function asToggle(el: Field): HTMLInputElement | null {
  return el instanceof HTMLInputElement && (el.type === 'checkbox' || el.type === 'radio')
    ? el
    : null;
}

/** What this field currently says, as a string. */
function fieldValue(el: Field): string {
  const t = asToggle(el);
  return t ? (t.checked ? '1' : '0') : el.value;
}

/** What it would say on a freshly loaded page, from the DOM alone.
 *
 *  UNRELIABLE FOR `type="hidden"`. Its `value` IDL attribute reflects the
 *  content attribute directly, so assigning `.value` moves `defaultValue` with
 *  it and a hidden field always looks untouched -- which silently dropped the
 *  mirrors that carry a page's view state out of every share link. The
 *  snapshot taken in `initPermalinks` is what covers that; this is the
 *  fallback for fields that did not exist yet when it was taken. */
function domDefault(el: Field): string {
  const t = asToggle(el);
  if (t) return t.defaultChecked ? '1' : '0';
  if (el instanceof HTMLSelectElement) {
    const def = [...el.options].find(o => o.defaultSelected) ?? el.options[0];
    return def ? def.value : '';
  }
  return el.defaultValue;
}

/** Apply a string from the URL, in whichever way this field takes one. */
function setField(el: Field, value: string): void {
  const t = asToggle(el);
  if (t) t.checked = value !== '0' && value !== '' && value !== 'false';
  else el.value = value;
}

export function initPermalinks(paramMap: ParamMap): void {
  const url = new URL(window.location.href);
  // `hasOwn`, not `in`: `paramMap` is a plain object, so `?toString=1` or
  // `?constructor=1` on an otherwise bare link matched through the prototype
  // chain and started a restore that had nothing to restore.
  const hasParams = [...url.searchParams.keys()].some(k => Object.hasOwn(paramMap, k));

  // What every tracked field said before anything touched it. Taken now, which
  // is after the page's own setup ran and before the restore below, so it is
  // the state a reader opening the bare URL would see -- exactly what a share
  // link may leave out. Fields built later (a layer row, a card that appears
  // with a count) are absent here and fall back to `domDefault`.
  const atLoad = new Map<string, string>();
  for (const elementId of Object.values(paramMap)) {
    const el = document.getElementById(elementId);
    if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement) {
      atLoad.set(elementId, fieldValue(el));
    }
  }

  const fieldDefault = (id: string, el: Field): string => atLoad.get(id) ?? domDefault(el);

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

    let restored = false;
    const restore = () => {
      if (restored) return;
      restored = true;
      // In map order, which is the order the page declared them: a field that
      // decides how many of the later ones exist, or what their numbers mean,
      // is listed before them, so its handler has run by the time they land.
      for (const [id, value] of inputs) {
        const el = document.getElementById(id) as HTMLInputElement | null;
        if (!el) continue;
        setField(el, value);
        el.dispatchEvent(new Event('input', { bubbles: true }));
        if (asToggle(el)) el.dispatchEvent(new Event('change', { bubbles: true }));
      }
    };

    // Deferred so the selects' own handlers have rebuilt whatever they rebuild
    // before these land -- but NOT on a paint alone. `requestAnimationFrame`
    // never fires in a tab that is not being painted: a link opened in a
    // background tab, a session restored at startup, an offscreen preview.
    // Measured in this app's own browser pane, where no frame arrives although
    // `document.visibilityState` reads "visible". Restoring is not a visual
    // effect -- skip it and the reader gets a page of defaults that looks
    // exactly like a link that carried no settings, with nothing to say
    // otherwise. So a timer races the frame, and `restored` keeps it to one.
    requestAnimationFrame(restore);
    setTimeout(restore, 0);
  }

  /** Write one field into `u`, or drop it when it sits at its default. */
  function writeParam(u: URL, param: string, elementId: string, el: Field): void {
    const value = fieldValue(el);
    if (value === fieldDefault(elementId, el)) u.searchParams.delete(param);
    else u.searchParams.set(param, value);
  }

  // Update URL silently on any tracked field change
  for (const [param, elementId] of Object.entries(paramMap)) {
    const el = document.getElementById(elementId);
    if (!(el instanceof HTMLInputElement) && !(el instanceof HTMLSelectElement)) continue;
    const eventName = el instanceof HTMLSelectElement ? 'change' : 'input';
    el.addEventListener(eventName, () => {
      const u = new URL(window.location.href);
      writeParam(u, param, elementId, el);
      history.replaceState(null, '', u.toString());
    });
  }

  /**
   * The current state of every tracked field, as a URL.
   *
   * NOT `window.location.href`. The listeners above only write a parameter
   * once its field has fired an event, so a link copied from a freshly loaded
   * page -- where the reader changed nothing, or changed only things those
   * listeners cannot see, such as which panel they are looking at -- came out
   * as the bare page address and restored nothing. The button offers to share
   * the settings, so it reads them, all of them, when it is pressed.
   */
  function shareUrl(): string {
    const u = new URL(window.location.href);
    for (const [param, elementId] of Object.entries(paramMap)) {
      const el = document.getElementById(elementId);
      if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement) {
        writeParam(u, param, elementId, el);
      }
    }
    return u.toString();
  }

  // What a copy-link button looked like before it said "Copied!", and the timer
  // that will put it back. Held across clicks, because a second press while the
  // first is still showing would otherwise capture the string "Copied!" as the
  // thing to restore -- and the later timer wins, so the button kept that text,
  // without its icon, for good.
  const restoring = new WeakMap<Element, { html: string; timer: ReturnType<typeof setTimeout> }>();

  // Wire up any copy-link buttons on the page
  document.querySelectorAll('[data-copy-link]').forEach(btn => {
    btn.addEventListener('click', () => {
      // Into the address bar too, so the copied link and the one the reader
      // can see are the same thing.
      const href = shareUrl();
      history.replaceState(null, '', href);

      /** Say something on the button, then put it back as it was.
       *
       *  innerHTML, not textContent, in BOTH directions. Several of these
       *  buttons are an inline <svg> plus a <span> (PresetStrip's "Copy
       *  link"), and writing textContent deletes the icon element outright --
       *  putting a captured string back then restores the words without it. */
      const say = (msg: string) => {
        const prior = restoring.get(btn);
        if (prior) clearTimeout(prior.timer);
        const html = prior ? prior.html : btn.innerHTML;
        btn.textContent = msg;
        const timer = setTimeout(() => {
          btn.innerHTML = html;
          restoring.delete(btn);
        }, 1500);
        restoring.set(btn, { html, timer });
      };

      // The clipboard CAN refuse -- it rejects when the document is not
      // focused, and on any page served without a secure context. Without a
      // catch that rejection went unhandled, the button sat unchanged, and the
      // reader was left thinking a copy had happened. The link is in the
      // address bar either way, which is what the fallback points at.
      navigator.clipboard
        .writeText(href)
        .then(() => say('Copied!'))
        .catch(() => say('Press Ctrl+C'));
    });
  });
}
