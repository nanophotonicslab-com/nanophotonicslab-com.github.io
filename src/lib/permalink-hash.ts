/**
 * Make a tool react to its permalink changing while the page is already open.
 *
 * Every Lab tool encodes its full state in the URL hash and reads it once, on
 * load. Nothing listened for the hash changing afterwards, so opening a second
 * permalink for the same tool from an already-open page did nothing visible:
 * the URL changed, the tool kept the previous configuration, and the numbers on
 * screen silently belonged to the wrong link. Following two figure permalinks
 * from one article in the same tab hit exactly that. The browser's back button
 * had the same problem.
 *
 * The fix reloads rather than re-applying in place. That is deliberate: the
 * whole state lives in the URL by design, so a reload is not lossy, and it puts
 * the incoming permalink through the identical code path as a cold load instead
 * of a second, less-travelled one that each tool would have to get right on its
 * own — several restore in stages, with renders and availability updates
 * sequenced around them.
 *
 * A hash written by the page itself does not trigger this: the tools update the
 * URL with history.replaceState, which fires no hashchange event.
 */

export interface PermalinkChangeOptions {
  /**
   * Whether a hash payload is an encoded tool state rather than a plain
   * anchor. Defaults to requiring an "=", which every encoded state carries
   * (`sp=…`, `g=…`, `exc=…`) and no navigation anchor does (`#drude`,
   * `#graphene`, a tab slug), so switching tabs does not reload the page.
   */
  isState?: (hash: string) => boolean;
  /** Event target to listen on. Defaults to `window`. */
  target?: EventTarget;
  /** Reads the current hash payload. Defaults to `location.hash.slice(1)`. */
  readHash?: () => string;
  /** Performs the reload. Defaults to `location.reload()`. */
  reload?: () => void;
}

const looksLikeState = (hash: string): boolean => hash.includes('=');

/**
 * Reload the page when the URL hash changes to a different encoded state.
 *
 * Returns a function that removes the listener.
 */
export function reloadOnPermalinkChange(options: PermalinkChangeOptions = {}): () => void {
  const {
    isState = looksLikeState,
    target = typeof window !== 'undefined' ? window : undefined,
    readHash = () => location.hash.slice(1),
    reload = () => location.reload(),
  } = options;

  if (!target) return () => {};

  const applied = readHash();
  const handler = () => {
    const hash = readHash();
    // Nothing to restore, or the hash came back to what is already on screen.
    if (!hash || hash === applied) return;
    if (!isState(hash)) return;
    reload();
  };

  target.addEventListener('hashchange', handler);
  return () => target.removeEventListener('hashchange', handler);
}
