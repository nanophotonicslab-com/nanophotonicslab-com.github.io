import { describe, it, expect, afterEach, vi } from 'vitest';

/**
 * `permalink.ts` is DOM code and this suite runs on bare Node, so the handful
 * of globals it actually touches are built here by hand: `getElementById`,
 * `querySelectorAll`, `window.location.href`, `history.replaceState`,
 * `navigator.clipboard`, `requestAnimationFrame`.
 *
 * The CLASSES matter as much as the objects. The module decides what counts as
 * a field with `instanceof HTMLInputElement` / `instanceof HTMLSelectElement`,
 * so the fakes have to be genuine instances of globals under those names. A
 * duck-typed object is not a field as far as the module is concerned, and every
 * test below would pass vacuously while tracking nothing at all.
 */

/** Everything a dispatched event needs: listeners, plus a name for the log. */
class FakeNode {
  private readonly listeners = new Map<string, Array<(e: Event) => void>>();

  constructor(readonly id: string) {}

  addEventListener(type: string, fn: (e: Event) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(fn);
    this.listeners.set(type, list);
  }

  dispatchEvent(event: Event): boolean {
    // Logged on dispatch rather than from a listener, so the order recorded is
    // the order the module applied things, whether or not anyone was listening.
    events.push(`${this.id}:${event.type}`);
    for (const fn of this.listeners.get(event.type) ?? []) fn(event);
    return true;
  }
}

class FakeInput extends FakeNode {
  readonly type: string;
  checked: boolean;
  defaultChecked: boolean;
  private current: string;
  private asLoaded: string;

  constructor(id: string, init: { type?: string; value?: string; checked?: boolean } = {}) {
    super(id);
    this.type = init.type ?? 'number';
    // `value="..."` in the markup is both the value and the default value.
    this.current = init.value ?? '';
    this.asLoaded = this.current;
    this.checked = init.checked ?? false;
    this.defaultChecked = this.checked;
  }

  get value(): string {
    return this.current;
  }

  set value(v: string) {
    this.current = v;
    // The quirk the init snapshot exists for. For `type="hidden"` the `value`
    // IDL attribute reflects the content attribute, so the browser moves
    // `defaultValue` along with every assignment and the field can never look
    // edited. Any other input keeps the value it was loaded with.
    if (this.type === 'hidden') this.asLoaded = v;
  }

  get defaultValue(): string {
    return this.asLoaded;
  }
}

class FakeSelect extends FakeNode {
  value: string;
  readonly options: Array<{ value: string; defaultSelected: boolean }>;

  constructor(id: string, values: string[], selected = 0) {
    super(id);
    this.options = values.map((value, i) => ({ value, defaultSelected: i === selected }));
    this.value = values[selected] ?? '';
  }
}

function unescapeHtml(s: string): string {
  return s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}

class FakeButton extends FakeNode {
  private html: string;

  constructor(id: string, html: string) {
    super(id);
    this.html = html;
  }

  get innerHTML(): string {
    return this.html;
  }

  set innerHTML(v: string) {
    this.html = v;
  }

  get textContent(): string {
    return unescapeHtml(this.html.replace(/<[^>]*>/g, ''));
  }

  /** Writing `textContent` replaces the children with a single text node, which
   *  is exactly what deletes an inline <svg> icon in a browser. Markup written
   *  this way comes back ESCAPED, so putting the captured string back through
   *  `textContent` would leave the button reading `&lt;svg ...` rather than
   *  showing an icon -- modelled here so a test can tell the two apart. */
  set textContent(v: string) {
    this.html = v.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
}

/* The mutable state the stubbed globals read. `page()` resets all of it. */
let fields = new Map<string, FakeInput | FakeSelect>();
let buttons: FakeButton[] = [];
let frames: Array<() => void> = [];
let pushed: string[] = [];
let copied: string[] = [];
let events: string[] = [];
const loc = { href: '' };

/** `defineProperty`, not assignment: Node already has a getter-only
 *  `globalThis.navigator`, and plain assignment to it throws in a module. */
function define(name: string, value: unknown): void {
  Object.defineProperty(globalThis, name, { value, writable: true, configurable: true });
}

define('HTMLInputElement', FakeInput);
define('HTMLSelectElement', FakeSelect);
define('document', {
  getElementById: (id: string) => fields.get(id) ?? null,
  // The module only ever asks for '[data-copy-link]'.
  querySelectorAll: () => buttons,
});
define('window', { location: loc });
define('history', {
  replaceState: (_state: unknown, _title: string, url: string) => {
    // A real `replaceState` moves the address bar, and the module re-reads
    // `window.location.href` on every later write. Without that feedback a
    // parameter could never be observed being deleted again.
    loc.href = url;
    pushed.push(url);
  },
});
define('navigator', {
  clipboard: {
    writeText: (text: string) => {
      copied.push(text);
      return Promise.resolve();
    },
  },
});
define('requestAnimationFrame', (cb: () => void) => frames.push(cb));

// Imported after the globals exist, because a static import is hoisted above
// all of the above and the module resolves those names from the global scope.
const { initPermalinks } = await import('./permalink');

const PAGE = 'https://nanophotonicslab.com/lab/pulse/';

function page(spec: {
  url?: string;
  fields: Record<string, FakeInput | FakeSelect>;
  buttons?: FakeButton[];
}) {
  fields = new Map(Object.entries(spec.fields));
  buttons = spec.buttons ?? [];
  frames = [];
  pushed = [];
  copied = [];
  events = [];
  loc.href = spec.url ?? PAGE;

  return {
    /** Run the queued animation frame, i.e. let the input restore happen. */
    frame(): void {
      const queued = frames;
      frames = [];
      for (const cb of queued) cb();
    },
    pending: () => frames.length,
    events: () => events,
    pushed: () => pushed,
    copied: () => copied,
    href: () => loc.href,
    params: () => new URL(loc.href).searchParams,
    /** What a reader's edit looks like: a new value plus the event it fires. */
    edit(field: FakeInput | FakeSelect, value: string): void {
      field.value = value;
      field.dispatchEvent(new Event(field instanceof FakeSelect ? 'change' : 'input'));
    },
    toggle(field: FakeInput, checked: boolean): void {
      field.checked = checked;
      field.dispatchEvent(new Event('input'));
    },
    click(btn: FakeButton): void {
      btn.dispatchEvent(new Event('click'));
    },
  };
}

/** Let the `writeText(...).then(...)` continuation in the click handler run. */
const settle = () => Promise.resolve().then(() => {});

/** Fake the two timer functions the copy-link handler uses, and only those.
 *
 *  `clearTimeout` has to come along with `setTimeout`: the handler cancels a
 *  pending restore with the handle `setTimeout` gave it, and the real
 *  `clearTimeout` cannot cancel a fake timer. `requestAnimationFrame` is
 *  deliberately left out -- it is this suite's own stub above, and sinon would
 *  otherwise take it over. */
const useTimers = () => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });

afterEach(() => {
  vi.useRealTimers();
});

describe('initPermalinks', () => {
  describe('restoring from the URL', () => {
    it('applies a parameter from the URL to its field and fires its event', () => {
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({ url: `${PAGE}?r=25`, fields: { 'r-num': r } });

      initPermalinks({ r: 'r-num' });
      h.frame();

      expect(r.value).toBe('25');
      // The event is the restore: the page's own handler is what turns a number
      // in a box into a redrawn figure.
      expect(h.events()).toContain('r-num:input');
    });

    it('restores even when no animation frame ever arrives', () => {
      // A tab that is never painted gets no `requestAnimationFrame` callback at
      // all -- a link opened in a background tab, a session restored at
      // startup, an offscreen preview. This was real, not hypothetical: in the
      // desktop app's own browser pane a frame never arrives although
      // `document.visibilityState` reads "visible", and a shared link opened
      // there restored nothing and said nothing, which is indistinguishable
      // from a link that carried no settings. So the restore races a timer
      // against the frame, and this pins the timer leg by never calling
      // `h.frame()`.
      useTimers();
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({ url: `${PAGE}?r=25`, fields: { 'r-num': r } });

      initPermalinks({ r: 'r-num' });
      expect(r.value).toBe('50'); // still deferred, as it always was
      vi.advanceTimersByTime(0);

      expect(r.value).toBe('25');
      expect(h.events()).toContain('r-num:input');

      // And exactly once: the frame is still queued, so flushing it as well
      // must not fire a second round of events at the page.
      h.frame();
      expect(h.events().filter(e => e === 'r-num:input')).toHaveLength(1);
    });

    it('leaves a field alone when its parameter is absent from the URL', () => {
      const r = new FakeInput('r-num', { value: '50' });
      const nh = new FakeInput('nh-num', { value: '1.33' });
      const h = page({ url: `${PAGE}?r=25`, fields: { 'r-num': r, 'nh-num': nh } });

      initPermalinks({ r: 'r-num', nh: 'nh-num' });
      h.frame();

      expect(nh.value).toBe('1.33');
      expect(h.events()).not.toContain('nh-num:input');
    });

    it('applies selects before inputs, synchronously, ahead of the frame', () => {
      // A select usually decides what the numbers MEAN -- which material, which
      // model -- so it has to be in force before any number lands. The map here
      // lists the select LAST, to show this is the select/input split doing it
      // and not map order.
      const r = new FakeInput('r-num', { value: '50' });
      const mat = new FakeSelect('material-select', ['bk7', 'sf11']);
      const h = page({
        url: `${PAGE}?r=25&material=sf11`,
        fields: { 'r-num': r, 'material-select': mat },
      });

      initPermalinks({ r: 'r-num', material: 'material-select' });

      // Before the frame runs the select is already done and the input untouched.
      expect(mat.value).toBe('sf11');
      expect(r.value).toBe('50');

      h.frame();
      expect(h.events()).toEqual(['material-select:change', 'r-num:input']);
    });

    it('applies inputs in paramMap insertion order, not alphabetically', () => {
      // The pages treat this order as a contract: a count that builds the later
      // rows, or a unit that decides what their numbers mean, is declared first
      // so its handler has run before they land. The URL below deliberately
      // carries the parameters in a different order, and alphabetically these
      // ids would come out a-num, m-num, n-layers -- so only map order gives
      // the sequence asserted here.
      const layers = new FakeInput('n-layers', { value: '1' });
      const a = new FakeInput('a-num', { value: '10' });
      const m = new FakeInput('m-num', { value: '2' });
      const h = page({
        url: `${PAGE}?a=20&mode=3&nl=4`,
        fields: { 'n-layers': layers, 'a-num': a, 'm-num': m },
      });

      initPermalinks({ nl: 'n-layers', a: 'a-num', mode: 'm-num' });
      h.frame();

      expect(h.events()).toEqual(['n-layers:input', 'a-num:input', 'm-num:input']);
    });

    it("round-trips checkbox and radio fields as '1'/'0' through `checked`", () => {
      const log = new FakeInput('log-cb', { type: 'checkbox', checked: false });
      const tm = new FakeInput('tm-radio', { type: 'radio', value: 'tm', checked: true });
      const h = page({
        url: `${PAGE}?log=1&tm=0`,
        fields: { 'log-cb': log, 'tm-radio': tm },
      });

      initPermalinks({ log: 'log-cb', tm: 'tm-radio' });
      h.frame();

      expect(log.checked).toBe(true);
      expect(tm.checked).toBe(false);
      // A radio's `value` is its identity, not its state, and has to survive
      // the restore untouched -- it is how the page knows which radio this is.
      expect(tm.value).toBe('tm');
      // Toggles get a `change` as well as an `input`, because plenty of
      // handlers listen for only one of the two.
      expect(h.events()).toEqual([
        'log-cb:input',
        'log-cb:change',
        'tm-radio:input',
        'tm-radio:change',
      ]);

      // Both now differ from what they said at load, so both stay in the link.
      expect(h.params().get('log')).toBe('1');
      expect(h.params().get('tm')).toBe('0');

      // And unticking the checkbox again takes it back out.
      h.toggle(log, false);
      expect(h.params().has('log')).toBe(false);
    });

    it('queues no frame when the URL carries none of the tracked parameters', () => {
      // Nothing to restore, so nothing is applied and no field is disturbed --
      // a tracking parameter on a bare link must not look like a shared state.
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({ url: `${PAGE}?utm_source=mastodon`, fields: { 'r-num': r } });

      initPermalinks({ r: 'r-num' });

      expect(h.pending()).toBe(0);
      expect(h.events()).toEqual([]);
    });

    it('does not mistake an inherited property name for a tracked parameter', () => {
      // `paramMap` is a plain object, so testing membership with `in` answered
      // yes for every name on `Object.prototype` -- a link carrying
      // ?toString=1 looked like a shared state and started a restore with
      // nothing in it to restore.
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({
        url: `${PAGE}?toString=1&constructor=2`,
        fields: { 'r-num': r },
      });

      initPermalinks({ r: 'r-num' });

      expect(h.pending()).toBe(0);
      expect(h.events()).toEqual([]);
      expect(r.value).toBe('50');
    });

    it('skips tracked ids that are not on the page', () => {
      // The electrostatics map is generated -- five ribbons of eight numbers and
      // eight layer rows -- and most of those ids do not exist until the counts
      // say so. Every loop has to tolerate a missing element rather than throw
      // and leave the rest of the page unwired.
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({ url: `${PAGE}?r=25&l3=7`, fields: { 'r-num': r } });

      expect(() => initPermalinks({ r: 'r-num', l3: 'efd-l-3' })).not.toThrow();
      h.frame();

      expect(r.value).toBe('25');
    });
  });

  describe('writing the URL', () => {
    it('does not write a field that still sits at its default', () => {
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({ fields: { 'r-num': r } });

      initPermalinks({ r: 'r-num' });
      // An event without a change: a slider handler re-assigning the same
      // number, or a reset back to where the page started. A link that carries
      // the defaults is unreadable, and that is the whole point of the module.
      r.dispatchEvent(new Event('input'));

      expect(h.params().has('r')).toBe(false);
      expect(h.href()).toBe(PAGE);
    });

    it('deletes the parameter when a field is edited back to its default', () => {
      const r = new FakeInput('r-num', { value: '50' });
      const h = page({ fields: { 'r-num': r } });

      initPermalinks({ r: 'r-num' });

      h.edit(r, '99');
      expect(h.params().get('r')).toBe('99');

      // Going back to 50 has to REMOVE r=99 rather than leave the link
      // describing a radius the page is no longer showing.
      h.edit(r, '50');
      expect(h.params().has('r')).toBe(false);
      expect(h.href()).toBe(PAGE);
    });

    it('writes a hidden mirror that changed after init, which the DOM default cannot see', () => {
      // THE BUG THE SNAPSHOT EXISTS FOR. `view-input` is the pattern the labs
      // use to carry which tab is open into a link: a plot-tab click writes the
      // hidden input and fires `input` on it.
      const view = new FakeInput('view-input', { type: 'hidden', value: 'tau' });
      const h = page({ fields: { 'view-input': view } });

      initPermalinks({ view: 'view-input' });

      view.value = 'gvd';
      // This is what the field now claims about itself in a browser: a hidden
      // input's `value` and `defaultValue` move together, so asking the DOM
      // "were you edited?" answers no, and the open tab silently dropped out of
      // every share link. Only the snapshot taken at init still remembers
      // 'tau'.
      expect(view.defaultValue).toBe('gvd');

      view.dispatchEvent(new Event('input'));
      expect(h.params().get('view')).toBe('gvd');
    });

    it('keeps a hidden mirror out of the link while it still says what it said at load', () => {
      // The other side of the snapshot: remembering the load value must not
      // turn into writing every hidden field unconditionally.
      const view = new FakeInput('view-input', { type: 'hidden', value: 'tau' });
      const h = page({ fields: { 'view-input': view } });

      initPermalinks({ view: 'view-input' });
      view.dispatchEvent(new Event('input'));

      expect(h.params().has('view')).toBe(false);
    });
  });

  describe('the Share button', () => {
    function sharePage() {
      const mat = new FakeSelect('material-select', ['bk7', 'sf11']);
      const tau = new FakeInput('tau-num', { value: '30' });
      const view = new FakeInput('view-input', { type: 'hidden', value: 'tau' });
      const copy = new FakeButton('copy', '<svg class="icon" aria-hidden="true"></svg><span>Copy link</span>');
      const h = page({
        fields: { 'material-select': mat, 'tau-num': tau, 'view-input': view },
        buttons: [copy],
      });
      initPermalinks({ material: 'material-select', tau: 'tau-num', view: 'view-input' });
      return { h, mat, tau, view, copy };
    }

    it('copies the state of every tracked field even when no field ever fired an event', () => {
      const { h, mat, tau, view, copy } = sharePage();

      // A freshly loaded page, then changes made the way the labs make them:
      // a tab click and a slider drag assign `.value` in script, which fires
      // nothing at all. The per-field listeners have therefore never run and
      // the address bar is still the bare page.
      mat.value = 'sf11';
      tau.value = '12';
      view.value = 'gvd';
      expect(h.pushed()).toEqual([]);

      h.click(copy);

      const link = new URL(h.copied()[0]);
      expect(link.searchParams.get('material')).toBe('sf11');
      expect(link.searchParams.get('tau')).toBe('12');
      expect(link.searchParams.get('view')).toBe('gvd');
    });

    it('puts the copied link in the address bar, so the two cannot disagree', () => {
      const { h, tau, copy } = sharePage();

      tau.value = '12';
      h.click(copy);

      expect(h.copied()).toHaveLength(1);
      expect(h.pushed()).toEqual(h.copied());
      expect(h.href()).toBe(h.copied()[0]);
    });

    it('restores the button innerHTML after "Copied!", so the icon survives', async () => {
      useTimers();
      const { h, copy } = sharePage();
      const icon = copy.innerHTML;

      h.click(copy);
      await settle();

      expect(copy.textContent).toBe('Copied!');
      // The <svg> element is gone at this point, which is the whole problem:
      // the confirmation has to be undone from a captured innerHTML, because
      // nothing in the DOM still holds the icon.
      expect(copy.innerHTML).toBe('Copied!');

      vi.advanceTimersByTime(1500);
      expect(copy.innerHTML).toBe(icon);
      expect(copy.innerHTML).toContain('<svg');
    });

    it('survives a second press while "Copied!" is still showing', async () => {
      // Two presses inside the same 1500 ms window used to leave the button
      // reading "Copied!" for good: the second press captured the confirmation
      // text itself as the thing to restore, and its timer fired last, so it
      // overwrote the icon the first press had just put back.
      useTimers();
      const { h, copy } = sharePage();
      const icon = copy.innerHTML;

      h.click(copy);
      await settle();
      vi.advanceTimersByTime(500);

      h.click(copy);
      await settle();
      expect(copy.textContent).toBe('Copied!');

      // Past both timers: the one the first press started and the one the
      // second did. Whichever lands last has to leave the icon behind.
      vi.advanceTimersByTime(5000);
      expect(copy.innerHTML).toBe(icon);
      expect(copy.innerHTML).toContain('<svg');
    });
  });
});
