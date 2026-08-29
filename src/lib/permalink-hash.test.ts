import { describe, it, expect } from 'vitest';
import { reloadOnPermalinkChange } from './permalink-hash';

/** Minimal stand-in for `window`, so these run without a DOM. */
function fakeTarget() {
  const listeners: Record<string, ((e?: unknown) => void)[]> = {};
  return {
    addEventListener(type: string, fn: () => void) {
      (listeners[type] ||= []).push(fn);
    },
    removeEventListener(type: string, fn: () => void) {
      listeners[type] = (listeners[type] || []).filter(f => f !== fn);
    },
    dispatchEvent(event: Event) {
      (listeners[event.type] || []).forEach(fn => fn(event));
      return true;
    },
    count(type: string) {
      return (listeners[type] || []).length;
    },
  };
}

function harness(initial: string) {
  const target = fakeTarget();
  let hash = initial;
  let reloads = 0;
  const stop = reloadOnPermalinkChange({
    target: target as unknown as EventTarget,
    readHash: () => hash,
    reload: () => { reloads += 1; },
  });
  return {
    stop,
    reloads: () => reloads,
    listeners: () => target.count('hashchange'),
    navigate(next: string) {
      hash = next;
      target.dispatchEvent({ type: 'hashchange' } as Event);
    },
  };
}

describe('reloadOnPermalinkChange', () => {
  it('reloads when the hash moves to a different encoded state', () => {
    const h = harness('sp=AAA');
    h.navigate('sp=BBB');
    expect(h.reloads()).toBe(1);
  });

  it('reloads when a state arrives on a page opened without one', () => {
    const h = harness('');
    h.navigate('exc=electron&r=25');
    expect(h.reloads()).toBe(1);
  });

  it('does not reload when the hash returns to the state already applied', () => {
    // What the back button does after the page itself rewrote the URL.
    const h = harness('sp=AAA');
    h.navigate('sp=BBB');
    h.navigate('sp=AAA');
    expect(h.reloads()).toBe(1);
  });

  it('ignores navigation anchors, so switching tabs does not reload', () => {
    const h = harness('');
    for (const anchor of ['drude', 'graphene', 'diffusion-and-tracking']) {
      h.navigate(anchor);
    }
    expect(h.reloads()).toBe(0);
  });

  it('ignores an emptied hash', () => {
    const h = harness('sp=AAA');
    h.navigate('');
    expect(h.reloads()).toBe(0);
  });

  it('honours a custom state test', () => {
    const target = fakeTarget();
    let hash = '';
    let reloads = 0;
    reloadOnPermalinkChange({
      target: target as unknown as EventTarget,
      readHash: () => hash,
      reload: () => { reloads += 1; },
      isState: h => h.startsWith('g='),
    });
    hash = 'sp=AAA';
    target.dispatchEvent({ type: 'hashchange' } as Event);
    expect(reloads).toBe(0);
    hash = 'g=BBB';
    target.dispatchEvent({ type: 'hashchange' } as Event);
    expect(reloads).toBe(1);
  });

  it('stops listening when the returned function is called', () => {
    const h = harness('sp=AAA');
    expect(h.listeners()).toBe(1);
    h.stop();
    expect(h.listeners()).toBe(0);
    h.navigate('sp=BBB');
    expect(h.reloads()).toBe(0);
  });

  it('is inert where there is no window to listen on', () => {
    expect(() => reloadOnPermalinkChange({ target: undefined })()).not.toThrow();
  });
});
