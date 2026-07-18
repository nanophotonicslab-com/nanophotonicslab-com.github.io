import { describe, it, expect } from 'vitest';
import { Graph, type NodeId } from './graph';

function makeGraph(edges: Record<NodeId, NodeId[]>, log: NodeId[]) {
  const g = new Graph();
  for (const [id, deps] of Object.entries(edges)) {
    g.add({ id, deps: () => deps, run: () => log.push(id) });
  }
  return g;
}

describe('Graph', () => {
  it('runs a chain dependency-first on invalidate of the root', () => {
    const log: NodeId[] = [];
    const g = makeGraph({ a: [], b: ['a'], c: ['b'] }, log);
    expect(g.invalidate('a')).toEqual(['a', 'b', 'c']);
    expect(log).toEqual(['a', 'b', 'c']);
  });

  it('does not touch upstream nodes on a mid-chain invalidate', () => {
    const log: NodeId[] = [];
    const g = makeGraph({ a: [], b: ['a'], c: ['b'] }, log);
    g.invalidate('b');
    expect(log).toEqual(['b', 'c']);
  });

  it('runs a diamond once per node, in a valid order', () => {
    const log: NodeId[] = [];
    const g = makeGraph({ src: [], l: ['src'], r: ['src'], sink: ['l', 'r'] }, log);
    g.invalidate('src');
    expect(log.filter((n) => n === 'sink')).toHaveLength(1);
    expect(log.indexOf('sink')).toBeGreaterThan(log.indexOf('l'));
    expect(log.indexOf('sink')).toBeGreaterThan(log.indexOf('r'));
  });

  it('honours dynamic re-wiring through deps()', () => {
    const log: NodeId[] = [];
    let source = 'a';
    const g = new Graph();
    g.add({ id: 'a', deps: () => [], run: () => log.push('a') });
    g.add({ id: 'b', deps: () => [], run: () => log.push('b') });
    g.add({ id: 'consumer', deps: () => [source], run: () => log.push('consumer') });
    g.invalidate('b');
    expect(log).toEqual(['b']);           // consumer wired to 'a' — untouched
    source = 'b';
    log.length = 0;
    g.invalidate('b');
    expect(log).toEqual(['b', 'consumer']); // re-wired — now downstream of 'b'
  });

  it('throws on cycles and on unknown nodes', () => {
    const log: NodeId[] = [];
    const g = makeGraph({ a: ['b'], b: ['a'] }, log);
    expect(() => g.invalidate('a')).toThrow(/cycle/);
    expect(() => g.invalidate('nope')).toThrow(/unknown/);
  });

  it('runAll covers every node dependency-first', () => {
    const log: NodeId[] = [];
    const g = makeGraph({ z: ['m'], m: [], q: ['z'] }, log);
    g.runAll();
    expect(log).toHaveLength(3);
    expect(log.indexOf('m')).toBeLessThan(log.indexOf('z'));
    expect(log.indexOf('z')).toBeLessThan(log.indexOf('q'));
  });
});
