/**
 * Minimal dependency graph for the simulation workbench — the evaluation core
 * the future node editor sits on (docs/node-editor-spec.md: "graph engine,
 * eval order + caching").
 *
 * Nodes declare their dependencies dynamically via deps() (so a UI selector
 * can re-wire an edge between evaluations) and do their work in run().
 * invalidate(id) re-runs the node and every transitive dependent, in
 * topological order, exactly once each. Nothing else is touched.
 */

export type NodeId = string;

export interface GraphNode {
  id: NodeId;
  /** Current upstream dependencies; may change between calls (re-wiring). */
  deps: () => NodeId[];
  run: () => void;
}

export class Graph {
  private nodes = new Map<NodeId, GraphNode>();

  add(node: GraphNode): void {
    this.nodes.set(node.id, node);
  }

  /** Direct dependents of `id` under the current wiring. */
  private dependents(id: NodeId): NodeId[] {
    const out: NodeId[] = [];
    for (const n of this.nodes.values()) if (n.deps().includes(id)) out.push(n.id);
    return out;
  }

  /** Topological order of `ids` under the current wiring (edges dep → node). */
  private topo(ids: Set<NodeId>): NodeId[] {
    const order: NodeId[] = [];
    const state = new Map<NodeId, 'visiting' | 'done'>();
    const visit = (id: NodeId) => {
      const s = state.get(id);
      if (s === 'done') return;
      if (s === 'visiting') throw new Error(`graph cycle through "${id}"`);
      state.set(id, 'visiting');
      for (const d of this.nodes.get(id)!.deps()) if (ids.has(d)) visit(d);
      state.set(id, 'done');
      order.push(id);
    };
    for (const id of ids) visit(id);
    return order;
  }

  /** Re-run `id` and all transitive dependents, dependency-first, once each. */
  invalidate(id: NodeId): NodeId[] {
    if (!this.nodes.has(id)) throw new Error(`unknown node "${id}"`);
    const dirty = new Set<NodeId>([id]);
    const queue = [id];
    while (queue.length) {
      for (const dep of this.dependents(queue.shift()!)) {
        if (!dirty.has(dep)) { dirty.add(dep); queue.push(dep); }
      }
    }
    const order = this.topo(dirty);
    for (const n of order) this.nodes.get(n)!.run();
    return order;
  }

  /** Run every node once, dependency-first. */
  runAll(): NodeId[] {
    const order = this.topo(new Set(this.nodes.keys()));
    for (const n of order) this.nodes.get(n)!.run();
    return order;
  }
}
