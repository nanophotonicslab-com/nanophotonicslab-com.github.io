/**
 * Client-side module implementations, keyed by URL slug.
 *
 * The module page is one generic route, so its bundle needs every module's
 * implementation and picks one at runtime from the slug in the markup. Adding
 * module 2 means one line here, alongside its line in the lab registry.
 */
import type { ModuleImpl } from './types';
import { diffusionTrackingModule } from './diffusion-tracking';

export const MODULE_IMPLS: Record<string, ModuleImpl> = {
  'diffusion-tracking': diffusionTrackingModule,
};

/**
 * Resolve a module implementation, throwing if the slug is not registered —
 * that can only mean a route exists without its implementation, which is a
 * build-time mistake worth surfacing loudly rather than a runtime condition.
 * Returning a non-optional type also keeps the page free of null checks in
 * every closure.
 */
export function implFor(slug: string): ModuleImpl {
  const impl = MODULE_IMPLS[slug];
  if (!impl) throw new Error(`no Imaging module implementation registered for "${slug}"`);
  return impl;
}
