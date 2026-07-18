export interface ToolVersion {
  version: string;
  updated: string; // ISO date
}

export const TOOL_VERSIONS: Record<string, ToolVersion> = {
  'mie-scattering':          { version: '1.0.0', updated: '2026-06-10' },
  'plasmonic-nanoparticles':  { version: '1.0.0', updated: '2026-06-10' },
  'cylinder':                { version: '1.0.0', updated: '2026-06-10' },
  'bem-solver':              { version: '0.1.0', updated: '2026-06-10' },
  'photothermal':            { version: '1.0.0', updated: '2026-06-10' },
  'bpm':                     { version: '1.0.0', updated: '2026-06-10' },
  'rcwa':                    { version: '0.1.0', updated: '2026-06-10' },
  'photon':                  { version: '1.0.0', updated: '2026-06-10' },
  'electron':                { version: '1.0.0', updated: '2026-06-10' },
  'laser':                   { version: '1.0.0', updated: '2026-06-10' },
  'units':                   { version: '1.0.0', updated: '2026-06-10' },
  'materials':               { version: '0.2.0', updated: '2026-06-24' },
  'assistant':               { version: '0.1.0', updated: '2026-06-10' },
  'heterostructures':        { version: '1.0.0', updated: '2026-06-20' },
  'tweezers':                { version: '0.1.0', updated: '2026-07-18' },
  'purcell':                 { version: '0.1.0', updated: '2026-07-18' },
};

export function formatCitation(toolName: string, toolSlug: string): string {
  const v = TOOL_VERSIONS[toolSlug];
  if (!v) return '';
  const today = new Date().toISOString().slice(0, 10);
  return (
    `NanophotonicsLab, "${toolName}", v${v.version}, ` +
    `https://nanophotonicslab.com/lab/${toolSlug}/, ` +
    `accessed ${today}.`
  );
}
