// Ambient declarations for dependencies that ship no types of their own.

// KaTeX publishes types for its main entry but not for the auto-render
// contrib bundle, which is the one the assistant page loads.
declare module 'katex/dist/contrib/auto-render.mjs' {
  interface RenderDelimiter { left: string; right: string; display: boolean }
  interface RenderOptions {
    delimiters?: RenderDelimiter[];
    ignoredTags?: string[];
    ignoredClasses?: string[];
    errorCallback?: (msg: string, err: Error) => void;
    throwOnError?: boolean;
    macros?: Record<string, string>;
  }
  export default function renderMathInElement(el: HTMLElement, options?: RenderOptions): void;
}
