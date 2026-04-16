export const languages = { en: 'English', es: 'Español' } as const;
export type Locale = keyof typeof languages;
export const defaultLocale: Locale = 'en';

const translations = {
  // Navbar
  'nav.home': { en: 'Home', es: 'Inicio' },
  'nav.lab': { en: 'Lab', es: 'Lab' },

  // Home
  'home.title': { en: 'Home', es: 'Inicio' },
  'home.hero.pre': { en: 'Light at the', es: 'Luz a' },
  'home.hero.gradient': { en: 'nanoscale', es: 'nanoescala' },
  'home.hero.subtitle': {
    en: 'We investigate light-matter interactions in nanostructures — from plasmonics and photonic crystals to quantum optics. Building the tools and knowledge to control photons at their smallest scale.',
    es: 'Investigamos las interacciones luz-materia en nanoestructuras — desde plasmónica y cristales fotónicos hasta óptica cuántica. Construimos las herramientas y el conocimiento para controlar fotones a su escala más pequeña.',
  },
  'home.hero.cta': { en: 'Try our simulators', es: 'Prueba nuestros simuladores' },

  // Lab index
  'lab.title': { en: 'Lab', es: 'Laboratorio' },
  'lab.hero.title': { en: 'Simulation', es: 'Entorno de' },
  'lab.hero.gradient': { en: 'playground', es: 'simulación' },
  'lab.hero.subtitle': {
    en: 'Interactive tools built by our lab. Explore nanophotonic phenomena right in your browser — no installation required.',
    es: 'Herramientas interactivas construidas por nuestro laboratorio. Explora fenómenos nanofotónicos directamente en tu navegador — sin instalación.',
  },
  'lab.cta': { en: 'Launch simulator', es: 'Abrir simulador' },

  // 404
  '404.title': { en: 'Page not found', es: 'Página no encontrada' },
  '404.subtitle': {
    en: "The page you're looking for doesn't exist or has been moved.",
    es: 'La página que buscas no existe o ha sido movida.',
  },
  '404.home': { en: 'Go home', es: 'Ir al inicio' },
  '404.lab': { en: 'Try the lab', es: 'Probar el lab' },

  // Footer
  'footer.about': {
    en: 'Exploring light-matter interactions at the nanoscale.',
    es: 'Explorando las interacciones luz-materia a nanoescala.',
  },
  'footer.nav': { en: 'Navigate', es: 'Navegar' },
  'footer.contact': { en: 'Contact', es: 'Contacto' },
  'footer.rights': { en: 'All rights reserved.', es: 'Todos los derechos reservados.' },
} as const;

export type TranslationKey = keyof typeof translations;

export function t(locale: Locale, key: TranslationKey): string {
  return translations[key]?.[locale] ?? translations[key]?.en ?? key;
}

/** Get locale from URL path. Returns 'es' if path starts with /es/, otherwise 'en'. */
export function getLocaleFromPath(path: string): Locale {
  return path.startsWith('/es/') || path === '/es' ? 'es' : 'en';
}

/** Get the equivalent path in another locale. */
export function getLocalePath(path: string, targetLocale: Locale): string {
  const cleanPath = path.replace(/^\/es(\/|$)/, '/');
  if (targetLocale === 'en') return cleanPath || '/';
  return '/es' + (cleanPath === '/' ? '/' : cleanPath);
}
