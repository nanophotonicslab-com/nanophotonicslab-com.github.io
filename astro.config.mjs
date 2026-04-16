// @ts-check
import { defineConfig } from 'astro/config';

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://nanophotonicslab.com',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'es'],
    routing: { prefixDefaultLocale: false },
  },

  redirects: {
    '/research/': '/',
    '/publications/': '/',
    '/team/': '/',
  },

  integrations: [sitemap()],
});