// @ts-check
import { defineConfig } from 'astro/config';

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://nanophotonicslab.com',

  redirects: {
    '/research/': '/lab/',
    '/publications/': '/about/',
    '/team/': '/about/',
  },

  // Hidden routes: built, and reachable by anyone who has the URL, but not
  // advertised -- out of the sitemap here, and unlinked from the lab index,
  // the quick-tools bar and the README.
  integrations: [
    sitemap({
      filter: (page) =>
        !['/lab/workbench', '/lab/electrostatics-fd'].some(hidden => page.includes(hidden)),
    }),
  ],
});