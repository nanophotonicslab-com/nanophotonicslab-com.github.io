// Service Worker — NanophotonicsLab PWA
const CACHE = 'nanolab-v3';
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);
const IS_LOCAL = LOCAL_HOSTS.has(self.location.hostname);

// Pre-cache the app shell on install
const SHELL = [
  '/',
  '/lab/',
  '/manifest.json',
  '/favicon.svg',
  '/favicon.ico',
  '/icon-192.png',
];

self.addEventListener('install', (e) => {
  if (IS_LOCAL) {
    self.skipWaiting();
    return;
  }

  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  if (IS_LOCAL) {
    e.waitUntil(
      caches.keys()
        .then((keys) => Promise.all(keys.map((k) => caches.delete(k))))
        .then(() => self.clients.claim())
    );
    return;
  }

  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (IS_LOCAL) return;

  const url = new URL(e.request.url);

  // Skip non-GET and cross-origin
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return;

  // Skip API calls and analytics
  if (url.pathname.startsWith('/api') || url.hostname.includes('umami')) return;

  // Static assets (JS, CSS, images, fonts): cache-first
  if (/\.(js|css|png|svg|ico|woff2?|whl|json)$/.test(url.pathname) || url.pathname.includes('/_astro/')) {
    e.respondWith(
      caches.match(e.request).then((cached) =>
        cached || fetch(e.request).then((res) => {
          if (res.ok) {
            const clone = res.clone();
            caches.open(CACHE).then((c) => c.put(e.request, clone));
          }
          return res;
        })
      )
    );
    return;
  }

  // HTML pages: network-first with cache fallback
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
