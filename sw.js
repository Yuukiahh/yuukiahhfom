/* Service worker: prefer network for shell so updates land; never cache data JSON. */
var SHELL = 'desk-shell-v5';
var FILES = ['./', 'index.html', 'app.css', 'app.js', 'chart.js', 'manifest.json', 'icons/icon-192.png', 'icons/icon-512.png'];
self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(SHELL).then(function (c) { return c.addAll(FILES); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (ks) {
    return Promise.all(ks.filter(function (k) { return k !== SHELL; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener('fetch', function (e) {
  var u = new URL(e.request.url);
  if (u.pathname.indexOf('/d/') !== -1 || u.pathname.indexOf('/api/') !== -1) return; // data: network only
  e.respondWith(fetch(e.request).then(function (r) {
    var copy = r.clone();
    caches.open(SHELL).then(function (c) { c.put(e.request, copy); });
    return r;
  }).catch(function () { return caches.match(e.request); }));
});
self.addEventListener('push', function (e) {
  var d = {}; try { d = e.data.json(); } catch (x) { d = { title: 'FOMO desk', body: e.data && e.data.text() }; }
  e.waitUntil(self.registration.showNotification(d.title || 'FOMO desk', { body: d.body || '', icon: 'icons/icon-192.png', data: { url: d.url } }));
});
self.addEventListener('notificationclick', function (e) {
  e.notification.close(); var u = e.notification.data && e.notification.data.url;
  if (u && new URL(u).hostname === 'fomo.family') e.waitUntil(clients.openWindow(u));
});
