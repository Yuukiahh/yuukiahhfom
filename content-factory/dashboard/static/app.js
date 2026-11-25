/* ContentFactory Dashboard – vanilla JS single-page app (no build step, no CDN). */
"use strict";

/* ================================================================== core utils */
const $ = (sel, root = document) => root.querySelector(sel);

/** Tiny element builder: h("div.card#id", {onclick, class, style, ...}, children...) – always uses textContent. */
function h(tag, attrs, ...children) {
  const m = tag.match(/^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i);
  const el = document.createElement((m && m[1]) || "div");
  if (m && m[2]) for (const part of m[2].match(/[.#][\w-]+/g)) {
    if (part[0] === ".") el.classList.add(part.slice(1)); else el.id = part.slice(1);
  }
  if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs); attrs = null;
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className += " " + v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "html") el.innerHTML = v; // only used with trusted constants (icons)
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}
function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
function svg(markup, cls) {
  const s = document.createElement("span");
  s.innerHTML = markup.trim();
  const el = s.firstElementChild;
  if (cls) el.classList.add(cls);
  return el;
}

const ICONS = {
  home: '<svg viewBox="0 0 24 24"><path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/></svg>',
  trend: '<svg viewBox="0 0 24 24"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>',
  wand: '<svg viewBox="0 0 24 24"><path d="M15 4V2M15 10V8M11 6h2M17 6h2M4 20L14 10"/><path d="M19 13v2M18 14h2M7 3v2M6 4h2"/></svg>',
  queue: '<svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg>',
  film: '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M7 3v18M17 3v18M3 8h4M3 16h4M17 8h4M17 16h4"/></svg>',
  captions: '<svg viewBox="0 0 24 24"><rect x="2" y="5" width="20" height="14" rx="3"/><path d="M7 12h4M13 12h4M7 15h7"/></svg>',
  send: '<svg viewBox="0 0 24 24"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4z"/></svg>',
  chart: '<svg viewBox="0 0 24 24"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>',
  coin: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M15 9.5a3 3 0 0 0-3-1.5c-1.7 0-3 .9-3 2s1.3 2 3 2 3 .9 3 2-1.3 2-3 2a3 3 0 0 1-3-1.5M12 6v2M12 16v2"/></svg>',
  gear: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
  log: '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 4.5l13 7.5-13 7.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-2.6-6.4L21 8"/><path d="M21 3v5h-5"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  warn: '<svg viewBox="0 0 24 24"><path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.01"/></svg>',
  shield: '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/></svg>',
  link: '<svg viewBox="0 0 24 24"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/></svg>',
  save: '<svg viewBox="0 0 24 24"><path d="M5 3h11l5 5v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M7 3v6h8M7 21v-7h10v7"/></svg>',
  calendar: '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>',
  inbox: '<svg viewBox="0 0 24 24"><path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.5 5h13L22 12v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6z"/></svg>',
  undo: '<svg viewBox="0 0 24 24"><path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/></svg>',
  key: '<svg viewBox="0 0 24 24"><circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M17 6l3 3M15 8l2 2"/></svg>',
  cpu: '<svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
  target: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg>',
};
const icon = (name, cls) => svg(ICONS[name] || ICONS.info, cls);

/* ---------------------------------------------------------------- API */
class ApiError extends Error {
  constructor(status, data) {
    super((data && (data.error || data.detail)) || `HTTP ${status}`);
    this.status = status; this.data = data || {};
  }
}
async function api(path, { method = "GET", body } = {}) {
  const opts = { method, headers: { "X-CF-Dashboard": "1" } };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(path, opts); } catch (e) { throw new ApiError(0, { error: "Server nicht erreichbar" }); }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}
const apiPost = (p, body) => api(p, { method: "POST", body });
const apiPut = (p, body) => api(p, { method: "PUT", body });
/** Run fn and return [result, error] instead of throwing. */
async function safe(promise) { try { return [await promise, null]; } catch (e) { return [null, e]; } }

/* ---------------------------------------------------------------- formatting */
const nf = new Intl.NumberFormat("de-DE");
const fmt = {
  num: (n) => (n == null ? "–" : nf.format(n)),
  compact: (n) => (n == null ? "–" : new Intl.NumberFormat("de-DE", { notation: "compact", maximumFractionDigits: 1 }).format(n)),
  eur: (n, d = 2) => (n == null ? "–" : new Intl.NumberFormat("de-DE", { style: "currency", currency: "EUR", minimumFractionDigits: d, maximumFractionDigits: Math.max(d, 2) }).format(n)),
  pct: (r, d = 0) => (r == null ? "–" : `${(r * 100).toFixed(d).replace(".", ",")} %`),
  dur: (s) => { if (s == null) return "–"; s = Math.round(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; },
  sec: (s) => (s == null ? "" : Number(s).toFixed(2)),
  date: (ts) => (ts ? new Date(ts * 1000).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "–"),
  short: (ts) => (ts ? new Date(ts * 1000).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "–"),
  rel: (ts) => {
    if (!ts) return "–";
    const d = Date.now() / 1000 - ts, a = Math.abs(d);
    const u = a < 60 ? [Math.round(a), "Sek."] : a < 3600 ? [Math.round(a / 60), "Min."] : a < 86400 ? [Math.round(a / 3600), "Std."] : [Math.round(a / 86400), "Tg."];
    return d >= 0 ? `vor ${u[0]} ${u[1]}` : `in ${u[0]} ${u[1]}`;
  },
};

const STATUS = {
  queued: ["Wartend", "blue"], running: ["Läuft", "violet"], done: ["Fertig", "green"], failed: ["Fehler", "red"],
  cancelled: ["Abgebrochen", ""], review: ["Prüfung", "amber"],
  pending: ["Offen", "amber"], approved: ["Freigegeben", "green"], rejected: ["Abgelehnt", "red"],
  scheduled: ["Geplant", "blue"], uploaded: ["Hochgeladen", "green"],
};
const STAGES = {
  research: "Recherche", script: "Skript", factcheck: "Faktencheck", voice: "Sprachaufnahme", visuals: "Visuals",
  music: "Musik & SFX", render_scenes: "Szenen-Rendering", assemble: "Zusammenschnitt", qc: "Qualitätskontrolle",
  done: "Fertig", queued: "Wartet",
};
const statusBadge = (s) => { const [t, c] = STATUS[s] || [s || "–", ""]; return h("span.badge", { class: c }, t); };
const WORKER = {
  running: { t: "Läuft", dot: "green", icon: "play" },
  paused: { t: "Pausiert", dot: "amber", icon: "pause" },
  stopped: { t: "Gestoppt", dot: "red", icon: "stop" },
};

/* ---------------------------------------------------------------- UI helpers */
function toast(msg, kind = "info", ms = 4200) {
  const el = h("div.toast", { class: kind }, msg);
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.transition = "opacity .3s"; el.style.opacity = "0"; setTimeout(() => el.remove(), 300); }, ms);
}
function errToast(e, prefix = "") {
  let msg = e.message || String(e);
  if (e.status === 503) msg = `Modul noch nicht verfügbar – ${msg}`;
  if (e.status === 501) msg = `Noch nicht implementiert – ${msg}`;
  if (e.data && e.data.problems) msg += ": " + (Array.isArray(e.data.problems) ? e.data.problems.join("; ") : Object.entries(e.data.problems).map(([k, v]) => `${k}: ${v}`).join("; "));
  toast(prefix + msg, "err", 6500);
}
function unavailable(err, what) {
  const msg = err ? (err.message || err) : "nicht verfügbar";
  return h("div.unavail", icon("warn"), h("div", h("b", what ? `${what}: ` : ""), msg));
}
function empty(title, text, ic = "inbox") {
  return h("div.empty", icon(ic), h("b", title), text ? h("div.small", text) : null);
}
function card(title, ...children) {
  return h("div.card", title ? (typeof title === "string" ? h("h2", title) : title) : null, ...children);
}
function cardHead(title, ...right) { return h("h2", title, h("span.spacer"), ...right); }
async function withBusy(btn, fn) {
  const old = Array.from(btn.childNodes);
  btn.disabled = true;
  btn.replaceChildren(h("span.spin"), ...old.filter((n) => n.nodeType === 3));
  try { return await fn(); } finally { btn.disabled = false; btn.replaceChildren(...old); }
}
function button(label, { kind = "", ic, size = "", onclick, title, disabled } = {}) {
  return h("button.btn", { class: `${kind} ${size}`.trim(), onclick, title, disabled, type: "button" }, ic ? icon(ic) : null, label);
}
function progressBar(p, { striped = false, lg = false } = {}) {
  const pct = Math.max(0, Math.min(100, (p || 0) * 100));
  return h("div.progress", { class: `${striped ? "striped" : ""} ${lg ? "lg" : ""}` }, h("i", { style: { width: pct + "%" } }));
}
function seg(options, value, onchange) {
  const wrap = h("div.seg");
  for (const [v, label] of options) {
    const b = h("button", { type: "button", class: v === value ? "on" : "" }, label);
    b.addEventListener("click", () => { wrap.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); onchange(v); });
    wrap.appendChild(b);
  }
  return wrap;
}
function scoreRing(score) {
  const s = score == null ? 0 : score > 1 ? Math.min(1, score / 100) : score;
  const r = 20, c = 2 * Math.PI * r;
  const id = "g" + Math.random().toString(36).slice(2, 8);
  const el = h("div.score-ring");
  el.appendChild(svg(`<svg viewBox="0 0 48 48"><defs><linearGradient id="${id}"><stop offset="0" stop-color="#8B5CF6"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>
    <circle cx="24" cy="24" r="${r}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="5"/>
    <circle cx="24" cy="24" r="${r}" fill="none" stroke="url(#${id})" stroke-width="5" stroke-linecap="round" stroke-dasharray="${(c * s).toFixed(1)} ${c.toFixed(1)}"/></svg>`));
  el.appendChild(h("span", score == null ? "–" : score > 1 ? Math.round(score) : Math.round(score * 100)));
  return el;
}

/** Horizontal/vertical SVG bar chart. items: [{label, value, title}] */
function barChart(items, { height = 220, format = fmt.compact, color } = {}) {
  if (!items.length) return empty("Keine Daten", null, "chart");
  const W = 640, H = height, padL = 44, padB = 54, padT = 14;
  const max = Math.max(...items.map((i) => i.value || 0)) || 1;
  const bw = Math.min((W - padL - 10) / items.length, 90);
  const x0 = padL + ((W - padL - 10) - bw * items.length) / 2;
  const id = "cg" + Math.random().toString(36).slice(2, 8);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"><defs><linearGradient id="${id}" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="${color || "#8B5CF6"}"/><stop offset="1" stop-color="${color || "#22D3EE"}"/></linearGradient></defs><g class="grid">`;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (H - padT - padB) * (1 - i / 4);
    s += `<line x1="${padL}" x2="${W}" y1="${y}" y2="${y}"/><text x="${padL - 8}" y="${y + 4}" text-anchor="end">${esc(format(max * i / 4))}</text>`;
  }
  s += "</g>";
  items.forEach((it, i) => {
    const bh = (H - padT - padB) * ((it.value || 0) / max);
    const x = x0 + i * bw + bw * 0.18, w = bw * 0.64, y = H - padB - bh;
    s += `<rect class="bar" x="${x}" y="${y}" width="${w}" height="${Math.max(bh, 1)}" rx="5" fill="url(#${id})"><title>${esc(it.title || it.label)}: ${esc(format(it.value))}</title></rect>`;
    const lbl = String(it.label).length > 14 ? String(it.label).slice(0, 13) + "…" : String(it.label);
    s += `<text class="lbl" x="${x + w / 2}" y="${H - padB + 16}" text-anchor="end" transform="rotate(-28 ${x + w / 2} ${H - padB + 16})">${esc(lbl)}</text>`;
  });
  return svg(s + "</svg>");
}
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function hbars(items, { format = (v) => v.toFixed(2), max } = {}) {
  if (!items.length) return empty("Keine Daten", null, "chart");
  const m = max || Math.max(...items.map((i) => i.value || 0), 1e-9);
  return h("div.bars", items.map((it) => [
    h("div.bar-row", h("div.name", { title: it.label }, it.label), h("div.track", h("i", { style: { width: `${Math.max(2, (it.value || 0) / m * 100)}%` } })), h("div.v", format(it.value || 0)),
      it.meta ? h("div.meta", { title: it.meta }, it.meta) : null),
  ]));
}

/* ================================================================== router */
const SECTIONS = [
  { id: "uebersicht", title: "Übersicht", sub: "Produktion, Warteschlange und Worker auf einen Blick", icon: "home", render: renderOverview },
  { id: "trends", title: "Trends", sub: "Aktuelle Themen und Nischen-Ranking aus echten Signalen", icon: "trend", render: renderTrends },
  { id: "generator", title: "Generator", sub: "Ein neues Video manuell in Auftrag geben", icon: "wand", render: renderGenerator },
  { id: "warteschlange", title: "Warteschlange", sub: "Live-Status aller Produktionsjobs", icon: "queue", render: renderQueue, badge: "queue" },
  { id: "videos", title: "Videos", sub: "Fertige Videos prüfen, freigeben und planen", icon: "film", render: renderVideos, badge: "review" },
  { id: "untertitel", title: "Untertitel-Editor", sub: "Wörter und Timings korrigieren, dann neu rendern", icon: "captions", render: renderSubtitles },
  { id: "veroeffentlichung", title: "Veröffentlichung", sub: "Plattformen, Freigaben und Zeitplan", icon: "send", render: renderPublishing },
  { id: "performance", title: "Performance", sub: "Echte Plattform-Metriken und Learnings", icon: "chart", render: renderPerformance },
  { id: "kosten", title: "Kosten", sub: "Ausgaben der letzten 30 Tage", icon: "coin", render: renderCosts },
  { id: "einstellungen", title: "Einstellungen", sub: "Produktion, Design, Zeitplan und API-Schlüssel", icon: "gear", render: renderSettings },
  { id: "protokoll", title: "Protokoll", sub: "Die letzten 200 Ereignisse", icon: "log", render: renderEvents },
];
const NAV_GROUPS = { uebersicht: "Studio", videos: "Inhalte", performance: "Auswertung", einstellungen: "System" };

const page = { timers: [], token: 0 };
/** Register an interval that is cleared automatically on navigation. */
function every(ms, fn) { page.timers.push(setInterval(fn, ms)); }
function setActions(...nodes) { $("#topActions").replaceChildren(...nodes); }

function buildNav() {
  const nav = $("#nav");
  for (const s of SECTIONS) {
    if (NAV_GROUPS[s.id]) nav.appendChild(h("div.nav-group", NAV_GROUPS[s.id]));
    nav.appendChild(h("a", { href: `#${s.id}`, "data-id": s.id }, icon(s.icon), h("span", s.title), s.badge ? h("span.badge.violet", { "data-badge": s.badge, hidden: true }) : null));
  }
}

async function route() {
  page.timers.forEach(clearInterval); page.timers = [];
  const token = ++page.token;
  const [id, ...rest] = (location.hash.slice(1) || "uebersicht").split("/");
  const sec = SECTIONS.find((s) => s.id === id) || SECTIONS[0];
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.id === sec.id));
  $("#pageTitle").textContent = sec.title;
  $("#pageSub").textContent = sec.sub;
  document.title = `${sec.title} · ContentFactory`;
  setActions();
  $("#sidebar").classList.remove("open");
  const view = $("#view");
  view.replaceChildren(h("div.grid.g-4", [1, 2, 3, 4].map(() => h("div.card", h("div.skeleton", { style: { width: "50%" } }), h("div.skeleton", { style: { marginTop: "14px", height: "30px" } })))));
  try {
    await sec.render(view, { isCurrent: () => token === page.token, params: rest });
  } catch (e) {
    console.error(e);
    if (token === page.token) view.replaceChildren(card("Fehler", unavailable(e, "Ansicht konnte nicht geladen werden")));
  }
}

/* global poll: worker pill + nav badges */
let lastStatus = null;
async function pollGlobal() {
  const [st] = await safe(api("/api/status"));
  if (st) lastStatus = st;
  const pill = $("#workerPill");
  if (!st) { pill.replaceChildren(h("span.dot.red"), h("span", "Server nicht erreichbar")); return; }
  const ws = st.worker_state && st.worker_state.state;
  const w = WORKER[ws];
  pill.replaceChildren(h("span.dot", { class: w ? w.dot : "" }), h("span", w ? `Worker: ${w.t}` : "Worker nicht verfügbar"),
    h("span.muted.small", { style: { marginLeft: "auto" } }, `${st.produced_today}/${st.videos_per_day}`));
  const q = st.queue || {};
  setBadge("queue", (q.queued || 0) + (q.running || 0));
  setBadge("review", (st.approvals || {}).pending || 0);
}
function setBadge(key, n) {
  const b = document.querySelector(`[data-badge="${key}"]`);
  if (!b) return;
  b.hidden = !n; b.textContent = n;
}

/* ================================================================== Übersicht */
async function renderOverview(view, ctx) {
  const btnPlan = button("Tagesplan erstellen", { ic: "calendar", onclick: async (e) => {
    await withBusy(e.currentTarget, async () => {
      const [r, err] = await safe(apiPost("/api/queue/plan"));
      if (err) return errToast(err);
      toast(`${(r.job_ids || []).length} Jobs für heute eingeplant`, "ok"); load();
    });
  } });
  setActions(btnPlan, button("Neues Video", { kind: "primary", ic: "plus", onclick: () => (location.hash = "#generator") }));

  async function load() {
    const [[st, e1], [ev], [vids]] = await Promise.all([safe(api("/api/status")), safe(api("/api/events?limit=10")), safe(api("/api/videos?limit=6"))]);
    if (!ctx.isCurrent()) return;
    if (e1) { view.replaceChildren(card("Status", unavailable(e1))); return; }
    const q = st.queue || {};
    const target = st.videos_per_day || 0;
    const costs = st.costs || {};
    const kpis = h("div.grid.g-4",
      h("div.card.kpi.accent", h("div.label", icon("target"), "Heute produziert"),
        h("div.value", st.produced_today, h("small", ` / ${target}`)), h("div", { style: { marginTop: "12px" } }, progressBar(target ? st.produced_today / target : 0, { lg: true })),
        h("div.foot", target ? (st.produced_today >= target ? "Tagesziel erreicht" : `noch ${target - st.produced_today} bis zum Tagesziel`) : "Kein Tagesziel gesetzt")),
      h("div.card.kpi", h("div.label", icon("queue"), "Warteschlange"),
        h("div.value", (q.queued || 0) + (q.running || 0), h("small", " aktiv")),
        h("div.foot", `${q.running || 0} läuft · ${q.queued || 0} wartend · ${q.failed || 0} Fehler`)),
      h("div.card.kpi", h("div.label", icon("shield"), "QC-Quote"),
        h("div.value", st.qc_pass_rate == null ? "–" : fmt.pct(st.qc_pass_rate)),
        h("div.foot", `${st.qc_passed} von ${st.videos_total} Videos bestanden`)),
      h("div.card.kpi", h("div.label", icon("coin"), "Kosten (30 Tage)"),
        h("div.value", fmt.eur(costs.last_30_days_eur)),
        h("div.foot", costs.budget_eur ? `Budget ${fmt.eur(costs.budget_eur)} / Monat` : "Budget 0 € – nur kostenlose Dienste")),
    );
    view.replaceChildren(kpis, h("div.mt"), workerCard(st, load),
      h("div.grid.g-main.mt",
        card(cardHead("Neueste Videos", h("a.small", { href: "#videos" }, "Alle anzeigen →")), recentVideos(vids)),
        card(cardHead("Aktivität", h("a.small", { href: "#protokoll" }, "Protokoll →")), eventList(ev ? ev.events : [], true))));
  }
  await load();
  every(5000, load);
}

function workerCard(st, reload) {
  const wsObj = st.worker_state;
  const ws = wsObj && wsObj.state;
  const w = WORKER[ws] || { t: "Nicht verfügbar", icon: "info" };
  const set = (state) => async (e) => {
    await withBusy(e.currentTarget, async () => {
      const [, err] = await safe(apiPost("/api/state", { state }));
      if (err) return errToast(err);
      toast(`Worker: ${WORKER[state].t}`, "ok"); reload(); pollGlobal();
    });
  };
  const cur = wsObj && wsObj.current_job;
  const worker = st.worker || {};
  const next = worker.next_run || worker.next_run_at;
  return h("div.card",
    h("div.control",
      h("div.state-big", h("div.state-orb", { class: ws || "" }, icon(w.icon)),
        h("div", h("div.muted.small", "Worker-Status"), h("div.t", w.t),
          h("div.small.muted", cur ? `Job #${cur} in Arbeit` : ws === "running" ? "Wartet auf den nächsten Job" : st.errors && st.errors.worker ? st.errors.worker : "Keine Produktion aktiv"))),
      next ? h("div.small.muted", "Nächster Lauf: ", h("b", { style: { color: "var(--text)" } }, typeof next === "number" ? fmt.date(next) : JSON.stringify(next))) : null,
      h("div.small.muted", `Aktive Stunden: ${(st.active_hours || []).join("–")} Uhr`),
      h("div.buttons",
        h("button.btn.lg.success", { class: ws === "running" ? "is-on" : "", onclick: set("running"), disabled: !wsObj }, icon("play"), "Start"),
        h("button.btn.lg.warn", { class: ws === "paused" ? "is-on" : "", onclick: set("paused"), disabled: !wsObj }, icon("pause"), "Pause"),
        h("button.btn.lg.danger", { class: ws === "stopped" ? "is-on" : "", onclick: set("stopped"), disabled: !wsObj }, icon("stop"), "Stopp"))),
    !wsObj ? h("div.mt", unavailable(st.errors && st.errors.worker, "Worker")) : null);
}

function recentVideos(data) {
  const vids = data ? data.videos : [];
  if (!vids.length) return empty("Noch keine Videos", "Starte den Worker oder erstelle ein Video im Generator.", "film");
  return h("div.vgrid", { style: { gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))" } }, vids.map(videoCard));
}

function eventList(events, compact = false) {
  if (!events.length) return empty("Keine Ereignisse", null, "log");
  return h("div.events", { style: compact ? { maxHeight: "360px" } : null }, events.map((e) =>
    h("div.ev", h("span.time", compact ? fmt.short(e.created_at) : fmt.date(e.created_at)), h("span.lvl", { class: e.level || "info" }, e.level || "info"),
      h("span.msg", e.job_id ? h("span.muted", `#${e.job_id} `) : null, e.message))));
}

/* ================================================================== Trends */
async function renderTrends(view, ctx) {
  const refreshBtn = button("Trends aktualisieren", { kind: "primary", ic: "refresh" });
  refreshBtn.addEventListener("click", async () => {
    const [r, err] = await safe(apiPost("/api/trends/refresh"));
    if (err) return errToast(err);
    toast(r.running ? "Trend-Aktualisierung läuft im Hintergrund …" : "Aktualisierung gestartet", "info");
    load();
  });
  setActions(refreshBtn);
  let wasRunning = false;

  async function load() {
    const [d, err] = await safe(api("/api/trends?limit=40"));
    if (!ctx.isCurrent()) return;
    if (err) { view.replaceChildren(card("Trends", unavailable(err))); return; }
    const rf = d.refresh || {};
    refreshBtn.disabled = !!rf.running;
    refreshBtn.replaceChildren(rf.running ? h("span.spin") : icon("refresh"), rf.running ? "Aktualisiere …" : "Trends aktualisieren");
    if (wasRunning && !rf.running) toast(rf.error ? `Trend-Refresh fehlgeschlagen: ${rf.error}` : "Trends aktualisiert", rf.error ? "err" : "ok");
    wasRunning = !!rf.running;

    const topics = d.topics || [];
    const topicCard = card(cardHead(`Top-Themen`, h("span.badge", `${topics.length}`)),
      d.errors.top_topics ? h("div", { style: { marginBottom: "12px" } }, unavailable(d.errors.top_topics + " – zeige gespeicherte Kandidaten", "Trend-Engine")) : null,
      topics.length ? h("div", topics.map(topicRow)) : empty("Keine Themen", "Klicke „Trends aktualisieren“, um Signale zu sammeln.", "trend"));

    const niches = (d.niches || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
    const nicheCard = card("Nischen-Ranking",
      d.errors.niche_report ? unavailable(d.errors.niche_report, "Nischen-Report") :
        hbars(niches.map((n) => ({ label: n.niche, value: n.score || 0, meta: [n.signals != null ? `${n.signals} Signale` : null, ...(n.sample_titles || []).slice(0, 2)].filter(Boolean).join(" · ") }))));
    const sources = d.sources;
    const srcCard = card("Quellen",
      h("div.small.muted", { style: { marginBottom: "10px" } }, `${fmt.num(d.signals.count)} Signale gespeichert · letzter Abruf ${fmt.rel(d.signals.last_fetch)}`),
      d.errors.source_status ? unavailable(d.errors.source_status, "Quellenstatus") : sources ? sourceList(sources) : null,
      rf.finished_at ? h("div.small.muted.mt", `Letzte Aktualisierung über Dashboard: ${fmt.rel(rf.finished_at)}`, rf.error ? h("div", { style: { color: "var(--red)" } }, rf.error) : null) : null);
    view.replaceChildren(h("div.grid.g-main", topicCard, h("div.stack", nicheCard, srcCard)));
  }

  function topicRow(t) {
    const evidence = Array.isArray(t.evidence) ? t.evidence : t.evidence ? [t.evidence] : [];
    const links = evidence.slice(0, 4).map((e) => {
      if (typeof e === "string" && /^https?:/.test(e)) return h("a", { href: e, target: "_blank", rel: "noopener noreferrer" }, icon("link"), hostOf(e));
      if (e && typeof e === "object" && e.url) return h("a", { href: e.url, target: "_blank", rel: "noopener noreferrer", title: e.title || e.url }, icon("link"), e.source || e.title || hostOf(e.url));
      if (e && typeof e === "object") return h("span", e.source || e.title || JSON.stringify(e).slice(0, 40));
      return h("span", `Signal #${e}`);
    });
    const detail = t.score_detail && typeof t.score_detail === "object" ? Object.entries(t.score_detail).filter(([, v]) => typeof v === "number").slice(0, 4) : [];
    const btn = button("In Warteschlange", { size: "sm", ic: "plus", kind: t.status === "queued" ? "" : "primary", disabled: t.status === "queued" });
    if (t.status === "queued") btn.lastChild.textContent = "Eingereiht";
    btn.addEventListener("click", async () => {
      const ok = await withBusy(btn, async () => {
        let [r, err] = await safe(apiPost(`/api/topics/${t.id}/enqueue`, {}));
        if (err && err.status === 409 && confirm(`${err.message}\n\nTrotzdem einreihen?`)) [r, err] = await safe(apiPost(`/api/topics/${t.id}/enqueue`, { force: true }));
        if (err) { errToast(err); return false; }
        toast(`Job #${r.job_id} eingereiht: ${t.title}`, "ok");
        return true;
      });
      if (ok) { btn.disabled = true; btn.classList.remove("primary"); btn.lastChild.textContent = "Eingereiht"; }
    });
    return h("div.topic", scoreRing(t.score),
      h("div", { style: { minWidth: 0 } }, h("div.title", t.title),
        h("div.tev", t.niche ? h("span.badge.violet", t.niche) : null,
          detail.map(([k, v]) => h("span.badge", { title: k }, `${k}: ${v.toFixed(2)}`)), links,
          evidence.length > 4 ? h("span", `+${evidence.length - 4}`) : null)),
      btn);
  }

  function sourceList(src) {
    const inner = src && typeof src.sources === "object" ? src.sources : src;
    const entries = Object.entries(inner || {}).filter(([, s]) => s !== null && typeof s !== "number" || typeof s === "boolean");
    if (!entries.length) return empty("Noch kein Abruf", "Quellenstatus erscheint nach dem ersten Trend-Refresh.", "trend");
    return h("div",
      src.fetched_at ? h("div.small.muted", { style: { marginBottom: "6px" } }, `Letzter Lauf ${fmt.rel(src.fetched_at)}${src.signals != null ? ` · ${src.signals} Signale` : ""}`) : null,
      entries.map(([name, s]) => {
        let cls = "", note = "", tip = "";
        if (typeof s === "object") {
          const okN = typeof s.ok === "number" ? s.ok : s.ok ? 1 : 0, failN = s.failed || 0;
          cls = failN && !okN ? "red" : failN ? "amber" : okN || s.ok === true ? "green" : "";
          note = s.note || [s.signals != null ? `${s.signals} Signale` : null, failN ? `${failN} Fehler` : null].filter(Boolean).join(" · ");
          tip = (s.errors || []).join("\n") || s.error || "";
        } else { cls = s ? "green" : "red"; }
        return h("div.row", { style: { padding: "7px 0", borderBottom: "1px solid var(--border)", flexWrap: "nowrap" }, title: tip || null },
          h("span.dot", { class: cls }), h("b", { style: { fontWeight: 500 } }, name.replace(/_/g, " ")), h("span.small.muted", { style: { marginLeft: "auto" } }, note));
      }));
  }

  await load();
  every(4000, () => { if (wasRunning) load(); });
}
function hostOf(u) { try { return new URL(u).hostname.replace(/^www\./, ""); } catch (_) { return u; } }

/* ================================================================== Generator */
let catalogCache = null;
async function getCatalog() {
  if (!catalogCache) catalogCache = await api("/api/catalog");
  return catalogCache;
}

async function renderGenerator(view, ctx) {
  const [cat, err] = await safe(getCatalog());
  if (!ctx.isCurrent()) return;
  if (err) { view.replaceChildren(card("Generator", unavailable(err))); return; }
  const defs = cat.defaults || {};
  const state = { format: "short", language: defs.language || "en" };

  const topic = h("textarea", { placeholder: "z. B. Was passiert, wenn man in ein Schwarzes Loch fällt?", rows: 3, maxlength: 300 });
  const niche = h("input", { list: "nicheList", placeholder: "z. B. space, psychology, technology" });
  const nicheList = h("datalist#nicheList", (cat.niches || []).map((n) => h("option", { value: n })));
  const voice = h("select");
  const preset = h("select", (cat.presets || []).map((p) => h("option", { value: p, selected: p === defs.preset }, presetLabel(p))));
  const auto = h("input", { type: "checkbox", checked: true });
  const scriptIn = h("textarea.mono", { rows: 10, placeholder: '{\n  "title": "…",\n  "hook": "…",\n  "scenes": [ { "narration": "…", "visual": "nebula" } ]\n}' });
  const scriptBox = h("label.field.full", { hidden: true }, "Eigenes Skript (JSON)", scriptIn,
    h("span.hint", "Wird unverändert an die Produktion übergeben (Recherche & Faktencheck entfallen). Format wie script.json eines fertigen Videos."));
  auto.addEventListener("change", () => (scriptBox.hidden = auto.checked));

  function fillVoices() {
    const engines = cat.engines || {};
    const preferred = typeof engines === "object" ? engines[state.language] : engines;
    const defVoice = typeof defs.voice === "object" && defs.voice ? defs.voice[state.language] : defs.voice;
    voice.replaceChildren(h("option", { value: "" }, `Standard (${defVoice || "Konfiguration"})`));
    const groups = Object.entries(cat.voices || {}).sort(([a], [b]) => (a === preferred ? -1 : b === preferred ? 1 : 0));
    for (const [engine, list] of groups) {
      const filtered = list.filter((v) => engine !== "edge" || v.toLowerCase().startsWith(state.language) || v.includes("Multilingual"));
      if (!filtered.length) continue;
      voice.appendChild(h("optgroup", { label: `${engine}${engine === preferred ? " · empfohlen" : ""}` }, filtered.map((v) => h("option", { value: v }, v))));
    }
  }
  fillVoices();

  const submit = h("button.btn.primary.lg", { type: "submit" }, icon("wand"), "Video in Auftrag geben");
  const form = h("form.card", { onsubmit: async (e) => {
    e.preventDefault();
    if (topic.value.trim().length < 2) { toast("Bitte ein Thema eingeben", "err"); topic.focus(); return; }
    let script = null;
    if (!auto.checked) {
      try { script = JSON.parse(scriptIn.value); } catch (e2) { toast(`Skript ist kein gültiges JSON: ${e2.message}`, "err"); scriptIn.focus(); return; }
    }
    await withBusy(submit, async () => {
      const payload = {
        topic: topic.value.trim(), niche: niche.value.trim() || null, format: state.format, language: state.language,
        preset: preset.value || null, voice: voice.value || null, auto_script: auto.checked, script,
      };
      let [r, err2] = await safe(apiPost("/api/queue", payload));
      if (err2 && err2.status === 409 && confirm(`${err2.message}\n\nTrotzdem einreihen?`)) [r, err2] = await safe(apiPost("/api/queue", { ...payload, force: true }));
      if (err2) return errToast(err2, "Konnte nicht eingereiht werden: ");
      toast(`Job #${r.job_id} eingereiht`, "ok");
      topic.value = "";
      resultBox.replaceChildren(h("div.notice.safe", icon("check"), h("div", h("b", `Job #${r.job_id} ist in der Warteschlange`), h("p", "Der Fortschritt ist live in der ", h("a", { href: "#warteschlange" }, "Warteschlange"), " sichtbar."))));
    });
  } },
    h("h2", "Neues Video"),
    h("div.form-grid",
      h("label.field.full", "Thema", topic, h("span.hint", "Eine konkrete Frage oder Aussage funktioniert am besten.")),
      h("label.field", "Nische", niche, nicheList),
      h("label.field", "Design-Preset", preset),
      h("div.field", h("label.field", "Format"), seg([["short", "Short · 25–45 s"], ["long", "Long · 60–90 s"]], state.format, (v) => (state.format = v))),
      h("div.field", h("label.field", "Sprache"), seg([["en", "Englisch"], ["de", "Deutsch"]], state.language, (v) => { state.language = v; fillVoices(); })),
      h("label.field.full", "Stimme", voice),
      h("label.check.full", auto, h("div", h("b", "Skript automatisch"), h("span", "Recherche, Skript und Faktencheck werden vollständig automatisch erstellt. Deaktivieren, um ein eigenes Skript (JSON) vorzugeben."))),
      scriptBox,
    ),
    h("div.row.end.mt", submit));
  const resultBox = h("div");

  const tips = card("So entsteht dein Video",
    h("div.stack", { style: { gap: "12px" } },
      ["Recherche & Quellen", "Skript, Hook & Faktencheck", "Stimme mit Wort-Timings", "Visuals, Musik & SFX", "Rendering 1080×1920", "Automatische QC & Metadaten"].map((t, i) =>
        h("div.row", h("span.badge.violet", { style: { width: "26px", justifyContent: "center" } }, i + 1), h("span", t)))),
    h("div.hr"),
    h("div.small.muted", "Fertige Videos landen unter „Videos“ und werden erst nach deiner Freigabe veröffentlicht."),
    Object.keys(cat.errors || {}).length ? h("div.mt", Object.values(cat.errors).map((e) => unavailable(e))) : null);

  view.replaceChildren(h("div.grid.g-main", h("div.stack", form, resultBox), h("div", tips)));
}
function presetLabel(p) { return ({ cinematic: "Cinematic – ruhig & hochwertig", energetic: "Energetic – schnell & laut", documentary: "Documentary – sachlich" })[p] || p; }

/* ================================================================== Warteschlange */
async function renderQueue(view, ctx) {
  let filter = ctx.params[0] || "";
  setActions(button("Tagesplan erstellen", { ic: "calendar", onclick: (e) => withBusy(e.currentTarget, async () => {
    const [r, err] = await safe(apiPost("/api/queue/plan"));
    if (err) return errToast(err);
    toast(`${(r.job_ids || []).length} Jobs eingeplant`, "ok"); load();
  }) }), button("Neues Video", { kind: "primary", ic: "plus", onclick: () => (location.hash = "#generator") }));

  const filtersEl = h("div.filters");
  const listCard = h("div.card");
  view.replaceChildren(h("div.row", { style: { marginBottom: "14px" } }, filtersEl, h("span.small.muted", { style: { marginLeft: "auto" } }, "Aktualisiert alle 3 s")), listCard);

  async function load() {
    const [d, err] = await safe(api("/api/queue?limit=300"));
    if (!ctx.isCurrent()) return;
    if (err) { listCard.replaceChildren(unavailable(err)); return; }
    const counts = {};
    d.jobs.forEach((j) => (counts[j.status] = (counts[j.status] || 0) + 1));
    filtersEl.replaceChildren(...[["", "Alle"], ["running", "Läuft"], ["queued", "Wartend"], ["review", "Prüfung"], ["failed", "Fehler"], ["done", "Fertig"], ["cancelled", "Abgebrochen"]].map(([k, label]) =>
      h("button.chip", { class: filter === k ? "on" : "", onclick: () => { filter = k; load(); } }, label, h("span.n", k ? counts[k] || 0 : d.jobs.length))));
    const jobs = filter ? d.jobs.filter((j) => j.status === filter) : d.jobs;
    listCard.replaceChildren(jobs.length ? h("div", jobs.map(jobRow)) : empty("Keine Jobs", filter ? "Kein Job mit diesem Status." : "Die Warteschlange ist leer.", "queue"));
  }

  function jobRow(j) {
    const running = j.status === "running";
    const actions = h("div.row", { style: { justifyContent: "flex-end" } });
    if (["failed", "cancelled"].includes(j.status)) actions.appendChild(button("Wiederholen", { size: "sm", ic: "refresh", onclick: (e) => act(e, j, "retry") }));
    if (["queued", "running", "review"].includes(j.status)) actions.appendChild(button("Abbrechen", { size: "sm", kind: "danger", ic: "x", onclick: (e) => act(e, j, "cancel") }));
    if (j.video_id) actions.appendChild(button("Video", { size: "sm", ic: "film", onclick: () => openVideo(j.video_id) }));
    const opts = j.options && typeof j.options === "object" ? j.options : {};
    return h("div.job",
      h("div", { style: { minWidth: 0 } },
        h("div.title", h("span.muted", `#${j.id} `), j.topic),
        h("div.meta", statusBadge(j.status), j.niche ? h("span.badge.violet", j.niche) : null, h("span.badge", j.format === "long" ? "Long" : "Short"),
          h("span.badge", (j.language || "").toUpperCase()), j.preset ? h("span.badge", j.preset) : null, j.voice ? h("span.badge", j.voice) : null,
          opts.auto_script === false ? h("span.badge.amber", "Skript manuell") : null,
          j.attempts ? h("span.small.muted", `Versuch ${j.attempts}`) : null,
          h("span.small.muted", `· ${fmt.rel(j.updated_at || j.created_at)}`),
          j.not_before && j.not_before > Date.now() / 1000 ? h("span.small.muted", `· frühestens ${fmt.short(j.not_before)}`) : null),
        j.error ? h("div.err", { title: j.error }, j.error) : null),
      h("div", h("div.stage", h("span", STAGES[j.stage] || j.stage || (j.status === "queued" ? "Wartet" : "–")), h("span", `${Math.round((j.progress || 0) * 100)} %`)),
        progressBar(j.status === "done" ? 1 : j.progress, { striped: running })),
      actions);
  }
  async function act(e, j, what) {
    if (what === "cancel" && !confirm(`Job #${j.id} abbrechen?`)) return;
    await withBusy(e.currentTarget, async () => {
      const [, err] = await safe(apiPost(`/api/jobs/${j.id}/${what}`));
      if (err) return errToast(err);
      toast(what === "retry" ? `Job #${j.id} wird erneut versucht` : `Job #${j.id} abgebrochen`, "ok"); load();
    });
  }
  await load();
  every(3000, load);
}

/* ================================================================== Videos */
function videoCard(v) {
  const qc = v.qc_passed ? h("span.badge.green", "QC ✓") : h("span.badge.red", "QC ✗");
  return h("div.vcard", { onclick: () => openVideo(v.id) },
    h("div.vthumb", { style: { backgroundImage: `url("${v.urls.thumb}")` } },
      h("div.badges", qc, statusBadge(v.approval)),
      h("div.play", h("div", icon("play"))),
      h("span.dur", fmt.dur(v.duration))),
    h("div.vbody", h("div.t", { title: v.title }, v.title || v.topic),
      h("div.m", h("span", v.niche || "–"), h("span", "·"), h("span", (v.language || "").toUpperCase()), h("span", "·"), h("span", fmt.short(v.created_at)))));
}

async function renderVideos(view, ctx) {
  let filter = ctx.params[0] || "";
  const filtersEl = h("div.filters");
  const grid = h("div");
  view.replaceChildren(h("div.row", { style: { marginBottom: "16px" } }, filtersEl), grid);
  async function load() {
    const [d, err] = await safe(api("/api/videos?limit=300"));
    if (!ctx.isCurrent()) return;
    if (err) { grid.replaceChildren(card(null, unavailable(err))); return; }
    const counts = {};
    d.videos.forEach((v) => (counts[v.approval] = (counts[v.approval] || 0) + 1));
    filtersEl.replaceChildren(...[["", "Alle"], ["pending", "Offen"], ["approved", "Freigegeben"], ["rejected", "Abgelehnt"]].map(([k, l]) =>
      h("button.chip", { class: filter === k ? "on" : "", onclick: () => { filter = k; load(); } }, l, h("span.n", k ? counts[k] || 0 : d.videos.length))));
    const vids = filter ? d.videos.filter((v) => v.approval === filter) : d.videos;
    grid.replaceChildren(vids.length ? h("div.vgrid", vids.map(videoCard)) : card(null, empty("Keine Videos", "Hier erscheinen fertig produzierte Videos.", "film")));
  }
  videosReload = load;
  await load();
  every(15000, load);
}
let videosReload = null;

function openModal(content) {
  $("#modalBody").replaceChildren(content);
  $("#modal").hidden = false;
  document.body.style.overflow = "hidden";
}
function closeModal() {
  const m = $("#modal");
  if (m.hidden) return;
  m.querySelectorAll("video").forEach((v) => v.pause());
  m.hidden = true;
  $("#modalBody").replaceChildren();
  document.body.style.overflow = "";
}

async function openVideo(id, tab = "info") {
  openModal(h("div", h("div.skeleton", { style: { height: "400px" } })));
  const [v, err] = await safe(api(`/api/videos/${id}`));
  if (err) { openModal(unavailable(err, `Video ${id}`)); return; }
  const meta = v.metadata || {};
  const yt = meta.youtube || {};
  const qc = v.qc || {};
  const tabBody = h("div");
  const tabsEl = h("div.tabs");
  const tabs = {
    info: ["Metadaten", () => h("div.stack",
      h("dl.kv",
        h("dt", "Thema"), h("dd", v.topic || "–"), h("dt", "Hook"), h("dd", v.hook || "–"),
        h("dt", "Nische / Format"), h("dd", `${v.niche || "–"} · ${v.format || "–"} · ${(v.language || "").toUpperCase()}`),
        h("dt", "Dauer"), h("dd", `${fmt.dur(v.duration)} (${(v.duration || 0).toFixed(1)} s)`),
        h("dt", "Erstellt"), h("dd", fmt.date(v.created_at)),
        h("dt", "Ordner"), h("dd.mono", v.folder || "–"),
        meta.ai_generated_disclosure != null ? [h("dt", "KI-Kennzeichnung"), h("dd", String(meta.ai_generated_disclosure))] : null),
      yt.title ? h("div", h("div.small.muted", "YouTube-Titel"), h("div", { style: { fontWeight: 600, marginTop: "4px" } }, yt.title)) : null,
      yt.description ? h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "Beschreibung"), h("div.pre", yt.description)) : null,
      (meta.hashtags || []).length ? h("div.row", meta.hashtags.map((t) => h("span.badge.cyan", `#${t}`))) : null,
      meta.tiktok && meta.tiktok.caption ? h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "TikTok-Caption"), h("div.pre", meta.tiktok.caption)) : null)],
    qc: ["QC-Checkliste", () => h("div.stack",
      h("div.row", qc.passed ? h("span.badge.green", "Bestanden") : h("span.badge.red", "Nicht bestanden"), h("span.badge", `Score ${qc.score ?? "–"}`),
        h("span.badge", `${qc.errors ?? 0} Fehler`), h("span.badge", `${qc.warnings ?? 0} Warnungen`), qc.loudness_lufs != null ? h("span.badge", `${qc.loudness_lufs} LUFS`) : null),
      h("div.qc-list", (qc.checks || []).map((c) => h("div.qc-item", { class: c.ok ? "ok" : c.severity === "warn" ? "warn" : "bad" },
        h("span.ic", c.ok ? icon("check") : icon("x")), h("div", { style: { minWidth: 0 } }, h("b", qcLabel(c.check)), h("span", c.detail || ""))))),
      h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "Kontaktabzug"), h("img.contact", { src: v.urls.contact, alt: "Kontaktabzug", loading: "lazy", onerror: (e) => e.target.replaceWith(h("div.small.muted", "Kein Kontaktabzug vorhanden")) })))],
    sources: ["Quellen", () => h("div.stack",
      v.fact_check ? h("div.row", h("span.small.muted", "Faktencheck:"), h("span.badge", { class: ["ok", "pass", "passed", "manual"].includes(v.fact_check.verdict) ? "green" : "amber" }, v.fact_check.verdict || "–"),
        v.fact_check.by ? h("span.small.muted", v.fact_check.by) : null) : null,
      (v.claims || []).length ? h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "Belegte Aussagen"), h("div.stack", { style: { gap: "8px" } }, v.claims.map((c) =>
        h("div.qc-item.ok", h("span.ic", icon("check")), h("div", { style: { minWidth: 0 } }, h("b", c.claim), linkOrText(c.source)))))) : null,
      (v.research_sources || []).length ? h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "Recherche-Quellen"),
        h("ul", { style: { margin: 0, paddingLeft: "18px" } }, v.research_sources.map((s) => h("li", linkOrText(s.url, s.title))))) : null,
      v.licenses ? h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "Lizenzen"), licenseTable(v.licenses)) : null,
      !(v.claims || []).length && !(v.research_sources || []).length && v.sources_md ? h("div.pre", v.sources_md) : null)],
    publish: ["Freigabe & Planung", () => publishPanel(v)],
  };
  function show(k) {
    tabsEl.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === k));
    tabBody.replaceChildren(tabs[k][1]());
  }
  for (const [k, [label]] of Object.entries(tabs)) tabsEl.appendChild(h("button", { "data-k": k, onclick: () => show(k) }, label));
  show(tabs[tab] ? tab : "info");

  openModal(h("div.vmodal",
    h("div", h("video", { src: v.urls.video, controls: true, preload: "metadata", poster: v.urls.thumb, playsinline: true }),
      h("div.row.mt", button("Untertitel bearbeiten", { size: "sm", ic: "captions", onclick: () => { closeModal(); location.hash = `#untertitel/${v.id}`; } }),
        h("a.btn.sm", { href: v.urls.video, download: `${v.slug || "video"}.mp4` }, "Download"))),
    h("div", { style: { minWidth: 0 } },
      h("h2", v.title || v.topic),
      h("div.row", statusBadge(v.approval), v.qc_passed ? h("span.badge.green", "QC bestanden") : h("span.badge.red", "QC nicht bestanden"),
        v.niche ? h("span.badge.violet", v.niche) : null, h("span.badge", `#${v.id}`)),
      tabsEl, tabBody)));
}
function qcLabel(k) {
  return ({ resolution: "Auflösung", fps: "Bildrate", codec: "Codec", audio: "Audio", duration: "Dauer", bitrate: "Bitrate", loudness: "Lautheit", true_peak: "True Peak",
    no_dead_air: "Keine Stille", no_black_frames: "Keine Schwarzbilder", no_frozen_video: "Kein Standbild", subtitle_words: "Untertitel-Wörter", subtitle_coverage: "Untertitel-Abdeckung",
    subtitle_monotonic: "Untertitel-Reihenfolge", hook_length: "Hook-Länge", hook_timing: "Hook-Timing", facts_sourced: "Fakten belegt", fact_check: "Faktencheck", licenses: "Lizenzen",
    visual_brightness: "Helligkeit", visual_variety: "Visuelle Abwechslung" })[k] || k;
}
function linkOrText(url, title) {
  if (typeof url === "string" && /^https?:\/\//.test(url)) return h("a.small", { href: url, target: "_blank", rel: "noopener noreferrer", style: { wordBreak: "break-all" } }, title || url);
  return h("span.small", title || url || "");
}
function licenseTable(lic) {
  const rows = [];
  (lic.media || []).forEach((m) => rows.push([`Szene ${m.scene ?? "–"}`, m.source || m.type || "", m.license || ""]));
  for (const k of ["music", "sfx", "font"]) if (lic[k]) rows.push([k, "", String(lic[k])]);
  if (lic.voice) rows.push(["voice", "", typeof lic.voice === "object" ? Object.values(lic.voice).join(" · ") : String(lic.voice)]);
  return h("div.table-wrap", h("table", h("tbody", rows.map((r) => h("tr", r.map((c) => h("td.small", c)))))));
}

function publishPanel(v) {
  const wrap = h("div.stack");
  const approveBtn = button("Freigeben", { kind: "success", ic: "check" });
  const rejectBtn = button("Ablehnen", { kind: "danger", ic: "x" });
  const decide = (what, btn) => withBusy(btn, async () => {
    const [r, err] = await safe(apiPost(`/api/videos/${v.id}/${what}`));
    if (err) return errToast(err);
    v.approval = r.approval || (what === "approve" ? "approved" : "rejected");
    toast(what === "approve" ? "Video freigegeben" : "Video abgelehnt", "ok");
    wrap.replaceWith(publishPanel(v));
    if (videosReload) videosReload();
    pollGlobal();
  });
  approveBtn.addEventListener("click", () => decide("approve", approveBtn));
  rejectBtn.addEventListener("click", () => decide("reject", rejectBtn));

  const platform = h("select", ["youtube", "tiktok", "instagram"].map((p) => h("option", { value: p }, p)));
  const when = h("input", { type: "datetime-local", value: nextSlotLocal() });
  const schedBtn = button("Einplanen", { kind: "primary", ic: "calendar", disabled: v.approval !== "approved" });
  schedBtn.addEventListener("click", () => withBusy(schedBtn, async () => {
    const ts = new Date(when.value).getTime() / 1000;
    if (!ts) return toast("Bitte Datum/Uhrzeit wählen", "err");
    const [, err] = await safe(apiPost(`/api/videos/${v.id}/schedule`, { platform: platform.value, when_ts: ts }));
    if (err) return errToast(err);
    toast(`Für ${platform.value} am ${fmt.date(ts)} eingeplant`, "ok");
  }));

  append(wrap, [
    lastStatus && lastStatus.publish_enabled
      ? h("div.notice.warn", icon("warn"), h("div", h("b", "Veröffentlichung ist aktiviert"),
        h("p", "Freigegebene und eingeplante Videos werden zum geplanten Zeitpunkt hochgeladen.")))
      : h("div.notice.info", icon("shield"), h("div", h("b", "Nichts wird ohne dich veröffentlicht"),
        h("p", "Die Freigabe markiert das Video nur als geprüft. Öffentliches Posten bleibt deaktiviert, bis du es ausdrücklich per ", h("code", "python -m factory publish enable"), " aktivierst."))),
    h("div.card", { style: { background: "var(--bg-2)" } }, h("h3", "Freigabe"), h("div.row", h("span.small.muted", "Aktuell:"), statusBadge(v.approval), h("span.spacer", { style: { flex: 1 } }), rejectBtn, approveBtn)),
    h("div.card", { style: { background: "var(--bg-2)" } }, h("h3", "Veröffentlichung planen"),
      v.approval !== "approved" ? h("div.small.muted", { style: { marginBottom: "10px" } }, "Erst nach Freigabe möglich.") : null,
      h("div.form-grid", h("label.field", "Plattform", platform), h("label.field", "Zeitpunkt", when)),
      h("div.row.end.mt", schedBtn)),
    (v.publications || []).length ? h("div", h("div.small.muted", { style: { marginBottom: "6px" } }, "Bisherige Planungen"), pubTable(v.publications)) : null,
  ]);
  return wrap;
}
function nextSlotLocal() {
  const d = new Date(Date.now() + 3600e3);
  d.setMinutes(0, 0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function pubTable(pubs) {
  return h("div.table-wrap", h("table",
    h("thead", h("tr", ["Video", "Plattform", "Status", "Geplant", "Veröffentlicht", "Link"].map((t) => h("th", t)))),
    h("tbody", pubs.map((p) => h("tr",
      h("td", p.title ? h("a", { href: "#", onclick: (e) => { e.preventDefault(); openVideo(p.video_id, "publish"); } }, p.title) : `#${p.video_id}`),
      h("td", p.platform), h("td", statusBadge(p.status), p.error ? h("div.small", { style: { color: "var(--red)" } }, p.error) : null),
      h("td.nowrap", fmt.short(p.scheduled_for)), h("td.nowrap", fmt.short(p.published_at)),
      h("td", p.url ? h("a", { href: p.url, target: "_blank", rel: "noopener noreferrer" }, "öffnen") : "–"))))));
}

/* ================================================================== Untertitel-Editor */
async function renderSubtitles(view, ctx) {
  const [d, err] = await safe(api("/api/videos?limit=300"));
  if (!ctx.isCurrent()) return;
  if (err) { view.replaceChildren(card(null, unavailable(err))); return; }
  if (!d.videos.length) { view.replaceChildren(card(null, empty("Keine Videos", "Sobald ein Video fertig ist, kannst du hier die Untertitel korrigieren.", "captions"))); return; }
  const selId = Number(ctx.params[0]) || d.videos[0].id;
  const picker = h("select", { style: { minWidth: "320px" }, onchange: () => (location.hash = `#untertitel/${picker.value}`) },
    d.videos.map((v) => h("option", { value: v.id, selected: v.id === selId }, `#${v.id} · ${v.title || v.topic}`)));
  setActions(picker);

  const [w, err2] = await safe(api(`/api/videos/${selId}/words`));
  if (!ctx.isCurrent()) return;
  if (err2) { view.replaceChildren(card(null, unavailable(err2, "Untertitel"))); return; }
  let words = w.words.map((x) => ({ ...x }));
  let dirty = false;
  const video = h("video", { src: `/media/${selId}/video`, controls: true, preload: "metadata", poster: `/media/${selId}/thumb` });
  const caption = h("div.live-caption.idle", "Abspielen für Live-Vorschau");
  const tbody = h("tbody");
  const status = h("span.small.muted");
  const saveBtn = button("Speichern", { kind: "primary", ic: "save" });
  const rerenderBtn = button("Neu rendern", { ic: "refresh" });
  const restoreBtn = button("Original", { kind: "ghost", ic: "undo", disabled: !w.has_backup });
  const shiftInput = h("input", { type: "number", step: "0.05", value: "0", style: { width: "90px" } });

  function validate() {
    const bad = new Set();
    words.forEach((x, i) => {
      if (!(x.end >= x.start) || x.start < 0 || !String(x.text).trim()) bad.add(i);
      if (i && x.start < words[i - 1].start) bad.add(i);
    });
    return bad;
  }
  function markDirty() { dirty = true; status.textContent = "Ungespeicherte Änderungen"; status.style.color = "var(--amber)"; refreshValidity(); }
  function refreshValidity() {
    const bad = validate();
    tbody.querySelectorAll("tr").forEach((tr, i) => tr.classList.toggle("invalid", bad.has(i)));
    saveBtn.disabled = bad.size > 0;
    if (bad.size) { status.textContent = `${bad.size} ungültige Zeile(n)`; status.style.color = "var(--red)"; }
  }
  function numInput(x, key) {
    return h("input.n", { type: "number", step: "0.01", min: "0", value: fmt.sec(x[key]), oninput: (e) => { x[key] = parseFloat(e.target.value); markDirty(); } });
  }
  function renderRows() {
    tbody.replaceChildren(...words.map((x, i) => h("tr",
      h("td.muted.small", i + 1),
      h("td", h("input.t", { value: x.text, oninput: (e) => { x.text = e.target.value; markDirty(); } })),
      h("td", numInput(x, "start")), h("td", numInput(x, "end")),
      h("td", h("input.s", { type: "number", min: "0", step: "1", value: x.scene ?? 0, oninput: (e) => { x.scene = parseInt(e.target.value || "0", 10); markDirty(); } })),
      h("td.nowrap",
        h("button.icon-btn", { title: "Ab hier abspielen", style: { width: "30px", height: "30px", display: "inline-grid" }, onclick: () => { video.currentTime = Math.max(0, x.start - 0.3); video.play(); } }, icon("play")),
        " ",
        h("button.icon-btn", { title: "Wort danach einfügen", style: { width: "30px", height: "30px", display: "inline-grid" }, onclick: () => {
          const nx = words[i + 1];
          words.splice(i + 1, 0, { text: "…", start: x.end, end: nx ? Math.max(x.end, nx.start) : x.end + 0.3, scene: x.scene });
          renderRows(); markDirty();
        } }, icon("plus")),
        " ",
        h("button.icon-btn", { title: "Wort löschen", style: { width: "30px", height: "30px", display: "inline-grid" }, onclick: () => { words.splice(i, 1); renderRows(); markDirty(); } }, icon("x"))))));
    refreshValidity();
  }
  // live caption + row highlight
  let lastIdx = -1;
  video.addEventListener("timeupdate", () => {
    const t = video.currentTime;
    const idx = words.findIndex((x) => t >= x.start && t <= x.end + 0.05);
    if (idx === lastIdx) return;
    lastIdx = idx;
    tbody.querySelectorAll("tr.current").forEach((r) => r.classList.remove("current"));
    if (idx < 0) return;
    const row = tbody.children[idx];
    if (row) { row.classList.add("current"); row.scrollIntoView({ block: "nearest" }); }
    const g0 = Math.max(0, idx - 1), g = words.slice(g0, g0 + 3);
    caption.classList.remove("idle");
    caption.replaceChildren(...g.map((x, j) => [h("span", { class: g0 + j === idx ? "hl" : "" }, x.text), " "]).flat());
  });

  saveBtn.addEventListener("click", () => withBusy(saveBtn, async () => {
    const [r, e] = await safe(apiPut(`/api/videos/${selId}/words`, { words }));
    if (e) return errToast(e, "Speichern fehlgeschlagen: ");
    dirty = false; status.textContent = `Gespeichert (${r.count} Wörter)`; status.style.color = "var(--green)";
    restoreBtn.disabled = false;
    toast("Untertitel gespeichert", "ok");
  }));
  rerenderBtn.addEventListener("click", async () => {
    if (dirty && !confirm("Es gibt ungespeicherte Änderungen. Trotzdem mit dem gespeicherten Stand rendern?")) return;
    const [r, e] = await safe(apiPost(`/api/videos/${selId}/rerender-subtitles`));
    if (e) return errToast(e, "Neu-Rendern: ");
    toast("Neu-Rendern gestartet …", "info");
    watchRerender(r);
  });
  restoreBtn.addEventListener("click", async () => {
    if (!confirm("Original-Timings wiederherstellen? Deine Änderungen gehen verloren.")) return;
    const [, e] = await safe(apiPost(`/api/videos/${selId}/words/restore`));
    if (e) return errToast(e);
    toast("Original wiederhergestellt", "ok"); route();
  });
  function watchRerender(st) {
    const upd = (s) => {
      rerenderBtn.disabled = !!s.running;
      rerenderBtn.replaceChildren(s.running ? h("span.spin") : icon("refresh"), s.running ? "Rendert …" : "Neu rendern");
    };
    upd(st);
    if (!st.running) return;
    const t = setInterval(async () => {
      const [s] = await safe(api(`/api/videos/${selId}/rerender-subtitles`));
      if (!s || !ctx.isCurrent()) return clearInterval(t);
      upd(s);
      if (!s.running) {
        clearInterval(t);
        if (s.error) toast(`Neu-Rendern fehlgeschlagen: ${s.error}`, "err", 8000);
        else { toast("Video mit neuen Untertiteln gerendert", "ok"); video.src = `/media/${selId}/video?t=${Date.now()}`; }
      }
    }, 2000);
    page.timers.push(t);
  }
  const shiftBtn = button("Verschieben", { title: "Alle Wörter um den Versatz verschieben", onclick: () => {
    const off = parseFloat(shiftInput.value) || 0;
    if (!off) return;
    words.forEach((x) => { x.start = Math.max(0, +(x.start + off).toFixed(3)); x.end = Math.max(0, +(x.end + off).toFixed(3)); });
    renderRows(); markDirty();
  } });

  renderRows();
  status.textContent = `${words.length} Wörter · Videodauer ${fmt.dur(w.duration)}`;
  view.replaceChildren(h("div.sub-layout",
    h("div.sub-preview.card", video, caption, h("div.small.muted.mt", "Tipp: Klicke ▶ in einer Zeile, um ab diesem Wort abzuspielen.")),
    h("div.card",
      h("div.row", { style: { marginBottom: "14px" } }, saveBtn, rerenderBtn, restoreBtn, h("span", { style: { flex: 1 } }), h("div.row", { style: { flexWrap: "nowrap", gap: "8px" } }, h("span.small.muted.nowrap", "Versatz (s)"), shiftInput, shiftBtn)),
      h("div.row", { style: { marginBottom: "10px" } }, status),
      h("div.words-scroll", h("table.words-table", h("thead", h("tr", h("th", "#"), h("th", "Wort"), h("th.num", "Start (s)"), h("th.num", "Ende (s)"), h("th.num", "Szene"), h("th", ""))), tbody)))));
  if (w.rerender && w.rerender.running) watchRerender(w.rerender);
}

/* ================================================================== Veröffentlichung */
async function renderPublishing(view, ctx) {
  const [d, err] = await safe(api("/api/publish"));
  if (!ctx.isCurrent()) return;
  if (err) { view.replaceChildren(card(null, unavailable(err))); return; }
  const notice = d.enabled
    ? h("div.notice.warn", icon("warn"), h("div", h("b", "Veröffentlichung ist AKTIVIERT"),
      h("p", `Freigegebene und eingeplante Videos werden automatisch hochgeladen (Standard-Sichtbarkeit: ${d.default_privacy}). `,
        d.require_manual_approval ? "Jedes Video braucht weiterhin deine manuelle Freigabe." : "Achtung: manuelle Freigabe ist deaktiviert.")))
    : h("div.notice.safe", icon("shield"), h("div", h("b", "Öffentliche Veröffentlichung ist deaktiviert"),
      h("p", "Es wird nichts hochgeladen. Das bleibt so, bis du es ausdrücklich aktivierst: ", h("code", "python -m factory publish enable"),
        ` – und die Plattform-Konten per OAuth verbunden sind. Standard-Sichtbarkeit danach: „${d.default_privacy}“.`)));

  const plats = d.platforms;
  const platformCards = plats
    ? h("div.grid.g-3", Object.entries(plats).map(([name, s]) => h("div.platform",
      h("div.logo", { class: name }, ({ youtube: "YT", tiktok: "TT", instagram: "IG" })[name] || name.slice(0, 2).toUpperCase()),
      h("div.info", h("b", ({ youtube: "YouTube", tiktok: "TikTok", instagram: "Instagram" })[name] || name), h("div.small.muted.clamp2", { title: s.note || "" }, s.note || (s.connected ? "Konto verbunden" : "Nicht verbunden"))),
      h("div.stack", { style: { gap: "6px", alignItems: "flex-end" } },
        s.connected ? h("span.badge.green", "Verbunden") : h("span.badge", "Nicht verbunden"),
        s.enabled ? h("span.badge.amber", "Aktiv") : h("span.badge", "Inaktiv")))))
    : unavailable(d.error, "Plattformstatus");

  const scheduled = d.publications || [];
  const approved = d.approved_unscheduled || [];
  view.replaceChildren(h("div.stack", notice,
    card(cardHead("Plattformen", h("span.small.muted", `Konfiguriert: ${(d.configured_platforms || []).join(", ") || "–"}`)), platformCards),
    h("div.grid.g-main",
      card(cardHead("Zeitplan & Verlauf", h("span.badge", scheduled.length)), scheduled.length ? pubTable(scheduled) : empty("Nichts geplant", "Gib Videos frei und plane sie im Video-Dialog ein.", "calendar")),
      h("div.stack",
        card("Freigegeben, noch nicht geplant", approved.length ? h("div", approved.map((v) => h("div.row", { style: { padding: "8px 0", borderBottom: "1px solid var(--border)" } },
          h("span.ellipsis", { style: { flex: 1, minWidth: 0 } }, v.title), button("Planen", { size: "sm", ic: "calendar", onclick: () => openVideo(v.id, "publish") })))) : empty("Keine offenen Freigaben", null, "check")),
        card("Posting-Slots", h("div.slot-list", (d.slots || []).map((s) => h("span.badge.violet", { style: { paddingRight: "12px" } }, s))),
          h("div.small.muted.mt", "Änderbar unter Einstellungen."))))));
}

/* ================================================================== Performance */
async function renderPerformance(view, ctx) {
  const refreshBtn = button("Metriken abrufen", { ic: "refresh" });
  const analyseBtn = button("Learnings neu berechnen", { kind: "primary", ic: "cpu" });
  setActions(refreshBtn, analyseBtn);
  refreshBtn.addEventListener("click", async () => {
    const [, e] = await safe(apiPost("/api/analytics/refresh"));
    if (e) return errToast(e);
    toast("Metriken werden im Hintergrund abgerufen …", "info");
    setTimeout(() => ctx.isCurrent() && load(), 4000);
  });
  analyseBtn.addEventListener("click", () => withBusy(analyseBtn, async () => {
    const [, e] = await safe(apiPost("/api/learnings/analyse"));
    if (e) return errToast(e);
    toast("Learnings aktualisiert", "ok"); load();
  }));

  async function load() {
    const [[a, e1], [l]] = await Promise.all([safe(api("/api/analytics")), safe(api("/api/learnings"))]);
    if (!ctx.isCurrent()) return;
    if (e1) { view.replaceChildren(card(null, unavailable(e1))); return; }
    const m = a.metrics || [];
    const totals = m.reduce((acc, r) => { acc.views += r.views || 0; acc.likes += r.likes || 0; acc.comments += r.comments || 0; acc.shares += r.shares || 0; return acc; }, { views: 0, likes: 0, comments: 0, shares: 0 });
    const ret = m.filter((r) => r.avg_view_pct != null);
    const avgRet = ret.length ? ret.reduce((s, r) => s + r.avg_view_pct, 0) / ret.length : null;
    const kpis = h("div.grid.g-4",
      h("div.card.kpi", h("div.label", "Aufrufe gesamt"), h("div.value", fmt.compact(totals.views)), h("div.foot", `${m.length} Video/Plattform-Paare`)),
      h("div.card.kpi", h("div.label", "Ø Wiedergabe"), h("div.value", avgRet == null ? "–" : `${avgRet.toFixed(0)} %`), h("div.foot", "Durchschnittlich angesehener Anteil")),
      h("div.card.kpi", h("div.label", "Likes"), h("div.value", fmt.compact(totals.likes)), h("div.foot", `${fmt.compact(totals.comments)} Kommentare`)),
      h("div.card.kpi", h("div.label", "Shares"), h("div.value", fmt.compact(totals.shares)), h("div.foot", totals.views ? `Engagement ${fmt.pct((totals.likes + totals.comments + totals.shares) / totals.views, 1)}` : "–")));

    const top = m.slice(0, 12);
    const charts = h("div.grid.g-2.mt",
      card("Aufrufe je Video", barChart(top.map((r) => ({ label: r.title || `#${r.video_id}`, value: r.views || 0 })))),
      card("Ø Wiedergabe-Anteil (%)", barChart(top.map((r) => ({ label: r.title || `#${r.video_id}`, value: r.avg_view_pct || 0 })), { format: (v) => `${Math.round(v)}%`, color: "#22D3EE" })));

    const table = card(cardHead("Metriken", a.refresh && a.refresh.running ? h("span.badge.violet", h("span.spin"), "Abruf läuft") : null),
      a.error ? h("div", { style: { marginBottom: "12px" } }, unavailable(a.error, "Analytics-Modul")) : null,
      m.length ? h("div.table-wrap", h("table",
        h("thead", h("tr", ["Video", "Plattform", "Aufrufe", "Likes", "Komm.", "Shares", "Ø %", "Ø s", "Abos", "Stand"].map((t, i) => h("th", { class: i > 1 && i < 9 ? "num" : "" }, t)))),
        h("tbody", m.map((r) => h("tr",
          h("td", { style: { minWidth: "220px" } }, h("a", { href: "#", onclick: (e) => { e.preventDefault(); openVideo(r.video_id); } }, r.title || `#${r.video_id}`)), h("td", r.platform),
          h("td.num", fmt.num(r.views)), h("td.num", fmt.num(r.likes)), h("td.num", fmt.num(r.comments)), h("td.num", fmt.num(r.shares)),
          h("td.num", r.avg_view_pct == null ? "–" : r.avg_view_pct.toFixed(1)), h("td.num", r.avg_view_duration == null ? "–" : r.avg_view_duration.toFixed(1)),
          h("td.num", fmt.num(r.subscribers_gained)), h("td.nowrap.small.muted", fmt.short(r.fetched_at)))))))
        : empty("Noch keine Metriken", "Sobald Videos veröffentlicht sind, werden echte Plattform-Daten hier angezeigt.", "chart"),
      a.summary ? h("details.mt", h("summary.small.muted", { style: { cursor: "pointer" } }, "Zusammenfassung (Rohdaten)"), h("div.pre.mt", JSON.stringify(a.summary, null, 2))) : null);

    view.replaceChildren(kpis, m.length ? charts : null, h("div.grid.g-main.mt", table, learningsCard(l && l.learnings)));
  }
  await load();
}

function learningsCard(l) {
  if (!l) return card("Learnings", empty("Noch keine Analyse", "Klicke „Learnings neu berechnen“.", "cpu"));
  if (l.status !== "ok") {
    return card("Learnings", h("div.notice.info", icon("info"), h("div", h("b", "Zu wenig echte Daten"),
      h("p", `Es liegen Metriken für ${l.videos_with_metrics ?? 0} Videos vor; benötigt werden mindestens ${l.needed ?? 5}. Bis dahin ändert das System nichts an seiner Strategie.`))),
      h("div.small.muted.mt", `Stand: ${fmt.date(l.updated_at)}`));
  }
  const nw = Object.entries(l.niche_weights || {}).map(([k, v]) => ({ label: k, value: v })).sort((a, b) => b.value - a.value);
  const hf = Object.entries(l.hook_features || {}).map(([k, v]) => ({ label: k, value: v.lift, meta: `n = ${v.n}` })).sort((a, b) => b.value - a.value);
  return card("Learnings",
    h("div.small.muted", { style: { marginBottom: "12px" } }, `${l.videos_with_metrics} Videos ausgewertet · ${fmt.date(l.updated_at)}`),
    h("h3", "Nischen-Gewichtung"), hbars(nw, { format: (v) => `×${v.toFixed(2)}` }),
    h("h3.mt", "Hook-Merkmale (Lift)"), hbars(hf, { format: (v) => `×${v.toFixed(2)}` }),
    h("div.hr"),
    h("dl.kv", h("dt", "Long-Anteil empf."), h("dd", fmt.pct(l.recommended_long_share)),
      h("dt", "Beste Stunden"), h("dd", (l.best_hours || []).length ? l.best_hours.map((x) => `${x}:00`).join(", ") : "–")),
    (l.top_hooks || []).length ? h("div.mt", h("div.small.muted", { style: { marginBottom: "6px" } }, "Stärkste Hooks"), h("div.stack", { style: { gap: "6px" } }, l.top_hooks.map((t) => h("div.pre", { style: { maxHeight: "none" } }, t)))) : null);
}

/* ================================================================== Kosten */
async function renderCosts(view, ctx) {
  const [c, err] = await safe(api("/api/costs"));
  if (!ctx.isCurrent()) return;
  if (err) { view.replaceChildren(card(null, unavailable(err))); return; }
  const budget = c.budget_eur || 0;
  const spent = c.last_30_days_eur || 0;
  const svc = (c.by_service || []).slice().sort((a, b) => (b.eur || 0) - (a.eur || 0));
  view.replaceChildren(
    h("div.grid.g-3",
      h("div.card.kpi.accent", h("div.label", icon("coin"), "Letzte 30 Tage"), h("div.value", fmt.eur(spent)),
        budget ? h("div", { style: { marginTop: "12px" } }, progressBar(spent / budget, { lg: true })) : null,
        h("div.foot", budget ? `${fmt.pct(spent / budget)} des Monatsbudgets` : "Kein Budget – nur kostenlose Dienste erlaubt")),
      h("div.card.kpi", h("div.label", "Monatsbudget"), h("div.value", fmt.eur(budget)), h("div.foot", "Änderbar unter Einstellungen")),
      h("div.card.kpi", h("div.label", "Kostenpflichtige Dienste"), h("div.value", svc.filter((s) => (s.eur || 0) > 0).length), h("div.foot", `${svc.length} Dienste mit Buchungen`))),
    h("div.grid.g-2.mt",
      card("Tageskosten", barChart((c.by_day || []).map((d) => ({ label: d.day.slice(5), value: d.eur || 0, title: d.day })), { format: (v) => fmt.eur(v) })),
      card("Nach Dienst", svc.length ? h("div.table-wrap", h("table", h("thead", h("tr", h("th", "Dienst"), h("th.num", "Einheiten"), h("th.num", "Kosten"))),
        h("tbody", svc.map((s) => h("tr", h("td", s.service), h("td.num", fmt.num(Math.round(s.units || 0))), h("td.num", fmt.eur(s.eur, 4))))))) : empty("Keine Kosten", "Alles lief über kostenlose Dienste.", "coin"))),
    h("div.grid.g-main.mt",
      card("Letzte Buchungen", (c.recent || []).length ? h("div.table-wrap", h("table", h("thead", h("tr", ["Zeit", "Dienst", "Einheiten", "Job", "Betrag"].map((t, i) => h("th", { class: i === 2 || i === 4 ? "num" : "" }, t)))),
        h("tbody", c.recent.map((r) => h("tr", h("td.nowrap.small", fmt.short(r.created_at)), h("td", r.service), h("td.num", `${fmt.num(r.units)} ${r.unit_name || ""}`), h("td", r.job_id ? `#${r.job_id}` : "–"), h("td.num", fmt.eur(r.amount_eur, 4))))))) : empty("Keine Buchungen", null, "coin")),
      card("Kostenlose Dienste im Einsatz", h("div.row", (c.free_services || []).map((s) => h("span.badge.green", s))))));
}

/* ================================================================== Einstellungen */
async function renderSettings(view, ctx) {
  const [[s, e1], [sec, e2], [cat], [llm, e4]] = await Promise.all([safe(api("/api/settings")), safe(api("/api/secrets")), safe(getCatalog()), safe(api("/api/llm"))]);
  if (!ctx.isCurrent()) return;
  if (e1) { view.replaceChildren(card(null, unavailable(e1))); return; }
  const v = JSON.parse(JSON.stringify(s.values));
  const voices = (cat && cat.voices) || {};
  const allVoices = Object.values(voices).flat();
  const presets = (cat && cat.presets) || ["cinematic", "energetic", "documentary"];

  const numIn = (key, attrs = {}) => h("input", { type: "number", value: v[key], ...attrs, oninput: (e) => (v[key] = e.target.value === "" ? null : Number(e.target.value)) });
  const voiceSel = (lang) => h("select", { onchange: (e) => (v["tts.voice"] = { ...(v["tts.voice"] || {}), [lang]: e.target.value }) },
    Object.entries(voices).map(([eng, list]) => h("optgroup", { label: eng }, list.map((x) => h("option", { value: x, selected: (v["tts.voice"] || {})[lang] === x }, x)))),
    !allVoices.includes((v["tts.voice"] || {})[lang]) ? h("option", { value: (v["tts.voice"] || {})[lang], selected: true }, (v["tts.voice"] || {})[lang]) : null);

  // format mix slider
  const mix = v["schedule.format_mix"] || { short: 0.7, long: 0.3 };
  const mixLabel = h("span.small.muted");
  const mixIn = h("input", { type: "range", min: 0, max: 100, step: 5, value: Math.round((mix.short || 0) * 100), style: { padding: 0, accentColor: "#8B5CF6" } });
  const updMix = () => { const sh = Number(mixIn.value) / 100; v["schedule.format_mix"] = { short: sh, long: +(1 - sh).toFixed(2) }; mixLabel.textContent = `${Math.round(sh * 100)} % Short · ${Math.round((1 - sh) * 100)} % Long`; };
  mixIn.addEventListener("input", updMix); updMix();

  // colors
  const colorsWrap = h("div.colors");
  const renderColors = () => colorsWrap.replaceChildren(
    ...(v["subtitles.highlight_colors"] || []).map((c, i) => h("div.row", { style: { gap: "4px" } },
      h("input", { type: "color", value: c, oninput: (e) => (v["subtitles.highlight_colors"][i] = e.target.value.toUpperCase()) }),
      v["subtitles.highlight_colors"].length > 1 ? h("button.btn.ghost.sm", { type: "button", onclick: () => { v["subtitles.highlight_colors"].splice(i, 1); renderColors(); } }, "×") : null)),
    (v["subtitles.highlight_colors"] || []).length < 6 ? button("Farbe", { size: "sm", ic: "plus", onclick: () => { v["subtitles.highlight_colors"].push("#FFFFFF"); renderColors(); } }) : null);
  renderColors();

  // slots
  const slotsWrap = h("div.slot-list");
  const slotIn = h("input", { type: "time", style: { width: "120px" } });
  const renderSlots = () => slotsWrap.replaceChildren(...(v["publish.slots"] || []).map((t, i) => h("span.badge.violet", t, h("button", { type: "button", title: "Entfernen", onclick: () => { v["publish.slots"].splice(i, 1); renderSlots(); } }, "×"))),
    slotIn, button("Slot", { size: "sm", ic: "plus", onclick: () => { if (slotIn.value && !v["publish.slots"].includes(slotIn.value)) { v["publish.slots"].push(slotIn.value); v["publish.slots"].sort(); renderSlots(); } } }));
  v["publish.slots"] = v["publish.slots"] || [];
  renderSlots();

  const hours = v["schedule.active_hours"] || [9, 23];
  const hFrom = h("input", { type: "number", min: 0, max: 23, value: hours[0], oninput: (e) => (v["schedule.active_hours"] = [Number(e.target.value), v["schedule.active_hours"][1]]) });
  const hTo = h("input", { type: "number", min: 1, max: 24, value: hours[1], oninput: (e) => (v["schedule.active_hours"] = [v["schedule.active_hours"][0], Number(e.target.value)]) });
  v["schedule.active_hours"] = hours;

  const saveBtn = button("Einstellungen speichern", { kind: "primary", ic: "save" });
  saveBtn.addEventListener("click", () => withBusy(saveBtn, async () => {
    const [r, e] = await safe(apiPut("/api/settings", v));
    if (e) return errToast(e, "Nicht gespeichert: ");
    toast(r.changed.length ? `Gespeichert: ${r.changed.length} Änderung(en)` : "Keine Änderungen", "ok");
    catalogCache = null;
  }));
  setActions(saveBtn);

  const production = card("Produktion",
    h("div.form-grid",
      h("label.field", "Videos pro Tag", numIn("schedule.videos_per_day", { min: 0, max: 20 }), h("span.hint", "Empfohlen: 3–5")),
      h("label.field", "Standardsprache", h("select", { onchange: (e) => (v["general.language"] = e.target.value) },
        [["en", "Englisch"], ["de", "Deutsch"]].map(([k, l]) => h("option", { value: k, selected: v["general.language"] === k }, l)))),
      h("label.field.full", "Format-Mix", mixIn, mixLabel),
      h("div.field", h("label.field", "Aktive Stunden"), h("div.row", { style: { flexWrap: "nowrap" } }, hFrom, h("span.muted", "bis"), hTo)),
      h("label.field", "Monatsbudget (EUR)", numIn("costs.monthly_budget_eur", { min: 0, step: 1 }), h("span.hint", "0 = nur kostenlose Dienste"))));

  const design = card("Stimme & Design",
    h("div.form-grid",
      h("label.field", "Stimme Englisch", voiceSel("en")),
      h("label.field", "Stimme Deutsch", voiceSel("de")),
      h("label.field", "Sprechtempo", numIn("tts.speed", { min: 0.7, max: 1.5, step: 0.01 })),
      h("label.field", "Design-Preset", h("select", { onchange: (e) => (v["render.preset"] = e.target.value) }, presets.map((p) => h("option", { value: p, selected: v["render.preset"] === p }, presetLabel(p))))),
      h("label.field", "Untertitel-Größe (px)", numIn("subtitles.size", { min: 40, max: 200 })),
      h("label.field", "Wörter pro Einblendung", numIn("subtitles.words_per_group", { min: 1, max: 6 })),
      h("label.field", "Untertitel-Position (0–1)", numIn("subtitles.position_y", { min: 0.3, max: 0.9, step: 0.01 }), h("span.hint", "Anteil der Bildhöhe, 0,66 = unteres Drittel")),
      h("div.field.full", h("label.field", "Highlight-Farben"), colorsWrap)));

  const ro = s.read_only || {};
  const publish = card("Veröffentlichung",
    h("div.field", h("label.field", "Posting-Slots"), slotsWrap),
    h("div.hr"),
    h("dl.kv", Object.entries(ro).map(([k, o]) => [h("dt", o.label.split(" (")[0]), h("dd", typeof o.value === "boolean" ? (o.value ? h("span.badge.amber", "AN") : h("span.badge.green", "AUS")) : String(o.value))])),
    h("div.small.muted.mt", "Öffentliches Posten lässt sich bewusst nicht im Dashboard einschalten – nur per ", h("code.mono", "python -m factory publish enable"), "."));

  const llmCard = card("KI-Provider",
    e4 ? unavailable(e4) : h("div", Object.entries(llm.available || {}).map(([k, ok]) => h("div.row", { style: { padding: "8px 0", borderBottom: "1px solid var(--border)" } },
      h("span.dot", { class: ok ? "green" : "red" }), h("b", { style: { fontWeight: 500 } }, ({ claude_cli: "Claude Code CLI (Abo)", ollama: "Ollama (lokal)", anthropic: "Claude API (kostenpflichtig)" })[k] || k),
      (llm.providers || []).includes(k) ? h("span.badge.violet", `Priorität ${(llm.providers || []).indexOf(k) + 1}`) : h("span.badge", "nicht aktiv"),
      h("span.small.muted", { style: { marginLeft: "auto" } }, ok ? "verfügbar" : "nicht erreichbar")))));

  const secretsCard = card(cardHead("API-Schlüssel", h("span.small.muted", "Werte werden nie angezeigt")),
    e2 ? unavailable(e2) : h("div", sec.secrets.map(secretRow)));

  view.replaceChildren(h("div.grid.g-2", h("div.stack", production, design), h("div.stack", publish, llmCard)), h("div.mt", secretsCard),
    h("div.small.muted.mt", `Änderungen werden in ${s.local_file} gespeichert.`));
}

function secretRow(s) {
  const input = h("input", { type: "password", placeholder: s.configured ? "•••••••• (gesetzt – neu überschreiben)" : "Wert eingeben", autocomplete: "new-password", spellcheck: false });
  const badge = s.configured ? h("span.badge.green", "Gesetzt") : h("span.badge", "Fehlt");
  const saveBtn = button("Speichern", { size: "sm", kind: "primary" });
  const delBtn = s.configured ? button("Löschen", { size: "sm", kind: "ghost" }) : null;
  const row = h("div.secret",
    h("div", h("div.row", { style: { gap: "8px" } }, h("span.name", s.name), badge), h("div.desc", s.description)),
    s.automatic ? h("span.small.muted", "wird automatisch per OAuth gesetzt") : input,
    h("div.row", s.automatic ? null : saveBtn, delBtn));
  saveBtn.addEventListener("click", () => withBusy(saveBtn, async () => {
    if (!input.value.trim()) return toast("Bitte einen Wert eingeben", "err");
    const [, e] = await safe(apiPost(`/api/secrets/${encodeURIComponent(s.name)}`, { value: input.value }));
    input.value = "";
    if (e) return errToast(e);
    toast(`${s.name} gespeichert`, "ok");
    row.replaceWith(secretRow({ ...s, configured: true }));
  }));
  if (delBtn) delBtn.addEventListener("click", async () => {
    if (!confirm(`${s.name} wirklich löschen?`)) return;
    const [, e] = await safe(api(`/api/secrets/${encodeURIComponent(s.name)}`, { method: "DELETE" }));
    if (e) return errToast(e);
    toast(`${s.name} gelöscht`, "ok");
    row.replaceWith(secretRow({ ...s, configured: false }));
  });
  return row;
}

/* ================================================================== Protokoll */
async function renderEvents(view, ctx) {
  let level = "";
  const listEl = h("div");
  const filters = h("div.filters");
  view.replaceChildren(card(cardHead("Ereignisse", filters), listEl));
  async function load() {
    const [d, err] = await safe(api(`/api/events?limit=200${level ? `&level=${level}` : ""}`));
    if (!ctx.isCurrent()) return;
    filters.replaceChildren(...[["", "Alle"], ["info", "Info"], ["warn", "Warnung"], ["error", "Fehler"]].map(([k, l]) => h("button.chip", { class: level === k ? "on" : "", onclick: () => { level = k; load(); } }, l)));
    listEl.replaceChildren(err ? unavailable(err) : eventList(d.events));
    const ev = listEl.querySelector(".events");
    if (ev) ev.style.maxHeight = "70vh";
  }
  await load();
  every(5000, load);
}

/* ================================================================== boot */
function boot() {
  buildNav();
  $("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#modalClose").addEventListener("click", closeModal);
  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  window.addEventListener("hashchange", () => { closeModal(); route(); });
  route();
  pollGlobal();
  setInterval(pollGlobal, 5000);
}
boot();
