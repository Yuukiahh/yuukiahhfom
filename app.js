/* FOMO Desk PWA. Plan B: polls JSON under the secret path (#d=). Plan A: bearer token (#t=).
   Every FOMO link shown here comes from the server JSON (built by links.py) — never built in the browser. */
(function () {
  'use strict';
  var LS = window.localStorage;
  (function () {
    var h = new URLSearchParams(location.hash.replace(/^#/, ''));
    var q = new URLSearchParams(location.search);
    ['d', 't', 'b'].forEach(function (k) {
      var v = h.get(k) || q.get(k);
      if (v) LS.setItem('desk_' + k, v);
    });
    // Strip secrets from the visible URL after storing them (hash + query).
    if (location.hash || q.get('d') || q.get('t') || q.get('b')) {
      history.replaceState(null, '', location.pathname);
    }
  })();
  var D = LS.getItem('desk_d'), T = LS.getItem('desk_t'), B = LS.getItem('desk_b') || './d/';
  var POLL_MS = 15000, state = null, chartData = null, range = 24, scrub = false;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function safeUrl(u) { try { var x = new URL(u); return x.protocol === 'https:' && x.hostname === 'fomo.family' ? u : null; } catch (e) { return null; } }
  function usd(v, d) { if (v == null || !isFinite(v)) return '—'; d = d == null ? 2 : d;
    return (v < 0 ? '-' : '') + '$' + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); }
  function px(v) { if (v == null || !isFinite(v)) return '—'; return v >= 1 ? usd(v) : '$' + Number(v).toPrecision(4); }
  function pct(v) { return v == null || !isFinite(v) ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%'; }
  function ago(ts) { if (!ts) return 'never'; var s = Math.round(Date.now() / 1000 - ts); return s < 90 ? s + 's ago' : s < 5400 ? Math.round(s / 60) + 'm ago' : Math.round(s / 3600) + 'h ago'; }
  function url(f) { return B.replace(/\/?$/, '/') + encodeURIComponent(D) + '/' + f + '?_=' + Date.now(); }
  function get(f) {
    return fetch(url(f), { cache: 'no-store', headers: T ? { Authorization: 'Bearer ' + T } : {}, referrerPolicy: 'no-referrer' })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
  }
  function sym(o) { return o.symbol ? '$' + o.symbol : (o.mint || '').slice(0, 6); }

  document.querySelectorAll('.tabs button').forEach(function (b) {
    b.onclick = function () {
      document.querySelectorAll('.tabs button').forEach(function (x) { x.classList.toggle('on', x === b); });
      document.querySelectorAll('.tab').forEach(function (t) { t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab); });
      if (b.dataset.tab === 'activity') loadActivity();
      if (b.dataset.tab === 'home') main.draw();
    };
  });
  document.querySelectorAll('#ranges button').forEach(function (b) {
    b.onclick = function () { range = +b.dataset.h; document.querySelectorAll('#ranges button').forEach(function (x) { x.classList.toggle('on', x === b); }); renderChart(); };
  });

  var main = new LineChart($('chart'), { onScrub: function (p) {
    scrub = !!p; if (p) { $('equity').textContent = usd(p[1]); $('delta').textContent = new Date(p[0] * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
    else renderHero();
  } });
  var coinChart = new LineChart($('coin-chart'));

  function renderHero() {
    if (!state || scrub) return;
    $('equity').textContent = usd(state.equity);
    var d = state.equity - state.start_stack, dp = state.start_stack ? 100 * d / state.start_stack : 0;
    $('delta').textContent = (d >= 0 ? '+' : '') + usd(d) + ' (' + pct(dp) + ') all time';
    document.body.classList.toggle('down', d < 0);
  }
  function renderChart() {
    if (!chartData) return;
    var since = Date.now() / 1000 - range * 3600;
    main.set(chartData.balance.filter(function (p) { return p[0] >= since; }));
  }
  function item(title, sub, pillText, pillCls, href, onclick) {
    var a = el(href ? 'a' : 'div', 'item'); if (href) { a.href = href; a.target = '_blank'; a.rel = 'noopener'; }
    if (onclick) { a.onclick = function (e) { e.preventDefault(); onclick(); }; a.style.cursor = 'pointer'; }
    var l = el('div', 'l'); l.appendChild(el('div', 't', title)); if (sub) l.appendChild(el('div', 's', sub));
    a.appendChild(l); if (pillText != null) a.appendChild(el('div', 'pill ' + (pillCls || 'mu'), pillText)); return a;
  }
  function fill(id, nodes, empty) { var c = $(id); c.textContent = ''; if (!nodes.length && empty) c.appendChild(el('div', 'note', empty)); nodes.forEach(function (n) { c.appendChild(n); }); }

  function render() {
    var s = state; renderHero();
    $('badge').textContent = s.mode; $('badge').className = 'badge ' + (s.mode === 'PAPER' ? 'paper' : 'live');
    $('cash').textContent = usd(s.stack); $('realized').textContent = usd(s.realized); $('unreal').textContent = usd(s.unrealized);
    fill('positions', s.positions.map(function (p) {
      return item(sym(p), usd(p.size_usd) + ' · stop ' + pct(p.stop_pct) + (p.trail ? ' (trail)' : '') + ' · ' + ago(p.opened_at),
        pct(p.pnl_pct), p.pnl_pct >= 0 ? 'up' : 'dn', null, function () { openCoin(p.mint); });
    }), 'No open positions.');
    var watch = s.waiting.map(function (w) { return item(sym(w), 'Waiting room', px(w.price), 'mu', null, function () { openCoin(w.mint); }); })
      .concat(s.recent.filter(function (r) { return !s.waiting.some(function (w) { return w.mint === r.mint; }); }).slice(0, 15).map(function (r) {
        return item(sym(r), (r.verdict || 'new') + (r.reason ? ' · ' + r.reason : ''), px(r.price), r.verdict === 'pass' ? 'up' : r.verdict === 'kill' ? 'dn' : 'mu', null, function () { openCoin(r.mint); });
      }));
    fill('watch', watch, 'Waiting for migrations…');
    fill('agents', s.agents.map(function (a) {
      var stale = !a.ts || Date.now() / 1000 - a.ts > 300;
      var n = item(a.name[0].toUpperCase() + a.name.slice(1), (a.enabled ? 'On' : 'Paused') + ' · ' + ago(a.ts));
      n.firstChild.firstChild.prepend(el('span', 'dot ' + (!a.enabled ? 'none' : stale ? 'stale' : 'ok'))); return n;
    }));
    var L = s.links || {};
    fill('account', [item('FOMO', '@' + (L.handle || '—'), 'Profile', 'mu', safeUrl(L.profile)),
                     item('Invite', 'Share your FOMO link', 'Invite', 'mu', safeUrl(L.signup)),
                     item('Mode', s.mode === 'PAPER' ? 'Paper — no orders, ever' : 'Live tracking — you trade on FOMO, the desk never places orders', s.mode, 'mu'),
                     item('Stack', usd(s.stack), null)]);
    var keys = ['executor.stack_pct', 'executor.min_ticket_usd', 'executor.below_min', 'risk.stop_pct', 'risk.trail_start_pct', 'risk.trail_pct', 'risk.timeout_h',
                'scanner.min_liquidity_usd', 'scanner.max_top10_pct', 'analyst.min_score', 'desk.report_time'];
    fill('settings', keys.filter(function (k) { return k in s.settings; }).map(function (k) { return item(k, null, String(s.settings[k]), 'mu'); }));
    $('hostnote').textContent = s.hosting === 'A' ? 'Plan A: live API.' : 'Plan B: read-only snapshot every 15s. Change settings in Telegram (e.g. "set stop to -8%").';
    var age = Date.now() / 1000 - s.ts, lv = $('live'); lv.className = 'live ' + (age < 180 ? 'ok' : 'stale'); lv.lastChild.textContent = 'Updated ' + ago(s.ts);
  }

  var KIND = { bought: 'up', sold: 'mu', stopped: 'dn', rejected: 'mu' };
  function loadActivity() {
    get('activity.json').then(function (a) {
      fill('activity', a.items.map(function (i) {
        var when = new Date(i.ts * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        return item(i.title, (i.subtitle || '') + ' · ' + i.agent + ' · ' + when, i.kind, KIND[i.kind], null, i.mint ? function () { openCoin(i.mint); } : null);
      }), 'No activity yet.');
    }).catch(function () {});
  }

  function openCoin(mint) {
    $('coin').hidden = false; $('coin-title').textContent = mint.slice(0, 6) + '…'; $('coin-open').removeAttribute('href');
    get('coin/' + encodeURIComponent(mint) + '.json').then(function (c) {
      $('coin-title').textContent = sym(c) + (c.name ? ' · ' + c.name : '');
      var pr = c.prices || [], last = pr.length ? pr[pr.length - 1][1] : null;
      $('coin-price').textContent = px(last);
      $('coin-delta').textContent = pr.length > 1 ? pct(100 * (last / pr[0][1] - 1)) + ' since first seen' : '';
      var u = safeUrl(c.link); if (u) $('coin-open').href = u;
      coinChart.set(pr);
      fill('coin-gate', (c.audits || []).map(function (a) { return item('L' + a.layer + ' ' + a.name.replace(/_/g, ' '), a.value_text + ' · ' + a.source, a.pass ? 'PASS' : 'FAIL', a.pass ? 'up' : 'dn'); }), 'Not audited yet.');
      var v = c.verdict;
      fill('coin-signal', v ? [item((v.pass ? 'PASS' : 'REJECT') + ' · score ' + v.score,
        'alive ' + v.alive_6h.toFixed(2) + ' · organic ' + v.organic.toFixed(2) + ' · flush ' + v.flush.toFixed(2) + ' · chase ' + v.chase.toFixed(2) + ' · ' + v.reason)] : [], 'No verdict.');
    }).catch(function () { $('coin-title').textContent = 'Unavailable'; });
  }
  $('coin-back').onclick = function () { $('coin').hidden = true; };
  $('forget').onclick = function () { ['desk_d', 'desk_t', 'desk_b'].forEach(function (k) { LS.removeItem(k); }); location.reload(); };

  function poll() {
    if (!D) { $('locked').hidden = false; document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); }); return; }
    get('state.json').then(function (s) { state = s; render(); }).catch(function () { var lv = $('live'); lv.className = 'live stale'; lv.lastChild.textContent = 'Offline'; });
    get('chart.json').then(function (c) { chartData = c; renderChart(); }).catch(function () {});
  }
  poll(); setInterval(poll, POLL_MS);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) poll(); });
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(function () {});
})();
