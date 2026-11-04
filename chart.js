/* Minimal Robinhood-style canvas line chart with touch scrubbing. */
(function () {
  function LineChart(canvas, opts) {
    this.c = canvas; this.opts = opts || {}; this.pts = []; this.hover = null;
    var self = this;
    function pos(e) {
      var r = self.c.getBoundingClientRect();
      var x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      self.hover = Math.max(0, Math.min(1, x / r.width)); self.draw();
      if (self.opts.onScrub) self.opts.onScrub(self.at());
    }
    function end() { self.hover = null; self.draw(); if (self.opts.onScrub) self.opts.onScrub(null); }
    ['mousemove', 'touchstart', 'touchmove'].forEach(function (ev) { canvas.addEventListener(ev, pos, { passive: true }); });
    ['mouseleave', 'touchend', 'touchcancel'].forEach(function (ev) { canvas.addEventListener(ev, end); });
    window.addEventListener('resize', function () { self.draw(); });
  }
  LineChart.prototype.set = function (pts) { this.pts = (pts || []).filter(function (p) { return p && isFinite(p[1]); }); this.draw(); };
  LineChart.prototype.at = function () {
    if (this.hover == null || !this.pts.length) return null;
    return this.pts[Math.round(this.hover * (this.pts.length - 1))];
  };
  LineChart.prototype.draw = function () {
    var c = this.c, dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    c.width = w * dpr; c.height = h * dpr;
    var g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    var p = this.pts;
    var col = getComputedStyle(document.body).getPropertyValue('--acc').trim() || '#00c805';
    if (p.length < 2) {
      g.strokeStyle = '#3a2a3e'; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
      return;
    }
    var min = Infinity, max = -Infinity;
    p.forEach(function (q) { if (q[1] < min) min = q[1]; if (q[1] > max) max = q[1]; });
    if (max === min) { max += 1e-9; min -= 1e-9; }
    var pad = 12, X = function (i) { return (i / (p.length - 1)) * w; },
        Y = function (v) { return pad + (1 - (v - min) / (max - min)) * (h - 2 * pad); };
    // baseline (first value), dotted
    g.strokeStyle = '#4a3550'; g.lineWidth = 1; g.setLineDash([1, 4]);
    g.beginPath(); g.moveTo(0, Y(p[0][1])); g.lineTo(w, Y(p[0][1])); g.stroke(); g.setLineDash([]);
    // soft area fill under the line (visual only)
    if (/^#[0-9a-f]{6}$/i.test(col)) {
      var gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, col + '33'); gr.addColorStop(1, col + '00');
      g.fillStyle = gr; g.beginPath(); g.moveTo(0, h);
      p.forEach(function (q, i) { g.lineTo(X(i), Y(q[1])); }); g.lineTo(w, h); g.closePath(); g.fill();
    }
    g.strokeStyle = col; g.lineWidth = 2.4; g.lineJoin = 'round'; g.beginPath();
    p.forEach(function (q, i) { i ? g.lineTo(X(i), Y(q[1])) : g.moveTo(X(i), Y(q[1])); });
    g.stroke();
    if (this.hover != null) {
      var i = Math.round(this.hover * (p.length - 1)), x = X(i), y = Y(p[i][1]);
      g.strokeStyle = '#6a5070'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
      g.fillStyle = col; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill();
    } else {
      var lx = X(p.length - 1), ly = Y(p[p.length - 1][1]);
      g.fillStyle = col; g.beginPath(); g.arc(lx - 3, ly, 4, 0, 7); g.fill();
    }
  };
  window.LineChart = LineChart;
})();
