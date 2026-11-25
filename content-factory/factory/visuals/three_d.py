"""Small, fast numpy "true 3D" renderer + dynamic 3D scenes.

Everything is real geometry in world space, projected through a perspective
:class:`Camera` that moves over time (orbits, fly-throughs), so the parallax is
genuine.  Rendering primitives:

* point splatting with sub-pixel bilinear footprint (``np.bincount``) into a
  float accumulation buffer (additive blending), multi-tap footprints for
  medium points, a quarter-resolution "soft" buffer for big / out-of-focus
  points;
* anti-aliased lines = segments resampled into sub-pixel splats (length
  weighted, so brightness per pixel of line length is constant);
* cheap bloom: every splat is also deposited (at low weight) into the soft
  buffer, which is blurred with PIL at 1/4 resolution and bilinearly upsampled;
* a soft-knee tone curve so dense accumulations roll off instead of clipping
  into flat white blobs.

All random data is generated in ``build()``; ``render(t)`` is a pure, smooth
function of ``t`` (safe for 60 fps).  Each render targets <= ~120 ms at the
1080x1920 layer size (1208x2150 with overscan) on one core.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .procedural import Scene

TAU = 2 * math.pi
REF_W = 1208.0  # layer width for a 1080 px wide output; pixel sizes are tuned for it


# ======================================================================== math helpers
def _norm(v):
    v = np.asarray(v, np.float64)
    return v / max(1e-12, float(np.linalg.norm(v)))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def smoothstep(e0, e1, x):
    x = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0, 1)
    return x * x * (3 - 2 * x)


def orbit(target, dist, azim, elev):
    """Camera position on a sphere around ``target`` (y is up)."""
    t = np.asarray(target, np.float64)
    return t + dist * np.array([math.cos(elev) * math.cos(azim), math.sin(elev), math.cos(elev) * math.sin(azim)])


def periodic_noise(shape, rng, beta=2.2, lo=1):
    """Smooth, seamlessly tiling noise (any dimensionality) via a power-law FFT spectrum. Returns 0..1."""
    freqs = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
    k = np.sqrt(sum(f.astype(np.float64) ** 2 for f in freqs))
    amp = np.where(k >= lo, 1.0 / np.maximum(k, 1e-9) ** (beta / 2 * len(shape) / 2 + 0.5), 0.0)
    spec = amp * np.exp(1j * rng.uniform(0, TAU, shape))
    n = np.real(np.fft.ifftn(spec))
    n -= n.min()
    return (n / max(1e-9, n.max())).astype(np.float32)


def trilinear(vol, p):
    """Sample periodic volume ``vol`` (D,D,D) at points p (N,3) given in voxel units."""
    d0, d1, d2 = vol.shape
    f = np.floor(p)
    t = (p - f).astype(np.float32)
    i = f.astype(np.int64)
    out = np.zeros(len(p), np.float32)
    for dx in (0, 1):
        wx = t[:, 0] if dx else 1 - t[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            for dz in (0, 1):
                wz = t[:, 2] if dz else 1 - t[:, 2]
                out += wx * wy * wz * vol[(i[:, 0] + dx) % d0, (i[:, 1] + dy) % d1, (i[:, 2] + dz) % d2]
    return out


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def random_dirs(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lerp_col(a, b, t):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    t = np.asarray(t, np.float32)[..., None]
    return a + (b - a) * t


# ======================================================================== camera
class Camera:
    """Perspective pinhole camera. ``fov`` is the vertical field of view in degrees."""

    def __init__(self, pos, target, up=(0, 1, 0), fov=50.0, roll=0.0):
        self.pos = np.asarray(pos, np.float64)
        fwd = _norm(np.asarray(target, np.float64) - self.pos)
        right = np.cross(fwd, np.asarray(up, np.float64))
        if np.linalg.norm(right) < 1e-8:
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right = _norm(right)
        upv = np.cross(right, fwd)
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            right, upv = right * c + upv * s, upv * c - right * s
        self.R = np.stack([right, upv, fwd])  # rows: camera axes in world coordinates
        self.fov = fov


# ======================================================================== renderer
class Renderer:
    def __init__(self, W: int, H: int, ds: int = 4):
        self.W, self.H, self.ds = W, H, ds
        self.Wp, self.Hp = -(-W // ds) * ds, -(-H // ds) * ds
        self.Ws, self.Hs = self.Wp // ds, self.Hp // ds
        self.Wc, self.Hc = -(-self.Ws // 4), -(-self.Hs // 4)
        self.k = W / REF_W
        self.cam = None
        self.f = 1.0
        self.begin(Camera((0, 0, -1), (0, 0, 0)))

    # ---------------------------------------------------------------- frame setup
    def begin(self, cam: Camera):
        self.cam = cam
        self.f = 0.5 * self.H / math.tan(math.radians(cam.fov) / 2)
        self._fine = {"back": [], "front": []}
        self._soft = []
        self._coarse = []

    def project(self, P, near=0.02):
        """World points (N,3) -> screen x, y (px), depth z (along view), px-per-world-unit at that depth."""
        P = np.asarray(P, np.float32)
        px, py, pz = P[:, 0] - np.float32(self.cam.pos[0]), P[:, 1] - np.float32(self.cam.pos[1]), P[:, 2] - np.float32(self.cam.pos[2])
        (a, b, c), (d, e, g), (h, i, j) = self.cam.R.astype(np.float32)  # small matmuls are slow in numpy
        z = px * h + py * i + pz * j
        inv = self.f / np.maximum(z, near)
        return self.W * 0.5 + (px * a + py * b + pz * c) * inv, self.H * 0.5 - (px * d + py * e + pz * g) * inv, z, inv

    def project_dirs(self, D):
        """Directions at infinity (sky). Returns x, y, visible-mask."""
        D = np.asarray(D, np.float32)
        (a, b, c), (d, e, g), (h, i, j) = self.cam.R.astype(np.float32)
        z = D[:, 0] * h + D[:, 1] * i + D[:, 2] * j
        inv = self.f / np.maximum(z, 1e-3)
        return (self.W * 0.5 + (D[:, 0] * a + D[:, 1] * b + D[:, 2] * c) * inv,
                self.H * 0.5 - (D[:, 0] * d + D[:, 1] * e + D[:, 2] * g) * inv, z > 0.05)

    # ---------------------------------------------------------------- splatting
    @staticmethod
    def _taps(x, y, Wb, Hb):
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        ix = x0.astype(np.int64)
        iy = y0.astype(np.int64)
        ok = (ix >= 0) & (iy >= 0) & (ix < Wb - 1) & (iy < Hb - 1)
        if not ok.all():
            ix, iy, fx, fy = ix[ok], iy[ok], fx[ok], fy[ok]
        i0 = iy * Wb + ix
        gx, gy = 1 - fx, 1 - fy
        idx = np.concatenate([i0, i0 + 1, i0 + Wb, i0 + Wb + 1])
        w = np.concatenate([gx * gy, fx * gy, gx * fy, fx * fy])
        return idx, w, ok

    def _push(self, store, x, y, col, Wb, Hb):
        if len(x) == 0:
            return
        idx, w, ok = self._taps(x, y, Wb, Hb)
        if len(idx) == 0:
            return
        c = col if ok.all() else col[ok]
        c4 = np.tile(c.T.astype(np.float32), (1, 4))  # (3, 4n)
        store.append((idx, c4 * w))

    def points(self, x, y, col, r=None, layer="back", bloom=0.35):
        """Screen-space points. col (N,3) = energy (~ peak value of a tiny pixel-centred point).

        r (px radius, scalar or array) selects the footprint: <0.8 single bilinear tap, <3 four
        offset taps (3x3-ish tent), <10 the blurred quarter-res buffer, larger a 1/16-res buffer
        (bokeh-sized blobs). Energy is conserved, so big points get dimmer per pixel."""
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = np.broadcast_to(col, (len(x), 3))
        if r is not None:
            r = np.broadcast_to(np.asarray(r, np.float32), x.shape)
            mg = 8 + r * 1.5
        else:
            mg = 8
        m = (x > -mg) & (x < self.W + mg) & (y > -mg) & (y < self.H + mg)
        if not m.all():
            x, y, col = x[m], y[m], col[m]
            if r is not None:
                r = r[m]
        if len(x) == 0:
            return
        fine = self._fine[layer]
        ds = self.ds
        if r is None:
            self._push(fine, x, y, col, self.W, self.H)
        else:
            small = r < 0.8
            mid = (~small) & (r < 3.0)
            big = (r >= 3.0) & (r < 10.0)
            huge = r >= 10.0
            if small.any():
                self._push(fine, x[small], y[small], col[small], self.W, self.H)
            if mid.any():
                xm, ym, cm, rm = x[mid], y[mid], col[mid] * 0.25, r[mid] * 0.5
                for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    self._push(fine, xm + ox * rm, ym + oy * rm, cm, self.W, self.H)
            if big.any():
                self._push(self._soft, x[big] / ds, y[big] / ds, col[big] * (1.0 / ds ** 2), self.Ws, self.Hs)
            if huge.any():
                dc = ds * 4
                xh, yh, rh, ch = x[huge] / dc, y[huge] / dc, r[huge] / dc, col[huge] * (1.0 / dc ** 2)
                one = rh < 1.5
                if one.any():
                    self._push(self._coarse, xh[one], yh[one], ch[one], self.Wc, self.Hc)
                if (~one).any():
                    xl, yl, rl, cl = xh[~one], yh[~one], np.minimum(rh[~one] * 0.5, 0.8), ch[~one] / 9
                    for ox in (-1, 0, 1):
                        for oy in (-1, 0, 1):
                            self._push(self._coarse, xl + ox * rl, yl + oy * rl, cl, self.Wc, self.Hc)
        if bloom:
            self._push(self._soft, x / ds, y / ds, col * (bloom / ds ** 2), self.Ws, self.Hs)

    def soft(self, x, y, col):
        """Deposit directly into the blurred quarter-res buffer (screen px coords, energy units)."""
        x = np.asarray(x, np.float32) / self.ds
        y = np.asarray(y, np.float32) / self.ds
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = np.broadcast_to(col, (len(x), 3))
        self._push(self._soft, x, y, col * (1.0 / self.ds ** 2), self.Ws, self.Hs)

    def points3d(self, P, col, size=0.0, size_px=0.0, near=0.05, layer="back", bloom=0.35, dof=None):
        """World points. size = world radius (perspective scaled) + size_px. dof=(focus_z, strength)
        enlarges points away from the focus distance (fake depth of field). Returns depth z of kept."""
        x, y, z, inv = self.project(P)
        m = z > near
        col = np.asarray(col, np.float32)
        if col.ndim == 2:
            col = col[m]
        x, y, z, inv = x[m], y[m], z[m], inv[m]
        r = np.asarray(size, np.float32)
        r = (r[m] if r.ndim else r) * inv + size_px * self.k
        if dof is not None:
            fz, s = dof
            r = r + np.abs(z - fz) / fz * s * self.k
        self.points(x, y, col, r, layer, bloom)
        return z

    @staticmethod
    def sample_segments(x0, y0, x1, y1, spacing=0.8, maxn=400):
        """Resample screen segments into points. Returns xs, ys, seg index, param s, step length (px)."""
        dx, dy = x1 - x0, y1 - y0
        L = np.sqrt(dx * dx + dy * dy)
        n = np.clip(np.ceil(L / spacing), 1, maxn).astype(np.int64)
        seg = np.repeat(np.arange(len(n)), n)
        start = np.cumsum(n) - n
        s = ((np.arange(len(seg)) - start[seg]) + 0.5) / n[seg]
        s = s.astype(np.float32)
        return x0[seg] + dx[seg] * s, y0[seg] + dy[seg] * s, seg, s, (L / n)[seg].astype(np.float32)

    def lines(self, x0, y0, x1, y1, c0, c1=None, width=0.0, layer="back", bloom=0.35, spacing=0.8, maxn=400):
        """Anti-aliased additive screen-space lines. c0/c1 (N,3): colour (per px of length) at both ends."""
        x0, y0, x1, y1 = (np.asarray(a, np.float32) for a in (x0, y0, x1, y1))
        if len(x0) == 0:
            return
        c0 = np.broadcast_to(np.asarray(c0, np.float32), (len(x0), 3))
        c1 = c0 if c1 is None else np.broadcast_to(np.asarray(c1, np.float32), (len(x0), 3))
        # cull segments fully off-screen
        m = ~(((x0 < -2) & (x1 < -2)) | ((x0 > self.W + 2) & (x1 > self.W + 2)) |
              ((y0 < -2) & (y1 < -2)) | ((y0 > self.H + 2) & (y1 > self.H + 2)))
        if not m.all():
            x0, y0, x1, y1, c0, c1 = x0[m], y0[m], x1[m], y1[m], c0[m], c1[m]
            if np.ndim(width):
                width = np.asarray(width)[m]
        xs, ys, seg, s, step = self.sample_segments(x0, y0, x1, y1, spacing, maxn)
        col = (c0[seg] + (c1[seg] - c0[seg]) * s[:, None]) * step[:, None]
        r = np.asarray(width, np.float32)
        r = r[seg] if r.ndim else r
        self.points(xs, ys, col, r if np.any(r) else None, layer, bloom)

    def lines3d(self, A, B, c0, c1=None, width=0.0, width_world=0.0, near=0.05, layer="back", bloom=0.35):
        xa, ya, za, ia = self.project(A)
        xb, yb, zb, ib = self.project(B)
        m = (za > near) & (zb > near)
        c0 = np.broadcast_to(np.asarray(c0, np.float32), (len(xa), 3))[m]
        c1 = None if c1 is None else np.broadcast_to(np.asarray(c1, np.float32), (len(xa), 3))[m]
        w = width * self.k + width_world * 0.5 * (ia[m] + ib[m])
        self.lines(xa[m], ya[m], xb[m], yb[m], c0, c1, w, layer, bloom)

    def tubes(self, A, B, radius, c0, c1=None, core=0.25, rim=1.0, back_rim=0.45, spec=1.3, width=1.0,
              light=(-0.55, -0.83), dof=None, near=0.05, bloom=0.35, chain=False):
        """Glassy neon tubes along 3D segments A->B (world radius): soft glowing core, crisp
        anti-aliased silhouette rims and a specular stripe that slides towards the light side.
        chain=True means B[i] == A[i+1] (a polyline): vertex normals are shared so rims stay continuous.
        dof=(focus_z, px) widens rims away from the focus distance (they migrate into the soft buffers)."""
        xa, ya, za, ia = self.project(A)
        xb, yb, zb, ib = self.project(B)
        n = len(xa)
        c0 = np.broadcast_to(np.asarray(c0, np.float32), (n, 3))
        c1 = c0 if c1 is None else np.broadcast_to(np.asarray(c1, np.float32), (n, 3))
        dx, dy = xb - xa, yb - ya
        L = np.sqrt(dx * dx + dy * dy) + 1e-6
        nx, ny = -dy / L, dx / L
        if chain and n > 1:
            vx = np.concatenate([nx[:1], nx[:-1] + nx[1:], nx[-1:]])
            vy = np.concatenate([ny[:1], ny[:-1] + ny[1:], ny[-1:]])
            vl = np.sqrt(vx * vx + vy * vy) + 1e-6
            vx, vy = vx / vl, vy / vl
            nax, nay, nbx, nby = vx[:-1], vy[:-1], vx[1:], vy[1:]
        else:
            nax, nay, nbx, nby = nx, ny, nx, ny
        m = (za > near) & (zb > near)
        if not m.any():
            return
        sel = lambda *arrs: [q[m] for q in arrs]
        xa, ya, za, ia, xb, yb, zb, ib, nax, nay, nbx, nby, c0, c1 = sel(xa, ya, za, ia, xb, yb, zb, ib, nax, nay, nbx, nby, c0, c1)
        ra, rb = radius * ia, radius * ib
        sa = np.clip(nax * light[0] + nay * light[1], -1, 1)  # how much the +n side faces the light
        sb = np.clip(nbx * light[0] + nby * light[1], -1, 1)
        w = np.full(len(xa), width * self.k, np.float32)
        if dof is not None:
            w = w + np.abs(0.5 * (za + zb) - dof[0]) / dof[0] * dof[1] * self.k
        sm = 0.5 * (sa + sb)
        for side in (1.0, -1.0):
            g = (rim + back_rim) * 0.5 + (rim - back_rim) * 0.5 * side * sm
            self.lines(xa + nax * ra * side, ya + nay * ra * side, xb + nbx * rb * side, yb + nby * rb * side,
                       c0 * g[:, None], c1 * g[:, None], w, bloom=bloom)
        if spec > 0:
            oa, ob = 0.5 * sa, 0.5 * sb
            g = (spec * (0.4 + 0.6 * np.abs(sm)))[:, None]
            self.lines(xa + nax * ra * oa, ya + nay * ra * oa, xb + nbx * rb * ob, yb + nby * rb * ob,
                       (c0 * 0.6 + 120) * g, (c1 * 0.6 + 120) * g, w * 1.6 + 0.6 * self.k, bloom=bloom)
        if core > 0:
            rr = 0.5 * (ra + rb)
            xs, ys, seg, s, step = self.sample_segments(xa, ya, xb, yb, 1.0, 400)
            stride = np.maximum(1, (rr[seg] * 0.3).astype(np.int64))
            keep = (np.arange(len(seg)) % stride) == 0
            xs, ys, seg, s, step = xs[keep], ys[keep], seg[keep], s[keep], step[keep] * stride[keep]
            col = (c0[seg] + (c1[seg] - c0[seg]) * s[:, None]) * (core * step * 2 * rr[seg])[:, None]
            self.points(xs, ys, col, rr[seg] * 0.9, bloom=0)

    # ---------------------------------------------------------------- compositing
    def _accumulate(self, store, Wb, Hb, out=None):
        """Sum splats into an (Hb, Wb, 3) float32 image (added into ``out`` if given)."""
        if out is None:
            out = np.zeros((Hb, Wb, 3), np.float32)
        if not store:
            return out
        idx = np.concatenate([s[0] for s in store])
        wc = np.concatenate([s[1] for s in store], axis=1)
        n = Wb * Hb
        for c in range(3):
            out[..., c] += np.bincount(idx, wc[c], minlength=n)[:n].reshape(Hb, Wb)
        return out

    def finish(self, base=None, canvas=None, bloom=(0.55, 0.45), radii=(1.6, 7.0), exposure=1.0,
               white=325.0, knee=260.0, soft_extra=None):
        """Resolve buffers -> float32 (H, W, 3) image in 0..~255.

        Everything is additive, so order does not matter.  base: optional full-res float image (or a
        (y0, x0, region) tuple) added in.  canvas: optional opaque uint8 RGB PIL image (painter's
        geometry), also bloomed.  soft_extra: (Hs, Ws, 3) array added to the soft buffer.
        Tone curve: Reinhard  y = white * x / (x + knee / exposure)  (never hard-clips)."""
        soft = self._accumulate(self._soft, self.Ws, self.Hs)
        if self._coarse:
            co = self._accumulate(self._coarse, self.Wc, self.Hc)
            for _ in range(2):  # small separable [1 2 1] blur hides the tap lattice of big blobs
                co[1:-1] = co[1:-1] * 0.5 + (co[:-2] + co[2:]) * 0.25
                co[:, 1:-1] = co[:, 1:-1] * 0.5 + (co[:, :-2] + co[:, 2:]) * 0.25
            box = (0, 0, self.Ws / 4, self.Hs / 4)
            for c in range(3):
                soft[..., c] += np.asarray(Image.fromarray(np.ascontiguousarray(co[..., c]), "F").resize(
                    (self.Ws, self.Hs), Image.BILINEAR, box=box))
        if soft_extra is not None:
            soft += soft_extra
        if canvas is not None:
            small = np.asarray(canvas.reduce(self.ds), np.float32)
            soft[:small.shape[0], :small.shape[1]] += small * 0.5
        if isinstance(base, tuple):
            y0, x0, reg = base
            sm = reg[::self.ds, ::self.ds]
            sy, sx = y0 // self.ds, x0 // self.ds
            hh, ww = min(sm.shape[0], self.Hs - sy), min(sm.shape[1], self.Ws - sx)
            soft[sy:sy + hh, sx:sx + ww] += sm[:hh, :ww] * 0.35
        # bloom: two gaussian radii on the quarter-res buffer, recombined, upsampled once (bilinear)
        s8 = Image.fromarray(np.clip(soft, 0, 255).astype(np.uint8))
        b1 = np.asarray(s8.filter(ImageFilter.GaussianBlur(radii[0] * self.k)), np.float32)
        b2 = np.asarray(s8.filter(ImageFilter.GaussianBlur(radii[1] * self.k)), np.float32)
        comb = b1 * bloom[0] + b2 * bloom[1] + soft * 0.25
        up = Image.fromarray(np.clip(comb, 0, 255).astype(np.uint8)).resize(
            (self.W, self.H), Image.BILINEAR, box=(0, 0, self.W / self.ds, self.H / self.ds))
        up = np.asarray(up)
        out = np.empty((self.H, self.W, 3), np.float32)
        store = self._fine["back"] + self._fine["front"]
        n = self.W * self.H
        if store:
            idx = np.concatenate([s[0] for s in store])
            wc = np.concatenate([s[1] for s in store], axis=1)
            for c in range(3):
                acc = np.bincount(idx, wc[c], minlength=n)[:n].reshape(self.H, self.W)
                acc += up[..., c]
                out[..., c] = acc
        else:
            out[:] = up
        # tone-map the additive light only; designed base colours (sky, planet, canvas) are added after
        den = out  # in place: y = white - white*k/(x+k)
        den += knee / exposure
        np.divide(-white * knee / exposure, den, out=den)
        den += white
        if canvas is not None:
            den += np.asarray(canvas)[:self.H, :self.W]
        if isinstance(base, tuple):
            y0, x0, reg = base
            den[y0:y0 + reg.shape[0], x0:x0 + reg.shape[1]] += reg
        elif base is not None:
            den += base
        return den


# ======================================================================== scene base
class Scene3D(Scene):
    dynamic = True

    def build(self):
        self.R = Renderer(self.W, self.H)
        self.k = self.R.k
        self.speed = float(self.p.get("speed", 1.0))
        self.setup()

    def setup(self):
        pass

    def render(self, t):
        return self.frame(float(t) * self.speed)

    def frame(self, t):
        raise NotImplementedError

    # common: background stars at infinity
    def make_sky(self, n=3500, rng=None, bright=1.0):
        rng = rng or self.rng
        self.sky_dir = random_dirs(rng, n).astype(np.float32)
        b = 0.08 + rng.random(n) ** 5 * 0.9 + rng.random(n) ** 40 * 2.5
        tint = rng.choice(np.array([[1, 1, 1], [0.75, 0.85, 1], [1, 0.88, 0.7], [0.7, 0.8, 1]], np.float32), n)
        self.sky_col = (tint * b[:, None] * 160 * bright).astype(np.float32)
        self.sky_r = np.where(b > 1.2, 1.2, 0.0).astype(np.float32)

    def draw_sky(self, mask_fn=None):
        x, y, vis = self.R.project_dirs(self.sky_dir)
        if mask_fn is not None:
            vis &= mask_fn(self.sky_dir)
        self.R.points(x[vis], y[vis], self.sky_col[vis], self.sky_r[vis] * self.k, bloom=0.5)

    def gradient(self, top, bottom, power=1.0):
        """Static vertical background gradient (float HxWx3); cheap because camera roll is small."""
        yy = np.linspace(0, 1, self.H, dtype=np.float32)[:, None] ** power
        col = lerp_col(top, bottom, yy)  # (H,1,3)
        return np.broadcast_to(col, (self.H, self.W, 3))


PAL3D = {
    "cyan": ((70, 210, 255), (150, 90, 255)),
    "violet": ((170, 90, 255), (70, 220, 255)),
    "ember": ((255, 140, 60), (255, 60, 120)),
    "gold": ((255, 200, 110), (255, 120, 60)),
    "teal": ((40, 230, 200), (60, 140, 255)),
    "pink": ((255, 90, 190), (90, 160, 255)),
}


def _pal(p, default):
    pal = p.get("palette", default)
    if isinstance(pal, (list, tuple)) and len(pal) == 2:
        return tuple(np.asarray(c, np.float32) for c in pal)
    a, b = PAL3D.get(pal, PAL3D[default])
    return np.asarray(a, np.float32), np.asarray(b, np.float32)


# ======================================================================== 1. galaxy
class Galaxy3D(Scene3D):
    """Spiral galaxy: ~150k particles, log-spiral arms, dust lanes, HII knots, differential rotation."""

    def setup(self):
        rng = self.rng
        n = int(self.p.get("particles", 150000) * min(1.0, self.k * 1.2 + 0.05))
        arms = int(self.p.get("arms", 2))
        pitch = math.radians(self.p.get("pitch", 17.0))
        tanp = math.tan(pitch)
        armw = TAU / arms
        # ---- disk
        nd = int(n * 0.74)
        r = 0.04 + rng.gamma(1.7, 0.26, nd)
        bad = r > 1.35
        r[bad] = rng.uniform(0.1, 1.35, bad.sum())
        arm = rng.integers(0, arms, nd)
        second = rng.random(nd) < 0.3  # fainter secondary arms between the main ones
        base = np.log(np.maximum(r, 0.06) / 0.06) / tanp + arm * armw + second * armw * 0.5
        in_arm = rng.random(nd) < 0.75
        th = np.where(in_arm, base + rng.normal(0, 1, nd) * (0.2 + 0.12 * r), rng.uniform(0, TAU, nd))
        phase = th - np.log(np.maximum(r, 0.06) / 0.06) / tanp
        d = ((phase / armw + 0.5) % 1.0 - 0.5) * armw  # angular offset to nearest arm centre
        d2 = ((phase / armw) % 1.0 - 0.5) * armw  # ... to nearest secondary arm
        armness = np.maximum(np.exp(-(d / (0.14 + 0.1 * r)) ** 2), 0.45 * np.exp(-(d2 / (0.12 + 0.08 * r)) ** 2))
        y = rng.normal(0, 1, nd) * (0.012 + 0.05 * np.exp(-r / 0.14))
        dust = 1 - 0.8 * np.exp(-((d + 0.13 + 0.05 * r) / 0.055) ** 2) * smoothstep(0.12, 0.3, r)
        core = np.exp(-r / 0.22)
        col = lerp_col((255, 214, 165), (165, 195, 255), np.clip(armness * 1.1 - core * 0.6, 0, 1))
        col = lerp_col(col, (255, 196, 140), np.clip(core * 1.2, 0, 1))
        clump_tex = periodic_noise((arms, 256), rng, beta=1.6)
        along = (np.log(np.maximum(r, 0.06) / 0.06) * 40).astype(np.int64) % 256
        clump = 0.35 + 1.3 * clump_tex[arm, along] ** 2
        L = (0.22 + 1.1 * armness * clump) * (np.exp(-r / 0.5) + 0.04) * dust * (0.55 + 0.9 * rng.random(nd) ** 2)
        dcol = col * L[:, None] * 13
        # ---- bulge
        nb = int(n * 0.16)
        rb = rng.gamma(1.3, 0.075, nb)
        vb = random_dirs(rng, nb) * rb[:, None] * np.array([1, 0.62, 1])
        rbx = np.sqrt(vb[:, 0] ** 2 + vb[:, 2] ** 2)
        bcol = lerp_col((255, 236, 205), (255, 180, 110), np.clip(rb / 0.3, 0, 1)) * (11 * np.exp(-rb / 0.25))[:, None]
        # ---- knots (HII regions pink, young clusters blue)
        nk = 140
        kr = 0.2 + 0.95 * rng.random(nk) ** 0.8
        ka = rng.integers(0, arms, nk)
        kth = np.log(kr / 0.06) / tanp + ka * armw + rng.normal(0, 0.08, nk)
        per = max(8, int(n * 0.07 / nk))
        kidx = np.repeat(np.arange(nk), per)
        ks = rng.uniform(0.006, 0.016, nk)[kidx]
        kx = kr[kidx] * np.cos(kth[kidx]) + rng.normal(0, 1, len(kidx)) * ks
        kz = kr[kidx] * np.sin(kth[kidx]) + rng.normal(0, 1, len(kidx)) * ks
        ky = rng.normal(0, 0.008, len(kidx))
        pink = (rng.random(nk) < 0.45)[kidx]
        kcol = np.where(pink[:, None], np.array([255, 85, 165], np.float32), np.array([120, 165, 255], np.float32))
        kcol = kcol * (rng.random(len(kidx)) ** 2 * 30 + 6)[:, None]
        # ---- combine into polar arrays (radius in the plane, angle, height)
        self.r = np.concatenate([r, rbx, np.sqrt(kx ** 2 + kz ** 2)]).astype(np.float32)
        self.th = np.concatenate([th, np.arctan2(vb[:, 2], vb[:, 0]), np.arctan2(kz, kx)]).astype(np.float32)
        self.y = np.concatenate([y, vb[:, 1], ky]).astype(np.float32)
        self.col = np.concatenate([dcol, bcol, kcol]).astype(np.float32)
        self.size = np.concatenate([np.full(nd, 0.0004), np.full(nb, 0.0005), np.full(len(kidx), 0.0011)]).astype(np.float32)
        # flat-ish rotation curve, mild differential so arms keep their shape over a clip
        self.omega = (-(0.035 + 0.025 / (self.r + 0.25))).astype(np.float32)
        self.make_sky(4000, bright=1.6)
        # faint nebular puffs on the sky sphere
        npf = 260
        self.puff_dir = random_dirs(rng, npf).astype(np.float32)
        self.puff_col = (lerp_col((90, 40, 160), (30, 70, 160), rng.random(npf)) * rng.random(npf)[:, None] * 0.9)

    def frame(self, t):
        R = self.R
        dist = 1.95 - 0.3 * (1 - math.exp(-t / 14.0))
        elev = 0.82 + 0.16 * math.sin(t * 0.11 + 0.4)
        azim = 0.9 + t * 0.07
        R.begin(Camera(orbit((0, 0, 0), dist, azim, elev), (0, -0.03, 0), fov=56, roll=0.42 + 0.05 * math.sin(t * 0.09)))
        x, y, vis = R.project_dirs(self.puff_dir)
        R.soft(x[vis], y[vis], self.puff_col[vis] * 160)
        self.draw_sky()
        th = self.th + self.omega * t
        P = np.stack([self.r * np.cos(th), self.y, self.r * np.sin(th)], 1)
        R.points3d(P, self.col, self.size, 0.0, bloom=0.5)
        # galactic core glow
        cx, cy, cz, _ = R.project(np.zeros((1, 3)))
        R.points(np.repeat(cx, 2), np.repeat(cy, 2), np.array([[255, 190, 130], [255, 210, 160]]) * 900, [60 * self.k, 14 * self.k], bloom=0)
        return R.finish(bloom=(0.7, 0.55), radii=(1.4, 8.0))


# ======================================================================== 2. warp
class Warp3D(Scene3D):
    """Hyperspace flight: 3D stars rushing past the camera as speed-length streaks."""

    def setup(self):
        rng = self.rng
        n = int(self.p.get("stars", 2600))
        self.L = 70.0
        ang = rng.uniform(0, TAU, n)
        rad = 0.35 + 7.5 * rng.random(n) ** 1.6
        self.sx, self.sy = (rad * np.cos(ang)).astype(np.float32), (rad * np.sin(ang)).astype(np.float32)
        self.sz = rng.uniform(0, self.L, n).astype(np.float32)
        a, b = _pal(self.p, "cyan")
        tints = np.stack([np.array([235, 242, 255], np.float32), a, b, np.array([255, 255, 255], np.float32)])
        pick = rng.choice(4, n, p=[0.55, 0.2, 0.12, 0.13])
        self.col = tints[pick] * (0.4 + 0.8 * rng.random(n) ** 2)[:, None]
        # tunnel gas: big soft blobs on a wider cylinder
        m = 1400
        ga = rng.uniform(0, TAU, m)
        gr = rng.uniform(2.2, 6.0, m)
        self.gx, self.gy = (gr * np.cos(ga)).astype(np.float32), (gr * np.sin(ga)).astype(np.float32)
        self.gz = rng.uniform(0, self.L, m).astype(np.float32)
        self.gcol = lerp_col(a, b, rng.random(m)) * (0.3 + 0.7 * rng.random(m))[:, None]
        self.v0 = float(self.p.get("v0", 5.0))
        self.v1 = float(self.p.get("v1", 55.0))
        self.tau = float(self.p.get("ramp", 4.5))
        self.ca, self.cb = a, b

    def _s(self, t):  # distance travelled (closed form of the speed ramp)
        return self.v1 * t - (self.v1 - self.v0) * self.tau * (1 - math.exp(-t / self.tau))

    def frame(self, t):
        R = self.R
        v = self.v1 + (self.v0 - self.v1) * math.exp(-t / self.tau)
        s = self._s(t)
        yaw = 0.06 * math.sin(t * 0.21)
        pitch = 0.05 * math.sin(t * 0.17 + 1.0)
        fwd = np.array([math.sin(yaw), math.sin(pitch), math.cos(yaw)])
        R.begin(Camera((0, 0, 0), fwd, fov=70, roll=t * 0.05 + 0.15 * math.sin(t * 0.3)))
        near = 0.4
        z = (self.sz - s) % self.L + near
        dt = 0.035 + 0.02 * min(1.0, v / self.v1)  # streak exposure time
        zt = z + v * dt  # where it was a moment ago (farther away)
        head = np.stack([self.sx, self.sy, z], 1)
        tail = np.stack([self.sx, self.sy, zt], 1)
        fade = smoothstep(self.L + near, self.L * 0.3, z) * smoothstep(near, near + 1.2, z) * np.clip(9.0 / z, 0, 1.6)
        # energy per streak roughly constant -> per-length brightness drops as streaks get long
        xa, ya, za, ia = R.project(head)
        xb, yb, zb, ib = R.project(tail)
        plen = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) + 1.0
        per_len = (fade * 26.0 / np.sqrt(plen))[:, None]
        width = np.clip(0.03 * ia, 0, 2.6)
        R.lines(xb, yb, xa, ya, self.col * per_len * 0.15, self.col * per_len, width, bloom=0.5)
        R.points(xa, ya, self.col * (fade * 60)[:, None], np.clip(0.04 * ia, 0.5, 6), bloom=0.6)
        # tunnel gas: big dim blobs smeared along the flight direction + glow at the vanishing point
        gz = (self.gz - s * 0.9) % self.L + near
        gfade = smoothstep(self.L * 0.65, self.L * 0.2, gz) * smoothstep(near + 1.0, 9.0, gz)
        nsm = 6
        for j in range(nsm):
            off = j * 0.5 * (0.4 + v / self.v1)
            gx, gy, gzz, gi = R.project(np.stack([self.gx, self.gy, gz + off], 1))
            gr = np.clip(0.9 * gi, 6, 220)
            R.points(gx, gy, self.gcol * (gfade * (0.004 + 0.0002 * v) / nsm * 3 * math.pi * gr * gr)[:, None], gr, bloom=0)
        cx, cy, _, _ = R.project(np.array([[0, 0, 60.0]]))
        R.soft(np.repeat(cx, 2), np.repeat(cy, 2), np.stack([self.ca * 30, np.array([255, 255, 255]) * (12 + v)]))
        return R.finish(bloom=(0.6, 0.6), radii=(1.5, 9.0))


# ======================================================================== 3. globe
class Globe3D(Scene3D):
    """Holographic Earth: lat/long wireframe, continent dot cloud, atmospheric rim, orbit rings."""

    def setup(self):
        rng = self.rng
        a, b = _pal(self.p, "cyan")
        self.ca, self.cb = a, b
        nd = int(self.p.get("dots", 70000) * min(1.0, self.k + 0.1))
        pts = fib_sphere(nd)
        vol = periodic_noise((48, 48, 48), rng, beta=2.6)
        n = trilinear(vol, (pts * 0.5 + 0.5) * 30 + 7)
        thr = np.quantile(n, 0.63)
        land = n > thr
        coast = land & (n < thr + 0.035)
        self.land = pts[land].astype(np.float32)
        self.land_b = np.where(coast[land], 1.7, 1.0).astype(np.float32) * (0.6 + 0.5 * rng.random(land.sum()))
        self.city = np.where(rng.random(land.sum()) < 0.025, 3.5, 1.0).astype(np.float32)  # sparkle "cities"
        # sparse ocean dots (very faint)
        self.ocean = pts[~land][::5].astype(np.float32)
        # wireframe segments (lat every 15 deg, meridians every 15 deg)
        segs = []
        lon = np.linspace(0, TAU, 181)
        for la in np.radians(np.arange(-75, 76, 15)):
            p = np.stack([np.cos(la) * np.cos(lon), np.full_like(lon, np.sin(la)), np.cos(la) * np.sin(lon)], 1)
            segs.append(np.stack([p[:-1], p[1:]], 1))
        lat = np.linspace(-math.pi / 2, math.pi / 2, 91)
        for lo in np.radians(np.arange(0, 180, 15)):
            for off in (0, math.pi):
                p = np.stack([np.cos(lat) * np.cos(lo + off), np.sin(lat), np.cos(lat) * np.sin(lo + off)], 1)
                segs.append(np.stack([p[:-1], p[1:]], 1))
        segs = np.concatenate(segs) * 1.004
        self.segA, self.segB = segs[:, 0].astype(np.float32), segs[:, 1].astype(np.float32)
        # orbit rings: (radius, tilt matrix, speed, satellites)
        self.rings = []
        for i, (rad, tx, tz, sp) in enumerate([(1.38, 0.35, 0.1, 0.5), (1.62, -0.55, 0.45, -0.35), (1.95, 0.15, -0.7, 0.25)]):
            M = rot_z(tz) @ rot_x(tx)
            self.rings.append((rad, M.astype(np.float32), sp, rng.uniform(0, TAU, 2)))
        th = np.linspace(0, TAU, 721)
        self.ring_unit = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1).astype(np.float32)
        self.dash = ((np.arange(720) // 6) % 2 == 0)
        # holo pedestal rings + rising motes
        self.mote = np.stack([rng.uniform(0, TAU, 700), rng.uniform(0, 1.15, 700) ** 0.5, rng.random(700)], 1).astype(np.float32)
        self.tilt = rot_z(math.radians(23.4)).astype(np.float32)
        self.make_sky(1800, bright=0.6)

    def frame(self, t):
        R, a, b = self.R, self.ca, self.cb
        azim = 0.3 + t * 0.12
        elev = 0.28 + 0.1 * math.sin(t * 0.15)
        cam_pos = orbit((0, -0.15, 0), 6.4 - 0.4 * math.sin(t * 0.08), azim, elev)
        R.begin(Camera(cam_pos, (0, -0.3, 0), fov=42, roll=0.05 * math.sin(t * 0.13)))
        self.draw_sky()
        M = (self.tilt @ rot_y(t * 0.22).astype(np.float32)).T  # row-vector convention
        view = cam_pos.astype(np.float32)
        scan = math.sin(t * 0.7) * 1.05  # hologram scan band (world y)

        def facing(P):
            v = view[None, :] - P
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            return np.sum(P * v, 1)

        # continents
        L = self.land @ M
        f = facing(L)
        bright = np.where(f > 0, 0.35 + 0.65 * f, 0.10) * self.land_b
        band = np.exp(-((L[:, 1] - scan) / 0.035) ** 2)
        colL = lerp_col(a, (235, 255, 255), np.clip(band * 0.6, 0, 1)) * (bright * (20 + 25 * band) * self.city)[:, None]
        R.points3d(L, colL, 0.0, 0.5, bloom=0.5)
        O = self.ocean @ M
        fo = facing(O)
        R.points3d(O, (b * 0.5 + a * 0.5) * (np.where(fo > 0, 7.0, 1.5))[:, None], 0.0, 0.5, bloom=0)
        # wireframe
        A, B = self.segA @ M, self.segB @ M
        fa = facing((A + B) * 0.5)
        wb = np.where(fa > 0, 0.55 + 0.45 * fa, 0.12)
        R.lines3d(A, B, (b * 0.35 + a * 0.25)[None, :] * wb[:, None], width=0.9, bloom=0.4)
        # atmospheric rim: tangent circle of the sphere seen from the camera
        d = float(np.linalg.norm(cam_pos))
        nrm = cam_pos / d
        cen = nrm * (1.0 / d)
        rr = math.sqrt(max(1e-6, 1 - 1 / d ** 2))
        u = _norm(np.cross(nrm, [0, 1, 0]))
        w = np.cross(nrm, u)
        th = np.linspace(0, TAU, 900, endpoint=False)
        rim = cen + rr * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
        for scale, colr in ((1.0, a * 1.6), (1.03, a * 5.0), (1.07, b * 4.0)):
            x, y, z, _ = R.project(rim * scale)
            if scale == 1.0:
                R.points(x, y, np.broadcast_to(colr, (len(x), 3)), 1.2, bloom=1.5)
            else:
                R.soft(x, y, np.broadcast_to(colr, (len(x), 3)))
        # orbit rings with satellites and trails
        for rad, Mt, sp, phases in self.rings:
            P = (self.ring_unit[:-1] * rad) @ Mt.T
            fP = np.where(facing(P) > -0.6, 1.0, 0.35)  # crude: dim the part hidden behind the globe
            behind = (np.linalg.norm(P - view, axis=1) > d) & (np.linalg.norm(np.cross(P - view, nrm), axis=1) < 1.0)
            fP = np.where(behind, 0.25, 1.0)
            cols = (a * 0.6)[None, :] * (fP * self.dash)[:, None] * 0.8
            R.points3d(P, cols, 0.004, bloom=0.4)
            for ph in phases:
                ang = ph + t * sp
                tr = np.linspace(0, 0.9 * np.sign(sp), 60)
                angs = ang - tr
                S = np.stack([np.cos(angs), np.zeros_like(angs), np.sin(angs)], 1) * rad @ Mt.T
                fade = np.exp(-np.linspace(0, 4, 60))
                hid = ((np.linalg.norm(S - view, axis=1) > d) & (np.linalg.norm(np.cross(S - view, nrm), axis=1) < 1.0))
                fade = fade * np.where(hid, 0.15, 1.0)
                cS = lerp_col(a, (255, 255, 255), fade) * (fade * 3.0)[:, None]
                R.lines3d(S[:-1], S[1:], cS[:-1], cS[1:], width=0.8 + 1.5 * fade[:-1], bloom=0.8)
                R.points3d(S[:1], np.array([[230, 250, 255]], np.float32) * 1.5 * fade[0], 0.0, 2.2, bloom=1.0)
        # pedestal: concentric rings + rising motes
        ped_y = -1.42
        for i, pr in enumerate((0.55, 0.85, 1.15, 1.25)):
            th2 = np.linspace(0, TAU, 360, endpoint=False) + t * (0.3 if i % 2 else -0.2)
            P = np.stack([pr * np.cos(th2), np.full_like(th2, ped_y), pr * np.sin(th2)], 1)
            seg = ((np.arange(360) // (12 + 6 * i)) % 3 != 0)
            R.points3d(P, (a * (0.9 if i < 3 else 0.5))[None, :] * seg[:, None] * 0.9, 0.006, bloom=0.6)
        mt = (self.mote[:, 2] + t * 0.12) % 1.0
        P = np.stack([self.mote[:, 1] * np.cos(self.mote[:, 0]), ped_y + mt * 2.2, self.mote[:, 1] * np.sin(self.mote[:, 0])], 1)
        R.points3d(P, a[None, :] * (np.sin(mt * math.pi) ** 2 * 35)[:, None], 0.005, bloom=0.6)
        return R.finish(bloom=(0.75, 0.6), radii=(1.3, 7.0))


# ======================================================================== 4. DNA
class DNA3D(Scene3D):
    """Double helix: glassy neon backbone tubes, coloured base-pair rungs, depth-of-field bokeh."""

    def setup(self):
        rng = self.rng
        self.ca, self.cb = _pal(self.p, "teal")
        self.base_cols = np.array([[255, 70, 120], [255, 190, 60], [60, 235, 150], [90, 140, 255]], np.float32)
        self.seq = rng.integers(0, 4, 4096)
        nb = 1400
        self.bok = np.stack([rng.uniform(-10, 10, nb), rng.uniform(-30, 30, nb), rng.uniform(-10, 10, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.ca, self.cb, rng.random(nb)) * (0.25 + rng.random(nb) ** 3)[:, None]
        self.bg = self.gradient((2, 16, 26), (1, 3, 9), 0.9)
        self.groove = math.pi * 0.78  # minor/major groove offset of the second strand
        self.focus = 6.6

    def _helix(self, u, off):
        ang = u * (TAU / 10.0) + off
        return np.stack([np.cos(ang), u * 0.34, np.sin(ang)], -1).astype(np.float32)

    def frame(self, t):
        R = self.R
        cy = t * 0.5
        azim = t * 0.2
        cam = orbit((0, cy, 0), 6.6, azim, -0.16)
        R.begin(Camera(cam, (0, cy + 0.9, 0), fov=52, roll=0.32))
        dof = (self.focus, 2.2)
        k0 = int(math.floor(cy / 0.34))
        ks = np.arange(k0 - 22, k0 + 52)
        # fog: fade with distance from camera
        def fog(P):
            d = np.linalg.norm(P - cam.astype(np.float32), axis=1)
            return np.clip(np.exp(-(d - 5.0) / 6.0), 0, 1.3)[:, None]
        # backbone strands (polyline segments)
        u = np.arange(ks[0] * 8, ks[-1] * 8 + 1, dtype=np.float32) / 8.0
        for off, col in ((0.0, self.ca), (self.groove, self.cb)):
            C = self._helix(u, off)
            A, B = C[:-1], C[1:]
            cc = col * fog((A + B) * 0.5) * 0.9
            R.tubes(A, B, 0.15, cc, core=0.22, rim=1.0, back_rim=0.3, spec=1.0, dof=dof, chain=True)
            # phosphate beads at every base pair
            Pb = self._helix(ks.astype(np.float32), off)
            R.points3d(Pb, (col * 0.5 + 120) * fog(Pb) * 1.2, 0.0, 2.2, bloom=0.8, dof=dof)
        # base-pair rungs: two coloured halves from each strand towards the axis
        kk = ks.astype(np.float32) + 0.0
        Pa, Pb = self._helix(kk, 0.0), self._helix(kk, self.groove)
        mid = (Pa + Pb) * 0.5
        bases = self.seq[ks % 4096]
        comp = 3 - bases
        ca = self.base_cols[bases] * fog(mid) * 1.0
        cb_ = self.base_cols[comp] * fog(mid) * 1.0
        Pa_in = Pa + (mid - Pa) * 0.12
        Pb_in = Pb + (mid - Pb) * 0.12
        R.tubes(np.concatenate([Pa_in, Pb_in]), np.concatenate([mid, mid]), 0.055,
                np.concatenate([ca, cb_]), core=0.35, rim=0.8, back_rim=0.3, spec=0.8, dof=dof)
        # bokeh dust around (wraps vertically with the camera)
        Bk = self.bok.copy()
        Bk[:, 1] = (Bk[:, 1] - cy) % 60 - 30 + cy
        x, y, z, inv = R.project(Bk)
        m = z > 0.3
        rb = 0.03 * inv[m] + np.abs(z[m] - self.focus) * 2.0 * self.k
        R.points(x[m], y[m], self.bok_c[m] * (90 * np.exp(-z[m] / 16))[:, None], rb, bloom=0.2)
        return R.finish(base=self.bg, bloom=(0.6, 0.55), radii=(1.6, 8.0))


# ======================================================================== 5. neural
class Neural3D(Scene3D):
    """3D neural cloud: ~250 nodes, nearest-neighbour edges, pulses travelling along edges."""

    def setup(self):
        rng = self.rng
        n = int(self.p.get("nodes", 250))
        self.ca, self.cb = _pal(self.p, "violet")
        # clustered cloud: a few lobes
        centers = rng.normal(0, 0.9, (6, 3)) * np.array([1.2, 1.6, 1.2])
        pts = centers[rng.integers(0, 6, n)] + rng.normal(0, 0.55, (n, 3))
        self.nodes = pts.astype(np.float32)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        edges = set()
        for i in range(n):
            for j in np.argsort(d[i])[1:4]:
                edges.add((min(i, int(j)), max(i, int(j))))
        e = np.array(sorted(edges))
        self.ei, self.ej = e[:, 0], e[:, 1]
        ne = len(e)
        self.ph = rng.random(ne).astype(np.float32)
        self.sp = rng.uniform(0.25, 0.6, ne).astype(np.float32)
        self.dir = rng.random(ne) < 0.5
        self.nsize = rng.uniform(0.6, 1.3, n).astype(np.float32)
        self.make_sky(1500, bright=0.5)
        nd = 2500
        self.dust = rng.normal(0, 3.5, (nd, 3)).astype(np.float32)
        self.dust_c = lerp_col(self.ca, self.cb, rng.random(nd)) * (rng.random(nd) ** 3 * 0.25)[:, None]

    def frame(self, t):
        R, a, b = self.R, self.ca, self.cb
        cam = orbit((0, 0, 0), 7.4 - 0.8 * math.sin(t * 0.1), 0.4 + t * 0.14, 0.22 * math.sin(t * 0.12))
        R.begin(Camera(cam, (0, 0, 0), fov=50, roll=0.08 * math.sin(t * 0.1)))
        self.draw_sky()
        view = cam.astype(np.float32)
        x, y, z, inv = R.project(self.dust)
        m = z > 0.3
        R.points(x[m], y[m], self.dust_c[m] * 60, np.clip(0.025 * inv[m], 0.5, 60), bloom=0)

        def fog(P):
            return np.exp(-np.maximum(np.linalg.norm(P - view, axis=1) - 5.5, 0) / 4.5).astype(np.float32)

        A, B = self.nodes[self.ei], self.nodes[self.ej]
        fe = fog((A + B) * 0.5)
        R.lines3d(A, B, (a * 0.42 + b * 0.08)[None, :] * fe[:, None], width=1.0, bloom=0.5)
        # pulses: each edge fires periodically (active ~40% of the time), short comet trail
        cyc = 2.4
        ph = (t * self.sp + self.ph * cyc) % cyc
        act = ph < 1.0
        f = np.clip(ph, 0, 1)
        src = np.where(self.dir[:, None], A, B)
        dst = np.where(self.dir[:, None], B, A)
        ntr = 10
        tr = np.linspace(0, 0.14, ntr, dtype=np.float32)
        ff = np.clip(f[:, None] - tr[None, :], 0, 1)
        P = (src[:, None, :] + (dst - src)[:, None, :] * ff[..., None]).reshape(-1, 3)
        w = np.exp(-tr * 22)[None, :] * (f[:, None] >= tr[None, :]) * act[:, None]
        w = (w * smoothstep(1.0, 0.9, f)[:, None] * fe[:, None]).ravel()
        keep = w > 0.01
        cols = lerp_col(b, (235, 255, 255), w[keep] ** 2) * (w[keep] * 90)[:, None]
        R.points3d(P[keep], cols, 0.0, 1.4, bloom=1.0)
        # nodes: fire when a pulse arrives
        arrive = np.exp(-((ph - 1.0) / 0.08) ** 2) * (ph >= 0.9)
        fire = np.zeros(len(self.nodes), np.float32)
        np.add.at(fire, np.where(self.dir, self.ej, self.ei), arrive.astype(np.float32))
        fire = np.clip(fire, 0, 1.2)
        fn = fog(self.nodes)[:, None]
        hot = lerp_col(a, (255, 235, 255), np.clip(fire, 0, 1))
        R.points3d(self.nodes, hot * (220 + 300 * fire)[:, None] * fn, 0.006 * self.nsize, 1.4, bloom=1.2)
        _, _, _, ninv = R.project(self.nodes)
        hr = 0.09 * self.nsize * ninv
        R.points3d(self.nodes, (a * 0.8 + b * 0.2) * ((0.04 + 0.3 * fire) * math.pi * hr * hr)[:, None] * fn,
                   0.09 * self.nsize, 0.0, bloom=0)
        return R.finish(bloom=(0.75, 0.65), radii=(1.5, 8.0))


# ======================================================================== 6. terrain
class Terrain3D(Scene3D):
    """Synthwave wireframe terrain flyover with occlusion, sun and sky (seamless forward motion)."""

    def setup(self):
        rng = self.rng
        self.C = 49  # columns
        self.dx = 1.0
        self.rows = 64
        self.Tz = 256
        noise = periodic_noise((self.Tz, 64), rng, beta=2.4)[:, :self.C]
        fine = periodic_noise((self.Tz, 64), rng, beta=1.4)[:, :self.C]
        xs = (np.arange(self.C) - (self.C - 1) / 2) * self.dx
        self.xs = xs.astype(np.float32)
        valley = smoothstep(1.2, 9.0, np.abs(xs)) ** 1.2
        h = (noise ** 1.5) * 14.0 * valley[None, :] + fine * 0.3 * (0.3 + valley[None, :])
        self.hmap = h.astype(np.float32)
        self.hmax = float(h.max())
        self.ca, self.cb = _pal(self.p, "pink")
        # sky: gradient, stars, striped sun (camera orientation is fixed -> static image)
        W, H = self.W, self.H
        self.pitch = -0.1
        cam = Camera((0, 2.2, 0), (0, 2.2 + math.tan(self.pitch) * 10, 10), fov=62)
        self.R.begin(cam)
        _, hy, _, _ = self.R.project(np.array([[0, 2.2, 1e5]]))
        self.hy = float(hy[0])
        yy = np.arange(H, dtype=np.float32)[:, None]
        v = np.clip((self.hy - yy) / self.hy, 0, 1)
        sky = lerp_col((150, 30, 110), (4, 2, 16), v[..., 0] ** 0.45)[:, None, :]
        sky = np.repeat(sky.reshape(H, 1, 3), W, 1)
        st = np.zeros((H, W, 3), np.float32)
        ns = int(W * H * 0.0004)
        sx, sy = rng.integers(0, W, ns), rng.integers(0, int(max(1, self.hy * 0.85)), ns)
        st[sy, sx] = (rng.random(ns) ** 3 * 220)[:, None] * np.array([0.9, 0.9, 1.0])
        sky += st
        xx = np.arange(W, dtype=np.float32)[None, :]
        sr = W * 0.27
        scx, scy = W * 0.5, self.hy - sr * 0.42
        dd = np.sqrt((xx - scx) ** 2 + (yy - scy) ** 2)
        sg = np.clip((yy - (scy - sr)) / (2 * sr), 0, 1)
        sun_col = lerp_col((255, 225, 90), (255, 40, 140), sg ** 1.1)
        stripes = ((yy - scy) / sr * 10 + 0.0) % 1.0 > np.clip((yy - scy + sr * 0.15) / sr, 0, 0.75) * 0.9
        disc = np.clip(sr - dd + 0.5, 0, 1) * stripes
        sky = sky * (1 - disc[..., None]) + sun_col * disc[..., None] * 1.05
        glow = np.exp(-np.clip(dd - sr, 0, None) / (W * 0.10))[..., None] * np.array([255, 60, 150], np.float32) * 0.3
        self.sky = (sky + glow).astype(np.float32)
        gy = np.clip((yy - self.hy) / (H - self.hy), 0, 1)
        self.ground = (lerp_col((40, 6, 46), (3, 1, 8), gy[..., 0] ** 0.3)[:, None, :] * np.ones((1, W, 1), np.float32))
        self.cols_x = np.arange(W, dtype=np.float32)

    def frame(self, t):
        R = self.R
        vz = 6.0
        cz = t * vz
        cx = 0.9 * math.sin(t * 0.23)
        cyh = 2.2 + 0.25 * math.sin(t * 0.31)
        cam = Camera((cx, cyh, cz), (cx, cyh + math.tan(self.pitch) * 10, cz + 10), fov=62)
        R.begin(cam)
        k0 = int(math.floor(cz / 1.0)) + 1
        ks = np.arange(k0, k0 + self.rows)
        Hm = self.hmap[ks % self.Tz]  # (rows, C)
        X = np.broadcast_to(self.xs[None, :], Hm.shape)
        Z = np.broadcast_to(ks[:, None].astype(np.float32), Hm.shape)
        P = np.stack([X, Hm, Z], -1).reshape(-1, 3)
        sx, sy, sz, inv = R.project(P)
        sx, sy, sz = sx.reshape(Hm.shape), sy.reshape(Hm.shape), sz.reshape(Hm.shape)
        inv = inv.reshape(Hm.shape)
        # occlusion horizon: for each row, the min screen-y of all nearer rows (per pixel column)
        big = 1e6
        rev = sx[0, 0] > sx[0, -1]  # np.interp needs increasing x
        sxi, syi = (sx[:, ::-1], sy[:, ::-1]) if rev else (sx, sy)
        yrow = np.stack([np.interp(self.cols_x, sxi[i], syi[i], left=big, right=big) for i in range(self.rows)])
        occ = np.minimum.accumulate(yrow, axis=0)
        occ_prev = np.vstack([np.full((1, self.W), big, np.float32), occ[:-1]])
        dist = sz
        fogf = (0.25 + 0.75 * np.exp(-dist / 22.0)) * smoothstep(self.rows * 1.0, self.rows * 0.6, dist)
        hn = np.clip(Hm / (self.hmax * 0.75), 0, 1)
        col = lerp_col(self.ca, self.cb, hn ** 0.8) * (fogf * 1.9)[..., None]
        col = col + np.array([255, 255, 255], np.float32) * (hn ** 3 * fogf * 0.5)[..., None]
        wid = np.clip(0.035 * inv, 0, 2.4)
        tol = 1.2
        # row segments
        r_i, c_i = np.meshgrid(np.arange(self.rows), np.arange(self.C - 1), indexing="ij")
        r_i, c_i = r_i.ravel(), c_i.ravel()
        segs = [(sx[r_i, c_i], sy[r_i, c_i], sx[r_i, c_i + 1], sy[r_i, c_i + 1], col[r_i, c_i], col[r_i, c_i + 1],
                 wid[r_i, c_i], occ_prev, r_i)]
        r2, c2 = np.meshgrid(np.arange(self.rows - 1), np.arange(self.C), indexing="ij")
        r2, c2 = r2.ravel(), c2.ravel()
        segs.append((sx[r2, c2], sy[r2, c2], sx[r2 + 1, c2], sy[r2 + 1, c2], col[r2, c2], col[r2 + 1, c2],
                     wid[r2, c2], occ, r2))
        for x0, y0, x1, y1, c0, c1, w, occa, ri in segs:
            on = ~(((x0 < 0) & (x1 < 0)) | ((x0 > self.W) & (x1 > self.W)) | ((y0 > self.H) & (y1 > self.H)))
            x0, y0, x1, y1, c0, c1, w, ri = x0[on], y0[on], x1[on], y1[on], c0[on], c1[on], w[on], ri[on]
            xs, ys, seg, s, step = R.sample_segments(x0, y0, x1, y1, 0.9, 500)
            xi = np.clip(xs.astype(np.int64), 0, self.W - 1)
            vis = ys <= occa[ri[seg], xi] + tol
            seg, s, step = seg[vis], s[vis], step[vis]
            cc = (c0[seg] + (c1[seg] - c0[seg]) * s[:, None]) * step[:, None]
            R.points(xs[vis], ys[vis], cc, w[seg], bloom=0.5)
        sil = occ[-1]
        yy = np.arange(self.H, dtype=np.float32)[:, None]
        mask = (yy > sil[None, :])[..., None]
        base = np.where(mask, self.ground, self.sky)
        return R.finish(base=base, bloom=(0.6, 0.55), radii=(1.4, 7.0))


# ======================================================================== 7. atom
class Atom3D(Scene3D):
    """Atom: glowing nucleon cluster, three tilted electron orbits with trails, faint orbital cloud."""

    def setup(self):
        rng = self.rng
        self.ca, self.cb = _pal(self.p, "cyan")
        # pack nucleons
        pos = [np.zeros(3)]
        while len(pos) < 22:
            c = rng.normal(0, 0.22, 3)
            if np.linalg.norm(c) < 0.36 and all(np.linalg.norm(c - p) > 0.17 for p in pos):
                pos.append(c)
        self.nuc = np.array(pos, np.float32)
        self.proton = rng.random(len(pos)) < 0.5
        self.sph = fib_sphere(2200).astype(np.float32)
        self.nr = 0.115
        self.orbits = []
        for i, (rad, rx, rz, sp) in enumerate([(1.55, 0.25, 0.0, 1.3), (1.75, 1.15, 0.6, -1.05), (1.95, -1.0, -0.7, 0.9)]):
            M = (rot_y(i * 1.1) @ rot_z(rz) @ rot_x(rx)).astype(np.float32)
            self.orbits.append((rad, M, sp, rng.uniform(0, TAU)))
        th = np.linspace(0, TAU, 1200, endpoint=False)
        self.circ = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1).astype(np.float32)
        # orbital probability cloud (2p-like lobes along 3 axes + spherical 1s)
        nc = 26000
        rr = rng.gamma(3.0, 0.28, nc)
        d = random_dirs(rng, nc)
        axis = rng.integers(0, 4, nc)
        keep = np.ones(nc, bool)
        for ax in range(3):
            sel = axis == ax
            keep[sel] = rng.random(sel.sum()) < d[sel, ax] ** 2
        self.cloud = (d * rr[:, None])[keep].astype(np.float32)
        self.cloud_c = lerp_col(self.ca, self.cb, rng.random(keep.sum())) * 0.9
        self.light = _norm([-0.4, 0.7, -0.5]).astype(np.float32)
        self.make_sky(1500, bright=0.6)

    def frame(self, t):
        R, a, b = self.R, self.ca, self.cb
        cam = orbit((0, 0, 0), 5.4 - 0.4 * math.sin(t * 0.11), 0.2 + t * 0.19, 0.3 + 0.18 * math.sin(t * 0.14))
        R.begin(Camera(cam, (0, 0, 0), fov=50, roll=0.12 * math.sin(t * 0.1)))
        view = cam.astype(np.float32)
        self.draw_sky()
        R.points3d(self.cloud, self.cloud_c * 0.9, 0.006, bloom=0.2)
        # nucleus: dense point-sampled spheres, front hemisphere only, with true sphere-sphere occlusion
        jig = 0.01 * np.sin(t * 2.6 + np.arange(len(self.nuc))[:, None] * 1.7 + np.arange(3)[None, :])
        C = self.nuc + jig.astype(np.float32)
        nS = len(self.sph)
        P = (C[:, None, :] + self.sph[None, :, :] * self.nr).reshape(-1, 3)
        N = np.tile(self.sph, (len(C), 1))
        own = np.repeat(np.arange(len(C)), nS)
        V = view - P
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        fac = np.sum(N * V, 1)
        fr = fac > 0.0
        P, N, V, fac, own = P[fr], N[fr], V[fr], fac[fr], own[fr]
        oc = P[:, None, :] - C[None, :, :]
        bq = np.einsum("mjk,mk->mj", oc, V)
        cq = np.einsum("mjk,mjk->mj", oc, oc) - self.nr ** 2
        disc = bq * bq - cq
        hit = (disc > 0) & ((-bq - np.sqrt(np.maximum(disc, 0))) > 1e-3)
        vis = ~hit.any(1)
        P, N, V, fac, own = P[vis], N[vis], V[vis], fac[vis], own[vis]
        diff = np.clip(N @ self.light, 0, 1)
        Hh = V + self.light
        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
        spec = np.clip(np.sum(N * Hh, 1), 0, 1) ** 40
        cdir = C[own] / np.maximum(np.linalg.norm(C[own], axis=1, keepdims=True), 1e-3)
        ao = np.clip(0.55 + 0.6 * np.sum(N * cdir, 1), 0.25, 1.0)
        base = np.where(self.proton[own][:, None], np.array([255, 85, 55], np.float32), np.array([80, 150, 255], np.float32))
        rim = (1 - fac) ** 3
        sh = base * ((0.08 + 0.8 * diff) * ao)[:, None] + (spec * 255)[:, None] + base * (rim * 0.9)[:, None]
        _, _, zc, invc = R.project(C[own])
        area = 2 * math.pi * (self.nr * invc) ** 2 / nS  # px^2 covered by each surface sample
        R.points3d(P, sh * area[:, None] * 0.9, 0.0, 1.1, bloom=0.35)
        cx, cy, _, ci = R.project(np.zeros((1, 3)))
        R.points(np.repeat(cx, 2), np.repeat(cy, 2), np.array([[255, 140, 100], [140, 170, 255]]) * 2500,
                 [0.45 * ci[0], 0.9 * ci[0]], bloom=0)
        # orbits, electrons with trails
        dn = float(np.linalg.norm(view))
        nrm = view / dn
        for rad, M, sp, ph in self.orbits:
            P = (self.circ * rad) @ M.T
            behind = (np.sum((P - view) * -nrm, 1) > dn) & (np.linalg.norm(np.cross(P - view, nrm), axis=1) < 0.42)
            fo = np.where(behind, 0.15, 1.0)[:, None]
            R.lines3d(P, np.roll(P, -1, 0), (a * 0.30)[None, :] * fo, width=0.9, bloom=0.5)
            ang = ph + t * sp
            ntr = 140
            tr = np.linspace(0, 2.2, ntr) * np.sign(sp)
            angs = ang - tr
            S = np.stack([np.cos(angs), np.zeros_like(angs), np.sin(angs)], 1) * rad @ M.T
            fade = np.exp(-np.linspace(0, 5, ntr))
            col = lerp_col(b, (255, 255, 255), fade ** 2) * (fade * 3.0)[:, None]
            R.lines3d(S[:-1], S[1:], col[:-1], col[1:], width=0.8 + 1.8 * fade[:-1] ** 2, bloom=0.9)
            R.points3d(S[:1], np.array([[200, 240, 255]]) * 2.2, 0.0, 2.6, bloom=1.0)
            ex, ey, ez, ei = R.project(S[:1])
            if ez[0] > 0.1:
                hr = 0.12 * ei[0]
                R.points(ex, ey, (b * 0.5 + 120)[None, :] * 0.22 * math.pi * hr * hr, hr, bloom=0)
        return R.finish(bloom=(0.7, 0.6), radii=(1.5, 8.0))


# ======================================================================== 8. data city
class DataCity3D(Scene3D):
    """Low flight down a street of glowing data towers (flat-shaded boxes, painter's algorithm)."""

    def setup(self):
        rng = self.rng
        self.ca, self.cb = _pal(self.p, "cyan")
        self.P = 64  # pattern repeats every P rows -> seamless forever
        self.nx = 14
        hraw = rng.gamma(1.6, 0.9, (self.P, self.nx)) * (0.6 + 0.8 * periodic_noise((self.P, self.nx * 2), rng, 2.0)[:, :self.nx])
        self.hgt = np.clip(hraw, 0.15, 6.0).astype(np.float32)
        self.hue = rng.random((self.P, self.nx)).astype(np.float32)
        self.fp = rng.uniform(0.55, 0.85, (self.P, self.nx)).astype(np.float32)  # footprint
        self.pulse = rng.random((self.P, self.nx)).astype(np.float32)
        self.cell = 1.0
        npk = 26
        self.pk_x = rng.integers(-3, 4, npk).astype(np.float64)
        self.pk_z0 = rng.uniform(0, 30, npk)
        self.pk_v = rng.uniform(3.0, 7.0, npk)
        self.pk_h = rng.random(npk)
        W, H = self.W, self.H
        yy = np.linspace(0, 1, H, dtype=np.float32)
        self.bg = lerp_col((2, 4, 12), (30, 12, 60), np.clip(yy / 0.42, 0, 1) ** 2.2)
        self.make_sky(900, bright=0.4)

    def frame(self, t):
        R = self.R
        vz = 2.2
        cz = t * vz
        camx = 0.5 * self.cell * 0  # in the middle of a street (street at x = 0)
        cy = 1.15 + 0.2 * math.sin(t * 0.4)
        sway = 0.25 * math.sin(t * 0.27)
        cam = Camera((camx + sway, cy, cz), (camx + sway * 0.4, cy - 0.3, cz + 4), fov=64,
                     roll=0.06 * math.sin(t * 0.33))
        R.begin(cam)
        W, H = self.W, self.H
        hx, hy, _, _ = R.project(np.array([[cam.pos[0], cy, cz + 2000.0]]))
        hy = float(hy[0])
        yy = np.arange(H, dtype=np.float32)
        above = np.clip((hy - yy) / max(hy, 1.0), 0, 1)
        below = np.clip((yy - hy) / max(H - hy, 1.0), 0, 1)
        col = np.where((yy < hy)[:, None], lerp_col((46, 16, 80), (2, 3, 12), above ** 0.5),
                       lerp_col((22, 8, 40), (2, 2, 6), below ** 0.35))
        col += (np.exp(-np.abs(yy - hy) / (H * 0.025))[:, None] * np.array([90, 40, 140], np.float32))
        canvas = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)[:, None, :]).resize((W, H), Image.NEAREST)
        d = ImageDraw.Draw(canvas)
        Rm = cam.R.astype(np.float64)
        cpos = cam.pos
        near = 0.08
        f = R.f

        def proj_poly(V):
            """Clip polygon (camera space) against near plane and project; returns list of xy or None."""
            out = []
            n = len(V)
            for i in range(n):
                a_, b_ = V[i], V[(i + 1) % n]
                ina, inb = a_[2] >= near, b_[2] >= near
                if ina:
                    out.append(a_)
                if ina != inb:
                    s = (near - a_[2]) / (b_[2] - a_[2])
                    out.append(a_ + (b_ - a_) * s)
            if len(out) < 3:
                return None
            return [(W * 0.5 + p[0] * f / p[2], H * 0.5 - p[1] * f / p[2]) for p in out]

        fogc = np.array([34, 12, 58], np.float32)
        far = 34.0
        # ground grid lines (streets)
        gl = []
        k0 = int(math.floor(cz))
        for zz in range(k0, k0 + int(far)):
            gl.append(((-8.0, 0.0, zz + 0.0), (8.0, 0.0, zz + 0.0)))
        for xx in np.arange(-7.0, 7.5, 1.0):
            gl.append(((xx, 0.0, cz + near + 0.01), (xx, 0.0, cz + far)))
        for A, B in gl:
            V = (np.array([A, B]) - cpos) @ Rm.T
            if V[0, 2] < near and V[1, 2] < near:
                continue
            if V[0, 2] < near:
                V[0] = V[0] + (V[1] - V[0]) * ((near - V[0, 2]) / (V[1, 2] - V[0, 2]))
            if V[1, 2] < near:
                V[1] = V[1] + (V[0] - V[1]) * ((near - V[1, 2]) / (V[0, 2] - V[1, 2]))
            zc = float(V[:, 2].mean())
            ff = math.exp(-zc / 10.0)
            c = tuple(int(v) for v in (self.ca * 0.5 * ff + fogc * (1 - ff) * 0.3))
            pts = [(W * 0.5 + p[0] * f / p[2], H * 0.5 - p[1] * f / p[2]) for p in V]
            d.line(pts, fill=c, width=1)
        # data packets streaming along the streets towards the camera
        Lp = 30.0
        rel = Lp - ((self.pk_z0 + self.pk_v * t) % Lp) - 1.0
        for px_, rz, hue in zip(self.pk_x, rel, self.pk_h):
            if rz < near + 0.3:
                continue
            A = np.array([[px_, 0.01, cz + rz], [px_, 0.01, cz + rz + 0.7]])
            V = (A - cpos) @ Rm.T
            ff = math.exp(-max(0.0, rz - 2.0) / 9.0)
            c = (self.ca * (1 - hue) + self.cb * hue) * ff + fogc * (1 - ff) * 0.3
            pts = [(W * 0.5 + p[0] * f / p[2], H * 0.5 - p[1] * f / p[2]) for p in V]
            d.line(pts, fill=tuple(int(v) for v in np.clip(c * 1.2 + 40 * ff, 0, 255)), width=max(1, int(0.05 * f / V[0, 2])))
        # towers, far to near
        items = []
        for row in range(k0 - 1, k0 + int(far)):
            for ci in range(self.nx):
                xc = (ci - self.nx / 2 + 0.5) * self.cell
                if abs(xc) < 0.6:
                    continue  # the street we fly down
                pr = row % self.P
                h = float(self.hgt[pr, ci])
                fp = float(self.fp[pr, ci]) * 0.5
                zc = row + 0.5
                dz = zc - cz
                if dz < -1.0:
                    continue
                items.append((math.hypot(xc - cpos[0], dz), xc, zc, h, fp, pr, ci))
        items.sort(reverse=True)
        for dist, xc, zc, h, fp, pr, ci in items:
            x0, x1, z0, z1 = xc - fp, xc + fp, zc - fp, zc + fp
            corners = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1],
                                [x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1]], np.float64)
            V = (corners - cpos) @ Rm.T
            if (V[:, 2] < near).all():
                continue
            ff = math.exp(-max(0.0, dist - 2.0) / 11.0)
            hue = float(self.hue[pr, ci])
            emis = self.ca * (1 - hue) + self.cb * hue
            pul = 0.5 + 0.5 * math.sin(t * 2.0 + self.pulse[pr, ci] * TAU)
            body = np.array([5, 7, 16], np.float32)
            faces = []
            if cpos[2] < z0:
                faces.append(([0, 1, 5, 4], body * 1.6 + emis * 0.05))  # front (-z)
            if cpos[0] < x0:
                faces.append(([3, 0, 4, 7], body * 1.0 + emis * 0.03))  # left side
            elif cpos[0] > x1:
                faces.append(([1, 2, 6, 5], body * 1.0 + emis * 0.03))  # right side
            if cpos[1] > h:
                faces.append(([4, 5, 6, 7], emis * (0.55 + 0.35 * pul)))  # emissive top
            for idx, c in faces:
                pts = proj_poly(V[idx])
                if pts is None:
                    continue
                cc = c * ff + fogc * (1 - ff) * 0.55
                edge = emis * (0.5 + 0.5 * pul) * ff + fogc * (1 - ff) * 0.6
                d.polygon(pts, fill=tuple(int(v) for v in np.clip(cc, 0, 255)),
                          outline=tuple(int(v) for v in np.clip(edge, 0, 255)))
            # emissive window bands on the visible vertical faces (near towers only)
            if dist < 14 and h > 0.8:
                nb = int(h / 0.35)
                for j in range(1, nb):
                    yb = j * 0.35
                    lit = ((j * 7 + int(self.pulse[pr, ci] * 13) + int(t * 1.5)) % 5) != 0
                    if not lit:
                        continue
                    segs = []
                    if cpos[2] < z0:
                        segs.append(((x0 + 0.04, yb, z0), (x1 - 0.04, yb, z0)))
                    if cpos[0] < x0:
                        segs.append(((x0, yb, z1 - 0.04), (x0, yb, z0 + 0.04)))
                    elif cpos[0] > x1:
                        segs.append(((x1, yb, z0 + 0.04), (x1, yb, z1 - 0.04)))
                    for A, B in segs:
                        Vs = (np.array([A, B]) - cpos) @ Rm.T
                        if (Vs[:, 2] < near).any():
                            continue
                        pts = [(W * 0.5 + p[0] * f / p[2], H * 0.5 - p[1] * f / p[2]) for p in Vs]
                        cc = emis * 0.85 * ff + fogc * (1 - ff) * 0.4
                        d.line(pts, fill=tuple(int(v) for v in cc), width=max(1, int(0.03 * f / max(0.3, Vs[0, 2]))))
        # stars above the skyline (additive, under the opaque canvas is fine: sky is dark)
        self.draw_sky()
        return R.finish(canvas=canvas, bloom=(0.7, 0.6), radii=(1.5, 8.0))


# ======================================================================== 9. ringed planet
class RingPlanet3D(Scene3D):
    """Banded gas giant with a ring system: per-pixel ray-cast planet + ring plane (ring gaps, planet
    shadow on the rings, ring shadow on the planet, rings in front partially covering the planet)
    plus a 3D particle layer on the rings for sparkle/parallax, all correctly occluded."""

    def setup(self):
        rng = self.rng
        self.ring_in, self.ring_out = 1.3, 2.45
        rr = np.linspace(self.ring_in, self.ring_out, 1200)
        prof = 0.3 + 0.7 * periodic_noise((1200,), rng, beta=1.1)
        prof *= smoothstep(1.3, 1.36, rr) * smoothstep(2.45, 2.4, rr)
        prof *= 0.45 + 0.55 * smoothstep(1.3, 1.55, rr)  # faint inner C-ring
        prof *= 1 - 0.95 * np.exp(-((rr - 1.95) / 0.03) ** 2)  # Cassini-like division
        prof *= 1 - 0.7 * np.exp(-((rr - 2.27) / 0.01) ** 2)  # Encke-like gap
        self.prof_r, self.prof = rr.astype(np.float32), (prof / prof.max()).astype(np.float32)
        tone = periodic_noise((1200,), rng, beta=1.0)
        self.ring_lut = lerp_col((150, 118, 92), (240, 222, 190), tone).astype(np.float32)
        # sparkle particles
        n = int(self.p.get("particles", 50000) * min(1.0, self.k * 1.2 + 0.05))
        pr = self.prof / self.prof.sum()
        r = rng.choice(rr, n, p=pr) + rng.uniform(-0.001, 0.001, n)
        self.r = r.astype(np.float32)
        self.th = rng.uniform(0, TAU, n).astype(np.float32)
        self.yy = rng.normal(0, 0.006, n).astype(np.float32)
        self.rcol = (lerp_col((170, 140, 110), (255, 240, 210), rng.random(n)) * (0.4 + rng.random(n) ** 3 * 1.5)[:, None]).astype(np.float32)
        self.omega = (0.12 * r ** -1.5).astype(np.float32)
        # planet texture (latitude bands + turbulence), tiles in longitude
        tw, th_ = 1024, 512
        lat = np.linspace(-1, 1, th_)[:, None]
        turb = periodic_noise((th_, tw), rng, beta=2.0)
        bands = np.sin(lat * 23 + turb * 2.2) * 0.5 + 0.5
        bands2 = np.sin(lat * 61 + turb * 4.0) * 0.5 + 0.5
        v = np.clip(bands * 0.65 + bands2 * 0.25 + turb * 0.3, 0, 1.2) / 1.2
        pal = self.p.get("planet", "saturn")
        stops = {"saturn": [(0, (95, 60, 40)), (0.35, (190, 140, 95)), (0.6, (235, 205, 160)), (0.8, (200, 160, 120)), (1, (250, 235, 205))],
                 "ice": [(0, (30, 60, 110)), (0.35, (70, 120, 180)), (0.6, (140, 190, 230)), (0.8, (100, 150, 210)), (1, (220, 240, 255))]}
        self.tex = colorize3(v, stops.get(pal, stops["saturn"])).reshape(-1, 3).astype(np.float32)
        self.tw, self.th_ = tw, th_
        self.tilt = (rot_z(0.25) @ rot_x(0.05)).astype(np.float64)  # planet/ring frame -> world
        self.sun = _norm([-0.95, 0.45, 0.35])
        self.make_sky(3000, bright=0.9)

    def _surface(self, cam, sun_l, cam_l, t):
        """Per-pixel planet + ring plane at half resolution inside their screen bbox."""
        R = self.R
        T = self.tilt
        # bbox from the projected ring rim and the planet limb
        th = np.linspace(0, TAU, 72, endpoint=False)
        rim = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1) * self.ring_out
        pts = np.concatenate([rim, fib_sphere(64) * 1.05]) @ T.T
        x, y, z, _ = R.project(pts)
        if (z <= 0.05).any():
            x0, x1, y0, y1 = 0, self.W, 0, self.H
        else:
            x0, x1 = int(max(0, x.min() - 4)), int(min(self.W, x.max() + 4))
            y0, y1 = int(max(0, y.min() - 4)), int(min(self.H, y.max() + 4))
        hs = 2
        x0, y0 = x0 // hs * hs, y0 // hs * hs
        x1, y1 = min(self.W, -(-x1 // hs) * hs), min(self.H, -(-y1 // hs) * hs)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        gx = (np.arange(x0, x1, hs, dtype=np.float32) + hs * 0.5 - self.W * 0.5) / R.f
        gy = -(np.arange(y0, y1, hs, dtype=np.float32) + hs * 0.5 - self.H * 0.5) / R.f
        Tf = T.astype(np.float32)
        right, up, fwd = (cam.R[i].astype(np.float32) @ Tf for i in range(3))
        Dx = gx[None, :] * right[0] + gy[:, None] * up[0] + fwd[0]
        Dy = gx[None, :] * right[1] + gy[:, None] * up[1] + fwd[1]
        Dz = gx[None, :] * right[2] + gy[:, None] * up[2] + fwd[2]
        inv = 1.0 / np.sqrt(Dx * Dx + Dy * Dy + Dz * Dz)
        Dx *= inv
        Dy *= inv
        Dz *= inv
        o = cam_l.astype(np.float32)
        sl = sun_l.astype(np.float32)
        b = Dx * o[0] + Dy * o[1] + Dz * o[2]
        disc = b * b - (float(o @ o) - 1.0)
        pxw = R.f / float(np.linalg.norm(o)) / hs
        alpha = np.clip(disc * pxw * 0.5 + 0.5, 0, 1)
        img = np.zeros(Dx.shape + (3,), np.float32)
        hit = alpha > 0
        s_hit = np.full(Dx.shape, np.inf, np.float32)
        if hit.any():
            dxh, dyh, dzh, bh = Dx[hit], Dy[hit], Dz[hit], b[hit]
            sh = -bh - np.sqrt(np.maximum(disc[hit], 0))
            s_hit[hit] = sh
            px, py, pz = o[0] + dxh * sh, o[1] + dyh * sh, o[2] + dzh * sh
            nrm = 1.0 / np.maximum(np.sqrt(px * px + py * py + pz * pz), 1e-6)
            px, py, pz = px * nrm, py * nrm, pz * nrm
            lat = np.arcsin(np.clip(py, -1, 1))
            lon = np.arctan2(pz, px) + t * 0.05
            u = ((lon * (1 / TAU)) % 1.0 * self.tw).astype(np.int32) % self.tw
            v = ((lat * (1 / math.pi) + 0.5) * (self.th_ - 1)).astype(np.int32)
            col = self.tex[v * self.tw + u]
            lam = np.clip(px * sl[0] + py * sl[1] + pz * sl[2], 0, 1)
            lit = 0.025 + 0.975 * lam ** 0.85
            # ring shadow on the planet
            sd = -py / (sl[1] if abs(sl[1]) > 1e-4 else 1e-4)
            qx, qz = px + sd * sl[0], pz + sd * sl[2]
            dens = np.interp(np.sqrt(qx * qx + qz * qz), self.prof_r, self.prof) * (sd > 0)
            lit *= 1 - 0.7 * dens
            mu = np.clip(-(dxh * px + dyh * py + dzh * pz), 0, 1)
            lit *= 0.5 + 0.5 * np.sqrt(mu)
            rimv = np.exp(-mu / 0.07) * (0.2 + lam) * 70
            img[hit] = (col * lit[:, None] + rimv[:, None] * np.array([1.0, 0.86, 0.66], np.float32)) * alpha[hit][:, None]
        # ring plane
        dys = np.where(np.abs(Dy) > 1e-6, Dy, 1e-6)
        sr = -o[1] / dys
        qx, qz = o[0] + Dx * sr, o[2] + Dz * sr
        qr = np.sqrt(qx * qx + qz * qz)
        ring = (sr > 0) & (qr > self.ring_in) & (qr < self.ring_out)
        if ring.any():
            qrr = qr[ring]
            ri = np.clip(((qrr - self.ring_in) / (self.ring_out - self.ring_in) * (len(self.prof) - 1)).astype(np.int32), 0, len(self.prof) - 1)
            dens = self.prof[ri]
            qxa, qza = qx[ring], qz[ring]
            along = qxa * sl[0] + qza * sl[2]
            shadow = (along < 0) & ((qrr * qrr - along * along) < 1.0)
            lit_face = 1.0 if (o[1] * sl[1]) > 0 else 0.45  # unlit face of the rings is dimmer
            light = np.where(shadow, 0.06, 1.0) * lit_face
            behind = sr[ring] > s_hit[ring]  # ring behind the planet: hidden where the planet is opaque
            cover = np.where(behind, alpha[ring], 0.0)
            op = np.clip(dens * 0.92, 0, 0.92) * (1 - cover)
            base = img[ring]
            img[ring] = base * (1 - op[:, None]) + self.ring_lut[ri] * (op * light * 0.85)[:, None]
        # upsample to full res
        h, w = y1 - y0, x1 - x0
        reg = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c]), "F").resize((w, h), Image.BILINEAR))
                        for c in range(3)], -1)
        return y0, x0, reg

    def frame(self, t):
        R = self.R
        azim = 2.2 + t * 0.09
        elev = 0.08 + 0.08 * math.sin(t * 0.13)
        cam_pos = orbit((0, 0, 0), 6.4 - 0.5 * (1 - math.exp(-t / 12)), azim, elev)
        cam = Camera(cam_pos, (0, -0.1, 0), fov=50, roll=0.18)
        R.begin(cam)
        T = self.tilt
        sun_l = self.sun @ T  # sun direction in planet frame
        cam_l = cam_pos @ T
        # stars (drop those behind the planet)
        x, y, vis = R.project_dirs(self.sky_dir)
        dl = self.sky_dir @ T
        b_ = dl @ cam_l
        hit = (b_ ** 2 - (np.dot(cam_l, cam_l) - 1) > 0) & (b_ < 0)
        vis &= ~hit
        R.points(x[vis], y[vis], self.sky_col[vis], self.sky_r[vis] * self.k, bloom=0.5)
        # sparkle particles on the ring plane (hidden behind the planet / in its shadow)
        th = self.th + self.omega * t
        Pl = np.stack([self.r * np.cos(th), self.yy, self.r * np.sin(th)], 1).astype(np.float32)
        cl = cam_l.astype(np.float32)
        D = Pl - cl
        dd = np.sum(D * D, 1)
        sc = np.clip(-(D @ cl) / dd, 0, 1)
        cpt = cl + D * sc[:, None]
        occ = np.sum(cpt * cpt, 1) < 1.0
        along = Pl @ sun_l.astype(np.float32)
        shadow = (along < 0) & ((np.sum(Pl * Pl, 1) - along ** 2) < 1.0)
        keep = ~occ & ~shadow
        Pw = Pl[keep] @ T.T.astype(np.float32)
        lit_face = 1.0 if (cam_l[1] * sun_l[1]) > 0 else 0.45
        R.points3d(Pw, self.rcol[keep] * (0.6 * lit_face), 0.0, 0.0, bloom=0.4)
        base = self._surface(cam, sun_l, cam_l, t)
        return R.finish(base=base, bloom=(0.5, 0.45), radii=(1.4, 7.0))


def colorize3(v, stops):
    xs = [s[0] for s in stops]
    out = np.zeros(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, xs, [s[1][c] for s in stops])
    return out


KINDS_3D = {
    "galaxy3d": Galaxy3D, "spiral_galaxy": Galaxy3D, "milkyway": Galaxy3D, "milky_way": Galaxy3D,
    "warp3d": Warp3D, "warp": Warp3D, "hyperspace": Warp3D, "lightspeed": Warp3D,
    "globe3d": Globe3D, "globe": Globe3D, "hologlobe": Globe3D, "world": Globe3D,
    "dna3d": DNA3D, "helix": DNA3D, "genome": DNA3D,
    "neural3d": Neural3D, "neurons": Neural3D, "network3d": Neural3D,
    "terrain3d": Terrain3D, "retrowave": Terrain3D, "flyover": Terrain3D,
    "atom3d": Atom3D, "atom": Atom3D, "quantum": Atom3D, "physics": Atom3D,
    "datacity3d": DataCity3D, "datacity": DataCity3D, "server": DataCity3D, "data": DataCity3D,
    "ringplanet3d": RingPlanet3D, "saturn": RingPlanet3D, "ringplanet": RingPlanet3D,
}

# register with the procedural scene table (procedural.py imports this module at its bottom)
from . import procedural as _procedural  # noqa: E402

_procedural.KINDS.update(KINDS_3D)
