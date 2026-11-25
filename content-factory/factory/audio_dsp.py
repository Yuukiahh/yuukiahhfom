"""Small, fast numpy DSP toolkit shared by the music engine and the SFX designer.

Everything is vectorised (FFT filters, STFT filters with moving cutoffs, FFT
convolution reverb) or loops over coarse blocks only (compressor, limiter), so a
90 s stereo mix stays well under a second of processing.
"""
from __future__ import annotations

import numpy as np


def db(x: float) -> float:
    return 10 ** (x / 20)


def fft_len(n: int) -> int:
    """Smallest 2^a*3^b*5^c >= n (fast for pocketfft, far less padding than the next power of two)."""
    n = max(2, int(n))
    best = 1 << int(np.ceil(np.log2(n)))
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


# ---------------------------------------------------------------- filters
def _resp(kind: str, f: np.ndarray, fc: float, q: float = 0.707, order: int = 2) -> np.ndarray:
    """Magnitude response of an analogue-style filter evaluated at frequencies f."""
    r = np.maximum(f, 1e-6) / max(fc, 1e-6)
    if kind == "lowpass":
        return 1 / np.sqrt(1 + r ** (2 * order))
    if kind == "highpass":
        return 1 / np.sqrt(1 + r ** (-2 * order))
    if kind in ("hshelf", "lshelf"):  # q carries the shelf gain in dB here
        g = db(q)
        w = r ** 2 / (1 + r ** 2) if kind == "hshelf" else 1 / (1 + r ** 2)
        return 1 + (g - 1) * w
    if kind == "bandpass":  # 2nd-order resonator, unity at centre
        return 1 / np.sqrt(1 + q ** 2 * (r - 1 / r) ** 2)
    raise ValueError(kind)


def fft_filter(x: np.ndarray, sr: int, kind: str, fc: float, q: float = 0.707, order: int = 2) -> np.ndarray:
    """Zero-phase FFT filter (fine for one-shots and whole buses)."""
    n = x.shape[0]
    L = fft_len(n + min(n, sr // 2) + 64)  # zero-pad so filter tails do not wrap around
    X = np.fft.rfft(x, L, axis=0)
    h = _resp(kind, np.fft.rfftfreq(L, 1 / sr), fc, q, order)
    return np.fft.irfft(X * (h[:, None] if x.ndim == 2 else h), L, axis=0)[:n]


def fft_chain(x: np.ndarray, sr: int, specs) -> np.ndarray:
    """Several zero-phase filters in one FFT. specs: [(kind, fc[, q[, order]]), ...]"""
    n = x.shape[0]
    L = fft_len(n + min(n, sr // 2) + 64)  # zero-pad so filter tails do not wrap around
    X = np.fft.rfft(x.astype(np.float32, copy=False) if n > 1_000_000 else x, L, axis=0)
    f = np.fft.rfftfreq(L, 1 / sr)
    h = np.ones_like(f)
    for sp in specs:
        h = h * _resp(*((sp[0], f) + tuple(sp[1:])))
    h = h.astype(X.real.dtype)
    return np.fft.irfft(X * (h[:, None] if x.ndim == 2 else h), L, axis=0)[:n].astype(x.dtype, copy=False)


def tv_filter(x: np.ndarray, sr: int, kind: str, cutoff, q: float = 0.707, order: int = 2,
              frame: int = 1024) -> np.ndarray:
    """Time-varying filter via STFT overlap-add. cutoff: callable(t_seconds_array) -> Hz,
    or [(t, hz), ...] breakpoints interpolated exponentially."""
    mono = x.ndim == 1
    X2 = x[:, None] if mono else x
    n = X2.shape[0]
    hop = frame // 4
    pad = np.zeros((n + 2 * frame, X2.shape[1]))
    pad[frame:frame + n] = X2
    nfr = (pad.shape[0] - frame) // hop + 1
    idx = np.arange(frame)[None, :] + hop * np.arange(nfr)[:, None]
    win = np.sqrt(np.hanning(frame + 1)[:-1])
    t = (np.arange(nfr) * hop + frame / 2 - frame) / sr
    if callable(cutoff):
        fcs = np.asarray(cutoff(t), float)
    else:
        bp = np.asarray(cutoff, float)
        fcs = np.exp(np.interp(t, bp[:, 0], np.log(bp[:, 1])))
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    H = np.stack([_resp(kind, freqs, fc, q, order) for fc in fcs])  # (nfr, bins)
    out = np.zeros_like(pad)
    norm = np.zeros(pad.shape[0])
    for ch in range(X2.shape[1]):
        fr = pad[idx, ch] * win
        y = np.fft.irfft(np.fft.rfft(fr, axis=1) * H, frame, axis=1) * win
        # overlap-add: column block k of every frame tiles the output contiguously
        for k in range(frame // hop):
            sl = y[:, k * hop:(k + 1) * hop].ravel()
            start = k * hop
            out[start:start + sl.size, ch] += sl
    wsum = win ** 2
    for k in range(frame // hop):
        seg = np.tile(wsum[k * hop:(k + 1) * hop], nfr)
        norm[k * hop:k * hop + seg.size] += seg
    out /= np.maximum(norm, 1e-3)[:, None]
    res = out[frame:frame + n]
    return res[:, 0] if mono else res


# ---------------------------------------------------------------- envelopes / helpers
def adsr(n: int, sr: int, a: float = 0.005, d: float = 0.1, s: float = 1.0, r: float = 0.05,
         curve: float = 1.0) -> np.ndarray:
    """Attack/decay/sustain/release envelope filling n samples (release at the end)."""
    e = np.full(n, s, float)
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    na = min(na, n)
    if na:
        e[:na] = np.linspace(0, 1, na)
    nd = min(nd, max(0, n - na))
    if nd:
        e[na:na + nd] = 1 + (s - 1) * np.linspace(0, 1, nd) ** (1 / curve)
    nr = min(nr, n)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** curve
    return e


def fade(x: np.ndarray, sr: int, fin: float = 0.002, fout: float = 0.01) -> np.ndarray:
    n = x.shape[0]
    a, b = min(n, int(fin * sr)), min(n, int(fout * sr))
    if a:
        x[:a] *= np.linspace(0, 1, a)[:, None] if x.ndim == 2 else np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)[:, None] if x.ndim == 2 else np.linspace(1, 0, b)
    return x


def pan(x: np.ndarray, p: float) -> np.ndarray:
    """Equal-power pan of a mono signal, p in [-1, 1]."""
    th = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], 1) * np.sqrt(2)


def sweep_phase(f: np.ndarray, sr: int) -> np.ndarray:
    return 2 * np.pi * np.cumsum(f) / sr


def saw(phase_cycles: np.ndarray, dt: np.ndarray | float) -> np.ndarray:
    """PolyBLEP band-limited saw. phase_cycles: running phase in cycles, dt: f/sr."""
    p = phase_cycles % 1.0
    y = 2 * p - 1
    dt = np.broadcast_to(np.asarray(dt, float), p.shape)
    m = p < dt
    tt = p[m] / dt[m]
    y[m] -= tt + tt - tt * tt - 1
    m = p > 1 - dt
    tt = (p[m] - 1) / dt[m]
    y[m] -= tt * tt + tt + tt + 1
    return y


# ---------------------------------------------------------------- space
def make_ir(sr: int, decay: float, rng: np.random.Generator, predelay: float = 0.02,
            damp_hz: float = 5000, stereo: bool = True) -> np.ndarray:
    n = int(decay * sr)
    t = np.arange(n) / sr
    chans = []
    for _ in range(2 if stereo else 1):
        bright = rng.standard_normal(n) * np.exp(-t * 6.9 / (decay * 0.45))
        dark = fft_filter(rng.standard_normal(n), sr, "lowpass", damp_hz, order=1) * np.exp(-t * 6.9 / decay)
        ir = 0.5 * bright + dark
        ir[: int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))
        ir = np.concatenate([np.zeros(int(predelay * sr)), ir])
        chans.append(ir / np.sqrt((ir ** 2).sum()))
    return np.stack(chans, 1)


def convolve(x: np.ndarray, ir: np.ndarray, hp: float | None = None, sr: int = 44100,
             lp: float | None = None) -> np.ndarray:
    """Stereo FFT convolution, optional high/low-pass of the wet signal (free in freq domain)."""
    n = x.shape[0]
    L = fft_len(n + ir.shape[0])
    X = np.fft.rfft(x.astype(np.float32, copy=False), L, axis=0)
    H = np.fft.rfft(ir.astype(np.float32), L, axis=0)
    if hp or lp:
        f = np.fft.rfftfreq(L, 1 / sr)
        g = np.ones_like(f)
        if hp:
            g *= _resp("highpass", f, hp)
        if lp:
            g *= _resp("lowpass", f, lp)
        H = H * g.astype(np.float32)[:, None]
    return np.fft.irfft(X * H, L, axis=0)[:n]


def delay(x: np.ndarray, sr: int, time: float, feedback: float = 0.35, taps: int = 4,
          pingpong: bool = True) -> np.ndarray:
    """Feedback delay built from shifted copies (no per-sample loop)."""
    out = np.zeros_like(x)
    d = int(time * sr)
    g = 1.0
    mono_all = (x[:, 0] + x[:, 1]) * 0.5 if (pingpong and x.ndim == 2) else None
    for k in range(1, taps + 1):
        g *= feedback
        if k * d >= x.shape[0]:
            break
        src = x[:-k * d]
        if mono_all is not None:
            ch = (k - 1) % 2
            out[k * d:, ch] += mono_all[:-k * d] * (g * 1.4)
        else:
            out[k * d:] += src * g
    return out


# ---------------------------------------------------------------- dynamics
def _block_env(x: np.ndarray, block: int, kind: str = "peak") -> np.ndarray:
    x2 = x if x.ndim == 2 else x[:, None]
    n, ch = x2.shape
    nb = int(np.ceil(n / block))
    if nb * block != n:
        x2 = np.concatenate([x2, np.zeros((nb * block - n, ch), x2.dtype)])
    a = x2.reshape(nb, block * ch)
    if kind == "peak":
        return np.maximum(a.max(1), -a.min(1))
    return np.sqrt(np.einsum("ij,ij->i", a, a) / (block * ch))


def compress(x: np.ndarray, sr: int, threshold_db: float = -18, ratio: float = 2.0, attack: float = 0.01,
             release: float = 0.15, knee_db: float = 6, makeup_db: float = 0.0, block: int = 256) -> np.ndarray:
    """RMS glue compressor (feed-forward, soft knee)."""
    env = _block_env(x, block, "rms")
    lvl = 20 * np.log10(np.maximum(env, 1e-9))
    over = lvl - threshold_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    bt = block / sr
    ca, cr = np.exp(-bt / attack), np.exp(-bt / release)
    sm = np.empty_like(gr)
    g = 0.0
    for i, v in enumerate(gr):
        c = ca if v > g else cr
        g = c * g + (1 - c) * v
        sm[i] = g
    centers = (np.arange(len(sm)) + 0.5) * block
    gain = db(makeup_db - np.interp(np.arange(x.shape[0]), centers, sm))
    return x * (gain[:, None] if x.ndim == 2 else gain)


_K = np.arange(-8, 8) + 0.5
_HALF = np.sinc(_K) * np.kaiser(16, 6.0)
_HALF /= _HALF.sum()


def _midpoints(x: np.ndarray) -> np.ndarray:
    """Band-limited estimate of the signal halfway between samples (for true-peak detection)."""
    x2 = x if x.ndim == 2 else x[:, None]
    out = np.empty_like(x2)
    for c in range(x2.shape[1]):
        out[:, c] = np.convolve(x2[:, c], _HALF[::-1], mode="same")
    return out if x.ndim == 2 else out[:, 0]


def limit(x: np.ndarray, sr: int, ceiling: float = 0.79, release: float = 0.08, lookahead: float = 0.003,
          block: int = 64, true_peak: bool = False) -> np.ndarray:
    """Look-ahead brickwall limiter with smooth attack ramp and release, plus a final
    soft-clip safety stage. Output |x| <= ceiling."""
    pk = _block_env(x, block, "peak")
    if true_peak:  # also detect inter-sample peaks (half-sample sinc interpolation)
        pk = np.maximum(pk, _block_env(_midpoints(x), block, "peak"))
    req = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    la = max(1, int(lookahead * sr / block))
    # window min over [i-1, i+la]
    padded = np.pad(req, (1, la), constant_values=1.0)
    win = np.lib.stride_tricks.sliding_window_view(padded, la + 2).min(1)
    g = win.copy()
    step_a = 1.0 / la
    for i in range(len(g) - 2, -1, -1):  # attack ramp (backwards)
        g[i] = min(g[i], g[i + 1] + step_a * (1 - g[i + 1]) + 1e-4)
    rc = 1 - np.exp(-block / (release * sr))
    for i in range(1, len(g)):  # release (forwards)
        g[i] = min(g[i], g[i - 1] + (1 - g[i - 1]) * rc)
    centers = (np.arange(len(g)) + 0.5) * block
    gain = np.interp(np.arange(x.shape[0]), centers, g)
    y = x * (gain[:, None] if x.ndim == 2 else gain)
    # soft safety clip above 90 % of the ceiling
    k = 0.9 * ceiling
    a = np.abs(y)
    over = a > k
    y[over] = np.sign(y[over]) * (k + (ceiling - k) * np.tanh((a[over] - k) / (ceiling - k)))
    return y


# ---------------------------------------------------------------- metering
def _k_weight(f: np.ndarray) -> np.ndarray:
    """|H| of the BS.1770 K-weighting (high shelf + RLB high-pass), analogue approximation."""
    A = db(4.0) ** 0.5  # shelf: +4 dB above ~1.7 kHz
    w = f / 1681.97
    q = 0.7072
    # analogue high shelf: H(s) = A (A s^2 + sqrt(A)/Q s + 1) / (A + sqrt(A)/Q s + s^2)
    s2 = -(w ** 2)
    re_n, im_n = A * s2 + 1, np.sqrt(A) / q * w
    re_d, im_d = A + s2, np.sqrt(A) / q * w
    shelf = A * np.sqrt((re_n ** 2 + im_n ** 2) / (re_d ** 2 + im_d ** 2))
    r = np.maximum(f, 1e-6) / 38.13
    hp = r ** 2 / np.sqrt((1 - r ** 2) ** 2 + (r / 0.5003) ** 2)
    return shelf * hp


def lufs(x: np.ndarray, sr: int, fast: bool = False) -> float:
    """Integrated loudness (BS.1770-4 gating), K-weighting applied in the frequency domain.
    fast=True measures at half the sample rate (pair averaging; error ~0.1 dB on music)."""
    x2 = x if x.ndim == 2 else x[:, None]
    if fast and x2.shape[0] > 4 * sr:
        m = x2.shape[0] // 2 * 2
        x2 = (x2[0:m:2] + x2[1:m:2]) * 0.5
        sr = sr // 2
    n = x2.shape[0]
    if n < int(0.4 * sr):
        x2 = np.pad(x2, ((0, int(0.4 * sr) - n), (0, 0)))
        n = x2.shape[0]
    L = fft_len(n)
    X = np.fft.rfft(x2.astype(np.float32, copy=False), L, axis=0)
    X *= _k_weight(np.fft.rfftfreq(L, 1 / sr)).astype(np.float32)[:, None]
    y = np.fft.irfft(X, L, axis=0)[:n]
    sub = int(0.1 * sr)
    nb = n // sub
    ms = (y[: nb * sub].astype(np.float64) ** 2).reshape(nb, sub, -1).mean(1).sum(1)  # channel sum per 100 ms
    if nb < 4:
        blocks = np.array([ms.mean()])
    else:
        blocks = np.lib.stride_tricks.sliding_window_view(ms, 4).mean(1)
    lk = -0.691 + 10 * np.log10(np.maximum(blocks, 1e-12))
    g = blocks[lk > -70]
    if not len(g):
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = g[(-0.691 + 10 * np.log10(g)) > rel]
    return float(-0.691 + 10 * np.log10(g2.mean()))


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    """4x FFT-oversampled peak (dBTP estimate)."""
    x2 = x if x.ndim == 2 else x[:, None]
    n = x2.shape[0]
    X = np.fft.rfft(x2, axis=0)
    L = n * oversample
    Y = np.zeros((L // 2 + 1, x2.shape[1]), complex)
    Y[: X.shape[0]] = X
    y = np.fft.irfft(Y, L, axis=0) * oversample
    return float(20 * np.log10(max(np.abs(y).max(), 1e-12)))


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited FFT resampling (one-shots / short tracks)."""
    if sr_in == sr_out:
        return x
    n = x.shape[0]
    m = int(round(n * sr_out / sr_in))
    X = np.fft.rfft(x, axis=0)
    k = min(X.shape[0], m // 2 + 1)
    Y = np.zeros((m // 2 + 1,) + X.shape[1:], complex)
    Y[:k] = X[:k]
    return np.fft.irfft(Y, m, axis=0) * (m / n)
