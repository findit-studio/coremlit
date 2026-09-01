"""Deterministic synthetic corpus for the ReDimNet-B5 conversion — 16 kHz mono, exactly
``WINDOW_SAMPLES`` long, regenerated bit-for-bit from a seed. NO downloaded or licensed
audio, and none committed.

What this corpus is FOR, and what it is not for. Every check in this recipe is a
CROSS-IMPLEMENTATION comparison: the same function computed by PyTorch and by CoreML on
the same input. That is a numerics question, not a speech question, so it needs inputs
that exercise the graph's dynamic range — not real speech. Realistic speech would prove
nothing extra here and would import a corpus licence into a repository whose whole
licence story is "CI downloads; it does not redistribute".

What it IS chosen to do is stress the places this graph can go wrong in fp16:

  * ``silence`` — the degenerate case for a mean-normalized log-mel front end
    (``spec_norm='mn'``) and for ASTP's ``var.clamp(min=1e-7)`` guard. If the fp16 graph
    saturates anywhere, it shows here first.
  * ``dc_offset`` — the pre-emphasis filter's null; separates a working ``PreEmphasis``
    from an identity one.
  * ``tone_*`` / ``sweep`` — single mel bins lit at a time, so a frequency-axis
    misalignment in the mel filterbank cannot hide behind a broadband average.
  * ``noise`` / ``clipped`` — full-band, full-scale, the widest activation range.
  * ``formant`` — a source-filter synthesis (impulse train through three resonators) with
    an amplitude envelope: the closest thing to speech obtainable from a seed, and the
    only clip whose embedding should look like a voice to the model at all.
"""
import numpy as np

SR = 16_000
_RNG_SEED = 20260902


def _t(n):
    return np.arange(n, dtype=np.float64) / SR


def _sine(freq, n, amp=0.5):
    return (amp * np.sin(2.0 * np.pi * freq * _t(n))).astype(np.float32)


def _silence(n):
    return np.zeros(n, dtype=np.float32)


def _dc(n, level=0.25):
    return np.full(n, level, dtype=np.float32)


def _noise(n, amp=0.1, seed=_RNG_SEED):
    return (amp * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def _sweep(n, f0=80.0, f1=7000.0, amp=0.4):
    """Exponential chirp — sweeps every mel bin exactly once."""
    t = _t(n)
    k = (f1 / f0) ** (1.0 / t[-1])
    phase = 2.0 * np.pi * f0 * (k ** t - 1.0) / np.log(k)
    return (amp * np.sin(phase)).astype(np.float32)


def _clipped(n, freq=220.0):
    return np.clip(_sine(freq, n, amp=2.0), -0.95, 0.95).astype(np.float32)


def _formant(n, f0=120.0, seed=_RNG_SEED + 1):
    """Source-filter voice-like signal: a glottal impulse train at ``f0`` driven through
    three second-order resonators (~700/1220/2600 Hz), then a slow amplitude envelope so
    the time axis is not stationary (ASTP's attention has something to prefer)."""
    rng = np.random.default_rng(seed)
    src = np.zeros(n, dtype=np.float64)
    period = SR / f0
    pos = 0.0
    while pos < n:
        src[int(pos)] = 1.0
        pos += period * (1.0 + 0.03 * rng.standard_normal())
    out = np.zeros(n, dtype=np.float64)
    for fc, bw, gain in ((700.0, 80.0, 1.0), (1220.0, 110.0, 0.55), (2600.0, 170.0, 0.25)):
        r = np.exp(-np.pi * bw / SR)
        theta = 2.0 * np.pi * fc / SR
        a1, a2 = 2.0 * r * np.cos(theta), -(r ** 2)
        y = np.zeros(n, dtype=np.float64)
        y1 = y2 = 0.0
        for i in range(n):
            y0 = src[i] + a1 * y1 + a2 * y2
            y[i] = y0
            y2, y1 = y1, y0
        out += gain * y
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.8 * _t(n))
    out *= env
    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.6 / peak
    return out.astype(np.float32)


# id -> generator over `n` samples. Every clip is exactly one fixed window.
CORPUS = {
    "silence":    _silence,
    "dc_offset":  _dc,
    "tone_220":   lambda n: _sine(220.0, n),
    "tone_3000":  lambda n: _sine(3000.0, n, amp=0.3),
    "sweep":      _sweep,
    "noise":      _noise,
    "clipped":    _clipped,
    "formant":    _formant,
}


def samples_f32(clip_id: str, n: int) -> np.ndarray:
    """The clip, exactly ``n`` samples of float32 in [-1, 1]."""
    x = CORPUS[clip_id](n)
    if x.shape != (n,):
        raise SystemExit(f"{clip_id}: generated {x.shape}, expected ({n},)")
    return np.ascontiguousarray(x, dtype=np.float32)
