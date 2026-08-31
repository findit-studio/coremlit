"""Deterministic synthetic CED corpus — 16 kHz mono clips that hit clear AudioSet classes
(Sine wave 501, White noise 520, Silence 500), so parity is fully reproducible and e2e top-k is
meaningful, with NO downloaded/licensed audio. Every clip is regenerated bit-for-bit from a seed.

WAV encoding is int16 PCM; the golden generator decodes the written int16 EXACTLY as the Rust
``read_wav_16k_mono`` does (``i16 as f32 * (1/32768)``), so the committed WAV, the Rust decode,
and the oracle all see identical samples.
"""
import wave

import numpy as np

SR = 16_000
WINDOW_SAMPLES = 160_000
_I16_SCALE = np.float32(1.0 / 32768.0)   # matches Rust: 1.0 / (1 << (bits-1)) for bits=16


def _sine(freq, n, amp=0.5):
    t = np.arange(n, dtype=np.float64) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _silence(n):
    return np.zeros(n, dtype=np.float32)


def _noise(n, amp=0.1, seed=1234):
    return (amp * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def _long_sine_then_noise():
    # 15 s: 10 s of 440 Hz sine then 5 s of white noise -> two CED windows (10 s + 5 s tail).
    return np.concatenate([_sine(440.0, 160_000), _noise(80_000, amp=0.1, seed=4321)])


# id -> (generator, kind). kind: "full" (=WINDOW_SAMPLES), "sub" (<WINDOW_SAMPLES), "long" (>).
CORPUS = {
    "sine440_10s":   (lambda: _sine(440.0, 160_000),  "full"),   # -> Sine wave (501)
    "noise_10s":     (lambda: _noise(160_000),        "full"),   # -> White noise / Noise
    "sine1000_3s":   (lambda: _sine(1000.0, 48_000),  "sub"),    # tail-padding, tone
    "silence_2s":    (lambda: _silence(32_000),       "sub"),    # -> Silence (500)
    "long_15s":      (_long_sine_then_noise,          "long"),   # e2e multi-window aggregation
}

# Where each clip's WAV lives, corpus.json-relative (see tests/ced/common GoldenClip.file):
#   parity/e2e-single clips are shared via fixtures/mel/  (-> "../../mel/<id>.wav")
#   the long clip is shared via fixtures/goldens/clips/   (-> "../clips/<id>.wav")
def rel_path(clip_id: str) -> str:
    return f"../clips/{clip_id}.wav" if CORPUS[clip_id][1] == "long" else f"../../mel/{clip_id}.wav"


def samples_f32(clip_id: str) -> np.ndarray:
    return CORPUS[clip_id][0]()


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, np.float64) * 32767.0), -32768, 32767).astype("<i2")


def write_wav(path, x_f32) -> int:
    """Write ``x_f32`` as 16 kHz mono int16 PCM. Returns the sample count."""
    i16 = to_int16(x_f32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(i16.tobytes())
    return int(i16.shape[0])


def read_wav_like_rust(path) -> np.ndarray:
    """Decode a 16 kHz mono int16 WAV to f32 EXACTLY as Rust ``read_wav_16k_mono`` does."""
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == SR and r.getnchannels() == 1 and r.getsampwidth() == 2
        raw = r.readframes(r.getnframes())
    i16 = np.frombuffer(raw, dtype="<i2")
    return (i16.astype(np.float32) * _I16_SCALE).astype(np.float32)
