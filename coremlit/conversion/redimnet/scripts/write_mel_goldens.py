"""Emit the committed mel goldens the Rust door's front end is checked against.

The Rust door reproduces ``MEL_FRONT_END`` parameter for parameter, and a wrong parameter
there is silently wrong EMBEDDINGS with no shape error to catch it (``assert_front_end``
makes the same argument for the recipe side). So the door needs an oracle it can be
compared against **with no CoreML model present**, and this is what writes it.

The oracle is the checkpoint's OWN ``MelBanks`` — ``mel_for_waveform``, i.e. the very
module ``assert_front_end`` validated — not a reimplementation of it. Each clip is written
BOTH ways:

  * ``<clip>.wav``      16 kHz mono signed-16-bit PCM, exactly ``WINDOW_SAMPLES`` frames;
  * ``<clip>_mel.npy``  float32 ``[N_MELS, N_FRAMES]``, the oracle's output for the
                        DEQUANTIZED contents of that wav.

Quantizing first and computing the golden from the quantized samples is what makes the
pair exact: the Rust test reads the same 16-bit integers, scales them by the same 1/32768,
and must land on the same mel. Nothing about the fixture depends on Python and Rust
agreeing about ``sin``.

``provenance.json`` records what produced them, and the one residual a reader has to know
about — see RESIDUAL_NOTE below.

Not part of ``run_redimnet.sh``: this writes into the crate's committed test fixtures, so
it is run deliberately, by hand, when the front end changes.

    REDIMNET_CONV=... REDIMNET_PY=... python scripts/write_mel_goldens.py
"""
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fixtures as fixtures                                     # noqa: E402
import _redimnet_common as common                                # noqa: E402

# The committed corpus, and why each one is here. Every entry is a clip of
# `_fixtures.CORPUS`, so it regenerates bit-for-bit from the same seed; the wav is
# committed anyway, because a fixture whose input is REGENERATED rather than stored
# would silently depend on numpy and Rust rounding `sin` the same way.
#
# Three is enough, and it is chosen rather than defaulted: `tone_220` alone separates the
# oracle from EVERY front-end mutation the door's mutation table lists (its closest miss
# is the log epsilon at 0.25, against a Rust-vs-oracle residual four orders of magnitude
# below that). The other two exist so the gate is not resting on one clip's geometry.
#
# `sweep` is deliberately NOT here. It is the widest-dynamic-range clip in the corpus and
# would be the natural stressor, but its mel is where the checkpoint's stored fp32 window
# buffer (see RESIDUAL_NOTE) is amplified hardest — a 6e-8 window perturbation moves its
# mel by 2.3e-4, two orders above any other clip — so committing it would buy dynamic
# range at the price of a tolerance loose enough to wave a real defect through.
GOLDEN_CLIPS = {
    "tone_220": "a single mel bin lit, everything else at the log epsilon's floor: the "
                "clip that separates the oracle from every mutation in the door's table, "
                "and the only one where the epsilon itself is visible",
    "clipped":  "a hard-clipped full-scale square-ish wave: the widest activation range "
                "in the committed set, and the largest mel value (~9.0)",
    "formant":  "a source-filter synthesis with a moving amplitude envelope — broadband, "
                "non-stationary, and the closest thing to a voice obtainable from a seed",
}

# The checkpoint's `spectrogram.window` is a SAVED fp32 buffer, and it is not bit-equal to
# a freshly computed `torch.hamming_window(400, periodic=True)` (6 of its 400 taps differ
# by one fp32 ULP), nor to the exact analytic window (up to 2.3e-7, torch's own fp32
# rounding: it evaluates 0.54f - 0.46f*cos in fp32, so w[0] is 0.08000001 rather than
# 0.08). `assert_front_end` compares them at atol 1e-6 and passes.
#
# The Rust door computes the analytic window in f64, which is MORE accurate than either.
# So there is an irreducible Rust-vs-oracle residual, it is dominated by this window and
# by the oracle's fp32 STFT, and it is stated rather than tuned away — the Rust test pins
# the measured number and says so.
RESIDUAL_NOTE = (
    "The oracle computes the whole front end in fp32 and uses the checkpoint's SAVED fp32 "
    "hamming window, whose taps sit up to 2.3e-7 from the exact analytic window (torch "
    "evaluates 0.54f - 0.46f*cos in fp32). The Rust door computes the window and the STFT "
    "in f64. The residual between them is therefore not zero and cannot be made zero "
    "without making the Rust port less accurate; the door's test pins the MEASURED value "
    "with margin instead of choosing a threshold in advance."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantize_i16(x: np.ndarray) -> np.ndarray:
    """f32 in [-1, 1] -> the signed 16-bit samples a 16-bit PCM wav would store."""
    return np.clip(np.round(x.astype(np.float64) * 32768.0), -32768.0, 32767.0).astype("<i2")


def write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(common.SAMPLE_RATE)
        fh.writeframes(pcm.tobytes())


def write_npy(path: Path, array: np.ndarray) -> None:
    np.save(str(path), np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)


def out_dir() -> Path:
    d = Path(common._env(
        "REDIMNET_GOLDENS_OUT",
        str(common.repo_root() / "coremlit" / "tests" / "identity" / "fixtures" / "mel"),
    ))
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    toolchain = common.observed_toolchain()
    model, _cfg = common.load_model()
    dest = out_dir()

    import torch

    # The traced sub-forward the graph IS: `mel -> embedding`, the exact tail of the
    # unmodified `ReDimNetWrap.forward` with only `self.spec` removed — the same factory
    # `convert_redimnet.py` traces, so the reference below is the function that was
    # converted rather than a restatement of it.
    sub_forward = common.MelToEmbedding.build(model)

    # The two tables the front end is built from, taken from the checkpoint's OWN loaded
    # buffers rather than recomputed. `mel_scale.fb` is byte-identical to a freshly built
    # one; `spectrogram.window` is NOT (see RESIDUAL_NOTE), and the one the model actually
    # uses is the one worth pinning.
    mel_transform = model.spec.torchfbank[2]
    window = mel_transform.spectrogram.window.detach().numpy()
    if window.shape != (common.MEL_FRONT_END["stft"]["win_length"],):
        raise SystemExit(f"window is {window.shape}, expected "
                         f"({common.MEL_FRONT_END['stft']['win_length']},)")
    write_npy(dest / "window.npy", window)
    # torchaudio stores the filterbank [n_freqs, n_mels]; transpose to the mel-major
    # [n_mels, n_freqs] the Rust port keeps its rows in.
    fbank = mel_transform.mel_scale.fb.detach().numpy().T
    n_freq = common.MEL_FRONT_END["stft"]["n_fft"] // 2 + 1
    if fbank.shape != (common.N_MELS, n_freq):
        raise SystemExit(f"filterbank is {fbank.shape}, expected ({common.N_MELS}, {n_freq})")
    write_npy(dest / "filterbank.npy", fbank)
    print(f"[ok] window.npy {window.shape} + filterbank.npy {fbank.shape}")

    entries = []
    for clip_id, why in GOLDEN_CLIPS.items():
        raw = fixtures.samples_f32(clip_id, common.WINDOW_SAMPLES)
        pcm = quantize_i16(raw)
        wav_path = dest / f"{clip_id}.wav"
        write_wav(wav_path, pcm)

        # Read the wav BACK, so the golden is computed from the bytes that were committed
        # rather than from what they were meant to be.
        with wave.open(str(wav_path), "rb") as fh:
            if (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) != (
                1, 2, common.SAMPLE_RATE
            ):
                raise SystemExit(f"{wav_path}: not 16 kHz mono 16-bit after writing")
            frames = fh.readframes(fh.getnframes())
        back = np.frombuffer(frames, dtype="<i2")
        if back.shape != (common.WINDOW_SAMPLES,):
            raise SystemExit(f"{wav_path}: {back.shape[0]} frames, expected "
                             f"{common.WINDOW_SAMPLES}")
        if not np.array_equal(back, pcm):
            raise SystemExit(f"{wav_path}: read-back samples differ from what was written")
        samples = (back.astype(np.float32) / 32768.0).astype(np.float32)

        mel = common.mel_for_waveform(model, torch.as_tensor(samples)[None, :]).numpy()
        if mel.shape != (1, common.N_MELS, common.N_FRAMES):
            raise SystemExit(f"{clip_id}: oracle emitted {mel.shape}, expected "
                             f"(1, {common.N_MELS}, {common.N_FRAMES})")
        if not np.all(np.isfinite(mel)):
            raise SystemExit(f"{clip_id}: oracle emitted a non-finite mel")
        mel = mel[0]

        npy_path = dest / f"{clip_id}_mel.npy"
        write_npy(npy_path, mel)

        # The PyTorch fp32 reference embedding for that same mel, from the EXACT
        # sub-forward the graph was traced from. This is what turns the door's gates from
        # "the mel is right" into "the whole chain is right": the Rust front end plus the
        # fp16 CoreML graph, against the function the conversion was verified against, in
        # one comparison. RAW — the norm is recorded so the no-L2 claim is visible here too.
        with torch.no_grad():
            emb = sub_forward(torch.as_tensor(mel)[None, :, :]).numpy()[0]
        if emb.shape != (common.EMBED_DIM,):
            raise SystemExit(f"{clip_id}: reference embedding is {emb.shape}, expected "
                             f"({common.EMBED_DIM},)")
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(emb).all() or norm == 0.0:
            raise SystemExit(f"{clip_id}: reference embedding is degenerate")

        entries.append({
            "id": clip_id,
            "why": why,
            "wav": wav_path.name,
            "wav_sha256": sha256_bytes(wav_path.read_bytes()),
            "mel": npy_path.name,
            "mel_sha256": sha256_bytes(npy_path.read_bytes()),
            "mel_min": float(mel.min()),
            "mel_max": float(mel.max()),
            "embedding_l2_norm": norm,
            "embedding": [float(v) for v in emb],
        })
        print(f"[ok] {clip_id}: {wav_path.name} + {npy_path.name} "
              f"range [{mel.min():+.4f}, {mel.max():+.4f}]  ||e|| {norm:.4f}")

    provenance = {
        "what": "Mel goldens for coremlit's `audio::identity` Rust front end.",
        "oracle": "conversion/redimnet/scripts/_redimnet_common.py::mel_for_waveform — the "
                  "checkpoint's own MelBanks (`model.spec`), the module assert_front_end "
                  "validates against MEL_FRONT_END.",
        "generator": "conversion/redimnet/scripts/write_mel_goldens.py",
        "contract": common.CONTRACT,
        "front_end": common.MEL_FRONT_END,
        "sample_rate_hz": common.SAMPLE_RATE,
        "window_samples": common.WINDOW_SAMPLES,
        "n_mels": common.N_MELS,
        "n_frames": common.N_FRAMES,
        "mel_layout": "float32 [n_mels, n_frames], row-major — freq-major, the graph's "
                      "`mel [1, 72, 401]` input with the batch axis dropped",
        "wav_format": "16 kHz mono signed 16-bit PCM; the Rust side scales by 1/32768",
        "checkpoint": {
            "asset": common.ASSET_NAME,
            "sha256": common.ASSET_SHA256,
            "model_source_rev": common.SOURCE_CODE_REV,
        },
        "toolchain": toolchain,
        "residual_note": RESIDUAL_NOTE,
        "embedding_note": (
            "Each clip's `embedding` is the PyTorch fp32 reference for that clip's `mel`, "
            "from `MelToEmbedding.build(model)` — the sub-forward `convert_redimnet.py` "
            "traces. RAW and un-normalized (see `embedding_l2_norm`), because the "
            "checkpoint's tail is ASTP -> BatchNorm1d -> Linear with no L2 to strip. A "
            "Rust gate comparing the door's end-to-end output against it measures the "
            "Rust front end and the fp16 graph together, at the same >= 0.99 cosine floor "
            "the recipe's own fp16 parity check uses."
        ),
        "tables": {
            "window.npy": {
                "what": "float32 [win_length] — the checkpoint's OWN loaded "
                        "`spectrogram.window` buffer, i.e. hamming_window(400, "
                        "periodic=True) as the model actually holds it. NOT zero-padded to "
                        "n_fft: the Rust port does that padding itself, and where the 400 "
                        "taps sit inside the 512-point frame is a separate assertion.",
                "sha256": sha256_bytes((dest / "window.npy").read_bytes()),
            },
            "filterbank.npy": {
                "what": "float32 [n_mels, n_freqs] — the checkpoint's OWN loaded "
                        "`mel_scale.fb` buffer, transposed to mel-major. Pins the htk mel "
                        "scale, norm=None, f_min/f_max and the frequency grid in one "
                        "comparison.",
                "sha256": sha256_bytes((dest / "filterbank.npy").read_bytes()),
            },
        },
        "clips": entries,
    }
    prov_path = dest / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=False) + "\n")
    print(f"[ok] wrote {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
