"""Write the committed CED corpus + per-size goldens.

  * Shared WAVs (written once): the <=window parity/e2e-single clips -> tests/ced/fixtures/mel/,
    the >window long clip -> tests/ced/fixtures/goldens/clips/.
  * Per size: tests/ced/fixtures/goldens/<size>/corpus.json = OracleProvenance (PyTorch fp32
    ``model.safetensors``) + each <=window clip's [527] PRE-sigmoid ``CedMelToLogits`` logits,
    computed from the clip DECODED EXACTLY as Rust does (int16 / 32768), zero-padded to the fixed
    window. The long clip is NOT in corpus.json (raw_scores rejects >window); e2e loads it
    directly and pins its own measured aggregate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _ced_common import (MODELS, SIZES, CedMelToLogits, load_feature_extractor, load_model,
                         mel_for_waveform, sha256_file, src_dir)
from _fixtures import CORPUS, read_wav_like_rust, samples_f32, write_wav

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "ced" / "fixtures"


def write_shared_wavs():
    (FIXTURES / "mel").mkdir(parents=True, exist_ok=True)
    (FIXTURES / "goldens" / "clips").mkdir(parents=True, exist_ok=True)
    counts = {}
    for cid, (_gen, kind) in CORPUS.items():
        dst = (FIXTURES / "goldens" / "clips" / f"{cid}.wav") if kind == "long" \
            else (FIXTURES / "mel" / f"{cid}.wav")
        counts[cid] = write_wav(dst, samples_f32(cid))
        print(f"  wrote {dst.relative_to(FIXTURES)}  ({counts[cid]} samples, kind={kind})")
    return counts


def golden_logits(size):
    """PyTorch fp32 pre-sigmoid [527] per <=window clip, from the Rust-decoded WAV."""
    model = load_model(size)
    fe = load_feature_extractor(size)
    wrap = CedMelToLogits(model).eval()
    out = []
    for cid, (_gen, kind) in CORPUS.items():
        if kind == "long":
            continue
        wav = read_wav_like_rust(FIXTURES / "mel" / f"{cid}.wav")  # decode == Rust
        mel = mel_for_waveform(fe, wav)                            # zero-pad to window + mel
        with torch.no_grad():
            logits = wrap(mel).numpy().ravel()
        out.append((cid, int(wav.shape[0]), logits))
    return out


def main():
    print("writing shared WAVs…")
    counts = write_shared_wavs()
    for size in SIZES:
        repo, rev, st_sha, _onnx_sha, _e, _h = MODELS[size]
        gl = golden_logits(size)
        corpus = {
            "oracle": {
                "repo": repo,
                "revision": rev,
                "file": "model.safetensors",
                "sha256": sha256_file(src_dir(size) / "model.safetensors"),
            },
            "clips": [
                {
                    "id": cid,
                    "file": f"../../mel/{cid}.wav",
                    "n_samples": n,
                    "logits": [float(np.float32(v)) for v in logits],
                }
                for cid, n, logits in gl
            ],
        }
        dst = FIXTURES / "goldens" / size / "corpus.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(corpus, indent=2) + "\n")
        print(f"  {size}: {len(gl)} clips -> {dst.relative_to(FIXTURES)}")
    print("goldens DONE")


if __name__ == "__main__":
    main()
