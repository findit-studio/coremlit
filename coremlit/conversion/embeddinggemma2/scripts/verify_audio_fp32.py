"""Verify the fp32 audio package before it is staged, fail-closed.

1. **The contract.** The package's I/O, as its spec states it, equals the fp16
   package's (FluidInference's, from the staged tree) and ``AUDIO_CONTRACT``:
   the names, the shapes and the data types, inputs and outputs.
2. **Token-level parity (G14).** Every window of the six committed clips, as
   the kit plans and feeds it, through the package under Core ML
   (``CPU_AND_GPU``, the kit's default for audio), against Hugging Face's own
   fp32 PyTorch audio path on the same window: each window's worst token
   cosine must hold ``TOKEN_COSINE``. This covers the re-implemented front
   end at the token level. The package is judged alone: never against the
   fp16 package, whose outputs are compared to R1 on their own (M5, M5f).

It runs the producer's environment (``require_producer_toolchain``) and writes
``verify.json`` beside the package, bound to the package's digest, which
``stage_manifest.py`` requires.

Environment: ``EG2_SRC``, ``EG2_TESTS``, ``EG2_MODELS`` (the staged
FluidInference tree, read only) and ``EG2_STAGE``.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _corpus  # noqa: E402
from _eg2_common import (  # noqa: E402
    AUDIO_CONTRACT,
    FP16_AUDIO_PACKAGE,
    FP32_AUDIO_PACKAGE,
    PRODUCER_FILE,
    VERIFY_FILE,
    audio_tokens,
    digest_tree,
    env,
    observed_toolchain,
    package_contract,
    padded_window,
    require_producer_toolchain,
    sha256_bytes,
    verify_source,
    write_json_atomic,
)
from convert_audio_fp32 import cosines, load, reference_tokens  # noqa: E402
from generate_goldens import read_wav, window_plan  # noqa: E402

# Each window's worst token cosine against PyTorch fp32. An fp32 graph is
# expected to hold it with orders of magnitude to spare; a FLOAT16 conversion
# must fall under it.
TOKEN_COSINE = 0.9999


def main():
    src, tests, models, stage = env("EG2_SRC"), env("EG2_TESTS"), env("EG2_MODELS"), env("EG2_STAGE")
    observed = observed_toolchain()
    with open(os.path.join(stage, PRODUCER_FILE), encoding="utf-8") as f:
        producer = json.load(f)
    require_producer_toolchain(producer, observed, "verify_audio_fp32.py")
    verify_source(src)
    import coremltools as ct

    path = os.path.join(stage, FP32_AUDIO_PACKAGE)
    ours = package_contract(path)
    fp16 = package_contract(os.path.join(models, FP16_AUDIO_PACKAGE))
    if ours != fp16 or ours != AUDIO_CONTRACT:
        raise SystemExit(
            "the package's contract is not the fp16 package's:\n"
            f"  ours  {json.dumps(ours)}\n  fp16  {json.dumps(fp16)}\n  pinned {json.dumps(AUDIO_CONTRACT)}"
        )
    print("[ok] the contract is the fp16 package's", flush=True)

    model, extractor = load(src)
    package = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    windows, failures = [], []
    for clip_id, rel, limit in _corpus.CLIPS:
        samples = read_wav(os.path.join(tests, rel), limit)
        for start, length in window_plan(len(samples)):
            window = samples[start : start + length]
            waveform, frame_mask = padded_window(window)
            started = time.time()
            out = package.predict({"waveform": waveform, "frame_mask": frame_mask})["audio_tokens"]
            elapsed = time.time() - started
            tokens = audio_tokens(length)
            got = np.asarray(out, dtype=np.float32)[0, :tokens]
            want = reference_tokens(model, extractor, window)
            if got.shape != want.shape:
                raise SystemExit(f"{clip_id}@{start}: {got.shape} tokens against PyTorch's {want.shape}")
            worst = float(cosines(got, want).min())
            largest = float(np.abs(got - want).max())
            windows.append(
                {
                    "clip": clip_id,
                    "start": start,
                    "samples": length,
                    "tokens": tokens,
                    "worst_cosine": worst,
                    "max_abs": largest,
                    "seconds": round(elapsed, 3),
                }
            )
            print(f"[parity] {clip_id}@{start}: {tokens} tokens, worst cosine {worst:.9f}, max |d| {largest:.3e}", flush=True)
            if worst < TOKEN_COSINE:
                failures.append(f"{clip_id}@{start}: worst token cosine {worst} under {TOKEN_COSINE}")
    if failures:
        raise SystemExit("token-level parity failed:\n  " + "\n  ".join(failures))

    files = digest_tree(path)
    write_json_atomic(
        os.path.join(stage, VERIFY_FILE),
        {
            "package": FP32_AUDIO_PACKAGE,
            "package_digest": sha256_bytes("".join(f"{sha}  {rel}\n" for rel, sha in files).encode()),
            "contract": ours,
            "compute_units": "CPU_AND_GPU",
            "token_cosine_floor": TOKEN_COSINE,
            "windows": windows,
            "worst_cosine": min(w["worst_cosine"] for w in windows),
            "passed": True,
        },
    )
    print(f"[ok] {len(windows)} windows hold {TOKEN_COSINE}; the worst is {min(w['worst_cosine'] for w in windows):.9f}", flush=True)


if __name__ == "__main__":
    main()
