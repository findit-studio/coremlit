"""Verify the fp32 audio package before it is staged, fail-closed.

1. **The contract.** The package's I/O, as its spec states it, equals the fp16
   package's (FluidInference's, from the staged tree) and ``AUDIO_CONTRACT``:
   the names, the shapes and the data types, inputs and outputs.
2. **Every ``layer_norm`` has a ``beta``.** Core ML's fp32 CPU road adds a stray
   offset to a beta-less one (``probes/layer_norm_cpu_probe.py``).
3. **Token-level parity (G14), on both roads.** Every window of the six
   committed clips, as the kit plans and feeds it, through the package under
   Core ML on the CPU (``CPU_ONLY``) and on the GPU (``CPU_AND_GPU``, the
   kit's default for audio), against Hugging Face's own fp32 PyTorch audio path
   on the same window: each window's worst token cosine must hold
   ``TOKEN_COSINE`` on each road. The CPU road is the one a host without a
   usable GPU takes. This covers the re-implemented front end at the token
   level. The package is judged alone: never against the fp16 package, whose
   outputs are compared to R1 on their own (M5, M5f).

It runs the producer's environment (``require_producer_toolchain``) and writes
``verify.json`` beside the package, an output for people: ``stage_manifest.py``
reads no report. It re-derives the same attestation (``_attest.attest``) on the
bytes it stages.

Environment: ``EG2_SRC``, ``EG2_TESTS``, ``EG2_MODELS`` (the staged
FluidInference tree, read only) and ``EG2_STAGE``.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _attest import TOKEN_COSINE, attest, real_inputs  # noqa: E402
from _eg2_common import (  # noqa: E402
    FP32_AUDIO_PACKAGE,
    PRODUCER_FILE,
    VERIFY_FILE,
    VERIFY_ROADS,
    observed_toolchain,
    env,
    require_producer_toolchain,
    sha256_bytes,
    verify_source,
    write_json_atomic,
)


def main():
    stage = env("EG2_STAGE")
    # The report is an output for people and nothing reads it back: a stale one
    # is cleared first so that a failed run does not leave a passing report
    # beside a package it did not judge.
    try:
        os.unlink(os.path.join(stage, VERIFY_FILE))
    except FileNotFoundError:
        pass
    src, tests, models = env("EG2_SRC"), env("EG2_TESTS"), env("EG2_MODELS")
    observed = observed_toolchain()
    with open(os.path.join(stage, PRODUCER_FILE), encoding="utf-8") as f:
        producer = json.load(f)
    require_producer_toolchain(producer, observed, "verify_audio_fp32.py")
    verify_source(src)

    path = os.path.join(stage, FP32_AUDIO_PACKAGE)
    result = attest(path, **real_inputs(src, tests, models))
    for record in result["windows"]:
        print(
            f"[parity] {record['clip']}@{record['start']} {record['road']}: {record['tokens']} tokens, "
            f"worst cosine {record['worst_cosine']:.9f}",
            flush=True,
        )
    files = result["files"]
    write_json_atomic(
        os.path.join(stage, VERIFY_FILE),
        {
            "note": "an output for people; staging re-derives its own attestation and reads nothing from here",
            "package": FP32_AUDIO_PACKAGE,
            "package_digest": sha256_bytes("".join(f"{sha}  {rel}\n" for rel, sha in files).encode()),
            "contract": result["checks"]["contract"],
            "compute_units": list(VERIFY_ROADS),
            "layer_norm_operations": result["checks"]["layer_norm_operations"],
            "token_cosine_floor": TOKEN_COSINE,
            "windows": result["windows"],
            "worst_cosine": min(w["worst_cosine"] for w in result["windows"]),
            "passed": True,
        },
    )
    for road in VERIFY_ROADS:
        worst = min(w["worst_cosine"] for w in result["windows"] if w["road"] == road)
        print(f"[ok] {road}: every window holds {TOKEN_COSINE}; the worst is {worst:.9f}", flush=True)


if __name__ == "__main__":
    main()
