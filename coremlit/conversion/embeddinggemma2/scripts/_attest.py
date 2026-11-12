"""One attestation of the fp32 audio package, in one process, over one set of bytes.

Staging trusts no stored report. ``stage_manifest.py`` and ``verify_audio_fp32.py``
both call ``attest`` on the package they hold, and ``stage_manifest.py`` writes
``CHECKSUMS.sha256`` from the digests ``attest`` returns: the digests of the
bytes the checks and the parity ran on.

``attest`` digests the package, runs the checks (the contract, an explicit
``beta`` on every ``layer_norm``), loads one predictor per Core ML road
(``VERIFY_ROADS``: the CPU road and the GPU road), judges every window on every
road against the reference at the floor, and digests the package again: bytes
that changed while it was being attested are refused. A run with no window is
refused, so it cannot hold vacuously.

Everything heavy is injected (the windows, the reference, the predictors, the
checks), so the unit tests drive the real logic with fakes. ``real_inputs``
wires the producer's environment.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from _eg2_common import (  # noqa: E402
    FP16_AUDIO_PACKAGE,
    AUDIO_CONTRACT,
    VERIFY_ROADS,
    audio_tokens,
    digest_tree,
    package_contract,
    padded_window,
    require_explicit_beta,
    worst_token_cosine,
)

# Each window's worst token cosine against PyTorch fp32. An fp32 graph is
# expected to hold it with orders of magnitude to spare; a FLOAT16 conversion
# must fall under it.
TOKEN_COSINE = 0.9999


def attest(path, *, windows, build_predictors, reference, checks, floor=TOKEN_COSINE, digest=digest_tree):
    """Attests the package at ``path``; returns ``{"files", "checks", "windows"}``.

    ``windows`` yields mappings with ``clip``, ``start``, ``samples`` and
    ``tokens``; ``reference(window)`` is the expected tokens; ``checks`` is a
    sequence of ``(name, callable(path))`` whose results are recorded;
    ``build_predictors(path)`` returns ``{road: callable(window) -> tokens}``
    for exactly ``VERIFY_ROADS``. Raises ``SystemExit`` naming what failed.
    """
    before = digest(path)
    recorded = {name: check(path) for name, check in checks}
    predictors = build_predictors(path)
    if tuple(predictors) != VERIFY_ROADS:
        raise SystemExit(f"the predictors are for {tuple(predictors)}, not {VERIFY_ROADS}")
    records, failures = [], []
    for window in windows:
        want = reference(window)
        for road in VERIFY_ROADS:
            started = time.time()
            got = predictors[road](window)
            elapsed = time.time() - started
            if getattr(got, "shape", None) != getattr(want, "shape", None):
                raise SystemExit(
                    f"{window['clip']}@{window['start']} ({road}): {getattr(got, 'shape', None)} tokens "
                    f"against the reference's {getattr(want, 'shape', None)}"
                )
            worst = worst_token_cosine(got, want, f"{window['clip']}@{window['start']} ({road})")
            records.append(
                {
                    "clip": window["clip"],
                    "start": window["start"],
                    "samples": window["samples"],
                    "road": road,
                    "tokens": window["tokens"],
                    "worst_cosine": worst,
                    "seconds": round(elapsed, 3),
                }
            )
            if worst < floor:
                failures.append(f"{window['clip']}@{window['start']} ({road}): worst token cosine {worst} under {floor}")
    if failures:
        raise SystemExit("token-level parity failed:\n  " + "\n  ".join(failures))
    if not records:
        raise SystemExit("no window was judged: the parity check would hold vacuously")
    if digest(path) != before:
        raise SystemExit("the package's bytes changed while it was being attested")
    return {"files": before, "checks": recorded, "floor": floor, "windows": records}


def real_inputs(src, tests, models):
    """The producer's environment: the corpus windows, Hugging Face's PyTorch
    tokens as the reference, Core ML predictors on both roads, and the checks."""
    import numpy as np

    import _corpus
    from convert_audio_fp32 import load, reference_tokens
    from generate_goldens import read_wav, window_plan

    model, extractor = load(src)

    def windows():
        for clip_id, rel, limit in _corpus.CLIPS:
            samples = read_wav(os.path.join(tests, rel), limit)
            for start, length in window_plan(len(samples)):
                window = samples[start : start + length]
                waveform, frame_mask = padded_window(window)
                yield {
                    "clip": clip_id,
                    "start": start,
                    "samples": length,
                    "tokens": audio_tokens(length),
                    "window": window,
                    "waveform": waveform,
                    "frame_mask": frame_mask,
                }

    def reference(window):
        return reference_tokens(model, extractor, window["window"])

    def build_predictors(path):
        import coremltools as ct

        def predictor(road):
            loaded = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, road))

            def predict(window):
                out = loaded.predict({"waveform": window["waveform"], "frame_mask": window["frame_mask"]})
                return np.asarray(out["audio_tokens"], dtype=np.float32)[0, : window["tokens"]]

            return predict

        return {road: predictor(road) for road in VERIFY_ROADS}

    def contract(path):
        import json

        ours = package_contract(path)
        fp16 = package_contract(os.path.join(models, FP16_AUDIO_PACKAGE))
        if ours != fp16 or ours != AUDIO_CONTRACT:
            raise SystemExit(
                "the package's contract is not the fp16 package's:\n"
                f"  ours  {json.dumps(ours)}\n  fp16  {json.dumps(fp16)}\n  pinned {json.dumps(AUDIO_CONTRACT)}"
            )
        return ours

    return {
        "windows": windows(),
        "build_predictors": build_predictors,
        "reference": reference,
        "checks": [("contract", contract), ("layer_norm_operations", require_explicit_beta)],
    }
