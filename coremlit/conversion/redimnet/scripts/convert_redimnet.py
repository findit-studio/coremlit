"""Convert the selected ReDimNet variant (``REDIMNET_VARIANT``: ``b5`` | ``b2`` | ``b2_ptn``)
to a CoreML fp16 graph (shipped) plus an fp32 graph (the verification reference). Usage:
``REDIMNET_VARIANT=<v> python convert_redimnet.py``.

Steps:
  1. Load the pinned, SHA-verified asset and rebuild ``ReDimNetWrap`` from its own
     ``model_config``; assert the config, a total state-dict match, a RAW tail, and that
     the caller-side ``MEL_FRONT_END`` spec describes this checkpoint's own ``MelBanks``.
  2. Pre-trace faithfulness: the ``mel -> embedding`` wrapper is the EXACT tail of the
     unmodified forward (``wrapper(model.spec(x)) == model(x)``), and its output is a raw
     vector (``||e||`` nowhere near 1) on every corpus clip — the numeric half of "there
     is no L2 to strip".
  3. ``torch.jit.trace`` at the fixed ``[1, N_MELS, N_FRAMES]`` shape; assert traced ==
     eager.
  4. ``ct.convert`` fp32 then fp16 into ``staging/``, FIXED shape (never ``RangeDim``: a
     flexible input takes the graph off the ANE, already measured on this repo's CAM++
     probe and visible in ``src/audio/lid``'s placement table).
  5. Record the producer: run id, the OBSERVED toolchain, and the source pins, so every
     later stage can refuse to describe a build it did not run under.

The graph is the unmodified forward with ONE module removed: ``self.spec``, the in-graph
mel front end, which moves to the caller. That removal is measured, not stylistic — see
``MEL_FRONT_END`` in ``_redimnet_common.py`` and ``probe_waveform_contract.py``. Nothing
else is stripped or substituted; in particular there is no L2 to remove.
"""
import datetime
import json
import sys
import time
import uuid

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import (CONTRACT, EMBED_DIM, INPUT_NAME, MEL_FRONT_END, MelToEmbedding,
                              N_FRAMES, N_MELS, OUTPUT_NAME, SOURCE_CODE_REV, SOURCE_RELEASE_TAG,
                              SOURCE_REPO, WINDOW_SAMPLES, cos, load_model, mel_for_waveform,
                              observed_toolchain, staging_dir)
from _fixtures import CORPUS, samples_f32

RUN_ID_KEY = "run_id"
# A raw embedding's norm is ~19 for B5. Anything within a whisker of 1.0 on EVERY clip
# would mean an L2 tail we failed to spot, so the assertion is on the whole corpus rather
# than on one clip that might legitimately be short.
L2_SUSPICION_BAND = (0.99, 1.01)


def main():
    toolchain = observed_toolchain()
    model, cfg, v = load_model()
    print(f"[..] variant {v.key}: {v.title} <- {v.asset} -> {v.mlmodelc}")
    wrap = MelToEmbedding.build(model)

    # (2) faithfulness + the numeric half of the raw-tail proof.
    print(f"[..] wrapper faithfulness and embedding norms over {len(CORPUS)} corpus clips:")
    mels, worst_faith = {}, 1.0
    normalized = []
    for cid in CORPUS:
        wav = samples_f32(cid, WINDOW_SAMPLES)[None, :]
        mel = mel_for_waveform(model, wav)
        mels[cid] = mel
        with torch.no_grad():
            full = model(torch.from_numpy(wav)).numpy().ravel()
            part = wrap(mel).numpy().ravel()
        if part.shape != (EMBED_DIM,):
            raise SystemExit(f"{cid}: embedding shape {part.shape}, expected ({EMBED_DIM},)")
        if tuple(mel.shape) != (1, N_MELS, N_FRAMES):
            raise SystemExit(f"{cid}: mel shape {tuple(mel.shape)}, expected "
                             f"(1, {N_MELS}, {N_FRAMES})")
        c = cos(part, full)
        worst_faith = min(worst_faith, c)
        n = float(np.linalg.norm(part))
        if L2_SUSPICION_BAND[0] <= n <= L2_SUSPICION_BAND[1]:
            normalized.append(cid)
        print(f"    {cid:12s} ||e|| = {n:8.4f}   wrapper-vs-forward cos = {c:.8f}")
    if not worst_faith >= 0.999999:
        raise SystemExit(f"wrapper is NOT the unmodified forward's tail: worst cos {worst_faith}")
    if len(normalized) == len(CORPUS):
        raise SystemExit("every clip's embedding has unit norm — this checkpoint HAS an L2 "
                         "tail; strip it before shipping (coremlit's contract is raw).")
    print("[ok] wrapper == unmodified forward on the same mel, and the output is RAW "
          "(un-normalized) — coremlit's `Embedding::normalize_from` owns L2")

    # (3) trace at the FIXED mel shape.
    ex = mels["formant"]
    print(f"[..] tracing at fixed [1, {N_MELS}, {N_FRAMES}] "
          f"({WINDOW_SAMPLES / 16000:.0f} s of 16 kHz audio)…")
    t0 = time.perf_counter()
    ts = torch.jit.trace(wrap, ex, check_trace=False)
    print(f"    traced in {time.perf_counter() - t0:.1f}s")
    with torch.no_grad():
        eager = wrap(ex).numpy().ravel()
        traced = ts(ex).numpy().ravel()
    c = cos(traced, eager)
    if not c >= 0.999999:
        raise SystemExit(f"traced vs eager cosine {c} — the trace is not the model")
    print(f"    traced-vs-eager cos={c:.8f} max|Δ|={float(np.abs(traced - eager).max()):.2e}")

    # (4) convert. FIXED shape, both precisions.
    common = dict(minimum_deployment_target=ct.target.iOS17, convert_to="mlprogram")
    for prec, tag in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
        print(f"[..] converting {tag}…")
        t0 = time.perf_counter()
        m = ct.convert(
            ts,
            inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, N_MELS, N_FRAMES),
                                  dtype=np.float32)],
            outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float32)],
            compute_precision=prec, **common,
        )
        m.author = "coremlit ReDimNet conversion (conversion/redimnet)"
        m.short_description = (
            f"{v.title.split(' (')[0]} speaker embedder. {CONTRACT} "
            f"Source {SOURCE_REPO}@{SOURCE_RELEASE_TAG}/{v.asset} (sha256 {v.asset_sha256[:16]}…), "
            f"model source @{SOURCE_CODE_REV[:12]}.")
        out = staging_dir() / (v.mlpackage if tag == "fp16" else v.mlpackage_fp32)
        m.save(str(out))
        spec = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
        ins = [(i.name, list(i.type.multiArrayType.shape)) for i in spec.description.input]
        outs = [(o.name, list(o.type.multiArrayType.shape)) for o in spec.description.output]
        if ins != [(INPUT_NAME, [1, N_MELS, N_FRAMES])]:
            raise SystemExit(f"{tag}: input contract is {ins}, expected "
                             f"[('{INPUT_NAME}', [1, {N_MELS}, {N_FRAMES}])]")
        if outs != [(OUTPUT_NAME, [1, EMBED_DIM])]:
            raise SystemExit(f"{tag}: output contract is {outs}, expected "
                             f"[('{OUTPUT_NAME}', [1, {EMBED_DIM}])]")
        print(f"    saved {out.name} in {time.perf_counter() - t0:.1f}s  io={ins} -> {outs}")

    # (5) producer record — WHICH run, WHEN, under WHICH toolchain.
    producer = {
        RUN_ID_KEY: uuid.uuid4().hex,
        "converted_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toolchain": toolchain,
        "variant": v.key,
        "bundle": v.mlmodelc,
        "title": v.title,
        "training_crop_s": v.training_crop_s,
        "published_metrics": v.published_metrics,
        "source": {
            "repo": SOURCE_REPO,
            "release_tag": SOURCE_RELEASE_TAG,
            "asset": v.asset,
            "asset_sha256": v.asset_sha256,
            "model_source_revision": SOURCE_CODE_REV,
        },
        "model_config": cfg,
        "contract": CONTRACT,
        "window_samples": WINDOW_SAMPLES,
        "mel_frames": N_FRAMES,
        "mel_front_end": MEL_FRONT_END,
    }
    v.staging_file("producer.json").write_text(json.dumps(producer, indent=2) + "\n")
    print(f"[ok] producer recorded for {v.mlmodelc} (run {producer[RUN_ID_KEY][:12]}…)")
    print("convert DONE")


if __name__ == "__main__":
    main()
