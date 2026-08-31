"""Convert ONE CED size's ``mel -> logits`` graph to CoreML fp16 (shipped) + fp32 (verification
reference). Usage: ``python convert_ced.py <tiny|mini|small|base>``.

Steps (mirrors the clap convert_*.py recipe):
  1. Load the pinned PyTorch model (SHA-verified) + build the pre-sigmoid ``CedMelToLogits``.
  2. Pre-trace faithfulness assert: ``sigmoid(wrapper(mel)) == forward(mel).logits`` (the exact
     pre-sigmoid of the UNMODIFIED forward) for every corpus mel.
  3. ``torch.jit.trace`` on a fixed ``[1, 64, 1001]`` example; post-trace assert traced==eager.
  4. ``ct.convert`` fp32 then fp16 -> ``staging/ced_<size>{,_fp32}.mlpackage`` (io names
     mel/logits, iOS17, mlprogram).
"""
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _ced_common import (N_FRAMES, N_MELS, CedMelToLogits, cos, download, load_feature_extractor,
                         load_model, mel_for_waveform, staging_dir)
from _fixtures import CORPUS, samples_f32


def build_corpus_mels(fe):
    """torchaudio mels for a few corpus clips (skip the >window 'long' clip — it isn't a single
    fixed-window graph input)."""
    mels = []
    for cid, (_gen, kind) in CORPUS.items():
        if kind == "long":
            continue
        mels.append((cid, mel_for_waveform(fe, samples_f32(cid))))
    return mels


def main(size):
    download(size)
    model = load_model(size)
    fe = load_feature_extractor(size)
    wrap = CedMelToLogits(model).eval()
    mels = build_corpus_mels(fe)

    # (2) pre-trace faithfulness: wrapper is the exact pre-sigmoid of the unmodified forward.
    print(f"[{size}] pre-trace faithfulness (sigmoid(wrapper) == forward.logits):")
    for cid, mel in mels:
        with torch.no_grad():
            pre = wrap(mel)
            post = model(mel).logits
        d = float((torch.sigmoid(pre) - post).abs().max())
        assert d <= 1e-6, f"{size}/{cid}: sigmoid(wrapper) vs forward.logits max|Δ|={d}"
        print(f"    {cid:14s} max|Δ|={d:.2e} OK")

    # (3) trace on a fixed [1,64,1001] example, assert traced == eager.
    ex = mels[0][1]
    print(f"[{size}] tracing…")
    ts = torch.jit.trace(wrap, ex, check_trace=False)
    with torch.no_grad():
        ref0 = wrap(ex).numpy().ravel()
        tr0 = ts(ex).numpy().ravel()
    c = cos(tr0, ref0)
    assert c >= 0.999999, f"{size}: traced vs eager cosine {c}"
    print(f"    traced-vs-eager cos={c:.8f} max|Δ|={float(np.abs(tr0 - ref0).max()):.2e}")

    # (4) convert fp32 then fp16.
    common = dict(minimum_deployment_target=ct.target.iOS17, convert_to="mlprogram")
    from _ced_common import MODELS
    repo, rev = MODELS[size][0], MODELS[size][1]
    for prec, tag in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
        print(f"[{size}] converting {tag}…")
        m = ct.convert(
            ts,
            inputs=[ct.TensorType(name="mel", shape=(1, N_MELS, N_FRAMES), dtype=np.float32)],
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
            compute_precision=prec, **common,
        )
        m.author = "coremlit CED conversion (conversion/ced)"
        m.short_description = (f"CED {size} mel[1,64,1001]->logits[1,527] PRE-sigmoid "
                               f"(sigmoid by the caller). Source {repo}@{rev}.")
        name = f"ced_{size}.mlpackage" if tag == "fp16" else f"ced_{size}_fp32.mlpackage"
        out = staging_dir() / name
        m.save(str(out))
        d = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_ONLY).get_spec().description
        ins = [i.name for i in d.input]
        outs = [o.name for o in d.output]
        assert ins == ["mel"] and outs == ["logits"], (ins, outs)
        print(f"    saved {out.name}  io={ins}->{outs}")
    print(f"[{size}] convert DONE")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("tiny", "mini", "small", "base"):
        raise SystemExit("usage: convert_ced.py <tiny|mini|small|base>")
    main(sys.argv[1])
