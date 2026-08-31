"""Convert the granite text encoder -> CoreML.

Contract:
  inputs : input_ids      int32 [1, 512]
           attention_mask int32 [1, 512]
  output : embedding      fp32  [1, 384]   (CLS, PRE-L2-norm; the caller normalizes)

Order of operations, each fail-closed:

  1. faithfulness — the ``GraniteGraph`` driver vs the UNMODIFIED canonical
     sentence-transformers pipeline over all 16 corpus entries, floor
     ``DRIVER_FLOOR``. Nothing is traced until the wrapper is proven equivalent
     to the stock model.
  2. trace — ``torch.jit.trace`` at the fixed 512 window, then traced-vs-eager
     over every entry at the same floor (a trace that specialized on the example
     would show up here).
  3. convert — fp16 (shipped) and fp32 (verification reference) mlpackages.

fp16 is the shipped precision: the graph's only guard class is ``layer_norm``
with epsilon 1e-5, which is ~2^-17 and far above the 2^-24 fp16 vanishing
threshold; there is no rsqrt/real_div/log tail, because the L2 norm stays in
Rust and the RoPE tables are constants. ``verify_granite.py`` is what actually
proves fp16 is safe here, per compute unit.
"""
import os
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _granite_common import (
    DRIVER_FLOOR,
    EMBED_DIM,
    MODEL_ID,
    MODEL_STEM,
    REV,
    SEQ_LEN,
    GraniteGraph,
    assert_prompt_free,
    begin_run,
    cos,
    digest_tree,
    driver_crosscheck,
    load_encoder,
    load_sentence_transformer,
    padded_inputs,
    stage_dir,
    worst_update,
    write_producer_record,
    write_staged_crosscheck,
)
from _fixtures import CORPUS

AUTHOR = (
    f"embedkit T1: {MODEL_ID} (ModernBERT encoder + CLS pooling), pre-L2-norm; "
    "static per-layer local/global masks + fp32 dual-RoPE at fixed 512"
)
SHORT_DESCRIPTION = (
    "granite-embedding-97m text encoder: input_ids/attention_mask [1,512] -> 384-d CLS "
    "embedding (PRE-L2-norm; caller L2-normalizes). Prompt-free (raw strings)."
)


def main():
    stage = stage_dir()
    # Before ANY failure-prone work: mint this run's id and discard every
    # attestation a previous run left behind, so a conversion that dies during
    # loading, the driver check or tracing cannot leave records that downstream
    # stages would read as describing the packages still on disk.
    run_id = begin_run(stage)
    encoder = load_encoder()
    st = load_sentence_transformer()
    assert_prompt_free(st)
    net = GraniteGraph(encoder, SEQ_LEN).eval()

    # ---- 1. faithfulness vs the UNMODIFIED canonical pipeline ----------------
    print("\n=== driver faithfulness (GraniteGraph vs the canonical pipeline) ===")
    crosscheck = driver_crosscheck(st, net)
    for row in crosscheck["per_entry"]:
        print(f"  [faithful] {row['id']:<10} cos = {row['cosine_canonical_vs_driver']:.16f}")
    worst = crosscheck["worst_cosine_canonical_vs_driver"]
    print(f"[CHECK] driver worst faithfulness cos = {worst:.16f} (floor {DRIVER_FLOOR})")
    if not (worst >= DRIVER_FLOOR):
        raise SystemExit(f"driver UNFAITHFUL: worst {worst:.16f} < {DRIVER_FLOOR}")
    write_staged_crosscheck(stage, run_id, crosscheck)

    # ---- 2. trace at the fixed window ---------------------------------------
    tokenizer = st.tokenizer
    pad_id = encoder.config.pad_token_id
    feeds = []
    for entry in CORPUS:
        input_ids, attention_mask, _ids = padded_inputs(tokenizer, entry["text"], pad_id)
        feeds.append((
            entry["id"],
            torch.from_numpy(input_ids).to(torch.long),
            torch.from_numpy(attention_mask).to(torch.long),
        ))
    example = (feeds[0][1], feeds[0][2])
    traced = torch.jit.trace(net, example, check_trace=False)

    print("\n=== traced-vs-eager ===")
    worst_trace = 1.0
    for name, input_ids, attention_mask in feeds:
        with torch.no_grad():
            traced_out = traced(input_ids, attention_mask).numpy()
            eager_out = net(input_ids, attention_mask).numpy()
        for label, out in (("traced", traced_out), ("eager", eager_out)):
            if not np.all(np.isfinite(out)):
                raise SystemExit(f"{label} output is non-finite on `{name}`; refusing to convert")
        worst_trace = worst_update(worst_trace, cos(traced_out, eager_out))
    print(f"[CHECK] traced-vs-eager worst cos = {worst_trace:.16f} (floor {DRIVER_FLOOR})")
    if not (worst_trace >= DRIVER_FLOOR):
        raise SystemExit(f"trace UNFAITHFUL: {worst_trace:.16f} < {DRIVER_FLOOR} (or non-finite)")

    # ---- 3. convert both precisions -----------------------------------------
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, SEQ_LEN), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, SEQ_LEN), dtype=np.int32),
    ]
    outputs = [ct.TensorType(name="embedding", dtype=np.float32)]
    for tag, precision in (("", ct.precision.FLOAT16), ("_fp32", ct.precision.FLOAT32)):
        ml = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=precision,
            convert_to="mlprogram",
        )
        ml.author = AUTHOR
        ml.short_description = SHORT_DESCRIPTION
        spec = ml.get_spec()
        got_in = {i.name: (i.type.multiArrayType.dataType, list(i.type.multiArrayType.shape))
                  for i in spec.description.input}
        got_out = {o.name: (o.type.multiArrayType.dataType, list(o.type.multiArrayType.shape))
                   for o in spec.description.output}
        assert set(got_in) == {"input_ids", "attention_mask"}, sorted(got_in)
        assert set(got_out) == {"embedding"}, sorted(got_out)
        for name, (_dtype, shape) in got_in.items():
            assert shape == [1, SEQ_LEN], (name, shape)
        assert got_out["embedding"][1] == [1, EMBED_DIM], got_out["embedding"]
        out = os.path.join(stage, f"{MODEL_STEM}{tag}.mlpackage")
        ml.save(out)
        print(f"SAVED {out}  ({precision})")

    produced = {
        name: digest_tree(os.path.join(stage, name))
        for name in (f"{MODEL_STEM}.mlpackage", f"{MODEL_STEM}_fp32.mlpackage")
    }
    write_producer_record(stage, run_id, produced)
    print(f"DONE convert ({MODEL_ID}@{REV[:12]}, driver faithfulness {worst:.16f})")


if __name__ == "__main__":
    main()
