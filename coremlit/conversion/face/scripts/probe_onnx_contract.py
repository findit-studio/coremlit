"""Establish, from the pinned ONNX itself, every fact the CoreML contract rests on.
Usage: ``python probe_onnx_contract.py``.

Four questions, each answered by reading the graph rather than a model card:

1. **Does the graph end in an L2 normalisation?** ``coremlit``'s contract is that the DOOR
   normalises (``FaceEmbedder`` L2-normalises what the model returns), so an artifact that
   normalises internally would double-count. ``w600k_r50`` is expected to emit the raw
   pre-norm 512-d feature; if it does not, the tail is stripped and this recipe says so.
2. **What is the declared I/O?** Names, element types and shapes — including whether the
   batch axis is symbolic, which decides what ``ct.convert`` has to pin.
3. **What preprocessing does InsightFace itself apply to these bytes?**
   ``ArcFaceONNX.__init__`` DERIVES ``input_mean``/``input_std`` by scanning the graph, so
   this stage runs that derivation on the actual file instead of quoting the constant.
4. **Is the op set one CoreML can take?** Recorded, not assumed.

The answers are written to ``contract.json`` and asserted against ``_arcface_common``'s
declared contract, so a later stage cannot convert under a contract this stage disproved.
"""
import collections
import json
import sys

import numpy as np
import onnx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (CONTRACT, EMBED_DIM, INSIGHTFACE_REV, INPUT_SHAPE, ONNX_INPUT_NAME,
                             ONNX_OUTPUT_NAME, OUTPUT_SHAPE, PREPROCESSING, TEMPLATE_SIZE,
                             conv_dir, observed_toolchain, require_source)

#: Op types that would mean the graph normalises its own output. ONNX spells an L2 four
#: ways and this recipe refuses all of them rather than pattern-matching one: the direct
#: op, the MXNet-era contrib op, and the two decompositions a tracer emits.
NORMALISING_OPS = {"LpNormalization", "L2Normalization", "Normalize", "MeanVarianceNormalization"}
DECOMPOSED_L2 = {"ReduceL2", "Sqrt", "Div", "ReciprocalSqrt", "Reciprocal", "Pow"}


def declared_shape(value_info):
    t = value_info.type.tensor_type
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in t.shape.dim], onnx.TensorProto.DataType.Name(t.elem_type)


def insightface_normalisation(graph):
    """``ArcFaceONNX.__init__``'s own derivation, run on THIS graph.

    Reproduced from ``python-package/insightface/model_zoo/arcface_onnx.py`` at
    ``INSIGHTFACE_REV``. Their rule: if a ``Sub`` and a ``Mul`` appear among the first
    eight node NAMES the preprocessing is fused into the graph (the MXNet-era export, mean
    0 / std 1); otherwise the caller applies mean 127.5 / std 127.5. Quoting the constant
    would be quoting the branch we hope was taken — this runs the branch."""
    find_sub = find_mul = False
    for node in graph.node[:8]:
        if node.name.startswith("Sub") or node.name.startswith("_minus"):
            find_sub = True
        if node.name.startswith("Mul") or node.name.startswith("_mul"):
            find_mul = True
    if find_sub and find_mul:
        return 0.0, 1.0, "fused (mxnet-era export)"
    return 127.5, 127.5, "caller-applied"


def main():
    observed = observed_toolchain()
    path = require_source()
    model = onnx.load(str(path))
    graph = model.graph
    failures = []

    init_names = {i.name for i in graph.initializer}
    real_inputs = [i for i in graph.input if i.name not in init_names]
    if len(real_inputs) != 1:
        failures.append(f"expected exactly one non-initializer input, got "
                        f"{[i.name for i in real_inputs]}")
    if len(graph.output) != 1:
        failures.append(f"expected exactly one output, got {[o.name for o in graph.output]}")

    in_shape, in_type = declared_shape(real_inputs[0])
    out_shape, out_type = declared_shape(graph.output[0])
    in_name, out_name = real_inputs[0].name, graph.output[0].name

    if in_name != ONNX_INPUT_NAME or out_name != ONNX_OUTPUT_NAME:
        failures.append(f"ONNX feature names moved: {in_name!r}/{out_name!r}, pinned "
                        f"{ONNX_INPUT_NAME!r}/{ONNX_OUTPUT_NAME!r}")
    if in_type != "FLOAT" or out_type != "FLOAT":
        failures.append(f"element types {in_type}/{out_type}, expected FLOAT/FLOAT")
    if list(in_shape[1:]) != list(INPUT_SHAPE[1:]):
        failures.append(f"input spatial shape {in_shape}, expected [*, {INPUT_SHAPE[1]}, "
                        f"{TEMPLATE_SIZE}, {TEMPLATE_SIZE}]")
    if list(out_shape[1:]) != list(OUTPUT_SHAPE[1:]):
        failures.append(f"output width {out_shape}, expected [*, {EMBED_DIM}]")

    ops = collections.Counter(n.op_type for n in graph.node)
    normalising = sorted((NORMALISING_OPS | DECOMPOSED_L2) & set(ops))
    if normalising:
        failures.append(f"the graph carries op(s) that can normalise its output: "
                        f"{normalising}. coremlit's contract is that the DOOR normalises, "
                        f"so this tail must be inspected and stripped before conversion.")

    tail = [(n.op_type, list(n.input), list(n.output)) for n in graph.node[-4:]]

    mean, std, kind = insightface_normalisation(graph)
    if (mean, std) != (127.5, 127.5):
        failures.append(f"InsightFace's own derivation gives mean {mean} / std {std} "
                        f"({kind}); this recipe's PREPROCESSING assumes caller-applied "
                        f"127.5/127.5")
    want_scale = 1.0 / std if std else None
    if want_scale is not None and abs(want_scale - PREPROCESSING["scale"]) > 1e-12:
        failures.append(f"scale {PREPROCESSING['scale']!r} != 1/{std}")
    if want_scale is not None and any(abs(b - (-mean / std)) > 1e-12 for b in PREPROCESSING["bias"]):
        failures.append(f"bias {PREPROCESSING['bias']!r} != -{mean}/{std}")

    # The output magnitude, on the input the graph is cheapest to evaluate at. A model that
    # normalised internally would put this at 1.0 whatever it was fed; this is a second,
    # independent witness to the structural finding above.
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    norms = []
    for _ in range(4):
        x = rng.uniform(-1.0, 1.0, INPUT_SHAPE).astype(np.float32)
        norms.append(float(np.linalg.norm(session.run(None, {in_name: x})[0])))
    if min(norms) > 0.99 and max(norms) < 1.01:
        failures.append(f"every output norm is ~1 ({norms}) — the graph appears to "
                        f"normalise internally after all")

    record = {
        "file": path.name,
        "onnx": {
            "ir_version": model.ir_version,
            "producer": f"{model.producer_name} {model.producer_version}".strip(),
            "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
            "graph_name": graph.name,
            "initializers": len(graph.initializer),
            "nodes": len(graph.node),
            "op_counts": dict(sorted(ops.items())),
            "input": {"name": in_name, "type": in_type, "shape": in_shape},
            "output": {"name": out_name, "type": out_type, "shape": out_shape},
            "tail": tail,
        },
        "l2_at_the_tail": False,
        "output_norms_on_uniform_input": norms,
        "insightface_preprocessing": {
            "revision": INSIGHTFACE_REV,
            "derivation": kind,
            "input_mean": mean,
            "input_std": std,
            "channel_order_fed_to_the_model": "rgb",
            "why": "ArcFaceONNX.get_feat calls cv2.dnn.blobFromImages(..., swapRB=True) on "
                   "the aligned crop, and face_align.norm_crop warps an OpenCV BGR frame; "
                   "swapRB therefore hands the model RGB. Confirmed numerically by "
                   "verify_arcface.py's channel-order arm.",
        },
        "coremlit_contract": CONTRACT,
        "preprocessing": PREPROCESSING,
        "toolchain": observed,
    }

    print(f"  producer      {record['onnx']['producer']}, opset "
          f"{record['onnx']['opset'][0]['version']}, ir {model.ir_version}")
    print(f"  input         {in_name!r} {in_type} {in_shape}")
    print(f"  output        {out_name!r} {out_type} {out_shape}")
    print(f"  ops           {dict(sorted(ops.items()))}")
    print(f"  tail          {' -> '.join(t[0] for t in tail)}")
    print(f"  L2 at tail    NO (no normalising op in the graph; ‖out‖ = "
          f"{min(norms):.2f}..{max(norms):.2f} on uniform input)")
    print(f"  insightface   mean {mean} / std {std} ({kind}), fed RGB")

    if failures:
        raise SystemExit("CONTRACT PROBE FAILED:\n  " + "\n  ".join(failures))
    (conv_dir() / "contract.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[ok] wrote {conv_dir() / 'contract.json'}")


if __name__ == "__main__":
    main()
