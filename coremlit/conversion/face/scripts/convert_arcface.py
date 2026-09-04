"""``w600k_r50.onnx`` -> two ``.mlpackage``s (fp16, the shipped one, and fp32, the parity
reference). Usage: ``python convert_arcface.py``.

**The ONNX -> PyTorch hop is not optional.** ``coremltools`` removed its ONNX front end;
``coremltools.converters`` exposes libsvm, lightgbm, sklearn and xgboost and nothing else.
So the graph is rebuilt as a ``torch.nn.Module`` by ``onnx2torch``, traced, and handed to
``ct.convert`` through the torch front end. That hop is a place a conversion can go wrong
silently, so it is CHECKED before anything is converted: the rebuilt module is compared
against ``onnxruntime`` on the same random inputs and must agree to ``ONNX2TORCH_COS``.

**The input shape is pinned, not flexible.** The ONNX declares a symbolic batch
(``['None', 3, 112, 112]``); the CoreML input is ``[1, 3, 112, 112]`` fixed. A ``RangeDim``
or ``EnumeratedShapes`` input takes the graph off the Neural Engine for every shape but the
default (Apple developer forum 724930; coremltools #2370 measures ANE residency going 78% ->
0%), and ``coremlit``'s face door refuses a non-``Fixed`` geometry at load anyway — it reads
the batch back off the feature and would otherwise be reading a default rather than a fact.

**Batch 1 is a decision with a cost, recorded rather than defaulted.** Issue #115's census
measured fixed batch 2 buying ~1.7x on the ANE and then plateauing. Batch 1 is the honest
first artifact — the door chunks a slice to whatever capacity it reads back, so a batch-8
export is a drop-in follow-up — and ``measure_throughput.py`` records what batch 1 costs.
"""
import json
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (BUNDLE, EMBED_DIM, INPUT_NAME, INPUT_SHAPE, MLPACKAGE_FP16,
                             MLPACKAGE_FP32, ONNX_INPUT_NAME, OUTPUT_NAME, OUTPUT_SHAPE,
                             PACK_SHA256, PACK_URL, PREPROCESSING, RECOGNITION_SHA256,
                             conv_dir, observed_toolchain, require_source, staging_dir)

#: The floor the ONNX -> PyTorch rebuild must clear before anything is converted. This hop
#: is arithmetic-preserving by construction (op for op, weights copied), so the residual is
#: float-association noise and nothing else; the floor is set high enough that a genuinely
#: different graph cannot pass it.
ONNX2TORCH_COS = 0.99999
#: And the same rebuild's worst absolute elementwise difference, on a feature whose
#: elements run to ~1. A cosine alone can hide a uniform scale.
ONNX2TORCH_MAXABS = 1e-4

#: iOS17 / macOS14: the floor that gives ``mlprogram`` the fp16 ANE path this artifact is
#: for. Older targets fall back to ``neuralnetwork``, whose outputs report an
#: ``Unspecified`` shape constraint that ``coremlit``'s face door refuses at load.
DEPLOYMENT_TARGET = "iOS17"


def rebuilt_module(onnx_path):
    from onnx2torch import convert

    module = convert(str(onnx_path)).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def check_rebuild(module, onnx_path, trials=8, seed=115):
    """Refuse to convert a module that is not the ONNX."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    worst_cos, worst_abs = 1.0, 0.0
    for _ in range(trials):
        # The real input domain: preprocessed uint8 pixels land in [-1, 1].
        x = rng.uniform(-1.0, 1.0, INPUT_SHAPE).astype(np.float32)
        want = session.run(None, {ONNX_INPUT_NAME: x})[0]
        with torch.no_grad():
            got = module(torch.from_numpy(x)).numpy()
        if got.shape != want.shape:
            raise SystemExit(f"rebuild shape {got.shape} != onnx {want.shape}")
        a, b = want.astype(np.float64).ravel(), got.astype(np.float64).ravel()
        worst_cos = min(worst_cos, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        worst_abs = max(worst_abs, float(np.abs(a - b).max()))
    if worst_cos < ONNX2TORCH_COS or worst_abs > ONNX2TORCH_MAXABS:
        raise SystemExit(f"ONNX -> PyTorch rebuild does not reproduce the graph: worst cosine "
                         f"{worst_cos!r} (floor {ONNX2TORCH_COS}), worst |diff| {worst_abs!r} "
                         f"(ceiling {ONNX2TORCH_MAXABS})")
    print(f"[ok] onnx2torch rebuild reproduces onnxruntime: worst cosine {worst_cos:.9f}, "
          f"worst |diff| {worst_abs:.3e} over {trials} trials")
    return {"trials": trials, "worst_cosine": worst_cos, "worst_abs": worst_abs}


def convert_one(traced, precision, out_path):
    import coremltools as ct

    model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, DEPLOYMENT_TARGET),
        inputs=[ct.TensorType(name=INPUT_NAME, shape=INPUT_SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float32)],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    model.short_description = (
        "InsightFace w600k_r50 (buffalo_l recognition head), IResNet-50 / 512-d. "
        "RESEARCH USE ONLY: non-commercial weights, WebFace600K corpus. "
        f"Input {INPUT_NAME} [1, 3, 112, 112] f32 NCHW RGB, (x - 127.5) / 127.5. "
        f"Output {OUTPUT_NAME} [1, 512] f32, RAW — the caller L2-normalises."
    )
    model.author = "converted by coremlit/conversion/face from deepinsight/insightface buffalo_l"
    model.license = "InsightFace model licence: non-commercial research use only"
    model.version = "buffalo_l/w600k_r50"
    model.save(str(out_path))
    return model


def assert_declared_contract(model, label):
    spec = model.get_spec()
    desc = spec.description
    failures = []
    if [f.name for f in desc.input] != [INPUT_NAME]:
        failures.append(f"inputs {[f.name for f in desc.input]} != [{INPUT_NAME!r}]")
    if [f.name for f in desc.output] != [OUTPUT_NAME]:
        failures.append(f"outputs {[f.name for f in desc.output]} != [{OUTPUT_NAME!r}]")
    if desc.HasField("stateTypes") if hasattr(desc, "stateTypes") else False:
        failures.append("the model declares state")
    if list(getattr(desc, "state", [])):
        failures.append("the model declares an MLState buffer; coremlit's face door "
                        "refuses a stateful graph at load")
    for feature, want in ((desc.input[0], INPUT_SHAPE), (desc.output[0], OUTPUT_SHAPE)):
        array = feature.type.multiArrayType
        if not feature.type.HasField("multiArrayType"):
            failures.append(f"{feature.name}: not a MultiArray (an ImageType input is the "
                            f"blocker this whole conversion exists to remove)")
            continue
        if list(array.shape) != list(want):
            failures.append(f"{feature.name}: shape {list(array.shape)} != {list(want)}")
        if array.WhichOneof("ShapeFlexibility") is not None:
            failures.append(f"{feature.name}: shape is FLEXIBLE "
                            f"({array.WhichOneof('ShapeFlexibility')}); a RangeDim or "
                            f"EnumeratedShapes input is off the ANE for every non-default "
                            f"shape and coremlit's door refuses it at load")
        import coremltools.proto.FeatureTypes_pb2 as ft
        if array.dataType != ft.ArrayFeatureType.FLOAT32:
            failures.append(f"{feature.name}: dataType {array.dataType} != FLOAT32")
    if failures:
        raise SystemExit(f"DECLARED CONTRACT WRONG ({label}):\n  " + "\n  ".join(failures))
    print(f"[ok] {label}: {INPUT_NAME} {list(desc.input[0].type.multiArrayType.shape)} f32 "
          f"-> {OUTPUT_NAME} {list(desc.output[0].type.multiArrayType.shape)} f32, fixed, "
          f"MultiArray, stateless")


def main():
    import coremltools as ct

    observed = observed_toolchain()
    onnx_path = require_source()
    staging = staging_dir()

    print("=== rebuild the ONNX as a torch module ===")
    t0 = time.perf_counter()
    module = rebuilt_module(onnx_path)
    params = sum(p.numel() for p in module.parameters())
    print(f"[ok] rebuilt in {time.perf_counter() - t0:.1f}s, {params:,} parameters")
    rebuild = check_rebuild(module, onnx_path)
    rebuild["parameters"] = params

    example = torch.zeros(*INPUT_SHAPE)
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=True)
    out = traced(example)
    if tuple(out.shape) != OUTPUT_SHAPE:
        raise SystemExit(f"traced output shape {tuple(out.shape)} != {OUTPUT_SHAPE}")

    for precision, name in ((ct.precision.FLOAT16, MLPACKAGE_FP16),
                            (ct.precision.FLOAT32, MLPACKAGE_FP32)):
        label = "fp16" if precision is ct.precision.FLOAT16 else "fp32"
        print(f"=== ct.convert -> {label} ===")
        t0 = time.perf_counter()
        model = convert_one(traced, precision, staging / name)
        print(f"[ok] {name} in {time.perf_counter() - t0:.1f}s")
        assert_declared_contract(model, label)

    producer = {
        "run_id": uuid.uuid4().hex,
        "converted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle": BUNDLE,
        "source": {"pack_url": PACK_URL, "pack_sha256": PACK_SHA256,
                   "member": onnx_path.name, "member_sha256": RECOGNITION_SHA256},
        "onnx_to_torch": rebuild,
        "contract": {"input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "dtype": "float32"},
                     "output": {"name": OUTPUT_NAME, "shape": list(OUTPUT_SHAPE),
                                "dtype": "float32", "normalisation": "none (raw feature)"},
                     "embed_dim": EMBED_DIM,
                     "minimum_deployment_target": DEPLOYMENT_TARGET,
                     "convert_to": "mlprogram"},
        "preprocessing": PREPROCESSING,
        "toolchain": observed,
    }
    (conv_dir() / "producer.json").write_text(json.dumps(producer, indent=2) + "\n")
    print(f"[ok] wrote {conv_dir() / 'producer.json'} (run {producer['run_id'][:12]}…)")


if __name__ == "__main__":
    main()
