"""The committed ONNX reference embeddings. Usage: ``python write_onnx_reference.py``.

``verify_arcface.py`` measures the CoreML bundle against a LIVE ``onnxruntime`` session and
then throws the vectors away — it keeps only the cosines. That is enough for a conversion
report and not enough for a gate: ``tests/face/parity.rs`` has to make the same comparison
inside ``cargo test``, where there is no ONNX runtime at all (the ``face`` feature pulls
none, and issue #115's cross-platform ``ort`` road is not built). So the reference is cut
here, once, and committed as data — the shape ``granite`` and ``siglip`` already use for
their transformers-fp32 goldens.

**What is committed is the ONNX's output, not the ONNX.** The weights are InsightFace's
non-commercial research terms and this repository redistributes no weight bytes; 18 × 512
floats of a model's output over six public-domain NASA photographs are a measurement of
this conversion, not a redistribution of the model. The bytes that produced them are pinned
by hash in the fixture's own provenance block, so a reader can regenerate and compare.

Everything about the measurement is the recipe's own: the same ``preprocess`` arithmetic
every other stage goes through, the same pinned ``w600k_r50.onnx``, the same committed
crops, and ``CPUExecutionProvider`` in fp32. The known-pairs statistics are recomputed here
and written beside the vectors, so a regeneration that silently changed the numbers is
visible in the diff rather than only in a test run.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (EMBED_DIM, ONNX_INPUT_NAME, ONNX_OUTPUT_NAME, PREPROCESSING,
                             PACK_NAME, PACK_SHA256, RECOGNITION_BYTES, RECOGNITION_MEMBER,
                             RECOGNITION_SHA256, TEMPLATE_SIZE, cos, fixtures_dir,
                             observed_toolchain, require_source, sha256_file)

#: Where the fixture lands, relative to ``tests/face/fixtures/``.
REFERENCE_NAME = "onnx_reference.json"

#: InsightFace's own operating point, recomputed here so the committed file carries the
#: numbers the Rust gate asserts rather than only the vectors it asserts them from.
SAME_MIN = 0.28
DIFFERENT_MAX = 0.20


def load_crops():
    """The committed aligned crops, each re-verified against the fixture manifest's hash.

    A reference cut from bytes that have moved is a reference for a different corpus, so
    this refuses rather than warns — the same rule ``build_fixtures.py`` applies to the NASA
    source assets."""
    faces_dir = fixtures_dir() / "faces"
    manifest = json.loads((faces_dir / "manifest.json").read_text())
    rows = []
    for row in manifest["faces"]:
        path = faces_dir / row["crop"]
        got = sha256_file(path)
        if got != row["crop_sha256"]:
            raise SystemExit(f"{row['crop']}: sha256 {got}, manifest {row['crop_sha256']}")
        raw = path.read_bytes()
        want = TEMPLATE_SIZE * TEMPLATE_SIZE * 3
        if len(raw) != want:
            raise SystemExit(f"{row['crop']}: {len(raw)} bytes, expected {want}")
        rows.append((row, np.frombuffer(raw, np.uint8).reshape(TEMPLATE_SIZE, TEMPLATE_SIZE, 3)))
    return manifest, rows


def pair_stats(unit, labels):
    """(min same-person cosine, max different-person cosine, the margin, and the pair that
    sets each) over L2-normalised rows."""
    sims = unit @ unit.T
    same, diff = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (same if labels[i] == labels[j] else diff).append((float(sims[i, j]), i, j))
    worst_same, worst_diff = min(same), max(diff)
    return {
        "same_pairs": len(same),
        "different_pairs": len(diff),
        "min_same": worst_same[0],
        "max_different": worst_diff[0],
        "margin": worst_same[0] - worst_diff[0],
        "worst_same_pair": [worst_same[1], worst_same[2]],
        "worst_different_pair": [worst_diff[1], worst_diff[2]],
    }


def main():
    import onnxruntime as ort

    # Only what this stage IMPORTS. Recording torch or coremltools here would be recording
    # versions that never ran — see `observed_toolchain`'s `keys`.
    observed = observed_toolchain(keys=("numpy", "onnxruntime"))
    onnx = require_source()
    manifest, rows = load_crops()
    print(f"[ok] {len(rows)} committed crops, verified against the fixture manifest")

    crops = np.stack([crop for _, crop in rows])
    x = np.asarray(crops, dtype=np.float32) * np.float32(PREPROCESSING["scale"])
    x = np.ascontiguousarray((x + np.asarray(PREPROCESSING["bias"], np.float32))
                             .transpose(0, 3, 1, 2))

    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    if (in_name, out_name) != (ONNX_INPUT_NAME, ONNX_OUTPUT_NAME):
        raise SystemExit(f"ONNX feature names moved: {in_name!r} -> {out_name!r}")
    embeddings = np.stack([session.run(None, {in_name: v[None]})[0].ravel() for v in x])
    if embeddings.shape != (len(rows), EMBED_DIM) or embeddings.dtype != np.float32:
        raise SystemExit(f"onnxruntime returned {embeddings.shape} {embeddings.dtype}")

    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if norms.min() < 2.0:
        raise SystemExit(f"reference ‖e‖ min {norms.min()} — these are not RAW embeddings")
    unit = embeddings.astype(np.float64) / norms[:, None]
    labels = [row["person"] for row, _ in rows]
    stats = pair_stats(unit, labels)
    ok = stats["min_same"] >= SAME_MIN and stats["max_different"] < DIFFERENT_MAX
    print(f"[ok] ‖e‖ = {norms.min():.2f}..{norms.max():.2f} (median {np.median(norms):.2f})")
    print(f"  known pairs: min same {stats['min_same']:.4f}  max different "
          f"{stats['max_different']:.4f}  margin {stats['margin']:+.4f}  "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("the committed crops no longer separate at InsightFace's own "
                         "operating point — regenerate the fixtures before the reference")

    names = [row["id"] for row, _ in rows]
    out = {
        "what": ("fp32 onnxruntime embeddings of the committed aligned crops — the "
                 "cross-implementation reference tests/face/parity.rs compares the CoreML "
                 "door against. Cut by conversion/face/scripts/write_onnx_reference.py."),
        "source": {
            "pack": PACK_NAME,
            "pack_sha256": PACK_SHA256,
            "member": RECOGNITION_MEMBER,
            "member_sha256": RECOGNITION_SHA256,
            "member_bytes": RECOGNITION_BYTES,
            "onnx_input": ONNX_INPUT_NAME,
            "onnx_output": ONNX_OUTPUT_NAME,
            "execution_provider": "CPUExecutionProvider",
            "precision": "fp32",
        },
        "preprocessing": PREPROCESSING,
        "dim": EMBED_DIM,
        "fixtures_revision": manifest.get("revision"),
        "known_pairs": {
            "same_min": SAME_MIN,
            "different_max": DIFFERENT_MAX,
            **stats,
            "worst_same_ids": [names[i] for i in stats["worst_same_pair"]],
            "worst_different_ids": [names[i] for i in stats["worst_different_pair"]],
        },
        "toolchain": observed,
        "faces": [
            {
                "id": row["id"],
                "person": row["person"],
                "crop": row["crop"],
                "crop_sha256": row["crop_sha256"],
                "l2_norm": float(norms[i]),
                "embedding": [float(v) for v in embeddings[i]],
            }
            for i, (row, _) in enumerate(rows)
        ],
    }
    # A round trip through the JSON before it is written: `float(np.float32)` is exact and
    # `repr` round-trips, so the committed text holds the fp32 values bit for bit — and a
    # reader that gets something else has a defect worth failing on here rather than in a
    # Rust gate three files away.
    text = json.dumps(out, indent=2) + "\n"
    back = json.loads(text)
    for i, face in enumerate(back["faces"]):
        if not np.array_equal(np.asarray(face["embedding"], np.float32), embeddings[i]):
            raise SystemExit(f"{face['id']}: the serialised embedding does not round-trip")
        if abs(cos(face["embedding"], embeddings[i]) - 1.0) > 1e-15:
            raise SystemExit(f"{face['id']}: round-trip cosine moved")
    path = fixtures_dir() / REFERENCE_NAME
    path.write_text(text)
    print(f"[ok] wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
