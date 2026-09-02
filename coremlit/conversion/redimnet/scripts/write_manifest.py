"""Emit ``CHECKSUMS.sha256`` (exact per-file manifest) + ``MANIFEST.json`` for EVERY staged
ReDimNet bundle under the models-out root. Usage: ``python write_manifest.py``.

One manifest for the whole publish tree, not one per bundle. The tree carries several
``.mlmodelc`` bundles that share every file NAME (``model.mil``, ``weights/weight.bin``,
…), so the manifest's paths are ROOT-RELATIVE with the bundle as their first component —
``./redimnet_b2.mlmodelc/weights/weight.bin`` — the same layout ``speakerkit``'s
``CHECKSUMS.sha256`` uses for its two bundles and the one ``shasum -c`` verifies from the
kit root in CI. A bundle-relative manifest, which is what this script wrote when the tree
held one bundle, would list ``weights/weight.bin`` three times with three hashes.

Every bundle listed must have been PRODUCED by this recipe: its ``<bundle>_producer.json``
is required, its toolchain must equal the one running this script entry for entry, and a
``.mlmodelc`` in the tree with no producer record is refused rather than described — a
manifest that recorded the versions a comment claims, rather than the versions that ran,
is a provenance record nobody can replay (issue #97).
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import (CONTRACT, EMBED_DIM, N_FRAMES, N_MELS, SAMPLE_RATE, VARIANTS,
                              WINDOW_SAMPLES, models_out_dir, observed_compiler,
                              observed_toolchain, sha256_file)

BUNDLE_SUFFIX = ".mlmodelc"


def walk(root: Path, bundle: Path):
    """(root-relative forward-slash path with a leading ``./``, sha256) for every file under
    ``bundle``, skipping the OS sidecars the CoreML loader never reads."""
    rows = []
    for p in sorted(bundle.rglob("*")):
        if p.is_dir() or p.name.startswith("._") or p.name == ".DS_Store":
            continue
        rows.append(("./" + p.relative_to(root).as_posix(), sha256_file(p)))
    return rows


def require_producer(v, observed):
    path = v.staging_file("producer.json")
    if not path.is_file():
        raise SystemExit(f"missing {path} — run convert_redimnet.py for {v.key} first; the "
                         f"manifest records the toolchain that produced the bundle, not this one.")
    producer = json.loads(path.read_text())
    for field in ("run_id", "converted_utc", "toolchain", "source", "variant", "bundle"):
        if not producer.get(field):
            raise SystemExit(f"{path}: producer record names no {field!r}")
    if producer["variant"] != v.key or producer["bundle"] != v.mlmodelc:
        raise SystemExit(f"{path}: records variant {producer['variant']!r} / bundle "
                         f"{producer['bundle']!r}, expected {v.key!r} / {v.mlmodelc!r}")
    recorded = producer["toolchain"]
    diff = [f"{k}: producer {recorded.get(k)!r}, this run {observed.get(k)!r}"
            for k in sorted(set(recorded) | set(observed)) if recorded.get(k) != observed.get(k)]
    if diff:
        raise SystemExit("PRODUCER/MANIFEST TOOLCHAIN MISMATCH — refusing to describe a build "
                         "this environment did not make:\n  " + "\n  ".join(diff))
    print(f"[ok] {v.mlmodelc}: producer and manifest toolchains are one environment "
          f"(run {producer['run_id'][:12]}…)")
    return producer


def main():
    observed = observed_toolchain()
    out = models_out_dir()
    by_bundle = {v.mlmodelc: v for v in VARIANTS.values()}

    present = sorted(p.name for p in out.iterdir() if p.is_dir() and p.name.endswith(BUNDLE_SUFFIX))
    if not present:
        raise SystemExit(f"no {BUNDLE_SUFFIX} bundle under {out}")
    unknown = [b for b in present if b not in by_bundle]
    if unknown:
        raise SystemExit(f"{out} holds bundles this recipe did not produce and will not describe: "
                         f"{unknown}")

    all_rows, artifacts = [], {}
    for name in present:
        v = by_bundle[name]
        producer = require_producer(v, observed)
        rows = walk(out, out / name)
        all_rows.extend(rows)
        artifacts[name] = {
            "variant": v.key,
            "model": v.title,
            "training_crop_s": v.training_crop_s,
            "published_metrics": v.published_metrics,
            "pooled_dim": v.pooled_dim,
            "source": producer["source"],
            "model_config": producer["model_config"],
            "producer_run_id": producer["run_id"],
            "converted_utc": producer["converted_utc"],
            "files": {rel: sha for rel, sha in rows},
        }

    (out / "CHECKSUMS.sha256").write_text("".join(f"{sha}  {rel}\n" for rel, sha in all_rows))
    manifest = {
        "kit": "identity (issue #123) — single fixed window, no mask; one contract, several "
               "artifacts",
        "graph": CONTRACT,
        "sample_rate_hz": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "mel": {"n_mels": N_MELS, "frames": N_FRAMES, "in_graph": False,
                "note": "the log-mel front end runs in the CALLER (coremlit's Rust port); "
                        "the graph starts at the mel"},
        "embed_dim": EMBED_DIM,
        "l2_normalized": False,
        "toolchain": observed,
        "compiler": observed_compiler(),
        "conversion": "minimum_deployment_target=iOS17, mlprogram, compute_precision=FLOAT16, "
                      "fixed input shape (no RangeDim)",
        "manifested_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checksums": "CHECKSUMS.sha256 — root-relative paths with the bundle as their first "
                     "component; verify with `shasum -a 256 -c CHECKSUMS.sha256` from this "
                     "directory",
        "artifacts": artifacts,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {len(all_rows)} files across {len(present)} bundle(s) -> {out}/CHECKSUMS.sha256")
    for rel, sha in all_rows:
        print(f"      {rel}  {sha}")


if __name__ == "__main__":
    main()
