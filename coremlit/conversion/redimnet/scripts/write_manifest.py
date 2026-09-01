"""Emit ``CHECKSUMS.sha256`` (exact per-file manifest) + ``MANIFEST.json`` for the staged
``redimnet_b5.mlmodelc`` bundle. Usage: ``python write_manifest.py``.

The manifest records the OBSERVED toolchain of the run that produced the bundle — read out
of ``staging/producer.json``, and refused unless the toolchain running THIS script is the
same one entry for entry. A manifest that recorded the versions a comment claims, rather
than the versions that ran, is a provenance record nobody can replay (issue #97).
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import (CONTRACT, EMBED_DIM, N_FRAMES, N_MELS, SAMPLE_RATE, WINDOW_SAMPLES,
                              models_out_dir, observed_compiler, observed_toolchain, sha256_file,
                              staging_dir)

BUNDLE = "redimnet_b5.mlmodelc"


def walk(bundle: Path):
    """(relative forward-slash path, sha256) for every file under ``bundle``, skipping the
    OS sidecars the CoreML loader never reads."""
    rows = []
    for p in sorted(bundle.rglob("*")):
        if p.is_dir() or p.name.startswith("._") or p.name == ".DS_Store":
            continue
        rows.append((p.relative_to(bundle).as_posix(), sha256_file(p)))
    return rows


def require_producer(observed):
    path = staging_dir() / "producer.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} — run convert_redimnet.py first; the manifest "
                         f"records the toolchain that produced the bundle, not this one.")
    producer = json.loads(path.read_text())
    for field in ("run_id", "converted_utc", "toolchain", "source"):
        if not producer.get(field):
            raise SystemExit(f"{path}: producer record names no {field!r}")
    recorded = producer["toolchain"]
    diff = [f"{k}: producer {recorded.get(k)!r}, this run {observed.get(k)!r}"
            for k in sorted(set(recorded) | set(observed)) if recorded.get(k) != observed.get(k)]
    if diff:
        raise SystemExit("PRODUCER/MANIFEST TOOLCHAIN MISMATCH — refusing to describe a build "
                         "this environment did not make:\n  " + "\n  ".join(diff))
    print(f"[ok] producer and manifest toolchains are one environment "
          f"(run {producer['run_id'][:12]}…)")
    return producer


def main():
    observed = observed_toolchain()
    producer = require_producer(observed)
    out = models_out_dir()
    bundle = out / BUNDLE
    if not bundle.is_dir():
        raise SystemExit(f"missing compiled bundle {bundle}")
    rows = walk(bundle)
    (out / "CHECKSUMS.sha256").write_text("".join(f"{sha}  {rel}\n" for rel, sha in rows))
    manifest = {
        "model": "ReDimNet-B5 (vox2, ft_lm)",
        "lane": "identity (issue #123) — single fixed window, no mask",
        "source": producer["source"],
        "model_config": producer["model_config"],
        "graph": CONTRACT,
        "sample_rate_hz": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "mel": {"n_mels": N_MELS, "frames": N_FRAMES, "in_graph": True},
        "embed_dim": EMBED_DIM,
        "l2_normalized": False,
        "toolchain": observed,
        "compiler": observed_compiler(),
        "conversion": "minimum_deployment_target=iOS17, mlprogram, compute_precision=FLOAT16, "
                      "fixed input shape (no RangeDim)",
        "producer_run_id": producer["run_id"],
        "converted_utc": producer["converted_utc"],
        "manifested_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle": {rel: sha for rel, sha in rows},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {len(rows)} files -> {out}/CHECKSUMS.sha256")
    for rel, sha in rows:
        print(f"      {rel}  {sha}")


if __name__ == "__main__":
    main()
