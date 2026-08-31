"""Emit CHECKSUMS.sha256 (exact per-file manifest the io gate consumes) + MANIFEST.json for one
staged ``ced_<size>.mlmodelc`` bundle. Usage: ``python write_manifest.py <size>``."""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _ced_common import MODELS, models_out_dir, sha256_file


def walk(bundle: Path):
    """(relative forward-slash path, sha256) for every file under ``bundle``, skipping OS
    sidecars the CoreML loader never reads (matches the Rust ``collect_files_rel``)."""
    rows = []
    for p in sorted(bundle.rglob("*")):
        if p.is_dir():
            continue
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        rows.append((p.relative_to(bundle).as_posix(), sha256_file(p)))
    return rows


def main(size):
    repo, rev, st_sha, onnx_sha, embed_dim, num_heads = MODELS[size]
    out = models_out_dir() / f"ced-{size}"
    bundle = out / f"ced_{size}.mlmodelc"
    if not bundle.is_dir():
        raise SystemExit(f"missing compiled bundle {bundle}")
    rows = walk(bundle)
    (out / "CHECKSUMS.sha256").write_text(
        "".join(f"{sha}  {rel}\n" for rel, sha in rows))
    manifest = {
        "size": size,
        "source_repo": repo,
        "source_revision": rev,
        "source_safetensors_sha256": st_sha,
        "source_onnx_sha256": onnx_sha,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "graph": "mel[1,64,1001] f32 -> logits[1,527] f32 (PRE-sigmoid; sigmoid by the caller)",
        "toolchain": "torch 2.5.1, torchaudio 2.5.1, transformers 4.53.3, coremltools 9.0, "
                     "minimum_deployment_target=iOS17, mlprogram",
        "attention_impl": "eager (plain matmul/softmax; no custom ops)",
        "converted_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle": {rel: sha for rel, sha in rows},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {size}: {len(rows)} files -> {out}/CHECKSUMS.sha256")
    for rel, sha in rows:
        print(f"      {rel}  {sha[:16]}…")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_manifest.py <size>")
    main(sys.argv[1])
