"""Emit CHECKSUMS.sha256 + MANIFEST.json for the staged shipped bundle.

``CHECKSUMS.sha256`` covers EXACTLY the published file set — both
``granite_97m_512`` bundles (``.mlmodelc`` and ``.mlpackage``), the model card,
and the runtime ``tokenizer.json`` the Rust crate loads from beside the bundle —
as ``<sha256>  ./<relative/path>`` lines, so it can be checked in place
with ``shasum -a 256 -c`` and set-compared against a published manifest. That
set is pinned in ``_granite_common.EXPECTED_ARTIFACT_FILES`` and asserted against
a RECURSIVE walk of the artifact root before anything is written: a manifest
covering a different set is not comparable with the published one, so a missing
path — or a stray file anywhere under the root — is a hard failure rather than a
note. Only this step's own outputs (``CHECKSUMS.sha256``, ``MANIFEST.json``) are
excluded from that walk.

``MANIFEST.json`` is recipe provenance (source pins, the PRODUCER's observed toolchain,
contract, measured verify numbers) and is deliberately NOT listed in CHECKSUMS:
it is written after the hashes and is not part of the distributed artifact set.

Nothing here is written until ``require_verify_evidence`` has accepted complete,
passing verification evidence that is digest-bound to this exact build — which
includes the run-bound compilation record for the shipped ``.mlmodelc`` and both
committed goldens, re-validated here as a pair. Publishing an artifact whose
evidence is missing, partial, failing, or left over from an earlier run is the
failure this step exists to prevent.

Publication itself is a THREE-state transition, not two writes. Every
precondition runs first and both payloads are built in full; only then are the
previous outputs invalidated, and each new payload is renamed into place from a
unique temp — manifest first, checksums last. An interruption therefore leaves
the root unpublished or visibly half-published, never a truncated checksum list
that ``shasum -c`` accepts, and never new checksums beside a manifest from
another run.

"Interruption" there means this PROCESS stopping — an exception, ENOSPC, a
Ctrl-C, a SIGKILL — which is what ``os.replace``'s atomicity covers. It is not a
power-loss claim: the fsyncs around these renames are the best-effort flush
``fsync_dir`` describes, and recovery from a power cut is to re-run the
conversion.

Nothing before that block touches the previous publication, which is what keeps a
failed precondition — an unset ``GRANITE_GOLDENS``, a mistyped stage path — from
destroying a valid attestation in a tree this step then refuses to publish to.

The hashes recorded here are what a byte-identity check against a published
CHECKSUMS.sha256 compares. Re-derivation is proven by ``verify_granite.py``'s
floors, not by those bytes matching.
"""
import json
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from _fixtures import goldens_dir
from _granite_common import (
    CHECKSUMS_FILE,
    EMBED_DIM,
    MANIFEST_FILE,
    MODEL_CARD,
    MODEL_CARD_SHA256,
    MODEL_ID,
    REV,
    SEQ_LEN,
    SOURCE_SHA256,
    TOKENIZER_FILE,
    assert_artifact_file_set,
    digest_files,
    discard_publication_outputs,
    fsync_dir,
    fsync_file,
    FP32_REFERENCE,
    RUN_ID_KEY,
    SHIPPED_PACKAGE,
    digest_tree,
    model_root,
    read_producer_record,
    replace_file_atomic,
    require_verify_evidence,
    sha256_file,
    src_dir,
    stage_dir,
)


def stage_model_card(root):
    """Ensure the PUBLISHED model card — by digest — sits beside the bundles.

    The card is maintained in the artifact repo, not generated here, so the recipe
    either finds it already staged or copies the one ``GRANITE_MODEL_CARD`` points
    at. Both paths are validated against ``MODEL_CARD_SHA256``: an existing
    destination may be stale or half-written from an earlier failed copy, and a
    supplied path may simply be the wrong file, and either would otherwise be
    checksummed into the manifest as though it were the published card. The copy
    goes through a unique temp name and is renamed into place, so an interrupted
    copy cannot leave a truncated card behind — and the temp is removed on ANY
    exit from the copy, not only on a digest mismatch. A leftover
    ``README.md.<hex>.tmp`` is not adoptable, but it fails the exact-file-set gate
    on every retry. A hard kill runs no handler at all, so that case is not
    recoverable from inside the process; ``assert_no_stale_publication_temps``
    names the file and the remedy instead of guessing that it is safe to delete.

    Contents are flushed before the rename and the directory after it, for no
    reason beyond matching what ``replace_file_atomic`` does with its own temp.
    Both are the best-effort flush ``fsync_dir`` describes; neither is load-bearing
    for anything above, and the difference was only ever an inconsistency between
    two neighbouring writers."""
    dst = os.path.join(root, MODEL_CARD)
    if os.path.isfile(dst):
        got = sha256_file(dst)
        if got == MODEL_CARD_SHA256:
            return
        raise SystemExit(
            f"STAGED MODEL CARD IS NOT THE PUBLISHED ONE ({dst}):\n"
            f"  got  {got}\n  want {MODEL_CARD_SHA256}\n"
            f"  Remove it and re-run with GRANITE_MODEL_CARD pointing at the published card."
        )
    src = os.environ.get("GRANITE_MODEL_CARD")
    if not src or not os.path.isfile(src):
        raise SystemExit(
            f"{MODEL_CARD} is missing from {root} and GRANITE_MODEL_CARD is unset or not a file.\n"
            f"  The model card is part of the published artifact set; a CHECKSUMS.sha256 without it "
            f"cannot be set-compared against the published manifest.\n"
            f"  Stage the card under the bundle dir, or set GRANITE_MODEL_CARD to it, and re-run."
        )
    got = sha256_file(src)
    if got != MODEL_CARD_SHA256:
        raise SystemExit(
            f"GRANITE_MODEL_CARD IS NOT THE PUBLISHED CARD ({src}):\n"
            f"  got  {got}\n  want {MODEL_CARD_SHA256}"
        )
    tmp = f"{dst}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(src, tmp)
        if sha256_file(tmp) != MODEL_CARD_SHA256:
            raise SystemExit(f"model card copy from {src} did not land intact; retry")
        fsync_file(tmp)
        os.replace(tmp, dst)
        fsync_dir(root)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"[ok] staged the published model card from {src}")


def stage_tokenizer(root):
    """Ensure the pinned SOURCE ``tokenizer.json`` — by digest — sits beside the
    bundles.

    The Rust crate no longer embeds the granite tokenizer, so the artifact is the
    only place ``TextEmbedder::load`` can get it: an artifact published without
    this file has no working default constructor. It is copied verbatim from the
    verified source snapshot, so the digest the crate pins
    (``contract::TOKENIZER_SHA256_HEX``) and the digest CHECKSUMS.sha256 records
    are the same ``SOURCE_SHA256["tokenizer.json"]`` value by construction.

    Both paths are validated against that pin: an existing destination may be
    stale or half-written from an earlier failed copy, and a source snapshot may
    be the wrong revision — either would otherwise be checksummed into the
    manifest as though it were the pinned tokenizer. The copy goes through a
    unique temp name and is renamed into place, so an interrupted copy cannot
    leave a truncated tokenizer behind — and, exactly as in ``stage_model_card``,
    the temp is removed on ANY exit from the copy, not only on a digest mismatch.
    A hard kill runs no handler, so that case is left to
    ``assert_no_stale_publication_temps``, which recognises this file's temps too.

    The flushes around the rename carry no more weight here than they do there:
    both are the best-effort flush ``fsync_dir`` describes, and they exist so the
    two copiers into the artifact root behave identically."""
    want = SOURCE_SHA256[TOKENIZER_FILE]
    dst = os.path.join(root, TOKENIZER_FILE)
    if os.path.isfile(dst):
        got = sha256_file(dst)
        if got == want:
            return
        raise SystemExit(
            f"STAGED TOKENIZER IS NOT THE PINNED ONE ({dst}):\n"
            f"  got  {got}\n  want {want}\n"
            f"  Remove it and re-run; it is copied from the verified source snapshot."
        )
    src = os.path.join(src_dir(), TOKENIZER_FILE)
    if not os.path.isfile(src):
        raise SystemExit(
            f"{TOKENIZER_FILE} is missing from {root} and from the source snapshot {src}.\n"
            f"  The tokenizer is part of the published artifact set — the Rust crate loads it "
            f"from beside the bundle — so a CHECKSUMS.sha256 without it describes an artifact "
            f"no caller can use.\n"
            f"  Download the source checkpoint (GRANITE_SRC_MODEL) and re-run."
        )
    got = sha256_file(src)
    if got != want:
        raise SystemExit(
            f"SOURCE TOKENIZER IS NOT THE PINNED ONE ({src}):\n"
            f"  got  {got}\n  want {want}"
        )
    tmp = f"{dst}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(src, tmp)
        if sha256_file(tmp) != want:
            raise SystemExit(f"tokenizer copy from {src} did not land intact; retry")
        fsync_file(tmp)
        os.replace(tmp, dst)
        fsync_dir(root)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"[ok] staged the pinned tokenizer from {src}")


def main():
    root = model_root()
    stage = stage_dir()

    # Everything from here to the publication block is preflight: read-only, or
    # constructive (the model card and the tokenizer). The previous publication
    # is NOT touched until both payloads exist and every precondition has passed
    # — see the invalidation below.
    goldens = goldens_dir()
    produced = {
        name: digest_tree(os.path.join(stage, name))
        for name in (SHIPPED_PACKAGE, FP32_REFERENCE)
    }
    producer = read_producer_record(stage, produced)
    run_id = producer[RUN_ID_KEY]
    stage_model_card(root)
    stage_tokenizer(root)
    rels = assert_artifact_file_set(root)
    verify = require_verify_evidence(root, stage, goldens, run_id)
    # The PRODUCER's environment, carried in the digest-bound evidence — never
    # this writer's, which can trivially be a different shell.
    toolchain = verify["toolchain"]

    checks = digest_files(root, rels)
    checksums_text = "".join(f"{checks[rel]}  {rel}\n" for rel in rels)

    manifest = {
        "source": {
            "repo": MODEL_ID,
            "revision": REV,
            "license": "Apache-2.0",
            "files_sha256": SOURCE_SHA256,
        },
        RUN_ID_KEY: run_id,
        "toolchain": toolchain,
        "toolchain_source": (
            "observed in the environment that CONVERTED and VERIFIED this artifact, carried in "
            "the digest-bound verify_metrics.json; not resampled by this writer"
        ),
        "conversion_date": producer["converted_utc"],
        "graph": (
            "ModernBERT encoder (12 layers, hidden 384) with static per-layer local/global "
            "attention masks and precomputed fp32 dual-RoPE at a fixed 512 window, CLS-sliced"
        ),
        "attention_impl": "eager (additive float masks; the driver supplies them as constants)",
        "contract": {
            "inputs": {
                "input_ids": f"int32 [1, {SEQ_LEN}]",
                "attention_mask": f"int32 [1, {SEQ_LEN}]",
            },
            "output": {"embedding": f"float32 [1, {EMBED_DIM}] (CLS, pre-L2-norm)"},
            "note": "L2 normalization is applied by the caller (Rust), OUT of the graph.",
            "prompt_free": True,
        },
        "shipped_files_sha256": checks,
        "verify": verify,
        "replayability": (
            "re-derivable to the floors in verify_granite.py; NOT bit-reproducible — "
            "the CoreML compiler and torch's fp32 reduction order are not pinned by these versions"
        ),
    }
    manifest_text = json.dumps(manifest, indent=2)

    # ---- publication: the ONLY destructive region in this step ----------------
    #
    # Both payloads are complete in memory and every precondition has passed, so
    # from here the only ways out are success or a crash. That is why the
    # invalidation lives HERE and not on entry: an unset GRANITE_GOLDENS or a
    # mistyped stage path used to delete a valid CHECKSUMS/MANIFEST pair and only
    # then abort, and the directory sync pushed that loss straight out to the
    # drive. A failed precondition now leaves an existing publication exactly as
    # it was.
    discard_publication_outputs(root)

    # Each payload goes down as one atomic rename, so no interruption can leave
    # the checksum SUBSET a line-by-line writer would — the kind `shasum -c`
    # accepts as a full verdict.
    #
    # CHECKSUMS.sha256 is written LAST, deliberately. With both prior outputs just
    # invalidated, the reachable states are: neither file (unpublished), the
    # manifest alone (visibly unfinished), or both from this run. That ordering is
    # what holds the invariant — CHECKSUMS.sha256 present implies a MANIFEST.json
    # for the same run is already beside it, and no interruption can leave an
    # older manifest next to newer checksums.
    replace_file_atomic(os.path.join(root, MANIFEST_FILE), manifest_text)
    replace_file_atomic(os.path.join(root, CHECKSUMS_FILE), checksums_text)
    print(f"[ok] {MANIFEST_FILE} ({len(verify)} verify metrics recorded)")
    print(f"[ok] {CHECKSUMS_FILE}: {len(rels)} files")
    for rel in rels:
        print(f"      {rel}  {checks[rel][:16]}…")


if __name__ == "__main__":
    main()
