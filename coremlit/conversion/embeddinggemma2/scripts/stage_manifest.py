"""Verify and stage the fp32 audio package, and write its ``CHECKSUMS.sha256``,
in the shape of ``conversion/granite/scripts/write_manifest.py`` (G14).

One command, one process, no stored report. Staging copies the package into the
destination under a temporary name, attests THAT copy (``_attest.attest``: the
contract, an explicit ``beta`` on every ``layer_norm``, token parity against
PyTorch on the CPU road and the GPU road for every window), digests it once
more just before promotion, and writes ``CHECKSUMS.sha256`` from the digests
the attestation ran on. ``verify.json`` is an output of ``verify_audio_fp32.py``
for people; nothing here reads it, so a stale or edited one changes nothing.

``CHECKSUMS.sha256`` covers EXACTLY the published file set, the package's
files, as ``<sha256>  ./<relative/path>`` lines sorted by path, so it is
checked in place with ``shasum -a 256 -c`` and is the manifest row U uploads
beside the package (knife 2). The set is pinned in ``PACKAGE_FILES`` and
asserted before anything is promoted: a missing or a stray file is a failure,
not a note.

Promotion is transactional against a process stopping (an exception, ENOSPC, a
Ctrl-C, a SIGKILL): the old checksums are removed first, the previous package
is moved aside, the attested copy is renamed into place, and the new checksums
are renamed in last, so no interruption leaves checksums describing bytes that
are not there. It is not a power-loss claim; the recovery from one is to re-run
the recipe.

Environment: ``EG2_SRC``, ``EG2_TESTS``, ``EG2_MODELS`` (the staged
FluidInference tree, read only), ``EG2_STAGE`` (the converted package) and
``EG2_MODELS_OUT`` (the directory staged for upload). It runs the producer's
environment, as the verification does.
"""
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from _eg2_common import (  # noqa: E402
    CHECKSUMS_FILE,
    FP32_AUDIO_PACKAGE,
    PRODUCER_FILE,
    digest_tree,
    env,
    observed_toolchain,
    require_producer_toolchain,
    verify_source,
)

# The package's files: what CHECKSUMS.sha256 lists and the upload carries.
PACKAGE_FILES = {
    "Data/com.apple.CoreML/model.mlmodel",
    "Data/com.apple.CoreML/weights/weight.bin",
    "Manifest.json",
}


def stage_package(source, out, attest_fn, after_attest=lambda temp: None):
    """Stages ``source`` into ``out`` and returns the digests written.

    ``attest_fn(temp)`` attests the copy and returns ``{"files": ...}``;
    ``after_attest(temp)`` is a test seam called between the attestation and
    the promotion. Nothing is promoted unless the copy still has the attested
    digests; on any failure the copy is removed and ``out`` is left as it was.
    """
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, FP32_AUDIO_PACKAGE)
    checksums = os.path.join(out, CHECKSUMS_FILE)
    token = uuid.uuid4().hex
    temp = os.path.join(out, f".{FP32_AUDIO_PACKAGE}.tmp-{token}")
    aside = os.path.join(out, f".{FP32_AUDIO_PACKAGE}.old-{token}")
    shutil.copytree(source, temp)
    try:
        files = attest_fn(temp)["files"]
        after_attest(temp)
        if digest_tree(temp) != files:
            raise SystemExit("the package's bytes changed after they were attested")
        names = {rel for rel, _ in files}
        if names != PACKAGE_FILES:
            raise SystemExit(f"the package's files are not the pinned set: {sorted(names ^ PACKAGE_FILES)}")
        lines = "".join(f"{sha}  ./{FP32_AUDIO_PACKAGE}/{rel}\n" for rel, sha in files)
        temp_checksums = f"{checksums}.tmp-{token}"
        with open(temp_checksums, "w", encoding="utf-8") as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    if os.path.exists(checksums):
        os.remove(checksums)
    moved = False
    try:
        if os.path.exists(target):
            os.replace(target, aside)
            moved = True
        os.replace(temp, target)
    except BaseException:
        if moved and not os.path.exists(target):
            os.replace(aside, target)
        raise
    os.replace(temp_checksums, checksums)
    if moved:
        shutil.rmtree(aside)
    return files


def main():
    import json

    from _attest import attest, real_inputs

    src, tests, models = env("EG2_SRC"), env("EG2_TESTS"), env("EG2_MODELS")
    stage, out = env("EG2_STAGE"), env("EG2_MODELS_OUT")
    observed = observed_toolchain()
    with open(os.path.join(stage, PRODUCER_FILE), encoding="utf-8") as f:
        producer = json.load(f)
    require_producer_toolchain(producer, observed, "stage_manifest.py")
    verify_source(src)
    inputs = real_inputs(src, tests, models)
    files = stage_package(
        os.path.join(stage, FP32_AUDIO_PACKAGE),
        out,
        lambda temp: attest(temp, **inputs),
    )
    print(f"[ok] staged {os.path.join(out, FP32_AUDIO_PACKAGE)}; {CHECKSUMS_FILE} lists {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
