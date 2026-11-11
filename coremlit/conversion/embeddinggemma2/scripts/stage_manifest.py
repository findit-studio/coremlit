"""Stage the verified fp32 audio package and write its ``CHECKSUMS.sha256``,
in the shape of ``conversion/granite/scripts/write_manifest.py`` (G14).

``CHECKSUMS.sha256`` covers EXACTLY the published file set, the package's
files, as ``<sha256>  ./<relative/path>`` lines sorted by path, so it is
checked in place with ``shasum -a 256 -c`` and is the manifest row U uploads
beside the package (knife 2). The set is pinned in ``PACKAGE_FILES`` and
asserted against a recursive walk of the package before anything is written: a
missing or a stray file is a failure, not a note.

Nothing is staged unless ``verify.json`` passed and is bound to these exact
bytes (its package digest equals the package's now).

Promotion is transactional against a process stopping (an exception, ENOSPC, a
Ctrl-C, a SIGKILL): the package is copied beside the destination under a
unique temporary name and its copy checked digest for digest; the old
checksums are removed first, the previous package is moved aside, the new one
is renamed into place, and the new checksums are renamed in last, so no
interruption leaves checksums describing bytes that are not there. It is not a
power-loss claim; the recovery from one is to re-run the recipe.

Environment: ``EG2_STAGE`` (the converted, verified package) and
``EG2_MODELS_OUT`` (the directory staged for upload).
"""
import json
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from _eg2_common import (  # noqa: E402
    CHECKSUMS_FILE,
    FP32_AUDIO_PACKAGE,
    VERIFY_FILE,
    digest_tree,
    env,
    sha256_bytes,
)

# The package's files: what CHECKSUMS.sha256 lists and the upload carries.
PACKAGE_FILES = {
    "Data/com.apple.CoreML/model.mlmodel",
    "Data/com.apple.CoreML/weights/weight.bin",
    "Manifest.json",
}


def tree_digest(files):
    return sha256_bytes("".join(f"{sha}  {rel}\n" for rel, sha in files).encode())


def main():
    stage, out = env("EG2_STAGE"), env("EG2_MODELS_OUT")
    source = os.path.join(stage, FP32_AUDIO_PACKAGE)
    with open(os.path.join(stage, VERIFY_FILE), encoding="utf-8") as f:
        verify = json.load(f)
    files = digest_tree(source)
    if not verify.get("passed") or verify.get("package_digest") != tree_digest(files):
        raise SystemExit(f"{VERIFY_FILE} does not attest these bytes: re-run verify_audio_fp32.py")
    names = {rel for rel, _ in files}
    if names != PACKAGE_FILES:
        raise SystemExit(f"the package's files are not the pinned set: {sorted(names ^ PACKAGE_FILES)}")

    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, FP32_AUDIO_PACKAGE)
    checksums = os.path.join(out, CHECKSUMS_FILE)
    token = uuid.uuid4().hex
    temp = os.path.join(out, f".{FP32_AUDIO_PACKAGE}.tmp-{token}")
    aside = os.path.join(out, f".{FP32_AUDIO_PACKAGE}.old-{token}")
    shutil.copytree(source, temp)
    if digest_tree(temp) != files:
        shutil.rmtree(temp)
        raise SystemExit("the staged copy does not match its source")
    lines = "".join(f"{sha}  ./{FP32_AUDIO_PACKAGE}/{rel}\n" for rel, sha in files)
    temp_checksums = f"{checksums}.tmp-{token}"
    with open(temp_checksums, "w", encoding="utf-8") as f:
        f.write(lines)
        f.flush()
        os.fsync(f.fileno())

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
    print(f"[ok] staged {target}; {CHECKSUMS_FILE} lists {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
