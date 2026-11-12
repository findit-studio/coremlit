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

The destination is published whole by ONE exclusive rename and is never
replaced: an existing one is refused, with the command to move it aside (see
``stage_package``). It is not a power-loss claim.

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


def remove_tree(path):
    """Removes the tree at ``path`` if it exists, best effort: a read-only
    directory INSIDE it is made writable and the removal retried, and an error
    is not raised (a leftover temporary directory carries an obvious name and
    is only debris). Permissions are touched only on ``path`` and below it,
    never on its parent, whatever fails (an immutable file, say)."""
    import stat

    root = os.path.abspath(path)

    def inside(candidate):
        candidate = os.path.abspath(candidate)
        return candidate == root or candidate.startswith(root + os.sep)

    def writable(function, target, _info):
        try:
            if inside(os.path.dirname(target)):
                os.chmod(os.path.dirname(target), stat.S_IRWXU)
            if inside(target):
                os.chmod(target, stat.S_IRWXU)
            function(target)
        except OSError:
            pass

    if os.path.lexists(path):
        shutil.rmtree(path, onerror=writable)


def rename_exclusive(source, destination):
    """Renames ``source`` to ``destination`` in ONE system call that fails with
    ``EEXIST`` when ``destination`` exists in any form (a plain rename would
    replace an empty directory), so the kernel arbitrates concurrent stagings
    and an alias of the destination changes nothing: the rename lands on one
    path or fails. ``renamex_np(RENAME_EXCL)`` on macOS, ``renameat2
    (RENAME_NOREPLACE)`` on Linux."""
    import ctypes
    import errno as errno_module

    libc = ctypes.CDLL(None, use_errno=True)
    old, new = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "darwin":
        result = libc.renamex_np(old, new, ctypes.c_uint(0x4))  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        result = libc.renameat2(ctypes.c_int(-100), old, ctypes.c_int(-100), new, ctypes.c_uint(1))  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise SystemExit(f"no exclusive rename on {sys.platform}")
    if result != 0:
        code = ctypes.get_errno()
        if code in (errno_module.ENOTSUP, errno_module.EINVAL, errno_module.ENOSYS):
            raise SystemExit(f"this filesystem has no exclusive rename ({os.strerror(code)}): stage onto APFS or ext4")
        raise OSError(code, os.strerror(code), destination)


def refusal_for_existing(destination):
    return (
        f"{destination} already exists; staging never replaces a generation. Move the old one aside and run "
        f"again:\n  mv '{destination}' \"{destination}.old-$(date +%Y%m%d%H%M%S)\""
    )


def stage_package(source, out, attest_fn, after_attest=lambda temp: None):
    """Stages ``source`` as a NEW destination ``out`` and returns the digests
    written.

    The whole destination (the package and its ``CHECKSUMS.sha256``, written
    from the digests the attestation ran on) is built in a sibling directory
    and published by ONE exclusive rename (``rename_exclusive``). An existing
    destination is refused by name and left untouched: replacing a generation
    is the maintainer's explicit act (the refusal prints the command). So there
    is nothing to swap, back up, roll back or lock: of two concurrent stagings
    the kernel lets one rename succeed and refuses the other; an interrupt
    lands before or after the one system call; a leftover sibling directory
    (``.<name>.gen-*``) carries an obvious name and is removed best effort.

    ``attest_fn(package)`` attests the copy and returns ``{"files": ...}``;
    ``after_attest(package)`` is a test seam between the attestation and the
    checksums.
    """
    destination = os.path.abspath(out)
    if os.path.lexists(destination):
        raise SystemExit(refusal_for_existing(destination))
    parent, name = os.path.split(destination)
    os.makedirs(parent, exist_ok=True)
    generation = os.path.join(parent, f".{name}.gen-{uuid.uuid4().hex}")
    package = os.path.join(generation, FP32_AUDIO_PACKAGE)
    try:
        os.makedirs(generation)
        shutil.copytree(source, package)
        files = attest_fn(package)["files"]
        after_attest(package)
        if digest_tree(package) != files:
            raise SystemExit("the package's bytes changed after they were attested")
        names = {rel for rel, _ in files}
        if names != PACKAGE_FILES:
            raise SystemExit(f"the package's files are not the pinned set: {sorted(names ^ PACKAGE_FILES)}")
        lines = "".join(f"{sha}  ./{FP32_AUDIO_PACKAGE}/{rel}\n" for rel, sha in files)
        with open(os.path.join(generation, CHECKSUMS_FILE), "w", encoding="utf-8") as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())
        try:
            rename_exclusive(generation, destination)
        except FileExistsError:
            raise SystemExit(refusal_for_existing(destination)) from None
    except BaseException:
        remove_tree(generation)
        raise
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
