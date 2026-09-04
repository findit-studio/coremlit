"""Pins, paths and toolchain observation shared by the ArcFace (``w600k_r50``) recipe.

Every stage imports its versions from here and every stage OBSERVES the environment it
runs in rather than recording a literal (issue #97). A venv that resolved coremltools 8.2
would otherwise finish the recipe and be written down as 8.3.0 — a provenance record
nobody can replay.

**The source is one zip, pinned by the SHA-256 of the bytes we downloaded, because
InsightFace publishes no hash of its own.** ``insightface/utils/storage.py`` (the code that
fetches this pack for every user of the Python package) builds a URL and unzips it; there
is no manifest, no signature and no digest anywhere in that path. So the pin here is a
*witness* — "these are the bytes this conversion consumed on this date" — and not a
verification against an upstream claim. The distinction matters and is recorded in the
model card rather than glossed.
"""
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

# --- source pins ---------------------------------------------------------------------------

#: The official release asset. The tag ``v0.7`` is a GitHub *release* tag on
#: ``deepinsight/insightface`` and, unlike ReDimNet's literal ``latest``, is not obviously
#: mutable — but a tag is still a name and the SHA-256 below is the lock.
PACK_URL = ("https://github.com/deepinsight/insightface/releases/download/v0.7/"
            "buffalo_l.zip")
PACK_NAME = "buffalo_l.zip"
PACK_BYTES = 288_621_354
PACK_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"

#: The one member of the pack this recipe converts. The other four are a detector, two
#: landmark models and a gender/age head; only ``det_10g.onnx`` is read at all, and only to
#: BUILD FIXTURES (see ``build_fixtures.py``) — it is never converted and never published.
RECOGNITION_MEMBER = "w600k_r50.onnx"
RECOGNITION_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
RECOGNITION_BYTES = 174_383_860

DETECTOR_MEMBER = "det_10g.onnx"
DETECTOR_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"

#: Every member, so ``fetch_source.py`` can record the whole pack rather than one file.
#: A pack-level record is what lets a later reader confirm that the four files this recipe
#: does NOT convert are the ones they think they are.
PACK_MEMBERS = {
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "det_10g.onnx": DETECTOR_SHA256,
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "w600k_r50.onnx": RECOGNITION_SHA256,
}

#: The ``deepinsight/insightface`` revision whose ``ArcFaceONNX`` preprocessing and
#: ``face_align.norm_crop`` this recipe reproduces. The SAME commit ``align_oracle.py``
#: pins, so the alignment the fixtures are cut with and the alignment the Rust door is
#: goldened against are one specification.
INSIGHTFACE_REV = "ffa12d315041c0505b077c7ff057ca914bb8dc7e"

# --- the contract --------------------------------------------------------------------------

#: The CoreML feature names. Neither is the ONNX's own: that graph was traced out of
#: PyTorch 1.9 and its features are named ``input.1`` and ``683`` — a tracer's counters,
#: not a contract. ``data`` is InsightFace's own MXNet-era name for this tensor and
#: ``embedding`` is what every other coremlit embedder calls its output, so the CoreML
#: bundle reads the same as the rest of the crate. The ONNX names are recorded in the
#: manifest so the cross-platform twin can bind them.
INPUT_NAME = "data"
OUTPUT_NAME = "embedding"
ONNX_INPUT_NAME = "input.1"
ONNX_OUTPUT_NAME = "683"

BATCH = 1
CHANNELS = 3
TEMPLATE_SIZE = 112
EMBED_DIM = 512
INPUT_SHAPE = (BATCH, CHANNELS, TEMPLATE_SIZE, TEMPLATE_SIZE)
OUTPUT_SHAPE = (BATCH, EMBED_DIM)

CONTRACT = (f"{INPUT_NAME} [{', '.join(map(str, INPUT_SHAPE))}] f32 -> "
            f"{OUTPUT_NAME} [{', '.join(map(str, OUTPUT_SHAPE))}] f32 (RAW, un-normalised)")

#: Host-side preprocessing, as ``coremlit``'s ``Preprocessing`` spells it:
#: ``value = byte * scale + bias[channel]``. Read off InsightFace's ``ArcFaceONNX``
#: (``input_mean = 127.5``, ``input_std = 127.5``, ``blobFromImages(..., swapRB=True)``
#: over an OpenCV **BGR** crop, i.e. the tensor the model is fed is **RGB**), and asserted
#: numerically by ``probe_onnx_contract.py`` rather than trusted.
PREPROCESSING = {
    "order": "rgb",
    "layout": "nchw",
    "scale": 1.0 / 127.5,
    "bias": [-1.0, -1.0, -1.0],
    "equivalent": "(x - 127.5) / 127.5",
}

BUNDLE = "w600k_r50.mlmodelc"
MLPACKAGE_FP16 = "w600k_r50.mlpackage"
MLPACKAGE_FP32 = "w600k_r50_fp32.mlpackage"

# --- toolchain -----------------------------------------------------------------------------

#: coremltools 8.3.0 is not a preference — it is the version every other bundle this crate
#: ships was produced by. torch/onnx2torch carry the ONNX -> PyTorch hop coremltools no
#: longer has a front end for (``ct.converters`` exposes only libsvm/lightgbm/sklearn/
#: xgboost); ``onnxruntime`` is the parity oracle and is pinned for the same reason.
REQUIRED_TOOLCHAIN = {
    "python": "3.11",
    "numpy": "1.26.4",
    "torch": "2.5.0",
    "coremltools": "8.3.0",
    "onnx": "1.17.0",
    "onnxruntime": "1.20.1",
    "onnx2torch": "1.5.15",
    "pillow": "11.0.0",
}


def _env(name, default):
    value = os.environ.get(name)
    return value if value else default


def conv_dir() -> Path:
    d = Path(_env("ARCFACE_CONV", str(Path.home() / ".cache" / "coremlit-arcface-conv")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_dir() -> Path:
    d = conv_dir() / "source"
    d.mkdir(parents=True, exist_ok=True)
    return d


def staging_dir() -> Path:
    d = conv_dir() / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_root() -> Path:
    """The repository root, FOUND rather than counted: walk up to the directory carrying
    ``MODELS_LOCK``. A ``parents[n]`` hop count encodes this file's depth in the tree and
    fails silently when that depth changes."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "MODELS_LOCK").is_file():
            return parent
    raise SystemExit(f"no MODELS_LOCK at or above {here} — cannot locate the repository root")


def models_out_dir() -> Path:
    d = Path(_env("ARCFACE_MODELS_OUT", str(repo_root() / "Models" / "facekit")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fixtures_dir() -> Path:
    return repo_root() / "coremlit" / "tests" / "face" / "fixtures"


def onnx_path() -> Path:
    return source_dir() / RECOGNITION_MEMBER


def detector_path() -> Path:
    return source_dir() / DETECTOR_MEMBER


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_source() -> Path:
    """The ONNX, re-verified against its pin on every use.

    Verified at every entry rather than once at download: the file lives in a cache
    directory outside the repository, and a stage that trusts a previous stage's download
    is a stage that will one day convert whatever is sitting at that path."""
    path = onnx_path()
    if not path.is_file():
        raise SystemExit(f"missing {path} — run fetch_source.py first")
    got = sha256_file(path)
    if got != RECOGNITION_SHA256:
        raise SystemExit(f"{path}: sha256 {got}, pinned {RECOGNITION_SHA256}")
    return path


def observed_toolchain(keys=None):
    """The versions ACTUALLY running this recipe, after asserting each matches
    ``REQUIRED_TOOLCHAIN``. Aborts (``SystemExit``) on any mismatch, so no stage can record
    a version it did not run under.

    ``keys`` narrows the observation to the packages a stage actually imports, and narrowing
    is the same #97 rule pointed the other way: a stage that never imports torch must not
    record a torch version, and must not refuse to run because one is absent. ``None`` (the
    default) observes the whole pinned stack, which is what the conversion stages need."""
    from importlib.metadata import PackageNotFoundError, version

    wanted = REQUIRED_TOOLCHAIN if keys is None else {
        k: REQUIRED_TOOLCHAIN[k] for k in ("python", *keys)}
    observed = {"python": platform.python_version()}
    mismatches = []
    want_python = wanted["python"]
    if observed["python"].rsplit(".", 1)[0] != want_python:
        mismatches.append(f"python: observed {observed['python']}, pinned {want_python}.x")
    for key, want in wanted.items():
        if key == "python":
            continue
        try:
            got = version(key)
        except PackageNotFoundError:
            got = None
        observed[key] = got
        if got != want:
            mismatches.append(f"{key}: observed {got!r}, pinned {want!r}")
    if mismatches:
        raise SystemExit(
            "TOOLCHAIN MISMATCH — refusing to record versions that were not run:\n  "
            + "\n  ".join(mismatches))
    print("[ok] toolchain observed and matches the pins ("
          + ", ".join(f"{k} {v}" for k, v in observed.items()) + ")")
    return observed


def observed_compiler():
    """The toolchain that turns an ``.mlpackage`` into an ``.mlmodelc``. It is NOT pinned by
    the Python venv — a different Xcode compiles different bytes from identical input — so
    the manifest records which one ran."""
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    xcode = run("xcodebuild", "-version")
    return {
        "coremlcompiler": run("xcrun", "--find", "coremlcompiler"),
        "xcode": xcode.replace("\n", " ") if xcode else None,
        "macos": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
    }


def preprocess(rgb_u8):
    """The host-side preprocessing, applied exactly as ``PREPROCESSING`` declares it.

    ``rgb_u8`` is ``[n, 112, 112, 3]`` uint8 in RGB order (what ``AlignedFace`` holds).
    Returns ``[n, 3, 112, 112]`` float32. There is one implementation of this arithmetic in
    the recipe and every measurement goes through it, so a parity number and a known-pairs
    number cannot be taken under two different preprocessings."""
    import numpy as np

    x = np.asarray(rgb_u8, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (TEMPLATE_SIZE, TEMPLATE_SIZE, CHANNELS):
        raise SystemExit(f"preprocess: expected [n, 112, 112, 3] uint8, got {x.shape}")
    x = x * np.float32(PREPROCESSING["scale"])
    x = x + np.asarray(PREPROCESSING["bias"], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def cos(a, b):
    import numpy as np

    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def align_oracle():
    """``conversion/face/align_oracle.py``, imported as a module.

    The fixtures this recipe cuts are aligned by the SAME code the committed alignment
    golden is produced by, which is what makes a parity or known-pairs number a statement
    about the *embedder* rather than about two alignments that happen to be close."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "align_oracle.py"
    spec = importlib.util.spec_from_file_location("align_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["align_oracle"] = module
    spec.loader.exec_module(module)
    return module
