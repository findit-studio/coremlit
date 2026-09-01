"""Shared discipline for the ReDimNet-B5 -> CoreML conversion (identity lane, issue #123).

Source of truth: the OFFICIAL public release asset ``b5-vox2-ft_lm.pt`` from
``IDRnD/redimnet``, pinned by RELEASE TAG **and** by SHA-256 of the bytes, plus the
model source at a pinned commit. The release tag is literally named ``latest`` and is
therefore MUTABLE — a tag pin alone would let different bytes ride in on an unchanged
recipe — so ``ASSET_SHA256`` is the real lock and every load verifies it.

**Only the ``-vox2-`` lineage.** The same release also publishes ``M-``/``S-`` assets
trained on VoxBlink2 (``vb2``), whose authors assert CC BY-NC-SA 4.0 propagates to the
trained model. Those are commercially disqualifying and this recipe refuses to load one:
``verify_asset_name`` is not decoration.

The graph is the UNMODIFIED ``ReDimNetWrap.forward``. Nothing is stripped, because there
is nothing to strip: the checkpoint's tail is ``ASTP -> BatchNorm1d(4608) -> Linear(4608,
192)`` with ``emb_bn=False`` and ``num_classes=None``, so the model already emits a RAW
192-d vector (measured ||e|| ~ 19, not 1). That matches coremlit's embedder contract
exactly — ``src/audio/speaker/embed/mod.rs``: "L2 normalization is a HIGHER-level concern
(``Embedding::normalize_from``)". ``assert_raw_tail`` proves it on every run rather than
trusting this paragraph.

Paths are env-driven (no hardcoded scratchpad):
  REDIMNET_CONV        working dir holding ``src/`` (asset + pinned model source) and
                       ``staging`` (default: ``~/.cache/coremlit-redimnet-conv``).
  REDIMNET_MODELS_OUT  where the fp16 ``.mlmodelc`` bundle is staged (default: the repo's
                       gitignored ``Models/redimnet``).
"""
import hashlib
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

# --- pinned source ------------------------------------------------------------------------
# The weights asset. `ASSET_BYTES` is a cheap tripwire; `ASSET_SHA256` is the lock.
SOURCE_REPO = "IDRnD/redimnet"
SOURCE_RELEASE_TAG = "latest"          # MUTABLE upstream tag — see the module doc.
ASSET_NAME = "b5-vox2-ft_lm.pt"
ASSET_URL = f"https://github.com/{SOURCE_REPO}/releases/download/{SOURCE_RELEASE_TAG}/{ASSET_NAME}"
ASSET_BYTES = 31_174_382
ASSET_SHA256 = "8b0c11bbf5a3a8bb39e5c072c4192d0b694d8c447cf126d4cd3c7346a04b39c8"

# The MODEL SOURCE (`redimnet/` python package) at a pinned commit. A checkpoint is only
# half the provenance: `ReDimNetWrap` is reconstructed from `model_config`, so the code
# that reconstructs it is part of what produced these weights' outputs.
SOURCE_CODE_URL = f"https://github.com/{SOURCE_REPO}.git"
SOURCE_CODE_REV = "ce039a624cb99fe127702ceb94c6080090e5032f"

# The checkpoint's own `model_config`, asserted entry for entry at load. A silently
# different config would still load (the state dict would mismatch and `load_state_dict`
# would catch most of it) but these are the entries the CONTRACT depends on, so they are
# checked directly rather than inferred from a successful load.
EXPECTED_CONFIG = {
    "C": 32,
    "F": 72,
    "block_1d_type": "conv+att",
    "block_2d_type": "basic_resnet_fwse",
    "emb_bn": False,
    "embed_dim": 192,
    "global_context_att": True,
    "group_divisor": 16,
    "hop_length": 240,
    "out_channels": None,
    "pooling_func": "ASTP",
}

# --- frozen contract ----------------------------------------------------------------------
SAMPLE_RATE = 16_000
# 6 s. THE window decision — justified in README.md "The window length"; not a default.
WINDOW_SAMPLES = 96_000
HOP_LENGTH = 240
# torchaudio MelSpectrogram(center=True): 1 + n_samples // hop.
N_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH        # 401
N_MELS = 72
EMBED_DIM = 192
INPUT_NAME = "mel"
OUTPUT_NAME = "embedding"
CONTRACT = (f"{INPUT_NAME}[1, {N_MELS}, {N_FRAMES}] f32 -> {OUTPUT_NAME}[1, {EMBED_DIM}] f32 "
            f"(RAW, un-normalized; L2 by the caller). The mel front end runs in the CALLER.")

# THE FRONT END THE CALLER MUST REPRODUCE, parameter for parameter.
#
# The graph starts at the mel, not at the waveform, and that is a MEASURED decision rather
# than a stylistic one: the waveform-in variant converts cleanly and is exact in fp32, but
# its in-graph power spectrogram exceeds fp16's dynamic range at BOTH ends, and CoreML's
# default `All` placement sends it to the ANE, where fp32 does not exist. Measured worst
# cosine against the fp32 reference, waveform-in fp16: CpuOnly 0.930, CpuAndGpu 0.947,
# All 0.277, CpuAndNeuralEngine 0.277 — wrong on EVERY arm, not merely imprecise. With the
# same weights behind a mel-in contract: 0.9986 / 0.9999 / 0.9993 / 0.9993. Reproduce with
# `probe_waveform_contract.py`. `conversion/ced` made the same call for the same reason
# ("The log-mel front-end runs in Rust (MelExtractor), so the graph starts at the mel").
#
# Every entry below is READ OUT OF the checkpoint's own `MelBanks` construction, not
# copied from a paper. `hop_length` comes from `model_config`; the rest are `MelBanks`
# defaults, which is why they are asserted at load by `assert_front_end`.
MEL_FRONT_END = {
    "pre_emphasis": {"coefficient": 0.97, "pad": "reflect, 1 sample on the left",
                     "formula": "y[n] = x[n] - 0.97 * x[n-1]"},
    "stft": {"n_fft": 512, "win_length": 400, "hop_length": HOP_LENGTH,
             "window": "torch.hamming_window(400, periodic=True) zero-padded to 512",
             "center": True, "pad_mode": "reflect", "power": 2.0, "normalized": False},
    "mel_filterbank": {"n_mels": N_MELS, "f_min": 20.0, "f_max": 7600.0,
                       "norm": None, "mel_scale": "htk"},
    "log": {"epsilon": 1e-6, "formula": "log(power + 1e-6), natural log"},
    "spec_norm": {"mode": "mn", "formula": "x - mean(x, dim=time, keepdim=True)",
                  "note": "per-mel-bin mean over the 401 frames, subtracted"},
    "normalize_audio": False,   # MelBanks(norm_signal=False)
    "spec_augment": False,      # eval mode; do_spec_aug=False in this checkpoint
}

# --- toolchain ----------------------------------------------------------------------------
# `python` is a MAJOR.MINOR pin; every other entry is exact. `observed_toolchain()` reads
# what is ACTUALLY running and refuses to continue on a mismatch, so nothing can record a
# version nobody ran (issue #97's discipline, borrowed from conversion/granite).
#
# coremltools 8.3.0 is not a preference: it is the version that produced the graph this
# crate already ships (`Models/speakerkit/wespeaker.mlmodelc/model.mil` records
# `coremltools-version = "8.3.0"`), so a new audio artifact converted with a different
# major would be the only one in the tree whose backend passes nobody has seen.
# torch 2.5.0 is coremltools 8.3.0's most recent TESTED torch; 2.5.1 converts but makes
# coremltools print "has not been tested with coremltools", and a recipe should not ship
# a warning it could have removed.
REQUIRED_TOOLCHAIN = {
    "python": "3.11",
    "torch": "2.5.0",
    "torchaudio": "2.5.0",
    "coremltools": "8.3.0",
    "numpy": "1.26.4",
}


def _env(name, default=None):
    return os.environ.get(name, default)


def conv_dir() -> Path:
    d = Path(_env("REDIMNET_CONV", str(Path.home() / ".cache" / "coremlit-redimnet-conv")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def src_dir() -> Path:
    d = conv_dir() / "src"
    d.mkdir(parents=True, exist_ok=True)
    return d


def staging_dir() -> Path:
    d = conv_dir() / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_root() -> Path:
    """The repository root, FOUND rather than counted: walk up to the directory carrying
    ``MODELS_LOCK``. A ``parents[n]`` hop count encodes this file's depth in the tree and
    fails silently when that depth changes (run_ced.sh already makes this argument for the
    shell side; the Python side deserves the same)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "MODELS_LOCK").is_file():
            return parent
    raise SystemExit(f"no MODELS_LOCK at or above {here} — cannot locate the repository root")


def models_out_dir() -> Path:
    d = Path(_env("REDIMNET_MODELS_OUT", str(repo_root() / "Models" / "redimnet")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def observed_toolchain():
    """Return the versions ACTUALLY running this recipe, after asserting each matches
    ``REQUIRED_TOOLCHAIN``. Aborts (SystemExit) on any mismatch.

    The manifest must never record a CLAIMED version. A venv that resolved coremltools
    8.2 would otherwise complete the whole recipe and then be written down as 8.3.0 — a
    provenance record nobody can replay, which is exactly the defect issue #97 names. The
    returned dict carries the observed strings (python at full MAJOR.MINOR.PATCH)."""
    from importlib.metadata import PackageNotFoundError, version

    observed = {"python": platform.python_version()}
    mismatches = []
    want_python = REQUIRED_TOOLCHAIN["python"]
    if observed["python"].rsplit(".", 1)[0] != want_python:
        mismatches.append(f"python: observed {observed['python']}, pinned {want_python}.x")
    for key, want in REQUIRED_TOOLCHAIN.items():
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
    print(f"[ok] toolchain observed and matches the pins (python {observed['python']}, "
          f"coremltools {observed['coremltools']}, torch {observed['torch']})")
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


# --- source acquisition -------------------------------------------------------------------
def verify_asset_name(name: str) -> None:
    """Refuse any checkpoint outside the ``-vox2-`` lineage.

    ``IDRnD/redimnet``'s release publishes ``M-vb2+vox2+cnc-ft_mix.pt`` and friends beside
    the vox2 rows. VoxBlink2's authors state the licence propagates to trained models
    ("The license of the model is also CC BY-NC-SA 4.0, no commercial application is
    allowed"), so a ``vb2``/``cnc`` checkpoint is commercially disqualifying. The
    corpus-parity argument this artifact rests on — VoxCeleb2-dev, the same lineage
    coremlit already ships — covers the vox2 rows and NOTHING else."""
    if "-vox2-" not in name or "vb2" in name or "cnc" in name:
        raise SystemExit(
            f"REFUSING {name}: only the -vox2- lineage is commercially usable. "
            f"vb2/VoxBlink2 and cnc checkpoints are CC BY-NC-SA 4.0 (no commercial use).")


def asset_path() -> Path:
    return src_dir() / ASSET_NAME


def download() -> Path:
    """Fetch (once) and SHA-verify the pinned release asset."""
    verify_asset_name(ASSET_NAME)
    dst = asset_path()
    if not dst.is_file():
        print(f"[..] downloading {ASSET_URL}")
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(dst), ASSET_URL], check=True)
    size = dst.stat().st_size
    if size != ASSET_BYTES:
        raise SystemExit(f"{dst}: {size} bytes, pinned {ASSET_BYTES}")
    got = sha256_file(dst)
    if got != ASSET_SHA256:
        raise SystemExit(f"{dst}: sha256 {got}, pinned {ASSET_SHA256}")
    print(f"[ok] {ASSET_NAME}: {size} bytes, sha256 {got[:16]}… matches the pin")
    return dst


def source_code_dir() -> Path:
    """Clone (once) and pin the ``redimnet`` python package that reconstructs the model."""
    d = src_dir() / "redimnet-src"
    if not (d / ".git").is_dir():
        print(f"[..] cloning {SOURCE_CODE_URL}")
        subprocess.run(["git", "clone", "--quiet", SOURCE_CODE_URL, str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "--quiet", SOURCE_CODE_REV], check=True)
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != SOURCE_CODE_REV:
        raise SystemExit(f"{d}: HEAD {head}, pinned {SOURCE_CODE_REV}")
    print(f"[ok] model source pinned at {head[:12]}…")
    return d


# --- model ---------------------------------------------------------------------------------
def assert_raw_tail(model) -> None:
    """Prove the checkpoint emits a RAW vector — the fact coremlit's contract depends on.

    Read, not assumed, and checked at BOTH ends: structurally (no ``bn2``, no ``cls_head``,
    the tail is ``Linear(4608, 192)``) and numerically (the caller measures ||e||). If a
    future ``-ft_lm`` asset ever grew an L2 or an ``emb_bn``, this is what refuses it, and
    stripping it would then become a deliberate, visible edit rather than a silent one."""
    import torch.nn as nn

    problems = []
    if getattr(model, "bn2", None) is not None:
        problems.append("emb_bn is set: an extra BatchNorm1d sits after `linear`")
    if getattr(model, "cls_head", None) is not None:
        problems.append("cls_head is present: this is a classifier checkpoint, not an embedder")
    lin = getattr(model, "linear", None)
    if not isinstance(lin, nn.Linear) or lin.out_features != EMBED_DIM:
        problems.append(f"tail is {type(lin).__name__}, expected Linear(*, {EMBED_DIM})")
    # `forward` must end at `linear` — no functional normalize anywhere in the source.
    import inspect
    src = inspect.getsource(type(model).forward)
    for needle in ("normalize", "norm(", "/ out.norm"):
        if needle in src:
            problems.append(f"forward mentions {needle!r} — inspect for an L2 tail")
    if problems:
        raise SystemExit("RAW-TAIL ASSERTION FAILED:\n  " + "\n  ".join(problems))
    print("[ok] tail is ASTP -> BatchNorm1d -> Linear(4608, 192): RAW output, nothing to strip")


def load_model():
    """Build ``ReDimNetWrap`` from the pinned asset's own ``model_config`` and load its
    weights, asserting the config entries the contract rests on and a total key match."""
    import sys

    import torch

    ckpt_path = download()
    code = source_code_dir()
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from redimnet.model import ReDimNetWrap  # noqa: E402  (path set above)

    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = blob["model_config"]
    bad = {k: (cfg.get(k), v) for k, v in EXPECTED_CONFIG.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit("model_config MISMATCH — the pinned asset is not the one this "
                         "recipe describes:\n  " +
                         "\n  ".join(f"{k}: observed {g!r}, expected {w!r}"
                                     for k, (g, w) in bad.items()))
    model = ReDimNetWrap(**cfg)
    res = model.load_state_dict(blob["state_dict"])
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit(f"state_dict mismatch: missing={res.missing_keys} "
                         f"unexpected={res.unexpected_keys}")
    model.eval()
    assert_raw_tail(model)
    assert_front_end(model)
    return model, cfg


def assert_front_end(model) -> None:
    """Prove the caller-side mel spec in ``MEL_FRONT_END`` is the one this checkpoint was
    built with, by reading the live ``MelBanks`` rather than trusting the table.

    This is the single highest-consequence claim in the recipe: the graph no longer
    computes its own features, so a wrong parameter here is silently wrong EMBEDDINGS in
    the Rust door, with no shape error to catch it."""
    spec = model.spec
    mel_t = spec.torchfbank[2]
    problems = []

    def want(label, got, expected):
        if got != expected:
            problems.append(f"{label}: observed {got!r}, recorded {expected!r}")

    pre = spec.torchfbank[1]
    want("pre_emphasis.coefficient", float(pre.coef), MEL_FRONT_END["pre_emphasis"]["coefficient"])
    st = MEL_FRONT_END["stft"]
    want("stft.n_fft", int(mel_t.spectrogram.n_fft), st["n_fft"])
    want("stft.win_length", int(mel_t.spectrogram.win_length), st["win_length"])
    want("stft.hop_length", int(mel_t.spectrogram.hop_length), st["hop_length"])
    want("stft.center", bool(mel_t.spectrogram.center), st["center"])
    want("stft.pad_mode", str(mel_t.spectrogram.pad_mode), st["pad_mode"])
    want("stft.power", float(mel_t.spectrogram.power), st["power"])
    want("stft.normalized", bool(mel_t.spectrogram.normalized), st["normalized"])
    fb = MEL_FRONT_END["mel_filterbank"]
    want("mel.n_mels", int(mel_t.mel_scale.n_mels), fb["n_mels"])
    want("mel.f_min", float(mel_t.mel_scale.f_min), fb["f_min"])
    want("mel.f_max", float(mel_t.mel_scale.f_max), fb["f_max"])
    want("mel.norm", mel_t.mel_scale.norm, fb["norm"])
    want("mel.mel_scale", str(mel_t.mel_scale.mel_scale), fb["mel_scale"])
    want("mel.sample_rate", int(mel_t.mel_scale.sample_rate), SAMPLE_RATE)
    # The window is a buffer, so it is compared by VALUE against the recorded recipe.
    import torch
    win = mel_t.spectrogram.window
    ref = torch.hamming_window(st["win_length"], periodic=True)
    if win.shape[0] != st["win_length"] or not torch.allclose(win, ref, atol=1e-6):
        problems.append(f"stft.window: observed shape {tuple(win.shape)} does not match "
                        f"hamming_window({st['win_length']}, periodic=True)")
    if problems:
        raise SystemExit("FRONT-END ASSERTION FAILED — the caller-side mel spec in "
                         "MEL_FRONT_END does not describe this checkpoint:\n  "
                         + "\n  ".join(problems))
    print(f"[ok] front end matches MEL_FRONT_END: pre-emph 0.97, n_fft 512/win 400/hop "
          f"{HOP_LENGTH}, hamming, {N_MELS} mels 20-7600 Hz htk, log(x+1e-6), mean-normalized")


class MelToEmbedding:
    """Factory for the traced sub-forward: ``mel -> embedding``, the EXACT tail of the
    unmodified ``ReDimNetWrap.forward`` with only ``self.spec`` removed."""

    @staticmethod
    def build(model):
        import torch.nn as nn

        class _W(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                # `ReDimNetWrap.forward` after `self.spec(x)`: unsqueeze to (B,1,F,T),
                # backbone, bn(pool(.)), linear. `tf_optimized_arch` is False for this
                # checkpoint and `bn2`/`cls_head` are absent (asserted by assert_raw_tail).
                x = x.unsqueeze(1)
                out = self.m.backbone(x)
                out = self.m.bn(self.m.pool(out))
                return self.m.linear(out)

        return _W(model).eval()


def mel_for_waveform(model, wav_1xn):
    """The caller-side front end, computed by the checkpoint's OWN ``MelBanks``. The Rust
    door must reproduce this from ``MEL_FRONT_END``; the goldens it is checked against
    should come from here."""
    import torch

    with torch.no_grad():
        return model.spec(torch.as_tensor(wav_1xn))


# --- numerics ------------------------------------------------------------------------------
def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def worst_update(worst: float, c: float) -> float:
    """NaN-poisoning min: once a non-finite comparison appears the worst stays NaN, so a
    breach cannot be averaged away by a later good clip."""
    if worst != worst or c != c:
        return float("nan")
    return min(worst, c)
