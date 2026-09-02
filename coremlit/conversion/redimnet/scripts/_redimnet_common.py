"""Shared discipline for the ReDimNet -> CoreML conversion (identity lane, issue #123).

ONE recipe, several ARTIFACTS. The size (and training stage) being converted is a
:class:`Variant` selected by ``REDIMNET_VARIANT`` — ``b5``, ``b2`` or ``b2_ptn`` — and
there is deliberately no default: a recipe that silently converted one checkpoint when
asked for another would be the provenance defect issue #97 names, so :func:`variant`
refuses to run unselected. Everything a variant changes — the release asset and its
SHA-256, the checkpoint's ``model_config``, the bundle name, the pooled width — lives in
:data:`VARIANTS`; everything a variant does NOT change — the front end, the window, the
I/O contract — is a module constant and is asserted against every checkpoint at load.

Source of truth per variant: the OFFICIAL public release asset (``b5-vox2-ft_lm.pt``,
``b2-vox2-ft_lm.pt``, ``b2-vox2-ptn.pt``) from ``IDRnD/redimnet``, pinned by RELEASE TAG
**and** by SHA-256 of the bytes, plus the model source at a pinned commit. The release
tag is literally named ``latest`` and is therefore MUTABLE — a tag pin alone would let
different bytes ride in on an unchanged recipe — so the per-variant ``asset_sha256`` is
the real lock and every load verifies it.

**Only the ``-vox2-`` lineage.** The same release also publishes ``M-``/``S-`` assets
trained on VoxBlink2 (``vb2``), whose authors assert CC BY-NC-SA 4.0 propagates to the
trained model. Those are commercially disqualifying and this recipe refuses to load one:
``verify_asset_name`` is not decoration, and it runs on every variant's asset name.

The graph is the UNMODIFIED ``ReDimNetWrap.forward``. Nothing is stripped, because there
is nothing to strip: every checkpoint's tail is ``ASTP -> BatchNorm1d(pooled) ->
Linear(pooled, 192)`` (``pooled`` is 4608 for B5 and 2304 for B2) with ``emb_bn=False``
and ``num_classes=None``, so the model already emits a RAW 192-d vector (measured
``||e|| ~ 19`` for B5, not 1). That matches coremlit's embedder contract exactly —
``src/audio/speaker/embed/mod.rs``: "L2 normalization is a HIGHER-level concern
(``Embedding::normalize_from``)". ``assert_raw_tail`` proves it on every run rather than
trusting this paragraph.

Paths are env-driven (no hardcoded scratchpad):
  REDIMNET_VARIANT     which checkpoint: ``b5`` | ``b2`` | ``b2_ptn`` (REQUIRED).
  REDIMNET_CONV        working dir holding ``src/`` (assets + pinned model source) and
                       ``staging`` (default: ``~/.cache/coremlit-redimnet-conv``).
  REDIMNET_MODELS_OUT  where the fp16 ``.mlmodelc`` bundles are staged (default: the
                       repo's gitignored ``Models/redimnet``).
"""
import dataclasses
import hashlib
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

# --- pinned source ------------------------------------------------------------------------
SOURCE_REPO = "IDRnD/redimnet"
SOURCE_RELEASE_TAG = "latest"          # MUTABLE upstream tag — see the module doc.
ASSET_URL_BASE = f"https://github.com/{SOURCE_REPO}/releases/download/{SOURCE_RELEASE_TAG}"

# The MODEL SOURCE (`redimnet/` python package) at a pinned commit. A checkpoint is only
# half the provenance: `ReDimNetWrap` is reconstructed from `model_config`, so the code
# that reconstructs it is part of what produced these weights' outputs. One commit for
# every variant: the same package rebuilds every size.
SOURCE_CODE_URL = f"https://github.com/{SOURCE_REPO}.git"
SOURCE_CODE_REV = "ce039a624cb99fe127702ceb94c6080090e5032f"

# `model_config` entries EVERY variant must carry: the ones the CONTRACT rests on. They
# are asserted at load for each variant, beside that variant's own size entries.
SHARED_CONFIG = {
    "F": 72,
    "block_1d_type": "conv+att",
    "emb_bn": False,
    "embed_dim": 192,
    "global_context_att": True,
    "hop_length": 240,
    "out_channels": None,
    "pooling_func": "ASTP",
}


@dataclasses.dataclass(frozen=True)
class Variant:
    """One ReDimNet checkpoint this recipe knows how to convert.

    ``key`` is what ``REDIMNET_VARIANT`` selects. ``bundle`` names the staged
    ``.mlpackage``/``.mlmodelc`` and every per-variant staging file. ``training_crop_s``
    is the crop length the checkpoint's LAST training stage used — 6 s for a ``-ft_lm``
    large-margin fine-tune, 2 s for a ``-ptn`` pretrain (arXiv 2407.18223 §3.2) — and it
    is recorded because the door's fixed 6 s window matches one regime and not the other;
    that mismatch belongs in the manifest rather than in a reader's memory.
    ``published_metrics`` is whether ANY upstream evaluation of the checkpoint exists:
    the ``ptn`` rows of ``IDRnD/redimnet``'s ``EVALUATION.md`` are ``-`` in every column.
    ``pooled_dim`` is the width the tail's ``Linear`` takes, asserted by
    ``assert_raw_tail``."""
    key: str
    asset: str
    asset_bytes: int
    asset_sha256: str
    bundle: str
    title: str
    training_crop_s: int
    published_metrics: bool
    pooled_dim: int
    expected_config: dict

    @property
    def asset_url(self) -> str:
        return f"{ASSET_URL_BASE}/{self.asset}"

    @property
    def mlpackage(self) -> str:
        return f"{self.bundle}.mlpackage"

    @property
    def mlpackage_fp32(self) -> str:
        return f"{self.bundle}_fp32.mlpackage"

    @property
    def mlmodelc(self) -> str:
        return f"{self.bundle}.mlmodelc"

    def staging_file(self, suffix: str) -> Path:
        """A per-variant file under ``staging/`` (``producer.json``, ``placement.json``,
        ``sweep_inputs.npy``), so two variants converted into one working dir never
        overwrite each other's records."""
        return staging_dir() / f"{self.bundle}_{suffix}"


# The per-size `model_config` entries, read out of each archive rather than transcribed
# from a paper: B5 is `basic_resnet_fwse` at C=32 (32 fwSE gates, the op class the census
# suspected of being ANE-hostile); B2 is `convnext_like` at C=16 with none. `stages_setup`
# is not pinned here — a total state-dict key match already refuses a mismatched
# architecture — but the entries that decide WHICH op classes the placement sweep measures
# are.
_B5_CONFIG = {**SHARED_CONFIG, "C": 32, "block_2d_type": "basic_resnet_fwse", "group_divisor": 16}
_B2_CONFIG = {**SHARED_CONFIG, "C": 16, "block_2d_type": "convnext_like", "group_divisor": 4}

VARIANTS = {
    "b5": Variant(
        key="b5",
        asset="b5-vox2-ft_lm.pt",
        asset_bytes=31_174_382,
        asset_sha256="8b0c11bbf5a3a8bb39e5c072c4192d0b694d8c447cf126d4cd3c7346a04b39c8",
        bundle="redimnet_b5",
        title="ReDimNet-B5 (vox2, ft_lm)",
        training_crop_s=6,
        published_metrics=True,
        pooled_dim=4608,
        expected_config=_B5_CONFIG,
    ),
    "b2": Variant(
        key="b2",
        asset="b2-vox2-ft_lm.pt",
        asset_bytes=20_582_650,
        asset_sha256="c9b6bb2f6747caa28a41eaf2e372d66b0d1563baef186d18f5e99abd5e71e06f",
        bundle="redimnet_b2",
        title="ReDimNet-B2 (vox2, ft_lm)",
        training_crop_s=6,
        published_metrics=True,
        pooled_dim=2304,
        expected_config=_B2_CONFIG,
    ),
    "b2_ptn": Variant(
        key="b2_ptn",
        asset="b2-vox2-ptn.pt",
        asset_bytes=20_581_530,
        asset_sha256="c18a42926878bc8ac079623fbf36f0bc8054cda1199e96fbe1a3f8e131796647",
        bundle="redimnet_b2_ptn",
        title="ReDimNet-B2 (vox2, ptn: 2 s pretrain, no published metrics)",
        training_crop_s=2,
        published_metrics=False,
        pooled_dim=2304,
        expected_config=_B2_CONFIG,
    ),
}


# The ONE variant that is registered — `MODELS_LOCK`, a licence row, the identity door's
# gated tests and the committed goldens all name it. B2 and B2-ptn are converted, measured
# and preserved in the artifact repository (README.md, "B2: converted, measured, not
# registered") but deliberately not registered: the short-segment experiment on issue #123
# left B2 with no lane, and a registered artifact nothing consumes is maintenance.
REGISTERED_VARIANT = "b5"


def variant() -> Variant:
    """The variant ``REDIMNET_VARIANT`` selects. NO default, on purpose: an unselected
    recipe refuses rather than converting whichever size a default happened to name."""
    key = os.environ.get("REDIMNET_VARIANT")
    if not key:
        raise SystemExit(
            f"REDIMNET_VARIANT is unset — select one of {sorted(VARIANTS)} "
            f"(run_redimnet.sh <variant>). There is no default: a recipe that converts a size "
            f"nobody asked for records provenance nobody can replay.")
    if key not in VARIANTS:
        raise SystemExit(f"REDIMNET_VARIANT={key!r} is not one of {sorted(VARIANTS)}")
    return VARIANTS[key]


# --- frozen contract ----------------------------------------------------------------------
SAMPLE_RATE = 16_000
# 6 s. THE window decision — justified in README.md "The window length"; not a default.
# One window for every variant: the graph's input shape is the contract, and a `ptn`
# checkpoint trained on 2 s crops is fed the same 401 frames — a train/inference
# mismatch the manifest records (`training_crop_s`) rather than a second contract.
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
# cosine against the fp32 reference, waveform-in fp16 (B5): CpuOnly 0.930, CpuAndGpu
# 0.947, All 0.277, CpuAndNeuralEngine 0.277 — wrong on EVERY arm, not merely imprecise.
# With the same weights behind a mel-in contract: 0.9986 / 0.9999 / 0.9993 / 0.9993.
# Reproduce with `probe_waveform_contract.py`. `conversion/ced` made the same call for the
# same reason ("The log-mel front-end runs in Rust (MelExtractor), so the graph starts at
# the mel").
#
# Every entry below is READ OUT OF the checkpoint's own `MelBanks` construction, not
# copied from a paper. `hop_length` comes from `model_config`; the rest are `MelBanks`
# defaults, which is why they are asserted at load by `assert_front_end` — for EVERY
# variant, since the whole point of one Rust front end is that every checkpoint behind
# the door was built on the same one.
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
    corpus-parity argument every artifact here rests on — VoxCeleb2-dev, the same lineage
    coremlit already ships — covers the vox2 rows and NOTHING else."""
    if "-vox2-" not in name or "vb2" in name or "cnc" in name:
        raise SystemExit(
            f"REFUSING {name}: only the -vox2- lineage is commercially usable. "
            f"vb2/VoxBlink2 and cnc checkpoints are CC BY-NC-SA 4.0 (no commercial use).")


def asset_path(v: Variant) -> Path:
    return src_dir() / v.asset


def download(v: Variant) -> Path:
    """Fetch (once) and SHA-verify the pinned release asset of ``v``."""
    verify_asset_name(v.asset)
    dst = asset_path(v)
    if not dst.is_file():
        print(f"[..] downloading {v.asset_url}")
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(dst), v.asset_url], check=True)
    size = dst.stat().st_size
    if size != v.asset_bytes:
        raise SystemExit(f"{dst}: {size} bytes, pinned {v.asset_bytes}")
    got = sha256_file(dst)
    if got != v.asset_sha256:
        raise SystemExit(f"{dst}: sha256 {got}, pinned {v.asset_sha256}")
    print(f"[ok] {v.asset}: {size} bytes, sha256 {got[:16]}… matches the pin")
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
def assert_raw_tail(model, v: Variant) -> None:
    """Prove the checkpoint emits a RAW vector — the fact coremlit's contract depends on.

    Read, not assumed, and checked at BOTH ends: structurally (no ``bn2``, no ``cls_head``,
    the tail is ``Linear(pooled, 192)`` at the variant's own pooled width) and numerically
    (the caller measures ||e||). If a future asset ever grew an L2 or an ``emb_bn``, this
    is what refuses it, and stripping it would then become a deliberate, visible edit
    rather than a silent one."""
    import torch.nn as nn

    problems = []
    if getattr(model, "bn2", None) is not None:
        problems.append("emb_bn is set: an extra BatchNorm1d sits after `linear`")
    if getattr(model, "cls_head", None) is not None:
        problems.append("cls_head is present: this is a classifier checkpoint, not an embedder")
    lin = getattr(model, "linear", None)
    if not isinstance(lin, nn.Linear) or lin.out_features != EMBED_DIM:
        problems.append(f"tail is {type(lin).__name__}, expected Linear(*, {EMBED_DIM})")
    elif lin.in_features != v.pooled_dim:
        problems.append(f"tail is Linear({lin.in_features}, {EMBED_DIM}); {v.key} pins a pooled "
                        f"width of {v.pooled_dim}")
    # `forward` must end at `linear` — no functional normalize anywhere in the source.
    import inspect
    src = inspect.getsource(type(model).forward)
    for needle in ("normalize", "norm(", "/ out.norm"):
        if needle in src:
            problems.append(f"forward mentions {needle!r} — inspect for an L2 tail")
    if problems:
        raise SystemExit("RAW-TAIL ASSERTION FAILED:\n  " + "\n  ".join(problems))
    print(f"[ok] tail is ASTP -> BatchNorm1d -> Linear({v.pooled_dim}, {EMBED_DIM}): RAW "
          f"output, nothing to strip")


def load_model(v: Variant | None = None):
    """Build ``ReDimNetWrap`` from the selected variant's asset and its own ``model_config``,
    and load its weights, asserting the config entries the contract rests on, the variant's
    own size entries, and a total key match. Returns ``(model, cfg, variant)``."""
    import sys

    import torch

    v = v or variant()
    ckpt_path = download(v)
    code = source_code_dir()
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from redimnet.model import ReDimNetWrap  # noqa: E402  (path set above)

    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = blob["model_config"]
    bad = {k: (cfg.get(k), w) for k, w in v.expected_config.items() if cfg.get(k) != w}
    if bad:
        raise SystemExit(f"model_config MISMATCH — the pinned asset is not the {v.key} this "
                         "recipe describes:\n  " +
                         "\n  ".join(f"{k}: observed {g!r}, expected {w!r}"
                                     for k, (g, w) in bad.items()))
    model = ReDimNetWrap(**cfg)
    res = model.load_state_dict(blob["state_dict"])
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit(f"state_dict mismatch: missing={res.missing_keys} "
                         f"unexpected={res.unexpected_keys}")
    model.eval()
    assert_raw_tail(model, v)
    assert_front_end(model)
    return model, cfg, v


def assert_front_end(model) -> None:
    """Prove the caller-side mel spec in ``MEL_FRONT_END`` is the one this checkpoint was
    built with, by reading the live ``MelBanks`` rather than trusting the table.

    This is the single highest-consequence claim in the recipe: the graph no longer
    computes its own features, so a wrong parameter here is silently wrong EMBEDDINGS in
    the Rust door, with no shape error to catch it. It runs for EVERY variant, because one
    Rust front end serving every artifact behind the door is only sound if every
    checkpoint was built on the same one."""
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


def front_end_tables(model):
    """The two saved buffers the front end is built from — ``spectrogram.window`` (fp32,
    ``[win_length]``) and ``mel_scale.fb`` transposed to mel-major ``[n_mels, n_freqs]`` —
    as float32 arrays. What the committed goldens pin, and what every variant must share
    byte for byte for one Rust front end to serve them all."""
    mel_t = model.spec.torchfbank[2]
    window = mel_t.spectrogram.window.detach().numpy().astype(np.float32)
    fbank = mel_t.mel_scale.fb.detach().numpy().T.astype(np.float32)
    return window, fbank


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
                # backbone, bn(pool(.)), linear. `tf_optimized_arch` is False for these
                # checkpoints and `bn2`/`cls_head` are absent (asserted by assert_raw_tail).
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
