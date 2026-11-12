"""Pins and helpers shared by the EmbeddingGemma 2 recipe's scripts.

Every input is pinned twice: by revision, and by the SHA-256 of each file the
scripts read. ``verify_source`` and ``verify_staged`` check the digests on
every run and refuse a mismatch, so no golden is written from bytes nobody
pinned. ``observed_toolchain`` refuses an environment that differs from
``REQUIRED_TOOLCHAIN``, so no record names a version nobody ran.
"""
import hashlib
import os
import platform
import subprocess
import sys

# google/embeddinggemma-2, the original checkpoint: R1's reference and the
# fp32 audio conversion's source.
MODEL_ID = "google/embeddinggemma-2"
REV = "f9972ac22e696a105c482202646979e33e4b5a1e"

# Every file of the checkpoint the recipe reads, by SHA-256. The weights are
# bf16; the others carry the graph's shape, the tokenizer and
# sentence-transformers' pipeline (mean pooling with the prompt included, then
# Normalize).
SOURCE_FILES = {
    "model.safetensors": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
    "config.json": "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
    "config_sentence_transformers.json": "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5",
    "modules.json": "3d02572a0455b832de67fb8e63a54981bc7e8b46e337c95e917bd8122a533bfd",
    "1_Pooling/config.json": "8759bdf7c77efc7df7723f64856a593c8943b71ee38baf2a88771fbaf78438f9",
    "2_Normalize/config.json": "cdb09dfca347a56aa2d691744e38d5ad3c7cbc2834e7181272b9a15328b82524",
    "sentence_bert_config.json": "b1bcd9f2dce3ae863b359e87d0710b5dbc3314a59ecb4e2f97c7778fc8e4b228",
    "processor_config.json": "168f6a08522f3ce5dea596d94d003af2fd691742d4f41fe1f9d8cce76bfbf69c",
    "preprocessor_config.json": "ea2ae257e901064abdd98dceb19f2b0da06af600bed15e0f99f5c85c37ee9d78",
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "tokenizer_config.json": "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874",
    "chat_template.jinja": "4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e",
}

# FluidInference/embeddinggemma-2-coreml, the conversion coremlit runs: the
# files of its tree the goldens read (the token table for the inputs_embeds
# digests, the position table for its provenance, the fp16 audio package for
# the fp32 conversion's contract).
STAGED_ID = "FluidInference/embeddinggemma-2-coreml"
STAGED_REV = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"
STAGED_FILES = {
    "embeddings.bf16": "56ebbdbfd706c827f3fae1d97aa39135c7ddc0f1e6d9b12f8565f84bbfbda877",
    "position_embeddings.f16": "17de9480acc23638c0634a97ed89e51ed664433256d7d6f109587dcd91c79124",
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "config.json": "30668aba89f54a200a3fa38913f21e2a2f18aba7215de67c1806542a75b9629b",
}

# The tables' geometry, as config.json states it.
VOCAB_SIZE = 262_144
HIDDEN_SIZE = 512
EMBEDDING_DIM = 768
EMBED_SCALE_BITS = 0x41B504F3
PAD_ID, EOS_ID, BOS_ID = 0, 1, 2
BOI_ID, BOA_ID, EOI_ID, EOA_ID = 255_999, 256_000, 258_882, 258_883
TEXT_FUNCTIONS = (32, 48, 64, 128, 256, 512)
MAX_TOKENS = 512
VISION_BUDGETS = (70, 140, 280)
PATCH_SIZE = 16
POOLING_KERNEL = 3
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 160_000
MIN_TAIL_SAMPLES = 8_000
MIN_WINDOW_SAMPLES = 161

# Every version the recipe runs, pinned. Python is pinned to MAJOR.MINOR; every
# other entry exactly.
REQUIRED_TOOLCHAIN = {
    "python": "3.11",
    "torch": "2.14.1",
    "torchvision": "0.29.2",
    "transformers": "5.22.1",
    "sentence_transformers": "6.2.0",
    "tokenizers": "0.23.3",
    "coremltools": "9.0",
    "numpy": "2.4.6",
    "pillow": "13.0.0",
    "safetensors": "0.8.0",
}

_DISTRIBUTION_NAMES = {"sentence_transformers": "sentence-transformers"}


def env(name, required=True):
    """The environment variable ``name``; a required one that is unset stops
    the run."""
    value = os.environ.get(name)
    if required and not value:
        raise SystemExit(f"required environment variable {name} is unset")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _verify(root, pinned, what):
    mismatches = []
    for rel, want in sorted(pinned.items()):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            mismatches.append(f"{rel}: missing")
            continue
        got = sha256_file(path)
        if got != want:
            mismatches.append(f"{rel}: {got}, pinned {want}")
    if mismatches:
        raise SystemExit(f"{what} at {root} is not the pinned tree:\n  " + "\n  ".join(mismatches))
    print(f"[ok] {what}: {len(pinned)} files match their pins")


def verify_source(root):
    """Refuse a checkpoint directory whose files are not the pinned ones."""
    _verify(root, SOURCE_FILES, f"{MODEL_ID} @ {REV[:8]}")


def verify_staged(root):
    """Refuse a staged Core ML tree whose files are not the pinned ones."""
    _verify(root, STAGED_FILES, f"{STAGED_ID} @ {STAGED_REV[:8]}")


def observed_toolchain():
    """The versions actually running, after asserting each equals its pin."""
    from importlib.metadata import PackageNotFoundError, version

    observed = {"python": platform.python_version()}
    mismatches = []
    if observed["python"].rsplit(".", 1)[0] != REQUIRED_TOOLCHAIN["python"]:
        mismatches.append(f"python: observed {observed['python']}, pinned {REQUIRED_TOOLCHAIN['python']}.x")
    for key, want in REQUIRED_TOOLCHAIN.items():
        if key == "python":
            continue
        try:
            got = version(_DISTRIBUTION_NAMES.get(key, key))
        except PackageNotFoundError:
            got = None
        observed[key] = got
        if got != want:
            mismatches.append(f"{key}: observed {got!r}, pinned {want!r}")
    if mismatches:
        raise SystemExit("toolchain differs from REQUIRED_TOOLCHAIN:\n  " + "\n  ".join(mismatches))
    return observed


def require_producer_toolchain(producer, observed, script):
    """A stage that consumes an artifact must run the environment that produced
    it: the whole toolchain record, python's patch level included."""
    recorded = producer.get("toolchain")
    if not isinstance(recorded, dict) or not recorded:
        raise SystemExit(f"the producer record names no toolchain; {script} cannot show it runs the producer's environment")
    if recorded != observed:
        diverged = "\n".join(
            f"  {field}: producer {recorded.get(field)!r}, {script} {observed.get(field)!r}"
            for field in sorted(set(recorded) | set(observed))
            if recorded.get(field) != observed.get(field)
        )
        raise SystemExit(f"the artifact was produced under another environment:\n{diverged}")


def host_record():
    """The host the goldens were computed on: its machine, CPU and OS. torch's
    fp32 reductions and torchvision's uint8 antialias resize both depend on it
    (the resize takes an integer AVX2 path on x86 and a float path on Apple
    Silicon)."""
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or "unknown"
    machine = platform.machine()
    return {
        "machine": machine,
        "cpu": cpu,
        "os": platform.platform(),
        "torchvision_uint8_antialias_path": "integer AVX2" if machine in ("x86_64", "AMD64") else "float",
    }


def provenance(observed, extra=None):
    """The provenance block every golden file carries."""
    block = {
        "source": {"model": MODEL_ID, "revision": REV, "files": SOURCE_FILES},
        "staged": {"model": STAGED_ID, "revision": STAGED_REV, "files": STAGED_FILES},
        "toolchain": observed,
        "host": host_record(),
        "generator": "coremlit/conversion/embeddinggemma2/scripts/generate_goldens.py",
        "argv": sys.argv[1:],
    }
    if extra:
        block.update(extra)
    return block


# The fp32 audio conversion (§7.5). It keeps the fp16 package's I/O contract
# exactly, so one contract serves both packages in the kit; only the precision
# the graph computes in differs.
FP16_AUDIO_PACKAGE = "EmbeddingGemma2Audio.mlpackage"
FP32_AUDIO_PACKAGE = "EmbeddingGemma2AudioFp32.mlpackage"
WAVEFORM_SAMPLES = 160_160
FRAMES = 1_000
AUDIO_ROWS = 250
AUDIO_CONTRACT = {
    "inputs": [
        {"name": "frame_mask", "shape": [1, FRAMES], "dtype": "FLOAT16"},
        {"name": "waveform", "shape": [1, WAVEFORM_SAMPLES], "dtype": "FLOAT32"},
    ],
    "outputs": [
        {"name": "audio_tokens", "shape": [1, AUDIO_ROWS, HIDDEN_SIZE], "dtype": "FLOAT16"},
    ],
}

# What the recipe writes beside the package in the staging tree, and what it
# publishes. The producer record and the verify record are recipe provenance;
# only the package and its checksums are the published artifact set.
PRODUCER_FILE = "producer.json"
VERIFY_FILE = "verify.json"
CHECKSUMS_FILE = "CHECKSUMS.sha256"


def valid_frames(samples):
    """The frames of a window of ``samples`` whose mask is 1: frame ``i`` is
    valid when ``i * 160 + 321 <= 160 + samples`` (HF's framing, §4.3)."""
    if 160 + samples < 321:
        return 0
    return min(FRAMES, (160 + samples - 321) // 160 + 1)


def audio_tokens(samples):
    """The audio tokens of a window: four frames each, the last partial."""
    return (valid_frames(samples) + 3) // 4


def padded_window(window):
    """A window as the kit feeds it: ``waveform`` (160 zeros, the samples,
    zeros to 160,160) and ``frame_mask`` (1 over the valid frames)."""
    import numpy as np

    assert MIN_WINDOW_SAMPLES <= len(window) <= WINDOW_SAMPLES, len(window)
    waveform = np.zeros((1, WAVEFORM_SAMPLES), dtype=np.float32)
    waveform[0, 160 : 160 + len(window)] = window
    frame_mask = np.zeros((1, FRAMES), dtype=np.float16)
    frame_mask[0, : valid_frames(len(window))] = 1.0
    return waveform, frame_mask


def package_contract(path):
    """A package's I/O as its spec states it: each feature's name, shape and
    multi-array data type, inputs and outputs sorted by name."""
    import coremltools as ct

    spec = ct.models.MLModel(path, skip_model_load=True).get_spec()
    data_type = ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name

    def features(entries):
        return sorted(
            (
                {
                    "name": f.name,
                    "shape": [int(d) for d in f.type.multiArrayType.shape],
                    "dtype": data_type(f.type.multiArrayType.dataType),
                }
                for f in entries
            ),
            key=lambda f: f["name"],
        )

    return {"inputs": features(spec.description.input), "outputs": features(spec.description.output)}


# The Core ML roads the verification judges every window on, in the order it
# records them. The CPU road is the one a host without a usable GPU takes.
VERIFY_ROADS = ("CPU_ONLY", "CPU_AND_GPU")


def require_current_verification(verify):
    """Refuses a ``verify.json`` that is not from the current verification.

    The report staging reads is bound to the package's digest, but a digest
    match is not enough: a report written before the verification judged the
    CPU road (``compute_units`` a single string, no per-window road) attests a
    package nobody checked on that road. A current report names both roads,
    holds every window on each of them, and says how many ``layer_norm``
    operations it checked for an explicit ``beta``.
    """
    if not verify.get("passed"):
        raise SystemExit("verify.json does not record a pass")
    if verify.get("compute_units") != list(VERIFY_ROADS):
        raise SystemExit(
            f"verify.json is from an older verification (compute_units {verify.get('compute_units')!r}, "
            f"wanted {list(VERIFY_ROADS)}): re-run verify_audio_fp32.py"
        )
    if not verify.get("layer_norm_operations"):
        raise SystemExit("verify.json records no layer_norm check: re-run verify_audio_fp32.py")
    windows = verify.get("windows") or []
    for road in VERIFY_ROADS:
        if not any(w.get("road") == road for w in windows):
            raise SystemExit(f"verify.json holds no window judged on {road}: re-run verify_audio_fp32.py")
    by_window = {}
    for w in windows:
        by_window.setdefault((w.get("clip"), w.get("start")), set()).add(w.get("road"))
    short = sorted(str(k) for k, roads in by_window.items() if roads != set(VERIFY_ROADS))
    if short:
        raise SystemExit(f"verify.json judged these windows on one road only: {', '.join(short)}")


def betaless_layer_norms(operations):
    """The names of the ``layer_norm`` operations among ``operations`` that
    have no ``beta``, and how many ``layer_norm`` operations there are.

    Core ML's fp32 CPU road mishandles a ``layer_norm`` with no ``beta`` inside
    the audio tower's subsampling block (see ``probes/layer_norm_cpu_probe.py``):
    it adds a stray per-channel offset to the upper half of the channels, while
    the GPU road is right. torch's ``nn.LayerNorm(bias=False)`` emits exactly
    that operation, so the recipe gives every such norm an explicit zero bias
    and refuses a program that still holds one.

    ``operations`` is any iterable of objects with ``op_type``, ``inputs`` (a
    mapping that has ``"beta"`` only when the operation has one, or maps it to
    ``None``) and ``outputs`` (whose first item has a ``name``).
    """
    count, missing = 0, []
    for op in operations:
        if op.op_type != "layer_norm":
            continue
        count += 1
        if op.inputs.get("beta") is None:
            missing.append(op.outputs[0].name)
    return count, missing


def package_layer_norms(path):
    """``betaless_layer_norms`` over the saved package's ``main`` function."""
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as load_mil

    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    spec = model.get_spec()
    program = load_mil(
        spec,
        specification_version=spec.specificationVersion,
        file_weights_dir=model.weights_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    return betaless_layer_norms(program.functions["main"].operations)


def require_explicit_beta(path):
    """Refuses a package that holds a ``layer_norm`` without a ``beta``, or none
    at all (a check that matched nothing would hold for any package). Returns
    the number of ``layer_norm`` operations."""
    count, missing = package_layer_norms(path)
    if count == 0:
        raise SystemExit(f"{path}: no layer_norm operation found; the beta check would hold vacuously")
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} of {count} layer_norm operations have no beta ({', '.join(missing)}). "
            "Core ML's fp32 CPU road adds a stray offset to such a norm; give the module an explicit zero bias."
        )
    return count


def digest_tree(root):
    """Every file under ``root``, as ``(relative path, sha256)``, sorted by path."""
    entries = []
    for directory, _, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            entries.append((os.path.relpath(path, root), sha256_file(path)))
    return sorted(entries)


def worst_token_cosine(ours, theirs, what):
    """The worst cosine between corresponding tokens (rows of the last axis)
    of ``ours`` and ``theirs``, computed in float64.

    Fail-closed. A NaN or an infinity in either side, or a token of zero norm,
    makes a cosine that is not a finite number, and a NaN compares false
    against every floor: a ``worst < floor`` gate would pass it. So every such
    token is refused here, by name, before a floor is ever compared, and so is
    a window with no token.
    """
    import numpy as np

    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    if a.shape != b.shape:
        raise SystemExit(f"{what}: {a.shape} tokens against {b.shape}")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    cosine = np.atleast_1d(cosine)
    if cosine.size == 0:
        raise SystemExit(f"{what}: no token to compare")
    bad = np.flatnonzero(~np.isfinite(cosine))
    if bad.size:
        first = int(bad[0])
        raise SystemExit(
            f"{what}: token {first}'s cosine is {cosine.flat[first]}, not a finite number "
            f"({bad.size} of {cosine.size} tokens): a NaN, an infinity or a zero-norm token"
        )
    return float(cosine.min())


def write_json_atomic(path, body):
    """Writes ``body`` as JSON to ``path`` through a unique temporary file in
    the same directory and a rename, so a reader sees the old file or the new
    one, never a part."""
    import json
    import uuid

    temp = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
