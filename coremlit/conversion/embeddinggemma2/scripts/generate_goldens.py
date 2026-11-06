"""Write R1, the sentence-transformers fp32 goldens of coremlit's
``embeddings::embeddinggemma2`` kit, into ``$EG2_GOLDENS``.

Inputs, every one pinned and verified on each run:
  * ``$EG2_SRC``: ``google/embeddinggemma-2`` @ ``f9972ac2`` (the reference);
  * ``$EG2_MODELS``: ``FluidInference/embeddinggemma-2-coreml`` @ ``3f978dd``
    (its token table and position table, for the input digests);
  * ``$EG2_FLUIDUSE``: a FluidUse checkout @ ``b51f7ad`` (its 30 cases);
  * ``$EG2_TESTS``: ``coremlit/tests`` (the committed clips and images).

Outputs:
  * ``text.json``: per item the prompt, the text, and under BOTH
    ``split_special_tokens`` settings the ids, the UTF-8 byte offsets over
    ``prefix + text`` and the special mask; the 768-d vector; for the literal
    special and custom boundary items the vector under ``true`` too; the
    SHA-256 of the ``inputs_embeds`` the kit builds for the item's unpacked
    function; and the framing of the eight prompts and of ``Bare``.
  * ``audio.json``: per clip its window plan and, per window, the token count
    and the 768-d vector of the fp32 PyTorch model.
  * ``image.json``: per image and budget (70, 140, 280) the resized geometry,
    the 768-d vector, the SHA-256 of Hugging Face's resized pixels and two
    32 x 32 tiles of them; per budget the digests of the five vision inputs on
    a synthetic image at a fixed point of the resize (V3); and the position
    table's provenance.

The Python ``tokenizers`` binding reports CHARACTER offsets. The kit's offsets
are UTF-8 BYTE offsets, so every offset is converted here, over the joint
string, and the provenance block says so.

Vectors are computed by running the model on exactly the golden ids (or the
processor's media inputs), mean-pooled with the prompt included and
normalized, as sentence-transformers' pipeline does; each text vector is
cross-checked against ``SentenceTransformer.encode`` of the same input.
Truncations are derived, not stored: the first M coordinates of the 768-d
vector, re-normalized, which is ``encode(truncate_dim=M)`` re-normalized
(checked here at 512, 256 and 128 on every text vector).

Reproducible to the cosine floors, not byte-identical: torch's fp32
reductions depend on the host.
"""
import base64
import datetime
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
import _corpus  # noqa: E402
from _eg2_common import (  # noqa: E402
    BOA_ID,
    BOS_ID,
    EMBED_SCALE_BITS,
    EOA_ID,
    EOS_ID,
    HIDDEN_SIZE,
    MAX_TOKENS,
    MIN_TAIL_SAMPLES,
    MIN_WINDOW_SAMPLES,
    PATCH_SIZE,
    POOLING_KERNEL,
    SAMPLE_RATE,
    TEXT_FUNCTIONS,
    VISION_BUDGETS,
    VOCAB_SIZE,
    WINDOW_SAMPLES,
    env,
    observed_toolchain,
    provenance,
    sha256_bytes,
    sha256_file,
    verify_source,
    verify_staged,
)

SCALE = np.array([EMBED_SCALE_BITS], dtype=np.uint32).view(np.float32)[0]
CROSSCHECK_FLOOR = 0.99999
MEDIA_PLACEHOLDERS = ("<|image|>", "<|audio|>", "<|video|>")


def f32_list(values):
    """f32 components as the shortest decimals that round-trip as f32."""
    return [float(np.format_float_positional(np.float32(v), unique=True, trim="-")) for v in values]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def byte_offsets(joint, char_offsets):
    """Character offsets over ``joint`` as UTF-8 byte offsets."""
    starts = [0]
    for ch in joint:
        starts.append(starts[-1] + len(ch.encode("utf-8")))
    return [[starts[a], starts[b]] for a, b in char_offsets]


def tokenize(tok, joint):
    enc = tok.encode(joint)
    return {
        "ids": list(enc.ids),
        "offsets": byte_offsets(joint, enc.offsets),
        "special_mask": list(enc.special_tokens_mask),
    }


def unpacked_function(n):
    return next(s for s in TEXT_FUNCTIONS if s >= n)


def inputs_embeds_sha(table, ids):
    """The ``inputs_embeds`` bytes the kit builds for ``ids`` on its unpacked
    function: each row ``f16(f32(bf16) * f32(sqrt(512)))``, then pad rows of
    the pad token's row."""
    s = unpacked_function(len(ids))
    rows = np.asarray(list(ids) + [0] * (s - len(ids)), dtype=np.int64)
    bits = table[rows].astype(np.uint32) << np.uint32(16)
    gathered = (bits.view(np.float32) * SCALE).astype(np.float16)
    return f"embed_{s}", sha256_bytes(gathered.astype("<f2").tobytes())


class Reference:
    """The fp32 PyTorch model, through sentence-transformers' modules."""

    def __init__(self, src):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            src, device="cpu", model_kwargs={"torch_dtype": torch.float32}, local_files_only=True
        )
        self.model.eval()
        self.transformer = self.model[0]

    @torch.inference_mode()
    def embed_ids(self, ids):
        features = self.transformer.preprocess(["x"])
        features["input_ids"] = torch.tensor([ids], dtype=torch.long)
        features["attention_mask"] = torch.ones((1, len(ids)), dtype=torch.long)
        return self.model(features)["sentence_embedding"][0].numpy()

    @torch.inference_mode()
    def encode(self, inputs, prompt=None, **kwargs):
        return self.model.encode([inputs], prompt=prompt or None, batch_size=1, **kwargs)[0]

    def preprocess(self, inputs, **kwargs):
        return self.transformer.preprocess([inputs], **kwargs)


def text_goldens(ref, src, staged, fluiduse):
    recognize = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
    split = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
    split.encode_special_tokens = True
    table = np.memmap(os.path.join(staged, "embeddings.bf16"), dtype="<u2", mode="r", shape=(VOCAB_SIZE, HIDDEN_SIZE))

    items = []
    for item_id, prompt, text, fluid_ids in _corpus.fluiduse_cases(os.path.join(fluiduse, _corpus.FLUIDUSE_CASES)):
        items.append((item_id, "fluiduse", prompt, text, fluid_ids))
    items += [(i, "ours", p, t, None) for i, p, t in _corpus.OURS]
    items += [(i, "boundary", p, t, None) for i, p, t in _corpus.BOUNDARY]
    items += [(i, "long", p, t, None) for i, p, t in _corpus.long_texts()]

    out = []
    worst_crosscheck = 1.0
    for item_id, source, prompt, text, fluid_ids in items:
        prefix = _corpus.prefix_of(prompt)
        joint = prefix + text
        entry = {
            "id": item_id,
            "source": source,
            "prompt": _corpus.prompt_record(prompt),
            "prefix": prefix,
            "text": text,
            "recognize": tokenize(recognize, joint),
            "split": tokenize(split, joint),
        }
        ids = entry["recognize"]["ids"]
        if fluid_ids is not None:
            assert ids == fluid_ids, f"{item_id}: Hugging Face's ids differ from FluidUse's fixture"
            entry["fluiduse_ids_match"] = True
        if len(ids) <= MAX_TOKENS:
            function, digest = inputs_embeds_sha(table, ids)
            entry["function"] = function
            entry["inputs_embeds_sha256"] = digest
            vector = ref.embed_ids(ids)
            # The processor refuses text holding a media placeholder literal
            # with no media beside it, so those vectors come from the ids
            # alone and are not cross-checked.
            crosscheck = not any(p in joint for p in MEDIA_PLACEHOLDERS)
            entry["crosschecked"] = crosscheck
            if crosscheck:
                via_encode = ref.encode(text, prompt=prefix)
                cosine = float(unit(vector) @ unit(via_encode))
                assert cosine >= CROSSCHECK_FLOOR, f"{item_id}: the ids' vector and encode's disagree ({cosine})"
                worst_crosscheck = min(worst_crosscheck, cosine)
                for width in (512, 256, 128):
                    truncated = ref.encode(text, prompt=prefix, truncate_dim=width, normalize_embeddings=True)
                    derived = unit(np.asarray(vector)[:width])
                    assert float(derived @ unit(truncated)) >= CROSSCHECK_FLOOR, f"{item_id}: truncation {width}"
            entry["vector"] = f32_list(vector)
            if item_id in _corpus.SPECIAL_IDS or source == "boundary":
                split_ids = entry["split"]["ids"]
                if len(split_ids) <= MAX_TOKENS:
                    entry["vector_split"] = f32_list(ref.embed_ids(split_ids))
        out.append(entry)
        print(f"[text] {item_id}: {len(ids)} ids", flush=True)

    framing = {key: len(recognize.encode(p.rstrip(" ")).ids) for key, p in _corpus.NAMED.items()}
    framing["bare"] = len(recognize.encode("").ids)
    return {
        "items": out,
        "framing": framing,
        "crosscheck_worst_cosine": worst_crosscheck,
    }


# WAVE_FORMAT_PCM, and WAVE_FORMAT_EXTENSIBLE whose subformat GUID begins
# with the PCM tag: Python's `wave` reads only the first.
PCM, EXTENSIBLE = 1, 0xFFFE


def read_wav(path, limit=None):
    """16 kHz mono s16le samples of a RIFF/WAVE file, as f32 = s / 32768."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", path
    fmt, pcm, offset = None, None, 12
    while offset + 8 <= len(data):
        tag, size = data[offset : offset + 4], int.from_bytes(data[offset + 4 : offset + 8], "little")
        body = data[offset + 8 : offset + 8 + size]
        if tag == b"fmt ":
            fmt = body
        elif tag == b"data":
            pcm = body
        offset += 8 + size + (size & 1)
    assert fmt is not None and pcm is not None, path
    format_tag = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if format_tag == EXTENSIBLE:
        format_tag = int.from_bytes(fmt[24:26], "little")
    assert (format_tag, channels, rate, bits) == (PCM, 1, SAMPLE_RATE, 16), (path, format_tag, channels, rate, bits)
    samples = np.frombuffer(pcm, dtype="<i2")
    if limit is not None:
        samples = samples[:limit]
    return samples.astype(np.float32) / 32768.0


def window_plan(n):
    """The kit's plan: windows of 160,000 samples, no overlap; a final window
    shorter than 8,000 is dropped unless it is the only one."""
    spans = [(start, min(WINDOW_SAMPLES, n - start)) for start in range(0, n, WINDOW_SAMPLES)]
    if len(spans) > 1 and spans[-1][1] < MIN_TAIL_SAMPLES:
        spans.pop()
    assert spans and spans[0][1] >= MIN_WINDOW_SAMPLES
    return spans


def audio_tokens(samples):
    frames = 0 if 160 + samples < 321 else min(1000, (160 + samples - 321) // 160 + 1)
    return (frames + 3) // 4


def audio_goldens(ref, tests):
    clips = []
    for clip_id, rel, limit in _corpus.CLIPS:
        path = os.path.join(tests, rel)
        samples = read_wav(path, limit)
        windows = []
        for start, length in window_plan(len(samples)):
            window = samples[start : start + length]
            features = ref.preprocess({"audio": window})
            ids = features["input_ids"][0].tolist()
            tokens = len(ids) - 4
            assert ids[:2] == [BOS_ID, BOA_ID] and ids[-2:] == [EOA_ID, EOS_ID], clip_id
            assert tokens == audio_tokens(length), (clip_id, tokens, audio_tokens(length))
            vector = ref.encode({"audio": window})
            windows.append({"start": start, "samples": length, "tokens": tokens, "vector": f32_list(vector)})
        clips.append(
            {
                "id": clip_id,
                "file": rel,
                "file_sha256": sha256_file(path),
                "samples": len(samples),
                "cut": None if limit is None else {"first_samples": limit},
                "conversion": "16 kHz mono s16le read as f32 = s / 32768; no resampling",
                "windows": windows,
            }
        )
        print(f"[audio] {clip_id}: {len(windows)} windows", flush=True)
    return {"clips": clips}


def pattern_image(width, height):
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:height, 0:width]
    for c in range(3):
        pixels[:, :, c] = _corpus.pattern_pixel(xs, ys, c)
    return Image.fromarray(pixels, "RGB")


def hf_resized(ref, image, budget):
    """Hugging Face's resize, captured before rescaling: (C, H, W) uint8."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import pil_to_tensor

    processor = ref.transformer.processor.image_processor
    return processor.aspect_ratio_preserving_resize(
        image=pil_to_tensor(image),
        patch_size=PATCH_SIZE,
        max_patches=budget * POOLING_KERNEL**2,
        pooling_kernel_size=POOLING_KERNEL,
        resample=InterpolationMode.BICUBIC,
    )


def patchify(chw_unit):
    """``[C, H, W]`` -> ``[ph * pw, 16 * 16 * C]``, HF's layout."""
    c, h, w = chw_unit.shape
    p = PATCH_SIZE
    return chw_unit.reshape(c, h // p, p, w // p, p).permute(1, 3, 2, 4, 0).reshape((h // p) * (w // p), -1)


def tile(hwc, y, x, size=32):
    y = max(0, min(y, hwc.shape[0] - size))
    x = max(0, min(x, hwc.shape[1] - size))
    block = np.ascontiguousarray(hwc[y : y + size, x : x + size])
    return {"y": int(y), "x": int(x), "h": size, "w": size, "rgb_base64": base64.b64encode(block.tobytes()).decode()}


def vision_inputs(pixel_values, position_ids, budget, table):
    """The five vision inputs the kit assembles, from HF's patches and
    positions, with ``pool``, ``valid`` and ``position_embeddings``
    synthesized from the conversion's documented rules."""
    n = 9 * budget
    real = position_ids[:, 0] >= 0
    count = int(real.sum())
    cols = int(position_ids[real, 0].max()) + 1
    rows = int(position_ids[real, 1].max()) + 1
    assert rows * cols == count
    patches = pixel_values.astype(np.float16).reshape(1, n, 768)
    positions = position_ids.astype(np.float16).reshape(1, n, 2)
    valid = np.zeros((1, n), dtype=np.float16)
    valid[0, :count] = 1.0
    pool = np.zeros((budget, n), dtype=np.float16)
    embeds = np.zeros((1, n, 768), dtype=np.float16)
    ninth = np.float16(1.0 / 9.0)
    for p in range(count):
        x, y = int(position_ids[p, 0]), int(position_ids[p, 1])
        assert p == y * cols + x
        pool[x // 3 + (cols // 3) * (y // 3), p] = ninth
        embeds[0, p] = (table[0, x].astype(np.float32) + table[1, y].astype(np.float32)).astype(np.float16)
    digest = lambda a: sha256_bytes(np.ascontiguousarray(a).astype("<f2").tobytes())  # noqa: E731
    return {
        "patches": {"shape": [1, n, 768], "sha256": digest(patches)},
        "positions": {"shape": [1, n, 2], "sha256": digest(positions)},
        "valid": {"shape": [1, n], "sha256": digest(valid)},
        "pool": {"shape": [budget, n], "sha256": digest(pool)},
        "position_embeddings": {"shape": [1, n, 768], "sha256": digest(embeds)},
        "grid": {"rows": rows, "cols": cols, "patches": count},
    }


# Fixed points of the resize, one per budget, each short of its patch budget
# so padding patches are present: no resize runs, so the inputs depend on the
# assembly alone.
FIXED_POINTS = {70: (336, 432), 140: (480, 624), 280: (672, 912)}


def image_goldens(ref, staged, tests, src):
    table = np.fromfile(os.path.join(staged, "position_embeddings.f16"), dtype="<f2").reshape(2, 1024, 768)
    images = []
    for image_id, rel in _corpus.IMAGES:
        if rel is None:
            image = pattern_image(96, 48)
            source = {"pattern": "_corpus.pattern_pixel(x, y, c)", "width": 96, "height": 48}
        else:
            path = os.path.join(tests, rel)
            image = Image.open(path).convert("RGB")
            source = {"file": rel, "file_sha256": sha256_file(path)}
        budgets = {}
        for budget in VISION_BUDGETS:
            kwargs = {"processing_kwargs": {"image": {"max_soft_tokens": budget}}}
            features = ref.preprocess(image, **kwargs)
            resized = hf_resized(ref, image, budget)
            _, h, w = resized.shape
            # HF rescales by multiplying with its f64 factor, which is not
            # bit-identical to dividing by 255 in f32.
            factor = ref.transformer.processor.image_processor.rescale_factor
            patches = patchify(resized.to(torch.float32) * factor)
            count = patches.shape[0]
            assert torch.equal(features["pixel_values"][0][:count].to(torch.float32), patches), (image_id, budget)
            tokens = features["input_ids"].shape[1] - 4
            assert tokens == count // 9, (image_id, budget)
            hwc = resized.permute(1, 2, 0).contiguous().numpy()
            vector = ref.encode(image, **kwargs)
            budgets[str(budget)] = {
                "resized": [int(h), int(w)],
                "patches": int(count),
                "tokens": int(tokens),
                "resized_sha256": sha256_bytes(hwc.tobytes()),
                "tiles": [tile(hwc, h // 2 - 16, w // 2 - 16), tile(hwc, h // 4, w // 4)],
                "vector": f32_list(vector),
            }
        images.append({"id": image_id, "source": source, "width": image.width, "height": image.height, "budgets": budgets})
        print(f"[image] {image_id}", flush=True)

    synthetic = []
    for budget, (h, w) in FIXED_POINTS.items():
        image = pattern_image(w, h)
        features = ref.preprocess(image, processing_kwargs={"image": {"max_soft_tokens": budget}})
        resized = hf_resized(ref, image, budget)
        assert tuple(resized.shape[1:]) == (h, w), "a fixed point of the resize"
        inputs = vision_inputs(
            features["pixel_values"][0].numpy(), features["image_position_ids"][0].numpy(), budget, table
        )
        synthetic.append({"budget": budget, "height": h, "width": w, "inputs": inputs})

    return {
        "images": images,
        "synthetic": {
            "rule": "_corpus.pattern_pixel(x, y, c) at each budget's fixed point of the resize",
            "derived_from_processor": ["patches", "positions"],
            "synthesized": ["pool", "valid", "position_embeddings"],
            "budgets": synthetic,
        },
        "position_table": position_table_provenance(src, staged, table),
    }


def position_table_provenance(src, staged, staged_table):
    """The staged position table is the checkpoint's vision position
    embeddings, rounded to f16: x (patch column) in table 0, y (patch row) in
    table 1. Also the staged token table against the checkpoint's."""
    from safetensors import safe_open

    with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
        positions = f.get_tensor("vision_tower.patch_embedder.position_embedding_table")
        tokens = f.get_tensor("language_model.embed_tokens.weight")
    rounded = positions[:, :1024, :].to(torch.float32).numpy().astype(np.float16)
    token_bits = tokens.view(torch.int16).numpy().astype("<i2").tobytes()
    with open(os.path.join(staged, "embeddings.bf16"), "rb") as f:
        staged_tokens = f.read()
    return {
        "checkpoint_tensor": "vision_tower.patch_embedder.position_embedding_table",
        "checkpoint_shape": list(positions.shape),
        "rule": "rows 0..1024 of each table, bf16 -> f32 -> f16 (round to nearest even)",
        "equal_to_staged": bool(np.array_equal(rounded.view(np.uint16), staged_table.view(np.uint16))),
        "table_0": "the patch column (x)",
        "table_1": "the patch row (y)",
        "token_table": {
            "checkpoint_tensor": "language_model.embed_tokens.weight",
            "equal_to_staged_bytes": token_bits == staged_tokens,
        },
    }


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write(out, name, body, observed, extra=None):
    """One JSON document: the provenance block indented, and every list's
    entries one compact line each, so a diff names the entry that moved."""
    document = {"provenance": provenance(observed, extra), **body}
    lines = ["{"]
    keys = list(document)
    for index, key in enumerate(keys):
        value = document[key]
        tail = "," if index + 1 < len(keys) else ""
        if key == "provenance":
            text = json.dumps(value, ensure_ascii=False, indent=1).replace("\n", "\n ")
            lines.append(f" {compact(key)}: {text}{tail}")
        elif isinstance(value, list):
            lines.append(f" {compact(key)}: [")
            lines += [f"  {compact(entry)}" + ("," if i + 1 < len(value) else "") for i, entry in enumerate(value)]
            lines.append(f" ]{tail}")
        else:
            lines.append(f" {compact(key)}: {compact(value)}{tail}")
    lines.append("}")
    path = os.path.join(out, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(document)), name
    print(f"[ok] {name}: {sha256_file(path)}")


def main():
    src, staged, out = env("EG2_SRC"), env("EG2_MODELS"), env("EG2_GOLDENS")
    tests, fluiduse = env("EG2_TESTS"), env("EG2_FLUIDUSE")
    observed = observed_toolchain()
    verify_source(src)
    verify_staged(staged)
    torch.manual_seed(0)
    ref = Reference(src)
    os.makedirs(out, exist_ok=True)
    stamp = {"generated": datetime.date.today().isoformat(), "fluiduse": _corpus.FLUIDUSE_REV}
    offsets = {
        "offsets": "UTF-8 byte offsets over prefix + text, converted from the tokenizers binding's character offsets",
        "settings": "recognize: split_special_tokens = false; split: tokenizers' encode_special_tokens = true",
        "vectors": "the model on exactly the golden ids, mean pooling with the prompt included, Normalize; cross-checked against SentenceTransformer.encode",
        "truncation": "derived: the first M coordinates re-normalized; checked against encode(truncate_dim=M) at 512, 256, 128",
    }
    write(out, "text.json", text_goldens(ref, src, staged, fluiduse), observed, {**stamp, **offsets})
    write(out, "audio.json", audio_goldens(ref, tests), observed, stamp)
    write(out, "image.json", image_goldens(ref, staged, tests, src), observed, stamp)


if __name__ == "__main__":
    main()
