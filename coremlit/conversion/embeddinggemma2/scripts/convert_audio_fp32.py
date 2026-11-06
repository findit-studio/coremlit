"""Convert google/embeddinggemma-2's audio tower to an fp32 Core ML package
that keeps FluidInference's fp16 package's I/O contract exactly (§4.6, §7.5).

``waveform`` F32 ``[1, 160160]`` and ``frame_mask`` F16 ``[1, 1000]`` in,
``audio_tokens`` F16 ``[1, 250, 512]`` out, computed in FLOAT32. The front end
is re-implemented in the graph with Hugging Face's framing, as FluidInference's
package does it: frame ``i`` reads the padded waveform's samples ``i·160 ..
i·160 + 320``, a periodic Hann window, a 512-point DFT as a strided
convolution, the magnitude, the HTK mel filterbank's 128 bins, and
``log(mel + 0.001)``; the features are masked, and the audio tower and its
projection into the text model's input space follow. Only the precision the
graph computes in differs from the fp16 package. The waveform is NOT scaled by
1/8 as the fp16 graph scales it to stay inside fp16's range: fp32 does not
need it, and log-mel is the same either way.

Before tracing, a faithfulness gate runs the wrapper in PyTorch against
Hugging Face's own audio path (its feature extractor, the variable-length
tower, the projection) on three windows of ``jfk.wav``: a full one, a 3.48 s
cut and its 1 s tail. Every token must hold a cosine of at least
``FAITHFULNESS_COSINE``. Then the wrapper is traced over a PARTIAL window, so
the masked path is the one recorded, and converted.

Environment: ``EG2_SRC`` (the pinned checkpoint), ``EG2_TESTS``
(``coremlit/tests``, for the clips), ``EG2_STAGE`` (the staging directory the
package and ``producer.json`` are written to).
"""
import datetime
import os
import shutil
import sys
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from _eg2_common import (  # noqa: E402
    AUDIO_CONTRACT,
    FP32_AUDIO_PACKAGE,
    FRAMES,
    MODEL_ID,
    PRODUCER_FILE,
    REV,
    SOURCE_FILES,
    WAVEFORM_SAMPLES,
    audio_tokens,
    env,
    host_record,
    observed_toolchain,
    padded_window,
    verify_source,
    worst_token_cosine,
    write_json_atomic,
)
from generate_goldens import read_wav  # noqa: E402

FRAME, HOP, FFT = 320, 160, 512
BINS = FFT // 2 + 1
MEL_FLOOR = 1e-3

# The faithfulness gate's floor: the wrapper's tokens against Hugging Face's,
# both fp32 PyTorch. Measured at 0.99999976 (the DFT as an fp32 convolution
# against numpy's float64 FFT); the floor leaves room for nothing else.
FAITHFULNESS_COSINE = 0.99999


class AudioFp32(torch.nn.Module):
    """The fp16 package's contract over the checkpoint's own modules."""

    def __init__(self, model, extractor):
        super().__init__()
        self.tower = model.audio_tower
        self.embed = model.embed_audio
        n = np.arange(FRAME)
        k = np.arange(BINS)[:, None]
        angle = 2.0 * np.pi * k * n / FFT
        window = extractor.window.astype(np.float64)
        dft = np.concatenate([np.cos(angle) * window, -np.sin(angle) * window])
        self.register_buffer("dft", torch.from_numpy(dft[:, None, :].astype(np.float32)))
        self.register_buffer("mel", torch.from_numpy(extractor.mel_filters.T.astype(np.float32)))
        # The tower's chunked local attention reads its mask in blocks: for
        # block b, query i of the block and key slot o, the query at position
        # b·chunk + i and the key at b·chunk + o - past. Where that key may be
        # attended depends on the audio only through which keys are real; the
        # rest is a constant, computed here rather than traced.
        config = model.audio_tower.config
        tokens = FRAMES // 4
        chunk = config.attention_chunk_size
        past = config.attention_context_left - 1
        future = config.attention_context_right
        blocks = -(-tokens // chunk)
        query = np.arange(blocks)[:, None, None] * chunk + np.arange(chunk)[None, :, None]
        key = np.arange(blocks)[:, None, None] * chunk + np.arange(chunk + past + future)[None, None, :] - past
        distance = query - key
        window = ((distance >= 0) & (distance < past)) | ((distance < 0) & (-distance < future))
        window &= (query < tokens) & (key >= 0) & (key < tokens)
        self.register_buffer("window", torch.from_numpy(window[None, None].astype(np.float32)))
        self.blocked = (1, 1, blocks, chunk, chunk + past + future)
        keys = np.broadcast_to(np.clip(key, 0, tokens - 1), self.blocked[2:])
        self.register_buffer("keys", torch.from_numpy(np.ascontiguousarray(keys).reshape(-1)))
        # Each attention layer gathers every block's keys and values with
        # `unfold`, which the converter lacks; the same windows are an
        # `index_select` at constant positions, exact.
        context = chunk + past + future
        self.register_buffer(
            "contexts",
            torch.from_numpy((np.arange(blocks)[:, None] * chunk + np.arange(context)[None, :]).reshape(-1)),
        )
        for layer in self.tower.layers:
            layer.self_attn._extract_block_context = self.block_context(layer.self_attn, blocks)

    def block_context(self, attention, blocks):
        """``attention._extract_block_context`` as an ``index_select``."""
        contexts = self.contexts

        def extract(hidden_states):
            batch, _, heads, dim = hidden_states.shape
            padded = F.pad(
                hidden_states,
                (0, 0, 0, 0, attention.max_past_horizon, attention.max_future_horizon + attention.chunk_size - 1),
            )
            windows = padded.index_select(1, contexts)
            return windows.reshape(batch, blocks, attention.context_size, heads, dim)

        return extract

    def forward(self, waveform, frame_mask):
        spectrum = F.conv1d(waveform[:, None, :], self.dft, stride=HOP)
        real, imaginary = spectrum[:, :BINS], spectrum[:, BINS:]
        magnitude = torch.sqrt(real * real + imaginary * imaginary)
        mel = torch.matmul(self.mel, magnitude)
        mask = frame_mask.to(torch.float32)
        features = torch.log(mel + MEL_FLOOR).transpose(1, 2) * mask[..., None]
        return self.embed(self.encode(features, mask > 0.5))

    def encode(self, features, valid):
        """The audio tower's forward, step for step, with its attention mask
        built here in the blocked form the chunked attention reads:
        transformers' ``create_bidirectional_mask`` builds it under ``vmap``,
        which does not trace, and its blocking computes indices the converter
        types as floats. The mask is the one they build for eager attention: a
        key is attended when it is a real token and lies within the query's
        left context (``0 <= q - k < attention_context_left - 1``, or ``0 < k -
        q < attention_context_right``), 0 where attended and the dtype's
        minimum elsewhere, padding rows and keys included. The faithfulness
        gate holds it to the tower's own forward."""
        tower = self.tower
        config = tower.config
        hidden, valid = tower.subsample_conv_projection(features, valid)
        position_embeddings = tower.rel_pos_enc(hidden)
        # 0 where attended, the dtype's minimum elsewhere, in arithmetic rather
        # than boolean operations, which Core ML's converter lacks.
        real = valid.to(torch.float32).index_select(1, self.keys).reshape(self.blocked)
        mask = (1.0 - self.window * real) * torch.finfo(torch.float32).min
        for layer in tower.layers[: config.num_hidden_layers]:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=position_embeddings)
        return tower.output_proj(hidden)


def fold_size_one_casts():
    """coremltools 9.0 casts a folded constant to a Python scalar with
    ``int(value)``. The tracer hands it the size of a dimension as a
    one-element array, and numpy 2.4 refuses ``int`` on an array with a
    dimension. This reads the one value instead; every other cast is
    coremltools' own. The conversion's semantics do not change: the cast's
    input is that one value."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    original = ops._cast

    def cast(context, node, dtype, dtype_name):
        x = context[node.inputs[0]]
        if x.can_be_folded_to_const() and np.ndim(x.val) > 0 and np.size(x.val) == 1:
            value = np.asarray(x.val).reshape(()).item()
            context.add(mb.const(val=dtype(value), name=node.name), node.name)
            return
        original(context, node, dtype, dtype_name)

    ops._cast = cast


def load(src):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(src, dtype=torch.float32, local_files_only=True).eval()
    extractor = AutoProcessor.from_pretrained(src, local_files_only=True).feature_extractor
    return model, extractor


@torch.inference_mode()
def reference_tokens(model, extractor, window):
    """Hugging Face's own audio path: its features, the variable-length tower
    and the projection; the real tokens only."""
    features = extractor([window], return_tensors="pt")
    out = model.get_audio_features(features["input_features"], features["input_features_mask"])
    return out.pooler_output[out.attention_mask].numpy()


@torch.inference_mode()
def wrapper_tokens(wrapper, window):
    waveform, frame_mask = padded_window(window)
    tokens = wrapper(torch.from_numpy(waveform), torch.from_numpy(frame_mask))[0]
    return tokens[: audio_tokens(len(window))].numpy()


def faithfulness_windows(tests):
    samples = read_wav(os.path.join(tests, "whisper/fixtures/audio/jfk.wav"))
    return [
        ("jfk full window", samples[:160_000]),
        ("jfk 3.48 s cut", samples[:55_680]),
        ("jfk 1 s tail", samples[160_000:176_000]),
    ]


def faithfulness(references, wrapper, windows):
    """The wrapper against Hugging Face's tokens, computed on the unpatched
    model before the wrapper took it over."""
    record = []
    for (name, window), theirs in zip(windows, references):
        ours = wrapper_tokens(wrapper, window)
        if ours.shape != theirs.shape:
            raise SystemExit(f"faithfulness: {name}: {ours.shape} tokens against Hugging Face's {theirs.shape}")
        worst = worst_token_cosine(ours, theirs, f"faithfulness: {name}")
        record.append({"window": name, "tokens": int(ours.shape[0]), "worst_cosine": worst})
        print(f"[faithfulness] {name}: {ours.shape[0]} tokens, worst cosine {worst:.9f}", flush=True)
        if worst < FAITHFULNESS_COSINE:
            raise SystemExit(f"faithfulness: {name}: worst token cosine {worst} under {FAITHFULNESS_COSINE}")
    return record


def main():
    src, tests, stage = env("EG2_SRC"), env("EG2_TESTS"), env("EG2_STAGE")
    observed = observed_toolchain()
    verify_source(src)
    torch.manual_seed(0)
    import coremltools as ct

    model, extractor = load(src)
    windows = faithfulness_windows(tests)
    references = [reference_tokens(model, extractor, window) for _, window in windows]
    wrapper = AudioFp32(model, extractor).eval()
    record = faithfulness(references, wrapper, windows)

    # Trace over a PARTIAL window, so the masked path is what the trace records.
    samples = read_wav(os.path.join(tests, "whisper/fixtures/audio/jfk.wav"))
    waveform, frame_mask = padded_window(samples[:55_680])
    started = time.time()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (torch.from_numpy(waveform), torch.from_numpy(frame_mask)))
    print(f"[trace] {time.time() - started:.1f} s", flush=True)

    started = time.time()
    fold_size_one_casts()
    package = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="waveform", shape=(1, WAVEFORM_SAMPLES), dtype=np.float32),
            ct.TensorType(name="frame_mask", shape=(1, FRAMES), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="audio_tokens", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"[convert] {time.time() - started:.1f} s", flush=True)
    package.short_description = (
        f"EmbeddingGemma 2's audio tower, fp32, from {MODEL_ID} @ {REV[:8]}: "
        "waveform [1,160160] f32 and frame_mask [1,1000] f16 to audio_tokens [1,250,512] f16."
    )

    # Into the staging directory through a unique temporary name and a rename.
    os.makedirs(stage, exist_ok=True)
    target = os.path.join(stage, FP32_AUDIO_PACKAGE)
    # Core ML saves an ML program only under a `.mlpackage` name.
    temp = os.path.join(stage, f".tmp-{uuid.uuid4().hex}-{FP32_AUDIO_PACKAGE}")
    package.save(temp)
    if os.path.exists(target):
        shutil.rmtree(target)
    os.replace(temp, target)

    write_json_atomic(
        os.path.join(stage, PRODUCER_FILE),
        {
            "package": FP32_AUDIO_PACKAGE,
            "source": {"model": MODEL_ID, "revision": REV, "files": SOURCE_FILES},
            "contract": AUDIO_CONTRACT,
            "compute_precision": "FLOAT32",
            "minimum_deployment_target": "macOS15",
            "front_end": {
                "frame": FRAME,
                "hop": HOP,
                "fft": FFT,
                "window": "periodic Hann, the feature extractor's",
                "mel": "the feature extractor's HTK filterbank, 128 bins",
                "log": f"log(mel + {MEL_FLOOR})",
                "mask": "features times frame_mask; the tower's own mask is frame_mask > 0.5",
            },
            "faithfulness": {"floor": FAITHFULNESS_COSINE, "windows": record},
            "traced_over": "jfk.wav's first 55,680 samples (a partial window)",
            "toolchain": observed,
            "host": host_record(),
            "produced": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print(f"[ok] {target}", flush=True)


if __name__ == "__main__":
    main()
