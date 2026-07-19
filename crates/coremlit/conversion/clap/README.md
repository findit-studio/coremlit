# clapkit CoreML conversion recipes

These scripts re-derive the two CLAP CoreML graphs clapkit loads. They are the
committed record of the T1 conversion — the paths in `run_clapkit.sh` are T1's
scratchpad locations and must be adapted to re-run.

## Provenance

- **Source model:** [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused)
  (PyTorch, via `transformers.ClapModel`), pinned to revision
  `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.
- **Distributed CoreML artifacts:**
  [`FinDIT-Studio/clapkit-coreml`](https://huggingface.co/FinDIT-Studio/clapkit-coreml),
  revision `02a99c6a8be21da1e9a947499ea503a10c80c4f1` — ships both the **fp16**
  tier (byte-identical to the original fp16-only publication
  `97d631f3814e1e46b798a8e88c9aa2e2202fdf67`) and the 2×-smaller **int8** tier
  (`clap_{audio,text}_int8.mlmodelc`). Fetch at the immutable revision (never
  mutable `main`):
  ```sh
  hf download FinDIT-Studio/clapkit-coreml \
    --revision 02a99c6a8be21da1e9a947499ea503a10c80c4f1 \
    --local-dir Models/clapkit
  ```
  Every artifact file's SHA-256 (both tiers) + I/O shapes are pinned in
  `tests/clap/model_io.rs` / `tests/clap/text_model_io.rs`.
- **Toolchain (pinned):** coremltools 9.0 · torch 2.5.1 · transformers 5.14.0 ·
  numpy 1.26.4 · python 3.11.15.

## License (load-bearing)

The LAION weights are treated as **CC-BY-4.0** by the spec, textclap's MODELS.md,
and the clapkit HF README front-matter — attribution to `laion/clap-htsat-unfused`
is required and is carried in the crate `NOTICE`. T1 flagged that the live
`laion/clap-htsat-unfused` HF card declares **apache-2.0**; attribution to the
source repo satisfies both licenses, and reconciling the HF front-matter is an
owner decision (recorded, not made here).

## I/O contract

| encoder | inputs | output |
|---|---|---|
| audio | `input_features` fp32 `[1, 1, 1001, 64]` (log-mel spectrogram) | `audio_embeds` fp32 `[1, 512]` pre-norm |
| text | `input_ids` int32 `[1, 512]`, `attention_mask` int32 `[1, 512]` | `text_embeds` fp32 `[1, 512]` pre-norm |

The projection heads (`Linear 768→512, ReLU, Linear 512→512`) are IN-graph; **L2
normalization is OUT of the graph** (clapkit normalizes in Rust), which keeps the
fp16 rsqrt-guard class out of both graphs entirely. Embedding dim 512. CLAP logit
scales (for the T5 zero-shot pipeline): `logit_scale_a.exp() = 18.6612`,
`logit_scale_t.exp() = 14.2857`.

## Mel: spectrogram-input, not in-graph (decided by measurement)

The audio graph takes a log-mel **spectrogram**; the mel/STFT front-end is a Rust
port of textclap's `mel.rs` (`crates/coremlit/src/embeddings/clap/audio/mel/`), bit-validated
against textclap's committed golden mel (`src/audio/mel/tests.rs`, measured
max-abs-diff ≈ 7.6e-6 = one f32 ULP). An in-graph mel was attempted
(`mel_in_graph_probe.py`, `measure_melgraph.py`) and **rejected** on three
measured grounds:

1. **Faithfulness (decisive):** an in-graph **fp32** STFT lands only 0.58–0.96
   cosine (worst 0.5830 over 10 real clips) from the correct HF-mel path
   end-to-end, and the **fp16** in-graph STFT degrades further (0.95–1.00, worst
   0.9534 vs the fp32 melgraph) — nowhere near the spectrogram path's clean fp16.
   Reproducing HF's `ClapFeatureExtractor` exactly needs **float64** STFT
   numerics, hostile to an fp16 ANE graph. (Measured by `measure_melgraph.py`:
   both separately-converted melgraph precisions vs the shipped fp32 spectrogram
   graph, all arms fed the identical repeat-tiled 480 000-sample input.)
2. **Guard class:** `power_to_db`'s `amin = 1e-10` floor is 1680× below `2^-24` —
   the exact fp16-vanishing-guard class the campaign forbids.
3. **No ANE upside:** the audio graph does not compile for the ANE either way.

The spectrogram-input path gives perfect fp32 parity (cosine 1.0) and clean fp16.

Rust mel params (== HF feature extractor == textclap): `n_fft 1024`, `hop 480`,
`n_mels 64`, `fmin 50`, `fmax 14000`, periodic Hann, Slaney scale + norm,
`center=True` reflect, `10·log10(max(·, 1e-10))`, HTSAT input-norm `none`,
time-major `[1001, 64]`.

## Conversion shims (both exact/safe; carried in the recipes)

- **bicubic → matmul resize** (`_clap_common.py::patch_bicubic_resize`):
  coremltools 9.0 has no `upsample_bicubic2d`. HTSAT's `reshape_mel2img` bicubic
  resize is replaced by an EXACT baked-constant matmul derived from torch's own
  bicubic kernel (proven cosine 1.0 vs the un-patched model). Bicubic
  interpolation is linear in the pixel values, so a fixed resize is a fixed
  matmul.
- **`new_ones` custom op** (`_custom_ops.py`): coremltools 9.0 ships `new_zeros`
  but not `new_ones`; transformers 5.14's `masking_utils` needs it
  (`q_idx.new_ones((), dtype=torch.bool)`). Implemented as the exact sibling of
  the stock `new_zeros`.
- **`torch.load` CVE workaround** (`_clap_common.py`): the source repo ships only
  `pytorch_model.bin` (no safetensors); transformers 5.14 blocks `torch.load`
  under torch < 2.6 as a blanket CVE guard. The guard is neutralized for this one
  trusted, revision-pinned, hash-verified load (documented + deliberate).

## fp16 guard audit: ALL CLEAN

The shipped fp16 graphs pass the MIL guard audit (55 sites: audio 29 `layer_norm`
+ 1 `batch_norm` at `epsilon = 0x1.5p-17 ≈ 1.001e-5`; text 25 `layer_norm` at
`epsilon = 0x1p-24` — the `1e-12` source eps auto-raised by the fp16 conversion to
exactly the audit floor `2^-24`). No `log`/`sqrt`/`rsqrt`/`real_div` in either
graph (L2-norm kept out). clapkit joins coremlit's `tests/fp16_guards.rs` sweep as
a clean control (`accepts_clapkit_conversion_norm_guards`).

## Scripts

| file | role |
|---|---|
| `run_clapkit.sh` | end-to-end pipeline (convert → compile → audit → verify → mel probe) |
| `scripts/_clap_common.py` | shared loader, constants, the bicubic-resize shim, the `torch.load` workaround |
| `scripts/_custom_ops.py` | the `new_ones` coremltools op shim |
| `scripts/_fixtures.py` | real audio clips + text prompts for verification |
| `scripts/convert_audio.py` | HTSAT + `audio_projection` → fp16/fp32 mlpackage |
| `scripts/convert_text.py` | RoBERTa + `text_projection` → fp16/fp32 mlpackage |
| `scripts/verify_encoders.py` | PyTorch-vs-CoreML fp32 cosine; fp16-vs-fp32 per compute unit |
| `scripts/mel_in_graph_probe.py`, `scripts/measure_melgraph.py` | the mel-in-graph rejection evidence |
| `scripts/inspect_clap.py`, `scripts/inspect_struct.py`, `scripts/probe_text_pad.py`, `scripts/probe_wrappers.py` | investigation probes (structure, padding, wrappers) |
