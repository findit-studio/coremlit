# siglip2-naflex CoreML conversion

Deterministically re-derives the two **SigLIP 2** (`siglip2-base-patch16-naflex`)
CoreML towers that `coremlit::embeddings::siglip` runs, converted **from the
official public checkpoint** — not consumed from any pre-uploaded artifact repo.
Local staging only; a public re-upload of the converted artifacts is a later owner
decision.

## Source (pinned)

- Repo: [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex)
  — **Apache-2.0** (see the repo-root `NOTICE`, §8).
- Revision: `b53b807d3a2d5e2b3911292f2d69e5341cdc064c`
- Per-file SHA-256 (verified on load, fail-closed — `scripts/_siglip_common.py`):
  - `model.safetensors` — `ac5f28bbdf92c0c1696ccbd3ce716426049cd67ad8045b66d0d938b0f9c8bbec`
  - `tokenizer.json` — `58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0`
    (also the bundled `src/embeddings/siglip/assets/tokenizer.json` identity pin)
  - `tokenizer.model` — `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2`
    (advisory sentencepiece cross-check; not bundled)

## Toolchain (dedicated venv — clap's transformers-5 venv is a TRAP)

`python 3.11`, `torch==2.5.1`, `transformers==4.53.3`, `coremltools==9.0`,
`numpy==1.26.4`, `pillow==12.3.0`, `tokenizers==0.21.2`. transformers **4.53.3** is
load-bearing: v5's `Siglip2Tokenizer` pads **left** and reworks the image
processors, which would silently diverge from the frozen Wave-A contract (right
padding) and the pillow-12.3.0 uint8-resize oracles. Both towers convert clean on
`coremltools 9.0` with the checkpoint's default `sdpa` attention (no eager fallback
or head decomposition was needed).

## I/O contract

| tower | artifact | inputs | output |
|---|---|---|---|
| vision | `siglip2_vision_512.mlmodelc` | `pixel_values` f32 `[1,512,768]` · `position_embeddings` f32 `[1,512,768]` · `attention_mask` f32 `[1,512]` | `image_features` f32 `[1,768]` |
| text | `siglip2_text_64.mlmodelc` | `input_ids` i32 `[1,64]` (no attention_mask) | `text_features` f32 `[1,768]` |

Both outputs are **pre-L2-norm** — the Rust caller normalizes (keeps the fp16
rsqrt-guard class out of the graphs). Plus the sidecar
`pos_embed_16x16x768.f32le.bin` (the base position grid, 786432 bytes).

## The position-embedding lift (why the vision graph is static)

The stock `Siglip2VisionEmbeddings` runs a per-image
`F.interpolate(size=spatial_shapes, mode="bilinear", antialias=True)` of the base
16×16 position grid — a **data-dependent** resize that cannot trace to one static
CoreML graph. `convert_vision.py` hoists it OUT: the graph takes the resized
`position_embeddings` as an input, and the Rust runtime computes the lift per image
(`lift_position_embeddings`, hermetically tested). The wrapper is byte-for-byte the
stock `Siglip2VisionTransformer.forward` with the position embeddings supplied
instead of recomputed — proven before tracing by a faithfulness assert
(`cos(wrapper, model.get_image_features) >= 0.999999`, measured **1.00000000** over
all 6 fixtures) against the UNMODIFIED model, using the checkpoint's OWN
`resize_positional_embeddings` for the lift.

## Measured verification (this machine; `scripts/verify_towers.py`, fail-closed)

- **fp32-CoreML(CPU) vs PyTorch fp32** (artifact faithfulness floor ≥ 0.9999):
  vision **1.0000000**, text **1.0000000**.
- **fp16-CoreML vs fp32-CoreML**, per compute unit:
  - `CpuAndGpu` (THE ship gate, ≥ 0.99917): vision **0.99999487**, text **0.99999873**.
  - `CpuOnly`: vision 0.98197, text 0.99982.
  - `CpuAndNeuralEngine`: **vision 0.31369** (systematic across all 6 images —
    materially below the earlier probe's 0.998118), text 0.99999.
  - `All`: vision **0.31369** (the planner dispatches vision to the ANE — `max|Δ|`
    from GPU = 11.36), text 0.99998.
- **fp16-CoreML(CpuAndGpu) vs the committed torch goldens**: vision 0.99999488,
  text 0.99999874.

**Placement decision (measured, never marketed):** vision ships `CpuAndGpu` — the
ANE arm collapses (0.31) and `All` follows it, so `All` is unsafe for vision. Text
ships `CpuAndGpu` too (its whole-graph ANECCompile fails and falls back gracefully;
the GPU is granite-class). The ANE arm is characterized, never floor-gated.

## Replay

```sh
export SIGLIP_CONV=/path/to/scratch          # holds .venv + src-model
export SIGLIP_GOLDENS="$PWD/crates/coremlit/tests/siglip/fixtures/goldens"
export SIGLIP_MODELS_OUT="$PWD/Models/siglip2-naflex"
python3.11 -m venv "$SIGLIP_CONV/.venv"
"$SIGLIP_CONV/.venv/bin/pip" install torch==2.5.1 transformers==4.53.3 \
  coremltools==9.0 numpy==1.26.4 pillow==12.3.0 tokenizers==0.21.2 huggingface_hub
hf download google/siglip2-base-patch16-naflex \
  --revision b53b807d3a2d5e2b3911292f2d69e5341cdc064c --local-dir "$SIGLIP_CONV/src-model"
bash crates/coremlit/conversion/siglip/run_siglip.sh
```

The corpus PNGs (`$SIGLIP_GOLDENS/images/`) are committed; their source URLs +
licenses are in `scripts/_fixtures.py` and `corpus.json`.

## Scripts

| file | role |
|---|---|
| `scripts/_siglip_common.py` | pins, SHA verify-on-load, model/processor/tokenizer loaders, the official lift, config-default + pad-side asserts |
| `scripts/_fixtures.py` | the committed corpus registry (images + captions + sources/licenses) |
| `scripts/convert_vision.py` | vision wrapper + faithfulness assert + trace + convert (fp16/fp32) + sidecar |
| `scripts/convert_text.py` | text wrapper + faithfulness assert + trace + convert (fp16/fp32) |
| `scripts/stage_manifest.py` | `CHECKSUMS.sha256` + `MANIFEST.json` over the shipped bundle |
| `scripts/verify_towers.py` | the fail-closed fp32-vs-torch + per-unit fp16 matrix |
| `scripts/generate_goldens.py` | `corpus.json` + `preprocess.json` + staged `.npy` fixtures |
| `run_siglip.sh` | the env-driven end-to-end driver |
