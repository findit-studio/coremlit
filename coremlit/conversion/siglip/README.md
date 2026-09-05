# siglip2-naflex CoreML conversion

Deterministically re-derives the two **SigLIP 2** (`siglip2-base-patch16-naflex`)
CoreML towers that `coremlit::embeddings::siglip` runs, converted **from the
official public checkpoint** — not consumed from any pre-uploaded artifact repo.

The converted artifacts are **published**, as this recipe's OUTPUT, at
[`FinDIT-Studio/siglip2-naflex-coreml`](https://huggingface.co/FinDIT-Studio/siglip2-naflex-coreml)
revision `eb514c2ab66fb702d43c742add0be5b091b02dab`. That repo is the PRODUCT of
running this recipe, never an input to it: the conversion below still starts at
`google/siglip2-base-patch16-naflex` and re-derives the graphs from those weights.
Publishing changed where a *consumer* obtains the artifacts, not what the recipe
reads. That bundle holds the fp16 ship set — the two `.mlmodelc` trees, the
`pos_embed_16x16x768.f32le.bin` sidecar, `CHECKSUMS.sha256`, and the source
checkpoint's `tokenizer.json` copied verbatim (34 MB; the crate reads it from the
artifact root rather than embedding it, see below). The fp32 towers built below
are conversion intermediates for the verification matrix and are not published.

## Source (pinned)

- Repo: [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex)
  — **Apache-2.0** (see the repo-root `NOTICE`, §8).
- Revision: `b53b807d3a2d5e2b3911292f2d69e5341cdc064c`
- Per-file SHA-256 (verified on load, fail-closed — `scripts/_siglip_common.py`):
  - `model.safetensors` — `ac5f28bbdf92c0c1696ccbd3ce716426049cd67ad8045b66d0d938b0f9c8bbec`
  - `tokenizer.json` — `58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0`
    (the crate embeds no copy: this file is republished verbatim in the artifact
    bundle above, and `siglip::text::contract::TOKENIZER_SHA256_HEX` is this same
    digest, enforced fail-closed at `TextEmbedder::load` before any model load)
  - `tokenizer.model` — `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2`
    (advisory sentencepiece cross-check; not bundled)

## Toolchain (dedicated venv — clap's transformers-5 venv is a TRAP)

`python 3.11`, `torch==2.5.1`, `transformers==4.53.3`, `coremltools==9.0`,
`numpy==1.26.4`, `pillow==12.3.0`, `tokenizers==0.21.2`. These pins are asserted fail-closed
by every converter and by the manifest stager (`assert_toolchain_pins`, issue #97);
each converter records its observed versions beside its artifact
(`toolchain_<tower>.json`) and the stager refuses a `MANIFEST.json` whose
toolchain differs from what produced the bytes. transformers **4.53.3** is
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

Host class for every number below: **MacBookPro18,2 / Apple M1 Max / macOS 26.5**.
Every ANE statement is about THIS Neural Engine — the `macos-15-arm64` CI runner
has no real ANE (`tests/siglip/placement.rs` records it falling back to the GPU), so
none of the ANE rows is reproducible there.

Two columns: the **published** bundle at `eb514c2` (converted before the ANE
rewrite) and the recipe **at this revision** (the ANE rewrite; the owner publishes
it and drives the lock/manifest/band bump).

- **fp32-CoreML(CPU) vs PyTorch fp32** (artifact faithfulness floor ≥ 0.9999):
  vision **1.0000000**, text **1.0000000** — both columns. This assert is blind to a
  change that only moves fp16, which is why the verifier now also reports fp16
  `max|Δ|` per arm and gates the fp16 `CpuAndGpu` arm with a no-regression band
  (`GPU_FP16_NO_REGRESSION = 0.99999`; the published number is the reference).
- **fp16-CoreML vs fp32-CoreML**, vision tower, worst cosine over the six fixtures
  (`max|Δ|` for this recipe in parentheses):

  | arm | published `eb514c2` | this recipe |
  |---|---|---|
  | `CpuAndGpu` (THE ship gate, ≥ 0.99917) | **0.99999487** | **0.99999428** (0.0065) |
  | `CpuOnly` | 0.98197 | 0.99838 (0.1015) |
  | `CpuAndNeuralEngine` | **0.31369** (collapse) | **0.99992054** (0.0173) |
  | `All` | **0.31369** (follows the ANE; `max|Δ|` vs GPU 11.36) | **0.99993241** (0.0098; `max|Δ|` vs GPU 0.0078) |

  The `CpuAndGpu` move 0.99999487 → 0.99999428 is a **systematic** change, not
  rounding noise, and it is bisected: the explicit head alone reproduces the
  published number exactly (0.99999488 vs the torch goldens); the elementwise GELU
  is the whole delta. Text tower (untouched by the rewrite): `CpuAndGpu` 0.99999873
  (delta +4.8e-9), `CpuOnly` 0.99982, `CpuAndNeuralEngine` 0.99999, `All` 0.99998.
- **fp16-CoreML(CpuAndGpu) vs the committed torch goldens**: vision 0.99999429
  (published 0.99999488), text 0.99999874. The committed goldens are byte-identical
  across the rewrite (torch fp32 of the UNMODIFIED model).

**Latency** (`ms` per inference, this host, 50 timed predictions after 5 warm-ups,
3 repeats — min / median / max, spread = (max−min)/median; machine loadavg 10–15
from other work during the runs):

| tower · arm | published `eb514c2` | this recipe |
|---|---|---|
| vision · `CpuAndGpu` | 16.6 / 16.6 / 16.6 (0.4 %) | 16.8 / 16.9 / 16.9 (1.1 %) |
| vision · `CpuAndNeuralEngine` | 28.0 / 28.1 / 28.1 (0.4 %) — wrong answers | 51.5 / 51.7 / 52.2 (1.2 %) |
| vision · `All` | 28.1 / 28.1 / 28.1 (0.2 %) — wrong answers | 79.0 / 79.9 / 85.2 (7.8 %) |
| vision · `CpuOnly` | 44.8 / 47.5 / 53.5 (18 %) | 66.5 / 73.5 / 102.7 (49 %) |
| text · `CpuAndGpu` | 8.9 / 10.9 / 14.3 (49 %) and 4.9 / 5.0 / 5.8 (18 %) in two runs | 5.1 / 5.5 / 5.9 (14 %) and 9.5 / 10.1 / 10.7 (12 %) |
| text · `CpuAndNeuralEngine` | ANECCompile fails → fallback, 77–211 (95 %) | same, 79–96 (18 %) |

Read it as: the text tower did not change and its rows swap between runs, so a
2× text difference is noise on this loaded host; the vision GPU arm is the same;
the vision ANE arm went from 28 ms of wrong answers to 52 ms of right ones (the
explicit head alone measured ≈ 29–38 ms across runs; the GELU `where` selects add
≈ 13 ms on the ANE and nothing on the GPU). The ANE is the SLOWER arm here.

**Placement decision (measured, never marketed):** vision ships `CpuAndGpu`. On
the published bundle the ANE arm collapses (0.31) and `All` follows it, so `All`
is unsafe there. With this recipe the ANE arm holds the floor but is ≈ 3× slower
than the GPU on this host, so `CpuAndGpu` stays the default and the ANE becomes an
*available* arm for a power-constrained caller, chosen through `ComputeUnits`.
Energy per image was NOT measured (`powermetrics` needs `sudo`; command below).
Text ships `CpuAndGpu` too (its whole-graph ANECCompile fails and falls back
gracefully; the GPU is granite-class).

## The ANE rewrite (issue #51) — what was measured, and what the fix is

On the published fp16 bundle, per-layer debug conversion, `ANE` vs `GPU` vs torch
fp32, worst over the six fixtures:

- The encoder alone would NOT have passed on the ANE either: the residual stream
  is at cos ≥ 0.998 after every one of the 12 layers and `post_layernorm` at
  0.9953 — below the 0.99917 gate — and `features` then drops to **0.38**. Two
  separate effects, two separate fixes.
- Nothing overflows fp16 anywhere in fp32: the largest tensor in the tower is
  `fc1` in layer 9 at 1172, the residual stream ≤ 424, LayerNorm variance ≤ 378,
  pre-scale `QKᵀ` ≤ 628. The issue's overflow hypotheses (attention logits,
  LayerNorm variance) were **refuted** by measurement. The stock MIL `gelu` op does
  not overflow either (swept ±1200: `max|Δ|` 4.8e-7 on every compute unit) — only
  the elementwise decomposition's cubic would, which is what its clamp is for.
- **The head.** Head-only fp16 graphs: the stock `nn.MultiheadAttention` lowering
  scores 0.751 on the ANE (0.999999 GPU) with or without the pad mask and at every
  mask magnitude; the same math written out explicitly (`ManualMAPHead`) scores
  0.999995. With the stock head UNCHANGED and its averaged attention weights
  exported as a second output, the ANE's weights are a proper distribution
  (non-negative, sum 1.0000–1.0005 over `P`, exactly 0 on every pad token) but the
  WRONG one (cos 0.316 vs fp32): the fault is in the pre-softmax scores, not in the
  softmax or the masking. It was not isolated further — every piece of the stock
  lowering (the `(P, B, D)` packed `linear`, its 5-D-transpose K/V split, the
  `perm=(1,2,0)` K transpose, the constant-`q` matmul, the softmax) is correct on
  the ANE in isolation, and a step-for-step mirror with intermediate outputs is
  correct too; the fused whole is not. What is established: on this ANE the stock
  lowering computes wrong, and every reformulation tried computes right.
  `ManualMAPHead` reuses the head's parameters unchanged (verified identical to
  `nn.MultiheadAttention` at torch 2.5.1 / transformers 4.53.3; structural asserts
  pin that) and lowers to plain `linear`/`matmul`/`softmax`.
- **The GELU.** With the head fixed the ANE arm reached 0.99796 — under the gate.
  Pinning LayerNorm and softmax to fp32 changed nothing; pinning GELU too gave
  0.99986 at 546 ms/image: the ANE's fused `linear → gelu` path is coarser than the
  elementwise ops. `ClampedTanhGelu` is `gelu_pytorch_tanh` written out elementwise,
  the tanh on `clamp(x, ±10)` and the tails selected exactly (`x` above 10, `0`
  below −10). Measured deviation from the stock op in the fp16 pipeline (swept
  ±1200, dense on ±40, negative tail included): 0 for `x < −20`, 0.0083 on
  `|x| ≤ 20`, the input's own fp16 half-ulp for `x > 20`. Forms measured and
  rejected on this ANE: `x·sigmoid(2·inner)` (exact negative tail, but the ANE arm
  drops to 0.99899); the clamped tanh without the tail selects (emits `x/2048` for
  `x < −10` — `1 + tanh` never reaches 0 in fp16; 18 % of GELU inputs sit in the
  tails); a clamp-ramp gate (leaves ≤ 0.005 on `(−10, −9)`).
- Both rewrites reuse the checkpoint's parameters unchanged; the whole vision model
  is deep-copied once so the pre-trace faithfulness assert compares against the
  UNMODIFIED model and can never compare it to itself: **1.00000000** on all six
  fixtures, as before.

Energy, if you want the number the issue asks for (needs `sudo`):

```sh
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 40 &
# ... run N image embeds on CpuAndGpu, then on CpuAndNeuralEngine, and read the
# per-block mW columns; divide by images/s from the placement/e2e suites.
```

## Replay

```sh
export SIGLIP_CONV=/path/to/scratch          # holds .venv + src-model
export SIGLIP_GOLDENS="$PWD/coremlit/tests/siglip/fixtures/goldens"
export SIGLIP_MODELS_OUT="$PWD/Models/siglip2-naflex"
python3.11 -m venv "$SIGLIP_CONV/.venv"
"$SIGLIP_CONV/.venv/bin/pip" install torch==2.5.1 transformers==4.53.3 \
  coremltools==9.0 numpy==1.26.4 pillow==12.3.0 tokenizers==0.21.2 huggingface_hub
hf download google/siglip2-base-patch16-naflex \
  --revision b53b807d3a2d5e2b3911292f2d69e5341cdc064c --local-dir "$SIGLIP_CONV/src-model"
bash coremlit/conversion/siglip/run_siglip.sh
```

The corpus PNGs (`$SIGLIP_GOLDENS/images/`) are committed; their source URLs +
licenses are in `scripts/_fixtures.py` and `corpus.json`.

Consuming rather than re-deriving? Unlike the other kits, this checkout stages no
local `Models/siglip2-naflex/` tree, so the model-gated tests (`SIGLIP_TEST_MODELS`,
else `Models/siglip2-naflex/`) need the published bundle fetched first:

```sh
hf download FinDIT-Studio/siglip2-naflex-coreml \
  --revision eb514c2ab66fb702d43c742add0be5b091b02dab \
  --local-dir Models/siglip2-naflex
```

CI stages **less than that**: the `model-tests` job downloads only
`siglip2-base-patch16-naflex-512/tokenizer.json` (34 MB, per `MODELS_LOCK`'s
fourth table), because the two gates it runs —
`tests/siglip/tokenizer_identity.rs` — call no `Model::load` and need only the
tokenizer plus the committed golden corpus. The tower-dependent gates
(`model_io`, `text_model_io`, `parity_embed`, `placement`, `e2e`) need the full
~784 MB bundle above and stay local/dev gates.

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
