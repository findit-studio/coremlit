# CED ×4 → CoreML conversion (`audio::ced`, Waves B/C)

Re-derives the four CED `mel → logits` CoreML graphs from the OFFICIAL public
checkpoints, deterministically. Mirrors the `conversion/clap` recipe (pre-sigmoid
wrapper → trace → `ct.convert` fp16+fp32 → compile → fail-closed verify → goldens).

## Sources (pinned, SHA-verified at load)

All four public (Apache-2.0), `mispeech/ced-<size>`, converted from
`model.safetensors` at these revisions (also in `scripts/_ced_common.py::MODELS`,
`tests/ced/common/mod.rs`, and each bundle's `MANIFEST.json`):

| size | revision | embed_dim | heads | safetensors sha256 (head) |
|---|---|---|---|---|
| tiny  | `ace276d29dd0bb3f3517b0fa8cf300738c409019` | 192 | 3  | `0e086f0c…` |
| mini  | `26c3ebcae85d4330f4fc26763f029539a3afcda0` | 256 | 4  | `e4070d02…` |
| small | `06bb40c5ec089e96867ebc5246be02441f4a71e4` | 384 | 6  | `a495e319…` |
| base  | `db3e14a8db4c21b56b165261c39649741a900e7f` | 768 | 12 | `31493569…` |

Shared: depth 12, outputdim 527, n_mels 64, n_fft/win 512, hop 160, f_min 0,
f_max 8000, center=True, target_length 1012, pooling `mean`.

## Contract (matches `src/audio/ced` + `tests/ced/model_io.rs` EXACTLY)

`mel` f32 `[1, 64, 1001]` → `logits` f32 `[1, 527]`, **PRE-sigmoid** (the caller
applies the sigmoid; the CED analogue of "L2 by the caller"). The log-mel
front-end runs in Rust (`MelExtractor`), so the graph starts at the mel.

The wrapper (`CedMelToLogits`) reuses the REAL `CedForAudioClassification`
sub-modules and only drops the final `sigmoid` (`pooling="mean"` ends in
`.sigmoid()`); it is the EXACT pre-sigmoid of the unmodified forward
(`sigmoid(wrapper) == forward.logits`, measured max|Δ| = 0.0). CED's graph is
standard ops (Conv2d patch embed, BatchNorm2d eval, LayerNorm, GELU, matmul+
softmax attention), so NO custom ops are needed.

## Oracle (why PyTorch fp32, not the shipped ONNX)

The committed goldens are the **PyTorch fp32 pre-sigmoid** `CedMelToLogits`
output on the torchaudio `CedFeatureExtractor` mel. The repos also ship a
`model.onnx`, but it is **post-sigmoid** (probed: byte-identical to
`soundevents/models/tiny.onnx`, output ∈ [0, 0.944], cos to PyTorch post-sigmoid
0.9995) — so it cannot be the coremlit contract's PRE-sigmoid oracle (inverting
its exact 0.0/≈1.0 entries is degenerate). PyTorch fp32 is uniform across sizes,
exactly pre-sigmoid, and the same lineage as the conversion source. `verify_ced`
still records `sigmoid(CoreML)` vs the shipped `model.onnx` as a drop-in sanity
(the onnx computes its mel in-graph, so it diverges on degenerate clips — not a
gate). `ort` never enters the Rust crate, not even dev.

## Corpus (fully synthetic, license-free)

Deterministic 16 kHz mono clips (`scripts/_fixtures.py`) that hit clear AudioSet
classes: `sine440_10s`/`sine1000_3s` → Sine wave/Beep, `noise_10s` → White noise,
`silence_2s` → Silence, `long_15s` (sine+noise, two windows) for the e2e
aggregation gate. No third-party audio is committed.

## Measured (Apple silicon; the pinned bands live in the test sources)

- Conversion floor (CoreML fp32 vs PyTorch fp32 pre-sigmoid): worst cos **1.0**,
  max|Δlogit| ≤ **2.3e-5** (all four sizes) — the SHIP gate.
- fp16 vs fp32 per compute unit: worst cos ≥ **0.99999** on every arm
  (CpuOnly/CpuAndGpu/CpuAndNeuralEngine/All).
- End-to-end (`raw_scores` vs goldens): CpuOnly cos ≥ 0.999990, default (fp16
  ANE/GPU) cos ≥ 0.9999998; top-1/top-10 exact.
- e2e: a 440 Hz sine ranks **Sine wave** (class 501) top-1, confidence
  0.889–0.927 across sizes.

## Replay

```sh
export CED_PY=/path/to/venv/bin/python     # torch 2.5.1, torchaudio 2.5.1,
                                           # transformers 4.53.3, coremltools 9.0,
                                           # onnxruntime, huggingface_hub
export CED_CONV=/scratch/ced-conv          # snapshots + staging (default ~/.cache/…)
export CED_MODELS_OUT="$PWD/Models/ced"    # gitignored fp16 bundles (repo default)
crates/coremlit/conversion/ced/run_ced.sh  # convert → compile → manifest → verify → goldens
```

Then run the gates: `CED_TEST_MODELS="$CED_MODELS_OUT" cargo test -p coremlit
--features ced --test ced_model_io --test ced_parity_logits --test ced_placement
--test ced_e2e -- --include-ignored`.
