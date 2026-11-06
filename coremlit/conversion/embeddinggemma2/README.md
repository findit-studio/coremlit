# EmbeddingGemma 2: the fp32 audio package and the R1 goldens

This directory holds two recipes over the original checkpoint:

- `run_embeddinggemma2.sh` converts the model's audio tower to an fp32 Core ML package, `EmbeddingGemma2AudioFp32.mlpackage`, verifies it and stages it with its `CHECKSUMS.sha256` for upload. It uploads nothing: the upload is a separate step, run on its own.
- `scripts/generate_goldens.py` computes R1, the committed PyTorch fp32 goldens under `coremlit/tests/embeddinggemma2/fixtures/goldens/`. The run script runs it only with `EG2_REGEN_GOLDENS=1`, since a conversion should not move the committed oracle.

## Source pins

- `google/embeddinggemma-2` @ `f9972ac22e696a105c482202646979e33e4b5a1e`. `scripts/_eg2_common.py` pins every file the recipe reads by SHA-256, `model.safetensors` (`197a3296…3d79`) included, and every script refuses a checkpoint that differs.
- `FluidInference/embeddinggemma-2-coreml` @ `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`, read only: its fp16 audio package is the contract the fp32 package keeps.

## Toolchain

`REQUIRED_TOOLCHAIN` in `scripts/_eg2_common.py` pins Python 3.11 and, exactly, torch 2.14.1, torchvision 0.29.2, transformers 5.22.1, sentence-transformers 6.2.0, tokenizers 0.23.3, coremltools 9.0, numpy 2.4.6, Pillow 13.0.0 and safetensors 0.8.0. Every step refuses another environment, and each step after the conversion refuses to run under an environment other than the one recorded in `producer.json`.

coremltools 9.0 warns that it has not been tested with torch 2.14.1. FluidInference's fp16 package was converted with the same pair. One incompatibility is worked around in `convert_audio_fp32.py`: coremltools casts a folded one-element constant to a Python scalar with `int(value)`, which numpy 2.4 refuses for an array with a dimension. `fold_size_one_casts` reads the one value instead, and the conversion's semantics do not change.

## The contract

| | name | shape | type |
|---|---|---|---|
| input | `waveform` | `[1, 160160]` | float32 |
| input | `frame_mask` | `[1, 1000]` | float16 |
| output | `audio_tokens` | `[1, 250, 512]` | float16 |

These are FluidInference's fp16 package's, exactly, so the kit holds both packages to one contract. Only the precision the graph computes in differs: `FLOAT32` here. The output's final rounding to float16 costs nothing downstream: the text function's `inputs_embeds` is float16 either way.

## What the conversion builds

The graph re-implements the fp16 package's in-graph log-mel front end with Hugging Face's framing (§4.3 of the design):

- frame `i` reads the padded waveform's samples `i·160 .. i·160 + 320`;
- a periodic Hann window, then a 512-point DFT as a strided convolution;
- the magnitude, the HTK mel filterbank's 128 bins, and `log(mel + 0.001)`;
- the features masked by `frame_mask`.

The checkpoint's audio tower and its projection into the text model's input space follow. Two of the tower's steps are rebuilt so that they trace and convert, and the faithfulness gate holds both to the tower's own forward:

- **The attention mask.** transformers builds it under `vmap`, which does not trace. The wrapper builds the same mask, in the blocked form the chunked attention reads, from constants and the real tokens.
- **Each attention layer's key and value windows.** These use `unfold`, which the converter lacks. The wrapper uses an `index_select` at constant positions instead, which is exact.

Unlike the fp16 graph, this one does not scale the waveform by 1/8: fp32 has the range, and log-mel is the same either way.

## The floors

| gate | where | floor | measured at C13 (Apple M1 Max, macOS 27.0) |
|---|---|---|---|
| faithfulness: the PyTorch wrapper against Hugging Face's own audio path, three windows of `jfk.wav` | `convert_audio_fp32.py` | every token's cosine ≥ 0.99999 | worst 0.999999999 |
| the contract against the fp16 package's | `verify_audio_fp32.py` | equal | equal |
| token-level parity: the package under Core ML (`CPU_AND_GPU`) against PyTorch fp32, every window of the six committed clips | `verify_audio_fp32.py` | each window's worst token cosine ≥ 0.9999 | worst 0.999998901 (`sine440`); max abs difference 5.6e-3, the float16 output's rounding |
| M5f: the kit's vectors through the package against R1 | `coremlit/tests/embeddinggemma2/audio.rs` | worst ≥ 0.999 | worst 0.999885, mean 0.999968 |

**Both cosine gates fail closed.** A NaN compares false against every floor, so a NaN in an output, or a token of zero norm, would pass a `worst < floor` check. Both gates reduce a window through `_eg2_common.worst_token_cosine`, which refuses a cosine that is not a finite number by name before a floor is compared. `tests/test_cosine_gates.py` plants a NaN, a zero-norm token and an infinity, and checks that both gates call it (`python3 -m unittest discover -s coremlit/conversion/embeddinggemma2/tests`; stdlib `unittest` and numpy, no model or venv).

**Re-derivable to the floors, not bit-reproducible.** Neither the Core ML compiler nor torch's fp32 reduction order is pinned by these versions. A re-run must hold the floors. It need not reproduce the published bytes.

## What the gates defend against, and what they do not

They defend against:
- **A front end that drifts from Hugging Face's.** The faithfulness gate compares token for token before tracing, and token-level parity compares again after conversion.
- **A trace that froze the wrong path.** The trace runs over a partial window, so the masked path is the one recorded, and parity covers full, partial and short windows.
- **A precision mistake.** A `FLOAT16` conversion of this graph overflows: its worst token cosines are near zero. The precision is also recorded in `producer.json`.
- **A contract that differs from the fp16 package's**, in name, shape or type.
- **Bytes other than the verified ones being staged.** `verify.json` binds the package's digest, and `stage_manifest.py` refuses anything else.

They do not:
- **Compare the package with the fp16 package.** Each is judged against PyTorch and R1 on its own.
- **Prove anything beyond the six committed clips and three windows.**
- **Pin the bytes.**
- **Cover the placement or Neural Engine numerics.** Those are the kit's laws (M8 to M12).

## Running it

```sh
EG2_CONV=<scratch with .venv> EG2_SRC=<the pinned checkpoint> \
EG2_MODELS=<FluidInference's staged tree> EG2_TESTS=coremlit/tests \
EG2_MODELS_OUT=<the directory to stage for upload> \
coremlit/conversion/embeddinggemma2/run_embeddinggemma2.sh
```

The steps are:
1. the faithfulness gate, the trace and the conversion into `$EG2_STAGE`, with `producer.json`;
2. the verify, which writes `verify.json`;
3. the staging into `$EG2_MODELS_OUT`, under the package's own name and beside its `CHECKSUMS.sha256`.

To run the kit's laws over the package before it is published, place it beside the base files of the staged tree, for example by linking them into a scratch directory. Then point `EMBEDDINGGEMMA2_TEST_MODELS` at that directory: `compile_artifacts` compiles it there.
