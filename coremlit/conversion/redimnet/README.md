# ReDimNet → CoreML conversion (identity lane, issue #123)

Re-derives the ReDimNet speaker-embedding CoreML graphs from the OFFICIAL public
checkpoints, deterministically. Mirrors the `conversion/ced` recipe (sub-forward wrapper →
trace → `ct.convert` fp16+fp32 → compile → manifest → fail-closed verify), plus a
four-arm placement sweep.

**One recipe, three artifacts.** `run_redimnet.sh <variant>` selects `b5`, `b2` or
`b2_ptn`; there is deliberately no default, because a recipe that converts a size nobody
asked for records provenance nobody can replay. Everything a variant changes — asset, SHA,
`model_config`, bundle name, pooled width — is one row of `VARIANTS` in
`scripts/_redimnet_common.py`. Everything a variant does NOT change — the front end, the
window, the I/O contract — is a module constant, asserted against every checkpoint at
load, and that shared front end is what lets ONE Rust door (`src/audio/identity`, landed
in #136) serve all three bundles through one unchanged `LoadContract`.

**Status.** B5 is the REGISTERED artifact: the door, its test module, the licence row,
the `MODELS_LOCK` table and the CI shard landed in #136 for it, and `REGISTERED_VARIANT`
in `scripts/_redimnet_common.py` names it. B2 and B2-ptn are **converted, measured, and
deliberately not registered** — see "B2: converted, measured, not registered" below for
the reason, which is a measurement rather than a schedule. Their placement and parity
evidence is recorded here so nobody re-derives it.

## Sources (pinned, SHA-verified at load)

| variant | weights (`IDRnD/redimnet` release `latest`) | bytes | sha256 | bundle |
|---|---|---|---|---|
| `b5` | **`b5-vox2-ft_lm.pt`** — 6 s large-margin fine-tune | 31,174,382 | `8b0c11bbf5a3a8bb39e5c072c4192d0b694d8c447cf126d4cd3c7346a04b39c8` | `redimnet_b5.mlmodelc` |
| `b2` | **`b2-vox2-ft_lm.pt`** — 6 s large-margin fine-tune | 20,582,650 | `c9b6bb2f6747caa28a41eaf2e372d66b0d1563baef186d18f5e99abd5e71e06f` | `redimnet_b2.mlmodelc` |
| `b2_ptn` | **`b2-vox2-ptn.pt`** — 2 s PRETRAIN, **no published metric of any kind** | 20,581,530 | `c18a42926878bc8ac079623fbf36f0bc8054cda1199e96fbe1a3f8e131796647` | `redimnet_b2_ptn.mlmodelc` |

| what | pin |
|---|---|
| model source (every variant) | `github.com/IDRnD/redimnet` @ `ce039a624cb99fe127702ceb94c6080090e5032f` |

The release tag is literally named `latest` and is **mutable**, so the tag is not the
lock — the SHA-256 is, and every stage verifies it. The model source is pinned too: a
checkpoint is only half the provenance, because `ReDimNetWrap` is *reconstructed* from the
archive's own `model_config` and the reconstructing code decides what the weights compute.

`model_config`, read out of each archive and asserted entry for entry at load. Shared by
every variant (`SHARED_CONFIG`, the entries the CONTRACT rests on): `F 72`, `block_1d_type
conv+att`, `hop_length 240`, `out_channels null`, `pooling_func ASTP`, `global_context_att
true`, `embed_dim 192`, `emb_bn false`. Per size: B5 is `C 32`, `block_2d_type
basic_resnet_fwse`, `group_divisor 16` (1,052 tensors, 7,709,351 parameters, tail
`Linear(4608, 192)`); B2 and B2-ptn are `C 16`, `block_2d_type convnext_like`,
`group_divisor 4` (556 tensors, 5,100,983 parameters, tail `Linear(2304, 192)`) — the two
B2 checkpoints are ONE architecture with two weight sets, and their compiled `model.mil`
is byte-identical (`ca22edff…`); only `weights/weight.bin` differs. Every load is
`load_state_dict` with zero missing and zero unexpected keys.

**The 2 s pretrain is fed the 6 s window.** The graph's input shape is the contract and
there is one contract; `b2-vox2-ptn.pt`'s last training stage used 2 s crops, and that
train/inference mismatch is recorded on the artifact (`training_crop_s: 2` in the manifest
and in `tests/identity/common/mod.rs`) rather than silently absorbed. Whether it costs
anything is the unmeasured question issue #123's short-segment experiment exists to
answer; this recipe converts the checkpoint so that experiment has a CoreML artifact, and
makes no claim about its quality.

**Only the `-vox2-` lineage.** The same release publishes `M-vb2+vox2+cnc-ft_mix.pt` and
`S-vb2-ptn.pt`, trained on VoxBlink2, whose authors state the CC BY-NC-SA 4.0 term
propagates to the trained model. `_redimnet_common.verify_asset_name` refuses any asset
whose name is not `-vox2-`, for every variant; it is a guard, not decoration.

## Re-running B5 through the variant recipe: what is and is not byte-identical

The refactor from a B5-only recipe to a variant recipe was proven by re-running B5 and
comparing the output against the PUBLISHED bundle (HF `80c2d0a`, the bytes
`tests/identity/common/mod.rs::ARTIFACT_SHA256` pins), and the result is worth stating
precisely because the naive claim is false:

| file | re-run vs published |
|---|---|
| `model.mil` | **byte-identical** (`75f9abd2…`) |
| `weights/weight.bin` | **byte-identical** (`1735fc68…`) |
| `metadata.json` | **byte-identical** (`03610dd7…`) |
| `coremldata.bin` | differs |
| `analytics/coremldata.bin` | differs |

The two that differ are written by `coremlcompiler`, not by the conversion, and
`coremlcompiler` is **nondeterministic on identical input**: compiling the SAME
`.mlpackage` twice gives two `coremldata.bin`s differing in 116 bytes at offset 448 of 624
(a UUID/timestamp region), and one of the two draws reproduced the published hash exactly.
The `.mlpackage`s themselves differ only in `Manifest.json`'s minted UUIDs and in protobuf
serialization order — the MIL program is `functions equal: True` and a 200,941-line textual
dump has zero differing lines. So: **the graph, the weights and the declared contract
re-derive byte for byte; the compiler's own metadata blob does not, for anyone.** The
publish tree therefore carries the ORIGINALLY compiled B5 bytes — the ones round 4 of
review and every pin were measured against — beside the newly compiled B2 bundles, rather
than a re-compiled B5 that would differ in two files for no reason a reader could check.

## The contract

```
mel  [1, 72, 401] f32   →   embedding [1, 192] f32
```

- input feature name **`mel`**, output feature name **`embedding`**;
- both `MultiArray` `float32`; batch is **1** — this lane embeds one window, and the
  shipping embedder's batch-3 shape is a diarization-slot artifact that does not apply;
- **fixed shape, never `RangeDim`** — a flexible input takes the graph off the ANE;
- fp16 weights, `mlprogram`, `minimum_deployment_target=iOS17`;
- the output is **RAW**. Measured `‖e‖ ≈ 15.8 – 21.9` across the corpus, nowhere near 1.

### The caller's front end — 6 s of 16 kHz mono, and the Rust door must reproduce it exactly

The graph starts at the mel, so this table is part of the contract, not background. Every
entry is read out of the checkpoint's live `MelBanks` by `assert_front_end`, which fails
the run on any mismatch — the values are not transcribed from a paper.

| stage | parameters |
|---|---|
| input | 96,000 samples, 16 kHz, mono, `f32` in [-1, 1] |
| pre-emphasis | reflect-pad 1 sample on the left, `y[n] = x[n] − 0.97·x[n−1]` |
| STFT | `n_fft 512`, `win_length 400`, `hop_length 240`, `hamming_window(400, periodic=True)` zero-padded to 512, `center=True`, `pad_mode='reflect'`, `power=2.0`, `normalized=False` |
| mel filterbank | `n_mels 72`, `f_min 20.0`, `f_max 7600.0`, `norm=None`, `mel_scale='htk'` |
| log | `log(power + 1e-6)`, natural log |
| spec-norm | subtract the per-mel-bin mean over the 401 frames (`spec_norm='mn'`) |
| output | `[1, 72, 401]` — `1 + 96000 // 240 = 401` |

No waveform normalization (`norm_signal=False`) and no SpecAugment (eval).

### Why the graph starts at the mel — MEASURED, and the most important result here

The natural contract is `waveform [1, 96000] → embedding`: one graph, the whole published
function, and it is what the only existing third-party ReDimNet CoreML artifact does. It
converts cleanly and is exact in fp32 (worst cosine **0.99999994** against PyTorch). **It
is still wrong in fp16 on every compute unit**, and it is rejected on that evidence.

Reproduce with `scripts/probe_waveform_contract.py`:

| graph | fp32 vs PyTorch | fp16 CpuOnly | fp16 CpuAndGpu | fp16 All | fp16 CpuAndNeuralEngine |
|---|---|---|---|---|---|
| `waveform → embedding` (rejected) | 0.99999994 | **0.9306** | **0.9470** | **0.2770** | **0.2769** |
| `waveform → mel` alone | 0.99999976 | 0.9692 | 0.9764 | **0.0463** | **0.0463** |
| **`mel → embedding` (shipped)** | **1.00000000** | **0.99864** | **0.99990** | **0.99933** | **0.99930** |

The middle row localizes the damage: the **mel front end alone** is what breaks, and the
network — 32 `fwSE` gates, six 4-head attention blocks, ASTP with global context, the
whole `to1d`/`to2d` skeleton — is fp16-clean on every arm including the ANE.

The cause is dynamic range, failing at both ends. `MelBanks` computes a **power**
spectrogram (`power=2.0`) over a 400-sample window before taking the log:

- **high end** — a full-scale tone concentrates ~400 samples of energy into one bin; the
  squared magnitude summed across a mel filter passes fp16's 65504 ceiling. coremltools
  says so out loud while converting the waveform variant: `RuntimeWarning: overflow
  encountered in cast`.
- **low end** — the log guard is `+1e-6`, which is **subnormal** in fp16 (smallest normal
  6.10e-5). Hardware that flushes subnormals turns the guard into `log(0)`.

This is the defect class this repository already keeps a file for — `tests/fp16_guards.rs`,
issue #15, the pre-repair segmentation graph's "inert `log(epsilon = 0)`" saturating on the
default ANE placement.

One repair was attempted and **did not work**: `ct.transform.FP16ComputePrecision` with an
`op_selector` pinning the front end (the backward closure of the single `log` op, 55 ops)
to fp32. The numbers did not move — coremltools casts the graph input to fp16 *before* the
fp32 island (`waveform_to_fp16 = cast(...)` is the second op of the emitted MIL), so the
precision is already gone. That is recorded as a dead end rather than a possibility; a
genuine fp32 island would in any case force a CPU/GPU partition and give up ANE residency,
which is the thing the sweep exists to protect.

`conversion/ced` made the same call for the same reason: *"The log-mel front-end runs in
Rust (`MelExtractor`), so the graph starts at the mel."*

### There is no L2 to strip

coremlit's rule is that an embedder emits raw vectors and the crate normalizes
(`src/audio/speaker/embed/mod.rs`: *"L2 normalization is a HIGHER-level concern
(`Embedding::normalize_from`)"*). This checkpoint already complies, and the recipe proves
it at both ends rather than assuming it:

- **structurally** — `assert_raw_tail` reads the live module tree. The tail is
  `ASTP → BatchNorm1d(4608) → Linear(4608, 192)`; `emb_bn` is `false` so there is no `bn2`,
  `num_classes` is absent so there is no `cls_head`, and `forward`'s source mentions no
  normalization;
- **numerically** — the converter prints `‖e‖` for all eight corpus clips and fails if
  every one of them is 1.0.

If a future `-ft_lm` asset ever grows an L2, that assertion is what refuses it, and
removing it becomes a deliberate, visible edit.

## The window length: 6 s / 96,000 samples

A real decision, so here is the argument and the price rather than a default.

**The evidence that fixes it at 6 s.** The paper (arXiv 2407.18223) §3.2, verbatim: *"At
the finetuning stage, AAM-softmax margin was set to constant 0.5 value, **with length of
training utterances expanded to 6 seconds**."* Pretraining used 2 s (§3.1); the published
EER/minDCF numbers are scored *"utilizing full utterance length as input"* (§3.3), which
on VoxCeleb1-test averages ~8 s. So the three candidate lengths are 2 s (pretrain), 6 s
(the LM fine-tune this asset IS), and ~8 s (the evaluation protocol). The `-ft_lm` weights
are the ones being shipped, and 6 s is the regime they were optimized in. The only
existing ReDimNet CoreML artifact independently chose 96,000 samples.

It also suits the lane. A profile is the speaker's own cropped clean speech, and
enrollment naturally averages embeddings over several windows, so the window does not have
to be long enough to hold a whole utterance — it has to be the length at which the model
is best conditioned.

**What it costs.** ReDimNet never downsamples time (frequency-only striding), so cost is
linear in T except the six attention blocks, which are quadratic. Measured and computed by
`scripts/measure_window_cost.py` on this machine:

| window | samples | mel frames T | PyTorch fp32 warm | attention quadratic | 1-D activations fp16 |
|---|---|---|---|---|---|
| 2 s | 32,000 | 134 | 261 ms | 16.4 MMAC | 4.7 MiB |
| 3 s | 48,000 | 201 | 247 ms | 36.8 MMAC | 7.1 MiB |
| **6 s** | **96,000** | **401** | **417 ms** | **146.7 MMAC** | **14.1 MiB** |
| 8 s | 128,000 | 534 | 534 ms | 260.1 MMAC | 18.8 MiB |
| 10 s | 160,000 | 667 | 663 ms | 405.7 MMAC | 23.4 MiB |

**The quadratic term is not the constraint, and that corrects the framing.** At 6 s it is
146.7 MMAC against a linear part of roughly 29.6 GMAC (the paper's 9.87 GMAC at 2 s,
scaled ×3) — about **0.5%**. The measured curve is linear to within noise from 3 s on
(≈0.89 ms per mel frame plus ≈67 ms of fixed cost). Going to 10 s would cost ~1.6× the
time and ~1.7× the activation traffic, and buy a regime the weights were not fine-tuned
in. Going to 2 s would land on the *pretrain* crop, which this asset is no longer at.

In shipped precision the chosen window costs **20.4 ms warm on `CpuAndGpu`** for 6 s of
audio — ~294× real time for a single call.

**What it would take to revisit.** Mechanically, one constant: `WINDOW_SAMPLES` in
`_redimnet_common.py`; `N_FRAMES` is derived, the graph is otherwise T-agnostic, and a
re-run reproduces the conversion, parity and sweep. What would *justify* revisiting is the
thing that does not exist: **no evaluation of any ReDimNet checkpoint below full utterance
length is published, at any size, in any source** (recorded as an open item on issue #123).
Choosing 8 s over 6 s on published evidence is not currently possible; it would need an
EER/minDCF measurement at both lengths on a trial list we can licence.

## Placement — the four-arm sweep

The number the ReDimNet census line was waiting on. Arms run in **separate processes**
(`scripts/_placement_arm.py`) for two reasons: `BNNS Graph Shape Deduction` is written by
the runtime straight to fd 2 where no `sys.stderr` redirect can see it, and CoreML caches
compiled programs per process so only a fresh process gives an honest cold load.

Apple silicon, fp16 `.mlmodelc`, 8 corpus clips, warm latency = median of repeated runs.

**B5** (idle machine, reproduced twice with agreeing numbers):

| arm | load | first predict | warm predict | worst cos vs fp32 CPU | NaN-free | `BNNS Graph Shape Deduction` |
|---|---|---|---|---|---|---|
| `All` | 199 ms | 240.5 ms | 79.4 ms | 0.999329 | yes | **none** |
| `CpuAndGpu` | 164 ms | 183.1 ms | **20.4 ms** | 0.999901 | yes | **none** |
| `CpuOnly` | 104 ms | 87.0 ms | 80.1 ms | 0.998635 | yes | **none** |
| `CpuAndNeuralEngine` | 156 ms | 74.6 ms | 73.9 ms | 0.999304 | yes | **none** |

**B2** and **B2-ptn**, measured on the same host while a sibling task ran PyTorch on it
(the absolute latencies carry that load — B5 re-measured under the same load read 24.1 ms
warm on `CpuAndGpu` against 20.4 ms idle, and 459.6 ms on `CpuOnly` against 80.1 — so the
ORDERING across arms is what these tables are read for):

| B2 arm | load | first predict | warm predict | worst cos vs fp32 CPU | NaN-free | BNNS |
|---|---|---|---|---|---|---|
| `All` | 529 ms | 715.2 ms | 40.0 ms | 0.998715 | yes | **none** |
| `CpuAndGpu` | 249 ms | 403.6 ms | **12.1 ms** | 0.999894 | yes | **none** |
| `CpuOnly` | 193 ms | 41.1 ms | 54.6 ms | 0.999017 | yes | **none** |
| `CpuAndNeuralEngine` | 341 ms | 40.2 ms | 38.0 ms | 0.998845 | yes | **none** |

| B2-ptn arm | load | first predict | warm predict | worst cos vs fp32 CPU | NaN-free | BNNS |
|---|---|---|---|---|---|---|
| `All` | 249 ms | 340.7 ms | 45.0 ms | 0.998337 | yes | **none** |
| `CpuAndGpu` | 548 ms | 470.0 ms | **20.3 ms** | 0.999914 | yes | **none** |
| `CpuOnly` | 228 ms | 56.7 ms | 41.0 ms | 0.999344 | yes | **none** |
| `CpuAndNeuralEngine` | 178 ms | 66.6 ms | 32.5 ms | 0.998497 | yes | **none** |

**Every arm of every artifact loads, predicts, stays finite, and clears the floor. No arm
emits a BNNS line.** The ANE arm is not pathological for any of them: no 20× load, no 10×
predict — the opposite of what `src/audio/lid` records for its graph. And the answer the
door needs is the same for all three: `CpuAndGpu` is best on both warm latency and
numerics for each, so ONE default serves every registered artifact — a measured fact,
not an assumption that a smaller graph places like a larger one.

**This answers B4 too.** B4 and B5 are the same graph differing only in `group_divisor`, so
one conversion rules on both — and it rules on the op class the census flagged: the 32
`fwSE` gates (rank-4 → rank-2 → rank-4 round trips, visible as 32 `sigmoid` in the MIL)
present in B4/B5 and absent in B2/B6 **do not keep this graph off the accelerator**.

**But `All` is the wrong default here, and that is a finding for the Rust door.** `All`
tracks the ANE arm, not the GPU arm, on both timing (B5: 79.4 vs 73.9 ms; B2: 40.0 vs
38.0 ms) and numerics (B5: 0.999329 vs `CpuAndNeuralEngine` 0.999304, distinct from
`CpuAndGpu` 0.999901; B2: 0.998715 vs 0.998845, distinct from 0.999894) — CoreML's
heuristic sends these graphs to the ANE, where they are **~3–4× slower** than the GPU. B2
has none of B5's 32 `fwSE` gates (its 2-D blocks are `convnext_like`) and shows the same
pattern, so the op class the census suspected is not what decides the placement. The door
ships `DEFAULT_COMPUTE = ComputeUnits::CpuAndGpu`, MEASURED, with these tables as the
reason. That is the mirror image of `src/audio/lid`, where `All` is right only because
the heuristic declines the ANE — same lesson, opposite sign, and equally OS-version
dependent.

## Parity — PyTorch ↔ CoreML

`scripts/verify_redimnet.py`, fail-closed: any breach exits non-zero and `run_redimnet.sh`
halts.

| check | floor | B5 | B2 | B2-ptn |
|---|---|---|---|---|
| (a) CoreML fp32 (CPU) vs **the unmodified `ReDimNetWrap.forward`** | cos ≥ 0.9999 | worst cos **1.00000000**, max\|Δ\| 1.43e-4 | **1.00000000**, 2.41e-4 | **1.00000000**, 3.18e-4 |
| (b) fp16 `.mlmodelc` vs the fp32 CPU reference, per compute unit | cos ≥ 0.99 | All 0.99933 · CpuAndGpu 0.99990 · CpuOnly 0.99864 · ANE 0.99930 | 0.99872 · 0.99989 · 0.99902 · 0.99884 | 0.99834 · 0.99991 · 0.99934 · 0.99850 |
| (c) cross-clip cosine geometry vs PyTorch's | per-pair Δ ≤ 1e-3 | worst Δ **7.7e-6** | **1.06e-5** | **1.92e-5** |

**The thresholds, and why they are these.** The house precedent for a parity claim is
≥ 0.99 (`tests/*/placement.rs::SANITY_COS`, `conversion/*/verify_*.py::SANITY_COS_FLOOR`),
and that is where the fp16 arms are held — the shipped artifact clears it by two orders of
magnitude in the residual. The fp32-vs-fp32 floor is set far tighter at 0.9999 because
both sides compute the same arithmetic in the same precision and should agree to near
machine precision; a loose floor there would wave through a real conversion defect. Both
floors were chosen before measuring and neither was moved afterwards. Note that (a)'s
reference is the **unmodified forward on the waveform**, not the wrapper — so it measures
the whole published function even though the graph starts one module later, and the
converter separately asserts `wrapper(spec(x)) == model(x)`.

Check (c) exists because the corpus contains a **deliberate degenerate pair**: `silence`
and `dc_offset` both reduce to an all-zero mel, since a mean-normalized log-mel of any
*stationary* signal is identically zero. An absolute "no two clips may be identical"
collapse detector was written first and **measured to fire on a correct graph**; it was
replaced by a comparison against PyTorch's own cross-clip matrix, which allows identical
pairs exactly where PyTorch has them.

## Corpus (fully synthetic, licence-free)

Eight deterministic 16 kHz clips regenerated bit-for-bit from a seed
(`scripts/_fixtures.py`); no third-party audio is downloaded or committed. Every check
here is a cross-implementation comparison of the same function, which is a numerics
question rather than a speech question — so the clips are chosen to stress fp16 dynamic
range (silence, DC, single tones, an exponential sweep, full-scale noise, a clipped
square) plus one source-filter `formant` synthesis with an amplitude envelope, the closest
thing to a voice obtainable from a seed.

## Licence

**Corpus: no new exposure.** `b5-vox2-ft_lm.pt` is trained on VoxCeleb2-dev — the same
lineage the incumbent WeSpeaker embedder already carries, so the corpus layer does not
move the shipping decision.

**Weights: a step DOWN in artifact-level clarity, and the row must say so.**
`IDRnD/redimnet` ships MIT, but the grant is written over *"the Software"* and neither that
repository nor `PalabraAI/redimnet2` extends it to the released `.pt` assets in writing.
The incumbent is not in that position: WeSpeaker's own model-licence document places its
VoxCeleb-trained pretrained models under **CC BY 4.0** — an explicit weights grant with
attribution as a *condition* — which is what `tests/model_licences.rs` records today. So
this is not a step across; it is a step down, and the register row should read that way
rather than clean.

The row and the `MODELS_LOCK` table are drafted in **`LICENCE_ROW.md`** beside this file,
together with the checks that currently make them impossible to commit.

## Replay

```sh
export REDIMNET_PY=/path/to/venv/bin/python   # python 3.11, torch 2.5.0, torchaudio 2.5.0,
                                              # coremltools 8.3.0, numpy 1.26.4
export REDIMNET_CONV=/scratch/redimnet-conv   # pinned source + staging
export REDIMNET_MODELS_OUT=/scratch/models    # fp16 bundle (default: <repo>/Models/redimnet)
coremlit/conversion/redimnet/run_redimnet.sh b5      # convert → compile → manifest → verify → sweep
coremlit/conversion/redimnet/run_redimnet.sh b2      # into the same output root; the manifest
coremlit/conversion/redimnet/run_redimnet.sh b2_ptn  # step describes every bundle it finds there
```

Goldens (committed test fixtures, run by hand when the front end changes — for the
REGISTERED variant only; the script refuses any other, and refuses to overwrite a shared
front-end golden with different bytes):

```sh
REDIMNET_VARIANT=b5 python scripts/write_mel_goldens.py
```

Diagnostics, not part of the gated run:

```sh
python scripts/probe_waveform_contract.py   # why the graph starts at the mel
python scripts/measure_window_cost.py       # why the window is 6 s
```

**coremltools 8.3.0 is not a preference.** It is the version that produced the graph this
crate already ships: `Models/speakerkit/wespeaker.mlmodelc/model.mil` records
`"coremltools-version", "8.3.0"` (verified on disk, not quoted from a comment). torch 2.5.0
is coremltools 8.3.0's most recent *tested* torch — 2.5.1 converts identically but makes
coremltools print `has not been tested with coremltools`, and a recipe should not ship a
warning it could have removed.

Every stage **observes** its toolchain (`observed_toolchain()`) and aborts rather than
record a version it did not run under; `write_manifest.py` additionally refuses to describe
a build made by a different environment than the one running it. The emitted bundle records
its own producer: `coremltools-version 8.3.0`, `coremltools-component-torch 2.5.0`,
`coremlc-version 3520.5.1`, `func main<ios17>(tensor<fp32, [1, 72, 401]> mel)`.

## B2: converted, measured, not registered

**Converted and measured on 2026-09-02** through this recipe (`run_redimnet.sh b2` and
`run_redimnet.sh b2_ptn`), verified against the same floors as B5 (tables above: fp32
parity cos 1.00000000 on every clip, every fp16 arm ≥ 0.9983, cross-clip geometry within
2e-5 of PyTorch, no `BNNS Graph Shape Deduction` on any arm), and end-to-end through
`audio::identity::Embedder` — Rust mel plus the fp16 graph against PyTorch fp32 — at
worst cosine **0.99998341** (B2) and **0.99998363** (B2-ptn) on `CpuAndGpu`, against B5's
0.99998543. The two bundles, with a `CHECKSUMS.sha256` and `MANIFEST.json` covering all
three, are **preserved in the private artifact repository
`FinDIT-Studio/redimnetkit-coreml`** at a revision the owner records when uploading
(`MODELS_LOCK` stays pinned at `80c2d0a`, the B5-only revision, so CI is untouched).
Note for that day: from the new revision on, `CHECKSUMS.sha256` is kit-root-relative
(`./redimnet_b2.mlmodelc/…`, speakerkit's layout) because three bundles sharing every
file name cannot be listed bundle-relative; a shard that stages it verifies from
`Models/redimnet` rather than from inside one bundle.

**Deliberately not registered** — no `MODELS_LOCK` entry, no licence row, no gated
test, no golden, no change to the door — because of the short-segment discriminability
experiment on issue #123
([comment](https://github.com/findit-studio/coremlit/issues/123#issuecomment-5503587829)):
on the diarization fixtures no ReDimNet checkpoint beats the incumbent below 5 s, every
arm collapses at 2 s (minDCF ≈ 1.0), and at full length B5 leads by 0.013 — inside one
impostor's resolution. B2 therefore has no lane: on identity it is dominated by B5, and on
diarization there is no supporting number and the lane is blocked by the DER gate
regardless. A registered artifact nothing consumes is maintenance, so the recipe and the
bytes are kept and the registration is not.

**What "the door takes it through an unchanged contract" was proven by**, once, by hand
(not a committed gate): `audio::identity::Embedder::load` with the merged
`IDENTITY_CONTRACT` accepted `redimnet_b2.mlmodelc` and `redimnet_b2_ptn.mlmodelc` under
every compute placement and embedded a window to a finite, raw 192-d vector, while the
same contract refused the vendored silero VAD bundle. The output is in the PR that added
this section. Registering B2 later is the B5 registration with the asset, SHA-256 and pin
changed — `LICENCE_ROW.md`'s closing section lists the order.

## What is NOT here

Named rather than implied:

1. **A diarization-lane contract.** Every bundle here is the identity lane's single fixed
   window with no mask. The diarization embedder's contract is a 10 s mixture with a
   per-frame weight vector, and ReDimNet's ASTP with `global_context_att` was trained under
   no mask at all — its attention softmax and its global-context statistics would both need
   a mask semantics the checkpoint never saw. That is a design decision issue #123 records,
   not an integration detail, and no artifact here attempts it.
2. **A second reference implementation.** Issue #123's own conclusion: the DER gate is a
   cross-implementation equivalence assertion against a fixed WeSpeaker oracle and goes red
   on any embedder change. That is a diarization-lane concern and does not block the
   identity lane, but it is the reason every artifact here is converted for identity only —
   B2 included, whose 0.22× cost is a diarization-lane argument this recipe does not cash.
3. **A quality claim for `b2_ptn`.** Converted and verified as a conversion; nothing
   published evaluates the checkpoint at any length, and this recipe adds no number of its
   own. The short-segment experiment on #123 is where one would come from.
