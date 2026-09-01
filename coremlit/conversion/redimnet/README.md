# ReDimNet-B5 → CoreML conversion (identity lane, issue #123)

Re-derives the ReDimNet-B5 speaker-embedding CoreML graph from the OFFICIAL public
checkpoint, deterministically. Mirrors the `conversion/ced` recipe (sub-forward wrapper →
trace → `ct.convert` fp16+fp32 → compile → manifest → fail-closed verify), plus a
four-arm placement sweep.

**Status: discovery.** There is no Rust door for this model yet, no test module, no CI
shard, and no published artifact repository. What this recipe establishes is the
**contract** the door will be written against, and the placement/parity evidence the
decision rests on. See "What is NOT here" at the end for the exact list of what a
follow-up must add, and why the licence row cannot be committed today.

## Sources (pinned, SHA-verified at load)

| what | pin |
|---|---|
| weights | `IDRnD/redimnet` release `latest` → **`b5-vox2-ft_lm.pt`**, 31,174,382 bytes, sha256 `8b0c11bbf5a3a8bb39e5c072c4192d0b694d8c447cf126d4cd3c7346a04b39c8` |
| model source | `github.com/IDRnD/redimnet` @ `ce039a624cb99fe127702ceb94c6080090e5032f` |

The release tag is literally named `latest` and is **mutable**, so the tag is not the
lock — the SHA-256 is, and every stage verifies it. The model source is pinned too: a
checkpoint is only half the provenance, because `ReDimNetWrap` is *reconstructed* from the
archive's own `model_config` and the reconstructing code decides what the weights compute.

`model_config`, read out of the archive and asserted entry for entry at load:
`C 32`, `F 72`, `block_1d_type conv+att`, `block_2d_type basic_resnet_fwse`,
`group_divisor 16`, `hop_length 240`, `out_channels null`, `pooling_func ASTP`,
`global_context_att true`, `embed_dim 192`, `emb_bn false`. 1,052 tensors, 7,709,351
parameters, `load_state_dict` with zero missing and zero unexpected keys.

**Only the `-vox2-` lineage.** The same release publishes `M-vb2+vox2+cnc-ft_mix.pt` and
`S-vb2-ptn.pt`, trained on VoxBlink2, whose authors state the CC BY-NC-SA 4.0 term
propagates to the trained model. `_redimnet_common.verify_asset_name` refuses any asset
whose name is not `-vox2-`; it is a guard, not decoration.

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

Apple silicon, fp16 `.mlmodelc`, 8 corpus clips, warm latency = median of 30 runs.
Reproduced twice with agreeing numbers.

| arm | load | first predict | warm predict | worst cos vs fp32 CPU | NaN-free | `BNNS Graph Shape Deduction` |
|---|---|---|---|---|---|---|
| `All` | 199 ms | 240.5 ms | 79.4 ms | 0.999329 | yes | **none** |
| `CpuAndGpu` | 164 ms | 183.1 ms | **20.4 ms** | 0.999901 | yes | **none** |
| `CpuOnly` | 104 ms | 87.0 ms | 80.1 ms | 0.998635 | yes | **none** |
| `CpuAndNeuralEngine` | 156 ms | 74.6 ms | 73.9 ms | 0.999304 | yes | **none** |

**Every arm loads, predicts, stays finite, and clears the floor. No arm emits a BNNS line.**
The ANE arm is not pathological: no 20× load, no 10× predict — the opposite of what
`src/audio/lid` records for its graph.

**This answers B4 too.** B4 and B5 are the same graph differing only in `group_divisor`, so
one conversion rules on both — and it rules on the op class the census flagged: the 32
`fwSE` gates (rank-4 → rank-2 → rank-4 round trips, visible as 32 `sigmoid` in the MIL)
present in B4/B5 and absent in B2/B6 **do not keep this graph off the accelerator**.

**But `All` is the wrong default here, and that is a finding for the Rust door.** `All`
tracks the ANE arm, not the GPU arm, on both timing (79.4 vs 73.9 ms) and numerics
(0.999329 vs `CpuAndNeuralEngine` 0.999304, distinct from `CpuAndGpu` 0.999901) — CoreML's
heuristic sends this graph to the ANE, where it is **3.9× slower** than the GPU. So the
door should ship `DEFAULT_COMPUTE = ComputeUnits::CpuAndGpu`, MEASURED, with this table as
the reason. That is the mirror image of `src/audio/lid`, where `All` is right only because
the heuristic declines the ANE — same lesson, opposite sign, and equally OS-version
dependent.

## Parity — PyTorch ↔ CoreML

`scripts/verify_redimnet.py`, fail-closed: any breach exits non-zero and `run_redimnet.sh`
halts.

| check | floor | measured |
|---|---|---|
| (a) CoreML fp32 (CPU) vs **the unmodified `ReDimNetWrap.forward`** | cos ≥ 0.9999 | worst cos **1.00000000**, worst max\|Δ\| 1.43e-4 |
| (b) fp16 `.mlmodelc` vs the fp32 CPU reference, per compute unit | cos ≥ 0.99 | All 0.99933 · CpuAndGpu 0.99990 · CpuOnly 0.99864 · CpuAndNeuralEngine 0.99930 |
| (c) cross-clip cosine geometry vs PyTorch's | per-pair Δ ≤ 1e-3 | worst Δ **7.7e-6** |

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
coremlit/conversion/redimnet/run_redimnet.sh  # convert → compile → manifest → verify → sweep
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

## What is NOT here

Named rather than implied, because each is a precondition for the next step:

1. **The Rust door.** `EmbedModel::from_file_with` hard-requires `waveform [3, 160000]` /
   `mask [3, F]` / `embedding [3, 256]` and those feature *names*, so no ReDimNet can load
   through it. This needs a new door in the shape of `audio::lid` (#100), and it must carry
   the front end above — `MEL_FRONT_END` is the specification, and `mel_for_waveform()` is
   the oracle its goldens should be cut against.
2. **Goldens.** Deliberately not generated: the shape of a golden corpus is decided by the
   door's API, which does not exist. `scripts/_fixtures.py` is written so the same clips can
   be reused.
3. **`MODELS_LOCK` + the licence row + a CI shard.** These three are coupled and none can
   land alone — see `LICENCE_ROW.md`.
4. **A second reference implementation.** Issue #123's own conclusion: the DER gate is a
   cross-implementation equivalence assertion against a fixed WeSpeaker oracle and goes red
   on any embedder change. That is a diarization-lane concern and does not block the
   identity lane, but it is the reason this recipe converts B5 for identity only.
