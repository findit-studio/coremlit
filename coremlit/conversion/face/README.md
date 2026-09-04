# `w600k_r50` → CoreML, and the alignment oracle (`embeddings::face`, issue #115)

Two things live here, and they answer to different halves of the door.

- **The conversion recipe** — `run_arcface.sh` and `scripts/` — turns InsightFace's
  `w600k_r50` (the `buffalo_l` recognition head, IResNet-50, 512-d) into the fp16 CoreML
  bundle `embeddings::face`'s `FaceEmbedder` loads. **These are non-commercial research
  weights on a research-only corpus**, converted for development and CI on the standing
  basis that this repository redistributes no weight bytes; the artifact is published to a
  PRIVATE repository and rides a `commercial-`prefixed feature that is never in `default`.
  `LICENCE_ROW.md` carries the row, the gate and the exact wording the register demands.
- **The alignment oracle** — `align_oracle.py` — produces the committed alignment golden.
  It predates the conversion, it is unchanged by it, and it is documented in the second
  half of this file.

## Sources (pinned, SHA-verified at load)

| what | value |
|---|---|
| pack | `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip` |
| pack bytes / sha256 | 288 621 354 / `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` |
| **converted member** | **`w600k_r50.onnx`**, 174 383 860 bytes, `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| preprocessing / alignment reference | `deepinsight/insightface` @ `ffa12d315041c0505b077c7ff057ca914bb8dc7e` |

**InsightFace publishes no digest for this pack, and that is a finding rather than an
oversight.** `insightface/utils/storage.py` — the code every user of the Python package
runs — builds a CloudFront URL and unzips whatever arrives. There is no manifest, no
signature and no hash anywhere on that path. So the SHA-256 above is a **witness** to the
bytes this conversion consumed on 2026-09-03, not a verification against an upstream claim,
and the model card says so in those words rather than implying a check that does not exist.

`fetch_source.py` records **every** member of the pack, not only the one it converts:

| member | bytes | sha256 | used |
|---|---|---|---|
| `w600k_r50.onnx` | 174 383 860 | `4c06341c…fc619e43` | **converted** |
| `det_10g.onnx` | 16 923 827 | `5838f7fe…b85b5b91` | fixture cutting only — never converted, never published |
| `1k3d68.onnx` | 143 607 619 | `df5c06b8…1f9a45cc` | untouched |
| `2d106det.onnx` | 5 030 888 | `f001b856…a7109dbf` | untouched |
| `genderage.onnx` | 1 322 532 | `4fde69b1…6d6652fb` | untouched |

A pack-level record is the discipline issue #115's census had to invent the hard way: four
of `fal/AuraFace-v1`'s five files turned out to be byte-identical InsightFace artifacts
under terms that repository's own `apache-2.0` tag contradicted. A row keyed on a file and
its hash cannot make that mistake; a row keyed on a repository can.

## The contract

```
data [1, 3, 112, 112] f32   →   embedding [1, 512] f32   (RAW, un-normalised)
```

* both features are `MultiArray` `float32`, **fixed shape**, no `RangeDim` and no
  `EnumeratedShapes`;
* stateless — no `MLState` buffer, which `coremlit`'s face door refuses at load;
* fp16 weights, `mlprogram`, `minimum_deployment_target` **iOS17 / macOS14**;
* `convert_arcface.py` asserts all of that off the produced spec before the run is allowed
  to finish, so a conversion that silently emitted a flexible input fails the recipe rather
  than the door.

The CoreML feature names are **not** the ONNX's. That graph was traced out of PyTorch 1.9
and its features are called `input.1` and `683` — a tracer's counters, not a contract.
`data` is InsightFace's own MXNet-era name for this tensor and `embedding` is what every
other `coremlit` embedder calls its output. The ONNX names are recorded in `MANIFEST.json`
so the cross-platform `ort` twin can bind them.

### Why this conversion is ours

Both third-party CoreML ArcFace builds on the Hub declare **`ImageType`** inputs. `coremlit`'s
`Features` binds `MultiArray` only, so neither can be fed at all — that was the third
blocker in the face PR. Converting from the ONNX ourselves removes it, and it also puts the
preprocessing where the crate wants it: in the artifact's manifest as data, rather than in
CoreML's own scale/bias where the door cannot read it back.

### There is no L2 to strip — established, not assumed

`coremlit`'s contract is that the **door** normalises, so an artifact that normalised
internally would double-count. `probe_onnx_contract.py` settles it two ways before anything
is converted:

* **structurally** — the graph's entire op set is `Conv` 53, `BatchNormalization` 26,
  `PRelu` 25, `Add` 24, `Flatten` 1, `Gemm` 1. None of `LpNormalization`,
  `L2Normalization`, `Normalize`, `MeanVarianceNormalization`, `ReduceL2`, `Sqrt`, `Div`,
  `Reciprocal` or `Pow` appears anywhere, so there is no L2 and no decomposition of one. The
  tail is `BatchNormalization → Flatten → Gemm → BatchNormalization`: the standard
  InsightFace IResNet head, `bn2 → flatten → fc → features(BN1d)`, which is exactly the
  pre-norm 512-d feature;
* **numerically** — measured over the 18 fixture faces, `‖e‖` runs **17.01 – 24.91**
  (median 21.27). A graph that normalised would put every one of those at 1.

Nothing was stripped, and the recipe would have failed rather than strip something quietly:
the probe treats a normalising op as a hard error with instructions, not as a branch.

### Channel order: RGB, read off InsightFace's code and then measured

`Preprocessing::ARCFACE` says RGB, NCHW, `(x − 127.5) / 127.5`. Every part of that is
checked against InsightFace rather than transcribed:

* `ArcFaceONNX.__init__` **derives** `input_mean` and `input_std` by scanning the first
  eight node names for a `Sub` and a `Mul` — if both are present the preprocessing is fused
  into the graph (the MXNet-era export) and the caller applies nothing. `probe_onnx_contract.py`
  runs that derivation on the actual file rather than quoting its constant; the first node
  is a `Conv`, so the branch taken is the caller-applied one and the answer is
  `input_mean = 127.5`, `input_std = 127.5`;
* the **order** follows from `ArcFaceONNX.get_feat` calling
  `cv2.dnn.blobFromImages(imgs, 1.0/127.5, size, (127.5,)*3, swapRB=True)` over a crop that
  `face_align.norm_crop` warped out of an OpenCV **BGR** frame. `swapRB=True` therefore
  hands the model **RGB**.

And then it is measured, because a source reading is an argument and this is a place where
being wrong is silent. Embedding the identical 18 crops with the channels reversed:

| feeding | min same-person | max different-person | margin |
|---|---|---|---|
| **RGB** (shipped) | **0.2891** | 0.1407 | **+0.1484** |
| BGR | 0.2547 | 0.1532 | +0.1015 |

Mean `1 − cos` between the two feedings is **0.0652**. The consequence is sharper than the
margin: under BGR the worst same-person pair falls to 0.2547, **through InsightFace's own
0.28 "same person" line** — Peggy Whitson's 2002 profile and her 2018 frontal stop being the
same person. `verify_arcface.py` fails the run if BGR ever separates identities at least as
well as RGB, so the finding is a gate and not a paragraph.

### Why batch 1

The ONNX declares a symbolic batch (`['None', 3, 112, 112]`); the CoreML input pins it to 1.

* A flexible CoreML input is off the Neural Engine for **every shape but its default**
  (Apple developer forum 724930), and coremltools #2370 measured ANE residency going 78 % →
  0 % under a `RangeDim`. Even `EnumeratedShapes` costs 3–4× against a dedicated fixed
  shape.
* `coremlit`'s face door refuses a non-`Fixed` geometry at load anyway: `FeatureInfo::shape`
  reports a flexible feature's **default**, so the batch it read back would be a default
  rather than a fact.
* The door chunks a slice to whatever capacity it reads back, so batch 1 and batch 8 are the
  same call site. Batch 1 is the honest first artifact; issue #115's census sized a fixed
  batch 2 at ~1.7× on the ANE, plateauing thereafter, so a batch-8 export is a follow-up
  **with a measured throughput reason** rather than a default nobody asked for.

### The ONNX → PyTorch hop, and why it is checked

`coremltools` has no ONNX front end any more — `coremltools.converters` exposes libsvm,
lightgbm, sklearn and xgboost and nothing else. The graph is therefore rebuilt as a
`torch.nn.Module` by `onnx2torch`, traced, and handed to the torch front end. That is a
place a conversion can go wrong silently, so `convert_arcface.py` refuses to convert a
module it has not first checked against `onnxruntime` on the real input domain
(`uniform(−1, 1)`, which is where preprocessed pixels land): **worst cosine
0.999999999991, worst |diff| 7.7 × 10⁻⁶ over 8 trials**, against floors of 0.99999 and
10⁻⁴ set in the file above the code that measures them. 43 572 288 parameters, rebuilt.

## Parity — ONNX ↔ CoreML

`onnxruntime` 1.20.1 on CPU in fp32 is the reference. Cosine over the 18 fixture faces:

| path | min | median | max | worst `1 − cos` |
|---|---|---|---|---|
| CoreML **fp32**, CpuOnly | 1.0000000 | 1.0000000 | 1.0000000 | 3.3 × 10⁻¹² |
| CoreML fp16, `All` | 0.9997810 | 0.9998419 | 0.9998794 | 2.2 × 10⁻⁴ |
| CoreML fp16, `CpuAndGpu` | 0.9999987 | 0.9999991 | 0.9999993 | 1.3 × 10⁻⁶ |
| CoreML fp16, `CpuOnly` | 0.9998353 | 0.9999079 | 0.9999352 | 1.6 × 10⁻⁴ |
| CoreML fp16, `CpuAndNeuralEngine` | 0.9997798 | 0.9998398 | 0.9998802 | 2.2 × 10⁻⁴ |

**Both floors were set before the measurement**, in `verify_arcface.py` above the code that
measures against them: 0.9999 for fp32 (same precision on both sides, so this is the
conversion itself with no precision story to hide behind) and **0.99** for fp16 — issue
#115's own gate, placed ~4× above the ANE's fp16 noise and ~8× below the cheapest real
preprocessing bug. It is not tightened to 0.999, for the reason that issue records at
length.

**fp16 does not break parity, so fp16 ships.** The census predicted `1 − cos ≈ 0.0015`
typical / `0.0025` worst on the ANE for an IResNet, from a measurement on IResNet-**100**;
this is IResNet-**50**, whose 24 residual `Add` chains `probe_onnx_contract.py` counted
directly, and a 100-layer backbone has roughly twice as many places for fp16 error to
accumulate. The observed worst is **2.2 × 10⁻⁴** — about 7× better than the prediction and
45× inside the gate's `1 − cos ≤ 0.01` budget. There was no fp32-versus-fp16 decision to
make; had there been one, the number and not a preference would have made it.

The GPU arm's `1.3 × 10⁻⁶` against the ANE's `2.2 × 10⁻⁴` reproduces the census's finding
that the two differ by ~100×: the GPU stores fp16 and accumulates fp32, and the ANE does
not.

## Placement — the four-arm sweep

Each arm loads the compiled fp16 bundle in a **fresh process** (so the load is genuinely
cold and stderr is attributable) and predicts every fixture face, 100 warm predicts per
round, five rounds.

| arm | cold load ms | first predict ms | warm predict ms (median, range) | faces/s | min cos vs fp32 CoreML | BNNS |
|---|---|---|---|---|---|---|
| `All` | 357 | 71 | 4.46 (3.57 – 6.41) | 224 | 0.999781 | clean |
| `CpuAndGpu` | 592 | 291 | 8.66 (4.86 – 10.59) | 115 | 0.999999 | clean |
| `CpuOnly` | 95 | 181 | 10.42 (9.92 – 23.09) | 96 | 0.999835 | clean |
| **`CpuAndNeuralEngine`** | **160** | **68** | **3.48 (2.99 – 3.76)** | **287** | 0.999780 | clean |

**Recommendation: `CpuAndNeuralEngine`.** It is the fastest arm, it has the tightest spread
of the four, its cold load is second-cheapest, and its parity sits 45× inside the gate. No
arm emitted a `BNNS Graph Shape Deduction` line — the negative result `src/audio/lid`
records looks nothing like this — which is consistent with the census's prediction that
Conv/BatchNorm/PReLU/Add/Flatten/Gemm is entirely ANE-friendly. ReDimNet-B5's sweep chose
`CpuAndGpu`; nothing about that transferred, and it was not assumed.

**`All` is not the same thing as pinning the ANE, and that is the second result here.**
CoreML's own default policy is 1.3× slower at the median and swings across 3.57 – 6.41 ms
where the pinned arm holds 2.99 – 3.76. A caller that leaves the compute units at their
default gets a slower and less predictable embedder than one that asks for the ANE.

**Five rounds, not one, and the reason is in the data.** The first two single-pass runs of
this sweep disagreed about the winner — `CpuAndNeuralEngine` 4.38 ms against `All` 9.91 ms
in one, then 4.90 against 4.84 in the next. A warm predict of a few milliseconds on a shared
desktop is within reach of whatever else the machine is doing, and a recommendation taken
from one draw is a recommendation about the machine's mood. `sweep_placement.py` now
aggregates rounds by median and prints the per-round range beside it.

## Throughput

**287 faces/s** on the recommended arm, warm, on the machine in the toolchain table below.

The method, stated because a throughput number without one is a slogan: one model loaded
once, **one predict per face** — the artifact's batch is 1, so a keyframe with N faces is N
predicts — timed as the median of 100 back-to-back predicts after the graph is hot, then the
median of five such rounds, each in its own process. `1000 / 3.48 ms = 287`. Cold, the first
face additionally costs a 160 ms load and a 68 ms first predict, so a process that embeds a
single face and exits sees ~230 ms rather than 3.5.

## Known pairs

18 same-person pairs and 135 different-person pairs over 6 identities, at **InsightFace's
own operating point** — `sim >= 0.28` "They ARE the same person", `sim < 0.2` "They are NOT
the same person", from `web-demos/src_recognition/main.py`
(@ `f8aa2c17e18044a86bbfa04be40e00cd2ff40a4f`, sha256 `24a94180…9509`). The threshold is not
this recipe's and is not fitted to this set.

| embedding path | min same-person | max different-person | margin |
|---|---|---|---|
| ONNX fp32 (`onnxruntime`) | 0.2891 | 0.1407 | +0.1484 |
| CoreML fp16, `All` | 0.2882 | 0.1418 | +0.1464 |
| CoreML fp16, `CpuAndGpu` | 0.2892 | 0.1407 | +0.1485 |
| CoreML fp16, `CpuOnly` | 0.2898 | 0.1386 | +0.1512 |
| CoreML fp16, `CpuAndNeuralEngine` | 0.2883 | 0.1416 | +0.1467 |

Every same-person pair clears 0.28 and every different-person pair is under 0.20, on every
path, and the two populations do not touch: same-person spans 0.289 – 0.757, different-person
spans −0.098 – 0.141.

**The binding pair is the frontal-to-profile one**, which is the point of having built the
set that way. `whitson_iss005e07178` — 2002, a full side view, yaw proxy −0.82 — against
`whitson_NHQ201803020004` — 2018, frontal — is 0.2891, the minimum in every row above: 16
years and a profile apart, and still on the accept side of InsightFace's own line. (Side
view is what the photograph shows. The yaw proxy the fixture rule cuts on is a monotone
stand-in, not calibrated to degrees, so no angle is claimed from it.) Issue #115 measured
AuraFace splitting identity on frontal-to-profile pairs 38.55 % of the time against
`buffalo_l`'s 2.22 %; this is that regime, and this artifact holds it.

The worst different-person pair is `lindgren_NHQ202009160011` against
`hague_NHQ202001130002` at 0.1407 — two frontal studio portraits of similar-looking men,
which is the hardest impostor shape a set this size can offer.

**Six identities cannot estimate a false-accept rate and none is claimed.** What is claimed
is exactly what the table says: at a threshold taken from upstream, no pair in this set is
misclassified, on any compute arm, by either the ONNX or the CoreML path.

## The fixture corpus

18 photographs of 6 people, every one a **work of the U.S. federal government in the public
domain**, from NASA's image library. No LFW, no CelebA, no CFP, no WebFace, nothing scraped.
Per-image provenance, the selection rule, the rejected candidates and the licence basis are
in `coremlit/tests/face/fixtures/PROVENANCE.md`; `build_fixtures.py` pins each source
asset's SHA-256 and refuses to cut a crop from bytes that have moved.

Two things there are worth repeating because they are about method rather than about faces:
a NASA caption naming a person is **not** evidence that person's face is the one in the
frame (one candidate passed every mechanical rule and showed somebody else), and an
identical-twin identity was removed from the set entirely rather than left to make a
different-person pair that no embedder should be expected to split.

## Licence

**Research only, at both layers, and a conversion changes neither.**

| layer | terms |
|---|---|
| this recipe | MIT OR Apache-2.0, with the rest of `coremlit`. Covers the recipe. Covers nothing below it. |
| **weights** (`w600k_r50.onnx`, and so the bundle) | **research-only.** InsightFace's model zoo: *"ALL models are available for non-commercial research purposes only."* No commercial licence is offered for them. |
| **corpus** (WebFace600K) | **research-only.** WebFace260M/WebFace600K is released under a licence agreement restricting use to non-commercial academic research. |

`LICENCE_ROW.md` carries the `model_licences.rs` row verbatim, the `MODELS_LOCK` table, the
`commercial-face-arcface` feature with the first sentence the register's doc rule demands,
and the reason none of them could land before the artifact was published.

## Replaying the conversion

```sh
export ARCFACE_PY=/path/to/venv/bin/python
coremlit/conversion/face/run_arcface.sh            # the whole recipe
coremlit/conversion/face/run_arcface.sh fixtures   # re-cut the committed fixtures only
coremlit/conversion/face/run_arcface.sh reference  # re-cut onnx_reference.json only
```

The `reference` stage is the one that needs no torch and no coremltools: it observes only
the packages it imports (numpy, onnxruntime), because a stage that records a version it did
not run under is the defect issue #97 named.

Every stage verifies the source against its pin before it reads it, and every stage
**observes** its toolchain and aborts rather than record a version it did not run under.
`write_manifest.py` additionally refuses to describe a bundle whose `producer.json`
toolchain differs from the one running it, so a manifest cannot describe a build made in a
different environment.

## The conversion's observed toolchain (per #97: observed, not a literal)

The bundle in the publish tree was produced by this exact stack.

| component | observed |
|---|---|
| macOS | 26.5, arm64 |
| Xcode | 26.6 (17F113) |
| `coremlcompiler` | `/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/coremlcompiler` |
| python | 3.11.15 |
| numpy | 1.26.4 |
| torch | 2.5.0 |
| coremltools | 8.3.0 |
| onnx | 1.17.0 |
| onnxruntime | 1.20.1 |
| onnx2torch | 1.5.15 |
| Pillow | 11.0.0 |
| `w600k_r50.mlmodelc/model.mil` sha256 | `050f69f10f5687971fb8f808d9da53b01a8d512c7013346fbc22daa948e42d26` |
| `w600k_r50.mlmodelc/weights/weight.bin` sha256 | `aa08d7826a70f9bc237ea0532a5eec12cb83b8375148a1b0650f104cbb2ff492` |

coremltools 8.3.0 is not a preference — it is the version that produced every other bundle
this crate ships. `coremlcompiler` is **not** pinned by the Python venv (a different Xcode
compiles different bytes from identical input, and the ReDimNet recipe measured
`coremldata.bin` differing between two compiles of the *same* `.mlpackage`), so the row above
records which one ran.

## What is NOT here

* **Nothing about the registration — that landed.** The `MODELS_LOCK` table
  (`kit = "arcface"`), the `commercial-face-arcface` feature, the licence row, the `ci.yml`
  shard and the four gated suites are all in the tree; `LICENCE_ROW.md` records what was
  written out ahead of time and the one field of it that had to change.
* **No batch-8 export.** A follow-up with a measured throughput reason, not a default.
* **No LIVE ONNX twin.** Issue #115's cross-platform acceptance is cosine ≥ 0.99 between
  the CoreML and ONNX outputs, and `tests/face/parity.rs` now asserts exactly that in CI —
  against `tests/face/fixtures/onnx_reference.json`, the fp32 vectors this recipe cut and
  committed (`scripts/write_onnx_reference.py`). What is still absent is the `ort` road
  itself: no gate runs an ONNX session, so a reference that needs re-cutting is a
  deliberate regeneration rather than something CI recomputes.
* **No accuracy benchmark.** CFP-FP, IJB-C and the rest are measured on research-licensed
  corpora this repository will not consume. The accuracy case for `buffalo_l` is issue
  #115's, taken there against its own out-of-tree measurements; what is measured *here* is
  that the conversion preserves the model, not that the model is good.

---

# The alignment oracle

`align_oracle.py` predates the conversion above and is unchanged by it. It produces the
committed alignment golden — `tests/face/align_golden.rs`'s expected pixels — and it is also
what `build_fixtures.py` warps the known-pairs crops with, so the fixtures and the golden are
one specification. Alignment is the one place in this kit where a wrong answer raises no
error and moves every downstream cosine, so it gets an oracle written independently of the
code it checks.

## `align_oracle.py`

Reproduces `deepinsight/insightface`'s `estimate_norm` + `norm_crop`
(`python-package/insightface/utils/face_align.py`, commit
`ffa12d315041c0505b077c7ff057ca914bb8dc7e`, 2022-12-17), and writes two raw
RGB8 fixtures into `coremlit/tests/face/fixtures/`.

It imports **numpy only**. `skimage` and OpenCV are deliberately not used (and
are not installed here), so nothing in this file can quietly become a call into
the thing it is meant to check.

**What is independent, and what is a specification reproduced twice** — stated
here because "oracle" is worth nothing if the reader has to guess which half is
evidence:

- the **solve** is independently derived. InsightFace's `estimate_norm` calls
  `skimage`'s `SimilarityTransform.estimate`, which is Umeyama (1991) by way of
  an SVD with a determinant sign correction; neither this script nor the Rust
  runs an SVD, both reaching the same minimiser through the **complex/linear**
  formulation, where writing the scaled rotation as `[[a, -b], [b, a]]` makes
  the residual linear in `(a, b, tx, ty)` and the answer two dot products. The
  Rust's own optimality gate (`recovered_transform_is_the_least_squares_minimiser`)
  names no formula at all, so the golden is not the only leg. **Both sides
  evaluate that minimiser in `f64` where `skimage` evaluates it in `f32`, which
  is a divergence and not a rounding** — see "The solve is not bit-exact with
  `skimage`" below;
- the **resampler** is not independent. `cv2.warpAffine`'s `INTER_LINEAR` has
  exactly one right answer and both sides reproduce it, so byte agreement here
  catches a transcription slip and nothing more.

**The resampler is bit-exact with OpenCV, and that replaced a false claim.**
This file and the Rust module used to resample in `f64` and record the
divergence from OpenCV as "at most one LSB per channel". That was a measurement
on one fixture stated as a bound over the domain, and it is false: OpenCV
quantises the inverse-mapped coordinate onto a five-bit grid *before* choosing
a weight, so a fraction under `1/64` collapses to zero and takes the pure left
pixel. Measured against `cv2.warpAffine` over ArcFace-shaped warps of random
crops (`opencv-python-headless` 4.12.0, 451 584 bytes), the float sampler
differed on **11.6 % of bytes, worst case 6 levels**. Both files now implement
OpenCV's fixed-point pipeline, with every constant named after the OpenCV
symbol it comes from.

**Which OpenCV — the version is part of the contract.** The 4.x line: it is
what the pinned `face_align.py` runs against and what every published ArcFace
accuracy number was measured on. OpenCV **5.0 replaced the fixed-point path
with a float one** and is a different function — against 5.0.0 these same
committed bytes differ on 8 488 of 37 632, by up to 5 levels.

## The solve is not bit-exact with `skimage`, and there is no single `skimage` to be exact against

`skimage`'s `_umeyama` (`skimage/transform/_geometric.py` v0.19.3, L107-149)
keeps its **`f32`** input through the centroids, the covariance and the SVD,
storing only the result as `f64`. This script and the Rust promote to `f64`
first. Same minimiser, different numbers — enough to move a five-bit source
coordinate on **10 of 12 544** destination pixels for the witness landmarks
`align_oracle.py --reference-divergence` reports.

That gap is real. What makes it unclosable is that `_umeyama`'s `f32` path is
two library calls — a `sgemm` for the covariance, a `sgesdd` for the 2×2 SVD —
and neither is specified past returning *a* correct answer. Running the
identical `_umeyama` source, same machine, same landmarks, under numpy's
OpenBLAS 0.3.33 and Apple's Accelerate:

| measured over | OpenBLAS vs Accelerate |
|---|---|
| `f32` covariance, face-like landmark sets | 16 618 of 20 000 differ |
| `f32` `sgesdd` singular values | 13 657 of 20 000 differ |
| `f32` `sgesdd` rotation `U @ V` | 20 000 of 20 000 differ |
| `_umeyama` results that are not a similarity | 19 624 of 20 000 (Accelerate); 0 (OpenBLAS) |
| five-bit coordinates, the witness | **15** of 12 544 differ |
| five-bit coordinates, 20 000 face-like sets | mean **14.8**, median 11, worst 212 |

The two builds disagree with **each other** by more than this crate disagrees
with either (10 against OpenBLAS, 5 against Accelerate). OpenBLAS's aarch64
`sgemm` contracts its multiply-adds into `fma` — an `fma` chain reproduces it on
3 000 of 3 000 random inputs, a non-fused chain on 341 — and whether a kernel
contracts is a build flag, not a specification.

There is a structural obstruction too: a `f32` `U @ V` is only approximately
orthogonal, so under Accelerate 98 % of `_umeyama` results are not exactly
similarities — and under OpenBLAS none of them fail to be, which is itself the
point: the property belongs to the build, not to the reference. Rust's
`SimilarityTransform` stores `(a, b)` and cannot represent a shear, so the
reference's own output is routinely not a value that type can hold.

So "bit-exact with `skimage`" is a property of `skimage` *and the BLAS the
measuring machine linked*, not of `skimage`. This crate therefore claims only
what it can hold: the least-squares similarity minimiser of the `f32`
landmarks, evaluated in `f64` rather than in the reference's `f32`, handed to a
resampler that is bit-exact with `cv2.warpAffine` 4.x.
`the_solve_diverges_from_skimage_by_less_than_skimage_diverges_from_itself`
commits all three counts, so closing the gap toward one build turns that test
red on the gap that has no single target.

Deciding it on accuracy rather than on bit-exactness needs the embedding drift
the divergence causes, measured against a staged artifact. There is none (see
`src/embeddings/face/mod.rs`), so the divergence is recorded rather than traded
away.

```sh
python3 coremlit/conversion/face/align_oracle.py --reference-divergence --sweep 20000
```

prints every number on this page: the witness, all three matrices, the three
witness counts that `src/embeddings/face/align/tests.rs` commits as constants,
the accumulation identification, and the table above (about ten seconds; drop
`--sweep` for the matrices and the three counts alone). **This mode, unlike the
golden,
depends on the BLAS**: that is what it exists to demonstrate. It needs no
skimage and no OpenCV, only numpy plus — for the second backend — Apple's
Accelerate through `ctypes`, whose bridge asserts that it reproduces a matmul
and reconstructs an SVD before anything is measured through it. Where
Accelerate is unavailable the mode says so and reports the one backend it has.

## Regenerating the alignment golden

```sh
python3 coremlit/conversion/face/align_oracle.py
```

It prints the solved 2×3 matrix and both fixtures' SHA-256. Those digests are
pinned in `tests/face/align_golden.rs` (`CROP_SHA256`, `EXPECTED_SHA256`) and
the matrix in `ORACLE_MATRIX`, so a regeneration is a deliberate three-place
diff and never a silent re-baseline.

## The oracle's own observed toolchain (per #97: observed, not a literal)

The committed fixtures were produced by this exact stack:

| component | observed |
|---|---|
| macOS | 26.5, arm64 |
| python | 3.14.6 (`/opt/homebrew/opt/python@3.14/bin/python3.14`) |
| numpy | 2.5.1 |
| `align_oracle.py` sha256 | `e96bf45ab2a5a712e7b5f2f9029958cced273f8172f36a483ba0594dede30ba4` |
| `align_crop_64x48_rgb8.bin` sha256 | `a7d34a19107058c28c73633cc25b82a018fc279034d6670b45488022d5071ce0` |
| `align_expected_112x112_rgb8.bin` sha256 | `274b92b0002ab01af0c8967372b2aea7bf5a71096308f257ea50168cf671f13c` |

The solve is IEEE-754 `f64` throughout with no BLAS call on the hot path and
the resampler is integer arithmetic after it, so a different numpy or Python is
expected to reproduce these **fixture** bytes; the row records what was actually
run, not what is required.

`--reference-divergence` is the exception and deliberately so: its numbers are
BLAS-dependent by construction, and the row above pins the build the committed
`SKIMAGE_OPENBLAS` constant came from (numpy 2.5.1, OpenBLAS 0.3.33, aarch64).
A different build will print a different matrix, which is the finding.

The OpenCV comparisons quoted above were run **out of tree**, in a throwaway
virtualenv, purely to check this reproduction against the thing it reproduces.
Nothing in the repository imports OpenCV and no gate depends on it: the numbers
are recorded here as an observation, and what the committed tests actually hold
is the Rust and this script agreeing on all 37 632 bytes plus the unit gate
`a_fraction_below_the_five_bit_half_step_takes_the_pure_left_pixel`, which pins
the one behaviour separating the fixed-point pipeline from a float one.
