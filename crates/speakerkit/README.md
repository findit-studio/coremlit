# speakerkit

Native CoreML **inference backend** for speaker diarization: it runs
pyannote's segmentation net and the WeSpeaker embedding net on the Apple
Neural Engine (via [`coremlit`](../coremlit)) and produces the tensors that
feed [`diaric`](https://github.com/findit-studio/diaric)'s Rust VBx/PLDA
clustering. Product driver:
[`findit-studio/desktop#120`](https://github.com/findit-studio/desktop/issues/120)
(segmentation ~20x / embedding ~30x ANE uplift targets).

**The clustering algorithms are `diaric`'s, not this crate's.** speakerkit's own
work is native CoreML inference: it runs the segmentation and embedding nets on
the ANE and produces `diaric`-shaped tensors (`Extraction`). It does not
reimplement clustering. What it *does* provide — as of the clustering phase — is
a thin runtime clustering *stage* (`Extraction::diarize` / `diarize_with` /
`diarize_online`) that turns those tensors into speaker-labelled spans by
delegating to one of `diaric`'s two engines: the offline pyannote-community-1
pipeline (the default, DER-gated) or the online FluidAudio-semantics matcher.
See [Clustering](#clustering) below.

**Sans-I/O**, like `whisperkit`: audio enters as 16 kHz mono `&[f32]`. No
file I/O, no resampling, no device capture, no async. macOS only (Apple
Silicon; built on `coremlit`'s safe CoreML layer).

## Quick start

```rust,no_run
use speakerkit::extract::Options;
use speakerkit::source::{AnySource, ModelSource, Source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 16 kHz mono samples — this crate performs no I/O or resampling.
    let audio: Vec<f32> = vec![0.0; 16_000];

    // `Source::FluidAudio` is the default (see "Licensing" below for why).
    let options = Options::new().with_source(Source::FluidAudio);
    let source = AnySource::load("Models/speakerkit", options)?;
    let extraction = source.extract(&audio)?;

    // `extraction` now holds exactly the tensors `diaric`'s offline diarizer
    // consumes. `diaric` is a non-optional runtime dependency, so clustering is
    // available directly (see "Clustering" below):
    //   extraction.diarize(&plda)  ->  diaric::offline::OfflineOutput
    // or extraction.into_offline_input(&plda) for the raw diaric::offline::OfflineInput.
    println!(
        "{} chunks, {} output frames",
        extraction.num_chunks(),
        extraction.num_output_frames()
    );
    Ok(())
}
```

To use argmax's models instead, swap the source *and* the models directory in
the snippet above — the two vendors ship different on-disk layouts, so
`models_root` means a different thing per source (see below):

```rust,ignore
let options = Options::new().with_source(Source::Argmax);
let source = AnySource::load("Models/argmax-speakerkit", options)?;
```

Feature flags: `dia-oracle` (test-only — pulls in `dia`'s `ort`-backed ONNX
reference oracle, `dia/ort` + `dia/bundled-segmentation`, that the end-to-end
DER parity suites score the CoreML path against; it is what the `parity_e2e` /
`parity_shipping_der` / `generate_goldens` binaries compile under, and is never
meant for production) and `serde` (`Serialize`/`Deserialize` on `Options` and
friends). Neither is on by default.

The `diaric` core crate is the **non-optional runtime dependency** — the
clustering stage (`diarize` / `diarize_with` / `diarize_online`) and
`Extraction::into_offline_input` are always available, no feature required. It
is pinned to the public `diaric` repo by an **exact git rev, not a path
dependency** on a sibling checkout (a path dep breaks a fresh `cargo metadata`).
`diaric` is backend-free by construction — the ONNX/Torch model runners live in
the separate `diarization` crate, never here — so the runtime clustering path
pulls no ONNX Runtime and no bundled ONNX model, keeping ORT out of CoreML-only
deployments. (The `diarization` crate is pulled in only by the test-only
`dia-oracle` feature above, as the DER reference; a sibling `diarization`
checkout is used only as **test data** for the model-gated DER gates below.)

## Clustering

Turn an `Extraction` into speaker-labelled RTTM spans with one of two backends.
Both are `diaric`'s engines — this crate selects and drives them, it does not
reimplement clustering. Pick a backend with `ClusterBackend` and tune it with
`OfflineOptions` / `OnlineOptions`.

### `ClusterBackend::Offline(OfflineOptions)` — the default, DER-gated

`diaric`'s pyannote-community-1 offline pipeline (`diaric::offline::diarize_offline`:
AHC initialization → VBx refinement over PLDA-projected embeddings). This is the
path every DER number in this README is measured on, and the one
`Extraction::diarize` runs by default. `OfflineOptions` mirrors, one-for-one, the
five community-1 hyperparameters `diaric` exposes:

| knob | default | tunes |
|---|---|---|
| `threshold` | 0.6 | AHC linkage cut |
| `fa` | 0.07 | VBx `Fa` |
| `fb` | 0.8 | VBx `Fb` |
| `max_iters` | 20 | VBx iteration cap |
| `min_duration_off` | 0.0 | gap-merge threshold (s) for span post-processing |

Every default equals `diaric`'s, which equals pyannote's — pinned in code
against `diaric`'s own accessors (`cluster::defaults_equal_diaric`), so a drift on
*either* side fails the build. `OfflineOptions::default()` therefore produces
byte-identical clustering to feeding `diaric` directly.

```rust,no_run
use speakerkit::extract::Options;
use speakerkit::source::{AnySource, ModelSource, Source};
use speakerkit::{ClusterBackend, OfflineOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = vec![0.0; 16_000]; // 16 kHz mono; no I/O in this crate.
    let options = Options::new().with_source(Source::FluidAudio);
    let extraction = AnySource::load("Models/speakerkit", options)?.extract(&audio)?;

    // The frozen community-1 PLDA projection `diaric` clusters through.
    let plda = diaric::plda::PldaTransform::new()?;

    // `diarize` runs the default backend (offline, `diaric`'s community-1 defaults).
    let output = extraction.diarize(&plda)?;
    println!("{} spans", output.spans_slice().len());

    // `diarize_with` runs a specific, tuned backend.
    let tuned = ClusterBackend::Offline(OfflineOptions::default().with_threshold(0.7));
    let _ = extraction.diarize_with(&plda, tuned)?;
    Ok(())
}
```

### `ClusterBackend::Online(OnlineOptions)` — streaming, NOT pyannote-parity

FluidAudio's greedy online centroid matcher (`SpeakerManager` semantics, ported
in `diaric` as `diaric::cluster::online::OnlineClusterer`) — a genuinely *different*
algorithm class: it assigns each segment as it arrives against running centroids,
which AHC→VBx structurally cannot do. Three properties are load-bearing and **by
design**:

- **Order-dependent.** Feeding the same segments in a different order can yield
  different speakers. The engine is defined here at one order (chunk order, then
  slot order within a chunk — FluidAudio's own feed order).
- **Raw cosine space, no PLDA.** It matches raw L2-normalized WeSpeaker
  embeddings by cosine distance; the PLDA projection the offline pipeline applies
  has no part in it. `diarize_online` therefore takes **no** `plda` argument —
  the absence is a fact of the signature, not an argument silently ignored.
- **Gated against FluidAudio's Swift, never pyannote DER.** Its correctness gate
  is a committed 48-step Swift `SpeakerManager` trace (48/48 exact,
  `tests/parity_online_swift.rs`), *not* DER against the pyannote reference. Do
  not read online output as pyannote-parity diarization.

`OnlineOptions` mirrors the three `SpeakerManager` knobs — `speaker_threshold`
(0.65, the cosine *distance* for assignment), `embedding_threshold` (0.45, for
centroid update), and `min_speech_duration` (1.0 s, to spawn a new speaker) —
the defaults of a bare `SpeakerManager()`.
`OnlineOptions::from_clustering_threshold(0.7)` reproduces the shipping FluidAudio
diarizer's derived `0.84` / `0.56`.

```rust,no_run
use speakerkit::extract::Options;
use speakerkit::source::{AnySource, ModelSource, Source};
use speakerkit::OnlineOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = vec![0.0; 16_000];
    let options = Options::new().with_source(Source::FluidAudio);
    let extraction = AnySource::load("Models/speakerkit", options)?.extract(&audio)?;

    // No PLDA: the online engine matches raw cosine embeddings.
    let output = extraction.diarize_online(OnlineOptions::default())?;
    println!("{} spans", output.spans_slice().len());
    Ok(())
}
```

### Honesty boundaries

- **Offline is the validated path.** Every DER number in this README, and every
  ⚠ warning below, is the offline backend. In this crate "parity" means offline.
- **Online is not parity.** It is order-dependent and gated only against
  FluidAudio's Swift `SpeakerManager`, never against pyannote. It is a streaming
  *capability*, not a faithful-diarization claim — do not DER-score it.
- **`min_active_ratio` is not a knob.** An earlier plan floated it as a
  speakerkit-side pre-filter (OFF = parity, ON = argmax); that was **dropped**,
  because the premise was false. `diaric`'s offline pipeline *already* applies
  argmax's `minActiveRatio = 0.2` sparse-slot exclude-and-reassign
  **unconditionally, for every source** (`diaric`'s `offline/algo.rs`; see also "The
  two model sources" below) — it withholds those slots from cluster *formation*
  and re-attaches them at nearest-centroid reassignment, never dropping them. So
  there is nothing to toggle: the filter is live for both sources, applied inside
  `diaric`, and a speakerkit knob could only add a novel stricter drop semantic no
  one asked for.

## The two model sources

`speakerkit` supports two independent, user-selectable CoreML conversions of
(functionally) the same pyannote pipeline, behind one `Options::source`
switch (`Source::FluidAudio` / `Source::Argmax`). They are **not**
interchangeable `.mlmodelc` swaps — argmax bakes preprocessing and decoding
into its graph, FluidAudio doesn't — so each is a genuinely different code
path inside this crate, not just a different weights file.

| | **FluidAudio** — `Source::FluidAudio` (default) | **argmax** — `Source::Argmax` (optional) |
|---|---|---|
| HF repo | `FluidInference/speaker-diarization-coreml` | `argmaxinc/speakerkit-coreml` |
| Segmenter output | powerset log-probs `[1,589,7]` (`log(softmax)`, not raw logits) | already decoded, in-graph: `speaker_ids`/`speaker_activity`/`overlapped_speaker_activity`/… |
| Decode semantics | **host-side**, in this crate — ports `diaric`'s exact powerset/mask/window decode | **in-graph** — argmax's own semantics; this crate only reads the result |
| Tier-1 fidelity ("did we read the model right") | relies on the tier-2 dia-ort check below; a dedicated FluidAudio Swift oracle is deferred/optional (no FluidAudio CLI) | argmax's own Swift (`argmax-oss-swift`'s `SpeakerSegmenterModel`/`SpeakerEmbedderModel`, via an out-of-tree harness since argmax's `DiarizeCLI` only emits post-clustering RTTM) |
| Tier-2 agreement ("does the decision match dia-ort") vs fp32 `dia`-ort | seg **99.97%** decision-level agreement (99.9717%, 3533/3534 frames); embed cosine **0.99999989** worst | seg **99.98%** cell agreement (Baseline); embed cosine **mean ~0.94, worst ~0.83** |
| Tier-3 parity (DER vs the reference, end to end through `diaric`'s clustering) | **validated** on 1-2 speakers (0.0000%); on the multi-speaker clips in the table below it stays *decision*-faithful (the speaker count matches the reference on every one) but is **not** frame-exact — see that table | **CHARACTERIZED, NOT VALIDATED** — 0.0000% standard-collar DER at 1-2 speakers, then 3.3-9.3% DER on three of the four multi-speaker clips |
| Measured status | **validated** (default; see the multi-speaker caveat below) | tensor-fidelity **validated** (72447/72447 segmentation cells EXACT, 123/123 embedding rows bit-identical vs argmax's own Swift); clustering parity **CHARACTERIZED, NOT VALIDATED** |

**The two sources can produce different diarization results on the same
audio — by design.** `Options::source` is a real tradeoff the caller picks,
not two paths that happen to agree: different decode semantics, different
embedding space. See "Status" below for exactly what "validated" vs
"characterized" means for each.

### ⚠ The argmax source diverges on multi-speaker audio

**Do not use `Source::Argmax` on multi-speaker audio.** This is measured, not
theoretical. End-to-end DER through `diaric`'s clustering (standard 0.25 s collar,
overlap excluded — the NIST/pyannote definition), scored against the pyannote
reference on `dia`'s parity corpus, `CpuOnly`:

| clip | ref speakers | FluidAudio | argmax |
|---|---|---|---|
| four clips | 1-2 | 0.0000%, count exact | 0.0000%, count exact |
| `06_long_recording` | 3 | 0.0908%, finds 3 | 0.0908%, finds 3 |
| `14_mrbeast_strongman_robot` | 4 | 0.3961%, finds 4 | **9.29%**, finds **5** |
| `10_mrbeast_clean_water` | 7 | 0.0000%, finds 7 | **3.33%**, finds **8** |
| `12_mrbeast_schools` | 15 | 0.1178%, finds 15 | **3.46%**, finds 15 |

`dia`-ort (the upstream oracle) reproduces the reference to 0.0000% standard
(collar-scored) DER on every one of these clips, and FluidAudio tracks it, so the
audio, the framing, the clustering, the reference and the harness are all held
constant, so the argmax **embedding front-end warp is the leading explanation**
for the divergence — it is *consistent with* an embedding-front-end cause, not
an experimentally isolated single variable, since the argmax source also swaps
in its own segmenter and in-graph decode alongside the embedder. Where it fails,
the error is ~100% *confusion*
with zero miss and zero false alarm — argmax hears exactly the same speech and
assigns it to the wrong person. That is the signature of a clustering divergence,
not of boundary jitter, which is why no collar absorbs it.

**Read the table for what it says, and not for more.** It does *not* say "argmax
breaks at ≥3 speakers": the 3-speaker clip is clean. It does *not* say the defect
is just a spurious speaker: on the 15-speaker clip argmax gets the count exactly
right and still misassigns 3.46% of speech. And the one clean multi-speaker clip
is also the only non-MrBeast one, so speaker count and recording domain are
**confounded** in this corpus. The failure is large, real and reproducible; its
precise trigger is not isolated. Assume argmax is unsafe for multi-speaker audio
until it is.

**Why.** `diaric`'s clustering is not intra-space geometry, and this is the trap:
its AHC cuts at a **fixed 0.6 linkage threshold** inside a **frozen, pretrained**
PLDA projection. `PldaTransform::new()` takes no data — it `include_bytes!`s an
LDA (256→128) + PLDA fit on the **native kaldi-fbank WeSpeaker distribution**
that `dia` and FluidAudio both feed it. argmax's embedder instead consumes an
80-mel spectrogram from its own `SpeakerEmbedderPreprocessor`, so its vectors
land in a differently-scaled space the frozen projection was never fit for. Where
every pairwise distance sits far from the threshold, even a miscalibrated
projection cuts in the right place — which is exactly why this looked benign for
so long. Where distances crowd the threshold, merges flip.

This means the **~0.94 embedding cosine is NOT benign at DER**, and any earlier
claim in this repository's history that it was is **retracted**. It was measured
only on 1- and 2-speaker clips, where DER = 0 is necessary but not sufficient.

### ⚠ And the FluidAudio default is not frame-exact there either

The same gate turned up a second thing, which the 1-2 speaker corpus had hidden:
`speakerkit`'s FluidAudio path is **0.0000% vs `dia`-ort on every ≤2-speaker clip
and on the 7-speaker clip, but 0.1191% and 0.3948% on `12` and `14`** — over the
0.1% DER-parity bound this crate's spec sets for the faithful source. The error
there is *confusion* (289 of 293 error units on `14`), not boundary jitter, so
the CoreML conversion's numerical drift really does flip a small number of
clustering assignments once several speakers must be separated.

In practice FluidAudio remains the right default by a wide margin — it never gets
the speaker *count* wrong, and it is ~23× more faithful than argmax on the same
clip — but "0.1% DER parity" is a claim that was only ever tested on 1-2 speaker
audio, and on multi-speaker audio it is false. It is recorded here rather than
smoothed over, and the bound in the test suite has *not* been raised to make it
pass.

Both limitations are pinned in code, not merely documented: `tests/parity_e2e.rs`
asserts every one of the numbers above on 3-, 4-, 7- and 15-speaker clips, so the
gate fires if behaviour moves in *either* direction — including if someone fixes
it, which must be a deliberate re-baseline rather than a silent pass.

A few more decisions worth knowing before you pick a source:

- **argmax's embedding space diverges from `dia`'s WeSpeaker space at
  cosine ~0.94 mean / ~0.83 worst.** Measured genuine, reproduced
  independently, not a harness bug (inputs are FNV-proven identical).
  Dominant cause is the **fbank front-end**, not quantization or masking:
  masks agree ~99.98%, both argmax quantization tiers sit at ~0.94, but
  argmax's WeSpeaker consumes an 80-mel spectrogram from its own separate
  `SpeakerEmbedderPreprocessor` rather than the kaldi fbank `dia`/FluidAudio
  use. It is tempting to argue that `diaric`'s clustering only cares about the
  *internal* geometry of whichever space it is given, so a self-consistent
  rotation would cluster fine. **That argument is wrong, and the DER gate
  above is what disproved it** — the projection `diaric` clusters through is
  frozen and pretrained, so it is not rotation-invariant, and a divergent
  front-end is a domain mismatch against it.
- **Both sources share `diaric`'s mask/overlap-exclusion policy AND its
  clustering-stage `minActiveRatio` filter** — not each vendor's own, and not a
  filter this crate omits. argmax's `minActiveRatio` (which withholds
  sparse/overlap-heavy slots, ~12-17% on real clips, from cluster *formation*
  while still labeling them by nearest centroid) is the SAME pyannote
  community-1 `filter_embeddings` that `diaric` ports and runs **unconditionally on
  every `Extraction`, from either source** (`MIN_ACTIVE_RATIO = 0.2`, diaric's
  `offline/algo.rs`): the sparse slots are withheld from formation and then
  re-attached at nearest-centroid reassignment — exactly argmax's own
  withhold-then-reassign, reproduced downstream. So it is **not** an un-ported
  divergence, and the old worry that reproducing it would mean *dropping* a slot
  is moot: `diaric` re-attaches, it never drops. This also settles what was once an
  open watch item — the filter is **already live for argmax**, and it does
  **not** rescue the multi-speaker divergence: every failing clip above ran with
  it applied, `FluidAudio` through the same filter is clean, and the spurious
  argmax clusters form among the slots that *pass* it. That is what pins the
  cause on the 80-mel front end (above), not on any missing filter — the only
  rescue path is a kaldi-fbank front-end swap.
- **argmax ships two quantization tiers** (`ArgmaxVariant::Baseline`,
  un-palettized, the default; `ArgmaxVariant::W8A16`, 8-bit palettized).
  `AnySource::load` always loads `Baseline` — build `ArgmaxSource` via
  `ArgmaxSource::from_dir_with` directly for `W8A16`. Measured quantization
  cost is near-free either way (segmentation cell agreement −0.20pp,
  embedding mean cosine −0.0011 relative to Baseline, both vs fp32
  `dia`-ort).

## Licensing

Two independently-licensed vendors, one crate. Read this before enabling
`Source::Argmax` in anything you ship — this section is a factual summary of
what each repository declares, not legal advice.

### FluidAudio — `FluidInference/speaker-diarization-coreml` (the default)

- FluidAudio's own SDK/conversion tooling is **Apache-2.0**.
- The model repo's own README states plainly: *"the SDK itself is Apache
  2.0, but the parent model from Pyannote is `cc-by-4.0`"* — and its
  HuggingFace frontmatter tags the repo `license: cc-by-4.0`, tracing to
  `pyannote/speaker-diarization-community-1`. CC-BY-4.0 is permissive but
  **requires attribution** — see below.
- Net effect: clean to use, attribution owed on the weights. This is why
  it's the crate's default source.

### argmax — `argmaxinc/speakerkit-coreml` (optional, experimental — CAVEAT)

- The HF repo declares **no `license:` key at all** in its frontmatter
  (verified directly against the downloaded repo's `README.md`, and
  independently by `tests/argmax_model_io.rs`'s introspection). Under both
  HuggingFace and GitHub norms, undeclared defaults to **all rights
  reserved** on whatever is original to the converted/compiled artifact.
- Each model directory ships its own `README.txt`, but it points at the
  *original upstream weights'* licenses, not a license for argmax's own
  CoreML conversion:
  [pyannote/segmentation-3.0's LICENSE](https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE)
  for the segmenter,
  [wespeaker's model license](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md#model-license)
  for the embedder.
- argmax's **code** — the Swift `SpeakerKit` implementation this crate ports
  decode semantics from (`argmaxinc/argmax-oss-swift`) — is **MIT**
  (verified directly against that repo's `LICENSE` file). The underlying
  **nets** argmax converted are pyannote-3.0-derived, i.e. MIT lineage per
  the link above.
- **What this means for you:** the argmax source ships as an optional,
  experimental path specifically so the pipeline is available for
  evaluation/research. Flipping `Options::source` to `Source::Argmax` in
  something you distribute means *you* assume the licensing risk on the
  converted CoreML artifacts, or clarify redistribution terms with argmax
  directly. This crate ports the code and documents the situation here; it
  does not resolve the license gap on your behalf.

### pyannote attribution (CC-BY-4.0 — applies regardless of source)

`diaric`'s clustering — the destination for every tensor this crate produces,
from either source — runs pyannote's **community-1 PLDA**, which is
**CC-BY-4.0** and requires attribution. Separately, pyannote's
**segmentation-3.0** net (what both sources' segmenters derive from) is
**MIT**. If you ship a product built on this pipeline, include the
following attribution (reproduced from FluidAudio's model card, itself
citing the original papers):

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Wang2023,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={ICASSP 2023, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@article{Landini2022,
  author={Landini, Federico and Profant, J{\'a}n and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  title={{Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks}},
  year={2022},
  journal={Computer Speech \& Language},
}
```

(The third citation is VBx/clustering, which is `diaric`'s domain, not this
crate's — reproduced here because it ships in the same model-card citation
block and applies to the same end-to-end pipeline.)

## Compute units

Both sources default every model to `ComputeUnits::All` — the Neural Engine
gets first pick whenever CoreML judges it can run the graph, which is the
entire point of this crate (ANE throughput is the product driver above).

This has one sharp, *measured* edge: CoreML's `All` vs `CpuOnly` scheduling
for the identical model and identical input is not a bitwise no-op. On the
argmax segmenter+embedder: **0.0911%** of hard `speaker_ids` cells flip, and
**every** embedding row's cosine to the `CpuOnly` reference moves (worst
**0.9241**) — roughly the same order of change as an entire int8-vs-fp32
quantization tier. The set of *which* `(chunk, slot)` pairs get consumed
does not change (`slot_diffs == 0`, asserted in code, not just observed).

Every *tensor-level* fidelity gate in this crate's test suite pins
`ComputeUnits::CpuOnly` for determinism (partly forced: argmax's fbank
preprocessor hardcodes `.cpuOnly` regardless of what this crate requests). Those
gates prove the `CpuOnly` execution path is correct — they say nothing on their
own about the `All` configuration this crate actually ships by default, because
**a gate pinned to one compute unit only proves that compute unit.**

The end-to-end DER gate therefore runs on **`All`, the shipping default**, and
asserts (not merely reports) that the placement changes no diarization
*decision*. For both sources it asserts this **directly** — the two placements
are scored against *each other* and not one standard-collar speaker-frame is
labelled differently (`der_std(CpuOnly, All).err_units() == 0`) — and the
speaker count is identical. That is strictly stronger than an equal
distance-to-reference: `All` and `CpuOnly` being *equally far* from the
reference would pass even if each erred on different frames, so the gate
compares the placements to each other, not only to the reference. The placement
is genuinely exercised rather than silently falling back to CPU — the strict,
no-collar difference between the two placements is *non-zero* (0.12-0.29%),
which an identical execution could not produce, while their collar-scored
disagreement is 0.0000%. So the ANE's numerical drift is real but lands entirely
inside the scoring collar: it moves span edges, not speakers.

That pairwise-zero is reachable only because both placements score 0.0000%
standard-collar DER against the reference on these 1-2 speaker clips (mutual
exactness implies pairwise agreement); a nonzero-DER clip could expose a real
decision change, which the gate would pin rather than absorb.

Scope that claim honestly: it is measured on the 1-2 speaker clips. The
≥3-speaker gate runs `CpuOnly` (matched to `dia`-ort's CPU ONNX, which is what
isolates the embedding axis from the placement axis), so `All` on multi-speaker
audio is inferred rather than measured — see "Status".

Practical takeaway: if you need bit-for-bit reproducibility across
machines/runs, pin `ComputeUnits::CpuOnly` explicitly via `ComputeOptions`
(FluidAudio) / `ArgmaxComputeOptions` (argmax), and record which placement
you used alongside anything you persist — outputs are not
placement-invariant.

## Test models

Both model stores are gitignored — nothing under `Models/` is committed.
Model-gated tests are `#[ignore]`d by default; run them explicitly once the
models are present.

| Source | Env override | Default path | Fetch |
|---|---|---|---|
| FluidAudio | `SPEAKERKIT_TEST_MODELS` | `<workspace>/Models/speakerkit` | `hf download FluidInference/speaker-diarization-coreml --local-dir Models/speakerkit` |
| argmax | `ARGMAX_TEST_MODELS` | `<workspace>/Models/argmax-speakerkit` | `hf download argmaxinc/speakerkit-coreml --local-dir Models/argmax-speakerkit` |

Then, from the workspace root:

```sh
SPEAKERKIT_TEST_MODELS=Models/speakerkit \
ARGMAX_TEST_MODELS=Models/argmax-speakerkit \
cargo test -p speakerkit -- --ignored
```

That runs the model-gated segmentation, embedding and argmax suites. It does
**not** run the end-to-end **DER** gates: `tests/parity_e2e.rs` and
`tests/parity_shipping_der.rs` are additionally `#![cfg(feature = "dia-oracle")]`
(they need dia's own ort path as the parity oracle), so without
`--features dia-oracle` they compile to *nothing* and the sweep above is a
green run containing **zero** DER tests. Run them explicitly, in **two phases** —
because `-- --ignored` runs the *ignored* tests **only**, and each DER binary
also carries an *ordinary* (non-ignored) suite: the DER-math unit tests and the
mutation-proof pin guards (e.g. `assert_pinned_fires_...`,
`clip09_known_defect_pins_every_field`) that keep the DER numbers honest. A
single `-- --ignored` invocation silently SKIPS them, so a dropped `assert_pinned`
clause would stay green.

Phase 1 — the ordinary (hermetic) suite. No models, no fixtures; run it WITHOUT
`--ignored`:

```sh
cargo test -p speakerkit --features dia-oracle --test parity_e2e
cargo test -p speakerkit --features dia-oracle --test parity_shipping_der
```

Phase 2 — the model-gated end-to-end DER gates (they also need the sibling
`diarization` ONNX/fixtures + `ort`, so set `ORT_DYLIB_PATH` if `ort` cannot
self-provision `libonnxruntime`):

```sh
SPEAKERKIT_TEST_MODELS=Models/speakerkit \
cargo test -p speakerkit --features dia-oracle --test parity_e2e -- --ignored
SPEAKERKIT_TEST_MODELS=Models/speakerkit \
cargo test -p speakerkit --features dia-oracle --test parity_shipping_der -- --ignored
```

To confirm those DER gates are actually **compiled** — not silently feature-gated
out, the exact trap above — and that the Phase 1 hermetic suite passes, run the
inventory check. It lists each DER binary's tests, hard-fails if the list is empty
or an expected gate is missing, and then EXECUTES each binary's ordinary suite —
so a feature-selection no-op is distinguishable from a real pass, and a weakened
pin guard is caught even if you skipped Phase 1:

```sh
crates/speakerkit/tests/der_gate_inventory.sh
```

The argmax Swift fidelity gate (`tests/parity_argmax_swift.rs`) compares
against **committed goldens** (`tests/fixtures/golden_argmax_swift/`), so
running it needs only `ARGMAX_TEST_MODELS` like everything else above.
*Regenerating* those goldens needs a Swift toolchain and a checkout of
[`argmaxinc/argmax-oss-swift`](https://github.com/argmaxinc/argmax-oss-swift) —
see `tests/swift/regen_goldens.sh`.

## Status

- **FluidAudio source: validated as the default; not frame-exact on
  multi-speaker audio.** Segmentation decision-level agreement and embedding
  cosine measured against fp32 `dia`-ort (the table above), and now DER through
  `diaric`'s clustering: **0.0000%** on every ≤2-speaker clip and on the 7-speaker
  clip; **0.09-0.39%** on the other multi-speaker clips, exceeding the spec's
  0.1% parity bound on two of them (see the second warning above). Its speaker
  *count* decision matches the reference on every clip in that table. (The one
  place the CoreML path does get the count wrong is 8-speaker audio: the shipping
  int8 embedder undercounts, 5-6 of 8, and the fp32 control cannot cluster at all
  — a separate known defect, pinned in `tests/parity_shipping_der.rs`.) A
  dedicated tier-1 fidelity oracle (FluidAudio's own Swift) is deferred/optional —
  FluidAudio ships a library, not a CLI, so the tier-2 `dia`-ort agreement check
  is what stands in for it.
- **argmax source: tensor-fidelity validated, clustering parity CHARACTERIZED,
  NOT VALIDATED.** Bit-exact/near-exact against argmax's own Swift decode
  (tier 1) — the strongest gate this crate has for any source. But end to end it
  **fails on multi-speaker audio** (see the first warning above): 0.0000%
  standard-collar DER at 1-2 speakers, then 3.3-9.3% DER on three of the four
  multi-speaker clips. It
  remains available and user-selectable, with the limitation pinned by test and
  documented here. Treat it as experimental, and do not point it at
  multi-speaker audio.
- **Known gap, stated rather than buried:** the multi-speaker DER numbers are
  measured on `CpuOnly` — the placement matched to `dia`-ort's CPU ONNX runtime,
  which is what isolates the *conversion* and *embedding* axes from the
  *placement* axis. The shipping `All` placement is DER-verified (ΔDER = 0.0000%,
  speaker count invariant) only on the 1-2 speaker clips. So "`All` changes no
  decision" is established on easy audio and *inferred*, not measured, on
  multi-speaker audio. That is precisely the shape of assumption that produced
  both failures above, so it is written down as an open item rather than treated
  as settled.
- **What "DER" means here, precisely.** The reference RTTMs this crate scores
  against are **pyannote.audio 4.0.4's own output**, captured and committed by
  `dia` — *not* human annotation. (Their segment durations are multiples of
  pyannote's 16.875 ms frame step; no human placed those boundaries.) So every
  DER number above is a **distance to pyannote 4.0.4**, never a distance to the
  truth: a source scoring 0.0000% has reproduced the upstream reference
  implementation to 0.0000% standard-collar DER, which is what this crate
  promises — it has **not** been
  shown to be *correct*. Human-labelled benchmark RTTM (AMI, DIHARD) is not part
  of this repository, so "are we right?" is a question none of these gates
  answer. They answer "do we match the reference implementation?".
- **`0.1.0` (unreleased), `publish = false`** until the always-on `diaric`
  (runtime) git dependency — and the optional `dia` (`diarization`) oracle —
  has a registry version: an exact-rev git dependency, like a path dependency,
  carries no version Cargo can publish against.

## See also

- Design spec: `docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`
  (the source of truth for the sources/DER material above) and its predecessor,
  `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md` — both
  referenced the same way from this crate's own rustdoc (e.g. `lib.rs`). The
  clustering surface documented under "Clustering" above has its own design of
  record, `docs/superpowers/specs/2026-07-16-clustering-backends-design.md`
  (whose 2026-07-16 amendment the `cluster` module and `lib.rs` cite).
  `docs/` is gitignored (local planning artifacts from this feature's
  spec/plan workflow, not shipped documentation), so these paths exist only
  where the feature was planned/built, not in every checkout — not
  browsable repo links.
- [`diaric`](https://github.com/findit-studio/diaric) — the core clustering
  crate this one feeds at runtime.
- [`dia`/`diarization`](https://github.com/Findit-AI/diarization) — its
  ONNX/`ort` superset, the DER oracle behind the `dia-oracle` feature.
- [`coremlit`](../coremlit) — the safe CoreML runtime layer this crate is
  built on.
- Workspace root [`README.md`](../../README.md) — `whisperkit`, MSRV, and
  the repo-wide dual MIT/Apache-2.0 license for this crate's *own* Rust
  source code (distinct from the model-weight licenses discussed above).
