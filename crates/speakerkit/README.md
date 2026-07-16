# speakerkit

Native CoreML **inference backend** for speaker diarization: it runs
pyannote's segmentation net and the WeSpeaker embedding net on the Apple
Neural Engine (via [`coremlit`](../coremlit)) and produces the tensors that
feed [`dia`](https://github.com/Findit-AI/diarization)'s Rust VBx/PLDA
clustering. Product driver:
[`findit-studio/desktop#120`](https://github.com/findit-studio/desktop/issues/120)
(segmentation ~20x / embedding ~30x ANE uplift targets).

**This is not a standalone diarizer.** Clustering — the step that turns
embeddings into speaker labels — stays in `dia`, unchanged, by design. This
crate's job stops the moment it has produced `dia`-shaped tensors
(`Extraction`); it never assigns a speaker label to anything.

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

    // `extraction` now holds exactly the tensors `dia`'s offline diarizer
    // consumes. Behind the `dia` feature:
    //   extraction.into_offline_input(&plda)  ->  dia::offline::OfflineInput
    // which feeds `dia::offline::diarize_offline` for the actual clustering.
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

Feature flags: `dia` (optional path dependency on the `diarization` sibling
crate; enables `Extraction::into_offline_input`) and `serde`
(`Serialize`/`Deserialize` on `Options` and friends). Neither is on by
default.

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
| Segmenter output | raw powerset logits `[1,589,7]` | already decoded, in-graph: `speaker_ids`/`speaker_activity`/`overlapped_speaker_activity`/… |
| Decode semantics | **host-side**, in this crate — ports `dia`'s exact powerset/mask/window decode | **in-graph** — argmax's own semantics; this crate only reads the result |
| Tier-1 fidelity ("did we read the model right") | relies on the tier-2 dia-ort check below; a dedicated FluidAudio Swift oracle is deferred/optional (no FluidAudio CLI) | argmax's own Swift (`argmax-oss-swift`'s `SpeakerSegmenterModel`/`SpeakerEmbedderModel`, via an out-of-tree harness since argmax's `DiarizeCLI` only emits post-clustering RTTM) |
| Tier-2 agreement ("does the decision match dia-ort") vs fp32 `dia`-ort | seg **99.97%** decision-level agreement (99.9717%, 3533/3534 frames); embed cosine **0.99999989** worst | seg **99.98%** cell agreement (Baseline); embed cosine **mean ~0.94, worst ~0.83** |
| Tier-3 parity (DER vs the reference, end to end through `dia`'s clustering) | **validated** on 1-2 speakers (0.0000%); on the multi-speaker clips in the table below it stays *decision*-faithful (the speaker count matches the reference on every one) but is **not** frame-exact — see that table | **CHARACTERIZED, NOT VALIDATED** — exact at 1-2 speakers, then 3.3-9.3% DER on three of the four multi-speaker clips |
| Measured status | **validated** (default; see the multi-speaker caveat below) | tensor-fidelity **validated** (72447/72447 segmentation cells EXACT, 123/123 embedding rows bit-identical vs argmax's own Swift); clustering parity **CHARACTERIZED, NOT VALIDATED** |

**The two sources can produce different diarization results on the same
audio — by design.** `Options::source` is a real tradeoff the caller picks,
not two paths that happen to agree: different decode semantics, different
embedding space. See "Status" below for exactly what "validated" vs
"characterized" means for each.

### ⚠ The argmax source diverges on multi-speaker audio

**Do not use `Source::Argmax` on multi-speaker audio.** This is measured, not
theoretical. End-to-end DER through `dia`'s clustering (standard 0.25 s collar,
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
constant: **argmax's embedding is the only variable.** Where it fails, the error is ~100% *confusion*
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

**Why.** `dia`'s clustering is not intra-space geometry, and this is the trap:
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
  use. It is tempting to argue that `dia`'s clustering only cares about the
  *internal* geometry of whichever space it is given, so a self-consistent
  rotation would cluster fine. **That argument is wrong, and the DER gate
  above is what disproved it** — the projection `dia` clusters through is
  frozen and pretrained, so it is not rotation-invariant, and a divergent
  front-end is a domain mismatch against it.
- **Both sources share `dia`'s mask/overlap-exclusion policy**, not each
  vendor's own. In particular, argmax's own `minActiveRatio` filter — which
  withholds sparse/overlap-heavy slots (~12-17% of slots on real clips) from
  cluster *formation* while still labeling them — is deliberately **not**
  ported: clustering is `dia`'s, and `Extraction` has no "present but
  excluded from clustering" channel, so reproducing that filter would mean
  *dropping* the slot and losing its speech attribution entirely, which is
  usually worse than clustering on a slightly sparse embedding. If DER
  measurements later show this mattering, revisiting is an open watch item.
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

`dia`'s clustering — the destination for every tensor this crate produces,
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

(The third citation is VBx/clustering, which is `dia`'s domain, not this
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
*decision*: for both sources, ΔDER(`All` − `CpuOnly`) against the reference is
0.0000% and the speaker count is identical. The placement is genuinely exercised
rather than silently falling back to CPU — the strict, no-collar frame-exact
difference between the two placements is *non-zero* (0.12-0.29%), which an
identical execution could not produce, while the collar-scored DER against the
reference is unchanged. So the ANE's numerical drift is real but lands entirely
inside the scoring collar: it moves span edges, not speakers.

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
  `dia`'s clustering: **0.0000%** on every ≤2-speaker clip and on the 7-speaker
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
  **fails on multi-speaker audio** (see the first warning above): exact at 1-2
  speakers, then 3.3-9.3% DER on three of the four multi-speaker clips. It
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
  implementation exactly, which is what this crate promises — it has **not** been
  shown to be *correct*. Human-labelled benchmark RTTM (AMI, DIHARD) is not part
  of this repository, so "are we right?" is a question none of these gates
  answer. They answer "do we match the reference implementation?".
- **`0.1.0` (unreleased), `publish = false`** until the `dia`/`diarization`
  path dependency has a registry version.

## See also

- Design spec: `docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`
  (the source of truth for everything above) and its predecessor,
  `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md` — both
  referenced the same way from this crate's own rustdoc (e.g. `lib.rs`).
  `docs/` is gitignored (local planning artifacts from this feature's
  spec/plan workflow, not shipped documentation), so these paths exist only
  where the feature was planned/built, not in every checkout — not
  browsable repo links.
- [`dia`/`diarization`](https://github.com/Findit-AI/diarization) — the
  clustering crate this one feeds.
- [`coremlit`](../coremlit) — the safe CoreML runtime layer this crate is
  built on.
- Workspace root [`README.md`](../../README.md) — `whisperkit`, MSRV, and
  the repo-wide dual MIT/Apache-2.0 license for this crate's *own* Rust
  source code (distinct from the model-weight licenses discussed above).
