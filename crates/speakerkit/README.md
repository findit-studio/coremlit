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
| Tier-2 accuracy ("is the decision right") vs fp32 `dia`-ort | seg **99.97%** decision-level agreement (99.9717%, 3533/3534 frames); embed cosine **0.99999989** worst | seg **99.98%** cell agreement (Baseline); embed cosine **mean ~0.94, worst ~0.83** |
| Measured status | **validated** | tensor-fidelity **validated** (72447/72447 segmentation cells EXACT, 123/123 embedding rows bit-identical vs argmax's own Swift); accuracy **characterized**, not yet DER-validated |

**The two sources can produce different diarization results on the same
audio — by design.** `Options::source` is a real tradeoff the caller picks,
not two paths that happen to agree: different decode semantics, different
embedding space. See "Status" below for exactly what "validated" vs
"characterized" means for each.

A few decisions worth knowing before you pick a source:

- **argmax's embedding space diverges from `dia`'s WeSpeaker space at
  cosine ~0.94 mean / ~0.83 worst.** Measured genuine, reproduced
  independently, not a harness bug (inputs are FNV-proven identical).
  Dominant cause is the **fbank front-end**, not quantization or masking:
  masks agree ~99.98%, both argmax quantization tiers sit at ~0.94, but
  argmax's WeSpeaker consumes an 80-mel spectrogram from its own separate
  `SpeakerEmbedderPreprocessor` rather than the kaldi fbank `dia`/FluidAudio
  use. `dia`'s clustering operates on the *internal* geometry of whichever
  embedding space it's given, not on cross-space cosine to `dia`'s own
  space, so this does not directly predict clustering quality — the
  end-to-end DER gate is what adjudicates whether it's good enough. Until
  that gate runs, the argmax source is **characterized, not
  production-validated**.
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

Every fidelity/parity gate in this crate's test suite pins `ComputeUnits::CpuOnly`
for determinism (partly forced: argmax's fbank preprocessor hardcodes
`.cpuOnly` regardless of what this crate requests). That means those gates
prove the `CpuOnly` execution path is correct — they say nothing about the
`All` configuration this crate actually ships by default. End-to-end
accuracy (DER) validation, when it lands, has to run against the shipping
default and compare to ground truth directly, not against a `CpuOnly`
reference: **a gate pinned to one compute unit only proves that compute
unit.**

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

- **FluidAudio source: validated.** Segmentation decision-level agreement
  and embedding cosine both measured against fp32 `dia`-ort (the table
  above). A dedicated tier-1 fidelity oracle (FluidAudio's own Swift) is
  deferred/optional — FluidAudio ships a library, not a CLI, so the tier-2
  `dia`-ort accuracy check is what stands in for it.
- **argmax source: tensor-fidelity validated, clustering accuracy
  characterized.** Bit-exact/near-exact against argmax's own Swift decode
  (tier 1, T4) — the strongest gate this crate has for any source. Its
  accuracy relative to `dia`'s embedding space (tier 2, T5) is measured and
  understood (fbank-dominant divergence, ~0.94 mean cosine), but whether
  that's *sufficient* for clustering is an open question only an end-to-end
  DER gate can answer. Treat this source as experimental until that lands.
- **Clustering accuracy (DER), end to end, for either source: not yet
  measured.** Out of this crate's scope by design (clustering is `dia`'s) —
  this is the gate referenced throughout this README as the thing that
  adjudicates the argmax embedding-space question and the compute-unit
  question above.
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
