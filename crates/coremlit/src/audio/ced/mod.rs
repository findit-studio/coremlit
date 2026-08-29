//! Native CoreML **CED** (tiny/mini/small/base) AudioSet sound-event tagging —
//! coremlit's first multi-label classifier: 16 kHz mono waveform in, ranked
//! AudioSet predictions out (527 rated classes: name + `/m/…` id + class index +
//! sigmoid confidence), long clips via windowed chunking + Mean/Max aggregation.
//!
//! CED (Consistent Ensemble Distillation, arXiv 2308.11957; upstream
//! RicherMans/CED, `mispeech/ced-{tiny,mini,small,base}`) is a distilled
//! AudioSet transformer. The four sizes are contract-identical here — one
//! size-invariant mel→logits I/O; they differ only in internal transformer
//! width (see [`CedModel`]). The mel front-end runs in Rust (the private `mel`
//! submodule) and the mel→logits transformer runs natively on Apple silicon as
//! one fp16 `.mlmodelc` — an in-graph STFT/mel is the exact fragility class
//! behind the ORT CoreML EP zeroed-logits bug this feature closes. NO `ort`
//! anywhere.
//!
//! Design spec: `docs/superpowers/specs/2026-07-23-ced-native-ane-design.md`.
//!
//! # Model artifacts
//!
//! No model is bundled (a `.mlmodelc` is a directory artifact). Each size's
//! fp16 CED graph is converted owner-side (Wave B), distributed via Hugging
//! Face, and staged as a gitignored dev-time download under
//! `Models/ced/ced-<size>/` (env override `CED_TEST_MODELS` points at the
//! `Models/ced` family root); per-file SHA-256 and I/O contract are pinned per
//! size by `tests/ced/model_io.rs` once staged. [`CedModel`] owns the repo ids
//! and the `<dir>/<bundle>` path spelling ([`CedModel::mlmodelc_path`]). The
//! four graphs are I/O-identical, so model identity is caller-supplied:
//! coremlit cannot — and does not — detect which size a `.mlmodelc` is.
//!
//! # Rust front-end around an fp16 CoreML graph
//!
//! The graph takes the believed `[1, 64, 1001]` log-mel (`mel`, f32) computed
//! by this module's Rust front-end and emits `[1, 527]` **pre-sigmoid** logits
//! (`logits`, f32); sigmoid, ranking, and long-clip aggregation run in Rust.
//! The believed mel numerics are probe-pinned in Wave B (see the `mel`
//! submodule docs). This `[1, 64, 1001]` shape is shared by all four sizes.
//!
//! Upstream's `target_length = 1012` is NOT this input width: it is the
//! transformer's time positional-embedding capacity and its long-form mel
//! chunk size. A canonical 10 s window is 160 000 samples → 1001 mel frames
//! (hop 160, `center=True`), consumed unpadded with the pos embed sliced to 62
//! of its 63 patch columns; padding to 1012 would compute a different
//! function. So `1001 <= 1012` is on-distribution, not a truncation (verified
//! against RicherMans/CED `audiotransformer.py` and the mispeech feature
//! extractor); the `mel` submodule carries the full derivation.
//!
//! # From window scores to events
//!
//! [`Classifier::classify_windows`] is the seam between this crate and event
//! detection. It returns `Vec<`[`WindowConfidences`]`>`, and
//! [`WindowConfidences`] *is* [`windit::windowed::Windowed`]`<`[`Confidences`]`>`
//! — just as [`Span`] *is* [`windit::plan::Span`]. coremlit's long-clip output
//! is already `windit`'s own value type, so the post-processing stack composes
//! with no adapter and no repacking:
//!
//! ```text
//! Classifier::classify_windows -> Vec<Windowed<Confidences>>  coremlit (this module)
//!   index one class            -> Vec<Windowed<f32>>          one slice read per window
//!   windit::smooth             -> Vec<Windowed<f32>>          Ema / CadenceEma
//!   zuoer::RunSegmenter        -> Run { span, mean, peak }    hysteresis + durations
//! ```
//!
//! ## coremlit ships no CED convenience layer, deliberately
//!
//! `audio::vad` offers a one-call `detect_speech` because Silero has
//! *upstream-authored* defaults — 0.5 threshold, 250 ms minimum speech, 100 ms
//! minimum silence — that `zuoer`'s hysteresis derives from, so "the default"
//! there is a real, attributable thing. CED's 527 classes have no equivalent. A
//! threshold and minimum duration right for `Glass` (class 441, a sub-second
//! transient) are wrong for `Music` (class 137, continuous for minutes); a hop
//! fine enough to localize a single bark wastes an order of magnitude of
//! inference on ambient scene tagging. Any defaults coremlit shipped would be
//! silently wrong in some scenario, and silently wrong is the failure mode a
//! classifier can least afford.
//!
//! So the glue below stays in your application. It is about a dozen lines, and
//! **every parameter in it is scenario-dependent** — which is precisely why it
//! is not a coremlit function. This module chooses no threshold, no smoothing
//! constant, no minimum event duration, and no set of classes to watch.
//!
//! ## Per-class orchestration is the consumer's job
//!
//! CED emits 527 *independent* sigmoids per window: not a softmax, and they do
//! not sum to one. An event detector picks the handful of classes it cares
//! about and runs one independent smoother + segmenter per class, each with its
//! own parameters. Running all 527 is possible but rarely wanted — a real
//! recording has a sparse active set at any moment, and 527 segmenters is 527
//! parameter sets nobody has tuned.
//!
//! ## Timestamps are window-resolution, not event-resolution
//!
//! Each score summarizes a whole [`WINDOW_SAMPLES`] (10 s) window, and a
//! segmenter treats it as one point sample placed at that window's start. Run
//! boundaries are therefore quantized to [`WindowPlan::hop_samples`], and the
//! audio a run actually observed is its reported interval extended forward by
//! one whole window — so an event reported at `3 s..8 s` happened somewhere in
//! `3 s..18 s`. A shorter hop buys finer quantization, never a narrower smear.
//!
//! ## Dependencies
//!
//! `windit` and `zuoer` are coremlit's own dependencies. Only [`Span`] and
//! [`WindowConfidences`] cross into *this* module's API, and this module
//! re-exports no smoothing tier, so depend on `windit` directly. (Under the
//! `clap` feature, `embeddings::clap::smooth` does re-export windit's smoothing
//! seam — but only the parts a 512-wide *embedding* can use: the scalar `Ema`
//! and `CadenceEma` this table names are not among them.)
//!
//! ```toml
//! windit = "0.3"   # smoothing; already in your graph via `ced`
//! ```
//!
//! `zuoer` is the other case. With the `vad` feature on, `audio::vad`
//! re-exports the whole set needed to drive a segmenter — `Run`,
//! `RunSegmenter`, `RunOptions`, `SampleRate` and its `Error` / `Result` — so
//! the segmenting block below names them through coremlit and needs no direct
//! dependency. Under `ced` alone, `zuoer` is not in your graph at all, and you
//! add it yourself:
//!
//! ```toml
//! zuoer = "0.2"    # only for `ced` WITHOUT `vad`
//! ```
//!
//! `windit` also ships its own gate/segment tier
//! (`windit::segment::{Hysteresis, Segmenter, SegmentOptions}`, composed by
//! `windit::decode`) which needs no extra dependency at all. It returns element
//! `Range`s and *no* probability aggregates, so prefer it when a plain interval
//! is enough, and `zuoer::RunSegmenter` when the event needs a confidence
//! attached — which is what the rest of this section shows.
//!
//! ## Scoring the clip
//!
//! Loading a model and running it is the ONE step that needs a staged
//! `.mlmodelc`, so this block — and only this block — is `no_run`:
//! `cargo test --doc` **compiles it and never executes it**. Nothing in it is
//! verified behavior. Everything downstream of it is, because everything
//! downstream of it is arithmetic on the returned numbers:
//!
//! ```no_run
//! use coremlit::audio::ced::{CedModel, Classifier, WindowConfidences, WindowPlan};
//!
//! # let samples_16k: Vec<f32> = Vec::new();
//! let classifier = Classifier::from_file(CedModel::Small.mlmodelc_path("Models/ced"))?;
//! // A 1 s hop across the fixed 10 s window: 90% overlap, one score per second.
//! let plan = WindowPlan::new().with_hop_samples(16_000);
//! let windows: Vec<WindowConfidences> = classifier.classify_windows(&samples_16k, &plan)?;
//! # Ok::<(), coremlit::audio::ced::Error>(())
//! ```
//!
//! ## Projecting one class, and smoothing it
//!
//! [`Confidences::try_from_slice`] builds that `windows` vector by hand, which
//! is what lets the rest of the pipeline **run** here with no model staged —
//! and what lets a consumer unit-test their own event logic the same way:
//!
//! ```
//! use coremlit::audio::ced::{
//!   Confidences, Error, NUM_CLASSES, RatedSoundEvent, Span, WINDOW_SAMPLES, WindowConfidences,
//! };
//! use windit::{
//!   smooth::{Ema, SmoothPolicy},
//!   windowed::Windowed,
//! };
//!
//! let hop = 16_000; // the `WindowPlan` hop the scores were produced at
//! let dog = RatedSoundEvent::from_key("Dog")[0].index();
//! let music = RatedSoundEvent::from_key("Music")[0].index();
//! assert_eq!((dog, music), (74, 137));
//!
//! // Twelve windows of 527 scores, standing in for `classify_windows` output.
//! // `Music` outscores `Dog` in every one of them: the 527 sigmoids are
//! // independent, so two classes can both be loud and they never sum to one.
//! let barks = [0.02, 0.04, 0.71, 0.86, 0.31, 0.90, 0.88, 0.09, 0.03, 0.01, 0.01, 0.02];
//! let windows = barks
//!   .iter()
//!   .enumerate()
//!   .map(|(i, &p)| {
//!     let mut scores = vec![0.0; NUM_CLASSES];
//!     scores[dog] = p;
//!     scores[music] = 0.93;
//!     Ok(WindowConfidences::new(
//!       Confidences::try_from_slice(&scores)?,
//!       Span::new(i * hop, WINDOW_SAMPLES, WINDOW_SAMPLES),
//!     ))
//!   })
//!   .collect::<Result<Vec<WindowConfidences>, Error>>()?;
//!
//! // Stored exactly as handed over: `try_from_slice` takes confidences, not
//! // logits, so it applies no sigmoid (which would read 0.7027 here) and no
//! // renormalization (0.4804 here, and a sum that could never pass one).
//! assert_eq!(windows[3].value().as_slice()[dog], 0.86);
//! assert!(windows[3].value().as_slice().iter().sum::<f32>() > 1.0);
//!
//! // One column out of 527. The span rides along untouched.
//! let track: Vec<Windowed<f32>> = windows
//!   .iter()
//!   .map(|w| Windowed::new(w.value().as_slice()[dog], w.span()))
//!   .collect();
//!
//! // The asked-for class, never the loudest one — projecting an argmax would
//! // have followed `Music` and returned a flat 0.93 track.
//! assert_eq!(*track[3].value(), 0.86);
//! assert_eq!(track[3].span(), windows[3].span());
//!
//! // `Ema`'s alpha is per push; `CadenceEma` denominates its time constant in
//! // input samples instead, so one setting survives an irregular hop.
//! let smoothed = Ema::new(0.6).smooth(&track)?;
//!
//! // Spans are preserved and values rewritten: the lone 0.31 dip at window 4
//! // lifts to ~0.46, so a 0.5/0.35 hysteresis will not tear the event in two.
//! // Unsmoothed that window still reads 0.31; read alpha as the decay weight
//! // rather than the innovation weight and it reads 0.4387.
//! assert_eq!(smoothed[4].span(), track[4].span());
//! assert!((smoothed[4].value() - 0.46).abs() < 5e-3);
//! # Ok::<(), Error>(())
//! ```
//!
#![cfg_attr(
  feature = "vad",
  doc = "## Segmenting the track into events",
  doc = "",
  doc = "`zuoer` reaches this crate through the `vad` feature, so this block is shown",
  doc = "and run under `ced` + `vad`. Like the projection above, it **runs** — the",
  doc = "segmenter takes probabilities, not audio — and it continues that same",
  doc = "example: the hidden preamble is the previous block's code verbatim, so the",
  doc = "whole chain from hand-built window scores to this event's confidence",
  doc = "executes. Note where the segmenter comes from: `audio::vad` re-exports it,",
  doc = "so nothing here names `zuoer`.",
  doc = "",
  doc = "```",
  doc = "use core::time::Duration;",
  doc = "",
  doc = "use coremlit::audio::vad::{RunOptions, RunSegmenter, SampleRate};",
  doc = "# use coremlit::audio::ced::{",
  doc = "#   Confidences, Error, NUM_CLASSES, RatedSoundEvent, Span, WINDOW_SAMPLES, WindowConfidences,",
  doc = "# };",
  doc = "# use windit::{",
  doc = "#   smooth::{Ema, SmoothPolicy},",
  doc = "#   windowed::Windowed,",
  doc = "# };",
  doc = "",
  doc = "let hop = 16_000;",
  doc = "// `smoothed` is the previous block's result verbatim: twelve hand-built",
  doc = "// `WindowConfidences`, the `Dog` column projected out, `Ema::new(0.6)`.",
  doc = "# let dog = RatedSoundEvent::from_key(\"Dog\")[0].index();",
  doc = "# let barks = [0.02, 0.04, 0.71, 0.86, 0.31, 0.90, 0.88, 0.09, 0.03, 0.01, 0.01, 0.02];",
  doc = "# let windows = barks",
  doc = "#   .iter()",
  doc = "#   .enumerate()",
  doc = "#   .map(|(i, &p)| {",
  doc = "#     let mut scores = vec![0.0; NUM_CLASSES];",
  doc = "#     scores[dog] = p;",
  doc = "#     Ok(WindowConfidences::new(",
  doc = "#       Confidences::try_from_slice(&scores)?,",
  doc = "#       Span::new(i * hop, WINDOW_SAMPLES, WINDOW_SAMPLES),",
  doc = "#     ))",
  doc = "#   })",
  doc = "#   .collect::<Result<Vec<WindowConfidences>, Error>>()?;",
  doc = "# let track: Vec<Windowed<f32>> = windows",
  doc = "#   .iter()",
  doc = "#   .map(|w| Windowed::new(w.value().as_slice()[dog], w.span()))",
  doc = "#   .collect();",
  doc = "# let smoothed = Ema::new(0.6).smooth(&track)?;",
  doc = "",
  doc = "// Every number here is a scenario choice for this one class. coremlit picks",
  doc = "// none of them, and there is no CED default set to fall back on.",
  doc = "let options = RunOptions::default()",
  doc = "  .with_sample_rate(SampleRate::Rate16k)",
  doc = "  .with_start_threshold(0.5)",
  doc = "  .with_end_threshold(0.35)",
  doc = "  .with_min_run_duration(Duration::from_secs(2))",
  doc = "  .with_min_gap_duration(Duration::from_secs(2))",
  doc = "  .with_pad(Duration::ZERO);",
  doc = "let mut segmenter = RunSegmenter::new(options);",
  doc = "// One score per planned window, so the segmenter's frame hop IS the plan's.",
  doc = "segmenter.set_frame_hop(hop);",
  doc = "",
  doc = "let mut events = Vec::new();",
  doc = "for window in &smoothed {",
  doc = "  if let Some(run) = segmenter.push_probability(*window.value()) {",
  doc = "    events.push(run);",
  doc = "  }",
  doc = "}",
  doc = "if let Some(run) = segmenter.finish() {",
  doc = "  events.push(run);",
  doc = "}",
  doc = "",
  doc = "assert_eq!(events.len(), 1);",
  doc = "let event = events[0];",
  doc = "// Both thresholds are load-bearing, and the track straddles them: windows 2",
  doc = "// (0.4388) and 7 (0.3812) clear the 0.35 end threshold but not the 0.5 start",
  doc = "// one. Collapse the pair to a single 0.35 gate and the run opens a window",
  doc = "// early, at 2.0; raise the end threshold to 0.40 and it closes a window",
  doc = "// early, at 7.0. (An end threshold at or above the start one is not a third",
  doc = "// option: `RunOptions` normalizes that back to its derived value.)",
  doc = "assert_eq!((event.start_seconds(), event.end_seconds()), (3.0, 8.0));",
  doc = "",
  doc = "// The event's confidence, on zuoer's terms: mean and peak over the run's own",
  doc = "// frames — padding excluded, bridged frames included. See `audio::vad`'s",
  doc = "// \"Segment confidence\" section for the full statement of those rules. So",
  doc = "// this is the mean of the five SMOOTHED in-run windows — over the whole clip",
  doc = "// it would be 0.3230, and smoothing left out entirely (`Ema::new(1.0)`) the",
  doc = "// same options report a 2.0..7.0 event with mean 0.7320.",
  doc = "assert!((event.mean_probability() - 0.6157).abs() < 1e-3);",
  doc = "assert!((event.peak_probability() - 0.8180).abs() < 1e-3);",
  doc = "# Ok::<(), Error>(())",
  doc = "```"
)]
//!
//! # Compute placement (measured, never marketed)
//!
//! [`DEFAULT_COMPUTE`] ships as [`crate::ComputeUnits::All`], MEASURED: the
//! Wave-C pass (`tests/ced/placement.rs`) characterized per-unit parity and
//! latency across all four sizes and this default is what it pinned. See
//! [`DEFAULT_COMPUTE`] for the numbers.
//!
//! # Performance: construct once, reuse, prewarm
//!
//! Construction pays model load/specialization; [`Classifier::prewarm`] runs
//! one throwaway inference to absorb first-prediction specialization before
//! serving. Fan-out is one [`Classifier`] per worker ([`crate::Model`] is
//! `Send` but deliberately not `Sync`).
//!
//! macOS only (built on [`crate`]).

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, MultiArray};

pub mod aggregate;
pub mod error;
pub mod model;
pub mod prediction;
pub mod window;

mod mel;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use aggregate::{ChunkAggregation, aggregate_windows};
pub use error::Error;
pub use model::{CedModel, ParseCedModelError};
pub use prediction::{Confidences, EventPrediction, RatedSoundEvent, WindowConfidences};
pub use window::{Span, TailPolicy, WindowPlan};

use crate::audio::ced::{
  error::{Result, WinditError},
  mel::{MelExtractor, N_FRAMES, N_MELS},
};

#[cfg(test)]
mod tests;

/// The sample rate this module's contract is defined at: callers decode and
/// resample to **16 kHz mono f32** before calling (sans-I/O — the workspace
/// convention; CED natively matches it).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// The fixed inference-window length in samples: 160 000 = 10 s at 16 kHz,
/// CED's training window. The CoreML export is fixed-shape, so this is model
/// geometry, not a knob (soundevents exposes `window_samples` only because its
/// ONNX graph is dynamic-length — recorded non-goal).
pub const WINDOW_SAMPLES: usize = 160_000;

/// Number of AudioSet classes the model scores: the 527 released rated classes.
/// Compile-time-pinned to `RatedSoundEvent::events().len()` below, so the
/// dataset crate and this module can never drift apart silently.
pub const NUM_CLASSES: usize = 527;

const _: () = assert!(
  soundevents_dataset::RatedSoundEvent::events().len() == NUM_CLASSES,
  "soundevents-dataset's rated label set must have exactly NUM_CLASSES entries"
);

/// Default compute placement: [`ComputeUnits::All`].
///
/// MEASURED, not provisional: the Wave-C placement pass
/// (`tests/ced/placement.rs`) characterized every unit (`CpuOnly`,
/// `CpuAndGpu`, `CpuAndNeuralEngine`, `All`) across all four sizes. Every
/// unit agrees with the `CpuOnly` reference at ≥ 0.99999 cosine and is
/// NaN-free, and warm latency is flat across units (~0.6–0.8 s/clip,
/// dominated by the Rust mel front end, not the CoreML forward) — so
/// `CpuAndGpu` is not faster here, contra the spec's original expectation.
/// The default `All` arm is in fact the numerically
/// *tightest* vs the committed PyTorch fp32 goldens
/// (`tests/ced/parity_logits.rs`: worst cos ~0.99999988, max|Δlogit| ~0.03),
/// and unlike siglip's vision tower the `CpuAndNeuralEngine` arm did not
/// collapse either, so `All` stays the default. Only a *measured* per-size
/// divergence would promote this to a per-[`CedModel`] table; Wave-C found
/// none, so one shared default stands.
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::All;

/// Declared feature names on the CED `.mlmodelc` (pinned by
/// `tests/ced/model_io.rs`). Wave A DECLARES these; the Wave-B export must
/// emit exactly them (we own the conversion), or they change with the probe —
/// the recorded rework seam.
mod names {
  pub const MEL: &str = "mel";
  pub const LOGITS: &str = "logits";
}

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_COMPUTE
}

/// Construction options for the CED [`Classifier`] (rust-options-pattern): a
/// single `compute` knob with one source of truth shared by
/// `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassifierOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_compute",
      with = "crate::audio::ced::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for ClassifierOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ClassifierOptions {
  /// Options matching the module default: [`DEFAULT_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the graph on.
  #[inline]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }

  /// Sets [`Self::compute`] in place.
  #[inline]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// CED sound-event classifier: 16 kHz mono `&[f32]` in, ranked AudioSet
/// predictions out. Loads any of the four [`CedModel`] sizes — they share one
/// mel→logits contract, so this type is size-agnostic and stores no identity.
///
/// The front-end is a Rust log-mel port (the private `mel` submodule); the
/// fp16 CoreML transformer maps the believed `[1, 64, 1001]` mel to `[1, 527]`
/// PRE-sigmoid logits, and sigmoid + ranking run in Rust.
///
/// Point [`Self::from_file`] / [`Self::load`] at the size you staged, composing
/// the path with [`CedModel::mlmodelc_path`]:
///
/// ```no_run
/// use coremlit::audio::ced::{CedModel, Classifier};
/// let models_root = "Models/ced";
/// Classifier::from_file(CedModel::Small.mlmodelc_path(models_root))?;
/// # Ok::<(), coremlit::audio::ced::Error>(())
/// ```
///
/// `&self` inference (no mutable scratch): the FFT plan and filterbank are
/// built once at load and per-call buffers are local, so fan-out means one
/// [`Classifier`] per worker over a `Send` [`crate::Model`] (`crate::Model` is
/// deliberately `!Sync`).
#[derive(Debug)]
pub struct Classifier {
  model: Model,
  mel: MelExtractor,
}

impl Classifier {
  /// Loads the CED `.mlmodelc` from `model_path` with custom `options` — the
  /// primary constructor. Pins the model's believed I/O contract against the
  /// metadata at load (`mel` `[1, 64, 1001]` f32 in, `logits` `[1, 527]` f32
  /// out — the ground truth lives in `tests/ced/model_io.rs`).
  ///
  /// No model is bundled: the `.mlmodelc` is a directory artifact, distributed
  /// via Hugging Face and staged gitignored under `Models/ced/` (Wave B).
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model; [`Error::ContractMismatch`]
  /// if its I/O contract mismatches.
  pub fn load(model_path: impl AsRef<Path>, options: ClassifierOptions) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let description = model.description();

    let input_expected = format!("[1, {N_MELS}, {N_FRAMES}] float32");
    let input = description
      .input(names::MEL)
      .ok_or_else(|| Error::ContractMismatch {
        feature: names::MEL,
        expected: input_expected.clone(),
        actual: "missing".to_string(),
      })?;
    if input.shape() != [1, N_MELS, N_FRAMES] || input.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch {
        feature: names::MEL,
        expected: input_expected,
        actual: describe(input.shape(), input.data_type()),
      });
    }

    let output_expected = format!("[1, {NUM_CLASSES}] float32");
    let output = description
      .output(names::LOGITS)
      .ok_or_else(|| Error::ContractMismatch {
        feature: names::LOGITS,
        expected: output_expected.clone(),
        actual: "missing".to_string(),
      })?;
    if output.shape() != [1, NUM_CLASSES] || output.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch {
        feature: names::LOGITS,
        expected: output_expected,
        actual: describe(output.shape(), output.data_type()),
      });
    }

    Ok(Self {
      model,
      mel: MelExtractor::new(),
    })
  }

  /// Loads the CED `.mlmodelc` with [`ClassifierOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, ClassifierOptions::new())
  }

  /// Scores one fixed window: the `[527]` **PRE-sigmoid** logits — the parity
  /// seam and the power-user escape (custom thresholds in logit space).
  ///
  /// `samples_16k` is 16 kHz mono and must be `1..=`[`WINDOW_SAMPLES`] long; a
  /// shorter input is zero-padded to the fixed window (the believed sub-window
  /// policy, probe-pinned in Wave B); a longer input is rejected — never
  /// silently truncated (route long clips to [`Self::classify_windows`] /
  /// [`Self::classify_long`]).
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples_16k` is empty; [`Error::AudioTooLong`]
  /// if it exceeds [`WINDOW_SAMPLES`]; [`Error::NonFiniteInput`] if any sample
  /// is NaN/infinite (it would silently poison the mel); [`Error::Tensor`] /
  /// [`Error::Prediction`] on a tensor or CoreML failure;
  /// [`Error::OutputShape`] if the predicted `logits` shape diverges from
  /// `[1, `[`NUM_CLASSES`]`]`; [`Error::NonFiniteOutput`] if the model output
  /// has a NaN/infinite logit (model corruption — never reaches sigmoid).
  pub fn raw_scores(&self, samples_16k: &[f32]) -> Result<Vec<f32>> {
    validate_window_input(samples_16k)?;

    let mut features = vec![0.0f32; N_MELS * N_FRAMES];
    self.mel.extract_into(samples_16k, &mut features)?;

    // Freq-major mel [64, 1001] maps directly onto the row-major believed
    // `mel [1, 64, 1001]` contract.
    let input = MultiArray::from_slice(&[1, N_MELS, N_FRAMES], &features)?;
    let mut outputs = self.model.predict_with(&[(names::MEL, &input)])?;
    let logits =
      outputs
        .take(names::LOGITS)
        .ok_or_else(|| crate::PredictionError::MissingOutput {
          name: names::LOGITS.to_string(),
        })?;
    if logits.shape() != [1, NUM_CLASSES] {
      return Err(Error::OutputShape {
        got: logits.shape().to_vec(),
        expected: vec![1, NUM_CLASSES],
      });
    }

    let mut row = vec![0.0f32; NUM_CLASSES];
    logits.copy_into::<f32>(&mut row)?;
    check_finite_logits(&row)?;
    Ok(row)
  }

  /// Classifies one window: the top `k` classes, descending confidence, ties
  /// broken by ascending class index (the soundevents contract) — ties in the
  /// raw logit are broken by ascending class index; distinct logits that
  /// saturate to equal confidences keep logit order. Runs the min-heap over
  /// raw logits and maps sigmoid at extraction. `k == 0` returns an empty vec
  /// without running the model; `k > `[`NUM_CLASSES`] saturates.
  ///
  /// # Errors
  /// As [`Self::raw_scores`]; [`Error::UnknownClassIndex`] is defensive-only.
  pub fn classify(&self, samples_16k: &[f32], k: usize) -> Result<Vec<EventPrediction>> {
    if k == 0 {
      validate_window_input(samples_16k)?;
      return Ok(Vec::new());
    }
    let logits = self.raw_scores(samples_16k)?;
    prediction::top_k_from_scores(logits.into_iter().enumerate(), k, prediction::sigmoid)
  }

  /// All [`NUM_CLASSES`] classes, **ranked** (descending confidence,
  /// soundevents tie-break) — caller-side thresholding. Note this deliberately
  /// differs from soundevents' `classify_all`, which returns model order; the
  /// spec (§4) pins the ranked form.
  ///
  /// # Errors
  /// As [`Self::classify`].
  pub fn classify_all(&self, samples_16k: &[f32]) -> Result<Vec<EventPrediction>> {
    self.classify(samples_16k, NUM_CLASSES)
  }

  /// The long-clip primitive: per-window sigmoid confidences + their
  /// [`Span`]s, ALWAYS exposed — so time-localized tagging ("when did the dog
  /// bark") is a caller-side read of `windows[i].value().as_slice()[class]`
  /// against `windows[i].span()`, no second API needed.
  ///
  /// Slices `samples_16k` at the plan's offsets and runs one
  /// [`Self::raw_scores`] per span (a short tail is zero-padded by the mel
  /// front-end). Runs sequentially: [`crate::Model`] is `!Sync`, so windows
  /// share one classifier on one thread.
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples_16k` is empty; [`Error::Windowing`] if
  /// the plan exceeds [`WindowPlan::max_windows`]
  /// ([`WinditError::TooManyWindows`]) or the span/result buffer cannot be
  /// allocated ([`WinditError::AllocFailed`]); otherwise any per-window
  /// [`Self::raw_scores`] error (a [`Error::NonFiniteInput`] index is relative
  /// to the offending window's start).
  pub fn classify_windows(
    &self,
    samples_16k: &[f32],
    plan: &WindowPlan,
  ) -> Result<Vec<WindowConfidences>> {
    if samples_16k.is_empty() {
      return Err(Error::EmptyAudio);
    }
    let spans = plan.spans(samples_16k.len())?;
    // Fallible reservation: the cap already bounds `spans.len()`, but the result
    // vector is still caller-geometry-sized, so reserve it checked rather than
    // risk an infallible `with_capacity` abort under memory pressure.
    let mut out = Vec::new();
    out.try_reserve_exact(spans.len()).map_err(|_| {
      Error::Windowing(WinditError::AllocFailed {
        elements: spans.len(),
      })
    })?;
    for span in spans {
      let logits = self.raw_scores(&samples_16k[span.start()..span.end()])?;
      out.push(WindowConfidences::new(
        Confidences::from_logits(&logits),
        span,
      ));
    }
    Ok(out)
  }

  /// The composed long-clip convenience: scores each planned window and folds
  /// the per-window confidences (`aggregation`, in confidence space) into one
  /// clip-level [`Confidences`], then returns its [`Confidences::top_k`]`(k)`.
  ///
  /// The fold streams through a shared O([`NUM_CLASSES`]) accumulator — the
  /// per-window vectors are never all held at once, so a long clip that plans
  /// many windows does not retain one 527-float vector per window; use
  /// [`Self::classify_windows`] when per-window access is wanted. `k == 0`
  /// returns an empty vec without running the model OR any windowing, so the
  /// [`WindowPlan::max_windows`] cap does not apply to it (it does the same
  /// finite-sample check the model path would).
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples_16k` is empty; [`Error::Windowing`] if
  /// the plan exceeds [`WindowPlan::max_windows`]
  /// ([`WinditError::TooManyWindows`]) or a buffer cannot be allocated
  /// ([`WinditError::AllocFailed`]); otherwise any per-window
  /// [`Self::raw_scores`] error. ([`Error::EmptyWindows`] is unreachable — a
  /// nonempty clip always plans at least one span.)
  pub fn classify_long(
    &self,
    samples_16k: &[f32],
    k: usize,
    plan: &WindowPlan,
    aggregation: ChunkAggregation,
  ) -> Result<Vec<EventPrediction>> {
    if samples_16k.is_empty() {
      return Err(Error::EmptyAudio);
    }
    if k == 0 {
      // k == 0 skips ALL windowing (and thus the cap), matching the pre-stream
      // behavior; it still rejects a NaN/±∞ clip rather than wave it through.
      check_finite_samples(samples_16k)?;
      return Ok(Vec::new());
    }
    let spans = plan.spans(samples_16k.len())?;
    let mut acc = aggregate::Accumulator::new(aggregation);
    for span in spans {
      let logits = self.raw_scores(&samples_16k[span.start()..span.end()])?;
      acc.push(&Confidences::from_logits(&logits));
    }
    let confidences = acc.finish()?;
    confidences.top_k(k)
  }

  /// Runs one throwaway [`Self::raw_scores`] on a fixed synthetic window to
  /// fully specialize the prediction path, so the first user-facing request is
  /// warm. Construction pays the model load / device specialization; what it
  /// does NOT pay is the first prediction's own graph specialization — calling
  /// `prewarm` once, after construction and before serving, moves that
  /// one-time cost off the first real clip. Then reuse this same classifier
  /// for every request (`&self` — it stays resident).
  ///
  /// The warm-up runs a fixed 1 s 440 Hz tone (zero-padded to the fixed
  /// window), so it neither reads caller audio nor allocates a full-window
  /// buffer up front.
  ///
  /// # Errors
  /// As [`Self::raw_scores`]; a failure here surfaces a broken model at
  /// prewarm time rather than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    let sr = SAMPLE_RATE_HZ as f32;
    let signal: Vec<f32> = (0..SAMPLE_RATE_HZ as usize)
      .map(|i| 0.5 * (std::f32::consts::TAU * 440.0 * (i as f32 / sr)).sin())
      .collect();
    self.raw_scores(&signal)?;
    Ok(())
  }
}

/// Reject a per-window input the pipeline must not see: empty (nothing to
/// classify), longer than the fixed window (never silently truncated — long
/// clips are windowed explicitly), or carrying a NaN/±∞ sample (it would
/// silently poison the mel). Free fn so the guards are hermetically testable
/// without a model.
fn validate_window_input(samples: &[f32]) -> Result<()> {
  if samples.is_empty() {
    return Err(Error::EmptyAudio);
  }
  if samples.len() > WINDOW_SAMPLES {
    return Err(Error::AudioTooLong {
      len: samples.len(),
      max: WINDOW_SAMPLES,
    });
  }
  check_finite_samples(samples)
}

/// Reject a NaN/±∞ sample ([`Error::NonFiniteInput`]) — it would silently
/// poison the mel. The finite-scan shared by [`validate_window_input`] (the
/// single-window path) and `Classifier::classify_long`'s `k == 0` early
/// return, which must skip `validate_window_input`'s `AudioTooLong` bound (a
/// long clip is expected to exceed [`WINDOW_SAMPLES`]) but must still not
/// wave a NaN/∞ clip through as an empty result.
fn check_finite_samples(samples: &[f32]) -> Result<()> {
  if let Some(index) = samples.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteInput { index });
  }
  Ok(())
}

/// Classify a NaN/∞ the CoreML runtime produced as model-output corruption
/// ([`Error::NonFiniteOutput`]) before it can reach sigmoid — a NaN logit
/// would silently rank via `total_cmp` and poison downstream aggregation.
fn check_finite_logits(logits: &[f32]) -> Result<()> {
  if let Some(index) = logits.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteOutput { index });
  }
  Ok(())
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}
