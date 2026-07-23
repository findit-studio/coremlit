//! Native CoreML **CED-tiny** AudioSet sound-event tagging — coremlit's first
//! multi-label classifier: 16 kHz mono waveform in, ranked AudioSet predictions
//! out (527 rated classes: name + `/m/…` id + class index + sigmoid
//! confidence), long clips via windowed chunking + Mean/Max aggregation.
//!
//! CED (Consistent Ensemble Distillation, arXiv 2308.11957; upstream
//! RicherMans/CED, `mispeech/ced-tiny`) is a distilled AudioSet transformer.
//! The mel front-end runs in Rust (the private `mel` submodule) and the
//! mel→logits transformer runs natively on Apple silicon as one fp16
//! `.mlmodelc` — an in-graph STFT/mel is the exact fragility class behind the
//! ORT CoreML EP zeroed-logits bug this feature closes. NO `ort` anywhere.
//!
//! Design spec: `docs/superpowers/specs/2026-07-23-ced-native-ane-design.md`.
//!
//! # Model artifacts
//!
//! No model is bundled (a `.mlmodelc` is a directory artifact). The fp16 CED
//! graph is converted owner-side (Wave B), distributed via Hugging Face, and
//! staged as a gitignored dev-time download under `Models/ced/` (env override
//! `CED_TEST_MODELS`); its per-file SHA-256 and I/O contract are pinned by
//! `tests/ced/model_io.rs` once staged.
//!
//! # Rust front-end around an fp16 CoreML graph
//!
//! The graph takes the believed `[1, 64, 1001]` log-mel (`mel`, f32) computed
//! by this module's Rust front-end and emits `[1, 527]` **pre-sigmoid** logits
//! (`logits`, f32); sigmoid, ranking, and long-clip aggregation run in Rust.
//! The believed mel numerics are probe-pinned in Wave B (see the `mel`
//! submodule docs).
//!
//! # Compute placement (measured, never marketed)
//!
//! [`DEFAULT_COMPUTE`] ships as [`crate::ComputeUnits::All`] and is
//! **PROVISIONAL**: no CED conversion has been measured yet. The Wave-C
//! placement pass (`tests/ced/placement.rs`) characterizes per-unit parity and
//! latency and re-pins the measured winner.
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
pub mod prediction;
pub mod window;

mod mel;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use aggregate::{ChunkAggregation, aggregate_windows};
pub use error::Error;
pub use prediction::{Confidences, EventPrediction, RatedSoundEvent, WindowConfidences};
pub use window::{Span, TailPolicy, WindowPlan};

use crate::audio::ced::{
  error::Result,
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
/// **PROVISIONAL** — placement is measured, never marketed, and the CED
/// conversion has not been measured yet: the Wave-C placement pass
/// (`tests/ced/placement.rs`) re-pins this to the measured winner (the spec
/// anticipates `CpuAndGpu`; ANE-capable ≠ floor-holding — the siglip lesson)
/// and this doc then carries the measured latency × placement table.
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
  /// Options matching the module default: [`DEFAULT_COMPUTE`] (PROVISIONAL —
  /// see its doc).
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

/// CED-tiny sound-event classifier: 16 kHz mono `&[f32]` in, ranked AudioSet
/// predictions out.
///
/// The front-end is a Rust log-mel port (the private `mel` submodule); the
/// fp16 CoreML transformer maps the believed `[1, 64, 1001]` mel to `[1, 527]`
/// PRE-sigmoid logits, and sigmoid + ranking run in Rust.
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
  /// broken by ascending class index (the soundevents contract). Runs the
  /// min-heap over raw logits and maps sigmoid at extraction. `k == 0` returns
  /// an empty vec without running the model; `k > `[`NUM_CLASSES`] saturates.
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
  /// [`Error::EmptyAudio`] if `samples_16k` is empty; otherwise any per-window
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
    let spans = plan.spans(samples_16k.len());
    let mut out = Vec::with_capacity(spans.len());
    for span in spans {
      let logits = self.raw_scores(&samples_16k[span.start()..span.end()])?;
      out.push(WindowConfidences::new(
        Confidences::from_logits(&logits),
        span,
      ));
    }
    Ok(out)
  }

  /// The composed long-clip convenience: [`Self::classify_windows`] →
  /// [`aggregate_windows`] (`aggregation`, in confidence space) →
  /// [`Confidences::top_k`]`(k)`. `k == 0` returns an empty vec without
  /// running the model.
  ///
  /// # Errors
  /// As [`Self::classify_windows`] and [`aggregate_windows`]
  /// ([`Error::EmptyWindows`] is unreachable here — a nonempty clip always
  /// plans at least one span).
  pub fn classify_long(
    &self,
    samples_16k: &[f32],
    k: usize,
    plan: &WindowPlan,
    aggregation: ChunkAggregation,
  ) -> Result<Vec<EventPrediction>> {
    if k == 0 {
      if samples_16k.is_empty() {
        return Err(Error::EmptyAudio);
      }
      return Ok(Vec::new());
    }
    let windows = self.classify_windows(samples_16k, plan)?;
    let confidences = aggregate_windows(aggregation, &windows)?;
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
