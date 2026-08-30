//! Native CoreML **spoken-language identification** — 16 kHz mono waveform in,
//! ranked languages out ([`NUM_LANGUAGES`] of them: code + English name +
//! model column + natural-log probability).
//!
//! The mel front end runs in Rust (the private `mel` submodule) and the
//! mel→log-probabilities network runs natively on Apple silicon as one
//! `.mlmodelc`. NO `ort` anywhere.
//!
//! # A backend-neutral door
//!
//! Nothing in this module's public surface names the model behind it. The
//! types ([`Identifier`], [`LanguageScore`], [`Language`]), the geometry
//! constants, the feature name, the env var and the models directory are all
//! spelled for the *task*, not the network — so a second or third LID backend
//! can land behind the same seam without renaming a caller's code. The
//! provenance that IS backend-specific — the artifact, its revision, its label
//! roster's origin — lives in the `labels` submodule's docs and in
//! `tests/lid/`, where it belongs.
//!
//! Today that backend is `aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML`
//! @ `2aa4d715a79e410d5f9aa32bd7a4fc9225bf9eb0` (Apache-2.0), an export of
//! `speechbrain/lang-id-voxlingua107-ecapa`.
//!
//! # Model artifacts
//!
//! No model is bundled (a `.mlmodelc` is a directory artifact). It is staged as
//! a gitignored dev-time download under `Models/lid/`, overridable with the
//! `LID_TEST_MODELS` environment variable; its per-file SHA-256 and I/O
//! contract are pinned by `tests/lid/model_io.rs`.
//!
//! # The contract
//!
//! ```text
//! input   mel_features       f32  [1, frames, 60]   frames in 10..=3001, TIME-major
//! output  log_probabilities  f32  [1, 107]          natural log, already softmaxed
//! ```
//!
//! `frames = 1 + n_samples / 160` (integer division), so the accepted envelope
//! is [`MIN_SAMPLES`]..=[`MAX_SAMPLES`] — **0.09 s to 30.01 s** at 16 kHz. The
//! runtime rejects anything outside that; this module rejects it FIRST, as
//! [`Error::FrameCountOutOfRange`], so a caller never has to string-match
//! CoreML's own axis-indexed complaint.
//!
//! Both tensors are fp32 at the boundary, but the graph casts to **fp16**
//! immediately and computes in it throughout. That is why the placements do not
//! all agree to the last digit — the reference clip's top score reads -0.010064
//! on the GPU arm and -0.015625 on `CpuOnly` — and why a parity gate against
//! this door wants a tolerance rather than an equality.
//!
//! ## Clips longer than 30 s
//!
//! Not handled here, deliberately. There is no upstream-authored windowing
//! policy for this model, and the choices that would have to be invented —
//! window length, hop, and how to combine per-window log-probability vectors
//! (mean in log space? in probability space? a vote?) — change the answer.
//! Long-clip support is a follow-up with its own measurements; until then a
//! long clip is a typed error, and the caller slices the audio with a policy
//! it can defend. Note that padding a short clip up to some bucket length is
//! NOT free either — see the performance note below.
//!
//! # Reading the scores
//!
//! The graph's last op is a log-softmax, so the returned values are natural-log
//! probabilities that already sum to 1 under `exp`; no softmax runs in Rust.
//! [`Identifier::identify`] returns the top `k` of them, descending. See
//! [`LanguageScore`] for why the element is a struct rather than a tuple, and
//! why comparisons want the log form.
//!
//! ```no_run
//! use coremlit::audio::lid::{Error, Identifier};
//!
//! # let samples_16k: Vec<f32> = Vec::new();
//! let identifier = Identifier::from_file("Models/lid/lid.mlmodelc")?;
//! for score in identifier.identify(&samples_16k, 3)? {
//!   println!("{:>3} {:<12} {:.4}", score.code(), score.name(), score.probability());
//! }
//! # Ok::<(), Error>(())
//! ```
//!
//! # Compute placement
//!
//! [`DEFAULT_COMPUTE`] is [`ComputeUnits::All`], the house default, and the
//! measurements say it is the right one here — but read [`DEFAULT_COMPUTE`]'s
//! own docs before assuming it is right for a reason. It is not: the ANE arm of
//! this graph is pathological, and `All` currently happens to avoid it.
//!
//! # Performance notes (measured)
//!
//! - **Every unseen frame count costs a one-off specialization**, roughly
//!   55–97 ms, against a steady state of 9–23 ms. A service that sees many
//!   distinct clip lengths pays it repeatedly.
//! - **Padding to bucket lengths does not fix that for free.** The fused
//!   in-graph mean subtraction reduces over the time axis, so it sees the
//!   padding: bucketing shifts tail log-probabilities by up to 3 nats. Bucket
//!   only if you have measured that the shift does not matter for your
//!   decision.
//! - [`Identifier::prewarm`] pays the first prediction's graph specialization
//!   once, off the first real request — for ONE frame count.
//!
//! Fan-out is one [`Identifier`] per worker ([`crate::Model`] is `Send` but
//! deliberately not `Sync`).
//!
//! macOS only (built on [`crate`]).

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, MultiArray};

pub mod error;
pub mod labels;
pub mod prediction;

mod mel;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use error::{ContractMismatch, Error, FrameCountOutOfRange, OutputShape, Result};
pub use labels::{LABELS_JSON_LEN, Language, labels_json_bytes, languages};
pub use prediction::LanguageScore;

use crate::audio::lid::mel::{HOP, MelExtractor, N_MELS};

#[cfg(test)]
mod tests;

/// The sample rate this module's contract is defined at: callers decode and
/// resample to **16 kHz mono f32** before calling (sans-I/O — the workspace
/// convention; the model natively matches it).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Number of languages the model scores — the width of its output row and the
/// length of [`languages`].
pub const NUM_LANGUAGES: usize = 107;

// The roster's length is enforced by its TYPE — `labels`'s table is a
// `[Language; NUM_LANGUAGES]`, so a row added or dropped is a build error, not
// a runtime surprise. A `const` assert restating it here would be true by
// construction and would prove nothing; what the sibling `labels/tests.rs`
// proves instead is the claim that is NOT free: that those rows still match the
// committed asset, entry for entry.

/// Fewest mel frames the graph accepts. Below this the CoreML runtime rejects
/// the input outright; [`Identifier`] rejects it first, as
/// [`Error::FrameCountOutOfRange`].
pub const MIN_FRAMES: usize = 10;

/// Most mel frames the graph accepts (its `RangeDims` upper bound).
pub const MAX_FRAMES: usize = 3_001;

/// Fewest 16 kHz samples that reach [`MIN_FRAMES`]: `(MIN_FRAMES - 1) · 160`
/// = 1 440, or 0.09 s. One sample fewer produces 9 frames and is rejected.
pub const MIN_SAMPLES: usize = (MIN_FRAMES - 1) * HOP;

/// Most 16 kHz samples that still fit [`MAX_FRAMES`]:
/// `MAX_FRAMES · 160 - 1` = 480 159, or ~30.0099 s. Integer division means the
/// last 159 samples of that window are free — 480 000 (exactly 30 s) and
/// 480 159 both give 3 001 frames.
pub const MAX_SAMPLES: usize = MAX_FRAMES * HOP - 1;

/// Number of mel frames a clip of `n_samples` 16 kHz samples produces:
/// `1 + n_samples / 160`, integer division.
///
/// The graph's time axis is exactly this, so it is also the check
/// [`Identifier`] runs before calling the model. Total, and `const` — it
/// answers "will this clip be accepted?" without constructing anything:
///
/// ```
/// use coremlit::audio::lid::{MAX_FRAMES, MAX_SAMPLES, MIN_FRAMES, MIN_SAMPLES, frame_count};
///
/// assert_eq!(frame_count(MIN_SAMPLES), MIN_FRAMES);
/// assert_eq!(frame_count(MIN_SAMPLES - 1), MIN_FRAMES - 1); // rejected
/// assert_eq!(frame_count(MAX_SAMPLES), MAX_FRAMES);
/// assert_eq!(frame_count(MAX_SAMPLES + 1), MAX_FRAMES + 1); // rejected
///
/// // 16 000 samples is one second of audio.
/// assert_eq!(frame_count(16_000), 101);
/// ```
#[inline]
#[must_use]
pub const fn frame_count(n_samples: usize) -> usize {
  1 + n_samples / HOP
}

/// Default compute placement: [`ComputeUnits::All`].
///
/// MEASURED — and correct here for a reason worth writing down, because the
/// reason is luck rather than design:
///
/// | `computeUnits`         | load          | 13 s clip  | 3 s clip |
/// |------------------------|---------------|------------|----------|
/// | `All`                  | 113 ms        | 13.9 ms    | 4.8 ms   |
/// | `CpuAndGpu`            | 50–100 ms     | 13.8 ms    | 7.2 ms   |
/// | `CpuOnly`              | 21 ms         | 24.7 ms    | 6.7 ms   |
/// | `CpuAndNeuralEngine`   | **2 440 ms**  | **145 ms** | **36 ms**|
///
/// `All` is **bit-identical** to `CpuAndGpu` on both clips: it dispatches to
/// the GPU and never touches the ANE. The `CpuAndNeuralEngine` arm is
/// pathological — twenty times the load time, ten times the inference time —
/// and additionally emits `BNNS Graph Shape Deduction: Unsupported kernel id
/// 512` on stderr, i.e. the ANE compiler is falling back mid-graph rather than
/// running it.
///
/// So the house default is right today only because CoreML's own placement
/// heuristic declines the ANE for this graph. That is an OS-version-dependent
/// decision, not a property of the model, and a future macOS that chose
/// differently would make `All` ten times slower with no code change here.
/// This is a KNOWN RISK, recorded so it is a regression rather than a mystery:
/// if this door's latency ever jumps by an order of magnitude, check the
/// placement before anything else, and pin
/// [`IdentifierOptions::with_compute`]`(`[`ComputeUnits::CpuAndGpu`]`)`.
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::All;

/// Declared feature names on the `.mlmodelc` (pinned by
/// `tests/lid/model_io.rs`).
mod names {
  pub const MEL_FEATURES: &str = "mel_features";
  pub const LOG_PROBABILITIES: &str = "log_probabilities";
}

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_COMPUTE
}

/// Construction options for the [`Identifier`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IdentifierOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_compute",
      with = "crate::audio::lid::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for IdentifierOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl IdentifierOptions {
  /// Options matching the module default: [`DEFAULT_COMPUTE`].
  #[must_use]
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

/// Spoken-language identifier: 16 kHz mono `&[f32]` in, ranked
/// [`LanguageScore`]s out.
///
/// The front end is a Rust log-mel port (the private `mel` submodule); the
/// CoreML graph maps `[1, frames, 60]` log-mel features to `[1, `
/// [`NUM_LANGUAGES`] `]` natural-log probabilities, already normalized.
///
/// `&self` inference (no mutable scratch): the FFT plan, window and filterbank
/// are built once at load and per-call buffers are local, so fan-out means one
/// [`Identifier`] per worker over a `Send` [`crate::Model`] (which is
/// deliberately `!Sync`).
#[derive(Debug)]
pub struct Identifier {
  model: Model,
  mel: MelExtractor,
}

impl Identifier {
  /// Loads the `.mlmodelc` from `model_path` with custom `options` — the
  /// primary constructor. Pins the model's I/O contract against its metadata at
  /// load.
  ///
  /// The input's time axis is flexible, so it cannot be pinned to one value
  /// here: [`crate::FeatureInfo::shape`] reports the graph's DEFAULT shape for
  /// a `RangeDims` input (`[1, 301, 60]` for this artifact), not its bounds,
  /// which CoreML does not expose through that snapshot. What is checked is
  /// everything that is fixed — rank 3, unit batch, 60 mel columns, f32 — and
  /// the frame bounds themselves are pinned by `tests/lid/model_io.rs` against
  /// the live runtime.
  ///
  /// No model is bundled: the `.mlmodelc` is a directory artifact, staged
  /// gitignored under `Models/lid/`.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model; [`Error::ContractMismatch`]
  /// if its I/O contract mismatches.
  pub fn load(model_path: impl AsRef<Path>, options: IdentifierOptions) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let description = model.description();

    let input_expected = format!("[1, {MIN_FRAMES}..={MAX_FRAMES}, {N_MELS}] float32");
    let input = description.input(names::MEL_FEATURES).ok_or_else(|| {
      ContractMismatch::new(
        names::MEL_FEATURES,
        input_expected.clone(),
        "missing".to_owned(),
      )
    })?;
    let shape = input.shape();
    let time_axis_plausible = shape.len() == 3
      && shape[0] == 1
      && shape[2] == N_MELS
      && (MIN_FRAMES..=MAX_FRAMES).contains(&shape[1]);
    if !time_axis_plausible || input.data_type() != Some(DataType::F32) {
      return Err(
        ContractMismatch::new(
          names::MEL_FEATURES,
          input_expected,
          describe(shape, input.data_type()),
        )
        .into(),
      );
    }

    let output_expected = format!("[1, {NUM_LANGUAGES}] float32");
    let output = description
      .output(names::LOG_PROBABILITIES)
      .ok_or_else(|| {
        ContractMismatch::new(
          names::LOG_PROBABILITIES,
          output_expected.clone(),
          "missing".to_owned(),
        )
      })?;
    if output.shape() != [1, NUM_LANGUAGES] || output.data_type() != Some(DataType::F32) {
      return Err(
        ContractMismatch::new(
          names::LOG_PROBABILITIES,
          output_expected,
          describe(output.shape(), output.data_type()),
        )
        .into(),
      );
    }

    Ok(Self {
      model,
      mel: MelExtractor::new(),
    })
  }

  /// Loads the `.mlmodelc` with [`IdentifierOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, IdentifierOptions::new())
  }

  /// The full `[`[`NUM_LANGUAGES`]`]` row of **natural-log probabilities**, in
  /// model column order (index `i` is `languages()[i]`) — the parity seam and
  /// the power-user escape.
  ///
  /// Already log-softmaxed by the graph: the values are `<= 0` and `exp` over
  /// the row sums to 1. Nothing is applied on top of them here.
  ///
  /// `samples_16k` is 16 kHz mono and must produce [`MIN_FRAMES`]..=
  /// [`MAX_FRAMES`] frames ([`MIN_SAMPLES`]..=[`MAX_SAMPLES`] samples); it is
  /// never padded or truncated to fit.
  ///
  /// # Errors
  /// [`Error::FrameCountOutOfRange`] if the clip is too short or too long
  /// (empty audio lands here too — zero samples is one frame, below
  /// [`MIN_FRAMES`]); [`Error::NonFiniteInput`] if any sample is NaN or
  /// infinite (it would silently poison the mel); [`Error::Tensor`] /
  /// [`Error::Prediction`] on a tensor or CoreML failure; [`Error::OutputShape`]
  /// if the predicted row's shape diverges from `[1, `[`NUM_LANGUAGES`]`]`;
  /// [`Error::NonFiniteOutput`] if the model emits a NaN or infinite score
  /// (model corruption — never reaches ranking).
  pub fn log_probabilities(&self, samples_16k: &[f32]) -> Result<Vec<f32>> {
    let frames = validate_frame_range(samples_16k.len())?;

    let mut features = vec![0.0f32; frames * N_MELS];
    self.mel.extract_into(samples_16k, &mut features)?;

    // Time-major mel [frames, 60] maps directly onto the row-major
    // `mel_features [1, frames, 60]` contract.
    let input = MultiArray::from_slice(&[1, frames, N_MELS], &features)?;
    let mut outputs = self.model.predict_with(&[(names::MEL_FEATURES, &input)])?;
    let scores = outputs.take(names::LOG_PROBABILITIES).ok_or_else(|| {
      crate::PredictionError::MissingOutput {
        name: names::LOG_PROBABILITIES.to_owned(),
      }
    })?;
    if scores.shape() != [1, NUM_LANGUAGES] {
      return Err(OutputShape::new(scores.shape().to_vec(), vec![1, NUM_LANGUAGES]).into());
    }

    let mut row = vec![0.0f32; NUM_LANGUAGES];
    scores.copy_into::<f32>(&mut row)?;
    if let Some(index) = row.iter().position(|value| !value.is_finite()) {
      return Err(Error::NonFiniteOutput(index));
    }
    Ok(row)
  }

  /// The top `k` languages for `samples_16k`, **descending** by log
  /// probability, ties broken by ascending model column.
  ///
  /// `k == 0` returns an empty vec without running the model (it still applies
  /// the same input validation, so a bad clip is still reported); `k` above
  /// [`NUM_LANGUAGES`] saturates.
  ///
  /// # Errors
  /// As [`Self::log_probabilities`]; [`Error::UnknownLanguageIndex`] is
  /// defensive-only.
  pub fn identify(&self, samples_16k: &[f32], k: usize) -> Result<Vec<LanguageScore>> {
    if k == 0 {
      validate_frame_range(samples_16k.len())?;
      check_finite_samples(samples_16k)?;
      return Ok(Vec::new());
    }
    let scores = self.log_probabilities(samples_16k)?;
    prediction::top_k_from_scores(scores.into_iter().enumerate(), k)
  }

  /// Runs one throwaway inference on a fixed synthetic clip to fully specialize
  /// the prediction path, so the first user-facing request is warm.
  /// Construction pays the model load; what it does NOT pay is the first
  /// prediction's own graph specialization.
  ///
  /// **This warms ONE frame count** — 1 001 frames, a 10 s clip. The
  /// specialization is per frame count (module docs, "Performance notes"), so a
  /// service that will see one clip length should prewarm at that length
  /// instead, by calling [`Self::log_probabilities`] on a throwaway buffer of
  /// the right size.
  ///
  /// # Errors
  /// As [`Self::log_probabilities`]; a failure here surfaces a broken model at
  /// prewarm time rather than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    let rate = SAMPLE_RATE_HZ as f32;
    let signal: Vec<f32> = (0..10 * SAMPLE_RATE_HZ as usize)
      .map(|i| 0.5 * (core::f32::consts::TAU * 440.0 * (i as f32 / rate)).sin())
      .collect();
    self.log_probabilities(&signal)?;
    Ok(())
  }
}

/// Reject a clip whose mel frame count the graph would refuse, returning that
/// frame count when it is in range.
///
/// A free fn so the guard is hermetically testable without a model, and so the
/// rejection happens strictly before [`Model::predict_with`] — the CoreML
/// runtime's own message names an internal axis index and would have to be
/// string-matched to act on.
fn validate_frame_range(n_samples: usize) -> Result<usize> {
  let frames = frame_count(n_samples);
  if !(MIN_FRAMES..=MAX_FRAMES).contains(&frames) {
    return Err(FrameCountOutOfRange::for_samples(n_samples).into());
  }
  Ok(frames)
}

/// Reject a NaN/±∞ sample ([`Error::NonFiniteInput`]) — it would poison the mel
/// frames it touches and, through the whole-utterance `top_db` floor, every
/// other frame as well.
fn check_finite_samples(samples: &[f32]) -> Result<()> {
  match samples.iter().position(|value| !value.is_finite()) {
    Some(index) => Err(Error::NonFiniteInput(index)),
    None => Ok(()),
  }
}

/// Human-readable `shape dtype` rendering for [`ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}
