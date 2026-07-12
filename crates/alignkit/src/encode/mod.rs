//! CoreML wrapper over `base960h_aligner.mlmodelc` (design spec §3
//! Candidate A): the fixed-window wav2vec2 CTC acoustic encoder,
//! `waveform [1, 960_000]` f32 in, `emissions [1, 2999, 29]` f32 out,
//! 20 ms/frame (stride 320 samples @ 16 kHz) — ground truth pinned by
//! `tests/model_io.rs::base960h_aligner_io_matches_spec`.
//!
//! # Fixed-window bridging
//!
//! [`Encoder::emissions`] hides the model's fixed 60 s window behind a
//! variable-length `&[f32]` contract, mirroring asry's own encoder call
//! shape (`(1, T) -> (1, T', V)`, `asry/src/runner/aligner/algorithm/
//! encode.rs`) as closely as a fixed-window CoreML graph allows:
//!
//! - **Longer than [`ENCODER_WINDOW_SAMPLES`]**: rejected with
//!   [`AlignError::InputTooLong`] rather than silently truncated. The
//!   caller (a future `Aligner`, spec §6/§7) is responsible for chunking
//!   audio to at most [`ENCODER_WINDOW_SAMPLES`] before calling — see
//!   "60 s clamp vs asry's `MAX_CHUNK_SIZE`" below for why this crate's
//!   ceiling is far tighter than asry's own per-chunk cap.
//! - **Shorter**: zero-padded up to exactly [`ENCODER_WINDOW_SAMPLES`],
//!   never rejected — unlike `dia-coreml`'s `SegmentModel::infer`, which
//!   rejects rather than pads short input
//!   (`crates/dia-coreml/src/segment/mod.rs`'s "dia contract match"
//!   section). wav2vec2 is not causal, so whether padding perturbs
//!   in-range emissions is an open empirical question, not an assumption —
//!   exactly what Gate 1 (`tests/parity_emissions.rs`) exists to measure,
//!   per design spec §3's Candidate A note.
//! - **`emissions` frames past the real (non-padded) audio**: truncated
//!   away — see [`Encoder::emissions`]'s doc for the exact formula and why
//!   it must be clamped to the model's actual frame count.
//!
//! # 60 s clamp vs asry's `MAX_CHUNK_SIZE`
//!
//! asry's own chunk-size ceiling is `pub const MAX_CHUNK_SIZE: Duration =
//! Duration::from_secs(600);` (`asry/src/core/transcriber.rs:137`) — 10
//! minutes. [`ENCODER_WINDOW_SAMPLES`] is 60 s, ten times tighter. This is
//! a deliberate, DOCUMENTED divergence, not a parity target: asry's 600 s
//! cap bounds per-chunk RAM for its own (non-fixed-window) ONNX wav2vec2
//! path, which allocates proportionally to whatever length it is given.
//! `base960h_aligner.mlmodelc` allocates a *fixed* 960,000-sample
//! input / `[1, 2999, 29]` output tensor pair regardless of how much of it
//! is real audio, so there is no equivalent "let it grow" option on this
//! side — the model's own fixed graph is the ceiling, not a tunable. A
//! caller (the future `Aligner`, spec §6/§7) must chunk audio to at most
//! [`ENCODER_WINDOW_SAMPLES`] before calling [`Encoder::emissions`]; that
//! chunking responsibility is explicitly out of scope here (design spec
//! §7's data flow already assumes per-chunk audio, not a whole-file
//! stream).
//!
//! # No re-softmax
//!
//! [`Encoder::emissions`] builds an [`asry::emissions::LogProbsTV`]
//! directly from the raw
//! `emissions` tensor with **no softmax or log-softmax applied**. This is
//! not an oversight: `tests/model_io.rs`'s
//! `emissions_are_log_probs_not_raw_logits` (Task B1's pinned graph-truth
//! test) empirically confirmed the model's `emissions` output is *already*
//! log-softmaxed (per-frame `logsumexp` ≈ 0 within `1e-2`, and every value
//! bounded by the `log(p) <= 0` ceiling that only a genuine
//! log-probability tensor can hit) — re-applying softmax here would
//! double-normalize and silently corrupt every downstream trellis/beam
//! score. Contrast asry's own `encode_log_softmax`
//! (`asry/src/runner/aligner/algorithm/encode.rs`), which DOES apply
//! `log_softmax_with_finite_guard` to its ONNX model's raw output — that
//! model emits logits, this one does not; the two encoders are not
//! interchangeable drop-ins at the "what does the raw tensor mean" level,
//! only at the "both eventually produce a `LogProbsTV`" level.

use std::{borrow::Cow, path::Path};

use asry::emissions::LogProbsTV;
use coremlit::{ComputeUnits, DataType, Model, MultiArray};

use crate::error::{AlignError, AlignerError};

/// Fixed sample count of the encoder's input window (60 s @ 16 kHz).
/// Pinned by `tests/model_io.rs::base960h_aligner_io_matches_spec`
/// (`waveform [1, 960_000]`). See the module doc's "Fixed-window bridging"
/// and "60 s clamp vs asry's `MAX_CHUNK_SIZE`" sections.
pub const ENCODER_WINDOW_SAMPLES: usize = 960_000;

/// Frame stride: 20 ms @ 16 kHz, matching wav2vec2-base's convention and
/// asry's own `hop_samples` default (`asry/src/runner/aligner/
/// aligner.rs`'s `Aligner::from_paths` doc: "`hop_samples` defaults to
/// 320"). Pinned by the design spec §3/§7 and the model card's "20
/// ms/frame" claim; unlike [`ENCODER_WINDOW_SAMPLES`], this is not itself
/// one of the model's declared tensor dimensions, so it is not
/// introspectable from `coremlit::ModelDescription` — see
/// [`Encoder::emissions`]'s doc for how it combines with the model's
/// actual (introspected) frame count.
pub const HOP_SAMPLES: usize = 320;

/// Declared feature names on `base960h_aligner.mlmodelc`
/// (pinned by `tests/model_io.rs::base960h_aligner_io_matches_spec`).
mod names {
  pub const WAVEFORM: &str = "waveform";
  pub const EMISSIONS: &str = "emissions";
}

/// Default [`EncoderOptions::compute`]. `ComputeUnits::All` lets CoreML
/// schedule across ANE/GPU/CPU, mirroring `dia-coreml`'s
/// `DEFAULT_SEGMENT_COMPUTE` (`crates/dia-coreml/src/segment/mod.rs`).
/// Model-gated tests in this module instead load with
/// `ComputeUnits::CpuOnly` for determinism (no ANE compile-latency
/// variance across runs), matching `tests/model_io.rs`'s introspection
/// convention — production code keeps this default.
pub const DEFAULT_ENCODER_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_encoder_compute() -> ComputeUnits {
  DEFAULT_ENCODER_COMPUTE
}

/// Construction options for [`Encoder`] (rust-options-pattern), mirroring
/// `dia-coreml::segment::SegmentModelOptions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_encoder_compute",
      with = "crate::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for EncoderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl EncoderOptions {
  /// Options matching the crate's default: [`DEFAULT_ENCODER_COMPUTE`]
  /// (`ComputeUnits::All`).
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_ENCODER_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the encoder model on.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  #[inline(always)]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// Human-readable `shape dtype` rendering for
/// [`AlignerError::ContractMismatch`]'s `actual`/`expected` fields.
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// Validates a loaded model's `waveform` input against the pinned
/// contract, in isolation — hermetically testable without a loaded model
/// (unlike `dia-coreml::SegmentModel`'s analogous checks, this crate has
/// no second local model fixture with a mismatched contract to
/// model-gate against; `Models/alignkit/` holds exactly one model, so the
/// validation logic itself is what gets tested, not a specific wrong
/// artifact).
fn check_waveform_contract(shape: &[usize], dtype: Option<DataType>) -> Result<(), AlignerError> {
  if shape != [1, ENCODER_WINDOW_SAMPLES] || dtype != Some(DataType::F32) {
    return Err(AlignerError::ContractMismatch {
      feature: names::WAVEFORM,
      expected: format!("[1, {ENCODER_WINDOW_SAMPLES}] float32"),
      actual: describe(shape, dtype),
    });
  }
  Ok(())
}

/// Validates a loaded model's `emissions` output against the pinned
/// contract, in isolation (see [`check_waveform_contract`]'s doc for why
/// this is hermetic rather than model-gated). Returns the frame count
/// (`shape[1]`) on success — read dynamically rather than hardcoded, see
/// [`Encoder::frames`].
///
/// `shape[1] >= 1`: a zero-frame model would "load fine" and then make
/// every [`Encoder::emissions`] call return an empty result with no
/// error — reject the degenerate contract at construction instead
/// (mirrors `dia-coreml::SegmentModel::from_file_with`'s identical
/// guard).
fn check_emissions_contract(
  shape: &[usize],
  dtype: Option<DataType>,
) -> Result<usize, AlignerError> {
  let shape_ok =
    shape.len() == 3 && shape[0] == 1 && shape[1] >= 1 && shape[2] == crate::vocab::VOCAB_SIZE;
  if !shape_ok || dtype != Some(DataType::F32) {
    return Err(AlignerError::ContractMismatch {
      feature: names::EMISSIONS,
      expected: format!("[1, >=1, {}] float32", crate::vocab::VOCAB_SIZE),
      actual: describe(shape, dtype),
    });
  }
  Ok(shape[1])
}

/// CoreML wrapper over `base960h_aligner.mlmodelc`: one
/// [`ENCODER_WINDOW_SAMPLES`]-sample fixed window in, per-frame CTC
/// log-probabilities out — see the module doc for the padding/truncation
/// contract that bridges this fixed window to asry's variable-length
/// encoder shape.
#[derive(Debug)]
pub struct Encoder {
  model: Model,
  frames: usize,
}

impl Encoder {
  /// Loads the model with [`EncoderOptions::new`] (`ComputeUnits::All`).
  ///
  /// # Errors
  /// As [`Self::from_file_with`].
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AlignerError> {
    Self::from_file_with(path, EncoderOptions::new())
  }

  /// Loads the model with custom options, introspecting and validating its
  /// I/O contract against the ground truth pinned by
  /// `tests/model_io.rs::base960h_aligner_io_matches_spec`.
  ///
  /// # Errors
  /// [`AlignerError::Load`] if CoreML rejects the model.
  /// [`AlignerError::ContractMismatch`] if the loaded model's `waveform`
  /// input isn't `[1, ENCODER_WINDOW_SAMPLES]` f32, or its `emissions`
  /// output isn't rank 3 with `shape[0] == 1`, `shape[1] >= 1`, and
  /// `shape[2] == crate::vocab::VOCAB_SIZE` f32. The frame count
  /// (`shape[1]`) is read dynamically, not hardcoded — mirrors
  /// `dia-coreml::SegmentModel`'s `num_frames` field
  /// (`crates/dia-coreml/src/segment/mod.rs`) — see [`Self::frames`].
  pub fn from_file_with(
    path: impl AsRef<Path>,
    options: EncoderOptions,
  ) -> Result<Self, AlignerError> {
    let model = Model::load(path, options.compute())?;
    let description = model.description();

    let waveform =
      description
        .input(names::WAVEFORM)
        .ok_or_else(|| AlignerError::ContractMismatch {
          feature: names::WAVEFORM,
          expected: format!("[1, {ENCODER_WINDOW_SAMPLES}] float32"),
          actual: "missing".to_string(),
        })?;
    check_waveform_contract(waveform.shape(), waveform.data_type())?;

    let emissions =
      description
        .output(names::EMISSIONS)
        .ok_or_else(|| AlignerError::ContractMismatch {
          feature: names::EMISSIONS,
          expected: format!("[1, >=1, {}] float32", crate::vocab::VOCAB_SIZE),
          actual: "missing".to_string(),
        })?;
    let frames = check_emissions_contract(emissions.shape(), emissions.data_type())?;

    Ok(Self { model, frames })
  }

  /// Output frame count for one full (unpadded) window — the introspected
  /// `emissions` shape's middle dimension (2999 for
  /// `base960h_aligner.mlmodelc`, pinned by
  /// `tests/model_io.rs::base960h_aligner_io_matches_spec`; read
  /// dynamically at construction, not hardcoded).
  #[inline(always)]
  pub const fn frames(&self) -> usize {
    self.frames
  }

  /// Runs the encoder on `samples`, returning per-frame CTC
  /// log-probabilities as an [`asry::emissions::LogProbsTV`] with
  /// `t = ceil(samples.len() / HOP_SAMPLES)` (clamped to [`Self::frames`],
  /// see below) and `v = crate::vocab::VOCAB_SIZE`.
  ///
  /// `samples` shorter than [`ENCODER_WINDOW_SAMPLES`] is zero-padded up to
  /// the full window before prediction; frames computed from the padded
  /// tail are truncated away before this returns, so the result always
  /// reflects only `samples`' own real audio. See the module doc's
  /// "Fixed-window bridging" and "No re-softmax" sections.
  ///
  /// # Truncation formula
  ///
  /// Nominal: `ceil(samples.len() / HOP_SAMPLES)` (design spec §3:
  /// `T_frames = ceil(real_samples / 320)`) — each [`HOP_SAMPLES`]-sample
  /// stride of real audio should contribute (at least) one real frame.
  /// Clamped to [`Self::frames`]: wav2vec2's convolutional feature
  /// extractor is not an exact `samples / HOP_SAMPLES` divider (its
  /// multi-layer kernel/stride chain has kernels slightly wider than their
  /// strides, so a handful of samples at the very end of a full window
  /// contribute no additional frame). Concretely, for `samples.len() ==
  /// ENCODER_WINDOW_SAMPLES` (960,000 — no padding at all, exactly the
  /// `ted_60.wav` fixture's own case), the nominal formula evaluates to
  /// 3,000 (`960_000 / 320`), one more than
  /// `base960h_aligner.mlmodelc`'s actual 2,999
  /// (`tests/model_io.rs::base960h_aligner_io_matches_spec`). Without the
  /// clamp, this method would try to keep a 3,000th frame that was never
  /// written into its `copy_into`-filled buffer for any `samples.len()` in
  /// `(Self::frames() * HOP_SAMPLES, ENCODER_WINDOW_SAMPLES]` — see
  /// `tests.rs` for a regression pinning exactly this boundary.
  ///
  /// # Known gap
  ///
  /// `coremlit::MultiArray::copy_into` validates only the predict-time
  /// `emissions` tensor's *total element count* against
  /// `Self::frames * crate::vocab::VOCAB_SIZE` (established once at
  /// construction) — an axes-swapped runtime output carrying the identical
  /// element count (e.g. `[1, VOCAB_SIZE, frames]` instead of `[1, frames,
  /// VOCAB_SIZE]`) is not independently re-validated per call the way
  /// `dia-coreml::SegmentModel::infer` re-validates its own output shape
  /// on every call (`crates/dia-coreml/src/segment/mod.rs`'s
  /// `check_output_shape`). Accepted here rather than ported: unlike
  /// `dia-coreml`'s `SegmentModel` (which validates a whole family of
  /// possible models sharing one contract), this crate's `Encoder`
  /// contract is pinned to one SHA-tracked model revision
  /// (`tests/model_io.rs`'s module doc), so an axis swap surfacing between
  /// two predictions of an already-loaded, already-contract-validated
  /// `Model` would be a CoreML runtime regression, not a data-dependent
  /// outcome this crate's own inputs can trigger.
  ///
  /// # Errors
  /// [`AlignError::InputTooLong`] if `samples.len() > ENCODER_WINDOW_SAMPLES`.
  /// [`AlignError::Tensor`] if building the input tensor or reading the
  /// output tensor fails. [`AlignError::Prediction`] on a CoreML prediction
  /// failure, including a prediction whose runtime output set omits
  /// `emissions` entirely.
  pub fn emissions(&self, samples: &[f32]) -> Result<LogProbsTV, AlignError> {
    if samples.len() > ENCODER_WINDOW_SAMPLES {
      return Err(AlignError::InputTooLong {
        got: samples.len(),
        max: ENCODER_WINDOW_SAMPLES,
      });
    }

    let waveform: Cow<'_, [f32]> = if samples.len() == ENCODER_WINDOW_SAMPLES {
      Cow::Borrowed(samples)
    } else {
      let mut buf = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
      buf[..samples.len()].copy_from_slice(samples);
      Cow::Owned(buf)
    };

    let input = MultiArray::from_slice(&[1, ENCODER_WINDOW_SAMPLES], waveform.as_ref())?;
    let mut outputs = self.model.predict_with(&[(names::WAVEFORM, &input)])?;
    let emissions =
      outputs
        .take(names::EMISSIONS)
        .ok_or_else(|| coremlit::PredictionError::MissingOutput {
          name: names::EMISSIONS.to_string(),
        })?;

    let mut raw = vec![0.0f32; self.frames * crate::vocab::VOCAB_SIZE];
    emissions.copy_into::<f32>(&mut raw)?;

    let real_frames = truncated_frame_count(samples.len(), self.frames);
    // `real_frames <= self.frames` always (see `truncated_frame_count`'s
    // clamp), so `real_frames * VOCAB_SIZE <= raw.len() == self.frames *
    // VOCAB_SIZE` and `truncate` below always shrinks to exactly that
    // length (never a no-op past `raw.len()`, which would leave `raw`
    // longer than `real_frames * VOCAB_SIZE`).
    raw.truncate(real_frames * crate::vocab::VOCAB_SIZE);

    Ok(
      LogProbsTV::new(real_frames, crate::vocab::VOCAB_SIZE, raw).expect(
        "real_frames * VOCAB_SIZE == raw.len() by construction (see the truncate call above), \
         and VOCAB_SIZE is a nonzero crate constant, so LogProbsTV::new's only two failure \
         modes cannot fire here",
      ),
    )
  }
}

/// See [`Encoder::emissions`]'s "Truncation formula" doc section.
fn truncated_frame_count(real_samples: usize, available_frames: usize) -> usize {
  real_samples.div_ceil(HOP_SAMPLES).min(available_frames)
}

#[cfg(test)]
mod tests;
