//! CoreML wrapper for `pyannote_segmentation.mlmodelc` (spec §4) and the
//! powerset→multilabel decode that turns its raw logits into dia's
//! `segmentations` tensor layout.
//!
//! Ports the model-facing half of dia's `segment` stage — `SegmentModel`
//! (`diarization/src/segment/model.rs`) — over `coremlit` instead of `ort`,
//! plus the powerset argmax dia's owned audio-in pipeline performs inline
//! (`diarization/src/offline/owned.rs:476-499`; dia's own
//! `powerset_to_speakers_hard`, `diarization/src/segment/powerset.rs:
//! 68-87`). Ground truth: `tests/model_io.rs`'s
//! `pyannote_segmentation_io_matches_spec` introspection test
//! (`pyannote_segmentation.mlmodelc`: `audio [1, 1, 160_000]` f32 in,
//! `segments [1, 589, 7]` f32 out) and the design spec §4/§5.
//!
//! # dia contract match
//!
//! - **Padding**: dia's `SegmentModel::infer` (the ort analog,
//!   `diarization/src/segment/model.rs:280-357`) requires its caller to
//!   already have exactly `WINDOW_SAMPLES` samples —
//!   `debug_assert_eq!(samples.len(), WINDOW_SAMPLES as usize)`
//!   (`model.rs:281`) — and does not pad internally; zero-padding happens
//!   one layer up, in `Segmenter::emit_window`
//!   (`diarization/src/segment/segmenter.rs:250-266`) or, for the owned
//!   audio-in pipeline, in `OwnedDiarizationPipeline`'s own chunk loop
//!   (`diarization/src/offline/owned.rs:453-475`, `padded_chunk`).
//!   [`SegmentModel::infer`] mirrors that same boundary: it REJECTS
//!   (`InferError::InputLength`) rather than pads, so a caller (a future
//!   `Extractor`, mirroring `owned.rs`'s loop) is responsible for
//!   zero-padding short chunks before calling — exactly dia's contract,
//!   made a checked `Result` instead of a debug-only assertion.
//! - **Non-finite outputs**: dia's `infer` rejects non-finite logits AFTER
//!   extracting them (`Error::NonFiniteOutput`, `model.rs:346-355`) —
//!   [`SegmentModel::infer`] does the same (`InferError::NonFiniteOutput`).
//!   dia additionally rejects non-finite *input* samples before running
//!   inference (`Error::NonFiniteInput`, `model.rs:283-290`); this crate's
//!   `InferError` (pinned by Task 1, outside this task's file scope) has no
//!   matching input-side variant, so that specific boundary isn't
//!   duplicated here. A non-finite input sample is caught by the
//!   (already-present) output scan whenever it propagates to a non-finite
//!   logit — the typical IEEE-arithmetic outcome, but not a guarantee
//!   (a kernel could in principle absorb it into finite garbage), so this
//!   is a KNOWN, documented gap vs dia's earlier, more specific guard;
//!   adding an input-side variant when `error/mod.rs` is next in scope
//!   closes it.
//! - **Layout re-validation**: dia's `infer` re-validates its output's
//!   shape/layout on every call (`model.rs:313-338`) because, per its own
//!   doc comment, "Load-time dimension verification ... is reserved for a
//!   future revision once a stable ort metadata API is available"
//!   (`model.rs:134-138`). `coremlit::Model::description()` already IS
//!   that stable metadata API, so [`SegmentModel::from_file_with`]
//!   validates the declared contract (presence, dtype, base shape) once
//!   at construction (`ModelError::ContractMismatch`) instead of
//!   re-deriving it from scratch on every call — a deliberate, documented
//!   improvement dia's own comment anticipates. A stable *declared*
//!   contract at load time is not, however, a guarantee that every
//!   individual CoreML prediction actually returns output matching it —
//!   the runtime is a trust boundary independent of whether its metadata
//!   API is introspectable — so [`SegmentModel::infer`] also re-validates
//!   the predict-time `segments` tensor's shape on every call
//!   (`InferError::OutputShape`), matching dia's own per-call re-check.
//! - **dtype**: [`multilabel`] returns `Vec<f64>` because dia's
//!   `OfflineInput::segmentations` field is `&'a [f64]`
//!   (`diarization/src/offline/algo.rs:179`); dia itself produces those
//!   values by computing the hard 0/1 mask in `f32` and casting `as f64`
//!   at the point it writes into its own segmentations buffer
//!   (`diarization/src/offline/owned.rs:496`). `0.0_f32 as f64` and
//!   `1.0_f32 as f64` are exact (both values are exactly representable in
//!   both formats), so building the lookup table directly in `f64` here is
//!   bit-identical to dia's f32-then-cast.
//!
//! # No-softmax equivalence
//!
//! dia's audio-in pipeline computes `softmax_row(&row)` before argmaxing
//! (`diarization/src/offline/owned.rs:484-494`). [`multilabel`] argmaxes
//! the RAW logits instead — no softmax. These are equivalent because
//! softmax, evaluated over one fixed row, is a strictly monotonic, order-
//! AND tie-preserving transform of that row's logits: every class in a row
//! shares the same normalizing denominator `D = Σ exp(logits)`, so
//! `softmax(logits)_i = exp(logits_i) / D`, and since `exp` is strictly
//! increasing and `D > 0`:
//!
//! - `logits_i > logits_j  <=>  softmax(logits)_i > softmax(logits)_j`
//! - `logits_i == logits_j <=> softmax(logits)_i == softmax(logits)_j`
//!
//! in exact (real-number) arithmetic — so `argmax` over raw logits and
//! `argmax` over softmaxed probabilities always select the same class,
//! including which side of a tie wins. Skipping softmax also saves 7
//! `exp` calls and a division per frame for an identical result.
//!
//! **Floating-point caveat** (why this holds "in practice", not as an
//! absolute guarantee): the equivalence above assumes exact arithmetic.
//! `expf`'s rounding means it is (astronomically unlikely, but not
//! impossible) for two DISTINCT `f32` logits close enough together to
//! round to the IDENTICAL `f32` softmax output after `exp`+divide, while
//! the raw logits themselves still compare unequal. In that specific edge
//! case, raw-logit argmax and softmax-then-argmax could pick different
//! classes on what raw-logit argmax sees as a clear (if extremely close)
//! winner but softmax-then-argmax sees as an exact tie. Real segmentation-
//! model logits are not adversarially constructed to sit on this boundary,
//! so this is a documented theoretical caveat, not an observed divergence.
//! An EXACT tie in the raw logits, by contrast, is provably also an exact
//! tie after softmax (same input bits in, same `exp` output bits out), so
//! the ordinary tie-handling case below is exact, not approximate.
//!
//! # Tie handling
//!
//! [`multilabel`]'s argmax uses strict `>` against a running max seeded
//! from class 0, exactly mirroring dia's `powerset_to_speakers_hard`
//! (`diarization/src/segment/powerset.rs:69-76`): on an exact tie, the
//! class with the LOWEST index wins, because a later equal value never
//! satisfies `p > max`.

use std::path::Path;

use coremlit::{ComputeUnits, DataType, Model, MultiArray};

use crate::error::{InferError, ModelError};

/// Sample count of one segmentation-model chunk (10 s at 16 kHz). Matches
/// dia's `WINDOW_SAMPLES` (`diarization/src/segment/options.rs:18`) and the
/// introspected `pyannote_segmentation.mlmodelc` `audio` input's fixed
/// shape `[1, 1, 160_000]` (design spec §4; pinned by
/// `tests/model_io.rs::pyannote_segmentation_io_matches_spec`).
pub const SEG_CHUNK_SAMPLES: usize = 160_000;

/// Maximum simultaneous speakers the powerset encoding represents. Matches
/// dia's `MAX_SPEAKER_SLOTS` (`diarization/src/segment/options.rs:43`) —
/// `usize` here (not dia's `u8`) because every use in this crate is a
/// slice length or index, not a compact wire value.
pub const SEG_NUM_SLOTS: usize = 3;

/// Powerset class count: silence, A, B, C, A+B, A+C, B+C. Matches dia's
/// `POWERSET_CLASSES` (`diarization/src/segment/options.rs:40`) and the
/// introspected `segments` output's trailing dimension (design spec §4;
/// `[1, 589, 7]`).
pub const POWERSET_CLASSES: usize = 7;

/// Declared feature names on `pyannote_segmentation.mlmodelc`
/// (pinned by `tests/model_io.rs::pyannote_segmentation_io_matches_spec`).
mod names {
  pub const AUDIO: &str = "audio";
  pub const SEGMENTS: &str = "segments";
}

/// Default [`SegmentModelOptions::compute`]. `ComputeUnits::All` lets
/// CoreML schedule across ANE/GPU/CPU — the whole point of this crate
/// (design spec §1's ~20x segmentation uplift target). Model-gated tests
/// in this module instead load with `ComputeUnits::CpuOnly` for
/// determinism (no ANE compile-latency variance across runs), matching
/// `tests/model_io.rs`'s introspection convention (every load there also
/// uses `ComputeUnits::CpuOnly`) — production code keeps this default.
pub const DEFAULT_SEGMENT_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_segment_compute() -> ComputeUnits {
  DEFAULT_SEGMENT_COMPUTE
}

// `coremlit::ComputeUnits` carries no serde impl of its own (coremlit has
// no serde dependency at all) — bridge it through its existing
// `as_str`/`FromStr`, the same shape whisperkit's private
// `options::compute_units_serde` module uses
// (crates/whisperkit/src/options/mod.rs).
#[cfg(feature = "serde")]
mod compute_units_serde {
  use core::str::FromStr;

  use coremlit::ComputeUnits;
  use serde::{Deserialize, Deserializer, Serializer};

  pub(super) fn serialize<S: Serializer>(
    value: &ComputeUnits,
    serializer: S,
  ) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(value.as_str())
  }

  pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
  ) -> Result<ComputeUnits, D::Error> {
    let name = String::deserialize(deserializer)?;
    ComputeUnits::from_str(&name).map_err(serde::de::Error::custom)
  }
}

/// Construction options for [`SegmentModel`] (rust-options-pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentModelOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_segment_compute", with = "compute_units_serde")
  )]
  compute: ComputeUnits,
}

impl Default for SegmentModelOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl SegmentModelOptions {
  /// Options matching the crate's default: [`DEFAULT_SEGMENT_COMPUTE`]
  /// (`ComputeUnits::All`).
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_SEGMENT_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the segmentation model on.
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
/// [`ModelError::ContractMismatch`]'s `actual`/`expected` fields.
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// CoreML wrapper over `pyannote_segmentation.mlmodelc`: one
/// [`SEG_CHUNK_SAMPLES`]-sample chunk in, flattened `[num_frames *
/// POWERSET_CLASSES]` raw powerset logits out — layout-identical to dia's
/// `SegmentModel::infer` (`diarization/src/segment/model.rs:280-357`; see
/// the module doc's "dia contract match" section).
#[derive(Debug)]
pub struct SegmentModel {
  model: Model,
  num_frames: usize,
}

impl SegmentModel {
  /// Loads the model with [`SegmentModelOptions::new`] (`ComputeUnits::All`).
  ///
  /// # Errors
  /// As [`Self::from_file_with`].
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
    Self::from_file_with(path, SegmentModelOptions::new())
  }

  /// Loads the model with custom options, introspecting and validating its
  /// I/O contract against the ground truth pinned by
  /// `tests/model_io.rs::pyannote_segmentation_io_matches_spec`.
  ///
  /// # Errors
  /// [`ModelError::Load`] if CoreML rejects the model.
  /// [`ModelError::ContractMismatch`] if the loaded model's `audio` input
  /// isn't `[1, 1, SEG_CHUNK_SAMPLES]` f32, or its `segments` output isn't
  /// rank 3 with `shape[0] == 1` and `shape[2] == POWERSET_CLASSES` f32.
  /// The frame count (`shape[1]`) is read dynamically, not hardcoded — see
  /// [`Self::num_frames`].
  pub fn from_file_with(
    path: impl AsRef<Path>,
    options: SegmentModelOptions,
  ) -> Result<Self, ModelError> {
    let model = Model::load(path, options.compute())?;
    let description = model.description();

    let audio = description
      .input(names::AUDIO)
      .ok_or_else(|| ModelError::ContractMismatch {
        feature: names::AUDIO,
        expected: format!("[1, 1, {SEG_CHUNK_SAMPLES}] float32"),
        actual: "missing".to_string(),
      })?;
    if audio.shape() != [1, 1, SEG_CHUNK_SAMPLES] || audio.data_type() != Some(DataType::F32) {
      return Err(ModelError::ContractMismatch {
        feature: names::AUDIO,
        expected: format!("[1, 1, {SEG_CHUNK_SAMPLES}] float32"),
        actual: describe(audio.shape(), audio.data_type()),
      });
    }

    let segments =
      description
        .output(names::SEGMENTS)
        .ok_or_else(|| ModelError::ContractMismatch {
          feature: names::SEGMENTS,
          expected: format!("[1, >=1, {POWERSET_CLASSES}] float32"),
          actual: "missing".to_string(),
        })?;
    let shape = segments.shape();
    // `shape[1] >= 1`: a zero-frame model would "load fine" and then make
    // every infer() return an empty Vec with no error — reject the
    // degenerate contract at construction instead.
    let shape_ok =
      shape.len() == 3 && shape[0] == 1 && shape[1] >= 1 && shape[2] == POWERSET_CLASSES;
    if !shape_ok || segments.data_type() != Some(DataType::F32) {
      return Err(ModelError::ContractMismatch {
        feature: names::SEGMENTS,
        expected: format!("[1, >=1, {POWERSET_CLASSES}] float32"),
        actual: describe(shape, segments.data_type()),
      });
    }
    let num_frames = shape[1];

    Ok(Self { model, num_frames })
  }

  /// Output frame count for one chunk — the introspected `segments`
  /// shape's middle dimension (589 for `pyannote_segmentation.mlmodelc`,
  /// pinned by `tests/model_io.rs::pyannote_segmentation_io_matches_spec`;
  /// read dynamically at construction, not hardcoded).
  #[inline(always)]
  pub const fn num_frames(&self) -> usize {
    self.num_frames
  }

  /// Runs one segmentation chunk, returning flattened `[num_frames *
  /// POWERSET_CLASSES]` raw powerset logits, row-major `[frame][class]` —
  /// layout-identical to dia's `SegmentModel::infer`
  /// (`diarization/src/segment/model.rs:280-357`).
  ///
  /// # Errors
  /// [`InferError::InputLength`] unless `samples.len() == SEG_CHUNK_SAMPLES`
  /// — see the module doc's "dia contract match" section for why this
  /// rejects rather than pads. [`InferError::Prediction`] /
  /// [`InferError::Tensor`] on a CoreML or tensor-construction failure —
  /// including a prediction whose runtime output set omits `segments`
  /// entirely ([`coremlit::PredictionError::MissingOutput`]; the
  /// construction-time contract pins the *declared* outputs, but the
  /// runtime provider's name set is CoreML's to produce per call).
  /// [`InferError::OutputShape`] if the predict-time `segments` tensor's
  /// shape diverges from `[1, num_frames, POWERSET_CLASSES]`. The
  /// construction-time contract validated once in [`Self::from_file_with`]
  /// is a claim about the model's declared shape, not a guarantee about
  /// every individual prediction; the CoreML runtime is a trust boundary
  /// re-checked here on every call, exactly as dia's `infer` re-validates
  /// output layout on every call
  /// (`diarization/src/segment/model.rs:313-338`). This also covers what
  /// [`coremlit::MultiArray::copy_into`] cannot: it validates only total
  /// element count, so an axes-swapped runtime output would otherwise be
  /// silently transposed into `logits` instead of erroring.
  /// [`InferError::NonFiniteOutput`] if any output logit is NaN or
  /// infinite — the exact `ort` CoreML-EP corruption mode (design spec §1)
  /// this crate exists to replace.
  pub fn infer(&self, samples: &[f32]) -> Result<Vec<f32>, InferError> {
    check_input_length(samples.len())?;

    let audio = MultiArray::from_slice(&[1, 1, SEG_CHUNK_SAMPLES], samples)?;
    let mut outputs = self.model.predict_with(&[(names::AUDIO, &audio)])?;
    let segments =
      outputs
        .take(names::SEGMENTS)
        .ok_or_else(|| coremlit::PredictionError::MissingOutput {
          name: names::SEGMENTS.to_string(),
        })?;
    // The construction-time contract pins the model's DECLARED shape; the
    // CoreML runtime producing a specific prediction's tensor is a
    // separate trust boundary, re-checked here on every call exactly as
    // dia's `infer` re-validates output layout on every call
    // (`diarization/src/segment/model.rs:313-338`). `copy_into` below only
    // validates total element count, so this must run first.
    check_output_shape(segments.shape(), self.num_frames)?;

    let mut logits = vec![0.0f32; self.num_frames * POWERSET_CLASSES];
    segments.copy_into::<f32>(&mut logits)?;
    check_finite(&logits)?;

    Ok(logits)
  }
}

/// Validates `infer`'s input-length contract in isolation — hermetically
/// testable without a loaded model (`infer` itself needs `&self`, i.e. a
/// real loaded [`SegmentModel`], to reach the CoreML call this guards).
fn check_input_length(got: usize) -> Result<(), InferError> {
  if got != SEG_CHUNK_SAMPLES {
    return Err(InferError::InputLength {
      got,
      expected: SEG_CHUNK_SAMPLES,
    });
  }
  Ok(())
}

/// Validates a predict-time `segments` tensor's shape against the
/// construction-time contract (`[1, num_frames, POWERSET_CLASSES]`) in
/// isolation — hermetically testable without a loaded model (`infer`
/// itself needs `&self`, i.e. a real loaded [`SegmentModel`], to reach the
/// CoreML call this guards). Catches exactly what
/// [`coremlit::MultiArray::copy_into`] cannot: it validates only total
/// element count, so an axes-swapped `[1, POWERSET_CLASSES, num_frames]`
/// tensor — the same element count as the expected
/// `[1, num_frames, POWERSET_CLASSES]` — would otherwise pass `copy_into`
/// silently and transpose frames and classes into `logits`.
fn check_output_shape(shape: &[usize], num_frames: usize) -> Result<(), InferError> {
  let shape_ok =
    shape.len() == 3 && shape[0] == 1 && shape[1] == num_frames && shape[2] == POWERSET_CLASSES;
  if !shape_ok {
    return Err(InferError::OutputShape {
      got: shape.to_vec(),
      expected: vec![1, num_frames, POWERSET_CLASSES],
    });
  }
  Ok(())
}

/// Scans `logits` for the first non-finite value — "the exact `ort`
/// CoreML-EP corruption mode this crate exists to replace" (see the module
/// doc). Extracted from [`SegmentModel::infer`] so it is hermetically
/// testable without a loaded model.
fn check_finite(logits: &[f32]) -> Result<(), InferError> {
  if let Some(index) = logits.iter().position(|v| !v.is_finite()) {
    return Err(InferError::NonFiniteOutput { index });
  }
  Ok(())
}

/// Powerset class index → hard 3-slot speaker mask. Order and encoding
/// match dia's `TABLE` exactly (`diarization/src/segment/powerset.rs:
/// 77-85`): index 0 is silence, 1..=3 are single speakers A/B/C, 4..=6 are
/// the three two-speaker overlaps A+B/A+C/B+C. `f64` here (dia's table is
/// `f32`) matches dia's `OfflineInput::segmentations` dtype — see the
/// module doc.
const POWERSET_TABLE: [[f64; SEG_NUM_SLOTS]; POWERSET_CLASSES] = [
  [0.0, 0.0, 0.0], // silence
  [1.0, 0.0, 0.0], // A
  [0.0, 1.0, 0.0], // B
  [0.0, 0.0, 1.0], // C
  [1.0, 1.0, 0.0], // A+B
  [1.0, 0.0, 1.0], // A+C
  [0.0, 1.0, 1.0], // B+C
];

/// Hard argmax over one frame's [`POWERSET_CLASSES`] logits, ties broken
/// toward the lowest class index — see the module doc's "Tie handling"
/// section.
///
/// The plain `>` comparison is dia's exact semantics
/// (`diarization/src/segment/powerset.rs:72`), including for non-finite
/// input (which [`SegmentModel::infer`] has already rejected on the
/// production path, exactly as dia's own callers do): `NaN > x` and
/// `x > NaN` are both false, so a NaN never displaces the running max, and
/// a NaN *seed* (class 0) sticks there. A NaN-total-ordering comparison
/// (`total_cmp`) would NOT be equivalent — it also orders `-0.0 < +0.0`,
/// breaking dia's lowest-index rule on that exact-tie case.
fn hard_argmax(row: &[f32; POWERSET_CLASSES]) -> usize {
  let mut argmax = 0usize;
  let mut max = row[0];
  for (k, &v) in row.iter().enumerate().skip(1) {
    if v > max {
      max = v;
      argmax = k;
    }
  }
  argmax
}

/// Powerset argmax → hard multilabel over a whole `[num_frames *
/// POWERSET_CLASSES]` logits buffer (e.g. [`SegmentModel::infer`]'s return
/// value): `[num_frames * SEG_NUM_SLOTS]` flattened `f64` 0.0/1.0,
/// frame-major (`frame * SEG_NUM_SLOTS + slot`) — the same layout dia's
/// `segmentations` buffer uses for one chunk
/// (`diarization/src/offline/owned.rs:496`:
/// `segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s]`, `c = 0`). See
/// the module doc's "No-softmax equivalence" and "Tie handling" sections
/// for why this operates on raw logits, no softmax, and how ties resolve.
///
/// `logits` is expected to be finite, as [`SegmentModel::infer`] guarantees
/// for its own output — the same precondition dia's callers establish
/// before its powerset decode (`Segmenter::push_inference` rejects
/// `NonFiniteScores`; the owned pipeline's `infer` rejects
/// `NonFiniteOutput`). Non-finite values are not rejected here, matching
/// dia: a NaN simply never wins the argmax (see the private `hard_argmax`
/// helper's doc for the exact comparison semantics).
///
/// # Panics
/// Panics if `logits.len() != num_frames * POWERSET_CLASSES` — in every
/// build profile, mirroring dia's own fail-fast (dia's inline decode
/// indexes `logits[f * POWERSET_CLASSES + k]` directly,
/// `diarization/src/offline/owned.rs:482`, which panics on a short buffer
/// in release too; silently truncating instead would hand downstream
/// consumers a misaligned `segmentations` buffer).
pub fn multilabel(logits: &[f32], num_frames: usize) -> Vec<f64> {
  assert_eq!(
    logits.len(),
    num_frames * POWERSET_CLASSES,
    "logits.len() must equal num_frames * POWERSET_CLASSES"
  );
  let mut out = Vec::with_capacity(num_frames * SEG_NUM_SLOTS);
  for row in logits.as_chunks::<POWERSET_CLASSES>().0 {
    out.extend_from_slice(&POWERSET_TABLE[hard_argmax(row)]);
  }
  out
}

#[cfg(test)]
mod tests;
