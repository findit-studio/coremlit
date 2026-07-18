//! CoreML wrapper for the FluidInference unified Silero VAD graph
//! (`silero-vad-unified-256ms-v6.2.1.mlmodelc`) plus the 64-sample context
//! stitching it expects (design spec §4).
//!
//! # The graph, exactly as its `metadata.json` declares it
//!
//! One stateful `MLProgram`. Ground truth: `tests/model_io.rs`'s
//! `silero_vad_unified_io_matches_metadata` introspection test and the
//! artifact's own `metadata.json` (pinned there by SHA-256).
//!
//! | Direction | Feature | Shape | Dtype |
//! |---|---|---|---|
//! | in  | `audio_input`      | `[1, 4160]` | f32 |
//! | in  | `hidden_state`     | `[1, 128]`  | f32 |
//! | in  | `cell_state`       | `[1, 128]`  | f32 |
//! | out | `vad_output`       | `[1, 1, 1]` | f32 |
//! | out | `new_hidden_state` | `[1, 128]`  | f32 |
//! | out | `new_cell_state`   | `[1, 128]`  | f32 |
//!
//! `audio_input` is **4096 new samples + a 64-sample leading context** =
//! [`MODEL_INPUT_SAMPLES`]; the recurrent LSTM state is carried across chunks
//! as two explicit `[1, 128]` I/O pairs (the artifact declares an EMPTY
//! `stateSchema` — it is NOT a CoreML `MLState` model; the state is ordinary
//! feature I/O, exactly as FluidAudio's `VadManager` drives it). `vad_output`
//! is a single speech probability in `[0, 1]` (the graph's tail is a noisy-OR
//! of eight sigmoids: `1 − Π(1 − pᵢ)`), returned as a `[1, 1, 1]` tensor whose
//! sole element this wrapper reads.
//!
//! ## Deltas from the plan's expectation
//!
//! The plan (spec §4, from FluidAudio's `VadManager.swift:21-26`) expected
//! "4160 in, one probability out". The artifact confirms the 4160 input and
//! the probability, and additionally declares the explicit `[1, 128]` LSTM
//! state I/O (`hidden_state`/`cell_state` → `new_hidden_state`/
//! `new_cell_state`) that `VadManager` also drives; `vad_output` is a rank-3
//! `[1, 1, 1]` tensor, not a bare scalar. The v6.2.1 artifact ships **only**
//! as a compiled `.mlmodelc` (no `.mlpackage`), so — unlike alignkit — there
//! is no `coremlcompiler` step: [`crate::Model::load`] consumes it directly.
//!
//! # Context stitching (FluidAudio `VadManager` semantics)
//!
//! Ports `VadManager.processChunk` / `processUnifiedModel`
//! (`FluidAudio/Sources/FluidAudio/VAD/VadManager.swift:162-329`) exactly, so
//! the Swift trace oracle (`tests/parity_swift.rs`) reproduces bit-for-bit:
//!
//! - **Window assembly** (`assemble_window`): the input state's 64-sample
//!   `context` occupies `audio_input[0..64]`, the 4096 new samples occupy
//!   `audio_input[64..4160]` — `VadManager.swift:232-243` copies context at
//!   offset 0, chunk at offset `contextSize`.
//! - **First chunk**: [`VadState::initial`]'s context is 64 zeros, so the
//!   first window's leading 64 samples are zero — the zeroed pooled buffer
//!   `VadManager` starts from.
//! - **Carry-forward** (`next_context`): the next chunk's context is the
//!   **last 64 samples of the (padded) current chunk** —
//!   `VadManager.swift:184`'s `processedChunk.suffix(contextSize)`. The
//!   context is NOT a sliding overlap of the raw stream; it is exactly the
//!   previous chunk's tail, which is why a one-sample skew is detectable
//!   (`tests/model_state.rs`'s `misaligned_context_changes_the_probability`).
//! - **Short final chunk** (`prepare_chunk`): padded to 4096 by repeating
//!   the LAST sample, not with zeros — `VadManager.swift:174-178` ("repeat-
//!   last padding instead of zeros to avoid energy distortion"). An empty
//!   chunk pads with `0.0` (`chunk.last ?? 0.0`). A chunk LONGER than 4096 is
//!   rejected ([`InferError::ChunkTooLong`]) rather than truncated: the
//!   streaming caller (silero's detector, T5) and the trace harness both feed
//!   at most one 256 ms window per call, so an over-long chunk is a caller
//!   bug whose silently-dropped tail would be lost speech, not a case to
//!   paper over. (`VadManager` truncates it, but never reaches that branch.)
//! - **No normalization**: samples feed the graph at their original
//!   amplitude (`VadManager.swift:185`).

use std::path::Path;

use crate::{ComputeUnits, DataType, FeatureInfo, Model, MultiArray};

use crate::audio::vad::error::{InferError, ModelError};

/// New audio samples consumed per VAD chunk — 256 ms at 16 kHz. Matches
/// FluidAudio's `VadManager.chunkSize` (`VadManager.swift:22`) and the
/// `audio_input` window minus its [`CONTEXT_SAMPLES`] leading context.
pub const CHUNK_SAMPLES: usize = 4096;

/// Leading context samples prepended to each chunk — the previous chunk's
/// tail. Matches FluidAudio's `VadState.contextLength` (`VadTypes.swift:94`)
/// / `VadManager.contextSize` (`VadManager.swift:23`).
pub const CONTEXT_SAMPLES: usize = 64;

/// Total `audio_input` length: [`CONTEXT_SAMPLES`] + [`CHUNK_SAMPLES`] = 4160.
/// Matches FluidAudio's `VadManager.modelInputSize` (`VadManager.swift:25`)
/// and the artifact's declared `audio_input` shape `[1, 4160]`.
pub const MODEL_INPUT_SAMPLES: usize = CONTEXT_SAMPLES + CHUNK_SAMPLES;

/// Length of each recurrent-state vector (`hidden_state`/`cell_state` and
/// their `new_*` outputs). Matches FluidAudio's `VadManager.stateSize`
/// (`VadManager.swift:24`) and the artifact's declared `[1, 128]` state
/// shapes.
pub const STATE_SIZE: usize = 128;

/// Declared feature names on the artifact (pinned by
/// `tests/model_io.rs::silero_vad_unified_io_matches_metadata`).
mod names {
  pub const AUDIO_INPUT: &str = "audio_input";
  pub const HIDDEN_STATE: &str = "hidden_state";
  pub const CELL_STATE: &str = "cell_state";
  pub const VAD_OUTPUT: &str = "vad_output";
  pub const NEW_HIDDEN_STATE: &str = "new_hidden_state";
  pub const NEW_CELL_STATE: &str = "new_cell_state";
}

/// Default [`VadModelOptions::compute`]. [`ComputeUnits::All`] lets CoreML
/// schedule across ANE/GPU/CPU — the production default. Model-gated tests
/// instead load with [`ComputeUnits::CpuOnly`] for determinism and to match
/// the Swift trace oracle's placement (`tests/parity_swift.rs`), exactly as
/// the sibling kits do.
pub const DEFAULT_VAD_COMPUTE: ComputeUnits = ComputeUnits::All;

/// Human-readable `shape dtype` rendering for
/// [`ModelError::ContractMismatch`]'s `actual`/`expected` fields (mirrors
/// `speakerkit::segment::describe`).
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// Construction options for [`VadModel`] (rust-options-pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadModelOptions {
  compute: ComputeUnits,
}

impl Default for VadModelOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl VadModelOptions {
  /// Options matching the crate's default: [`DEFAULT_VAD_COMPUTE`]
  /// ([`ComputeUnits::All`]).
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_VAD_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the VAD model on.
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

/// The VAD graph's recurrent state carried between chunks: the LSTM hidden
/// and cell vectors plus the 64-sample audio context (the previous chunk's
/// tail). Mirrors FluidAudio's `VadState` (`VadTypes.swift:93-116`), which
/// likewise bundles all three so a single value fully determines the next
/// chunk's output.
///
/// Not `Eq` (the arrays are `f32`); `PartialEq` compares element-for-element,
/// which is exact for the bit-identical state a deterministic re-run produces
/// (`tests/model_io.rs`'s `state_round_trips_across_chunks`).
#[derive(Debug, Clone, PartialEq)]
pub struct VadState {
  hidden: [f32; STATE_SIZE],
  cell: [f32; STATE_SIZE],
  context: [f32; CONTEXT_SAMPLES],
}

impl VadState {
  /// The zero state for the first chunk: 128 zero hidden units, 128 zero cell
  /// units, and 64 zero context samples. Matches FluidAudio's
  /// `VadState.initial()` (`VadTypes.swift:109-115`).
  pub const fn initial() -> Self {
    Self {
      hidden: [0.0; STATE_SIZE],
      cell: [0.0; STATE_SIZE],
      context: [0.0; CONTEXT_SAMPLES],
    }
  }

  /// Reconstructs a state from its parts — the restore half of streaming
  /// state persistence (the save half is [`Self::hidden`]/[`Self::cell`]/
  /// [`Self::context`]). Also how a test builds a deliberately misaligned
  /// context to prove the context is load-bearing
  /// (`tests/model_io.rs`'s `misaligned_context_changes_the_probability`).
  pub const fn from_parts(
    hidden: [f32; STATE_SIZE],
    cell: [f32; STATE_SIZE],
    context: [f32; CONTEXT_SAMPLES],
  ) -> Self {
    Self {
      hidden,
      cell,
      context,
    }
  }

  /// The LSTM hidden state (`hidden_state` input / `new_hidden_state` output).
  #[inline(always)]
  pub const fn hidden(&self) -> &[f32; STATE_SIZE] {
    &self.hidden
  }

  /// The LSTM cell state (`cell_state` input / `new_cell_state` output).
  #[inline(always)]
  pub const fn cell(&self) -> &[f32; STATE_SIZE] {
    &self.cell
  }

  /// The 64-sample audio context (the previous chunk's tail) prepended to the
  /// next chunk's `audio_input`.
  #[inline(always)]
  pub const fn context(&self) -> &[f32; CONTEXT_SAMPLES] {
    &self.context
  }
}

impl Default for VadState {
  fn default() -> Self {
    Self::initial()
  }
}

/// CoreML wrapper over the unified Silero VAD graph, carrying the recurrent
/// [`VadState`] between chunks. One [`CHUNK_SAMPLES`]-sample chunk in, one
/// speech probability out, state advanced in place — the shape the silero
/// backend seam's `predict`/`reset` expects (spec §3; the trait impl is T5).
#[derive(Debug)]
pub struct VadModel {
  model: Model,
  state: VadState,
}

impl VadModel {
  /// Loads the model with [`VadModelOptions::new`] ([`ComputeUnits::All`]),
  /// starting from [`VadState::initial`].
  ///
  /// # Errors
  /// As [`Self::load_with`].
  pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
    Self::load_with(path, VadModelOptions::new())
  }

  /// Loads the model with custom options, introspecting and validating its
  /// full I/O contract against the ground truth pinned by
  /// `tests/model_io.rs::silero_vad_unified_io_matches_metadata`.
  ///
  /// # Errors
  /// [`ModelError::Load`] if CoreML rejects the model.
  /// [`ModelError::ContractMismatch`] if any of the six declared features is
  /// missing or does not match its pinned shape/dtype: `audio_input`
  /// `[1, 4160]` f32, `hidden_state`/`cell_state`/`new_hidden_state`/
  /// `new_cell_state` `[1, 128]` f32, `vad_output` `[1, 1, 1]` f32.
  pub fn load_with(path: impl AsRef<Path>, options: VadModelOptions) -> Result<Self, ModelError> {
    let model = Model::load(path, options.compute())?;
    let description = model.description();

    check_feature(
      description.input(names::AUDIO_INPUT),
      names::AUDIO_INPUT,
      &[1, MODEL_INPUT_SAMPLES],
    )?;
    check_feature(
      description.input(names::HIDDEN_STATE),
      names::HIDDEN_STATE,
      &[1, STATE_SIZE],
    )?;
    check_feature(
      description.input(names::CELL_STATE),
      names::CELL_STATE,
      &[1, STATE_SIZE],
    )?;
    check_feature(
      description.output(names::VAD_OUTPUT),
      names::VAD_OUTPUT,
      &[1, 1, 1],
    )?;
    check_feature(
      description.output(names::NEW_HIDDEN_STATE),
      names::NEW_HIDDEN_STATE,
      &[1, STATE_SIZE],
    )?;
    check_feature(
      description.output(names::NEW_CELL_STATE),
      names::NEW_CELL_STATE,
      &[1, STATE_SIZE],
    )?;

    Ok(Self {
      model,
      state: VadState::initial(),
    })
  }

  /// The current recurrent state (advanced by [`Self::predict_chunk`], cleared
  /// by [`Self::reset`]).
  #[inline(always)]
  pub const fn state(&self) -> &VadState {
    &self.state
  }

  /// Clears the recurrent state back to [`VadState::initial`] — the silero
  /// seam's `reset` (spec §3). The next [`Self::predict_chunk`] then behaves
  /// as if it were the first chunk of a fresh stream.
  pub fn reset(&mut self) {
    self.state = VadState::initial();
  }

  /// Runs one chunk of up to [`CHUNK_SAMPLES`] new samples, advancing the
  /// internal [`VadState`] in place and returning the speech probability in
  /// `[0, 1]` — the silero seam's `predict` shape (spec §3).
  ///
  /// Context stitching, short-chunk padding, and the state carry-forward
  /// follow FluidAudio's `VadManager` exactly (see the module doc).
  ///
  /// # Errors
  /// As [`Self::predict_chunk_with_state`].
  pub fn predict_chunk(&mut self, chunk: &[f32]) -> Result<f32, InferError> {
    let (probability, next) = self.predict_chunk_with_state(chunk, &self.state)?;
    self.state = next;
    Ok(probability)
  }

  /// Runs one chunk from an EXPLICIT input state, returning the speech
  /// probability and the resulting output state, without touching the
  /// internal state. The functional core [`Self::predict_chunk`] delegates to;
  /// exposed so streaming state can be saved and restored (and so the state
  /// round-trip / misaligned-context gates can drive the model deterministically).
  ///
  /// # Errors
  /// [`InferError::ChunkTooLong`] if `chunk.len() > CHUNK_SAMPLES` (see the
  /// module doc). [`InferError::NonFiniteInput`] if the assembled window
  /// (context + chunk) holds a NaN or infinite sample, scanned before the
  /// CoreML call. [`InferError::Prediction`]/[`InferError::Tensor`] on a
  /// CoreML or tensor-construction failure — including a prediction whose
  /// output set omits a declared feature
  /// ([`crate::PredictionError::MissingOutput`]).
  /// [`InferError::OutputShape`] if any predict-time output tensor's shape
  /// diverges from its construction-time contract (the CoreML runtime is a
  /// trust boundary re-checked every call).
  /// [`InferError::NonFiniteOutput`] if the probability or any recurrent-state
  /// element comes back non-finite.
  pub fn predict_chunk_with_state(
    &self,
    chunk: &[f32],
    state: &VadState,
  ) -> Result<(f32, VadState), InferError> {
    let padded = prepare_chunk(chunk)?;
    let window = assemble_window(&state.context, &padded);
    check_finite_input(&window)?;

    let audio = MultiArray::from_slice(&[1, MODEL_INPUT_SAMPLES], &window)?;
    let hidden = MultiArray::from_slice(&[1, STATE_SIZE], &state.hidden)?;
    let cell = MultiArray::from_slice(&[1, STATE_SIZE], &state.cell)?;

    let mut outputs = self.model.predict_with(&[
      (names::AUDIO_INPUT, &audio),
      (names::HIDDEN_STATE, &hidden),
      (names::CELL_STATE, &cell),
    ])?;

    let probability = take_scalar(&mut outputs, names::VAD_OUTPUT)?;
    let next_hidden = take_state(&mut outputs, names::NEW_HIDDEN_STATE)?;
    let next_cell = take_state(&mut outputs, names::NEW_CELL_STATE)?;

    Ok((
      probability,
      VadState {
        hidden: next_hidden,
        cell: next_cell,
        context: next_context(&padded),
      },
    ))
  }
}

/// Validates one loaded [`FeatureInfo`] against its pinned shape (all six VAD
/// features are f32), yielding [`ModelError::ContractMismatch`] on a missing
/// feature, a shape mismatch, or a non-f32 dtype. Hermetically testable
/// without a loaded model.
fn check_feature(
  feature: Option<&FeatureInfo>,
  name: &'static str,
  expected_shape: &[usize],
) -> Result<(), ModelError> {
  let expected = describe(expected_shape, Some(DataType::F32));
  let Some(feature) = feature else {
    return Err(ModelError::ContractMismatch {
      feature: name,
      expected,
      actual: "missing".to_string(),
    });
  };
  if feature.shape() != expected_shape || feature.data_type() != Some(DataType::F32) {
    return Err(ModelError::ContractMismatch {
      feature: name,
      expected,
      actual: describe(feature.shape(), feature.data_type()),
    });
  }
  Ok(())
}

/// Pads/validates a caller chunk into exactly [`CHUNK_SAMPLES`] samples,
/// FluidAudio `VadManager.processChunk` semantics
/// (`VadManager.swift:172-182`): exact length passes through; a short chunk is
/// padded by repeating its LAST sample (`0.0` if empty); an over-long chunk is
/// rejected (see the module doc for why this rejects where `VadManager`
/// truncates). Hermetically testable without a loaded model.
fn prepare_chunk(chunk: &[f32]) -> Result<[f32; CHUNK_SAMPLES], InferError> {
  if chunk.len() > CHUNK_SAMPLES {
    return Err(InferError::ChunkTooLong {
      got: chunk.len(),
      max: CHUNK_SAMPLES,
    });
  }
  let mut padded = [0.0f32; CHUNK_SAMPLES];
  padded[..chunk.len()].copy_from_slice(chunk);
  // Repeat-last padding (not zeros) — the last real sample, or 0.0 for an
  // empty chunk (`chunk.last ?? 0.0`).
  let last = chunk.last().copied().unwrap_or(0.0);
  for slot in &mut padded[chunk.len()..] {
    *slot = last;
  }
  Ok(padded)
}

/// Assembles the `audio_input` window: `context` at `[0..CONTEXT_SAMPLES]`,
/// the padded chunk at `[CONTEXT_SAMPLES..MODEL_INPUT_SAMPLES]` — FluidAudio's
/// copy order (`VadManager.swift:235-243`). Hermetically testable.
fn assemble_window(
  context: &[f32; CONTEXT_SAMPLES],
  chunk: &[f32; CHUNK_SAMPLES],
) -> [f32; MODEL_INPUT_SAMPLES] {
  let mut window = [0.0f32; MODEL_INPUT_SAMPLES];
  window[..CONTEXT_SAMPLES].copy_from_slice(context);
  window[CONTEXT_SAMPLES..].copy_from_slice(chunk);
  window
}

/// The context carried into the next chunk: the last [`CONTEXT_SAMPLES`]
/// samples of the (padded) current chunk — FluidAudio's
/// `processedChunk.suffix(contextSize)` (`VadManager.swift:184`). Hermetically
/// testable.
fn next_context(chunk: &[f32; CHUNK_SAMPLES]) -> [f32; CONTEXT_SAMPLES] {
  let mut context = [0.0f32; CONTEXT_SAMPLES];
  context.copy_from_slice(&chunk[CHUNK_SAMPLES - CONTEXT_SAMPLES..]);
  context
}

/// Scans the assembled model window for the first non-finite sample, BEFORE
/// the CoreML call — a NaN would otherwise reach CoreML and can be absorbed
/// into finite garbage no output scan catches (see
/// [`InferError::NonFiniteInput`]). Hermetically testable.
fn check_finite_input(window: &[f32]) -> Result<(), InferError> {
  if let Some(index) = window.iter().position(|v| !v.is_finite()) {
    return Err(InferError::NonFiniteInput { index });
  }
  Ok(())
}

/// Extracts the single `[1, 1, 1]` `vad_output` probability, re-validating its
/// runtime shape (the CoreML runtime is a trust boundary — `copy_into` alone
/// only checks element count) and rejecting a non-finite value.
fn take_scalar(outputs: &mut crate::Features, name: &'static str) -> Result<f32, InferError> {
  let tensor = outputs
    .take(name)
    .ok_or_else(|| crate::PredictionError::MissingOutput {
      name: name.to_string(),
    })?;
  check_output_shape(tensor.shape(), name, &[1, 1, 1])?;
  let mut buf = [0.0f32; 1];
  tensor.copy_into::<f32>(&mut buf)?;
  if !buf[0].is_finite() {
    return Err(InferError::NonFiniteOutput {
      feature: name,
      index: 0,
    });
  }
  Ok(buf[0])
}

/// Extracts a `[1, 128]` recurrent-state output, re-validating its runtime
/// shape and rejecting any non-finite element.
fn take_state(
  outputs: &mut crate::Features,
  name: &'static str,
) -> Result<[f32; STATE_SIZE], InferError> {
  let tensor = outputs
    .take(name)
    .ok_or_else(|| crate::PredictionError::MissingOutput {
      name: name.to_string(),
    })?;
  check_output_shape(tensor.shape(), name, &[1, STATE_SIZE])?;
  let mut buf = [0.0f32; STATE_SIZE];
  tensor.copy_into::<f32>(&mut buf)?;
  if let Some(index) = buf.iter().position(|v| !v.is_finite()) {
    return Err(InferError::NonFiniteOutput {
      feature: name,
      index,
    });
  }
  Ok(buf)
}

/// Validates a predict-time output tensor's shape against its construction-time
/// contract. Hermetically testable without a loaded model.
fn check_output_shape(
  shape: &[usize],
  feature: &'static str,
  expected: &[usize],
) -> Result<(), InferError> {
  if shape != expected {
    return Err(InferError::OutputShape {
      feature,
      got: shape.to_vec(),
      expected: expected.to_vec(),
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests;
