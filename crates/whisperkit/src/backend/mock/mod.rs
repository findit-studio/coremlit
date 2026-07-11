//! [`MockBackend`]: a scripted, deterministic [`crate::backend::InferenceBackend`]
//! test double with no compiled model — kept in the public tree
//! deliberately (spec §5.4 `MockBackend`), not behind a test-only `cfg`, so
//! downstream crates get the same hermetic decode-loop/fallback/windowing
//! tests this crate does.

use std::sync::{Arc, Mutex};

use crate::backend::{AlignmentView, BackendError, InferenceBackend, ModelDims};

#[cfg(test)]
mod tests;

/// One scripted decode step: full-vocabulary logits, plus an optional
/// alignment row (mirrors the model's `alignment_heads_weights` output) for
/// steps that exercise word-timestamp support.
#[derive(Debug, Clone, PartialEq)]
struct ScriptedStep {
  logits: Vec<f32>,
  alignment_row: Option<Vec<f32>>,
}

/// Snapshot of how many times each [`InferenceBackend`] method has been
/// called on a [`MockBackend`] — lets a test assert the decode loop's shape
/// (step count, reset count, encode/extract call count) after a pipeline
/// ran against it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MockCounters {
  extract_calls: usize,
  encode_calls: usize,
  decode_steps: usize,
  resets: usize,
}

impl MockCounters {
  /// Number of [`InferenceBackend::extract_features`] calls.
  #[inline(always)]
  pub const fn extract_calls(&self) -> usize {
    self.extract_calls
  }

  /// Number of [`InferenceBackend::encode`] calls.
  #[inline(always)]
  pub const fn encode_calls(&self) -> usize {
    self.encode_calls
  }

  /// Number of successful [`InferenceBackend::decode_step`] calls.
  #[inline(always)]
  pub const fn decode_steps(&self) -> usize {
    self.decode_steps
  }

  /// Number of [`InferenceBackend::reset_decoder_state`] calls.
  #[inline(always)]
  pub const fn resets(&self) -> usize {
    self.resets
  }
}

/// Pre-allocated per-window decoder state for [`MockBackend`] — a Swift
/// `DecodingInputs` analogue scaled down to what the mock needs: a script
/// cursor, the KV-advance record, and the alignment-weight scratch buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct MockDecoderState {
  step: usize,
  consumed: Vec<(u32, usize)>,
  alignment: Vec<f32>,
  written_rows: usize,
}

impl MockDecoderState {
  /// Every `(token, position)` pair consumed by
  /// [`InferenceBackend::decode_step`] so far, in call order — the
  /// KV-advance record a test inspects instead of a real KV cache.
  #[inline(always)]
  pub fn consumed_slice(&self) -> &[(u32, usize)] {
    self.consumed.as_slice()
  }
}

/// Deterministic, scripted [`InferenceBackend`] test double (spec §5.4):
/// `extract_features`/`encode` just move bytes and count calls;
/// `decode_step` replays a pre-scripted sequence of logits (and,
/// optionally, alignment rows) in order. Backs every hermetic
/// decode-loop, fallback-ladder, windowing, and early-stop test this port
/// needs before a compiled model exists at all (spec §9.1).
#[derive(Debug)]
pub struct MockBackend {
  dims: ModelDims,
  script: Vec<ScriptedStep>,
  counters: Arc<Mutex<MockCounters>>,
}

impl Default for MockBackend {
  fn default() -> Self {
    Self::new()
  }
}

impl MockBackend {
  /// A mock backend with the tiny model's dimensions ([`ModelDims::new`])
  /// and no scripted steps.
  pub fn new() -> Self {
    Self {
      dims: ModelDims::new(),
      script: Vec::new(),
      counters: Arc::new(Mutex::new(MockCounters::default())),
    }
  }

  /// Builder form of [`Self::set_dims`].
  #[must_use]
  pub fn with_dims(mut self, dims: ModelDims) -> Self {
    self.set_dims(dims);
    self
  }

  /// Sets the dimensions [`InferenceBackend::dims`] reports, and that
  /// [`Self::push_step`]/[`Self::push_token_step`] validate scripted
  /// logits against.
  pub fn set_dims(&mut self, dims: ModelDims) -> &mut Self {
    self.dims = dims;
    self
  }

  /// Appends one scripted step with explicit `logits`.
  ///
  /// # Panics
  /// If `logits.len() != self.dims().vocab()` — a test-authoring error,
  /// not a runtime [`BackendError`].
  pub fn push_step(&mut self, logits: Vec<f32>) -> &mut Self {
    assert_eq!(
      logits.len(),
      self.dims.vocab(),
      "scripted logits len {} != dims.vocab() {}",
      logits.len(),
      self.dims.vocab()
    );
    self.script.push(ScriptedStep {
      logits,
      alignment_row: None,
    });
    self
  }

  /// Appends one scripted step whose logits are one-hot on `token` (`10.0`
  /// at `token`'s index, `0.0` elsewhere) — scripts "the model picks this
  /// exact token next" without hand-building a full logits vector.
  ///
  /// # Panics
  /// If `token as usize >= self.dims().vocab()`.
  pub fn push_token_step(&mut self, token: u32) -> &mut Self {
    let mut logits = vec![0.0_f32; self.dims.vocab()];
    logits[token as usize] = 10.0;
    self.push_step(logits)
  }

  /// Appends one scripted, one-hot step (see [`Self::push_token_step`])
  /// per entry of `tokens`, in order.
  pub fn push_token_steps(&mut self, tokens: &[u32]) -> &mut Self {
    for &token in tokens {
      self.push_token_step(token);
    }
    self
  }

  /// Appends one scripted step with explicit `logits` and an
  /// `alignment_row`, written at row `position + 1` when this step runs
  /// (see [`InferenceBackend::alignment_weights`]).
  ///
  /// # Panics
  /// If `logits.len() != self.dims().vocab()`.
  pub fn push_step_with_alignment(
    &mut self,
    logits: Vec<f32>,
    alignment_row: Vec<f32>,
  ) -> &mut Self {
    self.push_step(logits);
    self
      .script
      .last_mut()
      .expect("push_step just appended one entry")
      .alignment_row = Some(alignment_row);
    self
  }

  /// Snapshot of how many times each [`InferenceBackend`] method has been
  /// called so far.
  pub fn counters(&self) -> MockCounters {
    *self
      .counters
      .lock()
      .expect("mock backend counters lock poisoned")
  }
}

impl InferenceBackend for MockBackend {
  type Features = Vec<f32>;
  type EncoderOutput = Vec<f32>;
  type DecoderState = MockDecoderState;

  fn extract_features(&self, audio: &[f32]) -> Result<Self::Features, BackendError> {
    self
      .counters
      .lock()
      .expect("mock backend counters lock poisoned")
      .extract_calls += 1;
    Ok(audio.to_vec())
  }

  fn encode(&self, features: &Self::Features) -> Result<Self::EncoderOutput, BackendError> {
    self
      .counters
      .lock()
      .expect("mock backend counters lock poisoned")
      .encode_calls += 1;
    Ok(features.clone())
  }

  fn new_decoder_state(&self) -> Result<Self::DecoderState, BackendError> {
    Ok(MockDecoderState {
      step: 0,
      consumed: Vec::new(),
      // One row of headroom: `decode_step` at `position` writes its
      // alignment row at `position + 1`, so the last legal position
      // (`max_token_context - 1`) must still have a row to land in.
      alignment: vec![0.0; (self.dims.max_token_context() + 1) * self.dims.n_audio_ctx()],
      written_rows: 0,
    })
  }

  fn reset_decoder_state(&self, state: &mut Self::DecoderState) {
    state.step = 0;
    state.consumed.clear();
    state.alignment.fill(0.0);
    state.written_rows = 0;
    self
      .counters
      .lock()
      .expect("mock backend counters lock poisoned")
      .resets += 1;
  }

  fn decode_step(
    &self,
    token: u32,
    position: usize,
    _encoder_output: &Self::EncoderOutput,
    state: &mut Self::DecoderState,
    logits: &mut Vec<f32>,
  ) -> Result<(), BackendError> {
    let Some(scripted) = self.script.get(state.step) else {
      return Err(BackendError::ScriptExhausted { step: state.step });
    };

    state.consumed.push((token, position));
    logits.clear();
    logits.extend_from_slice(&scripted.logits);

    if let Some(row) = &scripted.alignment_row {
      let cols = self.dims.n_audio_ctx();
      let start = (position + 1) * cols;
      state.alignment[start..start + cols].copy_from_slice(row);
      // Gated exactly like the real backend (and Swift's
      // `if let ... = cache?.alignmentWeights`): steps without an
      // alignment row do not extend the view.
      state.written_rows = state.written_rows.max(position + 2);
    }
    state.step += 1;

    self
      .counters
      .lock()
      .expect("mock backend counters lock poisoned")
      .decode_steps += 1;
    Ok(())
  }

  fn alignment_weights<'state>(
    &self,
    state: &'state Self::DecoderState,
  ) -> Option<AlignmentView<'state>> {
    let cols = self.dims.n_audio_ctx();
    let rows = state.written_rows;
    Some(AlignmentView::new(
      &state.alignment[..rows * cols],
      rows,
      cols,
    ))
  }

  fn dims(&self) -> ModelDims {
    self.dims
  }
}
