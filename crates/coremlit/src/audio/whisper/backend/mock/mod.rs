//! [`MockBackend`]: a scripted, deterministic [`crate::audio::whisper::backend::InferenceBackend`]
//! test double with no compiled model — kept in the public tree
//! deliberately (spec §5.4 `MockBackend`), not behind a test-only `cfg`, so
//! downstream crates get the same hermetic decode-loop/fallback/windowing
//! tests this crate does.

use std::sync::{Arc, Mutex};

use crate::audio::whisper::backend::{AlignmentView, BackendError, InferenceBackend, ModelDims};

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
/// cursor, the KV-advance record, the fixed alignment-weight buffer, the
/// row staged by the last step (`pending`), and the per-window
/// `hasAlignment` gate.
#[derive(Debug, Clone, PartialEq)]
pub struct MockDecoderState {
  step: usize,
  consumed: Vec<(u32, usize)>,
  alignment: Vec<f32>,
  /// `(position, row)` staged by the last [`InferenceBackend::decode_step`],
  /// awaiting a [`InferenceBackend::commit_alignment_row`]; `None` when that
  /// step scripted no row.
  pending: Option<(usize, Vec<f32>)>,
  window_has_alignment: bool,
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
  // 1-based decode_step call ordinals scripted to fail, and the running
  // attempt count they are checked against. Deliberately reset-IMMUNE
  // (unlike the script cursor, which reset rewinds): a "transient"
  // failure on the n-th call stays consumed across window resets, which
  // is what lets a replayed script fail on one window/attempt only.
  fail_calls: Vec<usize>,
  attempted: Arc<Mutex<usize>>,
  // When set, `reset_decoder_state` does NOT rewind the script cursor, so
  // consecutive windows/attempts consume consecutive script slices (see
  // `with_continuous_script`). Reset-immune scripting state, like
  // `fail_calls` above; the default (false) rewinds for fallback-replay.
  continuous_script: bool,
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
      fail_calls: Vec::new(),
      attempted: Arc::new(Mutex::new(0)),
      continuous_script: false,
    }
  }

  /// Puts this mock in continuous-script mode: [`InferenceBackend::reset_decoder_state`]
  /// no longer rewinds the script cursor, so consecutive windows (and
  /// fallback attempts) consume consecutive script slices instead of
  /// replaying the same one from the top. Reset-immune like
  /// [`Self::fail_on_call`]'s ordinal (see the `fail_calls` field) — the
  /// default rewind backs the fallback-replay tests; this backs multi-window
  /// fixtures where each window must decode a different token/alignment
  /// sequence (whisper #41's cross-window alignment staleness is observable
  /// only when a later, shorter window consults a row an earlier, longer
  /// window committed).
  #[must_use]
  #[inline(always)]
  pub fn with_continuous_script(mut self) -> Self {
    self.set_continuous_script(true);
    self
  }

  /// Sets [`Self::with_continuous_script`]'s cursor mode in place.
  #[inline(always)]
  pub fn set_continuous_script(&mut self, continuous: bool) -> &mut Self {
    self.continuous_script = continuous;
    self
  }

  /// Scripts the `call`-th (1-based, counted across resets)
  /// [`InferenceBackend::decode_step`] call to fail with
  /// [`BackendError::ScriptedFailure`] instead of consuming a script
  /// step. Because the ordinal survives [`InferenceBackend::reset_decoder_state`]'s
  /// script rewind, a replayed script can fail on exactly one
  /// window/attempt — e.g. a transiently failing language probe.
  pub fn fail_on_call(&mut self, call: usize) -> &mut Self {
    self.fail_calls.push(call);
    self
  }

  /// Builder form of [`Self::set_dims`].
  #[must_use]
  #[inline(always)]
  pub fn with_dims(mut self, dims: ModelDims) -> Self {
    self.set_dims(dims);
    self
  }

  /// Sets the dimensions [`InferenceBackend::dims`] reports, and that
  /// [`Self::push_step`]/[`Self::push_token_step`] validate scripted
  /// logits against.
  #[inline(always)]
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
  /// `alignment_row`, staged by that step and committed to row `position + 1`
  /// only when the decode loop calls [`InferenceBackend::commit_alignment_row`]
  /// (see that method and [`InferenceBackend::alignment_weights`]).
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
      // One row of headroom: `decode_step` at `position` commits its
      // alignment row at `position + 1`, so the last legal position
      // (`max_token_context - 1`) must still have a row to land in. Zeroed
      // here once per run — reset never re-clears it (Swift's once-allocated
      // tensor, TextDecoder.swift:141).
      alignment: vec![0.0; (self.dims.max_token_context() + 1) * self.dims.n_audio_ctx()],
      pending: None,
      window_has_alignment: false,
    })
  }

  fn reset_decoder_state(&self, state: &mut Self::DecoderState) {
    // Continuous-script mode leaves the cursor so consecutive windows
    // consume consecutive script slices (see `with_continuous_script`); the
    // default rewinds it for the fallback-replay tests.
    if !self.continuous_script {
      state.step = 0;
    }
    state.consumed.clear();
    // Alignment buffer deliberately NOT cleared — mirrors the real backend
    // and Swift's once-allocated tensor (Models.swift:312-322 resets only
    // cacheLength + masks). Only the per-window commit bookkeeping resets.
    state.window_has_alignment = false;
    state.pending = None;
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
    let call = {
      let mut attempted = self
        .attempted
        .lock()
        .expect("mock backend attempted-call lock poisoned");
      *attempted += 1;
      *attempted
    };
    if self.fail_calls.contains(&call) {
      return Err(BackendError::ScriptedFailure { call });
    }
    let Some(scripted) = self.script.get(state.step) else {
      return Err(BackendError::ScriptExhausted { step: state.step });
    };

    state.consumed.push((token, position));
    logits.clear();
    logits.extend_from_slice(&scripted.logits);

    if position == 0 {
      // A fresh window begins at position 0: Swift's per-window
      // `var hasAlignment = false` (:568), plus dropping any row the prior
      // window's completing step staged but never committed. Reset clears
      // both too; this keeps them honest on the dormant silent-window
      // `continue` that skips reset.
      state.window_has_alignment = false;
      state.pending = None;
    }
    // Stage this step's scripted row (if any) — committed into the buffer
    // only by a following `commit_alignment_row`, exactly like the real
    // backend (Swift updates alignment on non-completing steps only,
    // TextDecoder.swift:709-717).
    state.pending = scripted
      .alignment_row
      .as_ref()
      .map(|row| (position, row.clone()));
    state.step += 1;

    self
      .counters
      .lock()
      .expect("mock backend counters lock poisoned")
      .decode_steps += 1;
    Ok(())
  }

  fn commit_alignment_row(&self, state: &mut Self::DecoderState) {
    // Take the row staged by the preceding `decode_step` and write it at row
    // `position + 1` (Swift's non-completing-step slot,
    // TextDecoder.swift:709-717); no-op when nothing was staged.
    let Some((position, row)) = state.pending.take() else {
      return;
    };
    let cols = self.dims.n_audio_ctx();
    let start = (position + 1) * cols;
    state.alignment[start..start + cols].copy_from_slice(&row);
    state.window_has_alignment = true;
  }

  fn alignment_weights<'state>(
    &self,
    state: &'state Self::DecoderState,
  ) -> Option<AlignmentView<'state>> {
    // Same gate + full-buffer view as the real backend: `None` for a
    // zero-commit window (Swift's `hasAlignment ? tensor : nil`,
    // TextDecoder.swift:764-771), else the fixed `(max_ctx + 1) * cols`
    // accumulator with its stale/committed rows intact (whisper #41).
    state.window_has_alignment.then(|| {
      let cols = self.dims.n_audio_ctx();
      AlignmentView::new(&state.alignment, self.dims.max_token_context() + 1, cols)
    })
  }

  fn dims(&self) -> ModelDims {
    self.dims
  }
}
