//! [`InferenceBackend`]: the seam every Whisper pipeline stage (mel
//! extraction, encoding, autoregressive decoding) drives without knowing
//! whether it is talking to a real CoreML model or a scripted test double
//! (spec §5.4). Also home to [`ModelDims`] (the static shape/vocabulary
//! description backends report — drives variant detection and buffer
//! sizing) and [`AlignmentView`] (the borrowed cross-attention alignment
//! slice word-timestamp code reads).
//!
//! Two implementations live here: [`coreml::CoreMlBackend`] — the real
//! one, owning the three `coremlit::Model`s (spec §5.4) — and
//! [`mock::MockBackend`], the scripted, hermetic test double every
//! decode-loop/fallback/windowing test downstream needs before a compiled
//! model exists at all (spec §9.1).

use crate::model::is_model_multilingual;

pub mod coreml;
pub mod mock;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// ModelDims
// ---------------------------------------------------------------------

/// Default [`ModelDims::vocab`] — tiny model decoder vocabulary size (Task
/// 1 ground truth: `logits` output shape `[1, 1, 51865]`).
pub const DEFAULT_VOCAB: usize = 51_865;
/// Default [`ModelDims::n_mels`] — tiny model mel-spectrogram bin count
/// (Task 1 ground truth: `melspectrogram_features` shape `[1, 80, 1,
/// 3000]`).
pub const DEFAULT_N_MELS: usize = 80;
/// Default [`ModelDims::embed_dim`] — tiny model encoder embedding width
/// (Task 1 ground truth: `encoder_output_embeds` shape `[1, 384, 1,
/// 1500]`).
pub const DEFAULT_EMBED_DIM: usize = 384;
/// Default [`ModelDims::kv_dim`] — tiny model decoder KV-cache channel
/// width, `embed_dim * decoder_layers` (Task 1 ground truth:
/// `key_cache`/`value_cache` shape `[1, 1536, 1, 224]`).
pub const DEFAULT_KV_DIM: usize = 1536;
/// Default [`ModelDims::max_token_context`] — shared by every model size
/// (Whisper's fixed `448 / 2` token budget, not tiny-specific), so this
/// reuses [`crate::constants::MAX_TOKEN_CONTEXT`] rather than restating it.
pub const DEFAULT_MAX_TOKEN_CONTEXT: usize = crate::constants::MAX_TOKEN_CONTEXT;
/// Default [`ModelDims::n_audio_ctx`] — tiny model encoder audio-context
/// length (Task 1 ground truth: `encoder_output_embeds`/
/// `alignment_heads_weights` trailing dim `1500`).
pub const DEFAULT_N_AUDIO_CTX: usize = 1500;
/// Default [`ModelDims::window_samples`] — shared by every model size
/// (fixed 30 s @ 16 kHz, not tiny-specific), so this reuses
/// [`crate::constants::WINDOW_SAMPLES`] rather than restating it.
pub const DEFAULT_WINDOW_SAMPLES: usize = crate::constants::WINDOW_SAMPLES;

/// Static shape/vocabulary description of a loaded Whisper model —
/// [`InferenceBackend::dims`] reports these for variant detection, buffer
/// sizing, and multilinguality (spec §5.4: "vocab, n_mels, n_ctx, embed —
/// drives variant detection").
///
/// [`Self::new`]/[`Default`] return the tiny model's dimensions
/// (introspected live from the real `openai_whisper-tiny` `.mlmodelc`
/// bundles and pinned by `tests/model_io.rs`) — every unit/doc test
/// in this crate implicitly targets that model. A real backend overrides
/// every field from its own model's introspection at load time; these
/// defaults exist for the tiny model and the mock/fallback case only, not
/// as a general-purpose guess for an arbitrary model size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
  vocab: usize,
  n_mels: usize,
  embed_dim: usize,
  kv_dim: usize,
  max_token_context: usize,
  n_audio_ctx: usize,
  window_samples: usize,
}

impl Default for ModelDims {
  fn default() -> Self {
    Self::new()
  }
}

impl ModelDims {
  /// Dimensions matching the tiny model (Task 1 ground truth); backends
  /// override every field from real introspection at load time.
  pub const fn new() -> Self {
    Self {
      vocab: DEFAULT_VOCAB,
      n_mels: DEFAULT_N_MELS,
      embed_dim: DEFAULT_EMBED_DIM,
      kv_dim: DEFAULT_KV_DIM,
      max_token_context: DEFAULT_MAX_TOKEN_CONTEXT,
      n_audio_ctx: DEFAULT_N_AUDIO_CTX,
      window_samples: DEFAULT_WINDOW_SAMPLES,
    }
  }

  // -- vocab ----------------------------------------------------------
  /// Decoder output vocabulary size.
  #[inline(always)]
  pub const fn vocab(&self) -> usize {
    self.vocab
  }
  /// Builder form of [`Self::set_vocab`].
  #[must_use]
  #[inline(always)]
  pub const fn with_vocab(mut self, vocab: usize) -> Self {
    self.set_vocab(vocab);
    self
  }
  /// Sets [`Self::vocab`] in place.
  #[inline(always)]
  pub const fn set_vocab(&mut self, vocab: usize) -> &mut Self {
    self.vocab = vocab;
    self
  }

  // -- n_mels -----------------------------------------------------------
  /// Mel-spectrogram feature bins the mel stage produces.
  #[inline(always)]
  pub const fn n_mels(&self) -> usize {
    self.n_mels
  }
  /// Builder form of [`Self::set_n_mels`].
  #[must_use]
  #[inline(always)]
  pub const fn with_n_mels(mut self, n_mels: usize) -> Self {
    self.set_n_mels(n_mels);
    self
  }
  /// Sets [`Self::n_mels`] in place.
  #[inline(always)]
  pub const fn set_n_mels(&mut self, n_mels: usize) -> &mut Self {
    self.n_mels = n_mels;
    self
  }

  // -- embed_dim ----------------------------------------------------------
  /// Encoder output embedding width.
  #[inline(always)]
  pub const fn embed_dim(&self) -> usize {
    self.embed_dim
  }
  /// Builder form of [`Self::set_embed_dim`].
  #[must_use]
  #[inline(always)]
  pub const fn with_embed_dim(mut self, embed_dim: usize) -> Self {
    self.set_embed_dim(embed_dim);
    self
  }
  /// Sets [`Self::embed_dim`] in place.
  #[inline(always)]
  pub const fn set_embed_dim(&mut self, embed_dim: usize) -> &mut Self {
    self.embed_dim = embed_dim;
    self
  }

  // -- kv_dim -----------------------------------------------------------
  /// Decoder KV-cache channel width (`embed_dim * decoder_layers`).
  #[inline(always)]
  pub const fn kv_dim(&self) -> usize {
    self.kv_dim
  }
  /// Builder form of [`Self::set_kv_dim`].
  #[must_use]
  #[inline(always)]
  pub const fn with_kv_dim(mut self, kv_dim: usize) -> Self {
    self.set_kv_dim(kv_dim);
    self
  }
  /// Sets [`Self::kv_dim`] in place.
  #[inline(always)]
  pub const fn set_kv_dim(&mut self, kv_dim: usize) -> &mut Self {
    self.kv_dim = kv_dim;
    self
  }

  // -- max_token_context ------------------------------------------------
  /// Maximum decoder token context.
  #[inline(always)]
  pub const fn max_token_context(&self) -> usize {
    self.max_token_context
  }
  /// Builder form of [`Self::set_max_token_context`].
  #[must_use]
  #[inline(always)]
  pub const fn with_max_token_context(mut self, max_token_context: usize) -> Self {
    self.set_max_token_context(max_token_context);
    self
  }
  /// Sets [`Self::max_token_context`] in place.
  #[inline(always)]
  pub const fn set_max_token_context(&mut self, max_token_context: usize) -> &mut Self {
    self.max_token_context = max_token_context;
    self
  }

  // -- n_audio_ctx --------------------------------------------------------
  /// Encoder audio-context length (time steps in `encoder_output_embeds`).
  #[inline(always)]
  pub const fn n_audio_ctx(&self) -> usize {
    self.n_audio_ctx
  }
  /// Builder form of [`Self::set_n_audio_ctx`].
  #[must_use]
  #[inline(always)]
  pub const fn with_n_audio_ctx(mut self, n_audio_ctx: usize) -> Self {
    self.set_n_audio_ctx(n_audio_ctx);
    self
  }
  /// Sets [`Self::n_audio_ctx`] in place.
  #[inline(always)]
  pub const fn set_n_audio_ctx(&mut self, n_audio_ctx: usize) -> &mut Self {
    self.n_audio_ctx = n_audio_ctx;
    self
  }

  // -- window_samples -----------------------------------------------------
  /// Samples per encoder window.
  #[inline(always)]
  pub const fn window_samples(&self) -> usize {
    self.window_samples
  }
  /// Builder form of [`Self::set_window_samples`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window_samples(mut self, window_samples: usize) -> Self {
    self.set_window_samples(window_samples);
    self
  }
  /// Sets [`Self::window_samples`] in place.
  #[inline(always)]
  pub const fn set_window_samples(&mut self, window_samples: usize) -> &mut Self {
    self.window_samples = window_samples;
    self
  }

  /// Whether this model understands languages other than English — every
  /// vocabulary size except English-only's `51864` (Swift
  /// `ModelUtilities.isModelMultilingual(logitsDim:)`,
  /// `WhisperKit/Utilities/ModelUtilities.swift:124-126`). Delegates to
  /// [`is_model_multilingual`] — the same check
  /// [`crate::model::detect_variant`] uses — instead of
  /// restating the `51864` literal a third time in this crate.
  #[inline(always)]
  pub const fn is_multilingual(&self) -> bool {
    is_model_multilingual(self.vocab)
  }
}

// ---------------------------------------------------------------------
// AlignmentView
// ---------------------------------------------------------------------

/// Borrowed, read-only view over a flat `rows * cols` cross-attention
/// alignment-weight buffer — one row per decoded token, one column per
/// encoder audio-context step (spec §5.4's `AlignmentView<'_>`; mirrors
/// Swift's raw `alignmentWeights` `MLMultiArray` without copying it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignmentView<'a> {
  data: &'a [f32],
  rows: usize,
  cols: usize,
}

impl<'a> AlignmentView<'a> {
  /// Wraps `data` as a `rows * cols` row-major view.
  ///
  /// # Panics
  /// If `data.len() != rows * cols`.
  pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
    assert_eq!(
      data.len(),
      rows * cols,
      "AlignmentView: data.len() ({}) != rows ({rows}) * cols ({cols})",
      data.len()
    );
    Self { data, rows, cols }
  }

  /// Number of decoded-token rows.
  #[inline(always)]
  pub const fn rows(&self) -> usize {
    self.rows
  }

  /// Number of encoder audio-context columns.
  #[inline(always)]
  pub const fn cols(&self) -> usize {
    self.cols
  }

  /// The alignment weights for token row `index`.
  ///
  /// # Panics
  /// If `index >= self.rows()`.
  #[inline(always)]
  pub fn row(&self, index: usize) -> &'a [f32] {
    &self.data[index * self.cols..(index + 1) * self.cols]
  }

  /// The full flat `rows * cols` buffer, row-major.
  #[inline(always)]
  pub const fn data(&self) -> &'a [f32] {
    self.data
  }
}

// ---------------------------------------------------------------------
// BackendError
// ---------------------------------------------------------------------

/// Failure calling into an [`InferenceBackend`] — mel extraction, encoding,
/// or a decode step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BackendError {
  /// The CoreML runtime failed to run a prediction.
  #[error("backend prediction failed: {0}")]
  Prediction(#[from] coremlit::PredictionError),
  /// A backend tensor failed to construct or view.
  #[error("backend tensor failed: {0}")]
  Tensor(#[from] coremlit::TensorError),
  /// A model's output feature dictionary lacks an expected named output.
  #[error("{model} model output is missing feature `{name}`")]
  MissingFeature {
    /// Which model produced the incomplete output (e.g. `"encoder"`).
    model: &'static str,
    /// The missing feature's name.
    name: &'static str,
  },
  /// The audio window's sample count doesn't match what the backend
  /// expects.
  #[error("audio window has {got} samples, backend expects {expected}")]
  AudioLength {
    /// Samples actually provided.
    got: usize,
    /// Samples the backend expects.
    expected: usize,
  },
  /// [`mock::MockBackend`]'s scripted logits ran out before the decode loop
  /// finished — the test forgot to script an explicit end-of-text step.
  #[error("mock script exhausted at step {step}")]
  ScriptExhausted {
    /// The step index the script had no entry for.
    step: usize,
  },
}

// ---------------------------------------------------------------------
// InferenceBackend
// ---------------------------------------------------------------------

/// Seam over the three CoreML-shaped inference stages a Whisper pipeline
/// drives per window: mel feature extraction, encoding, and the
/// autoregressive decode step (spec §5.4). A real backend implements this
/// against `coremlit::Model`s; [`mock::MockBackend`] implements it here
/// with no compiled model at all, for hermetic decode-loop,
/// fallback-ladder, windowing, and early-stop tests (spec §9.1).
pub trait InferenceBackend {
  /// Mel features for one 30 s window.
  type Features;
  /// Encoder output for one window.
  type EncoderOutput;
  /// Pre-allocated, reusable decoder tensors (ports Swift `DecodingInputs`).
  type DecoderState;

  /// Extracts mel-spectrogram features from one window of audio samples.
  ///
  /// # Errors
  /// [`BackendError`] if feature extraction fails.
  fn extract_features(&self, audio: &[f32]) -> Result<Self::Features, BackendError>;

  /// Runs the encoder over previously extracted features.
  ///
  /// # Errors
  /// [`BackendError`] if encoding fails.
  fn encode(&self, features: &Self::Features) -> Result<Self::EncoderOutput, BackendError>;

  /// Allocates a fresh decoder state (Swift `DecodingInputs::init`).
  ///
  /// # Errors
  /// [`BackendError`] if the backend cannot allocate its decoder tensors.
  fn new_decoder_state(&self) -> Result<Self::DecoderState, BackendError>;

  /// Ports `DecodingInputs.reset(maxTokenContext:)` (Models.swift:312-322).
  fn reset_decoder_state(&self, state: &mut Self::DecoderState);

  /// One autoregressive step: consume `token` at `position`, write the full
  /// vocab logits (converted to f32 once) into the reused `logits` buffer,
  /// and advance the KV cache/masks/alignment inside `state`.
  ///
  /// `position` is 0-based and must be in `0..dims().max_token_context()`:
  /// it is the KV-cache slot this step fills (the caches are
  /// `[1, kv, 1, max_token_context]`, per the introspected tiny model).
  /// On success the backend leaves `logits` exactly `dims().vocab()` long,
  /// holding only this step's values — the buffer is fully overwritten,
  /// never appended to, so no stale data from prior steps survives.
  ///
  /// # Errors
  /// [`BackendError`] if the step fails.
  fn decode_step(
    &self,
    token: u32,
    position: usize,
    encoder_output: &Self::EncoderOutput,
    state: &mut Self::DecoderState,
    logits: &mut Vec<f32>,
  ) -> Result<(), BackendError>;

  /// Accumulated per-token alignment weights, when the model has the
  /// word-timestamp head (rows = alignment rows written so far — one per
  /// step that produced a row — cols = audio ctx).
  fn alignment_weights<'state>(
    &self,
    state: &'state Self::DecoderState,
  ) -> Option<AlignmentView<'state>>;

  /// Static shape/vocabulary description of the loaded model.
  fn dims(&self) -> ModelDims;
}
