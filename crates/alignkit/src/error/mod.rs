//! Structured error types for `alignkit` (design spec §8). Foreign errors
//! from `coremlit` and `asry` are wrapped as typed `#[from]` variants — no
//! `Box<dyn Error>`, no string blobs.
//!
//! Two top-level enums, matching the spec's construction-vs-per-call split:
//!
//! - [`AlignerError`]: construction-time — loading and contract-validating
//!   the CoreML model ([`AlignerError::Load`],
//!   [`AlignerError::ContractMismatch`]) and building asry's alignment seam
//!   from the tokenizer + normalizer ([`AlignerError::Seam`]).
//! - [`AlignError`]: per-call — returned by both
//!   [`crate::encode::Encoder::emissions`] and
//!   [`crate::aligner::Aligner::align_chunk`], which sit at the same "one
//!   chunk's worth of work" layer.
//!
//! # The recoverable subset lives in `Aligner`, not here
//!
//! Two of the seam's [`asry::emissions::EmissionsError`] variants —
//! `NoAlignmentPath` and `SemanticOutOfVocab` — are *recoverable*: a chunk
//! that hits them yields an empty `AlignmentResult` (the ASR text is kept,
//! only per-word timings are dropped), not a hard error. That mapping is a
//! policy of [`crate::aligner::Aligner::align_chunk`], which converts those
//! two into `Ok(empty)` before they ever become an [`AlignError`] —
//! mirroring asry's own `alignment_failure_is_recoverable`
//! (`asry/src/runner/alignment_pool/mod.rs`). Every `EmissionsError` that
//! DOES reach [`AlignError::Alignment`] is therefore a genuine failure. The
//! pre-seam `EmptyText` recoverable case is gone: empty / untokenizable text
//! is now `PreparedChunk::is_trivial()`, short-circuited to an empty result
//! with no error at all.

/// Failure locating, loading, or validating the CoreML wav2vec2 forced-
/// aligner model (design spec §8's `AlignerError`, model-loading subset).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AlignerError {
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] coremlit::LoadError),
  /// A loaded model's input or output feature does not match the
  /// shape/dtype contract this crate was built against (see
  /// `tests/model_io.rs` for the pinned ground truth).
  #[error("model contract mismatch on `{feature}`: expected {expected}, got {actual}")]
  ContractMismatch {
    /// Name of the input/output feature that mismatched.
    feature: &'static str,
    /// The contract this crate expects, rendered for display.
    expected: String,
    /// What the loaded model actually declares, rendered for display.
    actual: String,
  },
  /// Building asry's alignment seam
  /// ([`asry::emissions::EmissionsAligner`]) failed: the tokenizer JSON did
  /// not parse, the CTC blank token could not be resolved, the language has
  /// no default text normalizer, or the normalizer needs a `|`
  /// word-delimiter the tokenizer lacks. Surfaced by
  /// [`crate::aligner::Aligner::from_paths`].
  #[error("alignment seam construction failed: {0}")]
  Seam(#[from] asry::emissions::EmissionsError),
}

/// Failure computing per-chunk CTC emissions or a full word-level
/// alignment (design spec §8's `AlignError`).
///
/// Wraps [`asry::emissions::EmissionsError`] — asry's own per-chunk
/// alignment failures from the emissions seam
/// ([`crate::aligner::Aligner::align_chunk`] feeds
/// [`crate::encode::Encoder::emissions`]'s output through
/// `prepare`/`finish`) — alongside the CoreML-sourced variants
/// [`crate::encode::Encoder::emissions`] itself can raise and the
/// [`asry::emissions::SpanError`] the VAD bridge can produce. The
/// CoreML-sourced shape (`Prediction` + `Tensor`) mirrors `dia-coreml`'s
/// analogous `InferError` (`crates/dia-coreml/src/error/mod.rs`) rather than
/// design spec §8's literal "one CoreML-sourced variant" —
/// `coremlit::Model::predict_with` and
/// `coremlit::MultiArray::from_slice`/`copy_into` fail with two distinct
/// foreign error types ([`coremlit::PredictionError`] and
/// [`coremlit::TensorError`] respectively), and collapsing both into one
/// variant would mean re-stringifying one of them instead of preserving it
/// as a typed `#[from]` source, the exact thing this module's opening
/// paragraph rules out.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum AlignError {
  /// A per-chunk alignment failure from asry's emissions seam — stride /
  /// vocab / blank-id validation, a non-finite or positive log-probability
  /// from the encoder, tokenization, or the trellis. The *recoverable*
  /// subset (`NoAlignmentPath`, `SemanticOutOfVocab`) never reaches here;
  /// see the module doc.
  #[error(transparent)]
  Alignment(#[from] asry::emissions::EmissionsError),
  /// The VAD sub-segments were not in the chunk-local 1/16000 analysis
  /// timebase (or exceeded the representable sample range) when
  /// [`crate::aligner::Aligner::align_chunk`] bridged them into
  /// [`asry::emissions::SpeechSpans`].
  #[error(transparent)]
  Span(#[from] asry::emissions::SpanError),
  /// The CoreML runtime failed to run the encoder.
  #[error("prediction failed: {0}")]
  Prediction(#[from] coremlit::PredictionError),
  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] coremlit::TensorError),
  /// `samples` exceeded [`crate::encode::Encoder::emissions`]'s fixed
  /// input window.
  #[error("input exceeds encoder window: {got} samples > {max} samples")]
  InputTooLong {
    /// Samples the caller supplied.
    got: usize,
    /// The encoder's fixed window size
    /// ([`crate::encode::ENCODER_WINDOW_SAMPLES`]).
    max: usize,
  },
}

#[cfg(test)]
mod tests;
