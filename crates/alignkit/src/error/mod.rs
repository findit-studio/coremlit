//! Structured error types for `alignkit` (design spec §8). Foreign errors
//! from `coremlit` and `asry` are wrapped as typed `#[from]` variants — no
//! `Box<dyn Error>`, no string blobs.
//!
//! Two top-level enums, matching the spec's construction-vs-per-call split:
//!
//! - [`AlignerError`]: construction-time — model loading, contract
//!   validation, (later) tokenizer/vocab parsing.
//! - [`AlignError`]: per-call — returned by
//!   [`crate::encode::Encoder::emissions`] today; a future
//!   `Aligner::align_chunk` (spec §6/§7) will return the same type once it
//!   lands, since both sit at the same "one chunk's worth of work" layer.
//!
//! # Deferred
//!
//! - [`AlignerError`]'s tokenizer parse/vocab-mismatch variants join once a
//!   concrete tokenizer type exists to report on (the vocab bridge, spec
//!   §3.1/§6).
//! - [`AlignError`]'s "recoverable subset" semantics (spec §8:
//!   `NoAlignmentPath | EmptyText | SemanticOutOfVocab` mapping to empty
//!   `words: []` rather than a hard error) are an `Aligner`-layer policy,
//!   not an `Encoder`-layer one — [`crate::encode::Encoder::emissions`] has
//!   no transcript/tokenization step, so it never produces those
//!   `asry::AlignmentError` variants itself; it only wraps CoreML
//!   prediction/tensor failures and its own fixed-window length contract.
//!   [`asry::AlignmentError`] variants that *do* reach a caller through
//!   [`AlignError::Alignment`] pass through unconverted for now — the
//!   empty-words mapping lands with the future `Aligner` that actually
//!   drives asry's trellis/beam/compose stage on
//!   [`crate::encode::Encoder::emissions`]'s output.

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
}

/// Failure computing per-chunk CTC emissions or (later) a full word-level
/// alignment (design spec §8's `AlignError`).
///
/// Wraps [`asry::AlignmentError`] — asry's own per-chunk alignment
/// failures, reachable once a caller feeds
/// [`crate::encode::Encoder::emissions`]'s output into asry's
/// `align_emissions`/`compose_words` (spec §7) — alongside the two
/// CoreML-sourced variants [`crate::encode::Encoder::emissions`] itself can
/// raise. The CoreML-sourced shape (`Prediction` + `Tensor`) mirrors
/// `dia-coreml`'s analogous `InferError`
/// (`crates/dia-coreml/src/error/mod.rs`) rather than design spec §8's
/// literal "one CoreML-sourced variant" — `coremlit::Model::predict_with`
/// and `coremlit::MultiArray::from_slice`/`copy_into` fail with two
/// distinct foreign error types ([`coremlit::PredictionError`] and
/// [`coremlit::TensorError`] respectively), and collapsing both into one
/// variant would mean re-stringifying one of them instead of preserving it
/// as a typed `#[from]` source, the exact thing this module's opening
/// paragraph rules out.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum AlignError {
  /// A per-chunk alignment failure from asry's algorithm layer.
  #[error(transparent)]
  Alignment(#[from] asry::AlignmentError),
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
