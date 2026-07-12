//! Structured error types for `alignkit` (design spec §8). Foreign errors
//! from `coremlit` are wrapped as typed `#[from]` variants — no
//! `Box<dyn Error>`, no string blobs.
//!
//! # Deferred
//!
//! The design spec names two top-level enums: [`AlignerError`] (construction
//! — model loading, tokenizer/vocab parsing) and `AlignError` (per-chunk
//! alignment — wraps `asry::AlignmentError` plus a CoreML prediction
//! variant). Only [`AlignerError`]'s model-loading/contract subset lands
//! here:
//!
//! - Tokenizer parse/vocab-mismatch variants join [`AlignerError`] once a
//!   concrete tokenizer type exists to report on (the vocab bridge, spec
//!   §3.1/§6).
//! - `AlignError` is not defined at all yet: its whole shape is wrapping
//!   `asry::AlignmentError`, and the `asry` dependency is not wired into
//!   this crate (see the crate root `Cargo.toml`'s `publish = false` note).
//!   Introducing it early would mean guessing at a foreign error type this
//!   crate cannot yet compile against.

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

#[cfg(test)]
mod tests;
