//! CoreML wav2vec2 forced word-level alignment: audio + a known transcript
//! -> per-word time spans with confidence.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`.
//! The CTC acoustic encoder
//! (`chordai/wav2vec2-base960h-aligner-coreml`, Apache-2.0 — see
//! `tests/model_io.rs` for its pinned I/O contract, provenance, and the
//! logits-vs-log-probs ground truth) runs through [`coremlit`]; everything
//! downstream of the emission matrix — CTC trellis, WhisperX-parity beam
//! backtrack, silence-aware word composition — is `asry`'s existing,
//! parity-tested implementation (not yet wired into this crate; see
//! [`error`]'s module doc for what that means for the error surface today).
//!
//! macOS only (built on [`coremlit`]).

pub mod error;
pub mod vocab;
