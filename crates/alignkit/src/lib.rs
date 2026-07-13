//! CoreML wav2vec2 forced word-level alignment: audio + a known transcript
//! → per-word time spans with confidence.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`.
//!
//! [`Aligner`] is the entry point. It pairs alignkit's CoreML CTC acoustic
//! encoder (`chordai/wav2vec2-base960h-aligner-coreml`, Apache-2.0 — see
//! `tests/model_io.rs` for its pinned I/O contract and provenance), reached
//! through [`coremlit`] by [`encode::Encoder`], with `asry`'s parity-tested
//! alignment seam ([`asry::emissions::EmissionsAligner`]): alignkit runs the
//! encoder, and asry owns everything else — the tokenizer, the silence mask,
//! the CTC trellis / beam / silence-aware word composition. [`AlignmentSet`]
//! keys aligners by language for a multi-language pipeline.
//!
//! ```text
//!   Aligner::align_chunk:  VAD → prepare → [CoreML encode] → finish → Words
//! ```
//!
//! The result vocabulary ([`AlignmentResult`], [`Word`], [`Lang`],
//! [`TimeRange`], and the OOV / speech-span types) is re-exported FROM
//! `asry`, so a caller speaks one vocabulary across the ASR and alignment
//! halves.
//!
//! macOS only (built on [`coremlit`]).

pub mod aligner;
#[cfg(feature = "serde")]
mod compute_units_serde;
pub mod encode;
pub mod error;
pub mod registry;
pub mod vocab;

pub use aligner::{Aligner, AlignerOptions};
pub use error::{AlignError, AlignerError};
pub use registry::{
  AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet, AlignmentSetBuilder,
};

// `ComputeUnits` is on this crate's own public surface
// ([`AlignerOptions::with_compute`], [`encode::EncoderOptions::with_compute`]),
// so re-export it rather than force every consumer to depend on `coremlit`
// directly just to name a compute placement.
pub use coremlit::ComputeUnits;

// The one vocabulary (design spec §6): result, language, time, OOV, and the
// validated seam input types come straight from `asry`, so a consumer never
// re-imports them from two crates.
pub use asry::{
  AlignmentResult, Lang, TimeRange, Timebase, Word,
  emissions::{
    DynTextNormalizer, Emissions, EmissionsError, EnglishNormalizer, NormalizationError,
    OovDecision, OovEvent, OovKind, OutputClock, ResolvedOov, SampleSpan, SpanError,
    SpeechCoverage, SpeechSpans, TextNormalizer, default_normalizer_for, default_oov_decisions,
    fail_closed_all_decisions, wildcard_all_decisions,
  },
  time::ANALYSIS_TIMEBASE,
};
