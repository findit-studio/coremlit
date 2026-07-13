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
//!
//! # How far you can trust the timings
//!
//! Because alignkit and `asry` share every stage except the encoder, the
//! encoder swap can be measured on its own — and it has been, at the word
//! level, on real speech, on the shipping compute default
//! (`tests/parity_words.rs`). On `jfk.wav`, against asry's ONNX-Runtime
//! aligner: **38 of 44 word boundaries agree to within one 20 ms frame**, the
//! median disagreement is **12.8 ms** (under a single frame), and the p90 is
//! 47 ms.
//!
//! One boundary disagrees by 908 ms, and there **the oracle is the one that is
//! wrong**: it places the second `ask` 873 ms before the audio contains any
//! evidence for it, inside a silent pause across which the acoustic model's
//! `logP(blank)` is fp16-saturated at exactly `0.0` for 41 consecutive frames.
//! alignkit puts that word 35 ms from its true acoustic onset. The lesson
//! generalises and is worth stating in the crate's own docs: **a forced
//! aligner's word boundaries are only as determined as the acoustic evidence
//! under them.** Across a long pause, with no VAD, the onset frame is a
//! tie-break among numerically identical paths — supply `sub_segments` from a
//! real VAD when you have one.
//!
//! # Features
//!
//! | feature | default | what it does |
//! |---|---|---|
//! | `serde` | no | `Serialize`/`Deserialize` for [`AlignerOptions`] and [`encode::EncoderOptions`] |
//! | `tracing` | no | structured spans over load and per-chunk alignment |
//! | `parity-oracle` | no | **dev/test only.** Turns on `asry`'s ONNX aligner (and with it `ort` + whisper.cpp) as the oracle for the word-timing parity gate. Adds nothing to this library; see `Cargo.toml`. |
//!
//! # Gates
//!
//! ```text
//! cargo test -p alignkit -- --ignored                        # e2e + determinism + model I/O
//! cargo test -p alignkit --features parity-oracle -- --ignored   # + the word-timing parity gate
//! cargo bench -p alignkit --bench align                      # encode / align_chunk, RTF
//! ```
//!
//! None of them skip: a missing model or fixture is a hard failure, never a
//! green `0 passed`.

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
