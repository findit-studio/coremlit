//! CoreML wav2vec2 forced word-level alignment: audio + a known transcript
//! → per-word time spans with confidence.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`.
//!
//! [`Aligner`] is the entry point. It pairs alignkit's CoreML CTC acoustic
//! encoder (`chordai/wav2vec2-base960h-aligner-coreml`, Apache-2.0 — see
//! `tests/model_io.rs` for its pinned I/O contract and provenance), reached
//! through [`crate`] by [`encode::Encoder`], with `asry`'s parity-tested
//! alignment seam ([`asry::emissions::EmissionsAligner`]): alignkit runs the
//! encoder, and asry owns everything else — the tokenizer, the silence mask,
//! the CTC trellis / beam / silence-aware word composition. [`AlignmentSet`]
//! keys aligners by language for a multi-language pipeline.
//!
//! ```text
//!   Aligner::align_chunk:  VAD → prepare → [CoreML encode] → finish → Words
//! ```
//!
//! # The canonical call
//!
//! [`Aligner::align_chunk`] takes six arguments and three of them have contracts
//! that are not obvious from their types, so here is the shape in full. This is
//! a compiled doctest: it type-checks against the real signature on every
//! `cargo test`.
//!
//! ```no_run
//! use core::sync::atomic::AtomicBool;
//! use std::path::Path;
//!
//! use coremlit::audio::align::{
//!   ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock,
//!   default_oov_decisions,
//! };
//!
//! let aligner = Aligner::from_paths(
//!   Lang::En,
//!   Path::new("Models/alignkit/base960h_aligner.mlmodelc"),
//!   Box::new(EnglishNormalizer::new()),
//! )?;
//!
//! // 16 kHz mono f32, at most `aligner.window_samples()` (60 s on this model).
//! let samples: Vec<f32> = vec![0.0; 16_000];
//! let text = "the transcript of what is said in `samples`";
//!
//! // OOV is DATA, not policy: detect the events, then resolve them. The
//! // decisions must stay in the order `detect_oov` reported them.
//! let events = aligner.detect_oov(text)?;
//! let decisions = default_oov_decisions(&events);
//!
//! let result = aligner.align_chunk(
//!   &samples,
//!   // VAD speech spans in the chunk-local 1/16000 timebase. EMPTY means
//!   // "no VAD" — i.e. all speech, NOT all silence (which would drop every
//!   // word).
//!   &[],
//!   text,
//!   // How stream sample indices map back to the output timebase.
//!   OutputClock::new(0, ANALYSIS_TIMEBASE, 0)?,
//!   // Cooperative cancellation, polled throughout prepare and finish.
//!   &AtomicBool::new(false),
//!   &decisions,
//! )?;
//!
//! for word in result.words() {
//!   println!("{:?} {}", word.range(), word.text());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! `no_run` because it needs the CoreML model on disk; it is compiled, so a
//! change to `align_chunk`'s signature breaks it.
//!
//! The result vocabulary ([`AlignmentResult`], [`Word`], [`Lang`],
//! [`TimeRange`], and the OOV / speech-span types) is re-exported FROM
//! `asry`, so a caller speaks one vocabulary across the ASR and alignment
//! halves.
//!
//! # A model spells with its own vocabulary, under its own contract
//!
//! [`Aligner::from_paths`] binds the bundled 29-class English table and
//! [`AcousticContract::BASE960H`], the vocabulary and the contract of the
//! staged `base960h_aligner.mlmodelc`. A model trained on another alphabet
//! ships its own `{token: id}` table beside it; read it with
//! [`Vocabulary::from_file`]. What neither the model nor the table declares —
//! which class is the CTC blank, and the receptive field and stride of the
//! front end — the caller states in an [`AcousticContract`]:
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use std::path::Path;
//!
//! use coremlit::audio::align::{
//!   AcousticContract, AcousticGeometry, Aligner, AlignerOptions, EnglishNormalizer, Lang,
//!   Vocabulary,
//! };
//!
//! // A conversion of HuggingFace's 32-class `wav2vec2-base-960h`, and the
//! // `vocab.json` beside it. Its `config.json` names the blank:
//! // `pad_token_id: 0`, the `<pad>` entry.
//! let vocabulary = Vocabulary::from_file("Models/hf-base960h/vocab.json")?;
//! // Its front end is wav2vec2's: 16 kHz audio, a 400-sample receptive field
//! // and a 320-sample stride (`AcousticGeometry::WAV2VEC2` spells the same).
//! let geometry =
//!   AcousticGeometry::new(16_000, NonZeroU32::new(400).unwrap(), NonZeroU32::new(320).unwrap())?;
//! let contract = AcousticContract::new(0, geometry);
//! let aligner = Aligner::from_paths_with_vocabulary(
//!   Lang::En,
//!   Path::new("Models/hf-base960h/model.mlmodelc"),
//!   &vocabulary,
//!   &contract,
//!   Box::new(EnglishNormalizer::new()),
//!   AlignerOptions::new(),
//! )?;
//! assert_eq!(aligner.contract(), &contract);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`Aligner::from_paths_with_vocabulary`] checks the statement against what it
//! can see and refuses a disagreement by name at load: a table without one
//! entry per class of the model's CTC head
//! ([`AlignerError::VocabularyMismatch`]), a blank that is no id of the table
//! ([`AlignerError::BlankOutOfVocabulary`]), a geometry that does not make the
//! model's declared frame count of its declared window
//! ([`AlignerError::FrameCountMismatch`]). Nothing is guessed: not the blank
//! from the table's names, not the geometry from the declared shapes, and not
//! a floor under the log-probabilities, which only the staged artifact's
//! contract carries ([`SentinelBand`]). That is how one aligner per language is
//! built; an [`AlignmentSet`] then keys them by language.
//!
//! macOS only (built on [`crate`]).
//!
//! # How far you can trust the timings
//!
//! Because alignkit and `asry` share every stage except the encoder, the
//! encoder swap can be measured on its own — and it has been, at the word
//! level, on real speech, on the shipping compute default
//! (`tests/parity_words.rs`), against asry's ONNX-Runtime aligner. It is
//! measured on **two** clips, and the difference between them is the most
//! useful thing this section can tell you:
//!
//! | | `ted_60.wav` (60 s — **fills the window**) | `jfk.wav` (11 s — **zero-padded**) |
//! |---|---|---|
//! | boundaries within one 20 ms frame | **367 / 372 (98.7%)** | 33 / 44 (75.0%) |
//! | median disagreement | **0.0 ms** — frame-identical | **0.0 ms** — frame-identical |
//! | p90 disagreement | **0.0 ms** | 40.1 ms |
//!
//! **Feed the encoder a full window and its word boundaries are frame-exact
//! against the reference implementation.** The encoder's CoreML fp16 29-class
//! conversion costs essentially nothing.
//!
//! On a short, zero-padded chunk the *typical* boundary is still frame-exact —
//! jfk's median disagreement is also 0.0 ms — but the **tail** spreads: its p90
//! is 40.1 ms where ted_60's is 0.0. That spread is **padding**, not encoder
//! error. The CoreML graph takes a fixed `[1, 960_000]` input
//! ([`encode::ENCODER_WINDOW_SAMPLES`]), so a chunk shorter than 60 s is
//! zero-padded, and wav2vec2-base group-norms over the whole sequence axis and
//! attends globally with no padding mask: the zeros perturb *every* real frame,
//! not just the tail. So, practically: **a short chunk costs you roughly a
//! couple of frames of extra spread on the worst boundaries — fill the window
//! when you can.**
//!
//! ## Where a forced aligner cannot help you
//!
//! On each clip exactly one boundary diverges grossly from the oracle, and on
//! **both** it is the ORACLE that is wrong — the same mechanism twice:
//!
//! - `jfk.wav`: it places the second `ask` 873 ms before the audio contains any
//!   evidence for it, inside a pause across which `logP(blank)` is fp16-saturated
//!   at exactly `0.0` for 41 consecutive frames. alignkit puts that word 50.7 ms
//!   from its true acoustic onset — within the unchanged 3-frame (60 ms) anchor
//!   bound the parity gate holds it to.
//! - `ted_60.wav`: the speaker says `would` twice and the ASR transcript names
//!   it once; the oracle ends the word at the *first* realisation and calls the
//!   second — 120 ms of confidently-decoded speech — blank. alignkit spans the
//!   word's real acoustic support.
//!
//! The lesson generalises and is worth stating in the crate's own docs: **a
//! forced aligner's word boundaries are only as determined as the acoustic
//! evidence under them.** Across a blank-saturated pause, or where the
//! transcript does not name what was actually said, the boundary frame is a
//! tie-break among numerically identical paths. Two things help, and both are
//! yours to supply: pass `sub_segments` from a real VAD when you have one, and
//! give the aligner a transcript that says what the speaker said.
//!
//! # Features
//!
//! | feature | default | what it does |
//! |---|---|---|
//! | `serde` | no | `Serialize`/`Deserialize` for [`AlignerOptions`], [`encode::EncoderOptions`] and [`AlignmentFallback`] |
//! | `tracing` | no | structured spans over load and per-chunk alignment — the four below |
//! | `align-oracle` | no | **dev/test only.** Turns on `asry`'s ONNX aligner (and with it `ort` + whisper.cpp) as the oracle for the word-timing parity gate. Adds nothing to this library; see `Cargo.toml`. |
//!
//! ## `tracing` spans
//!
//! | span | level | opened by |
//! |---|---|---|
//! | `alignkit.aligner.load` | `INFO` | [`Aligner::from_paths`] / [`Aligner::from_paths_with`] / [`Aligner::from_paths_with_vocabulary`] |
//! | `alignkit.encoder.load` | `INFO` | [`encode::Encoder::from_file`] — nested in the above |
//! | `alignkit.align_chunk` | `DEBUG` | one per [`Aligner::align_chunk`] call |
//! | `alignkit.encoder.emissions` | `DEBUG` | the CoreML predict — nested in the above |
//!
//! The two `INFO` spans carry the compute placement, which is the field that
//! explains a load time (0.68 s on the default; **308 s** the first time
//! [`encode::DEFAULT_ENCODER_COMPUTE`] is overridden to an ANE placement). The
//! `DEBUG` spans separate the CoreML predict from the trellis, which is the
//! first question a slow or mis-timed chunk raises.
//!
//! # Gates
//!
//! ```text
//! cargo test -p coremlit --features align -- --ignored           # e2e + determinism + model I/O
//! cargo test -p coremlit --features align-oracle -- --ignored    # + the word-timing parity gate
//! cargo test -p coremlit --features align,tracing -- --ignored   # + the per-chunk span instrumentation
//! cargo bench -p coremlit --features align --bench align_align   # encode / align_chunk, RTF
//! ```
//!
//! None of them skip: a missing model or fixture is a hard failure, never a
//! green `0 passed`.
//!
//! The `tracing` gate is listed on its own for a reason. The per-chunk span
//! test (`alignkit.align_chunk` / `alignkit.encoder.emissions`, in
//! `aligner::tests`) sits behind BOTH `feature = "tracing"` AND `#[ignore]` — it
//! needs a real model to open an alignment span — and **no other gate reaches
//! that combination**: `cargo hack test --each-feature` enables `tracing` but
//! skips ignored tests, and the `--ignored` runs above enable no features (or
//! `align-oracle`). Drop this line and deleting the per-call `#[instrument]`
//! attributes stops being caught by anything.
//!
//! The `align-oracle` gate additionally needs ONNX Runtime at **run** time
//! (`ort` is `load-dynamic`, so the *build* needs nothing), and on Apple
//! Silicon `brew install onnxruntime` alone is not enough: Homebrew's
//! `/opt/homebrew/lib` is on neither `DYLD_LIBRARY_PATH` nor dyld's fallback
//! list, so the bare `dlopen` fails. Point `ORT_DYLIB_PATH` at it:
//!
//! ```text
//! ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
//!   cargo test -p coremlit --features align-oracle -- --ignored
//! ```
//!
//! Without it `ort` does not return an error: the `ort` 2.0.0-rc.13 that asry
//! 0.2 pins panics inside whatever first touches its API (rc.12 deadlocked
//! there instead), deep inside the oracle's session build. So
//! `tests/parity_words.rs` probes the library itself up front, in a child it
//! kills if the load hangs, and panics with an actionable message.

pub mod acoustic;
pub mod aligner;
pub mod encode;
pub mod error;
pub mod registry;
pub mod vocab;

pub use acoustic::{AcousticContract, AcousticGeometry, SentinelBand};
pub use aligner::{Aligner, AlignerOptions};
pub use error::{
  AlignError, AlignerError, BlankOutOfVocabulary, ContractMismatch, CorruptEmissions,
  DecisionLanguage, FrameCountMismatch, GeometryError, InputTooLong, MissingId, OutputShape,
  PaddedChunk, Refusal, UnnormalizedEmissions, VocabularyError, VocabularyMismatch, VocabularyRead,
};
pub use registry::{
  AlignerKey, AlignmentBinding, AlignmentFallback, AlignmentHandle, AlignmentSet,
  AlignmentSetBuilder, ParseAlignmentFallbackError,
};
pub use vocab::Vocabulary;

// `ComputeUnits` is on this crate's own public surface
// ([`AlignerOptions::with_compute`], [`encode::EncoderOptions::with_compute`]),
// so re-export it rather than force every consumer to depend on `coremlit`
// directly just to name a compute placement.
pub use crate::ComputeUnits;

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
