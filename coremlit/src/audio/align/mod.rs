//! CoreML wav2vec2 forced word-level alignment: audio + a known transcript
//! → per-word time spans with confidence.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`.
//!
//! [`Aligner`] is the entry point. It pairs alignkit's CoreML CTC acoustic
//! encoder (`chordai/wav2vec2-base960h-aligner-coreml`, Apache-2.0 — see
//! `tests/model_io.rs` for its pinned I/O contract and provenance), reached
//! through [`crate`] by `Encoder`, with `asry`'s parity-tested
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
//!   ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock, default_oov_policy,
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
//! // OOV is DATA, not policy: detect the events, then decide them. The
//! // resolution is bound to this text and this aligner, and applies once.
//! let resolution = aligner.detect_oov(text)?.decide(default_oov_policy);
//!
//! let alignment = aligner.align_chunk(
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
//!   resolution,
//! )?;
//!
//! // The chunk's words, or why it has none (`alignment.cause()`).
//! for word in alignment.words() {
//!   println!("{:?} {}", word.range(), word.text());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! `no_run` because it needs the CoreML model on disk; it is compiled, so a
//! change to `align_chunk`'s signature breaks it.
//!
//! The result vocabulary ([`UnitAlignment`], [`Word`], [`Lang`],
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
//! which class is the CTC blank, the receptive field and stride of the front
//! end, how the head spells a word, and whether it emits log-probabilities or
//! logits — the caller states in an [`AcousticContract`]:
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use std::path::Path;
//!
//! use coremlit::audio::align::{
//!   AcousticContract, AcousticGeometry, Aligner, AlignerOptions, EnglishNormalizer, Granularity,
//!   Lang, LetterCase, OutputKind, Tokenization, Vocabulary, WordDelimiter,
//! };
//!
//! // A conversion of HuggingFace's 32-class `wav2vec2-base-960h`, and the
//! // `vocab.json` beside it. Its `config.json` names the blank:
//! // `pad_token_id: 0`, the `<pad>` entry; `<s>`, `</s>` and `<unk>` are three
//! // more columns of its 32-class head that are never a letter.
//! let vocabulary = Vocabulary::from_file("Models/hf-base960h/vocab.json")?;
//! // Its front end is wav2vec2's: 16 kHz audio, a 400-sample receptive field
//! // and a 320-sample stride (`AcousticGeometry::WAV2VEC2` spells the same).
//! let geometry =
//!   AcousticGeometry::new(16_000, NonZeroU32::new(400).unwrap(), NonZeroU32::new(320).unwrap())?;
//! // It delimits words with `|` (`word_delimiter_token`) and spells letters in
//! // upper case, one character at a time (the seam's only granularity); `<s>`, `</s>` and
//! // `<unk>` are named specials rather than letters (`<pad>` is the blank
//! // above, already exempt by id). Its head ends in a linear layer: raw logits.
//! let tokenization = Tokenization::new(
//!   WordDelimiter::from_token("|")?,
//!   LetterCase::Upper,
//!   Granularity::Character,
//!   &["<s>", "</s>", "<unk>"],
//! );
//! let contract = AcousticContract::new(0, geometry, tokenization, OutputKind::Logits);
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
//! ([`AlignerError::BlankOutOfVocabulary`]), a tokenization the table or the
//! normalizer contradicts ([`AlignerError::Tokenization`]), a seam that reserves
//! other columns than the contract declares non-lexical
//! ([`AlignerError::ReservedSetMismatch`]), a geometry that does
//! not make the model's declared frame count of its declared window
//! ([`AlignerError::FrameCountMismatch`]), log-probabilities from a head too
//! wide to check ([`AlignerError::UnprovableNormalization`]). Nothing is
//! guessed: not the blank from the table's names, not the geometry from the
//! declared shapes, and not a floor under the log-probabilities, which only the
//! staged artifact's contract carries ([`SentinelBand`]). What asry's seam lets
//! its caller state — the blank, the word delimiter, the letter case, the
//! receptive field and the stride — it is handed from the contract, not left at
//! asry's English wav2vec2 defaults. That is how one aligner per language is
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
//! | boundaries within one 20 ms frame | **367 / 370 (99.2%)** | 37 / 44 (84.1%) |
//! | median disagreement | **0.0 ms** — frame-identical | **0.0 ms** — frame-identical |
//! | p90 disagreement | **0.0 ms** | 20.1 ms |
//!
//! **Feed the encoder a full window and its word boundaries are frame-exact
//! against the reference implementation.** The encoder's CoreML fp16 29-class
//! conversion costs essentially nothing.
//!
//! On a short, zero-padded chunk the *typical* boundary is still frame-exact —
//! jfk's median disagreement is also 0.0 ms — but the **tail** spreads: its p90
//! is 20.1 ms where ted_60's is 0.0. That spread is **padding**, not encoder
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
//! Two boundaries are not determined by the transcript and the emissions alone,
//! and the parity gate holds both to the audio, within one frame:
//!
//! - `jfk.wav`: the second `ask` follows a pause across which `logP(blank)` is
//!   fp16-saturated at exactly `0.0` for 41 consecutive frames, so the tensor
//!   says nothing about where the word begins in it. asry 0.3.0 put the onset
//!   927 ms before the audio contains any evidence for it; asry 0.3.1 starts a
//!   word after a pause at the first frame the model emits its first character,
//!   and the aligner places it at 8,375.2 ms, 4.8 ms before the acoustic onset
//!   at 8,380 ms.
//! - `ted_60.wav`: the speaker says `would` twice and the ASR transcript names
//!   it once. asry 0.3.0 ended the word at the *first* realisation (31,710.6
//!   ms), calling the second — 120 ms of confidently-decoded speech — blank;
//!   asry 0.3.1 ends it at the second, 31,950.7 ms, 10.7 ms after the acoustic
//!   offset at 31,940 ms.
//!
//! Where the transcript has no word for what the speaker said, the audio cannot
//! referee at all, and the CoreML aligner and asry's ONNX oracle part: ted_60's
//! speaker stammers `then` before `actually` and the transcript names it once, so
//! the two end `then` 480 ms apart (30,270.1 against 30,750.3 ms) and start
//! `actually` 200 ms apart (30,690.2 against 30,890.2 ms), neither within a frame
//! of the audio's 30,800 ms. The parity gate pins exactly that divergence.
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
//! | `serde` | no | `Serialize`/`Deserialize` for [`AlignerOptions`] and [`AlignmentFallback`] |
//! | `tracing` | no | structured spans over load and per-chunk alignment — the five below |
//! | `align-oracle` | no | **dev/test only.** Turns on `asry`'s ONNX aligner (and with it `ort` + whisper.cpp) as the oracle for the word-timing parity gate. Adds nothing to this library; see `Cargo.toml`. |
//!
//! ## `tracing` spans
//!
//! | span | level | opened by |
//! |---|---|---|
//! | `alignkit.aligner.load` | `INFO` | [`Aligner::from_paths`] / [`Aligner::from_paths_with`] / [`Aligner::from_paths_with_vocabulary`] |
//! | `alignkit.encoder.load` | `INFO` | `Encoder::load` — nested in the above |
//! | `alignkit.registry.align_chunk` | `DEBUG` | one per [`AlignmentSet::align_chunk`] call (and [`AlignmentHandle::align_chunk`]) |
//! | `alignkit.align_chunk` | `DEBUG` | one per [`Aligner::align_chunk`] call — nested in the above when a registry dispatches it |
//! | `alignkit.encoder.emissions` | `DEBUG` | the CoreML predict — nested in the above |
//!
//! A language is never a bare `language` field: the registry's span carries
//! the `requested_language` and the `route` the lookup took (the
//! [`AlignmentBinding`]: exact, an [`AlignerKey::Any`] fallback naming its own
//! language, or a miss), and an aligner's spans carry its own
//! `aligner_language`. A request an English fallback serves for Korean traces
//! both, each by its own name.
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
//! pins (0.2 and 0.3 alike) panics inside whatever first touches its API (rc.12 deadlocked
//! there instead), deep inside the oracle's session build. So
//! `tests/parity_words.rs` probes the library itself up front, in a child it
//! kills if the load hangs, and panics with an actionable message.

pub mod acoustic;
pub mod aligner;
pub mod encode;
pub mod error;
pub mod registry;
pub mod vocab;

pub use acoustic::{
  AcousticContract, AcousticGeometry, Granularity, LetterCase, OutputKind, SentinelBand,
  Tokenization, WordDelimiter,
};
pub use aligner::{Aligner, AlignerOptions};
pub use error::{
  AlignError, AlignerError, BlankOutOfVocabulary, ContractMismatch, CorruptEmissions,
  DecisionLanguage, ForeignResolution, FrameCountMismatch, GeometryError, InputTooLong,
  MisroutedResolution, MissingId, OutputShape, Refusal, RefusedOov, ReservedSetMismatch,
  TokenizationError, UnnormalizedEmissions, UnprovableNormalization, VocabularyError,
  VocabularyMismatch, VocabularyRead,
};
pub use registry::{
  AlignerKey, AlignmentBinding, AlignmentFallback, AlignmentHandle, AlignmentSet,
  AlignmentSetBuilder, ParseAlignmentFallbackError, SetDetection, SetId, SetOovEvent,
  SetResolution, SetResolvedOov,
};
pub use vocab::Vocabulary;

// `ComputeUnits` is on this crate's own public surface
// ([`AlignerOptions::with_compute`]),
// so re-export it rather than force every consumer to depend on `coremlit`
// directly just to name a compute placement.
pub use crate::ComputeUnits;

// The one vocabulary (design spec §6): result, language, time, OOV, and the
// validated seam input types come straight from `asry`, so a consumer never
// re-imports them from two crates.
pub use asry::{
  Lang, TimeRange, Timebase, Word,
  emissions::{
    AlignedWords, DynTextNormalizer, EmissionsError, EnglishNormalizer, NormalizationError,
    OovDecision, OovDetection, OovEvent, OovKind, OovResolution, OutputClock, ResolvedOov,
    SampleSpan, SpanError, SpeechCoverage, SpeechSpans, TextNormalizer, UnalignedCause,
    UnitAlignment, default_normalizer_for, default_oov_policy, fail_closed_all_policy,
    wildcard_all_policy,
  },
  time::ANALYSIS_TIMEBASE,
};
