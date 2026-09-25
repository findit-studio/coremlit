//! Structured error types for `alignkit` (design spec §8). Foreign errors
//! from `coremlit` and `asry` are wrapped as typed `#[from]` variants — no
//! `Box<dyn Error>`, no string blobs.
//!
//! Three enums, matching the spec's construction-vs-per-call split, with the
//! vocabulary a model ships beside it read before either:
//!
//! - [`VocabularyError`]: reading a model's own `{token: id}` table into a
//!   [`crate::audio::align::vocab::Vocabulary`].
//! - [`AlignerError`]: construction-time — loading and contract-validating
//!   the CoreML model ([`AlignerError::Load`],
//!   [`AlignerError::ContractMismatch`]), building asry's alignment seam
//!   from the vocabulary + normalizer ([`AlignerError::Seam`]), and pairing the
//!   two ([`AlignerError::VocabularyMismatch`]).
//! - [`AlignError`]: per-call — returned by both
//!   [`crate::audio::align::encode::Encoder::emissions`] and
//!   [`crate::audio::align::aligner::Aligner::align_chunk`], which sit at the same "one
//!   chunk's worth of work" layer.
//!
//! # A refusal and an unalignable chunk are named, never an empty result
//!
//! Two of the seam's [`asry::emissions::EmissionsError`] variants are
//! per-chunk outcomes rather than a broken setup: `SemanticOutOfVocab` (the
//! caller's OOV decisions resolved a position `FailClosed`) and
//! `NoAlignmentPath` (the CTC lattice admits no path for this chunk's audio and
//! tokens). [`crate::audio::align::aligner::Aligner::align_chunk`] names each:
//! [`AlignError::Refused`] carries every position the caller's policy refused,
//! and [`AlignError::NoAlignmentPath`] carries asry's diagnostic. Neither is an
//! empty `AlignmentResult`, so a caller can tell them apart from each other and
//! from an empty SUCCESS: text that normalizes to nothing or yields no tokens
//! (`PreparedChunk::is_trivial()`, short-circuited before the encoder), or an
//! alignment whose every word fell outside the chunk's speech. Either way the
//! ASR text is the caller's to keep; only per-word timings are missing. Every
//! other `EmissionsError` reaches [`AlignError::Alignment`] and is a genuine
//! failure.

/// A loaded model's input or output feature does not match the
/// shape/dtype contract this crate was built against (see
/// `tests/model_io.rs` for the pinned ground truth).
///
/// Payload of [`AlignerError::ContractMismatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractMismatch {
  /// Name of the input/output feature that mismatched.
  feature: &'static str,
  /// The contract this crate expects, rendered for display.
  expected: String,
  /// What the loaded model actually declares, rendered for display.
  actual: String,
}

impl ContractMismatch {
  /// Construct from the mismatched feature, the expected contract, and what
  /// the loaded model actually declares.
  #[inline(always)]
  pub const fn new(feature: &'static str, expected: String, actual: String) -> Self {
    Self {
      feature,
      expected,
      actual,
    }
  }

  /// Name of the input/output feature that mismatched.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The contract this crate expects, rendered for display.
  #[inline(always)]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the loaded model actually declares, rendered for display.
  #[inline(always)]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// Failure locating, loading, or validating the CoreML wav2vec2 forced-
/// aligner model (design spec §8's `AlignerError`, model-loading subset).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AlignerError {
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] crate::LoadError),
  /// A loaded model's input or output feature does not match the
  /// shape/dtype contract this crate was built against (see
  /// `tests/model_io.rs` for the pinned ground truth).
  #[error(
    "model contract mismatch on `{}`: expected {}, got {}",
    .0.feature(),
    .0.expected(),
    .0.actual()
  )]
  ContractMismatch(ContractMismatch),
  /// The loaded graph declares a REQUIRED input this door never supplies, so
  /// every prediction through it would fail.
  ///
  /// Carries the offending feature name. An OPTIONAL extra input is not this:
  /// CoreML runs a prediction that omits one, so only a required input the
  /// door cannot fill makes the contract unsatisfiable.
  #[error(
    "model declares a required input `{0}` that this door never supplies; \
     it sends `waveform` and nothing else, so every prediction would fail"
  )]
  UnsatisfiableInput(String),
  /// The loaded graph declares CoreML STATE buffers, and this door predicts
  /// through the stateless API.
  ///
  /// Carries the offending state feature name. A stateful model must receive an
  /// `MLState` on every prediction; a door that never makes one either fails
  /// the prediction outright or silently discards the persistence the graph was
  /// built around. Neither is something to discover at predict time.
  #[error(
    "model declares the state buffer `{0}`, and this door predicts through the \
     stateless API; a stateful graph needs an `MLState` on every prediction"
  )]
  UnsatisfiableState(String),
  /// Building asry's alignment seam
  /// ([`asry::emissions::EmissionsAligner`]) failed: the tokenizer JSON did
  /// not parse, the CTC blank token could not be resolved, the language has
  /// no default text normalizer, or the normalizer needs a `|`
  /// word-delimiter the vocabulary lacks. Surfaced by
  /// [`crate::audio::align::aligner::Aligner::from_paths`].
  #[error("alignment seam construction failed: {0}")]
  Seam(#[from] asry::emissions::EmissionsError),
  /// The vocabulary does not have one entry per class of the model's CTC
  /// head: it names one number of classes, the model's `emissions` scores
  /// another per frame.
  ///
  /// Refused at load because the pair is wrong for every chunk: asry would read
  /// each token's posterior from a column that does not belong to it. A model
  /// is paired with the vocabulary that ships beside it
  /// ([`crate::audio::align::vocab::Vocabulary::from_file`]); the bundled table
  /// [`crate::audio::align::aligner::Aligner::from_paths`] binds is the 29-class
  /// English one.
  #[error(
    "the vocabulary names {} classes but the model's CTC head scores {} per frame; \
     pair the model with the vocabulary that ships beside it",
    .0.vocabulary(),
    .0.model()
  )]
  VocabularyMismatch(VocabularyMismatch),
}

/// A vocabulary and a model's CTC head disagree on the number of classes.
///
/// Payload of [`AlignerError::VocabularyMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabularyMismatch {
  /// Entries in the vocabulary the seam was built from.
  vocabulary: usize,
  /// Classes the model's `emissions` scores per frame.
  model: usize,
}

impl VocabularyMismatch {
  /// Construct from the vocabulary's entry count and the model's CTC head
  /// width.
  #[inline(always)]
  pub const fn new(vocabulary: usize, model: usize) -> Self {
    Self { vocabulary, model }
  }

  /// Entries in the vocabulary the seam was built from.
  #[inline(always)]
  pub const fn vocabulary(&self) -> usize {
    self.vocabulary
  }

  /// Classes the model's `emissions` scores per frame.
  #[inline(always)]
  pub const fn model(&self) -> usize {
    self.model
  }
}

/// Failure reading a model's own CTC vocabulary into a
/// [`crate::audio::align::vocab::Vocabulary`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VocabularyError {
  /// The vocabulary file could not be read.
  #[error(transparent)]
  Read(VocabularyRead),
  /// The bytes are not a JSON object mapping each token to a non-negative
  /// integer id. Carries the parser's diagnostic.
  #[error("a vocabulary is a JSON object mapping each token to its id: {0}")]
  Parse(String),
  /// An id in `0..n` names no token, where `n` is the number of entries: the
  /// table skips an id, or gives two tokens the same one. Each of a CTC head's
  /// columns is one class, so a table that does not name every id exactly once
  /// would leave a column unnamed or read one column for two tokens.
  #[error(
    "no token has id {}: a vocabulary of {} entries names every id in 0..{} exactly once, \
     one per class of its model's CTC head",
    .0.id(),
    .0.entries(),
    .0.entries()
  )]
  MissingId(MissingId),
  /// No entry is a CTC blank: the table holds none of `<pad>`, `[PAD]`,
  /// `<blank>` or `-`.
  #[error(
    "no entry is the CTC blank: a vocabulary names its blank `<pad>`, `[PAD]`, `<blank>` \
     or `-`"
  )]
  NoBlank,
}

/// The vocabulary file at [`Self::path`] could not be read.
///
/// Payload of [`VocabularyError::Read`], which is `#[error(transparent)]`: this
/// struct owns the message and the `#[source]`, so the error chain has one
/// link, to the [`std::io::Error`].
#[derive(Debug, thiserror::Error)]
#[error("failed to read the vocabulary `{path}`: {source}")]
pub struct VocabularyRead {
  /// The file that could not be read.
  path: std::path::PathBuf,
  /// The underlying I/O failure.
  #[source]
  source: std::io::Error,
}

impl VocabularyRead {
  /// Construct from the file that could not be read and the underlying I/O
  /// failure.
  #[inline(always)]
  pub const fn new(path: std::path::PathBuf, source: std::io::Error) -> Self {
    Self { path, source }
  }

  /// The file that could not be read.
  #[inline(always)]
  pub fn path(&self) -> &std::path::Path {
    &self.path
  }
}

/// An id of `0..entries` that no token of a vocabulary holds.
///
/// Payload of [`VocabularyError::MissingId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingId {
  /// The lowest id in `0..entries` no token holds.
  id: usize,
  /// The number of entries in the table.
  entries: usize,
}

impl MissingId {
  /// Construct from the lowest unnamed id and the table's entry count.
  #[inline(always)]
  pub const fn new(id: usize, entries: usize) -> Self {
    Self { id, entries }
  }

  /// The lowest id in `0..entries` no token holds.
  #[inline(always)]
  pub const fn id(&self) -> usize {
    self.id
  }

  /// The number of entries in the table.
  #[inline(always)]
  pub const fn entries(&self) -> usize {
    self.entries
  }
}

/// `samples` exceeded [`crate::audio::align::encode::Encoder::emissions`]'s fixed
/// input window.
///
/// Payload of [`AlignError::InputTooLong`].
#[derive(Debug, Clone)]
pub struct InputTooLong {
  /// Samples the caller supplied.
  got: usize,
  /// The encoder's fixed window size
  /// ([`crate::audio::align::encode::ENCODER_WINDOW_SAMPLES`]).
  max: usize,
}

impl InputTooLong {
  /// Construct from the sample count the caller supplied and the encoder's
  /// fixed window size.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// Samples the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The encoder's fixed window size
  /// ([`crate::audio::align::encode::ENCODER_WINDOW_SAMPLES`]).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// The encoder returned an emission matrix that is **not log-probabilities**:
/// at least one cell sits below [`crate::audio::align::encode::LOG_PROB_FLOOR`], the fp16
/// `log(0)` saturation sentinel (`≈ -45440`).
///
/// This is the loud form of what used to be a silent one. The values are
/// finite and negative, so they pass `Emissions::from_log_probs`' own
/// `finite ∧ <= 0` scan untouched and would align to *plausible, wrong*
/// timings (in the pre-truncation-fix measurement `ask` landed 881.6 ms early
/// on `jfk.wav`) — which is why the floor is checked separately. See
/// [`crate::audio::align::encode::DEFAULT_ENCODER_COMPUTE`] for the mechanism and
/// [`crate::audio::align::encode::LOG_PROB_FLOOR`] for why the guard keys on the value
/// domain rather than on the compute placement.
///
/// The corruption is a defect of the **model artifact**, not of the caller's
/// audio: no input makes a *correctly-converted* artifact produce it. But on a
/// *corrupted* artifact its DETECTION is input-dependent — this error fires
/// only when the input drives a class posterior under the fp16 floor and so
/// exposes the `log(0)` sentinel. Real speech can (measured `min ≈ -45440` on
/// `jfk.wav`); 960,000 samples of digital silence (`min ≈ -8.55`) and a
/// low-amplitude sine (`≈ -9.07`) stay ABOVE the floor and pass clean even on
/// the corrupt placement — the recorded evidence in
/// `tests::emissions_reject_an_ane_corrupted_matrix`'s doc, and why real speech
/// is load-bearing there. The fix is the placement named in this error, or a
/// re-converted model; nothing in this crate can recover the underflowed cells.
///
/// Payload of [`AlignError::CorruptEmissions`].
#[derive(Debug, Clone)]
pub struct CorruptEmissions {
  /// The compute placement the encoder was loaded on — the knob the caller
  /// can actually turn, hence the one the message names.
  compute: crate::ComputeUnits,
  /// The most negative cell in the matrix (`≈ -45440` on an ANE placement;
  /// `-30.81` on the `CpuOnly` default, measured on `jfk.wav`).
  min: f32,
  /// How many cells fell below [`crate::audio::align::encode::LOG_PROB_FLOOR`] (2,667 on
  /// `jfk.wav`'s ANE run).
  cells: usize,
  /// Cells scanned: `frames × `[`Encoder::vocab_size`](crate::audio::align::encode::Encoder::vocab_size)
  /// (15,921 on `jfk.wav`).
  total: usize,
}

impl CorruptEmissions {
  /// Construct from the compute placement, the most negative cell, the number
  /// of cells below the floor, and the number of cells scanned.
  #[inline(always)]
  pub const fn new(compute: crate::ComputeUnits, min: f32, cells: usize, total: usize) -> Self {
    Self {
      compute,
      min,
      cells,
      total,
    }
  }

  /// The compute placement the encoder was loaded on — the knob the caller
  /// can actually turn, hence the one the message names.
  #[inline(always)]
  pub const fn compute(&self) -> crate::ComputeUnits {
    self.compute
  }

  /// The most negative cell in the matrix (`≈ -45440` on an ANE placement;
  /// `-30.81` on the `CpuOnly` default, measured on `jfk.wav`).
  #[inline(always)]
  pub const fn min(&self) -> f32 {
    self.min
  }

  /// How many cells fell below [`crate::audio::align::encode::LOG_PROB_FLOOR`] (2,667 on
  /// `jfk.wav`'s ANE run).
  #[inline(always)]
  pub const fn cells(&self) -> usize {
    self.cells
  }

  /// Cells scanned: `frames × `[`Encoder::vocab_size`](crate::audio::align::encode::Encoder::vocab_size)
  /// (15,921 on `jfk.wav`).
  #[inline(always)]
  pub const fn total(&self) -> usize {
    self.total
  }
}

/// The encoder returned an emission matrix that is **not normalized
/// log-probabilities**: frame `row`'s `logsumexp` over the vocab axis is
/// `logsumexp`, exceeding [`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`] in
/// magnitude. A genuine CTC log-probability frame sums to 1 in probability
/// space, so its `logsumexp` is `0` (`ln Σ exp(log p_j) = ln Σ p_j = ln 1`); a
/// whole-unit deviation means the tensor carries raw logits — or another
/// un-normalized distribution — not the log-softmaxed output this crate's
/// encoder contract requires.
///
/// Unlike [`AlignError::CorruptEmissions`] — a placement-dependent fp16 underflow of
/// THIS reviewed artifact — this is a **model-artifact contract** failure that
/// no placement causes and no placement cures: a revision shipping a raw-logit
/// CTC head (the *standard* wav2vec2 export, and what asry's own ONNX model
/// emits) is rejected here rather than silently re-normalized by
/// `Emissions::from_logits` and aligned on forever. It is the check that makes
/// [`crate::audio::align::encode::Encoder::emissions`]'s "these really are log-probs" a
/// verified contract for any same-contract artifact loaded through the public
/// API, not merely for the one reviewed here. The finite ∧ `<= 0` scan
/// `Emissions::from_log_probs` runs cannot catch it: raw logits shifted wholly
/// into `[-20, -10]`, or an all-zeros frame, are finite and `<= 0` on every
/// cell yet no distribution at all. See
/// [`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`] for the measured tolerance and the
/// [`crate::audio::align::encode`] module doc's "The normalization guard".
///
/// Payload of [`AlignError::UnnormalizedEmissions`].
#[derive(Debug, Clone)]
pub struct UnnormalizedEmissions {
  /// The compute placement the encoder was loaded on. Carried for parity with
  /// [`AlignError::CorruptEmissions`]; unlike that error the placement is not the
  /// cause here (the model artifact is), but it remains useful context.
  compute: crate::ComputeUnits,
  /// Index of the worst frame — the one with the largest `|logsumexp|`.
  row: usize,
  /// That frame's `logsumexp` over the vocab axis (`≈ 0` for real log-probs;
  /// `ln 29 ≈ 3.367` for an all-zeros frame; `>= 6.6` for a `[-20, -10]`
  /// shifted raw-logit frame). Accumulated in `f64`.
  logsumexp: f64,
  /// The bound it exceeded ([`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`]).
  tolerance: f64,
}

impl UnnormalizedEmissions {
  /// Construct from the compute placement, the worst frame's index, that
  /// frame's `logsumexp`, and the tolerance it exceeded.
  #[inline(always)]
  pub const fn new(
    compute: crate::ComputeUnits,
    row: usize,
    logsumexp: f64,
    tolerance: f64,
  ) -> Self {
    Self {
      compute,
      row,
      logsumexp,
      tolerance,
    }
  }

  /// The compute placement the encoder was loaded on. Carried for parity with
  /// [`AlignError::CorruptEmissions`]; unlike that error the placement is not the
  /// cause here (the model artifact is), but it remains useful context.
  #[inline(always)]
  pub const fn compute(&self) -> crate::ComputeUnits {
    self.compute
  }

  /// Index of the worst frame — the one with the largest `|logsumexp|`.
  #[inline(always)]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// That frame's `logsumexp` over the vocab axis (`≈ 0` for real log-probs;
  /// `ln 29 ≈ 3.367` for an all-zeros frame; `>= 6.6` for a `[-20, -10]`
  /// shifted raw-logit frame). Accumulated in `f64`.
  #[inline(always)]
  pub const fn logsumexp(&self) -> f64 {
    self.logsumexp
  }

  /// The bound it exceeded ([`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`]).
  #[inline(always)]
  pub const fn tolerance(&self) -> f64 {
    self.tolerance
  }
}

/// A caller-supplied OOV decision does not carry the requested language.
///
/// Returned by [`crate::audio::align::registry::AlignmentSet::align_chunk`] when the
/// `ResolvedOov` at position `index` carries `found` rather than the
/// `requested` language the chunk is being aligned for. The registry checks
/// this BEFORE crossing the decisions into an
/// [`AlignerKey::Any`](crate::audio::align::registry::AlignerKey::Any) fallback aligner's
/// own language: a foreign-language decision would otherwise be re-stamped and
/// silently apply another language's wildcard / fail-closed policy at a
/// matching position (asry's `ResolvedOov` identity ignores language on
/// purpose, so nothing downstream would catch it). Resolve decisions against
/// the SAME language you pass to `align_chunk` — the one
/// [`AlignmentSet::detect_oov`](crate::audio::align::registry::AlignmentSet::detect_oov)
/// stamped them with.
///
/// Payload of [`AlignError::DecisionLanguage`].
#[derive(Debug, Clone)]
pub struct DecisionLanguage {
  /// Index of the offending decision in the caller's `oov_decisions` slice.
  index: usize,
  /// The language the chunk is being aligned for (the `align_chunk` argument).
  requested: asry::Lang,
  /// The language the decision actually carries.
  found: asry::Lang,
}

impl DecisionLanguage {
  /// Construct from the offending decision's index, the language the chunk is
  /// being aligned for, and the language the decision actually carries.
  #[inline(always)]
  pub const fn new(index: usize, requested: asry::Lang, found: asry::Lang) -> Self {
    Self {
      index,
      requested,
      found,
    }
  }

  /// Index of the offending decision in the caller's `oov_decisions` slice.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// The language the chunk is being aligned for (the `align_chunk` argument).
  #[inline(always)]
  pub const fn requested(&self) -> &asry::Lang {
    &self.requested
  }

  /// The language the decision actually carries.
  #[inline(always)]
  pub const fn found(&self) -> &asry::Lang {
    &self.found
  }
}

/// Every position a caller's OOV decisions refused in one chunk.
///
/// Payload of [`AlignError::Refused`]. The events are those the caller's
/// decisions resolved `FailClosed`, in the order the caller passed them (the
/// order `detect_oov` reported them). A `Symbol` or `InternalPunct` event names
/// its character ([`OovEvent::char`](asry::emissions::OovEvent::char)); a
/// `BoundaryPunct` event carries none, because the normalizer stripped that mark
/// before tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
  /// The refused positions, as the caller's decisions carried them.
  events: Vec<asry::emissions::OovEvent>,
}

impl Refusal {
  /// Construct from the refused positions.
  #[inline(always)]
  pub const fn new(events: Vec<asry::emissions::OovEvent>) -> Self {
    Self { events }
  }

  /// The refused positions, as the caller's decisions carried them.
  #[inline(always)]
  pub fn events(&self) -> &[asry::emissions::OovEvent] {
    &self.events
  }

  /// This refusal with every event stamped `language` — the language a
  /// registry request named, when the aligner that ran was an
  /// [`AlignerKey::Any`](crate::audio::align::registry::AlignerKey::Any) fallback
  /// handed the decisions crossed into its own.
  pub(crate) fn stamped(mut self, language: &asry::Lang) -> Self {
    for event in &mut self.events {
      event.set_language(language.clone());
    }
    self
  }
}

/// Each refused position, comma-separated: a character quoted with the
/// zero-based index of its word, or a boundary mark (whose character the
/// normalizer removed) with the index of its word.
impl core::fmt::Display for Refusal {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    if self.events.is_empty() {
      return f.write_str("no position");
    }
    for (at, event) in self.events.iter().enumerate() {
      if at > 0 {
        f.write_str(", ")?;
      }
      match event.char() {
        Some(symbol) => write!(f, "{symbol:?} (word {})", event.word_index())?,
        None => write!(f, "a boundary mark (word {})", event.word_index())?,
      }
    }
    Ok(())
  }
}

/// Failure computing per-chunk CTC emissions or a full word-level
/// alignment (design spec §8's `AlignError`).
///
/// Wraps [`asry::emissions::EmissionsError`] — asry's own per-chunk
/// alignment failures from the emissions seam
/// ([`crate::audio::align::aligner::Aligner::align_chunk`] feeds
/// [`crate::audio::align::encode::Encoder::emissions`]'s output through
/// `prepare`/`finish`) — alongside the CoreML-sourced variants
/// [`crate::audio::align::encode::Encoder::emissions`] itself can raise and the
/// [`asry::emissions::SpanError`] the VAD bridge can produce. The
/// CoreML-sourced shape (`Prediction` + `Tensor`) mirrors `dia-coreml`'s
/// analogous `InferError` (`crates/dia-coreml/src/error/mod.rs`) rather than
/// design spec §8's literal "one CoreML-sourced variant" —
/// `crate::Model::predict_with` and
/// `crate::MultiArray::from_slice`/`copy_into` fail with two distinct
/// foreign error types ([`crate::PredictionError`] and
/// [`crate::TensorError`] respectively), and collapsing both into one
/// variant would mean re-stringifying one of them instead of preserving it
/// as a typed `#[from]` source, the exact thing this module's opening
/// paragraph rules out.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum AlignError {
  /// A per-chunk alignment failure from asry's emissions seam — stride /
  /// vocab / blank-id validation, a non-finite or positive log-probability
  /// from the encoder, tokenization, or abort. A refusal and an unalignable
  /// chunk never reach here: they are [`Self::Refused`] and
  /// [`Self::NoAlignmentPath`]; see the module doc.
  #[error(transparent)]
  Alignment(#[from] asry::emissions::EmissionsError),
  /// The caller's OOV policy refused the chunk: at least one decision it passed
  /// resolved a position `FailClosed`, so no word timings were produced.
  ///
  /// Carries every refused position ([`Refusal::events`]), read off the
  /// decisions the caller passed. A refusal is the caller's own policy at
  /// work, not a failure of the aligner: the text is the caller's to keep and
  /// only its word timings are missing. It is never an empty result, so it
  /// cannot be mistaken for [`Self::NoAlignmentPath`] or for a chunk with
  /// nothing to align.
  #[error("the OOV policy refused this chunk at {0}; no word timings were produced")]
  Refused(Refusal),
  /// The CTC lattice admits no alignment path for this chunk: its audio is too
  /// short for its tokens (one frame cannot carry three), a trellis boundary
  /// cell is non-finite, or the lattice overran its cell budget.
  ///
  /// Carries asry's diagnostic. Like [`Self::Refused`] it is an outcome of this
  /// chunk's data, not a broken setup: the text is the caller's to keep and only
  /// its word timings are missing. It is never an empty result, so it cannot be
  /// mistaken for a refusal or for a chunk with nothing to align.
  #[error("no alignment path for this chunk: {0}")]
  NoAlignmentPath(asry::emissions::EmissionsFailure),
  /// The VAD sub-segments were not in the chunk-local 1/16000 analysis
  /// timebase (or exceeded the representable sample range) when
  /// [`crate::audio::align::aligner::Aligner::align_chunk`] bridged them into
  /// [`asry::emissions::SpeechSpans`].
  #[error(transparent)]
  Span(#[from] asry::emissions::SpanError),
  /// The CoreML runtime failed to run the encoder.
  #[error("prediction failed: {0}")]
  Prediction(#[from] crate::PredictionError),
  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] crate::TensorError),
  /// `samples` exceeded [`crate::audio::align::encode::Encoder::emissions`]'s fixed
  /// input window.
  #[error("input exceeds encoder window: {} samples > {} samples", .0.got(), .0.max())]
  InputTooLong(InputTooLong),
  /// The encoder returned an emission matrix that is **not log-probabilities**:
  /// at least one cell sits below [`crate::audio::align::encode::LOG_PROB_FLOOR`], the fp16
  /// `log(0)` saturation sentinel (`≈ -45440`).
  ///
  /// This is the loud form of what used to be a silent one. The values are
  /// finite and negative, so they pass `Emissions::from_log_probs`' own
  /// `finite ∧ <= 0` scan untouched and would align to *plausible, wrong*
  /// timings (in the pre-truncation-fix measurement `ask` landed 881.6 ms early
  /// on `jfk.wav`) — which is why the floor is checked separately. See
  /// [`crate::audio::align::encode::DEFAULT_ENCODER_COMPUTE`] for the mechanism and
  /// [`crate::audio::align::encode::LOG_PROB_FLOOR`] for why the guard keys on the value
  /// domain rather than on the compute placement.
  ///
  /// The corruption is a defect of the **model artifact**, not of the caller's
  /// audio: no input makes a *correctly-converted* artifact produce it. But on a
  /// *corrupted* artifact its DETECTION is input-dependent — this error fires
  /// only when the input drives a class posterior under the fp16 floor and so
  /// exposes the `log(0)` sentinel. Real speech can (measured `min ≈ -45440` on
  /// `jfk.wav`); 960,000 samples of digital silence (`min ≈ -8.55`) and a
  /// low-amplitude sine (`≈ -9.07`) stay ABOVE the floor and pass clean even on
  /// the corrupt placement — the recorded evidence in
  /// `tests::emissions_reject_an_ane_corrupted_matrix`'s doc, and why real speech
  /// is load-bearing there. The fix is the placement named in this error, or a
  /// re-converted model; nothing in this crate can recover the underflowed cells.
  #[error(
    "encoder emissions are not log-probabilities: {} of {} cells are below {floor} \
     (min = {}), the fp16 `log(0)` saturation sentinel. The encoder was scheduled on \
     {:?}: this model's fp16 `log(softmax(·))` tail underflows on the Apple Neural \
     Engine and its word timings shift by hundreds of milliseconds. Load the encoder on \
     `coremlit::audio::align::encode::DEFAULT_ENCODER_COMPUTE` (the default, and the fastest correct \
     placement) — or re-convert the model with a fused `log_softmax` tail.",
    .0.cells(),
    .0.total(),
    .0.min(),
    .0.compute(),
    floor = crate::audio::align::encode::LOG_PROB_FLOOR,
  )]
  CorruptEmissions(CorruptEmissions),
  /// The encoder returned an emission matrix that is **not normalized
  /// log-probabilities**: frame `row`'s `logsumexp` over the vocab axis is
  /// `logsumexp`, exceeding [`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`] in
  /// magnitude. A genuine CTC log-probability frame sums to 1 in probability
  /// space, so its `logsumexp` is `0` (`ln Σ exp(log p_j) = ln Σ p_j = ln 1`); a
  /// whole-unit deviation means the tensor carries raw logits — or another
  /// un-normalized distribution — not the log-softmaxed output this crate's
  /// encoder contract requires.
  ///
  /// Unlike [`Self::CorruptEmissions`] — a placement-dependent fp16 underflow of
  /// THIS reviewed artifact — this is a **model-artifact contract** failure that
  /// no placement causes and no placement cures: a revision shipping a raw-logit
  /// CTC head (the *standard* wav2vec2 export, and what asry's own ONNX model
  /// emits) is rejected here rather than silently re-normalized by
  /// `Emissions::from_logits` and aligned on forever. It is the check that makes
  /// [`crate::audio::align::encode::Encoder::emissions`]'s "these really are log-probs" a
  /// verified contract for any same-contract artifact loaded through the public
  /// API, not merely for the one reviewed here. The finite ∧ `<= 0` scan
  /// `Emissions::from_log_probs` runs cannot catch it: raw logits shifted wholly
  /// into `[-20, -10]`, or an all-zeros frame, are finite and `<= 0` on every
  /// cell yet no distribution at all. See
  /// [`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`] for the measured tolerance and the
  /// [`crate::audio::align::encode`] module doc's "The normalization guard".
  #[error(
    "encoder emissions are not normalized log-probabilities: frame {} has logsumexp \
     {} over the vocab axis (tolerance ±{}), but a CTC log-probability frame \
     sums to 1 in probability space so its logsumexp is 0. This emission tensor carries raw \
     logits or another un-normalized distribution, not the log-softmaxed output alignkit's \
     encoder contract requires — most likely the model artifact was swapped for a raw-logit CTC \
     head (the standard wav2vec2 export). The encoder was scheduled on {:?}.",
    .0.row(),
    .0.logsumexp(),
    .0.tolerance(),
    .0.compute()
  )]
  UnnormalizedEmissions(UnnormalizedEmissions),
  /// A caller-supplied OOV decision does not carry the requested language.
  ///
  /// Returned by [`crate::audio::align::registry::AlignmentSet::align_chunk`] when the
  /// `ResolvedOov` at position `index` carries `found` rather than the
  /// `requested` language the chunk is being aligned for. The registry checks
  /// this BEFORE crossing the decisions into an
  /// [`AlignerKey::Any`](crate::audio::align::registry::AlignerKey::Any) fallback aligner's
  /// own language: a foreign-language decision would otherwise be re-stamped and
  /// silently apply another language's wildcard / fail-closed policy at a
  /// matching position (asry's `ResolvedOov` identity ignores language on
  /// purpose, so nothing downstream would catch it). Resolve decisions against
  /// the SAME language you pass to `align_chunk` — the one
  /// [`AlignmentSet::detect_oov`](crate::audio::align::registry::AlignmentSet::detect_oov)
  /// stamped them with.
  #[error(
    "oov_decisions[{}] carries language {:?} but the chunk is being aligned for \
     {:?}; resolve the decisions against the language you request (the one \
     `AlignmentSet::detect_oov` stamped them with)",
    .0.index(),
    .0.found(),
    .0.requested()
  )]
  DecisionLanguage(DecisionLanguage),
  /// No aligner is registered for the requested language, no
  /// [`AlignerKey::Any`](crate::audio::align::registry::AlignerKey::Any) fallback exists, and
  /// the registry's miss policy is
  /// [`AlignmentFallback::Error`](crate::audio::align::registry::AlignmentFallback).
  ///
  /// Returned by [`crate::audio::align::registry::AlignmentSet::align_chunk`]. Under the
  /// default `SkipChunk` policy a miss instead yields an empty alignment result
  /// (the ASR text survives, only per-word timings are dropped); this variant is
  /// the opt-in loud form, for a pipeline that wants a missing language to stop
  /// it rather than pass silently.
  ///
  /// Carries the requested language with no registered aligner and no `Any`
  /// fallback.
  #[error(
    "no aligner registered for language {0:?}, no `Any` fallback, and the registry miss \
     policy is `Error`"
  )]
  LanguageUnsupported(asry::Lang),
}

#[cfg(test)]
mod tests;
