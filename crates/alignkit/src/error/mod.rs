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
  /// The encoder returned an emission matrix that is **not log-probabilities**:
  /// at least one cell sits below [`crate::encode::LOG_PROB_FLOOR`], the fp16
  /// `log(0)` saturation sentinel (`≈ -45440`).
  ///
  /// This is the loud form of what used to be a silent one. The values are
  /// finite and negative, so they pass `Emissions::from_log_probs`' own
  /// `finite ∧ <= 0` scan untouched and would align to *plausible, wrong*
  /// timings (measured: `ask` 881 ms early on `jfk.wav`) — which is why the
  /// floor is checked separately. See
  /// [`crate::encode::DEFAULT_ENCODER_COMPUTE`] for the mechanism and
  /// [`crate::encode::LOG_PROB_FLOOR`] for why the guard keys on the value
  /// domain rather than on the compute placement.
  ///
  /// It is a defect of the **model artifact**, not of the caller's audio: no
  /// input can make a correctly-converted artifact produce it, and no input
  /// avoids it on a corrupted one (though only real speech reaches the regime
  /// that exposes it — synthetic silence and tones never drive a class
  /// posterior under the fp16 floor). The fix is the placement named in this
  /// error, or a re-converted model; nothing in this crate can recover the
  /// underflowed cells.
  #[error(
    "encoder emissions are not log-probabilities: {cells} of {total} cells are below {floor} \
     (min = {min}), the fp16 `log(0)` saturation sentinel. The encoder was scheduled on \
     {compute:?}: this model's fp16 `log(softmax(·))` tail underflows on the Apple Neural \
     Engine and its word timings shift by hundreds of milliseconds. Load the encoder on \
     `alignkit::encode::DEFAULT_ENCODER_COMPUTE` (the default, and the fastest correct \
     placement) — or re-convert the model with a fused `log_softmax` tail.",
    floor = crate::encode::LOG_PROB_FLOOR,
  )]
  CorruptEmissions {
    /// The compute placement the encoder was loaded on — the knob the caller
    /// can actually turn, hence the one the message names.
    compute: coremlit::ComputeUnits,
    /// The most negative cell in the matrix (`≈ -45440` on an ANE placement;
    /// `-30.81` on the `CpuOnly` default, measured on `jfk.wav`).
    min: f32,
    /// How many cells fell below [`crate::encode::LOG_PROB_FLOOR`] (2,667 on
    /// `jfk.wav`'s ANE run).
    cells: usize,
    /// Cells scanned: `frames × `[`crate::vocab::VOCAB_SIZE`] (15,921 on
    /// `jfk.wav`).
    total: usize,
  },
  /// A caller-supplied OOV decision does not carry the requested language.
  ///
  /// Returned by [`crate::registry::AlignmentSet::align_chunk`] when the
  /// `ResolvedOov` at position `index` carries `found` rather than the
  /// `requested` language the chunk is being aligned for. The registry checks
  /// this BEFORE crossing the decisions into an
  /// [`AlignerKey::Any`](crate::registry::AlignerKey::Any) fallback aligner's
  /// own language: a foreign-language decision would otherwise be re-stamped and
  /// silently apply another language's wildcard / fail-closed policy at a
  /// matching position (asry's `ResolvedOov` identity ignores language on
  /// purpose, so nothing downstream would catch it). Resolve decisions against
  /// the SAME language you pass to `align_chunk` — the one
  /// [`AlignmentSet::detect_oov`](crate::registry::AlignmentSet::detect_oov)
  /// stamped them with.
  #[error(
    "oov_decisions[{index}] carries language {found:?} but the chunk is being aligned for \
     {requested:?}; resolve the decisions against the language you request (the one \
     `AlignmentSet::detect_oov` stamped them with)"
  )]
  DecisionLanguage {
    /// Index of the offending decision in the caller's `oov_decisions` slice.
    index: usize,
    /// The language the chunk is being aligned for (the `align_chunk` argument).
    requested: asry::Lang,
    /// The language the decision actually carries.
    found: asry::Lang,
  },
  /// No aligner is registered for the requested language, no
  /// [`AlignerKey::Any`](crate::registry::AlignerKey::Any) fallback exists, and
  /// the registry's miss policy is
  /// [`AlignmentFallback::Error`](crate::registry::AlignmentFallback).
  ///
  /// Returned by [`crate::registry::AlignmentSet::align_chunk`]. Under the
  /// default `SkipChunk` policy a miss instead yields an empty alignment result
  /// (the ASR text survives, only per-word timings are dropped); this variant is
  /// the opt-in loud form, for a pipeline that wants a missing language to stop
  /// it rather than pass silently.
  #[error(
    "no aligner registered for language {language:?}, no `Any` fallback, and the registry miss \
     policy is `Error`"
  )]
  LanguageUnsupported {
    /// The requested language with no registered aligner and no `Any` fallback.
    language: asry::Lang,
  },
}

#[cfg(test)]
mod tests;
