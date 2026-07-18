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
//!   [`crate::audio::align::encode::Encoder::emissions`] and
//!   [`crate::audio::align::aligner::Aligner::align_chunk`], which sit at the same "one
//!   chunk's worth of work" layer.
//!
//! # The recoverable subset lives in `Aligner`, not here
//!
//! Two of the seam's [`asry::emissions::EmissionsError`] variants —
//! `NoAlignmentPath` and `SemanticOutOfVocab` — are *recoverable*: a chunk
//! that hits them yields an empty `AlignmentResult` (the ASR text is kept,
//! only per-word timings are dropped), not a hard error. That mapping is a
//! policy of [`crate::audio::align::aligner::Aligner::align_chunk`], which converts those
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
  Load(#[from] crate::LoadError),
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
  /// [`crate::audio::align::aligner::Aligner::from_paths`].
  #[error("alignment seam construction failed: {0}")]
  Seam(#[from] asry::emissions::EmissionsError),
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
  /// from the encoder, tokenization, or the trellis. The *recoverable*
  /// subset (`NoAlignmentPath`, `SemanticOutOfVocab`) never reaches here;
  /// see the module doc.
  #[error(transparent)]
  Alignment(#[from] asry::emissions::EmissionsError),
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
  #[error("input exceeds encoder window: {got} samples > {max} samples")]
  InputTooLong {
    /// Samples the caller supplied.
    got: usize,
    /// The encoder's fixed window size
    /// ([`crate::audio::align::encode::ENCODER_WINDOW_SAMPLES`]).
    max: usize,
  },
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
    "encoder emissions are not log-probabilities: {cells} of {total} cells are below {floor} \
     (min = {min}), the fp16 `log(0)` saturation sentinel. The encoder was scheduled on \
     {compute:?}: this model's fp16 `log(softmax(·))` tail underflows on the Apple Neural \
     Engine and its word timings shift by hundreds of milliseconds. Load the encoder on \
     `alignkit::encode::DEFAULT_ENCODER_COMPUTE` (the default, and the fastest correct \
     placement) — or re-convert the model with a fused `log_softmax` tail.",
    floor = crate::audio::align::encode::LOG_PROB_FLOOR,
  )]
  CorruptEmissions {
    /// The compute placement the encoder was loaded on — the knob the caller
    /// can actually turn, hence the one the message names.
    compute: crate::ComputeUnits,
    /// The most negative cell in the matrix (`≈ -45440` on an ANE placement;
    /// `-30.81` on the `CpuOnly` default, measured on `jfk.wav`).
    min: f32,
    /// How many cells fell below [`crate::audio::align::encode::LOG_PROB_FLOOR`] (2,667 on
    /// `jfk.wav`'s ANE run).
    cells: usize,
    /// Cells scanned: `frames × `[`crate::audio::align::vocab::VOCAB_SIZE`] (15,921 on
    /// `jfk.wav`).
    total: usize,
  },
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
    "encoder emissions are not normalized log-probabilities: frame {row} has logsumexp \
     {logsumexp} over the vocab axis (tolerance ±{tolerance}), but a CTC log-probability frame \
     sums to 1 in probability space so its logsumexp is 0. This emission tensor carries raw \
     logits or another un-normalized distribution, not the log-softmaxed output alignkit's \
     encoder contract requires — most likely the model artifact was swapped for a raw-logit CTC \
     head (the standard wav2vec2 export). The encoder was scheduled on {compute:?}."
  )]
  UnnormalizedEmissions {
    /// The compute placement the encoder was loaded on. Carried for parity with
    /// [`Self::CorruptEmissions`]; unlike that error the placement is not the
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
  },
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
  /// [`AlignerKey::Any`](crate::audio::align::registry::AlignerKey::Any) fallback exists, and
  /// the registry's miss policy is
  /// [`AlignmentFallback::Error`](crate::audio::align::registry::AlignmentFallback).
  ///
  /// Returned by [`crate::audio::align::registry::AlignmentSet::align_chunk`]. Under the
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
