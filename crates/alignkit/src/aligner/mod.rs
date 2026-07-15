//! The public forced aligner: [`Aligner`] wraps `asry`'s
//! [`EmissionsAligner`] around alignkit's
//! CoreML [`Encoder`] and drives one chunk
//! end-to-end — VAD → `prepare` → CoreML encode → `finish` — into per-word
//! [`TimeRange`]s.
//!
//! # What the aligner no longer owns
//!
//! Everything except the CoreML encoder. The redesigned asry seam
//! ([`EmissionsAligner`]) owns the
//! tokenizer, the normalizer, the CTC blank id, the vocab-size handshake,
//! the silence mask, and every validator; alignkit hands it exactly one
//! thing it cannot compute — the emissions — and reads back the words. So
//! this type is thin: an [`Encoder`], the seam, and
//! the [`AlignerOptions`] baked into that seam at construction.

use core::{num::NonZeroU32, sync::atomic::AtomicBool, time::Duration};
use std::path::Path;

use asry::{
  AlignmentResult, Lang, TimeRange,
  emissions::{
    DynTextNormalizer, EmissionsAligner, EmissionsError, OovEvent, OutputClock, ResolvedOov,
    SpeechCoverage, SpeechSpans,
  },
};
use coremlit::ComputeUnits;

use crate::{
  encode::{DEFAULT_ENCODER_COMPUTE, Encoder, EncoderInput, EncoderOptions},
  error::{AlignError, AlignerError},
};

/// The frame stride handed to asry's seam, in 16 kHz samples — the SAME
/// number [`Encoder`]'s truncation formula divides by, re-typed as the
/// [`NonZeroU32`] the seam builder wants.
///
/// Derived from [`crate::encode::HOP_SAMPLES`] rather than re-spelled as
/// `320`, so the stride that TRUNCATES the emissions and the stride that
/// TIMES the words are one constant and cannot drift apart. It is private and
/// there is no option to override it: this crate wraps exactly one model,
/// whose stride is fixed at 320 by its graph, so no other value is ever
/// correct.
///
/// This was a caller-settable `AlignerOptions::hop_samples` knob, which was a
/// latent corruption: it reached only the seam (the timing half), never the
/// encoder (`truncated_frame_count` divides by the hardcoded
/// [`crate::encode::HOP_SAMPLES`]). asry's own `validate_stride_extent` slack
/// is `chunk_extent ± 2·hop`, far too loose to catch the skew — measured on
/// `jfk.wav`, a hop of 319, 320 or 321 all returned `Ok` with 22 words and no
/// error, so a caller setting 319 got words timed at the wrong stride,
/// silently. asry can afford the knob because its `T` comes from the ONNX
/// model's own output shape, making `hop_samples` the single place stride is
/// declared there; alignkit computes `T` itself, so a second declaration is a
/// second source of truth. Pinned by
/// `tests::seam_stride_is_the_encoder_stride`.
const SEAM_HOP_SAMPLES: NonZeroU32 = match NonZeroU32::new(crate::encode::HOP_SAMPLES as u32) {
  Some(v) => v,
  None => unreachable!(),
};

/// Default minimum speech coverage a word must clear to survive (`0.5`) —
/// asry's [`SpeechCoverage::DEFAULT`](asry::emissions::SpeechCoverage::DEFAULT).
pub const DEFAULT_MIN_SPEECH_COVERAGE: f32 = SpeechCoverage::DEFAULT.get();

/// Default maximum contiguous silent run tolerated inside a word's span
/// (80 ms) — asry's `DEFAULT_MAX_INTRA_SILENT_RUN`.
pub const DEFAULT_MAX_INTRA_SILENT_RUN: Duration = asry::emissions::DEFAULT_MAX_INTRA_SILENT_RUN;

#[cfg(feature = "serde")]
fn default_min_speech_coverage() -> f32 {
  DEFAULT_MIN_SPEECH_COVERAGE
}
#[cfg(feature = "serde")]
fn default_max_intra_silent_run() -> Duration {
  DEFAULT_MAX_INTRA_SILENT_RUN
}
#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_ENCODER_COMPUTE
}

/// Construction options for [`Aligner`] (rust-options-pattern): the two seam
/// knobs asry's [`EmissionsAligner`] builder exposes that have a meaningful
/// range for this model, plus the CoreML compute placement handed to the
/// [`Encoder`].
///
/// Deliberately NOT here: `hop_samples`. The seam builder accepts one, but
/// this crate wraps a single model whose stride is fixed at 320 by its graph
/// ([`crate::encode::HOP_SAMPLES`]), and the encoder's own truncation divides
/// by that same constant without consulting any option — so a caller-set
/// stride would apply to only half the pipeline and skew every word. The seam
/// is wired from that one constant instead (`SEAM_HOP_SAMPLES`, private).
///
/// These are **construction-time**: they are fed to the builder / model load
/// once and baked in, so there are no post-construction setters on
/// [`Aligner`] — rebuild via [`Aligner::from_paths_with`] to change them.
/// (The `with_`/`set_` pairs here mutate an `AlignerOptions` *value* before
/// it reaches construction.)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlignerOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_min_speech_coverage"))]
  min_speech_coverage: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_max_intra_silent_run"))]
  max_intra_silent_run: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_compute", with = "crate::compute_units_serde")
  )]
  compute: ComputeUnits,
}

impl Default for AlignerOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl AlignerOptions {
  /// Options matching the crate defaults: [`DEFAULT_MIN_SPEECH_COVERAGE`],
  /// [`DEFAULT_MAX_INTRA_SILENT_RUN`], [`DEFAULT_ENCODER_COMPUTE`].
  #[must_use]
  pub const fn new() -> Self {
    Self {
      min_speech_coverage: DEFAULT_MIN_SPEECH_COVERAGE,
      max_intra_silent_run: DEFAULT_MAX_INTRA_SILENT_RUN,
      compute: DEFAULT_ENCODER_COMPUTE,
    }
  }

  /// Minimum speech coverage a word must clear to survive.
  ///
  /// Coerced through
  /// [`SpeechCoverage::clamped`](asry::emissions::SpeechCoverage::clamped) at
  /// construction (`NaN` → default, out-of-range clamps to `[0, 1]`), so a
  /// bad value here can never silently disable the coverage filter.
  #[must_use]
  pub const fn min_speech_coverage(&self) -> f32 {
    self.min_speech_coverage
  }
  /// Builder form of [`Self::set_min_speech_coverage`].
  #[must_use]
  pub const fn with_min_speech_coverage(mut self, coverage: f32) -> Self {
    self.set_min_speech_coverage(coverage);
    self
  }
  /// Sets [`Self::min_speech_coverage`] in place.
  pub const fn set_min_speech_coverage(&mut self, coverage: f32) -> &mut Self {
    self.min_speech_coverage = coverage;
    self
  }

  /// Maximum contiguous silent run tolerated inside a word's span.
  #[must_use]
  pub const fn max_intra_silent_run(&self) -> Duration {
    self.max_intra_silent_run
  }
  /// Builder form of [`Self::set_max_intra_silent_run`].
  #[must_use]
  pub const fn with_max_intra_silent_run(mut self, run: Duration) -> Self {
    self.set_max_intra_silent_run(run);
    self
  }
  /// Sets [`Self::max_intra_silent_run`] in place.
  pub const fn set_max_intra_silent_run(&mut self, run: Duration) -> &mut Self {
    self.max_intra_silent_run = run;
    self
  }

  /// Which hardware CoreML may schedule the encoder on. Defaults to
  /// [`DEFAULT_ENCODER_COMPUTE`] (`ComputeUnits::CpuOnly`).
  ///
  /// **Overriding this to an ANE placement (`ComputeUnits::All` or
  /// `CpuAndNeuralEngine`) corrupts the emissions** — the model's fp16
  /// `log(softmax(·))` tail underflows to a `-45440` sentinel on 16.7% of cells
  /// and shifts real word timings by hundreds of milliseconds. That is a
  /// property of the model artifact, not of this crate, and nothing here can
  /// recover the underflowed cells.
  ///
  /// It is not silent: [`Aligner::align_chunk`] fails such a chunk with
  /// [`AlignError::CorruptEmissions`], which names this placement (see
  /// [`crate::encode::LOG_PROB_FLOOR`]). The guard is on the emission VALUES,
  /// not on the placement, so a numerically-clean non-default placement —
  /// `CpuAndGpu`, measured `min = -30.02` — still works. There is simply
  /// nothing to buy: `CpuOnly` is also the fastest correct placement. Read
  /// [`DEFAULT_ENCODER_COMPUTE`]'s doc before changing this.
  #[must_use]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// Build asry's [`EmissionsAligner`] the
/// way [`Aligner::from_paths_with`] does: bundled 29-class chordai
/// tokenizer, the MANDATORY explicit blank id, the model's fixed stride, and
/// `options` fed to the builder.
///
/// Factored out of [`Aligner::from_paths_with`] so the wiring — above all
/// the blank-id override and the stride — is unit-testable without a CoreML
/// model.
fn build_seam(
  language: Lang,
  normalizer: DynTextNormalizer,
  options: &AlignerOptions,
) -> Result<EmissionsAligner, EmissionsError> {
  EmissionsAligner::builder(language, crate::vocab::tokenizer_json_bytes())
    .normalizer(normalizer)
    // NOT an option (see `SEAM_HOP_SAMPLES`): the stride that times the words
    // here must be the stride that truncates the emissions in
    // `Encoder::emissions`, and asry's `chunk_extent ± 2·hop` validator is too
    // loose to catch them disagreeing.
    .hop_samples(SEAM_HOP_SAMPLES)
    .min_speech_coverage(SpeechCoverage::clamped(options.min_speech_coverage()))
    .max_intra_silent_run(options.max_intra_silent_run())
    // MANDATORY (DECISION 5): the chordai vocab's blank is `"-"`@0 and there
    // is no `<pad>` / `[PAD]` / `<blank>` entry, so the builder's default
    // auto-detect would FAIL construction. Pass the id explicitly.
    .blank_token_id(crate::vocab::BLANK_ID)
    .build()
}

/// Per-language forced aligner over the CoreML wav2vec2 encoder.
///
/// Wraps alignkit's CoreML [`Encoder`] (its head
/// width validated `== `[`VOCAB_SIZE`](crate::vocab::VOCAB_SIZE) at load),
/// asry's [`EmissionsAligner`] seam, and
/// the [`AlignerOptions`] baked into that seam. Build one per language with
/// [`from_paths`](Self::from_paths), then drive it per chunk with
/// [`align_chunk`](Self::align_chunk).
///
/// [`align_chunk`](Self::align_chunk) takes `&self`: the CoreML `Model`
/// predicts without `&mut`, and asry's `prepare`/`finish` are `&self`, so —
/// unlike asry's own ORT `Aligner`, which its registry wraps in a `Mutex` —
/// this one needs no interior mutability.
pub struct Aligner {
  encoder: crate::encode::Encoder,
  inner: EmissionsAligner,
  options: AlignerOptions,
}

impl Aligner {
  /// Load an aligner for `language` from the compiled CoreML model at
  /// `model_path`, using the crate's **bundled** 29-class chordai tokenizer
  /// ([`crate::vocab::tokenizer_json_bytes`]) and the default
  /// [`AlignerOptions`].
  ///
  /// There is no `tokenizer_path` parameter: alignkit wraps exactly one
  /// model whose only correct tokenizer is the bundled asset (any other
  /// vocab would fail the CTC-head handshake), and `crate::vocab`'s own
  /// "bytes, not a path" decision documents why a filesystem tokenizer path
  /// is the wrong shape for a packaged consumer. To align with a different
  /// tokenizer, build an [`EmissionsAligner`]
  /// directly.
  ///
  /// # Errors
  /// [`AlignerError::Load`] / [`AlignerError::ContractMismatch`] if CoreML
  /// rejects the model or its I/O contract disagrees with the pinned one;
  /// [`AlignerError::Seam`] if asry's builder rejects the bundled tokenizer
  /// or the normalizer (e.g. a normalizer that needs a `|` delimiter).
  pub fn from_paths(
    language: Lang,
    model_path: &Path,
    normalizer: DynTextNormalizer,
  ) -> Result<Self, AlignerError> {
    Self::from_paths_with(language, model_path, normalizer, AlignerOptions::new())
  }

  /// [`Self::from_paths`] with explicit [`AlignerOptions`].
  ///
  /// With the `tracing` feature: an `alignkit.aligner.load` span at `INFO`,
  /// with the CoreML load (`alignkit.encoder.load`) nested inside it.
  ///
  /// # Errors
  /// As [`Self::from_paths`].
  #[cfg_attr(
    feature = "tracing",
    tracing::instrument(
      name = "alignkit.aligner.load",
      level = "info",
      skip_all,
      fields(
        language = ?language,
        model_path = ?model_path,
        compute = ?options.compute(),
      ),
    )
  )]
  pub fn from_paths_with(
    language: Lang,
    model_path: &Path,
    normalizer: DynTextNormalizer,
    options: AlignerOptions,
  ) -> Result<Self, AlignerError> {
    let encoder = Encoder::from_file_with(
      model_path,
      EncoderOptions::new().with_compute(options.compute()),
    )?;
    let inner = build_seam(language, normalizer, &options)?;
    // Handshake: the bundled tokenizer's vocab must equal the CTC head width
    // the encoder validated at load. Both are VOCAB_SIZE for the pinned
    // model + asset, so this never fires in practice; `finish` re-runs it
    // per chunk (as `EmissionsError::VocabMismatch`) as the real enforcement
    // for any future mismatched build. Cheap startup sanity, not the guard.
    debug_assert_eq!(
      inner.vocab_size().get(),
      crate::vocab::VOCAB_SIZE,
      "bundled tokenizer vocab must equal the CTC head width"
    );
    Ok(Self {
      encoder,
      inner,
      options,
    })
  }

  /// The language this aligner was built for.
  #[must_use]
  pub const fn language_ref(&self) -> &Lang {
    self.inner.language()
  }

  /// The [`AlignerOptions`] baked into this aligner's seam.
  #[must_use]
  pub const fn options(&self) -> AlignerOptions {
    self.options
  }

  /// The audio sample rate this aligner expects: 16 kHz, asry's analysis
  /// rate. Callers resample to this first.
  #[must_use]
  pub const fn sample_rate(&self) -> u32 {
    asry::time::SAMPLE_RATE_HZ
  }

  /// Detect out-of-vocabulary characters in `text`, as data — no policy
  /// decision is made.
  ///
  /// Resolve the returned events with
  /// [`default_oov_decisions`](asry::emissions::default_oov_decisions) (or
  /// [`wildcard_all_decisions`](asry::emissions::wildcard_all_decisions),
  /// [`fail_closed_all_decisions`](asry::emissions::fail_closed_all_decisions),
  /// or your own policy), then pass the result to
  /// [`align_chunk`](Self::align_chunk). Events are returned in the order the
  /// tokenizer encounters them; a `&[ResolvedOov]` handed to `align_chunk`
  /// must be in the same order.
  ///
  /// # Errors
  /// [`AlignError::Alignment`] on a text-normalizer or tokenizer-engine
  /// failure. Punctuation-only input yields an empty vec, not an error.
  pub fn detect_oov(&self, text: &str) -> Result<Vec<OovEvent>, AlignError> {
    Ok(self.inner.detect_oov(text)?)
  }

  /// Align one chunk end-to-end into per-word [`TimeRange`]s in `clock`'s
  /// output timebase.
  ///
  /// - `samples`: the chunk's 16 kHz f32 mono audio, at most
  ///   [`ENCODER_WINDOW_SAMPLES`](crate::encode::ENCODER_WINDOW_SAMPLES).
  /// - `sub_segments`: VAD speech spans in the chunk-local 1/16000 analysis
  ///   timebase. **Empty means "no VAD"** →
  ///   [`SpeechSpans::all_speech`](asry::emissions::SpeechSpans::all_speech),
  ///   not "all silence" (which would drop every word).
  /// - `text`: the transcript to align against `samples`.
  /// - `clock`: how stream sample indices map back to output-timebase
  ///   ranges; build with
  ///   [`OutputClock::new`](asry::emissions::OutputClock::new). This is
  ///   `asry`'s replacement for the old `Fn(u64, u64) -> TimeRange` closure.
  /// - `abort_flag`: cooperative cancellation, polled throughout `prepare`
  ///   and `finish`.
  /// - `oov_decisions`: caller-resolved decisions for the events
  ///   [`Self::detect_oov`] reported, in that same order.
  ///
  /// A trivial chunk (text that normalises to nothing / yields no tokens)
  /// and the recoverable seam failures (`NoAlignmentPath`,
  /// `SemanticOutOfVocab`) both return an **empty** [`AlignmentResult`]: the
  /// ASR text survives, only per-word timings are dropped. See the
  /// [`crate::error`] module doc.
  ///
  /// With the `tracing` feature: one `alignkit.align_chunk` span at `DEBUG` per
  /// call, wrapping the whole VAD → prepare → encode → finish pass, with
  /// `alignkit.encoder.emissions` nested inside it. Both of the empty-result
  /// paths above are *successes* that produce no words, which is exactly the
  /// state a caller ends up staring at a debugger over — the span's
  /// `sub_segments` / `text_bytes` / `samples` fields are there to tell those
  /// two apart from a chunk that simply had nothing in it.
  ///
  /// # Errors
  /// [`AlignError::InputTooLong`] if `samples` exceeds the encoder window;
  /// [`AlignError::Span`] if `sub_segments` are not in the 1/16000 timebase;
  /// [`AlignError::Prediction`] / [`AlignError::Tensor`] from the CoreML
  /// encode; [`AlignError::CorruptEmissions`] if the encoder's emission matrix
  /// left the log-probability domain (an ANE placement set through
  /// [`AlignerOptions::with_compute`] — see
  /// [`crate::encode::LOG_PROB_FLOOR`]); [`AlignError::Alignment`] for any
  /// non-recoverable seam failure (stride / vocab / blank-id validation, a
  /// non-finite or positive log-probability, tokenization, abort).
  #[cfg_attr(
    feature = "tracing",
    tracing::instrument(
      name = "alignkit.align_chunk",
      level = "debug",
      skip_all,
      fields(
        language = ?self.language_ref(),
        samples = samples.len(),
        sub_segments = sub_segments.len(),
        text_bytes = text.len(),
        oov_decisions = oov_decisions.len(),
      ),
    )
  )]
  pub fn align_chunk(
    &self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    clock: OutputClock,
    abort_flag: &AtomicBool,
    oov_decisions: &[ResolvedOov],
  ) -> Result<AlignmentResult, AlignError> {
    if samples.len() > crate::encode::ENCODER_WINDOW_SAMPLES {
      return Err(AlignError::InputTooLong {
        got: samples.len(),
        max: crate::encode::ENCODER_WINDOW_SAMPLES,
      });
    }

    let speech = if sub_segments.is_empty() {
      SpeechSpans::all_speech()
    } else {
      SpeechSpans::from_time_ranges(sub_segments)?
    };

    let prepared = match self
      .inner
      .prepare(samples, &speech, text, oov_decisions, abort_flag)
    {
      Ok(prepared) => prepared,
      Err(err) => return recover_or_error(err),
    };
    if prepared.is_trivial() {
      return Ok(AlignmentResult::new(Vec::new()));
    }

    // asry has already silence-masked + receptive-field-padded the buffer; the
    // encoder consumes exactly THAT, and the truncation formula needs the real
    // sample count, which alignkit owns (asry keeps its own
    // `PreparedChunk::real_samples` crate-private). Binding the padded buffer to
    // the unpadded `samples.len()` in one `EncoderInput` is what makes a
    // mismatched real length unrepresentable (F1): the guarded pipeline can only
    // pass the length it actually prepared.
    let input = EncoderInput::from_prepared(prepared.encoder_input(), samples)?;
    let emissions = self.encoder.emissions(input)?;

    match self.inner.finish(prepared, &emissions, clock, abort_flag) {
      Ok(result) => Ok(result),
      Err(err) => recover_or_error(err),
    }
  }
}

/// The seam's recoverable subset → empty words (ASR text preserved); every
/// other failure → a hard [`AlignError`].
///
/// Mirrors asry's `alignment_failure_is_recoverable`
/// (`asry/src/runner/alignment_pool/mod.rs`): `NoAlignmentPath` (too-short
/// chunk / lattice-budget overflow / no finite path) and `SemanticOutOfVocab`
/// (a pronounced OOV symbol resolved fail-closed) are data-dependent
/// per-chunk misses, not broken-setup errors. asry's third recoverable case,
/// `EmptyText`, cannot arise here — empty / untokenizable text is
/// `PreparedChunk::is_trivial()`, handled before the encoder.
fn recover_or_error(err: EmissionsError) -> Result<AlignmentResult, AlignError> {
  match err {
    EmissionsError::NoAlignmentPath(_) | EmissionsError::SemanticOutOfVocab(_) => {
      Ok(AlignmentResult::new(Vec::new()))
    }
    other => Err(AlignError::Alignment(other)),
  }
}

#[cfg(test)]
mod tests;
