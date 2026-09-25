//! The public forced aligner: [`Aligner`] wraps `asry`'s
//! [`EmissionsAligner`] around alignkit's
//! CoreML [`Encoder`] and drives one chunk
//! end-to-end — VAD → `prepare` → CoreML encode → `finish` — into per-word
//! [`TimeRange`]s.
//!
//! # What the aligner no longer owns
//!
//! Everything except the CoreML encoder and the pairing of the encoder with
//! a vocabulary. The redesigned asry seam ([`EmissionsAligner`]) owns the
//! tokenizer, the normalizer, the CTC blank id, the per-chunk vocab-size
//! handshake, the silence mask, and every validator; alignkit hands it
//! exactly one thing it cannot compute — the emissions — and reads back the
//! words. So this type is thin: an [`Encoder`], the seam built from a
//! [`Vocabulary`], and the [`AlignerOptions`] baked into that seam at
//! construction.

use core::{
  num::{NonZeroU32, NonZeroUsize},
  sync::atomic::AtomicBool,
  time::Duration,
};
use std::path::Path;

use crate::ComputeUnits;
use asry::{
  AlignmentResult, Lang, TimeRange,
  emissions::{
    DynTextNormalizer, EmissionsAligner, EmissionsError, OovDecision, OovEvent, OutputClock,
    ResolvedOov, SpeechCoverage, SpeechSpans,
  },
};

use crate::audio::align::{
  encode::{DEFAULT_ENCODER_COMPUTE, Encoder, EncoderInput, EncoderOptions},
  error::{AlignError, AlignerError, InputTooLong, Refusal, VocabularyMismatch},
  vocab::Vocabulary,
};

/// The frame stride handed to asry's seam, in 16 kHz samples — the SAME
/// number [`Encoder`]'s truncation formula divides by, re-typed as the
/// [`NonZeroU32`] the seam builder wants.
///
/// Derived from [`crate::audio::align::encode::HOP_SAMPLES`] rather than re-spelled as
/// `320`, so the stride that TRUNCATES the emissions and the stride handed to
/// the seam are one constant and cannot drift apart. That truncation stride is
/// also what TIMES the words: it fixes the frame count `T`, on which asry maps
/// boundaries by the effective `n_samples / (T - 1)` ratio (~20 ms; see
/// `tests/parity_words.rs`), not a literal 320-sample grid. It is private and
/// there is no option to override it: this crate wraps exactly one model,
/// whose stride is fixed at 320 by its graph, so no other value is ever
/// correct.
///
/// This was a caller-settable `AlignerOptions::hop_samples` knob, which was a
/// latent corruption: it reached only the seam (its stride check and `T < 2`
/// fallback), never the encoder (`truncated_frame_count` divides by the
/// hardcoded [`crate::audio::align::encode::HOP_SAMPLES`]). At `T >= 2` that never re-timed
/// the words — asry maps boundaries by the effective `n_samples / (T - 1)`
/// ratio, driven by the encoder's frame count, not the seam's hop — but it let
/// the seam DECLARE a stride the encoder never used, which asry's own
/// `validate_stride_extent` slack (`chunk_extent ± 2·hop`) is far too loose to
/// reject: on `jfk.wav`, a hop of 319, 320 or 321 all returned `Ok` with 22
/// words and no error. asry can afford the knob because its `T` comes from the ONNX
/// model's own output shape, making `hop_samples` the single place stride is
/// declared there; alignkit computes `T` itself, so a second declaration is a
/// second source of truth. Pinned by
/// `tests::seam_stride_is_the_encoder_stride`.
const SEAM_HOP_SAMPLES: NonZeroU32 =
  match NonZeroU32::new(crate::audio::align::encode::HOP_SAMPLES as u32) {
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
/// ([`crate::audio::align::encode::HOP_SAMPLES`]), and the encoder's own truncation divides
/// by that same constant without consulting any option — so a caller-set
/// stride would reach only the seam half, declaring a stride the encoder never
/// used: a second, driftable source of truth (at `T >= 2` it would not even
/// move the boundaries, which asry maps by the encoder-driven
/// `n_samples / (T - 1)` ratio — see `SEAM_HOP_SAMPLES`). The seam is wired
/// from that one constant instead (`SEAM_HOP_SAMPLES`, private).
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
  #[cfg_attr(feature = "serde", serde(default = "default_compute"))]
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
  /// It is not silent on the audio that exposes it: [`Aligner::align_chunk`]
  /// fails a real-speech chunk on an ANE placement with
  /// [`AlignError::CorruptEmissions`], which names this placement (see
  /// [`crate::audio::align::encode::LOG_PROB_FLOOR`]). Detection is input-dependent — the
  /// `log(0)` sentinel only appears once a class posterior falls under the fp16
  /// floor, so a pure-silence or low-tone chunk can pass even here. Real speech
  /// can expose it, measured on `jfk.wav`. The guard is on the emission VALUES,
  /// not the input category or the placement, so a numerically-clean non-default
  /// placement — `CpuAndGpu`,
  /// measured `min = -30.02` — still works. There is simply nothing to buy:
  /// `CpuOnly` is also the fastest correct placement. Read
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

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a major bump.**
///
/// `key=value` pairs in declaration order, joined by `,`: [`Self::min_speech_coverage`]
/// as `{}` of `f32` (the shortest round-tripping repr); [`Self::max_intra_silent_run`]
/// as [`humantime::format_duration`]'s text (`"80ms"`, `"1s"`, …) — the same
/// grammar this workspace already renders `Duration` as elsewhere (ingraph's
/// `[registry]` seat); [`Self::compute`] composing [`ComputeUnits`]'s own
/// `Display` verbatim (its snake_case wire word, e.g. `"cpu_only"`) — that is
/// the one place ITS spelling can change, and a drift there fails the same
/// pinning test this one does.
///
/// For example, `AlignerOptions::new()` prints
/// `min_speech_coverage=0.5,max_intra_silent_run=80ms,compute=cpu_only`.
impl core::fmt::Display for AlignerOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "min_speech_coverage={},max_intra_silent_run={},compute={}",
      self.min_speech_coverage,
      humantime::format_duration(self.max_intra_silent_run),
      self.compute
    )
  }
}

/// Build asry's [`EmissionsAligner`] the way
/// [`Aligner::from_paths_with_vocabulary`] does: `vocabulary`'s tokenizer
/// document, its blank id passed EXPLICITLY, the model's fixed stride, and
/// `options` fed to the builder.
///
/// Factored out of [`Aligner::from_paths_with_vocabulary`] so the wiring —
/// above all the blank-id override and the stride — is unit-testable without
/// a CoreML model.
fn build_seam(
  language: Lang,
  vocabulary: &Vocabulary,
  normalizer: DynTextNormalizer,
  options: &AlignerOptions,
) -> Result<EmissionsAligner, EmissionsError> {
  EmissionsAligner::builder(language, vocabulary.tokenizer_json())
    .normalizer(normalizer)
    // NOT an option (see `SEAM_HOP_SAMPLES`): the stride handed to the seam
    // here must equal the stride that truncates the emissions in
    // `Encoder::emissions` — the one that, via `T`, times the words on asry's
    // effective `n_samples / (T - 1)` grid — and asry's `chunk_extent ± 2·hop`
    // validator is too loose to catch them disagreeing.
    .hop_samples(SEAM_HOP_SAMPLES)
    .min_speech_coverage(SpeechCoverage::clamped(options.min_speech_coverage()))
    .max_intra_silent_run(options.max_intra_silent_run())
    // MANDATORY (DECISION 5): the bundled chordai table's blank is `"-"`@0 and
    // it has no `<pad>` / `[PAD]` / `<blank>` entry, so the builder's default
    // auto-detect would FAIL construction. A vocabulary resolves its blank when
    // it is read (`Vocabulary::from_json`), so the id is always passed.
    .blank_token_id(vocabulary.blank_id())
    .build()
}

/// The load-time handshake: the seam's `vocabulary` must have exactly one
/// entry per class of the `model`'s CTC head.
///
/// asry re-checks the width on every chunk (`EmissionsError::VocabMismatch` in
/// `finish`), but a pair that disagrees is wrong for EVERY chunk, so it is
/// refused at load, by name, before the first one.
fn check_vocabulary_width(
  vocabulary: NonZeroUsize,
  model: NonZeroUsize,
) -> Result<(), AlignerError> {
  if vocabulary == model {
    Ok(())
  } else {
    Err(AlignerError::VocabularyMismatch(VocabularyMismatch::new(
      vocabulary.get(),
      model.get(),
    )))
  }
}

/// The `requested` [`AlignerOptions`] as the seam actually APPLIES them — the
/// effective state [`Aligner::options`] must report.
///
/// The seam coerces `min_speech_coverage` through
/// [`SpeechCoverage::clamped`](asry::emissions::SpeechCoverage::clamped) at
/// construction (`NaN` → default, out-of-range → `[0, 1]`), so the requested
/// value and the applied one can differ. This reads the applied coverage back
/// OUT of the built `seam` rather than re-clamping here, so `options()` can never
/// drift from what the seam does: the seam's own clamp is the single source of
/// truth. `max_intra_silent_run` and `compute` are not coerced, so they pass
/// through from `requested` unchanged.
fn effective_options(seam: &EmissionsAligner, requested: &AlignerOptions) -> AlignerOptions {
  requested.with_min_speech_coverage(seam.min_speech_coverage().get())
}

/// Per-language forced aligner over the CoreML wav2vec2 encoder.
///
/// Wraps alignkit's CoreML [`Encoder`], asry's [`EmissionsAligner`] seam built
/// from a [`Vocabulary`] — the encoder's CTC head width checked equal to the
/// vocabulary's size at load — and the [`AlignerOptions`] baked into that seam.
/// Build one per language with [`from_paths`](Self::from_paths) (the bundled
/// English table) or
/// [`from_paths_with_vocabulary`](Self::from_paths_with_vocabulary) (the table
/// the model ships beside it), then drive it per chunk with
/// [`align_chunk`](Self::align_chunk).
///
/// [`align_chunk`](Self::align_chunk) takes `&self`: the CoreML `Model`
/// predicts without `&mut`, and asry's `prepare`/`finish` are `&self`, so —
/// unlike asry's own ORT `Aligner`, which its registry wraps in a `Mutex` —
/// this one needs no interior mutability.
pub struct Aligner {
  encoder: crate::audio::align::encode::Encoder,
  inner: EmissionsAligner,
  options: AlignerOptions,
}

impl Aligner {
  /// Load an aligner for `language` from the compiled CoreML model at
  /// `model_path`, using the crate's **bundled** 29-class English table
  /// ([`Vocabulary::bundled`]) and the default [`AlignerOptions`].
  ///
  /// The bundled table is the vocabulary of the staged
  /// `base960h_aligner.mlmodelc`. A model that ships its own table — any model
  /// whose CTC head spells another alphabet — is loaded with
  /// [`Self::from_paths_with_vocabulary`].
  ///
  /// # Errors
  /// As [`Self::from_paths_with_vocabulary`].
  pub fn from_paths(
    language: Lang,
    model_path: &Path,
    normalizer: DynTextNormalizer,
  ) -> Result<Self, AlignerError> {
    Self::from_paths_with(language, model_path, normalizer, AlignerOptions::new())
  }

  /// [`Self::from_paths`] with explicit [`AlignerOptions`].
  ///
  /// # Errors
  /// As [`Self::from_paths_with_vocabulary`].
  pub fn from_paths_with(
    language: Lang,
    model_path: &Path,
    normalizer: DynTextNormalizer,
    options: AlignerOptions,
  ) -> Result<Self, AlignerError> {
    Self::from_paths_with_vocabulary(
      language,
      model_path,
      &Vocabulary::bundled(),
      normalizer,
      options,
    )
  }

  /// Load an aligner for `language` from the compiled CoreML model at
  /// `model_path` and the `vocabulary` it was trained with — the table that
  /// ships beside it, read with [`Vocabulary::from_file`].
  ///
  /// The model's CTC head width is read at load
  /// ([`Encoder::vocab_size`](crate::audio::align::encode::Encoder::vocab_size))
  /// and must equal `vocabulary`'s size: each id of the table is the column
  /// its token is scored in, so a table of another size is refused here, by
  /// name, rather than aligned against the wrong columns. This is the door a
  /// per-language aligner is built through: the model supplies the alphabet.
  ///
  /// With the `tracing` feature: an `alignkit.aligner.load` span at `INFO`,
  /// with the CoreML load (`alignkit.encoder.load`) nested inside it. The
  /// [`Self::from_paths`] and [`Self::from_paths_with`] loads open the same
  /// span, through this constructor.
  ///
  /// # Errors
  /// [`AlignerError::Load`] / [`AlignerError::ContractMismatch`] /
  /// [`AlignerError::UnsatisfiableInput`] / [`AlignerError::UnsatisfiableState`]
  /// if CoreML rejects the model or its I/O contract disagrees with this door's
  /// ([`Encoder::from_file_with`](crate::audio::align::encode::Encoder::from_file_with));
  /// [`AlignerError::Seam`] if asry's builder rejects the vocabulary or the
  /// normalizer (e.g. a normalizer that needs a `|` delimiter the table lacks);
  /// [`AlignerError::VocabularyMismatch`] if the table's size is not the model's
  /// CTC head width.
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
        vocabulary = vocabulary.size().get(),
      ),
    )
  )]
  pub fn from_paths_with_vocabulary(
    language: Lang,
    model_path: &Path,
    vocabulary: &Vocabulary,
    normalizer: DynTextNormalizer,
    options: AlignerOptions,
  ) -> Result<Self, AlignerError> {
    let encoder = Encoder::from_file_with(
      model_path,
      EncoderOptions::new().with_compute(options.compute()),
    )?;
    let inner = build_seam(language, vocabulary, normalizer, &options)?;
    check_vocabulary_width(inner.vocab_size(), encoder.vocab_size())?;
    // `options()` must report EFFECTIVE state (F3): the seam coerced
    // `min_speech_coverage` through `SpeechCoverage::clamped`, so store what the
    // seam actually applies — read back out of it — not the requested value that
    // may have been out of range or NaN.
    let options = effective_options(&inner, &options);
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

  /// The **effective** [`AlignerOptions`] baked into this aligner's seam — the
  /// values actually in force, which are not always the ones requested at
  /// construction. In particular `min_speech_coverage` is the seam's *clamped*
  /// value ([`SpeechCoverage::clamped`](asry::emissions::SpeechCoverage::clamped):
  /// `NaN` → default, out-of-range → `[0, 1]`), so a caller that constructed the
  /// aligner with `2.0` reads back `1.0` here — the coverage the aligner will
  /// actually apply — never the un-applied request.
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
  /// A character the vocabulary cannot spell is an event
  /// ([`OovKind::Symbol`](asry::emissions::OovKind::Symbol)), never an error:
  /// asry looks each character up in the vocabulary and never runs the
  /// tokenizer's `encode`, whose `MissingUnkToken` on a table with no unknown
  /// token used to fail the whole chunk.
  ///
  /// # Errors
  /// [`AlignError::Alignment`] if the text normalizer rejects the text, or its
  /// output disagrees with itself (its word count against its boundary map).
  /// Punctuation-only input yields an empty vec, not an error.
  pub fn detect_oov(&self, text: &str) -> Result<Vec<OovEvent>, AlignError> {
    Ok(self.inner.detect_oov(text)?)
  }

  /// Align one chunk end-to-end into per-word [`TimeRange`]s in `clock`'s
  /// output timebase.
  ///
  /// - `samples`: the chunk's 16 kHz f32 mono audio, at most
  ///   [`ENCODER_WINDOW_SAMPLES`](crate::audio::align::encode::ENCODER_WINDOW_SAMPLES).
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
  /// returns an **empty** [`AlignmentResult`]: there was nothing to align. The
  /// two per-chunk outcomes that are not faults of the setup are NAMED instead,
  /// never returned empty: [`AlignError::Refused`] when the caller's decisions
  /// resolved a position `FailClosed` (it carries every refused position), and
  /// [`AlignError::NoAlignmentPath`] when the CTC lattice admits no path for
  /// this chunk's audio and tokens. Either way the ASR text is the caller's to
  /// keep; only per-word timings are missing. See the
  /// [`crate::audio::align::error`] module doc.
  ///
  /// With the `tracing` feature: one `alignkit.align_chunk` span at `DEBUG` per
  /// call, wrapping the whole VAD → prepare → encode → finish pass, with
  /// `alignkit.encoder.emissions` nested inside it. The empty-result path above
  /// is a *success* that produces no words, which is exactly the state a caller
  /// ends up staring at a debugger over — the span's `sub_segments` /
  /// `text_bytes` / `samples` fields are there to tell it apart from a chunk
  /// whose words simply fell outside its speech.
  ///
  /// # Errors
  /// [`AlignError::InputTooLong`] if `samples` exceeds the encoder window;
  /// [`AlignError::Span`] if `sub_segments` are not in the 1/16000 timebase;
  /// [`AlignError::Refused`] if a decision in `oov_decisions` is `FailClosed`;
  /// [`AlignError::Prediction`] / [`AlignError::Tensor`] from the CoreML
  /// encode; [`AlignError::CorruptEmissions`] if the encoder's emission matrix
  /// left the log-probability domain from below (an ANE placement set through
  /// [`AlignerOptions::with_compute`] — see [`crate::audio::align::encode::LOG_PROB_FLOOR`]);
  /// [`AlignError::UnnormalizedEmissions`] if the encoder's emission matrix is not
  /// normalized log-probabilities (a raw-logit model swap — see
  /// [`crate::audio::align::encode::LOG_PROB_SUM_TOLERANCE`]);
  /// [`AlignError::NoAlignmentPath`] if the lattice admits no path;
  /// [`AlignError::Alignment`] for any other seam failure (stride / vocab /
  /// blank-id validation, a non-finite or positive log-probability,
  /// tokenization, abort).
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
    if samples.len() > crate::audio::align::encode::ENCODER_WINDOW_SAMPLES {
      return Err(AlignError::InputTooLong(InputTooLong::new(
        samples.len(),
        crate::audio::align::encode::ENCODER_WINDOW_SAMPLES,
      )));
    }

    let speech = if sub_segments.is_empty() {
      SpeechSpans::all_speech()
    } else {
      SpeechSpans::from_time_ranges(sub_segments)?
    };

    let prepared = self
      .inner
      .prepare(samples, &speech, text, oov_decisions, abort_flag)
      .map_err(|err| seam_error(err, oov_decisions))?;
    if prepared.is_trivial() {
      return Ok(AlignmentResult::new(Vec::new()));
    }

    // asry has already silence-masked + receptive-field-padded the buffer; the
    // encoder consumes exactly THAT, and the truncation formula needs the real
    // (pre-pad) sample count. Both come off the one `PreparedChunk` via
    // `EncoderInput::from_prepared`: the padded buffer from `encoder_input()`, the
    // real length from asry's own `real_samples()` (the same `samples.len()` we
    // handed `prepare`). Reading both from one authoritative object is what makes
    // a mismatched real length unrepresentable (F1) — there is no second length
    // for this call site to get out of step, and an external prepare → Encoder →
    // finish composer reaches the identical public door.
    let input = EncoderInput::from_prepared(&prepared)?;
    let emissions = self.encoder.emissions(input)?;

    self
      .inner
      .finish(prepared, &emissions, clock, abort_flag)
      .map_err(|err| seam_error(err, oov_decisions))
  }
}

/// The seam's error, NAMED: a refusal by the caller's OOV decisions is
/// [`AlignError::Refused`] carrying every refused position, a chunk the
/// lattice cannot align is [`AlignError::NoAlignmentPath`], and every other
/// failure is [`AlignError::Alignment`].
///
/// The one classifier both seam calls in [`Aligner::align_chunk`] go through,
/// so each case is named wherever it arises — in practice a refusal arises in
/// `prepare`, where asry tokenizes, and a no-path chunk in `finish`, where the
/// trellis runs. Neither may become an empty result, which is a SUCCESS's
/// answer — a chunk with nothing to align, or one whose words all fell outside
/// its speech — and which a caller could then not tell from either.
///
/// asry's `SemanticOutOfVocab` carries only a message, so the refused positions
/// are read off the decisions the caller passed. That is exact, not a guess:
/// asry validates EVERY decision against the text's freshly detected events
/// before applying any, and refuses only at a `FailClosed` one, so the
/// `FailClosed` decisions are precisely the positions the caller's policy
/// refused. With none of them — which asry's contract rules out — the error
/// stays asry's own rather than become a refusal that names nothing.
fn seam_error(err: EmissionsError, oov_decisions: &[ResolvedOov]) -> AlignError {
  match err {
    EmissionsError::SemanticOutOfVocab(failure) => {
      let refused: Vec<OovEvent> = oov_decisions
        .iter()
        .filter(|resolved| resolved.decision() == OovDecision::FailClosed)
        .map(|resolved| resolved.event().clone())
        .collect();
      if refused.is_empty() {
        AlignError::Alignment(EmissionsError::SemanticOutOfVocab(failure))
      } else {
        AlignError::Refused(Refusal::new(refused))
      }
    }
    EmissionsError::NoAlignmentPath(failure) => AlignError::NoAlignmentPath(failure),
    other => AlignError::Alignment(other),
  }
}

#[cfg(test)]
mod tests;
