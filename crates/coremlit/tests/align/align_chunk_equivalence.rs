//! **Codex round-3 regression** — [`Aligner::align_chunk`]'s encoder-input door
//! (`src/aligner/mod.rs`, `EncoderInput::from_prepared`) is the SAME door an
//! external `prepare → Encoder → finish` composer reaches, and the two MUST
//! produce the identical alignment. Nothing gated that: `align_chunk` builds
//! its input via `EncoderInput::from_prepared`, but no test drove THAT call
//! site — `tests/prepared_composition.rs` calls `from_prepared` directly (never
//! through `align_chunk`), and every other model-gated test uses whole-audio
//! spans, where the prepared buffer EQUALS the raw samples. So mutating that
//! line to `EncoderInput::from_samples(samples)` compiled and left every gate
//! green while sending UNMASKED audio into CoreML.
//!
//! This test closes the gap end-to-end with a **deliberately partial VAD mask**
//! that makes the mask load-bearing — the prepared (silence-masked) buffer no
//! longer equals the raw samples. It drives the SAME real audio (`jfk.wav`)
//! through both routes:
//!
//! - the canonical [`Aligner::align_chunk`] (which reaches the door under test);
//! - the explicit public composition — [`EmissionsAligner::prepare`] →
//!   [`EncoderInput::from_prepared`] → [`Encoder::emissions`] →
//!   [`EmissionsAligner::finish`] — on a standalone seam wired EXACTLY as
//!   `Aligner::from_paths` wires its own (see [`matching_seam`]).
//!
//! then asserts the reference composition produced words and that the two agree
//! on every word's text, both PTS integers, and the score's raw BITS.
//!
//! # Why this is the mutation proof
//!
//! Substitute `EncoderInput::from_samples(samples)` for the `from_prepared` call
//! in [`Aligner::align_chunk`]: the canonical route then feeds CoreML the raw,
//! UNMASKED buffer while `finish` still applies the prepared speech-span policy.
//! wav2vec2-base group-norms over the whole sequence and attends globally with
//! no padding mask, so the excluded audio perturbs the retained frames — the
//! surviving words shift in timing and/or score and this comparison FAILS.
//! Revert, and the two routes are bit-identical again. That gap is the standing
//! proof the call site must read the prepared buffer, not the raw samples.
//!
//! Model-gated like the sibling end-to-end tests: `#[ignore]` is the only gate,
//! a missing model is a hard failure (see `tests/align_chunk.rs`'s module doc).

mod common;

use core::{num::NonZeroU32, sync::atomic::AtomicBool};

use alignkit::{
  ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock, SpeechCoverage, SpeechSpans,
  TimeRange,
  aligner::{DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE},
  default_oov_decisions,
  encode::{Encoder, EncoderInput, HOP_SAMPLES},
  vocab,
};
use asry::emissions::EmissionsAligner;

/// Builds asry's [`EmissionsAligner`] with the SAME wiring
/// `Aligner::from_paths_with` bakes in through its private `build_seam`:
/// bundled 29-class chordai tokenizer, [`EnglishNormalizer`], the model's fixed
/// 320-sample stride ([`HOP_SAMPLES`]), the default coverage / silent-run, and
/// the MANDATORY explicit blank id ([`vocab::BLANK_ID`] — that vocab has no
/// `<pad>`/`[PAD]`/`<blank>` for asry's auto-detect).
///
/// Every knob is set EXPLICITLY rather than left to asry's builder defaults.
/// asry's defaults happen to match today, but pinning them to alignkit's own
/// constants is what makes the equivalence below isolate exactly the
/// `from_prepared` door in `align_chunk`: if this reference seam and the
/// `Aligner`'s inner seam ever configured differently, the two routes would
/// diverge for a reason other than the mutation under test, and the baseline
/// (unmutated) run would fail loudly rather than silently pass.
fn matching_seam() -> EmissionsAligner {
  let hop = NonZeroU32::new(HOP_SAMPLES as u32).expect("HOP_SAMPLES (320) is nonzero");
  EmissionsAligner::builder(Lang::En, vocab::tokenizer_json_bytes())
    .normalizer(Box::new(EnglishNormalizer::new()))
    .hop_samples(hop)
    .min_speech_coverage(SpeechCoverage::clamped(DEFAULT_MIN_SPEECH_COVERAGE))
    .max_intra_silent_run(DEFAULT_MAX_INTRA_SILENT_RUN)
    .blank_token_id(vocab::BLANK_ID)
    .build()
    .expect("build the reference EmissionsAligner matching Aligner::from_paths")
}

/// A stream-sample-0 clock in [`ANALYSIS_TIMEBASE`], so a word's PTS **is** its
/// 16 kHz sample index — directly comparable across the two routes. Built fresh
/// per call because both [`Aligner::align_chunk`] and
/// [`EmissionsAligner::finish`] consume the clock by value.
fn analysis_clock() -> OutputClock {
  OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction")
}

/// [`Aligner::align_chunk`] and the explicit public `prepare → from_prepared →
/// emissions → finish` composition agree bit-for-bit on real audio under a
/// partial VAD mask — the regime in which the prepared buffer diverges from the
/// raw samples and the `from_prepared` door at `align_chunk`'s call site is
/// load-bearing.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_equals_public_prepared_composition_under_partial_vad() {
  // LEFT — the shipping aligner, driven through its canonical `align_chunk`
  // (the route that reaches the `from_prepared` call site under test).
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect(
    "build the En aligner from the CoreML model + bundled tokenizer (set ALIGNKIT_TEST_MODELS \
     to the model directory)",
  );

  // RIGHT — the reference public composition: a standalone seam wired exactly
  // as `Aligner` wires its own, plus its own load of the same model. The
  // determinism gate (`tests/align_chunk.rs`) proves two loads emit
  // bit-identical tensors, so this second encoder is a faithful stand-in for
  // the `Aligner`'s private one.
  let seam = matching_seam();
  let encoder = Encoder::from_file(common::model_path())
    .expect("load base960h_aligner.mlmodelc for the reference composition");

  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  assert_eq!(
    samples.len(),
    176_000,
    "jfk fixture is 176,000 samples (11.000 s @ 16 kHz); the mask boundaries below assume it"
  );

  // The DELIBERATELY PARTIAL VAD mask: two speech spans that EXCLUDE the loud
  // 5.25–7.50 s middle (samples 84,000..120,000 — ~0.15 RMS of continuous real
  // speech, the "...country can do for you, ask..." stretch). Because the gap
  // is speech and not silence, `prepare`'s silence mask genuinely zeroes real
  // audio, so the prepared buffer no longer equals the raw samples — the exact
  // regime the `from_prepared` door exists for. Words inside the gap fall below
  // coverage and drop on BOTH routes identically; the retained words on either
  // side are what the equivalence compares.
  let sub_segments = [
    TimeRange::new(0, 84_000, ANALYSIS_TIMEBASE),
    TimeRange::new(120_000, 176_000, ANALYSIS_TIMEBASE),
  ];

  let text = common::JFK_TRANSCRIPT;
  let decisions = default_oov_decisions(&aligner.detect_oov(text).expect("detect_oov"));
  let abort = AtomicBool::new(false);

  // ——— LEFT: canonical align_chunk ———
  let left = aligner
    .align_chunk(
      &samples,
      &sub_segments,
      text,
      analysis_clock(),
      &abort,
      &decisions,
    )
    .expect("align_chunk succeeds end-to-end under the partial VAD mask")
    .words()
    .to_vec();

  // ——— RIGHT: prepare → from_prepared → emissions → finish, by hand ———
  //
  // `align_chunk` derives its speech spans the same way for a non-empty
  // `sub_segments`: `SpeechSpans::from_time_ranges(sub_segments)`. Same spans,
  // same text, same OOV decisions — so the only thing that can differ between
  // the two routes is how the encoder input is built.
  let speech =
    SpeechSpans::from_time_ranges(&sub_segments).expect("valid analysis-timebase speech spans");
  let prepared = seam
    .prepare(&samples, &speech, text, &decisions, &abort)
    .expect("prepare the reference composition");
  assert!(
    !prepared.is_trivial(),
    "the jfk transcript must tokenize to alignable tokens, or there is no encoder buffer to test"
  );
  let input = EncoderInput::from_prepared(&prepared).expect("from_prepared geometry is valid");
  let emissions = encoder
    .emissions(input)
    .expect("reference emissions on the prepared buffer");
  let right = seam
    .finish(prepared, &emissions, analysis_clock(), &abort)
    .expect("finish the reference composition")
    .words()
    .to_vec();

  // The reference route MUST have produced words: two empty vecs compare equal
  // and would silently vacate the mutation proof below.
  assert!(
    !right.is_empty(),
    "the reference composition produced no words — the partial mask dropped everything, so the \
     comparison cannot witness the from_prepared door; widen the retained spans"
  );
  // Assert length before zipping: `zip` truncates to the shorter, so a word the
  // mutation adds or drops must be caught here, not swallowed.
  assert_eq!(
    left.len(),
    right.len(),
    "align_chunk and the public composition disagree on word COUNT ({} vs {})",
    left.len(),
    right.len()
  );

  // Exact equivalence: text, both PTS integers, and the score's raw BITS (so a
  // NaN or a signed-zero flip cannot slip past the `f32` `==`). Substituting
  // `EncoderInput::from_samples(samples)` for `from_prepared` at
  // `align_chunk`'s call site diverges the LEFT route here and fails this loop.
  for (l, r) in left.iter().zip(&right) {
    assert_eq!(
      l.text(),
      r.text(),
      "word text differs between align_chunk and the public composition: `{}` vs `{}`",
      l.text(),
      r.text()
    );
    assert_eq!(
      (l.range().start_pts(), l.range().end_pts()),
      (r.range().start_pts(), r.range().end_pts()),
      "word `{}`: range differs between align_chunk and the public composition",
      l.text()
    );
    assert_eq!(
      l.score().to_bits(),
      r.score().to_bits(),
      "word `{}`: score differs between align_chunk and the public composition ({} vs {})",
      l.text(),
      l.score(),
      r.score()
    );
  }
}
