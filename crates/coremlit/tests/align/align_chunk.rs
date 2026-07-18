//! End-to-end model-gated forced alignment: alignkit's [`Aligner`] drives
//! one real chunk (`jfk.wav` + its known transcript) through
//! prepare → CoreML encode → finish, proving the whole pipeline against the
//! merged asry emissions seam produces monotonic per-word timings inside the
//! audio.
//!
//! This is Gate-1 (word-timing) PLUMBING, not the parity gate: it asserts
//! alignkit's OWN output is well-formed, NOT that it agrees with asry-ort
//! (that oracle comparison is Task B5).
//!
//! # This test does not skip
//!
//! `#[ignore]` is the opt-in gate, and it is the ONLY gate. A missing model or
//! a missing fixture is a hard FAILURE, never an early `return`. It used to
//! self-skip on both, which made it a fake gate: pointing
//! `ALIGNKIT_TEST_MODELS` at an empty directory reported `test result: ok. 1
//! passed` — green, having aligned nothing. The live exposure was the fixture,
//! reached by a cross-crate relative path into whisperkit
//! (`tests/common/mod.rs`): had that file ever moved, B4's only end-to-end
//! proof would have evaporated silently while the gate stayed green. A skip
//! that looks like a pass in the test summary is worse than no test.

mod common;

use core::sync::atomic::AtomicBool;

use alignkit::{
  ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock, Word, default_oov_decisions,
};

/// Builds the aligner and drives one real chunk (`jfk.wav` + its known
/// transcript) end-to-end, on the crate's shipping configuration.
///
/// `Aligner::from_paths` → `AlignerOptions::new()` → `DEFAULT_ENCODER_COMPUTE`.
/// Deliberately NOT a hardcoded compute placement: these are the crate's only
/// end-to-end proofs, so they must run the configuration that actually ships.
/// A gate pinned to a compute unit proves only that compute unit — the previous
/// default (`ComputeUnits::All`) corrupted every emission tensor while every
/// model-gated test, each pinned to `CpuOnly`, stayed green.
fn align_jfk(samples: &[f32]) -> Vec<Word> {
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect(
    "build the En aligner from the CoreML model + bundled tokenizer (set ALIGNKIT_TEST_MODELS \
     to the model directory)",
  );

  let text = common::JFK_TRANSCRIPT;
  let events = aligner.detect_oov(text).expect("detect_oov");
  let decisions = default_oov_decisions(&events);

  // No VAD (whole chunk is speech). OutputClock anchored at stream sample 0
  // in the 1/16000 analysis timebase, so per-word PTS ARE 16 kHz sample
  // indices — directly comparable to the audio length.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);

  aligner
    .align_chunk(samples, &[], text, clock, &abort, &decisions)
    .expect("align_chunk succeeds end-to-end")
    .words()
    .to_vec()
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_produces_monotonic_word_timings() {
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  assert!(!samples.is_empty(), "fixture decoded to no samples");

  let words = &align_jfk(&samples);
  assert!(
    !words.is_empty(),
    "a real transcript over matching audio must produce words"
  );

  let bound = samples.len() as i64;
  let mut prev_start = 0_i64;
  for word in words {
    let range = word.range();
    let (start, end) = (range.start_pts(), range.end_pts());
    assert!(
      start <= end,
      "word `{}`: start {start} exceeds end {end}",
      word.text()
    );
    assert!(
      start >= 0 && end <= bound,
      "word `{}`: range [{start}, {end}] escapes the audio [0, {bound}]",
      word.text()
    );
    assert!(
      start >= prev_start,
      "word `{}`: start {start} precedes the previous word's start {prev_start} (not monotonic)",
      word.text()
    );
    let score = word.score();
    assert!(
      (0.0..=1.0).contains(&score),
      "word `{}`: score {score} outside [0, 1]",
      word.text()
    );
    prev_start = start;
  }
}

/// **Gate 3 — determinism** (design spec §7): two runs, bit-identical.
///
/// Not a tolerance and not a statistic: `assert_eq!` on the PTS integers and on
/// the score's raw bits (`f32::to_bits`, so a `NaN` or a `-0.0`/`+0.0` flip
/// cannot slip through the `==` that `f32: PartialEq` would give). Every stage
/// downstream of the encoder is deterministic dynamic programming, so this is
/// really a statement about CoreML: the same weights on the same input on the
/// same placement must return the same tensor, twice.
///
/// It matters because the alternative was live. When this model was scheduled
/// on the ANE, 16.7% of its emission cells saturated to a `-45440` sentinel —
/// and that corruption was **bit-identical run to run**. Had it instead been
/// *non-deterministic*, every other gate in this crate would have flickered
/// rather than failed, which is far harder to diagnose. This test is what says
/// which of the two we are in.
///
/// Deliberately reloads the model on each run (via [`align_jfk`]) rather than
/// reusing one `Aligner`: a load-time nondeterminism — a compute-placement
/// decision CoreML makes differently on a second load, say — is exactly the
/// class of defect a same-session double-`predict` would hide.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_is_bit_identical_across_runs() {
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());

  let first = align_jfk(&samples);
  let second = align_jfk(&samples);

  assert!(
    !first.is_empty(),
    "a real transcript over matching audio must produce words"
  );
  assert_eq!(
    first.len(),
    second.len(),
    "two runs over identical input produced different word counts"
  );

  for (a, b) in first.iter().zip(&second) {
    assert_eq!(a.text(), b.text(), "word text differs between runs");
    assert_eq!(
      (a.range().start_pts(), a.range().end_pts()),
      (b.range().start_pts(), b.range().end_pts()),
      "word `{}`: timing differs between two runs over identical input",
      a.text()
    );
    assert_eq!(
      a.score().to_bits(),
      b.score().to_bits(),
      "word `{}`: score differs between two runs over identical input ({} vs {})",
      a.text(),
      a.score(),
      b.score()
    );
  }
}

/// **The codex-fence regression, on the canonical `Aligner` path.** 641 real
/// samples carrying three distinct tokens (`ABC`) truncate to one emission frame,
/// and one frame cannot carry three tokens — so the seam returns
/// `NoAlignmentPath`, which [`Aligner::align_chunk`] RECOVERS into an EMPTY result:
/// the ASR text survives, only the per-word timings are dropped (see
/// `align_chunk`'s doc and `recover_or_error`).
///
/// This is the canonical-path half of `tests/prepared_composition.rs`'s
/// `public_prepared_composition_641_abc_has_no_alignment_path` (which pins the raw
/// `NoAlignmentPath` error at the seam). Both are mutation proofs: revert
/// `truncated_frame_count` to `ceil(641/320) = 3` and the trellis threads `ABC`
/// across three phantom, padding-derived frames, so `align_chunk` returns a
/// non-empty word list — failing this assertion. asry's `chunk_extent ± 2·hop`
/// stride check (`3×320 = 960` inside `641 ± 640`) is too loose to catch the
/// phantom frames; this end-to-end test is the guard.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_641_abc_recovers_to_no_words() {
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("build the En aligner (set ALIGNKIT_TEST_MODELS to the model directory)");

  let jfk = common::load_wav_mono_f32(&common::jfk_wav_path());
  let samples = &jfk[80_000..80_641];
  assert_eq!(
    samples.len(),
    641,
    "the fence case is exactly 641 real samples"
  );

  let text = "ABC";
  let events = aligner.detect_oov(text).expect("detect_oov");
  assert!(
    events.is_empty(),
    "A, B, C must be in-vocab, or the OOV path — not the frame count — would drive the result"
  );
  let decisions = default_oov_decisions(&events);
  // No VAD; clock anchored at stream sample 0 in the analysis timebase.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);

  let result = aligner
    .align_chunk(samples, &[], text, clock, &abort, &decisions)
    .expect("align_chunk recovers NoAlignmentPath into an empty Ok, never an Err");
  assert!(
    result.words().is_empty(),
    "one frame cannot carry three distinct tokens: the recovered result must have zero words, got \
     {:?}",
    result.words().iter().map(Word::text).collect::<Vec<_>>()
  );
}
