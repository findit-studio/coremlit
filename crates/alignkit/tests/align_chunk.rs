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
  ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock, default_oov_decisions,
};

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_produces_monotonic_word_timings() {
  let model = common::model_path();
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  assert!(!samples.is_empty(), "fixture decoded to no samples");

  // `Aligner::from_paths` → `AlignerOptions::new()` → `DEFAULT_ENCODER_COMPUTE`.
  // Deliberately NOT a hardcoded compute placement: this is the crate's only
  // end-to-end proof, so it must run the configuration that actually ships. A
  // gate pinned to a compute unit proves only that compute unit.
  let aligner = Aligner::from_paths(Lang::En, &model, Box::new(EnglishNormalizer::new())).expect(
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

  let result = aligner
    .align_chunk(&samples, &[], text, clock, &abort, &decisions)
    .expect("align_chunk succeeds end-to-end");

  let words = result.words();
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
