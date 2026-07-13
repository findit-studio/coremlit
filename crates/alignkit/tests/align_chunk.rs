//! End-to-end model-gated forced alignment: alignkit's [`Aligner`] drives
//! one real chunk (`jfk.wav` + its known transcript) through
//! prepare → CoreML encode → finish, proving the whole pipeline against the
//! merged asry emissions seam produces monotonic per-word timings inside the
//! audio.
//!
//! This is Gate-1 (word-timing) PLUMBING, not the parity gate: it asserts
//! alignkit's OWN output is well-formed, NOT that it agrees with asry-ort
//! (that oracle comparison is Task B5). Self-skips when
//! `ALIGNKIT_TEST_MODELS` / the fixture are absent, matching the workspace
//! `-- --ignored` convention.

mod common;

use core::sync::atomic::AtomicBool;

use alignkit::{
  ANALYSIS_TIMEBASE, Aligner, AlignerOptions, ComputeUnits, EnglishNormalizer, Lang, OutputClock,
  default_oov_decisions,
};

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_produces_monotonic_word_timings() {
  let model = common::model_path();
  if !model.exists() {
    eprintln!("SKIP: model not found at {model:?} (set ALIGNKIT_TEST_MODELS)");
    return;
  }
  let wav = common::jfk_wav_path();
  if !wav.exists() {
    eprintln!("SKIP: jfk.wav fixture not found at {wav:?}");
    return;
  }

  let samples = common::load_wav_mono_f32(&wav);
  assert!(!samples.is_empty(), "fixture decoded to no samples");

  // `ComputeUnits::CpuOnly`, matching this crate's model-gated convention
  // (`src/encode/tests.rs`, `tests/model_io.rs`): deterministic, and it skips
  // the one-time multi-minute CoreML ANE compilation an `All` load of this
  // model's fixed 960,000-sample input pays. `ComputeUnits::All` remains the
  // default placement — and is the one B5's word-timing parity gate must
  // measure, per the plan's compute-unit rule.
  let aligner = Aligner::from_paths_with(
    Lang::En,
    &model,
    Box::new(EnglishNormalizer::new()),
    AlignerOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("build the En aligner from the CoreML model + bundled tokenizer");

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
