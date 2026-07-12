//! Long-form VAD chunking end-to-end on ted_60 (60 s) / tiny — the first
//! real exercise of spec §5.5's long-form flow: EnergyVad chunking, the
//! scoped-thread worker pool, seek-offset re-application (incl. word
//! timings), and merge_transcription_results.

mod common;

use whisperkit::{
  options::{ChunkingStrategy, DecodingOptions, Options},
  transcribe::WhisperKit,
};

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn ted_60_vad_chunked_transcription_with_word_timestamps() {
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/ted_60.wav"));
  assert_eq!(audio.len(), 960_000, "60 s at 16 kHz");
  // `Options::new` takes both folders directly (two-arg constructor, not a
  // zero-arg `new()` plus `with_model_folder`/`with_tokenizer_folder`
  // builders) — same brief-vs-shipped-API fix as tests/pipeline.rs's
  // `tiny_options`/tests/parity_jfk.rs.
  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir())).unwrap();
  let options = DecodingOptions::new()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .with_word_timestamps();
  let result = kit.transcribe(&audio, &options).unwrap();

  assert_eq!(result.language(), "en");
  assert!(
    result.text().len() > 200,
    "60 s of speech transcribes substantially"
  );
  let segments = result.segments_slice();
  assert!(segments.len() >= 4, "got {} segments", segments.len());
  // Chunk re-anchoring: at least one segment lives past the 30 s window
  // boundary — impossible without VAD chunking + seek offsets.
  assert!(
    segments.iter().any(|s| s.start() > 30.0),
    "no segment beyond 30 s: offsets not re-applied"
  );
  // Segment times are globally plausible and non-inverted.
  for segment in segments {
    assert!(segment.end() >= segment.start());
    assert!(segment.end() <= 61.0);
    // Word path flowed through the chunked workers AND got shifted:
    // words must sit inside their (already offset) segment, ± merge slack.
    for word in segment.words_slice() {
      assert!(
        word.start() >= segment.start() - 1.0,
        "word before its segment"
      );
      assert!(word.end() <= segment.end() + 1.0, "word after its segment");
    }
  }
  let worded = segments
    .iter()
    .filter(|s| !s.words_slice().is_empty())
    .count();
  assert!(
    worded * 2 >= segments.len(),
    "most segments carry word timings"
  );
}
