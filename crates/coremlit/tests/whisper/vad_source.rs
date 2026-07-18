//! The opt-in Silero VAD source (`vadkit` feature): the source-selection
//! contract (the energy VAD stays the default; `SileroVad` swaps in through the
//! existing `set_vad_detector` seam and changes the frame geometry to 256 ms)
//! and one model-gated end-to-end long-form transcription driven by Silero VAD
//! chunking, segments pinned two-sided.
//!
//! Gated on the `vadkit` feature, so the whole file compiles only when the
//! source under test exists.
#![cfg(feature = "vadkit")]

mod common;

use std::path::PathBuf;

use coremlit::ComputeUnits;
use vadkit::VadModelOptions;
use whisperkit::{
  audio::vad::DEFAULT_FRAME_LENGTH_SAMPLES,
  options::{ChunkingStrategy, DecodingOptions, Options},
  silero_vad::SileroVad,
  transcribe::WhisperKit,
};

/// Path to the compiled vadkit VAD artifact — `VADKIT_TEST_MODELS` or
/// `<workspace>/Models/vadkit` (mirrors `vadkit`'s own test-model resolution).
fn vadkit_model_path() -> PathBuf {
  std::env::var_os("VADKIT_TEST_MODELS")
    .map_or_else(
      || {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
          .join("../..")
          .join("Models")
          .join("vadkit")
      },
      PathBuf::from,
    )
    .join("silero-vad-unified-256ms-v6.2.1.mlmodelc")
}

/// Loads a `SileroVad` on `cpu_only` (deterministic; matches the vadkit trace
/// oracle's placement).
fn load_silero_vad() -> SileroVad {
  SileroVad::load_with(
    vadkit_model_path(),
    VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load SileroVad from Models/vadkit")
}

/// Hermetic: the source under test satisfies the seam's `Send + Sync` bound and
/// declares the 256 ms (4096-sample) frame geometry — provable without a model.
#[test]
fn silero_vad_is_send_sync_and_declares_256ms_frames() {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<SileroVad>();
  assert_eq!(
    SileroVad::FRAME_LENGTH_SAMPLES,
    4096,
    "256 ms at 16 kHz — 8× the energy VAD's 100 ms frame"
  );
  // The two sources declare distinct geometries, so a swap is observable.
  assert_ne!(
    SileroVad::FRAME_LENGTH_SAMPLES,
    DEFAULT_FRAME_LENGTH_SAMPLES
  );
}

/// **Source-selection contract** (model-gated): the default pipeline uses the
/// energy VAD; swapping in `SileroVad` through the existing seam takes effect
/// (the frame geometry changes to 256 ms) and the detector actually runs,
/// returning one flag per 256 ms frame with both speech and silence on a real
/// clip.
#[test]
#[ignore = "requires local tiny + vadkit models (WHISPERKIT_TEST_MODELS / VADKIT_TEST_MODELS)"]
fn vad_source_selection_swaps_energy_for_silero() {
  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir())).unwrap();

  // Default: the energy VAD (0.1 s = 1600-sample frames), byte-untouched.
  assert_eq!(
    kit.vad_detector().frame_length_samples(),
    DEFAULT_FRAME_LENGTH_SAMPLES,
    "default detector must be the energy VAD"
  );
  assert_eq!(DEFAULT_FRAME_LENGTH_SAMPLES, 1600);

  // Opt in to Silero VAD through the runtime seam: geometry becomes 256 ms.
  let kit = kit.with_vad_detector(Box::new(load_silero_vad()));
  assert_eq!(
    kit.vad_detector().frame_length_samples(),
    4096,
    "Silero VAD source must be selected (256 ms frames)"
  );

  // And it runs: one flag per 256 ms frame, with speech and silence present.
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/ted_60.wav"));
  let flags = kit.vad_detector().voice_activity(&audio);
  assert_eq!(
    flags.len(),
    audio.len().div_ceil(4096),
    "one flag per 256 ms frame (final partial frame included)"
  );
  assert!(
    flags.iter().any(|&f| f),
    "Silero must find speech in ted_60"
  );
  assert!(flags.iter().any(|&f| !f), "ted_60 has silence too");
}

// Measured e2e pins (tiny model, Silero VAD chunking, `cpu_only` VAD): 19
// segments, 904-char transcript. The bands are two-sided around the measured
// values with headroom for cross-silicon ANE fp16 drift — the whisper decode
// runs on the shipping compute units, where a borderline argmax flip can shift
// timestamp tokens and thus segment splits, so a tight count would be brittle
// on other Apple Silicon (the sibling `longform.rs` uses a one-sided `>= 4` for
// the same reason). The bands still catch gross regressions: a broken VAD
// collapsing to the un-chunked seek loop (≈ 4–8 segments), runaway repetition,
// or an empty transcript all fall outside.
const E2E_SEGMENTS_MIN: usize = 10;
const E2E_SEGMENTS_MAX: usize = 30;
const E2E_TEXT_LEN_MIN: usize = 200;
const E2E_TEXT_LEN_MAX: usize = 3_000;

/// **End-to-end Silero-VAD long-form transcription** (model-gated): the whole
/// opt-in path — `WhisperKit` + `ChunkingStrategy::Vad` driving `SileroVad`'s
/// 256 ms chunk boundaries through `VadChunker`, then per-chunk decode + merge —
/// on the 60 s ted_60 clip, with the transcript and segments pinned. The
/// companion `longform.rs` pins the same clip under the energy VAD; this proves
/// the Silero source produces a correct transcription through the same pipeline.
#[test]
#[ignore = "requires local tiny + vadkit models (WHISPERKIT_TEST_MODELS / VADKIT_TEST_MODELS)"]
fn ted_60_transcription_with_silero_vad_chunking() {
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/ted_60.wav"));
  assert_eq!(audio.len(), 960_000, "60 s at 16 kHz");

  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir()))
    .unwrap()
    .with_vad_detector(Box::new(load_silero_vad()));
  let options = DecodingOptions::new()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .with_word_timestamps();
  let result = kit.transcribe(&audio, &options).unwrap();

  println!(
    "[vad_source] ted_60 silero-vad: lang={} segments={} text_len={}",
    result.language(),
    result.segments_slice().len(),
    result.text().len(),
  );

  assert_eq!(result.language(), "en");
  // The opening clause of the TED clip must survive Silero chunking + merging
  // (same content anchor as the energy-VAD longform e2e).
  assert!(
    result.text().to_lowercase().contains("in college"),
    "expected the opening clause, got: {}",
    &result.text()[..result.text().len().min(120)]
  );
  assert!(
    (E2E_TEXT_LEN_MIN..=E2E_TEXT_LEN_MAX).contains(&result.text().len()),
    "transcript length {} outside [{E2E_TEXT_LEN_MIN}, {E2E_TEXT_LEN_MAX}] — \
     60 s of speech transcribes substantially but does not run away",
    result.text().len()
  );

  let segments = result.segments_slice();
  assert!(
    (E2E_SEGMENTS_MIN..=E2E_SEGMENTS_MAX).contains(&segments.len()),
    "segment count {} outside [{E2E_SEGMENTS_MIN}, {E2E_SEGMENTS_MAX}]",
    segments.len()
  );
  // Chunk re-anchoring: segments live past the 30 s window boundary.
  assert!(
    segments.iter().any(|s| s.start() > 30.0),
    "no segment beyond 30 s: chunk offsets not re-applied"
  );
  // Segment times globally plausible, non-inverted, words inside their segment.
  for segment in segments {
    assert!(segment.end() >= segment.start());
    assert!(segment.end() <= 61.0);
    for word in segment.words_slice() {
      assert!(
        word.start() >= segment.start() - 1.0,
        "word before its segment"
      );
      assert!(word.end() <= segment.end() + 1.0, "word after its segment");
    }
  }
}
