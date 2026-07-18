//! Simulated-stream LocalAgreement-2 on jfk.wav / tiny (ports the
//! whisperkit-cli `transcribeStreamSimulated` loop, TranscribeCLI.swift:322-424).

mod common;

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_simulated_stream_confirms_the_transcript() {
  // `Options::new` takes both folders directly (two-arg constructor, not a
  // zero-arg `new()` plus `with_model_folder`/`with_tokenizer_folder`
  // builders) — same brief-vs-shipped-API fix as tests/pipeline.rs's
  // `tiny_options`/tests/parity_jfk.rs.
  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir())).unwrap();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  let mut streamer = kit.local_agreement_transcriber(DecodingOptions::new());
  // 1 s pushes — 11 strides, each re-transcribing the grown prefix.
  for chunk in audio.chunks(16_000) {
    streamer.push_samples(chunk).unwrap();
  }
  let final_result = streamer.finalize();
  let normalized = coremlit::audio::whisper::text::normalized(final_result.text());
  assert!(
    normalized.contains("ask not what your country can do for you"),
    "confirmed stream text diverged: {normalized}"
  );
}
