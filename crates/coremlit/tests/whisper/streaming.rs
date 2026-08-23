//! Simulated-stream LocalAgreement-2 on jfk.wav / tiny (ports the
//! whisperkit-cli `transcribeStreamSimulated` loop, TranscribeCLI.swift:322-424).
//!
//! Host provenance. This gate owns no golden of its own — it asserts that the
//! CONFIRMED stream still contains the clip's canonical phrase — but it decodes
//! the same clip on the same model through the same shipping compute path as
//! `whisper_parity_jfk`, so the phrase it looks for is the Swift oracle's own
//! output and drifts with exactly the same fp16 argmax flips. It therefore
//! borrows `jfk_tiny_golden.json`'s `generationHost` as its host reference: a
//! golden stamped for a different host-class stops this test with the
//! regeneration diagnosis, and an unstamped one appends the ambiguity note to
//! the divergence message. Regeneration terms are `parity_jfk.rs`'s — from the
//! `whisperkit-cli` oracle only, never from this crate's own output.

mod common;

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};

/// The jfk golden whose host-class provenance this gate reads. It does not read
/// the golden's TOKENS — the LocalAgreement-2 confirmed prefix is a different
/// quantity from a full transcribe — only the host it was generated on.
const HOST_REFERENCE_GOLDEN: &str = "jfk_tiny_golden.json";

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_simulated_stream_confirms_the_transcript() {
  // Before any CoreML number: whose hardware produced the phrase below.
  let host_note = common::golden_host_note(
    HOST_REFERENCE_GOLDEN,
    &common::load_golden_json(HOST_REFERENCE_GOLDEN),
  );
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
    "confirmed stream text diverged: {normalized}\n\n  \
     This phrase is the Swift oracle's own transcript of jfk.wav on tiny. A greedy \
     argmax\n  flipped by host fp16 drift cascades through the decode and can rewrite \
     it, so read\n  the note below before treating this as a LocalAgreement-2 defect. \
     Do NOT weaken the\n  phrase to make this pass.{host_note}"
  );
}
