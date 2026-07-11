//! End-to-end parity against Swift WhisperKit on jfk.wav / openai_whisper-tiny.
//!
//! Golden: tests/fixtures/golden/jfk_tiny_golden.json (see plan Task 13 for
//! the pinned whisperkit-cli invocation). Contract (spec §2.1): exact token
//! ids; segment boundaries within epsilon (timestamps are quantized to
//! 0.02 s tokens, so epsilon 1e-3 catches any real divergence).

mod common;

use whisperkit::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};

#[derive(serde::Deserialize)]
struct Golden {
  text: String,
  language: String,
  tokens: Vec<u32>,
  segments: Vec<GoldenSegment>,
}

#[derive(serde::Deserialize)]
struct GoldenSegment {
  id: usize,
  start: f32,
  end: f32,
  text: String,
  tokens: Vec<u32>,
}

fn golden_path() -> std::path::PathBuf {
  common::fixtures_dir().join("golden/jfk_tiny_golden.json")
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_tiny_matches_golden_tokens_and_segments() {
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  // `Options::new` takes both folders directly (two-arg constructor, not a
  // zero-arg `new()` plus `with_model_folder`/`with_tokenizer_folder`
  // builders) — same brief-vs-shipped-API fix as tests/pipeline.rs's
  // `tiny_options`.
  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir())).unwrap();
  let result = kit.transcribe(&audio, &DecodingOptions::new()).unwrap();

  if std::env::var_os("UPDATE_GOLDEN").is_some() {
    // Fallback-path writer (plan Task 13 Step 1-FALLBACK): pin the Rust
    // output as the golden. Human verification + decision-issue REQUIRED.
    let doc = serde_json::json!({
        "model": "openai_whisper-tiny",
        "source": "rust-coreml (self-golden); swift cross-check pending",
        "text": result.text(),
        "language": result.language(),
        "tokens": result.segments_slice().iter().flat_map(|s| s.tokens_slice().iter().copied()).collect::<Vec<u32>>(),
        "segments": result.segments_slice().iter().map(|s| serde_json::json!({
            "id": s.id(), "start": s.start(), "end": s.end(),
            "text": s.text(), "tokens": s.tokens_slice(),
        })).collect::<Vec<_>>(),
    });
    std::fs::write(golden_path(), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("golden written — human-verify the transcript, then re-run without UPDATE_GOLDEN");
    return;
  }

  let golden: Golden =
    serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();

  assert_eq!(golden.language, result.language());

  // Keystone: exact token-id parity across the whole file.
  let rust_tokens: Vec<u32> = result
    .segments_slice()
    .iter()
    .flat_map(|s| s.tokens_slice().iter().copied())
    .collect();
  assert_eq!(rust_tokens, golden.tokens, "token ids must match exactly");

  // Segment-level parity: count, ids, boundaries within epsilon, text.
  assert_eq!(result.segments_slice().len(), golden.segments.len());
  const EPSILON: f32 = 1e-3;
  for (rust, gold) in result.segments_slice().iter().zip(&golden.segments) {
    assert_eq!(rust.id(), gold.id);
    assert!(
      (rust.start() - gold.start).abs() < EPSILON,
      "start {} vs {}",
      rust.start(),
      gold.start
    );
    assert!(
      (rust.end() - gold.end).abs() < EPSILON,
      "end {} vs {}",
      rust.end(),
      gold.end
    );
    assert_eq!(rust.tokens_slice(), gold.tokens.as_slice());
    assert_eq!(rust.text(), gold.text);
  }
  assert_eq!(result.text(), golden.text);
}
