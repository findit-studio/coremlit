//! End-to-end parity against Swift WhisperKit on jfk.wav / openai_whisper-tiny.
//!
//! Golden: tests/fixtures/golden/jfk_tiny_golden.json (see plan Task 13 for
//! the pinned whisperkit-cli invocation). Contract (spec §2.1): exact token
//! ids; segment boundaries within epsilon (timestamps are quantized to
//! 0.02 s tokens, so epsilon 1e-3 catches any real divergence).
//!
//! Compute path — THE RULE: **a gate validating a shipping default must run
//! on the shipping default.** This test runs on the DEFAULT compute units
//! (mel CPU+GPU, encoder/decoder CPU+ANE — spec Goal 2, and byte-identical
//! to Swift's own `ModelComputeOptions` defaults, `Models.swift:92-118`),
//! because that is the path the crate ships AND the path `whisperkit-cli`
//! produced this golden on: an ANE-to-ANE external parity check. The
//! assertion below pins it, so a future `CpuOnly` pin fails loudly instead
//! of silently narrowing the coverage to a compute unit nobody runs.
//!
//! This is not hypothetical. The sibling crate `alignkit` shipped
//! `ComputeUnits::All` while every one of its tests pinned `CpuOnly`; when
//! the shipping path was finally exercised, the ANE returned a corrupted
//! output matrix (fp16 `log(0)` saturating to -45440 across 16.7% of cells,
//! words shifted by up to 881 ms). The suite was green throughout, because
//! it validated a compute unit the crate did not ship.
//!
//! `tests/pipeline.rs` and `tests/model_io.rs` may keep pinning `CpuOnly` —
//! they assert shapes and dtypes, not numerics. The golden tests own the
//! shipping compute path, and must never be pinned away from it.
//!
//! Numeric drift: this decode's greedy margins are THIN at two steps (step
//! 17 -> token 11, margin 0.1562; step 27 -> token 50889, margin 0.2500)
//! against a worst observed cross-placement logit delta of ~1.0. No flip
//! occurs on the development machine, but a different Apple Silicon
//! generation could flip one, and greedy autoregression would cascade it.
//! `common::assert_golden_tokens` reports the first diverging step's
//! competing tokens and their margin on failure, so that shows up as a
//! borderline argmax rather than a mystery. Suspect ANE drift before a
//! pipeline logic bug — but never "fix" either by regenerating the golden or
//! loosening the comparison.

mod common;

use whisperkit::{
  options::{
    DEFAULT_DECODER_COMPUTE_UNITS, DEFAULT_ENCODER_COMPUTE_UNITS, DEFAULT_MEL_COMPUTE_UNITS,
    DecodingOptions, Options,
  },
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
  let options = Options::new(common::tiny_dir(), common::tokenizer_dir());
  // THE RULE (see this file's module doc): this golden is an ANE-captured
  // Swift oracle, so the gate must run on the compute units the crate SHIPS.
  // Pinning any of these to `CpuOnly` — the tempting "fix" for a flaky
  // golden — would validate a path nobody runs. Fail here instead.
  assert_eq!(options.compute().mel(), DEFAULT_MEL_COMPUTE_UNITS);
  assert_eq!(options.compute().encoder(), DEFAULT_ENCODER_COMPUTE_UNITS);
  assert_eq!(options.compute().decoder(), DEFAULT_DECODER_COMPUTE_UNITS);
  let kit = WhisperKit::new(&options).unwrap();
  let result = kit.transcribe(&audio, &DecodingOptions::new()).unwrap();
  // Clean speech at temperature 0 must never draw from the token sampler —
  // the fallback ladder's t != 0 attempts sample from an unseeded RNG, so a
  // ladder-triggering regression would make this decode non-reproducible.
  // Asserted via the carried sampling flag, NOT
  // `total_decoding_fallbacks()`: that counter stores the ZERO-BASED index
  // of the last fallback (transcribe/mod.rs:846), so its FIRST fallback
  // writes 0.0 — indistinguishable from "never fell back", making
  // `== 0.0` vacuous. The flag is unambiguous, and also catches a sampled
  // window that a later lossy filter removed.
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(false),
    "clean speech must decode greedily; no window drew from the unseeded sampler"
  );

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

  // Keystone: exact token-id parity across the whole file. Exact — the
  // helper only DIAGNOSES a mismatch (first diverging step, the two
  // competing token ids, their logit margin); it never tolerates one.
  let rust_tokens: Vec<u32> = result
    .segments_slice()
    .iter()
    .flat_map(|s| s.tokens_slice().iter().copied())
    .collect();
  common::assert_golden_tokens("jfk_tiny_golden.json", &rust_tokens, &golden.tokens, &audio);

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
