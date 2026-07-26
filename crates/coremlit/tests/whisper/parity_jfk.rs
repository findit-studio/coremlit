//! End-to-end parity against Swift WhisperKit on jfk.wav / openai_whisper-tiny.
//!
//! Golden: tests/whisper/fixtures/golden/jfk_tiny_golden.json (see plan Task 13 for
//! the pinned whisperkit-cli invocation). Contract (spec §2.1): exact token
//! ids; segment boundaries within epsilon (timestamps are quantized to
//! 0.02 s tokens, so epsilon 1e-3 catches any real divergence).
//!
//! Second golden: tests/whisper/fixtures/golden/jfk_tiny_words_golden.json —
//! the same clip and model with **word timestamps on**, captured from the
//! same Swift library at the same pin (provenance and the exact
//! `DecodingOptions` in `tests/whisper_swift_probes/README.md`). It carries a
//! stricter contract than the token golden: every word matches exactly, no
//! epsilon. It exists because `alignment_gather` (whisper #41) runs on every
//! word-timestamp window, short-form included, and the token golden above
//! leaves `word_timestamps` off — so nothing here covered that path.
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

use coremlit::audio::whisper::{
  options::{
    AlignmentGather, ChunkingStrategy, DEFAULT_DECODER_COMPUTE_UNITS,
    DEFAULT_ENCODER_COMPUTE_UNITS, DEFAULT_MEL_COMPUTE_UNITS, DecodingOptions, Options,
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

// ---------------------------------------------------------------------
// Short-form word timestamps (whisper #41 follow-up)
// ---------------------------------------------------------------------

/// The Swift oracle's own word record, verbatim from
/// `fixtures/golden/jfk_tiny_words_golden.json`.
#[derive(serde::Deserialize)]
struct WordsGolden {
  text: String,
  language: String,
  #[serde(rename = "totalDecodingWindows")]
  total_decoding_windows: f64,
  segments: Vec<WordsGoldenSegment>,
}

#[derive(serde::Deserialize)]
struct WordsGoldenSegment {
  id: usize,
  seek: usize,
  start: f32,
  end: f32,
  text: String,
  tokens: Vec<u32>,
  words: Vec<GoldenWord>,
}

#[derive(serde::Deserialize)]
struct GoldenWord {
  word: String,
  start: f32,
  end: f32,
  probability: f32,
  tokens: Vec<u32>,
}

/// `(word, start, end, probability, tokens)` for every word of every
/// segment, flattened — the comparison unit for both directions below.
type WordRow = (String, f32, f32, f32, Vec<u32>);

fn golden_word_rows(golden: &WordsGolden) -> Vec<WordRow> {
  golden
    .segments
    .iter()
    .flat_map(|segment| {
      segment.words.iter().map(|word| {
        (
          word.word.clone(),
          word.start,
          word.end,
          word.probability,
          word.tokens.clone(),
        )
      })
    })
    .collect()
}

fn result_word_rows(
  result: &coremlit::audio::whisper::result::TranscriptionResult,
) -> Vec<WordRow> {
  result
    .segments_slice()
    .iter()
    .flat_map(|segment| {
      segment.words_slice().iter().map(|word| {
        (
          word.word().to_string(),
          word.start(),
          word.end(),
          word.probability(),
          word.tokens_slice().to_vec(),
        )
      })
    })
    .collect()
}

/// The options the Swift oracle ran under, mirrored exactly: this is the
/// same pinned invocation the whisper #41 long-form evidence used
/// (`crates/coremlit/tests/whisper_swift_probes/README.md`), so one option
/// set covers the short-form and long-form halves of the same claim.
///
/// `alignment_gather` is deliberately NOT set — the point is to exercise the
/// shipping default, which after the round-3 owner decision is
/// [`AlignmentGather::Complete`].
fn oracle_options() -> DecodingOptions {
  DecodingOptions::new()
    .with_temperature(0.0)
    .with_temperature_fallback_count(0)
    .with_use_prefill_prompt()
    .with_skip_special_tokens()
    .with_word_timestamps()
    .with_concurrent_worker_count(std::num::NonZeroUsize::new(1).unwrap())
    .with_chunking_strategy(ChunkingStrategy::Disabled)
}

/// jfk/tiny's word timestamps, pinned against official Swift and against
/// themselves across both [`AlignmentGather`] modes.
///
/// **Scope, stated first because it was once overstated.** This pins ONE
/// clip on ONE model. It does not establish — and must not be cited as
/// establishing — that the gather leaves short-form output alone in general.
/// It does not: `segment::tests::
/// swift_parity_gather_truncates_final_alignment_row` measures the two
/// gathers producing last-word and segment ends of 0.88 s against 1.58 s
/// over a single `add_word_timestamps` call, and `transcribe::tests::
/// swift_parity_gather_moves_the_first_windows_end_and_the_next_seek`
/// measures a first-window divergence with no cascade behind it. What this
/// test establishes is that jfk/tiny is not one of the clips where that
/// happens, and that under the shipping default it reproduces official
/// Swift.
///
/// **Why it exists.** `alignment_gather` (whisper #41) is selected for
/// EVERY word-timestamp window, not only for long-form or for windows after
/// the first: `TranscribeTask::run` passes `options.alignment_gather()`
/// straight into `add_word_timestamps` with no guard. So a caller who opts
/// into `SwiftParity` runs a single-window clip through the truncating gather
/// too: at 30 gathered rows over 1500 columns and the measured pitch of 1504
/// the copied prefix ends inside the last row, which keeps 1384 of its 1500
/// columns. What this test then compares is the two modes' WORD LISTS — it
/// inspects neither mode's DTW input, so what it records is that the OUTPUTS
/// agree on this clip, not anything about whether the inputs did. #41
/// asserted the short-form outcome with nothing committed checking it — the
/// sibling `jfk_tiny_matches_golden_tokens_and_segments` runs with
/// `word_timestamps` OFF.
///
/// Two assertions, in the order that matters:
///
/// 1. Under the shipping default ([`AlignmentGather::Complete`]) every word
///    matches the Swift oracle exactly — text, start, end, probability and
///    token ids, no epsilon. The default flipped in round 3 of the #41
///    review, so THIS is the claim that had to be re-established: the
///    correct-everywhere gather still reproduces official Swift on this clip.
/// 2. The opt-in [`AlignmentGather::SwiftParity`] produces the identical word
///    list *on this clip*. That is the whole of the gather-invariance the
///    branch has measured on real model output, and it is an assertion rather
///    than a comment: if the truncated final row ever does move this clip's
///    boundaries, this fails instead of the documented scope quietly becoming
///    wrong.
///
/// Compute path: the shipping default, for the same reason this file's
/// module doc gives — the oracle was captured on the ANE, so the gate runs
/// there too.
#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_tiny_word_timestamps_match_swift_and_this_clip_is_gather_invariant() {
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  let model_options = Options::new(common::tiny_dir(), common::tokenizer_dir());
  assert_eq!(model_options.compute().mel(), DEFAULT_MEL_COMPUTE_UNITS);
  assert_eq!(
    model_options.compute().encoder(),
    DEFAULT_ENCODER_COMPUTE_UNITS
  );
  assert_eq!(
    model_options.compute().decoder(),
    DEFAULT_DECODER_COMPUTE_UNITS
  );
  let kit = WhisperKit::new(&model_options).unwrap();

  let golden: WordsGolden = serde_json::from_str(
    &std::fs::read_to_string(common::fixtures_dir().join("golden/jfk_tiny_words_golden.json"))
      .unwrap(),
  )
  .unwrap();

  let default_options = oracle_options();
  assert_eq!(
    default_options.alignment_gather(),
    AlignmentGather::Complete,
    "this gate must run the shipping gather, not an explicit one"
  );
  let default_result = kit.transcribe(&audio, &default_options).unwrap();

  // (1) exact parity with Swift, under the DEFAULT (un-truncated) gather.
  assert_eq!(default_result.language(), golden.language);
  assert_eq!(default_result.text(), golden.text);
  assert_eq!(
    default_result.timings().total_decoding_windows(),
    golden.total_decoding_windows,
    "jfk is a single window; a second one would mean the seek moved"
  );
  assert_eq!(default_result.segments_slice().len(), golden.segments.len());
  for (rust, gold) in default_result.segments_slice().iter().zip(&golden.segments) {
    assert_eq!(rust.id(), gold.id);
    assert_eq!(rust.seek(), gold.seek);
    assert_eq!(rust.tokens_slice(), gold.tokens.as_slice());
    assert_eq!(rust.text(), gold.text);
    // Exact, not epsilon: the word-timestamp path is integer encoder
    // columns times a constant, so a real divergence lands whole frames
    // away and a tolerance would only hide it.
    assert_eq!((rust.start(), rust.end()), (gold.start, gold.end));
  }
  let default_words = result_word_rows(&default_result);
  assert_eq!(
    default_words,
    golden_word_rows(&golden),
    "jfk/tiny's word timestamps under the DEFAULT gather must match official Swift exactly"
  );
  assert_eq!(default_words.len(), 22, "jfk/tiny yields 22 words");

  // (2) the opt-in gather does not move any of it.
  let parity = kit
    .transcribe(
      &audio,
      &oracle_options().with_alignment_gather(AlignmentGather::SwiftParity),
    )
    .unwrap();
  assert_eq!(
    result_word_rows(&parity),
    default_words,
    "the #41 gather moved a word timing on jfk/tiny: the one clip on which the two gathers \
     were measured to agree no longer agrees, so the scope documented on \
     `AlignmentGather::SwiftParity` is now wrong too"
  );
  assert_eq!(parity.text(), default_result.text());
  assert_eq!(
    parity
      .segments_slice()
      .iter()
      .map(|s| (s.seek(), s.start(), s.end(), s.tokens_slice().to_vec()))
      .collect::<Vec<_>>(),
    default_result
      .segments_slice()
      .iter()
      .map(|s| (s.seek(), s.start(), s.end(), s.tokens_slice().to_vec()))
      .collect::<Vec<_>>(),
    "and on this clip neither the seek nor the segment bounds move with it"
  );
}
