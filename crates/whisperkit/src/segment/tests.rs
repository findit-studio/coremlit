use super::*;
use crate::{
  backend::AlignmentView,
  constants::{APPEND_PUNCTUATION, PREPEND_PUNCTUATION},
  options::DecodingOptions,
  result::{DecodingResult, WordTiming},
  tokenizer::{SpecialTokens, WhisperTokenizer},
};

fn tiny_tokenizer() -> WhisperTokenizer {
  let root = std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    std::path::PathBuf::from,
  );
  WhisperTokenizer::from_folder(root.join("tokenizers/whisper-tiny")).unwrap()
}

fn ts(index: u32) -> u32 {
  SpecialTokens::whisper_defaults().time_token_begin() + index
}

fn result_with_tokens(tokens: Vec<u32>, no_speech: f32, avg_logprob: f32) -> DecodingResult {
  let log_probs: Vec<(u32, f32)> = tokens.iter().map(|&t| (t, -0.1)).collect();
  let mut r = DecodingResult::new();
  r.set_tokens(tokens)
    .set_token_log_probs(log_probs)
    .set_no_speech_prob(no_speech)
    .set_avg_logprob(avg_logprob);
  r
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn silence_skips_full_segment() {
  // SegmentSeeker.swift:57-74: noSpeech 0.9 > 0.6 and avgLogProb -1.5
  // NOT > -1.0 -> skip; seek advances one full segment; no segments.
  let t = tiny_tokenizer();
  let r = result_with_tokens(vec![], 0.9, -1.5);
  let (seek, segments) =
    find_seek_point_and_segments(&r, &DecodingOptions::new(), 0, 16_000, 480_000, &t).unwrap();
  assert_eq!(seek, 16_000 + 480_000);
  assert!(segments.is_none());

  // Confident text overrides silence: avgLogProb -0.2 > -1.0 -> not skipped.
  let r = result_with_tokens(vec![50258, 100, ts(0), ts(50)], 0.9, -0.2);
  let (_, segments) =
    find_seek_point_and_segments(&r, &DecodingOptions::new(), 0, 0, 480_000, &t).unwrap();
  assert!(segments.is_some());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn consecutive_timestamps_slice_into_segments_and_seek_to_last() {
  // Tokens: <|0.00|> hello <|1.00|> <|1.00|> world <|2.00|> EOT — two
  // segments; double-timestamp ending -> seek advances by the LAST
  // timestamp (2.00 s).
  //
  // Correction to this task's brief: the brief's token list ended at
  // `ts(100)` with no trailing token. Without a trailing non-timestamp
  // token, `isTimestampToken`'s last three flags are
  // `[true, false, true]` (ts(50), world, ts(100)) — neither the
  // singleTimestampEnding ([false, true, false]) nor noTimestampEnding
  // ([false, false, false]) pattern (SegmentSeeker.swift:84-86) — so
  // `sliceIndexes` stays at its single main-loop entry ([3]) and the
  // algorithm emits only ONE segment, with seek advancing by the FIRST
  // internal timestamp (1.00 s), not the two segments / 2.00 s advance
  // this test's own name and comments describe. Every real
  // `DecodingResult` this function actually receives ends in EOT
  // (`decode::finalize_decoding_result`, `TextDecoder.swift:780-783`),
  // which makes the true trailing flags [false, true, false]
  // (world, ts(100), EOT) -> singleTimestampEnding, producing the second
  // segment and 2.00 s seek advance below.
  let t = tiny_tokenizer();
  let hello = 15947u32; // any word-token id below specialTokenBegin works
  let world = 1002u32;
  let tokens = vec![
    ts(0),
    hello,
    ts(50),
    ts(50),
    world,
    ts(100),
    t.special_tokens().end_token(),
  ];
  let r = result_with_tokens(tokens, 0.0, -0.2);
  let (seek, segments) =
    find_seek_point_and_segments(&r, &DecodingOptions::new(), 3, 32_000, 480_000, &t).unwrap();
  let segments = segments.unwrap();
  assert_eq!(segments.len(), 2);
  // timeOffset = 32000/16000 = 2.0 s (SegmentSeeker.swift:55)
  assert_eq!(segments[0].id(), 3); // allSegmentsCount + index (:124)
  assert!((segments[0].start() - 2.0).abs() < 1e-4);
  assert!((segments[0].end() - 3.0).abs() < 1e-4);
  assert!((segments[1].start() - 3.0).abs() < 1e-4);
  assert!((segments[1].end() - 4.0).abs() < 1e-4);
  assert_eq!(segments[0].seek(), 32_000);
  // seek += lastTimestamp(2.00 s) * 16000 (:140-145)
  assert_eq!(seek, 32_000 + 32_000);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn single_timestamp_ending_appends_final_slice() {
  // Single-timestamp ending = last three timestamp-flags [false, true, false]
  // (SegmentSeeker.swift:84-86). Tokens ts(0) a ts(50) ts(50) b ts(75) c end
  // with [b(text), ts(75), c(text)] -> single ending.
  let t = tiny_tokenizer();
  let tokens = vec![ts(0), 100, ts(50), ts(50), 101, ts(75), 102];
  let r = result_with_tokens(tokens, 0.0, -0.2);
  let (seek, segments) =
    find_seek_point_and_segments(&r, &DecodingOptions::new(), 0, 0, 480_000, &t).unwrap();
  let segments = segments.unwrap();
  // Slice ends: pair at index 3, then appended lastIndex(ts)+1 = 6 (:100-107)
  assert_eq!(segments.len(), 2);
  assert!((segments[1].end() - 1.5).abs() < 1e-4);
  // Single ending: seek uses tokens[lastSliceStart - 1] = ts(75) (:141-145)
  assert_eq!(seek, (1.5 * 16_000.0) as usize);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn no_consecutive_timestamps_lumps_window_and_refines_duration() {
  // SegmentSeeker.swift:149-186: one segment covering the window; a lone
  // nonzero trailing timestamp refines the end time; seek += segmentSize.
  let t = tiny_tokenizer();
  let tokens = vec![ts(0), 100, 101, ts(150)]; // 3.00 s end
  let r = result_with_tokens(tokens, 0.0, -0.2);
  let (seek, segments) =
    find_seek_point_and_segments(&r, &DecodingOptions::new(), 0, 0, 160_000, &t).unwrap();
  let segments = segments.unwrap();
  assert_eq!(segments.len(), 1);
  assert!((segments[0].start() - 0.0).abs() < 1e-4);
  assert!(
    (segments[0].end() - 3.0).abs() < 1e-4,
    "refined by trailing timestamp"
  );
  assert_eq!(seek, 160_000);

  // Without any timestamp > timeTokenBegin: duration = segmentSize/sampleRate.
  let r = result_with_tokens(vec![ts(0), 100, 101], 0.0, -0.2);
  let (_, segments) =
    find_seek_point_and_segments(&r, &DecodingOptions::new(), 0, 0, 160_000, &t).unwrap();
  assert!((segments.unwrap()[0].end() - 10.0).abs() < 1e-4);
}

// SegmentSeeker.swift:195-278 -- dynamic time warping over an alignment matrix.

#[test]
fn dtw_diagonal_identity() {
  // Correction to this task's brief: the brief names this test for a
  // "strong diagonal" path and expects `[0, 1, 2]`, assuming the diagonal
  // wins cost ties. `minCostAndTrace` (`SegmentSeeker.swift:239-251`)
  // does not: an exact three-way cost tie falls through to its final
  // `else` and picks LEFT, never the diagonal. A perfect identity
  // matrix's 0.0 off-diagonal cells create exactly this tie repeatedly
  // (verified by hand-tracing the cost/trace matrices, and independently
  // cross-checked against Swift's own passing
  // `testDynamicTimeWarpingSimpleMatrix` ground truth in
  // `dtw_matches_swift_unit_test_ground_truth` below), so the real path
  // revisits row 1 and row 2 once each via LEFT steps before advancing.
  #[rustfmt::skip]
  let matrix = [
    1.0f32, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  ];
  let view = AlignmentView::new(&matrix, 3, 3);
  let path = dynamic_time_warping(&view).unwrap();
  assert_eq!(path.text_indices_slice(), &[0, 1, 1, 2, 2]);
  assert_eq!(path.time_indices_slice(), &[0, 0, 1, 1, 2]);
}

#[test]
fn dtw_wide_matrix_repeats_text_indices() {
  // 2 tokens x 4 frames: token 0 aligned to frames 0-1, token 1 to
  // 2-3 -- see the correction below for why the real path has one extra
  // step.
  //
  // Correction to this task's brief: the brief expects
  // `text_indices=[0,0,1,1]`/`time_indices=[0,1,2,3]` (four steps). The
  // actual path has five: at row 2/column 3, `up` (-1.9) and `left`
  // (-1.9) are an exact cost tie, and `minCostAndTrace`'s strict `<` +
  // final-`else` structure (`SegmentSeeker.swift:239-251`) makes LEFT
  // win ties, not UP -- so the backtrace takes one extra LEFT step at
  // column 3 before reaching column 4, repeating text index 1 a third
  // time.
  #[rustfmt::skip]
  let matrix = [
    0.9f32, 0.9, 0.1, 0.1,
    0.1,    0.1, 0.9, 0.9,
  ];
  let view = AlignmentView::new(&matrix, 2, 4);
  let path = dynamic_time_warping(&view).unwrap();
  assert_eq!(path.text_indices_slice(), &[0, 0, 1, 1, 1]);
  assert_eq!(path.time_indices_slice(), &[0, 1, 1, 2, 3]);
}

#[test]
fn dtw_matches_swift_unit_test_ground_truth() {
  // Cross-check against Swift's own ground truth for the exact matrix
  // `testDynamicTimeWarpingSimpleMatrix` uses (`UnitTests.swift:
  // 2337-2367`) -- independent confirmation, beyond hand-tracing, that
  // this port's tie-breaking matches a real, passing upstream Swift
  // assertion and not just this task's own (corrected, see below)
  // synthetic test matrices.
  #[rustfmt::skip]
  let matrix = [
    1.0f32, 1.0, 1.0,
    5.0, 2.0, 1.0,
    1.0, 5.0, 2.0,
  ];
  let view = AlignmentView::new(&matrix, 3, 3);
  let path = dynamic_time_warping(&view).unwrap();
  assert_eq!(path.text_indices_slice(), &[0, 1, 1, 2, 2]);
  assert_eq!(path.time_indices_slice(), &[0, 0, 1, 1, 2]);
}

#[test]
fn dtw_rejects_empty() {
  let view = AlignmentView::new(&[], 0, 0);
  assert!(dynamic_time_warping(&view).is_err());
}

fn word(text: &str, start: f32, end: f32) -> WordTiming {
  // Passing `text` bare (not `text.into()`): with two `impl Into<_>`
  // parameters in the same call, `.into()`'s target type is unresolvable
  // (E0283) even though `&str: Into<String>` is the only fit -- the
  // callee's own bound performs the conversion instead.
  WordTiming::new(text, vec![1], start, end, 0.9)
}

#[test]
fn merge_punctuations_english() {
  // Ports testMergePunctuations shape: " Hey" "," " you" "!" -> " Hey," " you!"
  let alignment = [
    word(" Hey", 0.0, 0.2),
    word(",", 0.2, 0.3),
    word(" you", 0.3, 0.6),
    word("!", 0.6, 0.7),
  ];
  let merged = merge_punctuations(&alignment, PREPEND_PUNCTUATION, APPEND_PUNCTUATION);
  let words: Vec<&str> = merged.iter().map(|w| w.word()).collect();
  assert_eq!(words, vec![" Hey,", " you!"]);
  assert_eq!(merged[0].tokens_slice().len(), 2, "tokens concatenated");
}

#[test]
fn merge_punctuations_prepended() {
  // A leading prepend punctuation (space + quote) glues onto the NEXT word
  // (SegmentSeeker.swift:296-315).
  let alignment = [
    word(" \u{00bf}", 0.0, 0.1),
    word("Que", 0.1, 0.4),
    word("?", 0.4, 0.5),
  ];
  let merged = merge_punctuations(&alignment, PREPEND_PUNCTUATION, APPEND_PUNCTUATION);
  let words: Vec<&str> = merged.iter().map(|w| w.word()).collect();
  assert_eq!(words, vec![" \u{00bf}Que?"]);
}

#[test]
fn merge_punctuations_ignores_whitespace_only_words() {
  // Regression (task-7 review, Important): a word that trims to nothing
  // (a standalone space BPE token can form one) must match NO punctuation
  // set — Swift's `String.contains("")` is false. `str::contains("")` is
  // true, which would glue the space word onto its neighbor as prepend
  // punctuation.
  let alignment = [
    word(" a", 0.0, 0.2),
    word(" ", 0.2, 0.3),
    word(" b", 0.3, 0.6),
  ];
  let merged = merge_punctuations(&alignment, PREPEND_PUNCTUATION, APPEND_PUNCTUATION);
  let words: Vec<&str> = merged.iter().map(|w| w.word()).collect();
  assert_eq!(words, vec![" a", " ", " b"], "no merges, nothing dropped");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn find_alignment_produces_monotonic_word_timings() {
  let t = tiny_tokenizer();
  let ids = t.encode(" Hello world again").unwrap();
  // Diagonal-ish synthetic weights: token i peaks at frame i * 10.
  let cols = 100usize;
  let mut matrix = vec![0.0f32; ids.len() * cols];
  for (i, row) in matrix.chunks_mut(cols).enumerate() {
    row[i * 10] = 1.0;
  }
  let view = AlignmentView::new(&matrix, ids.len(), cols);
  let log_probs = vec![-0.2f32; ids.len()];
  let words = find_alignment(&ids, &view, &log_probs, &t, "en").unwrap();
  assert!(!words.is_empty());
  for pair in words.windows(2) {
    assert!(pair[0].end() <= pair[1].start() + 1e-4, "monotonic timings");
  }
  for w in &words {
    assert!((0.0..=1.0).contains(&w.probability()));
  }
}

#[test]
fn dtw_add_before_compare_matches_swift_rounding_ties() {
  // Regression (phase-gate round 2): Swift adds the cell value BEFORE
  // comparing (SegmentSeeker.swift:239-251); a large-magnitude value can
  // round distinct incoming costs into exact ties, which fall to left.
  // Comparing the bare incoming costs picks up (strictly smallest) here
  // and walks a different path.
  #[rustfmt::skip]
  let matrix = [
    0.0f32, 1.0,
    0.0,    1.0e30,
  ];
  let view = AlignmentView::new(&matrix, 2, 2);
  let path = dynamic_time_warping(&view).unwrap();
  assert_eq!(path.text_indices_slice(), &[0, 1, 1]);
  assert_eq!(path.time_indices_slice(), &[0, 0, 1]);
}

// SegmentSeeker.swift:498-526 -- word-duration constraints and
// sentence-boundary truncation. Reuses the `word` helper above rather than
// a separate `timing` helper of the same shape.

#[test]
fn duration_constraints_take_the_capped_upper_median() {
  // SegmentSeeker.swift:498-507. Durations 0.2/0.4/0.6 (plus one
  // zero-length word that must be filtered): sorted[3/2] = sorted[1] = 0.4.
  let alignment = [
    word("a", 0.0, 0.2),
    word("b", 0.2, 0.6),
    word("c", 0.6, 1.2),
    word("z", 1.2, 1.2), // zero duration -> filtered before the median
  ];
  let constraints = calculate_word_duration_constraints(&alignment);
  assert!((constraints.median() - 0.4).abs() < 1e-6);
  assert!((constraints.max_duration() - 0.8).abs() < 1e-6);

  // Median above the cap clamps to 0.7 (max 1.4).
  let long = [word("a", 0.0, 1.0), word("b", 1.0, 2.0)];
  let constraints = calculate_word_duration_constraints(&long);
  assert!((constraints.median() - 0.7).abs() < 1e-6);
  assert!((constraints.max_duration() - 1.4).abs() < 1e-6);

  // Empty (or all-zero-duration) input -> zeros.
  let constraints = calculate_word_duration_constraints(&[]);
  assert_eq!(constraints.median(), 0.0);
  assert_eq!(constraints.max_duration(), 0.0);
}

#[test]
fn truncation_fires_only_at_sentence_boundaries() {
  // SegmentSeeker.swift:509-526.
  // Case A: the overlong word IS a sentence mark -> end pulled to start+max.
  let alignment = vec![word(" ok", 0.0, 0.3), word(".", 0.3, 2.0)];
  let out = truncate_long_words_at_sentence_boundaries(alignment, 0.5);
  assert!((out[1].end() - 0.8).abs() < 1e-6);

  // Case B: the PREVIOUS word is a mark -> start pulled to end-max.
  let alignment = vec![word("!", 0.0, 0.1), word(" Next", 0.1, 2.0)];
  let out = truncate_long_words_at_sentence_boundaries(alignment, 0.5);
  assert!((out[1].start() - 1.5).abs() < 1e-6);

  // Case C: no boundary involvement -> untouched, even when overlong;
  // and " ." with whitespace is NOT a mark (exact whole-word match).
  let alignment = vec![
    word(" a", 0.0, 0.1),
    word(" long", 0.1, 3.0),
    word(" .", 3.0, 6.0),
  ];
  let out = truncate_long_words_at_sentence_boundaries(alignment, 0.5);
  assert_eq!(out[1].end(), 3.0);
  assert_eq!(out[2].end(), 6.0);

  // Index 0 is never truncated (loop starts at 1).
  let alignment = vec![word(".", 0.0, 5.0)];
  let out = truncate_long_words_at_sentence_boundaries(alignment, 0.5);
  assert_eq!(out[0].end(), 5.0);
}

// FoundationExtensions.swift:9-13 -- Float.rounded(_:), half-away-from-zero.
// Correction to this task's brief, which cited :8-12: line 8 is the
// enclosing `extension Float {`, and the function's closing brace is on
// line 13, not included in :8-12.

#[test]
fn rounded_to_places_matches_swift_rounding() {
  assert_eq!(rounded_to_places(1.234, 2), 1.23);
  assert_eq!(rounded_to_places(1.235, 2), 1.24);
  assert_eq!(rounded_to_places(-1.235, 2), -1.24);
}
