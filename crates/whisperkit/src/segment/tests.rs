use std::path::PathBuf;

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
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
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
fn even_count_median_takes_the_upper_middle_value() {
  // Review finding: sorted[count/2] vs sorted[(count-1)/2] needs a
  // genuinely even, distinct-valued input to discriminate. Durations
  // 0.2/0.4/0.6/0.8 -> sorted[2] = 0.6 (the UPPER middle), max 1.2; the
  // lower-median regression would report 0.4/0.8.
  let alignment = [
    word("a", 0.0, 0.2),
    word("b", 0.2, 0.6),
    word("c", 0.6, 1.2),
    word("d", 1.2, 2.0),
  ];
  let constraints = calculate_word_duration_constraints(&alignment);
  assert!((constraints.median() - 0.6).abs() < 1e-6);
  assert!((constraints.max_duration() - 1.2).abs() < 1e-6);
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

  // Case D (review finding): BOTH branches hold — the overlong word is a
  // mark AND its predecessor is a mark. Swift's if/else-if takes the
  // first branch only: end is pulled in, start stays.
  let alignment = vec![word("!", 0.0, 0.1), word(".", 0.1, 2.0)];
  let out = truncate_long_words_at_sentence_boundaries(alignment, 0.5);
  assert!(
    (out[1].end() - 0.6).abs() < 1e-6,
    "first branch: end = start + max"
  );
  assert_eq!(out[1].start(), 0.1, "second branch must not also fire");

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

// SegmentSeeker.swift:528-659 -- update_segments_with_word_timings: the
// final word-timing re-anchoring step `addWordTimestamps` runs after
// `findAlignment` -> duration constraints/truncation -> mergePunctuations.

fn plain_segment(tokens: Vec<u32>, start: f32, end: f32) -> TranscriptionSegment {
  let mut segment = TranscriptionSegment::new();
  segment.set_tokens(tokens).set_start(start).set_end(end);
  segment
}

fn aligned(text: &str, tokens: Vec<u32>, start: f32, end: f32) -> WordTiming {
  // Passing `text` bare, not `text.into()`: see the `word` helper above --
  // the same E0283 trap applies here (`WordTiming::new` takes two
  // `impl Into<_>` parameters in this one call).
  WordTiming::new(text, tokens, start, end, 0.9)
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_walk_assigns_words_and_pulls_short_words_back() {
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let world = t.encode(" world").unwrap()[0];
  let segments = [plain_segment(vec![hello, world], 0.0, 2.0)];
  // Second word is near-zero-length with a 1.4 s gap: start moves back by
  // min(gap, median/2) = 0.3 (SegmentSeeker.swift:564-583).
  let alignment = [
    aligned(" Hello", vec![hello], 0.0, 0.5),
    aligned(" world", vec![world], 1.9, 2.0),
  ];
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 0, 0.0, 0.6, 1.2, &t).unwrap();
  assert_eq!(updated.len(), 1);
  let words = updated[0].words_slice();
  assert_eq!(words.len(), 2);
  assert!(
    (words[1].start() - 1.6).abs() < 1e-4,
    "0.1s word pulled back by median/2"
  );
  assert!((words[1].end() - 2.0).abs() < 1e-4);
  // Segment boundaries follow the words (:636-649 else-branches).
  assert!((updated[0].start() - 0.0).abs() < 1e-4);
  assert!((updated[0].end() - 2.0).abs() < 1e-4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn special_only_alignment_entries_are_skipped() {
  // SegmentSeeker.swift:551-554: a timing whose tokens are all specials is
  // consumed from the cursor but emits no word.
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  let segments = [plain_segment(vec![hello], 0.0, 1.0)];
  let alignment = [
    aligned("<|0.00|>", vec![s.time_token_begin()], 0.0, 0.0),
    aligned(" Hello", vec![hello], 0.0, 0.5),
  ];
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 0, 0.0, 0.6, 1.2, &t).unwrap();
  let words = updated[0].words_slice();
  assert_eq!(words.len(), 1);
  assert_eq!(words[0].word(), " Hello");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn seek_offset_and_pause_hack_apply() {
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  // seek 32000 -> +2.0 s offset (:537). Pause: last_speech_timestamp 0,
  // first word ends at 2.0+3.0=5.0 -> pause 5.0 > 0.6*4; word duration
  // 3.0 > max 1.2 -> w0.start = max(0, 5.0 - 1.2) = 3.8 (:615-632).
  let segments = [plain_segment(vec![hello], 2.0, 5.0)];
  let alignment = [aligned(" Hello", vec![hello], 0.0, 3.0)];
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 32_000, 0.0, 0.6, 1.2, &t).unwrap();
  let words = updated[0].words_slice();
  assert!((words[0].end() - 5.0).abs() < 1e-4, "offset applied");
  assert!(
    (words[0].start() - 3.8).abs() < 1e-4,
    "pause-hack clamped the first word"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_index_cursor_is_shared_and_previous_segment_gap_pulls_first_word_back() {
  // SegmentSeeker.swift:538: `wordIndex` is a single cursor walked across
  // every segment, never reset per segment; and :584-595: a segment's own
  // first word (only reachable while its local `wordsInSegment` is still
  // empty) pulls back against the PREVIOUS segment's already-finalized
  // `end`, not a word in this segment.
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let world = t.encode(" world").unwrap()[0];
  let segments = [
    plain_segment(vec![hello], 0.0, 1.0),
    plain_segment(vec![world], 2.0, 3.0),
  ];
  let alignment = [
    aligned(" Hello", vec![hello], 0.0, 1.0),
    aligned(" world", vec![world], 2.4, 2.45),
  ];
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 0, 0.0, 0.6, 1.2, &t).unwrap();
  assert_eq!(updated.len(), 2);
  assert!((updated[0].end() - 1.0).abs() < 1e-4);
  let second_words = updated[1].words_slice();
  assert_eq!(
    second_words.len(),
    1,
    "cursor advanced past segment 0's word, not reused"
  );
  // gap = 2.4 - 1.0 = 1.4; desired = min(1.4, 0.6/2=0.3) = 0.3 -> 2.4-0.3=2.1.
  assert!(
    (second_words[0].start() - 2.1).abs() < 1e-4,
    "first word of segment 1 pulled back against segment 0's end"
  );
  assert!((second_words[0].end() - 2.45).abs() < 1e-4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn second_word_too_long_triggers_boundary_resplit() {
  // SegmentSeeker.swift:621-633: an over-long pause followed by a
  // too-long SECOND word re-splits the 0/1 boundary at
  // max(w1.end/2, w1.end-max) before clamping the first word's start.
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let world = t.encode(" world").unwrap()[0];
  let segments = [plain_segment(vec![hello, world], 0.0, 10.0)];
  let alignment = [
    aligned(" Hello", vec![hello], 3.0, 3.3),
    aligned(" world", vec![world], 3.3, 6.0),
  ];
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 0, 0.0, 0.6, 1.2, &t).unwrap();
  let words = updated[0].words_slice();
  assert_eq!(words.len(), 2);
  // boundary = max(6.0/2=3.0, 6.0-1.2=4.8) = 4.8.
  assert!((words[0].end() - 4.8).abs() < 1e-4, "resplit boundary");
  assert!((words[1].start() - 4.8).abs() < 1e-4, "resplit boundary");
  // w0.start = max(last_speech_timestamp=0, w0.end(4.8) - max(1.2)) = 3.6.
  assert!(
    (words[0].start() - 3.6).abs() < 1e-4,
    "first word clamped after resplit"
  );
  assert!(
    (words[1].end() - 6.0).abs() < 1e-4,
    "second word's end untouched by resplit"
  );
  assert!((updated[0].start() - 3.6).abs() < 1e-4);
  assert!((updated[0].end() - 6.0).abs() < 1e-4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn segment_level_bounds_preferred_when_words_drift_far_from_segment() {
  // SegmentSeeker.swift:635-640 and :642-649's IF branches (not the
  // ubiquitous else): the first/last word's timing is replaced by a
  // segment-anchored clamp when it has drifted more than half a second
  // from the segment's own start/end.
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let segments = [plain_segment(vec![hello], 3.0, 5.0)];
  let alignment = [aligned(" Hello", vec![hello], 2.0, 10.0)];
  // last_speech_timestamp = 9.9 keeps the pause (:618-621) small, so the
  // pause-hack itself never fires -- isolating the segment-bounds
  // preference branches below.
  // median 2.5 (review follow-up): the end clamp's word-anchored term
  // must WIN its max so a stale pre-clamp `last.start` read becomes
  // detectable — live 3.0 + 2.5 = 5.5 beats segment.end 5.0, where the
  // stale 2.0 + 2.5 = 4.5 would collapse back to 5.0.
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 0, 9.9, 2.5, 1.2, &t).unwrap();
  let words = updated[0].words_slice();
  assert_eq!(words.len(), 1);
  // start: segment.start(3.0) < w0.end(10.0) && segment.start-0.5=2.5 >
  // w0.start(2.0) -> true -> clamped to segment.start = 3.0.
  assert!(
    (words[0].start() - 3.0).abs() < 1e-4,
    "segment start preferred"
  );
  // end: updatedSegment.end(5.0) > lastWord.start(3.0, POST-clamp) &&
  // segment.end+0.5=5.5 < lastWord.end(10.0) -> true -> max(3.0+2.5,
  // 5.0) = 5.5 — the word-anchored term, provably reading the mutated
  // start.
  assert!(
    (words[0].end() - 5.5).abs() < 1e-4,
    "word-anchored clamp term reads the live start"
  );
  assert!((updated[0].start() - 3.0).abs() < 1e-4);
  assert!(
    (updated[0].end() - 5.0).abs() < 1e-4,
    "IF branch leaves segment.end"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn mixed_special_and_word_tokens_retokenize_the_surviving_ones() {
  // SegmentSeeker.swift:556-559: when only SOME of a timing's tokens are
  // filtered out (not all -- that's the :551-554 skip case), the word is
  // retokenized from just the survivors rather than reusing the timing's
  // own (here deliberately wrong) `.word` text.
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  let segments = [plain_segment(vec![hello], 0.0, 1.0)];
  let alignment = [aligned(
    "WRONG",
    vec![s.time_token_begin(), hello],
    0.0,
    0.5,
  )];
  let updated =
    update_segments_with_word_timings(&segments, &alignment, 0, 0.0, 0.6, 1.2, &t).unwrap();
  let words = updated[0].words_slice();
  assert_eq!(words.len(), 1);
  assert_eq!(
    words[0].word(),
    " Hello",
    "retokenized from the surviving token, not `.word`"
  );
  assert_eq!(
    words[0].tokens_slice().to_vec(),
    vec![hello],
    "special token filtered out of stored tokens too"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn empty_segments_returns_empty_without_consuming_alignment() {
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let alignment = [aligned(" Hello", vec![hello], 0.0, 0.5)];
  let updated = update_segments_with_word_timings(&[], &alignment, 0, 0.0, 0.6, 1.2, &t).unwrap();
  assert!(updated.is_empty());
}

// SegmentSeeker.swift:410-496 -- add_word_timestamps: the orchestration
// wrapper composing gather -> prefix-take/zero-pad -> find_alignment ->
// duration constraints/truncation -> merge_punctuations -> word-timing
// re-anchoring.

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn add_word_timestamps_attaches_merged_monotonic_words() {
  // End-to-end over the pure stack: DTW -> find_alignment -> constraints ->
  // truncation -> merge_punctuations -> segment updates. Timestamp tokens
  // ride along exactly as in the real flow (SegmentSeeker.swift:427-442
  // gathers ALL segment tokens; specials drop inside the word split /
  // update walk). Verified rather than assumed: these ids resolve in the
  // tiny tokenizer's real vocabulary, so `split_to_word_tokens` decodes
  // them without error, and it is `update_segments_with_word_timings`'s
  // own special-token filter (:551-554) that drops them from the joined
  // text asserted below -- no correction to Plan 2's split was needed.
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  let world = t.encode(" world").unwrap()[0];
  let tokens = vec![
    s.time_token_begin(),
    hello,
    world,
    s.time_token_begin() + 100,
  ];
  let log_probs: Vec<(u32, f32)> = tokens.iter().map(|&tok| (tok, -0.2)).collect();
  let mut segment = TranscriptionSegment::new();
  segment
    .set_tokens(tokens)
    .set_token_log_probs(log_probs)
    .set_start(0.0)
    .set_end(2.0);

  // 4 token rows x 150 frames; row i peaks at frame i*25 (0.5 s apart).
  let cols = 150usize;
  let mut weights = vec![0.0f32; 4 * cols];
  for (i, row) in weights.chunks_mut(cols).enumerate() {
    row[i * 25] = 1.0;
  }
  let view = AlignmentView::new(&weights, 4, cols);

  let updated = add_word_timestamps(
    &[segment],
    &view,
    &t,
    "en",
    0,
    crate::constants::PREPEND_PUNCTUATION,
    crate::constants::APPEND_PUNCTUATION,
    0.0,
  )
  .unwrap();
  assert_eq!(updated.len(), 1);
  let words = updated[0].words_slice();
  assert!(!words.is_empty(), "text tokens produced word timings");
  let joined: String = words.iter().map(|w| w.word()).collect();
  assert_eq!(crate::text::normalized(&joined), "hello world");
  for pair in words.windows(2) {
    assert!(
      pair[0].start() <= pair[1].start() + 1e-4,
      "monotonic starts"
    );
  }
  for word in words {
    assert!(word.end() >= word.start());
    assert!((0.0..=1.0).contains(&word.probability()));
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn add_word_timestamps_zero_pads_missing_rows() {
  // More gathered tokens than written alignment rows: Swift reads zeros
  // from its preallocation (:444-461); the port zero-fills and must not
  // error or panic.
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let world = t.encode(" world").unwrap()[0];
  let mut segment = TranscriptionSegment::new();
  segment
    .set_tokens(vec![hello, world])
    .set_token_log_probs(vec![(hello, -0.1), (world, -0.1)])
    .set_start(0.0)
    .set_end(1.0);
  let weights = vec![1.0f32; 3]; // only ONE row written
  let view = AlignmentView::new(&weights, 1, 3);
  let updated = add_word_timestamps(
    &[segment],
    &view,
    &t,
    "en",
    0,
    crate::constants::PREPEND_PUNCTUATION,
    crate::constants::APPEND_PUNCTUATION,
    0.0,
  )
  .unwrap();
  assert_eq!(updated.len(), 1); // structure survives; timings degrade gracefully
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn add_word_timestamps_errors_on_empty_segments() {
  // Correction to this task's brief: its "Assertion note" claims
  // `needed == 0` flows through `find_alignment`'s `<= 1 word` early
  // return into an empty, word-less result, "exactly Swift's degenerate
  // path." It does not: `find_alignment` calls `dynamic_time_warping`
  // FIRST and unconditionally (see that function's own doc -- "DTW itself
  // still runs first regardless ... so a malformed alignment still errors
  // even on that trivial path"), and `dynamic_time_warping` rejects zero
  // rows before the word-count check is ever reached. An empty `segments`
  // input (zero gathered tokens) surfaces `InvalidAlignmentShape` here,
  // matching Swift's own crash on the equivalent `1...0` `ClosedRange`
  // (see `dynamic_time_warping`'s doc) rather than a silent empty result.
  let t = tiny_tokenizer();
  let view = AlignmentView::new(&[], 0, 3);
  let err = add_word_timestamps(
    &[],
    &view,
    &t,
    "en",
    0,
    crate::constants::PREPEND_PUNCTUATION,
    crate::constants::APPEND_PUNCTUATION,
    0.0,
  )
  .unwrap_err();
  assert!(matches!(
    err,
    SegmentError::InvalidAlignmentShape { rows: 0, .. }
  ));
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn add_word_timestamps_errors_on_zero_columns() {
  // Regression (task-4 review, High): a zero-column view reached
  // `chunks_mut(0)`, which panics even over an empty buffer — the
  // documented InvalidAlignmentShape must surface instead.
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];
  let segments = [plain_segment(vec![hello], 0.0, 1.0)];
  let view = AlignmentView::new(&[], 5, 0);
  let err = add_word_timestamps(&segments, &view, &t, "en", 0, "", "", 0.0).unwrap_err();
  assert!(matches!(
    err,
    SegmentError::InvalidAlignmentShape { cols: 0, .. }
  ));
}
