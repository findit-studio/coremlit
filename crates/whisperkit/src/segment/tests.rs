use super::*;
use crate::{
  options::DecodingOptions,
  result::DecodingResult,
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
