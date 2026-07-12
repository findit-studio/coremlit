use super::*;
use crate::result::{TranscriptionResult, TranscriptionSegment, TranscriptionTimings, WordTiming};

// The brief's literal snippet called `TranscriptionResult::new()` with no
// arguments and passed `.into()`'d string literals to `impl Into<String>`
// parameters; neither compiles against the shipped API. Same two fixes
// already established at
// `result::tests::merge_transcription_results_concatenates_and_reids`
// (result/tests.rs:10-19): the constructor takes all four fields with no
// zero-arg/`Default` form, and `"literal".into()` against a generic `impl
// Into<String>` parameter is ambiguous (E0283 -- `&str` implements
// `Into<T>` for several `T`, verified with a standalone rustc repro);
// passing the `&str` literal directly resolves unambiguously through the
// callee's own generic bound instead.
fn result_with_segment(words: Vec<WordTiming>) -> TranscriptionResult {
  let mut segment = TranscriptionSegment::new();
  segment
    .set_start(0.0)
    .set_end(2.0)
    .set_text(" Hello world")
    .set_words(words);
  let mut result = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  result.set_segments(vec![segment]).set_text(" Hello world");
  result
}

#[test]
fn format_time_matches_swift_markers_and_truncation() {
  // ResultWriter.swift:14-25 -- msec TRUNCATES (Int cast), not rounds.
  assert_eq!(format_time(0.0, true, ','), "00:00:00,000");
  assert_eq!(format_time(2.5, true, ','), "00:00:02,500");
  assert_eq!(format_time(2.5, false, '.'), "00:02.500");
  assert_eq!(format_time(3661.25, false, '.'), "01:01:01.250"); // hrs>0 forces hours
  assert_eq!(format_time(1.9995, true, ','), "00:00:01,999");
}

#[test]
fn srt_uses_segment_blocks_without_words() {
  // ResultWriter.swift:89-93 + 27-31.
  let srt = srt_content(&result_with_segment(vec![]));
  assert_eq!(srt, "1\n00:00:00,000 --> 00:00:02,000\n Hello world\n\n");
}

#[test]
fn srt_emits_one_block_per_word_and_increments_indices() {
  // ResultWriter.swift:83-88.
  let words = vec![
    WordTiming::new(" Hello", vec![1], 0.0, 0.5, 0.9),
    WordTiming::new(" world", vec![2], 0.5, 1.0, 0.9),
  ];
  let srt = srt_content(&result_with_segment(words));
  assert_eq!(
    srt,
    "1\n00:00:00,000 --> 00:00:00,500\n Hello\n\n\
     2\n00:00:00,500 --> 00:00:01,000\n world\n\n"
  );
}

#[test]
fn vtt_has_header_and_dot_markers_without_indices() {
  // ResultWriter.swift:111-133 + 33-37.
  let vtt = vtt_content(&result_with_segment(vec![]));
  assert_eq!(vtt, "WEBVTT\n\n00:00.000 --> 00:02.000\n Hello world\n\n");
}

#[test]
fn writers_emit_files_with_the_right_extension() {
  let dir = tempfile::tempdir().unwrap();
  let result = result_with_segment(vec![]);
  let path = SrtWriter::new(dir.path()).write(&result, "out").unwrap();
  assert!(path.ends_with("out.srt"));
  assert_eq!(
    std::fs::read_to_string(&path).unwrap(),
    srt_content(&result)
  );
  let path = VttWriter::new(dir.path()).write(&result, "out").unwrap();
  assert!(path.ends_with("out.vtt"));
  // Unwritable directory -> structured error carrying the path.
  let err = SrtWriter::new("/nonexistent/dir")
    .write(&result, "out")
    .unwrap_err();
  assert!(matches!(err, WriteError::Write { .. }));
}

#[cfg(feature = "serde")]
#[test]
fn json_round_trips_through_serde() {
  let dir = tempfile::tempdir().unwrap();
  let result = result_with_segment(vec![WordTiming::new(" Hello", vec![1], 0.0, 0.5, 0.9)]);
  let path = JsonWriter::new(dir.path()).write(&result, "out").unwrap();
  assert!(path.ends_with("out.json"));
  let parsed: TranscriptionResult =
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
  assert_eq!(parsed, result);
}

#[test]
fn writers_replace_existing_files_without_leaving_staging_artifacts() {
  // Phase-gate follow-up: writes stage into a sibling .tmp then rename
  // (Swift's atomically: true). Overwrite works and no staging file
  // survives either write.
  let dir = tempfile::tempdir().unwrap();
  let writer = SrtWriter::new(dir.path());
  let first = result_with_segment(vec![]);
  let path = writer.write(&first, "again").unwrap();
  let before = std::fs::read_to_string(&path).unwrap();
  writer.write(&first, "again").unwrap();
  assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
  let leftovers: Vec<_> = std::fs::read_dir(dir.path())
    .unwrap()
    .filter_map(Result::ok)
    .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
    .collect();
  assert!(leftovers.is_empty(), "staging file leaked: {leftovers:?}");
}
