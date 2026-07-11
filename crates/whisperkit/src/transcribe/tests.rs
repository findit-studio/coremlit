use super::*;
use crate::{
  backend::{ModelDims, mock::MockBackend},
  options::DecodingOptions,
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

fn special() -> SpecialTokens {
  SpecialTokens::whisper_defaults()
}

fn ts(index: u32) -> u32 {
  special().time_token_begin() + index
}

/// One clean scripted window: prompt predictions + " Hello" + closing
/// timestamps + EOT. Timestamp pair <|0.00|> .. <|2.00|>.
fn script_clean_window(mock: &mut MockBackend, word: u32) {
  let s = special();
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    word,
    ts(100),
    ts(100),
    s.end_token(),
  ]);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn audio_shorter_than_window_clip_time_yields_no_windows() {
  // Swift-faithful guard: `while seek < seekClipEnd - windowPadding`
  // (TranscribeTask.swift:113-116) never runs for audio shorter than
  // windowClipTime (1 s default) — port with guarded usize subtraction.
  let t = tiny_tokenizer();
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 14_400], &DecodingOptions::new())
    .unwrap();
  assert!(result.segments_slice().is_empty());
  assert_eq!(result.text(), "");
  assert_eq!(mock.counters().encode_calls(), 0);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn single_window_run_produces_segments_and_text() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  // 2 s of audio, 1 s mock windows (window_samples 16_000): window at seek 0
  // runs (0 < 32000 - 16000); its <|2.00|> ending advances seek to 32000,
  // which fails the guard -> exactly one window.
  let result = task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert_eq!(result.text(), "Hello"); // decoded word tokens, trimmed (TranscribeTask.swift:304-305)
  assert_eq!(result.segments_slice().len(), 1);
  let segment = &result.segments_slice()[0];
  assert!((segment.start() - 0.0).abs() < 1e-4);
  assert!((segment.end() - 2.0).abs() < 1e-4);
  assert_eq!(mock.counters().resets(), 1, "state reset after the window");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn windowing_advances_seek_by_last_timestamp_and_decodes_again() {
  let t = tiny_tokenizer();
  // 3 s of audio with 1 s windows: the scripted window ends with a
  // <|1.00|> pair -> seek advances 1 s per window; the state reset between
  // windows rewinds the mock's script cursor (it lives in the state), so
  // one scripted window replays for every window.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  let s = special();
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    hello,
    ts(50), // <|1.00|>
    ts(50),
    s.end_token(),
  ]);
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 48_000], &DecodingOptions::new())
    .unwrap();
  // window 0 covers [0, 16000) -> seek 16000; window 1 -> seek 32000;
  // guard: 32000 < 48000 - 16000 is false -> stop.
  assert_eq!(result.segments_slice().len(), 2);
  assert!(
    (result.segments_slice()[1].start() - 1.0).abs() < 1e-4,
    "time offset applied"
  );
  assert_eq!(mock.counters().encode_calls(), 2);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn fallback_ladder_retries_with_rising_temperature_then_accepts() {
  let t = tiny_tokenizer();
  let s = special();
  // A window whose avg logprob is catastrophically low. without_timestamps
  // keeps TimestampRulesFilter out (near-flat logits would otherwise trip
  // its sum-vs-max forcing) and makes the seeker take the lump branch with
  // a FULL segment_size seek advance — guaranteed loop progress.
  // Per-attempt script (prompt [SOT, en, transcribe, <|notimestamps|>]):
  // 3 forced-prompt predictions + 1 sampled step + EOT = 5 steps.
  // The sampled step's logits are flat-with-EOT-pinned-low: every candidate
  // equally (un)likely at lp = ln(1/51864) ~ -10.86, and EOT can never land
  // in the top-k -> deterministic step COUNTS at any temperature.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  for _ in 0..3 {
    for _ in 0..4 {
      let mut flat = vec![0.0f32; 51865];
      flat[s.end_token() as usize] = -20.0;
      mock.push_step(flat);
    }
    mock.push_token_step(s.end_token());
  }
  // avg logprob over SOT..=EOT: (4 prompt zeros + -10.86 + EOT 0)/6 ~ -1.81
  // < -1.0 default -> LogProbThreshold fallback on EVERY attempt; the
  // ladder exhausts and the final attempt's result stands
  // (TranscribeTask.swift:394-410).
  let options = DecodingOptions::new()
    .with_without_timestamps()
    .maybe_first_token_logprob_threshold(None)
    .maybe_compression_ratio_threshold(None)
    .with_temperature_fallback_count(2); // 3 attempts total
  let task = TranscribeTask::new(&mock, &t);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  let counters = mock.counters();
  // Swift resets on every needsFallback, INCLUDING the exhausted final
  // attempt (TranscribeTask.swift:394-404), plus the per-window reset:
  assert_eq!(
    counters.resets(),
    3 + 1,
    "3 fallback resets + 1 window reset"
  );
  assert_eq!(
    result.segments_slice().len(),
    1,
    "lump branch: one full-window segment"
  );
  assert!(
    (result.segments_slice()[0].temperature() - 0.4).abs() < 1e-3,
    "final attempt at temperature 0.0 + 2 * 0.2"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn segment_discovery_callback_fires_per_window() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let discovered = std::sync::Mutex::new(Vec::<usize>::new());
  let callback: &(dyn Fn(&[crate::result::TranscriptionSegment]) + Sync) =
    &|segments| discovered.lock().unwrap().push(segments.len());
  let task = TranscribeTask::new(&mock, &t).with_segment_callback(callback);
  task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert_eq!(discovered.lock().unwrap().as_slice(), &[1]);
}
