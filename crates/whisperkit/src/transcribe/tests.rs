use std::path::PathBuf;

use super::*;
use crate::{
  backend::{ModelDims, mock::MockBackend},
  error::SegmentError,
  options::{ChunkingStrategy, DecodingOptions},
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

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn failed_probe_rederives_language_from_that_attempts_decode() {
  // Regression (task-11 review, Important): TranscribeTask.swift:351-352
  // assigns the probe's outcome unconditionally (`try?` yields nil on
  // failure), so a later attempt's FAILED probe must clear the earlier
  // value and re-derive from that attempt's own decode result — never
  // stay sticky at the stale probe language.
  let t = tiny_tokenizer();
  let es = t.token_to_id("<|es|>").unwrap();
  let hello = t.encode(" Hello").unwrap()[0];
  let s = special();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  // Replayed from position 0 after every reset: serves attempt 0's probe
  // (one step) AND, after the probe's internal rewind, the decode itself.
  mock.push_token_steps(&[
    es,
    s.transcribe_token(),
    ts(0),
    hello,
    ts(100),
    ts(100),
    s.end_token(),
  ]);
  // Call ordinals (measured): probe0 = call 1 (succeeds, "es");
  // attempt-0 decode = call 2 only — decode_text's own first-token early
  // stop fires (forced "es" logprob ~-1.21 < -0.5 at t = 0) -> fallback;
  // probe1 = call 3 -> scripted to fail. Attempt 1's decode then rebuilds
  // the prompt with the default language and early-stops the same way,
  // so the ladder exhausts with attempt 1's result.
  mock.fail_on_call(3);

  let options = DecodingOptions::new()
    .with_detect_language()
    .with_temperature_fallback_count(1)
    .maybe_first_token_logprob_threshold(Some(-0.5))
    .maybe_logprob_threshold(None);
  let task = TranscribeTask::new(&mock, &t);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(
    result.language(),
    "en",
    "stale probe language must not survive a failed re-probe"
  );
  // The only enabled fallback trigger is the first-token comparison; it
  // fires on attempt 0 (stored counter = that fallback's 0-based attempt
  // index, TranscribeTask.swift:397) and attempt 1 accepts. The reset
  // count pins the full structure — probe0 + fallback + probe1 + window
  // — so a broken first-token comparison (no fallback, no re-probe)
  // fails here even though 0.0 is also the counter's default.
  assert_eq!(result.timings().total_decoding_fallbacks(), 0.0);
  assert_eq!(mock.counters().resets(), 4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn transcribe_all_preserves_order_across_scoped_threads() {
  let t = tiny_tokenizer();
  // The mock's script cursor lives in each worker's OWN MockDecoderState
  // (Task 2), so concurrent workers replay the same script independently;
  // only the counters are shared. One scripted window serves both audios.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t);
  let a = vec![0.1f32; 32_000];
  let b = vec![0.1f32; 32_000];
  let results = kit.transcribe_all(&[&a, &b], &DecodingOptions::new());
  assert_eq!(results.len(), 2);
  for result in results {
    assert_eq!(result.unwrap().text(), "Hello");
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn vad_chunked_transcribe_reanchors_and_merges() {
  // Regression (task-12 review, Important): the VAD branch itself —
  // chunk_all splitting, per-chunk clip clearing, apply_seek_offsets
  // re-anchoring, and merge_transcription_results — had no direct test.
  // Silence frames at 32_000..35_200 and 64_000..67_200 make
  // split_on_middle_of_longest_silence cut at 33_600 and 65_600, giving
  // three chunks (offsets 0 / 33_600 / 65_600), each decoding one
  // scripted "Hello" window.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(48_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t);
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let options = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(result.text(), "Hello Hello Hello", "per-chunk texts joined");
  let segments = result.segments_slice();
  assert_eq!(segments.len(), 3);
  let ids: Vec<usize> = segments.iter().map(|s| s.id()).collect();
  assert_eq!(
    ids,
    vec![0, 1, 2],
    "merge re-ids result_index + segment_index"
  );
  let starts: Vec<f32> = segments.iter().map(|s| s.start()).collect();
  assert!((starts[0] - 0.0).abs() < 1e-3);
  // 33_600 / 16_000 = 2.1 s and 65_600 / 16_000 = 4.1 s: the .1 offsets
  // are the chunk boundaries and CANNOT come from the un-chunked path
  // (whose window seeks land on whole timestamps, 2.0 / 4.0).
  assert!(
    (starts[1] - 2.1).abs() < 1e-3,
    "chunk 2 re-anchored, got {}",
    starts[1]
  );
  assert!(
    (starts[2] - 4.1).abs() < 1e-3,
    "chunk 3 re-anchored, got {}",
    starts[2]
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn zero_max_window_seek_terminates() {
  // Regression (phase-gate round 3): `Some(0)` reached the seek clamp as
  // `min(seek, previous_seek + 0)`, re-decoding the same window forever.
  // The cap now floors at one sample of progress: audio 3 samples past
  // the window guard yields exactly 3 one-sample windows, then stops.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new().with_max_window_seek(0);
  let result = task.run(&vec![0.1; 16_003], &options).unwrap();
  assert_eq!(
    result.segments_slice().len(),
    3,
    "one window per floored sample"
  );
  // Saturation half: a cap near usize::MAX must not overflow the sum.
  let options = DecodingOptions::new().with_max_window_seek(usize::MAX);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(
    result.segments_slice().len(),
    1,
    "uncapped-in-practice advance"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn out_of_range_clip_end_terminates_at_physical_audio() {
  // Regression (phase-gate round 4, High): a clip end beyond the audio
  // keeps `clip_guard` unsatisfied after `seek` reaches the physical
  // end; `segment_size` hits zero and — `without_timestamps`, where the
  // seek advance IS `segment_size` — the loop re-decoded padded silence
  // forever. The zero-size guard now breaks to the next clip.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  let s = special();
  // without_timestamps prompt: [sot, en, transcribe, no_ts]; forced
  // predictions consume script[0..2], then sampled hello + EOT.
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.no_timestamps_token(),
    hello,
    s.end_token(),
  ]);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new()
    .with_without_timestamps()
    .with_clip_timestamps(vec![0.0, 4.0]); // 64_000 samples of clip over 32_000 of audio
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  // Two real 1-second windows fit the physical audio; the third
  // iteration's zero segment breaks instead of decoding padding forever.
  assert_eq!(result.segments_slice().len(), 2);
  assert_eq!(
    mock.counters().encode_calls(),
    2,
    "no empty-window inference"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn early_stop_does_not_leak_into_fallback_retries() {
  // Regression (phase-gate round 5): the early-stop latch was allocated
  // once per window, so a callback-stopped attempt whose partial result
  // triggered an ordinary fallback handed the still-true flag to the
  // retry, truncating it after one step — Swift initializes a fresh
  // latch per decodeText (TextDecoder.swift:570).
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let calls = std::sync::Mutex::new(0usize);
  let callback: &(dyn Fn(&crate::result::TranscriptionProgress) -> Option<bool> + Sync) =
    &|_progress| {
      let mut seen = calls.lock().unwrap();
      *seen += 1;
      // Call 6 = attempt 0's third sampled step (after hello + both
      // timestamps landed): request the stop. Every other call continues.
      Some(*seen != 6)
    };
  let task = TranscribeTask::new(&mock, &t).with_progress_callback(callback);
  // At t = 0 the window averages ~-0.19 (hello's one-hot logprob -1.21
  // diluted by prefill/EOT zeros and the mass-rule-boosted timestamp
  // logprobs of ~-0.07); at t = 0.2 every draw scores ~0. A -0.1
  // threshold therefore fails attempt 0's stopped partial and accepts
  // the full retry. A leaked stale flag instead breaks the retry after
  // one step. (The first-token threshold stays disabled: it would
  // complete attempt 0 at its very first iteration, before any callback
  // can fire.)
  let options = DecodingOptions::new()
    .with_temperature_fallback_count(1)
    .maybe_first_token_logprob_threshold(None)
    .maybe_logprob_threshold(Some(-0.1));
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(result.text(), "Hello", "the retry ran to completion");
  assert_eq!(
    mock.counters().decode_steps(),
    13,
    "6 stopped steps + 7 full-retry steps"
  );
  assert_eq!(result.timings().total_decoding_fallbacks(), 0.0);
}

fn one_hot(token: u32) -> Vec<f32> {
  let mut logits = vec![0.0f32; 51865];
  logits[token as usize] = 10.0;
  logits
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn window_loop_attaches_word_timings_when_enabled() {
  let t = tiny_tokenizer();
  let s = special();
  let hello = t.encode(" Hello").unwrap()[0];
  // Mock with alignment rows: dims n_audio_ctx 100 -> 100-col rows; each
  // step's row peaks 10 frames (0.2 s) later than the last.
  let mut mock = MockBackend::new().with_dims(
    ModelDims::new()
      .with_window_samples(16_000)
      .with_n_audio_ctx(100),
  );
  let script = [
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    hello,
    ts(100),
    ts(100),
    s.end_token(),
  ];
  for (step, token) in script.iter().enumerate() {
    let mut row = vec![0.0f32; 100];
    row[(step + 1) * 10] = 1.0;
    mock.push_step_with_alignment(one_hot(*token), row);
  }
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new().with_word_timestamps();
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(result.segments_slice().len(), 1);
  let words = result.segments_slice()[0].words_slice();
  assert!(!words.is_empty(), "word timings attached");
  let joined: String = words.iter().map(|w| w.word()).collect();
  assert_eq!(crate::text::normalized(&joined), "hello");
  for word in words {
    assert!(word.end() >= word.start());
    assert!(
      (0.0..=2.5).contains(&word.end()),
      "timings inside the window"
    );
  }
  // Segment boundary follows the last word (update heuristics ran).
  assert!((result.segments_slice()[0].end() - words.last().unwrap().end()).abs() < 1e-4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_timestamps_off_leaves_segments_wordless() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert!(result.segments_slice()[0].words_slice().is_empty());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn silence_skipped_window_with_word_timestamps_surfaces_a_segment_error() {
  // Self-review finding (task-5 review): the brief's insert-code comment
  // describes a silence-skipped window's `None` `current_segments`
  // becoming `Some(vec![])` under word timestamps, mirroring Swift's
  // `currentSegments ?? []`. That is not what the already-shipped
  // `add_word_timestamps` (task 4) does: called with an empty `segments`
  // slice, its prefix-take alignment always ends up zero rows (sized off
  // `segments`' own token count, independent of the real alignment
  // matrix's shape), which `find_alignment`'s DTW guard unconditionally
  // rejects as `SegmentError::InvalidAlignmentShape`. `run` therefore
  // surfaces a `TranscribeError::Segment`, not a silent empty segment
  // list — the faithful analogue of Swift's own unconditional crash on
  // the same input (`SegmentSeeker.swift:208`/`:211`'s unguarded `1...0`
  // range) as a typed, recoverable error instead of a hard process abort.
  //
  // Reached here via an explicit negative `no_speech_threshold`, since
  // `no_speech_prob` is permanently `0.0` (`decode/mod.rs`'s own
  // faithfully-ported upstream TODO) — no positive threshold, including
  // the default, can trigger the silence skip through a real decode.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new()
    .with_word_timestamps()
    .with_no_speech_threshold(-0.1)
    .maybe_logprob_threshold(None);
  let err = task.run(&vec![0.1; 32_000], &options).unwrap_err();
  assert!(
    matches!(
      err,
      TranscribeError::Segment(SegmentError::InvalidAlignmentShape { rows: 0, .. })
    ),
    "got: {err:?}"
  );
}
