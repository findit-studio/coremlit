use std::sync::Mutex;

use super::*;
use crate::{
  backend::{ModelDims, mock::MockBackend},
  options::DecodingOptions,
  result::{TranscriptionProgress, TranscriptionSegment, TranscriptionTimings},
  tokenizer::{SpecialTokens, WhisperTokenizer},
};

#[test]
fn state_change_callback_type_is_constructible_and_send() {
  // `(old, new)` per assignment, mirroring Swift's `didSet`
  // (AudioStreamTranscriber.swift:27-31). `Sync` on the callback should
  // make a shared reference to it `Send` (`&F: Send` iff `F: Sync`) —
  // verified here, not just asserted in the type's doc comment.
  let old = StreamState::new();
  let mut newer = StreamState::new();
  newer.set_current_fallbacks(1);

  let cb: StateChangeCallback<'_> = &|prev, next| {
    assert_eq!(prev.current_fallbacks(), 0);
    assert_eq!(next.current_fallbacks(), 1);
  };
  cb(&old, &newer);

  fn assert_send<T: Send>(_: &T) {}
  assert_send(&cb);
}

#[test]
fn stream_options_defaults_match_swift_init() {
  // AudioStreamTranscriber.swift:51-54.
  let options = AudioStreamOptions::new();
  assert_eq!(options.required_segments_for_confirmation(), 2);
  assert_eq!(options.silence_threshold(), 0.3);
  assert_eq!(options.compression_check_window(), 60);
  assert!(options.use_vad());
  assert_eq!(AudioStreamOptions::default(), AudioStreamOptions::new());
  let options = options
    .with_silence_threshold(0.5)
    .with_required_segments_for_confirmation(3);
  assert_eq!(options.silence_threshold(), 0.5);
  assert_eq!(options.required_segments_for_confirmation(), 3);
}

#[test]
fn stream_update_vocabulary() {
  assert_eq!(StreamUpdate::AwaitingVoice.as_str(), "awaiting_voice");
  assert_eq!(StreamUpdate::Transcribed.to_string(), "transcribed");
  assert!(StreamUpdate::AwaitingAudio.is_awaiting_audio());
}

fn progress_with(tokens: Vec<u32>, avg_logprob: Option<f32>) -> TranscriptionProgress {
  let mut progress = TranscriptionProgress::new(TranscriptionTimings::new(), String::new(), tokens);
  if let Some(avg) = avg_logprob {
    progress.set_avg_logprob(avg);
  }
  progress
}

#[test]
fn should_stop_early_matches_swift_decision_table() {
  let options = DecodingOptions::new();
  // 61 identical tokens (> window 60): ratio >> 2.4 -> stop.
  assert_eq!(
    should_stop_early(&progress_with(vec![42; 61], None), &options, 60),
    Some(false)
  );
  // Below the window, repetitive or not: no compression verdict.
  assert_eq!(
    should_stop_early(&progress_with(vec![42; 60], None), &options, 60),
    None
  );
  // Bad average logprob -> stop.
  assert_eq!(
    should_stop_early(&progress_with(vec![1, 2, 3], Some(-2.0)), &options, 60),
    Some(false)
  );
  // Clean -> keep decoding.
  assert_eq!(
    should_stop_early(&progress_with(vec![1, 2, 3], Some(-0.1)), &options, 60),
    None
  );
  // Faithful quirk (AudioStreamTranscriber.swift:217): a DISABLED compression
  // threshold compares against 0.0, so any long token run trips the stop.
  let disabled = DecodingOptions::new().maybe_compression_ratio_threshold(None);
  let varied: Vec<u32> = (0..61).collect();
  assert_eq!(
    should_stop_early(&progress_with(varied, None), &disabled, 60),
    Some(false)
  );
}

#[test]
fn energy_tracker_frames_and_first_frame_zero() {
  // AudioProcessor.swift:906-921: one entry per 1600-sample frame; the
  // first frame has no reference history and records 0 (NaN-clamp parity).
  let mut tracker = EnergyTracker::default();
  let mut buffer = vec![0.001f32; 2 * ENERGY_FRAME_SAMPLES];
  tracker.absorb(&buffer);
  assert_eq!(tracker.relative_energies().len(), 2);
  assert_eq!(tracker.relative_energies()[0], 0.0);
  // Loud frames against the quiet reference read near 1.
  buffer.extend(std::iter::repeat_n(0.5, ENERGY_FRAME_SAMPLES));
  tracker.absorb(&buffer);
  let energies = tracker.relative_energies();
  assert_eq!(energies.len(), 3);
  assert!(
    energies[2] > 0.5,
    "loud-after-quiet is high relative energy, got {}",
    energies[2]
  );
  // Partial frames wait for completion.
  buffer.extend(std::iter::repeat_n(0.5, 10));
  tracker.absorb(&buffer);
  assert_eq!(tracker.relative_energies().len(), 3);
}

#[test]
fn stream_state_defaults_and_pub_crate_mutation() {
  // AudioStreamTranscriber.swift:7-17 (State), minus `isRecording` (mic
  // lifecycle, dropped — see module doc). `set_*` is `pub(crate)`: only
  // this crate's own state machine (Plan 4 T8) mutates a session's state
  // in practice, but the vocabulary earns its own coverage here, ahead of
  // that consumer.
  let mut state = StreamState::new();
  assert_eq!(state, StreamState::default());
  assert_eq!(state.current_fallbacks(), 0);
  assert_eq!(state.last_buffer_size(), 0);
  assert_eq!(state.last_confirmed_segment_end_seconds(), 0.0);
  assert!(state.buffer_energy_slice().is_empty());
  assert!(state.current_text().is_empty());
  assert!(state.confirmed_segments_slice().is_empty());
  assert!(state.unconfirmed_segments_slice().is_empty());
  assert!(state.unconfirmed_text_slice().is_empty());

  state.set_current_fallbacks(2);
  state.set_last_buffer_size(1_600);
  state.set_last_confirmed_segment_end_seconds(3.5);
  state.set_buffer_energy(vec![0.1, 0.2]);
  state.set_current_text("hello");
  let segment = TranscriptionSegment::new().with_text("hi");
  state.confirmed_segments_mut().push(segment.clone());
  state.set_unconfirmed_segments(vec![segment]);
  state.set_unconfirmed_text(vec!["stale".to_string()]);

  assert_eq!(state.current_fallbacks(), 2);
  assert_eq!(state.last_buffer_size(), 1_600);
  assert_eq!(state.last_confirmed_segment_end_seconds(), 3.5);
  assert_eq!(state.buffer_energy_slice().to_vec(), vec![0.1, 0.2]);
  assert_eq!(state.current_text(), "hello");
  assert_eq!(state.confirmed_segments_slice().len(), 1);
  assert_eq!(state.confirmed_segments_slice()[0].text(), "hi");
  assert_eq!(state.unconfirmed_segments_slice().len(), 1);
  assert_eq!(state.unconfirmed_segments_slice()[0].text(), "hi");
  assert_eq!(
    state.unconfirmed_text_slice().to_vec(),
    vec!["stale".to_string()]
  );
}

#[test]
fn contains_subsequence_true_false_and_edges() {
  // Hermetic unit test for the private `contains_subsequence` helper
  // (mod.rs, ports Swift's SE-0357 `Array.contains(_:)` subsequence
  // check). Previously exercised only indirectly, through the
  // tokenizer-gated `AudioStreamTranscriber` tests below, and only ever
  // on its FALSE branch (`confirmed_segments_slice()` starts empty, so
  // the one production call site never observed a hit).
  let a = TranscriptionSegment::new().with_text("a");
  let b = TranscriptionSegment::new().with_text("b");
  let c = TranscriptionSegment::new().with_text("c");
  let haystack = [a.clone(), b.clone(), c.clone()];

  // Needle present as a contiguous run -> true.
  assert!(contains_subsequence(&haystack, &[b.clone(), c.clone()]));
  // Present but non-contiguous (skips `b`) -> false: a strict
  // contiguous-window check, not a general subsequence test.
  assert!(!contains_subsequence(&haystack, &[a.clone(), c.clone()]));
  // Absent entirely -> false.
  let z = TranscriptionSegment::new().with_text("z");
  assert!(!contains_subsequence(&haystack, &[z]));
  // Needle longer than haystack -> false (`slice::windows` yields no
  // windows once its size exceeds the slice length; no panic).
  assert!(!contains_subsequence(std::slice::from_ref(&a), &haystack));
  // Empty needle: Swift's SE-0357 `Array.contains(_:)` returns `true` for
  // an empty subsequence (any collection trivially contains the empty
  // one). This port's `!needle.is_empty()` guard — added only to dodge
  // `slice::windows(0)`'s panic, per the function's own doc — makes it
  // DIVERGE and return `false` instead. Pinned here, not "fixed": the
  // one real call site never passes an empty needle (guarded by
  // `segments.len() > required`), so the divergence is unreachable in
  // production.
  assert!(!contains_subsequence(&haystack, &[]));
}

#[cfg(feature = "serde")]
#[test]
fn stream_options_partial_config_falls_back_to_defaults() {
  // Options-pattern serde pairing (review finding): every field carries a
  // fn-default, so a partial document inherits new()'s values.
  let partial: AudioStreamOptions = serde_json::from_str(r#"{"use_vad":false}"#).unwrap();
  assert!(!partial.use_vad());
  assert_eq!(
    partial.required_segments_for_confirmation(),
    DEFAULT_REQUIRED_SEGMENTS_FOR_CONFIRMATION
  );
  assert_eq!(partial.silence_threshold(), DEFAULT_SILENCE_THRESHOLD);
  assert_eq!(
    partial.compression_check_window(),
    DEFAULT_COMPRESSION_CHECK_WINDOW
  );
  let round: AudioStreamOptions =
    serde_json::from_str(&serde_json::to_string(&AudioStreamOptions::new()).unwrap()).unwrap();
  assert_eq!(round, AudioStreamOptions::new());
}

// ---------------------------------------------------------------------
// AudioStreamTranscriber
// ---------------------------------------------------------------------

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

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn short_push_awaits_audio_with_waiting_text() {
  // AudioStreamTranscriber.swift:131-140.
  let t = tiny_tokenizer();
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let fired = Mutex::new(0usize);
  let callback: &(dyn Fn(&StreamState, &StreamState) + Sync) = &|_old, _new| {
    *fired.lock().unwrap() += 1;
  };
  let mut streamer =
    AudioStreamTranscriber::new(&mock, &t, DecodingOptions::new()).with_state_callback(callback);
  let update = streamer.push_samples(&vec![0.5; 8_000]).unwrap();
  assert!(update.is_awaiting_audio());
  assert_eq!(streamer.state().current_text(), "Waiting for speech...");
  assert!(
    *fired.lock().unwrap() >= 2,
    "buffer_energy + waiting-text assignments fired"
  );
  assert_eq!(mock.counters().encode_calls(), 0);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn silent_audio_is_vad_skipped() {
  // AudioStreamTranscriber.swift:142-157 — 2 s of near-silence: enough new
  // audio, but relative energies stay ~0 -> no transcription.
  let t = tiny_tokenizer();
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let mut streamer = AudioStreamTranscriber::new(&mock, &t, DecodingOptions::new());
  let update = streamer.push_samples(&vec![0.001; 32_000]).unwrap();
  assert!(update.is_awaiting_voice());
  assert_eq!(
    streamer.state().last_buffer_size(),
    0,
    "skipped runs do not consume the buffer"
  );
  assert_eq!(mock.counters().encode_calls(), 0);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn voice_after_silence_transcribes_and_promotes_segments() {
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  // Per-window script ending in a <|1.00|> pair: each 1 s mock window
  // yields one 1 s segment (Plan 3 windowing precedent).
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    hello,
    s.time_token_begin() + 50,
    s.time_token_begin() + 50,
    s.end_token(),
  ]);
  let mut streamer = AudioStreamTranscriber::new(&mock, &t, DecodingOptions::new());

  // 2 s quiet builds a low-energy reference (VAD-skipped, buffer kept)...
  assert!(
    streamer
      .push_samples(&vec![0.001; 32_000])
      .unwrap()
      .is_awaiting_voice()
  );
  // ...then 2 s loud: 4 s buffer, seek clips [(0, 64000)] -> windows at
  // 0/1/2 s -> 3 segments; required=2 -> confirm 1, keep 2 unconfirmed.
  let update = streamer.push_samples(&vec![0.5; 32_000]).unwrap();
  assert!(update.is_transcribed());
  let state = streamer.state();
  assert_eq!(state.confirmed_segments_slice().len(), 1);
  assert_eq!(state.unconfirmed_segments_slice().len(), 2);
  assert!((state.last_confirmed_segment_end_seconds() - 1.0).abs() < 1e-4);
  assert_eq!(state.current_text(), "", "cleared after the run");
  assert_eq!(state.last_buffer_size(), 64_000);

  // Second round: push 2 s louder still (adaptive reference now includes
  // loud frames; 0.9 vs 0.5 stays > 0.3 relative). Clip starts at the
  // 1.0 s watermark -> 4 windows -> confirm 2 more.
  let update = streamer.push_samples(&vec![0.9; 32_000]).unwrap();
  assert!(update.is_transcribed());
  let state = streamer.state();
  assert_eq!(state.confirmed_segments_slice().len(), 3);
  assert!((state.last_confirmed_segment_end_seconds() - 3.0).abs() < 1e-4);
  assert_eq!(state.unconfirmed_segments_slice().len(), 2);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn segments_at_the_confirmation_boundary_stay_unconfirmed() {
  // Regression (task-8 review): the confirmation ELSE branch
  // (`segments.len() <= required`, mod.rs's `push_samples`) had no direct
  // test — only `segments.len() > required` was exercised, above. Two 1 s
  // windows (Plan 3 windowing precedent,
  // `transcribe::tests::windowing_advances_seek_by_last_timestamp_and_decodes_again`)
  // over a 3 s buffer yield exactly two segments; `required_segments_for_confirmation`
  // stays at its default (2), so `2 <= 2` takes the else branch AT THE
  // EXACT BOUNDARY — everything lands in `unconfirmed_segments_slice`,
  // nothing is promoted, and the watermark stays untouched.
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    hello,
    s.time_token_begin() + 50, // <|1.00|>
    s.time_token_begin() + 50,
    s.end_token(),
  ]);
  // use_vad stays at its default (true) elsewhere in this file, but VAD
  // gating is not this test's concern — disable it (the finding's own
  // suggested simplification) so a single loud push suffices.
  let mut stream_options = AudioStreamOptions::new();
  stream_options.clear_use_vad();
  let mut streamer = AudioStreamTranscriber::new(&mock, &t, DecodingOptions::new())
    .with_stream_options(stream_options);

  let update = streamer.push_samples(&vec![0.5; 48_000]).unwrap();
  assert!(update.is_transcribed());
  let state = streamer.state();
  assert_eq!(
    state.confirmed_segments_slice().len(),
    0,
    "2 <= required(2): nothing promoted"
  );
  assert_eq!(state.unconfirmed_segments_slice().len(), 2);
  assert_eq!(
    state.last_confirmed_segment_end_seconds(),
    0.0,
    "watermark untouched"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn fallback_retry_stashes_superseded_text_as_unconfirmed() {
  // Regression (task-8 review, Required): `on_progress_callback`'s stash
  // branch (mod.rs:719-740) — push the superseded `current_text` onto
  // `unconfirmed_text` when a retry's first callback reports fewer chars
  // than the prior attempt's text AT THE SAME fallback count — had no
  // test. `push_samples` unconditionally clears BOTH `current_text` and
  // `unconfirmed_text` right after the run finishes (its own step 4), so
  // the stash is only observable transiently, through the state-change
  // callback firing mid-run — never through `streamer.state()` after
  // `push_samples` returns. This test therefore records every `(old,
  // new)` snapshot via `with_state_callback` and inspects the recorded
  // history, not the final state.
  //
  // Single-window, single-retry fallback ladder, modeled directly on
  // `transcribe::tests::fallback_ladder_retries_with_rising_temperature_then_accepts`
  // and `failed_probe_rederives_language_from_that_attempts_decode`:
  // one-hot-scripted steps (deterministic token identity at any positive
  // temperature, since scaling logits by `1 / temperature` preserves
  // their rank order), `maybe_first_token_logprob_threshold(None)` so
  // only the average-logprob threshold can trigger a fallback, and
  // `with_temperature_fallback_count(1)` for exactly two attempts
  // (t=0.0, t=0.2).
  //
  // `use_vad` stays default `true` everywhere else in this file, but VAD
  // gating is not this test's concern; `clear_use_vad()` (offered by the
  // finding itself) sidesteps it so one loud push suffices.
  //
  // Threshold derivation (empirical, not just hand-computed — see below
  // for why the naive hand computation is wrong): `" Hello"`'s one-hot
  // logprob is ~-1.21 at t=0 (`10.0 - log_sum_exp` over the 51_865-entry
  // mock vocab), but the two closing `<|2.00|>` timestamps are NOT
  // similarly penalized — `TimestampRulesFilter`'s mass-comparison rule
  // boosts timestamp logits, landing their logprob near 0 even at t=0
  // (`transcribe::tests::early_stop_does_not_leak_into_fallback_retries`'s
  // own comment already notes this: "mass-rule-boosted timestamp
  // logprobs of ~-0.07"). A threshold placed between the NAIVE
  // (all-tokens-penalized) estimate and 0 does nothing; empirically,
  // `Some(-0.1)` is where attempt 0 needs a fallback and attempt 1 (t=0.2,
  // where every one-hot logprob collapses to ~0) is accepted — pinned by
  // running with `--nocapture` and reading `mock.counters().resets()`
  // (2: one fallback reset + one window reset — confirmed below) and each
  // segment's `avg_logprob()`/`temperature()`.
  //
  // One more wrinkle the callback count depends on ("prefill callbacks
  // count"): `decode_text`'s progress callback fires on EVERY
  // non-completing loop iteration, including the three forced/discarded
  // prefill ones (feeding SOT/en/transcribe) whose `current_tokens` is
  // still just the unchanged 4-token initial prompt — not only on
  // iterations that push a new token. Combined with `should_stop_early`
  // reading this same `logprob_threshold` from every one of those
  // callbacks (`transcribe_audio_samples` wires it as the progress
  // callback's return value), attempt 0's own PARTIAL average dips below
  // -0.1 right after " Hello" is pushed (`(4 prompt zeros + -1.21) / 5 ≈
  // -0.24`, well before its closing timestamps would dilute it back up) —
  // so attempt 0 is cut short there, before ever reaching its own
  // `<|2.00|>` pair, and the FINAL judgment
  // (`decode::finalize_decoding_result` + `result::needs_fallback`) on
  // that same short decode agrees. Attempt 0's cumulative text at the
  // point it stops is therefore `"...<|0.00|> Hello"`, not a full
  // `"...<|0.00|> Hello<|2.00|><|2.00|>"` — exactly the value pinned
  // below (read via `--nocapture`, not guessed).
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    hello,
    s.time_token_begin() + 100, // <|2.00|>
    s.time_token_begin() + 100,
    s.end_token(),
  ]);

  let options = DecodingOptions::new()
    .maybe_first_token_logprob_threshold(None)
    .maybe_logprob_threshold(Some(-0.1))
    .with_temperature_fallback_count(1);
  let mut stream_options = AudioStreamOptions::new();
  stream_options.clear_use_vad();

  let history: Mutex<Vec<StreamState>> = Mutex::new(Vec::new());
  let callback: &(dyn Fn(&StreamState, &StreamState) + Sync) =
    &|_old, new| history.lock().unwrap().push(new.clone());
  let mut streamer = AudioStreamTranscriber::new(&mock, &t, options)
    .with_stream_options(stream_options)
    .with_state_callback(callback);

  let update = streamer.push_samples(&vec![0.5; 32_000]).unwrap();
  assert!(update.is_transcribed());
  assert_eq!(
    mock.counters().resets(),
    2,
    "one fallback retry + one window reset -- confirms a real 2-attempt ladder ran"
  );

  let history = history.lock().unwrap();
  let stashed = history
    .iter()
    .find(|snapshot| !snapshot.unconfirmed_text_slice().is_empty())
    .unwrap_or_else(|| panic!("no snapshot ever recorded a stashed unconfirmed_text"));
  assert_eq!(
    stashed.unconfirmed_text_slice(),
    &["<|startoftranscript|><|en|><|transcribe|><|0.00|> Hello".to_string()],
    "attempt 0's cumulative text at its (should_stop_early-triggered) cutoff"
  );

  // push_samples' own documented step 4 clears unconfirmed_text
  // unconditionally once the run completes: the stash is real but
  // transient, never surviving past the call that produced it.
  assert!(streamer.state().unconfirmed_text_slice().is_empty());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn panicking_state_callback_resets_state_to_default() {
  // Ledgered at task 8 for this phase gate (progress-plan4.md P4.T8):
  // `StateChangeCallback`'s doc and `push_samples`' doc both warn that a
  // panicking callback poisons the internal `state_mutex`
  // `transcribe_audio_samples` swaps the live `StreamState` into for a
  // run's duration (mod.rs), silently resetting the transcriber's
  // accumulated state to `StreamState::default()` instead of restoring
  // it once the panic unwinds past the `self.state = state_mutex.
  // into_inner()...` restore line. That was "empirically established, not
  // just theorized" per the doc, but had no regression test pinning it —
  // closed here, template = nth-firing panic + catch_unwind + assert
  // reset.
  //
  // `push_samples` fires the state-change callback exactly twice
  // (`set_buffer_energy`, then `set_last_buffer_size`) BEFORE
  // `transcribe_audio_samples` ever swaps `self.state` into the Mutex —
  // both mutate `&mut self.state` directly, outside the swap (`use_vad`
  // is disabled and the push below is loud/long enough in one call to
  // skip both early-return branches, so neither's `set_waiting_text_if_empty`
  // adds an extra firing here). Every firing from the third on happens
  // from INSIDE `on_progress_callback`, reached only through the
  // Mutex-guarded `progress_callback` closure `decode_text` calls once
  // per loop iteration (`decode/mod.rs`) — including forced prefill
  // iterations, which run a real `decode_step` first, so the mock's
  // script is exercised before this fires. The third firing is therefore
  // always `on_progress_callback`'s own first `apply()` call
  // (`set_current_text`), on the very first (prefill) loop iteration of
  // the run's only decode attempt — squarely inside the swap window.
  // Confirmed empirically, not just derived: `--nocapture`'s captured
  // panic backtrace shows frame 3 (`apply`) called from frame 4
  // (`on_progress_callback`) called from frame 5
  // (`transcribe_audio_samples`'s closure) called from `decode_text`
  // (`decode/mod.rs:412`) on this run's first loop iteration — exactly
  // this path, every time, across repeated runs.
  let t = tiny_tokenizer();
  let s = SpecialTokens::whisper_defaults();
  let hello = t.encode(" Hello").unwrap()[0];
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    hello,
    s.time_token_begin() + 100, // <|2.00|>
    s.time_token_begin() + 100,
    s.end_token(),
  ]);
  let mut stream_options = AudioStreamOptions::new();
  stream_options.clear_use_vad();

  let fired = std::sync::atomic::AtomicUsize::new(0);
  let callback: &(dyn Fn(&StreamState, &StreamState) + Sync) = &|_old, _new| {
    if fired.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 == 3 {
      panic!("scripted panic: the state callback must not panic (see StateChangeCallback's doc)");
    }
  };
  let mut streamer = AudioStreamTranscriber::new(&mock, &t, DecodingOptions::new())
    .with_stream_options(stream_options)
    .with_state_callback(callback);

  let samples = vec![0.5; 32_000];
  let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    streamer.push_samples(&samples)
  }));
  assert!(
    outcome.is_err(),
    "the scripted callback panic must propagate out of push_samples, not get swallowed"
  );
  assert_eq!(
    fired.load(std::sync::atomic::Ordering::SeqCst),
    3,
    "the run must not reach a 4th callback firing once the 3rd one panics"
  );

  // The accumulated state (buffer_energy and last_buffer_size, at least,
  // were both set before the panic) is gone: push_samples never reached
  // its restore line, so `self.state` still holds the `mem::take`
  // placeholder from the moment of the swap.
  assert_eq!(*streamer.state(), StreamState::default());
}
