//! Full-pipeline integration: mel → encode → scripted decode steps through
//! `coremlit::audio::whisper::backend::coreml::CoreMlBackend` against the real tiny model.
//!
//! Ground truth pinned by `tests/model_io.rs` (Task 1 introspection); the
//! decode-step expectations mirror Swift's first predictions on
//! `jfk.wav`/tiny (`TextDecoder.swift:541-855`).

mod common;

use coremlit::{
  ComputeUnits, Model,
  audio::whisper::{
    backend::{InferenceBackend, coreml::CoreMlBackend},
    model::{ModelState, manager::ModelManager},
    options::{ChunkingStrategy, ComputeOptions, DecodingOptions, Options},
    transcribe::WhisperKit,
  },
};

/// CpuOnly is legitimate HERE — and only because of what this file asserts.
///
/// THE RULE: **a gate validating a shipping default must run on the shipping
/// default.** The crate ships mel = CPU+GPU and encoder/decoder = CPU+ANE
/// (`options::DEFAULT_*_COMPUTE_UNITS`, matching Swift's
/// `ModelComputeOptions`), so anything asserting NUMERICS — above all
/// `tests/parity_jfk.rs`/`parity_es.rs`, whose goldens are an ANE-captured
/// external Swift oracle — must run on those units, and those tests assert
/// so explicitly. **The golden tests own the shipping compute path and must
/// never be pinned away from it**, however tempting that looks as a fix for
/// a "flaky" golden. The sibling crate `alignkit` learned this the hard way:
/// it shipped `ComputeUnits::All` while every test pinned `CpuOnly`, and the
/// ANE turned out to produce a corrupted output matrix that the green suite
/// never saw.
///
/// The tests in this file assert SHAPES, DTYPES, and CONTROL FLOW — window
/// counts, feature dimensions, segment structure, callback wiring — none of
/// which the compute unit can change. Pinning them to CpuOnly buys
/// determinism and skips the ANE compilation stall, and costs no coverage
/// the goldens do not already own. Do not extend that reasoning to a test
/// that compares numbers against a reference.
fn load_backend() -> CoreMlBackend {
  let tiny = common::tiny_dir();
  let mel = Model::load(tiny.join("MelSpectrogram.mlmodelc"), ComputeUnits::CpuOnly).unwrap();
  let encoder = Model::load(tiny.join("AudioEncoder.mlmodelc"), ComputeUnits::CpuOnly).unwrap();
  let decoder = Model::load(tiny.join("TextDecoder.mlmodelc"), ComputeUnits::CpuOnly).unwrap();
  CoreMlBackend::new(mel, encoder, decoder).unwrap()
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn dims_derive_from_model_descriptions() {
  let backend = load_backend();
  let dims = backend.dims();
  assert_eq!(dims.vocab(), 51865);
  assert_eq!(dims.n_mels(), 80);
  assert_eq!(dims.embed_dim(), 384);
  assert_eq!(dims.kv_dim(), 1536);
  assert_eq!(dims.max_token_context(), 224);
  assert_eq!(dims.n_audio_ctx(), 1500);
  assert_eq!(dims.window_samples(), 480_000);
  assert!(dims.is_multilingual());
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_first_decode_steps_produce_language_then_task_tokens() {
  // The strongest possible pre-parity smoke test: on real audio, greedy
  // decoding from SOT must predict <|en|> (50259) then <|transcribe|>
  // (50359) — the same first predictions Swift makes on jfk.wav/tiny.
  let backend = load_backend();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  let mut window = audio.clone();
  window.resize(480_000, 0.0); // pad_or_trim
  let features = backend.extract_features(&window).unwrap();
  let encoded = backend.encode(&features).unwrap();
  let mut state = backend.new_decoder_state().unwrap();
  let mut logits = Vec::new();

  backend
    .decode_step(50258, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  assert_eq!(logits.len(), 51865);
  assert!(logits.iter().all(|v| v.is_finite()));
  let argmax = |l: &[f32]| {
    l.iter()
      .enumerate()
      .max_by(|a, b| a.1.total_cmp(b.1))
      .unwrap()
      .0
  };
  assert_eq!(argmax(&logits), 50259, "step 0 from SOT predicts <|en|>");

  backend
    .decode_step(50259, 1, &encoded, &mut state, &mut logits)
    .unwrap();
  assert_eq!(argmax(&logits), 50359, "step 1 predicts <|transcribe|>");
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn reset_restores_step_zero_logits_exactly() {
  // KV state isolation: step 0 logits after a reset must match a fresh
  // state bit-for-bit (CPU-only prediction is deterministic).
  let backend = load_backend();
  let mut window = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  window.resize(480_000, 0.0);
  let encoded = backend
    .encode(&backend.extract_features(&window).unwrap())
    .unwrap();
  let mut state = backend.new_decoder_state().unwrap();
  let (mut first, mut second) = (Vec::new(), Vec::new());
  backend
    .decode_step(50258, 0, &encoded, &mut state, &mut first)
    .unwrap();
  backend
    .decode_step(50259, 1, &encoded, &mut state, &mut second)
    .unwrap();
  backend.reset_decoder_state(&mut state);
  let mut replay = Vec::new();
  backend
    .decode_step(50258, 0, &encoded, &mut state, &mut replay)
    .unwrap();
  assert_eq!(first, replay, "reset produced a clean-slate step");
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn alignment_weights_accumulate_when_supported() {
  let backend = load_backend();
  if !backend.supports_word_timestamps() {
    eprintln!("skipping: model lacks alignment_heads_weights (recorded in Task 1)");
    return;
  }
  let mut window = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  window.resize(480_000, 0.0);
  let encoded = backend
    .encode(&backend.extract_features(&window).unwrap())
    .unwrap();
  let mut state = backend.new_decoder_state().unwrap();
  let mut logits = Vec::new();
  backend
    .decode_step(50258, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  let view = backend
    .alignment_weights(&state)
    .expect("supported => view");
  assert_eq!(view.cols(), 1500);
  assert!(view.rows() >= 2, "row position+1 written");
  assert!(view.row(1).iter().any(|&v| v != 0.0), "weights landed");
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn wrong_audio_length_is_structured_error() {
  let backend = load_backend();
  let err = backend.extract_features(&[0.0; 100]).unwrap_err();
  assert!(matches!(
    err,
    coremlit::audio::whisper::backend::BackendError::AudioLength {
      got: 100,
      expected: 480_000
    }
  ));
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn last_kv_slot_decode_step_succeeds() {
  // Regression (task-9 review, Critical): the trait legalizes every
  // position in 0..max_token_context, but the mask flips prepare
  // position + 1 — at the last slot there is no next slot to prepare and
  // the flips must be skipped, while the KV append and the alignment row
  // (headroom row max_ctx) still land. Pre-fix this returned a structured
  // IndexOutOfBounds AFTER mutating the KV cache.
  let backend = load_backend();
  let dims = backend.dims();
  let last = dims.max_token_context() - 1;
  let features = backend
    .extract_features(&vec![0.0; dims.window_samples()])
    .unwrap();
  let encoded = backend.encode(&features).unwrap();
  let mut state = backend.new_decoder_state().unwrap();
  let mut logits = Vec::new();
  backend
    .decode_step(50258, last, &encoded, &mut state, &mut logits)
    .expect("the last KV slot is a legal position");
  assert_eq!(logits.len(), dims.vocab());
  if let Some(view) = backend.alignment_weights(&state) {
    assert_eq!(view.rows(), dims.max_token_context() + 1);
    assert_eq!(view.cols(), dims.n_audio_ctx());
  }
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn manager_loads_tiny_idempotently_and_backend_builds() {
  let mut manager = ModelManager::new(
    common::tiny_dir(),
    // CpuOnly (no ANE compilation stalls) — legitimate here for the same
    // reason as in `load_backend` above: this asserts LOAD STATE MACHINERY,
    // not numerics. The goldens own the shipping compute path; see
    // `load_backend`'s doc for the rule.
    ComputeOptions::new()
      .with_mel(ComputeUnits::CpuOnly)
      .with_encoder(ComputeUnits::CpuOnly)
      .with_decoder(ComputeUnits::CpuOnly),
  );
  manager.ensure_loaded().unwrap();
  assert_eq!(manager.state(), ModelState::Loaded);
  manager.ensure_loaded().unwrap(); // idempotent — no reload, still Loaded
  assert_eq!(manager.state(), ModelState::Loaded);
  let backend = CoreMlBackend::from_loaded(manager.into_loaded().unwrap()).unwrap();
  assert_eq!(backend.dims().vocab(), 51865);
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn manager_prewarm_sequences_states() {
  let mut manager = ModelManager::new(
    common::tiny_dir(),
    ComputeOptions::new()
      .with_mel(ComputeUnits::CpuOnly)
      .with_encoder(ComputeUnits::CpuOnly)
      .with_decoder(ComputeUnits::CpuOnly),
  );
  manager.prewarm().unwrap();
  assert_eq!(manager.state(), ModelState::Prewarmed);
  manager.ensure_loaded().unwrap();
  assert_eq!(manager.state(), ModelState::Loaded);
  manager.unload();
  assert_eq!(manager.state(), ModelState::Unloaded);
}

// NOTE: the brief's literal snippet called `Options::new()` with no
// arguments, then chained `.with_model_folder(...)`/
// `.with_tokenizer_folder(...)`. The shipped constructor is two-argument
// (`Options::new(model_folder, tokenizer_folder)` — Plan 2's own doc: "No
// Default/zero-arg new(): there is no honest default model or tokenizer
// folder"), so both folders are passed directly here instead.
fn tiny_options() -> Options {
  Options::new(common::tiny_dir(), common::tokenizer_dir())
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_end_to_end_produces_english_transcript() {
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  let result = kit.transcribe(&audio, &DecodingOptions::new()).unwrap();
  assert_eq!(result.language(), "en");
  let lowered = result.text().to_lowercase();
  assert!(
    lowered.contains("fellow americans"),
    "got: {}",
    result.text()
  );
  assert!(!result.segments_slice().is_empty());
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn detect_language_on_jfk_is_english() {
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  let detection = kit.detect_language(&audio).unwrap();
  assert_eq!(detection.language(), "en");
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn prewarm_over_loaded_models_is_rejected() {
  // Regression (phase-gate round 3, ModelManager.swift:131-139): prewarm
  // over a resident triple would relabel it Prewarmed, double-load on the
  // next ensure_loaded, and on failure strand the models behind unload's
  // state guard. It must reject instead, leaving state and models intact;
  // an already-Prewarmed manager skips silently.
  use coremlit::audio::whisper::{
    error::ModelError,
    model::{ModelState, manager::ModelManager},
    options::ComputeOptions,
  };
  let mut manager = ModelManager::new(common::tiny_dir(), ComputeOptions::new());
  manager.ensure_loaded().unwrap();
  assert_eq!(manager.state(), ModelState::Loaded);
  let err = manager.prewarm().unwrap_err();
  assert!(matches!(err, ModelError::InvalidState { .. }));
  assert_eq!(manager.state(), ModelState::Loaded, "state untouched");
  manager.unload();
  assert_eq!(
    manager.state(),
    ModelState::Unloaded,
    "models still unloadable"
  );

  let mut manager = ModelManager::new(common::tiny_dir(), ComputeOptions::new());
  manager.prewarm().unwrap();
  assert_eq!(manager.state(), ModelState::Prewarmed);
  manager.prewarm().unwrap(); // silent skip, no re-prewarm transitions
  assert_eq!(manager.state(), ModelState::Prewarmed);
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_word_timestamps_are_monotonic_and_cover_the_transcript() {
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  let options = DecodingOptions::new().with_word_timestamps();
  let result = kit.transcribe(&audio, &options).unwrap();
  let words: Vec<_> = result
    .segments_slice()
    .iter()
    .flat_map(|s| s.words_slice().iter().cloned())
    .collect();
  assert!(words.len() >= 10, "jfk has ~22 words; got {}", words.len());
  for pair in words.windows(2) {
    assert!(
      pair[0].start() <= pair[1].start() + 1e-3,
      "monotonic word starts"
    );
  }
  for word in &words {
    assert!(word.end() >= word.start());
    assert!((0.0..=1.0).contains(&word.probability()));
    assert!(word.end() <= 11.5, "inside the 11 s clip");
  }
  let joined: String = words.iter().map(|w| w.word()).collect();
  let normalized = coremlit::audio::whisper::text::normalized(&joined);
  assert!(
    normalized.contains("ask not what your country"),
    "got: {normalized}"
  );
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn detect_language_on_es_and_ja_clips() {
  // Ports the language expectations of Swift's detectLanguage tests: the
  // clips' languages are the goldens.
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let es = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/es_test_clip.wav"));
  assert_eq!(kit.detect_language(&es).unwrap().language(), "es");
  let ja = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/ja_test_clip.wav"));
  assert_eq!(kit.detect_language(&ja).unwrap().language(), "ja");
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn silence_transcribes_to_the_blank_audio_marker_when_drop_is_cleared() {
  // Pins coremlit issue #9's silence edge case: 5 s of digital silence,
  // VAD off / prefill on, decoded to exactly `[BLANK_AUDIO]`, one segment,
  // on both Rust and Swift in the issue's own validation run --
  // upstream-compatible model behavior, not a Rust-only quirk.
  // `use_prefill_prompt`/`chunking_strategy` are set explicitly rather
  // than left to their (already-matching) defaults, per the issue's own
  // P1 policy recommendation to pin decode options rather than rely on
  // them silently.
  //
  // This contract now lives on the `drop_blank_audio == false` path
  // (coremlit issue #14): dropping the marker became the DEFAULT, a
  // deliberate product divergence from Swift, and clearing the option is
  // the exact Swift-parity escape hatch. Everything asserted below is
  // byte-for-byte what this test asserted before the option existed --
  // that is the point: opting out restores the old behavior exactly. The
  // default path is pinned by `silence_is_dropped_by_default` below.
  //
  // The byte-exact strings below are pins captured empirically against
  // this exact tiny-model/option combination (`cargo test -p coremlit
  // --test whisper_pipeline -- --ignored --nocapture
  // silence_transcribes_to_the_blank_audio_marker_when_drop_is_cleared`),
  // not derived from a spec: `result.text()` is exactly the marker, with
  // no surrounding whitespace. The sole segment's raw `text()` retains its
  // special/timestamp tokens because this test leaves `skip_special_tokens`
  // at its default (`false`, unset here) -- segment text is the
  // undecorated decode, `TranscriptionResult::text()` is the cleaned
  // aggregate view assembled from it, so the two having different
  // shapes is expected, not a bug in either. The representation
  // invariant that falls out of this -- result-level equality against
  // the marker holds, segment-level equality never does under this
  // configuration (`BLANK_AUDIO_MARKER`'s own doc) -- is asserted
  // explicitly below, not left for a reader to infer from the strings. It
  // is also exactly why the drop filter matches a segment's CLEAN decode
  // rather than this raw `text()`.
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let audio = vec![0.0f32; 5 * coremlit::audio::whisper::constants::SAMPLE_RATE as usize];
  let options = DecodingOptions::new()
    .with_use_prefill_prompt()
    .with_chunking_strategy(ChunkingStrategy::Disabled)
    .maybe_drop_blank_audio(false);
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(
    result.text(),
    coremlit::audio::whisper::constants::BLANK_AUDIO_MARKER,
    "got: {:?}",
    result.text()
  );
  let segments = result.segments_slice();
  assert_eq!(segments.len(), 1, "got: {segments:?}");
  assert_eq!(
    segments[0].text(),
    "<|startoftranscript|><|en|><|transcribe|><|0.00|> [BLANK_AUDIO]<|10.00|><|endoftext|>",
    "got: {:?}",
    segments[0].text()
  );
  assert_ne!(
    segments[0].text(),
    coremlit::audio::whisper::constants::BLANK_AUDIO_MARKER,
    "segment text must retain its special/timestamp tokens here, unlike result.text()"
  );
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn silence_is_dropped_by_default() {
  // The DEFAULT path over the very same 5 s of digital silence its
  // drop=false twin above decodes (coremlit issue #14): `drop_blank_audio`
  // defaults `true`, so the sole `[BLANK_AUDIO]` segment is filtered out
  // after decoding and the caller gets a genuinely empty result -- zero
  // segments, empty text -- instead of a one-segment marker transcript.
  //
  // Same audio, same model, same prefill/chunking as that twin; the ONLY
  // difference is the option left at its default. That pairing is the
  // mutation evidence that this outcome is the filter's doing, not the
  // decode's.
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let audio = vec![0.0f32; 5 * coremlit::audio::whisper::constants::SAMPLE_RATE as usize];
  let options = DecodingOptions::new()
    .with_use_prefill_prompt()
    .with_chunking_strategy(ChunkingStrategy::Disabled);
  assert!(
    options.drop_blank_audio(),
    "dropping must be the default, with no caller opt-in"
  );
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(result.text(), "", "got: {:?}", result.text());
  assert!(
    result.segments_slice().is_empty(),
    "got: {:?}",
    result.segments_slice()
  );
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn half_second_clip_yields_no_segments() {
  // Pins coremlit issue #9's other validated edge case: a clip too short
  // to form a segment decodes to empty text with zero segments on both
  // runtimes -- distinct from the silence case above (real speech
  // content, just too little of it: the first 0.5 s of `jfk.wav`).
  let kit = WhisperKit::new(&tiny_options()).unwrap();
  let mut audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  audio.truncate(coremlit::audio::whisper::constants::SAMPLE_RATE as usize / 2); // 0.5 s
  let options = DecodingOptions::new()
    .with_use_prefill_prompt()
    .with_chunking_strategy(ChunkingStrategy::Disabled);
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(result.text(), "", "got: {:?}", result.text());
  assert!(result.segments_slice().is_empty());
}
