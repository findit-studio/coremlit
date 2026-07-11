//! Full-pipeline integration: mel → encode → scripted decode steps through
//! `whisperkit::backend::coreml::CoreMlBackend` against the real tiny model.
//!
//! Ground truth pinned by `tests/model_io.rs` (Task 1 introspection); the
//! decode-step expectations mirror Swift's first predictions on
//! `jfk.wav`/tiny (`TextDecoder.swift:541-855`).

mod common;

use coremlit::{ComputeUnits, Model};
use whisperkit::backend::{InferenceBackend, coreml::CoreMlBackend};

fn load_backend() -> CoreMlBackend {
  let tiny = common::tiny_dir();
  // CpuOnly in tests: deterministic and no ANE compilation latency.
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
    whisperkit::backend::BackendError::AudioLength {
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
