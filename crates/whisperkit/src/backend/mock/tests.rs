use super::*;
use crate::backend::{BackendError, InferenceBackend, ModelDims};

fn tiny_mock(tokens: &[u32]) -> MockBackend {
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_vocab(100).with_n_audio_ctx(4));
  mock.push_token_steps(tokens);
  mock
}

#[test]
fn scripted_steps_replay_in_order_and_record_positions() {
  let mock = tiny_mock(&[7, 9]);
  let features = mock.extract_features(&[0.0; 16]).unwrap();
  let encoded = mock.encode(&features).unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();

  mock
    .decode_step(50258, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  assert_eq!(logits.len(), 100);
  assert_eq!(logits[7], 10.0);
  mock
    .decode_step(7, 1, &encoded, &mut state, &mut logits)
    .unwrap();
  assert_eq!(logits[9], 10.0);
  assert_eq!(state.consumed_slice(), &[(50258, 0), (7, 1)]);

  // Script exhausted -> structured error, not silence.
  let err = mock
    .decode_step(9, 2, &encoded, &mut state, &mut logits)
    .unwrap_err();
  assert!(matches!(err, BackendError::ScriptExhausted { step: 2 }));
}

#[test]
fn reset_replays_script_and_counts() {
  let mock = tiny_mock(&[7]);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();
  mock
    .decode_step(1, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.reset_decoder_state(&mut state);
  mock
    .decode_step(1, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  assert_eq!(logits[7], 10.0);
  let counters = mock.counters();
  assert_eq!(counters.decode_steps(), 2);
  assert_eq!(counters.resets(), 1);
  assert_eq!(counters.encode_calls(), 1);
}

#[test]
fn alignment_rows_accumulate_at_position_plus_one() {
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_vocab(10).with_n_audio_ctx(3));
  mock.push_step_with_alignment(vec![0.0; 10], vec![0.5, 0.6, 0.7]);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();
  mock
    .decode_step(1, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  let view = mock
    .alignment_weights(&state)
    .expect("mock always has alignment");
  // Ports updateAlignmentWeights: slice lands at row tokenIndex + 1
  // (TextDecoder.swift:286).
  assert_eq!(view.row(1), &[0.5, 0.6, 0.7]);
  assert_eq!(view.row(0), &[0.0, 0.0, 0.0]);
}
