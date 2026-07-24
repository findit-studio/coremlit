use super::*;
use crate::audio::whisper::backend::{BackendError, InferenceBackend, ModelDims};

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
  // Stage/commit split (whisper #41): a step alone STAGES its row and leaves
  // the `hasAlignment` gate shut (view `None`, TextDecoder.swift:764-771);
  // the following `commit_alignment_row` lands the slice at row
  // `position + 1` (TextDecoder.swift:286, :709-717) and opens the gate.
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
  assert!(
    mock.alignment_weights(&state).is_none(),
    "staging alone must not open the gate"
  );
  mock.commit_alignment_row(&mut state);
  let view = mock
    .alignment_weights(&state)
    .expect("commit opens the hasAlignment gate");
  assert_eq!(view.row(1), &[0.5, 0.6, 0.7]);
  assert_eq!(view.row(0), &[0.0, 0.0, 0.0], "row 0 is never written");
}

#[test]
fn full_token_budget_reaches_last_position_without_panicking() {
  // Regression (task-2 review, Critical): the last legal position is
  // `max_token_context - 1`, and its committed alignment row lands at row
  // `position + 1` — stepping there and committing must neither panic in
  // `decode_step`/`commit_alignment_row` nor in `alignment_weights`.
  let dims = ModelDims::new()
    .with_vocab(8)
    .with_max_token_context(4)
    .with_n_audio_ctx(3);
  let mut mock = MockBackend::new().with_dims(dims);
  for _ in 0..dims.max_token_context() - 1 {
    mock.push_step(vec![0.0; dims.vocab()]);
  }
  mock.push_step_with_alignment(vec![0.0; dims.vocab()], vec![0.5; dims.n_audio_ctx()]);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 16]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();
  for position in 0..dims.max_token_context() {
    mock
      .decode_step(0, position, &encoded, &mut state, &mut logits)
      .unwrap();
    mock.commit_alignment_row(&mut state);
    assert_eq!(logits.len(), dims.vocab());
  }
  let view = mock
    .alignment_weights(&state)
    .expect("the last step's committed row opened the gate");
  // Fixed-size view: `rows == max_token_context + 1` always, and the
  // headroom row is reachable without a panic.
  assert_eq!(view.rows(), dims.max_token_context() + 1);
  assert_eq!(view.row(dims.max_token_context()), &[0.5, 0.5, 0.5]);
}

#[test]
fn steps_without_alignment_rows_keep_the_has_alignment_gate_shut() {
  // The `hasAlignment` gate (TextDecoder.swift:568,711,764-771): a window
  // whose committed steps carry no alignment row yields no view at all; the
  // first committed real row flips the gate open. Recast from the old
  // "extend the view" pin — the view is fixed-size now, so presence is the
  // gate, not the row count.
  let dims = ModelDims::new().with_vocab(4).with_n_audio_ctx(2);
  let mut mock = MockBackend::new().with_dims(dims);
  mock.push_step(vec![0.0; 4]);
  mock.push_step_with_alignment(vec![0.0; 4], vec![0.7, 0.7]);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();
  mock
    .decode_step(0, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.commit_alignment_row(&mut state);
  assert!(
    mock.alignment_weights(&state).is_none(),
    "a rowless committed step keeps the gate shut"
  );
  mock
    .decode_step(0, 1, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.commit_alignment_row(&mut state);
  let view = mock
    .alignment_weights(&state)
    .expect("the committed real row opened the gate");
  assert_eq!(view.rows(), dims.max_token_context() + 1);
  assert_eq!(view.row(2), &[0.7, 0.7]);
}

#[test]
fn a_committed_row_survives_reset_as_stale_data() {
  // Reset deliberately leaves the accumulator (Swift's once-allocated
  // tensor, Models.swift:312-322 / TextDecoder.swift:141), so a later,
  // shorter window reads an earlier window's committed row wherever its own
  // steps never reached (whisper #41). Continuous-script mode keeps the
  // cursor across reset so the two windows script different rows.
  let dims = ModelDims::new().with_vocab(4).with_n_audio_ctx(2);
  let mut mock = MockBackend::new().with_dims(dims).with_continuous_script();
  mock.push_step_with_alignment(vec![0.0; 4], vec![9.0, 9.0]);
  mock.push_step_with_alignment(vec![0.0; 4], vec![1.0, 1.0]);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();

  // Window 1 commits a row high up (row 3).
  mock
    .decode_step(0, 2, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.commit_alignment_row(&mut state);
  assert_eq!(mock.alignment_weights(&state).unwrap().row(3), &[9.0, 9.0]);

  // The fallback/next-window reset shuts the gate but keeps the buffer.
  mock.reset_decoder_state(&mut state);
  assert!(
    mock.alignment_weights(&state).is_none(),
    "reset shuts the hasAlignment gate"
  );

  // Window 2 commits only row 1; it never touches row 3.
  mock
    .decode_step(0, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.commit_alignment_row(&mut state);
  let view = mock.alignment_weights(&state).unwrap();
  assert_eq!(view.row(1), &[1.0, 1.0], "window 2's own committed row");
  assert_eq!(
    view.row(3),
    &[9.0, 9.0],
    "window 1's row survived reset as stale data"
  );
}

#[test]
fn a_staged_but_uncommitted_row_is_dropped_by_reset() {
  // A completing step stages its row but the decode loop never commits it
  // (Swift breaks before :709-717): the row must not leak into a later
  // window. Continuous-script mode gives the later window a real committed
  // row so its view opens, exposing the dropped slot as the initial zero.
  let dims = ModelDims::new().with_vocab(4).with_n_audio_ctx(2);
  let mut mock = MockBackend::new().with_dims(dims).with_continuous_script();
  mock.push_step_with_alignment(vec![0.0; 4], vec![9.0, 9.0]);
  mock.push_step_with_alignment(vec![0.0; 4], vec![1.0, 1.0]);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut logits = Vec::new();

  // Window 1 stages a row at position 2 (would be row 3) but never commits.
  mock
    .decode_step(0, 2, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.reset_decoder_state(&mut state);

  // Window 2 commits row 1, opening the gate.
  mock
    .decode_step(0, 0, &encoded, &mut state, &mut logits)
    .unwrap();
  mock.commit_alignment_row(&mut state);
  let view = mock.alignment_weights(&state).unwrap();
  assert_eq!(view.row(1), &[1.0, 1.0]);
  assert_eq!(
    view.row(3),
    &[0.0, 0.0],
    "the staged-but-uncommitted row never landed"
  );
}
