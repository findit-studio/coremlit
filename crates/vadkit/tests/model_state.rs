//! Model-gated behavioral gates for the stateful VAD wrapper (design spec §6
//! "state round-trip + reset contracts; context-stitching pinned"). Run on
//! `ComputeUnits::CpuOnly` for determinism, exactly as the Swift trace oracle
//! and the sibling kits' model-gated suites do.

mod common;

use coremlit::ComputeUnits;
use vadkit::{CHUNK_SAMPLES, CONTEXT_SAMPLES, VadModel, VadModelOptions, VadState};

/// Loads the VAD model on `cpu_only` (deterministic).
fn load() -> VadModel {
  VadModel::load_with(
    common::model_path(),
    VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load vad model")
}

/// Two consecutive 256 ms chunks from a real-speech fixture.
fn two_chunks() -> (Vec<f32>, Vec<f32>) {
  let samples = common::load_wav_16k_mono(&common::fixture_wav_path("02_pyannote_sample"));
  assert!(
    samples.len() >= 2 * CHUNK_SAMPLES,
    "fixture must have at least two full chunks"
  );
  (
    samples[..CHUNK_SAMPLES].to_vec(),
    samples[CHUNK_SAMPLES..2 * CHUNK_SAMPLES].to_vec(),
  )
}

/// `reset()` restores EXACTLY the initial state, so the first chunk after a
/// reset reproduces its very first probability bit-for-bit. Mutation: a
/// `reset()` that fails to clear any state field (hidden, cell, OR context)
/// leaves a residue, and `p_after_reset != p_first` turns this red.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn reset_returns_to_initial_state() {
  let (chunk0, chunk1) = two_chunks();
  let mut model = load();

  assert_eq!(model.state(), &VadState::initial(), "starts at initial");
  let p_first = model.predict_chunk(&chunk0).expect("chunk 0");
  assert_ne!(
    model.state(),
    &VadState::initial(),
    "one chunk must advance the state"
  );
  model.predict_chunk(&chunk1).expect("chunk 1");

  model.reset();
  assert_eq!(
    model.state(),
    &VadState::initial(),
    "reset must restore the initial state exactly"
  );
  let p_after_reset = model.predict_chunk(&chunk0).expect("chunk 0 again");
  assert_eq!(
    p_after_reset, p_first,
    "after reset, chunk 0 must reproduce its first probability bit-for-bit"
  );
}

/// State fully determines the continuation: re-running a chunk from a SAVED
/// input state reproduces the same probability and the same output state, and
/// the internal-state path (`predict_chunk`) matches the explicit-state path
/// (`predict_chunk_with_state`) step for step. Mutation: dropping any carried
/// state field (e.g. not feeding `cell_state`) makes the saved-state re-run
/// diverge from the original continuation.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn state_round_trips_across_chunks() {
  let (chunk0, chunk1) = two_chunks();
  let model = load();

  let (p0, s1) = model
    .predict_chunk_with_state(&chunk0, &VadState::initial())
    .expect("chunk 0");
  let (p1, s2) = model
    .predict_chunk_with_state(&chunk1, &s1)
    .expect("chunk 1 from s1");

  // Re-run chunk 1 from the SAVED state s1 → identical probability and state.
  let (p1_again, s2_again) = model
    .predict_chunk_with_state(&chunk1, &s1)
    .expect("chunk 1 from saved s1");
  assert_eq!(p1, p1_again, "same input state → same probability");
  assert_eq!(s2, s2_again, "same input state → same output state");

  // The internal-state convenience path matches the explicit path exactly.
  let mut streaming = load();
  assert_eq!(streaming.predict_chunk(&chunk0).expect("stream 0"), p0);
  assert_eq!(streaming.predict_chunk(&chunk1).expect("stream 1"), p1);
  assert_eq!(
    streaming.state(),
    &s2,
    "streaming state tracks the explicit one"
  );
}

/// The 64-sample context is load-bearing and its stitch offset is EXACT.
///
/// Three pins over a whole real-speech fixture:
/// 1. **Exact offset (deterministic).** The context carried after chunk 0 is
///    bit-identical to chunk 0's last 64 samples — a one-sample skew in
///    `crate::model::next_context` turns this red with no floating-point fuzz.
///    This IS the "skew the context by one sample → red" mutation gate.
/// 2. **Consumed.** Re-running each chunk from the SAME LSTM state but with the
///    context ZEROED changes the probability on at least one chunk — the graph
///    genuinely reads the context, so the offset in pin 1 matters.
/// 3. **A one-sample skew is observable end-to-end.** Re-running each chunk
///    from the same state with the context ROTATED by one sample changes the
///    probability on at least one chunk — corroborating that the Swift trace
///    gate (`parity_swift.rs`) would catch a stitching skew, not only pin 1.
///
/// The context feeds only the first STFT frame, so its effect on the fp16
/// output is chunk-DEPENDENT: below fp16 resolution on most chunks, observable
/// on a minority (measured on this fixture: zeroed moves ~9 %, a one-sample
/// skew ~10 % of chunks, max |Δ| on the order of 1e-2). Scanning the whole clip
/// — rather than betting on one chunk — is why this is robust; the counts are
/// reported, not asserted exactly.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn misaligned_context_changes_the_probability() {
  let samples = common::load_wav_16k_mono(&common::fixture_wav_path("02_pyannote_sample"));
  let (chunks, _tail) = samples.as_chunks::<CHUNK_SAMPLES>();
  assert!(chunks.len() >= 40, "need enough chunks to scan");
  let model = load();

  // Pin 1: exact stitch offset after the first chunk.
  let (_p0, s1) = model
    .predict_chunk_with_state(&chunks[0], &VadState::initial())
    .expect("chunk 0");
  assert_eq!(
    &s1.context()[..],
    &chunks[0][CHUNK_SAMPLES - CONTEXT_SAMPLES..],
    "carried context must be the previous chunk's last 64 samples, no skew"
  );

  // Pins 2 & 3: stream with the correct context, and at each chunk also probe a
  // zeroed context and a one-sample-skewed context from the SAME running state.
  let mut state = VadState::initial();
  let (mut zeroed_diffs, mut skew_diffs) = (0usize, 0usize);
  let (mut zeroed_max, mut skew_max) = (0.0f64, 0.0f64);
  for chunk in chunks {
    let (correct, next) = model
      .predict_chunk_with_state(chunk, &state)
      .expect("correct context");

    let zeroed = VadState::from_parts(*state.hidden(), *state.cell(), [0.0f32; CONTEXT_SAMPLES]);
    let (p_zeroed, _) = model
      .predict_chunk_with_state(chunk, &zeroed)
      .expect("zeroed");

    let mut skewed_ctx = *state.context();
    skewed_ctx.rotate_left(1);
    let skewed = VadState::from_parts(*state.hidden(), *state.cell(), skewed_ctx);
    let (p_skewed, _) = model
      .predict_chunk_with_state(chunk, &skewed)
      .expect("skewed");

    if p_zeroed != correct {
      zeroed_diffs += 1;
    }
    if p_skewed != correct {
      skew_diffs += 1;
    }
    zeroed_max = zeroed_max.max((f64::from(p_zeroed) - f64::from(correct)).abs());
    skew_max = skew_max.max((f64::from(p_skewed) - f64::from(correct)).abs());
    state = next;
  }
  println!(
    "[misaligned] {} chunks | zeroed-context differs on {zeroed_diffs} (max |Δ| {zeroed_max:.3e}) \
     | 1-sample skew differs on {skew_diffs} (max |Δ| {skew_max:.3e})",
    chunks.len()
  );
  assert!(
    zeroed_diffs > 0,
    "a zeroed context must change the probability on at least one chunk — the graph consumes it"
  );
  assert!(
    skew_diffs > 0,
    "a one-sample context skew must be observable on at least one chunk (corroborating the trace gate)"
  );
}
