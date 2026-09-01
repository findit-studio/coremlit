//! Ground-truth introspection + provenance pins for the language-identification
//! CoreML artifact.
//!
//! Every shape/dtype claim here comes from loading the real `.mlmodelc` through
//! `coremlit::Model::load` + `.description()`; every SHA comes from the
//! downloaded bytes; and the accepted frame range is established by ASKING THE
//! RUNTIME rather than by trusting the library's constants.
//!
//! The model gates are `#[ignore]`d and run only with the artifact staged
//! (`LID_TEST_MODELS`, default `Models/lid`). The hermetic self-checks below
//! run with no model at all.
//!
//! # Pinned contract
//!
//! ```text
//! mel_features       f32  [1, frames, 60]   frames in 10..=3001 (RangeDims)
//! log_probabilities  f32  [1, 107]
//! ```

mod common;

use coremlit::{
  ComputeUnits, DataType, Model, MultiArray,
  audio::lid::{
    Error, Identifier, IdentifierOptions, MAX_FRAMES, MAX_SAMPLES, MIN_FRAMES, MIN_SAMPLES,
    NUM_LANGUAGES, frame_count,
  },
};

/// Mel width the graph declares — the front end's `n_mels`, restated here so
/// the pin is independent of the library constant it validates.
const N_MELS: usize = 60;

/// A `[1, frames, 60]` f32 input tensor of zeros, for probing the runtime's own
/// accepted frame range.
fn mel_tensor(frames: usize) -> MultiArray {
  MultiArray::from_slice(&[1, frames, N_MELS], &vec![0.0f32; frames * N_MELS])
    .expect("build mel tensor")
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The provenance strings are pinned so a re-download from a different revision
/// cannot quietly reuse this file's SHA table.
#[test]
fn provenance_is_pinned() {
  assert_eq!(
    common::HF_REPO,
    "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML"
  );
  assert_eq!(common::HF_REVISION.len(), 40);
  assert_eq!(
    common::SOURCE_MODEL,
    "speechbrain/lang-id-voxlingua107-ecapa"
  );
  assert_eq!(common::SOURCE_REVISION.len(), 40);
  assert_eq!(common::ARTIFACT_SHA256.len(), 4);
  for (path, sha) in common::ARTIFACT_SHA256 {
    assert_eq!(sha.len(), 64, "{path} needs a full SHA-256");
    assert!(
      sha
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
    );
  }
}

/// The library's published bounds are the ones this file pins against the
/// runtime below, in both unit systems.
#[test]
fn published_bounds_agree_with_this_files_pins() {
  assert_eq!((MIN_FRAMES, MAX_FRAMES), (10, 3_001));
  assert_eq!((MIN_SAMPLES, MAX_SAMPLES), (1_440, 480_159));
  assert_eq!(frame_count(MIN_SAMPLES), MIN_FRAMES);
  assert_eq!(frame_count(MAX_SAMPLES), MAX_FRAMES);
  assert_eq!(NUM_LANGUAGES, 107);
}

/// A clip outside the accepted range is rejected WITHOUT a model — the guard
/// runs before construction of any tensor, so the check is reachable even here.
#[test]
fn out_of_range_clips_are_rejected_before_any_model_is_needed() {
  let detail = coremlit::audio::lid::FrameCountOutOfRange::for_samples(MAX_SAMPLES + 1);
  assert!(!detail.is_too_short());
  assert_eq!(detail.frames(), MAX_FRAMES + 1);
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// The declared I/O contract, read off the real model.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn model_declares_the_pinned_io_contract() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");
  let description = model.description();

  let input = description
    .input("mel_features")
    .expect("`mel_features` input");
  assert_eq!(input.data_type(), Some(DataType::F32));
  let shape = input.shape();
  assert_eq!(shape.len(), 3, "rank 3, got {shape:?}");
  assert_eq!(shape[0], 1, "unit batch, got {shape:?}");
  assert_eq!(shape[2], N_MELS, "60 mel columns, got {shape:?}");
  // A flexible (`RangeDims`) axis reports its DEFAULT size through this
  // snapshot, not its bounds — CoreML does not surface the range here, which is
  // exactly why the boundary probe below asks the runtime instead.
  assert!(
    (MIN_FRAMES..=MAX_FRAMES).contains(&shape[1]),
    "the default time axis must sit inside the accepted range, got {shape:?}"
  );
  // And pinned to the exact value, not merely to the range, because two things
  // downstream now READ it rather than tolerate it: the frame count
  // `runtime_accepts_exactly_the_pinned_frame_range` singles out below, and
  // `common::default_shape_refusal`, whose whole diagnosis is that one host
  // refuses the graph's OWN default shape while answering every other length.
  // A re-exported artifact that moved this would turn both into tests of an
  // arbitrary number, silently.
  assert_eq!(
    shape[1],
    common::GRAPH_DEFAULT_SHAPE_FRAMES,
    "the artifact's declared DefaultShapes time axis moved, got {shape:?}"
  );

  let output = description
    .output("log_probabilities")
    .expect("`log_probabilities` output");
  assert_eq!(output.shape(), [1, NUM_LANGUAGES]);
  assert_eq!(output.data_type(), Some(DataType::F32));

  assert_eq!(description.inputs().len(), 1, "exactly one input");
  assert_eq!(description.outputs().len(), 1, "exactly one output");
}

/// The staged bundle is byte-for-byte the pinned artifact — exact file set,
/// exact SHA-256 per file.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn artifact_matches_the_pinned_sha_manifest() {
  common::assert_exact_sha_manifest(&common::model_path(), common::ARTIFACT_SHA256);
}

/// The accepted frame range comes from the RUNTIME, not from this crate's
/// constants: [`MIN_FRAMES`] and [`MAX_FRAMES`] are accepted, and one frame
/// outside either end is refused. This is the gate that would catch a
/// re-exported artifact whose `RangeDims` moved — at which point the library's
/// pre-check would start rejecting audio the model would have taken (or worse,
/// waving through audio it would not).
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn runtime_accepts_exactly_the_pinned_frame_range() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");

  // `GRAPH_DEFAULT_SHAPE_FRAMES` rather than a bare 301: this row is here
  // BECAUSE it is the graph's declared default shape, and the assertion above
  // is what keeps that true.
  for frames in [
    MIN_FRAMES,
    MIN_FRAMES + 1,
    common::GRAPH_DEFAULT_SHAPE_FRAMES,
    MAX_FRAMES - 1,
    MAX_FRAMES,
  ] {
    let input = mel_tensor(frames);
    let outputs = model
      .predict_with(&[("mel_features", &input)])
      .unwrap_or_else(|e| panic!("{frames} frames must be accepted: {e}"));
    let scores = outputs.get("log_probabilities").expect("output present");
    assert_eq!(scores.shape(), [1, NUM_LANGUAGES]);
  }

  for frames in [MIN_FRAMES - 1, MAX_FRAMES + 1] {
    let input = mel_tensor(frames);
    assert!(
      model.predict_with(&[("mel_features", &input)]).is_err(),
      "{frames} frames must be refused by the runtime"
    );
  }
}

/// `Identifier::load` accepts the real artifact under every compute placement,
/// so the contract check is not accidentally tied to one of them.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn identifier_loads_under_every_compute_placement() {
  for compute in [
    ComputeUnits::All,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
  ] {
    Identifier::load(
      common::model_path(),
      IdentifierOptions::new().with_compute(compute),
    )
    .unwrap_or_else(|e| panic!("load under {compute:?}: {e}"));
  }
}

/// The library's own guard fires before the model does, at both boundaries: the
/// two accepted sample counts go through, and the two rejected ones come back
/// as [`Error::FrameCountOutOfRange`] rather than as a CoreML prediction error.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn identifier_guards_the_range_before_calling_the_model() {
  let identifier = Identifier::load(
    common::model_path(),
    IdentifierOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load identifier");

  for accepted in [MIN_SAMPLES, MAX_SAMPLES] {
    let scores = identifier
      .log_probabilities(&vec![0.0f32; accepted])
      .unwrap_or_else(|e| panic!("{accepted} samples must be accepted: {e}"));
    assert_eq!(scores.len(), NUM_LANGUAGES);
  }

  for rejected in [MIN_SAMPLES - 1, MAX_SAMPLES + 1] {
    let error = identifier
      .log_probabilities(&vec![0.0f32; rejected])
      .expect_err("must be rejected");
    assert!(
      matches!(error, Error::FrameCountOutOfRange(_)),
      "{rejected} samples must be a typed range error, got {error:?}"
    );
  }
}
