//! Door gates.
//!
//! Until a CI shard stages the artifact, nothing in this repository loads a
//! model through [`Embedder`]. So everything the load path decides is factored
//! into free functions over shapes, dtypes and slices — [`check_feature`],
//! [`validate_window_input`], [`check_finite_embedding`], [`describe`] — and
//! those are exercised here in full, with no model present. What is NOT covered
//! by any of this is that a real `.mlmodelc` declares the contract these
//! functions accept; that is `tests/identity/model_io.rs`'s job, and it is
//! `#[ignore]`d until the artifact is staged.

use super::*;

// ── Geometry ───────────────────────────────────────────────────────────────

/// The window is 6 s at the contract rate, and the mel geometry it implies is
/// the graph's declared input shape.
#[test]
fn geometry_is_the_converted_contract() {
  assert_eq!(SAMPLE_RATE_HZ, 16_000);
  assert_eq!(WINDOW_SAMPLES, 6 * SAMPLE_RATE_HZ as usize);
  assert_eq!(N_MELS, 72);
  assert_eq!(N_FRAMES, 401);
  assert_eq!(EMBEDDING_DIM, 192);
}

/// This door's embedding dimension is NOT the diarization embedder's, and the
/// two are not interchangeable — a caller who mixes them gets a length error
/// rather than a wrong answer, and this pins the two numbers apart so a future
/// edit cannot quietly unify them.
#[cfg(feature = "speaker")]
#[test]
fn identity_and_diarization_embedding_dims_are_different_numbers() {
  assert_ne!(EMBEDDING_DIM, crate::audio::speaker::embed::EMBEDDING_DIM);
  assert_eq!(
    EMBEDDING_DIM,
    crate::audio::speaker::calibrate::Scoring::IdentityCosine.row_len(),
    "the identity score source must take exactly this door's raw row"
  );
}

/// The feature names the graph declares. Owned by the conversion recipe, which
/// this repository also owns, so they are pinned on both sides.
#[test]
fn feature_names_are_the_converted_ones() {
  assert_eq!(names::MEL, "mel");
  assert_eq!(names::EMBEDDING, "embedding");
}

/// The placement default is the MEASURED `CpuAndGpu`, and deliberately not the
/// `All` the crate's other doors take — see [`DEFAULT_COMPUTE`] for the
/// four-arm table. A change here without a new sweep is the thing this asserts
/// against.
#[test]
fn default_compute_is_the_measured_cpu_and_gpu() {
  assert_eq!(DEFAULT_COMPUTE, ComputeUnits::CpuAndGpu);
  assert_ne!(DEFAULT_COMPUTE, ComputeUnits::All);
}

// ── EmbedderOptions (rust-options-pattern) ─────────────────────────────────

#[test]
fn options_default_equals_new() {
  assert_eq!(EmbedderOptions::default(), EmbedderOptions::new());
  assert_eq!(EmbedderOptions::new().compute(), DEFAULT_COMPUTE);
}

#[test]
fn options_with_and_set_compute() {
  let opts = EmbedderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
  let mut opts = EmbedderOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip_and_pinned_spelling() {
  let opts = EmbedderOptions::new().with_compute(ComputeUnits::All);
  let json = serde_json::to_string(&opts).unwrap();
  assert_eq!(json, "{\"compute\":\"all\"}");
  let back: EmbedderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

#[cfg(feature = "serde")]
#[test]
fn options_missing_compute_defaults_to_the_measured_placement() {
  let opts: EmbedderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(opts.compute(), DEFAULT_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn options_unknown_compute_spelling_is_rejected() {
  assert!(serde_json::from_str::<EmbedderOptions>("{\"compute\":\"gpu\"}").is_err());
}

// ── The load-time contract check, as a pure function ───────────────────────

/// The exact contract the conversion emits is accepted.
#[test]
fn check_feature_accepts_the_converted_contract() {
  assert!(
    check_feature(
      names::MEL,
      &[1, N_MELS, N_FRAMES],
      Some((&[1, N_MELS, N_FRAMES], Some(DataType::F32)))
    )
    .is_ok()
  );
  assert!(
    check_feature(
      names::EMBEDDING,
      &[1, EMBEDDING_DIM],
      Some((&[1, EMBEDDING_DIM], Some(DataType::F32)))
    )
    .is_ok()
  );
}

/// A model with no feature of that name is refused, and the error says
/// `missing` rather than rendering a shape nobody declared.
#[test]
fn check_feature_refuses_a_missing_feature() {
  let err = check_feature(names::MEL, &[1, N_MELS, N_FRAMES], None).unwrap_err();
  let Error::ContractMismatch(m) = err else {
    panic!("expected ContractMismatch, got {err:?}")
  };
  assert_eq!(m.feature(), "mel");
  assert_eq!(m.expected(), "[1, 72, 401] float32");
  assert_eq!(m.actual(), "missing");
}

/// A transposed mel — the single most likely conversion mistake, since
/// `[1, 401, 72]` has exactly as many elements — is refused at load rather than
/// producing an embedding of noise.
#[test]
fn check_feature_refuses_a_transposed_shape_of_the_same_size() {
  let err = check_feature(
    names::MEL,
    &[1, N_MELS, N_FRAMES],
    Some((&[1, N_FRAMES, N_MELS], Some(DataType::F32))),
  )
  .unwrap_err();
  let Error::ContractMismatch(m) = err else {
    panic!("expected ContractMismatch, got {err:?}")
  };
  assert_eq!(m.actual(), "[1, 401, 72] float32");
}

/// The dtype is checked as hard as the shape. A graph whose boundary is fp16
/// has the right shape and the right name and is still not this contract — the
/// door takes f32 at the boundary and the graph casts internally.
#[test]
fn check_feature_refuses_a_right_shaped_wrong_dtype_feature() {
  for dtype in [Some(DataType::F16), Some(DataType::I32), None] {
    let err = check_feature(
      names::EMBEDDING,
      &[1, EMBEDDING_DIM],
      Some((&[1, EMBEDDING_DIM], dtype)),
    )
    .unwrap_err();
    assert!(
      matches!(err, Error::ContractMismatch(_)),
      "dtype {dtype:?} must be refused, got {err:?}"
    );
  }
}

/// A flexible-shape input declares no constrained dimensions at all, and is
/// refused: the conversion pins a fixed shape on purpose, because a `RangeDim`
/// input takes the graph off the ANE.
#[test]
fn check_feature_refuses_an_unconstrained_shape() {
  let err = check_feature(
    names::MEL,
    &[1, N_MELS, N_FRAMES],
    Some((&[], Some(DataType::F32))),
  )
  .unwrap_err();
  let Error::ContractMismatch(m) = err else {
    panic!("expected ContractMismatch, got {err:?}")
  };
  assert_eq!(m.actual(), "[] float32");
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(
    describe(&[1, N_MELS, N_FRAMES], Some(DataType::F32)),
    "[1, 72, 401] float32"
  );
  assert_eq!(describe(&[1, EMBEDDING_DIM], None), "[1, 192] none");
}

// ── Input and output guards ────────────────────────────────────────────────

/// Exactly one window, or a typed refusal carrying both counts. Not padded and
/// not truncated at any length.
#[test]
fn validate_window_input_requires_an_exact_window() {
  assert!(validate_window_input(&vec![0.0f32; WINDOW_SAMPLES]).is_ok());
  for len in [0usize, 1, WINDOW_SAMPLES - 1, WINDOW_SAMPLES + 1] {
    let err = validate_window_input(&vec![0.0f32; len]).unwrap_err();
    assert!(
      matches!(err, Error::WindowLength(w) if w.got() == len && w.expected() == WINDOW_SAMPLES),
      "len {len}: got {err:?}"
    );
  }
}

/// A NaN or ±∞ sample is refused with its index, before it can reach the mel —
/// where the per-bin mean would spread it across every frame of that bin.
#[test]
fn validate_window_input_reports_the_first_non_finite_sample() {
  let mut samples = vec![0.0f32; WINDOW_SAMPLES];
  samples[4_100] = f32::INFINITY;
  samples[41] = f32::NAN;
  assert!(matches!(
    validate_window_input(&samples),
    Err(Error::NonFiniteInput(41))
  ));
}

/// The length check runs BEFORE the finite scan, so a short clip full of NaNs
/// is reported as the length error it is — the caller's actual mistake.
#[test]
fn validate_window_input_checks_length_before_finiteness() {
  let short = vec![f32::NAN; 16];
  assert!(matches!(
    validate_window_input(&short),
    Err(Error::WindowLength(_))
  ));
}

/// A non-finite model output is caught before it reaches the caller's L2, where
/// one NaN component would make all 192 of them NaN.
#[test]
fn finite_embedding_check_reports_the_index() {
  let mut row = [0.0f32; EMBEDDING_DIM];
  assert!(check_finite_embedding(&row).is_ok());
  row[7] = f32::NEG_INFINITY;
  assert!(matches!(
    check_finite_embedding(&row),
    Err(Error::NonFiniteOutput(7))
  ));
}
