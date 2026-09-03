use super::*;

#[test]
fn options_default_equals_new() {
  assert_eq!(AudioEncoderOptions::default(), AudioEncoderOptions::new());
  assert_eq!(AudioEncoderOptions::new().compute(), DEFAULT_AUDIO_COMPUTE);
  assert_eq!(DEFAULT_AUDIO_COMPUTE, ComputeUnits::All);
}

#[test]
fn options_with_and_set_compute() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);

  let mut opts = AudioEncoderOptions::new();
  opts.set_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndGpu);
}

#[test]
fn first_non_finite_finds_offenders() {
  assert_eq!(first_non_finite(&[0.0, 1.0, 2.0]), None);
  assert_eq!(first_non_finite(&[0.0, f32::NAN, 2.0]), Some(1));
  assert_eq!(first_non_finite(&[f32::INFINITY]), Some(0));
  assert_eq!(first_non_finite(&[1.0, 2.0, f32::NEG_INFINITY]), Some(2));
  // Subnormals and signed zeros are finite.
  assert_eq!(
    first_non_finite(&[0.0, -0.0, f32::MIN_POSITIVE / 2.0]),
    None
  );
}

/// `embed_window` accepts `1..=TARGET_SAMPLES` and rejects an over-length clip
/// with [`Error::AudioTooLong`] (naming `embed_windows`) instead of silently
/// head-truncating it. Gated at the `check_window_len` seam so it needs no model.
///
/// Mutation tripwire: relaxing the bound (`>` → `>=`, or `TARGET_SAMPLES` →
/// `TARGET_SAMPLES + 1`) makes the over-length case pass, and dropping the guard
/// re-admits the silent-truncation defect.
#[test]
fn check_window_len_rejects_over_length_only() {
  // The exact window and anything shorter are accepted.
  assert!(check_window_len(TARGET_SAMPLES).is_ok());
  assert!(check_window_len(TARGET_SAMPLES - 1).is_ok());
  assert!(check_window_len(1).is_ok());
  // One sample past the window is rejected, and the error carries len + limit and
  // points the caller at the long-audio path.
  let err = check_window_len(TARGET_SAMPLES + 1).unwrap_err();
  let msg = err.to_string();
  assert!(
    matches!(err, Error::AudioTooLong(ref e) if e.len() == TARGET_SAMPLES + 1 && e.max() == TARGET_SAMPLES),
    "expected AudioTooLong with len {} and max {TARGET_SAMPLES}, got {err:?}",
    TARGET_SAMPLES + 1
  );
  assert!(
    msg.contains("embed_windows"),
    "AudioTooLong should name the long-audio path: {msg}"
  );
}

/// The codex [high] at-cap geometry: with `hop = 5`, a 500 000-sample (~2 MiB)
/// clip plans EXACTLY [`crate::embeddings::clap::window::DEFAULT_MAX_WINDOWS`]
/// (100 000) spans and is ADMITTED by the O(1) cap — `planned == max`, not
/// `> max`. `embed_windows` then reserves one ~2 KiB [`WindowEmbedding`] per span
/// (~207 MiB), which the fix does FALLIBLY (`try_reserve_exact` →
/// [`Error::Windowing`]`(`[`WinditError::AllocFailed`]`)`) instead of the prior
/// infallible `Vec::with_capacity`, so an allocator refusal on a small at-cap
/// clip is a typed error rather than a process abort.
///
/// This is the achievable seam assertion (mirroring `check_window_len_*`, which
/// tests `embed_window`'s guard without a model): it pins that `spans()` admits
/// the exact-cap plan the caller now reserves for. Asserting the actual typed
/// `AllocFailed` would require injecting an allocator failure — impractical and
/// unavailable here (there is no fault-injection allocator, and ~207 MiB is
/// ordinarily allocatable, so no real OOM fires) — and `embed_windows` itself
/// needs a loaded model for the per-span loop that follows the reservation. The
/// `try_reserve_exact` call is the structural guarantee; this test pins the
/// geometry that reaches it.
///
/// Mutation tripwire: reverting the reservation to `with_capacity` restores the
/// process-abort path for exactly this admitted-at-cap input.
#[test]
fn at_cap_plan_is_admitted_and_reserved_fallibly() {
  let plan = WindowPlan::new().with_hop_samples(5);
  // Default cap is on; the exact-cap geometry sits AT it (not over).
  assert_eq!(plan.max_windows(), 100_000);
  let spans = plan
    .spans(500_000)
    .expect("at-cap plan must be admitted, not refused");
  assert_eq!(
    spans.len(),
    100_000,
    "hop=5 over 500_000 samples plans exactly 100_000 spans"
  );
  // EXACTLY at the cap — the boundary the caller's fallible reservation covers.
  assert_eq!(spans.len() as u32, plan.max_windows());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_gpu"), "serialized as as_str: {json}");
  let back: AudioEncoderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself — its feature names, its
// element type, its geometry and its state clause — against descriptions built
// with the same fixture machinery, so a mis-stated contract is caught here and
// a mis-implemented checker is caught there.

use crate::{AxisRange, FeatureInfo, ModelDescription, model::RawShapeConstraint};

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  multi_array(name, shape, dtype, false, 2, vec![shape.to_vec()], shape)
}

/// One multi-array feature, spelled out: the constraint's raw type code, its
/// enumerated shapes, and the axes its per-axis ranges pin.
fn multi_array(
  name: &str,
  shape: &[usize],
  dtype: DataType,
  optional: bool,
  raw_type: isize,
  enumerated: Vec<Vec<usize>>,
  pinned: &[usize],
) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    optional,
    Some(RawShapeConstraint::new(
      raw_type,
      enumerated,
      pinned.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// The staged HTSAT bundle's description, as the CoreML probe reads it back:
/// `input_features [1, 1, 1001, 64]` f32 in, `audio_embeds [1, 512]` f32 out,
/// no state — identical across the fp16 and int8 tiers.
fn htsat_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_FEATURES,
      &[1, 1, T_FRAMES, N_MELS],
      DataType::F32,
    )],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into the CLAP
/// error vocabulary — exactly what `AudioEncoder::from_file_with` does after
/// `Model::load`.
fn check(description: &ModelDescription) -> Result<()> {
  crate::model::contract::check_load_contract(description, &audio_contract())
    .map_err(contract_violation)
}

/// The contract states exactly the geometry the conversion emits.
#[test]
fn the_contract_accepts_the_converted_geometry() {
  assert!(check(&htsat_description()).is_ok());
}

/// **The flexible-shape refusal.** [`crate::FeatureInfo::shape`] reports the
/// DEFAULT shape of a `RangeDims` input, so a flexible graph converted at
/// `[1, 1, 1001, 64]` declares this contract's exact numbers AND reports
/// `(d, 1)` on every axis. Only the whole-feature verdict separates the two.
#[test]
fn the_contract_refuses_a_flexible_input_declaring_its_exact_numbers() {
  let description = ModelDescription::from_parts(
    vec![multi_array(
      names::INPUT_FEATURES,
      &[1, 1, T_FRAMES, N_MELS],
      DataType::F32,
      false,
      3,
      Vec::new(),
      &[1, 1, T_FRAMES, N_MELS],
    )],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::INPUT_FEATURES),
    "{err}"
  );
}

/// An mlprogram converted at `compute_precision=FLOAT16` without an explicit
/// `dtype=np.float32` reports Float16 I/O, so this clause catches a
/// conversion-recipe regression at load rather than restating a constant.
#[test]
fn the_contract_refuses_a_right_shaped_fp16_graph() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_FEATURES,
      &[1, 1, T_FRAMES, N_MELS],
      DataType::F16,
    )],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::INPUT_FEATURES),
    "{err}"
  );
}

/// A transposed spectrogram of the same total size — the mutation no
/// element-count check can see, and the one this door's row-major mel write
/// depends on.
#[test]
fn the_contract_refuses_a_transposed_spectrogram() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_FEATURES,
      &[1, 1, N_MELS, T_FRAMES],
      DataType::F32,
    )],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(matches!(
    check(&description),
    Err(Error::ContractMismatch(_))
  ));
}

/// The projection width is CLAP's 512, and a differently sized head is a
/// different model.
#[test]
fn the_contract_refuses_a_different_projection_width() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_FEATURES,
      &[1, 1, T_FRAMES, N_MELS],
      DataType::F32,
    )],
    vec![fixed(names::AUDIO_EMBEDS, &[1, 768], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::AUDIO_EMBEDS),
    "{err}"
  );
}

/// **A graph carrying `input_features` plus another REQUIRED input** clears
/// every per-feature clause and then fails on every prediction, because
/// [`AudioEncoder::embed_window`] supplies `input_features` and nothing else.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(
        names::INPUT_FEATURES,
        &[1, 1, T_FRAMES, N_MELS],
        DataType::F32,
      ),
      fixed("is_longer", &[1, 1], DataType::I32),
    ],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableInput(name)) if name == "is_longer"),
    "{:?}",
    check(&description)
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail.
#[test]
fn the_contract_accepts_an_extra_optional_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(
        names::INPUT_FEATURES,
        &[1, 1, T_FRAMES, N_MELS],
        DataType::F32,
      ),
      multi_array(
        "is_longer",
        &[1, 1],
        DataType::I32,
        true,
        2,
        vec![vec![1, 1]],
        &[1, 1],
      ),
    ],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(check(&description).is_ok());
}

/// An output the door READS that the graph may leave out: every geometry
/// clause passes and the prediction is still free to omit it.
#[test]
fn the_contract_refuses_an_optional_embeds_output() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_FEATURES,
      &[1, 1, T_FRAMES, N_MELS],
      DataType::F32,
    )],
    vec![multi_array(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
      true,
      2,
      vec![vec![1, EMBEDDING_DIM]],
      &[1, EMBEDDING_DIM],
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::AUDIO_EMBEDS),
    "{err}"
  );
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly this door's two features plus a state clears every per-feature
/// clause AND the input set — and only then meets
/// [`AudioEncoder::embed_window`], which predicts through the STATELESS API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_FEATURES,
      &[1, 1, T_FRAMES, N_MELS],
      DataType::F32,
    )],
    vec![fixed(
      names::AUDIO_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableState(name)) if name == "kv_cache")
  );
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **This door's `Checked::new` call site, pinned on a REAL model, in every
/// `cargo test`.**
///
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is COMMITTED, so
/// unlike everything else in this repository that loads a model this needs no
/// staged artifact and carries no `#[ignore]`. Silero is a real, fixed-shape,
/// six-feature CoreML graph that is simply not this door's model — the exact
/// shape of a mis-pointed `path`.
#[test]
fn the_audio_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; \
     looked for {}",
    bundle.display()
  );

  let model = Model::load(&bundle, ComputeUnits::CpuOnly).expect("the committed bundle loads");
  assert!(
    model.description().input(names::INPUT_FEATURES).is_none(),
    "silero declares no `input_features`, which is what makes it this gate's model"
  );

  let violation = Checked::new(model, &audio_contract())
    .expect_err("silero does not satisfy the CLAP audio contract");
  assert!(
    matches!(&violation, crate::model::contract::ContractViolation::Missing(m)
      if m.feature() == names::INPUT_FEATURES),
    "expected `input_features` missing, got {violation}"
  );
}
