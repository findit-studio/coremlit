use super::*;

use crate::model::RawShapeConstraint;

// ── Fixtures ───────────────────────────────────────────────────────────────
//
// `ModelDescription::from_parts` / `FeatureInfo::from_parts` exist for these:
// one fixture family drives every clause of `check_load_contract`, so no door
// needs a fake of its own and no clause needs a staged artifact to be
// exercised. The verdict is never stated by a fixture — `from_parts` runs
// `classify_shape_constraint` over the raw contents, so a fixture cannot claim
// a `Fixed` its own numbers do not support.

/// The per-axis ranges a PINNED shape reports.
fn pinned(shape: &[usize]) -> Vec<AxisRange> {
  shape.iter().map(|d| AxisRange::new(*d, 1)).collect()
}

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    false,
    Some(RawShapeConstraint::new(
      2,
      vec![shape.to_vec()],
      pinned(shape),
    )),
  )
}

/// A `RangeDims` multi-array feature: raw type 3, no enumerated shapes, and
/// the per-axis bounds it was converted with. `shape` is the DEFAULT.
fn ranged(name: &str, shape: &[usize], dtype: DataType, ranges: &[AxisRange]) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    false,
    Some(RawShapeConstraint::new(3, Vec::new(), ranges.to_vec())),
  )
}

/// A fixed-shape feature the model declares OPTIONAL.
fn optional(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    true,
    Some(RawShapeConstraint::new(
      2,
      vec![shape.to_vec()],
      pinned(shape),
    )),
  )
}

// ── The identity door's contract, as the first consumer states it ──────────

const MEL: &str = "mel";
const EMBEDDING: &str = "embedding";
const MEL_SHAPE: &[usize] = &[1, 72, 401];
const EMBEDDING_SHAPE: &[usize] = &[1, 192];

/// `mel [1, 72, 401]` f32 in, `embedding [1, 192]` f32 out, no state.
fn identity_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      MEL,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(72), Dim::Exactly(401)],
    )],
    vec![FeatureContract::new(
      EMBEDDING,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(192)],
    )],
    StateContract::None,
  )
}

/// The converted redimnet artifact's description, as the Swift probe read it
/// back off the published bundle: `mel` raw 2, `[1, 72, 401]`, ranges
/// `1+1, 72+1, 401+1`, Float32; `embedding [1, 192]`; `states = []`.
fn redimnet_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  )
}

#[test]
fn the_identity_contract_accepts_the_published_redimnet_description() {
  assert_eq!(
    check_load_contract(&redimnet_description(), &identity_contract()),
    Ok(())
  );
}

// ── One clause at a time ───────────────────────────────────────────────────

#[test]
fn a_named_feature_the_model_does_not_declare_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed("audio", MEL_SHAPE, DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Missing(m) if m.feature() == MEL),
    "{error}"
  );
  assert!(error.to_string().contains("declares no feature `mel`"));
}

/// A named OUTPUT is checked the same way — the door reads it back, so its
/// absence fails every prediction just as an input's does.
#[test]
fn a_missing_output_is_refused_too() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![fixed("logits", EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Missing(m) if m.feature() == EMBEDDING),
    "{error}"
  );
}

/// **The dtype clause is not vacuous.** An mlprogram converted at fp16 without
/// an explicit `dtype=np.float32` reports Float16 I/O — measured on all three
/// mlprogram probes in `model/tests.rs` — so this is the clause that catches a
/// conversion-recipe regression at load.
#[test]
fn a_feature_of_the_wrong_element_type_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F16)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::DataType(d) if d.feature() == MEL),
    "{error}"
  );
  assert!(error.to_string().contains("float16"), "{error}");
}

#[test]
fn a_feature_with_a_different_number_of_axes_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, &[72, 401], DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Rank(r) if r.feature() == MEL),
    "{error}"
  );
  assert!(error.to_string().contains("rank 2"), "{error}");
}

/// **The equal-bound `RangeDim`.** It declares this contract's exact numbers
/// and reports `(d, 1)` on every axis, so every per-axis clause passes — and
/// the graph is still symbolic, which is what takes it off the accelerator.
/// The whole-feature verdict is the only thing that separates the two, which is
/// why an all-`Exactly` contract requires one.
#[test]
fn an_all_fixed_contract_refuses_a_flexible_feature_declaring_its_numbers() {
  let description = ModelDescription::from_parts(
    vec![ranged(MEL, MEL_SHAPE, DataType::F32, &pinned(MEL_SHAPE))],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Flexibility(f) if f.feature() == MEL),
    "{error}"
  );
  assert!(error.to_string().contains("is range"), "{error}");
}

/// **The ranges report the DEFAULT under an enumerated constraint** — measured
/// on the `enum3` probes — so a three-shape graph converted with `[1, 72, 401]`
/// as its default reports ranges indistinguishable from the fixed export's.
/// The verdict is again the only separator.
#[test]
fn an_all_fixed_contract_refuses_an_enumerated_feature_whose_ranges_look_pinned() {
  let mel = FeatureInfo::from_parts(
    MEL.to_string(),
    MEL_SHAPE.to_vec(),
    Some(DataType::F32),
    false,
    Some(RawShapeConstraint::new(
      2,
      vec![MEL_SHAPE.to_vec(), vec![1, 72, 201]],
      pinned(MEL_SHAPE),
    )),
  );
  let description = ModelDescription::from_parts(
    vec![mel],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Flexibility(f) if f.feature() == MEL),
    "{error}"
  );
  assert!(error.to_string().contains("is enumerated"), "{error}");
}

/// A feature whose constraint records nothing — the reading of every
/// `neuralnetwork` output — establishes nothing, so it cannot satisfy a fixed
/// axis.
#[test]
fn an_all_fixed_contract_refuses_an_unspecified_feature() {
  let embedding = FeatureInfo::from_parts(
    EMBEDDING.to_string(),
    EMBEDDING_SHAPE.to_vec(),
    Some(DataType::F32),
    false,
    Some(RawShapeConstraint::new(1, Vec::new(), Vec::new())),
  );
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![embedding],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Flexibility(f) if f.feature() == EMBEDDING),
    "{error}"
  );
  assert!(error.to_string().contains("is unspecified"), "{error}");
}

#[test]
fn an_exactly_axis_pinned_at_another_size_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, &[1, 72, 400], DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Axis(a) if a.feature() == MEL),
    "{error}"
  );
  let rendered = error.to_string();
  assert!(rendered.contains("axis 2 400"), "{rendered}");
  assert!(rendered.contains("axis 2 401"), "{rendered}");
}

/// A transposed shape of the same total size — the mutation no element-count
/// check can see.
#[test]
fn a_transposed_shape_of_the_same_size_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, &[1, 401, 72], DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  assert!(matches!(
    check_load_contract(&description, &identity_contract()),
    Err(ContractViolation::Axis(_))
  ));
}

// ── `AnyFixed`: the axis whose value the door READS rather than requires ───

/// A contract for a door configured by a manifest: the batch axis must be
/// pinned, and whatever it is pinned at is what the door then allocates for.
fn any_fixed_batch_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      MEL,
      DataType::F32,
      vec![Dim::AnyFixed, Dim::Exactly(72), Dim::Exactly(401)],
    )],
    Vec::new(),
    StateContract::None,
  )
}

#[test]
fn an_any_fixed_axis_accepts_whatever_one_size_is_pinned_and_is_read_back() {
  for batch in [1_usize, 3, 32] {
    let description = ModelDescription::from_parts(
      vec![fixed(MEL, &[batch, 72, 401], DataType::F32)],
      Vec::new(),
      Vec::new(),
    );
    assert_eq!(
      check_load_contract(&description, &any_fixed_batch_contract()),
      Ok(())
    );
    // What the door does after the check: read the value the contract
    // established is the only one.
    assert_eq!(description.input(MEL).expect("mel").shape()[0], batch);
  }
}

/// `AnyFixed` still requires the axis to be PINNED — "any size" is not "a range
/// of sizes", which is the whole distinction that keeps a flexible graph from
/// being bound at its default.
#[test]
fn an_any_fixed_axis_refuses_an_axis_admitting_more_than_one_size() {
  let description = ModelDescription::from_parts(
    vec![ranged(
      MEL,
      &[3, 72, 401],
      DataType::F32,
      &[
        AxisRange::inclusive(1, 8),
        AxisRange::new(72, 1),
        AxisRange::new(401, 1),
      ],
    )],
    Vec::new(),
    Vec::new(),
  );
  let error = check_load_contract(&description, &any_fixed_batch_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Flexibility(f) if f.feature() == MEL),
    "{error}"
  );
}

// ── `AtLeast`: the same read-back, with a floor ────────────────────────────

/// `audio::speaker`'s shape: the frame axis is the ARTIFACT's and is read
/// back, but a zero-frame graph is refused rather than read.
fn at_least_frames_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      MEL,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(72), Dim::AtLeast(1)],
    )],
    Vec::new(),
    StateContract::None,
  )
}

#[test]
fn an_at_least_axis_accepts_any_pinned_size_from_the_floor_up_and_is_read_back() {
  for frames in [1_usize, 401, 589, 4096] {
    let description = ModelDescription::from_parts(
      vec![fixed(MEL, &[1, 72, frames], DataType::F32)],
      Vec::new(),
      Vec::new(),
    );
    assert_eq!(
      check_load_contract(&description, &at_least_frames_contract()),
      Ok(())
    );
    assert_eq!(description.input(MEL).expect("mel").shape()[2], frames);
  }
}

/// **FALSIFIER (red first).** A zero-frame axis satisfies [`Dim::AnyFixed`] —
/// it admits exactly one size, and that size is `0` — so a graph declaring one
/// loads clean and every prediction then allocates zero-length rows. That is
/// the degenerate contract `audio::speaker`'s two doors each refused with a
/// hand-written `>= 1` beside the check, and the whole point of the floor is
/// that it is now checked with the rest rather than beside them.
#[test]
fn an_at_least_axis_refuses_a_zero_sized_axis_that_any_fixed_would_accept() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, &[1, 72, 0], DataType::F32)],
    Vec::new(),
    Vec::new(),
  );
  // The clause it tightens: without a floor this passes.
  let floorless = LoadContract::new(
    vec![FeatureContract::new(
      MEL,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(72), Dim::AnyFixed],
    )],
    Vec::new(),
    StateContract::None,
  );
  assert_eq!(check_load_contract(&description, &floorless), Ok(()));

  let error = check_load_contract(&description, &at_least_frames_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Axis(a) if a.feature() == MEL),
    "{error}"
  );
  let rendered = error.to_string();
  assert!(rendered.contains("axis 2 0"), "{rendered}");
  assert!(
    rendered.contains("axis 2 any one fixed size, at least 1"),
    "{rendered}"
  );
}

/// The floor does not weaken the pinning: a flexible axis whose bounds all sit
/// above the floor is still refused, because "at least" is a statement about
/// the ONE size the axis admits, not about a range of them.
#[test]
fn an_at_least_axis_still_requires_the_axis_to_be_pinned() {
  let description = ModelDescription::from_parts(
    vec![ranged(
      MEL,
      &[1, 72, 589],
      DataType::F32,
      &[
        AxisRange::new(1, 1),
        AxisRange::new(72, 1),
        AxisRange::inclusive(10, 4096),
      ],
    )],
    Vec::new(),
    Vec::new(),
  );
  let error = check_load_contract(&description, &at_least_frames_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Flexibility(f) if f.feature() == MEL),
    "{error}"
  );
}

// ── `lid` is expressible as a contract, not as an exemption ────────────────

/// **The proof that #137's one exception fits.** `audio::lid`'s `mel_features`
/// is `RangeDims [[1, 1], [10, 3001], [60, 60]]` with default shape
/// `[1, 301, 60]` — flexible BY DESIGN, because `lid::window` scores a ragged
/// tail at its own length. A blanket fixed-shape rule would hard-fail it; this
/// contract states the flexibility instead, so `lid` migrates onto the same
/// type as every other door rather than being carved out of it.
fn lid_shaped_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      "mel_features",
      DataType::F32,
      vec![
        Dim::Exactly(1),
        Dim::Range(AxisRange::inclusive(10, 3001)),
        Dim::Exactly(60),
      ],
    )],
    Vec::new(),
    StateContract::None,
  )
}

/// A description shaped like `lid`'s: the exact `RangeDims` its `model_io`
/// gate pins, at the artifact's own default shape.
fn lid_shaped_description(ranges: &[AxisRange]) -> ModelDescription {
  ModelDescription::from_parts(
    vec![ranged("mel_features", &[1, 301, 60], DataType::F32, ranges)],
    Vec::new(),
    Vec::new(),
  )
}

#[test]
fn a_lid_shaped_flexible_contract_is_expressible() {
  let description = lid_shaped_description(&[
    AxisRange::new(1, 1),
    AxisRange::inclusive(10, 3001),
    AxisRange::new(60, 1),
  ]);
  assert_eq!(
    check_load_contract(&description, &lid_shaped_contract()),
    Ok(())
  );
}

/// The flexible axis is checked against its BOUNDS, not against the default
/// shape — the check `lid`'s current door cannot make, because
/// [`FeatureInfo::shape`] reports `[1, 301, 60]` whatever the bounds are.
#[test]
fn a_flexible_axis_whose_bounds_differ_is_refused() {
  let description = lid_shaped_description(&[
    AxisRange::new(1, 1),
    AxisRange::inclusive(10, 1500),
    AxisRange::new(60, 1),
  ]);
  let error = check_load_contract(&description, &lid_shaped_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Axis(a) if a.feature() == "mel_features"),
    "{error}"
  );
  let rendered = error.to_string();
  assert!(rendered.contains("axis 1 10..=1500"), "{rendered}");
  assert!(rendered.contains("axis 1 10..=3001"), "{rendered}");
}

/// A fixed axis inside a flexible feature is still checked: under a `Range`
/// verdict an axis reading `(60, 1)` admits exactly 60, which is all
/// `Exactly(60)` claims — and one reading `(80, 1)` does not.
#[test]
fn a_fixed_axis_inside_a_flexible_feature_is_still_checked() {
  let description = lid_shaped_description(&[
    AxisRange::new(1, 1),
    AxisRange::inclusive(10, 3001),
    AxisRange::new(80, 1),
  ]);
  assert!(matches!(
    check_load_contract(&description, &lid_shaped_contract()),
    Err(ContractViolation::Axis(_))
  ));
}

/// A contract with a `Range` axis requires a flexible graph: a graph that
/// PINNED the time axis could not be fed the ragged tail the door exists to
/// score.
#[test]
fn a_flexible_contract_refuses_a_fully_fixed_feature() {
  let description = ModelDescription::from_parts(
    vec![fixed("mel_features", &[1, 301, 60], DataType::F32)],
    Vec::new(),
    Vec::new(),
  );
  let error = check_load_contract(&description, &lid_shaped_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::Flexibility(f) if f.feature() == "mel_features"),
    "{error}"
  );
}

/// A constraint carrying fewer ranges than the shape has axes pins only the
/// axes it lists, so the unlisted one is refused rather than assumed.
#[test]
fn an_axis_the_constraint_lists_no_range_for_is_refused() {
  let description = lid_shaped_description(&[AxisRange::new(1, 1), AxisRange::inclusive(10, 3001)]);
  let error = check_load_contract(&description, &lid_shaped_contract()).unwrap_err();
  assert!(matches!(&error, ContractViolation::Axis(_)), "{error}");
  assert!(error.to_string().contains("axis 2 none"), "{error}");
}

// ── The input set, and the state set ───────────────────────────────────────

/// **A graph carrying `mel` plus another REQUIRED input** passes every
/// per-feature clause and then fails on every prediction, because the door
/// supplies the features its contract names and nothing else.
#[test]
fn a_required_input_the_contract_does_not_name_is_refused() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(MEL, MEL_SHAPE, DataType::F32),
      fixed("speaker_mask", &[1, 401], DataType::F32),
    ],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::UnsatisfiableInput(i) if i.name() == "speaker_mask"),
    "{error}"
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one. Optionality is exactly the distinction this needs, and a count of
/// inputs cannot make it.
#[test]
fn an_optional_extra_input_is_accepted() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(MEL, MEL_SHAPE, DataType::F32),
      optional("mask", &[1, 401], DataType::F32),
    ],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  assert_eq!(
    check_load_contract(&description, &identity_contract()),
    Ok(())
  );
}

/// An extra OUTPUT is accepted: the door reads the outputs it names and
/// ignores the rest, so an extra one cannot make a prediction fail.
#[test]
fn an_extra_output_is_accepted() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![
      fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32),
      fixed("logits", &[1, 5994], DataType::F32),
    ],
    Vec::new(),
  );
  assert_eq!(
    check_load_contract(&description, &identity_contract()),
    Ok(())
  );
}

/// **FALSIFIER (red first).** A NAMED output the model declares OPTIONAL used
/// to pass every clause, because every clause is a statement about a feature
/// that IS declared: the dtype, the rank, the flexibility verdict and the axes
/// are all read off a `FeatureInfo` this description really has, and none of
/// them consulted `is_optional`.
///
/// What that blessed is a graph free to OMIT the feature from a prediction.
/// [`Checked::predict_with`] asks `Model::predict_with_outputs` for exactly the
/// names the contract carries, so the omission comes back as
/// `PredictionError::MissingOutput` at predict time — on a contract whose whole
/// job was to establish at LOAD time that the prediction can run.
#[test]
fn a_named_output_the_model_declares_optional_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![optional(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::OptionalOutput(o) if o.feature() == EMBEDDING),
    "{error}"
  );
  assert!(error.to_string().contains("`embedding`"), "{error}");

  // The clause is about the OPTIONALITY and nothing else: the same feature,
  // same geometry, declared required, still passes.
  assert_eq!(
    check_load_contract(&redimnet_description(), &identity_contract()),
    Ok(())
  );
}

/// **The asymmetry, pinned rather than left to a reader.** A NAMED INPUT the
/// model declares optional is deliberately ACCEPTED, and it is not the same
/// question: the door SUPPLIES the inputs its contract names, so one that is
/// merely permitted to be absent is supplied anyway and its optionality changes
/// nothing about the prediction. It is the OUTPUT direction that is asymmetric
/// — there the model decides whether the feature comes back.
///
/// The rule about inputs is the separate one this file's
/// `a_required_input_the_contract_does_not_name_is_refused` carries: a REQUIRED
/// input the contract does NOT name. Both still hold, and adding the output
/// rule to the input side would refuse an artifact that works.
#[test]
fn a_named_input_the_model_declares_optional_is_accepted() {
  let description = ModelDescription::from_parts(
    vec![optional(MEL, MEL_SHAPE, DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  assert_eq!(
    check_load_contract(&description, &identity_contract()),
    Ok(())
  );
}

/// The offender reported is the first BY NAME. `snapshot_features` sorts, so
/// it is stable across loads rather than an artefact of CoreML's dictionary
/// order.
#[test]
fn the_reported_unsatisfiable_input_is_stable() {
  let description = ModelDescription::from_parts(
    vec![
      fixed("aaa", &[1], DataType::F32),
      fixed(MEL, MEL_SHAPE, DataType::F32),
      fixed("zzz", &[1], DataType::F32),
    ],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::UnsatisfiableInput(i) if i.name() == "aaa"),
    "{error}"
  );
}

/// **State is not an input.** It lives in its own dictionary and never appears
/// among the ordinary inputs, so a stateful graph declaring exactly `mel` and
/// `embedding` plus a state clears every other clause — and then meets a door
/// predicting through the stateless API, which CoreML does not let a stateful
/// model be called with.
#[test]
fn a_declared_state_buffer_is_refused() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::UnsatisfiableState(s) if s.name() == "kv_cache"),
    "{error}"
  );
  assert!(
    error.to_string().contains("state buffer `kv_cache`"),
    "{error}"
  );
}

#[test]
fn the_reported_state_buffer_is_stable() {
  let description = ModelDescription::from_parts(
    vec![fixed(MEL, MEL_SHAPE, DataType::F32)],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    vec![
      fixed("aaa", &[1], DataType::F32),
      fixed("zzz", &[1], DataType::F32),
    ],
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::UnsatisfiableState(s) if s.name() == "aaa"),
    "{error}"
  );
}

// ── Rendering ──────────────────────────────────────────────────────────────

#[test]
fn a_dim_renders_for_a_violation_message() {
  assert_eq!(Dim::Exactly(401).to_string(), "401");
  assert_eq!(Dim::AnyFixed.to_string(), "any one fixed size");
  assert_eq!(
    Dim::AtLeast(1).to_string(),
    "any one fixed size, at least 1"
  );
  assert_eq!(
    Dim::Range(AxisRange::inclusive(10, 3001)).to_string(),
    "10..=3001"
  );
}

/// A non-multi-array feature carries neither element type nor constraint, and
/// both clauses render that as `none` rather than panicking or guessing.
#[test]
fn a_feature_with_no_multi_array_constraint_renders_as_none() {
  let description = ModelDescription::from_parts(
    vec![FeatureInfo::from_parts(
      MEL.to_string(),
      Vec::new(),
      None,
      false,
      None,
    )],
    vec![fixed(EMBEDDING, EMBEDDING_SHAPE, DataType::F32)],
    Vec::new(),
  );
  let error = check_load_contract(&description, &identity_contract()).unwrap_err();
  assert!(
    matches!(&error, ContractViolation::DataType(d) if d.observed() == "none"),
    "{error}"
  );
  assert!(error.to_string().contains("is none"), "{error}");
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **FALSIFIER (red first), on a REAL model, in every `cargo test`.**
///
/// Every other gate in this file drives [`check_load_contract`] over a
/// fixture. This one runs a real CoreML prediction through [`Checked`] against
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc`, which is COMMITTED
/// — 1.1 MiB, staged by no download — so unlike everything else in this
/// repository that predicts through a model it carries no `#[ignore]`.
///
/// Silero is this crate's only committed multi-output graph: three f32 inputs,
/// **three** f32 outputs, no state. The contract below names ONE of the three
/// outputs, which is exactly the situation
/// [`check_load_contract`]'s `an_extra_output_is_accepted` blesses — and
/// [`Model::predict_with`] answered it by converting all three anyway. A door
/// whose extra output were a string or a dictionary would have failed every
/// prediction on it; silero's are all tensors, so what this can measure is the
/// COUNT, which is the same fact one step before the failure.
///
/// Point `Checked::predict_with` back at [`Model::predict_with`] and this goes
/// red with three names where one was asked for.
#[test]
fn checked_materialises_only_the_outputs_its_contract_names() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; looked for {}",
    bundle.display()
  );
  let model =
    Model::load(&bundle, crate::ComputeUnits::CpuOnly).expect("the committed bundle loads");
  assert_eq!(
    model
      .description()
      .outputs()
      .iter()
      .map(FeatureInfo::name)
      .collect::<Vec<_>>(),
    vec!["new_cell_state", "new_hidden_state", "vad_output"],
    "this gate is about a graph with MORE outputs than the contract names"
  );

  // A contract over all three REQUIRED inputs — anything less is
  // `UnsatisfiableInput` — naming exactly one of the three outputs.
  let contract = LoadContract::new(
    vec![
      FeatureContract::new(
        "audio_input",
        DataType::F32,
        vec![Dim::Exactly(1), Dim::Exactly(4160)],
      ),
      FeatureContract::new(
        "cell_state",
        DataType::F32,
        vec![Dim::Exactly(1), Dim::Exactly(128)],
      ),
      FeatureContract::new(
        "hidden_state",
        DataType::F32,
        vec![Dim::Exactly(1), Dim::Exactly(128)],
      ),
    ],
    vec![FeatureContract::new(
      "vad_output",
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(1), Dim::Exactly(1)],
    )],
    StateContract::None,
  );
  let checked = Checked::new(model, &contract).expect("silero satisfies this contract");

  let audio = MultiArray::zeros(&[1, 4160], DataType::F32).expect("one 256 ms window");
  let hidden = MultiArray::zeros(&[1, 128], DataType::F32).expect("the LSTM's hidden state");
  let cell = MultiArray::zeros(&[1, 128], DataType::F32).expect("the LSTM's cell state");
  let outputs = checked
    .predict_with(&[
      ("audio_input", &audio),
      ("hidden_state", &hidden),
      ("cell_state", &cell),
    ])
    .expect("a real prediction through the door's own entry point");

  assert_eq!(
    outputs.names().collect::<Vec<_>>(),
    vec!["vad_output"],
    "the door asked for one output; the other two must not have been materialised"
  );
  assert_eq!(outputs.len(), 1);
  assert_eq!(outputs.get("vad_output").unwrap().shape(), &[1, 1, 1]);
}
