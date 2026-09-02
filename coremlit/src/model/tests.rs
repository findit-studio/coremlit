use super::*;
use crate::ComputeUnits;

#[test]
fn model_is_send() {
  fn assert_send<T: Send>() {}
  assert_send::<Model>();
}

#[test]
fn load_missing_path_is_not_found() {
  let err = Model::load("/nonexistent/Foo.mlmodelc", ComputeUnits::CpuOnly).unwrap_err();
  assert!(matches!(err, crate::LoadError::NotFound(_)));
}

#[test]
fn compile_missing_source_is_not_found() {
  let err = Model::compile("/nonexistent/foo.mlpackage").unwrap_err();
  assert!(matches!(err, crate::CompileError::NotFound(_)));
}

// ── ShapeConstraint ────────────────────────────────────────────────────────

/// Enumerated-shape lists, spelled as the constraint reports them.
fn shapes(list: &[&[usize]]) -> Vec<Vec<usize>> {
  list.iter().map(|s| s.to_vec()).collect()
}

/// The per-axis ranges a PINNED shape reports: `(d, 1)` for every axis.
fn pinned(shape: &[usize]) -> Vec<AxisRange> {
  shape.iter().map(|d| AxisRange::new(*d, 1)).collect()
}

/// One `MLMultiArrayShapeConstraint` as the Swift probe read it back.
struct Reading {
  /// `<bundle> <input|output> <feature>`, as the probe prints it.
  what: &'static str,
  raw_type: isize,
  declared: &'static [usize],
  enumerated: &'static [&'static [usize]],
  /// `(location, length)` per axis, straight off `sizeRangeForDimension`.
  ranges: &'static [(usize, usize)],
  /// `MLMultiArrayDataType` raw code.
  dtype: isize,
  verdict: ShapeConstraint,
}

/// **The measurements this vocabulary rests on, committed.**
///
/// Probe artifacts were built with the conversion recipes' own coremltools
/// 8.3.0 — one traced graph exported six ways, three as `mlprogram` at
/// `compute_precision=FLOAT16` and three as `neuralnetwork` — compiled to
/// `.mlmodelc`, and their `MLMultiArrayShapeConstraint` read back with a Swift
/// probe (`MLModelConfiguration.computeUnits = .cpuOnly`). Every row is one
/// line of that probe's output, transcribed.
///
/// The table is committed rather than the artifacts because the artifacts are
/// 4 MB of build output whose whole information content is these numbers, and
/// because a test over the numbers runs in every `cargo test` with no
/// coremltools, no Python and no model present. Re-derive it by rebuilding the
/// probes; the recipe is recorded in coremlit issue #138 §5.
const PROBE_READINGS: &[Reading] = &[
  // A plain fixed export does NOT report `…TypeUnspecified`. It reports raw 2
  // with ONE enumerated shape equal to the declared one and `(d, 1)` on every
  // axis — the measurement a door demanding a dedicated "fixed" code would
  // reject every fixed-shape artifact this crate ships over.
  Reading {
    what: "fixed.mlmodelc input mel",
    raw_type: 2,
    declared: &[1, 72, 401],
    enumerated: &[&[1, 72, 401]],
    ranges: &[(1, 1), (72, 1), (401, 1)],
    dtype: 65552,
    verdict: ShapeConstraint::Fixed,
  },
  // An mlprogram output downstream of a FIXED input keeps the same reading —
  // it is flexibility upstream, not being an output, that erases it (contrast
  // the `enum3` and `nn_fixed` outputs below).
  Reading {
    what: "fixed.mlmodelc output y",
    raw_type: 2,
    declared: &[1, 72, 401],
    enumerated: &[&[1, 72, 401]],
    ranges: &[(1, 1), (72, 1), (401, 1)],
    dtype: 65552,
    verdict: ShapeConstraint::Fixed,
  },
  // **The ranges report the DEFAULT only.** Three enumerated shapes
  // (`[1,72,401]`, `[1,72,201]`, `[1,72,801]`) and the ranges still read
  // `(401, 1)` on the time axis — not a bounding box over the list. So the
  // count is the sole discriminator between this row and the first, and the
  // classifier does not consult the ranges under raw 2 with two or more
  // shapes.
  Reading {
    what: "enum3.mlmodelc input mel",
    raw_type: 2,
    declared: &[1, 72, 401],
    enumerated: &[&[1, 72, 401], &[1, 72, 201], &[1, 72, 801]],
    ranges: &[(1, 1), (72, 1), (401, 1)],
    dtype: 65552,
    verdict: ShapeConstraint::Enumerated,
  },
  // **The equal-bound `RangeDim` hole.** `RangeDim(401, 401)` reports raw 3
  // with `(d, 1)` on every axis — indistinguishable from the fixed export
  // above on the ranges alone, and still a symbolic dimension, which is what
  // takes a graph off the accelerator.
  Reading {
    what: "range_equal.mlmodelc input mel",
    raw_type: 3,
    declared: &[1, 72, 401],
    enumerated: &[],
    ranges: &[(1, 1), (72, 1), (401, 1)],
    dtype: 65552,
    verdict: ShapeConstraint::Range,
  },
  // `RangeDim(10, 3001)` — `audio::lid`'s `mel_features` shape exactly.
  Reading {
    what: "range_open.mlmodelc input mel",
    raw_type: 3,
    declared: &[1, 72, 401],
    enumerated: &[],
    ranges: &[(1, 1), (72, 1), (10, 2992)],
    dtype: 65552,
    verdict: ShapeConstraint::Range,
  },
  // A `neuralnetwork` export's fixed input reads exactly like an mlprogram's.
  Reading {
    what: "nn_fixed.mlmodelc input mel",
    raw_type: 2,
    declared: &[1, 72, 401],
    enumerated: &[&[1, 72, 401]],
    ranges: &[(1, 1), (72, 1), (401, 1)],
    dtype: 65568,
    verdict: ShapeConstraint::Fixed,
  },
  // **`Unspecified` is the COMMON case, not an exotic one.** Every output of a
  // `neuralnetwork` export reads this way even when its input is fixed, as
  // does every output downstream of a flexible input in either converter. Such
  // a feature carries no ranges and an EMPTY declared shape, so nothing about
  // its geometry can be read off the description at all.
  Reading {
    what: "nn_fixed.mlmodelc output y",
    raw_type: 1,
    declared: &[],
    enumerated: &[],
    ranges: &[],
    dtype: 65568,
    verdict: ShapeConstraint::Unspecified,
  },
  // An UNBOUNDED range (`RangeDim(1, -1)`), which only a `neuralnetwork`
  // export produces: `location = 1`, `length = isize::MAX`.
  Reading {
    what: "nn_range_unbounded.mlmodelc input mel",
    raw_type: 3,
    declared: &[1, 72, 401],
    enumerated: &[],
    ranges: &[(1, 1), (72, 1), (1, 9_223_372_036_854_775_807)],
    dtype: 65568,
    verdict: ShapeConstraint::Range,
  },
];

/// Every committed probe reading classifies to the verdict it was measured at.
///
/// This is the hermetic half of the measurement: the artifacts are gone, the
/// numbers are not, and a change to [`classify_shape_constraint`] that
/// disagrees with any of them is a change that disagrees with CoreML.
#[test]
fn every_measured_probe_reading_classifies_to_its_verdict() {
  for reading in PROBE_READINGS {
    let ranges: Vec<AxisRange> = reading
      .ranges
      .iter()
      .map(|(location, length)| AxisRange::new(*location, *length))
      .collect();
    assert_eq!(
      classify_shape_constraint(
        reading.raw_type,
        reading.declared,
        &shapes(reading.enumerated),
        &ranges
      ),
      reading.verdict,
      "{}",
      reading.what
    );
  }
}

/// **The dtype clause is not vacuous.** The three `mlprogram` probes were
/// converted at `compute_precision=FLOAT16` with no explicit
/// `dtype=np.float32`, and every one of them reports **Float16** I/O; the
/// `neuralnetwork` exports report Float32. So a door contracting for `f32`
/// would catch exactly that recipe regression at load — the check earns its
/// place rather than restating a constant.
#[test]
fn an_fp16_mlprogram_conversion_reports_float16_io() {
  for reading in PROBE_READINGS {
    let declared = crate::DataType::from_raw(reading.dtype);
    let expected = if reading.what.starts_with("nn_") {
      crate::DataType::F32
    } else {
      crate::DataType::F16
    };
    assert_eq!(declared, expected, "{}", reading.what);
  }
}

/// **The measurement this vocabulary rests on.** A graph converted at a plain
/// fixed shape does NOT report `MLMultiArrayShapeConstraintTypeUnspecified`
/// (raw 1). Every one of the six features of the staged
/// `silero-vad-unified-256ms-v6.2.1.mlmodelc` — `hasShapeFlexibility: "0"` on
/// all of them in its own `metadata.json` — reports raw type 2
/// (`…TypeEnumerated`) with ONE enumerated shape equal to the declared one and
/// one `NSRange` per axis reading `(size, 1)`. A door that demanded a dedicated
/// "fixed" code would reject every fixed-shape artifact this crate ships.
///
/// What the measurement does NOT license is ignoring the code:
/// [`a_range_constraint_stays_range_even_when_every_range_is_one`] is the other
/// half, and the two together are the rule.
#[test]
fn a_fixed_shape_graph_is_classified_without_a_dedicated_fixed_type_code() {
  // `audio_input [1, 4160]`, exactly as the runtime reports it.
  assert_eq!(
    classify_shape_constraint(2, &[1, 4160], &shapes(&[&[1, 4160]]), &pinned(&[1, 4160])),
    ShapeConstraint::Fixed
  );
}

/// A `RangeDims` axis admits more than one size, and that is what makes the
/// feature flexible — `audio::lid`'s `mel_features`, whose time axis is
/// `RangeDims [[1, 1], [10, 3001], [60, 60]]`.
#[test]
fn a_range_axis_is_classified_range() {
  assert_eq!(
    classify_shape_constraint(
      3,
      &[1, 401, 60],
      &[],
      &[
        AxisRange::new(1, 1),
        AxisRange::new(10, 2992),
        AxisRange::new(60, 1)
      ]
    ),
    ShapeConstraint::Range
  );
}

/// More than one enumerated shape is enumerated, not fixed — and the ranges
/// cannot say so, because under this code they report the DEFAULT shape (see
/// the `enum3` probe reading). The count is the whole discriminator.
#[test]
fn several_enumerated_shapes_are_classified_enumerated() {
  assert_eq!(
    classify_shape_constraint(
      2,
      &[1, 72, 401],
      &shapes(&[&[1, 72, 401], &[1, 72, 201]]),
      &pinned(&[1, 72, 401])
    ),
    ShapeConstraint::Enumerated
  );
}

/// **`…TypeUnspecified` is named for what it is, and it is the common case.**
/// It was `Unknown(1)`, documented as "a code this door has never measured";
/// the probes measured it as the reading of every output downstream of a
/// flexible input and of every `neuralnetwork` output even when fixed.
///
/// Naming it changes nothing about the refusal — a constraint that records
/// nothing establishes nothing, unit ranges or not.
#[test]
fn an_unspecified_constraint_is_named_and_is_never_fixed() {
  assert_eq!(
    classify_shape_constraint(1, &[], &[], &[]),
    ShapeConstraint::Unspecified
  );
  assert_eq!(
    classify_shape_constraint(1, &[1, 1], &shapes(&[&[1, 1]]), &pinned(&[1, 1])),
    ShapeConstraint::Unspecified,
    "`…TypeUnspecified` establishes nothing, unit ranges or not"
  );
}

/// A code outside the three this door has measured is reported as exactly that
/// rather than resolved from whatever the contents happen to look like, and it
/// carries itself into the diagnosis.
#[test]
fn an_unmeasured_type_code_is_unknown_and_keeps_itself() {
  assert_eq!(
    classify_shape_constraint(7, &[1], &[], &[]),
    ShapeConstraint::Unknown(7)
  );
  assert_eq!(
    classify_shape_constraint(99, &[1], &shapes(&[&[1]]), &pinned(&[1])),
    ShapeConstraint::Unknown(99),
    "a code this door has never seen must carry itself into the diagnosis"
  );
}

/// **The `RangeDim`-with-unit-ranges hole.** coremltools permits a `RangeDim`
/// whose lower and upper bounds are EQUAL. The dimension stays symbolic and the
/// converter still serialises a `shapeRange`, so CoreML reports raw type 3
/// (`…TypeRange`) with `(d, 1)` on every axis — measured, as the
/// `range_equal.mlmodelc` reading above. A classifier that reads the ranges
/// alone calls that fixed and lets the graph through the door whose whole
/// reason for existing is that fixed shape is what keeps the model on the
/// accelerator.
#[test]
fn a_range_constraint_stays_range_even_when_every_range_is_one() {
  assert_eq!(
    classify_shape_constraint(3, &[1, 72, 401], &[], &pinned(&[1, 72, 401])),
    ShapeConstraint::Range,
    "an equal-bound `RangeDim` reports unit ranges and is still symbolic"
  );
  assert_eq!(
    classify_shape_constraint(
      3,
      &[1, 72, 401],
      &shapes(&[&[1, 72, 401]]),
      &pinned(&[1, 72, 401])
    ),
    ShapeConstraint::Range,
    "a `shapeRange` that also lists one enumerated shape is still a range"
  );
}

// ── The `Fixed` conjuncts, each falsified on its own ───────────────────────
//
// `Fixed` is `raw == 2` AND `enumerated == [declared]` AND one range per axis
// AND every range `(d, 1)`. Each test below breaks exactly one of those and
// nothing else, so each is the falsifier for one conjunct: delete that clause
// from `classify_shape_constraint` and exactly one of them reds.

/// **Conjunct 1: the raw code.** Was already covered by the range and
/// unspecified tests above; restated here as the mutation it guards, so the
/// four sit together.
#[test]
fn the_raw_code_conjunct_is_load_bearing() {
  assert_ne!(
    classify_shape_constraint(3, &[1, 72, 401], &[], &pinned(&[1, 72, 401])),
    ShapeConstraint::Fixed
  );
}

/// **Conjunct 2a: a constraint listing NO enumerated shape.** Unmeasured —
/// coremltools refuses an `EnumeratedShapes` of length 1, and no producer of a
/// zero-length list was found. The old rule accepted it (`enumerated <= 1`) and
/// called it fixed.
#[test]
fn raw_two_listing_no_enumerated_shape_fails_closed() {
  assert_eq!(
    classify_shape_constraint(2, &[1, 72, 401], &[], &pinned(&[1, 72, 401])),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::NoShapes)
  );
}

/// **Conjunct 2b: the sole shape must BE the declared one.** The old rule
/// counted the shapes and never compared them, so a constraint accepting only
/// `[1, 72, 201]` while declaring `[1, 72, 401]` — a model whose default is not
/// among the shapes it accepts — was fixed at the declared numbers, and a door
/// would have allocated for a shape the graph refuses.
#[test]
fn a_sole_enumerated_shape_that_is_not_the_declared_one_fails_closed() {
  assert_eq!(
    classify_shape_constraint(
      2,
      &[1, 72, 401],
      &shapes(&[&[1, 72, 201]]),
      &pinned(&[1, 72, 401])
    ),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SoleShapeIsNotDeclared)
  );
}

/// **Conjunct 3: one range per axis.** A constraint carrying fewer ranges than
/// the shape has axes pins only the axes it lists; the old rule read "every
/// range is 1" over whatever list it was given and never checked that the list
/// covered the shape.
#[test]
fn a_range_list_that_does_not_cover_every_axis_fails_closed() {
  assert_eq!(
    classify_shape_constraint(
      2,
      &[1, 72, 401],
      &shapes(&[&[1, 72, 401]]),
      &pinned(&[1, 72])
    ),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape)
  );
}

/// **Conjunct 4: every range is `(d, 1)` — the declared size, admitting one.**
/// Both halves: a range wide enough for a second size, and a range of exactly
/// one size that is not the declared one.
#[test]
fn a_range_that_does_not_pin_the_declared_size_fails_closed() {
  assert_eq!(
    classify_shape_constraint(
      2,
      &[1, 72, 401],
      &shapes(&[&[1, 72, 401]]),
      &[
        AxisRange::new(1, 1),
        AxisRange::new(72, 1),
        AxisRange::new(401, 2)
      ]
    ),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape),
    "an axis admitting a second size is not pinned"
  );
  assert_eq!(
    classify_shape_constraint(
      2,
      &[1, 72, 401],
      &shapes(&[&[1, 72, 401]]),
      &[
        AxisRange::new(1, 1),
        AxisRange::new(72, 1),
        AxisRange::new(400, 1)
      ]
    ),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape),
    "an axis pinned at a size other than the declared one is not this shape"
  );
}

/// A zero-length range accepts no size at all: degenerate, and not fixed.
#[test]
fn a_zero_length_range_is_not_fixed() {
  assert_eq!(
    classify_shape_constraint(
      2,
      &[1, 72],
      &shapes(&[&[1, 72]]),
      &[AxisRange::new(1, 1), AxisRange::new(72, 0)]
    ),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape)
  );
}

/// A shape with no axes pins nothing — "every axis admits one size" is
/// vacuously true of no axes, and that is not the same fact. Unmeasured under
/// raw 2 (the shape-less features the probes found all report raw 1), so it
/// fails closed rather than being resolved by the vacuous truth.
#[test]
fn a_declared_shape_with_no_axes_pins_nothing() {
  assert_eq!(
    classify_shape_constraint(2, &[], &shapes(&[&[]]), &[]),
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape)
  );
}

#[test]
fn shape_constraint_renders_for_a_contract_mismatch_message() {
  assert_eq!(ShapeConstraint::Fixed.to_string(), "fixed");
  assert_eq!(ShapeConstraint::Enumerated.to_string(), "enumerated");
  assert_eq!(ShapeConstraint::Range.to_string(), "range");
  assert_eq!(ShapeConstraint::Unspecified.to_string(), "unspecified");
  assert_eq!(ShapeConstraint::Unknown(7).to_string(), "unknown(7)");
  assert_eq!(
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::NoShapes).to_string(),
    "unmeasured(no enumerated shape)"
  );
  assert_eq!(
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SoleShapeIsNotDeclared).to_string(),
    "unmeasured(sole enumerated shape is not the declared shape)"
  );
  assert_eq!(
    ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape).to_string(),
    "unmeasured(per-axis ranges do not pin the declared shape)"
  );
}

// ── AxisRange ──────────────────────────────────────────────────────────────

/// The two spellings agree, and `inclusive` is the one a contract states a
/// `RangeDim(min, max)` axis in: `lid`'s `RangeDim(10, 3001)` is the 2992
/// consecutive sizes from 10, which is what the `range_open` probe read back.
#[test]
fn an_inclusive_axis_range_is_the_measured_span() {
  assert_eq!(AxisRange::inclusive(10, 3001), AxisRange::new(10, 2992));
  assert_eq!(AxisRange::inclusive(401, 401), AxisRange::new(401, 1));
  assert_eq!(AxisRange::inclusive(10, 3001).min(), 10);
  assert_eq!(AxisRange::inclusive(10, 3001).count(), 2992);
}

/// `max < min` is a range no constraint produces; it saturates to one size
/// rather than wrapping, so a mis-stated contract refuses a model instead of
/// accepting `usize::MAX` sizes.
#[test]
fn an_inverted_inclusive_range_saturates_rather_than_wrapping() {
  assert_eq!(AxisRange::inclusive(400, 10), AxisRange::new(400, 1));
}

#[test]
fn an_axis_range_renders_for_a_contract_mismatch_message() {
  assert_eq!(AxisRange::new(401, 1).to_string(), "401");
  assert_eq!(AxisRange::new(10, 2992).to_string(), "10..=3001");
  assert_eq!(AxisRange::new(7, 0).to_string(), "(no size)");
}
