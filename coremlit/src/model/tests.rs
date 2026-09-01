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

/// **The measurement this vocabulary rests on.** A graph converted at a plain
/// fixed shape does NOT report `MLMultiArrayShapeConstraintTypeUnspecified`
/// (raw 1). Every one of the six features of the staged
/// `silero-vad-unified-256ms-v6.2.1.mlmodelc` — `hasShapeFlexibility: "0"` on
/// all of them in its own `metadata.json` — reports raw type 2
/// (`…TypeEnumerated`) with ONE enumerated shape and one `NSRange` per axis of
/// length 1. A door that demanded a dedicated "fixed" code would reject every
/// fixed-shape artifact this crate ships.
///
/// What the measurement does NOT license is ignoring the code. It says raw 2
/// with unit spans is fixed — not that unit spans are fixed whatever the code
/// says; `a_range_constraint_stays_range_even_when_every_span_is_one` is the
/// other half, and the two together are the rule.
#[test]
fn a_fixed_shape_graph_is_classified_without_a_dedicated_fixed_type_code() {
  // `audio_input [1, 4160]`, exactly as the runtime reports it.
  assert_eq!(
    classify_shape_constraint(2, 1, &[1, 1]),
    ShapeConstraint::Fixed
  );
  // The spans stay load-bearing under that code: widen one axis and the same
  // code classifies enumerated.
  assert_eq!(
    classify_shape_constraint(2, 1, &[1, 4, 1]),
    ShapeConstraint::Enumerated
  );
}

/// A `RangeDims` axis admits more than one size, and that is what makes the
/// feature flexible — `audio::lid`'s `mel_features`, whose time axis is
/// `RangeDims [[1, 1], [10, 3001], [60, 60]]`, spans 1, 2992 and 1.
#[test]
fn a_range_axis_is_classified_range() {
  assert_eq!(
    classify_shape_constraint(3, 0, &[1, 2992, 1]),
    ShapeConstraint::Range
  );
}

/// More than one enumerated shape is enumerated, not a range — the per-axis
/// spans are then a bounding box over the list and cannot tell the two apart on
/// their own.
#[test]
fn several_enumerated_shapes_are_classified_enumerated() {
  assert_eq!(
    classify_shape_constraint(2, 32, &[32, 1, 1]),
    ShapeConstraint::Enumerated
  );
}

/// A code outside the two this door has measured is reported as exactly that
/// rather than resolved from whatever the contents happen to look like, and it
/// carries itself into the diagnosis.
#[test]
fn an_unmeasured_type_code_is_unknown_and_keeps_itself() {
  assert_eq!(
    classify_shape_constraint(1, 0, &[]),
    ShapeConstraint::Unknown(1)
  );
  assert_eq!(
    classify_shape_constraint(7, 3, &[]),
    ShapeConstraint::Unknown(7)
  );
}

/// Under a measured code, a constraint listing no axes at all pins nothing and
/// so is not fixed — "every axis admits one size" is vacuously true of no axes,
/// and that is not the same fact.
#[test]
fn an_empty_span_list_is_not_fixed() {
  assert_eq!(
    classify_shape_constraint(2, 1, &[]),
    ShapeConstraint::Enumerated
  );
}

/// A zero-length span accepts no size at all: degenerate, and not fixed.
#[test]
fn a_zero_length_span_is_not_fixed() {
  assert_ne!(
    classify_shape_constraint(2, 1, &[1, 0]),
    ShapeConstraint::Fixed
  );
}

#[test]
fn shape_constraint_renders_for_a_contract_mismatch_message() {
  assert_eq!(ShapeConstraint::Fixed.to_string(), "fixed");
  assert_eq!(ShapeConstraint::Enumerated.to_string(), "enumerated");
  assert_eq!(ShapeConstraint::Range.to_string(), "range");
  assert_eq!(ShapeConstraint::Unknown(1).to_string(), "unknown(1)");
}

/// **The `RangeDim`-with-unit-spans hole.** coremltools permits a `RangeDim`
/// whose lower and upper bounds are EQUAL. The dimension stays symbolic and the
/// converter still serialises a `shapeRange`, so CoreML reports raw type 3
/// (`…TypeRange`) with a `sizeRangeForDimension` length of 1 on every axis. A
/// classifier that reads the spans alone calls that fixed and lets the graph
/// through the door whose whole reason for existing is that fixed shape is what
/// keeps this model on the accelerator.
///
/// Spans alone cannot be trusted, and neither can the raw code alone (see
/// [`a_one_shape_enumerated_constraint_is_fixed`]). The rule that satisfies
/// both measurements uses both.
#[test]
fn a_range_constraint_stays_range_even_when_every_span_is_one() {
  assert_eq!(
    classify_shape_constraint(3, 0, &[1, 1, 1]),
    ShapeConstraint::Range,
    "an equal-bound `RangeDim` reports unit spans and is still a symbolic dimension"
  );
  assert_eq!(
    classify_shape_constraint(3, 1, &[1, 1]),
    ShapeConstraint::Range,
    "a `shapeRange` that also lists one enumerated shape is still a range"
  );
}

/// **The measurement this vocabulary rests on, restated as the rule it
/// supports.** A plain fixed export reports raw type 2 with ONE enumerated
/// shape and unit spans — never `…TypeUnspecified` — so a one-shape
/// `Enumerated` whose axes each admit exactly one size is fixed.
#[test]
fn a_one_shape_enumerated_constraint_is_fixed() {
  assert_eq!(
    classify_shape_constraint(2, 1, &[1, 1]),
    ShapeConstraint::Fixed
  );
  assert_eq!(
    classify_shape_constraint(2, 0, &[1, 1, 1]),
    ShapeConstraint::Fixed
  );
}

/// **Fails closed on a code this door has never measured.**
/// `…TypeUnspecified` (raw 1) says the constraint records nothing that decides
/// the question; an unknown code says the same thing louder. Neither may be
/// read as fixed just because the spans happen to be 1 — that is the
/// over-correction the span-only rule made.
#[test]
fn an_unspecified_or_unknown_raw_code_is_never_fixed() {
  assert_eq!(
    classify_shape_constraint(1, 1, &[1, 1]),
    ShapeConstraint::Unknown(1),
    "`…TypeUnspecified` establishes nothing, unit spans or not"
  );
  assert_eq!(
    classify_shape_constraint(99, 1, &[1]),
    ShapeConstraint::Unknown(99),
    "a code this door has never seen must carry itself into the diagnosis"
  );
}
