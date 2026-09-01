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
/// length 1. So the verdict comes from the per-axis spans, and a door that
/// matched the raw code would reject every fixed-shape artifact this crate
/// ships.
#[test]
fn a_fixed_shape_graph_is_classified_from_its_axis_spans_not_its_type_code() {
  // `audio_input [1, 4160]`, exactly as the runtime reports it.
  assert_eq!(
    classify_shape_constraint(2, 1, &[1, 1]),
    ShapeConstraint::Fixed
  );
  // ... and the raw code is not what decided it.
  for raw in [1, 2, 3, 99] {
    assert_eq!(
      classify_shape_constraint(raw, 1, &[1, 1, 1]),
      ShapeConstraint::Fixed,
      "raw type {raw} must not change a verdict the spans already decide"
    );
  }
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

/// A constraint carrying no per-axis sizes decides nothing, and is reported as
/// exactly that rather than assumed fixed. The raw code rides along for
/// diagnosis.
#[test]
fn an_empty_constraint_is_unknown_and_keeps_its_raw_code() {
  assert_eq!(
    classify_shape_constraint(1, 0, &[]),
    ShapeConstraint::Unknown(1)
  );
  assert_eq!(
    classify_shape_constraint(7, 3, &[]),
    ShapeConstraint::Unknown(7)
  );
}

/// A zero-length span accepts no size at all: degenerate, and not fixed.
#[test]
fn a_zero_length_span_is_not_fixed() {
  assert_ne!(
    classify_shape_constraint(3, 0, &[1, 0]),
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
