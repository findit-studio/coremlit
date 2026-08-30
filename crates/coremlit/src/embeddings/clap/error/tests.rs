use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch {
    feature: "input_features",
    expected: "[1, 1, 1001, 64] float32".to_string(),
    actual: "[1, 1, 1001, 64] float16".to_string(),
  };
  let msg = e.to_string();
  assert!(msg.contains("input_features"), "{msg}");
  assert!(msg.contains("float16"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape {
    got: vec![512, 1],
    expected: vec![1, 512],
  };
  let msg = e.to_string();
  assert!(
    msg.contains("[512, 1]") && msg.contains("[1, 512]"),
    "{msg}"
  );
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into clapkit's Error.
  let e = Error::from(crate::PredictionError::MissingOutput {
    name: "audio_embeds".to_string(),
  });
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_carry_index() {
  assert!(Error::NonFiniteInput { index: 7 }.to_string().contains('7'));
  assert!(
    Error::NonFiniteEmbedding { component_index: 3 }
      .to_string()
      .contains('3')
  );
}

#[test]
fn from_winditerror_is_total_and_does_not_special_case_empty() {
  // The blanket `From<WinditError> for Error` must be a lossless, total wrap
  // into `Error::Windowing` — no variant gets special-cased here. This is the
  // conversion a downstream `SmoothPolicy`/`Smoother` caller takes on a plain
  // `?` (both re-exported at `clap::{SmoothPolicy, Smoother}`), so a custom
  // policy's `WinditError::Empty` on a NONEMPTY stream must surface as
  // `Windowing(Empty)`, not the misleading `EmptyWindows` ("cannot aggregate
  // zero window embeddings"). Only `aggregate()` may map `Empty ->
  // EmptyWindows`, and only at its own call site, where an empty error and an
  // empty input slice are actually the same event.
  let e = Error::from(WinditError::Empty);
  assert!(
    matches!(e, Error::Windowing(WinditError::Empty)),
    "the blanket From<WinditError> impl must not special-case Empty; got {e:?}"
  );
}
