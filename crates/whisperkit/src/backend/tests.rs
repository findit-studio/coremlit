use super::*;

#[test]
fn dims_defaults_are_tiny_and_projections_work() {
  let d = ModelDims::new();
  assert_eq!(d.vocab(), 51865);
  assert_eq!(d.n_mels(), 80);
  assert_eq!(d.embed_dim(), 384);
  assert_eq!(d.kv_dim(), 1536);
  assert_eq!(d.max_token_context(), 224);
  assert_eq!(d.n_audio_ctx(), 1500);
  assert_eq!(d.window_samples(), 480_000);
  assert!(d.is_multilingual());
  assert!(!d.with_vocab(51864).is_multilingual()); // ModelUtilities.isModelMultilingual
  assert_eq!(ModelDims::default(), ModelDims::new());
}

#[test]
fn alignment_view_rows() {
  let data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
  let view = AlignmentView::new(&data, 2, 3);
  assert_eq!(view.rows(), 2);
  assert_eq!(view.row(1), &[3.0, 4.0, 5.0]);
}

#[test]
fn alignment_matrix_round_trips_a_view() {
  let data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
  let view = AlignmentView::new(&data, 2, 3);
  let matrix = view.to_matrix();
  assert_eq!(matrix.rows(), 2);
  assert_eq!(matrix.cols(), 3);
  assert_eq!(matrix.view().row(1), &[3.0, 4.0, 5.0]);
}

#[test]
fn backend_error_displays_structured() {
  let e = BackendError::MissingFeature {
    model: "decoder",
    name: "logits",
  };
  assert_eq!(
    e.to_string(),
    "decoder model output is missing feature `logits`"
  );
  let e = BackendError::AudioLength {
    got: 100,
    expected: 480_000,
  };
  assert!(e.to_string().contains("480000"));
}
