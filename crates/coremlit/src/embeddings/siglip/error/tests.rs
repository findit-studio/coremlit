use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch(ContractMismatch::new(
    "pixel_values",
    "[1, 512, 768] float32".to_string(),
    "[1, 512, 768] float16".to_string(),
  ));
  let msg = e.to_string();
  assert!(msg.contains("pixel_values"), "{msg}");
  assert!(msg.contains("float16"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape(OutputShape::new(vec![768, 1], vec![1, 768]));
  let msg = e.to_string();
  assert!(
    msg.contains("[768, 1]") && msg.contains("[1, 768]"),
    "{msg}"
  );
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into siglip's Error.
  let e = Error::from(crate::PredictionError::MissingOutput(
    "image_features".to_string(),
  ));
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_carry_index() {
  assert!(Error::NonFiniteOutput(7).to_string().contains('7'));
  assert!(Error::NonFiniteEmbedding(3).to_string().contains('3'));
}

#[test]
fn image_dimensions_display_shows_both_dims() {
  let e = Error::ImageDimensions(ImageDimensions::new(640, 0));
  let msg = e.to_string();
  assert!(msg.contains("640") && msg.contains('0'), "{msg}");
}

#[test]
fn image_data_length_display_shows_expected_and_got() {
  let e = Error::ImageDataLength(ImageDataLength::new(100, 640 * 480 * 3));
  let msg = e.to_string();
  assert!(
    msg.contains("100") && msg.contains(&(640 * 480 * 3).to_string()),
    "{msg}"
  );
}

#[test]
fn pos_embed_length_display_shows_expected_and_got() {
  let e = Error::PosEmbedLength(PosEmbedLength::new(123, 16 * 16 * 768 * 4));
  let msg = e.to_string();
  assert!(
    msg.contains("123") && msg.contains(&(16 * 16 * 768 * 4).to_string()),
    "{msg}"
  );
}

#[test]
fn pos_embed_load_wraps_io_error_as_source() {
  let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
  let e = Error::PosEmbedLoad(io);
  // The source chain is preserved (`#[source]`).
  assert!(std::error::Error::source(&e).is_some(), "source chain lost");
}

#[test]
fn patch_count_display_shows_both() {
  let e = Error::PatchCount(PatchCount::new(600, 512));
  let msg = e.to_string();
  assert!(msg.contains("600") && msg.contains("512"), "{msg}");
}

#[test]
fn tokenizer_placeholder_display_names_the_placeholder() {
  let msg = Error::TokenizerPlaceholder.to_string();
  assert!(msg.contains("placeholder"), "{msg}");
}

#[test]
fn token_variants_carry_values() {
  assert!(
    Error::TokenCount(TokenCount::new(70, 64))
      .to_string()
      .contains("70")
  );
  assert!(
    Error::TokenIdRange(u32::MAX)
      .to_string()
      .contains(&u32::MAX.to_string())
  );
}

#[test]
fn preprocessed_length_display_names_feature() {
  let msg =
    Error::PreprocessedLength(PreprocessedLength::new("pixel_values", 100, 393_216)).to_string();
  assert!(msg.contains("pixel_values"), "{msg}");
  assert!(msg.contains("100"), "{msg}");
  assert!(msg.contains("393216"), "{msg}");
  assert!(Error::PreprocessedPatchBudget(0).to_string().contains('0'));
}

#[test]
fn preprocessed_mask_and_pad_variants_display_carry_diagnostics() {
  let non_finite =
    Error::PreprocessedNonFinite(PreprocessedNonFinite::new("position_embeddings", 7)).to_string();
  assert!(
    non_finite.contains("position_embeddings") && non_finite.contains('7'),
    "{non_finite}"
  );

  let mask_value = Error::PreprocessedMaskValue(PreprocessedMaskValue::new(1, 0.5)).to_string();
  assert!(
    mask_value.contains('1') && mask_value.contains("0.5"),
    "{mask_value}"
  );

  assert!(Error::PreprocessedMaskOrder(2).to_string().contains('2'));
  assert!(Error::PreprocessedMaskEmpty.to_string().contains("no real"));

  let pad =
    Error::PreprocessedPadNonZero(PreprocessedPadNonZero::new("pixel_values", 9)).to_string();
  assert!(pad.contains("pixel_values") && pad.contains('9'), "{pad}");
}

#[test]
fn patch_budget_mismatch_display_shows_both() {
  let msg = Error::PatchBudgetMismatch(PatchBudgetMismatch::new(256, 512)).to_string();
  assert!(msg.contains("256") && msg.contains("512"), "{msg}");
}
