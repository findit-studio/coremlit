use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch {
    feature: "pixel_values",
    expected: "[1, 512, 768] float32".to_string(),
    actual: "[1, 512, 768] float16".to_string(),
  };
  let msg = e.to_string();
  assert!(msg.contains("pixel_values"), "{msg}");
  assert!(msg.contains("float16"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape {
    got: vec![768, 1],
    expected: vec![1, 768],
  };
  let msg = e.to_string();
  assert!(
    msg.contains("[768, 1]") && msg.contains("[1, 768]"),
    "{msg}"
  );
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into siglip's Error.
  let e = Error::from(crate::PredictionError::MissingOutput {
    name: "image_features".to_string(),
  });
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_carry_index() {
  assert!(
    Error::NonFiniteOutput { index: 7 }
      .to_string()
      .contains('7')
  );
  assert!(
    Error::NonFiniteEmbedding { component_index: 3 }
      .to_string()
      .contains('3')
  );
}

#[test]
fn image_dimensions_display_shows_both_dims() {
  let e = Error::ImageDimensions {
    width: 640,
    height: 0,
  };
  let msg = e.to_string();
  assert!(msg.contains("640") && msg.contains('0'), "{msg}");
}

#[test]
fn image_data_length_display_shows_expected_and_got() {
  let e = Error::ImageDataLength {
    got: 100,
    expected: 640 * 480 * 3,
  };
  let msg = e.to_string();
  assert!(
    msg.contains("100") && msg.contains(&(640 * 480 * 3).to_string()),
    "{msg}"
  );
}

#[test]
fn pos_embed_length_display_shows_expected_and_got() {
  let e = Error::PosEmbedLength {
    got: 123,
    expected: 16 * 16 * 768 * 4,
  };
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
  let e = Error::PatchCount { got: 600, max: 512 };
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
    Error::TokenCount { got: 70, max: 64 }
      .to_string()
      .contains("70")
  );
  assert!(
    Error::TokenIdRange { id: u32::MAX }
      .to_string()
      .contains(&u32::MAX.to_string())
  );
}

#[test]
fn preprocessed_length_display_names_feature() {
  let msg = Error::PreprocessedLength {
    feature: "pixel_values",
    got: 100,
    expected: 393_216,
  }
  .to_string();
  assert!(msg.contains("pixel_values"), "{msg}");
  assert!(msg.contains("100"), "{msg}");
  assert!(msg.contains("393216"), "{msg}");
  assert!(
    Error::PreprocessedPatchBudget { max_num_patches: 0 }
      .to_string()
      .contains('0')
  );
}

#[test]
fn preprocessed_mask_and_pad_variants_display_carry_diagnostics() {
  let non_finite = Error::PreprocessedNonFinite {
    feature: "position_embeddings",
    index: 7,
  }
  .to_string();
  assert!(
    non_finite.contains("position_embeddings") && non_finite.contains('7'),
    "{non_finite}"
  );

  let mask_value = Error::PreprocessedMaskValue {
    index: 1,
    value: 0.5,
  }
  .to_string();
  assert!(
    mask_value.contains('1') && mask_value.contains("0.5"),
    "{mask_value}"
  );

  assert!(
    Error::PreprocessedMaskOrder { index: 2 }
      .to_string()
      .contains('2')
  );
  assert!(Error::PreprocessedMaskEmpty.to_string().contains("no real"));

  let pad = Error::PreprocessedPadNonZero {
    feature: "pixel_values",
    index: 9,
  }
  .to_string();
  assert!(pad.contains("pixel_values") && pad.contains('9'), "{pad}");
}

#[test]
fn patch_budget_mismatch_display_shows_both() {
  let msg = Error::PatchBudgetMismatch {
    input: 256,
    model: 512,
  }
  .to_string();
  assert!(msg.contains("256") && msg.contains("512"), "{msg}");
}
