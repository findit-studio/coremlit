use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch(ContractMismatch::new(
    "input_ids",
    "[1, 512] int32".to_string(),
    "[1, 512] float16".to_string(),
  ));
  let msg = e.to_string();
  assert!(msg.contains("input_ids"), "{msg}");
  assert!(msg.contains("float16"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape(OutputShape::new(vec![384, 1], vec![1, 384]));
  let msg = e.to_string();
  assert!(
    msg.contains("[384, 1]") && msg.contains("[1, 384]"),
    "{msg}"
  );
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into granite's Error.
  let e = Error::from(crate::PredictionError::MissingOutput(
    "embedding".to_string(),
  ));
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_carry_index() {
  assert!(Error::NonFiniteOutput(7).to_string().contains('7'));
  assert!(Error::NonFiniteEmbedding(3).to_string().contains('3'));
}

#[test]
fn tokenizer_contract_mismatch_display_names_check() {
  let e = Error::TokenizerContractMismatch(TokenizerContractMismatch::new(
    "vocab size",
    "180000".to_string(),
    "32".to_string(),
  ));
  let msg = e.to_string();
  assert!(msg.contains("vocab size"), "{msg}");
  assert!(msg.contains("180000") && msg.contains("32"), "{msg}");
}

#[test]
fn input_too_large_display_shows_sizes() {
  let e = Error::InputTooLarge(InputTooLarge::new(8_388_608, 1_048_576));
  let msg = e.to_string();
  assert!(msg.contains("8388608") && msg.contains("1048576"), "{msg}");
}

#[test]
fn contentless_input_over_budget_display_shows_span_and_counts() {
  let e = Error::ContentlessInputOverBudget(ContentlessInputOverBudget::new(1, 100_001, 784, 512));
  let msg = e.to_string();
  assert!(
    msg.contains("100001") && msg.contains("784") && msg.contains("512"),
    "{msg}"
  );
}

#[test]
fn embedding_dim_mismatch_display_shows_expected_then_got() {
  let msg = Error::EmbeddingDimMismatch(EmbeddingDimMismatch::new(384, 128)).to_string();
  assert_eq!(msg, "embedding dimension mismatch: expected 384, got 128");
}

#[test]
fn embedding_not_unit_norm_display_carries_the_deviation() {
  let msg = Error::EmbeddingNotUnitNorm(0.5).to_string();
  assert!(msg.contains("unit-norm"), "{msg}");
  assert!(msg.contains("0.5"), "{msg}");
}

#[test]
fn token_budget_errors_display_their_counts_and_caps() {
  // `TokenCount::new` takes (got, max); `WindowOverBudget::new` (window, max).
  let count = Error::TokenCount(TokenCount::new(600, 512)).to_string();
  assert_eq!(
    count,
    "tokenized input has 600 tokens, exceeding the fixed 512-token window"
  );
  let budget = Error::WindowOverBudget(WindowOverBudget::new(600, 512)).to_string();
  assert!(budget.contains("600") && budget.contains("512"), "{budget}");
  assert!(budget.contains("embed_long"), "{budget}");
  assert!(
    Error::TokenIdRange(u32::MAX)
      .to_string()
      .contains(&u32::MAX.to_string())
  );
}

#[test]
fn artifact_tokenizer_read_is_transparent_and_keeps_the_chain_at_depth_one() {
  // The path carries a SPACE, so a `Debug` rendering could not pass for the
  // `Display` one thiserror's `{path}` shorthand produces.
  let e = Error::ArtifactTokenizerRead(ArtifactTokenizerRead::new(
    std::path::PathBuf::from("/tmp/a b/tokenizer.json"),
    std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
  ));
  assert_eq!(
    e.to_string(),
    "failed to read the artifact tokenizer `/tmp/a b/tokenizer.json`: missing"
  );
  let mut depth = 0;
  let mut cur: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&e);
  while let Some(c) = cur {
    depth += 1;
    cur = std::error::Error::source(c);
  }
  assert_eq!(depth, 1, "the source chain must stay at depth 1");
}
