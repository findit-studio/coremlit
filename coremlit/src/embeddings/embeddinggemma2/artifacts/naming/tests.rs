use super::*;

#[test]
fn a_compiled_bundle_name_round_trips_to_its_package() {
  let name = compiled_bundle("EmbeddingGemma2Text", "7b25760cb168");
  assert_eq!(name, "EmbeddingGemma2Text.7b25760cb168.mlmodelc");
  assert_eq!(
    parse_compiled_bundle(&name),
    Some(("EmbeddingGemma2Text", "7b25760cb168"))
  );
  assert_eq!(
    source_package("EmbeddingGemma2Text"),
    "EmbeddingGemma2Text.mlpackage"
  );
}

#[test]
fn a_name_that_is_not_a_compiled_bundle_does_not_parse() {
  for name in [
    "EmbeddingGemma2Text.mlmodelc",
    "EmbeddingGemma2Text.7b25760cb16.mlmodelc",
    "EmbeddingGemma2Text.7B25760CB168.mlmodelc",
    "EmbeddingGemma2Text.7b25760cb168.mlpackage",
    ".7b25760cb168.mlmodelc",
    "tokenizer.v1-4d777ef5bdc1.gemmabpe",
  ] {
    assert_eq!(parse_compiled_bundle(name), None, "{name}");
  }
}
