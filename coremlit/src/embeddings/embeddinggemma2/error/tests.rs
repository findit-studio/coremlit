use super::*;

/// The refusals name what they are about: the artifact, the function, the
/// directory and the command that stamps it.
#[test]
fn the_refusals_name_their_subject() {
  let cases: [(Error, &str); 11] = [
    (
      Error::Unstamped(Unstamped::new(Artifact::TokenTable, PathBuf::from("/m"))),
      "the stamp in `/m` lists nothing for `embeddings.bf16`: run `compile_artifacts` over the directory first",
    ),
    (
      Error::Load(LoadRefusal::new(
        Artifact::TextBundle,
        Some(Function::Text(TextFunction::Embed128)),
        LoadError::FunctionsUnsupported,
      )),
      "Core ML refused to load the EmbeddingGemma2Text bundle (function `embed_128`): loading one function of a model is unavailable on this OS (requires macOS 15)",
    ),
    (
      Error::NonFiniteOutput(NonFiniteOutput::new(
        Artifact::AudioBundle(AudioPrecision::Fp32),
        None,
        7,
      )),
      "the EmbeddingGemma2AudioFp32 bundle produced a non-finite value at index 7 of a row it was asked for",
    ),
    (
      Error::Contract(ContractRefusal::new(
        Artifact::VisionBundle,
        Some(Function::Vision(VisionBudget::Tokens140)),
        "patches".to_owned(),
        "F16 [1, 1260, 768]".to_owned(),
        "F16 [1, 630, 768]".to_owned(),
      )),
      "the EmbeddingGemma2Vision bundle (function `vision_140`) does not keep the kit's contract at feature `patches`: expected F16 [1, 1260, 768], got F16 [1, 630, 768]",
    ),
    (
      Error::TableIdentity(TableIdentity::new(
        Artifact::TokenTable,
        TableCell::token(506, 160),
        0xbf06,
        0x3f06,
      )),
      "`embeddings.bf16` differs from its stamp at row 506, column 160: the stamp recorded 0xbf06, the file holds 0x3f06",
    ),
    (
      Error::StaleStamp(StaleStamp::new(
        Artifact::PositionTable,
        3_145_728,
        3_145_726,
      )),
      "`position_embeddings.f16` has changed since it was stamped: the stamp recorded 3145728 bytes and the file has 3145726; run `compile_artifacts` over the directory again",
    ),
    (
      Error::AmbiguousArtifact(AmbiguousArtifact::new(
        Artifact::TextBundle,
        vec![
          PathBuf::from("EmbeddingGemma2Text.mlpackage"),
          PathBuf::from("EmbeddingGemma2Text.mlmodelc"),
        ],
      )),
      "the EmbeddingGemma2Text bundle has more than one source in the directory (`EmbeddingGemma2Text.mlpackage`, `EmbeddingGemma2Text.mlmodelc`): remove all but one; none is chosen",
    ),
    (
      Error::Allocation(Allocation::new(AllocationSite::GatherRows, 524_288)),
      "could not reserve 524288 bytes for gather_rows",
    ),
    (
      Error::Traversal(TraversalRefusal::new(
        TraversalDefect::BrokenLink,
        PathBuf::from("/m/tokenizer.json"),
      )),
      "`/m/tokenizer.json` is a broken link: the kit follows every link in a model directory, and this one leads to nothing",
    ),
    (
      Error::Traversal(TraversalRefusal::new(
        TraversalDefect::Cycle,
        PathBuf::from("/m/EmbeddingGemma2Text.mlpackage/Data/loop"),
      )),
      "`/m/EmbeddingGemma2Text.mlpackage/Data/loop` leads back into a directory that holds it: a directory cycle",
    ),
    (
      Error::Traversal(TraversalRefusal::new(
        TraversalDefect::TooManyEntries,
        PathBuf::from("/m/EmbeddingGemma2Audio.mlmodelc"),
      )),
      "`/m/EmbeddingGemma2Audio.mlmodelc` holds more than 1024 files and directories, the most one walked tree may hold",
    ),
  ];
  for (error, message) in cases {
    assert_eq!(error.to_string(), message);
  }
}

/// L14: an identity refusal names the part, the pinned label and the stamp's.
#[test]
fn an_identity_refusal_names_all_three_values() {
  let actual: ArtifactLabel = "sha256:9c1e00003b0a".parse().unwrap();
  let error = Error::ArtifactIdentity(ArtifactIdentity::new(
    Part::Audio,
    ArtifactLabel::FLUIDINFERENCE,
    actual,
  ));
  assert_eq!(
    error.to_string(),
    "the audio part's artifacts are sha256:9c1e00003b0a, but the deployment expects fluidinference@3f978dd"
  );
  match error {
    Error::ArtifactIdentity(identity) => {
      assert_eq!(identity.part(), Part::Audio);
      assert_eq!(identity.expected(), ArtifactLabel::FLUIDINFERENCE);
      assert_eq!(identity.actual(), actual);
    }
    other => panic!("expected ArtifactIdentity, got {other:?}"),
  }
}

/// The shared long-text refusals keep their variant and their message.
#[test]
fn the_long_text_refusals_keep_their_variant_and_message() {
  use crate::embeddings::long_text::Error as LongText;

  let cases = [
    LongText::WindowOverBudget(WindowOverBudget::new(513, 512)),
    LongText::InputTooLarge(InputTooLarge::new(10, 4)),
    LongText::ContentlessInputOverBudget(ContentlessInputOverBudget::new(0, 9, 600, 512)),
    LongText::Windowing(WinditError::TooManyWindows { got: 3, max: 2 }),
  ];
  for shared in cases {
    let message = shared.to_string();
    let converted = Error::from(shared);
    assert_eq!(converted.to_string(), message);
    assert!(matches!(
      converted,
      Error::WindowOverBudget(_)
        | Error::InputTooLarge(_)
        | Error::ContentlessInputOverBudget(_)
        | Error::Windowing(_)
    ));
  }
}

/// A source chain survives: the load refusal's source is Core ML's error.
#[test]
fn a_wrapped_error_keeps_its_source() {
  use std::error::Error as _;

  let error = Error::Load(LoadRefusal::new(
    Artifact::VisionBundle,
    None,
    LoadError::FunctionsUnsupported,
  ));
  let source = error.source().expect("the load error is the source");
  assert_eq!(
    source.to_string(),
    LoadError::FunctionsUnsupported.to_string()
  );
  assert_eq!(Artifact::Config.to_string(), "`config.json`");
  assert_eq!(
    TableCell::position(1, 0, 767).to_string(),
    "table 1, row 0, column 767"
  );
}

/// A refusal quotes at most 64 bytes of an artifact's own text, cut on a
/// character boundary and marked; a shorter text is quoted whole.
#[test]
fn a_quoted_excerpt_is_at_most_64_bytes() {
  assert_eq!(excerpt("<eos>"), "<eos>");
  let long = "a".repeat(100);
  assert_eq!(excerpt(&long), format!("{}…", "a".repeat(64)));
  // 63 ASCII bytes, then a two-byte character across the 64th byte.
  let straddling = format!("{}é{}", "b".repeat(63), "c".repeat(10));
  assert_eq!(excerpt(&straddling), format!("{}…", "b".repeat(63)));
  let value = serde_json::json!({ "content": "x".repeat(200) });
  let quoted = excerpt(&value);
  assert!(
    quoted.len() <= 64 + '…'.len_utf8() && quoted.ends_with('…'),
    "{quoted}"
  );
}

/// A refusal that quotes two texts shares its 64 bytes between them: two of
/// 40 bytes are cut to 32 each, and a short one is quoted whole and leaves the
/// rest to the other.
#[test]
fn two_quotations_share_one_refusals_64_bytes() {
  let (a, b) = excerpt_pair(&"a".repeat(40), &"b".repeat(40));
  assert_eq!(a, format!("{}…", "a".repeat(32)));
  assert_eq!(b, format!("{}…", "b".repeat(32)));
  let (a, b) = excerpt_pair("<eos>", &"b".repeat(100));
  assert_eq!(a, "<eos>");
  assert_eq!(b, format!("{}…", "b".repeat(64 - 5)));
  let (a, b) = excerpt_pair(&"a".repeat(100), "<bos>");
  assert_eq!(a, format!("{}…", "a".repeat(64 - 5)));
  assert_eq!(b, "<bos>");
}
