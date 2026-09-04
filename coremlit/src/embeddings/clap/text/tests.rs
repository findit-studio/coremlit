use super::*;

// ── Options ────────────────────────────────────────────────────────────────

#[test]
fn options_default_equals_new() {
  assert_eq!(TextEncoderOptions::default(), TextEncoderOptions::new());
  assert_eq!(TextEncoderOptions::new().compute(), DEFAULT_TEXT_COMPUTE);
  // Measure-then-pin default: moved off `All` to `CpuAndGpu` by the #30 perf pass
  // (the tiny RoBERTa graph pays ANE dispatch overhead on `All`; `CpuAndGpu` is
  // ~43% faster warm and holds the placement parity floor — see the const's docs
  // and `benches/clap/encode.rs`).
  assert_eq!(DEFAULT_TEXT_COMPUTE, ComputeUnits::CpuAndGpu);
}

#[test]
fn options_with_and_set_compute() {
  let opts = TextEncoderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
  let mut opts = TextEncoderOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = TextEncoderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_neural_engine"), "serialized: {json}");
  let back: TextEncoderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

// ── Tokenizer identity gate (hermetic; the real tokenizer seam) ─────────────

/// SHA-256 of the bundled tokenizer must equal the identical artifact textclap
/// pins (`textclap/models/MODELS.sha256`) — byte-identity is the foundation of
/// token-id identity. Any drift in `assets/tokenizer.json` fails here.
#[test]
fn bundled_tokenizer_sha_matches_textclap_pin() {
  use sha2::{Digest, Sha256};
  let sha: String = Sha256::digest(crate::embeddings::clap::BUNDLED_TOKENIZER)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  assert_eq!(
    sha, "dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5",
    "bundled tokenizer.json diverged from textclap's pinned Xenova artifact"
  );
}

/// Encode `text` through clapkit's ACTUAL configured tokenizer seam (the same
/// path [`TextEncoder::token_ids`] uses), hermetically (no model).
fn ids(text: &str) -> Vec<u32> {
  let tok = configured_tokenizer_from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER)
    .expect("configure tokenizer");
  tok.encode(text, true).expect("encode").get_ids().to_vec()
}

/// Token-id EXACT-equality over a pinned corpus (English, CJK, emoji). These ids
/// are identity-comparable to textclap: the tokenizer artifact is byte-identical
/// (SHA above) and the truncation config matches textclap's
/// `force_max_length_truncation`. The live cross-check against the textclap crate
/// itself is `coremlit-parity`'s `tests/clap/tokenizer_identity_textclap.rs`
/// (feature `clap-oracle`).
///
/// Measure-then-pin: mutate the tokenizer bytes or the encode call and these
/// exact sequences change.
#[test]
fn token_ids_match_pinned_golden() {
  // <s>=0, </s>=2 (RoBERTa specials) bracket every sequence.
  let cases: &[(&str, &[u32])] = &[
    ("a dog barking", &[0, 102, 2335, 35828, 2]),
    (
      "一只猫在喵喵叫",
      &[
        0, 48105, 45262, 10278, 36714, 14285, 4958, 46537, 11423, 42393, 25448, 8906, 42393, 25448,
        8906, 45262, 4958, 2,
      ],
    ),
    (
      "a cat 🐱 meowing 😺",
      &[0, 102, 4758, 8103, 16948, 15389, 162, 6932, 17841, 3070, 2],
    ),
  ];
  for (text, expected) in cases {
    let got = ids(text);
    assert_eq!(&got, expected, "token-id drift for {text:?}");
  }
}

/// Truncation identity — the DIRECTION, not just the length, is gated.
///
/// A *non-repetitive* input longer than the 512-token window (ascending integers,
/// every token distinct) truncates to EXACTLY [`TEXT_MAX_TOKENS`] without
/// overflowing the RoBERTa position table, and — because clapkit configures
/// `TruncationDirection::Right` (matching textclap's `LongestFirst@512`) — the
/// kept interior is the untruncated encoding's PREFIX (the first 510 content
/// tokens). The old gate used repetitive text and checked only length + the two
/// specials, so a `Right → Left` flip (which keeps the SUFFIX instead) stayed
/// green; here the full 510-id interior is asserted, so the flip trips it.
#[test]
fn long_input_truncation_keeps_the_right_directional_prefix() {
  // Non-repetitive, comfortably over one window: "1 2 3 … 1000", all distinct.
  let long: String = (1..=1000)
    .map(|n| n.to_string())
    .collect::<Vec<_>>()
    .join(" ");

  // clapkit's real configured seam (LongestFirst@512, Right).
  let truncated = ids(&long);
  assert_eq!(
    truncated.len(),
    TEXT_MAX_TOKENS,
    "truncation must cap ids at the window"
  );
  assert_eq!(truncated[0], 0, "leading <s> kept");
  assert_eq!(truncated[TEXT_MAX_TOKENS - 1], 2, "trailing </s> kept");

  // Untruncated reference: the SAME tokenizer bytes with truncation OFF.
  let full = tokenizers::Tokenizer::from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER)
    .expect("load tokenizer")
    .encode(long.as_str(), true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert!(
    full.len() > TEXT_MAX_TOKENS,
    "reference must actually overflow the window (got {})",
    full.len()
  );
  assert_eq!(full[0], 0, "reference leading <s>");

  // RIGHT truncation ⇒ the 510 interior ids equal the untruncated PREFIX — the
  // FULL-interior assertion the byte-only / repetitive gates lacked. Under
  // `Left` the interior would be the untruncated SUFFIX, which (distinct tokens)
  // differs ⇒ red.
  assert_eq!(
    &truncated[1..TEXT_MAX_TOKENS - 1],
    &full[1..TEXT_MAX_TOKENS - 1],
    "Right-truncation interior must equal the untruncated first-510 content tokens"
  );

  // Measure-then-pin: the exact 512-id sequence, nailed to a SHA-256 constant so
  // the whole interior is pinned absolutely (not only relative to the reference).
  // Any tokenizer-artifact or truncation-config drift changes it.
  use sha2::{Digest, Sha256};
  let mut hasher = Sha256::new();
  for id in &truncated {
    hasher.update(id.to_le_bytes());
  }
  let sha: String = hasher
    .finalize()
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  assert_eq!(
    sha, "87b94fa2c2c74ccc9ee354f15d1b865d960f4c3cef19030159fdc8364dbf38f0",
    "truncated 512-id sequence drifted (tokenizer artifact or truncation config changed)"
  );
}

// ── Special-token overhead vs the fixed window ─────────────────────────────

/// A minimal WordLevel tokenizer: the overhead guard is about the
/// post-processor and the window, so the model underneath only has to tokenize.
const OVERHEAD_TINY_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"a":1,"b":2},"unk_token":"<pad>"}}"#;

/// Serialize [`OVERHEAD_TINY_TOKENIZER`] with a `TemplateProcessing`
/// post-processor whose single-sequence template adds exactly `added` special
/// tokens — the shape a caller can hand `from_files`, and the number the
/// tokenizers crate subtracts from the truncation window without checking. One
/// `SpecialToken` carrying `added` ids is used rather than `added` template
/// pieces because `TemplateProcessing::added_tokens` sums `ids.len()` per piece,
/// so both count the same and only this one scales to a 512-token window.
fn tokenizer_bytes_with_special_overhead(added: usize) -> Vec<u8> {
  use tokenizers::processors::template::{SpecialToken, TemplateProcessing};

  let mut tokenizer =
    Tokenizer::from_bytes(OVERHEAD_TINY_TOKENIZER.as_bytes()).expect("load the tiny tokenizer");
  let special = SpecialToken::new(
    "<sp>".to_string(),
    vec![0u32; added],
    vec!["<pad>".to_string(); added],
  )
  .expect("ids and tokens are the same length");
  let template = TemplateProcessing::builder()
    .try_single("<sp> $A")
    .expect("single template")
    .try_pair("<sp> $A $B")
    .expect("pair template")
    .special_tokens(vec![special])
    .build()
    .expect("build the template post-processor");
  tokenizer.with_post_processor(Some(template));
  tokenizer
    .to_string(false)
    .expect("serialize the tokenizer")
    .into_bytes()
}

/// The helper actually installs the overhead it claims — otherwise every case
/// below would pass vacuously against a post-processor that adds nothing.
#[test]
fn overhead_fixture_installs_the_claimed_special_token_count() {
  use tokenizers::PostProcessor;
  for added in [
    1usize,
    TEXT_MAX_TOKENS - 1,
    TEXT_MAX_TOKENS,
    TEXT_MAX_TOKENS + 1,
  ] {
    let bytes = tokenizer_bytes_with_special_overhead(added);
    let tok = Tokenizer::from_bytes(&bytes).expect("reload the fixture");
    let post = tok.get_post_processor().expect("the fixture has one");
    assert_eq!(post.added_tokens(false), added, "single-sequence overhead");
  }
}

/// `added > TEXT_MAX_TOKENS` is the tokenizers crate's unchecked
/// `max_length - added_tokens` subtraction: it PANICS with "attempt to subtract
/// with overflow" under overflow checks and wraps to a near-`usize::MAX` window
/// in release. The door refuses the tokenizer first, naming both numbers.
#[test]
fn configure_tokenizer_refuses_overhead_over_the_window() {
  let bytes = tokenizer_bytes_with_special_overhead(TEXT_MAX_TOKENS + 1);
  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::SpecialTokenOverhead(overhead)) => {
      assert_eq!(overhead.added(), TEXT_MAX_TOKENS + 1);
      assert_eq!(overhead.window(), TEXT_MAX_TOKENS);
    }
    other => panic!("expected SpecialTokenOverhead, got {other:?}"),
  }
}

/// `added == TEXT_MAX_TOKENS` does not overflow — and is refused anyway, because
/// the effective text window is then zero and every encoding would be the
/// special tokens alone.
#[test]
fn configure_tokenizer_refuses_overhead_equal_to_the_window() {
  let bytes = tokenizer_bytes_with_special_overhead(TEXT_MAX_TOKENS);
  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::SpecialTokenOverhead(overhead)) => {
      assert_eq!(overhead.added(), TEXT_MAX_TOKENS);
      assert_eq!(overhead.window(), TEXT_MAX_TOKENS);
    }
    other => panic!("expected SpecialTokenOverhead, got {other:?}"),
  }
}

/// One token of room is enough: `added < TEXT_MAX_TOKENS` configures, and the
/// window then holds the specials plus at least one real token.
#[test]
fn configure_tokenizer_accepts_overhead_below_the_window() {
  let bytes = tokenizer_bytes_with_special_overhead(TEXT_MAX_TOKENS - 1);
  let tok = configured_tokenizer_from_bytes(&bytes).expect("511 specials fit a 512-token window");
  let ids = tok
    .encode("a b a b", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(ids.len(), TEXT_MAX_TOKENS, "specials plus one real token");
  assert_eq!(ids[TEXT_MAX_TOKENS - 1], 1, "the real token survives (`a`)");
}

/// The guard does not fire on the tokenizer this crate actually ships: the
/// bundled Xenova RoBERTa post-processor adds two specials, far inside 512.
#[test]
fn the_bundled_tokenizer_has_room_to_spare() {
  use tokenizers::PostProcessor;
  let tok = Tokenizer::from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER).expect("load");
  let added = tok
    .get_post_processor()
    .map_or(0, |post| post.added_tokens(false));
  assert_eq!(added, 2, "<s> … </s>");
  assert!(added < TEXT_MAX_TOKENS);
  configured_tokenizer_from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER)
    .expect("the shipped tokenizer configures");
}

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself — its feature names, its
// element type, its geometry and its state clause — against descriptions built
// with the same fixture machinery, so a mis-stated contract is caught here and
// a mis-implemented checker is caught there.

use crate::{AxisRange, FeatureInfo, ModelDescription, model::RawShapeConstraint};

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  multi_array(name, shape, dtype, false, 2, vec![shape.to_vec()], shape)
}

/// One multi-array feature, spelled out: the constraint's raw type code, its
/// enumerated shapes, and the axes its per-axis ranges pin.
fn multi_array(
  name: &str,
  shape: &[usize],
  dtype: DataType,
  optional: bool,
  raw_type: isize,
  enumerated: Vec<Vec<usize>>,
  pinned: &[usize],
) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    optional,
    Some(RawShapeConstraint::new(
      raw_type,
      enumerated,
      pinned.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// The staged RoBERTa bundle's description, as the CoreML probe reads it back:
/// `attention_mask` and `input_ids` both `[1, 512]` i32 in, `text_embeds
/// [1, 512]` f32 out, no state — identical across the fp16 and int8 tiers.
fn roberta_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed(names::INPUT_IDS, &[1, TEXT_MAX_TOKENS], DataType::I32),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into the CLAP
/// error vocabulary — exactly what `TextEncoder::from_parts` does after
/// `Model::load`.
fn check(description: &ModelDescription) -> Result<()> {
  crate::model::contract::check_load_contract(description, &text_contract())
    .map_err(contract_violation)
}

/// The contract states exactly the geometry the conversion emits.
#[test]
fn the_contract_accepts_the_converted_geometry() {
  assert!(check(&roberta_description()).is_ok());
}

/// **The flexible-shape refusal**, which matters here twice over: the window is
/// what this door PADS to, so a graph that would also accept a shorter sequence
/// is one whose mask means something else. `shape()` reports the DEFAULT shape
/// of a `RangeDims` input, so such a graph declares this contract's exact
/// numbers and reports `(d, 1)` on every axis; only the whole-feature verdict
/// separates the two.
#[test]
fn the_contract_refuses_a_flexible_window_declaring_its_exact_numbers() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      multi_array(
        names::INPUT_IDS,
        &[1, TEXT_MAX_TOKENS],
        DataType::I32,
        false,
        3,
        Vec::new(),
        &[1, TEXT_MAX_TOKENS],
      ),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::INPUT_IDS),
    "{err}"
  );
}

/// The token window is int32, not the float the mask's `1.0`/`0.0` spelling
/// would suggest — a recipe that exported either input as f32 would build a
/// tensor this door never writes.
#[test]
fn the_contract_refuses_a_float_token_window() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::F32),
      fixed(names::INPUT_IDS, &[1, TEXT_MAX_TOKENS], DataType::I32),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::ATTENTION_MASK),
    "{err}"
  );
}

/// A shorter window is not this graph: this door right-pads to exactly
/// [`TEXT_MAX_TOKENS`], so a `[1, 256]` export would be handed twice the tokens
/// it declares.
#[test]
fn the_contract_refuses_a_shorter_window() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed(names::INPUT_IDS, &[1, 256], DataType::I32),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::INPUT_IDS),
    "{err}"
  );
}

/// **A graph carrying this door's two inputs plus another REQUIRED one** clears
/// every per-feature clause and then fails on every prediction, because
/// [`TextEncoder::embed`] supplies `input_ids` and `attention_mask` and nothing
/// else.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed(names::INPUT_IDS, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed("token_type_ids", &[1, TEXT_MAX_TOKENS], DataType::I32),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableInput(name)) if name == "token_type_ids"),
    "{:?}",
    check(&description)
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail.
#[test]
fn the_contract_accepts_an_extra_optional_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed(names::INPUT_IDS, &[1, TEXT_MAX_TOKENS], DataType::I32),
      multi_array(
        "token_type_ids",
        &[1, TEXT_MAX_TOKENS],
        DataType::I32,
        true,
        2,
        vec![vec![1, TEXT_MAX_TOKENS]],
        &[1, TEXT_MAX_TOKENS],
      ),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(check(&description).is_ok());
}

/// An output the door READS that the graph may leave out: every geometry
/// clause passes and the prediction is still free to omit it.
#[test]
fn the_contract_refuses_an_optional_embeds_output() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed(names::INPUT_IDS, &[1, TEXT_MAX_TOKENS], DataType::I32),
    ],
    vec![multi_array(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
      true,
      2,
      vec![vec![1, EMBEDDING_DIM]],
      &[1, EMBEDDING_DIM],
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::TEXT_EMBEDS),
    "{err}"
  );
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly this door's three features plus a state clears every per-feature
/// clause AND the input set — and only then meets [`TextEncoder::embed`], which
/// predicts through the STATELESS API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::ATTENTION_MASK, &[1, TEXT_MAX_TOKENS], DataType::I32),
      fixed(names::INPUT_IDS, &[1, TEXT_MAX_TOKENS], DataType::I32),
    ],
    vec![fixed(
      names::TEXT_EMBEDS,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableState(name)) if name == "kv_cache")
  );
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **This door's `Checked::new` call site, pinned on a REAL model, in every
/// `cargo test`.**
///
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is COMMITTED, so
/// unlike everything else in this repository that loads a model this needs no
/// staged artifact and carries no `#[ignore]`. Silero is a real, fixed-shape,
/// six-feature CoreML graph that is simply not this door's model — the exact
/// shape of a mis-pointed `model_path`.
#[test]
fn the_text_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; \
     looked for {}",
    bundle.display()
  );

  let model = Model::load(&bundle, ComputeUnits::CpuOnly).expect("the committed bundle loads");
  assert!(
    model.description().input(names::INPUT_IDS).is_none(),
    "silero declares no `input_ids`, which is what makes it this gate's model"
  );

  let violation = Checked::new(model, &text_contract())
    .expect_err("silero does not satisfy the CLAP text contract");
  assert!(
    matches!(&violation, crate::model::contract::ContractViolation::Missing(m)
      if m.feature() == names::INPUT_IDS),
    "expected `input_ids` missing, got {violation}"
  );
}

use crate::embeddings::clap::error::PostProcessorTemplate;

// ── The caller tokenizer's own padding policy ───────────────────────────────

/// A `Fixed(513)` padding policy on a caller-supplied tokenizer. The tokenizer
/// pads AFTER truncating, so the door's `LongestFirst` truncation at 512 does
/// not contain it: measured against the unconfigured tokenizer, `encode` returns
/// 513 ids for any input.
const PADDING_PAST_THE_WINDOW: &str = r#"{"strategy":{"Fixed":513},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"<pad>"}"#;

/// [`OVERHEAD_TINY_TOKENIZER`] with a padding policy spliced in — the shape a
/// caller can hand `from_files`.
fn tokenizer_bytes_with_padding(padding: &str) -> Vec<u8> {
  OVERHEAD_TINY_TOKENIZER
    .replace(r#""padding":null"#, &format!(r#""padding":{padding}"#))
    .into_bytes()
}

/// Non-vacuity: the fixture really does over-pad when the door does NOT disable
/// padding, so the test below is not asserting against an inert policy.
#[test]
fn the_padding_fixture_really_over_pads() {
  let bytes = tokenizer_bytes_with_padding(PADDING_PAST_THE_WINDOW);
  let tok = Tokenizer::from_bytes(&bytes).expect("parse the fixture");
  assert_eq!(
    tok.encode("a b", true).expect("encode").get_ids().len(),
    513,
    "the raw fixture pads past the window"
  );
}

/// The door owns the padding policy, not the caller's file. Without
/// `with_padding(None)` the configured tokenizer emits 513 ids for a 512-token
/// window — which `build_window` would then have to refuse, and which `embed`
/// used to write straight past a `[i32; 512]`.
#[test]
fn configure_tokenizer_disables_the_tokenizers_own_padding() {
  let bytes = tokenizer_bytes_with_padding(PADDING_PAST_THE_WINDOW);
  let tok = configured_tokenizer_from_bytes(&bytes).expect("configure");
  assert!(tok.get_padding().is_none(), "padding must be disabled");
  let ids = tok.encode("a b", true).expect("encode").get_ids().to_vec();
  assert_eq!(ids.len(), 2, "no pads, and truncation caps at the window");
}

/// The same holds for `BatchLongest` with `pad_to_multiple_of`, which reaches
/// past the window without naming a length.
#[test]
fn configure_tokenizer_disables_a_pad_to_multiple_of_policy() {
  let bytes = tokenizer_bytes_with_padding(
    r#"{"strategy":"BatchLongest","direction":"Right","pad_to_multiple_of":513,"pad_id":0,"pad_type_id":0,"pad_token":"<pad>"}"#,
  );
  let tok = configured_tokenizer_from_bytes(&bytes).expect("configure");
  assert!(tok.get_padding().is_none(), "padding must be disabled");
  assert_eq!(
    tok.encode("a b", true).expect("encode").get_ids().len(),
    2,
    "no pads"
  );
}

// ── The fixed window is built, never asserted about ─────────────────────────

/// One id too many is a typed refusal, not the out-of-bounds write a fixed-size
/// window takes in release (`index out of bounds: the len is 512 but the index
/// is 512`) and a `debug_assert!` only documents under test.
#[test]
fn build_window_refuses_more_ids_than_the_window_holds() {
  match build_window(&vec![1u32; TEXT_MAX_TOKENS + 1], 0) {
    Err(Error::TokenCount(count)) => {
      assert_eq!(count.got(), TEXT_MAX_TOKENS + 1);
      assert_eq!(count.max(), TEXT_MAX_TOKENS);
    }
    other => panic!("expected TokenCount, got {other:?}"),
  }
}

/// A full window is not one too many: the boundary accepts.
#[test]
fn build_window_accepts_exactly_the_window() {
  let (ids, mask) = build_window(&vec![7u32; TEXT_MAX_TOKENS], 1).expect("a full window fits");
  assert!(
    ids.iter().all(|&id| id == 7),
    "every position is a real token"
  );
  assert!(
    mask.iter().all(|&m| m == 1),
    "and every position is attended"
  );
}

/// The real tokens occupy the prefix, the pad id fills the rest, and the mask
/// separates them — the geometry `embed` feeds CoreML.
#[test]
fn build_window_right_pads_and_masks() {
  let (ids, mask) = build_window(&[5, 6], 1).expect("two ids fit");
  assert_eq!(&ids[..3], &[5, 6, 1]);
  assert_eq!(&mask[..3], &[1, 1, 0]);
  assert_eq!(ids[TEXT_MAX_TOKENS - 1], 1, "the tail is padding");
  assert_eq!(mask[TEXT_MAX_TOKENS - 1], 0, "and it is not attended");
}

// ── Token ids are converted, never cast ─────────────────────────────────────

/// The first id above `i32::MAX`. `as i32` maps it to `i32::MIN`, so CoreML
/// would gather `input_ids` at a NEGATIVE index.
const FIRST_ID_PAST_I32: u32 = 2_147_483_648;

#[test]
fn build_window_refuses_a_token_id_past_int32() {
  match build_window(&[FIRST_ID_PAST_I32], 0) {
    Err(Error::TokenIdRange(id)) => assert_eq!(id, FIRST_ID_PAST_I32),
    other => panic!("expected TokenIdRange, got {other:?}"),
  }
  assert_eq!(
    FIRST_ID_PAST_I32 as i32,
    i32::MIN,
    "the cast this replaced wrapped to a negative id"
  );
}

/// `i32::MAX` itself converts: the boundary is exclusive.
#[test]
fn build_window_accepts_the_largest_int32_id() {
  let (ids, _) = build_window(&[i32::MAX as u32], 0).expect("i32::MAX fits int32");
  assert_eq!(ids[0], i32::MAX);
}

/// A tokenizer whose `<pad>` is out of `int32` range is refused at construction,
/// through the same resolution `from_parts` uses — no model needed.
#[test]
fn resolve_pad_id_refuses_a_pad_token_past_int32() {
  let bytes = OVERHEAD_TINY_TOKENIZER
    .replace(r#""<pad>":0"#, &format!(r#""<pad>":{FIRST_ID_PAST_I32}"#))
    .into_bytes();
  let tok = Tokenizer::from_bytes(&bytes).expect("parse");
  match resolve_pad_id(&tok) {
    Err(Error::TokenIdRange(id)) => assert_eq!(id, FIRST_ID_PAST_I32),
    other => panic!("expected TokenIdRange, got {other:?}"),
  }
}

/// The ordinary paths still resolve: the tokenizer's own `<pad>`, and RoBERTa's
/// conventional `1` when it has none.
#[test]
fn resolve_pad_id_reads_the_vocabulary_then_falls_back() {
  let tok = Tokenizer::from_bytes(OVERHEAD_TINY_TOKENIZER.as_bytes()).expect("parse");
  assert_eq!(
    resolve_pad_id(&tok).expect("in range"),
    0,
    "the vocab's own"
  );

  let bytes = OVERHEAD_TINY_TOKENIZER
    .replace(r#""<pad>":0,"#, "")
    .replace(r#""unk_token":"<pad>""#, r#""unk_token":"a""#)
    .into_bytes();
  let tok = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    resolve_pad_id(&tok).expect("no <pad> is not an error"),
    FALLBACK_PAD_ID
  );

  let bundled = Tokenizer::from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER).expect("bundled");
  assert_eq!(
    resolve_pad_id(&bundled).expect("the bundled pad id is in range"),
    FALLBACK_PAD_ID,
    "RoBERTa's <pad> is 1, which is also the fallback"
  );
}

// ── A parseable-but-inconsistent post-processor template ────────────────────

/// The three defective single templates, as a `tokenizer.json` post-processor
/// writes them. Each PARSES — the tokenizers deserializer skips its builder's
/// `validate` — and each is a wrong answer or a panic at the first `encode`.
const TEMPLATE_UNDECLARED: &str = r#"{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{}}"#;
const TEMPLATE_PAIR_IN_SINGLE: &str = r#"{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{}}"#;
const TEMPLATE_NO_INPUT: &str = r#"{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{"<s>":{"id":"<s>","ids":[0],"tokens":["<pad>"]}}}"#;

/// Two individually SOUND templates in a `Sequence`: neither breaks a rule of
/// its own, and both passed the guard before it simulated the encoding count.
/// The first applies a three-piece single template, and `apply_template` emits
/// one encoding per piece, so the second is handed THREE — a count whose arm in
/// `process_encodings` is a `todo!()`. Measured before the count was simulated:
/// `not yet implemented` at `processors/template.rs:681`.
const SEQUENCE_FEEDING_THREE_ENCODINGS: &str = r#"{"type":"Sequence","processors":[{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<s>","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{"<s>":{"id":"<s>","ids":[0],"tokens":["<pad>"]}}},{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{}}]}"#;

/// A single template that places the input sequence TWICE. Individually sound
/// by every rule the round-3 guard had, and its `added_tokens(false)` is ZERO —
/// a `Sequence` piece counts as no overhead however often it appears — so the
/// truncation and the overhead guard are both sized for one copy of a text the
/// post-processor returns two copies of.
const TEMPLATE_PLACING_THE_TEXT_TWICE: &str = r#"{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{}}"#;

/// A chain that reaches a template at TWO encodings: the first member's
/// two-piece single template emits two, so the second applies its `pair`
/// template — which here places no sequence at all, so the text is erased.
const SEQUENCE_REACHING_A_PAIR_TEMPLATE: &str = r#"{"type":"Sequence","processors":[{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<s>","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{"<s>":{"id":"<s>","ids":[0],"tokens":["<pad>"]}}},{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{}}]}"#;

/// [`OVERHEAD_TINY_TOKENIZER`] with `post_processor` spliced in.
fn tokenizer_bytes_with_post_processor(post: &str) -> Vec<u8> {
  OVERHEAD_TINY_TOKENIZER
    .replace(
      r#""post_processor":null"#,
      &format!(r#""post_processor":{post}"#),
    )
    .into_bytes()
}

/// Each defective template is refused at CONFIGURATION, with the reason named —
/// so no `encode` is ever reached. Measured before the guard: the first two
/// panicked inside the dependency (`no entry found for key` at
/// `processors/template.rs:563`, `index out of bounds: the len is 1 but the
/// index is 1` at `:556`) and the third returned `[0]` for every input.
#[test]
fn configure_tokenizer_refuses_an_inconsistent_single_template() {
  for (post, expected) in [
    (
      TEMPLATE_UNDECLARED,
      PostProcessorTemplate::UndeclaredSpecialToken("<s>".to_string()),
    ),
    (
      TEMPLATE_PAIR_IN_SINGLE,
      PostProcessorTemplate::PairSequenceInSingleTemplate,
    ),
    (
      TEMPLATE_NO_INPUT,
      PostProcessorTemplate::NoInputSequenceInSingleTemplate,
    ),
  ] {
    let bytes = tokenizer_bytes_with_post_processor(post);
    match configured_tokenizer_from_bytes(&bytes) {
      Err(Error::PostProcessorTemplate(why)) => assert_eq!(why, expected),
      other => panic!("expected PostProcessorTemplate({expected:?}), got {other:?}"),
    }
  }
}

/// A `Sequence` post-processor applies each member, so wrapping a defective
/// template in one must not smuggle it past the guard.
#[test]
fn configure_tokenizer_refuses_a_defective_template_inside_a_sequence() {
  let post = format!(r#"{{"type":"Sequence","processors":[{TEMPLATE_UNDECLARED}]}}"#);
  match configured_tokenizer_from_bytes(&tokenizer_bytes_with_post_processor(&post)) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UndeclaredSpecialToken(id))) => {
      assert_eq!(id, "<s>");
    }
    other => panic!("expected UndeclaredSpecialToken, got {other:?}"),
  }
}

/// The guard must not fire on the tokenizer this crate actually ships: CLAP's
/// bundled Xenova tokenizer carries a `RobertaProcessing`, which has no template
/// at all.
#[test]
fn the_bundled_tokenizer_carries_a_template_free_post_processor() {
  let tok = configured_tokenizer_from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER)
    .expect("the bundled tokenizer configures");
  let post = tok.get_post_processor().expect("the bundled one has one");
  let rendered = serde_json::to_value(post).expect("serialize");
  assert_eq!(
    rendered.get("type").and_then(serde_json::Value::as_str),
    Some("RobertaProcessing"),
  );
}

/// The overhead guard cannot substitute for the structural one: an undeclared
/// `SpecialToken` id contributes ZERO to `added_tokens(false)`, so a template
/// that fills the whole window with undeclared specials reads as NO overhead and
/// sails past the overhead guard. Only the structural check sees it.
#[test]
fn the_overhead_reading_is_blind_to_undeclared_special_tokens() {
  let single: Vec<String> = (0..TEXT_MAX_TOKENS)
    .map(|i| format!(r#"{{"SpecialToken":{{"id":"<undeclared{i}>","type_id":0}}}}"#))
    .collect();
  let post = format!(
    r#"{{"type":"TemplateProcessing","single":[{}],"pair":[{{"Sequence":{{"id":"A","type_id":0}}}}],"special_tokens":{{}}}}"#,
    single.join(",")
  );
  let bytes = tokenizer_bytes_with_post_processor(&post);

  let raw = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    tokenizers::PostProcessor::added_tokens(raw.get_post_processor().expect("has one"), false),
    0,
    "the overhead reading is blind to undeclared ids"
  );

  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UndeclaredSpecialToken(id))) => {
      assert_eq!(id, "<undeclared0>");
    }
    other => panic!("expected UndeclaredSpecialToken, got {other:?}"),
  }
}

/// …and where BOTH guards would refuse, the structural one is the diagnostic
/// that wins. The overhead here is real and declared — `count_added` reports the
/// full window, so [`SpecialTokenOverhead`] would refuse this tokenizer on its
/// own — which is what makes this a falsifier for the ORDER rather than for the
/// presence of either check. A count derived from a malformed template is not a
/// fact about the tokenizer, so the malformation is what gets named.
#[test]
fn a_structural_defect_outranks_an_overhead_that_would_also_refuse() {
  let bytes = tokenizer_bytes_with_overhead_and_a_pair_sequence(TEXT_MAX_TOKENS);
  let raw = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    tokenizers::PostProcessor::added_tokens(raw.get_post_processor().expect("has one"), false),
    TEXT_MAX_TOKENS,
    "the overhead guard would refuse this tokenizer on its own"
  );
  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::PairSequenceInSingleTemplate)) => {}
    other => panic!("expected the structural refusal to be the one reported, got {other:?}"),
  }
}

/// [`tokenizer_bytes_with_special_overhead`] with a `$B` piece appended to the
/// SINGLE template: a tokenizer that breaks the structural rule AND carries an
/// overhead that fills the window. The tokenizers BUILDER refuses `$B` in a
/// single template, so the piece is spliced into the serialized JSON — which is
/// exactly how a hand-written `tokenizer.json` gets one past the deserializer.
fn tokenizer_bytes_with_overhead_and_a_pair_sequence(added: usize) -> Vec<u8> {
  let mut json: serde_json::Value =
    serde_json::from_slice(&tokenizer_bytes_with_special_overhead(added))
      .expect("the fixture serializes as JSON");
  json["post_processor"]["single"]
    .as_array_mut()
    .expect("the single template serializes as an array")
    .push(serde_json::json!({"Sequence": {"id": "B", "type_id": 1}}));
  serde_json::to_vec(&json).expect("serialize the spliced tokenizer")
}

/// The cardinality rule reaches this door too: a chain of templates that are
/// each individually sound can still hand a later one a number of encodings the
/// dependency has no template for, and that is a panic at the first `encode`.
#[test]
fn configure_tokenizer_refuses_a_chain_that_feeds_a_template_an_unsupported_count() {
  let bytes = tokenizer_bytes_with_post_processor(SEQUENCE_FEEDING_THREE_ENCODINGS);
  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UnsupportedEncodingCount(n))) => {
      assert_eq!(n, 3, "the count the second template would have received");
    }
    other => panic!("expected UnsupportedEncodingCount(3), got {other:?}"),
  }
}

/// The placement rule reaches this door too, and it is the round-4 MEDIUM
/// reproduced end to end. With the guard out of the way, the configured
/// tokenizer advertises no overhead, truncates to the full 512-token window and
/// then hands `build_window` twice that many ids — so ordinary text longer than
/// half the window fails at `embed` with a typed `TokenCount`, although
/// construction succeeded. The guard refuses the tokenizer instead, naming the
/// number of placements.
#[test]
fn configure_tokenizer_refuses_a_template_that_places_the_text_twice() {
  let bytes = tokenizer_bytes_with_post_processor(TEMPLATE_PLACING_THE_TEXT_TWICE);

  // What the refusal buys, measured with the guard bypassed.
  let mut raw = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    tokenizers::PostProcessor::added_tokens(raw.get_post_processor().expect("has one"), false),
    0,
    "a repeated `$A` reads as no overhead at all"
  );
  raw
    .with_truncation(Some(TruncationParams {
      max_length: TEXT_MAX_TOKENS,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .expect("zero overhead does not overflow");
  raw.with_padding(None);
  let text = "a b ".repeat(300);
  let ids = raw
    .encode(text.as_str(), true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(ids.len(), 1024, "512 truncated tokens, placed twice");
  match build_window(&ids, 1) {
    Err(Error::TokenCount(count)) => {
      assert_eq!(count.got(), 1024);
      assert_eq!(count.max(), TEXT_MAX_TOKENS);
    }
    other => panic!("expected the TokenCount backstop to fire, got {other:?}"),
  }

  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::RepeatedInputSequence(n))) => {
      assert_eq!(n, 2, "the number of `$A` placements");
    }
    other => panic!("expected RepeatedInputSequence(2), got {other:?}"),
  }
}

/// …and a chain that reaches a template at TWO encodings is refused with the
/// count, because the dependency then applies the `pair` template while the
/// truncation stays sized on `added_single`. Measured with the guard bypassed:
/// this chain encodes every input to nothing at all.
#[test]
fn configure_tokenizer_refuses_a_chain_that_reaches_a_pair_template() {
  let bytes = tokenizer_bytes_with_post_processor(SEQUENCE_REACHING_A_PAIR_TEMPLATE);

  let raw = Tokenizer::from_bytes(&bytes).expect("parse");
  for text in ["a b", "a b a b a b"] {
    assert!(
      raw.encode(text, true).expect("encode").get_ids().is_empty(),
      "the pair template places no sequence, so the text is gone"
    );
  }

  match configured_tokenizer_from_bytes(&bytes) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UnsupportedEncodingCount(n))) => {
      assert_eq!(n, 2, "the count the second template would have received");
    }
    other => panic!("expected UnsupportedEncodingCount(2), got {other:?}"),
  }
}
