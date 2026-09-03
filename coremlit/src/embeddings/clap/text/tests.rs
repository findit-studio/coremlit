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
