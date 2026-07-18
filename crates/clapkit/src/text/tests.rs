use super::*;

// ── Options ────────────────────────────────────────────────────────────────

#[test]
fn options_default_equals_new() {
  assert_eq!(TextEncoderOptions::default(), TextEncoderOptions::new());
  assert_eq!(TextEncoderOptions::new().compute(), DEFAULT_TEXT_COMPUTE);
  assert_eq!(DEFAULT_TEXT_COMPUTE, ComputeUnits::All);
}

#[test]
fn options_with_and_set_compute() {
  let opts = TextEncoderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
  let mut opts = TextEncoderOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(describe(&[1, 512], Some(DataType::I32)), "[1, 512] int32");
  assert_eq!(describe(&[1, 512], None), "[1, 512] none");
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
  let sha: String = Sha256::digest(crate::BUNDLED_TOKENIZER)
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
  let tok = configured_tokenizer_from_bytes(crate::BUNDLED_TOKENIZER).expect("configure tokenizer");
  tok.encode(text, true).expect("encode").get_ids().to_vec()
}

/// Token-id EXACT-equality over a pinned corpus (English, CJK, emoji). These ids
/// are identity-comparable to textclap: the tokenizer artifact is byte-identical
/// (SHA above) and the truncation config matches textclap's
/// `force_max_length_truncation`. The live cross-check against the textclap crate
/// itself is `tests/tokenizer_identity_textclap.rs` (feature `parity-oracle`).
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

/// Truncation identity: an input far longer than the 512-token window truncates
/// to EXACTLY [`TEXT_MAX_TOKENS`], never overflowing the RoBERTa position table
/// (matches textclap's `LongestFirst@512`).
#[test]
fn long_input_truncates_to_window() {
  let long = "word ".repeat(2000); // ≫ 512 tokens
  let got = ids(&long);
  assert_eq!(
    got.len(),
    TEXT_MAX_TOKENS,
    "truncation must cap ids at the window"
  );
  // RoBERTa keeps the specials on truncation: first is <s>, last is </s>.
  assert_eq!(got[0], 0, "leading <s>");
  assert_eq!(got[TEXT_MAX_TOKENS - 1], 2, "trailing </s>");
}
