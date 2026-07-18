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
  let full = tokenizers::Tokenizer::from_bytes(crate::BUNDLED_TOKENIZER)
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
