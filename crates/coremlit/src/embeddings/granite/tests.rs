use super::*;

// ── Options ────────────────────────────────────────────────────────────────

#[test]
fn options_default_equals_new() {
  assert_eq!(TextEmbedderOptions::default(), TextEmbedderOptions::new());
  assert_eq!(TextEmbedderOptions::new().compute(), DEFAULT_COMPUTE);
  assert_eq!(DEFAULT_COMPUTE, ComputeUnits::All);
}

#[test]
fn options_with_and_set_compute() {
  let opts = TextEmbedderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
  let mut opts = TextEmbedderOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(describe(&[1, 512], Some(DataType::I32)), "[1, 512] int32");
  assert_eq!(describe(&[1, 384], None), "[1, 384] none");
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = TextEmbedderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_neural_engine"), "serialized: {json}");
  let back: TextEmbedderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

// ── Tokenizer identity gate (hermetic; the real tokenizer seam) ─────────────

/// SHA-256 of the bundled tokenizer must equal the tokenizer that produced the
/// committed goldens (the source model repo revision) — byte-identity is the
/// foundation of token-id identity. Any drift in `assets/tokenizer.json` fails
/// here.
#[test]
fn bundled_tokenizer_sha_matches_golden_source_pin() {
  use sha2::{Digest, Sha256};
  let sha: String = Sha256::digest(BUNDLED_TOKENIZER)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  assert_eq!(
    sha, "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f",
    "bundled tokenizer.json diverged from the granite tokenizer that cut the goldens"
  );
}

/// Encode `text` through granite's ACTUAL configured tokenizer seam (the same
/// path [`TextEmbedder::token_ids`] uses), hermetically (no model).
fn ids(text: &str) -> Vec<u32> {
  let tok = configured_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("configure tokenizer");
  tok.encode(text, true).expect("encode").get_ids().to_vec()
}

/// Token-id EXACT-equality against a pinned subset of the committed corpus. The
/// full 16-entry corpus identity gate is `tests/granite/tokenizer_identity.rs`;
/// these two hermetic sequences keep the in-lib seam honest without the fixture
/// file. `<|startoftext|>`=179934 (CLS, pooled) and `<|return|>`=179938 (EOS)
/// bracket every sequence.
#[test]
fn token_ids_match_pinned_golden_subset() {
  let cases: &[(&str, &[u32])] = &[
    ("hello world", &[179934, 24313, 2318, 179938]),
    (
      "how do I build a Rust CoreML inference library for text embeddings?",
      &[
        179934, 8775, 579, 317, 2966, 221, 54305, 15984, 4051, 86068, 11087, 355, 2145, 158816, 30,
        179938,
      ],
    ),
  ];
  for (text, expected) in cases {
    let got = ids(text);
    assert_eq!(&got, expected, "token-id drift for {text:?}");
  }
}

/// Truncation identity — the DIRECTION, not just the length, is gated.
///
/// A *non-repetitive* input longer than the 512-token window (ascending
/// integers, every token distinct) truncates to EXACTLY [`MAX_TOKENS`] without
/// overflowing the export sequence length, and — because the module configures
/// `TruncationDirection::Right` — the kept interior is the untruncated
/// encoding's PREFIX. A `Right → Left` flip (which keeps the SUFFIX instead)
/// changes the interior of distinct tokens and trips this.
#[test]
fn long_input_truncation_keeps_the_right_directional_prefix() {
  // Non-repetitive, comfortably over one window: "1 2 3 … 1000", all distinct.
  let long: String = (1..=1000)
    .map(|n| n.to_string())
    .collect::<Vec<_>>()
    .join(" ");

  let truncated = ids(&long);
  assert_eq!(
    truncated.len(),
    MAX_TOKENS,
    "truncation must cap ids at the window"
  );
  assert_eq!(truncated[0], 179934, "leading <|startoftext|> kept");
  assert_eq!(
    truncated[MAX_TOKENS - 1],
    179938,
    "trailing <|return|> kept"
  );

  // Untruncated reference: the SAME tokenizer bytes with truncation OFF.
  let full = tokenizers::Tokenizer::from_bytes(BUNDLED_TOKENIZER)
    .expect("load tokenizer")
    .encode(long.as_str(), true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert!(
    full.len() > MAX_TOKENS,
    "reference must actually overflow the window (got {})",
    full.len()
  );

  // RIGHT truncation ⇒ the 510 interior ids equal the untruncated PREFIX. Under
  // `Left` the interior would be the untruncated SUFFIX, which (distinct tokens)
  // differs ⇒ red.
  assert_eq!(
    &truncated[1..MAX_TOKENS - 1],
    &full[1..MAX_TOKENS - 1],
    "Right-truncation interior must equal the untruncated first-510 content tokens"
  );

  // Measure-then-pin: the exact 512-id sequence nailed to a SHA-256 constant, so
  // the whole interior is pinned absolutely. Any tokenizer-artifact or
  // truncation-config drift changes it.
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
    sha, "aec64c84fc8328d01b518a7cb4e63b42a00a659ba5d39789fc10a272667416af",
    "truncated 512-id sequence drifted (tokenizer artifact or truncation config changed)"
  );
}
