use super::*;

// ── A4: options ──────────────────────────────────────────────────────────────

#[test]
fn options_default_equals_new_and_is_cpu_and_gpu() {
  assert_eq!(TextEmbedderOptions::default(), TextEmbedderOptions::new());
  assert_eq!(TextEmbedderOptions::new().compute(), DEFAULT_TEXT_COMPUTE);
  assert_eq!(DEFAULT_TEXT_COMPUTE, ComputeUnits::CpuAndGpu);
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
  assert_eq!(describe(&[1, 64], Some(DataType::I32)), "[1, 64] int32");
  assert_eq!(describe(&[1, 768], None), "[1, 768] none");
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = TextEmbedderOptions::new().with_compute(ComputeUnits::All);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("all"), "serialized: {json}");
  let back: TextEmbedderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_defaults_missing_compute() {
  let back: TextEmbedderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(back, TextEmbedderOptions::new());
}

// ── A10/A11: build_window (hermetic; the fixed padded-window contract) ────────

const T: usize = 64;

#[test]
fn build_window_right_pad_places_prefix_and_pads_suffix() {
  let ids = [10u32, 20, 30];
  let w = build_window(&ids, 7, PadSide::Right, T).expect("window");
  assert_eq!(w.len(), T);
  assert_eq!(&w[..3], &[10i32, 20, 30]);
  assert!(w[3..].iter().all(|&x| x == 7), "suffix must be pad_id");
}

#[test]
fn build_window_left_pad_places_suffix_and_pads_prefix() {
  let ids = [10u32, 20, 30];
  let w = build_window(&ids, 7, PadSide::Left, T).expect("window");
  assert_eq!(w.len(), T);
  assert!(w[..T - 3].iter().all(|&x| x == 7), "prefix must be pad_id");
  assert_eq!(&w[T - 3..], &[10i32, 20, 30]);
}

#[test]
fn build_window_full_window_has_no_pad() {
  let ids: Vec<u32> = (0..T as u32).collect();
  let w_right = build_window(&ids, 7, PadSide::Right, T).expect("full window");
  let w_left = build_window(&ids, 7, PadSide::Left, T).expect("full window");
  // A full window is identical regardless of pad side (no pad positions).
  let expected: Vec<i32> = (0..T as i32).collect();
  assert_eq!(w_right, expected);
  assert_eq!(w_left, expected);
}

#[test]
fn build_window_rejects_overlong_ids_with_typed_error() {
  let overlong = vec![1u32; T + 1];
  match build_window(&overlong, 0, PadSide::Right, T) {
    Err(Error::TokenCount { got, max }) => {
      assert_eq!(got, T + 1);
      assert_eq!(max, T);
    }
    other => panic!("expected TokenCount, got {other:?}"),
  }
}

#[test]
fn build_window_rejects_out_of_range_token_id() {
  match build_window(&[u32::MAX], 0, PadSide::Right, T) {
    Err(Error::TokenIdRange { id }) => assert_eq!(id, u32::MAX),
    other => panic!("expected TokenIdRange, got {other:?}"),
  }
}

// ── A11: tokenizer seam (hermetic; a caller-supplied synthetic tokenizer) ─────

/// A minimal valid WordLevel `tokenizer.json` — enough to exercise the module's
/// truncation/padding configuration seam without the multi-megabyte Gemma
/// tokenizer the artifact carries.
const TINY_TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "<pad>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5 },
    "unk_token": "<pad>"
  }
}"#;

/// The configured tokenizer seam applies this module's truncation (`LongestFirst`
/// at the resolved `T`) and disables the tokenizer's own padding — so an
/// over-length input truncates to exactly `T` real ids (which `build_window`
/// then pads), regardless of what policy the tokenizer carried.
#[test]
fn configured_tokenizer_truncates_to_window_and_disables_padding() {
  let max_tokens = 4;
  let tok = configured_tokenizer_from_bytes(TINY_TOKENIZER.as_bytes(), max_tokens)
    .expect("configure tiny tokenizer");
  // 8 whitespace tokens; truncation must cap the encoding at 4.
  let ids = tok
    .encode("a b c d e a b c", false)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(ids.len(), max_tokens, "must truncate to the window");
  // Padding disabled: a short input is NOT padded by the tokenizer (the module
  // owns the fixed-window pad).
  let short = tok.encode("a b", false).expect("encode").get_ids().to_vec();
  assert_eq!(
    short,
    vec![1u32, 2],
    "short input stays unpadded by the tokenizer"
  );

  // The module then pads the short ids into the fixed window.
  let window = build_window(&short, 0, PadSide::Right, max_tokens).expect("window");
  assert_eq!(window, vec![1i32, 2, 0, 0]);
}

// ── E1: fail-closed placeholder tokenizer ─────────────────────────────────────

/// The placeholder guard does not short-circuit a REAL tokenizer: `from_memory`
/// with non-placeholder bytes proceeds PAST it to `Model::load`, which fails on a
/// nonexistent path with [`Error::Load`]. (Wave-A shipped this asserting
/// [`Error::TokenizerPlaceholder`] against the stub asset; the tokenizer-swap
/// flipped it, exactly as the Wave-A doc anticipated. The asset itself is gone —
/// the tokenizer ships with the model artifact — so the real-bytes half of this
/// now lives in the artifact gates.)
#[test]
fn from_memory_accepts_a_real_tokenizer_past_the_guard() {
  let real = TINY_TOKENIZER.as_bytes();
  assert!(
    ensure_not_placeholder(real).is_ok(),
    "a real (non-sentinel) tokenizer must pass the placeholder guard"
  );
  match TextEmbedder::from_memory(
    "/nonexistent/model.mlmodelc",
    real,
    TextEmbedderOptions::new(),
  ) {
    Err(Error::Load(_)) => {}
    other => panic!("expected Error::Load past the guard, got {other:?}"),
  }
}

/// `load` resolves the tokenizer from the ARTIFACT ROOT — the directory
/// CONTAINING the `.mlmodelc`, where the published bundle stages it beside the
/// two graphs and the pos-emb sidecar. Hermetic: path arithmetic only.
#[test]
fn artifact_tokenizer_path_is_the_bundle_sibling() {
  assert_eq!(
    artifact_tokenizer_path(Path::new(
      "/m/siglip2-base-patch16-naflex-512/siglip2_text_64.mlmodelc"
    )),
    Path::new("/m/siglip2-base-patch16-naflex-512/tokenizer.json"),
  );
  // A bare bundle name has an empty parent: the sidecar resolves in the current
  // directory, the same place the bundle itself would.
  assert_eq!(
    artifact_tokenizer_path(Path::new("siglip2_text_64.mlmodelc")),
    Path::new("tokenizer.json"),
  );
}

/// A missing sidecar is its own typed, actionable failure — not a confusing
/// `Model::load` error — and it names the file.
#[test]
fn load_reports_a_missing_artifact_tokenizer() {
  match TextEmbedder::load("/nonexistent/model.mlmodelc", TextEmbedderOptions::new()) {
    Err(Error::ArtifactTokenizerRead { path, .. }) => {
      assert_eq!(path, Path::new("/nonexistent/tokenizer.json"));
    }
    other => panic!("expected ArtifactTokenizerRead, got {other:?}"),
  }
}

/// BOTH guards the embedded bytes used to carry now run on the file `load`
/// actually reads, BEFORE any model load: a placeholder staged into an artifact
/// tree fails closed, and so does a tokenizer that is merely *not the pinned
/// one*. Without the second, moving the tokenizer out of the crate would have
/// traded guaranteed-correct bytes for unverified ones.
#[test]
fn load_guards_the_sidecar_it_reads() {
  let dir = tempfile::tempdir().expect("tempdir");
  // The model path never exists, so reaching `Model::load` would surface as
  // `Error::Load` — anything else proves the guard fired first.
  let model_path = dir.path().join("siglip2_text_64.mlmodelc");
  let tokenizer_path = dir.path().join("tokenizer.json");

  let mut placeholder = br#"{"junk":""#.to_vec();
  placeholder.extend_from_slice(PLACEHOLDER_SENTINEL);
  placeholder.extend_from_slice(br#""}"#);
  std::fs::write(&tokenizer_path, &placeholder).expect("write placeholder sidecar");
  match TextEmbedder::load(&model_path, TextEmbedderOptions::new()) {
    Err(Error::TokenizerPlaceholder) => {}
    other => panic!("expected TokenizerPlaceholder for a placeholder sidecar, got {other:?}"),
  }

  std::fs::write(&tokenizer_path, TINY_TOKENIZER.as_bytes()).expect("write foreign sidecar");
  match TextEmbedder::load(&model_path, TextEmbedderOptions::new()) {
    Err(Error::ArtifactTokenizerIdentity {
      path,
      expected,
      actual,
    }) => {
      assert_eq!(path, tokenizer_path);
      assert_eq!(expected, contract::TOKENIZER_SHA256_HEX);
      assert_ne!(actual, contract::TOKENIZER_SHA256_HEX);
    }
    other => panic!("expected ArtifactTokenizerIdentity for a foreign sidecar, got {other:?}"),
  }
}

/// The guard is a placeholder sentinel scan, not a blanket reject: a small
/// non-placeholder tokenizer (the length fast-path is an optimization, not a
/// semantic) passes.
#[test]
fn placeholder_guard_accepts_real_tokenizer_bytes() {
  assert!(ensure_not_placeholder(TINY_TOKENIZER.as_bytes()).is_ok());
}

/// The durable regression guard: a small buffer carrying the sentinel is refused
/// with
/// [`Error::TokenizerPlaceholder`], so staging the build-time placeholder
/// `tokenizer.json` fails closed rather than shipping a meaningless tokenizer.
#[test]
fn placeholder_guard_rejects_the_sentinel_buffer() {
  let mut buf = br#"{"junk":""#.to_vec();
  buf.extend_from_slice(PLACEHOLDER_SENTINEL);
  buf.extend_from_slice(br#""}"#);
  match ensure_not_placeholder(&buf) {
    Err(Error::TokenizerPlaceholder) => {}
    other => panic!("expected TokenizerPlaceholder for a sentinel buffer, got {other:?}"),
  }
}

// ── E2: lowercase composition (mixed-case oracles) ────────────────────────────

/// A tiny WordLevel tokenizer whose vocab carries BOTH a lowercase and an
/// uppercase entry for the same letter — proves the composed `Lowercase`
/// normalizer runs before the model lookup (the uppercase id is never chosen).
const CASE_COLLISION_TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "<pad>": 0, "a": 1, "b": 2, "A": 6 },
    "unk_token": "<pad>"
  }
}"#;

/// A tiny WordLevel tokenizer carrying its OWN `Replace` normalizer (`x` → `a`).
/// Composing `Lowercase` AHEAD of it turns `X` into `x` into `a`; composing it
/// AFTER would leave `X` unmatched — so the encoded id discriminates the order.
const REPLACE_NORMALIZER_TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": { "type": "Replace", "pattern": { "String": "x" }, "content": "a" },
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "<pad>": 0, "a": 1, "b": 2 },
    "unk_token": "<pad>"
  }
}"#;

/// The configured tokenizer lowercases before the model lookup: mixed-case
/// `"A B"` encodes to the lowercase ids `[1, 2]`. Non-vacuity: `TINY_TOKENIZER`
/// carries NO normalizer, so without the composition `A`/`B` are out-of-vocab
/// and fall to the `<pad>` unk (`[0, 0]`) — the mixed-case oracle is sharp.
/// (Covers the `None`-normalizer arm of the composition.)
#[test]
fn configured_tokenizer_lowercases_before_lookup() {
  let tok =
    configured_tokenizer_from_bytes(TINY_TOKENIZER.as_bytes(), 8).expect("configure tokenizer");
  let ids = tok.encode("A B", false).expect("encode").get_ids().to_vec();
  assert_eq!(
    ids,
    vec![1u32, 2],
    "mixed case must lowercase to [a, b] ids"
  );
}

/// With both `"a"` and `"A"` in the vocab, `"A"` still resolves to the lowercase
/// id `1` (never the uppercase `6`) — the normalizer runs before the lookup.
#[test]
fn configured_tokenizer_prefers_lowercase_vocab_entry() {
  let tok = configured_tokenizer_from_bytes(CASE_COLLISION_TOKENIZER.as_bytes(), 8)
    .expect("configure tokenizer");
  let ids = tok.encode("A", false).expect("encode").get_ids().to_vec();
  assert_eq!(ids, vec![1u32], "must pick the lowercase entry, not id 6");
}

/// `Lowercase` is composed AHEAD of the loaded normalizer, and the loaded
/// normalizer is preserved (not clobbered): `"X b"` lowercases to `"x b"`, then
/// the loaded `Replace` maps `x` → `a`, giving `[1, 2]`. Composed the other way,
/// `X` never reaches `Replace` and would fall to unk `[0, 2]` — so this pins the
/// ordering.
#[test]
fn configured_tokenizer_composes_ahead_of_existing_normalizer() {
  let tok = configured_tokenizer_from_bytes(REPLACE_NORMALIZER_TOKENIZER.as_bytes(), 8)
    .expect("configure tokenizer");
  let ids = tok.encode("X b", false).expect("encode").get_ids().to_vec();
  assert_eq!(
    ids,
    vec![1u32, 2],
    "Lowercase must run before the loaded Replace normalizer"
  );
}
