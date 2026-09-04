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
    Err(Error::TokenCount(e)) => {
      assert_eq!(e.got(), T + 1);
      assert_eq!(e.max(), T);
    }
    other => panic!("expected TokenCount, got {other:?}"),
  }
}

#[test]
fn build_window_rejects_out_of_range_token_id() {
  match build_window(&[u32::MAX], 0, PadSide::Right, T) {
    Err(Error::TokenIdRange(id)) => assert_eq!(id, u32::MAX),
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

// ── Special-token overhead vs the read-back window ───────────────────────────

/// Serialize [`TINY_TOKENIZER`] with a `TemplateProcessing` post-processor whose
/// single-sequence template adds exactly `added` special tokens — the shape a
/// caller can hand `from_files`, and the number the tokenizers crate subtracts
/// from the truncation window without checking. One `SpecialToken` carrying
/// `added` ids is used rather than `added` template pieces because
/// `TemplateProcessing::added_tokens` sums `ids.len()` per piece, so both count
/// the same and only this one scales to a 512-token window.
fn tokenizer_bytes_with_special_overhead(added: usize) -> Vec<u8> {
  use tokenizers::processors::template::{SpecialToken, TemplateProcessing};

  let mut tokenizer =
    Tokenizer::from_bytes(TINY_TOKENIZER.as_bytes()).expect("load the tiny tokenizer");
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
  for added in [1usize, 2, 5] {
    let bytes = tokenizer_bytes_with_special_overhead(added);
    let tok = Tokenizer::from_bytes(&bytes).expect("reload the fixture");
    let post = tok.get_post_processor().expect("the fixture has one");
    assert_eq!(post.added_tokens(false), added, "single-sequence overhead");
  }
}

/// `added > max_tokens` is the tokenizers crate's unchecked
/// `max_length - added_tokens` subtraction: it PANICS with "attempt to subtract
/// with overflow" under overflow checks and wraps to a near-`usize::MAX` window
/// in release. The door refuses the pairing first, naming both numbers.
#[test]
fn configure_tokenizer_refuses_overhead_over_the_window() {
  let bytes = tokenizer_bytes_with_special_overhead(2);
  match configured_tokenizer_from_bytes(&bytes, 1) {
    Err(Error::SpecialTokenOverhead(overhead)) => {
      assert_eq!(overhead.added(), 2);
      assert_eq!(overhead.window(), 1);
    }
    other => panic!("expected SpecialTokenOverhead, got {other:?}"),
  }
}

/// `added == max_tokens` does not overflow — and is refused anyway, because the
/// effective text window is then zero and every encoding is the special tokens
/// alone. The second half of this test is that fact, measured through a
/// tokenizer configured BEHIND the guard: "a b c d e" encodes to nothing but the
/// two specials, which is a wrong answer rather than a reported failure.
#[test]
fn configure_tokenizer_refuses_overhead_equal_to_the_window() {
  let bytes = tokenizer_bytes_with_special_overhead(2);
  match configured_tokenizer_from_bytes(&bytes, 2) {
    Err(Error::SpecialTokenOverhead(overhead)) => {
      assert_eq!(overhead.added(), 2);
      assert_eq!(overhead.window(), 2);
    }
    other => panic!("expected SpecialTokenOverhead, got {other:?}"),
  }

  // What the refusal buys, configured directly so the guard is out of the way.
  let mut raw = Tokenizer::from_bytes(&bytes).expect("load the fixture");
  raw
    .with_truncation(Some(TruncationParams {
      max_length: 2,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .expect("added == window does not overflow");
  let ids = raw
    .encode("a b c d e", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(
    ids,
    vec![0u32, 0],
    "at a zero effective window every input is the specials alone"
  );
}

/// One token of room is enough: `added < max_tokens` configures, and the window
/// then holds the specials plus at least one real token.
#[test]
fn configure_tokenizer_accepts_overhead_below_the_window() {
  let bytes = tokenizer_bytes_with_special_overhead(2);
  let tok = configured_tokenizer_from_bytes(&bytes, 3).expect("2 specials fit a 3-token window");
  let ids = tok
    .encode("a b c d e", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(ids.len(), 3, "two specials plus one real token");
  assert_eq!(ids[2], 1, "the real token survives (`a`)");
}

/// A tokenizer with no post-processor at all reads as zero overhead, so the
/// guard never fires on it — the ordinary path stays open at every window.
#[test]
fn configure_tokenizer_accepts_a_tokenizer_without_a_post_processor() {
  for max_tokens in [1usize, 4, 64] {
    configured_tokenizer_from_bytes(TINY_TOKENIZER.as_bytes(), max_tokens)
      .expect("no post-processor is zero overhead");
  }
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
    Err(Error::ArtifactTokenizerRead(e)) => {
      assert_eq!(e.path(), Path::new("/nonexistent/tokenizer.json"));
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
    Err(Error::ArtifactTokenizerIdentity(e)) => {
      assert_eq!(e.path(), tokenizer_path.as_path());
      assert_eq!(e.expected(), contract::TOKENIZER_SHA256_HEX);
      assert_ne!(e.actual(), contract::TOKENIZER_SHA256_HEX);
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

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself — its feature names, its
// element type, its geometry, its state clause, and the one axis it READS back
// rather than requires — against descriptions built with the same fixture
// machinery. They are the whole of this door's contract coverage: no siglip
// `.mlmodelc` is staged in this repository, so `tests/siglip/text_model_io.rs`
// runs against nothing.

use crate::{
  AxisRange, FeatureInfo, ModelDescription, embeddings::siglip::error::contract_violation,
  model::RawShapeConstraint,
};

/// The text window the staged conversion pins
/// (`conversion/siglip/scripts/_siglip_common.py`: `TEXT_WINDOW = 64`). The
/// door never spells this number — it reads whatever the graph pins — so it
/// appears here only as the fixture's own choice, and
/// `the_contract_reads_back_whatever_window_the_graph_pins` proves a different
/// one is equally acceptable.
const STAGED_TEXT_WINDOW: usize = 64;

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

/// A text description at window `t`: the single `input_ids` input and the one
/// projection output, no state.
fn text_description(t: usize) -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(names::INPUT_IDS, &[1, t], DataType::I32)],
    vec![fixed(
      names::TEXT_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into the siglip
/// error vocabulary — exactly what `TextEmbedder::from_parts` does after
/// `Model::load`. There is nothing after it: the window is READ from a
/// description this has accepted, and carries no refusal of its own.
fn check(description: &ModelDescription) -> Result<()> {
  crate::model::contract::check_load_contract(description, &text_contract())
    .map_err(contract_violation)
}

/// The contract states exactly the geometry the staged conversion emits.
#[test]
fn the_contract_accepts_the_staged_geometry() {
  assert!(check(&text_description(STAGED_TEXT_WINDOW)).is_ok());
}

/// **The `AnyFixed` clause.** The window is the conversion's, not this crate's,
/// so any pinned window is acceptable — and the value is read back rather than
/// required.
#[test]
fn the_contract_reads_back_whatever_window_the_graph_pins() {
  for t in [1usize, 16, STAGED_TEXT_WINDOW, 512] {
    let description = text_description(t);
    assert!(check(&description).is_ok(), "window {t}");
    // What `from_parts` does after the check: read the value the contract
    // established is the only one.
    assert_eq!(
      description
        .input(names::INPUT_IDS)
        .expect("input_ids")
        .shape()[1],
      t,
      "the window read back must be the one the graph pins"
    );
  }
}

/// **The flexible-shape refusal**, and it bites on exactly the axis this door
/// reads back: [`crate::FeatureInfo::shape`] reports the DEFAULT shape of a
/// `RangeDims` input, so a graph whose window is a RANGE declares one number
/// and accepts others — and this door would pad every request to that default
/// and truncate the tokenizer at it.
#[test]
fn the_contract_refuses_a_flexible_window() {
  let description = ModelDescription::from_parts(
    vec![multi_array(
      names::INPUT_IDS,
      &[1, STAGED_TEXT_WINDOW],
      DataType::I32,
      false,
      3,
      Vec::new(),
      &[1, STAGED_TEXT_WINDOW],
    )],
    vec![fixed(
      names::TEXT_FEATURES,
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

/// **FALSIFIER for deleting this door's own zero refusal.** A window of ZERO is
/// refused BY the contract and by nothing else: an axis pinned at zero admits
/// exactly one size, so no clause but `Dim::AnyFixed`'s own can see it, and that
/// clause is now the whole guard — the read-back in `from_parts` states no
/// `>= 1` of its own to catch it a second time.
#[test]
fn a_zero_window_is_refused_by_the_contract() {
  let description = text_description(0);
  let violation = crate::model::contract::check_load_contract(&description, &text_contract())
    .expect_err("`AnyFixed`'s zero clause refuses a window pinned at zero");
  assert!(
    matches!(
      &violation,
      crate::model::contract::ContractViolation::ZeroSizedAxis(zero)
        if zero.feature() == names::INPUT_IDS
    ),
    "{violation}"
  );
}

/// The token window is int32 and the projection is 768-wide f32 — the §0
/// contract, and the two facts a wrong conversion would move.
#[test]
fn the_contract_refuses_a_wrong_dtype_or_projection_width() {
  let float_ids = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_IDS,
      &[1, STAGED_TEXT_WINDOW],
      DataType::F32,
    )],
    vec![fixed(
      names::TEXT_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&float_ids).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::INPUT_IDS),
    "{err}"
  );

  let wrong_width = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_IDS,
      &[1, STAGED_TEXT_WINDOW],
      DataType::I32,
    )],
    vec![fixed(names::TEXT_FEATURES, &[1, 512], DataType::F32)],
    Vec::new(),
  );
  let err = check(&wrong_width).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::TEXT_FEATURES),
    "{err}"
  );
}

/// **The input-SET clause this door's docs used to delegate to a model-gated
/// test.** The SigLIP text graph takes `input_ids` and nothing else; a graph
/// that grew an `attention_mask` clears every per-feature clause and then fails
/// every prediction, because [`TextEmbedder::embed`] supplies one input. The
/// assertion is hermetic now, which matters because no siglip artifact is
/// staged for the `#[ignore]`d gate to run against.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::INPUT_IDS, &[1, STAGED_TEXT_WINDOW], DataType::I32),
      fixed("attention_mask", &[1, STAGED_TEXT_WINDOW], DataType::I32),
    ],
    vec![fixed(
      names::TEXT_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableInput(name)) if name == "attention_mask"),
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
      fixed(names::INPUT_IDS, &[1, STAGED_TEXT_WINDOW], DataType::I32),
      multi_array(
        "attention_mask",
        &[1, STAGED_TEXT_WINDOW],
        DataType::I32,
        true,
        2,
        vec![vec![1, STAGED_TEXT_WINDOW]],
        &[1, STAGED_TEXT_WINDOW],
      ),
    ],
    vec![fixed(
      names::TEXT_FEATURES,
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
fn the_contract_refuses_an_optional_features_output() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_IDS,
      &[1, STAGED_TEXT_WINDOW],
      DataType::I32,
    )],
    vec![multi_array(
      names::TEXT_FEATURES,
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
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::TEXT_FEATURES),
    "{err}"
  );
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly `input_ids` and `text_features` plus a state clears every
/// per-feature clause AND the input set — and only then meets
/// [`TextEmbedder::embed`], which predicts through the STATELESS API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::INPUT_IDS,
      &[1, STAGED_TEXT_WINDOW],
      DataType::I32,
    )],
    vec![fixed(
      names::TEXT_FEATURES,
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
/// staged artifact and carries no `#[ignore]` — which matters more here than
/// anywhere else in this crate, because the siglip `.mlmodelc` bundles are the
/// one kit `Models/` stages nothing of (only the tokenizer sidecar).
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
    .expect_err("silero does not satisfy the siglip text contract");
  assert!(
    matches!(&violation, crate::model::contract::ContractViolation::Missing(m)
      if m.feature() == names::INPUT_IDS),
    "expected `input_ids` missing, got {violation}"
  );
}

use crate::embeddings::siglip::error::PostProcessorTemplate;

// ── A parseable-but-inconsistent post-processor template ────────────────────

/// The three defective single templates, as a `tokenizer.json` post-processor
/// writes them. Each PARSES — the tokenizers deserializer skips its builder's
/// `validate` — and each is a panic or a wrong answer at the first `encode`.
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

/// [`TINY_TOKENIZER`] with `post_processor` spliced in — the shape a caller can
/// hand `from_files` / `from_memory`, neither of which pins a tokenizer.
fn tiny_tokenizer_with_post_processor(post: &str) -> Vec<u8> {
  TINY_TOKENIZER
    .replace(
      r#""post_processor": null"#,
      &format!(r#""post_processor": {post}"#),
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
    let bytes = tiny_tokenizer_with_post_processor(post);
    match configured_tokenizer_from_bytes(&bytes, 64) {
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
  match configured_tokenizer_from_bytes(&tiny_tokenizer_with_post_processor(&post), 64) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UndeclaredSpecialToken(id))) => {
      assert_eq!(id, "<s>");
    }
    other => panic!("expected UndeclaredSpecialToken, got {other:?}"),
  }
}

/// A sound template still configures — the guard is a structural rule, not a
/// refusal of `TemplateProcessing` as a kind. (Gemma's own post-processor is a
/// template, so refusing the kind would refuse the pinned artifact.)
#[test]
fn configure_tokenizer_accepts_a_well_formed_single_template() {
  let post = r#"{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{"<s>":{"id":"<s>","ids":[0],"tokens":["<pad>"]}}}"#;
  let tok = configured_tokenizer_from_bytes(&tiny_tokenizer_with_post_processor(post), 64)
    .expect("a sound template configures");
  assert_eq!(
    tok.encode("a b", true).expect("encode").get_ids(),
    &[0, 1, 2],
    "the special token, then the text"
  );
}

/// The overhead guard cannot substitute for the structural one: an undeclared
/// `SpecialToken` id contributes ZERO to `added_tokens(false)`, so a template
/// that fills the whole window with undeclared specials reads as NO overhead and
/// sails past the overhead guard. Only the structural check sees it.
#[test]
fn the_overhead_reading_is_blind_to_undeclared_special_tokens() {
  const T: usize = 8;
  let single: Vec<String> = (0..T)
    .map(|i| format!(r#"{{"SpecialToken":{{"id":"<undeclared{i}>","type_id":0}}}}"#))
    .collect();
  let post = format!(
    r#"{{"type":"TemplateProcessing","single":[{}],"pair":[{{"Sequence":{{"id":"A","type_id":0}}}}],"special_tokens":{{}}}}"#,
    single.join(",")
  );
  let bytes = tiny_tokenizer_with_post_processor(&post);

  let raw = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    raw
      .get_post_processor()
      .expect("has one")
      .added_tokens(false),
    0,
    "the overhead reading is blind to undeclared ids"
  );

  match configured_tokenizer_from_bytes(&bytes, T) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UndeclaredSpecialToken(id))) => {
      assert_eq!(id, "<undeclared0>");
    }
    other => panic!("expected UndeclaredSpecialToken, got {other:?}"),
  }
}

// ── The pad id is converted, never quietly replaced ─────────────────────────

/// The first id above `i32::MAX`.
const FIRST_ID_PAST_I32: u32 = 2_147_483_648;

/// SigLIP attends every position and pools the final one, so the pad id is part
/// of the answer (D6). An out-of-range `<pad>` is therefore REPORTED — it used
/// to fall through to `0`, silently embedding against a padding token the caller
/// never chose.
#[test]
fn resolve_pad_id_refuses_a_pad_token_past_int32() {
  let bytes = TINY_TOKENIZER
    .replace(r#""<pad>": 0"#, &format!(r#""<pad>": {FIRST_ID_PAST_I32}"#))
    .into_bytes();
  let tok = Tokenizer::from_bytes(&bytes).expect("parse");
  match resolve_pad_id(&tok) {
    Err(Error::TokenIdRange(id)) => assert_eq!(id, FIRST_ID_PAST_I32),
    other => panic!("expected TokenIdRange, got {other:?}"),
  }
}

/// The ordinary paths still resolve: the tokenizer's own `<pad>`, and `0` when
/// it has none — the case the range failure used to be confused with.
#[test]
fn resolve_pad_id_reads_the_vocabulary_then_falls_back() {
  let tok = Tokenizer::from_bytes(TINY_TOKENIZER.as_bytes()).expect("parse");
  assert_eq!(resolve_pad_id(&tok).expect("in range"), 0);

  let bytes = TINY_TOKENIZER
    .replace(r#""<pad>": 0, "#, "")
    .replace(r#""unk_token": "<pad>""#, r#""unk_token": "a""#)
    .into_bytes();
  let tok = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(resolve_pad_id(&tok).expect("no <pad> is not an error"), 0);
}

// ── The artifact tokenizer is judged before it is parsed ────────────────────

/// `load` hashes the RAW sidecar bytes and refuses them BEFORE handing them to
/// the tokenizers parser. Bytes that are not even JSON prove the ordering: the
/// failure is the identity pin, not `TokenizerLoad`. That is what makes the
/// dependency's deserializer unreachable for a foreign artifact.
#[test]
fn load_hashes_the_sidecar_before_parsing_it() {
  let dir = tempfile::tempdir().expect("tempdir");
  let model_path = dir.path().join("siglip2_text_64.mlmodelc");
  let tokenizer_path = dir.path().join("tokenizer.json");
  std::fs::write(&tokenizer_path, b"this is not json at all").expect("write sidecar");

  match TextEmbedder::load(&model_path, TextEmbedderOptions::new()) {
    Err(Error::ArtifactTokenizerIdentity(e)) => {
      assert_eq!(e.path(), tokenizer_path.as_path());
      assert_eq!(e.expected(), contract::TOKENIZER_SHA256_HEX);
    }
    other => panic!("expected ArtifactTokenizerIdentity before any parse, got {other:?}"),
  }
}

/// …and where BOTH guards would refuse, the structural one is the diagnostic
/// that wins. The overhead here is real and declared — `count_added` reports the
/// full window, so `Error::SpecialTokenOverhead` would refuse this tokenizer on
/// its own — which is what makes this a falsifier for the ORDER rather than for
/// the presence of either check.
#[test]
fn a_structural_defect_outranks_an_overhead_that_would_also_refuse() {
  const T: usize = 8;
  let bytes = tokenizer_bytes_with_overhead_and_a_pair_sequence(T);
  let raw = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    raw
      .get_post_processor()
      .expect("has one")
      .added_tokens(false),
    T,
    "the overhead guard would refuse this tokenizer on its own"
  );
  match configured_tokenizer_from_bytes(&bytes, T) {
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
  let bytes = tiny_tokenizer_with_post_processor(SEQUENCE_FEEDING_THREE_ENCODINGS);
  match configured_tokenizer_from_bytes(&bytes, 64) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UnsupportedEncodingCount(n))) => {
      assert_eq!(n, 3, "the count the second template would have received");
    }
    other => panic!("expected UnsupportedEncodingCount(3), got {other:?}"),
  }
}

/// The placement rule reaches this door too, and it is the round-4 MEDIUM
/// reproduced end to end at a small window. With the guard out of the way the
/// configured tokenizer advertises no overhead, truncates to the full window and
/// then hands `build_window` twice that many ids, so ordinary text longer than
/// half the window fails with a typed `TokenCount` although construction
/// succeeded. The guard refuses the tokenizer instead, naming the number of
/// placements.
#[test]
fn configure_tokenizer_refuses_a_template_that_places_the_text_twice() {
  const WINDOW: usize = 4;
  let bytes = tiny_tokenizer_with_post_processor(TEMPLATE_PLACING_THE_TEXT_TWICE);

  // What the refusal buys, measured with the guard bypassed.
  let mut raw = Tokenizer::from_bytes(&bytes).expect("parse");
  assert_eq!(
    raw
      .get_post_processor()
      .expect("has one")
      .added_tokens(false),
    0,
    "a repeated `$A` reads as no overhead at all"
  );
  raw
    .with_truncation(Some(TruncationParams {
      max_length: WINDOW,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .expect("zero overhead does not overflow");
  raw.with_padding(None);
  let ids = raw
    .encode("a b a", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(ids, vec![1, 2, 1, 1, 2, 1], "three tokens in, six out");
  match build_window(&ids, 0, PadSide::Right, WINDOW) {
    Err(Error::TokenCount(count)) => {
      assert_eq!(count.got(), 6);
      assert_eq!(count.max(), WINDOW);
    }
    other => panic!("expected the TokenCount backstop to fire, got {other:?}"),
  }

  match configured_tokenizer_from_bytes(&bytes, WINDOW) {
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
  let bytes = tiny_tokenizer_with_post_processor(SEQUENCE_REACHING_A_PAIR_TEMPLATE);

  let raw = Tokenizer::from_bytes(&bytes).expect("parse");
  for text in ["a b", "a b a b a b"] {
    assert!(
      raw.encode(text, true).expect("encode").get_ids().is_empty(),
      "the pair template places no sequence, so the text is gone"
    );
  }

  match configured_tokenizer_from_bytes(&bytes, 64) {
    Err(Error::PostProcessorTemplate(PostProcessorTemplate::UnsupportedEncodingCount(n))) => {
      assert_eq!(n, 2, "the count the second template would have received");
    }
    other => panic!("expected UnsupportedEncodingCount(2), got {other:?}"),
  }
}
