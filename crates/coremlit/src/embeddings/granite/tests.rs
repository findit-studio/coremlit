use super::*;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};

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

// ── Fixed-window contract: padding override + build_window (hermetic) ────────
//
// A caller-supplied `tokenizer.json` can carry a padding policy; if it survived
// into `token_ids`, `embed` would mask PAD positions as real (corrupt embedding),
// pool CLS off position 0 (left padding), or overflow the window (fixed padding
// beyond 512, a release panic). These prove `configure_tokenizer` neutralizes
// every such policy and `build_window` is a typed guard, not a panic — with no
// model.

/// "hello world" through the granite tokenizer: `<|startoftext|>`=179934 (CLS,
/// pooled), then `hello`/`world`, then `<|return|>`=179938 (EOS). The exact
/// sequence is pinned by `token_ids_match_pinned_golden_subset` above; sourced
/// from the module contract (single source of truth).
const HELLO_WORLD_IDS: [u32; 4] = contract::SENTINEL_IDS;

/// A fresh bundled tokenizer carrying an adversarial fixed-window padding policy
/// (the kind a caller-supplied tokenizer might inherit), BEFORE this module's
/// config runs.
fn bundled_with_padding(direction: PaddingDirection) -> Tokenizer {
  let mut tok = Tokenizer::from_bytes(BUNDLED_TOKENIZER).expect("load bundled tokenizer");
  tok.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(MAX_TOKENS),
    direction,
    ..Default::default()
  }));
  tok
}

/// Fixed-512 RIGHT padding — the corrupt-mask case. Without the override the
/// tokenizer pads a short input to the full window, so `embed`'s mask would mark
/// the trailing PADs as real tokens. `configure_tokenizer` must disable the
/// tokenizer's own padding so only the real ids survive, and the window then
/// masks EXACTLY those.
#[test]
fn configured_tokenizer_disables_fixed_right_padding_mask_stays_correct() {
  let mut tok = bundled_with_padding(PaddingDirection::Right);

  // Precondition: the adversarial policy really does pad to the full window.
  let padded = tok
    .encode("hello world", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(
    padded.len(),
    MAX_TOKENS,
    "adversarial fixture must actually pad to the window"
  );

  // Override strips the padding: token_ids sees only the real, unpadded ids.
  configure_tokenizer(&mut tok).expect("configure");
  let real = tok
    .encode("hello world", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(
    real, HELLO_WORLD_IDS,
    "padding must be stripped, real ids only"
  );

  // The fixed window masks EXACTLY the real tokens — no PAD marked real.
  let (input_ids, mask) = build_window(&real, 0).expect("build window");
  assert_eq!(
    mask.iter().sum::<i32>(),
    i32::try_from(real.len()).unwrap(),
    "attention mask must count only the real tokens"
  );
  assert!(
    mask[..real.len()].iter().all(|&m| m == 1),
    "real tokens masked 1"
  );
  assert!(
    mask[real.len()..].iter().all(|&m| m == 0),
    "pad positions masked 0"
  );
  assert_eq!(input_ids[0], 179934, "CLS at position 0");
}

/// Fixed-512 LEFT padding — the wrong-CLS-pooling case. Without the override the
/// leading PADs push CLS (`<|startoftext|>`) off position 0, so CLS pooling would
/// read a PAD. `configure_tokenizer` must disable padding so CLS stays at 0.
#[test]
fn configured_tokenizer_disables_left_padding_keeps_cls_at_zero() {
  let mut tok = bundled_with_padding(PaddingDirection::Left);

  // Precondition: left padding pushes CLS off position 0 (the hazard).
  let padded = tok
    .encode("hello world", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(padded.len(), MAX_TOKENS);
  assert_ne!(
    padded[0], 179934,
    "left padding must push CLS off position 0 (the hazard being defended)"
  );

  // Override removes the leading pads: CLS is back at position 0.
  configure_tokenizer(&mut tok).expect("configure");
  let real = tok
    .encode("hello world", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(real, HELLO_WORLD_IDS);
  assert_eq!(
    real[0], 179934,
    "CLS must be at position 0 after the override"
  );
  let (input_ids, _mask) = build_window(&real, 0).expect("build window");
  assert_eq!(
    input_ids[0], 179934,
    "CLS stays at position 0 in the window"
  );
}

/// An over-long input (real text past the window) truncates to exactly
/// [`MAX_TOKENS`] through the configured seam and fills the window with real
/// tokens — no panic, CLS still at position 0.
#[test]
fn overlong_input_truncates_and_fills_the_window_without_panic() {
  // Non-repetitive, comfortably over one window: "1 2 3 … 1000".
  let long: String = (1..=1000)
    .map(|n| n.to_string())
    .collect::<Vec<_>>()
    .join(" ");
  let real = ids(&long); // configured seam: truncation on, padding off.
  assert_eq!(
    real.len(),
    MAX_TOKENS,
    "over-long input truncates to the window"
  );

  let (input_ids, mask) = build_window(&real, 0).expect("full window must build, not panic");
  assert!(
    mask.iter().all(|&m| m == 1),
    "a full window is entirely real tokens"
  );
  assert_eq!(input_ids[0], 179934, "CLS stays at position 0");
}

/// `build_window` returns a typed [`Error::TokenCount`] — never the release
/// out-of-bounds panic the old `debug_assert!` hid — if a tokenizer ever yields
/// more ids than the window.
#[test]
fn build_window_rejects_overlong_ids_with_typed_error() {
  let overlong = vec![7u32; MAX_TOKENS + 1];
  match build_window(&overlong, 0) {
    Err(Error::TokenCount { got, max }) => {
      assert_eq!(got, MAX_TOKENS + 1);
      assert_eq!(max, MAX_TOKENS);
    }
    other => panic!("expected Err(TokenCount), got {other:?}"),
  }
}

/// `build_window` returns a typed [`Error::TokenIdRange`] — never a silently
/// wrapping cast — for a token id outside the model's int32 range.
#[test]
fn build_window_rejects_out_of_range_token_id() {
  match build_window(&[u32::MAX], 0) {
    Err(Error::TokenIdRange { id }) => assert_eq!(id, u32::MAX),
    other => panic!("expected Err(TokenIdRange), got {other:?}"),
  }
}

/// `build_window` on a short real sequence masks exactly the real prefix and
/// right-pads the remainder with `pad_id` (masked 0) — the internal fixed-window
/// pad, done correctly.
#[test]
fn build_window_masks_prefix_and_right_pads_remainder() {
  let (input_ids, mask) = build_window(&[10, 20, 30], 7).expect("build");
  assert_eq!(&input_ids[..3], &[10i32, 20, 30]);
  assert!(
    input_ids[3..].iter().all(|&x| x == 7),
    "remainder is pad_id"
  );
  assert_eq!(&mask[..3], &[1i32, 1, 1]);
  assert!(mask[3..].iter().all(|&m| m == 0), "pad positions masked 0");
}

/// A full window (exactly [`MAX_TOKENS`] real ids) is accepted and entirely
/// masked — the boundary the old guard treated as `<=` must remain valid.
#[test]
fn build_window_accepts_a_full_window() {
  let (_input_ids, mask) = build_window(&vec![1u32; MAX_TOKENS], 0).expect("full window builds");
  assert_eq!(mask.iter().sum::<i32>(), i32::try_from(MAX_TOKENS).unwrap());
}

// ── embed_long: content-aware chunk geometry (hermetic; measuring tokenizer,
//    no model). The CoreML aggregation path is proven model-gated in
//    tests/granite/embed_long.rs. ─────────────────────────────────────────────

/// A deterministic multi-paragraph document comfortably over several 512-token
/// windows: 24 paragraphs of 40 distinct words each, `\n\n`-separated.
fn long_doc() -> String {
  (0..24)
    .map(|p| {
      (0..40)
        .map(|w| format!("para{p}word{w}"))
        .collect::<Vec<_>>()
        .join(" ")
    })
    .collect::<Vec<_>>()
    .join("\n\n")
}

/// THE hazard regression (design correction #1): the CONFIGURED (production)
/// tokenizer truncates a long input's id count to exactly [`MAX_TOKENS`], while
/// the MEASURING tokenizer (truncation disabled) reports the true, larger count.
/// `embed_long`'s chunker MUST measure with the latter — measuring with the
/// former would judge EVERY long document to "fit one window" and silently
/// degenerate `embed_long` into a truncated `embed`.
#[test]
fn measuring_tokenizer_reports_untruncated_counts() {
  // Non-repetitive, comfortably over one window: "1 2 3 … 1000".
  let long: String = (1..=1000)
    .map(|n| n.to_string())
    .collect::<Vec<_>>()
    .join(" ");
  let configured = configured_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("configure");
  let measuring = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");

  let configured_count = configured
    .encode(long.as_str(), true)
    .expect("encode")
    .get_ids()
    .len();
  let measuring_count = measuring
    .encode(long.as_str(), true)
    .expect("encode")
    .get_ids()
    .len();

  assert_eq!(
    configured_count, MAX_TOKENS,
    "the production tokenizer saturates a long input at the window"
  );
  assert!(
    measuring_count > MAX_TOKENS,
    "the measuring tokenizer must see the true (untruncated) count, got {measuring_count}"
  );
}

/// A long document splits into multiple chunks that PARTITION the text under the
/// default (overlap-free) geometry — the first starts at byte 0, each begins
/// where the previous ended, the last ends at `doc.len()` — with every chunk
/// within the token budget. The partition triplet is the coverage regression:
/// pre-repair windit left `\n\n` gaps between chunks, so `chunk.start()` ran
/// strictly ahead of the previous end rather than meeting it.
#[test]
fn long_text_chunks_multi_window_within_budget() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let doc = long_doc();
  let chunks = chunk_long(&mt, &doc, &WindowOptions::new(MAX_TOKENS)).expect("chunk");

  assert!(
    chunks.len() > 1,
    "a document over several windows must split into multiple chunks, got {}",
    chunks.len()
  );
  assert_eq!(chunks[0].start(), 0, "the first chunk starts at byte 0");
  let mut prev_end = 0usize;
  for chunk in &chunks {
    let s = chunk
      .as_str(&doc)
      .expect("chunk falls on a char boundary of its own text");
    let count = mt.encode(s, true).expect("encode").get_ids().len();
    assert!(
      count <= MAX_TOKENS,
      "every chunk stays within the token budget, got {count}"
    );
    assert_eq!(
      chunk.start(),
      prev_end,
      "each chunk begins where the previous ended (no gap, no overlap)"
    );
    prev_end = chunk.end();
  }
  assert_eq!(prev_end, doc.len(), "the last chunk ends at doc.len()");
}

/// Every byte of the document survives chunking, and every paragraph separator
/// stays in the token stream exactly once. windit drops the `\n\n` runs that fall
/// on chunk boundaries; `attach_gaps` reattaches them, so (a) the chunks
/// concatenate back to the document byte-for-byte and (b) the ByteLevel separator
/// token appears once per `\n\n` across the union of the chunk encodings —
/// interior and reattached-boundary separators alike.
#[test]
fn boundary_separators_stay_in_the_token_stream() {
  // `\n\n` tokenizes to `[<|startoftext|>, ĊĊ, <|return|>]`, so id 239 is the
  // paragraph separator's sole content token; counting it counts separators.
  const PARAGRAPH_SEPARATOR_TOKEN: u32 = 239;
  assert_eq!(
    ids("\n\n"),
    vec![179934, PARAGRAPH_SEPARATOR_TOKEN, 179938],
    "the paragraph separator's token id is pinned"
  );

  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let doc = long_doc();
  let chunks = chunk_long(&mt, &doc, &WindowOptions::new(MAX_TOKENS)).expect("chunk");

  let concat: String = chunks
    .iter()
    .map(|c| {
      c.as_str(&doc)
        .expect("chunk falls on a char boundary of its own text")
    })
    .collect();
  assert_eq!(
    concat, doc,
    "the chunks must concatenate back to the document byte-for-byte"
  );

  let separators: usize = chunks
    .iter()
    .map(|c| {
      let s = c.as_str(&doc).expect("char boundary");
      mt.encode(s, true)
        .expect("encode")
        .get_ids()
        .iter()
        .filter(|&&id| id == PARAGRAPH_SEPARATOR_TOKEN)
        .count()
    })
    .sum();
  assert_eq!(
    separators,
    doc.matches("\n\n").count(),
    "every `\\n\\n` is tokenized exactly once across the chunks"
  );
}

/// The word-level fallback (an oversized sentence with no paragraph or sentence
/// break) excludes inter-word punctuation from its chunks; `attach_gaps`
/// reattaches it. One 400-term comma-separated sentence at window 128 partitions
/// into byte-exact chunks — every `", "` preserved, none over budget.
#[test]
fn word_fallback_punctuation_is_reattached() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let sentence = (0..400)
    .map(|w| format!("term{w}"))
    .collect::<Vec<_>>()
    .join(", ");
  let chunks = chunk_long(&mt, &sentence, &WindowOptions::new(128)).expect("chunk");

  assert!(
    chunks.len() > 1,
    "a 400-term sentence must split into multiple chunks, got {}",
    chunks.len()
  );
  assert_eq!(chunks[0].start(), 0, "the first chunk starts at byte 0");
  let mut prev_end = 0usize;
  for chunk in &chunks {
    let s = chunk.as_str(&sentence).expect("char boundary");
    assert_eq!(
      chunk.start(),
      prev_end,
      "each chunk begins where the previous ended"
    );
    assert!(
      mt.encode(s, true).expect("encode").get_ids().len() <= 128,
      "every chunk stays within the 128-token budget"
    );
    prev_end = chunk.end();
  }
  assert_eq!(
    prev_end,
    sentence.len(),
    "the last chunk ends at the text length"
  );

  let concat: String = chunks
    .iter()
    .map(|c| c.as_str(&sentence).expect("char boundary"))
    .collect();
  assert_eq!(
    concat, sentence,
    "the chunks reproduce the sentence byte-for-byte"
  );
}

/// Leading and trailing separators are covered too: a document wrapped in `\n\n`
/// still partitions — the first chunk starts at byte 0 despite the leading
/// separator and the last ends at the text length despite the trailing one
/// (`attach_gaps`' leading and trailing branches).
#[test]
fn leading_and_trailing_separators_are_covered() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let doc = format!("\n\n{}\n\n", long_doc());
  let chunks = chunk_long(&mt, &doc, &WindowOptions::new(MAX_TOKENS)).expect("chunk");

  assert!(
    chunks.len() > 1,
    "the wrapped document still splits, got {}",
    chunks.len()
  );
  assert_eq!(
    chunks[0].start(),
    0,
    "the first chunk starts at 0 despite the leading separator"
  );
  let mut prev_end = 0usize;
  for chunk in &chunks {
    assert_eq!(
      chunk.start(),
      prev_end,
      "each chunk begins where the previous ended"
    );
    prev_end = chunk.end();
  }
  assert_eq!(
    prev_end,
    doc.len(),
    "the last chunk ends at len despite the trailing separator"
  );

  let concat: String = chunks
    .iter()
    .map(|c| c.as_str(&doc).expect("char boundary"))
    .collect();
  assert_eq!(concat, doc, "the chunks reproduce the wrapped document");
}

/// The overflow fallback chain — right-prepend, own-chunk, leading, trailing — is
/// unreachable with the real tokenizer on natural corpora (packed chunks never
/// sit exactly at the window), so pin it with a `char`-count measure that drives
/// `ContentAware` + `attach_gaps` directly. Each windit trace is checked by hand
/// against the pinned rev; each case asserts the exact repaired ranges, which are
/// a partition of the input.
#[test]
fn gap_attachment_falls_back_right_then_own_chunk() {
  use windit::split::ContentAware;

  let measure = |s: &str| -> usize { s.chars().count() };
  // `attach_gaps` now takes a fallible measure; a `char`-count never fails, so
  // the own-chunks (all far below MAX_TOKENS) attach exactly as before.
  let measure_checked = |s: &str| -> Result<usize> { Ok(s.chars().count()) };
  // windit's raw chunks for `text` at `window`, repaired by `attach_gaps`, as
  // (start, end) byte ranges.
  let repair = |text: &str, window: usize| -> Vec<(usize, usize)> {
    let chunks = ContentAware::new(&measure)
      .chunk(text, &WindowOptions::new(window))
      .expect("chunk");
    attach_gaps(text, chunks, &measure_checked, window)
      .expect("own-chunks measure within MAX_TOKENS")
      .iter()
      .map(|c| (c.start(), c.end()))
      .collect()
  };

  let cases: &[(&str, &[(usize, usize)])] = &[
    // Left neighbor full (`aaaaa` = 5); the `\n\n` gap cannot append (`aaaaa\n\n`
    // = 7 > 5) but prepends to the right neighbor, which still fits (`\n\nbbb`
    // = 5). windit: [0,5),[7,10).
    ("aaaaa\n\nbbb", &[(0, 5), (5, 10)]),
    // Both neighbors full; neither can absorb the `\n\n`, so it becomes its own
    // chunk between them. windit: [0,5),[7,12).
    ("aaaaa\n\nbbbbb", &[(0, 5), (5, 7), (7, 12)]),
    // windit's lone chunk [2,7) omits the leading `\n\n` (the 1-chunk coverage
    // hole at micro scale); it cannot prepend (`\n\naaaaa` = 7 > 5), so the
    // leading run is its own chunk.
    ("\n\naaaaa", &[(0, 2), (2, 7)]),
    // The trailing `\n\n` cannot append (`aaaaa\n\n` = 7 > 5), so it is its own
    // chunk. windit: [0,5).
    ("aaaaa\n\n", &[(0, 5), (5, 7)]),
  ];

  for &(text, expected) in cases {
    let got = repair(text, 5);
    assert_eq!(got.as_slice(), expected, "repaired ranges for {text:?}");
    // The exact ranges above are a partition: first start 0, adjacent tiling,
    // last end == text length.
    assert_eq!(got.first().unwrap().0, 0, "{text:?}: first start 0");
    assert_eq!(
      got.last().unwrap().1,
      text.len(),
      "{text:?}: last end == text length"
    );
    for w in got.windows(2) {
      assert_eq!(w[0].1, w[1].0, "{text:?}: adjacent chunks tile");
    }
  }
}

/// Gap repair must not silently defeat the caller's `max_windows` work bound
/// (each chunk is one CoreML prediction): windit's own cap passes pre-repair,
/// but an unabsorbable separator run becomes an extra own-chunk, so the cap is
/// re-enforced on the FINAL chunk count. With the bundled tokenizer at window
/// 3, `a`/`b` pack a window exactly (3 ids with specials) while `a\n\n` /
/// `\n\nb` measure 4, so a `\n\n` between them fits neither neighbor.
#[test]
fn gap_repair_cannot_exceed_max_windows() {
  use windit::WinditError;

  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");

  // windit passes at two content chunks; repair inserts the interior `\n\n` as
  // a third. `got` is the full repaired count.
  match chunk_long(&mt, "a\n\nb", &WindowOptions::new(3).with_max_windows(2)) {
    Err(Error::Windowing(WinditError::TooManyWindows { got, max })) => {
      assert_eq!(got, 3, "the full repaired chunk count is reported");
      assert_eq!(max, 2);
    }
    other => panic!("expected Err(Windowing(TooManyWindows)), got {other:?}"),
  }

  // Uncapped, the same geometry chunks fine — three covering chunks — so the
  // error above is the cap, not the geometry.
  let uncapped = chunk_long(&mt, "a\n\nb", &WindowOptions::new(3)).expect("uncapped");
  let ranges: Vec<_> = uncapped.iter().map(|c| (c.start(), c.end())).collect();
  assert_eq!(ranges, vec![(0, 1), (1, 3), (3, 4)]);

  // Leading, interior, and trailing insertions co-occur and are all counted:
  // windit yields `a` and `b` (2 content chunks, within the cap of 3, so
  // windit's own check passes); repair adds all three `\n\n` runs, so the
  // final count (5) exceeds `max + 1` (4) by one — `got` is the full
  // repaired count, not windit's abort-at-`max + 1` value.
  match chunk_long(
    &mt,
    "\n\na\n\nb\n\n",
    &WindowOptions::new(3).with_max_windows(3),
  ) {
    Err(Error::Windowing(WinditError::TooManyWindows { got, max })) => {
      assert_eq!(
        got, 5,
        "leading + interior + trailing insertions all counted"
      );
      assert_eq!(max, 3);
    }
    other => panic!("expected Err(Windowing(TooManyWindows)), got {other:?}"),
  }
}

/// Contentless (whitespace-only) nonempty text chunks to no content, yet
/// embedding it still costs one whole-input CoreML prediction; the cap must
/// see that cost. Cap 0 refuses before any model work with the true count;
/// cap 1 (and no cap) admits it as a single whole-input chunk — full
/// coverage, one prediction.
#[test]
fn whitespace_only_text_counts_one_window_against_the_cap() {
  use windit::WinditError;

  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");

  match chunk_long(
    &mt,
    "   ",
    &WindowOptions::new(MAX_TOKENS).with_max_windows(0),
  ) {
    Err(Error::Windowing(WinditError::TooManyWindows { got, max })) => {
      assert_eq!(got, 1, "the whole-input fallback counts as one window");
      assert_eq!(max, 0);
    }
    other => panic!("expected Err(Windowing(TooManyWindows)), got {other:?}"),
  }

  let capped = chunk_long(
    &mt,
    "   ",
    &WindowOptions::new(MAX_TOKENS).with_max_windows(1),
  )
  .expect("cap 1 admits the whole-input fallback");
  assert_eq!(
    capped
      .iter()
      .map(|c| (c.start(), c.end()))
      .collect::<Vec<_>>(),
    vec![(0, 3)],
    "one chunk spanning the whole input"
  );

  let uncapped =
    chunk_long(&mt, "   ", &WindowOptions::new(MAX_TOKENS)).expect("uncapped whitespace");
  assert_eq!(
    uncapped
      .iter()
      .map(|c| (c.start(), c.end()))
      .collect::<Vec<_>>(),
    vec![(0, 3)],
    "the fallback chunk is synthesized regardless of any cap"
  );
}

/// Contentless over-budget input is REFUSED, not silently truncated. windit
/// drops the whitespace-only text; the whole-input fallback would then embed it
/// through the truncating production tokenizer, dropping every token past the
/// 512-window. The measured fallback refuses it instead. Fixtures span spaces,
/// tabs, CRLF, NBSP, and a mixed run (em/thin spaces); `tokens` is compared to
/// the test's own untruncated encode so the pin is self-consistent under any
/// tokenizer.
#[test]
fn contentless_over_budget_input_is_refused_not_truncated() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let fixtures = [
    " ".repeat(100_000),
    "\t".repeat(100_000),
    "\r\n".repeat(50_000),
    "\u{00A0}".repeat(100_000),
    " \t\r\n\u{00A0}\u{2003}\u{2009}".repeat(15_000),
  ];
  for s in &fixtures {
    let expected_tokens = mt.encode(s.as_str(), true).expect("encode").get_ids().len();
    assert!(
      expected_tokens > MAX_TOKENS,
      "fixture must actually exceed the window (got {expected_tokens})"
    );
    match chunk_long(&mt, s, &WindowOptions::new(MAX_TOKENS)) {
      Err(Error::ContentlessInputOverBudget {
        start,
        end,
        tokens,
        max,
      }) => {
        assert_eq!(start, 0, "the whole input is the offending run");
        assert_eq!(end, s.len());
        assert_eq!(
          tokens, expected_tokens,
          "reported count is the untruncated measure"
        );
        assert_eq!(max, MAX_TOKENS);
      }
      other => panic!("expected ContentlessInputOverBudget, got {other:?}"),
    }
  }
}

/// The at-budget boundary is embedded whole; the first over-budget count is
/// refused. Binary-searches the largest space run that fits the window, so the
/// boundary is exact regardless of the pinned tokenizer.
#[test]
fn contentless_input_at_or_under_budget_still_embeds_whole() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let measure = |n: usize| {
    mt.encode(" ".repeat(n).as_str(), true)
      .expect("encode")
      .get_ids()
      .len()
  };
  let mut lo = 1usize;
  let mut hi = 100_000usize;
  assert!(measure(lo) <= MAX_TOKENS, "one space fits");
  assert!(measure(hi) > MAX_TOKENS, "100k spaces overflow");
  while lo + 1 < hi {
    let mid = (lo + hi) / 2;
    if measure(mid) <= MAX_TOKENS {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  // `lo` is the largest in-budget count; `hi == lo + 1` the first over-budget.
  let at_budget = " ".repeat(lo);
  let chunks = chunk_long(&mt, &at_budget, &WindowOptions::new(MAX_TOKENS))
    .expect("in-budget contentless input embeds whole");
  assert_eq!(
    chunks
      .iter()
      .map(|c| (c.start(), c.end()))
      .collect::<Vec<_>>(),
    vec![(0, at_budget.len())],
    "in-budget contentless input is one whole-input chunk"
  );
  let over = " ".repeat(hi);
  match chunk_long(&mt, &over, &WindowOptions::new(MAX_TOKENS)) {
    Err(Error::ContentlessInputOverBudget { .. }) => {}
    other => panic!("expected ContentlessInputOverBudget just past the budget, got {other:?}"),
  }
}

/// A pure-separator gap between two content chunks that neither neighbor can
/// absorb (the `attach_gaps` own-chunk escape, interior case) is refused when
/// its run measures past the window. `a<100k spaces>b` at window 3 forces the
/// escape; the in-budget own-chunk escape (`a\n\nb`) still chunks Ok.
#[test]
fn separator_gap_over_budget_is_refused() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let text = format!("a{}b", " ".repeat(100_000));
  match chunk_long(&mt, &text, &WindowOptions::new(3)) {
    Err(Error::ContentlessInputOverBudget {
      start,
      end,
      tokens,
      max,
    }) => {
      assert_eq!(start, 1, "the gap starts right after `a`");
      assert_eq!(end, 100_001, "the gap ends right before `b`");
      assert!(tokens > MAX_TOKENS, "the gap run measures over the window");
      assert_eq!(max, MAX_TOKENS);
    }
    other => panic!("expected ContentlessInputOverBudget, got {other:?}"),
  }
  // Control: an in-budget own-chunk gap still chunks Ok — the escape stays.
  let ok = chunk_long(&mt, "a\n\nb", &WindowOptions::new(3)).expect("in-budget own-chunk escape");
  assert_eq!(
    ok.iter().map(|c| (c.start(), c.end())).collect::<Vec<_>>(),
    vec![(0, 1), (1, 3), (3, 4)]
  );
}

/// Leading and trailing pure-separator gaps that measure past the window are
/// refused with the correct byte span (the `attach_gaps` leading and trailing
/// branches, measuring the GAP itself, not the extended candidate).
#[test]
fn leading_and_trailing_over_budget_gaps_are_refused() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");

  let leading = format!("{}a", " ".repeat(100_000));
  match chunk_long(&mt, &leading, &WindowOptions::new(MAX_TOKENS)) {
    Err(Error::ContentlessInputOverBudget { start, end, .. }) => {
      assert_eq!(start, 0, "leading gap starts at byte 0");
      assert_eq!(end, 100_000, "leading gap ends right before `a`");
    }
    other => panic!("expected leading ContentlessInputOverBudget, got {other:?}"),
  }

  let trailing = format!("a{}", " ".repeat(100_000));
  match chunk_long(&mt, &trailing, &WindowOptions::new(MAX_TOKENS)) {
    Err(Error::ContentlessInputOverBudget { start, end, .. }) => {
      assert_eq!(start, 1, "trailing gap starts right after `a`");
      assert_eq!(end, 100_001, "trailing gap ends at text length");
    }
    other => panic!("expected trailing ContentlessInputOverBudget, got {other:?}"),
  }
}

/// An encode failure during the fallback measure surfaces as `Error::Tokenize`,
/// NOT a bogus `ContentlessInputOverBudget { tokens: usize::MAX }`: the fallible
/// measure closure must preserve the tokenizer error's identity. A tiny
/// WordLevel tokenizer whose unk token is absent from its vocab fails to encode
/// the whitespace-only fallback input.
#[test]
fn tokenizer_failure_on_fallback_measure_keeps_tokenize_identity() {
  // WordLevel, no pre-tokenizer, unk token `<unk>` not in vocab: encoding the
  // whole-string `"   "` (not in vocab) errors instead of truncating.
  const TINY_NO_UNK: &[u8] = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"world":1},"unk_token":"<unk>"}}"#;
  let mt = tokenizers::Tokenizer::from_bytes(TINY_NO_UNK).expect("load tiny WordLevel");
  match chunk_long(&mt, "   ", &WindowOptions::new(MAX_TOKENS)) {
    Err(Error::Tokenize(_)) => {}
    other => panic!("expected Err(Tokenize), got {other:?}"),
  }
}

/// `""` costs zero predictions (`embed_long_with` delegates it to `embed`,
/// which fails `EmptyText` before the model), so no fallback chunk is
/// synthesized and even a cap of 0 passes chunking.
#[test]
fn empty_text_chunks_to_nothing_under_any_cap() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let chunks = chunk_long(&mt, "", &WindowOptions::new(MAX_TOKENS).with_max_windows(0))
    .expect("empty text chunks to nothing under a cap of 0");
  assert!(chunks.is_empty());
}

/// A short text that fits one window is a single chunk spanning the whole text.
#[test]
fn single_window_text_is_one_whole_chunk() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let text = "a compact sentence that fits comfortably inside one window";
  let chunks = chunk_long(&mt, text, &WindowOptions::new(MAX_TOKENS)).expect("chunk");
  assert_eq!(chunks.len(), 1, "short text is one chunk");
  assert_eq!(chunks[0].start(), 0);
  assert_eq!(chunks[0].end(), text.len());
}

/// The chunk geometry adapts to `WindowOptions` alone (the spec's genericity at
/// granite's seam): a smaller window yields more, smaller chunks, each within its
/// own budget.
#[test]
fn chunk_geometry_adapts_by_window_options_alone() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let doc = long_doc();
  let coarse = chunk_long(&mt, &doc, &WindowOptions::new(128)).expect("chunk @128");
  let fine = chunk_long(&mt, &doc, &WindowOptions::new(64)).expect("chunk @64");

  assert!(
    fine.len() > coarse.len(),
    "a smaller window yields more chunks: {} @64 vs {} @128",
    fine.len(),
    coarse.len()
  );
  for chunk in &coarse {
    let s = chunk.as_str(&doc).expect("char boundary");
    assert!(mt.encode(s, true).expect("encode").get_ids().len() <= 128);
  }
  for chunk in &fine {
    let s = chunk.as_str(&doc).expect("char boundary");
    assert!(mt.encode(s, true).expect("encode").get_ids().len() <= 64);
  }
}

/// With a non-zero overlap, consecutive chunks repeat a trailing region whose
/// measured length stays within the overlap token budget.
#[test]
fn overlap_repeats_trailing_tokens_within_budget() {
  let mt = measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring");
  let doc = long_doc();
  let opts = WindowOptions::new(128).with_overlap(16);
  let chunks = chunk_long(&mt, &doc, &opts).expect("chunk");

  assert!(chunks.len() > 1, "an overlapped long doc still splits");
  for pair in chunks.windows(2) {
    // Consecutive chunks share a trailing region…
    assert!(
      pair[1].start() < pair[0].end(),
      "consecutive chunks overlap: next start {} vs prev end {}",
      pair[1].start(),
      pair[0].end()
    );
    // …and that repeated text measures within the overlap budget (the exact text
    // the packer measured, with special tokens, is `<= 16`).
    let repeated = &doc[pair[1].start()..pair[0].end()];
    let n = mt.encode(repeated, true).expect("encode").get_ids().len();
    assert!(
      n <= 16,
      "repeated region within the 16-token overlap budget, got {n}"
    );
  }
}

/// `validate_long_input` (and thus `embed_long_with`) rejects a per-chunk budget
/// above the model's fixed window before any tokenization — hermetically, no
/// model.
#[test]
fn window_over_budget_is_rejected() {
  match validate_long_input(
    "any text",
    &LongTextOptions::from(WindowOptions::new(MAX_TOKENS + 1)),
  ) {
    Err(Error::WindowOverBudget { window, max }) => {
      assert_eq!(window, MAX_TOKENS + 1);
      assert_eq!(max, MAX_TOKENS);
    }
    other => panic!("expected Err(WindowOverBudget), got {other:?}"),
  }
  // The exact budget is accepted.
  assert!(
    validate_long_input(
      "any text",
      &LongTextOptions::from(WindowOptions::new(MAX_TOKENS)),
    )
    .is_ok()
  );
}

// ── #2: LongTextOptions + input-size gate (hermetic) ─────────────────────────

/// `Default` == `new`, the documented defaults (full window, no byte limit), the
/// builder/setter round-trips, and the `From<WindowOptions>` geometry-only form.
#[test]
fn long_text_options_default_equals_new() {
  assert_eq!(LongTextOptions::default(), LongTextOptions::new());
  assert_eq!(
    LongTextOptions::new().window_options(),
    WindowOptions::new(MAX_TOKENS)
  );
  assert_eq!(LongTextOptions::new().max_input_bytes(), None);

  let built = LongTextOptions::new()
    .with_window_options(WindowOptions::new(64))
    .with_max_input_bytes(4096);
  assert_eq!(built.window_options(), WindowOptions::new(64));
  assert_eq!(built.max_input_bytes(), Some(4096));

  let mut set = LongTextOptions::new();
  set.set_window_options(WindowOptions::new(32));
  set.set_max_input_bytes(2048);
  assert_eq!(set.window_options(), WindowOptions::new(32));
  assert_eq!(set.max_input_bytes(), Some(2048));

  let from = LongTextOptions::from(WindowOptions::new(64));
  assert_eq!(from.window_options().window(), 64);
  assert_eq!(from.max_input_bytes(), None);
}

/// The input-size gate refuses an oversized input reading only `text.len()` (no
/// tokenizer); at-limit passes (`>` rejects, `==` accepts), and a `None` limit
/// accepts the same oversized input.
#[test]
fn input_too_large_is_rejected_before_any_tokenizer_work() {
  let big = "x".repeat(8 * 1024 * 1024);
  let opts = LongTextOptions::new().with_max_input_bytes(1024 * 1024);
  match validate_long_input(&big, &opts) {
    Err(Error::InputTooLarge { got, max }) => {
      assert_eq!(got, big.len());
      assert_eq!(max, 1024 * 1024);
    }
    other => panic!("expected InputTooLarge, got {other:?}"),
  }
  // At-limit is accepted.
  let at_limit = "x".repeat(1024 * 1024);
  assert!(validate_long_input(&at_limit, &opts).is_ok());
  // No limit accepts the same 8 MiB input.
  assert!(validate_long_input(&big, &LongTextOptions::new()).is_ok());
}

/// The untrusted-input gate is the outermost shield: an oversized input AND an
/// over-budget window yields `InputTooLarge`, not `WindowOverBudget`.
#[test]
fn input_too_large_takes_precedence_over_window_budget() {
  let big = "x".repeat(2 * 1024 * 1024);
  let opts =
    LongTextOptions::from(WindowOptions::new(MAX_TOKENS + 1)).with_max_input_bytes(1024 * 1024);
  match validate_long_input(&big, &opts) {
    Err(Error::InputTooLarge { .. }) => {}
    other => panic!("expected InputTooLarge to win over WindowOverBudget, got {other:?}"),
  }
}

// ── #6: tokenizer contract validation (hermetic) ─────────────────────────────

/// Parse the bundled tokenizer JSON, apply `mutate`, re-serialize, and build the
/// module's CONFIGURED tokenizer from the result — so a contract check sees
/// exactly the production tokenization. `serde_json` is a dev-dependency.
fn mutated_bundled_tokenizer(mutate: impl FnOnce(&mut serde_json::Value)) -> Tokenizer {
  let mut value: serde_json::Value =
    serde_json::from_slice(BUNDLED_TOKENIZER).expect("parse bundled tokenizer.json");
  mutate(&mut value);
  let bytes = serde_json::to_vec(&value).expect("re-serialize mutated tokenizer.json");
  configured_tokenizer_from_bytes(&bytes).expect("configure mutated tokenizer")
}

/// The bundled granite tokenizer passes every contract check — the invariant on
/// which all keep-green constructor/golden behavior rests. If this fails the
/// contract constants are wrong; strengthen the constants, never the checks.
#[test]
fn tokenizer_contract_accepts_the_bundled_tokenizer() {
  let tok = configured_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("configure bundled");
  validate_tokenizer_contract(&tok).expect("bundled tokenizer must satisfy the contract");
}

/// A tiny foreign tokenizer with none of granite's specials fails the first
/// check (`<|startoftext|>`), reported as `missing`.
#[test]
fn tokenizer_contract_rejects_missing_specials() {
  const TINY: &[u8] = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"world":1},"unk_token":"<unk>"}}"#;
  let tok = configured_tokenizer_from_bytes(TINY).expect("configure tiny");
  match validate_tokenizer_contract(&tok) {
    Err(Error::TokenizerContractMismatch { check, actual, .. }) => {
      assert!(
        check.contains("<|startoftext|>"),
        "check names the missing special: {check}"
      );
      assert_eq!(actual, "missing");
    }
    other => panic!("expected TokenizerContractMismatch, got {other:?}"),
  }
}

/// The bundled tokenizer with one trailing (non-special) added token removed
/// keeps granite's three specials but drops the vocabulary to 179999, so the
/// vocab-size check fires. The pinned `tokenizers` reassigns added-token ids
/// densely as `base + array_index` (ignoring the JSON `id` field), and the three
/// specials are the array's first entries, so removing the LAST entry leaves
/// their ids intact — the fixture edits structure, not a declared id.
#[test]
fn tokenizer_contract_rejects_wrong_vocab_size() {
  let tok = mutated_bundled_tokenizer(|value| {
    let added = value["added_tokens"]
      .as_array_mut()
      .expect("added_tokens array");
    added.pop().expect("added_tokens is non-empty");
  });
  match validate_tokenizer_contract(&tok) {
    Err(Error::TokenizerContractMismatch {
      check,
      expected,
      actual,
    }) => {
      assert_eq!(check, "vocab size");
      assert!(
        expected.contains("180000"),
        "expected names the contract size: {expected}"
      );
      assert!(
        actual.contains("179999"),
        "actual names the reduced size: {actual}"
      );
    }
    other => panic!("expected TokenizerContractMismatch on vocab size, got {other:?}"),
  }
}

/// The bundled tokenizer with its highest BASE-vocab id pushed past the model's
/// table (to 180000) keeps the count at 180000 and the specials intact, so the
/// specials and vocab-size checks pass but the max-token-id gate fires — the
/// out-of-vocabulary case a larger foreign tokenizer would hit. (Added-token ids
/// are reassigned densely by this `tokenizers`, so an OOV id can only come from
/// the base vocab; the fixture moves a base id, leaving a hole the count check
/// tolerates.)
#[test]
fn tokenizer_contract_rejects_out_of_model_vocab_id() {
  let tok = mutated_bundled_tokenizer(|value| {
    let vocab = value["model"]["vocab"]
      .as_object_mut()
      .expect("model.vocab object");
    let key = vocab
      .iter()
      .max_by_key(|(_, id)| id.as_u64().unwrap_or(0))
      .map(|(k, _)| k.clone())
      .expect("non-empty base vocab");
    vocab.insert(key, serde_json::json!(180_000));
  });
  match validate_tokenizer_contract(&tok) {
    Err(Error::TokenizerContractMismatch { check, actual, .. }) => {
      assert_eq!(check, "max token id");
      assert!(
        actual.contains("180000"),
        "actual carries the offending id: {actual}"
      );
    }
    other => panic!("expected TokenizerContractMismatch on max token id, got {other:?}"),
  }
}

/// Two base-vocab entries with their ids swapped (`hello`↔`Ġworld`, 24313↔2318)
/// leave the specials, vocab size, and max id intact but change the sentinel
/// encoding, so only the final check fires.
#[test]
fn tokenizer_contract_rejects_divergent_encoding() {
  let tok = mutated_bundled_tokenizer(|value| {
    let vocab = value["model"]["vocab"]
      .as_object_mut()
      .expect("model.vocab object");
    let mut key_a = None;
    let mut key_b = None;
    for (key, id) in vocab.iter() {
      match id.as_u64() {
        Some(24_313) => key_a = Some(key.clone()),
        Some(2_318) => key_b = Some(key.clone()),
        _ => {}
      }
    }
    let key_a = key_a.expect("id 24313 present in base vocab");
    let key_b = key_b.expect("id 2318 present in base vocab");
    vocab.insert(key_a, serde_json::json!(2_318));
    vocab.insert(key_b, serde_json::json!(24_313));
  });
  match validate_tokenizer_contract(&tok) {
    Err(Error::TokenizerContractMismatch { check, .. }) => {
      assert_eq!(check, "sentinel encoding");
    }
    other => panic!("expected TokenizerContractMismatch on sentinel encoding, got {other:?}"),
  }
}
