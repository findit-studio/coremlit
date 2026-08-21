//! Layer-1 differential: [`TokenIndex::measure_range`] must equal
//! `encode(&text[a..b], true).len()` EXACTLY, for every range windit or
//! `attach_gaps` can present. This is the merge gate for output identity — a
//! measure that shifted a window boundary reds here deterministically.
//!
//! Coverage: every committed multilingual golden text + a battery of
//! class-targeted adversarial strings (digit runs, NBSP/CRLF/tab/thin-space
//! whitespace runs, emoji-ZWJ families, combining sequences, apostrophe suffixes,
//! vulgar fractions), each cut at (i) all pre-token boundary pairs where the count
//! is small, (ii) all `(0, b)` / `(a, len)` hot-path shapes, and (iii) thousands
//! of seeded-random `char`-boundary pairs.

use tokenizers::Tokenizer;

use super::TokenIndex;
use crate::embeddings::granite::{measuring_tokenizer_from_bytes, test_artifact};

/// The truncation-disabled MEASURING tokenizer — the exact one `chunk_long` builds
/// its index with, from the granite `tokenizer.json` the ARTIFACT carries (the
/// crate embeds none). Every test that calls this is `#[ignore]`d on that staged
/// artifact.
fn measuring_tok() -> Tokenizer {
  measuring_tokenizer_from_bytes(test_artifact::tokenizer_bytes()).expect("measuring tokenizer")
}

/// The oracle: the count the OLD per-call closure returned.
fn oracle(tok: &Tokenizer, s: &str) -> usize {
  tok
    .encode(s, true)
    .expect("encode substring")
    .get_ids()
    .len()
}

/// Deterministic SplitMix64 — a seeded RNG with no new dependency.
struct Rng(u64);
impl Rng {
  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }
  fn below(&mut self, n: usize) -> usize {
    (self.next_u64() % (n as u64)) as usize
  }
}

/// Every byte offset that is a `char` boundary of `text`, `0..=len` inclusive.
fn char_boundaries(text: &str) -> Vec<usize> {
  (0..=text.len())
    .filter(|&i| text.is_char_boundary(i))
    .collect()
}

/// Assert `measure_range == oracle` at one `(a, b)`.
#[track_caller]
fn check(index: &TokenIndex, tok: &Tokenizer, text: &str, a: usize, b: usize) {
  let got = index
    .measure_range(tok, text, a, b)
    .expect("measure_range must not fail on the granite tokenizer");
  let want = oracle(tok, &text[a..b]);
  assert_eq!(
    got,
    want,
    "measure_range({a}, {b}) = {got} but encode({:?}) = {want} (full text {:?})",
    &text[a..b],
    text
  );
}

/// The full differential over one text: exhaustive boundary pairs when small (the
/// adversarial battery and most corpus entries), or sampled hot-path + random
/// pairs when large (the slow oracle is bounded).
fn differential_over(tok: &Tokenizer, text: &str, rng: &mut Rng) {
  if text.is_empty() {
    return;
  }
  let index = TokenIndex::build(tok, text).expect("build index");
  let bounds = char_boundaries(text);
  let m = bounds.len();

  // Small strings: EXHAUSTIVE over every boundary pair — the class-targeted
  // adversarial strings and short goldens are covered completely.
  if m <= 96 {
    for i in 0..m {
      for j in (i + 1)..m {
        check(&index, tok, text, bounds[i], bounds[j]);
      }
    }
    return;
  }

  // Large strings: sample the hot-path `(0, b)` / `(a, len)` shapes (pack's
  // growing prefix and the leading/trailing/whole-input repairs) plus seeded
  // random pairs. The oracle re-encodes multi-KB substrings, so the count is
  // bounded rather than exhaustive.
  for _ in 0..400 {
    let b = bounds[rng.below(m)];
    if b > 0 {
      check(&index, tok, text, 0, b);
    }
    let a = bounds[rng.below(m)];
    if a < text.len() {
      check(&index, tok, text, a, text.len());
    }
  }
  for _ in 0..1_500 {
    let mut a = bounds[rng.below(m)];
    let mut b = bounds[rng.below(m)];
    if a == b {
      continue;
    }
    if a > b {
      std::mem::swap(&mut a, &mut b);
    }
    check(&index, tok, text, a, b);
  }
}

/// Class-targeted adversarial strings: each exercises a specific dirty-zone rule
/// or an unenumerated boundary class the fallback must still get exactly right.
const ADVERSARIAL: &[&str] = &[
  // Digit triplet re-anchoring (the `\p{N}{1,3}` branch, left rule).
  "1234567890",
  "a1234567890b",
  "192.168.100.254",
  "Order #A1234-99 shipped 2026-07-18 at 09:41:59",
  "1000000 2000000 3000000 phone 555-0142 zip 94107-1234",
  // Whitespace-run lookahead (`\s+(?!\S)`, right rule) across every ws flavour.
  "a   b   c",
  "a\t\t\tb\tc",
  "a\r\nb\r\n\r\nc",
  "word\n\n\nword\n\nword",
  "a\u{00A0}\u{00A0}\u{00A0}b",
  "a\u{2009}\u{2009}\u{2009}b",
  "mix \u{00A0}\t \u{2003}\u{2009}  run x",
  "half\u{3000}width\u{3000}\u{3000}ideographic",
  "   leading and trailing   ",
  "   ",
  "\n\n",
  // Word branches: leading-char attachment + apostrophe suffixes.
  "can't won't I'll we've they're it's",
  " spaced words with  double  gaps ",
  // Combining marks (`\p{M}` joins its word).
  "cafe\u{0301} na\u{0308}ive re\u{0301}sume\u{0301}",
  // Emoji + ZWJ families and variation selectors (unenumerated class → fallback).
  "best 🍕🍅🧀 crust 🌿 done",
  "👍🏽 family 👨\u{200D}👩\u{200D}👧\u{200D}👦 flag 🏳️\u{200D}🌈 end",
  // Vulgar fractions (category No) and mixed scripts.
  "x½ + ¼ = ¾ y² z₃",
  "使用 transformers 库 for retrieval 检索 テスト 테스트",
  "Café — naïve façade; ½ + ¼ = ¾. \"Quotes\" & <tags> and\ttabs.",
];

/// The committed multilingual golden texts (`corpus.json` `entries[].text`),
/// parsed hermetically (`serde_json` is a dev-dependency).
fn golden_texts() -> Vec<String> {
  const CORPUS: &str = include_str!("../../../../tests/granite/fixtures/goldens/corpus.json");
  let v: serde_json::Value = serde_json::from_str(CORPUS).expect("parse corpus.json");
  v["entries"]
    .as_array()
    .expect("entries array")
    .iter()
    .map(|e| e["text"].as_str().expect("entry text").to_string())
    .collect()
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn measure_range_matches_encode_over_adversarial_corpus() {
  let tok = measuring_tok();
  let mut rng = Rng(0x0DDB_1A5E_5EED_1234);
  for &text in ADVERSARIAL {
    differential_over(&tok, text, &mut rng);
  }
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn measure_range_matches_encode_over_goldens() {
  let tok = measuring_tok();
  let mut rng = Rng(0xF00D_CAFE_1357_9BDF);
  for text in golden_texts() {
    differential_over(&tok, &text, &mut rng);
  }
}

/// A long single-alphabet-mixed document (the `long_document` shape) with digit
/// storms and whitespace pathologies interleaved, cut at thousands of random
/// boundaries — the density the corpus alone does not reach.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn measure_range_matches_encode_over_seeded_random_document() {
  let tok = measuring_tok();
  let mut rng = Rng(0xABCD_1234_5678_9EF0);
  let mut doc = String::new();
  for p in 0..40u32 {
    for w in 0..12u32 {
      doc.push_str(&format!("para{p}word{w} "));
    }
    doc.push_str(&format!(
      "{}.{}.{}.{} ",
      p,
      p * 7,
      p * 13 % 256,
      p * 251 % 256
    ));
    doc.push_str("can't 你好 café\u{0301} 🍕 ");
    doc.push_str("\n\n");
  }
  differential_over(&tok, &doc, &mut rng);
}

/// RED-FIRST (non-vacuity): the equality assert catches a boundary shift. For
/// witnesses spanning each dirty class, shifting the cut by one `char` changes the
/// true token count, so any measure that placed the boundary one char off — the
/// exact defect the single-pass rewrite risks — would red the differential above.
/// `measure_range` returns the un-shifted truth at each witness.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn differential_is_load_bearing_against_a_shifted_boundary() {
  let tok = measuring_tok();
  // (text, a, b): a one-char right shift of `b` must change the encoded count.
  let witnesses: &[(&str, usize, usize)] = &[
    ("a b c", 0, 1),         // "a" (3) vs "a " (4): the trailing space adds a token
    ("1234567890", 0, 6),    // "123456" (2 triplets) vs "1234567" (3): triplet count
    ("hello world", 0, 5),   // "hello" vs "hello " (trailing space adds a token)
    ("return (a, b)", 0, 6), // "return" vs "return " — the punct-attach neighbourhood
  ];
  for &(text, a, b) in witnesses {
    let index = TokenIndex::build(&tok, text).expect("build");
    let truth = oracle(&tok, &text[a..b]);
    // measure_range returns the truth at the correct boundary…
    assert_eq!(index.measure_range(&tok, text, a, b).unwrap(), truth);
    // …and the next char boundary yields a DIFFERENT count, so the equality
    // assert is not vacuous: a one-char boundary shift is observable.
    let b2 = (b + 1..=text.len())
      .find(|&i| text.is_char_boundary(i))
      .expect("a right neighbour boundary exists");
    let shifted = oracle(&tok, &text[a..b2]);
    assert_ne!(
      truth, shifted,
      "witness {text:?} [{a},{b}) vs [{a},{b2}) must differ for the differential to bite"
    );
  }
}

/// The `direct_only` fail-safe answers exactly (by full substring encode), so even
/// if reconstruction were ever rejected the measures stay correct — only slower.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn direct_only_fallback_is_still_exact() {
  let tok = measuring_tok();
  let mut rng = Rng(0x1122_3344_5566_7788);
  for &text in &[
    "hello world foo bar",
    "1234567 abc\u{00A0}\u{00A0}def",
    "a\n\nb",
  ] {
    // Force the fail-safe: build normally, then rebuild a direct-only twin.
    let real = TokenIndex::build(&tok, text).expect("build");
    let direct = TokenIndex {
      pretoken_ends: Vec::new(),
      count_prefix: vec![0],
      digit: Vec::new(),
      direct_only: true,
    };
    let bounds = char_boundaries(text);
    for _ in 0..200 {
      let m = bounds.len();
      let mut a = bounds[rng.below(m)];
      let mut b = bounds[rng.below(m)];
      if a == b {
        continue;
      }
      if a > b {
        std::mem::swap(&mut a, &mut b);
      }
      let want = oracle(&tok, &text[a..b]);
      assert_eq!(real.measure_range(&tok, text, a, b).unwrap(), want);
      assert_eq!(direct.measure_range(&tok, text, a, b).unwrap(), want);
    }
  }
}

// ── F7: the killer classes Fable's adversarial differential surfaced ──────
//
// The layer-1/2 corpora above miss every class that can actually dissolve a
// full-parse boundary under a cut: their whitespace probe `"a   b   c"` has runs
// GLUED by the surrounding letters (branch-1's leading ` ?` pulls the run's last
// space into the next word), so it can never diverge. These generators add the
// classes that CAN — swept over every `char`-boundary pair they reproduce the
// ~500 pre-fix divergences and MUST now be zero. K1-K4 keep the INDEX path live (no
// dropped byte, no added literal), so the big cross-class join built from them
// exercises the fast path; the byte-coverage FALLBACK classes live in
// `DROPPABLE_KILLERS`, and the added-token class in the F9/K6 tests below.
//   K1  a whitespace run split by a NON-glue follower (digit/punct/emoji): the
//       forward glue is narrow — the word branches pull a single non-CR/LF
//       whitespace into a following letter/mark, the punct branch only a literal
//       ' ', a digit nothing — so a tab/NBSP/NEL/thin/ideographic run before a
//       digit, symbol, or emoji is split by the full parse yet merged by
//       `\s+(?!\S)` at end-of-substring.
//   K2  the punct branch's `[\r\n/]*` tail folds CRLFs into a symbol pre-token, so
//       a whitespace back-scan can land mid-pre-token.
//   K3  a contraction suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)` whose letters rejoin
//       the following word once the left context is cut.
//   K4  combining / Other_Alphabetic marks (`\p{M}`, and Mn/Mc the regex glues but
//       `is_alphanumeric` does not).
const KILLERS: &[&str] = &[
  // K1 — whitespace run split by a non-glue follower.
  "456  1",
  "a  1",
  "12  34  56",
  "x   9",
  "a\t\t9",
  "a\u{00A0}\u{00A0}9",
  "a\u{0085}\u{0085}9",
  "a\u{2009}\u{2009}9",
  "a  !",
  "a\t\t.",
  "a\u{00A0}\u{00A0}#",
  "a  🍕",
  "b\u{2009}\u{2009}🌿",
  "a  b",
  "a   b   c",
  "a\t\tb",
  "a\u{2003}\u{2003}b",
  "a\u{3000}\u{3000}b",
  "1 2  3   4",
  "n\u{00A0}9\u{00A0}\u{00A0}8",
  // K2 — punct `[\r\n/]*` tail folds CRLF into a symbol pre-token.
  "a!\r\n\r\n Next t",
  "x.\r\n\r\ny",
  "p!\r\nq",
  "u/\r\n/v",
  "end.\r\n\r\n\r\nStart here",
  "a?!\r\n\r\nB",
  "a!\r\n1",
  "a!\r\n  b",
  "a!\r\n\r\nb",
  // K3 — contraction suffix letters rejoin the next word once cut.
  " it'station end",
  "can'ther",
  "we'reunited now",
  "I'lloop back",
  "he'daily",
  "you'venue",
  "she'small",
  "It'STELLAR",
  // K4 — combining / Other_Alphabetic marks.
  "cafe\u{0301}s here",
  "a\u{0345}b",
  "क\u{093E}ख",
  "ন\u{09BE}দ",
  "a\u{05B4}b",
  "a\u{064B}c",
  "re\u{0301}sume\u{0301} now",
  // Digit runs across scripts (`\p{N}{1,3}` triplet re-anchoring).
  "1234567890",
  "٠١٢٣٤٥٦٧٨٩",
  "０１２３４５６７８９",
  "a1234567890b",
  "192.168.100.254",
  "v1234\u{0660}\u{0661}\u{0662}z",
  // Emoji-ZWJ families / variation selectors, glue-punct siblings, slashes/URLs.
  "👨\u{200D}👩\u{200D}👧\u{200D}👦x",
  "🏳️\u{200D}🌈y",
  "a👍🏽b",
  "!!!???...",
  "a,,,b",
  "(()){}[]",
  "http://a/b/c/d",
  "/usr/local/bin",
  "a//b//c",
  "x.\r\n/y/z",
  // 4-byte chars with a 0xF3 lead (TAG plane 14, PUA-A plane 15) are PRESENT in the
  // vocab → NO drop; the index path stays live and must measure exactly (the
  // correction to the wrong "0xF1-0xF4 all drop" claim — 0xF3 is in the vocab).
  "tag\u{E0067}\u{E0067}\nﬀ\u{5b4}7", // plane-14 TAG chars (0xF3 lead) + ligature+mark
];

/// K5 — vocab-DROPPED byte-level chars. Each string carries a char whose single-byte
/// ByteLevel token is MISSING from the pinned BPE (`unk_token: null`), so
/// `tokenizers` silently DROPS it and mis-attributes the surviving tokens' offsets;
/// `TokenIndex::build` chains those into `ends` that stay monotone, covering, and
/// char-aligned (every tiling check passes), so only the byte-coverage guard catches
/// it → the `direct_only` fallback (exact by direct encode). The missing single-byte
/// set is NUL 0x00, the C0 controls 0x04-0x07 / 0x0B / 0x0C / 0x0E-0x1A / 0x1C-0x1F
/// (NOT 0x01-0x03 / 0x08-0x0A / 0x0D / 0x1B / 0x7F, which are present), and the
/// 4-byte lead bytes 0xC0 / 0xC1 and 0xF1 / 0xF2 / 0xF4-0xFF (0xF3 — planes 12-15,
/// incl. TAG/PUA — IS present). Each dropped char sits inside a piece with OTHER
/// surviving tokens (a LONE dropped char is a zero-token piece the word-id-gap check
/// already catches).
const DROPPABLE_KILLERS: &[&str] = &[
  // C0 controls (the byte-coverage witnesses).
  "\u{c}\n\u{2003}\u{202f}\nﬃ\u{5b4}\u{64b}23", // FF leads a ws run (minimal witness)
  "\u{b}\n\u{2003}\u{202f}\nﬃ\u{5b4}\u{64b}23", // VT variant
  "a\u{c}\n\u{2003}word 12",
  "x\u{b}\ny\u{2009}\u{2009}9",
  "p\u{0}q\nﬃ\u{5b4}z8",       // NUL inside a run
  "m\u{7}\n\u{2003}n\u{64b}5", // BEL (a 0x04-0x1F C0 control)
  "\u{c}\u{c}\n\u{3000}\u{3000}ﬃ\u{93e}9",
  "list:\u{1f}item\n\u{2003}\u{5b4}4", // 0x1F control mid-word
  // Real non-BMP drops (0xF1/0xF2/0xF4 leads — the planes the old K5 "TAG" string, a
  // 0xF3 lead, wrongly claimed to cover; verified codepoint→lead-byte in the tests).
  "a\u{40000}b 12",                // U+40000 → 0xF1 lead (plane 4)
  "q\u{80000}\n\u{2003}r 3",       // U+80000 → 0xF2 lead (plane 8)
  "t\u{100000}u\u{2009}\u{2009}8", // U+100000 → 0xF4 lead (plane 16)
];

/// A seeded DROPPABLE fragment-soup: random short fragments over an alphabet that
/// ALWAYS includes vocab-dropped byte-level chars, so `build` takes the byte-coverage
/// FALLBACK arm — the merge gate uses it to sweep the `direct_only` path under
/// cross-class adjacency. (The K5-free index-path twin is
/// `k5_free_soup_exercises_index_path`.)
fn droppable_fragment_soup(seed: u64, target_chars: usize) -> String {
  const FRAGS: &[&str] = &[
    "a",
    "b",
    "c",
    "Z",
    "it",
    "café",
    "no",
    "  ",
    "   ",
    "\t",
    "\t\t",
    "\u{00A0}",
    "\u{00A0}\u{00A0}",
    "\u{2009}",
    "\u{2003}",
    "\u{3000}",
    "\r\n",
    "\r\n\r\n",
    "\n",
    "\n\n",
    "1",
    "12",
    "123",
    "1234",
    "٧",
    "٨٩",
    "５",
    "'s",
    "'t",
    "'re",
    "'ll",
    "!",
    "!!",
    ".",
    "...",
    "/",
    "//",
    "#",
    ",",
    "(",
    ")",
    "🍕",
    "🌿",
    "\u{0301}",
    "\u{093E}",
    "\u{064B}",
    " ",
    "x",
    "y",
    // K5 — vocab-droppable byte-level chars (silently dropped → byte-coverage guard →
    // direct_only). C0 controls plus REAL non-BMP droppers (0xF1/0xF2/0xF4 leads); the
    // old TAG entry (0xF3) is PRESENT and did NOT drop, so it is removed.
    "\u{000B}",   // VT (0x0B)
    "\u{000C}",   // FF (0x0C)
    "\u{0000}",   // NUL (0x00)
    "\u{0007}",   // BEL (a 0x04-0x1F C0 control)
    "\u{40000}",  // U+40000 → 0xF1 lead (plane 4)
    "\u{100000}", // U+100000 → 0xF4 lead (plane 16)
  ];
  let mut rng = Rng(seed);
  let mut s = String::new();
  while s.chars().count() < target_chars {
    s.push_str(FRAGS[rng.below(FRAGS.len())]);
  }
  s
}

/// Every `char`-boundary pair `(a, b)`, `a < b`, exhaustive up to `cap` boundaries
/// then seeded-sampled, pushing each `measure_range != oracle` into `out` as
/// `(text, a, b, got, want)`. Returns the number of pairs tested.
fn count_divergences(
  tok: &Tokenizer,
  text: &str,
  cap: usize,
  out: &mut Vec<(String, usize, usize, usize, usize)>,
) -> usize {
  if text.is_empty() {
    return 0;
  }
  let index = TokenIndex::build(tok, text).expect("build index");
  let bounds = char_boundaries(text);
  let m = bounds.len();
  let mut tested = 0usize;
  let one = |a: usize, b: usize, out: &mut Vec<_>| {
    let got = index.measure_range(tok, text, a, b).expect("measure_range");
    let want = oracle(tok, &text[a..b]);
    if got != want {
      out.push((text.to_string(), a, b, got, want));
    }
  };
  if m <= cap {
    for i in 0..m {
      for j in (i + 1)..m {
        one(bounds[i], bounds[j], out);
        tested += 1;
      }
    }
  } else {
    let mut rng = Rng(0xC0DE_F00D ^ text.len() as u64);
    for _ in 0..(cap * cap) {
      let mut a = bounds[rng.below(m)];
      let mut b = bounds[rng.below(m)];
      if a == b {
        continue;
      }
      if a > b {
        std::mem::swap(&mut a, &mut b);
      }
      one(a, b, out);
      tested += 1;
    }
  }
  tested
}

/// THE MERGE GATE (exhaustive adversarial differential). Over the killer classes,
/// one big cross-class concatenation, and seeded fragment-soups, every
/// `measure_range(a, b)` MUST equal `encode(&text[a..b], true).len()`. This
/// reproduced ~500 divergences against the pre-fix single pass; it must be zero.
///
/// F10 integrity: the clean `KILLERS` and the big join are droppable-free and
/// added-literal-free, and the join is LIVENESS-ASSERTED `!direct_only`, so they
/// exercise the INDEX path — not the trivial `direct_only` arm. `DROPPABLE_KILLERS`
/// and the droppable soups separately sweep the byte-coverage FALLBACK arm.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn measure_range_zero_divergence_over_killers() {
  let tok = measuring_tok();
  let mut out: Vec<(String, usize, usize, usize, usize)> = Vec::new();
  let mut pairs = 0usize;

  // Clean killers — the INDEX path.
  for &text in KILLERS {
    pairs += count_divergences(&tok, text, 96, &mut out);
  }
  // Droppable killers — the byte-coverage FALLBACK arm (still exactly 0-divergence by
  // direct encode); assert each genuinely trips a build-time guard.
  for &text in DROPPABLE_KILLERS {
    assert!(
      TokenIndex::build(&tok, text)
        .expect("build droppable")
        .direct_only,
      "droppable killer must trip a build-time guard (direct_only): {text:?}"
    );
    pairs += count_divergences(&tok, text, 96, &mut out);
  }
  // One big cross-class string, built from the CLEAN killers so every clean boundary
  // meets every other ON THE INDEX PATH. Liveness-asserted: a dropped byte or added
  // literal sneaking in would trip a build guard and silently gut this strongest
  // cross-class generator (the F10 regression this guards against).
  let big: String = KILLERS.join(" | ");
  assert!(
    !TokenIndex::build(&tok, &big)
      .expect("build big join")
      .direct_only,
    "the big cross-class join must exercise the INDEX path, not the direct_only arm"
  );
  pairs += count_divergences(&tok, &big, 200, &mut out);
  // Droppable fragment-soups — the fallback arm under cross-class adjacency.
  for seed in 0..6u64 {
    let soup = droppable_fragment_soup(0xA5A5_0000 ^ (seed.wrapping_mul(0x9E37_79B9)), 160);
    pairs += count_divergences(&tok, &soup, 200, &mut out);
  }

  eprintln!(
    "[killer-sweep] pairs_tested={pairs} divergences={}",
    out.len()
  );
  for (t, a, b, got, want) in out.iter().take(25) {
    eprintln!(
      "  DIVERGE measure_range({a},{b})={got} != encode({:?})={want}  in {:.48?}",
      &t[*a..*b],
      t
    );
  }
  assert!(
    out.is_empty(),
    "{} divergences over {pairs} killer-class pairs (see the list above) — the single-pass \
     measure is not exact",
    out.len()
  );
}

/// F1 witness (whitespace run split by a following digit, overcount +1): the two
/// spaces of `"456  1"[3..5]` are ONE pre-token at end-of-substring (`\s+(?!\S)`
/// merges), but the full parse splits them because a digit — which takes no
/// space-glue — follows at `b`.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn witness_f1_ws_run_split_by_following_digit() {
  let tok = measuring_tok();
  let text = "456  1";
  let index = TokenIndex::build(&tok, text).expect("build");
  assert_eq!(
    index.measure_range(&tok, text, 3, 5).unwrap(),
    oracle(&tok, "  "),
    "F1: [3,5) must be encode(\"  \"), not the split-run overcount"
  );
}

/// F2 witness (back-scan lands mid-pre-token, undercount): the punct `[\r\n/]*`
/// tail makes `"!\r\n\r\n"` one pre-token; scanning whitespace back from the right
/// zone walks into it, and without snapping DOWN to its start the partition drops
/// the `!` head.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn witness_f2_scan_back_snaps_out_of_punct_crlf_tail() {
  let tok = measuring_tok();
  let text = "a!\r\n\r\n Next t";
  let index = TokenIndex::build(&tok, text).expect("build");
  assert_eq!(
    index.measure_range(&tok, text, 0, 8).unwrap(),
    oracle(&tok, &text[0..8]),
    "F2: the `!` head of the punct+CRLF pre-token must not be dropped"
  );
}

/// F3 witness (contraction-suffix letter|letter adjacency, overcount): the word
/// branch ends at the `'s` suffix even with letters after, so `" it's"|"tation"`
/// is a real full boundary; cutting at the `'s` re-joins `"station"` as one word.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn witness_f3_contraction_suffix_letter_adjacency() {
  let tok = measuring_tok();
  let text = " it'station end";
  let index = TokenIndex::build(&tok, text).expect("build");
  assert_eq!(
    index.measure_range(&tok, text, 4, 11).unwrap(),
    oracle(&tok, "station"),
    "F3: [4,11) must be encode(\"station\"), not \"s\"+\"tation\""
  );
}

/// F8 witness (a vocab-dropped byte corrupts the reconstructed offsets): the
/// leading form feed `\u{c}` has NO single-byte vocab token and the BPE
/// `unk_token` is null, so `tokenizers` silently DROPS its byte-level char and
/// mis-attributes the surviving tokens' offsets. The chained `ends` stay
/// monotone, covering, and char-aligned, so no tiling check trips `direct_only` —
/// only the byte-coverage guard sees the shortfall. Pre-guard,
/// `measure_range(8, 16)` returned 6 for what `encode` counts as 7.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn witness_f8_vocab_dropped_byte_corrupts_index_offsets() {
  let tok = measuring_tok();
  let text = "\u{c}\n\u{2003}\u{202f}\nﬃ\u{5b4}\u{64b}23";
  // The drop is real and invisible to the tiling checks: ByteLevel emits exactly
  // one byte-level char per original byte, so fewer surviving chars than
  // `text.len()` proves a byte-level char vanished.
  let survived: usize = tok
    .encode(text, false)
    .expect("encode")
    .get_tokens()
    .iter()
    .map(|t| t.chars().count())
    .sum();
  assert!(
    survived < text.len(),
    "the form feed's byte-level char must be dropped: {survived} chars vs {} bytes",
    text.len()
  );
  // The public contract holds exactly at the ranges the corrupted offsets skewed.
  let index = TokenIndex::build(&tok, text).expect("build");
  for &(a, b) in &[(8usize, 16usize), (8, 17), (8, 18)] {
    assert_eq!(
      index.measure_range(&tok, text, a, b).unwrap(),
      oracle(&tok, &text[a..b]),
      "F8: [{a},{b}) must equal the direct encode, not the dropped-byte undercount"
    );
  }
  // The guard's chosen repair: the whole index falls back to exact direct encoding.
  assert!(
    index.direct_only,
    "the byte-coverage guard must reject the corrupted reconstruction (direct_only)"
  );
}

/// F9 witness (added-token lookaround, over-count): `<|reserved_200020|>` is a
/// `single_word` added token — it matches ONLY when flanked by non-word chars or a
/// text edge. In `"a<|reserved_200020|>b 12  3"` the literal is glued between the
/// word chars `a` and `b`, so the FULL parse tokenizes it as ordinary Split+BPE; but
/// the substring `[1,20)` IS the bare literal, which re-encodes to the single
/// added-token id. Pre-guard the index measured 10 for what `encode` counts as 3 —
/// the forward-determinism premise fails for single-word added tokens. The
/// added-token guard rejects the reconstruction (direct_only), restoring exactness.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn witness_f9_added_token_lookaround_guard() {
  let tok = measuring_tok();
  let text = "a<|reserved_200020|>b 12  3";
  assert_eq!(&text[1..20], "<|reserved_200020|>");
  let index = TokenIndex::build(&tok, text).expect("build");
  assert!(
    index.direct_only,
    "the added-token guard must reject the reconstruction (direct_only)"
  );
  assert_eq!(
    index.measure_range(&tok, text, 1, 20).unwrap(),
    oracle(&tok, "<|reserved_200020|>"),
    "F9: [1,20) must equal encode(\"<|reserved_200020|>\") = 3, not the index over-count"
  );
}

/// K6 — added-vocabulary literals in text (44 `single_word` reserved + 22 specials,
/// 66 total). ANY of them can make a SUBSTRING match a literal the full text never
/// did (F9), so the added-token guard fires on every one: `build` is `direct_only`
/// and every pair measures exactly by direct encode. (Fable's review template proved
/// the RED pre-fix behaviour; this asserts the FIXED behaviour.)
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn added_literals_guard_fires_and_measures_exactly() {
  let tok = measuring_tok();
  for text in [
    "pre <|endoftext|> post",
    "<|startoftext|>x<|return|>",
    "a<|reserved_200020|>b 12  3",
    "no<|end|>break\u{2009}\u{2009}9",
    "[MASK] in text",
    "mix <|reserved_200063|> and <|call|> here",
  ] {
    let index = TokenIndex::build(&tok, text).expect("build");
    assert!(
      index.direct_only,
      "the added-token guard must fire (direct_only) for {text:?}"
    );
    let bounds = char_boundaries(text);
    for i in 0..bounds.len() {
      for j in (i + 1)..bounds.len() {
        check(&index, &tok, text, bounds[i], bounds[j]);
      }
    }
  }
}

/// F11 — REAL non-BMP droppers. The vocab is missing the 4-byte lead bytes 0xF1 /
/// 0xF2 / 0xF4-0xFF (planes 4-11 and 16), so a codepoint in those planes drops a
/// byte-level char → byte-coverage guard → `direct_only` (exact by direct encode).
/// The codepoint→lead-byte mappings are verified in-test. (The old K5 "TAG" string,
/// a 0xF3 lead, is PRESENT and does NOT drop; these are the classes that actually
/// do.)
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn real_nonbmp_drops_fall_back_and_stay_exact() {
  // Lead byte of a codepoint's UTF-8: confirm the plane→lead mapping the fixtures
  // rely on, so a wrong codepoint cannot silently pass by NOT dropping.
  fn lead(cp: char) -> u8 {
    let mut buf = [0u8; 4];
    cp.encode_utf8(&mut buf).as_bytes()[0]
  }
  assert_eq!(lead('\u{40000}'), 0xF1);
  assert_eq!(lead('\u{80000}'), 0xF2);
  assert_eq!(lead('\u{100000}'), 0xF4);
  assert_eq!(lead('\u{7FFFF}'), 0xF1);
  assert_eq!(lead('\u{10FFFD}'), 0xF4);

  let tok = measuring_tok();
  for text in [
    "a\u{40000}b 12",                           // U+40000 → 0xF1 (plane 4)
    "a\u{40000}b 12  3",                        // + trailing ws-run split
    "q\u{80000}\n\u{2003}r 3",                  // U+80000 → 0xF2 (plane 8)
    "q\u{80000}\n\u{2003}\u{FB03}\u{5b4}r 3",   // + ligature + combining mark
    "t\u{100000}u\u{2009}\u{2009}8",            // U+100000 → 0xF4 (plane 16)
    "x \u{7FFFF}\u{7FFFF} y9\u{2009}\u{2009}8", // U+7FFFF → 0xF1
    "\u{10FFFD}\u{10FFFD} end.\r\n\r\nNext",    // U+10FFFD → 0xF4
  ] {
    let index = TokenIndex::build(&tok, text).expect("build");
    assert!(
      index.direct_only,
      "non-BMP drop must fall back to direct_only for {text:?}"
    );
    let bounds = char_boundaries(text);
    for i in 0..bounds.len() {
      for j in (i + 1)..bounds.len() {
        check(&index, &tok, text, bounds[i], bounds[j]);
      }
    }
  }
}

/// F11 — 0xF3-lead 4-byte chars (TAG plane 14, PUA-A plane 15) are PRESENT in the
/// vocab: NO drop, so the index path stays LIVE and must still measure every pair
/// exactly. This is the correction to the wrong "0xF1-0xF4 all drop" claim — 0xF3
/// is in the vocab, so its planes index cleanly.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn tag_and_pua_do_not_drop_and_index_stays_exact() {
  fn lead(cp: char) -> u8 {
    let mut buf = [0u8; 4];
    cp.encode_utf8(&mut buf).as_bytes()[0]
  }
  assert_eq!(lead('\u{E0067}'), 0xF3); // TAG (plane 14)
  assert_eq!(lead('\u{F0000}'), 0xF3); // PUA-A (plane 15)

  let tok = measuring_tok();
  for text in [
    "tag\u{E0067}\u{E0067}\nﬀ\u{5b4}7",
    "lang\u{E0001}tag 12  3",
    "pua\u{F0000}\u{FFFFD}z\u{2009}\u{2009}9",
  ] {
    let index = TokenIndex::build(&tok, text).expect("build");
    assert!(
      !index.direct_only,
      "0xF3-lead chars do not drop; the index path must stay live for {text:?}"
    );
    let bounds = char_boundaries(text);
    for i in 0..bounds.len() {
      for j in (i + 1)..bounds.len() {
        check(&index, &tok, text, bounds[i], bounds[j]);
      }
    }
  }
}

/// F10 — the cross-class INDEX-path coverage the accidental K5 fragments removed from
/// the committed soups (which now all trip the byte-coverage guard and run the direct
/// arm). These soups draw ONLY from droppable-free, added-literal-free fragments, so
/// `build` stays on the index fast path (asserted), and every pair must still measure
/// exactly. This restores the strongest index-path cross-class generator.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn k5_free_soup_exercises_index_path() {
  fn clean_soup(seed: u64, target_chars: usize) -> String {
    // No vocab-droppable byte-level char and no added-token literal, so `build` keeps
    // the index path.
    const FRAGS: &[&str] = &[
      "a",
      "b",
      "c",
      "Z",
      "it",
      "café",
      "no",
      "  ",
      "   ",
      "\t",
      "\t\t",
      "\u{00A0}",
      "\u{00A0}\u{00A0}",
      "\u{2009}",
      "\u{2003}",
      "\u{3000}",
      "\r\n",
      "\r\n\r\n",
      "\n",
      "\n\n",
      "1",
      "12",
      "123",
      "1234",
      "٧",
      "٨٩",
      "５",
      "'s",
      "'t",
      "'re",
      "'ll",
      "!",
      "!!",
      ".",
      "...",
      "/",
      "//",
      "#",
      ",",
      "(",
      ")",
      "🍕",
      "🌿",
      "\u{0301}",
      "\u{093E}",
      "\u{064B}",
      " ",
      "x",
      "y",
    ];
    let mut rng = Rng(seed);
    let mut s = String::new();
    while s.chars().count() < target_chars {
      s.push_str(FRAGS[rng.below(FRAGS.len())]);
    }
    s
  }
  let tok = measuring_tok();
  let mut out: Vec<(String, usize, usize, usize, usize)> = Vec::new();
  let mut pairs = 0usize;
  for seed in 100..106u64 {
    let soup = clean_soup(0x5EED_0000 ^ (seed.wrapping_mul(0x9E37_79B9)), 160);
    let index = TokenIndex::build(&tok, &soup).expect("build");
    assert!(
      !index.direct_only,
      "clean soup must take the index path (not direct_only): {soup:?}"
    );
    pairs += count_divergences(&tok, &soup, 200, &mut out);
  }
  eprintln!("[k5-free-soup] pairs={pairs} divergences={}", out.len());
  for (t, a, b, got, want) in out.iter().take(10) {
    eprintln!("  DIVERGE ({a},{b}) got={got} want={want} in {t:?}");
  }
  assert!(
    out.is_empty(),
    "{} divergences over {pairs} pairs",
    out.len()
  );
}

/// codex #57 (perf) — an OVERSIZED single pre-token double-encoded every probe.
/// A multi-megabyte lowercase run is ONE word-branch pre-token (`pretoken_ends ==
/// [n]`), so no interior full-parse boundary exists for any cut to land on. windit's
/// char fallback then measures growing prefixes `[a, b)` with `0 < a < b < n`; each
/// makes `left_resync` cap its window at `b` and encode the whole `[a, b)`, find no
/// boundary, and — pre-fix — return `None`, whereupon the caller RE-ENCODED the
/// identical `[a, b)`: two tokenizer encodes of the same range per probe. The fix
/// hands that window's own already-computed whole-query count back
/// (`Resync::WholeQuery`) so the caller reuses it: ONE encode per probe.
///
/// RED pre-fix / GREEN post-fix, asserted on the hermetic encode meter: each probe
/// now encodes its range EXACTLY ONCE (`calls == 1`, `bytes == b - a`) where the
/// pre-fix double-encode metered `calls == 2`, `bytes == 2 * (b - a)`. The reused
/// count stays output-identical — spot-checked against the direct oracle here, and
/// swept over the full killer differential (`measure_range == encode`) elsewhere.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn oversized_single_pretoken_encodes_each_probe_once() {
  let tok = measuring_tok();
  // A multi-megabyte single unbroken pre-token: the lowercase run matches one
  // word-branch span, so the whole input tiles as ONE pre-token.
  let n = 4 * 1024 * 1024;
  let text = "a".repeat(n);
  let index = TokenIndex::build(&tok, &text).expect("build");
  assert!(
    !index.direct_only,
    "'a'*n stays on the live index path (no dropped byte, no added literal)"
  );
  assert_eq!(
    index.pretoken_ends.len(),
    1,
    "the oversized-single case requires the whole input to be ONE pre-token"
  );

  // windit's char-fallback probe shape: a growing prefix from a fixed post-first
  // segment start, plus interior windows and one reaching the last char — every
  // one has `0 < a < b < n` and no interior boundary, so each takes the reuse path.
  let probes = [
    (1_000usize, 1_001usize), // minimal window
    (1_000, 1_010),
    (1_000, 11_000),
    (1_000, 211_000),       // a large growing prefix
    (2_000_000, 2_050_000), // an interior window far from the start
    (n - 50, n - 1),        // b at the last char (still strictly < n)
  ];
  for &(a, b) in &probes {
    super::encode_meter::reset();
    let got = index
      .measure_range(&tok, &text, a, b)
      .expect("measure_range must not fail on the granite tokenizer");
    let calls = super::encode_meter::calls();
    let bytes = super::encode_meter::get();
    // ONE encode of EXACTLY the probe range — the single-encode property (pre-fix:
    // 2 calls, 2*(b-a) bytes).
    assert_eq!(
      calls, 1,
      "probe [{a},{b}) must encode ONCE, not twice: metered {calls} encode calls"
    );
    assert_eq!(
      bytes,
      b - a,
      "probe [{a},{b}) must encode exactly its own {} bytes once, not {} (the pre-fix \
       double-encode)",
      b - a,
      2 * (b - a)
    );
    // Output identity: the reused whole-query count equals a fresh direct encode.
    assert_eq!(
      got,
      oracle(&tok, &text[a..b]),
      "reused whole-query count must equal encode(&text[{a}..{b}], true).len()"
    );
  }
}
