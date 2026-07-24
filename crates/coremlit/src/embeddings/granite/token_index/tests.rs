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
use crate::embeddings::granite::{BUNDLED_TOKENIZER, measuring_tokenizer_from_bytes};

/// The truncation-disabled MEASURING tokenizer — the exact one `chunk_long` builds
/// its index with.
fn measuring_tok() -> Tokenizer {
  measuring_tokenizer_from_bytes(BUNDLED_TOKENIZER).expect("measuring tokenizer")
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
fn measure_range_matches_encode_over_adversarial_corpus() {
  let tok = measuring_tok();
  let mut rng = Rng(0x0DDB_1A5E_5EED_1234);
  for &text in ADVERSARIAL {
    differential_over(&tok, text, &mut rng);
  }
}

#[test]
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
