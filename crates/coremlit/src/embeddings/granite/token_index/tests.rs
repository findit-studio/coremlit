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

// ── F7: the four killer classes Fable's adversarial differential surfaced ──────
//
// The layer-1/2 corpora above miss every class that can actually dissolve a
// full-parse boundary under a cut: their whitespace probe `"a   b   c"` has runs
// GLUED by the surrounding letters (branch-1's leading ` ?` pulls the run's last
// space into the next word), so it can never diverge. These generators add the
// classes that CAN — swept over every `char`-boundary pair they reproduce the
// ~500 pre-fix divergences and MUST now be zero:
//   K1  a whitespace run split by a NON-glue follower (digit/punct/emoji): only a
//       literal ' ' glues forward, so a tab/NBSP/NEL/thin/ideographic run before a
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
];

/// A seeded fragment-soup: random short fragments drawn from an alphabet of every
/// killer char, concatenated — the cross-class adjacencies the fixed strings miss.
fn fragment_soup(seed: u64, target_chars: usize) -> String {
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

/// THE MERGE GATE (exhaustive adversarial differential). Over the four killer
/// classes, one big cross-class concatenation, and seeded fragment-soup, every
/// `measure_range(a, b)` MUST equal `encode(&text[a..b], true).len()`. This
/// reproduced ~500 divergences against the pre-fix single pass; it must be zero.
#[test]
fn measure_range_zero_divergence_over_killers() {
  let tok = measuring_tok();
  let mut out: Vec<(String, usize, usize, usize, usize)> = Vec::new();
  let mut pairs = 0usize;

  for &text in KILLERS {
    pairs += count_divergences(&tok, text, 96, &mut out);
  }
  // One big cross-class string: every killer boundary meets every other.
  let big: String = KILLERS.join(" | ");
  pairs += count_divergences(&tok, &big, 200, &mut out);
  // Seeded fragment-soup, several documents.
  for seed in 0..6u64 {
    let soup = fragment_soup(0xA5A5_0000 ^ (seed.wrapping_mul(0x9E37_79B9)), 160);
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
