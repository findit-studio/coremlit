//! The mirror is exact: its final tokens equal the crate's over every word the
//! artifact tokenizer can be handed, and its recorded process is internally
//! consistent (spines, pop ends, tiling).

use tokenizers::{Tokenizer, utils::SysRegex};

use super::{HEAD_CLASS, LEAD_CLASS, MergeTable, SPLIT_PATTERN, TAIL_CLASS, TailClass};
use crate::embeddings::granite::{measuring_tokenizer_from_bytes, test_artifact};

fn measuring_tok() -> Tokenizer {
  measuring_tokenizer_from_bytes(test_artifact::tokenizer_bytes()).expect("measuring tokenizer")
}

fn table(tok: &Tokenizer) -> MergeTable {
  MergeTable::from_tokenizer(tok).expect("the artifact's BPE is mirrorable")
}

/// Crate tokens of a single pre-token `s` (no specials), as `(id, end)`.
fn crate_tokens(tok: &Tokenizer, s: &str) -> Vec<(u32, usize)> {
  let enc = tok.encode(s, false).expect("encode");
  enc
    .get_ids()
    .iter()
    .zip(enc.get_offsets())
    .map(|(&id, &(_, e))| (id, e))
    .collect()
}

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

/// Words that pre-tokenize as ONE piece (all-alphabetic runs), so the crate's
/// encode of the word IS the BPE of its bytes: CJK runs of every length, Latin
/// lowercase runs, Cyrillic, Greek, Thai, mixed scripts, and random draws from
/// a CJK alphabet.
fn single_piece_words() -> Vec<String> {
  const CJK: &str = "你好世界模型推理文本嵌入检索的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种向死去所有";
  let cjk: Vec<char> = CJK.chars().collect();
  let mut words = vec![
    "你".to_string(),
    "你好".to_string(),
    "你好世界".to_string(),
    "模型推理文本嵌入检索".to_string(),
    "internationalization".to_string(),
    "x".repeat(300),
    "ab".repeat(200),
    "привет".to_string(),
    "καλημέρα".to_string(),
    "สวัสดี".to_string(),
    "你好hello世界world".to_string(),
  ];
  let mut rng = Rng(7);
  for len in [2usize, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610] {
    for _ in 0..8 {
      let w: String = (0..len).map(|_| cjk[rng.below(cjk.len())]).collect();
      words.push(w);
    }
  }
  words
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn mirrored_process_matches_the_crate_on_single_piece_words() {
  let tok = measuring_tok();
  let table = table(&tok);
  for w in single_piece_words() {
    let run = table
      .process(w.as_bytes())
      .expect("every byte of the word has a token");
    let want = crate_tokens(&tok, &w);
    // `ignore_merges`: a whole word in the vocabulary is one token to the
    // crate; the mirror deliberately runs the process instead (module docs).
    if tok.token_to_id(&table.spell(w.as_bytes())).is_some() {
      assert_eq!(want.len(), 1, "{w:?}: crate applies ignore_merges");
      continue;
    }
    // Ids must agree token for token. Ends agree wherever the crate's offsets
    // are byte-exact; a byte token INSIDE a multi-byte char is reported by the
    // crate at the whole char's offsets, so only the ends of tokens that close
    // a char are compared.
    let got_ids: Vec<u32> = run.ids.clone();
    let want_ids: Vec<u32> = want.iter().map(|&(id, _)| id).collect();
    assert_eq!(got_ids, want_ids, "mirror diverges from the crate on {w:?}");
    for (&e, &(_, we)) in run.ends.iter().zip(&want) {
      if w.is_char_boundary(e as usize) {
        assert_eq!(e as usize, we, "{w:?}: end of a char-closing token");
      }
    }
  }
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn recorded_process_is_internally_consistent() {
  let tok = measuring_tok();
  let table = table(&tok);
  for w in single_piece_words() {
    let run = table
      .process(w.as_bytes())
      .expect("every byte of the word has a token");
    // Tiling: ends strictly increase and cover the word.
    assert!(run.ends.windows(2).all(|p| p[0] < p[1]), "{w:?}: ends");
    assert_eq!(
      *run.ends.last().unwrap() as usize,
      w.len(),
      "{w:?}: coverage"
    );
    assert_eq!(run.ends.len(), run.ids.len());
    assert_eq!(run.ends.len(), run.rspine.len());
    assert_eq!(run.ends.len(), run.lspine.len());
    // Pops: as many as symbols minus tokens; each pop's end is a token end of
    // SOME later or final state — in particular the last pop of each spine
    // produces that final token.
    assert_eq!(run.pops.len(), w.len() - run.ends.len(), "{w:?}: pop count");
    for (t, (&end, &id)) in run.ends.iter().zip(&run.ids).enumerate() {
      let start = if t == 0 { 0 } else { run.ends[t - 1] };
      for spine in [&run.rspine[t], &run.lspine[t]] {
        assert!(spine.windows(2).all(|p| p[0] < p[1]), "{w:?}: spine order");
        if end - start == 1 {
          assert!(spine.is_empty(), "{w:?}: a single-byte token has no merges");
        } else {
          let last = *spine.last().expect("multi-byte token was merged") as usize;
          assert_eq!(run.pops[last].new_id, id, "{w:?}: spine ends at the token");
          assert_eq!(
            run.pops[last].end, end,
            "{w:?}: spine's last pop ends the token"
          );
        }
      }
      // Right-spine pops all end at the token's end (they absorb the last byte);
      // left-spine pops all start at the token's start (the produced token's end
      // grows monotonically along it).
      for &i in &run.rspine[t] {
        assert_eq!(run.pops[i as usize].end, end, "{w:?}: right spine end");
      }
      let mut prev = start;
      for &i in &run.lspine[t] {
        let e = run.pops[i as usize].end;
        assert!(
          e > prev && e <= end,
          "{w:?}: left spine grows within the token"
        );
        prev = e;
      }
    }
  }
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn the_table_reads_the_artifacts_geometry() {
  let tok = measuring_tok();
  let table = table(&tok);
  assert_eq!(table.max_token_bytes(), 128);
  assert_eq!(table.merge_count(), 413_540, "the artifact's merge list");
  // The byte-level single-char tokens the vocabulary has are exactly the ones
  // the table maps (the crate's own vocabulary is the oracle), and the artifact
  // is missing some — NUL among them — as the index's coverage guard documents.
  for b in 0..=255u8 {
    let s = table.spell(&[b]);
    assert_eq!(tok.token_to_id(&s), table.symbol(b), "byte {b:#04x}");
  }
  assert_eq!(
    table.symbol(0),
    None,
    "NUL has no single-char token in the artifact"
  );
  assert!(table.symbol(b'a').is_some());
  assert!(
    table.process(b"a\0b").is_none(),
    "a word with a token-less byte is refused, not silently dropped"
  );
}

/// TIMING (run locally for PR notes): building the merge table from the loaded
/// model — the one-time cost an embedder pays on the lane's first engagement.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS); timing"]
fn merge_table_build_time() {
  let tok = measuring_tok();
  let t0 = std::time::Instant::now();
  let table = table(&tok);
  let ms = t0.elapsed().as_secs_f64() * 1e3;
  println!(
    "[merge-table] build={ms:.1}ms max_token_bytes={}",
    table.max_token_bytes()
  );
}

/// The pinned Split pattern IS the pattern the lane's lemma is about: its two
/// letter branches, in this order, built from the three classes the gate is
/// derived from.
#[test]
fn the_pinned_split_pattern_opens_with_the_two_letter_branches() {
  let suffix = "(?i:'s|'t|'re|'ve|'m|'ll|'d)?";
  let letters = format!(
    "{LEAD_CLASS}?{HEAD_CLASS}*{TAIL_CLASS}+{suffix}|{LEAD_CLASS}?{HEAD_CLASS}+{TAIL_CLASS}*{suffix}|"
  );
  assert!(SPLIT_PATTERN.starts_with(&letters), "{SPLIT_PATTERN}");
  assert!(
    SysRegex::new(SPLIT_PATTERN).is_ok(),
    "the pinned pattern compiles on the crate's engine"
  );
  assert!(TailClass::new().is_some());
}

/// The tail class, compiled on the crate's engine, agrees with the pinned
/// Split regex itself — `"a" + c` is ONE pre-token exactly when `c` is a tail
/// char (the first letter branch's `TAIL+` runs through it, and no branch can
/// glue anything else to a lowercase `a`) — over every scalar below U+3400
/// (Latin, Greek, Cyrillic, the Semitic and Indic scripts with their marks,
/// the titlecase digraphs and Greek titlecase vowels, all of the C0/C1
/// controls, digits and punctuation), a stride of the rest of the space, and
/// hand-picked exemplars of every general category. Hermetic: the pinned
/// pattern is a constant.
#[test]
fn the_tail_class_agrees_with_the_split_regex_over_a_dense_sample_of_scalars() {
  let tail = TailClass::new().expect("compiles");
  let split = SysRegex::new(SPLIT_PATTERN).expect("compiles");
  let is_tail = |c: char| tail.contains(c);
  let glued = |c: char| {
    let s = format!("a{c}");
    split.find_iter(&s).next() == Some((0, s.len()))
  };
  let exemplars = [
    0x41, 0x391, 0x410, 0x1C4, 0x1F08, 0x1F88, // Lu / Lt
    0x1C5, 0x1C8, 0x1CB, 0x1F2, 0x1F98, 0x1FA8, 0x1FBC, 0x1FCC, 0x1FFC, // Lt
    0x61, 0x3B1, 0x430, 0xDF, 0x1E9F, 0x1D00, // Ll
    0x2B0, 0x2C6, 0x3005, 0x309D, 0x30FC, 0xA015, 0xFF70, // Lm
    0x4E2D, 0x5D0, 0xAC00, 0x3042, 0x20000, 0x2A6D6, 0x10400, // Lo
    0x300, 0x5BF, 0x903, 0x20DD, 0xFE0F, 0xE0100, // Mn / Mc / Me
    0x30, 0x660, 0xB2, 0x2160, 0x3007, // Nd / No / Nl
    0x20, 0x9, 0xA, 0xD, 0xA0, 0x2028, 0x3000, // whitespace
    0x27, 0x2E, 0x3002, 0xFF01, 0x24, 0x2B, 0x1F600, 0x200D, // punctuation / symbols / format
    0x0, 0x7F, 0x85, 0xE000, 0x10FFFF, // controls, private use, the last scalar
  ];
  let scalars = (0u32..0x3400)
    .chain((0x3400..0x110000).step_by(97))
    .chain(exemplars);
  let (mut total, mut tails) = (0usize, 0usize);
  for cp in scalars {
    let Some(c) = char::from_u32(cp) else {
      continue;
    };
    if c == 'a' {
      continue;
    }
    let t = is_tail(c);
    assert_eq!(t, glued(c), "U+{cp:04X} {c:?}");
    total += 1;
    tails += usize::from(t);
  }
  assert!(
    tails > 5_000 && total - tails > 5_000,
    "{tails} tail chars of {total}"
  );
  for cp in [
    0x61u32, 0x3B1, 0x430, 0x2B0, 0x3005, 0x4E2D, 0x5D0, 0xAC00, 0x300, 0x903, 0x20DD,
  ] {
    assert!(
      is_tail(char::from_u32(cp).expect("scalar")),
      "U+{cp:04X} is a tail char"
    );
  }
  for cp in [
    0x41u32, 0x391, 0x410, 0x1C5, 0x1F88, 0x1FFC, 0x30, 0x20, 0x27, 0x3002, 0xA, 0x0,
  ] {
    assert!(
      !is_tail(char::from_u32(cp).expect("scalar")),
      "U+{cp:04X} is not a tail char"
    );
  }
}

/// A tokenizer that is the artifact's in every way but one of the pinned
/// configuration points gets NO table — the lane stays off, the exact path
/// stays — while the unmutated artifact does (the premise).
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn the_table_refuses_a_configuration_the_lane_is_not_pinned_to() {
  fn mutated(mutate: impl FnOnce(&mut serde_json::Value)) -> Tokenizer {
    let mut value: serde_json::Value =
      serde_json::from_slice(test_artifact::tokenizer_bytes()).expect("parse the artifact");
    mutate(&mut value);
    Tokenizer::from_bytes(serde_json::to_vec(&value).expect("serialize")).expect("load")
  }
  assert!(
    MergeTable::from_tokenizer(&mutated(|_| {})).is_some(),
    "premise: the re-serialized artifact is mirrorable"
  );
  let swapped = {
    let (first, rest) = SPLIT_PATTERN.split_once('|').expect("alternation");
    let (second, tail) = rest.split_once('|').expect("alternation");
    format!("{second}|{first}|{tail}")
  };
  type Mutation = Box<dyn FnOnce(&mut serde_json::Value)>;
  let cases: [(&str, Mutation); 8] = [
    (
      "ignore_merges off",
      Box::new(|v| v["model"]["ignore_merges"] = serde_json::json!(false)),
    ),
    (
      "a normalizer",
      Box::new(|v| v["normalizer"] = serde_json::json!({ "type": "NFC" })),
    ),
    (
      "ByteLevel add_prefix_space",
      Box::new(|v| {
        v["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = serde_json::json!(true)
      }),
    ),
    (
      "ByteLevel use_regex",
      Box::new(|v| v["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = serde_json::json!(true)),
    ),
    (
      "the letter branches swapped",
      Box::new(move |v| {
        v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = serde_json::json!(swapped)
      }),
    ),
    (
      "Split behavior Removed",
      Box::new(|v| {
        v["pre_tokenizer"]["pretokenizers"][0]["behavior"] = serde_json::json!("Removed")
      }),
    ),
    (
      "ByteLevel alone",
      Box::new(|v| {
        v["pre_tokenizer"] = serde_json::json!({
          "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true
        });
      }),
    ),
    (
      "byte_fallback",
      Box::new(|v| v["model"]["byte_fallback"] = serde_json::json!(true)),
    ),
  ];
  for (what, mutate) in cases {
    assert!(
      MergeTable::from_tokenizer(&mutated(mutate)).is_none(),
      "{what}: no table"
    );
  }
}
