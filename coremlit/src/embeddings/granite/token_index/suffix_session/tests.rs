//! The fast lane is exact: every prefix count it answers equals the crate's
//! `encode` of the same bytes, over the separatorless corpora of #72, random
//! CJK draws, mixed scripts, and words built to make BPE non-prefix-stable —
//! and the cascade walk really does fire (the certificate is not vacuous).

use tokenizers::Tokenizer;

use super::{INITIAL_CAP, Prefix, Session};
use crate::embeddings::granite::{
  measuring_tokenizer_from_bytes, test_artifact, token_index::MergeTable,
};

fn measuring_tok() -> Tokenizer {
  measuring_tokenizer_from_bytes(test_artifact::tokenizer_bytes()).expect("measuring tokenizer")
}

fn table(tok: &Tokenizer) -> MergeTable {
  MergeTable::from_tokenizer(tok).expect("the artifact's BPE is mirrorable")
}

/// Content-token count the crate reports for a single-piece word.
fn oracle(tok: &Tokenizer, s: &str) -> usize {
  tok.encode(s, false).expect("encode").get_ids().len()
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

fn cjk_run(len_chars: usize, seed: u64) -> String {
  const CJK: &str = "你好世界模型推理文本嵌入检索的一是不了人我在有他这中大来上国个到说们为子和地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种向";
  let cjk: Vec<char> = CJK.chars().collect();
  let mut rng = Rng(seed);
  (0..len_chars).map(|_| cjk[rng.below(cjk.len())]).collect()
}

/// Every char-aligned prefix of `text[start..]` measured through a session,
/// against the crate. Returns how many probes the cascade answered from a cut
/// strictly left of the token containing `b` (the certificate refused at least
/// one cut), and how many fell through to `Direct`.
#[track_caller]
fn check_every_prefix(
  tok: &Tokenizer,
  table: &MergeTable,
  text: &str,
  start: usize,
) -> (usize, usize) {
  let end = text.len().min(start + INITIAL_CAP);
  let mut session = Session::build(tok, table, text, start, end)
    .expect("build")
    .expect("session usable");
  let (mut cascaded, mut direct) = (0usize, 0usize);
  for b in (start + 1..=end).filter(|&b| text.is_char_boundary(b)) {
    let want = oracle(tok, &text[start..b]);
    match session.measure_prefix(tok, table, b) {
      Prefix::Count(got) => {
        assert_eq!(got, want, "prefix [{start}, {b}) of {:?}", &text[start..b]);
        if session.last_cascade_depth() > 0 {
          cascaded += 1;
        }
      }
      Prefix::Direct => direct += 1,
      Prefix::PastCap => panic!("b={b} within the cap"),
    }
  }
  (cascaded, direct)
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn every_prefix_of_the_issue_corpora_measures_like_the_crate() {
  let tok = measuring_tok();
  let table = table(&tok);
  let cjk = "你好世界模型推理文本嵌入检索".repeat(120); // 5,040 bytes
  let xs = "x".repeat(3000);
  for (label, text) in [("cjk", cjk.as_str()), ("x", xs.as_str())] {
    for start in [0usize, 3, 7, 30, 301] {
      let start = (start..).find(|&i| text.is_char_boundary(i)).unwrap();
      let (cascaded, direct) = check_every_prefix(&tok, &table, text, start);
      println!("[fast-lane:{label}] start={start} cascaded={cascaded} direct={direct}");
      // A cascade that walks back to the start of `Z` is answered by the direct
      // encode — exact, and only ever seen on the first few bytes of the `x`
      // run (the prefix sits inside the first token). It must stay rare: a
      // lane that fell through routinely would be the old cost back.
      assert!(
        direct <= 4,
        "{label}: {direct} probes walked back to the start of Z"
      );
    }
  }
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn every_prefix_of_random_cjk_measures_like_the_crate() {
  let tok = measuring_tok();
  let table = table(&tok);
  for seed in 1..=6u64 {
    let text = cjk_run(900, seed);
    for start in [0usize, 3, 6, 33, 150] {
      check_every_prefix(&tok, &table, &text, start);
    }
  }
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn every_prefix_of_mixed_script_runs_measures_like_the_crate() {
  let tok = measuring_tok();
  let table = table(&tok);
  let texts = [
    "internationalizationinternationalization".repeat(20),
    "приветмирпривет".repeat(60),
    "καλημέρακόσμε".repeat(60),
    "สวัสดีชาวโลก".repeat(50),
    "你好hello世界world".repeat(60),
    "ab".repeat(600),
    "abcabcabd".repeat(150),
  ];
  for text in &texts {
    for start in [0usize, 1, 5, 17] {
      let start = (start..).find(|&i| text.is_char_boundary(i)).unwrap();
      check_every_prefix(&tok, &table, text, start);
    }
  }
}

/// The certificate's non-trivial branch is exercised: on the issue's own CJK
/// corpus, some probe ends inside a token whose cut is NOT a boundary of the
/// prefix's tokenization, so the walk moved a cut left and still matched the
/// crate. (If no corpus ever cascaded, the branch would be untested — and BPE
/// non-prefix-stability guarantees it happens on real vocabularies.)
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn the_cascade_walk_fires_and_stays_exact() {
  let tok = measuring_tok();
  let table = table(&tok);
  let mut total = 0usize;
  for seed in 1..=12u64 {
    let text = cjk_run(600, seed);
    total += check_every_prefix(&tok, &table, &text, 0).0;
  }
  let issue = "你好世界模型推理文本嵌入检索".repeat(120);
  total += check_every_prefix(&tok, &table, &issue, 0).0;
  assert!(
    total > 0,
    "no probe cascaded — the certificate's refusal branch is unexercised"
  );
}

/// A whole probe that is itself a vocabulary entry is ONE token however the
/// merges would tile it (`ignore_merges`), and the session says so.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn a_probe_that_is_a_vocabulary_entry_is_one_token() {
  let tok = measuring_tok();
  let table = table(&tok);
  let text = "你好世界模型推理文本嵌入检索".repeat(4);
  let mut session = Session::build(&tok, &table, &text, 0, text.len())
    .expect("build")
    .expect("usable");
  // "你好" is a vocabulary entry of the artifact.
  assert!(tok.token_to_id(&table.spell("你好".as_bytes())).is_some());
  match session.measure_prefix(&tok, &table, "你好".len()) {
    Prefix::Count(1) => {}
    other => panic!("expected Count(1), got {}", describe(&other)),
  }
}

/// A suffix that is itself a vocabulary entry is not a session: the crate would
/// answer it with the whole-word shortcut, not the process.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn a_suffix_that_is_a_vocabulary_entry_gets_no_session() {
  let tok = measuring_tok();
  let table = table(&tok);
  let text = "模型推理你好";
  let start = "模型推理".len();
  assert!(
    Session::build(&tok, &table, text, start, text.len())
      .expect("build")
      .is_none()
  );
}

fn describe(p: &Prefix) -> String {
  match p {
    Prefix::Count(n) => format!("Count({n})"),
    Prefix::Direct => "Direct".into(),
    Prefix::PastCap => "PastCap".into(),
  }
}

// ─── Adversarial corpora derived from the merge ranks ────────────────────────

/// Single-CJK-char vocabulary tokens: `(char, id)`.
fn cjk_char_tokens(tok: &Tokenizer, table: &MergeTable) -> Vec<(char, u32)> {
  let mut out = Vec::new();
  for cp in 0x4E00u32..=0x9FFF {
    let Some(ch) = char::from_u32(cp) else {
      continue;
    };
    let mut buf = [0u8; 4];
    let s = ch.encode_utf8(&mut buf);
    if let Some(id) = tok.token_to_id(&table.spell(s.as_bytes())) {
      out.push((ch, id));
    }
  }
  out
}

/// Non-prefix-stability witnesses mined from the merge table: triples
/// `(A, B, C)` of single-char tokens with merges `A·B` and `B·C` such that
/// `rank(B·C) < rank(A·B)`. Then `T("ABC")` starts `A, BC` while `T("AB")` is
/// `AB`: the prefix `"AB"` of `"ABC"` does NOT tokenize as the prefix of
/// `T("ABC")`. A probe ending after `B` inside such a run lands inside the
/// token `BC`, whose cut (the start of `BC`) is NOT a boundary of the prefix's
/// tokenization — exactly the case the certificate must refuse and the cascade
/// must walk past.
fn merge_rank_witnesses(
  tok: &Tokenizer,
  table: &MergeTable,
  limit: usize,
) -> Vec<(char, char, char)> {
  let chars = cjk_char_tokens(tok, table);
  // Index pair merges by their left operand so the triple search is linear
  // in the number of pair merges rather than cubic in the alphabet.
  let mut right_of: std::collections::HashMap<u32, Vec<(char, u32, u32)>> =
    std::collections::HashMap::new();
  for &(_, ib) in &chars {
    for &(c, ic) in &chars {
      if let Some((rank, _)) = table.merge(ib, ic) {
        right_of.entry(ib).or_default().push((c, ic, rank));
      }
    }
  }
  let mut out = Vec::new();
  'outer: for &(a, ia) in &chars {
    for &(b, ib) in &chars {
      let Some((rank_ab, _)) = table.merge(ia, ib) else {
        continue;
      };
      if let Some(nexts) = right_of.get(&ib) {
        for &(c, _, rank_bc) in nexts {
          if rank_bc < rank_ab {
            out.push((a, b, c));
            if out.len() >= limit {
              break 'outer;
            }
          }
        }
      }
    }
  }
  out
}

#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn merge_rank_witnesses_exist_and_break_prefix_stability() {
  let tok = measuring_tok();
  let table = table(&tok);
  let witnesses = merge_rank_witnesses(&tok, &table, 64);
  assert!(
    witnesses.len() >= 16,
    "expected many witnesses, got {}",
    witnesses.len()
  );
  let mut broke = 0usize;
  for &(a, b, c) in &witnesses {
    let abc: String = [a, b, c].iter().collect();
    let ab: String = [a, b].iter().collect();
    let t_abc = tok
      .encode(abc.as_str(), false)
      .expect("encode")
      .get_ids()
      .to_vec();
    let t_ab = tok
      .encode(ab.as_str(), false)
      .expect("encode")
      .get_ids()
      .to_vec();
    // The crate's own view: the prefix's tokens are not a prefix of the whole's.
    if !t_abc.starts_with(&t_ab) {
      broke += 1;
    }
  }
  assert!(
    broke * 2 >= witnesses.len(),
    "witnesses should mostly break prefix stability under the crate: {broke}/{}",
    witnesses.len()
  );
}

/// Texts woven from the witnesses — every prefix from several starts measures
/// like the crate, and the cascade walk fires on them.
#[test]
#[ignore = "requires the granite tokenizer.json staged beside the model bundle (EMBEDKIT_TEST_MODELS)"]
fn every_prefix_of_witness_woven_text_measures_like_the_crate() {
  let tok = measuring_tok();
  let table = table(&tok);
  let witnesses = merge_rank_witnesses(&tok, &table, 200);
  assert!(!witnesses.is_empty());
  let mut rng = Rng(0x7212);
  let filler = cjk_run(40, 99);
  let filler: Vec<char> = filler.chars().collect();
  let mut texts: Vec<String> = Vec::new();
  // 1. witnesses back to back; 2. witnesses separated by one filler char;
  // 3. overlapping chains A B C B' C' ... where consecutive witnesses share a
  //    char when they can; 4. witnesses embedded in random filler.
  texts.push(
    witnesses
      .iter()
      .map(|&(a, b, c)| [a, b, c].iter().collect::<String>())
      .collect(),
  );
  texts.push(
    witnesses
      .iter()
      .enumerate()
      .map(|(i, &(a, b, c))| {
        let mut s: String = [a, b, c].iter().collect();
        s.push(filler[i % filler.len()]);
        s
      })
      .collect(),
  );
  {
    let mut s = String::new();
    for &(a, b, c) in &witnesses {
      if s.ends_with(a) {
        s.push(b);
        s.push(c);
      } else {
        s.push(a);
        s.push(b);
        s.push(c);
      }
    }
    texts.push(s);
  }
  {
    let mut s = String::new();
    for &(a, b, c) in witnesses.iter().take(80) {
      for _ in 0..rng.below(4) {
        s.push(filler[rng.below(filler.len())]);
      }
      s.push(a);
      s.push(b);
      s.push(c);
    }
    texts.push(s);
  }
  let mut cascaded = 0usize;
  for text in &texts {
    for start in [0usize, 3, 6, 9, 30] {
      let start = (start..).find(|&i| text.is_char_boundary(i)).unwrap();
      let (c, direct) = check_every_prefix(&tok, &table, text, start);
      cascaded += c;
      assert_eq!(direct, 0);
    }
  }
  assert!(
    cascaded > 0,
    "the witnesses must make the cascade walk fire"
  );
  println!(
    "[witness-corpus] texts={} cascaded_probes={cascaded}",
    texts.len()
  );
}
