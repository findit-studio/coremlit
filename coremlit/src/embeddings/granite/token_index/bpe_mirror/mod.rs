//! A mirror of the `tokenizers` BPE merge process that also records HOW the
//! tokens formed — the merge (pop) sequence and each final token's spines —
//! which is what [`super::suffix_session`] needs to decide, without re-encoding,
//! whether a cut inside one pre-token is a boundary of the substring's own
//! tokenization.
//!
//! # What is mirrored, exactly
//!
//! `tokenizers::models::bpe::Word::merge_all` (0.23): every adjacent pair that
//! is in the merge table goes into a min-heap keyed by `(rank, position)`; the
//! minimum is popped, skipped if stale (its left symbol is dead, is the last
//! symbol, or no longer forms the pair the entry was pushed for), else merged,
//! after which the pair with the previous symbol and the pair with the next
//! symbol are pushed. Positions are the ORIGINAL symbol indices, so "leftmost
//! first" among equal ranks means what it means in the crate. `merge_word`'s
//! symbol construction is mirrored for the artifact's configuration only: no
//! `continuing_subword_prefix`, no `end_of_word_suffix`, no `unk_token`, no
//! `byte_fallback`, no dropout — [`MergeTable::from_tokenizer`] refuses any
//! other configuration. A byte whose byte-level single-char token the
//! vocabulary lacks is one the crate silently DROPS; [`MergeTable::process`]
//! refuses a word containing one instead (the index's byte-coverage guard has
//! already routed such a text to direct encoding, so none reaches here).
//!
//! `ignore_merges` (a whole word that is itself a vocabulary entry is returned
//! as one token, no merges) is NOT applied by [`MergeTable::process`]: this
//! module models the merge PROCESS as it runs inside a longer word, where the
//! shortcut never fires for a sub-range. The caller applies the shortcut to the
//! whole query itself through [`MergeTable::whole_word_id`] — the MODEL
//! vocabulary, the map `BPE::tokenize` reads for `ignore_merges`, never
//! `Tokenizer::token_to_id`, which resolves the ADDED vocabulary first: an
//! added token whose content is the byte-level spelling of a word would answer
//! that lookup while the crate, matching added tokens against the raw text and
//! finding no such literal, runs the merges.
//!
//! # Where the table comes from
//!
//! `BPE::merges` is crate-private in `tokenizers`, so the table is rebuilt from
//! the model's serde form (`serde_json::to_value` of the loaded model — the same
//! JSON the artifact file holds), pairing each merge's two strings through the
//! vocabulary exactly as the crate's builder does, later duplicates of a pair
//! overwriting earlier ones as `HashMap::insert` does there. Built once per
//! embedder, on the fast lane's FIRST ENGAGEMENT (the first probe longer than
//! any token can be into a QUALIFYING pre-token — see the index docs), not on
//! the first `embed_long`: about half a second and ~74 MB transient (the serde
//! value of the whole model plus a vocabulary clone), retaining the merge map
//! alone — one 16-byte entry per merge, about 10 MB with the map's overhead
//! for the artifact's 413,540 merges — for the embedder's lifetime.
//!
//! # The configuration the lane is pinned to
//!
//! [`MergeTable::from_tokenizer`] returns `None` — the lane stays off and every
//! probe keeps the exact index path — unless the tokenizer is EXACTLY what the
//! lane's arguments are about: a BPE with `ignore_merges` (the whole-word
//! vocabulary lookups mirror it), no dropout / unk / affixes / byte fallback
//! (the merge process is mirrored for that configuration only), NO normalizer
//! (the bytes tokenized are the text's own, which the index's pointer/offset
//! recovery reads), and the pre-tokenizer `Split(`[`SPLIT_PATTERN`]`,
//! Isolated) → ByteLevel { add_prefix_space: false, use_regex: false }`
//! verbatim — the lane's engagement gate is a lemma about THAT pattern's two
//! letter branches, and a second regex pass or a prefixed space would void it.
//! NOT pinned here: the post-processor. The lane's `+ 2` for the template
//! specials — inherited from `measure_range`, whose count it must equal — is
//! held by #48's tokenizer SHA gate, not by this table.

use std::{
  cmp::Reverse,
  collections::{BinaryHeap, HashMap},
};

use tokenizers::{Model, ModelWrapper, Tokenizer, utils::SysRegex};

/// The Split regex of the pre-tokenizer the lane is built for — o200k's,
/// pinned verbatim from the artifact. Its first two alternatives are the
/// letter branches `LEAD? HEAD* TAIL+ SUFFIX?` and `LEAD? HEAD+ TAIL* SUFFIX?`
/// ([`LEAD_CLASS`], [`HEAD_CLASS`], [`TAIL_CLASS`], and the apostrophe
/// contractions), in that order; the engagement lemma in the index docs is
/// about exactly this pattern, and [`MergeTable::from_tokenizer`] refuses any
/// other.
pub(crate) const SPLIT_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// The letter branches' optional single leading char: anything that is not a
/// letter, a digit, CR or LF. Named for the test that pins the pattern's
/// structure; the lane itself needs only [`TAIL_CLASS`].
#[cfg(test)]
pub(crate) const LEAD_CLASS: &str = r"[^\r\n\p{L}\p{N}]";
/// The letter branches' HEAD class — note `\p{Lu}` and `\p{Lt}`, which the
/// tail class lacks. Named for the test that pins the pattern's structure.
#[cfg(test)]
pub(crate) const HEAD_CLASS: &str = r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]";
/// The letter branches' TAIL class: the chars a pre-token may hold after its
/// first for the lane to engage on it ([`TailClass`]).
pub(crate) const TAIL_CLASS: &str = r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]";

/// [`TAIL_CLASS`] anchored to one whole string, compiled by the regex engine
/// the tokenizer's own Split runs on — hence over the same Unicode tables —
/// and independent of the merge table, so the lane can test a pre-token
/// BEFORE deciding whether the table is worth building.
pub(crate) struct TailClass(SysRegex);

impl TailClass {
  /// Compile the class; `None` only if the engine refuses the constant.
  pub(crate) fn new() -> Option<Self> {
    SysRegex::new(&format!(r"\A{TAIL_CLASS}\z")).ok().map(Self)
  }

  /// Whether `c` is in the tail class — decided by the tokenizer's own regex
  /// engine, so exactly as its Split sees it.
  pub(crate) fn contains(&self, c: char) -> bool {
    let mut buf = [0u8; 4];
    self.0.find_iter(c.encode_utf8(&mut buf)).next().is_some()
  }
}

/// The pre-tokenizer configuration the lane is pinned to, in the tokenizer's
/// serde form (what `serde_json::to_value` of the loaded pre-tokenizer yields,
/// and what the artifact file holds).
pub(crate) fn expected_pre_tokenizer() -> serde_json::Value {
  serde_json::json!({
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": { "Regex": SPLIT_PATTERN },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
      }
    ]
  })
}

/// One executed merge of the process, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pop {
  /// The merge's rank (its index in the merge list; the heap key).
  pub(crate) rank: u32,
  /// Exclusive byte end, relative to the word start, of the token the merge
  /// produced. A pop lies left of a boundary `q` of the final tokenization iff
  /// `end <= q` — tokens never straddle a final boundary.
  pub(crate) end: u32,
  /// The vocabulary id of the produced token.
  pub(crate) new_id: u32,
}

/// The recorded process of one word: its final tokens and every merge that
/// formed them, plus each final token's two spines.
#[derive(Debug, Clone, Default)]
pub(crate) struct WordRun {
  /// Exclusive byte ends of the final tokens, relative to the word start,
  /// strictly increasing, last `== word.len()`.
  pub(crate) ends: Vec<u32>,
  /// The final tokens' vocabulary ids, parallel to `ends`.
  pub(crate) ids: Vec<u32>,
  /// Every executed merge, in execution order.
  pub(crate) pops: Vec<Pop>,
  /// Per final token, the pop indices along its RIGHT spine, increasing: the
  /// merges that successively produced the token containing the token's LAST
  /// byte. State `j` of that byte's token is the initial single-byte symbol for
  /// `j == 0` and `pops[rspine[j - 1]].new_id` after; it lives until pop
  /// `rspine[j]` absorbs it.
  pub(crate) rspine: Vec<Vec<u32>>,
  /// Per final token, the pop indices along its LEFT spine — the same for the
  /// token containing the token's FIRST byte.
  pub(crate) lspine: Vec<Vec<u32>>,
}

/// The merge table of the artifact's BPE, plus the byte-level symbol map.
pub(crate) struct MergeTable {
  /// `(left id, right id) -> (rank, produced id)`.
  merges: HashMap<(u32, u32), (u32, u32)>,
  /// Raw byte -> vocabulary id of its byte-level single-char token, `None`
  /// for a byte the vocabulary lacks (the crate DROPS such a char; the index's
  /// byte-coverage guard keeps any text containing one off this path, and
  /// [`MergeTable::process`] refuses such a word regardless).
  byte_symbol: [Option<u32>; 256],
  /// Raw byte -> its byte-level char, for spelling a word the way the vocabulary
  /// does (the whole-word lookup).
  byte_char: [char; 256],
  /// The longest token in bytes the crate can emit — the model's vocabulary
  /// (byte-level chars, one per byte) or an added token (raw bytes) — so a
  /// whole-word lookup is skipped past it, and the measure floor's per-token
  /// byte bound is over everything the crate can produce.
  max_token_bytes: usize,
}

impl core::fmt::Debug for MergeTable {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("MergeTable")
      .field("merges", &self.merges.len())
      .field("max_token_bytes", &self.max_token_bytes)
      .finish_non_exhaustive()
  }
}

impl MergeTable {
  /// Rebuild the merge table from the tokenizer's loaded BPE model. `None` when
  /// the model is not a BPE, when the tokenizer is not the configuration the
  /// lane is pinned to (see the module docs: `ignore_merges`, no normalizer,
  /// the exact Split + ByteLevel pre-tokenizer), or when its serde form cannot
  /// be read — in every case the caller simply keeps today's direct-encode
  /// measurement. Not pinned: the post-processor — the lane's `+ 2` for the
  /// template specials is what `measure_range` already assumes, held by #48's
  /// tokenizer SHA gate (with `post_processor: null` the lane and the index
  /// agree with each other and both differ from a direct encode).
  pub(crate) fn from_tokenizer(tok: &Tokenizer) -> Option<Self> {
    let bpe = match tok.get_model() {
      ModelWrapper::BPE(bpe) => bpe,
      _ => return None,
    };
    if bpe.dropout.is_some_and(|d| d > 0.0)
      || bpe.unk_token.is_some()
      || bpe.continuing_subword_prefix.is_some()
      || bpe.end_of_word_suffix.is_some()
      || bpe.byte_fallback
      || !bpe.ignore_merges
      || tok.get_normalizer().is_some()
    {
      return None;
    }
    if serde_json::to_value(tok.get_pre_tokenizer()?).ok()? != expected_pre_tokenizer() {
      return None;
    }
    let vocab: HashMap<String, u32> = bpe.get_vocab();
    let value = serde_json::to_value(tok.get_model()).ok()?;
    let merges_json = value.get("merges")?.as_array()?;
    let mut merges: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(merges_json.len());
    for (rank, entry) in merges_json.iter().enumerate() {
      // Two serde spellings exist: `["a", "b"]` (current) and `"a b"` (legacy).
      let (a, b) = match entry {
        serde_json::Value::Array(pair) if pair.len() == 2 => {
          (pair[0].as_str()?.to_owned(), pair[1].as_str()?.to_owned())
        }
        serde_json::Value::String(s) => {
          let (a, b) = s.split_once(' ')?;
          (a.to_owned(), b.to_owned())
        }
        _ => return None,
      };
      let a_id = *vocab.get(&a)?;
      let b_id = *vocab.get(&b)?;
      let new_id = *vocab.get(&format!("{a}{b}"))?;
      merges.insert((a_id, b_id), (u32::try_from(rank).ok()?, new_id));
    }
    let byte_char = bytes_char();
    let mut byte_symbol = [None; 256];
    for b in 0..=255u8 {
      let mut s = String::new();
      s.push(byte_char[b as usize]);
      byte_symbol[b as usize] = vocab.get(&s).copied();
    }
    // One byte-level char per original byte, so a token's byte length is its
    // char count — over the model's vocabulary AND the added vocabulary, since
    // the crate can emit either (an added literal in the text sends the index
    // to direct encoding anyway, but the bound must not depend on that).
    let added = tok.get_added_vocabulary().get_vocab();
    let max_token_bytes = vocab
      .keys()
      .map(|k| k.chars().count())
      .chain(added.keys().map(|k| k.len()))
      .max()?;
    Some(Self {
      merges,
      byte_symbol,
      byte_char,
      max_token_bytes,
    })
  }

  /// The MODEL vocabulary's id of `word` as one token — the lookup
  /// `ignore_merges` makes (`BPE::tokenize` reads the model's `vocab` alone),
  /// which every whole-word shortcut of the lane must mirror. NOT
  /// `Tokenizer::token_to_id`: that resolves the ADDED vocabulary first, and
  /// an added token whose content is the byte-level spelling of a word
  /// (`"Ġzzqxjkw"` for `" zzqxjkw"`) would answer it while the crate, which
  /// matches added tokens against the RAW text and finds no such literal,
  /// runs the merges. A word longer than any token is never one.
  pub(crate) fn whole_word_id(&self, tok: &Tokenizer, word: &[u8]) -> Option<u32> {
    if word.len() > self.max_token_bytes {
      return None;
    }
    tok.get_model().token_to_id(&self.spell(word))
  }

  /// The number of merges the table holds (its retained footprint is one
  /// 16-byte entry each, plus the map's overhead).
  #[cfg(test)]
  pub(crate) fn merge_count(&self) -> usize {
    self.merges.len()
  }

  /// The longest token the crate can emit, in bytes (model and added
  /// vocabularies alike).
  #[inline]
  pub(crate) const fn max_token_bytes(&self) -> usize {
    self.max_token_bytes
  }

  /// The merge rank and produced id of the pair `(left, right)`, if it is one.
  #[inline]
  pub(crate) fn merge(&self, left: u32, right: u32) -> Option<(u32, u32)> {
    self.merges.get(&(left, right)).copied()
  }

  /// The vocabulary id of `byte`'s byte-level single-char token, `None` when
  /// the vocabulary lacks it.
  #[inline]
  pub(crate) fn symbol(&self, byte: u8) -> Option<u32> {
    self.byte_symbol[byte as usize]
  }

  /// `word` spelled as the vocabulary spells it (byte-level chars), for the
  /// whole-word lookup that mirrors `ignore_merges`.
  pub(crate) fn spell(&self, word: &[u8]) -> String {
    word.iter().map(|&b| self.byte_char[b as usize]).collect()
  }

  /// Run the merge process over `word` (raw bytes, one symbol per byte) and
  /// record it. Never applies the whole-word shortcut (see the module docs).
  /// `None` when a byte has no single-char token — the crate would drop that
  /// char, which this process does not model.
  pub(crate) fn process(&self, word: &[u8]) -> Option<WordRun> {
    struct Sym {
      c: u32,
      prev: i32,
      next: i32,
      len: u32,
      start: u32,
      lspine: Vec<u32>,
      rspine: Vec<u32>,
    }
    let n = word.len();
    let mut syms: Vec<Sym> = Vec::with_capacity(n);
    for (i, &b) in word.iter().enumerate() {
      syms.push(Sym {
        c: self.symbol(b)?,
        prev: i as i32 - 1,
        next: if i + 1 < n { (i + 1) as i32 } else { -1 },
        len: 1,
        start: i as u32,
        lspine: Vec::new(),
        rspine: Vec::new(),
      });
    }
    // Min-heap on (rank, pos): the crate's `Merge` ordering.
    let mut heap: BinaryHeap<Reverse<(u32, u32, u32)>> = BinaryHeap::with_capacity(n);
    for i in 0..n.saturating_sub(1) {
      if let Some((rank, new_id)) = self.merge(syms[i].c, syms[i + 1].c) {
        heap.push(Reverse((rank, i as u32, new_id)));
      }
    }
    let mut pops: Vec<Pop> = Vec::new();
    while let Some(Reverse((rank, pos, new_id))) = heap.pop() {
      let p = pos as usize;
      if syms[p].len == 0 || syms[p].next == -1 {
        continue;
      }
      let nx = syms[p].next as usize;
      match self.merge(syms[p].c, syms[nx].c) {
        Some((_, id)) if id == new_id => {}
        _ => continue,
      }
      // Merge slot `nx` into slot `p`.
      let right_len = syms[nx].len;
      let right_next = syms[nx].next;
      let right_rspine = std::mem::take(&mut syms[nx].rspine);
      syms[nx].len = 0;
      let pop_idx = pops.len() as u32;
      {
        let left = &mut syms[p];
        left.c = new_id;
        left.len += right_len;
        left.next = right_next;
        left.rspine = right_rspine;
        left.rspine.push(pop_idx);
        left.lspine.push(pop_idx);
      }
      if right_next >= 0 {
        syms[right_next as usize].prev = p as i32;
      }
      pops.push(Pop {
        rank,
        end: syms[p].start + syms[p].len,
        new_id,
      });
      let prev = syms[p].prev;
      if prev >= 0 {
        let pv = prev as usize;
        if let Some((r, id)) = self.merge(syms[pv].c, syms[p].c) {
          heap.push(Reverse((r, pv as u32, id)));
        }
      }
      let next = syms[p].next;
      if next >= 0 {
        let nn = next as usize;
        if let Some((r, id)) = self.merge(syms[p].c, syms[nn].c) {
          heap.push(Reverse((r, p as u32, id)));
        }
      }
    }
    let mut run = WordRun {
      pops,
      ..WordRun::default()
    };
    for s in syms.into_iter().filter(|s| s.len != 0) {
      run.ends.push(s.start + s.len);
      run.ids.push(s.c);
      run.rspine.push(s.rspine);
      run.lspine.push(s.lspine);
    }
    Some(run)
  }
}

/// The byte-level map the crate's `ByteLevel` pre-tokenizer uses: printable
/// bytes map to themselves, the rest to `U+0100 + n` in order.
fn bytes_char() -> [char; 256] {
  let mut printable = [false; 256];
  for b in b'!'..=b'~' {
    printable[b as usize] = true;
  }
  for b in 0xA1u8..=0xAC {
    printable[b as usize] = true;
  }
  for b in 0xAEu8..=0xFF {
    printable[b as usize] = true;
  }
  let mut out = ['\0'; 256];
  let mut n = 0u32;
  for b in 0..=255usize {
    out[b] = if printable[b] {
      char::from(b as u8)
    } else {
      let c = char::from_u32(256 + n).expect("256..512 are scalar values");
      n += 1;
      c
    };
  }
  out
}

#[cfg(test)]
mod tests;
