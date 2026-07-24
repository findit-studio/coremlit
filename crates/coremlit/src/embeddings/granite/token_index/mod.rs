//! Single-pass token index over one `text` under the granite MEASURING tokenizer
//! (truncation disabled): tokenize the whole input ONCE, then answer every exact
//! range measure [`chunk_long`](super::chunk_long) needs without re-encoding more
//! than tiny edge fragments.
//!
//! # Why this exists
//!
//! windit's `ContentAware` packer measures every candidate byte range by its
//! token count, and its dominant cost is the growing-prefix probe
//! (`text[chunk_start..atom_i.end())` for each atom `i`), which re-encodes ~`m²/2`
//! bytes per `m`-atom chunk. Encoding the exact substring per query is correct but
//! quadratic; on a 4 MiB input the old closure re-encoded ~11× the input. This
//! index replaces that with one full encode plus O(log n) range answers.
//!
//! # Exactness contract (the non-negotiable invariant)
//!
//! [`TokenIndex::measure_range`]`(a, b)` returns **exactly**
//! `measure_tok.encode(&text[a..b], true).get_ids().len()` — the count the old
//! per-call closure returned — for every `a < b` on `char` boundaries. Byte-equal
//! measures at every windit/`attach_gaps` decision point ⇒ identical chunk
//! boundaries ⇒ identical `Vec<Chunk>` ⇒ (unchanged embed tail) bit-identical
//! embeddings. Output identity reduces to measure equality, which the layered
//! differential suite ([`tests`], `granite/tests.rs`) proves red-on-divergence.
//!
//! # Tokenizer facts the exactness argument rests on
//!
//! Guaranteed for the pinned granite `tokenizer.json` by #48's construction-time
//! contract + SHA-identity gate (production `chunk_long` only ever sees this one
//! artifact):
//! * `normalizer: null` → offsets are literal original bytes; no cross-boundary
//!   normalization.
//! * `pre_tokenizer` = `Sequence[ Split(GPT-4-style regex, Isolated),
//!   ByteLevel(add_prefix_space=false, use_regex=false) ]` → pre-tokens are the
//!   Split-regex pieces; they **tile** the text (every char matches some branch,
//!   Isolated drops nothing), and BPE merges never cross a pre-token boundary, so
//!   `encode(&text[a..b]).len()` is the sum of the per-pre-token BPE counts of
//!   `text[a..b]`'s OWN pre-tokenization.
//! * `post_processor` = TemplateProcessing `<|startoftext|> A <|return|>` →
//!   `encode(s, true).len() == encode(s, false).len() + 2` for any `s`.
//! * The Split regex has **no lookbehind**; its only branches that make a
//!   substring's pre-tokenization diverge from the full text's at a cut edge are
//!   the word branches' one leading `[^\r\n\p{L}\p{N}]?` char, `\p{N}{1,3}`
//!   (digits group in left-anchored triplets), and `\s+(?!\S)` (whitespace runs
//!   shaped by one char of lookahead). Those three are exactly the dirty-zone
//!   rules [`TokenIndex::measure_range`] applies.

use tokenizers::Tokenizer;

use crate::embeddings::granite::error::{Error, Result};

/// Counts the UTF-8 bytes handed to every internal `encode` on the measurement
/// path (index build + edge fragments + the zone-overlap/`direct_only` direct
/// arm + the windit slow-fallback), so the hermetic byte-ratio gate can prove the
/// single pass really replaced the old ~11× re-encode. Test-only; compiled out of
/// production entirely.
#[cfg(test)]
pub(crate) mod encode_meter {
  use std::cell::Cell;

  thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
  }

  /// Zero the per-thread counter before a measured run.
  pub(crate) fn reset() {
    BYTES.with(|b| b.set(0));
  }

  /// The bytes encoded since the last [`reset`].
  pub(crate) fn get() -> usize {
    BYTES.with(Cell::get)
  }

  /// Record `n` encoded bytes.
  pub(crate) fn add(n: usize) {
    BYTES.with(|b| b.set(b.get().saturating_add(n)));
  }
}

/// Content-token count (`add_special_tokens = false`) of `s`, the unit
/// `count_prefix` and the edge fragments are expressed in. Every direct encode on
/// the measurement path funnels through here so the test byte-meter sees all of
/// them; the caller adds the fixed `+ 2` template tokens once per whole range.
///
/// # Errors
/// [`Error::Tokenize`] if `s` fails to encode.
fn encode_content_len(tok: &Tokenizer, s: &str) -> Result<usize> {
  #[cfg(test)]
  encode_meter::add(s.len());
  tok
    .encode(s, false)
    .map(|e| e.get_ids().len())
    .map_err(Error::Tokenize)
}

/// Single-pass token index over one `text` (see the module docs).
///
/// Built from ONE `encode(text, add_special_tokens = false)`; the `Encoding` is
/// dropped once the three arrays are derived. Retained size is ~9 B per pre-token
/// (two `u32`s + a `bool`), freed when [`chunk_long`](super::chunk_long) returns.
pub(crate) struct TokenIndex {
  /// Exclusive byte end of pre-token `i` — the TRUE tiling boundaries
  /// (`start_0 = 0`, `start_i = pretoken_ends[i-1]`, last `== text.len()`),
  /// reconstructed by chaining the per-word END offsets. Ends are used, never the
  /// reported starts: ByteLevel `trim_offsets` can shrink a `Ġ`-word's reported
  /// START (dropping the leading space from its first token's offset) but never
  /// its end, and chaining ends recovers the starts the trim hid.
  pretoken_ends: Vec<u32>,
  /// `count_prefix[i]` = content tokens of pre-tokens `0..i` (len =
  /// `n_pretokens + 1`), from the run-lengths of `Encoding::get_word_ids()`. The
  /// interior of a range is a single subtraction over this.
  count_prefix: Vec<u32>,
  /// Pre-token `i` is entirely `\p{N}` chars (a member of a left-anchored digit
  /// run under the `\p{N}{1,3}` branch), so a cut inside it re-anchors the triplet
  /// grouping through to the run's end. `char::is_numeric` == Unicode general
  /// category `N` == the regex `\p{N}`.
  digit: Vec<bool>,
  /// Build-time tiling validation failed (a `None`/non-contiguous word id, a
  /// non-monotone end, a last end `!= text.len()`, or `text.len() > u32::MAX`).
  /// When set, every [`measure_range`](TokenIndex::measure_range) answers by a
  /// direct substring encode — today's exact behavior at today's cost.
  /// Unreachable for the pinned granite tokenizer on real text; insurance, not a
  /// path to design around.
  direct_only: bool,
}

impl TokenIndex {
  /// Tokenize `text` once with `measure_tok` (truncation + padding already
  /// disabled by the caller) and build the index. On any tiling anomaly the index
  /// is still returned, in `direct_only` mode.
  ///
  /// # Errors
  /// [`Error::Tokenize`] if the single full encode fails — the same variant the
  /// old path surfaced one call later from the per-chunk `token_ids`.
  pub(crate) fn build(measure_tok: &Tokenizer, text: &str) -> Result<Self> {
    #[cfg(test)]
    encode_meter::add(text.len());
    let enc = measure_tok.encode(text, false).map_err(Error::Tokenize)?;
    let offsets = enc.get_offsets();
    let word_ids = enc.get_word_ids();

    // `u32` arrays keep the index compact; a text past `u32::MAX` bytes (never
    // reachable behind `max_input_bytes`, and absurd without it) falls back to
    // direct encoding rather than truncating an offset.
    if text.len() > u32::MAX as usize {
      return Ok(Self::direct_only());
    }

    let mut pretoken_ends: Vec<u32> = Vec::new();
    let mut count_prefix: Vec<u32> = vec![0];
    let mut acc: u32 = 0;

    // Group consecutive tokens by word id (word ids are non-decreasing within an
    // encoding). A word's byte end is the max end offset over its tokens (its last
    // token's end); the count is its token count. Any `None` word id or a gap in
    // the 0,1,2,… sequence means the reconstruction cannot be trusted → direct.
    let mut expected: u32 = 0;
    let mut i = 0usize;
    let n_tokens = offsets.len();
    while i < n_tokens {
      let Some(wid) = word_ids[i] else {
        return Ok(Self::direct_only());
      };
      if wid != expected {
        return Ok(Self::direct_only());
      }
      let mut j = i;
      let mut end: usize = 0;
      while j < n_tokens && word_ids[j] == Some(wid) {
        end = end.max(offsets[j].1);
        j += 1;
      }
      pretoken_ends.push(end as u32);
      acc = acc.saturating_add((j - i) as u32);
      count_prefix.push(acc);
      expected += 1;
      i = j;
    }

    // Tiling checks: strictly increasing ends (each pre-token non-empty), and the
    // last end covers the whole input. Empty text tiles trivially (no pre-tokens).
    let mut prev: u32 = 0;
    for (k, &e) in pretoken_ends.iter().enumerate() {
      if k == 0 {
        if e == 0 {
          return Ok(Self::direct_only());
        }
      } else if e <= prev {
        return Ok(Self::direct_only());
      }
      prev = e;
    }
    match pretoken_ends.last() {
      Some(&last) if last as usize == text.len() => {}
      None if text.is_empty() => {}
      _ => return Ok(Self::direct_only()),
    }

    // Digit flags from the reconstructed byte ranges.
    let mut digit: Vec<bool> = Vec::with_capacity(pretoken_ends.len());
    let mut start = 0usize;
    for &e in &pretoken_ends {
      let word = &text[start..e as usize];
      digit.push(!word.is_empty() && word.chars().all(char::is_numeric));
      start = e as usize;
    }

    Ok(Self {
      pretoken_ends,
      count_prefix,
      digit,
      direct_only: false,
    })
  }

  /// The fail-safe index: every measure answers by direct substring encode.
  fn direct_only() -> Self {
    Self {
      pretoken_ends: Vec::new(),
      count_prefix: vec![0],
      digit: Vec::new(),
      direct_only: true,
    }
  }

  /// Exactly `tok.encode(&text[a..b], true).get_ids().len()` for `a <= b` on
  /// `char` boundaries of `text`, computed from the index plus at most two tiny
  /// edge-fragment encodes.
  ///
  /// The range decomposes as `left fragment ++ interior pre-tokens ++ right
  /// fragment`, where the interior aligns with the full-text pre-tokenization and
  /// its count is one prefix-sum subtraction. The fragments exist only where a cut
  /// falls inside a pre-token whose grouping the cut reshapes; the three
  /// dirty-zone rules pick fragment ends that are provably pre-token boundaries of
  /// BOTH the full parse and `text[a..b]`'s parse, so each fragment re-encodes to
  /// exactly its in-range contribution:
  ///
  /// 1. **Left.** `a` on a pre-token boundary ⇒ clean (no lookbehind: the parse
  ///    from a shared boundary reproduces the full parse). Else `a` is inside
  ///    pre-token `p`; its end is preserved under the cut for a letter, whitespace,
  ///    or (extend right through the run first, `while digit[p+1]`) digit tail. The
  ///    one class whose end can still dissolve is a punctuation/symbol tail with
  ///    text after it — cutting off `p`'s left context can leave a lone
  ///    `[^\r\n\p{L}\p{N}]` char that branch-1/2 pulls forward into the next word
  ///    (`" ("` → `"(a"`); that fragile case is detected cheaply and re-synced by a
  ///    bounded re-encode ([`left_resync`](TokenIndex::left_resync)). The left
  ///    fragment `[a, z)` is then re-encoded directly.
  /// 2. **Right.** `b` on a pre-token boundary ⇒ clean (truncating a parse at one
  ///    of its own boundaries cannot reshape earlier maximal-munch matches). Else
  ///    `b` is inside pre-token `q`; a cut inside/adjacent to a whitespace run
  ///    reshapes it via the `\s+(?!\S)` lookahead, so extend LEFT across the whole
  ///    contiguous whitespace-char run preceding `q`'s start (its start is a
  ///    boundary — the char before it is non-whitespace). The right fragment
  ///    `[y, b)` is re-encoded directly.
  /// 3. **Zones meet or cross** (`z >= y`, which subsumes every tiny range: single
  ///    atoms, gaps, `"\n\n"`) ⇒ measure the whole substring directly. Exact by
  ///    definition; these strings are small on the hot path.
  ///
  /// Over-extending a fragment to any boundary at or past the minimal safe one
  /// stays exact (forward determinism makes every full-parse boundary from the
  /// first re-synced one onward a substring boundary too), so the digit rule is
  /// safe even when it reaches past what a given cut strictly needs.
  ///
  /// # Errors
  /// [`Error::Tokenize`] if an edge-fragment or direct encode fails.
  pub(crate) fn measure_range(
    &self,
    tok: &Tokenizer,
    text: &str,
    a: usize,
    b: usize,
  ) -> Result<usize> {
    // Empty content is the two template specials. windit never queries an empty
    // range; this only guards the pointer-recovery edge and keeps `text[a..b]`
    // below from ever slicing backwards.
    if a >= b {
      return Ok(2);
    }
    if self.direct_only {
      return Ok(encode_content_len(tok, &text[a..b])? + 2);
    }

    let ends = &self.pretoken_ends;
    let n = ends.len();
    let a32 = a as u32;
    let b32 = b as u32;

    // ── Left zone ──
    // `z` = first pre-token boundary at/after `a` that is ALSO a boundary of
    // `text[a..]`'s own parse; `i` = interior's first word; `left_count` and/or a
    // deferred `left_fragment` supply the token count of `[a, z)`.
    let z: usize;
    let i: usize;
    let mut left_count: usize = 0;
    let mut left_fragment: Option<(usize, usize)> = None;
    if a == 0 {
      (z, i) = (0, 0);
    } else {
      // First pre-token whose end is past `a`.
      let p = ends.partition_point(|&e| e <= a32);
      if p > 0 && ends[p - 1] == a32 {
        // `a` is a boundary: the interior begins with the pre-token starting at
        // `a`, no fragment.
        (z, i) = (a, p);
      } else {
        // `a` is strictly inside pre-token `p`; extend right through the digit run
        // (harmless when `p` is not itself a digit — it only re-encodes more of an
        // already-exact prefix).
        let mut pp = p;
        while pp + 1 < n && self.digit[pp + 1] {
          pp += 1;
        }
        let e_pp = ends[pp] as usize;
        // `e_pp` is a substring boundary for a letter/whitespace/digit tail (those
        // self-resync at their end). It is FRAGILE only for a punctuation/symbol
        // tail with more text after it: cutting off `p`'s left context can leave a
        // lone `[^\r\n\p{L}\p{N}]` char that branch-1/2 pulls forward into the next
        // word (" (" → "(a"), or that merges with an adjacent punct run — dissolving
        // the boundary. The cheap test is the fragment's last char; a false positive
        // only costs one bounded re-encode.
        let last_char = text[..e_pp].chars().next_back();
        let fragile = e_pp < text.len()
          && matches!(last_char, Some(c) if !c.is_alphanumeric() && !c.is_whitespace());
        if !fragile {
          (z, i) = (e_pp, pp + 1);
          left_fragment = Some((a, e_pp));
        } else if let Some((zz, cnt)) = self.left_resync(tok, text, a, b, pp)? {
          z = zz;
          i = ends.partition_point(|&e| e <= zz as u32);
          left_count = cnt;
        } else {
          // No re-sync inside the bounded window (unreachable for the pinned
          // tokenizer): the always-exact whole-substring encode.
          return Ok(encode_content_len(tok, &text[a..b])? + 2);
        }
      }
    }

    // ── Right zone ──
    let (y, j, right_fragment): (usize, usize, Option<(usize, usize)>);
    if b == text.len() {
      (y, j, right_fragment) = (text.len(), n, None);
    } else {
      // First pre-token whose end is at or past `b`; it contains `b`.
      let q = ends.partition_point(|&e| e < b32);
      if ends[q] == b32 {
        // `b` is a boundary: the interior ends with the pre-token ending at `b`.
        (y, j, right_fragment) = (b, q + 1, None);
      } else {
        let y0 = if q == 0 { 0 } else { ends[q - 1] as usize };
        let yy = scan_back_whitespace(text, y0);
        let jj = ends.partition_point(|&e| e <= yy as u32);
        (y, j, right_fragment) = (yy, jj, Some((yy, b)));
      }
    }

    // ── Zones meet or cross → direct ──
    if z >= y {
      return Ok(encode_content_len(tok, &text[a..b])? + 2);
    }

    // ── Assemble: left fragment + interior prefix-sum + right fragment + 2 ──
    // `left_count` already holds the fragment count on the fragile re-sync path;
    // the non-fragile path defers its fragment encode to here.
    let mut total = left_count;
    if let Some((fa, fz)) = left_fragment {
      total += encode_content_len(tok, &text[fa..fz])?;
    }
    total += (self.count_prefix[j] - self.count_prefix[i]) as usize;
    if let Some((fy, fb)) = right_fragment {
      total += encode_content_len(tok, &text[fy..fb])?;
    }
    Ok(total + 2)
  }

  /// The fragile-boundary re-sync: re-encode a bounded window from `a` and return
  /// `(z, count)` where `z` is the first position that is a pre-token boundary of
  /// BOTH `text[a..]`'s parse (a word-id change) and the full parse, and `count`
  /// is the content-token count of `[a, z)` read off that same encode.
  ///
  /// The window ends two pre-tokens past the digit-extended `pp` (capped at `b`) —
  /// the forward-attachment reach is one pre-token, so a boundary that survives is
  /// found with margin, and `[a, z)`'s tokens cannot be reshaped by the window's
  /// own right edge (nothing spans the pre-token boundary `z`). `None` (unreachable
  /// for the pinned tokenizer) tells the caller to encode `[a, b)` whole.
  ///
  /// # Errors
  /// [`Error::Tokenize`] if the windowed encode fails.
  fn left_resync(
    &self,
    tok: &Tokenizer,
    text: &str,
    a: usize,
    b: usize,
    pp: usize,
  ) -> Result<Option<(usize, usize)>> {
    let ends = &self.pretoken_ends;
    let n = ends.len();
    let hi = (ends[(pp + 2).min(n - 1)] as usize).min(b);
    if hi <= a {
      return Ok(None);
    }
    #[cfg(test)]
    encode_meter::add(hi - a);
    let enc = tok.encode(&text[a..hi], false).map_err(Error::Tokenize)?;
    let offsets = enc.get_offsets();
    let word_ids = enc.get_word_ids();
    let m = offsets.len();
    for k in 0..m {
      let abs = a + offsets[k].1;
      // A pre-token boundary of the substring is where the next token's word id
      // differs (or the encode ends).
      let sub_boundary = k + 1 == m || word_ids[k + 1] != word_ids[k];
      if sub_boundary && abs > a && self.is_full_boundary(abs as u32) {
        return Ok(Some((abs, k + 1)));
      }
    }
    Ok(None)
  }

  /// Whether `pos` is a full-parse pre-token boundary (`0`, or an entry of
  /// `pretoken_ends`).
  fn is_full_boundary(&self, pos: u32) -> bool {
    pos == 0 || self.pretoken_ends.binary_search(&pos).is_ok()
  }
}

/// Start byte of the maximal run of `char::is_whitespace` chars ending at `from`
/// (a `char` boundary). `char::is_whitespace` == Unicode `White_Space` == the
/// regex `\s` here (empirically incl. NBSP `\u{A0}` and thin space `\u{2009}`),
/// so this recovers exactly the whitespace run the `\s+(?!\S)` lookahead reshapes
/// under a right-edge cut. Walks back at most the run length (short in real text).
fn scan_back_whitespace(text: &str, from: usize) -> usize {
  let mut y = from;
  for (idx, ch) in text[..from].char_indices().rev() {
    if ch.is_whitespace() {
      y = idx;
    } else {
      break;
    }
  }
  y
}

/// windit [`MeasureText`](windit::split::MeasureText) adapter backed by a
/// [`TokenIndex`]: recovers the `(a, b)` byte range of each queried subslice by
/// pointer arithmetic against `text` and answers from the index.
///
/// windit's five measure sites all pass `&text[a..b]` (descent whole-ranges,
/// char-fallback growing slices, the pack growing prefix, the overlap back-probe),
/// so `s.as_ptr() - text.as_ptr()` recovers `a` exactly; a foreign or zero-length
/// `&str` (never produced by windit) falls through to an exact slow encode.
pub(crate) struct IndexMeasure<'a> {
  text: &'a str,
  index: &'a TokenIndex,
  tok: &'a Tokenizer,
}

impl<'a> IndexMeasure<'a> {
  /// Adapter over `text`, its prebuilt `index`, and the measuring `tok`.
  pub(crate) fn new(text: &'a str, index: &'a TokenIndex, tok: &'a Tokenizer) -> Self {
    Self { text, index, tok }
  }
}

impl windit::split::MeasureText for IndexMeasure<'_> {
  fn measure(&self, s: &str) -> usize {
    // Recover the range by pointer offset. Live allocations are disjoint, so an
    // in-range offset can only mean `s` aliases `self.text`; anything else (a
    // foreign pointer, or an offset past the end) is not a windit subslice and
    // takes the exact slow path. No `unsafe`: this is arithmetic on addresses,
    // never a dereference.
    let base = self.text.as_ptr() as usize;
    let sp = s.as_ptr() as usize;
    if let Some(off) = sp.checked_sub(base)
      && off <= self.text.len()
      && off + s.len() <= self.text.len()
    {
      return self
        .index
        .measure_range(self.tok, self.text, off, off + s.len())
        .unwrap_or(usize::MAX);
    }
    // Unreachable from windit; fold an encode error to `usize::MAX` ("does not
    // fit"), exactly as the old closure did.
    self
      .tok
      .encode(s, true)
      .map(|e| e.get_ids().len())
      .unwrap_or(usize::MAX)
  }

  // `measure_within` keeps the default (full `measure` + compare): `measure` is
  // already O(log n) here, so the early-stop the trait offers buys nothing, and
  // the default trivially satisfies the "must agree with `measure`" contract that
  // keeps chunk boundaries put.
}

#[cfg(test)]
mod tests;
