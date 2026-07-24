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
//! * `pre_tokenizer` = `Sequence[ Split(o200k-style regex, Isolated),
//!   ByteLevel(add_prefix_space=false, use_regex=false) ]` → pre-tokens are the
//!   Split-regex pieces; they **tile** the text (every char matches some branch,
//!   Isolated drops nothing), and BPE merges never cross a pre-token boundary, so
//!   `encode(&text[a..b]).len()` is the sum of the per-pre-token BPE counts of
//!   `text[a..b]`'s OWN pre-tokenization.
//! * `post_processor` = TemplateProcessing `<|startoftext|> A <|return|>` →
//!   `encode(s, true).len() == encode(s, false).len() + 2` for any `s`.
//! * The Split regex, verbatim from the artifact, is the o200k_base pattern —
//!   materially richer than a plain GPT-2 split:
//!   ```text
//!   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//!   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//!   |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!   ```
//!   It has **no lookbehind**, so two deterministic parses agree forward from any
//!   shared boundary. The branches that make a substring's parse diverge from the
//!   full text's at a cut EDGE — exactly the [`TokenIndex::measure_range`]
//!   dirty-zone rules — are:
//!   1. the word branches' leading `[^\r\n\p{L}\p{N}]?` char (a lone punct/symbol
//!      pulled FORWARD into the next word once left context is cut) and their
//!      in-branch contraction suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` (whose letters
//!      rejoin the following word once the word before them is cut off); the word
//!      char classes glue `\p{L}∪\p{M}` — letters AND combining marks, wider than
//!      `char::is_alphanumeric`;
//!   2. `\p{N}{1,3}` — digits group in left-anchored triplets, re-anchored by a cut;
//!   3. ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — the punct branch, whose `[\r\n/]*` tail folds
//!      a trailing CR/LF/`/` run INTO the symbol pre-token (so a whitespace
//!      back-scan can walk into a symbol pre-token, not a real boundary);
//!   4. `\s+(?!\S)` vs `\s+` — a whitespace run is one pre-token only when nothing
//!      but whitespace follows; a NON-glue char after the run (a digit/punct/emoji
//!      — only a literal ' ' glues forward via rule 1) SPLITS it. A cut whose right
//!      edge lands in a run merges it (end-of-substring), so `b` on a pre-token
//!      boundary is not enough — the run must be re-encoded looking beyond `b`.

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
  /// `char` boundaries of `text`, computed from the index plus at most one bounded
  /// left-resync encode and one tiny right-fragment encode.
  ///
  /// The range decomposes as `left [a, z) ++ interior [z, y) ++ right [y, b)`,
  /// where the interior is a whole run of full-parse pre-tokens and its count is
  /// one prefix-sum subtraction. `z` and `y` are chosen to be pre-token boundaries
  /// of BOTH the full parse and `text[a..b]`'s parse (with no lookbehind, two
  /// deterministic parses agree forward from any shared boundary), so each edge
  /// re-encodes to exactly its in-range contribution:
  ///
  /// 1. **Left.** `a` on a pre-token boundary ⇒ clean. Else `a` is strictly inside
  ///    pre-token `p` and the cut can dissolve `p`'s end (a punct/symbol char
  ///    pulled forward into the next word, a contraction `'s`/`'t` suffix whose
  ///    letters rejoin the next word, a re-anchored digit triplet). No char-class
  ///    test separates safe cuts from fragile ones — `is_alphanumeric` is not the
  ///    regex's `\p{L}∪\p{M}∪\p{N}` glue set — so EVERY inside-`p` cut re-syncs by
  ///    a bounded re-encode ([`left_resync`](TokenIndex::left_resync)) that yields
  ///    `z` and the exact count of `[a, z)`; the window is first extended past any
  ///    digit run (`while digit[p+1]`) so a re-anchored run reaches its end.
  /// 2. **Right.** `b` is clean only when it is a full boundary AND `text[b-1]` is
  ///    non-whitespace. A whitespace char at `b-1` means a run touches `b`:
  ///    `text[a..b]` merges it under `\s+(?!\S)`, but the full parse may split it
  ///    at a non-glue char after `b` (only a literal ' ' glues forward), so even a
  ///    full boundary at `b` is NOT clean. Otherwise extend LEFT across the maximal
  ///    whitespace run touching `b`, then snap the landing DOWN to a true boundary
  ///    ([`snap_down_boundary`](TokenIndex::snap_down_boundary) — the back-scan can
  ///    stop inside the `[\r\n/]*` CRLF tail of a symbol pre-token); the right
  ///    fragment `[y, b)` is re-encoded directly.
  /// 3. **Zones meet or cross** (`z >= y`, which subsumes every tiny range: single
  ///    atoms, gaps, `"\n\n"`) ⇒ measure the whole substring directly. Exact by
  ///    definition; these strings are small on the hot path.
  ///
  /// Over-extending an edge to any boundary at or past the minimal safe one stays
  /// exact (forward determinism makes every full-parse boundary from the first
  /// re-synced one onward a substring boundary too), so the digit-run extension
  /// and the whitespace-run snap are safe even when they reach past what a given
  /// cut strictly needs.
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
    // `text[a..]`'s own parse; `i` = interior's first word; `left_count` supplies
    // the token count of `[a, z)`.
    let z: usize;
    let i: usize;
    let mut left_count: usize = 0;
    if a == 0 {
      (z, i) = (0, 0);
    } else {
      // First pre-token whose end is past `a`.
      let p = ends.partition_point(|&e| e <= a32);
      if p > 0 && ends[p - 1] == a32 {
        // `a` is a boundary: the interior begins with the pre-token starting at
        // `a`, no fragment (no lookbehind → the parse from a shared boundary
        // reproduces the full parse).
        (z, i) = (a, p);
      } else {
        // `a` is strictly INSIDE pre-token `p`, and the cut can dissolve `p`'s end
        // in ways the full parse hides: a punct/symbol char branch-1/2 pulls
        // forward into the next word (" (" → "(a"); a contraction `'s`/`'t` suffix
        // whose letters rejoin the following word (" it's"|"tation" → "station");
        // a digit triplet run re-anchored from the cut. No cheap char-class test
        // separates the safe cuts from the fragile ones — `is_alphanumeric` is not
        // the regex's `\p{L}∪\p{M}∪\p{N}` glue set — so classify nothing and
        // re-sync EVERY inside-`p` cut by a bounded re-encode. Extend the window
        // past any digit run first (`while digit[pp+1]`) so a re-anchored triplet
        // run reaches its end (its only shared boundary) inside the window.
        let mut pp = p;
        while pp + 1 < n && self.digit[pp + 1] {
          pp += 1;
        }
        if let Some((zz, cnt)) = self.left_resync(tok, text, a, b, pp)? {
          z = zz;
          i = ends.partition_point(|&e| e <= zz as u32);
          left_count = cnt;
        } else {
          // No shared boundary inside the bounded window (unreachable for the
          // pinned tokenizer): the always-exact whole-substring encode.
          return Ok(encode_content_len(tok, &text[a..b])? + 2);
        }
      }
    }

    // ── Right zone ──
    let (y, j, right_fragment): (usize, usize, Option<(usize, usize)>);
    if b == text.len() {
      // End of input: `text[a..b]` ends its parse exactly where the full parse
      // ends; nothing after `b` can reshape it.
      (y, j, right_fragment) = (text.len(), n, None);
    } else {
      // First pre-token whose end is at or past `b`; it contains `b`.
      let q = ends.partition_point(|&e| e < b32);
      // `b` is TRULY clean only when it is a full boundary AND the char just before
      // it is non-whitespace. A whitespace char at `b-1` means a run touches `b`:
      // `text[a..b]` ENDS the run, so `\s+(?!\S)` merges it, while the full parse
      // may split it at whatever follows `b` (a digit/punct/emoji takes no
      // space-glue — only a literal ' ' glues forward), so a full boundary at `b`
      // is not sufficient; extend LEFT across the run.
      let b_is_boundary = ends[q] == b32;
      let prev_ws = text[..b]
        .chars()
        .next_back()
        .is_some_and(char::is_whitespace);
      if b_is_boundary && !prev_ws {
        // `b` is a boundary and no run touches it: the interior ends with the
        // pre-token ending at `b`.
        (y, j, right_fragment) = (b, q + 1, None);
      } else {
        // Extend LEFT across the maximal whitespace run touching `b` — start at `b`
        // itself when a run splits on the boundary, else at the enclosing
        // pre-token's start — then snap the landing DOWN to a true full-parse
        // boundary: the back-scan can stop inside the `[\r\n/]*` tail the punct
        // branch folds into a symbol pre-token, which is not a boundary.
        let y0 = if b_is_boundary {
          b
        } else if q == 0 {
          0
        } else {
          ends[q - 1] as usize
        };
        let yy = self.snap_down_boundary(scan_back_whitespace(text, y0));
        let jj = ends.partition_point(|&e| e <= yy as u32);
        (y, j, right_fragment) = (yy, jj, Some((yy, b)));
      }
    }

    // ── Zones meet or cross → direct ──
    if z >= y {
      return Ok(encode_content_len(tok, &text[a..b])? + 2);
    }

    // ── Assemble: left count (from resync) + interior prefix-sum + right fragment
    // + 2 template specials ──
    let mut total = left_count;
    total += (self.count_prefix[j] - self.count_prefix[i]) as usize;
    if let Some((fy, fb)) = right_fragment {
      total += encode_content_len(tok, &text[fy..fb])?;
    }
    Ok(total + 2)
  }

  /// The inside-pre-token re-sync: re-encode a bounded window from `a` and return
  /// `(z, count)` where `z` is the first position that is a pre-token boundary of
  /// BOTH `text[a..]`'s parse (a word-id change) and the full parse, and `count`
  /// is the content-token count of `[a, z)` read off that same encode. EVERY cut
  /// strictly inside a pre-token takes this path — no char-class test pre-screens
  /// it — so a dissolved boundary (a forward-attached punct char, a contraction
  /// suffix's letters, a re-anchored digit triplet) is always re-synced, never
  /// trusted.
  ///
  /// The window ends two pre-tokens past the digit-extended `pp` (capped at `b`) —
  /// the forward-attachment reach is one pre-token, so a surviving boundary is
  /// found with margin. A boundary at the window's own right edge (`abs == hi`)
  /// with `hi < b` is REJECTED: the window's EOS can merge a whitespace run that
  /// `[a, b)` keeps split (the `\s+(?!\S)` mechanism), so it need not be a boundary
  /// of `[a, b)`'s parse and its count can be stale. `None` (unreachable for the
  /// pinned tokenizer) tells the caller to encode `[a, b)` whole.
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
        // A boundary at the window's own right edge (`abs == hi`) is trustworthy
        // only when the window IS the whole query (`hi == b`). With `hi < b` the
        // window's EOS can merge a whitespace run that `[a, b)` keeps split (the
        // `\s+(?!\S)` mechanism), so `abs` need not be a boundary of `[a, b)`'s
        // parse and `k + 1` may miscount — reject and encode `[a, b)` whole.
        if abs == hi && hi < b {
          return Ok(None);
        }
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

  /// The largest full-parse boundary `<= pos` (`0`, or an entry of
  /// `pretoken_ends`). A whitespace back-scan can land INSIDE a pre-token — the
  /// `[\r\n/]*` tail the punct branch folds into a symbol pre-token is whitespace
  /// yet interior — so the landing is snapped DOWN to that pre-token's start, a
  /// true boundary of both parses. Returns `pos` unchanged when it is already a
  /// boundary.
  fn snap_down_boundary(&self, pos: usize) -> usize {
    let c = self.pretoken_ends.partition_point(|&e| (e as usize) <= pos);
    if c == 0 {
      0
    } else {
      self.pretoken_ends[c - 1] as usize
    }
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
