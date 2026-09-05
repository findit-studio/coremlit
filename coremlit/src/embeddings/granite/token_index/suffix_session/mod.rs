//! The separatorless fast lane: exact prefix measures inside ONE pre-token
//! without re-encoding the growing prefix.
//!
//! # The problem this solves
//!
//! On text the Split regex parses as a single document-spanning pre-token
//! (unspaced CJK, `"x".repeat(n)`), [`super::TokenIndex`] has no interior
//! boundary, so every packer probe `[a, b)` with `a` inside that pre-token
//! re-encoded `[a, b)` whole — quadratic in the chunk length, 500–2000× the
//! input (#72). The probes come in one shape: a FIXED start `a` (the chunk
//! start) and an end `b` that grows one atom at a time. This module answers them
//! from one recorded merge process over the suffix `Z = text[a..]` (capped, see
//! [`Session::build`]) plus, per probe, a tiny re-run over the last token.
//!
//! # Why a cut inside a pre-token is hard
//!
//! Inside one pre-token the count is the BPE token count of the byte string,
//! and BPE is not prefix-stable: with `rank(bc) < rank(ab)`, `encode("ab")` is
//! not a prefix of `encode("abc")`. What IS true is the independence lemma —
//! stated and proved on [`Session::cut_is_boundary`]: if `q` is a boundary of
//! the tokenization of `w`, the two sides tokenize as they would in isolation,
//! so `T(w) = T(w[..q]) ++ T(w[q..])`. For a probe `Z_b = Z[..b]` the question
//! is therefore only: which boundary `q` of `T(Z)` is ALSO a boundary of
//! `T(Z_b)`? Then `count(Z_b) = (tokens of T(Z) before q) + |T(Z[q..b])|`, the
//! second term a re-run over at most one token's worth of bytes. Deciding that
//! needs the merge PROCESS, not the final tokens — which is what
//! [`super::bpe_mirror`] records.
//!
//! # Exactness guards, in production
//!
//! * The session's process over `Z` is checked token-for-token against the
//!   crate's own `encode` of the same bytes; a mismatch voids the session and
//!   the query takes today's direct encode.
//! * `ignore_merges` — a whole query that is itself a vocabulary entry is one
//!   token, whatever the merges say — is applied through the tokenizer's own
//!   vocabulary lookup before any process result is consulted.
//! * A cascade that walks back to the start of `Z` is answered by the direct
//!   encode, never by a guess.

use tokenizers::Tokenizer;

use super::bpe_mirror::{MergeTable, WordRun};

/// Test-only record of every session built — `(start, end)` in order — so a
/// gate can see how many suffix encodes a chunking really performed and from
/// which starts, not just their byte total.
#[cfg(test)]
pub(crate) mod build_meter {
  use std::cell::RefCell;

  thread_local! {
    static BUILDS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
  }

  /// Forget every recorded build.
  pub(crate) fn reset() {
    BUILDS.with(|b| b.borrow_mut().clear());
  }

  /// The builds recorded since the last [`reset`].
  pub(crate) fn builds() -> Vec<(usize, usize)> {
    BUILDS.with(|b| b.borrow().clone())
  }

  pub(crate) fn add(start: usize, end: usize) {
    BUILDS.with(|b| b.borrow_mut().push((start, end)));
  }
}
use crate::embeddings::granite::error::{Error, Result};

/// Initial suffix cap: how many bytes of `text[a..]` one session records. A
/// window of `MAX_TOKENS` tokens is far shorter than this on every real
/// tokenizer (tokens are ≥ 1 byte), and a probe past the cap rebuilds a session
/// twice as long, so the cap is a cost bound, not a limit.
pub(crate) const INITIAL_CAP: usize = 8192;

/// A range-max tree over the recorded pops of `Z`, holding `rank + 1` for an
/// ACTIVE pop (one whose token ends at or before the current cut `q`, i.e. a pop
/// of the left side `Z[..q]`) and `0` otherwise. Pops become active as the cut
/// moves right and never deactivate, because the cut only moves right across
/// the probes of one session.
struct MaxTree {
  size: usize,
  node: Vec<u32>,
}

impl MaxTree {
  fn new(len: usize) -> Self {
    let size = len.max(1).next_power_of_two();
    Self {
      size,
      node: vec![0; 2 * size],
    }
  }

  fn set(&mut self, i: usize, v: u32) {
    let mut k = i + self.size;
    self.node[k] = v;
    while k > 1 {
      k /= 2;
      self.node[k] = self.node[2 * k].max(self.node[2 * k + 1]);
    }
  }

  /// Max over `[lo, hi)`; `0` when empty.
  fn max(&self, lo: usize, hi: usize) -> u32 {
    let (mut l, mut r) = (lo + self.size, hi.min(self.size) + self.size);
    let mut best = 0;
    while l < r {
      if l & 1 == 1 {
        best = best.max(self.node[l]);
        l += 1;
      }
      if r & 1 == 1 {
        r -= 1;
        best = best.max(self.node[r]);
      }
      l /= 2;
      r /= 2;
    }
    best
  }

  /// Largest index `k` with `max(0..=k) <= t`, as `Option<usize>` (`None` when
  /// even index 0 exceeds `t`). Prefix maxima are monotone, so this bisects.
  fn last_prefix_at_most(&self, t: u32, len: usize) -> Option<usize> {
    if len == 0 || self.max(0, 1) > t {
      return None;
    }
    let (mut lo, mut hi) = (0usize, len - 1);
    while lo < hi {
      let mid = lo + (hi - lo).div_ceil(2);
      if self.max(0, mid + 1) <= t {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    Some(lo)
  }
}

/// One recorded suffix `Z = text[start..end]` and the machinery to answer exact
/// prefix counts over it.
pub(crate) struct Session {
  /// Absolute byte start of `Z` — the chunk start every probe shares.
  start: usize,
  /// Absolute exclusive byte end of `Z` (the cap, or the pre-token's end).
  end: usize,
  /// `text[start..end]`, owned so the session carries no borrow.
  word: Vec<u8>,
  /// The recorded merge process over `word`.
  z: WordRun,
  /// Pop indices of `z` ordered by their token's byte end — the activation order.
  by_end: Vec<u32>,
  /// How many of `by_end` are active; `active_q` is the cut they were activated
  /// for.
  activated: usize,
  active_q: u32,
  tree: MaxTree,
  /// How many cuts the last probe's cascade walk moved left before a cut was
  /// certified (0: the cut at the token containing `b` was a boundary, or the
  /// probe was answered before any walk). Test-visible, so the refusal branch
  /// of the certificate is provably exercised.
  last_cascade_depth: usize,
}

/// What a probe resolved to.
pub(crate) enum Prefix {
  /// The exact content-token count of `text[start..b]`.
  Count(usize),
  /// `b` lies past this session's cap; the caller rebuilds a longer session.
  PastCap,
  /// The cascade reached the start of `Z`: encode the whole probe directly.
  Direct,
}

impl Session {
  /// Absolute byte start of `Z`.
  pub(crate) const fn start(&self) -> usize {
    self.start
  }

  /// Absolute exclusive byte end of `Z`.
  pub(crate) const fn end(&self) -> usize {
    self.end
  }

  /// Record the process over `text[start..end]`, or `None` when the fast lane
  /// must not be used for this suffix: the suffix is itself a vocabulary entry
  /// (`ignore_merges` would answer, and the process view is not what the
  /// tokenizer runs), or the mirrored process disagrees with the tokenizer's
  /// own tokenization of the same bytes.
  ///
  /// # Errors
  /// [`Error::Tokenize`] if the checking encode fails.
  pub(crate) fn build(
    tok: &Tokenizer,
    table: &MergeTable,
    text: &str,
    start: usize,
    end: usize,
  ) -> Result<Option<Self>> {
    let word = text.as_bytes()[start..end].to_vec();
    if word.is_empty() {
      return Ok(None);
    }
    if table.whole_word_id(tok, &word).is_some() {
      return Ok(None);
    }
    let Some(z) = table.process(&word) else {
      return Ok(None);
    };
    // Cross-check against the tokenizer: the same ids in the same order. Equal
    // ids fix the byte ends (every id has one byte length), so the ends need no
    // separate comparison — and could not get one: the crate reports a byte
    // token that sits inside a multi-byte char at the CHAR's offsets, not the
    // byte's. An encode error is "cannot use", never "use".
    #[cfg(test)]
    super::encode_meter::add(end - start);
    let enc = tok
      .encode(&text[start..end], false)
      .map_err(Error::Tokenize)?;
    if enc.get_ids() != z.ids.as_slice() {
      return Ok(None);
    }
    #[cfg(test)]
    build_meter::add(start, end);
    let mut by_end: Vec<u32> = (0..z.pops.len() as u32).collect();
    by_end.sort_by_key(|&i| z.pops[i as usize].end);
    let tree = MaxTree::new(z.pops.len());
    Ok(Some(Self {
      start,
      end,
      word,
      z,
      by_end,
      activated: 0,
      active_q: 0,
      tree,
      last_cascade_depth: 0,
    }))
  }

  /// Make the active set exactly the pops whose token ends at or before `q`
  /// (relative). `by_end` orders pops by end, so moving the cut right activates
  /// a prefix of it and moving it left — the cascade walk stepping one token
  /// back — deactivates the tail, each pop touched once per move.
  fn activate_upto(&mut self, q: u32) {
    while self.activated < self.by_end.len() {
      let i = self.by_end[self.activated] as usize;
      if self.z.pops[i].end > q {
        break;
      }
      self.tree.set(i, self.z.pops[i].rank + 1);
      self.activated += 1;
    }
    while self.activated > 0 {
      let i = self.by_end[self.activated - 1] as usize;
      if self.z.pops[i].end <= q {
        break;
      }
      self.tree.set(i, 0);
      self.activated -= 1;
    }
    self.active_q = q;
  }

  /// The exact content-token count of `text[start..b]`.
  ///
  /// Order of decision: the whole-word shortcut (the tokenizer's own vocabulary,
  /// mirroring `ignore_merges`); a `b` on a boundary of `T(Z)` (the independence
  /// lemma: the prefix tokenizes as its own tokens); otherwise the cascade walk
  /// of [`Self::cut_is_boundary`] from the token containing `b` leftward.
  pub(crate) fn measure_prefix(&mut self, tok: &Tokenizer, table: &MergeTable, b: usize) -> Prefix {
    self.last_cascade_depth = 0;
    if b > self.end {
      return Prefix::PastCap;
    }
    let rel = b - self.start;
    if rel == 0 {
      return Prefix::Count(0);
    }
    let rel32 = rel as u32;
    if table.whole_word_id(tok, &self.word[..rel]).is_some() {
      return Prefix::Count(1);
    }
    // Index of the first token whose end is >= rel: the token containing rel,
    // or the token ending exactly at rel.
    let t = self.z.ends.partition_point(|&e| e < rel32);
    if self.z.ends[t] == rel32 {
      return Prefix::Count(t + 1);
    }
    // `t` tokens end strictly before `rel`; the cut candidates are their ends,
    // walked leftward while the cut proves not to be a boundary of `T(Z_b)`.
    let mut cut_token = t; // the cut is the START of token `cut_token`
    loop {
      if cut_token == 0 {
        return Prefix::Direct;
      }
      let q = self.z.ends[cut_token - 1];
      let Some(y) = table.process(&self.word[q as usize..rel]) else {
        return Prefix::Direct;
      };
      if self.cut_is_boundary(table, q, cut_token - 1, &y) {
        return Prefix::Count(cut_token + y.ends.len());
      }
      cut_token -= 1;
      self.last_cascade_depth += 1;
    }
  }

  /// Cuts the last probe's cascade walk moved left (see the field).
  #[cfg(test)]
  pub(crate) const fn last_cascade_depth(&self) -> usize {
    self.last_cascade_depth
  }

  /// Whether `q` — a boundary of `T(Z)` at the end of token `x_token` — is also
  /// a boundary of `T(Z[..b])`, where `y` is the recorded process over
  /// `Y = Z[q..b]` in isolation.
  ///
  /// # The argument
  ///
  /// **Independence.** BPE pops the pending adjacent pair of least `(rank,
  /// position)` and merges it, until none is pending. Take `w = XY`. As long as
  /// no pop has merged across the `X|Y` seam, the pending pairs are the X-pairs,
  /// the Y-pairs and the one seam pair `(x, y)` formed by X's last and Y's first
  /// token; the X-pairs' pending set equals what X's isolated process has pending
  /// at the same number of X-pops, likewise Y, because pops on one side never
  /// touch the other. So, until a seam merge, the process is exactly X's and Y's
  /// isolated pop sequences interleaved by taking, at each step, the least of
  /// (X's next pop, the seam pair, Y's next pop) — X before the seam before Y on
  /// equal rank, by position. If no seam pair is ever popped, `q` is a boundary
  /// of `T(w)` and `T(w) = T(X) ++ T(Y)` (merges never split). Conversely a seam
  /// merge makes a token straddle `q`, so `q` is a boundary iff no seam pair is
  /// ever popped. Since `q` IS a boundary of `T(Z)` (never crossed there), X's
  /// isolated sequence is exactly the pops of the recorded `Z`-process whose
  /// token ends at or before `q`, in order — the ACTIVE pops.
  ///
  /// **Deciding the seam.** The seam pair changes only when a side's fringe
  /// token changes: X's fringe walks the right spine of its last token, Y's the
  /// left spine of its first. With `(x, y)` the current fringe pair and `ρ` its
  /// merge rank (none ⇒ it cannot pop), and both sides at some position in
  /// their sequences: the seam pops iff it becomes the least pending, i.e. iff X
  /// reaches a next pop of rank `> ρ` while `x` is still alive AND Y reaches a
  /// next pop of rank `>= ρ` (the seam is left of every Y pair, so it wins ties
  /// with Y and loses them to X) while `y` is still alive. Each side's clause
  /// depends on that side alone: X runs its pops of rank `<= ρ` first whatever Y
  /// does, so `x` survives iff some pop between X's current position and the pop
  /// that absorbs `x` (inclusive) has rank `> ρ`, or nothing ever absorbs `x`
  /// (X's sequence ends, its next rank being `+∞`). Symmetrically for Y with
  /// `>= ρ`. Both clauses true ⇒ the seam pops ⇒ not a boundary. Otherwise the
  /// side that fails the clause absorbs its fringe first, a new pair forms, and
  /// the question repeats for it.
  ///
  /// **Ordering the fringe events.** Two sequences interleaved by least-next-
  /// rank emit their elements in order of *prefix maximum* rank (an element pops
  /// when the running maximum of what has popped reaches it), the left side first
  /// on ties. So fringe events are visited in order of their side's prefix-max at
  /// the event's pop, X first on ties; a side's "current position" at a foreign
  /// event of prefix-max `t` is its last pop with prefix-max `<= t`. That is
  /// [`MaxTree::last_prefix_at_most`] over the active pops for X, and a scan of
  /// the short recorded Y process for Y.
  fn cut_is_boundary(&mut self, table: &MergeTable, q: u32, x_token: usize, y: &WordRun) -> bool {
    self.activate_upto(q);
    let n_pops = self.z.pops.len();
    let x_spine = &self.z.rspine[x_token];
    let y_spine = &y.lspine[0];
    // Prefix maxima of the isolated Y process, in rank+1 space.
    let mut y_prefix: Vec<u32> = Vec::with_capacity(y.pops.len());
    let mut run = 0u32;
    for p in &y.pops {
      run = run.max(p.rank + 1);
      y_prefix.push(run);
    }
    let y_max = |lo: usize, hi: usize| -> u32 {
      y.pops[lo..hi.min(y.pops.len())]
        .iter()
        .map(|p| p.rank + 1)
        .max()
        .unwrap_or(0)
    };
    // Fringe states.
    // A byte without a single-char token cannot occur here (`process` refused
    // it); `u32::MAX` is not a vocabulary id, so it merges with nothing.
    let x_state = |j: usize| -> u32 {
      if j == 0 {
        table.symbol(self.word[q as usize - 1]).unwrap_or(u32::MAX)
      } else {
        self.z.pops[x_spine[j - 1] as usize].new_id
      }
    };
    let y_state = |j: usize| -> u32 {
      if j == 0 {
        table.symbol(self.word[q as usize]).unwrap_or(u32::MAX)
      } else {
        y.pops[y_spine[j - 1] as usize].new_id
      }
    };
    // Event times (prefix-max at the forming pop), rank+1 space.
    let x_time = |j: usize| -> u32 {
      let idx = x_spine[j - 1] as usize;
      self.tree.max(0, idx + 1)
    };
    let y_time = |j: usize| -> u32 { y_prefix[y_spine[j - 1] as usize] };

    let (mut jx, mut jy) = (0usize, 0usize);
    // The side that produced the current event and the pop index it stood at,
    // or the initial state (no pops on either side).
    enum At {
      Init,
      X(usize),
      Y(usize),
    }
    let mut at = At::Init;
    loop {
      // Evaluate the current seam pair.
      if let Some((rank, _)) = table.merge(x_state(jx), y_state(jy)) {
        let rho = rank + 1;
        let (x_pos, y_pos): (Option<usize>, Option<usize>) = match at {
          At::Init => (None, None),
          At::X(i) => {
            // X wins ties: at an X event of time `t`, Y's pops of time `t` have
            // NOT happened yet, so Y stands at its last pop of time `< t`.
            let t = self.tree.max(0, i + 1);
            (Some(i), y_last_at_most(&y_prefix, t.saturating_sub(1)))
          }
          At::Y(i) => (self.tree.last_prefix_at_most(y_prefix[i], n_pops), Some(i)),
        };
        let x_from = x_pos.map_or(0, |p| p + 1);
        let x_alive = match x_spine.get(jx) {
          None => true,
          Some(&e) => self.tree.max(x_from, e as usize + 1) > rho,
        };
        let y_from = y_pos.map_or(0, |p| p + 1);
        let y_alive = match y_spine.get(jy) {
          None => true,
          Some(&e) => y_max(y_from, e as usize + 1) >= rho,
        };
        if x_alive && y_alive {
          return false;
        }
      }
      // Advance to the next fringe event in time order, X first on ties.
      let nx = (jx < x_spine.len()).then(|| x_time(jx + 1));
      let ny = (jy < y_spine.len()).then(|| y_time(jy + 1));
      match (nx, ny) {
        (None, None) => return true,
        (Some(_), None) => {
          jx += 1;
          at = At::X(x_spine[jx - 1] as usize);
        }
        (None, Some(_)) => {
          jy += 1;
          at = At::Y(y_spine[jy - 1] as usize);
        }
        (Some(tx), Some(ty)) => {
          if tx <= ty {
            jx += 1;
            at = At::X(x_spine[jx - 1] as usize);
          } else {
            jy += 1;
            at = At::Y(y_spine[jy - 1] as usize);
          }
        }
      }
    }
  }
}

/// Largest index `k` with `prefix[k] <= t`, or `None`.
fn y_last_at_most(prefix: &[u32], t: u32) -> Option<usize> {
  let n = prefix.partition_point(|&p| p <= t);
  n.checked_sub(1)
}

#[cfg(test)]
mod tests;
