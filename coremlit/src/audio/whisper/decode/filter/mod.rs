//! Whisper's per-step logits filter chain: each [`LogitsFilter`] runs
//! after a decode step produces raw logits and masks disallowed
//! vocabulary entries to `-inf` in place, before sampling picks the next
//! token. Ports `LogitsFiltering` and its four concrete filters
//! (argmax-oss-swift `Sources/WhisperKit/Core/Text/LogitsFilter.swift`).
//!
//! Swift's protocol method mutates and returns the same `MLMultiArray`
//! (`filterLogits(_:withTokens:) -> MLMultiArray`) to support chaining;
//! [`LogitsFilter::filter`] instead mutates `logits: &mut [f32]` in place
//! and returns nothing. The masking logic is plain `f32` — the f16→f32
//! conversion already happened at the backend boundary (see
//! [`crate::audio::whisper::backend`]). The one place precision is
//! load-bearing is the timestamp-mass comparison
//! (`bnns_mass_rule_scalars`), which deliberately replicates the f16
//! rounding *structure* of Swift's BNNS pipeline rather than comparing in
//! f32 — see that function's doc for the probed contract.

use half::f16;

use crate::audio::whisper::tokenizer::SpecialTokens;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// LogitsFilter
// ---------------------------------------------------------------------

/// A vocabulary id a filter must mask that is not a position in the step's
/// logits — the id and the width it overran.
///
/// # Why this is fallible rather than skipped
///
/// The ids reaching these filters are not all this crate's. A caller sets
/// [`DecodingOptions::suppress_tokens`](crate::audio::whisper::options::DecodingOptions::set_suppress_tokens)
/// to whatever it likes, and [`SuppressTokensFilter`] is public and takes the
/// list directly, so `logits[id]` is an unbounded write driven by untrusted
/// input. `tokens` is equally the caller's: [`TimestampRulesFilter`] derives
/// the end of a mask range from the largest timestamp id in it.
///
/// Masking past the end is not a maskable event — the entry the caller wants
/// suppressed does not exist — so silently dropping it (`get_mut(..).map(..)`
/// and carry on) would decode against a vocabulary the caller thinks it
/// constrained. The refusal names the id and the width so the mismatch is
/// diagnosable from the message alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmaskableToken {
  token: u32,
  vocab: usize,
}

impl UnmaskableToken {
  /// Construct from the id that could not be masked and the logits width it
  /// overran.
  #[inline(always)]
  pub const fn new(token: u32, vocab: usize) -> Self {
    Self { token, vocab }
  }

  /// The id the filter could not mask.
  #[inline(always)]
  pub const fn token(&self) -> u32 {
    self.token
  }

  /// The number of logits the step produced.
  #[inline(always)]
  pub const fn vocab(&self) -> usize {
    self.vocab
  }
}

/// `token` as a POSITION in a `vocab`-wide logits vector (`token < vocab`).
#[inline]
fn position(token: u32, vocab: usize) -> Result<usize, UnmaskableToken> {
  let index = token as usize;
  if index < vocab {
    Ok(index)
  } else {
    Err(UnmaskableToken::new(token, vocab))
  }
}

/// `token` as a slice BOUND into a `vocab`-wide logits vector
/// (`token <= vocab`) — the end of a `..token` mask range, or the start of a
/// `token..` one, either of which may legally sit one past the last position.
#[inline]
fn bound(token: u32, vocab: usize) -> Result<usize, UnmaskableToken> {
  let index = token as usize;
  if index <= vocab {
    Ok(index)
  } else {
    Err(UnmaskableToken::new(token, vocab))
  }
}

/// One step of the logits filter chain: masks disallowed vocabulary
/// entries of `logits` to [`f32::NEG_INFINITY`] in place, given the
/// tokens sampled so far (prompt included). Ports Swift's
/// `LogitsFiltering` protocol (`LogitsFilter.swift:8-10`).
pub trait LogitsFilter {
  /// Masks `logits` in place for the next sampling step, given `tokens`
  /// sampled so far.
  ///
  /// # Errors
  /// [`UnmaskableToken`] when an id this filter must mask is not a position in
  /// `logits`. Swift indexes an `MLMultiArray` with no such bound and traps;
  /// see that type for why the id is refused rather than skipped.
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) -> Result<(), UnmaskableToken>;
}

// ---------------------------------------------------------------------
// SuppressTokensFilter
// ---------------------------------------------------------------------

/// Unconditionally masks a fixed list of token ids on every call. Ports
/// Swift's `SuppressTokensFilter` (`LogitsFilter.swift:12-25`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuppressTokensFilter {
  suppress_tokens: Vec<u32>,
}

impl SuppressTokensFilter {
  /// Builds a filter that masks every id in `suppress_tokens`, every call.
  pub fn new(suppress_tokens: Vec<u32>) -> Self {
    Self { suppress_tokens }
  }
}

impl LogitsFilter for SuppressTokensFilter {
  fn filter(&self, logits: &mut [f32], _tokens: &[u32]) -> Result<(), UnmaskableToken> {
    let vocab = logits.len();
    for &token in &self.suppress_tokens {
      logits[position(token, vocab)?] = f32::NEG_INFINITY;
    }
    Ok(())
  }
}

// ---------------------------------------------------------------------
// SuppressBlankFilter
// ---------------------------------------------------------------------

/// Masks the whitespace and end-of-text tokens on the very first sampling
/// step only (`tokens.len() == sample_begin`), so the decoder cannot open
/// a segment with a blank. Ports Swift's `SuppressBlankFilter`
/// (`LogitsFilter.swift:27-51`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuppressBlankFilter {
  whitespace_token: u32,
  end_token: u32,
  sample_begin: usize,
}

impl SuppressBlankFilter {
  /// Builds a filter over `special`'s whitespace/end-of-text ids, firing
  /// only when the sampled sequence is exactly `sample_begin` tokens long.
  pub fn new(special: &SpecialTokens, sample_begin: usize) -> Self {
    Self {
      whitespace_token: special.whitespace_token(),
      end_token: special.end_token(),
      sample_begin,
    }
  }
}

impl LogitsFilter for SuppressBlankFilter {
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) -> Result<(), UnmaskableToken> {
    if tokens.len() != self.sample_begin {
      return Ok(());
    }
    let vocab = logits.len();
    logits[position(self.whitespace_token, vocab)?] = f32::NEG_INFINITY;
    logits[position(self.end_token, vocab)?] = f32::NEG_INFINITY;
    Ok(())
  }
}

// ---------------------------------------------------------------------
// TimestampRulesFilter
// ---------------------------------------------------------------------

/// Enforces Whisper's paired-timestamp decoding rules: timestamps must
/// appear in pairs (except directly before EOT), must not decrease, and
/// each segment must have nonzero length; also forces timestamp sampling
/// once the timestamp tokens' combined probability mass exceeds every
/// individual text token's. Ports Swift's `TimestampRulesFilter`
/// (`LogitsFilter.swift:54-243`), itself a port of OpenAI Whisper's
/// `ApplyTimestampRules`
/// (<https://github.com/openai/whisper/blob/master/whisper/decoding.py#L441>).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampRulesFilter {
  no_timestamps_token: u32,
  end_token: u32,
  time_token_begin: u32,
  transcribe_token: u32,
  translate_token: u32,
  sample_begin: usize,
  // Stored for constructor parity with Swift but never read: the
  // initial-timestamp clamp that would consume it is commented out
  // upstream (`LogitsFilter.swift:112-122`) — see the NOTE in
  // `filter` below for how to re-enable it.
  max_initial_timestamp_index: Option<usize>,
  is_multilingual: bool,
}

impl TimestampRulesFilter {
  /// Builds a filter enforcing the paired-timestamp rules for a decode run
  /// whose content tokens start at `sample_begin`. `max_initial_timestamp_index`
  /// is stored for constructor parity with Swift but not applied (see
  /// [`LogitsFilter::filter`]'s impl below). `is_multilingual` selects
  /// between the fixed `sample_begin` and Swift's task-token scan
  /// (`LogitsFilter.swift:131-142`).
  pub fn new(
    special: &SpecialTokens,
    sample_begin: usize,
    max_initial_timestamp_index: Option<usize>,
    is_multilingual: bool,
  ) -> Self {
    Self {
      no_timestamps_token: special.no_timestamps_token(),
      end_token: special.end_token(),
      time_token_begin: special.time_token_begin(),
      transcribe_token: special.transcribe_token(),
      translate_token: special.translate_token(),
      sample_begin,
      max_initial_timestamp_index,
      is_multilingual,
    }
  }

  /// Resolves the effective `sample_begin` for `tokens`: the fixed value
  /// for a non-multilingual model, or `None` while a multilingual prompt
  /// is still being prefilled (no `<|transcribe|>`/`<|translate|>` task
  /// token in its first 3 tokens yet). Ports `sampleBegin(for:)`
  /// (`LogitsFilter.swift:131-142`).
  fn effective_sample_begin(&self, tokens: &[u32]) -> Option<usize> {
    if !self.is_multilingual {
      return Some(self.sample_begin);
    }
    tokens
      .iter()
      .take(3)
      .position(|&t| t == self.transcribe_token || t == self.translate_token)
      .map(|task_index| (task_index + 1).max(self.sample_begin))
  }
}

impl LogitsFilter for TimestampRulesFilter {
  /// Ports `filterLogits(_:withTokens:)` (`LogitsFilter.swift:72-129`).
  ///
  /// # The four bounds, resolved once at the top
  ///
  /// This filter is the one that masks RANGES, so it needs more than "the id
  /// is a position": `logits[..end_token]` and `logits[time_begin..]` both
  /// need their id as a slice BOUND, which may legally sit one past the last
  /// position. Both are resolved before any masking so a refusal leaves the
  /// vector as the step produced it, and so the range fills below are indexing
  /// numbers this function has already established rather than raw ids.
  ///
  /// The fourth bound is the only one that is not the tokenizer's:
  /// `timestamp_last` is derived from the largest timestamp id in `tokens`,
  /// which is the CALLER's slice.
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) -> Result<(), UnmaskableToken> {
    let Some(sample_begin) = self.effective_sample_begin(tokens) else {
      return Ok(()); // still prefilling a multilingual prompt without a task token
    };
    if sample_begin > tokens.len() {
      return Ok(());
    }

    // In the order the masking below would have reached them, so the id
    // reported is the one that would have been the first bad subscript.
    let vocab = logits.len();
    let no_timestamps = position(self.no_timestamps_token, vocab)?;
    let time_begin = bound(self.time_token_begin, vocab)?;
    let end_token = bound(self.end_token, vocab)?;

    // suppress <|notimestamps|>, which is handled by `withoutTimestamps`.
    logits[no_timestamps] = f32::NEG_INFINITY;

    if tokens.len() > sample_begin {
      // Timestamps have to appear in pairs, except directly before EOT;
      // mask logits accordingly.
      let sampled = &tokens[sample_begin..];
      let last_was_timestamp = sampled.last().is_some_and(|&t| t >= self.time_token_begin);
      let penultimate_was_timestamp =
        sampled.len() < 2 || sampled[sampled.len() - 2] >= self.time_token_begin;
      if last_was_timestamp {
        if penultimate_was_timestamp {
          // has to be non-timestamp
          logits[time_begin..vocab].fill(f32::NEG_INFINITY);
        } else {
          // cannot be normal text tokens
          logits[..end_token].fill(f32::NEG_INFINITY);
        }
      }

      if let Some(last_timestamp) = sampled
        .iter()
        .copied()
        .rfind(|&t| t >= self.time_token_begin)
      {
        // Timestamps shouldn't decrease: forbid timestamp tokens smaller
        // than the last. Also force each segment to have a nonzero
        // length, to prevent infinite looping, unless the sequence so far
        // is a single opening timestamp directly after text
        // (LogitsFilter.swift:100-108).
        let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
          last_timestamp
        } else {
          // `tokens` is the caller's, so this may be one past `u32::MAX` as
          // well as past the vocabulary; both are the same refusal.
          last_timestamp
            .checked_add(1)
            .ok_or(UnmaskableToken::new(last_timestamp, vocab))?
        };
        let timestamp_last = bound(timestamp_last, vocab)?;
        logits[time_begin..timestamp_last].fill(f32::NEG_INFINITY);
      }
    }

    // NOTE: the initial-timestamp rule is intentionally not applied here —
    // it is commented out upstream (LogitsFilter.swift:112-122), so the
    // real model is never forced into `<|0.00|>` at the first sampled
    // token. Re-enabling it is a one-liner: when `tokens.len() ==
    // sample_begin`, mask `logits[..time_begin]`, then, if
    // `self.max_initial_timestamp_index` is `Some(index)`, additionally
    // mask `logits[time_begin + index + 1..]`.

    // If the sum of probability over timestamps is above any other token,
    // sample a timestamp.
    if timestamp_mass_exceeds_text(logits, time_begin) {
      logits[..time_begin].fill(f32::NEG_INFINITY);
    }
    Ok(())
  }
}

/// The two `Float16` comparands of Swift's timestamp-mass rule
/// (`LogitsFilter.swift:144-242`), replicated at BNNS's probed precision
/// structure (tests/whisper_swift_probes/probe_massrule2.out, macOS 26.5/M1
/// Max): BNNS computes internally in f32 and rounds to f16 only at each
/// operation's output — a stable (max-subtracted) `logSoftmax` over the full
/// vector rounded per-element to f16, a NAIVE (no max subtraction; probed:
/// `LSE([-110 x1101]) = -inf`, not the stable `-103`) f32 sum-of-exp over the
/// f16 timestamp logprobs rounded to f16, and an exact max over the f16 text
/// logprobs. `logits[time_begin..]` is the timestamp region,
/// `logits[..time_begin]` the text region.
///
/// Returns `None` when the whole vector is masked (Swift's pipeline then
/// yields NaN/-inf comparands and never fires; callers treat `None` as "don't
/// fire"). NaN logits are outside the parity contract either way: this port
/// poisons the comparison to non-firing on any NaN, anywhere in the vector.
/// BNNS is not uniformly this conservative — its `.max`/`logSoftmax` skip NaN
/// lanes, so a NaN confined to the TEXT region leaves BNNS's comparands
/// finite and its rule CAN still fire (only a NaN in the TIMESTAMP region
/// poisons BNNS's naive, non-skipping `.logSumExp` the same way, so the two
/// agree there); a model emitting NaN logits is already undefined upstream.
///
/// Numeric domain (all probe-verified): `sum >= 1` always (the max element
/// contributes `exp(0)`), so `l` is finite in `[0, ln(vocab)]`; inputs are
/// exact f16 values `<= 65504` so no overflow; a fully-masked timestamp region
/// gives `ts_sum = 0 -> ts = -inf -> false`, exactly BNNS's probed
/// `LSE(all -inf) = -inf`. The per-element `v - m - l` double-subtraction is
/// the probed formula (probe emuA; `v - (m + l)` was indistinguishable).
fn bnns_mass_rule_scalars(logits: &[f32], time_begin: usize) -> Option<(f16, f16)> {
  let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  if !m.is_finite() {
    return None;
  }
  let mut sum = 0f32;
  for &v in logits {
    sum += (v - m).exp();
  }
  let l = sum.ln();
  // Timestamp scalar: BNNS `.logSumExp` (naive) over f16-rounded logprobs.
  let mut ts_sum = 0f32;
  for &v in &logits[time_begin..] {
    ts_sum += f16::from_f32(v - m - l).to_f32().exp();
  }
  let ts = f16::from_f32(ts_sum.ln());
  // Text scalar: BNNS `.max` over f16-rounded logprobs. (Probed BNNS quirk:
  // an all-(-inf) input returns -65504, not -inf — boolean-immaterial, the
  // timestamp scalar is then finite and exceeds both; plain max here.)
  let mut mx = f16::NEG_INFINITY;
  for &v in &logits[..time_begin] {
    let lp = f16::from_f32(v - m - l);
    if lp > mx {
      mx = lp;
    }
  }
  Some((ts, mx))
}

/// Whether the timestamp region's combined probability mass exceeds every
/// individual text token's, deciding whether text is masked so a timestamp is
/// forced (`LogitsFilter.swift:124-127` + `144-242`). Compares the two
/// f16-rounded scalars of [`bnns_mass_rule_scalars`]; a fully-masked vector
/// (`None`) does not fire. This resolves the margin on the same f16 grid as
/// Swift instead of in f32 — the #41 rank-1 token-divergence channel. Bounded
/// parity contract: boolean agreement with Swift's BNNS pipeline except where
/// BNNS's unspecified f32 vector-kernel rounding differs from sequential libm
/// by enough to cross the f16 rounding boundary of a comparand; empirically
/// `<= 0.3%` of deliberately margin-straddling adversarial inputs, 0 observed
/// on ordinary inputs and at both probed flip-point scans.
fn timestamp_mass_exceeds_text(logits: &[f32], time_begin: usize) -> bool {
  match bnns_mass_rule_scalars(logits, time_begin) {
    Some((ts, mx)) => ts > mx,
    None => false,
  }
}

// ---------------------------------------------------------------------
// LanguageLogitsFilter
// ---------------------------------------------------------------------

/// Masks every vocabulary index that is not a language token, once the
/// sampled sequence reaches `sample_begin` tokens — keeps
/// language-detection sampling confined to the language tokens. Ports
/// Swift's `LanguageLogitsFilter` (`LogitsFilter.swift:245-276`).
///
/// Swift precomputes `nonLanguageTokenIndexes: [[Int]]`, one 3-element
/// index array per non-language vocabulary entry
/// (`getNonLanguageTokenIndexes`, `LogitsFilter.swift:267-275`) —
/// effectively a ~51k-entry allocation for a full Whisper vocabulary. This
/// instead keeps a sorted `language_tokens` and masks by `binary_search`
/// at filter time: identical result, no large precomputed table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageLogitsFilter {
  language_tokens: Vec<u32>,
  sample_begin: usize,
}

impl LanguageLogitsFilter {
  /// Builds a filter over `language_tokens`, active once the sampled
  /// sequence reaches `sample_begin` tokens.
  pub fn new(language_tokens: &[u32], sample_begin: usize) -> Self {
    let mut language_tokens = language_tokens.to_vec();
    language_tokens.sort_unstable();
    Self {
      language_tokens,
      sample_begin,
    }
  }
}

impl LogitsFilter for LanguageLogitsFilter {
  /// Infallible in practice, and structurally so: this is the one filter that
  /// walks the logits rather than indexing them, so no id of its own reaches a
  /// subscript and there is nothing for [`UnmaskableToken`] to report. A
  /// language id past the vocabulary simply matches no position.
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) -> Result<(), UnmaskableToken> {
    if tokens.len() < self.sample_begin {
      return Ok(());
    }
    for (index, value) in logits.iter_mut().enumerate() {
      if self.language_tokens.binary_search(&(index as u32)).is_err() {
        *value = f32::NEG_INFINITY;
      }
    }
    Ok(())
  }
}
