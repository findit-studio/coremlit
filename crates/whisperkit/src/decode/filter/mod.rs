//! Whisper's per-step logits filter chain: each [`LogitsFilter`] runs
//! after a decode step produces raw logits and masks disallowed
//! vocabulary entries to `-inf` in place, before sampling picks the next
//! token. Ports `LogitsFiltering` and its four concrete filters
//! (argmax-oss-swift `Sources/WhisperKit/Core/Text/LogitsFilter.swift`).
//!
//! Swift's protocol method mutates and returns the same `MLMultiArray`
//! (`filterLogits(_:withTokens:) -> MLMultiArray`) to support chaining;
//! [`LogitsFilter::filter`] instead mutates `logits: &mut [f32]` in place
//! and returns nothing. The buffer is plain `f32` math throughout, not
//! BNNS `FloatType` — the f16→f32 conversion already happened at the
//! backend boundary (spec §4.8; see [`crate::backend`]).

use crate::tokenizer::SpecialTokens;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// LogitsFilter
// ---------------------------------------------------------------------

/// One step of the logits filter chain: masks disallowed vocabulary
/// entries of `logits` to [`f32::NEG_INFINITY`] in place, given the
/// tokens sampled so far (prompt included). Ports Swift's
/// `LogitsFiltering` protocol (`LogitsFilter.swift:8-10`).
pub trait LogitsFilter {
  /// Masks `logits` in place for the next sampling step, given `tokens`
  /// sampled so far.
  fn filter(&self, logits: &mut [f32], tokens: &[u32]);
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
  fn filter(&self, logits: &mut [f32], _tokens: &[u32]) {
    for &token in &self.suppress_tokens {
      logits[token as usize] = f32::NEG_INFINITY;
    }
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
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) {
    if tokens.len() != self.sample_begin {
      return;
    }
    logits[self.whitespace_token as usize] = f32::NEG_INFINITY;
    logits[self.end_token as usize] = f32::NEG_INFINITY;
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
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) {
    let Some(sample_begin) = self.effective_sample_begin(tokens) else {
      return; // still prefilling a multilingual prompt without a task token
    };
    if sample_begin > tokens.len() {
      return;
    }

    // suppress <|notimestamps|>, which is handled by `withoutTimestamps`.
    let time_begin = self.time_token_begin as usize;
    logits[self.no_timestamps_token as usize] = f32::NEG_INFINITY;

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
          let len = logits.len();
          logits[time_begin..len].fill(f32::NEG_INFINITY);
        } else {
          // cannot be normal text tokens
          logits[..self.end_token as usize].fill(f32::NEG_INFINITY);
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
          last_timestamp + 1
        };
        logits[time_begin..timestamp_last as usize].fill(f32::NEG_INFINITY);
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
  }
}

/// Numerically stable equivalent of the BNNS `logSoftmax` +
/// `logSumExp`/`max` reduction pair Swift runs to decide whether the
/// timestamp region's combined probability mass exceeds every individual
/// text token's (`LogitsFilter.swift:144-242`), computed here as plain
/// `f32` math instead of BNNS (spec §4.8). `logits[time_begin..]` is the
/// timestamp region, `logits[..time_begin]` the text region.
fn timestamp_mass_exceeds_text(logits: &[f32], time_begin: usize) -> bool {
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  if !max.is_finite() {
    // Every entry is masked (-inf): there is no distribution to compare.
    return false;
  }
  let log_z = max + logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();

  let timestamps = &logits[time_begin..];
  let ts_max = timestamps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let timestamp_logprob = if ts_max.is_finite() {
    ts_max - log_z
      + timestamps
        .iter()
        .map(|&v| (v - ts_max).exp())
        .sum::<f32>()
        .ln()
  } else {
    f32::NEG_INFINITY
  };

  let max_text_logprob = logits[..time_begin]
    .iter()
    .copied()
    .fold(f32::NEG_INFINITY, f32::max)
    - log_z;

  timestamp_logprob > max_text_logprob
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
  fn filter(&self, logits: &mut [f32], tokens: &[u32]) {
    if tokens.len() < self.sample_begin {
      return;
    }
    for (index, value) in logits.iter_mut().enumerate() {
      if self.language_tokens.binary_search(&(index as u32)).is_err() {
        *value = f32::NEG_INFINITY;
      }
    }
  }
}
