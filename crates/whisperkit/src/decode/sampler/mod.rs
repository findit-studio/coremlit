//! Greedy next-token sampling: the decode loop's per-step choice of which
//! token to emit next, given that step's raw logits — argmax at
//! `temperature == 0`, seeded top-k multinomial otherwise — plus the
//! end-of-decoding finalization step. Ports Swift's `GreedyTokenSampler`
//! (argmax-oss-swift `Sources/WhisperKit/Core/Text/TokenSampler.swift:29-252`).
//!
//! Swift picks between two sampling implementations at runtime by OS
//! availability — `sampleWithMLTensor` (`TokenSampler.swift:40-84`, macOS
//! 15+/iOS 18+) and `sampleWithBNNS` (`TokenSampler.swift:86-213`, the
//! fallback, itself flagged with a `TODO` for replacement) — that
//! collapse into the one plain `f32` path here: the f16→f32 conversion
//! already happened at the backend boundary (spec §4.8; see
//! [`crate::backend`]), matching [`crate::decode::filter`]'s convention.
//!
//! **Deliberate mechanical deviation, documented per spec:** Swift's
//! `SamplingResult` clones the whole `tokens`/`logProbs` arrays on every
//! step (`TokenSampler.swift:231-239`); [`SamplingResult`] here carries
//! only the new step's `(token, logprob, completed)`, leaving the decode
//! loop (a later task) to append it — the observable token/logprob
//! sequence is identical, with zero per-step clones.
//!
//! `BeamSearchTokenSampler` (`TokenSampler.swift:254-290`) is not ported:
//! every method it defines is an unconditional `fatalError`, including
//! its own initializer on an invalid beam size/patience, so there is no
//! real behavior to port — a spec non-goal.

use std::num::NonZeroUsize;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::options::DecodingOptions;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// SamplingResult
// ---------------------------------------------------------------------

/// One sampling step's outcome: the sampled token id, its log probability
/// under the (possibly temperature-scaled) distribution, and whether that
/// token completes decoding. Ports Swift's `SamplingResult`
/// (`TokenSampler.swift:13-27`) — see the module docs for why this only
/// carries the new step instead of Swift's whole-history clone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingResult {
  token: u32,
  logprob: f32,
  completed: bool,
}

impl SamplingResult {
  /// The sampled token id.
  #[inline(always)]
  pub const fn token(&self) -> u32 {
    self.token
  }

  /// Log probability of [`Self::token`] under the sampling distribution.
  #[inline(always)]
  pub const fn logprob(&self) -> f32 {
    self.logprob
  }

  /// Whether [`Self::token`] is the end-of-text token, i.e. decoding is
  /// complete.
  #[inline(always)]
  pub const fn completed(&self) -> bool {
    self.completed
  }
}

// ---------------------------------------------------------------------
// GreedyTokenSampler
// ---------------------------------------------------------------------

/// Greedy (argmax at `temperature() == 0.0`) / seeded top-k multinomial
/// (otherwise) next-token sampler. Ports Swift's `GreedyTokenSampler`
/// (`TokenSampler.swift:29-252`); see the module docs for the
/// `sampleWithMLTensor`/`sampleWithBNNS` collapse and the
/// [`SamplingResult`] deviation.
#[derive(Debug)]
pub struct GreedyTokenSampler {
  temperature: f32,
  eot_token: u32,
  top_k: NonZeroUsize,
  rng: StdRng,
  // Reused across `sample` calls at `temperature != 0` to avoid a
  // per-step allocation; cleared and repopulated at the top of that path.
  probs: Vec<f32>,
}

impl GreedyTokenSampler {
  /// Builds a sampler for `temperature`/`eot_token`, capturing `options`'
  /// [`DecodingOptions::top_k`] (`top_k` is a plain `usize` knob with no
  /// non-zero invariant of its own, so `0` clamps up to
  /// `NonZeroUsize::MIN`). Seeds its RNG from the OS
  /// (`StdRng::from_os_rng`) — `temperature == 0.0` never consults the
  /// RNG, so this only matters for reproducibility at `temperature !=
  /// 0.0`; see [`Self::with_seed`] for the deterministic alternative.
  pub fn new(temperature: f32, eot_token: u32, options: &DecodingOptions) -> Self {
    Self {
      temperature,
      eot_token,
      top_k: NonZeroUsize::new(options.top_k()).unwrap_or(NonZeroUsize::MIN),
      rng: StdRng::from_os_rng(),
      probs: Vec::new(),
    }
  }

  /// Reseeds this sampler's RNG deterministically from `seed`
  /// (`StdRng::seed_from_u64`). Swift's sampler draws `Float.random(in:
  /// 0..<sum)` unseeded (`TokenSampler.swift:169`); [`Self::new`]'s
  /// OS-seeded default matches that non-determinism, and this builder
  /// adds a reproducibility knob Swift has no equivalent for, so
  /// `temperature != 0.0` callers (e.g. tests) can assert an exact draw.
  #[must_use]
  pub fn with_seed(mut self, seed: u64) -> Self {
    self.rng = StdRng::seed_from_u64(seed);
    self
  }

  /// The configured sampling temperature; `0.0` selects argmax decoding.
  #[inline(always)]
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }

  /// The end-of-text token id that marks a sampled sequence complete.
  #[inline(always)]
  pub const fn eot_token(&self) -> u32 {
    self.eot_token
  }

  /// Samples the next token from a decode step's `logits`. At
  /// `temperature() == 0.0`: argmax, with `logprob` the exact log-softmax
  /// value at the chosen index (`TokenSampler.swift:75,182-197,202-204`).
  /// Otherwise: `logits` scaled by `1 / temperature`, passed through a
  /// numerically stable softmax, restricted to the
  /// [`DecodingOptions::top_k`] highest-probability indices, and one
  /// drawn by a multinomial scan over `0..sum(top-k probabilities)`
  /// (`TokenSampler.swift:140-180`) — `logprob` is the log of the drawn
  /// token's probability under the *full* (not top-k-renormalized)
  /// distribution, matching Swift's `topKProbs...log()` /
  /// `log(softmaxResult[nextToken])`. Either way, `completed` is `token
  /// == eot_token()` (`TokenSampler.swift:233`).
  ///
  /// # Panics
  ///
  /// Panics if `logits` is empty.
  pub fn sample(&mut self, logits: &[f32]) -> SamplingResult {
    let (token, logprob) = if self.temperature == 0.0 {
      let index = argmax(logits);
      (index as u32, logits[index] - super::log_sum_exp(logits))
    } else {
      self.probs.clear();
      let inv_t = 1.0 / self.temperature;
      // logits scaled by inv_t, max-shifted by the scaled max — avoids a
      // second pass over `logits` to find the max of the scaled values.
      let scaled_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
      let mut sum = 0.0f32;
      self.probs.extend(logits.iter().map(|&v| {
        let e = (v * inv_t - scaled_max).exp();
        sum += e;
        e
      }));

      // Indices of the top-k entries by (unnormalized) probability: a
      // partition instead of a full sort, since only top-k membership and
      // its sum matter, not a fully sorted order. Clamped to the actual
      // candidate count so a `top_k` misconfigured above `logits.len()`
      // can't panic.
      let k = self.top_k.get().min(self.probs.len());
      let mut indices: Vec<usize> = (0..self.probs.len()).collect();
      indices.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        self.probs[b].total_cmp(&self.probs[a])
      });
      indices.truncate(k);

      let top_sum: f32 = indices.iter().map(|&i| self.probs[i]).sum();
      let rnd = self.rng.random_range(0.0..top_sum);
      let mut accumulator = 0.0f32;
      let mut chosen = indices[0];
      for &i in &indices {
        accumulator += self.probs[i];
        if rnd < accumulator {
          chosen = i;
          break;
        }
      }
      (chosen as u32, (self.probs[chosen] / sum).ln())
    };
    SamplingResult {
      token,
      logprob,
      completed: token == self.eot_token,
    }
  }

  /// Appends the end-of-text token (with `logprob` `0.0`) to `tokens`/
  /// `logprobs` if the sequence does not already end with it; a no-op
  /// once it does. Ports Swift's `finalize(tokens:logProbs:)`
  /// (`TokenSampler.swift:242-251`) — Swift's return value (a fresh
  /// `SamplingResult` cloning the whole appended arrays) is dropped here
  /// for the same reason as the module docs' `SamplingResult` deviation:
  /// the caller already owns `tokens`/`logprobs` and mutates them in
  /// place.
  pub fn finalize(&self, tokens: &mut Vec<u32>, logprobs: &mut Vec<f32>) {
    if tokens.last() != Some(&self.eot_token) {
      tokens.push(self.eot_token);
      logprobs.push(0.0);
    }
  }
}

/// Index of the largest entry in `logits`, comparing with `f32::total_cmp`
/// and keeping the first index on an exact tie (matching vDSP/numpy
/// argmax convention). Ports the argmax branch of Swift's
/// `sampleWithMLTensor`/`sampleWithBNNS` (`TokenSampler.swift:75,182-196`).
fn argmax(logits: &[f32]) -> usize {
  let mut best = 0;
  for (index, &value) in logits.iter().enumerate().skip(1) {
    if value.total_cmp(&logits[best]).is_gt() {
      best = index;
    }
  }
  best
}
