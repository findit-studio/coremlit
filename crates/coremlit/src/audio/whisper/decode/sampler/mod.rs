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
//! [`crate::audio::whisper::backend`]), matching [`crate::audio::whisper::decode::filter`]'s convention.
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
//!
//! **Rust-only addition, no Swift equivalent:** [`derive_attempt_seed`]
//! turns one caller-chosen [`DecodingOptions::seed`] into a distinct
//! [`GreedyTokenSampler::with_seed`] seed per (window, attempt), so
//! [`crate::audio::whisper::transcribe::TranscribeTask`]'s temperature-fallback ladder can
//! be both reproducible end to end and free of correlated draws across
//! windows/attempts — see that function's own doc for the exact mixing
//! function and contract (coremlit issue #9).

use std::num::NonZeroUsize;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::audio::whisper::options::DecodingOptions;

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
  /// Latches `true` the first time [`Self::sample`] consults `rng` — a real
  /// multinomial draw, which happens only at a non-zero temperature on a
  /// non-masked buffer. [`crate::audio::whisper::transcribe::TranscribeTask`]'s fallback
  /// ladder reads this to record whether an RNG draw *actually occurred*,
  /// rather than inferring it from the accepted attempt's temperature — which
  /// misses a rejected attempt's draw and overcounts a zero-iteration decode
  /// (F2, codex round 3). See [`Self::drew_from_rng`].
  drew_from_rng: bool,
  // Reused across `sample` calls at `temperature != 0` to avoid
  // per-step allocations; cleared and repopulated at the top of that
  // path. `indices` is the top-k candidate list — full-vocab sized, so
  // reallocating it every sampled token would churn ~400 KiB/step.
  probs: Vec<f32>,
  indices: Vec<usize>,
}

impl GreedyTokenSampler {
  /// Builds a sampler for `temperature`/`eot_token`, capturing `options`'
  /// [`DecodingOptions::top_k`] (`top_k` is a plain `usize` knob with no
  /// non-zero invariant of its own, so `0` clamps up to
  /// `NonZeroUsize::MIN`). Seeds its RNG from the OS
  /// (`StdRng::from_os_rng`) — `temperature == 0.0` never consults the
  /// RNG, so this only matters for reproducibility at `temperature !=
  /// 0.0`; see [`Self::with_seed`] for the deterministic alternative,
  /// which [`crate::audio::whisper::transcribe::TranscribeTask`]'s fallback ladder now
  /// calls automatically (via [`derive_attempt_seed`]) whenever
  /// [`DecodingOptions::seed`] is set — this constructor's OS-seeded
  /// behavior is exactly what running with `seed` left `None` still gets.
  pub fn new(temperature: f32, eot_token: u32, options: &DecodingOptions) -> Self {
    Self {
      temperature,
      eot_token,
      top_k: NonZeroUsize::new(options.top_k()).unwrap_or(NonZeroUsize::MIN),
      rng: StdRng::from_os_rng(),
      drew_from_rng: false,
      probs: Vec::new(),
      indices: Vec::new(),
    }
  }

  /// Reseeds this sampler's RNG deterministically from `seed`
  /// (`StdRng::seed_from_u64`). Swift's sampler draws `Float.random(in:
  /// 0..<sum)` unseeded (`TokenSampler.swift:169`); [`Self::new`]'s
  /// OS-seeded default matches that non-determinism, and this builder
  /// adds a reproducibility knob Swift has no equivalent for, so
  /// `temperature != 0.0` callers (e.g. tests, and — via
  /// [`derive_attempt_seed`] — [`crate::audio::whisper::transcribe::TranscribeTask`]'s
  /// own fallback ladder when [`DecodingOptions::seed`] is set) can
  /// assert or reproduce an exact draw.
  #[must_use]
  #[inline(always)]
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

  /// Whether [`Self::sample`] has consulted the RNG at least once on this
  /// sampler — a real multinomial draw (non-zero temperature on a non-masked
  /// buffer; argmax and the all-masked degenerate path never draw).
  /// [`crate::audio::whisper::transcribe::TranscribeTask`]'s fallback ladder ORs this across a
  /// window's language probe and every attempt (rejected and retained) to
  /// record whether the transcript depended on an RNG draw at all — the
  /// reproducibility fact, recorded rather than inferred from a temperature
  /// (F2, codex round 3).
  #[inline(always)]
  pub const fn drew_from_rng(&self) -> bool {
    self.drew_from_rng
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
  /// The scaling treats a filter MASK (`-inf`, what [`crate::audio::whisper::decode::filter`]
  /// writes for a suppressed token) as a mask rather than a number: it stays
  /// `-inf` (probability 0) whatever the sign of `temperature`, so a negative
  /// temperature cannot flip `-inf * (1 / temperature)` to `+inf` and turn a
  /// suppressed token into the most-probable one (F1, codex round 4). When a
  /// near-zero temperature would overflow a finite logit's scaled value
  /// (`1 / temperature` or `v / temperature` past the f32 range), the scaled
  /// form collapses distinct logits to one endpoint, so scaling instead falls
  /// back to a stable-difference softmax `(logit − stab) / temperature` that
  /// keeps the ordering — the argmax (at a positive temperature; the argmin at
  /// a negative one) taking probability 1 and logprob 0, every other distinct
  /// logit −∞ — while ordinary temperatures keep the scaled softmax bit-for-bit
  /// (F1, codex round 14). Either way `# Panics` below stays true.
  ///
  /// A `top_k` configured above `logits.len()` is clamped to it (Swift's
  /// BNNS path has no defined behavior for that misconfiguration — `try!
  /// BNNS.applyTopK` with an oversized `k` is crash territory), and a
  /// `top_k` of `0` was already clamped up to `1` at construction.
  ///
  /// When every entry of `logits` is masked (`-inf`) there is no
  /// distribution to sample; both temperature paths then return the same
  /// defined degenerate result — the argmax convention's first index with
  /// `logprob` `-inf` — without consulting the RNG, so the decode loop's
  /// logprob threshold triggers fallback naturally. (Swift has no defined
  /// behavior here either: BNNS softmax over all-`-inf` yields NaNs.)
  ///
  /// # Panics
  ///
  /// Panics if `logits` is empty.
  pub fn sample(&mut self, logits: &[f32]) -> SamplingResult {
    assert!(!logits.is_empty(), "sample() requires non-empty logits");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
      // Fully masked buffer (see the doc paragraph above): intercept
      // before the paths' natural failure modes — `-inf - -inf = NaN`
      // logprob at `temperature == 0`, and a panic on the empty
      // `0.0..0.0` multinomial range otherwise. Mirrors the all-masked
      // guard in `decode::log_sum_exp`.
      let token = argmax(logits) as u32;
      return SamplingResult {
        token,
        logprob: f32::NEG_INFINITY,
        completed: token == self.eot_token,
      };
    }
    let (token, logprob) = if self.temperature == 0.0 {
      let index = argmax(logits);
      (index as u32, logits[index] - super::log_sum_exp(logits))
    } else {
      self.probs.clear();
      let inv_t = 1.0 / self.temperature;
      // Scale each logit by `1 / temperature`, exactly as Swift does before it
      // softmaxes the scaled vector (`TokenSampler.swift:109-138`) — but a
      // filter MASK is a mask, not a number to scale. `decode::filter` writes
      // `-inf` for a suppressed token (Swift's own filters do too,
      // `LogitsFilter.swift:81`), and that entry must stay EXCLUDED
      // (probability 0) whatever the sign of `inv_t`. Scaling it directly is
      // the F1 bug (codex round 4): a NEGATIVE temperature turns `-inf * inv_t`
      // into `+inf`, so the masked index becomes the single largest scaled
      // value.
      //
      // A near-zero temperature is a second F1 failure the per-entry clamp
      // hides (codex round 14): `1/T` (or a finite `v/T`) leaves the f32 range,
      // so the clamp maps EVERY same-sign finite logit to the same endpoint and
      // `scale(v) - scaled_max` is `0` for all of them — a UNIFORM draw over
      // distinct logits, which flips the token and reports `ln(1/n)` where the
      // limit is `0`. That saturation is exactly the set of FINITE logits whose
      // `v * inv_t` overflows; when any does, take a stable-difference softmax
      // that subtracts the stabilizer LOGIT *before* dividing by `temperature`
      // (`(v - stab) / temperature`), which stays order-preserving and free of
      // the `finite * inf` / `inf - inf` the scaled form produces. ORDINARY
      // temperatures never saturate and keep the exact arithmetic below,
      // bit-for-bit — the Swift-parity path the sampler goldens pin.
      let saturates = logits
        .iter()
        .any(|&v| v.is_finite() && !(v * inv_t).is_finite());
      let mut sum = 0.0f32;
      if saturates {
        // The stabilizer is the logit whose scaled value `v / temperature` is
        // largest — the MAX finite logit at a positive temperature, the MIN at
        // a negative one (dividing by a negative reverses the order). Masks
        // (`-inf`) and any NaN are never the stabilizer and stay excluded. Then
        // `(v - stab) / temperature` is `0` for the stabilizer (mass `1`,
        // logprob `0`) and `-inf` for every other distinct finite logit (mass
        // `0`, logprob `-inf`) — the true softmax limit as `temperature → 0`,
        // computed with no overflow (`0 / T` and `finite / T` never form a NaN).
        let stab = if self.temperature > 0.0 {
          logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max)
        } else {
          logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min)
        };
        self.probs.extend(logits.iter().map(|&v| {
          let e = if v.is_finite() {
            ((v - stab) / self.temperature).exp()
          } else {
            0.0 // mask (or a NaN backend logit): excluded from the distribution.
          };
          sum += e;
          e
        }));
      } else {
        // ORDINARY temperature: `v * inv_t` stays in range for every finite
        // logit, so this is the pre-fix scaled softmax unchanged — masks scale
        // to `-inf`, and `scale(v) - scaled_max` is numerically stable because
        // `scaled_max` is the max of the finite scaled values. Bit-identical to
        // the code the sampler parity goldens pin (F1, codex round 14).
        let scale = |v: f32| -> f32 {
          if v == f32::NEG_INFINITY {
            return f32::NEG_INFINITY; // mask: excluded at any temperature sign.
          }
          let scaled = v * inv_t;
          if scaled.is_nan() {
            // `0 * ±inf`, the only NaN a finite logit produces here (a subnormal
            // temperature drove `inv_t` non-finite — unreachable now that such a
            // temperature routes to the stable path above, but kept so this
            // branch remains a total, self-contained scaling of any input).
            0.0
          } else {
            scaled.clamp(f32::MIN, f32::MAX)
          }
        };
        let scaled_max = logits
          .iter()
          .copied()
          .map(scale)
          .fold(f32::NEG_INFINITY, f32::max);
        self.probs.extend(logits.iter().map(|&v| {
          let e = (scale(v) - scaled_max).exp();
          sum += e;
          e
        }));
      }

      // Indices of the top-k entries by (unnormalized) probability: a
      // partition instead of a full sort, since only top-k membership and
      // its sum matter, not a fully sorted order. Clamped to the actual
      // candidate count so a `top_k` misconfigured above `logits.len()`
      // can't panic.
      let k = self.top_k.get().min(self.probs.len());
      let probs = &self.probs;
      let indices = &mut self.indices;
      indices.clear();
      indices.extend(0..probs.len());
      indices.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| probs[b].total_cmp(&probs[a]));
      indices.truncate(k);

      let top_sum: f32 = indices.iter().map(|&i| probs[i]).sum();
      // A real RNG draw — the fact the fallback ladder records for
      // reproducibility (F2). Only reached at non-zero temperature on a
      // non-masked buffer; the argmax and all-masked paths never get here.
      self.drew_from_rng = true;
      let rnd = self.rng.random_range(0.0..top_sum);
      let mut accumulator = 0.0f32;
      let mut chosen = indices[0];
      for &i in indices.iter() {
        accumulator += probs[i];
        if rnd < accumulator {
          chosen = i;
          break;
        }
      }
      (chosen as u32, (probs[chosen] / sum).ln())
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

// ---------------------------------------------------------------------
// Seed derivation
// ---------------------------------------------------------------------

/// SplitMix64's avalanche finalizer (Steele, Lea & Flood, *Fast Splittable
/// Pseudorandom Number Generators*, OOPSLA 2014): two multiply-xorshift
/// rounds that turn a single-bit input difference into an unrelated
/// (on average half-flipped) 64-bit output. The same finalizer backs
/// `java.util.SplittableRandom`'s stream splitting; [`derive_attempt_seed`]
/// below chains one round of it per coordinate (base seed, worker, window,
/// attempt) to fold all four into a single sub-seed.
const fn splitmix64(x: u64) -> u64 {
  let mut z = x;
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^ (z >> 31)
}

/// Distinct, nonzero *odd* multipliers — one per coordinate — that spread a
/// coordinate across the full 64-bit width before it is XORed into the
/// running `splitmix64` state in [`derive_attempt_seed`]. Oddness makes
/// `coord.wrapping_mul(GAMMA)` a bijection on `u64` (odd values are units
/// modulo 2^64), so distinct coordinate values never collapse to the same
/// contribution; using a *different* constant per coordinate — applied at a
/// *different* mixing round — is what stops two coordinates from aliasing
/// when their raw values are swapped. `GAMMA_SEED` is SplitMix64's
/// golden-ratio increment; the other three are the MurmurHash3 / SplitMix64
/// finalizer multipliers — all well-known full-period mixing constants.
const GAMMA_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const GAMMA_WORKER: u64 = 0xD1B5_4A32_D192_ED03;
const GAMMA_WINDOW: u64 = 0xFF51_AFD7_ED55_8CCD;
const GAMMA_ATTEMPT: u64 = 0xC4CE_B9FE_1A85_EC53;

/// Deterministically derives a per-(worker, window, attempt) sampler
/// sub-seed from a caller-chosen base `seed` ([`DecodingOptions::seed`]).
///
/// Reusing `seed` verbatim for every [`GreedyTokenSampler`] the
/// temperature-fallback ladder builds would make every window/attempt draw
/// the exact same RNG stream — wrong the moment two of them share a logits
/// shape, since they would then sample identical sequences. Instead the
/// three coordinates are *domain-separated*: each is folded into the running
/// state in its own `splitmix64` round (this module's private SplitMix64
/// finalizer, above), after an initial round that mixes the base seed with a
/// nonzero constant. With `s` the running state:
///
/// ```text
/// s = splitmix64(seed           ^ GAMMA_SEED)
/// s = splitmix64(s ^ worker_index .wrapping_mul(GAMMA_WORKER))
/// s = splitmix64(s ^ window_index .wrapping_mul(GAMMA_WINDOW))
/// s = splitmix64(s ^ attempt_index.wrapping_mul(GAMMA_ATTEMPT))
/// ```
///
/// The coordinates are **never summed or otherwise collapsed into one
/// number**: `worker_index` and `window_index` enter at *different* rounds
/// through *different* multipliers, so `(worker = 0, window = 1)` and
/// `(worker = 1, window = 0)` derive unrelated sub-seeds — the earlier
/// `offset + window_index` sum aliased exactly that pair. And because the
/// first round mixes `seed` with a nonzero constant, `seed = 0` does not
/// collapse to `0`, so cross-seed pairs the earlier XOR mixer folded onto a
/// single value (`(seed = 0, window = 0)` and `(seed = 1, window = 1)` both
/// became `0` under `splitmix64(seed ^ window)`, since `splitmix64(0) == 0`)
/// now differ too.
///
/// Two guaranteed properties:
///
/// * **Reproducible.** This is a pure function of its four arguments, so an
///   identical `(seed, worker_index, window_index, attempt_index)` tuple
///   always reproduces the identical result — which is what lets
///   [`DecodingOptions::seed`] alone reproduce a whole transcription's
///   sampled tokens bit-for-bit across separate runs.
/// * **No single-coordinate collapse.** Changing *exactly one* coordinate
///   (holding the others fixed) is *guaranteed* to change the output, never
///   merely usually: each `splitmix64` round is a bijection, and a
///   coordinate reaches its round through an odd-constant multiply then an
///   XOR — both bijections — so the map from that coordinate to the
///   post-round state is injective and every later round preserves the
///   difference. No coordinate (the base seed included) has a value that
///   aliases another, and none collapses at zero.
///
/// A `u64` result cannot be globally injective over the full four-`u64`
/// input space (pigeonhole), so "collision-free" here means the
/// per-single-coordinate injectivity above plus being empirically
/// collision-free across the realistic worker/window/attempt ranges a
/// transcription reaches — see this module's `derive_attempt_seed` collision
/// test.
///
/// [`crate::audio::whisper::transcribe::TranscribeTask`]'s fallback ladder calls this with
/// `worker_index` the task's own
/// [`window_id_offset`](crate::audio::whisper::transcribe::TranscribeTask::set_window_id_offset)
/// (a genuinely unique per-chunk / per-audio / per-worker id, so distinct
/// chunks and concurrently-running workers now get distinct seed streams as
/// a real guarantee rather than the pre-fix *nudge*), `window_index` a
/// strictly monotonic per-`run` window counter *local* to that task (reset
/// to 0 for every task — precisely why it must be a coordinate separate from
/// `worker_index`), and `attempt_index` the fallback loop's own
/// `0..=temperature_fallback_count` counter. The function itself has no
/// dependency on that caller: it is a pure, public mixing primitive, so a
/// direct [`decode_text`](crate::audio::whisper::decode::decode_text)/
/// [`detect_language`](crate::audio::whisper::decode::detect_language) caller that wants to
/// replicate (or deliberately diverge from) the pipeline's exact seed
/// schedule can call it the same way.
#[must_use]
pub const fn derive_attempt_seed(
  seed: u64,
  worker_index: u64,
  window_index: u64,
  attempt_index: u64,
) -> u64 {
  let mut state = splitmix64(seed ^ GAMMA_SEED);
  state = splitmix64(state ^ worker_index.wrapping_mul(GAMMA_WORKER));
  state = splitmix64(state ^ window_index.wrapping_mul(GAMMA_WINDOW));
  splitmix64(state ^ attempt_index.wrapping_mul(GAMMA_ATTEMPT))
}

/// Index of the largest entry in `logits` under IEEE `>` comparison: the
/// FIRST index wins every tie (exact ties and signed-zero ties — IEEE `==`
/// treats `-0.0 == +0.0`), and NaN entries are skipped wherever they sit.
/// Probe-verified Swift parity (tests/whisper_swift_probes/probe_argmax2.out,
/// macOS 26.5/M1 Max): `MLTensor.argmax(alongAxis:)` on the f32-cast logits
/// (the macOS 15+ path, `TokenSampler.swift:45,75`) and
/// `BNNS.ReductionFunction.argMax` at f16 (legacy, `:182-196`) both return
/// the first index on every crafted tie and both skip NaN. All-NaN input is
/// unspecified upstream (MLTensor -> 0, BNNS -> last); this port pins 0,
/// matching the shipping MLTensor path. Empty input returns 0 (unreachable:
/// the decoder always emits `vocab()` logits).
fn argmax(logits: &[f32]) -> usize {
  let mut best: Option<usize> = None;
  for (index, &value) in logits.iter().enumerate() {
    match best {
      None if !value.is_nan() => best = Some(index),
      Some(b) if value > logits[b] => best = Some(index),
      _ => {}
    }
  }
  best.unwrap_or(0)
}
