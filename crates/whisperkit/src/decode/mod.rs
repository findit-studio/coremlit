//! Autoregressive decoding.
//!
//! Currently home to [`filter`], the logits-filter chain the decode loop
//! runs after every step's raw logits are produced, and [`sampler`], the
//! greedy/seeded-top-k next-token sampler it calls afterward; the loop
//! itself (Swift's `TextDecoder.decodeText`) lands in a later task.

pub mod filter;
pub mod sampler;

/// Numerically stable `max + ln(Σ exp(v - max))` (the log-sum-exp
/// normalizer of `logits`): subtracts the running max before
/// exponentiating so large logits don't overflow `f32::exp`, the same
/// shape as Swift's BNNS/MLTensor `logSoftmax` normalizer at `f32`
/// precision (spec §4.8). Shared by [`filter`]'s timestamp-mass
/// comparison and [`sampler`]'s zero-temperature log-softmax. Returns
/// [`f32::NEG_INFINITY`] when `logits` is empty or every entry is already
/// [`f32::NEG_INFINITY`], rather than the `NaN` that `(-inf) - (-inf)`
/// would otherwise produce.
fn log_sum_exp(logits: &[f32]) -> f32 {
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  if !max.is_finite() {
    return f32::NEG_INFINITY;
  }
  max + logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}
