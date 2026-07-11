use super::*;
use crate::options::DecodingOptions;

fn greedy(temperature: f32) -> GreedyTokenSampler {
  GreedyTokenSampler::new(temperature, 3, &DecodingOptions::new()).with_seed(42)
}

#[test]
fn argmax_at_zero_temperature_with_exact_logprob() {
  let logits = [1.0f32, 3.0, 2.0, 0.0];
  let result = greedy(0.0).sample(&logits);
  assert_eq!(result.token(), 1);
  assert!(!result.completed());
  // log softmax by hand: 3.0 - ln(e^1 + e^3 + e^2 + e^0)
  let log_z = logits.iter().map(|v| v.exp()).sum::<f32>().ln();
  assert!((result.logprob() - (3.0 - log_z)).abs() < 1e-5);
}

#[test]
fn eot_completes() {
  let logits = [0.0f32, 0.0, 0.0, 5.0]; // index 3 == eot
  let result = greedy(0.0).sample(&logits);
  assert_eq!(result.token(), 3);
  assert!(result.completed());
}

#[test]
fn nonzero_temperature_is_seed_deterministic_and_top_k_bounded() {
  // top_k = 5 (DecodingOptions default); only indices 0..5 by probability
  // can ever be drawn.
  let mut logits = vec![0.0f32; 16];
  for (i, v) in [9.0, 8.0, 7.0, 6.0, 5.0].iter().enumerate() {
    logits[i + 8] = *v; // the top-5 live at 8..13
  }
  let mut a = greedy(0.7);
  let mut b = greedy(0.7);
  for _ in 0..20 {
    let (ra, rb) = (a.sample(&logits), b.sample(&logits));
    assert_eq!(ra.token(), rb.token(), "same seed, same draw");
    assert!(
      (8..13).contains(&(ra.token() as usize)),
      "outside top-k drawn"
    );
    assert!(ra.logprob() <= 0.0);
  }
}

#[test]
fn fully_masked_logits_degenerate_without_panic_or_nan() {
  // Regression (task-4 review, Important): every entry masked (-inf)
  // panicked on the empty multinomial range at t != 0 and produced a NaN
  // logprob at t == 0. Both paths must return the same defined result.
  let masked = [f32::NEG_INFINITY; 8];
  for temperature in [0.0, 0.7] {
    let result = greedy(temperature).sample(&masked);
    assert_eq!(result.token(), 0, "t={temperature}");
    assert_eq!(result.logprob(), f32::NEG_INFINITY, "t={temperature}");
    assert!(!result.completed(), "t={temperature}");
  }
  // eot at the degenerate index still reports completion.
  let eot_zero = GreedyTokenSampler::new(0.7, 0, &DecodingOptions::new())
    .with_seed(42)
    .sample(&masked);
  assert!(eot_zero.completed());
}

#[test]
fn fully_masked_sample_does_not_consume_rng() {
  // The degenerate path must not consult the RNG: a sampler that first
  // saw a fully masked buffer draws the same stream afterwards as a
  // fresh same-seeded sampler.
  let logits: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
  let mut interrupted = greedy(0.7);
  let mut fresh = greedy(0.7);
  interrupted.sample(&[f32::NEG_INFINITY; 16]);
  for _ in 0..10 {
    assert_eq!(
      interrupted.sample(&logits).token(),
      fresh.sample(&logits).token()
    );
  }
}

#[test]
#[should_panic(expected = "non-empty logits")]
fn empty_logits_panic() {
  greedy(0.0).sample(&[]);
}

#[test]
fn finalize_appends_eot_once() {
  let sampler = greedy(0.0);
  let (mut tokens, mut logprobs) = (vec![1u32, 2], vec![-0.5f32, -0.25]);
  sampler.finalize(&mut tokens, &mut logprobs);
  assert_eq!(tokens, vec![1, 2, 3]);
  assert_eq!(logprobs, vec![-0.5, -0.25, 0.0]);
  sampler.finalize(&mut tokens, &mut logprobs); // idempotent: already ends in EOT
  assert_eq!(tokens.len(), 3);
}
