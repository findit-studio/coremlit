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
fn finalize_appends_eot_once() {
  let sampler = greedy(0.0);
  let (mut tokens, mut logprobs) = (vec![1u32, 2], vec![-0.5f32, -0.25]);
  sampler.finalize(&mut tokens, &mut logprobs);
  assert_eq!(tokens, vec![1, 2, 3]);
  assert_eq!(logprobs, vec![-0.5, -0.25, 0.0]);
  sampler.finalize(&mut tokens, &mut logprobs); // idempotent: already ends in EOT
  assert_eq!(tokens.len(), 3);
}
