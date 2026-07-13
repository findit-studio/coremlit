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

// ---------------------------------------------------------------------
// derive_attempt_seed
// ---------------------------------------------------------------------

#[test]
fn derive_attempt_seed_is_pure_and_deterministic() {
  // Same triple, same result -- every time, no hidden state.
  assert_eq!(
    derive_attempt_seed(1, 2, 3),
    derive_attempt_seed(1, 2, 3),
    "pure function: identical inputs must reproduce identical output"
  );
  // Each coordinate independently changes the result.
  assert_ne!(
    derive_attempt_seed(1, 2, 3),
    derive_attempt_seed(1, 2, 4),
    "attempt_index must change the derived seed"
  );
  assert_ne!(
    derive_attempt_seed(1, 2, 3),
    derive_attempt_seed(1, 3, 3),
    "window_index must change the derived seed"
  );
  assert_ne!(
    derive_attempt_seed(1, 2, 3),
    derive_attempt_seed(2, 2, 3),
    "the base seed must change the derived seed"
  );
}

#[test]
fn derive_attempt_seed_has_no_collisions_over_realistic_ranges() {
  // Statistical decorrelation check over a generous window/attempt range
  // (temperature_fallback_count defaults to 5, so 0..=8 already exceeds
  // any default configuration; 0..64 windows covers long-form audio's
  // window count many times over). A broken derivation that ignored (or
  // truncated) either coordinate would collide immediately here.
  let mut seen = std::collections::HashSet::new();
  for window in 0..64u64 {
    for attempt in 0..=8u64 {
      let derived = derive_attempt_seed(0xABCD_1234_5678_9ABC, window, attempt);
      assert!(
        seen.insert(derived),
        "collision at window={window} attempt={attempt}"
      );
    }
  }
}

/// A seeded, non-degenerate (multi-candidate) sampler at `temperature =
/// 0.7` -- top_k defaults to 5, so this is a genuine multinomial draw
/// among several candidates, not a coin flip between two or a foregone
/// argmax.
fn seeded_sampler(seed: u64) -> GreedyTokenSampler {
  GreedyTokenSampler::new(0.7, 999, &DecodingOptions::new()).with_seed(seed)
}

fn draw_sequence(sampler: &mut GreedyTokenSampler, logits: &[f32], n: usize) -> Vec<u32> {
  (0..n).map(|_| sampler.sample(logits).token()).collect()
}

#[test]
fn attempt_seed_derivation_changes_sampled_draws_across_attempts() {
  // Proves the (window, attempt) sub-seed derivation is actually wired
  // into distinct SAMPLING streams, not just distinct numbers in the
  // abstract: two samplers seeded from adjacent attempt indices at the
  // same window draw different sequences from the exact same logits.
  //
  // Mutation check performed by hand (not left in the tree): temporarily
  // making `derive_attempt_seed` ignore `attempt_index` (returning the
  // same sub-seed for every attempt at a fixed window) made this
  // `assert_ne!` fail, confirming the test is sensitive to exactly the
  // bug class it exists to catch.
  let seed = 0xC0FFEE_u64;
  let window = 3u64;
  let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.6).collect();

  let mut attempt0 = seeded_sampler(derive_attempt_seed(seed, window, 0));
  let mut attempt1 = seeded_sampler(derive_attempt_seed(seed, window, 1));
  let draws0 = draw_sequence(&mut attempt0, &logits, 20);
  let draws1 = draw_sequence(&mut attempt1, &logits, 20);
  assert_ne!(
    draws0, draws1,
    "different attempt_index must decorrelate the sampled stream"
  );

  // Reproducibility half: the identical (window, attempt) pair always
  // replays the identical stream (this is what makes a whole
  // transcription reproducible from one base seed).
  let mut replay0 = seeded_sampler(derive_attempt_seed(seed, window, 0));
  let replay_draws0 = draw_sequence(&mut replay0, &logits, 20);
  assert_eq!(draws0, replay_draws0);
}

#[test]
fn attempt_seed_derivation_changes_sampled_draws_across_windows() {
  // Same proof as above, along the window_index coordinate instead of
  // attempt_index: two different windows at the same attempt must not
  // share a draw stream either.
  let seed = 0xC0FFEE_u64;
  let attempt = 1u64;
  let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.6).collect();

  let mut window0 = seeded_sampler(derive_attempt_seed(seed, 0, attempt));
  let mut window1 = seeded_sampler(derive_attempt_seed(seed, 1, attempt));
  let draws0 = draw_sequence(&mut window0, &logits, 20);
  let draws1 = draw_sequence(&mut window1, &logits, 20);
  assert_ne!(
    draws0, draws1,
    "different window_index must decorrelate the sampled stream"
  );
}
