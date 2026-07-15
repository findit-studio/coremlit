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
fn drew_from_rng_tracks_real_rng_draws() {
  // F2 (codex round 3). The fallback ladder records reproducibility from THIS
  // fact, not from the temperature: it must be true iff `sample` actually
  // consulted the RNG -- a non-zero-temperature draw on a non-masked buffer.
  let logits = [1.0f32, 3.0, 2.0, 0.0];

  // Argmax (temperature 0) never draws.
  let mut argmax_sampler = greedy(0.0);
  assert!(!argmax_sampler.drew_from_rng(), "a fresh sampler has not drawn");
  argmax_sampler.sample(&logits);
  assert!(
    !argmax_sampler.drew_from_rng(),
    "argmax decoding does not consult the RNG"
  );

  // A non-zero temperature draws, and the flag latches.
  let mut sampling = greedy(0.7);
  sampling.sample(&logits);
  assert!(
    sampling.drew_from_rng(),
    "a non-zero-temperature sample draws from the RNG"
  );

  // The all-masked degenerate path returns without drawing.
  let mut masked = greedy(0.7);
  masked.sample(&[f32::NEG_INFINITY; 4]);
  assert!(
    !masked.drew_from_rng(),
    "the all-masked degenerate path must not consult the RNG"
  );
}

#[test]
fn negative_temperature_wide_logits_sample_without_panic() {
  // F1 (codex round 3, High). Swift scales the logits by `1 / temperature`
  // FIRST, then softmaxes the *scaled* vector (`TokenSampler.swift:109-138`).
  // The port stabilized the softmax with `max(raw) * inv_t`, but for a
  // NEGATIVE temperature `inv_t < 0` reverses order, so that constant is the
  // *minimum* scaled value, not the max: the true-largest scaled entry then
  // computes `exp(scaled - min) = exp(huge) = +inf`, making `top_sum`
  // non-finite and panicking `random_range(0.0..inf)`.
  //
  // Reproduction (pre-fix): `sample([-10, 10])` at `-0.2` panicked. Post-fix
  // it must return a finite draw with no panic.
  let result = greedy(-0.2).sample(&[-10.0f32, 10.0]);
  assert!(result.logprob().is_finite(), "logprob must be finite");
  assert!((result.token() as usize) < 2, "token must index the logits");

  // Every drawn token must lie in the top-k of a numerically stable softmax
  // over the SCALED logits (`v / temperature`) -- exactly what Swift's
  // scale-then-softmax computes. Under a negative temperature the *smallest*
  // raw logit is the most probable, so a correct fix inverts the ordering
  // rather than overflowing.
  let wide = [-10.0f32, 10.0, -8.0, 5.0, -3.0, 2.0, 9.0, -1.0];
  let inv_t = 1.0f32 / -0.2;
  let scaled: Vec<f32> = wide.iter().map(|&v| v * inv_t).collect();
  let scaled_max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let reference: Vec<f32> = scaled.iter().map(|&s| (s - scaled_max).exp()).collect();
  assert!(
    reference.iter().all(|p| p.is_finite()),
    "the reference stable softmax over scaled logits must stay finite"
  );
  // top_k defaults to 5: the five highest-probability scaled entries.
  let mut order: Vec<usize> = (0..wide.len()).collect();
  order.sort_by(|&a, &b| reference[b].total_cmp(&reference[a]));
  let top_k: std::collections::HashSet<usize> = order.into_iter().take(5).collect();

  let mut sampler = greedy(-0.2);
  for _ in 0..50 {
    let r = sampler.sample(&wide);
    assert!(r.logprob().is_finite(), "finite logprob under negative temperature");
    assert!(
      top_k.contains(&(r.token() as usize)),
      "drawn token {} fell outside the stable-softmax top-k {top_k:?}",
      r.token()
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
  // Same tuple, same result -- every time, no hidden state.
  assert_eq!(
    derive_attempt_seed(1, 2, 3, 4),
    derive_attempt_seed(1, 2, 3, 4),
    "pure function: identical inputs must reproduce identical output"
  );
  // Each of the four coordinates independently changes the result: the
  // mixer folds every one into its own bijective `splitmix64` round, so
  // changing exactly one is *guaranteed* to change the output (see
  // `derive_attempt_seed`'s doc), not merely likely to.
  assert_ne!(
    derive_attempt_seed(1, 2, 3, 4),
    derive_attempt_seed(2, 2, 3, 4),
    "the base seed must change the derived seed"
  );
  assert_ne!(
    derive_attempt_seed(1, 2, 3, 4),
    derive_attempt_seed(1, 9, 3, 4),
    "worker_index must change the derived seed"
  );
  assert_ne!(
    derive_attempt_seed(1, 2, 3, 4),
    derive_attempt_seed(1, 2, 9, 4),
    "window_index must change the derived seed"
  );
  assert_ne!(
    derive_attempt_seed(1, 2, 3, 4),
    derive_attempt_seed(1, 2, 3, 9),
    "attempt_index must change the derived seed"
  );
}

#[test]
fn derive_attempt_seed_domain_separates_worker_and_window() {
  // Regression for the caller-side `offset + window_index` SUM alias
  // (coremlit#13): `transcribe_all` feeds each audio's global index as the
  // worker id while every task resets its window counter to 0, so
  // audio-0/window-1 (0 + 1) and audio-1/window-0 (1 + 0) summed to the
  // same coordinate `1` and shared one StdRng stream -- identical draws on
  // any shared logits shape (silent/repeated windows, or the MockBackend
  // that ignores encoder output). Passed as SEPARATE coordinates they must
  // derive different sub-seeds.
  for seed in [0u64, 1, 7, u64::MAX] {
    for attempt in 0..=5u64 {
      assert_ne!(
        derive_attempt_seed(seed, 0, 1, attempt),
        derive_attempt_seed(seed, 1, 0, attempt),
        "(worker=0, window=1) must not alias (worker=1, window=0) \
         [seed={seed} attempt={attempt}]"
      );
    }
  }
}

#[test]
fn derive_attempt_seed_has_no_zero_collapse_across_base_seeds() {
  // Regression for the mixer's XOR/zero alias (coremlit#13): the old
  // `splitmix64(seed ^ window)` mixer folded distinct coordinates together
  // because `splitmix64(0) == 0`. `(seed=0, window=0)` and
  // `(seed=1, window=1)` both derived 0, and `(seed=0, window=1)` aliased
  // `(seed=1, window=0)`. None of these may alias now, and the all-zero
  // tuple must not derive 0.
  assert_ne!(
    derive_attempt_seed(0, 0, 0, 0),
    0,
    "the all-zero tuple must not collapse to 0"
  );
  assert_ne!(
    derive_attempt_seed(0, 0, 0, 0),
    derive_attempt_seed(1, 0, 1, 0),
    "(seed=0, window=0) must not alias (seed=1, window=1)"
  );
  assert_ne!(
    derive_attempt_seed(0, 0, 1, 0),
    derive_attempt_seed(1, 0, 0, 0),
    "(seed=0, window=1) must not alias (seed=1, window=0)"
  );
}

#[test]
fn derive_attempt_seed_has_no_collisions_over_realistic_ranges() {
  // Statistical decorrelation check over a generous worker/window/attempt
  // range (temperature_fallback_count defaults to 5, so 0..=8 already
  // exceeds any default configuration; 0..64 windows covers long-form
  // audio's window count many times over; 0..64 workers covers a large
  // concurrent batch or VAD-chunk count). A broken derivation that ignored,
  // truncated, or summed any coordinate would collide immediately here.
  let mut seen = std::collections::HashSet::new();
  for worker in 0..64u64 {
    for window in 0..64u64 {
      for attempt in 0..=8u64 {
        let derived = derive_attempt_seed(0xABCD_1234_5678_9ABC, worker, window, attempt);
        assert!(
          seen.insert(derived),
          "collision at worker={worker} window={window} attempt={attempt}"
        );
      }
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
  // Proves the sub-seed derivation is actually wired into distinct SAMPLING
  // streams, not just distinct numbers in the abstract: two samplers seeded
  // from adjacent attempt indices at the same (worker, window) draw
  // different sequences from the exact same logits.
  //
  // Mutation check performed by hand (not left in the tree): temporarily
  // making `derive_attempt_seed` ignore `attempt_index` (returning the
  // same sub-seed for every attempt at a fixed window) made this
  // `assert_ne!` fail, confirming the test is sensitive to exactly the
  // bug class it exists to catch.
  let seed = 0xC0FFEE_u64;
  let worker = 2u64;
  let window = 3u64;
  let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.6).collect();

  let mut attempt0 = seeded_sampler(derive_attempt_seed(seed, worker, window, 0));
  let mut attempt1 = seeded_sampler(derive_attempt_seed(seed, worker, window, 1));
  let draws0 = draw_sequence(&mut attempt0, &logits, 20);
  let draws1 = draw_sequence(&mut attempt1, &logits, 20);
  assert_ne!(
    draws0, draws1,
    "different attempt_index must decorrelate the sampled stream"
  );

  // Reproducibility half: the identical tuple always replays the identical
  // stream (this is what makes a whole transcription reproducible from one
  // base seed).
  let mut replay0 = seeded_sampler(derive_attempt_seed(seed, worker, window, 0));
  let replay_draws0 = draw_sequence(&mut replay0, &logits, 20);
  assert_eq!(draws0, replay_draws0);
}

#[test]
fn attempt_seed_derivation_changes_sampled_draws_across_windows() {
  // Same proof as above, along the window_index coordinate: two different
  // windows at the same (worker, attempt) must not share a draw stream.
  let seed = 0xC0FFEE_u64;
  let worker = 2u64;
  let attempt = 1u64;
  let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.6).collect();

  let mut window0 = seeded_sampler(derive_attempt_seed(seed, worker, 0, attempt));
  let mut window1 = seeded_sampler(derive_attempt_seed(seed, worker, 1, attempt));
  let draws0 = draw_sequence(&mut window0, &logits, 20);
  let draws1 = draw_sequence(&mut window1, &logits, 20);
  assert_ne!(
    draws0, draws1,
    "different window_index must decorrelate the sampled stream"
  );
}

#[test]
fn attempt_seed_derivation_changes_sampled_draws_across_workers() {
  // The Class-A alias (coremlit#13) at the SAMPLED-STREAM level: the two
  // (worker, window) pairs the old `offset + window_index` sum collapsed --
  // (worker=0, window=1) and (worker=1, window=0) -- must now draw
  // different sequences from identical logits, exactly as two real
  // transcription windows sharing a logits shape would need.
  let seed = 0xC0FFEE_u64;
  let attempt = 0u64;
  let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.6).collect();

  let mut worker0_window1 = seeded_sampler(derive_attempt_seed(seed, 0, 1, attempt));
  let mut worker1_window0 = seeded_sampler(derive_attempt_seed(seed, 1, 0, attempt));
  let draws_a = draw_sequence(&mut worker0_window1, &logits, 20);
  let draws_b = draw_sequence(&mut worker1_window0, &logits, 20);
  assert_ne!(
    draws_a, draws_b,
    "(worker=0, window=1) and (worker=1, window=0) must not share a stream"
  );
}
