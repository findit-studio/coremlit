use super::*;
use crate::audio::lid::Span;

const WINDOW: usize = 160_000;

/// A normalized log-probability row from unnormalized scores, so a test can
/// state "this window is 90 % sure of column 3" without hand-computing a
/// softmax.
fn row_from_logits(logits: &[(usize, f64)]) -> LogProbabilities {
  let mut values = vec![-30.0f64; NUM_LANGUAGES];
  for &(index, logit) in logits {
    values[index] = logit;
  }
  let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let log_sum_exp = max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
  LogProbabilities::new(
    values
      .into_iter()
      .map(|v| (v - log_sum_exp) as f32)
      .collect(),
  )
}

fn window(row: LogProbabilities, start: usize, len: usize) -> WindowLogProbabilities {
  WindowLogProbabilities::new(row, Span::new(start, len, WINDOW))
}

/// Total probability mass under `exp`, in f64 — every pooling must leave this
/// at 1.
fn mass(row: &LogProbabilities) -> f64 {
  row.as_slice().iter().map(|v| f64::from(*v).exp()).sum()
}

fn all_poolings() -> [ScorePooling; 4] {
  [
    ScorePooling::MeanLogProbability,
    ScorePooling::MeanProbability,
    ScorePooling::Max,
    ScorePooling::Vote,
  ]
}

// ── The default ─────────────────────────────────────────────────────────────

/// The shipped default is the logarithmic pool, and `Default` agrees with the
/// variant the docs and the module's measurement table name.
#[test]
fn the_default_pooling_is_the_logarithmic_pool() {
  assert_eq!(ScorePooling::default(), ScorePooling::MeanLogProbability);
}

// ── Identity ────────────────────────────────────────────────────────────────

/// A single window aggregates to ITSELF, bit for bit, under every pooling —
/// no divide, no renormalization, no f64 round trip. This is what makes
/// `identify_long` on a clip that fits one window identical to `identify`, and
/// it is why `finish` short-circuits on `count == 1`.
#[test]
fn one_window_is_the_bit_exact_identity() {
  let row = row_from_logits(&[(94, 6.0), (3, 1.5), (17, 0.25)]);
  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &[window(row.clone(), 0, WINDOW)]).expect("aggregate");
    assert_eq!(out.as_slice(), row.as_slice(), "{pooling:?}");
  }
  // Even a partial-coverage sole window: the weight cancels, it is not applied.
  let out = aggregate_windows(
    ScorePooling::MeanProbability,
    &[window(row.clone(), 0, 4_321)],
  )
  .expect("aggregate");
  assert_eq!(out.as_slice(), row.as_slice());
}

/// Two IDENTICAL windows aggregate back to that same row (to f32 rounding)
/// under every pooling — the sanity floor beneath the identity above, and the
/// one property all four policies genuinely share.
#[test]
fn identical_windows_reproduce_their_common_row() {
  let row = row_from_logits(&[(94, 6.0), (3, 1.5)]);
  let windows = vec![
    window(row.clone(), 0, WINDOW),
    window(row.clone(), WINDOW, WINDOW),
  ];
  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &windows).expect("aggregate");
    if pooling == ScorePooling::Vote {
      // A vote keeps only the winner, so it cannot reproduce the row — it
      // reproduces the CHOICE, at probability 1.
      assert_eq!(out.as_slice()[94], 0.0);
      continue;
    }
    for (index, (&got, &want)) in out.as_slice().iter().zip(row.as_slice()).enumerate() {
      assert!(
        (got - want).abs() < 1e-5,
        "{pooling:?} column {index}: {got} vs {want}"
      );
    }
  }
}

// ── Each pooling's arithmetic ───────────────────────────────────────────────

/// Log-space and probability-space means are DIFFERENT operations on this
/// domain — the claim the whole module rests on. Two windows, one sure of
/// column 0 and one split between 0 and 1: the linear pool reports the
/// mixture's own average, the logarithmic pool multiplies the evidence and
/// lands somewhere else entirely.
#[test]
fn the_log_pool_and_the_linear_pool_disagree() {
  // Window A: p(0) = 0.98, p(1) = 0.02 (rest negligible).
  // Window B: p(0) = 0.30, p(1) = 0.70.
  let a = LogProbabilities::try_from_slice(&distribution(&[(0, 0.98), (1, 0.02)])).expect("row");
  let b = LogProbabilities::try_from_slice(&distribution(&[(0, 0.30), (1, 0.70)])).expect("row");
  let windows = vec![window(a, 0, WINDOW), window(b, WINDOW, WINDOW)];

  let linear = aggregate_windows(ScorePooling::MeanProbability, &windows).expect("aggregate");
  let logarithmic =
    aggregate_windows(ScorePooling::MeanLogProbability, &windows).expect("aggregate");

  // Linear: the arithmetic mean of the two probabilities, exactly.
  assert!((f64::from(linear.as_slice()[0]).exp() - 0.64).abs() < 1e-4);
  assert!((f64::from(linear.as_slice()[1]).exp() - 0.36).abs() < 1e-4);

  // Logarithmic: the renormalized geometric mean, sqrt(.98·.30) : sqrt(.02·.70)
  // = 0.5422 : 0.1183, i.e. 0.8209 : 0.1791.
  assert!(
    (f64::from(logarithmic.as_slice()[0]).exp() - 0.8209).abs() < 1e-3,
    "{}",
    f64::from(logarithmic.as_slice()[0]).exp()
  );
  assert!((f64::from(logarithmic.as_slice()[1]).exp() - 0.1791).abs() < 1e-3);

  // And they are genuinely far apart — this is not a rounding difference.
  assert!(
    (f64::from(linear.as_slice()[0]) - f64::from(logarithmic.as_slice()[0])).abs() > 0.2,
    "the two means must not coincide"
  );
}

/// `Max` takes the per-language peak and renormalizes; the peak language keeps
/// its rank and the row is a distribution again.
#[test]
fn max_takes_the_per_language_peak() {
  let a = LogProbabilities::try_from_slice(&distribution(&[(0, 0.90), (1, 0.10)])).expect("row");
  let b = LogProbabilities::try_from_slice(&distribution(&[(0, 0.20), (2, 0.80)])).expect("row");
  let out = aggregate_windows(
    ScorePooling::Max,
    &[window(a, 0, WINDOW), window(b, WINDOW, WINDOW)],
  )
  .expect("aggregate");

  // Pre-normalization peaks are 0.90, 0.10, 0.80; the sum is 1.80, so the
  // renormalized row is 0.5, 0.0556, 0.4444.
  assert!((f64::from(out.as_slice()[0]).exp() - 0.5).abs() < 1e-3);
  assert!((f64::from(out.as_slice()[1]).exp() - 0.0556).abs() < 1e-3);
  assert!((f64::from(out.as_slice()[2]).exp() - 0.4444).abs() < 1e-3);
  assert!((mass(&out) - 1.0).abs() < 1e-5);
}

/// A vote counts window winners and nothing else: a 51 % window and a 99.9 %
/// window carry the same weight, and every language no window chose lands at
/// exactly `-∞` (probability 0).
#[test]
fn a_vote_counts_winners_and_discards_magnitude() {
  let landslide =
    LogProbabilities::try_from_slice(&distribution(&[(0, 0.999), (1, 0.001)])).expect("row");
  let squeaker =
    LogProbabilities::try_from_slice(&distribution(&[(0, 0.490), (1, 0.510)])).expect("row");
  let out = aggregate_windows(
    ScorePooling::Vote,
    &[
      window(landslide, 0, WINDOW),
      window(squeaker, WINDOW, WINDOW),
    ],
  )
  .expect("aggregate");

  assert_eq!(out.as_slice()[0], 0.5f32.ln());
  assert_eq!(out.as_slice()[1], 0.5f32.ln());
  assert_eq!(out.as_slice()[2], f32::NEG_INFINITY);
  assert!((mass(&out) - 1.0).abs() < 1e-6);

  // `-inf` survives the type's invariant and reads as probability zero.
  let ranked = out.top_k(NUM_LANGUAGES).expect("rank");
  assert_eq!(ranked[0].index(), 0, "ties break by ascending column");
  assert_eq!(ranked[1].index(), 1);
  assert_eq!(ranked[2].probability(), 0.0);
}

/// A window votes for the same language `top_k(1)` on that window would
/// return, including the ascending-column tie-break on an exact tie.
#[test]
fn a_windows_vote_agrees_with_its_own_top_1() {
  let tied = LogProbabilities::try_from_slice(&distribution(&[(40, 0.5), (7, 0.5)])).expect("row");
  assert_eq!(argmax(tied.as_slice()), 7);
  assert_eq!(tied.top_k(1).expect("rank")[0].index(), 7);

  let out = aggregate_windows(
    ScorePooling::Vote,
    &[
      window(tied.clone(), 0, WINDOW),
      window(tied, WINDOW, WINDOW),
    ],
  )
  .expect("aggregate");
  assert_eq!(out.as_slice()[7], 0.0);
  assert_eq!(out.as_slice()[40], f32::NEG_INFINITY);
}

// ── Duration weighting ──────────────────────────────────────────────────────

/// Weighting is by REAL audio, not by window count: a tail covering a tenth of
/// a window carries a tenth of the vote. Stated on the linear pool, where the
/// expected number is exact arithmetic rather than a pooling artefact.
#[test]
fn windows_weigh_by_the_audio_they_actually_saw() {
  let full = LogProbabilities::try_from_slice(&distribution(&[(0, 1.0)])).expect("row");
  let sliver = LogProbabilities::try_from_slice(&distribution(&[(1, 1.0)])).expect("row");

  // 160 000 samples of "column 0" against 16 000 samples of "column 1".
  let weighted = aggregate_windows(
    ScorePooling::MeanProbability,
    &[
      window(full.clone(), 0, WINDOW),
      window(sliver.clone(), WINDOW, WINDOW / 10),
    ],
  )
  .expect("aggregate");
  assert!((f64::from(weighted.as_slice()[0]).exp() - 10.0 / 11.0).abs() < 1e-4);
  assert!((f64::from(weighted.as_slice()[1]).exp() - 1.0 / 11.0).abs() < 1e-4);

  // Equal spans: exactly the unweighted mean, so the weighting is inert under
  // the tail policies that produce only full-length windows.
  let equal = aggregate_windows(
    ScorePooling::MeanProbability,
    &[window(full, 0, WINDOW), window(sliver, WINDOW, WINDOW)],
  )
  .expect("aggregate");
  assert!((f64::from(equal.as_slice()[0]).exp() - 0.5).abs() < 1e-4);
}

/// A vote is weighted too — "vote with your seconds" — so a short tail cannot
/// outvote the body of the clip.
#[test]
fn a_vote_is_weighted_by_duration() {
  let body = LogProbabilities::try_from_slice(&distribution(&[(0, 0.9), (1, 0.1)])).expect("row");
  let tail = LogProbabilities::try_from_slice(&distribution(&[(1, 0.9), (0, 0.1)])).expect("row");
  let out = aggregate_windows(
    ScorePooling::Vote,
    &[window(body, 0, WINDOW), window(tail, WINDOW, WINDOW / 4)],
  )
  .expect("aggregate");
  assert!((f64::from(out.as_slice()[0]).exp() - 0.8).abs() < 1e-6);
  assert!((f64::from(out.as_slice()[1]).exp() - 0.2).abs() < 1e-6);
}

/// `Max` is the one pooling a window's length does not influence — a maximum
/// has no weighted form, and the docs say so.
#[test]
fn max_ignores_duration() {
  let a = LogProbabilities::try_from_slice(&distribution(&[(0, 0.9), (1, 0.1)])).expect("row");
  let b = LogProbabilities::try_from_slice(&distribution(&[(1, 0.7), (0, 0.3)])).expect("row");
  let long = aggregate_windows(
    ScorePooling::Max,
    &[
      window(a.clone(), 0, WINDOW),
      window(b.clone(), WINDOW, WINDOW),
    ],
  )
  .expect("aggregate");
  let short = aggregate_windows(
    ScorePooling::Max,
    &[window(a, 0, WINDOW), window(b, WINDOW, 2_000)],
  )
  .expect("aggregate");
  assert_eq!(long.as_slice(), short.as_slice());
}

// ── Invariants across the board ─────────────────────────────────────────────

/// Whatever the pooling, the aggregate is still a natural-log distribution:
/// every value `<= 0`, none NaN, and `exp` summing to 1. That is what lets an
/// aggregated row be used anywhere a single-window row can.
#[test]
fn every_pooling_returns_a_normalized_log_distribution() {
  let windows = vec![
    window(row_from_logits(&[(94, 7.0), (3, 2.0)]), 0, WINDOW),
    window(
      row_from_logits(&[(3, 4.0), (94, 3.5), (61, 3.0)]),
      WINDOW,
      WINDOW,
    ),
    window(row_from_logits(&[(61, 9.0)]), 2 * WINDOW, WINDOW / 3),
  ];
  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &windows).expect("aggregate");
    assert_eq!(out.as_slice().len(), NUM_LANGUAGES, "{pooling:?}");
    assert!(
      out.as_slice().iter().all(|v| !v.is_nan() && *v <= 0.0),
      "{pooling:?} produced a NaN or a positive log-probability"
    );
    assert!(
      (mass(&out) - 1.0).abs() < 1e-4,
      "{pooling:?} mass {}",
      mass(&out)
    );
    // The invariant the TYPE enforces admits the same row back.
    assert!(
      LogProbabilities::try_from_slice(out.as_slice()).is_ok(),
      "{pooling:?}"
    );
  }
}

/// The batch entry point and the streaming accumulator `identify_long` drives
/// are the same fold, bit for bit — they must be, because only one of them is
/// exercised against the model.
#[test]
fn the_batch_and_streaming_folds_agree_bit_for_bit() {
  let windows = vec![
    window(row_from_logits(&[(94, 7.0), (3, 2.0)]), 0, WINDOW),
    window(row_from_logits(&[(3, 4.0), (94, 3.5)]), WINDOW, WINDOW),
    window(row_from_logits(&[(61, 9.0)]), 2 * WINDOW, 7_777),
  ];
  for pooling in all_poolings() {
    let batch = aggregate_windows(pooling, &windows).expect("batch");
    let mut acc = Accumulator::new(pooling);
    for w in &windows {
      acc.push(w.value(), w.span().len());
    }
    let streamed = acc.finish().expect("streamed");
    assert_eq!(batch.as_slice(), streamed.as_slice(), "{pooling:?}");
  }
}

/// Folding nothing is a typed refusal, not a panic or an all-zero row.
#[test]
fn an_empty_window_list_is_a_typed_refusal() {
  for pooling in all_poolings() {
    assert!(matches!(
      aggregate_windows(pooling, &[]),
      Err(Error::EmptyWindows)
    ));
  }
}

/// `renormalize` leaves an all-`-∞` row alone rather than producing NaN from
/// `-∞ − (−∞)`. Not reachable from a model row (a log-softmax always has a
/// finite argmax); pinned because the guard is invisible otherwise.
#[test]
fn renormalize_does_not_turn_an_impossible_row_into_nan() {
  let mut values = vec![f64::NEG_INFINITY; NUM_LANGUAGES];
  renormalize(&mut values);
  assert!(values.iter().all(|v| *v == f64::NEG_INFINITY));
}

/// A row of exact probabilities, as log-probabilities. Unlisted columns get the
/// remaining mass spread thin enough to be negligible but never zero, so a
/// pooling that multiplies cannot be handed a `-∞` it did not earn.
fn distribution(entries: &[(usize, f64)]) -> Vec<f32> {
  let named: f64 = entries.iter().map(|(_, p)| *p).sum();
  let rest = ((1.0 - named) / (NUM_LANGUAGES - entries.len()) as f64).max(f64::MIN_POSITIVE);
  let mut values = vec![rest.ln() as f32; NUM_LANGUAGES];
  for &(index, p) in entries {
    values[index] = p.ln() as f32;
  }
  values
}
