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
  // The shift comes off before the log does, for the reason `renormalize`
  // spells out: a helper that quietly stopped normalizing on a large logit
  // would surface as the fold's postcondition failing, which reads as a
  // production defect rather than a broken fixture.
  let log_sum = values.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
  LogProbabilities::new(
    values
      .into_iter()
      .map(|v| ((v - max) - log_sum) as f32)
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
      acc.push(w.value(), w.span().len()).expect("push");
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

// ── Totality over the accepted domain ───────────────────────────────────────

/// Disjoint zero-probability supports pool, in log space, to a row whose
/// exponentials sum to ZERO.
///
/// The arithmetic is right: a logarithmic pool is a geometric mean, so a
/// language ANY window scores at `-∞` is zero in the pool, and two windows
/// certain of different languages zero out every language between them. What
/// is wrong is returning that as a distribution — its "top" languages are
/// whichever ones the tie-break happens to surface, each at probability zero.
#[test]
fn a_zero_mass_logarithmic_pool_is_refused_rather_than_returned() {
  let certain_of = |index: usize| {
    let mut values = vec![f32::NEG_INFINITY; NUM_LANGUAGES];
    values[index] = 0.0;
    LogProbabilities::try_from_slice(&values).expect("one zero among -inf normalizes exactly")
  };
  let windows = vec![
    window(certain_of(100), 0, WINDOW),
    window(certain_of(101), WINDOW, WINDOW),
  ];

  let pooled = aggregate_windows(ScorePooling::MeanLogProbability, &windows);
  assert!(
    matches!(
      pooled,
      Err(Error::ZeroMassAggregate(ScorePooling::MeanLogProbability))
    ),
    "a zero-mass pool must be a typed refusal, got {}",
    describe(&pooled)
  );
}

/// A finite log-probability far below the row maximum keeps its RANK through
/// the linear pool, even where `exp` of it underflows to zero.
///
/// `f64`'s smallest subnormal is 4.94e-324, so anything under ln of it
/// (≈ −744.4) exponentiates to exactly `0.0`; a pool that sums those and takes
/// the log re-emits `-∞` for every tail and hands the caller a ranking that is
/// whatever the tie-break says. Two identical windows must pool back to their
/// common row exactly, tail included.
#[test]
fn the_linear_pool_keeps_a_finite_tail_through_exp_underflow() {
  let mut values = vec![-1_000.0f32; NUM_LANGUAGES];
  values[0] = 0.0;
  values[100] = -800.0;
  values[101] = -900.0;
  let row = LogProbabilities::try_from_slice(&values).expect("a finite row, normalized to 1");

  let pooled = aggregate_windows(
    ScorePooling::MeanProbability,
    &[
      window(row.clone(), 0, WINDOW),
      window(row.clone(), WINDOW, WINDOW),
    ],
  )
  .expect("aggregate");

  let ranked: Vec<usize> = pooled
    .top_k(3)
    .expect("rank")
    .iter()
    .map(|score| score.index())
    .collect();
  assert_eq!(
    ranked,
    vec![0, 100, 101],
    "the finite tail must keep its rank; row[100]={} row[101]={} row[1]={}",
    pooled.as_slice()[100],
    pooled.as_slice()[101],
    pooled.as_slice()[1],
  );
  // Two identical windows pool back to their common row, and in log space the
  // shifted sum makes that exact rather than approximate.
  assert_eq!(pooled.as_slice(), row.as_slice());
}

/// The survey the findings prompted, pinned rather than described: over the
/// domain [`LogProbabilities`] accepts, every pooling either returns a
/// distribution or refuses.
///
/// Windows that are `-∞` in every column are the hardest case that domain
/// admits, and they are refused at the door for every pooling alike — a row
/// that rules every language out is not evidence about any of them. That is one
/// rule rather than four: it used to be that three poolings folded such a row
/// into a zero-mass pool and said so, while [`Vote`] cast its ballot for
/// whatever column the ranking tie-break surfaced and returned a perfectly
/// well-formed distribution over a language nothing had chosen.
///
/// [`Vote`]: ScorePooling::Vote
#[test]
fn every_pooling_either_returns_a_distribution_or_refuses() {
  let nothing = LogProbabilities::try_from_slice(&vec![f32::NEG_INFINITY; NUM_LANGUAGES])
    .expect("an all-zero-probability row is accepted");

  for pooling in all_poolings() {
    let pooled = aggregate_windows(
      pooling,
      &[
        window(nothing.clone(), 0, WINDOW),
        window(nothing.clone(), WINDOW, WINDOW),
      ],
    );
    assert!(
      matches!(&pooled, Err(Error::UnnormalizableWindow(0))),
      "{pooling:?}: {}",
      describe(&pooled)
    );
  }

  // The single-window identity path is not an exception: a lone zero-mass row
  // is refused, not handed back verbatim. Unreachable from the model, whose
  // rows are log-softmax rows; reachable by hand, because `try_from_slice`
  // accepts `-∞` on purpose.
  for pooling in all_poolings() {
    let pooled = aggregate_windows(pooling, &[window(nothing.clone(), 0, WINDOW)]);
    assert!(
      matches!(&pooled, Err(Error::UnnormalizableWindow(0))),
      "{pooling:?} identity: {}",
      describe(&pooled)
    );
  }

  // What the refusal must NOT swallow: a row that is `-∞` in all but one
  // column still has mass, so it is evidence, and a vote over rows like it is
  // still the distribution it always was.
  let certain_of = |index: usize| {
    let mut values = vec![f32::NEG_INFINITY; NUM_LANGUAGES];
    values[index] = 0.0;
    LogProbabilities::try_from_slice(&values).expect("one zero among -inf")
  };
  let voted = aggregate_windows(
    ScorePooling::Vote,
    &[
      window(certain_of(50), 0, WINDOW),
      window(certain_of(3), WINDOW, WINDOW),
    ],
  )
  .expect("a vote over rows that chose something");
  assert_eq!(voted.as_slice()[50], 0.5f32.ln());
  assert_eq!(voted.as_slice()[3], 0.5f32.ln());
  assert_eq!(voted.as_slice()[0], f32::NEG_INFINITY);
  assert!((mass(&voted) - 1.0).abs() < 1e-6, "mass {}", mass(&voted));
}

/// One language ruled out by one window is where the refusal must NOT reach:
/// the row still has mass, so it is an answer and it is returned.
///
/// It also separates the three poolings that meet a `-∞` differently. The
/// logarithmic pool multiplies, so a single window's zero is final. The linear
/// pool adds, so the other window's 0.7 survives. `Max` takes the peak, so it
/// survives there too — and neither of those two can be dragged to zero mass by
/// a `-∞` at all.
#[test]
fn one_language_ruled_out_does_not_rule_out_the_row() {
  let mut values = distribution(&[(0, 0.6), (1, 0.4)]);
  values[5] = f32::NEG_INFINITY;
  let rules_out = LogProbabilities::try_from_slice(&values).expect("row");
  let votes_for =
    LogProbabilities::try_from_slice(&distribution(&[(5, 0.7), (0, 0.3)])).expect("row");
  let windows = vec![
    window(rules_out, 0, WINDOW),
    window(votes_for, WINDOW, WINDOW),
  ];

  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &windows).expect("aggregate");
    assert!(
      (mass(&out) - 1.0).abs() < 1e-5,
      "{pooling:?} mass {}",
      mass(&out)
    );
    let zeroed = out.as_slice()[5] == f32::NEG_INFINITY;
    assert_eq!(
      zeroed,
      pooling == ScorePooling::MeanLogProbability,
      "{pooling:?} column 5 = {}",
      out.as_slice()[5]
    );
  }
}

/// Render an aggregate result for an assertion message: what a caller would
/// actually receive, including the mass that makes it not a distribution.
fn describe(pooled: &Result<LogProbabilities>) -> String {
  match pooled {
    Ok(row) => format!(
      "Ok(mass {}, top-3 {:?}, values[0..3] {:?})",
      mass(row),
      row
        .top_k(3)
        .expect("rank")
        .iter()
        .map(|score| score.index())
        .collect::<Vec<_>>(),
      &row.as_slice()[..3],
    ),
    Err(error) => format!("Err({error})"),
  }
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

// ── The normalizer's own arithmetic ─────────────────────────────────────────

/// A pool that lands far below zero is still normalized.
///
/// Two equal-weight rows, each certain of a different language among columns
/// pinned at a huge finite negative. Both rows have mass 1, so nothing screens
/// them out; the logarithmic pool's per-column mean lands at −5e19, where f64's
/// ULP is 8192. Forming `max + ln(sum)` there returns `max` unchanged — the
/// normalization constant is absorbed whole — and both leading columns come
/// back as exactly `0.0`: two languages, each reported at probability 1, in a
/// row whose mass is 2.
#[test]
fn a_pool_far_below_zero_is_still_normalized() {
  let certain_among_giants = |index: usize| {
    let mut values = vec![-1e20f32; NUM_LANGUAGES];
    values[index] = 0.0;
    LogProbabilities::try_from_slice(&values).expect("finite and non-positive")
  };
  let windows = vec![
    window(certain_among_giants(0), 0, WINDOW),
    window(certain_among_giants(1), WINDOW, WINDOW),
  ];
  // All four, not only the pooling that provoked it: `Max` reaches the same
  // normalizer, and `MeanProbability` carries a shift of its own that this row
  // drives to −1e20 in every column but two.
  for pooling in all_poolings() {
    let pooled = aggregate_windows(pooling, &windows);
    let row = match &pooled {
      Ok(row) => row,
      Err(error) => panic!("{pooling:?} expected a distribution, got Err({error})"),
    };
    assert!(
      (mass(row) - 1.0).abs() < 1e-6,
      "{pooling:?} mass {}, columns 0 and 1 = {} / {}",
      mass(row),
      row.as_slice()[0],
      row.as_slice()[1]
    );
    // Each of the two survivors takes half; the giants keep their own scale.
    assert!(
      (f64::from(row.as_slice()[0]).exp() - 0.5).abs() < 1e-6,
      "{pooling:?} column 0 = {}",
      row.as_slice()[0]
    );
    assert!(
      (f64::from(row.as_slice()[1]).exp() - 0.5).abs() < 1e-6,
      "{pooling:?} column 1 = {}",
      row.as_slice()[1]
    );
  }
}

/// `renormalize` is written so the shift comes off before the small constant
/// does, which is the whole of the fix above. Stated directly on the function
/// so it cannot be lost by a change to which poolings call it — and, since
/// `renormalize` and `push` now share one `DistributionShift`, it pins the
/// arithmetic at BOTH ends of the fold rather than only at the exit.
#[test]
fn renormalize_does_not_lose_its_constant_against_a_huge_shift() {
  let mut values = vec![-1e20f64; NUM_LANGUAGES];
  values[0] = -5e19;
  values[1] = -5e19;
  renormalize(&mut values);
  let total: f64 = values.iter().map(|v| v.exp()).sum();
  assert!((total - 1.0).abs() < 1e-12, "total {total}");
  assert!(
    (values[0] - 0.5f64.ln()).abs() < 1e-12,
    "column 0 = {}",
    values[0]
  );
}

// ── The fold's precondition: a window must be evidence ──────────────────────

/// A window that rules EVERY language out is refused rather than folded.
///
/// It contributes to no numerator — the linear pool skips its terms, which is
/// what keeps `(-∞) − (-∞)` unreachable — while its weight still lands in the
/// denominator, so folding it diluted the pool: a one-hot window beside it came
/// back at probability 0.5, in a row whose total mass was 0.5.
#[test]
fn a_window_with_no_probability_mass_is_refused_not_folded() {
  let mut one_hot = vec![f32::NEG_INFINITY; NUM_LANGUAGES];
  one_hot[0] = 0.0;
  let says_something = LogProbabilities::try_from_slice(&one_hot).expect("row");
  let says_nothing = LogProbabilities::try_from_slice(&vec![f32::NEG_INFINITY; NUM_LANGUAGES])
    .expect("an all-zero-probability row is accepted");
  let windows = vec![
    window(says_something, 0, WINDOW),
    window(says_nothing, WINDOW, WINDOW),
  ];
  let pooled = aggregate_windows(ScorePooling::MeanProbability, &windows);
  assert!(
    matches!(&pooled, Err(Error::UnnormalizableWindow(1))),
    "expected the SECOND window to be named, got {}",
    describe(&pooled)
  );
}

/// The refusal names the offending window's position, so a caller pooling forty
/// windows learns which one it was.
#[test]
fn the_refusal_names_the_window_that_ruled_everything_out() {
  let mut one_hot = vec![f32::NEG_INFINITY; NUM_LANGUAGES];
  one_hot[7] = 0.0;
  let says_something = LogProbabilities::try_from_slice(&one_hot).expect("row");
  let says_nothing =
    LogProbabilities::try_from_slice(&vec![f32::NEG_INFINITY; NUM_LANGUAGES]).expect("row");
  for position in 0..3usize {
    let windows: Vec<_> = (0..3usize)
      .map(|i| {
        let row = if i == position {
          says_nothing.clone()
        } else {
          says_something.clone()
        };
        window(row, i * WINDOW, WINDOW)
      })
      .collect();
    let pooled = aggregate_windows(ScorePooling::Max, &windows);
    assert!(
      matches!(&pooled, Err(Error::UnnormalizableWindow(got)) if *got == position),
      "position {position}: {}",
      describe(&pooled)
    );
  }
}

/// A vote from a window that ruled every language out used to be cast for the
/// TIE-BREAK's column: a language nothing chose taking half the clip's vote
/// share, in a row whose mass was a perfectly respectable 1. The mass check
/// cannot see that one — only refusing the row at the door can.
#[test]
fn a_vote_is_not_cast_by_a_window_that_ruled_everything_out() {
  let mut one_hot = vec![f32::NEG_INFINITY; NUM_LANGUAGES];
  one_hot[50] = 0.0;
  let says_fifty = LogProbabilities::try_from_slice(&one_hot).expect("row");
  let says_nothing =
    LogProbabilities::try_from_slice(&vec![f32::NEG_INFINITY; NUM_LANGUAGES]).expect("row");
  let pooled = aggregate_windows(
    ScorePooling::Vote,
    &[
      window(says_fifty, 0, WINDOW),
      window(says_nothing, WINDOW, WINDOW),
    ],
  );
  assert!(
    matches!(&pooled, Err(Error::UnnormalizableWindow(1))),
    "expected a typed refusal, got {}",
    describe(&pooled)
  );
}

/// `Max` reaches `renormalize` too, and a row of huge finite negatives
/// renormalized to all-zeros once gave 107 languages at probability 1 apiece.
///
/// The row is NOT refused. Its maximum is finite, so it normalizes, and a
/// uniform row written at −1e20 is a uniform DISTRIBUTION — the honest answer,
/// and the one this returns. What closed the defect is `DistributionShift`'s
/// two subtractions held apart, so the log of the shifted sum survives a
/// maximum that far from zero; the door was never the right place for it, and
/// using the door for it is exactly what made the door decide on a row's
/// absolute scale.
#[test]
fn max_over_rows_of_huge_negatives_pools_to_a_distribution() {
  let row = LogProbabilities::try_from_slice(&vec![-1e20f32; NUM_LANGUAGES]).expect("row");
  let pooled = aggregate_windows(
    ScorePooling::Max,
    &[window(row.clone(), 0, WINDOW), window(row, WINDOW, WINDOW)],
  );
  let out = match &pooled {
    Ok(out) => out,
    Err(_) => panic!(
      "a row with a finite maximum must fold, got {}",
      describe(&pooled)
    ),
  };
  // Not 107 columns at exactly 0.0, which is what the fused shift produced.
  let total = mass(out);
  assert!((total - 1.0).abs() < 1e-6, "mass {total}");
  let uniform = (1.0 / NUM_LANGUAGES as f64).ln();
  for (index, value) in out.as_slice().iter().enumerate() {
    assert!(
      (f64::from(*value) - uniform).abs() < 1e-6,
      "column {index}: {value} against the uniform {uniform}"
    );
  }
}

// ── The fold's postcondition: a distribution comes back ─────────────────────

/// The linear pool returns a distribution even from rows that only nearly are.
///
/// It is the one pooling whose output mass would otherwise be its inputs'.
/// Model rows are log-softmax rows narrowed through fp16 and do not sum to 1
/// exactly — measured 7.7e-3 short on `CpuOnly` — and this pooling alone would
/// hand that deficit straight to the caller.
///
/// TWO independent guards now stop it, and this test holds the PROPERTY rather
/// than either mechanism: `push` makes every row a distribution before folding
/// it (which is also what stops the deficit acting as a per-window weight —
/// module docs, "A row's own scale is not evidence"), and the fold still closes
/// with a renormalization whatever the rows were.
#[test]
fn the_linear_pool_returns_a_distribution_from_rows_that_only_nearly_are() {
  let short = (0.99f64 / NUM_LANGUAGES as f64).ln() as f32;
  let deficient = LogProbabilities::try_from_slice(&vec![short; NUM_LANGUAGES]).expect("row");
  let windows = vec![
    window(deficient.clone(), 0, WINDOW),
    window(deficient, WINDOW, WINDOW),
  ];
  let pooled = aggregate_windows(ScorePooling::MeanProbability, &windows);
  let row = match &pooled {
    Ok(row) => row,
    Err(error) => panic!("expected a distribution, got Err({error})"),
  };
  assert!((mass(row) - 1.0).abs() < 1e-6, "mass {}", mass(row));
}

/// The postcondition is a mass CHECK, not a mass repair: it reports the row it
/// refused, so a defect that reaches it is diagnosable rather than merely
/// blocked.
#[test]
fn a_row_that_is_not_a_distribution_carries_the_mass_it_left() {
  let error = Error::from(NotADistribution::new(ScorePooling::MeanProbability, 0.5));
  let rendered = error.to_string();
  assert!(rendered.contains("MeanProbability"), "{rendered}");
  assert!(rendered.contains("0.5"), "{rendered}");
  assert!(rendered.contains("not a distribution"), "{rendered}");
  let Error::NotADistribution(payload) = error else {
    panic!("wrong variant")
  };
  assert_eq!(payload.pooling(), ScorePooling::MeanProbability);
  assert!((payload.mass() - 0.5).abs() < f64::EPSILON);
}

/// Every pooling's folded row sits far inside the tolerance the postcondition
/// enforces — the measured basis for `MAX_MASS_DEVIATION`, pinned so a future
/// change that eats into the margin shows up here rather than in a caller's
/// refused clip.
#[test]
fn every_folded_row_is_normalized_to_far_inside_the_tolerance() {
  let windows = vec![
    window(row_from_logits(&[(94, 7.0), (3, 2.0)]), 0, WINDOW),
    window(
      row_from_logits(&[(3, 4.0), (94, 3.5), (61, 3.0)]),
      WINDOW,
      WINDOW,
    ),
    window(row_from_logits(&[(61, 9.0)]), 2 * WINDOW, WINDOW / 3),
    // A near-uniform row, which is where the f32 narrowing costs the most.
    window(
      row_from_logits(&(0..NUM_LANGUAGES).map(|i| (i, 0.0)).collect::<Vec<_>>()),
      3 * WINDOW,
      WINDOW,
    ),
  ];
  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &windows).expect("aggregate");
    let deviation = (mass(&out) - 1.0).abs();
    assert!(
      deviation < MAX_MASS_DEVIATION / 10.0,
      "{pooling:?} deviation {deviation:e} is within an order of magnitude of \
       the tolerance {MAX_MASS_DEVIATION:e}"
    );
  }
}

// ── A row's own scale ───────────────────────────────────────────────────────

/// The mass a real log-softmax row comes back with on `CpuOnly`: 7.7e-3 short
/// of 1. Its log is what that deficit subtracts from every column of the row,
/// and subtracting a constant from a whole row changes no ratio inside it — so
/// it is the exact shape of "carries no evidence about any language".
const CPU_ONLY_ROW_MASS: f64 = 0.99235;

/// `row` with `by` added to every column — the row at a different overall
/// scale, saying exactly the same thing about every language.
fn rescaled(row: &LogProbabilities, by: f32) -> LogProbabilities {
  let values: Vec<f32> = row.as_slice().iter().map(|v| v + by).collect();
  LogProbabilities::try_from_slice(&values).expect("a non-positive row stays one under a shift")
}

/// The largest per-column gap between two pooled rows, in log space. Equal
/// columns count as zero so a shared `-∞` is a match rather than a NaN.
fn max_abs_difference(a: &LogProbabilities, b: &LogProbabilities) -> f64 {
  a.as_slice()
    .iter()
    .zip(b.as_slice())
    .map(|(&x, &y)| {
      if x == y {
        0.0
      } else {
        (f64::from(x) - f64::from(y)).abs()
      }
    })
    .fold(0.0f64, f64::max)
}

/// A window's own probability-mass deficit is fp noise, and it used to decide
/// the clip.
///
/// Two equal-length windows, each one-hot and so each perfectly certain of its
/// own language: window 0 on column 0 at `ln(0.99235)`, the mass a real
/// `CpuOnly` row comes back with, and window 1 on column 1 at exactly `0.0`.
/// Nothing separates them but the first row's narrowing error. Folding the RAW
/// rows let that error act as a per-window WEIGHT: both `MeanProbability` and
/// `Max` returned col0 = −0.69699425 against col1 = −0.68931484, which is
/// p0 = 0.498080163 against p1 = 0.501919846, so column 1 won both. (`Max`
/// carries the deficit in as `-0.00767941` against `0.0` and the exit shift
/// then lands it in the same place.) The folded mass was 1.0000000086 — four
/// orders inside `MAX_MASS_DEVIATION`, so the postcondition could not see it.
///
/// Normalized at the door the two columns tie exactly, and the crate's
/// ascending-index tie-break takes column 0.
#[test]
fn a_windows_own_mass_deficit_does_not_outvote_an_equally_certain_window() {
  let one_hot = |index: usize, value: f32| {
    let mut values = vec![f32::NEG_INFINITY; NUM_LANGUAGES];
    values[index] = value;
    LogProbabilities::try_from_slice(&values).expect("one finite value among -inf")
  };
  let deficit = CPU_ONLY_ROW_MASS.ln() as f32;
  let windows = vec![
    window(one_hot(0, deficit), 0, WINDOW),
    window(one_hot(1, 0.0), WINDOW, WINDOW),
  ];

  // Every pooling is measured before anything is asserted, so a failure names
  // all of them rather than the first one to go.
  let mut wrong = Vec::new();
  for pooling in [ScorePooling::MeanProbability, ScorePooling::Max] {
    let out = aggregate_windows(pooling, &windows).expect("aggregate");
    let (col0, col1) = (out.as_slice()[0], out.as_slice()[1]);
    let winner = argmax(out.as_slice());
    println!(
      "codex trigger  {pooling:>18?}  winner {winner}  col0 {col0:.8} col1 {col1:.8}  \
       p0 {:.9} p1 {:.9}  mass {:.10}",
      f64::from(col0).exp(),
      f64::from(col1).exp(),
      mass(&out)
    );
    if winner != 0 || col0 != col1 {
      wrong.push(format!(
        "{pooling:?}: winner {winner}, col0 {col0} vs col1 {col1} (p0 {}, p1 {})",
        f64::from(col0).exp(),
        f64::from(col1).exp()
      ));
    }
  }
  assert!(
    wrong.is_empty(),
    "a 7.7e-3 mass deficit, which is fp noise and not evidence, decided the clip:\n  {}",
    wrong.join("\n  ")
  );

  // The other two are unmoved either way, and both are pinned here so the fix
  // is held to changing only what it had to.
  //
  // The logarithmic pool over these two windows has disjoint supports, so its
  // honest answer is that every language has probability zero — before the fix
  // and after it.
  assert!(
    matches!(
      aggregate_windows(ScorePooling::MeanLogProbability, &windows),
      Err(Error::ZeroMassAggregate(ScorePooling::MeanLogProbability))
    ),
    "the logarithmic pool over disjoint supports is refused, not repaired"
  );
  // A vote is an argmax, and an argmax is a comparison: the deficit never
  // reached it, so nothing here moves.
  let voted = aggregate_windows(ScorePooling::Vote, &windows).expect("aggregate");
  assert_eq!(voted.as_slice()[0], 0.5f32.ln());
  assert_eq!(voted.as_slice()[1], 0.5f32.ln());
  assert_eq!(argmax(voted.as_slice()), 0);
}

/// Scale invariance, as a property rather than a comment: rescaling ONE
/// window's row by a constant — which changes no probability ratio inside it,
/// and so tells the fold nothing new about any language — must not move the
/// pooled row by more than the f32 narrowing at the exit.
///
/// This is the invariant the module was missing. The largest per-column gap the
/// rescale opens, over the two overlapping-support rows below:
///
/// | pooling  | folding raw rows | folding normalized rows |
/// |----------|------------------|-------------------------|
/// | meanlog  | 0.0              | 0.0                     |
/// | meanprob | 1.2791e-3        | 0.0                     |
/// | max      | 4.0376e-3        | 5.9605e-8               |
/// | vote     | 0.0              | 0.0                     |
///
/// The test prints the right-hand column on every run, so the table is
/// re-measured rather than remembered.
#[test]
fn the_fold_is_invariant_to_a_rows_own_scale() {
  let baseline =
    LogProbabilities::try_from_slice(&distribution(&[(94, 0.62), (3, 0.23), (61, 0.15)]))
      .expect("row");
  let other = LogProbabilities::try_from_slice(&distribution(&[(3, 0.44), (94, 0.31), (61, 0.25)]))
    .expect("row");
  let deficit = CPU_ONLY_ROW_MASS.ln() as f32;
  let scaled_other = rescaled(&other, deficit);
  // Non-vacuity: the rescale must actually change the row it is given.
  assert!(deficit < 0.0, "the deficit must be a real shift");
  assert_ne!(scaled_other.as_slice(), other.as_slice());

  let plain = vec![
    window(baseline.clone(), 0, WINDOW),
    window(other, WINDOW, WINDOW),
  ];
  let scaled = vec![
    window(baseline, 0, WINDOW),
    window(scaled_other, WINDOW, WINDOW),
  ];

  // All four are measured before anything is asserted, so the failure carries
  // the whole table rather than the first row of it.
  let mut moved = Vec::new();
  for pooling in all_poolings() {
    let from_plain = aggregate_windows(pooling, &plain).expect("aggregate");
    let from_scaled = aggregate_windows(pooling, &scaled).expect("aggregate");
    let deviation = max_abs_difference(&from_plain, &from_scaled);
    println!("scale-invariance  {pooling:>18?}  {deviation:.4e}");
    if deviation >= SCALE_INVARIANCE_FLOOR {
      moved.push(format!("{pooling:?}: {deviation:e}"));
    }
  }
  assert!(
    moved.is_empty(),
    "rescaling one window's row by a constant moved the pooled row further than the \
     f32 narrowing floor {SCALE_INVARIANCE_FLOOR:e}:\n  {}",
    moved.join("\n  ")
  );
}

/// How far a pooled row may move when one window's row is rescaled by a
/// constant.
///
/// Measured, not chosen: the fold runs in f64 and narrows once at the exit, so
/// the only gap that survives is that narrowing. Three of the four poolings
/// come back bit-identical and `Max` moves by 5.96e-8 — one f32 ULP at
/// `ln(0.5)`. This is 17× that, leaving room for a platform whose `exp`/`ln`
/// round a unit differently, and still four orders of magnitude below what
/// folding raw rows produced (1.28e-3 and 4.04e-3), which is the thing it has
/// to catch.
const SCALE_INVARIANCE_FLOOR: f64 = 1e-6;

/// The identity path keeps the row it was GIVEN — deficit and all.
///
/// Normalizing at the door must not reach it: a lone window is the caller's or
/// the model's row, returned verbatim, and that is precisely what makes
/// `identify_long` a bit-for-bit drop-in for `identify`. The row used here is
/// deliberately NOT a distribution, so a normalization that leaked into
/// `self.first` would move every column of it.
#[test]
fn a_lone_window_keeps_its_own_mass_deficit() {
  let deficit = CPU_ONLY_ROW_MASS.ln() as f32;
  let row = rescaled(
    &LogProbabilities::try_from_slice(&distribution(&[(94, 0.62), (3, 0.23), (61, 0.15)]))
      .expect("row"),
    deficit,
  );
  let remaining = mass(&row);
  assert!(
    (remaining - CPU_ONLY_ROW_MASS).abs() < 1e-4,
    "the fixture must not already be a distribution, or this proves nothing: mass {remaining}"
  );

  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &[window(row.clone(), 0, WINDOW)]).expect("aggregate");
    assert_eq!(out.as_slice(), row.as_slice(), "{pooling:?}");
  }
}

// ── The door in front of the fold: what it may and may not decide on ────────

/// A row of [`NUM_LANGUAGES`] values descending by one nat from `top` — the
/// SAME evidence at whatever absolute scale `top` names.
///
/// Every value is an integer, so for any integer `top` whose row stays under
/// 2^24 the whole row is exact in f32 and two rows built at different `top`s
/// differ by EXACTLY their shift. That is what lets the assertions below be
/// bit-equalities rather than tolerances.
fn ramp(top: f32) -> LogProbabilities {
  let values: Vec<f32> = (0..NUM_LANGUAGES).map(|i| top - i as f32).collect();
  LogProbabilities::try_from_slice(&values).expect("a descending non-positive ramp")
}

/// Fold two equal-length windows of `row`.
fn fold_pair(pooling: ScorePooling, row: &LogProbabilities) -> Result<LogProbabilities> {
  aggregate_windows(
    pooling,
    &[
      window(row.clone(), 0, WINDOW),
      window(row.clone(), WINDOW, WINDOW),
    ],
  )
}

/// A row's own scale decided whether the door would take it at all.
///
/// `[-800, -801, …, -906]` and `[0, -1, …, -106]` differ by exactly 800 in
/// every column, so no probability RATIO differs between them and they
/// normalize to the identical distribution — and the door refused the first
/// and folded the second. The guard was `exp(max) > 0.0`, and `exp` underflows
/// f64 to exactly zero below `ln(f64::MIN_POSITIVE)` ≈ −744.44, so a perfectly
/// well-formed row was refused for being WRITTEN low.
///
/// This is the same leak the fold itself was carrying one round earlier, one
/// step further upstream: normalizing at the door stopped a row's scale acting
/// as a weight, but the guard standing in front of the door was still deciding
/// on it.
#[test]
fn a_rows_own_scale_does_not_decide_whether_the_door_accepts_it() {
  let low = ramp(-800.0);
  let high = ramp(0.0);

  // Non-vacuity, both ways: these are different rows, and they say the same
  // thing. Every column sits the same distance below its own row's maximum,
  // exactly, and the two normalize to the identical distribution.
  assert_ne!(low.as_slice(), high.as_slice());
  for (&a, &b) in low.as_slice().iter().zip(high.as_slice()) {
    assert_eq!(a - low.as_slice()[0], b - high.as_slice()[0]);
  }
  assert_eq!(
    as_distribution(low.as_slice()),
    as_distribution(high.as_slice())
  );

  // The arithmetic that used to separate them, pinned so the fixture cannot
  // quietly stop reproducing the regime it was built for.
  assert_eq!(f64::from(low.as_slice()[0]).exp(), 0.0);
  assert!(f64::from(high.as_slice()[0]).exp() > 0.0);

  for pooling in all_poolings() {
    let from_low = fold_pair(pooling, &low);
    let from_high = fold_pair(pooling, &high);
    match (&from_low, &from_high) {
      (Ok(a), Ok(b)) => assert_eq!(
        a.as_slice(),
        b.as_slice(),
        "{pooling:?}: two rows carrying identical evidence must fold alike"
      ),
      _ => panic!(
        "{pooling:?}: the SAME evidence at two scales got two verdicts — \
         low {} / high {}",
        describe(&from_low),
        describe(&from_high)
      ),
    }
  }
}

/// Scale invariance as a PROPERTY of the door, not of one pair: shifting a
/// whole row by a constant changes neither the verdict nor the fold.
///
/// `the_fold_is_invariant_to_a_rows_own_scale` holds the same property one
/// stage later, over rows the door had already accepted. It could not see this
/// one, because a row the door refuses never reaches the fold to be compared.
#[test]
fn the_door_is_invariant_to_a_rows_own_scale() {
  let reference: Vec<Vec<f32>> = all_poolings()
    .into_iter()
    .map(|pooling| {
      fold_pair(pooling, &ramp(0.0))
        .expect("the unshifted ramp folds")
        .as_slice()
        .to_vec()
    })
    .collect();

  // Every shift below keeps the ramp integral and under 2^24, where f32 is
  // exact — so "the same row at another scale" is not itself an approximation
  // and the fold must come back BIT-identical. −744/−745/−746 straddle
  // `ln(f64::MIN_POSITIVE)`, the cliff the old guard fell off.
  for top in [
    -1.0f32,
    -100.0,
    -744.0,
    -745.0,
    -746.0,
    -800.0,
    -10_000.0,
    -16_000_000.0,
  ] {
    let row = ramp(top);
    for (pooling, want) in all_poolings().into_iter().zip(&reference) {
      let got = fold_pair(pooling, &row);
      match &got {
        Ok(folded) => assert_eq!(folded.as_slice(), want.as_slice(), "top {top}, {pooling:?}"),
        Err(_) => panic!("top {top}, {pooling:?}: {}", describe(&got)),
      }
    }
  }

  // Past the point where f32 can hold the ramp the row genuinely changes: at
  // −3e38 the ULP is 2e31, so every column rounds onto `top` and the row
  // really IS uniform. The VERDICT must still not change — a uniform row is a
  // perfectly good distribution, whatever scale it is written at — and what
  // comes back is one.
  let flattened = ramp(-3.0e38);
  assert!(flattened.as_slice().iter().all(|v| *v == -3.0e38));
  for pooling in all_poolings() {
    let got = fold_pair(pooling, &flattened);
    let folded = match &got {
      Ok(folded) => folded,
      Err(_) => panic!("{pooling:?}: {}", describe(&got)),
    };
    let total = mass(folded);
    assert!((total - 1.0).abs() < 1e-6, "{pooling:?}: mass {total}");
  }
}

/// A LONE low-scale window is answered rather than refused, and the answer is
/// the caller's row VERBATIM — which is exactly what a single-shot
/// `identify` over the same row returns.
///
/// The consequence of relaxing the door, stated as a contract rather than
/// discovered: the identity path now hands back a row whose f64 mass is
/// exactly zero. That is not a regression, it is the `identify_long` ==
/// `identify` promise. `finish`'s postcondition does not apply to it (the row
/// is the caller's, not one this module computed) and the ranking is the same
/// ranking `top_k` gives the row on its own.
#[test]
fn a_lone_low_scale_window_comes_back_verbatim_and_ranks_as_identify_would() {
  let low = ramp(-800.0);
  assert_eq!(
    mass(&low),
    0.0,
    "the fixture must be in the underflow regime"
  );

  for pooling in all_poolings() {
    let out = aggregate_windows(pooling, &[window(low.clone(), 0, WINDOW)]);
    let row = match &out {
      Ok(row) => row,
      Err(_) => panic!("{pooling:?}: {}", describe(&out)),
    };
    assert_eq!(row.as_slice(), low.as_slice(), "{pooling:?}");
    // The rank a caller actually receives, against the rank the same row
    // ranked on its own — what `identify` would have reported for it.
    assert_eq!(
      row.top_k(3).expect("rank"),
      low.top_k(3).expect("rank"),
      "{pooling:?}"
    );
  }
}

/// `+∞` is refused at the door, deliberately.
///
/// It cannot arrive from outside this crate — [`LogProbabilities::try_from_slice`]
/// rejects every value above zero, `+∞` among them — and it cannot arrive from
/// the model, whose row [`Identifier::log_probabilities`] refuses outright if
/// any entry is non-finite. What could reach here is a future in-crate producer
/// going through the unvalidated `pub(crate)` `LogProbabilities::new`, and the
/// door's answer to that is a typed refusal naming the window.
///
/// The alternative was to accept it and let the fold deal with it, which is
/// what it did before: three of the four poolings carried the `∞` through to
/// `finish` and failed the mass postcondition with `NotADistribution(inf)`,
/// while a LONE `+∞` window took the identity path and came back to the caller
/// verbatim — a `LogProbabilities` holding a POSITIVE value, which the type's
/// own documented invariant forbids. One predicate at the door closes both.
///
/// [`Identifier::log_probabilities`]: crate::audio::lid::Identifier::log_probabilities
#[test]
fn a_window_whose_maximum_is_not_finite_upward_is_refused_at_the_door() {
  let mut values = vec![-1.0f32; NUM_LANGUAGES];
  values[7] = f32::INFINITY;
  assert!(
    matches!(
      LogProbabilities::try_from_slice(&values),
      Err(Error::InvalidLogProbability(detail)) if detail.index() == 7
    ),
    "the public constructor must still be the first line of defence"
  );
  let row = LogProbabilities::new(values);

  for pooling in all_poolings() {
    let lone = aggregate_windows(pooling, &[window(row.clone(), 0, WINDOW)]);
    assert!(
      matches!(&lone, Err(Error::UnnormalizableWindow(0))),
      "{pooling:?}, lone window: {}",
      describe(&lone)
    );
    let pair = fold_pair(pooling, &row);
    assert!(
      matches!(&pair, Err(Error::UnnormalizableWindow(0))),
      "{pooling:?}, two windows: {}",
      describe(&pair)
    );
  }
}
