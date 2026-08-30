use super::*;
use crate::audio::lid::NUM_LANGUAGES;

/// A full log-probability row: everything at `base`, with `(index, value)`
/// overrides. Not normalized — ranking never depends on normalization.
fn row(base: f32, overrides: &[(usize, f32)]) -> Vec<f32> {
  let mut values = vec![base; NUM_LANGUAGES];
  for &(index, value) in overrides {
    values[index] = value;
  }
  values
}

fn top_k(values: &[f32], k: usize) -> Vec<LanguageScore> {
  top_k_from_scores(values.iter().copied().enumerate(), k).expect("ranking")
}

/// The ranked output is descending by log probability and carries the resolved
/// roster row, not a bare index.
#[test]
fn ranking_is_descending_and_carries_the_roster_row() {
  // The real Thai anchor's shape: one dominant language, a distant runner-up.
  let values = row(-25.0, &[(94, -0.0101), (55, -4.6038), (52, -18.7132)]);
  let ranked = top_k(&values, 3);

  assert_eq!(ranked.len(), 3);
  assert_eq!(ranked[0].index(), 94);
  assert_eq!(ranked[0].code(), "th");
  assert_eq!(ranked[0].name(), "Thai");
  assert_eq!(ranked[1].code(), "lo");
  assert_eq!(ranked[2].code(), "la");

  for pair in ranked.windows(2) {
    assert!(
      pair[0].log_probability() >= pair[1].log_probability(),
      "output must be descending"
    );
  }

  // `language()` is the same row `labels` hands out, not a copy.
  assert_eq!(
    ranked[0].language(),
    crate::audio::lid::Language::from_index(94).expect("Thai")
  );
}

/// `probability` is `exp` of the log score, and the log form keeps resolution
/// the probability form throws away — the reason the accessor pair exists.
#[test]
fn probability_is_exp_of_the_log_score() {
  let values = row(-30.0, &[(94, -0.0101), (55, -4.6038), (52, -18.7132)]);
  let ranked = top_k(&values, 3);

  assert!((ranked[0].probability() - 0.98995).abs() < 1e-4);
  assert!((ranked[1].probability() - 0.01001).abs() < 1e-4);
  assert_eq!(ranked[0].probability(), ranked[0].log_probability().exp());

  // The third and a hypothetical much worse language both vanish in
  // probability space while staying far apart in log space.
  assert!(ranked[2].probability() < 1e-6);
  assert!(ranked[2].log_probability() - (-30.0) > 10.0);
}

/// Ranking in log space gives the same order as ranking the exponentials, which
/// is what lets the heap run on the raw model output with no `exp` per element.
#[test]
fn log_space_ranking_matches_probability_space_ranking() {
  let mut values = row(-9.0, &[]);
  for (i, slot) in values.iter_mut().enumerate() {
    *slot = -((i % 17) as f32) - 0.25 * (i as f32 % 5.0);
  }
  let by_log: Vec<usize> = top_k(&values, NUM_LANGUAGES)
    .iter()
    .map(LanguageScore::index)
    .collect();

  let mut by_probability: Vec<(usize, f32)> = values.iter().map(|v| v.exp()).enumerate().collect();
  by_probability.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

  assert_eq!(
    by_log,
    by_probability.iter().map(|(i, _)| *i).collect::<Vec<_>>()
  );
}

/// Ties break by ASCENDING language index, so the ranking is total and
/// reproducible — a flat row must come back in roster order.
#[test]
fn ties_break_by_ascending_language_index() {
  let flat = row(-4.5, &[]);
  let ranked = top_k(&flat, 5);
  assert_eq!(
    ranked.iter().map(LanguageScore::index).collect::<Vec<_>>(),
    vec![0, 1, 2, 3, 4]
  );

  // A tie at the top of an otherwise-varied row resolves the same way.
  let values = row(-9.0, &[(80, -1.0), (12, -1.0), (44, -2.0)]);
  let ranked = top_k(&values, 3);
  assert_eq!(
    ranked.iter().map(LanguageScore::index).collect::<Vec<_>>(),
    vec![12, 80, 44]
  );
}

/// `k` degenerates safely: zero yields nothing, oversized `k` saturates at the
/// roster size, and `usize::MAX` cannot overflow the heap's pre-allocation.
#[test]
fn k_zero_and_oversized_k_are_safe() {
  let values = row(-3.0, &[(7, -0.5)]);
  assert!(top_k(&values, 0).is_empty());
  assert_eq!(top_k(&values, 1).len(), 1);
  assert_eq!(top_k(&values, NUM_LANGUAGES).len(), NUM_LANGUAGES);
  assert_eq!(top_k(&values, NUM_LANGUAGES + 50).len(), NUM_LANGUAGES);
  assert_eq!(top_k(&values, usize::MAX).len(), NUM_LANGUAGES);
  assert_eq!(top_k(&values, usize::MAX)[0].index(), 7);
}

/// The whole roster, ranked, is a permutation of it — nothing dropped, nothing
/// duplicated by the heap's replace-the-smallest loop.
#[test]
fn ranking_everything_is_a_permutation_of_the_roster() {
  let values: Vec<f32> = (0..NUM_LANGUAGES)
    .map(|i| -((i as f32) * 0.37 % 11.0))
    .collect();
  let mut indices: Vec<usize> = top_k(&values, NUM_LANGUAGES)
    .iter()
    .map(LanguageScore::index)
    .collect();
  assert_eq!(indices.len(), NUM_LANGUAGES);
  indices.sort_unstable();
  assert_eq!(indices, (0..NUM_LANGUAGES).collect::<Vec<_>>());
}

/// An out-of-roster index is a typed error, never a panic. Unreachable through
/// the model path (the output row is width-checked first), so it is exercised
/// by feeding the ranker directly.
#[test]
fn an_out_of_roster_index_is_a_typed_error() {
  let error = top_k_from_scores([(NUM_LANGUAGES, -1.0f32)], 1).expect_err("must reject");
  assert!(matches!(error, Error::UnknownLanguageIndex(i) if i == NUM_LANGUAGES));
}
