//! Unit tests for [`super`] — the AS-Norm calibration door.
//!
//! Every fixture here is CONSTRUCTED, not measured: unit vectors placed by
//! hand in a handful of coordinate axes of the 256-d space, chosen so the
//! geometry each test claims is arithmetic rather than a property of any
//! recording. Nothing in this file asserts anything about WeSpeaker's real
//! output distribution — that is `tests/speaker/`'s job, and it needs models.

use diaric::embed::{Embedding, cosine_similarity};

use crate::audio::speaker::{
  calibrate::{
    AsNormOptions, Cohort, CohortStats, Scoring, VoiceProfile, as_norm,
    cohort_stats_assuming_disjoint, cohort_stats_excluding,
  },
  embed::EMBEDDING_DIM,
  error::CalibrateError,
};

// ── fixtures ─────────────────────────────────────────────────────────────

/// A raw row built from `(axis, weight)` pairs. Un-normalized on purpose:
/// this is what an embedder hands back, and normalizing it is the door's job.
fn row(components: &[(usize, f32)]) -> Vec<f32> {
  let mut v = vec![0.0f32; EMBEDDING_DIM];
  for &(axis, weight) in components {
    v[axis] = weight;
  }
  v
}

/// Turn a `Result` into an ASSERTION rather than a panic, so a red round
/// reports a failed expectation instead of an unwrap.
#[track_caller]
fn ok<T, E: core::fmt::Debug>(r: Result<T, E>, what: &str) -> T {
  assert!(r.is_ok(), "{what} must succeed, got {:?}", r.as_ref().err());
  match r {
    Ok(v) => v,
    Err(_) => unreachable!("asserted Ok immediately above"),
  }
}

/// The 32 impostors of the shared library fixture: a crowd around axis 0, at
/// varying radial distance so their scores against any probe actually spread,
/// and with a small varying component on axis 1 so a speaker sitting there is
/// not scored against a constant.
fn crowd() -> Vec<Vec<f32>> {
  (0..32)
    .map(|i| {
      let radius = 0.20 + 0.01 * i as f32;
      let axis_one = 0.02 + 0.004 * i as f32;
      row(&[(0, 1.0), (1, axis_one), (10 + i, radius)])
    })
    .collect()
}

/// Speaker A, sitting INSIDE the crowd — the speaker every impostor scores
/// highly against.
fn speaker_a() -> Vec<f32> {
  row(&[(0, 1.0), (100, 0.30)])
}

/// A second recording of speaker A.
fn speaker_a_again() -> Vec<f32> {
  row(&[(0, 1.0), (100, 0.30), (101, 0.08)])
}

/// Speaker B, sitting alone — nearly orthogonal to the crowd.
fn speaker_b() -> Vec<f32> {
  row(&[(0, 0.05), (1, 1.0)])
}

/// A second, noisier recording of speaker B.
fn speaker_b_again() -> Vec<f32> {
  row(&[(0, 0.05), (1, 1.0), (200, 0.45)])
}

/// Keys for the shared library fixture. `Impostor` carries the crowd index so
/// a single impostor can be named as the other side of a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Speaker {
  A,
  B,
  Impostor(usize),
}

/// The library every cross-recording test scores against: the crowd, plus
/// speakers A and B themselves. #123's cohort is "sampled from the library
/// itself", so A and B ARE in their own cohorts — which is the whole reason
/// [`cohort_stats_excluding`] has to exist.
fn library(scoring: Scoring) -> Cohort<Speaker, VoiceProfile> {
  let mut cohort = Cohort::new();
  for (i, v) in crowd().iter().enumerate() {
    cohort.push(
      Speaker::Impostor(i),
      ok(scoring.prepare(v), "prepare an impostor"),
    );
  }
  cohort.push(Speaker::A, ok(scoring.prepare(&speaker_a()), "prepare A"));
  cohort.push(Speaker::B, ok(scoring.prepare(&speaker_b()), "prepare B"));
  cohort
}

/// One side of a trial, excluding the probe speaker's own library entries.
fn side(
  cohort: &Cohort<Speaker, VoiceProfile>,
  speaker: Speaker,
  profile: &VoiceProfile,
) -> CohortStats {
  ok(
    cohort_stats_excluding(cohort, &speaker, profile, &AsNormOptions::new()),
    "cohort statistics for a trial side",
  )
}

// ── the score sources ────────────────────────────────────────────────────

/// The cosine door must not be a second cosine implementation: it has to be
/// the SAME arithmetic the online clusterer matches on, or a threshold read
/// off one does not transfer to the other.
#[test]
fn a_cosine_trial_score_is_bit_identical_to_diarics_own_cosine() {
  let (a, b) = (speaker_a(), speaker_b());
  let profile_a = ok(Scoring::Cosine.prepare(&a), "prepare A");
  let profile_b = ok(Scoring::Cosine.prepare(&b), "prepare B");

  let mut arr_a = [0.0f32; EMBEDDING_DIM];
  arr_a.copy_from_slice(&a);
  let mut arr_b = [0.0f32; EMBEDDING_DIM];
  arr_b.copy_from_slice(&b);
  let expected = f64::from(cosine_similarity(
    &Embedding::normalize_from(arr_a).expect("A normalizes"),
    &Embedding::normalize_from(arr_b).expect("B normalizes"),
  ));

  let got = ok(profile_a.score(&profile_b), "score A against B");
  assert_eq!(
    got.to_bits(),
    expected.to_bits(),
    "the cosine score source must BE diaric::embed::cosine_similarity, not agree with it"
  );
}

/// A self-match is the top of the cohort — the fact that makes
/// self-contamination guaranteed rather than merely possible.
#[test]
fn a_cosine_self_match_is_the_largest_score_a_profile_can_obtain() {
  let scoring = Scoring::Cosine;
  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let self_score = ok(a.score(&a), "score A against itself");
  assert!(
    (self_score - 1.0).abs() < 1e-6,
    "an L2-normalized self-match is 1.0, got {self_score}"
  );
  for (i, v) in crowd().iter().enumerate() {
    let impostor = ok(scoring.prepare(v), "prepare an impostor");
    let s = ok(a.score(&impostor), "score A against an impostor");
    assert!(
      s < self_score,
      "impostor {i} scored {s}, at or above A's own self-match {self_score}"
    );
  }
}

/// Both score sources are reachable, and they are genuinely different spaces
/// — a 256-d cosine and a 128-d one — not two spellings of one number.
#[test]
fn both_score_sources_are_reachable_and_are_not_the_same_number() {
  let (a, b) = (speaker_a(), speaker_b());
  let cosine = ok(
    ok(Scoring::Cosine.prepare(&a), "prepare A for cosine")
      .score(&ok(Scoring::Cosine.prepare(&b), "prepare B for cosine")),
    "cosine trial score",
  );
  let plda = ok(
    ok(Scoring::PldaCosine.prepare(&a), "prepare A for plda")
      .score(&ok(Scoring::PldaCosine.prepare(&b), "prepare B for plda")),
    "plda trial score",
  );

  assert!(cosine.is_finite(), "cosine score {cosine} must be finite");
  assert!(plda.is_finite(), "plda score {plda} must be finite");
  assert!(
    (-1.0..=1.0).contains(&cosine) && (-1.0..=1.0).contains(&plda),
    "both sources are cosines and must land in [-1, 1]; got {cosine} and {plda}"
  );
  assert!(
    (cosine - plda).abs() > 1e-6,
    "the two score sources returned {cosine} and {plda}; a PLDA projection that \
     leaves the score unchanged is not a second score source"
  );
}

/// A profile remembers which source prepared it.
#[test]
fn a_profile_reports_the_score_source_it_was_prepared_for() {
  let raw = speaker_a();
  assert_eq!(
    ok(Scoring::Cosine.prepare(&raw), "prepare for cosine").scoring(),
    Scoring::Cosine
  );
  assert_eq!(
    ok(Scoring::PldaCosine.prepare(&raw), "prepare for plda").scoring(),
    Scoring::PldaCosine
  );
}

// ── the self-contamination choice ────────────────────────────────────────

/// The ruling this door exists to preserve: BOTH entrypoints are present, and
/// they answer differently on the cohort #123 actually has — one sampled from
/// the library being scored, so the speaker is in it.
#[test]
fn the_two_self_contamination_entrypoints_answer_differently_on_a_shared_library() {
  let scoring = Scoring::Cosine;
  let cohort = library(scoring);
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let options = AsNormOptions::new();

  let excluding = ok(
    cohort_stats_excluding(&cohort, &Speaker::B, &b, &options),
    "statistics excluding B's own entries",
  );
  let disjoint = ok(
    cohort_stats_assuming_disjoint(&cohort, &b, &options),
    "statistics over the whole cohort",
  );

  assert_eq!(
    disjoint.considered(),
    cohort.len(),
    "the disjoint door scores every member"
  );
  assert_eq!(
    excluding.considered(),
    cohort.len() - 1,
    "the excluding door drops B's one library entry"
  );
  assert!(
    excluding.mean() < disjoint.mean(),
    "B's own entry scores 1.0 against B, so keeping it must raise the mean: \
     excluding {} vs disjoint {}",
    excluding.mean(),
    disjoint.mean()
  );
}

/// Exclusion is by IDENTITY, so it removes ALL of a speaker's entries, not
/// just the one that happens to score highest.
#[test]
fn exclusion_drops_every_entry_a_speaker_owns_not_only_the_self_match() {
  let scoring = Scoring::Cosine;
  let mut cohort = library(scoring);
  // A second library entry for B — a different recording, same identity.
  cohort.push(
    Speaker::B,
    ok(scoring.prepare(&speaker_b_again()), "prepare B again"),
  );

  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let excluding = ok(
    cohort_stats_excluding(&cohort, &Speaker::B, &b, &AsNormOptions::new()),
    "statistics excluding B",
  );
  assert_eq!(
    excluding.considered(),
    cohort.len() - 2,
    "both of B's entries must go, not just the exact self-match"
  );
}

/// A cohort holding nothing but the excluded speaker is a refusal, not an
/// empty-but-usable side.
#[test]
fn a_cohort_that_is_entirely_the_excluded_speaker_is_refused() {
  let scoring = Scoring::Cosine;
  let mut cohort: Cohort<Speaker, VoiceProfile> = Cohort::new();
  cohort.push(Speaker::B, ok(scoring.prepare(&speaker_b()), "prepare B"));
  cohort.push(
    Speaker::B,
    ok(scoring.prepare(&speaker_b_again()), "prepare B again"),
  );
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");

  let refused = cohort_stats_excluding(&cohort, &Speaker::B, &b, &AsNormOptions::new());
  assert!(
    matches!(refused, Err(CalibrateError::ScoreNorm(_))),
    "self-exclusion emptying the cohort must refuse, got {refused:?}"
  );
}

// ── what #123 asked for ──────────────────────────────────────────────────

/// The claim from #123, made falsifiable: raw cosine admits NO single
/// threshold that separates same-speaker trials from different-speaker ones
/// across two differently-placed speakers, and AS-Norm does.
#[test]
fn as_norm_separates_two_differently_placed_speakers_where_no_raw_threshold_can() {
  let scoring = Scoring::Cosine;
  let cohort = library(scoring);

  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let a2 = ok(scoring.prepare(&speaker_a_again()), "prepare A again");
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");
  let impostor = ok(
    scoring.prepare(&crowd()[0]),
    "prepare the impostor nearest A",
  );

  // Raw trial scores.
  let raw_genuine_a = ok(a.score(&a2), "A vs A'");
  let raw_genuine_b = ok(b.score(&b2), "B vs B'");
  let raw_impostor = ok(a.score(&impostor), "A vs an impostor");

  // A threshold separates iff the weakest genuine trial outscores the
  // strongest impostor one. Raw scores fail that test.
  assert!(
    raw_genuine_b < raw_impostor,
    "the fixture must reproduce #123's problem: a genuine trial for the \
     isolated speaker ({raw_genuine_b}) has to score BELOW an impostor trial \
     for the crowded one ({raw_impostor})"
  );

  // The same trials, normalized. Every side excludes its own speaker's
  // library entries — only the caller knows a probe's identity.
  let side_a = side(&cohort, Speaker::A, &a);
  let side_a2 = side(&cohort, Speaker::A, &a2);
  let side_b = side(&cohort, Speaker::B, &b);
  let side_b2 = side(&cohort, Speaker::B, &b2);
  let side_impostor = side(&cohort, Speaker::Impostor(0), &impostor);

  let norm_genuine_a = ok(
    as_norm(raw_genuine_a, &side_a, &side_a2),
    "A vs A' normalized",
  );
  let norm_genuine_b = ok(
    as_norm(raw_genuine_b, &side_b, &side_b2),
    "B vs B' normalized",
  );
  let norm_impostor = ok(
    as_norm(raw_impostor, &side_a, &side_impostor),
    "A vs impostor normalized",
  );

  assert!(
    norm_genuine_a > norm_impostor && norm_genuine_b > norm_impostor,
    "after AS-Norm a single threshold must separate: genuine A {norm_genuine_a}, \
     genuine B {norm_genuine_b}, impostor {norm_impostor}"
  );
}

/// The PLDA-projected source runs the same road end to end.
#[test]
fn the_plda_score_source_normalizes_end_to_end() {
  let scoring = Scoring::PldaCosine;
  let cohort = library(scoring);
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");

  let raw = ok(b.score(&b2), "B vs B'");
  let side_b = side(&cohort, Speaker::B, &b);
  let side_b2 = side(&cohort, Speaker::B, &b2);
  let normalized = ok(as_norm(raw, &side_b, &side_b2), "B vs B' normalized");
  assert!(
    normalized.is_finite(),
    "a PLDA-space normalization must produce a usable number, got {normalized}"
  );

  // The statistics have to come out of the PLDA space, not out of a
  // `PldaCosine` that quietly degraded to `Cosine` somewhere in `prepare`.
  let cosine_cohort = library(Scoring::Cosine);
  let cosine_b = ok(
    Scoring::Cosine.prepare(&speaker_b()),
    "prepare B for cosine",
  );
  let cosine_side = side(&cosine_cohort, Speaker::B, &cosine_b);
  assert!(
    (side_b.mean() - cosine_side.mean()).abs() > 1e-6
      || (side_b.deviation() - cosine_side.deviation()).abs() > 1e-6,
    "the PLDA side ({}, {}) is indistinguishable from the cosine side ({}, {})",
    side_b.mean(),
    side_b.deviation(),
    cosine_side.mean(),
    cosine_side.deviation()
  );
}

// ── refusals ─────────────────────────────────────────────────────────────

/// A row of the wrong width is a typed refusal carrying both numbers, not a
/// silent truncation or pad.
#[test]
fn a_raw_row_of_the_wrong_width_is_refused() {
  for scoring in [Scoring::Cosine, Scoring::PldaCosine] {
    for len in [0usize, EMBEDDING_DIM - 1, EMBEDDING_DIM + 1] {
      let refused = scoring.prepare(&vec![0.5f32; len]);
      match refused {
        Err(CalibrateError::ProfileLength(p)) => {
          assert_eq!(p.got(), len);
          assert_eq!(p.expected(), EMBEDDING_DIM);
        }
        other => panic!("{scoring:?} must refuse a {len}-element row, got {other:?}"),
      }
    }
  }
}

/// A row with no direction is refused by BOTH sources, each through its own
/// backend's floor rather than a floor invented here.
#[test]
fn a_row_with_no_usable_direction_is_refused_by_both_sources() {
  let zero = vec![0.0f32; EMBEDDING_DIM];
  assert!(
    matches!(
      Scoring::Cosine.prepare(&zero),
      Err(CalibrateError::DegenerateProfile(Scoring::Cosine))
    ),
    "the cosine source must refuse a zero row"
  );
  let plda = Scoring::PldaCosine.prepare(&zero);
  assert!(
    matches!(
      plda,
      Err(CalibrateError::DegenerateProfile(Scoring::PldaCosine)) | Err(CalibrateError::Plda(_))
    ),
    "the plda source must refuse a zero row, got {plda:?}"
  );
}

/// A non-finite row never becomes a profile.
#[test]
fn a_non_finite_row_never_becomes_a_profile() {
  for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    let mut v = row(&[(0, 1.0), (5, 0.3)]);
    v[7] = bad;
    for scoring in [Scoring::Cosine, Scoring::PldaCosine] {
      assert!(
        scoring.prepare(&v).is_err(),
        "{scoring:?} must refuse a row containing {bad}"
      );
    }
  }
}

/// Mixing the two score sources in one trial is refused, and the refusal
/// names both sides.
#[test]
fn scoring_across_two_score_sources_is_refused() {
  let raw = speaker_a();
  let cosine = ok(Scoring::Cosine.prepare(&raw), "prepare for cosine");
  let plda = ok(Scoring::PldaCosine.prepare(&raw), "prepare for plda");

  match cosine.score(&plda) {
    Err(CalibrateError::ScoringMismatch(m)) => {
      assert_eq!(m.side(), Scoring::Cosine);
      assert_eq!(m.other(), Scoring::PldaCosine);
    }
    other => panic!("a cross-source trial must be refused, got {other:?}"),
  }
  match plda.score(&cosine) {
    Err(CalibrateError::ScoringMismatch(m)) => {
      assert_eq!(m.side(), Scoring::PldaCosine);
      assert_eq!(m.other(), Scoring::Cosine);
    }
    other => panic!("a cross-source trial must be refused, got {other:?}"),
  }
}

/// A cohort holding a foreign-source entry poisons a mean silently unless the
/// door refuses it, so it refuses it — through BOTH entrypoints.
#[test]
fn a_cohort_mixing_two_score_sources_is_refused_rather_than_averaged() {
  let mut cohort = library(Scoring::Cosine);
  cohort.push(
    Speaker::Impostor(999),
    ok(
      Scoring::PldaCosine.prepare(&speaker_b()),
      "prepare a foreign-source entry",
    ),
  );
  let a = ok(Scoring::Cosine.prepare(&speaker_a()), "prepare A");
  let options = AsNormOptions::new();

  let excluding = cohort_stats_excluding(&cohort, &Speaker::A, &a, &options);
  assert!(
    matches!(excluding, Err(CalibrateError::ScoringMismatch(_))),
    "a mixed cohort must be refused by the excluding door, got {excluding:?}"
  );
  let disjoint = cohort_stats_assuming_disjoint(&cohort, &a, &options);
  assert!(
    matches!(disjoint, Err(CalibrateError::ScoringMismatch(_))),
    "a mixed cohort must be refused by the disjoint door, got {disjoint:?}"
  );
}

/// The refusal survives the closure boundary: `diaric` takes an INFALLIBLE
/// scoring function, so a mismatch has to be carried out by hand. This pins
/// that the carried error is the one reported, not the `NonFiniteScore` the
/// poison value would otherwise produce.
#[test]
fn a_mismatch_inside_the_cohort_reports_the_mismatch_not_the_poison_value() {
  let mut cohort: Cohort<Speaker, VoiceProfile> = Cohort::new();
  for (i, v) in crowd().iter().enumerate() {
    cohort.push(
      Speaker::Impostor(i),
      ok(Scoring::PldaCosine.prepare(v), "prepare a foreign entry"),
    );
  }
  let a = ok(Scoring::Cosine.prepare(&speaker_a()), "prepare A");
  let refused = cohort_stats_assuming_disjoint(&cohort, &a, &AsNormOptions::new());
  assert!(
    matches!(refused, Err(CalibrateError::ScoringMismatch(_))),
    "every entry mismatches, so the mismatch must be reported: {refused:?}"
  );
}

/// The options are `diaric`'s own, and their defaults reach this door
/// unchanged — a threshold read off published AS-Norm numbers depends on it.
#[test]
fn the_options_defaults_are_diarics_own() {
  let options = AsNormOptions::new();
  assert_eq!(
    options.top_n().get(),
    super::DEFAULT_TOP_N,
    "top_n default must be the re-exported constant"
  );
  assert_eq!(options.min_deviation(), super::DEFAULT_MIN_DEVIATION);
  assert_eq!(super::DEFAULT_TOP_N, diaric::score_norm::DEFAULT_TOP_N);
  assert_eq!(
    super::MIN_COHORT_SCORES,
    diaric::score_norm::MIN_COHORT_SCORES
  );
  assert_eq!(
    super::MAX_NORMALIZED_ERROR,
    diaric::score_norm::MAX_NORMALIZED_ERROR
  );
}

/// The caller's [`AsNormOptions`] has to REACH `diaric`'s selection. A wrapper
/// that quietly substituted the defaults would pass every other test here.
#[test]
fn the_callers_options_reach_diarics_selection() {
  use core::num::NonZeroUsize;

  let scoring = Scoring::Cosine;
  let cohort = library(scoring);
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let narrow = AsNormOptions::new().with_top_n(NonZeroUsize::new(4).expect("4 is non-zero"));

  let wide_side = ok(
    cohort_stats_excluding(&cohort, &Speaker::B, &b, &AsNormOptions::new()),
    "statistics at the default top_n",
  );
  let narrow_side = ok(
    cohort_stats_excluding(&cohort, &Speaker::B, &b, &narrow),
    "statistics at top_n = 4",
  );
  assert_eq!(
    wide_side.selected(),
    cohort.len() - 1,
    "the default top_n is far above this cohort, so every considered score is selected"
  );
  assert_eq!(narrow_side.selected(), 4);

  // Through the other door too, so neither wrapper can be the one that drops
  // the argument.
  let narrow_disjoint = ok(
    cohort_stats_assuming_disjoint(&cohort, &b, &narrow),
    "whole-cohort statistics at top_n = 4",
  );
  assert_eq!(narrow_disjoint.selected(), 4);
}

/// `diaric`'s minimum-usable-cohort floor reaches through the wrapper, and
/// arrives as its OWN refusal — a cohort that is merely absent must not be
/// reported as a degenerate one.
#[test]
fn a_cohort_below_the_minimum_usable_size_is_refused_as_too_small() {
  let scoring = Scoring::Cosine;
  let mut cohort: Cohort<Speaker, VoiceProfile> = Cohort::new();
  cohort.push(
    Speaker::Impostor(0),
    ok(scoring.prepare(&crowd()[0]), "prepare an impostor"),
  );
  assert!(cohort.len() < super::MIN_COHORT_SCORES);

  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let refused = cohort_stats_assuming_disjoint(&cohort, &b, &AsNormOptions::new());
  match refused {
    Err(CalibrateError::ScoreNorm(diaric::score_norm::Error::CohortTooSmall(t))) => {
      assert_eq!(t.available(), 1);
      assert_eq!(t.required(), super::MIN_COHORT_SCORES);
    }
    other => panic!("a one-member cohort must be refused as too small, got {other:?}"),
  }
}

/// A profile is plain data and must stay movable and shareable across threads:
/// a library holds thousands of them and a confusion experiment fans them out.
/// Pinned at compile time so a future field type cannot regress the auto-derive
/// silently.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<VoiceProfile>();
  assert_send_sync::<Scoring>();
  assert_send_sync::<CalibrateError>();
};
