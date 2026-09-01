//! Unit tests for [`super`] — the AS-Norm calibration door.
//!
//! Every fixture here is CONSTRUCTED, not measured: unit vectors placed by
//! hand in a handful of coordinate axes of the 256-d space, chosen so the
//! geometry each test claims is arithmetic rather than a property of any
//! recording. Nothing in this file asserts anything about WeSpeaker's real
//! output distribution — that is `tests/speaker/`'s job, and it needs models.

use diaric::{
  embed::{Embedding, cosine_similarity},
  // `CohortStats` is `diaric`'s own untagged statistic, and it is deliberately
  // NOT re-exported by `super` any more (module docs, "The untagged road is a
  // different crate, not a re-export"). These tests reach it the way a caller
  // now has to — through `diaric` itself — which is also what lets them build
  // the arrangements the doors refuse, in order to pin WHY they refuse them.
  score_norm::CohortStats,
};

use crate::audio::speaker::{
  calibrate::{
    AsNormOptions, CohortId, Enrolled, HeldOutCohort, LibraryCohort, Scoring, SideStats,
    TrialScore, VoiceProfile, as_norm, enrolled_stats, held_out_stats,
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

/// Where the held-out partition of [`crowd`] starts. Members below this index
/// can appear as a trial partner; members at or above it are cohort-only and
/// never sit on a side of a trial — which is what makes
/// [`HeldOutCohort::assuming_disjoint`] a true statement about this fixture
/// rather than a convenient one.
const HELD_OUT_FROM: usize = 8;

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

/// The library-sampled cohort: the crowd, plus speakers A and B themselves.
/// #123's cohort is "sampled from the library itself", so A and B ARE in their
/// own cohorts — which is the whole reason [`enrolled_stats`] excludes by
/// identity.
fn library(scoring: Scoring) -> LibraryCohort<Speaker> {
  let mut cohort = LibraryCohort::new();
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

/// The held-out cohort: crowd members [`HELD_OUT_FROM`]`..`, which hold no
/// material from A, from B, or from the one impostor used as a trial partner.
fn held_out(scoring: Scoring) -> HeldOutCohort {
  HeldOutCohort::assuming_disjoint(
    crowd()
      .iter()
      .skip(HELD_OUT_FROM)
      .map(|v| ok(scoring.prepare(v), "prepare a held-out impostor"))
      .collect(),
  )
}

/// One side of a trial over the held-out cohort — the arrangement a trial with
/// an unidentified probe in it has to use, since both sides must come from one
/// cohort and a probe's can only be a held-out one.
fn side(cohort: &HeldOutCohort, profile: &VoiceProfile) -> SideStats {
  ok(
    held_out_stats(cohort, profile, &AsNormOptions::new()),
    "cohort statistics for a trial side",
  )
}

/// The worked case behind the finding this shape answers: a three-member
/// cohort in which a probe scores `[0, 0.8, 0.2]`. Returns the probe row and
/// the three member rows.
fn worked_case() -> (Vec<f32>, [Vec<f32>; 3]) {
  let probe = row(&[(0, 1.0)]);
  let members = [
    row(&[(1, 1.0)]),                 // cos(probe, ·) = 0.0
    row(&[(0, 0.8), (1, 0.6)]),       // cos(probe, ·) = 0.8
    row(&[(0, 0.2), (2, 0.979_796)]), // cos(probe, ·) = 0.2
  ];
  (probe, members)
}

/// A threshold the worked case straddles: the whole-cohort normalization lands
/// below it, and the candidate-truncated one above it.
const WORKED_CASE_THRESHOLD: f64 = 3.0;

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

  let got = ok(profile_a.score(&profile_b), "score A against B").raw();
  assert_eq!(
    got.to_bits(),
    expected.to_bits(),
    "the cosine score source must BE diaric::embed::cosine_similarity, not agree with it"
  );
}

/// A self-match is the top of the cohort — the fact that makes
/// self-contamination guaranteed rather than merely possible, and therefore
/// the fact [`HeldOutCohort::assuming_disjoint`] exists to keep out.
#[test]
fn a_cosine_self_match_is_the_largest_score_a_profile_can_obtain() {
  let scoring = Scoring::Cosine;
  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let self_score = ok(a.score(&a), "score A against itself").raw();
  assert!(
    (self_score - 1.0).abs() < 1e-6,
    "an L2-normalized self-match is 1.0, got {self_score}"
  );
  for (i, v) in crowd().iter().enumerate() {
    let impostor = ok(scoring.prepare(v), "prepare an impostor");
    let s = ok(a.score(&impostor), "score A against an impostor").raw();
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
  )
  .raw();
  let plda = ok(
    ok(Scoring::PldaCosine.prepare(&a), "prepare A for plda")
      .score(&ok(Scoring::PldaCosine.prepare(&b), "prepare B for plda")),
    "plda trial score",
  )
  .raw();

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

/// The tag has to SURVIVE each step, not merely be checked at the first one:
/// a trial score and both kinds of cohort statistic carry the source they were
/// computed in, which is what makes the final check in [`as_norm`] possible at
/// all.
#[test]
fn a_trial_score_and_a_side_statistic_carry_the_source_they_were_computed_in() {
  let options = AsNormOptions::new();
  for scoring in [Scoring::Cosine, Scoring::PldaCosine] {
    let a = ok(scoring.prepare(&speaker_a()), "prepare A");
    let b = ok(scoring.prepare(&speaker_b()), "prepare B");

    assert_eq!(
      ok(a.score(&b), "a trial score").scoring(),
      scoring,
      "a trial score must report the source both profiles were prepared for"
    );
    assert_eq!(
      side(&held_out(scoring), &a).scoring(),
      scoring,
      "a held-out side must report the source it was computed in"
    );
    assert_eq!(
      ok(
        enrolled_stats(&library(scoring), Enrolled::new(&Speaker::A, &a), &options),
        "A's enrolled side",
      )
      .scoring(),
      scoring,
      "an enrolled side must report the source it was computed in"
    );
  }
}

// ── the two cohorts, and what each side may use ──────────────────────────

/// Why [`HeldOutCohort::assuming_disjoint`] is an assertion worth making: a
/// cohort that keeps a speaker's own entry scores it at `1.0` and pulls the
/// mean up, and the contaminated side still looks perfectly healthy.
#[test]
fn keeping_a_speakers_own_entry_raises_its_cohort_mean() {
  let scoring = Scoring::Cosine;
  let cohort = library(scoring);
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let options = AsNormOptions::new();

  let excluding = ok(
    enrolled_stats(&cohort, Enrolled::new(&Speaker::B, &b), &options),
    "statistics excluding B's own entries",
  );
  // The same profiles, wrongly asserted to be held out — B's own entry
  // included. Only a test would state that; it is the failure the assertion
  // exists to keep out.
  let contaminated_members: Vec<VoiceProfile> =
    cohort.entries.entries().iter().map(|e| *e.item()).collect();
  let contaminated = ok(
    held_out_stats(
      &HeldOutCohort::assuming_disjoint(contaminated_members),
      &b,
      &options,
    ),
    "statistics over a cohort that still holds B",
  );

  assert_eq!(
    contaminated.considered(),
    cohort.len(),
    "the held-out door scores every member"
  );
  assert_eq!(
    excluding.considered(),
    cohort.len() - 1,
    "the enrolled door drops B's one library entry"
  );
  assert!(
    excluding.mean() < contaminated.mean(),
    "B's own entry scores 1.0 against B, so keeping it must raise the mean: \
     excluding {} vs contaminated {}",
    excluding.mean(),
    contaminated.mean()
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
    enrolled_stats(
      &cohort,
      Enrolled::new(&Speaker::B, &b),
      &AsNormOptions::new(),
    ),
    "statistics excluding B",
  );
  assert_eq!(
    excluding.considered(),
    cohort.len() - 2,
    "both of B's entries must go, not just the exact self-match"
  );
}

/// The SCOPE of the exclusion, pinned as arithmetic: an enrolled side drops
/// its own speaker and nobody else — least of all the partner it is about to
/// be scored against, which is what keeps the side a property of one speaker
/// plus the cohort, and therefore reusable.
#[test]
fn an_enrolled_side_drops_only_its_own_speaker_never_the_partner() {
  let scoring = Scoring::Cosine;
  let (_, member_rows) = worked_case();
  let options = AsNormOptions::new();

  let profiles: Vec<VoiceProfile> = member_rows
    .iter()
    .map(|v| ok(scoring.prepare(v), "prepare a cohort member"))
    .collect();
  let mut cohort: LibraryCohort<Speaker> = LibraryCohort::new();
  for (i, p) in profiles.iter().enumerate() {
    cohort.push(Speaker::Impostor(i), *p);
  }

  let enrolled = profiles[1];
  let got = ok(
    enrolled_stats(
      &cohort,
      Enrolled::new(&Speaker::Impostor(1), &enrolled),
      &options,
    ),
    "the enrolled side",
  );

  // Exactly the scores against the other two members, member 2 included —
  // member 2 is the trial partner in the finding-1 test below.
  let by_hand = ok(
    CohortStats::from_scores(
      [
        ok(enrolled.score(&profiles[0]), "score against member 0").raw(),
        ok(enrolled.score(&profiles[2]), "score against member 2").raw(),
      ],
      &options,
    ),
    "the statistics written out by hand",
  );

  assert_eq!(got.considered(), 2, "only member 1's own entry may go");
  assert_eq!(
    got.mean().to_bits(),
    by_hand.mean().to_bits(),
    "an enrolled side must be exactly the scores against every OTHER speaker: \
     got {} against {}",
    got.mean(),
    by_hand.mean()
  );
  assert_eq!(got.deviation().to_bits(), by_hand.deviation().to_bits());
}

/// A cohort holding nothing but the excluded speaker is a refusal, not an
/// empty-but-usable side.
#[test]
fn a_cohort_that_is_entirely_the_excluded_speaker_is_refused() {
  let scoring = Scoring::Cosine;
  let mut cohort: LibraryCohort<Speaker> = LibraryCohort::new();
  cohort.push(Speaker::B, ok(scoring.prepare(&speaker_b()), "prepare B"));
  cohort.push(
    Speaker::B,
    ok(scoring.prepare(&speaker_b_again()), "prepare B again"),
  );
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");

  let refused = enrolled_stats(
    &cohort,
    Enrolled::new(&Speaker::B, &b),
    &AsNormOptions::new(),
  );
  assert!(
    matches!(refused, Err(CalibrateError::ScoreNorm(_))),
    "self-exclusion emptying the cohort must refuse, got {refused:?}"
  );
}

/// FINDING 1, pinned as numbers. An unenrolled probe's identity is what
/// identification is trying to discover, so its side covers the WHOLE cohort;
/// the alternative this module's first version recommended — dropping the
/// candidate's entry from the probe's side — moves the normalized trial across
/// a threshold.
///
/// The truncated statistics have to be built by hand out of `diaric`'s own
/// constructor, because no entrypoint here can produce them any more — and so
/// does the whole-cohort normalization, for a second reason that arrived with
/// the cohort identity: the candidate's side comes from the library cohort and
/// the probe's from the held-out one, and [`as_norm`] now refuses that pairing
/// outright (`two_sides_taken_over_different_cohorts_are_refused` says why).
/// Both numbers here therefore run through `diaric`'s own `normalize` with the
/// SAME enrolment statistics, which is what makes this a comparison of the two
/// PROBE sides and of nothing else.
#[test]
fn an_unidentified_probes_side_covers_its_whole_cohort() {
  let scoring = Scoring::Cosine;
  let (probe_row, member_rows) = worked_case();
  let options = AsNormOptions::new();

  let profiles: Vec<VoiceProfile> = member_rows
    .iter()
    .map(|v| ok(scoring.prepare(v), "prepare a cohort member"))
    .collect();
  let probe = ok(scoring.prepare(&probe_row), "prepare the probe");
  let candidate = profiles[1];

  // The probe's side: a held-out cohort, nothing excluded, and no candidate
  // anywhere in the call.
  let cohort = HeldOutCohort::assuming_disjoint(profiles.clone());
  let probe_side = ok(
    held_out_stats(&cohort, &probe, &options),
    "the probe's cohort statistics",
  );
  assert_eq!(
    probe_side.considered(),
    cohort.len(),
    "a probe has no identity to exclude, so every cohort member must be scored"
  );
  assert_eq!(probe_side.selected(), 3);

  // The candidate's side, from the library-sampled cohort that names it.
  let mut library_cohort: LibraryCohort<Speaker> = LibraryCohort::new();
  for (i, p) in profiles.iter().enumerate() {
    library_cohort.push(Speaker::Impostor(i), *p);
  }
  let enrolled_side = ok(
    enrolled_stats(
      &library_cohort,
      Enrolled::new(&Speaker::Impostor(1), &candidate),
      &options,
    ),
    "the candidate's cohort statistics",
  );

  let trial = ok(candidate.score(&probe), "the trial score");

  // Two cohorts, so the tagged door refuses the pair before any arithmetic.
  assert!(
    matches!(
      as_norm(trial, &enrolled_side, &probe_side),
      Err(CalibrateError::CohortMismatch(_))
    ),
    "a library-sampled enrolment side and a held-out probe side are two \
     impostor populations, and averaging their z-scores is the refusal"
  );

  // The same trial, with the candidate's entry dropped from the PROBE's side.
  let truncated = ok(
    CohortStats::from_scores(
      [
        ok(probe.score(&profiles[0]), "probe against member 0").raw(),
        ok(probe.score(&profiles[2]), "probe against member 2").raw(),
      ],
      &options,
    ),
    "the candidate-truncated probe statistics",
  );
  let enrolled_untagged = ok(
    CohortStats::from_scores(
      [
        ok(candidate.score(&profiles[0]), "candidate against member 0").raw(),
        ok(candidate.score(&profiles[2]), "candidate against member 2").raw(),
      ],
      &options,
    ),
    "the candidate's statistics written out by hand",
  );
  assert_eq!(
    enrolled_untagged.mean().to_bits(),
    enrolled_side.mean().to_bits(),
    "the hand-built enrolment side must be the door's own, or the comparison \
     below is between two different trials"
  );
  // One enrolment side, two probe sides: the door's whole-cohort one, and the
  // candidate-truncated alternative. `probe_side.stats` is the door's own
  // statistic, read through the private field these tests can see, so the
  // honest half is not a second hand-built copy of it.
  let honest = ok(
    enrolled_untagged.normalize(trial.raw(), &probe_side.stats),
    "the normalized trial against whole-cohort probe statistics",
  );
  let flipped = ok(
    enrolled_untagged.normalize(trial.raw(), &truncated),
    "the normalized trial against truncated probe statistics",
  );

  assert!(
    (honest - 1.640_951_863_461_705).abs() < 1e-9,
    "the whole-cohort normalization moved: {honest}"
  );
  assert!(
    (flipped - 4.454_545_979_780_686).abs() < 1e-9,
    "the candidate-truncated normalization moved: {flipped}"
  );
  assert!(
    honest < WORKED_CASE_THRESHOLD && flipped > WORKED_CASE_THRESHOLD,
    "dropping the candidate from the probe's cohort must be the difference \
     between {honest} and {flipped}, which straddle a threshold of \
     {WORKED_CASE_THRESHOLD}"
  );
}

// ── what #123 asked for ──────────────────────────────────────────────────

/// The claim from #123, made falsifiable: raw cosine admits NO single
/// threshold that separates same-speaker trials from different-speaker ones
/// across two differently-placed speakers, and AS-Norm does.
///
/// Every side here comes from the one held-out cohort — the recommended
/// arrangement, and the only one under which the two z-scores AS-Norm averages
/// are commensurable.
#[test]
fn as_norm_separates_two_differently_placed_speakers_where_no_raw_threshold_can() {
  let scoring = Scoring::Cosine;
  let cohort = held_out(scoring);

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
    raw_genuine_b.raw() < raw_impostor.raw(),
    "the fixture must reproduce #123's problem: a genuine trial for the \
     isolated speaker ({}) has to score BELOW an impostor trial for the \
     crowded one ({})",
    raw_genuine_b.raw(),
    raw_impostor.raw()
  );

  // The same trials, normalized. A', B' and the impostor partner all sit
  // OUTSIDE the held-out cohort, so nothing needs excluding and no side
  // depends on the other end of its trial.
  let side_a = side(&cohort, &a);
  let side_a2 = side(&cohort, &a2);
  let side_b = side(&cohort, &b);
  let side_b2 = side(&cohort, &b2);
  let side_impostor = side(&cohort, &impostor);

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
  let cohort = held_out(scoring);
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");

  let raw = ok(b.score(&b2), "B vs B'");
  let side_b = side(&cohort, &b);
  let side_b2 = side(&cohort, &b2);
  let normalized = ok(as_norm(raw, &side_b, &side_b2), "B vs B' normalized");
  assert!(
    normalized.is_finite(),
    "a PLDA-space normalization must produce a usable number, got {normalized}"
  );

  // The statistics have to come out of the PLDA space, not out of a
  // `PldaCosine` that quietly degraded to `Cosine` somewhere in `prepare`.
  let cosine_b = ok(
    Scoring::Cosine.prepare(&speaker_b()),
    "prepare B for cosine",
  );
  let cosine_side = side(&held_out(Scoring::Cosine), &cosine_b);
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

/// The wrapper adds a tag check and NOTHING else: on values that agree, its
/// answer is `diaric`'s own arithmetic, bit for bit. A second implementation
/// of eq. (7) hiding in here would be a second set of `diaric`'s cancellation
/// bugs.
#[test]
fn a_matching_normalization_is_diarics_own_arithmetic_bit_for_bit() {
  let scoring = Scoring::Cosine;
  let cohort = held_out(scoring);
  let options = AsNormOptions::new();
  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");

  let trial = ok(a.score(&b), "A vs B");
  let got = ok(
    as_norm(trial, &side(&cohort, &a), &side(&cohort, &b)),
    "this module's as_norm",
  );

  let scores = |profile: &VoiceProfile| {
    crowd()
      .iter()
      .skip(HELD_OUT_FROM)
      .map(|v| {
        let entry = ok(scoring.prepare(v), "prepare a held-out impostor");
        ok(profile.score(&entry), "a cohort score").raw()
      })
      .collect::<Vec<_>>()
  };
  let expected = ok(
    diaric::score_norm::as_norm(
      trial.raw(),
      &ok(CohortStats::from_scores(scores(&a), &options), "A's side"),
      &ok(CohortStats::from_scores(scores(&b), &options), "B's side"),
    ),
    "diaric's own as_norm",
  );

  assert_eq!(
    got.to_bits(),
    expected.to_bits(),
    "the tagged door must return diaric's number unchanged: {got} vs {expected}"
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

/// FINDING 2. The check at `score()` covers the trial and nothing else: the
/// final combination reads a number and two statistics and no profile at all,
/// so without a tag on each of the three a `PldaCosine` trial score is
/// silently calibrated by `Cosine` cohort statistics and comes back finite and
/// plausible.
#[test]
fn a_trial_score_cannot_be_normalized_by_another_metrics_statistics() {
  let cosine_cohort = held_out(Scoring::Cosine);
  let cosine_b = ok(Scoring::Cosine.prepare(&speaker_b()), "prepare B");
  let cosine_b2 = ok(
    Scoring::Cosine.prepare(&speaker_b_again()),
    "prepare B again",
  );
  let cosine_side = side(&cosine_cohort, &cosine_b);
  let cosine_side2 = side(&cosine_cohort, &cosine_b2);

  let plda_b = ok(Scoring::PldaCosine.prepare(&speaker_b()), "prepare B");
  let plda_b2 = ok(
    Scoring::PldaCosine.prepare(&speaker_b_again()),
    "prepare B again",
  );
  let plda_trial = ok(plda_b.score(&plda_b2), "a PldaCosine trial score");

  match as_norm(plda_trial, &cosine_side, &cosine_side2) {
    Err(CalibrateError::NormalizationMismatch(m)) => {
      assert_eq!(m.trial(), Scoring::PldaCosine);
      assert_eq!(m.enrolled(), Scoring::Cosine);
      assert_eq!(m.probe(), Scoring::Cosine);
    }
    other => panic!(
      "a PldaCosine trial score calibrated by Cosine cohort statistics must be \
       refused, got {other:?}"
    ),
  }

  // And one stale side among otherwise matching values, which is the shape a
  // cache of per-speaker statistics actually produces.
  let plda_side = side(&held_out(Scoring::PldaCosine), &plda_b);
  match as_norm(plda_trial, &plda_side, &cosine_side2) {
    Err(CalibrateError::NormalizationMismatch(m)) => {
      assert_eq!(m.trial(), Scoring::PldaCosine);
      assert_eq!(m.enrolled(), Scoring::PldaCosine);
      assert_eq!(m.probe(), Scoring::Cosine);
    }
    other => panic!("one foreign side must be refused, got {other:?}"),
  }
}

/// A cohort holding a foreign-source entry poisons a mean silently unless the
/// door refuses it, so it refuses it — through BOTH entrypoints. This is also
/// what makes a [`SideStats`]'s own tag sound: a surviving statistic can only
/// have been computed over entries that all matched its side.
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

  let excluding = enrolled_stats(&cohort, Enrolled::new(&Speaker::A, &a), &options);
  assert!(
    matches!(excluding, Err(CalibrateError::ScoringMismatch(_))),
    "a mixed cohort must be refused by the enrolled door, got {excluding:?}"
  );

  let mut mixed: Vec<VoiceProfile> = crowd()
    .iter()
    .map(|v| ok(Scoring::Cosine.prepare(v), "prepare an impostor"))
    .collect();
  mixed.push(ok(
    Scoring::PldaCosine.prepare(&speaker_b()),
    "prepare a foreign-source entry",
  ));
  let held = held_out_stats(&HeldOutCohort::assuming_disjoint(mixed), &a, &options);
  assert!(
    matches!(held, Err(CalibrateError::ScoringMismatch(_))),
    "a mixed cohort must be refused by the held-out door, got {held:?}"
  );
}

/// The refusal survives the closure boundary: `diaric` takes an INFALLIBLE
/// scoring function, so a mismatch has to be carried out by hand. This pins
/// that the carried error is the one reported, not the `NonFiniteScore` the
/// poison value would otherwise produce.
#[test]
fn a_mismatch_inside_the_cohort_reports_the_mismatch_not_the_poison_value() {
  let cohort = held_out(Scoring::PldaCosine);
  let a = ok(Scoring::Cosine.prepare(&speaker_a()), "prepare A");
  let refused = held_out_stats(&cohort, &a, &AsNormOptions::new());
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
    enrolled_stats(
      &cohort,
      Enrolled::new(&Speaker::B, &b),
      &AsNormOptions::new(),
    ),
    "statistics at the default top_n",
  );
  let narrow_side = ok(
    enrolled_stats(&cohort, Enrolled::new(&Speaker::B, &b), &narrow),
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
  let narrow_held_out = ok(
    held_out_stats(&held_out(scoring), &b, &narrow),
    "held-out statistics at top_n = 4",
  );
  assert_eq!(narrow_held_out.selected(), 4);
}

/// `diaric`'s minimum-usable-cohort floor reaches through the wrapper, and
/// arrives as its OWN refusal — a cohort that is merely absent must not be
/// reported as a degenerate one.
#[test]
fn a_cohort_below_the_minimum_usable_size_is_refused_as_too_small() {
  let scoring = Scoring::Cosine;
  let cohort = HeldOutCohort::assuming_disjoint(vec![ok(
    scoring.prepare(&crowd()[0]),
    "prepare an impostor",
  )]);
  assert!(cohort.len() < super::MIN_COHORT_SCORES);
  assert!(!cohort.is_empty());

  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let refused = held_out_stats(&cohort, &b, &AsNormOptions::new());
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
/// The tagged values travel with them. Pinned at compile time so a future field
/// type cannot regress the auto-derive silently.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<VoiceProfile>();
  assert_send_sync::<Scoring>();
  assert_send_sync::<TrialScore>();
  assert_send_sync::<SideStats>();
  assert_send_sync::<HeldOutCohort>();
  assert_send_sync::<LibraryCohort<u32>>();
  assert_send_sync::<CohortId>();
  assert_send_sync::<Enrolled<'static, u32>>();
  assert_send_sync::<CalibrateError>();
};

// ── one cohort per trial ─────────────────────────────────────────────────

/// codex's round-2 case, reproduced to its digits. `P` is the probe; `A` and
/// `B` are the two candidates the trial has to rank; `X` is the third library
/// member that keeps each enrolled side above [`MIN_COHORT_SCORES`] once its
/// own entry is dropped. `X` is placed so that `A·X = 0.8` and `B·X = 0`,
/// which is what pulls the two enrolment statistics apart.
fn ranking_case() -> [Vec<f32>; 4] {
  [
    row(&[(0, 1.0)]),                                             // P
    row(&[(0, 0.8), (1, 0.6)]),                                   // A
    row(&[(0, 0.7), (2, 0.51f32.sqrt())]),                        // B
    row(&[(0, 0.341_056_75), (1, 0.878_591), (2, -0.334_302_5)]), // X
  ]
}

/// The axis-aligned impostor cohort of the case above, over `axes`.
fn axis_cohort(scoring: Scoring, axes: &[(usize, f32)]) -> HeldOutCohort {
  HeldOutCohort::assuming_disjoint(
    axes
      .iter()
      .map(|&(axis, sign)| {
        ok(
          scoring.prepare(&row(&[(axis, sign)])),
          "prepare an axis impostor",
        )
      })
      .collect(),
  )
}

/// One run of the ranking case: an impostor cohort, and the numbers the two
/// candidates normalize to with both sides over it and with only the probe's
/// side over it.
struct Case<'a> {
  /// The cohort's members, as `(axis, sign)` unit vectors.
  axes: &'a [(usize, f32)],
  /// A and B normalized with BOTH sides over `axes`.
  shared: [f64; 2],
  /// A and B normalized with the enrolment side over the library cohort
  /// instead — the pairing the door refuses.
  mixed: [f64; 2],
}

/// ROUND 2, FINDING 2. Two sides taken over DIFFERENT cohorts are refused —
/// and the refusal is not pedantry: on this fixture the mixed pairing does not
/// merely shift the number, it REVERSES which candidate ranks first.
///
/// Every value in both arrangements is tagged [`Scoring::Cosine`], every
/// profile is valid, and every deviation clears [`DEFAULT_MIN_DEVIATION`], so
/// nothing already in this module could see it: the metric was never what
/// differed. The mixed numbers are computed here through `diaric`'s own
/// `normalize`, because the tagged door will not produce them any more.
///
/// Run over two cohorts. The first is codex's as reported, `{±e0, ±e1, ±e2}`,
/// which holds a member on the probe's own axis — so `assuming_disjoint` is a
/// geometric falsehood there, stated only to keep the reported digits
/// checkable. The second drops that member, which makes the fixture's
/// assertion true and shows the reversal does not depend on it.
#[test]
fn two_sides_taken_over_different_cohorts_are_refused() {
  let scoring = Scoring::Cosine;
  let options = AsNormOptions::new();
  let [probe_row, a_row, b_row, x_row] = ranking_case();

  let probe = ok(scoring.prepare(&probe_row), "prepare the probe");
  let a = ok(scoring.prepare(&a_row), "prepare A");
  let b = ok(scoring.prepare(&b_row), "prepare B");
  let x = ok(scoring.prepare(&x_row), "prepare X");

  let trial_a = ok(a.score(&probe), "A against the probe");
  let trial_b = ok(b.score(&probe), "B against the probe");
  assert!(
    trial_a.raw() > trial_b.raw(),
    "the fixture must start with A ahead on the raw score: A {} vs B {}",
    trial_a.raw(),
    trial_b.raw()
  );

  // The library-sampled cohort: it holds A and B themselves, so each enrolled
  // side drops its own entries and keeps the other two.
  let mut library: LibraryCohort<Speaker> = LibraryCohort::new();
  library.push(Speaker::A, a);
  library.push(Speaker::B, b);
  library.push(Speaker::Impostor(0), x);
  let a_enrolled = ok(
    enrolled_stats(&library, Enrolled::new(&Speaker::A, &a), &options),
    "A's side over the library cohort",
  );
  let b_enrolled = ok(
    enrolled_stats(&library, Enrolled::new(&Speaker::B, &b), &options),
    "B's side over the library cohort",
  );
  assert_eq!(a_enrolled.considered(), 2);
  assert_eq!(b_enrolled.considered(), 2);

  let codex_axes = [
    (0, 1.0),
    (0, -1.0),
    (1, 1.0),
    (1, -1.0),
    (2, 1.0),
    (2, -1.0),
  ];
  let cases = [
    Case {
      axes: &codex_axes,
      shared: [1.385_640_6, 1.212_435_6],
      mixed: [1.192_820_1, 1.356_217_7],
    },
    Case {
      // The same case with the probe's own axis dropped, so
      // `assuming_disjoint` is true of the fixture as well as asserted.
      axes: &codex_axes[1..],
      shared: [2.216_987_6, 1.915_345_4],
      mixed: [1.75, 1.875],
    },
  ];

  for Case {
    axes,
    shared: [shared_a, shared_b],
    mixed: [mixed_a, mixed_b],
  } in cases
  {
    let held_out = axis_cohort(scoring, axes);
    let probe_side = ok(
      held_out_stats(&held_out, &probe, &options),
      "the probe's side over the held-out cohort",
    );

    // The arrangement that IS commensurable: both sides over the one cohort.
    let a_shared = ok(
      as_norm(
        trial_a,
        &ok(held_out_stats(&held_out, &a, &options), "A over the cohort"),
        &probe_side,
      ),
      "A normalized, both sides over one cohort",
    );
    let b_shared = ok(
      as_norm(
        trial_b,
        &ok(held_out_stats(&held_out, &b, &options), "B over the cohort"),
        &probe_side,
      ),
      "B normalized, both sides over one cohort",
    );
    assert!(
      (a_shared - shared_a).abs() < 1e-6,
      "A over one cohort: {a_shared}"
    );
    assert!(
      (b_shared - shared_b).abs() < 1e-6,
      "B over one cohort: {b_shared}"
    );
    assert!(
      a_shared > b_shared,
      "one cohort for both sides must keep A ahead, as the raw score has it: \
       A {a_shared} vs B {b_shared}"
    );

    // The arrangement that is not, and what it does to the order. `diaric`'s
    // own arithmetic, because the door refuses to run it.
    let a_mixed = ok(
      a_enrolled.stats.normalize(trial_a.raw(), &probe_side.stats),
      "A normalized across two cohorts",
    );
    let b_mixed = ok(
      b_enrolled.stats.normalize(trial_b.raw(), &probe_side.stats),
      "B normalized across two cohorts",
    );
    assert!(
      (a_mixed - mixed_a).abs() < 1e-6,
      "A across two cohorts: {a_mixed}"
    );
    assert!(
      (b_mixed - mixed_b).abs() < 1e-6,
      "B across two cohorts: {b_mixed}"
    );
    assert!(
      b_mixed > a_mixed,
      "the whole point: two cohorts put B first, against both the raw order \
       and the one-cohort order — A {a_mixed} vs B {b_mixed}"
    );

    // Which is why the door will not do it.
    for (side, trial) in [(&a_enrolled, trial_a), (&b_enrolled, trial_b)] {
      match as_norm(trial, side, &probe_side) {
        Err(CalibrateError::CohortMismatch(m)) => {
          assert_eq!(m.enrolled(), library.id());
          assert_eq!(m.probe(), held_out.id());
          assert_ne!(m.enrolled(), m.probe());
        }
        other => panic!("two cohorts in one trial must be refused, got {other:?}"),
      }
    }
  }
}

/// The identity is the COHORT's, not the variable's: a side taken before a
/// cohort grows cannot be averaged against one taken after. That is the shape
/// a per-speaker cache produces when the library gains a member, and it is the
/// one case where two sides really did come from "the same" cohort by name.
#[test]
fn growing_a_cohort_makes_it_a_different_cohort() {
  let scoring = Scoring::Cosine;
  let options = AsNormOptions::new();
  let mut cohort = library(scoring);

  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let before = ok(
    enrolled_stats(&cohort, Enrolled::new(&Speaker::A, &a), &options),
    "A's side before the cohort grew",
  );
  assert_eq!(before.cohort(), cohort.id());

  cohort.push(
    Speaker::Impostor(999),
    ok(
      scoring.prepare(&speaker_b_again()),
      "prepare a new impostor",
    ),
  );
  let after = ok(
    enrolled_stats(&cohort, Enrolled::new(&Speaker::B, &b), &options),
    "B's side after the cohort grew",
  );
  assert_ne!(
    before.cohort(),
    after.cohort(),
    "pushing a member must mint a new identity"
  );

  let trial = ok(a.score(&b), "A vs B");
  assert!(
    matches!(
      as_norm(trial, &before, &after),
      Err(CalibrateError::CohortMismatch(_))
    ),
    "a side from before the push and one from after are two populations"
  );

  // A clone is the same population, so it keeps the identity and still pairs.
  let clone = cohort.clone();
  let cloned_side = ok(
    enrolled_stats(&clone, Enrolled::new(&Speaker::A, &a), &options),
    "A's side over a clone of the grown cohort",
  );
  assert_eq!(cloned_side.cohort(), after.cohort());
  assert!(
    ok(
      as_norm(trial, &cloned_side, &after),
      "cloned cohort normalizes"
    )
    .is_finite()
  );
}

/// Two cohorts built from the very same profiles are two identities, because
/// nothing in this crate can tell they are one population. Stated as a test
/// because it is the conservative half of the trade — a caller who rebuilds an
/// identical cohort gets a refusal, not a silent pass — and a future "compare
/// the members instead" would be a behaviour change, not an optimisation.
#[test]
fn two_cohorts_of_identical_profiles_are_two_identities() {
  let scoring = Scoring::Cosine;
  let one = held_out(scoring);
  let two = held_out(scoring);
  assert_eq!(one, one.clone(), "a clone is the same cohort");
  assert_ne!(one.id(), two.id());
  assert_ne!(
    one, two,
    "equality is identity, so two separately assembled cohorts differ even \
     when their members do not"
  );

  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");
  let trial = ok(b.score(&b2), "B vs B'");
  assert!(
    matches!(
      as_norm(trial, &side(&one, &b), &side(&two, &b2)),
      Err(CalibrateError::CohortMismatch(_))
    ),
    "two identically-built cohorts are still two cohorts"
  );
  assert!(
    ok(
      as_norm(trial, &side(&one, &b), &side(&one, &b2)),
      "one cohort for both sides"
    )
    .is_finite()
  );
}

/// The metric check runs FIRST. A caller who has mixed both needs to hear
/// about the metric: a `PldaCosine` side and a `Cosine` side necessarily came
/// from two cohorts as well, so reporting the cohort would name the symptom.
#[test]
fn a_metric_mismatch_is_reported_ahead_of_the_cohort_mismatch() {
  let cosine_cohort = held_out(Scoring::Cosine);
  let plda_cohort = held_out(Scoring::PldaCosine);
  assert_ne!(cosine_cohort.id(), plda_cohort.id());

  let plda_b = ok(Scoring::PldaCosine.prepare(&speaker_b()), "prepare B");
  let plda_b2 = ok(
    Scoring::PldaCosine.prepare(&speaker_b_again()),
    "prepare B again",
  );
  let trial = ok(plda_b.score(&plda_b2), "a PldaCosine trial");

  let plda_side = side(&plda_cohort, &plda_b);
  let cosine_side = side(
    &cosine_cohort,
    &ok(
      Scoring::Cosine.prepare(&speaker_b_again()),
      "prepare B again for cosine",
    ),
  );
  assert!(
    matches!(
      as_norm(trial, &plda_side, &cosine_side),
      Err(CalibrateError::NormalizationMismatch(_))
    ),
    "the metric mismatch must win over the cohort one"
  );
}
