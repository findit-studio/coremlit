//! Unit tests for [`super`] — the AS-Norm calibration door.
//!
//! Every fixture here is CONSTRUCTED, not measured: unit vectors placed by
//! hand in a handful of coordinate axes of the 256-d space, chosen so the
//! geometry each test claims is arithmetic rather than a property of any
//! recording. Nothing in this file asserts anything about WeSpeaker's real
//! output distribution — that is `tests/speaker/`'s job, and it needs models.

use std::collections::HashMap;

use diaric::{
  embed::{Embedding, cosine_similarity},
  // `CohortStats` is `diaric`'s own unbound statistic, and it is deliberately
  // NOT re-exported by `super` (module docs, "What leaves this module"). These
  // tests reach it the way a caller now has to — through `diaric` itself —
  // which is also what lets them build the arrangements the doors refuse, in
  // order to pin WHY they refuse them.
  score_norm::CohortStats,
};

use crate::audio::speaker::{
  calibrate::{
    AsNormOptions, CalibratedTrial, Calibration, CalibrationId, Enrolled, HeldOutCohort,
    LibraryCohort, LibraryCohortBuilder, Scoring, SpeakerToken, TrialSide, VoiceProfile,
  },
  embed::EMBEDDING_DIM,
  error::{CalibrateError, CohortSelection, ScoreNormRefusal},
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

/// The CALLER's library keys for the shared fixture. `Impostor` carries the
/// crowd index so a single impostor can be named as the other side of a trial.
///
/// Nothing on the surface under test ever sees one of these: a cohort names its
/// speakers with tokens it minted, and the map from a key to a token is the
/// caller's own — which is what [`Roster`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Speaker {
  A,
  B,
  Impostor(usize),
}

/// The caller-side half of an identity: their key, and the token the cohort
/// minted for it. A real caller's library holds exactly this.
type Roster = HashMap<Speaker, SpeakerToken>;

/// Push one profile under `speaker`, minting that speaker's token the first
/// time the roster is asked for it.
fn enrol(
  cohort: &mut LibraryCohortBuilder,
  roster: &mut Roster,
  speaker: Speaker,
  profile: VoiceProfile,
) -> SpeakerToken {
  let token = *roster.entry(speaker).or_insert_with(|| cohort.speaker());
  cohort.push(token, profile);
  token
}

/// The library-sampled cohort: the crowd, plus speakers A and B themselves.
/// #123's cohort is "sampled from the library itself", so A and B ARE in their
/// own cohorts — which is the whole reason the enrolled door excludes by
/// identity.
fn library(scoring: Scoring) -> (LibraryCohortBuilder, Roster) {
  let mut cohort = LibraryCohortBuilder::new();
  let mut roster = Roster::new();
  for (i, v) in crowd().iter().enumerate() {
    enrol(
      &mut cohort,
      &mut roster,
      Speaker::Impostor(i),
      ok(scoring.prepare(v), "prepare an impostor"),
    );
  }
  enrol(
    &mut cohort,
    &mut roster,
    Speaker::A,
    ok(scoring.prepare(&speaker_a()), "prepare A"),
  );
  enrol(
    &mut cohort,
    &mut roster,
    Speaker::B,
    ok(scoring.prepare(&speaker_b()), "prepare B"),
  );
  (cohort, roster)
}

/// [`library`] under the default options, with one speaker's token picked out
/// of the roster — the shape an enrolled side takes.
fn library_calibration_for(
  scoring: Scoring,
  speaker: Speaker,
) -> (Calibration<LibraryCohort>, SpeakerToken) {
  let (cohort, roster) = library(scoring);
  let token = roster[&speaker];
  (Calibration::new(cohort, AsNormOptions::new()), token)
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

/// [`held_out`] fixed under the default options — the shape a trial with an
/// unidentified probe in it has to take, since a probe's side can only be a
/// held-out one and both sides must come from the one calibration.
fn held_out_calibration(scoring: Scoring) -> Calibration<HeldOutCohort> {
  Calibration::new(held_out(scoring), AsNormOptions::new())
}

/// One side of a trial over a held-out calibration.
fn side(calibration: &Calibration<HeldOutCohort>, profile: &VoiceProfile) -> TrialSide {
  ok(calibration.side(profile), "a trial side")
}

/// One trial of a held-out calibration, between two profiles.
fn trial_of(
  calibration: &Calibration<HeldOutCohort>,
  a: &VoiceProfile,
  b: &VoiceProfile,
) -> CalibratedTrial {
  ok(
    calibration.trial(&side(calibration, a), &side(calibration, b)),
    "a calibrated trial",
  )
}

/// The worked case behind the shape this module answers: a three-member cohort
/// in which a probe scores `[0, 0.8, 0.2]`. Returns the probe row and the
/// three member rows.
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

  // Through the public door as well as the private one, since the raw score a
  // caller can actually see is the trial's.
  let got = ok(profile_a.score(&profile_b), "score A against B");
  let published = trial_of(
    &held_out_calibration(Scoring::Cosine),
    &profile_a,
    &profile_b,
  )
  .raw();
  assert_eq!(
    got.to_bits(),
    expected.to_bits(),
    "the cosine score source must BE diaric::embed::cosine_similarity, not agree with it"
  );
  assert_eq!(
    published.to_bits(),
    expected.to_bits(),
    "a calibrated trial's raw score must be that same number"
  );
}

/// A self-match is the top of the cohort — the fact that makes
/// self-contamination guaranteed rather than merely possible, and therefore
/// the fact [`HeldOutCohort::assuming_disjoint`] exists to keep out.
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
  let cosine = trial_of(
    &held_out_calibration(Scoring::Cosine),
    &ok(Scoring::Cosine.prepare(&a), "prepare A for cosine"),
    &ok(Scoring::Cosine.prepare(&b), "prepare B for cosine"),
  )
  .raw();
  let plda = trial_of(
    &held_out_calibration(Scoring::PldaCosine),
    &ok(Scoring::PldaCosine.prepare(&a), "prepare A for plda"),
    &ok(Scoring::PldaCosine.prepare(&b), "prepare B for plda"),
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

/// The source has to SURVIVE each step: both kinds of side and the calibrated
/// trial itself report the source they were computed in, and the trial takes
/// its from the profiles it scored rather than from a tag handed in.
#[test]
fn a_side_and_a_calibrated_trial_carry_the_source_they_were_computed_in() {
  for scoring in [Scoring::Cosine, Scoring::PldaCosine] {
    let a = ok(scoring.prepare(&speaker_a()), "prepare A");
    let b = ok(scoring.prepare(&speaker_b()), "prepare B");
    let calibration = held_out_calibration(scoring);

    assert_eq!(
      side(&calibration, &a).scoring(),
      scoring,
      "a held-out side must report the source it was computed in"
    );
    assert_eq!(
      trial_of(&calibration, &a, &b).scoring(),
      scoring,
      "a calibrated trial must report the source both its profiles carry"
    );
    let (library_cal, a_token) = library_calibration_for(scoring, Speaker::A);
    assert_eq!(
      ok(
        library_cal.enrolled_side(Enrolled::new(a_token, &a)),
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
/// mean up, so every z-score taken against it is smaller — and the
/// contaminated side still looks perfectly healthy.
///
/// The mean itself is not published any more (module docs, "What leaves this
/// module"), so the effect is read where a caller would actually meet it: on
/// the calibrated number, for the same trial, under two calibrations.
#[test]
fn keeping_a_speakers_own_entry_lowers_every_score_taken_against_it() {
  let scoring = Scoring::Cosine;
  let options = AsNormOptions::new();
  let (cohort, roster) = library(scoring);
  let b_token = roster[&Speaker::B];
  let members = cohort.len();
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");

  let clean = Calibration::new(cohort.clone(), options);
  let clean_b = ok(
    clean.enrolled_side(Enrolled::new(b_token, &b)),
    "B's side, own entries excluded",
  );
  let clean_b2 = ok(
    clean.enrolled_side(Enrolled::new(b_token, &b2)),
    "B''s side, own entries excluded",
  );

  // The same profiles, wrongly asserted to be held out — B's own entry
  // included. Only a test would state that; it is the failure the assertion
  // exists to keep out.
  let contaminated = Calibration::new(
    HeldOutCohort::assuming_disjoint(
      library(scoring)
        .0
        .entries
        .entries()
        .iter()
        .map(|e| *e.item())
        .collect(),
    ),
    options,
  );
  let dirty_b = side(&contaminated, &b);
  let dirty_b2 = side(&contaminated, &b2);

  assert_eq!(
    dirty_b.considered(),
    members,
    "the held-out door scores every member"
  );
  assert_eq!(
    clean_b.considered(),
    members - 1,
    "the enrolled door drops B's one library entry"
  );

  let honest = ok(clean.trial(&clean_b, &clean_b2), "B vs B', B excluded");
  let poisoned = ok(
    contaminated.trial(&dirty_b, &dirty_b2),
    "B vs B', B still in the cohort",
  );
  assert_eq!(
    honest.raw().to_bits(),
    poisoned.raw().to_bits(),
    "the same two profiles, so the raw score must be identical and only the \
     calibration can differ"
  );
  assert!(
    poisoned.calibrated() < honest.calibrated(),
    "B's own entry scores 1.0 against B, so keeping it must raise the cohort \
     mean and shrink the z-score: honest {} vs contaminated {}",
    honest.calibrated(),
    poisoned.calibrated()
  );
}

/// Exclusion is by IDENTITY, so it removes ALL of a speaker's entries, not
/// just the one that happens to score highest.
#[test]
fn exclusion_drops_every_entry_a_speaker_owns_not_only_the_self_match() {
  let scoring = Scoring::Cosine;
  let (mut cohort, mut roster) = library(scoring);
  // A second library entry for B — a different recording, same identity, so it
  // goes under the token B already has.
  let b_token = enrol(
    &mut cohort,
    &mut roster,
    Speaker::B,
    ok(scoring.prepare(&speaker_b_again()), "prepare B again"),
  );
  let members = cohort.len();

  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let excluding = ok(
    Calibration::new(cohort, AsNormOptions::new()).enrolled_side(Enrolled::new(b_token, &b)),
    "statistics excluding B",
  );
  assert_eq!(
    excluding.considered(),
    members - 2,
    "both of B's entries must go, not just the exact self-match"
  );
}

/// The SCOPE of the exclusion, pinned as arithmetic: an enrolled side drops
/// its own speaker and nobody else — least of all the partner it is about to
/// be scored against, which is what keeps the side a property of one speaker
/// plus the calibration, and therefore reusable.
#[test]
fn an_enrolled_side_drops_only_its_own_speaker_never_the_partner() {
  let scoring = Scoring::Cosine;
  let (_, member_rows) = worked_case();
  let options = AsNormOptions::new();

  let profiles: Vec<VoiceProfile> = member_rows
    .iter()
    .map(|v| ok(scoring.prepare(v), "prepare a cohort member"))
    .collect();
  let mut cohort = LibraryCohortBuilder::new();
  let mut roster = Roster::new();
  for (i, p) in profiles.iter().enumerate() {
    enrol(&mut cohort, &mut roster, Speaker::Impostor(i), *p);
  }

  let enrolled = profiles[1];
  let token = roster[&Speaker::Impostor(1)];
  let got = ok(
    Calibration::new(cohort, options).enrolled_side(Enrolled::new(token, &enrolled)),
    "the enrolled side",
  );

  // Exactly the scores against the other two members, member 2 included —
  // member 2 is the trial partner in the probe test below.
  let by_hand = ok(
    CohortStats::from_scores(
      [
        ok(enrolled.score(&profiles[0]), "score against member 0"),
        ok(enrolled.score(&profiles[2]), "score against member 2"),
      ],
      &options,
    ),
    "the statistics written out by hand",
  );

  assert_eq!(got.considered(), 2, "only member 1's own entry may go");
  // `got.stats` is the private field these tests can see. A caller cannot: the
  // mean and the deviation are exactly what this surface no longer publishes.
  assert_eq!(
    got.stats.mean().to_bits(),
    by_hand.mean().to_bits(),
    "an enrolled side must be exactly the scores against every OTHER speaker: \
     got {} against {}",
    got.stats.mean(),
    by_hand.mean()
  );
  assert_eq!(
    got.stats.deviation().to_bits(),
    by_hand.deviation().to_bits()
  );
}

/// A cohort holding nothing but the excluded speaker is a refusal, not an
/// empty-but-usable side.
#[test]
fn a_cohort_that_is_entirely_the_excluded_speaker_is_refused() {
  let scoring = Scoring::Cosine;
  let mut cohort = LibraryCohortBuilder::new();
  let mut roster = Roster::new();
  let b_token = enrol(
    &mut cohort,
    &mut roster,
    Speaker::B,
    ok(scoring.prepare(&speaker_b()), "prepare B"),
  );
  enrol(
    &mut cohort,
    &mut roster,
    Speaker::B,
    ok(scoring.prepare(&speaker_b_again()), "prepare B again"),
  );
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");

  let refused =
    Calibration::new(cohort, AsNormOptions::new()).enrolled_side(Enrolled::new(b_token, &b));
  assert!(
    matches!(refused, Err(CalibrateError::ScoreNorm(_))),
    "self-exclusion emptying the cohort must refuse, got {refused:?}"
  );
}

/// No duplicate of a cohort can re-decide what a token names.
///
/// This is where three rounds of review ended, and why the caller's key type is
/// gone rather than sealed once more. [`Eq`] does not forbid interior
/// mutability, so an `Rc<Cell<u64>>` key was `Eq` and still writable through a
/// handle the caller kept; rewriting it so one speaker's key equalled another's
/// made a lookup for the SECOND speaker hand back the FIRST one's token, and
/// `enrolled_side` then honoured it — dropping the wrong speaker's entries and
/// leaving the subject's own in place, on a finite, plausible number.
///
/// Each round sealed the road it had found and left the next: between two
/// derivations; through a `Clone` of the calibration's own cohort, which is an
/// OWNED resolver and so walks around the `&mut self` on the mint; and with no
/// calibration in sight at all, before any freeze. The road was never the
/// defect — resolving a token from caller-owned state was.
///
/// There is nothing to resolve now. `speaker` takes no argument, and a frozen
/// cohort holds token membership and token-keyed entries and nothing else. So
/// every duplicate below answers exactly what the original answers, the only
/// identity a duplicate can produce is a NEW one the original refuses, and what
/// a duplicate is given stays in the duplicate.
#[test]
fn no_duplicate_of_a_cohort_can_re_decide_what_a_token_names() {
  let scoring = Scoring::Cosine;
  let (cohort, roster) = library(scoring);
  let a_token = roster[&Speaker::A];
  let b_token = roster[&Speaker::B];
  assert_ne!(a_token, b_token, "two speakers must be two identities");
  let members = cohort.len();

  // A copy of the cohort kept aside BEFORE the freeze — the road the last
  // round's fix would not have reached.
  let mut spare = cohort.clone();

  let calibration = Calibration::new(cohort, AsNormOptions::new());
  let a = ok(scoring.prepare(&speaker_a_again()), "prepare A again");
  let first = ok(
    calibration.enrolled_side(Enrolled::new(a_token, &a)),
    "A's side",
  );
  assert_eq!(
    first.considered(),
    members - 1,
    "A's own entry must be dropped"
  );

  // Every duplicate this surface permits, each asked for the same side.
  let cloned_calibration = calibration.clone();
  let over_cloned_cohort = Calibration::new(spare.clone(), AsNormOptions::new());
  for (what, side) in [
    (
      "a clone of the calibration",
      ok(
        cloned_calibration.enrolled_side(Enrolled::new(a_token, &a)),
        "A's side under a cloned calibration",
      ),
    ),
    (
      "a clone of the cohort, freshly calibrated",
      ok(
        over_cloned_cohort.enrolled_side(Enrolled::new(a_token, &a)),
        "A's side under a cloned cohort",
      ),
    ),
  ] {
    assert_eq!(
      side.considered(),
      first.considered(),
      "{what} must consider the same cohort"
    );
    assert_eq!(
      side.stats.mean().to_bits(),
      first.stats.mean().to_bits(),
      "{what} must name the same exclusion set, and so the same statistic"
    );
    assert_eq!(
      side.stats.deviation().to_bits(),
      first.stats.deviation().to_bits(),
      "{what} must name the same exclusion set, and so the same statistic"
    );
  }

  // The one identity a duplicate can produce is a NEW speaker, never one of the
  // original's — so a duplicate cannot manufacture a token the original honours.
  let stranger = spare.speaker();
  assert_ne!(
    stranger, a_token,
    "a mint reads nothing, so it repeats nothing"
  );
  assert_ne!(stranger, b_token);
  let refused = calibration.enrolled_side(Enrolled::new(stranger, &a));
  assert!(
    matches!(refused, Err(CalibrateError::ForeignSpeaker)),
    "an identity minted by a duplicate is not this cohort's, got {refused:?}"
  );

  // And what a duplicate is given stays in the duplicate: the frozen cohort has
  // no mutator, so nothing can grow under a side already taken.
  spare.push(
    a_token,
    ok(scoring.prepare(&speaker_a_again()), "prepare A again"),
  );
  let after = ok(
    calibration.enrolled_side(Enrolled::new(a_token, &a)),
    "A's side, taken again",
  );
  assert_eq!(
    after.considered(),
    first.considered(),
    "a push into a duplicate must not reach the frozen cohort"
  );
  assert_eq!(
    after.stats.mean().to_bits(),
    first.stats.mean().to_bits(),
    "one calibration and one token must be one statistic, for the life of both"
  );
  assert_eq!(
    after.stats.deviation().to_bits(),
    first.stats.deviation().to_bits()
  );
}

/// A token names a speaker in the cohort that MINTED it and in no other, so a
/// token from somewhere else is a refusal rather than an exclusion of nothing.
///
/// Excluding nothing is the self-contamination the enrolled door exists to
/// prevent, and it looks perfectly healthy: a self-match is the largest score
/// a profile can obtain, so top-N selection is guaranteed to keep it.
#[test]
fn a_token_from_another_cohort_is_refused_rather_than_excluding_nothing() {
  let scoring = Scoring::Cosine;
  let (_, foreign_roster) = library(scoring);
  let (own, own_roster) = library(scoring);
  let foreign_token = foreign_roster[&Speaker::A];
  let own_token = own_roster[&Speaker::A];
  assert_ne!(
    foreign_token, own_token,
    "two cohorts mint two speakers for one library record: nothing here can \
     tell they are one population"
  );

  let calibration = Calibration::new(own, AsNormOptions::new());
  let a = ok(scoring.prepare(&speaker_a()), "prepare A");

  let refused = calibration.enrolled_side(Enrolled::new(foreign_token, &a));
  assert!(
    matches!(refused, Err(CalibrateError::ForeignSpeaker)),
    "a foreign token must be refused, got {refused:?}"
  );
  let accepted = ok(
    calibration.enrolled_side(Enrolled::new(own_token, &a)),
    "A's side under her own cohort's token",
  );
  assert_eq!(accepted.considered(), calibration.cohort().len() - 1);
}

/// A speaker the cohort holds nothing of still gets a token, and it names an
/// empty exclusion set — the ordinary case, since a cohort is a sample of a
/// library and most enrolled speakers are not in it. The token stays the same
/// one if that speaker IS pushed later, so an exclusion cannot be lost by the
/// order the caller assembled things in.
#[test]
fn a_speaker_the_cohort_does_not_hold_excludes_nothing_and_keeps_its_token() {
  let scoring = Scoring::Cosine;
  let (mut cohort, _) = library(scoring);
  let members = cohort.len();

  let token = cohort.speaker();
  let profile = ok(scoring.prepare(&speaker_b_again()), "prepare a stranger");

  let absent = Calibration::new(cohort.clone(), AsNormOptions::new());
  let side = ok(
    absent.enrolled_side(Enrolled::new(token, &profile)),
    "a side for a speaker the cohort does not hold",
  );
  assert_eq!(
    side.considered(),
    members,
    "there is nothing of this speaker's here to drop"
  );

  // Pushed afterwards, under the token that speaker was already given: the
  // entry is filed under it, so the exclusion still happens.
  cohort.push(token, profile);
  let present = Calibration::new(cohort, AsNormOptions::new());
  let after = ok(
    present.enrolled_side(Enrolled::new(token, &profile)),
    "a side once the cohort holds this speaker",
  );
  assert_eq!(
    after.considered(),
    members,
    "the entry pushed under an already-minted token must be dropped by it"
  );
}

/// An unenrolled probe's identity is what identification is trying to
/// discover, so its side covers the WHOLE cohort; the alternative this
/// module's first version recommended — dropping the candidate's entry from
/// the probe's side — moves the normalized trial across a threshold.
///
/// The truncated statistics have to be built by hand out of `diaric`'s own
/// constructor, because no entrypoint here can produce them any more — and so
/// does the whole-cohort normalization, for a second reason: the candidate's
/// side comes from the library calibration and the probe's from the held-out
/// one, and a trial refuses that pairing outright
/// (`two_sides_from_different_calibrations_are_refused` says why). Both numbers
/// here therefore run through `diaric`'s own `normalize` with the SAME
/// enrolment statistics, which is what makes this a comparison of the two
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
  let members = profiles.len();
  let held_out_cal = Calibration::new(HeldOutCohort::assuming_disjoint(profiles.clone()), options);
  let probe_side = side(&held_out_cal, &probe);
  assert_eq!(
    probe_side.considered(),
    members,
    "a probe has no identity to exclude, so every cohort member must be scored"
  );
  assert_eq!(probe_side.selected(), 3);

  // The candidate's side, from the library-sampled calibration that names it.
  let mut library_cohort = LibraryCohortBuilder::new();
  let mut roster = Roster::new();
  for (i, p) in profiles.iter().enumerate() {
    enrol(&mut library_cohort, &mut roster, Speaker::Impostor(i), *p);
  }
  let candidate_token = roster[&Speaker::Impostor(1)];
  let library_cal = Calibration::new(library_cohort, options);
  let enrolled_side = ok(
    library_cal.enrolled_side(Enrolled::new(candidate_token, &candidate)),
    "the candidate's cohort statistics",
  );

  // Two calibrations, so the door refuses the pair before any arithmetic —
  // from either end.
  for (cal_name, refused) in [
    (
      "the library calibration",
      library_cal.trial(&enrolled_side, &probe_side),
    ),
    (
      "the held-out calibration",
      held_out_cal.trial(&enrolled_side, &probe_side),
    ),
  ] {
    assert!(
      matches!(refused, Err(CalibrateError::CalibrationMismatch(_))),
      "a library-sampled enrolment side and a held-out probe side are two \
       impostor populations, and {cal_name} must refuse them: {refused:?}"
    );
  }

  // The same trial, with the candidate's entry dropped from the PROBE's side.
  let trial_raw = ok(candidate.score(&probe), "the trial score");
  let truncated = ok(
    CohortStats::from_scores(
      [
        ok(probe.score(&profiles[0]), "probe against member 0"),
        ok(probe.score(&profiles[2]), "probe against member 2"),
      ],
      &options,
    ),
    "the candidate-truncated probe statistics",
  );
  let enrolled_unbound = ok(
    CohortStats::from_scores(
      [
        ok(candidate.score(&profiles[0]), "candidate against member 0"),
        ok(candidate.score(&profiles[2]), "candidate against member 2"),
      ],
      &options,
    ),
    "the candidate's statistics written out by hand",
  );
  assert_eq!(
    enrolled_unbound.mean().to_bits(),
    enrolled_side.stats.mean().to_bits(),
    "the hand-built enrolment side must be the door's own, or the comparison \
     below is between two different trials"
  );
  // One enrolment side, two probe sides: the door's whole-cohort one, and the
  // candidate-truncated alternative. `probe_side.stats` is the door's own
  // statistic, read through the private field these tests can see, so the
  // honest half is not a second hand-built copy of it.
  let honest = ok(
    enrolled_unbound.normalize(trial_raw, &probe_side.stats),
    "the normalized trial against whole-cohort probe statistics",
  );
  let flipped = ok(
    enrolled_unbound.normalize(trial_raw, &truncated),
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
/// Every side here comes from the one held-out calibration — the recommended
/// arrangement, and the only one under which the two z-scores AS-Norm averages
/// are commensurable. Each trial hands back both numbers at once, which is how
/// the comparison is between one pair of trials rather than two.
#[test]
fn as_norm_separates_two_differently_placed_speakers_where_no_raw_threshold_can() {
  let scoring = Scoring::Cosine;
  let calibration = held_out_calibration(scoring);

  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let a2 = ok(scoring.prepare(&speaker_a_again()), "prepare A again");
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");
  let impostor = ok(
    scoring.prepare(&crowd()[0]),
    "prepare the impostor nearest A",
  );

  // A', B' and the impostor partner all sit OUTSIDE the held-out cohort, so
  // nothing needs excluding and no side depends on the other end of its trial.
  let side_a = side(&calibration, &a);
  let side_a2 = side(&calibration, &a2);
  let side_b = side(&calibration, &b);
  let side_b2 = side(&calibration, &b2);
  let side_impostor = side(&calibration, &impostor);

  let genuine_a = ok(calibration.trial(&side_a, &side_a2), "A vs A'");
  let genuine_b = ok(calibration.trial(&side_b, &side_b2), "B vs B'");
  let impostor_trial = ok(
    calibration.trial(&side_a, &side_impostor),
    "A vs an impostor",
  );

  // A threshold separates iff the weakest genuine trial outscores the
  // strongest impostor one. Raw scores fail that test.
  assert!(
    genuine_b.raw() < impostor_trial.raw(),
    "the fixture must reproduce #123's problem: a genuine trial for the \
     isolated speaker ({}) has to score BELOW an impostor trial for the \
     crowded one ({})",
    genuine_b.raw(),
    impostor_trial.raw()
  );

  assert!(
    genuine_a.calibrated() > impostor_trial.calibrated()
      && genuine_b.calibrated() > impostor_trial.calibrated(),
    "after AS-Norm a single threshold must separate: genuine A {}, genuine B \
     {}, impostor {}",
    genuine_a.calibrated(),
    genuine_b.calibrated(),
    impostor_trial.calibrated()
  );
}

/// The PLDA-projected source runs the same road end to end.
#[test]
fn the_plda_score_source_normalizes_end_to_end() {
  let scoring = Scoring::PldaCosine;
  let calibration = held_out_calibration(scoring);
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");

  let normalized = trial_of(&calibration, &b, &b2).calibrated();
  assert!(
    normalized.is_finite(),
    "a PLDA-space normalization must produce a usable number, got {normalized}"
  );

  // The statistics have to come out of the PLDA space, not out of a
  // `PldaCosine` that quietly degraded to `Cosine` somewhere in `prepare`.
  // Read through the private field, since the surface publishes neither.
  let side_b = side(&calibration, &b);
  let cosine_b = ok(
    Scoring::Cosine.prepare(&speaker_b()),
    "prepare B for cosine",
  );
  let cosine_side = side(&held_out_calibration(Scoring::Cosine), &cosine_b);
  assert!(
    (side_b.stats.mean() - cosine_side.stats.mean()).abs() > 1e-6
      || (side_b.stats.deviation() - cosine_side.stats.deviation()).abs() > 1e-6,
    "the PLDA side ({}, {}) is indistinguishable from the cosine side ({}, {})",
    side_b.stats.mean(),
    side_b.stats.deviation(),
    cosine_side.stats.mean(),
    cosine_side.stats.deviation()
  );
}

/// The wrapper adds binding and NOTHING else: on sides of one calibration, its
/// answer is `diaric`'s own arithmetic, bit for bit. A second implementation
/// of eq. (7) hiding in here would be a second set of `diaric`'s cancellation
/// bugs.
#[test]
fn a_matching_normalization_is_diarics_own_arithmetic_bit_for_bit() {
  let scoring = Scoring::Cosine;
  let calibration = held_out_calibration(scoring);
  let options = AsNormOptions::new();
  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");

  let got = trial_of(&calibration, &a, &b);

  let scores = |profile: &VoiceProfile| {
    crowd()
      .iter()
      .skip(HELD_OUT_FROM)
      .map(|v| {
        let entry = ok(scoring.prepare(v), "prepare a held-out impostor");
        ok(profile.score(&entry), "a cohort score")
      })
      .collect::<Vec<_>>()
  };
  let expected = ok(
    diaric::score_norm::as_norm(
      got.raw(),
      &ok(CohortStats::from_scores(scores(&a), &options), "A's side"),
      &ok(CohortStats::from_scores(scores(&b), &options), "B's side"),
    ),
    "diaric's own as_norm",
  );

  assert_eq!(
    got.calibrated().to_bits(),
    expected.to_bits(),
    "the bound door must return diaric's number unchanged: {} vs {expected}",
    got.calibrated()
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

/// One metric per calibration, structurally. A `PldaCosine` profile cannot
/// take a side of a `Cosine` calibration at all, so two sides that could be
/// handed to one trial cannot disagree about the metric — and a side from the
/// other metric's calibration is refused as the foreign calibration it is.
///
/// This replaces a three-way tag comparison at the last step. That step used
/// to read a number and two statistics and no profile: `Cosine` cohort scores
/// of `[-1, 1]` have mean `0` and deviation `1`, so any finite `PldaCosine`
/// trial score normalized against them came back finite and plausible. It
/// reads profiles again, and they are the ones the sides were built from.
#[test]
fn one_calibration_cannot_hold_two_metrics() {
  let cosine_cal = held_out_calibration(Scoring::Cosine);
  let plda_cal = held_out_calibration(Scoring::PldaCosine);

  let cosine_b = ok(Scoring::Cosine.prepare(&speaker_b()), "prepare B");
  let plda_b = ok(Scoring::PldaCosine.prepare(&speaker_b()), "prepare B");
  let plda_b2 = ok(
    Scoring::PldaCosine.prepare(&speaker_b_again()),
    "prepare B again",
  );

  // A foreign-metric profile cannot become a side of this calibration, so no
  // pair of sides of one calibration can disagree about the metric.
  let refused = cosine_cal.side(&plda_b);
  assert!(
    matches!(refused, Err(CalibrateError::ScoringMismatch(_))),
    "a PldaCosine profile must not take a side of a Cosine calibration, got \
     {refused:?}"
  );

  // And the only way to hold two metrics' sides at once is to hold two
  // calibrations, which the trial refuses as such.
  let plda_side = side(&plda_cal, &plda_b);
  let plda_side2 = side(&plda_cal, &plda_b2);
  let cosine_side = side(&cosine_cal, &cosine_b);
  for (name, refused) in [
    (
      "the PldaCosine calibration",
      plda_cal.trial(&plda_side, &cosine_side),
    ),
    (
      "the Cosine calibration",
      cosine_cal.trial(&plda_side, &plda_side2),
    ),
  ] {
    match refused {
      Err(CalibrateError::CalibrationMismatch(_)) => {}
      other => panic!("{name} must refuse a foreign side, got {other:?}"),
    }
  }
}

/// A cohort holding a foreign-source entry poisons a mean silently unless the
/// door refuses it, so it refuses it — through BOTH entrypoints. This is also
/// what makes a [`TrialSide`]'s own source sound: a surviving statistic can
/// only have been computed over entries that all matched its side.
#[test]
fn a_cohort_mixing_two_score_sources_is_refused_rather_than_averaged() {
  let (mut cohort, roster) = library(Scoring::Cosine);
  let outsider = cohort.speaker();
  cohort.push(
    outsider,
    ok(
      Scoring::PldaCosine.prepare(&speaker_b()),
      "prepare a foreign-source entry",
    ),
  );
  let a = ok(Scoring::Cosine.prepare(&speaker_a()), "prepare A");
  let options = AsNormOptions::new();
  let a_token = roster[&Speaker::A];

  let excluding = Calibration::new(cohort, options).enrolled_side(Enrolled::new(a_token, &a));
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
  let held = Calibration::new(HeldOutCohort::assuming_disjoint(mixed), options).side(&a);
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
  let a = ok(Scoring::Cosine.prepare(&speaker_a()), "prepare A");
  let refused = held_out_calibration(Scoring::PldaCosine).side(&a);
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

/// The caller's [`AsNormOptions`] has to REACH `diaric`'s selection, through
/// BOTH side doors — a calibration that quietly substituted the defaults would
/// pass every other test here — and it has to be readable back off the
/// calibration that holds it.
#[test]
fn the_callers_options_reach_diarics_selection() {
  use core::num::NonZeroUsize;

  let scoring = Scoring::Cosine;
  let (cohort, roster) = library(scoring);
  let b_token = roster[&Speaker::B];
  let members = cohort.len();
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let narrow = AsNormOptions::new().with_top_n(NonZeroUsize::new(4).expect("4 is non-zero"));

  let wide = Calibration::new(cohort.clone(), AsNormOptions::new());
  let narrow_library = Calibration::new(cohort, narrow);
  assert_eq!(narrow_library.options(), &narrow);

  // A clone of the cohort keeps the tokens it was cloned with, so one token
  // names the same speaker in both — which is what makes this a comparison of
  // two `top_n` values and of nothing else.
  let wide_side = ok(
    wide.enrolled_side(Enrolled::new(b_token, &b)),
    "statistics at the default top_n",
  );
  let narrow_side = ok(
    narrow_library.enrolled_side(Enrolled::new(b_token, &b)),
    "statistics at top_n = 4",
  );
  assert_eq!(
    wide_side.selected(),
    members - 1,
    "the default top_n is far above this cohort, so every considered score is selected"
  );
  assert_eq!(narrow_side.selected(), 4);

  // Through the other door too, so neither one can be the one that drops it.
  let narrow_held_out = ok(
    Calibration::new(held_out(scoring), narrow).side(&b),
    "held-out statistics at top_n = 4",
  );
  assert_eq!(narrow_held_out.selected(), 4);
}

/// `diaric`'s minimum-usable-cohort floor reaches through the wrapper, and
/// arrives as its OWN refusal — a cohort that is merely absent must not be
/// reported as a degenerate one — and it arrives translated, carrying the two
/// COUNTS and none of `diaric`'s arithmetic.
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
  let refused = Calibration::new(cohort, AsNormOptions::new()).side(&b);
  match refused {
    Err(CalibrateError::ScoreNorm(ScoreNormRefusal::CohortTooSmall(selection))) => {
      assert_eq!(selection.selected(), 1);
      assert_eq!(selection.considered(), 1);
      // The floor itself is named by the variant and rendered from the public
      // constant, so nothing has to carry it.
      assert!(
        CalibrateError::ScoreNorm(ScoreNormRefusal::CohortTooSmall(selection))
          .to_string()
          .contains(&super::MIN_COHORT_SCORES.to_string())
      );
    }
    other => panic!("a one-member cohort must be refused as too small, got {other:?}"),
  }
}

/// A REFUSAL must not hand back the operand the success path withholds.
///
/// A `min_deviation` above what a valid cohort spreads is a refusal path: no
/// side is produced, so none of `TrialSide`'s doors are involved at all. The
/// refusal itself carried the exact standard deviation of the side that was
/// rejected — `diaric`'s `DegenerateDeviation::deviation`, rendered by both
/// its `Debug` and its `Display` — which is one half of eq. (7)'s operands,
/// reached without a `TrialSide` ever existing.
#[test]
fn a_refused_side_discloses_no_deviation() {
  let scoring = Scoring::Cosine;
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");

  // The very statistics the refusal is about, computed here so the test knows
  // the number it is looking for. Every cosine is in `[-1, 1]`, so a floor of
  // `2.0` refuses this cohort while leaving it perfectly well formed.
  let members: Vec<VoiceProfile> = crowd()
    .iter()
    .skip(HELD_OUT_FROM)
    .map(|v| ok(scoring.prepare(v), "prepare a held-out impostor"))
    .collect();
  let truth = ok(
    CohortStats::from_scores(
      members
        .iter()
        .map(|m| ok(b.score(m), "score B against a cohort member")),
      &AsNormOptions::new(),
    ),
    "B's own statistics under a usable floor",
  );
  let sigma = truth.deviation();
  assert!(sigma > 0.0 && sigma < 2.0, "the fixture must be refusable");

  let refused = Calibration::new(
    held_out(scoring),
    AsNormOptions::new().with_min_deviation(2.0),
  )
  .side(&b);
  let e = match refused {
    Err(e) => e,
    Ok(side) => panic!("a deviation below the floor must refuse, got {side:?}"),
  };

  // The category and the counts survive; the number does not.
  match &e {
    CalibrateError::ScoreNorm(ScoreNormRefusal::DegenerateCohort(selection)) => {
      assert_eq!(selection.selected(), members.len());
      assert_eq!(selection.considered(), members.len());
    }
    other => panic!("a deviation below the floor is a degenerate cohort, got {other:?}"),
  }

  let debugged = format!("{e:?}");
  let displayed = e.to_string();
  for rendering in [&debugged, &displayed] {
    for spelling in [
      format!("{sigma}"),
      format!("{sigma:?}"),
      format!("{sigma:.3e}"),
      format!("{sigma:.6e}"),
    ] {
      assert!(
        !rendering.contains(&spelling),
        "a refusal must not disclose the deviation it refused: {spelling} in {rendering}"
      );
    }
  }
}

/// A profile is plain data and must stay movable and shareable across threads:
/// a library holds thousands of them and a confusion experiment fans them out.
/// The bound values travel with them. Pinned at compile time so a future field
/// type cannot regress the auto-derive silently.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<VoiceProfile>();
  assert_send_sync::<Scoring>();
  assert_send_sync::<TrialSide>();
  assert_send_sync::<CalibratedTrial>();
  assert_send_sync::<HeldOutCohort>();
  assert_send_sync::<LibraryCohortBuilder>();
  assert_send_sync::<LibraryCohort>();
  assert_send_sync::<Calibration<HeldOutCohort>>();
  assert_send_sync::<Calibration<LibraryCohort>>();
  assert_send_sync::<CalibrationId>();
  assert_send_sync::<SpeakerToken>();
  assert_send_sync::<Enrolled<'static>>();
  assert_send_sync::<CalibrateError>();
  assert_send_sync::<ScoreNormRefusal>();
};

/// A refusal must be unable to carry an arithmetic intermediate AT ALL, rather
/// than one variant at a time.
///
/// An `f64` anywhere in [`ScoreNormRefusal`] — at any depth, in any variant,
/// added at any time — makes [`Eq`] and [`Hash`](core::hash::Hash)
/// underivable. So this bound is a whole-type, compile-time proof that no
/// deviation, mean, z-score or normalized value can leave through a refusal;
/// the round that put a σ back would not compile rather than needing a test
/// that anticipated it.
const _: fn() = || {
  fn assert_sealed<T: Eq + core::hash::Hash>() {}
  assert_sealed::<ScoreNormRefusal>();
  assert_sealed::<CohortSelection>();
  // The token is the other value this surface mints, and it must stay a plain
  // comparable handle for the same reason.
  assert_sealed::<SpeakerToken>();
};

// ── one calibration per trial ────────────────────────────────────────────

/// codex's round-2 case, reproduced to its digits. `P` is the probe; `A` and
/// `B` are the two candidates the trial has to rank; `X` is the third library
/// member that keeps each enrolled side above [`MIN_COHORT_SCORES`] once its
/// own entry is dropped. `X` is placed so that `A·X = 0.8` and `B·X = 0`,
/// which is what pulls the two enrolment statistics apart.
///
/// [`MIN_COHORT_SCORES`]: super::MIN_COHORT_SCORES
fn ranking_case() -> [Vec<f32>; 4] {
  [
    row(&[(0, 1.0)]),                                             // P
    row(&[(0, 0.8), (1, 0.6)]),                                   // A
    row(&[(0, 0.7), (2, 0.51f32.sqrt())]),                        // B
    row(&[(0, 0.341_056_75), (1, 0.878_591), (2, -0.334_302_5)]), // X
  ]
}

/// The axis-aligned impostor cohort of the case above, over `axes`.
fn axis_calibration(scoring: Scoring, axes: &[(usize, f32)]) -> Calibration<HeldOutCohort> {
  Calibration::new(
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
    ),
    AsNormOptions::new(),
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

/// ROUND 2, FINDING 2. Two sides taken under DIFFERENT calibrations are
/// refused — and the refusal is not pedantry: on this fixture the mixed
/// pairing does not merely shift the number, it REVERSES which candidate ranks
/// first.
///
/// Every value in both arrangements is [`Scoring::Cosine`], every profile is
/// valid, and every deviation clears [`DEFAULT_MIN_DEVIATION`], so nothing
/// about the metric could see it: the metric was never what differed. The
/// mixed numbers are computed here through `diaric`'s own `normalize`, because
/// the bound door will not produce them any more.
///
/// Run over two cohorts. The first is codex's as reported, `{±e0, ±e1, ±e2}`,
/// which holds a member on the probe's own axis — so `assuming_disjoint` is a
/// geometric falsehood there, stated only to keep the reported digits
/// checkable. The second drops that member, which makes the fixture's
/// assertion true and shows the reversal does not depend on it.
///
/// [`DEFAULT_MIN_DEVIATION`]: super::DEFAULT_MIN_DEVIATION
#[test]
fn two_sides_from_different_calibrations_are_refused() {
  let scoring = Scoring::Cosine;
  let options = AsNormOptions::new();
  let [probe_row, a_row, b_row, x_row] = ranking_case();

  let probe = ok(scoring.prepare(&probe_row), "prepare the probe");
  let a = ok(scoring.prepare(&a_row), "prepare A");
  let b = ok(scoring.prepare(&b_row), "prepare B");
  let x = ok(scoring.prepare(&x_row), "prepare X");

  let raw_a = ok(a.score(&probe), "A against the probe");
  let raw_b = ok(b.score(&probe), "B against the probe");
  assert!(
    raw_a > raw_b,
    "the fixture must start with A ahead on the raw score: A {raw_a} vs B {raw_b}"
  );

  // The library-sampled cohort: it holds A and B themselves, so each enrolled
  // side drops its own entries and keeps the other two.
  let mut library = LibraryCohortBuilder::new();
  let mut roster = Roster::new();
  let a_token = enrol(&mut library, &mut roster, Speaker::A, a);
  let b_token = enrol(&mut library, &mut roster, Speaker::B, b);
  enrol(&mut library, &mut roster, Speaker::Impostor(0), x);
  let library_cal = Calibration::new(library, options);
  let a_enrolled = ok(
    library_cal.enrolled_side(Enrolled::new(a_token, &a)),
    "A's side over the library cohort",
  );
  let b_enrolled = ok(
    library_cal.enrolled_side(Enrolled::new(b_token, &b)),
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
    let held_out = axis_calibration(scoring, axes);
    let probe_side = side(&held_out, &probe);

    // The arrangement that IS commensurable: both sides of one calibration.
    let a_shared = ok(
      held_out.trial(&side(&held_out, &a), &probe_side),
      "A normalized, both sides of one calibration",
    );
    let b_shared = ok(
      held_out.trial(&side(&held_out, &b), &probe_side),
      "B normalized, both sides of one calibration",
    );
    assert_eq!(a_shared.raw().to_bits(), raw_a.to_bits());
    assert_eq!(b_shared.raw().to_bits(), raw_b.to_bits());
    assert!(
      (a_shared.calibrated() - shared_a).abs() < 1e-6,
      "A over one calibration: {}",
      a_shared.calibrated()
    );
    assert!(
      (b_shared.calibrated() - shared_b).abs() < 1e-6,
      "B over one calibration: {}",
      b_shared.calibrated()
    );
    assert!(
      a_shared.calibrated() > b_shared.calibrated(),
      "one calibration for both sides must keep A ahead, as the raw score has \
       it: A {} vs B {}",
      a_shared.calibrated(),
      b_shared.calibrated()
    );

    // The arrangement that is not, and what it does to the order. `diaric`'s
    // own arithmetic, because the door refuses to run it.
    let a_mixed = ok(
      a_enrolled.stats.normalize(raw_a, &probe_side.stats),
      "A normalized across two calibrations",
    );
    let b_mixed = ok(
      b_enrolled.stats.normalize(raw_b, &probe_side.stats),
      "B normalized across two calibrations",
    );
    assert!(
      (a_mixed - mixed_a).abs() < 1e-6,
      "A across two calibrations: {a_mixed}"
    );
    assert!(
      (b_mixed - mixed_b).abs() < 1e-6,
      "B across two calibrations: {b_mixed}"
    );
    assert!(
      b_mixed > a_mixed,
      "the whole point: two calibrations put B first, against both the raw \
       order and the one-calibration order — A {a_mixed} vs B {b_mixed}"
    );

    // Which is why the door will not do it, from either end.
    for (side_of_a_b, name) in [(&a_enrolled, "A"), (&b_enrolled, "B")] {
      match held_out.trial(side_of_a_b, &probe_side) {
        Err(CalibrateError::CalibrationMismatch(m)) => {
          assert_eq!(m.expected(), held_out.id());
          assert_eq!(m.enrolled(), library_cal.id());
          assert_eq!(m.probe(), held_out.id());
          assert_ne!(m.enrolled(), m.probe());
        }
        other => panic!("{name}'s library side must be refused here, got {other:?}"),
      }
      match library_cal.trial(side_of_a_b, &probe_side) {
        Err(CalibrateError::CalibrationMismatch(m)) => {
          assert_eq!(m.expected(), library_cal.id());
          assert_eq!(m.enrolled(), library_cal.id());
          assert_eq!(m.probe(), held_out.id());
        }
        other => {
          panic!("{name}'s trial against a foreign probe side must be refused, got {other:?}")
        }
      }
    }
  }
}

/// The identity is the CALIBRATION's: a side taken before a cohort grew cannot
/// be averaged against one taken after. A cohort cannot grow under a
/// calibration at all — [`Calibration::new`] takes it by value — so growing
/// one means building a second calibration, and that is what makes the two
/// populations two identities.
///
/// [`Calibration::new`]: super::Calibration::new
#[test]
fn a_grown_cohort_is_a_different_calibration() {
  let scoring = Scoring::Cosine;
  let options = AsNormOptions::new();
  let (mut cohort, roster) = library(scoring);
  let a_token = roster[&Speaker::A];
  let b_token = roster[&Speaker::B];

  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let before_cal = Calibration::new(cohort.clone(), options);
  let before = ok(
    before_cal.enrolled_side(Enrolled::new(a_token, &a)),
    "A's side before the cohort grew",
  );
  assert_eq!(before.calibration(), before_cal.id());

  let newcomer = cohort.speaker();
  cohort.push(
    newcomer,
    ok(
      scoring.prepare(&speaker_b_again()),
      "prepare a new impostor",
    ),
  );
  let after_cal = Calibration::new(cohort, options);
  // A cohort CLONED and then grown keeps the tokens it was cloned with, so
  // one token still names one speaker in both lineages — the new member got a
  // token of its own, minted fresh.
  let after = ok(
    after_cal.enrolled_side(Enrolled::new(b_token, &b)),
    "B's side after the cohort grew",
  );
  assert_ne!(
    before.calibration(),
    after.calibration(),
    "a cohort with a member added is a different calibration"
  );

  assert!(
    matches!(
      after_cal.trial(&before, &after),
      Err(CalibrateError::CalibrationMismatch(_))
    ),
    "a side from before the push and one from after are two populations"
  );

  // A clone of the CALIBRATION is the same population under the same options,
  // so it keeps the identity and its sides still pair.
  let clone = after_cal.clone();
  assert_eq!(clone.id(), after_cal.id());
  let cloned_side = ok(
    clone.enrolled_side(Enrolled::new(a_token, &a)),
    "A's side over a clone of the grown calibration",
  );
  assert_eq!(cloned_side.calibration(), after.calibration());
  assert!(
    ok(
      after_cal.trial(&cloned_side, &after),
      "a cloned calibration normalizes"
    )
    .calibrated()
    .is_finite()
  );
}

/// Two calibrations built from the very same profiles are two identities,
/// because nothing in this crate can tell they are one population. Stated as a
/// test because it is the conservative half of the trade — a caller who
/// rebuilds an identical cohort gets a refusal, not a silent pass — and a
/// future "compare the members instead" would be a behaviour change, not an
/// optimisation.
#[test]
fn two_calibrations_over_identical_cohorts_are_two_identities() {
  let scoring = Scoring::Cosine;
  let one = held_out_calibration(scoring);
  let two = held_out_calibration(scoring);
  assert_eq!(one.cohort(), two.cohort(), "the members are the same");
  assert_ne!(one.id(), two.id());
  assert_ne!(
    one, two,
    "equality includes the identity, so two separately built calibrations \
     differ even when their cohorts and options do not"
  );

  let b = ok(scoring.prepare(&speaker_b()), "prepare B");
  let b2 = ok(scoring.prepare(&speaker_b_again()), "prepare B again");
  assert!(
    matches!(
      one.trial(&side(&one, &b), &side(&two, &b2)),
      Err(CalibrateError::CalibrationMismatch(_))
    ),
    "two identically-built calibrations are still two calibrations"
  );
  assert!(
    trial_of(&one, &b, &b2).calibrated().is_finite(),
    "one calibration for both sides"
  );
}

// ── round 3: what a calibration binds ────────────────────────────────────

/// The five-axis cohort of [`ranking_case`] — codex's `{±e0, ±e1, ±e2}` with
/// the probe's own axis dropped, so `assuming_disjoint` is true of the fixture
/// rather than merely asserted.
const ROUND3_AXES: [(usize, f32); 5] = [(0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)];

/// ROUND 3, FINDING 1. A trial is bound to its ENDPOINTS: the raw score is
/// computed from the two sides' own profiles, so a side belonging to some
/// other speaker cannot calibrate this trial.
///
/// Under the shape this replaces, `as_norm(A.score(P), B_side, P_side)`
/// returned `Ok(2.134434)` — B's statistics, valid, `Cosine`, over the very
/// same cohort, against A's trial score. A threshold of `2.18` separates that
/// from A's own `2.216988`, so the substitution changed the decision. There is
/// now no argument to make it with: `trial` is handed no score, and the two
/// sides it is handed are the two endpoints.
#[test]
fn a_trial_is_between_the_two_sides_it_was_given() {
  let scoring = Scoring::Cosine;
  let [probe_row, a_row, b_row, _x_row] = ranking_case();

  let probe = ok(scoring.prepare(&probe_row), "prepare P");
  let a = ok(scoring.prepare(&a_row), "prepare A");
  let b = ok(scoring.prepare(&b_row), "prepare B");

  let calibration = axis_calibration(scoring, &ROUND3_AXES);
  let a_side = side(&calibration, &a);
  let b_side = side(&calibration, &b);
  let probe_side = side(&calibration, &probe);

  let a_trial = ok(calibration.trial(&a_side, &probe_side), "A against P");
  let b_trial = ok(calibration.trial(&b_side, &probe_side), "B against P");

  // Each trial's raw score is its own endpoints', not the other pair's.
  assert!(
    (a_trial.raw() - 0.8).abs() < 1e-6,
    "A against P is 0.8, got {}",
    a_trial.raw()
  );
  assert!(
    (b_trial.raw() - 0.7).abs() < 1e-6,
    "B against P is 0.7, got {}",
    b_trial.raw()
  );
  assert!(
    (a_trial.calibrated() - 2.216_987_6).abs() < 1e-6,
    "A's own calibrated score: {}",
    a_trial.calibrated()
  );

  // The number the old shape produced from B's statistics and A's trial score.
  // No pairing of sides here can reach it: swapping the enrolment side swaps
  // the raw score with it.
  const FOREIGN: f64 = 2.134_434;
  for (name, got) in [
    ("A against P", a_trial.calibrated()),
    ("B against P", b_trial.calibrated()),
    (
      "P against A",
      ok(calibration.trial(&probe_side, &a_side), "P against A").calibrated(),
    ),
    (
      "P against B",
      ok(calibration.trial(&probe_side, &b_side), "P against B").calibrated(),
    ),
  ] {
    assert!(
      (got - FOREIGN).abs() > 1e-3,
      "{name} produced {got}, the value B's statistics used to give A's trial \
       score — a side of one speaker must not calibrate another's trial"
    );
  }

  // And the substitution mattered: a threshold of 2.18 sits between the two.
  assert!(
    a_trial.calibrated() > 2.18 && FOREIGN < 2.18,
    "the fixture must keep the threshold between them: A {} vs the foreign \
     {FOREIGN}",
    a_trial.calibrated()
  );
}

/// ROUND 3, FINDING 2. The public surface must not hand out the operands of
/// eq. (7). `Debug` is the half that can be asserted at run time — a derived
/// one printed `CohortStats`'s `mean` and `deviation` in full; the accessors
/// themselves are pinned by the `compile_fail` doctests on the module page,
/// since no runtime assertion can say a method does not exist.
#[test]
fn a_side_never_renders_the_operands_of_the_formula() {
  let scoring = Scoring::Cosine;
  let calibration = held_out_calibration(scoring);
  let profile = ok(scoring.prepare(&speaker_a()), "prepare A");
  let rendered = format!("{:?}", side(&calibration, &profile));

  assert!(
    !rendered.contains("mean"),
    "a side must not render its cohort mean: {rendered}"
  );
  assert!(
    !rendered.contains("deviation"),
    "a side must not render its cohort deviation: {rendered}"
  );
  // It still has to be useful: what a caller needs from a `{:?}` is which
  // calibration the side belongs to and how much cohort it actually covered.
  for expected in ["Cosine", "selected", "considered", "calibration"] {
    assert!(
      rendered.contains(expected),
      "a side's Debug must still name {expected}: {rendered}"
    );
  }
}

/// ROUND 3, FINDING 3. Two sides derived under different [`AsNormOptions`] are
/// not commensurable, and they cannot meet.
///
/// Under the shape this replaces, A's side at `top_n = 2` and P's side at the
/// defaults passed every check and returned `Ok(2.083333)` — a value
/// corresponding to no single AS-Norm configuration. The options now belong to
/// the calibration, so two configurations are two calibrations, and a side of
/// one is refused by the other.
#[test]
fn two_sides_derived_under_different_options_cannot_meet() {
  use core::num::NonZeroUsize;

  let scoring = Scoring::Cosine;
  let [probe_row, a_row, b_row, _x_row] = ranking_case();

  let probe = ok(scoring.prepare(&probe_row), "prepare P");
  let a = ok(scoring.prepare(&a_row), "prepare A");

  let members: Vec<VoiceProfile> = ROUND3_AXES
    .iter()
    .map(|&(axis, sign)| {
      ok(
        scoring.prepare(&row(&[(axis, sign)])),
        "prepare an axis impostor",
      )
    })
    .collect();
  let top_two = AsNormOptions::new().with_top_n(NonZeroUsize::new(2).expect("2 is non-zero"));

  let defaults_cal = Calibration::new(
    HeldOutCohort::assuming_disjoint(members.clone()),
    AsNormOptions::new(),
  );
  let top_two_cal = Calibration::new(HeldOutCohort::assuming_disjoint(members), top_two);
  assert_eq!(
    defaults_cal.cohort(),
    top_two_cal.cohort(),
    "the same members: only the configuration differs"
  );

  let a_narrow = ok(top_two_cal.side(&a), "A's side at top_n = 2");
  let probe_wide = ok(defaults_cal.side(&probe), "P's side at the defaults");
  assert_eq!(a_narrow.selected(), 2);
  assert_eq!(probe_wide.selected(), 5);

  for (name, refused) in [
    ("the defaults", defaults_cal.trial(&a_narrow, &probe_wide)),
    ("top_n = 2", top_two_cal.trial(&a_narrow, &probe_wide)),
  ] {
    match refused {
      Err(CalibrateError::CalibrationMismatch(m)) => {
        assert_ne!(m.enrolled(), m.probe());
        assert!(
          m.expected() == m.enrolled() || m.expected() == m.probe(),
          "the calibration asked must be one of the two: {m:?}"
        );
      }
      other => panic!(
        "sides derived under two configurations must not be averaged by \
         {name}, got {other:?}"
      ),
    }
  }

  // And each configuration on its own still normalizes, so the refusal is
  // about the MIXTURE rather than about `top_n = 2` being unusable. Scored
  // between A and B, whose top-2 cohort scores still spread — the probe's do
  // not, which is `diaric`'s own floor and a different refusal entirely.
  let b = ok(scoring.prepare(&b_row), "prepare B");
  for (name, cal) in [("the defaults", &defaults_cal), ("top_n = 2", &top_two_cal)] {
    let value = ok(
      cal.trial(&ok(cal.side(&a), "A's side"), &ok(cal.side(&b), "B's side")),
      name,
    );
    assert!(value.calibrated().is_finite(), "{name}: {value:?}");
  }
}

/// A side of a calibration is reusable across every trial of THAT calibration
/// and across no other — the guarantee the module docs' cost table states, and
/// the one an invariant-lifetime brand would have taken away.
#[test]
fn one_side_serves_every_trial_of_its_own_calibration() {
  let scoring = Scoring::Cosine;
  let calibration = held_out_calibration(scoring);

  let a = ok(scoring.prepare(&speaker_a()), "prepare A");
  let partners = [
    ok(scoring.prepare(&speaker_a_again()), "prepare A again"),
    ok(scoring.prepare(&speaker_b()), "prepare B"),
    ok(scoring.prepare(&speaker_b_again()), "prepare B again"),
  ];

  // One side for A, taken once, and every partner scored against it.
  let a_side = side(&calibration, &a);
  for (i, partner) in partners.iter().enumerate() {
    let reused = ok(
      calibration.trial(&a_side, &side(&calibration, partner)),
      "a trial reusing A's side",
    );
    // The same numbers a freshly taken side would give: the side depends on
    // its own profile and its own calibration, and on nothing about the
    // partner.
    let fresh = trial_of(&calibration, &a, partner);
    assert_eq!(
      reused.raw().to_bits(),
      fresh.raw().to_bits(),
      "partner {i}: a reused side must not change the raw score"
    );
    assert_eq!(
      reused.calibrated().to_bits(),
      fresh.calibrated().to_bits(),
      "partner {i}: a reused side must not change the calibrated score"
    );
  }
}

/// **The residual, measured rather than asserted.** μ and σ leave this module
/// by no door at all — no accessor, no [`Debug`], no refusal — and they are
/// still RECOVERABLE in closed form from the two numbers a
/// [`CalibratedTrial`] publishes on purpose.
///
/// Write `aᵢ = 1/σᵢ` and `bᵢ = −μᵢ/σᵢ`, so a z-score is the affine `aᵢ·s + bᵢ`
/// and eq. (7) reads `c(i,j) = ½·[(aᵢ + aⱼ)·s(i,j) + bᵢ + bⱼ]`. Trialling a
/// side against ITSELF gives `c(i,i) = aᵢ·s(i,i) + bᵢ`, which eliminates every
/// `bᵢ`; three sides then leave three linear equations in three unknowns, and
/// the solve returns each side's σ and μ to the last few digits. Nothing here
/// reads a private field until the final comparison: only `raw`, `calibrated`,
/// and one calibration.
///
/// This is why the claim in the module docs is about what is HANDED OUT and
/// not about what can be reconstructed. A door that publishes both a raw score
/// and its normalization publishes the affine map between them, and no library
/// can withhold the map while publishing both of its ends — which #123's own
/// raw-versus-calibrated comparison is the reason to do.
#[test]
fn the_two_published_numbers_still_determine_every_side_that_produced_them() {
  let scoring = Scoring::Cosine;
  let calibration = held_out_calibration(scoring);

  let rows = [speaker_a(), speaker_b(), crowd()[0].clone()];
  let profiles: Vec<VoiceProfile> = rows
    .iter()
    .map(|v| ok(scoring.prepare(v), "prepare a subject"))
    .collect();
  let sides: Vec<TrialSide> = profiles.iter().map(|p| side(&calibration, p)).collect();

  // Everything below this line is what a caller can read.
  let trial = |i: usize, j: usize| {
    let t = ok(
      calibration.trial(&sides[i], &sides[j]),
      "a calibrated trial",
    );
    (t.raw(), t.calibrated())
  };
  let own: Vec<(f64, f64)> = (0..3).map(|i| trial(i, i)).collect();

  // 2·c(i,j) − c(i,i) − c(j,j) = aᵢ·(s(i,j) − s(i,i)) + aⱼ·(s(i,j) − s(j,j)),
  // one row per pair. Self-cosines are not assumed equal to each other or to
  // one: `Scoring::Cosine` is an f32 dot product, and they are not.
  let equation = |i: usize, j: usize| {
    let (s_ij, c_ij) = trial(i, j);
    let (s_ii, c_ii) = own[i];
    let (s_jj, c_jj) = own[j];
    (s_ij - s_ii, s_ij - s_jj, 2.0 * c_ij - c_ii - c_jj)
  };
  let (ab_a, ab_b, ab_r) = equation(0, 1);
  let (ac_a, ac_c, ac_r) = equation(0, 2);
  let (bc_b, bc_c, bc_r) = equation(1, 2);

  let m = [[ab_a, ab_b, 0.0], [ac_a, 0.0, ac_c], [0.0, bc_b, bc_c]];
  let rhs = [ab_r, ac_r, bc_r];
  let det = |m: &[[f64; 3]; 3]| {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  };
  let base = det(&m);
  assert!(base.abs() > 1e-12, "the three sides must be independent");

  for (i, (s_ii, c_ii)) in own.iter().enumerate() {
    // Cramer's rule for aᵢ.
    let mut replaced = m;
    for (row, r) in replaced.iter_mut().zip(rhs.iter()) {
      row[i] = *r;
    }
    let a_i = det(&replaced) / base;
    let sigma = 1.0 / a_i;
    let mu = (a_i * s_ii - c_ii) * sigma;

    // `sides[i].stats` is the private field these tests can see and a caller
    // cannot. The point of the test is that a caller did not need it.
    let truth_sigma = sides[i].stats.deviation();
    let truth_mu = sides[i].stats.mean();
    assert!(
      (sigma - truth_sigma).abs() <= 1e-9 * truth_sigma.abs(),
      "side {i}: recovered sigma {sigma} against {truth_sigma}"
    );
    assert!(
      (mu - truth_mu).abs() <= 1e-9 * truth_mu.abs(),
      "side {i}: recovered mean {mu} against {truth_mu}"
    );
  }
}
