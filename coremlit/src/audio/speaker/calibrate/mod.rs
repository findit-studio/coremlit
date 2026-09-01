//! Adaptive score normalization (AS-Norm1) for **cross-recording** speaker
//! identification — turning raw similarity scores, which are not comparable
//! across speakers, into scores one fixed threshold can be read against.
//!
//! [`findit-studio/coremlit#123`][coremlit-123] states the problem the rest of
//! this module answers:
//!
//! > Raw cosine scores are not comparable across speakers. A single global
//! > threshold over-merges some identities and under-merges others.
//!
//! One speaker sits in a crowded region of the embedding space and scores
//! highly against everyone; another sits alone and scores lower against
//! everyone. AS-Norm rescales each trial score against the cohort distribution
//! of the *two speakers involved*, so one threshold means the same thing
//! everywhere. [`diaric::score_norm`]'s module docs carry the arithmetic, the
//! literature, and the accuracy postcondition; this door carries the two things
//! that are `coremlit`'s: what an embedding *is* here, and how it reaches a
//! score.
//!
//! [coremlit-123]: https://github.com/findit-studio/coremlit/issues/123
//!
//! # This is the cross-recording door, not a diarization one
//!
//! Everything else in [`speaker`](crate::audio::speaker) is *within*-recording:
//! [`extract`](crate::audio::speaker::extract) turns one clip into per-chunk
//! tensors and [`diarize`](crate::audio::speaker::extract::Extraction::diarize)
//! labels that clip's own speakers. This module is the first surface here that
//! looks *across* recordings: a library of voice profiles at cluster-centroid
//! grain, scored against each other so the same person found in two different
//! recordings can be recognized as one identity.
//!
//! Nothing here is wired into the clustering path, deliberately. Within a
//! recording the clusterer sees every embedding at once and the pyannote
//! pipeline it reproduces is parity-gated end to end; changing the score it
//! merges on is a different question with a different oracle, and it is not
//! this one.
//!
//! # `coremlit` scores. It does not hold the library
//!
//! Sans-I/O, like the rest of the crate. The caller owns the store, the
//! identities, and the policy that decides which profiles become a cohort.
//! What this module takes is a cohort the caller assembled and the candidates
//! the caller chose; what it returns is a number. There is no enrolment, no
//! persistence, and no cohort-selection heuristic here, because none of those
//! can be answered without the library this crate cannot see.
//!
//! # The arithmetic is `diaric`'s, not a second copy
//!
//! Every statistic and every normalization on this page is
//! [`diaric::score_norm`] called through — its `CohortStats`,
//! [`diaric::score_norm::as_norm`], the [`AsNormOptions`] defaults,
//! [`MAX_NORMALIZED_ERROR`]'s accuracy postcondition. That module is where the
//! cancellation analysis, the subnormal handling and the two-tier error
//! predicate live, and a second implementation of AS-Norm1 would be a second
//! set of those bugs. What this module adds is everything between "a raw
//! WeSpeaker row" and "a number a threshold can be read against":
//!
//! - the [`Scoring`] a profile is prepared for, and the preparation itself;
//! - a **fallible** boundary. `diaric`'s cohort statistics take an INFALLIBLE
//!   `FnMut(&S, &T) -> f64`, so a refusal inside scoring has no way out of the
//!   closure. The side constructors carry it out by hand rather than letting a
//!   caller decide what number a failed score should be — an `unwrap_or(0.0)`
//!   inside that closure poisons a mean silently, which is the one failure
//!   AS-Norm exists to prevent;
//! - the [`Calibration`] that holds every attribute a trial has to agree on,
//!   so no two of them can be got out of step. That is the rest of this page,
//!   and it is the third correction to what this module said when it was first
//!   written.
//!
//! # One scoped operation, not three loose values
//!
//! A calibrated trial is the product of a single [`Calibration`]. Nothing else
//! produces one, and a caller assembles no operands at all.
//!
//! ```text
//! Calibration::new(cohort, options)     one cohort, one configuration
//!   .side(&probe)             -> TrialSide        bound to that profile
//!   .enrolled_side(Enrolled)  -> TrialSide        bound to that profile and speaker
//!   .trial(&a, &b)            -> CalibratedTrial  the raw score AND the calibrated one
//! ```
//!
//! AS-Norm1's eq. (7) has four things that must agree, and each of the first
//! three was once a value the caller carried between calls:
//!
//! | must agree | how it is fixed |
//! |---|---|
//! | the **metric** both numbers were computed in | a side exists only if its profile's [`Scoring`] matched every cohort entry it was scored against; [`Calibration::trial`] then computes the raw score **from those same two profiles** |
//! | the **cohort** both sides selected their top-N from | the [`Calibration`] owns it, by value, for its whole life |
//! | the **endpoints** the trial score is between | [`Calibration::trial`] takes no score. It computes one, from the two sides' own profiles, so the trial is between whoever the sides are |
//! | the **configuration** both sides were derived under | the [`Calibration`] owns the [`AsNormOptions`] too, and both side constructors read it from there |
//!
//! **Why this shape and not one more tag.** The three rounds of review that
//! produced it each bound one more attribute onto three loose values — a raw
//! score and two statistics — that a caller assembled by hand: first the
//! metric, then the cohort, then the endpoints and the configuration. Each
//! round's fix was correct and each left the next attribute unbound, because
//! the shape was the defect: a value a caller carries between two calls is a
//! value they can carry from the wrong place. Removing the assembly removes
//! the class. There is no public constructor for a [`TrialSide`] or a
//! [`CalibratedTrial`], and no function anywhere on this page that takes a raw
//! score and two statistics.
//!
//! **What is left, stated rather than left to be found.** Two [`Calibration`]s
//! are two values, so a caller holding both can still hand one's side to the
//! other's [`trial`](Calibration::trial). That is a refusal
//! ([`CalibrateError::CalibrationMismatch`]) rather than a wrong number, and
//! it is *one* check over *one* opaque identity — not four checks over four
//! attributes — because a calibration fixes all four at once. Making even that
//! a compile error would take an invariant lifetime and a closure-scoped API,
//! which would end the reuse guarantee below: a cached side could not outlive
//! the scope that built it. The trade is deliberate, and it is the residual.
//!
//! # Two sides, and only one of them has an identity
//!
//! AS-Norm needs a cohort statistic for *both* speakers in a trial, and the
//! two sides are not symmetric.
//!
//! The **enrolled** side is a speaker the caller's library already names. Its
//! identity is known, so a cohort drawn from that same library can have
//! exactly that speaker's entries removed —
//! [`Calibration::enrolled_side`], which is
//! [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding)
//! with the cohort's own [`SpeakerToken`] for that speaker and the profile
//! handed over as one [`Enrolled`] value.
//!
//! The **probe** side is a recording whose speaker is not yet known. That
//! identity is what identification is *trying to discover*, so the caller has
//! no identity of the probe's to exclude — and neither of the two things they
//! could reach for instead is answerable:
//!
//! - excluding nothing from a library-sampled cohort is self-contamination
//!   whenever the probe's speaker is enrolled, which is precisely the case a
//!   positive identification is. A self-match is the largest score obtainable,
//!   so top-N selection is *guaranteed* to pick it up;
//! - excluding the **candidate's** identity is worse, and it is what this
//!   module's first version told a caller to do. It drops a valid impostor,
//!   and it makes the probe's statistics depend on which candidate the probe
//!   is being scored against. `diaric` names that failure at the entrypoint
//!   itself:
//!
//!   > Only `speaker`'s **own** entries are removed — never the other side of
//!   > the trial. […] Excluding the partner too would make the statistics
//!   > trial-dependent and give back the quadratic cost the precomputation
//!   > exists to avoid.
//!
//!   It moves the answer, too. In a three-member cohort where a probe scores
//!   `[0, 0.8, 0.2]`, dropping the candidate that scored `0.8` leaves
//!   `[0, 0.2]`; both sets give finite, healthy-looking statistics, and the
//!   normalized trial moves from `1.641` to `4.455`, across any threshold
//!   between them (`an_unidentified_probes_side_covers_its_whole_cohort` pins
//!   both numbers).
//!
//! So a probe gets a **held-out** cohort: [`HeldOutCohort`], profiles the
//! caller asserts hold no material from any speaker that will be scored
//! against them. Nothing is excluded because nothing needs to be — the
//! assumption Matějka et al. 2017 §2.1 actually makes is *restored* rather
//! than approximated.
//!
//! That assertion is answerable where the exclusion question is not, and the
//! difference is the whole point. It is a fact about where the cohort **came
//! from** — a public corpus of strangers, or a partition of the library
//! reserved for this and never enrolled — and the caller knows it without
//! knowing who the probe is. *"Is this probe Alice?"* is the question being
//! asked. *"Was this cohort drawn from people who are not in my library?"* is
//! provenance, and the caller answered it when they assembled the cohort.
//!
//! One cohort serves both sides of a trial. AS-Norm averages two z-scores, and
//! they are commensurable only when they measure the same trial score against
//! the same impostor population — [`diaric::score_norm`]'s own statement of
//! eq. (7) has each side selecting its top-N of *the shared cohort*. So a
//! trial with an unidentified probe in it is calibrated over one
//! `Calibration<HeldOutCohort>`, which is what the example below does.
//! `Calibration<LibraryCohort>` is the other shape: one cohort serving two
//! NAMED sides, each dropping its own speaker's entries. The MIXTURE of those
//! two — a library-sampled enrolment side against a held-out probe side —
//! reverses the candidate ranking on valid, healthy-looking numbers
//! (`two_sides_from_different_calibrations_are_refused` has the case), and
//! there is no way to express it: the two sides come from two calibrations,
//! and neither calibration will normalize the other's.
//!
//! ## Why the signatures changed, and not only the prose
//!
//! The shape this replaces permitted correct use: a caller who reached for
//! `cohort_stats_assuming_disjoint` with a genuinely held-out cohort got
//! exactly the statistics above. What was wrong was everything that told them
//! what to reach for — the module docs, the runnable example, and the tests
//! all excluded the partner's key from the probe's side, which is the failure
//! `diaric` names verbatim.
//!
//! Fixing the prose alone would have left two properties unpinnable. Under
//! that signature "a probe's side does not depend on the candidate" is caller
//! discipline, not an API property, so no test in this crate could hold it —
//! the guidance would have been the only guard over exactly the mistake the
//! guidance had just made. Binding the identity to the profile turns it into a
//! type property with a `compile_fail` proof, and an identity-free
//! [`HeldOutCohort`] means the probe's road has nothing to pass at all. What is
//! left of the hazard is stated below rather than claimed away.
//!
//! ## The token travels with the profile
//!
//! [`Calibration::enrolled_side`] takes one [`Enrolled`] rather than a speaker
//! *and* a side, so there is no second argument left that could name a
//! different speaker.
//!
//! The identity half is a [`SpeakerToken`] the cohort minted, and there is
//! **no key type on this surface for it to have been resolved from**.
//! [`LibraryCohortBuilder::speaker`] takes no argument; a
//! [`LibraryCohort`] holds token membership and token-keyed entries and
//! nothing else. So no value the caller owns takes part in deciding which
//! speaker a token names, at any point in a cohort's life, and no duplicate of
//! a cohort — a [`Clone`], a borrow, a re-derived copy — can answer that
//! question differently, because none of them can be asked it:
//!
//! ```compile_fail,E0599
//! use coremlit::audio::speaker::calibrate::{
//!   AsNormOptions, Calibration, LibraryCohortBuilder, Scoring,
//! };
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let mut cohort = LibraryCohortBuilder::new();
//! let alice = cohort.speaker();
//! cohort.push(alice, Scoring::Cosine.prepare(&raw).unwrap());
//! let calibration = Calibration::new(cohort, AsNormOptions::new());
//! // An OWNED duplicate of the calibration's own cohort — no borrow in the
//! // way. It still has no mutator on it, and no key to look anything up by.
//! let mut duplicate = calibration.cohort().clone();
//! let _ = duplicate.speaker();
//! ```
//!
//! ```compile_fail,E0061
//! use coremlit::audio::speaker::calibrate::LibraryCohortBuilder;
//! let mut cohort = LibraryCohortBuilder::new();
//! // A token is minted, not resolved: there is no argument to hand it, and so
//! // nothing whose later value could re-decide the answer.
//! let _ = cohort.speaker(7u32);
//! ```
//!
//! **Why the whole key type went, rather than one more seal on it.** [`Eq`]
//! does not forbid interior mutability: `Rc<Cell<u64>>` is a perfectly good
//! `Eq` key whose comparison reads a cell the caller can still write to.
//! Rewriting that cell so one speaker's key equalled another's made a lookup
//! for the *second* speaker hand back the *first* one's token, and the side
//! taken under it then dropped the wrong speaker's entries while leaving the
//! subject's own in place — a finite, plausible, self-contaminated number,
//! with the [`CalibrationId`] never moving because no cohort had changed.
//! Three separate roads reached it: between two derivations, through a
//! [`Clone`] of the calibration's cohort (an owned copy, so the `&mut` on the
//! mint sealed nothing), and with no calibration in sight at all. Each round
//! sealed the road it had found and left the next, because the road was never
//! the defect — resolving a token from caller-owned state was. [`SpeakerToken`]
//! carries the three in full.
//!
//! What remains is not a mis-passed argument but a false statement:
//! `Enrolled::new(candidate_token, &probe)` *claims* the probe belongs to that
//! speaker, and `push(some_other_token, profile)` claims the same thing while
//! the cohort is being assembled. No type in a sans-I/O crate can refute
//! either — `coremlit` does not hold the library — but both are claims spelled
//! out at the point a caller reads a library record, not an argument slot two
//! positions along from the one that matters, and neither is an ANSWER this
//! crate gave. A probe has no such value to build, and that is the point: a
//! bare [`VoiceProfile`] reaches only [`Calibration::side`], the held-out door.
//!
//! ```compile_fail,E0308
//! use coremlit::audio::speaker::calibrate::{
//!   AsNormOptions, Calibration, LibraryCohortBuilder, Scoring,
//! };
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let mut cohort = LibraryCohortBuilder::new();
//! let alice = cohort.speaker();
//! cohort.push(alice, Scoring::Cosine.prepare(&raw).unwrap());
//! let calibration = Calibration::new(cohort, AsNormOptions::new());
//! // An unidentified probe: a prepared vector and no identity at all. There is
//! // nothing to exclude, so the excluding door must not accept it.
//! let probe = Scoring::Cosine.prepare(&raw).unwrap();
//! let _ = calibration.enrolled_side(&probe);
//! ```
//!
//! # The score source is structural, not re-checked
//!
//! [`Scoring`] used to be a tag carried on a raw score and on both statistics
//! and compared three ways at the last step. It no longer needs to be, because
//! the last step is no longer handed a number.
//!
//! A [`TrialSide`] exists only if its profile was scored against the cohort
//! and every entry it reached agreed with it
//! ([`CalibrateError::ScoringMismatch`] refuses the whole call otherwise, and
//! `diaric`'s own [`MIN_COHORT_SCORES`] floor refuses a side that reached too
//! few entries to have one). Two sides of one calibration therefore agree with
//! the cohort, and so with each other. Then [`Calibration::trial`] computes the
//! raw score from those two profiles, in the source they were prepared for,
//! rather than accepting one computed elsewhere.
//!
//! That is the failure a single check at score time did not cover, and it is
//! silent by construction: [`Scoring::Cosine`] cohort scores of `[-1, 1]` have
//! mean `0` and deviation `1`, so *any* finite [`Scoring::PldaCosine`] trial
//! score normalized against them comes back finite and plausible — one metric
//! calibrated by another, with no value out of range to notice. It is most
//! reachable exactly where this design pushes a caller: caching a
//! [`TrialSide`] per speaker while iterating over both score sources, which is
//! what #123's ask 3 is. A cached side carries its calibration, and a
//! calibration is over one cohort, whose entries are prepared for one source.
//!
//! # What leaves this module
//!
//! `diaric`'s own [`as_norm`](diaric::score_norm::as_norm) is arithmetic over
//! two unbound `CohortStats`, and this module re-exports neither it nor the
//! containers that produce one. Nor does it hand out the **operands**:
//!
//! - a [`TrialSide`] has no `mean` and no `deviation`, and no accessor to the
//!   `CohortStats` inside it. Its [`Debug`] is hand-written for the same
//!   reason — the derived one printed both numbers, which is the same
//!   disclosure through a different door;
//! - a [`VoiceProfile`] has no public way to produce a number at all. Scoring
//!   two profiles is private to this module;
//! - a **refusal** carries no arithmetic either. `diaric`'s own
//!   `DegenerateDeviation` carries the exact σ that failed the floor, and its
//!   `ZScoreCancellation` carries both z-scores and the value they cancelled
//!   to — so [`CalibrateError::ScoreNorm`] re-states them as
//!   [`ScoreNormRefusal`], which keeps the category, the floor that was
//!   breached and how many scores were selected out of how many considered,
//!   and carries no `f64` at all. This door needs no [`TrialSide`] to exist:
//!   a `min_deviation` set above what a perfectly valid cohort spreads refuses
//!   before any side is produced, which is how a σ left by a road the previous
//!   round did not look at;
//! - the only `f64`s this page emits are [`CalibratedTrial`]'s two, produced
//!   together by [`Calibration::trial`], which fixed all four of eq. (7)'s
//!   attributes before it computed either of them.
//!
//! [`ScoreNormRefusal`]: crate::audio::speaker::error::ScoreNormRefusal
//!
//! **This is a correction.** The version of this page that shipped published
//! `TrialScore::raw`, `SideStats::mean` and `SideStats::deviation` — every
//! operand of `0.5·((s−μₐ)/σₐ + (s−μᵦ)/σᵦ)` — and its derived `Debug` printed
//! the last two as well. So the cross-cohort normalization this door refuses
//! was a two-line function over values `coremlit` itself handed out, needing
//! no `diaric` dependency and no statistic of the caller's own. Calling that
//! pairing "a typed refusal rather than a caveat" was true of the door, and
//! the door was not the only way through.
//!
//! **And here is what this page does not claim.** A caller owns the embedding
//! rows they prepared their profiles from, and `diaric` is a published crate.
//! They can compute their own cosines, select their own top-N and evaluate
//! eq. (7) over two cohorts of their own — and, holding a [`Calibration`],
//! they can take a side per cohort member and read the raw scores back out of
//! [`CalibratedTrial::raw`] to rebuild a mean and a deviation. Nothing in a
//! library that does not hold the library can prevent arithmetic over the
//! caller's own numbers. **Cross-cohort normalization is not, and cannot be
//! made, structurally unavailable.**
//!
//! What is true is narrower, and is the whole claim: **no cohort statistic
//! `coremlit` computes reaches a caller — not through an accessor, not through
//! a [`Debug`], and not through a refusal.** There are exactly two such
//! numbers, μ and σ; they are enumerable, every door onto them is pinned one
//! at a time below, and that is what makes the claim checkable rather than
//! open-ended.
//!
//! It is deliberately *not* the broader "no operand of eq. (7) escapes", and
//! the reason is arithmetic rather than taste. [`CalibratedTrial::raw`] is
//! `s`, the formula's third operand, and it is published on purpose: #123's
//! own comparison is between the raw scores and the calibrated ones, so
//! withholding it would take the question with it. Publishing both a score and
//! its normalization publishes the affine map between them, and **that map
//! determines the two statistics behind it**:
//!
//! Write `aᵢ = 1/σᵢ` and `bᵢ = −μᵢ/σᵢ`, so a z-score is the affine `aᵢ·s + bᵢ`
//! and eq. (7) reads `c(i,j) = ½·[(aᵢ + aⱼ)·s(i,j) + bᵢ + bⱼ]`. A side
//! trialled against ITSELF gives `c(i,i) = aᵢ·s(i,i) + bᵢ`, which eliminates
//! every `bᵢ`; three sides of one calibration then leave three linear
//! equations in three unknowns, and the solve returns each side's σ and μ to
//! the last few digits — no accessor, no [`Debug`], no refusal, nothing but
//! [`CalibratedTrial::raw`] and [`CalibratedTrial::calibrated`].
//! `the_two_published_numbers_still_determine_every_side_that_produced_them`
//! carries it out and compares the result against the private field.
//!
//! **So that is the residual, stated once with its closed form rather than
//! left for a later round to find as a new "escaping operand".** Nothing short
//! of withholding one of the two published numbers closes it, and doing so
//! would end the comparison this door exists for. What the claim above is
//! worth is what it says: every door that HANDED a statistic over is shut, and
//! shut uniformly — the accessors, the derived [`Debug`], and the refusal —
//! so a caller who wants μ and σ has to solve for them rather than read them.
//! The road to a cross-cohort normalization starts where it always did:
//! building the statistic out of the caller's own numbers.
//!
//! The cohorts are `coremlit`'s own types — [`LibraryCohortBuilder`],
//! [`LibraryCohort`] and [`HeldOutCohort`] — and the `diaric` container inside
//! each is private, so
//! `diaric`'s generic [`Cohort<K, T>`](diaric::score_norm::Cohort) and its
//! public
//! [`stats_excluding`](diaric::score_norm::Cohort::stats_excluding) — whose
//! `T` is unconstrained, and so accepts a [`VoiceProfile`] like any other item
//! type — are not reachable through this surface:
//!
//! ```compile_fail,E0599
//! use coremlit::audio::speaker::calibrate::{
//!   AsNormOptions, LibraryCohortBuilder, Scoring, VoiceProfile,
//! };
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let mut cohort = LibraryCohortBuilder::new();
//! let alice = cohort.speaker();
//! let bob = cohort.speaker();
//! cohort.push(alice, Scoring::Cosine.prepare(&raw).unwrap());
//! cohort.push(bob, Scoring::Cosine.prepare(&raw).unwrap());
//! let probe = Scoring::Cosine.prepare(&raw).unwrap();
//! // `diaric`'s unbound statistic takes ANY key — the candidate's included —
//! // and returns a number with no calibration on it. It is not reachable from
//! // a cohort this crate hands out.
//! let _ = cohort.stats_excluding(
//!   &alice,
//!   &probe,
//!   |_a: &VoiceProfile, _b: &VoiceProfile| 0.0,
//!   &AsNormOptions::new(),
//! );
//! ```
//!
//! One `compile_fail` is not enough on its own, because that one only pins
//! that no PASSTHROUGH was added to the wrapper (and that it does not `Deref`
//! to what it wraps). The storage itself is pinned separately, once per value
//! that holds a `diaric` statistic or the container behind one — a field going
//! public is the regression each of these catches and the one above does not:
//!
//! ```compile_fail,E0616
//! use coremlit::audio::speaker::calibrate::LibraryCohortBuilder;
//! let cohort = LibraryCohortBuilder::new();
//! let _ = cohort.entries;
//! ```
//!
//! ```compile_fail,E0616
//! use coremlit::audio::speaker::calibrate::{LibraryCohort, LibraryCohortBuilder};
//! let cohort = LibraryCohort::from(LibraryCohortBuilder::new());
//! let _ = cohort.entries;
//! ```
//!
//! ```compile_fail,E0616
//! use coremlit::audio::speaker::calibrate::HeldOutCohort;
//! let cohort = HeldOutCohort::assuming_disjoint(Vec::new());
//! let _ = cohort.entries;
//! ```
//!
//! ```compile_fail,E0616
//! use coremlit::audio::speaker::calibrate::{
//!   AsNormOptions, Calibration, HeldOutCohort, Scoring,
//! };
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let profile = Scoring::Cosine.prepare(&raw).unwrap();
//! let calibration = Calibration::new(
//!   HeldOutCohort::assuming_disjoint(vec![profile; 4]),
//!   AsNormOptions::new(),
//! );
//! let side = calibration.side(&profile).unwrap();
//! let _ = side.stats;
//! ```
//!
//! The two operands are pinned one at a time as well, for the same reason one
//! shared pin is not enough — either accessor coming back reopens the road on
//! its own:
//!
//! ```compile_fail,E0599
//! use coremlit::audio::speaker::calibrate::{
//!   AsNormOptions, Calibration, HeldOutCohort, Scoring,
//! };
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let profile = Scoring::Cosine.prepare(&raw).unwrap();
//! let calibration = Calibration::new(
//!   HeldOutCohort::assuming_disjoint(vec![profile; 4]),
//!   AsNormOptions::new(),
//! );
//! let side = calibration.side(&profile).unwrap();
//! let _ = side.mean();
//! ```
//!
//! ```compile_fail,E0599
//! use coremlit::audio::speaker::calibrate::{
//!   AsNormOptions, Calibration, HeldOutCohort, Scoring,
//! };
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let profile = Scoring::Cosine.prepare(&raw).unwrap();
//! let calibration = Calibration::new(
//!   HeldOutCohort::assuming_disjoint(vec![profile; 4]),
//!   AsNormOptions::new(),
//! );
//! let side = calibration.side(&profile).unwrap();
//! let _ = side.deviation();
//! ```
//!
//! Scoring two profiles is private, so a raw similarity leaves this module
//! only inside a [`CalibratedTrial`]:
//!
//! ```compile_fail,E0624
//! use coremlit::audio::speaker::calibrate::Scoring;
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let a = Scoring::Cosine.prepare(&raw).unwrap();
//! let b = Scoring::Cosine.prepare(&raw).unwrap();
//! let _ = a.score(&b);
//! ```
//!
//! And neither product can be assembled by hand, which is what makes "a
//! calibration produced it" a fact rather than a convention:
//!
//! ```compile_fail,E0451
//! use coremlit::audio::speaker::calibrate::TrialSide;
//! let _ = TrialSide {
//!   profile: todo!(),
//!   stats: todo!(),
//!   calibration: todo!(),
//! };
//! ```
//!
//! ```compile_fail,E0451
//! use coremlit::audio::speaker::calibrate::CalibratedTrial;
//! let _ = CalibratedTrial {
//!   raw: todo!(),
//!   calibrated: todo!(),
//!   scoring: todo!(),
//! };
//! ```
//!
//! # The score sources
//!
//! [`Scoring`] names them. Both are cosines; they differ in the space.
//!
//! ## `Cosine` — what the clusterers actually compare with
//!
//! Cosine over L2-normalized raw WeSpeaker embeddings, computed by calling
//! [`diaric::embed::cosine_similarity`] rather than reimplementing it, so a
//! threshold read off one transfers to the other **bit-exactly**
//! (`a_cosine_trial_score_is_bit_identical_to_diarics_own_cosine`).
//!
//! This is the similarity BOTH clustering backends compare with:
//!
//! - the online matcher scores raw cosine over
//!   [`Embedding`] directly, with no PLDA
//!   ([`diaric::cluster::online`]);
//! - the offline pipeline's linkage stage, [`diaric::cluster::ahc::ahc_init`],
//!   L2-normalizes the raw 256-d rows and takes Euclidean distances over them.
//!   On unit vectors `d² = 2 − 2·cos`, so its dendrogram cut is a cosine
//!   threshold wearing a different unit.
//!
//! ## `PldaCosine` — the offline pipeline's projection, scored pairwise
//!
//! [`diaric::plda::PldaTransform::project`] maps a raw row into the frozen
//! community-1 128-d PLDA space (LDA whitening, then rotation onto the
//! descending eigenvectors); this door L2-normalizes that vector and takes a
//! cosine there. The projection is not invented here — it is the same
//! process-wide transform [`extract`](crate::audio::speaker::extract) already
//! caches, the same one
//! [`diarize`](crate::audio::speaker::extract::Extraction::diarize) is handed.
//!
//! **The honesty boundary, stated rather than left to be discovered.** The
//! offline pipeline does not score *pairs* in that space. Its PLDA features
//! feed `vbx_iterate` as a variational-Bayes feature matrix alongside the
//! across-class covariance diagonal `phi` — a set-level EM over every
//! embedding at once, which never emits a trial score. Pairwise cosine in the
//! PLDA space is therefore **this door's own construction**: a second score
//! source in a different dimension and a different metric space, which no DER
//! gate and no parity oracle in this repository covers. What is measured here
//! is only that the two disagree numerically on the same rows
//! (`both_score_sources_are_reachable_and_are_not_the_same_number`); whether
//! they RANK a library differently, and which ranks it better, is the question
//! ask 3 exists to answer and nothing on this page claims an answer to it. It
//! is offered because AS-Norm is generic
//! over the score source precisely so the choice does not have to be made here
//! — and because #123's own follow-up, a library-scale confusion experiment,
//! is a comparison *between* score sources and needs more than one. Treat
//! [`Scoring::Cosine`] as the validated default and [`Scoring::PldaCosine`] as
//! characterized, not validated, until that experiment says otherwise.
//!
//! **Feed it RAW rows.** `PldaTransform` is calibrated for the unnormalized
//! WeSpeaker distribution (norm typically `0.5..7`), and
//! [`diaric::plda::RawEmbedding::from_wespeaker`] says so: handing it an
//! already-L2-normalized vector drifts the projection off the captured pyannote
//! distribution. A centroid must therefore be averaged from **raw** rows, not
//! from normalized ones, if it is going to be scored under
//! [`Scoring::PldaCosine`]. [`Scoring::Cosine`] does not care — it normalizes
//! anyway.
//!
//! # What a profile is made of
//!
//! One raw 256-d WeSpeaker row: [`EMBEDDING_DIM`] `f32`s, exactly what
//! [`Extraction::raw_embeddings`] slices into and what
//! [`Extractor::extract_chunk_embeddings`] returns per slot. No existing type
//! changed to make this door fit — a caller who has already computed
//! cluster-centroid embeddings hands over the centroid, and a caller who has
//! not hands over a row.
//!
//! [`EMBEDDING_DIM`]: crate::audio::speaker::embed::EMBEDDING_DIM
//! [`Extraction::raw_embeddings`]: crate::audio::speaker::extract::Extraction::raw_embeddings
//! [`Extractor::extract_chunk_embeddings`]: crate::audio::speaker::extract::Extractor::extract_chunk_embeddings
//!
//! # Cost: prepare once, and take each side once
//!
//! Both halves matter, and they are why the surface is three steps rather than
//! one convenience call.
//!
//! [`Scoring::prepare`] is where the per-row work happens — an L2 normalization
//! for [`Scoring::Cosine`], and for [`Scoring::PldaCosine`] a two-stage
//! projection measured at ~8.6 µs a row (see
//! [`extract`](crate::audio::speaker::extract)'s shared-transform doc). A
//! [`VoiceProfile`] is prepared once and scored many times.
//!
//! A [`TrialSide`] depends only on its own profile and its own calibration,
//! never on the other side of the trial, so it is taken once and reused across
//! every trial that profile appears in. **That holds for both sides here**,
//! and it is what the shape above buys:
//!
//! | side | the calibration's cohort | the statistic depends on | reused across |
//! |---|---|---|---|
//! | enrolled | held-out (nothing to exclude), or library-sampled with its own token excluded | that speaker and the calibration | every trial of that calibration the speaker appears in |
//! | probe | held-out only | that probe and the calibration | every candidate of that calibration the probe is scored against |
//!
//! A cached [`TrialSide`] is reusable across trials, not across calibrations.
//! It carries the [`CalibrationId`] it was taken under, so pairing it with a
//! side from a later-built calibration is a refusal rather than a silent
//! mismatch — which is exactly the shape a per-speaker cache produces once a
//! caller rebuilds their cohort or changes their [`AsNormOptions`].
//!
//! `N` enrolled speakers against `M` probes over a cohort of `C` therefore
//! costs `(N + M)·C` cohort scores and `N·M` trial scores, not `N·M·C` — for
//! 1 000 library profiles, 100 probes and a 300-member cohort, 330 000 cohort
//! scores rather than 30 million. [`diaric::score_norm`]'s cost table has the
//! derivation.
//!
//! **This is a correction, not a restatement.** The version of this page that
//! shipped claimed the same `N·C` reuse while telling a caller to exclude the
//! candidate's identity from the probe's side — which makes the probe's
//! statistic trial-dependent, so it has to be recomputed per candidate and the
//! claim was false by `M` for exactly the road the page recommended. The claim
//! is true again because the probe's side no longer has a candidate in it.
//!
//! **Where it does not hold**: a probe against a library-sampled cohort. There
//! is no entrypoint for that pairing, because there is no correct statistic to
//! return — see above — and there is no half-measure either, now that a trial
//! is one calibration's product. Taking the enrolment side over the library
//! and the probe side over a held-out cohort is exactly the ranking-reversing
//! pairing above, so a caller with no held-out cohort has to assemble one and
//! calibrate BOTH sides over it before an unidentified probe can be normalized
//! at all. That is a real restriction on what this door can do, and it is
//! stated here rather than papered over with a default that would be wrong in
//! a way nothing could detect.
//!
//! # End to end
//!
//! ```
//! use std::collections::HashMap;
//!
//! use coremlit::audio::speaker::{
//!   calibrate::{
//!     AsNormOptions, Calibration, Enrolled, HeldOutCohort, LibraryCohortBuilder, Scoring,
//!     SpeakerToken,
//!   },
//!   embed::EMBEDDING_DIM,
//! };
//!
//! # fn stored_centroid(seed: usize) -> Vec<f32> {
//! #   // Stand-in for a real cluster centroid: raw, unnormalized rows with
//! #   // enough spread that the cohort scores actually vary. A real caller
//! #   // passes whatever their library holds.
//! #   (0..EMBEDDING_DIM)
//! #     .map(|i| (((i * 37 + seed * 101) % 97) as f32) / 97.0 - 0.5)
//! #     .collect()
//! # }
//! # fn probe_of(seed: usize) -> Vec<f32> {
//! #   let mut v = stored_centroid(seed);
//! #   v[3] += 0.05;
//! #   v
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // The caller's library key. It stays the caller's: this door names a
//! // speaker with a `SpeakerToken` it minted, and never with a value of
//! // theirs.
//! type PersonId = u32;
//!
//! let scoring = Scoring::Cosine;
//! let options = AsNormOptions::new();
//!
//! // 1. The impostor cohort. The caller asserts these profiles are nobody in
//! //    the library being scored — a fact about where they came from, which
//! //    is knowable without knowing who any probe is.
//! let held_out = HeldOutCohort::assuming_disjoint(
//!   (0..64)
//!     .map(|i| scoring.prepare(&stored_centroid(10_000 + i)))
//!     .collect::<Result<Vec<_>, _>>()?,
//! );
//!
//! // 2. One calibration. It takes the cohort and the configuration by value,
//! //    and from here on it is the only thing that produces a side or a
//! //    trial — so no two of them can disagree about either.
//! let over_held_out = Calibration::new(held_out, options);
//!
//! // 3. The trial: a library profile the caller can name, against a probe
//! //    from a recording that has just been diarized — whose speaker is the
//! //    thing being looked up, not something the caller can hand over.
//! let alice: PersonId = 7;
//! let stored = scoring.prepare(&stored_centroid(alice as usize))?;
//! let probe = scoring.prepare(&probe_of(alice as usize))?;
//!
//! // 4. One side per profile, both reusable. Neither depends on the other
//! //    side of the trial.
//! let stored_side = over_held_out.side(&stored)?;
//! let probe_side = over_held_out.side(&probe)?;
//! assert_eq!(probe_side.considered(), 64); // nothing to exclude
//!
//! // 5. The trial itself. It is handed no score: it computes the raw one from
//! //    the two sides' own profiles and returns it beside the calibrated
//! //    number a fixed threshold reads.
//! let trial = over_held_out.trial(&stored_side, &probe_side)?;
//! assert!(trial.raw() <= 1.0);
//! assert!(trial.calibrated().is_finite());
//!
//! // The other cohort shape: sampled from the library itself, so it CONTAINS
//! // the speakers being scored. Only NAMED sides may use it — the identity
//! // travels with the profile, so no other speaker's entries can be dropped by
//! // mistake — and each side drops its own.
//! let bob: PersonId = 11;
//! let mut library = LibraryCohortBuilder::new();
//! // The map from the caller's own key to this cohort's identity lives with
//! // the library, because that is whose it is. `speaker` takes no argument, so
//! // nothing the caller does to their own keys afterwards — and no duplicate
//! // of this cohort — can change what a token names.
//! let mut tokens: HashMap<PersonId, SpeakerToken> = HashMap::new();
//! for id in 0..64u32 {
//!   let token = *tokens.entry(id).or_insert_with(|| library.speaker());
//!   library.push(token, scoring.prepare(&stored_centroid(id as usize))?);
//! }
//! let over_library = Calibration::new(library, options);
//! let bobs_profile = scoring.prepare(&stored_centroid(bob as usize))?;
//!
//! let alice_side = over_library.enrolled_side(Enrolled::new(tokens[&alice], &stored))?;
//! let bobs_side = over_library.enrolled_side(Enrolled::new(tokens[&bob], &bobs_profile))?;
//! assert_eq!(alice_side.considered(), 63); // 64 members, less Alice's own
//! let merged = over_library.trial(&alice_side, &bobs_side)?;
//! assert!(merged.calibrated().is_finite());
//!
//! // What no longer normalizes: a side from one calibration in another's
//! // trial. Both are valid statistics and both are `Cosine`, and averaging
//! // their z-scores can invert the candidate order — so it is refused from
//! // either end, rather than the caller having to know.
//! assert!(over_library.trial(&alice_side, &probe_side).is_err());
//! assert!(over_held_out.trial(&alice_side, &probe_side).is_err());
//! # Ok(())
//! # }
//! ```
//!
//! # Not implemented here
//!
//! - **Cohort selection.** Which profiles belong in a cohort, and how many, is
//!   a policy over a library this crate does not hold. Matějka et al. 2017 §4.1
//!   assumes one entry per speaker;
//!   [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding)
//!   does not require it, and neither does this door.
//! - **A threshold.** #123's ask 3 — a library-scale confusion experiment
//!   against the WeSpeaker baseline — is what produces one, and it needs data
//!   this repository does not have. Nothing here publishes a default.
//! - **AS-Norm2**, the crossed variant. See [`diaric::score_norm`]; it is not
//!   implemented there either, and for a stated reason.
//! - **A third score source.** A [`Calibration`] is built over a cohort of
//!   [`VoiceProfile`]s, so it covers the sources [`Scoring`] names and no
//!   others — it exists to carry a `coremlit` refusal out of an infallible
//!   closure and to hold eq. (7)'s four attributes together, neither of which
//!   a caller's own scoring function needs. Genericity is not lost by that,
//!   but it is not *re-exported*: a caller scoring in some other space takes a
//!   `diaric` dependency and fills `diaric`'s own `Cohort<K, T>` with THEIR
//!   item type. "What leaves this module" says why handing the same container
//!   out from here — already able to hold a [`VoiceProfile`] — was a defect
//!   rather than a convenience.

use core::sync::atomic::{AtomicU64, Ordering};

use diaric::{
  embed::Embedding,
  plda::{PLDA_DIMENSION, RawEmbedding},
  score_norm::{Cohort as DiaricCohort, CohortEntry, CohortStats},
};

use crate::audio::speaker::{
  embed::EMBEDDING_DIM,
  error::{CalibrateError, CalibrationMismatch, ProfileLength, ScoreNormRefusal, ScoringMismatch},
  extract::shared_plda_transform,
};

/// `diaric`'s AS-Norm **configuration** vocabulary, re-exported so a caller of
/// this door does not need a direct `diaric` dependency to name what it takes.
///
/// These are `diaric`'s own, unchanged: [`AsNormOptions`] tunes the per-side
/// statistics and the four constants are the numbers its defaults and its
/// floors are read against. A `coremlit`-side mirror of [`AsNormOptions`] (the
/// way [`OnlineOptions`](crate::audio::speaker::OnlineOptions) mirrors
/// `diaric`'s online configuration) would be two constants and two builders
/// with nowhere to drift to but out of step, so there is none.
///
/// **`diaric`'s CONTAINERS and its arithmetic are deliberately not among
/// these** — not `as_norm`, and not `Cohort`, `CohortEntry` or `CohortStats`
/// either. The free `as_norm` takes two unbound statistics and cannot tell one
/// score source, cohort or subject from another; the generic `Cohort<K, T>`
/// accepts a [`VoiceProfile`] as its `T` and hands out a candidate-dependent,
/// unbound statistic over one. Both are reachable from `diaric` itself, and
/// neither is reachable from here — the module docs' "What leaves this module"
/// has the worked case and the boundary.
///
/// One consequence, stated because it is not visible from the signatures:
/// [`AsNormOptions`]'s `serde` impls are `diaric`'s own and sit behind
/// `diaric`'s `serde` feature, which this crate does NOT enable — `coremlit`'s
/// `serde` feature covers `coremlit`'s types, [`Scoring`] included. A caller
/// serializing an AS-Norm configuration enables `serde` on their own `diaric`
/// dependency; feature unification then applies it here.
pub use diaric::score_norm::{
  AsNormOptions, DEFAULT_MIN_DEVIATION, DEFAULT_TOP_N, MAX_NORMALIZED_ERROR, MIN_COHORT_SCORES,
};

/// Which score source a [`VoiceProfile`] is prepared for.
///
/// Both are cosines; they differ in the space the cosine is taken in, and so in
/// what a threshold over them means. The module docs' "The score sources" says
/// what each one is, and which of the two the clustering backends actually
/// compare with.
///
/// `#[non_exhaustive]` because a third source is exactly what #123's confusion
/// experiment could ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum Scoring {
  /// Cosine over L2-normalized raw WeSpeaker embeddings —
  /// [`diaric::embed::cosine_similarity`], called rather than reimplemented.
  ///
  /// The similarity both clustering backends compare with, and the validated
  /// default.
  Cosine,
  /// Cosine in the frozen community-1 PLDA-projected 128-d space.
  ///
  /// The projection is the offline pipeline's own
  /// ([`diaric::plda::PldaTransform::project`]); **scoring pairs by cosine in
  /// that space is this door's construction, not the pipeline's** — the
  /// pipeline hands those features to VBx, which is a set-level EM and emits no
  /// trial score. Characterized, not validated: no DER gate covers it. Needs
  /// RAW, unnormalized rows (module docs).
  PldaCosine,
}

impl Scoring {
  /// Prepare one raw WeSpeaker row for this score source.
  ///
  /// `raw` is a single unnormalized 256-d row —
  /// [`EMBEDDING_DIM`] `f32`s: a
  /// row out of
  /// [`Extraction::raw_embeddings`](crate::audio::speaker::extract::Extraction::raw_embeddings),
  /// a slot out of
  /// [`Extractor::extract_chunk_embeddings`](crate::audio::speaker::extract::Extractor::extract_chunk_embeddings),
  /// or a cluster centroid the caller averaged from either. Preparing is the
  /// per-row cost (module docs); a profile is prepared once and scored many
  /// times.
  ///
  /// A slice rather than a `[f32; EMBEDDING_DIM]` because that is the shape a
  /// caller actually holds, and the array form only moves the same length check
  /// to their `try_into`. The two dimensions are still pinned together: the
  /// array this builds is handed to `diaric`'s array-typed constructors, so
  /// `EMBEDDING_DIM` disagreeing with `diaric`'s own dimension is a compile
  /// error here rather than a runtime one.
  ///
  /// # Errors
  ///
  /// - [`CalibrateError::ProfileLength`] if `raw` is not exactly
  ///   `EMBEDDING_DIM` elements. Never truncated, never padded — a short row is
  ///   a caller bug, and silently completing it would produce a profile for a
  ///   speaker who does not exist.
  /// - [`CalibrateError::DegenerateProfile`] if the prepared vector has no
  ///   usable direction.
  /// - [`CalibrateError::Plda`] and
  ///   [`CalibrateError::PldaTransformUnavailable`] for
  ///   [`Scoring::PldaCosine`] only, from `diaric`'s projection and from the
  ///   shared transform it needs.
  pub fn prepare(self, raw: &[f32]) -> Result<VoiceProfile, CalibrateError> {
    if raw.len() != EMBEDDING_DIM {
      return Err(CalibrateError::ProfileLength(ProfileLength::new(
        raw.len(),
        EMBEDDING_DIM,
      )));
    }
    let mut row = [0.0f32; EMBEDDING_DIM];
    row.copy_from_slice(raw);

    match self {
      // `normalize_from` owns the floor; this door does not re-derive it. Same
      // discipline `extract::PLDA_MIN_NORM`'s doc describes, for the same
      // reason: two spellings of one threshold drift apart.
      Self::Cosine => Embedding::normalize_from(row)
        .map(|e| VoiceProfile(Prepared::Cosine(e)))
        .ok_or(CalibrateError::DegenerateProfile(Self::Cosine)),

      Self::PldaCosine => {
        // Resolved before the projection for the reason
        // `Extractor::extract_chunk_embeddings` resolves it before inference:
        // an unavailable transform must refuse the call outright rather than
        // surface halfway through it.
        let plda = shared_plda_transform().map_err(|_| CalibrateError::PldaTransformUnavailable)?;
        let projected = plda.project(&RawEmbedding::from_wespeaker(row)?)?;
        Ok(VoiceProfile(Prepared::PldaCosine(unit_vector(projected)?)))
      }
    }
  }
}

/// L2-normalize a projected PLDA vector, so scoring is a dot product and every
/// trial score is bounded without a division per pair.
///
/// # There is deliberately no floor above zero
///
/// `diaric` publishes none for the 128-d PLDA space: `NORM_EPSILON` is
/// calibrated for the 256-d WeSpeaker one and `RAW_EMBEDDING_MIN_NORM` for its
/// input. A third constant invented here would refuse real projections on a
/// number nothing measured — the "fabricated variance" failure
/// [`diaric::score_norm`] records from its own review history. What is refused
/// instead is a vector that genuinely has no direction, or one whose
/// normalization leaves f64's range: `‖v‖` zero or non-finite, or a component
/// non-finite afterwards.
///
/// The `Σv²` overflow the second check would catch is not reachable from
/// [`diaric::plda::PldaTransform::project`], whose stage-1 output
/// [`PostXvecEmbedding`](diaric::plda::PostXvecEmbedding) has norm exactly
/// `sqrt(128)` by construction and whose stage 2 is a fixed rotation by shipped
/// weights. It is checked anyway: "cannot happen" is a claim about today's
/// weights, and this is a refusal rather than an assertion.
fn unit_vector(v: [f64; PLDA_DIMENSION]) -> Result<[f64; PLDA_DIMENSION], CalibrateError> {
  let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
  if !norm.is_finite() || norm <= 0.0 {
    return Err(CalibrateError::DegenerateProfile(Scoring::PldaCosine));
  }
  let mut out = [0.0f64; PLDA_DIMENSION];
  for (o, x) in out.iter_mut().zip(v.iter()) {
    *o = x / norm;
  }
  if !out.iter().all(|x| x.is_finite()) {
    return Err(CalibrateError::DegenerateProfile(Scoring::PldaCosine));
  }
  Ok(out)
}

/// One speaker's voice profile, prepared for a chosen [`Scoring`] and ready to
/// be calibrated.
///
/// Build one with [`Scoring::prepare`]. A profile is a *prepared vector*, not a
/// stored one: it holds no identity, no provenance and no recording. The
/// caller's library holds those, and binds one back to a profile with
/// [`Enrolled`] when — and only when — the speaker is known.
///
/// A bare `VoiceProfile` is therefore exactly what an unidentified probe is,
/// and it reaches only [`Calibration::side`].
///
/// **No number comes out of a profile.** Scoring two of them is private to
/// this module: a raw similarity leaves only inside a [`CalibratedTrial`],
/// beside the calibrated number, and only after a [`Calibration`] has fixed
/// everything eq. (7) needs the two to agree on. The module docs' "What leaves
/// this module" says what that closes and, precisely, what it does not.
///
/// # Why the score source is a tag and not a type parameter
///
/// Mixing sources is refused at runtime ([`CalibrateError::ScoringMismatch`])
/// rather than made unrepresentable by a `VoiceProfile<S>`. Two reasons, the
/// second deciding:
///
/// - a type parameter here would spread onto every container that holds a
///   profile — [`LibraryCohortBuilder`], [`LibraryCohort`], [`HeldOutCohort`],
///   [`Calibration`] — so the caller's whole library would be monomorphized on
///   a choice they naturally make at run time: which score source to run *this*
///   comparison in;
/// - #123's follow-up is a confusion experiment *between* score sources, whose
///   natural shape is a loop over `[Scoring::Cosine, Scoring::PldaCosine]`. A
///   type parameter turns that loop into duplicated generic code.
///
/// What the tag buys in exchange is that a cross-source result is a typed
/// refusal rather than a finite number. An earlier version of this paragraph
/// claimed that on the ground that "every path that reads two profiles
/// compares the tags first", which was true and not sufficient, because the
/// last step then read no profiles at all. It reads them again: a trial is
/// computed from the two sides' own profiles, so the source they agree on is
/// the source the number is in — see the module docs' "The score source is
/// structural, not re-checked".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceProfile(Prepared);

/// The prepared vector, one shape per [`Scoring`]. Private: these shapes
/// implement the score sources; they are not a vocabulary a caller needs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Prepared {
  /// L2-normalized 256-d, carried as `diaric`'s own type so scoring can call
  /// `diaric`'s own cosine.
  Cosine(Embedding),
  /// L2-normalized 128-d PLDA projection.
  PldaCosine([f64; PLDA_DIMENSION]),
}

impl VoiceProfile {
  /// The score source this profile was prepared for.
  pub const fn scoring(&self) -> Scoring {
    match self.0 {
      Prepared::Cosine(_) => Scoring::Cosine,
      Prepared::PldaCosine(_) => Scoring::PldaCosine,
    }
  }

  /// The raw trial score between this profile and `other`.
  ///
  /// **Private, and that is the point.** A raw similarity with nothing bound
  /// to it is exactly the loose operand three rounds of review kept finding a
  /// use for; it reaches a caller only through [`CalibratedTrial::raw`], which
  /// a [`Calibration`] produced. Inside this module it is called twice: once
  /// per cohort entry when a side is taken, and once per trial.
  ///
  /// Both sources are cosines between unit vectors, so the result is finite and
  /// lands in `[-1, 1]` to within the rounding of the normalization that
  /// produced them — Cauchy-Schwarz bounds the exact dot product by the product
  /// of the norms, and those are `1` by construction. That normalization
  /// happened once, at [`Scoring::prepare`], rather than once per trial.
  ///
  /// # Errors
  ///
  /// [`CalibrateError::ScoringMismatch`] if the two profiles were prepared for
  /// different score sources. That is the only way this can fail.
  fn score(&self, other: &Self) -> Result<f64, CalibrateError> {
    match (&self.0, &other.0) {
      // Calls `diaric`'s cosine rather than repeating its loop, so this score
      // is bit-identical to the one the online clusterer matches on.
      (Prepared::Cosine(a), Prepared::Cosine(b)) => {
        Ok(f64::from(diaric::embed::cosine_similarity(a, b)))
      }
      (Prepared::PldaCosine(a), Prepared::PldaCosine(b)) => {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
      }
      _ => Err(CalibrateError::ScoringMismatch(ScoringMismatch::new(
        self.scoring(),
        other.scoring(),
      ))),
    }
  }
}

/// An opaque handle to one speaker of one [`LibraryCohortBuilder`] — what an
/// enrolled side is excluded by, and the only name a speaker has here.
///
/// Minted by [`LibraryCohortBuilder::speaker`], which is the only way to obtain
/// one. There is no constructor, no accessor to the number inside, and nothing
/// a caller can write down; all it can do is compare equal to itself, which is
/// all an exclusion needs.
///
/// # A token is MINTED, never resolved
///
/// [`LibraryCohortBuilder::speaker`] **takes no argument**. Nothing on this page
/// maps a value the caller owns to a `SpeakerToken`, so there is no question
/// that can be asked twice and answered differently — and a token therefore
/// names the same speaker for as long as it exists, whoever holds it and
/// however the value holding it was arrived at.
///
/// That is the whole property, and it is what a cohort keyed by the caller's
/// own `K` could not buy. [`Eq`] does not forbid interior mutability:
/// `Rc<Cell<u64>>` is a perfectly good `Eq` key whose comparison reads a cell
/// the caller can still write to. Rewriting that cell so one speaker's key
/// equals another's made a *lookup for the second speaker hand back the first
/// one's token*; the side taken under it then dropped the wrong speaker's
/// entries and left the subject's own in place, which is the
/// self-contamination [`diaric::score_norm`] names — on finite, plausible
/// numbers, with the [`CalibrationId`] never moving because no cohort had
/// changed.
///
/// Three roads reached that one answer, and each round of review sealed one:
///
/// - rewriting the cell between two derivations, back when the cohort compared
///   `K` at every side;
/// - rewriting it after the cohort had become a [`Calibration`], and resolving
///   through a [`Clone`] of the calibration's cohort — an owned, and therefore
///   mutable, copy carrying the original tokens. Lending the cohort out shared
///   seals `&mut`, and `Clone` does not go through a borrow;
/// - rewriting it with no calibration in sight, while the cohort was still
///   open, or through a copy of the cohort kept aside before it was frozen.
///
/// Sealing a road leaves the next one, because the road was never the defect:
/// the defect was that a token could be *resolved* from caller-owned state at
/// all. There is no `K` on this surface now, at any point in a cohort's life,
/// so there is nothing to resolve — cloned, borrowed, serialized or otherwise.
/// What the caller keeps instead is the map from their own library key to the
/// token this minted, which is the map their library already is.
///
/// This token's OWN [`Eq`] is the one comparison an exclusion still makes, and
/// it is safe for the reason an arbitrary `K` was not: it compares a private
/// `u64` with no accessor, no constructor and no interior mutability, so it
/// cannot answer differently on a second call. That is the difference between
/// closing the class and moving it.
///
/// Tokens come from a process-wide counter, so one never collides with a token
/// another cohort minted; [`Calibration::enrolled_side`] refuses a token its
/// own cohort does not hold ([`CalibrateError::ForeignSpeaker`]) rather than
/// silently excluding nothing.
///
/// It is deliberately **not** `serde`, and neither cohort is. A `Deserialize`
/// would be a constructor from data — the one remaining shape that could put a
/// token this process never minted beside a profile, or rebuild a cohort whose
/// token-to-entry map nothing here decided. A caller persists their own library
/// key and re-mints on load, which is the same work assembling a cohort already
/// is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerToken(u64);

impl SpeakerToken {
  /// The next token, from a counter of its own.
  ///
  /// `Relaxed`, and unguarded against wrapping, for the reasons
  /// [`CalibrationId::mint`] states: a `fetch_add` hands every caller a
  /// distinct previous value under any ordering, this counter guards no other
  /// memory, and `2^64` mints is not a number a program reaches.
  fn mint() -> Self {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    Self(NEXT.fetch_add(1, Ordering::Relaxed))
  }
}

/// A [`VoiceProfile`] bound to the identity it belongs to.
///
/// The one value [`Calibration::enrolled_side`] takes in place of a speaker
/// *and* a side, so no argument is left that could name a different speaker
/// than the profile belongs to. The module docs' "The token travels with the
/// profile" says what that removes and what it cannot.
///
/// The identity half is a [`SpeakerToken`] the cohort minted — see that type
/// for why it is not, and can never again be, anything of the caller's. The
/// profile half stays a borrow: a caller's library already holds it and a
/// [`VoiceProfile`] is a kilobyte of prepared vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Enrolled<'a> {
  /// Whose profile this is, as this cohort's own immutable handle.
  speaker: SpeakerToken,
  /// The prepared vector itself.
  profile: &'a VoiceProfile,
}

impl<'a> Enrolled<'a> {
  /// Bind a profile to the speaker it belongs to.
  ///
  /// This is an assertion about identity, and the only one in this module: it
  /// says `profile` is material from the speaker `speaker` names, which is
  /// what makes dropping that speaker's cohort entries the right thing to do.
  /// It is answerable for a library record and it is *not* answerable for a
  /// probe — so a probe never gets one, having no token to bind.
  pub const fn new(speaker: SpeakerToken, profile: &'a VoiceProfile) -> Self {
    Self { speaker, profile }
  }

  /// Whose profile this is.
  pub const fn speaker(&self) -> SpeakerToken {
    self.speaker
  }

  /// The prepared vector itself — how an enrolled speaker reaches a
  /// [`Calibration`] or a cohort.
  pub const fn profile(&self) -> &VoiceProfile {
    self.profile
  }
}

/// Where a library-sampled cohort is assembled: speakers are minted here, and
/// profiles filed under them.
///
/// [`Calibration::new`] takes one of these **by value** and freezes it into a
/// [`LibraryCohort`], which is what the calibration holds and the only cohort
/// it will ever lend out. The split is the design and not a convenience: the
/// frozen value has no mutator on it at all, so "nothing can change under a
/// side already taken" is a property of the type rather than a property of who
/// holds which borrow — and a borrow is what [`Clone`] goes around.
///
/// **Identity is minted, never resolved.** [`speaker`](Self::speaker) takes no
/// argument and reads nothing of the caller's; [`push`](Self::push) files a
/// profile under a token the caller already holds. There is no key type here,
/// so nothing a caller owns takes part in deciding which speaker a token names
/// — before the freeze or after it. [`SpeakerToken`] carries the three roads
/// that closes, and why sealing them one at a time did not.
///
/// The caller's own library key stays the caller's: a `HashMap<TheirId,
/// SpeakerToken>` filled while this cohort is assembled is the whole of what a
/// key type used to buy, and it lives where the library does.
#[derive(Debug, Clone, PartialEq)]
pub struct LibraryCohortBuilder {
  /// One entry per speaker this cohort knows: minted by
  /// [`speaker`](Self::speaker), or adopted by [`push`](Self::push).
  ///
  /// It answers the question `entries` cannot — whether a token handed to
  /// [`Calibration::enrolled_side`] is one this cohort holds — for a speaker
  /// the cohort has no profiles of, whose exclusion set is legitimately empty.
  speakers: Vec<SpeakerToken>,
  /// `diaric`'s own container, private, keyed by the token.
  entries: DiaricCohort<SpeakerToken, VoiceProfile>,
}

impl Default for LibraryCohortBuilder {
  /// An empty cohort.
  fn default() -> Self {
    Self::new()
  }
}

impl LibraryCohortBuilder {
  /// An empty cohort.
  pub fn new() -> Self {
    Self {
      speakers: Vec::new(),
      entries: DiaricCohort::new(),
    }
  }

  /// Mint a token for one speaker of this cohort.
  ///
  /// **It takes no argument**, which is the point: a token is not derived from
  /// anything, so there is no input whose later value could re-decide the
  /// answer. Call it once per distinct speaker and keep the token beside that
  /// speaker's own library key; calling it twice mints two speakers, which is
  /// what two calls asked for.
  ///
  /// A speaker with no profiles pushed under it is not a mistake — it names an
  /// empty exclusion set, which is the right answer for an enrolled speaker who
  /// is simply not among the impostors.
  pub fn speaker(&mut self) -> SpeakerToken {
    let token = SpeakerToken::mint();
    self.speakers.push(token);
    token
  }

  /// File one library profile under the speaker it belongs to.
  ///
  /// Every profile pushed under one token is dropped together by
  /// [`Calibration::enrolled_side`], so a speaker's second recording goes under
  /// the same token their first did.
  ///
  /// A token this cohort has not seen is **adopted**: it becomes one of this
  /// cohort's speakers, naming exactly what is filed under it here. That is the
  /// only sound reading — a token is a bare identity, so one minted from
  /// another cohort names, in this one, precisely the entries this cohort was
  /// given for it. Refusing instead would turn a caller reusing one identity
  /// across two cohorts into a [`CalibrateError::ForeignSpeaker`] on a cohort
  /// that genuinely holds that speaker, which is an exclusion lost.
  ///
  /// Like [`Enrolled::new`], this is an **assertion**: it says the profile is
  /// material from the speaker the token names. A sans-I/O crate cannot refute
  /// that, and it is the same irreducible residual stated there. What is no
  /// longer possible is the other thing — a *query* of this crate's handing
  /// back a token that names somebody else.
  pub fn push(&mut self, speaker: SpeakerToken, profile: VoiceProfile) {
    if !self.speakers.contains(&speaker) {
      self.speakers.push(speaker);
    }
    self.entries.push(speaker, profile);
  }

  /// Number of cohort members, before any exclusion.
  pub fn len(&self) -> usize {
    self.entries.len()
  }

  /// Whether the cohort holds no members. An empty one is not a refusal here;
  /// it becomes [`CalibrateError::ScoreNorm`] at
  /// [`Calibration::enrolled_side`], where `diaric`'s own floor lives.
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }
}

/// A frozen cohort sampled from the caller's own library, whose members carry
/// the identities that name them.
///
/// #123's cohort shape, and the one [`Calibration::enrolled_side`] needs.
/// Because it may hold an enrolled speaker's own entries, every side over it
/// drops that speaker's entries by identity — which is correct for a side whose
/// speaker is NAMED and impossible for one whose speaker is what the trial is
/// trying to discover. A probe therefore has no door onto this type at all; it
/// goes to [`HeldOutCohort`].
///
/// **It holds token membership and token-keyed entries, and nothing else.**
/// There is no caller key in it, so no lookup of the caller's can be answered
/// against it — which is not a statement about what it lends out. A [`Clone`]
/// of it, a [`Clone`] of the [`Calibration`] around it, or any other duplicate
/// is the same immutable pair of lists: nothing to mutate and nothing to
/// resolve. [`SpeakerToken`] has the case that shape closes, and the three
/// separate roads that reached it while a key was still here.
///
/// **Built only by freezing a [`LibraryCohortBuilder`]**, and it has no mutator,
/// so it cannot grow. [`Calibration::new`] does the freeze; growing means
/// assembling a new cohort and a new calibration, whose sides do not pair with
/// the old ones — the refusal a silently grown cohort would otherwise skip.
///
/// **The `diaric` container inside is private, and that is the point.**
/// `diaric`'s [`Cohort<K, T>`](diaric::score_norm::Cohort) is generic in `T`
/// and carries a public
/// [`stats_excluding`](diaric::score_norm::Cohort::stats_excluding), so
/// re-exporting it — which this module did — handed a caller a way to compute
/// an unbound, candidate-dependent statistic over a [`VoiceProfile`] through
/// `coremlit`'s own surface. Wrapping it means a [`Calibration`] is this
/// crate's only statistic constructor over a profile.
#[derive(Debug, Clone, PartialEq)]
pub struct LibraryCohort {
  /// The speakers this cohort knows, as frozen at the moment of the freeze.
  speakers: Vec<SpeakerToken>,
  /// `diaric`'s own container, private, keyed by the token.
  entries: DiaricCohort<SpeakerToken, VoiceProfile>,
}

impl From<LibraryCohortBuilder> for LibraryCohort {
  /// Freeze an assembled cohort: the same speakers and the same entries, with
  /// every mutator left behind.
  fn from(builder: LibraryCohortBuilder) -> Self {
    Self {
      speakers: builder.speakers,
      entries: builder.entries,
    }
  }
}

impl LibraryCohort {
  /// Number of cohort members, before any exclusion.
  pub fn len(&self) -> usize {
    self.entries.len()
  }

  /// Whether the cohort holds no members. An empty one is not a refusal here;
  /// it becomes [`CalibrateError::ScoreNorm`] at
  /// [`Calibration::enrolled_side`], where `diaric`'s own floor lives.
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  /// Whether this cohort holds `token` as one of its speakers.
  ///
  /// A linear scan over the distinct speakers, which is nothing beside the one
  /// cosine per member the side that follows costs. It is what makes a token
  /// from some other cohort a refusal rather than an exclusion of nothing.
  fn holds(&self, token: SpeakerToken) -> bool {
    self.speakers.contains(&token)
  }
}

/// A cohort the caller asserts holds no material from any speaker that will be
/// scored against it.
///
/// The literature's own arrangement (Matějka et al. 2017 §2.1) and the only one
/// an unidentified probe has, because nothing about a probe can be excluded
/// from a cohort when the probe's identity is the thing being looked up. See
/// the module docs' "Two sides, and only one of them has an identity".
///
/// **It carries no speaker identities at all, and that is the design.** A
/// held-out cohort has nothing to exclude, so there is no token to pass wrongly
/// and no entrypoint that could take one: candidate-independence is structural
/// here rather than documented. It needs no builder for the same reason — there
/// is nothing to mint — so it is its own frozen form.
///
/// The disjointness itself cannot be checked — `coremlit` does not hold the
/// library — so [`assuming_disjoint`](HeldOutCohort::assuming_disjoint) is the
/// single, named place the caller states it, once, where they know the cohort's
/// provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutCohort {
  /// `diaric`'s own container with the key type erased and kept private, so
  /// the selection this runs is `diaric`'s and not a second copy of it, and a
  /// caller cannot reach `diaric`'s unbound statistic through it.
  entries: DiaricCohort<(), VoiceProfile>,
}

impl HeldOutCohort {
  /// Assert that these profiles are held out: none of them is material from a
  /// speaker that will be scored against this cohort.
  ///
  /// The one way to build a [`HeldOutCohort`], so the assertion cannot be
  /// skipped, and it is made once at assembly rather than at every scoring
  /// call. It is a claim about **provenance** — a public corpus of strangers,
  /// or a partition of the library reserved for this and never enrolled —
  /// which a caller can answer without knowing who any probe is.
  ///
  /// Getting it wrong is the self-contamination failure
  /// [`diaric::score_norm`] documents: a speaker's own material scores at the
  /// top of the cohort, top-N selection is guaranteed to keep it, and the
  /// resulting statistics look perfectly healthy while every score derived
  /// from them is wrong. Nothing can catch that here. What this shape does is
  /// make the claim explicit, singular and reviewable, rather than an argument
  /// re-passed at every call site.
  pub fn assuming_disjoint(profiles: Vec<VoiceProfile>) -> Self {
    Self {
      entries: DiaricCohort::from_entries(
        profiles
          .into_iter()
          .map(|profile| CohortEntry::new((), profile))
          .collect(),
      ),
    }
  }

  /// Number of cohort members.
  pub fn len(&self) -> usize {
    self.entries.len()
  }

  /// Whether the cohort holds no members. An empty one is not a refusal here;
  /// it becomes [`CalibrateError::ScoreNorm`] at [`Calibration::side`], where
  /// `diaric`'s own floor lives.
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }
}

/// The cohort shapes a [`Calibration`] can HOLD: frozen, with no mutator on
/// them and no key of the caller's in them.
///
/// **Sealed**: [`LibraryCohort`] and [`HeldOutCohort`] implement it, nothing
/// else can, and there is no method on it to call. It exists so
/// [`Calibration`]'s shared methods are offered for exactly the two containers
/// that can produce a side, rather than for any type at all.
pub trait CalibrationCohort: sealed::Sealed {}

impl CalibrationCohort for LibraryCohort {}

impl CalibrationCohort for HeldOutCohort {}

/// The cohort shapes [`Calibration::new`] can be handed, and the frozen cohort
/// each becomes.
///
/// **Sealed**, and the one place the freeze is expressed:
/// [`LibraryCohortBuilder`] becomes a [`LibraryCohort`] — the caller's assembly
/// step left behind, along with every mutator — and a [`HeldOutCohort`], which
/// mints nothing and has nothing to freeze, becomes itself.
pub trait CohortSource: sealed::Sealed + Sized {
  /// The frozen cohort a [`Calibration`] holds once it owns this one.
  type Cohort: CalibrationCohort + From<Self>;
}

impl CohortSource for LibraryCohortBuilder {
  type Cohort = LibraryCohort;
}

impl CohortSource for HeldOutCohort {
  type Cohort = Self;
}

mod sealed {
  /// Not nameable outside this module, so
  /// [`CohortSource`](super::CohortSource) and
  /// [`CalibrationCohort`](super::CalibrationCohort) cannot be implemented
  /// outside it either.
  pub trait Sealed {}

  impl Sealed for super::LibraryCohortBuilder {}

  impl Sealed for super::LibraryCohort {}

  impl Sealed for super::HeldOutCohort {}
}

/// The identity of one [`Calibration`].
///
/// Opaque and unforgeable: no constructor, no accessor to the number inside,
/// nothing a caller can write down. All it can do is compare equal to itself —
/// which is all [`Calibration::trial`] needs, because the property that has to
/// hold is *this calibration and no other*, and a calibration is one cohort and
/// one configuration.
///
/// Minted per calibration VALUE. A [`Clone`] of a calibration carries its
/// identity — same cohort, same options, so the same statistics — while every
/// [`Calibration::new`] mints a fresh one, including one over a cohort equal to
/// another's. That is the conservative direction: nothing in a sans-I/O crate
/// can tell that two separately assembled cohorts are one population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CalibrationId(u64);

impl CalibrationId {
  /// The next identity.
  ///
  /// `Relaxed` is the entire requirement. A `fetch_add` hands every caller a
  /// distinct previous value under any ordering, and this counter guards no
  /// other memory, so there is nothing for an `Acquire`/`Release` pair to
  /// order it against.
  ///
  /// The counter wraps after `2^64` calibrations, which is not a number a
  /// program reaches — one mint per nanosecond takes 584 years — and a wrap
  /// would make two calibrations compare equal, i.e. it would MISS a mismatch
  /// rather than invent one. That is why there is no panic path here: the only
  /// failure it could report belongs to a program that cannot exist, and a
  /// library that aborts is a worse answer than a check that has already done
  /// its work 2^64 times.
  fn mint() -> Self {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    Self(NEXT.fetch_add(1, Ordering::Relaxed))
  }
}

/// One cohort and one [`AsNormOptions`] — and the only thing on this page that
/// produces a side or a calibrated trial.
///
/// AS-Norm1 averages two z-scores, and they are commensurable only when four
/// things agree: the metric both numbers were computed in, the cohort both
/// sides selected their top-N from, the two endpoints the trial score is
/// between, and the configuration both sides were derived under. A
/// `Calibration` holds all four — so a caller assembles none of them, and there
/// is no loose triple of a score and two statistics left to mismatch. The
/// module docs' "One scoped operation, not three loose values" has the table
/// and the history.
///
/// It takes its cohort **by value and freezes it**: what it holds is a
/// [`LibraryCohort`] or a [`HeldOutCohort`], neither of which has a mutator on
/// it at all. A cohort that could still be pushed to would be a population that
/// changed under sides already taken, and making that a property of the TYPE
/// rather than of who holds which borrow is deliberate — a borrow is what a
/// [`Clone`] goes around.
///
/// Build one per `(cohort, options)` pair and keep it for as long as the sides
/// taken under it: a [`TrialSide`] is reusable across every trial of *this*
/// calibration and across no other (module docs' cost table).
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration<C> {
  /// The impostor population every side selects its top-N from.
  cohort: C,
  /// The selection and floor every side is derived under.
  options: AsNormOptions,
  /// This calibration's identity, minted at construction.
  id: CalibrationId,
}

impl<C: CalibrationCohort> Calibration<C> {
  /// Freeze a cohort and a configuration together, under a fresh
  /// [`CalibrationId`].
  ///
  /// It takes a [`LibraryCohortBuilder`] or a [`HeldOutCohort`] and holds what
  /// that freezes into — a [`LibraryCohort`], or the [`HeldOutCohort`] itself.
  /// Which one decides whether sides are taken with
  /// [`enrolled_side`](Self::enrolled_side) or with [`side`](Self::side), and
  /// that is the asymmetry between a named speaker and an unidentified probe
  /// that the module docs argue.
  ///
  /// **The freeze is about what the cohort loses, not about who may borrow
  /// it.** A frozen cohort holds token membership and token-keyed entries and
  /// nothing else, so a duplicate of it — or of this calibration — answers no
  /// question the original would have answered differently. That is
  /// deliberately not a claim about `&mut`: [`SpeakerToken`] carries the round
  /// where it was, and the [`Clone`] that walked around it.
  // `C: From<S>` is what `CohortSource::Cohort`'s own bound already says, and
  // it is restated because rustc does not carry an associated type's item
  // bounds across the `Cohort = C` equality. It constrains no caller: the only
  // two `S` that exist satisfy it by that same bound.
  pub fn new<S>(cohort: S, options: AsNormOptions) -> Self
  where
    S: CohortSource<Cohort = C>,
    C: From<S>,
  {
    Self {
      cohort: cohort.into(),
      options,
      id: CalibrationId::mint(),
    }
  }

  /// This calibration's identity — what every [`TrialSide`] it produces
  /// carries.
  ///
  /// [`trial`](Self::trial) compares it for the caller; it is exposed so a
  /// caller caching a side per speaker can tell a stale entry from a live one
  /// before they get as far as a refusal.
  pub const fn id(&self) -> CalibrationId {
    self.id
  }

  /// The configuration every side of this calibration is derived under.
  pub const fn options(&self) -> &AsNormOptions {
    &self.options
  }

  /// The cohort every side of this calibration selects its top-N from.
  pub const fn cohort(&self) -> &C {
    &self.cohort
  }

  /// AS-Norm1: the raw trial score between two sides, and the calibrated score
  /// a fixed threshold reads.
  ///
  /// The arithmetic is [`diaric::score_norm::as_norm`] called through — eq. (7)
  /// of Matějka et al. 2017, its `0.5`, its population standard deviation and
  /// its [`MAX_NORMALIZED_ERROR`] accuracy postcondition, none of them
  /// re-derived here.
  ///
  /// **It is handed no score.** It computes the raw one from the two sides'
  /// own profiles, which is what binds the trial to its endpoints: there is no
  /// argument that could carry a score between two other speakers, and none
  /// that could carry statistics taken over another cohort or under other
  /// options, because a side of this calibration has neither. The one thing
  /// left to check is that both sides really are this calibration's.
  ///
  /// `enrolled` and `probe` name the two sides of eq. (7) — `s(e,t)`'s
  /// enrolment and test terms. The order does not change the result (the
  /// formula is symmetric in the two z-scores) but it does change which side an
  /// error names, so it matches the trial being described.
  ///
  /// # Errors
  ///
  /// - [`CalibrateError::CalibrationMismatch`] if either side was taken under a
  ///   different calibration. The error names all three identities — this one
  ///   and both sides' — because which side is the stale one is the diagnosis.
  /// - [`CalibrateError::ScoringMismatch`] if the two sides' profiles were
  ///   prepared for different score sources. Not reachable through a single
  ///   calibration — a side exists only if its profile agreed with every cohort
  ///   entry it was scored against, so two sides of one cohort agree with each
  ///   other — and propagated rather than asserted away, on the same grounds as
  ///   the overflow check in this module's PLDA normalization: "cannot happen"
  ///   is a claim about today's code.
  /// - [`CalibrateError::ScoreNorm`] for `diaric`'s own refusals — a non-finite
  ///   trial score, or a z-score cancellation that leaves the result outside
  ///   the accuracy postcondition.
  pub fn trial(
    &self,
    enrolled: &TrialSide,
    probe: &TrialSide,
  ) -> Result<CalibratedTrial, CalibrateError> {
    if enrolled.calibration != self.id || probe.calibration != self.id {
      return Err(CalibrateError::CalibrationMismatch(
        CalibrationMismatch::new(self.id, enrolled.calibration, probe.calibration),
      ));
    }
    let raw = enrolled.profile.score(&probe.profile)?;
    // `as_norm` is `enrolled.stats.normalize(raw, &probe.stats)`, so the three
    // refusals it can make are about the trial and about the ENROLMENT side's
    // statistics; the count handed to the translation is that side's own. The
    // other four belong to the statistics constructor, which ran before either
    // side existed — mapped rather than asserted away, on this module's
    // standing discipline that "cannot happen" is a claim about today's code.
    let calibrated =
      diaric::score_norm::as_norm(raw, &enrolled.stats, &probe.stats).map_err(|e| {
        CalibrateError::ScoreNorm(ScoreNormRefusal::translate(e, enrolled.considered()))
      })?;
    Ok(CalibratedTrial {
      raw,
      calibrated,
      scoring: enrolled.profile.scoring(),
    })
  }
}

impl Calibration<LibraryCohort> {
  /// One side for an **enrolled** speaker, over a cohort that may contain that
  /// speaker's own entries.
  ///
  /// The entrypoint for a cohort sampled from the same library being scored —
  /// #123's own arrangement, and the one where a speaker's material is
  /// guaranteed to be selected into its own top-N. Exclusion is by identity:
  /// all of `enrolled`'s speaker's entries go, not merely the exact self-match,
  /// and only that speaker's — never the other side of the trial, which is what
  /// keeps a side reusable across every trial the speaker appears in.
  ///
  /// **An unidentified probe has no entrypoint here, deliberately.** It takes
  /// an [`Enrolled`], which binds an identity to a profile, and a probe has no
  /// identity to bind; passing the candidate's instead is the failure this
  /// shape removes. A probe goes to a `Calibration<HeldOutCohort>` and
  /// [`side`](Calibration::side). The module docs argue the asymmetry.
  ///
  /// **What this side can be averaged with.** Any other side of this same
  /// calibration, and nothing else. That makes it the door for a trial between
  /// two speakers the caller can both name (binding two enrolled recordings to
  /// one identity, say), each side dropping its own entries from the shared
  /// cohort, which is eq. (7)'s own arrangement. It is NOT a door onto
  /// "enrolment side from the library, probe side from somewhere else": that
  /// pairing averages two z-scores taken over two different impostor
  /// populations, and the module docs carry the case where doing so reverses
  /// which candidate ranks first.
  ///
  /// **Exclusion is by a [`SpeakerToken`], and there is no key anywhere to
  /// resolve one from.** A token is minted by
  /// [`LibraryCohortBuilder::speaker`], which takes no argument, so the
  /// exclusion set it names is fixed by what was filed under it and by nothing
  /// the caller can subsequently change or ask again — that type has the three
  /// roads a caller's `K` left open, and why sealing them one at a time did
  /// not close the class.
  ///
  /// # Errors
  ///
  /// - [`CalibrateError::ForeignSpeaker`] if this calibration's own cohort does
  ///   not hold that speaker. Excluding nothing instead would be the
  ///   self-contamination this door exists to prevent.
  /// - [`CalibrateError::ScoringMismatch`] if any scored cohort entry was
  ///   prepared for a different [`Scoring`] than the enrolled profile.
  /// - [`CalibrateError::ScoreNorm`] for the score-normalization refusals: an
  ///   empty selection (including a cohort that was entirely this speaker), too
  ///   few usable scores, or a selected set that does not spread.
  pub fn enrolled_side(&self, enrolled: Enrolled<'_>) -> Result<TrialSide, CalibrateError> {
    if !self.cohort.holds(enrolled.speaker) {
      return Err(CalibrateError::ForeignSpeaker);
    }
    let mut bridge = Bridge::default();
    let stats = self.cohort.entries.stats_excluding(
      &enrolled.speaker,
      enrolled.profile,
      scorer(&mut bridge),
      &self.options,
    );
    finish(bridge, stats, *enrolled.profile, self.id)
  }
}

impl Calibration<HeldOutCohort> {
  /// One side for any profile, over a [`HeldOutCohort`].
  ///
  /// Nothing is excluded, because a held-out cohort has nothing of this
  /// speaker's in it to exclude — the precondition is the cohort's, asserted
  /// once at [`HeldOutCohort::assuming_disjoint`], rather than a choice made
  /// again at every call.
  ///
  /// **This is the only door an unidentified probe has**, and it is therefore
  /// the door for the enrolled side of any trial a probe is in: AS-Norm
  /// averages two z-scores, they are commensurable only when both sides select
  /// their top-N from the same impostor population, and a calibration is what
  /// makes that one population rather than something to get right.
  ///
  /// # Errors
  ///
  /// As [`Calibration::enrolled_side`], minus the self-exclusion case.
  pub fn side(&self, profile: &VoiceProfile) -> Result<TrialSide, CalibrateError> {
    let mut bridge = Bridge::default();
    let stats =
      self
        .cohort
        .entries
        .stats_assuming_disjoint(profile, scorer(&mut bridge), &self.options);
    finish(bridge, stats, *profile, self.id)
  }
}

/// One side of a trial: a subject's profile and its cohort statistics, bound to
/// the [`Calibration`] that produced them.
///
/// Built only by [`Calibration::enrolled_side`] and [`Calibration::side`], and
/// not constructible any other way — every field is private and there is no
/// public constructor, which is what makes "this calibration produced it" a
/// fact rather than a convention.
///
/// It depends on its own profile and its own calibration and on nothing else,
/// so it is taken once per profile and reused across every trial that profile
/// appears in — the reuse guarantee in the module docs' cost table. Reused
/// across *calibrations* it is not, and the [`CalibrationId`] is what makes
/// that a refusal instead of a silently incommensurable average.
///
/// **It carries the profile, not only the statistics.** That is what binds a
/// trial to its endpoints: [`Calibration::trial`] computes the raw score from
/// the two sides' profiles rather than accepting one, so there is no argument
/// left that could carry a score between two other speakers.
///
/// **It publishes neither μ nor σ**, its [`Debug`] does not print them either,
/// and neither does the refusal that stands in its place when a side cannot be
/// derived — see the module docs' "What leaves this module" for what that
/// closes and for the residual it does not.
#[derive(Clone, Copy, PartialEq)]
pub struct TrialSide {
  /// Whose side this is — the subject the statistics were taken for, and one
  /// endpoint of every trial this side appears in.
  profile: VoiceProfile,
  /// `diaric`'s own statistic, untouched.
  stats: CohortStats,
  /// The calibration that produced it: one cohort, one configuration.
  calibration: CalibrationId,
}

// Hand-written rather than derived, and the difference is the whole point of
// the type: `#[derive(Debug)]` prints every field, and `CohortStats`'s own
// derived `Debug` prints `mean` and `deviation`. That is every operand of
// eq. (7) on a `{:?}` of a value this crate handed out — the same disclosure
// the removed accessors made, through a door a reviewer would not think to
// look at. The profile is left out for a duller reason: it is a kilobyte of
// prepared vector, and printing it helps nobody.
impl core::fmt::Debug for TrialSide {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("TrialSide")
      .field("scoring", &self.scoring())
      .field("selected", &self.selected())
      .field("considered", &self.considered())
      .field("calibration", &self.calibration)
      .finish_non_exhaustive()
  }
}

impl TrialSide {
  /// The score source this side's profile was prepared for, and therefore the
  /// one every cohort score behind its statistics was computed in.
  pub const fn scoring(&self) -> Scoring {
    self.profile.scoring()
  }

  /// The calibration that produced this side.
  ///
  /// [`Calibration::trial`] compares it for the caller; it is exposed so a
  /// caller caching a side per speaker can tell a stale entry from a live one
  /// before they get as far as a refusal.
  pub const fn calibration(&self) -> CalibrationId {
    self.calibration
  }

  /// How many cohort scores the top-N selection kept.
  pub const fn selected(&self) -> usize {
    self.stats.selected()
  }

  /// How many cohort members were scored at all — every member, less whatever
  /// [`Calibration::enrolled_side`] dropped by identity.
  pub const fn considered(&self) -> usize {
    self.stats.considered()
  }
}

/// One trial, calibrated: the raw similarity and the AS-Norm1 score a fixed
/// threshold reads.
///
/// Built only by [`Calibration::trial`], so the two numbers are one trial's —
/// computed together, from two sides of one calibration, between the endpoints
/// those sides are. There is no constructor that takes them apart.
///
/// Both are published because #123's own comparison needs both: the claim being
/// tested is that no fixed threshold over the raw scores separates a library
/// and that one over the calibrated scores does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibratedTrial {
  /// The raw, uncalibrated similarity between the two sides' profiles.
  raw: f64,
  /// eq. (7) over that similarity and the two sides' statistics.
  calibrated: f64,
  /// The source both profiles were prepared for, and so the space both numbers
  /// live in.
  scoring: Scoring,
}

impl CalibratedTrial {
  /// The raw, uncalibrated similarity.
  pub const fn raw(&self) -> f64 {
    self.raw
  }

  /// The AS-Norm1 score — what a fixed threshold is read against.
  pub const fn calibrated(&self) -> f64 {
    self.calibrated
  }

  /// The score source both numbers were computed in.
  pub const fn scoring(&self) -> Scoring {
    self.scoring
  }
}

/// What one pass over a cohort produces besides the scores themselves.
///
/// `diaric`'s cohort statistics take an INFALLIBLE `FnMut(&S, &T) -> f64`, so
/// a refusal has to be carried out of the closure by hand — and so does the
/// one count its refusals do not report.
#[derive(Debug, Default)]
struct Bridge {
  /// The FIRST scoring refusal, if any.
  carried: Option<CalibrateError>,
  /// How many cohort entries the side actually reached: every member, less
  /// whatever the exclusion dropped. `diaric` publishes this on a SUCCESSFUL
  /// [`CohortStats`] and on none of its refusals, and it is the count that
  /// says whether an exclusion ate the cohort — so a refusal that carries no
  /// arithmetic can still carry it.
  considered: usize,
}

/// The fallible-scorer bridge.
///
/// The poison value is `NaN`, not a plausible score: `CohortStats::from_scores`
/// rejects a non-finite score outright, so even if the carried error were
/// dropped the statistics would refuse rather than quietly absorb a fabricated
/// number. Only the FIRST error is kept — the rest are one defect repeated, and
/// keeping the last would report a mixed cohort's final entry instead of the
/// one that broke it.
///
/// The count is incremented once per entry reached, which is `considered`
/// exactly — `from_scores` pulls the whole iterator unless a non-finite score
/// stops it, and the refusal that produces carries no count.
fn scorer(bridge: &mut Bridge) -> impl FnMut(&VoiceProfile, &VoiceProfile) -> f64 + '_ {
  move |side, entry| {
    bridge.considered += 1;
    match side.score(entry) {
      Ok(v) => v,
      Err(e) => {
        bridge.carried.get_or_insert(e);
        f64::NAN
      }
    }
  }
}

/// Report a carried scoring refusal ahead of `diaric`'s own, so a mixed cohort
/// is named as a mismatch rather than as the `NonFiniteScore` the poison value
/// produces downstream, and bind the surviving statistic to the profile it was
/// taken for and the calibration it was taken under.
///
/// Reporting the mismatch here is what makes the side's own source sound: a
/// [`TrialSide`] whose profile is [`Scoring::Cosine`] can only have been scored
/// against entries that were all `Cosine` too, since one foreign entry refuses
/// the whole call rather than contributing a number.
///
/// The other two bindings are sound for a simpler reason: both callers pass
/// their own calibration's identity and the profile they were handed, and
/// neither is something a caller can supply.
fn finish(
  bridge: Bridge,
  stats: Result<CohortStats, diaric::score_norm::Error>,
  profile: VoiceProfile,
  calibration: CalibrationId,
) -> Result<TrialSide, CalibrateError> {
  match bridge.carried {
    Some(e) => Err(e),
    None => stats
      .map(|stats| TrialSide {
        profile,
        stats,
        calibration,
      })
      // `diaric`'s refusals carry the deviation, the z-scores and the value
      // they cancelled to; this crate's do not. See [`ScoreNormRefusal`].
      .map_err(|e| CalibrateError::ScoreNorm(ScoreNormRefusal::translate(e, bridge.considered))),
  }
}

#[cfg(test)]
mod tests;
