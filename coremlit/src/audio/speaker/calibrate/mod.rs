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
//! [`diaric::score_norm`] called through — [`CohortStats`],
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
//!   closure. [`enrolled_stats`] and [`held_out_stats`] carry it out by hand
//!   rather than letting a caller decide what number a failed score should be
//!   — an `unwrap_or(0.0)` inside that closure poisons a mean silently, which
//!   is the one failure AS-Norm exists to prevent;
//! - the shape that says which cohort each side of a trial may use, and the
//!   [`Scoring`] tag carried through every value so one metric cannot
//!   calibrate another. Those two are the rest of this page, and both are
//!   corrections to what this module said when it was first written.
//!
//! # Two sides, and only one of them has an identity
//!
//! AS-Norm needs a cohort statistic for *both* speakers in a trial, and the
//! two sides are not symmetric.
//!
//! The **enrolled** side is a speaker the caller's library already names. Its
//! identity is known, so a cohort drawn from that same library can have
//! exactly that speaker's entries removed — [`enrolled_stats`], which is
//! [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding)
//! with the key and the profile handed over as one [`Enrolled`] value.
//!
//! The **probe** side is a recording whose speaker is not yet known. That
//! identity is what identification is *trying to discover*, so the caller has
//! no key of the probe's to exclude — and neither of the two things they could
//! reach for instead is answerable:
//!
//! - excluding nothing from a library-sampled cohort is self-contamination
//!   whenever the probe's speaker is enrolled, which is precisely the case a
//!   positive identification is. A self-match is the largest score obtainable,
//!   so top-N selection is *guaranteed* to pick it up;
//! - excluding the **candidate's** key is worse, and it is what this module's
//!   first version told a caller to do. It drops a valid impostor, and it
//!   makes the probe's statistics depend on which candidate the probe is being
//!   scored against. `diaric` names that failure at the entrypoint itself:
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
//! One cohort should serve both sides of a trial. AS-Norm averages two
//! z-scores, and they are only commensurable when they measure the same trial
//! score against the same impostor population — [`diaric::score_norm`]'s own
//! statement of eq. (7) has each side selecting its top-N of *the shared
//! cohort*. So the recommended arrangement is a single [`HeldOutCohort`] for
//! both sides, which is what the example below does. [`enrolled_stats`] exists
//! for the library-sampled cohort #123 describes, where the enrolled side
//! still has a correct answer and the probe side does not.
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
//! guidance had just made. Binding the key to the profile turns it into a type
//! property with a `compile_fail` proof, and a keyless [`HeldOutCohort`] means
//! the probe's road has no key to pass at all. What is left of the hazard is
//! stated below rather than claimed away.
//!
//! ## The key travels with the profile
//!
//! [`enrolled_stats`] takes one [`Enrolled`] rather than a key *and* a side,
//! so there is no second argument left that could name a different speaker.
//!
//! What remains is not a mis-passed argument but a false statement:
//! `Enrolled::new(&candidate_key, &probe)` *claims* the probe belongs to that
//! speaker. No type in a sans-I/O crate can refute that — `coremlit` does not
//! hold the library — but it is a claim spelled out at the point a caller
//! reads a library record, not an argument slot two positions along from the
//! one that matters. A probe has no such value to build, and that is the
//! point: a bare [`VoiceProfile`] reaches only [`held_out_stats`].
//!
//! ```compile_fail,E0308
//! use coremlit::audio::speaker::calibrate::{AsNormOptions, Cohort, Scoring, enrolled_stats};
//! # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
//! let mut cohort: Cohort<u32, _> = Cohort::new();
//! cohort.push(1, Scoring::Cosine.prepare(&raw).unwrap());
//! // An unidentified probe: a prepared vector and no key at all. There is no
//! // identity to exclude, so the excluding door must not accept it.
//! let probe = Scoring::Cosine.prepare(&raw).unwrap();
//! let _ = enrolled_stats(&cohort, &probe, &AsNormOptions::new());
//! ```
//!
//! # Every value carries its score source
//!
//! [`Scoring`] is checked at every step rather than once.
//! [`VoiceProfile::score`] returns a [`TrialScore`], both statistics doors
//! return a [`SideStats`], each carrying the source it was computed in, and
//! [`as_norm`] refuses unless all three agree
//! ([`CalibrateError::NormalizationMismatch`]).
//!
//! That is the failure a single check at `score()` did not cover, and it is
//! silent by construction: [`Scoring::Cosine`] cohort scores of `[-1, 1]` have
//! mean `0` and deviation `1`, so *any* finite [`Scoring::PldaCosine`] trial
//! score normalized against them comes back finite and plausible — one metric
//! calibrated by another, with no value out of range to notice. It is most
//! reachable exactly where this design pushes a caller: caching a
//! [`SideStats`] per speaker while iterating over both score sources, which is
//! what #123's ask 3 is.
//!
//! The tag is trustworthy because there is no way to write one down. A
//! [`TrialScore`] exists only as the output of [`VoiceProfile::score`], which
//! produces the number and the source together; a [`SideStats`] exists only as
//! the output of the two statistics doors, which take theirs from the side
//! profile — and every cohort entry that reached those statistics had to match
//! that profile or the whole call is [`CalibrateError::ScoringMismatch`].
//!
//! **What it does not promise.** The tag says which *metric* a number was
//! computed in. It says nothing about which *cohort*: two [`SideStats`] both
//! tagged [`Scoring::Cosine`] but taken over different impostor populations
//! normalize without complaint, and the z-scores they average are
//! commensurable only to the extent the two populations are. That is the
//! caller's to get right, and it is why the recommended arrangement above is
//! one cohort for both sides.
//!
//! `diaric`'s own [`as_norm`](diaric::score_norm::as_norm) is arithmetic over
//! two untagged [`CohortStats`], and **this module does not re-export it**.
//! Nothing here hands out the [`CohortStats`] inside a [`SideStats`] either,
//! so a `coremlit` statistic has no route into the untagged door. A caller
//! scoring in their own space still has one, and should: they build their own
//! [`CohortStats`] through
//! [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding) and
//! normalize with
//! [`CohortStats::normalize`](diaric::score_norm::CohortStats::normalize),
//! `diaric`'s own inherent method on `diaric`'s own type. That road is
//! described under "Not implemented here" and it is not closed; what is closed
//! is a `coremlit`-tagged value entering it without its tag.
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
//! A [`SideStats`] depends only on its own profile and its own cohort, never
//! on the other side of the trial, so it is computed once and reused across
//! every trial that profile appears in. **That holds for both sides here**,
//! and it is what the shape above buys:
//!
//! | side | cohort | the statistic depends on | reused across |
//! |---|---|---|---|
//! | enrolled | held-out, or library-sampled with its own key excluded | that speaker and the cohort | every trial the speaker appears in |
//! | probe | held-out only | that probe and the cohort | every candidate the probe is scored against |
//!
//! `N` enrolled speakers against `M` probes over a cohort of `C` therefore
//! costs `(N + M)·C` cohort scores and `N·M` trial scores, not `N·M·C` — for
//! 1 000 library profiles, 100 probes and a 300-member cohort, 330 000 cohort
//! scores rather than 30 million. [`diaric::score_norm`]'s cost table has the
//! derivation.
//!
//! **This is a correction, not a restatement.** The version of this page that
//! shipped claimed the same `N·C` reuse while telling a caller to exclude the
//! candidate's key from the probe's side — which makes the probe's statistic
//! trial-dependent, so it has to be recomputed per candidate and the claim was
//! false by `M` for exactly the road the page recommended. The claim is true
//! again because the probe's side no longer has a candidate in it.
//!
//! **Where it does not hold**: a probe against a library-sampled cohort. There
//! is no entrypoint for that pairing, because there is no correct statistic to
//! return — see above. A caller who has no held-out cohort has to assemble one
//! before an unidentified probe can be normalized at all. That is a real
//! restriction on what this door can do, and it is stated here rather than
//! papered over with a default that would be wrong in a way nothing could
//! detect.
//!
//! # End to end
//!
//! ```
//! use coremlit::audio::speaker::{
//!   calibrate::{
//!     AsNormOptions, Cohort, Enrolled, HeldOutCohort, Scoring, as_norm, enrolled_stats,
//!     held_out_stats,
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
//! // Identities are the caller's. Anything `Eq` names a speaker.
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
//! // 2. The trial: a library profile the caller can name, against a probe
//! //    from a recording that has just been diarized — whose speaker is the
//! //    thing being looked up, not something the caller can hand over.
//! let alice: PersonId = 7;
//! let stored = scoring.prepare(&stored_centroid(alice as usize))?;
//! let probe = scoring.prepare(&probe_of(alice as usize))?;
//!
//! // 3. One side per profile, both over the same held-out cohort, both
//! //    reusable. Neither depends on the other side of the trial.
//! let stored_side = held_out_stats(&held_out, &stored, &options)?;
//! let probe_side = held_out_stats(&held_out, &probe, &options)?;
//! assert_eq!(probe_side.considered(), 64); // nothing to exclude
//!
//! // 4. The raw trial score, then the calibrated one a fixed threshold reads.
//! //    All three values carry `Cosine`, so this cannot be a PLDA score
//! //    calibrated by cosine statistics.
//! let trial = stored.score(&probe)?;
//! let normalized = as_norm(trial, &stored_side, &probe_side)?;
//! assert!(normalized.is_finite());
//!
//! // The other cohort shape: sampled from the library itself, so it CONTAINS
//! // Alice. Only an enrolled side may use it, and the key travels with the
//! // profile, so no other speaker's entries can be dropped by mistake.
//! let mut library: Cohort<PersonId, _> = Cohort::new();
//! for id in 0..64u32 {
//!   library.push(id, scoring.prepare(&stored_centroid(id as usize))?);
//! }
//! let alice_side = enrolled_stats(&library, Enrolled::new(&alice, &stored), &options)?;
//! assert_eq!(alice_side.considered(), 63); // 64 members, less Alice's own
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
//! - **A third score source.** [`enrolled_stats`] and [`held_out_stats`] take
//!   a [`VoiceProfile`], so they cover the sources [`Scoring`] names and no
//!   others — they exist to carry a `coremlit` refusal out of an infallible
//!   closure, which a caller's own scoring function does not need. Genericity
//!   is not lost by that: [`Cohort`] is `diaric`'s own type, generic over its
//!   item type and re-exported whole, so a caller scoring in some other space
//!   fills a `Cohort<K, T>` with THEIR item type, calls
//!   [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding)
//!   directly with their own `FnMut(&S, &T) -> f64`, and normalizes with
//!   [`CohortStats::normalize`](diaric::score_norm::CohortStats::normalize).
//!   Nothing here has to be widened for that to work — and [`VoiceProfile`] is
//!   not in that road's way, because it is not on it.

use diaric::{
  embed::Embedding,
  plda::{PLDA_DIMENSION, RawEmbedding},
};

use crate::audio::speaker::{
  embed::EMBEDDING_DIM,
  error::{CalibrateError, NormalizationMismatch, ProfileLength, ScoringMismatch},
  extract::shared_plda_transform,
};

/// `diaric`'s AS-Norm vocabulary, re-exported so a caller of this door does not
/// need a direct `diaric` dependency to name what it takes and returns.
///
/// These are `diaric`'s own types, unchanged: [`Cohort`] is the container the
/// caller fills for [`enrolled_stats`], [`AsNormOptions`] tunes the per-side
/// statistics, and [`CohortStats`] is the untagged statistic a caller scoring
/// in their own space computes for themselves. A `coremlit`-side mirror of
/// [`AsNormOptions`] (the way
/// [`OnlineOptions`](crate::audio::speaker::OnlineOptions) mirrors `diaric`'s
/// online configuration) would be two constants and two builders with nowhere
/// to drift to but out of step, so there is none.
///
/// **`diaric::score_norm::as_norm` is deliberately NOT among these.** This
/// module's own [`as_norm`] replaces it, because the free function takes two
/// untagged [`CohortStats`] and cannot tell one score source from another —
/// see "Every value carries its score source".
///
/// One consequence, stated because it is not visible from the signatures:
/// [`AsNormOptions`]'s `serde` impls are `diaric`'s own and sit behind
/// `diaric`'s `serde` feature, which this crate does NOT enable — `coremlit`'s
/// `serde` feature covers `coremlit`'s types, [`Scoring`] included. A caller
/// serializing an AS-Norm configuration enables `serde` on their own `diaric`
/// dependency; feature unification then applies it here.
pub use diaric::score_norm::{
  AsNormOptions, Cohort, CohortEntry, CohortStats, DEFAULT_MIN_DEVIATION, DEFAULT_TOP_N,
  MAX_NORMALIZED_ERROR, MIN_COHORT_SCORES,
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
/// be scored.
///
/// Build one with [`Scoring::prepare`]. A profile is a *prepared vector*, not a
/// stored one: it holds no identity, no provenance and no recording. The
/// caller's library holds those, and binds one back to a profile with
/// [`Enrolled`] when — and only when — the speaker is known.
///
/// A bare `VoiceProfile` is therefore exactly what an unidentified probe is,
/// and it reaches only [`held_out_stats`].
///
/// # Why the score source is a tag and not a type parameter
///
/// Mixing sources is refused at runtime ([`CalibrateError::ScoringMismatch`],
/// [`CalibrateError::NormalizationMismatch`]) rather than made unrepresentable
/// by a `VoiceProfile<S>`. Two reasons, the second deciding:
///
/// - the cohort type is `diaric`'s [`Cohort<K, T>`], so a type parameter here
///   would monomorphize the caller's whole cohort on a choice they naturally
///   make at run time — which score source to run *this* comparison in;
/// - #123's follow-up is a confusion experiment *between* score sources, whose
///   natural shape is a loop over `[Scoring::Cosine, Scoring::PldaCosine]`. A
///   type parameter turns that loop into duplicated generic code.
///
/// What the tag buys in exchange is that a cross-source result is a typed
/// refusal rather than a finite number, and that is a property of the whole
/// chain rather than of this one call — the tag travels into [`TrialScore`]
/// and [`SideStats`] and is re-checked by [`as_norm`]. An earlier version of
/// this paragraph claimed the guarantee on the ground that "every path that
/// reads two profiles compares the tags first", which was true and not
/// sufficient: the last step reads no profiles at all. What the tag still does
/// **not** say is which cohort a statistic came from — module docs, "What it
/// does not promise".
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

  /// The raw trial score between this profile and `other` — the number
  /// [`as_norm`] normalizes, tagged with the source it was computed in.
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
  pub fn score(&self, other: &Self) -> Result<TrialScore, CalibrateError> {
    match (&self.0, &other.0) {
      // Calls `diaric`'s cosine rather than repeating its loop, so this score
      // is bit-identical to the one the online clusterer matches on.
      (Prepared::Cosine(a), Prepared::Cosine(b)) => Ok(TrialScore {
        raw: f64::from(diaric::embed::cosine_similarity(a, b)),
        scoring: Scoring::Cosine,
      }),
      (Prepared::PldaCosine(a), Prepared::PldaCosine(b)) => Ok(TrialScore {
        raw: a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        scoring: Scoring::PldaCosine,
      }),
      _ => Err(CalibrateError::ScoringMismatch(ScoringMismatch::new(
        self.scoring(),
        other.scoring(),
      ))),
    }
  }
}

/// A raw trial score, carrying the [`Scoring`] it was computed in.
///
/// Built only by [`VoiceProfile::score`], so the tag cannot be a claim: the
/// number and the source it came from are produced together and there is no
/// constructor that takes them apart. [`as_norm`] re-checks the tag against
/// both sides' statistics — the module docs' "Every value carries its score
/// source" says what a bare `f64` here would let through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrialScore {
  /// The similarity itself.
  raw: f64,
  /// The score source both profiles were prepared for.
  scoring: Scoring,
}

impl TrialScore {
  /// The raw, uncalibrated similarity.
  ///
  /// #123's own comparison needs it: the claim being tested is that no fixed
  /// threshold over these separates a library, and that one over [`as_norm`]'s
  /// output does.
  pub const fn raw(&self) -> f64 {
    self.raw
  }

  /// The score source this trial was computed in.
  pub const fn scoring(&self) -> Scoring {
    self.scoring
  }
}

/// One side of a trial: a cohort statistic, carrying the [`Scoring`] it was
/// computed in.
///
/// Built only by [`enrolled_stats`] and [`held_out_stats`]. It depends on its
/// own profile and its own cohort and on nothing else, so it is computed once
/// per profile and reused across every trial that profile appears in — the
/// reuse guarantee in the module docs' cost table.
///
/// The [`CohortStats`] inside is deliberately not handed back. Doing so would
/// re-open the untagged [`as_norm`](diaric::score_norm::as_norm) this module
/// refuses to re-export; the four numbers a caller actually reads are
/// forwarded instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SideStats {
  /// `diaric`'s own statistic, untouched.
  stats: CohortStats,
  /// The score source the side profile — and therefore every cohort score
  /// behind these numbers — was prepared for.
  scoring: Scoring,
}

impl SideStats {
  /// The score source these statistics were computed in.
  pub const fn scoring(&self) -> Scoring {
    self.scoring
  }

  /// Mean of the selected cohort scores.
  pub const fn mean(&self) -> f64 {
    self.stats.mean()
  }

  /// Population standard deviation of the selected cohort scores.
  pub const fn deviation(&self) -> f64 {
    self.stats.deviation()
  }

  /// How many cohort scores the top-N selection kept.
  pub const fn selected(&self) -> usize {
    self.stats.selected()
  }

  /// How many cohort members were scored at all — every member, less whatever
  /// [`enrolled_stats`] dropped by identity.
  pub const fn considered(&self) -> usize {
    self.stats.considered()
  }
}

/// A [`VoiceProfile`] bound to the identity it belongs to.
///
/// The one value [`enrolled_stats`] takes in place of a key *and* a side, so
/// no argument is left that could name a different speaker than the profile
/// belongs to. The module docs' "The key travels with the profile" says what
/// that removes and what it cannot.
///
/// A borrowing view rather than an owning pair: a caller's library already
/// holds the key and the profile, `K` may be a `String`, and a
/// [`VoiceProfile`] is a kilobyte of prepared vector. Binding them should cost
/// two pointers.
#[derive(Debug)]
pub struct Enrolled<'a, K> {
  /// Whose profile this is.
  speaker: &'a K,
  /// The prepared vector itself.
  profile: &'a VoiceProfile,
}

// Hand-written rather than derived: `#[derive(Clone, Copy)]` would demand
// `K: Clone` / `K: Copy`, which a pair of shared references does not need.
impl<K> Clone for Enrolled<'_, K> {
  fn clone(&self) -> Self {
    *self
  }
}

impl<K> Copy for Enrolled<'_, K> {}

impl<'a, K> Enrolled<'a, K> {
  /// Bind a profile to the speaker it belongs to.
  ///
  /// This is an assertion about identity, and the only one in this module: it
  /// says `profile` is material from `speaker`, which is what makes dropping
  /// `speaker`'s cohort entries the right thing to do. It is answerable for a
  /// library record and it is *not* answerable for a probe — so a probe never
  /// gets one.
  pub const fn new(speaker: &'a K, profile: &'a VoiceProfile) -> Self {
    Self { speaker, profile }
  }

  /// Whose profile this is.
  pub const fn speaker(&self) -> &K {
    self.speaker
  }

  /// The prepared vector itself — how an enrolled speaker reaches
  /// [`held_out_stats`], [`VoiceProfile::score`], or a cohort.
  pub const fn profile(&self) -> &VoiceProfile {
    self.profile
  }
}

/// A cohort the caller asserts holds no material from any speaker that will be
/// scored against it.
///
/// The literature's own arrangement (Matějka et al. 2017 §2.1) and the only
/// one an unidentified probe has, because nothing about a probe can be
/// excluded from a cohort when the probe's identity is the thing being looked
/// up. See the module docs' "Two sides, and only one of them has an identity".
///
/// **It carries no speaker keys, and that is the design.** A held-out cohort
/// has nothing to exclude, so there is no key to pass wrongly and no
/// entrypoint that could take one: candidate-independence is structural here
/// rather than documented.
///
/// The disjointness itself cannot be checked — `coremlit` does not hold the
/// library — so [`assuming_disjoint`](HeldOutCohort::assuming_disjoint) is the
/// single, named place the caller states it, once, where they know the
/// cohort's provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutCohort {
  /// `diaric`'s own container with the key type erased, so the selection this
  /// runs is `diaric`'s and not a second copy of it.
  entries: Cohort<(), VoiceProfile>,
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
      entries: Cohort::from_entries(
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
  /// it becomes [`CalibrateError::ScoreNorm`] at [`held_out_stats`], where
  /// `diaric`'s own floor lives.
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }
}

/// Cohort statistics for an **enrolled** speaker, over a cohort that may
/// contain that speaker's own entries.
///
/// The entrypoint for a cohort sampled from the same library being scored —
/// #123's own arrangement, and the one where a speaker's material is guaranteed
/// to be selected into its own top-N. Exclusion is by identity: all of
/// `enrolled`'s speaker's entries go, not merely the exact self-match, and only
/// that speaker's — never the other side of the trial, which is what keeps a
/// side reusable across every trial the speaker appears in.
///
/// **An unidentified probe has no entrypoint here, deliberately.** It takes an
/// [`Enrolled`], which binds a key to a profile, and a probe has no key to
/// bind; passing the candidate's key instead is the failure this shape
/// removes. A probe goes to [`held_out_stats`]. The module docs argue the
/// asymmetry.
///
/// [`Eq`] rather than [`PartialEq`] on `K` because the filter is
/// `entry.speaker != *speaker`, and that is a correct exclusion only if a key
/// equals itself; `f64::NAN` is the standard counterexample, and it would keep
/// a self-entry that scores `1.0`. See
/// [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding) for
/// the full argument. The bound is pinned at the call site, not merely
/// inherited:
///
/// ```compile_fail,E0277
/// use coremlit::audio::speaker::calibrate::{
///   AsNormOptions, Cohort, Enrolled, Scoring, enrolled_stats,
/// };
/// # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
/// // `f64` is `PartialEq` but not `Eq`, so it cannot name a speaker — and a
/// // `NaN` key would not match its own entry, keeping the self-match in.
/// let mut cohort: Cohort<f64, _> = Cohort::new();
/// cohort.push(f64::NAN, Scoring::Cosine.prepare(&raw).unwrap());
/// let side = Scoring::Cosine.prepare(&raw).unwrap();
/// let _ = enrolled_stats(&cohort, Enrolled::new(&f64::NAN, &side), &AsNormOptions::new());
/// ```
///
/// # Errors
///
/// - [`CalibrateError::ScoringMismatch`] if any scored cohort entry was
///   prepared for a different [`Scoring`] than the enrolled profile.
/// - [`CalibrateError::ScoreNorm`] for `diaric`'s own refusals: an empty
///   selection (including a cohort that was entirely this speaker), too few
///   usable scores, or a selected set that does not spread.
pub fn enrolled_stats<K: Eq>(
  cohort: &Cohort<K, VoiceProfile>,
  enrolled: Enrolled<'_, K>,
  options: &AsNormOptions,
) -> Result<SideStats, CalibrateError> {
  let mut carried = None;
  let stats = cohort.stats_excluding(
    enrolled.speaker,
    enrolled.profile,
    scorer(&mut carried),
    options,
  );
  finish(carried, stats, enrolled.profile.scoring())
}

/// Cohort statistics for any profile, over a [`HeldOutCohort`].
///
/// Nothing is excluded, because a held-out cohort has nothing of this
/// speaker's in it to exclude — the precondition is the cohort's, asserted
/// once at [`HeldOutCohort::assuming_disjoint`], rather than a choice made
/// again at every call.
///
/// **This is the only door an unidentified probe has**, and it is the
/// recommended one for the enrolled side too: AS-Norm averages two z-scores,
/// and they are commensurable when both sides select their top-N from the same
/// impostor population.
///
/// # Errors
///
/// As [`enrolled_stats`], minus the self-exclusion case.
pub fn held_out_stats(
  cohort: &HeldOutCohort,
  side: &VoiceProfile,
  options: &AsNormOptions,
) -> Result<SideStats, CalibrateError> {
  let mut carried = None;
  let stats = cohort
    .entries
    .stats_assuming_disjoint(side, scorer(&mut carried), options);
  finish(carried, stats, side.scoring())
}

/// AS-Norm1: the calibrated score a fixed threshold reads, from a trial score
/// and the two sides' cohort statistics.
///
/// The arithmetic is [`diaric::score_norm::as_norm`] called through — eq. (7)
/// of Matějka et al. 2017, its `0.5`, its population standard deviation and
/// its [`MAX_NORMALIZED_ERROR`] accuracy postcondition, none of them re-derived
/// here. What this wrapper adds is the check `diaric` cannot make: that all
/// three values were computed in the same [`Scoring`].
///
/// `diaric`'s free function takes two untagged [`CohortStats`] and an `f64`, so
/// [`Scoring::Cosine`] statistics with mean `0` and deviation `1` will happily
/// calibrate a [`Scoring::PldaCosine`] trial score and return a finite,
/// plausible number. That is why this module re-exports the arithmetic's
/// vocabulary but not the function.
///
/// `enrolled` and `probe` name the two sides of eq. (7) — `s(e,t)`'s
/// enrolment and test terms. The order does not change the result (the
/// formula is symmetric in the two z-scores) but it does change which side an
/// error names, so it matches the trial being described.
///
/// # Errors
///
/// - [`CalibrateError::NormalizationMismatch`] if the trial score and the two
///   sides were not all computed in one score source. Reported before the
///   arithmetic, and naming all three, because which of them is the odd one
///   out is the whole diagnosis.
/// - [`CalibrateError::ScoreNorm`] for `diaric`'s own refusals — a non-finite
///   trial score, or a z-score cancellation that leaves the result outside the
///   accuracy postcondition.
pub fn as_norm(
  trial: TrialScore,
  enrolled: &SideStats,
  probe: &SideStats,
) -> Result<f64, CalibrateError> {
  if trial.scoring != enrolled.scoring || trial.scoring != probe.scoring {
    return Err(CalibrateError::NormalizationMismatch(
      NormalizationMismatch::new(trial.scoring, enrolled.scoring, probe.scoring),
    ));
  }
  diaric::score_norm::as_norm(trial.raw, &enrolled.stats, &probe.stats)
    .map_err(CalibrateError::ScoreNorm)
}

/// The fallible-scorer bridge. `diaric`'s cohort statistics take an INFALLIBLE
/// `FnMut(&S, &T) -> f64`, so a refusal has to be carried out of the closure by
/// hand.
///
/// The poison value is `NaN`, not a plausible score: `CohortStats::from_scores`
/// rejects a non-finite score outright, so even if the carried error were
/// dropped the statistics would refuse rather than quietly absorb a fabricated
/// number. Only the FIRST error is kept — the rest are one defect repeated, and
/// keeping the last would report a mixed cohort's final entry instead of the
/// one that broke it.
fn scorer(
  carried: &mut Option<CalibrateError>,
) -> impl FnMut(&VoiceProfile, &VoiceProfile) -> f64 + '_ {
  move |side, entry| match side.score(entry) {
    Ok(v) => v.raw,
    Err(e) => {
      carried.get_or_insert(e);
      f64::NAN
    }
  }
}

/// Report a carried scoring refusal ahead of `diaric`'s own, so a mixed cohort
/// is named as a mismatch rather than as the `NonFiniteScore` the poison value
/// produces downstream, and tag the surviving statistic with the source its
/// side was prepared for.
///
/// That tag is sound precisely because the mismatch is reported here: a
/// [`SideStats`] labelled [`Scoring::Cosine`] can only have come from a
/// `Cosine` side scored against entries that were all `Cosine` too, since one
/// foreign entry refuses the whole call rather than contributing a number.
fn finish(
  carried: Option<CalibrateError>,
  stats: Result<CohortStats, diaric::score_norm::Error>,
  scoring: Scoring,
) -> Result<SideStats, CalibrateError> {
  match carried {
    Some(e) => Err(e),
    None => stats
      .map(|stats| SideStats { stats, scoring })
      .map_err(CalibrateError::ScoreNorm),
  }
}

#[cfg(test)]
mod tests;
