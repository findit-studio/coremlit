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
//! What this module takes is a [`Cohort`] the caller assembled and the
//! candidates the caller chose; what it returns is a number. There is no
//! enrolment, no persistence, and no cohort-selection heuristic here, because
//! none of those can be answered without the library this crate cannot see.
//!
//! # The arithmetic is `diaric`'s, not a second copy
//!
//! Every statistic and every normalization on this page is
//! [`diaric::score_norm`] called through — [`CohortStats`], [`as_norm`], the
//! [`AsNormOptions`] defaults, [`MAX_NORMALIZED_ERROR`]'s accuracy
//! postcondition. That module is where the cancellation analysis, the
//! subnormal handling and the two-tier error predicate live, and a second
//! implementation of AS-Norm1 would be a second set of those bugs. What this
//! module adds is everything between "a raw WeSpeaker row" and "an `f64` a
//! cohort statistic can be taken over":
//!
//! - the [`Scoring`] a profile is prepared for, and the preparation itself;
//! - a **fallible** boundary. `diaric`'s cohort statistics take an INFALLIBLE
//!   `FnMut(&S, &T) -> f64`, so a refusal inside scoring has no way out of the
//!   closure. [`cohort_stats_excluding`] and [`cohort_stats_assuming_disjoint`]
//!   carry it out by hand rather than letting a caller decide what number a
//!   failed score should be — an `unwrap_or(0.0)` inside that closure poisons a
//!   mean silently, which is the one failure AS-Norm exists to prevent.
//!
//! # The self-contamination choice is the caller's, and it stays that way
//!
//! `diaric` offers two entrypoints whose names carry the precondition, and this
//! module offers exactly the same two under the same names. It does **not**
//! offer a convenience that picks one:
//!
//! - [`cohort_stats_excluding`] — drops every cohort entry belonging to the
//!   speaker being normalized.
//! - [`cohort_stats_assuming_disjoint`] — scores the whole cohort. Correct only
//!   for a cohort held out from the library being scored.
//!
//! #123's cohort is "sampled from the library itself", so a speaker's own
//! entries are **guaranteed** to be selected: top-N takes the largest scores
//! and a self-match is the largest score obtainable — `1.0` for L2-normalized
//! embeddings under cosine, to within the rounding of that normalization, and
//! the maximum either way. A contaminated side still looks
//! perfectly healthy: its mean is high, its deviation is real, and every score
//! derived from it is wrong. No type can catch it, because only the caller
//! knows a probe's identity — a probe is a *new recording*, and whether the
//! person speaking in it is already in the library is precisely the question
//! being asked. So the promise lives in the method name, and the choice is made
//! at the call site or not at all.
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
//! A [`CohortStats`] depends only on its own side and the cohort, so it is
//! computed once per speaker and reused across every trial that speaker appears
//! in — `N·C` cohort scores instead of `N(N−1)·C`, which for a 1 000-profile
//! library against a 300-member cohort is 300 000 scores rather than 300
//! million. [`diaric::score_norm`]'s cost table has the derivation.
//!
//! # End to end
//!
//! ```
//! use coremlit::audio::speaker::{
//!   calibrate::{AsNormOptions, Cohort, Scoring, as_norm, cohort_stats_excluding},
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
//! // 1. The cohort: impostor profiles sampled from the caller's own library,
//! //    so it CONTAINS the people being scored. `coremlit` never sees the
//! //    store this came out of.
//! let mut cohort: Cohort<PersonId, _> = Cohort::new();
//! for id in 0..64u32 {
//!   cohort.push(id, scoring.prepare(&stored_centroid(id as usize))?);
//! }
//!
//! // 2. The trial: a profile already in the library, against a probe from a
//! //    recording that has just been diarized.
//! let alice: PersonId = 7;
//! let stored = scoring.prepare(&stored_centroid(alice as usize))?;
//! let probe = scoring.prepare(&probe_of(alice as usize))?;
//!
//! // 3. One side per speaker, EXCLUDING that speaker's own library entries —
//! //    the cohort was drawn from the same library, so `assuming_disjoint`
//! //    would be scoring Alice against herself. Both sides carry Alice's
//! //    identity: the probe is a new recording of the same person.
//! let stored_side = cohort_stats_excluding(&cohort, &alice, &stored, &options)?;
//! let probe_side = cohort_stats_excluding(&cohort, &alice, &probe, &options)?;
//! assert_eq!(stored_side.considered(), 63); // 64 members, less Alice's own
//!
//! // 4. The raw trial score, then the calibrated one a fixed threshold reads.
//! let raw = stored.score(&probe)?;
//! let normalized = as_norm(raw, &stored_side, &probe_side)?;
//! assert!(normalized.is_finite());
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
//! - **A third score source.** [`cohort_stats_excluding`] and
//!   [`cohort_stats_assuming_disjoint`] take a [`VoiceProfile`], so they cover
//!   the sources [`Scoring`] names and no others — they exist to carry a
//!   `coremlit` refusal out of an infallible closure, which a caller's own
//!   scoring function does not need. Genericity is not lost by that: [`Cohort`]
//!   is `diaric`'s own type, generic over its item type and re-exported whole,
//!   so a caller scoring in some other space fills a `Cohort<K, T>` with THEIR
//!   item type and calls
//!   [`Cohort::stats_excluding`](diaric::score_norm::Cohort::stats_excluding)
//!   directly with their own `FnMut(&S, &T) -> f64`, reaching the same
//!   [`as_norm`]. Nothing here has to be widened for that to work — and
//!   [`VoiceProfile`] is not in that road's way, because it is not on it.

use diaric::{
  embed::Embedding,
  plda::{PLDA_DIMENSION, RawEmbedding},
};

use crate::audio::speaker::{
  embed::EMBEDDING_DIM,
  error::{CalibrateError, ProfileLength, ScoringMismatch},
  extract::shared_plda_transform,
};

/// `diaric`'s AS-Norm vocabulary, re-exported so a caller of this door does not
/// need a direct `diaric` dependency to name what it takes and returns.
///
/// These are `diaric`'s own types, unchanged: [`Cohort`] is the container the
/// caller fills, [`AsNormOptions`] tunes the per-side statistics, and
/// [`as_norm`] is the final combination step this module deliberately does not
/// wrap — it is arithmetic over two [`CohortStats`] and a score, with nothing
/// `coremlit`-shaped left to add. A `coremlit`-side mirror of [`AsNormOptions`]
/// (the way [`OnlineOptions`](crate::audio::speaker::OnlineOptions) mirrors
/// `diaric`'s online configuration) would be two constants and two builders
/// with nowhere to drift to but out of step, so there is none.
///
/// One consequence, stated because it is not visible from the signatures:
/// [`AsNormOptions`]'s `serde` impls are `diaric`'s own and sit behind
/// `diaric`'s `serde` feature, which this crate does NOT enable — `coremlit`'s
/// `serde` feature covers `coremlit`'s types, [`Scoring`] included. A caller
/// serializing an AS-Norm configuration enables `serde` on their own `diaric`
/// dependency; feature unification then applies it here.
pub use diaric::score_norm::{
  AsNormOptions, Cohort, CohortEntry, CohortStats, DEFAULT_MIN_DEVIATION, DEFAULT_TOP_N,
  MAX_NORMALIZED_ERROR, MIN_COHORT_SCORES, as_norm,
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
/// caller's library holds those, and hands the identity back at
/// [`cohort_stats_excluding`].
///
/// # Why the score source is a tag and not a type parameter
///
/// Mixing sources is refused at runtime ([`CalibrateError::ScoringMismatch`])
/// rather than made unrepresentable by a `VoiceProfile<S>`. Two reasons, the
/// second deciding:
///
/// - the cohort type is `diaric`'s [`Cohort<K, T>`], so a type parameter here
///   would monomorphize the caller's whole cohort on a choice they naturally
///   make at run time — which score source to run *this* comparison in;
/// - #123's follow-up is a confusion experiment *between* score sources, whose
///   natural shape is a loop over `[Scoring::Cosine, Scoring::PldaCosine]`. A
///   type parameter turns that loop into duplicated generic code.
///
/// What the tag has to buy in exchange is that a mismatch can never be silent,
/// and it is: every path that reads two profiles compares the tags first, so a
/// cohort holding one foreign entry is refused rather than averaged.
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
  /// [`as_norm`] normalizes.
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
  pub fn score(&self, other: &Self) -> Result<f64, CalibrateError> {
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

/// Cohort statistics for `side`, **excluding** every cohort entry belonging to
/// `speaker`.
///
/// The entrypoint for a cohort sampled from the same library being scored —
/// #123's own arrangement, and the one where a speaker's material is guaranteed
/// to be selected into its own top-N. Exclusion is by identity: all of
/// `speaker`'s entries go, not merely the exact self-match, and only this
/// side's speaker is removed, never the partner's — which is what keeps a side
/// reusable across every trial that speaker appears in.
///
/// `speaker` is the identity of whoever `side` belongs to. For a probe from a
/// fresh recording that is the identity being *tested*, not one the probe
/// already carries.
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
///   AsNormOptions, Cohort, Scoring, cohort_stats_excluding,
/// };
/// # let raw = vec![0.5f32; coremlit::audio::speaker::embed::EMBEDDING_DIM];
/// // `f64` is `PartialEq` but not `Eq`, so it cannot name a speaker — and a
/// // `NaN` key would not match its own entry, keeping the self-match in.
/// let mut cohort: Cohort<f64, _> = Cohort::new();
/// cohort.push(f64::NAN, Scoring::Cosine.prepare(&raw).unwrap());
/// let side = Scoring::Cosine.prepare(&raw).unwrap();
/// let _ = cohort_stats_excluding(&cohort, &f64::NAN, &side, &AsNormOptions::new());
/// ```
///
/// # Errors
///
/// - [`CalibrateError::ScoringMismatch`] if any scored cohort entry was
///   prepared for a different [`Scoring`] than `side`.
/// - [`CalibrateError::ScoreNorm`] for `diaric`'s own refusals: an empty
///   selection (including a cohort that was entirely `speaker`), too few usable
///   scores, or a selected set that does not spread.
pub fn cohort_stats_excluding<K: Eq>(
  cohort: &Cohort<K, VoiceProfile>,
  speaker: &K,
  side: &VoiceProfile,
  options: &AsNormOptions,
) -> Result<CohortStats, CalibrateError> {
  let mut carried = None;
  let stats = cohort.stats_excluding(speaker, side, scorer(&mut carried), options);
  finish(carried, stats)
}

/// Cohort statistics for `side` over **every** cohort member.
///
/// The literature's own path, and its precondition is in the name: correct only
/// when the cohort is genuinely disjoint from the speakers being scored — a
/// held-out corpus, which is what Matějka et al. 2017 §2.1 assumes and what
/// every reference implementation satisfies structurally.
///
/// Reaching for this with a cohort drawn from the library being scored is the
/// self-contamination failure [`cohort_stats_excluding`] describes, and nothing
/// can catch it for you. See the module docs.
///
/// # Errors
///
/// As [`cohort_stats_excluding`], minus the self-exclusion case.
pub fn cohort_stats_assuming_disjoint<K>(
  cohort: &Cohort<K, VoiceProfile>,
  side: &VoiceProfile,
  options: &AsNormOptions,
) -> Result<CohortStats, CalibrateError> {
  let mut carried = None;
  let stats = cohort.stats_assuming_disjoint(side, scorer(&mut carried), options);
  finish(carried, stats)
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
    Ok(v) => v,
    Err(e) => {
      carried.get_or_insert(e);
      f64::NAN
    }
  }
}

/// Report a carried scoring refusal ahead of `diaric`'s own, so a mixed cohort
/// is named as a mismatch rather than as the `NonFiniteScore` the poison value
/// produces downstream.
fn finish(
  carried: Option<CalibrateError>,
  stats: Result<CohortStats, diaric::score_norm::Error>,
) -> Result<CohortStats, CalibrateError> {
  match carried {
    Some(e) => Err(e),
    None => stats.map_err(CalibrateError::ScoreNorm),
  }
}

#[cfg(test)]
mod tests;
