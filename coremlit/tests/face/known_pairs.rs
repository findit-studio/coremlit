//! Known pairs at **InsightFace's own operating point**, not this kit's.
//!
//! 18 same-person and 135 different-person pairs over six identities, scored
//! against the two constants InsightFace's own recognition demo rules on:
//! `sim >= 0.28` "They ARE the same person", `sim < 0.2` "They are NOT the
//! same person" (`web-demos/src_recognition/main.py` @
//! `f8aa2c17e18044a86bbfa04be40e00cd2ff40a4f`, sha256 `24a94180…9509`).
//! Neither number is fitted to this set, and both are read out of the
//! committed reference fixture rather than written here, so a threshold cannot
//! be quietly relaxed to make a run pass.
//!
//! # Six identities cannot estimate a false-accept rate, and none is claimed
//!
//! What IS claimed is exactly what the assertions say: at a threshold taken
//! from upstream, no pair in this set is misclassified. The corpus is 18
//! photographs of six people, every one a work of the U.S. federal government
//! in the public domain — no LFW, no CelebA, no CFP, no WebFace, nothing
//! scraped (`tests/face/fixtures/PROVENANCE.md`).
//!
//! # The binding pair is the frontal-to-profile one
//!
//! `whitson_iss005e07178` — 2002, a full side view, yaw proxy −0.82 — against
//! `whitson_NHQ201803020004` — 2018, frontal. Sixteen years and a profile
//! apart, and it sets `min same-person` in every measurement the recipe took.
//! That pair is in the set deliberately: issue #115 measured AuraFace
//! splitting identity on frontal-to-profile pairs 38.55 % of the time against
//! `buffalo_l`'s 2.22 %, so a fixture set with no profile in it would not
//! exercise the regime that decided the model choice.
//!
//! **Research-only weights.** See `tests/face/arcface/mod.rs`.

#[path = "arcface/mod.rs"]
mod common;

use coremlit::embeddings::face::{AlignedFace, FaceEmbedder, FaceEmbedderOptions, arcface};

/// The pair counts a set of six identities × three photographs determines.
/// Spelled here so a fixture set that silently lost a face is a failure rather
/// than a smaller, easier corpus.
const SAME_PAIRS: usize = 18;
/// See [`SAME_PAIRS`].
const DIFFERENT_PAIRS: usize = 135;

/// One population's extremes, and the pair that set each.
struct Separation {
  min_same: f64,
  max_different: f64,
  worst_same: (String, String),
  worst_different: (String, String),
  same_pairs: usize,
  different_pairs: usize,
}

impl Separation {
  /// The margin between the two populations. Positive means they do not touch.
  fn margin(&self) -> f64 {
    self.min_same - self.max_different
  }
}

/// Score every pair of `vectors`, which must be parallel to `faces`.
fn separate(faces: &[common::Face], vectors: &[&[f32]]) -> Separation {
  let mut min_same = (2.0f64, String::new(), String::new());
  let mut max_different = (-2.0f64, String::new(), String::new());
  let (mut same_pairs, mut different_pairs) = (0usize, 0usize);
  for (i, a) in faces.iter().enumerate() {
    for (j, b) in faces.iter().enumerate().skip(i + 1) {
      let cos = common::cosine(vectors[i], vectors[j]);
      if a.person == b.person {
        same_pairs += 1;
        if cos < min_same.0 {
          min_same = (cos, a.id.clone(), b.id.clone());
        }
      } else {
        different_pairs += 1;
        if cos > max_different.0 {
          max_different = (cos, a.id.clone(), b.id.clone());
        }
      }
    }
  }
  Separation {
    min_same: min_same.0,
    max_different: max_different.0,
    worst_same: (min_same.1, min_same.2),
    worst_different: (max_different.1, max_different.2),
    same_pairs,
    different_pairs,
  }
}

/// Assert one path's separation at InsightFace's thresholds, printing the
/// margin and both binding pairs whatever the verdict.
fn assert_separated(what: &str, separation: &Separation, reference: &common::Reference) {
  eprintln!(
    "[arcface] {what}: {} same / {} different  min same {:.4} ({} / {})  max different {:.4} \
     ({} / {})  margin {:+.4}",
    separation.same_pairs,
    separation.different_pairs,
    separation.min_same,
    separation.worst_same.0,
    separation.worst_same.1,
    separation.max_different,
    separation.worst_different.0,
    separation.worst_different.1,
    separation.margin()
  );
  assert_eq!(separation.same_pairs, SAME_PAIRS);
  assert_eq!(separation.different_pairs, DIFFERENT_PAIRS);
  assert!(
    separation.min_same >= reference.same_min,
    "{what}: same-person pair ({} / {}) scores {:.4}, under InsightFace's own {} \"they ARE the \
     same person\" line — an identity SPLIT",
    separation.worst_same.0,
    separation.worst_same.1,
    separation.min_same,
    reference.same_min
  );
  assert!(
    separation.max_different < reference.different_max,
    "{what}: different-person pair ({} / {}) scores {:.4}, at or above InsightFace's own {} \
     \"they are NOT the same person\" line — an identity MERGE",
    separation.worst_different.0,
    separation.worst_different.1,
    separation.max_different,
    reference.different_max
  );
  assert!(
    separation.margin() > 0.0,
    "{what}: the two populations touch (margin {:+.4}); no threshold separates this set",
    separation.margin()
  );
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The committed reference itself separates, and the pair that binds it is the
/// frontal-to-profile one this set was built to contain.
///
/// It runs with no artifact, and it is what makes the gated assertion below a
/// statement about the CoreML bundle rather than about the corpus: if this
/// ever reds, the fixtures moved and the model gate would be measuring a
/// different set.
#[test]
fn the_onnx_reference_separates_at_insightfaces_operating_point() {
  let reference = common::load_reference();
  let vectors: Vec<&[f32]> = reference
    .faces
    .iter()
    .map(|f| f.reference.as_slice())
    .collect();
  let separation = separate(&reference.faces, &vectors);
  assert_separated("ONNX fp32 (committed reference)", &separation, &reference);

  // The fixture's own recorded statistics, recomputed here rather than
  // believed — the file is data, and data can be edited.
  assert!(
    (separation.min_same - reference.reference_min_same).abs() < 1e-9,
    "the fixture records min same {} but its vectors give {}",
    reference.reference_min_same,
    separation.min_same
  );
  assert!(
    (separation.max_different - reference.reference_max_different).abs() < 1e-9,
    "the fixture records max different {} but its vectors give {}",
    reference.reference_max_different,
    separation.max_different
  );

  // Named, not merely recorded: this pair is the reason the set has a profile
  // in it at all.
  let binding = [
    separation.worst_same.0.as_str(),
    separation.worst_same.1.as_str(),
  ];
  assert!(
    binding.contains(&"whitson_iss005e07178") && binding.contains(&"whitson_NHQ201803020004"),
    "the binding same-person pair is {binding:?}, not Whitson's 2002 profile against her 2018 \
     frontal. Either the corpus changed or the hardest pair is no longer the one this set was \
     built around — say which in PROVENANCE.md before re-baselining."
  );
  assert_eq!(
    reference.worst_same_ids,
    [
      separation.worst_same.0.clone(),
      separation.worst_same.1.clone()
    ],
    "the fixture's recorded worst same-person pair is not the one its vectors give"
  );
  assert_eq!(
    reference.worst_different_ids,
    [
      separation.worst_different.0.clone(),
      separation.worst_different.1.clone()
    ],
    "the fixture's recorded worst different-person pair is not the one its vectors give"
  );
}

/// The thresholds are upstream's, in the direction that matters.
///
/// A band between them is InsightFace's "LIKELY TO be the same person", so
/// `same_min > different_max` is what makes the pair of constants a
/// classifier at all. Pinned because a fixture edit that swapped or collapsed
/// them would make every assertion above trivially satisfiable.
#[test]
fn the_operating_point_is_insightfaces_own() {
  let reference = common::load_reference();
  assert!((reference.same_min - 0.28).abs() < 1e-12);
  assert!((reference.different_max - 0.20).abs() < 1e-12);
  assert!(
    reference.same_min > reference.different_max,
    "the accept line must sit above the reject line"
  );
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// **The shipped path separates the known pairs**, on the recommended arm, at
/// InsightFace's own operating point.
///
/// This is the claim the door's users actually rely on, and the one no shape
/// check can make: a graph fed the wrong channel order still returns a
/// plausible 512-d vector, and what it loses is exactly this separation. The
/// recipe measured that directly — BGR feeding drops the worst same-person
/// pair to 0.2547, through the 0.28 line — so a silent flip of
/// `Preprocessing::ARCFACE` fails HERE and nowhere else.
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn the_door_separates_the_known_pairs_at_insightfaces_threshold() {
  let reference = common::load_reference();
  let embedder = FaceEmbedder::load(
    common::model_path(),
    arcface::MODEL,
    FaceEmbedderOptions::new().with_compute(arcface::RECOMMENDED_COMPUTE),
  )
  .expect("load embedder");
  let faces: Vec<AlignedFace> = reference
    .faces
    .iter()
    .map(|face| {
      AlignedFace::from_template_pixels(&face.crop)
        .unwrap_or_else(|e| panic!("{}: wrap crop: {e}", face.id))
    })
    .collect();
  let embeddings = embedder.embed(&faces).expect("embed the whole corpus");
  let vectors: Vec<&[f32]> = embeddings
    .iter()
    .map(coremlit::embeddings::face::FaceEmbedding::as_slice)
    .collect();

  let separation = separate(&reference.faces, &vectors);
  assert_separated(
    &format!("CoreML fp16 {:?}", arcface::RECOMMENDED_COMPUTE),
    &separation,
    &reference,
  );

  // The fp16 path may not separate materially WORSE than the fp32 reference:
  // a margin that collapsed while every pair still landed on the right side of
  // the line is a regression the thresholds alone would not catch. The recipe
  // measured +0.1467 against the reference's +0.1484, so the tolerance is the
  // size of that gap and not a bound anybody hopes for.
  let margin_loss =
    reference.reference_min_same - reference.reference_max_different - separation.margin();
  eprintln!("[arcface] margin loss vs the fp32 reference: {margin_loss:+.4}");
  assert!(
    margin_loss < 0.02,
    "the fp16 path's margin is {:+.4} against the reference's {:+.4} — a loss of {margin_loss:+.4}",
    separation.margin(),
    reference.reference_min_same - reference.reference_max_different
  );
}
