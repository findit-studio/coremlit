//! End-to-end parity: the Rust door against the fp32 ONNX reference.
//!
//! `model_io.rs` proves the artifact is the pinned bytes and declares the
//! pinned contract. Neither says the door reproduces the FUNCTION those bytes
//! encode. This does: for each of the 18 committed aligned crops it runs
//! `FaceEmbedder::embed` — the manifest's preprocessing, the fp16 CoreML
//! graph, and the door's own L2 — and compares against the fp32
//! `onnxruntime` embedding of the same crop.
//!
//! The reference is `tests/face/fixtures/onnx_reference.json`, cut by
//! `conversion/face/scripts/write_onnx_reference.py` from the same pinned
//! `w600k_r50.onnx` the conversion consumed. It is committed as DATA because a
//! gate cannot depend on an ONNX runtime — the `face` feature pulls none, and
//! issue #115's cross-platform `ort` road is not built — which is the shape
//! `granite` and `siglip` already use for their transformers-fp32 goldens.
//!
//! **Research-only weights.** See `tests/face/arcface/mod.rs`.
//!
//! # Where the ALIGNMENT is measured
//!
//! Not here, and the reason is in the fixtures: the five landmarks live in the
//! coordinate space of the full `~medium` NASA asset, whose bytes are not
//! committed, so the warp cannot be re-run. It is covered in two pieces
//! instead, and between them they cover the whole of it:
//!
//! - **the solve** — `the_rust_solve_reproduces_every_committed_alignment_matrix`
//!   below runs [`SimilarityTransform::estimate`] over each face's committed
//!   landmarks and compares against the matrix `align_oracle.py` solved. This
//!   is the half where a wrong answer raises no error and moves every cosine
//!   on this page;
//! - **the resampler** — `tests/face/align_golden.rs` compares all 37 632
//!   bytes of a warped template against the oracle's, and the oracle is
//!   bit-exact with `cv2.warpAffine` 4.x.
//!
//! [`SimilarityTransform::estimate`]: coremlit::embeddings::face::SimilarityTransform::estimate

#[path = "arcface/mod.rs"]
mod common;

use coremlit::embeddings::face::{AlignedFace, FaceEmbedder, FaceEmbedderOptions, arcface};

/// How far the Rust solve may sit from the oracle's, relative to the
/// magnitude of the parameter.
///
/// Both sides minimise the same least-squares problem over the same `f32`
/// landmarks in `f64`, differing only in summation order, so the residual is
/// float association and nothing else — measured at 4.4e-16 relative over the
/// 18 committed faces. `1e-9` is ~7 orders above that, deliberately: it is a
/// bound on "the two are the same solve", not a re-pin of one machine's
/// rounding. What it must NOT accommodate is a different solve, and the one
/// this kit knows about is `skimage`'s `f32` path, whose divergence
/// `src/embeddings/face/align` measures in the units it can be measured in —
/// five-bit source coordinates, 10 of 12 544 moved on the witness landmarks —
/// rather than in a parameter delta nobody has recorded.
const SOLVE_TOLERANCE: f64 = 1e-9;

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The reference corpus loads, is the size it claims, and its vectors are RAW.
///
/// The same claim `model_io.rs` makes of the graph, made here of the ORACLE
/// those outputs are compared against. A unit-norm reference would let the
/// parity gate below pass while measuring the wrong function — the door's own
/// L2 against a vector that had already had one.
#[test]
fn the_reference_corpus_loads_and_its_vectors_are_raw() {
  let reference = common::load_reference();
  assert_eq!(reference.faces.len(), 18, "18 committed crops");
  assert_eq!(reference.dim, 512);
  assert_eq!(
    reference
      .faces
      .iter()
      .map(|f| f.person.as_str())
      .collect::<std::collections::BTreeSet<_>>()
      .len(),
    6,
    "six identities"
  );
  for face in &reference.faces {
    let measured = face
      .reference
      .iter()
      .map(|v| f64::from(*v) * f64::from(*v))
      .sum::<f64>()
      .sqrt();
    assert!(
      (measured - face.reference_norm).abs() < 1e-3,
      "{}: recorded norm {} but the vector measures {measured}",
      face.id,
      face.reference_norm
    );
    assert!(
      measured > 2.0,
      "{}: the reference must be RAW; norm {measured:.4}",
      face.id
    );
  }
}

/// The references are distinct directions, so the comparison below is
/// discriminating rather than satisfied by any vector at all.
///
/// Every DIFFERENT-person pair must sit far under the parity floor. Two crops
/// that happened to share a direction would let a constant-output graph clear
/// the gate.
#[test]
fn the_reference_directions_are_distinct_across_identities() {
  let reference = common::load_reference();
  let mut worst = -1.0f64;
  for (i, a) in reference.faces.iter().enumerate() {
    for b in reference.faces.iter().skip(i + 1) {
      if a.person == b.person {
        continue;
      }
      let cos = common::cosine(&a.reference, &b.reference);
      assert!(
        cos < common::SANITY_COS,
        "{} and {} embed to the same direction (cos {cos:.6}); the parity gate could not tell \
         them apart",
        a.id,
        b.id
      );
      worst = worst.max(cos);
    }
  }
  eprintln!("[arcface] reference: closest different-person pair = {worst:.6}");
}

/// **The alignment leg.** The Rust solve reproduces every committed matrix.
///
/// [`SimilarityTransform::estimate`][est] over each face's five landmarks
/// against [`ARCFACE_TEMPLATE`][tpl], compared with the 2×3 matrix
/// `align_oracle.py` solved and `faces/manifest.json` records. Both promote
/// the `f32` landmarks to `f64` and reach the same minimiser through the
/// complex/linear formulation rather than an SVD, so the two are expected to
/// agree to float-association noise — and the reason this matters is that
/// alignment is the one place in this kit where a wrong answer raises no error
/// and moves every cosine downstream.
///
/// It is hermetic: the landmarks and the matrices are committed, and no
/// artifact and no source photograph are needed.
///
/// [est]: coremlit::embeddings::face::SimilarityTransform::estimate
/// [tpl]: coremlit::embeddings::face::ARCFACE_TEMPLATE
#[test]
fn the_rust_solve_reproduces_every_committed_alignment_matrix() {
  let reference = common::load_reference();
  let mut worst = 0.0f64;
  for face in &reference.faces {
    let solved = common::solved_transform(face).matrix();
    for (index, (got, want)) in solved.iter().zip(face.align_matrix).enumerate() {
      let allowed = SOLVE_TOLERANCE * want.abs().max(1.0);
      let delta = (got - want).abs();
      worst = worst.max(delta / want.abs().max(1.0));
      assert!(
        delta <= allowed,
        "{}: matrix[{index}] solved {got} against the oracle's {want} (delta {delta:.3e}, \
         allowed {allowed:.3e})",
        face.id
      );
    }
  }
  eprintln!("[arcface] solve: worst relative deviation from the oracle = {worst:.3e}");
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// **The end-to-end gate.** The door reproduces the fp32 ONNX reference on the
/// recommended arm, at issue #115's own acceptance floor.
///
/// One `embed` call over all 18 crops, so the door's chunking to the
/// artifact's batch of 1 is exercised as well as the arithmetic. The worst
/// cosine is reported whatever the verdict: a gate that only speaks when it
/// fails leaves nobody able to see the margin shrinking.
///
/// The recipe measured `1 - cos` at 2.2e-4 worst on this arm — 45x inside the
/// floor — and that number is deliberately NOT the assertion; see
/// [`common::SANITY_COS`].
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn the_door_reproduces_the_onnx_reference_on_the_recommended_arm() {
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
  assert_eq!(embeddings.len(), reference.faces.len());

  let mut worst = (1.0f64, String::new());
  for (face, embedding) in reference.faces.iter().zip(&embeddings) {
    assert_eq!(embedding.dim(), reference.dim, "{}: width", face.id);
    assert!(
      embedding.as_slice().iter().all(|v| v.is_finite()),
      "{}: non-finite component",
      face.id
    );
    let cos = common::cosine(embedding.as_slice(), &face.reference);
    eprintln!(
      "[arcface] {:?} {}: cos vs ONNX fp32 = {cos:.8}",
      arcface::RECOMMENDED_COMPUTE,
      face.id
    );
    if cos < worst.0 {
      worst = (cos, face.id.clone());
    }
  }
  eprintln!(
    "[arcface] {:?} worst cos = {:.8} ({}), 1-cos = {:.3e}, floor {}",
    arcface::RECOMMENDED_COMPUTE,
    worst.0,
    worst.1,
    1.0 - worst.0,
    common::SANITY_COS
  );
  assert!(
    worst.0 >= common::SANITY_COS,
    "{}: cos {:.8} < {} on {:?}",
    worst.1,
    worst.0,
    common::SANITY_COS,
    arcface::RECOMMENDED_COMPUTE
  );
}

/// The cross-face GEOMETRY survives too, not just each vector separately.
///
/// A preprocessing error that affects all 18 crops the same way could clear
/// the per-face floor while rotating the whole corpus, and the thing this kit
/// actually sells is the ANGLES between faces. So the pairwise cosines the
/// door produces are compared against the pairwise cosines the ONNX produces
/// — the check `verify_arcface.py` makes for the same reason.
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn cross_face_geometry_matches_the_reference() {
  /// The fp16 graph reproduces each vector to `1 - cos ~ 2e-4`, so a pairwise
  /// cosine — a difference of two such vectors' directions — cannot move by
  /// more than a few multiples of that. `1e-2` is two orders looser and is the
  /// same bound `tests/identity/parity_embed.rs` holds.
  const MAX_PAIR_DELTA: f64 = 1e-2;

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
    .map(|face| AlignedFace::from_template_pixels(&face.crop).expect("wrap crop"))
    .collect();
  let ours = embedder.embed(&faces).expect("embed the whole corpus");

  let mut worst = (0.0f64, String::new());
  for (i, a) in reference.faces.iter().enumerate() {
    for (j, b) in reference.faces.iter().enumerate().skip(i + 1) {
      let theirs = common::cosine(&a.reference, &b.reference);
      let mine = common::cosine(ours[i].as_slice(), ours[j].as_slice());
      let delta = (mine - theirs).abs();
      if delta > worst.0 {
        worst = (delta, format!("{}/{}", a.id, b.id));
      }
      assert!(
        delta <= MAX_PAIR_DELTA,
        "{}/{}: cross-face cosine moved by {delta:.3e} (reference {theirs:.8}, door {mine:.8})",
        a.id,
        b.id
      );
    }
  }
  eprintln!(
    "[arcface] worst pairwise-cosine drift vs the reference: {:.3e} ({})",
    worst.0, worst.1
  );
}
