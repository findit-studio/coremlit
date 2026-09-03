//! Compute-placement characterisation (measured, never marketed).
//!
//! One table, four arms, over the 18 committed crops: cold load, warm predict,
//! throughput, and the worst cosine against the committed fp32 ONNX
//! reference. It is printed on every host and asserted only where the claim is
//! PORTABLE — that every arm is finite and clears issue #115's parity floor,
//! and that the recommended arm is one of the four rather than a special case.
//!
//! # Why the timings are printed and not pinned
//!
//! The recipe measured this on one machine (`conversion/face/README.md`'s
//! observed-toolchain table names it):
//!
//! | arm | cold load ms | warm predict ms (median, range) | faces/s | min cos |
//! |---|---|---|---|---|
//! | `All` | 357 | 4.46 (3.57 – 6.41) | 224 | 0.999781 |
//! | `CpuAndGpu` | 592 | 8.66 (4.86 – 10.59) | 115 | 0.999999 |
//! | `CpuOnly` | 95 | 10.42 (9.92 – 23.09) | 96 | 0.999835 |
//! | **`CpuAndNeuralEngine`** | **160** | **3.48 (2.99 – 3.76)** | **287** | 0.999780 |
//!
//! Which arm is fastest is a property of a Neural Engine generation, its
//! compiler and its firmware, and CoreML contracts none of that to reproduce
//! across chips or macOS builds (#36). `siglip`'s placement suite learned that
//! the expensive way — a band measured on one host went red on a CI runner
//! that had broken nothing. So the ORDERING is not asserted here at all; what
//! is asserted is the parity floor, which is portable, and the finiteness,
//! which is a correctness property rather than a performance one.
//!
//! **Research-only weights.** See `tests/face/arcface/mod.rs`.

#[path = "arcface/mod.rs"]
mod common;

use std::time::Instant;

use coremlit::{
  ComputeUnits,
  embeddings::face::{AlignedFace, FaceEmbedder, FaceEmbedderOptions, arcface},
};

/// The four placements, in the order the recipe's sweep reports them.
const ARMS: [ComputeUnits; 4] = [
  ComputeUnits::All,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuOnly,
  ComputeUnits::CpuAndNeuralEngine,
];

/// One arm's measurement.
struct Arm {
  compute: ComputeUnits,
  cold_load_ms: f64,
  corpus_ms: f64,
  worst_cos: f64,
  worst_id: String,
}

impl Arm {
  /// Faces per second over the warm corpus pass, at this arm's measured rate.
  fn faces_per_second(&self, faces: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "18 faces is exact in f64")]
    let n = faces as f64;
    n * 1000.0 / self.corpus_ms
  }
}

/// Load on `compute`, embed the whole corpus once to warm the graph, then
/// measure a second pass and score it against the reference.
fn measure(compute: ComputeUnits, reference: &common::Reference, faces: &[AlignedFace]) -> Arm {
  let started = Instant::now();
  let embedder = FaceEmbedder::load(
    common::model_path(),
    arcface::MODEL,
    FaceEmbedderOptions::new().with_compute(compute),
  )
  .unwrap_or_else(|e| panic!("load under {compute:?}: {e}"));
  let cold_load_ms = started.elapsed().as_secs_f64() * 1000.0;

  // The first pass pays the first-predict cost the recipe reports separately;
  // the second is what the number below describes.
  let _warm = embedder
    .embed(faces)
    .unwrap_or_else(|e| panic!("warm-up under {compute:?}: {e}"));
  let started = Instant::now();
  let embeddings = embedder
    .embed(faces)
    .unwrap_or_else(|e| panic!("embed under {compute:?}: {e}"));
  let corpus_ms = started.elapsed().as_secs_f64() * 1000.0;

  let mut worst = (1.0f64, String::new());
  for (face, embedding) in reference.faces.iter().zip(&embeddings) {
    assert!(
      embedding.as_slice().iter().all(|v| v.is_finite()),
      "{compute:?} produced a non-finite component for {}. A NaN here is not a precision \
       finding — it is a vector whose cosine against anything is NaN, which every downstream \
       threshold silently fails open on.",
      face.id
    );
    let cos = common::cosine(embedding.as_slice(), &face.reference);
    if cos < worst.0 {
      worst = (cos, face.id.clone());
    }
  }
  Arm {
    compute,
    cold_load_ms,
    corpus_ms,
    worst_cos: worst.0,
    worst_id: worst.1,
  }
}

/// **The four-arm table.**
///
/// Every arm is loaded, warmed and scored; the table is printed; and the
/// portable half is asserted — no NaN anywhere, every arm at or above the
/// parity floor, and the recommended arm among the four. A caller choosing a
/// placement is then choosing between measured arms rather than guessing.
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn every_placement_is_finite_and_within_parity() {
  let reference = common::load_reference();
  let faces: Vec<AlignedFace> = reference
    .faces
    .iter()
    .map(|face| {
      AlignedFace::from_template_pixels(&face.crop)
        .unwrap_or_else(|e| panic!("{}: wrap crop: {e}", face.id))
    })
    .collect();

  let measured: Vec<Arm> = ARMS
    .iter()
    .map(|compute| measure(*compute, &reference, &faces))
    .collect();

  eprintln!("[arcface] placement over {} faces", faces.len());
  eprintln!(
    "[arcface] {:<20} {:>12} {:>12} {:>10} {:>12}  worst face",
    "arm", "cold load ms", "corpus ms", "faces/s", "min cos"
  );
  for arm in &measured {
    eprintln!(
      "[arcface] {:<20} {:>12.1} {:>12.2} {:>10.0} {:>12.7}  {}",
      format!("{:?}", arm.compute),
      arm.cold_load_ms,
      arm.corpus_ms,
      arm.faces_per_second(faces.len()),
      arm.worst_cos,
      arm.worst_id
    );
  }

  for arm in &measured {
    assert!(
      arm.worst_cos >= common::SANITY_COS,
      "{:?}: worst cos {:.8} on {} is under the parity floor {}. Every arm must reproduce the \
       ONNX — an arm that does not is not a slower arm, it is a different function.",
      arm.compute,
      arm.worst_cos,
      arm.worst_id,
      common::SANITY_COS
    );
  }

  let recommended = measured
    .iter()
    .find(|arm| arm.compute == arcface::RECOMMENDED_COMPUTE)
    .expect("the recommended arm must be one of the four measured here");
  eprintln!(
    "[arcface] recommended {:?}: worst cos {:.8}, 1-cos {:.3e}, {:.0} faces/s",
    recommended.compute,
    recommended.worst_cos,
    1.0 - recommended.worst_cos,
    recommended.faces_per_second(faces.len())
  );
  assert!(
    recommended.worst_cos >= common::SANITY_COS,
    "the RECOMMENDED arm {:?} is under the parity floor: {:.8} on {}. Whatever else changes \
     about a host, the arm this crate tells callers to use has to reproduce the model.",
    recommended.compute,
    recommended.worst_cos,
    recommended.worst_id
  );
}
