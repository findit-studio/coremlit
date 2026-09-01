//! **The alignment golden: known landmarks in, known aligned pixels out.**
//!
//! Every downstream face cosine passes through the 5-point similarity
//! transform, and a wrong transform raises no error — it moves every embedding
//! by an amount no shape check and no finiteness check can see. So the
//! transform gets a golden of its own, and the golden's expected values do not
//! come from the code under test.
//!
//! # Where the expected pixels come from
//!
//! `coremlit/conversion/face/align_oracle.py` — numpy only, deliberately
//! importing neither `skimage` nor OpenCV, so nothing in it can quietly become
//! a call into the thing it is meant to check. It:
//!
//! 1. solves the same least-squares similarity through a **different
//!    derivation** — the complex/linear formulation, whose minimiser is two dot
//!    products, where the Rust follows Umeyama's statement of the same problem;
//! 2. resamples with an inverse-mapped bilinear kernel and a constant-0 border,
//!    reproducing `cv2.warpAffine(img, M, (112, 112), borderValue=0.0)`;
//! 3. writes both the source crop and the 112×112×3 result as raw RGB8.
//!
//! Both files are committed, and both digests are pinned below, so
//! regenerating a fixture is a visible diff in two places rather than a silent
//! re-baseline.
//!
//! # Why these landmarks
//!
//! They are LITERAL, not derived from the ArcFace template. A golden whose
//! landmarks are a similarity image of the template moves WITH the template, so
//! a mutated template coordinate would leave the expected pixels unchanged and
//! the golden would pass over the mutation. With fixed landmarks the template's
//! own ten numbers are load-bearing for all 37 632 committed bytes.
//!
//! The crop is 64×48 — deliberately not square, so a width/height
//! transposition changes the output — and its three channels use three
//! different generators, so a channel permutation changes it too.
//!
//! Hermetic: two file reads, no model, no network.

use std::path::PathBuf;

use coremlit::embeddings::face::{
  ARCFACE_TEMPLATE, FaceAlign, FaceCrop, LANDMARK_COUNT, Point, SimilarityTransform, TEMPLATE_BYTES,
};
use sha2::{Digest, Sha256};

/// The literal landmarks the oracle was run with, in template order.
const LANDMARKS: [Point; LANDMARK_COUNT] = [
  Point::new(18.5, 16.0),
  Point::new(41.0, 13.5),
  Point::new(30.5, 25.0),
  Point::new(21.0, 35.5),
  Point::new(40.0, 33.0),
];

const CROP_WIDTH: usize = 64;
const CROP_HEIGHT: usize = 48;

/// SHA-256 of `align_crop_64x48_rgb8.bin` as the oracle wrote it.
const CROP_SHA256: &str = "a7d34a19107058c28c73633cc25b82a018fc279034d6670b45488022d5071ce0";

/// SHA-256 of `align_expected_112x112_rgb8.bin` as the oracle wrote it.
const EXPECTED_SHA256: &str = "0b04d1c71bd97ee3ea42f01fde36cd36282ed6ba4a85843613597fa6f4dc45c4";

/// The oracle's own solved matrix, printed by `align_oracle.py` in row-major
/// `[a, −b, tx, b, a, ty]` order — the same six numbers
/// `skimage`'s `tform.params[0:2, :]` holds.
const ORACLE_MATRIX: [f64; 6] = [
  1.787_506_585_347_667_3,
  -0.125_255_686_473_033_28,
  5.124_750_677_705_819,
  0.125_255_686_473_033_28,
  1.787_506_585_347_667_3,
  24.145_397_518_961_772,
];

fn fixture(name: &str) -> Vec<u8> {
  let path: PathBuf = [
    env!("CARGO_MANIFEST_DIR"),
    "tests",
    "face",
    "fixtures",
    name,
  ]
  .iter()
  .collect();
  std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn digest(bytes: &[u8]) -> String {
  use core::fmt::Write;

  Sha256::digest(bytes)
    .iter()
    .fold(String::new(), |mut acc, b| {
      let _ = write!(acc, "{b:02x}");
      acc
    })
}

#[test]
fn the_committed_fixtures_are_the_bytes_the_oracle_wrote() {
  let crop = fixture("align_crop_64x48_rgb8.bin");
  let expected = fixture("align_expected_112x112_rgb8.bin");
  assert_eq!(
    crop.len(),
    CROP_WIDTH * CROP_HEIGHT * 3,
    "the source crop is not 64×48×3"
  );
  assert_eq!(
    expected.len(),
    TEMPLATE_BYTES,
    "the expected template is not 112×112×3"
  );
  assert_eq!(
    digest(&crop),
    CROP_SHA256,
    "the source fixture changed; regenerate the expected pixels with \
     conversion/face/align_oracle.py and update BOTH digests deliberately"
  );
  assert_eq!(
    digest(&expected),
    EXPECTED_SHA256,
    "the expected fixture changed without its digest changing; a golden that can be re-baselined \
     silently is not a golden"
  );
}

#[test]
fn the_solved_transform_matches_the_oracles_to_the_last_digits() {
  let solved = SimilarityTransform::estimate(&LANDMARKS, &ARCFACE_TEMPLATE)
    .expect("the fixture landmarks are non-degenerate");
  let matrix = solved.matrix();
  for (index, (got, want)) in matrix.into_iter().zip(ORACLE_MATRIX).enumerate() {
    assert!(
      (got - want).abs() <= want.abs().mul_add(1e-12, 1e-12),
      "matrix entry {index}: Rust solved {got}, the oracle solved {want}. Two derivations of one \
       least-squares minimiser have diverged."
    );
  }
}

#[test]
fn aligned_pixels_match_the_committed_oracle_byte_for_byte() {
  let source = fixture("align_crop_64x48_rgb8.bin");
  let expected = fixture("align_expected_112x112_rgb8.bin");
  let crop =
    FaceCrop::new(&source, CROP_WIDTH, CROP_HEIGHT).expect("the fixture geometry is valid");
  let aligned = FaceAlign::to_template(crop, &LANDMARKS).expect("the fixture is solvable");

  let got = aligned.pixels();
  let mismatches: Vec<String> = got
    .iter()
    .zip(expected.iter())
    .enumerate()
    .filter(|(_, (g, e))| g != e)
    .take(8)
    .map(|(index, (g, e))| {
      let pixel = index / 3;
      format!(
        "byte {index} (row {}, col {}, channel {}): got {g}, expected {e}",
        pixel / 112,
        pixel % 112,
        index % 3
      )
    })
    .collect();
  assert!(
    mismatches.is_empty(),
    "{} of {TEMPLATE_BYTES} bytes differ from the oracle. First few:\n{}",
    got
      .iter()
      .zip(expected.iter())
      .filter(|(g, e)| g != e)
      .count(),
    mismatches.join("\n")
  );
}

#[test]
fn a_one_pixel_template_shift_is_far_outside_the_oracle_tolerance() {
  // The golden's own falsifier. `the_solved_transform_matches_the_oracles_to_the_last_digits`
  // compares against a matrix the ORACLE derived from its own copy of the
  // template, so it is a template check — but only if its tolerance is tight
  // enough to notice a template that moved. Shift one template coordinate by a
  // single pixel and the divergence must exceed that tolerance by orders of
  // magnitude, or the comparison could sleep through exactly the mutation it
  // exists to catch.
  let real = SimilarityTransform::estimate(&LANDMARKS, &ARCFACE_TEMPLATE).expect("solvable");
  let mut shifted = ARCFACE_TEMPLATE;
  shifted[0] = Point::new(ARCFACE_TEMPLATE[0].x() + 1.0, ARCFACE_TEMPLATE[0].y());
  let moved = SimilarityTransform::estimate(&LANDMARKS, &shifted).expect("still non-degenerate");

  let worst = real
    .matrix()
    .into_iter()
    .zip(moved.matrix())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f64, f64::max);
  assert!(
    worst > 1e-3,
    "a one-pixel template shift moved the solved matrix by only {worst}, which the oracle \
     comparison's 1e-12 tolerance would still be measuring against noise"
  );

  // And the shift has to be visible in image terms, not just in the matrix:
  // the same landmark must land somewhere materially different on the
  // template, which is what makes the committed pixels move with it.
  let (rx, ry) = real.apply(LANDMARKS[0]);
  let (mx, my) = moved.apply(LANDMARKS[0]);
  assert!(
    (rx - mx).hypot(ry - my) > 0.1,
    "a one-pixel template shift moved landmark 0 by only {} px on the template",
    (rx - mx).hypot(ry - my)
  );
}
