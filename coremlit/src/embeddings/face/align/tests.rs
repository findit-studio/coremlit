//! Unit gates for the 5-point similarity alignment.
//!
//! Three of these establish the transform WITHOUT appealing to a second
//! implementation of it: an optimality proof by perturbation, an exact-fit
//! proof on landmarks that are a similarity image of the template, and an
//! analytic proof that the sampler is bilinear. The committed-pixel golden
//! (`tests/face/align_golden.rs`) is the fourth leg and the only one that can
//! see the template's own numbers.

use super::*;

/// A crop whose pixels are the linear function `2x + 3y` in every channel —
/// linear, so bilinear resampling of it is EXACT and the expected value of a
/// warped pixel is arithmetic rather than a golden.
fn linear_crop(width: usize, height: usize) -> Vec<u8> {
  let mut data = vec![0u8; width * height * 3];
  for y in 0..height {
    for x in 0..width {
      let value = u8::try_from(2 * x + 3 * y).expect("crop kept below 256");
      for channel in 0..3 {
        data[(y * width + x) * 3 + channel] = value;
      }
    }
  }
  data
}

/// The fixture landmarks the committed golden uses. Literal, never derived
/// from [`ARCFACE_TEMPLATE`].
const FIXTURE_LANDMARKS: [Point; LANDMARK_COUNT] = [
  Point::new(18.5, 16.0),
  Point::new(41.0, 13.5),
  Point::new(30.5, 25.0),
  Point::new(21.0, 35.5),
  Point::new(40.0, 33.0),
];

/// `Σ ‖M·pᵢ − qᵢ‖²` for an arbitrary parameter vector — the objective
/// [`SimilarityTransform::estimate`] claims to minimise, written out here so
/// the claim can be tested without naming a formula for the minimiser.
fn residual(
  params: [f64; 4],
  source: &[Point; LANDMARK_COUNT],
  target: &[Point; LANDMARK_COUNT],
) -> f64 {
  let [a, b, tx, ty] = params;
  let mut total = 0.0;
  for (s, t) in source.iter().zip(target.iter()) {
    let (px, py) = (f64::from(s.x()), f64::from(s.y()));
    let mapped_x = a * px - b * py + tx;
    let mapped_y = b * px + a * py + ty;
    let dx = mapped_x - f64::from(t.x());
    let dy = mapped_y - f64::from(t.y());
    total += dx * dx + dy * dy;
  }
  total
}

#[test]
fn arcface_template_matches_the_pinned_upstream_constants() {
  // deepinsight/insightface, python-package/insightface/utils/face_align.py,
  // commit ffa12d315041c0505b077c7ff057ca914bb8dc7e. Written out again rather
  // than referenced so a silent edit to the constant is a diff in two places.
  let expected: [(f32, f32); LANDMARK_COUNT] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
  ];
  for (index, (point, (x, y))) in ARCFACE_TEMPLATE.iter().zip(expected).enumerate() {
    assert_eq!(
      (point.x(), point.y()),
      (x, y),
      "template point {index} drifted from the pinned upstream value"
    );
  }
  assert_eq!(TEMPLATE_SIZE, 112);
  assert_eq!(TEMPLATE_BYTES, 112 * 112 * 3);
}

#[test]
fn exact_similarity_landmarks_recover_the_analytic_inverse() {
  // Landmarks built as an EXACT similarity image of the template: scale 0.5,
  // rotation 0.3 rad, translation (12, -7). The recovered transform must undo
  // exactly that, and the two numbers checked — scale and rotation — come from
  // the construction, not from any inverse formula.
  let (scale, theta) = (0.5f64, 0.3f64);
  let (tx, ty) = (12.0f64, -7.0f64);
  let (cos, sin) = (theta.cos() * scale, theta.sin() * scale);
  let mut landmarks = [Point::new(0.0, 0.0); LANDMARK_COUNT];
  for (slot, template) in landmarks.iter_mut().zip(ARCFACE_TEMPLATE.iter()) {
    let (x, y) = (f64::from(template.x()), f64::from(template.y()));
    *slot = Point::new(
      (cos * x - sin * y + tx) as f32,
      (sin * x + cos * y + ty) as f32,
    );
  }

  let recovered = SimilarityTransform::estimate(&landmarks, &ARCFACE_TEMPLATE)
    .expect("a non-degenerate similarity image is solvable");

  assert!(
    (recovered.scale() - 1.0 / scale).abs() < 1e-4,
    "recovered scale {} is not 1/{scale}",
    recovered.scale()
  );
  assert!(
    (recovered.rotation() + theta).abs() < 1e-5,
    "recovered rotation {} is not -{theta}",
    recovered.rotation()
  );
  for (i, (landmark, template)) in landmarks.iter().zip(ARCFACE_TEMPLATE.iter()).enumerate() {
    let (mx, my) = recovered.apply(*landmark);
    assert!(
      (mx - f64::from(template.x())).abs() < 2e-3 && (my - f64::from(template.y())).abs() < 2e-3,
      "landmark {i} maps to ({mx}, {my}), not onto its template point"
    );
  }
}

#[test]
fn recovered_transform_is_the_least_squares_minimiser() {
  // The optimality proof, and the only check here that names no formula for
  // the answer: perturb each solved parameter in both directions and the
  // residual must RISE. Any mutation of the solve moves the answer off the
  // minimum, and a minimum is exactly what this detects.
  let solved = SimilarityTransform::estimate(&FIXTURE_LANDMARKS, &ARCFACE_TEMPLATE)
    .expect("the fixture landmarks are non-degenerate");
  let params = [solved.a(), solved.b(), solved.tx(), solved.ty()];
  let best = residual(params, &FIXTURE_LANDMARKS, &ARCFACE_TEMPLATE);

  for (index, name) in ["a", "b", "tx", "ty"].into_iter().enumerate() {
    for step in [-1e-3f64, 1e-3] {
      let mut perturbed = params;
      perturbed[index] += step;
      let worse = residual(perturbed, &FIXTURE_LANDMARKS, &ARCFACE_TEMPLATE);
      assert!(
        worse > best,
        "moving {name} by {step} did not increase the residual ({worse} vs {best}); the solved \
         parameters are not the least-squares minimiser"
      );
    }
  }
}

#[test]
fn bilinear_sampling_is_exact_on_an_affine_ramp() {
  // Bilinear interpolation reproduces a linear function exactly, so with a
  // `2x + 3y` crop and a pure sub-pixel translation the expected template
  // pixel is `2u + 3v + 11` by arithmetic — no golden, no oracle. Nearest
  // neighbour would give `2u + 3v + 9`.
  let (width, height) = (40usize, 30usize);
  let data = linear_crop(width, height);
  let crop = FaceCrop::new(&data, width, height).expect("geometry is valid");
  let inverse = SimilarityTransform::new(1.0, 0.0, 3.25, 1.5);
  let warped = warp_bilinear(crop, &inverse);

  let mut checked = 0usize;
  for v in 0..TEMPLATE_SIZE {
    for u in 0..TEMPLATE_SIZE {
      // Only pixels whose whole 2×2 tap window lies inside the crop: outside
      // it the constant-0 border is the correct answer, not the ramp.
      if u + 4 >= width || v + 3 >= height {
        continue;
      }
      let expected = u8::try_from(2 * u + 3 * v + 11).expect("ramp stays below 256");
      let base = (v * TEMPLATE_SIZE + u) * 3;
      assert_eq!(
        &warped[base..base + 3],
        &[expected, expected, expected],
        "template pixel ({u}, {v}) is not the exact bilinear value of the ramp"
      );
      checked += 1;
    }
  }
  assert!(checked > 500, "only {checked} pixels were inside the crop");
}

#[test]
fn taps_outside_the_crop_contribute_the_zero_border() {
  // Landmarks so tight that the template's own corners map far outside the
  // crop. `cv2.warpAffine(..., borderValue=0.0)` reads black there; clamping
  // an edge pixel across the face instead would be a different, and wrong,
  // convention.
  // The landmarks span ~10 px where the template spans 112, so the template's
  // own corners map well outside a 20×20 crop while its centre stays inside.
  let (width, height) = (20usize, 20usize);
  let data = vec![200u8; width * height * 3];
  let crop = FaceCrop::new(&data, width, height).expect("geometry is valid");
  let landmarks = [
    Point::new(8.0, 9.0),
    Point::new(18.0, 9.0),
    Point::new(13.0, 14.0),
    Point::new(9.0, 19.0),
    Point::new(17.0, 19.0),
  ];
  let aligned = FaceAlign::to_template(crop, &landmarks).expect("solvable");
  let pixels = aligned.pixels();
  assert_eq!(
    &pixels[0..3],
    &[0, 0, 0],
    "the top-left corner is not the border"
  );
  let last = TEMPLATE_BYTES - 3;
  assert_eq!(
    &pixels[last..],
    &[0, 0, 0],
    "the bottom-right corner is not the border"
  );
  let centre = ((TEMPLATE_SIZE / 2) * TEMPLATE_SIZE + TEMPLATE_SIZE / 2) * 3;
  assert_eq!(
    &pixels[centre..centre + 3],
    &[200, 200, 200],
    "the template centre should sample the crop's flat interior"
  );
}

#[test]
fn inverse_round_trips_a_point() {
  let forward = SimilarityTransform::new(1.7875, -0.1252, 5.1247, 24.1454);
  let back = forward.inverse().expect("a nonzero scale is invertible");
  let p = Point::new(11.25, -3.5);
  let (fx, fy) = forward.apply(p);
  let mapped = Point::new(fx as f32, fy as f32);
  let (rx, ry) = back.apply(mapped);
  assert!(
    (rx - f64::from(p.x())).abs() < 1e-4 && (ry - f64::from(p.y())).abs() < 1e-4,
    "round trip landed at ({rx}, {ry}), not ({}, {})",
    p.x(),
    p.y()
  );
}

#[test]
fn a_zero_scale_transform_has_no_inverse() {
  assert!(
    SimilarityTransform::new(0.0, 0.0, 5.0, 5.0)
      .inverse()
      .is_none()
  );
}

#[test]
fn estimate_itself_rejects_landmarks_with_no_spread() {
  // Added after a mutation SURVIVED. Deleting `estimate`'s own spread guard
  // left `coincident_landmarks_are_rejected` GREEN: a NaN `a`/`b` makes
  // `inverse()` return `None`, and `to_template`'s backstop raises the same
  // `DegenerateLandmarks(0.0)` the guard would have. The end-to-end gate
  // therefore could not tell the guard from the backstop — and `estimate` is
  // PUBLIC, so with the guard gone a caller using it directly would get a
  // silent all-NaN transform. This gate calls `estimate` with no warp in the
  // way, so only the guard can satisfy it.
  let coincident = [Point::new(8.0, 8.0); LANDMARK_COUNT];
  let error = SimilarityTransform::estimate(&coincident, &ARCFACE_TEMPLATE)
    .expect_err("five coincident points determine no similarity");
  assert!(
    matches!(error, Error::DegenerateLandmarks(payload) if payload.spread() == 0.0),
    "expected DegenerateLandmarks(0.0) from `estimate` itself, got {error:?}"
  );
}

#[test]
fn coincident_landmarks_are_rejected() {
  let data = vec![0u8; 16 * 16 * 3];
  let crop = FaceCrop::new(&data, 16, 16).expect("geometry is valid");
  let landmarks = [Point::new(8.0, 8.0); LANDMARK_COUNT];
  let error = FaceAlign::to_template(crop, &landmarks).expect_err("no transform is determined");
  assert!(
    matches!(error, Error::DegenerateLandmarks(payload) if payload.spread() == 0.0),
    "expected DegenerateLandmarks, got {error:?}"
  );
}

#[test]
fn a_non_finite_landmark_is_rejected_by_index() {
  let data = vec![0u8; 16 * 16 * 3];
  let crop = FaceCrop::new(&data, 16, 16).expect("geometry is valid");
  let mut landmarks = FIXTURE_LANDMARKS;
  landmarks[3] = Point::new(f32::NAN, 4.0);
  let error = FaceAlign::to_template(crop, &landmarks).expect_err("NaN is not a landmark");
  assert!(
    matches!(error, Error::NonFiniteLandmark(payload) if payload.index() == 3),
    "expected NonFiniteLandmark(3), got {error:?}"
  );
}

#[test]
fn crop_geometry_is_validated() {
  let data = vec![0u8; 12];
  assert!(matches!(
    FaceCrop::new(&data, 0, 4).expect_err("a zero axis is unusable"),
    Error::CropDimensions(_)
  ));
  let error = FaceCrop::new(&data, 3, 3).expect_err("9 pixels need 27 bytes");
  assert!(
    matches!(error, Error::CropDataLength(payload) if payload.got() == 12 && payload.expected() == 27),
    "expected CropDataLength(12, 27), got {error:?}"
  );
  assert!(FaceCrop::new(&data, 2, 2).is_ok());
}

#[test]
fn from_template_pixels_requires_the_exact_length() {
  let exact = vec![0u8; TEMPLATE_BYTES];
  let short = vec![0u8; TEMPLATE_BYTES - 1];
  assert!(AlignedFace::from_template_pixels(&exact).is_ok());
  let error =
    AlignedFace::from_template_pixels(&short).expect_err("a short buffer is not a template");
  assert!(
    matches!(error, Error::CropDataLength(payload) if payload.expected() == TEMPLATE_BYTES),
    "expected CropDataLength, got {error:?}"
  );
  assert!(
    AlignedFace::from_template_pixels(&exact)
      .expect("valid")
      .transform()
      .is_none(),
    "pixels the caller aligned elsewhere must carry no transform"
  );
}

#[test]
fn aligning_records_the_transform_it_used() {
  let data = vec![7u8; 64 * 48 * 3];
  let crop = FaceCrop::new(&data, 64, 48).expect("geometry is valid");
  let aligned = FaceAlign::to_template(crop, &FIXTURE_LANDMARKS).expect("solvable");
  let transform = aligned
    .transform()
    .expect("alignment records its transform");
  let solved =
    SimilarityTransform::estimate(&FIXTURE_LANDMARKS, &ARCFACE_TEMPLATE).expect("solvable");
  assert_eq!(*transform, solved);
  assert_eq!(aligned.width(), TEMPLATE_SIZE);
  assert_eq!(aligned.height(), TEMPLATE_SIZE);
}
