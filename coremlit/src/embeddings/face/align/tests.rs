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
fn estimate_can_return_a_transform_with_no_inverse() {
  // `to_template`'s `inverse()` arm used to be commented as unreachable, on
  // the strength of `estimate` rejecting a zero SOURCE spread. That does not
  // follow: the solved scale is `|Σ conj(uᵢ)·vᵢ| / Σ‖uᵢ‖²` over the two
  // CENTRED sets, so it is the TARGET side (and the relative geometry) that
  // decides invertibility, and `estimate` is public and takes its target from
  // the caller. A zero-spread target is the shortest witness: every solved
  // parameter is finite, `estimate` is happy, and the result still inverts to
  // nothing.
  let flat_target = [Point::new(11.0, -4.0); LANDMARK_COUNT];
  let solved = SimilarityTransform::estimate(&FIXTURE_LANDMARKS, &flat_target)
    .expect("a spread source against any finite target is solvable");
  assert_eq!((solved.a(), solved.b()), (0.0, 0.0));
  assert_eq!(
    (solved.tx(), solved.ty()),
    (11.0, -4.0),
    "the whole plane collapses onto the target point"
  );
  assert!(
    solved.inverse().is_none(),
    "a zero-scale transform has no inverse, so `estimate` can hand back one that does not invert"
  );
}

#[test]
fn a_transform_that_does_not_invert_reports_its_own_scale_not_a_landmark_spread() {
  // Every execution of `to_template`'s no-inverse arm has already passed
  // `estimate`'s spread guard, so `Σ‖pᵢ−p̄‖²` is strictly positive there. The
  // old payload reported it as ZERO, which sends the reader hunting for
  // coincident landmarks that do not exist — the failure is the solved SCALE,
  // and that is what the payload has to carry.
  //
  // The first witness has a scale that is NONZERO and still has no inverse:
  // `1/(a² + b²)` overflows at 1e-160. A payload reporting zero here would be
  // a sentinel rather than the measurement, so this is the assertion that
  // discriminates one from the other.
  let collapsed = SimilarityTransform::new(1e-160, 0.0, 1.0, 2.0);
  assert!(collapsed.inverse().is_none(), "the witness has no inverse");
  assert_eq!(collapsed.scale(), 1e-160, "and its scale is not zero");

  let error = collapsed
    .checked_inverse()
    .expect_err("a transform with no inverse cannot align anything");
  assert!(
    matches!(&error, Error::NonInvertibleTransform(payload) if payload.scale() == 1e-160),
    "the payload must carry the scale that collapsed, got {error:?}"
  );
  // And what a reader actually sees, since that is where the falsehood was.
  let message = error.to_string();
  assert!(
    message.contains("1e-160"),
    "the message must name the collapsed scale, got {message:?}"
  );
  assert!(
    !message.contains("landmark"),
    "the landmarks are spread; blaming them is the falsehood, got {message:?}"
  );

  // The second witness is the one `estimate` itself reaches (see
  // `estimate_can_return_a_transform_with_no_inverse`): a zero-spread TARGET
  // collapses the plane onto a point. The source spread is ~9.3e4 — the
  // number the old payload reported as 0.0 — and the scale really is zero.
  let flat_target = [Point::new(11.0, -4.0); LANDMARK_COUNT];
  let solved = SimilarityTransform::estimate(&FIXTURE_LANDMARKS, &flat_target)
    .expect("a spread source against any finite target is solvable");
  let error = solved
    .checked_inverse()
    .expect_err("a zero-scale transform has no inverse");
  assert!(
    matches!(&error, Error::NonInvertibleTransform(payload) if payload.scale() == 0.0),
    "expected NonInvertibleTransform(0), got {error:?}"
  );
  assert!(
    !error.to_string().contains("landmark"),
    "a well-spread source must not be reported as landmarks with no spread"
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
    matches!(error, Error::NonFiniteLandmark(payload)
      if payload.index() == 3 && payload.set() == LandmarkSet::Source),
    "expected NonFiniteLandmark(source, 3), got {error:?}"
  );
}

#[test]
fn estimate_rejects_a_non_finite_target_and_names_the_set_it_came_from() {
  // `estimate` takes TWO point sets and used to validate only `source`. A NaN
  // in the public `target` reached the centroid and both dot products, and the
  // function returned `Ok` holding NaN parameters — after which `apply` gives
  // NaN and the sampler, refusing every mapped coordinate, emits an all-border
  // template. A silent black face is exactly what returning a `Result` here
  // was supposed to prevent.
  let mut target = ARCFACE_TEMPLATE;
  target[2] = Point::new(56.0252, f32::INFINITY);
  let error = SimilarityTransform::estimate(&FIXTURE_LANDMARKS, &target)
    .expect_err("an infinite target coordinate determines no transform");
  assert!(
    matches!(error, Error::NonFiniteLandmark(payload)
      if payload.index() == 2 && payload.set() == LandmarkSet::Target),
    "expected NonFiniteLandmark(target, 2), got {error:?}"
  );

  // And the payload has to DISTINGUISH the two sides, or the caller cannot
  // tell "my detector emitted NaN" from "the template I passed is broken".
  let mut source = FIXTURE_LANDMARKS;
  source[2] = Point::new(30.5, f32::NAN);
  let from_source = SimilarityTransform::estimate(&source, &ARCFACE_TEMPLATE)
    .expect_err("a NaN source coordinate determines no transform");
  assert!(
    matches!(from_source, Error::NonFiniteLandmark(payload)
      if payload.index() == 2 && payload.set() == LandmarkSet::Source),
    "expected NonFiniteLandmark(source, 2), got {from_source:?}"
  );
  assert_ne!(
    LandmarkSet::Source,
    LandmarkSet::Target,
    "the two sides must not compare equal, or naming them proves nothing"
  );
}

#[test]
fn a_solved_transform_with_a_non_finite_parameter_is_never_handed_out() {
  // The backstop `estimate` returns through. It is not reachable from
  // `estimate` itself — with both `f32` point sets finite the solved
  // parameters are bounded well inside `f64` (see `estimate`'s doc) — so it is
  // gated at the constructor, which is the only place that can see it. Without
  // this, `Ok` could carry a transform whose `apply` is NaN.
  for (index, parameter) in [
    TransformParameter::A,
    TransformParameter::B,
    TransformParameter::Tx,
    TransformParameter::Ty,
  ]
  .into_iter()
  .enumerate()
  {
    let mut params = [1.0f64, 0.5, 2.0, 3.0];
    params[index] = f64::NAN;
    let [a, b, tx, ty] = params;
    let error =
      SimilarityTransform::checked(a, b, tx, ty).expect_err("a NaN parameter is not a transform");
    assert!(
      matches!(error, Error::NonFiniteTransform(payload) if payload.parameter() == parameter),
      "expected NonFiniteTransform({parameter}), got {error:?}"
    );
  }
  assert!(SimilarityTransform::checked(1.0, 0.5, 2.0, 3.0).is_ok());
}

#[test]
fn an_inverse_is_refused_when_any_parameter_is_non_finite() {
  // The same one-sided-validation class as `estimate`'s, on the other public
  // constructor: `inverse` checked the ROTATION (through the determinant) and
  // never the TRANSLATION, so a perfectly good rotation with a NaN shift
  // returned `Some` holding a transform whose `apply` is NaN everywhere.
  // `new` is public AND `const`, so that is a value a caller can build.
  assert!(
    SimilarityTransform::new(1.0, 0.0, f64::NAN, 0.0)
      .inverse()
      .is_none(),
    "a NaN translation must not invert to Some"
  );
  assert!(
    SimilarityTransform::new(1.0, 0.0, 0.0, f64::NEG_INFINITY)
      .inverse()
      .is_none(),
    "an infinite translation must not invert to Some"
  );
  assert!(
    SimilarityTransform::new(f64::NAN, 0.0, 1.0, 2.0)
      .inverse()
      .is_none(),
    "the rotation side must still be refused"
  );
  // And the check on the way OUT is load-bearing on its own, not a duplicate
  // of the one on the way in: a scale small enough that `1/(a² + b²)`
  // overflows turns four finite parameters into an infinite one. OpenCV's
  // `warpAffine` computes the same `D = 1./D` and would produce the same
  // infinity, so refusing is both the safe answer and the faithful one.
  assert!(
    SimilarityTransform::new(1e-160, 0.0, 1.0, 2.0)
      .inverse()
      .is_none(),
    "a scale whose reciprocal overflows has no finite inverse"
  );
  assert!(
    SimilarityTransform::new(1.7875, -0.1252, 5.1247, 24.1454)
      .inverse()
      .is_some(),
    "a finite invertible transform must still invert"
  );
}

#[test]
fn cv_round_breaks_ties_to_even_and_saturates() {
  // `cvRound` is `lrint` under the default rounding mode, so an exact .5 goes
  // to the EVEN neighbour — not away from zero, which is what the pixel cast
  // below does. The two tie rules sit three lines apart in this module and
  // reproducing the wrong one at either site is invisible on any input that
  // never lands exactly on a half, which is most of them: the golden alone
  // could not tell the two apart.
  for (value, want) in [
    (0.5f64, 0i64),
    (1.5, 2),
    (2.5, 2),
    (3.5, 4),
    (-0.5, 0),
    (-1.5, -2),
    (-2.5, -2),
    (0.49, 0),
    (0.51, 1),
    (-0.51, -1),
  ] {
    assert_eq!(
      cv_round(value),
      want,
      "cvRound({value}) must be {want} (nearest, ties to even)"
    );
  }
  // `saturate_cast<int>` is undefined outside `int` in C++; here it saturates,
  // and either way the coordinate lands far outside every crop.
  assert_eq!(cv_round(1e300), i64::from(i32::MAX));
  assert_eq!(cv_round(-1e300), i64::from(i32::MIN));
}

#[test]
fn the_fixed_point_pixel_cast_rounds_half_up_and_saturates() {
  // OpenCV's `FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>`:
  // `saturate_cast<uchar>((value + (1 << 14)) >> 15)`. Half goes UP here,
  // where `cv_round` above sends it to even — pinned separately because the
  // difference only shows on an exact half.
  let one = 1i64 << REMAP_COEF_BITS;
  let half = 1i64 << (REMAP_COEF_BITS - 1);
  assert_eq!(fixed_point_to_u8(0), 0);
  assert_eq!(fixed_point_to_u8(half - 1), 0);
  assert_eq!(fixed_point_to_u8(half), 1, "an exact half must round UP");
  assert_eq!(fixed_point_to_u8(one + half), 2, "and so must the next one");
  assert_eq!(fixed_point_to_u8(255 * one), 255);
  assert_eq!(fixed_point_to_u8(256 * one), 255, "must saturate, not wrap");
  assert_eq!(fixed_point_to_u8(-one), 0, "must clamp, not wrap");
}

#[test]
fn the_saturating_weight_table_cell_is_invisible_for_u8_sources() {
  // `bilinear_weights` returns the EXACT 15-bit table, whose four entries sum
  // to `1 << 15`. OpenCV's `initInterTab2D` cannot: it fills its table through
  // `saturate_cast<short>`, so the single cell whose weight is the whole unit
  // — fraction (0, 0) — saturates to 32 767, and the sum-fixing step then puts
  // the missing 1 on the opposite corner. That one cell of 1 024 is the only
  // place the two tables differ, and this module would be claiming
  // bit-exactness while knowingly using the other one if the difference were
  // not proven invisible.
  //
  // Exhaustive over both taps the differing weights touch: with weights
  // [32768, 0, 0, 0] the accumulator is `v00 << 15`, and with OpenCV's
  // [32767, 0, 0, 1] it is `v00 · 32767 + v11`.
  for v00 in 0..=255i64 {
    for v11 in 0..=255i64 {
      assert_eq!(
        fixed_point_to_u8(v00 * 32768),
        fixed_point_to_u8(v00 * 32767 + v11),
        "taps ({v00}, {v11}) separate the exact weight table from OpenCV's saturated one"
      );
    }
  }

  // The property that makes the exact table the right one to carry: every cell
  // sums to one unit, so the fixed-point cast is unbiased.
  for fy in 0..INTER_TAB_SIZE {
    for fx in 0..INTER_TAB_SIZE {
      let weights = bilinear_weights(fx, fy);
      assert!(
        weights.iter().all(|w| *w >= 0),
        "weight table cell ({fx}, {fy}) has a negative entry: {weights:?}"
      );
      assert_eq!(
        weights.iter().sum::<i64>(),
        1 << REMAP_COEF_BITS,
        "weight table cell ({fx}, {fy}) does not sum to one unit: {weights:?}"
      );
    }
  }
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

#[test]
fn a_fraction_below_the_five_bit_half_step_takes_the_pure_left_pixel() {
  // The resampler falsifier, stated as OpenCV's own arithmetic rather than as
  // prose. `cv2.warpAffine`'s `INTER_LINEAR` carries the fractional source
  // coordinate in five bits (`INTER_BITS = 5`), so a true fraction below the
  // half-step 1/64 quantises to index 0 and the tap is the PURE LEFT PIXEL.
  //
  // On a 0-to-255 edge that is the difference between 0 and `255 · f`:
  //   f = 0.015  (just under 1/64 = 0.015625)
  //   unquantised: 255 · 0.015 = 3.825  ->  4
  //   OpenCV:      round(0.015 · 1024) = 15; (15 + 16) >> 5 = 0  ->  fraction
  //                index 0  ->  255 · 0 = 0
  //
  // A float sampler cannot produce 0 here, and every published ArcFace number
  // is measured against crops `cv2.warpAffine` produced.
  const FRACTION: f64 = 0.015;
  let (width, height) = (4usize, 2usize);
  let mut data = vec![255u8; width * height * 3];
  for y in 0..height {
    for channel in 0..3 {
      data[(y * width) * 3 + channel] = 0;
    }
  }
  let crop = FaceCrop::new(&data, width, height).expect("geometry is valid");
  let warped = warp_bilinear(crop, &SimilarityTransform::new(1.0, 0.0, FRACTION, 0.0));

  assert_eq!(
    warped[0], 0,
    "template pixel (0, 0) sampled a 0-to-255 edge at fraction {FRACTION} (under the 1/64 \
     half-step) and got {}, where cv2.warpAffine's 5-bit INTER_LINEAR quantises the fraction to 0 \
     and takes the pure left pixel",
    warped[0]
  );

  // The same quantisation at the right edge: u = 3 maps to source x = 3.015,
  // whose fraction also quantises to 0, so the pixel is the pure 255 and NOT
  // 0.985 · 255 = 251 blended against the zero border past the last column.
  let u3 = 3 * 3;
  assert_eq!(
    warped[u3], 255,
    "template pixel (3, 0) got {}, where OpenCV's quantised fraction 0 takes source column 3 \
     whole and never reaches the border",
    warped[u3]
  );
}
