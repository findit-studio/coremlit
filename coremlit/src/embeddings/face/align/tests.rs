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
  let warped = warp_bilinear(crop, &inverse).expect("an identity-scaled inverse stays inside int");

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
  // The first witness has a scale that is NONZERO and still has no inverse. It
  // is deliberately NOT `1e-160`, which this test used to use on the strength
  // of `1/(a² + b²)` overflowing there: `1/1e-160` is `1e160`, that transform
  // inverts, and building a truthful payload around an untrue predicate is
  // what the round before last actually did. The smallest subnormal is a real
  // witness — `1/5e-324` is genuinely not representable — and it is a scale
  // `f32` would flush to the zero this payload exists to stop reporting.
  const SUBNORMAL: f64 = f64::from_bits(1);
  let collapsed = SimilarityTransform::new(SUBNORMAL, 0.0, 1.0, 2.0);
  assert!(collapsed.inverse().is_none(), "the witness has no inverse");
  assert_eq!(collapsed.scale(), SUBNORMAL, "and its scale is not zero");
  assert_eq!(
    collapsed.scale() as f32,
    0.0,
    "and `f32` would render it as the zero this payload must not report"
  );

  let error = collapsed
    .checked_inverse()
    .expect_err("a transform with no inverse cannot align anything");
  assert!(
    matches!(&error, Error::NonInvertibleTransform(payload) if payload.scale() == SUBNORMAL),
    "the payload must carry the scale that collapsed, got {error:?}"
  );
  // And what a reader actually sees, since that is where the falsehood was.
  let message = error.to_string();
  assert!(
    message.contains("5e-324"),
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
  // The entry guard's OWN witness, and it did not exist before the scaled
  // reciprocal did. Under the old `D = a·a + b·b; 1./D` an infinite rotation
  // propagated a NaN into the result and the exit check caught it, so a guard
  // on the way in was unreachable. Scaling reaches a FINITE answer from it —
  // `∞` scales to `(0, −0)`, and a zero transform with a finite translation
  // passes the exit check — so without the guard this is a false `Some`
  // mapping every template pixel onto one source point.
  for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
    assert!(
      SimilarityTransform::new(infinite, 0.0, 1.0, 2.0)
        .inverse()
        .is_none(),
      "an infinite rotation coefficient has no inverse, and must not scale to a zero transform"
    );
    assert!(
      SimilarityTransform::new(0.0, infinite, 1.0, 2.0)
        .inverse()
        .is_none(),
      "the same on the other rotation coefficient, which takes the other scaling branch"
    );
  }
  // The check on the way OUT is load-bearing on its own, not a duplicate of
  // the one on the way in: a scale too small for `f64` to hold `1/s` turns
  // four finite parameters into an infinite one. `1e-160` is NOT that witness
  // — `1/1e-160 = 1e160` is finite, and treating it as one was the defect
  // `the_inverse_refuses_only_transforms_that_have_no_finite_inverse` covers.
  // The real boundary is around `5.6e-309`, below which no reciprocal exists.
  assert!(
    SimilarityTransform::new(f64::from_bits(1), 0.0, 1.0, 2.0)
      .inverse()
      .is_none(),
    "the smallest subnormal scale has no representable reciprocal"
  );
  // And the same check catching a TRANSLATION built from a good inverse
  // scale: `1e-300` inverts to `1e300`, which the shift then overflows.
  assert!(
    SimilarityTransform::new(1e-300, 0.0, 1e300, 0.0)
      .inverse()
      .is_none(),
    "an inverse translation that overflows is still no inverse"
  );
  assert!(
    SimilarityTransform::new(1.7875, -0.1252, 5.1247, 24.1454)
      .inverse()
      .is_some(),
    "a finite invertible transform must still invert"
  );
}

#[test]
fn cv_round_breaks_ties_to_even_and_refuses_what_leaves_int() {
  // `cvRound` is `lrint` under the default rounding mode, so an exact .5 goes
  // to the EVEN neighbour — not away from zero, which is what the pixel cast
  // below does. The two tie rules sit three lines apart in this module and
  // reproducing the wrong one at either site is invisible on any input that
  // never lands exactly on a half, which is most of them: the golden alone
  // could not tell the two apart.
  for (value, want) in [
    (0.5f64, 0i32),
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
      Some(want),
      "cvRound({value}) must be {want} (nearest, ties to even)"
    );
  }

  // Past `int` there is no reference answer at all: `saturate_cast<int>` is
  // UNDEFINED outside the range in C++. This used to saturate, and saturating
  // is precisely what let two out-of-domain terms cancel into a small,
  // valid-looking coordinate — see
  // `opposite_coordinate_saturations_must_not_cancel_into_a_valid_tap`. The
  // domain is reported instead.
  for out in [1e300, -1e300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
    assert_eq!(cv_round(out), None, "{out} is outside `int`");
  }

  // The boundary itself is inside. `i32::MIN` and `i32::MAX` are exactly
  // representable in `f64`, so this is the exact bound and not an
  // approximation of one.
  assert_eq!(cv_round(f64::from(i32::MAX)), Some(i32::MAX));
  assert_eq!(cv_round(f64::from(i32::MIN)), Some(i32::MIN));
  assert_eq!(cv_round(f64::from(i32::MAX) + 1.0), None);
  assert_eq!(cv_round(f64::from(i32::MIN) - 1.0), None);
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
fn the_tap_is_exact_rather_than_saturated_into_the_crop() {
  // The read this used to measure, on the geometry that separates the two
  // forms. The sampler once saturated its integer tap into `i16`, as OpenCV's
  // `saturate_cast<short>` does, and then decided the constant-0 border by
  // comparing the SATURATED tap against the crop's extent. Past `i16` the two
  // disagree: source column 33 000 is a perfectly good column of a 40 000-wide
  // crop, but the tap arrived as `i16::MAX`, which is ALSO inside that crop —
  // so the sampler read column 32 767 and reported nothing.
  //
  // The tap is written exactly now, so the caller's own column is read. The
  // crop bound below is still there, and is still OpenCV's, but it is no
  // longer what stands between an aliased tap and a wrong pixel.
  const WIDE: usize = 40_000;
  let mut row = vec![0u8; WIDE * 3];
  row[32_767 * 3] = 200; // what a saturated tap would have read
  row[33_000 * 3] = 25; // what the caller actually asked for
  let mut out = [0u8; 3];
  sample_fixed_point(
    &row,
    WIDE,
    1,
    33_000 << INTER_BITS, // an exact pixel centre: the whole weight on one tap
    0,
    &mut out,
  );
  assert_eq!(
    out[0], 25,
    "the exact tap must read the column asked for, not the one `i16` saturation aliases it onto"
  );

  // And the two forms agree on every crop the reference admits, which is what
  // makes this a total replacement rather than a divergence: at and past the
  // saturation value, both read the constant-0 border.
  let mut widest = vec![9u8; MAX_CROP_AXIS * 3];
  widest[(MAX_CROP_AXIS - 1) * 3] = 111;
  for column in [
    i64::from(i16::MAX) - 1,
    i64::from(i16::MAX),
    i64::from(i16::MAX) + 5_000,
    i64::from(i16::MIN),
    i64::from(i16::MIN) - 5_000,
  ] {
    let mut border = [7u8; 3];
    sample_fixed_point(
      &widest,
      MAX_CROP_AXIS,
      1,
      column << INTER_BITS,
      0,
      &mut border,
    );
    assert_eq!(
      border, [0u8; 3],
      "a tap at {column} is outside the widest admitted crop and must read the border"
    );
  }

  // The geometry past OpenCV's own assert is still refused at the door,
  // because the reference refuses it and this module's contract is to be
  // bit-exact with the reference.
  let data = vec![0u8; WIDE * 2 * 3];
  let error = FaceCrop::new(&data, WIDE, 2).expect_err("wider than the fixed-point tap domain");
  assert!(
    matches!(error, Error::CropDimensions(p) if p.width() == WIDE && p.height() == 2),
    "expected CropDimensions({WIDE}, 2), got {error:?}"
  );
  let tall = vec![0u8; 2 * WIDE * 3];
  assert!(
    matches!(
      FaceCrop::new(&tall, 2, WIDE).expect_err("the bound is per axis"),
      Error::CropDimensions(_)
    ),
    "the height axis is bounded too"
  );

  // The bound is exactly OpenCV's `remap` assert, `src.cols < SHRT_MAX`, and
  // it is a boundary rather than a round number: one pixel narrower is fine.
  assert_eq!(MAX_CROP_AXIS, i16::MAX as usize - 1);
  let admitted = vec![0u8; MAX_CROP_AXIS * 3];
  assert!(
    FaceCrop::new(&admitted, MAX_CROP_AXIS, 1).is_ok(),
    "the widest admitted crop must still be admitted"
  );
  let refused = vec![0u8; (MAX_CROP_AXIS + 1) * 3];
  assert!(
    FaceCrop::new(&refused, MAX_CROP_AXIS + 1, 1).is_err(),
    "one pixel past the bound must be refused"
  );

  // The widest admitted crop's own last column still reads as itself, so the
  // border assertions above are about the taps outside it and not about a
  // sampler that reads nothing.
  let mut last = [7u8; 3];
  sample_fixed_point(
    &widest,
    MAX_CROP_AXIS,
    1,
    (MAX_CROP_AXIS as i64 - 1) << INTER_BITS,
    0,
    &mut last,
  );
  assert_eq!(
    last[0], 111,
    "the last column of the widest admitted crop must still be readable"
  );
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

/// The reviewer's witness, `f32` as a detector emits them.
const WITNESS: [Point; LANDMARK_COUNT] = [
  Point::new(48.073_643, 97.059_7),
  Point::new(103.453_03, 115.633_26),
  Point::new(68.999_21, 127.547_72),
  Point::new(37.211_536, 152.986_66),
  Point::new(82.014_03, 169.196_21),
];

/// `skimage`'s `f32` `_umeyama` on [`WITNESS`] under numpy 2.5.1 / **OpenBLAS
/// 0.3.33**, row-major 2×3 — the same six numbers `tform.params[0:2, :]` holds.
///
/// Printed by `conversion/face/align_oracle.py --reference-divergence`.
const SKIMAGE_OPENBLAS: [f64; 6] = [
  0.628_153_825_248_663_7,
  0.202_666_161_104_246_54,
  -13.507_243_940_956_57,
  -0.202_666_161_104_246_54,
  0.628_153_825_248_663_7,
  2.451_227_246_254_319_4,
];

/// The IDENTICAL `_umeyama` source on the IDENTICAL landmarks, under **Apple
/// Accelerate** instead.
///
/// Note that `[0] != [4]` and `[1] != -[3]`: a `f32` `U @ V` is only
/// approximately orthogonal, so this is not a similarity at all and
/// [`SimilarityTransform`] could not hold it even if it were the target.
const SKIMAGE_ACCELERATE: [f64; 6] = [
  0.628_153_890_096_049_4,
  0.202_666_202_037_496_08,
  -13.507_253_770_385_248,
  -0.202_666_205_667_098_56,
  0.628_153_970_067_602_8,
  2.451_211_088_017_999,
];

/// `cv2.warpAffine`'s own inversion of a general 2×3, in ITS operation order.
///
/// [`SimilarityTransform::inverse`] is this specialised to a similarity, and
/// the two are asserted to agree wherever both apply. It is here in full
/// because the reference matrices above are not similarities and the
/// production type cannot carry them.
fn invert_2x3(m: [f64; 6]) -> [f64; 6] {
  let d = m[0] * m[4] - m[1] * m[3];
  let d = if d == 0.0 { 0.0 } else { 1.0 / d };
  let (n0, n1, n3, n4) = (m[4] * d, m[1] * -d, m[3] * -d, m[0] * d);
  [
    n0,
    n1,
    -n0 * m[2] - n1 * m[5],
    n3,
    n4,
    -n3 * m[2] - n4 * m[5],
  ]
}

/// Destination pixels whose five-bit source coordinate differs between two
/// source → template matrices.
///
/// Coordinates, not bytes: a moved coordinate leaves the output unchanged
/// wherever it lands in a flat neighbourhood, so counting differing pixels
/// under-reports a moved map.
fn coordinate_divergence(left: [f64; 6], right: [f64; 6]) -> usize {
  let (a, b) = (
    SourceGrid::new(invert_2x3(left)).expect("a face-shaped matrix stays inside int"),
    SourceGrid::new(invert_2x3(right)).expect("a face-shaped matrix stays inside int"),
  );
  (0..TEMPLATE_SIZE)
    .map(|v| {
      let (oa, ob) = (a.row_origin(v), b.row_origin(v));
      (0..TEMPLATE_SIZE)
        .filter(|&u| a.at(oa, u) != b.at(ob, u))
        .count()
    })
    .sum()
}

#[test]
fn the_solve_diverges_from_skimage_by_less_than_skimage_diverges_from_itself() {
  // The module doc's central claim, as three numbers rather than as prose.
  //
  // The ruling this answers was to reproduce `skimage`'s `f32` `_umeyama`
  // end to end so the bit-exact resampler would be fed the matrix the
  // reference computes. It cannot be done, and the third number is why: the
  // reference is not one matrix. `_umeyama`'s `f32` path is a `sgemm` and a
  // `sgesdd`, neither specified past returning *a* correct answer, and two
  // correct builds of it disagree on these very landmarks by MORE than this
  // module disagrees with either.
  //
  // If a later change closes the first gap by tracking one build, this goes
  // red and the gap that cannot be closed has to be confronted rather than
  // inherited.
  let solved = SimilarityTransform::estimate(&WITNESS, &ARCFACE_TEMPLATE)
    .expect("the witness landmarks are non-degenerate");

  // The watcher first: `invert_2x3` must agree with the production inverse
  // wherever the production type can carry the matrix at all, or the three
  // counts below are measuring this helper rather than the solve.
  let inverse = solved.inverse().expect("a solvable witness inverts");
  assert_eq!(
    invert_2x3(solved.matrix()),
    inverse.matrix(),
    "the test's general inversion must reproduce the production similarity one"
  );

  assert_eq!(
    coordinate_divergence(solved.matrix(), SKIMAGE_OPENBLAS),
    10,
    "the solve's distance from `skimage` under OpenBLAS"
  );
  assert_eq!(
    coordinate_divergence(solved.matrix(), SKIMAGE_ACCELERATE),
    5,
    "the solve's distance from the SAME `skimage` under Accelerate"
  );
  assert_eq!(
    coordinate_divergence(SKIMAGE_OPENBLAS, SKIMAGE_ACCELERATE),
    15,
    "two correct builds of the reference disagree with EACH OTHER by more than \
     this module disagrees with either; there is no single matrix to be exact against"
  );

  // The reference's own output is frequently not even a similarity, so the
  // production type could not carry it however the solve were written.
  // `SimilarityTransform` stores `(a, b)` and reconstitutes
  // `[a, −b, tx, b, a, ty]`, so the nearest value it can hold to Accelerate's
  // matrix is not that matrix: a `f32` `U @ V` is only approximately
  // orthogonal, and the shear that leaves behind has nowhere to live here.
  let nearest = SimilarityTransform::new(
    SKIMAGE_ACCELERATE[0],
    -SKIMAGE_ACCELERATE[1],
    SKIMAGE_ACCELERATE[2],
    SKIMAGE_ACCELERATE[5],
  );
  assert_ne!(
    nearest.matrix(),
    SKIMAGE_ACCELERATE,
    "the Accelerate reference was expected to carry a shear this type cannot hold"
  );
}

#[test]
fn the_divergence_counts_are_sensitive_to_the_matrix_they_measure() {
  // `the_solve_diverges_…` asserts three fixed numbers, so it is worth
  // something only if those numbers respond to the matrix. Three properties,
  // because "it changed once" is not one of them.
  let solved = SimilarityTransform::estimate(&WITNESS, &ARCFACE_TEMPLATE).expect("solvable");

  // Zero on identity — the metric is a distance, not a constant.
  for m in [solved.matrix(), SKIMAGE_OPENBLAS, SKIMAGE_ACCELERATE] {
    assert_eq!(
      coordinate_divergence(m, m),
      0,
      "a matrix cannot differ from itself"
    );
  }

  // Closing the gap really does close it. This is the shape a later "fix"
  // toward one build would take, and it is what makes the third assertion in
  // `the_solve_diverges_…` the one that has to be answered: adopting OpenBLAS's
  // matrix drives that count to zero and leaves the 15 exactly where it was.
  assert_eq!(
    coordinate_divergence(SKIMAGE_OPENBLAS, SKIMAGE_OPENBLAS),
    0,
    "tracking one build would zero its count"
  );
  assert_eq!(
    coordinate_divergence(SKIMAGE_OPENBLAS, SKIMAGE_ACCELERATE),
    15,
    "and would leave the other one untouched"
  );

  // And the count moves at the SCALE of the divergence it measures. That
  // scale is the translation's: the `f32` centroid `skimage` keeps is worth
  // 2.1e-6 on `tx` and 6.1e-6 on `ty` here, where the rotation block differs
  // by only 4.4e-8. A perturbation far under the measured divergence leaves a
  // quantised count alone, which is the metric behaving, not sleeping.
  let mut moved = solved.matrix();
  moved[2] += 1e-6;
  assert_ne!(
    coordinate_divergence(moved, SKIMAGE_OPENBLAS),
    coordinate_divergence(solved.matrix(), SKIMAGE_OPENBLAS),
    "a translation move the size of the measured divergence must be visible in the count"
  );
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
  let warped = warp_bilinear(crop, &SimilarityTransform::new(1.0, 0.0, FRACTION, 0.0))
    .expect("a unit-scale translation stays inside int");

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

/// The reviewer's round-3 witness: five FINITE, in-bounds landmarks in a
/// 256×256 crop whose solved inverse is large enough that both split halves of
/// the fixed-point coordinate leave `int` — in OPPOSITE directions.
const CANCELLING_LANDMARKS: [Point; LANDMARK_COUNT] = [
  Point::new(108.922_34, 130.0),
  Point::new(128.855_71, 130.0),
  Point::new(174.0, 130.0),
  Point::new(131.146_21, 130.0),
  Point::new(107.075_73, 130.0),
];

#[test]
fn opposite_coordinate_saturations_must_not_cancel_into_a_valid_tap() {
  const SIDE: usize = 256;
  let mut data = vec![0u8; SIDE * SIDE * 3];
  data[..3].copy_from_slice(&[200, 201, 202]); // crop pixel (0, 0)
  let crop = FaceCrop::new(&data, SIDE, SIDE).expect("geometry is valid");

  // Where destination (1, 0) TRULY comes from: nowhere near the crop.
  let transform = SimilarityTransform::estimate(&CANCELLING_LANDMARKS, &ARCFACE_TEMPLATE)
    .expect("finite, spread landmarks solve");
  let inverse = transform
    .inverse()
    .expect("a finite nonzero scale has a finite inverse");
  let m = inverse.matrix();
  let (source_x, source_y) = (m[0] + m[2], m[3] + m[5]);
  assert!(
    source_x < -1.8e9 && source_y > 7.6e7,
    "the witness must map destination (1,0) far outside the crop, got ({source_x:e}, {source_y:e})"
  );

  // Both halves of that coordinate leave `int`, in OPPOSITE directions, which
  // is what made the failure invisible: saturated, they summed to 15, which is
  // source pixel 0 after the shift onto the five-bit grid.
  let outcome = FaceAlign::to_template(crop, &CANCELLING_LANDMARKS);
  let sampled = outcome
    .as_ref()
    .ok()
    .map(|face| [face.pixels()[3], face.pixels()[4], face.pixels()[5]]);
  let error = match outcome {
    Err(error) => error,
    Ok(_) => panic!(
      "destination (1,0) inverse-maps to ({source_x:e}, {source_y:e}) — border — but the split \
       terms saturated to `i32::MIN` and `i32::MAX` and cancelled; it sampled {sampled:?}, which \
       is the crop's own pixel (0,0)"
    ),
  };
  assert!(
    matches!(&error, Error::CoordinateOverflow(_)),
    "expected CoordinateOverflow, got {error:?}"
  );
  // The refusal has to say WHERE, or it is a panic with a nicer type.
  let message = error.to_string();
  assert!(
    message.contains("outside the `int` domain"),
    "the message must name the domain that was left, got {message:?}"
  );

  // The terms themselves, so the gate does not rest on `to_template` alone:
  // the per-column half of `x` is past `i32::MAX`, and the per-row half of `y`
  // is past it too once `round_delta` is folded in — the addition that made
  // the round-2 `i16` bound insufficient one level up.
  let error = SourceGrid::new(m)
    .err()
    .expect("this map does not fit in `int`");
  assert!(
    matches!(&error, Error::CoordinateOverflow(p) if p.term() != CoordinateTerm::Sum),
    "a term leaves `int` before any sum does, got {error:?}"
  );

  // The `round_delta` FOLD is its own overflow site, and the witness above
  // reaches it only because its rounding already failed. So: a row origin that
  // rounds to exactly `i32::MAX` — inside `int` — and then leaves it by adding
  // 16. That is the `2147483663` the reviewer's witness reports, isolated.
  // `i32::MAX / 1024` is exact in `f64`, so this rounds to the boundary and not
  // near it.
  let at_the_boundary = f64::from(i32::MAX) / AB_SCALE;
  let error = SourceGrid::new([0.0, 0.0, at_the_boundary, 0.0, 0.0, 0.0])
    .err()
    .expect("`i32::MAX + round_delta` is not an `int`");
  assert!(
    matches!(
      &error,
      Error::CoordinateOverflow(p)
        if p.term() == CoordinateTerm::RowOrigin
          && p.axis() == CoordinateAxis::X
          && p.value() == f64::from(i32::MAX) + f64::from(ROUND_DELTA)
    ),
    "expected the fold past `i32::MAX` to be reported at 2147483663, got {error:?}"
  );
  // One less, and it fits — so this is the boundary and not a blanket refusal.
  assert!(
    SourceGrid::new([
      0.0,
      0.0,
      (f64::from(i32::MAX) - f64::from(ROUND_DELTA)) / AB_SCALE,
      0.0,
      0.0,
      0.0
    ])
    .is_ok(),
    "a row origin that folds to exactly `i32::MAX` is inside the domain"
  );

  // And the arm no per-term check can reach: two terms that each fit while
  // their SUM does not. The per-column term at u = 111 is about `+2.02e9` and
  // every row origin about `+1.07e9`, so both are inside `int` and
  // `origin + delta` is about `+3.09e9`, which is not.
  let column = f64::from(i32::MAX) * 0.94 / (111.0 * AB_SCALE);
  let row = f64::from(i32::MAX) * 0.5 / AB_SCALE;
  let error = SourceGrid::new([column, 0.0, row, 0.0, 0.0, 0.0])
    .err()
    .expect("two representable terms whose sum is not");
  assert!(
    matches!(
      &error,
      Error::CoordinateOverflow(p)
        if p.term() == CoordinateTerm::Sum
          && p.axis() == CoordinateAxis::X
          && p.value() > f64::from(i32::MAX)
    ),
    "expected a Sum overflow on the horizontal coordinate, got {error:?}"
  );
}

#[test]
fn the_inverse_refuses_only_transforms_that_have_no_finite_inverse() {
  // Forming `a² + b²` at the input's own magnitude used to decide
  // invertibility, and that product leaves `f64` on BOTH sides while the
  // inverse itself stays comfortably inside it.
  //
  // Underflow: `1e-160² = 1e-320` is subnormal, its reciprocal overflows, and
  // the whole inverse was refused — though `1/1e-160 = 1e160` is finite and
  // exactly representable.
  let tiny = SimilarityTransform::new(1e-160, 0.0, 1.0, 2.0);
  let tiny_inverse = tiny
    .inverse()
    .expect("1/1e-160 = 1e160 is finite, so this transform HAS an inverse");
  assert_eq!(
    tiny_inverse.a(),
    1e160,
    "the inverse coefficient is exactly 1e160"
  );

  // Overflow: `1e200² = inf`, so the determinant was infinite, its reciprocal
  // zero, and the result `Some` holding the ZERO transform — a false `Some`
  // that maps every template pixel onto one source point. The worse of the two
  // errors: a false `None` refuses, a false `Some` warps.
  let huge = SimilarityTransform::new(1e200, 0.0, 1.0, 2.0);
  let huge_inverse = huge
    .inverse()
    .expect("1/1e200 = 1e-200 is finite, so this transform HAS an inverse");
  assert_eq!(
    huge_inverse.a(),
    1e-200,
    "the inverse coefficient is exactly 1e-200, not the zero an infinite determinant produces"
  );

  // Both survive a round trip, which is the property a "zero transform"
  // inverse silently fails.
  for original in [tiny, huge] {
    let back = original
      .inverse()
      .and_then(|inverted| inverted.inverse())
      .expect("an invertible transform's inverse is invertible");
    assert_eq!(
      (back.a(), back.b()),
      (original.a(), original.b()),
      "inverting twice must return the rotation block it started from"
    );
  }

  // Across the band where `a² + b²` and its reciprocal are both normal —
  // about `1.5e-154` to `6.7e153`, the range the docs name — the inverse is
  // correct on BOTH sides of the boundary, which is the property that matters
  // rather than which association ran.
  for scale in [1e-160, 1e-154, 2e-154, 1.0, 6e153, 1e154, 1e200] {
    let inverted = SimilarityTransform::new(scale, 0.0, 0.0, 0.0)
      .inverse()
      .unwrap_or_else(|| panic!("a scale of {scale:e} has a finite inverse"));
    assert_eq!(
      inverted.a(),
      1.0 / scale,
      "the inverse coefficient at scale {scale:e} must be 1/{scale:e}"
    );
  }

  // The fast path is still OpenCV's own association wherever OpenCV's own
  // arithmetic is defined, so nothing about an ordinary alignment moved.
  let ordinary = SimilarityTransform::new(1.7875, -0.1252, 5.1247, 24.1454);
  let determinant = ordinary.a() * ordinary.a() + ordinary.b() * ordinary.b();
  let reciprocal = 1.0 / determinant;
  let inverted = ordinary.inverse().expect("an ordinary transform inverts");
  assert_eq!(
    (inverted.a(), inverted.b()),
    (ordinary.a() * reciprocal, ordinary.b() * -reciprocal),
    "an in-range transform must invert through `D = 1./D` exactly as cv2.warpAffine does"
  );
}
