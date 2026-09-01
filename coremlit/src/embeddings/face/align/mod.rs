//! The 5-point similarity alignment to the ArcFace 112×112 template.
//!
//! **Alignment lives OUTSIDE the embedder on purpose.** The embedder is then a
//! pure function of an [`AlignedFace`], and the template — the thing every
//! downstream cosine is measured through — is an explicit value with a golden
//! of its own rather than a private preprocessing step nobody can test in
//! isolation. A wrong transform does not fail; it degrades silently, moving
//! every embedding by an amount no shape check can see.
//!
//! # What is being reproduced
//!
//! deepinsight/insightface, `python-package/insightface/utils/face_align.py`,
//! pinned at commit `ffa12d315041c0505b077c7ff057ca914bb8dc7e` (2022-12-17):
//!
//! ```text
//! arcface_dst = np.array(
//!     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
//!      [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
//!
//! tform = trans.SimilarityTransform()
//! tform.estimate(lmk, dst)
//! M = tform.params[0:2, :]
//! warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
//! ```
//!
//! `estimate` is `skimage`'s Umeyama least-squares similarity, and
//! `cv2.warpAffine` without `WARP_INVERSE_MAP` **inverts** `M` itself and
//! samples the source at the inverse-mapped destination pixel centre,
//! `INTER_LINEAR`, constant-0 border. [`FaceAlign::to_template`] is that
//! pipeline in the same order.
//!
//! **The two halves are reproduced to different standards, and the difference
//! is measured rather than assumed.** The warp is bit-exact with OpenCV 4.x
//! given a matrix. The solve is not bit-exact with `skimage`, and the section
//! below measures both how far apart they are and — the part that decides
//! what may be claimed — how far the reference is from ITSELF.
//!
//! # The resampler is BIT-EXACT with `cv2.warpAffine`, and that is the contract
//!
//! `INTER_LINEAR` is **not** a float bilinear kernel. For an 8-bit image
//! OpenCV quantises the inverse-mapped coordinate to a five-bit fraction
//! (`INTER_BITS = 5`) and interpolates with 15-bit fixed-point weights, so a
//! true fraction below the half-step `1/64` collapses to **zero** and the tap
//! is the pure left pixel. On a 0-to-255 edge that is the difference between
//! `0` and `255/64 ≈ 4` — two thirds of a level short of `4` is not a rounding
//! difference, it is a different pixel.
//!
//! An earlier revision of this module resampled in `f64` and recorded the
//! divergence as "at most one LSB per channel". **That was a measurement on
//! one fixture stated as a bound over the domain, and it is false.** Measured
//! against `cv2.warpAffine` over ArcFace-shaped warps of random crops
//! (`opencv-python-headless` 4.12.0, 451 584 bytes): **11.6 % of bytes differ
//! and the worst differs by 6 levels.** Every published ArcFace accuracy
//! number is measured against crops `cv2.warpAffine` produced, so this module
//! reproduces OpenCV's fixed-point pipeline exactly rather than approximating
//! it. The sampler's constants are each named after the OpenCV symbol they
//! come from (`INTER_BITS`, `AB_SCALE`, `INTER_REMAP_COEF_BITS`, …), so the
//! pipeline can be read against `imgproc/src/imgwarp.cpp` line by line.
//!
//! **Which OpenCV — the version is part of the contract.** The 4.x line, which
//! is what InsightFace's pinned `face_align.py` runs against and what every
//! published number was measured on. OpenCV **5.0 replaced the fixed-point
//! path with a float one**: on the same warps it tracks an unquantised `f64`
//! sampler (one differing byte in 73 728, an exact-tie rounding) and so
//! differs from 4.x on the same 11.6 % of bytes. "Bit-exact with OpenCV" is
//! therefore version-bearing, and it is pinned here to **4.x** deliberately.
//!
//! # The SOLVE is not bit-exact with `skimage`, and there is no single
//! # `skimage` to be exact against
//!
//! The resampler above reproduces `cv2.warpAffine` exactly **given a matrix**.
//! Which matrix is a separate question, and the honest answer is that the
//! reference has no single one to reproduce.
//!
//! `skimage`'s `_umeyama` (`skimage/transform/_geometric.py` v0.19.3, L107-149)
//! keeps its **`f32`** input through the centroids, the covariance and the SVD,
//! storing only the result as `f64`. This module promotes to `f64` first. Same
//! minimiser, different numbers — and the difference is large enough to move a
//! five-bit source coordinate. On the landmarks
//!
//! ```text
//! [[48.073643, 97.0597], [103.45303, 115.63326], [68.99921, 127.54772],
//!  [37.211536, 152.98666], [82.01403, 169.19621]]
//! ```
//!
//! **10 of the 12 544 destination pixels** take a different five-bit source
//! coordinate than `skimage` gives under numpy 2.5.1 / OpenBLAS 0.3.33.
//!
//! That much is a real divergence. What makes it unclosable is the next
//! measurement. `_umeyama`'s `f32` path is two library calls — a `sgemm` for
//! the covariance and a `sgesdd` for the 2×2 SVD — and neither is specified
//! beyond returning *a* correct answer. Running the identical `_umeyama`
//! source, same machine, same landmarks, under two BLAS/LAPACK builds (numpy's
//! OpenBLAS 0.3.33 and Apple's Accelerate):
//!
//! - the `f32` covariance differs on **16 618 of 20 000** face-like landmark
//!   sets. OpenBLAS's aarch64 `sgemm` contracts its multiply-adds into `fma`:
//!   an `fma` chain reproduces it on 3 000 of 3 000 random inputs and a
//!   non-fused chain on 341. Whether a kernel contracts is a build flag, not a
//!   specification;
//! - the `f32` `sgesdd` differs on its singular values on 13 657 of 20 000,
//!   and on the rotation `U @ V` that `_umeyama` actually uses on all 20 000;
//! - end to end, **on the witness above the two builds differ from each other
//!   on 15 destination pixels** — more than either differs from this module —
//!   and over 20 000 face-like sets on a mean of 14.8 (median 11, worst 212).
//!
//! So "bit-exact with `skimage`" is not a property of `skimage`. It is a
//! property of `skimage` *and the BLAS the measuring machine happened to
//! link*, and picking one build to be exact against would be picking one of
//! several equally correct references while presenting it as having removed a
//! choice.
//!
//! A structural obstruction sits on top of the numeric one:
//! [`SimilarityTransform`] cannot hold a shear by construction, and under
//! Accelerate **19 624 of those same 20 000** `_umeyama` results are not
//! exactly similarities — `a ≠ d` or `b ≠ −c` in the last bits, because a
//! `f32` `U @ V` is only approximately orthogonal. (Under OpenBLAS, 0 of
//! 20 000, which is itself the point: the property is the build's, not the
//! reference's.) The reference's own output is routinely not a value this type
//! can represent.
//!
//! **What this module claims, therefore, and nothing more:** the transform is
//! the least-squares similarity minimiser of the `f32` landmarks, evaluated in
//! `f64` (not exactly — `f64` rounds too; it is `f64`-accurate where the
//! reference is `f32`-accurate), and the resampler is bit-exact with
//! `cv2.warpAffine`
//! 4.x given that transform. It sits 10 five-bit coordinates from one
//! reference build and 5 from another, inside a 15-wide band the reference
//! occupies on its own.
//! `the_solve_diverges_from_skimage_by_less_than_skimage_diverges_from_itself`
//! pins all three numbers, so an attempt to close the gap toward one build has
//! to confront the gap that cannot be closed.
//!
//! Regenerate every number above with
//! `python3 conversion/face/align_oracle.py --reference-divergence --sweep 20000`
//! (about ten seconds; without `--sweep` it prints the matrices and the three
//! witness counts alone). Deciding
//! it on accuracy instead of on bit-exactness needs a number this branch
//! cannot produce — the embedding drift the divergence causes, measured
//! against a staged artifact, and there is none (see the
//! [`crate::embeddings::face`] module doc). So it is recorded rather than
//! traded away.
//!
//! # The transform is solved without an SVD
//!
//! Umeyama's construction is stated with an SVD, but in 2-D **without
//! reflection** — which is what an alignment is; a mirrored face is not the
//! same face — the minimiser has a closed form. Writing the scaled rotation as
//! `[[a, −b], [b, a]]` makes the residual linear in `(a, b, tx, ty)`, so the
//! least-squares solution is a pair of dot products over the centred point
//! sets. See [`SimilarityTransform::estimate`].
//!
//! The evidence that this is the right minimiser deliberately does **not** rest
//! on agreeing with a second copy of the same derivation. Three independent
//! legs, in `tests.rs` and `tests/face/align_golden.rs`:
//!
//! - `recovered_transform_is_the_least_squares_minimiser` perturbs the solved
//!   parameters in all four directions and asserts the residual **rises** —
//!   an optimality proof that names no formula at all;
//! - `exact_similarity_landmarks_recover_the_analytic_inverse` feeds landmarks
//!   that are an exact similarity image of the template and asserts the
//!   recovered scale and rotation are the constructed ones inverted;
//! - the committed golden compares 112×112×3 bytes against
//!   `conversion/face/align_oracle.py`, which solves the same minimiser
//!   through a different derivation.
//!
//! All three legs are about the minimiser, not about `skimage`'s `f32`
//! evaluation of it; the section above is what covers that.
//!
//! The golden's THIRD leg covers the solve only. Since the resampler became
//! bit-exact, the oracle reproduces the same OpenCV specification this module
//! does, so their byte agreement catches a transcription slip on either side
//! and is not independent evidence about the pipeline. What carries that is
//! `a_fraction_below_the_five_bit_half_step_takes_the_pure_left_pixel`, which
//! pins the one behaviour separating the fixed-point pipeline from a float
//! one, plus `cv_round_breaks_ties_to_even_and_refuses_what_leaves_int` and
//! `the_fixed_point_pixel_cast_rounds_half_up_and_saturates` for the two tie
//! rules that no whole-image comparison can see.
//!
//! # The coordinate pipeline is TOTAL, and where it is not it says so
//!
//! Everything between the solved transform and a sampled byte rounds, casts,
//! clamps or accumulates, and each of those is a place an answer can be
//! invented. Two rounds of review found one invented answer each — an `i16`
//! tap that aliased a real column onto a saturated one, then two `i32` terms
//! that saturated in opposite directions and CANCELLED into a small,
//! plausible coordinate — and both were first met by bounding the input.
//! Bounding the input is an argument about every future caller; it does not
//! make the operation total, it makes it safe for the inputs someone thought
//! of. So the operation is fallible instead:
//!
//! - `cv_round` returns `Option<i32>` rather than saturating. OpenCV's
//!   `saturate_cast<int>(double)` is UNDEFINED past `int`, so there is no
//!   reference answer to reproduce there — only a domain to stay inside;
//! - the `round_delta` fold and the per-row/per-column SUM are checked
//!   additions, the sum for all 112² pairs at once by an extremes argument
//!   (`check_sum_domain`);
//! - the whole map is built and validated in `SourceGrid::new` BEFORE the
//!   first pixel is sampled, so a transform outside the domain produces
//!   [`Error::CoordinateOverflow`] rather than a partially warped face;
//! - the source tap is written EXACTLY rather than saturated into `i16`,
//!   which agrees with the reference on every crop the reference admits and
//!   is total on the ones it does not;
//! - [`SimilarityTransform::inverse`] refuses only when the INVERSE is
//!   unrepresentable, not when one expression for it overflowed.
//!
//! What remains a clamp is `fixed_point_to_u8`'s saturation into `u8`, which
//! is OpenCV's `FixedPtCast` and is unreachable for a `u8` source (the four
//! 15-bit weights sum to exactly `1 << 15`, so the accumulator cannot leave
//! `0..=255` after the shift); and [`MAX_CROP_AXIS`], which is now purely the
//! reference's own `CV_Assert` and no longer stands between an aliased tap
//! and a wrong pixel.

use crate::embeddings::face::error::{
  CoordinateAxis, CoordinateOverflow, CoordinateTerm, CropDataLength, CropDimensions,
  DegenerateLandmarks, Error, LandmarkSet, NonFiniteLandmark, NonFiniteTransform,
  NonInvertibleTransform, Result, TransformParameter,
};

/// The number of landmarks the ArcFace family aligns on.
pub const LANDMARK_COUNT: usize = 5;

/// The ArcFace template's side, in pixels.
pub const TEMPLATE_SIZE: usize = 112;

/// Bytes in one [`AlignedFace`]: `112 · 112 · 3`, RGB8 interleaved.
pub const TEMPLATE_BYTES: usize = TEMPLATE_SIZE * TEMPLATE_SIZE * 3;

/// The largest crop axis [`FaceCrop::new`] admits: one short of `i16::MAX`.
///
/// **This is the REFERENCE's admitted geometry, and nothing here depends on it
/// for safety.** OpenCV 4.x's `remap` — the fixed-point pipeline `warpAffine`
/// funnels into — opens with `CV_Assert( dst.cols < SHRT_MAX && dst.rows <
/// SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX )`, so a wider crop
/// is a shape the reference declines to define. This module's contract is to be
/// bit-exact with `cv2.warpAffine` (see the module doc); admitting geometry the
/// reference refuses would mean claiming exactness against an answer that does
/// not exist.
///
/// **It used to be load-bearing, and that is worth recording rather than
/// quietly dropping.** The sampler once saturated each integer source tap into
/// `i16` — OpenCV's own `saturate_cast<short>` on the `short XY[]` its
/// `WarpAffineInvoker` fills — and decided the constant-0 border by comparing
/// the SATURATED tap against the crop's extent. That comparison is right only
/// while the saturation value is not a coordinate the crop actually has, which
/// this bound was introduced to guarantee. Guaranteeing it is not the same as
/// removing it: the tap is now written exactly (`sample_fixed_point`), so a
/// coordinate outside the crop reads the border at any crop width, and the
/// bound no longer stands between an aliased tap and a wrong pixel.
/// `the_tap_is_exact_rather_than_saturated_into_the_crop` measures both forms
/// on the geometry that separates them.
///
/// **Why `i16::MAX − 1` and not `i16::MAX`.** OpenCV's bound is strictly less
/// than `SHRT_MAX`, so this takes the same one: the admitted set is exactly the
/// reference's and the two cannot disagree about a crop at the boundary. One
/// pixel of conservatism, chosen to keep a second number from existing.
pub const MAX_CROP_AXIS: usize = i16::MAX as usize - 1;

/// One 2-D point in a crop's pixel coordinates, pixel centres on integers.
///
/// `f32` because that is what a detector emits. The solve then promotes to
/// `f64`, where `skimage`'s stays in `f32` — a divergence the module doc
/// measures rather than waves at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
  /// Horizontal coordinate, increasing rightwards.
  x: f32,
  /// Vertical coordinate, increasing downwards.
  y: f32,
}

impl Point {
  /// A point at `(x, y)`.
  #[inline(always)]
  pub const fn new(x: f32, y: f32) -> Self {
    Self { x, y }
  }

  /// Horizontal coordinate, increasing rightwards.
  #[inline(always)]
  pub const fn x(&self) -> f32 {
    self.x
  }

  /// Vertical coordinate, increasing downwards.
  #[inline(always)]
  pub const fn y(&self) -> f32 {
    self.y
  }
}

/// The ArcFace 112×112 destination template, in the landmark order the whole
/// family uses: **left eye, right eye, nose tip, left mouth corner, right mouth
/// corner**.
///
/// Left and right are the VIEWER's, matching the upstream array — the first
/// entry has the smaller `x`. Passing the subject's own left/right instead
/// mirrors every face and is invisible to every check but the cosine.
///
/// Verbatim from the pinned `face_align.py` (see the module doc), `f32` because
/// upstream declares `dtype=np.float32`.
pub const ARCFACE_TEMPLATE: [Point; LANDMARK_COUNT] = [
  Point::new(38.2946, 51.6963),
  Point::new(73.5318, 51.5014),
  Point::new(56.0252, 71.7366),
  Point::new(41.5493, 92.3655),
  Point::new(70.7299, 92.2041),
];

/// A 2-D similarity transform `p ↦ [[a, −b], [b, a]] · p + t`, in the
/// source → template direction.
///
/// The rotation-and-uniform-scale block is stored as the two free parameters
/// `(a, b)` rather than a general 2×2, so a value of this type **cannot**
/// represent a shear, a non-uniform scale, or a reflection. That is the point:
/// the alignment contract is a similarity, and making the type unable to hold
/// anything else removes a whole class of silent corruption.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform {
  /// `s·cos θ` — the `[0][0]` and `[1][1]` entry.
  a: f64,
  /// `s·sin θ` — the `[1][0]` entry, and `−b` at `[0][1]`.
  b: f64,
  /// Horizontal translation.
  tx: f64,
  /// Vertical translation.
  ty: f64,
}

impl SimilarityTransform {
  /// A transform from its four free parameters.
  ///
  /// `a = s·cos θ`, `b = s·sin θ`, and `(tx, ty)` the translation.
  ///
  /// **Unvalidated**, because it is `const`: a non-finite argument produces a
  /// transform whose [`Self::apply`] is NaN everywhere. The two FALLIBLE
  /// constructors — [`Self::estimate`] and [`Self::inverse`] — both refuse
  /// one, so a transform that reaches [`FaceAlign::to_template`]'s sampler is
  /// finite in all four parameters.
  #[inline(always)]
  pub const fn new(a: f64, b: f64, tx: f64, ty: f64) -> Self {
    Self { a, b, tx, ty }
  }

  /// The first of `(a, b, tx, ty)` that is NaN or infinite, in that order.
  #[inline]
  fn first_non_finite(&self) -> Option<TransformParameter> {
    [
      (TransformParameter::A, self.a),
      (TransformParameter::B, self.b),
      (TransformParameter::Tx, self.tx),
      (TransformParameter::Ty, self.ty),
    ]
    .into_iter()
    .find_map(|(parameter, value)| (!value.is_finite()).then_some(parameter))
  }

  /// [`Self::new`] with the finiteness check — the one path every fallible
  /// constructor returns through, so no `Ok` can hold a transform whose
  /// [`Self::apply`] is NaN.
  fn checked(a: f64, b: f64, tx: f64, ty: f64) -> Result<Self> {
    let candidate = Self { a, b, tx, ty };
    match candidate.first_non_finite() {
      Some(parameter) => Err(Error::NonFiniteTransform(NonFiniteTransform::new(
        parameter,
      ))),
      None => Ok(candidate),
    }
  }

  /// `s·cos θ`.
  #[inline(always)]
  pub const fn a(&self) -> f64 {
    self.a
  }

  /// `s·sin θ`.
  #[inline(always)]
  pub const fn b(&self) -> f64 {
    self.b
  }

  /// Horizontal translation.
  #[inline(always)]
  pub const fn tx(&self) -> f64 {
    self.tx
  }

  /// Vertical translation.
  #[inline(always)]
  pub const fn ty(&self) -> f64 {
    self.ty
  }

  /// The uniform scale factor `s = √(a² + b²)`.
  #[inline]
  pub fn scale(&self) -> f64 {
    self.a.hypot(self.b)
  }

  /// The rotation `θ`, in radians.
  #[inline]
  pub fn rotation(&self) -> f64 {
    self.b.atan2(self.a)
  }

  /// The row-major `[a, −b, tx, b, a, ty]` 2×3 matrix — the same six numbers
  /// `tform.params[0:2, :]` holds, so a value here can be compared directly
  /// against the upstream reference.
  #[inline]
  pub const fn matrix(&self) -> [f64; 6] {
    [self.a, -self.b, self.tx, self.b, self.a, self.ty]
  }

  /// Maps one point through the transform.
  #[inline]
  pub fn apply(&self, p: Point) -> (f64, f64) {
    let (x, y) = (f64::from(p.x()), f64::from(p.y()));
    (
      self.a * x - self.b * y + self.tx,
      self.b * x + self.a * y + self.ty,
    )
  }

  /// The inverse transform (template → source), or `None` when no finite
  /// inverse exists.
  ///
  /// Closed form rather than a general 3×3 inversion: a similarity's inverse is
  /// a similarity, and `[[a, −b], [b, a]]⁻¹ = [[a, b], [−b, a]] / (a² + b²)`.
  ///
  /// # `None` is a fact about the RESULT, not about one expression for it
  ///
  /// [`Self::new`] is `const` and public, so all four parameters are a
  /// caller's to choose, and the predicate has to be a property of the inverse
  /// rather than of an intermediate. `None` is returned in exactly three
  /// cases, and each is a case where no inverse exists in `f64`:
  ///
  /// - a non-finite INPUT parameter — including a non-finite translation with
  ///   a perfectly good rotation, which no determinant sees;
  /// - `a = b = 0`, the rotation block that collapses the plane onto a point;
  /// - a final inverse parameter that is not finite, which is how a scale
  ///   below about `5.6e-309`, or a translation too large to carry through the
  ///   inverse scale, is refused.
  ///
  /// **Deciding it on `a² + b²` at the input's own magnitude got the answer
  /// wrong in BOTH directions**, and the fix is the predicate, not the payload
  /// the refusal carried:
  ///
  /// - at `(a, b) = (1e-160, 0)` the square is the subnormal `1e-320`, its
  ///   reciprocal overflows, and the inverse was refused — though `1/1e-160 =
  ///   1e160` is finite and exactly representable;
  /// - at `(1e200, 0)` the square is infinite, its reciprocal is zero, and the
  ///   result was `Some` holding the ZERO transform — an "inverse" mapping
  ///   every template pixel to one source point — where `1e-200` was the
  ///   answer. A false `Some` is worse than the false `None`: it warps.
  ///
  /// The entry guard on non-finite inputs is genuinely load-bearing now.
  /// Under the old arithmetic every output parameter was a product reaching
  /// every input, so a non-finite input always surfaced in the exit check and
  /// a guard on the way in would have been unreachable; the scaled reciprocal
  /// below breaks that — `(a, b) = (∞, 0)` scales to a finite `(0, −0)` — so
  /// the guard is now the only thing that catches it.
  ///
  /// # `cv2.warpAffine`'s own operation order, wherever it is defined
  ///
  /// The resampler this feeds is bit-exact with OpenCV (see the module doc)
  /// and a differently-associated inverse moves the sampled coordinate by an
  /// ulp, and with it the occasional quantised pixel. So the reference's order
  /// is used verbatim whenever the reference's own arithmetic stays inside
  /// `f64`:
  ///
  /// ```text
  /// D = M[0]*M[4] - M[1]*M[3];  D = 1./D;
  /// A11 = M[4]*D;  A22 = M[0]*D;
  /// M[0] = A11;  M[1] *= -D;  M[3] *= -D;  M[4] = A22;
  /// M[2] = -M[0]*M[2] - M[1]*M[5];
  /// M[5] = -M[3]*M[2] - M[4]*M[5];
  /// ```
  ///
  /// With `M = [a, −b, tx, b, a, ty]` the determinant `M[0]·M[4] − M[1]·M[3]`
  /// is `a·a − (−b)·b`, which rounds identically to `a² + b²`, and `A11` and
  /// `A22` coincide — so the inverse is again a similarity and fits this type.
  /// `inverse_rotation` takes that path for every scale between about
  /// `1.5e-154` and `6.7e153` — the band where `a² + b²` and its reciprocal
  /// are BOTH normal, and so every alignment a detector can produce — and
  /// falls back to a scaled reciprocal only where OpenCV's own expression has
  /// left `f64` and there is no bit-exactness left to preserve.
  #[inline]
  pub fn inverse(&self) -> Option<Self> {
    if self.first_non_finite().is_some() {
      return None;
    }
    let (a, b) = inverse_rotation(self.a, self.b)?;
    // `a` is OpenCV's `A11`/`A22`; `b` is its `M[3]` after `M[3] *= -D`, and
    // its `M[1]` is then exactly `-b`. Subtracting `(-b)·ty` and adding `b·ty`
    // are the same IEEE result, so the translation is written in the shorter
    // of the two forms.
    let inverted = Self {
      a,
      b,
      tx: -a * self.tx + b * self.ty,
      ty: -b * self.tx - a * self.ty,
    };
    inverted.first_non_finite().is_none().then_some(inverted)
  }

  /// [`Self::inverse`] with the failure REPORTED rather than swallowed — the
  /// form [`FaceAlign::to_template`] needs, which owes its caller a reason.
  ///
  /// The reason is [`Self::scale`], the quantity that decides two of
  /// [`Self::inverse`]'s three refusals, and it is read off THIS transform
  /// rather than defaulted. A payload here can only say what a
  /// `SimilarityTransform` knows: the landmark spread that produced it is not
  /// one of those things, and the old [`Error::DegenerateLandmarks`] said it
  /// anyway — as zero, on a path `estimate`'s spread guard has already proven
  /// it is not.
  ///
  /// The third refusal — a non-finite input parameter — renders as a NaN or
  /// infinite scale, which is the truth about such a transform. It is not
  /// reachable from [`FaceAlign::to_template`], whose transform comes from
  /// [`Self::estimate`] and is finite in all four parameters by construction.
  fn checked_inverse(&self) -> Result<Self> {
    self
      .inverse()
      .ok_or_else(|| Error::NonInvertibleTransform(NonInvertibleTransform::new(self.scale())))
  }

  /// The least-squares similarity mapping `source` onto `target`.
  ///
  /// Minimises `Σ ‖S·pᵢ + t − qᵢ‖²` over `S = [[a, −b], [b, a]]` and `t`.
  /// Centring removes `t` and leaves a residual that is **linear** in
  /// `(a, b)`, so the minimiser is two dot products over the centred sets:
  ///
  /// ```text
  /// a = Σ (Xᵢ · Yᵢ) / Σ ‖Xᵢ‖²        (dot)
  /// b = Σ (Xᵢ × Yᵢ) / Σ ‖Xᵢ‖²        (cross, z component)
  /// t = q̄ − S · p̄
  /// ```
  ///
  /// This is Umeyama's minimiser for the non-reflective 2-D case, reached
  /// without an SVD — see the module doc for why that is safe here and how the
  /// result is checked without appealing to either derivation.
  ///
  /// **`f64` throughout, where the reference evaluates the same minimiser in
  /// `f32`.** That is a divergence, it moves five-bit source coordinates, and
  /// it is not closable: the module doc measures it, and measures the wider
  /// spread the reference has between two builds of its own BLAS.
  ///
  /// # Errors
  /// [`Error::NonFiniteLandmark`] if any coordinate of EITHER point set is NaN
  /// or infinite, naming which set it came from;
  /// [`Error::DegenerateLandmarks`] if `Σ ‖Xᵢ‖²` is zero or non-finite, which
  /// is the case where no transform is determined;
  /// [`Error::NonFiniteTransform`] if a solved parameter is not finite.
  ///
  /// **Both sets, not just `source`.** `target` reaches the centroid and the
  /// dot products exactly as `source` does, so a NaN there used to return `Ok`
  /// holding NaN parameters and [`FaceAlign::to_template`] then emitted an
  /// all-black template instead of an error. `target` is
  /// [`ARCFACE_TEMPLATE`] on that path, but this function is PUBLIC and takes
  /// the target from its caller.
  ///
  /// The [`Error::NonFiniteTransform`] arm is a backstop rather than a
  /// reachable branch: with both `f32` point sets finite, `|a|` is bounded by
  /// `√(Σ‖Yᵢ‖² / Σ‖Xᵢ‖²)` (Cauchy–Schwarz), whose numerator is at most
  /// `10·f32::MAX² ≈ 1.2e78` and whose denominator, once nonzero, is at least
  /// the square of the smallest `f32` gap — so the quotient stays far inside
  /// `f64`. It is kept because the bound is an argument about the input type
  /// and not something the compiler enforces, and because `Ok` must never
  /// carry a NaN transform.
  pub fn estimate(
    source: &[Point; LANDMARK_COUNT],
    target: &[Point; LANDMARK_COUNT],
  ) -> Result<Self> {
    check_all_finite(source, LandmarkSet::Source)?;
    check_all_finite(target, LandmarkSet::Target)?;

    let (sx, sy) = centroid(source);
    let (tx_mean, ty_mean) = centroid(target);

    let (mut denom, mut dot, mut cross) = (0.0f64, 0.0f64, 0.0f64);
    for (s, t) in source.iter().zip(target.iter()) {
      let ux = f64::from(s.x()) - sx;
      let uy = f64::from(s.y()) - sy;
      let vx = f64::from(t.x()) - tx_mean;
      let vy = f64::from(t.y()) - ty_mean;
      denom += ux * ux + uy * uy;
      dot += ux * vx + uy * vy;
      cross += ux * vy - uy * vx;
    }

    if denom <= 0.0 || !denom.is_finite() {
      // The spread is reported for diagnostics only, and `f32` is the precision
      // the landmarks themselves carry, so narrowing it loses nothing.
      let spread = denom as f32;
      return Err(Error::DegenerateLandmarks(DegenerateLandmarks::new(spread)));
    }

    let (a, b) = (dot / denom, cross / denom);
    Self::checked(
      a,
      b,
      tx_mean - (a * sx - b * sy),
      ty_mean - (b * sx + a * sy),
    )
  }
}

/// The rotation block of `[[a, −b], [b, a]]⁻¹` — the complex reciprocal
/// `1/(a + bi)`, returned as `(re, im)`.
///
/// `None` only when no such reciprocal exists in `f64`: `a = b = 0`, or a
/// scale so small that even the scaled form overflows. Both arguments are
/// finite by [`SimilarityTransform::inverse`]'s entry guard.
///
/// Two paths, and which one runs is decided by whether the REFERENCE's
/// arithmetic is defined rather than by the size of the input:
///
/// 1. `cv2.warpAffine`'s own `D = a·a + b·b; D = 1./D;` whenever both `D` and
///    `1/D` are normal — a scale from about `1.5e-154` to `6.7e153`, so
///    every alignment a detector can produce takes it and the module's
///    bit-exactness with OpenCV is untouched.
/// 2. Otherwise Smith's scaling, which never forms the sum of squares at the
///    original magnitude. Dividing through by the larger component gives
///    `(a² + b²)/max = max + min·(min/max)`, an expression that stays inside
///    `f64` exactly when the reciprocal itself does — so `(1e-160, 0)` yields
///    `1e160` and `(1e200, 0)` yields `1e-200`, both of which the direct form
///    got wrong.
///
/// Path 2 is a different association and can differ from path 1 in the last
/// bits. That is not a divergence from OpenCV: it runs only where OpenCV's own
/// expression has already produced an infinity or a subnormal, i.e. where the
/// reference has no answer to be exact against.
fn inverse_rotation(a: f64, b: f64) -> Option<(f64, f64)> {
  let determinant = a * a + b * b;
  let reciprocal = 1.0 / determinant;
  if determinant.is_normal() && reciprocal.is_normal() {
    return Some((a * reciprocal, b * -reciprocal));
  }
  let a_dominates = a.abs() >= b.abs();
  let (larger, smaller) = if a_dominates { (a, b) } else { (b, a) };
  if larger == 0.0 {
    // `a = b = 0`. The only rotation block with no inverse at any precision:
    // it collapses the plane onto a point, and no scaling recovers a direction
    // from that.
    return None;
  }
  let ratio = smaller / larger;
  // `(a² + b²) / larger`, formed without ever holding `a² + b²`.
  let scaled = larger + smaller * ratio;
  let (re, im) = if a_dominates {
    (1.0 / scaled, -ratio / scaled)
  } else {
    (ratio / scaled, -1.0 / scaled)
  };
  (re.is_finite() && im.is_finite()).then_some((re, im))
}

/// Rejects a NaN or infinite coordinate in one NAMED point set, so the error
/// says which of [`SimilarityTransform::estimate`]'s two sides failed.
fn check_all_finite(points: &[Point; LANDMARK_COUNT], set: LandmarkSet) -> Result<()> {
  match points
    .iter()
    .position(|p| !p.x().is_finite() || !p.y().is_finite())
  {
    Some(index) => Err(Error::NonFiniteLandmark(NonFiniteLandmark::new(set, index))),
    None => Ok(()),
  }
}

/// The centroid of five landmarks, in `f64`.
fn centroid(points: &[Point; LANDMARK_COUNT]) -> (f64, f64) {
  let n = LANDMARK_COUNT as f64;
  let sum = points.iter().fold((0.0f64, 0.0f64), |(x, y), p| {
    (x + f64::from(p.x()), y + f64::from(p.y()))
  });
  (sum.0 / n, sum.1 / n)
}

/// A borrowed view of one decoded RGB8 face crop: `width · height · 3`
/// row-major, RGB-interleaved bytes.
///
/// The sans-I/O seam — decoding PNG/JPEG and cropping to the detector's box is
/// the caller's job, exactly as `clap` takes resampled 48 kHz audio and
/// `siglip` takes a decoded [`crate::embeddings::siglip::Rgb8Image`]. This
/// module deliberately keeps its own view type rather than reaching into
/// `siglip`: the `face` feature must not pull the `siglip` feature's
/// dependencies in to name a slice and two integers.
///
/// The landmarks passed alongside are in **this crop's** coordinates, not the
/// original frame's.
#[derive(Debug, Clone, Copy)]
pub struct FaceCrop<'a> {
  data: &'a [u8],
  width: usize,
  height: usize,
}

impl<'a> FaceCrop<'a> {
  /// Wrap a decoded RGB8 buffer, validating its geometry.
  ///
  /// Both axes must be nonzero and at most [`MAX_CROP_AXIS`]. The upper bound
  /// is the sampler's, not the allocator's — see that constant for the tap
  /// saturation it keeps unreachable and for OpenCV's identical assert.
  ///
  /// # Errors
  /// [`Error::CropDimensions`] if an axis is zero, exceeds [`MAX_CROP_AXIS`],
  /// or `width · height · 3` overflows `usize`; [`Error::CropDataLength`] if
  /// `data.len()` is not exactly `width · height · 3`.
  pub fn new(data: &'a [u8], width: usize, height: usize) -> Result<Self> {
    if width == 0 || height == 0 || width > MAX_CROP_AXIS || height > MAX_CROP_AXIS {
      return Err(Error::CropDimensions(CropDimensions::new(width, height)));
    }
    let expected = width
      .checked_mul(height)
      .and_then(|hw| hw.checked_mul(3))
      .ok_or(Error::CropDimensions(CropDimensions::new(width, height)))?;
    if data.len() != expected {
      return Err(Error::CropDataLength(CropDataLength::new(
        data.len(),
        expected,
      )));
    }
    Ok(Self {
      data,
      width,
      height,
    })
  }

  /// The crop width in pixels.
  #[inline]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// The crop height in pixels.
  #[inline]
  pub const fn height(&self) -> usize {
    self.height
  }

  /// The backing RGB8 bytes (`width · height · 3`, row-major, interleaved).
  #[inline]
  pub const fn data(&self) -> &'a [u8] {
    self.data
  }
}

/// One face warped onto the ArcFace 112×112 template: `112 · 112 · 3` RGB8
/// bytes, row-major and interleaved, plus the transform that produced them.
///
/// This is the embedder's ONLY input. Its pixels are still raw 0–255 bytes —
/// channel order, scale and bias belong to the model manifest
/// ([`crate::embeddings::face::Preprocessing`]), not to the alignment and not
/// to the caller.
#[derive(Debug, Clone)]
pub struct AlignedFace {
  /// `112 · 112 · 3` RGB8 bytes.
  pixels: Box<[u8; TEMPLATE_BYTES]>,
  /// The source → template transform, or `None` for pixels the caller aligned
  /// elsewhere.
  transform: Option<SimilarityTransform>,
}

impl AlignedFace {
  /// The template pixels: `112 · 112 · 3` RGB8, row-major, interleaved.
  #[inline]
  pub fn pixels(&self) -> &[u8; TEMPLATE_BYTES] {
    &self.pixels
  }

  /// The template width in pixels — always [`TEMPLATE_SIZE`].
  #[inline]
  pub const fn width(&self) -> usize {
    TEMPLATE_SIZE
  }

  /// The template height in pixels — always [`TEMPLATE_SIZE`].
  #[inline]
  pub const fn height(&self) -> usize {
    TEMPLATE_SIZE
  }

  /// The source → template transform [`FaceAlign::to_template`] solved, or
  /// `None` when the pixels came from [`Self::from_template_pixels`].
  ///
  /// Useful for mapping a template coordinate back into the original crop —
  /// invert it with [`SimilarityTransform::inverse`].
  #[inline]
  pub const fn transform(&self) -> Option<&SimilarityTransform> {
    self.transform.as_ref()
  }

  /// Wrap `112 · 112 · 3` RGB8 bytes a caller aligned elsewhere.
  ///
  /// The bring-your-own-alignment bypass, for a pipeline that already runs the
  /// ArcFace warp. **coremlit cannot check that these pixels came from the
  /// ArcFace template** — a crop aligned to some other template, or not aligned
  /// at all, passes this constructor and **silently degrades** every cosine
  /// computed from it. Prefer [`FaceAlign::to_template`].
  ///
  /// # Errors
  /// [`Error::CropDataLength`] if `pixels.len()` is not exactly
  /// [`TEMPLATE_BYTES`].
  pub fn from_template_pixels(pixels: &[u8]) -> Result<Self> {
    let exact: [u8; TEMPLATE_BYTES] = pixels
      .try_into()
      .map_err(|_| Error::CropDataLength(CropDataLength::new(pixels.len(), TEMPLATE_BYTES)))?;
    Ok(Self {
      pixels: Box::new(exact),
      transform: None,
    })
  }
}

/// The 5-point similarity alignment onto the ArcFace 112×112 template.
///
/// A unit type rather than a free function so the template it targets is named
/// at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceAlign;

impl FaceAlign {
  /// Warps `crop` onto the ArcFace 112×112 template using `landmarks5`.
  ///
  /// `landmarks5` are in `crop`'s own pixel coordinates, in
  /// [`ARCFACE_TEMPLATE`] order (left eye, right eye, nose tip, left mouth
  /// corner, right mouth corner — the **viewer's** left and right). Sampling is
  /// bilinear with a constant-0 border, so a template pixel whose source falls
  /// outside the crop reads black rather than clamping an edge pixel across the
  /// face; that is `cv2.warpAffine(..., borderValue=0.0)`, and it is what the
  /// ArcFace family was trained against.
  ///
  /// Diverges from issue #115 §1's signature by returning a [`Result`]: five
  /// coincident or non-finite landmarks determine no transform, and a silent
  /// all-NaN template would be strictly worse than an error.
  ///
  /// # Errors
  /// [`Error::NonFiniteLandmark`] on a NaN or infinite landmark coordinate;
  /// [`Error::DegenerateLandmarks`] if the five landmarks carry no spread;
  /// [`Error::NonInvertibleTransform`] if the solve nonetheless produced a
  /// transform with no inverse — a DIFFERENT geometry, and a different error,
  /// because the landmarks are spread in that case and only the scale
  /// collapsed; [`Error::CoordinateOverflow`] if the inverse is finite but so
  /// large that a destination → source coordinate leaves the `int` fixed-point
  /// domain the sampler computes it in.
  ///
  /// **The last of those is the arm a near-degenerate detection reaches.** Five
  /// finite, in-bounds landmarks that are nearly collinear solve to a nonzero
  /// scale, pass the spread guard, invert cleanly, and then map a destination
  /// pixel a billion columns outside the crop — which is not a face, and is now
  /// said so rather than sampled.
  pub fn to_template(
    crop: FaceCrop<'_>,
    landmarks5: &[Point; LANDMARK_COUNT],
  ) -> Result<AlignedFace> {
    let transform = SimilarityTransform::estimate(landmarks5, &ARCFACE_TEMPLATE)?;
    // A real backstop, against a real geometry — NOT an impossibility, which
    // is what the two obvious arguments for it both get wrong. The solved
    // scale is `|Σ conj(uᵢ)·vᵢ| / Σ‖uᵢ‖²` over the two CENTRED point sets, and
    // that vanishes when the sets are orthogonal in that inner product, which
    // neither `estimate`'s rejection of a zero SOURCE spread nor the fact that
    // [`ARCFACE_TEMPLATE`] has spread of its own rules out: a well-spread
    // source exists (`Σ‖uᵢ‖² ≈ 9.3e4`) whose exact solved scale against this
    // very template is 5e-18. Rounding such a source through `Point`'s `f32`
    // has always left the scale nonzero — 9e-10 for the constructed case — so
    // no input is KNOWN to reach the arm, and it is kept because that is a
    // failure to find one rather than a proof that none exists.
    //
    // Reachable directly, though, and pinned by
    // `estimate_can_return_a_transform_with_no_inverse`: `estimate` is public
    // and takes its target from the caller, so a zero-spread target hands back
    // a finite `a = b = 0` that inverts to `None`.
    //
    // Whichever way it is reached, the error names the SCALE. It used to name
    // a landmark spread of zero, which no execution of this line can have:
    // `estimate` returned `Ok`, so its own guard has already established that
    // the spread is positive and finite.
    let inverse = transform.checked_inverse()?;
    Ok(AlignedFace {
      pixels: Box::new(warp_bilinear(crop, &inverse)?),
      transform: Some(transform),
    })
  }
}

/// OpenCV's `INTER_BITS`: the inverse-mapped source coordinate keeps five
/// fractional bits, so its interpolation weight is drawn from a 32-step table
/// and a true fraction under `1/64` quantises to zero.
const INTER_BITS: u32 = 5;

/// OpenCV's `INTER_TAB_SIZE`, `1 << INTER_BITS`.
const INTER_TAB_SIZE: i64 = 1 << INTER_BITS;

/// OpenCV's `AB_BITS`, `MAX(10, INTER_BITS)`: the precision the per-row and
/// per-column halves of the mapped coordinate are rounded to BEFORE they are
/// added together.
const AB_BITS: u32 = 10;

/// OpenCV's `AB_SCALE`, `1 << AB_BITS`.
const AB_SCALE: f64 = (1i64 << AB_BITS) as f64;

/// OpenCV's `round_delta` for a non-nearest interpolation,
/// `AB_SCALE / INTER_TAB_SIZE / 2` — the half-step folded in so that the
/// truncating shift down to the five-bit grid becomes a round-to-nearest.
const ROUND_DELTA: i32 = (1i32 << AB_BITS) / (1i32 << INTER_BITS) / 2;

/// OpenCV's `INTER_REMAP_COEF_BITS`: the four interpolation weights are
/// 15-bit fixed point and sum to exactly `1 << 15`.
const REMAP_COEF_BITS: u32 = 15;

/// OpenCV's `cvRound`, which is `lrint` under the default rounding mode:
/// nearest, **ties to even** — not the half-up rounding used for pixels.
///
/// `None` outside `int`, rather than a saturated value, and that is the whole
/// point of the signature. OpenCV reaches this through
/// `saturate_cast<int>(double)`, which is *undefined* past `int`'s range: there
/// is no reference answer to reproduce, only a domain to stay inside. Rust has
/// to define something, and the two definitions on offer are not equally safe.
///
/// **Saturating was tried and is wrong.** The coordinate is SPLIT — a per-column
/// term and a per-row term, each rounded on its own, then added — so two
/// saturated terms can cancel: `i32::MIN + 16` plus `i32::MAX` is `15`, which
/// after the shift onto the five-bit grid is source pixel `0`. A destination
/// pixel whose true source is 1.9 billion columns outside the crop then reads
/// the crop's own first pixel and reports nothing. Refusing the term instead
/// makes the map total, because a term that has no answer cannot cancel against
/// another that has none either.
#[inline]
fn cv_round(value: f64) -> Option<i32> {
  let rounded = value.round_ties_even();
  // NaN fails both comparisons, so it is refused here rather than reaching the
  // cast. `i32::MIN` and `i32::MAX` are exactly representable in `f64`, so the
  // bounds are the exact ones and the cast below cannot saturate.
  (rounded >= f64::from(i32::MIN) && rounded <= f64::from(i32::MAX)).then_some(rounded as i32)
}

/// [`cv_round`] with the domain failure NAMED, so a refusal can say which term
/// of which coordinate left `int`.
#[inline]
fn rounded_term(value: f64, axis: CoordinateAxis, term: CoordinateTerm) -> Result<i32> {
  cv_round(value)
    .ok_or_else(|| Error::CoordinateOverflow(CoordinateOverflow::new(axis, term, value)))
}

/// `cv2.warpAffine(crop, M, (112, 112), flags=INTER_LINEAR,
/// borderMode=BORDER_CONSTANT, borderValue=0)`, given `M⁻¹`.
///
/// Destination-driven, and **fixed point throughout** — see the module doc for
/// why a float bilinear kernel is a different function and not a rounding of
/// this one. The structure mirrors `imgwarp.cpp`'s `WarpAffineInvoker`: the
/// per-destination-column contribution is rounded to `1/AB_SCALE` of a pixel
/// once (`adelta`/`bdelta`), the per-row half likewise, and only then are the
/// two added and dropped to the five-bit grid. That intermediate rounding is
/// part of the answer, so it is reproduced rather than folded into a single
/// expression: `cvRound(a·u·1024) + cvRound(m·v·1024)` is not
/// `cvRound((a·u + m·v)·1024)`.
///
/// # Errors
/// [`Error::CoordinateOverflow`] if any term of the destination → source map,
/// or the sum of two of them, leaves the `int` domain OpenCV computes it in.
/// The whole map is built and checked BEFORE the first sample, so this is a
/// refusal rather than a template warped from some pixels and not others.
fn warp_bilinear(
  crop: FaceCrop<'_>,
  inverse: &SimilarityTransform,
) -> Result<[u8; TEMPLATE_BYTES]> {
  let mut out = [0u8; TEMPLATE_BYTES];
  let (width, height) = (crop.width(), crop.height());
  let data = crop.data();
  // The same six numbers `cv2.warpAffine` holds in `M` after inverting it.
  let grid = SourceGrid::new(inverse.matrix())?;

  for v in 0..TEMPLATE_SIZE {
    let origin = grid.row_origin(v);
    for u in 0..TEMPLATE_SIZE {
      let (x, y) = grid.at(origin, u);
      let base = (v * TEMPLATE_SIZE + u) * 3;
      sample_fixed_point(data, width, height, x, y, &mut out[base..base + 3]);
    }
  }
  Ok(out)
}

/// The destination → five-bit source coordinate map `cv2.warpAffine` walks,
/// for one ALREADY-INVERTED 2×3 matrix.
///
/// Split out of [`warp_bilinear`] for two reasons. It names the intermediate
/// rounding that is part of the answer — the per-column and per-row halves are
/// each rounded to `1/AB_SCALE` before they are added, so
/// `cvRound(a·u·1024) + cvRound(m·v·1024)` is not `cvRound((a·u + m·v)·1024)`.
/// And it lets two matrices be compared on the coordinates themselves rather
/// than on pixels, which is what
/// `the_solve_diverges_from_skimage_by_less_than_skimage_diverges_from_itself`
/// needs: a moved coordinate leaves the output unchanged wherever the
/// neighbourhood it lands in happens to be flat, so counting differing bytes
/// under-reports a moved map.
///
/// Takes a raw `[f64; 6]` rather than a [`SimilarityTransform`] because the
/// reference's own solved matrix usually is NOT a similarity — see the module
/// doc — and comparing against one means being able to walk one.
///
/// **Every term is `i32` and every one of them was checked before this value
/// existed**, which is what makes the map total. A `SourceGrid` cannot be
/// constructed for a transform whose coordinates leave `int`, so [`Self::at`]
/// is infallible for the same reason a `FaceCrop`'s indices are: the
/// constructor is the only door.
struct SourceGrid {
  /// OpenCV's `adelta`: the per-destination-COLUMN half of the mapped `x`,
  /// rounded to `1/AB_SCALE` of a pixel on its own.
  adelta: [i32; TEMPLATE_SIZE],
  /// OpenCV's `bdelta`, the same for `y`.
  bdelta: [i32; TEMPLATE_SIZE],
  /// OpenCV's `X0` per destination ROW, [`ROUND_DELTA`] already folded in.
  x_origin: [i32; TEMPLATE_SIZE],
  /// OpenCV's `Y0`, the same for `y`.
  y_origin: [i32; TEMPLATE_SIZE],
}

impl SourceGrid {
  /// Builds and VALIDATES the whole destination → source map for a template →
  /// source 2×3.
  ///
  /// Both halves are computed here, not just the per-column one, because the
  /// per-row half is where `round_delta` is added and that addition is one of
  /// the places the coordinate can leave `int` — the reviewer's witness leaves
  /// it exactly there, at `i32::MAX + 16`.
  ///
  /// # Errors
  /// [`Error::CoordinateOverflow`] naming the axis and the term, for any of
  /// the three: a per-column term, a per-row term (rounding or the
  /// `round_delta` fold), or the sum the two form.
  fn new(inverse: [f64; 6]) -> Result<Self> {
    let mut adelta = [0i32; TEMPLATE_SIZE];
    let mut bdelta = [0i32; TEMPLATE_SIZE];
    for (u, (a, b)) in adelta.iter_mut().zip(bdelta.iter_mut()).enumerate() {
      // `u` is below 112, so `u as f64` is exact.
      let uf = u as f64;
      *a = rounded_term(
        inverse[0] * uf * AB_SCALE,
        CoordinateAxis::X,
        CoordinateTerm::ColumnDelta,
      )?;
      *b = rounded_term(
        inverse[3] * uf * AB_SCALE,
        CoordinateAxis::Y,
        CoordinateTerm::ColumnDelta,
      )?;
    }

    let mut x_origin = [0i32; TEMPLATE_SIZE];
    let mut y_origin = [0i32; TEMPLATE_SIZE];
    for (v, (x, y)) in x_origin.iter_mut().zip(y_origin.iter_mut()).enumerate() {
      // `v` is below 112, so `v as f64` is exact.
      let vf = v as f64;
      *x = row_origin_term((inverse[1] * vf + inverse[2]) * AB_SCALE, CoordinateAxis::X)?;
      *y = row_origin_term((inverse[4] * vf + inverse[5]) * AB_SCALE, CoordinateAxis::Y)?;
    }

    check_sum_domain(&x_origin, &adelta, CoordinateAxis::X)?;
    check_sum_domain(&y_origin, &bdelta, CoordinateAxis::Y)?;

    Ok(Self {
      adelta,
      bdelta,
      x_origin,
      y_origin,
    })
  }

  /// Destination row `v`'s accumulator origin, with [`ROUND_DELTA`] folded in
  /// so the shift down onto the five-bit grid rounds rather than truncates.
  ///
  /// Read from the table [`Self::new`] built: the rounding and the fold both
  /// happened there, where a term outside `int` could still be reported.
  #[inline]
  fn row_origin(&self, v: usize) -> (i32, i32) {
    (self.x_origin[v], self.y_origin[v])
  }

  /// The five-bit source coordinate destination `(u, v)` samples at, given
  /// that row's origin from [`Self::row_origin`].
  ///
  /// The sum is formed in `i64` and cannot overflow there; it is also
  /// guaranteed by [`check_sum_domain`] to fit in `i32`, so it holds exactly
  /// the value C++'s `int X = X0 + adelta[x1]` holds. Arithmetic shift, so it
  /// floors for a negative coordinate exactly as C++'s does.
  #[inline]
  fn at(&self, origin: (i32, i32), u: usize) -> (i64, i64) {
    (
      (i64::from(origin.0) + i64::from(self.adelta[u])) >> (AB_BITS - INTER_BITS),
      (i64::from(origin.1) + i64::from(self.bdelta[u])) >> (AB_BITS - INTER_BITS),
    )
  }
}

/// One row origin: [`cv_round`] plus OpenCV's `round_delta`, with BOTH steps
/// required to stay inside `int`.
///
/// The fold is checked separately because it is a real overflow site and not a
/// formality: the witness that motivated this whole path rounds to `i32::MAX`
/// and then adds 16.
fn row_origin_term(value: f64, axis: CoordinateAxis) -> Result<i32> {
  let rounded = rounded_term(value, axis, CoordinateTerm::RowOrigin)?;
  rounded.checked_add(ROUND_DELTA).ok_or_else(|| {
    Error::CoordinateOverflow(CoordinateOverflow::new(
      axis,
      CoordinateTerm::RowOrigin,
      // Exact: both operands are `i32`, so the sum is far inside `f64`'s
      // integer range.
      f64::from(rounded) + f64::from(ROUND_DELTA),
    ))
  })
}

/// Proves every one of the 112² sums `origin[v] + delta[u]` fits in the single
/// `int` OpenCV forms it in — with two checked additions rather than 12 544.
///
/// Addition is monotone in both arguments, so for every pair
/// `min(origin) + min(delta) ≤ origin[v] + delta[u] ≤ max(origin) + max(delta)`,
/// and `i32` is a contiguous interval: if both bounds are representable, so is
/// everything between them. Checking the two extreme pairs therefore covers
/// the whole grid, and it is why [`SourceGrid::at`] can be infallible.
///
/// # Errors
/// [`Error::CoordinateOverflow`] with [`CoordinateTerm::Sum`], carrying the
/// offending sum computed in `f64` where it does not overflow.
fn check_sum_domain(
  origins: &[i32; TEMPLATE_SIZE],
  deltas: &[i32; TEMPLATE_SIZE],
  axis: CoordinateAxis,
) -> Result<()> {
  let extremes = |values: &[i32; TEMPLATE_SIZE]| {
    values
      .iter()
      .fold((i32::MAX, i32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
  };
  let (origin_lo, origin_hi) = extremes(origins);
  let (delta_lo, delta_hi) = extremes(deltas);
  for (origin, delta) in [(origin_lo, delta_lo), (origin_hi, delta_hi)] {
    if origin.checked_add(delta).is_none() {
      return Err(Error::CoordinateOverflow(CoordinateOverflow::new(
        axis,
        CoordinateTerm::Sum,
        f64::from(origin) + f64::from(delta),
      )));
    }
  }
  Ok(())
}

/// One destination pixel from a five-bit fixed-point source coordinate:
/// `remapBilinear`'s tap gather, weight table and output cast.
///
/// `x` and `y` are in units of `1/INTER_TAB_SIZE` of a source pixel. The high
/// bits are the integer tap and the low [`INTER_BITS`] are the fraction's table
/// index, which is how OpenCV splits them:
///
/// ```text
/// xy[k]  = saturate_cast<short>(X >> INTER_BITS);
/// alpha  = (Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1));
/// ```
fn sample_fixed_point(data: &[u8], width: usize, height: usize, x: i64, y: i64, out: &mut [u8]) {
  // The integer tap, EXACT, where OpenCV writes `saturate_cast<short>(X >>
  // INTER_BITS)` into its `short XY[]` and then tests THAT against the source
  // extent.
  //
  // The two agree everywhere the reference is defined, and the exact form is
  // total where the saturating one is not. OpenCV's `remap` asserts
  // `src.cols < SHRT_MAX`, so a saturated tap — `i16::MIN`, or `i16::MAX`,
  // whose successor tap is `32 768` — is outside every admitted crop and reads
  // the constant-0 border, exactly as the unsaturated coordinate does; the
  // taps the clamp leaves alone are unchanged by definition. Past that assert
  // the two part company, and it is the saturating form that is wrong: column
  // 33 000 of a 40 000-wide crop arrives as `i16::MAX`, which IS a column of
  // that crop, so the sampler reads a different region and reports nothing.
  // Writing the tap exactly removes that aliasing from the operation instead
  // of keeping it unreachable by a bound on the caller's geometry.
  //
  // No overflow to guard here: `x` and `y` come from `SourceGrid::at`, whose
  // sums are inside `i32`, so both are within `±2³¹ ⁄ 32` before this shift.
  let sx = x >> INTER_BITS;
  let sy = y >> INTER_BITS;
  let fx = x & (INTER_TAB_SIZE - 1);
  let fy = y & (INTER_TAB_SIZE - 1);

  let weights = bilinear_weights(fx, fy);

  let mut acc = [0i64; 3];
  for ((dy, dx), weight) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)]
    .into_iter()
    .zip(weights)
  {
    // `BORDER_CONSTANT` with `borderValue = 0`: an out-of-range tap
    // contributes `0 · weight`, which is what skipping it adds. The bounds are
    // tested on the QUANTISED tap, as OpenCV tests them — deciding the border
    // from the unrounded coordinate instead would disagree wherever the
    // rounding crosses a pixel boundary.
    let Some(index) = tap_index(sx + dx, sy + dy, width, height) else {
      continue;
    };
    for (channel, slot) in acc.iter_mut().enumerate() {
      *slot += weight * i64::from(data[index + channel]);
    }
  }
  for (slot, value) in out.iter_mut().zip(acc) {
    *slot = fixed_point_to_u8(value);
  }
}

/// `BilinearTab_i[fy·INTER_TAB_SIZE + fx]`, in tap order
/// `(0,0), (0,1), (1,0), (1,1)`.
///
/// The products of the two 1-D weights `(1 − i/32, i/32)` scaled by
/// `1 << INTER_REMAP_COEF_BITS`. Every entry is an exact integer and the four
/// sum to exactly `1 << 15`, which is what makes the fixed-point cast below
/// unbiased.
///
/// OpenCV's own table differs in ONE of its 1 024 cells: `initInterTab2D`
/// fills it with `saturate_cast<short>`, so the unit weight at fraction
/// `(0, 0)` saturates to 32 767 and the sum-fixing step that follows puts the
/// missing 1 on the opposite corner — `[32767, 0, 0, 1]` where this returns
/// `[32768, 0, 0, 0]`. For a `u8` source the two are the same function;
/// `the_saturating_weight_table_cell_is_invisible_for_u8_sources` proves that
/// exhaustively over the 65 536 tap pairs the difference can see, so the exact
/// form is used here rather than a transcription of an overflow.
#[inline]
fn bilinear_weights(fx: i64, fy: i64) -> [i64; 4] {
  [
    (INTER_TAB_SIZE - fy) * (INTER_TAB_SIZE - fx) * INTER_TAB_SIZE,
    (INTER_TAB_SIZE - fy) * fx * INTER_TAB_SIZE,
    fy * (INTER_TAB_SIZE - fx) * INTER_TAB_SIZE,
    fy * fx * INTER_TAB_SIZE,
  ]
}

/// The byte offset of the source tap at `(x, y)`, or `None` when it lies
/// outside the crop and reads the constant-0 border.
fn tap_index(x: i64, y: i64, width: usize, height: usize) -> Option<usize> {
  let x = usize::try_from(x).ok()?;
  let y = usize::try_from(y).ok()?;
  if x >= width || y >= height {
    return None;
  }
  Some((y * width + x) * 3)
}

/// OpenCV's `FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>`: add half a unit,
/// shift down, saturate into `u8`.
///
/// Half-up on the fixed-point accumulator, which is NOT the same tie rule as
/// [`cv_round`]'s half-to-even on the coordinate; both are reproduced as
/// OpenCV has them.
#[inline]
fn fixed_point_to_u8(value: i64) -> u8 {
  let rounded = (value + (1i64 << (REMAP_COEF_BITS - 1))) >> REMAP_COEF_BITS;
  rounded.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests;
