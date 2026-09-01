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
//! one, plus `cv_round_breaks_ties_to_even_and_saturates` and
//! `the_fixed_point_pixel_cast_rounds_half_up_and_saturates` for the two tie
//! rules that no whole-image comparison can see.

use crate::embeddings::face::error::{
  CropDataLength, CropDimensions, DegenerateLandmarks, Error, LandmarkSet, NonFiniteLandmark,
  NonFiniteTransform, NonInvertibleTransform, Result, TransformParameter,
};

/// The number of landmarks the ArcFace family aligns on.
pub const LANDMARK_COUNT: usize = 5;

/// The ArcFace template's side, in pixels.
pub const TEMPLATE_SIZE: usize = 112;

/// Bytes in one [`AlignedFace`]: `112 · 112 · 3`, RGB8 interleaved.
pub const TEMPLATE_BYTES: usize = TEMPLATE_SIZE * TEMPLATE_SIZE * 3;

/// The largest crop axis [`FaceCrop::new`] admits: one short of `i16::MAX`.
///
/// **This is the sampler's fixed-point domain, not a buffer limit.**
/// [`FaceAlign::to_template`] saturates each integer source tap into `i16` —
/// OpenCV's own `saturate_cast<short>` on the `short XY[]` its
/// `WarpAffineInvoker` fills — and the constant-0 border is then decided by
/// comparing the SATURATED tap against the crop's extent. That test is right
/// only while the saturation value is not a coordinate the crop actually has.
/// Past this bound it is: a 40 000-column crop whose inverse asks for column
/// 33 000 gets `i16::MAX` back, and 32 767 IS a column of that crop, so the
/// sampler reads a wholly different region and reports nothing.
/// `the_i16_tap_limit_is_why_a_wide_crop_is_refused` measures that read.
///
/// **Refused rather than widened**, because the sampler's contract is to be
/// bit-exact with `cv2.warpAffine` (see the module doc) and OpenCV draws this
/// line itself: 4.x's `remap` — the fixed-point pipeline `warpAffine` funnels
/// into — opens with `CV_Assert( dst.cols < SHRT_MAX && dst.rows < SHRT_MAX &&
/// src.cols < SHRT_MAX && src.rows < SHRT_MAX )`. A wider tap would sample
/// geometry the reference refuses outright, which trades a silent wrong answer
/// for a silent divergence rather than removing one.
///
/// **Why `i16::MAX − 1` and not `i16::MAX`.** Correctness alone would admit an
/// axis of exactly `i16::MAX`: a crop that wide has its last column at
/// `i16::MAX − 1`, so the saturation value is still outside it and the border
/// test still holds. This takes OpenCV's strictly-less bound instead, so the
/// admitted set is exactly the one the reference admits and the two cannot
/// disagree about a crop at the boundary. One pixel of conservatism, chosen to
/// keep a second number from existing.
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
  /// `None` on a zero scale, and equally on a non-finite parameter — including
  /// a non-finite TRANSLATION with a perfectly good rotation, which the
  /// determinant alone does not see. [`Self::new`] is `const` and public, so
  /// that is a value a caller can hand in, and returning `Some` for it would
  /// mean handing back an inverse whose [`Self::apply`] is NaN everywhere.
  ///
  /// ONE check, on the way out, and that is deliberate: every parameter of the
  /// result is a sum of products of all four inputs, so a non-finite input
  /// reaches at least one output parameter and a guard on the way in would be
  /// unreachable — a line no test could distinguish from its absence. The exit
  /// check is not redundant with it, though: a scale small enough that
  /// `1.0 / (a² + b²)` overflows turns four finite parameters into an infinite
  /// one, which is what
  /// `an_inverse_is_refused_when_any_parameter_is_non_finite` pins.
  ///
  /// The arithmetic follows `cv2.warpAffine`'s own inversion in ITS operation
  /// order — one reciprocal, then multiplies — because the resampler this
  /// feeds is bit-exact with OpenCV (see the module doc) and a
  /// differently-associated inverse moves the sampled coordinate by an ulp,
  /// and with it the occasional quantised pixel:
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
  #[inline]
  pub fn inverse(&self) -> Option<Self> {
    let determinant = self.a * self.a + self.b * self.b;
    if determinant == 0.0 {
      return None;
    }
    let reciprocal = 1.0 / determinant;
    // `a` is OpenCV's `A11`/`A22`; `b` is its `M[3]` after `M[3] *= -D`, and
    // its `M[1]` is then exactly `-b`. Subtracting `(-b)·ty` and adding `b·ty`
    // are the same IEEE result, so the translation is written in the shorter
    // of the two forms.
    let a = self.a * reciprocal;
    let b = self.b * -reciprocal;
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
  /// The reason is [`Self::scale`], the quantity that actually decides it, and
  /// it is read off THIS transform rather than defaulted. A payload here can
  /// only say what a `SimilarityTransform` knows: the landmark spread that
  /// produced it is not one of those things, and the old
  /// [`Error::DegenerateLandmarks`] said it anyway — as zero, on a path
  /// `estimate`'s spread guard has already proven it is not.
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
  /// collapsed.
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
      pixels: Box::new(warp_bilinear(crop, &inverse)),
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
const ROUND_DELTA: i64 = (1i64 << AB_BITS) / INTER_TAB_SIZE / 2;

/// OpenCV's `INTER_REMAP_COEF_BITS`: the four interpolation weights are
/// 15-bit fixed point and sum to exactly `1 << 15`.
const REMAP_COEF_BITS: u32 = 15;

/// OpenCV's `cvRound`, which is `lrint` under the default rounding mode:
/// nearest, **ties to even** — not the half-up rounding used for pixels.
///
/// OpenCV reaches this through `saturate_cast<int>(double)`, which is
/// undefined outside `int`'s range; Rust has to define it, so it saturates
/// there. No solved alignment reaches that: [`SimilarityTransform::estimate`]
/// and [`SimilarityTransform::inverse`] both refuse a non-finite parameter,
/// and a saturated coordinate lands outside any crop and reads the border
/// either way.
#[inline]
fn cv_round(value: f64) -> i64 {
  let rounded = value.round_ties_even();
  if rounded >= f64::from(i32::MAX) {
    i64::from(i32::MAX)
  } else if rounded <= f64::from(i32::MIN) {
    i64::from(i32::MIN)
  } else {
    rounded as i64
  }
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
fn warp_bilinear(crop: FaceCrop<'_>, inverse: &SimilarityTransform) -> [u8; TEMPLATE_BYTES] {
  let mut out = [0u8; TEMPLATE_BYTES];
  let (width, height) = (crop.width(), crop.height());
  let data = crop.data();
  // The same six numbers `cv2.warpAffine` holds in `M` after inverting it.
  let grid = SourceGrid::new(inverse.matrix());

  for v in 0..TEMPLATE_SIZE {
    let origin = grid.row_origin(v);
    for u in 0..TEMPLATE_SIZE {
      let (x, y) = grid.at(origin, u);
      let base = (v * TEMPLATE_SIZE + u) * 3;
      sample_fixed_point(data, width, height, x, y, &mut out[base..base + 3]);
    }
  }
  out
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
struct SourceGrid {
  /// The inverted 2×3, row-major, as `cv2.warpAffine` holds `M`.
  m: [f64; 6],
  /// OpenCV's `adelta`: the per-destination-COLUMN half of the mapped `x`,
  /// rounded to `1/AB_SCALE` of a pixel on its own.
  adelta: [i64; TEMPLATE_SIZE],
  /// OpenCV's `bdelta`, the same for `y`.
  bdelta: [i64; TEMPLATE_SIZE],
}

impl SourceGrid {
  /// Precomputes the per-column halves for a template → source 2×3.
  fn new(inverse: [f64; 6]) -> Self {
    let mut adelta = [0i64; TEMPLATE_SIZE];
    let mut bdelta = [0i64; TEMPLATE_SIZE];
    for (u, (a, b)) in adelta.iter_mut().zip(bdelta.iter_mut()).enumerate() {
      // `u` is below 112, so `u as f64` is exact.
      let uf = u as f64;
      *a = cv_round(inverse[0] * uf * AB_SCALE);
      *b = cv_round(inverse[3] * uf * AB_SCALE);
    }
    Self {
      m: inverse,
      adelta,
      bdelta,
    }
  }

  /// Destination row `v`'s accumulator origin, with [`ROUND_DELTA`] folded in
  /// so the shift down onto the five-bit grid rounds rather than truncates.
  fn row_origin(&self, v: usize) -> (i64, i64) {
    let vf = v as f64;
    (
      cv_round((self.m[1] * vf + self.m[2]) * AB_SCALE) + ROUND_DELTA,
      cv_round((self.m[4] * vf + self.m[5]) * AB_SCALE) + ROUND_DELTA,
    )
  }

  /// The five-bit source coordinate destination `(u, v)` samples at, given
  /// that row's origin from [`Self::row_origin`].
  ///
  /// Arithmetic shift, so it floors for a negative coordinate exactly as
  /// C++'s does.
  #[inline]
  fn at(&self, origin: (i64, i64), u: usize) -> (i64, i64) {
    (
      (origin.0 + self.adelta[u]) >> (AB_BITS - INTER_BITS),
      (origin.1 + self.bdelta[u]) >> (AB_BITS - INTER_BITS),
    )
  }
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
  // `saturate_cast<short>` on the integer tap. It cannot pull an out-of-range
  // coordinate back INTO a crop — no crop is 32 768 pixels wide — so it only
  // ever keeps the arithmetic below in range.
  let sx = (x >> INTER_BITS).clamp(i64::from(i16::MIN), i64::from(i16::MAX));
  let sy = (y >> INTER_BITS).clamp(i64::from(i16::MIN), i64::from(i16::MAX));
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
