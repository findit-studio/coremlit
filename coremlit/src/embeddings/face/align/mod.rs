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
//! `INTER_LINEAR`, constant-0 border. [`FaceAlign::to_template`] does exactly
//! that.
//!
//! # Two deliberate numerical divergences, both recorded rather than chased
//!
//! 1. **Float weights, not OpenCV's 5-bit fixed point.** For 8-bit input
//!    OpenCV quantises the bilinear weights to `INTER_BITS = 5`; this module
//!    accumulates in `f64`. The two differ by at most one LSB per channel.
//! 2. **Half-up rounding.** OpenCV's `saturate_cast<uchar>` rounds half to
//!    even; this module rounds half away from zero (`⌊v + 0.5⌋`, then clamps).
//!    It bites only on an exact `.5`.
//!
//! Both sit around **1/255 ≈ 0.004 of a channel**, against a measured ANE fp16
//! embedding floor of `1 − cos ≈ 0.0015` typical / `0.0025` worst and a
//! cheapest-real-preprocessing-bug distance of `0.083` (issue #115's parity
//! census). Chasing bit-exact OpenCV would buy nothing measurable and would
//! make the sampler untestable against anything but OpenCV itself.
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
//!   through a different derivation and resamples in numpy.

use crate::embeddings::face::error::{
  CropDataLength, CropDimensions, DegenerateLandmarks, Error, NonFiniteLandmark, Result,
};

/// The number of landmarks the ArcFace family aligns on.
pub const LANDMARK_COUNT: usize = 5;

/// The ArcFace template's side, in pixels.
pub const TEMPLATE_SIZE: usize = 112;

/// Bytes in one [`AlignedFace`]: `112 · 112 · 3`, RGB8 interleaved.
pub const TEMPLATE_BYTES: usize = TEMPLATE_SIZE * TEMPLATE_SIZE * 3;

/// One 2-D point in a crop's pixel coordinates, pixel centres on integers.
///
/// `f32` because that is what a detector emits, and because every coordinate
/// the alignment consumes is promoted to `f64` for the solve anyway.
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
  #[inline(always)]
  pub const fn new(a: f64, b: f64, tx: f64, ty: f64) -> Self {
    Self { a, b, tx, ty }
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

  /// The inverse transform (template → source), or `None` when the scale is
  /// zero and no inverse exists.
  ///
  /// Closed form rather than a general 3×3 inversion: a similarity's inverse is
  /// a similarity, and `[[a, −b], [b, a]]⁻¹ = [[a, b], [−b, a]] / (a² + b²)`.
  #[inline]
  pub fn inverse(&self) -> Option<Self> {
    let det = self.a * self.a + self.b * self.b;
    if det == 0.0 || !det.is_finite() {
      return None;
    }
    let (ia, ib) = (self.a / det, -self.b / det);
    Some(Self {
      a: ia,
      b: ib,
      tx: -(ia * self.tx - ib * self.ty),
      ty: -(ib * self.tx + ia * self.ty),
    })
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
  /// # Errors
  /// [`Error::NonFiniteLandmark`] if any `source` coordinate is NaN or
  /// infinite; [`Error::DegenerateLandmarks`] if `Σ ‖Xᵢ‖²` is zero or
  /// non-finite, which is the case where no transform is determined.
  pub fn estimate(
    source: &[Point; LANDMARK_COUNT],
    target: &[Point; LANDMARK_COUNT],
  ) -> Result<Self> {
    for (index, p) in source.iter().enumerate() {
      if !p.x().is_finite() || !p.y().is_finite() {
        return Err(Error::NonFiniteLandmark(NonFiniteLandmark::new(index)));
      }
    }

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
    Ok(Self {
      a,
      b,
      tx: tx_mean - (a * sx - b * sy),
      ty: ty_mean - (b * sx + a * sy),
    })
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
  /// # Errors
  /// [`Error::CropDimensions`] if an axis is zero or `width · height · 3`
  /// overflows `usize`; [`Error::CropDataLength`] if `data.len()` is not
  /// exactly `width · height · 3`.
  pub fn new(data: &'a [u8], width: usize, height: usize) -> Result<Self> {
    if width == 0 || height == 0 {
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
  /// [`Error::DegenerateLandmarks`] if the five landmarks carry no spread.
  pub fn to_template(
    crop: FaceCrop<'_>,
    landmarks5: &[Point; LANDMARK_COUNT],
  ) -> Result<AlignedFace> {
    let transform = SimilarityTransform::estimate(landmarks5, &ARCFACE_TEMPLATE)?;
    // `estimate` rejects a zero spread, so the scale is nonzero and the
    // inverse exists; the `ok_or_else` is a total-function backstop, not a path
    // any input reaches.
    let inverse = transform
      .inverse()
      .ok_or_else(|| Error::DegenerateLandmarks(DegenerateLandmarks::new(0.0)))?;
    Ok(AlignedFace {
      pixels: Box::new(warp_bilinear(crop, &inverse)),
      transform: Some(transform),
    })
  }
}

/// `cv2.warpAffine(crop, M, (112, 112), borderValue=0.0)`, given `M⁻¹`.
///
/// Destination-driven: each template pixel centre is mapped back into the crop
/// and bilinearly sampled, with out-of-range taps contributing zero. `f64`
/// accumulation, then half-up rounding into `u8`.
fn warp_bilinear(crop: FaceCrop<'_>, inverse: &SimilarityTransform) -> [u8; TEMPLATE_BYTES] {
  let mut out = [0u8; TEMPLATE_BYTES];
  let (width, height) = (crop.width(), crop.height());
  let data = crop.data();

  for v in 0..TEMPLATE_SIZE {
    for u in 0..TEMPLATE_SIZE {
      // `u` and `v` are below 112, so both are exact in `f64`.
      let (uf, vf) = (u as f64, v as f64);
      let fx = inverse.a() * uf - inverse.b() * vf + inverse.tx();
      let fy = inverse.b() * uf + inverse.a() * vf + inverse.ty();
      let mut acc = [0.0f64; 3];
      accumulate_bilinear(data, width, height, fx, fy, &mut acc);
      let base = (v * TEMPLATE_SIZE + u) * 3;
      for (channel, value) in acc.iter().enumerate() {
        out[base + channel] = round_to_u8(*value);
      }
    }
  }
  out
}

/// Adds the four bilinear taps around `(fx, fy)` into `acc`, skipping taps
/// outside the crop (the constant-0 border).
fn accumulate_bilinear(
  data: &[u8],
  width: usize,
  height: usize,
  fx: f64,
  fy: f64,
  acc: &mut [f64; 3],
) {
  // A non-finite mapped coordinate cannot happen for a transform `estimate`
  // accepted, but `floor` on a NaN would produce a nonsense index, so the
  // sampler refuses it and leaves the pixel at the border value.
  if !fx.is_finite() || !fy.is_finite() {
    return;
  }
  let (x0, y0) = (fx.floor(), fy.floor());
  let (ax, ay) = (fx - x0, fy - y0);
  for (dy, wy) in [(0i64, 1.0 - ay), (1, ay)] {
    for (dx, wx) in [(0i64, 1.0 - ax), (1, ax)] {
      let weight = wy * wx;
      if weight == 0.0 {
        continue;
      }
      let Some(index) = tap_index(x0, y0, dx, dy, width, height) else {
        continue;
      };
      for (channel, slot) in acc.iter_mut().enumerate() {
        *slot += weight * f64::from(data[index + channel]);
      }
    }
  }
}

/// The byte offset of the tap at `(x0 + dx, y0 + dy)`, or `None` when it falls
/// outside the crop.
fn tap_index(x0: f64, y0: f64, dx: i64, dy: i64, width: usize, height: usize) -> Option<usize> {
  // `x0`/`y0` are `floor`ed and finite here; anything beyond `i64` is far
  // outside any crop, so the saturating conversion lands out of range and the
  // bounds test below rejects it.
  let (xi, yi) = (x0 as i64 + dx, y0 as i64 + dy);
  let x = usize::try_from(xi).ok()?;
  let y = usize::try_from(yi).ok()?;
  if x >= width || y >= height {
    return None;
  }
  Some((y * width + x) * 3)
}

/// Half-up rounding into `u8`, clamped to `0..=255`.
#[inline]
fn round_to_u8(value: f64) -> u8 {
  // Clamped into `0.0..=255.0` immediately before the narrowing cast.
  (value + 0.5).floor().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests;
