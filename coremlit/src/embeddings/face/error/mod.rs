//! Structured errors for [`crate::embeddings::face`].
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants, and
//! every variant is UNIT or NEWTYPE — several fields means a named payload
//! struct, per the workspace house rule
//! (`no_enum_in_the_workspace_has_a_struct_shaped_or_multi_field_variant`).

/// A crop's declared geometry is unusable: a zero axis, an axis past
/// [`crate::embeddings::face::MAX_CROP_AXIS`], or `width · height · 3` not
/// representable.
///
/// Payload of [`Error::CropDimensions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropDimensions {
  /// The declared width in pixels.
  width: usize,
  /// The declared height in pixels.
  height: usize,
}

impl CropDimensions {
  /// Construct from the declared geometry.
  #[inline(always)]
  pub const fn new(width: usize, height: usize) -> Self {
    Self { width, height }
  }

  /// The declared width in pixels.
  #[inline(always)]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// The declared height in pixels.
  #[inline(always)]
  pub const fn height(&self) -> usize {
    self.height
  }
}

/// A crop's backing slice is not exactly `width · height · 3` bytes.
///
/// Payload of [`Error::CropDataLength`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropDataLength {
  /// Bytes the caller supplied.
  got: usize,
  /// Bytes `width · height · 3` requires.
  expected: usize,
}

impl CropDataLength {
  /// Construct from the supplied and required byte counts.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// Bytes the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// Bytes `width · height · 3` requires.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// The five landmarks carry no usable spread, so no similarity transform is
/// determined.
///
/// The least-squares similarity divides by `Σ ‖pᵢ − p̄‖²`; five coincident (or
/// numerically coincident) points make that zero and every downstream
/// coordinate a NaN. Rejecting is the only honest answer — a face whose five
/// landmarks collapse to a point has not been detected.
///
/// Payload of [`Error::DegenerateLandmarks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegenerateLandmarks {
  /// `Σ ‖pᵢ − p̄‖²` over the five supplied landmarks — exactly zero, or
  /// non-finite, when this error is raised.
  ///
  /// Those are the only two values the guard admits, which is why narrowing
  /// the `f64` accumulator to `f32` here cannot lose anything: a positive
  /// SUBNORMAL spread is finite and greater than zero, so it is solved rather
  /// than rejected, and never reaches this field to be flushed to zero by the
  /// narrowing.
  spread: f32,
}

impl DegenerateLandmarks {
  /// Construct from the computed spread.
  #[inline(always)]
  pub const fn new(spread: f32) -> Self {
    Self { spread }
  }

  /// `Σ ‖pᵢ − p̄‖²` over the five supplied landmarks.
  #[inline(always)]
  pub const fn spread(&self) -> f32 {
    self.spread
  }
}

/// Which of [`crate::embeddings::face::SimilarityTransform::estimate`]'s TWO
/// point sets a rejected coordinate came from.
///
/// `estimate` takes a source and a target and is only as total as the weaker
/// of the two checks, so the error has to say which side failed: "landmark 3
/// is NaN" is not actionable when the caller supplied one of the point sets
/// and the crate supplied the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum LandmarkSet {
  /// The detector's landmarks — `estimate`'s `source`, and the argument
  /// [`crate::embeddings::face::FaceAlign::to_template`] takes from its caller.
  Source,
  /// The destination template — `estimate`'s `target`, which
  /// [`crate::embeddings::face::FaceAlign::to_template`] fills in with
  /// [`crate::embeddings::face::ARCFACE_TEMPLATE`].
  Target,
}

impl LandmarkSet {
  /// Stable name, as it appears in the error message.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Source => "source",
      Self::Target => "target",
    }
  }
}

/// A landmark coordinate is NaN or infinite, in one named point set.
///
/// Separated from [`DegenerateLandmarks`] because the causes differ: a
/// non-finite coordinate is a broken detector, a zero spread is a detection
/// that collapsed.
///
/// Payload of [`Error::NonFiniteLandmark`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteLandmark {
  /// Which point set the offending coordinate came from.
  set: LandmarkSet,
  /// Index of the offending landmark, `0..5` in template order.
  index: usize,
}

impl NonFiniteLandmark {
  /// Construct from the offending landmark's point set and index.
  #[inline(always)]
  pub const fn new(set: LandmarkSet, index: usize) -> Self {
    Self { set, index }
  }

  /// Which point set the offending coordinate came from.
  #[inline(always)]
  pub const fn set(&self) -> LandmarkSet {
    self.set
  }

  /// Index of the offending landmark, `0..5` in template order.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }
}

/// One of a [`crate::embeddings::face::SimilarityTransform`]'s four free
/// parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum TransformParameter {
  /// `s·cos θ`.
  A,
  /// `s·sin θ`.
  B,
  /// Horizontal translation.
  Tx,
  /// Vertical translation.
  Ty,
}

impl TransformParameter {
  /// Stable name, as it appears in the error message.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::A => "a",
      Self::B => "b",
      Self::Tx => "tx",
      Self::Ty => "ty",
    }
  }
}

/// A SOLVED similarity transform has a non-finite parameter.
///
/// The backstop that keeps a `SimilarityTransform` VALUE total: every way of
/// producing one checks its four parameters before handing it out, so no
/// caller can receive an `Ok` holding a transform whose `apply` returns NaN.
///
/// **No route reaches it, and the doc says so rather than naming one.**
/// [`crate::embeddings::face::SimilarityTransform::estimate`] is its only
/// producer, and both point sets are finite by the time the solve runs, which
/// bounds the four parameters well inside `f64` by Cauchy–Schwarz — see that
/// method for the bound. The 860 810-set measurement counted this variant zero
/// times. [`crate::embeddings::face::SimilarityTransform::inverse`] reports a
/// transform it cannot invert as `None` and raises nothing; a solved transform
/// that has no inverse is refused at the producer instead, as
/// [`Error::NonInvertibleTransform`]. This variant is kept because the bound is
/// an argument about the input type rather than something the compiler
/// enforces.
///
/// Its sibling's bound was an argument too, and it was WRONG — see
/// [`NonInvertibleTransform`]. The difference is that this one is
/// Cauchy–Schwarz, an inequality on the numerator, where that one reasoned
/// about how small a nonzero numerator can be.
///
/// Payload of [`Error::NonFiniteTransform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteTransform {
  /// The first parameter found to be NaN or infinite.
  parameter: TransformParameter,
}

impl NonFiniteTransform {
  /// Construct from the offending parameter.
  #[inline(always)]
  pub const fn new(parameter: TransformParameter) -> Self {
    Self { parameter }
  }

  /// The first parameter found to be NaN or infinite.
  #[inline(always)]
  pub const fn parameter(&self) -> TransformParameter {
    self.parameter
  }
}

/// A solved similarity transform has no inverse, so no template pixel can be
/// mapped back into the crop and nothing can be sampled through it.
///
/// **This is a producer postcondition over the CLOSURE, and it is the whole
/// reason [`crate::embeddings::face::SimilarityTransform::inverse`] can be one
/// arithmetic path.** A `SimilarityTransform` exists only as a value
/// [`crate::embeddings::face::SimilarityTransform::estimate`] returned or as
/// the inverse of one, and both doors check that `a² + b²` and its reciprocal
/// are normal — `inverse` on the value it CONSTRUCTS, `estimate` on its solve
/// and then on that solve's inverse. `estimate` raises this, naming the scale,
/// for the finite `f32` inputs whose minimiser has no inverse in
/// `cv2.warpAffine`'s arithmetic, and for those whose inverse has none in
/// turn.
///
/// The module twice claimed no such input existed and published the argument
/// as proven. Review round 5 on #135 exhibited a minimiser whose determinant
/// underflows to zero; round 6 exhibited one whose determinant is `0x1p-1022`
/// exactly — admitted by the band — and whose INVERSE's determinant lands one
/// ulp outside it. Both claims are withdrawn: the band is enforced at both
/// producers rather than argued about `f32`, and the module doc states where
/// the resulting guarantee stops.
///
/// **Deliberately not [`DegenerateLandmarks`].** That one means the SOURCE
/// points carried no spread, and
/// [`crate::embeddings::face::SimilarityTransform::estimate`] raises it only
/// where the spread really is zero. Anything reaching THIS error has already
/// passed that guard, so its landmarks are spread — what vanished is the
/// solved SCALE, which the source spread does not determine on its own (the
/// scale is `|Σ conj(uᵢ)·vᵢ| / Σ‖uᵢ‖²` over the two CENTRED sets, so the
/// target and the relative geometry decide it too). Reporting a zero landmark
/// spread here would send a reader hunting for coincident landmarks that do
/// not exist.
///
/// Payload of [`Error::NonInvertibleTransform`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonInvertibleTransform {
  /// The solved transform's uniform scale, `√(a² + b²)`.
  ///
  /// Zero when the solve collapsed the plane onto a point, which a target with
  /// no spread reaches. Otherwise a scale outside the band
  /// [`crate::embeddings::face::SimilarityTransform::inverse`] runs in — below
  /// about `1.5e-154`, or above about `6.7e153` — or one sitting so close to
  /// an edge of it that the INVERSE's scale falls outside, which is the
  /// round-6 case and where the payload's `f64` width earns itself twice over
  /// (that witness's scale is `2^-511`, and the two failing regimes are a
  /// handful of ulps apart).
  ///
  /// **`f64` rather than `f32`, and a public producer DOES reach the width
  /// that needs.** The whole lower half of that band is a range `f32` flushes
  /// to zero, and reporting zero is what this payload exists to stop doing:
  /// the round-5 witness on #135 is five finite `f32` landmarks whose scale is
  /// `6.1e-168`, which `f32` cannot tell from the collapsed case and which a
  /// reader has to be able to tell from it. When this doc said no producer
  /// reached that half, the width was a property of the type; it is now a
  /// property of a witness anyone can hand in.
  scale: f64,
}

impl NonInvertibleTransform {
  /// Construct from the solved transform's scale.
  #[inline(always)]
  pub const fn new(scale: f64) -> Self {
    Self { scale }
  }

  /// The solved transform's uniform scale, `√(a² + b²)`.
  #[inline(always)]
  pub const fn scale(&self) -> f64 {
    self.scale
  }
}

/// Which of the two source coordinates left OpenCV's `int` fixed-point domain.
///
/// Named after the variables `imgwarp.cpp`'s `WarpAffineInvoker` uses, so a
/// reader can put the failure on a line of the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum CoordinateAxis {
  /// The horizontal source coordinate — OpenCV's `X`, built from `M[0..3]`.
  X,
  /// The vertical source coordinate — OpenCV's `Y`, built from `M[3..6]`.
  Y,
}

impl CoordinateAxis {
  /// Stable name, as it appears in the error message.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::X => "horizontal",
      Self::Y => "vertical",
    }
  }
}

/// Which TERM of the split fixed-point coordinate left `int`.
///
/// `cv2.warpAffine` never forms the source coordinate in one expression: the
/// per-column half and the per-row half are each rounded to `1/1024` of a
/// pixel on their own and only then added. Each of the three is a separate
/// `int`, so each is a separate place the coordinate can leave the domain —
/// and naming which one is what distinguishes a transform whose columns run
/// away from one whose rows do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum CoordinateTerm {
  /// OpenCV's `adelta[x]` / `bdelta[x]`: the per-destination-COLUMN half.
  ColumnDelta,
  /// OpenCV's `X0` / `Y0`: the per-destination-ROW half, `round_delta`
  /// included.
  RowOrigin,
  /// OpenCV's `X` / `Y`: the two halves added into one `int`.
  ///
  /// **The arm that made this error necessary.** Both halves can be inside
  /// `int` — or be forced into it by a clamp — while their SUM is not, and a
  /// clamped pair can CANCEL: `i32::MIN + 16` plus `i32::MAX` is 15, a small,
  /// perfectly ordinary-looking coordinate that samples the crop's first pixel
  /// where the true source is 1.9 billion pixels away.
  Sum,
}

impl CoordinateTerm {
  /// Stable name, as it appears in the error message.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::ColumnDelta => "per-column term",
      Self::RowOrigin => "per-row term",
      Self::Sum => "summed term",
    }
  }
}

/// A destination → source coordinate leaves the `int` fixed-point domain
/// `cv2.warpAffine` computes it in, so no template pixel can be sampled
/// through the transform.
///
/// **Raised BEFORE any sampling.** The whole coordinate map is built and
/// checked up front, so this is a refusal rather than a partially warped
/// template: a transform that puts one destination pixel outside the domain
/// puts the operation outside its own definition.
///
/// Reaching it needs a transform whose inverse is enormous, which needs
/// landmarks whose solved scale is nearly zero — a detector emitting a
/// near-degenerate five-point set, finite and in-bounds and past every other
/// guard. That is exactly the input that used to produce a plausible corrupted
/// face.
///
/// Payload of [`Error::CoordinateOverflow`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordinateOverflow {
  /// Which source coordinate the offending term belongs to.
  axis: CoordinateAxis,
  /// Which of the three split terms left `int`.
  term: CoordinateTerm,
  /// The offending value, in units of `1/1024` of a source pixel.
  ///
  /// `f64` because two of the three terms are read before they are rounded.
  /// The third — [`CoordinateTerm::Sum`] — is an integer of magnitude at most
  /// 2³², so widening it to `f64` is exact and the field carries one kind of
  /// number rather than two.
  value: f64,
}

impl CoordinateOverflow {
  /// Construct from the offending axis, term and value.
  #[inline(always)]
  pub const fn new(axis: CoordinateAxis, term: CoordinateTerm, value: f64) -> Self {
    Self { axis, term, value }
  }

  /// Which source coordinate the offending term belongs to.
  #[inline(always)]
  pub const fn axis(&self) -> CoordinateAxis {
    self.axis
  }

  /// Which of the three split terms left `int`.
  #[inline(always)]
  pub const fn term(&self) -> CoordinateTerm {
    self.term
  }

  /// The offending value, in units of `1/1024` of a source pixel.
  #[inline(always)]
  pub const fn value(&self) -> f64 {
    self.value
  }
}

/// A loaded model's input or output feature does not match the contract the
/// caller's [`crate::embeddings::face::FaceModel`] manifest declares.
///
/// Payload of [`Error::ContractMismatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractMismatch {
  /// Name of the input/output feature that mismatched.
  feature: String,
  /// The contract the manifest declares, rendered for display.
  expected: String,
  /// What the loaded model actually declares, rendered for display.
  actual: String,
}

impl ContractMismatch {
  /// Construct from the mismatched feature, the manifest's contract, and what
  /// the loaded model declares.
  #[inline(always)]
  pub const fn new(feature: String, expected: String, actual: String) -> Self {
    Self {
      feature,
      expected,
      actual,
    }
  }

  /// Name of the input/output feature that mismatched.
  #[inline(always)]
  pub fn feature(&self) -> &str {
    &self.feature
  }

  /// The contract the manifest declares, rendered for display.
  #[inline(always)]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the loaded model actually declares, rendered for display.
  #[inline(always)]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// The predicted embedding tensor's shape diverges from the contract resolved
/// at load.
///
/// Payload of [`Error::OutputShape`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputShape {
  /// What the model produced.
  got: Vec<usize>,
  /// What the resolved contract requires.
  expected: Vec<usize>,
}

impl OutputShape {
  /// Construct from the produced and required shapes.
  #[inline(always)]
  pub const fn new(got: Vec<usize>, expected: Vec<usize>) -> Self {
    Self { got, expected }
  }

  /// What the model produced.
  #[inline(always)]
  pub fn got(&self) -> &[usize] {
    &self.got
  }

  /// What the resolved contract requires.
  #[inline(always)]
  pub fn expected(&self) -> &[usize] {
    &self.expected
  }
}

/// The predicted tensor holds a different NUMBER of elements than the resolved
/// contract, with axes that matched.
///
/// **Deliberately not [`OutputShape`].** The element count is CoreML's own
/// answer rather than a product of the cached shape, which is why it is
/// checked alongside the axes at all — so the two can disagree, and when only
/// the count does, the axes were right. `OutputShape` would then have to put
/// the same vector in both of its fields and report a shape mismatch that did
/// not happen.
///
/// Payload of [`Error::OutputElementCount`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputElementCount {
  /// Elements the model's tensor reports.
  got: usize,
  /// Elements the resolved contract requires.
  expected: usize,
}

impl OutputElementCount {
  /// Construct from the reported and required element counts.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// Elements the model's tensor reports.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// Elements the resolved contract requires.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// A per-batch-row failure, carrying which row of the batch it came from.
///
/// A batch call fails as a whole, but the caller still needs to know WHICH
/// face was the problem — with `N` faces in one call, "the embedding has zero
/// magnitude" without a row index is not actionable.
///
/// Payload of [`Error::EmbeddingZero`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRow {
  /// Index into the `faces` slice the caller passed.
  row: usize,
}

impl BatchRow {
  /// Construct from the offending row's index into the caller's slice.
  #[inline(always)]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Index into the `faces` slice the caller passed.
  #[inline(always)]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// A non-finite component in one row of the model's output.
///
/// Payload of [`Error::NonFiniteOutput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteOutput {
  /// Index into the `faces` slice the caller passed.
  row: usize,
  /// Component index within that row's embedding.
  component: usize,
}

impl NonFiniteOutput {
  /// Construct from the offending row and component.
  #[inline(always)]
  pub const fn new(row: usize, component: usize) -> Self {
    Self { row, component }
  }

  /// Index into the `faces` slice the caller passed.
  #[inline(always)]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Component index within that row's embedding.
  #[inline(always)]
  pub const fn component(&self) -> usize {
    self.component
  }
}

/// Which part of two [`crate::embeddings::face::EmbeddingSpace`]s first
/// disagrees.
///
/// Declared in the order [`crate::embeddings::face::FaceEmbedding::dot`]
/// reports them, so a pair that differs in several places at once names the
/// first of these rather than an arbitrary one.
///
/// **Every variant here is part of the FUNCTION that produced the vector**, and
/// that is the whole membership rule. [`Self::Artifact`] is the trained
/// parameters, which are most of that function; the two feature names say
/// which tensor was written and which was read; `dim` is the artifact's output
/// width; the rest are the pixels-to-tensor map the host applied before
/// inference. Change any of them and the numbers change, so a cosine across
/// the difference is undefined rather than merely inaccurate.
///
/// **The feature names were removed from this list once, as "IO routing", and
/// are back.** The argument was that renaming a graph's features re-exports
/// the same weights and produces the same numbers, so the name is not
/// evidence. It is not evidence *about the weights* — that is what
/// [`Self::Artifact`] is for — but for a model with two `[batch, dim]` heads
/// the OUTPUT name selects which function produced the numbers, and the
/// re-export the old argument worried about is now caught by the digest
/// instead, where it belongs. The two claims that motivated the removal are
/// both answered by adding the artifact rather than by dropping the names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum EmbeddingSpaceField {
  /// The two embeddings came out of artifacts whose BYTES differ — a
  /// fine-tune, a requantisation, or an unrelated export of the same width.
  ///
  /// First, because when this differs everything else is coincidence: the
  /// trained parameters are most of the function, so naming a matching width
  /// or a matching divisor would be true and misleading about the cause.
  Artifact,
  /// The pixels were written to differently named input features.
  InputFeature,
  /// The embeddings were read from differently named output features — which
  /// for a multi-head graph means two different functions of one artifact.
  OutputFeature,
  /// The embeddings are different widths — the one case that was previously
  /// reported as a cosine of `0.0`, a value a measured non-match also has.
  Dim,
  /// One model wants RGB and the other BGR.
  ChannelOrder,
  /// One model wants NCHW and the other NHWC.
  TensorLayout,
  /// The per-byte multiplier differs.
  PreprocessingScale,
  /// The per-channel offset differs.
  PreprocessingBias,
}

impl EmbeddingSpaceField {
  /// Stable name, as it appears in the error message.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Artifact => "artifact digest",
      Self::InputFeature => "input feature name",
      Self::OutputFeature => "output feature name",
      Self::Dim => "dim",
      Self::ChannelOrder => "preprocessing channel order",
      Self::TensorLayout => "preprocessing tensor layout",
      Self::PreprocessingScale => "preprocessing scale",
      Self::PreprocessingBias => "preprocessing bias",
    }
  }
}

/// Two [`crate::embeddings::face::FaceEmbedding`]s were compared across
/// different model or preprocessing spaces.
///
/// **This is a refusal, not a score.** The widths agreeing is not enough: two
/// 512-wide ArcFace-family artifacts, one graph's two `[batch, dim]` heads, or
/// one artifact fed BGR and RGB, put their vectors in unrelated spaces, and
/// their dot product lands in `[−1, 1]` looking exactly like a measurement.
/// The old return of `0.0` for a width mismatch had the same defect in its
/// narrow case — `0.0` is a legitimate cosine, so no caller could tell an
/// incompatible migration from a face that simply did not match.
///
/// Payload of [`Error::IncomparableEmbeddings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncomparableEmbeddings {
  /// The first space field found to differ.
  field: EmbeddingSpaceField,
}

impl IncomparableEmbeddings {
  /// Construct from the first space field found to differ.
  #[inline(always)]
  pub const fn new(field: EmbeddingSpaceField) -> Self {
    Self { field }
  }

  /// The first space field found to differ.
  ///
  /// The FIRST, not the only: two spaces can disagree in several places at
  /// once, and reporting one that is definitely wrong beats a list assembled
  /// to look thorough. When the artifacts differ that is what is named, and
  /// everything after it is coincidence.
  #[inline(always)]
  pub const fn field(&self) -> EmbeddingSpaceField {
    self.field
  }
}

/// The bytes of a compiled model artifact could not be read, so no
/// [`crate::embeddings::face::ArtifactDigest`] exists for it.
///
/// **Fails closed, and that is the whole point of the variant.** The digest is
/// what binds an embedding to the weights it came out of; a load that could
/// not read some of the bundle has no identity to stamp, and a partial digest
/// would be an identity for bytes nobody has. So a load that reaches this
/// fails rather than producing vectors whose space is a guess.
///
/// Raised for an unreadable directory or file anywhere under the artifact, for
/// a file symlink that cannot be followed, for a symlink to a DIRECTORY (which
/// is refused rather than walked, so the walk cannot become a walk of a
/// graph), for a root that is neither a directory nor a regular file, and for
/// a tree past either of the walk's two resource caps — its depth and its
/// total entry count.
///
/// Payload of [`Error::ArtifactDigest`].
///
/// Like its siblings elsewhere in the workspace this one derives
/// [`std::error::Error`] and owns both the variant's message and its
/// `#[source]`: the variant is `#[error(transparent)]`, which forwards
/// `Display` *and* `source()` straight through rather than inserting a link.
///
/// The inherent [`source`](Self::source) getter returns the concrete
/// `&`[`std::io::Error`], and so shadows [`std::error::Error::source`] for
/// method-call syntax; call the trait method by path
/// (`std::error::Error::source(&e)`) to walk the chain.
#[derive(Debug, thiserror::Error)]
#[error("failed to hash the model artifact at `{path}`: {source}")]
pub struct DigestFailure {
  /// The path that could not be read — the artifact root, or the entry under
  /// it that failed.
  path: std::path::PathBuf,
  /// The underlying I/O failure.
  #[source]
  source: std::io::Error,
}

impl DigestFailure {
  /// Construct from the path that could not be read and the underlying I/O
  /// failure.
  #[inline(always)]
  pub const fn new(path: std::path::PathBuf, source: std::io::Error) -> Self {
    Self { path, source }
  }

  /// The path that could not be read.
  #[inline(always)]
  pub fn path(&self) -> &std::path::Path {
    &self.path
  }

  /// The underlying I/O failure.
  #[inline(always)]
  pub const fn source(&self) -> &std::io::Error {
    &self.source
  }
}

/// What about a [`crate::embeddings::face::Preprocessing`] does not stay in
/// `f32` — a field, or the MAP the two fields make.
///
/// The third variant is the one that is not a field, and it exists because the
/// first two were an enumeration of what can go wrong that missed the thing
/// the fields are for: `scale = f32::MAX` with `bias = 0` is two perfectly
/// finite numbers whose map writes `+inf` for every byte from 2 upwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
pub enum PreprocessingField {
  /// [`crate::embeddings::face::Preprocessing::scale`], the per-byte
  /// multiplier.
  #[display("scale")]
  Scale,
  /// [`crate::embeddings::face::Preprocessing::bias`], at this channel index
  /// in the MODEL's own channel order — so a BGR manifest's `bias[0]` is its
  /// blue offset.
  #[display("bias[{_0}]")]
  Bias(usize),
  /// Both fields are finite and the MAP they make is not — see
  /// [`PreprocessingMap`] for the channel and the endpoint it names.
  #[display("{_0}")]
  Map(PreprocessingMap),
}

/// Where a [`crate::embeddings::face::Preprocessing`]'s map leaves `f32`: the
/// end of the byte range, and the channel whose bias carries it there.
///
/// `byte ↦ byte · scale + bias[channel]` is affine in `byte`, so its extremes
/// over `0..=255` are at the two endpoints — which is why naming one of them
/// names the whole failure rather than one sampled byte. Only the far endpoint
/// can be named in practice: `byte 0 · scale + bias` is exactly `bias` for any
/// finite `scale`, so once the two field checks have passed the near endpoint
/// is finite by construction, and it is evaluated because the PAIR is what
/// proves the 254 bytes between them.
///
/// Payload of [`PreprocessingField::Map`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("byte {byte} · scale + bias[{channel}]")]
pub struct PreprocessingMap {
  /// The channel whose bias carries the map out of `f32`, in the MODEL's own
  /// channel order — as [`PreprocessingField::Bias`]'s index is, so a BGR
  /// manifest's channel `0` is its blue one.
  channel: usize,
  /// The endpoint of the byte range at which it does: `0` or `255`.
  byte: u8,
}

impl PreprocessingMap {
  /// Construct from the channel and the byte-range endpoint.
  #[inline(always)]
  pub const fn new(channel: usize, byte: u8) -> Self {
    Self { channel, byte }
  }

  /// The channel whose bias carries the map out of `f32`, in the MODEL's own
  /// channel order.
  #[inline(always)]
  pub const fn channel(&self) -> usize {
    self.channel
  }

  /// The endpoint of the byte range at which the map leaves `f32`.
  #[inline(always)]
  pub const fn byte(&self) -> u8 {
    self.byte
  }
}

/// A manifest's preprocessing does not stay in `f32`: a NaN or infinite
/// `scale` or `bias`, or an affine map that leaves `f32` at an end of the byte
/// range.
///
/// **Refused at load, which is what keeps a NaN out of a stamped
/// [`crate::embeddings::face::EmbeddingSpace`].** `value = byte · scale +
/// bias` with a non-finite parameter makes every element of the input tensor
/// non-finite, so the model can only produce garbage or
/// [`Error::NonFiniteOutput`]; and the manifest is copied verbatim into the
/// space every vector carries, where a NaN would then have to be given an
/// equality of its own to stay comparable with itself.
///
/// **The two fields being finite is not the same claim, and the load used to
/// make only that one.** `scale = f32::MAX` and `bias = 0` are both finite and
/// write `+inf` for every byte from 2 upwards. The map is AFFINE in `byte`, so
/// its extremes over `0..=255` sit at the endpoints; the load evaluates the
/// exact expression the writer uses at byte `0` and byte `255` for each
/// channel, which proves the 254 bytes between them rather than sampling them.
/// [`PreprocessingField::Map`] is what that refusal names.
///
/// [`crate::embeddings::face::Preprocessing`] is public and its constructors
/// are `const`, so such a value can be BUILT — that is why
/// `Preprocessing`'s own `Eq` still folds NaN onto one representative. What
/// this variant removes is the road from there to an embedder.
///
/// Payload of [`Error::NonFinitePreprocessing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFinitePreprocessing {
  /// The first field found to be NaN or infinite, `scale` before `bias`.
  field: PreprocessingField,
}

impl NonFinitePreprocessing {
  /// Construct from the offending field.
  #[inline(always)]
  pub const fn new(field: PreprocessingField) -> Self {
    Self { field }
  }

  /// The first field found to be NaN or infinite, `scale` before `bias`.
  #[inline(always)]
  pub const fn field(&self) -> PreprocessingField {
    self.field
  }
}

/// One digest as lowercase hex, for an error message.
///
/// Cold path: two of these are formatted only when a load is being refused.
fn hex(digest: crate::embeddings::face::ArtifactDigest) -> String {
  digest
    .as_bytes()
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect()
}

/// The artifact's bytes changed between the digest taken before the load and
/// the one taken after, so which bytes CoreML actually read is not known.
///
/// **A digest and a load are two separate walks of a path the crate does not
/// own.** This crate is sans-I/O about the artifact: it takes a path, and
/// anything may replace what that path names while the load is in flight. A
/// bundle swapped in that window used to be loaded as A and STAMPED as B, and
/// every vector it produced then carried an identity belonging to weights that
/// never ran — the exact confusion
/// [`crate::embeddings::face::EmbeddingSpace`] exists to prevent, arriving
/// through the mechanism meant to prevent it.
/// [`crate::embeddings::face::FaceEmbedder::load`] therefore hashes, loads, and
/// hashes again, and raises this rather than choosing one of the two answers.
///
/// # The residual, stated exactly
///
/// This detects any SINGLE replacement inside the window. It does not detect an
/// **A→B→A** replacement completed inside one `load` — B's bytes are read and
/// A's digest is stamped twice — and that is not closed. Closing it needs a
/// private immutable snapshot of the bundle, which costs a full copy per load
/// and adds a new failure surface (disk space, a second path, its own cleanup)
/// for a race that requires two swaps inside roughly 150 ms with no adversary
/// in the threat model: a caller who controls the filesystem can already load
/// whatever bytes they choose, and the digest exists against *confusion*, not
/// against them. The snapshot was considered and declined for that reason.
///
/// Payload of [`Error::ArtifactChangedDuringLoad`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactChangedDuringLoad {
  /// The digest taken before the artifact was opened.
  before: crate::embeddings::face::ArtifactDigest,
  /// The digest taken after the load had read it.
  after: crate::embeddings::face::ArtifactDigest,
}

impl ArtifactChangedDuringLoad {
  /// Construct from the digests taken before and after the load.
  #[inline(always)]
  pub const fn new(
    before: crate::embeddings::face::ArtifactDigest,
    after: crate::embeddings::face::ArtifactDigest,
  ) -> Self {
    Self { before, after }
  }

  /// The digest taken before the artifact was opened.
  #[inline(always)]
  pub const fn before(&self) -> crate::embeddings::face::ArtifactDigest {
    self.before
  }

  /// The digest taken after the load had read it.
  #[inline(always)]
  pub const fn after(&self) -> crate::embeddings::face::ArtifactDigest {
    self.after
  }
}

/// Which of the two tensors one prediction allocates.
///
/// They are sized by different things, so a size failure names which one it is
/// about: the input's element count is `batch · 112 · 112 · 3` and is the
/// ARTIFACT's alone, while the output's is `batch · dim` and pairs the
/// artifact's batch with the manifest's declared width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
pub enum PredictionTensor {
  /// The `[batch, 3, 112, 112]` (or NHWC) tensor the preprocessed pixels are
  /// written into.
  #[display("input")]
  Input,
  /// The `[batch, dim]` tensor the embeddings are read out of.
  #[display("output")]
  Output,
}

/// One prediction tensor whose element count `batch · per_row` does not fit
/// `usize`, so no buffer for it can even be described.
///
/// # Why this is a load-time refusal and not a cap
///
/// The batch is the ARTIFACT's: this door's input contract states the batch
/// axis as "any one fixed size" and reads the number back off the checked
/// model, so it is a value neither this crate nor its caller chose, and nothing
/// in a `.mlmodelc` bounds it. Multiplied with `*`, `batch · per_row` wraps
/// silently in a release build; the too-short buffer that follows then panics
/// when a row is sliced out of it, which turns a model the door ACCEPTED into a
/// terminated caller.
///
/// A cap would be an enumeration of how big is too big. `checked_mul` is a
/// proof instead: the product either fits `usize` or it does not, and the
/// second case is this error.
///
/// Payload of [`Error::ElementCountOverflow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementCountOverflow {
  /// Which tensor's element count overflowed.
  tensor: PredictionTensor,
  /// The batch the artifact declares.
  batch: usize,
  /// Elements one row of that tensor holds: `112 · 112 · 3` for the input, the
  /// manifest's embedding width for the output.
  per_row: usize,
}

impl ElementCountOverflow {
  /// Construct from the tensor, the artifact's batch and that tensor's
  /// per-row element count.
  #[inline(always)]
  pub const fn new(tensor: PredictionTensor, batch: usize, per_row: usize) -> Self {
    Self {
      tensor,
      batch,
      per_row,
    }
  }

  /// Which tensor's element count overflowed.
  #[inline(always)]
  pub const fn tensor(&self) -> PredictionTensor {
    self.tensor
  }

  /// The batch the artifact declares.
  #[inline(always)]
  pub const fn batch(&self) -> usize {
    self.batch
  }

  /// Elements one row of that tensor holds.
  #[inline(always)]
  pub const fn per_row(&self) -> usize {
    self.per_row
  }
}

/// A prediction tensor whose element count fits `usize` and whose buffer the
/// allocator would not give.
///
/// The count fitting `usize` is what [`ElementCountOverflow`] establishes and
/// is a strictly weaker fact than the memory existing: a batch of `2⁵⁵` counts
/// fine and asks for petabytes. `vec![0.0; n]` answers that by aborting the
/// process, which is not something a caller can handle; the buffers this door
/// sizes from an artifact are reserved with `Vec::try_reserve_exact` instead,
/// and this payload is what the refusal carries.
///
/// Payload of [`Error::AllocationFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationFailed {
  /// Which tensor could not be allocated.
  tensor: PredictionTensor,
  /// The `f32` element count that was asked for.
  elements: usize,
}

impl AllocationFailed {
  /// Construct from the tensor and the element count that was refused.
  #[inline(always)]
  pub const fn new(tensor: PredictionTensor, elements: usize) -> Self {
    Self { tensor, elements }
  }

  /// Which tensor could not be allocated.
  #[inline(always)]
  pub const fn tensor(&self) -> PredictionTensor {
    self.tensor
  }

  /// The `f32` element count that was asked for.
  #[inline(always)]
  pub const fn elements(&self) -> usize {
    self.elements
  }
}

/// Everything [`crate::embeddings::face`] can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] crate::LoadError),
  /// A CoreML prediction failed.
  #[error("prediction failed: {0}")]
  Prediction(#[from] crate::PredictionError),
  /// Building or reading a CoreML tensor failed.
  #[error("tensor failed: {0}")]
  Tensor(#[from] crate::TensorError),
  /// The crop's declared geometry is unusable.
  #[error(
    "crop dimensions {}×{} are unusable (zero axis, an axis past {}, or w·h·3 overflows)",
    .0.width(),
    .0.height(),
    crate::embeddings::face::MAX_CROP_AXIS
  )]
  CropDimensions(CropDimensions),
  /// The crop's backing slice is not `width · height · 3` bytes.
  #[error("crop data length mismatch: expected {} bytes (w·h·3), got {}", .0.expected(), .0.got())]
  CropDataLength(CropDataLength),
  /// A landmark coordinate is NaN or infinite.
  #[error("{} landmark {} has a non-finite coordinate", .0.set(), .0.index())]
  NonFiniteLandmark(NonFiniteLandmark),
  /// A solved transform has a non-finite parameter.
  #[error("the solved similarity transform has a non-finite `{}`", .0.parameter())]
  NonFiniteTransform(NonFiniteTransform),
  /// A solved transform has no inverse, so the template cannot be sampled.
  #[error(
    "the solved similarity transform has no inverse (scale = {:e}); no template pixel can be \
     mapped back into the crop",
    .0.scale()
  )]
  NonInvertibleTransform(NonInvertibleTransform),
  /// The five landmarks carry no usable spread.
  #[error(
    "the five landmarks have no usable spread (Σ‖pᵢ−p̄‖² = {}); no similarity transform is \
     determined",
    .0.spread()
  )]
  DegenerateLandmarks(DegenerateLandmarks),
  /// A destination → source coordinate leaves OpenCV's `int` fixed-point
  /// domain, so the warp is refused before anything is sampled.
  #[error(
    "the {} source coordinate's {} is {:e} (units of 1/1024 px), outside the `int` domain \
     `cv2.warpAffine` computes it in; no template pixel can be sampled through this transform",
    .0.axis(),
    .0.term(),
    .0.value()
  )]
  CoordinateOverflow(CoordinateOverflow),
  /// The compiled artifact's bytes could not be hashed, so the embedder has no
  /// identity to stamp on the vectors it would produce.
  ///
  /// The message and the `source` live on [`DigestFailure`], which this variant
  /// forwards both of through `#[error(transparent)]`.
  #[error(transparent)]
  ArtifactDigest(#[from] DigestFailure),
  /// The artifact's bytes changed while it was being loaded, so the identity
  /// to stamp on its vectors is not known.
  #[error(
    "the model artifact changed while it was being loaded: it hashed to {} before and {} after, \
     so which bytes CoreML read is not known and no identity can be stamped on the vectors it \
     would produce",
    hex(.0.before()),
    hex(.0.after())
  )]
  ArtifactChangedDuringLoad(ArtifactChangedDuringLoad),
  /// The manifest's preprocessing does not stay in `f32`: a NaN or infinite
  /// scale or bias, or a map that leaves `f32` at an end of the byte range.
  #[error(
    "the manifest's preprocessing `{}` does not stay in `f32`; values it writes into the input \
     tensor would be non-finite, and the space stamped on the embeddings would carry it",
    .0.field()
  )]
  NonFinitePreprocessing(NonFinitePreprocessing),
  /// The loaded model does not match the manifest's declared contract.
  #[error("model contract mismatch on `{}`: expected {}, got {}", .0.feature(), .0.expected(), .0.actual())]
  ContractMismatch(ContractMismatch),
  /// The loaded graph declares a REQUIRED input beyond the manifest's, so every
  /// prediction through it would fail.
  ///
  /// Carries the offending feature name. An OPTIONAL extra input is not this:
  /// CoreML runs a prediction that omits one, so only a required input the
  /// embedder cannot fill makes the contract unsatisfiable.
  #[error(
    "model declares a required input `{0}` that this door never supplies; it sends the \
     manifest's input feature and nothing else, so every prediction would fail"
  )]
  UnsatisfiableInput(String),
  /// The loaded graph declares CoreML STATE buffers, and this door predicts
  /// through the stateless API.
  ///
  /// Carries the offending state feature name. A stateful model must receive an
  /// `MLState` on every prediction; a door that never makes one either fails the
  /// prediction outright or silently discards the persistence the graph was
  /// built around. Neither is something to discover at predict time.
  #[error(
    "model declares the state buffer `{0}`, and this door predicts through the stateless \
     API; a stateful graph needs an `MLState` on every prediction"
  )]
  UnsatisfiableState(String),
  /// The predicted embedding tensor's shape diverges from the resolved contract.
  #[error("output shape mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  OutputShape(OutputShape),
  /// The predicted embedding tensor holds the wrong number of elements, with
  /// axes that matched.
  #[error("output element count mismatch: expected {}, got {}", .0.expected(), .0.got())]
  OutputElementCount(OutputElementCount),
  /// The model produced a NaN or infinite embedding component.
  #[error("model output row {} contains a non-finite value at component {}", .0.row(), .0.component())]
  NonFiniteOutput(NonFiniteOutput),
  /// A (finite) embedding row has zero magnitude and cannot be L2-normalized.
  #[error("model output row {} has zero magnitude and cannot be normalized", .0.row())]
  EmbeddingZero(BatchRow),
  /// The artifact's declared batch makes one of this door's tensors
  /// uncountable: `batch · per_row` does not fit `usize`.
  #[error(
    "the graph's batch of {} makes the {} tensor {} · {} elements long, which does not fit \
     `usize`; no buffer for it can be described, let alone filled",
    .0.batch(), .0.tensor(), .0.batch(), .0.per_row()
  )]
  ElementCountOverflow(ElementCountOverflow),
  /// A prediction tensor whose element count fits `usize` could not be
  /// allocated.
  #[error(
    "could not allocate the {} tensor's {} `f32` elements",
    .0.tensor(), .0.elements()
  )]
  AllocationFailed(AllocationFailed),
  /// Two embeddings come from different artifact, routing or preprocessing
  /// spaces.
  #[error(
    "these two embeddings come from different spaces (their {} differs); no similarity between \
     them is defined",
    .0.field()
  )]
  IncomparableEmbeddings(IncomparableEmbeddings),
}

/// `Result` specialized to this module's [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
