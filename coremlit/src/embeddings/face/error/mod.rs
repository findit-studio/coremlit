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
/// **This is a producer postcondition, and it is the whole reason
/// [`crate::embeddings::face::SimilarityTransform::inverse`] can be one
/// arithmetic path.** A `SimilarityTransform` exists only as a value
/// [`crate::embeddings::face::SimilarityTransform::estimate`] returned after
/// checking that `a² + b²` and its reciprocal are normal, or as the inverse of
/// one. `estimate` raises this, naming the scale, for the finite `f32` inputs
/// whose minimiser has no inverse in `cv2.warpAffine`'s arithmetic. The module
/// once claimed no such input existed and published the argument as proven;
/// review round 5 on #135 exhibited one, and the claim is withdrawn — the band
/// is enforced here rather than argued about `f32`.
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
  /// about `1.5e-154`, or above about `6.7e153`.
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
/// a symlink that cannot be followed, for a root that is neither a directory
/// nor a regular file, and for a tree nested past the walk's depth limit.
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
