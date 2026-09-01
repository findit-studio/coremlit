//! Structured errors for [`crate::embeddings::face`].
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants, and
//! every variant is UNIT or NEWTYPE — several fields means a named payload
//! struct, per the workspace house rule
//! (`no_enum_in_the_workspace_has_a_struct_shaped_or_multi_field_variant`).

/// A crop's declared geometry is unusable: a zero axis, or `width · height · 3`
/// not representable.
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
/// **No public route reaches it, and the doc says so rather than naming one.**
/// [`crate::embeddings::face::SimilarityTransform::estimate`] is its only
/// producer, and both point sets are finite by the time the solve runs, which
/// bounds the four parameters well inside `f64` — see that function's doc for
/// the range argument. [`crate::embeddings::face::SimilarityTransform::inverse`]
/// takes a caller-built transform and CAN meet a non-finite parameter, but it
/// reports that as `None` and raises nothing; where
/// [`crate::embeddings::face::FaceAlign::to_template`] has to turn that `None`
/// into an error, the one it raises is [`Error::NonInvertibleTransform`].
/// This variant is kept because the bound is an argument about the input type
/// rather than something the compiler enforces.
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
  /// Zero when the solve collapsed the plane onto a point; nonzero but tiny
  /// when it is only the INVERSE that leaves `f64` — `1/(a² + b²)`, or a
  /// translation built from it, overflowing. `f64` for exactly that second
  /// case: narrowed to `f32`, a scale of `1e-160` would render as the zero
  /// this payload exists to stop reporting.
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
  #[error("crop dimensions {}×{} are unusable (zero axis, or w·h·3 overflows)", .0.width(), .0.height())]
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
  /// The loaded model does not match the manifest's declared contract.
  #[error("model contract mismatch on `{}`: expected {}, got {}", .0.feature(), .0.expected(), .0.actual())]
  ContractMismatch(ContractMismatch),
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
}

/// `Result` specialized to this module's [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
