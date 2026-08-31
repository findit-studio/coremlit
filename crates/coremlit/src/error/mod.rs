//! Structured error types for the CoreML layer.

use std::path::PathBuf;

use objc2_foundation::NSError;

use crate::DataType;

/// Structured capture of an `NSError` returned by CoreML.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{domain} (code {code}): {message}")]
pub struct NsErrorInfo {
  domain: String,
  code: isize,
  message: String,
}

impl NsErrorInfo {
  /// Construct from a live `NSError` reference.
  pub(crate) fn from_ns_error(error: &NSError) -> Self {
    // Plain accessor message sends on a live NSError reference.
    let (domain, code, message) = (
      error.domain().to_string(),
      error.code(),
      error.localizedDescription().to_string(),
    );
    Self {
      domain,
      code,
      message,
    }
  }

  /// The `NSError` domain.
  #[inline(always)]
  pub fn domain(&self) -> &str {
    &self.domain
  }

  /// The `NSError` code.
  #[inline(always)]
  pub const fn code(&self) -> isize {
    self.code
  }

  /// The localized description.
  #[inline(always)]
  pub fn message(&self) -> &str {
    &self.message
  }
}

/// Failure loading a compiled model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
  /// The model path does not exist.
  ///
  /// Carries the path that was checked.
  #[error("model not found at `{}`", .0.display())]
  NotFound(PathBuf),
  /// CoreML rejected the model.
  #[error("core ml failed to load model: {0}")]
  Native(NsErrorInfo),
}

/// Failure compiling an `.mlpackage`/`.mlmodel` into an `.mlmodelc`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CompileError {
  /// The source path does not exist.
  ///
  /// Carries the path that was checked.
  #[error("model source not found at `{}`", .0.display())]
  NotFound(PathBuf),
  /// CoreML rejected the compilation.
  #[error("core ml failed to compile model: {0}")]
  Native(NsErrorInfo),
}

/// Failure running a prediction.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PredictionError {
  /// The output feature dictionary lacks an expected name.
  ///
  /// Carries the feature name that was absent.
  #[error("prediction output is missing feature `{0}`")]
  MissingOutput(String),
  /// An output feature was not a multi-array.
  ///
  /// Carries the feature name with the wrong kind.
  #[error("prediction output `{0}` is not a multi-array")]
  NotMultiArray(String),
  /// CoreML reported a prediction failure.
  #[error("core ml prediction failed: {0}")]
  Native(NsErrorInfo),
  /// Stateful prediction requires macOS 15 (MLState).
  #[error("stateful prediction is unavailable on this OS (requires macOS 15)")]
  StateUnsupported,
  /// De-aliasing an output array that shared its native buffer with
  /// another live array (an input, or another output name) failed.
  #[error("failed to de-alias a prediction output: {0}")]
  AliasCopyFailed(TensorError),
}

/// The array's element type differs from the requested view type.
///
/// Payload of [`TensorError::DataTypeMismatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTypeMismatch {
  /// Requested element type.
  expected: DataType,
  /// The array's actual element type.
  actual: DataType,
}

impl DataTypeMismatch {
  /// Construct from the requested and the actual element type.
  #[inline(always)]
  pub const fn new(expected: DataType, actual: DataType) -> Self {
    Self { expected, actual }
  }

  /// Requested element type.
  #[inline(always)]
  pub const fn expected(&self) -> DataType {
    self.expected
  }

  /// The array's actual element type.
  #[inline(always)]
  pub const fn actual(&self) -> DataType {
    self.actual
  }
}

/// Element count differs from the shape's product.
///
/// Payload of [`TensorError::ShapeMismatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatch {
  /// Elements implied by the shape.
  expected: usize,
  /// Elements provided.
  actual: usize,
}

impl ShapeMismatch {
  /// Construct from the implied and the provided element count.
  #[inline(always)]
  pub const fn new(expected: usize, actual: usize) -> Self {
    Self { expected, actual }
  }

  /// Elements implied by the shape.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Elements provided.
  #[inline(always)]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// Index tuple rank differs from the array rank.
///
/// Payload of [`TensorError::RankMismatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankMismatch {
  /// The array's rank.
  expected: usize,
  /// Indices provided.
  actual: usize,
}

impl RankMismatch {
  /// Construct from the array's rank and the number of indices provided.
  #[inline(always)]
  pub const fn new(expected: usize, actual: usize) -> Self {
    Self { expected, actual }
  }

  /// The array's rank.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Indices provided.
  #[inline(always)]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// A linear or dimensional index is out of bounds.
///
/// Payload of [`TensorError::IndexOutOfBounds`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexOutOfBounds {
  /// Offending index.
  index: usize,
  /// Bound it violated.
  len: usize,
}

// `len` names the BOUND one failed index violated, not a collection length
// this payload owns, so there is nothing for an `is_empty` to mean here.
#[allow(clippy::len_without_is_empty)]
impl IndexOutOfBounds {
  /// Construct from the offending index and the bound it violated.
  #[inline(always)]
  pub const fn new(index: usize, len: usize) -> Self {
    Self { index, len }
  }

  /// Offending index.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// Bound it violated.
  #[inline(always)]
  pub const fn len(&self) -> usize {
    self.len
  }
}

/// The array's memory layout is not row-major contiguous.
///
/// Payload of [`TensorError::NonContiguous`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonContiguous {
  /// The array's shape.
  shape: Vec<usize>,
  /// The array's element strides.
  strides: Vec<usize>,
}

impl NonContiguous {
  /// Construct from the array's shape and its element strides.
  #[inline(always)]
  pub const fn new(shape: Vec<usize>, strides: Vec<usize>) -> Self {
    Self { shape, strides }
  }

  /// The array's shape.
  #[inline(always)]
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// The array's element strides.
  #[inline(always)]
  pub fn strides(&self) -> &[usize] {
    &self.strides
  }
}

/// The shape does not meet an operation's structural requirement.
///
/// Payload of [`TensorError::UnsupportedShape`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedShape {
  /// The offending shape.
  shape: Vec<usize>,
  /// Why the shape was rejected.
  reason: ShapeRequirement,
}

impl UnsupportedShape {
  /// Construct from the offending shape and the requirement it failed.
  #[inline(always)]
  pub const fn new(shape: Vec<usize>, reason: ShapeRequirement) -> Self {
    Self { shape, reason }
  }

  /// The offending shape.
  #[inline(always)]
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// Why the shape was rejected.
  #[inline(always)]
  pub const fn reason(&self) -> ShapeRequirement {
    self.reason
  }
}

/// Failure constructing or viewing a multi-array.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TensorError {
  /// The array's element type differs from the requested view type.
  #[error("data type mismatch: expected `{}`, got `{}`", .0.expected(), .0.actual())]
  DataTypeMismatch(DataTypeMismatch),
  /// Element count differs from the shape's product.
  #[error("shape mismatch: expected {} elements, got {}", .0.expected(), .0.actual())]
  ShapeMismatch(ShapeMismatch),
  /// Index tuple rank differs from the array rank.
  #[error("rank mismatch: expected {} indices, got {}", .0.expected(), .0.actual())]
  RankMismatch(RankMismatch),
  /// A linear or dimensional index is out of bounds.
  #[error("index {} out of bounds for length {}", .0.index(), .0.len())]
  IndexOutOfBounds(IndexOutOfBounds),
  /// The array's memory layout is not row-major contiguous.
  #[error("array layout is not contiguous (strides {:?} for shape {:?})", .0.strides(), .0.shape())]
  NonContiguous(NonContiguous),
  /// The data type cannot back an array (no known element size).
  ///
  /// Carries the rejected data type.
  #[error("unsupported data type `{0}` for array construction")]
  UnsupportedDataType(DataType),
  /// CoreML rejected the array construction.
  #[error("core ml multi-array failure: {0}")]
  Native(NsErrorInfo),
  /// CVPixelBuffer creation failed.
  ///
  /// Carries the CVReturn code.
  #[error("pixel buffer creation failed with CVReturn {0}")]
  PixelBuffer(i32),
  /// The shape does not meet an operation's structural requirement.
  #[error("shape {:?} is unsupported: {}", .0.shape(), .0.reason())]
  UnsupportedShape(UnsupportedShape),
  /// A shape's element count, or a size/offset derived from it, overflows
  /// `usize`.
  ///
  /// Carries the offending shape.
  #[error("shape {0:?} element count overflows usize")]
  ShapeOverflow(Vec<usize>),
  /// `MLMultiArray`'s pixel-buffer-backed initializer is unavailable on
  /// this OS.
  #[error("pixel-buffer-backed arrays require macOS 12 or newer")]
  SurfaceUnsupported,
}

/// Why a shape was rejected by [`TensorError::UnsupportedShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum ShapeRequirement {
  /// All dimensions before the last must be 1.
  LeadingDimsUnit,
  /// The shape must have at least one dimension.
  NonEmpty,
  /// Every dimension must be nonzero.
  NonZeroDims,
}

impl ShapeRequirement {
  /// Stable name of the requirement.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::LeadingDimsUnit => "all dimensions before the last must be 1",
      Self::NonEmpty => "the shape must have at least one dimension",
      Self::NonZeroDims => "every dimension must be nonzero",
    }
  }
}

#[cfg(test)]
mod tests;
