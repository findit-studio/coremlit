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
  #[error("model not found at `{path}`", path = path.display())]
  NotFound {
    /// Path that was checked.
    path: PathBuf,
  },
  /// CoreML rejected the model.
  #[error("core ml failed to load model: {0}")]
  Native(NsErrorInfo),
}

/// Failure compiling an `.mlpackage`/`.mlmodel` into an `.mlmodelc`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CompileError {
  /// The source path does not exist.
  #[error("model source not found at `{path}`", path = path.display())]
  NotFound {
    /// Path that was checked.
    path: PathBuf,
  },
  /// CoreML rejected the compilation.
  #[error("core ml failed to compile model: {0}")]
  Native(NsErrorInfo),
}

/// Failure running a prediction.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PredictionError {
  /// The output feature dictionary lacks an expected name.
  #[error("prediction output is missing feature `{name}`")]
  MissingOutput {
    /// Feature name that was absent.
    name: String,
  },
  /// An output feature was not a multi-array.
  #[error("prediction output `{name}` is not a multi-array")]
  NotMultiArray {
    /// Feature name with the wrong kind.
    name: String,
  },
  /// CoreML reported a prediction failure.
  #[error("core ml prediction failed: {0}")]
  Native(NsErrorInfo),
}

/// Failure constructing or viewing a multi-array.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TensorError {
  /// The array's element type differs from the requested view type.
  #[error("data type mismatch: expected `{expected}`, got `{actual}`")]
  DataTypeMismatch {
    /// Requested element type.
    expected: DataType,
    /// The array's actual element type.
    actual: DataType,
  },
  /// Element count differs from the shape's product.
  #[error("shape mismatch: expected {expected} elements, got {actual}")]
  ShapeMismatch {
    /// Elements implied by the shape.
    expected: usize,
    /// Elements provided.
    actual: usize,
  },
  /// Index tuple rank differs from the array rank.
  #[error("rank mismatch: expected {expected} indices, got {actual}")]
  RankMismatch {
    /// The array's rank.
    expected: usize,
    /// Indices provided.
    actual: usize,
  },
  /// A linear or dimensional index is out of bounds.
  #[error("index {index} out of bounds for length {len}")]
  IndexOutOfBounds {
    /// Offending index.
    index: usize,
    /// Bound it violated.
    len: usize,
  },
  /// The data type cannot back an array (no known element size).
  #[error("unsupported data type `{dtype}` for array construction")]
  UnsupportedDataType {
    /// The rejected data type.
    dtype: DataType,
  },
  /// CoreML rejected the array construction.
  #[error("core ml multi-array failure: {0}")]
  Native(NsErrorInfo),
}

#[cfg(test)]
mod tests;
