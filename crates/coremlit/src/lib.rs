//! Safe, synchronous CoreML runtime layer.
//!
//! Wraps `objc2-core-ml` behind owned types: [`Model`] (load / compile /
//! prewarm / predict / stateful predict), [`MultiArray`] (typed N-d tensors,
//! including IOSurface-backed `f16`), and [`Features`] (named model I/O).
//! All `unsafe` FFI lives inside this crate; the public API is safe.
//!
//! macOS only. Mirrors the CoreML surface used by Argmax's WhisperKit
//! (`MLModelExtensions` / `MLMultiArrayExtensions` in argmax-oss-swift).

mod dtype;
mod error;
mod features;
mod model;
mod multi_array;
mod units;

pub use dtype::DataType;
pub use error::{CompileError, LoadError, NsErrorInfo, PredictionError, TensorError};
pub use features::Features;
pub use model::{FeatureInfo, Model, ModelDescription};
pub use multi_array::{Element, MultiArray};
pub use units::{ComputeUnits, ParseComputeUnitsError};
