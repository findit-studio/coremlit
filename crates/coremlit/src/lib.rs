//! Safe, synchronous CoreML runtime layer.
//!
//! Wraps `objc2-core-ml` behind owned types: [`Model`] (load / compile /
//! prewarm / predict / stateful predict), [`MultiArray`] (typed N-d tensors,
//! including IOSurface-backed `f16`), and [`Features`] (named model I/O).
//! All `unsafe` FFI lives inside this crate; the public API is safe.
//!
//! macOS only. Mirrors the CoreML surface used by Argmax's WhisperKit
//! (`MLModelExtensions` / `MLMultiArrayExtensions` in argmax-oss-swift).
//!
//! # Example
//!
//! ```no_run
//! use coremlit::{ComputeUnits, DataType, Features, Model, MultiArray};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Model::load("MelSpectrogram.mlmodelc", ComputeUnits::CpuAndGpu)?;
//! let audio = MultiArray::zeros(&[480_000], DataType::F32)?;
//! let outputs = model.predict(&Features::new().with("audio", audio))?;
//! let mel = outputs.get("melspectrogram_features").unwrap();
//! assert_eq!(mel.data_type(), DataType::F16);
//! # Ok(())
//! # }
//! ```

mod dtype;
mod error;
mod features;
mod model;
mod multi_array;
mod state;
mod units;

pub use dtype::DataType;
pub use error::{CompileError, LoadError, NsErrorInfo, PredictionError, TensorError};
pub use features::Features;
pub use model::{FeatureInfo, Model, ModelDescription};
pub use multi_array::{Element, MultiArray};
pub use state::State;
pub use units::{ComputeUnits, ParseComputeUnitsError};
