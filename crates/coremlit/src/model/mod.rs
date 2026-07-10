//! Model loading, introspection, prediction.

use std::path::Path;

use objc2::rc::Retained;
use objc2_core_ml::{MLModel, MLModelConfiguration};
use objc2_foundation::NSURL;

use crate::{ComputeUnits, DataType, LoadError, NsErrorInfo};

/// A loaded CoreML model.
///
/// `MLModel` prediction is thread-safe, so `Model` is `Send + Sync`; share
/// one instance across worker threads.
#[derive(Debug)]
pub struct Model {
  inner: Retained<MLModel>,
  description: ModelDescription,
}

// SAFETY: Apple documents MLModel prediction as thread-safe; the wrapper
// exposes no unsynchronized interior mutation.
unsafe impl Send for Model {}
// SAFETY: as above.
unsafe impl Sync for Model {}

/// Shape/type info for one model input or output feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureInfo {
  name: String,
  shape: Vec<usize>,
  data_type: Option<DataType>,
}

impl FeatureInfo {
  /// The feature name.
  #[inline(always)]
  pub fn name(&self) -> &str {
    &self.name
  }

  /// Constrained dimensions; empty when the model leaves them open.
  #[inline(always)]
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// Element type for multi-array features; `None` otherwise.
  #[inline(always)]
  pub const fn data_type(&self) -> Option<DataType> {
    self.data_type
  }
}

/// Eagerly snapshotted model I/O description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescription {
  inputs: Vec<FeatureInfo>,
  outputs: Vec<FeatureInfo>,
}

impl ModelDescription {
  /// Input features.
  #[inline(always)]
  pub fn inputs(&self) -> &[FeatureInfo] {
    &self.inputs
  }

  /// Output features.
  #[inline(always)]
  pub fn outputs(&self) -> &[FeatureInfo] {
    &self.outputs
  }

  /// Input feature named `name`.
  pub fn input(&self, name: &str) -> Option<&FeatureInfo> {
    self.inputs.iter().find(|f| f.name == name)
  }

  /// Output feature named `name`.
  pub fn output(&self, name: &str) -> Option<&FeatureInfo> {
    self.outputs.iter().find(|f| f.name == name)
  }
}

fn snapshot_features(
  descriptions: &objc2_foundation::NSDictionary<
    objc2_foundation::NSString,
    objc2_core_ml::MLFeatureDescription,
  >,
) -> Vec<FeatureInfo> {
  let mut features = Vec::new();
  for name in descriptions.keys() {
    let description = descriptions.objectForKey(&name).expect("key from keys()");
    // SAFETY: accessor sends; multiArrayConstraint is nil for
    // non-multi-array features.
    let (shape, data_type) = unsafe {
      description
        .multiArrayConstraint()
        .map_or((Vec::new(), None), |constraint| {
          (
            constraint.shape().iter().map(|n| n.as_usize()).collect(),
            Some(DataType::from_raw(constraint.dataType().0)),
          )
        })
    };
    features.push(FeatureInfo {
      name: name.to_string(),
      shape,
      data_type,
    });
  }
  features.sort_by(|a, b| a.name.cmp(&b.name));
  features
}

impl Model {
  /// Loads a compiled `.mlmodelc` with the given compute units.
  ///
  /// # Errors
  /// [`LoadError::NotFound`] if `path` does not exist;
  /// [`LoadError::Native`] if CoreML rejects the model.
  pub fn load(path: impl AsRef<Path>, units: ComputeUnits) -> Result<Self, LoadError> {
    let path = path.as_ref();
    if !path.exists() {
      return Err(LoadError::NotFound {
        path: path.to_path_buf(),
      });
    }
    // `fileURLWithPath`/`NSString::from_str` are safe constructors in this
    // objc2-foundation version (no `unsafe` needed, unlike the brief's
    // assumption — see Task 8 report).
    let url = NSURL::fileURLWithPath(&objc2_foundation::NSString::from_str(
      &path.to_string_lossy(),
    ));
    // SAFETY: fresh configuration object; setComputeUnits is a setter.
    let configuration = unsafe {
      let configuration = MLModelConfiguration::new();
      configuration.setComputeUnits(units.to_raw());
      configuration
    };
    // SAFETY: URL and configuration are live; error checked via Result.
    let inner =
      unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration) }
        .map_err(|e| LoadError::Native(NsErrorInfo::from_ns_error(&e)))?;
    // SAFETY: accessor send.
    let raw_description = unsafe { inner.modelDescription() };
    // SAFETY: dictionary accessors on a live description.
    let (inputs, outputs) = unsafe {
      (
        snapshot_features(&raw_description.inputDescriptionsByName()),
        snapshot_features(&raw_description.outputDescriptionsByName()),
      )
    };
    Ok(Self {
      inner,
      description: ModelDescription { inputs, outputs },
    })
  }

  /// The model's I/O description (snapshotted at load).
  #[inline(always)]
  pub const fn description(&self) -> &ModelDescription {
    &self.description
  }

  #[allow(dead_code)] // consumed from Task 9 (Model::predict)
  pub(crate) fn raw(&self) -> &MLModel {
    &self.inner
  }
}

#[cfg(test)]
mod tests;
