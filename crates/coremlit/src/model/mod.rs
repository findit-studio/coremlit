//! Model loading, introspection, prediction.

use std::path::Path;

use objc2::rc::Retained;
use objc2_core_ml::{MLModel, MLModelConfiguration};
use objc2_foundation::NSURL;

use crate::{
  CompileError, ComputeUnits, DataType, Features, LoadError, NsErrorInfo, PredictionError,
};

/// A loaded CoreML model.
///
/// # Concurrency
///
/// `Model` is [`Send`] but deliberately **not** [`Sync`]: Apple documents,
/// "Use an MLModel instance on one thread or one dispatch queue at a
/// time" — concurrent `&Model` access from multiple threads is outside that
/// contract. Callers that want to fan prediction work out across threads
/// need one `Model` per worker (each independently loaded, or all
/// serialized behind an external `Mutex`) rather than sharing one instance.
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<coremlit::Model>();
/// ```
#[derive(Debug)]
pub struct Model {
  inner: Retained<MLModel>,
  description: ModelDescription,
}

// SAFETY: Apple's contract is about serialization ("one thread or one
// dispatch queue at a time"), not confinement to the thread that loaded the
// model, so moving a `Model` to another thread and continuing to use it
// only from there afterward is exactly the documented usage pattern; the
// wrapper also exposes no unsynchronized interior mutation for the move
// itself to race against. Deliberately not `Sync` (see the `# Concurrency`
// doc section above) — that would permit *concurrent* `&Model` access from
// multiple threads, which Apple's "one thread ... at a time" contract rules
// out.
unsafe impl Send for Model {}

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
    // objc2-foundation version, so no `unsafe` is needed here.
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

  pub(crate) fn raw(&self) -> &MLModel {
    &self.inner
  }

  /// Runs a synchronous prediction.
  ///
  /// # Errors
  /// [`PredictionError::Native`] if CoreML fails; missing/mistyped outputs
  /// surface as structured variants when extracted;
  /// [`PredictionError::AliasCopyFailed`] if de-aliasing an output that
  /// shared a buffer with an input (or another output) fails.
  pub fn predict(&self, inputs: &Features) -> Result<Features, PredictionError> {
    let provider = inputs.to_provider()?;
    // SAFETY: provider conforms to MLFeatureProvider; blocking call.
    let outputs = unsafe {
      self
        .raw()
        .predictionFromFeatures_error(objc2::runtime::ProtocolObject::from_ref(&*provider))
    }
    .map_err(|e| PredictionError::Native(NsErrorInfo::from_ns_error(&e)))?;
    // Seed with every input's buffer identity: inputs outlive this call (the
    // caller still owns `inputs`), so an identity/zero-copy model echoing
    // one back as an output is the same aliasing hazard as two output names
    // sharing one array, which `from_provider` also catches on its own.
    let mut known_ptrs = inputs.data_ptrs();
    Features::from_provider(&outputs, &mut known_ptrs)
  }

  /// Compiles an `.mlpackage`/`.mlmodel` to a temporary `.mlmodelc`.
  ///
  /// Callers move the returned directory to a permanent location.
  ///
  /// # Errors
  /// [`CompileError::NotFound`] / [`CompileError::Native`].
  pub fn compile(source: impl AsRef<Path>) -> Result<std::path::PathBuf, CompileError> {
    let source = source.as_ref();
    if !source.exists() {
      return Err(CompileError::NotFound {
        path: source.to_path_buf(),
      });
    }
    let url = NSURL::fileURLWithPath(&objc2_foundation::NSString::from_str(
      &source.to_string_lossy(),
    ));
    // SAFETY: blocking compile; Result-checked. The sync API is deprecated
    // in favor of the async block variant, which this sync crate
    // deliberately does not use.
    #[allow(deprecated)]
    let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
      .map_err(|e| CompileError::Native(NsErrorInfo::from_ns_error(&e)))?;
    let path = compiled.path().expect("compiled model URL has a path");
    Ok(std::path::PathBuf::from(path.to_string()))
  }

  /// Loads a model and immediately drops it.
  ///
  /// Serializes ANE compilation and caps peak memory before a real
  /// concurrent load — ports Swift's `prewarmMode`.
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn prewarm(path: impl AsRef<Path>, units: ComputeUnits) -> Result<(), LoadError> {
    Self::load(path, units).map(drop)
  }

  /// Whether this OS supports stateful prediction (macOS 15+).
  ///
  /// Backs the availability guard in both [`Self::make_state`] and
  /// [`Self::predict_with_state`].
  pub fn supports_state(&self) -> bool {
    use objc2::runtime::NSObjectProtocol;
    self.inner.respondsToSelector(objc2::sel!(newState))
  }

  /// Creates fresh model state for stateful prediction.
  ///
  /// CoreML defines `newState()` on a model with no declared state buffers
  /// (e.g. WhisperKit's `MelSpectrogram`) as returning an *empty* state;
  /// running [`Self::predict_with_state`] with that state then behaves
  /// identically to [`Self::predict`]. Confirmed against `MelSpectrogram` in
  /// this crate's integration tests — TTSKit's genuinely stateful models
  /// exercise the buffer-carrying path this type exists for.
  ///
  /// # Errors
  /// [`PredictionError::StateUnsupported`] before macOS 15.
  pub fn make_state(&self) -> Result<crate::State, PredictionError> {
    if !self.supports_state() {
      return Err(PredictionError::StateUnsupported);
    }
    // SAFETY: availability probed above.
    Ok(crate::State::from_raw(unsafe { self.inner.newState() }))
  }

  /// Runs a synchronous stateful prediction, mutating `state` in place.
  ///
  /// On an empty state (see [`Self::make_state`]) this behaves identically
  /// to [`Self::predict`].
  ///
  /// # Errors
  /// [`PredictionError::StateUnsupported`] before macOS 15;
  /// [`PredictionError::Native`] on CoreML failure;
  /// [`PredictionError::AliasCopyFailed`] if de-aliasing an output that
  /// shared a buffer with an input (or another output) fails.
  pub fn predict_with_state(
    &self,
    inputs: &Features,
    state: &mut crate::State,
  ) -> Result<Features, PredictionError> {
    if !self.supports_state() {
      return Err(PredictionError::StateUnsupported);
    }
    let provider = inputs.to_provider()?;
    // SAFETY: provider + state are live; &mut state gives exclusivity.
    let outputs = unsafe {
      self.inner.predictionFromFeatures_usingState_error(
        objc2::runtime::ProtocolObject::from_ref(&*provider),
        state.raw(),
      )
    }
    .map_err(|e| PredictionError::Native(NsErrorInfo::from_ns_error(&e)))?;
    // See `predict`'s comment: inputs outlive this call, so seed known_ptrs
    // with their buffer identities too.
    let mut known_ptrs = inputs.data_ptrs();
    Features::from_provider(&outputs, &mut known_ptrs)
  }
}

#[cfg(test)]
mod tests;
