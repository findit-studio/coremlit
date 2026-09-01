//! Model loading, introspection, prediction.

use std::path::{Path, PathBuf};

use objc2::rc::{Retained, autoreleasepool};
use objc2_core_ml::{MLDictionaryFeatureProvider, MLModel, MLModelConfiguration};
use objc2_foundation::NSURL;

use crate::{
  CompileError, ComputeUnits, DataType, Features, LoadError, MultiArray, NsErrorInfo,
  PredictionError,
};

/// Converts `path` to a file URL through the filesystem-representation API,
/// preserving the exact on-disk bytes.
///
/// `Path::to_string_lossy` would substitute U+FFFD into any non-UTF-8
/// component, silently pointing CoreML at a DIFFERENT path than the one the
/// caller's `exists()` check validated. APFS enforces UTF-8 names, but
/// network and foreign-filesystem mounts on macOS need not.
fn file_url(path: &Path, is_directory: bool) -> Retained<NSURL> {
  use std::os::unix::ffi::OsStrExt;
  let bytes = std::ffi::CString::new(path.as_os_str().as_bytes())
    .expect("callers verify the path exists, so it contains no interior NUL");
  // SAFETY: `bytes` is a valid NUL-terminated filesystem representation
  // borrowed for the duration of the call; the initializer copies it.
  unsafe {
    NSURL::fileURLWithFileSystemRepresentation_isDirectory_relativeToURL(
      core::ptr::NonNull::new(bytes.as_ptr().cast_mut()).expect("CString pointer is non-null"),
      is_directory,
      None,
    )
  }
}

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

/// How many shapes a model will accept for one multi-array feature.
///
/// # Derived from the raw `type` code AND the constraint's contents
///
/// Neither half decides this on its own, and the two measurements that say so
/// point in opposite directions.
///
/// **The code alone is not enough.** A graph converted at a plain fixed shape
/// — no `RangeDim`, no enumerated shapes — reports `…TypeEnumerated` (raw
/// `2`), never `…TypeUnspecified`. Measured on the staged
/// `silero-vad-unified-256ms-v6.2.1.mlmodelc`, whose `metadata.json` records
/// `hasShapeFlexibility: "0"` for every one of its six features: each reports
/// raw type `2`, one enumerated shape equal to [`FeatureInfo::shape`], and one
/// `sizeRangeForDimension` entry per axis with **length 1**. A door that
/// demanded a dedicated "fixed" code would reject every fixed-shape artifact
/// this crate ships.
///
/// **The contents alone are not enough either.** coremltools permits a
/// `RangeDim` whose lower and upper bounds are equal. The dimension stays
/// symbolic and the converter still serialises a `shapeRange`, so CoreML
/// reports raw type `3` (`…TypeRange`) with a span of 1 on every axis. Read
/// off the spans alone that is indistinguishable from the fixed export above —
/// and it is exactly what the fixed-shape invariant exists to refuse, because a
/// symbolic dimension is what takes the graph off the accelerator.
///
/// **So the rule uses both**, and fails closed on anything it has not measured:
///
/// | raw `type` | spans / enumerated shapes | verdict |
/// |---|---|---|
/// | `…TypeEnumerated` (`2`) | at least one axis, every span `1`, at most one enumerated shape | [`Self::Fixed`] |
/// | `…TypeEnumerated` (`2`) | anything else | [`Self::Enumerated`] |
/// | `…TypeRange` (`3`) | anything, unit spans included | [`Self::Range`] |
/// | `…TypeUnspecified` (`1`) | anything | [`Self::Unknown`] |
/// | any other code | anything | [`Self::Unknown`] |
///
/// Only [`Self::Fixed`] establishes a fixed shape. This vocabulary answers that
/// one question; it is deliberately not a count of accepted shapes, and no
/// caller should read it as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[non_exhaustive]
pub enum ShapeConstraint {
  /// Exactly one shape is accepted, and [`FeatureInfo::shape`] is it: every
  /// axis admits exactly one size.
  #[display("fixed")]
  Fixed,
  /// A list of two or more accepted shapes; [`FeatureInfo::shape`] is the
  /// default, not the only one.
  #[display("enumerated")]
  Enumerated,
  /// At least one axis admits a range of sizes (`RangeDims`);
  /// [`FeatureInfo::shape`] is the default, not a bound.
  #[display("range")]
  Range,
  /// The constraint carries a `MLMultiArrayShapeConstraintType` this door has
  /// never measured — `…TypeUnspecified`, or a code newer than this crate.
  /// Carries the raw code for diagnosis.
  ///
  /// Deliberately NOT read as "fixed", however narrow the per-axis sizes look:
  /// a caller that needs a fixed shape needs it established, and a code whose
  /// meaning is unmeasured establishes nothing.
  #[display("unknown({_0})")]
  Unknown(isize),
}

/// `MLMultiArrayShapeConstraintTypeEnumerated`.
const RAW_ENUMERATED: isize = 2;
/// `MLMultiArrayShapeConstraintTypeRange`.
const RAW_RANGE: isize = 3;

/// Classify one multi-array shape constraint from its raw type code and its
/// contents.
///
/// `axis_spans` is how many sizes each axis admits (one entry per dimension,
/// `NSRange::length` from `sizeRangeForDimension`); `enumerated_shapes` is
/// `enumeratedShapes.count`. See [`ShapeConstraint`] for the table this
/// implements and the two measurements that force both inputs to be consulted.
/// A free function over plain numbers so the whole vocabulary is exercisable
/// with no model present.
const fn classify_shape_constraint(
  raw_type: isize,
  enumerated_shapes: usize,
  axis_spans: &[usize],
) -> ShapeConstraint {
  match raw_type {
    // A symbolic dimension, whatever its bounds. An equal-bound `RangeDim`
    // reports unit spans and is still off the fixed-shape path.
    RAW_RANGE => ShapeConstraint::Range,
    RAW_ENUMERATED => {
      if enumerated_shapes <= 1 && all_spans_are_one(axis_spans) {
        ShapeConstraint::Fixed
      } else {
        ShapeConstraint::Enumerated
      }
    }
    other => ShapeConstraint::Unknown(other),
  }
}

/// Whether `spans` is non-empty and every axis admits exactly one size.
///
/// Empty is not "every axis is pinned"; it is a constraint that lists no axes
/// at all, which pins nothing.
const fn all_spans_are_one(spans: &[usize]) -> bool {
  if spans.is_empty() {
    return false;
  }
  let mut i = 0;
  while i < spans.len() {
    if spans[i] != 1 {
      return false;
    }
    i += 1;
  }
  true
}

/// Shape/type info for one model input or output feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureInfo {
  name: String,
  shape: Vec<usize>,
  data_type: Option<DataType>,
  optional: bool,
  shape_constraint: Option<ShapeConstraint>,
}

impl FeatureInfo {
  /// The feature name.
  #[inline(always)]
  pub fn name(&self) -> &str {
    &self.name
  }

  /// Constrained dimensions; empty when the model leaves them open.
  ///
  /// For a feature whose [`Self::shape_constraint`] is not
  /// [`ShapeConstraint::Fixed`] this is the graph's DEFAULT shape, not a
  /// bound: a `RangeDims` input reports a shape it will happily accept others
  /// beside. Pinning a value read from here is only sound once the constraint
  /// says the value is the only one.
  #[inline(always)]
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// Element type for multi-array features; `None` otherwise.
  #[inline(always)]
  pub const fn data_type(&self) -> Option<DataType> {
    self.data_type
  }

  /// Whether the model declares this feature optional — a prediction that
  /// omits it still runs.
  ///
  /// The complement is what matters at load: a REQUIRED input a caller never
  /// supplies makes every prediction fail, so a door that checks only the
  /// features it does send accepts a contract it cannot honour.
  #[inline(always)]
  pub const fn is_optional(&self) -> bool {
    self.optional
  }

  /// How many shapes the model accepts for this feature; `None` for a
  /// non-multi-array feature, which carries no such constraint.
  #[inline(always)]
  pub const fn shape_constraint(&self) -> Option<ShapeConstraint> {
    self.shape_constraint
  }
}

/// Eagerly snapshotted model I/O description.
///
/// # What this carries, and what it deliberately drops
///
/// `MLModelDescription` exposes nine things. This snapshot keeps the three that
/// decide whether a caller's prediction can run at all, and drops the six that
/// describe a model rather than constrain how it is called:
///
/// | `MLModelDescription` member | here | why |
/// |---|---|---|
/// | `inputDescriptionsByName` | [`Self::inputs`] | a REQUIRED input a caller never sends fails every prediction |
/// | `outputDescriptionsByName` | [`Self::outputs`] | the shapes and dtypes a caller reads back |
/// | `stateDescriptionsByName` | [`Self::states`] | a stateful graph must be predicted through `MLState`; the stateless API cannot honour it |
/// | `predictedFeatureName` | dropped | names one existing OUTPUT as a classifier's primary; it is already in [`Self::outputs`] and constrains no call |
/// | `predictedProbabilitiesName` | dropped | likewise, and likewise already an output |
/// | `classLabels` | dropped | the label vocabulary behind a classifier output; descriptive, not a calling constraint |
/// | `metadata` | dropped | author/version/description strings |
/// | `isUpdatable` | dropped | an updatable model still predicts through the same API |
/// | `trainingInputDescriptionsByName`, `parameterDescriptionsByKey` | dropped | on-device UPDATE inputs, reached through `MLUpdateTask`, never through a prediction |
///
/// The rule the table encodes: a member is snapshotted when its contents can
/// make an otherwise-conformant prediction fail, and dropped when it cannot.
/// The three kept members are the complete set that can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescription {
  inputs: Vec<FeatureInfo>,
  outputs: Vec<FeatureInfo>,
  states: Vec<FeatureInfo>,
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

  /// State features (`MLState` buffers) the model declares; empty for a
  /// stateless model.
  ///
  /// A model with a non-empty state set must be predicted through
  /// [`Model::predict_with_state`]: CoreML requires a stateful model to receive
  /// an `MLState`, and the stateless
  /// [`predict`](Model::predict) / [`predict_with`](Model::predict_with) path
  /// either fails or discards the persistence the graph was built around. State
  /// features are NOT ordinary inputs and never appear in [`Self::inputs`], so
  /// a door that reasons only over the input set cannot see them.
  ///
  /// Always empty before macOS 15, where CoreML has no state concept and no
  /// model can declare one — see [`Model::supports_state`].
  #[inline(always)]
  pub fn states(&self) -> &[FeatureInfo] {
    &self.states
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
    let (shape, data_type, shape_constraint) = unsafe {
      description
        .multiArrayConstraint()
        .map_or((Vec::new(), None, None), |constraint| {
          let shape_constraint = constraint.shapeConstraint();
          let axis_spans: Vec<usize> = shape_constraint
            .sizeRangeForDimension()
            .iter()
            .map(|range| range.rangeValue().length)
            .collect();
          (
            constraint.shape().iter().map(|n| n.as_usize()).collect(),
            Some(DataType::from_raw(constraint.dataType().0)),
            Some(classify_shape_constraint(
              shape_constraint.r#type().0,
              shape_constraint.enumeratedShapes().len(),
              &axis_spans,
            )),
          )
        })
    };
    // SAFETY: accessor send on a live feature description.
    let optional = unsafe { description.isOptional() };
    features.push(FeatureInfo {
      name: name.to_string(),
      shape,
      data_type,
      optional,
      shape_constraint,
    });
  }
  features.sort_by(|a, b| a.name.cmp(&b.name));
  features
}

/// The model's declared `MLState` buffers, or an empty set on an OS with no
/// state concept.
///
/// `stateDescriptionsByName` arrived with macOS 15, alongside `MLState` itself.
/// The selector is probed rather than assumed for the same reason
/// [`Model::supports_state`] probes `newState`: sending a message the runtime
/// does not implement traps. The empty set the probe falls back to is not an
/// approximation — CoreML gained state and this accessor in the same release,
/// so an OS without the accessor is an OS on which no loaded model can declare
/// state at all.
fn snapshot_states(description: &objc2_core_ml::MLModelDescription) -> Vec<FeatureInfo> {
  use objc2::runtime::NSObjectProtocol;
  if !description.respondsToSelector(objc2::sel!(stateDescriptionsByName)) {
    return Vec::new();
  }
  // SAFETY: selector availability probed immediately above; the description is
  // live for the call.
  let states = unsafe { description.stateDescriptionsByName() };
  snapshot_features(&states)
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
      return Err(LoadError::NotFound(path.to_path_buf()));
    }
    let url = file_url(path, path.is_dir());
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
    let states = snapshot_states(&raw_description);
    Ok(Self {
      inner,
      description: ModelDescription {
        inputs,
        outputs,
        states,
      },
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
  /// # Autorelease pool
  ///
  /// The whole body runs inside an [`autoreleasepool`], and so do
  /// [`Self::predict_with`] and [`Self::predict_with_state`]. These three
  /// are the only places this crate crosses into CoreML's prediction API,
  /// and every kit — whisper, speaker, siglip, granite, clap, vad, ced,
  /// align — reaches CoreML through one of them, so draining here bounds
  /// every consumer at once rather than one caller's loop.
  ///
  /// A pool is needed because a Rust binary has no ambient one. Objective-C
  /// hosts get theirs from the run loop or from the `@autoreleasepool` the
  /// caller writes; a plain `main` has neither, and libobjc's response to
  /// an `autorelease` with no pool in place is to install a first page and
  /// accumulate into it *for the lifetime of the process* — nothing ever
  /// pops it. Every object CoreML autoreleases per prediction (and every
  /// one this crate's own bridging autoreleases: the `MLFeatureValue`s
  /// `provider_from_pairs` builds, the `NSArray` shape/stride reads behind
  /// [`MultiArray`], the `NSString` names `Features::from_provider`
  /// converts) therefore stays live until the process exits. That is
  /// coremlit issue #62: the live count grows strictly with the number of
  /// predictions, so a long-form transcription's footprint is a function of
  /// audio length with no ceiling.
  ///
  /// **Draining does not invalidate anything this returns.** [`Features`]
  /// owns its arrays through `Retained<MLMultiArray>` — a `+1` strong
  /// reference that is entirely independent of the pool's — and
  /// [`PredictionError`] carries only owned `String`/`isize`
  /// ([`NsErrorInfo`]). Nothing pool-bound can escape by construction
  /// either: [`autoreleasepool`]'s bound is
  /// `for<'pool> F: FnOnce(AutoreleasePool<'pool>) -> T`, so `T` is chosen
  /// independently of `'pool` and a `&T` borrowed from the pool token
  /// cannot appear in the return type. This crate never calls the borrowing
  /// APIs (`Retained::autorelease*`) that would produce such a reference in
  /// the first place — the token is ignored at all three sites.
  ///
  /// # Errors
  /// [`PredictionError::Native`] if CoreML fails; missing/mistyped outputs
  /// surface as structured variants when extracted;
  /// [`PredictionError::AliasCopyFailed`] if de-aliasing an output that
  /// shared a buffer with an input (or another output) fails.
  pub fn predict(&self, inputs: &Features) -> Result<Features, PredictionError> {
    autoreleasepool(|_| {
      let provider = inputs.to_provider()?;
      // Seed with every input's buffer identity: inputs outlive this call (the
      // caller still owns `inputs`), so an identity/zero-copy model echoing
      // one back as an output is the same aliasing hazard as two output names
      // sharing one array, which `from_provider` also catches on its own.
      self.predict_from_provider(&provider, inputs.byte_ranges())
    })
  }

  /// Runs a synchronous prediction from borrowed inputs.
  ///
  /// The per-step decoder path reuses a fixed set of pre-allocated tensors
  /// every step; [`Features`] owns its arrays, so `predict(&Features)` would
  /// force a move-in/move-out of each one on every step, and could not
  /// include a borrowed encoder output at all. This builds the feature
  /// provider directly from borrowed `(name, array)` pairs instead of an
  /// owned [`Features`].
  ///
  /// Sound because `MLFeatureValue(multiArray:)` retains the array it
  /// wraps, so the provider built inside this call does not depend on any
  /// input outliving the call; `&MultiArray` guarantees no `&mut` alias of
  /// any input exists for the call's duration; and [`Model`] is [`Send`]
  /// but deliberately not [`Sync`] (see the `# Concurrency` section above),
  /// so no other thread can be predicting against — or otherwise touching —
  /// this same `Model` concurrently.
  ///
  /// Unlike [`Features`]-based construction (whose insert-by-name cannot
  /// produce duplicates), a raw slice can repeat a name; duplicates are
  /// not rejected — one entry silently wins per `NSDictionary`'s own
  /// construction semantics, and every entry's byte region still seeds
  /// the aliasing detector either way.
  ///
  /// # Errors
  /// As [`Self::predict`], whose `# Autorelease pool` section also covers
  /// this method's pool.
  pub fn predict_with(&self, inputs: &[(&str, &MultiArray)]) -> Result<Features, PredictionError> {
    // Per-prediction pool: see `predict`'s `# Autorelease pool`. This is the
    // per-step decoder entry, so it is also the site the accumulation scales
    // with — one pool per prediction, not one per window, is what keeps the
    // live count flat across a transcription of any length.
    autoreleasepool(|_| {
      let provider = crate::features::provider_from_pairs(inputs.iter().copied())?;
      // As in `predict`: these borrowed inputs outlive this call too (the
      // caller still owns each array), so seed `known_regions` the same way.
      let known_regions = inputs.iter().map(|(_, a)| a.byte_range()).collect();
      self.predict_from_provider(&provider, known_regions)
    })
  }

  // Shared prediction tail for `predict`/`predict_with`: runs
  // `predictionFromFeatures_error` against an already-built `provider` and
  // extracts its outputs, seeding `known_regions` so aliasing with any
  // still caller-owned input is caught by `Features::from_provider`. The
  // two callers differ only in how `provider`/`known_regions` are built
  // (from an owned `Features` vs. borrowed pairs); everything past that
  // point is identical, so it lives here once.
  fn predict_from_provider(
    &self,
    provider: &MLDictionaryFeatureProvider,
    mut known_regions: Vec<(usize, usize)>,
  ) -> Result<Features, PredictionError> {
    // SAFETY: provider conforms to MLFeatureProvider; blocking call.
    let outputs = unsafe {
      self
        .raw()
        .predictionFromFeatures_error(objc2::runtime::ProtocolObject::from_ref(provider))
    }
    .map_err(|e| PredictionError::Native(NsErrorInfo::from_ns_error(&e)))?;
    Features::from_provider(&outputs, &mut known_regions)
  }

  /// Compiles an `.mlpackage`/`.mlmodel` to a temporary `.mlmodelc`.
  ///
  /// Callers move the returned directory to a permanent location.
  ///
  /// # Errors
  /// [`CompileError::NotFound`] / [`CompileError::Native`].
  pub fn compile(source: impl AsRef<Path>) -> Result<PathBuf, CompileError> {
    let source = source.as_ref();
    if !source.exists() {
      return Err(CompileError::NotFound(source.to_path_buf()));
    }
    let url = file_url(source, source.is_dir());
    // SAFETY: blocking compile; Result-checked. The sync API is deprecated
    // in favor of the async block variant, which this sync crate
    // deliberately does not use.
    #[allow(deprecated)]
    let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
      .map_err(|e| CompileError::Native(NsErrorInfo::from_ns_error(&e)))?;
    let path = compiled.path().expect("compiled model URL has a path");
    Ok(PathBuf::from(path.to_string()))
  }

  /// Loads a model and immediately drops it, so the cost of a first load is
  /// paid where the caller puts it rather than inside a later one.
  ///
  /// Ports Swift's `prewarmMode`. The body is exactly [`Self::load`] followed
  /// by a drop: it takes no lock, compiles nothing, and bounds nothing on its
  /// own. Whether a later load is cheaper for having run this depends on what
  /// CoreML caches, which this crate does not measure; any serialization comes
  /// from the caller invoking it before the loads it cares about.
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
  ///
  /// The body runs inside an [`autoreleasepool`] for the same reason
  /// [`Self::predict`]'s does; `state` is mutated in place and outlives the
  /// pool as a `Retained<MLState>`, so the drain does not touch it.
  pub fn predict_with_state(
    &self,
    inputs: &Features,
    state: &mut crate::State,
  ) -> Result<Features, PredictionError> {
    if !self.supports_state() {
      return Err(PredictionError::StateUnsupported);
    }
    autoreleasepool(|_| {
      let provider = inputs.to_provider()?;
      // SAFETY: provider + state are live; &mut state gives exclusivity.
      let outputs = unsafe {
        self.inner.predictionFromFeatures_usingState_error(
          objc2::runtime::ProtocolObject::from_ref(&*provider),
          state.raw(),
        )
      }
      .map_err(|e| PredictionError::Native(NsErrorInfo::from_ns_error(&e)))?;
      // See `predict`'s comment: inputs outlive this call, so seed known_regions
      // with their buffer identities too.
      let mut known_regions = inputs.byte_ranges();
      Features::from_provider(&outputs, &mut known_regions)
    })
  }
}

#[cfg(test)]
mod tests;
