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

/// One axis's size range, exactly as `sizeRangeForDimension` reports it.
///
/// [`Self::min`] is the smallest size the axis admits and [`Self::count`] how
/// many consecutive sizes it admits, so a **fixed** axis of size `d` reads
/// `(d, 1)` — measured on every probe below and on both bundles this repository
/// ships. A `RangeDim(lower, upper)` axis reads `(lower, upper − lower + 1)`;
/// an unbounded one reads `(1, isize::MAX as usize)`, which only a
/// `neuralnetwork` export produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AxisRange {
  min: usize,
  count: usize,
}

impl AxisRange {
  /// From the raw `NSRange`: the smallest admitted size and how many
  /// consecutive sizes are admitted.
  #[inline(always)]
  pub const fn new(min: usize, count: usize) -> Self {
    Self { min, count }
  }

  /// The range an axis bounded by `min..=max` inclusive reports, i.e.
  /// `(min, max − min + 1)`. Saturates rather than wrapping on `max < min`,
  /// which no constraint produces.
  #[inline(always)]
  pub const fn inclusive(min: usize, max: usize) -> Self {
    Self::new(min, max.saturating_sub(min).saturating_add(1))
  }

  /// The smallest size this axis admits.
  #[inline(always)]
  pub const fn min(&self) -> usize {
    self.min
  }

  /// How many consecutive sizes this axis admits; `1` for a pinned axis.
  #[inline(always)]
  pub const fn count(&self) -> usize {
    self.count
  }
}

impl core::fmt::Display for AxisRange {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self.count {
      0 => write!(f, "(no size)"),
      1 => write!(f, "{}", self.min),
      _ => write!(f, "{}..={}", self.min, self.min + self.count - 1),
    }
  }
}

/// Why a `…TypeEnumerated` constraint could not be classified: what was
/// OBSERVED, in the words of the clause it failed.
///
/// Every one of these is a combination no producer this door has measured
/// emits, so none of them is resolved to a verdict — see [`ShapeConstraint`]
/// for the measured table and the probes behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[non_exhaustive]
pub enum UnmeasuredEnumeration {
  /// The constraint lists NO enumerated shape. coremltools refuses an
  /// `EnumeratedShapes` of length 1 and a plain fixed export still lists its
  /// one shape, so nothing measured here produces an empty list.
  #[display("no enumerated shape")]
  NoShapes,
  /// Its ONE enumerated shape is not [`FeatureInfo::shape`]. A one-shape
  /// constraint whose shape is not the declared one describes a model whose
  /// default is not among the shapes it accepts.
  #[display("sole enumerated shape is not the declared shape")]
  SoleShapeIsNotDeclared,
  /// Its per-axis ranges do not pin [`FeatureInfo::shape`]: a different number
  /// of them than the shape has axes, one that does not read `(size, 1)`, or a
  /// declared shape with no axes at all (which pins nothing).
  #[display("per-axis ranges do not pin the declared shape")]
  SpansDoNotPinDeclaredShape,
}

/// How many shapes a model will accept for one multi-array feature.
///
/// # A measured table, not an inference
///
/// Probe artifacts were built with the conversion recipes' own coremltools
/// 8.3.0 (a `mlprogram` and a `neuralnetwork` export of one traced graph, at a
/// fixed shape, three enumerated shapes, an equal-bound `RangeDim`, an open
/// `RangeDim` and an unbounded one), compiled, and their
/// `MLMultiArrayShapeConstraint` read back with a Swift probe. Every row below
/// says what it was measured against, and the rows nothing produced fail
/// closed.
///
/// `declared` is [`FeatureInfo::shape`]; `ranges` is
/// [`FeatureInfo::axis_ranges`]; `enumerated` is
/// [`FeatureInfo::enumerated_shapes`].
///
/// | raw `type` | enumerated | ranges | verdict | evidence |
/// |---|---|---|---|---|
/// | `2` | one, `== declared` | all `(d, 1)`, count = rank | [`Self::Fixed`] | mlprogram `fixed`, `nn_fixed`, shipped silero, published redimnet |
/// | `2` | ≥ 2 | all `(d, 1)` | [`Self::Enumerated`] | mlprogram & nn `enum3` — the ranges report the DEFAULT only, so the count is the sole discriminator |
/// | `2` | 0 | any | [`Self::Unmeasured`] | unmeasured — coremltools refuses an `EnumeratedShapes` of length 1, and no producer of 0 was found |
/// | `2` | one, `!= declared` | any | [`Self::Unmeasured`] | unmeasured |
/// | `2` | one, `== declared` | count ≠ rank, or one not `(d, 1)`, or no axes | [`Self::Unmeasured`] | unmeasured |
/// | `3` | 0 | all `(d, 1)` | [`Self::Range`] | `RangeDim(401, 401)` — an equal-bound range stays symbolic |
/// | `3` | 0 | some wider than 1 | [`Self::Range`] | `range_open`, `nn_range`, `nn_range_unbounded` (`1 + 2⁶³−1`) |
/// | `1` | 0 | `[]`, shape `[]` | [`Self::Unspecified`] | every output downstream of a flexible input, and every output of a `neuralnetwork` export even when fixed |
/// | other | any | any | [`Self::Unknown`] | unmeasured, fails closed |
///
/// # Why both halves are consulted
///
/// **The code alone is not enough.** A graph converted at a plain fixed shape
/// — no `RangeDim`, no enumerated shapes — reports `…TypeEnumerated` (raw
/// `2`), never `…TypeUnspecified`. Measured on the staged
/// `silero-vad-unified-256ms-v6.2.1.mlmodelc`, whose `metadata.json` records
/// `hasShapeFlexibility: "0"` for every one of its six features. A door that
/// demanded a dedicated "fixed" code would reject every fixed-shape artifact
/// this crate ships.
///
/// **The contents alone are not enough either.** coremltools permits a
/// `RangeDim` whose lower and upper bounds are equal. The dimension stays
/// symbolic and the converter still serialises a `shapeRange`, so CoreML
/// reports raw type `3` with a range of `(d, 1)` on every axis. Read off the
/// ranges alone that is indistinguishable from the fixed export above — and it
/// is exactly what the fixed-shape invariant exists to refuse, because a
/// symbolic dimension is what takes the graph off the accelerator.
///
/// # One cell the rows above resolve between them
///
/// Two shapes or more is decided BEFORE the ranges are looked at, so a `≥ 2`
/// constraint is [`Self::Enumerated`] whatever its ranges say — including a
/// range count that is not the rank. That is forced by the measurement in row
/// two: under this code the ranges report the default shape and nothing about
/// the alternatives, so they carry no information to fail closed on. The
/// rank clause is therefore a clause of the `Fixed` rule, not of the code.
///
/// Only [`Self::Fixed`] establishes a fixed shape. This vocabulary answers that
/// one question; it is deliberately not a count of accepted shapes, and no
/// caller should read it as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[non_exhaustive]
pub enum ShapeConstraint {
  /// Exactly one shape is accepted, and [`FeatureInfo::shape`] is it: it is the
  /// sole enumerated shape, and every axis admits exactly that size.
  #[display("fixed")]
  Fixed,
  /// A list of two or more accepted shapes; [`FeatureInfo::shape`] is the
  /// default, not the only one, and [`FeatureInfo::axis_ranges`] reports that
  /// default rather than a bounding box over the list.
  #[display("enumerated")]
  Enumerated,
  /// At least one axis is symbolic (`RangeDim`); [`FeatureInfo::shape`] is the
  /// default, not a bound, and [`FeatureInfo::axis_ranges`] carries the real
  /// per-axis bounds. An equal-bound `RangeDim` lands here too: it reports
  /// `(d, 1)` and is still off the fixed-shape path.
  #[display("range")]
  Range,
  /// `MLMultiArrayShapeConstraintTypeUnspecified` (raw `1`): the constraint
  /// records nothing that decides the question.
  ///
  /// This is the **common** case, not an exotic one, and naming it is the
  /// point: measured on every output downstream of a flexible input, and on
  /// every output of a `neuralnetwork` export even when its input is fixed.
  /// Such a feature carries no ranges and an empty shape, so nothing about its
  /// geometry can be read off the description at all.
  ///
  /// Deliberately NOT read as "fixed": a caller that needs a fixed shape needs
  /// it established, and a constraint that records nothing establishes nothing.
  #[display("unspecified")]
  Unspecified,
  /// `…TypeEnumerated` whose contents match no measured row of the table
  /// above. Carries what was observed.
  ///
  /// Deliberately NOT read as "fixed", however narrow the per-axis ranges look.
  #[display("unmeasured({_0})")]
  Unmeasured(UnmeasuredEnumeration),
  /// A `MLMultiArrayShapeConstraintType` code this door has never measured —
  /// one newer than this crate. Carries the raw code for diagnosis.
  ///
  /// Deliberately NOT read as "fixed", for the same reason
  /// [`Self::Unspecified`] is not.
  #[display("unknown({_0})")]
  Unknown(isize),
}

/// `MLMultiArrayShapeConstraintTypeUnspecified`.
const RAW_UNSPECIFIED: isize = 1;
/// `MLMultiArrayShapeConstraintTypeEnumerated`.
const RAW_ENUMERATED: isize = 2;
/// `MLMultiArrayShapeConstraintTypeRange`.
const RAW_RANGE: isize = 3;

/// Classify one multi-array shape constraint from its raw type code, the
/// feature's declared shape, and the constraint's own contents.
///
/// See [`ShapeConstraint`] for the measured table this implements and the two
/// measurements that force both the code and the contents to be consulted. A
/// free function over plain numbers so the whole vocabulary is exercisable with
/// no model present.
///
/// The `…TypeEnumerated` arm tests the [`ShapeConstraint::Fixed`] conjuncts in
/// the order the table lists them, so a constraint that fails several reports
/// the first — enough to refuse it, which is all a fail-closed verdict owes.
fn classify_shape_constraint(
  raw_type: isize,
  declared_shape: &[usize],
  enumerated_shapes: &[Vec<usize>],
  axis_ranges: &[AxisRange],
) -> ShapeConstraint {
  match raw_type {
    // A symbolic dimension, whatever its bounds. An equal-bound `RangeDim`
    // reports `(d, 1)` on every axis and is still off the fixed-shape path.
    RAW_RANGE => ShapeConstraint::Range,
    RAW_UNSPECIFIED => ShapeConstraint::Unspecified,
    RAW_ENUMERATED => {
      // Two or more shapes is the one discriminator that holds: the ranges
      // under this code report the DEFAULT shape, not a bounding box over the
      // list, so they cannot separate an enumerated constraint from a fixed
      // one and are not consulted here.
      if enumerated_shapes.len() >= 2 {
        return ShapeConstraint::Enumerated;
      }
      if enumerated_shapes.is_empty() {
        return ShapeConstraint::Unmeasured(UnmeasuredEnumeration::NoShapes);
      }
      if enumerated_shapes[0] != declared_shape {
        return ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SoleShapeIsNotDeclared);
      }
      // A shape with no axes pins nothing: "every axis admits one size" is
      // vacuously true of no axes, and that is not the same fact.
      if declared_shape.is_empty()
        || axis_ranges.len() != declared_shape.len()
        || !axis_ranges
          .iter()
          .zip(declared_shape)
          .all(|(range, size)| *range == AxisRange::new(*size, 1))
      {
        return ShapeConstraint::Unmeasured(UnmeasuredEnumeration::SpansDoNotPinDeclaredShape);
      }
      ShapeConstraint::Fixed
    }
    other => ShapeConstraint::Unknown(other),
  }
}

/// One multi-array feature's shape constraint as CoreML reports it, before
/// classification.
///
/// [`FeatureInfo::from_parts`] takes this rather than a [`ShapeConstraint`], so
/// the verdict has exactly one producer — [`classify_shape_constraint`] — and a
/// unit-test fixture cannot state a verdict its own contents do not support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawShapeConstraint {
  raw_type: isize,
  enumerated_shapes: Vec<Vec<usize>>,
  axis_ranges: Vec<AxisRange>,
}

impl RawShapeConstraint {
  /// From `shapeConstraint.type`, `enumeratedShapes` and
  /// `sizeRangeForDimension`.
  pub(crate) const fn new(
    raw_type: isize,
    enumerated_shapes: Vec<Vec<usize>>,
    axis_ranges: Vec<AxisRange>,
  ) -> Self {
    Self {
      raw_type,
      enumerated_shapes,
      axis_ranges,
    }
  }
}

/// Shape/type info for one model input or output feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureInfo {
  name: String,
  shape: Vec<usize>,
  data_type: Option<DataType>,
  optional: bool,
  axis_ranges: Vec<AxisRange>,
  enumerated_shapes: Vec<Vec<usize>>,
  shape_constraint: Option<ShapeConstraint>,
}

impl FeatureInfo {
  /// Build one feature's snapshot, classifying `constraint` on the way in.
  ///
  /// `constraint` is `None` for a non-multi-array feature, which carries no
  /// shape constraint at all. Crate-internal so `model::contract`'s clause
  /// tests can drive [`check_load_contract`](contract::check_load_contract)
  /// over fixtures with no model present.
  pub(crate) fn from_parts(
    name: String,
    shape: Vec<usize>,
    data_type: Option<DataType>,
    optional: bool,
    constraint: Option<RawShapeConstraint>,
  ) -> Self {
    let (axis_ranges, enumerated_shapes, shape_constraint) = match constraint {
      None => (Vec::new(), Vec::new(), None),
      Some(raw) => {
        let verdict = classify_shape_constraint(
          raw.raw_type,
          &shape,
          &raw.enumerated_shapes,
          &raw.axis_ranges,
        );
        (raw.axis_ranges, raw.enumerated_shapes, Some(verdict))
      }
    };
    Self {
      name,
      shape,
      data_type,
      optional,
      axis_ranges,
      enumerated_shapes,
      shape_constraint,
    }
  }

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
  /// says the value is the only one — [`Self::axis_ranges`] is where that is
  /// stated per axis.
  #[inline(always)]
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// The raw per-axis size ranges (`sizeRangeForDimension`), one per axis;
  /// empty for a non-multi-array feature and for one whose constraint carries
  /// none.
  ///
  /// This is the per-AXIS statement [`Self::shape_constraint`] summarises for
  /// the whole feature, and it is what a load-time contract checks a
  /// dimension against: a pinned axis of size `d` reads `d`, and a
  /// `RangeDim(lower, upper)` axis reads `lower..=upper`. Trustworthy under
  /// [`ShapeConstraint::Fixed`] and [`ShapeConstraint::Range`]; under
  /// [`ShapeConstraint::Enumerated`] it reports the DEFAULT shape rather than
  /// the alternatives, which is measured and is why the enumerated arm of the
  /// classifier does not consult it.
  #[inline(always)]
  pub fn axis_ranges(&self) -> &[AxisRange] {
    &self.axis_ranges
  }

  /// The constraint's own list of accepted shapes; empty when it lists none.
  ///
  /// A fixed export lists exactly one — its declared shape — which is half of
  /// what makes [`ShapeConstraint::Fixed`] a measured fact rather than a
  /// reading of the type code.
  #[inline(always)]
  pub fn enumerated_shapes(&self) -> &[Vec<usize>] {
    &self.enumerated_shapes
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
  /// Assemble a description from already-snapshotted features.
  ///
  /// Crate-internal so `model::contract`'s clause tests can drive
  /// [`check_load_contract`](contract::check_load_contract) over fixtures with
  /// no model present — one fixture family for every door, rather than a fake
  /// per door.
  pub(crate) const fn from_parts(
    inputs: Vec<FeatureInfo>,
    outputs: Vec<FeatureInfo>,
    states: Vec<FeatureInfo>,
  ) -> Self {
    Self {
      inputs,
      outputs,
      states,
    }
  }

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
    let (shape, data_type, raw_constraint) = unsafe {
      description
        .multiArrayConstraint()
        .map_or((Vec::new(), None, None), |constraint| {
          let shape_constraint = constraint.shapeConstraint();
          let axis_ranges: Vec<AxisRange> = shape_constraint
            .sizeRangeForDimension()
            .iter()
            .map(|range| {
              let range = range.rangeValue();
              AxisRange::new(range.location, range.length)
            })
            .collect();
          let enumerated_shapes: Vec<Vec<usize>> = shape_constraint
            .enumeratedShapes()
            .iter()
            .map(|dims| dims.iter().map(|n| n.as_usize()).collect())
            .collect();
          (
            constraint.shape().iter().map(|n| n.as_usize()).collect(),
            Some(DataType::from_raw(constraint.dataType().0)),
            Some(RawShapeConstraint::new(
              shape_constraint.r#type().0,
              enumerated_shapes,
              axis_ranges,
            )),
          )
        })
    };
    // SAFETY: accessor send on a live feature description.
    let optional = unsafe { description.isOptional() };
    features.push(FeatureInfo::from_parts(
      name.to_string(),
      shape,
      data_type,
      optional,
      raw_constraint,
    ));
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
      description: ModelDescription::from_parts(inputs, outputs, states),
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

// A load contract has a consumer only in a build that compiles a door. With no
// door — `cargo build --features whisper --examples` in CI, or the default
// feature set, which pulls no pipeline at all — every item in this module is
// unused, and that is the feature set rather than rot. The list grows as
// coremlit #137 migrates the remaining eight doors onto it; the module itself
// stays compiled under every feature set, so a change that breaks it is caught
// everywhere rather than only where a door happens to be on.
#[cfg_attr(
  not(any(feature = "identity", feature = "face", test)),
  allow(dead_code, reason = "no door in this feature set holds a `Checked`")
)]
pub(crate) mod contract;

#[cfg(test)]
mod tests;
