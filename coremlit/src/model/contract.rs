//! The load-time contract a door states about the model it opens, and the
//! type that proves the contract was checked.
//!
//! # Why this is a type and not a function
//!
//! Every door in this crate opens a `.mlmodelc` and then predicts into it. What
//! makes a prediction possible is not the door's own care but a set of facts
//! about the artifact: the features it sends exist, carry the element type it
//! writes and the geometry it allocates for; no OTHER required input exists,
//! because the door supplies only its own; and no state buffer exists, because
//! the door predicts through the stateless API.
//!
//! Written as free functions those facts are checks a `load` can forget to
//! call, and deleting one from a door fails no runnable test — a door's
//! integration assertions need a staged artifact, and the model-gated ones are
//! `#[ignore]`d. That is the shape every review round on this crate has found:
//! a check that lives beside the value instead of inside its constructor.
//!
//! So the value gets one door. [`Checked`] wraps a [`Model`] and its ONLY
//! constructor is [`Checked::new`], which takes a [`LoadContract`] and runs
//! [`check_load_contract`]. A door holds a `Checked`, never a `Model`, so
//! deleting the check is a compile error rather than a survivable mutation.
//!
//! # The contract is data
//!
//! [`LoadContract`] is owned rather than `&'static` on purpose: a door whose
//! geometry comes from a manifest read at load builds its contract at load
//! too, and an axis whose value it means to READ back rather than require is
//! [`Dim::AnyFixed`].

use crate::{DataType, FeatureInfo, Features, Model, MultiArray, PredictionError, ShapeConstraint};

use super::{AxisRange, ModelDescription};

#[cfg(test)]
mod tests;

/// One axis of a feature's geometry, as the door states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Dim {
  /// The axis admits exactly this one size. The door allocates against the
  /// number, so nothing else can be accepted.
  Exactly(usize),
  /// The axis admits exactly one size, whatever it is. The door READS the
  /// value back from [`FeatureInfo::shape`] after the check rather than
  /// requiring it — the shape of a door configured by a manifest.
  // Constructed by the contract fixtures and by no door YET, so it is dead in a
  // shipped build. Its first producer is the face door's manifest-built
  // contract (coremlit #135 §4); the clause is specified and driven over
  // fixtures now so that door adopts this reading rather than adding a second
  // one. Drop the attribute when that lands.
  #[allow(dead_code, reason = "constructed by the face door in coremlit #135 §4")]
  AnyFixed,
  /// The axis is deliberately symbolic, over exactly this range. The door
  /// varies the size within it on purpose, so a graph that pins the axis is as
  /// wrong as one that opens it wider.
  // Likewise: `audio::lid` is its first producer, migrated in coremlit #137.
  // The clause is driven over a lid-shaped fixture now to prove lid is
  // expressible as a contract rather than as an exemption BEFORE that
  // migration. Drop the attribute when that lands.
  #[allow(dead_code, reason = "constructed by `audio::lid` in coremlit #137")]
  Range(AxisRange),
}

impl core::fmt::Display for Dim {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::Exactly(size) => write!(f, "{size}"),
      Self::AnyFixed => f.write_str("any one fixed size"),
      Self::Range(range) => write!(f, "{range}"),
    }
  }
}

/// What a door requires of one named input or output feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FeatureContract {
  name: &'static str,
  dtype: DataType,
  dims: Vec<Dim>,
}

impl FeatureContract {
  /// The feature `name` must exist, carry `dtype`, and have exactly `dims`
  /// axes matching them in order.
  pub(crate) const fn new(name: &'static str, dtype: DataType, dims: Vec<Dim>) -> Self {
    Self { name, dtype, dims }
  }

  /// The whole-feature verdict this contract's axes require.
  ///
  /// # Why the contract's own axes decide it
  ///
  /// [`FeatureInfo::axis_ranges`] is a true per-axis statement only under
  /// [`ShapeConstraint::Fixed`] and [`ShapeConstraint::Range`]; under
  /// [`ShapeConstraint::Enumerated`] it reports the DEFAULT shape and says
  /// nothing about the alternatives, and under the remaining verdicts it is
  /// absent. So the per-axis clauses below are only meaningful once the
  /// feature's verdict is one of those two, and WHICH of the two is decided by
  /// the contract:
  ///
  ///   - all axes [`Dim::Exactly`] / [`Dim::AnyFixed`] — the door needs a
  ///     graph with nothing symbolic anywhere, so the verdict must be
  ///     [`ShapeConstraint::Fixed`]. A `RangeDim(d, d)` graph declares this
  ///     contract's exact numbers and reports `(d, 1)` on every axis, so the
  ///     per-axis clauses alone would accept it — and a symbolic dimension is
  ///     what takes a graph off the accelerator, which for the identity door
  ///     is the entire reason its recipe pins a fixed shape.
  ///   - at least one [`Dim::Range`] axis — the door WANTS the graph flexible
  ///     there, so the verdict must be [`ShapeConstraint::Range`]. A fixed
  ///     graph cannot honour the range, and an enumerated one reports ranges
  ///     that are not its bounds.
  ///
  /// This is the reading of "an `Exactly`/`AnyFixed` axis whose feature is not
  /// `Fixed`" that also lets `audio::lid`'s
  /// `[Exactly(1), Range(10..=3001), Exactly(60)]` be a contract rather than an
  /// exemption: under a `Range` feature an axis reading `(d, 1)` still admits
  /// exactly `d`, which is all `Exactly(d)` claims about that axis.
  fn required_verdict(&self) -> ShapeConstraint {
    if self.dims.iter().any(|d| matches!(d, Dim::Range(_))) {
      ShapeConstraint::Range
    } else {
      ShapeConstraint::Fixed
    }
  }
}

/// What a door requires of the model's `MLState` buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StateContract {
  /// The model must declare none.
  ///
  /// A state buffer is not an input: it lives in its own dictionary
  /// ([`ModelDescription::states`]) and never appears among the ordinary
  /// inputs, so a stateful graph whose input and output sets are otherwise
  /// conformant clears every other clause. CoreML then requires an `MLState`
  /// on every prediction, which a door predicting through the stateless API
  /// never makes: the prediction fails, or the persistence the graph was built
  /// around is silently discarded.
  ///
  /// A `Stateful(..)` variant belongs here when a door needs one; until then
  /// the absent variant is a fact this crate has no measured shape for.
  None,
}

/// The complete set of facts a door requires of a model at load.
///
/// "Complete" over exactly the members of [`ModelDescription`] that can make an
/// otherwise-conformant prediction fail; that type's own documentation is the
/// table of what those are and what is deliberately dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadContract {
  inputs: Vec<FeatureContract>,
  outputs: Vec<FeatureContract>,
  state: StateContract,
}

impl LoadContract {
  /// The inputs the door sends, the outputs it reads, and what it requires of
  /// the state set.
  pub(crate) const fn new(
    inputs: Vec<FeatureContract>,
    outputs: Vec<FeatureContract>,
    state: StateContract,
  ) -> Self {
    Self {
      inputs,
      outputs,
      state,
    }
  }
}

/// The feature a violation is about, for a door mapping one into its own error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MissingFeature {
  feature: &'static str,
}

impl MissingFeature {
  const fn new(feature: &'static str) -> Self {
    Self { feature }
  }

  /// The feature the model does not declare.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }
}

/// A named feature whose declared element type is not the one the door writes
/// and reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DataTypeMismatch {
  feature: &'static str,
  expected: DataType,
  observed: Option<DataType>,
}

impl DataTypeMismatch {
  const fn new(feature: &'static str, expected: DataType, observed: Option<DataType>) -> Self {
    Self {
      feature,
      expected,
      observed,
    }
  }

  /// The feature whose element type mismatched.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The element type the contract states.
  pub(crate) fn expected(&self) -> String {
    self.expected.as_str().to_string()
  }

  /// The element type the model declares; `none` for a non-multi-array
  /// feature, which carries no element type at all.
  pub(crate) fn observed(&self) -> String {
    self
      .observed
      .map_or_else(|| "none".to_string(), |d| d.as_str().to_string())
  }
}

/// A named feature with a different number of axes than the contract states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RankMismatch {
  feature: &'static str,
  expected: usize,
  observed: usize,
}

impl RankMismatch {
  const fn new(feature: &'static str, expected: usize, observed: usize) -> Self {
    Self {
      feature,
      expected,
      observed,
    }
  }

  /// The feature whose rank mismatched.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }

  /// How many axes the contract states.
  pub(crate) fn expected(&self) -> String {
    format!("rank {}", self.expected)
  }

  /// How many axes the model declares.
  pub(crate) fn observed(&self) -> String {
    format!("rank {}", self.observed)
  }
}

/// A named feature whose whole-feature shape constraint is not the one its
/// contract's axes require — see [`FeatureContract::required_verdict`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlexibilityMismatch {
  feature: &'static str,
  expected: ShapeConstraint,
  observed: Option<ShapeConstraint>,
}

impl FlexibilityMismatch {
  const fn new(
    feature: &'static str,
    expected: ShapeConstraint,
    observed: Option<ShapeConstraint>,
  ) -> Self {
    Self {
      feature,
      expected,
      observed,
    }
  }

  /// The feature whose flexibility mismatched.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The shape constraint the contract's axes require.
  pub(crate) fn expected(&self) -> String {
    self.expected.to_string()
  }

  /// The shape constraint the model reports; `none` for a non-multi-array
  /// feature, which carries none.
  pub(crate) fn observed(&self) -> String {
    self
      .observed
      .map_or_else(|| "none".to_string(), |c| c.to_string())
  }
}

/// One axis of a named feature whose declared size range is not the one the
/// contract states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AxisMismatch {
  feature: &'static str,
  axis: usize,
  expected: Dim,
  observed: Option<AxisRange>,
}

impl AxisMismatch {
  const fn new(
    feature: &'static str,
    axis: usize,
    expected: Dim,
    observed: Option<AxisRange>,
  ) -> Self {
    Self {
      feature,
      axis,
      expected,
      observed,
    }
  }

  /// The feature whose axis mismatched.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The size range the contract states for this axis.
  pub(crate) fn expected(&self) -> String {
    format!("axis {} {}", self.axis, self.expected)
  }

  /// The size range the model declares for this axis; `none` when the
  /// constraint lists no range for it at all.
  pub(crate) fn observed(&self) -> String {
    match self.observed {
      Some(range) => format!("axis {} {range}", self.axis),
      None => format!("axis {} none", self.axis),
    }
  }
}

/// A REQUIRED input the contract does not name, so the door would never send
/// it and every prediction would fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnsatisfiableInput {
  name: String,
}

impl UnsatisfiableInput {
  const fn new(name: String) -> Self {
    Self { name }
  }

  /// The required input the door cannot fill.
  pub(crate) fn name(&self) -> &str {
    &self.name
  }
}

/// A declared `MLState` buffer under [`StateContract::None`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnsatisfiableState {
  name: String,
}

impl UnsatisfiableState {
  const fn new(name: String) -> Self {
    Self { name }
  }

  /// The state buffer the door cannot supply.
  pub(crate) fn name(&self) -> &str {
    &self.name
  }
}

/// A model that does not satisfy the [`LoadContract`] it was checked against.
///
/// One variant per clause of [`check_load_contract`], each naming the feature
/// it is about, so a door can map it into its own error vocabulary without
/// re-deriving what went wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub(crate) enum ContractViolation {
  /// The model declares no feature of this name.
  #[error("model declares no feature `{}`", .0.feature())]
  Missing(MissingFeature),
  /// A named feature's element type is not the contract's.
  #[error(
    "feature `{}` is {}, and the contract states {}",
    .0.feature(), .0.observed(), .0.expected()
  )]
  DataType(DataTypeMismatch),
  /// A named feature has a different number of axes than the contract states.
  #[error(
    "feature `{}` has {}, and the contract states {}",
    .0.feature(), .0.observed(), .0.expected()
  )]
  Rank(RankMismatch),
  /// A named feature's whole-feature shape constraint is not the one its
  /// axes require.
  #[error(
    "feature `{}` is {}, and the contract's axes require {}",
    .0.feature(), .0.observed(), .0.expected()
  )]
  Flexibility(FlexibilityMismatch),
  /// One axis of a named feature admits a different set of sizes.
  #[error(
    "feature `{}` declares {}, and the contract states {}",
    .0.feature(), .0.observed(), .0.expected()
  )]
  Axis(AxisMismatch),
  /// A REQUIRED input the contract does not name.
  #[error(
    "model declares a required input `{}` the contract does not name, so every \
     prediction would fail",
    .0.name()
  )]
  UnsatisfiableInput(UnsatisfiableInput),
  /// A declared state buffer under [`StateContract::None`].
  #[error(
    "model declares the state buffer `{}`, and the contract states none",
    .0.name()
  )]
  UnsatisfiableState(UnsatisfiableState),
}

/// Check `description` against `contract`, refusing on the first clause it
/// fails.
///
/// # What is refused
///
///   - a named input or output the model does not declare;
///   - a named feature whose element type is not the contract's;
///   - a named feature whose rank is not the contract's;
///   - a feature whose whole-feature shape constraint is not the one its
///     contract's axes require ([`FeatureContract::required_verdict`] carries
///     the rule and why the contract decides it);
///   - a [`Dim::Exactly`] axis that does not read exactly that one size, a
///     [`Dim::AnyFixed`] axis that admits more than one, or a [`Dim::Range`]
///     axis whose declared range is not the stated one;
///   - any REQUIRED input the contract does not name (an OPTIONAL extra passes
///     — CoreML runs a prediction that omits one, so only a required input the
///     door cannot fill makes the contract unsatisfiable);
///   - any declared state buffer under [`StateContract::None`].
///
/// # Errors
/// [`ContractViolation`], naming the feature and the clause.
pub(crate) fn check_load_contract(
  description: &ModelDescription,
  contract: &LoadContract,
) -> Result<(), ContractViolation> {
  for feature in &contract.inputs {
    check_feature_contract(feature, description.input(feature.name))?;
  }
  for feature in &contract.outputs {
    check_feature_contract(feature, description.output(feature.name))?;
  }

  // The inputs the door does NOT send. `snapshot_features` sorts by name, so
  // the offender reported here is stable across loads rather than an artefact
  // of CoreML's dictionary order.
  for declared in description.inputs() {
    if !declared.is_optional()
      && !contract
        .inputs
        .iter()
        .any(|feature| feature.name == declared.name())
    {
      return Err(ContractViolation::UnsatisfiableInput(
        UnsatisfiableInput::new(declared.name().to_string()),
      ));
    }
  }

  match contract.state {
    StateContract::None => {
      if let Some(state) = description.states().first() {
        return Err(ContractViolation::UnsatisfiableState(
          UnsatisfiableState::new(state.name().to_string()),
        ));
      }
    }
  }

  Ok(())
}

/// One feature's clauses, in the order [`check_load_contract`] documents.
fn check_feature_contract(
  contract: &FeatureContract,
  declared: Option<&FeatureInfo>,
) -> Result<(), ContractViolation> {
  let name = contract.name;
  let Some(declared) = declared else {
    return Err(ContractViolation::Missing(MissingFeature::new(name)));
  };

  if declared.data_type() != Some(contract.dtype) {
    return Err(ContractViolation::DataType(DataTypeMismatch::new(
      name,
      contract.dtype,
      declared.data_type(),
    )));
  }

  if declared.shape().len() != contract.dims.len() {
    return Err(ContractViolation::Rank(RankMismatch::new(
      name,
      contract.dims.len(),
      declared.shape().len(),
    )));
  }

  let required = contract.required_verdict();
  if declared.shape_constraint() != Some(required) {
    return Err(ContractViolation::Flexibility(FlexibilityMismatch::new(
      name,
      required,
      declared.shape_constraint(),
    )));
  }

  for (axis, dim) in contract.dims.iter().enumerate() {
    // Absent when the constraint lists fewer ranges than the shape has axes;
    // `None` fails every arm below, which is the fail-closed reading.
    let observed = declared.axis_ranges().get(axis).copied();
    let satisfied = match *dim {
      Dim::Exactly(size) => observed == Some(AxisRange::new(size, 1)),
      Dim::AnyFixed => observed.is_some_and(|range| range.count() == 1),
      Dim::Range(range) => observed == Some(range),
    };
    if !satisfied {
      return Err(ContractViolation::Axis(AxisMismatch::new(
        name, axis, *dim, observed,
      )));
    }
  }

  Ok(())
}

/// A [`Model`] that has been checked against a [`LoadContract`].
///
/// # The only door
///
/// [`Self::new`] is the ONLY constructor, and there is no accessor that hands
/// back the wrapped [`Model`]. A door's field is therefore a `Checked`, and
/// removing the contract check from that door does not compile.
///
/// # The exposed surface is the contract's, not the model's
///
/// This deliberately does NOT `Deref` to [`Model`]. A `&Model` cannot un-check
/// anything, so the reason is not safety but scope: `Deref` would make the
/// exposed surface open-ended, and a later door would silently gain methods the
/// contract has nothing to say about. Which prediction entry is right is a
/// function of the contract — under [`StateContract::None`],
/// [`Model::predict_with_state`] is incoherent, and a future `Stateful(..)`
/// contract would want the opposite pair — so every forwarded method is a
/// decision recorded here.
///
/// Forwarded today: [`Self::predict_with`] alone, the borrowed-input
/// prediction entry, which is the whole of what the identity door calls on a
/// [`Model`] and therefore the whole of what a stateless graph needs. A door
/// that means to read a [`Dim::AnyFixed`] axis's value back wants a
/// `description` accessor here; it is added with its first caller rather than
/// ahead of one, so the exposed surface never carries a method no contract has
/// been written against.
#[derive(Debug)]
pub(crate) struct Checked {
  model: Model,
}

impl Checked {
  /// Check `model` against `contract` and wrap it.
  ///
  /// # Errors
  /// [`ContractViolation`] if the model does not satisfy `contract`; the model
  /// is dropped rather than returned, because the only thing this type says is
  /// that the check passed.
  pub(crate) fn new(model: Model, contract: &LoadContract) -> Result<Self, ContractViolation> {
    check_load_contract(model.description(), contract)?;
    Ok(Self { model })
  }

  /// Runs a synchronous prediction from borrowed inputs.
  ///
  /// # Errors
  /// As [`Model::predict_with`].
  pub(crate) fn predict_with(
    &self,
    inputs: &[(&str, &MultiArray)],
  ) -> Result<Features, PredictionError> {
    self.model.predict_with(inputs)
  }
}
