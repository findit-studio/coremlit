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
  /// The axis admits exactly one size, whatever it is, and that size is not
  /// zero. The door READS the value back from [`FeatureInfo::shape`] after the
  /// check rather than requiring it — the shape of a door whose geometry is the
  /// artifact's.
  ///
  /// # Why the zero is refused HERE and not beside each door
  ///
  /// This is the one [`Dim`] whose number comes from the MODEL rather than from
  /// the contract, so it is the one where a degenerate declaration is the
  /// model's to make. And a zero-sized axis satisfies "exactly one size" —
  /// `(0, 1)` is a pinned axis — so nothing else in this checker refuses it.
  ///
  /// Every door that reads an axis back then ALLOCATES from the number:
  /// `audio::speaker`'s two frame counts size every mask row and every logit
  /// buffer, `audio::whisper`'s `kv_dim`/`max_token_context`/`vocab` size the KV
  /// caches, both attention masks and the logits gather, `embeddings::face`'s
  /// batch sizes the input tensor, and `embeddings::siglip`'s patch budget and
  /// token window size all three image tensors and the text one. A zero in any
  /// of them is a graph that loads clean and then computes nothing — which is
  /// why every one of those doors carried a hand-written `>= 1` beside its
  /// check before this clause existed. Beside the constructor that is a check a
  /// door can forget, which is the defect this whole type exists to close; here
  /// it is checked with everything else, once, where a door cannot skip it.
  ///
  /// Refused as [`ContractViolation::ZeroSizedAxis`] rather than as an ordinary
  /// [`ContractViolation::Axis`], because "the axis you left to me is empty" is
  /// a different sentence from "the axis you pinned reads a different size".
  //
  // `embeddings::face` is the producer the clause was specified for: that
  // door's geometry comes from a manifest read at load and its batch is the
  // ARTIFACT's. `audio::speaker`, `audio::whisper` and `embeddings::siglip`
  // joined it — both speaker doors' frame counts, every dimension that differs
  // across whisper's tiny, small and large-v3, and siglip's conversion-chosen
  // patch budget and token window. Still dead in a build with none of them,
  // which `--no-default-features` is: this module is always compiled and no
  // door is.
  #[cfg_attr(
    not(any(
      feature = "face",
      feature = "siglip",
      feature = "speaker",
      feature = "whisper"
    )),
    allow(dead_code, reason = "no door in this feature set reads an axis back")
  )]
  AnyFixed,
  /// The axis is deliberately symbolic, over exactly this range. The door
  /// varies the size within it on purpose, so a graph that pins the axis is as
  /// wrong as one that opens it wider.
  //
  // `audio::lid` is the producer: its `mel_features` time axis is `RangeDims`
  // BY DESIGN, because `lid::window` scores a ragged tail at its own length.
  #[cfg_attr(
    not(feature = "lid"),
    allow(dead_code, reason = "the lid door is this variant's only producer")
  )]
  Range(AxisRange),
}

impl core::fmt::Display for Dim {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::Exactly(size) => write!(f, "{size}"),
      Self::AnyFixed => f.write_str("any one non-zero fixed size"),
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
  /// # This is the only variant because it is the only MEASURED one
  ///
  /// Every `.mlmodelc` this repository stages was loaded and its
  /// [`ModelDescription::states`] read back (coremlit #137, PR B). All thirteen
  /// declare **none**:
  ///
  /// | artifact | `states()` |
  /// |---|---|
  /// | `silero-vad-unified-256ms-v6.2.1` (committed) | empty |
  /// | `wespeaker`, `wespeaker_v2`, `wespeaker_int8` | empty |
  /// | `pyannote_segmentation` | empty |
  /// | whisper `MelSpectrogram` / `AudioEncoder` / `TextDecoder`, × tiny, small, large-v3 | empty |
  /// | `SpeechBrainECAPAVoxLingua107` (lid) | empty |
  /// | published ReDimNet-B5 (`audio::identity`, probed in #136) | empty |
  ///
  /// The whisper `TextDecoder` is the one worth naming, because its shape
  /// invites the opposite guess: it is autoregressive and carries a KV cache
  /// across steps. That cache is **not** `MLState`. `key_cache` and
  /// `value_cache` are ordinary `[1, kv_dim, 1, max_token_context]` f16
  /// INPUTS, and `key_cache_updates` / `value_cache_updates` ordinary outputs;
  /// the host owns the buffers and appends one column per step, which is why
  /// `audio::whisper::backend::coreml` predicts through the stateless entry and
  /// is correct to.
  ///
  /// So a `Stateful(..)` variant would have no producer, no artifact to be
  /// checked against, and no measured shape for what it should carry — an arm
  /// added with a guess, which is exactly what issue #138 records this crate
  /// paying for repeatedly. It belongs here when a door needs one, and it
  /// arrives with the artifact that forces it.
  ///
  /// Until then the absence is load-bearing rather than incidental:
  /// [`Checked`] exposes NO stateful prediction entry at all, so a door holding
  /// one cannot call [`Model::predict_with_state`] — not by a runtime refusal
  /// but because the method does not exist on the type (`E0599`). Adding the
  /// variant is what would open that up, and it would then owe a typestate to
  /// close it again.
  None,
}

/// The complete set of facts a door requires of a model at load.
///
/// "Complete" over exactly the members of [`ModelDescription`] that can make an
/// otherwise-conformant prediction fail; that type's own documentation is the
/// table of what those are and what is deliberately dropped.
///
/// # What NAMING an output guarantees
///
/// The `outputs` list is not a filter over what the graph declares — it is the
/// list the door will READ, and [`Checked`] keeps it and hands it to every
/// prediction. So naming an output is three statements at once, and all three
/// are established at load:
///
///   1. the feature EXISTS, with the contract's element type and geometry;
///   2. the model declares it REQUIRED, so the graph does not carry a DECLARED
///      freedom to leave it out — an optional one is refused as
///      [`ContractViolation::OptionalOutput`], because every other clause here
///      is a statement about the declaration and none of them says anything
///      about the feature being in a RESULT;
///   3. it is the only kind of output that is materialised —
///      [`Checked::predict_with`] asks for exactly these names, which is what
///      lets an EXTRA output the graph declares be accepted (it cannot make a
///      prediction fail if nobody asks for it).
///
/// Those three rule out every reason a door's own `outputs.take(name)` could
/// come back empty that the DESCRIPTION could have shown at load: the feature
/// absent from the graph, optional in the graph, and unrequested at the call.
/// The doors still map a `None` to [`PredictionError::MissingOutput`] rather
/// than unwrapping, and correctly — what the contract removes is the declared
/// licence to omit, not a guarantee about the runtime.
///
/// The `inputs` list carries no equivalent optionality clause, deliberately.
/// The door SUPPLIES those, so an optional one is supplied anyway;
/// [`OptionalOutput`] carries the asymmetry.
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

/// A [`Dim::AnyFixed`] axis the model pins at size ZERO.
///
/// Separate from [`AxisMismatch`] because the two are different sentences. An
/// `AxisMismatch` says the model's axis is not the size the CONTRACT states;
/// this says the axis the contract deliberately left to the MODEL came back
/// empty. [`Dim::AnyFixed`] carries why that is the one degenerate declaration
/// no other clause can see, and what every door then allocates from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ZeroSizedAxis {
  feature: &'static str,
  axis: usize,
}

impl ZeroSizedAxis {
  const fn new(feature: &'static str, axis: usize) -> Self {
    Self { feature, axis }
  }

  /// The feature whose read-back axis is empty.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }

  /// What the contract states for that axis.
  pub(crate) fn expected(&self) -> String {
    format!("axis {} {}", self.axis, Dim::AnyFixed)
  }

  /// What the model declares for it.
  pub(crate) fn observed(&self) -> String {
    format!("axis {} 0", self.axis)
  }
}

/// An output the contract NAMES that the model declares OPTIONAL, so the graph
/// may leave it out of the very prediction the contract was checked to make
/// possible.
///
/// # Why this is a clause and the input direction is not
///
/// The two directions are not symmetric, and the asymmetry is about who
/// decides. A door SUPPLIES the inputs its contract names, so an input the
/// model merely permits to be absent is supplied anyway and its optionality
/// changes nothing — which is why an optional NAMED input is deliberately
/// accepted, and why the input clause below is the different one: a REQUIRED
/// input the contract does not name.
///
/// An output is the model's to produce. Every other per-feature clause is a
/// statement about a feature that IS declared — its element type, its rank, its
/// flexibility verdict, its axes — and all of them pass for an optional one,
/// because the feature really is there in the description. What none of them
/// says is that it will be there in a RESULT.
/// [`Checked::predict_with`] asks [`Model::predict_with_outputs`] for exactly
/// the contract's own names, so a graph that omits one answers
/// [`PredictionError::MissingOutput`] at predict time — on a contract whose
/// whole job was to establish at LOAD time that the prediction can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OptionalOutput {
  feature: &'static str,
}

impl OptionalOutput {
  const fn new(feature: &'static str) -> Self {
    Self { feature }
  }

  /// The output the door reads and the model may omit.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
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
  /// A [`Dim::AnyFixed`] axis the model pins at size ZERO — an axis the
  /// contract left to the model, come back empty.
  #[error(
    "feature `{}` declares {}, and the contract states {}; a door that READS an \
     axis back allocates from it, so an empty one loads clean and computes nothing",
    .0.feature(), .0.observed(), .0.expected()
  )]
  ZeroSizedAxis(ZeroSizedAxis),
  /// A named OUTPUT the model declares optional.
  #[error(
    "model declares the output `{}` OPTIONAL, and the contract names it as one the door reads; \
     a prediction that omits it satisfies the model and fails the door",
    .0.feature()
  )]
  OptionalOutput(OptionalOutput),
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

/// One clause about a NAMED feature, rendered for a door's own error type.
///
/// Payload of [`Rendered::Feature`]. Every per-feature clause — element type,
/// rank, flexibility, one axis, a zero-sized read-back axis, an optional output
/// — collapses to this triple, because every door's error vocabulary makes
/// exactly one distinction over them: which feature, what was wanted, what was
/// declared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FeatureRendering {
  feature: &'static str,
  expected: String,
  actual: String,
}

impl FeatureRendering {
  const fn new(feature: &'static str, expected: String, actual: String) -> Self {
    Self {
      feature,
      expected,
      actual,
    }
  }

  /// The feature the clause is about.
  pub(crate) const fn feature(&self) -> &'static str {
    self.feature
  }

  /// What the contract states.
  pub(crate) fn expected(self) -> String {
    self.expected
  }

  /// What the model declares.
  pub(crate) fn actual(self) -> String {
    self.actual
  }
}

/// A [`ContractViolation`] reduced to the THREE cases every door's error
/// vocabulary distinguishes.
///
/// # Why this exists rather than each door matching the violation directly
///
/// [`ContractViolation`] has one variant per clause, which is right for
/// diagnosis and wrong for the six doors that map it: each wrote an exhaustive
/// match that collapsed all but two arms into one, so ADDING a clause — the
/// zero-sized-axis one, say — broke every door at once, in six places that all
/// wanted the same new arm. This is the reduction they were all performing,
/// written once. A clause added later lands in [`Self::Feature`] and no door
/// changes.
///
/// The two it does NOT collapse are the two that are not about a declared
/// feature at all: they name something the door cannot SUPPLY, and every door
/// gives them error variants of their own for that reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Rendered {
  /// A named feature's declaration is not the contract's.
  Feature(FeatureRendering),
  /// A REQUIRED input the contract does not name.
  UnsatisfiableInput(String),
  /// A declared state buffer under [`StateContract::None`].
  UnsatisfiableState(String),
}

impl ContractViolation {
  /// Reduce to the three cases a door's error vocabulary distinguishes — see
  /// [`Rendered`].
  pub(crate) fn rendered(self) -> Rendered {
    let (feature, expected, actual) = match self {
      Self::UnsatisfiableInput(input) => {
        return Rendered::UnsatisfiableInput(input.name);
      }
      Self::UnsatisfiableState(state) => {
        return Rendered::UnsatisfiableState(state.name);
      }
      Self::Missing(missing) => (
        missing.feature(),
        "a declared feature".to_string(),
        "missing".to_string(),
      ),
      Self::DataType(mismatch) => (mismatch.feature(), mismatch.expected(), mismatch.observed()),
      Self::Rank(mismatch) => (mismatch.feature(), mismatch.expected(), mismatch.observed()),
      Self::Flexibility(mismatch) => (mismatch.feature(), mismatch.expected(), mismatch.observed()),
      Self::Axis(mismatch) => (mismatch.feature(), mismatch.expected(), mismatch.observed()),
      Self::ZeroSizedAxis(zero) => (zero.feature(), zero.expected(), zero.observed()),
      Self::OptionalOutput(output) => (
        output.feature(),
        "a required output".to_string(),
        "optional".to_string(),
      ),
    };
    Rendered::Feature(FeatureRendering::new(feature, expected, actual))
  }
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
///   - a [`Dim::AnyFixed`] axis the model pins at size ZERO, which is the one
///     degenerate declaration only the axis's own clause can see — the size
///     came from the model, not from the contract ([`Dim::AnyFixed`] carries
///     why);
///   - a named OUTPUT the model declares OPTIONAL, which the door reads and the
///     graph may omit — see [`OptionalOutput`] for why this direction is a
///     clause and the input direction deliberately is not;
///   - any REQUIRED input the contract does not name (an OPTIONAL extra passes
///     — CoreML runs a prediction that omits one, so only a required input the
///     door cannot fill makes the contract unsatisfiable);
///   - any declared state buffer under [`StateContract::None`].
///
/// **A named INPUT the model declares optional is ACCEPTED**, and that is a
/// decision rather than an omission: the door sends the inputs its contract
/// names, so an input that is merely permitted to be absent is sent anyway.
/// `a_named_input_the_model_declares_optional_is_accepted` pins it.
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
    let declared = description.output(feature.name);
    check_feature_contract(feature, declared)?;
    // The clause `check_feature_contract` cannot carry, because it is shared
    // with the inputs and the two directions differ — see [`OptionalOutput`].
    // `check_feature_contract` has already refused an absent feature as
    // `Missing`, so what reaches here is a declared one.
    if declared.is_some_and(FeatureInfo::is_optional) {
      return Err(ContractViolation::OptionalOutput(OptionalOutput::new(
        feature.name,
      )));
    }
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
    // The zero an `AnyFixed` axis may be pinned at is its own refusal: it
    // satisfies "exactly one size", and the number is the MODEL's rather than
    // the contract's, so no other clause here can see it.
    if *dim == Dim::AnyFixed && observed == Some(AxisRange::new(0, 1)) {
      return Err(ContractViolation::ZeroSizedAxis(ZeroSizedAxis::new(
        name, axis,
      )));
    }
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
/// Forwarded today, each landed with the caller that needed it:
/// [`Self::predict_with`], the borrowed-input prediction entry, which is the
/// whole of what `audio::identity` calls on a [`Model`] and therefore the whole
/// of what a stateless graph needs; and [`Self::description`], for a door that
/// means to READ a [`Dim::AnyFixed`] axis's value back
/// rather than require it — `embeddings::face`, whose batch is the artifact's
/// and not its own, and `audio::speaker` / `audio::whisper`, whose frame counts
/// and per-model-size dimensions are. Neither was added ahead of its caller, so
/// the exposed surface carries no method no contract has been written against.
///
/// [`Model::predict_with_state`] is the method the omission is LOAD-BEARING
/// for: no contract can state a stateful graph ([`StateContract`] has one
/// variant, and its doc carries the measurement), so no `Checked` should be
/// able to make a stateful prediction — and none can, because the method is not
/// on this type. Calling it is `E0599`, decided by the compiler rather than by
/// a runtime check on something already known at load.
///
/// # The prediction it forwards is SELECTIVE, and that is the contract's doing
///
/// [`Self::predict_with`] is not [`Model::predict_with`] with a check in front
/// of it: it materialises only the outputs the contract NAMES, by handing that
/// list to [`Model::predict_with_outputs`]. The contract is therefore not only
/// what was checked at load but what is asked for at every prediction, which is
/// what keeps an extra output — legal, and correctly accepted by
/// [`check_load_contract`] — from deciding whether a call succeeds. That method
/// carries the defect it closes.
#[derive(Debug)]
pub(crate) struct Checked {
  model: Model,
  /// The output features this value's contract NAMES — the only ones
  /// [`Self::predict_with`] materialises. Kept from the contract at
  /// construction, so the set the door declared and the set it reads back are
  /// one list and cannot drift.
  outputs: Vec<&'static str>,
}

impl Checked {
  /// Check `model` against `contract` and wrap it.
  ///
  /// # The output list this keeps is a list of REQUIRED features
  ///
  /// [`Self::outputs`] is taken from the contract here and handed to every
  /// prediction, so the names kept are precisely the names asked for. That is
  /// why [`check_load_contract`] refuses an output the model declares OPTIONAL:
  /// without that clause a description could pass every geometry check and
  /// still be free to omit the feature, and the omission would surface as
  /// [`PredictionError::MissingOutput`] — at predict time, from a door whose
  /// load had already succeeded. With the clause, the list this constructor
  /// stores names only features the model declares REQUIRED, so nothing in the
  /// description says the door may be handed a result without them.
  ///
  /// # Errors
  /// [`ContractViolation`] if the model does not satisfy `contract`; the model
  /// is dropped rather than returned, because the only thing this type says is
  /// that the check passed.
  pub(crate) fn new(model: Model, contract: &LoadContract) -> Result<Self, ContractViolation> {
    check_load_contract(model.description(), contract)?;
    Ok(Self {
      model,
      outputs: contract.outputs.iter().map(|output| output.name).collect(),
    })
  }

  /// The description of the model this value's contract was checked against.
  ///
  /// **Why a door reads it HERE rather than off the [`Model`] before the
  /// check.** [`Dim::AnyFixed`] is specified as an axis whose value the door
  /// reads back AFTERWARDS, and the two moments are not the same fact. Before
  /// the check [`FeatureInfo::shape`] can be the DEFAULT shape of a flexible
  /// feature — a `RangeDim` or enumerated graph reports one it will happily
  /// accept others beside. After it the feature is
  /// [`ShapeConstraint::Fixed`], which is what an `AnyFixed` axis requires, so
  /// the same number is a fact about the graph rather than a reading of its
  /// declaration — and the check has also refused a zero, which a raw
  /// [`FeatureInfo::shape`] read would have handed back.
  ///
  /// Its readers: `embeddings::face` (the batch its manifest does not state),
  /// `embeddings::siglip`'s two doors, `audio::speaker`'s two frame counts, and
  /// `audio::whisper`, whose seven per-model-size dimensions differ across
  /// tiny, small and large-v3 and are read rather than tabled.
  pub(crate) const fn description(&self) -> &ModelDescription {
    self.model.description()
  }

  /// Runs a synchronous prediction from borrowed inputs, materialising only
  /// the outputs this value's contract NAMES.
  ///
  /// # The door asks for exactly what it declared
  ///
  /// [`check_load_contract`] accepts an EXTRA output, and correctly: it is not
  /// a required input, so it cannot make a prediction fail — except that
  /// [`Model::predict_with`] converted every advertised output into a
  /// [`MultiArray`] before the door got to select its own. A graph carrying the
  /// contract's f32 tensor head beside a string, dictionary, image or sequence
  /// output therefore loaded clean and then failed EVERY prediction with
  /// [`PredictionError::NotMultiArray`], on a feature no door had asked for.
  ///
  /// The fix is here rather than as a load-time rule, because a rule would have
  /// to enumerate which output kinds the generic extraction path can represent
  /// — a list that is wrong the moment CoreML grows a kind — and would refuse
  /// artifacts that work. Asking for the contract's own names refuses nothing
  /// and materialises nothing extra, and every door reaching CoreML through a
  /// `Checked` gets it without a change of its own.
  ///
  /// # Errors
  /// As [`Model::predict_with_outputs`].
  pub(crate) fn predict_with(
    &self,
    inputs: &[(&str, &MultiArray)],
  ) -> Result<Features, PredictionError> {
    self.model.predict_with_outputs(inputs, &self.outputs)
  }
}
