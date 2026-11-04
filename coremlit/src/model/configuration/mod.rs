//! What a load sends to a fresh `MLModelConfiguration` beyond its compute
//! units, and the probes that decide it.
//!
//! # Probe, then send
//!
//! The selectors sent here arrived in later macOS releases than the oldest one
//! coremlit runs on, and coremlit declares no deployment floor. Sending a
//! selector the runtime does not implement raises `NSInvalidArgumentException`
//! (`doesNotRecognizeSelector:`), and an Objective-C exception that reaches
//! Rust aborts the process. So nothing is sent until its receiver has answered
//! for it: the discipline [`Model::supports_state`] and the state snapshot
//! already follow.
//!
//! | send | receiver | from |
//! |---|---|---|
//! | `setFunctionName:` | `MLModelConfiguration` | macOS 15.0 |
//! | `setAllowLowPrecisionAccumulationOnGPU:` | `MLModelConfiguration` | macOS 10.14 |
//! | `setOptimizationHints:` | `MLModelConfiguration` | macOS 14.4 |
//! | `setSpecializationStrategy:` | `MLOptimizationHints` | macOS 15.0 |
//!
//! The last two are one choice, [`Specialization`], and they are probed in that
//! order. `MLOptimizationHints` arrived with `setOptimizationHints:`, and
//! looking up a class the runtime lacks panics, so the class is not touched
//! until the configuration has answered `setOptimizationHints:`. Its own
//! `setSpecializationStrategy:` arrived later than the class (macOS 15.0
//! against 14.4), so it is probed on its own.
//!
//! The decision is made in full before anything is sent. [`plan_sends`] is a
//! pure function of the request and of the probes' answers, and [`apply`] sends
//! only what it returned. A default is never sent and never probed, so a load
//! that asks for nothing new sends exactly what a load sent before these
//! choices existed. The laws drive [`plan_sends`] with a stubbed responder,
//! which is how an older OS's refusals are reached on a host that answers every
//! probe.
//!
//! [`Model::supports_state`]: crate::Model::supports_state

use objc2::runtime::Sel;
use objc2_core_ml::{MLModelConfiguration, MLOptimizationHints, MLSpecializationStrategy};
use objc2_foundation::NSString;

use crate::{ComputeUnits, LoadError};

/// The name of one function of a multifunction model, as it crosses into
/// Core ML. Borrowed for the load, and never empty.
///
/// An ML Program may carry several functions over one set of weights; each
/// loaded `MLModel` runs exactly one of them. [`Model::load_function`] takes
/// this name; [`Model::load`] loads the model's default function.
///
/// The check is the one a type can make: an empty string names no function.
/// Whether the model HAS a function of this name is Core ML's to say at load,
/// and it says so as [`LoadError::Native`].
///
/// ```
/// use coremlit::FunctionName;
///
/// assert_eq!(FunctionName::new("embed_128").map(|f| f.as_str()), Some("embed_128"));
/// assert_eq!(FunctionName::new(""), None);
/// ```
///
/// [`Model::load_function`]: crate::Model::load_function
/// [`Model::load`]: crate::Model::load
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionName<'a>(&'a str);

impl<'a> FunctionName<'a> {
  /// The function `name`, or `None` for the empty string, which names no
  /// function.
  #[inline]
  pub const fn new(name: &'a str) -> Option<Self> {
    if name.is_empty() {
      None
    } else {
      Some(Self(name))
    }
  }

  /// The name, exactly as it is sent to Core ML.
  #[inline]
  pub const fn as_str(&self) -> &'a str {
    self.0
  }
}

/// How Core ML specializes a model's compute graph for the device it runs on:
/// `MLSpecializationStrategy`, carried by the configuration's
/// `optimizationHints`.
///
/// # Wire form
///
/// Under the `serde` feature, the snake_case string [`Self::as_str`] returns:
/// `"default"` or `"fast_prediction"`. [`Display`](core::fmt::Display) prints
/// the same word.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum Specialization {
  /// `MLSpecializationStrategyDefault`, Core ML's own default: "the strategy
  /// that works well for most applications". Never sent.
  #[default]
  Default,
  /// `MLSpecializationStrategyFastPrediction`. Apple's documentation: "Prefer
  /// the prediction latency at the potential cost of specialization time,
  /// memory footprint, and the disk space usage of specialized artifacts."
  /// Requires macOS 15.0.
  FastPrediction,
}

impl Specialization {
  /// Stable snake_case name of the variant.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Default => "default",
      Self::FastPrediction => "fast_prediction",
    }
  }

  #[inline]
  const fn to_raw(self) -> MLSpecializationStrategy {
    match self {
      Self::Default => MLSpecializationStrategy::Default,
      Self::FastPrediction => MLSpecializationStrategy::FastPrediction,
    }
  }
}

/// A [`LoadOptions`] choice whose non-default value the OS may be unable to
/// honour: the payload of [`LoadError::OptionUnsupported`].
///
/// [`Display`](core::fmt::Display) prints [`Self::as_str`], the option's key in
/// [`LoadOptions`]' wire form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum LoadOption {
  /// [`LoadOptions::low_precision_gpu_accumulation`]:
  /// `allowLowPrecisionAccumulationOnGPU`, from macOS 10.14.
  LowPrecisionGpuAccumulation,
  /// [`LoadOptions::specialization`]: `optimizationHints` (macOS 14.4) and its
  /// `specializationStrategy` (macOS 15.0).
  Specialization,
}

impl LoadOption {
  /// The option's key in [`LoadOptions`]' wire form.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::LowPrecisionGpuAccumulation => "low_precision_gpu_accumulation",
      Self::Specialization => "specialization",
    }
  }
}

/// Every choice a load makes through `MLModelConfiguration`, besides the
/// function: the compute units, low-precision accumulation on the GPU, and the
/// specialization strategy (rust-options-pattern).
///
/// The compute units are required, as [`Model::load`]'s `units` argument is.
/// The other two default to Core ML's own defaults, and a default is never
/// sent: [`Model::load_with`] over `LoadOptions::new(units)` sends exactly what
/// [`Model::load`] sends. A non-default value is sent only after the OS has
/// answered for its selector, and one the OS cannot honour is refused by name
/// as [`LoadError::OptionUnsupported`], never silently dropped.
///
/// Two of Core ML's choices are deliberately absent: `reshapeFrequency`, which
/// applies only to flexible-shape models, and `preferredMetalDevice`, since
/// coremlit's one target, `aarch64-apple-darwin`, has one GPU.
///
/// # Wire form
///
/// Under the `serde` feature, a document with the three keys, `compute`
/// required and the other two defaulted; an unknown key is refused by name, so
/// a misspelled option is an error rather than a silent default:
///
/// ```json
/// {"compute":"cpu_and_gpu","low_precision_gpu_accumulation":false,"specialization":"default"}
/// ```
///
/// `deny_unknown_fields` and `flatten` do not compose, so a configuration type
/// that carries these options nests them under a key of its own.
///
/// [`Model::load`]: crate::Model::load
/// [`Model::load_with`]: crate::Model::load_with
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LoadOptions {
  compute: ComputeUnits,
  #[cfg_attr(feature = "serde", serde(default))]
  low_precision_gpu_accumulation: bool,
  #[cfg_attr(feature = "serde", serde(default))]
  specialization: Specialization,
}

impl LoadOptions {
  /// Options on `compute`, with Core ML's defaults for everything else: no
  /// low-precision accumulation, [`Specialization::Default`].
  #[inline]
  pub const fn new(compute: ComputeUnits) -> Self {
    Self {
      compute,
      low_precision_gpu_accumulation: false,
      specialization: Specialization::Default,
    }
  }

  /// Which hardware Core ML may schedule the model on.
  #[inline]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }

  /// Sets [`Self::compute`] in place.
  #[inline]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }

  /// Whether the GPU may accumulate in reduced precision
  /// (`allowLowPrecisionAccumulationOnGPU`). `false`, Core ML's default.
  ///
  /// Changes GPU numerics, and so the values a model produces on the GPU.
  #[inline]
  pub const fn low_precision_gpu_accumulation(&self) -> bool {
    self.low_precision_gpu_accumulation
  }

  /// Builder form of [`Self::set_low_precision_gpu_accumulation`].
  #[must_use]
  #[inline]
  pub const fn with_low_precision_gpu_accumulation(mut self, allow: bool) -> Self {
    self.set_low_precision_gpu_accumulation(allow);
    self
  }

  /// Sets [`Self::low_precision_gpu_accumulation`] in place.
  #[inline]
  pub const fn set_low_precision_gpu_accumulation(&mut self, allow: bool) -> &mut Self {
    self.low_precision_gpu_accumulation = allow;
    self
  }

  /// How Core ML specializes the graph for the device.
  /// [`Specialization::Default`], Core ML's default.
  #[inline]
  pub const fn specialization(&self) -> Specialization {
    self.specialization
  }

  /// Builder form of [`Self::set_specialization`].
  #[must_use]
  #[inline]
  pub const fn with_specialization(mut self, specialization: Specialization) -> Self {
    self.set_specialization(specialization);
    self
  }

  /// Sets [`Self::specialization`] in place.
  #[inline]
  pub const fn set_specialization(&mut self, specialization: Specialization) -> &mut Self {
    self.specialization = specialization;
    self
  }
}

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a major bump.**
///
/// `key=value` pairs in declaration order, joined by `,`, each value in its own
/// vocabulary's spelling. `LoadOptions::new(ComputeUnits::CpuAndGpu)` prints
/// `compute=cpu_and_gpu,low_precision_gpu_accumulation=false,specialization=default`.
/// A door that nests these options nests the whole spelling in parentheses.
impl core::fmt::Display for LoadOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "compute={},low_precision_gpu_accumulation={},specialization={}",
      self.compute, self.low_precision_gpu_accumulation, self.specialization
    )
  }
}

/// A runtime class a load sends to, as [`plan_sends`] asks about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Receiver {
  /// The load's own `MLModelConfiguration`.
  Configuration,
  /// The `MLOptimizationHints` class. Asked about only after the
  /// configuration answered `setOptimizationHints:`, which arrived with the
  /// class: looking up a class the runtime lacks panics.
  OptimizationHints,
}

/// The sends a load makes to its configuration beyond the compute units. Only
/// [`plan_sends`] builds one, so every send it holds was probed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sends<'a> {
  /// `setFunctionName:` with this name, or no send at all: a load without a
  /// name leaves the configuration's `functionName` nil, which is Core ML's
  /// own "the default function".
  function: Option<FunctionName<'a>>,
  /// `setAllowLowPrecisionAccumulationOnGPU:` with `YES`, or no send at all
  /// (the default, `NO`, is never sent).
  low_precision_gpu_accumulation: bool,
  /// An `MLOptimizationHints` carrying this strategy, set as the
  /// configuration's `optimizationHints`; `None` for the default, which is
  /// never sent.
  specialization: Option<Specialization>,
}

#[cfg(test)]
impl<'a> Sends<'a> {
  /// The function `setFunctionName:` selects, if it is sent at all.
  const fn function(&self) -> Option<FunctionName<'a>> {
    self.function
  }

  /// Whether `setAllowLowPrecisionAccumulationOnGPU:` is sent (with `YES`).
  const fn low_precision_gpu_accumulation(&self) -> bool {
    self.low_precision_gpu_accumulation
  }

  /// The strategy the optimization hints are sent with, if they are sent.
  const fn specialization(&self) -> Option<Specialization> {
    self.specialization
  }
}

/// Decides every send a load makes, asking `responds` whether a receiver
/// answers each selector before deciding to send it.
///
/// A default asks nothing: no function name, no low-precision accumulation and
/// [`Specialization::Default`] probe no selector, so the plain [`Model::load`]
/// path sends exactly what it sent before these choices existed.
///
/// [`Specialization::FastPrediction`] asks the configuration about
/// `setOptimizationHints:` FIRST, and the `MLOptimizationHints` class about
/// `setSpecializationStrategy:` only after a yes: the class arrived with the
/// configuration's selector, and the production responder answers a
/// [`Receiver::OptimizationHints`] question by looking the class up.
///
/// # Errors
/// [`LoadError::FunctionsUnsupported`] when a function is named and the
/// configuration does not answer `setFunctionName:` (before macOS 15);
/// [`LoadError::OptionUnsupported`] naming the option when a non-default value
/// is asked for and a receiver does not answer its selector.
///
/// [`Model::load`]: crate::Model::load
fn plan_sends<'a>(
  function: Option<FunctionName<'a>>,
  options: &LoadOptions,
  mut responds: impl FnMut(Receiver, Sel) -> bool,
) -> Result<Sends<'a>, LoadError> {
  if function.is_some() && !responds(Receiver::Configuration, objc2::sel!(setFunctionName:)) {
    return Err(LoadError::FunctionsUnsupported);
  }
  let low_precision_gpu_accumulation = options.low_precision_gpu_accumulation();
  if low_precision_gpu_accumulation
    && !responds(
      Receiver::Configuration,
      objc2::sel!(setAllowLowPrecisionAccumulationOnGPU:),
    )
  {
    return Err(LoadError::OptionUnsupported(
      LoadOption::LowPrecisionGpuAccumulation,
    ));
  }
  let specialization = match options.specialization() {
    Specialization::Default => None,
    chosen @ Specialization::FastPrediction => {
      // `||` short-circuits: the class is asked about only once the
      // configuration has shown it exists.
      if !responds(Receiver::Configuration, objc2::sel!(setOptimizationHints:))
        || !responds(
          Receiver::OptimizationHints,
          objc2::sel!(setSpecializationStrategy:),
        )
      {
        return Err(LoadError::OptionUnsupported(LoadOption::Specialization));
      }
      Some(chosen)
    }
  };
  Ok(Sends {
    function,
    low_precision_gpu_accumulation,
    specialization,
  })
}

/// Probes for every send `function` and `options` need, then makes them.
///
/// # Errors
/// As [`plan_sends`]; nothing is sent when it refuses.
pub(super) fn configure(
  configuration: &MLModelConfiguration,
  function: Option<FunctionName<'_>>,
  options: &LoadOptions,
) -> Result<(), LoadError> {
  use objc2::{ClassType, runtime::NSObjectProtocol};
  let sends = plan_sends(function, options, |receiver, selector| match receiver {
    // The configuration's own class answers, so a "no" means this OS's
    // `MLModelConfiguration` lacks the method. The configuration is live: the
    // caller created it and holds it for the whole call.
    Receiver::Configuration => configuration.respondsToSelector(selector),
    // `plan_sends` asks this only after the configuration answered
    // `setOptimizationHints:`, so the class exists and `class()` finds it
    // rather than panicking. `class_respondsToSelector` answers for instances
    // without sending anything.
    Receiver::OptimizationHints => MLOptimizationHints::class().responds_to(selector),
  })?;
  // SAFETY: `sends` is `plan_sends`'s answer for this very `configuration`,
  // planned on the line above from its own answers and those of this
  // process's `MLOptimizationHints` class, which is `apply`'s whole contract.
  unsafe { apply(configuration, &sends) };
  Ok(())
}

/// Makes the sends `sends` holds.
///
/// # Safety
///
/// `sends` must be [`plan_sends`]'s answer for `configuration`: every selector
/// it asks for must have been answered YES by its receiver, the configuration's
/// own `respondsToSelector:` or the `MLOptimizationHints` class's
/// `class_respondsToSelector`, the latter asked only after the configuration
/// answered `setOptimizationHints:`.
unsafe fn apply(configuration: &MLModelConfiguration, sends: &Sends<'_>) {
  if let Some(function) = sends.function {
    let name = NSString::from_str(function.as_str());
    // SAFETY: the configuration answered `setFunctionName:` (this function's
    // contract). Both objects are live for the call: the caller holds the
    // configuration, and `name` is owned here. The property is declared
    // `copy`, so Core ML keeps its own copy of the string and nothing
    // borrowed outlives the send.
    unsafe { configuration.setFunctionName(Some(&name)) };
  }
  if sends.low_precision_gpu_accumulation {
    // SAFETY: the configuration answered
    // `setAllowLowPrecisionAccumulationOnGPU:` (this function's contract); a
    // `BOOL` setter on a configuration the caller holds live.
    unsafe { configuration.setAllowLowPrecisionAccumulationOnGPU(true) };
  }
  if let Some(specialization) = sends.specialization {
    // SAFETY: the `MLOptimizationHints` class exists. The configuration
    // answered `setOptimizationHints:` before the class was first looked up
    // (this function's contract), and the class arrived with that selector, so
    // the lookup inside `new` finds it. `new` is `NSObject`'s, and returns an
    // owned, initialized instance.
    let hints = unsafe { MLOptimizationHints::new() };
    // SAFETY: the class answered `setSpecializationStrategy:` (this function's
    // contract; it arrived in macOS 15.0, after the class); a plain enum setter
    // on the instance owned just above.
    unsafe { hints.setSpecializationStrategy(specialization.to_raw()) };
    // SAFETY: the configuration answered `setOptimizationHints:` (this
    // function's contract). Both objects are live for the call. The property is
    // declared `copy`, so Core ML keeps its own copy of the hints and `hints`
    // need not outlive the send.
    unsafe { configuration.setOptimizationHints(&hints) };
  }
}

#[cfg(test)]
mod tests;
