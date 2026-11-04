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
//! `respondsToSelector:` for it: the discipline [`Model::supports_state`] and
//! the state snapshot already follow.
//!
//! The decision is made in full before anything is sent. [`plan_sends`] is a
//! pure function of the request and of the probes' answers, and [`apply`] sends
//! only what it returned. The laws drive [`plan_sends`] with a stubbed
//! responder, which is how an older OS's refusals are reached on a host that
//! answers every probe.
//!
//! [`Model::supports_state`]: crate::Model::supports_state

use objc2::runtime::Sel;
use objc2_core_ml::MLModelConfiguration;
use objc2_foundation::NSString;

use crate::LoadError;

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

/// The sends a load makes to its configuration beyond the compute units. Only
/// [`plan_sends`] builds one, so every send it holds was probed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sends<'a> {
  /// `setFunctionName:` with this name, or no send at all: a load without a
  /// name leaves the configuration's `functionName` nil, which is Core ML's
  /// own "the default function".
  function: Option<FunctionName<'a>>,
}

#[cfg(test)]
impl<'a> Sends<'a> {
  /// The function `setFunctionName:` selects, if it is sent at all.
  const fn function(&self) -> Option<FunctionName<'a>> {
    self.function
  }
}

/// Decides every send a load makes, asking `responds` whether the
/// configuration answers each selector before deciding to send it.
///
/// A request for nothing asks nothing: a load without a function name probes
/// no selector, so the plain [`Model::load`] path sends exactly what it sent
/// before function loading existed.
///
/// # Errors
/// [`LoadError::FunctionsUnsupported`] when a function is named and the
/// configuration does not answer `setFunctionName:` (before macOS 15).
///
/// [`Model::load`]: crate::Model::load
fn plan_sends<'a>(
  function: Option<FunctionName<'a>>,
  mut responds: impl FnMut(Sel) -> bool,
) -> Result<Sends<'a>, LoadError> {
  if function.is_some() && !responds(objc2::sel!(setFunctionName:)) {
    return Err(LoadError::FunctionsUnsupported);
  }
  Ok(Sends { function })
}

/// Probes `configuration` for every send `function` needs, then makes them.
///
/// # Errors
/// As [`plan_sends`]; nothing is sent when it refuses.
pub(super) fn configure(
  configuration: &MLModelConfiguration,
  function: Option<FunctionName<'_>>,
) -> Result<(), LoadError> {
  use objc2::runtime::NSObjectProtocol;
  // The probe. `respondsToSelector:` is answered by the configuration's own
  // class, so a "no" means this OS's `MLModelConfiguration` lacks the method
  // and sending it would raise an exception. The configuration is live: the
  // caller created it and holds it for the whole call.
  let sends = plan_sends(function, |selector| {
    configuration.respondsToSelector(selector)
  })?;
  // SAFETY: `sends` is `plan_sends`'s answer for this very `configuration`,
  // planned on the line above from its own `respondsToSelector:` answers,
  // which is `apply`'s whole contract.
  unsafe { apply(configuration, &sends) };
  Ok(())
}

/// Makes the sends `sends` holds.
///
/// # Safety
///
/// `sends` must be [`plan_sends`]'s answer for `configuration`: every selector
/// it asks for must have been answered YES by `configuration`'s own
/// `respondsToSelector:`.
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
}

#[cfg(test)]
mod tests;
