//! What a `prewarm` did: per function it warmed, the load and the first
//! prediction, which carries Core ML's device specialization when it is cold.

use core::time::Duration;

use super::error::Function;

/// One warmed function: its load and its first, throwaway prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Warmed {
  function: Option<Function>,
  load: Duration,
  first_prediction: Duration,
}

impl Warmed {
  /// The record of `function` (`None` for the audio bundle's one function).
  #[inline]
  pub(crate) const fn new(function: Option<Function>, load: Duration, first: Duration) -> Self {
    Self {
      function,
      load,
      first_prediction: first,
    }
  }

  /// The function, or `None` for a one-function bundle.
  #[inline]
  pub const fn function(&self) -> Option<Function> {
    self.function
  }

  /// How long Core ML took to load it.
  #[inline]
  pub const fn load(&self) -> Duration {
    self.load
  }

  /// How long its first prediction took: the device specialization, when it
  /// was cold, and one inference.
  #[inline]
  pub const fn first_prediction(&self) -> Duration {
    self.first_prediction
  }
}

/// What a `prewarm` did, in the order it warmed the functions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrewarmReport {
  warmed: Vec<Warmed>,
}

impl PrewarmReport {
  /// The record of every function warmed, in order.
  #[inline]
  pub fn warmed(&self) -> &[Warmed] {
    &self.warmed
  }

  /// Appends one function's record.
  #[inline]
  pub(crate) fn push(&mut self, warmed: Warmed) {
    self.warmed.push(warmed);
  }
}
