//! Opaque state for stateful models (MLState, macOS 15+).

use core::{cell::Cell, marker::PhantomData};

use objc2::rc::Retained;
use objc2_core_ml::MLState;

/// Opaque per-session state for a stateful model.
///
/// Created by [`Model::make_state`](crate::Model::make_state); mutated in
/// place by [`Model::predict_with_state`](crate::Model::predict_with_state).
///
/// # Concurrency
///
/// `State` is [`Send`] but deliberately **not** [`Sync`]: Apple requires
/// stateful predictions sharing an `MLState` to be serialized, and
/// [`Model::predict_with_state`](crate::Model::predict_with_state) enforces
/// that through `&mut State` exclusivity.
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<coremlit::State>();
/// ```
#[derive(Debug)]
pub struct State {
  inner: Retained<MLState>,
  // `MLState` is marked both `Send` and `Sync` in objc2-core-ml (unlike
  // `MLMultiArray`, which carries neither), so `Retained<MLState>` is
  // already `Sync` and this struct would auto-derive `Sync` from `inner`
  // alone. Apple documents that concurrent stateful predictions sharing one
  // `MLState` are undefined behavior ("each ... prediction that uses the
  // same MLState must be serialized"), and a future `&self` accessor
  // mirroring `MLState::getMultiArrayForStateNamed_handler` (itself `&self`
  // despite mutating the buffer, per the ObjC binding) would let two
  // threads race on the buffer if `&State` could be shared. `Cell<()>` is
  // `Send` but never `Sync`, so this marker blocks the auto-derived `Sync`
  // without affecting `Send`.
  _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: exclusive mutation is enforced via &mut in predict_with_state;
// ownership transfer across threads is sound. Every field here is already
// `Send`, so this impl documents the invariant explicitly rather than
// leaving it to auto-derivation.
unsafe impl Send for State {}

impl State {
  pub(crate) fn from_raw(inner: Retained<MLState>) -> Self {
    Self {
      inner,
      _not_sync: PhantomData,
    }
  }

  pub(crate) fn raw(&self) -> &MLState {
    &self.inner
  }
}

#[cfg(test)]
mod tests;
