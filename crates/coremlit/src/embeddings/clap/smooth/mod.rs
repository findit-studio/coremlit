//! Per-window smoothing: coremlit re-exports windit's smoothing tier — the
//! [`Smoother`] / [`SmoothPolicy`] seam and the vector low-pass [`VectorEma`] —
//! and adds a thin clap-typed [`smooth`] wrapper.
//!
//! This is the streaming half of the long-audio stack, and the sibling of
//! [`aggregate`](crate::embeddings::clap::aggregate). Where aggregation folds a
//! finished [`WindowEmbedding`] slice to ONE clip-level
//! [`Embedding`](crate::embeddings::clap::Embedding), smoothing rewrites one
//! window in / one window out with the input
//! [`Span`](crate::embeddings::clap::window::Span) intact — so a per-window
//! embedding stream is denoised without being collapsed to a point. A consumer
//! that wants every window to keep emitting, only quieter, wants this tier and
//! not that one.
//!
//! [`VectorEma`] is the streaming sibling of
//! [`EmaRenormalized`](crate::embeddings::clap::aggregate::EmaRenormalized): at
//! the same smoothing factor, window `i` of the smoothed stream carries the
//! direction the EMA aggregate folds over the prefix `[0..=i]`. The equivalence
//! is exact in exact arithmetic and close-but-not-bit-exact in floating point
//! (the aggregate materializes each weight and folds with Neumaier
//! compensation; the smoother carries a two-term recurrence), which the module
//! tests pin as a tolerance rather than as equality.
//!
//! # What is and is not re-exported
//!
//! windit's other two non-identity smoothers, `Ema` and `CadenceEma`, are
//! `Smoother<f32>` scalar low-passes — the right shape for a per-window
//! *probability* (which is how `audio::ced` uses them) and the wrong one for a
//! 512-wide unit-norm embedding. They are deliberately absent here for the same
//! reason `SaliencyWeighted` is absent from
//! [`aggregate`](crate::embeddings::clap::aggregate): re-exporting a knob this
//! module's value type cannot use would ship a misleading surface. Reach them
//! through `windit::smooth` directly.
//!
//! [`Identity`] and [`VectorEmaState`] are reachable at this module path but are
//! deliberately NOT flattened onto [`embeddings::clap`](crate::embeddings::clap)
//! the way [`VectorEma`] and the two traits are: `Identity` is a bare name
//! likely to collide in a flat namespace, and `VectorEmaState` is the policy's
//! associated state type, named only by a caller that holds a streaming stage in
//! a struct field.
//!
//! Import [`VectorEma`] by path rather than through a glob: windit keeps it out
//! of its own prelude precisely because a glob addition can collide (`E0659`) at
//! a downstream use site that globs two modules.

use crate::embeddings::clap::{
  error::{Error, Result},
  window::WindowEmbedding,
};

pub use windit::smooth::{Identity, SmoothPolicy, Smoother, VectorEma, VectorEmaState};

#[cfg(test)]
mod tests;

/// Smooth a per-window embedding stream under `policy`, returning one output
/// window per input window with its [`Span`] unchanged.
///
/// [`Span`]: crate::embeddings::clap::window::Span
///
/// This is the clap-typed wrapper over
/// [`SmoothPolicy::smooth`](windit::smooth::SmoothPolicy::smooth), the batch
/// convenience: it drives a FRESH [`Smoother`] over `windows` on every call, so
/// smoothing a stream chunk by chunk through separate calls is **not**
/// equivalent to one whole-stream call — a running average does not carry across
/// calls. To smooth incrementally, hold one
/// [`smoother`](SmoothPolicy::smoother) and
/// [`push`](Smoother::push) each window into it; that path also sheds the
/// per-window [`Embedding`](crate::embeddings::clap::Embedding) clone this
/// convenience makes.
///
/// An empty `windows` smooths to an empty stream. That is deliberately unlike
/// [`aggregate`](crate::embeddings::clap::aggregate::aggregate), which refuses
/// an empty slice with [`Error::EmptyWindows`]: a fold of nothing has no
/// direction, while a rewrite of nothing is nothing.
///
/// # Errors
/// [`Error::Windowing`] carrying windit's typed error for any smoothing failure.
/// For [`VectorEma`] over clap's fixed-width, always-unit, always-finite
/// [`Embedding`](crate::embeddings::clap::Embedding) the reachable member of
/// that set is the determinacy gate's `WinditError::NonFinite` — a window whose
/// accumulator has cancelled to within its own error bound of zero, so no
/// direction can be reported — plus `WinditError::AllocFailed` for a refused
/// buffer and `WinditError::EpochTooLong` past
/// [`VectorEma::MAX_EPOCH_STEPS`](windit::smooth::VectorEma::MAX_EPOCH_STEPS).
/// The dimension, finiteness and magnitude refusals windit documents cannot
/// arise here, because [`Embedding`](crate::embeddings::clap::Embedding)'s own
/// constructors have already excluded them.
pub fn smooth<P>(policy: &P, windows: &[WindowEmbedding]) -> Result<Vec<WindowEmbedding>>
where
  P: SmoothPolicy<crate::embeddings::clap::Embedding>,
{
  policy.smooth(windows).map_err(Error::from)
}
