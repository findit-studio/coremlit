//! The pluggable model-source abstraction (design spec §4,
//! `docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`):
//! [`ModelSource`] is the common interface every seg+embed backend
//! implements, all normalizing to the same [`Extraction`] that feeds
//! `dia`'s clustering via `Extraction::into_offline_input`.
//!
//! # Why this exists
//!
//! A ground-truth model comparison (design spec §2) found that FluidAudio,
//! argmax, and `dia` all run the *same* pyannote pipeline — same
//! segmentation net, same WeSpeaker embedder, same PLDA/VBx clustering —
//! three independent conversions of one model family, differing only in
//! packaging and in-graph preprocessing (design spec §3). So `speakerkit`
//! is a *multi-source* backend: a caller selects which vendor's CoreML
//! conversion computes the seg+embed tensors, and every source normalizes
//! to the identical [`Extraction`] shape so `dia`'s clustering runs
//! unchanged regardless of which source produced it.
//!
//! # [`FluidAudioSource`]: the existing pipeline, unchanged
//!
//! This crate's segmentation + embedding pipeline ([`crate::segment`],
//! [`crate::embed`], [`crate::window`], [`crate::extract`] — built before
//! the multi-source split, when this crate had only one source) already
//! implements the FluidAudio path in full. [`FluidAudioSource`] does not
//! reimplement any of it: it owns a loaded [`SegmentModel`]/[`EmbedModel`]
//! pair plus an [`Options`], and its [`ModelSource::extract`] delegates
//! directly to [`Extractor::extract`] — the exact orchestration every
//! existing model-gated `extract_*` test in [`crate::extract`] already
//! exercises. No behavior changes here; this module only adds an
//! owns-its-models, trait-shaped wrapper around it. [`Extractor`] itself
//! is untouched and stays a fully working, independent public API (a
//! caller who wants to swap models per call without owning them keeps
//! that option).
//!
//! # [`Source`]: today just FluidAudio, `Argmax` reserved
//!
//! [`crate::extract::Options`] carries a [`Source`] selector so
//! configuration can name which vendor's source to build — `FluidAudio`
//! (default, cleanly licensed — design spec §6) or `Argmax`. Only
//! [`FluidAudioSource`] exists today: `Source::Argmax` is a real,
//! exhaustively-matchable variant (nothing hides it behind a wildcard —
//! see this module's `source_variants_are_exhaustively_matchable` test),
//! but no `ModelSource` impl is built from it yet — that is the multi-source
//! plan's Task 3 (`ArgmaxSource`). Selecting `Argmax` on an `Options`
//! value has no effect on its own: nothing in this crate reads
//! `Options::source` to decide which source to construct yet, so this is
//! forward-compatible configuration surface, not yet a working switch.
//! `Source` is deliberately NOT `#[non_exhaustive]`: unlike this crate's
//! error enums (which reserve growth room because callers must match
//! them defensively), `Source`'s whole point right now is that its
//! variant set is exactly and honestly `{FluidAudio, Argmax}` — a future
//! dispatcher matching on it should be forced by the compiler to handle
//! `Argmax` explicitly, not silently fall through a catch-all arm.

use crate::{
  embed::EmbedModel,
  error::ExtractError,
  extract::{Extraction, Extractor, Options},
  segment::SegmentModel,
};

/// A pluggable seg+embed backend: given 16 kHz mono `samples`, produces the
/// [`Extraction`] tensor set `dia`'s offline diarizer consumes. See the
/// module doc for why this crate has more than one implementation.
pub trait ModelSource {
  /// Runs the full extraction over `samples`. Every implementation
  /// normalizes to the same [`Extraction`] shape, but each owns its own
  /// model(s) and decode semantics — see the implementing type's own
  /// documentation for exactly which [`ExtractError`] variants it can
  /// return.
  ///
  /// # Errors
  /// Implementation-defined; see the implementing type.
  fn extract(&self, samples: &[f32]) -> Result<Extraction, ExtractError>;
}

/// The FluidAudio model source: `pyannote_segmentation.mlmodelc` +
/// `wespeaker_v2.mlmodelc` via [`SegmentModel`]/[`EmbedModel`], decoded
/// host-side by [`Extractor::extract`] — this crate's original (and,
/// until the multi-source split, only) pipeline. See the module doc's
/// "`FluidAudioSource`: the existing pipeline, unchanged" section.
#[derive(Debug)]
pub struct FluidAudioSource {
  seg: SegmentModel,
  embed: EmbedModel,
  options: Options,
}

impl FluidAudioSource {
  /// A source over already-loaded models, using default [`Options`].
  pub fn new(seg: SegmentModel, embed: EmbedModel) -> Self {
    Self::with_options(seg, embed, Options::new())
  }

  /// A source over already-loaded models and explicit [`Options`].
  #[must_use]
  pub fn with_options(seg: SegmentModel, embed: EmbedModel, options: Options) -> Self {
    Self {
      seg,
      embed,
      options,
    }
  }

  /// The source's [`Options`].
  #[inline(always)]
  pub const fn options_ref(&self) -> &Options {
    &self.options
  }
}

impl ModelSource for FluidAudioSource {
  /// Delegates to [`Extractor::extract`] with this source's own
  /// [`SegmentModel`]/[`EmbedModel`]/[`Options`] — see that method's own
  /// doc for the exact stage-by-stage behavior and the full `# Errors`
  /// list, inherited verbatim: no orchestration logic lives here, this is
  /// composition only (module doc).
  fn extract(&self, samples: &[f32]) -> Result<Extraction, ExtractError> {
    Extractor::with_options(self.options).extract(&self.seg, &self.embed, samples)
  }
}

/// Default [`crate::extract::Options::source`] — [`Source::FluidAudio`],
/// the cleanly licensed default (design spec §6).
pub const DEFAULT_SOURCE: Source = Source::FluidAudio;

/// Which vendor's CoreML conversion computes the seg+embed tensors —
/// [`crate::extract::Options`]'s source selector (design spec §4). See the
/// module doc's "`Source`: today just FluidAudio, `Argmax` reserved"
/// section: only `FluidAudio` has a working [`ModelSource`] impl today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Source {
  /// [`FluidAudioSource`] — this crate's original pipeline. The default.
  FluidAudio,
  /// The argmax `speakerkit-coreml` source. Reserved for the multi-source
  /// plan's Task 3 (`ArgmaxSource`): this variant exists and is fully
  /// matchable today, but no [`ModelSource`] impl is built from it yet —
  /// selecting it has no effect on its own (module doc).
  Argmax,
}

impl Default for Source {
  fn default() -> Self {
    DEFAULT_SOURCE
  }
}

#[cfg(test)]
mod tests;
