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
//! # [`argmax::ArgmaxSource`]: the in-graph-decoded source
//!
//! argmax's segmenter does NOT emit raw logits — it takes 30 s of waveform
//! and returns already-decoded per-window/frame/speaker activity, having
//! done the windowing, the powerset decode and the overlap detection inside
//! the CoreML graph with its OWN semantics. So [`argmax::ArgmaxSource`]
//! reuses none of the host-side decode above: it maps argmax's decoded
//! tensors straight into the same [`Extraction`]. The two sources can
//! therefore diarize the same audio differently — by design (spec §4). See
//! [`argmax`]'s module doc for the full decode semantics, the index mapping,
//! and every deliberate divergence from argmax's own Swift.
//!
//! # [`Source`] and [`AnySource`]: the selector and the dispatcher
//!
//! [`crate::extract::Options`] carries a [`Source`] selector naming which
//! vendor's source to build — `FluidAudio` (default, cleanly licensed —
//! design spec §6) or `Argmax`. [`AnySource`] is the runtime counterpart: a
//! built, dispatchable `ModelSource`, one variant per `Source`, constructed
//! by [`AnySource::load`]. Both its `load` match and its
//! [`ModelSource::extract`] match are exhaustive with no wildcard arm, so
//! neither source can silently fall through to the other.
//!
//! `Source` is deliberately NOT `#[non_exhaustive]`: unlike this crate's
//! error enums (which reserve growth room because callers must match them
//! defensively), `Source`'s whole point is that its variant set is exactly
//! and honestly `{FluidAudio, Argmax}` — the dispatcher matching on it is
//! forced by the compiler to handle every variant explicitly.

use std::path::Path;

use crate::{
  embed::EmbedModel,
  error::{ExtractError, ModelError},
  extract::{Extraction, Extractor, Options},
  segment::SegmentModel,
};

pub mod argmax;

pub use argmax::{ArgmaxComputeOptions, ArgmaxOptions, ArgmaxSource, ArgmaxVariant};

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
/// [`crate::extract::Options`]'s source selector (design spec §4). Build the
/// named source with [`AnySource::load`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Source {
  /// [`FluidAudioSource`] — this crate's original, host-side-decoding
  /// pipeline. The default.
  FluidAudio,
  /// [`ArgmaxSource`] — the `argmaxinc/speakerkit-coreml` source, decoded
  /// in-graph (see [`argmax`]'s module doc).
  Argmax,
}

impl Default for Source {
  fn default() -> Self {
    DEFAULT_SOURCE
  }
}

/// A built, dispatchable [`ModelSource`] — the runtime counterpart to the
/// [`Source`] selector, owning whichever source's models were loaded.
///
/// Both this type's [`ModelSource::extract`] impl and [`Self::load`] match
/// [`Source`] exhaustively with no wildcard arm, so no path can silently fall
/// back from one source to the other.
#[derive(Debug)]
pub enum AnySource {
  /// A loaded [`FluidAudioSource`].
  FluidAudio(FluidAudioSource),
  /// A loaded [`ArgmaxSource`].
  Argmax(ArgmaxSource),
}

impl AnySource {
  /// Loads the source [`Options::source`] names, from that VENDOR's own
  /// artifact root.
  ///
  /// The two vendors ship different layouts, so `models_root` means a
  /// different thing per arm — there is no single directory that could serve
  /// both:
  ///
  /// - [`Source::FluidAudio`]: a directory holding
  ///   `pyannote_segmentation.mlmodelc` and `wespeaker_v2.mlmodelc`
  ///   (this crate's `Models/speakerkit`).
  /// - [`Source::Argmax`]: the `speakerkit-coreml` root holding
  ///   `speaker_segmenter/` and `speaker_embedder/` (this crate's
  ///   `Models/argmax-speakerkit`) — see [`ArgmaxSource::from_dir_with`].
  ///
  /// `options`'s [`crate::window::WindowOptions`] and
  /// [`crate::extract::ComputeOptions`] are threaded into both arms. The
  /// argmax arm additionally needs an [`ArgmaxVariant`] (quantization tier)
  /// and a third compute placement (its fbank preprocessor), neither of which
  /// exists on the shared [`Options`]; it uses [`ArgmaxOptions::new`]'s
  /// defaults for those, mapping the preprocessor onto
  /// [`crate::extract::ComputeOptions::embedder`] (argmax's own Swift likewise
  /// owns the preprocessor inside its embedder model,
  /// `SpeakerEmbedderModel.swift:142,148`). A caller who needs a different
  /// variant builds [`ArgmaxSource::from_dir_with`] directly and wraps it in
  /// [`Self::Argmax`].
  ///
  /// # Errors
  /// [`ModelError::Load`] / [`ModelError::ContractMismatch`] from whichever
  /// source's loader runs.
  pub fn load(models_root: impl AsRef<Path>, options: Options) -> Result<Self, ModelError> {
    let root = models_root.as_ref();
    let compute = options.compute();
    match options.source() {
      Source::FluidAudio => {
        let seg = SegmentModel::from_file_with(
          root.join("pyannote_segmentation.mlmodelc"),
          crate::segment::SegmentModelOptions::new().with_compute(compute.segmenter()),
        )?;
        let embed = EmbedModel::from_file_with(
          root.join("wespeaker_v2.mlmodelc"),
          crate::embed::EmbedModelOptions::new().with_compute(compute.embedder()),
        )?;
        Ok(Self::FluidAudio(FluidAudioSource::with_options(
          seg, embed, options,
        )))
      }
      Source::Argmax => {
        let argmax_options = ArgmaxOptions::new()
          .with_window(options.window())
          .with_compute(
            ArgmaxComputeOptions::new()
              .with_segmenter(compute.segmenter())
              .with_preprocessor(compute.embedder())
              .with_embedder(compute.embedder()),
          );
        Ok(Self::Argmax(ArgmaxSource::from_dir_with(
          root,
          argmax_options,
        )?))
      }
    }
  }

  /// The [`Source`] this was built from.
  #[inline(always)]
  pub const fn source(&self) -> Source {
    match self {
      Self::FluidAudio(_) => Source::FluidAudio,
      Self::Argmax(_) => Source::Argmax,
    }
  }
}

impl ModelSource for AnySource {
  /// Dispatches to the loaded source's own `extract`. Exhaustive match — a
  /// new [`Source`] variant cannot silently route to an existing source.
  fn extract(&self, samples: &[f32]) -> Result<Extraction, ExtractError> {
    match self {
      Self::FluidAudio(source) => source.extract(samples),
      Self::Argmax(source) => source.extract(samples),
    }
  }
}

#[cfg(test)]
mod tests;
