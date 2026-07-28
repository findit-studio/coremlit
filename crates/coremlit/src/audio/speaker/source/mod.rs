//! The pluggable model-source abstraction (design spec §4,
//! `docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`):
//! [`ModelSource`] is the common interface every seg+embed backend
//! implements, all normalizing to the same [`Extraction`] that feeds
//! `diaric`'s clustering via `Extraction::into_offline_input`.
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
//! to the identical [`Extraction`] shape so `diaric`'s clustering runs
//! unchanged regardless of which source produced it.
//!
//! # [`FluidAudioSource`]: the existing pipeline, unchanged
//!
//! This crate's segmentation + embedding pipeline ([`crate::audio::speaker::segment`],
//! [`crate::audio::speaker::embed`], [`crate::audio::speaker::window`], [`crate::audio::speaker::extract`] — built before
//! the multi-source split, when this crate had only one source) already
//! implements the FluidAudio path in full. [`FluidAudioSource`] does not
//! reimplement any of it: it owns a loaded [`SegmentModel`]/[`EmbedModel`]
//! pair plus an [`Options`], and its [`ModelSource::extract`] delegates
//! directly to [`Extractor::extract`] — the exact orchestration every
//! existing model-gated `extract_*` test in [`crate::audio::speaker::extract`] already
//! exercises. No behavior changes here; this module only adds an
//! owns-its-models, trait-shaped wrapper around it. [`Extractor`] itself
//! is untouched and stays a fully working, independent public API (a
//! caller who wants to swap models per call without owning them keeps
//! that option).
//!
//! # [`argmax::ArgmaxSource`]: the in-graph-decoded source
//!
//! argmax's segmenter does NOT emit a per-frame powerset row for this
//! crate to decode (FluidAudio's emits `log(softmax(·))` log-probabilities;
//! see [`crate::audio::speaker::segment`]'s module doc) — it takes 30 s of waveform and
//! returns already-decoded per-window/frame/speaker activity, having
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
//! [`crate::audio::speaker::extract::Options`] carries a [`Source`] selector naming which
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

use std::path::{Path, PathBuf};

use crate::audio::speaker::{
  embed::EmbedModel,
  error::{ExtractError, ModelError},
  extract::{Extraction, Extractor, Options},
  segment::SegmentModel,
};

pub mod argmax;

pub use argmax::{ArgmaxComputeOptions, ArgmaxOptions, ArgmaxSource, ArgmaxVariant};

/// A pluggable seg+embed backend: given 16 kHz mono `samples`, produces the
/// [`Extraction`] tensor set `diaric`'s offline diarizer consumes. See the
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
/// `wespeaker.mlmodelc` via [`SegmentModel`]/[`EmbedModel`], decoded
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

  /// Loads the shipping FluidAudio artifact pair under `models_root` with
  /// `options` — exactly [`Self::load_with`] at the empty
  /// [`FluidAudioArtifactConfig`], i.e. the fixed filenames and `options`'s own
  /// compute placement. This is the body of [`AnySource::load`]'s
  /// [`Source::FluidAudio`] arm, which delegates here.
  ///
  /// # Errors
  /// As [`Self::load_with`].
  pub fn load(models_root: impl AsRef<Path>, options: Options) -> Result<Self, ModelError> {
    Self::load_with(models_root, options, &FluidAudioArtifactConfig::new())
  }

  /// Loads the FluidAudio artifact pair `config` selects under `models_root`,
  /// with `options` supplying the extraction geometry the returned source runs.
  ///
  /// Paths come from [`FluidAudioArtifacts::resolve_with`] — an explicit
  /// override verbatim, an omitted one from the fixed-name convention.
  ///
  /// # Precedence
  /// Compute placement is taken per model from `config` when it names one, and
  /// from `options`'s [`crate::audio::speaker::extract::ComputeOptions`]
  /// otherwise; `options` in turn already carries the crate defaults. The
  /// config is the more specific input and wins, and its `None` — not some
  /// sentinel value — is what defers. That direction is forced: `ComputeOptions`
  /// has no absent state, so an options-wins rule would make the config's
  /// placements unreachable at every call site.
  ///
  /// `options`'s [`Options::source`] is NOT consulted here: this constructor
  /// always builds the FluidAudio source, exactly as [`Extractor::extract`]
  /// always runs the FluidAudio orchestration. Wrap the result in
  /// [`AnySource::FluidAudio`] for the dispatchable form.
  ///
  /// **A substituted artifact carries none of this crate's safety evidence** —
  /// read [`FluidAudioArtifactConfig`]'s own documentation before pointing this
  /// at bytes the crate does not pin.
  ///
  /// # Errors
  /// [`ModelError::Load`] if CoreML cannot load either artifact — including
  /// [`crate::LoadError::NotFound`], which names the exact resolved path.
  /// [`ModelError::ContractMismatch`] if either model's declared I/O diverges
  /// from the pinned shapes and dtypes; that check validates SHAPE, never
  /// numerics, so it does not qualify a substitute artifact.
  pub fn load_with(
    models_root: impl AsRef<Path>,
    options: Options,
    config: &FluidAudioArtifactConfig,
  ) -> Result<Self, ModelError> {
    let artifacts = FluidAudioArtifacts::resolve_with(models_root, config);
    let (segmenter_options, embedder_options) = resolve_model_options(options, config);
    let seg = SegmentModel::from_file_with(artifacts.segmenter(), segmenter_options)?;
    let embed = EmbedModel::from_file_with(artifacts.embedder(), embedder_options)?;
    Ok(Self::with_options(seg, embed, options))
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

/// Default [`crate::audio::speaker::extract::Options::source`] — [`Source::FluidAudio`],
/// the cleanly licensed default (design spec §6).
pub const DEFAULT_SOURCE: Source = Source::FluidAudio;

/// Which vendor's CoreML conversion computes the seg+embed tensors —
/// [`crate::audio::speaker::extract::Options`]'s source selector (design spec §4). Build the
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

/// Declarative overrides for the [`Source::FluidAudio`] artifact set: which two
/// `.mlmodelc` bundles [`FluidAudioSource::load_with`] loads, and which hardware
/// each is placed on.
///
/// Every field is `Option`, and `None` means **"use the convention"** — never
/// "the default value". For a path the convention is [`FluidAudioArtifacts::resolve`]'s
/// fixed filename under the caller's `models_root`; for a placement it is
/// whatever the caller's own [`Options`] carries. Keeping absence *semantic* is
/// what lets a config and a separately supplied [`Options`] compose with a
/// stated precedence (see [`FluidAudioSource::load_with`]).
///
/// This crate reads no configuration file: the library graph carries no
/// config-format dependency and no search path. Parse your own TOML/JSON into
/// this type (behind the `serde` feature) and hand it over. (`toml` is a
/// `[dev-dependencies]` entry only, so the wire-format example below is executed
/// rather than trusted; `cargo tree -e normal` shows it absent from the
/// library.)
///
/// ```
/// use coremlit::ComputeUnits;
/// use coremlit::audio::speaker::source::{FluidAudioArtifactConfig, FluidAudioArtifacts};
///
/// let config = FluidAudioArtifactConfig::new()
///   .with_embedder("models/my_embed.mlmodelc")
///   .with_embedder_compute(ComputeUnits::CpuOnly);
///
/// let artifacts = FluidAudioArtifacts::resolve_with("Models/speakerkit", &config);
/// assert_eq!(
///   artifacts.embedder(),
///   std::path::Path::new("models/my_embed.mlmodelc")
/// );
/// // The omitted segmenter keeps the convention under the root.
/// assert_eq!(
///   artifacts.segmenter(),
///   std::path::Path::new("Models/speakerkit/pyannote_segmentation.mlmodelc")
/// );
/// ```
///
/// # ⚠ A custom artifact carries none of this crate's safety evidence
///
/// Bringing your own artifact is supported and legitimate. What does NOT come
/// with it is the evidence: every gate that makes the shipping selection
/// trustworthy is a pin on *the shipping bytes*, and none of it transfers.
/// `fp16_safe_wespeaker_fp32_matches_pinned_sha256` and
/// `fp16_safe_segmentation_matches_pinned_sha256` (`tests/speaker/model_io.rs`)
/// hash the canonical artifacts and say nothing about a substitute; the
/// shipping-DER suite scores the canonical artifacts; and the
/// [`ModelError::ContractMismatch`] check run at load time validates tensor
/// SHAPES and DTYPES, never numerics.
///
/// Issue #15 measured how little that leaves. Two facts from it, both scoped to
/// the host and clips they were measured on:
///
/// - The retired int8 embedder is contract-identical to the shipping fp32 one
///   and passes every shape/dtype check, yet its per-tensor palettization error
///   is a COHERENT displacement that compresses between-speaker margins in the
///   frozen community-1 PLDA space — on `09_mrbeast_dollar_date` it finds 5 of
///   8 speakers at 16.59 % DER.
/// - The shipping fp32 embedder and FluidInference's pre-repair fp32 conversion
///   have byte-identical weights and differ only in two attentive-stat pooling
///   guard constants (`1e-8` against the `0x1p-24` fp16 floor) plus buildInfo
///   strings. They measured EQUAL to the last DER error unit on every clip-09
///   remedy-matrix arm — those guard sites add their epsilon to mask sums that
///   are never near zero, so on that audio the guard never had to act. The
///   repair is a STATIC fp16-floor guarantee, **not** a measured quality fix:
///   no comparison here shows the pre-repair artifact failing where the
///   repaired one does not. What it does show is that contract equality and
///   measured DER equality can BOTH hold while one artifact carries a guard
///   below the fp16 floor — so neither establishes that two artifacts agree on
///   every input and placement, only on what was measured.
///
/// Point this config at an int8, fp16-unsafe, or otherwise unvalidated artifact
/// and no gate here will object: the int8 collapse above happened with every
/// one of them green. Validate a substitute yourself — at DER, on multi-speaker
/// audio, on the placement you ship.
///
/// # Wire format (`serde` feature)
///
/// `deny_unknown_fields`: a misspelled key is a hard error, never a silent
/// fallback to the pinned default. Silently loading the shipping artifact while
/// the user believes their `embeder = "..."` took effect is precisely the
/// wrong-artifact failure issue #15 is about. Every key is optional; omitting
/// one selects the convention.
///
/// This type IS the table, so its keys sit at the top level of the document you
/// hand to the parser:
///
/// ```text
/// segmenter         = "models/my_seg.mlmodelc"
/// embedder          = "models/my_embed.mlmodelc"
/// segmenter_compute = "cpu_only"
/// embedder_compute  = "all"
/// ```
///
/// A `[speaker]` header does NOT deserialize into this type. That header makes
/// `speaker` a key of the ENCLOSING document, and `deny_unknown_fields` rejects
/// it — correctly, because the document is then not this struct. A caller who
/// wants the artifacts nested under one owns that outer type:
///
/// ```text
/// #[derive(serde::Deserialize)]
/// struct AppConfig {
///   speaker: FluidAudioArtifactConfig,
/// }
/// ```
///
/// Both directions are pinned by `config_parses_the_documented_toml` and
/// `config_rejects_a_speaker_headed_document_but_a_wrapper_accepts_it`, which
/// parse the exact documents shown above — this section's claims are executable,
/// not asserted.
///
/// Compute names are [`crate::ComputeUnits::as_str`]'s snake_case forms
/// (`cpu_only`, `cpu_and_gpu`, `cpu_and_neural_engine`, `all`) — the spelling
/// [`crate::audio::speaker::extract::ComputeOptions`] already accepts, shared
/// through the same bridge so the two cannot drift.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct FluidAudioArtifactConfig {
  // `skip_serializing_if` is not cosmetic here: TOML — the format this config
  // is shaped for — cannot represent a null, so an absent field must be
  // omitted rather than encoded.
  #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
  segmenter: Option<PathBuf>,
  #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
  embedder: Option<PathBuf>,
  // `default` is required on these two, not blanket-applied: a field with
  // `with = ...` and no default makes OMISSION a hard error, which would
  // contradict the type's "absent means use the convention" contract. The
  // unknown-key rejection that `deny_unknown_fields` provides is unaffected.
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      with = "crate::audio::speaker::compute_units_serde::option"
    )
  )]
  segmenter_compute: Option<crate::ComputeUnits>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      with = "crate::audio::speaker::compute_units_serde::option"
    )
  )]
  embedder_compute: Option<crate::ComputeUnits>,
}

impl Default for FluidAudioArtifactConfig {
  fn default() -> Self {
    Self::new()
  }
}

impl FluidAudioArtifactConfig {
  /// The empty config: every field absent, so every choice falls through to the
  /// convention. [`FluidAudioArtifacts::resolve_with`] at this value IS
  /// [`FluidAudioArtifacts::resolve`], and [`FluidAudioSource::load_with`] at it
  /// IS [`FluidAudioSource::load`].
  pub const fn new() -> Self {
    Self {
      segmenter: None,
      embedder: None,
      segmenter_compute: None,
      embedder_compute: None,
    }
  }

  /// The segmentation artifact override, if any. `None` selects
  /// `<models_root>/pyannote_segmentation.mlmodelc`.
  #[inline]
  pub fn segmenter(&self) -> Option<&Path> {
    self.segmenter.as_deref()
  }
  /// The embedder artifact override, if any. `None` selects
  /// `<models_root>/wespeaker.mlmodelc`.
  #[inline]
  pub fn embedder(&self) -> Option<&Path> {
    self.embedder.as_deref()
  }
  /// The segmentation model's compute placement, if any. `None` defers to the
  /// caller's [`crate::audio::speaker::extract::ComputeOptions`].
  #[inline(always)]
  pub const fn segmenter_compute(&self) -> Option<crate::ComputeUnits> {
    self.segmenter_compute
  }
  /// The embedding model's compute placement, if any. `None` defers to the
  /// caller's [`crate::audio::speaker::extract::ComputeOptions`].
  #[inline(always)]
  pub const fn embedder_compute(&self) -> Option<crate::ComputeUnits> {
    self.embedder_compute
  }

  /// Builder form of [`Self::set_segmenter`].
  #[must_use]
  pub fn with_segmenter(mut self, segmenter: impl Into<PathBuf>) -> Self {
    self.set_segmenter(segmenter);
    self
  }
  /// Sets [`Self::segmenter`] in place — the path is used verbatim, never
  /// joined onto a `models_root`.
  pub fn set_segmenter(&mut self, segmenter: impl Into<PathBuf>) -> &mut Self {
    self.segmenter = Some(segmenter.into());
    self
  }
  /// Builder form of [`Self::set_embedder`].
  #[must_use]
  pub fn with_embedder(mut self, embedder: impl Into<PathBuf>) -> Self {
    self.set_embedder(embedder);
    self
  }
  /// Sets [`Self::embedder`] in place — the path is used verbatim, never joined
  /// onto a `models_root`.
  pub fn set_embedder(&mut self, embedder: impl Into<PathBuf>) -> &mut Self {
    self.embedder = Some(embedder.into());
    self
  }
  /// Builder form of [`Self::set_segmenter_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_segmenter_compute(mut self, compute: crate::ComputeUnits) -> Self {
    self.set_segmenter_compute(compute);
    self
  }
  /// Sets [`Self::segmenter_compute`] in place.
  #[inline(always)]
  pub const fn set_segmenter_compute(&mut self, compute: crate::ComputeUnits) -> &mut Self {
    self.segmenter_compute = Some(compute);
    self
  }
  /// Builder form of [`Self::set_embedder_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_embedder_compute(mut self, compute: crate::ComputeUnits) -> Self {
    self.set_embedder_compute(compute);
    self
  }
  /// Sets [`Self::embedder_compute`] in place.
  #[inline(always)]
  pub const fn set_embedder_compute(&mut self, compute: crate::ComputeUnits) -> &mut Self {
    self.embedder_compute = Some(compute);
    self
  }
}

/// The per-model load options [`FluidAudioSource::load_with`] hands to
/// [`SegmentModel::from_file_with`] / [`EmbedModel::from_file_with`], resolved
/// from the two configuration sources it composes. The single place that
/// precedence is decided, so a test can pin it without a loaded model.
///
/// Precedence per model: `config`'s placement when present, else `options`'s
/// [`crate::audio::speaker::extract::ComputeOptions`] (which already carries the
/// crate defaults). Config-wins is forced rather than chosen — `ComputeOptions`
/// stores plain [`crate::ComputeUnits`] with no absent state, so an
/// options-wins rule would leave the config's placements unreachable at every
/// call site.
///
/// Returns the two option types rather than bare [`crate::ComputeUnits`] so the
/// call site cannot transpose them: `SegmentModelOptions` and
/// `EmbedModelOptions` are distinct types, and a swap fails to compile.
fn resolve_model_options(
  options: Options,
  config: &FluidAudioArtifactConfig,
) -> (
  crate::audio::speaker::segment::SegmentModelOptions,
  crate::audio::speaker::embed::EmbedModelOptions,
) {
  let compute = options.compute();
  (
    crate::audio::speaker::segment::SegmentModelOptions::new().with_compute(
      config
        .segmenter_compute()
        .unwrap_or_else(|| compute.segmenter()),
    ),
    crate::audio::speaker::embed::EmbedModelOptions::new().with_compute(
      config
        .embedder_compute()
        .unwrap_or_else(|| compute.embedder()),
    ),
  )
}

/// The exact on-disk artifacts the [`Source::FluidAudio`] arm of
/// [`AnySource::load`] loads under a `models_root` — the SINGLE definition of
/// *which files ship*.
///
/// [`AnySource::load`] resolves the FluidAudio paths through this and nothing
/// else, so a gate can assert the shipping selection at its source of truth
/// instead of re-encoding it. [`Self::embedder`] is the fp32
/// `wespeaker.mlmodelc` (issue #15): the previously shipped int8-palettized
/// `wespeaker_v2.mlmodelc` silently collapses 8-speaker audio (5 of 8
/// speakers, 16.59 % DER on the measured clip) because its per-tensor
/// palettization error is a COHERENT shared displacement that compresses
/// between-speaker margins in the frozen community-1 PLDA space — while
/// holding no stable extraction-speed edge over fp32 on any placement
/// (≤ ~15 % apart with the sign varying run to run; the measured trade is in
/// `tests/speaker/model_io.rs`'s DECISION). The mechanism, the factorial
/// that isolated
/// it, and the retirement rationale live in `tests/speaker/model_io.rs` (the
/// DECISION section) and `tests/speaker/backend_factorial.rs`
/// (`quantization_error_structure`). Repointing this resolver moves
/// production AND every gate that pins the selection through it, so the two
/// cannot silently diverge.
///
/// **Pure**: path selection only — no filesystem access, no model load — so a
/// hermetic unit test can pin the selection with no models present.
///
/// A caller bringing their own artifacts overrides either path through
/// [`Self::resolve_with`] and a [`FluidAudioArtifactConfig`]; that path wraps
/// this fixed-name selection rather than replacing it, so what is pinned here
/// remains the default-of-record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FluidAudioArtifacts {
  segmenter: PathBuf,
  embedder: PathBuf,
}

impl FluidAudioArtifacts {
  /// Resolve the FluidAudio artifact paths under `models_root`. Pure: joins the
  /// two fixed artifact names onto the root, touching no filesystem.
  ///
  /// Exactly [`Self::resolve_with`] at the empty [`FluidAudioArtifactConfig`],
  /// so the fixed-name selection stays the default-of-record: the config layer
  /// defaults THROUGH this resolver rather than replacing it.
  #[must_use]
  pub fn resolve(models_root: impl AsRef<Path>) -> Self {
    Self::resolve_with(models_root, &FluidAudioArtifactConfig::new())
  }

  /// Resolve the FluidAudio artifact paths under `models_root`, letting
  /// `config` substitute either artifact. Pure, exactly as [`Self::resolve`]:
  /// path selection only, no filesystem access, no model load.
  ///
  /// Per field, independently:
  ///
  /// - `Some(path)` is used **verbatim** — NOT joined onto `models_root`. A
  ///   relative override is therefore relative to the process's working
  ///   directory, and an override may point outside the root entirely.
  /// - `None` falls back to this source's fixed filename under `models_root` —
  ///   the selection [`Self::resolve`] pins.
  ///
  /// Overriding one artifact leaves the other on the convention; see
  /// [`FluidAudioArtifactConfig`] for what a substituted artifact does and does
  /// not inherit from this crate's gates.
  #[must_use]
  pub fn resolve_with(models_root: impl AsRef<Path>, config: &FluidAudioArtifactConfig) -> Self {
    let root = models_root.as_ref();
    Self {
      segmenter: config.segmenter().map_or_else(
        || root.join("pyannote_segmentation.mlmodelc"),
        Path::to_path_buf,
      ),
      embedder: config
        .embedder()
        .map_or_else(|| root.join("wespeaker.mlmodelc"), Path::to_path_buf),
    }
  }

  /// The segmentation artifact path — `<root>/pyannote_segmentation.mlmodelc`
  /// unless a [`FluidAudioArtifactConfig`] substituted it.
  #[inline]
  #[must_use]
  pub fn segmenter(&self) -> &Path {
    &self.segmenter
  }

  /// The embedder artifact path — `<root>/wespeaker.mlmodelc`, the fp32
  /// shipping default (issue #15), unless a [`FluidAudioArtifactConfig`]
  /// substituted it.
  #[inline]
  #[must_use]
  pub fn embedder(&self) -> &Path {
    &self.embedder
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
  ///   `pyannote_segmentation.mlmodelc` and `wespeaker.mlmodelc`
  ///   (this crate's `Models/speakerkit`).
  /// - [`Source::Argmax`]: the `speakerkit-coreml` root holding
  ///   `speaker_segmenter/` and `speaker_embedder/` (this crate's
  ///   `Models/argmax-speakerkit`) — see [`ArgmaxSource::from_dir_with`].
  ///
  /// `options`'s [`crate::audio::speaker::window::WindowOptions`] and
  /// [`crate::audio::speaker::extract::ComputeOptions`] are threaded into both arms. The
  /// argmax arm additionally needs an [`ArgmaxVariant`] (quantization tier)
  /// and a third compute placement (its fbank preprocessor), neither of which
  /// exists on the shared [`Options`]; it uses [`ArgmaxOptions::new`]'s
  /// defaults for those, mapping the preprocessor onto
  /// [`crate::audio::speaker::extract::ComputeOptions::embedder`] (argmax's own Swift likewise
  /// owns the preprocessor inside its embedder model,
  /// `SpeakerEmbedderModel.swift:142,148`). A caller who needs a different
  /// variant builds [`ArgmaxSource::from_dir_with`] directly and wraps it in
  /// [`Self::Argmax`].
  ///
  /// # Bring-your-own artifacts
  /// Both arms here load their vendor's CONVENTIONAL filenames. To substitute an
  /// artifact — or to override a placement RELATIVE TO the `options` passed here
  /// — build the source directly and wrap it: [`FluidAudioSource::load_with`]
  /// with a [`FluidAudioArtifactConfig`] into [`Self::FluidAudio`], or
  /// [`ArgmaxSource::from_dir_with`] into [`Self::Argmax`].
  ///
  /// Per-model placement alone needs neither. `options`'s
  /// [`crate::audio::speaker::extract::ComputeOptions`] already carries one
  /// [`crate::ComputeUnits`] per model, and this method threads both through
  /// unchanged — a single [`Options`] can put the segmenter and the embedder on
  /// different hardware. What a [`FluidAudioArtifactConfig`] adds is artifact
  /// PATHS plus a placement override that outranks whatever `options` says:
  /// precedence, not reach.
  ///
  /// The config is per-source deliberately: the two vendors'
  /// artifact SETS differ in count, in directory shape, and (for argmax) in
  /// carrying a variant that itself participates in path resolution, so one
  /// shared path config could only describe both by leaving keys inert under
  /// the other source.
  ///
  /// # Errors
  /// [`ModelError::Load`] / [`ModelError::ContractMismatch`] from whichever
  /// source's loader runs.
  pub fn load(models_root: impl AsRef<Path>, options: Options) -> Result<Self, ModelError> {
    let root = models_root.as_ref();
    match options.source() {
      // The artifact paths come from the pure resolver, the single place "which
      // FluidAudio files ship" is defined, so a gate pins production's exact
      // selection rather than a parallel copy of it (finding 3).
      Source::FluidAudio => Ok(Self::FluidAudio(FluidAudioSource::load(root, options)?)),
      Source::Argmax => {
        let compute = options.compute();
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
