//! Native CoreML segmentation and embedding backends for `dia`'s
//! pyannote-community-1 diarization pipeline.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md`.
//! Product driver: [`findit-studio/desktop#120`][desktop-120] (components
//! 3+4: speaker segmentation ~20x and speaker embedding ~30x uplift
//! targets). Clustering stays in `dia` (the issue's hard scope line) — this
//! crate only replaces `dia`'s `ort`-backed segmentation and embedding
//! inference with native CoreML/ANE execution, producing tensors
//! bit-compatible with `dia`'s public compute entry points so its
//! parity-proven clustering/reconstruction runs unchanged.
//!
//! [desktop-120]: https://github.com/findit-studio/desktop/issues/120
//!
//! macOS only (built on [`coremlit`]).
//!
//! # Multi-source backend
//!
//! As of the multi-source split
//! (`docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`),
//! the segmentation/embedding pipeline above is [`source::ModelSource`]'s
//! first implementation ([`source::FluidAudioSource`]), not this crate's
//! only one — see that module for the pluggable-source abstraction and
//! why it exists.

#[cfg(feature = "serde")]
mod compute_units_serde;
pub mod embed;
pub mod error;
pub mod extract;
pub mod segment;
pub mod source;
pub mod window;

/// dia's offline clustering option vocabulary, re-exported as speakerkit's
/// public surface for the forthcoming `ClusterBackend` config type (design
/// spec §Architecture: `Offline(OfflineClusterOptions)`), which will wrap it.
///
/// [`OfflineClusterOptions`] configures dia's batch
/// [`cluster_offline`](dia::cluster::cluster_offline) entry point (method,
/// similarity threshold, target speakers, seed); [`OfflineMethod`] and
/// [`Linkage`] are its constituent enums.
///
/// NB the current runtime clustering path, [`extract::Extraction::diarize`],
/// drives dia's `diarize_offline` pipeline with that pipeline's own
/// community-1 hyperparameter defaults (its inline `OfflineInput` knobs) and
/// does not consume these options — `cluster_offline` and `diarize_offline`
/// are distinct dia entry points. Reconciling the config surface with the
/// runtime pipeline is T2's task; T1 only exposes the type T2 will wrap.
pub use dia::cluster::{Linkage, OfflineClusterOptions, OfflineMethod};
