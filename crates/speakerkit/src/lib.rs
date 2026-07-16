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

pub mod cluster;
#[cfg(feature = "serde")]
mod compute_units_serde;
pub mod embed;
pub mod error;
pub mod extract;
pub mod segment;
pub mod source;
pub mod window;

/// The runtime clustering config surface (design spec §Architecture): select a
/// backend with [`ClusterBackend`] and tune the offline engine with
/// [`OfflineOptions`]; [`extract::Extraction::diarize_with`] runs it. See
/// [`cluster`] for the full surface.
///
/// Removed re-export (was T1's): speakerkit no longer re-exports dia's
/// `OfflineClusterOptions`/`OfflineMethod`/`Linkage`. T1 exposed them expecting
/// [`ClusterBackend::Offline`] to wrap them, but T1 also discovered dia has TWO
/// disjoint offline entry points, and the runtime path drives the OTHER one:
/// [`extract::Extraction::diarize`] runs dia's pyannote-parity PIPELINE
/// (`dia::offline::diarize_offline`, tuned by [`OfflineOptions`]), NOT dia's
/// batch clusterer (`dia::cluster::cluster_offline`, which those three types
/// configure — a different algorithm surface never validated against the parity
/// corpus). Re-exporting the batch vocabulary as speakerkit's clustering
/// surface was therefore misleading, so it is gone (design spec AMENDMENT
/// 2026-07-16). A caller who genuinely wants dia's batch clusterer can still
/// reach it through the `dia` dependency directly; a first-class batch mode, if
/// ever wanted, would arrive as its own [`ClusterBackend`] variant with its own
/// gates.
pub use cluster::{ClusterBackend, OfflineOptions, ParseClusterBackendError};
