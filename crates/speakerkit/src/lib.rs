//! Native CoreML segmentation and embedding backends for `dia`'s
//! pyannote-community-1 diarization pipeline.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md`.
//! Product driver: [`findit-studio/desktop#120`][desktop-120] (components
//! 3+4: speaker segmentation ~20x and speaker embedding ~30x uplift
//! targets). The clustering *algorithms* are `dia`'s — this crate replaces
//! `dia`'s `ort`-backed segmentation and embedding inference with native
//! CoreML/ANE execution, producing tensors bit-compatible with `dia`'s public
//! compute entry points so its parity-proven clustering/reconstruction runs
//! unchanged. As of the clustering phase it also *drives* that clustering at
//! runtime through a thin backend-selecting stage (see the clustering example
//! below and the [`cluster`] module) — it does not reimplement the clustering
//! algorithms.
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
//!
//! # Clustering
//!
//! An [`extract::Extraction`] is not the end of the road: as of the clustering
//! phase this crate drives `dia`'s clustering directly, turning the extracted
//! tensors into speaker-labelled RTTM spans. Select a backend with
//! [`ClusterBackend`] — the offline pyannote-community-1 pipeline (the default,
//! DER-gated) or the online FluidAudio-semantics matcher (streaming,
//! order-dependent, and NOT pyannote-parity) — and run it with
//! [`diarize`](extract::Extraction::diarize) /
//! [`diarize_with`](extract::Extraction::diarize_with) /
//! [`diarize_online`](extract::Extraction::diarize_online). See the [`cluster`]
//! module for the full two-engine surface and its honesty boundaries.
//!
//! ```no_run
//! use speakerkit::extract::Options;
//! use speakerkit::source::{AnySource, ModelSource, Source};
//! use speakerkit::{ClusterBackend, OfflineOptions, OnlineOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let audio: Vec<f32> = vec![0.0; 16_000]; // 16 kHz mono; this crate does no I/O.
//! let source =
//!   AnySource::load("Models/speakerkit", Options::new().with_source(Source::FluidAudio))?;
//! let extraction = source.extract(&audio)?;
//!
//! // Offline (the default backend): dia's pyannote-community-1 AHC→VBx pipeline,
//! // through the frozen community-1 PLDA projection it clusters in.
//! let plda = dia::plda::PldaTransform::new()?;
//! let offline = extraction.diarize(&plda)?;
//! println!("offline: {} spans", offline.spans_slice().len());
//!
//! // The same engine, tuned (every default already equals dia's = pyannote's).
//! let tuned = ClusterBackend::Offline(OfflineOptions::default().with_threshold(0.7));
//! let _ = extraction.diarize_with(&plda, tuned)?;
//!
//! // Online: FluidAudio's streaming greedy matcher — raw cosine, NO plda.
//! let online = extraction.diarize_online(OnlineOptions::default())?;
//! println!("online: {} spans", online.spans_slice().len());
//! # Ok(())
//! # }
//! ```

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
/// [`OfflineOptions`] or the online engine with [`OnlineOptions`];
/// [`extract::Extraction::diarize_with`] runs it (or
/// [`extract::Extraction::diarize_online`] for the plda-free online engine). See
/// [`cluster`] for the full surface.
///
/// speakerkit deliberately does NOT re-export dia's batch-clusterer vocabulary
/// (`OfflineClusterOptions`/`OfflineMethod`/`Linkage`) that T1 briefly exposed:
/// [`ClusterBackend::Offline`] wraps dia's pyannote-parity *pipeline*
/// ([`extract::Extraction::diarize`] → `dia::offline::diarize_offline`), not the
/// batch clusterer those types configure. The [`cluster`] module documents which
/// dia entry point `Offline` wraps and why the batch vocabulary was removed
/// (design spec AMENDMENT 2026-07-16).
pub use cluster::{ClusterBackend, OfflineOptions, OnlineOptions, ParseClusterBackendError};
