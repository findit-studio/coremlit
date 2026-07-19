//! Native CoreML segmentation and embedding backends for the
//! pyannote-community-1 diarization pipeline (its clustering core `diaric`,
//! extracted from the ancestor `dia` crate).
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md`.
//! Product driver: [`findit-studio/desktop#120`][desktop-120] (components
//! 3+4: speaker segmentation ~20x and speaker embedding ~30x uplift
//! targets). The clustering *algorithms* are pyannote's, their implementation
//! extracted from `dia` into `diaric` — the backend-free runtime clustering core
//! this crate drives. This crate replaces `dia`'s `ort`-backed segmentation and
//! embedding inference with native CoreML/ANE execution, producing tensors
//! bit-compatible with `diaric`'s public compute entry points so its
//! parity-proven clustering/reconstruction runs unchanged. As of the clustering
//! phase it also *drives* that clustering at runtime through a thin
//! backend-selecting stage (see the clustering example below and the [`cluster`]
//! module) — it does not reimplement the clustering algorithms.
//!
//! [desktop-120]: https://github.com/findit-studio/desktop/issues/120
//!
//! macOS only (built on [`crate`]).
//!
//! Sans-I/O, like `whisper`: audio enters as 16 kHz mono `&[f32]`. Behind the
//! `speaker` feature the runtime clustering core is `diaric` — the backend-free,
//! `ort`-free crate that owns the clustering algorithms this module drives: the
//! offline pyannote-community-1 AHC→VBx pipeline
//! ([`diaric::offline::diarize_offline`]) and the online FluidAudio
//! `SpeakerManager`-parity centroid matcher
//! ([`diaric::cluster::online::OnlineClusterer`]), plus PLDA projection and
//! reconstruction. So this module DOES assign speaker labels at runtime: cluster
//! an [`extract::Extraction`] through its public
//! [`extract::Extraction::diarize`] / [`extract::Extraction::diarize_with`] /
//! [`extract::Extraction::diarize_online`] methods (backend selection lives in
//! the [`cluster`] module), each returning a speaker-labelled
//! [`diaric::offline::OfflineOutput`]. `dia` (the former speakerkit `dia`
//! feature) is NOT the runtime dependency — it is the test-only DER reference
//! oracle behind `speaker-oracle`, which alone pulls dia's `ort` inference.
//!
//! ```no_run
//! use coremlit::audio::speaker::extract::Options;
//! use coremlit::audio::speaker::source::{AnySource, ModelSource, Source};
//! # let audio: Vec<f32> = vec![0.0; 16_000];
//! let options = Options::new().with_source(Source::FluidAudio); // the default
//! let source = AnySource::load("Models/speakerkit", options)?;
//! let extraction = source.extract(&audio)?;
//! // cluster behind `speaker`: usual entry `extraction.diarize(&plda)?` (speaker-
//! // labelled spans); lower-level bridge `into_offline_input(&plda) -> diaric::offline::OfflineInput`
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Multi-source backend
//!
//! As of the multi-source split
//! (`docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`),
//! the segmentation/embedding pipeline above is [`source::ModelSource`]'s
//! first implementation ([`source::FluidAudioSource`]), not the only one — see
//! [`source`] for the pluggable-source abstraction. Two user-selectable CoreML
//! conversions of (functionally) the same pyannote pipeline sit behind one
//! [`extract::Options::with_source`] switch — genuinely different code paths
//! (argmax bakes preprocessing/decoding into its graph; FluidAudio does host-
//! side powerset/mask/window decode), not interchangeable weight swaps:
//!
//! - **FluidAudio** (`Source::FluidAudio`, default; `FluidInference/speaker-diarization-coreml`):
//!   powerset log-probs `[1,589,7]`, host-side decode; seg 99.97% / embed
//!   cosine 0.99999989 vs fp32 dia-ort. **Validated** default.
//! - **argmax** (`Source::Argmax`, optional; `argmaxinc/speakerkit-coreml`):
//!   already-decoded in-graph; tensor-fidelity **validated** bit-exact vs
//!   argmax's own Swift, but clustering parity **CHARACTERIZED, NOT VALIDATED**.
//!
//! ## ⚠ Two safety-critical, test-pinned caveats
//!
//! **1. Do not use `Source::Argmax` on multi-speaker audio.** Measured: 0.0000%
//! standard-collar DER at 1-2 speakers, then 3.3-9.3% DER on three of four
//! multi-speaker clips (e.g. `14_mrbeast` 9.29% finding 5 of 4 speakers). The
//! error is ~100% confusion — argmax's embedder consumes an 80-mel spectrogram
//! from its own preprocessor, landing in a differently-scaled space the
//! **frozen, pretrained** community-1 PLDA (fixed 0.6 AHC linkage) was never
//! fit for. The ~0.94 embedding cosine is NOT benign at DER; any earlier claim
//! it was is retracted (it was measured only on 1-2 speaker clips).
//!
//! **2. The FluidAudio default is not frame-exact on multi-speaker audio
//! either** — 0.0000% vs dia-ort on every ≤2-speaker clip and the 7-speaker
//! clip, but 0.12% / 0.39% on `12`/`14`, over this module's 0.1% DER-parity
//! bound. It never gets the speaker *count* wrong and is ~23× more faithful
//! than argmax; the "0.1% parity" claim was only ever tested on 1-2 speaker
//! audio. Both limitations are pinned in `tests/speaker/parity_e2e.rs`, which
//! fires if behavior moves in either direction (a fix must be a deliberate
//! re-baseline, not a silent pass).
//!
//! **What "DER" means here:** the reference RTTMs are pyannote.audio 4.0.4's
//! own OUTPUT (captured by `dia`), *not* human annotation — every DER number is
//! a distance to the reference implementation, never to ground truth.
//!
//! # Compute units, gates, and licensing
//!
//! Both sources default to `ComputeUnits::All`; every tensor-fidelity gate pins
//! `CpuOnly` for determinism, and the end-to-end DER gate runs on the shipping
//! `All` and asserts the placement changes no *decision* (pairwise `der_std ==
//! 0`) on the 1-2 speaker clips. For bit-reproducibility, pin `CpuOnly`
//! explicitly and record the placement. Run the model-gated suites:
//!
//! ```text
//! SPEAKERKIT_TEST_MODELS=Models/speakerkit \
//!   cargo test -p coremlit --features speaker -- --ignored          # seg/embed/argmax
//! cargo test -p coremlit --features speaker-oracle --test speaker_parity_e2e -- --ignored
//! crates/coremlit/tests/speaker/der_gate_inventory.sh               # DER binaries compiled + hermetic suite
//! ```
//!
//! The DER binaries are `#![cfg(feature = "speaker-oracle")]` (dia's own ort
//! path as the oracle); without it they compile to nothing. Model attribution
//! (pyannote community-1 **CC-BY-4.0 — attribution required**, segmentation-3.0
//! MIT, WeSpeaker, and the argmax repo's **undeclared** license) is recorded in
//! the crate `NOTICE` (§4).
//!
//! # Clustering
//!
//! An [`extract::Extraction`] is not the end of the road: as of the clustering
//! phase this crate drives `diaric`'s clustering directly, turning the extracted
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
//! use coremlit::audio::speaker::extract::Options;
//! use coremlit::audio::speaker::source::{AnySource, ModelSource, Source};
//! use coremlit::audio::speaker::{ClusterBackend, OfflineOptions, OnlineOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let audio: Vec<f32> = vec![0.0; 16_000]; // 16 kHz mono; this crate does no I/O.
//! let source =
//!   AnySource::load("Models/speakerkit", Options::new().with_source(Source::FluidAudio))?;
//! let extraction = source.extract(&audio)?;
//!
//! // Offline (the default backend): diaric's pyannote-community-1 AHC→VBx pipeline,
//! // through the frozen community-1 PLDA projection it clusters in.
//! let plda = diaric::plda::PldaTransform::new()?;
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
/// The speaker module deliberately does NOT re-export diaric's batch-clusterer
/// vocabulary (`OfflineClusterOptions`/`OfflineMethod`/`Linkage`) that T1 briefly
/// exposed: [`ClusterBackend::Offline`] wraps diaric's pyannote-parity *pipeline*
/// ([`extract::Extraction::diarize`] → `diaric::offline::diarize_offline`), not the
/// batch clusterer those types configure. The [`cluster`] module documents which
/// diaric entry point `Offline` wraps and why the batch vocabulary was removed
/// (design spec AMENDMENT 2026-07-16).
pub use cluster::{ClusterBackend, OfflineOptions, OnlineOptions, ParseClusterBackendError};
