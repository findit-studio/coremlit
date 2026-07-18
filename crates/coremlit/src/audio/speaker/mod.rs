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
//! macOS only (built on [`crate`]).
//!
//! Sans-I/O, like `whisper`: audio enters as 16 kHz mono `&[f32]`. **Not a
//! standalone diarizer** — it never assigns a speaker label; behind the
//! `speaker` feature, [`extract::Extraction::into_offline_input`] bridges into
//! `dia::offline::diarize_offline` for the actual clustering (the former
//! speakerkit `dia` feature; `speaker-oracle` adds dia's own ort DER oracle).
//!
//! ```no_run
//! use coremlit::audio::speaker::extract::Options;
//! use coremlit::audio::speaker::source::{AnySource, ModelSource, Source};
//! # let audio: Vec<f32> = vec![0.0; 16_000];
//! let options = Options::new().with_source(Source::FluidAudio); // the default
//! let source = AnySource::load("Models/speakerkit", options)?;
//! let extraction = source.extract(&audio)?;
//! // behind `speaker`: extraction.into_offline_input(&plda) -> dia::offline::OfflineInput
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

#[cfg(feature = "serde")]
mod compute_units_serde;
pub mod embed;
pub mod error;
pub mod extract;
pub mod segment;
pub mod source;
pub mod window;
