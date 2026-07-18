//! Silero VAD on CoreML (feature `vad`) — the FluidInference unified 256 ms
//! artifact (`silero-vad-unified-256ms-v6.2.1`), run through the [`crate`]
//! runtime instead of ONNX Runtime, with all voice-activity *detection* logic
//! single-homed in the published `silero` crate behind a backend seam.
//!
//! Design spec:
//! `docs/superpowers/specs/2026-07-18-vadkit-design.md` (§4 model layer, §5
//! adoption, §6 gates). Plan:
//! `docs/superpowers/plans/2026-07-18-vadkit-plan.md`.
//!
//! macOS only (built on [`crate`]).
//!
//! # Scope: model layer only, no detection logic
//!
//! This module wraps ONE stateful CoreML graph — 256 ms (4096-sample) chunks
//! of 16 kHz mono audio in, one speech probability per chunk out, with the
//! recurrent LSTM state carried across chunks — and the 64-sample context
//! stitching that graph expects (the FluidAudio `VadManager` semantics,
//! `FluidAudio/Sources/FluidAudio/VAD/VadManager.swift:21-26`). It authors
//! **zero** speech-detection or streaming-segmentation logic: that lives, and
//! stays single-homed, in the published `silero` crate behind its backend
//! seam (spec §2-§3). [`CoreMlBackend`] implements that seam over CoreML and
//! [`detect_speech`] plus the re-exported [`silero`] detector surface
//! ([`SpeechOptions`], [`SpeechSegment`], [`SpeechSegmenter`],
//! [`detect_speech_with`]) wire it up (spec §4) — so a consumer gets the full
//! offline + streaming detection API with zero segmentation logic authored
//! here. The `src/audio/vad/` grep gate in `tests/vad/reexport.rs` pins that
//! single-home invariant.
//!
//! The module depends on `silero` with `default-features = false` (logic
//! only), so **`ort`/ONNX never enters the `vad` runtime graph** — nor a
//! downstream `whisper`'s. The ONNX stack appears only behind the DEV/TEST
//! `vad-bundled` feature (`silero/bundled`), for the cross-backend gate. The
//! git-pinned `silero` re-pins to `silero = "0.5.0"` once it publishes (no
//! behavior change).
//!
//! ```no_run
//! use coremlit::audio::vad::{CoreMlBackend, SpeechOptions, detect_speech};
//! # let samples: Vec<f32> = vec![0.0; 16_000];
//! let mut backend =
//!   CoreMlBackend::load("Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc")?;
//! for seg in detect_speech(&mut backend, &samples, SpeechOptions::default())? {
//!   println!("speech {:.2}s..{:.2}s", seg.start_seconds(), seg.end_seconds());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The `whisper` module consumes this one behind the `whisper`+`vad`
//! composition: `audio::whisper::silero_vad::SileroVad` plugs the Silero model
//! into whisper's own frame-level VAD seam for long-form chunking.
//!
//! # Model & geometry
//!
//! Adopted from Hugging Face, revision-pinned, never republished:
//! `FluidInference/silero-vad-coreml` rev
//! `b419383c55c110e2c9271fa6ee0ea83d03c70d96`, artifact
//! `silero-vad-unified-256ms-v6.2.1.mlmodelc` (ships pre-compiled), MIT. The
//! revision and per-file SHA-256 are pinned in `tests/vad/model_io.rs`; the
//! model is not committed (`Models/vadkit/` is gitignored, fetched dev-time,
//! `VADKIT_TEST_MODELS` overrides the path). I/O contract (all f32, pinned):
//! `audio_input [1, 4160]` (64 context + 4096 new) → `vad_output [1, 1, 1]`
//! (a noisy-OR of eight sigmoids); the recurrent LSTM state is explicit
//! feature I/O (`hidden_state`/`cell_state [1, 128]`, an empty `stateSchema`
//! — not an `MLState` model). One probability per 256 ms — an 8× coarser
//! frame than `silero`'s ONNX geometry, consumed unchanged by its
//! geometry-parameterized detector.
//!
//! # Compute placement, oracles & gates
//!
//! Defaults to `ComputeUnits::All`; the module states MEASURED behavior rather
//! than marketing placement — every `ComputeUnits` selection produces
//! bit-identical output on the fixture audio (worst |Δ| = 0), and the tail is
//! LSTM-dominated (CPU-placed). Pinned against the real FluidAudio Swift
//! `VadManager`: committed per-chunk probability traces
//! (`tests/vad/parity_swift.rs`, `tests/vad/fixtures/golden_swift/`, worst
//! |Δ| = 0 across 217 chunks) regenerable via `tests/vad/swift/regen_goldens.sh`;
//! the exact I/O + state contract in `tests/vad/model_io.rs` /
//! `tests/vad/model_state.rs`; the no-duplication + re-export gate
//! (`tests/vad/reexport.rs`); the cross-backend characterization against the
//! `silero` ONNX stack (`tests/vad/cross_backend.rs`, feature `vad-bundled`);
//! and the fp16-guard sweep in the crate's `tests/fp16_guards.rs` (the graph
//! is fp16-clean). Model-gated tests are `#[ignore]`d.
//!
//! # Licensing
//!
//! MIT end to end — see the crate `NOTICE` (§1-2) for the two model
//! attributions (upstream Silero VAD, and FluidInference's CoreML
//! conversion). The Rust source is MIT OR Apache-2.0.

pub mod backend;
pub mod error;
pub mod model;

pub use backend::{CoreMlBackend, detect_speech};
pub use error::{InferError, ModelError};
pub use model::{
  CHUNK_SAMPLES, CONTEXT_SAMPLES, MODEL_INPUT_SAMPLES, STATE_SIZE, VadModel, VadModelOptions,
  VadState,
};

// The silero detector surface, re-exported unchanged and wired (via
// [`CoreMlBackend`] / [`detect_speech`]) to run over the CoreML backend. vadkit
// adds NO detection logic (spec §2-§4); these are silero's own types. `Error`
// / `Result` are silero's detector error (into which the model layer's
// [`InferError`] bridges through [`silero::Error::Backend`]), distinct from the
// model-layer [`ModelError`] / [`InferError`] above.
pub use silero::{
  Error, Result, SampleRate, SpeechDetector, SpeechOptions, SpeechSegment, SpeechSegmenter,
  VadBackend, detect_speech_with,
};
