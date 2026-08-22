//! Silero VAD on CoreML (feature `vad`) — the FluidInference unified 256 ms
//! artifact (`silero-vad-unified-256ms-v6.2.1`), run through the [`crate`]
//! runtime instead of ONNX Runtime, with all voice-activity *detection* logic
//! single-homed in the published `zuoer` crate behind a backend seam.
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
//! stays single-homed, in the published `zuoer` crate behind its backend
//! seam (spec §2-§3). [`CoreMlBackend`] implements that seam over CoreML and
//! [`detect_speech`] plus the re-exported [`zuoer`] detector surface
//! ([`SpeechOptions`], [`SpeechSegment`], [`SpeechSegmenter`],
//! [`detect_speech_with`]) wire it up (spec §4) — so a consumer gets the full
//! offline + streaming detection API with zero segmentation logic authored
//! here. The `src/audio/vad/` grep gate in `tests/vad/reexport.rs` pins that
//! single-home invariant.
//!
//! The module depends on `zuoer`, which owns no model, no inference runtime
//! and no dependencies at all, so **`ort`/ONNX never enters the `vad` runtime
//! graph** — nor a downstream `whisper`'s. The ONNX stack appears only behind
//! the DEV/TEST `vad-bundled` feature of the sibling `coremlit-parity` package,
//! the only thing that pulls the `silero` crate (`silero/bundled`), for the
//! cross-backend gate.
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
//! # Two spellings, one set of types
//!
//! zuoer names its detector types for the general thing they detect — a [`Run`]
//! of frames above threshold, assembled by a [`RunSegmenter`] configured with
//! [`RunOptions`] — and declares the speech-flavoured names as plain type
//! **aliases** of those:
//!
//! | neutral          | speech alias(es)                        |
//! | ---------------- | --------------------------------------- |
//! | [`Run`]          | [`SpeechSegment`]                       |
//! | [`RunSegmenter`] | [`SpeechSegmenter`], [`SpeechDetector`] |
//! | [`RunOptions`]   | [`SpeechOptions`]                       |
//!
//! Both spellings are re-exported here and they are the SAME items, not
//! parallel types: rustdoc renders each type once under its neutral name with
//! the aliases pointing at it, and a value produced through one spelling is
//! consumed through the other with no conversion. So this is a naming choice,
//! never a behavioural one — pick by what the probabilities you feed in
//! actually mean. Driven by [`CoreMlBackend`] they are speech probabilities and
//! [`detect_speech`] plus the `Speech*` names read true. Driven by anything
//! else — one class column of a CED sound-event track is the worked example in
//! `audio::ced` — nothing about them is speech, and the `Run*` spelling is the
//! honest one.
//!
//! Those three names plus [`SampleRate`] (already neutral, and shared) and
//! [`Error`] / [`Result`] are the whole set needed to construct a
//! [`RunSegmenter`], push probabilities into it and read the [`Run`]s back, so
//! a consumer doing per-class segmentation never needs `zuoer` as a direct
//! dependency. The `detect_speech*` entry points are the speech-only part of
//! the surface: they own a [`VadBackend`], which is where the audio — and the
//! speech assumption — enters.
//!
//! # Segment confidence
//!
//! A [`SpeechSegment`] carries more than a timespan. `zuoer` accumulates the
//! model frame probabilities each segment was built from and hands them back as
//! [`mean_probability`](zuoer::Run::mean_probability) /
//! [`peak_probability`](zuoer::Run::peak_probability) — the segment's
//! confidence. They need no opt-in: [`SpeechSegment`] is a plain type **alias**
//! for [`zuoer::Run`], not a wrapper, so every accessor zuoer defines is already
//! on the values [`detect_speech`] returns.
//!
//! Three semantics decide what those numbers mean, and none of them can be read
//! off the signature. They are zuoer's contract, not this module's invention —
//! [`zuoer::Run`](zuoer::Run#probability-aggregates) is authoritative, and this
//! summary must not be trusted over it:
//!
//! - **Padding is excluded.** The aggregate covers the segment's raw
//!   model-frame span only. [`speech_pad`](zuoer::RunOptions::speech_pad) widens
//!   the emitted `[start_sample, end_sample)` on both sides as a timeline
//!   courtesy; there are no observations out there to average.
//! - **Bridged frames are included.** A dip shorter than
//!   [`min_silence_duration`](zuoer::RunOptions::min_silence_duration) does not
//!   close the segment, and its frames stay in the mean and pull it down. That
//!   is the correct signal: a mean well below the peak is a segment with quiet
//!   stretches inside it, not a defect.
//! - **A force-split cuts the accumulator too.** When
//!   [`max_speech_duration`](zuoer::RunOptions::max_speech_duration)
//!   force-splits a long segment, the emitted segment carries only the
//!   pre-split aggregate and the continuation restarts its own; frames in the
//!   gap the split landed on appear in neither.
//!
//! Both values are finite and inside `[0, 1]` on every segment the segmenter
//! emits. A [`SpeechSegment`] built by hand has them zeroed — only
//! segmenter-emitted segments carry observations.
//!
//! [`SpeechSegmenter`] consumes probabilities rather than audio, so the
//! aggregates are demonstrable with no model at all. This is the same state
//! machine [`detect_speech`] drives over [`CoreMlBackend`], fed by hand at the
//! hop the backend reports:
//!
//! ```
//! use coremlit::audio::vad::{CHUNK_SAMPLES, SpeechOptions, SpeechSegmenter};
//!
//! // The upstream Silero defaults zuoer's hysteresis derives from: 0.5 start
//! // threshold, 250 ms minimum speech, 100 ms minimum silence, 30 ms pad.
//! let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
//! segmenter.set_frame_hop(CHUNK_SAMPLES); // one probability per 256 ms chunk
//!
//! // Frame 3 is a one-chunk dip inside the speech; frames 6-8 end it.
//! let mut segments = Vec::new();
//! for p in [0.02, 0.90, 0.80, 0.10, 0.85, 0.95, 0.02, 0.01, 0.01] {
//!   if let Some(segment) = segmenter.push_probability(p) {
//!     segments.push(segment);
//!   }
//! }
//! if let Some(segment) = segmenter.finish() {
//!   segments.push(segment);
//! }
//!
//! assert_eq!(segments.len(), 1);
//! let segment = segments[0];
//!
//! // The emitted span is padded: 480 samples (30 ms) either side of the raw
//! // model frames at 4096..24576.
//! assert_eq!((segment.start_sample(), segment.end_sample()), (3616, 25056));
//!
//! assert!((segment.peak_probability() - 0.95).abs() < 1e-6);
//! // Five raw frames, mean 0.72 — the bridged 0.10 is in it. Dropping that
//! // frame would read 0.875; the padding contributes to neither aggregate.
//! assert!((segment.mean_probability() - 0.72).abs() < 1e-4);
//! ```
//!
//! # Model & geometry
//!
//! Adopted from Hugging Face and revision-pinned:
//! `FluidInference/silero-vad-coreml` rev
//! `b419383c55c110e2c9271fa6ee0ea83d03c70d96`, artifact
//! `silero-vad-unified-256ms-v6.2.1.mlmodelc` (ships pre-compiled), MIT. The
//! revision and per-file SHA-256 are pinned in `tests/vad/model_io.rs`.
//!
//! Alone among this crate's models, it is COMMITTED to the coremlit repository
//! at `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc/` — 1.1 MiB, so
//! CI and every clone get it with no download step and the model gates run on
//! a fresh checkout (`VADKIT_TEST_MODELS` still overrides the path). That is a
//! REDISTRIBUTION: MIT permits it, and the notice it obliges is preserved in
//! the repository `NOTICE` (sections 1-2) and in a `LICENSE` file inside the
//! artifact directory. Nothing changes for a crates.io consumer — the vendored
//! tree sits outside the published package, so `cargo add coremlit` still
//! fetches the model itself. I/O contract (all f32, pinned):
//! `audio_input [1, 4160]` (64 context + 4096 new) → `vad_output [1, 1, 1]`
//! (a noisy-OR of eight sigmoids); the recurrent LSTM state is explicit
//! feature I/O (`hidden_state`/`cell_state [1, 128]`, an empty `stateSchema`
//! — not an `MLState` model). One probability per 256 ms — an 8× coarser
//! frame than the ONNX Silero geometry, consumed unchanged by `zuoer`'s
//! geometry-parameterized detector.
//!
//! Because zuoer's seam is push-based, [`CoreMlBackend`] also owns the input
//! windowing and the end-of-stream policy: it buffers the un-chunked PCM tail
//! across [`VadBackend::push`] calls and, on [`VadBackend::finish`],
//! zero-pads a non-empty trailing partial frame and emits its probability.
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
//! `silero` crate's ONNX stack (`coremlit-parity`'s
//! `tests/vad/cross_backend.rs`, feature `vad-bundled`);
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

// The zuoer detector surface, re-exported unchanged and wired (via
// [`CoreMlBackend`] / [`detect_speech`]) to run over the CoreML backend. vadkit
// adds NO detection logic (spec §2-§4); these are zuoer's own types. `Error`
// / `Result` are zuoer's detector error (into which the model layer's
// [`InferError`] bridges through [`zuoer::Error::Backend`]), distinct from the
// model-layer [`ModelError`] / [`InferError`] above.
pub use zuoer::{
  Error, Result, Run, RunOptions, RunSegmenter, SampleRate, SpeechDetector, SpeechOptions,
  SpeechSegment, SpeechSegmenter, VadBackend, detect_speech_with,
};
