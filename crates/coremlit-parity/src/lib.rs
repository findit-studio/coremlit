//! Parity-oracle harness for [`coremlit`](../coremlit) — **not a library**.
//!
//! This package exists so `coremlit` itself can be published. Its parity gates
//! score coremlit's CoreML output against three third-party reference stacks:
//!
//! | oracle | crate | feature |
//! |---|---|---|
//! | pyannote DER reference (ONNX Runtime) | `dia` (`diarization`) | `speaker-oracle` |
//! | CLAP-HTSAT model-level parity (ONNX Runtime) | `textclap` | `clap-oracle` |
//! | Silero VAD cross-backend characterization (ONNX Runtime) | `silero` | `vad-bundled` |
//!
//! Two of those (`dia`, `textclap`) are unpublished rev-pinned **git**
//! dependencies, which `cargo publish` rejects outright — even behind an
//! optional feature. Hosting them here, under `publish = false`, keeps
//! `coremlit`'s own manifest publishable while the gates keep running
//! unchanged.
//!
//! Everything lives in `tests/`; this lib target exists only because Cargo
//! requires a package to have one. The shared test-support modules
//! (`common`, `der_calc`) are NOT copied here — each oracle binary
//! `#[path]`-includes the single copy that lives in `crates/coremlit/tests/`,
//! which the 13 non-oracle binaries that stayed behind also use.
