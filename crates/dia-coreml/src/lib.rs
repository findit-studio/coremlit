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

pub mod error;
pub mod segment;
