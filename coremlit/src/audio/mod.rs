//! On-device audio-understanding pipelines.
//!
//! A namespace of feature-gated modules, each a self-contained CoreML pipeline
//! built over the always-compiled runtime core (`Model` / `MultiArray` /
//! `Features`). Every module is a former standalone kit crate, collapsed here
//! per the mono-crate restructure; enabling a module's feature is the only way
//! it compiles, and `default = []` pulls none of them.
//!
//! - `whisper` — Whisper speech-to-text (feature `whisper`).
//! - `align` — wav2vec2 forced word-level alignment (feature `align`;
//!   `align-oracle` adds the asry ONNX parity oracle).
//! - `speaker` — native CoreML segmentation/embedding backends for the `dia`
//!   diarization pipeline (feature `speaker`; the dia-ort DER oracle lives in
//!   the sibling `coremlit-parity` package's `speaker-oracle` feature).
//! - `vad` — Silero voice-activity detection (feature `vad`; the `silero`-crate
//!   ONNX cross-backend oracle lives in `coremlit-parity`'s `vad-bundled`).
//! - `ced` — CED (tiny/mini/small/base) AudioSet sound-event tagging (feature `ced`).
//! - `lid` — spoken-language identification (feature `lid`).
//! - `identity` — speaker-identity embedding, one fixed window at a time
//!   (feature `identity`). Distinct from `speaker`: that module's embedder is
//!   the diarization lane's batch-3, mask-taking, 256-d graph, pinned by a DER
//!   gate; this one embeds a single 6 s window into 192 raw dimensions and is
//!   additive to it.
//!
//! See the crate README's layering map for module authority and the dependency
//! arrows to the `zuoer`, `asry`, and `dia` seams.

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "align")]
pub mod align;

#[cfg(feature = "speaker")]
pub mod speaker;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "ced")]
pub mod ced;

#[cfg(feature = "lid")]
pub mod lid;

#[cfg(feature = "identity")]
pub mod identity;
