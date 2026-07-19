//! Embedding producers.
//!
//! A namespace of feature-gated submodules mirroring [`crate::audio`], each a
//! self-contained CoreML embedding pipeline built over the always-compiled
//! runtime core (`Model` / `MultiArray` / `Features`). Enabling a module's
//! feature is the only way it compiles, and `default = []` pulls none of them.
//!
//! - `clap` — CLAP-HTSAT dual-tower audio+text embeddings, a former standalone
//!   kit crate collapsed here per the mono-crate restructure (feature `clap`;
//!   `clap-oracle` adds the textclap parity oracle).
//! - `granite` / `gemma` — sentence-embedding backends (the embedkit phase,
//!   re-targeted from a crate to module form; feature `embeddinggemma`) — still
//!   reserved, not yet landed.
//!
//! See the crate README's layering map for module authority. The `video/`
//! sibling namespace is likewise reserved, but is not created until a video kit
//! exists (README note).

#[cfg(feature = "clap")]
pub mod clap;
