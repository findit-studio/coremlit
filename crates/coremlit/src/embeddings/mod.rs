//! Embedding producers — reserved namespace.
//!
//! A documented placeholder. The embedding pipelines land in later waves as
//! feature-gated submodules mirroring [`crate::audio`]:
//!
//! - `clap` — CLAP-HTSAT dual-tower audio+text embeddings (arrives with the
//!   clapkit port; features `clap` / `clap-oracle`).
//! - `granite` / `gemma` — sentence-embedding backends (the embedkit phase,
//!   re-targeted from a crate to module form; feature `embeddinggemma`).
//!
//! No embedding code compiles yet; the namespace is reserved so the crate's
//! module map and the README layering map name a stable home for it. The
//! `video/` sibling namespace is likewise reserved, but is not created until a
//! video kit exists (README note).
