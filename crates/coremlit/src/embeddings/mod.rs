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
//! - `granite` — general text sentence-embeddings on CoreML (the embedkit
//!   phase), first model `granite-embedding-97m-multilingual-r2` (feature
//!   `granite`; committed transformers-fp32 goldens as the parity oracle — NO
//!   ort). The T2/T3 core is landed; long-input windowing (T4) is a later phase.
//!   `gemma` was DROPPED per the embedkit design's Amendment 3.
//! - `siglip` — SigLIP 2 (`siglip2-base-patch16-naflex`) dual-tower image+text
//!   embeddings on CoreML, a shared 768-dim joint space with cross-modal `rank`
//!   (feature `siglip`; committed transformers-fp32 goldens as the parity
//!   oracle — NO ort). NaFlex resizes natively to a fixed patch budget, so no
//!   windowing is used. The hermetic preprocessing + embedding core is landed;
//!   the model-gated parity gates await the staged conversion.
//!
//! See the crate README's layering map for module authority. The `video/`
//! sibling namespace is likewise reserved, but is not created until a video kit
//! exists (README note).

#[cfg(feature = "clap")]
pub mod clap;

#[cfg(feature = "granite")]
pub mod granite;

#[cfg(feature = "siglip")]
pub mod siglip;
