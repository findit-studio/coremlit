//! EmbeddingGemma 2 on Core ML: one 768-d embedding space for text, audio
//! windows and images.
//!
//! The model is `google/embeddinggemma-2`, run through FluidInference's Core ML
//! conversion (`FluidInference/embeddinggemma-2-coreml` @ `3f978dd`): three
//! packages and two lifted tables.
//!
//! - **Text** (`EmbeddingGemma2Text.mlpackage`, the Neural Engine by
//!   default): seven functions over one set of weights, `embed_32` …
//!   `embed_512` and `pack_256`. The token table (`embeddings.bf16`) is read
//!   outside the graph, one row per id.
//! - **Audio** (`EmbeddingGemma2Audio.mlpackage`, fp16, or this crate's own
//!   fp32 conversion; the GPU by default): 16 kHz mono windows of 10 s to 250
//!   audio tokens.
//! - **Vision** (`EmbeddingGemma2Vision.mlpackage`, the GPU by default): an
//!   image's patches to 70, 140 or 280 image tokens. The patch-position tables
//!   (`position_embeddings.f16`) are read outside the graph.
//!
//! Audio and image tokens are embedded by the TEXT functions, as a sequence
//! framed by their start and end tokens, so every modality lands in the one
//! space and a text query ranks audio windows, images and texts alike.
//!
//! # Matryoshka
//!
//! [`Embedding`] is 768-d; [`Embedding::truncate`] keeps the first 512, 256 or
//! 128 coordinates, re-normalized. The width is the type's, so vectors of two
//! widths cannot be compared by accident.
//!
//! # Options
//!
//! [`TextOptions`], [`AudioOptions`] and [`ImageOptions`] carry every choice
//! the model and Core ML allow, each with an opinionated default, and
//! [`EmbedderOptions`] composes them. A part spells the choices that change
//! its vectors, and the label of the artifacts it runs, as its
//! [`DerivationTerms`].
//!
//! macOS 15 or later (the packages are Core ML specification 9).

pub mod error;

mod artifacts;
mod audio;
mod config;
mod embedding;
mod map;
mod options;
mod prompt;
mod table;
mod text;
mod tokenizer;
mod vision;

pub use artifacts::{ArtifactLabel, ParseArtifactLabelError, Part};
pub use audio::AudioPrecision;
pub use config::MAX_TOKENS;
pub use embedding::{EMBEDDING_DIM, Embedding, Matryoshka, Width};
pub use error::Error;
pub use options::{
  AudioOptions, Batching, DEFAULT_AUDIO_BATCHING, DEFAULT_AUDIO_COREML, DEFAULT_AUDIO_PRECISION,
  DEFAULT_BUDGET, DEFAULT_BUDGETS, DEFAULT_FOLD, DEFAULT_HOP_SAMPLES, DEFAULT_MIN_TAIL_SAMPLES,
  DEFAULT_RESIZE, DEFAULT_SPLIT_SPECIAL_TOKENS, DEFAULT_TEXT_BATCHING, DEFAULT_TEXT_COREML,
  DEFAULT_VISION_BATCHING, DEFAULT_VISION_COREML, DerivationTerms, EmbedderOptions, ImageOptions,
  TextOptions,
};
pub use prompt::Prompt;
pub use text::{TextFunction, TextFunctions};
pub use tokenizer::SplitSpecialTokens;
pub use vision::{ResizeRoad, VisionBudget, VisionBudgets};

/// The long-text geometry and input bound, shared with the crate's other text
/// doors; its window counts total ids and may not exceed [`MAX_TOKENS`].
pub use crate::embeddings::long_text::LongTextOptions;

/// How a long text's windows fold into one vector, shared with the crate's
/// other doors that fold windows.
pub use crate::embeddings::aggregate::{AggregatePolicyKind, EmaRenormalizedOptions};

/// The borrowed RGB8 image view the vision part takes, shared with the crate's
/// other image doors, and the refusal its constructor returns.
pub use crate::embeddings::image::{ImageGeometryError, Rgb8Image};

/// windit's window geometry and tail policy, carried by [`LongTextOptions`].
pub use windit::{TailPolicy, plan::WindowOptions};
