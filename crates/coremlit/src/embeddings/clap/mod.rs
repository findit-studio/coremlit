//! Native CoreML LAION **CLAP** (`clap-htsat-unfused`) inference: an audio
//! encoder and a text encoder that project into a shared 512-dim joint
//! embedding space, plus the improved long-audio pipeline textclap lacks —
//! overlapped chunking ([`window`]), customizable window-embedding aggregation
//! ([`aggregate`]), and zero-shot scoring ([`score`](mod@score)).
//!
//! Design spec: `docs/superpowers/specs/2026-07-18-clapkit-design.md` (including
//! its `AggregatePolicy` amendment). textclap is the model-level oracle: the
//! logic here is being *improved*, not merely reused (the asry→alignkit
//! relationship, not silero→vadkit).
//!
//! macOS only (built on [`crate`]).
//!
//! # Sample rate: 48 kHz (a documented deviation)
//!
//! CLAP is a **48 kHz** model. clapkit's audio API therefore takes 48 kHz mono
//! `&[f32]` — a deliberate, documented deviation from the workspace's 16 kHz
//! convention. Resampling to 48 kHz is the caller's responsibility (sans-I/O).
//!
//! # Model artifacts
//!
//! The CoreML graphs are distributed on the Hugging Face Hub at
//! [`FinDIT-Studio/clapkit-coreml`](https://huggingface.co/FinDIT-Studio/clapkit-coreml),
//! revision `02a99c6a8be21da1e9a947499ea503a10c80c4f1` (converted from
//! `laion/clap-htsat-unfused` — **CC-BY-4.0**, attribution required; see the
//! crate `NOTICE`). That revision ships **two tiers**, both loaded through the
//! same encoders (identical I/O contract; pick per tower at load time):
//!
//! - **fp16** (`clap_audio.mlmodelc` / `clap_text.mlmodelc`) — the validated,
//!   shipped default; every parity / placement / e2e gate runs on it.
//! - **int8** (`clap_audio_int8.mlmodelc` / `clap_text_int8.mlmodelc`) — a
//!   2×-smaller **opt-in** tier, measured-parity-clean per the int8 gates in
//!   `tests/clap/`. (The per-tower production default is the owner's decision;
//!   this crate records only that fp16 is validated and int8 is opt-in.)
//!
//! They are gitignored dev-time downloads under `Models/clapkit/`, fetched at the
//! immutable revision (never mutable `main`):
//!
//! ```text
//! hf download FinDIT-Studio/clapkit-coreml \
//!   --revision 02a99c6a8be21da1e9a947499ea503a10c80c4f1 \
//!   --local-dir Models/clapkit
//! ```
//!
//! Every artifact file's SHA-256 and the I/O contracts are pinned by
//! `tests/clap/model_io.rs` / `tests/clap/text_model_io.rs`.
//!
//! # Encoder split: Rust front-ends around fp16 CoreML graphs
//!
//! Each graph emits the **pre-normalization** joint embedding; clapkit applies
//! the final L2 normalization in Rust (keeping the fp16 rsqrt-guard class out of
//! the graphs). The audio graph takes a log-mel **spectrogram** (`[1, 1, 1001,
//! 64]`), not raw audio: the mel/STFT front-end is a Rust port of textclap's
//! `mel.rs` (see [`audio`]). The text graph takes tokenized `input_ids` /
//! `attention_mask` (`[1, 512]`), produced from the pinned Xenova tokenizer (see
//! [`text`]).
//!
//! # Compute placement (measured, never marketed)
//!
//! Placement is characterized, not asserted (`tests/clap/placement.rs`). As converted
//! (T1), the **audio** (HTSAT Swin) graph does **not** compile for the ANE and
//! falls back to GPU/CPU; the **text** (RoBERTa) graph does compile for the ANE.
//! [`crate::ComputeUnits::All`] lets CoreML schedule either way; the crate
//! never claims ANE residency for the audio tower.
//!
//! # Long-audio pipeline
//!
//! A clip longer than one 10 s window is handled in three composable, sans-model
//! steps (nothing is hidden inside a monolithic pipeline object):
//!
//! 1. [`WindowPlan`] maps the clip length to overlapped [`WindowSpan`]s
//!    (configurable hop; explicit tail policy; serde-validated construction).
//! 2. [`AudioEncoder::embed_windows`] embeds each span into a [`WindowEmbedding`]
//!    (embedding + span + tail-padding-aware coverage). These per-window
//!    embeddings are always returned to the caller.
//! 3. An [`AggregatePolicy`] combines them into one clip embedding. The seam is
//!    an **open trait**: the built-ins [`MeanRenormalized`] (default),
//!    [`EmaRenormalized`], and [`CoverageWeightedMean`] ship, and end users
//!    implement the trait for their own strategies. A serde-able
//!    [`AggregatePolicyKind`] names the built-ins for config surfaces.
//!
//! [`score()`] ranks text labels against any embedding (a window's or the
//! aggregate's) by audio↔text cosine, raw or CLAP-logit-scaled; per-window
//! scoring ([`score_windows`]) is exposed so score-level smoothing stays
//! caller-side.

pub mod aggregate;
pub mod audio;
#[cfg(feature = "serde")]
mod compute_units_serde;
pub mod embedding;
pub mod error;
pub mod score;
pub mod text;
pub mod window;

pub use aggregate::{
  AggregatePolicy, AggregatePolicyKind, CoverageWeightedMean, EmaRenormalized, MeanRenormalized,
};
pub use audio::{AudioEncoder, AudioEncoderOptions};
pub use embedding::Embedding;
pub use error::Error;
pub use score::{LabeledScore, ScoreMode, TextAnchor, score, score_windows};
pub use text::{TextEncoder, TextEncoderOptions};
pub use window::{TailPolicy, WindowEmbedding, WindowPlan, WindowSpan};

/// Bytes of the pinned Xenova `tokenizer.json` bundled with the crate (~2 MB).
///
/// This is the **identical** artifact textclap pins (SHA-256
/// `dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5`,
/// `textclap/models/MODELS.sha256`), so tokenization is identity-comparable —
/// see `tests/clap/tokenizer_identity.rs`. Exposed for callers who construct
/// [`TextEncoder`] via [`TextEncoder::from_memory`]; the
/// [`TextEncoder::from_bundled_tokenizer`] / [`TextEncoder::from_file`]
/// constructors use it directly.
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("assets/tokenizer.json");
