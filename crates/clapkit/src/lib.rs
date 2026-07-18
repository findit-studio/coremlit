//! Native CoreML LAION **CLAP** (`clap-htsat-unfused`) inference: an audio
//! encoder and a text encoder that project into a shared 512-dim joint
//! embedding space, plus (in later tasks) an improved long-audio pipeline
//! textclap lacks.
//!
//! Design spec: `docs/superpowers/specs/2026-07-18-clapkit-design.md` (including
//! its `AggregatePolicy` amendment). textclap is the model-level oracle: the
//! logic here is being *improved*, not merely reused (the asry→alignkit
//! relationship, not silero→vadkit).
//!
//! macOS only (built on [`coremlit`]).
//!
//! # Sample rate: 48 kHz (a documented deviation)
//!
//! CLAP is a **48 kHz** model. clapkit's audio API therefore takes 48 kHz mono
//! `&[f32]` — a deliberate, documented deviation from the workspace's 16 kHz
//! convention. Resampling to 48 kHz is the caller's responsibility (sans-I/O).
//!
//! # Model artifacts
//!
//! The two CoreML graphs are distributed on the Hugging Face Hub at
//! [`FinDIT-Studio/clapkit-coreml`](https://huggingface.co/FinDIT-Studio/clapkit-coreml),
//! revision `97d631f3814e1e46b798a8e88c9aa2e2202fdf67` (fp16, converted from
//! `laion/clap-htsat-unfused` — **CC-BY-4.0**, attribution required; see the
//! crate `NOTICE`). They are gitignored dev-time downloads under
//! `Models/clapkit/`; their SHA-256 and I/O contracts are pinned by
//! `tests/model_io.rs` / `tests/text_model_io.rs`. The conversion recipes live
//! under `conversion/`.
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
//! Placement is characterized, not asserted (`tests/placement.rs`). As converted
//! (T1), the **audio** (HTSAT Swin) graph does **not** compile for the ANE and
//! falls back to GPU/CPU; the **text** (RoBERTa) graph does compile for the ANE.
//! [`coremlit::ComputeUnits::All`] lets CoreML schedule either way; the crate
//! never claims ANE residency for the audio tower.

#![doc(html_root_url = "https://docs.rs/clapkit/0.1.0")]

pub mod audio;
#[cfg(feature = "serde")]
mod compute_units_serde;
pub mod embedding;
pub mod error;
pub mod text;

pub use crate::{
  audio::{AudioEncoder, AudioEncoderOptions},
  embedding::Embedding,
  error::Error,
  text::{TextEncoder, TextEncoderOptions},
};

/// Bytes of the pinned Xenova `tokenizer.json` bundled with the crate (~2 MB).
///
/// This is the **identical** artifact textclap pins (SHA-256
/// `dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5`,
/// `textclap/models/MODELS.sha256`), so tokenization is identity-comparable —
/// see `tests/tokenizer_identity.rs`. Exposed for callers who construct
/// [`TextEncoder`] via [`TextEncoder::from_memory`]; the
/// [`TextEncoder::from_bundled_tokenizer`] / [`TextEncoder::from_file`]
/// constructors use it directly.
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../assets/tokenizer.json");
