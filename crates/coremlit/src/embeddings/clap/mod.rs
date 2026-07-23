//! Native CoreML LAION **CLAP** (`clap-htsat-unfused`) inference: an audio
//! encoder and a text encoder that project into a shared 512-dim joint
//! embedding space, plus the improved long-audio pipeline textclap lacks —
//! overlapped chunking ([`window`]), customizable window-embedding aggregation
//! ([`aggregate`](mod@aggregate)), and zero-shot scoring ([`score`](mod@score)).
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
//! The crate never claims ANE residency for the audio tower.
//!
//! The per-tower **defaults are measure-then-pin** (the `clap_encode` bench, issue
//! #30): the **audio** default is [`crate::ComputeUnits::All`] (no non-`All` unit
//! is meaningfully faster — see [`audio::DEFAULT_AUDIO_COMPUTE`]); the **text**
//! default is [`crate::ComputeUnits::CpuAndGpu`], ~43 % faster warm than `All`
//! because the tiny text graph pays ANE-dispatch overhead on `All` (see
//! [`text::DEFAULT_TEXT_COMPUTE`]). Both stay overridable per encoder via
//! `with_compute` / `set_compute`; the full latency × placement table lives in
//! `tests/clap/placement.rs` and is reproduced by the bench.
//!
//! # Performance: construct once, reuse, prewarm (measured, never marketed)
//!
//! The committed `clap_encode` bench (`benches/clap/encode.rs`) separates the four
//! cost phases per tower × placement, model-gated. Regenerate with
//! `CLAPKIT_TEST_MODELS=… cargo bench -p coremlit --features clap --bench clap_encode`
//! (the numbers below are Apple M1 Max / macOS 26.5, fp16 — the bench is the
//! reproducible source of truth, not this prose).
//!
//! **Construct each encoder once and reuse it.** An [`AudioEncoder`] /
//! [`TextEncoder`] loads its CoreML model at construction and reuses that one
//! resident [`crate::Model`] across every `embed*` call (`&self` inference, no
//! per-call load). Reconstructing per request would re-pay the whole load below —
//! don't.
//!
//! **Prewarm before the first user-facing request.** Construction pays the model
//! *load* / device specialization, but the *first* `embed` still pays the
//! prediction path's own graph specialization — measured at several × the warm
//! latency (audio ~200 ms first vs ~48 ms warm; text ~120 ms first vs ~17 ms
//! warm). [`AudioEncoder::prewarm`] / [`TextEncoder::prewarm`] run one throwaway
//! inference to absorb that up front, so the first real request is warm. The
//! recipe:
//!
//! ```no_run
//! # use coremlit::embeddings::clap::{AudioEncoder, TextEncoder};
//! # fn demo(audio_path: &str, text_path: &str) -> Result<(), coremlit::embeddings::clap::Error> {
//! let audio = AudioEncoder::from_file(audio_path)?; // load (one-time)
//! let text = TextEncoder::from_file(text_path)?;
//! audio.prewarm()?; // absorb first-inference specialization before serving
//! text.prewarm()?;
//! // …reuse `audio` / `text` for every request from here on.
//! # Ok(())
//! # }
//! ```
//!
//! **Cold vs cached load (the one-time specialization).** The first-ever load of
//! these graphs on a given device pays a large **one-time** OS specialization cost
//! (the #30 audit measured ~24.2 s audio / ~7.25 s text on `All`, vs sub-second
//! for the ONNX reference). It is cached thereafter: on an already-specialized
//! host every later process start is fast (measured cached loads: text ~0.3–0.7 s,
//! audio ~0.05–0.16 s on `CpuOnly`/`CpuAndGpu`), and the specialization survives
//! process restarts — clearing the ANE (`e5rt`) cache barely moved load time
//! (audio-`All` 2.0 s → 2.3 s, text-`All` 0.28 s → 0.34 s), so the bulk of it is
//! cached outside that ANE cache and persists. Net: budget the one-time
//! specialization at install/first-run, and prewarm at process start so the
//! steady-state cached-load + warm-inference path is what users actually hit.
//! (The audio tower's ANE-naming placements re-attempt the failing HTSAT
//! `ANECCompile` on *every* load — see [`audio::DEFAULT_AUDIO_COMPUTE`] — an
//! `All`-only load cost, not a per-request one.)
//!
//! **int8 is a size tier, not a speed tier.** Its weight-only quantization keeps
//! the fp16 activations, so it is not faster (the audit measured text ~21 %
//! *slower*); use it to halve on-disk / weight memory, not to cut latency.
//!
//! # Long-audio pipeline
//!
//! A clip longer than one 10 s window is handled in three composable, sans-model
//! steps (nothing is hidden inside a monolithic pipeline object). The window
//! GEOMETRY and per-window AGGREGATION ride the generic `windit` engine; the mel
//! `repeatpad` front-end and the golden-pinned serde config types stay clap's own
//! (windit's `serde` feature is off):
//!
//! 1. [`WindowPlan`] maps the clip length to overlapped [`Span`]s via
//!    `windit::plan::WindowPlan` plus two clap-contract guards (short-clip and
//!    multi-tail; see [`window`]). Its hop / tail policy keep clap's own
//!    serde-validated construction.
//! 2. [`AudioEncoder::embed_windows`] slices each span and embeds it into a
//!    [`WindowEmbedding`] (`windit::windowed::WindowEmbedding<Embedding>`) — the
//!    mel front-end `repeatpad`s a short tail up to the fixed window. These
//!    per-window embeddings are always returned to the caller.
//! 3. An [`AggregatePolicy`] (windit's object-safe trait) combines them into one
//!    clip embedding via [`aggregate()`]. The built-ins [`MeanRenormalized`],
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
  aggregate,
};
pub use audio::{AudioEncoder, AudioEncoderOptions};
pub use embedding::Embedding;
pub use error::Error;
pub use score::{LabeledScore, ScoreMode, TextAnchor, score, score_windows};
pub use text::{TextEncoder, TextEncoderOptions};
pub use window::{Span, TailPolicy, WindowEmbedding, WindowPlan};

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
