<div align="center">
<h1>coremlit</h1>

**On-device audio understanding for macOS in Rust**: a safe CoreML runtime layer, plus opt-in feature-gated pipelines — a faithful port of [WhisperKit](https://github.com/argmaxinc/WhisperKit), forced alignment, speaker diarization, and Silero VAD — in one crate.

[<img alt="CI" src="https://img.shields.io/github/actions/workflow/status/findit-studio/coremlit/ci.yml?branch=main&style=for-the-badge&logo=github-actions" height="22">](https://github.com/findit-studio/coremlit/actions/workflows/ci.yml)
<img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.95-orange?style=for-the-badge&logo=rust" height="22">
<img alt="license" src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue?style=for-the-badge" height="22">

</div>

## One crate, grouped modules, flat features

`coremlit` is a single crate. The runtime **core** is always compiled; each audio pipeline is a feature-gated module under `audio::`. `default = []` — the core pulls no pipeline dependencies; consumers opt in per feature.

| Module | Feature | What it is |
|---|---|---|
| `coremlit` (core) | always | Safe, synchronous CoreML runtime over `objc2-core-ml`: model load / compile / prewarm, prediction, stateful prediction (`MLState`), typed multi-arrays (incl. IOSurface-backed `f16`), eager I/O introspection. Every `unsafe` FFI call lives here behind a safe API. |
| [`audio::whisper`](crates/coremlit/src/audio/whisper) | `whisper` | The Whisper pipeline on CoreML: mel → encoder → autoregressive decoder with prefill, KV caching, temperature-fallback ladder; energy-VAD long-form chunking (opt-in Silero VAD via `whisper`+`vad`); scoped-thread batch pool; DTW word timestamps; push-based streaming with LocalAgreement-2; SRT/VTT/JSON writers. Token-for-token parity-tested against Swift WhisperKit on `openai_whisper-tiny`. |
| [`audio::align`](crates/coremlit/src/audio/align) | `align` (`align-oracle`) | CoreML wav2vec2 forced word-level alignment: audio + a known transcript → per-word time spans with confidence, over `asry`'s parity-tested alignment seam. |
| [`audio::speaker`](crates/coremlit/src/audio/speaker) | `speaker` (`speaker-oracle`) | CoreML segmentation + embedding backends for `dia`'s diarization: runs pyannote's `segmentation-3.0` and WeSpeaker on the ANE and produces the `dia`-shaped tensors (`Extraction`) that feed [`dia`](https://github.com/findit-studio/diarization)'s VBx/PLDA clustering. Multi-source (FluidAudio default + Argmax). Never assigns a speaker label — clustering stays in `dia`. |
| [`audio::vad`](crates/coremlit/src/audio/vad) | `vad` (`vad-bundled`) | Silero VAD on CoreML: runs the FluidInference unified 256 ms model and implements the published [`silero`](https://github.com/Findit-AI/silero) crate's `VadBackend` seam, re-exporting its detector so a consumer gets the full offline + streaming API with **zero** detection logic duplicated; `ort`/ONNX never enters the runtime graph. |
| [`embeddings`](crates/coremlit/src/embeddings) | `clap` / `granite` / `siglip` | Embedding producers, each a feature-gated CoreML pipeline projecting into a shared joint space, L2-normalized in Rust: CLAP-HTSAT audio+text (`clap`), granite sentence embeddings (`granite`), and SigLIP 2 (`siglip2-base-patch16-naflex`) image+text (`siglip`, NaFlex — no windowing). Parity against committed transformers-fp32 goldens, no `ort`. `video` is likewise reserved and **not** created until a video kit exists. |

## Layering map

The owner's architecture-confusion fix: who is authoritative for what, where the logic seams sit, and when a module's logic core gets pulled out into its own crate.

```
                         coremlit  (this crate — macOS/CoreML only)
   ┌───────────────────────────────────────────────────────────────────────┐
   │  core: Model / MultiArray / Features / State   (all unsafe FFI; safe API)│
   │                              ▲  ▲  ▲  ▲                                  │
   │        ┌─────────────────────┘  │  │  └─────────────────────┐           │
   │   audio::whisper          audio::align   audio::speaker   audio::vad     │
   │   (STT pipeline,          (encoder +      (CoreML seg +   (CoreML model  │
   │    authoritative)          asry seam)      embed backends) layer + wiring)│
   └────────┬───────────────────────┬───────────────┬───────────────┬────────┘
            │ whisper+vad            │ align          │ speaker        │ vad
            ▼ (opt-in)               ▼                ▼                ▼
      audio::vad                 asry  (git)    dia / diarization  silero (git)
   (Silero long-form chunking) (alignment seam:  (git; VBx/PLDA    (detector logic
                                emissions +       clustering —      single-home:
                                ONNX oracle)      backend-free      thresholding,
                                                  offline core)     hysteresis,
                                                                    segmentation)
```

**Authority.** The runtime **core** owns every CoreML FFI call. Each `audio::*` module owns its pipeline's CoreML execution and host-side glue, but **not** the backend-agnostic algorithm it drives:

- `audio::align` runs the CoreML CTC encoder; **`asry`** owns the tokenizer, silence mask, and CTC trellis/beam (the alignment vocabulary is re-exported from `asry`). `align-oracle` adds asry's ONNX aligner as the word-timing parity oracle.
- `audio::speaker` runs CoreML segmentation/embedding; **`dia`** owns clustering/PLDA/reconstruction. `speaker` pulls dia's **backend-free offline core** (no `ort`); `speaker-oracle` adds dia's own ort inference as the DER oracle.
- `audio::vad` runs the CoreML Silero graph and implements **`silero`**'s `VadBackend` seam; **`silero`** owns all detection logic. `vad-bundled` adds silero's ONNX reference stack (DEV/TEST only).

**Dependency arrows are rev-pinned git deps** (`asry`, `dia`/`diarization`, `silero`), gated behind their feature so a fresh, sibling-free clone resolves; co-develop against a local checkout via an uncommitted workspace-root `[patch]` (see `Cargo.toml`).

**Extraction triggers** (the `diaric` naming pattern — a model-branded crate's pure, backend-agnostic logic core is pulled into a standalone `*ic` crate). Two triggers fire an extraction:

1. **A second backend/consumer needs the logic core.** coremlit's `audio::speaker` depends on the pinned [`dia`](https://github.com/findit-studio/diarization) crate, which owns clustering/PLDA/reconstruction **in-tree** — the backend-free offline core (no `ort`) that `speaker` pulls; [`diaric`](https://github.com/findit-studio/diaric) is a SEPARATE downstream extraction lineage — a different consumer's pull of that logic core, **not** a coremlit dependency and not the authority for coremlit's speaker path. `vadic` is **RESERVED** for `silero`'s detector logic under the same pattern — today `silero` single-homes it and coremlit's `audio::vad` is its only CoreML consumer, so no VAD extraction has fired.
2. **The pure surface must escape backend-coupled CI/versioning.** The `--no-default-features` (ort/tch-free) surface moves out so it can build and publish free of the backend infrastructure's rot — the `diaric` split's second rationale.

`coremlit` is downstream of all three seams; it authors CoreML execution, never the algorithms.

## The contract: sans-I/O, synchronous, macOS

- **Audio enters as 16 kHz mono `&[f32]`.** The library never opens files or devices and never resamples. Decoding and capture belong to your app — [`examples/whisper/transcribe_wav.rs`](crates/coremlit/examples/whisper/transcribe_wav.rs) (hound) and [`examples/whisper/mic_stream.rs`](crates/coremlit/examples/whisper/mic_stream.rs) (cpal + rubato) show both sides; those crates are dev-dependencies, never library dependencies.
- **Synchronous.** No async runtime; batch transcription parallelizes internally with scoped threads over `Sync` backends. VAD long-form chunking on the CoreML backend runs each chunk sequentially — `CoreMlBackend` is deliberately not `Sync` (Apple's one-`MLModel`-per-thread contract).
- **macOS on Apple Silicon** (CI: `macos-15`). Per-stage compute-unit selection (`CPU`/`GPU`/`Neural Engine`); `MLState` stateful prediction requires macOS 15, probed at runtime (`Model::supports_state`).

## Quick start

Transcribe (the compile-checked doctest from the `whisper` module docs):

```rust,no_run
use coremlit::audio::whisper::options::{DecodingOptions, Options};
use coremlit::audio::whisper::transcribe::WhisperKit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // You provide 16 kHz mono samples — see examples/ for WAV + mic sources.
    let audio: Vec<f32> = vec![0.0; 16_000];

    let options = Options::new(
        "Models/whisperkit-coreml/openai_whisper-tiny",
        "Models/tokenizers/whisper-tiny",
    );
    let kit = WhisperKit::new(&options)?;
    let result = kit.transcribe(&audio, &DecodingOptions::new())?;
    println!("{}", result.text());
    Ok(())
}
```

Raw CoreML, without any pipeline (core, no features):

```rust,no_run
use coremlit::{ComputeUnits, DataType, Features, Model, MultiArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load("MelSpectrogram.mlmodelc", ComputeUnits::CpuAndGpu)?;
    let audio = MultiArray::zeros(&[480_000], DataType::F32)?;
    let outputs = model.predict(&Features::new().with("audio", audio))?;
    let mel = outputs.get("melspectrogram_features").unwrap();
    assert_eq!(mel.data_type(), DataType::F16);
    Ok(())
}
```

## Installation

Not yet on crates.io (release pending — `publish = false` while `asry`/`dia`/`silero` are git deps). Use a git dependency and enable the pipelines you need:

```toml
[dependencies]
coremlit = { git = "https://github.com/findit-studio/coremlit", features = ["whisper"] }
# add "align", "speaker", "vad" (and "serde"/"tracing") as needed; default = [] is the bare core.
```

## Getting models

Models are plain local folders — the library performs no downloads. Fetch the WhisperKit CoreML bundles and the matching tokenizer with the Hugging Face CLI:

```sh
hf download argmaxinc/whisperkit-coreml --include "openai_whisper-tiny/*" \
  --local-dir Models/whisperkit-coreml
hf download openai/whisper-tiny tokenizer.json tokenizer_config.json config.json \
  --local-dir Models/tokenizers/whisper-tiny
```

Each pipeline resolves its models root from its own env var (`WHISPERKIT_TEST_MODELS`, `ALIGNKIT_TEST_MODELS`, `SPEAKERKIT_TEST_MODELS`/`ARGMAX_TEST_MODELS`, `VADKIT_TEST_MODELS`), defaulting to `<repo>/Models/...` (gitignored). See each module's `tests/<kit>/model_io.rs` for the pinned repo id, revision, and per-file SHA-256, and the module docs for fetch commands. The model-gated suites load multi-hundred-MB CoreML models per test and libtest runs tests in one binary concurrently by default; on memory-constrained hosts (< 16 GB) append `--test-threads=1` after `--ignored` to run them serially.

## Examples & benches

| Command | What it shows |
|---|---|
| `cargo run -p coremlit --features whisper --example whisper_transcribe_wav -- [wav]` | File transcription with timestamps + timings (defaults to the committed JFK clip) |
| `cargo run -p coremlit --features whisper --example whisper_mic_stream` | Live mic streaming: cpal capture → rubato resample → `push_samples` |
| `cargo bench -p coremlit --features whisper --bench whisper_stages` | Hermetic criterion benches: logits filters, DTW, VAD chunking, compression ratio |
| `cargo bench -p coremlit --features whisper --bench whisper_rtf` | End-to-end tokens/sec + real-time factor on the tiny model (skips without models) |
| `cargo bench -p coremlit --features align --bench align_align` | Alignment encode / align_chunk RTF |
| `cargo bench -p coremlit --features clap --bench clap_encode` | CLAP dual-tower encode phases (first-observed / cached load, first + warm inference) per tower × ComputeUnits, with output hash / cosine / RSS (skips without models) |

## Feature flags

Flat and additive; `default = []`.

| Feature | Enables |
|---|---|
| `whisper` | the `audio::whisper` STT pipeline |
| `align` / `align-oracle` | forced alignment / + the asry ONNX word-timing parity oracle (DEV/TEST) |
| `speaker` / `speaker-oracle` | diarization backends + dia offline bridge / + dia's ort DER oracle (DEV/TEST) |
| `vad` / `vad-bundled` | Silero VAD model layer / + silero's ONNX cross-backend oracle (DEV/TEST) |
| `clap` / `granite` / `siglip` | embedding producers: CLAP audio+text / granite sentence / SigLIP 2 image+text — each committed-golden parity, no `ort` (`clap-oracle` adds the textclap ort oracle) |
| `serde` | `Serialize`/`Deserialize` on options/results/provenance (+ the whisper JSON writer) |
| `tracing` | internal log events additionally emitted as `tracing` events |

`whisper`+`vad` together light up `audio::whisper::silero_vad` (the former whisperkit `vadkit` feature). See [`crates/coremlit/FEATURE_MAP.md`](crates/coremlit/FEATURE_MAP.md) for the old-crate-feature → flat-feature rename table and the curated CI feature-combination list (pinned by the `feature_map` golden test).

## MSRV & platform

Rust **1.95**, edition 2024. macOS only (Apple Silicon primary; `x86_64-apple-darwin` untested). Not sandboxed-Linux-buildable by design — this is a CoreML binding.

## Acknowledgments & licensing

`audio::whisper` is a Rust port of [Argmax's WhisperKit](https://github.com/argmaxinc/WhisperKit) (MIT); the underlying model is [OpenAI's Whisper](https://github.com/openai/whisper). The forced aligner, diarization backends, and VAD build on the `asry`, `dia`, and `silero` seams respectively. Every third-party **model** attribution the crate's pipelines load at runtime — Silero/FluidInference, Whisper/argmax/OpenAI, pyannote community-1 (**CC-BY-4.0, attribution required**)/segmentation-3.0/WeSpeaker/argmax, and the chordai wav2vec2 aligner — is recorded in [`NOTICE`](NOTICE).

#### License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
