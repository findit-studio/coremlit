<div align="center">
<h1>coremlit</h1>

**On-device speech-to-text for macOS in Rust**: a safe CoreML runtime layer, and a faithful port of [WhisperKit](https://github.com/argmaxinc/WhisperKit).

[<img alt="CI" src="https://img.shields.io/github/actions/workflow/status/findit-studio/coremlit/ci.yml?branch=main&style=for-the-badge&logo=github-actions" height="22">](https://github.com/findit-studio/coremlit/actions/workflows/ci.yml)
<img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.95-orange?style=for-the-badge&logo=rust" height="22">
<img alt="license" src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue?style=for-the-badge" height="22">

</div>

## Crates

| Crate | What it is |
|---|---|
| [`coremlit`](crates/coremlit) | Safe, synchronous CoreML runtime layer over `objc2-core-ml`: model load / compile / prewarm, prediction, stateful prediction (`MLState`), typed multi-arrays (including IOSurface-backed `f16` for the Neural Engine), eager I/O introspection. Every `unsafe` FFI call lives inside this crate behind a safe API. |
| [`whisperkit`](crates/whisperkit) | The Whisper pipeline on CoreML: mel → encoder → autoregressive decoder with prefill, KV caching, and the temperature-fallback ladder; energy-VAD long-form chunking (sequential per chunk on the CoreML backend); a scoped-thread worker pool for batch transcription over `Sync` backends; DTW word timestamps; push-based streaming with confirmed/unconfirmed promotion and LocalAgreement-2; SRT/VTT/JSON writers. Token-for-token parity-tested against Swift WhisperKit's CLI on `openai_whisper-tiny`. |
| [`speakerkit`](crates/speakerkit) | Speaker-diarization front-end on CoreML: runs pyannote's `segmentation-3.0` net and the WeSpeaker embedder on the Neural Engine (via `coremlit`) and produces the `dia`-shaped tensors (`Extraction`) that feed [`dia`](https://github.com/findit-studio/diarization)'s Rust VBx/PLDA clustering. Multi-source — the FluidAudio (default) and Argmax model layouts behind one `Source` API — with host-side powerset/mask/window decode ported from `dia`. Not a standalone diarizer: it never assigns a speaker label, and clustering stays in `dia` by design; behind the optional `dia` feature, `Extraction::into_offline_input` bridges straight into `dia::offline::diarize_offline`. DER-parity-gated end to end against pyannote-output references through `dia`'s clustering. |

## The contract: sans-I/O, synchronous, macOS

- **Audio enters as 16 kHz mono `&[f32]`.** The library never opens files or devices and never resamples. Decoding and capture belong to your app — [`examples/transcribe_wav.rs`](crates/whisperkit/examples/transcribe_wav.rs) (hound) and [`examples/mic_stream.rs`](crates/whisperkit/examples/mic_stream.rs) (cpal + rubato) show both sides of the boundary; those crates are dev-dependencies here, never library dependencies.
- **Synchronous.** No async runtime anywhere; batch transcription (`transcribe_all`) parallelizes internally with scoped threads over `Sync` backends. VAD long-form chunking on the CoreML backend runs each chunk sequentially instead — `CoreMlBackend` is deliberately not `Sync` (Apple's contract is one `MLModel` on one thread at a time), so it can never satisfy `transcribe_all`'s bound; see `WhisperKit::transcribe`'s docs.
- **macOS on Apple Silicon** (CI: `macos-15`). Compute-unit selection (`CPU`/`GPU`/`Neural Engine`) per model stage; `MLState`-based stateful prediction requires macOS 15 and is probed at runtime (`Model::supports_state`).

## Quick start

Transcribe (the snippet is the compile-checked doctest from `whisperkit`'s crate docs):

```rust,no_run
use whisperkit::options::{DecodingOptions, Options};
use whisperkit::transcribe::WhisperKit;

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

Stream from pushed samples:

```rust,no_run
use whisperkit::options::{DecodingOptions, Options};
use whisperkit::transcribe::WhisperKit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::new(
        "Models/whisperkit-coreml/openai_whisper-tiny",
        "Models/tokenizers/whisper-tiny",
    );
    let kit = WhisperKit::new(&options)?;
    let mut streamer = kit.audio_stream_transcriber(DecodingOptions::new());
    loop {
        let samples: Vec<f32> = vec![0.0; 16_000]; // 1 s of 16 kHz mono from your source
        let update = streamer.push_samples(&samples)?;
        if update.is_transcribed() {
            for segment in streamer.state().confirmed_segments_slice() {
                println!("confirmed: {}", segment.text());
            }
            break;
        }
    }
    Ok(())
}
```

Raw CoreML, without the pipeline:

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

Not yet on crates.io (release pending). Until then, use a git dependency:

```toml
[dependencies]
whisperkit = { git = "https://github.com/findit-studio/coremlit" }
# or just the CoreML layer:
coremlit = { git = "https://github.com/findit-studio/coremlit" }
```

## Getting models

Models are plain local folders — the library performs no downloads. Fetch the
WhisperKit CoreML bundles and the matching tokenizer with the Hugging Face CLI:

```sh
hf download argmaxinc/whisperkit-coreml --include "openai_whisper-tiny/*" \
  --local-dir Models/whisperkit-coreml
hf download openai/whisper-tiny tokenizer.json tokenizer_config.json config.json \
  --local-dir Models/tokenizers/whisper-tiny
```

Examples, benches, and the `--ignored` test suites resolve the models root via
the `WHISPERKIT_TEST_MODELS` environment variable, defaulting to `<repo>/Models`
(gitignored). Any `openai_whisper-*` folder from
[argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml)
works; swap the tokenizer repo to match the model size.

## Examples & benches

| Command | What it shows |
|---|---|
| `cargo run -p whisperkit --example transcribe_wav -- [wav]` | File transcription with timestamps + timings (defaults to the committed JFK clip) |
| `cargo run -p whisperkit --example mic_stream` | Live mic streaming: cpal capture → rubato resample → `push_samples` |
| `cargo bench -p whisperkit --bench stages` | Hermetic criterion benches: logits filters, DTW, VAD chunking, compression ratio |
| `cargo bench -p whisperkit --bench rtf` | End-to-end tokens/sec + real-time factor on the tiny model (skips without models) |

## Feature flags (`whisperkit`)

| Feature | Default | Enables |
|---|---|---|
| `serde` | off | `Serialize`/`Deserialize` on options and results (partial configs fill with defaults) + the JSON result writer |
| `tracing` | off | Internal log events additionally emitted as `tracing` events |

`coremlit` has no feature flags.

## MSRV & platform

Rust **1.95**, edition 2024. macOS only (Apple Silicon primary; `x86_64-apple-darwin` is untested). Not sandboxed-Linux-buildable by design — this is a CoreML binding.

## Status

`0.1.0` (unreleased). The pipeline is parity-pinned against Swift WhisperKit
(`whisperkit-cli`) token goldens on `openai_whisper-tiny` for English and
Spanish clips, plus a 60 s VAD-chunked long-form fixture. See
[CHANGELOG.md](CHANGELOG.md).

## Acknowledgments

`whisperkit` is a Rust port of [Argmax's WhisperKit](https://github.com/argmaxinc/WhisperKit)
(MIT). Model weights come from [argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml);
Whisper itself is [OpenAI's](https://github.com/openai/whisper).

#### License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any additional
terms or conditions.
