# Changelog

Notable changes to the crates in this workspace. Versions follow SemVer per crate.

## coremlit 0.1.0 (unreleased)

Initial release: a safe, synchronous CoreML runtime layer.

- `Model`: load / compile / prewarm, synchronous `predict`, eager `ModelDescription` I/O snapshot.
- Stateful prediction via `MLState` (macOS 15+, probed at runtime with `supports_state`).
- `MultiArray`: typed views for `f16`/`f32`/`f64`/`i32`, IOSurface-backed `f16` construction for the Neural Engine, stride-aware `read_at`/`copy_into` for row-padded outputs, index math and scatter fills.
- `Features`: insertion-ordered named I/O bridging `MLFeatureProvider` in both directions (output extraction de-aliases buffers shared with inputs or other outputs).
- Threading: `Model`, `MultiArray`, and `State` are `Send` but deliberately **not** `Sync` — move them between threads or hold one per worker; concurrent shared access is outside the contract.
- `ComputeUnits`/`DataType` vocabularies; structured `thiserror` errors capturing `NSError` domain/code/message.

## whisperkit 0.1.0 (unreleased)

Initial release: a Rust port of [WhisperKit](https://github.com/argmaxinc/WhisperKit) (Swift) on CoreML, sans-I/O (16 kHz mono `&[f32]` in).

- Full pipeline: mel → encoder → autoregressive decoder with prefill prompts, KV caching, logits filters, and the temperature-fallback ladder; token parity with Swift `whisperkit-cli` on `openai_whisper-tiny` (en + es goldens).
- Long-form: energy-VAD chunking, scoped-thread worker pool, seek re-anchoring, result merging (60 s fixture-proven).
- Word timestamps: DTW over decoder alignment weights, duration constraints, punctuation merging (`DecodingOptions::word_timestamps`).
- Streaming: push-based `AudioStreamTranscriber` (VAD-gated, confirmed/unconfirmed segment promotion) and LocalAgreement-2 word confirmation (`LocalAgreementTranscriber`).
- Language detection; SRT/VTT/JSON result writers; `serde` and `tracing` optional features.
- Examples (`transcribe_wav`, `mic_stream`) and benches (criterion stage benches + an end-to-end RTF harness).
