# coremlit feature map

The mono-crate restructure collapsed five crates into one crate with
feature-gated modules. This file is the authoritative **rename table** (old
per-crate feature → new flat feature) and the **curated CI feature-combination
list**. It is pinned by the golden test `tests/feature_map.rs`, which parses
`Cargo.toml` and fails if the declared feature set drifts from this table, so a
rename or a dropped feature cannot land silently.

## Rename table (old crate feature → new flat feature)

| Old crate | Old feature | New flat feature | Notes |
|---|---|---|---|
| whisperkit | (crate) | `whisper` | the former unconditional deps (libc, mach2, rand, serde_json, tokenizers, unicode_categories) now ride this feature |
| whisperkit | `nl-recognizer` | `nl-recognizer` | kept; now implies `whisper` |
| whisperkit | `vadkit` | `whisper` + `vad` | the cross-crate feature becomes a composition |
| whisperkit / alignkit / speakerkit | `serde` | `serde` | unified cross-cutting |
| whisperkit / alignkit | `tracing` | `tracing` | unified cross-cutting |
| alignkit | (crate) | `align` | asry's `emissions` seam rides this |
| alignkit | `parity-oracle` | `align-oracle` | asry ONNX aligner oracle (DEV/TEST) |
| speakerkit | (crate) | `speaker` | the CoreML segmentation + embedding backends (module `audio::speaker`) ride this |
| speakerkit | `dia` | `speaker` | dia's backend-free offline bridge |
| speakerkit | `dia-oracle` | `speaker-oracle` | dia's ort DER oracle (DEV/TEST) |
| vadkit | (crate) | `vad` | silero's logic-only detector rides this |
| vadkit | dev-dep `silero/bundled` | `vad-bundled` | silero ONNX cross-backend oracle (DEV/TEST) |
| clapkit | (crate) | `clap` | CLAP-HTSAT dual-tower audio+text encoders (module `embeddings::clap`) ride this; Rust mel front-end + shared `tokenizers`, no ort |
| clapkit | `parity-oracle` | `clap-oracle` | textclap model-level parity oracle (DEV/TEST) |
| clapkit | `serde` | `serde` | unified cross-cutting |

## Flat feature set

`default = []` (the bare CoreML runtime core). Additive features:

`whisper`, `nl-recognizer`, `align`, `align-oracle`, `speaker`,
`speaker-oracle`, `vad`, `vad-bundled`, `clap`, `clap-oracle`, `serde`,
`tracing`.

Compositions (pinned by the golden test): `nl-recognizer` → `whisper`;
`align-oracle` → `align`; `speaker-oracle` → `speaker`; `vad-bundled` → `vad`;
`clap-oracle` → `clap`.

## Curated CI feature-combination list

The former per-crate `cargo hack --each-feature` powerset is replaced by this
curated combo list — each kit feature alone, each oracle combo, all-on, and
none. It is pinned here and driven by CI (`.github/workflows/ci.yml`):

| Combo | Purpose |
|---|---|
| (none, `default = []`) | the bare core builds/tests dependency-lean |
| `whisper` | the STT pipeline alone |
| `align` | forced alignment alone (asry emissions, no ort) |
| `speaker` | diarization backends + dia offline core (no ort) |
| `vad` | Silero model layer alone (silero logic-only, no ort) |
| `whisper,vad` | the `silero_vad` composition (former `vadkit` feature) |
| `align-oracle` | + asry ONNX aligner (ort + whisper.cpp) |
| `speaker-oracle` | + dia ort DER oracle |
| `vad-bundled` | + silero ONNX cross-backend oracle |
| `clap` | CLAP audio+text encoders alone (Rust mel + tokenizers, no ort) |
| `clap-oracle` | + textclap model-level parity oracle (ort) |
| `whisper,align,speaker,vad,serde,tracing,nl-recognizer` | all non-oracle features on |
| `whisper,align-oracle,speaker-oracle,vad-bundled,serde,tracing,nl-recognizer` | all-on (every feature, oracles included) |

`serde` and `tracing` are cross-cutting and covered by the all-on runs. The
list embodies the combinatorial-honesty rule: it is explicit and reviewable,
not an implicit powerset.
