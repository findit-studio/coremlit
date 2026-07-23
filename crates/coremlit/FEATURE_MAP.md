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
| speakerkit | `dia` | `speaker` | diaric's backend-free runtime clustering core (formerly the `dia` offline bridge) |
| speakerkit | `dia-oracle` | `speaker-oracle` | dia's ort DER oracle (DEV/TEST) |
| vadkit | (crate) | `vad` | silero's logic-only detector rides this |
| vadkit | dev-dep `silero/bundled` | `vad-bundled` | silero ONNX cross-backend oracle (DEV/TEST) |
| clapkit | (crate) | `clap` | CLAP-HTSAT dual-tower audio+text encoders (module `embeddings::clap`) ride this; Rust mel front-end + shared `tokenizers`, no ort; the long-audio window geometry + aggregation ride the rev-pinned `windit` git dep |
| clapkit | `parity-oracle` | `clap-oracle` | textclap model-level parity oracle (DEV/TEST) |
| clapkit | `serde` | `serde` | unified cross-cutting |

## Flat feature set

`default = []` (the bare CoreML runtime core). Additive features:

`whisper`, `nl-recognizer`, `align`, `align-oracle`, `speaker`,
`speaker-oracle`, `vad`, `vad-bundled`, `clap`, `clap-oracle`, `granite`,
`serde`, `tracing`.

`granite` is not a former per-crate kit but a NEW module (`embeddings::granite`,
the embedkit phase): general text sentence-embeddings on CoreML, first model
`granite-embedding-97m-multilingual-r2`. Its parity oracle is COMMITTED
transformers-fp32 goldens, not a live crate, so it has NO `granite-oracle`
sibling and pulls no `ort` — hence it appears in the rename table below only as a
new-module note, not an old-crate row. Its long-input `embed_long` path pulls the
rev-pinned `windit` git dep (with `windit/text` for content-aware chunking); the
single-text `embed` path does not depend on it.

Compositions (pinned by the golden test): `nl-recognizer` → `whisper`;
`align-oracle` → `align`; `speaker-oracle` → `speaker`; `vad-bundled` → `vad`;
`clap-oracle` → `clap`. (`granite` composes with nothing — a single leaf
feature.)

## Curated CI feature-combination list

The former per-crate `cargo hack --each-feature` powerset is replaced by this
curated combo list — each kit feature alone, each oracle combo, all-on, and
none. It is pinned here and driven by CI (`.github/workflows/ci.yml`):

| Combo | Purpose |
|---|---|
| (none, `default = []`) | the bare core builds/tests dependency-lean |
| `whisper` | the STT pipeline alone |
| `align` | forced alignment alone (asry emissions, no ort) |
| `speaker` | diarization backends + diaric clustering core (no ort) |
| `vad` | Silero model layer alone (silero logic-only, no ort) |
| `whisper,vad` | the `silero_vad` composition (former `vadkit` feature) |
| `align-oracle` | + asry ONNX aligner (ort + whisper.cpp) |
| `speaker-oracle` | + dia ort DER oracle |
| `vad-bundled` | + silero ONNX cross-backend oracle |
| `clap` | CLAP audio+text encoders alone (Rust mel + tokenizers, no ort) |
| `clap-oracle` | + textclap model-level parity oracle (ort) |
| `granite` | granite text embeddings alone (bundled tokenizer + committed transformers-fp32 goldens, no ort; `embed_long` rides the rev-pinned `windit` engine + `windit/text`) |
| `whisper,align,speaker,vad,clap,granite,serde,tracing,nl-recognizer` | all non-oracle features on |
| `whisper,align-oracle,speaker-oracle,vad-bundled,clap-oracle,granite,serde,tracing,nl-recognizer` | all-on (every feature, oracles included) |

`serde` and `tracing` are cross-cutting and covered by the all-on runs. The
list embodies the combinatorial-honesty rule: it is explicit and reviewable,
not an implicit powerset.
