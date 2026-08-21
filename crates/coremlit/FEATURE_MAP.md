# coremlit feature map

The mono-crate restructure collapsed five crates into one crate with
feature-gated modules. This file is the authoritative **rename table** (old
per-crate feature → new flat feature) and the **curated CI feature-combination
list**. It is pinned by the golden test `tests/feature_map.rs`, which parses
`Cargo.toml` and fails if the declared feature set drifts from this table, so a
rename or a dropped feature cannot land silently.

**Two packages.** `coremlit` is the publishable crate. The three third-party
parity oracles — `speaker-oracle` (dia/ort DER), `clap-oracle` (textclap),
`vad-bundled` (the `silero` crate's ONNX stack) — are **no longer coremlit
features**: `dia` and `textclap` are unpublished rev-pinned git sources that
`cargo publish` rejects even behind an optional feature, so those features, their
dependencies and their nine test binaries live in the never-published
`crates/coremlit-parity` package. Their NAMES are unchanged; only the package
you pass to `-p` changed. `align-oracle` did NOT move — it only turns on a
feature of `asry`, a dependency `coremlit` keeps either way, so relocating it
would move code without removing a git dependency. The golden test pins BOTH
manifests and BOTH CI matrices.

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
| speakerkit | `dia-oracle` | `speaker-oracle` | dia's ort DER oracle (DEV/TEST) — now a **`coremlit-parity`** feature |
| vadkit | (crate) | `vad` | the `zuoer` detector core rides this |
| vadkit | dev-dep `silero/bundled` | `vad-bundled` | the `silero` crate's ONNX cross-backend oracle (DEV/TEST) — now a **`coremlit-parity`** feature |
| clapkit | (crate) | `clap` | CLAP-HTSAT dual-tower audio+text encoders (module `embeddings::clap`) ride this; Rust mel front-end + shared `tokenizers`, no ort; the long-audio window geometry + aggregation ride the rev-pinned `windit` git dep |
| clapkit | `parity-oracle` | `clap-oracle` | textclap model-level parity oracle (DEV/TEST) — now a **`coremlit-parity`** feature |
| clapkit | `serde` | `serde` | unified cross-cutting |

## Flat feature set

`default = []` (the bare CoreML runtime core). Additive features:

`whisper`, `nl-recognizer`, `align`, `align-oracle`, `speaker`, `vad`, `clap`,
`granite`, `siglip`, `ced`, `serde`, `tracing`.

`coremlit-parity`'s own `default = []` plus three additive oracle features:
`speaker-oracle` (⇒ `coremlit/speaker`), `clap-oracle` (⇒ `coremlit/clap`),
`vad-bundled` (⇒ `coremlit/vad`). Each rides its own feature so one oracle's
`ort` build is not forced on the other two.

`granite` is not a former per-crate kit but a NEW module (`embeddings::granite`,
the embedkit phase): general text sentence-embeddings on CoreML, first model
`granite-embedding-97m-multilingual-r2`. Its parity oracle is COMMITTED
transformers-fp32 goldens, not a live crate, so it has NO `granite-oracle`
sibling and pulls no `ort` — hence it appears in the rename table below only as a
new-module note, not an old-crate row. Its long-input `embed_long` path pulls the
rev-pinned `windit` git dep (with `windit/text` for content-aware chunking); the
single-text `embed` path does not depend on it.

`siglip` is likewise a NEW module (`embeddings::siglip`): SigLIP 2
(`siglip2-base-patch16-naflex`) dual-tower image+text embeddings on CoreML (a
shared 768-dim joint space). Same posture as `granite` — COMMITTED
transformers-fp32 goldens, so NO `siglip-oracle` sibling and no `ort` — and it
composes with nothing (a single leaf feature). NaFlex resizes natively to a fixed
patch budget, so it is a `windit` non-consumer (no windowing).

`ced` is likewise a NEW module (`audio::ced`): CED (tiny/mini/small/base)
AudioSet sound-event tagging on CoreML — 16 kHz mono waveform in, ranked
predictions over the 527 rated AudioSet classes out (`soundevents-dataset`,
the ort-free data crate; the ort-based `soundevents` crate is never a
dependency). A Rust log-mel
front-end (`rustfft`) feeds one fp16 mel→logits graph; long clips ride the
rev-pinned `windit` engine (geometry only). Same posture as `granite` —
COMMITTED fp32 goldens, so NO `ced-oracle` sibling and no `ort` — and it
composes with nothing (a single leaf feature).

Compositions (pinned by the golden test): `nl-recognizer` → `whisper`;
`align-oracle` → `align`. Across the package boundary, `coremlit-parity`'s
`speaker-oracle` → `coremlit/speaker`, `clap-oracle` → `coremlit/clap`,
`vad-bundled` → `coremlit/vad`. (`granite`, `siglip`, and `ced` each compose
with nothing — a single leaf feature.)

## Curated CI feature-combination list

The former per-crate `cargo hack --each-feature` powerset is replaced by this
curated combo list — each kit feature alone, all-on, and none. It is pinned here
and driven by the `features` job of CI (`.github/workflows/ci.yml`), which runs
`cargo test -p coremlit --features <combo>`:

| Combo | Purpose |
|---|---|
| (none, `default = []`) | the bare core builds/tests dependency-lean |
| `whisper` | the STT pipeline alone |
| `align` | forced alignment alone (asry emissions, no ort) |
| `speaker` | diarization backends + diaric clustering core (no ort) |
| `vad` | Silero model layer alone (`zuoer` detector core, no ort) |
| `whisper,vad` | the `silero_vad` composition (former `vadkit` feature) |
| `align-oracle` | + asry ONNX aligner (ort + whisper.cpp) |
| `clap` | CLAP audio+text encoders alone (Rust mel + tokenizers, no ort) |
| `granite` | granite text embeddings alone (bundled tokenizer + committed transformers-fp32 goldens, no ort; `embed_long` rides the rev-pinned `windit` engine + `windit/text`) |
| `siglip` | SigLIP 2 image+text embeddings alone (bundled tokenizer + committed transformers-fp32 goldens, no ort) |
| `ced` | CED (tiny/mini/small/base) sound-event tagging alone (Rust mel + `soundevents-dataset` + `windit`, no ort) |
| `whisper,align,speaker,vad,clap,granite,siglip,ced,serde,tracing,nl-recognizer` | all non-oracle features on |
| `whisper,align-oracle,speaker,vad,clap,granite,siglip,ced,serde,tracing,nl-recognizer` | all-on (every coremlit feature, `align-oracle` included) |

`serde` and `tracing` are cross-cutting and covered by the all-on runs. The
list embodies the combinatorial-honesty rule: it is explicit and reviewable,
not an implicit powerset.

## Curated CI parity-oracle list

The three third-party oracles get their own CI job (`parity`), which runs
`cargo test -p coremlit-parity --features <combo>`. Pinned by the same golden
test, per job, so a dropped row cannot silently stop building an oracle:

| Combo | Purpose |
|---|---|
| `speaker-oracle` | dia's ort DER reference oracle |
| `clap-oracle` | textclap model-level parity oracle (ort) |
| `vad-bundled` | the `silero` crate's ONNX cross-backend oracle |
| `speaker-oracle,clap-oracle,vad-bundled` | all-on |
