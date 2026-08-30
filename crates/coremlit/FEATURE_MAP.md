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
| clapkit | (crate) | `clap` | CLAP-HTSAT dual-tower audio+text encoders (module `embeddings::clap`) ride this; Rust mel front-end + shared `tokenizers`, no ort; the long-audio window geometry + aggregation ride the crates.io `windit` dep |
| clapkit | `parity-oracle` | `clap-oracle` | textclap model-level parity oracle (DEV/TEST) — now a **`coremlit-parity`** feature |
| clapkit | `serde` | `serde` | unified cross-cutting |

## Flat feature set

`default = []` (the bare CoreML runtime core). Additive features:

`whisper`, `nl-recognizer`, `align`, `align-oracle`, `speaker`, `vad`, `clap`,
`granite`, `siglip`, `ced`, `lid`, `serde`, `tracing`.

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
crates.io `windit` dep (with `windit/text` for content-aware chunking); the
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
crates.io `windit` engine (geometry only). Same posture as `granite` —
COMMITTED fp32 goldens, so NO `ced-oracle` sibling and no `ort` — and it
composes with nothing (a single leaf feature).

`lid` is likewise a NEW module (`audio::lid`): spoken-language identification
on CoreML — 16 kHz mono waveform in, ranked languages out over a 107-language
roster. A Rust log-mel front-end (`rustfft`, the module's only dependency)
feeds one fp16 mel→log-probabilities graph, so the feature adds exactly
`dep:rustfft` and nothing else. It is a **backend-neutral door**: no public name
spells the model behind it, and today's backend is
`aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML` (Apache-2.0), an export of
`speechbrain/lang-id-voxlingua107-ecapa`. Unlike `granite` and `siglip` the
label roster is COMMITTED in-crate (`include_bytes!`, 10.7 kB) rather than read
from the artifact, so nothing about the vocabulary is a model gate. Clips are
capped at 30.01 s by the graph's own frame range, so `lid` is a `windit`
non-consumer (no windowing), and it composes with nothing — a single leaf
feature.

Compositions (pinned by the golden test): `nl-recognizer` → `whisper`;
`align-oracle` → `align`. Across the package boundary, `coremlit-parity`'s
`speaker-oracle` → `coremlit/speaker`, `clap-oracle` → `coremlit/clap`,
`vad-bundled` → `coremlit/vad`. (`granite`, `siglip`, `ced`, and `lid` each
compose with nothing — a single leaf feature.)

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
| `granite` | granite text embeddings alone (artifact-sidecar tokenizer + committed transformers-fp32 goldens, no ort; `embed_long` rides the crates.io `windit` engine + `windit/text`) |
| `siglip` | SigLIP 2 image+text embeddings alone (artifact-sidecar tokenizer + committed transformers-fp32 goldens, no ort) |
| `ced` | CED (tiny/mini/small/base) sound-event tagging alone (Rust mel + `soundevents-dataset` + `windit`, no ort) |
| `lid` | spoken-language identification alone (Rust mel + committed 107-label roster, no ort) |
| `whisper,align,speaker,vad,clap,granite,siglip,ced,lid,serde,tracing,nl-recognizer` | all non-oracle features on |
| `whisper,align-oracle,speaker,vad,clap,granite,siglip,ced,lid,serde,tracing,nl-recognizer` | all-on (every coremlit feature, `align-oracle` included) |

`serde` and `tracing` are cross-cutting and covered by the all-on runs. The
list embodies the combinatorial-honesty rule: it is explicit and reviewable,
not an implicit powerset.

"Artifact-sidecar tokenizer" (`granite`, `siglip`) means the crate embeds no
`tokenizer.json` for those two — each is a multi-megabyte file the published
model artifact ships beside the `.mlmodelc`, which `TextEmbedder::load` reads
and hash-checks against a pinned SHA-256. Only `clap` and `align` still
`include_bytes!` their tokenizers. Consequence for this table: the `features`
job is hermetic, so the `granite`/`siglip` rows build and run everything EXCEPT
the tokenizer gates — those are `#[ignore]`d on a staged artifact and belong to
the `model-tests` job, which stages both via `MODELS_LOCK`. SigLIP's entry is
the 34 MB `tokenizer.json` ALONE, not the ~784 MB bundle: its two tokenizer
gates call no `Model::load`, so the towers would buy them nothing. The
tower-dependent siglip gates (`model_io`, `text_model_io`, `parity_embed`,
`placement`, `e2e`) still run only locally.

`ced` has no tokenizer, but the same split applies to its model gates: the
`features` job runs the hermetic `ced` suite, and the model-gated
`ced_model_io`/`ced_parity_logits`/`ced_placement`/`ced_e2e` targets belong to
`model-tests`, which stages the artifact via `MODELS_LOCK`. The entry is
`ced-tiny` ALONE — 10.64 MB of the repo's 234 MB, since the four sizes are
I/O-identical and `ced-base` at 163.62 MB is past GitHub's 100 MB file limit
that let vadkit be committed instead. Each target declares its gates once per
size, so the CI step filters on `tiny::`; the `mini`/`small`/`base` gates stay
local/dev gates against an owner-staged `CED_TEST_MODELS` tree.

`lid` splits the same way, but only half of it is wired today. The `features`
job runs the hermetic lid suite (the mel front end, the roster/asset agreement
checks, every typed-error path) under the `lid` row and both all-on rows. The
model-gated `lid_model_io` (5 gates) and `lid_e2e` (4 gates) targets have **no
`model-tests` shard yet** and stay local/dev gates against an owner-staged
`LID_TEST_MODELS` tree — the same posture `align` is in, and for a related
reason. It is not the download: it is that `Models/lid/` cannot enter any shard
until `tests/fp16_guards.rs`'s graph sweep can read the artifact. That sweep
walks `Models/` WHOLE, and this graph is a coremltools 9 export whose scalar
consts use MIL's terse `fp16 v = const()[val = fp16(…)]` spelling instead of the
`tensor<fp16, []>` form the reader knows, so all 36 of its guard sites come back
unresolved; and once they are readable its final `softmax -> log` carries
`epsilon = 0x1p-149`, the same vanishing-guard defect `alignkit` and
`speakerkit/Segmentation` are pinned for. Teaching the reader that dialect and
cutting a `KNOWN_DEFECTS` pin are prerequisites for the shard, and both are
model-numerics work rather than CI registration. There would be no `@lib` half
in any case — every lid model gate is a `tests/lid/*` target, so
`--features lid --lib -- --list --ignored` lists zero.

`speaker` splits the same way, with one wrinkle no other kit has: its artifact
set is staged from TWO repositories into one directory, and the second overlays
the first (MODELS_LOCK's last two tables, and the ORDER box in its header).
`model-tests` runs `speaker_model_io`, `speaker_parity_seg` and
`speaker_parity_embed` there. Four speaker targets deliberately stay out:
`speaker_parity_diarize_wiring`, whose fixtures live in the sibling
`diarization` repository (`DIA_PARITY_FIXTURES`) that no runner has, and the
three argmax targets (`speaker_argmax_model_io`,
`speaker_parity_argmax_accuracy`, `speaker_parity_argmax_swift`), because the
argmax artifact repo declares no license — so this repository does not fetch
those graphs in CI at all (NOTICE records the reasoning).

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
