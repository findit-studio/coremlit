# clapkit

Native CoreML **[CLAP](https://github.com/LAION-AI/CLAP)** (`laion/clap-htsat-unfused`)
on macOS: both encoders — HTSAT audio and RoBERTa text — running through
[`coremlit`](../coremlit), projecting into a shared 512-dim joint embedding
space, **plus the long-audio pipeline textclap lacks**: overlapped chunking,
customizable window-embedding aggregation, and zero-shot scoring.

Unlike the sibling backends, clapkit is not a thin port: the CLAP *logic* is
being **improved**, not merely reused (the asry→alignkit relationship, not
silero→vadkit). Its model-level oracle is
[`textclap`](https://github.com/Findit-AI/textclap), against which the encoders
are parity-pinned.

**Sans-I/O**, like the rest of the workspace — but at **48 kHz** mono `&[f32]`, a
deliberate, documented deviation from the workspace's 16 kHz convention (48 kHz
is CLAP's native rate; resampling to it is the caller's job). No file I/O, no
device capture, no async. macOS only (Apple Silicon; built on `coremlit`'s safe
CoreML layer).

## Quick start

Embed one 10 s window and score it against text labels (the canonical CLAP
zero-shot template, `"This is a sound of {label}"`):

```rust,no_run
use clapkit::{score, AudioEncoder, ScoreMode, TextAnchor, TextEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 48 kHz mono — this crate performs no I/O or resampling.
    let samples: Vec<f32> = vec![0.0; 480_000]; // one fixed 10 s window

    let audio = AudioEncoder::from_file("Models/clapkit/clap_audio.mlmodelc")?;
    let text = TextEncoder::from_file("Models/clapkit/clap_text.mlmodelc")?;

    let clip = audio.embed_window(&samples)?;

    let labels = [
        "This is a sound of a person speaking",
        "This is a sound of music",
        "This is a sound of a dog barking",
    ];
    let embeddings: Vec<_> = labels.iter().map(|l| text.embed(l)).collect::<Result<_, _>>()?;
    let anchors: Vec<_> = labels
        .iter()
        .zip(&embeddings)
        .map(|(l, e)| TextAnchor::new(l, e))
        .collect();

    for r in score(&clip, &anchors, ScoreMode::LogitScaled) {
        println!("{:>8.3}  {}", r.score(), r.label());
    }
    Ok(())
}
```

## The long-audio pipeline

A clip longer than one 10 s window is handled in three **composable, sans-model**
steps — nothing is hidden inside a monolithic pipeline object:

```rust,no_run
use clapkit::aggregate::AggregatePolicy;
use clapkit::{AudioEncoder, MeanRenormalized, WindowPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = AudioEncoder::from_file("Models/clapkit/clap_audio.mlmodelc")?;
    let long: Vec<f32> = vec![0.0; 3 * 60 * 48_000]; // 3 min, 48 kHz mono

    // 1. Plan overlapped windows (window fixed at 480 000 samples; hop + tail
    //    policy configurable, serde-validated).
    let plan = WindowPlan::new().with_hop_samples(240_000); // 50 % overlap

    // 2. Embed each window. The per-window embeddings are RETURNED (each carries
    //    its span + tail-padding-aware coverage), never hidden.
    let windows = audio.embed_windows(&long, &plan)?;

    // 3. Aggregate with a built-in policy — or your own (below).
    let clip = MeanRenormalized.aggregate(&windows)?;
    println!("{} windows -> one {}-d clip embedding", windows.len(), clip.dim());
    Ok(())
}
```

**Aggregation is a user-customizable seam** (spec amendment). Three built-ins
ship — `MeanRenormalized` (default), `EmaRenormalized { alpha }` (temporal
smoothing), `CoverageWeightedMean` (down-weights padded tails) — and a
serde-able `AggregatePolicyKind` names them for config surfaces. The set is
**open**: implement the object-safe `AggregatePolicy` trait for anything the
built-ins don't cover.

```rust
use clapkit::aggregate::AggregatePolicy;
use clapkit::window::WindowEmbedding;
use clapkit::{Embedding, Error};

// A custom policy: trust only the window with the most real coverage.
struct MostCovered;

impl AggregatePolicy for MostCovered {
    fn aggregate(&self, windows: &[WindowEmbedding]) -> Result<Embedding, Error> {
        windows
            .iter()
            .max_by(|a, b| a.coverage().total_cmp(&b.coverage()))
            .map(|w| w.embedding().clone())
            .ok_or(Error::EmptyWindows)
    }
}
```

Per-window embeddings **and** per-window zero-shot scores (`score_windows`) are
always exposed, so score-level smoothing or voting stays caller-side without a
second trait seam (the deliberate cut recorded in the spec).

## Model-level parity vs textclap (characterized, not bit-exact)

clapkit's fp16 CoreML encoders are pinned, per window, against `textclap`
running the **Xenova ONNX** graphs it ships (`tests/parity_textclap.rs`, feature
`parity-oracle`). Both crates receive the identical `&[f32]` / `&str`; the mel
front-end and tokenizer are the same ported/pinned artifacts (identity-gated in
`tests/`), so the residual gap is the encoder graph — precision + lowering.

**Honesty clause.** textclap ships the **quantized** (int8-class) Xenova graphs,
while clapkit converts fp16 from the fp32 source, so the primary gates pin the
cosine two-sided at **measured** values, not 1.0. A **same-precision control**
runs the identical comparison against Xenova's **unquantized fp32** graphs; its
near-perfect agreement attributes essentially the entire gap to textclap's
quantization, not to clapkit's fp16 (measured 2026-07-18):

| tower | vs quantized (worst cosine) | vs fp32 control (worst cosine) | quantization contribution |
|---|---|---|---|
| audio | **0.99804741** | 0.99999756 | ≈ 0.00195 |
| text | **0.96725219** (CJK) | 0.99994940 | ≈ 0.03270 |

That the fp32 control sits at ~1.0 for both towers is the load-bearing result:
clapkit's fp16 conversion is essentially bit-faithful to the fp32 ONNX source;
the parity gap is quantization on textclap's side. All four bands are pinned
two-sided — a shift in *either* direction is a finding.

## Licensing (read before you ship)

Two components, different provenance. This is a factual summary; see `NOTICE`
for the full attribution a redistributor must reproduce.

- **CLAP weights** — `laion/clap-htsat-unfused`, converted to CoreML at
  [`FinDIT-Studio/clapkit-coreml`](https://huggingface.co/FinDIT-Studio/clapkit-coreml)
  (`@97d631f3…`). Treated as **CC-BY-4.0** (attribution required — carried in
  `NOTICE`), consistent with textclap's `models/MODELS.md`. The current upstream
  HF card declares apache-2.0; both require attribution, which this repo
  provides. clapkit ships no model files — you fetch them (below); only a binary
  that distributes the fetched graphs redistributes the weights.
- **Tokenizer** — `tokenizer.json` from
  [`Xenova/clap-htsat-unfused`](https://huggingface.co/Xenova/clap-htsat-unfused)
  `@c28f2883…` (SHA `dc239041…`), the CLAP/RoBERTa (roberta-base, MIT lineage)
  tokenizer. It is **bundled** in the crate via `include_bytes!`
  (`clapkit::BUNDLED_TOKENIZER`), so every binary linking clapkit redistributes
  it.

The clapkit Rust source itself is dual **MIT OR Apache-2.0** like the rest of the
workspace.

## Compute units (measured, never marketed)

Both encoders default to `ComputeUnits::All` — CoreML schedules across the
available hardware. Placement is **characterized, not asserted**
(`tests/placement.rs`):

- **text** (RoBERTa): compiles for and runs on the ANE/GPU/CPU; fp16-clean on
  all, cross-placement cosine ≥ 0.9999.
- **audio** (HTSAT Swin): as converted, the graph **fails ANE compilation**
  (`ANECCompile()` — visible at runtime) and falls back to GPU/CPU, still
  fp16-clean (cross-placement cosine ≥ 0.9999). **Do not assume ANE placement
  for the audio tower.**

For bit-reproducibility across machines/runs, pin `ComputeUnits::CpuOnly`
explicitly and record the placement alongside anything you persist — outputs are
not placement-invariant.

## Test models

The CoreML store is gitignored — nothing under `Models/` is committed.
Model-gated tests are `#[ignore]`d by default; run them explicitly once the
models are present.

| Artifact | Env override | Default path | Fetch |
|---|---|---|---|
| clapkit CoreML | `CLAPKIT_TEST_MODELS` | `<workspace>/Models/clapkit` | `hf download FinDIT-Studio/clapkit-coreml --local-dir Models/clapkit` |
| textclap ONNX (parity oracle) | `CLAPKIT_TEXTCLAP_ONNX` | `<workspace>/Models/textclap-onnx` | `hf download Xenova/clap-htsat-unfused --include "onnx/*model*.onnx" --revision c28f2883575e590e04d3146ff0713c2448d691ba --local-dir Models/textclap-onnx` |

```sh
# Model-gated hermetic gates (I/O pins, placement, e2e pipeline):
CLAPKIT_TEST_MODELS=Models/clapkit cargo test -p clapkit -- --ignored

# The parity gate additionally needs the textclap oracle + its ONNX (the
# quantized pair is required; the fp32 pair enables the same-precision control):
CLAPKIT_TEST_MODELS=Models/clapkit CLAPKIT_TEXTCLAP_ONNX=Models/textclap-onnx \
  cargo test -p clapkit --features parity-oracle --test parity_textclap -- --ignored
```

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `serde` | off | `Serialize`/`Deserialize` on options, `WindowPlan`, `TailPolicy`, `AggregatePolicyKind`, `ScoreMode` (validated deserialization where it matters) |
| `parity-oracle` | off | **dev/test only** — links `textclap` (+ its `ort` runtime) for the live tokenizer-identity and model-level parity gates |

## Status

`0.1.0` (unreleased), `publish = false` — the `parity-oracle` gate depends on the
unpublished `textclap` git source (crates.io forbids git sources); the DEFAULT
build has no git dependencies. Both encoders are parity-pinned against textclap
(table above); the long-audio pipeline (geometry, aggregation math, serde,
zero-shot ranking) is hermetically pinned and mutation-verified; a multi-minute
end-to-end run pins the window count, aggregate, and top zero-shot label.

## See also

- Design spec: `docs/superpowers/specs/2026-07-18-clapkit-design.md` (and its
  `AggregatePolicy` amendment) — the source of truth for everything above.
  `docs/` is gitignored (local planning artifacts), so this path exists only
  where the feature was planned/built.
- [`textclap`](https://github.com/Findit-AI/textclap) — the model-level oracle.
- [`coremlit`](../coremlit) — the safe CoreML runtime layer this crate is built
  on.
- Workspace root [`README.md`](../../README.md) — the crate roster and the
  repo-wide dual MIT/Apache-2.0 license for this crate's *own* Rust source
  (distinct from the model-weight/tokenizer licenses above).
