//! Requires a commercial licence for the ArcFace `w600k_r50` weights and their
//! WebFace600K corpus.
//!
//! InsightFace publishes both for non-commercial research only and offers no
//! commercial grant over them, so what this module names may be used to
//! develop, evaluate and test — never to ship. It is a MANIFEST and nothing
//! else: four constants describing the artifact `MODELS_LOCK`'s `arcface` kit
//! stages, with no loader, no path resolution and no weight bytes. The
//! `commercial-face-arcface` feature that gates it is never in `default`, no
//! other feature pulls it in, and this repository still redistributes nothing
//! — the bundle lives in a private Hugging Face repository CI fetches with a
//! read token (`NOTICE`, "CI DOWNLOADS; IT DOES NOT REDISTRIBUTE").
//!
//! Everything else in [`super`] is unencumbered and ungated: the 5-point
//! alignment is `f64` arithmetic over a synthetic golden, and
//! [`FaceEmbedder`](super::FaceEmbedder) loads whatever artifact its caller
//! points it at — including one whose licence permits a product. What the gate
//! protects is this artifact, so this is the only thing behind it.
//!
//! **The gate therefore governs what coremlit WIRES** — this manifest, its
//! `MODELS_LOCK` staging and its gated tests — and not what anyone may load:
//! bytes a caller hands
//! [`FaceEmbedder::load`](super::FaceEmbedder::load) through the plain `face`
//! feature carry the caller's own licence, which is the residual issue #138 §8
//! states and `tests/model_licences.rs`'s module doc records.
//!
//! ```no_run
//! use coremlit::embeddings::face::{FaceEmbedder, FaceEmbedderOptions, arcface};
//!
//! let embedder = FaceEmbedder::load(
//!   arcface::STAGED_PATH,
//!   arcface::MODEL,
//!   FaceEmbedderOptions::new().with_compute(arcface::RECOMMENDED_COMPUTE),
//! )?;
//! assert_eq!(embedder.dim(), 512);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Where these values come from
//!
//! `coremlit/conversion/face/README.md`, which converted the artifact and
//! measured every number on this page. Nothing here is a preference:
//! `probe_onnx_contract.py` reads the feature names, the width and the absence
//! of an L2 tail off the ONNX before anything is converted, and
//! `verify_arcface.py` decides the channel order by measurement rather than by
//! transcription — feeding BGR drops the worst same-person pair to 0.2547,
//! through InsightFace's own 0.28 "same person" line.

use crate::{ComputeUnits, embeddings::face::FaceModel};

/// The compiled bundle's directory name.
///
/// The library never spells this anywhere else —
/// [`FaceEmbedder::load`](super::FaceEmbedder::load) takes whatever path its
/// caller staged — so the artifact's own name lives beside the rest of its
/// manifest.
pub const BUNDLE_NAME: &str = "w600k_r50.mlmodelc";

/// The workspace-relative path `MODELS_LOCK`'s `arcface` kit stages the bundle
/// to.
///
/// A convenience for a caller running from a checkout, not a resolution rule:
/// the door takes an absolute or relative path from its caller and this crate
/// resolves nothing on its behalf. CI's `arcface` shard downloads here, and
/// `FACEKIT_TEST_MODELS` overrides the directory for the gated tests.
pub const STAGED_PATH: &str = "Models/facekit/w600k_r50.mlmodelc";

/// The artifact's contract: `data [1, 3, 112, 112] f32 → embedding [1, 512]
/// f32`, RAW, preprocessed as [`Preprocessing::ARCFACE`].
///
/// Neither feature name is the ONNX's own. That graph was traced out of
/// PyTorch 1.9 and its features are called `input.1` and `683` — a tracer's
/// counters rather than a contract — so the conversion renames them to
/// InsightFace's own MXNet-era `data` and to the `embedding` every other
/// `coremlit` embedder emits.
///
/// **The graph does not normalise, and that was established rather than
/// assumed.** Its entire op set is `Conv` 53, `BatchNormalization` 26, `PRelu`
/// 25, `Add` 24, `Flatten` 1, `Gemm` 1 — no `LpNormalization`, no `ReduceL2`,
/// no decomposition of one — and the measured norms over the 18 fixture faces
/// run 17.01 – 24.91. So the L2 is the DOOR's, exactly as
/// [`FaceEmbedding`](super::FaceEmbedding) documents, and there is no double
/// normalisation to worry about.
///
/// [`Preprocessing::ARCFACE`]: super::Preprocessing::ARCFACE
pub const MODEL: FaceModel = FaceModel::new("data", "embedding", 512);

/// The compute placement this artifact was measured fastest and steadiest on.
///
/// **Measured, not preferred** — `conversion/face/scripts/sweep_placement.py`,
/// five rounds of 100 warm predicts per arm, each round in a fresh process so
/// the cold load is genuinely cold:
///
/// | arm | warm predict ms (median, range) | faces/s | min cos vs fp32 |
/// |---|---|---|---|
/// | `All` | 4.46 (3.57 – 6.41) | 224 | 0.999781 |
/// | `CpuAndGpu` | 8.66 (4.86 – 10.59) | 115 | 0.999999 |
/// | `CpuOnly` | 10.42 (9.92 – 23.09) | 96 | 0.999835 |
/// | **`CpuAndNeuralEngine`** | **3.48 (2.99 – 3.76)** | **287** | 0.999780 |
///
/// It is the fastest arm, it has the tightest spread of the four, its cold
/// load is second-cheapest at 160 ms, and its parity sits 45× inside the
/// `1 − cos ≤ 0.01` gate. No arm emitted a `BNNS Graph Shape Deduction` line.
///
/// **This is not [`DEFAULT_FACE_COMPUTE`](super::DEFAULT_FACE_COMPUTE), and
/// the difference is deliberate.** That default belongs to a door which loads
/// whatever artifact a caller supplies, so it stays CoreML's own planner
/// choice; this constant is a fact about ONE artifact. The two differ in
/// practice as well as in principle — leaving the placement at `All` costs
/// 1.3× at the median and swings across 3.57 – 6.41 ms where the pinned arm
/// holds 2.99 – 3.76.
///
/// Every number above is one machine's (the toolchain table in the recipe's
/// README names it). CoreML contracts none of it to reproduce across chips or
/// macOS builds, which is why `tests/face/placement.rs` asserts the PORTABLE
/// half — parity and finiteness on every arm — and prints the timings rather
/// than pinning them.
pub const RECOMMENDED_COMPUTE: ComputeUnits = ComputeUnits::CpuAndNeuralEngine;

#[cfg(test)]
mod tests;
