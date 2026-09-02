//! Native CoreML **face embedding**: the identity half of a video face
//! pipeline, as `audio::speaker` is for voices.
//!
//! Apple Vision supplies the face *spine* — rectangles, capture quality,
//! roll/yaw/pitch, landmarks — but not a face *embedding*:
//! `VNGenerateFaceprintRequest` is private, and the public
//! `VNGenerateImageFeaturePrintRequest` is a whole-image feature print, not an
//! identity-tuned space. So the identity half has to be a model of our own.
//!
//! ```no_run
//! use coremlit::embeddings::face::{FaceAlign, FaceCrop, FaceEmbedder, FaceModel, Point};
//!
//! # let (rgb, width, height) = (vec![0u8; 64 * 48 * 3], 64usize, 48usize);
//! let crop = FaceCrop::new(&rgb, width, height)?;
//! let landmarks = [
//!   Point::new(18.5, 16.0), // left eye  (the VIEWER's left)
//!   Point::new(41.0, 13.5), // right eye
//!   Point::new(30.5, 25.0), // nose tip
//!   Point::new(21.0, 35.5), // left mouth corner
//!   Point::new(40.0, 33.0), // right mouth corner
//! ];
//! let aligned = FaceAlign::to_template(crop, &landmarks)?;
//!
//! // Preprocessing is the MANIFEST's, never a constant here.
//! let manifest = FaceModel::new("data", "embedding", 512);
//! let embedder = FaceEmbedder::from_file("Models/facekit/arcface.mlmodelc", manifest)?;
//! let embeddings = embedder.embed(&[aligned])?; // batch is the unit
//! assert_eq!(embeddings.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! macOS only (built on [`crate`]).
//!
//! # The shapes that are deliberate
//!
//! - **Alignment is OUTSIDE the embedder.** [`FaceAlign::to_template`] is a
//!   pure function producing an explicit [`AlignedFace`], so the 5-point
//!   similarity transform every downstream cosine passes through has a golden
//!   of its own instead of hiding inside a preprocessing step. See the
//!   [`align`] module.
//! - **Batch is the unit.** A keyframe with N faces is one
//!   [`FaceEmbedder::embed`] call, whatever the graph's own batch dimension is.
//! - **Preprocessing belongs to the manifest.** Channel order, scale, bias and
//!   layout are [`FaceModel`] fields, so a second artifact with different
//!   preprocessing is a different value, not a second code path. See the
//!   [`embed`] module for the census table that makes this non-negotiable.
//! - **The space is produced by `load`, and it knows which bytes it came
//!   from.** [`FaceEmbedder::load`] hashes the artifact directory it loads
//!   into an [`ArtifactDigest`], and every [`FaceEmbedding`] carries that
//!   alongside the feature names, width and preprocessing. Two vectors compare
//!   only if byte-identical weights produced them, read from the same output
//!   feature — [`FaceEmbedding::dot`] cannot return a score across different
//!   weights. See the [`artifact`] module.
//! - **The load contract is a value, and a type proves it was checked.**
//!   [`FaceEmbedder`] holds a crate-internal `Checked` model whose only
//!   constructor runs this door's contract, so an extra required input, a
//!   declared state buffer, a wrong element type and a flexible input shape are
//!   all refused when the model is loaded rather than discovered at predict
//!   time — and deleting the check does not compile. The [`embed`] module
//!   carries the contract, and the legacy `neuralNetwork` export it refuses
//!   deliberately.
//!
//! # No artifact is staged, and that is a finding rather than an omission
//!
//! Every other kit in this crate names an artifact `MODELS_LOCK` stages and
//! gates itself against it. This one does not, and the reason is recorded here
//! because it is a property of the licence policy rather than of this module.
//!
//! Issue #115's census established that **no face-embedding model clears both
//! the licence bar and the off-angle accuracy bar**: `buffalo_l`'s `w600k_r50`
//! is the accuracy reference (CFP-FP 99.33; 2.22 % frontal↔profile identity
//! splits at FAR 1e-3) but InsightFace's zoo states "ALL models are available
//! for non-commercial research purposes only", and the best commercially
//! granted artifact, AuraFace, splits identity on those same pairs **38.55 %**
//! of the time. The owner's response (issue #115, "Owner decisions — recorded")
//! was that CI and development MAY use research-only weights, on the standing
//! basis that **coremlit never redistributes them**, with every such artifact
//! behind a `commercial-`prefixed feature outside `default`.
//!
//! That decision has no road to a staged artifact today, for four independent
//! reasons:
//!
//! 1. **The convert-and-publish road is the redistribution the policy forbids.**
//!    Granite, SigLIP, CED, CLAP and the speakerkit overlay are all
//!    FinDIT-Studio re-conversions, and `NOTICE` describes those as weights this
//!    project redistributes. A research-only model cannot travel that road.
//! 2. **No upstream publishes a CoreML build of an accuracy-adequate ArcFace.**
//!    InsightFace ships ONNX only.
//! 3. **Both third-party CoreML ArcFace builds a Hugging Face search surfaces
//!    declare `ImageType` inputs**, and [`crate::Model`] binds only
//!    [`crate::MultiArray`] feature values — `crate::Features` is a
//!    `Vec<(String, MultiArray)>` end to end, and an image feature needs
//!    `MLFeatureValue(pixelBuffer:)`. Wiring either one is a core-runtime
//!    change, not a face-module change. Both were downloaded and inspected
//!    (`xcrun coremlcompiler metadata`) on 2026-09-02, and each has a second
//!    disqualifier of its own:
//!
//!    - `RuiSumida/ArcFace-R100-CoreML` states its provenance (a conversion of
//!      the ONNX Model Zoo's `arcfaceresnet100-8.onnx`, Apache-2.0 repo,
//!      trained on refined MS-Celeb-1M — so research-only at the corpus layer,
//!      which is the row shape the register wants). But its graph emulates
//!      PReLU with **100 `NonZero` + 50 `scatterNd` + 50 `gatherNd`** ops,
//!      which forces data-dependent shapes and the ANE off the graph entirely;
//!      and the zoo's own published CFP-FP for that checkpoint is **94.21**,
//!      below issue #115's proven-bad line of 97.3.
//!    - `RuiSumida/InsightFace-glintr100-CoreML` is a clean IResNet-100 (103
//!      convolutions, 51 batch norms, 50 NATIVE `ActivationPReLU`) and would be
//!      the better graph — but it ships no README, no licence, and no statement
//!      of provenance beyond the repository name, which is exactly the evidence
//!      #73's provenance gate exists to require.
//! 4. **Converting inside the CI job has no legal row shape in the licence
//!    register.** Its two key exemptions are `Unpinned` (the lock pins a moving
//!    revision) and `Unmanifested` (a glob table with no per-file manifest);
//!    neither covers bytes that are produced by the job and never published, so
//!    a converted-in-CI artifact could not be keyed at all.
//!
//! The register is right to make this hard: a `commercial-` feature that gates
//! no licence row is refused by
//! `every_commercial_feature_gates_a_research_only_artifact` precisely so a
//! gate cannot stand as false reassurance. So this module ships behind a plain
//! `face` feature and takes the artifact path from the caller, and the
//! `commercial-face` gate arrives with the artifact it protects — not before.
//!
//! What that costs, stated plainly: the alignment golden is real and hermetic,
//! but there is **no embedding parity test, no known-pairs test and no
//! throughput number**, because there is nothing to run them against.

pub mod align;
pub mod artifact;
pub mod embed;
pub mod error;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use artifact::ArtifactDigest;

pub use align::{
  ARCFACE_TEMPLATE, AlignedFace, FaceAlign, FaceCrop, LANDMARK_COUNT, MAX_CROP_AXIS, Point,
  SimilarityTransform, TEMPLATE_BYTES, TEMPLATE_SIZE,
};
pub use embed::{
  ChannelOrder, DEFAULT_FACE_COMPUTE, EmbeddingSpace, FaceEmbedder, FaceEmbedderOptions,
  FaceEmbedding, FaceModel, Preprocessing, TensorLayout,
};
