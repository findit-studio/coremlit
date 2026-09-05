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
//! let embedder = FaceEmbedder::from_file("Models/facekit/w600k_r50.mlmodelc", manifest)?;
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
//!   only if byte-identical weights produced them **as read at `load`**, from
//!   the same output feature — [`FaceEmbedding::dot`] cannot return a score
//!   across different weights. The artifact must not be modified in place while
//!   an embedder holds it: replace a model by atomic `rename` and load a new
//!   embedder. See the [`artifact`] module.
//! - **The load contract is a value, and a type proves it was checked.**
//!   [`FaceEmbedder`] holds a crate-internal `Checked` model whose only
//!   constructor runs this door's contract, so an extra required input, a
//!   declared state buffer, a wrong element type and a flexible input shape are
//!   all refused when the model is loaded rather than discovered at predict
//!   time — and deleting the check does not compile. The [`embed`] module
//!   carries the contract, and the legacy `neuralNetwork` export it refuses
//!   deliberately.
//!
//! # ONE artifact is staged, and it is gated
//!
//! Issue #115's census established that **no face-embedding model clears both
//! the licence bar and the off-angle accuracy bar**: `buffalo_l`'s `w600k_r50`
//! is the accuracy reference (CFP-FP 99.33; 2.22 % frontal-to-profile identity
//! splits at FAR 1e-3) but InsightFace's zoo states "ALL models are available
//! for non-commercial research purposes only", and the best commercially
//! granted artifact, AuraFace, splits identity on those same pairs **38.55 %**
//! of the time. The owner's response (issue #115, "Owner decisions — recorded")
//! was that CI and development MAY use research-only weights, on the standing
//! basis that **coremlit never redistributes them**, with every such artifact
//! behind a `commercial-`prefixed feature outside `default`.
//!
//! That is what the `arcface` module is: a conversion of `w600k_r50`, published
//! to a PRIVATE Hugging Face repository CI fetches with a read token, behind
//! `commercial-face-arcface`. It is the register's first research-only row at
//! BOTH layers (the weights and WebFace600K), and the first artifact for which
//! `tests/model_licences.rs`'s directions 2 and 3 bind something rather than
//! standing as tripwires.
//!
//! **Only the ARTIFACT is gated, and the rest of this module is not.** Two
//! thirds of what is here is encumbered by nothing: [`align`] is `f64`
//! arithmetic and an integer resampler over a synthetic golden, with no
//! weights and no photograph in it, and [`FaceEmbedder`] loads a
//! **caller-supplied** path — including an artifact whose licence permits a
//! product. Pushing either behind a gate documented "requires a commercial
//! licence" would make the register say something false about code that
//! requires nothing, so `face` stays a plain feature and the manifest module
//! carries the gate alone.
//!
//! What the artifact buys, all of it measured in `conversion/face/README.md`
//! and asserted by the `commercial-face-arcface` gates in `tests/face/`:
//! cross-implementation parity against the committed fp32 ONNX reference on
//! every compute arm, the 18 same-person / 135 different-person pairs at
//! InsightFace's own 0.28 / 0.20 operating point, a four-arm placement table,
//! and 287 faces/s on the recommended arm.
//!
//! Three roads that do NOT exist are recorded here because each looks like the
//! obvious next step and none of them is:
//!
//! 1. **A public artifact repository.** Granite, SigLIP, CED, CLAP and the
//!    speakerkit overlay are FinDIT-Studio re-conversions `NOTICE` describes as
//!    weights this project redistributes. A research-only model cannot travel
//!    that road, which is why this one's repository is private.
//! 2. **A third-party CoreML build.** InsightFace ships ONNX only, and both
//!    CoreML ArcFace builds a Hugging Face search surfaces declare `ImageType`
//!    inputs that [`crate::Model`] cannot bind (`crate::Features` is a
//!    `Vec<(String, MultiArray)>` end to end). Each has a second disqualifier
//!    of its own — `RuiSumida/ArcFace-R100-CoreML` emulates PReLU with 100
//!    `NonZero` + 50 `scatterNd` + 50 `gatherNd`, forcing the ANE off the
//!    graph, and publishes CFP-FP 94.21 against issue #115's proven-bad line of
//!    97.3; `RuiSumida/InsightFace-glintr100-CoreML` is the better graph and
//!    ships no README, no licence and no statement of provenance.
//! 3. **Converting inside the CI job.** The licence register's two key
//!    exemptions are `Unpinned` (the lock pins a moving revision) and
//!    `Unmanifested` (a glob table with no per-file manifest); neither covers
//!    bytes produced by the job and never published, so such an artifact could
//!    not be keyed at all.

// **The absent half of the gate, provable only as a compile error.** Under
// plain `face` the manifest module does not exist, and no runtime assertion can
// see an absence — so the proof is a `compile_fail` doctest, and it is attached
// only in the configuration whose claim it is. Its two-sided partner, plus the
// deliberate assertion that a caller CAN write this artifact's manifest by hand
// under plain `face`, is `tests/face/gate.rs`.
#![cfg_attr(
  not(feature = "commercial-face-arcface"),
  doc = r#"
# Under plain `face`, the registered artifact's manifest is not here

`arcface` carries `#[cfg(feature = "commercial-face-arcface")]`, so naming it
without that feature is a resolution failure rather than a runtime refusal:

```compile_fail,E0433
let _ = coremlit::embeddings::face::arcface::MODEL;
```

What this feature still lets a caller do is write that manifest themselves —
`FaceModel::new("data", "embedding", 512)` — and load their own artifact with
it. That is deliberate and is asserted in `tests/face/gate.rs`: the gate
governs what coremlit WIRES, never which bytes a caller may hold.
"#
)]

pub mod align;
pub mod artifact;
pub mod embed;
pub mod error;

// The staged `w600k_r50` artifact's manifest: what to load, how it wants its
// pixels, and which placement it was measured fastest on. Behind
// `commercial-face-arcface` because the weights and their corpus are
// research-only; everything else in this module works on any artifact the
// caller supplies and is not gated.
//
// A PLAIN comment rather than a `///` one, and that is load-bearing: an outer
// doc comment on a `mod` declaration merges with the module file's own `//!`
// block and rustdoc then resolves the WHOLE merged block from the outer
// fragment's scope — so `super` inside `arcface/mod.rs` would mean
// `embeddings`, and every relative link in it would break or go redundant. The
// module documents itself.
#[cfg(feature = "commercial-face-arcface")]
pub mod arcface;

pub use artifact::ArtifactDigest;

pub use align::{
  ARCFACE_TEMPLATE, AlignedFace, FaceAlign, FaceCrop, LANDMARK_COUNT, MAX_CROP_AXIS, Point,
  SimilarityTransform, TEMPLATE_BYTES, TEMPLATE_SIZE,
};
pub use embed::{
  ChannelOrder, DEFAULT_FACE_COMPUTE, EmbeddingSpace, FaceEmbedder, FaceEmbedderOptions,
  FaceEmbedding, FaceModel, Preprocessing, TensorLayout,
};
