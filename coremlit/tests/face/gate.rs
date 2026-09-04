//! **What `commercial-face-arcface` gates, asserted from both sides — and what
//! it deliberately does not gate.**
//!
//! The feature governs what coremlit WIRES: the `embeddings::face::arcface`
//! manifest module, the `MODELS_LOCK` table that stages InsightFace's
//! `w600k_r50` bundle for CI, and the four gated suites in `tests/face/`. It
//! does not — and must not — govern what a caller may load.
//! [`FaceEmbedder::load`] takes a caller-supplied path and a caller-written
//! [`FaceModel`], and nothing but a digest would separate a product's own
//! commercially licensed ArcFace-shaped model from InsightFace's. That residual
//! is issue #138 §8's ("the register governs what the crate's *features* wire;
//! `Model::load` is public and any consumer can load any bytes") and is stated
//! in `tests/model_licences.rs`'s module doc.
//!
//! So this file asserts what is TRUE rather than installing a denylist, and it
//! is written to be read by whoever later reaches for one:
//!
//!   - **the gate's half.** Under `commercial-face-arcface` the manifest
//!     constant exists, and it is the same VALUE a caller can write by hand —
//!     which is the point: only provenance, never the value, distinguishes
//!     coremlit's registered manifest from anyone else's. The absent half is a
//!     `compile_fail` doctest on the `embeddings::face` module page, attached
//!     only when the feature is OFF, because absence is a resolution failure
//!     and no runtime assertion can see it.
//!   - **the door's half.** Under plain `face`,
//!     `FaceModel::new("data", "embedding", 512)` is constructible and this
//!     door accepts it, refusing a wrong artifact on the CONTRACT and never on
//!     the identity of its bytes. **That is deliberate, and this test exists so
//!     it cannot be "fixed" into a denylist by accident.** Refusing this
//!     manifest, or these bytes by digest, would be (1) policy enforcement on
//!     bytes, which issue #138 §8's stated residual already places outside this
//!     library, (2) bypassed by calling `coremlit::Model::load` directly, and
//!     (3) a contradiction of this door's founding principle that a manifest is
//!     a value — a caller with a commercially licensed ArcFace model of their
//!     own must be able to write its manifest and load it under plain `face`.
//!
//! Hermetic, and NOT `#[ignore]`d: the one bundle loaded here is
//! `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc`, 1.1 MiB of
//! committed bytes staged by no download.

// The workspace-root anchor `Models/` is resolved against. FOUND by searching
// upward for the `[workspace]` manifest rather than counted in `../` hops — see
// its module doc.
#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use coremlit::{
  ComputeUnits,
  embeddings::face::{FaceEmbedder, FaceEmbedderOptions, FaceModel, Preprocessing, error::Error},
};

/// The manifest for InsightFace's `w600k_r50`, spelled out here exactly as a
/// caller of the plain `face` feature has to spell it.
///
/// Hand-written on purpose. Importing `arcface::MODEL` would make this file
/// need the gated feature and would prove nothing about what a caller without
/// it can do.
const CALLER_WRITTEN: FaceModel = FaceModel::new("data", "embedding", 512);

/// **A caller can write this artifact's manifest under plain `face`.**
///
/// The deliberate assertion. `FaceModel::new` is `const` and total, so the
/// value exists whatever feature set is on; what this pins is that it stays
/// that way — a later refusal keyed on these strings and this width would be a
/// denylist over a caller's own property, and it would red here.
#[test]
fn the_arcface_shaped_manifest_is_a_value_any_plain_face_caller_can_write() {
  assert_eq!(CALLER_WRITTEN.input(), "data");
  assert_eq!(CALLER_WRITTEN.output(), "embedding");
  assert_eq!(CALLER_WRITTEN.dim(), 512);
  assert_eq!(
    CALLER_WRITTEN.preprocessing(),
    Preprocessing::ARCFACE,
    "the door's default preprocessing is the ArcFace one, so a caller writes the whole manifest \
     with one call"
  );
}

/// **The door refuses a wrong artifact on its CONTRACT, never on the identity
/// of its bytes.**
///
/// Driven with the hand-written manifest above over the committed silero
/// bundle, so it runs with no download: silero declares `audio_input`, not
/// `data`, and the refusal must name that missing feature. A refusal on any
/// other ground — a digest, a path, a bundle name — would be this door judging
/// bytes the caller supplied, which is exactly what issue #138 §8's residual
/// says this library does not do.
#[test]
fn the_door_refuses_a_wrong_artifact_on_its_contract_and_not_on_its_bytes() {
  let bundle = workspace_root::models_root()
    .join("vadkit")
    .join("silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; looked for {}",
    bundle.display()
  );

  let error = FaceEmbedder::load(
    &bundle,
    CALLER_WRITTEN,
    FaceEmbedderOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect_err("silero declares no `data` feature");
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == CALLER_WRITTEN.input() && m.actual().contains("audio_input")),
    "the refusal has to name the missing FEATURE. Anything else here would mean the door had \
     started deciding which artifacts a caller is allowed to name: {error}"
  );
}

/// **Under `commercial-face-arcface` the manifest module exists, and it is that
/// same value.**
///
/// The other side of the two-sided claim; the absent side is the `compile_fail`
/// doctest on the `embeddings::face` module page, which exists only when this
/// feature is off. What the equality says is the whole reason the register is a
/// claim about WIRING and not about bytes: coremlit's registered manifest is
/// not a privileged value, it is the same four fields any caller can write, and
/// what the feature adds is the staging, the tests and the licence row behind
/// them.
#[cfg(feature = "commercial-face-arcface")]
#[test]
fn under_the_commercial_gate_the_registered_manifest_is_that_same_value() {
  use coremlit::embeddings::face::arcface;

  assert_eq!(
    arcface::MODEL,
    CALLER_WRITTEN,
    "the registered manifest is a value a caller can write, not a privileged one"
  );
  assert!(
    arcface::STAGED_PATH.ends_with(arcface::BUNDLE_NAME),
    "the staged path names the bundle: {}",
    arcface::STAGED_PATH
  );
}
