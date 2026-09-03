//! Shared pins and fixture loaders for the `commercial-face-arcface` gates.
//!
//! **These gates run against research-only weights.** InsightFace publishes
//! `w600k_r50` for non-commercial research only and WebFace600K is a signed
//! research-only agreement, so the artifact they load may be used to develop,
//! evaluate and test and never to ship. That is why every one of them sits
//! behind a `commercial-`prefixed feature outside `default`, and why the
//! artifact repository is PRIVATE: CI fetching our own conversion is USE,
//! which is the line `NOTICE` draws ("CI DOWNLOADS; IT DOES NOT
//! REDISTRIBUTE"). A tokenless `hf download` cannot stage this kit — see the
//! `arcface` shard in `.github/workflows/ci.yml`.
//!
//! Two data sources, kept distinct:
//!
//! - **Committed fixtures** (`tests/face/fixtures/`) — 18 aligned 112×112
//!   crops of six people, every source photograph a work of the U.S. federal
//!   government in the public domain, plus `faces/manifest.json` (the
//!   detection, the five landmarks and the solved alignment matrix per face)
//!   and `onnx_reference.json` (the fp32 `onnxruntime` embedding of each crop).
//!   All of it reads with no artifact and no network.
//! - **CoreML artifact** ([`BUNDLE_NAME`]) — a gitignored dev-time download
//!   staged under `Models/facekit/`, overridable via `FACEKIT_TEST_MODELS`.
//!   Model-gated tests are `#[ignore]`d and run only when it is staged.
//!
//! # Why the crops are the input rather than the photographs
//!
//! The five landmarks in `faces/manifest.json` live in the coordinate space of
//! the full `~medium` NASA asset, and those bytes are NOT committed (the
//! `<id>.jpg` beside each crop is a 640 px re-encode for a human to look at,
//! not the crop's source). So the warp itself cannot be re-run here. What CAN
//! be re-run from committed data is the half where a wrong answer is silent:
//! `the_rust_solve_reproduces_every_committed_alignment_matrix` in `parity.rs`
//! puts each face's landmarks through
//! [`SimilarityTransform::estimate`][est] and compares against the matrix the
//! oracle solved. The resampler that follows it is bit-exact with
//! `cv2.warpAffine` and is goldened byte for byte by
//! `tests/face/align_golden.rs`.
//!
//! [est]: coremlit::embeddings::face::SimilarityTransform::estimate

// The workspace-root anchor `models_dir()` resolves against. FOUND by
// searching upward for the `[workspace]` manifest, never counted in `../`
// hops — see its module doc for why a count is the wrong shape here.
#[path = "../../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;
#[allow(unused_imports)]
pub use workspace_root::{checkout_parent, models_root, workspace_root};

use std::path::{Path, PathBuf};

use coremlit::embeddings::face::{
  ARCFACE_TEMPLATE, LANDMARK_COUNT, Point, SimilarityTransform, TEMPLATE_BYTES,
};

/// Hugging Face repository the artifact comes from. **Private** — see the
/// module doc.
#[allow(dead_code)]
pub const HF_REPO: &str = "FinDIT-Studio/facekit-coreml";

/// Pinned artifact-repo revision the SHA-256 table below was taken from.
#[allow(dead_code)]
pub const HF_REVISION: &str = "70e212696bd3c472e28718e2e39c79467b97805e";

/// The upstream release asset the conversion consumed.
///
/// **InsightFace publishes no digest for this pack**, and that is a finding
/// rather than an oversight: `insightface/utils/storage.py` — the code every
/// user of the Python package runs — builds a CloudFront URL and unzips
/// whatever arrives, with no manifest, no signature and no hash anywhere on
/// that path. So [`SOURCE_PACK_SHA256`] is a WITNESS to the bytes this
/// conversion consumed, not a verification against an upstream claim.
#[allow(dead_code)]
pub const SOURCE_PACK: &str = "buffalo_l.zip";

/// SHA-256 of [`SOURCE_PACK`], as downloaded on 2026-09-03.
#[allow(dead_code)]
pub const SOURCE_PACK_SHA256: &str =
  "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f";

/// The one member of the pack the recipe converts. The other four — a
/// detector, two landmark models and a gender/age head — are never converted
/// and never published; only `det_10g.onnx` is read at all, and only to cut
/// the committed fixtures.
#[allow(dead_code)]
pub const SOURCE_MEMBER: &str = "w600k_r50.onnx";

/// SHA-256 of [`SOURCE_MEMBER`] inside the pinned pack.
#[allow(dead_code)]
pub const SOURCE_MEMBER_SHA256: &str =
  "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43";

/// The `deepinsight/insightface` revision whose `ArcFaceONNX` preprocessing
/// and `face_align.norm_crop` this kit reproduces — the SAME commit
/// `align_oracle.py` pins, so the alignment the fixtures were cut with and the
/// alignment the Rust door is goldened against are one specification.
#[allow(dead_code)]
pub const INSIGHTFACE_REVISION: &str = "ffa12d315041c0505b077c7ff057ca914bb8dc7e";

/// The compiled bundle's directory name, restated here rather than imported.
///
/// The library's own `arcface::BUNDLE_NAME` is the value the door's users
/// read; this is the value the GATE resolves its path from, and
/// `the_library_and_this_suite_name_one_bundle` asserts the two agree. One
/// constant read twice proves nothing about either.
#[allow(dead_code)]
pub const BUNDLE_NAME: &str = "w600k_r50.mlmodelc";

/// Exact per-file SHA-256 of the staged `.mlmodelc` bundle, as published at
/// [`HF_REVISION`]. The set is exact: a missing OR an added file reds, not
/// just a changed one.
#[allow(dead_code)]
pub const ARTIFACT_SHA256: &[(&str, &str)] = &[
  (
    "analytics/coremldata.bin",
    "1320de26a121f36a6dde0a1faab329d69006560709c0671cc1254cfccf4cdb5f",
  ),
  (
    "coremldata.bin",
    "f95b43443adb213f1f96136b42e5113c539ce897efa916b8982b82f03e92a38f",
  ),
  (
    "metadata.json",
    "a0dd2ec43b8182a7184a4116df29c2e10afd454be1f239e4b5e1efb507fa22b8",
  ),
  (
    "model.mil",
    "050f69f10f5687971fb8f808d9da53b01a8d512c7013346fbc22daa948e42d26",
  ),
  (
    "weights/weight.bin",
    "aa08d7826a70f9bc237ea0532a5eec12cb83b8375148a1b0650f104cbb2ff492",
  ),
];

/// The house floor for a cross-implementation parity claim: `>= 0.99` cosine.
///
/// The same number `tests/*/placement.rs::SANITY_COS` and
/// `conversion/*/verify_*.py::SANITY_COS_FLOOR` hold, and the same one issue
/// #115 set for this kit on measured grounds: the ANE's own fp16 error for an
/// IResNet is `1 − cos ≈ 0.0015` typical while the cheapest REAL preprocessing
/// bug costs `0.083`, so 0.99 sits ~4× above the noise and ~8× below the
/// cheapest bug.
///
/// **Deliberately not tightened to the recipe's measured `1 − cos` of
/// 2.2 × 10⁻⁴.** That was taken on one machine and one OS version, and fp16
/// placement numerics are the most host-dependent thing in this crate; pinning
/// an observation as the requirement converts an OS update into a false
/// failure. A defect of the kind this floor exists for lands far below it.
#[allow(dead_code)]
pub const SANITY_COS: f64 = 0.99;

/// Directory holding the downloaded CoreML artifact tree.
///
/// Overridable via `FACEKIT_TEST_MODELS`; otherwise `<workspace>/Models/facekit`
/// — gitignored, fetched dev-time, the same convention `IDENTITY_TEST_MODELS`
/// and `LID_TEST_MODELS` follow. The directory is named for the KIT while the
/// module and the feature are named for the lane and the artifact
/// respectively, which is the split `MODELS_LOCK` records.
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("FACEKIT_TEST_MODELS").map_or_else(
    || workspace_root::models_root().join("facekit"),
    PathBuf::from,
  )
}

/// Path to the compiled graph under [`models_dir`].
#[allow(dead_code)]
pub fn model_path() -> PathBuf {
  models_dir().join(BUNDLE_NAME)
}

/// `tests/face/fixtures/`.
#[allow(dead_code)]
pub fn fixtures_dir() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("face")
    .join("fixtures")
}

/// Lowercase-hex SHA-256 of a byte slice.
#[allow(dead_code)]
pub fn sha256_hex(bytes: &[u8]) -> String {
  use core::fmt::Write;

  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .fold(String::new(), |mut acc, b| {
      let _ = write!(acc, "{b:02x}");
      acc
    })
}

/// Lowercase-hex SHA-256 of a file's contents.
#[allow(dead_code)]
pub fn sha256_file(path: &Path) -> String {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  sha256_hex(&bytes)
}

/// Recursively collect the forward-slash relative path of every FILE under
/// `dir`. OS-generated sidecars (AppleDouble `._*`, `.DS_Store`) are skipped:
/// CoreML's loader never reads them, so excluding them cannot mask a
/// functional change, while keeping them would false-fail the exact-set gate.
#[allow(dead_code)]
pub fn collect_files_rel(dir: &Path, prefix: &str, out: &mut Vec<String>) {
  let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"));
  for entry in entries {
    let entry = entry.unwrap_or_else(|e| panic!("dir entry under {dir:?}: {e}"));
    let name = entry.file_name().to_string_lossy().into_owned();
    if name.starts_with("._") || name == ".DS_Store" {
      continue;
    }
    let rel = if prefix.is_empty() {
      name
    } else {
      format!("{prefix}/{name}")
    };
    let file_type = entry
      .file_type()
      .unwrap_or_else(|e| panic!("file_type {:?}: {e}", entry.path()));
    if file_type.is_dir() {
      collect_files_rel(&entry.path(), &rel, out);
    } else {
      out.push(rel);
    }
  }
}

/// Assert the bundle at `dir` matches the EXACT pinned manifest: the
/// discovered file set must EQUAL the pinned key set (so a missing OR an added
/// artifact both red), and each file's SHA-256 must equal its pinned value.
#[allow(dead_code)]
pub fn assert_exact_sha_manifest(dir: &Path, cases: &[(&str, &str)]) {
  use std::collections::BTreeSet;

  let mut found = Vec::new();
  collect_files_rel(dir, "", &mut found);
  let on_disk: BTreeSet<String> = found.into_iter().collect();
  let pinned: BTreeSet<String> = cases.iter().map(|(rel, _)| (*rel).to_owned()).collect();

  if on_disk != pinned {
    let missing: Vec<&String> = pinned.difference(&on_disk).collect();
    let extra: Vec<&String> = on_disk.difference(&pinned).collect();
    panic!(
      "artifact manifest mismatch under {dir:?}:\n  \
       missing (pinned but not on disk): {missing:?}\n  \
       extra (on disk but not pinned): {extra:?}"
    );
  }

  for (relative, expected) in cases {
    assert_eq!(
      &sha256_file(&dir.join(relative)),
      expected,
      "sha256 drift on artifact {relative} under {dir:?}"
    );
  }
}

/// One committed fixture face: its aligned crop, who it is, the five landmarks
/// the oracle solved from, the matrix it solved, and the fp32 `onnxruntime`
/// embedding of the crop.
#[allow(dead_code)]
pub struct Face {
  /// The fixture id, e.g. `whitson_iss005e07178`.
  pub id: String,
  /// The person key the same-person pairs are formed on.
  pub person: String,
  /// The aligned 112×112 RGB8 crop, `TEMPLATE_BYTES` long.
  pub crop: Vec<u8>,
  /// The five landmarks in the ORIGINAL NASA asset's pixel space.
  pub landmarks: [Point; LANDMARK_COUNT],
  /// The 2×3 row-major matrix `align_oracle.py` solved, as
  /// [`SimilarityTransform::matrix`] orders it.
  pub align_matrix: [f64; 6],
  /// The fp32 `onnxruntime` embedding of [`Self::crop`], RAW.
  pub reference: Vec<f32>,
  /// The recorded L2 norm of [`Self::reference`].
  pub reference_norm: f64,
}

/// Everything `onnx_reference.json` records beside the vectors.
#[allow(dead_code)]
pub struct Reference {
  /// The 18 faces, in fixture-manifest order.
  pub faces: Vec<Face>,
  /// The embedding width the reference was cut at.
  pub dim: usize,
  /// InsightFace's own "same person" threshold, as the fixture records it.
  pub same_min: f64,
  /// InsightFace's own "not the same person" threshold.
  pub different_max: f64,
  /// The minimum same-person cosine the reference itself achieves.
  pub reference_min_same: f64,
  /// The maximum different-person cosine the reference itself achieves.
  pub reference_max_different: f64,
  /// The two ids setting [`Self::reference_min_same`].
  pub worst_same_ids: [String; 2],
  /// The two ids setting [`Self::reference_max_different`].
  pub worst_different_ids: [String; 2],
}

/// Read `faces/manifest.json` and `onnx_reference.json` and join them,
/// asserting every field the gates are about to rely on.
///
/// The strictness is the point. A reference cut against different crops, a
/// crop whose bytes have moved, a reference that lost its vectors or one that
/// was silently normalised would each let a parity comparison pass while
/// measuring the wrong thing, so each is refused here rather than downstream.
#[allow(dead_code)]
pub fn load_reference() -> Reference {
  let faces_dir = fixtures_dir().join("faces");
  let manifest = read_json(&faces_dir.join("manifest.json"));
  let reference = read_json(&fixtures_dir().join("onnx_reference.json"));

  assert_eq!(
    reference["source"]["member_sha256"].as_str(),
    Some(SOURCE_MEMBER_SHA256),
    "the reference was cut from a different ONNX than this kit converts"
  );
  assert_eq!(
    reference["source"]["pack_sha256"].as_str(),
    Some(SOURCE_PACK_SHA256),
    "the reference's pack pin is not the one the conversion consumed"
  );
  assert_eq!(
    reference["source"]["precision"].as_str(),
    Some("fp32"),
    "the reference must be fp32; an fp16 oracle would measure the artifact against itself"
  );
  assert_eq!(reference["preprocessing"]["order"].as_str(), Some("rgb"));
  assert_eq!(reference["preprocessing"]["layout"].as_str(), Some("nchw"));
  let dim = usize::try_from(reference["dim"].as_u64().expect("`dim`")).expect("dim fits");

  let manifest_faces = manifest["faces"].as_array().expect("`faces` array");
  let reference_faces = reference["faces"].as_array().expect("`faces` array");
  assert_eq!(
    manifest_faces.len(),
    reference_faces.len(),
    "the reference covers a different number of faces than the fixture manifest lists"
  );

  let faces = manifest_faces
    .iter()
    .zip(reference_faces)
    .map(|(row, reference)| {
      let id = row["id"].as_str().expect("fixture id").to_owned();
      assert_eq!(
        reference["id"].as_str(),
        Some(id.as_str()),
        "the reference and the fixture manifest disagree on face order"
      );
      let crop_sha = row["crop_sha256"].as_str().expect("crop sha");
      assert_eq!(
        reference["crop_sha256"].as_str(),
        Some(crop_sha),
        "{id}: the reference was cut from a different crop"
      );
      let crop_path = faces_dir.join(row["crop"].as_str().expect("crop name"));
      let crop = std::fs::read(&crop_path).unwrap_or_else(|e| panic!("read {crop_path:?}: {e}"));
      assert_eq!(crop.len(), TEMPLATE_BYTES, "{id}: crop length");
      assert_eq!(
        sha256_hex(&crop),
        crop_sha,
        "{id}: the committed crop's bytes have moved"
      );

      let points = row["detection"]["landmarks5"]
        .as_array()
        .unwrap_or_else(|| panic!("{id}: no landmarks5"));
      assert_eq!(points.len(), LANDMARK_COUNT, "{id}: landmark count");
      let mut landmarks = [Point::new(0.0, 0.0); LANDMARK_COUNT];
      for (slot, point) in landmarks.iter_mut().zip(points) {
        let xy = point.as_array().expect("landmark pair");
        *slot = Point::new(
          xy[0].as_f64().expect("landmark x") as f32,
          xy[1].as_f64().expect("landmark y") as f32,
        );
      }

      let rows = row["align_matrix"]
        .as_array()
        .unwrap_or_else(|| panic!("{id}: no align_matrix"));
      let mut align_matrix = [0.0f64; 6];
      for (i, matrix_row) in rows.iter().enumerate() {
        for (j, value) in matrix_row
          .as_array()
          .expect("matrix row")
          .iter()
          .enumerate()
        {
          align_matrix[i * 3 + j] = value.as_f64().expect("matrix entry");
        }
      }

      let embedding: Vec<f32> = reference["embedding"]
        .as_array()
        .unwrap_or_else(|| panic!("{id}: no reference embedding"))
        .iter()
        .map(|v| v.as_f64().expect("finite reference component") as f32)
        .collect();
      assert_eq!(embedding.len(), dim, "{id}: reference width");
      assert!(
        embedding.iter().all(|v| v.is_finite()),
        "{id}: non-finite reference component"
      );
      let reference_norm = reference["l2_norm"]
        .as_f64()
        .unwrap_or_else(|| panic!("{id}: no reference norm"));

      Face {
        id,
        person: row["person"].as_str().expect("person").to_owned(),
        crop,
        landmarks,
        align_matrix,
        reference: embedding,
        reference_norm,
      }
    })
    .collect();

  let pairs = &reference["known_pairs"];
  Reference {
    faces,
    dim,
    same_min: pairs["same_min"].as_f64().expect("same_min"),
    different_max: pairs["different_max"].as_f64().expect("different_max"),
    reference_min_same: pairs["min_same"].as_f64().expect("min_same"),
    reference_max_different: pairs["max_different"].as_f64().expect("max_different"),
    worst_same_ids: two_ids(&pairs["worst_same_ids"]),
    worst_different_ids: two_ids(&pairs["worst_different_ids"]),
  }
}

/// A two-element array of fixture ids out of the reference's pair record.
fn two_ids(value: &serde_json::Value) -> [String; 2] {
  let ids = value.as_array().expect("id pair");
  assert_eq!(ids.len(), 2, "a pair names two faces");
  [
    ids[0].as_str().expect("id").to_owned(),
    ids[1].as_str().expect("id").to_owned(),
  ]
}

/// Parse a committed JSON fixture, naming the file on either failure.
fn read_json(path: &Path) -> serde_json::Value {
  let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

/// The transform `SimilarityTransform::estimate` solves for one face's
/// committed landmarks onto [`ARCFACE_TEMPLATE`].
#[allow(dead_code)]
pub fn solved_transform(face: &Face) -> SimilarityTransform {
  SimilarityTransform::estimate(&face.landmarks, &ARCFACE_TEMPLATE)
    .unwrap_or_else(|e| panic!("{}: the committed landmarks solve to nothing: {e}", face.id))
}

/// Cosine similarity between two vectors, normalising both.
///
/// The door's own [`FaceEmbedding`][emb] is already unit-norm and has a
/// [`cosine`][cos] of its own; this is for comparing against the RAW ONNX
/// reference, whose norms run 17 – 25 on purpose.
///
/// [emb]: coremlit::embeddings::face::FaceEmbedding
/// [cos]: coremlit::embeddings::face::FaceEmbedding::cosine
#[allow(dead_code)]
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
  let dot: f64 = a
    .iter()
    .zip(b.iter())
    .map(|(x, y)| f64::from(*x) * f64::from(*y))
    .sum();
  let na = a
    .iter()
    .map(|x| f64::from(*x) * f64::from(*x))
    .sum::<f64>()
    .sqrt();
  let nb = b
    .iter()
    .map(|x| f64::from(*x) * f64::from(*x))
    .sum::<f64>()
    .sqrt();
  dot / (na * nb)
}
