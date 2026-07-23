//! Shared helpers for the siglip embedding tests.
//!
//! Two data sources, kept distinct:
//!
//! - **Committed goldens** (`tests/siglip/fixtures/goldens/`) — the in-tree
//!   transformers-fp32 oracle (per-image + per-text unit-normalized embeddings,
//!   the padded token windows, and small preprocessing oracles) plus the CC0
//!   corpus PNGs. Read hermetically; no model, no network. Staged by the port
//!   plan's golden-generation step (Wave B).
//! - **CoreML artifacts** (`siglip2_vision_512.mlmodelc`, `siglip2_text_64.mlmodelc`,
//!   and `pos_embed_16x16x768.f32le.bin`) — gitignored dev-time downloads from
//!   `FinDIT-Studio/siglip2-naflex-coreml` under `Models/siglip2-naflex/`
//!   (overridable via `SIGLIP_TEST_MODELS`). Model-gated tests are `#[ignore]` by
//!   default and run only when the owner stages the conversion (Wave C).

use std::path::{Path, PathBuf};

/// The HF revision (commit SHA) of `FinDIT-Studio/siglip2-naflex-coreml` the
/// model artifacts and their per-file SHA-256 pins are recorded at. Pinned by
/// the conversion runbook's upload step (Wave C / U2); a placeholder until then.
#[allow(dead_code)]
pub const SIGLIP_REVISION: &str = "<pending conversion upload (Wave C / runbook U2)>";

/// Directory containing the downloaded siglip CoreML artifact tree.
///
/// Overridable via `SIGLIP_TEST_MODELS`; otherwise
/// `<workspace>/Models/siglip2-naflex` — gitignored, fetched dev-time (mirrors
/// the `EMBEDKIT_TEST_MODELS` / `CLAPKIT_TEST_MODELS` conventions).
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("SIGLIP_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("siglip2-naflex")
    },
    PathBuf::from,
  )
}

/// The 512-tier bundle's parent directory
/// (`.../siglip2-base-patch16-naflex-512`), which holds the `.mlmodelc` bundles,
/// the pos-emb sidecar, and `CHECKSUMS.sha256`.
#[allow(dead_code)]
pub fn model_root() -> PathBuf {
  models_dir().join("siglip2-base-patch16-naflex-512")
}

/// Path to the compiled vision encoder, `siglip2_vision_512.mlmodelc`.
#[allow(dead_code)]
pub fn vision_model_path() -> PathBuf {
  model_root().join("siglip2_vision_512.mlmodelc")
}

/// Path to the compiled text encoder, `siglip2_text_64.mlmodelc`.
#[allow(dead_code)]
pub fn text_model_path() -> PathBuf {
  model_root().join("siglip2_text_64.mlmodelc")
}

/// Path to the base position-grid sidecar, `pos_embed_16x16x768.f32le.bin`.
#[allow(dead_code)]
pub fn pos_embed_path() -> PathBuf {
  model_root().join("pos_embed_16x16x768.f32le.bin")
}

/// Absolute path to a committed fixture under `crates/coremlit/tests/siglip/fixtures`.
#[allow(dead_code)]
pub fn fixture_path(relative: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("siglip")
    .join("fixtures")
    .join(relative)
}

/// One committed golden IMAGE entry: geometry, the measured `spatial_shapes`
/// grid, its matched caption id, and the transformers-fp32 **unit-normalized**
/// 768-d embedding.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub struct ImageGolden {
  /// Stable per-entry id.
  pub id: String,
  /// Corpus-relative PNG path (`images/<id>.png`).
  pub file: String,
  /// Source URL (provenance).
  pub source: String,
  /// Redistributable license (CC0 / public-domain / owner-authored).
  pub license: String,
  /// Decoded image width in pixels.
  pub width: usize,
  /// Decoded image height in pixels.
  pub height: usize,
  /// The measured `(grid_h, grid_w)` patch grid at the 512 budget.
  pub spatial_shapes: [usize; 2],
  /// The matched caption's `TextGolden::id`.
  pub caption_id: String,
  /// The transformers-fp32 unit-normalized 768-d image embedding.
  pub embedding: Vec<f32>,
}

/// One committed golden TEXT entry: the raw text, the EXACT padded `[T]` token
/// window (side/id load-bearing, D6), the real-token count, and the
/// transformers-fp32 **unit-normalized** 768-d embedding.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub struct TextGolden {
  /// Stable per-entry id.
  pub id: String,
  /// The raw input string.
  pub text: String,
  /// The processor's EXACT padded token window (length `T`).
  pub token_ids_padded: Vec<i32>,
  /// Real (non-pad) token count.
  pub n_real: usize,
  /// The transformers-fp32 unit-normalized 768-d text embedding.
  pub embedding: Vec<f32>,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct Corpus {
  images: Vec<ImageGolden>,
  texts: Vec<TextGolden>,
}

/// Load the committed golden corpus (images + texts). Hermetic — reads the
/// in-tree fixture, never `Models/`. (Staged by Wave B; until then the fixture
/// is absent and only the model-gated / goldens-gated tests that call this are
/// affected — they are `#[ignore]`d.)
#[allow(dead_code)]
pub fn golden_corpus() -> (Vec<ImageGolden>, Vec<TextGolden>) {
  let path = fixture_path("goldens/corpus.json");
  let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  let corpus: Corpus =
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
  (corpus.images, corpus.texts)
}

/// Lowercase-hex SHA-256 of a byte slice.
#[allow(dead_code)]
pub fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Lowercase-hex SHA-256 of a file's contents.
#[allow(dead_code)]
pub fn sha256_file(path: &Path) -> String {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  sha256_hex(&bytes)
}

/// Cosine of two equal-length finite vectors — FAIL-CLOSED (the #30 lesson):
/// panics on a length mismatch or any non-finite component rather than returning
/// a silently-wrong scalar.
#[allow(dead_code)]
pub fn cosine_checked(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), b.len(), "cosine operands differ in length");
  assert!(!a.is_empty(), "cosine of empty vectors is undefined");
  let mut dot = 0.0f64;
  let mut na = 0.0f64;
  let mut nb = 0.0f64;
  for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
    assert!(x.is_finite(), "left operand non-finite at {i}");
    assert!(y.is_finite(), "right operand non-finite at {i}");
    dot += (x as f64) * (y as f64);
    na += (x as f64) * (x as f64);
    nb += (y as f64) * (y as f64);
  }
  assert!(na > 0.0 && nb > 0.0, "cosine of a zero vector is undefined");
  (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Recursively collects the forward-slash relative path of every FILE under
/// `dir` into `out`. Pure path enumeration — hermetically testable without real
/// model bytes. OS-generated sidecars (AppleDouble `._*`, `.DS_Store`) are
/// skipped: CoreML's loader never reads them, so excluding them cannot mask a
/// functional change, whereas keeping them would false-fail the exact-set gate.
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

/// Assert that the compiled-model bundle at `dir` matches an EXACT pinned
/// manifest (`(relative_path, sha256)` list): the discovered file set must EQUAL
/// the pinned key set (so a missing OR an added artifact both red), and each
/// file's SHA-256 must equal its pinned value. `std`-only recursive walk;
/// forward-slash relative keys.
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
       extra (on disk but not pinned): {extra:?}\n  \
       if the bundle changed intentionally, update the pinned `cases` list AND \
       the doc SHA table"
    );
  }

  for (relative, expected) in cases {
    let actual = sha256_file(&dir.join(relative));
    assert_eq!(
      &actual, expected,
      "sha256 drift on artifact {relative} under {dir:?}"
    );
  }
}

/// Decode a committed corpus PNG to interleaved RGB8 `(bytes, width, height)`
/// via the `png` dev-dep — the sans-I/O crate takes decoded RGB, so decoding is
/// the test's job. Accepts 8-bit RGB or RGBA (alpha dropped).
#[allow(dead_code)]
pub fn decode_png_rgb8(path: &Path) -> (Vec<u8>, usize, usize) {
  let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
  let mut reader = png::Decoder::new(file)
    .read_info()
    .unwrap_or_else(|e| panic!("read png header {path:?}: {e}"));
  let mut buf = vec![0u8; reader.output_buffer_size()];
  let info = reader
    .next_frame(&mut buf)
    .unwrap_or_else(|e| panic!("decode png {path:?}: {e}"));
  assert_eq!(
    info.bit_depth,
    png::BitDepth::Eight,
    "corpus PNGs must be 8-bit ({path:?})"
  );
  let (w, h) = (info.width as usize, info.height as usize);
  let rgb: Vec<u8> = match info.color_type {
    png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
    png::ColorType::Rgba => buf[..info.buffer_size()]
      .as_chunks::<4>()
      .0
      .iter()
      .flat_map(|p| [p[0], p[1], p[2]])
      .collect(),
    other => panic!("unsupported PNG color type {other:?} for {path:?} (need RGB/RGBA)"),
  };
  assert_eq!(
    rgb.len(),
    w * h * 3,
    "decoded RGB length mismatch for {path:?}"
  );
  (rgb, w, h)
}
