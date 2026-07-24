//! The siglip parity SHIP GATE — CoreML embeddings vs the committed
//! transformers-fp32 goldens, per compute unit.
//!
//! # Status: Wave C (model-gated)
//!
//! `#[ignore]`d until the owner stages the conversion (`SIGLIP_TEST_MODELS`) and
//! the committed goldens (Wave B). The `CpuAndGpu` arm is THE GATE (floor never
//! below 0.99917; measured worst here ≈ 0.99999 both towers — the vision path
//! preprocesses via colconv and the text path feeds the golden ids). The ANE /
//! CpuOnly arms are CHARACTERIZED in `placement.rs`, not floor-gated. Non-vacuity:
//! zeroed `position_embeddings` collapses vision parity far below the floor.

mod common;

use coremlit::embeddings::siglip::{ImageEmbedder, PreprocessedImage, Rgb8Image, TextEmbedder};

/// THE hard ship floor (spec §3). The measured band floor below is tighter.
const PARITY_FLOOR: f32 = 0.99917;
/// Measured-then-pinned band floor: the CpuAndGpu worst sits at ≈ 0.99999 on both
/// towers (probe class), so 0.9999 pins the measurement with jitter margin while
/// staying above the 0.99917 spec floor — a regression that merely clears the spec
/// floor but drops the measured parity still REDs the gate.
const MEASURED_FLOOR: f32 = 0.9999;

fn corpus_worst() -> (f32, f32) {
  let (images, texts) = common::golden_corpus();
  let image = ImageEmbedder::from_files(common::vision_model_path(), common::pos_embed_path())
    .expect("load vision (CpuAndGpu default)");
  let text = TextEmbedder::from_file(common::text_model_path()).expect("load text");

  let mut worst_img = 1.0f32;
  for g in &images {
    let (rgb, w, h) =
      common::decode_png_rgb8(&common::fixture_path(&format!("goldens/{}", g.file)));
    let emb = image
      .embed(Rgb8Image::new(&rgb, w, h).expect("rgb view"))
      .expect("embed image");
    let c = common::cosine_checked(emb.as_slice(), &g.embedding);
    println!("  image {:9} cos {c:.8}", g.id);
    worst_img = worst_img.min(c);
  }
  let mut worst_txt = 1.0f32;
  for g in &texts {
    let emb = text.embed(&g.text).expect("embed text");
    let c = common::cosine_checked(emb.as_slice(), &g.embedding);
    println!("  text  {:14} cos {c:.8}", g.id);
    worst_txt = worst_txt.min(c);
  }
  (worst_img, worst_txt)
}

/// The `CpuAndGpu` GATE: worst-corpus cosine vs the fp32 goldens holds the floor.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn cpu_and_gpu_arm_holds_the_parity_floor() {
  let (worst_img, worst_txt) = corpus_worst();
  println!("[parity] CpuAndGpu worst: vision {worst_img:.8}  text {worst_txt:.8}");
  let worst = worst_img.min(worst_txt);
  assert!(
    worst >= PARITY_FLOOR,
    "CpuAndGpu parity {worst:.8} below the hard spec floor {PARITY_FLOOR}"
  );
  assert!(
    worst_img >= MEASURED_FLOOR,
    "vision CpuAndGpu parity {worst_img:.8} below the measured band floor {MEASURED_FLOOR}"
  );
  assert!(
    worst_txt >= MEASURED_FLOOR,
    "text CpuAndGpu parity {worst_txt:.8} below the measured band floor {MEASURED_FLOOR}"
  );
  // Two-sided: a cosine cannot exceed 1 by more than fp32 slack — a value well
  // over 1 would mean a broken cosine or a non-unit golden.
  assert!(
    worst <= 1.000_01,
    "cosine over 1 — broken metric or non-unit golden"
  );
}

/// Non-vacuity: zeroing `position_embeddings` (the port's central novel step) must
/// collapse vision parity far below the floor — the pos-emb lift is load-bearing.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn zeroed_position_embeddings_break_the_gate() {
  let (images, _texts) = common::golden_corpus();
  let embedder = ImageEmbedder::from_files(common::vision_model_path(), common::pos_embed_path())
    .expect("load vision");
  let g = &images[0];
  let (rgb, w, h) = common::decode_png_rgb8(&common::fixture_path(&format!("goldens/{}", g.file)));
  let pre = embedder
    .preprocess(Rgb8Image::new(&rgb, w, h).expect("rgb view"))
    .expect("preprocess");
  let p = embedder.max_num_patches();

  // Real lift: holds the floor.
  let real = embedder.embed_preprocessed(&pre).expect("embed real");
  let c_real = common::cosine_checked(real.as_slice(), &g.embedding);

  // Zeroed lift (a LEGAL PreprocessedImage — zero rows pass validation): must break.
  let zeroed = PreprocessedImage::try_new(
    pre.pixel_values().to_vec(),
    vec![0.0f32; p * coremlit::embeddings::siglip::embedding::EMBEDDING_DIM],
    pre.attention_mask().to_vec(),
    p,
  )
  .expect("zeroed bundle is a legal input");
  let broken = embedder.embed_preprocessed(&zeroed).expect("embed zeroed");
  let c_zero = common::cosine_checked(broken.as_slice(), &g.embedding);

  println!("[non-vacuity] real cos {c_real:.6}  zeroed-pos cos {c_zero:.6}");
  assert!(
    c_real >= PARITY_FLOOR,
    "real lift must hold the floor ({c_real:.6})"
  );
  assert!(
    c_zero < 0.95,
    "zeroing position_embeddings must collapse parity far below the floor (got {c_zero:.6})"
  );
}
