//! End-to-end cross-modal ranking + prewarm smoke.
//!
//! # Status: Wave C (model-gated)
//!
//! `#[ignore]`d until the conversion is staged (`SIGLIP_TEST_MODELS`) and the
//! corpus committed (Wave B). Loads both towers (defaults), embeds every corpus
//! image + caption, and asserts each caption ranks its own image top-1 (via
//! `siglip::rank`, the default `CpuAndGpu` arm), exercises `prewarm`, and one
//! `from_memory` construction path.

mod common;

use coremlit::embeddings::siglip::{
  Candidate, Embedding, ImageEmbedder, ImageEmbedderOptions, PreprocessedImage, Rgb8Image,
  TextEmbedder, rank,
};

fn decode(file: &str) -> (Vec<u8>, usize, usize) {
  common::decode_png_rgb8(&common::fixture_path(&format!("goldens/{file}")))
}

/// Each caption ranks its matched image top-1 (cross-modal retrieval), on the
/// default `CpuAndGpu` arm. Exercises `prewarm` and the `from_memory` construction
/// path (sidecar bytes) alongside `from_files`.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn each_caption_ranks_its_image_top1() {
  // from_memory: load the sidecar bytes ourselves (the bring-your-own-grid path).
  let pos_bytes = std::fs::read(common::pos_embed_path()).expect("read sidecar");
  let image = ImageEmbedder::from_memory(
    common::vision_model_path(),
    &pos_bytes,
    ImageEmbedderOptions::new(),
  )
  .expect("load vision (from_memory)");
  let text = TextEmbedder::from_file(common::text_model_path()).expect("load text");
  image.prewarm().expect("prewarm vision");
  text.prewarm().expect("prewarm text");

  let (images, texts) = common::golden_corpus();

  // Embed every corpus image (decode PNG -> Rgb8Image -> embed).
  let image_embs: Vec<(String, Embedding)> = images
    .iter()
    .map(|g| {
      let (rgb, w, h) = decode(&g.file);
      let emb = image
        .embed(Rgb8Image::new(&rgb, w, h).expect("rgb"))
        .expect("embed image");
      (g.id.clone(), emb)
    })
    .collect();

  // Each image's caption must rank that image top-1 over all six.
  let candidates: Vec<Candidate<'_>> = image_embs
    .iter()
    .map(|(id, e)| Candidate::new(id, e))
    .collect();
  for g in &images {
    let caption = texts
      .iter()
      .find(|t| t.id == g.caption_id)
      .unwrap_or_else(|| panic!("no caption {} for image {}", g.caption_id, g.id));
    let query = text.embed(&caption.text).expect("embed caption");
    let ranked = rank(&query, &candidates);
    println!(
      "  caption {:14} -> top1 {} ({:.4}), want {}",
      caption.id,
      ranked[0].label(),
      ranked[0].score(),
      g.id
    );
    assert_eq!(
      ranked[0].label(),
      g.id,
      "caption {:?} ({:?}) did not rank its image {:?} top-1",
      caption.id,
      caption.text,
      g.id
    );
  }
}

/// `embed(img)` and `embed_preprocessed(preprocess(img))` agree: `embed` routes
/// through `embed_preprocessed`, so both feed byte-identical tensors to CoreML and
/// the tolerance only absorbs run-to-run prediction jitter. Also round-trips the
/// pipeline's bundle through the public validator at the model's real budget.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn embed_preprocessed_matches_embed_identity() {
  let embedder = ImageEmbedder::from_files(common::vision_model_path(), common::pos_embed_path())
    .expect("load vision");
  let (images, _texts) = common::golden_corpus();
  let (rgb, w, h) = decode(&images[0].file);
  let img = Rgb8Image::new(&rgb, w, h).expect("rgb");

  let a = embedder.embed(img).expect("embed");
  let pre = embedder.preprocess(img).expect("preprocess");
  let b = embedder
    .embed_preprocessed(&pre)
    .expect("embed_preprocessed");
  assert!(
    a.is_close(&b, 1e-6),
    "embed and embed_preprocessed must agree"
  );

  assert!(
    PreprocessedImage::try_new(
      pre.pixel_values().to_vec(),
      pre.position_embeddings().to_vec(),
      pre.attention_mask().to_vec(),
      embedder.max_num_patches(),
    )
    .is_ok(),
    "the pipeline's own bundle must pass the public validator"
  );
}
