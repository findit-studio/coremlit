//! End-to-end cross-modal ranking + prewarm smoke.
//!
//! # Status: Wave C shell (model-gated)
//!
//! `#[ignore]`d until the conversion is staged (`SIGLIP_TEST_MODELS`) and the
//! corpus committed (Wave B). Wave C loads both towers (defaults), embeds every
//! corpus image + caption, and asserts each caption ranks its own image top-1
//! (via `siglip::rank`, GPU arm), exercises `prewarm`, and one `from_memory`
//! construction path.

mod common;

use coremlit::embeddings::siglip::{ImageEmbedder, TextEmbedder};

/// Each caption ranks its matched image top-1 (cross-modal retrieval), on the
/// default `CpuAndGpu` arm. Wave C implements against the staged models + corpus.
#[test]
#[ignore = "requires staged siglip models + committed corpus — Wave C"]
fn each_caption_ranks_its_image_top1() {
  let image = ImageEmbedder::from_files(common::vision_model_path(), common::pos_embed_path())
    .expect("load vision");
  let text = TextEmbedder::from_file(common::text_model_path()).expect("load text");
  image.prewarm().expect("prewarm vision");
  text.prewarm().expect("prewarm text");
  let (_images, _texts) = common::golden_corpus();
  // Wave C: embed corpus images (decode_png_rgb8 → Rgb8Image → image.embed) and
  //         captions (text.embed), then siglip::rank each caption over the images
  //         and assert its matched image is top-1.
}

/// `embed(img)` and `embed_preprocessed(preprocess(img))` agree: `embed` routes
/// through `embed_preprocessed` by construction, so the two calls feed
/// byte-identical tensors to CoreML and the tolerance only absorbs run-to-run
/// prediction jitter (tighten toward 0.0 if Wave C measures bit-stability on
/// the pinned GPU arm). Also round-trips the pipeline's bundle through the
/// public validator at the model's real budget.
#[test]
#[ignore = "requires staged siglip models + committed corpus — Wave C"]
fn embed_preprocessed_matches_embed_identity() {
  let _embedder = ImageEmbedder::from_files(common::vision_model_path(), common::pos_embed_path())
    .expect("load vision");
  // Wave C: decode one corpus PNG → Rgb8Image `img`, then:
  //   let a = embedder.embed(img).expect("embed");
  //   let pre = embedder.preprocess(img).expect("preprocess");
  //   let b = embedder.embed_preprocessed(&pre).expect("embed_preprocessed");
  //   assert!(a.is_close(&b, 1e-6));
  //   assert!(
  //     PreprocessedImage::try_new(
  //       pre.pixel_values().to_vec(),
  //       pre.position_embeddings().to_vec(),
  //       pre.attention_mask().to_vec(),
  //       embedder.max_num_patches(),
  //     )
  //     .is_ok()
  //   );
}
