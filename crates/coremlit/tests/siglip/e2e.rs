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
