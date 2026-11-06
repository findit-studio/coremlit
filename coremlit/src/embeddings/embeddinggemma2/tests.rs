use super::*;

/// The staged model directory: `EMBEDDINGGEMMA2_TEST_MODELS`, else the
/// workspace's `Models/embeddinggemma2`.
fn staged_dir() -> PathBuf {
  std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    PathBuf::from,
  )
}

/// A facade with both media lanes off: the text part opened, no function
/// loaded yet.
fn text_only() -> Embedder {
  Embedder::load(staged_dir(), &EmbedderOptions::new()).expect("the text part opens")
}

/// The refusal's message names the lane, exactly as ruled.
#[test]
fn a_disabled_lane_names_itself() {
  assert_eq!(
    Error::PartDisabled(Part::Audio).to_string(),
    "the audio lane is off: the options enable no audio part, so the facade loaded none"
  );
  assert_eq!(
    Error::PartDisabled(Part::Vision).to_string(),
    "the vision lane is off: the options enable no vision part, so the facade loaded none"
  );
}

/// A call into the audio lane while it is off is refused by name, and no
/// model work happened: no audio part was loaded on the side, and the text
/// part loaded no function.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_call_into_the_audio_lane_is_refused_before_any_work() {
  let mut embedder = text_only();
  let clip = vec![0.25f32; 32_000];
  let refused = embedder.embed_audio(&clip);
  assert!(
    matches!(refused, Err(Error::PartDisabled(Part::Audio))),
    "expected PartDisabled(Audio), got {refused:?}"
  );
  assert!(embedder.parts.audio.is_none(), "an audio part was loaded");
  let loaded = embedder.parts.text.loaded();
  assert!(loaded.is_empty(), "the text part loaded {loaded}");
}

/// A call into the vision lane while it is off is refused by name, through
/// both image doors, and no model work happened: no vision part was loaded on
/// the side, the text part loaded no function, and `out` is unchanged.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_call_into_the_vision_lane_is_refused_before_any_work() {
  let mut embedder = text_only();
  let pixels = [128u8; 48 * 48 * 3];
  let image = Rgb8Image::new(&pixels, 48, 48).expect("an image");
  let refused = embedder.embed_image(image, None);
  assert!(
    matches!(refused, Err(Error::PartDisabled(Part::Vision))),
    "expected PartDisabled(Vision) from embed_image, got {refused:?}"
  );
  let mut out = Vec::new();
  let refused = embedder.embed_images(&[image, image], None, &mut out);
  assert!(
    matches!(refused, Err(Error::PartDisabled(Part::Vision))),
    "expected PartDisabled(Vision) from embed_images, got {refused:?}"
  );
  assert!(
    out.is_empty(),
    "embed_images appended {} vectors",
    out.len()
  );
  assert!(embedder.parts.vision.is_none(), "a vision part was loaded");
  let loaded = embedder.parts.text.loaded();
  assert!(loaded.is_empty(), "the text part loaded {loaded}");
}

/// `derivation_terms` answers one part at a time: the text part's own terms,
/// and `None` for each lane that is off.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_terms_are_one_parts_and_none_for_a_lane_that_is_off() {
  let embedder = text_only();
  let text = embedder
    .derivation_terms(Part::Text)
    .expect("the text part is always on")
    .to_string();
  assert_eq!(text, embedder.parts.text.derivation_terms().to_string());
  assert!(text.starts_with("embeddinggemma2/text("), "{text}");
  assert!(embedder.derivation_terms(Part::Audio).is_none());
  assert!(embedder.derivation_terms(Part::Vision).is_none());
}
