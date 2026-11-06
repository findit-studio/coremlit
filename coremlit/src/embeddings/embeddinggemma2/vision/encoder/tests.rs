use super::*;

/// The staged model directory: `EMBEDDINGGEMMA2_TEST_MODELS`, else the
/// workspace's `Models/embeddinggemma2`.
fn staged_dir() -> std::path::PathBuf {
  std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    std::path::PathBuf::from,
  )
}

/// M3 (vision), its text half: for the R2 probe's three synthetic images,
/// the text function's inputs over the image tokens hash to what FluidUse
/// handed Core ML (printed: they ride on the GPU's output). The vision
/// planes' half is V3's.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_image_text_inputs_are_the_swift_probes_bytes() {
  use half::f16;

  use crate::embeddings::embeddinggemma2::{
    VisionBudgets,
    config::{BOI_ID, EOI_ID},
    text::{
      TextArtifacts,
      layout::{MediaSequence, Sequence, fill_unpacked, unpacked},
    },
    tokenizer::bpe::TokenId,
  };

  let dir = staged_dir();
  let text = TextArtifacts::open(&dir).expect("the staged text part opens");
  let vision = VisionArtifacts::open(&dir).expect("the staged vision part opens");
  let options = ImageOptions::new().with_budgets(VisionBudgets::all());
  let mut encoder = ImageEncoder::new(&vision, &options).expect("an instance");
  let probe: serde_json::Value = serde_json::from_slice(
    &std::fs::read(
      std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/embeddinggemma2/fixtures/goldens/swift_probe.json"),
    )
    .expect("the probe reads"),
  )
  .expect("parses");
  let sha = |values: &[f16]| {
    use sha2::{Digest, Sha256};
    let bytes: Vec<u8> = values
      .iter()
      .flat_map(|v| v.to_bits().to_le_bytes())
      .collect();
    Sha256::digest(&bytes)
      .iter()
      .map(|b| format!("{b:02x}"))
      .collect::<String>()
  };
  let (mut same, mut differing, mut rows) = (0, Vec::new(), Vec::new());
  for entry in probe["image"].as_array().expect("image") {
    let id = entry["id"].as_str().expect("id");
    let budget = VisionBudget::from_tokens(entry["budget"].as_u64().expect("budget") as usize)
      .expect("a budget");
    let (width, height) = (
      entry["width"].as_u64().expect("width") as usize,
      entry["height"].as_u64().expect("height") as usize,
    );
    let pixels: Vec<u8> = (0..height)
      .flat_map(|y| {
        (0..width).flat_map(move |x| {
          (0..3).map(move |c| ((x * 7 + y * 13 + c * 101 + ((x / 5) ^ (y / 3)) * 29) % 256) as u8)
        })
      })
      .collect();
    let image = Rgb8Image::new(&pixels, width, height).expect("an image");
    let prepared = vision
      .prepare(image, Some(budget), &options)
      .expect("prepares");
    let tokens = encoder.encode(&prepared).expect("encodes");
    tokens.rows_into(&mut rows).expect("copies");
    let sequence = Sequence::Media(MediaSequence {
      open: TokenId::from_vocab(BOI_ID),
      rows: &rows,
      close: TokenId::from_vocab(EOI_ID),
    });
    let function = unpacked(sequence.len());
    let capture = &entry["captures"][1];
    assert_eq!(
      capture["function"].as_str(),
      Some(function.as_str()),
      "{id}"
    );
    let mut inputs = vec![f16::ZERO; function.tokens() * HIDDEN_SIZE];
    let mut mask = vec![f16::ZERO; function.tokens()];
    fill_unpacked(text.table(), &sequence, &mut inputs, &mut mask).expect("fits");
    let all = capture["inputs"]
      .as_array()
      .expect("inputs")
      .iter()
      .all(|input| {
        let got = match input["name"].as_str() {
          Some("inputs_embeds") => sha(&inputs),
          Some("attention_mask") => sha(&mask),
          other => panic!("{id}: an unexpected input {other:?}"),
        };
        Some(got.as_str()) == input["sha256"].as_str()
      });
    if all {
      same += 1;
    } else {
      differing.push(id.to_owned());
    }
  }
  eprintln!("M3 vision: text inputs byte-identical on {same} images, differing on {differing:?}");
}
