//! The facade end to end over the staged EmbeddingGemma 2 tree: every lane
//! on, one space. M7's cross-modal ranking, `Embedder::prewarm` over every
//! enabled part (G14), and each part's derivation terms.
//!
//! Every gate here is `#[ignore]`d and model-gated. The tree is
//! `EMBEDDINGGEMMA2_TEST_MODELS`, else the workspace's `Models/embeddinggemma2`;
//! with neither present a gate says so and passes. Each gate calls
//! `compile_artifacts` first, idempotently.

#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::path::PathBuf;

use coremlit::embeddings::embeddinggemma2::{
  AggregatePolicyKind, AudioOptions, Embedder, EmbedderOptions, Embedding, ImageOptions, Part,
  Prompt, Rgb8Image, TextFunctions, VisionBudget, VisionBudgets, compile_artifacts,
  error::Function,
};
use serde_json::Value;
use windit::windowed::Windowed;

/// The staged tree, compiled, or `None` (with the reason printed).
fn staged() -> Option<PathBuf> {
  let dir = std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || workspace_root::models_root().join("embeddinggemma2"),
    PathBuf::from,
  );
  if !dir.is_dir() {
    eprintln!(
      "SKIP: no staged EmbeddingGemma 2 tree at {} (set EMBEDDINGGEMMA2_TEST_MODELS)",
      dir.display()
    );
    return None;
  }
  compile_artifacts(&dir).expect("the staged tree compiles");
  Some(dir)
}

fn golden(name: &str) -> Value {
  let path = workspace_root::workspace_root()
    .join("coremlit/tests/embeddinggemma2/fixtures/goldens")
    .join(name);
  serde_json::from_slice(&std::fs::read(&path).expect("the golden reads")).expect("parses")
}

/// Every lane on, under its defaults.
fn every_lane() -> EmbedderOptions {
  EmbedderOptions::new()
    .with_audio(AudioOptions::new())
    .with_vision(ImageOptions::new())
}

/// The synthetic images' rule, `_corpus.pattern_pixel`.
fn pattern(width: usize, height: usize) -> Vec<u8> {
  (0..height)
    .flat_map(|y| {
      (0..width).flat_map(move |x| {
        (0..3).map(move |c| ((x * 7 + y * 13 + c * 101 + ((x / 5) ^ (y / 3)) * 29) % 256) as u8)
      })
    })
    .collect()
}

/// A golden image's pixels: a committed PNG, or the pattern.
fn pixels(entry: &Value) -> (Vec<u8>, usize, usize) {
  match entry["source"]["file"].as_str() {
    Some(file) => {
      let path = workspace_root::workspace_root()
        .join("coremlit/tests")
        .join(file);
      let reader = std::fs::File::open(&path).expect("the image opens");
      let mut reader = png::Decoder::new(std::io::BufReader::new(reader))
        .read_info()
        .expect("a png");
      let mut buf = vec![0u8; reader.output_buffer_size().expect("a size")];
      let info = reader.next_frame(&mut buf).expect("decodes");
      let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
          .as_chunks::<4>()
          .0
          .iter()
          .flat_map(|p| [p[0], p[1], p[2]])
          .collect(),
        other => panic!("{other:?}"),
      };
      (rgb, info.width as usize, info.height as usize)
    }
    None => {
      let (w, h) = (
        entry["source"]["width"].as_u64().expect("width") as usize,
        entry["source"]["height"].as_u64().expect("height") as usize,
      );
      (pattern(w, h), w, h)
    }
  }
}

/// A committed 16 kHz mono clip as the goldens read it: s16le over 32,768,
/// cut to the golden's sample count.
fn clip(entry: &Value) -> Vec<f32> {
  let path = workspace_root::workspace_root()
    .join("coremlit/tests")
    .join(entry["file"].as_str().expect("a file"));
  let mut reader = hound::WavReader::open(&path).expect("the clip opens");
  let spec = reader.spec();
  assert_eq!(
    (spec.sample_rate, spec.channels),
    (16_000, 1),
    "{}",
    path.display()
  );
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("a sample")) / 32_768.0)
    .collect();
  let count = entry["samples"].as_u64().expect("samples") as usize;
  samples[..count].to_vec()
}

/// The ids of `scores` from the highest score down.
fn ranked(scores: &[(String, f32)]) -> Vec<&str> {
  let mut order: Vec<&(String, f32)> = scores.iter().collect();
  order.sort_by(|a, b| b.1.total_cmp(&a.1));
  order.iter().map(|(id, _)| id.as_str()).collect()
}

/// M7: one space for every modality. Through the facade, every lane at its
/// defaults: "a photo of a dog" ranks `dog.png` first of the seven images;
/// "a pure sine tone" ranks `sine440` first of the six clips; "a person
/// speaking" ranks one of the three speech clips first. Each clip is one
/// vector: its windows folded through windit under the default fold, as a
/// consumer of `AudioWindowEmbedding` does.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_text_query_ranks_images_and_clips_in_one_space() {
  let Some(dir) = staged() else { return };
  let mut embedder = Embedder::load(&dir, &every_lane()).expect("every lane loads");

  let images = golden("image.json");
  let mut pictures: Vec<(String, Embedding)> = Vec::new();
  for entry in images["images"].as_array().expect("images") {
    let (pixels, width, height) = pixels(entry);
    let image = Rgb8Image::new(&pixels, width, height).expect("an image");
    let vector = embedder.embed_image(image, None).expect("embeds");
    pictures.push((entry["id"].as_str().expect("id").to_owned(), vector));
  }
  assert_eq!(pictures.len(), 7);

  let fold = AggregatePolicyKind::CoverageWeightedMean.into_policy();
  let audio = golden("audio.json");
  let mut clips: Vec<(String, Embedding)> = Vec::new();
  for entry in audio["clips"].as_array().expect("clips") {
    let windows = embedder.embed_audio(&clip(entry)).expect("embeds");
    let windowed: Vec<Windowed<Embedding>> = windows
      .iter()
      .map(|window| Windowed::new(window.embedding().clone(), window.samples()))
      .collect();
    let vector = windit::aggregate::aggregate(fold.as_ref(), &windowed).expect("folds");
    clips.push((entry["id"].as_str().expect("id").to_owned(), vector));
  }
  assert_eq!(clips.len(), 6);

  let mut rank = |query: &str, against: &[(String, Embedding)]| -> Vec<String> {
    let query = embedder
      .embed_text(query, Prompt::SearchQuery)
      .expect("embeds");
    let scores: Vec<(String, f32)> = against
      .iter()
      .map(|(id, vector)| (id.clone(), query.cosine(vector)))
      .collect();
    eprintln!("M7 {scores:?}");
    ranked(&scores).into_iter().map(str::to_owned).collect()
  };

  let dog = rank("a photo of a dog", &pictures);
  eprintln!("M7 \"a photo of a dog\": {dog:?}");
  assert_eq!(dog[0], "dog", "{dog:?}");

  let tone = rank("a pure sine tone", &clips);
  eprintln!("M7 \"a pure sine tone\": {tone:?}");
  assert_eq!(tone[0], "sine440", "{tone:?}");

  let speech = rank("a person speaking", &clips);
  eprintln!("M7 \"a person speaking\": {speech:?}");
  assert!(
    ["jfk", "jfk-3.48s", "ted60"].contains(&speech[0].as_str()),
    "{speech:?}"
  );
}

/// G14: `Embedder::prewarm` warms every enabled part, in part order: the text
/// part's set, the audio bundle (a record that names no function), then each
/// loaded vision budget.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_facade_prewarms_every_enabled_part() {
  let Some(dir) = staged() else { return };
  let options = EmbedderOptions::new()
    .with_audio(AudioOptions::new())
    .with_vision(ImageOptions::new().with_budgets(VisionBudgets::all()));
  let mut embedder = Embedder::load(&dir, &options).expect("every lane loads");
  let report = embedder.prewarm().expect("warms");
  for warmed in report.warmed() {
    eprintln!(
      "G14: {:<12} load {:>9.1?}  first {:>9.1?}",
      warmed
        .function()
        .map_or_else(|| "audio".to_owned(), |f| f.to_string()),
      warmed.load(),
      warmed.first_prediction()
    );
  }
  let warmed: Vec<Option<Function>> = report.warmed().iter().map(|w| w.function()).collect();
  let expected: Vec<Option<Function>> = TextFunctions::all()
    .iter()
    .map(|f| Some(Function::Text(f)))
    .chain([None])
    .chain(VisionBudget::ALL.map(|b| Some(Function::Vision(b))))
    .collect();
  assert_eq!(warmed, expected);
}

/// The ruling on `derivation_terms` with every lane on: each part answers its
/// own terms, as its own instance spells them.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn every_enabled_part_answers_its_own_terms() {
  let Some(dir) = staged() else { return };
  let embedder = Embedder::load(&dir, &every_lane()).expect("every lane loads");
  let terms: Vec<String> = [Part::Text, Part::Audio, Part::Vision]
    .into_iter()
    .map(|part| {
      embedder
        .derivation_terms(part)
        .unwrap_or_else(|| panic!("the {part} lane is on"))
        .to_string()
    })
    .collect();
  let mut parts = embedder.into_parts();
  let text = parts.text_mut().derivation_terms().to_string();
  let audio = parts
    .audio_mut()
    .expect("the audio part")
    .derivation_terms()
    .to_string();
  let vision = parts
    .vision_mut()
    .expect("the vision part")
    .encoder_mut()
    .derivation_terms()
    .to_string();
  assert_eq!(terms, [text, audio, vision]);
  for (term, part) in terms.iter().zip(["text", "audio", "vision"]) {
    assert!(
      term.starts_with(&format!("embeddinggemma2/{part}(")),
      "{term}"
    );
  }
}
