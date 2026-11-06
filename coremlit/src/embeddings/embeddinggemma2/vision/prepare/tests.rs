use super::*;
use crate::embeddings::embeddinggemma2::table::fixture;

/// The goldens' image file, parsed.
fn goldens() -> serde_json::Value {
  let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests/embeddinggemma2/fixtures/goldens/image.json");
  serde_json::from_slice(&std::fs::read(path).expect("the golden reads")).expect("parses")
}

fn budget_of(tokens: u64) -> VisionBudget {
  VisionBudget::from_tokens(tokens as usize).expect("a budget")
}

/// The synthetic images' rule, as `_corpus.pattern_pixel` states it.
fn pattern(width: usize, height: usize) -> Vec<u8> {
  let mut pixels = Vec::with_capacity(width * height * 3);
  for y in 0..height {
    for x in 0..width {
      for c in 0..3 {
        pixels.push(((x * 7 + y * 13 + c * 101 + ((x / 5) ^ (y / 3)) * 29) % 256) as u8);
      }
    }
  }
  pixels
}

fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

fn halves_sha(values: &[f16]) -> String {
  sha256_hex(
    &values
      .iter()
      .flat_map(|v| v.to_bits().to_le_bytes())
      .collect::<Vec<u8>>(),
  )
}

/// V1: the target size is Hugging Face's on §5.3's table, FluidUse's five
/// cases, and the cases where one side rounds to zero.
#[test]
fn the_target_sizes_are_hugging_faces() {
  use VisionBudget::{Tokens70, Tokens140, Tokens280};
  // (width, height, budget) -> (width, height)
  let cases = [
    (1920, 1080, Tokens70, (528, 288)),
    (1920, 1080, Tokens140, (720, 384)),
    (1920, 1080, Tokens280, (1056, 576)),
    (1280, 720, Tokens140, (720, 384)),
    (3840, 2160, Tokens140, (720, 384)),
    (1080, 1920, Tokens140, (384, 720)),
    (1024, 1024, Tokens70, (384, 384)),
    (1024, 1024, Tokens140, (528, 528)),
    (1024, 1024, Tokens280, (768, 768)),
    (640, 480, Tokens140, (624, 480)),
    (500, 334, Tokens280, (960, 624)),
    (4000, 100, Tokens70, (2496, 48)),
    // FluidUse's five.
    (334, 500, Tokens70, (288, 480)),
    (64, 64, Tokens280, (768, 768)),
    // One side rounds to zero: 48 for it, the other the ratio's floor, capped.
    (100_000, 10, Tokens70, (3360, 48)),
    (10, 100_000, Tokens280, (48, 13_440)),
    (200, 3, Tokens140, (4608, 48)),
  ];
  for (width, height, budget, (want_width, want_height)) in cases {
    let (got_height, got_width) = target_size(height, width, budget).expect("a size");
    assert_eq!(
      (got_width, got_height),
      (want_width, want_height),
      "{width}x{height} at {budget}"
    );
    assert!((got_width / 16) * (got_height / 16) <= budget.patches());
    assert_eq!(got_width % 48, 0);
    assert_eq!(got_height % 48, 0);
  }
}

/// V2: the rescale table is `f16(v / 255)` for all 256 values, and the three
/// references' forms agree with it everywhere.
#[test]
fn the_rescale_table_is_f16_of_v_over_255() {
  for v in 0..=255u8 {
    let exact = f16::from_f64(f64::from(v) / 255.0);
    assert_eq!(RESCALE[usize::from(v)].to_bits(), exact.to_bits(), "{v}");
    let swift = f16::from_f32(f32::from(v) / 255.0);
    let hf = f16::from_f32((f64::from(v) * (1.0 / 255.0)) as f32);
    let f32_factor = f16::from_f32(f32::from(v) * (1.0f32 / 255.0));
    for (form, value) in [("swift", swift), ("hf", hf), ("f32 factor", f32_factor)] {
      assert_eq!(value.to_bits(), exact.to_bits(), "{v}: {form}");
    }
  }
}

/// The planes of the synthetic image at `budget`'s fixed point, over `table`.
fn synthetic_planes(
  budget: &serde_json::Value,
  table: &PositionTable,
) -> (Vec<Vec<f16>>, usize, usize) {
  let tokens = budget["budget"].as_u64().expect("budget");
  let vision = budget_of(tokens);
  let (height, width) = (
    budget["height"].as_u64().expect("height") as usize,
    budget["width"].as_u64().expect("width") as usize,
  );
  assert_eq!(
    target_size(height, width, vision).expect("a size"),
    (height, width)
  );
  let n = vision.patches();
  let mut planes = vec![
    vec![f16::ZERO; n * PATCH_VALUES],
    vec![f16::ZERO; n * 2],
    vec![f16::ZERO; n],
    vec![f16::ZERO; vision.tokens() * n],
    vec![f16::ZERO; n * POSITION_WIDTH],
  ];
  let [patches, positions, valid, pool, position_embeddings] = &mut planes[..] else {
    unreachable!()
  };
  fill(
    &pattern(width, height),
    height / 16,
    width / 16,
    table,
    Planes {
      patches,
      positions,
      valid,
      pool,
      position_embeddings,
    },
  );
  (planes, height / 16, width / 16)
}

/// V3, its table-free half: at each budget's fixed point, the synthetic
/// image's `patches`, `positions`, `valid` and `pool` are the golden bytes,
/// and its grid is the golden grid. (`position_embeddings` reads the staged
/// tables: the model-gated half checks it.)
#[test]
fn the_planes_are_the_goldens_bytes() {
  let (_file, table) = fixture::position_table();
  let golden = goldens();
  let budgets = golden["synthetic"]["budgets"].as_array().expect("budgets");
  assert_eq!(budgets.len(), 3);
  for budget in budgets {
    let (planes, rows, cols) = synthetic_planes(budget, &table);
    let inputs = &budget["inputs"];
    assert_eq!(rows as u64, inputs["grid"]["rows"].as_u64().expect("rows"));
    assert_eq!(cols as u64, inputs["grid"]["cols"].as_u64().expect("cols"));
    for (name, plane) in ["patches", "positions", "valid", "pool"]
      .iter()
      .zip(&planes)
    {
      assert_eq!(
        Some(halves_sha(plane).as_str()),
        inputs[*name]["sha256"].as_str(),
        "{name} at {}",
        budget["budget"]
      );
    }
  }
}

/// A committed corpus PNG, RGB8.
fn png_rgb8(path: &std::path::Path) -> (Vec<u8>, usize, usize) {
  let file = std::fs::File::open(path).expect("the image opens");
  let mut reader = png::Decoder::new(std::io::BufReader::new(file))
    .read_info()
    .expect("a png");
  let mut buf = vec![0u8; reader.output_buffer_size().expect("a size")];
  let info = reader.next_frame(&mut buf).expect("decodes");
  let (width, height) = (info.width as usize, info.height as usize);
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
  (rgb, width, height)
}

/// Standard base64, as the goldens carry the tiles.
fn base64(text: &str) -> Vec<u8> {
  let value = |c: u8| match c {
    b'A'..=b'Z' => c - b'A',
    b'a'..=b'z' => c - b'a' + 26,
    b'0'..=b'9' => c - b'0' + 52,
    b'+' => 62,
    b'/' => 63,
    _ => panic!("not base64: {c}"),
  };
  let bytes: Vec<u8> = text.bytes().filter(|&c| c != b'=').map(value).collect();
  bytes
    .chunks(4)
    .flat_map(|chunk| {
      let n = chunk.iter().fold(0u32, |n, &v| n << 6 | u32::from(v)) << (6 * (4 - chunk.len()));
      let [_, a, b, c] = n.to_be_bytes();
      [a, b, c].into_iter().take(chunk.len() - 1)
    })
    .collect()
}

/// V4's band: the worst channel difference against Hugging Face's
/// torchvision `uint8` antialias resize, pinned as measured on the goldens'
/// host (Apple M1 Max, torchvision's float path).
const RESIZE_LSB: u8 = 1;

/// V4: pixon's `CatmullRom` against Hugging Face's resized pixels, on the
/// goldens' 32 × 32 tiles of the seven images at every budget.
#[test]
fn the_resize_holds_hugging_faces_within_one_lsb() {
  let golden = goldens();
  let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
  let mut worst = 0u8;
  let mut compared = 0usize;
  for entry in golden["images"].as_array().expect("images") {
    let id = entry["id"].as_str().expect("id");
    let (pixels, width, height) = match entry["source"]["file"].as_str() {
      Some(file) => png_rgb8(&tests.join(file)),
      None => {
        let (w, h) = (
          entry["source"]["width"].as_u64().expect("width") as usize,
          entry["source"]["height"].as_u64().expect("height") as usize,
        );
        (pattern(w, h), w, h)
      }
    };
    let image = Rgb8Image::new(&pixels, width, height).expect("an image");
    for (tokens, budget) in entry["budgets"].as_object().expect("budgets") {
      let vision = budget_of(tokens.parse().expect("a count"));
      let (target_height, target_width) = target_size(height, width, vision).expect("a size");
      let resized_size = budget["resized"].as_array().expect("resized");
      // The goldens record the resized size as `[height, width]`.
      assert_eq!(
        (target_height as u64, target_width as u64),
        (
          resized_size[0].as_u64().expect("h"),
          resized_size[1].as_u64().expect("w")
        ),
        "{id} at {tokens}"
      );
      let mut ours = Vec::new();
      let ours: &[u8] = if (target_height, target_width) == (height, width) {
        &pixels
      } else {
        resize(image, target_height, target_width, &mut ours).expect("resizes");
        &ours
      };
      for tile in budget["tiles"].as_array().expect("tiles") {
        let (y0, x0) = (
          tile["y"].as_u64().expect("y") as usize,
          tile["x"].as_u64().expect("x") as usize,
        );
        let (h, w) = (
          tile["h"].as_u64().expect("h") as usize,
          tile["w"].as_u64().expect("w") as usize,
        );
        let theirs = base64(tile["rgb_base64"].as_str().expect("tile"));
        assert_eq!(theirs.len(), h * w * 3);
        for y in 0..h {
          for x in 0..w {
            for c in 0..3 {
              let a = ours[((y0 + y) * target_width + x0 + x) * 3 + c];
              let b = theirs[(y * w + x) * 3 + c];
              worst = worst.max(a.abs_diff(b));
              compared += 1;
            }
          }
        }
      }
    }
  }
  eprintln!("V4: {compared} channel values, the worst {worst} LSB from Hugging Face's");
  assert!(compared > 0);
  assert!(worst <= RESIZE_LSB, "the worst difference is {worst} LSB");
}

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

/// V3, the staged half: `position_embeddings` over the staged tables is the
/// golden bytes at each budget, as are the four table-free planes.
/// `position_embeddings` is compared first: it is the plane only this half
/// can check.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_position_embeddings_are_the_goldens_bytes() {
  let artifacts = VisionArtifacts::open(staged_dir()).expect("the staged vision part opens");
  let golden = goldens();
  for budget in golden["synthetic"]["budgets"].as_array().expect("budgets") {
    let (planes, _, _) = synthetic_planes(budget, &artifacts.shared.table);
    for (name, index) in [
      ("position_embeddings", 4),
      ("patches", 0),
      ("positions", 1),
      ("valid", 2),
      ("pool", 3),
    ] {
      assert_eq!(
        Some(halves_sha(&planes[index]).as_str()),
        budget["inputs"][name]["sha256"].as_str(),
        "{name} at {}",
        budget["budget"]
      );
    }
  }
}

/// A preparation that fails leaves no preparation behind: the budget is
/// `None` and no patch is real, so an encoder refuses `out` rather than
/// encoding the planes of the image before.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_failed_preparation_leaves_nothing_to_encode() {
  let artifacts = VisionArtifacts::open(staged_dir()).expect("the staged vision part opens");
  let options = ImageOptions::new();
  let pixels = pattern(624, 480);
  let image = Rgb8Image::new(&pixels, 624, 480).expect("an image");
  let mut out = PreparedImage::new();
  artifacts
    .prepare_into(image, None, &options, &mut out)
    .expect("prepares at the default budget");
  assert_eq!(out.budget(), Some(VisionBudget::Tokens140));
  assert!(out.patches() > 0);
  let refused = artifacts.prepare_into(image, Some(VisionBudget::Tokens280), &options, &mut out);
  assert!(
    matches!(refused, Err(Error::BudgetNotLoaded(_))),
    "{refused:?}"
  );
  assert_eq!(out.budget(), None);
  assert_eq!(out.patches(), 0);
}

/// G14's steady state: once a [`PreparedImage`] holds a budget's planes,
/// refilling it from an image already at that budget's geometry allocates
/// nothing of its own: exactly what borrowing its five planes costs on this
/// build (objc2's message check, in a debug build), no plane and no pixel
/// buffer. A resize allocates pixon's resample plan, printed.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_refill_at_the_budgets_geometry_allocates_nothing() {
  use crate::tests::alloc_probe::measure;

  let artifacts = VisionArtifacts::open(staged_dir()).expect("the staged vision part opens");
  let options =
    ImageOptions::new().with_budgets(crate::embeddings::embeddinggemma2::VisionBudgets::all());
  let mut prepared = PreparedImage::new();
  for (width, height, budget) in [
    (432, 336, VisionBudget::Tokens70),
    (624, 480, VisionBudget::Tokens140),
    (912, 672, VisionBudget::Tokens280),
  ] {
    let pixels = pattern(width, height);
    let image = Rgb8Image::new(&pixels, width, height).expect("an image");
    artifacts
      .prepare_into(image, Some(budget), &options, &mut prepared)
      .expect("prepares");
    let (_, borrowing) = measure(|| {
      for plane in [
        &mut prepared.patches,
        &mut prepared.positions,
        &mut prepared.valid,
        &mut prepared.pool,
        &mut prepared.position_embeddings,
      ] {
        plane
          .as_mut()
          .expect("shaped")
          .as_slice_mut::<f16>()
          .expect("a plane");
      }
    });
    let (result, allocated) =
      measure(|| artifacts.prepare_into(image, Some(budget), &options, &mut prepared));
    result.expect("prepares");
    assert_eq!(
      allocated.total, borrowing.total,
      "{budget}: a refill allocated {allocated:?}, borrowing the planes {borrowing:?}"
    );
    let bigger = pattern(width * 2, height * 2);
    let resized = Rgb8Image::new(&bigger, width * 2, height * 2).expect("an image");
    artifacts
      .prepare_into(resized, Some(budget), &options, &mut prepared)
      .expect("prepares");
    let (result, allocated) =
      measure(|| artifacts.prepare_into(resized, Some(budget), &options, &mut prepared));
    result.expect("prepares");
    eprintln!(
      "G14 {budget}: a refill allocates nothing of its own (borrowing the planes: {borrowing:?}); a resizing refill allocates {allocated:?}"
    );
  }
}
