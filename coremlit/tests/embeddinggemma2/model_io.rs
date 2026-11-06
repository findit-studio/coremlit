//! Model I/O over the staged EmbeddingGemma 2 tree.
//!
//! C1's law, carried here from knife 0 now that a shard stages the tree: the
//! ten functions of the compiled bundles each load with their own
//! description, and `Model::load` without a name gives the declared defaults
//! (`embed_128` for text, `vision_70` for vision). The audio bundle has one
//! function and loads by its default.
//!
//! The bundles are the ones `compile_artifacts` names: the gate calls it
//! first, idempotently, so it finds the shard's compile step done and a local
//! run compiles a freshly staged tree. The tree is
//! `EMBEDDINGGEMMA2_TEST_MODELS`, else the workspace's `Models/embeddinggemma2`;
//! with neither present the gate says so and passes.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops.
#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::path::{Path, PathBuf};

use coremlit::{
  ComputeUnits, DataType, FeatureInfo, FunctionName, Model,
  embeddings::embeddinggemma2::compile_artifacts,
};

/// The staged tree, or `None` (with the reason printed) when there is none.
fn staged() -> Option<PathBuf> {
  let dir = std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || workspace_root::models_root().join("embeddinggemma2"),
    PathBuf::from,
  );
  if dir.is_dir() {
    Some(dir)
  } else {
    eprintln!(
      "SKIP: no staged EmbeddingGemma 2 tree at {} (set EMBEDDINGGEMMA2_TEST_MODELS)",
      dir.display()
    );
    None
  }
}

/// The compiled bundle `compile_artifacts` names for a package `stem`.
fn bundle(dir: &Path, names: &[String], stem: &str) -> PathBuf {
  let prefix = format!("{stem}.");
  let name = names
    .iter()
    .find(|name| name.starts_with(&prefix) && name.ends_with(".mlmodelc"))
    .unwrap_or_else(|| panic!("compile_artifacts names no {stem} bundle among {names:?}"));
  dir.join(name)
}

type Io = Vec<(String, Vec<usize>, Option<DataType>)>;

fn io(model: &Model) -> (Io, Io) {
  let list = |features: &[FeatureInfo]| {
    features
      .iter()
      .map(|f| (f.name().to_owned(), f.shape().to_vec(), f.data_type()))
      .collect::<Io>()
  };
  (
    list(model.description().inputs()),
    list(model.description().outputs()),
  )
}

fn f(name: &str, shape: &[usize], dtype: DataType) -> (String, Vec<usize>, Option<DataType>) {
  (name.to_owned(), shape.to_vec(), Some(dtype))
}

fn embed(s: usize) -> (Io, Io) {
  (
    vec![
      f("attention_mask", &[1, s], DataType::F16),
      f("inputs_embeds", &[1, s, 512], DataType::F16),
    ],
    vec![f("embedding", &[1, 768], DataType::F32)],
  )
}

fn pack() -> (Io, Io) {
  (
    vec![
      f("attention_bias", &[1, 1, 256, 256], DataType::F16),
      f("inputs_embeds", &[1, 256, 512], DataType::F16),
      f("pool", &[8, 256], DataType::F16),
      f("positions", &[256, 1], DataType::F16),
    ],
    vec![f("embedding", &[8, 768], DataType::F32)],
  )
}

fn vision(t: usize) -> (Io, Io) {
  let n = 9 * t;
  (
    vec![
      f("patches", &[1, n, 768], DataType::F16),
      f("pool", &[t, n], DataType::F16),
      f("position_embeddings", &[1, n, 768], DataType::F16),
      f("positions", &[1, n, 2], DataType::F16),
      f("valid", &[1, n], DataType::F16),
    ],
    vec![f("image_tokens", &[1, t, 512], DataType::F16)],
  )
}

#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_ten_functions_load_with_their_own_descriptions_and_the_defaults_are_declared() {
  let Some(dir) = staged() else {
    return;
  };
  let report = compile_artifacts(&dir).expect("the staged tree compiles");
  let names: Vec<String> = report
    .compiled()
    .iter()
    .chain(report.kept())
    .cloned()
    .collect();
  let text = bundle(&dir, &names, "EmbeddingGemma2Text");
  let vision_bundle = bundle(&dir, &names, "EmbeddingGemma2Vision");
  let audio_bundle = bundle(&dir, &names, "EmbeddingGemma2Audio");

  let mut cases: Vec<(&Path, String, (Io, Io))> = Vec::new();
  for s in [32, 48, 64, 128, 256, 512] {
    cases.push((text.as_path(), format!("embed_{s}"), embed(s)));
  }
  cases.push((text.as_path(), "pack_256".to_owned(), pack()));
  for t in [70, 140, 280] {
    cases.push((vision_bundle.as_path(), format!("vision_{t}"), vision(t)));
  }
  assert_eq!(cases.len(), 10);

  let mut failures = Vec::new();
  for (path, name, expected) in &cases {
    let model = Model::load_function(
      path,
      FunctionName::new(name).expect("a function name"),
      ComputeUnits::CpuOnly,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    let actual = io(&model);
    if &actual == expected {
      eprintln!("{name}: its own description");
    } else {
      failures.push(format!("{name}: got {actual:?}"));
    }
  }

  let text_default = io(&Model::load(&text, ComputeUnits::CpuOnly).expect("text loads"));
  if text_default == embed(128) {
    eprintln!("text, no name: embed_128");
  } else {
    failures.push(format!("text without a name: got {text_default:?}"));
  }
  let vision_default =
    io(&Model::load(&vision_bundle, ComputeUnits::CpuOnly).expect("vision loads"));
  if vision_default == vision(70) {
    eprintln!("vision, no name: vision_70");
  } else {
    failures.push(format!("vision without a name: got {vision_default:?}"));
  }

  let audio = io(&Model::load(&audio_bundle, ComputeUnits::CpuOnly).expect("audio loads"));
  if audio
    == (
      vec![
        f("frame_mask", &[1, 1000], DataType::F16),
        f("waveform", &[1, 160_160], DataType::F32),
      ],
      vec![f("audio_tokens", &[1, 250, 512], DataType::F16)],
    )
  {
    eprintln!("audio, no name: its one function");
  } else {
    failures.push(format!("the single-function audio bundle: got {audio:?}"));
  }

  assert!(
    failures.is_empty(),
    "{} of 13 wrong:\n{}",
    failures.len(),
    failures.join("\n")
  );
}
