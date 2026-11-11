use super::*;
use crate::embeddings::embeddinggemma2::tokenizer::image::fixture;

/// §1.4's framing of the upstream tokenizer: the eight prompts in
/// [`Prompt::NAMED`] order, then `Bare`.
const UPSTREAM_FRAMING: [usize; 9] = [8, 9, 9, 9, 9, 8, 8, 9, 2];

/// The mini fixture's framing, from Hugging Face's `tokenizers` over
/// `mini_tokenizer.json`: each prompt without its trailing space, encoded with
/// its special tokens. Mostly byte fallback, so far from the upstream table.
const MINI_FRAMING: [usize; 9] = [21, 30, 34, 30, 31, 31, 27, 36, 2];

/// F1's hermetic twin (G13): the framing is computed from the loaded image,
/// not copied from a table, so the mini image gives its own counts, Hugging
/// Face's, which differ from §1.4's.
#[test]
fn the_framing_is_computed_from_the_loaded_image() {
  let (image, _file) = fixture::mini();
  let table = framing_table(&image, &mut TokenizerScratch::new()).expect("encodes");
  assert_eq!(table, MINI_FRAMING);
  assert_ne!(table, UPSTREAM_FRAMING);
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

/// A committed golden, parsed.
fn golden(name: &str) -> serde_json::Value {
  let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests/embeddinggemma2/fixtures/goldens")
    .join(name);
  serde_json::from_slice(&std::fs::read(&path).expect("the golden reads")).expect("parses")
}

fn sha256_of(values: &[f16]) -> String {
  use sha2::{Digest, Sha256};
  let bytes: Vec<u8> = values
    .iter()
    .flat_map(|v| v.to_bits().to_le_bytes())
    .collect();
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// M3's inputs half (T5 over the real artifacts): for every R2 text item whose
/// ids the kit and FluidUse agree on, the `embed_S` inputs the kit builds hash
/// to what the Swift probe captured FluidUse handing Core ML, byte for byte.
/// FluidUse's tokenizer splits no special literal, so the kit encodes under
/// `Split`; `bom-twin` is the one item where FluidUse's ids part from Hugging
/// Face's, and the kit follows Hugging Face.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_inputs_are_the_swift_probes_bytes() {
  let artifacts = TextArtifacts::open(staged_dir()).expect("the staged text part opens");
  let text = golden("text.json");
  let probe = golden("swift_probe.json");
  let joint: std::collections::BTreeMap<&str, String> = text["items"]
    .as_array()
    .expect("items")
    .iter()
    .map(|item| {
      (
        item["id"].as_str().expect("id"),
        format!(
          "{}{}",
          item["prefix"].as_str().expect("prefix"),
          item["text"].as_str().expect("text")
        ),
      )
    })
    .collect();
  let mut scratch = TokenizerScratch::new();
  let mut ids = Vec::new();
  let (mut compared, mut differing_ids) = (0, Vec::new());
  for entry in probe["text"].as_array().expect("text") {
    let id = entry["id"].as_str().expect("id");
    artifacts
      .token_ids(
        &joint[id],
        Prompt::Bare,
        SplitSpecialTokens::Split,
        &mut scratch,
        &mut ids,
      )
      .expect("encodes");
    let fluiduse: Vec<u32> = entry["ids"]
      .as_array()
      .expect("ids")
      .iter()
      .map(|v| v.as_u64().expect("an id") as u32)
      .collect();
    if ids.iter().map(|t| t.get()).collect::<Vec<_>>() != fluiduse {
      differing_ids.push(id.to_owned());
      continue;
    }
    let function = unpacked(ids.len());
    let mut inputs = vec![f16::ZERO; function.tokens() * HIDDEN_SIZE];
    let mut mask = vec![f16::ZERO; function.tokens()];
    fill_unpacked(
      &artifacts.shared.table,
      &Sequence::Text(&ids),
      &mut inputs,
      &mut mask,
    )
    .expect("fits");
    let capture = &entry["captures"][0];
    assert_eq!(
      capture["function"].as_str(),
      Some(function.as_str()),
      "{id}"
    );
    for input in capture["inputs"].as_array().expect("inputs") {
      let got = match input["name"].as_str() {
        Some("inputs_embeds") => sha256_of(&inputs),
        Some("attention_mask") => sha256_of(&mask),
        other => panic!("{id}: an unexpected input {other:?}"),
      };
      assert_eq!(
        Some(got.as_str()),
        input["sha256"].as_str(),
        "{id}: {}",
        input["name"]
      );
    }
    compared += 1;
  }
  eprintln!("M3 inputs: {compared} items byte-identical; ids differ on {differing_ids:?}");
  assert_eq!(
    differing_ids,
    ["bom-twin"],
    "FluidUse parts from Hugging Face on bom-twin alone"
  );
  assert_eq!(compared, 75);
}

/// §11, the gather against its call: laying out `embed_512`'s 512 rows from
/// the mapped table (warm pages), against one warm `embed_512` call under the
/// default placement. Printed; §2.9 builds a hand-written kernel only if the
/// safe path costs more than 2% of the call.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_gather_is_measured_against_its_call() {
  use std::time::Instant;

  let artifacts = TextArtifacts::open(staged_dir()).expect("opens");
  let ids: Vec<TokenId> = (0..512u32)
    .map(|i| TokenId::from_vocab(1_000 + i * 397))
    .collect();
  let mut inputs = vec![f16::ZERO; 512 * HIDDEN_SIZE];
  let mut mask = vec![f16::ZERO; 512];
  let layout = |inputs: &mut [f16], mask: &mut [f16]| {
    fill_unpacked(&artifacts.shared.table, &Sequence::Text(&ids), inputs, mask).expect("fits");
  };
  layout(&mut inputs, &mut mask);
  const RUNS: u32 = 200;
  let started = Instant::now();
  for _ in 0..RUNS {
    layout(&mut inputs, &mut mask);
  }
  let gather = started.elapsed() / RUNS;
  let text: String = (0..500)
    .map(|i| ["river", "boat", "light", "stone", "wind"][i % 5])
    .collect::<Vec<_>>()
    .join(" ");
  let mut embedder = TextEmbedder::new(
    &artifacts,
    &TextOptions::new().with_batching(Batching::Single),
  )
  .expect("an instance");
  embedder.embed(&text, Prompt::Bare).expect("warms");
  const CALLS: u32 = 10;
  let started = Instant::now();
  for _ in 0..CALLS {
    embedder.embed(&text, Prompt::Bare).expect("embeds");
  }
  let call = started.elapsed() / CALLS;
  eprintln!(
    "gather: {gather:.1?} per 512-row layout against {call:.1?} per embed_512 call ({:.2}%)",
    100.0 * gather.as_secs_f64() / call.as_secs_f64()
  );
}

/// M9's mapping half for the text part: two instances over one
/// `TextArtifacts` read one mapping of the table and the image, so a second
/// instance adds its models and no second copy of either.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn two_text_instances_read_one_mapping() {
  let artifacts = TextArtifacts::open(staged_dir()).expect("opens");
  let first = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let second = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("a second");
  assert!(first.artifacts.same_mappings(&second.artifacts));
  assert!(first.artifacts.same_mappings(&artifacts));
}

/// Codex round 1, findings 4 and 5: the batch road answers an allocation it
/// cannot make by name, before any model work. Each refusal is told apart by
/// its site and the bytes asked for: under `Packed`, the members' lengths
/// (the first pack-layout reservation) and the result slots (the third
/// batch-output one, after the sequences and the output); on the media door,
/// its row list (the first batch-output one). No text function loads.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_batch_road_refuses_an_allocation_by_name() {
  use core::num::NonZeroU16;

  use half::f16;

  use super::super::super::{
    alloc::inject, config::HIDDEN_SIZE, error::AllocationSite, vision::VisionBudget,
  };
  use crate::{DataType, MultiArray};

  let artifacts = TextArtifacts::open(staged_dir()).expect("the staged text part opens");
  let refusal = |result: Result<()>| match result {
    Err(Error::Allocation(refusal)) => Some((refusal.site(), refusal.bytes())),
    _ => None,
  };
  let texts = ["a river", "a boat", "a stone"];
  for (site, skip, bytes) in [
    (
      AllocationSite::PackLayout,
      0,
      texts.len() * size_of::<usize>(),
    ),
    (
      AllocationSite::BatchOutput,
      2,
      texts.len() * size_of::<Option<Embedding>>(),
    ),
  ] {
    let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
    let mut out = Vec::new();
    let refused = {
      let _guard = inject::refuse(site, skip);
      text.embed_batch(&texts, Prompt::SearchQuery, &mut out)
    };
    assert_eq!(refusal(refused), Some((site, bytes)), "{site} after {skip}");
    assert!(out.is_empty());
    let loaded = text.loaded();
    assert!(loaded.is_empty(), "the text part loaded {loaded}");
  }
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let tokens = ImageTokens::new(
    MultiArray::zeros(&[1, 140, HIDDEN_SIZE], DataType::F16).expect("an array"),
    VisionBudget::Tokens140,
    NonZeroU16::new(140).expect("non-zero"),
  );
  let mut out = Vec::new();
  let refused = {
    let _guard = inject::refuse(AllocationSite::BatchOutput, 0);
    text.embed_image_tokens_batch_with(&[&tokens], Batching::Packed, &mut out)
  };
  assert_eq!(
    refusal(refused),
    Some((AllocationSite::BatchOutput, size_of::<Vec<f16>>()))
  );
  let loaded = text.loaded();
  assert!(loaded.is_empty(), "the text part loaded {loaded}");
}
