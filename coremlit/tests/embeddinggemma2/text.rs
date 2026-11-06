//! The text part over the staged EmbeddingGemma 2 tree: framing, contracts,
//! parity against R1 and R2, packing, placement, footprint, run identity and
//! throughput.
//!
//! Every gate here is `#[ignore]`d and model-gated. The tree is
//! `EMBEDDINGGEMMA2_TEST_MODELS`, else the workspace's `Models/embeddinggemma2`;
//! with neither present a gate says so and passes. Each gate calls
//! `compile_artifacts` first, idempotently, so a freshly staged tree compiles
//! and a compiled one is left as it is.
//!
//! The measured bands go through `measured_band`: asserted on the host class
//! in [`CHARACTERIZED_ON`], computed and printed everywhere. The portable
//! floors (M2's 0.998) are asserted on every host.

#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

#[path = "../support/host_class.rs"]
#[allow(dead_code)]
mod host_class;

// `measured_band.rs` reads the host class through its parent, as in the
// siglip suite's `common`.
use host_class::HostClass;

#[path = "../support/measured_band.rs"]
#[allow(dead_code)]
mod measured_band;

use std::{
  path::PathBuf,
  time::{Duration, Instant},
};

use coremlit::{
  ComputeUnits, LoadOptions,
  embeddings::embeddinggemma2::{
    AggregatePolicyKind, ArtifactLabel, Batching, EmaRenormalizedOptions, Embedding,
    LongTextOptions, PACKED_SLOT_INVARIANT, Part, Prompt, SplitSpecialTokens, TextArtifacts,
    TextEmbedder, TextFunction, TextFunctions, TextOptions, TokenizerScratch, compile_artifacts,
    error::Error, run_identity,
  },
};
use measured_band::{BandGate, CharacterizedHost};
use serde_json::Value;

/// The host class the measured bands below were characterized on. There:
/// M3's worst coordinate against R2 is 2.7e-8; M8's worst cosine against
/// `CpuOnly` is 0.999918 (`CpuAndGpu`), 0.999898 (`CpuAndNeuralEngine`) and
/// 0.999911 (`All`); M4 found `pack_256` slot-dependent under every arm; M11
/// found every arm's text output repeating bit for bit.
const CHARACTERIZED_ON: Option<CharacterizedHost> = Some(CharacterizedHost {
  os_build: "26A428",
  os_product_version: "27.0",
  chip: "Apple M1 Max",
  arch: "arm64",
});

/// M8's band: the worst cosine of any placement against `CpuOnly`, measured
/// 0.999898 at worst, pinned with a 4e-4 margin.
const PLACEMENT_LO: f32 = 0.9995;

/// The command that re-characterizes these bands on this host.
const RECHARACTERIZE: &str = "EMBEDDINGGEMMA2_TEST_MODELS=<tree> cargo test -p coremlit \
  --features embeddinggemma2 --test embeddinggemma2_text -- --ignored --nocapture --test-threads=1";

/// M2's portable floor: the worst cosine against R1, at every width.
const R1_FLOOR: f64 = 0.998;

/// The four placements, `CpuOnly` first: M8's reference.
const ARMS: [ComputeUnits; 4] = [
  ComputeUnits::CpuOnly,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuAndNeuralEngine,
  ComputeUnits::All,
];

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

fn options(units: ComputeUnits, batching: Batching) -> TextOptions {
  TextOptions::new()
    .with_coreml(LoadOptions::new(units))
    .with_batching(batching)
}

/// A golden's prompt.
fn prompt_of(spec: &Value) -> Prompt<'_> {
  match spec["kind"].as_str().expect("a prompt kind") {
    "bare" => Prompt::Bare,
    "named" => match spec["key"].as_str().expect("a key") {
      "document" => Prompt::Document(None),
      "search_query" => Prompt::SearchQuery,
      "question_answering" => Prompt::QuestionAnswering,
      "fact_checking" => Prompt::FactChecking,
      "code_retrieval" => Prompt::CodeRetrieval,
      "classification" => Prompt::Classification,
      "clustering" => Prompt::Clustering,
      "sentence_similarity" => Prompt::SentenceSimilarity,
      other => panic!("an unknown named prompt {other}"),
    },
    "document" => Prompt::Document(spec["title"].as_str()),
    "custom" => Prompt::Custom(spec["prefix"].as_str().expect("a prefix")),
    other => panic!("an unknown prompt kind {other}"),
  }
}

fn vector(value: &Value) -> Vec<f64> {
  value
    .as_array()
    .expect("a vector")
    .iter()
    .map(|v| v.as_f64().expect("a number"))
    .collect()
}

/// The cosine of `a` against the first `a.len()` coordinates of `b`,
/// re-normalized.
fn cosine(a: &[f32], b: &[f64]) -> f64 {
  let b = &b[..a.len()];
  let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * y).sum();
  let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
  let nb: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
  dot / (na * nb)
}

fn bits(v: &Embedding) -> Vec<u32> {
  v.as_slice().iter().map(|x| x.to_bits()).collect()
}

/// F1 (G13): the framing `TextArtifacts::open` computes from the staged
/// tokenizer image is §1.4's table for the eight prompts and `Bare`.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_framing_is_the_tokenizers_table() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let expected = [
    (Prompt::Document(None), 8),
    (Prompt::SearchQuery, 9),
    (Prompt::QuestionAnswering, 9),
    (Prompt::FactChecking, 9),
    (Prompt::CodeRetrieval, 9),
    (Prompt::Classification, 8),
    (Prompt::Clustering, 8),
    (Prompt::SentenceSimilarity, 9),
    (Prompt::Bare, 2),
  ];
  for (prompt, ids) in expected {
    let framing = artifacts
      .framing(prompt, SplitSpecialTokens::Recognize)
      .expect("frames");
    assert_eq!(framing.ids(), ids, "{prompt}");
    assert!(!framing.is_conservative(), "{prompt}");
  }
  let titled = artifacts
    .framing(
      Prompt::Document(Some("Q3 earnings call")),
      SplitSpecialTokens::Recognize,
    )
    .expect("frames");
  assert!(titled.ids() > 8 && !titled.is_conservative());
  let custom = artifacts
    .framing(
      Prompt::Custom("task: legal clause retrieval | query:"),
      SplitSpecialTokens::Recognize,
    )
    .expect("frames");
  assert!(custom.is_conservative());
}

/// M1 (text) and M10's local half: every function the default options reach
/// loads under the kit's contract and warms; the report prints each one's
/// load and first prediction. The staged label is the upstream revision's.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn every_text_function_keeps_its_contract_and_warms() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  assert_eq!(artifacts.artifact_label(), ArtifactLabel::FLUIDINFERENCE);
  let mut embedder = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  assert_eq!(embedder.prewarm_set(), TextFunctions::all());
  let report = embedder.prewarm().expect("warms");
  assert_eq!(report.warmed().len(), 7);
  for warmed in report.warmed() {
    eprintln!(
      "M10 (local, CpuAndNeuralEngine): {:?} load {:.1?}, first prediction {:.1?}",
      warmed.function().map(|f| f.as_str()),
      warmed.load(),
      warmed.first_prediction()
    );
  }
}

/// L14 (a), the text wiring: a pin equal to the stamp's label loads, another
/// is refused naming the part and both labels, and no pin loads.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_pinned_text_label_is_honoured() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let matching = TextOptions::new().with_expect_artifacts(ArtifactLabel::FLUIDINFERENCE);
  assert!(TextEmbedder::new(&artifacts, &matching).is_ok());
  let other: ArtifactLabel = "sha256:0123456789ab".parse().expect("a label");
  match TextEmbedder::new(&artifacts, &TextOptions::new().with_expect_artifacts(other)) {
    Err(Error::ArtifactIdentity(refusal)) => {
      assert_eq!(refusal.part(), Part::Text);
      assert_eq!(refusal.expected(), other);
      assert_eq!(refusal.actual(), ArtifactLabel::FLUIDINFERENCE);
    }
    other => panic!("a mismatching pin is refused, got {other:?}"),
  }
  assert!(TextEmbedder::new(&artifacts, &TextOptions::new()).is_ok());
}

/// M2 and M2b: the kit's text vectors against R1 (sentence-transformers fp32)
/// at 768, 512, 256 and 128, under the default options, and the literal-
/// special items under `split_special_tokens = true`; the worst cosine at
/// every width clears the portable floor.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_text_vectors_hold_r1_at_every_width() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let text = golden("text.json");
  for split in [false, true] {
    let options = TextOptions::new().with_split_special_tokens(split);
    let mut embedder = TextEmbedder::new(&artifacts, &options).expect("an instance");
    let mut worst = [1.0f64; 4];
    let mut count = 0;
    for item in text["items"].as_array().expect("items") {
      let reference = if split {
        &item["vector_split"]
      } else {
        &item["vector"]
      };
      if reference.is_null() {
        continue;
      }
      let reference = vector(reference);
      let id = item["id"].as_str().expect("id");
      let got = embedder
        .embed(
          item["text"].as_str().expect("text"),
          prompt_of(&item["prompt"]),
        )
        .unwrap_or_else(|e| panic!("{id}: {e}"));
      let widths = [
        cosine(got.as_slice(), &reference),
        cosine(got.truncate::<512>().as_slice(), &reference),
        cosine(got.truncate::<256>().as_slice(), &reference),
        cosine(got.truncate::<128>().as_slice(), &reference),
      ];
      for (slot, value) in widths.iter().enumerate() {
        worst[slot] = worst[slot].min(*value);
      }
      count += 1;
    }
    eprintln!(
      "M2{} ({count} items, split = {split}): worst cosine 768 {:.6}, 512 {:.6}, 256 {:.6}, 128 {:.6}",
      if split { "b" } else { "" },
      worst[0],
      worst[1],
      worst[2],
      worst[3]
    );
    assert!(count > 0);
    for (width, value) in [768, 512, 256, 128].into_iter().zip(worst) {
      assert!(value >= R1_FLOOR, "split = {split}, width {width}: {value}");
    }
  }
}

/// M3's outputs half: the kit's vectors against what FluidUse got from Core
/// ML for the same items on the characterizing host (`embed_S`, the Neural
/// Engine, `Split`), each re-normalized in f64 as the kit's are, within 1e-6
/// in every coordinate.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_text_vectors_are_the_swift_probes() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 text vs R2",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let text = golden("text.json");
  let probe = golden("swift_probe.json");
  let mut embedder = TextEmbedder::new(
    &artifacts,
    &options(ComputeUnits::CpuAndNeuralEngine, Batching::Single).with_split_special_tokens(true),
  )
  .expect("an instance");
  let joint = |id: &str| {
    let item = text["items"]
      .as_array()
      .expect("items")
      .iter()
      .find(|item| item["id"].as_str() == Some(id))
      .expect("the item");
    format!(
      "{}{}",
      item["prefix"].as_str().expect("prefix"),
      item["text"].as_str().expect("text")
    )
  };
  let mut worst = 0f64;
  for entry in probe["text"].as_array().expect("text") {
    let id = entry["id"].as_str().expect("id");
    if id == "bom-twin" {
      continue;
    }
    let got = embedder.embed(&joint(id), Prompt::Bare).expect("embeds");
    // FluidUse hands out the graph's own output, normalized inside the fp16
    // graph (its norm is 1 within about 1e-3); the kit re-normalizes in f64.
    // So the reference is re-normalized the same way before the comparison.
    let raw = vector(&entry["vector"]);
    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    let reference: Vec<f64> = raw.iter().map(|v| v / norm).collect();
    let deviation = got
      .as_slice()
      .iter()
      .zip(&reference)
      .map(|(a, b)| (f64::from(*a) - b).abs())
      .fold(0f64, f64::max);
    worst = worst.max(deviation);
  }
  eprintln!("M3 outputs: worst coordinate deviation from R2 {worst:e}");
  println!(
    "{}",
    gate.check_ceiling("text vs R2, worst coordinate", worst as f32, 1e-6)
  );
}

/// M4: one sequence at every slot of `pack_256`, beside different companions,
/// against itself alone at slot 0 and against its `embed_S` vector, under
/// every placement. Prints the arms where every vector is bit-identical:
/// `PACKED_SLOT_INVARIANT` holds exactly those, and each `true` it holds is
/// asserted here on the characterizing host.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn packing_is_measured_slot_by_slot() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open("embeddinggemma2 packing", CHARACTERIZED_ON, RECHARACTERIZE);
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let subject = "A slow afternoon by the river, counting the boats that drift past.";
  let companions = [
    "the quick brown fox",
    "Café crème, naïve résumé",
    "fn main() { println!(\"hi\"); }",
    "雨の日の午後",
    "42",
    "Did 30 minutes easy this morning.",
    "<b> </b>",
  ];
  for units in ARMS {
    let mut embedder =
      TextEmbedder::new(&artifacts, &options(units, Batching::Packed)).expect("an instance");
    let alone = bits(&embedder.embed(subject, Prompt::Bare).expect("embeds"));
    let mut out = Vec::new();
    embedder
      .embed_batch_with(&[subject], Prompt::Bare, Batching::Single, &mut out)
      .expect("embeds");
    let single = bits(&out[0]);
    let mut invariant = true;
    for slot in 1..8 {
      let mut batch: Vec<&str> = companions[..slot].to_vec();
      batch.push(subject);
      let mut out = Vec::new();
      embedder
        .embed_batch_with(&batch, Prompt::Bare, Batching::Packed, &mut out)
        .expect("embeds");
      invariant &= bits(&out[slot]) == alone;
    }
    let matches_single = alone == single;
    eprintln!("M4 {units:?}: slot-invariant {invariant}; packed alone == embed_S {matches_single}");
    if PACKED_SLOT_INVARIANT.get(units) {
      println!(
        "{}",
        gate.check_holds(
          &format!("pack_256 slot invariance under {units:?}"),
          invariant,
          "every slot bit-identical",
          "a slot moved the vector",
        )
      );
    }
  }
}

/// Texts long enough to reach every unpacked function: 32, 48, 64, 128, 256
/// and 512 ids.
fn reaching_every_function() -> Vec<String> {
  [20usize, 40, 60, 120, 250, 500]
    .into_iter()
    .map(|words| {
      (0..words)
        .map(|i| ["river", "boat", "light", "stone", "wind"][i % 5])
        .collect::<Vec<_>>()
        .join(" ")
    })
    .collect()
}

/// M8: each text function under every placement, its vector's cosine against
/// the same function under `CpuOnly`; plus `pack_256` with two sequences.
/// Measured, printed, and asserted on the characterizing host as a band.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn every_text_function_is_characterized_per_placement() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 text placement",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let texts = reaching_every_function();
  let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
  let mut reference: Vec<Embedding> = Vec::new();
  TextEmbedder::new(
    &artifacts,
    &options(ComputeUnits::CpuOnly, Batching::Single),
  )
  .expect("an instance")
  .embed_batch(&refs, Prompt::Bare, &mut reference)
  .expect("embeds");
  let mut packed_reference = Vec::new();
  TextEmbedder::new(
    &artifacts,
    &options(ComputeUnits::CpuOnly, Batching::Packed),
  )
  .expect("an instance")
  .embed_batch(&refs[..2], Prompt::Bare, &mut packed_reference)
  .expect("embeds");
  for units in &ARMS[1..] {
    let mut out = Vec::new();
    TextEmbedder::new(&artifacts, &options(*units, Batching::Single))
      .expect("an instance")
      .embed_batch(&refs, Prompt::Bare, &mut out)
      .expect("embeds");
    let mut packed = Vec::new();
    TextEmbedder::new(&artifacts, &options(*units, Batching::Packed))
      .expect("an instance")
      .embed_batch(&refs[..2], Prompt::Bare, &mut packed)
      .expect("embeds");
    let pairs = out
      .iter()
      .zip(&reference)
      .chain(packed.iter().zip(&packed_reference));
    let worst = pairs
      .map(|(a, b)| a.cosine(b))
      .fold(f32::INFINITY, f32::min);
    eprintln!("M8 {units:?}: worst cosine against CpuOnly {worst:.6}");
    println!(
      "{}",
      gate.check_band(
        &format!("text under {units:?} vs CpuOnly"),
        worst,
        PLACEMENT_LO,
        1.0
      )
    );
  }
}

/// This process's physical footprint (`phys_footprint`), in bytes.
fn phys_footprint() -> u64 {
  // SAFETY: `task_vm_info` is plain-old-data; the all-zero bit pattern is a
  // valid value for every field.
  let mut info: mach2::task_info::task_vm_info = unsafe { core::mem::zeroed() };
  let mut count = (core::mem::size_of::<mach2::task_info::task_vm_info>()
    / core::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
  // SAFETY: `mach_task_self()` names the current task; `info` is a zeroed,
  // properly sized and aligned TASK_VM_INFO out-struct and `count` carries its
  // capacity in words per the in-out contract, so the kernel writes only
  // within `info`. The result code is checked before `info` is read.
  let result = unsafe {
    libc::task_info(
      mach2::traps::mach_task_self(),
      mach2::task_info::TASK_VM_INFO,
      (&raw mut info).cast(),
      &mut count,
    )
  };
  assert_eq!(result, libc::KERN_SUCCESS, "task_info(TASK_VM_INFO)");
  info.phys_footprint
}

/// Set in the child process a measurement runs in, naming what it measures.
const CHILD: &str = "EMBEDDINGGEMMA2_TEXT_CHILD";

/// Runs `name` alone in a child process with `CHILD` set to `what`, and
/// returns the child's line that starts with `marker`.
fn in_child(name: &str, what: &str, marker: &str) -> String {
  let child = std::process::Command::new(std::env::current_exe().expect("the test binary"))
    .args([
      "--exact",
      name,
      "--ignored",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(CHILD, what)
    .output()
    .expect("the child runs");
  let stdout = String::from_utf8_lossy(&child.stdout).into_owned();
  assert!(
    child.status.success(),
    "the child failed:\n{stdout}\n{}",
    String::from_utf8_lossy(&child.stderr)
  );
  // libtest prints `test <name> ... ` on the line the child's own output
  // begins, so the marker is found within a line, not at its start.
  stdout
    .lines()
    .find_map(|line| line.split_once(marker).map(|(_, rest)| rest.to_owned()))
    .unwrap_or_else(|| panic!("the child printed no `{marker}` line:\n{stdout}"))
}

/// M9a: the physical footprint of a text instance with one function loaded,
/// then all seven, then a second instance with all seven; and two instances
/// over one `TextArtifacts` read one mapping. Measured in a child process,
/// because the footprint is the whole process's.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_text_footprint_is_characterized() {
  const NAME: &str = "the_text_footprint_is_characterized";
  let Some(dir) = staged() else { return };
  if std::env::var_os(CHILD).is_some() {
    let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
    let start = phys_footprint();
    let one = TextOptions::new().with_prewarm(TextFunctions::empty().with(TextFunction::Embed128));
    let mut first = TextEmbedder::new(&artifacts, &one).expect("an instance");
    first.prewarm().expect("warms");
    let with_one = phys_footprint();
    let mut all = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
    drop(first);
    all.prewarm().expect("warms");
    let with_all = phys_footprint();
    let mut second = TextEmbedder::new(&artifacts.clone(), &TextOptions::new()).expect("a second");
    second.prewarm().expect("warms");
    let with_two = phys_footprint();
    println!("M9A {start} {with_one} {with_all} {with_two}");
    return;
  }
  let line = in_child(NAME, "footprint", "M9A ");
  let readings: Vec<u64> = line
    .split(' ')
    .map(|n| n.parse().expect("a reading"))
    .collect();
  let mib = |bytes: u64| bytes as f64 / f64::from(1 << 20);
  eprintln!(
    "M9a: one function +{:.1} MiB; all seven +{:.1} MiB; a second instance +{:.1} MiB",
    mib(readings[1].saturating_sub(readings[0])),
    mib(readings[2].saturating_sub(readings[0])),
    mib(readings[3].saturating_sub(readings[2]))
  );
}

/// One vector's bits, under `units`, from a fresh instance.
fn fresh_bits(artifacts: &TextArtifacts, units: ComputeUnits, text: &str) -> Vec<u32> {
  let mut embedder =
    TextEmbedder::new(artifacts, &options(units, Batching::Single)).expect("an instance");
  bits(&embedder.embed(text, Prompt::Bare).expect("embeds"))
}

/// M11: does one text input repeat bit for bit, twice on one instance, on two
/// instances and in two processes, under every placement? Prints the table
/// `run_identity` holds; each `Some(true)` it holds is asserted here on the
/// characterizing host.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn text_run_identity_is_measured_per_placement() {
  const NAME: &str = "text_run_identity_is_measured_per_placement";
  const TEXT: &str = "A slow afternoon by the river, counting the boats that drift past.";
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  if let Some(arm) = std::env::var_os(CHILD) {
    let index: usize = arm
      .to_str()
      .and_then(|a| a.parse().ok())
      .expect("an arm index");
    let units = ARMS[index];
    let bits = fresh_bits(&artifacts, units, TEXT);
    println!("M11 {bits:?}");
    return;
  }
  let gate = BandGate::open(
    "embeddinggemma2 text run identity",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  for (index, units) in ARMS.into_iter().enumerate() {
    let mut embedder =
      TextEmbedder::new(&artifacts, &options(units, Batching::Single)).expect("an instance");
    let first = bits(&embedder.embed(TEXT, Prompt::Bare).expect("embeds"));
    let twice = bits(&embedder.embed(TEXT, Prompt::Bare).expect("embeds")) == first;
    let two_instances = fresh_bits(&artifacts, units, TEXT) == first;
    let two_processes = in_child(NAME, &index.to_string(), "M11 ") == format!("{first:?}");
    let identical = twice && two_instances && two_processes;
    eprintln!(
      "M11 {units:?}: twice {twice}, two instances {two_instances}, two processes {two_processes}"
    );
    if run_identity(Part::Text, units) == Some(true) {
      println!(
        "{}",
        gate.check_holds(
          &format!("text run identity under {units:?}"),
          identical,
          "bit-identical",
          "a repeat moved the vector",
        )
      );
    }
  }
}

/// M12: text throughput against the number of instances (1 to 4, one thread
/// each) per placement, over `embed_128`-sized texts. Printed.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn text_throughput_is_characterized() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let text = reaching_every_function().swap_remove(3);
  const PER_INSTANCE: usize = 24;
  for units in ARMS {
    for instances in 1..=4 {
      let mut embedders: Vec<TextEmbedder> = (0..instances)
        .map(|_| {
          let mut e =
            TextEmbedder::new(&artifacts, &options(units, Batching::Single)).expect("an instance");
          e.embed(&text, Prompt::Bare).expect("warms");
          e
        })
        .collect();
      let started = Instant::now();
      std::thread::scope(|scope| {
        for embedder in &mut embedders {
          let text = &text;
          scope.spawn(move || {
            for _ in 0..PER_INSTANCE {
              embedder.embed(text, Prompt::Bare).expect("embeds");
            }
          });
        }
      });
      let elapsed: Duration = started.elapsed();
      eprintln!(
        "M12 {units:?} x{instances}: {:.1} texts/s",
        (instances * PER_INSTANCE) as f64 / elapsed.as_secs_f64()
      );
    }
  }
}

/// The four long texts' windows under their prompts and the default options,
/// per text: each window's `(byte_start, byte_end, ids)`. Recorded at the
/// first run: they depend on the tokenizer image and windit's chunker alone,
/// so a move is a change to one of them.
const LONG_WINDOWS: [(&str, &[(usize, usize, usize)]); 4] = [
  (
    "long-en",
    &[
      (0, 2328, 504),
      (2328, 4383, 460),
      (4383, 6441, 463),
      (6441, 7027, 137),
    ],
  ),
  (
    "long-ja",
    &[(0, 2459, 500), (2459, 4932, 506), (4932, 5298, 82)],
  ),
  (
    "long-code",
    &[
      (0, 1412, 504),
      (1412, 2699, 470),
      (2699, 3986, 470),
      (3986, 5273, 470),
      (5273, 5558, 106),
    ],
  ),
  (
    "long-unbroken",
    &[(0, 1277, 512), (1277, 2555, 512), (2555, 3000, 179)],
  ),
];

/// T7's real half, the geometry: the four long texts' windows under their
/// prompts and the default options partition each text on `char` boundaries;
/// every window carries the prompt (its ids are the joint encoding of `prompt
/// ⧺ chunk`) and at most 512 ids; the spans run on; and the geometry is the
/// pinned one.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_long_texts_window_as_pinned() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut embedder = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let text_golden = golden("text.json");
  let (mut scratch, mut ids) = (TokenizerScratch::new(), Vec::new());
  let mut observed = Vec::new();
  for item in text_golden["items"]
    .as_array()
    .expect("items")
    .iter()
    .filter(|item| item["source"] == "long")
  {
    let id = item["id"].as_str().expect("id");
    let text = item["text"].as_str().expect("text");
    let prompt = prompt_of(&item["prompt"]);
    let windows = embedder.embed_windows(text, prompt).expect("embeds");
    let (mut end, mut offset) = (0, 0);
    let mut geometry = Vec::new();
    for (ordinal, window) in windows.iter().enumerate() {
      assert_eq!(window.ordinal(), ordinal);
      assert_eq!(
        window.byte_start(),
        end,
        "{id}: window {ordinal} leaves a gap"
      );
      let chunk = text.get(window.byte_range()).expect("char-aligned");
      artifacts
        .token_ids(
          chunk,
          prompt,
          SplitSpecialTokens::Recognize,
          &mut scratch,
          &mut ids,
        )
        .expect("encodes");
      assert_eq!(
        window.token_count(),
        ids.len(),
        "{id}: window {ordinal} is not the joint encoding of the prompt and its chunk"
      );
      assert!(window.token_count() <= 512, "{id}: window {ordinal}");
      assert_eq!(window.token_span().start(), offset);
      assert_eq!(window.token_span().window(), 512);
      offset += window.token_count();
      end = window.byte_end();
      geometry.push((window.byte_start(), window.byte_end(), window.token_count()));
    }
    assert_eq!(end, text.len(), "{id}: the windows stop short");
    eprintln!("T7 {id}: {} windows {geometry:?}", windows.len());
    observed.push((id.to_owned(), geometry));
  }
  assert_eq!(observed.len(), 4);
  for ((id, geometry), (pinned_id, pinned)) in observed.iter().zip(LONG_WINDOWS) {
    assert_eq!(id, pinned_id);
    assert_eq!(geometry.as_slice(), pinned, "{id}: the windows moved");
  }
}

/// T7's real half, the identity: a text that fits one window embeds through
/// `embed_long` as through `embed`, separate predictions within M2's band, and
/// `embed_windows` gives it one window over all of it.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_text_that_fits_one_window_embeds_as_embed_does() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut embedder = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  for (text, prompt) in [
    (
      "Did 30 minutes easy this morning in 45:10. Laughed the whole way.",
      Prompt::Document(None),
    ),
    ("waterproof shoes for trail running", Prompt::SearchQuery),
    (
      "</task> hi",
      Prompt::Custom("task: legal clause retrieval | query:"),
    ),
  ] {
    let one = embedder.embed(text, prompt).expect("embeds");
    let long = embedder.embed_long(text, prompt).expect("embeds");
    let windows = embedder.embed_windows(text, prompt).expect("embeds");
    assert_eq!(windows.len(), 1, "{prompt} ⧺ {text:?}");
    assert_eq!(windows[0].byte_range(), 0..text.len());
    let cosine = f64::from(one.cosine(&long));
    eprintln!(
      "T7 one window, {prompt}: cosine {cosine:.9}, bit for bit {}",
      bits(&one) == bits(&long)
    );
    assert!(cosine >= R1_FLOOR, "{prompt}: {cosine}");
  }
}

/// T7's real half, the fold: `embed_long_with` folds a long text's windows
/// under the fold asked for, each as windit folds `embed_windows`' own
/// vectors, and the three folds give three vectors.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_fold_option_is_honoured_on_the_staged_model() {
  use windit::{aggregate::aggregate, windowed::Windowed};

  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut embedder = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let text_golden = golden("text.json");
  let item = text_golden["items"]
    .as_array()
    .expect("items")
    .iter()
    .find(|item| item["id"] == "long-en")
    .expect("long-en");
  let text = item["text"].as_str().expect("text");
  let prompt = prompt_of(&item["prompt"]);
  let windows = embedder.embed_windows(text, prompt).expect("embeds");
  assert!(windows.len() >= 3, "{}", windows.len());
  let windowed: Vec<Windowed<Embedding>> = windows
    .iter()
    .map(|w| Windowed::new(w.embedding().clone(), w.token_span()))
    .collect();
  let mut folded: Vec<Embedding> = Vec::new();
  for kind in [
    AggregatePolicyKind::CoverageWeightedMean,
    AggregatePolicyKind::MeanRenormalized,
    AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.5)),
  ] {
    let expected = aggregate(kind.into_policy().as_ref(), &windowed).expect("folds");
    let got = embedder
      .embed_long_with(text, prompt, &LongTextOptions::new(), &kind)
      .expect("embeds");
    let cosine = got.cosine(&expected);
    eprintln!("T7 fold {kind}: cosine {cosine:.9} against windit's fold of the windows");
    assert!(cosine >= 1.0 - 1e-6, "{kind}: {cosine}");
    folded.push(got);
  }
  for (a, b) in [(0, 1), (0, 2), (1, 2)] {
    let cosine = folded[a].cosine(&folded[b]);
    eprintln!("T7 folds {a} and {b}: cosine {cosine:.9}");
    assert!(cosine < 1.0 - 1e-5, "folds {a} and {b} agree: {cosine}");
  }
}
