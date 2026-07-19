//! Embedding parity vs the committed transformers-fp32 goldens — the
//! model-level gate (NO ort; the oracle is in-tree fixtures, never a live crate).
//!
//! For each committed corpus entry, the granite CoreML embedder embeds the raw
//! `text` and the result is scored by cosine against the entry's transformers-fp32
//! **unit-normalized** 384-d embedding golden. Both sides are unit-norm, so
//! cosine == dot; the comparison is fail-closed (`common::cosine_checked`
//! length- and finiteness-checks the `Vec` golden — the #30 lesson).
//!
//! # Bands (measure-then-pin)
//!
//! The fp32-CPU arm (`ComputeUnits::CpuOnly`) is the PRIMARY gate: CoreML fp32
//! against transformers fp32 agrees near 1.0. Its two-sided band clears the
//! MEASURED worst by a margin; a shift in EITHER direction is a finding, not a
//! threshold to loosen. The default-compute arm (`ComputeUnits::All`, which the
//! T1 probe measured at ~97.8% ANE / fp16) is CHARACTERIZED separately — its
//! band is the measured fp16 reality, not floored to the fp32 arm.
//!
//! Measured 2026-07-19 on this machine (printed by the tests below).

mod common;

use coremlit::{
  ComputeUnits, Model, MultiArray,
  embeddings::granite::{
    Embedding, MAX_TOKENS, TextEmbedder, TextEmbedderOptions, embedding::EMBEDDING_DIM,
  },
};

// ── MEASURED-then-pinned two-sided bands (measured 2026-07-19; worst cosine over
//    the 16-entry committed corpus; cosine of two unit-norm vectors == dot). ──
// fp32-CPU arm (the PRIMARY gate): measured worst = 0.99997884 (entry `near512`);
// floor pinned at 0.9998 — clears the worst by ~1.9e-4, comfortably over the
// ~1.8e-5 cross-placement drift (placement.rs) — ceiling just over 1.0 for ULP
// slop. A shift in EITHER direction is a finding, not a threshold to loosen.
const CPU_LO: f32 = 0.9998;
const CPU_HI: f32 = 1.0 + 1e-4;
// default-compute (All / fp16-ANE) arm, CHARACTERIZED separately (measured, not
// floored to the fp32 arm): measured worst = 0.99999928 (entries `url_num` /
// `short`). The fp16/ANE lowering agrees even TIGHTER than fp32-CPU here, so its
// floor is pinned higher (0.9999) — this is the measured fp16 reality.
const FP16_LO: f32 = 0.9999;
const FP16_HI: f32 = 1.0 + 1e-4;
// Non-vacuity ceiling for the negative controls: the measured cross-entry MAX
// cosine = 0.94094926 (entry 0 vs the other 15) and the shifted-slice mutation
// collapses to ~-0.034 — both must clear this ceiling, which itself sits far
// below the 0.9998 parity floor (so a real agreement is distinguished from
// everything trivially landing near 1.0).
const NEGATIVE_CEIL: f32 = 0.97;

fn embedder(units: ComputeUnits) -> TextEmbedder {
  TextEmbedder::load(
    common::model_path(),
    TextEmbedderOptions::new().with_compute(units),
  )
  .unwrap_or_else(|e| panic!("load granite [{}]: {e}", units.as_str()))
}

/// Worst (min) cosine of `embed(text)` vs the golden, over the whole corpus, on
/// `units`. Prints each so the measurement is visible.
fn parity_worst(units: ComputeUnits, tier: &str) -> f32 {
  let emb = embedder(units);
  let corpus = common::golden_corpus();
  let mut worst = 1.0f32;
  for e in &corpus {
    let got = emb
      .embed(&e.text)
      .unwrap_or_else(|err| panic!("embed `{}` [{tier}]: {err}", e.id));
    let cos = common::cosine_checked(got.as_slice(), &e.embedding);
    println!("[{tier}] {:<8} cos = {cos:.8}", e.id);
    worst = worst.min(cos);
  }
  println!("[{tier}] WORST cosine = {worst:.8}");
  worst
}

#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn parity_vs_goldens_cpu_fp32() {
  let worst = parity_worst(ComputeUnits::CpuOnly, "cpu-fp32");
  assert!(
    (CPU_LO..=CPU_HI).contains(&worst),
    "granite fp32-CPU parity worst cosine {worst:.8} outside pinned band [{CPU_LO}, {CPU_HI}] \
     — a shift in EITHER direction is a finding (re-measure, do not just widen)"
  );
}

#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn parity_vs_goldens_default_compute_fp16_characterized() {
  let worst = parity_worst(ComputeUnits::All, "all-fp16");
  assert!(
    (FP16_LO..=FP16_HI).contains(&worst),
    "granite default-compute (All/fp16) parity worst cosine {worst:.8} outside CHARACTERIZED band \
     [{FP16_LO}, {FP16_HI}] — the measured fp16 reality moved; re-measure"
  );
}

/// Negative control: an entry's embedding vs a DIFFERENT entry's golden is far
/// below the parity floor, so the ~1.0 cosines above are real agreement, not
/// everything trivially landing near 1.0.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn cross_entry_cosine_is_far_below_the_parity_floor() {
  let emb = embedder(ComputeUnits::CpuOnly);
  let corpus = common::golden_corpus();
  // Embed entry 0, score it against every OTHER entry's golden.
  let got = emb.embed(&corpus[0].text).expect("embed entry 0");
  let mut worst_cross = 0.0f32;
  for e in corpus.iter().skip(1) {
    let cos = common::cosine_checked(got.as_slice(), &e.embedding);
    worst_cross = worst_cross.max(cos);
  }
  println!("[negative-control] entry0-vs-others MAX cross cosine = {worst_cross:.8}");
  assert!(
    worst_cross < NEGATIVE_CEIL,
    "cross-entry cosine {worst_cross:.8} is implausibly high — the parity metric is not \
     discriminating"
  );
}

// ── Mutation proofs (non-vacuity): the parity gate WOULD red under a real bug ──
//
// CLS pooling and attention are IN-GRAPH, so the mutations are applied at the
// tensor boundary (the raw model output / the attention mask) to show that a
// pooling-slice or mask off-by-one bug collapses the golden cosine below the
// floor. Each loads a RAW `Model` and replicates the embedder's tensor pipeline,
// then perturbs exactly one step.

/// granite's pad token id (`<|endoftext|>`); the masked pad positions never
/// reach the CLS output, so the value is immaterial — documented for the raw
/// pipeline below.
const PAD_ID: i32 = 179_935;

/// The raw pre-norm [384] model output for `ids`, with the attention mask set to
/// 1 on the first `mask_len` positions (real tokens) and 0 elsewhere. `mask_len
/// == ids.len()` is the correct pipeline; a different `mask_len` is the
/// off-by-one mutation.
fn raw_output(model: &Model, ids: &[u32], mask_len: usize) -> [f32; EMBEDDING_DIM] {
  let mut input_ids = [PAD_ID; MAX_TOKENS];
  let mut mask = [0i32; MAX_TOKENS];
  for (i, &id) in ids.iter().enumerate() {
    input_ids[i] = id as i32;
  }
  for m in mask.iter_mut().take(mask_len.min(MAX_TOKENS)) {
    *m = 1;
  }
  let ids_t = MultiArray::from_slice(&[1, MAX_TOKENS], &input_ids).unwrap();
  let mask_t = MultiArray::from_slice(&[1, MAX_TOKENS], &mask).unwrap();
  let mut out = model
    .predict_with(&[("input_ids", &ids_t), ("attention_mask", &mask_t)])
    .expect("predict");
  let emb = out.take("embedding").expect("embedding output");
  let mut row = [0.0f32; EMBEDDING_DIM];
  emb.copy_into::<f32>(&mut row).expect("copy output");
  row
}

/// Mutation A — a MASK OFF-BY-ONE (drop the last real token from the mask) must
/// drive the golden cosine below the parity floor. Proves the gate is sensitive
/// to the exact mask the embedder builds. Uses a mid-length entry so dropping one
/// token materially changes the CLS output.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn mutation_mask_off_by_one_reds_the_gate() {
  let emb = embedder(ComputeUnits::CpuOnly);
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).unwrap();
  let corpus = common::golden_corpus();
  // `url_num` (29 tokens) — long enough that one dropped token matters.
  let e = corpus
    .iter()
    .find(|e| e.id == "url_num")
    .expect("url_num entry");
  let ids = emb.token_ids(&e.text).expect("tokenize");

  // Correct pipeline reproduces the golden…
  let correct = Embedding::from_slice_normalizing(&raw_output(&model, &ids, ids.len())).unwrap();
  let cos_ok = common::cosine_checked(correct.as_slice(), &e.embedding);
  // …and the off-by-one mask does NOT.
  let mutated =
    Embedding::from_slice_normalizing(&raw_output(&model, &ids, ids.len() - 1)).unwrap();
  let cos_bad = common::cosine_checked(mutated.as_slice(), &e.embedding);
  println!("[mutation/mask] correct cos = {cos_ok:.8}, off-by-one cos = {cos_bad:.8}");

  assert!(
    (CPU_LO..=CPU_HI).contains(&cos_ok),
    "the CORRECT mask must reproduce the golden ({cos_ok:.8})"
  );
  assert!(
    cos_bad < CPU_LO,
    "a mask off-by-one must RED the parity gate (cos {cos_bad:.8} should fall below the floor \
     {CPU_LO}) — otherwise the mask construction is not load-bearing"
  );
}

/// Mutation B — a WRONG POOLING SLICE (rotate the pooled output by one component,
/// the shifted-slice analog of mis-pooling) must drive the golden cosine below
/// the parity floor. Proves the 384-d alignment is exact and a pooling/slice bug
/// would be caught.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn mutation_shifted_pooling_slice_reds_the_gate() {
  let emb = embedder(ComputeUnits::CpuOnly);
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).unwrap();
  let corpus = common::golden_corpus();
  let e = &corpus[0];
  let ids = emb.token_ids(&e.text).expect("tokenize");
  let raw = raw_output(&model, &ids, ids.len());

  let correct = Embedding::from_slice_normalizing(&raw).unwrap();
  let cos_ok = common::cosine_checked(correct.as_slice(), &e.embedding);

  // Rotate the 384 components by one — a slice/pooling misalignment.
  let mut shifted = [0.0f32; EMBEDDING_DIM];
  for i in 0..EMBEDDING_DIM {
    shifted[i] = raw[(i + 1) % EMBEDDING_DIM];
  }
  let mutated = Embedding::from_slice_normalizing(&shifted).unwrap();
  let cos_bad = common::cosine_checked(mutated.as_slice(), &e.embedding);
  println!("[mutation/pooling] correct cos = {cos_ok:.8}, shifted-slice cos = {cos_bad:.8}");

  assert!(
    (CPU_LO..=CPU_HI).contains(&cos_ok),
    "the CORRECT pooling must reproduce the golden ({cos_ok:.8})"
  );
  assert!(
    cos_bad < NEGATIVE_CEIL,
    "a shifted pooling slice must RED the parity gate (cos {cos_bad:.8} should collapse below \
     {NEGATIVE_CEIL})"
  );
}
