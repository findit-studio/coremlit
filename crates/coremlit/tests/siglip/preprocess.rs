//! NaFlex preprocessing parity gates.
//!
//! # Status: Wave B/C shells
//!
//! The pure preprocessing MATH (budget solver against the probe's `spatial_shapes`
//! oracle, antialiased resize, patchify/mask, pos-emb lift, determinism) is
//! proven HERMETICALLY in the in-lib unit tests
//! (`src/embeddings/siglip/image/preprocess/tests.rs`, in the default
//! `cargo test --features siglip`). This integration file holds the
//! oracle-fixture-gated arms:
//!
//! - **Wave B** — the committed small-oracle cases (`fixtures/goldens/preprocess.json`):
//!   resize vs torch-oracle arrays, the `(H,W)→(h_p,w_p)` budget table (exact),
//!   a tiny patchify tensor, and the normalize constants.
//! - **Wave C** — full-tensor parity: the Rust `pixel_values`/`attention_mask`/
//!   `position_embeddings` vs the staged per-image `.npy` fixtures (via `npyz`),
//!   exact mask equality, bitwise-zero pad rows.

mod common;

/// Small committed oracles: resize / budget-table / patchify / normalize vs the
/// torch-generated `preprocess.json`. Wave B implements against the staged file.
#[test]
#[ignore = "requires committed preprocess small-oracles (fixtures/goldens/preprocess.json) — Wave B"]
fn small_oracles_match_committed_torch_reference() {
  let _oracle = common::fixture_path("goldens/preprocess.json");
  // Wave B: parse and compare (tolerance measured-then-pinned ~1e-6-class for the
  // resize; exact for the budget table and patchify).
}

/// Full-tensor parity of the Rust NaFlex tensors against the staged per-image
/// `.npy` fixtures. Wave C implements against `SIGLIP_TEST_MODELS`.
#[test]
#[ignore = "requires staged siglip preprocessing fixtures (SIGLIP_TEST_MODELS) — Wave C"]
fn full_tensor_parity_against_staged_npy() {
  let _dir = common::models_dir();
  // Wave C: for each corpus image, decode PNG (common::decode_png_rgb8), run the
  //         ImageEmbedder preprocessing, compare pixel_values/mask/pos-emb to the
  //         staged `.npy`. Exact mask equality; pixel_values exact against the
  //         slow-processor (use_fast=False, pillow 12.3.0) fixture on the u8
  //         resize. For position_embeddings, compare the lifted REAL rows
  //         (`.. h_p·w_p`) to the fixture but assert the pad rows bitwise zero
  //         against the coremlit contract — the reference fills them with
  //         resized[0] (masked, output-invariant), so canonicalize/exclude the
  //         pad rows rather than comparing them to the raw dump.
}
