//! Hermetic integrity checks on the committed dia-ort parity goldens
//! (`tests/fixtures/golden/*.json`) that run in the ORDINARY `cargo test`
//! suite — no local models, no `ort`, no `dia` feature, nothing `#[ignore]`d.
//!
//! # Why this exists: the desync the parity suites cannot see
//!
//! `tests/generate_goldens.rs` writes each golden's `seg_model` provenance
//! string, but that string is metadata the parity gates
//! (`tests/parity_seg.rs`, `tests/parity_embed.rs`, via `common::load_golden`)
//! never read. So a divergence between the string the generator writes and the
//! string frozen into the committed goldens is invisible to every other test
//! until someone runs the `#[ignore]`d regenerator — and under the old
//! unconditional writer, running it (as the standard `cargo test -p speakerkit
//! --features dia -- --ignored` gate does) silently REWROTE the oracle.
//!
//! Exactly that desync landed on this branch: a doc/label correction changed
//! the generator's `seg_model` string from "raw powerset logits" to
//! "powerset log-probabilities" WITHOUT regenerating the committed goldens (and
//! correctly so — the values are unchanged log-probabilities either way, the
//! label was only ever a description). The generator's write is now guarded on
//! `UPDATE_GOLDEN`, and both sides read one pinned constant
//! ([`common::SEG_MODEL_LABEL`]); this test makes any future re-divergence a
//! LOUD failure in the ordinary suite instead of a silent oracle rewrite on the
//! next `--ignored` sweep. It is the fast, hermetic guard for the slow,
//! model-gated regenerator.

mod common;

/// The `seg_model` string frozen into every committed golden must equal the
/// single source of truth [`common::SEG_MODEL_LABEL`] — the exact string
/// `tests/generate_goldens.rs` writes when it regenerates a golden. If they
/// diverge (a label edit in the generator that skipped regeneration, or a
/// hand-edit of a committed golden), an `UPDATE_GOLDEN=1 ... --ignored` run
/// would re-baseline the oracle to the new string; this fails first, here, in
/// the ordinary suite, before that can happen.
#[test]
fn committed_goldens_seg_model_matches_source_label() {
  for fixture in common::FIXTURES {
    let path = common::golden_path(fixture.name);
    let bytes =
      std::fs::read(&path).unwrap_or_else(|e| panic!("read golden {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_slice(&bytes)
      .unwrap_or_else(|e| panic!("parse golden {}: {e}", fixture.name));
    let seg_model = v["seg_model"].as_str().unwrap_or_else(|| {
      panic!(
        "{}: committed golden has no string `seg_model`",
        fixture.name
      )
    });

    assert_eq!(
      seg_model,
      common::SEG_MODEL_LABEL,
      "{}: committed golden `seg_model` has drifted from `common::SEG_MODEL_LABEL` \
       (the exact string `tests/generate_goldens.rs` writes). committed={seg_model:?} \
       source={:?}. Either the generator's label changed without regenerating the \
       committed goldens, or a committed golden was hand-edited. Do NOT run \
       `UPDATE_GOLDEN=1 ... --ignored` to paper over this — that would re-baseline the \
       oracle; reconcile the label deliberately (see the generate_goldens write-guard \
       comment and `common::SEG_MODEL_LABEL`).",
      fixture.name,
      common::SEG_MODEL_LABEL,
    );
  }
}
