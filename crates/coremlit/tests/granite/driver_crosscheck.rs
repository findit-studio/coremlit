//! Hermetic gate on the committed conversion-faithfulness record
//! (`tests/granite/fixtures/goldens/driver_crosscheck.json`) — NO model, NO
//! network, NO Python. Runs in the default `cargo test --features granite`.
//!
//! # What the fixture is
//!
//! The shipped `.mlmodelc` was not traced from the stock `ModernBertModel`
//! forward: that forward rebuilds the RoPE tables and both attention masks on
//! every call, which cannot lower to one static CoreML graph. The conversion
//! recipe (`crates/coremlit/conversion/granite/`) instead drives the stock
//! submodules with the RoPE tables and the sliding-window geometry hoisted to
//! fixed-512 constants, and proves — BEFORE tracing — that this rewrite did not
//! move the model, by scoring the driver against the UNMODIFIED canonical
//! sentence-transformers pipeline over the same 16-entry corpus.
//!
//! This file is that measurement, not a later re-run of it: `convert_granite.py`
//! computes it, gates the conversion on it, and stages it; `generate_goldens.py`
//! publishes the staged record verbatim, adding only the `corpus_sha256` binding,
//! and refuses to run when the staged record is missing, from a different
//! conversion run, or recorded over different ordered inputs than the corpus
//! being written beside it.
//!
//! Every other granite gate scores against `corpus.json`, whose embeddings come
//! from the canonical pipeline. So nothing else in this crate would notice if
//! the traced driver had silently diverged from that pipeline — the CoreML
//! artifact would simply be faithful to a different model. This gate closes
//! that hole, and keeps the fixture from being an orphan claim nobody checks.
//!
//! # A cosine near 1.0 is not evidence of anything
//!
//! Scoring any vector against ITSELF also yields ~1.0: over these 16 committed
//! embeddings a self-cosine reduced in `f64` is not even exactly 1.0, drifting by
//! up to 2.2e-16 on 7 of the 16. So an accidental `driven = canonical` regression
//! would regenerate a fixture whose cosines look perfect. The cosine cannot
//! distinguish "the driver agrees" from "the driver was never run".
//!
//! `max_abs_component_delta` is the evidence that can: the largest per-component
//! difference between the unit-normalized driver vector and the canonical one.
//! Two genuinely independent fp32 computations of this embedding differ by at
//! least **9.28e-08** per component over the corpus (**2.83e-07** at the loosest
//! entry).
//!
//! That separation is real but narrower than it first appears, and the honest
//! numbers matter: a byte-identical stand-in scores exactly **0.0**, but a driver
//! that returned the canonical vector merely RESCALED still scores **1.2e-08 to
//! 2.6e-08**, purely from fp32 quantization of the normalize-and-compare round
//! trip. Only ~4x separates that from the genuine minimum. So [`DELTA_FLOOR`] is
//! a secondary band sitting between the two measurements, not a wide moat; the
//! primary defense is in the generator, which refuses to emit any pair scoring
//! below the same floor.
//!
//! # The bound above 1.0
//!
//! The committed values are reduced in `f64` and none exceeds 1.0. An earlier
//! fixture, cut by the since-lost original script, reduced at `f32` precision and
//! carried several values a few ULP ABOVE the mathematical bound. Both are
//! legitimate readings of the same quantity, so the ceiling here is
//! `1.0 + EPS_ABOVE_ONE` rather than exactly 1.0 — wide enough for either regime,
//! tight enough that a value which could not be a cosine still fails.

mod common;

/// Floor every recorded cosine must clear — the same faithfulness floor the
/// conversion recipe gates on (`DRIVER_FLOOR` in `scripts/_granite_common.py`).
/// Measured worst = 0.9999999999996902 (entry `special`). A drop below is a
/// finding in the conversion, not a threshold to loosen.
const DRIVER_FLOOR: f64 = 0.99999997;

/// Tolerance ABOVE the mathematical bound of 1.0, for the `f32`-reduction regime
/// described in the module docs (measured max excess there was 1.006e-7). The
/// committed `f64` values do not reach 1.0 at all, so this is headroom for a
/// differently-reduced fixture rather than a fit to the current data.
const EPS_ABOVE_ONE: f64 = 1e-6;

/// Floor on the recorded per-component distinctness, mirroring
/// `DISTINCTNESS_FLOOR` in `scripts/_granite_common.py`. Measured minimum over
/// the corpus = 9.28e-08 (1.9x above); a canonical-vector stand-in scores at most
/// 2.6e-08 (1.9x below). The margins are deliberately symmetric because that is
/// where the two measurements actually sit — see the module docs.
const DELTA_FLOOR: f64 = 5e-8;

/// The recipe's own reported verdict must be the agreeing one — the coarsest
/// check here, and the one that catches a regenerated fixture coming back
/// `DIVERGE` even if nobody read the numbers.
#[test]
fn verdict_is_agree() {
  let cc = common::driver_crosscheck();
  assert_eq!(
    cc.verdict, "AGREE",
    "the committed granite driver crosscheck reports verdict `{}` — the fixed-512 static-mask \
     driver diverged from the canonical pipeline, so the shipped artifact is faithful to a \
     DIFFERENT model than the goldens describe",
    cc.verdict
  );
}

/// The recorded worst cosine clears the recipe's faithfulness floor, and sits
/// inside the documented band around 1.0.
#[test]
fn worst_cosine_clears_the_driver_floor() {
  let cc = common::driver_crosscheck();
  let worst = cc.worst_cosine_canonical_vs_driver;
  assert!(
    (DRIVER_FLOOR..=1.0 + EPS_ABOVE_ONE).contains(&worst),
    "granite driver worst cosine {worst:.16} outside [{DRIVER_FLOOR}, 1.0 + {EPS_ABOVE_ONE:e}]"
  );
  // The recorded verdict must be consistent with the recorded numbers — a
  // hand-edited `AGREE` over a diverged measurement is caught here.
  assert!(
    1.0 - worst <= cc.stop_threshold_divergence,
    "verdict `{}` contradicts the numbers: divergence {:.3e} exceeds the recorded budget {:.3e}",
    cc.verdict,
    1.0 - worst,
    cc.stop_threshold_divergence
  );
}

/// The crosscheck covers EXACTLY the corpus: same ids, same count, nothing
/// dropped and nothing invented. Without this, an entry could be removed from
/// the crosscheck (or the corpus could grow) and the worst-cosine check above
/// would still pass while silently covering less.
#[test]
fn per_entry_ids_match_the_corpus_exactly() {
  let cc = common::driver_crosscheck();
  let corpus = common::golden_corpus();

  let recorded: std::collections::BTreeSet<&str> =
    cc.per_entry.iter().map(|e| e.id.as_str()).collect();
  let expected: std::collections::BTreeSet<&str> = corpus.iter().map(|e| e.id.as_str()).collect();
  assert_eq!(
    recorded, expected,
    "granite driver crosscheck ids differ from the corpus ids — the two goldens were cut from \
     different corpora"
  );
  // A duplicated id would collapse in the set comparison above, so the raw
  // counts are compared too.
  assert_eq!(
    cc.per_entry.len(),
    corpus.len(),
    "granite driver crosscheck has {} rows for {} corpus entries (duplicate or repeated id)",
    cc.per_entry.len(),
    corpus.len()
  );
}

/// The two goldens are the PAIR that was published together.
///
/// Ids alone cannot show this: if a regenerated `corpus.json` landed beside a
/// previous run's `driver_crosscheck.json`, every id would still line up while
/// the crosscheck described embeddings that are no longer there.
///
/// What `corpus_sha256` proves is exactly that — publication together, and
/// nothing more. The crosscheck is measured during conversion from live
/// canonical vectors, before any corpus file exists; `generate_goldens.py`
/// serializes `corpus.json`, hashes those bytes, and stamps the digest onto the
/// staged record as it emits both. So the field binds the two files as a unit
/// and catches a separated pair. It is NOT evidence that the measurement
/// consumed these bytes.
///
/// The recipe records a separate `corpus_input_sha256` during the measurement
/// itself, over the ordered `(id, text)` it consumed, and refuses to publish
/// goldens whose ordered inputs differ from it — that is the measurement-input
/// binding. It is enforced at generation time and required UNCONDITIONALLY by
/// `verify_granite.py`. This COMMITTED fixture predates the field and gains it at
/// the next regeneration, so nothing in this crate checks it yet — and the
/// recipe's own verification refuses to attest to this pair until then.
#[test]
fn crosscheck_is_bound_to_this_corpus() {
  let cc = common::driver_crosscheck();
  let actual = common::corpus_sha256();
  assert_eq!(
    cc.corpus_sha256, actual,
    "granite driver crosscheck was published alongside a corpus.json hashing to {}, but the \
     committed corpus.json hashes to {} — these two goldens came from different runs and must \
     be regenerated together",
    cc.corpus_sha256, actual
  );
}

/// EVERY per-entry cosine clears the floor — not just the recorded minimum — and
/// the recorded minimum really is the minimum of the rows.
#[test]
fn every_entry_clears_the_driver_floor() {
  let cc = common::driver_crosscheck();
  let mut below = Vec::new();
  let mut worst = f64::INFINITY;
  for entry in &cc.per_entry {
    let cos = entry.cosine_canonical_vs_driver;
    assert!(
      cos.is_finite(),
      "granite driver crosscheck `{}` is non-finite ({cos})",
      entry.id
    );
    assert!(
      cos <= 1.0 + EPS_ABOVE_ONE,
      "granite driver crosscheck `{}` = {cos:.16} exceeds 1.0 by more than {EPS_ABOVE_ONE:e} — \
       that is not float drift, it is not a cosine",
      entry.id
    );
    if cos < DRIVER_FLOOR {
      below.push(format!("{}: {cos:.16}", entry.id));
    }
    worst = worst.min(cos);
  }
  assert!(
    below.is_empty(),
    "granite driver crosscheck entries below the {DRIVER_FLOOR} faithfulness floor:\n  {}",
    below.join("\n  ")
  );
  assert_eq!(
    worst, cc.worst_cosine_canonical_vs_driver,
    "the recorded worst cosine is not the minimum of the per-entry rows"
  );
}

/// The driver and the canonical pipeline were genuinely two computations.
///
/// This is the check a `driven = canonical` regression must fail. Cosine cannot
/// see that bug (see the module docs); a per-component difference of exactly
/// zero is its unmistakable signature, on every entry at once.
#[test]
fn every_entry_records_independent_computation() {
  let cc = common::driver_crosscheck();
  let mut degenerate = Vec::new();
  let mut tightest = f64::INFINITY;
  for entry in &cc.per_entry {
    let delta = entry.max_abs_component_delta;
    assert!(
      delta.is_finite() && delta >= 0.0,
      "granite driver crosscheck `{}` has a nonsensical component delta ({delta})",
      entry.id
    );
    if delta < DELTA_FLOOR {
      degenerate.push(format!("{}: {delta:.3e}", entry.id));
    }
    tightest = tightest.min(delta);
  }
  assert!(
    degenerate.is_empty(),
    "granite driver crosscheck entries below the {DELTA_FLOOR:e} distinctness floor — the \
     driver output is indistinguishable from the canonical output, so the crosscheck compared \
     one computation with itself and proves nothing:\n  {}",
    degenerate.join("\n  ")
  );
  assert_eq!(
    tightest, cc.min_max_abs_component_delta,
    "the recorded minimum component delta is not the minimum of the per-entry rows"
  );
  // The corpus-wide summary must clear the floor on its own, so a fixture that
  // carried only the summary would still be gated.
  assert!(
    cc.min_max_abs_component_delta >= DELTA_FLOOR,
    "recorded min component delta {:.3e} is below the {DELTA_FLOOR:e} distinctness floor",
    cc.min_max_abs_component_delta
  );
}
