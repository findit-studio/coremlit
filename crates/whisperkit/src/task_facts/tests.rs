use super::*;

/// `a` merged with `b`, as a pure value — the shape every merge entry point
/// applies, lifted out so associativity is testable directly on the record.
fn merged(a: &TaskFacts, b: &TaskFacts) -> TaskFacts {
  let mut out = a.clone();
  out.merge(b);
  out
}

/// A deliberately varied corpus spanning every field's interesting states:
/// each explicit-unknown boolean at all THREE of its values (`None`,
/// `Some(false)`, `Some(true)`) and — since the Kleene OR came apart from the
/// pre-round-8 free monoid on exactly the `Some(false)`/`None` mixes — both
/// booleans crossed at every combination that matters (`(T, F)`, `(T, T)`,
/// `(None, F)`); three languages plus the unknown; known single/empty/unknown
/// schedules; and exact / at-least / overflowing / identity (`Exact(0)`) spans,
/// whose round-12 `SpanKnowledge` sum saturates and degrades across states —
/// including the `unknown()` value and
/// an all-dropped-style `[]` schedule. The `N` elements below drive the
/// `N`-cubed associativity proof of [the Kleene contributor merge](TaskFacts::merge).
fn corpus() -> Vec<TaskFacts> {
  vec![
    TaskFacts::unknown(),
    TaskFacts::unknown()
      .with_worker(0)
      .with_decoded_span(SpanKnowledge::Exact(1)),
    TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_worker(2)
      .with_decoded_span(SpanKnowledge::Exact(3)),
    TaskFacts::unknown()
      .with_observed_language(Some("es".to_string()))
      .with_worker(1),
    TaskFacts::unknown()
      .with_early_stopped(true)
      .with_observed_language(Some("en".to_string()))
      .with_decoded_span(SpanKnowledge::Exact(2)),
    // POSITIVELY observed `Some(false)` facts — the shape a real greedy,
    // un-truncated decode carries — distinct from the `unknown()` `None` above,
    // so the merge's `Some(false)`/`None`/`Some(true)` mixes are all covered.
    TaskFacts::unknown()
      .with_drew_from_rng(false)
      .with_early_stopped(false)
      .with_worker(4)
      .with_decoded_span(SpanKnowledge::Exact(1)),
    TaskFacts::unknown()
      .with_drew_from_rng(false)
      .with_early_stopped(true),
    // A merge of zero workers (an empty-but-known schedule), distinct from the
    // unknown schedule above.
    TaskFacts::unknown().with_worker_schedule(Some(Vec::new())),
    // Kleene-cross additions (codex round 8): a drew-`true`/stop-`false` run, a
    // drew-`true`/stop-`true` run, and a `None`-draw/`Some(false)`-stop record —
    // the mixes on which `Some(false) | None = None` diverges from the old
    // `Some(false)`-as-identity monoid, so associativity is re-proved across them.
    TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_early_stopped(false)
      .with_worker(5),
    TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_early_stopped(true)
      .with_observed_language(Some("de".to_string()))
      .with_decoded_span(SpanKnowledge::Exact(4)),
    TaskFacts::unknown()
      .with_early_stopped(false)
      .with_worker(6),
    // An overflowing EXACT span (round 10, F3): `usize::MAX` alongside the span-1
    // and span-2 children above forms the MAX,1,2 triple whose grouping the pre-fix
    // identity-`None` made non-associative. The round-12 `SpanKnowledge` sum must
    // render every triple that mixes it associative.
    TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(usize::MAX)),
    // `AtLeast` spans (codex round 12): the KNOWN lower bound the round-12 redesign
    // adds, at a small bound, at zero (the wholly-unknown value distinct from the
    // `Exact(0)` identity below), and at the saturated `usize::MAX`. Crossing
    // exact/at-least/overflow spans is what proves `SpanKnowledge::merge`
    // associative — checked-add over exacts, saturating once any side is a bound.
    TaskFacts::unknown()
      .with_decoded_span(SpanKnowledge::AtLeast(2))
      .with_worker(8),
    TaskFacts::unknown().with_decoded_span(SpanKnowledge::AtLeast(0)),
    TaskFacts::unknown().with_decoded_span(SpanKnowledge::AtLeast(usize::MAX)),
    // The span monoid identity, `Exact(0)` — a KNOWN-empty span (a zero-chunk VAD
    // run), distinct from the wholly-unknown `AtLeast(0)` above.
    TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(0)),
    // Swallowed-error variants (codex round 11, M2): the third Kleene boolean at
    // its `Some(true)` (a swallowed drop) and `Some(false)` (an observed-clean
    // watch) states, crossed with a draw and a worker, so the associativity proof
    // below folds it through the same `Some(true)`/`Some(false)`/`None` mixes the
    // other two booleans are proved on.
    TaskFacts::unknown()
      .with_had_swallowed_error(true)
      .with_worker(7),
    TaskFacts::unknown()
      .with_drew_from_rng(false)
      .with_had_swallowed_error(false),
  ]
}

#[test]
fn merge_is_associative_over_three_children() {
  // THE property (coremlit issue #14, codex round 6; re-proved for the Kleene OR
  // in round 8): a one-shot merge and a staged merge -- the
  // VAD-result-then-streaming-finalize shape -- must agree. `(a . b) . c == a .
  // (b . c)` for every triple in the corpus, unknown-state and empty-schedule
  // children included. Kleene three-valued OR is itself associative, so swapping
  // the free-monoid `or_unknown` for `kleene_or` preserves this even though it
  // changed the `Some(false) | None` VALUE -- the Kleene-cross corpus additions
  // exercise exactly those diverging mixes.
  let corpus = corpus();
  for a in &corpus {
    for b in &corpus {
      for c in &corpus {
        let left = merged(&merged(a, b), c);
        let right = merged(a, &merged(b, c));
        assert_eq!(
          left, right,
          "merge is not associative for a={a:?} b={b:?} c={c:?}"
        );
      }
    }
  }
}

#[test]
fn accumulator_empty_is_the_fold_identity() {
  // The Accumulator -- NOT `unknown()` -- is the identity of a contributor fold
  // (codex round 8, F2). `Empty` takes the first contributor verbatim, so no
  // all-`None` `unknown()` boolean identity ever nulls a known `Some(false)`.
  for x in corpus() {
    let mut acc = TaskFactsAccumulator::new();
    acc.merge(&x);
    assert_eq!(acc.into_facts(), x, "Empty is a verbatim left identity");
  }
  // Zero contributors fold to `unknown()` -- the honest "nothing observed".
  assert_eq!(
    TaskFactsAccumulator::new().into_facts(),
    TaskFacts::unknown(),
    "an empty fold is unknown, not observed-clean",
  );
  // A multi-contributor fold equals the left-fold that takes the first verbatim
  // then merges the rest -- the exact shape `merge_results` relies on.
  for a in &corpus() {
    for b in &corpus() {
      let mut acc = TaskFactsAccumulator::new();
      acc.merge(a);
      acc.merge(b);
      assert_eq!(
        acc.into_facts(),
        merged(a, b),
        "fold == verbatim-first then merge for a={a:?} b={b:?}",
      );
    }
  }
}

#[test]
fn unknown_is_not_the_merge_identity_for_an_observed_clean_fact() {
  // The round-8 correction. Under the Kleene OR, `Some(false)` -- not `None` --
  // is the boolean identity, so folding the all-`None` `unknown()` onto an
  // observed-clean `Some(false)` NULLS it to unknown rather than preserving it.
  // This is precisely why a contributor fold must seed from
  // `TaskFactsAccumulator::Empty`, never from `unknown()`.
  let clean = TaskFacts::observed_clean();
  assert_eq!(clean.drew_from_rng(), Some(false));
  assert_eq!(clean.early_stopped(), Some(false));
  assert_eq!(
    merged(&TaskFacts::unknown(), &clean).drew_from_rng(),
    None,
    "unknown() is NOT a left identity for an observed-clean draw",
  );
  assert_eq!(merged(&clean, &TaskFacts::unknown()).drew_from_rng(), None);

  // A `Some(true)` or `None` boolean beside a `None` is unchanged by the Kleene
  // OR, and an absent language still adopts the other's (here also absent). But
  // round 10 (F2) extends the correction to the worker SCHEDULE: `unknown()` is
  // not an identity there either -- `None` is absorbing, so an unknown coordinate
  // beside a known one NULLS the whole schedule rather than passing it through.
  // (The decoded-span field's own absorbing correction is pinned by
  // `merge_sums_the_decoded_span` and the associativity corpus.)
  let drew = TaskFacts::unknown().with_drew_from_rng(true).with_worker(2);
  let left = merged(&TaskFacts::unknown(), &drew);
  assert_eq!(
    left.drew_from_rng(),
    Some(true),
    "None | Some(true) = Some(true)"
  );
  assert_eq!(left.observed_language(), None);
  assert_eq!(
    left.worker_schedule(),
    None,
    "an unknown coordinate absorbs a known one: unknown() is not a schedule identity (round 10, F2)",
  );
  assert_eq!(
    merged(&drew, &TaskFacts::unknown()).worker_schedule(),
    None,
    "and in the other order",
  );
}

#[test]
fn merge_ors_the_bools_by_the_kleene_table() {
  // Pin the FULL Kleene three-valued OR the merge folds the draw/early-stop
  // booleans with (codex round 8, F2). Vehicle: `drew_from_rng`, driven through
  // the merge from every (self, other) pair. `Some(true)` absorbs, `Some(false)
  // | Some(false)` stays `Some(false)`, and an unknown mixed with
  // anything-but-true stays unknown -- INCLUDING the corrected `None |
  // Some(false) = None` and `Some(false) | None = None`, the transitions the
  // pre-round-8 free monoid (`None` as identity) wrongly pinned as `Some(false)`.
  // That old oracle is deliberately replaced here, on codex round 8's authority:
  // a child that cannot observe the draw must not certify the other's `false`.
  let of = |b: Option<bool>| match b {
    Some(v) => TaskFacts::unknown().with_drew_from_rng(v),
    None => TaskFacts::unknown(),
  };
  let table = [
    (Some(true), Some(true), Some(true)),
    (Some(true), Some(false), Some(true)),
    (Some(true), None, Some(true)),
    (Some(false), Some(true), Some(true)),
    (Some(false), Some(false), Some(false)),
    (Some(false), None, None), // round-8 correction (was Some(false))
    (None, Some(true), Some(true)),
    (None, Some(false), None), // round-8 correction (was Some(false))
    (None, None, None),
  ];
  for (a, b, expected) in table {
    assert_eq!(
      merged(&of(a), &of(b)).drew_from_rng(),
      expected,
      "kleene_or({a:?}, {b:?})",
    );
  }

  // The same table drives `had_swallowed_error` too (codex round 11, M2): it is a
  // third `Option<bool>` folded through the very same `kleene_or`, so pin it on
  // the transitions that matter -- `Some(true)` absorbs, two `Some(false)` stay
  // clean, and an unknown beside a `Some(false)` poisons to `None`.
  let se = |b: Option<bool>| match b {
    Some(v) => TaskFacts::unknown().with_had_swallowed_error(v),
    None => TaskFacts::unknown(),
  };
  for (a, b, expected) in table {
    assert_eq!(
      merged(&se(a), &se(b)).had_swallowed_error(),
      expected,
      "kleene_or({a:?}, {b:?}) on had_swallowed_error",
    );
  }

  // The same law reaches all THREE booleans independently: a drew-true merged with
  // a stopped-true and a swallowed-true carries all three `Some(true)`.
  let both = merged(
    &TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_had_swallowed_error(true),
    &TaskFacts::unknown().with_early_stopped(true),
  );
  assert_eq!(both.drew_from_rng(), Some(true));
  assert_eq!(both.early_stopped(), Some(true));
  assert_eq!(both.had_swallowed_error(), Some(true));

  // `Some(false)` IS the OR identity (not `None`): a real greedy contributor
  // folded beside a draw or another greedy behaves as OR.
  let greedy = TaskFacts::unknown().with_drew_from_rng(false);
  let drew = TaskFacts::unknown().with_drew_from_rng(true);
  assert_eq!(merged(&greedy, &drew).drew_from_rng(), Some(true));
  assert_eq!(merged(&greedy, &greedy).drew_from_rng(), Some(false));
}

#[test]
fn merge_keeps_the_first_observed_language() {
  let es = TaskFacts::unknown().with_observed_language(Some("es".to_string()));
  let en = TaskFacts::unknown().with_observed_language(Some("en".to_string()));
  assert_eq!(
    merged(&es, &en).observed_language(),
    Some("es"),
    "first wins"
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &en).observed_language(),
    Some("en"),
    "an unknown-language child adopts the other's observation",
  );
}

#[test]
fn merge_concatenates_worker_schedules_in_order() {
  // R6-F2: a merge of [0] and [2] must be distinguishable from [0] and [1].
  let w = |n| TaskFacts::unknown().with_worker(n);
  assert_eq!(
    merged(&w(0), &w(2)).worker_schedule(),
    Some([0, 2].as_slice())
  );
  assert_ne!(
    merged(&w(0), &w(2)).worker_schedule(),
    merged(&w(0), &w(1)).worker_schedule(),
    "the collapsed pre-fix merge made these two indistinguishable",
  );

  // ORACLE CORRECTION (round 10, F2): `None` is ABSORBING, not the identity. A
  // child that cannot report its ordered coordinates taints the aggregate to
  // unknown -- partial knowledge must not read back as a fully-known schedule.
  // The pre-round-10 law pinned these two as `Some([0])` / `Some([2])` (`None`
  // as the identity); that oracle is replaced here on round 10's authority,
  // mirroring the Kleene-bool correction (codex round 8) that gave `None` its
  // absorbing role.
  //
  // Mutation proof: revert the merge to the `(None, Some(more)) => Some(...)`
  // identity arm and both `None` expectations below read back `Some([0])` /
  // `Some([2])`.
  assert_eq!(
    merged(&w(0), &TaskFacts::unknown()).worker_schedule(),
    None,
    "an unknown contributor absorbs a known coordinate (was Some([0]))",
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &w(2)).worker_schedule(),
    None,
    "and in the other order (was Some([2]))",
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &TaskFacts::unknown()).worker_schedule(),
    None,
    "two unknowns stay unknown, not [0]",
  );

  // `Some([])` (known-empty) IS the identity: it leaves a known schedule
  // unchanged on either side, and two known-empties stay known-empty -- distinct
  // from the absorbing unknown above.
  let empty = || TaskFacts::unknown().with_worker_schedule(Some(Vec::new()));
  let known_empty: &[usize] = &[];
  assert_eq!(
    merged(&w(0), &empty()).worker_schedule(),
    Some([0].as_slice()),
    "a known-empty child is the identity for a known coordinate",
  );
  assert_eq!(
    merged(&empty(), &w(2)).worker_schedule(),
    Some([2].as_slice()),
  );
  assert_eq!(
    merged(&empty(), &empty()).worker_schedule(),
    Some(known_empty),
    "known-empty is the identity of itself, never nulled to unknown",
  );
}

#[test]
fn merge_sums_the_decoded_span() {
  // R6-F3: two KNOWN spans sum to the aggregate ordinal count their children
  // allocated.
  let s = |n| TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(n));
  assert_eq!(
    merged(&s(2), &s(1)).decoded_span(),
    SpanKnowledge::Exact(3),
    "two exact spans sum exactly"
  );

  // ORACLE CORRECTION (codex round 12): a wholly-unknown (`AtLeast(0)`) child no
  // longer ABSORBS its known sibling's ordinals — the round-10/11 absorbing-`None`
  // that the round-12 redesign replaces. The KNOWN sibling survives as the
  // aggregate's LOWER BOUND: `AtLeast(0) + Exact(2) = AtLeast(2)`, so a staged
  // re-merge can still advance past the 2 ordinals the sibling allocated, which is
  // exactly what the pre-round-12 `None` threw away.
  //
  // Mutation proof: revert `SpanKnowledge::merge`'s `AtLeast` arm to the absorbing
  // `_ => None` and these `AtLeast(2)` expectations read back the wholly-unknown
  // `AtLeast(0)`.
  assert_eq!(
    merged(&s(2), &TaskFacts::unknown()).decoded_span(),
    SpanKnowledge::AtLeast(2),
    "an unknown child lower-bounds, never erases, the known sibling (was absorbing None)",
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &s(2)).decoded_span(),
    SpanKnowledge::AtLeast(2),
    "and in the other order",
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &TaskFacts::unknown()).decoded_span(),
    SpanKnowledge::AtLeast(0),
    "two wholly-unknown children stay wholly unknown",
  );
}

#[test]
fn merge_records_an_overflowing_span_as_a_saturated_lower_bound() {
  // F2 (codex round 9), corrected under round 12. Summing two exact spans that
  // overflow `usize` has no exact answer, so the sum degrades to a SATURATED LOWER
  // BOUND `AtLeast(usize::MAX)` — never a fabricated EXACT `usize::MAX` a staged
  // re-merge would trust as a precise ordinal count, and never (pre-round-12) a
  // bound-less `None` that threw the count away.
  //
  // Mutation proof: revert the `Exact + Exact` overflow arm to `Self::Exact(a
  // .saturating_add(b))` and this reads back the EXACT `usize::MAX`.
  let overflowing = merged(
    &TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(usize::MAX)),
    &TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(1)),
  );
  assert_eq!(
    overflowing.decoded_span(),
    SpanKnowledge::AtLeast(usize::MAX),
    "an overflowing exact sum is a saturated lower bound, not an exact usize::MAX",
  );
}

#[test]
fn merge_is_associative_over_an_overflowing_span_triple() {
  // F3 (round 10), corrected under round 12. THE associativity property the span
  // law must keep: spans MAX, 1, 2. The pre-round-10 identity-`None` gave
  // `(A·B)·C = Some(2)` but `A·(B·C) = None`. Under the round-12 `SpanKnowledge`
  // sum both groupings agree at the saturated lower bound `AtLeast(usize::MAX)`:
  // `Exact(MAX) + Exact(1)` overflows to `AtLeast(MAX)`, and any further `AtLeast`
  // participation saturates, so the true total (`MAX + 3`) is reported as the
  // grouping-independent `AtLeast(usize::MAX)`.
  //
  // Mutation proof: revert the `Exact + Exact` overflow arm to the absorbing
  // `_ => None` (or the identity arm) and the two groupings diverge, failing the
  // equality.
  let a = TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(usize::MAX));
  let b = TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(1));
  let c = TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(2));
  let left = merged(&merged(&a, &b), &c);
  let right = merged(&a, &merged(&b, &c));
  assert_eq!(
    left.decoded_span(),
    right.decoded_span(),
    "the documented merge associativity holds across an overflowing span triple",
  );
  assert_eq!(
    left.decoded_span(),
    SpanKnowledge::AtLeast(usize::MAX),
    "the overflowed total is the saturated lower bound, grouping-independent",
  );
}

#[test]
fn is_reproducible_under_is_conservative_on_the_explicit_unknown() {
  // F1 (codex round 6 post-consolidation). The bare `unknown()` — draw AND
  // truncation both unobserved — is NOT reproducible: a record that cannot know
  // whether a transcript-controlling event happened must not promise
  // byte-reproducibility. This is the case the old `false`-means-both
  // representation wrongly called reproducible.
  let unknown = TaskFacts::unknown();
  assert!(
    !unknown.is_reproducible_under(false),
    "unknown is not a promise"
  );
  assert!(!unknown.is_reproducible_under(true));

  // A genuinely greedy, un-truncated, no-swallow decode POSITIVELY observes all
  // THREE `false` — and only THAT earns the optimistic answer, seed or not. Each
  // fixture below pins ONE axis by holding the other two at their observed-clean
  // `Some(false)`, so the swallowed-error axis added in round 11 does not mask the
  // draw/truncation axes these assertions exist to catch.
  let greedy = TaskFacts::observed_clean();
  assert!(
    greedy.is_reproducible_under(false),
    "observed-greedy reproduces"
  );
  assert!(greedy.is_reproducible_under(true));

  // An observed unseeded draw is not reproducible; a seed makes it replayable —
  // but only because the truncation and swallow are ALSO positively `Some(false)`.
  let drew = TaskFacts::observed_clean().with_drew_from_rng(true);
  assert!(
    !drew.is_reproducible_under(false),
    "an unseeded draw is not reproducible"
  );
  assert!(
    drew.is_reproducible_under(true),
    "a seed makes the draw replayable"
  );

  // An observed early stop forces false regardless of the seed: the callback is
  // not in the record.
  let stopped = TaskFacts::observed_clean().with_early_stopped(true);
  assert!(!stopped.is_reproducible_under(false));
  assert!(!stopped.is_reproducible_under(true));

  // An observed swallowed child error forces false regardless of the seed (codex
  // round 11, M2): the hidden error controlled the transcript, and re-running the
  // same audio and options need not reproduce the swallow. This is the axis a
  // record built through the observed-clean sink flips to `Some(true)` at a VAD
  // chunk drop or a failed language probe.
  let swallowed = TaskFacts::observed_clean().with_had_swallowed_error(true);
  assert!(!swallowed.is_reproducible_under(false));
  assert!(!swallowed.is_reproducible_under(true));

  // An UNKNOWN factor poisons the answer even when the others are observed-clean:
  // an unobserved truncation (the `for_segment` shape), an unobserved draw, or an
  // unobserved swallow is conservatively non-reproducible, seed or not. Each holds
  // the OTHER two at `Some(false)` and leaves its own axis at `unknown()`'s `None`.
  let truncation_unknown = TaskFacts::unknown()
    .with_drew_from_rng(false)
    .with_had_swallowed_error(false);
  assert!(!truncation_unknown.is_reproducible_under(false));
  assert!(
    !truncation_unknown.is_reproducible_under(true),
    "an unobserved truncation is never reproducible, whatever the seed"
  );
  let draw_unknown = TaskFacts::unknown()
    .with_early_stopped(false)
    .with_had_swallowed_error(false);
  assert!(!draw_unknown.is_reproducible_under(false));
  assert!(!draw_unknown.is_reproducible_under(true));
  // The swallow axis, unobserved: the `None` the other two booleans have always
  // carried for a segment-/options-only record, mirrored (codex round 11, M2).
  let swallow_unknown = TaskFacts::unknown()
    .with_drew_from_rng(false)
    .with_early_stopped(false);
  assert!(!swallow_unknown.is_reproducible_under(false));
  assert!(!swallow_unknown.is_reproducible_under(true));
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_every_field() {
  let full = TaskFacts::unknown()
    .with_drew_from_rng(true)
    .with_observed_language(Some("es".to_string()))
    .with_early_stopped(true)
    .with_had_swallowed_error(true)
    .with_worker_schedule(Some(vec![0, 2]))
    .with_decoded_span(SpanKnowledge::Exact(5));
  let json = serde_json::to_string(&full).unwrap();
  assert_eq!(serde_json::from_str::<TaskFacts>(&json).unwrap(), full);

  // A POSITIVELY observed `Some(false)` draw/early-stop round-trips as `false`,
  // and reads back distinct from the `None` unknown — the whole point of the
  // tri-state (F1). If `false` and unknown serialized the same, this would fail.
  let greedy = TaskFacts::unknown()
    .with_drew_from_rng(false)
    .with_early_stopped(false)
    .with_worker(0);
  let json = serde_json::to_string(&greedy).unwrap();
  let read: TaskFacts = serde_json::from_str(&json).unwrap();
  assert_eq!(read, greedy);
  assert_eq!(read.drew_from_rng(), Some(false));
  assert_ne!(read.drew_from_rng(), TaskFacts::unknown().drew_from_rng());

  // The unknown record round-trips too: null draw/early-stop/language/schedule
  // read back as explicit unknown, and the absent span defaults to the
  // wholly-unknown `AtLeast(0)`.
  let unknown = TaskFacts::unknown();
  let json = serde_json::to_string(&unknown).unwrap();
  assert_eq!(serde_json::from_str::<TaskFacts>(&json).unwrap(), unknown);
}

#[cfg(feature = "serde")]
#[test]
fn the_reproducibility_and_coordinate_facts_are_required_on_deserialize() {
  // The optimistic direction is the dangerous one: were a dropped
  // `drew_from_rng`, `early_stopped`, or `had_swallowed_error` to default to
  // `None` and were `None` optimistic, a dropped key would leak a reproducibility
  // answer; a dropped `worker_schedule` would forge a worker (R6-F2). All five are
  // rejected on a missing key; only the transient `decoded_span` may be absent.
  let full = TaskFacts::unknown()
    .with_drew_from_rng(true)
    .with_observed_language(Some("es".to_string()))
    .with_early_stopped(true)
    .with_had_swallowed_error(true)
    .with_worker_schedule(Some(vec![0, 2]))
    .with_decoded_span(SpanKnowledge::Exact(5));
  let value: serde_json::Value = serde_json::to_value(&full).unwrap();
  assert_eq!(
    serde_json::from_str::<TaskFacts>(&value.to_string()).unwrap(),
    full,
    "the intact record must round-trip, or the removals below prove nothing",
  );

  for required in [
    "drew_from_rng",
    "observed_language",
    "early_stopped",
    "had_swallowed_error",
    "worker_schedule",
  ] {
    let mut without = value.clone();
    without.as_object_mut().unwrap().remove(required).unwrap();
    assert!(
      serde_json::from_str::<TaskFacts>(&without.to_string()).is_err(),
      "a missing `{required}` must fail, not default",
    );
  }

  // Present-but-null is the honest, ACCEPTED encoding of explicit unknown for
  // all five nullable fields — the draw, truncation, and swallowed-error among
  // them (F1; codex round 11, M2).
  let mut nulled = value.clone();
  for field in [
    "observed_language",
    "worker_schedule",
    "drew_from_rng",
    "early_stopped",
    "had_swallowed_error",
  ] {
    nulled.as_object_mut().unwrap()[field] = serde_json::Value::Null;
  }
  let read: TaskFacts = serde_json::from_str(&nulled.to_string()).unwrap();
  assert_eq!(read.observed_language(), None);
  assert_eq!(
    read.worker_schedule(),
    None,
    "null schedule is explicit unknown, never [0]"
  );
  assert_eq!(
    read.drew_from_rng(),
    None,
    "null draw is explicit unknown, never false"
  );
  assert_eq!(
    read.early_stopped(),
    None,
    "null early-stop is explicit unknown, never false"
  );
  assert_eq!(
    read.had_swallowed_error(),
    None,
    "null swallowed-error is explicit unknown, never false"
  );

  // The transient span may be dropped without error, reading back the
  // wholly-unknown `AtLeast(0)` (round 12: the old `None`).
  let mut without_span = value;
  without_span.as_object_mut().unwrap().remove("decoded_span");
  assert_eq!(
    serde_json::from_str::<TaskFacts>(&without_span.to_string())
      .unwrap()
      .decoded_span(),
    SpanKnowledge::wholly_unknown(),
  );
}

// ---------------------------------------------------------------------
// SpanKnowledge — the two-state id-span fact (coremlit issue #14, codex round 12)
// ---------------------------------------------------------------------

/// A span corpus spanning both states at the boundary values the merge's
/// checked/saturating arithmetic cares about: the identity, small counts, and
/// the `usize::MAX` edge where an exact sum overflows.
fn span_corpus() -> Vec<SpanKnowledge> {
  vec![
    SpanKnowledge::Exact(0), // the merge identity
    SpanKnowledge::Exact(1),
    SpanKnowledge::Exact(2),
    SpanKnowledge::Exact(usize::MAX - 1),
    SpanKnowledge::Exact(usize::MAX),
    SpanKnowledge::wholly_unknown(), // AtLeast(0)
    SpanKnowledge::AtLeast(1),
    SpanKnowledge::AtLeast(2),
    SpanKnowledge::AtLeast(usize::MAX),
  ]
}

#[test]
fn span_knowledge_accessors() {
  assert_eq!(SpanKnowledge::wholly_unknown(), SpanKnowledge::AtLeast(0));
  assert_eq!(SpanKnowledge::Exact(3).lower_bound(), 3);
  assert_eq!(SpanKnowledge::AtLeast(3).lower_bound(), 3);
  assert!(SpanKnowledge::Exact(0).is_exact());
  assert!(!SpanKnowledge::AtLeast(0).is_exact());
  // Only `AtLeast(0)` is wholly unknown — `Exact(0)` (a KNOWN-empty span) is not,
  // so it is serialized rather than skipped.
  assert!(SpanKnowledge::AtLeast(0).is_wholly_unknown());
  assert!(!SpanKnowledge::Exact(0).is_wholly_unknown());
  assert!(!SpanKnowledge::AtLeast(1).is_wholly_unknown());
}

#[test]
fn span_knowledge_merge_closed_form() {
  // Two exacts sum exactly; an overflowing exact sum degrades to the saturated
  // lower bound; any `AtLeast` participation yields the saturating sum of the two
  // lower bounds.
  assert_eq!(
    SpanKnowledge::Exact(2).merge(SpanKnowledge::Exact(3)),
    SpanKnowledge::Exact(5),
  );
  assert_eq!(
    SpanKnowledge::Exact(usize::MAX).merge(SpanKnowledge::Exact(1)),
    SpanKnowledge::AtLeast(usize::MAX),
    "an exact sum with no usize answer degrades to a saturated lower bound",
  );
  assert_eq!(
    SpanKnowledge::AtLeast(1).merge(SpanKnowledge::Exact(2)),
    SpanKnowledge::AtLeast(3),
    "a known lower bound plus an exact count is a lower bound of their sum",
  );
  assert_eq!(
    SpanKnowledge::AtLeast(usize::MAX).merge(SpanKnowledge::AtLeast(2)),
    SpanKnowledge::AtLeast(usize::MAX),
    "a lower bound sum saturates, never wraps",
  );
}

#[test]
fn span_knowledge_merge_identity_associative_commutative() {
  // THE property the round-12 redesign rests on: `SpanKnowledge::merge` is an
  // associative, commutative monoid with `Exact(0)` its identity, so a staged and
  // a one-shot merge store the same span in every grouping. Proven exhaustively
  // over the boundary corpus (identity, small, and `usize::MAX`-edge spans in both
  // states) — the same shape the `TaskFacts` corpus folds through `TaskFacts::merge`.
  let corpus = span_corpus();
  for &a in &corpus {
    // Two-sided identity.
    assert_eq!(
      SpanKnowledge::Exact(0).merge(a),
      a,
      "Exact(0) is a left identity"
    );
    assert_eq!(
      a.merge(SpanKnowledge::Exact(0)),
      a,
      "Exact(0) is a right identity"
    );
    for &b in &corpus {
      assert_eq!(
        a.merge(b),
        b.merge(a),
        "merge is commutative for {a:?}, {b:?}"
      );
      for &c in &corpus {
        assert_eq!(
          a.merge(b).merge(c),
          a.merge(b.merge(c)),
          "merge is not associative for {a:?}, {b:?}, {c:?}",
        );
      }
    }
  }
}
