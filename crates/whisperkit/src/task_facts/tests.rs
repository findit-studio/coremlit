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
/// schedules; and tracked/untracked spans — including the `unknown()` value and
/// an all-dropped-style `[]` schedule. The `N` elements below drive the
/// `N`-cubed associativity proof of [the Kleene contributor merge](TaskFacts::merge).
fn corpus() -> Vec<TaskFacts> {
  vec![
    TaskFacts::unknown(),
    TaskFacts::unknown()
      .with_worker(0)
      .with_decoded_span(Some(1)),
    TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_worker(2)
      .with_decoded_span(Some(3)),
    TaskFacts::unknown()
      .with_observed_language(Some("es".to_string()))
      .with_worker(1),
    TaskFacts::unknown()
      .with_early_stopped(true)
      .with_observed_language(Some("en".to_string()))
      .with_decoded_span(Some(2)),
    // POSITIVELY observed `Some(false)` facts — the shape a real greedy,
    // un-truncated decode carries — distinct from the `unknown()` `None` above,
    // so the merge's `Some(false)`/`None`/`Some(true)` mixes are all covered.
    TaskFacts::unknown()
      .with_drew_from_rng(false)
      .with_early_stopped(false)
      .with_worker(4)
      .with_decoded_span(Some(1)),
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
      .with_decoded_span(Some(4)),
    TaskFacts::unknown()
      .with_early_stopped(false)
      .with_worker(6),
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

  // It REMAINS an identity for the non-Kleene fields and for a `Some(true)` or
  // `None` boolean, which the Kleene OR leaves unchanged beside a `None`.
  let drew = TaskFacts::unknown()
    .with_drew_from_rng(true)
    .with_worker(2)
    .with_decoded_span(Some(3));
  assert_eq!(
    merged(&TaskFacts::unknown(), &drew),
    drew,
    "None|Some(true)=Some(true), and worker/span/language keep None as identity",
  );
  assert_eq!(merged(&drew, &TaskFacts::unknown()), drew);
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

  // The same law reaches BOTH booleans independently: a drew-true merged with a
  // stopped-true carries both `Some(true)`.
  let both = merged(
    &TaskFacts::unknown().with_drew_from_rng(true),
    &TaskFacts::unknown().with_early_stopped(true),
  );
  assert_eq!(both.drew_from_rng(), Some(true));
  assert_eq!(both.early_stopped(), Some(true));

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
  // An unknown-coordinate child is the identity, never a fabricated 0.
  assert_eq!(
    merged(&w(0), &TaskFacts::unknown()).worker_schedule(),
    Some([0].as_slice()),
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &w(2)).worker_schedule(),
    Some([2].as_slice()),
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &TaskFacts::unknown()).worker_schedule(),
    None,
    "two unknowns stay unknown, not [0]",
  );
}

#[test]
fn merge_sums_the_decoded_span() {
  // R6-F3: the merged result stores the aggregate span its children allocated.
  let s = |n| TaskFacts::unknown().with_decoded_span(Some(n));
  assert_eq!(merged(&s(2), &s(1)).decoded_span(), Some(3));
  assert_eq!(
    merged(&s(2), &TaskFacts::unknown()).decoded_span(),
    Some(2),
    "an untracked child contributes nothing",
  );
  assert_eq!(
    merged(&TaskFacts::unknown(), &TaskFacts::unknown()).decoded_span(),
    None,
    "the sum is unknown only when every child is untracked",
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

  // A genuinely greedy, un-truncated decode POSITIVELY observes both `false` —
  // and only THAT earns the optimistic answer, seed or not.
  let greedy = TaskFacts::unknown()
    .with_drew_from_rng(false)
    .with_early_stopped(false);
  assert!(
    greedy.is_reproducible_under(false),
    "observed-greedy reproduces"
  );
  assert!(greedy.is_reproducible_under(true));

  // An observed unseeded draw is not reproducible; a seed makes it replayable —
  // but only because the truncation is ALSO positively observed as `false`.
  let drew = TaskFacts::unknown()
    .with_drew_from_rng(true)
    .with_early_stopped(false);
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
  let stopped = TaskFacts::unknown()
    .with_drew_from_rng(false)
    .with_early_stopped(true);
  assert!(!stopped.is_reproducible_under(false));
  assert!(!stopped.is_reproducible_under(true));

  // An UNKNOWN factor poisons the answer even when the other is observed-clean:
  // an unobserved truncation (the `for_segment` shape) or an unobserved draw is
  // conservatively non-reproducible, seed or not.
  let truncation_unknown = TaskFacts::unknown().with_drew_from_rng(false);
  assert!(!truncation_unknown.is_reproducible_under(false));
  assert!(
    !truncation_unknown.is_reproducible_under(true),
    "an unobserved truncation is never reproducible, whatever the seed"
  );
  let draw_unknown = TaskFacts::unknown().with_early_stopped(false);
  assert!(!draw_unknown.is_reproducible_under(false));
  assert!(!draw_unknown.is_reproducible_under(true));
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_every_field() {
  let full = TaskFacts::unknown()
    .with_drew_from_rng(true)
    .with_observed_language(Some("es".to_string()))
    .with_early_stopped(true)
    .with_worker_schedule(Some(vec![0, 2]))
    .with_decoded_span(Some(5));
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
  // read back as explicit unknown, and the absent span defaults to None.
  let unknown = TaskFacts::unknown();
  let json = serde_json::to_string(&unknown).unwrap();
  assert_eq!(serde_json::from_str::<TaskFacts>(&json).unwrap(), unknown);
}

#[cfg(feature = "serde")]
#[test]
fn the_reproducibility_and_coordinate_facts_are_required_on_deserialize() {
  // The optimistic direction is the dangerous one: were a dropped
  // `drew_from_rng` or `early_stopped` to default to `None` and were `None`
  // optimistic, a dropped key would leak a reproducibility answer; a dropped
  // `worker_schedule` would forge a worker (R6-F2). All four are rejected on a
  // missing key; only the transient `decoded_span` may be absent.
  let full = TaskFacts::unknown()
    .with_drew_from_rng(true)
    .with_observed_language(Some("es".to_string()))
    .with_early_stopped(true)
    .with_worker_schedule(Some(vec![0, 2]))
    .with_decoded_span(Some(5));
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
  // all four nullable fields — the draw and truncation among them (F1).
  let mut nulled = value.clone();
  for field in [
    "observed_language",
    "worker_schedule",
    "drew_from_rng",
    "early_stopped",
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

  // The transient span may be dropped without error, reading back untracked.
  let mut without_span = value;
  without_span.as_object_mut().unwrap().remove("decoded_span");
  assert_eq!(
    serde_json::from_str::<TaskFacts>(&without_span.to_string())
      .unwrap()
      .decoded_span(),
    None,
  );
}
