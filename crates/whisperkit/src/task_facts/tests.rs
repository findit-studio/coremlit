use super::*;

/// `a` merged with `b`, as a pure value — the shape every merge entry point
/// applies, lifted out so associativity is testable directly on the record.
fn merged(a: &TaskFacts, b: &TaskFacts) -> TaskFacts {
  let mut out = a.clone();
  out.merge(b);
  out
}

/// A deliberately varied corpus spanning every field's interesting states:
/// draw on/off, three languages plus the unknown, truncated or not, known
/// single/empty/unknown schedules, and tracked/untracked spans — including the
/// `unknown()` identity and an all-dropped-style `[]` schedule.
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
    // A merge of zero workers (an empty-but-known schedule), distinct from the
    // unknown schedule above.
    TaskFacts::unknown().with_worker_schedule(Some(Vec::new())),
  ]
}

#[test]
fn merge_is_associative_over_three_children() {
  // THE property (coremlit issue #14, codex round 6): a one-shot merge and a
  // staged merge -- the VAD-result-then-streaming-finalize shape -- must agree.
  // `(a . b) . c == a . (b . c)` for every triple in the corpus, unknown-state
  // and empty-schedule children included.
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
fn unknown_is_the_merge_identity() {
  for x in corpus() {
    assert_eq!(merged(&TaskFacts::unknown(), &x), x, "left identity");
    assert_eq!(merged(&x, &TaskFacts::unknown()), x, "right identity");
  }
}

#[test]
fn merge_ors_the_bools() {
  let drew = TaskFacts::unknown().with_drew_from_rng(true);
  let stopped = TaskFacts::unknown().with_early_stopped(true);
  let both = merged(&drew, &stopped);
  assert!(both.drew_from_rng() && both.early_stopped());
  // OR, not last-write: a false does not clear a true.
  assert!(merged(&drew, &TaskFacts::unknown()).drew_from_rng());
  assert!(merged(&TaskFacts::unknown(), &drew).drew_from_rng());
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
fn is_reproducible_under_reads_the_two_carried_facts() {
  let greedy = TaskFacts::unknown();
  assert!(
    greedy.is_reproducible_under(false),
    "greedy reproduces unseeded"
  );
  assert!(greedy.is_reproducible_under(true));

  let drew = TaskFacts::unknown().with_drew_from_rng(true);
  assert!(
    !drew.is_reproducible_under(false),
    "an unseeded draw is not reproducible"
  );
  assert!(
    drew.is_reproducible_under(true),
    "a seed makes the draw replayable"
  );

  // An early stop forces false regardless of the seed: the callback is not in
  // the record.
  let stopped = TaskFacts::unknown().with_early_stopped(true);
  assert!(!stopped.is_reproducible_under(false));
  assert!(!stopped.is_reproducible_under(true));
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

  // The unknown record round-trips too: null language/schedule read back as
  // explicit unknown, and the absent span defaults to None.
  let unknown = TaskFacts::unknown();
  let json = serde_json::to_string(&unknown).unwrap();
  assert_eq!(serde_json::from_str::<TaskFacts>(&json).unwrap(), unknown);
}

#[cfg(feature = "serde")]
#[test]
fn the_reproducibility_and_coordinate_facts_are_required_on_deserialize() {
  // The optimistic direction is the dangerous one: a dropped `drew_from_rng`
  // or `early_stopped` reads back the reproducible answer, and a dropped
  // `worker_schedule` a fabricated worker (R6-F2). All four are rejected on a
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
  // the two nullable fields.
  let mut nulled = value.clone();
  nulled.as_object_mut().unwrap()["observed_language"] = serde_json::Value::Null;
  nulled.as_object_mut().unwrap()["worker_schedule"] = serde_json::Value::Null;
  let read: TaskFacts = serde_json::from_str(&nulled.to_string()).unwrap();
  assert_eq!(read.observed_language(), None);
  assert_eq!(
    read.worker_schedule(),
    None,
    "null schedule is explicit unknown, never [0]"
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
