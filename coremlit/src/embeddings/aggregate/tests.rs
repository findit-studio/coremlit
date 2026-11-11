use super::*;

/// The shared selector keeps resolving at the clap paths it was first public at,
/// and both paths name the one type.
#[cfg(feature = "clap")]
#[test]
fn the_clap_paths_resolve_to_the_shared_selector() {
  use crate::embeddings::clap::{
    self,
    aggregate::{
      AggregatePolicyKind as ViaAggregate, EmaRenormalizedOptions as OptionsViaAggregate,
    },
  };

  let ema_via_aggregate = ViaAggregate::EmaRenormalized(OptionsViaAggregate::new(0.25));
  let ema_via_root =
    clap::AggregatePolicyKind::EmaRenormalized(clap::EmaRenormalizedOptions::new(0.25));
  let shared = AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.25));
  assert_eq!(ema_via_aggregate, shared);
  assert_eq!(ema_via_root, shared);
  assert_eq!(clap::EmaRenormalizedOptions::new(0.25).alpha(), 0.25);
}

/// The lines of the pre-lift golden that start with one of `prefixes`, in file
/// order. The golden's `# ` lines are commentary.
#[cfg(all(feature = "clap", feature = "serde"))]
fn golden_lines(prefixes: &[&str]) -> Vec<&'static str> {
  include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lifted_observables_812da2d.txt"
  ))
  .lines()
  .filter(|line| !line.starts_with("# "))
  .filter(|line| prefixes.iter().any(|prefix| line.starts_with(prefix)))
  .collect()
}

/// Every mismatch between a rendering and its golden lines, so a failure names
/// all the moved observables at once.
#[cfg(all(feature = "clap", feature = "serde"))]
fn mismatches(rendered: &[String], golden: &[&str]) -> Vec<String> {
  let mut out = Vec::new();
  if rendered.len() != golden.len() {
    out.push(format!(
      "line count: rendered {}, golden {}",
      rendered.len(),
      golden.len()
    ));
  }
  for (index, (got, want)) in rendered.iter().zip(golden).enumerate() {
    if got != want {
      out.push(format!(
        "line {index}\n  rendered: {got}\n  golden:   {want}"
      ));
    }
  }
  out
}

/// Every observable the selector lift moved is what the build before the lift
/// printed: the `Debug`, JSON, postcard and TOML spelling of each kind, the
/// refusal of the old variant spellings and the options' JSON and missing-field
/// refusal.
#[cfg(all(feature = "clap", feature = "serde"))]
#[test]
fn the_aggregate_observables_are_the_pre_lift_observables() {
  use crate::embeddings::clap;

  let mut rendered = Vec::new();
  for kind in [
    clap::AggregatePolicyKind::MeanRenormalized,
    clap::aggregate::AggregatePolicyKind::EmaRenormalized(clap::EmaRenormalizedOptions::new(0.5)),
    clap::AggregatePolicyKind::CoverageWeightedMean,
  ] {
    rendered.push(format!(
      "AggregatePolicyKind: json={} | debug={kind:?} | postcard={:?} | toml={:?}",
      serde_json::to_string(&kind).unwrap(),
      postcard::to_allocvec(&kind).unwrap(),
      toml::to_string(&std::collections::BTreeMap::from([("fold", kind)])).ok(),
    ));
  }
  rendered.push(format!(
    "AggregatePolicyKind refusals: {} | {}",
    serde_json::from_str::<clap::AggregatePolicyKind>(r#""MeanRenormalized""#).unwrap_err(),
    serde_json::from_str::<clap::AggregatePolicyKind>(r#"{"EmaRenormalized":{"alpha":0.5}}"#)
      .unwrap_err(),
  ));
  rendered.push(format!(
    "EmaRenormalizedOptions: json={} | missing alpha: {}",
    serde_json::to_string(&clap::aggregate::EmaRenormalizedOptions::new(0.3)).unwrap(),
    serde_json::from_str::<clap::EmaRenormalizedOptions>("{}").unwrap_err(),
  ));

  let golden = golden_lines(&["AggregatePolicyKind", "EmaRenormalizedOptions"]);
  assert_eq!(golden.len(), 5, "the aggregate law covers 5 golden lines");
  let moved = mismatches(&rendered, &golden);
  assert!(
    moved.is_empty(),
    "{} observable(s) differ from the pre-lift golden:\n{}",
    moved.len(),
    moved.join("\n")
  );
}

/// The `fold=` spelling derivation fingerprints persist: each kind's snake_case
/// name, and an EMA's factor through `f64`'s own shortest round-trip
/// `Display`.
#[test]
fn the_fold_spelling_is_pinned() {
  let cases: [(AggregatePolicyKind, &str); 6] = [
    (
      AggregatePolicyKind::CoverageWeightedMean,
      "coverage_weighted_mean",
    ),
    (AggregatePolicyKind::MeanRenormalized, "mean_renormalized"),
    (
      AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.5)),
      "ema_renormalized(alpha=0.5)",
    ),
    (
      AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.1 + 0.2)),
      "ema_renormalized(alpha=0.30000000000000004)",
    ),
    (
      AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(1.0)),
      "ema_renormalized(alpha=1)",
    ),
    (
      AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(1e-7)),
      "ema_renormalized(alpha=0.0000001)",
    ),
  ];
  for (kind, spelled) in cases {
    assert_eq!(kind.to_string(), spelled);
  }
}

/// Codex round 1, finding 7: the EMA options refuse an unknown key, so a
/// misspelled `alpha` beside the real one is an error, never silently
/// ignored, whether the options are read alone or inside the selector.
#[cfg(feature = "serde")]
#[test]
fn the_ema_options_refuse_an_unknown_key() {
  let alone = serde_json::from_str::<EmaRenormalizedOptions>(r#"{"alpha":0.5,"alhpa":0.8}"#)
    .unwrap_err()
    .to_string();
  assert!(alone.contains("alhpa"), "{alone}");
  let kind = serde_json::from_str::<AggregatePolicyKind>(
    r#"{"ema_renormalized":{"alpha":0.5,"alhpa":0.8}}"#,
  )
  .unwrap_err()
  .to_string();
  assert!(kind.contains("alhpa"), "{kind}");
}
