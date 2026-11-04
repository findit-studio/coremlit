use super::*;

/// The shared selector keeps resolving at the clap paths it was first public at,
/// and both paths name the one type.
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
#[cfg(feature = "serde")]
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
#[cfg(feature = "serde")]
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
#[cfg(feature = "serde")]
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
