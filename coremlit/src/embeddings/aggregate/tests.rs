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
