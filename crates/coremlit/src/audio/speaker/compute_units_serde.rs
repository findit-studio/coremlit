//! Shared serde bridge for `crate::ComputeUnits` (crate-private).
//!
//! `crate::ComputeUnits` carries no serde impl of its own (coremlit has
//! no serde dependency at all) — bridge it through its existing
//! `as_str`/`FromStr`, the same shape whisperkit's private
//! `options::compute_units_serde` module uses
//! (`crates/whisperkit/src/options/mod.rs`). Used by both
//! [`crate::audio::speaker::segment::SegmentModelOptions`] and
//! [`crate::audio::speaker::embed::EmbedModelOptions`] via `serde(with = "crate::audio::speaker::compute_units_serde")`.
//!
//! Originally a private `segment`-local module (T2); T3's review queue
//! flagged that a second per-module copy for `EmbedModelOptions` would be a
//! *third* copy of this exact bridge counting whisperkit's — factored out
//! here instead so both option types (and any future one) share it.
//! `default_segment_compute`/`default_embed_compute` stay module-local:
//! each returns a different per-type default and has nothing to share.

use core::str::FromStr;

use crate::ComputeUnits;
use serde::{Deserialize, Deserializer, Serializer};

pub(crate) fn serialize<S: Serializer>(
  value: &ComputeUnits,
  serializer: S,
) -> Result<S::Ok, S::Error> {
  serializer.serialize_str(value.as_str())
}

pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
  deserializer: D,
) -> Result<ComputeUnits, D::Error> {
  let name = String::deserialize(deserializer)?;
  ComputeUnits::from_str(&name).map_err(serde::de::Error::custom)
}
