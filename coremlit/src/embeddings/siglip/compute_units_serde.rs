//! Shared serde bridge for [`crate::ComputeUnits`] (module-private, `serde`
//! feature only).
//!
//! `crate::ComputeUnits` carries no serde impl of its own, so bridge it
//! through its existing `as_str`/`FromStr` — the same shape the `granite` and
//! `clap` modules' private compute-units bridges use. Used by
//! [`crate::embeddings::siglip::ImageEmbedderOptions`] and
//! [`crate::embeddings::siglip::TextEmbedderOptions`] via
//! `serde(with = "crate::embeddings::siglip::compute_units_serde")`.

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
