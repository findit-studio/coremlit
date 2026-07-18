//! Serde bridge for `coremlit::ComputeUnits` (crate-private).
//!
//! `coremlit::ComputeUnits` carries no serde impl of its own (coremlit has
//! no serde dependency at all) — bridge it through its existing
//! `as_str`/`FromStr`, the same shape `dia-coreml`'s private
//! `compute_units_serde` module uses
//! (`crates/dia-coreml/src/compute_units_serde.rs`, itself mirroring
//! whisperkit's private `options::compute_units_serde`). Used by
//! [`crate::encode::EncoderOptions`] via
//! `serde(with = "crate::compute_units_serde")`.

use core::str::FromStr;

use coremlit::ComputeUnits;
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
