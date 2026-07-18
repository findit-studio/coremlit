//! Shared serde bridge for [`coremlit::ComputeUnits`] (crate-private, `serde`
//! feature only).
//!
//! `coremlit::ComputeUnits` carries no serde impl of its own, so bridge it
//! through its existing `as_str`/`FromStr` — the same shape speakerkit's and
//! whisperkit's private compute-units bridges use. Used by
//! [`crate::audio::AudioEncoderOptions`] and [`crate::text::TextEncoderOptions`]
//! via `serde(with = "crate::compute_units_serde")`.

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
