//! The kit's one naming rule for a compiled bundle: `<Name>.<id>.mlmodelc`,
//! compiled from `<Name>.mlpackage`, where `<id>` is the first 12 lowercase
//! hex digits of the package's manifest digest.
//!
//! Standard library only, with no path into the rest of the crate: the licence
//! register (`tests/model_licences.rs`) includes this same file to resolve a
//! compiled bundle the fp16 sweep pins to the package it was compiled from, so
//! the rule exists once.

/// The width of a compiled bundle's id, in hex digits.
pub(crate) const ID_HEX: usize = 12;

/// The name of the bundle `name`'s package compiles to under `id`.
pub(crate) fn compiled_bundle(name: &str, id: &str) -> String {
  format!("{name}.{id}.mlmodelc")
}

/// The `(name, id)` of a compiled bundle, or `None` when `bundle` is not one:
/// no `.mlmodelc` suffix, an empty name, or an id that is not [`ID_HEX`]
/// lowercase hex digits.
pub(crate) fn parse_compiled_bundle(bundle: &str) -> Option<(&str, &str)> {
  let (name, id) = bundle.strip_suffix(".mlmodelc")?.rsplit_once('.')?;
  let hex = id.len() == ID_HEX
    && id
      .bytes()
      .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
  (!name.is_empty() && hex).then_some((name, id))
}

/// The package a compiled bundle of `name` is compiled from.
pub(crate) fn source_package(name: &str) -> String {
  format!("{name}.mlpackage")
}

#[cfg(test)]
mod tests;
