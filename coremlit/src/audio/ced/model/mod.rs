//! The CED model-size vocabulary ([`CedModel`]) and its parse error.
//!
//! CED ships in four sizes that are **contract-identical for coremlit**: they
//! share one mel→logits I/O shape and every family constant the parent module
//! pins ([`crate::audio::ced::SAMPLE_RATE_HZ`],
//! [`crate::audio::ced::WINDOW_SAMPLES`], [`crate::audio::ced::NUM_CLASSES`],
//! the mel geometry), differing only in internal transformer width — which the
//! loaded `.mlmodelc` never exposes. [`CedModel`] is therefore a
//! naming/metadata vocabulary (repo ids and staging paths), not a runtime
//! switch: [`crate::audio::ced::Classifier`] stays size-agnostic and coremlit
//! cannot — and does not — detect which size a graph is.

use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

/// One of CED's four published sizes (`mispeech/ced-{tiny,mini,small,base}`).
///
/// **Size-invariant I/O.** All four graphs consume the believed
/// `mel [1, 64, 1001]` f32 and emit `logits [1, `[`crate::audio::ced::NUM_CLASSES`]`]`
/// f32 (upstream `target_length = 1012` is the transformer's pos-embed
/// capacity, not an input width — see the `mel` submodule). They differ ONLY
/// in internal transformer width, which the compiled artifact does not expose,
/// so this enum deliberately has NO `embed_dim`/`heads` accessor: coremlit
/// measures what a gate can pin and does not market a dimension it cannot
/// verify. For provenance the widths are (shared `depth = 12`; source: the four
/// `mispeech/ced-*` `config.json`):
///
/// | size  | embed_dim | num_heads |
/// |-------|-----------|-----------|
/// | tiny  | 192       | 3         |
/// | mini  | 256       | 4         |
/// | small | 384       | 6         |
/// | base  | 768       | 12        |
///
/// Because the sizes are indistinguishable at the I/O boundary, model identity
/// is **caller-supplied**: point [`crate::audio::ced::Classifier::from_file`]
/// at the size you staged and compose its path with [`Self::mlmodelc_path`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum CedModel {
  /// `ced-tiny`: embed_dim 192, 3 heads.
  Tiny,
  /// `ced-mini`: embed_dim 256, 4 heads.
  Mini,
  /// `ced-small`: embed_dim 384, 6 heads.
  Small,
  /// `ced-base`: embed_dim 768, 12 heads.
  Base,
}

impl CedModel {
  /// Every [`CedModel`] size, ascending in transformer width. Handy for the
  /// per-size test matrices; the closed enum keeps every metadata table total.
  pub const ALL: [CedModel; 4] = [Self::Tiny, Self::Mini, Self::Small, Self::Base];

  /// Stable snake_case size name (`"tiny"`/`"mini"`/`"small"`/`"base"`) — the
  /// single source shared by [`Display`](core::fmt::Display), serde, and
  /// [`FromStr`](core::str::FromStr).
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Tiny => "tiny",
      Self::Mini => "mini",
      Self::Small => "small",
      Self::Base => "base",
    }
  }

  /// Upstream Hugging Face repo id for this size (`"mispeech/ced-<size>"`) —
  /// the conversion source and the goldens' oracle provenance.
  #[inline(always)]
  pub const fn hf_repo(&self) -> &'static str {
    match self {
      Self::Tiny => "mispeech/ced-tiny",
      Self::Mini => "mispeech/ced-mini",
      Self::Small => "mispeech/ced-small",
      Self::Base => "mispeech/ced-base",
    }
  }

  /// Staging/distribution directory name — **hyphenated** (`"ced-<size>"`),
  /// the per-size parent under the `Models/ced` family root.
  #[inline(always)]
  pub const fn dir_name(&self) -> &'static str {
    match self {
      Self::Tiny => "ced-tiny",
      Self::Mini => "ced-mini",
      Self::Small => "ced-small",
      Self::Base => "ced-base",
    }
  }

  /// Compiled bundle name — **underscored** stem (`"ced_<size>.mlmodelc"`).
  /// The hyphen/underscore split ([`Self::dir_name`] vs this) is the exact
  /// spelling Wave A committed for tiny, frozen here once for all four.
  #[inline(always)]
  pub const fn mlmodelc_name(&self) -> &'static str {
    match self {
      Self::Tiny => "ced_tiny.mlmodelc",
      Self::Mini => "ced_mini.mlmodelc",
      Self::Small => "ced_small.mlmodelc",
      Self::Base => "ced_base.mlmodelc",
    }
  }

  /// Composes this size's compiled-graph path under `models_root`:
  /// `models_root/`[`dir_name`](Self::dir_name)`/`[`mlmodelc_name`](Self::mlmodelc_name).
  /// Pure path arithmetic — no filesystem access, so it serves both callers
  /// (resolving a staged bundle) and the hermetic path pins.
  pub fn mlmodelc_path(&self, models_root: impl AsRef<Path>) -> PathBuf {
    models_root
      .as_ref()
      .join(self.dir_name())
      .join(self.mlmodelc_name())
  }
}

/// Error returned when a string is not one of [`CedModel`]'s four size names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown CED model name")]
pub struct ParseCedModelError(());

impl core::str::FromStr for CedModel {
  type Err = ParseCedModelError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "tiny" => Self::Tiny,
      "mini" => Self::Mini,
      "small" => Self::Small,
      "base" => Self::Base,
      _ => return Err(ParseCedModelError(())),
    })
  }
}
