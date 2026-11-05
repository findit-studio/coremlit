//! The tokenizer: Gemma's BPE over a mapped, versioned image of its tables,
//! compiled at staging from `tokenizer.json`, and how the 24 special literals
//! in a text are read.

pub(crate) mod compile;
pub(crate) mod image;

/// Whether a special token's literal in the text (`<|image>`, `<eos>`, …) is
/// read as that special token or as plain text.
///
/// The typed face of the `split_special_tokens` option, named as Hugging Face
/// names it: `false` is [`Self::Recognize`], `true` is [`Self::Split`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum SplitSpecialTokens {
  /// The literal is the special token's id, as Hugging Face and
  /// sentence-transformers read it (`split_special_tokens = false`). The
  /// default.
  Recognize,
  /// The literal is split as plain text (`split_special_tokens = true`).
  Split,
}

impl SplitSpecialTokens {
  /// The behaviour the boolean option names: `true` splits, `false`
  /// recognizes.
  #[inline]
  pub const fn from_split(split: bool) -> Self {
    if split { Self::Split } else { Self::Recognize }
  }

  /// The boolean option's value: `true` for [`Self::Split`].
  #[inline]
  pub const fn is_split(&self) -> bool {
    matches!(self, Self::Split)
  }

  /// Stable snake_case name of the behaviour: `"recognize"` or `"split"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Recognize => "recognize",
      Self::Split => "split",
    }
  }
}

#[cfg(test)]
mod tests;
