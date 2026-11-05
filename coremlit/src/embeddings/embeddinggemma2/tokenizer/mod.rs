//! The tokenizer: Gemma's BPE over a mapped, versioned image of its tables,
//! compiled at staging from `tokenizer.json`, and how the 24 special literals
//! in a text are read.

pub(crate) mod bpe;
pub(crate) mod compile;
pub(crate) mod image;

use super::{
  error::{Error, ImageField, Result, TokenizerImageDefect},
  prompt::Prompt,
};
use bpe::{Encoded, TokenizerScratch};
use image::TokenizerImage;

/// One encoding an open checks: the image must encode `prompt ⧺ text` under
/// `split` to exactly `ids`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sentinel {
  pub(crate) prompt: Prompt<'static>,
  pub(crate) text: &'static str,
  pub(crate) split: SplitSpecialTokens,
  pub(crate) ids: &'static [u32],
}

/// The five encodings an open of the upstream image checks, the special
/// literal under both settings: the document fixture, the empty text, a
/// byte-fallback string, `<|image|>`, and `<b> </b>`, whose one bridging merge
/// makes the bridging path run at every open. Hugging Face's ids, from the
/// committed goldens.
pub(crate) const UPSTREAM_SENTINELS: [Sentinel; 6] = [
  Sentinel {
    prompt: Prompt::Document { title: None },
    text: "Café crème, naïve résumé",
    split: SplitSpecialTokens::Recognize,
    ids: &[
      2, 3250, 236787, 7293, 1109, 1816, 236787, 62367, 100479, 236764, 120362, 118515, 1,
    ],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "x\u{f0000}y",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 236781, 481, 414, 366, 366, 236762, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "<|image|>",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 258880, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "<|image|>",
    split: SplitSpecialTokens::Split,
    ids: &[2, 236820, 236909, 3304, 111038, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "<b> </b>",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 236820, 236763, 107068, 236763, 236813, 1],
  },
];

impl TokenizerImage {
  /// Encodes each sentinel and compares its ids. A few pages of the image,
  /// microseconds.
  ///
  /// # Errors
  /// [`Error::TokenizerImage`] naming [`ImageField::Sentinel`] and the first
  /// encoding that differs.
  pub(crate) fn check_sentinels(
    &self,
    sentinels: &[Sentinel],
    scratch: &mut TokenizerScratch,
  ) -> Result<()> {
    let mut ids = Vec::new();
    for sentinel in sentinels {
      ids.clear();
      self.encode_joint(
        &sentinel.prompt,
        sentinel.text,
        sentinel.split,
        scratch,
        Encoded {
          ids: &mut ids,
          spans: None,
        },
      )?;
      let got: Vec<u32> = ids.iter().map(|id| id.get()).collect();
      if got != sentinel.ids {
        return Err(Error::TokenizerImage(TokenizerImageDefect::new(
          ImageField::Sentinel,
          format!(
            "{:?} for {:?} under {}",
            sentinel.ids, sentinel.text, sentinel.split
          ),
          format!("{got:?}"),
        )));
      }
    }
    Ok(())
  }
}

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
