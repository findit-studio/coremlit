//! Prompts: the text a query or a document is prefixed with before it is
//! tokenized.

/// The prefix a text is embedded under.
///
/// The eight named prompts are `config.json`'s `prompts` map, checked at load
/// byte for byte against [`Self::NAMED`]. [`Self::Document`] with a title,
/// [`Self::Bare`] and [`Self::Custom`] are the other forms the model card
/// describes; none of them is checked against `config.json`.
///
/// # Prompt and text are one string
///
/// The ids of a prompt and a text are the encoding of their concatenation as
/// ONE string, never the sum of two encodings. The eight prompts end in a
/// space, whose `▁` merges into the text's first word (`▁Café`); a custom
/// prefix can end in anything, and its last piece may merge with the text's
/// first. A text embedded in several windows carries the prompt in every one.
///
/// # Text only
///
/// A prompt applies to text. Audio and image tokens are embedded without one,
/// by signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Prompt<'a> {
  /// A document; the payload is its optional title.
  ///
  /// - `Prompt::Document(None)` is `"title: none | text: "`, `config.json`'s
  ///   `document` prompt.
  /// - `Prompt::Document(Some("Q3 earnings call"))` is
  ///   `"title: Q3 earnings call | text: "`.
  ///
  /// Its derivation term spells the title `document(title="…")`.
  Document(Option<&'a str>),
  /// `"task: search result | query: "`: a search query.
  SearchQuery,
  /// `"task: question answering | query: "`.
  QuestionAnswering,
  /// `"task: fact checking | query: "`.
  FactChecking,
  /// `"task: code retrieval | query: "`.
  CodeRetrieval,
  /// `"task: classification | query: "`.
  Classification,
  /// `"task: clustering | query: "`.
  Clustering,
  /// `"task: sentence similarity | query: "`.
  SentenceSimilarity,
  /// No prefix. The model card: it works, and is less precise.
  Bare,
  /// A caller's own prefix, tokenized jointly with the text and repeated in
  /// every window exactly like the eight. For example
  /// `"task: legal clause retrieval | query: "`.
  Custom(&'a str),
}

impl<'a> Prompt<'a> {
  /// `config.json`'s eight prompts as `(key, text)`, in its order. The kit
  /// checks the loaded `config.json` against exactly these.
  pub const NAMED: [(&'static str, &'static str); 8] = [
    ("document", "title: none | text: "),
    ("search_query", "task: search result | query: "),
    ("question_answering", "task: question answering | query: "),
    ("fact_checking", "task: fact checking | query: "),
    ("code_retrieval", "task: code retrieval | query: "),
    ("classification", "task: classification | query: "),
    ("clustering", "task: clustering | query: "),
    ("sentence_similarity", "task: sentence similarity | query: "),
  ];

  /// The prompt's key in `config.json`'s `prompts` map, or `None` for a
  /// titled document, [`Self::Bare`] and [`Self::Custom`], which it does not
  /// carry.
  #[inline]
  pub const fn config_key(&self) -> Option<&'static str> {
    match self.named_index() {
      Some(index) => Some(Self::NAMED[index].0),
      None => None,
    }
  }

  /// The prefix as up to three pieces whose concatenation is the prompt's
  /// text. A titled document is `["title: ", title, " | text: "]`; every other
  /// form is one piece and two empty ones.
  pub(crate) const fn pieces(&self) -> [&'a str; 3] {
    match (self, self.named_index()) {
      (_, Some(index)) => [Self::NAMED[index].1, "", ""],
      (Self::Document(Some(title)), None) => ["title: ", title, " | text: "],
      (Self::Custom(prefix), None) => [prefix, "", ""],
      _ => ["", "", ""],
    }
  }

  /// The index of a named prompt in [`Self::NAMED`].
  const fn named_index(&self) -> Option<usize> {
    match self {
      Self::Document(None) => Some(0),
      Self::SearchQuery => Some(1),
      Self::QuestionAnswering => Some(2),
      Self::FactChecking => Some(3),
      Self::CodeRetrieval => Some(4),
      Self::Classification => Some(5),
      Self::Clustering => Some(6),
      Self::SentenceSimilarity => Some(7),
      Self::Document(Some(_)) | Self::Bare | Self::Custom(_) => None,
    }
  }
}

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a major bump.**
///
/// A named prompt prints its `config.json` key (`document`, `search_query`, …),
/// and [`Prompt::Bare`] prints `bare`. A titled document prints
/// `document(title="…")` and a custom prefix `custom("…")`, each string
/// verbatim, with `"` and `\` escaped the way Rust's `{:?}` escapes a `str`:
/// `Prompt::Custom("task: legal clause retrieval | query: ")` prints
/// `custom("task: legal clause retrieval | query: ")`.
impl core::fmt::Display for Prompt<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match (self, self.config_key()) {
      (_, Some(key)) => f.write_str(key),
      (Self::Document(Some(title)), None) => write!(f, "document(title={title:?})"),
      (Self::Custom(prefix), None) => write!(f, "custom({prefix:?})"),
      _ => f.write_str("bare"),
    }
  }
}

#[cfg(test)]
mod tests;
