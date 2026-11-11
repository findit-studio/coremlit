use super::*;

const TASKS: [Prompt<'static>; 7] = [
  Prompt::SearchQuery,
  Prompt::QuestionAnswering,
  Prompt::FactChecking,
  Prompt::CodeRetrieval,
  Prompt::Classification,
  Prompt::Clustering,
  Prompt::SentenceSimilarity,
];

/// The eight named prompts are `config.json`'s keys, in its order; the other
/// forms carry none.
#[test]
fn the_named_prompts_are_config_keys_and_the_others_are_not() {
  let named: Vec<&str> = core::iter::once(Prompt::Document(None))
    .chain(TASKS)
    .map(|prompt| prompt.config_key().expect("a named prompt"))
    .collect();
  let keys: Vec<&str> = Prompt::NAMED.iter().map(|(key, _)| *key).collect();
  assert_eq!(named, keys);
  for prompt in [
    Prompt::Document(Some("Q3 earnings call")),
    Prompt::Bare,
    Prompt::Custom("task: legal clause retrieval | query: "),
  ] {
    assert_eq!(prompt.config_key(), None, "{prompt}");
  }
}

/// The derivation spelling: a named prompt prints its key, `Bare` prints
/// `bare`, and a title or custom prefix prints verbatim, escaped as `{:?}`
/// escapes a `str`.
#[test]
fn the_derivation_spelling_is_pinned() {
  let cases: [(Prompt<'_>, &str); 7] = [
    (Prompt::Document(None), "document"),
    (Prompt::SearchQuery, "search_query"),
    (Prompt::SentenceSimilarity, "sentence_similarity"),
    (Prompt::Bare, "bare"),
    (
      Prompt::Document(Some("Q3 earnings call")),
      r#"document(title="Q3 earnings call")"#,
    ),
    (
      Prompt::Custom("task: legal clause retrieval | query: "),
      r#"custom("task: legal clause retrieval | query: ")"#,
    ),
    (
      Prompt::Custom("a \"quoted\" \\ path\n"),
      r#"custom("a \"quoted\" \\ path\n")"#,
    ),
  ];
  for (prompt, spelled) in cases {
    assert_eq!(prompt.to_string(), spelled);
  }
}
