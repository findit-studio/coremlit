use super::*;

/// A minimal WordLevel tokenizer around `post_processor` — the guard reads only
/// the post-processor, so the model underneath just has to parse.
fn tokenizer_with(post_processor: &str) -> Tokenizer {
  let json = format!(
    r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":{post_processor},"decoder":null,"model":{{"type":"WordLevel","vocab":{{"<pad>":0,"a":1,"b":2}},"unk_token":"<pad>"}}}}"#
  );
  Tokenizer::from_bytes(json.as_bytes()).expect("the fixture tokenizer must parse")
}

/// The template shape a `tokenizer.json` carries, with `single` and
/// `special_tokens` supplied per case. `pair` is always well-formed: these rules
/// are about the SINGLE template, which is the only one the doors apply.
fn template(single: &str, special_tokens: &str) -> String {
  format!(
    r#"{{"type":"TemplateProcessing","single":{single},"pair":[{{"Sequence":{{"id":"A","type_id":0}}}}],"special_tokens":{special_tokens}}}"#
  )
}

const SPECIAL_A: &str = r#"{"a":{"id":"a","ids":[1],"tokens":["a"]}}"#;
const PIECE_A: &str = r#"{"Sequence":{"id":"A","type_id":0}}"#;
const PIECE_B: &str = r#"{"Sequence":{"id":"B","type_id":1}}"#;
const PIECE_SPECIAL_A: &str = r#"{"SpecialToken":{"id":"a","type_id":0}}"#;
const PIECE_SPECIAL_MISSING: &str = r#"{"SpecialToken":{"id":"<s>","type_id":0}}"#;

/// `Sequence` and `ByteLevel` post-processor bodies used to build chains below.
const BYTE_LEVEL: &str =
  r#"{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#;
const ROBERTA: &str = r#"{"type":"RobertaProcessing","sep":["b",2],"cls":["a",1],"trim_offsets":true,"add_prefix_space":false}"#;

/// A post-processor `Sequence` over `members` (already-rendered JSON).
fn sequence(members: &[&str]) -> String {
  format!(
    r#"{{"type":"Sequence","processors":[{}]}}"#,
    members.join(",")
  )
}

/// A `TemplateProcessing` whose PAIR template is supplied too — the chain cases
/// below reach it, which [`template`]'s always-well-formed pair cannot express.
fn template_with_pair(single: &str, pair: &str, special_tokens: &str) -> String {
  format!(
    r#"{{"type":"TemplateProcessing","single":{single},"pair":{pair},"special_tokens":{special_tokens}}}"#
  )
}

/// A single template of `n` `$A` pieces: it places the input sequence `n` times
/// and so emits `n` encodings, each a copy of the text.
fn template_of_n_inputs(n: usize) -> String {
  let pieces: Vec<&str> = std::iter::repeat_n(PIECE_A, n).collect();
  template(&format!("[{}]", pieces.join(",")), "{}")
}

/// `[CLS] $A [SEP]` — three pieces, both specials declared.
fn cls_a_sep() -> String {
  template(
    &format!("[{PIECE_SPECIAL_A},{PIECE_A},{PIECE_SPECIAL_A}]"),
    SPECIAL_A,
  )
}

/// Non-vacuity: every defective fixture below really does reach the dependency's
/// panic (or its silent drop) when the guard is not consulted. Without this the
/// refusals could all be passing against templates that were never dangerous.
///
/// The two panicking cases are asserted through `catch_unwind`; the third is not
/// a panic at all but a wrong answer — the text is dropped and the encoding is
/// the special tokens alone.
#[test]
fn the_defective_fixtures_are_really_defective() {
  for single in [
    format!("[{PIECE_SPECIAL_MISSING},{PIECE_A}]"),
    format!("[{PIECE_A},{PIECE_B}]"),
  ] {
    let tokenizer = tokenizer_with(&template(&single, "{}"));
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = tokenizer.encode("a b", true);
    }))
    .is_err();
    assert!(panicked, "`{single}` must panic inside the dependency");
  }

  let tokenizer = tokenizer_with(&template(&format!("[{PIECE_SPECIAL_A}]"), SPECIAL_A));
  let ids = tokenizer
    .encode("a b a b", true)
    .expect("a template with no $A does not panic — it drops the text")
    .get_ids()
    .to_vec();
  assert_eq!(
    ids,
    vec![1],
    "the text is gone; only the special token remains"
  );
}

#[test]
fn a_tokenizer_without_a_post_processor_passes() {
  assert_eq!(check_post_processor(&tokenizer_with("null")), Ok(()));
}

/// The three template-free kinds. `RobertaProcessing` is the shape CLAP's
/// bundled tokenizer actually carries, so this is also the guard's
/// does-not-fire-on-production case.
#[test]
fn template_free_post_processors_pass() {
  for post in [
    r#"{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":false}"#,
    r#"{"type":"BertProcessing","sep":["[SEP]",2],"cls":["[CLS]",0]}"#,
    r#"{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#,
  ] {
    assert_eq!(
      check_post_processor(&tokenizer_with(post)),
      Ok(()),
      "{post} carries no template"
    );
  }
}

#[test]
fn a_well_formed_single_template_passes() {
  let post = template(&format!("[{PIECE_SPECIAL_A},{PIECE_A}]"), SPECIAL_A);
  assert_eq!(check_post_processor(&tokenizer_with(&post)), Ok(()));
}

#[test]
fn an_undeclared_special_token_is_refused_and_named() {
  let post = template(&format!("[{PIECE_SPECIAL_MISSING},{PIECE_A}]"), "{}");
  assert_eq!(
    check_post_processor(&tokenizer_with(&post)),
    Err(PostProcessorTemplate::UndeclaredSpecialToken(
      "<s>".to_string()
    ))
  );
}

/// The rule is membership in `special_tokens`, not "the map is non-empty": a
/// template that declares one id and uses another is still refused, naming the
/// one it used.
#[test]
fn an_undeclared_id_beside_a_declared_one_is_refused() {
  let post = template(
    &format!("[{PIECE_SPECIAL_A},{PIECE_SPECIAL_MISSING},{PIECE_A}]"),
    SPECIAL_A,
  );
  assert_eq!(
    check_post_processor(&tokenizer_with(&post)),
    Err(PostProcessorTemplate::UndeclaredSpecialToken(
      "<s>".to_string()
    ))
  );
}

#[test]
fn a_pair_sequence_in_the_single_template_is_refused() {
  let post = template(&format!("[{PIECE_A},{PIECE_B}]"), "{}");
  assert_eq!(
    check_post_processor(&tokenizer_with(&post)),
    Err(PostProcessorTemplate::PairSequenceInSingleTemplate)
  );
}

#[test]
fn a_single_template_that_never_places_the_text_is_refused() {
  let post = template(&format!("[{PIECE_SPECIAL_A}]"), SPECIAL_A);
  assert_eq!(
    check_post_processor(&tokenizer_with(&post)),
    Err(PostProcessorTemplate::NoInputSequenceInSingleTemplate)
  );
}

/// An empty single template places no text either — the same refusal, reached
/// through the shape rather than through a special token.
#[test]
fn an_empty_single_template_is_refused() {
  let post = template("[]", "{}");
  assert_eq!(
    check_post_processor(&tokenizer_with(&post)),
    Err(PostProcessorTemplate::NoInputSequenceInSingleTemplate)
  );
}

/// At a count of ONE the PAIR template is not judged: the dependency applies the
/// `single` one, and a `$B` in the pair template is what a correct pair template
/// looks like. `template`'s pair is well-formed above, so this pins the
/// converse — an unreached pair template that would fail every rule leaves the
/// guard silent. Nothing here reaches a pair template any more: a count of two
/// is refused before its template is read ([`a_template_fed_two_encodings_is_refused_with_the_count`]),
/// so `pair` is dead weight in every tokenizer this guard admits.
#[test]
fn the_pair_template_is_not_judged() {
  let post = format!(
    r#"{{"type":"TemplateProcessing","single":[{PIECE_A}],"pair":[{PIECE_SPECIAL_MISSING},{PIECE_B}],"special_tokens":{{}}}}"#
  );
  assert_eq!(check_post_processor(&tokenizer_with(&post)), Ok(()));
}

/// A `Sequence` post-processor applies each member, so the guard must recurse
/// into it — a defective template nested one level down is refused with the
/// same reason it would carry on its own.
#[test]
fn a_defective_template_nested_in_a_sequence_is_refused() {
  let inner = template(&format!("[{PIECE_SPECIAL_MISSING},{PIECE_A}]"), "{}");
  let post = format!(r#"{{"type":"Sequence","processors":[{inner}]}}"#);
  assert_eq!(
    check_post_processor(&tokenizer_with(&post)),
    Err(PostProcessorTemplate::UndeclaredSpecialToken(
      "<s>".to_string()
    ))
  );
}

/// …and a `Sequence` of sound members passes, so the recursion is not a blanket
/// refusal of the kind.
#[test]
fn a_sequence_of_sound_processors_passes() {
  let inner = template(&format!("[{PIECE_SPECIAL_A},{PIECE_A}]"), SPECIAL_A);
  let post = format!(
    r#"{{"type":"Sequence","processors":[{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},{inner}]}}"#
  );
  assert_eq!(check_post_processor(&tokenizer_with(&post)), Ok(()));
}

/// Each reason renders its own sentence, and the undeclared-id one carries the
/// id — the doors interpolate this into their error message.
#[test]
fn every_reason_displays_distinctly() {
  let rendered = [
    PostProcessorTemplate::UndeclaredSpecialToken("<s>".to_string()).to_string(),
    PostProcessorTemplate::PairSequenceInSingleTemplate.to_string(),
    PostProcessorTemplate::NoInputSequenceInSingleTemplate.to_string(),
    PostProcessorTemplate::RepeatedInputSequence(2).to_string(),
    PostProcessorTemplate::UnsupportedEncodingCount(3).to_string(),
    PostProcessorTemplate::Unreadable.to_string(),
  ];
  assert!(rendered[0].contains("`<s>`"), "{}", rendered[0]);
  assert!(rendered[1].contains("$B"), "{}", rendered[1]);
  assert!(rendered[2].contains("$A"), "{}", rendered[2]);
  assert!(rendered[3].contains('2'), "{}", rendered[3]);
  assert!(rendered[4].contains('3'), "{}", rendered[4]);
  for (i, a) in rendered.iter().enumerate() {
    for b in &rendered[i + 1..] {
      assert_ne!(a, b, "each reason must render distinctly");
    }
  }
}

// ── Cardinality: how many encodings reach each post-processor ───────────────

/// `tokenizer` carrying the truncation policy every door installs — `window`,
/// `LongestFirst`, no stride, right direction — and the tokenizer's own padding
/// disabled. A length measured through this is the length `build_window` gets.
fn truncated_at(tokenizer: &Tokenizer, window: usize) -> Tokenizer {
  use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};
  let mut tokenizer = tokenizer.clone();
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: window,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .expect("the fixtures here never over-fill the window");
  tokenizer.with_padding(None);
  tokenizer
}

/// `post_processor.added_tokens(false)` — the ONE number `with_truncation`,
/// `post_process` and the doors' special-token-overhead guard are all sized on.
fn declared_overhead(tokenizer: &Tokenizer) -> usize {
  tokenizers::PostProcessor::added_tokens(
    tokenizer.get_post_processor().expect("the fixture has one"),
    false,
  )
}

/// Non-vacuity for the cardinality rule, the same way
/// [`the_defective_fixtures_are_really_defective`] does it for the single-
/// template rules. Both chains are built only from templates the per-template
/// rules ACCEPT, and each still panics inside the dependency on the first
/// `encode`, because the count reaching a later template is not one
/// `process_encodings` has an arm for.
///
/// Measured against tokenizers 0.23.1: both panic "not yet implemented" at
/// `processors/template.rs:681` — the `todo!()` arm of the `encodings.len()`
/// match.
#[test]
fn the_cardinality_fixtures_are_really_defective() {
  for chain in [
    sequence(&[&cls_a_sep(), &template(&format!("[{PIECE_A}]"), "{}")]),
    sequence(&[&template_of_n_inputs(3), &cls_a_sep()]),
  ] {
    let tokenizer = tokenizer_with(&chain);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = tokenizer.encode("a b", true);
    }));
    std::panic::set_hook(hook);
    let payload = outcome.expect_err(&format!("`{chain}` must panic inside the dependency"));
    let message = payload
      .downcast_ref::<String>()
      .cloned()
      .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
      .unwrap_or_default();
    assert_eq!(message, "not yet implemented", "for `{chain}`");
  }
}

/// The headline case: two templates that are each individually sound, chained.
/// The first applies a THREE-piece single template, so the second is handed
/// three encodings — the `todo!()` arm. The reason carries the count, because
/// neither template is itself at fault.
#[test]
fn a_template_fed_more_than_two_encodings_is_refused_with_the_count() {
  let chain = sequence(&[&cls_a_sep(), &template(&format!("[{PIECE_A}]"), "{}")]);
  assert_eq!(
    check_post_processor(&tokenizer_with(&chain)),
    Err(PostProcessorTemplate::UnsupportedEncodingCount(3)),
    "for `{chain}`"
  );
}

/// A count of ZERO has the same `todo!()` arm, and no longer has a chain that
/// reaches it: the walk starts at one, every template it admits hands on its
/// piece count (at least the one `$A` piece), and the passthrough kinds preserve
/// what they get, so nothing downstream of an ACCEPTED post-processor can be
/// handed none. The arm is kept because the guard fails closed on a count it
/// cannot reason about, and it is exercised where it lives.
#[test]
fn a_template_fed_no_encodings_is_refused_with_the_count() {
  let template: serde_json::Value =
    serde_json::from_str(&cls_a_sep()).expect("the fixture is JSON");
  assert_eq!(
    check_value(&template, 0),
    Err(PostProcessorTemplate::UnsupportedEncodingCount(0))
  );
}

// ── The text is placed exactly once ─────────────────────────────────────────

/// Non-vacuity for the repeat rule, MEASURED rather than argued: `$A` twice is
/// not a shape the guard merely dislikes.
///
/// `apply_template` emits one encoding per piece and `count_added` scores a
/// `Sequence` piece as zero, so `Template("$A $A")` advertises an overhead of
/// ZERO and then returns twice the text. Truncation runs BEFORE the
/// post-processor and is sized `max_length - added_tokens(false)` — on ONE copy
/// — so nothing bounds the result: at a four-token window a three-token text
/// still comes back as six ids, which is the door's `build_window` refusing
/// ordinary text with a typed `TokenCount` although construction succeeded.
#[test]
fn a_repeated_input_sequence_really_doubles_the_text() {
  let tokenizer = tokenizer_with(&template_of_n_inputs(2));
  assert_eq!(
    declared_overhead(&tokenizer),
    0,
    "a `Sequence` piece counts as no overhead, however many times it appears"
  );
  assert_eq!(
    tokenizer.encode("a b a", true).expect("encode").get_ids(),
    &[1, 2, 1, 1, 2, 1],
    "three text tokens in, six out"
  );
  let ids = truncated_at(&tokenizer, 4)
    .encode("a b a", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(
    ids,
    vec![1, 2, 1, 1, 2, 1],
    "truncation is sized for one copy, so it does not fire and the window is blown"
  );
  assert!(ids.len() > 4, "past a four-token window: {} ids", ids.len());
}

/// …and the doubling composes with a later template into the HIGH case: the
/// second template is handed two encodings, applies its `pair`, and if that pair
/// is empty or holds only special-token pieces the text is gone. The guard
/// answered `Ok(())` here, and `encode` answers the same ids for every input —
/// a silently constant embedding rather than a reported failure.
#[test]
fn a_doubled_text_folded_by_a_pair_template_erases_it() {
  let doubler = template_of_n_inputs(2);
  for (pair, expected) in [
    ("[]".to_string(), Vec::new()),
    (
      format!("[{PIECE_SPECIAL_A},{PIECE_SPECIAL_A}]"),
      vec![1u32, 1],
    ),
  ] {
    let chain = sequence(&[
      &doubler,
      &template_with_pair(&format!("[{PIECE_A}]"), &pair, SPECIAL_A),
    ]);
    let tokenizer = tokenizer_with(&chain);
    for text in ["some words", "a b", "a b a b a b"] {
      assert_eq!(
        tokenizer.encode(text, true).expect("encode").get_ids(),
        expected.as_slice(),
        "every input encodes identically under pair `{pair}`"
      );
    }
  }
}

/// The rule: a single template must place the input sequence EXACTLY once, and
/// the reason carries how many placements it really has.
#[test]
fn a_single_template_that_places_the_text_more_than_once_is_refused() {
  for n in [2usize, 3, 5] {
    assert_eq!(
      check_post_processor(&tokenizer_with(&template_of_n_inputs(n))),
      Err(PostProcessorTemplate::RepeatedInputSequence(n)),
    );
  }
}

/// …and in a chain the repeat is named where it IS. The walk is in application
/// order, so the earliest post-processor that breaks a rule is the one reported.
/// Both fixtures here also hand two encodings to their second member, but that
/// count exists only because the first template doubled the text: reporting the
/// count would name the consequence and point the caller at the wrong template.
#[test]
fn a_chain_whose_first_template_repeats_the_text_is_refused_there() {
  let doubler = template_of_n_inputs(2);
  for pair in [
    "[]".to_string(),
    format!("[{PIECE_SPECIAL_A},{PIECE_SPECIAL_A}]"),
  ] {
    let chain = sequence(&[
      &doubler,
      &template_with_pair(&format!("[{PIECE_A}]"), &pair, SPECIAL_A),
    ]);
    assert_eq!(
      check_post_processor(&tokenizer_with(&chain)),
      Err(PostProcessorTemplate::RepeatedInputSequence(2)),
      "for `{chain}`"
    );
  }
}

// ── Pair mode: the count the dependency HAS a template for ──────────────────

/// Non-vacuity for the count-of-two refusal, which is the one that is not a
/// panic — `process_encodings` has an arm for two and applies the `pair`
/// template happily. What is wrong is the arithmetic around it, and there are
/// two ways it shows.
///
/// A pair template that places neither `$A` nor `$B` erases the text: measured,
/// every input encodes to `[]` (an empty pair) or to the constant `[1, 1]`
/// (special-token pieces only), with the guard answering `Ok(())`.
///
/// A well-formed pair template keeps the text and breaks the LENGTH: what it
/// adds is `added_pair`, and `added_tokens(false)` reports `added_single`. Here
/// the chain declares an overhead of 1 and, at a four-token window, returns six
/// ids for any input.
#[test]
fn the_pair_mode_fixtures_are_really_defective() {
  let a_then_special = template(&format!("[{PIECE_A},{PIECE_SPECIAL_A}]"), SPECIAL_A);
  for (pair, expected) in [
    ("[]".to_string(), Vec::new()),
    (
      format!("[{PIECE_SPECIAL_A},{PIECE_SPECIAL_A}]"),
      vec![1u32, 1],
    ),
  ] {
    let chain = sequence(&[
      &a_then_special,
      &template_with_pair(&format!("[{PIECE_A}]"), &pair, SPECIAL_A),
    ]);
    let tokenizer = tokenizer_with(&chain);
    for text in ["some words", "a b", "a b a b a b"] {
      assert_eq!(
        tokenizer.encode(text, true).expect("encode").get_ids(),
        expected.as_slice(),
        "every input encodes identically under pair `{pair}`"
      );
    }
  }

  let chain = sequence(&[
    &a_then_special,
    &template_with_pair(
      &format!("[{PIECE_A}]"),
      &format!("[{PIECE_SPECIAL_A},{PIECE_A},{PIECE_B},{PIECE_SPECIAL_A}]"),
      SPECIAL_A,
    ),
  ]);
  let tokenizer = tokenizer_with(&chain);
  assert_eq!(
    declared_overhead(&tokenizer),
    1,
    "the sum of the members' `added_single`, whichever mode each really runs in"
  );
  let ids = truncated_at(&tokenizer, 4)
    .encode("a b a b a b", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(
    ids,
    vec![1, 1, 2, 1, 1, 1],
    "sized on an overhead of 1, the pair template adds 3"
  );
  assert!(ids.len() > 4, "past a four-token window: {} ids", ids.len());
}

/// So a template a chain would reach at TWO encodings is refused outright,
/// whatever its pair template says — the sound one in the third case included,
/// because what is unsound there is the truncation sized on `added_single` while
/// the template adds `added_pair`. No door here encodes a pair, so a chain that
/// manufactures one is not a shape any of them needs.
#[test]
fn a_template_fed_two_encodings_is_refused_with_the_count() {
  let a_then_special = template(&format!("[{PIECE_A},{PIECE_SPECIAL_A}]"), SPECIAL_A);
  for pair in [
    "[]".to_string(),
    format!("[{PIECE_SPECIAL_A},{PIECE_SPECIAL_A}]"),
    format!("[{PIECE_SPECIAL_A},{PIECE_A},{PIECE_B},{PIECE_SPECIAL_A}]"),
  ] {
    let chain = sequence(&[
      &a_then_special,
      &template_with_pair(&format!("[{PIECE_A}]"), &pair, SPECIAL_A),
    ]);
    assert_eq!(
      check_post_processor(&tokenizer_with(&chain)),
      Err(PostProcessorTemplate::UnsupportedEncodingCount(2)),
      "for pair `{pair}`"
    );
  }
}

// ── The template-free kinds ────────────────────────────────────────────────

/// Non-vacuity for the passthrough half of the count rule.
/// `RobertaProcessing` and `BertProcessing` map encodings 1:1 but they are not
/// free: they wrap EVERY encoding they are handed — `cls … sep` around the
/// first, `sep … sep` around each of the rest — while `added_tokens(false)`
/// reports a flat 2. Measured behind a three-piece template at a window of 8:
/// the declared overhead is 4, and Roberta returns 12 ids, Bert 10. `ByteLevel`
/// adds nothing at any count and returns exactly 8, so the rule below is about
/// the overhead and not about the kind.
#[test]
fn the_token_adding_passthroughs_under_count_their_overhead() {
  const BERT: &str = r#"{"type":"BertProcessing","sep":["b",2],"cls":["a",1]}"#;
  for (passthrough, declared, length) in [(ROBERTA, 4, 12), (BERT, 4, 10), (BYTE_LEVEL, 2, 8)] {
    let chain = sequence(&[&cls_a_sep(), passthrough]);
    let tokenizer = tokenizer_with(&chain);
    assert_eq!(declared_overhead(&tokenizer), declared, "for {passthrough}");
    let ids = truncated_at(&tokenizer, 8)
      .encode("a b a b a b", true)
      .expect("encode")
      .get_ids()
      .to_vec();
    assert_eq!(ids.len(), length, "for {passthrough}: {ids:?}");
  }
}

/// …so `RobertaProcessing` and `BertProcessing` must be reached at ONE encoding
/// too, and are refused with the count when they are not. `ByteLevel` is not:
/// it adds nothing at any count, which is why it can stand after a multi-piece
/// template and the other two cannot.
#[test]
fn the_token_adding_passthroughs_are_refused_past_one_encoding() {
  const BERT: &str = r#"{"type":"BertProcessing","sep":["b",2],"cls":["a",1]}"#;
  for passthrough in [ROBERTA, BERT] {
    let chain = sequence(&[&cls_a_sep(), passthrough]);
    assert_eq!(
      check_post_processor(&tokenizer_with(&chain)),
      Err(PostProcessorTemplate::UnsupportedEncodingCount(3)),
      "for {passthrough}"
    );
  }
  let chain = sequence(&[&cls_a_sep(), BYTE_LEVEL]);
  assert_eq!(
    check_post_processor(&tokenizer_with(&chain)),
    Ok(()),
    "`ByteLevel` adds nothing at any count"
  );
}

/// `ByteLevel` must PROPAGATE the count rather than reset it: it is placed
/// between a two-piece template and a third member that is only sound at a count
/// of one, so crediting it with handing on one encoding would miss the refusal.
#[test]
fn byte_level_propagates_the_count() {
  let chain = sequence(&[
    &template(&format!("[{PIECE_A},{PIECE_SPECIAL_A}]"), SPECIAL_A),
    BYTE_LEVEL,
    &template(&format!("[{PIECE_A}]"), "{}"),
  ]);
  assert_eq!(
    check_post_processor(&tokenizer_with(&chain)),
    Err(PostProcessorTemplate::UnsupportedEncodingCount(2))
  );
}

/// …and with the count held at one the same kinds pass, so the rules above are
/// count rules and not a refusal of those kinds in a chain.
#[test]
fn the_template_free_kinds_pass_a_sound_chain() {
  const BERT: &str = r#"{"type":"BertProcessing","sep":["b",2],"cls":["a",1]}"#;
  for passthrough in [ROBERTA, BERT, BYTE_LEVEL] {
    for chain in [
      sequence(&[passthrough]),
      sequence(&[passthrough, &template(&format!("[{PIECE_A}]"), "{}")]),
      sequence(&[&template(&format!("[{PIECE_A}]"), "{}"), passthrough]),
    ] {
      assert_eq!(
        check_post_processor(&tokenizer_with(&chain)),
        Ok(()),
        "for `{chain}`"
      );
    }
  }
}

/// A nested `Sequence` is not a fresh start: the count it produces is the count
/// its SIBLING receives. The inner sequence's two-piece template makes the outer
/// one's next member a template reached at two.
#[test]
fn a_nested_sequence_hands_its_count_to_its_sibling() {
  let inner = sequence(&[&template(
    &format!("[{PIECE_A},{PIECE_SPECIAL_A}]"),
    SPECIAL_A,
  )]);
  let chain = sequence(&[&inner, &template(&format!("[{PIECE_A}]"), "{}")]);
  assert_eq!(
    check_post_processor(&tokenizer_with(&chain)),
    Err(PostProcessorTemplate::UnsupportedEncodingCount(2))
  );
}

// ── What stays open ────────────────────────────────────────────────────────

/// The accepting side, pinned by IDS and not only by `Ok(())`: a chain that
/// keeps the count at one encodes exactly as the single template does on its
/// own. The identity template `$A` ahead of it, a `ByteLevel` ahead of it, and
/// the bare `Sequence` wrapper all leave the answer alone.
#[test]
fn a_chain_that_keeps_the_count_at_one_is_accepted_and_encodes() {
  let alone = tokenizer_with(&cls_a_sep());
  assert_eq!(check_post_processor(&alone), Ok(()));
  let expected = alone
    .encode("a b", true)
    .expect("encode")
    .get_ids()
    .to_vec();
  assert_eq!(
    expected,
    vec![1, 1, 2, 1],
    "the special token, the text, the special token — merged into one encoding"
  );

  for chain in [
    sequence(&[&template(&format!("[{PIECE_A}]"), "{}"), &cls_a_sep()]),
    sequence(&[BYTE_LEVEL, &cls_a_sep()]),
    sequence(&[&cls_a_sep()]),
  ] {
    let tokenizer = tokenizer_with(&chain);
    assert_eq!(check_post_processor(&tokenizer), Ok(()), "for `{chain}`");
    assert_eq!(
      tokenizer.encode("a b", true).expect("encode").get_ids(),
      expected.as_slice(),
      "for `{chain}`"
    );
  }
}

/// The forward-compatibility arm, exercised where it is reachable. A kind the
/// tokenizers DESERIALIZER does not know cannot travel through a `Tokenizer` at
/// all — `Tokenizer::from_bytes` refuses the file, and `serde_json::to_value` of
/// a live post-processor can only emit one of the five tags the crate has — so
/// this addresses [`check_value`] directly. A kind added by a later tokenizers
/// version passes and is credited with mapping encodings 1:1, which is what
/// every kind that exists today does.
#[test]
fn an_unrecognized_kind_passes_and_keeps_the_count() {
  for count in [0, 1, 2, 3] {
    let unknown = serde_json::json!({"type": "SomeProcessorAddedLater"});
    assert_eq!(check_value(&unknown, count), Ok(count));
  }
  // …including nested inside a `Sequence`, where it must not swallow the count
  // its neighbours depend on.
  let chain = serde_json::json!({
    "type": "Sequence",
    "processors": [
      {"type": "SomeProcessorAddedLater"},
      serde_json::from_str::<serde_json::Value>(&template(
        &format!("[{PIECE_A},{PIECE_SPECIAL_A}]"),
        SPECIAL_A,
      ))
      .expect("json"),
      {"type": "SomeProcessorAddedLater"},
      serde_json::from_str::<serde_json::Value>(&template(&format!("[{PIECE_A}]"), "{}"))
        .expect("json"),
    ],
  });
  assert_eq!(
    check_value(&chain, 1),
    Err(PostProcessorTemplate::UnsupportedEncodingCount(2))
  );
}
