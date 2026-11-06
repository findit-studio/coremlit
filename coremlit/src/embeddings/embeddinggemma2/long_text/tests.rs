use super::*;
use windit::plan::WindowOptions;

use crate::embeddings::{
  aggregate::EmaRenormalizedOptions,
  embeddinggemma2::{
    embedding::EMBEDDING_DIM,
    text::{framing_of, framing_table},
    tokenizer::image::fixture,
  },
};

/// A small deterministic generator: no `rand` among the dev-dependencies.
struct Lcg(u64);

impl Lcg {
  fn next(&mut self) -> u64 {
    self.0 = self
      .0
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    self.0 >> 33
  }

  fn below(&mut self, n: usize) -> usize {
    (self.next() % n as u64) as usize
  }
}

/// The joint encoding of `prompt ⧺ text`: ids and joint-string spans.
fn encode(
  image: &TokenizerImage,
  prompt: Prompt<'_>,
  text: &str,
  split: SplitSpecialTokens,
) -> (Vec<TokenId>, Vec<Range<usize>>) {
  let (mut ids, mut spans) = (Vec::new(), Vec::new());
  image
    .encode_joint(
      &prompt,
      text,
      split,
      &mut TokenizerScratch::new(),
      Encoded {
        ids: &mut ids,
        spans: Some(&mut spans),
      },
    )
    .expect("encodes");
  (ids, spans)
}

/// What the generated texts are made of: the mini vocabulary's scalars and
/// merges, the bridging site, runs of spaces, the specials and their halves,
/// multi-byte scalars, a literal `▁`, and sentence and paragraph breaks.
const ATOMS: [&str; 30] = [
  "a", "b", "x", "y", "aa", "ab", "xy", " ", " ", " ", "  ", "   ", ">", "<", "/", "</", "> </",
  ":", "q", "<m>", "<m>!", "<|ima", "ge>", "<|image>", "<image|>", "é", "日", "\u{2581}", ". ",
  "\n\n",
];

fn generated(rng: &mut Lcg, atoms: usize) -> String {
  (0..atoms).map(|_| ATOMS[rng.below(ATOMS.len())]).collect()
}

/// A random range of `text` on `char` boundaries, sometimes empty, sometimes
/// the whole text.
fn range(rng: &mut Lcg, text: &str) -> (usize, usize) {
  let bounds: Vec<usize> = text
    .char_indices()
    .map(|(at, _)| at)
    .chain([text.len()])
    .collect();
  match rng.below(8) {
    0 => (0, text.len()),
    _ => {
      let (i, j) = (rng.below(bounds.len()), rng.below(bounds.len()));
      (bounds[i.min(j)], bounds[i.max(j)])
    }
  }
}

/// The prompts the measure is checked under: the named forms (byte fallback
/// in the mini vocabulary), a title, and custom prefixes ending in a colon, in
/// `>`, in a space, inside a special literal, and in a scalar that merges with
/// the content (`<` before `/`, the mini vocabulary's twin of `:</`).
const PROMPTS: [Prompt<'static>; 11] = [
  Prompt::Bare,
  Prompt::Document(None),
  Prompt::Document(Some("ab")),
  Prompt::SearchQuery,
  Prompt::Custom("task: legal clause retrieval | query:"),
  Prompt::Custom("markup: <b>"),
  Prompt::Custom("task: search result | query:"),
  Prompt::Custom("look <|ima"),
  Prompt::Custom("q <"),
  Prompt::Custom("a "),
  Prompt::Custom("<m>"),
];

/// §6.1's four `Custom` boundary cases (G6), with their content, and the mini
/// vocabulary's twin of the first: a prefix whose last scalar merges with the
/// content's first (`<` with `/`, as `:` with `</` upstream).
const BOUNDARY: [(&str, &str); 5] = [
  ("task: legal clause retrieval | query:", "</task> hi"),
  ("q <", "/x a"),
  ("markup: <b>", " </b> bold"),
  ("task: search result | query:", "  bar"),
  ("look <|ima", "ge> now"),
];

/// Checks one range: the measure, both faces, against the direct joint
/// encoding, and the first content token's span against the seam.
fn check_range(
  image: &TokenizerImage,
  measure: &JointMeasure<'_>,
  prompt: Prompt<'_>,
  split: SplitSpecialTokens,
  text: &str,
  (start, end): (usize, usize),
) {
  let (ids, spans) = encode(image, prompt, &text[start..end], split);
  let direct = ids.len();
  let context = || format!("{prompt} ⧺ {:?} under {split}", &text[start..end]);
  assert_eq!(
    measure.measure_range(start, end).expect("measures"),
    direct,
    "{}",
    context()
  );
  for limit in [direct.saturating_sub(1), direct, direct + 1, 0] {
    assert_eq!(
      measure.measure_within(&text[start..end], limit),
      (direct <= limit).then_some(direct),
      "{} within {limit}",
      context()
    );
  }
  // The first content token may straddle the seam; clamped into the range's
  // coordinates, it starts at the range's byte 0. Every token before it lies
  // wholly inside the prompt.
  if start < end {
    let seam = prompt.pieces().concat().len();
    let content = &spans[1..spans.len() - 1];
    let first = content
      .iter()
      .position(|span| span.end > seam)
      .expect("the content has a token");
    assert!(content[first].start <= seam, "{}", context());
    assert_eq!(content[first].start.max(seam) - seam, 0, "{}", context());
    assert!(
      content[..first].iter().all(|span| span.end <= seam),
      "{}",
      context()
    );
  }
}

/// T8 (G6): the measure of every range equals the direct joint encoding of
/// `prompt ⧺ text[a..b]`, through both faces and under both settings, over
/// generated texts under every prompt form, and the four boundary cases at the
/// seam. The first content token's span clamps to byte 0.
#[test]
fn the_measure_is_the_joint_encoding() {
  let (image, _file) = fixture::mini();
  let mut rng = Lcg(0x7e8);
  let mut compared = 0usize;
  for _ in 0..120 {
    let atoms = 4 + rng.below(40);
    let text = generated(&mut rng, atoms);
    for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
      for prompt in PROMPTS {
        let measure = JointMeasure::new(&image, &text, prompt, split).expect("measures");
        for _ in 0..6 {
          let bounds = range(&mut rng, &text);
          check_range(&image, &measure, prompt, split, &text, bounds);
          compared += 1;
        }
      }
    }
  }
  for (prefix, content) in BOUNDARY {
    for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
      let text = format!("{content} aa xy. </b> {content}");
      let prompt = Prompt::Custom(prefix);
      let measure = JointMeasure::new(&image, &text, prompt, split).expect("measures");
      for (end, _) in text.char_indices().skip(1) {
        check_range(&image, &measure, prompt, split, &text, (0, end));
        compared += 1;
      }
      check_range(&image, &measure, prompt, split, &text, (0, text.len()));
    }
  }
  assert!(compared > 15_000, "{compared}");
}

/// T8's cut condition: the mini image allows cuts (no literal holds a space,
/// the one bridging merge `>` + `▁</` holds no word boundary inside it), a
/// cut sits before every space after another scalar except at the bridging
/// site, and a range's interior is read from the table.
#[test]
fn the_piece_table_cuts_at_word_boundaries_off_the_bridging_site() {
  let (image, _file) = fixture::mini();
  let seams = Seams::of(&image);
  assert!(seams.cuts);
  assert_eq!(seams.reach, 1);
  let text = "aa xy > </b  ab";
  let table = PieceTable::build(
    &image,
    text,
    SplitSpecialTokens::Recognize,
    &seams,
    &mut TokenizerScratch::new(),
  )
  .expect("builds");
  // Word boundaries at 2 (` xy`), 5 (` >`), 7 (` </b`, bridged) and 11
  // (`  ab`); 12 is inside a run of spaces.
  assert_eq!(table.cuts, [2, 5, 11]);
  let (ids, spans) = encode(&image, Prompt::Bare, text, SplitSpecialTokens::Recognize);
  for (cut, before) in table.cuts.iter().zip(&table.before) {
    let counted = spans[1..ids.len() - 1]
      .iter()
      .filter(|span| span.start < *cut)
      .count();
    assert_eq!(*before, counted, "before {cut}");
  }
}

/// The prompt's last piece: after its last word boundary that no bridging
/// left text ends at. The eight prompts join at their trailing space; a
/// prefix without one joins its last word; a word boundary after `>` is not a
/// join, since content beginning ` </` would bridge it.
#[test]
fn the_prompt_joins_at_its_last_piece() {
  let (image, _file) = fixture::mini();
  let seams = Seams::of(&image);
  let at = |prompt: &str| join_point(prompt, &seams);
  assert_eq!(at("title: none | text: "), 19);
  assert_eq!(at("task: legal clause retrieval | query:"), 30);
  assert_eq!(at("q <"), 1);
  assert_eq!(at("x > "), 1);
  assert_eq!(at("y:"), 0);
  assert_eq!(at(""), 0);
}

/// Generated paragraphs: sentences of words over the mini vocabulary.
fn paragraphs(rng: &mut Lcg, count: usize) -> String {
  const WORDS: [&str; 10] = ["aa", "ab", "xy", "b", "a", "x", "y", "é", "日日", "<m>"];
  (0..count)
    .map(|_| {
      (0..1 + rng.below(4))
        .map(|_| {
          let words: Vec<&str> = (0..2 + rng.below(9))
            .map(|_| WORDS[rng.below(WORDS.len())])
            .collect();
          format!("{}.", words.join(" "))
        })
        .collect::<Vec<_>>()
        .join(" ")
    })
    .collect::<Vec<_>>()
    .join("\n\n")
}

/// The mini image's framing of `prompt` under `split`.
fn mini_framing(image: &TokenizerImage, prompt: Prompt<'_>, split: SplitSpecialTokens) -> Framing {
  let mut scratch = TokenizerScratch::new();
  let table = framing_table(image, &mut scratch).expect("frames");
  framing_of(image, &table, prompt, split, &mut scratch).expect("frames")
}

/// T7's hermetic half: over the mini image, the windows of generated texts
/// partition each text on `char` boundaries (with no overlap) and cover it
/// (with one); each window's ids are the joint encoding of `prompt ⧺ chunk`,
/// the prompt in every window, a custom one included; and each window holds at
/// most `window` ids, save one character that alone outgrows it.
#[test]
fn the_windows_cover_the_text_and_each_carries_the_prompt() {
  let (image, _file) = fixture::mini();
  let mut rng = Lcg(0x77);
  let mut scratch = TokenizerScratch::new();
  let mut multi = 0;
  for case in 0..60 {
    let count = 1 + rng.below(5);
    let text = paragraphs(&mut rng, count);
    let prompt = [
      Prompt::Bare,
      Prompt::Custom("q <"),
      Prompt::Custom("markup: <b>"),
      Prompt::Document(Some("ab")),
    ][case % 4];
    for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
      let framing = mini_framing(&image, prompt, split);
      for extra in [3, 7, 16] {
        let window = framing.ids() + extra;
        for hop in [window, window - 2] {
          let geometry = LongTextOptions::from(WindowOptions::new(window).with_hop(hop));
          let windows = plan(
            &image,
            &text,
            prompt,
            split,
            &geometry,
            framing,
            &mut scratch,
          )
          .expect("plans");
          assert!(!windows.is_empty());
          multi += usize::from(windows.len() > 1);
          let mut covered = 0;
          for (index, planned) in windows.iter().enumerate() {
            let chunk = text.get(planned.range.clone()).expect("char-aligned");
            assert!(planned.range.start <= covered, "{prompt} {index}: a gap");
            if hop == window && index > 0 {
              assert_eq!(
                planned.range.start, covered,
                "{prompt} {index}: not a partition"
              );
            }
            covered = covered.max(planned.range.end);
            assert_eq!(
              planned.ids,
              encode(&image, prompt, chunk, split).0,
              "{prompt}: window {index} ({chunk:?}) does not carry the prompt"
            );
            // windit keeps a character that alone outgrows the window, and the
            // repair a run of separators neither neighbour can take.
            let lone = chunk.chars().count() == 1 || !chunk.chars().any(char::is_alphanumeric);
            assert!(
              planned.ids.len() <= window || lone,
              "{prompt}: window {index} holds {} ids against {window}",
              planned.ids.len()
            );
            assert!(planned.ids.len() <= MAX_TOKENS);
          }
          assert_eq!(windows[0].range.start, 0);
          assert_eq!(covered, text.len());
        }
      }
    }
  }
  assert!(multi > 100, "{multi}");
}

/// T7: a text that fits one window plans to one window over all of it, whose
/// ids are exactly [`TextEmbedder::embed`]'s: the joint encoding of `prompt ⧺
/// text`.
#[test]
fn a_text_that_fits_one_window_is_embeds_ids() {
  let (image, _file) = fixture::mini();
  let mut scratch = TokenizerScratch::new();
  for prompt in PROMPTS {
    for text in ["ab", "aa xy. b", "\n\n ", "</b> x", "ge> now"] {
      let split = SplitSpecialTokens::Recognize;
      let framing = mini_framing(&image, prompt, split);
      let windows = plan(
        &image,
        text,
        prompt,
        split,
        &LongTextOptions::new(),
        framing,
        &mut scratch,
      )
      .expect("plans");
      assert_eq!(windows.len(), 1, "{prompt} ⧺ {text:?}");
      assert_eq!(windows[0].range, 0..text.len());
      assert_eq!(windows[0].ids, encode(&image, prompt, text, split).0);
    }
  }
}

/// K4 for the long-text call: in order, the input bound, the model's 512, a
/// geometry windit refuses, a window the prompt's framing fills, then an empty
/// text. A custom prefix's framing is the conservative count; a window one
/// past the framing plans.
#[test]
fn a_call_is_refused_by_name_in_order() {
  let (image, _file) = fixture::mini();
  let mut scratch = TokenizerScratch::new();
  let split = SplitSpecialTokens::Recognize;
  let custom = mini_framing(&image, Prompt::Custom("q <"), split);
  assert!(custom.is_conservative());
  assert_eq!(
    custom.ids(),
    encode(&image, Prompt::Bare, "q <", split).0.len()
  );
  let mut call = |text: &str, prompt: Prompt<'_>, geometry: LongTextOptions| {
    let framing = mini_framing(&image, prompt, split);
    plan(
      &image,
      text,
      prompt,
      split,
      &geometry,
      framing,
      &mut scratch,
    )
  };
  let window = |w: usize| LongTextOptions::from(WindowOptions::new(w));
  match call("aa ab", Prompt::Bare, window(513).with_max_input_bytes(4)) {
    Err(Error::InputTooLarge(e)) => assert_eq!((e.got(), e.max()), (5, 4)),
    other => panic!("expected InputTooLarge, got {other:?}"),
  }
  match call("aa ab", Prompt::Bare, window(513)) {
    Err(Error::WindowOverBudget(e)) => assert_eq!((e.window(), e.max()), (513, 512)),
    other => panic!("expected WindowOverBudget, got {other:?}"),
  }
  assert!(matches!(
    call("aa ab", Prompt::Bare, window(0)),
    Err(Error::Windowing(_))
  ));
  match call("aa ab", Prompt::Custom("q <"), window(custom.ids())) {
    Err(Error::PromptFillsWindow(e)) => {
      assert_eq!((e.window(), e.framing()), (custom.ids(), custom.ids()))
    }
    other => panic!("expected PromptFillsWindow, got {other:?}"),
  }
  assert!(matches!(
    call("", Prompt::Bare, window(2)),
    Err(Error::PromptFillsWindow(_))
  ));
  assert!(matches!(
    call("", Prompt::Bare, window(3)),
    Err(Error::EmptyText)
  ));
  let planned = call("aa ab", Prompt::Custom("q <"), window(custom.ids() + 1)).expect("plans");
  assert_eq!(planned.last().map(|w| w.range.end), Some(5));
}

/// The ruled refusal (C11): a prompt within a few ids of the model's 512
/// and one character that alone outgrows the window. windit keeps the
/// character; beside the prompt it outgrows the model too, and the window is
/// refused by name at the model's 512, never truncated.
#[test]
fn one_character_past_the_models_512_is_refused_by_name() {
  let (image, _file) = fixture::mini();
  let mut scratch = TokenizerScratch::new();
  let split = SplitSpecialTokens::Recognize;
  // 509 single-scalar tokens that no merge joins: a framing of 511.
  let prefix = "b".repeat(509);
  let prompt = Prompt::Custom(&prefix);
  let framing = mini_framing(&image, prompt, split);
  assert_eq!(framing.ids(), 511);
  // A scalar outside the mini vocabulary: four byte-fallback tokens, 515 ids
  // beside the prompt.
  let text = "\u{1f600}";
  assert_eq!(encode(&image, prompt, text, split).0.len(), 515);
  match plan(
    &image,
    text,
    prompt,
    split,
    &LongTextOptions::new(),
    framing,
    &mut scratch,
  ) {
    Err(Error::PromptFillsWindow(e)) => assert_eq!((e.window(), e.framing()), (512, 511)),
    other => panic!("expected PromptFillsWindow at 512, got {other:?}"),
  }
  // One character that fits is planned as one window.
  let fits = plan(
    &image,
    "b",
    prompt,
    split,
    &LongTextOptions::new(),
    framing,
    &mut scratch,
  )
  .expect("plans");
  assert_eq!(fits.len(), 1);
  assert_eq!(fits[0].ids.len(), 512);
}

/// A unit vector whose coordinates follow `seed`.
fn vector(seed: u32) -> Embedding {
  let mut row = [0f32; EMBEDDING_DIM];
  for (i, v) in row.iter_mut().enumerate() {
    *v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 1_000) as f32 / 1_000.0 - 0.5;
  }
  Embedding::from_output(&row).expect("a finite, non-zero row")
}

/// T7 (condition 3): granite's single-window identity re-pinned for
/// `Embedding<768>`. Folding exactly one window under the default fold gives
/// it back bit for bit, so returning one window as it is answers what folding
/// it would; under every fold, one window is returned as it is.
#[test]
fn one_window_folds_to_itself_bit_for_bit() {
  let window = WindowEmbedding::new(0, 0..10, Span::new(0, 300, MAX_TOKENS), vector(7));
  let folded = windit::aggregate::aggregate(
    &windit::aggregate::CoverageWeightedMean,
    &[Windowed::new(
      window.embedding().clone(),
      window.token_span(),
    )],
  )
  .expect("folds");
  assert_eq!(folded.as_slice(), window.embedding().as_slice());
  for kind in [
    AggregatePolicyKind::CoverageWeightedMean,
    AggregatePolicyKind::MeanRenormalized,
    AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.5)),
  ] {
    let alone = fold_windows(vec![window.clone()], &kind).expect("folds");
    assert_eq!(alone.as_slice(), window.embedding().as_slice(), "{kind}");
  }
}

/// T7's fold option, hermetic: windows of unequal coverage fold under the fold
/// asked for, each fold exactly windit's, and the folds differ.
#[test]
fn the_fold_option_is_honoured() {
  let windows: Vec<WindowEmbedding> = [(508, 1), (508, 2), (308, 3)]
    .iter()
    .scan(0, |offset, &(ids, seed)| {
      let span = Span::new(*offset, ids, MAX_TOKENS);
      *offset += ids;
      Some(WindowEmbedding::new(
        seed as usize - 1,
        0..1,
        span,
        vector(seed),
      ))
    })
    .collect();
  let kinds = [
    AggregatePolicyKind::CoverageWeightedMean,
    AggregatePolicyKind::MeanRenormalized,
    AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.5)),
  ];
  let mut folded = Vec::new();
  for kind in kinds {
    let windowed: Vec<Windowed<Embedding>> = windows
      .iter()
      .map(|w| Windowed::new(w.embedding().clone(), w.token_span()))
      .collect();
    let expected =
      windit::aggregate::aggregate(kind.into_policy().as_ref(), &windowed).expect("folds");
    let got = fold_windows(windows.clone(), &kind).expect("folds");
    assert_eq!(got.as_slice(), expected.as_slice(), "{kind}");
    folded.push(got);
  }
  assert_ne!(folded[0].as_slice(), folded[1].as_slice());
  assert_ne!(folded[0].as_slice(), folded[2].as_slice());
}

/// The staged model directory: `EMBEDDINGGEMMA2_TEST_MODELS`, else the
/// workspace's `Models/embeddinggemma2`.
fn staged_dir() -> std::path::PathBuf {
  std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    std::path::PathBuf::from,
  )
}

/// T8's real twin: over the staged tokenizer image, the measure equals the
/// direct joint encoding over random ranges of the four long texts under their
/// prompts, and at the seam of the four boundary cases, where the real `:</`
/// merges across it; both settings.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_measure_is_the_joint_encoding() {
  use crate::embeddings::embeddinggemma2::text::TextArtifacts;

  let artifacts = TextArtifacts::open(staged_dir()).expect("the staged text part opens");
  let image = artifacts.image();
  let golden: serde_json::Value = serde_json::from_slice(
    &std::fs::read(
      std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/embeddinggemma2/fixtures/goldens/text.json"),
    )
    .expect("the golden reads"),
  )
  .expect("parses");
  let mut rng = Lcg(0x8a);
  let mut compared = 0usize;
  for item in golden["items"].as_array().expect("items") {
    let source = item["source"].as_str().expect("source");
    if source != "long" && source != "boundary" {
      continue;
    }
    let prefix = item["prefix"].as_str().expect("prefix");
    let text = item["text"].as_str().expect("text");
    let prompt = if prefix.is_empty() {
      Prompt::Bare
    } else {
      Prompt::Custom(prefix)
    };
    for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
      let measure = JointMeasure::new(image, text, prompt, split).expect("measures");
      if source == "boundary" {
        for (end, _) in text.char_indices().skip(1) {
          check_range(image, &measure, prompt, split, text, (0, end));
          compared += 1;
        }
        check_range(image, &measure, prompt, split, text, (0, text.len()));
      } else {
        for _ in 0..200 {
          let bounds = range(&mut rng, text);
          check_range(image, &measure, prompt, split, text, bounds);
          compared += 1;
        }
      }
    }
  }
  eprintln!("T8 (staged): {compared} ranges equal the direct joint encoding");
  assert!(compared > 1_600, "{compared}");
}
