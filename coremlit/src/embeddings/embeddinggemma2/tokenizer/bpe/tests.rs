use std::io::Write;

use super::{
  super::{
    super::error::Error,
    Sentinel, UPSTREAM_SENTINELS,
    compile::compile_image,
    image::{Counts, Identity, UPSTREAM},
  },
  *,
};

/// The committed mini `tokenizer.json` (see the image's tests).
const MINI: &[u8] =
  include_bytes!("../../../../../tests/embeddinggemma2/fixtures/mini_tokenizer.json");

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

fn image_of(json: &[u8], identity: &Identity) -> (TokenizerImage, tempfile::NamedTempFile) {
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file
    .write_all(&compile_image(json).expect("compiles"))
    .expect("write");
  file.flush().expect("flush");
  let image = TokenizerImage::open(file.path(), identity).expect("opens");
  (image, file)
}

fn mini_identity() -> Identity {
  use sha2::{Digest, Sha256};
  Identity {
    source: Sha256::digest(MINI).into(),
    counts: Counts {
      vocab: 286,
      merges: 9,
      added: 8,
      scalars: 13,
      bridges: 1,
    },
  }
}

fn mini() -> (TokenizerImage, tempfile::NamedTempFile) {
  image_of(MINI, &mini_identity())
}

/// The kit's ids and joint-string spans for `prompt ⧺ text`.
fn encode(
  image: &TokenizerImage,
  prompt: Prompt<'_>,
  text: &str,
  split: SplitSpecialTokens,
) -> (Vec<u32>, Vec<Range<usize>>) {
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
  (ids.iter().map(|id| id.get()).collect(), spans)
}

/// Hugging Face's tokenizer under each setting, loaded once.
struct Oracle {
  recognize: tokenizers::Tokenizer,
  split: tokenizers::Tokenizer,
}

impl Oracle {
  fn from_bytes(json: &[u8]) -> Self {
    let recognize = tokenizers::Tokenizer::from_bytes(json).expect("the oracle loads");
    let mut split = recognize.clone();
    split.set_encode_special_tokens(true);
    Self { recognize, split }
  }
}

/// Hugging Face's ids and byte spans for `joint`, `<bos>`/`<eos>` spans empty.
fn oracle(
  tokenizers: &Oracle,
  joint: &str,
  split: SplitSpecialTokens,
) -> (Vec<u32>, Vec<Range<usize>>) {
  let tokenizer = if split.is_split() {
    &tokenizers.split
  } else {
    &tokenizers.recognize
  };
  let encoding = tokenizer.encode(joint, true).expect("the oracle encodes");
  let spans = encoding.get_offsets().iter().map(|&(a, b)| a..b).collect();
  (encoding.get_ids().to_vec(), spans)
}

fn mini_oracle() -> Oracle {
  Oracle::from_bytes(MINI)
}

/// TM1: the structural cases, each pinned and each equal to Hugging Face's:
/// rank first then leftmost, byte fallback spanning the whole character, the
/// longest special first, both settings, `▁` spanning its one space byte, and
/// no Unicode normalization.
#[test]
fn the_structural_cases_are_hugging_faces() {
  let (image, _file) = mini();
  let oracle_tokenizer = mini_oracle();
  let (r, s) = (SplitSpecialTokens::Recognize, SplitSpecialTokens::Split);
  let cases: [(&str, SplitSpecialTokens, &[u32]); 9] = [
    ("aaa", r, &[2, 277, 265, 1]),
    ("a a", r, &[2, 265, 278, 1]),
    ("<m>!", r, &[2, 5, 1]),
    ("<m>", r, &[2, 4, 1]),
    ("<m>!", s, &[2, 269, 117, 268, 41, 1]),
    ("\u{feff}a", r, &[2, 247, 195, 199, 265, 1]),
    ("\u{f900} \u{8c48}", r, &[2, 247, 172, 136, 264, 274, 1]),
    ("e\u{301}", r, &[2, 275, 276, 1]),
    ("> </x", r, &[2, 281, 271, 1]),
  ];
  for (text, split, want) in cases {
    let (ids, spans) = encode(&image, Prompt::Bare, text, split);
    assert_eq!(ids, want, "{text:?} under {split}");
    assert_eq!(
      (ids, spans),
      oracle(&oracle_tokenizer, text, split),
      "{text:?} under {split}"
    );
  }
  let (_, spans) = encode(&image, Prompt::Bare, "\u{feff}a b", r);
  assert_eq!(
    spans,
    [0..0, 0..3, 0..3, 0..3, 3..4, 4..5, 5..6, 0..0],
    "fallback bytes span their character; `▁` its one space byte"
  );
}

/// TM1 over generated text: on strings drawn from the mini fixture's
/// alphabet, specials and look-alikes included, the kit's ids and spans are
/// Hugging Face's under both settings.
#[test]
fn generated_text_encodes_as_hugging_face_does() {
  let (image, _file) = mini();
  let oracle_tokenizer = mini_oracle();
  let alphabet = [
    "a", "b", "c", "x", "y", ">", "<", "/", ":", " ", "  ", "\u{8c48}", "\u{f900}", "\u{feff}",
    "e", "\u{301}", "<m>", "<m>!", "<|image>", "<image|>", "\u{e9}", "\u{20ac}", "\n", "</",
    "> </", "<|ima", "ge>",
  ];
  let mut rng = Lcg(7);
  for case in 0..3_000 {
    let len = rng.below(12);
    let text: String = (0..len)
      .map(|_| alphabet[rng.below(alphabet.len())])
      .collect();
    for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
      assert_eq!(
        encode(&image, Prompt::Bare, &text, split),
        oracle(&oracle_tokenizer, &text, split),
        "case {case}: {text:?} under {split}"
      );
    }
  }
}

/// The prompt and the text are one string: a literal that spans the seam is
/// matched, and a merge across it is made, exactly as in the concatenation.
#[test]
fn the_seam_is_invisible() {
  let (image, _file) = mini();
  let oracle_tokenizer = mini_oracle();
  for (prefix, text) in [
    ("look <|ima", "ge> now"),
    ("a", "ab"),
    ("x >", " </x"),
    ("y:", "  b"),
  ] {
    for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
      let joint = format!("{prefix}{text}");
      assert_eq!(
        encode(&image, Prompt::Custom(prefix), text, split),
        oracle(&oracle_tokenizer, &joint, split),
        "{prefix:?} + {text:?} under {split}"
      );
    }
  }
  let (ids, _) = encode(
    &image,
    Prompt::Custom("look <|ima"),
    "ge> now",
    SplitSpecialTokens::Recognize,
  );
  assert!(
    ids.contains(&6),
    "the literal spanning the seam is its special id"
  );
}

/// TM3: cutting a segment into pieces at word boundaries (but not at a
/// bridging site) gives exactly the BPE of the whole segment.
#[test]
fn piece_by_piece_is_the_whole_segments_bpe() {
  let (image, _file) = mini();
  let alphabet = [
    "a", "b", "x", "y", ">", "<", "/", " ", "  ", "</", "> </", ":",
  ];
  let mut rng = Lcg(11);
  let mut scratch = TokenizerScratch::new();
  let scalars = |text: &str| -> Vec<Scalar> {
    let mut at = 0usize;
    text
      .chars()
      .map(|ch| {
        let end = at + ch.len_utf8();
        let scalar = Scalar { ch, start: at, end };
        at = end;
        scalar
      })
      .collect()
  };
  for case in 0..3_000 {
    let text: String = if case == 0 {
      "> </x".to_owned()
    } else {
      (0..rng.below(14))
        .map(|_| alphabet[rng.below(alphabet.len())])
        .collect()
    };
    let scalars = scalars(&text);
    let (mut pieced, mut whole) = (Vec::new(), Vec::new());
    image
      .encode_segment(
        &scalars,
        &mut scratch,
        &mut Encoded {
          ids: &mut pieced,
          spans: None,
        },
      )
      .expect("pieces");
    image
      .encode_piece(
        &scalars,
        &mut scratch,
        &mut Encoded {
          ids: &mut whole,
          spans: None,
        },
      )
      .expect("whole");
    assert_eq!(pieced, whole, "case {case}: {text:?}");
  }
}

/// The mini fixture's sentinels: its own five, the bridging site last.
const MINI_SENTINELS: [Sentinel; 5] = [
  Sentinel {
    prompt: Prompt::Bare,
    text: "aaa",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 277, 265, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "\u{feff}a",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 247, 195, 199, 265, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "<m>!",
    split: SplitSpecialTokens::Split,
    ids: &[2, 269, 117, 268, 41, 1],
  },
  Sentinel {
    prompt: Prompt::Bare,
    text: "> </x",
    split: SplitSpecialTokens::Recognize,
    ids: &[2, 281, 271, 1],
  },
];

/// G13: an open runs the sentinel encodings, the bridging site among them, so
/// an image whose bridging merge is wrong is refused by name although every
/// structural check passes.
#[test]
fn an_open_runs_the_bridging_site() {
  let (image, _file) = mini();
  image
    .check_sentinels(&MINI_SENTINELS, &mut TokenizerScratch::new())
    .expect("the mini image keeps its sentinels");

  let mut bytes = compile_image(MINI).expect("compiles");
  let bridges_at = u64::from_le_bytes(
    bytes[super::super::image::header::SECTIONS + 7 * 16..][..8]
      .try_into()
      .unwrap(),
  ) as usize;
  bytes[bridges_at + 4..bridges_at + 8].copy_from_slice(&278u32.to_le_bytes());
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file.write_all(&bytes).expect("write");
  let wrong = TokenizerImage::open(file.path(), &mini_identity()).expect("the structure is intact");
  match wrong.check_sentinels(&MINI_SENTINELS, &mut TokenizerScratch::new()) {
    Err(Error::TokenizerImage(defect)) => {
      assert_eq!(
        defect.field(),
        super::super::super::error::ImageField::Sentinel
      );
      assert!(defect.expected().contains("> </x"), "{}", defect.expected());
    }
    other => panic!("expected a sentinel refusal, got {other:?}"),
  }
}

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

fn staged() -> (TokenizerImage, tempfile::NamedTempFile, Oracle) {
  let json = std::fs::read(staged_dir().join("tokenizer.json")).expect("the staged tokenizer.json");
  let (image, file) = image_of(&json, &UPSTREAM);
  (image, file, Oracle::from_bytes(&json))
}

/// The golden record's prompt.
fn prompt_of(record: &serde_json::Value) -> Prompt<'_> {
  match record["kind"].as_str().expect("kind") {
    "bare" => Prompt::Bare,
    "document" => Prompt::Document(record["title"].as_str()),
    "custom" => Prompt::Custom(record["prefix"].as_str().expect("prefix")),
    _ => match record["key"].as_str().expect("key") {
      "document" => Prompt::Document(None),
      "search_query" => Prompt::SearchQuery,
      "question_answering" => Prompt::QuestionAnswering,
      "fact_checking" => Prompt::FactChecking,
      "code_retrieval" => Prompt::CodeRetrieval,
      "classification" => Prompt::Classification,
      "clustering" => Prompt::Clustering,
      _ => Prompt::SentenceSimilarity,
    },
  }
}

/// T1, and G13 over the real image: every corpus item's ids, byte offsets and
/// special mask under both settings are Hugging Face's goldens, and the
/// upstream image keeps its five sentinel encodings.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_corpus_encodes_to_the_goldens() {
  let (image, _file, _) = staged();
  image
    .check_sentinels(&UPSTREAM_SENTINELS, &mut TokenizerScratch::new())
    .expect("the upstream image keeps its sentinels");
  let goldens = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/embeddinggemma2/fixtures/goldens/text.json"
  );
  let doc: serde_json::Value =
    serde_json::from_str(&std::fs::read_to_string(goldens).expect("the text goldens"))
      .expect("JSON");
  let mut checked = 0;
  for item in doc["items"].as_array().expect("items") {
    let id = item["id"].as_str().expect("id");
    let text = item["text"].as_str().expect("text");
    for (key, split) in [
      ("recognize", SplitSpecialTokens::Recognize),
      ("split", SplitSpecialTokens::Split),
    ] {
      let (ids, spans) = encode(&image, prompt_of(&item["prompt"]), text, split);
      let want_ids: Vec<u32> = item[key]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
      let want_spans: Vec<Range<usize>> = item[key]["offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|span| span[0].as_u64().unwrap() as usize..span[1].as_u64().unwrap() as usize)
        .collect();
      let mask: Vec<u64> = (0..ids.len())
        .map(|i| u64::from(i == 0 || i + 1 == ids.len()))
        .collect();
      let want_mask: Vec<u64> = item[key]["special_mask"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
      assert_eq!(ids, want_ids, "{id} {key}: ids");
      assert_eq!(spans, want_spans, "{id} {key}: byte offsets");
      assert_eq!(
        mask, want_mask,
        "{id} {key}: the special mask marks the framing alone"
      );
      checked += 1;
    }
  }
  assert_eq!(checked, 160);
}

/// T2: a differential against `tokenizers` over the real tokenizer under both
/// settings, on four seeded generators, each behind no prompt and behind the
/// four custom boundary shapes.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_real_tokenizer_agrees_with_hugging_face_on_generated_text() {
  let (image, _file, tokenizer) = staged();
  let vocab_size = image.counts().vocab as usize;
  let specials: Vec<String> = image
    .added()
    .map(|id| String::from_utf8(image.token(id).unwrap().to_vec()).unwrap())
    .collect();
  let fixtures = [
    "Café crème, naïve résumé",
    "Did 30 minutes easy this morning in 45:10.",
    "<b> </b>",
    "東京で一番のラーメン",
  ];
  let prefixes = [
    "",
    "task: legal clause retrieval | query:",
    "markup: <b>",
    "task: search result | query:",
    "look <|ima",
  ];
  let mut rng = Lcg(2026);
  let mut cases = 0;
  for generator in 0..4 {
    for _ in 0..400 {
      let text: String = match generator {
        // Vocabulary pieces, decoded: merge-heavy.
        0 => (0..1 + rng.below(8))
          .map(|_| {
            let id = rng.below(vocab_size) as u32;
            String::from_utf8_lossy(image.token(id).unwrap()).replace('\u{2581}', " ")
          })
          .collect(),
        // Mutated fixtures.
        1 => {
          let mut chars: Vec<char> = fixtures[rng.below(fixtures.len())].chars().collect();
          for _ in 0..1 + rng.below(4) {
            let at = rng.below(chars.len() + 1);
            match rng.below(3) {
              0 if at < chars.len() => {
                chars.remove(at);
              }
              _ => chars.insert(
                at,
                [' ', 'é', '<', '>', '\u{2581}', '豈', '\n'][rng.below(7)],
              ),
            }
          }
          chars.into_iter().collect()
        }
        // Random scalars across the planes.
        2 => (0..1 + rng.below(10))
          .filter_map(|_| {
            char::from_u32(
              [rng.below(0x80), rng.below(0x3000), rng.below(0x11_0000)][rng.below(3)] as u32,
            )
          })
          .collect(),
        // Specials at random positions, the near-twins, next to spaces.
        _ => (0..1 + rng.below(6))
          .map(|_| match rng.below(4) {
            0 => specials[rng.below(specials.len())].clone(),
            1 => ["<|image>", "<|image|>", "<image|>"][rng.below(3)].to_owned(),
            2 => " ".to_owned(),
            _ => "ab".to_owned(),
          })
          .collect(),
      };
      for prefix in prefixes {
        for split in [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split] {
          let prompt = if prefix.is_empty() {
            Prompt::Bare
          } else {
            Prompt::Custom(prefix)
          };
          assert_eq!(
            encode(&image, prompt, &text, split),
            oracle(&tokenizer, &format!("{prefix}{text}"), split),
            "generator {generator}: {prefix:?} + {text:?} under {split}"
          );
          cases += 1;
        }
      }
    }
  }
  eprintln!("T2: {cases} encodings agreed with tokenizers");
}

/// G13: the upstream sentinels run the bridging site (`<b> </b>`) and the
/// special literal under both settings, so an open exercises every path the
/// structural checks cannot.
#[test]
fn the_upstream_sentinels_run_the_bridging_site() {
  assert!(
    UPSTREAM_SENTINELS.iter().any(|s| s.text.contains("> </")),
    "a sentinel holds the bridging site"
  );
  let literal: Vec<SplitSpecialTokens> = UPSTREAM_SENTINELS
    .iter()
    .filter(|s| s.text == "<|image|>")
    .map(|s| s.split)
    .collect();
  assert_eq!(
    literal,
    [SplitSpecialTokens::Recognize, SplitSpecialTokens::Split]
  );
  assert!(
    UPSTREAM_SENTINELS.iter().any(|s| s.text.is_empty()),
    "the empty text"
  );
}

/// Codex round 1, finding 4: a piece's merge heap grows with the piece, so it
/// is reserved before the merge loop, at its bound (fewer than three
/// candidates a symbol), through the kit's one door. Refused there, the
/// encoding answers `Allocation(TokenizerScratch)` instead of aborting.
#[test]
fn the_bpe_heap_reservation_is_refused_by_name() {
  use super::super::super::alloc::inject;

  let (image, _file) = mini();
  // One piece: the scalars are the site's first reservation, the piece's
  // symbols its second, and its heap its third.
  let _guard = inject::refuse(AllocationSite::TokenizerScratch, 2);
  let (mut ids, mut spans) = (Vec::new(), Vec::new());
  let refused = image.encode_joint(
    &Prompt::Bare,
    "aaaaaaaa",
    SplitSpecialTokens::Recognize,
    &mut TokenizerScratch::new(),
    Encoded {
      ids: &mut ids,
      spans: Some(&mut spans),
    },
  );
  assert!(
    matches!(&refused, Err(Error::Allocation(a)) if a.site() == AllocationSite::TokenizerScratch),
    "{refused:?}"
  );
}

/// Codex round 1, finding 6: no count narrows. The joint string's byte
/// offsets, a piece's symbol links and a candidate's position are as wide as
/// the string's length, so a span past the old `u32` limit (4 GiB) comes out
/// exactly: laid out from just under it, a text encodes to the same ids as
/// from 0, each span shifted by exactly the base, the last past 2^32.
#[test]
fn a_span_past_four_gib_is_kept_exactly() {
  let (image, _file) = mini();
  let encoded_from = |base: usize| {
    let mut scalars = Vec::new();
    lay_out(["", "", "", "ab cd"], base, &mut scalars).expect("lays out");
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    image
      .encode_scalars(
        &scalars,
        SplitSpecialTokens::Recognize,
        &mut TokenizerScratch::new(),
        &mut Encoded {
          ids: &mut ids,
          spans: Some(&mut spans),
        },
      )
      .expect("encodes");
    (ids, spans)
  };
  let base = u32::MAX as usize - 2;
  let (ids, spans) = encoded_from(0);
  let (far_ids, far_spans) = encoded_from(base);
  assert_eq!(far_ids, ids);
  let shifted: Vec<Range<usize>> = spans
    .iter()
    .map(|span| span.start + base..span.end + base)
    .collect();
  assert_eq!(far_spans, shifted);
  assert!(
    far_spans
      .last()
      .is_some_and(|span| span.end > u32::MAX as usize),
    "{far_spans:?}"
  );
}
