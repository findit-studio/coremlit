use std::io::Write;

use super::{
  super::{
    super::error::SourceField,
    compile::{compile_image, image_name},
  },
  *,
};

/// The committed mini `tokenizer.json`: the real file's configuration over
/// 286 tokens and 9 merges, one of them bridging.
const MINI: &[u8] =
  include_bytes!("../../../../../tests/embeddinggemma2/fixtures/mini_tokenizer.json");

/// The SHA-256 of the mini fixture's format-1 image. A change to the layout,
/// the hash or a section's meaning moves it, and must bump
/// [`TOKENIZER_IMAGE_VERSION`].
const MINI_IMAGE_SHA256: &str = "5122a5b4d706b28a8b8271b47ce292c86dd2b34fc19e0526b94a68ba1814332a";

fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
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

fn written(bytes: &[u8]) -> tempfile::NamedTempFile {
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file.write_all(bytes).expect("write");
  file.flush().expect("flush");
  file
}

fn defect_field(result: Result<TokenizerImage>) -> ImageField {
  match result {
    Err(Error::TokenizerImage(defect)) => defect.field(),
    other => panic!("expected a tokenizer image refusal, got {other:?}"),
  }
}

/// TM2: compiling the mini fixture twice gives the same bytes, and they are
/// the committed digest.
#[test]
fn the_mini_image_is_deterministic_and_pinned() {
  let first = compile_image(MINI).expect("the mini fixture compiles");
  let second = compile_image(MINI).expect("again");
  assert_eq!(first, second);
  assert_eq!(sha256_hex(&first), MINI_IMAGE_SHA256);
}

/// TM2: the header carries its fields, and the image opens under its own
/// identity and reads its tables back.
#[test]
fn the_mini_image_opens_and_reads_back_its_tables() {
  let bytes = compile_image(MINI).expect("compiles");
  assert_eq!(&bytes[..8], b"CMLGBPE\0");
  assert_eq!(u32_at(&bytes, header::VERSION), Some(1));
  assert_eq!(u32_at(&bytes, header::HEADER_SIZE), Some(4096));
  assert_eq!(
    &bytes[header::SOURCE_DIGEST..header::SOURCE_DIGEST + 32],
    mini_identity().source.as_slice()
  );
  assert_eq!(
    &bytes[header::IMAGE_DIGEST..header::IMAGE_DIGEST + 32],
    hex32(&sha256_hex(&bytes[HEADER_SIZE..])).as_slice()
  );
  assert!(
    bytes[header::END..HEADER_SIZE].iter().all(|b| *b == 0),
    "the header's tail is zero"
  );

  let file = written(&bytes);
  let image = TokenizerImage::open(file.path(), &mini_identity()).expect("opens");
  assert_eq!(image.counts(), mini_identity().counts);
  assert_eq!(image.token(277), Some("aa".as_bytes()));
  assert_eq!(image.token(286), None);
  assert_eq!(image.scalar_id('a'), Some(265));
  assert_eq!(image.scalar_id('\u{8c48}'), Some(274));
  assert_eq!(
    image.scalar_id('\u{f900}'),
    None,
    "the compatibility ideograph is not its canonical twin"
  );
  assert_eq!(image.merge(265, 265), Some((0, 277)));
  assert_eq!(image.merge(268, 280), Some((4, 281)));
  assert_eq!(image.merge(266, 265), None);
  assert_eq!(image.byte_id(0x00), Some(8));
  assert_eq!(image.byte_id(0xff), Some(263));
  assert_eq!(image.bridges().collect::<Vec<_>>(), [(268, 280)]);
  let added: Vec<u32> = image.added().collect();
  assert_eq!(added.len(), 8);
  let lengths: Vec<usize> = added
    .iter()
    .map(|id| image.token(*id).unwrap().len())
    .collect();
  assert!(
    lengths.windows(2).all(|w| w[0] >= w[1]),
    "longest literal first: {lengths:?}"
  );
  assert_eq!(
    image_name(&mini_identity().source),
    format!("tokenizer.v1-{}.gemmabpe", &sha256_hex(MINI)[..12])
  );
}

/// TM2: each kind of wrong image is refused by the field it is wrong in.
#[test]
fn a_wrong_image_is_refused_by_field() {
  let good = compile_image(MINI).expect("compiles");
  let open = |bytes: &[u8]| TokenizerImage::open(written(bytes).path(), &mini_identity());

  let mut magic = good.clone();
  magic[0] ^= 1;
  assert_eq!(defect_field(open(&magic)), ImageField::Magic);

  let mut version = good.clone();
  version[header::VERSION] = 2;
  assert_eq!(defect_field(open(&version)), ImageField::FormatVersion);

  let mut size = good.clone();
  size[header::HEADER_SIZE + 1] ^= 0x30;
  assert_eq!(defect_field(open(&size)), ImageField::HeaderSize);

  let mut count = good.clone();
  count[header::COUNTS] ^= 1;
  assert_eq!(defect_field(open(&count)), ImageField::Count);

  let mut source = good.clone();
  source[header::SOURCE_DIGEST] ^= 1;
  assert_eq!(defect_field(open(&source)), ImageField::SourceDigest);

  let mut section = good.clone();
  section[header::SECTIONS + 7 * 16 + 8] = 0xff;
  assert_eq!(defect_field(open(&section)), ImageField::Section);

  assert_eq!(
    defect_field(open(&good[..good.len() - 8])),
    ImageField::Section
  );
  assert_eq!(defect_field(open(&good[..100])), ImageField::Length);

  assert_eq!(
    defect_field(TokenizerImage::open(written(&good).path(), &UPSTREAM)),
    ImageField::Count,
    "the mini image is not the upstream tokenizer's"
  );
}

/// The compiler refuses a document outside the one configuration the
/// runtime implements, naming the part.
#[test]
fn the_compiler_refuses_another_configuration_by_field() {
  let doc: serde_json::Value = serde_json::from_slice(MINI).expect("JSON");
  let refused = |edit: &dyn Fn(&mut serde_json::Value)| {
    let mut copy = doc.clone();
    edit(&mut copy);
    match compile_image(&serde_json::to_vec(&copy).unwrap()) {
      Err(Error::TokenizerSource(defect)) => defect.field(),
      other => panic!(
        "expected a source refusal, got {:?}",
        other.map(|b| b.len())
      ),
    }
  };
  assert_eq!(
    refused(&|d| d["normalizer"]["content"] = "_".into()),
    SourceField::Normalizer
  );
  assert_eq!(
    refused(&|d| d["pre_tokenizer"]["behavior"] = "Isolated".into()),
    SourceField::PreTokenizer
  );
  assert_eq!(
    refused(&|d| d["model"]["byte_fallback"] = false.into()),
    SourceField::Model
  );
  assert_eq!(
    refused(&|d| d["added_tokens"][4]["lstrip"] = true.into()),
    SourceField::AddedTokens
  );
  assert_eq!(
    refused(&|d| d["model"]["merges"][0][1] = "zz".into()),
    SourceField::Merges
  );
  let crossing = |d: &mut serde_json::Value| {
    let vocab = d["model"]["vocab"].as_object_mut().unwrap();
    vocab.insert("a\u{2581}".into(), 286.into());
    vocab.insert("a\u{2581}a".into(), 287.into());
    d["model"]["merges"]
      .as_array_mut()
      .unwrap()
      .push(serde_json::json!(["a\u{2581}", "a"]));
  };
  let mut copy = doc.clone();
  crossing(&mut copy);
  match compile_image(&serde_json::to_vec(&copy).unwrap()) {
    Err(Error::TokenizerSource(defect)) => {
      assert_eq!(defect.field(), SourceField::Merges);
      assert!(
        defect.detail().contains("crosses a word boundary"),
        "{}",
        defect.detail()
      );
    }
    other => panic!("expected a merge refusal, got {:?}", other.map(|b| b.len())),
  }
  assert_eq!(
    refused(&|d| {
      d["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("<0x41>");
    }),
    SourceField::Vocabulary,
    "a missing id breaks the dense vocabulary"
  );
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

/// M13's compile half: the staged `tokenizer.json` compiles to an image that
/// opens under the upstream identity and reads back its known tokens; the
/// compile's duration and the image's size are printed.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_tokenizer_compiles_to_the_upstream_image() {
  let json = std::fs::read(staged_dir().join("tokenizer.json")).expect("the staged tokenizer.json");
  let started = std::time::Instant::now();
  let bytes = compile_image(&json).expect("the staged tokenizer.json compiles");
  let elapsed = started.elapsed();
  let file = written(&bytes);
  let image =
    TokenizerImage::open(file.path(), &UPSTREAM).expect("opens under the upstream identity");
  assert_eq!(image.counts(), UPSTREAM.counts);
  assert_eq!(image.token(2), Some("<bos>".as_bytes()));
  assert_eq!(image.scalar_id('\u{2581}'), Some(236_743));
  let bridges: Vec<(Vec<u8>, Vec<u8>)> = image
    .bridges()
    .map(|(l, r)| {
      (
        image.token(l).unwrap().to_vec(),
        image.token(r).unwrap().to_vec(),
      )
    })
    .collect();
  assert_eq!(bridges, [(b">".to_vec(), "\u{2581}</".as_bytes().to_vec())]);
  assert_eq!(
    image_name(&UPSTREAM.source),
    "tokenizer.v1-4d777ef5bdc1.gemmabpe"
  );
  eprintln!(
    "M13: compiled {} bytes ({:.2} MiB) in {elapsed:?}",
    bytes.len(),
    bytes.len() as f64 / (1024.0 * 1024.0)
  );
}

/// The lists `tokenizer.json`'s typed parse fills grow through the door, so
/// a list that cannot be held is refused by name at `source_parse`, not as a
/// malformed document.
#[test]
fn a_refused_parse_list_is_answered_by_name() {
  use super::super::super::{alloc, error::AllocationSite};

  let _refuse = alloc::inject::refuse(AllocationSite::SourceParse, 0);
  match compile_image(MINI) {
    Err(Error::Allocation(refused)) => assert_eq!(refused.site(), AllocationSite::SourceParse),
    other => panic!(
      "expected the parse list refused by name, got {:?}",
      other.map(|image| image.len())
    ),
  }
}

/// The mini document with `edit` applied, compiled.
fn compiled_with(edit: &dyn Fn(&mut serde_json::Value)) -> Result<Vec<u8>> {
  let mut doc: serde_json::Value = serde_json::from_slice(MINI).expect("JSON");
  edit(&mut doc);
  compile_image(&serde_json::to_vec(&doc).unwrap())
}

/// The part of the document the compiler names when it refuses `edit`.
fn refused_with(edit: &dyn Fn(&mut serde_json::Value)) -> SourceField {
  match compiled_with(edit) {
    Err(Error::TokenizerSource(defect)) => defect.field(),
    other => panic!(
      "expected a source refusal, got {:?}",
      other.map(|b| b.len())
    ),
  }
}

/// A refusal the parse raises inside a configuration object names that
/// object: a key the object does not declare, a value of another type. A
/// `truncation` that is not `null` is the document's.
#[test]
fn a_refusal_raised_inside_a_configuration_object_names_it() {
  assert_eq!(
    refused_with(&|d| d["normalizer"]["pattern"] = serde_json::json!({"Regex": " "})),
    SourceField::Normalizer
  );
  assert_eq!(
    refused_with(&|d| d["pre_tokenizer"]["invert"] = "false".into()),
    SourceField::PreTokenizer
  );
  assert_eq!(
    refused_with(&|d| d["post_processor"]["single"][0]["SpecialToken"]["type_id"] = "0".into()),
    SourceField::PostProcessor
  );
  assert_eq!(
    refused_with(&|d| d["truncation"] = serde_json::json!({"max_length": 512})),
    SourceField::Document
  );
}

/// A template piece carries one kind: a piece that carries both is refused
/// as the post-processor's, though each key is one a piece declares.
#[test]
fn a_template_piece_carrying_both_kinds_is_refused() {
  assert_eq!(
    refused_with(&|d| {
      d["post_processor"]["single"][1]["SpecialToken"] =
        serde_json::json!({"id": "<bos>", "type_id": 0});
    }),
    SourceField::PostProcessor
  );
}

/// The post-processor's other keys are skipped, not validated: a key it does
/// not declare, whatever it holds, compiles.
#[test]
fn the_post_processors_other_keys_are_skipped() {
  compiled_with(&|d| d["post_processor"]["note"] = serde_json::json!({"any": [1, 2.5, null]}))
    .expect("the post-processor's other keys are skipped");
}
