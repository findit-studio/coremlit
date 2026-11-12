use super::*;

/// A `char`-count measure over one text. Both halves count `char`s — windit's
/// [`MeasureText`] and the exact range measure alike — so every geometry below
/// is arithmetic a reader can check by hand.
struct Chars<'t>(&'t str);

impl MeasureText for Chars<'_> {
  fn measure(&self, text: &str) -> usize {
    text.chars().count()
  }
}

impl RangeMeasure for Chars<'_> {
  type Error = Error;

  fn measure_range(&self, start: usize, end: usize) -> Result<usize, Error> {
    Ok(self.0[start..end].chars().count())
  }
}

/// The shared chunker over [`Chars`], its chunks as `(start, end)` byte ranges.
fn cut(text: &str, opts: &WindowOptions, ceiling: usize) -> Result<Vec<(usize, usize)>, Error> {
  chunk_long(&Chars(text), text, opts, ceiling)
    .map(|chunks| chunks.iter().map(|c| (c.start(), c.end())).collect())
}

/// The ceiling the geometry and refusal tables run at: granite's model window,
/// the ceiling their literals were cut under.
const CEILING: usize = 512;

/// An expected chunk list, as `(start, end)` byte ranges.
type Spans = &'static [(usize, usize)];

/// The shared chunker cuts exactly what granite's own chunker cut before the
/// machinery was shared. Every expected list is the output of granite's
/// pipeline — its `ContentAware` call, the gap repair, the whole-input fallback
/// and the cap re-bind — for the same text and options under the same
/// `char`-count measure, at granite's ceiling. A changed literal is a moved
/// granite boundary.
#[test]
fn the_shared_chunker_cuts_granites_geometry() {
  let w = WindowOptions::new;
  let cases: &[(&str, WindowOptions, Spans)] = &[
    // The gap repair's three outcomes. windit cuts the content only; the `\n\n`
    // between two chunks is a gap.
    // Left absorb: `aa\n\n` (4) still fits; `\n\nbbbbb` (7) would not. windit:
    // [0,2),[4,9).
    ("aa\n\nbbbbb", w(5), &[(0, 4), (4, 9)]),
    // Both neighbours could absorb (4 each); the left takes it. windit:
    // [0,2),[4,6).
    ("aa\n\nbb", w(5), &[(0, 4), (4, 6)]),
    // Right absorb: the left neighbour is full, `\n\nbbb` (5) fits. windit:
    // [0,5),[7,10).
    ("aaaaa\n\nbbb", w(5), &[(0, 5), (5, 10)]),
    // Own chunk: both neighbours full. windit: [0,5),[7,12).
    ("aaaaa\n\nbbbbb", w(5), &[(0, 5), (5, 7), (7, 12)]),
    // A leading gap, absorbed by the first chunk (then a right absorb, then an
    // own chunk), or emitted alone. windit: [2,4),[6,9); [2,4),[6,11); [2,7).
    ("\n\naa\n\nbbb", w(5), &[(0, 4), (4, 9)]),
    ("\n\naa\n\nbbbbb", w(5), &[(0, 4), (4, 6), (6, 11)]),
    ("\n\naaaaa", w(5), &[(0, 2), (2, 7)]),
    // A trailing gap, absorbed by the last chunk (after a left absorb), or
    // emitted alone. windit: [0,3),[5,7); [0,5).
    ("aaa\n\nbb\n\n", w(5), &[(0, 5), (5, 9)]),
    ("aaaaa\n\n", w(5), &[(0, 5), (5, 7)]),
    // A text that fits one window is one chunk, its separators included: windit
    // keeps it whole ([0,5)), so the repair has no gap to close.
    ("\n\naaa", w(5), &[(0, 5)]),
    ("aaa\n\n", w(5), &[(0, 5)]),
    // The word fallback leaves the `, ` between words out; each is reattached
    // left. windit: [0,3),[5,8),[10,13).
    ("aaa, bbb, ccc", w(5), &[(0, 5), (5, 10), (10, 13)]),
    // Whitespace-only text chunks to nothing in windit; the whole-input
    // fallback gives it one chunk.
    ("   ", w(5), &[(0, 3)]),
    ("     ", w(5), &[(0, 5)]),
    // Space runs as gaps: interior, leading and trailing own-chunks. windit:
    // [0,5),[10,15); [5,10); [0,5).
    ("aaaaa     bbbbb", w(5), &[(0, 5), (5, 10), (10, 15)]),
    ("     aaaaa", w(5), &[(0, 5), (5, 10)]),
    ("aaaaa     ", w(5), &[(0, 5), (5, 10)]),
    // The tail policy. Kept (the default, `PadFull`, and a minimum the tail
    // meets; windit: [0,3),[5,7)), the `\n\n` goes left; dropped below a
    // minimum of 3, windit returns [0,3) alone and the repair gives the tail
    // back as its own chunk, the `\n\n` heading it: the last boundary moves, no
    // byte is lost.
    ("aaa\n\nbb", w(5), &[(0, 5), (5, 7)]),
    (
      "aaa\n\nbb",
      w(5).with_tail(TailPolicy::PadFull),
      &[(0, 5), (5, 7)],
    ),
    (
      "aaa\n\nbb",
      w(5).with_tail(TailPolicy::DropBelowMin(2)),
      &[(0, 5), (5, 7)],
    ),
    (
      "aaa\n\nbb",
      w(5).with_tail(TailPolicy::DropBelowMin(3)),
      &[(0, 3), (3, 7)],
    ),
    // A lone chunk below the minimum: windit returns nothing; the whole-input
    // fallback still covers the text.
    (
      "aaa",
      w(5).with_tail(TailPolicy::DropBelowMin(4)),
      &[(0, 3)],
    ),
    // Overlap: the repeats are negative gaps the repair leaves alone.
    (
      "aa bb cc dd ee",
      w(5).with_overlap(2),
      &[(0, 5), (3, 8), (6, 11), (9, 14)],
    ),
    // `""` stays chunkless, under any cap.
    ("", w(5), &[]),
    ("", w(5).with_max_windows(0), &[]),
    // A cap the repaired list fits.
    (
      "aaaaa\n\nbbbbb",
      w(5).with_max_windows(3),
      &[(0, 5), (5, 7), (7, 12)],
    ),
    ("   ", w(5).with_max_windows(1), &[(0, 3)]),
  ];
  let mut moved = Vec::new();
  for &(text, opts, expected) in cases {
    match cut(text, &opts, CEILING) {
      Ok(got) if got == expected => {}
      other => moved.push(format!(
        "{text:?} at {opts}: expected {expected:?}, got {other:?}"
      )),
    }
  }
  assert!(
    moved.is_empty(),
    "the shared chunker moved granite's geometry:\n{}",
    moved.join("\n")
  );
}

/// The shared chunker refuses what granite's chunker refused, with the same
/// payload: windit's own errors pass through `Windowing` unchanged, the
/// `max_windows` cap re-binds on the FINAL list with `got` the full repaired
/// count, and a contentless run past the ceiling is refused wherever it is
/// measured — the whole-input fallback (before the cap is re-checked) and the
/// leading, interior and trailing own-chunks. Literals as in
/// `the_shared_chunker_cuts_granites_geometry`.
#[test]
fn the_shared_chunker_refuses_with_granites_payloads() {
  let w = WindowOptions::new;
  let windowing: &[(&str, WindowOptions, WinditError)] = &[
    // The repair's own-chunk makes 3 against a cap of 2.
    (
      "aaaaa\n\nbbbbb",
      w(5).with_max_windows(2),
      WinditError::TooManyWindows { got: 3, max: 2 },
    ),
    // Leading, interior and trailing insertions all count: 5 against 3, past
    // windit's own abort count of 4.
    (
      "\n\naaaaa\n\nbbbbb\n\n",
      w(5).with_max_windows(3),
      WinditError::TooManyWindows { got: 5, max: 3 },
    ),
    // The whole-input fallback is a window too.
    (
      "   ",
      w(5).with_max_windows(0),
      WinditError::TooManyWindows { got: 1, max: 0 },
    ),
    // windit's own refusals, unchanged.
    (
      "aaaaa\n\nbbbbb\n\nccccc",
      w(5).with_max_windows(1),
      WinditError::TooManyWindows { got: 2, max: 1 },
    ),
    ("aaa", w(0), WinditError::ZeroWindow),
    (
      "aaa",
      w(5).with_overlap(5),
      WinditError::OverlapGeWindow {
        overlap: 5,
        window: 5,
      },
    ),
  ];
  let mut wrong = Vec::new();
  for &(text, opts, expected) in windowing {
    match cut(text, &opts, CEILING) {
      Err(Error::Windowing(got)) if got == expected => {}
      other => wrong.push(format!(
        "{text:?} at {opts}: expected {expected:?}, got {other:?}"
      )),
    }
  }

  let spaces = " ".repeat(513);
  let contentless = [
    (spaces.clone(), w(512), (0, 513, 513, 512)),
    // Refused before the cap re-check, so not `TooManyWindows`.
    (
      spaces.clone(),
      w(512).with_max_windows(0),
      (0, 513, 513, 512),
    ),
    (format!("a{spaces}b"), w(1), (1, 514, 513, 512)),
    (format!("{spaces}a"), w(1), (0, 513, 513, 512)),
    (format!("a{spaces}"), w(1), (1, 514, 513, 512)),
  ];
  for (text, opts, expected) in &contentless {
    match cut(text, opts, CEILING) {
      Err(Error::ContentlessInputOverBudget(o))
        if (o.start(), o.end(), o.tokens(), o.max()) == *expected => {}
      other => wrong.push(format!(
        "{} bytes at {opts}: expected ContentlessInputOverBudget {expected:?}, got {other:?}",
        text.len()
      )),
    }
  }
  // At the ceiling exactly, the same run is one whole-input chunk.
  assert_eq!(
    cut(&" ".repeat(512), &w(512), CEILING).ok(),
    Some(vec![(0, 512)])
  );
  assert!(
    wrong.is_empty(),
    "the shared chunker refused differently:\n{}",
    wrong.join("\n")
  );
}

/// The ceiling is the door's parameter, never a number this module knows. At a
/// ceiling of 4 a contentless run of 5 is refused carrying 4, at every place
/// one is measured: the whole-input fallback (ahead of a cap of 0), and the
/// leading, interior and trailing own-chunks. At a ceiling of 5 the same texts
/// cut the geometry granite cut for them. `validate_long_input` reads the same
/// parameter.
#[test]
fn the_ceiling_is_the_doors_parameter() {
  let w = WindowOptions::new;
  let cases: &[(&str, WindowOptions, Spans, (usize, usize))] = &[
    ("     ", w(5), &[(0, 5)], (0, 5)),
    ("     aaaaa", w(5), &[(0, 5), (5, 10)], (0, 5)),
    (
      "aaaaa     bbbbb",
      w(5),
      &[(0, 5), (5, 10), (10, 15)],
      (5, 10),
    ),
    ("aaaaa     ", w(5), &[(0, 5), (5, 10)], (5, 10)),
  ];
  let mut wrong = Vec::new();
  for &(text, opts, fits, (start, end)) in cases {
    match cut(text, &opts, 4) {
      Err(Error::ContentlessInputOverBudget(o))
        if (o.start(), o.end(), o.tokens(), o.max()) == (start, end, 5, 4) => {}
      other => wrong.push(format!(
        "{text:?} at {opts}, ceiling 4: expected a refusal of {start}..{end} carrying 4, got \
         {other:?}"
      )),
    }
    match cut(text, &opts, 5) {
      Ok(got) if got == fits => {}
      other => wrong.push(format!(
        "{text:?} at {opts}, ceiling 5: expected {fits:?}, got {other:?}"
      )),
    }
  }
  // Under a cap of 0 the fallback's measure still runs first.
  match cut("     ", &w(5).with_max_windows(0), 4) {
    Err(Error::ContentlessInputOverBudget(o)) if (o.tokens(), o.max()) == (5, 4) => {}
    other => wrong.push(format!(
      "cap 0, ceiling 4: expected ContentlessInputOverBudget, got {other:?}"
    )),
  }
  match cut("     ", &w(5).with_max_windows(0), 5) {
    Err(Error::Windowing(WinditError::TooManyWindows { got: 1, max: 0 })) => {}
    other => wrong.push(format!(
      "cap 0, ceiling 5: expected TooManyWindows, got {other:?}"
    )),
  }
  assert!(
    wrong.is_empty(),
    "the ceiling is not the caller's:\n{}",
    wrong.join("\n")
  );

  let opts = LongTextOptions::from(w(5));
  match validate_long_input("any", &opts, 4) {
    Err(Error::WindowOverBudget(over)) => assert_eq!((over.window(), over.max()), (5, 4)),
    other => panic!("expected WindowOverBudget carrying 4, got {other:?}"),
  }
  assert!(validate_long_input("any", &opts, 5).is_ok());
}

/// Each shared refusal prints, through granite's `Error`, exactly what granite
/// printed for that variant before the machinery was shared — and the shared
/// message is the same text, so no conversion moves a message.
#[cfg(feature = "granite")]
#[test]
fn each_shared_refusal_prints_granites_message() {
  use crate::embeddings::granite;

  let cases: [(Error, &str); 4] = [
    (
      Error::Windowing(WinditError::TooManyWindows { got: 3, max: 2 }),
      "windowed-sequence processing failed: produced 3 windows, exceeding the maximum of 2",
    ),
    (
      Error::WindowOverBudget(WindowOverBudget::new(513, 512)),
      "embed_long window budget 513 exceeds the model's fixed 512-token input window",
    ),
    (
      Error::InputTooLarge(InputTooLarge::new(8_388_608, 1_048_576)),
      "text input is 8388608 bytes, exceeding the configured 1048576-byte limit",
    ),
    (
      Error::ContentlessInputOverBudget(ContentlessInputOverBudget::new(5, 10, 5, 4)),
      "contentless text at bytes 5..10 measures 5 tokens, exceeding the model's fixed 4-token \
       window",
    ),
  ];
  for (shared, printed) in cases {
    assert_eq!(shared.to_string(), printed, "the shared message moved");
    assert_eq!(
      granite::Error::from(shared).to_string(),
      printed,
      "granite's message moved"
    );
  }
}

/// The lines of the pre-lift golden that start with one of `prefixes`, in file
/// order. The golden's `# ` lines are commentary.
#[cfg(all(feature = "granite", feature = "serde"))]
fn golden_lines(prefixes: &[&str]) -> Vec<&'static str> {
  include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lifted_observables_812da2d.txt"
  ))
  .lines()
  .filter(|line| !line.starts_with("# "))
  .filter(|line| prefixes.iter().any(|prefix| line.starts_with(prefix)))
  .collect()
}

/// Every mismatch between a rendering and its golden lines, so a failure names
/// all the moved observables at once.
#[cfg(all(feature = "granite", feature = "serde"))]
fn mismatches(rendered: &[String], golden: &[&str]) -> Vec<String> {
  let mut out = Vec::new();
  if rendered.len() != golden.len() {
    out.push(format!(
      "line count: rendered {}, golden {}",
      rendered.len(),
      golden.len()
    ));
  }
  for (index, (got, want)) in rendered.iter().zip(golden).enumerate() {
    if got != want {
      out.push(format!(
        "line {index}\n  rendered: {got}\n  golden:   {want}"
      ));
    }
  }
  out
}

/// Every observable the long-text lift moved is what the build before the lift
/// printed: granite's refusal texts and `source`, the payloads' `Debug`, and
/// the options' `Display`, `Debug`, JSON, postcard, accessors, defaults and
/// unknown-field refusal.
#[cfg(all(feature = "granite", feature = "serde"))]
#[test]
fn the_long_text_observables_are_the_pre_lift_observables() {
  use crate::embeddings::granite;

  let mut rendered = Vec::new();
  rendered.push(format!(
    "granite::Error::WindowOverBudget: {}",
    granite::Error::WindowOverBudget(granite::WindowOverBudget::new(513, 512))
  ));
  rendered.push(format!(
    "granite::Error::InputTooLarge: {}",
    granite::Error::InputTooLarge(granite::error::InputTooLarge::new(8_388_608, 1_048_576))
  ));
  rendered.push(format!(
    "granite::Error::ContentlessInputOverBudget: {}",
    granite::Error::ContentlessInputOverBudget(granite::ContentlessInputOverBudget::new(
      5, 10, 5, 512
    ))
  ));
  let windowing =
    granite::Error::Windowing(granite::error::WinditError::TooManyWindows { got: 3, max: 2 });
  rendered.push(format!(
    "granite::Error::Windowing: {windowing} | source={:?}",
    std::error::Error::source(&windowing).map(|s| s.to_string())
  ));
  rendered.push(format!(
    "payload debug: {:?} {:?} {:?}",
    granite::WindowOverBudget::new(1, 2),
    granite::InputTooLarge::new(3, 4),
    granite::error::ContentlessInputOverBudget::new(5, 6, 7, 8)
  ));
  for options in [
    granite::LongTextOptions::new(),
    granite::LongTextOptions::default(),
    granite::LongTextOptions::from(granite::WindowOptions::new(64).with_overlap(8)),
    granite::LongTextOptions::new()
      .with_max_input_bytes(4096)
      .with_tail_policy(granite::TailPolicy::DropBelowMin(3)),
  ] {
    rendered.push(format!(
      "LongTextOptions: display={options} | debug={options:?} | json={} | postcard={:?} | tail={:?} | window={}",
      serde_json::to_string(&options).unwrap(),
      postcard::to_allocvec(&options).unwrap(),
      options.tail_policy(),
      options.window_options().window(),
    ));
  }
  rendered.push(format!(
    "LongTextOptions from {{}}: {:?}",
    serde_json::from_str::<granite::LongTextOptions>("{}").unwrap()
  ));
  rendered.push(format!(
    "LongTextOptions refusal: {}",
    serde_json::from_str::<granite::LongTextOptions>(r#"{"max_input_byte":4096}"#).unwrap_err()
  ));

  let golden = golden_lines(&[
    "granite::Error::",
    "payload debug: WindowOverBudget",
    "LongTextOptions",
  ]);
  assert_eq!(golden.len(), 11, "the long-text law covers 11 golden lines");
  let moved = mismatches(&rendered, &golden);
  assert!(
    moved.is_empty(),
    "{} observable(s) differ from the pre-lift golden:\n{}",
    moved.len(),
    moved.join("\n")
  );
}

/// Codex round 2 (LOW): a cap misplaced inside the tail policy, beside its
/// `kind`, is refused through these options rather than dropped, which would
/// leave `max_windows` silently unbounded. windit 0.5.2 refuses the stray key
/// in its nested wire enum.
#[cfg(feature = "serde")]
#[test]
fn a_cap_misplaced_in_the_tail_policy_is_refused() {
  let document = r#"{"window_options":{"window":512,"hop":512,"tail":{"kind":"keep_with_coverage","max_windows":1}}}"#;
  let refused = serde_json::from_str::<LongTextOptions>(document).unwrap_err();
  assert!(refused.to_string().contains("max_windows"), "{refused}");
}
