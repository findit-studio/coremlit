use super::*;

use asry::emissions::{EmissionsFailure, EnglishNormalizer};

fn normalizer() -> DynTextNormalizer {
  Box::new(EnglishNormalizer::new())
}

// ---------------------------------------------------------------------
// AlignerOptions (rust-options-pattern)
// ---------------------------------------------------------------------

#[test]
fn options_new_matches_documented_defaults() {
  let o = AlignerOptions::new();
  assert_eq!(o.min_speech_coverage(), DEFAULT_MIN_SPEECH_COVERAGE);
  assert_eq!(o.min_speech_coverage(), 0.5);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  // The shipping placement is CpuOnly, and it is a correctness requirement:
  // the ANE placements underflow this model's fp16 `log(softmax(·))` tail to a
  // `-45440` sentinel. See `DEFAULT_ENCODER_COMPUTE`.
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(o.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_compute_overrides() {
  // Override with placements that are NOT the default, or this would pass
  // against a no-op `with_compute`.
  let o = AlignerOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(o.compute(), ComputeUnits::CpuAndGpu);

  let mut o = AlignerOptions::new();
  o.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(o.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(AlignerOptions::default(), AlignerOptions::new());
}

#[test]
fn options_with_builders_override() {
  let o = AlignerOptions::new()
    .with_min_speech_coverage(0.75)
    .with_max_intra_silent_run(Duration::from_millis(120));
  assert_eq!(o.min_speech_coverage(), 0.75);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(120));
}

#[test]
fn options_set_in_place() {
  let mut o = AlignerOptions::new();
  o.set_min_speech_coverage(0.25);
  o.set_max_intra_silent_run(Duration::from_millis(40));
  assert_eq!(o.min_speech_coverage(), 0.25);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(40));
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_fields_default() {
  let o: AlignerOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, AlignerOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_fills_defaults() {
  let o: AlignerOptions = serde_json::from_str(r#"{"min_speech_coverage":0.7}"#).unwrap();
  assert_eq!(o.min_speech_coverage(), 0.7);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips() {
  // A non-default compute, so the round-trip proves the field actually
  // survives serialization rather than being re-defaulted on the way back.
  let o = AlignerOptions::new()
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&o).unwrap();
  assert!(json.contains("cpu_and_gpu"), "round-tripped json: {json}");
  let back: AlignerOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(o, back);
}

// ---------------------------------------------------------------------
// Seam construction / blank-id wiring (DECISION 5) — hermetic: these
// build the asry seam alone (bundled tokenizer bytes + a normalizer), no
// CoreML model, so they run without ALIGNKIT_TEST_MODELS.
// ---------------------------------------------------------------------

#[test]
fn build_seam_wires_blank_id_zero_and_vocab_29() {
  let seam = build_seam(Lang::En, normalizer(), &AlignerOptions::new())
    .expect("bundled tokenizer + explicit blank id builds");
  assert_eq!(seam.blank_token_id(), crate::vocab::BLANK_ID);
  assert_eq!(seam.blank_token_id(), 0);
  assert_eq!(seam.vocab_size().get(), crate::vocab::VOCAB_SIZE);
}

#[test]
fn build_seam_threads_options_into_the_seam() {
  let options = AlignerOptions::new().with_max_intra_silent_run(Duration::from_millis(120));
  let seam = build_seam(Lang::En, normalizer(), &options).expect("builds");
  assert_eq!(seam.max_intra_silent_run(), options.max_intra_silent_run());
}

#[test]
fn seam_stride_is_the_encoder_stride() {
  // THE one-stride invariant. The stride that TIMES the words (asry's seam)
  // and the stride that TRUNCATES the emissions
  // (`encode::truncated_frame_count`, which divides by
  // `encode::HOP_SAMPLES`) must be the same number, or every word is skewed
  // in proportion to the difference. They are not independently checked
  // downstream: asry's `validate_stride_extent` allows `chunk_extent ± 2·hop`,
  // which on jfk.wav accepts 319, 320 AND 321 without error.
  //
  // This held only by coincidence while `AlignerOptions::hop_samples` existed
  // (it fed the seam, never the encoder); it now holds by construction, since
  // `SEAM_HOP_SAMPLES` is DERIVED from `encode::HOP_SAMPLES`. A mutant that
  // re-spells the seam's stride as a literal fails here.
  let seam = build_seam(Lang::En, normalizer(), &AlignerOptions::new()).expect("builds");
  assert_eq!(seam.hop_samples(), SEAM_HOP_SAMPLES);
  assert_eq!(
    seam.hop_samples().get() as usize,
    crate::encode::HOP_SAMPLES,
    "the seam's word-timing stride must equal the encoder's truncation stride"
  );
}

#[test]
fn bundled_tokenizer_has_no_autodetectable_blank() {
  // Proves the explicit `.blank_token_id(BLANK_ID)` in `build_seam` is
  // load-bearing: WITHOUT it, asry's default `<pad>` / `[PAD]` / `<blank>`
  // auto-detect finds nothing in the chordai vocab and construction FAILS.
  // A mutant dropping that override would regress to exactly this error.
  let result = EmissionsAligner::builder(Lang::En, crate::vocab::tokenizer_json_bytes())
    .normalizer(normalizer())
    .build();
  assert!(
    matches!(result, Err(EmissionsError::Config(_))),
    "auto-detect must fail without an explicit blank id"
  );
}

// ---------------------------------------------------------------------
// F3: options() reports EFFECTIVE (post-clamp) state, not the requested value.
// The transform is hermetic (build the seam, read its applied coverage back);
// the wiring through the real `Aligner::options()` is model-gated below.
// ---------------------------------------------------------------------

#[test]
fn effective_options_reports_the_seams_clamped_coverage_not_the_requested_value() {
  // Out-of-range requests are coerced by the seam; effective options must report
  // the coerced value, so `Aligner::options()` never lies about the filter in
  // force. A mutant that stored/returned the requested value fails here.
  for (requested, effective) in [(2.0_f32, 1.0_f32), (-0.25, 0.0)] {
    let options = AlignerOptions::new().with_min_speech_coverage(requested);
    let seam = build_seam(Lang::En, normalizer(), &options).expect("builds");
    let eff = effective_options(&seam, &options);
    assert_eq!(
      eff.min_speech_coverage(),
      effective,
      "requested {requested} must report as {effective}"
    );
    // ...and it equals the seam's own applied value exactly (the source of truth).
    assert_eq!(eff.min_speech_coverage(), seam.min_speech_coverage().get());
  }

  // NaN → the seam's default, never NaN.
  let options = AlignerOptions::new().with_min_speech_coverage(f32::NAN);
  let seam = build_seam(Lang::En, normalizer(), &options).expect("builds");
  let eff = effective_options(&seam, &options);
  assert!(
    !eff.min_speech_coverage().is_nan(),
    "NaN must not survive into effective options"
  );
  assert_eq!(eff.min_speech_coverage(), DEFAULT_MIN_SPEECH_COVERAGE);
  assert_eq!(eff.min_speech_coverage(), seam.min_speech_coverage().get());
}

#[test]
fn effective_options_passes_through_the_uncoerced_fields() {
  // Only min_speech_coverage is coerced; max_intra_silent_run and compute pass
  // through from the request untouched. A mutant that rebuilt options from seam
  // defaults (dropping the request) fails here.
  let options = AlignerOptions::new()
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuAndGpu)
    .with_min_speech_coverage(2.0);
  let seam = build_seam(Lang::En, normalizer(), &options).expect("builds");
  let eff = effective_options(&seam, &options);
  assert_eq!(eff.max_intra_silent_run(), Duration::from_millis(120));
  assert_eq!(eff.compute(), ComputeUnits::CpuAndGpu);
  assert_eq!(eff.min_speech_coverage(), 1.0); // the one field that IS coerced
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn aligner_options_reports_effective_coverage_after_construction() {
  // The wiring proof: a real Aligner built with an out-of-range coverage must
  // report the CLAMPED value through `options()`. Reverting `from_paths_with` to
  // store the requested value fails exactly here (the mutation proof for F3's
  // wiring, which the hermetic `effective_options` tests cannot see).
  let aligner = Aligner::from_paths_with(
    Lang::En,
    &models_dir().join("base960h_aligner.mlmodelc"),
    normalizer(),
    AlignerOptions::new().with_min_speech_coverage(2.0),
  )
  .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");
  assert_eq!(
    aligner.options().min_speech_coverage(),
    1.0,
    "options() must report the seam's clamped coverage, not the requested 2.0"
  );
}

/// `ALIGNKIT_TEST_MODELS`, or `<workspace>/Models/alignkit` — the crate's
/// convention, duplicated here (as `encode::tests` and `registry::tests` do)
/// because a `src/` unit test cannot import the `tests/` integration crate.
fn models_dir() -> std::path::PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("alignkit")
    },
    std::path::PathBuf::from,
  )
}

// ---------------------------------------------------------------------
// Recoverable-subset mapping — the `align_chunk` policy, tested directly.
// ---------------------------------------------------------------------

fn failure(message: &str) -> EmissionsFailure {
  EmissionsFailure::new(message.into())
}

#[test]
fn recover_maps_no_alignment_path_to_empty_words() {
  let result =
    recover_or_error(EmissionsError::NoAlignmentPath(failure("no finite path"))).unwrap();
  assert!(result.words().is_empty());
}

#[test]
fn recover_maps_semantic_oov_to_empty_words() {
  let result = recover_or_error(EmissionsError::SemanticOutOfVocab(failure(
    "fail-closed OOV",
  )))
  .unwrap();
  assert!(result.words().is_empty());
}

#[test]
fn recover_propagates_non_recoverable_errors() {
  // A config / abort failure is a HARD error, never empty words — the exact
  // distinction that stops a broken setup from silently emitting empty
  // alignments forever.
  assert!(matches!(
    recover_or_error(EmissionsError::Config(failure("blank id >= V"))),
    Err(AlignError::Alignment(EmissionsError::Config(_)))
  ));
  assert!(matches!(
    recover_or_error(EmissionsError::Aborted(failure("aborted"))),
    Err(AlignError::Alignment(EmissionsError::Aborted(_)))
  ));
}

// ---------------------------------------------------------------------
// The `tracing` feature actually emits spans.
//
// The feature was declared in Cargo.toml, advertised in `lib.rs` ("structured
// spans over load and per-chunk alignment") and implemented NOWHERE: not one
// `tracing::` call-site existed anywhere in `src/`. A user who built
// `--features tracing` with a subscriber installed got zero spans and lost the
// afternoon to their own setup.
//
// Every gate missed it, and the reason generalises: `cargo hack check
// --each-feature` only COMPILES each feature, and an unused optional dependency
// compiles perfectly clean. Only EXECUTING a test under the feature can see
// this class of bug, so these must run under `cargo hack test --each-feature` —
// which means the load half below is deliberately hermetic (a missing model
// still opens the span, because `#[instrument]` opens it before the body runs).
// ---------------------------------------------------------------------

#[cfg(feature = "tracing")]
mod tracing_spans {
  use core::cell::RefCell;
  use std::sync::{
    Once,
    atomic::{AtomicU64, Ordering},
  };

  use super::*;

  // Why a GLOBAL subscriber with thread-local capture, and not the obvious
  // `with_default(subscriber, || ...)`:
  //
  // `tracing` caches an `Interest` per callsite, PROCESS-WIDE, the first time
  // that callsite is reached. A callsite first reached while no subscriber is
  // installed caches `Interest::never()` and is then dead for the rest of the
  // process. `with_default` installs a THREAD-LOCAL subscriber and does NOT
  // rebuild that cache (`rebuild_interest_cache` recomputes from the list of
  // GLOBAL dispatchers, which is empty in that case — it cannot help). Other
  // tests in this binary call `Encoder::emissions` with no subscriber at all,
  // so on a full `--ignored` run they killed the `alignkit.encoder.emissions`
  // callsite before this test ever ran and it captured three of its four spans.
  // Found exactly that way: green alone, red in the suite.
  //
  // `set_global_default` DOES rebuild the interest cache, and `enabled()` below
  // is unconditionally true, so every callsite lands on `Interest::always()` and
  // can never be re-poisoned. Capture is then armed per-thread, which is also
  // what keeps parallel tests from seeing each other's spans.
  //
  // This is a property of scoped subscribers in a multi-test process, NOT of
  // this crate: a real user calls `init()` / `set_global_default`, which takes
  // the same path this does. There is nothing to fix in the library.

  thread_local! {
    /// `Some` while this thread is capturing; the spans it has collected.
    static CAPTURED: RefCell<Option<Capture>> = const { RefCell::new(None) };
  }

  /// One captured span: its NAME, the NAMES of its declared diagnostic fields,
  /// and the id of its (contextual) parent — everything the `tracing` contract
  /// documents, so the gate proves the fields and the nesting, not merely the
  /// count.
  #[derive(Debug)]
  struct CapturedSpan {
    id: u64,
    name: &'static str,
    fields: Vec<&'static str>,
    parent: Option<u64>,
  }

  /// The per-thread capture buffer: the spans opened so far, and the stack of
  /// currently-entered span ids that resolves each new span's CONTEXTUAL parent
  /// — exactly as `tracing-subscriber`'s registry does, since a `#[instrument]`
  /// span's parent is whichever span is entered when it is created.
  #[derive(Debug, Default)]
  struct Capture {
    spans: Vec<CapturedSpan>,
    entered: Vec<u64>,
  }

  /// A capturing [`tracing::Subscriber`]: records the NAME, FIELD names, and
  /// (contextual) PARENT of every span opened on a thread that has armed
  /// [`CAPTURED`], and ignores every other thread.
  ///
  /// Hand-rolled rather than `tracing-subscriber`: the whole surface is seven
  /// trait methods, and a test-only dependency on a second tracing crate (with
  /// its own feature matrix) to assert the nesting and fields would cost more
  /// than it explains.
  struct CaptureSpans {
    next_id: AtomicU64,
  }

  impl tracing::Subscriber for CaptureSpans {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
      // Every level: the load spans are INFO and the per-chunk ones DEBUG, and
      // this test is about whether they EXIST, not about filtering. Being
      // unconditional is also what pins every callsite to `Interest::always()`.
      true
    }

    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
      // `Id::from_u64` rejects 0; `next_id` starts at 1.
      let id = self.next_id.fetch_add(1, Ordering::Relaxed);
      CAPTURED.with(|captured| {
        if let Some(capture) = captured.borrow_mut().as_mut() {
          // A `#[instrument]` span sets no explicit parent, so its parent is the
          // span currently entered on this thread (top of the stack), or none at
          // the root.
          let parent = capture.entered.last().copied();
          let fields = span
            .metadata()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect();
          capture.spans.push(CapturedSpan {
            id,
            name: span.metadata().name(),
            fields,
            parent,
          });
        }
      });
      tracing::span::Id::from_u64(id)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {}

    fn enter(&self, span: &tracing::span::Id) {
      CAPTURED.with(|captured| {
        if let Some(capture) = captured.borrow_mut().as_mut() {
          capture.entered.push(span.into_u64());
        }
      });
    }

    fn exit(&self, span: &tracing::span::Id) {
      CAPTURED.with(|captured| {
        if let Some(capture) = captured.borrow_mut().as_mut() {
          // `#[instrument]` enters/exits are strictly nested, so the exiting span
          // is the top on the happy path; find-and-remove anyway so an unexpected
          // order cannot corrupt the stack.
          if let Some(pos) = capture
            .entered
            .iter()
            .rposition(|&id| id == span.into_u64())
          {
            capture.entered.remove(pos);
          }
        }
      });
    }
  }

  /// Runs `body` with the capturing subscriber armed on this thread and returns
  /// every span it opened, in order.
  fn spans_opened_by(body: impl FnOnce()) -> Vec<CapturedSpan> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
      tracing::subscriber::set_global_default(CaptureSpans {
        next_id: AtomicU64::new(1),
      })
      .expect("no other subscriber may claim the global default in this test binary");
    });

    CAPTURED.with(|captured| *captured.borrow_mut() = Some(Capture::default()));
    body();
    CAPTURED.with(|captured| {
      captured
        .borrow_mut()
        .take()
        .expect("capture was armed above")
        .spans
    })
  }

  fn count(spans: &[CapturedSpan], name: &str) -> usize {
    spans.iter().filter(|span| span.name == name).count()
  }

  /// The first captured span with `name` — for the field / parent assertions.
  fn first<'a>(spans: &'a [CapturedSpan], name: &str) -> &'a CapturedSpan {
    spans
      .iter()
      .find(|span| span.name == name)
      .unwrap_or_else(|| panic!("no `{name}` span was captured; got {spans:?}"))
  }

  /// The name of `span`'s parent, resolved through the captured ids.
  fn parent_name<'a>(spans: &'a [CapturedSpan], span: &CapturedSpan) -> Option<&'a str> {
    let parent = span.parent?;
    spans
      .iter()
      .find(|candidate| candidate.id == parent)
      .map(|candidate| candidate.name)
  }

  /// Asserts `span` declares every documented field in `expected`.
  fn assert_has_fields(span: &CapturedSpan, expected: &[&str]) {
    for field in expected {
      assert!(
        span.fields.contains(field),
        "`{}` span must carry the documented `{field}` field; got {:?}",
        span.name,
        span.fields
      );
    }
  }

  /// **HERMETIC, and that is the point**: this runs under `cargo hack test
  /// --each-feature`, the only gate that can see the feature do nothing.
  ///
  /// `#[instrument]` opens the span before the function body runs, so a load
  /// that FAILS still emits one — which lets the load half of the contract be
  /// proven with no CoreML model at all. The `Err` is asserted too: without it
  /// this test would keep passing if the model path silently started resolving
  /// to something real.
  #[test]
  fn load_emits_a_span_even_when_the_model_is_missing() {
    let spans = spans_opened_by(|| {
      let result = Aligner::from_paths_with(
        Lang::En,
        Path::new("/nonexistent/base960h_aligner.mlmodelc"),
        normalizer(),
        AlignerOptions::new(),
      );
      assert!(
        matches!(result, Err(AlignerError::Load(_))),
        "the point of this path is that it fails; a load that succeeded would prove nothing \
         about the span"
      );
    });

    // The span NAMES are the feature's observable contract — a subscriber
    // filters and groups on them — so they are asserted as literals here rather
    // than read back from a constant that a rename would silently carry along.
    assert!(
      count(&spans, "alignkit.aligner.load") >= 1,
      "`--features tracing` must emit a load span; got {spans:?}"
    );
    assert!(
      count(&spans, "alignkit.encoder.load") >= 1,
      "the CoreML load must be its own nested span (it is where the wall-clock hides — 308 s on \
       a cold ANE placement); got {spans:?}"
    );

    // The documented diagnostic FIELDS (a subscriber renders these) — stripping
    // any one must fail here, not slip past a name count.
    assert_has_fields(
      first(&spans, "alignkit.aligner.load"),
      &["language", "model_path", "compute"],
    );
    assert_has_fields(first(&spans, "alignkit.encoder.load"), &["path", "compute"]);

    // NESTING (aligner/mod.rs:303): the CoreML load is a CHILD of the aligner
    // load — this holds even on the failing path, since `#[instrument]` opens
    // both spans before either body runs. A load moved out from under the aligner
    // span would keep both counts but lose this parent link.
    assert_eq!(
      parent_name(&spans, first(&spans, "alignkit.encoder.load")),
      Some("alignkit.aligner.load"),
      "`alignkit.encoder.load` must nest inside `alignkit.aligner.load`; got {spans:?}"
    );
  }

  /// The per-chunk half: **one `alignkit.align_chunk` span per call**, with the
  /// CoreML predict nested inside it. Model-gated, because a span over an
  /// alignment needs an alignment — so it runs ONLY under
  /// `cargo test -p alignkit --features tracing -- --ignored`, the one gate that
  /// both enables `tracing` and runs `#[ignore]` tests. `cargo hack test
  /// --each-feature` enables the feature but skips ignored tests; the plain
  /// `--ignored` runs do not enable `tracing`. Without that gate in the matrix,
  /// deleting the per-call `#[instrument]` attributes on `Aligner::align_chunk`
  /// or `Encoder::emissions` would be caught by nothing at all (F4) — see the
  /// crate-root "Gates" section, which now lists it.
  #[test]
  #[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
  fn every_align_chunk_call_opens_exactly_one_span() {
    let samples = load_jfk_wav();
    let text = "And so my fellow Americans ask not what your country can do for you, ask what \
                you can do for your country.";

    let spans = spans_opened_by(|| {
      let aligner = Aligner::from_paths(
        Lang::En,
        &models_dir().join("base960h_aligner.mlmodelc"),
        normalizer(),
      )
      .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");

      let events = aligner.detect_oov(text).expect("detect_oov");
      let decisions = asry::emissions::default_oov_decisions(&events);
      let abort = AtomicBool::new(false);

      // TWICE: "at least one span" would also pass against an `#[instrument]`
      // that somehow fired once per Aligner rather than once per chunk.
      for _ in 0..2 {
        let clock = OutputClock::new(0, asry::time::ANALYSIS_TIMEBASE, 0).expect("clock");
        let result = aligner
          .align_chunk(&samples, &[], text, clock, &abort, &decisions)
          .expect("align_chunk on the shipping default");
        assert!(!result.words().is_empty(), "jfk.wav must align to words");
      }
    });

    assert_eq!(
      count(&spans, "alignkit.align_chunk"),
      2,
      "one span per align_chunk call, no more and no fewer; got {spans:?}"
    );
    // EXACTLY two, tightened from `>= 2`: one predict per chunk — a second predict
    // inside a chunk would be a real regression, and zero on a chunk would be the
    // nesting bug the loop below catches.
    assert_eq!(
      count(&spans, "alignkit.encoder.emissions"),
      2,
      "exactly one CoreML predict span per chunk, two over two calls; got {spans:?}"
    );
    assert!(
      count(&spans, "alignkit.aligner.load") >= 1,
      "load must still be spanned on the success path; got {spans:?}"
    );

    // The documented diagnostic FIELDS: the per-chunk debugger aids
    // (aligner/mod.rs:427 — `sub_segments` / `text_bytes` / `samples` tell the two
    // empty-result paths apart) and the encoder's geometry/placement.
    assert_has_fields(
      first(&spans, "alignkit.align_chunk"),
      &[
        "language",
        "samples",
        "sub_segments",
        "text_bytes",
        "oov_decisions",
      ],
    );
    assert_has_fields(
      first(&spans, "alignkit.encoder.emissions"),
      &["encoder_input", "real_samples", "compute"],
    );

    // NESTING (aligner/mod.rs:424): EVERY emissions predict runs INSIDE an
    // align_chunk pass, not beside it. Moving `align_chunk` onto a helper invoked
    // after prediction would preserve both counts while un-nesting the predict —
    // this loop is what catches that.
    for span in spans
      .iter()
      .filter(|span| span.name == "alignkit.encoder.emissions")
    {
      assert_eq!(
        parent_name(&spans, span),
        Some("alignkit.align_chunk"),
        "each `alignkit.encoder.emissions` must nest inside `alignkit.align_chunk`; got {spans:?}"
      );
    }
  }

  /// `ALIGNKIT_TEST_MODELS`, or `<workspace>/Models/alignkit` — the crate's
  /// convention (`tests/common/mod.rs`), duplicated here because a `src/` unit
  /// test cannot import the `tests/` integration crate (the same duplication,
  /// for the same reason, as `encode::tests` and `registry::tests`).
  fn models_dir() -> std::path::PathBuf {
    std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
      || {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
          .join("../..")
          .join("Models")
          .join("alignkit")
      },
      std::path::PathBuf::from,
    )
  }

  /// The 11 s `jfk.wav` fixture, borrowed from the whisperkit crate by relative
  /// path (as `encode::tests` does) and failing LOUDLY if it ever moves.
  fn load_jfk_wav() -> Vec<f32> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("../whisperkit/tests/fixtures/audio/jfk.wav");
    let mut reader = hound::WavReader::open(&path)
      .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
    assert_eq!(reader.spec().sample_rate, 16_000, "fixture must be 16 kHz");
    reader
      .samples::<i16>()
      .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
      .collect()
  }
}
