use super::*;

#[test]
fn aligner_error_wraps_load_via_from() {
  let inner = crate::LoadError::NotFound("base960h_aligner.mlmodelc".into());
  let e: AlignerError = inner.into();
  assert!(matches!(e, AlignerError::Load(_)));
}

#[test]
fn aligner_error_load_displays_inner_message() {
  let e: AlignerError = crate::LoadError::NotFound("/tmp/missing.mlmodelc".into()).into();
  assert!(e.to_string().contains("/tmp/missing.mlmodelc"));
}

#[test]
fn aligner_error_contract_mismatch_displays_feature_and_shapes() {
  let e = AlignerError::ContractMismatch(ContractMismatch::new(
    "emissions",
    "[1, 2999, 29] f32".to_string(),
    "[1, 2999, 32] f32".to_string(),
  ));
  let rendered = e.to_string();
  assert!(rendered.contains("emissions"));
  assert!(rendered.contains("2999, 29"));
  assert!(rendered.contains("2999, 32"));
}

#[test]
fn aligner_error_is_equatable_and_cloneable() {
  let a = AlignerError::ContractMismatch(ContractMismatch::new(
    "waveform",
    "[1, 960000] f32".to_string(),
    "[1, 480000] f32".to_string(),
  ));
  let b = a.clone();
  assert_eq!(a, b);
}

#[test]
fn align_error_wraps_prediction_via_from() {
  let inner = crate::PredictionError::MissingOutput("emissions".to_string());
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Prediction(_)));
  assert!(e.to_string().contains("emissions"));
}

#[test]
fn align_error_wraps_tensor_via_from() {
  let inner = crate::TensorError::ShapeMismatch(crate::ShapeMismatch::new(960_000, 100));
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Tensor(_)));
  assert!(e.to_string().contains("960000") || e.to_string().contains("960_000"));
}

#[test]
fn align_error_wraps_emissions_via_from_and_is_transparent() {
  let inner = asry::emissions::EmissionsError::StrideMismatch(
    asry::emissions::EmissionsFailure::new("T · hop outside real_samples ± 2·hop".into()),
  );
  let displayed_inner = inner.to_string();
  let e: AlignError = inner.clone().into();
  assert!(matches!(e, AlignError::Alignment(_)));
  // `#[error(transparent)]` forwards Display verbatim, no extra wrapper text.
  assert_eq!(e.to_string(), displayed_inner);
}

#[test]
fn align_error_wraps_span_via_from_and_is_transparent() {
  let inner = asry::emissions::SpanError::Timebase {
    expected: 16_000,
    num: 1,
    den: 1_000,
  };
  let displayed_inner = inner.to_string();
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Span(_)));
  assert_eq!(e.to_string(), displayed_inner);
}

#[test]
fn aligner_error_wraps_seam_via_from() {
  let inner = asry::emissions::EmissionsError::Config(asry::emissions::EmissionsFailure::new(
    "bad tokenizer".into(),
  ));
  let e: AlignerError = inner.into();
  assert!(matches!(e, AlignerError::Seam(_)));
  assert!(e.to_string().contains("bad tokenizer"));
}

#[test]
fn align_error_input_too_long_displays_both_counts() {
  let e = AlignError::InputTooLong(InputTooLong::new(
    1_000_000,
    crate::audio::align::encode::ENCODER_WINDOW_SAMPLES,
  ));
  let rendered = e.to_string();
  assert!(rendered.contains("1000000"));
  assert!(rendered.contains("960000"));
}

#[test]
fn align_error_corrupt_emissions_names_the_placement_and_the_band() {
  // The error exists to be SELF-DIAGNOSING: a caller who flipped
  // `with_compute` must be able to read the cause straight off the message,
  // without knowing anything about fp16 subnormals. So the placement, the band
  // that was tripped, the observed minimum and the blast radius all have to
  // survive into Display. The real ANE numbers, measured on jfk.wav.
  let e = AlignError::CorruptEmissions(CorruptEmissions::new(
    crate::ComputeUnits::All,
    crate::audio::align::acoustic::SentinelBand::Fp16Saturation,
    -45_440.0,
    2_667,
    15_921,
  ));
  let rendered = e.to_string();
  assert!(
    rendered.contains("All"),
    "must name the placement: {rendered}"
  );
  assert!(
    rendered.contains("-45440"),
    "must report the min: {rendered}"
  );
  assert!(
    rendered.contains("2667"),
    "must report the blast radius: {rendered}"
  );
  assert!(
    rendered.contains("15921"),
    "must report the total: {rendered}"
  );
  assert!(
    rendered.contains("-32768") && rendered.contains("Fp16Saturation"),
    "must name the band it tripped: {rendered}"
  );
  assert!(
    rendered.contains("DEFAULT_ENCODER_COMPUTE"),
    "must name the way out: {rendered}"
  );
}

#[test]
fn align_error_unnormalized_emissions_names_the_frame_and_logsumexp() {
  // Self-diagnosing, like CorruptEmissions: a caller must read the cause — a
  // model artifact swapped for a raw-logit head — straight off the message, so
  // the offending frame, its logsumexp, the tolerance and the placement all
  // survive into Display.
  let tolerance = crate::audio::align::encode::log_prob_sum_tolerance(
    core::num::NonZeroUsize::new(29).expect("nonzero"),
  );
  let e = AlignError::UnnormalizedEmissions(UnnormalizedEmissions::new(
    crate::ComputeUnits::All,
    2_832,
    6.63,
    tolerance,
  ));
  let rendered = e.to_string();
  assert!(rendered.contains("2832"), "must name the frame: {rendered}");
  assert!(
    rendered.contains("6.63"),
    "must report the logsumexp: {rendered}"
  );
  assert!(
    rendered.contains(&tolerance.to_string()),
    "must report the tolerance: {rendered}"
  );
  assert!(
    rendered.contains("All"),
    "must name the placement: {rendered}"
  );
  assert!(
    rendered.contains("raw-logit"),
    "must name the likely cause: {rendered}"
  );
}

#[test]
fn decision_language_display_separates_found_from_requested() {
  // `new` takes (requested, found); the message prints the FOUND language
  // first and the REQUESTED one second, which is the pair a
  // positional-argument slip would swap without changing the prose.
  let e = AlignError::DecisionLanguage(DecisionLanguage::new(asry::Lang::En, asry::Lang::Zh));
  let rendered = e.to_string();
  assert!(
    rendered.starts_with("the OOV decisions were made for language "),
    "{rendered}"
  );
  assert!(
    rendered.contains("made for language Zh but the chunk is being aligned for En"),
    "{rendered}"
  );
  assert!(rendered.contains("AlignmentSet::detect_oov"), "{rendered}");
}

#[test]
fn vocabulary_mismatch_names_both_widths_and_the_remedy() {
  let e = AlignerError::VocabularyMismatch(VocabularyMismatch::new(30, 29));
  let rendered = e.to_string();
  assert!(
    rendered.contains("names 30 classes"),
    "must name the vocabulary's width: {rendered}"
  );
  assert!(
    rendered.contains("scores 29 per frame"),
    "must name the model's width: {rendered}"
  );
  assert!(
    rendered.contains("ships beside it"),
    "must name the way out: {rendered}"
  );
  // The new variant keeps `AlignerError` equatable and cloneable.
  assert_eq!(e.clone(), e);
}

/// A normalizer that strips one trailing `!` from a word and reports it as one
/// trailing wildcard. asry's own normalizers report no boundary padding, so a
/// `BoundaryPunct` event comes only from a normalizer like this one.
struct TrailingBang;

impl asry::emissions::TextNormalizer for TrailingBang {
  fn normalize<'a>(
    &self,
    text: &'a str,
  ) -> Result<asry::emissions::NormalizedText<'a>, asry::emissions::NormalizationError> {
    let mut normalized = Vec::new();
    let mut original = Vec::new();
    let mut padding = Vec::new();
    for word in text.split_whitespace() {
      let bare = word.strip_suffix('!');
      normalized.push(bare.unwrap_or(word).to_lowercase());
      original.push(std::borrow::Cow::Borrowed(word));
      padding.push(asry::emissions::WildcardBoundary::new(
        0,
        u32::from(bare.is_some()),
      ));
    }
    Ok(asry::emissions::NormalizedText::with_wildcards(
      normalized.join(" "),
      original,
      padding,
    ))
  }
}

/// The events the bundled table's seam detects in `text` under
/// [`TrailingBang`]: an event is made only by detection.
fn detected_events(text: &str) -> Vec<asry::emissions::OovEvent> {
  asry::emissions::EmissionsAligner::builder(
    asry::Lang::En,
    crate::audio::align::vocab::tokenizer_json_bytes(),
  )
  .normalizer(Box::new(TrailingBang))
  .blank_token_id(crate::audio::align::vocab::BLANK_ID)
  .build()
  .expect("the bundled seam builds")
  .detect_oov(text)
  .expect("detect_oov")
  .events()
  .to_vec()
}

/// A refusal names every refused position: the character where one survives,
/// and the boundary mark — whose character the normalizer removed — as such.
#[test]
fn refused_display_names_every_refused_position() {
  let refusal = refusal_of("go AT&T b4d!");
  assert_eq!(
    refusal.to_string(),
    "'&' (word 1), '4' (word 2), a boundary mark (word 2)"
  );
  assert_eq!(
    AlignError::Refused(refusal).to_string(),
    "the OOV policy refused this chunk at '&' (word 1), '4' (word 2), a boundary mark (word \
     2); no word timings were produced"
  );
  assert_eq!(Refusal::new(Vec::new()).to_string(), "no position");
}

/// A refusal of every event the bundled seam detects in `text`, each under the
/// language it was read in (English).
fn refusal_of(text: &str) -> Refusal {
  Refusal::new(
    detected_events(text)
      .iter()
      .map(RefusedOov::detected)
      .collect(),
  )
}

/// **A refusal restated for a request names every position under it, and holds
/// no asry event**: read in English, restated for a Korean request, every
/// position reports Korean, and neither its debug form nor its error's names
/// English. The positions themselves are unchanged.
#[test]
fn a_refusal_restated_for_a_request_names_every_position_under_it() {
  let read = refusal_of("go AT&T b4d!");
  assert!(
    read
      .events()
      .iter()
      .all(|event| event.language() == &asry::Lang::En)
  );
  let restated = read.clone().under(&asry::Lang::Ko);
  assert_eq!(restated.events().len(), 3);
  for (before, after) in read.events().iter().zip(restated.events()) {
    assert_eq!(after.language(), &asry::Lang::Ko);
    assert_eq!(
      (
        after.kind(),
        after.char(),
        after.char_index(),
        after.word_index()
      ),
      (
        before.kind(),
        before.char(),
        before.char_index(),
        before.word_index()
      )
    );
  }
  let shown = format!("{:?}", AlignError::Refused(restated));
  assert!(shown.contains("language: Ko"), "{shown}");
  assert!(!shown.contains("En"), "{shown}");
}

#[test]
fn no_alignment_path_carries_asrys_diagnostic() {
  let e = AlignError::NoAlignmentPath(asry::emissions::EmissionsFailure::new(
    "emissions shorter than the token count".into(),
  ));
  assert_eq!(
    e.to_string(),
    "no alignment path for this chunk: emissions shorter than the token count"
  );
}

#[test]
fn vocabulary_errors_name_what_is_wrong_with_the_table() {
  assert_eq!(
    VocabularyError::MissingId(MissingId::new(1, 2)).to_string(),
    "no token has id 1: a vocabulary of 2 entries names every id in 0..2 exactly once, one per \
     class of its model's CTC head"
  );
  assert!(
    VocabularyError::Empty
      .to_string()
      .contains("names no token")
  );
  assert!(
    VocabularyError::DuplicateToken("A".to_owned())
      .to_string()
      .contains("names the token \"A\" twice")
  );
  let read = VocabularyError::Read(VocabularyRead::new(
    "/models/fr_dict.json".into(),
    std::io::Error::from(std::io::ErrorKind::NotFound),
  ));
  assert!(read.to_string().contains("/models/fr_dict.json"), "{read}");
}

/// A geometry the model's declaration contradicts is refused naming the
/// geometry, the window and both frame counts; the variant keeps `AlignerError`
/// equatable and cloneable.
#[test]
fn frame_count_mismatch_names_the_geometry_and_both_counts() {
  use core::num::NonZeroU32;

  let geometry = crate::audio::align::acoustic::AcousticGeometry::new(
    16_000,
    NonZeroU32::new(400).expect("nonzero"),
    NonZeroU32::new(160).expect("nonzero"),
  )
  .expect("a geometry");
  let e = AlignerError::FrameCountMismatch(FrameCountMismatch::new(geometry, 960_000, 2_999));
  assert_eq!(
    e.to_string(),
    "the contract's front end (400-sample receptive field, 160-sample stride) makes 5998 frames \
     of the model's 960000-sample window, but the model declares 2999: the geometry is not this \
     model's"
  );
  assert_eq!(e.clone(), e);
}

/// A blank that is no id of the table is refused naming the blank and the
/// table's ids.
#[test]
fn blank_out_of_vocabulary_names_the_blank_and_the_ids() {
  let e = AlignerError::BlankOutOfVocabulary(BlankOutOfVocabulary::new(29, 29));
  assert_eq!(
    e.to_string(),
    "the contract names id 29 the CTC blank, but the vocabulary's 29 entries hold ids 0..29; \
     the blank must be one of the table's own ids"
  );
  assert_eq!(e.clone(), e);
}

/// A geometry asry's seam cannot time names the reason: the rate.
#[test]
fn geometry_errors_name_what_the_seam_cannot_time() {
  assert!(
    GeometryError::SampleRate(8_000)
      .to_string()
      .starts_with("the front end takes 8000 Hz audio")
  );
}

/// A contradicted tokenization names what disagrees, and the load-time variant
/// keeps `AlignerError` equatable and cloneable.
#[test]
fn tokenization_errors_name_what_disagrees() {
  assert!(
    TokenizationError::UnsupportedDelimiter("_".to_owned())
      .to_string()
      .contains("delimits words with \"_\", and a contract states the `|` delimiter, the space")
  );
  assert!(
    TokenizationError::UpperWithLowercase('b')
      .to_string()
      .contains("also spells 'b'")
  );
  let e = AlignerError::Tokenization(TokenizationError::ProjectedAsWritten);
  assert!(
    e.to_string()
      .starts_with("the contract's tokenization cannot be honoured: ")
  );
  assert_eq!(e.clone(), e);
  assert!(
    TokenizationError::EmptySpecial(4)
      .to_string()
      .contains("the table's empty token, id 4, a special")
  );
}

/// A seam reserving other columns than the contract declares names both sets,
/// and the variant keeps `AlignerError` equatable and cloneable.
#[test]
fn reserved_set_mismatch_names_both_sets() {
  let e = AlignerError::ReservedSetMismatch(ReservedSetMismatch::new(vec![0, 1, 4], vec![0, 1]));
  let rendered = e.to_string();
  assert!(
    rendered.contains("the seam reserves the columns [0, 1] where the contract declares [0, 1, 4]"),
    "{rendered}"
  );
  assert_eq!(e.clone(), e);
}

/// A log-probability head too wide to check names its width, the allowance it
/// would need and the widest checkable width, and says what to state instead.
#[test]
fn unprovable_normalization_names_the_width_and_the_road_out() {
  let width = core::num::NonZeroUsize::new(5_000).expect("nonzero");
  let e = AlignerError::UnprovableNormalization(UnprovableNormalization::new(width, 353));
  let rendered = e.to_string();
  assert!(
    rendered.starts_with("a 5000-class head's log-probabilities"),
    "{rendered}"
  );
  assert!(rendered.contains("up to 4.884"), "{rendered}");
  assert!(rendered.contains("only up to 353 classes"), "{rendered}");
  assert!(
    rendered.contains("State the head's output as logits"),
    "{rendered}"
  );
  assert_eq!(e.clone(), e);
}

/// An output-shape mismatch names both shapes, so a transposed head reads as
/// what it is.
#[test]
fn output_shape_names_both_shapes() {
  let rendered =
    AlignError::OutputShape(OutputShape::new(vec![1, 29, 2999], vec![1, 2999, 29])).to_string();
  assert!(rendered.contains("[1, 29, 2999]"), "{rendered}");
  assert!(rendered.contains("[1, 2999, 29]"), "{rendered}");
}

#[test]
fn language_unsupported_display_names_the_language_and_the_policy() {
  let rendered = AlignError::LanguageUnsupported(asry::Lang::Zh).to_string();
  assert!(
    rendered.contains("no aligner registered for language Zh"),
    "{rendered}"
  );
  assert!(rendered.contains("no `Any` fallback"), "{rendered}");
  assert!(rendered.contains("`Error`"), "{rendered}");
}
