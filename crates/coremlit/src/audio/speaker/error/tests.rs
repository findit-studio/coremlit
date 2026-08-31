use super::*;

#[test]
fn model_error_wraps_load_via_from() {
  let inner = crate::LoadError::NotFound("seg.mlmodelc".into());
  let e: ModelError = inner.into();
  assert!(matches!(e, ModelError::Load(_)));
}

#[test]
fn model_error_contract_mismatch_displays_feature_and_shapes() {
  let e = ModelError::ContractMismatch(ContractMismatch::new(
    "segments",
    "[1, 589, 7] f32".to_string(),
    "[1, 592, 7] f32".to_string(),
  ));
  let rendered = e.to_string();
  assert!(rendered.contains("segments"));
  assert!(rendered.contains("589"));
  assert!(rendered.contains("592"));
}

#[test]
fn infer_error_wraps_prediction_and_tensor_via_from() {
  let e: InferError = crate::PredictionError::StateUnsupported.into();
  assert!(matches!(e, InferError::Prediction(_)));

  let e: InferError = crate::TensorError::ShapeMismatch(crate::ShapeMismatch::new(4, 2)).into();
  assert!(matches!(e, InferError::Tensor(_)));
}

#[test]
fn infer_error_non_finite_output_displays_index() {
  let e = InferError::NonFiniteOutput(42);
  assert_eq!(
    e.to_string(),
    "output contains a non-finite value at index 42"
  );
}

#[test]
fn infer_error_input_length_displays_got_and_expected() {
  let e = InferError::InputLength(InputLength::new(100, 160_000));
  let rendered = e.to_string();
  assert!(rendered.contains("100"));
  assert!(rendered.contains("160000"));
}

#[test]
fn infer_error_output_shape_displays_got_and_expected() {
  // Missing pin (T2 review-queue item): every other variant has a Display
  // test, but `OutputShape` (added in fix round 1, commit fcbce74) never
  // got one.
  let e = InferError::OutputShape(OutputShape::new(vec![1, 7, 589], vec![1, 589, 7]));
  let rendered = e.to_string();
  assert!(rendered.contains("[1, 7, 589]"));
  assert!(rendered.contains("[1, 589, 7]"));
}

#[test]
fn infer_error_non_finite_input_displays_index() {
  let e = InferError::NonFiniteInput(7);
  assert_eq!(
    e.to_string(),
    "input contains a non-finite value at index 7"
  );
}

#[test]
fn infer_error_f16_overflow_input_displays_index_and_honest_reason() {
  let e = InferError::F16OverflowInput(42);
  let rendered = e.to_string();
  assert!(rendered.contains("index 42"), "{rendered}");
  // The message must be HONEST that the value is finite in f32 (unlike
  // `NonFiniteInput`), and name f16 as the domain it overflows.
  assert!(
    rendered.contains("finite in f32") && rendered.contains("f16"),
    "{rendered}"
  );
}

#[test]
fn infer_error_empty_mask_displays_message() {
  let e = InferError::EmptyMask;
  assert_eq!(e.to_string(), "mask has no active (true) frame");
}

#[test]
fn extract_error_composes_model_arm() {
  let model_err: ModelError = crate::LoadError::NotFound("seg.mlmodelc".into()).into();
  let e: ExtractError = model_err.into();
  assert!(matches!(e, ExtractError::Model(ModelError::Load(_))));
}

#[test]
fn extract_error_composes_infer_arm() {
  let infer_err: InferError =
    crate::TensorError::ShapeMismatch(crate::ShapeMismatch::new(4, 2)).into();
  let e: ExtractError = infer_err.into();
  assert!(matches!(e, ExtractError::Infer(InferError::Tensor(_))));
}

#[test]
fn extract_error_empty_samples_displays_message() {
  assert_eq!(ExtractError::EmptySamples.to_string(), "samples is empty");
}

#[test]
fn extract_error_zero_step_samples_displays_message() {
  assert_eq!(
    ExtractError::ZeroStepSamples.to_string(),
    "step_samples must be > 0"
  );
}

#[test]
fn extract_error_step_samples_exceeds_window_displays_both() {
  let e = ExtractError::StepSamplesExceedsWindow(StepSamplesExceedsWindow::new(200_000, 160_000));
  let rendered = e.to_string();
  assert!(rendered.contains("200000"));
  assert!(rendered.contains("160000"));
}

/// Distinct from `StepSamplesExceedsWindow` above: that one rejects a step
/// too LARGE for any source, this one rejects a step the SELECTED source
/// cannot honor at all because its stride is compiled into the model graph
/// (`crate::audio::speaker::source::ArgmaxSource`). Both must render both numbers, so a
/// caller can see what it asked for AND what the source requires.
#[test]
fn extract_error_unsupported_step_samples_displays_both() {
  let e = ExtractError::UnsupportedStepSamples(UnsupportedStepSamples::new(8_000, 16_000));
  let rendered = e.to_string();
  assert!(rendered.contains("8000"));
  assert!(rendered.contains("16000"));
}

#[test]
fn extract_error_onset_out_of_range_displays_value() {
  let e = ExtractError::OnsetOutOfRange(1.5);
  let rendered = e.to_string();
  assert!(rendered.contains("1.5"));
  assert!(rendered.contains("(0.0, 1.0]"));
}

#[test]
fn extract_error_frame_count_mismatch_displays_both() {
  let e = ExtractError::FrameCountMismatch(FrameCountMismatch::new(589, 588));
  let rendered = e.to_string();
  assert!(rendered.contains("589"));
  assert!(rendered.contains("588"));
}

#[test]
fn extract_error_output_frame_count_overflow_displays_message() {
  let rendered = ExtractError::OutputFrameCountOverflow.to_string();
  assert!(rendered.contains("num_output_frames overflows usize"));
}

// ── try_from_parts payloads (issue #110) ────────────────────────────────
// The variants are NEWTYPES over named payload structs, per this repo's
// `rust-type-conventions` ("Variants are UNIT or NEWTYPE only — never
// struct-shaped … EXTRACT them into a named struct and wrap it"). These pin
// that each payload's accessors survive into the rendered message, which is
// what a caller debugging a message-assembly bug actually reads.

#[test]
fn zero_extraction_dimension_displays_the_part_name() {
  for (part, name) in [
    (ExtractionPart::NumChunks, "num_chunks"),
    (ExtractionPart::NumFramesPerChunk, "num_frames_per_chunk"),
    (ExtractionPart::Count, "count"),
  ] {
    let rendered = ExtractError::ZeroExtractionDimension(part).to_string();
    assert!(rendered.contains(name), "{rendered} must name {name}");
    assert!(rendered.contains("non-zero"), "{rendered}");
  }
}

#[test]
fn extraction_len_mismatch_displays_part_got_and_expected() {
  let m = ExtractionLenMismatch::new(ExtractionPart::RawEmbeddings, 767, 768);
  assert_eq!(m.part(), ExtractionPart::RawEmbeddings);
  assert_eq!(m.got(), 767);
  assert_eq!(m.expected(), 768);
  let rendered = ExtractError::ExtractionLenMismatch(m).to_string();
  assert!(rendered.contains("raw_embeddings"), "{rendered}");
  assert!(rendered.contains("767"), "{rendered}");
  assert!(rendered.contains("768"), "{rendered}");
}

#[test]
fn extraction_geometry_overflow_displays_part_and_both_dimensions() {
  let g = ExtractionGeometryOverflow::new(ExtractionPart::Segmentations, 4_294_967_296, 8);
  assert_eq!(g.part(), ExtractionPart::Segmentations);
  assert_eq!(g.num_chunks(), 4_294_967_296);
  assert_eq!(g.num_frames_per_chunk(), 8);
  let rendered = ExtractError::ExtractionGeometryOverflow(g).to_string();
  assert!(rendered.contains("segmentations"), "{rendered}");
  assert!(rendered.contains("4294967296"), "{rendered}");
  assert!(rendered.contains("overflows usize"), "{rendered}");
}

#[test]
fn invalid_sliding_window_displays_all_three_components() {
  let w = crate::audio::speaker::window::SlidingWindow::new(0.0, 10.0, 0.0);
  let e = InvalidSlidingWindow::new(ExtractionPart::ChunksSw, w);
  assert_eq!(e.part(), ExtractionPart::ChunksSw);
  assert_eq!(e.window(), w);
  let rendered = ExtractError::InvalidSlidingWindow(e).to_string();
  assert!(rendered.contains("chunks_sw"), "{rendered}");
  assert!(rendered.contains("10"), "{rendered}");
  assert!(rendered.contains("step 0"), "{rendered}");
}

#[test]
fn misaligned_chunk_placement_displays_the_chunk_and_both_frames() {
  let m = ChunkPlacementMismatch::new(1, 3, 2);
  assert_eq!((m.chunk(), m.aggregated(), m.reconstructed()), (1, 3, 2));
  let rendered = ExtractError::MisalignedChunkPlacement(m).to_string();
  assert!(rendered.contains("chunk 1"), "{rendered}");
  assert!(rendered.contains("output frame 3"), "{rendered}");
  assert!(rendered.contains("frame 2"), "{rendered}");
  assert!(rendered.contains("reconstruction"), "{rendered}");
}

#[test]
fn frame_step_not_representable_in_f32_displays_both_the_step_and_its_image() {
  let w = crate::audio::speaker::window::SlidingWindow::new(0.0, 1.0, 7.0e-46);
  let e = InvalidSlidingWindow::new(ExtractionPart::FramesSw, w);
  let rendered = ExtractError::FrameStepNotRepresentableInF32(e).to_string();
  assert!(rendered.contains("frames_sw"), "{rendered}");
  // The f64 step and the f32 it narrows to, so the message shows the loss.
  assert!(rendered.contains("7e-46"), "{rendered}");
  assert!(rendered.contains("narrows to 0e0"), "{rendered}");
}

#[test]
fn non_binary_segmentation_displays_the_index_the_value_and_the_slot() {
  use crate::audio::speaker::segment::SEG_NUM_SLOTS;

  // `[c][f][s]` with the slot axis innermost, so the slot falls out of the flat
  // index alone: (chunk 2, frame 5, slot 1) on a 7-frame chunk is
  // ((2 * 7 + 5) * 3) + 1 = 58.
  let flat = ((2 * 7 + 5) * SEG_NUM_SLOTS) + 1;
  assert_eq!(flat, 58);
  let n = NonBinarySegmentation::new(flat, 0.3);
  assert_eq!((n.index(), n.value(), n.slot()), (58, 0.3, 1));
  let rendered = ExtractError::NonBinarySegmentation(n).to_string();
  assert!(rendered.contains("segmentations[58]"), "{rendered}");
  assert!(rendered.contains("is 0.3"), "{rendered}");
  assert!(rendered.contains("slot 1"), "{rendered}");
  assert!(rendered.contains("exactly 0.0 or 1.0"), "{rendered}");

  // Index 0 is slot 0 — the boundary the decode must not shift — and a value
  // ABOVE the unit interval renders as readily as one inside it, because the
  // refusal is an equality and not a range.
  let rendered =
    ExtractError::NonBinarySegmentation(NonBinarySegmentation::new(0, 4.5)).to_string();
  assert!(rendered.contains("segmentations[0]"), "{rendered}");
  assert!(rendered.contains("is 4.5"), "{rendered}");
  assert!(rendered.contains("slot 0"), "{rendered}");

  // A non-finite cell reaches the same variant.
  let rendered =
    ExtractError::NonBinarySegmentation(NonBinarySegmentation::new(2, f64::NAN)).to_string();
  assert!(rendered.contains("segmentations[2]"), "{rendered}");
  assert!(rendered.contains("NaN"), "{rendered}");
  assert!(
    rendered.contains(&format!("slot {}", 2 % SEG_NUM_SLOTS)),
    "{rendered}"
  );
}

#[test]
fn active_slot_without_embedding_displays_chunk_and_slot() {
  let a = ActiveSlotWithoutEmbedding::new(7, 2);
  assert_eq!((a.chunk(), a.slot()), (7, 2));
  let rendered = ExtractError::ActiveSlotWithoutEmbedding(a).to_string();
  assert!(rendered.contains("chunk 7"), "{rendered}");
  assert!(rendered.contains("slot 2"), "{rendered}");
  // The message must name the floor the row failed. It is PLDA's `0.01`, the
  // one BOTH backends require and both in-crate producers drop below — not
  // `normalize_from`'s `1e-12`, which is the online engine's alone and which
  // this variant used to advertise.
  assert!(
    rendered.contains("cannot reach the clustering"),
    "{rendered}"
  );
  assert!(rendered.contains("0.01"), "{rendered}");
  assert!(!rendered.contains("NORM_EPSILON"), "{rendered}");
}

#[test]
fn count_not_segmentation_derived_displays_frame_got_and_expected() {
  let c = CountNotSegmentationDerived::new(41, 4, 1);
  assert_eq!((c.frame(), c.got(), c.expected()), (41, 4, 1));
  let rendered = ExtractError::CountNotSegmentationDerived(c).to_string();
  assert!(rendered.contains("count[41]"), "{rendered}");
  assert!(rendered.contains(" is 4 "), "{rendered}");
  assert!(rendered.contains("derive 1"), "{rendered}");

  // The UNDER-count direction renders through the same variant: the check is an
  // equality, so a count BELOW the derived one is the same defect class.
  let under = CountNotSegmentationDerived::new(0, 0, 1);
  let rendered = ExtractError::CountNotSegmentationDerived(under).to_string();
  assert!(rendered.contains("count[0]"), "{rendered}");
  assert!(rendered.contains(" is 0 "), "{rendered}");
  assert!(rendered.contains("derive 1"), "{rendered}");
}

#[test]
fn output_frame_count_too_large_displays_both_the_derived_count_and_the_cap() {
  let rendered =
    ExtractError::OutputFrameCountTooLarge(crate::audio::speaker::extract::MAX_OUTPUT_FRAMES + 1)
      .to_string();
  assert!(rendered.contains("4194305"), "{rendered}");
  assert!(rendered.contains("4194304"), "{rendered}");
  assert!(rendered.contains("MAX_OUTPUT_FRAMES"), "{rendered}");
}

#[test]
fn non_finite_raw_embedding_displays_the_index_it_decodes_to_chunk_slot_dimension() {
  use crate::audio::speaker::{embed::EMBEDDING_DIM, segment::SEG_NUM_SLOTS};

  // The newtype carries a flat `[c][s][d]` index; the message decodes it. The
  // arithmetic is the inverse of `extract::embedding_range`, so it is pinned
  // against hand-computed positions rather than restated.
  // (chunk 2, slot 1, dimension 5) = ((2 * 3 + 1) * 256) + 5 = 1797.
  let flat = ((2 * SEG_NUM_SLOTS + 1) * EMBEDDING_DIM) + 5;
  assert_eq!(flat, 1797);
  let rendered = ExtractError::NonFiniteRawEmbedding(flat).to_string();
  assert!(rendered.contains("raw_embeddings[1797]"), "{rendered}");
  assert!(rendered.contains("chunk 2"), "{rendered}");
  assert!(rendered.contains("slot 1"), "{rendered}");
  assert!(rendered.contains("dimension 5"), "{rendered}");

  // Index 0 is chunk 0, slot 0, dimension 0 — the boundary the decode must not
  // shift.
  let rendered = ExtractError::NonFiniteRawEmbedding(0).to_string();
  assert!(rendered.contains("raw_embeddings[0]"), "{rendered}");
  assert!(rendered.contains("chunk 0"), "{rendered}");
  assert!(rendered.contains("slot 0"), "{rendered}");
  assert!(rendered.contains("dimension 0"), "{rendered}");

  // The last lane of the last slot of chunk 0: one before the chunk boundary.
  let last = SEG_NUM_SLOTS * EMBEDDING_DIM - 1;
  let rendered = ExtractError::NonFiniteRawEmbedding(last).to_string();
  assert!(rendered.contains("chunk 0"), "{rendered}");
  assert!(
    rendered.contains(&format!("slot {}", SEG_NUM_SLOTS - 1)),
    "{rendered}"
  );
  assert!(
    rendered.contains(&format!("dimension {}", EMBEDDING_DIM - 1)),
    "{rendered}"
  );
}

#[test]
fn extraction_tensor_bytes_too_large_displays_the_total_and_the_cap() {
  use crate::audio::speaker::extract::MAX_EXTRACTION_TENSOR_BYTES;

  let rendered =
    ExtractError::ExtractionTensorBytesTooLarge(MAX_EXTRACTION_TENSOR_BYTES + 1).to_string();
  assert!(
    rendered.contains(&(MAX_EXTRACTION_TENSOR_BYTES + 1).to_string()),
    "{rendered}"
  );
  assert!(
    rendered.contains("MAX_EXTRACTION_TENSOR_BYTES"),
    "{rendered}"
  );
  assert!(
    rendered.contains(&MAX_EXTRACTION_TENSOR_BYTES.to_string()),
    "{rendered}"
  );
  // The message must say which grid it is talking about: a caller who has just
  // read `OutputFrameCountTooLarge`'s "output frames" needs to see that this one
  // is the chunk grid, not the same bound restated.
  assert!(rendered.contains("chunk grid"), "{rendered}");
}

#[test]
fn extraction_chunk_count_too_large_displays_the_count_and_the_cap() {
  use crate::audio::speaker::extract::MAX_EXTRACTION_CHUNKS;

  let rendered = ExtractError::ExtractionChunkCountTooLarge(MAX_EXTRACTION_CHUNKS + 1).to_string();
  assert!(
    rendered.contains(&(MAX_EXTRACTION_CHUNKS + 1).to_string()),
    "{rendered}"
  );
  assert!(rendered.contains("MAX_EXTRACTION_CHUNKS"), "{rendered}");
  assert!(
    rendered.contains(&MAX_EXTRACTION_CHUNKS.to_string()),
    "{rendered}"
  );
  // Same discrimination the byte cap's message needs, in the other direction:
  // a caller reading "chunk grid" must see this one counts CHUNKS, where
  // `ExtractionTensorBytesTooLarge` counts bytes of that same grid.
  assert!(rendered.contains("chunks"), "{rendered}");
  let bytes = ExtractError::ExtractionTensorBytesTooLarge(1).to_string();
  assert_ne!(
    rendered.split_whitespace().collect::<Vec<_>>(),
    bytes.split_whitespace().collect::<Vec<_>>()
  );
}

/// The repo's `rust-type-conventions` rule — "variants are UNIT or NEWTYPE only,
/// never struct-shaped" — counted rather than asserted per variant, so a new
/// struct-shaped variant is a failure here rather than a review miss.
///
/// Ten variants were grandfathered in when this test was written:
/// `ModelError::ContractMismatch`, `InferError`'s five (`NonFiniteOutput`,
/// `InputLength`, `OutputShape`, `NonFiniteInput`, `F16OverflowInput`), and
/// `ExtractError`'s four (`StepSamplesExceedsWindow`, `OnsetOutOfRange`,
/// `UnsupportedStepSamples`, `FrameCountMismatch`). All ten now carry a named
/// payload struct or a plain scalar, so the exemption is gone and the rule
/// holds here with no exceptions.
#[test]
fn error_enums_have_no_struct_shaped_variants() {
  let src = include_str!("mod.rs");
  let count = src
    .lines()
    .filter(|l| {
      let t = l.strip_prefix("  ").unwrap_or("");
      t.ends_with(" {")
        && t.chars().next().is_some_and(char::is_uppercase)
        && t[..t.len() - 2].chars().all(char::is_alphanumeric)
    })
    .count();
  assert_eq!(
    count, 0,
    "struct-shaped variants in audio/speaker/error/mod.rs"
  );
}
