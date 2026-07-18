use super::*;

// ---------------------------------------------------------------------
// truncated_frame_count: hermetic coverage of the truncation/clamp math.
// The comments on each case below call out which mutation of
// `truncated_frame_count` (module doc's "Truncation formula" section)
// the test would catch, per the task's mutation-evidence requirement.
// ---------------------------------------------------------------------

#[test]
fn truncated_frame_count_zero_samples_is_zero() {
  // No real audio → no real frames. The `real_samples == 0` short-circuit is
  // load-bearing: the conv formula would otherwise floor UP to 1
  // (`0.max(400)` → 400 → one frame). A trivial chunk's empty tensor must stay
  // empty.
  assert_eq!(truncated_frame_count(0, 2999), 0);
}

#[test]
fn truncated_frame_count_sub_receptive_field_is_one_frame() {
  // Anything from a single sample up to the full 400-sample receptive field
  // yields exactly ONE frame: the wav2vec2 conv stack needs a complete
  // receptive field for its first output, and asry pads a sub-400 chunk up to
  // 400 before it — so one real sample and a full receptive field are the same
  // one frame. (Reverting `saturating_sub` to a bare `-` underflows here.)
  //
  // HAND-COMPUTED literal inputs, deliberately NOT `HOP_SAMPLES` /
  // `RECEPTIVE_FIELD_SAMPLES`: a boundary derived from the constant it means to
  // pin moves WITH the constant under mutation and stays green (F3). 320 and 400
  // are spelled out; `receptive_field_and_hop_constants_are_pinned` pins the
  // constants themselves.
  for real_samples in [1, 200, 320, 399, 400] {
    assert_eq!(
      truncated_frame_count(real_samples, 2999),
      1,
      "real_samples={real_samples} is within the receptive field: one frame"
    );
  }
}

#[test]
fn truncated_frame_count_no_phantom_frame_from_receptive_field_slack() {
  // THE wrong-pinning fix. The old formula was `ceil(real_samples / 320)`,
  // which invented a phantom frame out of the receptive-field slack:
  // `ceil(321/320) == 2` and `ceil(641/320) == 3`. But 321 (and 641) real
  // samples do not fill a SECOND 400-wide receptive field, so the wav2vec2
  // conv stack — and asry's own ONNX encoder — yield exactly ONE frame. Those
  // phantom frames are pure padding-derived structure: a 641-sample chunk
  // carrying three distinct tokens rode them into a plausible-but-nonexistent
  // alignment where the reference returns `NoAlignmentPath`
  // (`tests/prepared_composition.rs`, `tests/align_chunk.rs`). Reverting to
  // `div_ceil` fails both assertions (2 and 3, not 1).
  assert_eq!(truncated_frame_count(321, 2999), 1); // 321: was ceil → 2
  assert_eq!(truncated_frame_count(641, 2999), 1); // was ceil → 3
}

#[test]
fn truncated_frame_count_adds_one_frame_per_hop_past_the_receptive_field() {
  // At and above the receptive field the count is `floor((L - 400)/320) + 1`:
  // each further 320-sample hop past the first full receptive field adds one
  // frame. Catches dropping the `+ 1` (401 → 0) or a wrong divisor.
  //
  // HAND-COMPUTED literals (NOT derived from the constants — see F3): 401 → 1,
  // 720 → 2, 1040 → 3.
  assert_eq!(truncated_frame_count(401, 2999), 1);
  assert_eq!(truncated_frame_count(720, 2999), 2);
  assert_eq!(truncated_frame_count(1040, 2999), 3);
}

#[test]
fn truncated_frame_count_receptive_field_boundary_is_pinned_by_hand() {
  // THE F3 pin. Literal, hand-computed frame counts across the first-frame
  // boundary, referencing NEITHER RECEPTIVE_FIELD_SAMPLES nor HOP_SAMPLES — so
  // mutating either constant (e.g. RECEPTIVE_FIELD_SAMPLES 400 → 399) cannot
  // slide the inputs and expectations to stay green, the exact defect this
  // replaces (a self-derived fixture became 719 → 2 and still passed).
  //
  // With RF = 400, HOP = 320 the count holds at 1 up to and including 719 real
  // samples (719 does not fill a SECOND 400-wide window past the first hop) and
  // steps to 2 at 720. Under the RF = 399 mutation the step falls to 719, so
  // `719 → 1` is the assertion that catches it (it would return 2); `720 → 2`
  // pins the true step.
  assert_eq!(truncated_frame_count(399, 2999), 1);
  assert_eq!(truncated_frame_count(400, 2999), 1);
  assert_eq!(truncated_frame_count(719, 2999), 1);
  assert_eq!(truncated_frame_count(720, 2999), 2);
}

#[test]
fn receptive_field_and_hop_constants_are_pinned() {
  // Direct literal pins on the geometry constants themselves, so a change to
  // either is a loud, single-line failure and not a silent re-derivation of the
  // frame-count fixtures. This is wav2vec2-base's fixed geometry: a 400-sample
  // receptive field and a 320-sample (20 ms @ 16 kHz) hop.
  assert_eq!(RECEPTIVE_FIELD_SAMPLES, 400);
  assert_eq!(HOP_SAMPLES, 320);
}

#[test]
fn truncated_frame_count_reference_short_clip() {
  // 48,000 samples (3 s), well under the model's 2,999-frame ceiling: the
  // reference conv output is `floor((48_000 - 400)/320) + 1 == 149`. The old
  // `ceil(48_000/320) == 150` over-counted by one, so reverting to `div_ceil`
  // fails here (150, not 149). Cross-validated against the LIVE model by
  // `emissions_on_short_input_truncates_to_hermetic_formula`.
  assert_eq!(truncated_frame_count(48_000, 2_999), 149);
}

#[test]
fn truncated_frame_count_full_window_is_the_model_frame_count() {
  // A full, zero-padding-free ENCODER_WINDOW_SAMPLES (960,000 — exactly the
  // `ted_60.wav` fixture's own case) evaluates to `floor((960_000 - 400)/320)
  // + 1 == 2_999`, `base960h_aligner.mlmodelc`'s ACTUAL frame count
  // (`tests/model_io.rs::base960h_aligner_io_matches_spec`): the model count
  // falls out of the formula NATURALLY, with the `.min(available_frames)` clamp
  // a no-op here. The old `ceil(960_000/320) == 3_000` overshot by one and
  // relied on the clamp to hide the phantom frame; this formula does not.
  assert_eq!(truncated_frame_count(ENCODER_WINDOW_SAMPLES, 2_999), 2_999);
}

#[test]
fn truncated_frame_count_approaches_the_full_window_without_overshoot() {
  // The formula climbs to 2,999 and stops there — it never exceeds the model
  // count for any in-window input, so the clamp is defensive, not corrective.
  // `2_999 * 320 == 959_680` is now 2_998 (not the old ceil's 2_999); the count
  // first reaches 2_999 at 959_760 and holds it through the full window.
  assert_eq!(truncated_frame_count(2_999 * HOP_SAMPLES, 2_999), 2_998);
  assert_eq!(truncated_frame_count(959_760, 2_999), 2_999);
  assert_eq!(
    truncated_frame_count(ENCODER_WINDOW_SAMPLES - 1, 2_999),
    2_999
  );
}

#[test]
fn truncated_frame_count_clamp_engages_only_below_the_formula() {
  // The `.min(available_frames)` clamp never fires for the real model (the
  // formula tops out at exactly its 2,999), so prove it against a SMALLER
  // hypothetical frame budget: 48,000 samples nominally yield 149, but a
  // 100-frame model must cap at 100. Catches a `.min` → `.max` mutant, which
  // would return 149 here.
  assert_eq!(truncated_frame_count(48_000, 100), 100);
  assert_eq!(truncated_frame_count(48_000, 149), 149); // exactly at the budget: no clamp
}

#[test]
fn truncated_frame_count_never_exceeds_available_frames_near_full_window() {
  // Sweep below/at/above the full window, cross-checking the invariant
  // `result <= available_frames` the clamp exists to guarantee. Below and at
  // the window the formula already tops out at exactly 2,999, so those cases
  // alone are a VACUOUS `<=` check — the `.min` never fires, and deleting it
  // (or flipping it to `.max`) still passes them. The ABOVE-window cases make
  // the clamp load-bearing: `ENCODER_WINDOW_SAMPLES + HOP_SAMPLES` evaluates to
  // 3,000 pre-clamp and `+ 10 * HOP_SAMPLES` to 3,009, so only
  // `.min(available_frames)` pulls each back to the model's 2,999.
  let available_frames = 2_999;
  for real_samples in [
    ENCODER_WINDOW_SAMPLES - 1,
    ENCODER_WINDOW_SAMPLES,
    available_frames * HOP_SAMPLES,
    available_frames * HOP_SAMPLES + 1,
    ENCODER_WINDOW_SAMPLES + HOP_SAMPLES,
    ENCODER_WINDOW_SAMPLES + 10 * HOP_SAMPLES,
  ] {
    let t = truncated_frame_count(real_samples, available_frames);
    assert!(
      t <= available_frames,
      "truncated_frame_count({real_samples}, {available_frames}) = {t} exceeds available_frames"
    );
  }
  // Load-bearing clamp: above the window the conv formula overshoots 2,999, and
  // ONLY the `.min` holds the result at exactly `base960h_aligner.mlmodelc`'s
  // frame count. These EXACT `== 2_999` checks fail the instant `.min` is
  // deleted or flipped to `.max` — the formula then returns 3,000 and 3,009.
  assert_eq!(
    truncated_frame_count(ENCODER_WINDOW_SAMPLES + HOP_SAMPLES, available_frames),
    2_999
  );
  assert_eq!(
    truncated_frame_count(ENCODER_WINDOW_SAMPLES + 10 * HOP_SAMPLES, available_frames),
    2_999
  );
}

// ---------------------------------------------------------------------
// EncoderInput: the F1 capability. Hermetic, and that is the whole point —
// a wrong real-sample length is unrepresentable at CONSTRUCTION, before any
// Encoder or model exists, so the mismatch the free `real_samples: usize`
// argument used to allow cannot reach a prediction.
// ---------------------------------------------------------------------

#[test]
fn encoder_input_from_samples_binds_real_length_to_the_slice() {
  // A 176,000-sample chunk fed as raw audio: `real_samples` IS the slice's own
  // length, 176,000. The F1 defect declared 175,360 (two hops short) for this
  // same buffer to get 547 frames where 549 belong; there is now no
  // `real_samples` argument to declare it into.
  let chunk = vec![0.0f32; 176_000];
  let input = EncoderInput::from_samples(&chunk).expect("176k <= window");
  assert_eq!(input.real_samples, 176_000);
  assert_eq!(input.encoder_input.len(), 176_000);
  assert_eq!(truncated_frame_count(input.real_samples, 2_999), 549);
  // The buggy answer is now unreachable: 175_360 gives 547, but nothing can
  // bind 175_360 to this 176,000-sample buffer.
  assert_eq!(truncated_frame_count(175_360, 2_999), 547);
  assert_ne!(
    truncated_frame_count(input.real_samples, 2_999),
    truncated_frame_count(175_360, 2_999)
  );
}

#[test]
fn encoder_input_gate_binds_real_length_independent_of_the_padded_buffer() {
  // The pipeline geometry: 200 real samples that asry silence-masks and zero-pads
  // to the 400-sample receptive field. The gate every constructor funnels through
  // records the real length as the UNPADDED count (200), never the padded buffer's
  // length (400) — the type-level F1 property. `from_prepared` reads exactly this
  // (buffer, real_samples) pair off an unforgeable `PreparedChunk`; here we drive
  // the gate directly so the binding is pinned with no model and no seam. The
  // `from_prepared` door's OWN provenance is pinned hermetically by the sibling
  // `from_prepared_records_the_true_pre_pad_provenance_not_the_padded_length`; the
  // end-to-end door on the CoreML encoder is `tests/prepared_composition.rs`.
  let real_len = 200usize;
  let padded_buffer = vec![0.0f32; 400];
  let input = EncoderInput::new(&padded_buffer, real_len).expect("valid geometry");
  assert_eq!(input.real_samples, 200); // the UNPADDED count, NOT 400
  assert_eq!(input.encoder_input.len(), 400);
  // Under the corrected conv-geometry truncation this sub-receptive-field slip is
  // BENIGN for the count: 200 real samples and the 400-sample pad both yield the
  // single receptive-field frame — `ceil` was the only thing that ever made them
  // 1 vs 2. The binding still matters (it records the honest length and stays
  // correct for the general case pinned by
  // `encoder_input_from_samples_binds_real_length_to_the_slice`: 176_000 vs
  // 175_360 → 549 vs 547, where a short real count genuinely moves the count).
  assert_eq!(truncated_frame_count(input.real_samples, 2_999), 1);
  assert_eq!(truncated_frame_count(padded_buffer.len(), 2_999), 1);
}

#[test]
fn from_prepared_records_the_true_pre_pad_provenance_not_the_padded_length() {
  // L2: the public `from_prepared` door must record the chunk's TRUE pre-pad
  // `real_samples` (200), never the padded encoder-buffer length (400). The
  // corrected conv geometry now maps BOTH 200 and 400 to a single frame, so the
  // frame COUNT can no longer distinguish the two doors below the receptive field
  // — which is exactly why `tests/prepared_composition.rs`'s `frames() == 1`
  // checks went vacuous for this mutation. The surviving distinguisher is this
  // recorded provenance.
  //
  // `real_samples` has no public accessor BY DESIGN (it must never be a
  // caller-supplied integer — see `EncoderInput`), and no crate dev-dependency
  // captures the `tracing` span field that carries it. So this pins the guarantee
  // the adjudicated way: a crate-private assertion, reading the field one module
  // in, driven through the SAME public `from_prepared` door on a real
  // `PreparedChunk` — no CoreML model, since only construction is under test.
  //
  // Mutating `from_prepared` to `Self::from_samples(prepared.encoder_input())`
  // records the padded 400 here and turns the `== 200` assertion RED — the exact
  // regression the count-based test can no longer catch.
  use asry::emissions::EmissionsAligner;
  use core::sync::atomic::AtomicBool;

  let aligner = EmissionsAligner::builder(
    crate::audio::align::Lang::En,
    crate::audio::align::vocab::tokenizer_json_bytes(),
  )
  .normalizer(Box::new(crate::audio::align::EnglishNormalizer::new()))
  .blank_token_id(crate::audio::align::vocab::BLANK_ID)
  .build()
  .expect("build the En seam from the bundled tokenizer");

  // 200 real samples of unambiguously non-silent audio. The content is irrelevant
  // to the recorded LENGTH (no encoder runs here), but the text must tokenize to
  // alignable tokens or `prepare` returns a trivial chunk with no buffer to test.
  let samples: Vec<f32> = (0..200).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
  let abort = AtomicBool::new(false);
  let prepared = aligner
    .prepare(
      &samples,
      &crate::audio::align::SpeechSpans::all_speech(),
      "test",
      &[],
      &abort,
    )
    .expect("prepare 200 real samples with alignable text");
  assert!(
    !prepared.is_trivial(),
    "`test` must tokenize to alignable tokens, or there is no prepared buffer to test"
  );
  assert_eq!(
    prepared.encoder_input().len(),
    400,
    "asry pads 200 real samples up to the 400-sample receptive field"
  );

  // The supported door records asry's honest pre-pad length.
  let via_prepared =
    EncoderInput::from_prepared(&prepared).expect("from_prepared geometry is valid");
  assert_eq!(
    via_prepared.real_samples, 200,
    "from_prepared must record the true pre-pad real_samples (200), never the padded 400"
  );

  // The raw door, handed the SAME padded buffer, records the padded length — the
  // provenance the frame-count coincidence (both truncate to one frame) hides, and
  // exactly what the vacuous mutation collapses `from_prepared` into.
  let via_raw =
    EncoderInput::from_samples(prepared.encoder_input()).expect("from_samples geometry is valid");
  assert_eq!(
    via_raw.real_samples, 400,
    "from_samples records the buffer length it is handed (400) — the distinguisher"
  );
  assert_ne!(
    via_prepared.real_samples, via_raw.real_samples,
    "the two doors record DIFFERENT provenance for the one buffer; only the frame \
     count coincides, which is why a count-only test cannot bind from_prepared"
  );
}

#[test]
fn encoder_input_rejects_a_buffer_longer_than_the_window_before_any_prediction() {
  // Invalid geometry is caught at construction, with no Encoder and no model in
  // sight — so it can never reach a prediction. (Formerly this check lived
  // inside `emissions_raw`, one predict away.)
  let too_long = vec![0.0f32; ENCODER_WINDOW_SAMPLES + 1];
  let err = EncoderInput::from_samples(&too_long).unwrap_err();
  assert!(matches!(
    err,
    AlignError::InputTooLong { got, max }
      if got == ENCODER_WINDOW_SAMPLES + 1 && max == ENCODER_WINDOW_SAMPLES
  ));
}

#[test]
fn encoder_input_accepts_a_buffer_exactly_the_window() {
  // The exact-window boundary is valid — it is the `ted_60.wav` case, where
  // `emissions_raw` borrows the buffer rather than padding it.
  let full = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
  let input = EncoderInput::from_samples(&full).expect("exactly the window is fine");
  assert_eq!(input.real_samples, ENCODER_WINDOW_SAMPLES);
  assert_eq!(input.encoder_input.len(), ENCODER_WINDOW_SAMPLES);
}

// ---------------------------------------------------------------------
// check_waveform_contract / check_emissions_contract: hermetic coverage
// without a loaded model (see their doc comments for why this crate
// tests the validation logic directly rather than model-gating against a
// second, deliberately-wrong local model fixture — `Models/alignkit/`
// holds exactly one model).
// ---------------------------------------------------------------------

#[test]
fn check_waveform_contract_accepts_correct_shape_and_dtype() {
  assert_eq!(
    check_waveform_contract(&[1, ENCODER_WINDOW_SAMPLES], Some(DataType::F32)),
    Ok(())
  );
}

#[test]
fn check_waveform_contract_rejects_wrong_shape() {
  let err = check_waveform_contract(&[1, 480_000], Some(DataType::F32)).unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "waveform",
      ..
    }
  ));
}

#[test]
fn check_waveform_contract_rejects_wrong_dtype() {
  let err = check_waveform_contract(&[1, ENCODER_WINDOW_SAMPLES], Some(DataType::F16)).unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "waveform",
      ..
    }
  ));
}

#[test]
fn check_waveform_contract_rejects_missing_dtype() {
  let err = check_waveform_contract(&[1, ENCODER_WINDOW_SAMPLES], None).unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn missing_waveform_input_diagnostic_names_the_exact_contract() {
  // The MISSING-`waveform`-input branch of `Encoder::from_file_with`
  // (`waveform_input_or_mismatch`) must name the SAME `[1, 960000]` contract the
  // shape check reports. The two copies are identical today — unlike the
  // `emissions` side, this diagnostic never drifted — but a second hand-written
  // literal is the same root cause: change `ENCODER_WINDOW_SAMPLES` in one place
  // and the other copy would report a window the next load rejects.
  // `check_waveform_contract` is only reached with a present input, so the tests
  // above cannot cover this separate branch; `None` drives it hermetically (no
  // loaded model — the one artifact in `Models/alignkit/` always has the input).
  // Hand-diverging `expected_waveform_contract` from the check's literal fails
  // the `expected` assertion below.
  match waveform_input_or_mismatch(None) {
    Err(AlignerError::ContractMismatch {
      feature,
      expected,
      actual,
    }) => {
      assert_eq!(feature, "waveform");
      assert_eq!(expected, "[1, 960000] float32");
      assert_eq!(actual, "missing");
    }
    other => panic!("expected a ContractMismatch, got {other:?}"),
  }
}

#[test]
fn check_emissions_contract_accepts_correct_shape_and_returns_frame_count() {
  assert_eq!(
    check_emissions_contract(
      &[1, 2_999, crate::audio::align::vocab::VOCAB_SIZE],
      Some(DataType::F32)
    ),
    Ok(2_999)
  );
}

#[test]
fn check_emissions_contract_rejects_wrong_rank() {
  let err = check_emissions_contract(
    &[2_999, crate::audio::align::vocab::VOCAB_SIZE],
    Some(DataType::F32),
  )
  .unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "emissions",
      ..
    }
  ));
}

#[test]
fn check_emissions_contract_rejects_wrong_batch_dim() {
  let err = check_emissions_contract(
    &[2, 2_999, crate::audio::align::vocab::VOCAB_SIZE],
    Some(DataType::F32),
  )
  .unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_zero_frames() {
  // A zero-frame model would "load fine" and make every `emissions()`
  // call silently return an empty result — reject at construction.
  let err = check_emissions_contract(
    &[1, 0, crate::audio::align::vocab::VOCAB_SIZE],
    Some(DataType::F32),
  )
  .unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_wrong_vocab_dim() {
  let err = check_emissions_contract(&[1, 2_999, 32], Some(DataType::F32)).unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_wrong_dtype() {
  let err = check_emissions_contract(
    &[1, 2_999, crate::audio::align::vocab::VOCAB_SIZE],
    Some(DataType::F64),
  )
  .unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_a_cropped_frame_count() {
  // 2998 — the fence's failing history. `floor((960_000 - 400)/320) + 1 == 2999`
  // is the ONLY frame count this fixed-window graph declares; a cropped
  // `[1, 2998, 29]` export used to pass the old `shape[1] >= 1` check, construct
  // fine, then silently drop the last acoustic frame (the full-window formula
  // requests 2999 but the introspected 2998 clamps it away). It is now a
  // ContractMismatch at construction. Reverting the check to `>= 1` accepts it
  // and fails this test.
  let err = check_emissions_contract(
    &[1, 2_998, crate::audio::align::vocab::VOCAB_SIZE],
    Some(DataType::F32),
  )
  .unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "emissions",
      ..
    }
  ));
}

#[test]
fn check_emissions_contract_rejects_an_overlong_frame_count() {
  // 3000 — one frame too many. The contract is EXACTLY EXPECTED_OUTPUT_FRAMES,
  // so an over-long declaration is a ContractMismatch at construction just like
  // the cropped one, in the other direction. Reverting the check to `>= 1`
  // accepts it and fails this test.
  let err = check_emissions_contract(
    &[1, 3_000, crate::audio::align::vocab::VOCAB_SIZE],
    Some(DataType::F32),
  )
  .unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "emissions",
      ..
    }
  ));
}

#[test]
fn missing_emissions_output_diagnostic_names_the_exact_contract() {
  // The MISSING-`emissions`-output branch of `Encoder::from_file_with`
  // (`emissions_output_or_mismatch`) must name the SAME `[1, 2999, 29]` contract
  // the shape check reports — not the stale `[1, >=1, 29]` this diagnostic once
  // hand-duplicated, which would tell a developer a `[1, 3000, 29]` export is
  // acceptable, only for the next load to reject it. `check_emissions_contract`
  // is only reached with a present output, so the tests above cannot cover this
  // separate branch; `None` drives it hermetically (no loaded model — the one
  // artifact in `Models/alignkit/` always has the output). Reverting
  // `expected_emissions_contract` to a `>=1` literal fails the `expected`
  // assertion below.
  match emissions_output_or_mismatch(None) {
    Err(AlignerError::ContractMismatch {
      feature,
      expected,
      actual,
    }) => {
      assert_eq!(feature, "emissions");
      assert_eq!(expected, "[1, 2999, 29] float32");
      assert_eq!(actual, "missing");
    }
    other => panic!("expected a ContractMismatch, got {other:?}"),
  }
}

// ---------------------------------------------------------------------
// check_log_prob_floor: hermetic coverage of the fp16 `log(0)` sentinel
// guard. The model-gated half (`emissions_reject_an_ane_corrupted_matrix`)
// proves the real ANE artifact trips it; these prove the predicate itself,
// including the two boundaries a mutant would move.
// ---------------------------------------------------------------------

#[test]
fn check_log_prob_floor_accepts_real_log_probs() {
  // The measured legitimate range on this model: max exactly 0.0, min -30.81
  // (`CpuOnly`) / -30.02 (`CpuAndGpu`). Nothing here is anywhere near the floor.
  let data = [0.0, -0.06, -19.0, -21.75, -30.02, -30.81];
  assert!(check_log_prob_floor(&data, ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_log_prob_floor_accepts_an_empty_matrix() {
  // `real_samples == 0` truncates to zero frames; the guard must not invent a
  // failure out of an empty scan (min would be +inf).
  assert!(check_log_prob_floor(&[], ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_log_prob_floor_rejects_the_fp16_log_zero_sentinel() {
  // One corrupt cell in an otherwise clean matrix is still a corrupt matrix:
  // the ANE run corrupts 16.7% of cells, but a single one is enough to move a
  // trellis path. Catches a mutant that thresholds on a FRACTION of cells.
  let data = [0.0, -1.5, -45_440.0, -20.0];
  let Err(err) = check_log_prob_floor(&data, ComputeUnits::All) else {
    panic!("the -45440 fp16 log(0) sentinel must be rejected");
  };
  let AlignError::CorruptEmissions {
    compute,
    min,
    cells,
    total,
  } = err
  else {
    panic!("expected AlignError::CorruptEmissions, got {err:?}");
  };
  assert_eq!(compute, ComputeUnits::All);
  assert_eq!(min, -45_440.0);
  assert_eq!(cells, 1);
  assert_eq!(total, 4);
}

#[test]
fn check_log_prob_floor_is_a_strict_lower_bound_at_the_floor_itself() {
  // The floor is INCLUSIVE (`< LOG_PROB_FLOOR` fails, `== LOG_PROB_FLOOR`
  // passes). Pins the comparison's direction and strictness together: a mutant
  // flipping `<` to `<=` fails the first assertion, one flipping it to `>`
  // fails the second.
  assert!(check_log_prob_floor(&[LOG_PROB_FLOOR], ComputeUnits::CpuOnly).is_ok());
  assert!(
    check_log_prob_floor(&[LOG_PROB_FLOOR - 1.0], ComputeUnits::CpuOnly).is_err(),
    "one ulp-plus below the floor is already outside the log-prob domain"
  );
}

#[test]
fn check_log_prob_floor_leaves_non_finite_values_to_from_log_probs() {
  // Deliberate division of labour, documented on `check_log_prob_floor`: the
  // floor guard is the LOWER bound only. `NaN` compares false against
  // everything and passes here; `Emissions::from_log_probs`' finite ∧ <= 0 scan
  // (which runs on the very next line of `Encoder::emissions`) is what rejects
  // it. Neither scan is redundant with the other, and this pins that seam so a
  // later "simplification" cannot silently drop one of them.
  assert!(check_log_prob_floor(&[f32::NAN], ComputeUnits::CpuOnly).is_ok());
  assert!(check_log_prob_floor(&[f32::INFINITY], ComputeUnits::CpuOnly).is_ok());
  // -inf is genuinely below the floor and IS the guard's business.
  assert!(check_log_prob_floor(&[f32::NEG_INFINITY], ComputeUnits::CpuOnly).is_err());
}

// LOG_PROB_FLOOR's separation property (strictly between the -30.81 legitimate
// minimum and the -45440 sentinel) is asserted in `mod.rs` at COMPILE time, not
// here: both operands are constants, so a runtime test of it is dead weight that
// only fires after a build already succeeded.

// ---------------------------------------------------------------------
// check_log_prob_normalization: hermetic coverage of the per-frame logsumexp
// guard — the check that makes the "these really are log-probs" contract true
// for a model-artifact swap the floor and `from_log_probs`'s finite ∧ <= 0 scan
// both miss. The model-gated half
// (`emissions_pass_the_normalization_guard_on_real_speech`) proves the real
// artifact passes on both clips and both clean placements; these prove the
// predicate rejects the two un-normalized inputs the finding names, AND that
// neither the floor nor the <= 0 scan would have caught them (the closed bypass).
// ---------------------------------------------------------------------

/// A frame filled with `value` on every one of the 29 classes.
fn uniform_frame(value: f32) -> [f32; crate::audio::align::vocab::VOCAB_SIZE] {
  [value; crate::audio::align::vocab::VOCAB_SIZE]
}

#[test]
fn check_log_prob_normalization_accepts_normalized_log_probs() {
  // A normalized log-prob frame has logsumexp == 0 by construction. Two shapes:
  // (1) uniform — 29 copies of ln(1/29) = -ln(29), the maximum-entropy
  // distribution; (2) a peaked distribution built as a genuine log-softmax, so
  // its probabilities sum to 1 and its logsumexp is 0 for a NON-uniform row too.
  let ln29 = f64::from(crate::audio::align::vocab::VOCAB_SIZE as u32).ln();
  let uniform = uniform_frame(-ln29 as f32);

  // log_softmax of arbitrary logits: row_j = z_j - logsumexp(z), which sums to 1
  // in probability space, so logsumexp(row) == 0.
  let logits: [f32; crate::audio::align::vocab::VOCAB_SIZE] =
    core::array::from_fn(|j| (j as f32) * 0.5 - 3.0);
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let z_lse = f64::from(max)
    + logits
      .iter()
      .map(|&z| (f64::from(z) - f64::from(max)).exp())
      .sum::<f64>()
      .ln();
  let peaked: Vec<f32> = logits
    .iter()
    .map(|&z| (f64::from(z) - z_lse) as f32)
    .collect();

  let mut data = Vec::new();
  data.extend_from_slice(&uniform);
  data.extend_from_slice(&peaked);
  assert!(
    check_log_prob_normalization(&data, ComputeUnits::CpuOnly).is_ok(),
    "normalized log-prob frames (logsumexp ≈ 0) must pass"
  );
}

#[test]
fn check_log_prob_normalization_accepts_an_empty_matrix() {
  // `real_samples == 0` truncates to zero frames; no frame to check, so Ok
  // (mirrors `check_log_prob_floor_accepts_an_empty_matrix`).
  assert!(check_log_prob_normalization(&[], ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_log_prob_normalization_rejects_shifted_raw_logits() {
  // THE bypass this guard closes. A full 2999 × 29 matrix of raw logits shifted
  // WHOLLY into [-20, -10] — the finding's exact fence. Every cell is finite and
  // <= 0, so it passes BOTH the floor (nothing below -100) and the finite ∧ <= 0
  // scan `from_log_probs` runs — yet no frame is a distribution: a row entirely
  // in [-20, -10] has logsumexp in [max, max + ln 29] ⊆ [-20, -6.63], so
  // |logsumexp| >= 6.63, orders of magnitude past the ±2e-2 tolerance.
  let mut data = Vec::with_capacity(2999 * crate::audio::align::vocab::VOCAB_SIZE);
  for _ in 0..2999 {
    // A ramp across the vocab, every value inside [-20, -10]; not normalized.
    for j in 0..crate::audio::align::vocab::VOCAB_SIZE {
      data
        .push(-10.0 - (j as f32) * (10.0 / (crate::audio::align::vocab::VOCAB_SIZE as f32 - 1.0)));
    }
  }
  // The floor does NOT catch it (nothing below -100)...
  assert!(
    check_log_prob_floor(&data, ComputeUnits::CpuOnly).is_ok(),
    "shifted raw logits in [-20, -10] are all above LOG_PROB_FLOOR — the floor cannot catch them"
  );
  // ...and `from_log_probs`'s finite ∧ <= 0 scan would not either.
  assert!(
    data.iter().all(|v| v.is_finite() && *v <= 0.0),
    "shifted raw logits are finite and <= 0 — the from_log_probs scan cannot catch them"
  );
  // Only the normalization guard does.
  let Err(err) = check_log_prob_normalization(&data, ComputeUnits::CpuOnly) else {
    panic!("raw logits shifted into [-20, -10] must be rejected as un-normalized");
  };
  let AlignError::UnnormalizedEmissions {
    logsumexp,
    tolerance,
    ..
  } = err
  else {
    panic!("expected AlignError::UnnormalizedEmissions, got {err:?}");
  };
  assert!(
    logsumexp.abs() > 6.6,
    "a [-20, -10] shifted frame's |logsumexp| is >= 6.63, got {logsumexp}"
  );
  assert_eq!(tolerance, LOG_PROB_SUM_TOLERANCE);
}

#[test]
fn check_log_prob_normalization_rejects_an_all_zero_frame() {
  // THE simplest un-normalized case: an all-zeros frame, exp(0) = 1 on every
  // class, so logsumexp = ln(29) ≈ 3.367 — again finite, <= 0, above the floor,
  // and again only the normalization guard rejects it.
  let data = uniform_frame(0.0);
  assert!(
    check_log_prob_floor(&data, ComputeUnits::CpuOnly).is_ok(),
    "an all-zeros frame is above the floor"
  );
  assert!(data.iter().all(|v| v.is_finite() && *v <= 0.0));
  let Err(AlignError::UnnormalizedEmissions {
    row,
    logsumexp,
    compute,
    ..
  }) = check_log_prob_normalization(&data, ComputeUnits::All)
  else {
    panic!("an all-zeros frame (logsumexp = ln 29) must be rejected");
  };
  assert_eq!(row, 0);
  assert_eq!(compute, ComputeUnits::All); // the placement is carried through
  let ln29 = f64::from(crate::audio::align::vocab::VOCAB_SIZE as u32).ln();
  assert!(
    (logsumexp - ln29).abs() < 1e-5,
    "all-zeros logsumexp must be ln(29) ≈ {ln29}, got {logsumexp}"
  );
}

#[test]
fn check_log_prob_normalization_names_the_worst_frame() {
  // Several normalized frames (logsumexp ≈ 0) with ONE un-normalized frame at a
  // known index: the error must name THAT frame, not the first or the last.
  let ln29 = f64::from(crate::audio::align::vocab::VOCAB_SIZE as u32).ln();
  let normalized = uniform_frame(-ln29 as f32);
  let bad_index = 2usize;
  let mut data = Vec::new();
  for i in 0..5 {
    if i == bad_index {
      data.extend_from_slice(&uniform_frame(0.0)); // logsumexp = ln 29
    } else {
      data.extend_from_slice(&normalized); // logsumexp ≈ 0
    }
  }
  let Err(AlignError::UnnormalizedEmissions { row, .. }) =
    check_log_prob_normalization(&data, ComputeUnits::CpuOnly)
  else {
    panic!("the un-normalized frame must be rejected");
  };
  assert_eq!(row, bad_index, "the error must name the worst frame");
}

#[test]
fn check_log_prob_normalization_thresholds_on_the_tolerance() {
  // Pins the threshold LOCATION and direction: a uniform frame constructed to
  // sit at logsumexp = TOL/2 passes, one at logsumexp = 2·TOL is rejected. (An
  // exact-at-TOL boundary is not pinned here — an f32-stored frame cannot hit an
  // f64 TOL exactly; the `>` strictness is stated on the function.)
  let ln29 = f64::from(crate::audio::align::vocab::VOCAB_SIZE as u32).ln();
  let tol = LOG_PROB_SUM_TOLERANCE;
  // uniform frame value `v` gives logsumexp = v + ln29; solve for the target.
  let inside = uniform_frame((tol / 2.0 - ln29) as f32);
  let outside = uniform_frame((2.0 * tol - ln29) as f32);
  assert!(
    check_log_prob_normalization(&inside, ComputeUnits::CpuOnly).is_ok(),
    "logsumexp = TOL/2 is within tolerance"
  );
  assert!(
    check_log_prob_normalization(&outside, ComputeUnits::CpuOnly).is_err(),
    "logsumexp = 2·TOL exceeds tolerance"
  );
}

// ---------------------------------------------------------------------
// RawEmissions::check_value_domain: the floor-then-normalization guard sequence
// `Encoder::emissions` actually mints through. Driving the MINTER (not the
// extracted `check_emission_value_domain` helper) binds BOTH predicates to the
// production door in one call — in particular the normalization half, which
// (unlike the floor half, bound end-to-end by the model-gated
// `emissions_reject_an_ane_corrupted_matrix`) has no real-model fixture. If the
// minter's guard were ever handed `&[]` in place of its real tensor, or skipped
// normalization, an un-normalized matrix would sail through unnoticed; because
// this test feeds the minter the same un-normalized tensors the door would AND
// asserts the sealed buffer is the validated one, that regression is red right
// here.
// ---------------------------------------------------------------------

#[test]
fn raw_emissions_check_value_domain_binds_the_guard_and_the_minted_buffer() {
  // A shifted-raw-logit matrix — every cell finite, <= 0, and above
  // LOG_PROB_FLOOR, so the floor and `from_log_probs`'s <= 0 scan both miss it
  // and only the normalization step in the sequence rejects it.
  let mut shifted = Vec::with_capacity(4 * crate::audio::align::vocab::VOCAB_SIZE);
  for _ in 0..4 {
    for j in 0..crate::audio::align::vocab::VOCAB_SIZE {
      shifted
        .push(-10.0 - (j as f32) * (10.0 / (crate::audio::align::vocab::VOCAB_SIZE as f32 - 1.0)));
    }
  }
  let raw = RawEmissions {
    frames: 4,
    data: shifted,
  };
  assert!(
    matches!(
      raw.check_value_domain(ComputeUnits::CpuOnly),
      Err(AlignError::UnnormalizedEmissions { .. })
    ),
    "the minter must reject a shifted-raw-logit tensor as un-normalized"
  );

  // The all-zeros frame: exp(0) = 1 on every class, logsumexp = ln 29, again
  // above the floor and <= 0 — only normalization catches it.
  let raw = RawEmissions {
    frames: 1,
    data: uniform_frame(0.0).to_vec(),
  };
  assert!(
    matches!(
      raw.check_value_domain(ComputeUnits::CpuOnly),
      Err(AlignError::UnnormalizedEmissions { .. })
    ),
    "the minter must reject an all-zeros frame as un-normalized"
  );

  // A genuinely normalized frame (logsumexp = 0) passes — and the minted token
  // owns EXACTLY the bytes the guard validated: the minter cannot clear one
  // buffer and seal another.
  let ln29 = f64::from(crate::audio::align::vocab::VOCAB_SIZE as u32).ln();
  let normalized = uniform_frame(-ln29 as f32).to_vec();
  let token = RawEmissions {
    frames: 1,
    data: normalized.clone(),
  }
  .check_value_domain(ComputeUnits::CpuOnly)
  .expect("the minter must accept a normalized log-prob frame");
  assert_eq!(token.frames, 1);
  assert_eq!(
    token.data, normalized,
    "the minted token must own the exact buffer the guard validated"
  );
}

// ---------------------------------------------------------------------
// ValueDomainChecked::into_emissions: the wrap the minted token feeds. The
// minter test above stops at MINTING — it never consumes its token — so the door
// choice inside `into_emissions` (`from_log_probs`, the log-prob door, vs
// `from_logits`, the raw-logit door) is invisible to it. This test consumes the
// token and pins that door: a frame that clears BOTH value-domain guards yet
// carries a positive cell must be rejected by `from_log_probs`, where
// `from_logits` would silently renormalize and accept.
// ---------------------------------------------------------------------

/// The value-domain guard is deliberately not the WHOLE log-prob contract:
/// [`check_log_prob_floor`] bounds each cell from below and
/// [`check_log_prob_normalization`] checks each frame is a distribution, but
/// neither enforces the per-cell `<= 0` ceiling. That half is
/// [`Emissions::from_log_probs`]'s own `finite ∧ <= 0` scan, run inside
/// [`ValueDomainChecked::into_emissions`] on the very tensor the guard sealed
/// (see [`check_log_prob_floor`]'s "Deliberately only the lower bound" note).
///
/// The distinguisher is a single frame `[0.001, -20.0 × 28]`:
///
/// - It clears [`check_log_prob_floor`]: the minimum cell is `-20.0`, far above
///   [`LOG_PROB_FLOOR`] (`-100`).
/// - It clears [`check_log_prob_normalization`]:
///   `logsumexp = ln(e^0.001 + 28·e^-20) ≈ 0.001` (the 28 `-20.0` cells add
///   `≈ 5.8e-8`), well inside [`LOG_PROB_SUM_TOLERANCE`] (`2e-2`). So both guards
///   pass and the token mints.
/// - But cell 0 is `0.001 > 0`, so it is not a log-probability.
///   [`Emissions::from_log_probs`] rejects it as `LogProbsValueClass::Positive`;
///   `Emissions::from_logits` would instead apply
///   `log_softmax_with_finite_guard`, renormalize it into a plausible
///   distribution, and return `Ok`.
///
/// So swapping the door in [`ValueDomainChecked::into_emissions`]
/// (`from_log_probs` → `from_logits`) turns this `Err` into `Ok` and this test
/// goes red, while every other test stays green: the minter test never consumes
/// its token, and the model-gated `emissions_wraps_into_validated_emissions`
/// feeds a genuine, already-normalized log-prob tensor both doors accept
/// identically. This is the test that pins the door at the wrap.
#[test]
fn into_emissions_takes_the_log_prob_door_not_the_logit_door() {
  use asry::emissions::{EmissionsError, LogProbsValueClass};

  // One frame that clears both value-domain guards yet holds a single positive
  // cell — the `<= 0` half of the log-prob contract the guards defer to
  // `from_log_probs`.
  let mut data = vec![-20.0f32; crate::audio::align::vocab::VOCAB_SIZE];
  data[0] = 0.001;

  // Neither guard rejects it: the floor sees a min of -20.0 (above -100), and
  // the frame's logsumexp is ≈ 0.001 (within 2e-2).
  assert!(
    check_log_prob_floor(&data, ComputeUnits::CpuOnly).is_ok(),
    "min cell -20.0 is far above LOG_PROB_FLOOR (-100): the floor cannot catch a positive cell"
  );
  assert!(
    check_log_prob_normalization(&data, ComputeUnits::CpuOnly).is_ok(),
    "logsumexp ≈ 0.001 is within LOG_PROB_SUM_TOLERANCE (2e-2): normalization cannot catch it"
  );

  // ...so the token mints through the real check sequence.
  let token = RawEmissions { frames: 1, data }
    .check_value_domain(ComputeUnits::CpuOnly)
    .expect("a frame that clears the floor and the normalization guard must mint a token");

  // Only the log-prob door catches the positive cell on consumption. Swapping
  // `from_log_probs` for `from_logits` in `into_emissions` renormalizes it and
  // returns Ok — the exact mutation this asserts red.
  let Err(err) = token.into_emissions() else {
    panic!(
      "into_emissions accepted a frame with a positive cell (0.001): from_log_probs must reject \
       it. Only from_logits — the wrong door — would renormalize and accept."
    );
  };
  let AlignError::Alignment(EmissionsError::Value(value)) = err else {
    panic!("expected AlignError::Alignment(EmissionsError::Value), got {err:?}");
  };
  assert_eq!(
    value.class(),
    LogProbsValueClass::Positive,
    "cell 0 (0.001) is finite and > 0 — the positive log-prob-domain class"
  );
  assert_eq!(value.frame(), 0, "the positive cell is in frame 0");
  assert_eq!(
    value.vocab_index(),
    0,
    "the positive cell is at vocab index 0"
  );
}

// ---------------------------------------------------------------------
// EncoderOptions
// ---------------------------------------------------------------------

#[test]
fn options_new_defaults_to_cpu_only_compute() {
  // Not a perf preference: the ANE placements corrupt this model's emissions.
  // See `DEFAULT_ENCODER_COMPUTE` and
  // `emissions_have_no_fp16_log_zero_sentinel`.
  assert_eq!(EncoderOptions::new().compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(EncoderOptions::new().compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(EncoderOptions::default(), EncoderOptions::new());
}

#[test]
fn options_with_compute_overrides() {
  // A NON-default placement, or this would also pass against a `with_compute`
  // that silently ignored its argument.
  let options = EncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(options.compute(), ComputeUnits::CpuAndGpu);
}

#[test]
fn options_set_compute_in_place() {
  let mut options = EncoderOptions::new();
  options.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(options.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_compute_defaults_to_cpu_only() {
  let options: EncoderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(options.compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_explicit_compute() {
  // Round-trip a non-default placement: deserializing `cpu_only` would now be
  // indistinguishable from the field defaulting.
  let options: EncoderOptions = serde_json::from_str(r#"{"compute":"cpu_and_gpu"}"#).unwrap();
  assert_eq!(options.compute(), ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&options).unwrap();
  assert!(json.contains("cpu_and_gpu"), "round-tripped json: {json}");
}

// ---------------------------------------------------------------------
// Encoder: model-gated (requires a local base960h_aligner.mlmodelc,
// ALIGNKIT_TEST_MODELS or Models/alignkit/, same convention as
// tests/model_io.rs's `common` module and tests/common/mod.rs).
// Duplicated here in miniature because unit tests under `src/` cannot
// import the separate `tests/` integration-test crate (mirrors
// dia-coreml::segment::tests's identical duplication and rationale).
//
// These load the encoder on DEFAULT_ENCODER_COMPUTE — never a hardcoded
// placement — so they validate the SHIPPING configuration for free. A gate
// pinned to a compute unit proves only that compute unit; pinning CpuOnly
// here is exactly how the `All`-path emission corruption survived review.
//
// Mostly synthetic signals, but not exclusively: the fp16 `log(0)` sentinel
// only appears on inputs whose probabilities fall under the fp16 floor, which
// silence and a low-amplitude sine never do — see
// `emissions_have_no_fp16_log_zero_sentinel`, which needs real speech.
// ---------------------------------------------------------------------

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

fn encoder_path() -> std::path::PathBuf {
  models_dir().join("base960h_aligner.mlmodelc")
}

/// Loads the real encoder model on [`DEFAULT_ENCODER_COMPUTE`] — the shipping
/// placement, via the same `EncoderOptions::new()` door production code takes.
/// Deliberately NOT a hardcoded `ComputeUnits::_`: every model-gated test
/// below is then a test OF the default.
fn load_encoder() -> Encoder {
  Encoder::from_file(encoder_path())
    .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS to the model directory)")
}

/// `EncoderInput::from_samples` for the model-gated tests below, whose fixtures
/// are always within the window. The fallible construction is F1's geometry
/// gate; its rejection path is proven hermetically by
/// `encoder_input_rejects_a_buffer_longer_than_the_window_before_any_prediction`
/// (no model needed), so there is no longer a model-gated too-long test — the
/// too-long buffer never reaches `emissions_raw` at all.
fn window_input(samples: &[f32]) -> EncoderInput<'_> {
  EncoderInput::from_samples(samples).expect("model-gated fixtures are <= ENCODER_WINDOW_SAMPLES")
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn from_file_loads_and_reports_frame_count() {
  let encoder = load_encoder();
  // Ground truth pinned by
  // `tests/model_io.rs::base960h_aligner_io_matches_spec`: 2,999 frames.
  assert_eq!(encoder.frames(), 2_999);
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_on_full_window_produces_correctly_shaped_finite_log_probs() {
  let encoder = load_encoder();
  let samples = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
  let raw = encoder
    .emissions_raw(window_input(&samples))
    .expect("emissions on silence");
  assert_eq!(raw.frames, encoder.frames());
  assert_eq!(
    raw.data.len(),
    raw.frames * crate::audio::align::vocab::VOCAB_SIZE
  );
  assert!(
    raw.data.iter().all(|v| v.is_finite()),
    "all log-probs finite"
  );
  // Log-probabilities are bounded above by log(1) == 0. This is also the
  // exact domain `Emissions::from_log_probs` enforces, so a pass here is a
  // canary that `Encoder::emissions` (the wrapped door) will not trip the
  // value-domain scan on this input.
  assert!(
    raw.data.iter().all(|&v| v <= 0.0),
    "log-probs must satisfy log(p) <= 0"
  );
}

/// **THE C1 REGRESSION ORACLE.** No emission cell may be an fp16 `log(0)`
/// saturation sentinel.
///
/// `base960h_aligner.mlmodelc` ends in an fp16 `softmax` followed by an fp16
/// `log` whose `epsilon = 0x1p-149` guard is far below fp16's smallest
/// subnormal and therefore inert (see [`DEFAULT_ENCODER_COMPUTE`]). On an ANE
/// placement every softmax output under the fp16 floor underflows to 0 and
/// `log(0)` saturates to ≈ `-45440`, silently replacing ordinary log-probs of
/// `-19.0` … `-21.75` and shifting real word timings by hundreds of ms.
///
/// The encoder is built from [`DEFAULT_ENCODER_COMPUTE`] — NEVER a hardcoded
/// placement — so this is a test of the shipping default. Flipping that
/// constant to `ComputeUnits::All` makes it fail (measured `min = -45440`,
/// 2,667 of 15,921 cells past the threshold); on `CpuOnly` it passes
/// (`min = -30.81`).
///
/// It must run on REAL SPEECH. This bug is invisible to synthetic input:
/// measured on the same model, 960,000 samples of silence bottom out at
/// `min = -8.55` and a low-amplitude sine at `-9.07` — both far ABOVE the fp16
/// floor (`log(2⁻²⁴) ≈ -16.6`), so nothing underflows and an `All` run of
/// either passes clean. Only real speech drives per-class probabilities down to
/// `e^-30.8 ≈ 4e-14`, deep under the floor. Hence the cross-crate `jfk.wav`
/// borrow.
///
/// [`LOG_PROB_FLOOR`] is not a tolerance to be relaxed: it separates two
/// populations three orders of magnitude apart (worst legitimate log-prob
/// measured anywhere on this model ≈ `-30.8`; the sentinel ≈ `-45440`).
/// Anything in between is already a broken emission matrix.
///
/// This measures the RAW tensor. `emissions_reject_an_ane_corrupted_matrix`
/// pins the same fact at the public door, where it is now an error rather than
/// a measurement.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_have_no_fp16_log_zero_sentinel() {
  let encoder = load_encoder();
  let samples = load_jfk_wav();
  let raw = encoder
    .emissions_raw(window_input(&samples))
    .expect("emissions on jfk.wav");

  let min = raw.data.iter().copied().fold(f32::INFINITY, f32::min);
  let sentinels = raw.data.iter().filter(|v| **v < LOG_PROB_FLOOR).count();
  assert_eq!(
    sentinels,
    0,
    "{sentinels} of {} emission cells are below {LOG_PROB_FLOOR} (min = {min}) — the fp16 \
     `log(0)` sentinel. The encoder is on {:?}; an ANE placement corrupts this model's emissions \
     and cannot be used. See DEFAULT_ENCODER_COMPUTE.",
    raw.data.len(),
    DEFAULT_ENCODER_COMPUTE,
  );
}

/// **THE SILENT-CORRUPTION REGRESSION.** An ANE-corrupted emission matrix must
/// be REJECTED by the public door, not returned as a plausible `Ok`.
///
/// [`EncoderOptions::with_compute`] is public and accepts `ComputeUnits::All`.
/// Before [`LOG_PROB_FLOOR`] existed, this exact call returned **`Ok`**: the
/// `-45440` sentinel is finite and `<= 0`, so it satisfies every check
/// [`Emissions::from_log_probs`] runs, and the caller got word timings that were
/// wrong by up to 881 ms with no diagnostic anywhere. Measured on the real
/// model, pre-guard: `Aligner::align_chunk(jfk, …)` → `Ok`, with `ask` at
/// 7533.7 ms instead of 8415.3 ms — a pre-truncation-fix measurement whose exact
/// ms shifted with the fix (see [`DEFAULT_ENCODER_COMPUTE`]).
///
/// REAL SPEECH is load-bearing, and a synthetic input cannot replace it: on the
/// corrupt path 960,000 samples of digital silence bottom out at `-8.55` and a
/// low-amplitude sine at `-9.07`, both ABOVE the fp16 floor
/// (`log(2⁻²⁴) ≈ -16.6`), so nothing underflows and this test would pass
/// **against the corrupt model**. Only real speech drives a class posterior
/// under the floor. Hence the cross-crate `jfk.wav` borrow.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_reject_an_ane_corrupted_matrix() {
  let encoder = Encoder::from_file_with(
    encoder_path(),
    EncoderOptions::new().with_compute(ComputeUnits::All),
  )
  .expect("load base960h_aligner.mlmodelc on ComputeUnits::All");
  let samples = load_jfk_wav();

  let Err(err) = encoder.emissions(window_input(&samples)) else {
    panic!(
      "an ANE-corrupted emission matrix was accepted. `Emissions::from_log_probs` cannot catch \
       this — -45440 is finite and <= 0 — so the caller now has plausible, silently wrong word \
       timings. LOG_PROB_FLOOR is the only thing standing here."
    );
  };
  let AlignError::CorruptEmissions {
    compute,
    min,
    cells,
    total,
  } = err
  else {
    panic!("expected AlignError::CorruptEmissions, got {err:?}");
  };
  // The measured ANE signature, pinned: 2,667 of 15,921 cells (16.7%),
  // min = -45440. Asserted as bounds rather than as equalities — the exact
  // count is a property of one OS/ANE firmware pair, but the ORDER of the
  // corruption is the fact worth pinning.
  assert_eq!(compute, ComputeUnits::All);
  assert_eq!(total, 549 * crate::audio::align::vocab::VOCAB_SIZE);
  assert!(
    cells > 0 && cells <= total,
    "corrupt cells: {cells}/{total}"
  );
  assert!(
    min < LOG_PROB_FLOOR,
    "reported min {min} must be past the floor it tripped"
  );
  // Self-diagnosing: the message must NAME the placement, or the caller is
  // left to rediscover a 450×-slower, 16.7%-corrupt configuration by hand.
  let rendered = AlignError::CorruptEmissions {
    compute,
    min,
    cells,
    total,
  }
  .to_string();
  assert!(
    rendered.contains("All"),
    "error must name the placement: {rendered}"
  );
  println!("rejected with: {rendered}");
}

/// The guard keys on the emission VALUES, never on the placement — so a
/// non-default but numerically-clean placement must still be accepted.
///
/// `CpuAndGpu` is that placement: measured `min = -30.02`, zero cells past
/// [`LOG_PROB_FLOOR`] on the same real speech the ANE corrupts. A guard that
/// rejected "any non-default compute" would fail here, and would also forbid a
/// future re-converted artifact that runs correctly on the ANE. This test is
/// what keeps the fix a value-domain check instead of a placement ban.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_accept_the_cpu_and_gpu_placement() {
  let encoder = Encoder::from_file_with(
    encoder_path(),
    EncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu),
  )
  .expect("load base960h_aligner.mlmodelc on ComputeUnits::CpuAndGpu");
  let samples = load_jfk_wav();

  let emissions = encoder
    .emissions(window_input(&samples))
    .expect("CpuAndGpu emissions are clean log-probs and must pass the floor guard");
  assert_eq!(emissions.frames(), 549);
  assert_eq!(
    emissions.vocab().get(),
    crate::audio::align::vocab::VOCAB_SIZE
  );
}

/// The shipping default on the same real speech, through the SAME guarded door
/// the ANE test fails at — the third leg of the placement-agnostic proof
/// (`CpuOnly` Ok, `CpuAndGpu` Ok, `All` Err).
///
/// `emissions_wraps_into_validated_emissions` covers the door on silence, which
/// (as `emissions_reject_an_ane_corrupted_matrix` explains) never reaches the
/// failure regime at all — so it cannot stand in for this.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_accept_the_default_placement_on_real_speech() {
  let encoder = load_encoder();
  let samples = load_jfk_wav();
  let emissions = encoder
    .emissions(window_input(&samples))
    .unwrap_or_else(|e| panic!("the SHIPPING placement must produce clean log-probs: {e}"));
  assert_eq!(emissions.frames(), 549);
  assert_eq!(
    emissions.vocab().get(),
    crate::audio::align::vocab::VOCAB_SIZE
  );
}

/// **THE NORMALIZATION-GUARD REGRESSION (c).** Real emissions from the shipping
/// artifact must PASS `check_log_prob_normalization` — the guard that rejects a
/// raw-logit model swap — on both gate clips and both numerically-clean gate
/// placements, with the measured worst per-frame `|logsumexp|` comfortably under
/// [`LOG_PROB_SUM_TOLERANCE`].
///
/// This is the model side of the tolerance calibration: it re-measures, at gate
/// time, the worst `|logsumexp|` [`LOG_PROB_SUM_TOLERANCE`]'s doc records
/// (`CpuOnly` `ted_60` 5.2485e-3, `jfk` 4.7453e-3; `CpuAndGpu` ~2.5e-7), so a
/// future artifact or firmware whose jitter crept toward the bound would fail
/// here rather than silently at a caller. It exercises the guarded door's exact
/// check pair (`check_log_prob_floor` then `check_log_prob_normalization`) on the
/// truncated real tensor; the end-to-end public door on real speech is covered by
/// `emissions_accept_the_default_placement_on_real_speech` (jfk `CpuOnly`) and
/// `emissions_accept_the_cpu_and_gpu_placement` (jfk `CpuAndGpu`), which now run
/// the guard too, and by `tests/parity_words.rs` on `ted_60`.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_pass_the_normalization_guard_on_real_speech() {
  for compute in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndGpu] {
    let encoder =
      Encoder::from_file_with(encoder_path(), EncoderOptions::new().with_compute(compute))
        .unwrap_or_else(|e| panic!("load base960h_aligner.mlmodelc on {compute:?}: {e}"));
    for (name, samples) in [("jfk", load_jfk_wav()), ("ted_60", load_ted_60_wav())] {
      let raw = encoder
        .emissions_raw(window_input(&samples))
        .unwrap_or_else(|e| panic!("{compute:?} {name}: emissions_raw: {e}"));
      // The exact guarded-door pair, on the exact truncated tensor the door checks.
      check_log_prob_floor(&raw.data, compute)
        .unwrap_or_else(|e| panic!("{compute:?} {name}: real emissions tripped the floor: {e}"));
      check_log_prob_normalization(&raw.data, compute).unwrap_or_else(|e| {
        panic!("{compute:?} {name}: real emissions tripped the normalization guard: {e}")
      });
      // The measurement of record: worst per-frame |logsumexp|, f64-accumulated,
      // over the real (truncated) frames the guard scans.
      let worst = raw
        .data
        .as_chunks::<{ crate::audio::align::vocab::VOCAB_SIZE }>()
        .0
        .iter()
        .map(|frame| {
          let max = f64::from(frame.iter().copied().fold(f32::NEG_INFINITY, f32::max));
          let sum: f64 = frame.iter().map(|&x| (f64::from(x) - max).exp()).sum();
          (max + sum.ln()).abs()
        })
        .fold(0.0f64, f64::max);
      println!(
        "{compute:?} {name}: {} frames, worst |logsumexp| = {worst:.6e} (tolerance {LOG_PROB_SUM_TOLERANCE:e})",
        raw.frames,
      );
      assert!(
        worst < LOG_PROB_SUM_TOLERANCE,
        "{compute:?} {name}: worst |logsumexp| {worst} is not under the guard tolerance \
         {LOG_PROB_SUM_TOLERANCE} — the tolerance's measured headroom has been lost"
      );
    }
  }
}

/// Decodes the 11 s `jfk.wav` fixture (16 kHz mono int16) to f32 samples.
///
/// Borrowed from the whisperkit crate by relative path rather than committing
/// a second copy — the same borrow `tests/common/mod.rs` makes, and it FAILS
/// LOUDLY (never skips) if that path ever moves.
fn load_jfk_wav() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../whisperkit/tests/fixtures/audio/jfk.wav");
  let mut reader = hound::WavReader::open(&path)
    .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

/// Decodes the 60 s `ted_60.wav` fixture (16 kHz mono int16) to f32 samples —
/// exactly [`ENCODER_WINDOW_SAMPLES`] (960,000), the full window with no padding,
/// so the normalization guard sees all 2,999 real frames. Borrowed cross-crate
/// exactly as [`load_jfk_wav`]; fails loudly (never skips) if the path moves or
/// the clip stops filling the window.
fn load_ted_60_wav() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../whisperkit/tests/fixtures/audio/ted_60.wav");
  let mut reader = hound::WavReader::open(&path)
    .unwrap_or_else(|e| panic!("open the ted_60.wav fixture at {path:?}: {e}"));
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect();
  assert_eq!(
    samples.len(),
    ENCODER_WINDOW_SAMPLES,
    "ted_60.wav must fill the encoder window exactly (the zero-padding-free path)"
  );
  samples
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_wraps_into_validated_emissions() {
  // The wrapped door: proves `Emissions::from_log_probs`' O(T·V) value scan
  // passes on the real model's output (the fp16 log-prob ceiling holds), and
  // that the shape handshake (`frames`/`vocab`) survives the wrap.
  let encoder = load_encoder();
  let samples = vec![0.0f32; 48_000];
  let emissions = encoder
    .emissions(window_input(&samples))
    .expect("emissions wraps into a validated Emissions");
  assert_eq!(emissions.frames(), 149);
  assert_eq!(
    emissions.vocab().get(),
    crate::audio::align::vocab::VOCAB_SIZE
  );
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_on_short_input_truncates_to_hermetic_formula() {
  let encoder = load_encoder();
  // 3 s @ 16 kHz: well under the model's 2,999-frame ceiling, so the
  // real model's output must match the pure hermetic formula exactly
  // (cross-validates `truncated_frame_count` against the live model,
  // not just itself).
  let samples = vec![0.0f32; 48_000];
  let raw = encoder
    .emissions_raw(window_input(&samples))
    .expect("emissions on short input");
  assert_eq!(raw.frames, truncated_frame_count(48_000, encoder.frames()));
  assert_eq!(raw.frames, 149);
  assert_eq!(raw.data.len(), 149 * crate::audio::align::vocab::VOCAB_SIZE);
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_is_deterministic_across_repeated_calls() {
  let encoder = load_encoder();
  // Small-amplitude non-zero signal, not pure silence, so this exercises
  // real signal-path compute rather than just a bias/floor.
  let samples: Vec<f32> = (0..ENCODER_WINDOW_SAMPLES)
    .map(|i| 0.01 * (i as f32 * 0.001).sin())
    .collect();
  let first = encoder
    .emissions_raw(window_input(&samples))
    .expect("first emissions call");
  let second = encoder
    .emissions_raw(window_input(&samples))
    .expect("second emissions call");
  assert_eq!(first.frames, second.frames);
  assert_eq!(
    first.data, second.data,
    "repeated emissions_raw() must be bit-identical"
  );
}
