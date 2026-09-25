use super::*;

use core::num::NonZeroU32;

/// [`truncated_frame_count`] on the staged model's geometry,
/// [`AcousticGeometry::WAV2VEC2`] — the one every fixture below was measured on.
fn staged(real_samples: usize, available_frames: usize) -> usize {
  truncated_frame_count(AcousticGeometry::WAV2VEC2, real_samples, available_frames)
}

/// A contract of a model's own with the staged tokenization and
/// log-probability output: only `blank` and `geometry` vary here.
fn contract(blank: u32, geometry: AcousticGeometry) -> AcousticContract {
  AcousticContract::new(
    blank,
    geometry,
    crate::audio::align::acoustic::Tokenization::new(
      crate::audio::align::acoustic::WordDelimiter::Pipe,
      crate::audio::align::acoustic::LetterCase::Upper,
      crate::audio::align::acoustic::Granularity::Character,
      &[],
    ),
    OutputKind::LogProbabilities,
  )
}

/// A geometry at 16 kHz, `receptive_field` and `stride` samples.
fn geometry(receptive_field: u32, stride: u32) -> AcousticGeometry {
  AcousticGeometry::new(
    16_000,
    NonZeroU32::new(receptive_field).expect("nonzero"),
    NonZeroU32::new(stride).expect("nonzero"),
  )
  .expect("a geometry asry's seam times")
}

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
  assert_eq!(staged(0, 2999), 0);
}

#[test]
fn truncated_frame_count_sub_receptive_field_is_one_frame() {
  // Anything from a single sample up to the full 400-sample receptive field
  // yields exactly ONE frame: the wav2vec2 conv stack needs a complete
  // receptive field for its first output, and asry pads a sub-400 chunk up to
  // 400 before it — so one real sample and a full receptive field are the same
  // one frame. (Reverting `saturating_sub` to a bare `-` underflows here.)
  //
  // HAND-COMPUTED literal inputs, deliberately NOT derived from the staged
  // geometry: a boundary derived from the value it means to pin moves WITH the
  // value under mutation and stays green (F3). 320 and 400 are spelled out;
  // `the_staged_geometry_is_pinned` pins the geometry itself.
  for real_samples in [1, 200, 320, 399, 400] {
    assert_eq!(
      staged(real_samples, 2999),
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
  assert_eq!(staged(321, 2999), 1); // 321: was ceil → 2
  assert_eq!(staged(641, 2999), 1); // was ceil → 3
}

#[test]
fn truncated_frame_count_adds_one_frame_per_hop_past_the_receptive_field() {
  // At and above the receptive field the count is `floor((L - 400)/320) + 1`:
  // each further 320-sample hop past the first full receptive field adds one
  // frame. Catches dropping the `+ 1` (401 → 0) or a wrong divisor.
  //
  // HAND-COMPUTED literals (NOT derived from the constants — see F3): 401 → 1,
  // 720 → 2, 1040 → 3.
  assert_eq!(staged(401, 2999), 1);
  assert_eq!(staged(720, 2999), 2);
  assert_eq!(staged(1040, 2999), 3);
}

#[test]
fn truncated_frame_count_receptive_field_boundary_is_pinned_by_hand() {
  // THE F3 pin. Literal, hand-computed frame counts across the first-frame
  // boundary, referencing NEITHER the staged receptive field nor its stride — so
  // mutating either (e.g. the receptive field 400 → 399) cannot slide the inputs
  // and expectations to stay green, the exact defect this replaces (a
  // self-derived fixture became 719 → 2 and still passed).
  //
  // With RF = 400, HOP = 320 the count holds at 1 up to and including 719 real
  // samples (719 does not fill a SECOND 400-wide window past the first hop) and
  // steps to 2 at 720. Under the RF = 399 mutation the step falls to 719, so
  // `719 → 1` is the assertion that catches it (it would return 2); `720 → 2`
  // pins the true step.
  assert_eq!(staged(399, 2999), 1);
  assert_eq!(staged(400, 2999), 1);
  assert_eq!(staged(719, 2999), 1);
  assert_eq!(staged(720, 2999), 2);
}

#[test]
fn the_staged_geometry_is_pinned() {
  // Direct literal pins on the staged geometry itself, so a change to it is a
  // loud, single-line failure and not a silent re-derivation of the frame-count
  // fixtures. This is wav2vec2's front end: 16 kHz audio, a 400-sample
  // receptive field and a 320-sample (20 ms @ 16 kHz) stride — the staged
  // contract's.
  let geometry = AcousticGeometry::WAV2VEC2;
  assert_eq!(geometry.sample_rate().get(), 16_000);
  assert_eq!(geometry.receptive_field().get(), 400);
  assert_eq!(geometry.stride().get(), 320);
  assert_eq!(HOP_SAMPLES, 320);
  assert_eq!(AcousticContract::BASE960H.geometry(), geometry);
}

#[test]
fn truncated_frame_count_reference_short_clip() {
  // 48,000 samples (3 s), well under the model's 2,999-frame ceiling: the
  // reference conv output is `floor((48_000 - 400)/320) + 1 == 149`. The old
  // `ceil(48_000/320) == 150` over-counted by one, so reverting to `div_ceil`
  // fails here (150, not 149). Cross-validated against the LIVE model by
  // `emissions_on_short_input_truncates_to_hermetic_formula`.
  assert_eq!(staged(48_000, 2_999), 149);
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
  assert_eq!(staged(ENCODER_WINDOW_SAMPLES, 2_999), 2_999);
}

#[test]
fn truncated_frame_count_approaches_the_full_window_without_overshoot() {
  // The formula climbs to 2,999 and stops there — it never exceeds the model
  // count for any in-window input, so the clamp is defensive, not corrective.
  // `2_999 * 320 == 959_680` is now 2_998 (not the old ceil's 2_999); the count
  // first reaches 2_999 at 959_760 and holds it through the full window.
  assert_eq!(staged(2_999 * HOP_SAMPLES, 2_999), 2_998);
  assert_eq!(staged(959_760, 2_999), 2_999);
  assert_eq!(staged(ENCODER_WINDOW_SAMPLES - 1, 2_999), 2_999);
}

#[test]
fn truncated_frame_count_clamp_engages_only_below_the_formula() {
  // The `.min(available_frames)` clamp never fires for the real model (the
  // formula tops out at exactly its 2,999), so prove it against a SMALLER
  // hypothetical frame budget: 48,000 samples nominally yield 149, but a
  // 100-frame model must cap at 100. Catches a `.min` → `.max` mutant, which
  // would return 149 here.
  assert_eq!(staged(48_000, 100), 100);
  assert_eq!(staged(48_000, 149), 149); // exactly at the budget: no clamp
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
    let t = staged(real_samples, available_frames);
    assert!(
      t <= available_frames,
      "staged({real_samples}, {available_frames}) = {t} exceeds available_frames"
    );
  }
  // Load-bearing clamp: above the window the conv formula overshoots 2,999, and
  // ONLY the `.min` holds the result at exactly `base960h_aligner.mlmodelc`'s
  // frame count. These EXACT `== 2_999` checks fail the instant `.min` is
  // deleted or flipped to `.max` — the formula then returns 3,000 and 3,009.
  assert_eq!(
    staged(ENCODER_WINDOW_SAMPLES + HOP_SAMPLES, available_frames),
    2_999
  );
  assert_eq!(
    staged(ENCODER_WINDOW_SAMPLES + 10 * HOP_SAMPLES, available_frames),
    2_999
  );
}

/// **The truncation is the contract's geometry, not the staged one's.** A
/// 640-sample receptive field at a 320-sample stride makes the same 2999 frames
/// of the staged 960,000-sample window as wav2vec2's 400, so no declaration
/// tells the two apart. On 720 real samples they part: the 640-sample field
/// yields ONE complete frame, where the staged geometry keeps two. Truncating
/// such a model by the staged 400 would keep a frame computed from padding.
#[test]
fn a_640_320_contract_truncates_720_samples_to_one_frame() {
  let wide = geometry(640, 320);
  assert_eq!(
    wide.frames(ENCODER_WINDOW_SAMPLES),
    AcousticGeometry::WAV2VEC2.frames(ENCODER_WINDOW_SAMPLES),
    "both geometries make the staged window's 2999 frames: the declaration cannot tell them apart"
  );
  assert_eq!(truncated_frame_count(wide, 720, 2_999), 1);
  assert_eq!(staged(720, 2_999), 2);
  // Its own boundaries, by hand: one frame up to 959 samples, two from 960.
  assert_eq!(truncated_frame_count(wide, 1, 2_999), 1);
  assert_eq!(truncated_frame_count(wide, 959, 2_999), 1);
  assert_eq!(truncated_frame_count(wide, 960, 2_999), 2);
  assert_eq!(truncated_frame_count(wide, 0, 2_999), 0);
}

/// **A geometry the declaration contradicts is refused by name.** The check the
/// load makes of the contract: its geometry must make exactly the declared
/// frame count of the declared window. The staged model's own pair passes under
/// wav2vec2's geometry and under the 640-sample one (the ambiguity the
/// declaration leaves); a 160-sample stride would make 5998 frames, a 321-sample
/// one 2990, and a window shorter than the receptive field none — each refused,
/// naming the geometry and both counts.
#[test]
fn check_frame_count_refuses_a_geometry_the_declaration_contradicts() {
  let window = NonZeroUsize::new(ENCODER_WINDOW_SAMPLES).expect("nonzero");
  let frames = NonZeroUsize::new(EXPECTED_OUTPUT_FRAMES).expect("nonzero");
  assert_eq!(
    check_frame_count(AcousticGeometry::WAV2VEC2, window, frames),
    Ok(())
  );
  assert_eq!(
    check_frame_count(geometry(640, 320), window, frames),
    Ok(())
  );

  for (geometry, window, derived) in [
    (geometry(400, 160), ENCODER_WINDOW_SAMPLES, 5_998),
    (geometry(400, 321), ENCODER_WINDOW_SAMPLES, 2_990),
    (geometry(400, 320), 399, 0),
  ] {
    let window = NonZeroUsize::new(window).expect("nonzero");
    let Err(AlignerError::FrameCountMismatch(mismatch)) =
      check_frame_count(geometry, window, frames)
    else {
      panic!("{geometry:?} must be refused against a {window}-sample window of 2999 frames");
    };
    assert_eq!(mismatch.geometry(), geometry);
    assert_eq!(mismatch.window(), window.get());
    assert_eq!(mismatch.declared(), EXPECTED_OUTPUT_FRAMES);
    assert_eq!(mismatch.derived(), derived);
  }
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
  let input = EncoderInput::from_samples(&chunk);
  assert_eq!(input.real_samples, 176_000);
  assert_eq!(input.encoder_input.len(), 176_000);
  assert_eq!(staged(input.real_samples, 2_999), 549);
  // The buggy answer is now unreachable: 175_360 gives 547, but nothing can
  // bind 175_360 to this 176,000-sample buffer.
  assert_eq!(staged(175_360, 2_999), 547);
  assert_ne!(staged(input.real_samples, 2_999), staged(175_360, 2_999));
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
  let input = EncoderInput::new(&padded_buffer, real_len);
  assert_eq!(input.real_samples, 200); // the UNPADDED count, NOT 400
  assert_eq!(input.encoder_input.len(), 400);
  // Under the corrected conv-geometry truncation this sub-receptive-field slip is
  // BENIGN for the count: 200 real samples and the 400-sample pad both yield the
  // single receptive-field frame — `ceil` was the only thing that ever made them
  // 1 vs 2. The binding still matters (it records the honest length and stays
  // correct for the general case pinned by
  // `encoder_input_from_samples_binds_real_length_to_the_slice`: 176_000 vs
  // 175_360 → 549 vs 547, where a short real count genuinely moves the count).
  assert_eq!(staged(input.real_samples, 2_999), 1);
  assert_eq!(staged(padded_buffer.len(), 2_999), 1);
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
  let via_prepared = EncoderInput::from_prepared(&prepared);
  assert_eq!(
    via_prepared.real_samples, 200,
    "from_prepared must record the true pre-pad real_samples (200), never the padded 400"
  );

  // The raw door, handed the SAME padded buffer, records the padded length — the
  // provenance the frame-count coincidence (both truncate to one frame) hides, and
  // exactly what the vacuous mutation collapses `from_prepared` into.
  let via_raw = EncoderInput::from_samples(prepared.encoder_input());
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

/// The window is the MODEL's, read at load, so the check that a buffer fits it
/// belongs to the encoder: `check_window`, which `emissions_raw` runs before any
/// prediction. A buffer one sample past the window is refused naming both
/// lengths; exactly the window — the `ted_60.wav` case, where `emissions_raw`
/// borrows the buffer rather than padding it — and anything shorter pass. The
/// model-gated `emissions_refuse_a_buffer_longer_than_the_window_before_predicting`
/// drives it through the public door.
#[test]
fn check_window_refuses_a_buffer_longer_than_the_models_window() {
  let window = NonZeroUsize::new(ENCODER_WINDOW_SAMPLES).expect("nonzero");
  let Err(AlignError::InputTooLong(too_long)) = check_window(ENCODER_WINDOW_SAMPLES + 1, window)
  else {
    panic!("one sample past the window must be refused");
  };
  assert_eq!(
    (too_long.got(), too_long.max()),
    (ENCODER_WINDOW_SAMPLES + 1, ENCODER_WINDOW_SAMPLES)
  );
  assert!(check_window(ENCODER_WINDOW_SAMPLES, window).is_ok());
  assert!(check_window(0, window).is_ok());

  // Another model's window is its own ceiling.
  let short = NonZeroUsize::new(480_000).expect("nonzero");
  assert!(check_window(480_001, short).is_err());
  assert!(check_window(480_000, short).is_ok());
}

#[test]
fn encoder_input_binds_a_full_window_as_real() {
  // The exact-window buffer is real audio end to end: nothing to pad, and
  // nothing past the real samples to truncate.
  let full = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
  let input = EncoderInput::from_samples(&full);
  assert_eq!(input.real_samples, ENCODER_WINDOW_SAMPLES);
  assert_eq!(input.encoder_input.len(), ENCODER_WINDOW_SAMPLES);
}

// ---------------------------------------------------------------------
// check_sentinel_band: hermetic coverage of the band a contract may carry —
// the staged model's fp16 `log(0)` sentinel. The model-gated half
// (`emissions_reject_an_ane_corrupted_matrix`) proves the real ANE artifact
// trips it; these prove the predicate itself, including the boundaries a mutant
// would move, and that a contract without a band refuses nothing finite.
// ---------------------------------------------------------------------

/// The staged contract's band.
const STAGED_BAND: Option<SentinelBand> = AcousticContract::BASE960H.sentinel_band();

#[test]
fn check_sentinel_band_accepts_real_log_probs() {
  // The measured legitimate range on the staged model: max exactly 0.0, min
  // -30.81 (`CpuOnly`) / -30.02 (`CpuAndGpu`). Nothing here is near the band.
  let data = [0.0, -0.06, -19.0, -21.75, -30.02, -30.81];
  assert!(check_sentinel_band(&data, STAGED_BAND, ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_sentinel_band_accepts_an_empty_matrix() {
  // `real_samples == 0` truncates to zero frames; the guard must not invent a
  // failure out of an empty scan (min would be +inf).
  assert!(check_sentinel_band(&[], STAGED_BAND, ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_sentinel_band_rejects_the_fp16_log_zero_sentinel() {
  // One corrupt cell in an otherwise clean matrix is still a corrupt matrix:
  // the ANE run corrupts 16.7% of cells, but a single one is enough to move a
  // trellis path. Catches a mutant that thresholds on a FRACTION of cells.
  let data = [0.0, -1.5, -45_440.0, -20.0];
  let Err(err) = check_sentinel_band(&data, STAGED_BAND, ComputeUnits::All) else {
    panic!("the -45440 fp16 log(0) sentinel must be rejected");
  };
  let AlignError::CorruptEmissions(ref e) = err else {
    panic!("expected AlignError::CorruptEmissions, got {err:?}");
  };
  assert_eq!(e.compute(), ComputeUnits::All);
  assert_eq!(e.band(), SentinelBand::Fp16Saturation);
  assert_eq!(e.min(), -45_440.0);
  assert_eq!(e.cells(), 1);
  assert_eq!(e.total(), 4);
}

#[test]
fn the_band_holds_its_ceiling_and_nothing_above_it() {
  // The band is fp16's saturation binade, and `-32768` is its top value: the
  // ceiling is IN the band, anything above it is not. Pins the comparison's
  // direction and strictness together: a mutant flipping `<=` to `<` fails the
  // first assertion, one flipping it to `>=` fails the second.
  let ceiling = SentinelBand::Fp16Saturation.ceiling();
  assert_eq!(ceiling, -32_768.0);
  assert!(check_sentinel_band(&[ceiling], STAGED_BAND, ComputeUnits::CpuOnly).is_err());
  assert!(
    check_sentinel_band(&[-32_767.0], STAGED_BAND, ComputeUnits::CpuOnly).is_ok(),
    "a value above the binade is no saturated fp16 log(0)"
  );
}

#[test]
fn check_sentinel_band_leaves_non_finite_values_to_from_log_probs() {
  // Deliberate division of labour, documented on `check_sentinel_band`: the
  // band guard refuses the band only. `NaN` compares false against everything
  // and passes here; `Emissions::from_log_probs`' finite ∧ <= 0 scan (which runs
  // on the very next line of `Encoder::emissions`) is what rejects it. Neither
  // scan is redundant with the other, and this pins that seam so a later
  // "simplification" cannot silently drop one of them.
  assert!(check_sentinel_band(&[f32::NAN], STAGED_BAND, ComputeUnits::CpuOnly).is_ok());
  assert!(check_sentinel_band(&[f32::INFINITY], STAGED_BAND, ComputeUnits::CpuOnly).is_ok());
  // -inf is below the band's ceiling and IS the band's business.
  assert!(check_sentinel_band(&[f32::NEG_INFINITY], STAGED_BAND, ComputeUnits::CpuOnly).is_err());
}

/// The rows below, each a normalized pair of log-probabilities
/// (`logsumexp([0, x]) ≈ 0`), through the door's whole value-domain guard and
/// the wrap, under `band`.
fn guard_row(tail: f32, band: Option<SentinelBand>) -> Result<Emissions, AlignError> {
  let two = NonZeroUsize::new(2).expect("nonzero");
  RawEmissions {
    frames: 1,
    vocab_size: two,
    data: vec![0.0, tail],
  }
  .check_value_domain(band, ComputeUnits::All)?
  .into_emissions()
}

/// **A contract of a model's own refuses no finite log-probability, and the
/// staged contract still refuses its sentinel.** No law bounds a
/// log-probability from below: `[0, -101]`, `[0, -32000]` and `[0, -40000]` are
/// normalized rows, and so is `[0, -45440]`. A contract made with
/// [`AcousticContract::new`] carries no band, so every one of them clears the
/// whole value-domain guard and the wrap. The staged contract's band is a
/// measurement of that artifact, where a value at or below `-32768` is its
/// saturated fp16 `log(0)`: it still refuses `-45440`, and `-40000` with it,
/// while `-101` and `-32000` pass there too.
#[test]
fn a_generic_contract_refuses_no_finite_log_probability() {
  let generic = contract(0, AcousticGeometry::WAV2VEC2);
  assert_eq!(generic.sentinel_band(), None);
  for tail in [-101.0f32, -500.0, -32_000.0, -40_000.0, -45_440.0] {
    let emissions = guard_row(tail, generic.sentinel_band())
      .unwrap_or_else(|err| panic!("[0, {tail}] is a row of log-probabilities: {err:?}"));
    assert_eq!(emissions.vocab().get(), 2);
  }

  for tail in [-40_000.0f32, -45_440.0] {
    assert!(
      matches!(
        guard_row(tail, STAGED_BAND),
        Err(AlignError::CorruptEmissions(_))
      ),
      "[0, {tail}] is in the staged model's band"
    );
  }
  for tail in [-101.0f32, -500.0, -32_000.0] {
    assert!(
      guard_row(tail, STAGED_BAND).is_ok(),
      "[0, {tail}] is above the staged model's band"
    );
  }
}

// The band's place (at or below the top of fp16's saturation binade, above the
// -45440 sentinel) is asserted in `acoustic/mod.rs` at COMPILE time, not here:
// both operands are constants, so a runtime test of it is dead weight that only
// fires after a build already succeeded.

// ---------------------------------------------------------------------
// read_emissions: the tensor a prediction RETURNS, against the declared
// `[1, frames, V]`, before a single cell is copied.
// ---------------------------------------------------------------------

/// **The shape a prediction returns is checked, not assumed.** `copy_into`
/// checks only the element count, which a transposed `[1, V, frames]` — or
/// any reshape of the same count — shares with the declared `[1, frames, V]`.
/// Driven through the door's own reader over real `MultiArray`s: every such
/// shape is refused with both shapes named, as are a head of another width and
/// a cropped frame axis, and only the declared shape is copied — cell for cell.
#[test]
fn read_emissions_refuses_every_shape_but_the_declared_one() {
  const FRAMES: usize = 3;
  let width = WIDTH.get();
  let cells = |count: usize| (0..count).map(|i| -(i as f32)).collect::<Vec<f32>>();

  let declared = cells(FRAMES * width);
  let tensor = MultiArray::from_slice(&[1, FRAMES, width], &declared).expect("build a tensor");
  assert_eq!(
    read_emissions(&tensor, FRAMES, WIDTH).expect("the declared shape is read"),
    declared
  );

  for shape in [
    vec![1, width, FRAMES],
    vec![FRAMES, width, 1],
    vec![FRAMES, width],
    vec![1, FRAMES * width],
    vec![1, FRAMES, 32],
    vec![1, FRAMES - 1, width],
  ] {
    let tensor =
      MultiArray::from_slice(&shape, &cells(shape.iter().product())).expect("build a tensor");
    let Err(AlignError::OutputShape(mismatch)) = read_emissions(&tensor, FRAMES, WIDTH) else {
      panic!("{shape:?} must be refused as an output-shape mismatch");
    };
    assert_eq!(mismatch.got(), shape.as_slice());
    assert_eq!(mismatch.expected(), [1, FRAMES, width]);
  }
}

// ---------------------------------------------------------------------
// check_log_prob_normalization: hermetic coverage of the per-frame logsumexp
// guard — the check that makes the "these really are log-probs" contract true
// for a model-artifact swap the sentinel band and `from_log_probs`'s finite ∧
// <= 0 scan both miss. The model-gated half
// (`emissions_pass_the_normalization_guard_on_real_speech`) proves the real
// artifact passes on both clips and both clean placements; these prove the
// predicate rejects the two un-normalized inputs the finding names, AND that
// neither the band nor the <= 0 scan would have caught them (the closed bypass),
// and that the allowance grows with the head's width.
// ---------------------------------------------------------------------

/// The staged `base960h` head's width, as the guards take it: 29 classes.
const WIDTH: NonZeroUsize = match NonZeroUsize::new(crate::audio::align::vocab::VOCAB_SIZE) {
  Some(width) => width,
  None => unreachable!(),
};

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
    check_log_prob_normalization(&data, WIDTH, ComputeUnits::CpuOnly).is_ok(),
    "normalized log-prob frames (logsumexp ≈ 0) must pass"
  );
}

#[test]
fn check_log_prob_normalization_accepts_an_empty_matrix() {
  // `real_samples == 0` truncates to zero frames; no frame to check, so Ok
  // (mirrors `check_sentinel_band_accepts_an_empty_matrix`).
  assert!(check_log_prob_normalization(&[], WIDTH, ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_log_prob_normalization_rejects_shifted_raw_logits() {
  // THE bypass this guard closes. A full 2999 × 29 matrix of raw logits shifted
  // WHOLLY into [-20, -10] — the finding's exact fence. Every cell is finite and
  // <= 0, so it passes BOTH the staged band (nothing near -32768) and the finite
  // ∧ <= 0 scan `from_log_probs` runs — yet no frame is a distribution: a row
  // entirely in [-20, -10] has logsumexp in [max, max + ln 29] ⊆ [-20, -6.63],
  // so |logsumexp| >= 6.63, orders of magnitude past the 29-class allowance.
  let mut data = Vec::with_capacity(2999 * crate::audio::align::vocab::VOCAB_SIZE);
  for _ in 0..2999 {
    // A ramp across the vocab, every value inside [-20, -10]; not normalized.
    for j in 0..crate::audio::align::vocab::VOCAB_SIZE {
      data
        .push(-10.0 - (j as f32) * (10.0 / (crate::audio::align::vocab::VOCAB_SIZE as f32 - 1.0)));
    }
  }
  // The band does NOT catch it (nothing near -32768)...
  assert!(
    check_sentinel_band(&data, STAGED_BAND, ComputeUnits::CpuOnly).is_ok(),
    "shifted raw logits in [-20, -10] are all above the staged band — it cannot catch them"
  );
  // ...and `from_log_probs`'s finite ∧ <= 0 scan would not either.
  assert!(
    data.iter().all(|v| v.is_finite() && *v <= 0.0),
    "shifted raw logits are finite and <= 0 — the from_log_probs scan cannot catch them"
  );
  // Only the normalization guard does.
  let Err(err) = check_log_prob_normalization(&data, WIDTH, ComputeUnits::CpuOnly) else {
    panic!("raw logits shifted into [-20, -10] must be rejected as un-normalized");
  };
  let AlignError::UnnormalizedEmissions(ref e) = err else {
    panic!("expected AlignError::UnnormalizedEmissions, got {err:?}");
  };
  let logsumexp = e.logsumexp();
  assert!(
    logsumexp.abs() > 6.6,
    "a [-20, -10] shifted frame's |logsumexp| is >= 6.63, got {logsumexp}"
  );
  assert_eq!(e.tolerance(), log_prob_sum_tolerance(WIDTH));
}

#[test]
fn check_log_prob_normalization_rejects_an_all_zero_frame() {
  // THE simplest un-normalized case: an all-zeros frame, exp(0) = 1 on every
  // class, so logsumexp = ln(29) ≈ 3.367 — again finite, <= 0, above the band,
  // and again only the normalization guard rejects it.
  let data = uniform_frame(0.0);
  assert!(
    check_sentinel_band(&data, STAGED_BAND, ComputeUnits::CpuOnly).is_ok(),
    "an all-zeros frame is above the band"
  );
  assert!(data.iter().all(|v| v.is_finite() && *v <= 0.0));
  let Err(AlignError::UnnormalizedEmissions(e)) =
    check_log_prob_normalization(&data, WIDTH, ComputeUnits::All)
  else {
    panic!("an all-zeros frame (logsumexp = ln 29) must be rejected");
  };
  let logsumexp = e.logsumexp();
  assert_eq!(e.row(), 0);
  assert_eq!(e.compute(), ComputeUnits::All); // the placement is carried through
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
  let Err(AlignError::UnnormalizedEmissions(e)) =
    check_log_prob_normalization(&data, WIDTH, ComputeUnits::CpuOnly)
  else {
    panic!("the un-normalized frame must be rejected");
  };
  assert_eq!(e.row(), bad_index, "the error must name the worst frame");
}

#[test]
fn check_log_prob_normalization_thresholds_on_the_tolerance() {
  // Pins the threshold LOCATION and direction: a uniform frame constructed to
  // sit at logsumexp = TOL/2 passes, one at logsumexp = 2·TOL is rejected. (An
  // exact-at-TOL boundary is not pinned here — an f32-stored frame cannot hit an
  // f64 TOL exactly; the `>` strictness is stated on the function.)
  let ln29 = f64::from(crate::audio::align::vocab::VOCAB_SIZE as u32).ln();
  let tol = log_prob_sum_tolerance(WIDTH);
  // uniform frame value `v` gives logsumexp = v + ln29; solve for the target.
  let inside = uniform_frame((tol / 2.0 - ln29) as f32);
  let outside = uniform_frame((2.0 * tol - ln29) as f32);
  assert!(
    check_log_prob_normalization(&inside, WIDTH, ComputeUnits::CpuOnly).is_ok(),
    "logsumexp = TOL/2 is within tolerance"
  );
  assert!(
    check_log_prob_normalization(&outside, WIDTH, ComputeUnits::CpuOnly).is_err(),
    "logsumexp = 2·TOL exceeds tolerance"
  );
}

/// **The allowance is the arithmetic's, and grows with the head.** A
/// log-softmax computed in fp16 normalizes a frame of `V` classes only to within
/// `(V + 2 + 2 ln V)` roundings of `2^-11`; the allowance is `2·(V + 1)·2^-11`,
/// which covers that at every width: `0.0293` for the staged 29 classes (5.6×
/// its measured worst jitter, `5.2485e-3`), `3.42` for a 3,500-class
/// character head. A fixed allowance measured on the 29-class head refused such a
/// head's genuine fp16 frames. What the guard exists to refuse stays whole units
/// away for the heads a CTC aligner uses: an all-zeros frame is off by `ln V`,
/// refused at every width up to about 9,000 classes.
#[test]
fn the_allowance_grows_with_the_head_and_still_refuses_an_all_zeros_frame() {
  let width = |v: usize| NonZeroUsize::new(v).expect("nonzero");
  assert_eq!(log_prob_sum_tolerance(width(29)), 60.0 / 2048.0);
  assert_eq!(log_prob_sum_tolerance(width(3_500)), 7_002.0 / 2048.0);
  for v in [2usize, 29, 32, 100, 1_000, 3_500, 8_000] {
    let tolerance = log_prob_sum_tolerance(width(v));
    let ln_v = (v as f64).ln();
    // The derived worst case of a genuine fp16 frame fits inside it...
    assert!(
      (v as f64 + 2.0 + 2.0 * ln_v) / 2048.0 <= tolerance,
      "{v} classes: the fp16 rounding bound must fit the allowance"
    );
    // ...and an all-zeros frame, off by ln V, does not.
    let zeros = vec![0.0f32; v];
    assert!(
      matches!(
        check_log_prob_normalization(&zeros, width(v), ComputeUnits::CpuOnly),
        Err(AlignError::UnnormalizedEmissions(_))
      ),
      "{v} classes: an all-zeros frame (logsumexp ln {v} = {ln_v}) must be refused"
    );
  }
  // A genuine frame of a wide head whose rounding outgrows the staged 29-class
  // allowance: 3,500 classes with a logsumexp of 0.5 — past `0.0293`, inside
  // the width's own allowance.
  let v = 3_500usize;
  let wide = vec![(0.5 - (v as f64).ln()) as f32; v];
  assert!(
    check_log_prob_normalization(&wide, width(v), ComputeUnits::CpuOnly).is_ok(),
    "a 3,500-class frame within its own allowance must pass"
  );
  assert!(0.5 > log_prob_sum_tolerance(WIDTH));
}

/// The guard frames its rows by the width it is HANDED — the model's head, read
/// at load — not by the bundled table's 29. Two uniform 4-class log-prob frames
/// (`-ln 4` each, `logsumexp = 0`) are normalized read four cells to a row; read
/// two to a row, every row is `[-ln 4, -ln 4]` with `logsumexp = -ln 2`, which is
/// no distribution. A guard that framed by a constant would pass or fail both
/// readings alike.
#[test]
fn check_log_prob_normalization_frames_rows_by_the_width_it_is_given() {
  let data = [-(4.0f32.ln()); 8];
  let four = NonZeroUsize::new(4).expect("nonzero");
  let two = NonZeroUsize::new(2).expect("nonzero");
  assert!(
    check_log_prob_normalization(&data, four, ComputeUnits::CpuOnly).is_ok(),
    "two 4-class frames of -ln 4 are normalized"
  );
  let Err(AlignError::UnnormalizedEmissions(e)) =
    check_log_prob_normalization(&data, two, ComputeUnits::CpuOnly)
  else {
    panic!("the same cells read as 2-class frames are not distributions");
  };
  assert!(
    (e.logsumexp() + 2.0f64.ln()).abs() < 1e-6,
    "a [-ln 4, -ln 4] row has logsumexp -ln 2, got {}",
    e.logsumexp()
  );
}

/// The wrap hands asry the width the encoder read, so a head of another width
/// than the bundled table's 29 wraps into emissions of THAT width — the width
/// asry's `finish` then checks against the seam's vocabulary.
#[test]
fn the_wrap_carries_the_width_the_encoder_read() {
  let four = NonZeroUsize::new(4).expect("nonzero");
  let emissions = RawEmissions {
    frames: 2,
    vocab_size: four,
    data: vec![-(4.0f32.ln()); 8],
  }
  .check_value_domain(None, ComputeUnits::CpuOnly)
  .expect("two normalized 4-class frames clear the guard")
  .into_emissions()
  .expect("and wrap as log-probabilities");
  assert_eq!(emissions.frames(), 2);
  assert_eq!(emissions.vocab(), four);
}

// ---------------------------------------------------------------------
// RawEmissions::check_value_domain: the band-then-normalization guard sequence
// `Encoder::emissions` actually mints through. Driving the MINTER (not the
// extracted `check_emission_value_domain` helper) binds BOTH predicates to the
// production door in one call — in particular the normalization half, which
// (unlike the band half, bound end-to-end by the model-gated
// `emissions_reject_an_ane_corrupted_matrix`) has no real-model fixture. If the
// minter's guard were ever handed `&[]` in place of its real tensor, or skipped
// normalization, an un-normalized matrix would sail through unnoticed; because
// this test feeds the minter the same un-normalized tensors the door would AND
// asserts the sealed buffer is the validated one, that regression is red right
// here.
// ---------------------------------------------------------------------

#[test]
fn raw_emissions_check_value_domain_binds_the_guard_and_the_minted_buffer() {
  // A shifted-raw-logit matrix — every cell finite, <= 0, and above the staged
  // band, so the band and `from_log_probs`'s <= 0 scan both miss it and only the
  // normalization step in the sequence rejects it.
  let mut shifted = Vec::with_capacity(4 * crate::audio::align::vocab::VOCAB_SIZE);
  for _ in 0..4 {
    for j in 0..crate::audio::align::vocab::VOCAB_SIZE {
      shifted
        .push(-10.0 - (j as f32) * (10.0 / (crate::audio::align::vocab::VOCAB_SIZE as f32 - 1.0)));
    }
  }
  let raw = RawEmissions {
    frames: 4,
    vocab_size: WIDTH,
    data: shifted,
  };
  assert!(
    matches!(
      raw.check_value_domain(STAGED_BAND, ComputeUnits::CpuOnly),
      Err(AlignError::UnnormalizedEmissions(_))
    ),
    "the minter must reject a shifted-raw-logit tensor as un-normalized"
  );

  // The all-zeros frame: exp(0) = 1 on every class, logsumexp = ln 29, again
  // above the band and <= 0 — only normalization catches it.
  let raw = RawEmissions {
    frames: 1,
    vocab_size: WIDTH,
    data: uniform_frame(0.0).to_vec(),
  };
  assert!(
    matches!(
      raw.check_value_domain(STAGED_BAND, ComputeUnits::CpuOnly),
      Err(AlignError::UnnormalizedEmissions(_))
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
    vocab_size: WIDTH,
    data: normalized.clone(),
  }
  .check_value_domain(STAGED_BAND, ComputeUnits::CpuOnly)
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
/// [`check_sentinel_band`] refuses the contract's band and
/// [`check_log_prob_normalization`] checks each frame is a distribution, but
/// neither enforces the per-cell `<= 0` ceiling. That half is
/// [`Emissions::from_log_probs`]'s own `finite ∧ <= 0` scan, run inside
/// [`ValueDomainChecked::into_emissions`] on the very tensor the guard sealed
/// (see [`check_sentinel_band`]'s "Deliberately only the band" note).
///
/// The distinguisher is a single frame `[0.001, -20.0 × 28]`:
///
/// - It clears [`check_sentinel_band`]: the minimum cell is `-20.0`, far above
///   the staged band's `-32768`.
/// - It clears [`check_log_prob_normalization`]:
///   `logsumexp = ln(e^0.001 + 28·e^-20) ≈ 0.001` (the 28 `-20.0` cells add
///   `≈ 5.8e-8`), well inside [`log_prob_sum_tolerance`] of 29 classes
///   (`0.0293`). So both guards pass and the token mints.
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

  // Neither guard rejects it: the band sees a min of -20.0 (above -32768), and
  // the frame's logsumexp is ≈ 0.001 (within the 29-class allowance, 0.0293).
  assert!(
    check_sentinel_band(&data, STAGED_BAND, ComputeUnits::CpuOnly).is_ok(),
    "min cell -20.0 is far above the staged band: the band cannot catch a positive cell"
  );
  assert!(
    check_log_prob_normalization(&data, WIDTH, ComputeUnits::CpuOnly).is_ok(),
    "logsumexp ≈ 0.001 is within the 29-class allowance: normalization cannot catch it"
  );

  // ...so the token mints through the real check sequence.
  let token = RawEmissions {
    frames: 1,
    vocab_size: WIDTH,
    data,
  }
  .check_value_domain(STAGED_BAND, ComputeUnits::CpuOnly)
  .expect("a frame that clears the band and the normalization guard must mint a token");

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
    || crate::tests::models_root().join("alignkit"),
    std::path::PathBuf::from,
  )
}

fn encoder_path() -> std::path::PathBuf {
  models_dir().join("base960h_aligner.mlmodelc")
}

/// Loads the real encoder model on [`DEFAULT_ENCODER_COMPUTE`] — the shipping
/// placement, the one an aligner's options default to.
/// Deliberately NOT a hardcoded `ComputeUnits::_`: every model-gated test
/// below is then a test OF the default.
fn load_encoder() -> Encoder {
  staged_encoder(DEFAULT_ENCODER_COMPUTE)
}

/// The staged model under its own contract, on `compute`: the road
/// [`crate::audio::align::Aligner::from_paths`] takes.
fn staged_encoder(compute: ComputeUnits) -> Encoder {
  Encoder::load(encoder_path(), &AcousticContract::BASE960H, compute)
    .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS to the model directory)")
}

/// `EncoderInput::from_samples` for the model-gated tests below, whose fixtures
/// are always within the window.
fn window_input(samples: &[f32]) -> EncoderInput<'_> {
  EncoderInput::from_samples(samples)
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn from_file_loads_and_reports_frame_count() {
  let encoder = load_encoder();
  // Ground truth pinned by
  // `tests/model_io.rs::base960h_aligner_io_matches_spec`: a 960,000-sample
  // window, 2,999 frames and a 29-class head — all three read at load now, and
  // the staged contract's geometry checked against the first two.
  assert_eq!(encoder.window_samples(), ENCODER_WINDOW_SAMPLES);
  assert_eq!(encoder.frames(), 2_999);
  assert_eq!(
    encoder.vocab_size().get(),
    crate::audio::align::vocab::VOCAB_SIZE
  );
  assert_eq!(encoder.contract(), &AcousticContract::BASE960H);
}

/// **A geometry the staged model's declaration contradicts is refused at
/// load.** Its declared window and frames, 960,000 and 2,999, are what a
/// 400-sample receptive field makes at a 320-sample stride. A contract stating a
/// 160-sample stride (5,998 frames) or a 321-sample one (2,990) is not this
/// model's front end, and the door refuses it by name before any chunk, naming
/// both counts. A 640-sample receptive field at 320 fits the same pair, and
/// loads: the declaration cannot tell the two apart, which is why the geometry
/// is the caller's to state.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn a_geometry_the_model_contradicts_is_refused_at_load() {
  for (stride, derived) in [(160u32, 5_998usize), (321, 2_990)] {
    let contract = contract(0, geometry(400, stride));
    let Err(AlignerError::FrameCountMismatch(mismatch)) =
      Encoder::load(encoder_path(), &contract, DEFAULT_ENCODER_COMPUTE)
    else {
      panic!("a {stride}-sample stride must be refused against the staged model's 2999 frames");
    };
    assert_eq!(
      (mismatch.window(), mismatch.declared(), mismatch.derived()),
      (ENCODER_WINDOW_SAMPLES, 2_999, derived)
    );
  }
  let wide = contract(0, geometry(640, 320));
  let encoder = Encoder::load(encoder_path(), &wide, DEFAULT_ENCODER_COMPUTE)
    .expect("a 640/320 geometry fits the staged declaration");
  assert_eq!(encoder.contract(), &wide);
}

/// The window check runs before any prediction, through the public door: a
/// buffer one sample past the staged model's window is refused naming both
/// lengths.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_refuse_a_buffer_longer_than_the_window_before_predicting() {
  let encoder = load_encoder();
  let too_long = vec![0.0f32; ENCODER_WINDOW_SAMPLES + 1];
  let Err(AlignError::InputTooLong(err)) = encoder.emissions(window_input(&too_long)) else {
    panic!("a buffer past the window must be refused");
  };
  assert_eq!(
    (err.got(), err.max()),
    (ENCODER_WINDOW_SAMPLES + 1, ENCODER_WINDOW_SAMPLES)
  );
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
/// The staged contract's [`SentinelBand::Fp16Saturation`] is not a tolerance to
/// be relaxed: it separates two populations three orders of magnitude apart
/// (worst legitimate log-prob measured anywhere on this model ≈ `-30.8`; the
/// sentinel ≈ `-45440`).
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
  let band = SentinelBand::Fp16Saturation;
  let sentinels = raw.data.iter().filter(|v| band.holds(**v)).count();
  assert_eq!(
    sentinels,
    0,
    "{sentinels} of {} emission cells are at or below {} (min = {min}) — the fp16 `log(0)` \
     sentinel. The encoder is on {:?}; an ANE placement corrupts this model's emissions and \
     cannot be used. See DEFAULT_ENCODER_COMPUTE.",
    raw.data.len(),
    band.ceiling(),
    DEFAULT_ENCODER_COMPUTE,
  );
}

/// **THE SILENT-CORRUPTION REGRESSION.** An ANE-corrupted emission matrix must
/// be REJECTED by the public door, not returned as a plausible `Ok`.
///
/// [`crate::audio::align::AlignerOptions::with_compute`] is public and accepts
/// `ComputeUnits::All`.
/// Before the value-domain guard existed, this exact call returned **`Ok`**: the
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
  let encoder = Encoder::load(
    encoder_path(),
    &AcousticContract::BASE960H,
    ComputeUnits::All,
  )
  .expect("load base960h_aligner.mlmodelc on ComputeUnits::All");
  let samples = load_jfk_wav();

  let Err(err) = encoder.emissions(window_input(&samples)) else {
    panic!(
      "an ANE-corrupted emission matrix was accepted. `Emissions::from_log_probs` cannot catch \
       this — -45440 is finite and <= 0 — so the caller now has plausible, silently wrong word \
       timings. The staged contract's sentinel band is the only thing standing here."
    );
  };
  let AlignError::CorruptEmissions(ref e) = err else {
    panic!("expected AlignError::CorruptEmissions, got {err:?}");
  };
  let (compute, min, cells, total) = (e.compute(), e.min(), e.cells(), e.total());
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
  assert_eq!(e.band(), SentinelBand::Fp16Saturation);
  assert!(
    e.band().holds(min),
    "reported min {min} must be in the band it tripped"
  );
  // Self-diagnosing: the message must NAME the placement, or the caller is
  // left to rediscover a 450×-slower, 16.7%-corrupt configuration by hand.
  let rendered = AlignError::CorruptEmissions(e.clone()).to_string();
  assert!(
    rendered.contains("All"),
    "error must name the placement: {rendered}"
  );
  println!("rejected with: {rendered}");
}

/// The guard keys on the emission VALUES, never on the placement — so a
/// non-default but numerically-clean placement must still be accepted.
///
/// `CpuAndGpu` is that placement: measured `min = -30.02`, zero cells in the
/// staged band on the same real speech the ANE corrupts. A guard that
/// rejected "any non-default compute" would fail here, and would also forbid a
/// future re-converted artifact that runs correctly on the ANE. This test is
/// what keeps the fix a value-domain check instead of a placement ban.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_accept_the_cpu_and_gpu_placement() {
  let encoder = Encoder::load(
    encoder_path(),
    &AcousticContract::BASE960H,
    ComputeUnits::CpuAndGpu,
  )
  .expect("load base960h_aligner.mlmodelc on ComputeUnits::CpuAndGpu");
  let samples = load_jfk_wav();

  let emissions = encoder
    .emissions(window_input(&samples))
    .expect("CpuAndGpu emissions are clean log-probs and must pass the band");
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
/// [`log_prob_sum_tolerance`] of its 29 classes.
///
/// This is the model side of the allowance: it re-measures, at gate time, the
/// worst `|logsumexp|` [`log_prob_sum_tolerance`]'s doc records (`CpuOnly`
/// `ted_60` 5.2485e-3, `jfk` 4.7453e-3; `CpuAndGpu` ~2.5e-7), so a future
/// artifact or firmware whose jitter crept toward the bound would fail here
/// rather than silently at a caller. It exercises the guarded door's exact check
/// pair (`check_sentinel_band` then `check_log_prob_normalization`) on the
/// truncated real tensor; the end-to-end public door on real speech is covered by
/// `emissions_accept_the_default_placement_on_real_speech` (jfk `CpuOnly`) and
/// `emissions_accept_the_cpu_and_gpu_placement` (jfk `CpuAndGpu`), which now run
/// the guard too, and by `tests/parity_words.rs` on `ted_60`.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_pass_the_normalization_guard_on_real_speech() {
  for compute in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndGpu] {
    let encoder = Encoder::load(encoder_path(), &AcousticContract::BASE960H, compute)
      .unwrap_or_else(|e| panic!("load base960h_aligner.mlmodelc on {compute:?}: {e}"));
    for (name, samples) in [("jfk", load_jfk_wav()), ("ted_60", load_ted_60_wav())] {
      let raw = encoder
        .emissions_raw(window_input(&samples))
        .unwrap_or_else(|e| panic!("{compute:?} {name}: emissions_raw: {e}"));
      // The exact guarded-door pair, on the exact truncated tensor the door checks.
      check_sentinel_band(&raw.data, STAGED_BAND, compute)
        .unwrap_or_else(|e| panic!("{compute:?} {name}: real emissions tripped the band: {e}"));
      check_log_prob_normalization(&raw.data, raw.vocab_size, compute).unwrap_or_else(|e| {
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
      let tolerance = log_prob_sum_tolerance(raw.vocab_size);
      println!(
        "{compute:?} {name}: {} frames, worst |logsumexp| = {worst:.6e} (allowance {tolerance:e})",
        raw.frames,
      );
      assert!(
        worst < tolerance,
        "{compute:?} {name}: worst |logsumexp| {worst} is not under the allowance {tolerance} — \
         the allowance's measured headroom has been lost"
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
    .join("tests/whisper/fixtures/audio/jfk.wav");
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
    .join("tests/whisper/fixtures/audio/ted_60.wav");
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
  assert_eq!(raw.frames, staged(48_000, encoder.frames()));
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

// ---------------------------------------------------------------------
// The door's own contract.
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`.
// What these drive is this door's `LoadContract` itself — its feature
// names, its element type, its geometry and its state clause — against
// descriptions built with the same fixture machinery, so a mis-stated
// contract is caught here and a mis-implemented checker is caught there.
// They replace the per-shape `check_waveform_contract` /
// `check_emissions_contract` gates, which could only see a shape the
// constructor remembered to hand them.
// ---------------------------------------------------------------------

use crate::{AxisRange, FeatureInfo, ModelDescription, model::RawShapeConstraint};

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis — which is what the staged
/// `base960h_aligner.mlmodelc` reports for both of its features.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  multi_array(name, shape, dtype, false, 2, vec![shape.to_vec()], shape)
}

/// One multi-array feature, spelled out: the constraint's raw type code, its
/// enumerated shapes, and the axes its per-axis ranges pin.
fn multi_array(
  name: &str,
  shape: &[usize],
  dtype: DataType,
  optional: bool,
  raw_type: isize,
  enumerated: Vec<Vec<usize>>,
  pinned: &[usize],
) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    optional,
    Some(RawShapeConstraint::new(
      raw_type,
      enumerated,
      pinned.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// The staged aligner's description, as the CoreML probe reads it back:
/// `waveform [1, 960000]` f32 in, `emissions [1, 2999, 29]` f32 out, no state.
fn aligner_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(
      names::WAVEFORM,
      &[1, ENCODER_WINDOW_SAMPLES],
      DataType::F32,
    )],
    vec![fixed(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
    )],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into this
/// module's errors — exactly what `Encoder::load` does after
/// `Model::load`, before it reads the declaration back.
fn check(description: &ModelDescription) -> Result<(), AlignerError> {
  crate::model::contract::check_load_contract(description, &align_contract())
    .map_err(contract_violation)
}

/// The contract, then the staged contract's geometry against what the
/// description declares: the whole load-time judgement of a graph.
fn check_staged(description: &ModelDescription) -> Result<Declared, AlignerError> {
  check(description)?;
  let declared = declared(description);
  check_frame_count(
    AcousticContract::BASE960H.geometry(),
    declared.window,
    declared.frames,
  )?;
  Ok(declared)
}

/// The contract and the staged geometry accept exactly what the staged artifact
/// declares, and read it back.
#[test]
fn the_contract_accepts_the_staged_geometry() {
  let declared = check_staged(&aligner_description()).expect("the staged declaration loads");
  assert_eq!(declared.window.get(), ENCODER_WINDOW_SAMPLES);
  assert_eq!(declared.frames.get(), EXPECTED_OUTPUT_FRAMES);
  assert_eq!(
    declared.vocab_size.get(),
    crate::audio::align::vocab::VOCAB_SIZE
  );
}

/// **The clause the module's "fixed in every dimension" sentence was missing.**
/// The old code never consulted the `waveform` input's shape CONSTRAINT — only
/// its declared shape, which [`crate::FeatureInfo::shape`] reports identically
/// for a `RangeDims` graph converted at `[1, 960000]`. Such a graph loaded, and
/// its variable window is exactly what the fixed-window bridging assumes away.
#[test]
fn the_contract_refuses_a_flexible_waveform_declaring_its_exact_numbers() {
  let description = ModelDescription::from_parts(
    vec![multi_array(
      names::WAVEFORM,
      &[1, ENCODER_WINDOW_SAMPLES],
      DataType::F32,
      false,
      3,
      Vec::new(),
      &[1, ENCODER_WINDOW_SAMPLES],
    )],
    vec![fixed(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, AlignerError::ContractMismatch(m) if m.feature() == names::WAVEFORM),
    "{err}"
  );
}

/// The same clause on the output side: a flexible `emissions` declaring 2999
/// frames is a graph whose frame count is a default, not a guarantee — and the
/// truncation formula sizes its destination from that count.
#[test]
fn the_contract_refuses_a_flexible_emissions_declaring_its_exact_numbers() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::WAVEFORM,
      &[1, ENCODER_WINDOW_SAMPLES],
      DataType::F32,
    )],
    vec![multi_array(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
      false,
      3,
      Vec::new(),
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, AlignerError::ContractMismatch(m) if m.feature() == names::EMISSIONS),
    "{err}"
  );
}

/// A missing `waveform` names the feature it cannot find. The old code carried
/// a hand-written `expected` string for this branch, kept in sync with the
/// check's own literal by a test; the contract has one statement of the
/// geometry and nothing to keep in sync.
#[test]
fn the_contract_refuses_a_missing_waveform() {
  let description = ModelDescription::from_parts(
    vec![fixed("audio", &[1, ENCODER_WINDOW_SAMPLES], DataType::F32)],
    vec![fixed(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, AlignerError::ContractMismatch(m)
      if m.feature() == names::WAVEFORM && m.actual() == "missing"),
    "{err}"
  );
}

/// A wrong dtype on either feature, and an empty axis — a zero window, zero
/// frames, a zero-width head, the one head width no vocabulary can pair with —
/// are the contract's to refuse. Each is a `ContractMismatch` naming the
/// feature it is about.
#[test]
fn the_contract_refuses_a_wrong_dtype_or_an_empty_axis() {
  const VOCAB: usize = crate::audio::align::vocab::VOCAB_SIZE;
  let staged_emissions = || {
    fixed(
      names::EMISSIONS,
      &[1, EXPECTED_OUTPUT_FRAMES, VOCAB],
      DataType::F32,
    )
  };
  let staged_waveform = || fixed(names::WAVEFORM, &[1, ENCODER_WINDOW_SAMPLES], DataType::F32);
  let cases: [(FeatureInfo, FeatureInfo, &str); 5] = [
    (
      fixed(names::WAVEFORM, &[1, ENCODER_WINDOW_SAMPLES], DataType::F16),
      staged_emissions(),
      names::WAVEFORM,
    ),
    (
      staged_waveform(),
      fixed(
        names::EMISSIONS,
        &[1, EXPECTED_OUTPUT_FRAMES, VOCAB],
        DataType::F16,
      ),
      names::EMISSIONS,
    ),
    (
      fixed(names::WAVEFORM, &[1, 0], DataType::F32),
      staged_emissions(),
      names::WAVEFORM,
    ),
    (
      staged_waveform(),
      fixed(names::EMISSIONS, &[1, 0, VOCAB], DataType::F32),
      names::EMISSIONS,
    ),
    (
      staged_waveform(),
      fixed(
        names::EMISSIONS,
        &[1, EXPECTED_OUTPUT_FRAMES, 0],
        DataType::F32,
      ),
      names::EMISSIONS,
    ),
  ];
  for (waveform, emissions, feature) in cases {
    let description = ModelDescription::from_parts(vec![waveform], vec![emissions], Vec::new());
    let err = check(&description).unwrap_err();
    assert!(
      matches!(&err, AlignerError::ContractMismatch(m) if m.feature() == feature),
      "expected a {feature} mismatch, got {err}"
    );
  }
}

/// **A window or a frame count the staged geometry does not make is refused by
/// name.** The two frame counts the old `>= 1` check waved through — 2998
/// (drops the last acoustic frame) and 3000 — and the staged 2999 frames on a
/// 30 s window pass the contract, which fixes each axis and not how they
/// relate, and are refused by the geometry check the load runs next: a
/// `FrameCountMismatch` naming the window and both counts.
#[test]
fn the_staged_geometry_refuses_a_window_or_frame_count_it_does_not_make() {
  const VOCAB: usize = crate::audio::align::vocab::VOCAB_SIZE;
  for (window, frames, derived) in [
    (ENCODER_WINDOW_SAMPLES, 2_998, 2_999),
    (ENCODER_WINDOW_SAMPLES, 3_000, 2_999),
    (480_000, EXPECTED_OUTPUT_FRAMES, 1_499),
  ] {
    let description = ModelDescription::from_parts(
      vec![fixed(names::WAVEFORM, &[1, window], DataType::F32)],
      vec![fixed(names::EMISSIONS, &[1, frames, VOCAB], DataType::F32)],
      Vec::new(),
    );
    assert!(check(&description).is_ok(), "the contract fixes each axis");
    let Err(AlignerError::FrameCountMismatch(mismatch)) = check_staged(&description) else {
      panic!("{frames} frames of a {window}-sample window must be refused");
    };
    assert_eq!(
      (mismatch.window(), mismatch.declared(), mismatch.derived()),
      (window, frames, derived)
    );
  }
}

/// **The window, the frame count and the head width are the model's, and they
/// are READ.** A 29-class head (the staged `base960h`), HuggingFace's 32-class
/// `wav2vec2-base-960h` head, and a one-class and a 64-class head all satisfy
/// the contract, and each reads back as exactly the width it declares — the
/// number the aligner then pairs with the vocabulary that ships beside the
/// model. The same holds for the window: a 30 s conversion, `[1, 480000]` in and
/// the 1499 frames wav2vec2's front end makes of it out, loads under the staged
/// geometry and reads back its own window and frame count. The contract used to
/// pin 960,000, 2999 and 29, which refused every model converted at another
/// window, or spelling another alphabet, before its own contract and vocabulary
/// could be consulted.
#[test]
fn the_contract_reads_the_window_frames_and_head_width_back() {
  for (window, frames, width) in [
    (ENCODER_WINDOW_SAMPLES, EXPECTED_OUTPUT_FRAMES, 29),
    (ENCODER_WINDOW_SAMPLES, EXPECTED_OUTPUT_FRAMES, 32),
    (ENCODER_WINDOW_SAMPLES, EXPECTED_OUTPUT_FRAMES, 1),
    (ENCODER_WINDOW_SAMPLES, EXPECTED_OUTPUT_FRAMES, 64),
    (480_000, 1_499, 29),
  ] {
    let description = ModelDescription::from_parts(
      vec![fixed(names::WAVEFORM, &[1, window], DataType::F32)],
      vec![fixed(names::EMISSIONS, &[1, frames, width], DataType::F32)],
      Vec::new(),
    );
    let declared = check_staged(&description).unwrap_or_else(|err| {
      panic!("a {window}-sample window of {frames} frames and {width} classes loads: {err}")
    });
    assert_eq!(
      (
        declared.window.get(),
        declared.frames.get(),
        declared.vocab_size.get()
      ),
      (window, frames, width)
    );
  }
}

/// **A window under asry's 400-sample pad is refused at load.** asry's
/// `prepare` pads every chunk shorter than 400 samples up to 400, so every
/// chunk that reaches the encoder is at least that long: a model declaring a
/// 100-sample window (one frame of a 100-sample receptive field at a 301-sample
/// stride, which its own geometry check would pass) could align nothing, and
/// is refused by the load contract on `waveform`. A 400-sample window loads.
///
/// Mutation check: stating the window `Dim::AnyFixed` again lets the 100-sample
/// window through, and this test fails.
#[test]
fn a_window_under_asrys_pad_is_refused_at_load() {
  const VOCAB: usize = crate::audio::align::vocab::VOCAB_SIZE;
  let with_window = |window: usize| {
    ModelDescription::from_parts(
      vec![fixed(names::WAVEFORM, &[1, window], DataType::F32)],
      vec![fixed(names::EMISSIONS, &[1, 1, VOCAB], DataType::F32)],
      Vec::new(),
    )
  };
  let err = check(&with_window(100)).unwrap_err();
  assert!(
    matches!(&err, AlignerError::ContractMismatch(m) if m.feature() == names::WAVEFORM),
    "{err}"
  );
  assert!(check(&with_window(399)).is_err());
  assert!(check(&with_window(400)).is_ok());
}

/// **The V=5000 raw -4 row is refused: a log-probability head that wide is
/// refused at load.** fp16 rounding over 5,000 classes can move a genuine
/// frame's logsumexp by up to `log_prob_sum_tolerance(5000)` = 4.88, and a
/// frame of `-4` in every column — probability mass about 92, logsumexp 4.52 —
/// sits inside that. No allowance separates the two at that width, so the
/// contract's `LogProbabilities` statement is refused for it by name, and only
/// `Logits` (normalized by asry) loads. The check is exact at its edge: 353
/// classes are separated, 354 are not.
///
/// Mutation check: widening `MAX_LOG_PROB_WIDTH` past 5,000, or deleting the
/// check, lets the 5,000-class log-probability head load, and this test fails.
#[test]
fn a_log_probability_head_too_wide_to_check_is_refused_at_load() {
  let width = |v: usize| NonZeroUsize::new(v).expect("nonzero");
  let v = 5_000usize;
  let raw_row = vec![-4.0f32; v];
  let lse = -4.0 + (v as f64).ln();
  assert!(
    lse < log_prob_sum_tolerance(width(v)),
    "the reason: at 5,000 classes the allowance ({}) admits the raw -4 row (logsumexp {lse})",
    log_prob_sum_tolerance(width(v))
  );
  let Err(AlignerError::UnprovableNormalization(refused)) =
    check_output_width(OutputKind::LogProbabilities, width(v))
  else {
    panic!("a 5,000-class log-probability head must be refused at load");
  };
  assert_eq!(
    (refused.vocab_size(), refused.widest()),
    (v, MAX_LOG_PROB_WIDTH)
  );
  assert_eq!(check_output_width(OutputKind::Logits, width(v)), Ok(()));

  assert_eq!(MAX_LOG_PROB_WIDTH, 353);
  assert_eq!(
    check_output_width(OutputKind::LogProbabilities, width(353)),
    Ok(())
  );
  assert!(check_output_width(OutputKind::LogProbabilities, width(354)).is_err());

  // Stated as logits, the same row is what asry normalizes: a uniform
  // distribution over its 5,000 classes.
  let emissions = RawEmissions {
    frames: 1,
    vocab_size: width(v),
    data: raw_row,
  }
  .into_logit_emissions()
  .expect("logits are normalized, not checked");
  assert_eq!(emissions.vocab(), width(v));
}

/// At the widest checkable head the allowance stays separated from an
/// unnormalized frame: a frame of 353 equal cells whose probabilities sum to
/// two (logsumexp `ln 2`) is refused, and one within the allowance passes.
#[test]
fn the_widest_checkable_head_still_refuses_a_frame_off_by_a_factor_of_two() {
  let v = MAX_LOG_PROB_WIDTH;
  let width = NonZeroUsize::new(v).expect("nonzero");
  let ln_v = (v as f64).ln();
  let doubled = vec![(UNNORMALIZED_LOGSUMEXP - ln_v) as f32; v];
  assert!(matches!(
    check_log_prob_normalization(&doubled, width, ComputeUnits::CpuOnly),
    Err(AlignError::UnnormalizedEmissions(_))
  ));
  let within = vec![((log_prob_sum_tolerance(width) / 2.0) - ln_v) as f32; v];
  assert!(check_log_prob_normalization(&within, width, ComputeUnits::CpuOnly).is_ok());
}

/// **A graph carrying `waveform` plus another REQUIRED input** clears every
/// per-feature clause and then fails on every prediction, because
/// [`Encoder::emissions`] supplies `waveform` and nothing else.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::WAVEFORM, &[1, ENCODER_WINDOW_SAMPLES], DataType::F32),
      fixed(
        "attention_mask",
        &[1, ENCODER_WINDOW_SAMPLES],
        DataType::I32,
      ),
    ],
    vec![fixed(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(AlignerError::UnsatisfiableInput(ref name))
      if name == "attention_mask"),
    "{:?}",
    check(&description)
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail.
#[test]
fn the_contract_accepts_an_extra_optional_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::WAVEFORM, &[1, ENCODER_WINDOW_SAMPLES], DataType::F32),
      multi_array(
        "attention_mask",
        &[1, ENCODER_WINDOW_SAMPLES],
        DataType::I32,
        true,
        2,
        vec![vec![1, ENCODER_WINDOW_SAMPLES]],
        &[1, ENCODER_WINDOW_SAMPLES],
      ),
    ],
    vec![fixed(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(check(&description).is_ok());
}

/// An output the door READS that the graph may leave out: every geometry
/// clause passes and the prediction is still free to omit it.
#[test]
fn the_contract_refuses_an_optional_emissions_output() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::WAVEFORM,
      &[1, ENCODER_WINDOW_SAMPLES],
      DataType::F32,
    )],
    vec![multi_array(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
      true,
      2,
      vec![vec![
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ]],
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, AlignerError::ContractMismatch(m) if m.feature() == names::EMISSIONS),
    "{err}"
  );
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly `waveform` and `emissions` plus a state clears every per-feature
/// clause AND the input set — and only then meets [`Encoder::emissions`],
/// which predicts through the STATELESS API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::WAVEFORM,
      &[1, ENCODER_WINDOW_SAMPLES],
      DataType::F32,
    )],
    vec![fixed(
      names::EMISSIONS,
      &[
        1,
        EXPECTED_OUTPUT_FRAMES,
        crate::audio::align::vocab::VOCAB_SIZE,
      ],
      DataType::F32,
    )],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  assert!(
    matches!(check(&description), Err(AlignerError::UnsatisfiableState(ref name))
      if name == "kv_cache")
  );
}

// ---------------------------------------------------------------------
// The one gate here that loads a real artifact.
// ---------------------------------------------------------------------

/// **This door's `Checked::new` call site, pinned on a REAL model, in every
/// `cargo test`.**
///
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is COMMITTED, so
/// unlike everything else in this repository that loads a model this needs no
/// staged artifact and carries no `#[ignore]`. Silero is a real, fixed-shape,
/// six-feature CoreML graph that is simply not this door's model — the exact
/// shape of a mis-pointed `path`.
#[test]
fn the_align_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; \
     looked for {}",
    bundle.display()
  );

  let model = Model::load(&bundle, ComputeUnits::CpuOnly).expect("the committed bundle loads");
  assert!(
    model.description().input(names::WAVEFORM).is_none(),
    "silero declares no `waveform`, which is what makes it this gate's model"
  );

  let violation = Checked::new(model, &align_contract())
    .expect_err("silero does not satisfy the aligner contract");
  assert!(
    matches!(&violation, ContractViolation::Missing(m) if m.feature() == names::WAVEFORM),
    "expected `waveform` missing, got {violation}"
  );
}
