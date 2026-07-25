use super::*;

#[test]
fn options_default_equals_new() {
  assert_eq!(AudioEncoderOptions::default(), AudioEncoderOptions::new());
  assert_eq!(AudioEncoderOptions::new().compute(), DEFAULT_AUDIO_COMPUTE);
  assert_eq!(DEFAULT_AUDIO_COMPUTE, ComputeUnits::All);
}

#[test]
fn options_with_and_set_compute() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);

  let mut opts = AudioEncoderOptions::new();
  opts.set_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndGpu);
}

#[test]
fn first_non_finite_finds_offenders() {
  assert_eq!(first_non_finite(&[0.0, 1.0, 2.0]), None);
  assert_eq!(first_non_finite(&[0.0, f32::NAN, 2.0]), Some(1));
  assert_eq!(first_non_finite(&[f32::INFINITY]), Some(0));
  assert_eq!(first_non_finite(&[1.0, 2.0, f32::NEG_INFINITY]), Some(2));
  // Subnormals and signed zeros are finite.
  assert_eq!(
    first_non_finite(&[0.0, -0.0, f32::MIN_POSITIVE / 2.0]),
    None
  );
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(
    describe(&[1, 1, 1001, 64], Some(DataType::F32)),
    "[1, 1, 1001, 64] float32"
  );
  assert_eq!(describe(&[1, 512], None), "[1, 512] none");
}

/// `embed_window` accepts `1..=TARGET_SAMPLES` and rejects an over-length clip
/// with [`Error::AudioTooLong`] (naming `embed_windows`) instead of silently
/// head-truncating it. Gated at the `check_window_len` seam so it needs no model.
///
/// Mutation tripwire: relaxing the bound (`>` → `>=`, or `TARGET_SAMPLES` →
/// `TARGET_SAMPLES + 1`) makes the over-length case pass, and dropping the guard
/// re-admits the silent-truncation defect.
#[test]
fn check_window_len_rejects_over_length_only() {
  // The exact window and anything shorter are accepted.
  assert!(check_window_len(TARGET_SAMPLES).is_ok());
  assert!(check_window_len(TARGET_SAMPLES - 1).is_ok());
  assert!(check_window_len(1).is_ok());
  // One sample past the window is rejected, and the error carries len + limit and
  // points the caller at the long-audio path.
  let err = check_window_len(TARGET_SAMPLES + 1).unwrap_err();
  let msg = err.to_string();
  assert!(
    matches!(err, Error::AudioTooLong { len, max } if len == TARGET_SAMPLES + 1 && max == TARGET_SAMPLES),
    "expected AudioTooLong{{ len: {}, max: {TARGET_SAMPLES} }}, got {err:?}",
    TARGET_SAMPLES + 1
  );
  assert!(
    msg.contains("embed_windows"),
    "AudioTooLong should name the long-audio path: {msg}"
  );
}

/// The codex [high] at-cap geometry: with `hop = 5`, a 500 000-sample (~2 MiB)
/// clip plans EXACTLY [`crate::embeddings::clap::window::DEFAULT_MAX_WINDOWS`]
/// (100 000) spans and is ADMITTED by the O(1) cap — `planned == max`, not
/// `> max`. `embed_windows` then reserves one ~2 KiB [`WindowEmbedding`] per span
/// (~207 MiB), which the fix does FALLIBLY (`try_reserve_exact` →
/// [`Error::Windowing`]`(`[`WinditError::AllocFailed`]`)`) instead of the prior
/// infallible `Vec::with_capacity`, so an allocator refusal on a small at-cap
/// clip is a typed error rather than a process abort.
///
/// This is the achievable seam assertion (mirroring `check_window_len_*`, which
/// tests `embed_window`'s guard without a model): it pins that `spans()` admits
/// the exact-cap plan the caller now reserves for. Asserting the actual typed
/// `AllocFailed` would require injecting an allocator failure — impractical and
/// unavailable here (there is no fault-injection allocator, and ~207 MiB is
/// ordinarily allocatable, so no real OOM fires) — and `embed_windows` itself
/// needs a loaded model for the per-span loop that follows the reservation. The
/// `try_reserve_exact` call is the structural guarantee; this test pins the
/// geometry that reaches it.
///
/// Mutation tripwire: reverting the reservation to `with_capacity` restores the
/// process-abort path for exactly this admitted-at-cap input.
#[test]
fn at_cap_plan_is_admitted_and_reserved_fallibly() {
  let plan = WindowPlan::new().with_hop_samples(5);
  // Default cap is on; the exact-cap geometry sits AT it (not over).
  assert_eq!(plan.max_windows(), 100_000);
  let spans = plan
    .spans(500_000)
    .expect("at-cap plan must be admitted, not refused");
  assert_eq!(
    spans.len(),
    100_000,
    "hop=5 over 500_000 samples plans exactly 100_000 spans"
  );
  // EXACTLY at the cap — the boundary the caller's fallible reservation covers.
  assert_eq!(spans.len() as u32, plan.max_windows());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_gpu"), "serialized as as_str: {json}");
  let back: AudioEncoderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}
