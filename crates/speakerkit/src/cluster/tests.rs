use super::*;

/// A minimal, self-consistent dia [`OfflineInput`](dia::offline::OfflineInput)
/// over caller-owned buffers (num_chunks=1, num_speakers=3, one frame, one
/// output frame). Its data tensors are irrelevant to the default/`apply_to`
/// pins — only its (dia-default) hyperparameters and passthrough matter — so
/// the caller owns trivial zero buffers and this wires them into dia's
/// constructor with the community-1 sliding windows.
fn base_offline_input<'a>(
  raw: &'a [f32],
  segs: &'a [f64],
  count: &'a [u8],
  plda: &'a dia::plda::PldaTransform,
) -> dia::offline::OfflineInput<'a> {
  dia::offline::OfflineInput::new(
    raw,
    1,
    3,
    segs,
    1,
    count,
    1,
    dia::reconstruct::SlidingWindow::new(0.0, 10.0, 1.0),
    dia::reconstruct::SlidingWindow::new(0.0, 0.0619375, 0.016875),
    plda,
  )
}

// =====================================================================
// Golden-enum contract — driven by the SINGLE table roster
// (CLUSTER_BACKEND_SPELLINGS), so a new variant extends every assertion
// automatically. Mirrors coremlit::units::tests and whisperkit's Task tests.
// =====================================================================

#[test]
fn as_str_round_trips_from_str_for_every_spelling() {
  for &sp in CLUSTER_BACKEND_SPELLINGS {
    let parsed: ClusterBackend = sp.parse().expect("table spelling parses");
    assert_eq!(
      parsed.as_str(),
      sp,
      "FromStr → as_str must round-trip the discriminant"
    );
  }
}

#[test]
fn display_matches_as_str_for_every_spelling() {
  for &sp in CLUSTER_BACKEND_SPELLINGS {
    let parsed: ClusterBackend = sp.parse().expect("table spelling parses");
    assert_eq!(
      parsed.to_string(),
      sp,
      "Display must equal as_str (derive_more delegation)"
    );
  }
}

#[test]
fn spellings_roster_is_the_known_set() {
  // Pins the current roster concretely. Adding a variant to the macro table
  // updates this AND the loops above together.
  assert_eq!(CLUSTER_BACKEND_SPELLINGS, &["offline", "online"]);
}

#[test]
fn offline_spelling_maps_to_default_payload() {
  // FromStr selects an engine with a DEFAULT payload — the string form tunes
  // nothing.
  assert_eq!(
    "offline".parse::<ClusterBackend>().unwrap(),
    ClusterBackend::Offline(OfflineOptions::new())
  );
}

#[test]
fn online_spelling_maps_to_default_payload() {
  // Same for the online engine: the discriminant selects it, the payload
  // defaults.
  assert_eq!(
    "online".parse::<ClusterBackend>().unwrap(),
    ClusterBackend::Online(OnlineOptions::new())
  );
}

#[test]
fn default_is_still_offline_not_online() {
  // Adding the Online variant must NOT change the default backend (every DER
  // gate drives the default; a flipped default would silently reroute them).
  assert_eq!(
    ClusterBackend::default(),
    ClusterBackend::Offline(OfflineOptions::new())
  );
  assert_ne!(
    ClusterBackend::default(),
    ClusterBackend::Online(OnlineOptions::new())
  );
}

#[test]
fn unknown_name_is_opaque_error() {
  // Both real spellings parse (covered by the roster loops +
  // `*_spelling_maps_to_default_payload`); empty, arbitrary, and wrong-case
  // names fail.
  assert!("".parse::<ClusterBackend>().is_err());
  assert!("Offline".parse::<ClusterBackend>().is_err()); // case-sensitive
  assert!("Online".parse::<ClusterBackend>().is_err()); // case-sensitive
  // The error is opaque: constructed only by the parser, comparable, no payload.
  assert_eq!(
    "nope".parse::<ClusterBackend>().unwrap_err(),
    "also-nope".parse::<ClusterBackend>().unwrap_err()
  );
}

// =====================================================================
// ClusterBackend default + Copy
// =====================================================================

#[test]
fn default_is_offline_with_default_options() {
  assert_eq!(
    ClusterBackend::default(),
    ClusterBackend::Offline(OfflineOptions::new())
  );
}

#[test]
fn cluster_backend_is_copy() {
  // Copy (matches the crate's options-type convention); a use-after-move
  // compiles only because the value is copied.
  let a = ClusterBackend::default();
  let b = a;
  assert_eq!(a, b);
}

// =====================================================================
// OfflineOptions defaults — pinned BOTH to the literal consts AND to dia's
// OWN OfflineInput accessors, so a drift on EITHER side fails.
// =====================================================================

#[test]
fn offline_options_new_matches_default() {
  assert_eq!(OfflineOptions::new(), OfflineOptions::default());
}

#[test]
fn offline_options_defaults_match_const_literals() {
  let o = OfflineOptions::new();
  assert_eq!(o.threshold(), DEFAULT_THRESHOLD);
  assert_eq!(o.fa(), DEFAULT_FA);
  assert_eq!(o.fb(), DEFAULT_FB);
  assert_eq!(o.max_iters(), DEFAULT_MAX_ITERS);
  assert_eq!(o.min_duration_off(), DEFAULT_MIN_DURATION_OFF);
  // The concrete community-1 values, pinned so a mutation to any default fails
  // here as well as at the dia cross-check below.
  assert_eq!(o.threshold(), 0.6);
  assert_eq!(o.fa(), 0.07);
  assert_eq!(o.fb(), 0.8);
  assert_eq!(o.max_iters(), 20);
  assert_eq!(o.min_duration_off(), 0.0);
}

#[test]
fn defaults_equal_dia() {
  // The load-bearing pin: speakerkit's OfflineOptions defaults MUST equal dia's
  // OfflineInput defaults (which are pyannote-community-1's). Read dia's own
  // defaults off a freshly-constructed OfflineInput and compare. A drift on
  // EITHER side — speakerkit's const or dia's `OfflineInput::new` — fails this.
  let plda = dia::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let raw = vec![0.0f32; crate::embed::EMBEDDING_DIM * 3];
  let segs = vec![0.0f64; 3];
  let count = vec![0u8; 1];
  let input = base_offline_input(&raw, &segs, &count, &plda);

  let o = OfflineOptions::default();
  assert_eq!(
    o.threshold(),
    input.threshold(),
    "threshold drifted from dia"
  );
  assert_eq!(o.fa(), input.fa(), "fa drifted from dia");
  assert_eq!(o.fb(), input.fb(), "fb drifted from dia");
  assert_eq!(
    o.max_iters(),
    input.max_iters(),
    "max_iters drifted from dia"
  );
  assert_eq!(
    o.min_duration_off(),
    input.min_duration_off(),
    "min_duration_off drifted from dia"
  );
}

// =====================================================================
// OfflineOptions builders / setters (rust-options-pattern)
// =====================================================================

#[test]
fn offline_options_builders_and_setters() {
  let o = OfflineOptions::new()
    .with_threshold(0.33)
    .with_fa(0.11)
    .with_fb(0.77)
    .with_max_iters(9)
    .with_min_duration_off(0.5);
  assert_eq!(o.threshold(), 0.33);
  assert_eq!(o.fa(), 0.11);
  assert_eq!(o.fb(), 0.77);
  assert_eq!(o.max_iters(), 9);
  assert_eq!(o.min_duration_off(), 0.5);

  let mut m = OfflineOptions::new();
  m.set_threshold(0.1);
  m.set_fa(0.2);
  m.set_fb(0.3);
  m.set_max_iters(4);
  m.set_min_duration_off(0.05);
  assert_eq!(m.threshold(), 0.1);
  assert_eq!(m.fa(), 0.2);
  assert_eq!(m.fb(), 0.3);
  assert_eq!(m.max_iters(), 4);
  assert_eq!(m.min_duration_off(), 0.05);
}

#[test]
fn threshold_fa_fb_builders_accept_non_finite_like_dia() {
  // dia's OfflineInput::with_threshold/with_fa/with_fb range-check NOTHING;
  // OfflineOptions mirrors that (only the serde boundary rejects non-finite for
  // these three). This pins the deliberate asymmetry with min_duration_off — if
  // a guard is ever added to one of these builders, it diverges from dia and
  // this fails.
  let o = OfflineOptions::new()
    .with_threshold(f64::NAN)
    .with_fa(f64::INFINITY)
    .with_fb(f64::NEG_INFINITY);
  assert!(o.threshold().is_nan());
  assert_eq!(o.fa(), f64::INFINITY);
  assert_eq!(o.fb(), f64::NEG_INFINITY);
}

#[test]
#[should_panic(expected = "min_duration_off must be finite and >= 0")]
fn with_min_duration_off_panics_on_negative() {
  let _ = OfflineOptions::new().with_min_duration_off(-1.0);
}

#[test]
#[should_panic(expected = "min_duration_off must be finite and >= 0")]
fn with_min_duration_off_panics_on_nan() {
  let _ = OfflineOptions::new().with_min_duration_off(f64::NAN);
}

#[test]
#[should_panic(expected = "min_duration_off must be finite and >= 0")]
fn with_min_duration_off_panics_on_positive_infinity() {
  let _ = OfflineOptions::new().with_min_duration_off(f64::INFINITY);
}

#[test]
fn with_min_duration_off_accepts_zero_and_positive() {
  // The boundary value 0.0 (the default) and positive finites are valid.
  assert_eq!(
    OfflineOptions::new()
      .with_min_duration_off(0.0)
      .min_duration_off(),
    0.0
  );
  assert_eq!(
    OfflineOptions::new()
      .with_min_duration_off(2.5)
      .min_duration_off(),
    2.5
  );
}

// =====================================================================
// apply_to — the single OfflineOptions → dia OfflineInput mapping. Hermetic,
// ort-free (dia's PLDA weights are compile-time embedded).
// =====================================================================

#[test]
fn apply_to_maps_each_knob_to_its_dia_field() {
  let plda = dia::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let raw = vec![0.0f32; crate::embed::EMBEDDING_DIM * 3];
  let segs = vec![0.0f64; 3];
  let count = vec![0u8; 1];
  let base = base_offline_input(&raw, &segs, &count, &plda);

  // Five distinct non-default values, so a swapped mapping (e.g. fa↔fb) fails.
  let opts = OfflineOptions::new()
    .with_threshold(0.31)
    .with_fa(0.12)
    .with_fb(0.73)
    .with_max_iters(7)
    .with_min_duration_off(0.4);
  let input = opts.apply_to(base);
  assert_eq!(input.threshold(), 0.31);
  assert_eq!(input.fa(), 0.12);
  assert_eq!(input.fb(), 0.73);
  assert_eq!(input.max_iters(), 7);
  assert_eq!(input.min_duration_off(), 0.4);
  // The knobs never touch the data tensors: they pass through unchanged.
  assert_eq!(input.raw_embeddings(), raw.as_slice());
  assert_eq!(input.count(), count.as_slice());
}

#[test]
fn apply_to_default_is_a_no_op_over_dia_defaults() {
  // Applying the DEFAULT OfflineOptions re-writes each field with dia's own
  // default value, leaving the input's hyperparameters unchanged — the property
  // `Extraction::diarize` relies on for byte-identical default clustering.
  let plda = dia::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let raw = vec![0.0f32; crate::embed::EMBEDDING_DIM * 3];
  let segs = vec![0.0f64; 3];
  let count = vec![0u8; 1];
  let base = base_offline_input(&raw, &segs, &count, &plda);
  // Capture the input's own (dia-default) hyperparameters before apply_to
  // consumes it.
  let (t, fa, fb, mi, md) = (
    base.threshold(),
    base.fa(),
    base.fb(),
    base.max_iters(),
    base.min_duration_off(),
  );

  let out = OfflineOptions::default().apply_to(base);
  assert_eq!(out.threshold(), t);
  assert_eq!(out.fa(), fa);
  assert_eq!(out.fb(), fb);
  assert_eq!(out.max_iters(), mi);
  assert_eq!(out.min_duration_off(), md);
}

// =====================================================================
// serde — discriminant tag, omitted-field defaults, no silent flip, and
// non-finite rejection at the boundary (whisperkit round-3 F6).
// =====================================================================

#[cfg(feature = "serde")]
#[test]
fn serde_discriminant_tag_equals_as_str_for_every_spelling() {
  // The externally-tagged single key IS the discriminant; it must equal
  // as_str, so serde `rename_all` and the table spelling cannot silently drift.
  for &sp in CLUSTER_BACKEND_SPELLINGS {
    let backend: ClusterBackend = sp.parse().unwrap();
    let value = serde_json::to_value(backend).unwrap();
    let obj = value
      .as_object()
      .expect("externally-tagged enum is an object");
    assert_eq!(obj.len(), 1, "exactly one discriminant key");
    assert_eq!(
      obj.keys().next().unwrap(),
      sp,
      "serde tag must equal as_str"
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn serde_offline_empty_payload_is_full_defaults() {
  let b: ClusterBackend = serde_json::from_str(r#"{"offline":{}}"#).unwrap();
  assert_eq!(b, ClusterBackend::Offline(OfflineOptions::new()));
}

#[cfg(feature = "serde")]
#[test]
fn serde_offline_partial_payload_defaults_other_knobs() {
  // Only threshold given: the other four knobs default individually (per-field
  // serde defaults), not the whole struct.
  let b: ClusterBackend = serde_json::from_str(r#"{"offline":{"threshold":0.42}}"#).unwrap();
  let ClusterBackend::Offline(o) = b else {
    panic!("expected Offline")
  };
  assert_eq!(o.threshold(), 0.42);
  assert_eq!(o.fa(), DEFAULT_FA);
  assert_eq!(o.fb(), DEFAULT_FB);
  assert_eq!(o.max_iters(), DEFAULT_MAX_ITERS);
  assert_eq!(o.min_duration_off(), DEFAULT_MIN_DURATION_OFF);
}

#[cfg(feature = "serde")]
#[test]
fn serde_non_default_round_trips_without_silent_flip() {
  // A fully non-default backend survives serialize → deserialize unchanged (the
  // whisperkit "no silent flip on round trip" lesson).
  let b = ClusterBackend::Offline(
    OfflineOptions::new()
      .with_threshold(0.55)
      .with_fa(0.09)
      .with_fb(0.71)
      .with_max_iters(33)
      .with_min_duration_off(1.25),
  );
  let json = serde_json::to_string(&b).unwrap();
  let back: ClusterBackend = serde_json::from_str(&json).unwrap();
  assert_eq!(back, b);
}

#[cfg(feature = "serde")]
#[test]
fn serde_options_empty_object_is_full_defaults() {
  let o: OfflineOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, OfflineOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn serde_serialize_rejects_non_finite_threshold_fa_fb() {
  // The F6 hole: serde_json would silently write NaN/±∞ as `null`. The
  // finite_f64 helper refuses instead, on the serialize side (where the lossy
  // null would be produced). Reachable because these three builders are
  // unchecked.
  assert!(serde_json::to_string(&OfflineOptions::new().with_threshold(f64::NAN)).is_err());
  assert!(serde_json::to_string(&OfflineOptions::new().with_fa(f64::INFINITY)).is_err());
  assert!(serde_json::to_string(&OfflineOptions::new().with_fb(f64::NEG_INFINITY)).is_err());
  // ...and through the enum wrapper too.
  assert!(
    serde_json::to_string(&ClusterBackend::Offline(
      OfflineOptions::new().with_threshold(f64::NAN)
    ))
    .is_err()
  );
}

#[cfg(feature = "serde")]
#[test]
fn serde_deserialize_rejects_negative_min_duration_off() {
  // finite_nonneg_f64 refuses a negative (finite) min_duration_off at the wire,
  // closing the serde-bypass path into dia's with_min_duration_off panic.
  assert!(serde_json::from_str::<OfflineOptions>(r#"{"min_duration_off":-1.0}"#).is_err());
  assert!(
    serde_json::from_str::<ClusterBackend>(r#"{"offline":{"min_duration_off":-0.001}}"#).is_err()
  );
}

#[cfg(feature = "serde")]
#[test]
fn serde_deserialize_accepts_valid_min_duration_off() {
  let o: OfflineOptions = serde_json::from_str(r#"{"min_duration_off":0.75}"#).unwrap();
  assert_eq!(o.min_duration_off(), 0.75);
  // Zero (the boundary / default) is valid.
  let z: OfflineOptions = serde_json::from_str(r#"{"min_duration_off":0.0}"#).unwrap();
  assert_eq!(z.min_duration_off(), 0.0);
}

// =====================================================================
// OnlineOptions defaults — pinned BOTH to the literal consts AND to dia's OWN
// OnlineClusterOptions accessors, so a drift on EITHER side fails (the mutation
// target: flip a default here → this pin fails).
// =====================================================================

#[test]
fn online_options_new_matches_default() {
  assert_eq!(OnlineOptions::new(), OnlineOptions::default());
}

#[test]
fn online_options_defaults_match_const_literals() {
  let o = OnlineOptions::new();
  assert_eq!(o.speaker_threshold(), DEFAULT_SPEAKER_THRESHOLD);
  assert_eq!(o.embedding_threshold(), DEFAULT_EMBEDDING_THRESHOLD);
  assert_eq!(o.min_speech_duration(), DEFAULT_MIN_SPEECH_DURATION);
  // The concrete bare-`SpeakerManager()` values, pinned so a mutation to any
  // default fails here as well as at the dia cross-check below.
  assert_eq!(o.speaker_threshold(), 0.65);
  assert_eq!(o.embedding_threshold(), 0.45);
  assert_eq!(o.min_speech_duration(), 1.0);
}

#[test]
fn online_defaults_equal_dia() {
  // The load-bearing pin: speakerkit's OnlineOptions defaults MUST equal dia's
  // OnlineClusterOptions defaults (which are FluidAudio's bare
  // `SpeakerManager()`). Read dia's own defaults off a freshly-constructed
  // OnlineClusterOptions and compare. A drift on EITHER side fails this.
  let dia_default = dia::cluster::online::OnlineClusterOptions::default();
  let o = OnlineOptions::default();
  assert_eq!(
    o.speaker_threshold(),
    dia_default.speaker_threshold(),
    "speaker_threshold drifted from dia"
  );
  assert_eq!(
    o.embedding_threshold(),
    dia_default.embedding_threshold(),
    "embedding_threshold drifted from dia"
  );
  assert_eq!(
    o.min_speech_duration(),
    dia_default.min_speech_duration(),
    "min_speech_duration drifted from dia"
  );
}

#[test]
fn online_from_clustering_threshold_matches_dia_ratios() {
  // Production DiarizerManager derivation: speaker = base×1.2, embedding =
  // base×0.8. Pinned against dia's OWN from_clustering_threshold so the two
  // convenience constructors cannot drift.
  let o = OnlineOptions::from_clustering_threshold(0.7);
  let dia = dia::cluster::online::OnlineClusterOptions::from_clustering_threshold(0.7);
  assert_eq!(o.speaker_threshold(), dia.speaker_threshold());
  assert_eq!(o.embedding_threshold(), dia.embedding_threshold());
  assert_eq!(o.min_speech_duration(), dia.min_speech_duration());
  // And the concrete shipping values (0.84 / 0.56).
  assert!((o.speaker_threshold() - 0.84).abs() < 1e-6);
  assert!((o.embedding_threshold() - 0.56).abs() < 1e-6);
  assert_eq!(o.min_speech_duration(), 1.0);
}

// =====================================================================
// OnlineOptions builders / setters (rust-options-pattern) — ALL three
// panic-validate (unlike OfflineOptions' unchecked threshold/fa/fb), mirroring
// dia's OnlineClusterOptions setters.
// =====================================================================

#[test]
fn online_options_builders_and_setters() {
  let o = OnlineOptions::new()
    .with_speaker_threshold(0.9)
    .with_embedding_threshold(0.3)
    .with_min_speech_duration(2.5);
  assert_eq!(o.speaker_threshold(), 0.9);
  assert_eq!(o.embedding_threshold(), 0.3);
  assert_eq!(o.min_speech_duration(), 2.5);

  let mut m = OnlineOptions::new();
  m.set_speaker_threshold(1.1);
  m.set_embedding_threshold(0.2);
  m.set_min_speech_duration(0.0);
  assert_eq!(m.speaker_threshold(), 1.1);
  assert_eq!(m.embedding_threshold(), 0.2);
  assert_eq!(m.min_speech_duration(), 0.0);
}

#[test]
fn online_threshold_boundaries_accept_zero_and_two() {
  // Cosine distance codomain is [0.0, 2.0] inclusive; both endpoints are valid.
  assert_eq!(
    OnlineOptions::new()
      .with_speaker_threshold(0.0)
      .speaker_threshold(),
    0.0
  );
  assert_eq!(
    OnlineOptions::new()
      .with_embedding_threshold(2.0)
      .embedding_threshold(),
    2.0
  );
}

#[test]
#[should_panic(expected = "speaker_threshold must be a finite cosine distance")]
fn with_speaker_threshold_panics_on_nan() {
  let _ = OnlineOptions::new().with_speaker_threshold(f32::NAN);
}

#[test]
#[should_panic(expected = "speaker_threshold must be a finite cosine distance")]
fn with_speaker_threshold_panics_above_two() {
  let _ = OnlineOptions::new().with_speaker_threshold(2.5);
}

#[test]
#[should_panic(expected = "speaker_threshold must be a finite cosine distance")]
fn with_speaker_threshold_panics_on_negative() {
  let _ = OnlineOptions::new().with_speaker_threshold(-0.1);
}

#[test]
#[should_panic(expected = "embedding_threshold must be a finite cosine distance")]
fn with_embedding_threshold_panics_on_infinity() {
  let _ = OnlineOptions::new().with_embedding_threshold(f32::INFINITY);
}

#[test]
#[should_panic(expected = "min_speech_duration must be finite and >= 0")]
fn with_min_speech_duration_panics_on_negative() {
  let _ = OnlineOptions::new().with_min_speech_duration(-1.0);
}

#[test]
#[should_panic(expected = "min_speech_duration must be finite and >= 0")]
fn with_min_speech_duration_panics_on_infinity() {
  let _ = OnlineOptions::new().with_min_speech_duration(f32::INFINITY);
}

#[test]
#[should_panic(expected = "speaker_threshold must be a finite cosine distance")]
fn from_clustering_threshold_overflow_panics() {
  // 2.0 × 1.2 = 2.4 > 2.0 → the derived speaker_threshold is rejected (dia
  // panics identically).
  let _ = OnlineOptions::from_clustering_threshold(2.0);
}

// =====================================================================
// to_dia_options — the single OnlineOptions → dia OnlineClusterOptions mapping.
// =====================================================================

#[test]
fn online_to_dia_options_maps_each_knob() {
  // Three distinct non-default values, so a swapped mapping (e.g.
  // speaker↔embedding) fails.
  let opts = OnlineOptions::new()
    .with_speaker_threshold(0.71)
    .with_embedding_threshold(0.33)
    .with_min_speech_duration(1.75);
  let dia = opts.to_dia_options();
  assert_eq!(dia.speaker_threshold(), 0.71);
  assert_eq!(dia.embedding_threshold(), 0.33);
  assert_eq!(dia.min_speech_duration(), 1.75);
}

#[test]
fn online_to_dia_options_default_equals_dia_default() {
  // The default OnlineOptions maps to exactly dia's default OnlineClusterOptions
  // — the property diarize_online relies on for FluidAudio-default clustering.
  let dia = OnlineOptions::default().to_dia_options();
  let dia_default = dia::cluster::online::OnlineClusterOptions::default();
  assert_eq!(dia.speaker_threshold(), dia_default.speaker_threshold());
  assert_eq!(dia.embedding_threshold(), dia_default.embedding_threshold());
  assert_eq!(dia.min_speech_duration(), dia_default.min_speech_duration());
}

// =====================================================================
// Determinism — the online engine, driven through the speakerkit wiring
// (`to_dia_options` → dia's OnlineClusterer), is total-deterministic given a
// fixed feed order. Hermetic (no models, no golden).
// =====================================================================

#[test]
fn online_clusterer_is_deterministic_given_fixed_order() {
  use dia::{
    cluster::online::{Assignment, OnlineClusterer},
    embed::{EMBEDDING_DIM, Embedding},
  };

  // Three well-separated one-hot-block directions → three distinct speakers.
  let make = |block: usize| -> Embedding {
    let mut raw = [0.0f32; EMBEDDING_DIM];
    raw[(block * 64)..((block + 1) * 64)].fill(1.0);
    Embedding::normalize_from(raw).expect("nonzero")
  };
  let seq = [
    (make(0), 2.0f32),
    (make(1), 2.0),
    (make(0), 2.0), // reuse speaker 1
    (make(2), 2.0),
    (make(1), 2.0), // reuse speaker 2
  ];

  let run = || -> (Vec<Assignment>, Vec<[f32; EMBEDDING_DIM]>) {
    let mut c = OnlineClusterer::new(OnlineOptions::default().to_dia_options());
    let mut assigns = Vec::new();
    for (e, d) in &seq {
      assigns.push(c.assign(e, *d));
    }
    let centroids: Vec<[f32; EMBEDDING_DIM]> =
      c.speaker_ids().map(|id| *c.centroid(id).unwrap()).collect();
    (assigns, centroids)
  };

  let (a1, cent1) = run();
  let (a2, cent2) = run();
  assert_eq!(a1, a2, "assignments must be identical across runs");
  assert_eq!(cent1, cent2, "centroids must be bit-identical across runs");
  // The sequence exercised New (×3) and Existing (×2) with a stable roster.
  assert_eq!(
    a1,
    vec![
      Assignment::New(1),
      Assignment::New(2),
      Assignment::Existing(1),
      Assignment::New(3),
      Assignment::Existing(2),
    ]
  );
}

// =====================================================================
// OnlineOptions serde — per-field defaults, no silent flip, and the SAME
// finite/range rejection OfflineOptions applies (extended to the [0,2] cosine
// bound and the >= 0 duration bound dia's setters assert).
// =====================================================================

#[cfg(feature = "serde")]
#[test]
fn serde_online_empty_payload_is_full_defaults() {
  let b: ClusterBackend = serde_json::from_str(r#"{"online":{}}"#).unwrap();
  assert_eq!(b, ClusterBackend::Online(OnlineOptions::new()));
  let o: OnlineOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, OnlineOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn serde_online_partial_payload_defaults_other_knobs() {
  let b: ClusterBackend = serde_json::from_str(r#"{"online":{"speaker_threshold":0.9}}"#).unwrap();
  let ClusterBackend::Online(o) = b else {
    panic!("expected Online")
  };
  assert_eq!(o.speaker_threshold(), 0.9);
  assert_eq!(o.embedding_threshold(), DEFAULT_EMBEDDING_THRESHOLD);
  assert_eq!(o.min_speech_duration(), DEFAULT_MIN_SPEECH_DURATION);
}

#[cfg(feature = "serde")]
#[test]
fn serde_online_non_default_round_trips_without_silent_flip() {
  let b = ClusterBackend::Online(
    OnlineOptions::new()
      .with_speaker_threshold(0.71)
      .with_embedding_threshold(0.33)
      .with_min_speech_duration(1.75),
  );
  let json = serde_json::to_string(&b).unwrap();
  let back: ClusterBackend = serde_json::from_str(&json).unwrap();
  assert_eq!(back, b);
}

#[cfg(feature = "serde")]
#[test]
fn serde_online_rejects_non_finite_and_out_of_range_thresholds() {
  // Deserialize side: NaN / ±inf / >2 / <0 all refused for both thresholds,
  // closing the serde-bypass path into dia's panicking threshold setter.
  assert!(serde_json::from_str::<OnlineOptions>(r#"{"speaker_threshold":2.5}"#).is_err());
  assert!(serde_json::from_str::<OnlineOptions>(r#"{"speaker_threshold":-0.1}"#).is_err());
  assert!(serde_json::from_str::<OnlineOptions>(r#"{"embedding_threshold":3.0}"#).is_err());
  // NaN/inf have no JSON literal; serde_json parses `null` — the field's
  // deserializer sees a non-number and errors before the predicate, which is
  // also a rejection (the value never lands). A finite-but-out-of-range value
  // (above) exercises the predicate itself.
  assert!(
    serde_json::from_str::<ClusterBackend>(r#"{"online":{"speaker_threshold":2.5}}"#).is_err()
  );
}

#[cfg(feature = "serde")]
#[test]
fn serde_online_serialize_helper_rejects_out_of_range() {
  // A real OnlineOptions cannot hold an out-of-range value (its setters and the
  // deserialize gate both reject one), so the `finite_threshold_f32` serialize
  // branch is only reachable if a future unchecked constructor is ever added.
  // Pin its behaviour now, directly, via a minimal field using the helper —
  // symmetric with the offline `finite_f64` serialize check.
  #[derive(serde::Serialize)]
  struct Bare {
    #[serde(with = "super::finite_threshold_f32")]
    v: f32,
  }
  assert!(serde_json::to_string(&Bare { v: 2.5 }).is_err());
  assert!(serde_json::to_string(&Bare { v: f32::NAN }).is_err());
  // A valid value serializes fine.
  assert!(serde_json::to_string(&Bare { v: 0.65 }).is_ok());
}

#[cfg(feature = "serde")]
#[test]
fn serde_online_rejects_negative_min_speech_duration() {
  assert!(serde_json::from_str::<OnlineOptions>(r#"{"min_speech_duration":-1.0}"#).is_err());
  assert!(
    serde_json::from_str::<ClusterBackend>(r#"{"online":{"min_speech_duration":-0.001}}"#).is_err()
  );
  // Zero (the boundary) is valid.
  let o: OnlineOptions = serde_json::from_str(r#"{"min_speech_duration":0.0}"#).unwrap();
  assert_eq!(o.min_speech_duration(), 0.0);
}
