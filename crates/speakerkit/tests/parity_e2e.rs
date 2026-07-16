//! Gates 3-4 (spec §5, the end-to-end tier): **DER through `dia`'s
//! clustering** — the first speakerkit gate that measures the PRODUCT metric
//! (does the whole pipeline reproduce the reference diarization) rather than
//! tensor-level fidelity.
//!
//! Every prior speakerkit gate is tensor-level: segmentation multilabel
//! agreement (`parity_seg`), embedding cosine (`parity_embed`), argmax
//! Swift bit-exactness (`parity_argmax_swift`). This suite closes the loop:
//! it runs the whole pipeline — seg + embed → `Extraction` →
//! `Extraction::into_offline_input(plda)` → `dia::offline::diarize_offline`
//! → RTTM speaker spans — and scores the spans with **DER** (Diarization
//! Error Rate). It also adjudicates the two questions the design spec
//! explicitly deferred "to the DER gate":
//!
//! - **§5.4** — argmax's embedding cosine vs dia is ~0.94 (a fbank
//!   front-end difference). Does that DEGRADE clustering? **It does, on
//!   multi-speaker audio** (§5.6, Part D below). The answer depends entirely
//!   on the speaker count, which is why a gate that only saw ≤2-speaker clips
//!   got it wrong.
//! - **§5.3** — every fidelity gate pins `ComputeUnits::CpuOnly`; the
//!   shipping default is `All`. The workspace rule: a gate validating a
//!   shipping default MUST run on the shipping default. So the absolute-DER
//!   study runs on `All` and asserts ΔDER(All vs CpuOnly) (Part C).
//!
//! # The four result sets
//!
//! - **A (core T7): FluidAudio-source DER PARITY vs dia-ort.** The same
//!   clips through (i) speakerkit's `FluidAudioSource` → `dia` clustering
//!   and (ii) dia's OWN ort path (`OwnedDiarizationPipeline`: dia-ort
//!   seg+embed → the SAME clustering). Standard DER ≤ 0.1 % absolute;
//!   speaker-count decisions identical; two CpuOnly runs bit-identical
//!   (determinism). See [`fluidaudio_der_parity_vs_dia_ort_and_determinism`].
//! - **B (§5.4/§5.6): argmax-source DER on ≤2-speaker clips.** The same
//!   clips through `ArgmaxSource` → the SAME clustering. argmax matches the
//!   faithful sources to 0.0000 % standard DER here (pinned) — which is precisely
//!   what made the multi-speaker defect invisible for a whole task cycle. See
//!   [`argmax_source_der_characterization`].
//! - **C (§5.3): compute-unit DER.** DER on the shipping default (`All`) and
//!   on `CpuOnly`, for BOTH sources: speaker-count invariance and
//!   ΔDER(All − CpuOnly) are ASSERTED, not merely printed. See
//!   [`compute_unit_der_study_all_vs_cpuonly`].
//! - **D (§5.6): the multi-speaker stress — where the ArgmaxSource FAILS.**
//!   One test per clip over dia's 3-, 4-, 7- and 15-speaker references. This
//!   is the gate whose absence let a real defect ship: A/B/C above score clips
//!   with 1-2 reference speakers, where the clustering decision is near-trivial
//!   (every pairwise distance sits far from AHC's fixed threshold, so even a
//!   miscalibrated embedding lands the same cut) and DER = 0 is NECESSARY but
//!   NOT SUFFICIENT. On the multi-speaker clips argmax scores **3.3-9.3 % DER**
//!   — inventing a spurious speaker on two of them — where the faithful source
//!   scores 0.0-0.4 %. See [`DER_PINS`] for the measured table, and
//!   [`stress_10_mrbeast_clean_water_7_speakers`].
//!
//! # What Part D ASSERTS (and why none of it is a bound)
//!
//! Both sources are pinned by MEASUREMENT ([`DER_PINS`]), not held to a bound,
//! and the table records two findings — read it before trusting either source.
//!
//! **argmax is CHARACTERIZED, NOT VALIDATED** (spec §5.6): its multi-speaker
//! degradation is real, large, reproduced by independent harnesses, and
//! DOCUMENTED (crate README). Note the data does not license the simple rule
//! "argmax fails at ≥3 speakers" — the 3-speaker clip is CLEAN, the 15-speaker
//! clip gets the speaker count RIGHT and still misassigns 3.5 % of speech, and
//! the one clean multi-speaker clip is also the only non-MrBeast one, so
//! speaker count and recording domain are confounded here. [`DER_PINS`] states
//! exactly what was measured and no more.
//!
//! **The FAITHFUL source breaches the spec's 0.1 % parity bound** on two of the
//! four multi-speaker clips (0.1191 % and 0.3948 %, and with CONFUSION rather
//! than boundary units). That bound was only ever measured on ≤2-speaker audio.
//! [`PARITY_DER_MAX`] is left UNCHANGED and still gates Parts A and C; Part D
//! pins reality and the finding is escalated rather than absorbed.
//!
//! Part D deliberately does NOT assert a ≤0.1 % bound on either source here (we
//! know it is violated — asserting it would be a knowingly-failing gate), and it
//! deliberately does NOT carry a loose "sanity ceiling" (a ≤50 % bound cannot
//! fire on a 3.3 % failure — a gate that cannot fail is not a gate; that is
//! exactly the hole this defect shipped through). Every pin fires in BOTH
//! directions.
//!
//! # Which DER metric gates (the `parity_seg` re-scope, one level up)
//!
//! Two DER variants are computed and both REPORTED; only one gates. The
//! **standard DER** ([`der_std`]: 0.25 s collar, overlap excluded — the
//! NIST/DIHARD/pyannote/FluidAudio definition, i.e. what "DER" means) is the
//! GATE, held to the original 0.1 % bound ([`PARITY_DER_MAX`]). The **strict
//! frame-exact DER** ([`der_strict`]: no collar) is REPORTED but not the
//! pass/fail bound. Reason (measured): the FluidAudio source diarizes
//! IDENTICALLY to dia-ort at standard DER (0.0000 %) and reproduces the
//! independent pyannote reference exactly as dia-ort does (0.0000 %
//! each), yet strict DER reads 0.08-0.29 % — a handful of 10 ms frames, ALL
//! within a boundary collar (proven: standard DER = 0). That is the
//! span-level image of the T6-accepted 99.97 % seg agreement (a 1-frame seg
//! flip shifts a span edge by 1-3 frames), i.e. the "boundary jitter DER
//! absorbs" §5.3 anticipated — not a clustering divergence. Gating the
//! spec's 0.1 % on the strict variant would gate an unachievable
//! cross-conversion proxy, the exact anti-pattern `parity_seg.rs` re-scoped
//! (gate the decision metric; report the raw). This re-scope does NOT relax
//! the 0.1 % bound (standard DER meets it at 0.0000 %); the raw strict number
//! is printed every run so any growth stays visible.
//!
//! Both variants are hand-verified in the unit tests, `der_strict` by
//! [`der_identical_is_zero`] … [`optimal_mapping_is_global`] and the GATING
//! `der_std` by [`der_std_forgives_error_on_overlapped_reference_frames`] …
//! [`der_std_skips_only_reference_overlap_not_hypothesis_overlap`].
//!
//! # The DER definition (implemented here)
//!
//! `dia` exposes no public Rust DER helper (its `test_util` is only
//! `repo_root`/`parity_fixtures_root`; its pyannote-parity DER lives in
//! `tests/parity/python/score.py`, out of process). FluidAudio's DER is
//! Swift (`Sources/FluidAudioCLI/Utils/DiarizationMetrics.swift`) — a
//! reference definition, not reusable from Rust. So this suite implements
//! the **standard frame-based DER** (NIST `md-eval` / `pyannote.metrics`
//! `DiarizationErrorRate`):
//!
//! On a 10 ms frame grid (the Kaldi/`md-eval` *scoring* convention — NOT the
//! reference's own resolution: the RTTM's segment DURATIONS are integer
//! multiples of pyannote's 16.875 ms output-frame step, and its absolute
//! boundaries carry the sliding-window origin offset, so they lie on neither a
//! 16.875 ms nor a 10 ms grid; the 10 ms scoring grid is finer than that
//! 16.875 ms data step, so it oversamples the reference and changes no verdict),
//! after (a) a 0.25 s no-score collar on each side
//! of every reference-segment boundary and (b) optionally excluding frames
//! with more than one reference speaker (`skip_overlap`), with the optimal
//! one-to-one speaker mapping (the assignment that maximizes matched
//! reference speech — Hungarian-equivalent; computed exactly by DP over
//! reference subsets):
//!
//! ```text
//! DER = ( missed + false_alarm + confusion ) / total_reference_speech
//!   missed(i)      = max(0, N_ref(i) - N_hyp(i))
//!   false_alarm(i) = max(0, N_hyp(i) - N_ref(i))
//!   confusion(i)   = min(N_ref(i), N_hyp(i)) - N_correct(i)
//! ```
//! summed over scored frames `i`, where `N_correct(i)` counts reference
//! speakers whose mapped hypothesis speaker is also active. Denominator is
//! `Σ N_ref(i)` (total reference speech). This is the pyannote.metrics
//! decomposition verbatim.
//!
//! # Input-match proof (the alignkit lesson)
//!
//! Before any DER number is trusted, every side is proven to consume the
//! identical audio AND the identical framing. Every side is fed the SAME
//! `common::load_wav_16k_mono` buffer (one variable, FNV-fingerprinted), whose
//! length is asserted against [`FIXTURE_FACTS`] so a swapped clip cannot pass
//! as the intended one. The grid geometry is asserted equal: speakerkit's
//! `Extraction::num_chunks` / `Extraction::num_output_frames` must equal dia's
//! own pipeline's `hard_clusters` length / discrete-grid frame count. A
//! misaligned comparison would otherwise fabricate a DER exactly as alignkit's
//! fake 86 % did.
//!
//! # Why the FluidAudio side uses the **fp32** embedder for parity
//!
//! The parity gate (A) isolates the CoreML-vs-ONNX *conversion* physics, so
//! it must hold precision constant against dia-ort's fp32 `wespeaker` ONNX:
//! it loads the fp32 `wespeaker.mlmodelc` (`common::embed_fp32_path`, the
//! same artifact `parity_embed` matched to cosine 0.99999989), NOT the
//! shipping int8 `wespeaker_v2.mlmodelc`. Mixing in int8 quantization
//! (~0.90-0.92 cosine, T6) would confound the conversion-fidelity signal
//! with the quantization axis. The fp32 embedder is likewise the
//! apples-to-apples precision for the §5.4/§5.6 comparison against argmax's
//! Baseline tier (W32A32 seg / W16A16 embed). The int8 shipping-tier DER is a
//! separate axis, gated separately.
//!
//! # "Reference" here means pyannote's OUTPUT, not the truth
//!
//! Absolute DER is scored against `diarization/tests/parity/fixtures/<name>/
//! reference.rttm`. Provenance: dia's `manifest.json` records
//! `pyannote_audio_version: 4.0.4`, so these are **pyannote.audio 4.0.4's own
//! diarization output** on the clip, captured and committed — NOT human
//! annotation. The files carry their own machine lineage: their segment
//! durations are multiples of 0.017 s — pyannote's 16.875 ms output-frame step
//! (`speakerkit::window::FRAME_STEP_S`) at the RTTM's 3 decimal places — and 10
//! of the 14 contain a segment exactly one frame long. No human placed those
//! boundaries.
//!
//! So pyannote 4.0.4 is a genuine THIRD independent reference (distinct from
//! both dia-ort and speakerkit-CoreML — it is the upstream implementation the
//! whole stack targets), and "**DER vs the reference**" in this suite means
//! **distance to pyannote 4.0.4**, never *distance to the truth*. A source
//! scoring 0.0000 % here has reproduced pyannote exactly; it has NOT been
//! shown to be correct. Human-labelled benchmark RTTM (AMI, DIHARD) is not
//! committed locally, so "are we RIGHT?" — as opposed to "do we match the
//! reference implementation?" — remains out of reach of this suite. Every
//! claim it makes is a parity claim.
//!
//! `#[ignore]` (needs the gitignored `Models/speakerkit` +
//! `Models/argmax-speakerkit` artifacts, the sibling `diarization` ONNX +
//! fixtures, and `ort`); the DER-calc unit tests need none of that and run
//! in the ordinary `--features dia` suite. Run the gate with:
//!
//! ```text
//! cargo test -p speakerkit --features dia --test parity_e2e -- --ignored --nocapture
//! ```
//!
//! Part D dominates the runtime (~46 min of audio through three pipelines each,
//! ~77 min of CPU); its four clips are separate `#[test]`s so cargo's default
//! thread pool runs them concurrently.
#![cfg(feature = "dia")]

mod common;
mod der_calc;

use std::{
  collections::BTreeMap,
  path::{Path, PathBuf},
};

use coremlit::ComputeUnits;
use der_calc::{Seg, approx, der_std, der_strict, distinct_speakers, fmt_der, parse_rttm};
use speakerkit::{
  embed::{EmbedModel, EmbedModelOptions},
  extract::{Extraction, Options},
  segment::{SegmentModel, SegmentModelOptions},
  source::{
    ArgmaxComputeOptions, ArgmaxOptions, ArgmaxSource, ArgmaxVariant, FluidAudioSource, ModelSource,
  },
};

// ══════════════════════════════════════════════════════════════════════
// DER bounds (the calculation itself lives in `der_calc` — one definition,
// shared with `parity_shipping_der.rs`, unit-tested where it is defined)
// ══════════════════════════════════════════════════════════════════════

/// The DER parity bound (spec §6 / original T7): 0.1 % absolute. Applied to
/// the STANDARD DER — the conventional metric the spec's "DER" names (0.25 s
/// collar, overlap excluded; the NIST/DIHARD/pyannote/FluidAudio definition,
/// [`der_std`]) — under BOTH readings of "DER delta ≤ 0.1 %": the parity DER
/// of speakerkit's spans against dia-ort's, AND the gap between each source's
/// absolute DER vs the pyannote reference. This is the ORIGINAL spec bound,
/// UNCHANGED; it is never relaxed.
///
/// It binds the FAITHFUL source (FluidAudio), and it GATES Parts A and C, where
/// FluidAudio meets it at 0.0000 %. **It does not hold everywhere.** On the
/// multi-speaker clips FluidAudio scores 0.1191 % and 0.3948 % against dia-ort —
/// over this bound (finding 2 in [`DER_PINS`]). T7 measured the bound only on
/// ≤2-speaker audio and generalized; that generalization is false, and it is
/// escalated as a finding rather than absorbed by editing this constant. **Do
/// not raise it.** Part D pins the measured values instead.
///
/// It is deliberately NOT applied to the ArgmaxSource either: a different
/// embedding space that never promised parity and does not achieve it (§5.6).
/// Asserting a bound already known to be violated is a knowingly-failing gate,
/// so both sources are pinned by measurement in Part D ([`DER_PINS`]).
///
/// It is likewise NOT applied to the strict no-collar [`der_strict`]
/// frame-exact variant: that variant is dominated by unavoidable ±1-3 frame
/// boundary quantization from the accepted 99.97 % segmentation agreement
/// (T6) — the same "unachievable raw proxy across two conversions" that
/// `parity_seg.rs` re-scoped from a gate to a REPORTED stat (spec §5). Strict
/// is reported here for the same reason, with [`STRICT_JITTER_TRIPWIRE`] as a
/// gross-regression guard only.
const PARITY_DER_MAX: f64 = 0.001;

/// A loose gross-regression tripwire on the REPORTED strict (no-collar)
/// frame-exact parity DER — NOT the spec's 0.1 % bound (that is
/// [`PARITY_DER_MAX`] on the standard DER). Set well above the measured
/// boundary jitter (worst 0.29 %, all within a boundary collar — proven by
/// the standard parity DER being 0.0000 %) and far below a genuine
/// clustering regression (which moves whole spans, many %). It exists so a
/// catastrophic seg/embed regression still fails loudly through the strict
/// metric even though benign sub-collar jitter does not. Its value is a
/// controller decision (like `parity_seg`'s agreement floor), never loosened
/// to hide a regression.
const STRICT_JITTER_TRIPWIRE: f64 = 0.01;

// ══════════════════════════════════════════════════════════════════════
// Fixture facts — the corpus, as it IS (never as its filenames claim)
// ══════════════════════════════════════════════════════════════════════

/// Every fixture in dia's parity corpus, with the two facts a DER gate must
/// never guess: how many speakers its reference actually contains, and how
/// many samples its clip actually holds.
///
/// **The speaker counts are read from the reference RTTMs. The clip NAMES are
/// wrong.** `04_three_speaker`'s reference holds ONE speaker; `05_four_speaker`'s
/// holds two. They are the only two clips whose names advertise a speaker
/// count, and they are the only two that lie about it.
///
/// This is not trivia — it is the root cause of spec §5.6. T7 wanted
/// multi-speaker coverage, selected `04_three_speaker` and `05_four_speaker`
/// **by name**, actually measured a 1-speaker and a 2-speaker clip, saw a clean
/// DER, and concluded that the corpus "caps at 1-2 speakers". It does not: it
/// ships EIGHT clips with 3-15 speaker references, all already resolvable by
/// this suite's loader. The one clip that would have exposed argmax's
/// multi-speaker defect (`10_mrbeast_clean_water`, 7 speakers) was in the repo
/// the whole time.
///
/// So: [`reference_segments`] re-derives the count from the RTTM on every load
/// and asserts it against this table, and [`fixture_audio`] does the same for
/// the sample count. A test cannot obtain a reference or a clip without both
/// being checked, which is what makes that mistake unrepeatable.
struct FixtureFacts {
  /// Fixture directory name in dia's parity corpus (`<name>/clip_16k.wav`,
  /// `<name>/reference.rttm`).
  name: &'static str,
  /// Distinct speakers in the pyannote reference RTTM — counted from the file,
  /// NOT inferred from [`FixtureFacts::name`].
  ref_speakers: usize,
  /// Exact 16 kHz mono sample count of `clip_16k.wav`. Asserted on load, so a
  /// corpus update that swaps the audio under a name cannot slip through.
  samples: usize,
}

/// The corpus (dia's `tests/parity/fixtures/`). See [`FixtureFacts`] — the
/// speaker counts come from the RTTMs, not the names.
const FIXTURE_FACTS: &[FixtureFacts] = &[
  FixtureFacts {
    name: "01_dialogue",
    ref_speakers: 2,
    samples: 3_631_361,
  },
  FixtureFacts {
    name: "02_pyannote_sample",
    ref_speakers: 2,
    samples: 480_000,
  },
  FixtureFacts {
    name: "03_dual_speaker",
    ref_speakers: 2,
    samples: 960_000,
  },
  // NAME LIES: one speaker in the reference, not three.
  FixtureFacts {
    name: "04_three_speaker",
    ref_speakers: 1,
    samples: 639_573,
  },
  // NAME LIES: two speakers in the reference, not four.
  FixtureFacts {
    name: "05_four_speaker",
    ref_speakers: 2,
    samples: 960_000,
  },
  FixtureFacts {
    name: "06_long_recording",
    ref_speakers: 3,
    samples: 15_643_627,
  },
  FixtureFacts {
    name: "07_yuhewei_dongbei_english",
    ref_speakers: 2,
    samples: 404_213,
  },
  FixtureFacts {
    name: "08_luyu_jinjing_freedom",
    ref_speakers: 3,
    samples: 22_675_308,
  },
  FixtureFacts {
    name: "09_mrbeast_dollar_date",
    ref_speakers: 8,
    samples: 16_671_744,
  },
  FixtureFacts {
    name: "10_mrbeast_clean_water",
    ref_speakers: 7,
    samples: 9_911_979,
  },
  FixtureFacts {
    name: "11_mrbeast_age_race",
    ref_speakers: 6,
    samples: 22_568_310,
  },
  FixtureFacts {
    name: "12_mrbeast_schools",
    ref_speakers: 15,
    samples: 15_426_781,
  },
  FixtureFacts {
    name: "13_mrbeast_saved_animals",
    ref_speakers: 11,
    samples: 16_882_005,
  },
  FixtureFacts {
    name: "14_mrbeast_strongman_robot",
    ref_speakers: 4,
    samples: 17_648_640,
  },
];

/// The [`FixtureFacts`] row for `name`.
///
/// # Panics
/// If `name` is not in the corpus — a typo'd fixture name must be a hard error,
/// never a silently-skipped test.
fn facts(name: &str) -> &'static FixtureFacts {
  FIXTURE_FACTS
    .iter()
    .find(|f| f.name == name)
    .unwrap_or_else(|| panic!("{name}: not a fixture in FIXTURE_FACTS"))
}

/// The ≥3-speaker clips this suite GATES on (Part D), one `#[test]` each.
///
/// Chosen from the eight ≥3-speaker clips in [`FIXTURE_FACTS`] to span the
/// speaker-count ladder — 3, 4, 7, 15 — at the lowest runtime that still
/// covers it:
///
/// - `10_mrbeast_clean_water` (7): **the counterexample.** This is the clip
///   that overturns §5.4; it must be in the gate forever.
/// - `06_long_recording` (3): the MINIMAL ≥3 case — the boundary where the
///   clustering decision stops being trivial — and the only non-MrBeast clip
///   in the set, so the finding is not an artifact of one recording style.
/// - `12_mrbeast_schools` (15): the richest reference in the corpus, i.e. the
///   maximum stress AHC's fixed threshold can face here.
/// - `14_mrbeast_strongman_robot` (4): fills the 3→7 gap.
///
/// The other four (`08`=3, `09`=8, `11`=6, `13`=11) are redundant in kind and
/// would roughly double the CPU cost (they are the four longest clips), so they
/// are documented in [`FIXTURE_FACTS`] but not gated. Adding one back is a
/// three-line `#[test]` plus its [`DER_PINS`] row.
const STRESS_FIXTURES: &[&str] = &[
  "06_long_recording",
  "10_mrbeast_clean_water",
  "12_mrbeast_schools",
  "14_mrbeast_strongman_robot",
];

/// The pyannote reference RTTM for a fixture, from the sibling `diarization`
/// checkout (override root via `DIA_PARITY_FIXTURES`). Same three-levels-up
/// convention as the crate's `dia` path dependency and
/// `generate_goldens.rs`'s wespeaker ONNX resolution.
fn reference_rttm_path(name: &str) -> PathBuf {
  let root = std::env::var_os("DIA_PARITY_FIXTURES").map_or_else(
    || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../diarization/tests/parity/fixtures"),
    PathBuf::from,
  );
  root.join(name).join("reference.rttm")
}

/// Extra dia-only fixtures (not committed to speakerkit) that extend the
/// §5.4/§5.3 studies past the two committed ~25-30 s clips: longer, multi-turn
/// audio (40 s / 60 s) and the FIRST argmax multi-chunk (>30 s) coverage.
///
/// What they are NOT is multi-speaker coverage. Their references hold ONE and
/// TWO speakers respectively ([`FIXTURE_FACTS`]) — the names are aspirational.
/// Reading speaker coverage off those names is the mistake that produced spec
/// §5.6; the ≥3-speaker stress is [`STRESS_FIXTURES`]' job, and no other test
/// may claim it.
const EXTRA_DIA_FIXTURES: &[&str] = &["04_three_speaker", "05_four_speaker"];

/// The audio path for a fixture: the byte-verified committed WAV for the two
/// speakerkit fixtures, else the sibling dia checkout's `clip_16k.wav`.
fn fixture_audio_path(name: &str) -> PathBuf {
  if common::FIXTURES.iter().any(|f| f.name == name) {
    common::audio_path(name)
  } else {
    reference_rttm_path(name)
      .parent()
      .expect("fixture dir")
      .join("clip_16k.wav")
  }
}

/// Load a fixture's 16 kHz mono audio, asserting it is the clip this suite
/// believes it is (exact sample count, [`FixtureFacts::samples`]).
///
/// One of the two loading chokepoints (the other is [`reference_segments`]).
/// The pair is what makes a gate structurally unable to test something other
/// than what it claims to test.
fn fixture_audio(name: &str) -> Vec<f32> {
  let path = fixture_audio_path(name);
  assert!(
    path.exists(),
    "{name}: clip not found at {} — this suite requires the sibling `diarization` \
     checkout (override with DIA_PARITY_FIXTURES)",
    path.display()
  );
  let samples = common::load_wav_16k_mono(&path);
  assert_eq!(
    samples.len(),
    facts(name).samples,
    "{name}: loaded {} samples but FIXTURE_FACTS says {} — the corpus changed under this \
     name; re-derive the table before trusting any DER it produces",
    samples.len(),
    facts(name).samples
  );
  samples
}

/// The pyannote reference for a fixture, with its TRUE speaker count asserted
/// against [`FIXTURE_FACTS`].
///
/// The only way this suite obtains a reference. Deriving the count from the
/// file it just parsed — rather than trusting the fixture's name, or a table
/// nobody re-checks — is what makes T7's misnaming trap unrepeatable: a test
/// that believes it has 3-speaker coverage but has loaded a 1-speaker clip now
/// fails instead of quietly reporting a meaningless DER.
///
/// (What it returns is pyannote 4.0.4's OUTPUT, not human ground truth — see
/// the module doc. Every "absolute DER" downstream is a distance to pyannote.)
fn reference_segments(name: &str) -> Vec<Seg> {
  let segs = parse_rttm(&reference_rttm_path(name));
  let counted = distinct_speakers(&segs).len();
  assert_eq!(
    counted,
    facts(name).ref_speakers,
    "{name}: its reference.rttm holds {counted} speakers but FIXTURE_FACTS says {} — the \
     corpus changed, so every speaker-count claim this suite makes about it is now unverified. \
     Re-derive the table from the RTTMs; do NOT edit the expectation to match a clip you have \
     not looked at.",
    facts(name).ref_speakers
  );
  segs
}

/// The names driving the §5.4/§5.3 studies (Parts B and C): the two committed
/// fixtures plus [`EXTRA_DIA_FIXTURES`]. All four have ≤2 reference speakers —
/// this is the EASY half of the corpus, and its clean DER is only meaningful
/// next to Part D's.
fn e2e_fixture_names() -> Vec<&'static str> {
  let mut names: Vec<&'static str> = common::FIXTURES.iter().map(|f| f.name).collect();
  names.extend_from_slice(EXTRA_DIA_FIXTURES);
  names
}

// ══════════════════════════════════════════════════════════════════════
// The end-to-end characterization (spec §5.6) — pinned, not bounded
// ══════════════════════════════════════════════════════════════════════

/// Tolerance on a pinned DER, absolute (0.05 pp). Tight: the CpuOnly pipelines
/// are deterministic (Part A asserts bit-identical reruns), and the smallest
/// argmax degradation pinned below is 3.3 pp — 66× this band. Any
/// clustering-decision change moves DER by whole points and fires immediately;
/// the band exists only to absorb a stray flipped frame on a different CoreML
/// build, not to hide movement. Never widen it to make a pin pass.
///
/// **This band is WIDER than the margin by which the faithful source breaches
/// [`PARITY_DER_MAX`]** (clip 12's `fa_vs_dia` clears the 0.1 % bound by only
/// 0.0191 pp, well inside 0.05 pp). So the tolerance ALONE would let a
/// pinned-breaching clip drift back under the bound in-tolerance — the breach
/// warning would vanish and the crate README's "breaches on two clips" claim
/// would go silently false with no red gate (finding 1). [`assert_pinned`]
/// closes that: a measurement and its pin must also land on the SAME SIDE of
/// [`PARITY_DER_MAX`], so a bound crossing is always a deliberate re-baseline,
/// never a tolerance no-op.
const DER_PIN_TOL: f64 = 0.0005;

/// What the pipeline ACTUALLY does, per clip, end to end — the executable form
/// of spec §5.6. Every number here is measured, not predicted.
///
/// | clip | ref spk | FluidAudio spk / DER vs dia-ort | argmax spk / DER vs pyannote |
/// |---|---|---|---|
/// | `02_pyannote_sample` | 2 | 2 / 0.0000 % | 2 / 0.0000 % |
/// | `07_yuhewei_dongbei_english` | 2 | 2 / 0.0000 % | 2 / 0.0000 % |
/// | `04_three_speaker` | 1 | 1 / 0.0000 % | 1 / 0.0000 % |
/// | `05_four_speaker` | 2 | 2 / 0.0000 % | 2 / 0.0000 % |
/// | `06_long_recording` | 3 | 3 / 0.0909 % | **3** / 0.0908 % |
/// | `14_mrbeast_strongman_robot` | 4 | 4 / **0.3948 %** | **5** / **9.2934 %** |
/// | `10_mrbeast_clean_water` | 7 | 7 / 0.0000 % | **8** / **3.3282 %** |
/// | `12_mrbeast_schools` | 15 | 15 / **0.1191 %** | 15 / **3.4582 %** |
///
/// Two findings live in that table, and neither was visible on the ≤2-speaker
/// corpus T7 measured.
///
/// **1. The ArgmaxSource is CHARACTERIZED, NOT VALIDATED.** It reproduces the
/// faithful sources to 0.0000 % standard DER at 1-2 speakers, and it holds up on the 3-speaker
/// clip. On the other three multi-speaker clips it diverges hard: 3.3-9.3 % DER,
/// against a faithful source scoring 0.0-0.4 % on the same audio, through the
/// same framing, the same clustering, the same reference and the same harness —
/// so **argmax's embedding is the only variable**. Where it fails, the divergence
/// is essentially pure CONFUSION ([`DerPin::ax_vs_fluidaudio_confusion`] ≈
/// [`DerPin::ax_vs_fluidaudio`], with zero miss and zero false alarm): argmax
/// hears the same speech and assigns it to the wrong person, which is why no
/// collar absorbs it.
///
/// Note what the data does NOT say. It does not say "argmax fails at ≥3
/// speakers": `06_long_recording` has 3 and is clean. It does not say the defect
/// is only a spurious speaker: on `12_mrbeast_schools` argmax gets the speaker
/// COUNT exactly right (15) and still misassigns 3.46 % of speech. And the one
/// clean multi-speaker clip is also the only non-MrBeast one, so speaker count
/// and recording domain are **confounded** in this corpus. The failure is real,
/// large and reproducible; its precise trigger is NOT isolated by these four
/// clips. Claiming a clean "≥N speakers" threshold would be repeating T7's
/// mistake — reading a rule off a corpus that cannot support it.
///
/// Mechanism (§5.6, corroborated in `dia`'s source): `dia`'s AHC cuts at a FIXED
/// 0.6 linkage threshold inside a FROZEN, PRETRAINED PLDA projection —
/// `PldaTransform::new()` takes no data, it `include_bytes!`s an LDA (256→128) +
/// PLDA fit on the native kaldi-fbank WeSpeaker distribution. argmax's embedder
/// eats an 80-mel spectrogram from its own preprocessor instead, so its vectors
/// land in a differently-scaled space the frozen projection was never fit for.
/// Where every pairwise distance sits far from the threshold, a miscalibrated
/// projection still cuts correctly; where distances crowd it, merges flip.
///
/// **2. NEW — the FAITHFUL source breaches the spec's 0.1 % parity bound on
/// multi-speaker audio.** FluidAudio is 0.0000 % against dia-ort on every
/// ≤2-speaker clip and on the 7-speaker clip, but scores **0.1191 %** on
/// `12_mrbeast_schools` and **0.3948 %** on `14_mrbeast_strongman_robot` — over
/// [`PARITY_DER_MAX`]. Its error there is CONFUSION (289 of 293 units on `14`),
/// not boundary jitter, so the collar cannot explain it away: the CoreML
/// conversion's numerical drift does flip a small number of clustering
/// assignments once several speakers must be separated. FluidAudio never gets the
/// speaker COUNT wrong (asserted below), and it stays ~23× more faithful than
/// argmax on the same clip — but "0.1 % DER parity" is a claim T7 only ever
/// tested on 1-2 speaker audio, and on multi-speaker audio it is FALSE.
///
/// That is a finding for the spec owner to adjudicate, not for this test to paper
/// over. [`PARITY_DER_MAX`] is therefore UNCHANGED and still gates Parts A and C.
/// Part D pins the measured reality instead — because asserting a bound already
/// known to be violated is a knowingly-failing gate, and quietly raising the
/// bound to 0.4 % would be exactly the loosening this suite exists to prevent.
/// **Do not "fix" a red Part D by touching `PARITY_DER_MAX`.**
///
/// **These are pins, not bounds.** They fire in BOTH directions: if a source gets
/// worse, and if it gets better — because "better" means a root cause moved, and
/// that must be a deliberate re-baseline (re-measure, update this table, update
/// the crate README), never a silent pass.
struct DerPin {
  /// Fixture name (a key into [`FIXTURE_FACTS`]).
  name: &'static str,
  /// FluidAudio's standard parity DER against dia-ort — the T7 claim, measured
  /// per clip. Above [`PARITY_DER_MAX`] on `12` and `14`; see finding 2 above.
  fa_vs_dia: f64,
  /// ...of which CONFUSION — the KIND of the parity result, pinned so a breach
  /// cannot silently change character. On the two clips that breach
  /// [`PARITY_DER_MAX`] (`12`, `14`) essentially all of `fa_vs_dia` is confusion
  /// (a clustering divergence the collar cannot absorb), which is the whole
  /// reason the breach is a finding and not benign boundary jitter; on the clean
  /// and near-clean clips it is zero (clip 06's small error is boundary MISS, so
  /// its confusion is 0.0 too). The breach warning already prints these units —
  /// pinning the fraction turns that report into a gate (finding 1).
  fa_vs_dia_confusion: f64,
  /// FluidAudio's standard DER against the pyannote reference.
  fa_vs_reference: f64,
  /// Speakers the ArgmaxSource's clustering DECIDES on. Compare with
  /// [`FixtureFacts::ref_speakers`]: 5-vs-4 on `14` and 8-vs-7 on `10` are
  /// spurious speakers; on `12` the count is right and the assignment is not.
  ax_speakers: usize,
  /// argmax's standard DER against the pyannote reference.
  ax_vs_reference: f64,
  /// argmax's standard DER against FluidAudio's spans.
  ax_vs_fluidaudio: f64,
  /// ...of which CONFUSION. Pinned separately from the total on purpose: where
  /// argmax fails, essentially ALL of its error is confusion (same speech, wrong
  /// speaker), whereas on the clean `06` its whole 0.0037 % is boundary MISS and
  /// confusion is zero. The two are different in KIND, and only this field says
  /// which — the size alone cannot.
  ax_vs_fluidaudio_confusion: f64,
}

/// The pinned end-to-end characterization — see [`DerPin`].
const DER_PINS: &[DerPin] = &[
  // ── ≤2 reference speakers: every source scores 0.0000 % standard-collar DER. (Parts B/C.)
  DerPin {
    name: "02_pyannote_sample",
    fa_vs_dia: 0.0,
    fa_vs_dia_confusion: 0.0,
    fa_vs_reference: 0.0,
    ax_speakers: 2,
    ax_vs_reference: 0.0,
    ax_vs_fluidaudio: 0.0,
    ax_vs_fluidaudio_confusion: 0.0,
  },
  DerPin {
    name: "07_yuhewei_dongbei_english",
    fa_vs_dia: 0.0,
    fa_vs_dia_confusion: 0.0,
    fa_vs_reference: 0.0,
    ax_speakers: 2,
    ax_vs_reference: 0.0,
    ax_vs_fluidaudio: 0.0,
    ax_vs_fluidaudio_confusion: 0.0,
  },
  DerPin {
    name: "04_three_speaker",
    fa_vs_dia: 0.0,
    fa_vs_dia_confusion: 0.0,
    fa_vs_reference: 0.0,
    ax_speakers: 1,
    ax_vs_reference: 0.0,
    ax_vs_fluidaudio: 0.0,
    ax_vs_fluidaudio_confusion: 0.0,
  },
  DerPin {
    name: "05_four_speaker",
    fa_vs_dia: 0.0,
    fa_vs_dia_confusion: 0.0,
    fa_vs_reference: 0.0,
    ax_speakers: 2,
    ax_vs_reference: 0.0,
    ax_vs_fluidaudio: 0.0,
    ax_vs_fluidaudio_confusion: 0.0,
  },
  // ── ≥3 reference speakers (Part D). argmax holds on 06 and breaks on 14/10/12;
  //    FluidAudio exceeds PARITY_DER_MAX on 12 and 14 (finding 2).
  DerPin {
    name: "06_long_recording",
    fa_vs_dia: 0.000_909,
    fa_vs_dia_confusion: 0.0,
    fa_vs_reference: 0.000_908,
    ax_speakers: 3,
    ax_vs_reference: 0.000_908,
    ax_vs_fluidaudio: 0.000_037,
    ax_vs_fluidaudio_confusion: 0.0,
  },
  DerPin {
    name: "14_mrbeast_strongman_robot",
    fa_vs_dia: 0.003_948,
    fa_vs_dia_confusion: 0.003_895,
    fa_vs_reference: 0.003_961,
    ax_speakers: 5,
    ax_vs_reference: 0.092_934,
    ax_vs_fluidaudio: 0.089_887,
    ax_vs_fluidaudio_confusion: 0.089_887,
  },
  DerPin {
    name: "10_mrbeast_clean_water",
    fa_vs_dia: 0.0,
    fa_vs_dia_confusion: 0.0,
    fa_vs_reference: 0.0,
    ax_speakers: 8,
    ax_vs_reference: 0.033_282,
    ax_vs_fluidaudio: 0.033_266,
    ax_vs_fluidaudio_confusion: 0.033_266,
  },
  DerPin {
    name: "12_mrbeast_schools",
    fa_vs_dia: 0.001_191,
    fa_vs_dia_confusion: 0.001_178,
    fa_vs_reference: 0.001_178,
    ax_speakers: 15,
    ax_vs_reference: 0.034_582,
    ax_vs_fluidaudio: 0.034_831,
    ax_vs_fluidaudio_confusion: 0.034_831,
  },
];

/// The [`DerPin`] for `name`.
///
/// # Panics
/// If `name` has no pin — a DER measured against no expectation is a number
/// nobody has adjudicated, which is what this suite exists to prevent.
fn der_pin(name: &str) -> &'static DerPin {
  DER_PINS
    .iter()
    .find(|p| p.name == name)
    .unwrap_or_else(|| panic!("{name}: no DER_PINS row — measure it and pin it before gating"))
}

/// Assert a pinned DER, in both directions — AND that a drift within the
/// tolerance band cannot silently carry the value ACROSS the spec's parity
/// bound (finding 1). [`DER_PIN_TOL`] is wider than the margin by which the
/// breach clips clear [`PARITY_DER_MAX`], so `(measured − pinned).abs() ≤ tol`
/// alone would let a pinned-breaching clip drift back under the bound — the
/// breach warning would disappear with no red gate. Requiring the measurement
/// and its pin to sit on the SAME SIDE of the bound makes any bound crossing a
/// deliberate re-baseline instead of a tolerance no-op.
fn assert_pinned(name: &str, what: &str, measured: f64, pinned: f64) {
  assert!(
    (measured - pinned).abs() <= DER_PIN_TOL,
    "{name}: {what} is {:.4}%, pinned at {:.4}% (±{:.4}%). The characterization has MOVED. \
     Worse is a regression; BETTER means a root cause changed and needs a deliberate \
     re-baseline (re-measure, update DER_PINS, update the crate README). Either way, do not \
     just edit the number, and do NOT widen DER_PIN_TOL.",
    measured * 100.0,
    pinned * 100.0,
    DER_PIN_TOL * 100.0
  );
  // Same side of PARITY_DER_MAX: a value must not cross the spec bound while
  // staying inside DER_PIN_TOL. This is what keeps the faithful source's
  // multi-speaker breach from silently un-breaching (finding 1).
  assert_eq!(
    measured > PARITY_DER_MAX,
    pinned > PARITY_DER_MAX,
    "{name}: {what} is {:.4}% but its pin is {:.4}% — on OPPOSITE sides of the PARITY_DER_MAX \
     {:.4}% bound, a crossing DER_PIN_TOL ({:.4}%) would otherwise absorb. If a breach clip \
     dropped under the bound, the crate README's 'breaches on two clips' claim just went \
     silently false; if a clean clip rose over it, the faithful source just started breaching. \
     Either way, re-baseline deliberately (re-measure, update DER_PINS + README) — do NOT let \
     the tolerance hide a bound crossing.",
    measured * 100.0,
    pinned * 100.0,
    PARITY_DER_MAX * 100.0,
    DER_PIN_TOL * 100.0
  );
}

/// Score all three pipelines on one clip and assert everything the §5.6 verdict
/// rests on: the structural invariants that must hold on EVERY clip, and the
/// per-clip [`DerPin`]. Nothing a conclusion depends on is merely printed.
///
/// Shared by Part B (the ≤2-speaker clips) and Part D (the ≥3-speaker stress) —
/// the two halves of one characterization, so they must score identically.
fn assert_clip_pins(
  name: &str,
  dia: &DiaOrtRun,
  fa_segs: &[Seg],
  ax_segs: &[Seg],
  reference: &[Seg],
) {
  let pin = der_pin(name);
  let ref_speakers = facts(name).ref_speakers;

  let fa_vs_dia = der_std(&dia.segs, fa_segs);
  let ax_vs_dia = der_std(&dia.segs, ax_segs);
  let ax_vs_fa = der_std(fa_segs, ax_segs);
  let dia_vs_ref = der_std(reference, &dia.segs);
  let fa_vs_ref = der_std(reference, fa_segs);
  let ax_vs_ref = der_std(reference, ax_segs);

  let (n_dia, n_fa, n_ax) = (
    distinct_speakers(&dia.segs).len(),
    distinct_speakers(fa_segs).len(),
    distinct_speakers(ax_segs).len(),
  );
  println!(
    "[{name}] {}",
    fmt_der("PARITY fluidaudio vs dia-ort ", &fa_vs_dia)
  );
  println!(
    "[{name}] {}",
    fmt_der("PARITY argmax     vs dia-ort ", &ax_vs_dia)
  );
  println!(
    "[{name}] {}",
    fmt_der("PARITY argmax     vs fluidaud", &ax_vs_fa)
  );
  println!(
    "[{name}] {}",
    fmt_der("ABS    dia-ort    vs pyannote", &dia_vs_ref)
  );
  println!(
    "[{name}] {}",
    fmt_der("ABS    fluidaudio vs pyannote", &fa_vs_ref)
  );
  println!(
    "[{name}] {}",
    fmt_der("ABS    argmax     vs pyannote", &ax_vs_ref)
  );
  println!(
    "[{name}] SPEAKER COUNTS: reference={ref_speakers} dia-ort={n_dia} fluidaudio={n_fa} \
     argmax={n_ax} (argmax pin {})",
    pin.ax_speakers
  );

  // ── Structural invariants. These hold on EVERY clip in the corpus, 1 to 15
  // speakers, and they are what make the argmax pin attributable to argmax: the
  // oracle reproduces the reference on every collar-scored frame, and the
  // faithful source reproduces the oracle's speaker-count DECISION exactly. If
  // either breaks, nothing measured on this clip means anything — so they are
  // exact, not toleranced.
  assert_eq!(
    dia_vs_ref.err_units(),
    0,
    "{name}: dia-ort no longer reproduces the pyannote reference on every collar-scored frame \
     ({} scored-frame error units) — the CONTROL is broken. Every DER this suite reports is \
     scored against that reference, so no conclusion from this clip is trustworthy until this \
     is explained.",
    dia_vs_ref.err_units()
  );
  assert_eq!(
    n_dia, ref_speakers,
    "{name}: dia-ort found {n_dia} speakers against a {ref_speakers}-speaker reference"
  );
  assert_eq!(
    n_fa, n_dia,
    "{name}: FluidAudio's speaker-count DECISION differs from dia-ort's ({n_fa} vs {n_dia}). \
     Boundary jitter cannot add or drop a speaker — this is a clustering divergence in the \
     CoreML conversion. Do NOT loosen; investigate."
  );

  // ── The per-clip pins: both sources, both directions.
  assert_pinned(
    name,
    "FluidAudio parity DER vs dia-ort",
    fa_vs_dia.der,
    pin.fa_vs_dia,
  );
  // The KIND of the parity result, pinned alongside its size: the two breach
  // clips are a finding precisely because their error is CONFUSION (clustering),
  // not collar-absorbable boundary jitter. Gating the confusion component stops
  // a breach from silently changing character (finding 1).
  assert_pinned(
    name,
    "FluidAudio parity CONFUSION vs dia-ort",
    fa_vs_dia.confusion,
    pin.fa_vs_dia_confusion,
  );
  assert_pinned(
    name,
    "FluidAudio DER vs pyannote",
    fa_vs_ref.der,
    pin.fa_vs_reference,
  );
  assert_eq!(
    n_ax, pin.ax_speakers,
    "{name}: argmax now decides {n_ax} speakers, pinned at {} (reference has {ref_speakers}). \
     The §5.6 characterization has MOVED — re-measure and re-baseline deliberately.",
    pin.ax_speakers
  );
  assert_pinned(
    name,
    "argmax DER vs pyannote",
    ax_vs_ref.der,
    pin.ax_vs_reference,
  );
  assert_pinned(
    name,
    "argmax DER vs FluidAudio",
    ax_vs_fa.der,
    pin.ax_vs_fluidaudio,
  );
  // The KIND of argmax's divergence, not just its size — see the field doc.
  assert_pinned(
    name,
    "argmax-vs-FluidAudio CONFUSION",
    ax_vs_fa.confusion,
    pin.ax_vs_fluidaudio_confusion,
  );

  // Loud, so a reader of the log cannot miss finding 2 (see [`DerPin`]): the
  // FAITHFUL source is over the spec's parity bound on this clip. This is now a
  // report on top of a GATE, not the gate itself: `assert_pinned` above already
  // fails if this clip's `fa_vs_dia` drifts across PARITY_DER_MAX (same-side
  // guard) or if its confusion character changes, so the breach cannot silently
  // vanish (finding 1).
  if fa_vs_dia.der > PARITY_DER_MAX {
    println!(
      "[{name}] ⚠ SPEC BOUND BREACHED BY THE FAITHFUL SOURCE: FluidAudio parity DER {:.4}% > \
       PARITY_DER_MAX {:.4}% ({} of {} error units are CONFUSION, i.e. clustering, not \
       boundary jitter). Pinned, NOT waived — see DER_PINS. Do not raise PARITY_DER_MAX.",
      fa_vs_dia.der * 100.0,
      PARITY_DER_MAX * 100.0,
      fa_vs_dia.conf_units,
      fa_vs_dia.err_units(),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════
// Pipeline runners
// ══════════════════════════════════════════════════════════════════════

/// The shared community-1 PLDA both the speakerkit path
/// (`into_offline_input`) and dia's own pipeline consume — one instance, so
/// the two clustering runs cannot diverge on the projection.
///
/// NB `new()` takes no data: it is a FROZEN, pretrained projection
/// (`include_bytes!`) fit on the native kaldi-fbank WeSpeaker distribution.
/// That is not an implementation detail — it is the mechanism behind §5.6 (see
/// [`DerPin`]): a source whose embeddings live in a differently-scaled space
/// is projected through a basis that was never fit for it.
fn load_plda() -> dia::plda::PldaTransform {
  dia::plda::PldaTransform::new().expect("load community-1 PldaTransform")
}

/// dia's OWN ort path (the parity oracle): dia-ort seg (bundled
/// `segmentation-3.0`) + dia-ort embed (fp32 `wespeaker_resnet34_lm.onnx`) →
/// `OwnedDiarizationPipeline` → the SAME `diarize_offline` clustering. This
/// is exactly the fp32-dia oracle Task 6 held FluidAudio to, run end-to-end.
struct DiaOrtRun {
  segs: Vec<Seg>,
  num_chunks: usize,
  num_output_frames: usize,
  num_clusters: usize,
}

/// Resolves the BYO WeSpeaker fp32 ONNX exactly as `generate_goldens.rs`:
/// `DIA_EMBED_MODEL_PATH`, else the sibling `diarization/models/`.
fn dia_wespeaker_onnx() -> PathBuf {
  std::env::var_os("DIA_EMBED_MODEL_PATH").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../diarization/models/wespeaker_resnet34_lm.onnx")
    },
    PathBuf::from,
  )
}

fn dia_ort_run(samples: &[f32], plda: &dia::plda::PldaTransform) -> DiaOrtRun {
  let mut seg = dia::segment::SegmentModel::bundled().expect("dia bundled segmentation-3.0");
  let onnx = dia_wespeaker_onnx();
  assert!(
    onnx.exists(),
    "dia WeSpeaker ONNX not found at {}; set DIA_EMBED_MODEL_PATH",
    onnx.display()
  );
  let mut embed = dia::embed::EmbedModel::from_file(&onnx).expect("dia WeSpeaker fp32 ONNX");
  let pipeline = dia::offline::OwnedDiarizationPipeline::new();
  let out = pipeline
    .run(&mut seg, &mut embed, plda, samples)
    .expect("dia OwnedDiarizationPipeline::run");
  let num_clusters = out.num_clusters();
  let num_chunks = out.hard_clusters_slice().len();
  let disc = out.discrete_diarization_slice();
  let num_output_frames = disc.len().checked_div(num_clusters).unwrap_or(0);
  DiaOrtRun {
    segs: output_segs(&out),
    num_chunks,
    num_output_frames,
    num_clusters,
  }
}

/// dia `OfflineOutput` RTTM spans → [`Seg`]s (cluster id is already a
/// 0-indexed integer speaker id). Names only `OfflineOutput`, never
/// `RttmSpan`, so no extra import.
fn output_segs(out: &dia::offline::OfflineOutput) -> Vec<Seg> {
  out
    .spans_slice()
    .iter()
    .map(|s| Seg {
      start: s.start(),
      end: s.end(),
      spk: s.cluster(),
    })
    .collect()
}

/// speakerkit's FluidAudio source over explicitly-loaded models (fp32
/// embedder for parity — see the module doc), producing the `Extraction`.
fn fluidaudio_extraction(
  samples: &[f32],
  seg_cu: ComputeUnits,
  embed_cu: ComputeUnits,
  embed_path: &Path,
) -> Extraction {
  let seg = SegmentModel::from_file_with(
    common::seg_path(),
    SegmentModelOptions::new().with_compute(seg_cu),
  )
  .expect("load pyannote_segmentation.mlmodelc");
  let embed =
    EmbedModel::from_file_with(embed_path, EmbedModelOptions::new().with_compute(embed_cu))
      .expect("load wespeaker embedder");
  FluidAudioSource::with_options(seg, embed, Options::new())
    .extract(samples)
    .expect("FluidAudioSource::extract")
}

/// speakerkit's argmax source over the given variant/placement.
fn argmax_extraction(
  root: &Path,
  samples: &[f32],
  variant: ArgmaxVariant,
  seg_cu: ComputeUnits,
  pre_cu: ComputeUnits,
  emb_cu: ComputeUnits,
) -> Extraction {
  let opts = ArgmaxOptions::new().with_variant(variant).with_compute(
    ArgmaxComputeOptions::new()
      .with_segmenter(seg_cu)
      .with_preprocessor(pre_cu)
      .with_embedder(emb_cu),
  );
  ArgmaxSource::from_dir_with(root, opts)
    .expect("load ArgmaxSource")
    .extract(samples)
    .expect("ArgmaxSource::extract")
}

/// The argmax model root, asserted present — never a silent skip. A gate that
/// quietly does not run is the same failure class as a gate that cannot fail.
fn argmax_models_root() -> PathBuf {
  let root = common::argmax_models_dir();
  assert!(
    root.join("speaker_segmenter").exists(),
    "argmax models not found under {} (set ARGMAX_TEST_MODELS)",
    root.display()
  );
  root
}

/// Run `diarize_offline` on an `Extraction` + shared PLDA → its spans as
/// [`Seg`]s. Borrows keep the `Extraction` alive across the call.
fn diarize_extraction_segs(ext: &Extraction, plda: &dia::plda::PldaTransform) -> Vec<Seg> {
  let input = ext.into_offline_input(plda);
  let out = dia::offline::diarize_offline(&input).expect("diarize_offline over speakerkit tensors");
  output_segs(&out)
}

/// Assert speakerkit's sliding-window grid is dia-ort's — the framing half of
/// the input-match proof. A mismatch fabricates DER out of an offset (the
/// alignkit lesson: that is how a fake 86 % divergence once appeared), so this
/// is a hard assert on EVERY source, not a report.
fn assert_grid_matches(name: &str, tag: &str, ext: &Extraction, dia: &DiaOrtRun) {
  assert_eq!(
    ext.num_chunks(),
    dia.num_chunks,
    "{name}/{tag}: grid num_chunks mismatch (speakerkit {} vs dia-ort {}) — framing diverged",
    ext.num_chunks(),
    dia.num_chunks
  );
  assert_eq!(
    ext.num_output_frames(),
    dia.num_output_frames,
    "{name}/{tag}: grid num_output_frames mismatch (speakerkit {} vs dia-ort {}) — framing \
     diverged",
    ext.num_output_frames(),
    dia.num_output_frames
  );
}

// ══════════════════════════════════════════════════════════════════════
// `der_std`'s overlap-skip, hand-verified (no models — runs in the normal
// suite). The DER calc's own unit tests live with the calc, in `der_calc`,
// and run in this binary too.
// ══════════════════════════════════════════════════════════════════════

/// The reference used by the three [`der_std`] unit tests below: speaker 0
/// talks for `[0,10)` and speaker 1 interjects over the top of them for
/// `[4,6)`. On the 10 ms grid that is 1000 frames, of which frames 400..=599
/// (centres in `[4,6)`) carry TWO reference speakers.
#[cfg(test)]
fn overlapping_reference() -> Vec<Seg> {
  vec![
    Seg {
      start: 0.0,
      end: 10.0,
      spk: 0,
    },
    Seg {
      start: 4.0,
      end: 6.0,
      spk: 1,
    },
  ]
}

/// `der_std`'s overlap-skip, hand-verified: an error that happens ONLY on
/// overlapped reference frames is not scored at all.
///
/// Hypothesis hears speaker 0 throughout and misses the interjection entirely.
///
/// - **Strict** (no collar, overlap scored): all 1000 frames score.
///   `ref_units = 800·1 + 200·2 = 1200`; the 200 overlap frames each miss one
///   of their two speakers ⇒ `miss_units = 200` ⇒ DER = 200/1200 = 1/6.
/// - **Standard** (`der_std`, the GATE): the 200 overlap frames are dropped,
///   and so are the collar frames — 0.25 s each side of the reference
///   boundaries at 0, 4, 6 and 10 s, i.e. frames 0..=24, 375..=424, 575..=624
///   and 975..=999. Union with the overlap = 300 dropped, leaving exactly 700
///   scored frames, each with one reference speaker ⇒ `ref_units = 700`,
///   zero error ⇒ DER = 0.
///
/// This is the branch every gating number in this suite flows through, and it
/// removes 2-8 % of reference speech on the real clips.
#[test]
fn der_std_forgives_error_on_overlapped_reference_frames() {
  let reference = overlapping_reference();
  let hypothesis = vec![Seg {
    start: 0.0,
    end: 10.0,
    spk: 0,
  }];

  let strict = der_strict(&reference, &hypothesis);
  assert_eq!(strict.scored_frames, 1000, "strict scores every frame");
  assert_eq!(
    strict.ref_units, 1200,
    "800 single + 200 double-speaker frames"
  );
  assert_eq!(
    strict.miss_units, 200,
    "one missed speaker on each overlap frame"
  );
  assert_eq!(strict.fa_units, 0);
  assert_eq!(strict.conf_units, 0);
  assert!(approx(strict.der, 200.0 / 1200.0), "got {}", strict.der);

  let std = der_std(&reference, &hypothesis);
  assert_eq!(std.scored_frames, 700, "1000 − 300 collar/overlap frames");
  assert_eq!(
    std.ref_units, 700,
    "every scored frame has exactly one speaker"
  );
  assert_eq!(std.err_units(), 0, "the only error was on unscored frames");
  assert!(approx(std.der, 0.0), "got {}", std.der);
}

/// `der_std` is not a blanket amnesty: on the SCORED frames it still catches a
/// confusion in full. Same reference; the hypothesis now hands the back half of
/// speaker 0's turn to a different speaker.
///
/// The optimal mapping can claim only one of the two hypothesis speakers for
/// reference speaker 0 (each covers 350 scored frames), so the other's 350
/// frames are pure confusion ⇒ DER = 350/700 = 50 %, with zero miss and zero
/// false alarm.
///
/// That miss = fa = 0, confusion = everything shape is exactly the signature the
/// ArgmaxSource shows on ≥3-speaker audio ([`DerPin`]) — the same speech,
/// attributed to the wrong speaker — which is why no collar can absorb it.
#[test]
fn der_std_still_scores_confusion_on_scored_frames() {
  let reference = overlapping_reference();
  let hypothesis = vec![
    Seg {
      start: 0.0,
      end: 4.0,
      spk: 0,
    },
    Seg {
      start: 4.0,
      end: 10.0,
      spk: 9,
    },
  ];

  let d = der_std(&reference, &hypothesis);
  assert_eq!(d.scored_frames, 700);
  assert_eq!(d.ref_units, 700);
  assert_eq!(
    d.miss_units, 0,
    "the hypothesis speaks on every scored frame"
  );
  assert_eq!(d.fa_units, 0, "and never where the reference is silent");
  assert_eq!(
    d.conf_units, 350,
    "the unmapped speaker's 350 scored frames"
  );
  assert!(approx(d.der, 0.5), "got {}", d.der);
}

/// The overlap-skip keys on the REFERENCE, never the hypothesis — the pyannote
/// definition, and a real subtlety: a hypothesis that hallucinates a second
/// speaker on top of a single-speaker reference must be penalised, not excused.
///
/// Reference: one speaker, `[0,10)`. Hypothesis: the same speaker plus a
/// phantom second one over the identical span. The 0.25 s collar drops frames
/// 0..=24 and 975..=999 ⇒ 950 scored frames, each with one reference speaker
/// and TWO hypothesis speakers ⇒ 950 false-alarm units over 950 reference units
/// ⇒ DER = 100 %, all false alarm. If `skip_overlap` looked at the hypothesis,
/// every frame would be dropped and this would score 0 %.
#[test]
fn der_std_skips_only_reference_overlap_not_hypothesis_overlap() {
  let reference = vec![Seg {
    start: 0.0,
    end: 10.0,
    spk: 0,
  }];
  let hypothesis = vec![
    Seg {
      start: 0.0,
      end: 10.0,
      spk: 0,
    },
    Seg {
      start: 0.0,
      end: 10.0,
      spk: 1,
    },
  ];

  let d = der_std(&reference, &hypothesis);
  assert_eq!(
    d.scored_frames, 950,
    "1000 − 50 collar frames; NOT overlap-skipped"
  );
  assert_eq!(d.ref_units, 950);
  assert_eq!(
    d.fa_units, 950,
    "the phantom speaker, on every scored frame"
  );
  assert_eq!(d.miss_units, 0);
  assert_eq!(d.conf_units, 0, "the real speaker is still matched");
  assert!(approx(d.der, 1.0), "got {}", d.der);
}

/// Finding 1, hermetic: [`assert_pinned`] must FAIL when a measurement crosses
/// [`PARITY_DER_MAX`] while staying inside [`DER_PIN_TOL`] of its pin — the
/// silent un-breach the same-side guard exists to stop. No models: it feeds
/// synthetic `(measured, pinned)` pairs straight through the helper, catching
/// the panic (`assert_pinned` aborts on failure).
#[test]
fn assert_pinned_fires_when_a_value_crosses_the_parity_bound() {
  // A pin that BREACHES (clip 12's 0.1191 % fa_vs_dia) and a measurement that
  // drifts back UNDER the 0.1 % bound but stays inside the 0.05 pp band: the
  // magnitude check passes, so only the same-side guard can catch it.
  let breach_pin: f64 = 0.001_191;
  let drifted_under: f64 = 0.000_999; // |Δ| = 0.000192 < DER_PIN_TOL, yet < PARITY_DER_MAX
  assert!((drifted_under - breach_pin).abs() <= DER_PIN_TOL);
  assert!(breach_pin > PARITY_DER_MAX && drifted_under < PARITY_DER_MAX);
  let crossed = std::panic::catch_unwind(|| {
    assert_pinned(
      "synthetic",
      "fa_vs_dia (breach → under)",
      drifted_under,
      breach_pin,
    );
  });
  assert!(
    crossed.is_err(),
    "a measurement crossing PARITY_DER_MAX from above to below, within tolerance, must fail \
     assert_pinned — otherwise the breach silently un-breaches (finding 1)"
  );

  // The mirror: a clean pin (clip 06's 0.0909 %, just under the bound) with a
  // measurement that rises OVER it in-tolerance — the faithful source silently
  // STARTS breaching.
  let clean_pin: f64 = 0.000_909;
  let drifted_over: f64 = 0.001_400; // |Δ| = 0.000491 < DER_PIN_TOL, yet > PARITY_DER_MAX
  assert!((drifted_over - clean_pin).abs() <= DER_PIN_TOL);
  let rose = std::panic::catch_unwind(|| {
    assert_pinned(
      "synthetic",
      "fa_vs_dia (clean → over)",
      drifted_over,
      clean_pin,
    );
  });
  assert!(
    rose.is_err(),
    "a clean clip rising over PARITY_DER_MAX within tolerance must also fire (finding 1)"
  );

  // Control: an in-tolerance drift that stays on the SAME side of the bound must
  // still pass (the guard is not a hair-trigger on the tolerance itself).
  assert_pinned("synthetic", "same side, above", 0.001_150, breach_pin);
  assert_pinned("synthetic", "same side, below", 0.000_800, clean_pin);
}

// ══════════════════════════════════════════════════════════════════════
// The corpus guard — no test may believe a fixture's NAME (spec §5.6)
// ══════════════════════════════════════════════════════════════════════

/// Every fixture is the clip [`FIXTURE_FACTS`] says it is.
///
/// Cheap (parses 14 RTTMs, loads no models) and deliberately separate from the
/// DER gates: if the corpus moves under this suite, this fails FIRST and names
/// the fixture, rather than some DER number silently becoming meaningless.
///
/// It also asserts the property the ≥3-speaker gate depends on — that every
/// [`STRESS_FIXTURES`] clip really does have ≥3 reference speakers. Without
/// that, "we gate on multi-speaker audio" is a claim about filenames, which is
/// precisely how §5.6 happened.
#[test]
#[ignore = "requires the sibling diarization parity fixtures (no models needed)"]
fn fixture_facts_match_the_corpus_on_disk() {
  for f in FIXTURE_FACTS {
    let rttm = reference_rttm_path(f.name);
    assert!(
      rttm.exists(),
      "{}: reference.rttm not found at {} — this suite requires the sibling `diarization` \
       checkout (override with DIA_PARITY_FIXTURES)",
      f.name,
      rttm.display()
    );
    // `reference_segments` is the assertion (count re-derived from the file).
    let segs = reference_segments(f.name);
    println!(
      "[{}] reference: {} speakers, {} turns — {} samples expected",
      f.name,
      distinct_speakers(&segs).len(),
      segs.len(),
      f.samples
    );
  }

  for &name in STRESS_FIXTURES {
    let n = facts(name).ref_speakers;
    assert!(
      n >= 3,
      "{name} is in STRESS_FIXTURES but its reference holds {n} speaker(s). The ≥3-speaker \
       gate would be testing easy audio while claiming otherwise — the exact §5.6 failure."
    );
  }

  // The two clips whose names lie. Pinned so the corpus cannot quietly acquire
  // the speakers their names promise and leave the §5.6 narrative stale.
  assert_eq!(facts("04_three_speaker").ref_speakers, 1);
  assert_eq!(facts("05_four_speaker").ref_speakers, 2);
}

// ══════════════════════════════════════════════════════════════════════
// A — FluidAudio DER parity vs dia-ort + determinism (core T7)
// ══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Models/speakerkit + sibling diarization ONNX/fixtures + ort"]
fn fluidaudio_der_parity_vs_dia_ort_and_determinism() {
  let plda = load_plda();
  // GATED: standard-collar parity DER + the absolute-DER delta (both spec
  // readings of "DER delta ≤ 0.1%"). REPORTED: strict frame-exact jitter.
  let mut worst_parity_std = 0.0_f64;
  let mut worst_abs_delta = 0.0_f64;
  let mut worst_parity_strict = 0.0_f64;

  for fixture in common::FIXTURES {
    let samples = fixture_audio(fixture.name);
    let audio_fnv = common::fnv1a_f32(&samples);
    println!(
      "\n=== [{}] {} samples (fnv1a={}) ===",
      fixture.name,
      samples.len(),
      common::fnv_hex(audio_fnv)
    );

    // (ii) dia's own ort path — the parity oracle.
    let dia = dia_ort_run(&samples, &plda);

    // (i) speakerkit FluidAudio, fp32 embedder, CpuOnly (matched to dia-ort's
    // CPU EP — the fidelity control, spec §5.3).
    let ext = fluidaudio_extraction(
      &samples,
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
      &common::embed_fp32_path(),
    );
    assert_grid_matches(fixture.name, "fluidaudio", &ext, &dia);

    let sk_segs = diarize_extraction_segs(&ext, &plda);

    // ── DETERMINISM: a second CpuOnly run must be bit-identical, tensors AND
    // spans.
    let ext2 = fluidaudio_extraction(
      &samples,
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
      &common::embed_fp32_path(),
    );
    assert!(
      ext == ext2,
      "{}: FluidAudio extraction is non-deterministic on CpuOnly (tensors differ)",
      fixture.name
    );
    let sk_segs2 = diarize_extraction_segs(&ext2, &plda);
    let span_key = |segs: &[Seg]| -> Vec<(usize, u64, u64)> {
      segs
        .iter()
        .map(|s| (s.spk, s.start.to_bits(), (s.end - s.start).to_bits()))
        .collect()
    };
    assert_eq!(
      span_key(&sk_segs),
      span_key(&sk_segs2),
      "{}: FluidAudio spans non-deterministic on CpuOnly",
      fixture.name
    );

    // ── SPEAKER-COUNT decisions identical (speakerkit vs dia-ort).
    let sk_spk = distinct_speakers(&sk_segs);
    let dia_spk = distinct_speakers(&dia.segs);
    println!(
      "[{}] speaker counts: speakerkit={} dia-ort={} (dia num_clusters={})",
      fixture.name,
      sk_spk.len(),
      dia_spk.len(),
      dia.num_clusters
    );
    assert_eq!(
      sk_spk.len(),
      dia_spk.len(),
      "{}: speaker-count decision differs (speakerkit {} vs dia-ort {})",
      fixture.name,
      sk_spk.len(),
      dia_spk.len()
    );

    // ── PARITY DER (dia-ort is the reference; speakerkit the hypothesis).
    // GATE = standard DER; strict is reported (see the module doc's metric
    // note + PARITY_DER_MAX). standard == 0 proves strict's diffs are ALL
    // within a boundary collar, i.e. pure boundary jitter.
    let parity_std = der_std(&dia.segs, &sk_segs);
    let parity_strict = der_strict(&dia.segs, &sk_segs);
    println!(
      "[{}] {}",
      fixture.name,
      fmt_der("PARITY std(0.25) [GATE]", &parity_std)
    );
    println!(
      "[{}] {}",
      fixture.name,
      fmt_der("PARITY strict     [report]", &parity_strict)
    );

    // ── ABSOLUTE DER vs pyannote 4.0.4 (its OUTPUT, not the truth — module
    // doc). The std delta between the two sources is the other reading of "DER
    // delta ≤ 0.1%" (GATE); strict absolute is reported to show both sources
    // carry the SAME small boundary jitter vs the independent reference
    // (speakerkit is not uniquely jittery).
    let reference = reference_segments(fixture.name);
    let abs_sk_std = der_std(&reference, &sk_segs);
    let abs_dia_std = der_std(&reference, &dia.segs);
    let abs_delta = (abs_sk_std.der - abs_dia_std.der).abs();
    println!(
      "[{}] {}",
      fixture.name,
      fmt_der("ABS speakerkit vs pyannote std", &abs_sk_std)
    );
    println!(
      "[{}] {}",
      fixture.name,
      fmt_der("ABS dia-ort    vs pyannote std", &abs_dia_std)
    );
    println!(
      "[{}] {}",
      fixture.name,
      fmt_der(
        "ABS speakerkit vs pyannote strict",
        &der_strict(&reference, &sk_segs)
      )
    );
    println!(
      "[{}] {}",
      fixture.name,
      fmt_der(
        "ABS dia-ort    vs pyannote strict",
        &der_strict(&reference, &dia.segs)
      )
    );
    println!(
      "[{}] abs-DER delta (speakerkit − dia-ort) std = {:+.4}%",
      fixture.name,
      abs_delta * 100.0
    );

    worst_parity_std = worst_parity_std.max(parity_std.der);
    worst_abs_delta = worst_abs_delta.max(abs_delta);
    worst_parity_strict = worst_parity_strict.max(parity_strict.der);
  }

  println!(
    "\nFLUIDAUDIO PARITY GATE: worst standard parity DER = {:.4}% | worst abs-DER \
     delta = {:.4}% (bound {:.4}%) | worst strict parity DER = {:.4}% (report; \
     tripwire {:.4}%)",
    worst_parity_std * 100.0,
    worst_abs_delta * 100.0,
    PARITY_DER_MAX * 100.0,
    worst_parity_strict * 100.0,
    STRICT_JITTER_TRIPWIRE * 100.0,
  );

  // THE GATE (original T7 / spec §6, 0.1% on the STANDARD DER — the metric the
  // spec's "DER" names). Both readings of "DER delta ≤ 0.1%": the parity DER
  // of speakerkit vs dia-ort, and the gap between the two sources' absolute
  // DER vs pyannote. Fails on a genuine clustering divergence (which moves
  // whole spans past the collar, or flips the speaker count — also asserted
  // above); tolerates sub-collar boundary jitter, which the strict metric
  // below still guards against gross regression. Never loosened.
  assert!(
    worst_parity_std <= PARITY_DER_MAX,
    "FluidAudio standard parity DER {:.4}% exceeds the {:.4}% bound — a \
     genuine seg/embed divergence propagated through clustering past the \
     collar. Do NOT loosen; investigate.",
    worst_parity_std * 100.0,
    PARITY_DER_MAX * 100.0
  );
  assert!(
    worst_abs_delta <= PARITY_DER_MAX,
    "FluidAudio vs dia-ort absolute-DER delta {:.4}% exceeds {:.4}% — the two \
     sources diverge in agreement with the pyannote reference. Do NOT \
     loosen; investigate.",
    worst_abs_delta * 100.0,
    PARITY_DER_MAX * 100.0
  );
  // Gross-regression guard on the REPORTED strict metric (NOT the spec bound).
  assert!(
    worst_parity_strict <= STRICT_JITTER_TRIPWIRE,
    "FluidAudio strict frame-exact parity DER {:.4}% exceeds the gross-\
     regression tripwire {:.4}% — this is far past benign boundary jitter and \
     indicates a real seg/embed regression. Investigate (do NOT raise the \
     tripwire to pass).",
    worst_parity_strict * 100.0,
    STRICT_JITTER_TRIPWIRE * 100.0
  );
}

// ══════════════════════════════════════════════════════════════════════
// B — argmax-source DER on the ≤2-speaker clips (§5.4 / §5.6)
// ══════════════════════════════════════════════════════════════════════

/// argmax on the EASY half of the corpus: exact agreement with the faithful
/// sources, pinned ([`DER_PINS`]).
///
/// This test's clean result is not a validation — it is the other half of the
/// §5.6 finding. All four of its clips have ≤2 reference speakers
/// ([`FIXTURE_FACTS`]), and argmax reproduces the faithful sources' spans to
/// 0.0000 % standard-collar DER on every one (collar-scored agreement, not
/// asserted frame-exactness — the pins are all [`der_std`]). Read alone, that
/// says "argmax is fine"; that is the reading T7 made, and it was wrong. Read
/// next to
/// [`stress_10_mrbeast_clean_water_7_speakers`], it says something sharper: the
/// argmax divergence is *invisible* until the clustering decision gets hard,
/// which is why the speaker count of the corpus — not its size, not its
/// duration — is what this gate lives or dies by.
#[test]
#[ignore = "requires Models/argmax-speakerkit + Models/speakerkit + sibling diarization + ort"]
fn argmax_source_der_characterization() {
  let argmax_root = argmax_models_root();
  let plda = load_plda();

  for name in e2e_fixture_names() {
    let samples = fixture_audio(name);
    let reference = reference_segments(name);
    let ref_spk = facts(name).ref_speakers;
    println!(
      "\n=== [{name}] argmax §5.4/§5.6 DER ({} samples, {ref_spk} reference speakers) ===",
      samples.len()
    );
    assert!(
      ref_spk <= 2,
      "{name} has {ref_spk} reference speakers — this part scores the EASY half of the \
       corpus. A ≥3-speaker clip belongs in STRESS_FIXTURES, where its argmax divergence is \
       actually gated."
    );

    // dia-ort oracle + FluidAudio (fp32, CpuOnly) for the side-by-side.
    let dia = dia_ort_run(&samples, &plda);
    let fa_ext = fluidaudio_extraction(
      &samples,
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
      &common::embed_fp32_path(),
    );
    let fa_segs = diarize_extraction_segs(&fa_ext, &plda);

    // argmax Baseline (W32A32 seg / W16A16 embed), CpuOnly — the parity
    // control (spec §5.3/§5.4). Its ~0.94 embedding divergence is the risk
    // under test.
    let ax_ext = argmax_extraction(
      &argmax_root,
      &samples,
      ArgmaxVariant::Baseline,
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
    );
    assert_grid_matches(name, "fluidaudio", &fa_ext, &dia);
    assert_grid_matches(name, "argmax", &ax_ext, &dia);
    let ax_segs = diarize_extraction_segs(&ax_ext, &plda);

    // THE §5.6 CHARACTERIZATION, asserted — the same scoring Part D applies to
    // the ≥3-speaker clips, so the easy and hard halves are directly comparable.
    assert_clip_pins(name, &dia, &fa_segs, &ax_segs, &reference);
  }
}

// ══════════════════════════════════════════════════════════════════════
// C — compute-unit DER: shipping default (All) vs CpuOnly (§5.3)
// ══════════════════════════════════════════════════════════════════════

/// §5.3, asserted for BOTH sources: does running on the shipping default
/// (`All` — the ANE gets first pick) change a diarization DECISION, or only
/// jitter span boundaries that DER absorbs?
///
/// The verdict (ΔDER = 0.0000 %, speaker counts invariant) stands; what changed
/// here is that it is now ASSERTED rather than printed, and asserted for the
/// argmax source too — which previously ran its half of the study with no
/// assertion at all, i.e. it could have diverged arbitrarily and still passed.
///
/// The placement really is exercised (not silently falling back to CPU): the
/// strict no-collar jitter between the two placements is NON-zero (0.12-0.29 %),
/// which an identical execution could not produce, while the standard DER
/// against the same reference is identical — so they agree on every scored
/// frame and differ only inside the collar.
#[test]
#[ignore = "requires Models/speakerkit (+ argmax) + sibling diarization + ort; runs the ANE"]
fn compute_unit_der_study_all_vs_cpuonly() {
  let plda = load_plda();
  let argmax_root = argmax_models_root();

  // Finding 4: a per-source WITNESS that the ANE placement was genuinely
  // exercised. Every upper bound below (ΔDER, speaker count, strict tripwire) is
  // satisfied by ZERO jitter — i.e. by both arms silently running the SAME
  // placement (a CPU fallback), which is exactly what would make "All is
  // decision-equivalent to CpuOnly" a vacuous claim (the README says the
  // placement is genuinely exercised; without this, the test does not check it).
  // So the strict no-collar jitter must be NON-zero on at least one pinned
  // fixture per source. Weakest sufficient witness: non-zero err_units on ANY
  // fixture — NOT a magnitude bound — measured 0.12-0.29 % here, four
  // independent chances per source. TRADE-OFF: this is a new gate that would
  // flake if Apple ever made this graph bit-identical across ANE and CPU; the
  // floor is deliberately the weakest thing that still tells "the ANE ran"
  // apart from "it didn't", so it fires only on a true no-op placement.
  let mut strict_jitter_units: BTreeMap<&str, u64> = BTreeMap::new();

  for name in e2e_fixture_names() {
    let samples = fixture_audio(name);
    let reference = reference_segments(name);
    println!(
      "\n=== [{name}] §5.3 compute-unit DER (fp32 embedder held constant, {} samples) ===",
      samples.len()
    );

    // Only the placement varies (precision fixed), so ΔDER isolates the
    // ANE-vs-CPU scheduling drift (spec §5.3).
    let fa_cpu = diarize_extraction_segs(
      &fluidaudio_extraction(
        &samples,
        ComputeUnits::CpuOnly,
        ComputeUnits::CpuOnly,
        &common::embed_fp32_path(),
      ),
      &plda,
    );
    let fa_all = diarize_extraction_segs(
      &fluidaudio_extraction(
        &samples,
        ComputeUnits::All,
        ComputeUnits::All,
        &common::embed_fp32_path(),
      ),
      &plda,
    );
    let ax_cpu = diarize_extraction_segs(
      &argmax_extraction(
        &argmax_root,
        &samples,
        ArgmaxVariant::Baseline,
        ComputeUnits::CpuOnly,
        ComputeUnits::CpuOnly,
        ComputeUnits::CpuOnly,
      ),
      &plda,
    );
    let ax_all = diarize_extraction_segs(
      &argmax_extraction(
        &argmax_root,
        &samples,
        ArgmaxVariant::Baseline,
        ComputeUnits::All,
        ComputeUnits::All,
        ComputeUnits::All,
      ),
      &plda,
    );

    for (tag, cpu, all) in [
      ("fluidaudio", &fa_cpu, &fa_all),
      ("argmax", &ax_cpu, &ax_all),
    ] {
      let cpu_der = der_std(&reference, cpu);
      let all_der = der_std(&reference, all);
      let delta = (all_der.der - cpu_der.der).abs();
      let jitter_full = der_strict(cpu, all);
      let jitter = jitter_full.der;
      // Record the placement-exercised witness for this source (finding 4).
      let w = strict_jitter_units.entry(tag).or_insert(0);
      *w = (*w).max(jitter_full.err_units());
      let (n_cpu, n_all) = (distinct_speakers(cpu).len(), distinct_speakers(all).len());
      println!(
        "[{name}] {}",
        fmt_der(&format!("{tag} CpuOnly vs ref"), &cpu_der)
      );
      println!(
        "[{name}] {}",
        fmt_der(&format!("{tag} All     vs ref"), &all_der)
      );
      println!(
        "[{name}] §5.3 {tag}: ΔDER(All−CpuOnly vs ref) = {:+.4}% (bound {:.4}%) | placement \
         strict jitter (All vs CpuOnly spans) = {:.4}% (tripwire {:.4}%) | speaker counts \
         All={n_all} CpuOnly={n_cpu}",
        (all_der.der - cpu_der.der) * 100.0,
        PARITY_DER_MAX * 100.0,
        jitter * 100.0,
        STRICT_JITTER_TRIPWIRE * 100.0,
      );

      // §5.3 decision 4, in force. The DER-level analogue of the tensor gate's
      // `slot_diffs == 0`: the placement must not add or drop a speaker...
      assert_eq!(
        n_all, n_cpu,
        "{name}/{tag}: compute-unit placement changed the speaker count (All {n_all} vs \
         CpuOnly {n_cpu}) — that is a DECISION change, not the boundary jitter §5.3 accepted. \
         The shipping default does not diarize like the gated configuration."
      );
      // ...nor move a span past the collar (the reference-agreement claim itself)...
      assert!(
        delta <= PARITY_DER_MAX,
        "{name}/{tag}: ΔDER(All − CpuOnly) is {:+.4}%, past the {:.4}% bound — the shipping \
         default is measurably further from the reference than the configuration every fidelity \
         gate pins. Do NOT loosen; investigate.",
        (all_der.der - cpu_der.der) * 100.0,
        PARITY_DER_MAX * 100.0
      );
      // ...and the raw sub-collar drift stays within the same gross-regression
      // tripwire the parity gate uses.
      assert!(
        jitter <= STRICT_JITTER_TRIPWIRE,
        "{name}/{tag}: strict placement jitter {:.4}% exceeds the gross-regression tripwire \
         {:.4}% — far past the sub-collar drift §5.3 measured.",
        jitter * 100.0,
        STRICT_JITTER_TRIPWIRE * 100.0
      );
    }
  }

  // Finding 4: the placement-exercised witness, one per claimed source. A source
  // that showed ZERO strict jitter across EVERY pinned fixture produced
  // bit-identical All and CpuOnly spans — the ANE never diverged from the CPU,
  // so "All is decision-equivalent to CpuOnly" would be vacuously true here (it
  // is only meaningful once All has been observed doing something different).
  // The upper bounds above cannot catch this; only a non-zero floor can. See the
  // accumulator's comment for the (deliberately accepted) flake trade-off.
  for tag in ["fluidaudio", "argmax"] {
    let units = strict_jitter_units.get(tag).copied().unwrap_or(0);
    assert!(
      units > 0,
      "{tag}: strict All-vs-CpuOnly jitter was ZERO on every pinned fixture — the two placements \
       produced bit-identical spans, so ComputeUnits::All never actually diverged from the CPU on \
       this source. This study cannot claim 'All is decision-equivalent to CpuOnly' when it never \
       observed All doing anything different: either the ANE silently fell back to CPU, or the \
       fixtures stopped exercising it. Investigate before trusting the ΔDER=0 result (finding 4)."
    );
  }
}

// ══════════════════════════════════════════════════════════════════════
// D — the ≥3-speaker stress (spec §5.6): where the ArgmaxSource FAILS
// ══════════════════════════════════════════════════════════════════════

/// The ≥3-speaker gate, for one clip.
///
/// Three pipelines over ONE audio buffer — dia-ort (the oracle), FluidAudio and
/// argmax, all `CpuOnly` (the fidelity control: dia-ort runs the ONNX CPU EP, so
/// matching the placement is what isolates the CONVERSION and EMBEDDING axes from
/// the PLACEMENT axis, which is Part C's job). One consequence worth stating
/// plainly rather than burying: this gate therefore does NOT prove anything about
/// the shipping `All` placement on multi-speaker audio. Part C proves `All` is
/// decision-equivalent on ≤2-speaker clips; extending that to ≥3 speakers is
/// inferred, not measured. Recorded in the crate README as an open item.
///
/// Everything is asserted through [`assert_clip_pins`] — the same scoring Part B
/// applies to the ≤2-speaker clips, so the easy and hard halves of the
/// characterization are directly comparable and cannot drift apart.
fn stress_clip(name: &str) {
  let argmax_root = argmax_models_root();
  let plda = load_plda();

  let ref_speakers = facts(name).ref_speakers;
  assert!(
    ref_speakers >= 3,
    "{name}: this gate exists to stress ≥3-speaker clustering; {ref_speakers} is not a stress \
     case"
  );

  // ── ONE audio buffer, shared by every pipeline (the input-match proof starts
  // here: there is only one `samples` variable to feed), its length asserted
  // against FIXTURE_FACTS by `fixture_audio`.
  let samples = fixture_audio(name);
  let reference = reference_segments(name);
  println!(
    "\n═══ [{name}] ≥3-SPEAKER STRESS — {} samples (fnv1a={}), {ref_speakers} reference \
     speakers ═══",
    samples.len(),
    common::fnv_hex(common::fnv1a_f32(&samples)),
  );

  let dia = dia_ort_run(&samples, &plda);
  let fa_ext = fluidaudio_extraction(
    &samples,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuOnly,
    &common::embed_fp32_path(),
  );
  let ax_ext = argmax_extraction(
    &argmax_root,
    &samples,
    ArgmaxVariant::Baseline,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuOnly,
  );

  // Framing: every source built the same grid over the same audio. On these
  // multi-chunk clips (10-24 min) this is the first argmax long-audio grid
  // check, so a framing bug in EITHER source cannot masquerade as a §5.6
  // embedding finding.
  assert_grid_matches(name, "fluidaudio", &fa_ext, &dia);
  assert_grid_matches(name, "argmax", &ax_ext, &dia);
  println!(
    "[{name}] INPUT MATCH: grid num_chunks={} num_output_frames={} identical across dia-ort / \
     fluidaudio / argmax",
    dia.num_chunks, dia.num_output_frames
  );

  let fa_segs = diarize_extraction_segs(&fa_ext, &plda);
  let ax_segs = diarize_extraction_segs(&ax_ext, &plda);

  assert_clip_pins(name, &dia, &fa_segs, &ax_segs, &reference);
}

/// **The counterexample** (7 reference speakers). argmax invents a spurious 8th
/// speaker and scores 3.33 % DER — 33× the spec's 0.1 % bound — where BOTH
/// faithful sources score 0.0000 % standard-collar DER on the same audio. That
/// makes this the cleanest attribution in the corpus: nothing else varies.
/// This single clip is what overturned §5.4's "the ~0.94 embedding divergence
/// is benign" verdict, and it was in the repo the whole time — T7 simply never
/// ran it, because it picked its "multi-speaker" clips by NAME
/// (see [`FIXTURE_FACTS`]).
#[test]
#[ignore = "requires Models/speakerkit + Models/argmax-speakerkit + sibling diarization + ort"]
fn stress_10_mrbeast_clean_water_7_speakers() {
  stress_clip("10_mrbeast_clean_water");
}

/// The MINIMAL multi-speaker case (3 reference speakers), and the only
/// non-MrBeast clip in the stress set — **the clip where argmax HOLDS.**
///
/// It is in the gate precisely because it does not fail: argmax lands 3 speakers
/// and 0.0908 % DER, matching FluidAudio to 0.0037 % (all of it boundary MISS,
/// zero confusion). That is what forbids the tidy rule "argmax breaks at ≥3
/// speakers" — and it is why [`DER_PINS`] says the trigger is NOT isolated. It
/// also means speaker count and recording domain are confounded across this set,
/// since the one clean multi-speaker clip is also the one non-MrBeast one.
/// Pinning a clean result is as load-bearing as pinning a broken one: if argmax
/// ever starts failing here too, the picture changes and this fires.
#[test]
#[ignore = "requires Models/speakerkit + Models/argmax-speakerkit + sibling diarization + ort"]
fn stress_06_long_recording_3_speakers() {
  stress_clip("06_long_recording");
}

/// The richest reference in the corpus (15 speakers) — and the clip that shows
/// the defect is **not** merely "an extra speaker": argmax gets the speaker count
/// exactly RIGHT (15) and still misassigns 3.46 % of speech, all of it confusion.
/// A gate that only watched the speaker count would have called this a pass.
#[test]
#[ignore = "requires Models/speakerkit + Models/argmax-speakerkit + sibling diarization + ort"]
fn stress_12_mrbeast_schools_15_speakers() {
  stress_clip("12_mrbeast_schools");
}

/// Four reference speakers — and argmax's WORST clip by a wide margin: 5 speakers
/// and **9.29 % DER**, ~23× the faithful source on the same audio. It also carries
/// FluidAudio's worst parity (0.3948 %, over [`PARITY_DER_MAX`] — finding 2 in
/// [`DER_PINS`]), which is what makes it the corpus's sharpest clustering cliff:
/// a small perturbation moves assignments, a large one invents a speaker.
#[test]
#[ignore = "requires Models/speakerkit + Models/argmax-speakerkit + sibling diarization + ort"]
fn stress_14_mrbeast_strongman_robot_4_speakers() {
  stress_clip("14_mrbeast_strongman_robot");
}
