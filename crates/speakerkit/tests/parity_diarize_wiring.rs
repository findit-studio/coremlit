//! Model-gated proof that speakerkit's public runtime clustering entry point,
//! [`speakerkit::extract::Extraction::diarize`], is ONE code path with the
//! manual `into_offline_input → diarize_offline` plumbing the parity harness
//! scores (the alignkit canonical-wiring lesson): SAME extraction, SAME PLDA ⇒
//! identical speaker-labelled spans.
//!
//! # Why this is NOT `dia-oracle`-gated
//!
//! Unlike `parity_e2e` / `parity_shipping_der` (which score against dia's
//! `ort`-backed ONNX oracle and so need `dia-oracle`), this suite runs the
//! entire pipeline — CoreML seg+embed via [`FluidAudioSource`] → diaric's
//! ort-free `diarize_offline` — with NO onnxruntime. That is the point: it
//! demonstrates T1's thesis directly, that offline clustering is a runtime
//! capability of the ort-free build, not a test-only harness step. `ort` never
//! links here.
//!
//! `#[ignore]`d (needs the gitignored `Models/speakerkit` and the sibling
//! `diarization` parity clips). Run with:
//!
//! ```text
//! SPEAKERKIT_TEST_MODELS=… cargo test -p speakerkit --test parity_diarize_wiring -- --ignored --nocapture
//! ```
//!
//! # Clip selection
//!
//! The two clips are chosen by their RTTM-derived distinct-speaker count,
//! asserted at run time and NEVER read off the filename (the T7 misnaming
//! trap — the `NN_something_speaker` names in this corpus lie). One clip is
//! ≤2-speaker (clustering trivial) and one is ≥3-speaker (non-trivial
//! multi-cluster AHC+VBx), so the wiring is proven across both regimes.

mod common;
mod der_calc;

use std::path::PathBuf;

use coremlit::ComputeUnits;
use der_calc::{Seg, distinct_speakers, parse_rttm};
use speakerkit::{
  embed::{EmbedModel, EmbedModelOptions},
  extract::{Extraction, Options},
  segment::{SegmentModel, SegmentModelOptions},
  source::{FluidAudioSource, ModelSource},
};

/// A wiring-proof clip: its corpus basename plus the inclusive distinct-speaker
/// BAND its reference RTTM must fall in. The count is asserted from the parsed
/// RTTM at run time, never inferred from `name`.
struct WiringClip {
  name: &'static str,
  spk_lo: usize,
  spk_hi: usize,
}

/// One ≤2-speaker clip (trivial clustering) and one ≥3-speaker clip (the
/// non-trivial multi-cluster regime). `02_pyannote_sample` is the committed
/// 30 s fixture; `10_mrbeast_clean_water` is the shortest ≥3-speaker clip in
/// the corpus (7 speakers per its RTTM) and is a permanent member of the
/// end-to-end stress gate, so it is known to cluster cleanly on the fp32 path.
const WIRING_CLIPS: &[WiringClip] = &[
  WiringClip {
    name: "02_pyannote_sample",
    spk_lo: 1,
    spk_hi: 2,
  },
  WiringClip {
    name: "10_mrbeast_clean_water",
    spk_lo: 3,
    spk_hi: usize::MAX,
  },
];

/// dia `OfflineOutput` RTTM spans → [`Seg`]s (cluster id is already a 0-indexed
/// integer speaker id) — the observable clustering output this suite compares.
fn output_segs(out: &diaric::offline::OfflineOutput) -> Vec<Seg> {
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

/// The sibling `diarization` checkout's fixture dir for `name` (override root
/// via `DIA_PARITY_FIXTURES`) — same three-levels-up convention as the crate's
/// `dia` path dependency.
fn fixture_dir(name: &str) -> PathBuf {
  let root = std::env::var_os("DIA_PARITY_FIXTURES").map_or_else(
    || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../diarization/tests/parity/fixtures"),
    PathBuf::from,
  );
  root.join(name)
}

/// A fixture's 16 kHz mono audio: the byte-verified committed WAV for the two
/// speakerkit fixtures, else the sibling checkout's `clip_16k.wav`.
fn fixture_audio(name: &str) -> Vec<f32> {
  let path = if common::FIXTURES.iter().any(|f| f.name == name) {
    common::audio_path(name)
  } else {
    fixture_dir(name).join("clip_16k.wav")
  };
  assert!(
    path.exists(),
    "{name}: clip not found at {} — this suite requires the sibling `diarization` checkout \
     (override with DIA_PARITY_FIXTURES)",
    path.display()
  );
  common::load_wav_16k_mono(&path)
}

/// The pyannote reference RTTM's distinct-speaker count for `name`, parsed from
/// the file — the ground for the ≤2 / ≥3 band assertion.
fn rttm_speaker_count(name: &str) -> usize {
  let rttm = fixture_dir(name).join("reference.rttm");
  assert!(
    rttm.exists(),
    "{name}: reference.rttm not found at {}",
    rttm.display()
  );
  distinct_speakers(&parse_rttm(&rttm)).len()
}

/// speakerkit's FluidAudio source over the fp32 embedder (`wespeaker.mlmodelc`
/// — the clean parity path) → the [`Extraction`]. ANE compute: the extraction
/// runs ONCE and feeds both clustering calls, so any ANE nondeterminism cancels
/// and only the deterministic f64 clustering is under test.
fn fluidaudio_extraction(samples: &[f32]) -> Extraction {
  let cu = ComputeUnits::CpuAndNeuralEngine;
  let seg = SegmentModel::from_file_with(
    common::seg_path(),
    SegmentModelOptions::new().with_compute(cu),
  )
  .expect("load pyannote_segmentation.mlmodelc");
  let embed = EmbedModel::from_file_with(
    common::embed_fp32_path(),
    EmbedModelOptions::new().with_compute(cu),
  )
  .expect("load wespeaker.mlmodelc (fp32)");
  FluidAudioSource::with_options(seg, embed, Options::new())
    .extract(samples)
    .expect("FluidAudioSource::extract")
}

/// Prove the public `Extraction::diarize` path equals the manual
/// `into_offline_input → diarize_offline` plumbing on `clip`, asserting the
/// clip's RTTM-derived speaker count lands in its declared band first.
fn prove_wiring(clip: &WiringClip) {
  let counted = rttm_speaker_count(clip.name);
  assert!(
    (clip.spk_lo..=clip.spk_hi).contains(&counted),
    "{}: reference.rttm holds {counted} speakers, outside the required [{}, {}] band — \
     re-pick the clip, do not trust the filename",
    clip.name,
    clip.spk_lo,
    clip.spk_hi,
  );

  let samples = fixture_audio(clip.name);
  let ext = fluidaudio_extraction(&samples);
  let plda = diaric::plda::PldaTransform::new().expect("load community-1 PldaTransform");

  // Subject: the public runtime method.
  let via_public = output_segs(
    &ext
      .diarize(&plda)
      .expect("Extraction::diarize over speakerkit tensors"),
  );
  // Reference: the pre-refactor plumbing, reconstructed through the still-
  // public `into_offline_input` bridge (what the harness used to inline).
  let via_manual = output_segs(
    &diaric::offline::diarize_offline(&ext.into_offline_input(&plda))
      .expect("manual into_offline_input → diarize_offline"),
  );

  assert_eq!(
    via_public.len(),
    via_manual.len(),
    "{}: diarize() produced {} spans, manual plumbing {} — the public path diverged",
    clip.name,
    via_public.len(),
    via_manual.len(),
  );
  for (i, (p, m)) in via_public.iter().zip(&via_manual).enumerate() {
    assert!(
      (p.start - m.start).abs() < f64::EPSILON
        && (p.end - m.end).abs() < f64::EPSILON
        && p.spk == m.spk,
      "{}: span {i} diverged — diarize() = ({:.6},{:.6},spk{}) vs manual = ({:.6},{:.6},spk{}); \
       diarize() is NOT one code path with into_offline_input → diarize_offline",
      clip.name,
      p.start,
      p.end,
      p.spk,
      m.start,
      m.end,
      m.spk,
    );
  }
  eprintln!(
    "[wiring] {}: {counted} ref speakers, {} spans identical via diarize() and manual plumbing",
    clip.name,
    via_public.len(),
  );
}

/// ≤2-speaker regime (trivial clustering).
#[test]
#[ignore = "needs Models/speakerkit + sibling diarization parity clips"]
fn diarize_is_one_code_path_le2_speaker() {
  prove_wiring(&WIRING_CLIPS[0]);
}

/// ≥3-speaker regime (non-trivial multi-cluster AHC+VBx).
#[test]
#[ignore = "needs Models/speakerkit + sibling diarization parity clips"]
fn diarize_is_one_code_path_ge3_speaker() {
  prove_wiring(&WIRING_CLIPS[1]);
}
