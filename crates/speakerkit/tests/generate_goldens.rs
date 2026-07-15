//! Regenerates the committed dia-ort parity goldens (spec §6 Gates 1-2).
//!
//! This is the *executable, reproducible* oracle: rather than pinning opaque
//! reference blobs, it RUNS dia's own `ort` pipeline — the very
//! `pyannote/segmentation-3.0` + WeSpeaker ResNet34-LM models speakerkit
//! re-implements over CoreML — and writes each fixture's reference tensors to
//! `tests/fixtures/golden/<name>.json`. The parity suites
//! (`tests/parity_seg.rs`, `tests/parity_embed.rs`) then check CoreML against
//! those committed goldens WITHOUT needing dia/ort at all.
//!
//! Gated on the `dia` feature so the default build compiles this file to
//! nothing (it links `ort` + a 27 MB ONNX otherwise). `#[ignore]` so it never
//! runs in an ordinary `cargo test`, AND gated on the `UPDATE_GOLDEN`
//! environment variable so it never rewrites the committed oracle as a *side
//! effect* of a routine `--ignored` sweep (whisperkit's convention, see
//! `crates/whisperkit/tests/parity_jfk.rs`): `cargo test -p speakerkit
//! --features dia -- --ignored` runs every `#[ignore]` test including this one,
//! so without the env guard that standard gate would silently re-baseline the
//! goldens it exists to validate. Without `UPDATE_GOLDEN` set this test is an
//! explicit no-op that touches nothing; regeneration is a deliberate,
//! human-verified act:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p speakerkit --features dia --test generate_goldens -- --ignored --nocapture
//! ```
//!
//! Provisioning (proven working standalone before this harness was wired):
//! - Segmentation ONNX is `dia::segment::SegmentModel::bundled()` — embedded
//!   in the `dia` crate via `include_bytes!` (`bundled-segmentation`, on by
//!   default), no file needed.
//! - Embedding ONNX is the BYO fp32 `wespeaker_resnet34_lm.onnx`; path via
//!   `DIA_EMBED_MODEL_PATH` or the sibling `diarization/models/` checkout.
//! - `ort` self-provisions `libonnxruntime` via its default `download-binaries`
//!   feature (cached at `~/Library/Caches/ort.pyke.io`); no `ORT_DYLIB_PATH`.
//!
//! Both models run on ort's CPU EP (dia registers no execution provider here —
//! `speakerkit`'s `dia` feature enables none of dia's per-EP features), the
//! matched reference for CoreML's own deterministic `CpuOnly` parity runs.
#![cfg(feature = "dia")]

mod common;

use std::io::Write as _;

use dia::{embed::EmbedModel, segment::SegmentModel};
use speakerkit::segment::{POWERSET_CLASSES, SEG_NUM_SLOTS, multilabel};

/// dia's community-1 onset (`diarization/src/offline/owned.rs:144`;
/// speakerkit's `window::DEFAULT_ONSET`). On the hard 0/1 multilabel a slot is
/// active at a frame iff its value is `>= ONSET`.
const ONSET: f64 = 0.5;

/// pyannote's `embedding_exclude_overlap` minimum clean-frame count
/// (`diarization/src/offline/owned.rs:522`; speakerkit's
/// `extract::EXCLUDE_OVERLAP_MIN_FRAMES`): the overlap-excluded mask is used
/// only with STRICTLY more clean frames than this, else the slot falls back to
/// its raw active mask.
const EXCLUDE_OVERLAP_MIN_FRAMES: usize = 2;

/// Resolves the BYO WeSpeaker fp32 ONNX: `DIA_EMBED_MODEL_PATH`, else the
/// sibling `diarization/models/wespeaker_resnet34_lm.onnx` (relative to this
/// crate, mirroring `Cargo.toml`'s `dia` path dependency).
fn wespeaker_onnx_path() -> std::path::PathBuf {
  if let Some(p) = std::env::var_os("DIA_EMBED_MODEL_PATH") {
    return std::path::PathBuf::from(p);
  }
  std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../../../diarization/models/wespeaker_resnet34_lm.onnx")
}

/// Per-slot embedding masks for one chunk, reproducing dia's offline
/// overlap-exclusion (`derive_slot_plans`, speakerkit `extract/mod.rs`, itself
/// a bit-for-bit port of `owned.rs:507-591`) so the reference embeddings use
/// the exact masks dia's pipeline would feed. `None` = the slot has no active
/// frame (dia skips it — no embed call).
fn derive_slot_masks(chunk_segs: &[f64], num_frames: usize) -> [Option<Vec<bool>>; SEG_NUM_SLOTS] {
  assert_eq!(chunk_segs.len(), num_frames * SEG_NUM_SLOTS);

  // Per-frame "clean" indicator: fewer than 2 of the slots active. Computed
  // once over all slots, before the per-slot loop (owned.rs:536-549).
  let mut clean = vec![false; num_frames];
  for (f, clean_f) in clean.iter_mut().enumerate() {
    let active = (0..SEG_NUM_SLOTS)
      .filter(|&s| chunk_segs[f * SEG_NUM_SLOTS + s] >= ONSET)
      .count();
    *clean_f = active < 2;
  }

  core::array::from_fn(|s| {
    let mut frame_mask = vec![false; num_frames];
    let mut any = false;
    for (f, m) in frame_mask.iter_mut().enumerate() {
      *m = chunk_segs[f * SEG_NUM_SLOTS + s] >= ONSET;
      any |= *m;
    }
    if !any {
      return None; // dia drops a slot with no active frame (owned.rs:561-571).
    }
    // Overlap-excluded mask = raw AND clean; fall back to raw when too few
    // clean frames remain (`<=`, owned.rs:573-591).
    let mut used = vec![false; num_frames];
    let mut clean_count = 0usize;
    for (f, u) in used.iter_mut().enumerate() {
      *u = frame_mask[f] && clean[f];
      if *u {
        clean_count += 1;
      }
    }
    if clean_count <= EXCLUDE_OVERLAP_MIN_FRAMES {
      used = frame_mask;
    }
    Some(used)
  })
}

#[test]
#[ignore = "rewrites committed goldens; set UPDATE_GOLDEN=1 + `dia` feature + ort + wespeaker ONNX"]
fn generate_goldens() {
  // WRITE GUARD (whisperkit's `UPDATE_GOLDEN` convention, `parity_jfk.rs`):
  // this test's whole body OVERWRITES the committed golden oracle
  // (`tests/fixtures/golden/*.json`), so it must never fire from a routine
  // `cargo test -p speakerkit --features dia -- --ignored` — that gate runs
  // every `#[ignore]` test, and an unconditional writer here silently
  // re-baselines the very oracle `tests/parity_seg.rs` / `parity_embed.rs`
  // validate against. Unset ⇒ explicit no-op: no models loaded, no files
  // touched, so the standard gate leaves the working tree clean. The freshly
  // computed tensors are gated against the committed goldens by those parity
  // suites; the `seg_model` provenance label by the hermetic
  // `tests/golden_metadata.rs`. Regenerating is deliberate and human-verified.
  if std::env::var_os("UPDATE_GOLDEN").is_none() {
    eprintln!(
      "generate_goldens: UPDATE_GOLDEN unset — no-op (committed goldens left \
       untouched). Set UPDATE_GOLDEN=1 to regenerate, then human-verify the diff."
    );
    return;
  }

  let onnx = wespeaker_onnx_path();
  assert!(
    onnx.exists(),
    "WeSpeaker ONNX not found at {}; set DIA_EMBED_MODEL_PATH or provision \
     diarization/models/wespeaker_resnet34_lm.onnx",
    onnx.display()
  );

  for fixture in common::FIXTURES {
    let wav = common::audio_path(fixture.name);
    let samples = common::load_wav_16k_mono(&wav);
    let chunks = common::chunk_and_pad(&samples);
    println!(
      "[{}] {} samples -> {} chunks",
      fixture.name,
      samples.len(),
      chunks.len()
    );

    // Fresh sessions per fixture (both are `&mut self`).
    let mut seg = SegmentModel::bundled().expect("load bundled segmentation ONNX");
    let mut embed = EmbedModel::from_file(&onnx).expect("load WeSpeaker fp32 ONNX");

    let mut num_frames_seen: Option<usize> = None;
    let mut chunk_values: Vec<serde_json::Value> = Vec::with_capacity(chunks.len());

    for (c, chunk) in chunks.iter().enumerate() {
      // dia-ort segmentation: [num_frames * 7] powerset LOG-PROBABILITIES,
      // not raw logits (this comment claimed "no softmax" until the values
      // were checked: every element is <= 0 and each 7-class row satisfies
      // sum(exp(row)) == 1.0). The CoreML side emits the same quantity —
      // its MIL ends `softmax` -> `log` — so `parity_seg.rs` compares like
      // with like. The `logits`/`seg_logits` names are kept to avoid
      // churning every committed golden; the values are log-probs.
      let logits = seg.infer(chunk).expect("dia-ort segmentation infer");
      assert_eq!(
        logits.len() % POWERSET_CLASSES,
        0,
        "seg logits not a multiple of POWERSET_CLASSES"
      );
      let num_frames = logits.len() / POWERSET_CLASSES;
      num_frames_seen = Some(*num_frames_seen.get_or_insert(num_frames));
      assert_eq!(num_frames_seen, Some(num_frames), "frame count drift");

      // Hard multilabel (speakerkit's decode == dia's softmax+argmax) → the
      // per-slot overlap-excluded masks dia's pipeline feeds to embed.
      let slab = multilabel(&logits, num_frames);
      let masks = derive_slot_masks(&slab, num_frames);

      let mut slot_values: Vec<serde_json::Value> = Vec::new();
      for (s, mask) in masks.iter().enumerate() {
        let Some(mask) = mask else { continue };
        // dia-ort embedding: full chunk → Rust kaldi-fbank → ONNX weighted
        // pooling with this mask → raw 256-d embedding (spec §2.4 cross-fbank).
        let emb = embed
          .embed_chunk_with_frame_mask(chunk, mask)
          .expect("dia-ort embedding infer");
        slot_values.push(serde_json::json!({
          "slot": s,
          "mask": common::mask_to_string(mask),
          "embedding": emb.to_vec(),
        }));
      }

      println!(
        "  chunk {c}: {num_frames} frames, {} embedded slot(s)",
        slot_values.len()
      );

      chunk_values.push(serde_json::json!({
        "input_len": chunk.len(),
        "input_fnv1a": common::fnv_hex(common::fnv1a_f32(chunk)),
        "seg_logits": logits,
        "slots": slot_values,
      }));
    }

    let golden = serde_json::json!({
      "fixture": fixture.name,
      "source": fixture.source,
      "wav_sha256": fixture.sha256,
      "sample_count": samples.len(),
      // Legacy provenance label, pinned in `common` and kept verbatim to match
      // the committed oracle (the values are log-probabilities — see
      // `common::SEG_MODEL_LABEL` and `speakerkit::segment`'s module doc).
      "seg_model": common::SEG_MODEL_LABEL,
      "embed_model": "wespeaker_resnet34_lm.onnx (dia BYO, ort CPU EP, fp32)",
      "onset": ONSET,
      "num_chunks": chunks.len(),
      "num_frames": num_frames_seen.expect("at least one chunk"),
      "powerset_classes": POWERSET_CLASSES,
      "chunks": chunk_values,
    });

    let path = common::golden_path(fixture.name);
    std::fs::create_dir_all(path.parent().unwrap()).expect("create golden dir");
    let mut f = std::fs::File::create(&path).expect("create golden file");
    serde_json::to_writer(&mut f, &golden).expect("write golden json");
    f.write_all(b"\n").expect("trailing newline");
    println!("[{}] wrote {}", fixture.name, path.display());
  }
}
