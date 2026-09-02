//! End-to-end parity: the Rust door against the PyTorch fp32 reference.
//!
//! `model_io.rs` proves the artifact is the pinned bytes and declares the
//! pinned contract; `src/audio/identity/mel/tests.rs` proves the Rust front end
//! reproduces the checkpoint's own `MelBanks`. Neither of them shows that the
//! two halves compose. This does: for each committed clip it runs
//! `Embedder::embed` on the same 16-bit WAV the goldens were cut from — Rust
//! mel, then the fp16 CoreML graph — and compares the result against the
//! PyTorch fp32 embedding of that clip's golden mel.
//!
//! The reference comes from `conversion/redimnet/scripts/write_mel_goldens.py`,
//! which computes it with `MelToEmbedding.build(model)` — the exact sub-forward
//! `convert_redimnet.py` traces, so this measures the shipped chain against the
//! function that was converted rather than against a restatement of it.
//!
//! Model-gated and `#[ignore]`d; the hermetic checks below run the fixture
//! loader with no model, so a malformed or drifted fixture reds even when the
//! artifact is not staged.

mod common;

use std::path::PathBuf;

use coremlit::{
  ComputeUnits,
  audio::identity::{EMBEDDING_DIM, Embedder, EmbedderOptions, WINDOW_SAMPLES},
};

/// The house floor for a cross-implementation parity claim: `>= 0.99` cosine,
/// the same number `tests/*/placement.rs::SANITY_COS` and
/// `conversion/*/verify_*.py::SANITY_COS_FLOOR` hold, and the same one the
/// recipe's own fp16-vs-fp32 check used (it measured 0.99990 on `CpuAndGpu`).
///
/// Deliberately NOT re-tightened to the recipe's measured value. That number
/// was taken on one machine and one OS version over a synthetic corpus, and
/// fp16 placement numerics are the most host-dependent thing in this crate;
/// pinning the observation as the requirement would convert an OS update into a
/// false failure. The floor is what a correct graph must clear, and a defect of
/// the kind this gate exists for — a transposed mel, a front-end parameter
/// wrong, a graph that ignores its input — lands far below it, not just under
/// it.
const SANITY_COS: f64 = 0.99;

/// One committed clip: its WAV and the PyTorch fp32 reference embedding of its
/// golden mel.
struct GoldenClip {
  id: String,
  wav: PathBuf,
  reference: Vec<f32>,
  reference_norm: f64,
}

fn fixture_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("identity")
    .join("fixtures")
    .join("mel")
}

/// Read `provenance.json` and rebuild the clip table, asserting every field it
/// is about to rely on. The strictness is the point: a fixture that lost its
/// embeddings, or whose provenance stopped naming the artifact this crate
/// pins, must fail here rather than feed a vacuous comparison.
fn load_goldens() -> Vec<GoldenClip> {
  let path = fixture_dir().join("provenance.json");
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  let json: serde_json::Value =
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));

  assert_eq!(
    json["checkpoint"]["sha256"].as_str(),
    Some(common::SOURCE_ASSET_SHA256),
    "the goldens were cut from a different checkpoint than this crate pins"
  );
  assert_eq!(
    json["checkpoint"]["model_source_rev"].as_str(),
    Some(common::SOURCE_CODE_REVISION),
    "the goldens were cut against a different model-source revision"
  );
  assert_eq!(json["window_samples"].as_u64(), Some(WINDOW_SAMPLES as u64));

  let clips = json["clips"].as_array().expect("`clips` array");
  assert!(!clips.is_empty(), "the golden corpus is empty");
  let out: Vec<GoldenClip> = clips
    .iter()
    .map(|clip| {
      let id = clip["id"].as_str().expect("clip id").to_owned();
      let wav = fixture_dir().join(clip["wav"].as_str().expect("clip wav"));
      let reference: Vec<f32> = clip["embedding"]
        .as_array()
        .unwrap_or_else(|| panic!("{id}: no reference embedding"))
        .iter()
        .map(|v| v.as_f64().expect("finite reference component") as f32)
        .collect();
      assert_eq!(reference.len(), EMBEDDING_DIM, "{id}: reference width");
      assert!(
        reference.iter().all(|v| v.is_finite()),
        "{id}: non-finite reference component"
      );
      let reference_norm = clip["embedding_l2_norm"]
        .as_f64()
        .unwrap_or_else(|| panic!("{id}: no reference norm"));
      GoldenClip {
        id,
        wav,
        reference,
        reference_norm,
      }
    })
    .collect();
  out
}

/// Read a committed golden clip, asserting its 16 kHz mono 16-bit header and
/// exact length — a fixture at the wrong rate would still decode to numbers.
fn read_golden_wav(path: &std::path::Path) -> Vec<f32> {
  let mut reader =
    hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 16_000, "{}", path.display());
  assert_eq!(spec.channels, 1, "{}", path.display());
  assert_eq!(spec.bits_per_sample, 16, "{}", path.display());
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("decode sample")) / 32_768.0)
    .collect();
  assert_eq!(samples.len(), WINDOW_SAMPLES, "{}", path.display());
  samples
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The fixture set loads, is non-trivial, and its references are RAW — the same
/// claim `embeddings_are_raw_and_not_unit_norm` makes of the model, made here
/// of the oracle those model outputs are compared against. If the reference
/// were unit-norm the parity gate would still pass while measuring the wrong
/// function.
#[test]
fn the_golden_corpus_loads_and_its_references_are_raw() {
  let clips = load_goldens();
  assert_eq!(clips.len(), 3, "three committed clips");
  for clip in &clips {
    assert!(clip.wav.is_file(), "{}: wav missing", clip.id);
    let measured = clip
      .reference
      .iter()
      .map(|v| f64::from(*v) * f64::from(*v))
      .sum::<f64>()
      .sqrt();
    assert!(
      (measured - clip.reference_norm).abs() < 1e-3,
      "{}: recorded norm {} but the vector measures {measured}",
      clip.id,
      clip.reference_norm
    );
    assert!(
      measured > 2.0,
      "{}: the reference must be RAW; norm {measured:.4}",
      clip.id
    );
  }
}

/// The three references are distinct directions, so the parity comparison below
/// is discriminating rather than satisfied by any vector at all. Two clips that
/// happened to share a direction would let a constant-output graph pass.
#[test]
fn the_golden_references_are_distinct_directions() {
  let clips = load_goldens();
  for (i, a) in clips.iter().enumerate() {
    for b in clips.iter().skip(i + 1) {
      let cos = common::cosine(&a.reference, &b.reference);
      assert!(
        cos < SANITY_COS,
        "{} and {} embed to the same direction (cos {cos:.6}); the parity gate below \
         would not be able to tell them apart",
        a.id,
        b.id
      );
    }
  }
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// **The end-to-end gate.** Rust mel plus fp16 CoreML, against PyTorch fp32,
/// on every compute placement — including the two the module does NOT default
/// to, so a caller who chooses one is choosing between measured arms.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn door_matches_the_pytorch_reference_on_every_placement() {
  let clips = load_goldens();
  for compute in [
    ComputeUnits::All,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
  ] {
    let embedder = Embedder::load(
      common::model_path(),
      EmbedderOptions::new().with_compute(compute),
    )
    .unwrap_or_else(|e| panic!("load under {compute:?}: {e}"));

    let mut worst = 1.0f64;
    for clip in &clips {
      let samples = read_golden_wav(&clip.wav);
      let raw = embedder
        .embed(&samples)
        .unwrap_or_else(|e| panic!("{} under {compute:?}: {e}", clip.id));
      let cos = common::cosine(&raw, &clip.reference);
      eprintln!("[identity] {compute:?} {}: cos = {cos:.8}", clip.id);
      assert!(
        cos >= SANITY_COS,
        "{} under {compute:?}: cos {cos:.8} < {SANITY_COS}",
        clip.id
      );
      worst = worst.min(cos);
    }
    eprintln!("[identity] {compute:?} worst cos = {worst:.8}");
  }
}

/// The cross-clip GEOMETRY survives too, not just each vector separately.
///
/// A front end wrong in a way that affects all three clips the same way could
/// still clear the per-clip floor while collapsing the corpus onto one
/// direction. So the pairwise cosines the door produces are compared against
/// the pairwise cosines PyTorch produces, which is the check the conversion
/// recipe's own `verify_redimnet.py` makes for the same reason.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn cross_clip_geometry_matches_the_reference() {
  const MAX_PAIR_DELTA: f64 = 1e-2;

  let clips = load_goldens();
  let embedder = Embedder::from_file(common::model_path()).expect("load embedder");
  let ours: Vec<[f32; EMBEDDING_DIM]> = clips
    .iter()
    .map(|c| embedder.embed(&read_golden_wav(&c.wav)).expect("embed"))
    .collect();

  for (i, a) in clips.iter().enumerate() {
    for (j, b) in clips.iter().enumerate().skip(i + 1) {
      let theirs = common::cosine(&a.reference, &b.reference);
      let mine = common::cosine(&ours[i], &ours[j]);
      let delta = (mine - theirs).abs();
      eprintln!(
        "[identity] pair {}/{}: reference {theirs:.8}, door {mine:.8}, delta {delta:.3e}",
        a.id, b.id
      );
      assert!(
        delta <= MAX_PAIR_DELTA,
        "{}/{}: cross-clip cosine moved by {delta:.3e} (reference {theirs:.8}, door {mine:.8})",
        a.id,
        b.id
      );
    }
  }
}
