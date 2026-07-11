//! Integration test against the real `openai_whisper-tiny` mel model
//! (requires `Models/whisperkit-coreml/openai_whisper-tiny`, see Task 8).

mod common;

use coremlit::{ComputeUnits, DataType, Features, Model, MultiArray};

#[test]
#[ignore = "requires local tiny model (see plan Task 8 Step 1)"]
fn loads_mel_model_and_reads_description() {
  let model = Model::load(
    common::tiny_dir().join("MelSpectrogram.mlmodelc"),
    ComputeUnits::CpuOnly,
  )
  .unwrap();
  let description = model.description();

  let input = description
    .input("audio")
    .expect("mel input feature `audio`");
  assert_eq!(input.shape(), &[480_000]);

  let output = description
    .output("melspectrogram_features")
    .expect("mel output feature `melspectrogram_features`");
  assert_eq!(output.shape(), &[1, 80, 1, 3000]);
  assert_eq!(output.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local tiny model (see plan Task 8 Step 1)"]
fn mel_predict_produces_f16_spectrogram() {
  let model = Model::load(
    common::tiny_dir().join("MelSpectrogram.mlmodelc"),
    ComputeUnits::CpuOnly,
  )
  .unwrap();
  let audio = MultiArray::zeros(&[480_000], DataType::F32).unwrap();
  let outputs = model
    .predict(&Features::new().with("audio", audio))
    .unwrap();
  let mel = outputs
    .get("melspectrogram_features")
    .expect("mel output present");
  assert_eq!(mel.shape(), vec![1, 80, 1, 3000]);
  assert_eq!(mel.data_type(), DataType::F16);
  // Log-mel of silence is a constant floor: finite and uniform-ish. On real
  // hardware CoreML backs this model's output with an IOSurface-aligned
  // buffer (width 3000 padded to a stride of 3008 elements, i.e. the 6000
  // logical bytes per row rounded up to the next 128-byte boundary) — the
  // same ANE-friendly layout `MultiArray::f16_surface` documents for
  // inputs. `copy_into` is stride-aware, so — unlike `as_slice`, which
  // refuses padded buffers outright — it gathers every logical element
  // correctly whether or not CoreML happens to hand back a dense buffer.
  let mut values = vec![half::f16::from_f32(0.0); mel.count()];
  mel.copy_into(&mut values).unwrap();
  assert!(values.iter().all(|v| v.to_f32().is_finite()));
}

#[test]
#[ignore = "requires local tiny model (see plan Task 8 Step 1)"]
fn prewarm_loads_and_drops() {
  coremlit::Model::prewarm(
    common::tiny_dir().join("MelSpectrogram.mlmodelc"),
    ComputeUnits::CpuOnly,
  )
  .unwrap();
}

#[test]
#[ignore = "requires local tiny model (see plan Task 8 Step 1)"]
fn stateless_model_accepts_state_prediction() {
  let model = Model::load(
    common::tiny_dir().join("MelSpectrogram.mlmodelc"),
    ComputeUnits::CpuOnly,
  )
  .unwrap();
  if !model.supports_state() {
    eprintln!("skipping: MLState unavailable on this OS");
    return;
  }
  let mut state = model.make_state().unwrap();
  let stateful_audio = MultiArray::zeros(&[480_000], DataType::F32).unwrap();
  let stateful_outputs = model
    .predict_with_state(&Features::new().with("audio", stateful_audio), &mut state)
    .unwrap();
  let stateful_mel = stateful_outputs.get("melspectrogram_features").unwrap();
  assert_eq!(stateful_mel.shape(), vec![1, 80, 1, 3000]);

  let plain_audio = MultiArray::zeros(&[480_000], DataType::F32).unwrap();
  let plain_outputs = model
    .predict(&Features::new().with("audio", plain_audio))
    .unwrap();
  let plain_mel = plain_outputs.get("melspectrogram_features").unwrap();

  let mut stateful_values = vec![half::f16::from_f32(0.0); stateful_mel.count()];
  stateful_mel.copy_into(&mut stateful_values).unwrap();
  let mut plain_values = vec![half::f16::from_f32(0.0); plain_mel.count()];
  plain_mel.copy_into(&mut plain_values).unwrap();

  // MelSpectrogram declares no state buffers, so `newState()` yields an
  // empty MLState and stateful prediction must behave identically to plain
  // `predict` — the invariant `Model::make_state`'s doc promises, now
  // checked value-for-value instead of shape-only.
  assert_eq!(stateful_values, plain_values);
}

#[test]
#[ignore = "requires local tiny model (see plan Task 8 Step 1)"]
fn loads_through_non_ascii_symlinked_path() {
  // The exact-bytes path contract: a non-ASCII (multi-byte UTF-8) component
  // must reach CoreML unaltered. (A lossy conversion would survive valid
  // UTF-8 too, but this pins the filesystem-representation route end to
  // end; APFS enforces UTF-8, so invalid bytes are not constructible here.)
  let link_dir = std::env::temp_dir().join("coremlit-tests");
  std::fs::create_dir_all(&link_dir).unwrap();
  let link = link_dir.join("modèle-mel.mlmodelc");
  let _ = std::fs::remove_file(&link);
  std::os::unix::fs::symlink(common::tiny_dir().join("MelSpectrogram.mlmodelc"), &link).unwrap();
  let model = Model::load(&link, ComputeUnits::CpuOnly).unwrap();
  assert!(model.description().input("audio").is_some());
  let _ = std::fs::remove_file(&link);
}
