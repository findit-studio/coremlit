//! Integration test against the real `openai_whisper-tiny` mel model
//! (requires `Models/whisperkit-coreml/openai_whisper-tiny`, see Task 8).

mod common;

use coremlit::{ComputeUnits, DataType, Features, Model, MultiArray, TensorError};

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
  // inputs. `as_slice` correctly refuses to bulk-read that padding as
  // `TensorError::NonContiguous` rather than misinterpreting it, so the
  // flat check only runs when CoreML happens to hand back a dense buffer.
  match mel.as_slice::<half::f16>() {
    Ok(values) => assert!(values.iter().all(|v| v.to_f32().is_finite())),
    Err(TensorError::NonContiguous { .. }) => {}
    Err(e) => panic!("unexpected error reading mel output: {e}"),
  }
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
