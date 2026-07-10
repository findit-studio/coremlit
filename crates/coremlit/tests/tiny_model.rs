//! Integration test against the real `openai_whisper-tiny` mel model
//! (requires `Models/whisperkit-coreml/openai_whisper-tiny`, see Task 8).

mod common;

use coremlit::{ComputeUnits, DataType, Model};

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
