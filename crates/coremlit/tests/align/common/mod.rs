use std::path::{Path, PathBuf};

/// Directory containing the downloaded alignkit model artifacts.
///
/// Overridable via `ALIGNKIT_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/alignkit` — gitignored, fetched dev-time (mirrors
/// whisperkit's `WHISPERKIT_TEST_MODELS`/`Models/` and dia-coreml's
/// `DIA_COREML_TEST_MODELS`/`Models/dia-coreml` conventions, one directory
/// level down for this crate's own model set).
pub fn models_dir() -> PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("alignkit")
    },
    PathBuf::from,
  )
}

/// Path to the compiled forced-aligner artifact.
///
/// Compiled from the downloaded `base960h_aligner.mlpackage` via `xcrun
/// coremlcompiler compile` at model-acquisition time (`coremlit::Model::load`
/// only accepts a compiled `.mlmodelc`; see `tests/model_io.rs`'s module doc
/// for the full acquisition record: source, revision, licence, per-file
/// SHA-256).
pub fn model_path() -> PathBuf {
  models_dir().join("base960h_aligner.mlmodelc")
}

/// Path to the 60 s @ 16 kHz mono fixture used by the graph-truth test and
/// by `tests/parity_words.rs`'s **unpadded** half.
///
/// alignkit has no committed audio fixtures of its own. `ted_60.wav` in the
/// whisperkit crate's `tests/fixtures/audio/` is already exactly 960,000
/// samples (60.000000 s @ 16 kHz mono int16, `afinfo`-verified at write
/// time) — precisely the `[1, 960000]` window `base960h_aligner.mlmodelc`
/// requires, with no padding needed — so this crate borrows it by relative
/// path instead of committing a second copy of a ~1.9 MB binary fixture that
/// would then need to stay byte-identical to the original forever. Both
/// crates live in this workspace and move together.
///
/// That exact-window property is the whole reason the parity gate wants this
/// clip: see [`TED_60_TRANSCRIPT`].
///
/// `#[allow(dead_code)]`: only `tests/model_io.rs` and `tests/parity_words.rs`
/// use it; the per-binary `common` copy in `tests/align_chunk.rs` does not.
#[allow(dead_code)]
pub fn ted_60_wav_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/whisper/fixtures/audio/ted_60.wav")
}

/// The verified transcript for [`ted_60_wav_path`]'s audio — the opening 60 s
/// of Tim Urban's TED talk *Inside the mind of a master procrastinator*.
///
/// # Why this clip has a transcript at all
///
/// `jfk.wav` is 176,000 samples; the encoder window is 960,000, so alignkit
/// zero-pads it by 81.7% and the **unpadded path has never been gated**.
/// `ted_60.wav` is exactly 960,000 samples, so
/// `Encoder::emissions_raw` takes its `Cow::Borrowed` branch — *no zeros are
/// ever appended* — and `truncated_frame_count(960_000, 2999)` keeps all 2,999
/// frames instead of jfk's 549. Forced alignment needs a transcript, and this
/// clip shipped without one, which is exactly why the gap survived B5.
///
/// # Provenance: ASR, because ASR is what feeds forced alignment
///
/// Produced by **this workspace's own whisperkit** — the real production
/// pipeline, ASR → forced alignment — via
/// `cargo run -p coremlit --example whisper_transcribe_wav`, on
/// `openai_whisper-large-v3` (`argmaxinc/whisperkit-coreml`). It is therefore
/// the *kind* of text a caller actually aligns: readable ASR output, not a
/// hand-made verbatim transcription.
///
/// # Verification — three sources, and the ASR lost twice
///
/// A transcript is an **input** to forced alignment: a wrong word does not
/// fail, it silently becomes a wrong alignment target. So this text was
/// cross-checked against two independent readings of the same audio —
/// whisper-small (the same pipeline, a weaker checkpoint) and a **greedy CTC
/// decode of alignkit's own wav2vec2 emissions**, which is the acoustic model
/// that will actually consume it — and every disagreement was settled against
/// the emission posteriors and the RMS envelope, never by vote:
///
/// - **`ninety-page`, not `90-page`** (large-v3's spelling). The 29-class CTC
///   vocabulary has **no digits**, so `90` can only align as out-of-vocabulary
///   wildcards. The greedy decode reads the audio as `NINETY | PAGE` — two
///   words — and [`asry::EnglishNormalizer`] splits on the hyphen, so the
///   spoken form lands as exactly those two words. Spoken form is the correct
///   register for a forced-alignment target.
/// - **`happen to every single paper`**, not whisper-small's `happen in`. The
///   greedy decode emits a bare `T` at 37,520 ms between `HAPPEN` and `EVERY`
///   — the /t/ that `in` does not have. large-v3 agrees.
/// - **`everything gets done and things stay civil`** — the `and` is real,
///   though the greedy decode drops it. Frames 1055–1057 (21,100–21,140 ms)
///   carry `A`/`I` → `N` → `D` posterior mass in sequence beneath a dominant
///   word-delimiter, over a non-zero RMS of 0.081 → 0.062 → 0.021: a reduced,
///   unstressed /ənd/ in the 160 ms gap. Greedy CTC routinely swallows those
///   (it also lost the `may` of `maybe` and the `st` of `stay` right here).
/// - **`I knew for a paper like that`** — large-v3's leading **`And` is a
///   hallucination and is NOT in this transcript.** Frames 2285–2296
///   (45,700–45,920 ms) are digital silence: RMS 0.001–0.003, `logP(blank)`
///   fp16-saturated at exactly `0.0`, and no letter posterior above −8. Speech
///   resumes at 45,920 ms and the model fires `I` at 45,940 ms with a −0.06
///   log-prob — there is neither room nor evidence for an `And`. It is a
///   textbook Whisper segment-initial discourse-marker insertion: large-v3
///   opened a new segment at exactly 45.84 s, right after that pause, and
///   whisper-small — which did not break a segment there — never wrote it.
///
/// The clip's last word is complete: the greedy decode closes `IT` at
/// 59,980 ms, 20 ms inside the 60,000 ms edge.
///
/// # What is deliberately NOT here
///
/// Whisper elides disfluencies, and two survive in the audio: a false start
/// (`I would`… ~27,520–28,220 ms) before `I would have it all ready to go`,
/// and a `would would` repetition near 31,800 ms. They are **left out on
/// purpose** — this is ASR output, which is what production feeds an aligner,
/// and *both* aligners receive the identical text, so the omission cannot bias
/// the comparison. It does leave real speech with no transcript word under it,
/// which makes those two spots the natural places for the trellis to diverge;
/// `tests/parity_words.rs`'s divergence ledger is where that shows up, and it
/// is pinned rather than tolerated.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub const TED_60_TRANSCRIPT: &str = concat!(
  "So in college, I was a government major, which means I had to write a lot of papers. ",
  "Now, when a normal student writes a paper, they might spread the work out a little like this. ",
  "So, you know, you get started maybe a little slowly, but you get enough done in the first week ",
  "that with some heavier days later on, everything gets done and things stay civil. ",
  "And I would want to do that like that. That would be the plan. ",
  "I would have it all ready to go, but then actually the paper would come along, ",
  "and then I would kind of do this. ",
  "And that would happen to every single paper. ",
  "But then came my ninety-page senior thesis, a paper you're supposed to spend a year on. ",
  "I knew for a paper like that, my normal workflow was not an option. ",
  "It was way too big a project. ",
  "So I planned things out, and I decided I kind of had to go something like this. ",
  "This is how the year would go. So I'd start off light and I'd bump it",
);

/// SHA-256 of [`ted_60_wav_path`]'s **decoded** buffer — the 960,000 f32
/// samples [`load_wav_mono_f32`] returns, hashed as little-endian bytes.
/// Exactly [`JFK_SAMPLES_SHA256`]'s role, for the second clip: it pins the
/// audio the gate's ted_60 bounds were measured on, and it is *also* the pin
/// that the clip still fills the window exactly — a re-encode that changed the
/// length by one sample would silently move alignkit onto its zero-padding
/// branch and quietly retire the very path this fixture exists to cover.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub const TED_60_SAMPLES_SHA256: &str =
  "b14ed488eb68545e49893bd424d78a0849941b97c3f042c3e4461e3bfb513dd5";

/// Path to the 11 s @ 16 kHz mono `jfk.wav` fixture (176,000 samples, well
/// inside the encoder's 960,000 window), borrowed from the whisperkit crate
/// exactly as [`ted_60_wav_path`] is. Its known transcript is
/// [`JFK_TRANSCRIPT`]; together they drive `tests/align_chunk.rs`'s
/// end-to-end alignment.
///
/// `#[allow(dead_code)]`: only `tests/align_chunk.rs` uses it.
#[allow(dead_code)]
pub fn jfk_wav_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/whisper/fixtures/audio/jfk.wav")
}

/// The known transcript for [`jfk_wav_path`]'s audio (whisperkit's
/// `tests/fixtures/golden/jfk_tiny_golden.json`).
///
/// `#[allow(dead_code)]`: only `tests/align_chunk.rs` uses it.
#[allow(dead_code)]
pub const JFK_TRANSCRIPT: &str = "And so my fellow Americans ask not what your country can do for \
                                  you, ask what you can do for your country.";

/// SHA-256 of [`jfk_wav_path`]'s **decoded** buffer — the 176,000 f32
/// samples [`load_wav_mono_f32`] returns, hashed as little-endian bytes.
///
/// This is the input-identity pin for `tests/parity_words.rs`. That gate
/// compares alignkit's word timings against asry's ONNX aligner, and such a
/// comparison is worth exactly nothing if the two sides are not looking at
/// the same audio: the FIRST attempt at an alignkit-vs-asry comparison
/// (`.superpowers/sdd/alignkit-gate1-diagnostic.md`) reported an alarming
/// "86.6% divergence" that turned out to be a harness bug — one side got a
/// padded buffer, the other an unpadded one. The number was measuring the
/// harness, not the models.
///
/// The gate feeds one `Vec<f32>`, by reference, to both aligners, so
/// buffer identity holds by construction; this digest additionally pins the
/// FIXTURE, so a `jfk.wav` that is silently re-encoded, resampled, or
/// swapped out from under the cross-crate relative path fails loudly instead
/// of re-measuring parity on different audio.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub const JFK_SAMPLES_SHA256: &str =
  "ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0";

/// Lowercase-hex SHA-256 of a decoded sample buffer, over its little-endian
/// `f32` bytes. Backs [`JFK_SAMPLES_SHA256`].
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn sha256_samples_hex(samples: &[f32]) -> String {
  use sha2::{Digest, Sha256};
  let mut hasher = Sha256::new();
  for sample in samples {
    hasher.update(sample.to_le_bytes());
  }
  hasher
    .finalize()
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Directory holding asry's ONNX wav2vec2 oracle — the `models/` directory of a
/// co-located `asry` checkout (a sibling of this repo). This is TEST DATA, not
/// the code dependency: alignkit depends on asry as a rev-pinned git source
/// (`crates/alignkit/Cargo.toml`), so building the crate does NOT put asry's
/// `models/` on disk. The default path below assumes the dev-worktree layout (a
/// sibling `asry`); set `ALIGNKIT_ASRY_MODELS` when it lives elsewhere.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn asry_models_dir() -> PathBuf {
  std::env::var_os("ALIGNKIT_ASRY_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../asry")
        .join("models")
    },
    PathBuf::from,
  )
}

/// asry's ONNX wav2vec2-base-960h export (`onnx-community/
/// wav2vec2-base-960h-ONNX`, fetched by asry's own `build.rs`). Raw
/// **logits**, 32-class head — the oracle's encoder.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn asry_onnx_model_path() -> PathBuf {
  asry_models_dir().join("wav2vec2-base-960h.onnx")
}

/// The 32-class HuggingFace tokenizer matching [`asry_onnx_model_path`].
/// **Not** alignkit's bundled 29-class chordai asset: each tokenizer belongs
/// to its own CTC head, and asry's `Aligner::from_paths` validates the width.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn asry_tokenizer_path() -> PathBuf {
  asry_models_dir().join("wav2vec2-base-960h-tokenizer.json")
}

/// Reads a 16 kHz mono 16-bit PCM WAV into normalized f32 samples.
///
/// Mirrors whisperkit's `tests/common::load_wav_mono_f32`.
pub fn load_wav_mono_f32(path: &Path) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("fixture wav opens");
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

/// Lowercase-hex SHA-256 digest of a file's contents.
///
/// Backs `tests/model_io.rs`'s provenance/integrity pin over the downloaded
/// model artifacts. `common` is a `mod`, not a separate crate, so each
/// `tests/*.rs` integration-test binary compiles its own copy; binaries
/// that don't happen to call this one (e.g. `tests/align_chunk.rs`)
/// would otherwise warn `dead_code` on it.
#[allow(dead_code)]
pub fn sha256_hex(path: &Path) -> String {
  use sha2::{Digest, Sha256};
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}
