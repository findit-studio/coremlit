use std::path::PathBuf;

pub fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  )
}

pub fn tiny_dir() -> PathBuf {
  models_dir()
    .join("whisperkit-coreml")
    .join("openai_whisper-tiny")
}

// `tests/common/mod.rs` is compiled fresh into each integration-test
// binary that declares `mod common;`; not every binary uses every helper.
// Most do need a tokenizer path (anything that builds a `WhisperKit` via
// `Options::new`), but `model_io.rs` drives `Model::load` directly with no
// tokenizer involved, so an unused-in-THAT-binary helper is expected here,
// not a real dead-code bug.
#[allow(dead_code)]
pub fn tokenizer_dir() -> PathBuf {
  models_dir().join("tokenizers").join("whisper-tiny")
}

pub fn fixtures_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("whisper")
    .join("fixtures")
}

/// Reads a 16 kHz mono 16-bit PCM WAV into normalized f32 samples.
///
/// All three committed fixtures (`jfk.wav`, `es_test_clip.wav`,
/// `ja_test_clip.wav`) are already 16 kHz mono 16-bit PCM as copied from
/// `argmax-oss-swift` (`afinfo`-verified at plan time: jfk 11.000s /
/// 176,000 samples, es_test_clip 7.664562s / 122,633 samples, ja_test_clip
/// 2.773s / 44,368 samples) — no `afconvert` resampling was needed for any
/// of them, though only `jfk.wav`'s sample count is asserted below.
pub fn load_wav_mono_f32(path: &std::path::Path) -> Vec<f32> {
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

// ---------------------------------------------------------------------
// Golden token parity + first-divergence diagnostic
// ---------------------------------------------------------------------

/// Asserts `rust` matches `golden` token for token; on a mismatch, panics
/// with a diagnosis of the **first diverging decode step** — the two
/// competing token ids, their raw decoder logits, and the step's top-1 /
/// top-2 margin.
///
/// This is a diagnostic, not a tolerance. It changes no pass/fail verdict:
/// identical streams pass, divergent ones fail, exactly as `assert_eq!`
/// would. What it adds is *why*.
///
/// The goldens are an external Swift oracle (`whisperkit-cli @
/// argmax-oss-swift`) captured on the Neural Engine, and this pipeline
/// decodes **greedily** — so one borderline argmax flipped by ANE fp16 drift
/// on a different Apple Silicon generation cascades through every token
/// after it. A bare `assert_eq!` on two 30-element vectors cannot tell that
/// apart from a real pipeline bug; the margin at the first divergence can.
/// On `openai_whisper-tiny` the two thinnest steps of the jfk decode sit at
/// margins of 0.1562 and 0.2500 against a worst observed cross-placement
/// logit delta of ~1.0 — i.e. a flip is *possible* on other silicon, and
/// this is what makes it legible when it happens.
///
/// `audio` is the clip the tokens were decoded from. The diagnostic replays
/// the decode against the **shipping** compute units and teacher-forces the
/// golden prefix, so `golden[k]` is the token fed at cache position `k` —
/// see [`replay_step_logits`] for the invariant that makes this exact, and
/// the guard that refuses the replay when it would not be.
#[allow(dead_code)]
pub fn assert_golden_tokens(label: &str, rust: &[u32], golden: &[u32], audio: &[f32]) {
  if rust == golden {
    return;
  }

  let first_diff = rust
    .iter()
    .zip(golden)
    .position(|(ours, gold)| ours != gold)
    .unwrap_or_else(|| rust.len().min(golden.len()));

  let mut report = format!(
    "GOLDEN TOKEN MISMATCH [{label}]\n\
     \x20 golden stream: {} tokens; ours: {} tokens\n\
     \x20 first divergence at token index {first_diff}\n",
    golden.len(),
    rust.len(),
  );

  match (rust.get(first_diff), golden.get(first_diff)) {
    (Some(&ours), Some(&gold)) => {
      // Step k feeds tokens[k] at cache position k and predicts tokens[k+1]
      // (`decode::decode_text`'s loop), so the step that produced the token
      // at `first_diff` is `first_diff - 1`.
      let Some(step) = first_diff.checked_sub(1) else {
        report.push_str(
          "  ...at index 0 — the start-of-transcript token itself. That is a \
           prefill/prompt-construction bug, not a sampled-token flip; no decode \
           step produced it, so there is no margin to report.\n",
        );
        panic!("{report}");
      };
      report.push_str(&format!(
        "  produced by decode step {step} (fed golden token {})\n\
         \x20 ours:   {ours}\n\
         \x20 golden: {gold}\n",
        golden[step],
      ));

      match replay_step_logits(audio, &golden[..=step]) {
        Ok(logits) => {
          let ours_logit = logits.get(ours as usize).copied().unwrap_or(f32::NAN);
          let gold_logit = logits.get(gold as usize).copied().unwrap_or(f32::NAN);
          let ((top1, top1_logit), (top2, top2_logit)) = top_two(&logits);
          report.push_str(&format!(
            "\n  raw decoder logits at step {step}\n\
             \x20   ours   {ours:>6}: {ours_logit:>10.4}\n\
             \x20   golden {gold:>6}: {gold_logit:>10.4}\n\
             \x20   MARGIN (ours - golden): {:>+.4}\n\
             \x20   raw top-1 {top1:>6}: {top1_logit:>10.4}\n\
             \x20   raw top-2 {top2:>6}: {top2_logit:>10.4}\n\
             \x20   MARGIN (top1 - top2):   {:>+.4}\n",
            ours_logit - gold_logit,
            top1_logit - top2_logit,
          ));
          report.push_str(
            "\n  A THIN margin here (order 0.1-1.0) means the two machines \
             disagreed on a\n  BORDERLINE ARGMAX: ANE fp16 drift on a different \
             Apple Silicon generation can\n  flip it, and greedy autoregression \
             then cascades the flip through every token\n  after this one. \
             Suspect hardware drift before a pipeline logic bug. A WIDE\n  margin \
             (many logits apart) is the opposite: the model was not close to \
             agreeing\n  with the golden, so look for a real defect in \
             prefill/filters/sampling.\n\n\
             \x20 These are RAW logits, read straight from the decoder before the \
             pipeline's\n  logits-filter chain. The chain adds the same 0 to two \
             unsuppressed candidates,\n  so for a genuine near-tie this IS the \
             margin the sampler saw; if one of the two\n  is a token the chain \
             suppresses, expect the raw numbers to disagree with the\n  sampled \
             outcome, and read that as the tell.\n",
          );
        }
        Err(why) => report.push_str(&format!("\n  (no logit replay: {why})\n")),
      }
    }
    _ => report.push_str(
      "  ...as a LENGTH difference: one stream is a strict prefix of the other, \
       so there is\n  no competing token pair at this index to weigh. The \
       divergence is structural\n  (segment count / early or late EOT), not a \
       single flipped argmax.\n",
    ),
  }

  report.push_str(
    "\n  DO NOT regenerate the golden, and DO NOT add a tolerance. The golden is \
     an\n  EXTERNAL Swift oracle (whisperkit-cli @ argmax-oss-swift); a \
     divergence from it is\n  a real difference to be explained, never smoothed \
     over.\n",
  );
  panic!("{report}");
}

/// Replays `prefix` through a freshly-built pipeline on the **shipping**
/// compute units and returns the raw logits of the last fed step — the
/// distribution the sampler saw when it predicted the token *after*
/// `prefix`.
///
/// Exactness rests on one invariant: `decode::decode_text`'s loop feeds
/// `tokens[k]` at cache position `k` (forcing the prompt tokens for
/// `k < prompt.len()`, then the token it sampled at `k - 1`), and the
/// segment splitter partitions that same stream at the *second* of each
/// adjacent timestamp pair (`segment::find_seek_point_and_segments`) — so
/// the golden's flattened per-segment tokens ARE the fed stream, index for
/// index, and teacher-forcing them reproduces the decode step for step.
///
/// That holds for a **single-window** decode only, which both committed
/// goldens are (jfk 11.0 s, es_test_clip 7.7 s, against a 30 s window). A
/// multi-window clip restarts the prompt and resets the KV cache at every
/// window, and the flat token list carries no window boundaries to
/// reconstruct that from — so this refuses the replay rather than reporting
/// logits from a stream the model never saw.
fn replay_step_logits(audio: &[f32], prefix: &[u32]) -> Result<Vec<f32>, String> {
  use coremlit::{
    Model,
    audio::whisper::{
      audio::pad_or_trim,
      backend::{InferenceBackend, coreml::CoreMlBackend},
      options::{
        DEFAULT_DECODER_COMPUTE_UNITS, DEFAULT_ENCODER_COMPUTE_UNITS, DEFAULT_MEL_COMPUTE_UNITS,
      },
    },
  };

  let tiny = tiny_dir();
  let load = |name: &str, units| {
    Model::load(tiny.join(name), units).map_err(|e| format!("{name} failed to load: {e}"))
  };
  // The SHIPPING compute units, deliberately: a diagnostic that read the
  // logits back on CpuOnly would describe a decode nobody runs, and the
  // whole point here is to characterize an ANE-vs-ANE divergence.
  let backend = CoreMlBackend::new(
    load("MelSpectrogram.mlmodelc", DEFAULT_MEL_COMPUTE_UNITS)?,
    load("AudioEncoder.mlmodelc", DEFAULT_ENCODER_COMPUTE_UNITS)?,
    load("TextDecoder.mlmodelc", DEFAULT_DECODER_COMPUTE_UNITS)?,
  )
  .map_err(|e| format!("backend construction failed: {e}"))?;

  let window_samples = backend.dims().window_samples();
  if audio.len() > window_samples {
    return Err(format!(
      "clip is {} samples, past the {window_samples}-sample window — a \
       multi-window decode re-prompts and resets the KV cache per window, and \
       the flat golden token list records no window boundaries, so a \
       teacher-forced replay would not reproduce the real decode",
      audio.len(),
    ));
  }

  let window = pad_or_trim(audio, window_samples);
  let features = backend
    .extract_features(&window)
    .map_err(|e| format!("mel extraction failed: {e}"))?;
  let encoded = backend
    .encode(&features)
    .map_err(|e| format!("encode failed: {e}"))?;
  let mut state = backend
    .new_decoder_state()
    .map_err(|e| format!("decoder state allocation failed: {e}"))?;

  let mut logits = Vec::new();
  for (position, &token) in prefix.iter().enumerate() {
    backend
      .decode_step(token, position, &encoded, &mut state, &mut logits)
      .map_err(|e| format!("decode step {position} failed: {e}"))?;
  }
  Ok(logits)
}

/// The two highest-scoring `(token, logit)` pairs, best first. Ties resolve
/// to the lower token id, matching the greedy sampler's own argmax.
fn top_two(logits: &[f32]) -> ((u32, f32), (u32, f32)) {
  let mut best = (u32::MAX, f32::NEG_INFINITY);
  let mut second = (u32::MAX, f32::NEG_INFINITY);
  for (token, &logit) in logits.iter().enumerate() {
    let token = u32::try_from(token).expect("vocab fits u32");
    if logit > best.1 {
      second = best;
      best = (token, logit);
    } else if logit > second.1 {
      second = (token, logit);
    }
  }
  (best, second)
}
