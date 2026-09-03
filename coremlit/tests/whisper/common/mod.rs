// The workspace-root anchor every `models_dir()` below resolves against, and
// the sibling-checkout anchor the oracle gates read. FOUND by searching upward
// for the `[workspace]` manifest, never counted in `../` hops — see its module
// doc for why a count is the wrong shape here. Re-exported so the binaries
// that pull this `common` in share the one resolver.
#[path = "../../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;
#[allow(unused_imports)]
pub use workspace_root::{checkout_parent, models_root, workspace_root};

use std::path::PathBuf;

// ── Host-class provenance ───────────────────────────────────────────────────
//
// The whisper goldens are an EXTERNAL Swift oracle captured on the Neural
// Engine, and this pipeline decodes greedily, so one borderline argmax flipped
// by fp16 drift on a different Apple Silicon generation or macOS build cascades
// through every token after it (CI run 97115941847: a +0.0078 margin, one fp16
// ULP, at decode step 11 of es_test_clip). The `generationHosts` set — the host
// classes this exact payload was REPRODUCED on — is what tells that apart from a
// port defect. The predicate lives in `tests/support/host_class.rs` — one copy,
// shared with the speaker and vad suites.
#[path = "../../support/host_class.rs"]
#[allow(dead_code)]
mod host_class;
#[allow(unused_imports)]
pub use host_class::{HostClass, HostVerdict, RecordedHost, check_host_class, legacy_failure_note};

// ── Host-class scoping for MEASURED observations ────────────────────────────
//
// The gates above compare against a COMMITTED Swift golden, so a foreign host
// panics: the comparison is the whole test and there is nothing left to run.
// `streaming.rs` asserts no golden — it characterizes what THIS machine's
// LocalAgreement-2 confirms — so it needs the opposite non-`Match` behaviour:
// measure, print, do not assert. `tests/support/measured_band.rs` is that
// contract, shared with the siglip band suites; `measured_band.rs` resolves
// `super::HostClass` through the re-export above.
#[path = "../../support/measured_band.rs"]
#[allow(dead_code)]
mod measured_band;
#[allow(unused_imports)]
pub use measured_band::{BandGate, BandVerdict, CharacterizedHost, band_verdict};

/// The exact command that regenerates the whisper goldens **from the external
/// Swift oracle**, quoted verbatim into every host-class diagnosis so the
/// failure names its own fix.
///
/// Two spellings of one script: run it locally to get a golden matching THIS
/// machine, or dispatch the workflow to get one matching the CI runner's
/// host-class. `.github/workflows/regen-whisper-goldens.yml` runs this same
/// script, so the two cannot diverge.
#[allow(dead_code)]
pub const WHISPER_REGEN_SCRIPT: &str = "coremlit/tests/whisper/swift/regen_goldens.sh\n\
   \x20   (for the CI runner's host-class instead: `gh workflow run \
   regen-whisper-goldens.yml`,\n\
   \x20    then download the `whisper-goldens` artifact and commit it — the job \
   never commits)";

/// Reads a committed golden from `fixtures/golden/` as untyped JSON, so the
/// host-class gate can read `generationHosts` before the suite deserializes the
/// rest into its own typed shape. One read, not two.
#[allow(dead_code)]
pub fn load_golden_json(fixture: &str) -> serde_json::Value {
  let path = fixtures_dir().join("golden").join(fixture);
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  serde_json::from_str(&text).unwrap_or_else(|e| panic!("{fixture}: not valid JSON: {e}"))
}

/// Runs the host-class gate for one golden and returns the note to append to a
/// fidelity failure — empty when this machine is one of the classes the golden
/// was reproduced on, the ambiguity note on a legacy (unstamped) one.
///
/// Call this BEFORE producing any CoreML number: a host outside the recorded set
/// panics here with the regeneration diagnosis, so the suite never reports host
/// drift as a token divergence.
#[allow(dead_code)]
pub fn golden_host_note(fixture: &str, golden: &serde_json::Value) -> String {
  let recorded = RecordedHost::all_from_golden(fixture, golden).unwrap_or_else(|e| panic!("{e}"));
  let running = HostClass::running();
  match check_host_class(fixture, &recorded, &running, WHISPER_REGEN_SCRIPT) {
    Ok(HostVerdict::Match) => {
      println!(
        "[host] {fixture}: this host is one of the {} class(es) the golden was reproduced on: \
         {running}",
        recorded.len()
      );
      String::new()
    }
    Ok(HostVerdict::LegacyUnknown) => {
      println!(
        "[host] {fixture}: golden records no host class (pre-host-provenance); exact token \
         parity still enforced — but a FAILURE would be ambiguous between a port defect and \
         host fp16 drift. Running host: {running}"
      );
      legacy_failure_note(WHISPER_REGEN_SCRIPT)
    }
    Err(diagnosis) => panic!("{diagnosis}"),
  }
}

pub fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(workspace_root::models_root, PathBuf::from)
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

/// The staged whisper-tiny tokenizer's indexed-ID domain — the `vocab` every
/// `CoreMlBackend` construction here states its decoder's `logits` width as.
/// READ from the tokenizer rather than spelled as 51 865, so a test that
/// builds a backend proves the two agree rather than asserting a literal past
/// them both.
#[allow(dead_code)]
pub fn tiny_vocab_size() -> usize {
  coremlit::audio::whisper::tokenizer::WhisperTokenizer::from_folder(tokenizer_dir())
    .expect("the staged whisper-tiny tokenizer loads")
    .vocab_size()
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
///
/// `allow(dead_code)` for the reason `tokenizer_dir` carries one: the hermetic
/// `golden_provenance.rs` binary compiles this module but decodes no audio.
#[allow(dead_code)]
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
///
/// `host_note` is [`golden_host_note`]'s verdict for this golden, appended
/// last. A mismatched host never reaches here — that panics up front — so this
/// carries either nothing (host matched: the divergence is the port's) or the
/// legacy-ambiguity note (golden unstamped: it could be either).
#[allow(dead_code)]
pub fn assert_golden_tokens(
  label: &str,
  rust: &[u32],
  golden: &[u32],
  audio: &[f32],
  host_note: &str,
) {
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
    // Step k feeds tokens[k] at cache position k and predicts tokens[k+1]
    // (`decode::decode_text`'s loop), so the step that produced the token at
    // `first_diff` is `first_diff - 1` — and at index 0 there is no such step.
    // That case falls THROUGH to the shared tail rather than panicking here, so
    // it still carries the regeneration rule and the host note.
    (Some(&ours), Some(&gold)) => match first_diff.checked_sub(1) {
      None => report.push_str(
        "  ...at index 0 — the start-of-transcript token itself. That is a \
         prefill/prompt-construction bug, not a sampled-token flip; no decode \
         step produced it, so there is no margin to report.\n",
      ),
      Some(step) => {
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
    },
    _ => report.push_str(
      "  ...as a LENGTH difference: one stream is a strict prefix of the other, \
       so there is\n  no competing token pair at this index to weigh. The \
       divergence is structural\n  (segment count / early or late EOT), not a \
       single flipped argmax.\n",
    ),
  }

  report.push_str(
    "\n  DO NOT add a tolerance, and DO NOT re-baseline this golden against our own \
     output.\n  The golden's whole value is that it comes from somewhere else: an EXTERNAL \
     Swift\n  oracle (whisperkit-cli @ argmax-oss-swift). A divergence from it is a real\n  \
     difference to be explained, never smoothed over — and a golden rewritten from \
     the\n  Rust side would assert only that coremlit agrees with coremlit.\n\n  \
     REGENERATION is permitted, but only on those terms:\n\
     \x20   - produced by `whisperkit-cli`, never by this crate. The regeneration script \
     runs\n\
     \x20     the CLI and reshapes its `--report` JSON; it does not build or link coremlit,\n\
     \x20     and `whisper_golden_provenance` fails if it ever does.\n\
     \x20   - recorded against the host classes it was REPRODUCED on \
     (`generationHosts`), so\n\
     \x20     the next reader of a failure can tell a port defect from this machine's fp16 \
     drift.\n\
     \x20   - reviewed as a diff by a human, because a changed oracle output is news.\n\
     \x20 Anything else — including a tolerance — puts the gate to sleep.\n",
  );
  report.push_str(host_note);
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
    tiny_vocab_size(),
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

// ── Model-gate visibility (#61) ─────────────────────────────────────────────
//
// NOT `#[ignore]`d, deliberately. This is the ordinary-run half of the gate
// accounting: an ignored-ONLY run (`-- --ignored`, what every CI gate uses)
// never selects it, and it never appears in an ignored-only `--list`, so the
// anti-vacuum counts those gates take are unchanged. What it adds is the case
// no gate covers — a plain, modelless run — where the skipped gates otherwise
// say nothing but `ignored`. Mechanism, and what it does and does not refuse,
// in the shared module.
#[path = "../../support/model_gate_report.rs"]
mod model_gate_report;

/// Reports how many of this binary's tests are `#[ignore]`d whisperkit model gates
/// that did not run, and whether the models root they read is on disk.
#[test]
fn model_gate_report() {
  model_gate_report::report(&[("WHISPERKIT_TEST_MODELS", models_dir())]);
}
