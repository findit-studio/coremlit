//! Gate 1 (design spec §7): [`alignkit::encode::Encoder::emissions`]
//! (CoreML) vs asry's own ONNX/`ort` emissions path, same audio through
//! both.
//!
//! # Reference mechanism
//!
//! asry has no PUBLIC function that just returns a raw emissions matrix
//! from its ONNX path: `encode_log_softmax`
//! (`asry/src/runner/aligner/algorithm/encode.rs`) is `pub(crate)`, and
//! even the doc-hidden `bench-internals` re-export list
//! (`asry::__bench`, `asry/src/lib.rs`) omits it — only `LogProbsTV`
//! itself and the normalize/tokenize/trellis_beam helpers are exposed
//! there, never the ONNX-calling function. The one public, purpose-built
//! hook is the `parity-dump-emission` Cargo feature (`asry/Cargo.toml`:
//! "Diagnostic-only: write `wy_seg<N>.{emission,trellis}.bin`... to
//! `ASRY_PARITY_DUMP_TRELLIS` whenever set"), wired into
//! `Aligner::align_chunk`/`align_chunk_with_abort`
//! (`asry/src/runner/aligner/aligner.rs:753-795`): set the env var, run a
//! real `align_chunk` call, then read back the dumped
//! `<T: u32 LE><V: u32 LE><T*V f32 LE>` buffer it writes IMMEDIATELY
//! after the ONNX encode step and BEFORE the trellis/beam stage — so the
//! dump exists even when `text` doesn't actually match the audio content
//! and the alignment result itself later errors. [`asry_ort_emissions`]
//! below is this crate's harness around that hook.
//!
//! `sub_segments` is set to cover the whole chunk (`[0, samples.len())`
//! in asry's chunk-local 1/16000 timebase, `asry::time::ANALYSIS_TIMEBASE`)
//! so asry's own silence mask zeroes nothing — the ONNX call sees exactly
//! the same raw samples the CoreML side does. Neither side applies
//! zero-mean/unit-variance normalisation: wav2vec2-base-960h expects raw
//! `[-1, 1]` waveform on both the CoreML and the ONNX-export side, and
//! the design spec's Decisions Log documents asry's own "raw-waveform
//! (non-normalized) encode" as a deliberate WhisperX-parity pin, not a
//! bug to fix — see
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`
//! §3's Scope item 1(a).
//!
//! # Required local artifacts — NOT present at authoring time
//!
//! (See the Task B3 report for the exact fetch commands documented below;
//! none of them were run.)
//!
//! - `Models/alignkit/base960h_aligner.mlmodelc` (this crate's CoreML
//!   model; `ALIGNKIT_TEST_MODELS`, `tests/common::model_path`).
//! - `<asry checkout>/models/wav2vec2-base-960h.onnx` +
//!   `models/wav2vec2-base-960h-tokenizer.json` (~378 MB; asry's own
//!   `README.md` documents the exact SHA-256-verified `curl` commands,
//!   also reachable via `ASRY_FETCH_MODEL=1 ASRY_FETCH_W2V=1 cargo build`
//!   against the asry checkout). Overridable here via `ASRY_ONNX_MODELS`
//!   / `ASRY_ONNX_W2V_MODEL` / `ASRY_ONNX_W2V_TOKENIZER`.
//! - A working `libonnxruntime` dylib discoverable via `ORT_DYLIB_PATH`
//!   (asry's `alignment` feature uses `ort` in load-dynamic mode: it
//!   LINKS without one, but needs one at run time to actually call
//!   `Session::builder()`). None found on this machine outside an
//!   unrelated bundled copy inside a browser application bundle, which
//!   this harness deliberately does not reach for.
//!
//! Because `cargo test -p alignkit -- --ignored` is a standing per-task
//! gate in this workspace and RUNS `#[ignore]`d tests, the test below
//! self-skips (early `return` with a loud `eprintln!` naming exactly
//! what's missing) when the asry ONNX artifacts are absent — the same
//! convention asry's own `tests/aligner_load.rs` uses for its fetched
//! fixtures. KNOWN HAZARD, accepted deliberately: a skip looks like a
//! pass in the test summary; the `eprintln!` (visible with
//! `--nocapture`) and this doc are the mitigations. It does NOT
//! self-skip on a missing/broken ONNX Runtime dylib: staging 378 MB of
//! ONNX artifacts is a deliberate act, and at that point a hard failure
//! naming `ORT_DYLIB_PATH` beats silently skipping the gate the
//! artifacts were staged for.
//!
//! # Vocab alignment (CoreML `V=29` vs asry-ONNX `V=32`)
//!
//! The two "reference" models are different conversions of the SAME
//! wav2vec2-base-960h CTC alphabet, with different-width output heads —
//! not different alphabets, just different label tables for what turns
//! out to be the same 29 symbols once the extra entries below are
//! accounted for:
//!
//! - **chordai's CoreML head** (`assets/chordai_base960h_tokenizer.json`,
//!   [`vocab`]): 29 entries — CTC blank `"-"` = [`vocab::BLANK_ID`]
//!   (`0`), word-delimiter `"|"` = `1`, then the 26 letters and `'` at
//!   `2..28`.
//! - **asry's ONNX/HF head** (the fetched
//!   `wav2vec2-base-960h-tokenizer.json`,
//!   `onnx-community/wav2vec2-base-960h-ONNX`, an export of HuggingFace
//!   `facebook/wav2vec2-base-960h`): 32 entries — the 4 HF
//!   sequence-to-sequence specials `"<pad>"` = `0`, `"<s>"` = `1`,
//!   `"</s>"` = `2`, `"<unk>"` = `3`, then `"|"` = `4`, then the same 26
//!   letters and `'` at `5..31`.
//!
//! [`build_vocab_map`] derives the id correspondence TOKEN-STRING-KEYED
//! from both files, AT RUNTIME, every time this test actually runs (i.e.
//! behind the same artifact self-skip gate as everything else described
//! above) — never a hard-coded/assumed offset. That construction, and
//! the assertions it makes against the live files as it goes, ARE this
//! harness's regression coverage for the mapping: there is no separately
//! pinned constant that could go stale, and a run against an
//! `ASRY_ONNX_W2V_TOKENIZER` override gets checked against exactly the
//! file it actually used, not a fixture snapshot.
//!
//! - **Blank/pad, paired by ROLE, not spelling**: chordai id
//!   [`vocab::BLANK_ID`] (`"-"`) and asry id `0` (`"<pad>"`) both encode
//!   the CTC blank in their own vocabulary, but don't share a spelling.
//!   [`build_vocab_map`] special-cases this ONE pair by role, and
//!   cross-checks the ROLE claim — not just the id-0/id-0 coincidence —
//!   against asry's own tokenizer file's `decoder.pad_token` field before
//!   relying on it. Every other chordai token (`"|"` and the 26
//!   letters/`'`) maps by exact string match instead.
//! - **Bijective over the 29**: every chordai id gets exactly one asry
//!   id, and [`build_vocab_map`] asserts no two chordai ids ever claim
//!   the same asry id.
//! - **Exactly 3 leftover asry ids**: [`EXPECTED_PLACEHOLDER_TOKENS`]
//!   (`<s>`, `</s>`, `<unk>`) — HF sequence-to-sequence framing tokens a
//!   fixed CTC alphabet has no use for and a CTC head structurally cannot
//!   emit. Asserted by string identity, order-independent.
//!
//! ## Renormalization
//!
//! asry's raw 32-wide row is a log-softmax over all 32 labels, including
//! the 3 placeholders the CoreML head has no output units for at all.
//! Comparing `lp32[map(i)]` directly against CoreML's `lp29[i]` would
//! therefore compare a log-probability normalized over 32 outcomes to one
//! normalized over 29 — a different distribution, not just a different
//! indexing of the same one. [`compare`] renormalizes each asry frame
//! onto the shared 29 BEFORE any comparison, dropping the 3 placeholder
//! columns and rescaling the remaining 29 back to sum to 1 in probability
//! space:
//!
//! ```text
//! lp29[i] = lp32[map(i)] - ln( Σ_{j mapped} exp(lp32[j]) )
//! ```
//!
//! — [`renormalize_frame`], the denominator computed
//! log-sum-exp-stabilized by [`log_sum_exp`] (max-shifted, so every
//! `exp(...)` term is `<= 0` and cannot overflow).
//!
//! **Placeholder mass is measured and reported, not just silently
//! discarded.** [`renormalize_frame`] also returns `Σ_{j placeholder}
//! exp(lp32[j])` per frame; [`report`] prints its max and mean across
//! frames for every scenario. If that mass isn't tiny, the
//! renormalization is materially RESHAPING the comparison — redistributing
//! real probability mass onto the 29 real classes, not just dropping
//! negligible noise — and [`report`] flags this loudly (`CAVEAT`, at a
//! `1e-3` threshold well below the `5e-2` Gate-1 max-abs-diff bound
//! itself) rather than letting a materially-altered comparison pass as if
//! it were a clean one.
//!
//! Argmax agreement is computed over the mapped 29 columns only, never the
//! raw 32. Separately, [`compare`] also checks asry's own RAW, unmapped
//! argmax (over all 32 columns, no renormalization involved) on every
//! frame: if that ever lands on one of the 3 placeholder ids, the
//! reference model genuinely favors an outcome the CoreML head cannot
//! represent at all, which would undermine renormalization's working
//! premise that placeholder mass is marginal rather than a real competing
//! hypothesis. Expected count: `0`; [`report`] warns if not.
//!
//! # Gate-1 bound
//!
//! Plan hypothesis (per-chunk emissions parity, plan Global Constraints):
//! per-frame log-prob max-abs-diff <= [`GATE1_MAX_ABS_DIFF`] AND
//! exact-argmax agreement >= [`GATE1_MIN_ARGMAX_AGREEMENT`] of frames,
//! over the frames both sides share. **Not measured at authoring time** —
//! the asry-side artifact classes above were absent; see the Task B3
//! report for exactly what is missing and where this was checked. The
//! bound may be tightened by a future run with real measurements in
//! hand; it must not be loosened to pass.

mod common;

use std::{
  path::PathBuf,
  sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
  },
};

use alignkit::{
  encode::{ENCODER_WINDOW_SAMPLES, Encoder, EncoderOptions},
  vocab,
};
use asry::{Aligner, EnglishNormalizer, Lang, TimeRange, emissions::LogProbsTV};
use coremlit::ComputeUnits;
use serde_json::Value;

/// Gate-1 max-abs log-prob difference bound (plan Global Constraints
/// hypothesis, per-frame, over shared frames). Must not be loosened
/// beyond this to pass — see the module doc's "Gate-1 bound" section.
const GATE1_MAX_ABS_DIFF: f32 = 5e-2;
/// Gate-1 minimum exact-argmax agreement fraction (plan Global
/// Constraints hypothesis). Must not be loosened beyond this to pass —
/// see the module doc's "Gate-1 bound" section.
const GATE1_MIN_ARGMAX_AGREEMENT: f64 = 0.999;

/// Guards every access to the process-global `ASRY_PARITY_DUMP_TRELLIS`
/// env var and to asry's own process-global `wy_seg<N>` dump-file
/// counter (`asry/src/runner/aligner/aligner.rs`'s `SEG_COUNTER`) — see
/// the module doc's "Reference mechanism" section. `cargo test` runs
/// `#[test]` functions in parallel threads by default within one binary;
/// even though this file currently has a single test that touches the
/// dump mechanism, the guard is cheap insurance against a future second
/// one racing it. It is also what makes the `unsafe` `env::set_var`/
/// `remove_var` calls below sound: those became `unsafe fn` in Rust
/// 1.82+ specifically because concurrent env mutation is not
/// thread-safe on every platform, and this lock is the synchronization
/// that rules that out here.
static DUMP_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Directory holding asry's fetched ONNX wav2vec2 artifacts. Mirrors
/// `crates/alignkit/tests/common::models_dir`'s override convention.
fn asry_onnx_models_dir() -> PathBuf {
  std::env::var_os("ASRY_ONNX_MODELS").map_or_else(
    || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../asry/models"),
    PathBuf::from,
  )
}

fn asry_w2v_onnx_path() -> PathBuf {
  std::env::var_os("ASRY_ONNX_W2V_MODEL").map_or_else(
    || asry_onnx_models_dir().join("wav2vec2-base-960h.onnx"),
    PathBuf::from,
  )
}

fn asry_w2v_tokenizer_path() -> PathBuf {
  std::env::var_os("ASRY_ONNX_W2V_TOKENIZER").map_or_else(
    || asry_onnx_models_dir().join("wav2vec2-base-960h-tokenizer.json"),
    PathBuf::from,
  )
}

/// Row-major `(T, V)` log-probabilities read back from asry's
/// `parity-dump-emission` dump file.
struct AsryReference {
  t: usize,
  v: usize,
  data: Vec<f32>,
}

/// Runs `samples` through asry's real ONNX wav2vec2 encoder (via
/// `Aligner::align_chunk` + the `parity-dump-emission` diagnostic hook,
/// see the module doc's "Reference mechanism" section) and returns the
/// resulting log-probabilities.
///
/// `text` only needs to TOKENIZE against the wav2vec2 vocab — it does
/// not need to describe `samples`' actual spoken content. The emission
/// dump happens immediately after the ONNX encode step, before the
/// trellis/beam stage that would need the transcript to actually agree
/// with the audio, so `align_chunk`'s own `Result` is discarded here:
/// any failure at or after the trellis/beam stage is irrelevant to this
/// helper's only job (capturing the encoder's raw output). A missing
/// dump file — meaning `align_chunk` failed BEFORE reaching the encode
/// step — is this helper's own failure and panics loudly instead of
/// silently returning an empty reference.
///
/// # Panics
/// If `Aligner::from_paths` fails to load the ONNX model/tokenizer (see
/// the module doc for the required artifact paths), or if no
/// `*.emission.bin` dump file appears after the `align_chunk` call.
fn asry_ort_emissions(samples: &[f32], text: &str) -> AsryReference {
  let _guard = DUMP_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

  let mut aligner = Aligner::from_paths(
    Lang::En,
    &asry_w2v_onnx_path(),
    &asry_w2v_tokenizer_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect(
    "load asry's ONNX wav2vec2 aligner — see this file's module doc for the required local \
     artifacts (asry ONNX model/tokenizer, ORT_DYLIB_PATH)",
  );

  static DUMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
  let dump_dir = std::env::temp_dir().join(format!(
    "alignkit-parity-emissions-{}-{}",
    std::process::id(),
    DUMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
  ));
  std::fs::create_dir_all(&dump_dir).expect("create scratch dump dir");

  // SAFETY: serialized by `DUMP_ENV_LOCK` above — no other thread in
  // this process reads or writes `ASRY_PARITY_DUMP_TRELLIS`, or
  // triggers asry's internal dump-file-name counter, while this guard
  // is held (see `DUMP_ENV_LOCK`'s doc).
  unsafe {
    std::env::set_var("ASRY_PARITY_DUMP_TRELLIS", &dump_dir);
  }

  let full_chunk = [TimeRange::new(
    0,
    samples.len() as i64,
    asry::time::ANALYSIS_TIMEBASE,
  )];
  // Result intentionally discarded — see this function's doc comment.
  let _ = aligner.align_chunk(samples, &full_chunk, text, 0, |start, end| {
    TimeRange::new(start as i64, end as i64, asry::time::ANALYSIS_TIMEBASE)
  });

  // SAFETY: still inside the `DUMP_ENV_LOCK` critical section started
  // above.
  unsafe {
    std::env::remove_var("ASRY_PARITY_DUMP_TRELLIS");
  }

  let dump_path = std::fs::read_dir(&dump_dir)
    .expect("read scratch dump dir")
    .filter_map(Result::ok)
    .map(|entry| entry.path())
    .find(|path| {
      path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".emission.bin"))
    })
    .unwrap_or_else(|| {
      panic!(
        "no *.emission.bin dump found under {dump_dir:?} — align_chunk must have failed before \
         reaching the ONNX encode step (see this file's module doc's \"Reference mechanism\" \
         section)"
      )
    });

  let bytes = std::fs::read(&dump_path).expect("read emission dump");
  let reference = parse_emission_dump(&bytes);
  let _ = std::fs::remove_dir_all(&dump_dir);
  reference
}

/// Parses asry's `parity-dump-emission` binary format:
/// `<T: u32 LE><V: u32 LE><T*V f32 LE, row-major (T, V)>`
/// (`asry/src/runner/aligner/aligner.rs`'s emission-dump block).
fn parse_emission_dump(bytes: &[u8]) -> AsryReference {
  assert!(bytes.len() >= 8, "dump too short for the T/V header");
  let t = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
  let v = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
  let expected_len = 8 + t * v * 4;
  assert_eq!(
    bytes.len(),
    expected_len,
    "dump byte length {} doesn't match header T={t} * V={v} * 4 + 8 = {expected_len}",
    bytes.len()
  );
  let (chunks, remainder) = bytes[8..].as_chunks::<4>();
  assert!(
    remainder.is_empty(),
    "payload length is a multiple of 4 per the length check above"
  );
  let data = chunks.iter().copied().map(f32::from_le_bytes).collect();
  AsryReference { t, v, data }
}

/// Chordai's CTC blank-token string (`assets/chordai_base960h_tokenizer.json`
/// id [`vocab::BLANK_ID`]) — see the module doc's "Vocab alignment"
/// section.
const CHORDAI_BLANK_TOKEN: &str = "-";

/// asry/HF's CTC blank-ROLE token string: the `<pad>` special token its
/// `tokenizer.json`'s `decoder.pad_token` field designates as the CTC
/// blank. Spelled differently from [`CHORDAI_BLANK_TOKEN`] but the same
/// ROLE; [`build_vocab_map`] pairs the two explicitly rather than relying
/// on string equality. See the module doc's "Vocab alignment" section.
const ASRY_PAD_TOKEN: &str = "<pad>";

/// The exact asry/HF vocab entries [`build_vocab_map`] requires to be the
/// ONLY ids with no chordai counterpart: HuggingFace
/// sequence-to-sequence framing tokens (beginning-of-sequence,
/// end-of-sequence, unknown-word) that CTC decoding never emits — a fixed
/// CTC alphabet has no notion of any of the three. See the module doc's
/// "Vocab alignment" section.
const EXPECTED_PLACEHOLDER_TOKENS: [&str; 3] = ["<s>", "</s>", "<unk>"];

/// Token-string-keyed CTC-role map from chordai's `V=`[`vocab::VOCAB_SIZE`]
/// vocabulary onto asry's wider HF vocabulary, plus the bookkeeping
/// [`compare`] and [`report`] need to renormalize and describe an asry
/// frame in terms of it. Built once per test run by [`build_vocab_map`] —
/// see the module doc's "Vocab alignment" section for the full
/// derivation.
struct VocabMap {
  /// `chordai_tokens[coreml_id]` — chordai's own token strings,
  /// id-ascending. Kept only for human-readable reporting (worst-frame
  /// samples print a token string, not just its numeric id).
  chordai_tokens: Vec<String>,
  /// Number of entries in the asry/HF vocab this map was built from —
  /// [`compare`] cross-checks this against the ACTUAL asry-ONNX
  /// emissions' vocab width before trusting `mapped_asry_id`/
  /// `placeholder_asry_ids` against that data.
  asry_vocab_size: usize,
  /// `mapped_asry_id[coreml_id]` is the asry-side id carrying the same
  /// token (or, for the CTC blank, the same ROLE — see the module doc).
  /// Indexed `0..`[`vocab::VOCAB_SIZE`]; every entry is a valid index
  /// into an asry-side row of width `asry_vocab_size`.
  mapped_asry_id: [usize; vocab::VOCAB_SIZE],
  /// The asry-side ids with no chordai counterpart. Asserted by
  /// [`build_vocab_map`] to name exactly [`EXPECTED_PLACEHOLDER_TOKENS`],
  /// by string; used at comparison time to measure placeholder
  /// probability mass and to check the reference's raw argmax never
  /// lands here (see the module doc).
  placeholder_asry_ids: Vec<usize>,
}

/// Parses a `tokenizers`-crate-schema JSON file's `model.vocab` object
/// (`{token: id}`) into an id-ascending `Vec<String>` with `tokens[id] ==
/// token`. Both the committed chordai asset
/// (`assets/chordai_base960h_tokenizer.json`,
/// [`vocab::tokenizer_json_bytes`]) and asry's fetched HF export use this
/// schema (`{"model": {"vocab": {...}, ...}, ...}`) — see the module
/// doc's "Vocab alignment" section.
///
/// # Panics
/// If `bytes` isn't valid JSON, has no `model.vocab` object, any vocab
/// value isn't representable as a `usize`, or the ids aren't a dense
/// permutation of `0..vocab.len()` (a gap or a duplicate would mean a
/// vocabulary this harness cannot safely reason about token-by-token).
fn vocab_tokens_by_id(bytes: &[u8]) -> Vec<String> {
  let root: Value = serde_json::from_slice(bytes).expect("valid tokenizer JSON");
  let vocab_obj = root
    .get("model")
    .and_then(|model| model.get("vocab"))
    .and_then(Value::as_object)
    .expect("tokenizer JSON has a model.vocab object");

  let len = vocab_obj.len();
  let mut by_id: Vec<Option<String>> = vec![None; len];
  for (token, id_value) in vocab_obj {
    let id = id_value
      .as_u64()
      .and_then(|raw| usize::try_from(raw).ok())
      .filter(|&id| id < len)
      .unwrap_or_else(|| {
        panic!(
          "vocab id {id_value} for token {token:?} is not a valid index into the dense 0..{len} \
           range implied by {len} total vocab entries"
        )
      });
    if let Some(existing) = &by_id[id] {
      panic!(
        "vocab id {id} is claimed by more than one token (at least {existing:?} and {token:?})"
      );
    }
    by_id[id] = Some(token.clone());
  }

  by_id
    .into_iter()
    .enumerate()
    .map(|(id, token)| {
      token
        .unwrap_or_else(|| panic!("vocab id {id} has no token — ids must densely cover 0..{len}"))
    })
    .collect()
}

/// Builds the token-string-keyed CTC-role map from chordai's
/// `V=`[`vocab::VOCAB_SIZE`] vocabulary onto asry's wider HF vocabulary,
/// asserting the full derivation described in the module doc's "Vocab
/// alignment" section against BOTH live tokenizer files as it goes. This
/// construction IS this harness's regression coverage for the mapping: it
/// only ever runs behind the same artifact self-skip gate as the rest of
/// this file's `#[ignore]`d test (module doc's "Required local
/// artifacts" section), so asserting inline here, every real run,
/// against the ACTUAL files that run is using — including under an
/// `ASRY_ONNX_W2V_TOKENIZER` override — gives strictly more protection
/// than a separately pinned constant checked against a fixed test
/// fixture could.
///
/// # Panics
/// See the module doc's "Vocab alignment" section for what's asserted
/// and why each check is load-bearing: chordai has exactly
/// [`vocab::VOCAB_SIZE`] entries; chordai id [`vocab::BLANK_ID`] is the
/// token [`CHORDAI_BLANK_TOKEN`]; asry id `0` is the token
/// [`ASRY_PAD_TOKEN`] AND asry's own `decoder.pad_token` field agrees;
/// every chordai token has a same-string asry counterpart (blank/pad
/// excepted, matched by role); the resulting map is injective (no two
/// chordai tokens claim the same asry id); and the leftover unmapped asry
/// ids are exactly [`EXPECTED_PLACEHOLDER_TOKENS`], by string, in any
/// order.
fn build_vocab_map(chordai_bytes: &[u8], asry_bytes: &[u8]) -> VocabMap {
  let chordai_tokens = vocab_tokens_by_id(chordai_bytes);
  assert_eq!(
    chordai_tokens.len(),
    vocab::VOCAB_SIZE,
    "chordai tokenizer asset has {} entries, expected vocab::VOCAB_SIZE ({})",
    chordai_tokens.len(),
    vocab::VOCAB_SIZE
  );
  assert_eq!(
    chordai_tokens[vocab::BLANK_ID as usize],
    CHORDAI_BLANK_TOKEN,
    "chordai vocab id {} (vocab::BLANK_ID) is {:?}, expected the CTC blank token {:?}",
    vocab::BLANK_ID,
    chordai_tokens[vocab::BLANK_ID as usize],
    CHORDAI_BLANK_TOKEN
  );

  let asry_tokens = vocab_tokens_by_id(asry_bytes);
  assert_eq!(
    asry_tokens.first().map(String::as_str),
    Some(ASRY_PAD_TOKEN),
    "asry/HF tokenizer vocab id 0 is {:?}, expected the CTC pad/blank-role token {:?}",
    asry_tokens.first(),
    ASRY_PAD_TOKEN
  );
  let asry_root: Value = serde_json::from_slice(asry_bytes).expect("valid tokenizer JSON");
  let decoder_pad_token = asry_root
    .get("decoder")
    .and_then(|decoder| decoder.get("pad_token"))
    .and_then(Value::as_str);
  assert_eq!(
    decoder_pad_token,
    Some(ASRY_PAD_TOKEN),
    "asry/HF tokenizer's decoder.pad_token is {decoder_pad_token:?}, expected {:?} — the \
     chordai-blank / asry-pad ROLE pairing this map relies on is unverified without this",
    Some(ASRY_PAD_TOKEN)
  );

  let mut mapped_asry_id = [usize::MAX; vocab::VOCAB_SIZE];
  let mut claimed = vec![false; asry_tokens.len()];
  for (coreml_id, token) in chordai_tokens.iter().enumerate() {
    let asry_id = if token == CHORDAI_BLANK_TOKEN {
      asry_tokens
        .iter()
        .position(|candidate| candidate == ASRY_PAD_TOKEN)
        .unwrap_or_else(|| {
          panic!(
            "asry/HF vocab has no {ASRY_PAD_TOKEN:?} entry to pair with chordai's blank token \
             {CHORDAI_BLANK_TOKEN:?} by role"
          )
        })
    } else {
      asry_tokens
        .iter()
        .position(|candidate| candidate == token)
        .unwrap_or_else(|| {
          panic!(
            "chordai token {token:?} (id {coreml_id}) has no same-string counterpart anywhere in \
             the asry/HF vocab — the two vocabularies are not the bijection this map assumes"
          )
        })
    };
    assert!(
      !claimed[asry_id],
      "asry id {asry_id} ({:?}) is claimed by more than one chordai token — the mapping is not \
       injective",
      asry_tokens[asry_id]
    );
    claimed[asry_id] = true;
    mapped_asry_id[coreml_id] = asry_id;
  }

  let mut placeholder_asry_ids = Vec::new();
  let mut placeholder_tokens: Vec<&str> = Vec::new();
  for (asry_id, is_claimed) in claimed.iter().enumerate() {
    if !is_claimed {
      placeholder_asry_ids.push(asry_id);
      placeholder_tokens.push(asry_tokens[asry_id].as_str());
    }
  }
  let mut sorted_placeholder_tokens = placeholder_tokens.clone();
  sorted_placeholder_tokens.sort_unstable();
  let mut expected_placeholders = EXPECTED_PLACEHOLDER_TOKENS;
  expected_placeholders.sort_unstable();
  assert_eq!(
    sorted_placeholder_tokens.as_slice(),
    expected_placeholders.as_slice(),
    "unmapped asry/HF ids (no chordai counterpart) are {placeholder_tokens:?}, expected exactly \
     the seq2seq placeholders {EXPECTED_PLACEHOLDER_TOKENS:?} (order-independent) — see the \
     module doc's \"Vocab alignment\" section"
  );

  VocabMap {
    chordai_tokens,
    asry_vocab_size: asry_tokens.len(),
    mapped_asry_id,
    placeholder_asry_ids,
  }
}

/// Natural-log `ln(Σ exp(x))` over `xs`, computed with the standard
/// max-shifted stabilization: factor `exp(max)` out of the sum so every
/// remaining exponent is `<= 0` and can never overflow. The
/// renormalization denominator in the module doc's "Vocab alignment"
/// section is exactly this function applied to one asry frame's 29
/// mapped columns.
///
/// # Panics
/// If `xs` is empty — every call site here passes a fixed-size,
/// non-empty window.
fn log_sum_exp(xs: &[f32]) -> f32 {
  assert!(!xs.is_empty(), "log_sum_exp requires at least one term");
  let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  if !max.is_finite() {
    // `xs` empty is ruled out above, so every remaining value must be
    // `-inf` (or `NaN`, propagated as-is) — `compare`'s upstream
    // finite/`<= 0` sanity check on the raw asry data means this branch
    // shouldn't be reachable in practice, but this function doesn't
    // depend on that caller-side guarantee to stay correct.
    return max;
  }
  let shifted_sum: f32 = xs.iter().map(|&x| (x - max).exp()).sum();
  max + shifted_sum.ln()
}

/// Nearest-rank `p`-th percentile (`0.0..=100.0`) of `sorted`, which must
/// already be sorted ascending. Purely a [`report`] diagnostic — the
/// distribution shape it summarizes plays no role in the Gate-1
/// pass/fail decision itself, which remains the single global max over
/// all frames (see the module doc's "Gate-1 bound" section and
/// [`assert_gate1`]).
///
/// # Panics
/// If `sorted` is empty.
fn percentile(sorted: &[f32], p: f64) -> f32 {
  assert!(
    !sorted.is_empty(),
    "percentile of an empty slice is undefined"
  );
  let rank = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
  sorted[rank.min(sorted.len() - 1)]
}

/// Renormalizes one asry frame's raw, wider-vocab log-probabilities onto
/// the shared `V=`[`vocab::VOCAB_SIZE`] alphabet: for each chordai id
/// `i`, `lp29[i] = lp32[map(i)] - ln(Σ_{j mapped} exp(lp32[j]))`,
/// computed log-sum-exp-stabilized via [`log_sum_exp`]. Also returns the
/// frame's placeholder probability mass (`Σ_{j placeholder}
/// exp(lp32[j])`). See the module doc's "Vocab alignment" section for the
/// full derivation and why both numbers matter.
fn renormalize_frame(
  asry_reference: &AsryReference,
  frame: usize,
  map: &VocabMap,
) -> ([f32; vocab::VOCAB_SIZE], f32) {
  let row = &asry_reference.data[frame * asry_reference.v..(frame + 1) * asry_reference.v];

  let mapped_lps: [f32; vocab::VOCAB_SIZE] = map.mapped_asry_id.map(|asry_id| row[asry_id]);
  let denom = log_sum_exp(&mapped_lps);
  let lp29 = mapped_lps.map(|lp| lp - denom);

  let placeholder_mass: f32 = map.placeholder_asry_ids.iter().map(|&j| row[j].exp()).sum();

  (lp29, placeholder_mass)
}

/// One worst-frame diagnostic sample: the single mapped column with the
/// largest `|diff|` within one frame, plus enough context to identify and
/// inspect it.
struct WorstFrame {
  frame: usize,
  token_id: usize,
  coreml_lp: f32,
  asry_lp_renorm: f32,
  abs_diff: f32,
}

/// How many worst-frame samples [`compare`] keeps for [`report`] to
/// print — enough to see a pattern (e.g. "always the same token") in
/// `--nocapture` output without flooding it with one row per frame.
const WORST_FRAMES_KEPT: usize = 10;

/// Per-frame comparison statistics between the CoreML head (native
/// `V=`[`vocab::VOCAB_SIZE`]) and asry's ONNX reference (renormalized
/// onto the same [`vocab::VOCAB_SIZE`] classes — see the module doc's
/// "Vocab alignment" section), over the frames both sides share
/// (`min(coreml.t(), asry.t())`).
struct ParityStats {
  frames_compared: usize,
  max_abs_diff: f32,
  max_abs_diff_frame: usize,
  argmax_agreement_count: usize,
  /// `(frame, coreml_argmax, asry_argmax)`, capped for readability.
  argmax_disagreements: Vec<(usize, usize, usize)>,
  /// Per-frame max-abs-diff, one entry per compared frame in frame order
  /// — [`report`] sorts a copy of this to derive percentiles.
  per_frame_max_abs_diff: Vec<f32>,
  /// Per-frame placeholder probability mass (`Σ exp(lp32)` over the 3
  /// unmapped asry ids) — see the module doc's "Vocab alignment" section.
  placeholder_mass: Vec<f32>,
  /// Frames where asry's RAW, unmapped argmax (over every column it
  /// actually has, before renormalization) lands on a placeholder id —
  /// see the module doc; expected to be empty.
  reference_argmax_is_placeholder: Vec<usize>,
  /// Up to [`WORST_FRAMES_KEPT`] worst frames by absolute diff,
  /// descending.
  worst_frames: Vec<WorstFrame>,
}

impl ParityStats {
  fn argmax_agreement_fraction(&self) -> f64 {
    if self.frames_compared == 0 {
      return 1.0;
    }
    self.argmax_agreement_count as f64 / self.frames_compared as f64
  }
}

/// Computes [`ParityStats`] between `coreml` (native
/// `V=`[`vocab::VOCAB_SIZE`]) and `asry_reference` (raw, wider-vocab),
/// renormalizing the latter onto `coreml`'s classes via `map` before any
/// comparison — see the module doc's "Vocab alignment" section for the
/// full derivation this function implements.
///
/// # Panics
/// If `coreml`'s vocab width isn't [`vocab::VOCAB_SIZE`], or
/// `asry_reference`'s isn't the asry vocab size `map` was built from —
/// either would mean `map` doesn't actually describe the models that
/// produced these emissions, which is a fixture/harness setup bug, not a
/// numeric parity question this stat block can summarize. Also panics if
/// any raw asry log-probability is outside the log-probability domain
/// (non-finite or `> 0.0`) — [`asry_ort_emissions`]'s doc explains why
/// the dump is expected to already satisfy this, so a violation here
/// means that expectation broke upstream.
fn compare(coreml: &LogProbsTV, asry_reference: &AsryReference, map: &VocabMap) -> ParityStats {
  assert_eq!(
    coreml.v(),
    vocab::VOCAB_SIZE,
    "CoreML emissions vocab width {} does not match vocab::VOCAB_SIZE ({}) — the chordai \
     model/vocab pairing itself is inconsistent, independent of asry's reference",
    coreml.v(),
    vocab::VOCAB_SIZE
  );
  assert_eq!(
    asry_reference.v, map.asry_vocab_size,
    "asry-ONNX emissions vocab width {} does not match the {}-entry asry/HF tokenizer the vocab \
     map was built from — the asry model/tokenizer pairing itself is inconsistent",
    asry_reference.v, map.asry_vocab_size
  );
  assert!(
    asry_reference
      .data
      .iter()
      .all(|&x| x.is_finite() && x <= 0.0),
    "asry-ONNX raw emission dump contains a value outside the log-probability domain (non-finite \
     or > 0.0) — renormalization assumes valid log-probabilities, see asry_ort_emissions's doc"
  );

  let frames_compared = coreml.t().min(asry_reference.t);

  let mut max_abs_diff = 0.0f32;
  let mut max_abs_diff_frame = 0usize;
  let mut argmax_agreement_count = 0usize;
  let mut argmax_disagreements = Vec::new();
  let mut per_frame_max_abs_diff = Vec::with_capacity(frames_compared);
  let mut placeholder_mass = Vec::with_capacity(frames_compared);
  let mut reference_argmax_is_placeholder = Vec::new();
  let mut worst_frames = Vec::with_capacity(frames_compared);

  for t in 0..frames_compared {
    let (lp29, frame_placeholder_mass) = renormalize_frame(asry_reference, t, map);
    placeholder_mass.push(frame_placeholder_mass);

    let (mut coreml_argmax, mut coreml_max) = (0usize, f32::NEG_INFINITY);
    let (mut asry_argmax, mut asry_max) = (0usize, f32::NEG_INFINITY);
    let (mut frame_worst_v, mut frame_worst_diff) = (0usize, 0.0f32);
    let (mut frame_worst_coreml_lp, mut frame_worst_asry_lp) = (0.0f32, 0.0f32);

    for (v, &b) in lp29.iter().enumerate() {
      let a = coreml.at(t, v);
      let diff = (a - b).abs();
      if diff > frame_worst_diff {
        frame_worst_diff = diff;
        frame_worst_v = v;
        frame_worst_coreml_lp = a;
        frame_worst_asry_lp = b;
      }
      if diff > max_abs_diff {
        max_abs_diff = diff;
        max_abs_diff_frame = t;
      }
      if a > coreml_max {
        coreml_max = a;
        coreml_argmax = v;
      }
      if b > asry_max {
        asry_max = b;
        asry_argmax = v;
      }
    }
    per_frame_max_abs_diff.push(frame_worst_diff);
    worst_frames.push(WorstFrame {
      frame: t,
      token_id: frame_worst_v,
      coreml_lp: frame_worst_coreml_lp,
      asry_lp_renorm: frame_worst_asry_lp,
      abs_diff: frame_worst_diff,
    });

    if coreml_argmax == asry_argmax {
      argmax_agreement_count += 1;
    } else if argmax_disagreements.len() < 20 {
      argmax_disagreements.push((t, coreml_argmax, asry_argmax));
    }

    // Independent sanity check: does asry's OWN raw argmax (over every
    // column it actually has, no renormalization/mapping involved) ever
    // land on a placeholder id? Expected: never — see the module doc's
    // "Vocab alignment" section.
    let raw_row = &asry_reference.data[t * asry_reference.v..(t + 1) * asry_reference.v];
    let (raw_argmax, _) = raw_row
      .iter()
      .copied()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.total_cmp(b))
      .expect(
        "map.asry_vocab_size > 0 (build_vocab_map asserts an id-0 pad token exists) and \
         asry_reference.v == map.asry_vocab_size (checked above), so raw_row is non-empty",
      );
    if map.placeholder_asry_ids.contains(&raw_argmax) {
      reference_argmax_is_placeholder.push(t);
    }
  }

  worst_frames.sort_unstable_by(|a, b| b.abs_diff.total_cmp(&a.abs_diff));
  worst_frames.truncate(WORST_FRAMES_KEPT);

  ParityStats {
    frames_compared,
    max_abs_diff,
    max_abs_diff_frame,
    argmax_agreement_count,
    argmax_disagreements,
    per_frame_max_abs_diff,
    placeholder_mass,
    reference_argmax_is_placeholder,
    worst_frames,
  }
}

/// Prints the measured Gate-1 numbers to stderr (visible with
/// `cargo test -- --ignored --nocapture`) — the report and this file's
/// doc comment both require the actual measured numbers recorded, not
/// just the pass/fail outcome. Always prints the placeholder-mass and
/// worst-frame diagnostics, regardless of whether [`assert_gate1`] ends
/// up passing — see the module doc's "Vocab alignment" section for why
/// those numbers matter to trusting this comparison at all.
fn report(label: &str, stats: &ParityStats, map: &VocabMap) {
  let mut sorted_diffs = stats.per_frame_max_abs_diff.clone();
  sorted_diffs.sort_unstable_by(f32::total_cmp);
  let p50 = percentile(&sorted_diffs, 50.0);
  let p95 = percentile(&sorted_diffs, 95.0);
  let p99 = percentile(&sorted_diffs, 99.0);
  let p100 = percentile(&sorted_diffs, 100.0);

  let placeholder_max = stats
    .placeholder_mass
    .iter()
    .copied()
    .fold(0.0f32, f32::max);
  let placeholder_mean = if stats.placeholder_mass.is_empty() {
    0.0
  } else {
    stats.placeholder_mass.iter().sum::<f32>() / stats.placeholder_mass.len() as f32
  };

  eprintln!(
    "[gate1] {label}: frames_compared={} max_abs_diff={:.6} (at frame {})",
    stats.frames_compared, stats.max_abs_diff, stats.max_abs_diff_frame,
  );
  eprintln!(
    "[gate1] {label}: per_frame_max_abs_diff p50={p50:.6} p95={p95:.6} p99={p99:.6} \
     max={p100:.6}",
  );
  eprintln!(
    "[gate1] {label}: argmax_agreement={:.4}% ({}/{}) disagreements(frame,coreml,asry)={:?}",
    stats.argmax_agreement_fraction() * 100.0,
    stats.argmax_agreement_count,
    stats.frames_compared,
    stats.argmax_disagreements,
  );
  eprintln!(
    "[gate1] {label}: placeholder_mass max={placeholder_max:.6e} mean={placeholder_mean:.6e} \
     reference_argmax_is_placeholder_frames={}",
    stats.reference_argmax_is_placeholder.len(),
  );
  if !stats.reference_argmax_is_placeholder.is_empty() {
    let shown = stats.reference_argmax_is_placeholder.len().min(20);
    eprintln!(
      "[gate1] {label}: WARNING — asry's raw (unmapped) argmax lands on a placeholder id on {} \
       frame(s), expected 0: {:?}{}",
      stats.reference_argmax_is_placeholder.len(),
      &stats.reference_argmax_is_placeholder[..shown],
      if stats.reference_argmax_is_placeholder.len() > shown {
        " (truncated)"
      } else {
        ""
      },
    );
  }
  if placeholder_max > 1e-3 {
    eprintln!(
      "[gate1] {label}: CAVEAT — max placeholder mass {placeholder_max:.6e} exceeds 1e-3; the \
       32-to-29 renormalization is materially reshaping the comparison, not just discarding \
       negligible mass — see this file's module doc \"Vocab alignment\" section",
    );
  }
  eprintln!(
    "[gate1] {label}: worst frames (frame, token[id], coreml_lp, asry_lp_renorm, abs_diff):"
  );
  for w in &stats.worst_frames {
    eprintln!(
      "[gate1] {label}:   t={} {:?}[{}] coreml={:.6} asry_renorm={:.6} abs_diff={:.6}",
      w.frame,
      map.chordai_tokens[w.token_id],
      w.token_id,
      w.coreml_lp,
      w.asry_lp_renorm,
      w.abs_diff,
    );
  }
}

/// Asserts `stats` meets the Gate-1 bound (module doc's "Gate-1 bound"
/// section). A failure here is a DIVERGENCE to report, not a bound to
/// loosen — see the module doc.
fn assert_gate1(label: &str, stats: &ParityStats) {
  assert!(
    stats.max_abs_diff <= GATE1_MAX_ABS_DIFF,
    "Gate 1 DIVERGENCE ({label}): max-abs log-prob diff {} exceeds the hypothesis bound \
     {GATE1_MAX_ABS_DIFF} at frame {}",
    stats.max_abs_diff,
    stats.max_abs_diff_frame
  );
  let agreement = stats.argmax_agreement_fraction();
  assert!(
    agreement >= GATE1_MIN_ARGMAX_AGREEMENT,
    "Gate 1 DIVERGENCE ({label}): argmax agreement {:.4}% below the hypothesis bound {:.1}% ({} \
     disagreement(s) of {} frames)",
    agreement * 100.0,
    GATE1_MIN_ARGMAX_AGREEMENT * 100.0,
    stats.frames_compared - stats.argmax_agreement_count,
    stats.frames_compared
  );
}

/// Simple, all-in-vocabulary English text. Its only job is to tokenize
/// successfully against the wav2vec2-base-960h CTC vocab — see
/// [`asry_ort_emissions`]'s doc for why it does not need to describe
/// either fixture's actual spoken content.
const PLACEHOLDER_TEXT: &str = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG";

/// **Gate 1.** Compares [`Encoder::emissions`] (CoreML) against asry's
/// ONNX/`ort` emissions path on two REQUIRED scenarios (plan Global
/// Constraints):
///
/// 1. `ted_60.wav`, exactly [`ENCODER_WINDOW_SAMPLES`] — the fixture
///    already occupies the model's full window, so this scenario carries
///    zero padding on the CoreML side.
/// 2. A short slice (well below the model's frame ceiling) — exercises
///    [`Encoder`]'s zero-pad + truncate path (module doc's "Fixed-window
///    bridging") against asry's own non-fixed-window short-chunk
///    handling, i.e. the truncation-rule case the design spec's
///    Candidate A note calls out.
///
/// See this file's module doc for the `#[ignore]` + self-skip layering
/// (and its accepted skip-looks-like-pass hazard), the "Vocab alignment"
/// section for how the two models' different-width CTC heads are bridged
/// before any numeric comparison happens, and the Task B3 report for the
/// measured-numbers gap this leaves.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS) plus asry's ONNX wav2vec2 \
artifacts (models/wav2vec2-base-960h.onnx + tokenizer.json, ~378 MB; absent at authoring time — \
self-skips with a message when missing) and a working libonnxruntime via ORT_DYLIB_PATH; see \
this file's module doc and the Task B3 report for exact fetch commands (intentionally not run) \
and where each path was checked"]
fn emissions_match_asry_ort_reference() {
  // Self-skip (NOT a pass — see the module doc's accepted hazard) when
  // the asry-side ONNX artifacts are absent, so the standing
  // `cargo test -p alignkit -- --ignored` gate stays runnable on
  // machines that have this crate's own CoreML model but not asry's
  // ~378 MB reference artifacts. Same convention as asry's own
  // `tests/aligner_load.rs`.
  let onnx = asry_w2v_onnx_path();
  let tokenizer = asry_w2v_tokenizer_path();
  if !onnx.exists() || !tokenizer.exists() {
    eprintln!(
      "SKIP emissions_match_asry_ort_reference: asry ONNX reference artifacts missing \
       (checked {onnx:?} and {tokenizer:?}); Gate 1 NOT measured. Fetch per asry/README.md's \
       SHA-pinned commands (or ASRY_FETCH_MODEL=1 ASRY_FETCH_W2V=1 cargo build in the asry \
       checkout), then set ORT_DYLIB_PATH and re-run."
    );
    return;
  }

  // Built once, from the two live tokenizer files, before either
  // scenario runs — see the module doc's "Vocab alignment" section. The
  // map depends only on the two vocabularies, never on the audio.
  let asry_tokenizer_bytes = std::fs::read(&tokenizer).expect("read asry tokenizer.json");
  let map = build_vocab_map(vocab::tokenizer_json_bytes(), &asry_tokenizer_bytes);

  let encoder = Encoder::from_file_with(
    common::model_path(),
    EncoderOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load base960h_aligner.mlmodelc");

  // Scenario 1 (REQUIRED): the full ted_60.wav window, no padding.
  let ted_60 = common::load_wav_mono_f32(&common::ted_60_wav_path());
  assert_eq!(
    ted_60.len(),
    ENCODER_WINDOW_SAMPLES,
    "ted_60.wav must be exactly ENCODER_WINDOW_SAMPLES samples (see tests/common's doc)"
  );
  let coreml_full = encoder
    .emissions(&ted_60)
    .expect("CoreML emissions on ted_60.wav");
  let asry_full = asry_ort_emissions(&ted_60, PLACEHOLDER_TEXT);
  let stats_full = compare(&coreml_full, &asry_full, &map);
  report("ted_60.wav (full 960,000-sample window)", &stats_full, &map);

  // Scenario 2 (REQUIRED): a short slice — the truncation-rule case.
  let short_slice = &ted_60[..48_000];
  let coreml_short = encoder
    .emissions(short_slice)
    .expect("CoreML emissions on short slice");
  let asry_short = asry_ort_emissions(short_slice, PLACEHOLDER_TEXT);
  let stats_short = compare(&coreml_short, &asry_short, &map);
  report(
    "48,000-sample slice (truncation-rule case)",
    &stats_short,
    &map,
  );

  // Both REQUIRED scenarios are measured and reported above BEFORE
  // either assertion below runs, so a DIVERGENCE in one scenario can
  // never suppress the measured numbers for the other — every
  // `--nocapture` run prints a complete Gate-1 measurement across both
  // scenarios regardless of which (if either) fails. Asserting only
  // after both are reported does not touch either bound (module doc's
  // "Gate-1 bound" section); it only changes when the failure is raised
  // relative to reporting.
  assert_gate1("ted_60.wav (full window)", &stats_full);
  assert_gate1("48,000-sample slice (truncation-rule case)", &stats_short);
}
