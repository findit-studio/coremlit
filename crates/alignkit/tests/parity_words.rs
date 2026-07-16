//! **The word-timing parity gate** (design spec §7 Gate 1, as reframed by
//! issue #12): alignkit's CoreML wav2vec2 encoder against asry's ONNX-Runtime
//! wav2vec2 encoder, compared where a user actually consumes the result —
//! **per-word start/end times and scores** on real speech.
//!
//! # Why the word level, and not the emission tensor
//!
//! The original Gate 1 compared raw emission tensors under a `max-abs ≤ 5e-2 /
//! argmax ≥ 99.9%` bound invented before anything was measured. That bound is
//! physically unreachable for two independently-converted copies of wav2vec2
//! (ours `torchaudio → CoreML`, fp16, 29-class; asry's `HuggingFace → ONNX`,
//! fp32, 32-class) and the one alarming number it produced turned out to be a
//! **harness bug**, not a model disagreement
//! (`.superpowers/sdd/alignkit-gate1-diagnostic.md`). Comparing at the word
//! level deliberately abstracts away the vocab width and the fp16/fp32 tail,
//! neither of which a caller can observe, and measures the thing that ships.
//!
//! asry is the right oracle at that level: it is the SAME forced-alignment
//! algorithm — literally the same `AlignerCore`, reached through the same
//! `prepare` / `finish` — and it is WhisperX-parity-proven. The two code paths
//! differ in exactly one place:
//!
//! ```text
//!   asry     prepare(samples) → encoder_input → [ ONNX fp32, 32-class ] → finish
//!   alignkit prepare(samples) → encoder_input → [ CoreML fp16, 29-class ] → finish
//! ```
//!
//! so a word-timing disagreement can only come from the encoder swap. That is
//! what this gate isolates.
//!
//! # The oracle is not ground truth, and this gate does not pretend it is
//!
//! Measured on `jfk.wav`: 38 of 44 word boundaries agree to within one 20 ms
//! frame, and the median disagreement is 12.8 ms — under a single frame. **One
//! boundary disagrees by 908 ms**, and it is the ORACLE that is wrong there.
//! The word is the second `ask`, whose onset asry places at 7507 ms and
//! alignkit at 8415 ms. The audio settles it:
//!
//! - A 20 ms RMS envelope of `jfk.wav` puts **silence** across 7460–8180 ms
//!   (RMS 0.009–0.037, against 0.2+ for speech). asry's onset, 7507 ms, is
//!   inside that silence.
//! - The acoustic model's own emissions are `logP(blank) == 0.0000` — exactly,
//!   fp16-saturated — for all 41 frames from 7560 to 8360 ms. The tensor
//!   carries *no information whatsoever* about where a token begins in that
//!   run, so the trellis is choosing between numerically identical paths and
//!   the tie can fall anywhere. This is not encoder noise; it is an absence of
//!   evidence.
//! - The first frame with real evidence is 419 (**8380 ms**), where the model
//!   fires `A` with a +6.64 margin over blank. A greedy CTC decode of the whole
//!   clip confirms it: `…|FOR|YOU|AND|WHAT|YOU|CAN|…` with `A@8380ms`.
//!
//! So alignkit's onset is **35 ms after** the true acoustic onset, and asry's is
//! **873 ms before** it. Requiring alignkit to reproduce the oracle's answer
//! here would be requiring it to be wrong. The gate is therefore built on
//! **robust statistics** (median, p90) plus an explicit, pinned **ledger of the
//! divergences** ([`JFK_EXPECTED_DIVERGENCES`]) — not on a max-delta bound, which
//! could only be satisfied by inflating it to 908 ms, at which point it would no
//! longer catch the 881 ms regression it exists to catch. A bound that cannot
//! distinguish the defect from the baseline is not a bound.
//!
//! # Two clips, because one of them is padded and the other is not
//!
//! | test | clip | samples | padding CoreML sees | what it isolates |
//! |---|---|---|---|---|
//! | [`word_timings_agree_with_asry_ort_on_jfk`] | `jfk.wav` | 176,000 | **784,000 zeros (81.7%)** | encoder swap **+** fixed-window padding, summed |
//! | [`word_timings_agree_with_asry_ort_on_ted_60`] | `ted_60.wav` | **960,000** | **none** | the encoder swap, **alone** |
//!
//! alignkit's CoreML graph takes a fixed `[1, 960_000]` input, so a short chunk
//! is zero-padded to 60 s before the encoder sees it; asry's ONNX graph is
//! variable-length and sees the buffer as-is. On `jfk.wav` that asymmetry is
//! **most of the input**, and it is not cosmetic — wav2vec2-base group-norms
//! over the whole sequence axis and attends globally with no padding mask, so
//! 49 s of zeros perturb every real frame. Every jfk number is therefore a
//! *sum* of two effects and cannot separate them;
//! [`fixed_window_padding_does_not_explain_the_divergence`] exists solely to
//! bound the second.
//!
//! `ted_60.wav` is **exactly** `ENCODER_WINDOW_SAMPLES`, so `emissions_raw`
//! borrows the buffer and appends nothing: both encoders see the identical
//! 960,000 real samples and the encoder swap is the only difference left. It
//! needs no control, and it is the stronger measurement. It also says something
//! jfk cannot:
//!
//! | | jfk (padded) | **ted_60 (unpadded)** |
//! |---|---|---|
//! | median \|Δ\| | 12.8 ms | **0.0 ms** |
//! | p90 \|Δ\| | 47.0 ms | **0.0 ms** |
//! | boundaries within one frame | 38/44 (86.4%) | **367/372 (98.7%)** |
//!
//! **With the padding gone, the CoreML fp16 29-class encoder and the ONNX fp32
//! 32-class encoder put 90%+ of all word boundaries on the same frame.** jfk's
//! 12.8 ms median really was the zeros.
//!
//! ted_60 also runs what jfk leaves cold: the full 2,999-frame emission tensor
//! instead of 550, a trellis over 186 words instead of 22, and a real
//! disfluency — the speaker says `would` twice and the ASR transcript names it
//! once — which is exactly where a forced aligner has to guess, and where the
//! two aligners disagree (see [`TED_60_EXPECTED_DIVERGENCES`]; the audio says
//! alignkit is right, and the ANE collapses it onto the oracle's answer).
//!
//! # What "same input" means here
//!
//! Both aligners receive the **same decoded `Vec<f32>`, by reference**, the
//! same transcript, the same `EnglishNormalizer`, the same OOV policy
//! (`default_oov_decisions`), the same whole-chunk speech span, the same
//! 320-sample stride, and the same coverage / silent-run defaults. Buffer
//! identity holds by construction; [`common::JFK_SAMPLES_SHA256`] and
//! [`common::TED_60_SAMPLES_SHA256`] additionally pin the fixtures, because the
//! first attempt at this comparison reported an "86.6% divergence" that was
//! really one side being handed a padded buffer and the other an unpadded one.
//! A parity number measured on two different inputs measures the harness.
//!
//! # This test does not skip
//!
//! `#[ignore]` is the opt-in gate and the only gate; `required-features =
//! ["parity-oracle"]` (see `Cargo.toml`) keeps ort out of the default test
//! build. A missing model, tokenizer or fixture is a hard `.expect()` failure,
//! never an early `return`. A model-gated test that silently returns and still
//! prints `ok` is a fake gate — that exact bug shipped in asry's CI and in
//! this crate's own `tests/align_chunk.rs`, where an empty models directory
//! reported `test result: ok. 1 passed` having aligned nothing.
//!
//! # Running it
//!
//! ```text
//! cargo test -p alignkit --features parity-oracle -- --ignored
//! ```
//!
//! Needs `Models/alignkit/` (or `ALIGNKIT_TEST_MODELS`), asry's ONNX oracle in
//! its `models/` directory (or `ALIGNKIT_ASRY_MODELS`), and — because `ort`
//! runs in `load-dynamic` mode — `libonnxruntime.dylib` on the loader path.

mod common;

use core::sync::atomic::AtomicBool;
use std::{
  ffi::{OsStr, OsString, c_void},
  path::{Path, PathBuf},
  process::{Command, Stdio},
  time::{Duration, Instant},
};

use alignkit::{
  ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock, TimeRange, Word,
  default_oov_decisions,
};
use asry::Aligner as OrtAligner;

/// 16 kHz: 16 samples per millisecond. Word PTS are 16 kHz sample indices
/// (both aligners are anchored at stream sample 0 in [`ANALYSIS_TIMEBASE`]),
/// so this is the only conversion the comparison needs.
const SAMPLES_PER_MS: f64 = 16.0;

/// One encoder frame in milliseconds: `HOP_SAMPLES` (320) @ 16 kHz. **The
/// quantum of this entire measurement.** A CTC trellis backtrack yields a frame
/// index per token, so no word boundary from either aligner can be more precise
/// than 20 ms, and the smallest disagreement either can express is one frame.
const FRAME_MS: f64 = 20.0;

/// Largest tolerated **median** boundary disagreement: **one frame**.
///
/// The bound on *systematic* shift: a defect that moves every word — a wrong
/// stride (319 vs 320), an off-by-one in the emissions truncation, a mis-anchored
/// clock — moves the median, however small each individual shift is. A worst-case
/// bound cannot see any of that.
///
/// Measured: **12.8 ms**, already under one frame.
///
/// It is **not** the bound that catches the ANE corruption, and this is recorded
/// rather than assumed because it was measured: on the corrupted path the median
/// moves only 12.8 → 16.7 ms, still inside this bound. What catches the ANE is
/// [`ACOUSTIC_ONSET_OF_ASK_MS`] and [`JFK_EXPECTED_DIVERGENCES`]. Do not "simplify"
/// the gate down to this bound.
const MAX_MEDIAN_BOUNDARY_DELTA_MS: f64 = FRAME_MS;

/// Largest tolerated **90th-percentile** boundary disagreement: **5 frames**.
///
/// Bounds the bulk of the distribution without being hostage to the
/// information-free outlier ([`JFK_EXPECTED_DIVERGENCES`]). Measured: **47.0 ms**.
///
/// The headroom above that is not slack, it is a *measured floor*: alignkit
/// cannot beat its own fixed-window padding, and
/// [`fixed_window_padding_does_not_explain_the_divergence`] measures the
/// padding's cost — asry-ort against **itself**, ONNX both times, nothing but
/// the zeros changing — at a p90 in this same band. A bound tighter than the
/// padding's own contribution would be demanding that alignkit outperform its
/// model's input shape.
const MAX_P90_BOUNDARY_DELTA_MS: f64 = 5.0 * FRAME_MS;

/// A boundary disagreement above this is "gross": no longer explicable as
/// encoder precision or as fixed-window padding, and therefore something that
/// must be *named* in [`JFK_EXPECTED_DIVERGENCES`] rather than absorbed by a
/// tolerance.
///
/// **150 ms = 7.5 frames.** It sits above the worst *acoustically anchored*
/// disagreement measured anywhere on this fixture (91.1 ms, on `country`) with
/// 1.6× of room, and far below the 881.6 ms by which the ANE's corrupted
/// emissions displace `ask`. It is a classifier, not a tolerance — nothing
/// passes merely by coming in under it.
const GROSS_DELTA_MS: f64 = 150.0;

/// **The ledger.** Every boundary that diverges from the oracle by more than
/// [`GROSS_DELTA_MS`], named — an `assert_eq!` on the set, so it is pinned in
/// **both** directions.
///
/// `(word index, boundary)`. The single entry is the second `ask` (word 14 of
/// 22), whose ONSET the oracle places 873 ms before the audio contains any
/// evidence for it. See the module doc for the acoustic proof.
///
/// # Why a ledger and not a max-delta bound
///
/// Because a max-delta bound here is not merely weak, it is **inverted** —
/// measured, not argued. Mutating [`alignkit::encode::DEFAULT_ENCODER_COMPUTE`]
/// to `ComputeUnits::All` (the ANE placement, whose fp16 `log(softmax)` tail
/// saturates 16.7% of emission cells to a `-45440` sentinel) gives:
///
/// | | correct (`CpuOnly`) | **corrupted (`All`)** |
/// |---|---|---|
/// | max \|Δ\| vs oracle | 908.0 ms | **87.1 ms** |
/// | median \|Δ\| | 12.8 ms | 16.7 ms |
/// | boundaries within 1 frame | 38/44 | 32/44 |
/// | gross divergences | `[(14, Start)]` | **`[]`** |
///
/// The corruption makes alignkit agree **better** with the oracle on every
/// worst-case statistic, because it destroys the fp16-saturated blank plateau
/// that was anchoring the `ask` onset to the real acoustic evidence, and lets
/// the trellis drift early into the silence — to 7533.7 ms, right next to the
/// oracle's own wrong 7507.3 ms. A `max |Δ| <= 100 ms` gate would have **passed
/// the corrupted build and failed the correct one.**
///
/// So the ledger pins the divergence set by identity. A **new** divergence fails
/// (a fresh defect), and — the case that matters — the **disappearance** of this
/// one fails too, because agreeing with a wrong oracle is itself the symptom.
const JFK_EXPECTED_DIVERGENCES: &[(usize, Boundary)] = &[(14, Boundary::Start)];

/// The true acoustic onset of the second `ask`, in ms: **8380**, frame 419.
///
/// Where the oracle cannot be trusted, the gate falls back to something that
/// can — the audio. This is the first frame at which the acoustic model's
/// posterior leaves its fp16-saturated blank plateau (`logP(blank) == 0.0000`
/// for all 41 frames from 7560 ms to 8360 ms) and fires a letter, `A`, with a
/// +6.64 log-prob margin over blank. It is corroborated independently by the
/// signal itself: a 20 ms RMS envelope shows silence (RMS ≤ 0.037) through
/// 8180 ms and speech (RMS 0.2+) after it.
///
/// So for this one boundary the gate asserts alignkit against the **audio**
/// rather than against asry. That is a strictly stronger check than the parity
/// comparison it replaces — and it is the check that catches the ANE
/// corruption, which moves this exact word by 881.6 ms.
const ACOUSTIC_ONSET_OF_ASK_MS: f64 = 8380.0;

/// How far alignkit's `ask` onset may sit from [`ACOUSTIC_ONSET_OF_ASK_MS`]:
/// **3 frames**. Measured: **+35.3 ms** (8415.3 ms), under two. A CTC onset
/// frame is the first frame of the token's *acoustic* realisation, which for a
/// vowel-initial word may legitimately lag the first frame at which the model
/// becomes confident by a frame or so; three frames of room covers that without
/// admitting anything that could be called a misplacement.
const MAX_ASK_ONSET_ERROR_MS: f64 = 3.0 * FRAME_MS;

/// Largest tolerated **median** per-word score disagreement: `0.10`.
///
/// Scores are the mean per-frame posterior along the word's path, so unlike a
/// frame index they are a *continuous* function of the emission values and
/// cannot be expected to agree closely across an fp16/fp32, 29-vs-32-class
/// encoder swap: the measured per-word spread runs from 0.0054 to 0.2997, with
/// a median of **0.0838**. The maximum is deliberately NOT bounded — it belongs
/// to `ask`, the same word whose span the two sides disagree about (140 ms vs
/// 1104 ms), so it is a restatement of the timing divergence, not independent
/// evidence.
///
/// This bounds a **systematic confidence regression** and nothing else. It does
/// NOT catch the ANE corruption — measured, not assumed: on the corrupted path
/// the median score delta *falls*, to 0.0465, for the same reason its timing
/// statistics improve (it converges on the oracle's wrong answer). Keeping a
/// bound honest about what it cannot see is the point of writing this down.
const MAX_MEDIAN_SCORE_DELTA: f32 = 0.10;

/// Largest tolerated boundary movement when the ORACLE's OWN input is
/// zero-padded to alignkit's fixed window: **5 frames** at p90.
/// [`fixed_window_padding_does_not_explain_the_divergence`] measures it.
const MAX_PADDING_P90_DELTA_MS: f64 = 5.0 * FRAME_MS;

// =========================================================================
// ted_60 — the UNPADDED clip. Its own bounds, because they are properties of
// its audio, not of the harness.
//
// Why the numbers below are so much tighter than jfk's: with the padding gone
// (`ted_60.wav` is exactly ENCODER_WINDOW_SAMPLES, so `emissions_raw` borrows
// the buffer and appends no zeros), the ONLY difference left between the two
// pipelines is the encoder itself — CoreML fp16 29-class against ONNX fp32
// 32-class. jfk cannot separate those two effects; it can only measure their
// sum. ted_60 measures the encoder swap ALONE, and the answer is that the two
// encoders put **90%+ of all word boundaries on exactly the same frame**:
//
// |                    | jfk (padded 81.7%) | ted_60 (unpadded) |
// |--------------------|--------------------|-------------------|
// | median |Δ|         | 12.8 ms            | **0.0 ms**        |
// | p90 |Δ|            | 47.0 ms            | **0.0 ms**        |
// | within one frame   | 38/44 (86.4%)      | **367/372 (98.7%)** |
//
// That is the affirmative result of this fixture, and it retroactively
// confirms `fixed_window_padding_does_not_explain_the_divergence`: jfk's
// 12.8 ms median really was the zero-padding, and with the zeros removed it
// collapses to nothing.
// =========================================================================

/// ted_60's largest tolerated **median** boundary disagreement: **one frame**.
/// Measured: **0.0 ms** — the median boundary is frame-IDENTICAL.
///
/// Bounds systematic shift (a wrong stride, an off-by-one in the emissions
/// truncation, a mis-anchored clock), exactly as its jfk counterpart does.
///
/// It does **not** catch the ANE corruption, and that is measured, not
/// assumed: on the corrupted path ted_60's median is *also* 0.0 ms. See
/// [`TED_60_EXPECTED_DIVERGENCES`] for what does.
const MAX_TED_60_MEDIAN_BOUNDARY_DELTA_MS: f64 = FRAME_MS;

/// ted_60's largest tolerated **90th-percentile** boundary disagreement: **one
/// frame**. Measured: **0.0 ms**.
///
/// Five times tighter than jfk's `MAX_P90_BOUNDARY_DELTA_MS` (5 frames), and
/// the gap is the whole point of this clip. jfk's p90 headroom is a *measured
/// floor* imposed by its zero-padding — the control test shows the padding
/// alone costs p90 80.9 ms, and alignkit cannot beat its own model's input
/// shape. Remove the padding and that floor disappears: on ted_60 at least 90%
/// of the 372 boundaries land on the **same frame**, so one frame of headroom
/// over a measured 0.0 ms is the honest bound.
///
/// Also does **not** catch the ANE (corrupted p90 is likewise 0.0 ms).
const MAX_TED_60_P90_BOUNDARY_DELTA_MS: f64 = FRAME_MS;

/// **ted_60's ledger.** The single gross (> [`GROSS_DELTA_MS`]) divergence:
/// word 96, `would`, its END — an `assert_eq!` on the set, so it is pinned in
/// **both** directions.
///
/// # The boundary, and which side the AUDIO says is right
///
/// The speaker says `would` **twice** — "the paper would… would come along" —
/// a disfluency the ASR transcript elides (see [`common::TED_60_TRANSCRIPT`],
/// which names this spot in advance as a place the trellis can diverge). One
/// transcript word, two acoustic realisations, and a 100 ms fp16-saturated
/// blank plateau between them: the trellis has to pick, and the two aligners
/// pick differently.
///
/// | | `would`.end |
/// |---|---|
/// | alignkit (`CpuOnly`, shipping) | **31,981.3 ms** |
/// | asry-ort (the oracle) | 31,741.2 ms |
/// | alignkit (`ComputeUnits::All`, ANE-corrupt) | **31,741.2 ms** — the oracle's value, exactly |
///
/// Three independent readings of the audio say **alignkit is right**:
///
/// 1. A **greedy CTC decode** of alignkit's own emissions reads `WOULD` at
///    31,560–31,680 ms and a **second** `WOULD` at 31,820–**31,940** ms.
/// 2. A **verbatim forced alignment** — the same clip with `would would` in
///    the transcript — places those two words at 31,601.1–31,741.2 and
///    31,861.2–**31,981.3** ms, with scores 0.758 and 0.664. The second
///    realisation is real, confident speech.
/// 3. The **RMS envelope** has no silent run anywhere in 28,400–33,720 ms, so
///    31,820–31,940 ms carries speech energy, not silence.
///
/// The oracle's answer, 31,741.2 ms, is *exactly* the offset of the FIRST
/// `would`. It therefore assigns a 120 ms, confidently-decoded `WOULD` to
/// **blank** — contradicted by its own posterior. alignkit's answer spans the
/// word's full acoustic support and hands off precisely where `come` begins
/// (both aligners put `come` at 32,021.4 ms). **Requiring alignkit to
/// reproduce the oracle here would be requiring it to call speech silence.**
///
/// # Why a ledger and not a max-delta bound — measured on THIS clip
///
/// Because on ted_60 a max-delta bound is not merely weak, it is **inverted**,
/// and more starkly than on jfk. Mutating [`alignkit::encode::DEFAULT_ENCODER_COMPUTE`]
/// to `ComputeUnits::All`:
///
/// | | correct (`CpuOnly`) | **corrupted (`All`)** |
/// |---|---|---|
/// | max \|Δ\| vs oracle | 240.1 ms | **20.1 ms** |
/// | median \|Δ\| | 0.0 ms | 0.0 ms |
/// | p90 \|Δ\| | 0.0 ms | 0.0 ms |
/// | boundaries within 1 frame | 367/372 | **369/372** |
/// | median \|Δscore\| | 0.0134 | **0.0076** |
/// | gross divergences | `[(96, End)]` | **`[]`** |
///
/// **Every agreement statistic IMPROVES under corruption.** max, within-one-frame
/// and score-delta all move the *wrong* way; median and p90 cannot see it at
/// all. The corruption destroys the blank plateau that was anchoring `would`
/// to its second realisation and lets the trellis collapse onto the oracle's
/// answer — to the exact millisecond.
///
/// So the ledger pins the divergence set by identity. A **new** divergence
/// fails (a fresh defect); the **disappearance** of this one fails too, and on
/// this clip that disappearance is the ANE's entire signature.
const TED_60_EXPECTED_DIVERGENCES: &[(usize, Boundary)] = &[(96, Boundary::End)];

/// The acoustic offset of the **second** spoken `would`, in ms: **31,940**,
/// frame 1597 — the last frame at which a greedy argmax over alignkit's
/// emissions still reads a letter of `WOULD` before the blank that precedes
/// `come`.
///
/// Deliberately taken from the **greedy argmax**, not from any forced
/// alignment, so the constant this test measures alignkit against is not
/// derived from alignkit's own trellis. Corroborated independently by the RMS
/// envelope (speech energy, no silent run) and by the verbatim two-`would`
/// alignment (which ends its second `would` at 31,981.3 ms, 41 ms later — two
/// frames, the usual CTC offset lag).
const ACOUSTIC_OFFSET_OF_SECOND_WOULD_MS: f64 = 31_940.0;

/// How far alignkit's `would` offset may sit from
/// [`ACOUSTIC_OFFSET_OF_SECOND_WOULD_MS`]: **3 frames** — the same tolerance
/// [`MAX_ASK_ONSET_ERROR_MS`] gives jfk's un-refereeable boundary, for the same
/// reason (a CTC offset frame trails the acoustic one by a frame or two).
///
/// Measured: **+41.3 ms**, inside two frames.
///
/// **This is the check that catches the ANE on this clip**, and it is the one
/// that runs FIRST: corrupted, alignkit puts the offset at 31,741.2 ms —
/// **198.8 ms** from the acoustic evidence, 3.3× over this bound. The oracle
/// cannot referee the boundary (it is the one calling that speech blank), so
/// the audio does.
const MAX_WOULD_OFFSET_ERROR_MS: f64 = 3.0 * FRAME_MS;

/// ted_60's largest tolerated **median** per-word score disagreement: `0.05`.
/// Measured: **0.0134** — four times tighter than jfk's 0.0838, again because
/// no zero-padding is perturbing the emissions.
///
/// Bounds a **systematic confidence regression**. It does **NOT** catch the
/// ANE — measured: the corrupted median score delta *falls* to 0.0076, for the
/// same reason its timing statistics improve. Recorded so nobody mistakes it
/// for a safety net.
const MAX_TED_60_MEDIAN_SCORE_DELTA: f32 = 0.05;

/// Which end of a word a delta belongs to. Named, because
/// [`JFK_EXPECTED_DIVERGENCES`] pins divergences by identity and "word 14" alone
/// would not say whether it is the onset (which is what the oracle gets wrong)
/// or the offset (which it does not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
  Start,
  End,
}

/// Word PTS as `(start_ms, end_ms)`. Both aligners are built with a clock
/// anchored at stream sample 0 in [`ANALYSIS_TIMEBASE`], so a PTS **is** a
/// 16 kHz sample index.
fn ms(range: TimeRange) -> (f64, f64) {
  (
    range.start_pts() as f64 / SAMPLES_PER_MS,
    range.end_pts() as f64 / SAMPLES_PER_MS,
  )
}

/// One clip's measured alignkit-vs-oracle comparison.
///
/// Built by [`compare`], which owns every mechanic the two clips share — the
/// tokenization-identity and word-sequence preconditions, the per-word delta
/// table, the robust statistics, and the gross-divergence classification. The
/// per-clip *bounds* deliberately do NOT live here: they are properties of the
/// audio, not of the harness, and each one is justified against its own clip's
/// measurements at its own call site.
///
/// # There is deliberately no `max` field
///
/// [`compare`] computes and PRINTS the maximum boundary delta, because it is
/// the first thing a human wants when a bound trips — but it does not hand one
/// back, because **a max-delta bound is the single trap this gate exists to
/// avoid**. On both clips the maximum moves the *wrong way* under the ANE
/// corruption (jfk 908.0 → 87.1 ms; ted_60 240.1 → 20.1 ms), so any assertion
/// built on it would pass the corrupt build and fail the correct one. The
/// worst-case boundary is named in a ledger ([`JFK_EXPECTED_DIVERGENCES`],
/// [`TED_60_EXPECTED_DIVERGENCES`]) and checked against the AUDIO instead.
/// Not offering the number is the cheapest way to stop someone reaching for it.
struct Comparison {
  ak_words: Vec<Word>,
  ort_words: Vec<Word>,
  median: f64,
  p90: f64,
  median_score: f32,
  gross: Vec<(usize, Boundary)>,
}

/// Runs both aligners on one clip, asserts the two preconditions that make a
/// per-word delta meaningful at all, and reduces the result to [`Comparison`].
///
/// The preconditions are asserted *here*, once, for both clips: identical OOV
/// event streams (the two heads carry different vocabularies — 29-class chordai
/// vs 32-class HuggingFace — so an equal transcript is not by itself proof that
/// equal TOKENS reach the two trellises) and identical word sequences (different
/// words means there is no per-word delta to take).
fn compare(
  clip: &str,
  alignkit: &Aligner,
  ort: &mut OrtAligner,
  samples: &[f32],
  text: &str,
) -> Comparison {
  assert_eq!(
    alignkit.detect_oov(text).expect("alignkit detect_oov"),
    ort.detect_oov(text).expect("asry-ort detect_oov"),
    "[{clip}] the two vocabularies disagree about which characters are out-of-vocabulary, so the \
     two trellises are not being handed the same tokens and their word timings are not comparable"
  );

  let ak_words = align_with_alignkit(alignkit, samples, text);
  let ort_words = align_with_asry_ort(ort, samples, text);

  assert!(
    !ak_words.is_empty(),
    "[{clip}] a real transcript over matching audio must produce words"
  );

  let ak_texts: Vec<&str> = ak_words.iter().map(Word::text).collect();
  let ort_texts: Vec<&str> = ort_words.iter().map(Word::text).collect();
  assert_eq!(
    ak_texts, ort_texts,
    "[{clip}] the two aligners produced different WORDS, not merely different timings for the same \
     words — there is no meaningful per-word delta to take"
  );

  let mut boundary_deltas: Vec<f64> = Vec::with_capacity(ak_words.len() * 2);
  let mut score_deltas: Vec<f32> = Vec::with_capacity(ak_words.len());
  let mut gross: Vec<(usize, Boundary)> = Vec::new();

  println!(
    "\n=== {clip} ===\n{:<12} {:>10} {:>10} {:>9} {:>10} {:>10} {:>9} {:>8}",
    "word", "ak.start", "ort.start", "Δstart", "ak.end", "ort.end", "Δend", "Δscore"
  );
  for (i, (ak, orw)) in ak_words.iter().zip(&ort_words).enumerate() {
    let (ak_start, ak_end) = ms(ak.range());
    let (ort_start, ort_end) = ms(orw.range());
    let (d_start, d_end) = (ak_start - ort_start, ak_end - ort_end);
    let d_score = ak.score() - orw.score();

    println!(
      "{:<12} {ak_start:>10.1} {ort_start:>10.1} {d_start:>+9.1} {ak_end:>10.1} {ort_end:>10.1} \
       {d_end:>+9.1} {d_score:>+8.4}",
      ak.text()
    );

    boundary_deltas.push(d_start.abs());
    boundary_deltas.push(d_end.abs());
    score_deltas.push(d_score.abs());
    if d_start.abs() > GROSS_DELTA_MS {
      gross.push((i, Boundary::Start));
    }
    if d_end.abs() > GROSS_DELTA_MS {
      gross.push((i, Boundary::End));
    }
  }

  boundary_deltas.sort_by(f64::total_cmp);
  score_deltas.sort_by(f32::total_cmp);
  let median = percentile(&boundary_deltas, 0.50);
  let p90 = percentile(&boundary_deltas, 0.90);
  let p95 = percentile(&boundary_deltas, 0.95);
  let max = *boundary_deltas.last().expect("at least two boundaries");
  let within_one_frame = boundary_deltas.iter().filter(|d| **d <= FRAME_MS).count();
  let median_score = score_deltas[score_deltas.len() / 2];

  println!(
    "\n[{clip}] {} words, {} boundaries | median {median:.1} ms | p90 {p90:.1} ms | p95 {p95:.1} \
     ms | max {max:.1} ms | within 1 frame ({FRAME_MS:.0} ms): {within_one_frame}/{} | median \
     Δscore {median_score:.4} | gross (>{GROSS_DELTA_MS:.0} ms): {gross:?}\n",
    ak_words.len(),
    boundary_deltas.len(),
    boundary_deltas.len(),
  );

  Comparison {
    ak_words,
    ort_words,
    median,
    p90,
    median_score,
    gross,
  }
}

/// Nearest-rank percentile over an already-sorted slice. `p` in `[0, 1]`.
fn percentile(sorted: &[f64], p: f64) -> f64 {
  assert!(!sorted.is_empty(), "percentile of an empty sample");
  let rank = (p * sorted.len() as f64).ceil() as usize;
  sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Loads alignkit's aligner on the shipping defaults.
///
/// `Aligner::from_paths` → `AlignerOptions::new()` → `DEFAULT_ENCODER_COMPUTE`.
/// **Never a hardcoded compute placement**: a gate pinned to a compute unit
/// proves only that compute unit, and this crate's default *is* the thing under
/// test — the previous default (`ComputeUnits::All`) silently corrupted every
/// emission tensor while every model-gated test, each pinned to `CpuOnly`,
/// stayed green.
fn load_alignkit() -> Aligner {
  Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect(
    "build alignkit's En aligner from the CoreML model + bundled tokenizer (set \
     ALIGNKIT_TEST_MODELS to the model directory)",
  )
}

/// ort rc.12's macOS fallback dylib name — the bare name it hands to `dlopen`
/// when `ORT_DYLIB_PATH` is unset, empty, or non-UTF-8
/// (`ort-2.0.0-rc.12/src/lib.rs:195`).
const ORT_DYLIB_BARE_NAME: &str = "libonnxruntime.dylib";

/// Resolve the `dlopen` target EXACTLY as ort rc.12 does, so the preflight probes
/// the library ort will actually load — never one ort would not use.
///
/// Mirrors ort's selection verbatim (`ort-2.0.0-rc.12/src/lib.rs:188-197`):
/// ```text
/// let path = match std::env::var("ORT_DYLIB_PATH") {
///     Ok(s) if !s.is_empty() => s,      // UTF-8 AND non-empty → that path
///     _ => "libonnxruntime.dylib",      // absent / empty / non-UTF-8 → bare name
/// };
/// ```
/// The three subtleties a hand-rolled resolver gets wrong — each pinned by a
/// regression test below:
/// - `std::env::var` (UTF-8), **not** `var_os`: a non-UTF-8 value is `Err`, so ort
///   IGNORES it and falls back to the bare name
///   ([`preflight_non_utf8_ort_dylib_path_is_ignored_like_ort`]).
/// - the `!s.is_empty()` guard: `ORT_DYLIB_PATH=""` is NOT an explicit path — ort
///   falls back to the bare name
///   ([`preflight_empty_ort_dylib_path_falls_back_to_bare_name_like_ort`]).
/// - a **relative** value (`libonnxruntime.dylib`) is used as-is and resolved by
///   `dlopen`/dyld — which consults `DYLD_LIBRARY_PATH` — never by a `cwd` check
///   ([`preflight_relative_ort_dylib_path_resolves_via_dyld_like_ort`]).
fn select_ort_dylib_target() -> OsString {
  match std::env::var("ORT_DYLIB_PATH") {
    Ok(explicit) if !explicit.is_empty() => OsString::from(explicit),
    _ => OsString::from(ORT_DYLIB_BARE_NAME),
  }
}

/// Fails loudly if `ort` will not be able to `dlopen` ONNX Runtime — because
/// if it cannot, it **hangs instead of erroring**, and a gate that hangs is
/// worse than one that fails.
///
/// `ort` runs in `load-dynamic` mode: it resolves `libonnxruntime.dylib` at
/// *runtime*, not link time. When the library is not resolvable, ort does not
/// return `Err` — it **deadlocks**. `ort::setup_api` runs inside a
/// `std::sync::Once`, and its failure path constructs the error through
/// `ort::error::Error::new_internal`, which re-enters the **same** `Once`; the
/// thread parks in `semaphore_wait_trap` and never comes back. Measured here,
/// not inferred: `cargo test -p alignkit --features parity-oracle` sat for ten
/// minutes printing "has been running for over 60 seconds" while the process
/// burned **0.01 s of CPU**, and `sample(1)` showed precisely that stack —
/// `load_asry_ort` → `SessionBuilder::new` → `environment::current` →
/// `Once::call` → `setup_api` → `Once::call` → `Error::new_internal` →
/// `Once::wait` → `semaphore_wait_trap`.
///
/// In CI that is a job that burns to its timeout and reports nothing
/// actionable; here it looked exactly like "the 60 s clip is just slow", which
/// is the kind of wrong conclusion that gets a gate weakened. So resolve the
/// library up front and panic with something a human can act on.
///
/// **Existence is not loadability** (F3). A file at the resolved path can still
/// be a text file, a wrong-architecture dylib, a library missing `OrtGetApiBase`,
/// or a runtime too OLD to provide the API version ort needs — each passes an
/// `is_file()` check and then hits the very deadlock above. So the preflight does
/// not stop at existence: it actually `dlopen`s the library, resolves
/// `OrtGetApiBase`, and CALLS it to confirm `GetApi` yields a usable `OrtApi` at
/// the required version — the entry point ort itself calls — in a CHILD PROCESS
/// with a timeout ([`probe_ort_dylib_loadable`]), so a hanging loader kills the
/// child instead of this test, and any unusable runtime becomes a fast,
/// actionable panic rather than a silent hang.
///
/// Target selection mirrors ort rc.12 EXACTLY ([`select_ort_dylib_target`],
/// `ort-2.0.0-rc.12/src/lib.rs:188`), and the bounded child's `dlopen` — the same
/// call ort ultimately makes (`load_dylib_from_path` → `libloading::Library::new`,
/// ort `lib.rs:92`) — is the SOLE loadability judge: no `is_file()` pre-check and
/// no hand-rolled dyld directory list, because `dlopen` already performs dyld's
/// real search (`DYLD_LIBRARY_PATH`, then the defaults, then
/// `DYLD_FALLBACK_LIBRARY_PATH`), so reimplementing it could only DIVERGE from
/// what ort will actually load. Homebrew's `/opt/homebrew/lib` is on none of those
/// default lists on Apple Silicon — which is why `brew install onnxruntime` on its
/// own is not enough, and `ORT_DYLIB_PATH` is effectively mandatory.
fn assert_onnxruntime_is_resolvable() {
  const HINT: &str = "ONNX Runtime is the parity gate's ORACLE; without it there is nothing to \
                      compare alignkit against. Install it (`brew install onnxruntime`) and point \
                      ORT_DYLIB_PATH at the dylib, e.g. \
                      ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib. Do NOT skip this \
                      test instead — a parity gate that opts itself out is not a gate.";

  // Resolve the dlopen TARGET exactly as ort would (mirrors ort's own selection;
  // see `select_ort_dylib_target`), then let the bounded child's `dlopen` be the
  // ONLY loadability judge — existence is not loadability (F3), and dlopen is the
  // same call ort ultimately makes, so any `is_file()`/dyld-list pre-check could
  // only diverge from what ort will actually load.
  let target = select_ort_dylib_target();
  if let Err(why) = probe_ort_dylib_loadable(&target) {
    panic!(
      "ort cannot load ONNX Runtime from {}: {why}. ort resolves ONNX Runtime with the SAME \
       dlopen and, when the load fails, DEADLOCKS rather than returning an error, so failing here \
       instead. {HINT}",
      PathBuf::from(&target).display()
    );
  }
}

/// Env var carrying, into the re-exec'd child, the dylib path/name it must try
/// to `dlopen`. Set only on the child's environment by
/// [`probe_ort_dylib_loadable`]; absent in an ordinary run, where the child test
/// is a no-op.
const ORT_PROBE_TARGET_ENV: &str = "ALIGNKIT_ORT_PREFLIGHT_DLOPEN";

/// Child exit code: the file could not be `dlopen`ed at all (not a loadable
/// dylib for this architecture — a text file, a wrong-arch binary, …).
const PROBE_EXIT_LOAD_FAILED: i32 = 2;
/// Child exit code: it loaded, but does not export `OrtGetApiBase` — so it is
/// not ONNX Runtime.
const PROBE_EXIT_NO_SYMBOL: i32 = 3;
/// Child exit code: it loaded and exports `OrtGetApiBase`, but that entry
/// point's `GetApi(`[`ORT_REQUIRED_API_VERSION`]`)` returned null — the runtime
/// is too OLD to provide the `OrtApi` version ort was built against, so ort
/// could not have used it either. This is the codex-round-5 gap: resolving the
/// symbol without calling it accepted such a runtime, which then wedged ort's
/// own init.
const PROBE_EXIT_API_REJECTED: i32 = 4;

/// The `ORT_API_VERSION` ort negotiates with the loaded runtime — the exact
/// argument ort's `setup_api` passes to `OrtApiBase::GetApi`
/// (`((*base).GetApi)(ort_sys::ORT_API_VERSION)`, `ort-2.0.0-rc.12/src/lib.rs:210`).
///
/// DERIVED from the authoritative compiled value, never hand-written. ort defines
/// `pub const MINOR_VERSION: u32 = ort_sys::ORT_API_VERSION;`
/// (`ort-2.0.0-rc.12/src/lib.rs:86`), so `asry::ort::MINOR_VERSION` — asry
/// re-exports `ort` under its `alignment` feature, which `parity-oracle` turns on
/// (`asry/src/lib.rs:100`) — IS the very number ort feeds to `GetApi`. Sourcing it
/// from that symbol is what makes probe/ORT drift impossible BY CONSTRUCTION: a
/// hand-written `24` could silently disagree with a bumped `ort_sys` (a runtime
/// that satisfies the stale probe yet fails ort's real init); this cannot, and a
/// future ort/ort-sys bump moves it automatically with no constant to forget.
///
/// The probe must pass THIS number to `GetApi`: a runtime that supports a lower
/// version but not this one is exactly the mismatch ort cannot survive.
const ORT_REQUIRED_API_VERSION: u32 = asry::ort::MINOR_VERSION;

/// Prove ONNX Runtime at `target` is USABLE by ort — `dlopen` it, resolve
/// `OrtGetApiBase`, and CALL it to confirm `GetApi(`[`ORT_REQUIRED_API_VERSION`]`)`
/// returns a non-null `OrtApi` (the exact negotiation ort's `setup_api` performs)
/// — in a CHILD PROCESS with a timeout, so a hanging loader kills the child
/// rather than wedging this test. `Ok(())` iff the child loaded the library,
/// found the symbol, AND the runtime supports the API version ort requires.
///
/// The load runs in a re-exec of THIS test binary
/// ([`ort_preflight_dlopen_child`]), so the `unsafe` `libloading` call is
/// confined to the test crate and the alignkit library stays unsafe-free. The
/// timeout is a poll loop over [`std::process::Child::try_wait`] (no extra
/// dependency) that kills the child if it overruns.
fn probe_ort_dylib_loadable(target: &OsStr) -> Result<(), String> {
  // Generous: a good dlopen of ONNX Runtime is sub-second, so this only bounds a
  // genuine hang.
  const TIMEOUT: Duration = Duration::from_secs(30);
  const POLL: Duration = Duration::from_millis(50);

  let exe = std::env::current_exe().map_err(|e| format!("cannot find the test binary: {e}"))?;
  let mut child = Command::new(exe)
    // Run ONLY the child probe test — libtest's exact filter, plus `--ignored`
    // because that test opts out of ordinary runs.
    .args(["--exact", "--ignored", "ort_preflight_dlopen_child"])
    .env(ORT_PROBE_TARGET_ENV, target)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .spawn()
    .map_err(|e| format!("cannot spawn the dlopen probe subprocess: {e}"))?;

  let start = Instant::now();
  loop {
    match child.try_wait() {
      Ok(Some(status)) => {
        return match status.code() {
          Some(0) => Ok(()),
          Some(PROBE_EXIT_LOAD_FAILED) => Err(
            "dlopen failed — not a loadable ONNX Runtime dylib for this architecture".to_owned(),
          ),
          Some(PROBE_EXIT_NO_SYMBOL) => {
            Err("loaded, but does not export OrtGetApiBase — not ONNX Runtime".to_owned())
          }
          Some(PROBE_EXIT_API_REJECTED) => Err(format!(
            "loaded and exports OrtGetApiBase, but GetApi({ORT_REQUIRED_API_VERSION}) returned \
             null — this ONNX Runtime does not support the ORT_API_VERSION ort was built against \
             (needs a runtime providing API {ORT_REQUIRED_API_VERSION}, e.g. ONNX Runtime >= \
             1.{ORT_REQUIRED_API_VERSION}); ort would fail unrecoverably resolving this very API \
             inside its init `Once`, so failing here instead"
          )),
          Some(other) => Err(format!("the dlopen probe exited with status {other}")),
          None => Err("the dlopen probe was killed by a signal".to_owned()),
        };
      }
      Ok(None) => {
        if start.elapsed() >= TIMEOUT {
          let _ = child.kill();
          let _ = child.wait();
          return Err(format!(
            "the loader did not return within {}s — a hanging dlopen, exactly the deadlock the \
             preflight exists to convert into a fast failure",
            TIMEOUT.as_secs()
          ));
        }
        std::thread::sleep(POLL);
      }
      Err(e) => return Err(format!("cannot wait on the dlopen probe subprocess: {e}")),
    }
  }
}

/// Just enough of ONNX Runtime's `OrtApiBase` for the child probe: its FIRST
/// field, `GetApi`, at offset 0. `#[repr(C)]` fixes that offset; the real
/// struct's trailing fields (`GetVersionString`, …) are omitted because the
/// probe only ever reads field 0 through a `*const OrtApiBase` and never handles
/// the value by size. Mirrors `ort_sys::OrtApiBase`; `extern "C"` equals
/// `ort_sys`' `extern "system"` on macOS (this crate's only target) and keeps
/// the test free of an `ort_sys` dependency, which would drag ort into every
/// build (see `Cargo.toml`).
#[repr(C)]
struct OrtApiBase {
  /// `GetApi(version) -> *const OrtApi`; returns null when the runtime does not
  /// support the requested `ORT_API_VERSION`. The result is only null-checked,
  /// never dereferenced, so its pointee is elided to `c_void`.
  get_api: unsafe extern "C" fn(u32) -> *const c_void,
}

/// The child half of [`probe_ort_dylib_loadable`], re-exec'd by it with
/// [`ORT_PROBE_TARGET_ENV`] set: it `dlopen`s the target, resolves
/// `OrtGetApiBase`, and CALLS it to check that
/// `GetApi(`[`ORT_REQUIRED_API_VERSION`]`)` is non-null, then exits with a status
/// the parent reads. Absent that env var — i.e. in any ordinary `--ignored`
/// run — it is a passing no-op.
#[test]
#[ignore = "internal ONNX Runtime dlopen-probe subprocess; a no-op unless re-exec'd by the parity preflight"]
fn ort_preflight_dlopen_child() {
  let Some(target) = std::env::var_os(ORT_PROBE_TARGET_ENV) else {
    return;
  };
  // SAFETY: loading an arbitrary dylib runs its initializers, which is why the
  // call is `unsafe` and why it runs in this short-lived CHILD process the parent
  // kills on timeout. The unsafe is confined to this test-crate helper; the
  // alignkit library is unsafe-free.
  let library = match unsafe { libloading::Library::new(&target) } {
    Ok(library) => library,
    Err(_) => std::process::exit(PROBE_EXIT_LOAD_FAILED),
  };
  // SAFETY: resolving `OrtGetApiBase` by name. The signature mirrors
  // `ort_sys::OrtGetApiBase` — `extern "system"`, which is ABI-identical to
  // `extern "C"` on macOS, this crate's only target. A missing symbol means this
  // is not ONNX Runtime.
  let get_api_base =
    match unsafe { library.get::<unsafe extern "C" fn() -> *const OrtApiBase>(b"OrtGetApiBase\0") }
    {
      Ok(get_api_base) => get_api_base,
      Err(_) => std::process::exit(PROBE_EXIT_NO_SYMBOL),
    };
  // SAFETY: `OrtGetApiBase` and `OrtApiBase::GetApi` are ONNX Runtime's own
  // version-negotiation entry points — pure accessors that return static
  // pointers and take no locks, so unlike ort's `setup_api` they cannot deadlock
  // or run environment init. Calling them HERE, in the timeout-bounded child, is
  // what makes the check safe: it reproduces ort's
  // `((*base).GetApi)(ort_sys::ORT_API_VERSION)` (ort `lib.rs`) and, on a null
  // result — the runtime refusing the required version — lets the parent fail
  // fast instead of the real process unwrapping that null inside its API-init
  // `Once`. A null base or a null `GetApi` result both mean "no usable OrtApi at
  // the required version".
  let api = unsafe {
    let base = get_api_base();
    if base.is_null() {
      std::process::exit(PROBE_EXIT_API_REJECTED);
    }
    ((*base).get_api)(ORT_REQUIRED_API_VERSION)
  };
  std::process::exit(if api.is_null() {
    PROBE_EXIT_API_REJECTED
  } else {
    0
  });
}

/// Switches [`ort_preflight_selection_child`] out of its no-op mode into running
/// the REAL selection + probe against the child's own environment. Set only on
/// that child's environment by [`run_ort_selection_probe`]; absent in an ordinary
/// run, where the child test is a no-op.
const ORT_SELECT_PROBE_ENV: &str = "ALIGNKIT_ORT_PREFLIGHT_SELECT";

/// [`ort_preflight_selection_child`] exit codes — the decoded verdict of the real
/// selection + probe for a crafted environment. Kept distinct from the
/// `PROBE_EXIT_*` codes (which the grandchild dlopen probe reports) so the two
/// process layers can never be confused.
const SELECT_EXIT_ACCEPTED: i32 = 10;
/// The selected target dlopened to a runtime that REFUSED the required API
/// version — unreachable unless selection produced a bare/relative name that dyld
/// resolved to the refusing stub, which is exactly what makes it a positive proof
/// of resolution (a real, accepting runtime could never forge it).
const SELECT_EXIT_API_REJECTED: i32 = 11;
/// The selected target did not `dlopen` at all (not found / not loadable).
const SELECT_EXIT_LOAD_FAILED: i32 = 12;
/// Any other probe verdict — surfaced rather than swallowed so a surprising
/// outcome fails a regression test loudly instead of masquerading as a pass.
const SELECT_EXIT_OTHER: i32 = 13;

/// Sibling of [`ort_preflight_dlopen_child`] for the SELECTION layer: re-exec'd by
/// [`run_ort_selection_probe`] with [`ORT_SELECT_PROBE_ENV`] set, it runs the REAL
/// [`select_ort_dylib_target`] against its own (parent-crafted) `ORT_DYLIB_PATH`
/// and hands the result to the REAL [`probe_ort_dylib_loadable`], then exits with
/// the decoded verdict — so the parent observes exactly what the preflight would
/// decide for that environment, with no reimplementation to drift. Absent the env
/// var — i.e. in any ordinary `--ignored` run — it is a passing no-op.
#[test]
#[ignore = "internal ORT_DYLIB_PATH selection-probe subprocess; a no-op unless re-exec'd by the L1 regression tests"]
fn ort_preflight_selection_child() {
  if std::env::var_os(ORT_SELECT_PROBE_ENV).is_none() {
    return;
  }
  let code = match probe_ort_dylib_loadable(&select_ort_dylib_target()) {
    Ok(()) => SELECT_EXIT_ACCEPTED,
    // The verdict strings are `probe_ort_dylib_loadable`'s own — the same anchors
    // the sibling `preflight_rejects_*` tests assert on.
    Err(why) if why.contains("returned null") => SELECT_EXIT_API_REJECTED,
    Err(why) if why.contains("dlopen failed") => SELECT_EXIT_LOAD_FAILED,
    Err(_) => SELECT_EXIT_OTHER,
  };
  std::process::exit(code);
}

/// **F3 unit test.** The loadability validator REJECTS a file that exists but is
/// not a loadable dylib. Hermetic — a text file in a temp dir, no ONNX Runtime,
/// no models — and NOT `#[ignore]`, so it runs wherever `parity-oracle` is built
/// (e.g. `cargo hack --each-feature`). It is the standing proof that
/// `Path::is_file()` alone — the old preflight — was never enough: this decoy
/// passes `is_file()` and would have sailed straight into ort's deadlock.
#[test]
fn preflight_rejects_a_non_dylib_file() {
  let dir = tempfile::tempdir().expect("create a temp dir");
  let not_a_dylib = dir.path().join("libonnxruntime.dylib");
  std::fs::write(&not_a_dylib, b"I am a text file, not a Mach-O dylib.\n")
    .expect("write the decoy file");
  assert!(
    not_a_dylib.is_file(),
    "the decoy must exist, so this proves loadability is checked BEYOND existence"
  );

  let err = probe_ort_dylib_loadable(not_a_dylib.as_os_str())
    .expect_err("a text file is not a loadable dylib and must be rejected");
  assert!(
    err.contains("dlopen failed"),
    "expected a load failure for a non-dylib, got: {err}"
  );
}

/// C source for a *loadable* dylib that impersonates ONNX Runtime's version
/// negotiation but REFUSES the API version ort requires: its `OrtGetApiBase`
/// returns a base whose `GetApi` yields non-null only for versions BELOW
/// `REQUIRED_API_VERSION` (a `-D` define) and null at or above it — modelling a
/// too-old runtime (e.g. ONNX Runtime 1.23 against ort's api-24). Returning
/// non-null for a lower version is deliberate: it pins that the probe calls
/// `GetApi` with the REQUIRED version, so a probe passing some lower version
/// would wrongly accept this stub and fail the test.
const API_REFUSING_STUB_C: &str = r#"
#include <stdint.h>

typedef struct OrtApiBase {
  const void *(*GetApi)(uint32_t version);
  const char *(*GetVersionString)(void);
} OrtApiBase;

static const char sentinel_api = 0;

static const void *stub_get_api(uint32_t version) {
  if (version < REQUIRED_API_VERSION) {
    return (const void *)&sentinel_api;
  }
  return (const void *)0;
}

static const char *stub_get_version_string(void) {
  return "0.0.0-alignkit-stub";
}

static const OrtApiBase base = { stub_get_api, stub_get_version_string };

const OrtApiBase *OrtGetApiBase(void) {
  return &base;
}
"#;

/// C source for a *loadable* dylib that is NOT ONNX Runtime: it exports a marker
/// but no `OrtGetApiBase`, so the probe must reach the "loaded, but does not
/// export OrtGetApiBase" verdict — a path a non-dylib file (which fails at
/// `dlopen`) can never exercise.
const SYMBOLLESS_STUB_C: &str = r#"
int alignkit_not_onnxruntime(void) {
  return 0;
}
"#;

/// Compiles `c_source` into a loadable dylib named `lib{name}.dylib` under `dir`
/// and returns its path, applying each `-D{key}={value}` define.
///
/// The C compiler (`$CC`, else `cc`) ships with the Xcode Command Line Tools that
/// building this crate already requires, and `cc -dynamiclib` is the lightest
/// mechanism that yields a genuinely *loadable* Mach-O dylib — the only kind that
/// exercises symbol and API-version resolution rather than just the
/// `dlopen`-fails path — with no build script, no fixture crate, and no `unsafe`
/// in this crate (the compile is a subprocess; the only FFI is the already-isolated
/// child probe). Hermetic: source and output live under the caller's temp `dir`.
fn compile_stub_dylib(dir: &Path, name: &str, c_source: &str, defines: &[(&str, &str)]) -> PathBuf {
  let source = dir.join(format!("{name}.c"));
  std::fs::write(&source, c_source).expect("write the stub C source");
  let dylib = dir.join(format!("lib{name}.dylib"));
  let compiler = std::env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));

  let mut command = Command::new(&compiler);
  command.arg("-dynamiclib").arg("-o").arg(&dylib);
  for (key, value) in defines {
    command.arg(format!("-D{key}={value}"));
  }
  command.arg(&source);

  let status = command.status().unwrap_or_else(|e| {
    panic!("cannot run the C compiler {compiler:?} to build stub dylib {name}: {e}")
  });
  assert!(
    status.success(),
    "the C compiler failed to build stub dylib {name}"
  );
  assert!(
    dylib.is_file(),
    "stub dylib was not produced at {}",
    dylib.display()
  );
  dylib
}

/// **F3 unit test.** The loadability validator REJECTS a *loadable* dylib that
/// exports `OrtGetApiBase` but whose `GetApi` refuses the API version ort
/// requires — the codex-round-5 gap: the old probe stopped at symbol resolution,
/// so such a runtime sailed through the preflight and into ort's own
/// unrecoverable init failure. Hermetic: a stub dylib compiled on the fly, no
/// ONNX Runtime, no models — and NOT `#[ignore]`, so it runs wherever
/// `parity-oracle` is built. Because the stub accepts every version BELOW
/// [`ORT_REQUIRED_API_VERSION`] and refuses that one, it also pins that the probe
/// negotiates the *correct* version, not merely some lower one.
#[test]
fn preflight_rejects_a_dylib_that_refuses_the_required_api_version() {
  let dir = tempfile::tempdir().expect("create a temp dir");
  // Source the stub's refusal threshold DIRECTLY from the authoritative symbol —
  // NOT from `ORT_REQUIRED_API_VERSION`, the probe's own binding. Two independent
  // reads of `asry::ort::MINOR_VERSION` (`== ort_sys::ORT_API_VERSION`, ort
  // lib.rs:86) are what make this test a probe/ORT DRIFT DETECTOR: were the probe
  // ever pinned to a stale literal below the compiled version, the stub would
  // accept the version the probe asks for, the rejection would vanish, and the
  // `expect_err` below would fail. (Mutation-proven: dropping the probe's constant
  // by one turns this test RED.)
  let required = asry::ort::MINOR_VERSION.to_string();
  let stub = compile_stub_dylib(
    dir.path(),
    "ort_api_refuse",
    API_REFUSING_STUB_C,
    &[("REQUIRED_API_VERSION", required.as_str())],
  );

  let err = probe_ort_dylib_loadable(stub.as_os_str())
    .expect_err("a runtime refusing the required API version must be rejected");
  assert!(
    err.contains(&format!("GetApi({ORT_REQUIRED_API_VERSION})")),
    "expected an API-version rejection naming GetApi({ORT_REQUIRED_API_VERSION}), got: {err}"
  );
}

/// **F3 unit test.** The validator REJECTS a *loadable* dylib that is not ONNX
/// Runtime — it loads but exports no `OrtGetApiBase`. Distinct from
/// [`preflight_rejects_a_non_dylib_file`], which fails during `dlopen`: this
/// proves the symbol-missing verdict on a dylib that genuinely loaded. Hermetic,
/// and NOT `#[ignore]`.
#[test]
fn preflight_rejects_a_loadable_dylib_without_ort_get_api_base() {
  let dir = tempfile::tempdir().expect("create a temp dir");
  let stub = compile_stub_dylib(dir.path(), "not_onnxruntime", SYMBOLLESS_STUB_C, &[]);

  let err = probe_ort_dylib_loadable(stub.as_os_str())
    .expect_err("a loadable dylib that is not ONNX Runtime must be rejected");
  assert!(
    err.contains("does not export OrtGetApiBase"),
    "expected a missing-symbol rejection, got: {err}"
  );
}

/// Compiles the API-refusing stub as `libonnxruntime.dylib` under a fresh temp dir
/// and returns the dir — put it on `DYLD_LIBRARY_PATH` and a bare
/// `dlopen("libonnxruntime.dylib")` resolves to it. Refusing (not accepting) is
/// deliberate: the resulting [`SELECT_EXIT_API_REJECTED`] verdict is UNREACHABLE
/// unless the selection produced a bare/relative name that dyld resolved to THIS
/// stub, so it is a positive proof of resolution no real, accepting runtime could
/// forge. The stub refuses `asry::ort::MINOR_VERSION`, the exact version the probe
/// requests (`ORT_REQUIRED_API_VERSION`).
fn onnxruntime_stub_on_dyld_path() -> tempfile::TempDir {
  let dir = tempfile::tempdir().expect("create a temp dir");
  let required = asry::ort::MINOR_VERSION.to_string();
  compile_stub_dylib(
    dir.path(),
    // `lib{name}.dylib` → `libonnxruntime.dylib`, the bare name dyld resolves.
    "onnxruntime",
    API_REFUSING_STUB_C,
    &[("REQUIRED_API_VERSION", required.as_str())],
  );
  dir
}

/// Spawns [`ort_preflight_selection_child`] with a crafted `ORT_DYLIB_PATH` and a
/// `DYLD_LIBRARY_PATH` pointing at `dyld_library_path`, and returns the child's
/// exit code — the preflight's decoded verdict for that environment. The env is
/// crafted on the CHILD (never mutated in-process — `set_var` is racy and its
/// safety cannot be guaranteed here), and `ort_dylib_path` is an `&OsStr` so a
/// caller can pass a non-UTF-8 value.
fn run_ort_selection_probe(ort_dylib_path: &OsStr, dyld_library_path: &Path) -> i32 {
  let exe = std::env::current_exe().expect("locate the test binary");
  Command::new(exe)
    .args(["--exact", "--ignored", "ort_preflight_selection_child"])
    .env(ORT_SELECT_PROBE_ENV, "1")
    .env("ORT_DYLIB_PATH", ort_dylib_path)
    .env("DYLD_LIBRARY_PATH", dyld_library_path)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status()
    .expect("spawn the selection-probe child")
    .code()
    .expect("the selection-probe child was killed by a signal, not a code")
}

/// **L1 regression.** `ORT_DYLIB_PATH=""` must be treated as ort treats it — NOT
/// as an explicit (missing) path, but as the bare-name fallback: ort's
/// `Ok(s) if !s.is_empty()` guard fails for `""` (`ort-2.0.0-rc.12/src/lib.rs:188`).
/// With the stub on `DYLD_LIBRARY_PATH`, the bare `dlopen` resolves to it and the
/// probe reports the stub's API refusal; the OLD preflight `is_file("")`-rejected
/// an empty path and diverged from ort. Hermetic (a compiled stub, no ONNX
/// Runtime, no models) and NOT `#[ignore]`, so it runs wherever `parity-oracle`
/// builds.
#[test]
fn preflight_empty_ort_dylib_path_falls_back_to_bare_name_like_ort() {
  let stub_dir = onnxruntime_stub_on_dyld_path();
  let verdict = run_ort_selection_probe(OsStr::new(""), stub_dir.path());
  assert_eq!(
    verdict, SELECT_EXIT_API_REJECTED,
    "empty ORT_DYLIB_PATH must fall back to the bare name (ort lib.rs:188) and dlopen the stub on \
     DYLD_LIBRARY_PATH, reaching its API refusal; got exit {verdict}"
  );
}

/// **L1 regression.** A RELATIVE `ORT_DYLIB_PATH=libonnxruntime.dylib` reachable
/// via `DYLD_LIBRARY_PATH` must be loaded through dyld — as ort does
/// (`load_dylib_from_path` → `libloading::Library::new`, ort `lib.rs:92`) — not
/// rejected by a `cwd`-relative `is_file()` check, the OLD preflight's bug (it
/// probed the cwd only, so ort would load this dylib while the preflight refused
/// it). Hermetic and NOT `#[ignore]`.
#[test]
fn preflight_relative_ort_dylib_path_resolves_via_dyld_like_ort() {
  let stub_dir = onnxruntime_stub_on_dyld_path();
  let verdict = run_ort_selection_probe(OsStr::new(ORT_DYLIB_BARE_NAME), stub_dir.path());
  assert_eq!(
    verdict, SELECT_EXIT_API_REJECTED,
    "a relative ORT_DYLIB_PATH must resolve through dyld's DYLD_LIBRARY_PATH search and reach the \
     stub's API refusal, not be cwd-`is_file`-rejected; got exit {verdict}"
  );
}

/// **L1 regression.** A NON-UTF-8 `ORT_DYLIB_PATH` must be IGNORED exactly as ort
/// ignores it: ort reads it with `std::env::var` (UTF-8 only,
/// `ort-2.0.0-rc.12/src/lib.rs:188`), so a non-UTF-8 value is `Err` and ort falls
/// back to the bare name. The OLD preflight used `var_os` and probed the garbage
/// path. With the stub on `DYLD_LIBRARY_PATH`, the bare-name fallback reaches its
/// API refusal. Hermetic and NOT `#[ignore]`.
#[test]
fn preflight_non_utf8_ort_dylib_path_is_ignored_like_ort() {
  use std::os::unix::ffi::OsStrExt;
  // "foo" + bytes that are not valid UTF-8, so the child's `std::env::var` returns
  // `Err` — the very case `var_os` would have wrongly surfaced as an explicit path.
  let non_utf8 = OsStr::from_bytes(&[b'f', b'o', b'o', 0x80, 0xff]);
  let stub_dir = onnxruntime_stub_on_dyld_path();
  let verdict = run_ort_selection_probe(non_utf8, stub_dir.path());
  assert_eq!(
    verdict, SELECT_EXIT_API_REJECTED,
    "a non-UTF-8 ORT_DYLIB_PATH must be ignored (ort reads UTF-8 via std::env::var, ort \
     lib.rs:188) and fall back to the bare name, reaching the stub's API refusal; got exit \
     {verdict}"
  );
}

/// Loads the oracle. Separate from [`align_with_asry_ort`] so the ONNX session
/// (a ~378 MB model) is built once and reused across a test's runs.
fn load_asry_ort() -> OrtAligner {
  assert_onnxruntime_is_resolvable();
  OrtAligner::from_paths(
    Lang::En,
    &common::asry_onnx_model_path(),
    &common::asry_tokenizer_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect(
    "load asry's ONNX wav2vec2 oracle + its 32-class tokenizer (set ALIGNKIT_ASRY_MODELS to \
     asry's models/ directory; `ort` is load-dynamic, so libonnxruntime.dylib must also be on \
     the loader path)",
  )
}

/// Runs alignkit's CoreML pipeline. `speech` is passed explicitly rather than
/// as `&[]` so this side and the oracle are spelled the same way — see
/// [`align_with_asry_ort`]'s doc for why an empty slice is a trap.
fn align_with_alignkit(aligner: &Aligner, samples: &[f32], text: &str) -> Vec<Word> {
  let events = aligner.detect_oov(text).expect("alignkit detect_oov");
  let decisions = default_oov_decisions(&events);
  // Clock anchored at stream sample 0 in the analysis timebase, so word PTS are
  // 16 kHz sample indices.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);

  aligner
    .align_chunk(
      samples,
      &whole_chunk_is_speech(samples),
      text,
      clock,
      &abort,
      &decisions,
    )
    .expect("alignkit align_chunk succeeds end-to-end")
    .words()
    .to_vec()
}

/// Runs asry's ONNX-Runtime pipeline — the oracle — on its own defaults.
///
/// `Aligner::from_paths` gives hop 320, `SpeechCoverage::DEFAULT` (0.5) and
/// `DEFAULT_MAX_INTRA_SILENT_RUN` (80 ms): the same three knobs alignkit's
/// `AlignerOptions::new()` sets to the same three values, so the two pipelines
/// differ in the encoder and nothing else.
///
/// `align_chunk` (rather than `align_chunk_with_abort`) is deliberate: it
/// applies `default_oov_decisions` to its own `detect_oov` output internally —
/// exactly the policy [`align_with_alignkit`] applies — so the OOV path cannot
/// drift between the two sides through a hand-written argument.
///
/// The closure is asry's pre-`OutputClock` bridge; anchoring the chunk at
/// stream sample 0 and emitting [`ANALYSIS_TIMEBASE`] PTS reproduces
/// `OutputClock::new(0, ANALYSIS_TIMEBASE, 0)` exactly (that clock's `range`
/// rescales `ANALYSIS_TIMEBASE → ANALYSIS_TIMEBASE`, the identity, and adds a
/// zero `base_pts`).
fn align_with_asry_ort(aligner: &mut OrtAligner, samples: &[f32], text: &str) -> Vec<Word> {
  aligner
    .align_chunk(
      samples,
      &whole_chunk_is_speech(samples),
      text,
      0,
      |start, end| TimeRange::new(start as i64, end as i64, ANALYSIS_TIMEBASE),
    )
    .expect("asry-ort align_chunk succeeds end-to-end")
    .words()
    .to_vec()
}

/// "No VAD" — one span covering the whole chunk, in the 1/16000 analysis
/// timebase.
///
/// # Why this is spelled out and never `&[]`
///
/// The two front ends disagree about what an empty `sub_segments` slice means,
/// and they disagree in the most dangerous possible direction:
///
/// | call | empty `sub_segments` means |
/// |---|---|
/// | `alignkit::Aligner::align_chunk` | `SpeechSpans::all_speech()` — every sample is speech |
/// | `asry::Aligner::align_chunk` | `SpeechSpans::from_time_ranges(&[])` → **no speech at all** |
///
/// asry's own `SpeechSpans` doc calls this out ("mean 'no VAD' and get 'all
/// silence', which dropped every word"), and its `EmissionsAligner` seam exists
/// partly to force the distinction into the type — but the ORT front end still
/// takes the raw slice. Handing `&[]` to the oracle would silence-mask the
/// entire buffer to zeros and then compare alignkit's real alignment against an
/// oracle that had been fed 11 seconds of digital silence: the same class of
/// defect as the padded-vs-unpadded harness bug this gate's fingerprint exists
/// to prevent, and just as quiet — it produces numbers, only meaningless ones.
///
/// One span over `[0, samples.len())` is the faithful translation on both
/// sides: `all_speech()` is `[0, SampleSpan::MAX_SAMPLE)`, which the mask and
/// the frame classifier both clamp to the chunk's real length, so the two are
/// the same mask and the same 1.0 coverage on every word. Spelling it
/// identically for both aligners removes the asymmetry entirely.
fn whole_chunk_is_speech(samples: &[f32]) -> [TimeRange; 1] {
  [TimeRange::new(0, samples.len() as i64, ANALYSIS_TIMEBASE)]
}

/// Decodes the fixture and asserts it is the audio these numbers were measured
/// on.
fn jfk_samples() -> Vec<f32> {
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  assert_eq!(
    common::sha256_samples_hex(&samples),
    common::JFK_SAMPLES_SHA256,
    "the decoded jfk.wav buffer is not the audio this gate's numbers were measured on; a parity \
     number from two different inputs measures the harness, not the models"
  );
  samples
}

/// Decodes `ted_60.wav` and asserts **both** things the unpadded half depends
/// on: that it is the audio these numbers were measured on, and that it still
/// fills the encoder window *exactly*.
///
/// The length assertion is not a tautology of the digest — it is the one that
/// says what this fixture is FOR. At exactly [`alignkit::encode::ENCODER_WINDOW_SAMPLES`]
/// samples, `Encoder::emissions_raw` borrows the caller's buffer and appends no
/// zeros at all; one sample fewer and it silently takes the zero-fill branch,
/// which would quietly convert this test back into a second copy of the padded
/// jfk case and retire the only coverage the unpadded path has. Spelled out, so
/// that a re-encoded fixture fails HERE, loudly, instead of passing while
/// measuring nothing new.
fn ted_60_samples() -> Vec<f32> {
  let samples = common::load_wav_mono_f32(&common::ted_60_wav_path());
  assert_eq!(
    samples.len(),
    alignkit::encode::ENCODER_WINDOW_SAMPLES,
    "ted_60.wav is {} samples, not the {} that exactly fill the encoder window. This fixture's \
     entire purpose is the ZERO-PADDING-FREE path (`emissions_raw`'s `Cow::Borrowed` branch); at \
     any other length alignkit pads, and this test degenerates into a second padded clip.",
    samples.len(),
    alignkit::encode::ENCODER_WINDOW_SAMPLES,
  );
  assert_eq!(
    common::sha256_samples_hex(&samples),
    common::TED_60_SAMPLES_SHA256,
    "the decoded ted_60.wav buffer is not the audio this gate's numbers were measured on; a parity \
     number from two different inputs measures the harness, not the models"
  );
  samples
}

/// **The gate.** Per-word start/end/score agreement between alignkit (CoreML)
/// and asry (ONNX Runtime) on real speech.
///
/// The fixture is `jfk.wav` — **real speech, and it has to be**. Synthetic input
/// is not merely weaker here, it is actively misleading: measured on the
/// known-corrupt ANE path, 60 s of digital silence bottoms out at `-8.55` and a
/// sine at `-9.07`, both *above* fp16's `log` floor (≈ `-16.6`), so neither
/// underflows and **both pass on a corrupted model**. Only real speech drives a
/// per-class posterior to `e^-30.8 ≈ 4e-14`, under the floor, where the failure
/// this crate ships a `CpuOnly` default to avoid actually lives.
#[test]
#[ignore = "requires local alignkit + asry models (ALIGNKIT_TEST_MODELS, ALIGNKIT_ASRY_MODELS)"]
fn word_timings_agree_with_asry_ort_on_jfk() {
  let samples = jfk_samples();
  let text = common::JFK_TRANSCRIPT;

  let alignkit = load_alignkit();
  let mut ort = load_asry_ort();
  let c = compare("jfk", &alignkit, &mut ort, &samples, text);

  // ---- FIRST: the boundary the oracle cannot referee, refereed by the audio
  // The one boundary the ledger permits is the one the oracle gets WRONG. It is
  // not exempt from checking — it is checked against something better than the
  // oracle, and this check runs before any comparison to the oracle does,
  // because the audio outranks it.
  assert_eq!(c.ak_words[14].text(), "ask", "word 14 is no longer `ask`");
  let (ask_start, _) = ms(c.ak_words[14].range());
  let onset_error = (ask_start - ACOUSTIC_ONSET_OF_ASK_MS).abs();
  println!(
    "`ask` onset: alignkit {ask_start:.1} ms vs ACOUSTIC onset {ACOUSTIC_ONSET_OF_ASK_MS:.1} ms \
     (error {onset_error:+.1} ms); the oracle says {:.1} ms, which is inside the silence.\n",
    ms(c.ort_words[14].range()).0,
  );
  assert!(
    onset_error <= MAX_ASK_ONSET_ERROR_MS,
    "alignkit places the second `ask` at {ask_start:.1} ms, {onset_error:.1} ms from the true \
     acoustic onset at {ACOUSTIC_ONSET_OF_ASK_MS:.1} ms (bound: {MAX_ASK_ONSET_ERROR_MS:.1} ms). \
     The oracle cannot referee this boundary — it places it 873 ms into digital silence — so the \
     audio does. This is exactly the word the ANE's corrupted emissions displace, to 7533.7 ms."
  );

  // ---- then the comparison to the oracle ---------------------------------
  assert!(
    c.median <= MAX_MEDIAN_BOUNDARY_DELTA_MS,
    "median word-boundary disagreement {:.1} ms exceeds {MAX_MEDIAN_BOUNDARY_DELTA_MS:.1} ms (one \
     frame) — more than half of all boundaries moved. That is a SYSTEMATIC shift (a stride, a \
     truncation off-by-one, a clock anchor), not encoder precision.",
    c.median
  );
  assert!(
    c.p90 <= MAX_P90_BOUNDARY_DELTA_MS,
    "p90 word-boundary disagreement {:.1} ms exceeds {MAX_P90_BOUNDARY_DELTA_MS:.1} ms — the BULK \
     of the distribution moved, not just a tail. Do NOT widen this bound; read the per-word table \
     above.",
    c.p90
  );
  assert_eq!(
    c.gross.as_slice(),
    JFK_EXPECTED_DIVERGENCES,
    "the ledger of gross (>{GROSS_DELTA_MS:.0} ms) divergences from the oracle changed. Each \
     entry must be a KNOWN, root-caused boundary — see JFK_EXPECTED_DIVERGENCES. A NEW one is a \
     finding to investigate and document, never an entry to append until the test is green. A \
     MISSING one is worse: agreeing with an oracle that is demonstrably wrong is the ANE \
     corruption's own signature."
  );
  assert!(
    c.median_score <= MAX_MEDIAN_SCORE_DELTA,
    "median per-word score disagreement {:.4} exceeds {MAX_MEDIAN_SCORE_DELTA:.4} — the two \
     encoders systematically disagree about how confident the alignment is",
    c.median_score
  );
}

/// **The gate's unpadded half** — the configuration
/// [`word_timings_agree_with_asry_ort_on_jfk`] structurally cannot reach.
///
/// `jfk.wav` is 176,000 samples against a 960,000-sample encoder window, so
/// **81.7% of what CoreML sees is zeros alignkit appended**. Every number that
/// test produces is a sum of two effects — the encoder swap and the padding —
/// and it cannot separate them;
/// `fixed_window_padding_does_not_explain_the_divergence` exists only to bound
/// the second one. `ted_60.wav` is **exactly** 960,000 samples
/// ([`ted_60_samples`] asserts it), so `Encoder::emissions_raw` borrows the
/// caller's buffer and appends **no zeros at all**: both encoders see the
/// identical 960,000 real samples, and the encoder swap is the *only*
/// difference left. This is the stronger comparison, and it needs no control.
///
/// It also runs the parts of the pipeline jfk leaves cold: the full
/// 2,999-frame emission tensor rather than 550 (`truncated_frame_count`'s
/// clamp engages, dropping nothing), a trellis over 186 words rather than 22,
/// and — because the transcript is real ASR output over spontaneous speech — a
/// disfluency the transcript elides, which is exactly where a forced aligner
/// is forced to guess.
#[test]
#[ignore = "requires local alignkit + asry models (ALIGNKIT_TEST_MODELS, ALIGNKIT_ASRY_MODELS)"]
fn word_timings_agree_with_asry_ort_on_ted_60() {
  let samples = ted_60_samples();
  let text = common::TED_60_TRANSCRIPT;

  let alignkit = load_alignkit();
  let mut ort = load_asry_ort();
  let c = compare("ted_60", &alignkit, &mut ort, &samples, text);

  // ---- FIRST: the boundary the oracle cannot referee, refereed by the audio
  // The oracle is the side that calls a confidently-decoded `WOULD` blank here
  // (see TED_60_EXPECTED_DIVERGENCES), so it does not get a vote. The audio
  // does, and it votes before any comparison to the oracle happens.
  assert_eq!(
    c.ak_words[96].text(),
    "would",
    "word 96 is no longer `would`"
  );
  let (_, would_end) = ms(c.ak_words[96].range());
  let offset_error = (would_end - ACOUSTIC_OFFSET_OF_SECOND_WOULD_MS).abs();
  println!(
    "`would` offset: alignkit {would_end:.1} ms vs ACOUSTIC offset of the second spoken `would` \
     {ACOUSTIC_OFFSET_OF_SECOND_WOULD_MS:.1} ms (error {offset_error:+.1} ms); the oracle says \
     {:.1} ms, which ends the word before that `would` is spoken.\n",
    ms(c.ort_words[96].range()).1,
  );
  assert!(
    offset_error <= MAX_WOULD_OFFSET_ERROR_MS,
    "alignkit ends `would` at {would_end:.1} ms, {offset_error:.1} ms from the acoustic offset of \
     the second spoken `would` at {ACOUSTIC_OFFSET_OF_SECOND_WOULD_MS:.1} ms (bound: \
     {MAX_WOULD_OFFSET_ERROR_MS:.1} ms). The speaker says `would` TWICE and the transcript names \
     it once; the oracle resolves that by calling the second one blank, so it cannot referee this \
     boundary — the audio does. This is exactly the boundary the ANE's corrupted emissions \
     collapse onto the oracle's answer, at 31741.2 ms."
  );

  // ---- then the comparison to the oracle ---------------------------------
  assert!(
    c.median <= MAX_TED_60_MEDIAN_BOUNDARY_DELTA_MS,
    "median word-boundary disagreement {:.1} ms exceeds {MAX_TED_60_MEDIAN_BOUNDARY_DELTA_MS:.1} \
     ms (one frame) — more than half of all boundaries moved. With no padding in play that is a \
     SYSTEMATIC shift (a stride, a truncation off-by-one, a clock anchor), not encoder precision: \
     unpadded, the median boundary is normally frame-IDENTICAL (0.0 ms).",
    c.median
  );
  assert!(
    c.p90 <= MAX_TED_60_P90_BOUNDARY_DELTA_MS,
    "p90 word-boundary disagreement {:.1} ms exceeds {MAX_TED_60_P90_BOUNDARY_DELTA_MS:.1} ms \
     (one frame). Unpadded, 90%+ of boundaries land on the SAME FRAME (measured p90: 0.0 ms) — \
     the two encoders genuinely agree that closely once the zero-padding is removed. Do NOT widen \
     this bound to jfk's 5 frames; jfk needs those frames for its padding, and this clip has none. \
     Read the per-word table above.",
    c.p90
  );
  assert_eq!(
    c.gross.as_slice(),
    TED_60_EXPECTED_DIVERGENCES,
    "the ledger of gross (>{GROSS_DELTA_MS:.0} ms) divergences from the oracle changed. Each entry \
     must be a KNOWN, root-caused boundary — see TED_60_EXPECTED_DIVERGENCES. A NEW one is a \
     finding to investigate and document, never an entry to append until the test is green. A \
     MISSING one is worse, and on THIS clip it is the ANE corruption's entire signature: every \
     other statistic here IMPROVES when the emissions are corrupted, and the vanishing of this \
     divergence is the only thing left that still says so."
  );
  assert!(
    c.median_score <= MAX_TED_60_MEDIAN_SCORE_DELTA,
    "median per-word score disagreement {:.4} exceeds {MAX_TED_60_MEDIAN_SCORE_DELTA:.4} — the two \
     encoders systematically disagree about how confident the alignment is",
    c.median_score
  );
}

/// **The control.** Answers the one question
/// [`word_timings_agree_with_asry_ort_on_jfk`]'s numbers cannot answer on their own:
/// **is the 908 ms `ask` divergence caused by alignkit's fixed 60 s window, or
/// by its CoreML conversion?**
///
/// alignkit zero-pads an 11 s chunk to 960,000 samples because its CoreML graph
/// takes no other shape; asry's ONNX graph is variable-length and does not. That
/// asymmetry is real, and at the *emission* level it is not small —
/// wav2vec2-base group-norms over the whole sequence axis and attends globally
/// with no padding mask, so on a 3 s slice padding alone moved 13.4% of frame
/// argmaxes and produced a max-abs log-prob difference of 27.17
/// (`.superpowers/sdd/alignkit-gate1-diagnostic.md`). It is the obvious suspect.
///
/// So: run the oracle against **itself** — ONNX both times, same fp32 weights,
/// same transcript, same span, the zeros the only difference. It exonerates the
/// window: padded, the oracle still puts the `ask` onset at ~7.5 s, within a
/// frame or so of where it puts it unpadded. **The padding does not move that
/// word.** What moves it is the encoder conversion, into a region where the
/// emissions carry no information at all (see the module doc).
///
/// # Why onsets only
///
/// Not cherry-picking — a structural difference in what the two sides even
/// compute. alignkit **truncates its emissions** to the real audio's frame count
/// (`ceil(176_000 / 320) = 550` of 2999) before `finish` ever sees them, so its
/// trellis cannot path a token into the pad. asry's ORT front end fuses encode
/// and finish and offers no seam to truncate at, so its padded trellis runs over
/// all 2999 frames and the FINAL token is free to run out into the zeros — which
/// it does, to ~60 s. (Marking the pad as non-speech instead does not rescue it:
/// the coverage post-pass then simply drops the last word, 22 → 21.) That
/// artifact is a property of *not truncating*, which alignkit does not do; it
/// would answer a different question, and a loud one, drowning out this one.
/// Word ONSETS are unaffected by it and are exactly the quantity the gate's
/// outlier is about.
///
/// The end-blowout is worth stating plainly, because it is the affirmative case
/// for a design decision B3 made and B4 kept: **the emissions truncation is not
/// an optimisation, it is what makes the fixed-window bridge correct at all.**
#[test]
#[ignore = "requires asry's ONNX oracle (ALIGNKIT_ASRY_MODELS)"]
fn fixed_window_padding_does_not_explain_the_divergence() {
  let samples = jfk_samples();
  let text = common::JFK_TRANSCRIPT;

  let mut padded = samples.clone();
  padded.resize(alignkit::encode::ENCODER_WINDOW_SAMPLES, 0.0);

  let mut ort = load_asry_ort();
  let bare = align_with_asry_ort(&mut ort, &samples, text);
  let with_pad = align_with_asry_ort(&mut ort, &padded, text);

  let bare_texts: Vec<&str> = bare.iter().map(Word::text).collect();
  let padded_texts: Vec<&str> = with_pad.iter().map(Word::text).collect();
  assert_eq!(
    bare_texts, padded_texts,
    "zero-padding the oracle's input changed which WORDS it found, not merely where — the \
     comparison below has no common basis"
  );

  let mut onset_deltas: Vec<f64> = Vec::with_capacity(bare.len());
  println!(
    "\n{:<12} {:>10} {:>10} {:>9}",
    "word", "bare", "padded", "Δonset"
  );
  for (b, p) in bare.iter().zip(&with_pad) {
    let (b_start, p_start) = (ms(b.range()).0, ms(p.range()).0);
    println!(
      "{:<12} {b_start:>10.1} {p_start:>10.1} {:>+9.1}",
      b.text(),
      p_start - b_start
    );
    onset_deltas.push((p_start - b_start).abs());
  }

  let ask_shift = (ms(with_pad[14].range()).0 - ms(bare[14].range()).0).abs();
  let mut sorted = onset_deltas.clone();
  sorted.sort_by(f64::total_cmp);
  let (median, p90) = (percentile(&sorted, 0.50), percentile(&sorted, 0.90));
  println!(
    "\nORACLE vs ITSELF (ONNX both sides), unpadded vs 60 s zero-padded ONSETS: median \
     {median:.1} ms | p90 {p90:.1} ms | `ask` onset moved {ask_shift:.1} ms\n\
     => alignkit's fixed window is NOT what moves `ask` by 908 ms.\n"
  );

  assert!(
    ask_shift <= GROSS_DELTA_MS,
    "zero-padding alone moves the oracle's `ask` onset by {ask_shift:.1} ms, past the \
     {GROSS_DELTA_MS:.0} ms gross threshold. The fixed window, not the CoreML conversion, would \
     then be the prime suspect for the parity gate's 908 ms divergence, and the module doc's \
     root-cause analysis needs redoing."
  );
  assert!(
    p90 <= MAX_PADDING_P90_DELTA_MS,
    "zero-padding the oracle's own input to alignkit's 60 s window moves its word onsets by p90 \
     {p90:.1} ms, past the {MAX_PADDING_P90_DELTA_MS:.1} ms bound. The fixed-window bridge has \
     become the dominant error source, and the parity gate's numbers must be re-attributed \
     before they mean anything."
  );
}
