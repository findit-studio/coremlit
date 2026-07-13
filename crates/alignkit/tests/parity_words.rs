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
use std::path::PathBuf;

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
/// The search mirrors what ort and dyld actually do: `ORT_DYLIB_PATH` if set,
/// otherwise a bare `dlopen("libonnxruntime.dylib")`, which consults
/// `DYLD_LIBRARY_PATH` and then `DYLD_FALLBACK_LIBRARY_PATH` (default
/// `$HOME/lib:/usr/local/lib:/usr/lib`). Homebrew's `/opt/homebrew/lib` is on
/// **neither** list on Apple Silicon — which is why `brew install onnxruntime`
/// on its own is not enough, and `ORT_DYLIB_PATH` is effectively mandatory.
fn assert_onnxruntime_is_resolvable() {
  const LIB: &str = "libonnxruntime.dylib";
  const HINT: &str = "ONNX Runtime is the parity gate's ORACLE; without it there is nothing to \
                      compare alignkit against. Install it (`brew install onnxruntime`) and point \
                      ORT_DYLIB_PATH at the dylib, e.g. \
                      ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib. Do NOT skip this \
                      test instead — a parity gate that opts itself out is not a gate.";

  if let Some(explicit) = std::env::var_os("ORT_DYLIB_PATH") {
    let path = PathBuf::from(&explicit);
    assert!(
      path.is_file(),
      "ORT_DYLIB_PATH points at {}, but there is no file there. ort would DEADLOCK on this (it \
       builds the load failure inside a `Once` it is already holding), so failing here instead. \
       {HINT}",
      path.display()
    );
    return;
  }

  let mut dirs: Vec<PathBuf> = std::env::var_os("DYLD_LIBRARY_PATH")
    .iter()
    .flat_map(std::env::split_paths)
    .collect();
  match std::env::var_os("DYLD_FALLBACK_LIBRARY_PATH") {
    Some(fallback) => dirs.extend(std::env::split_paths(&fallback)),
    None => {
      dirs.extend(std::env::var_os("HOME").map(|h| PathBuf::from(h).join("lib")));
      dirs.push(PathBuf::from("/usr/local/lib"));
      dirs.push(PathBuf::from("/usr/lib"));
    }
  }

  assert!(
    dirs.iter().any(|dir| dir.join(LIB).is_file()),
    "ORT_DYLIB_PATH is unset and no {LIB} is on dyld's search path (looked in {dirs:?}). ort \
     resolves ONNX Runtime with a bare dlopen and, when that fails, DEADLOCKS rather than \
     returning an error — so this test would hang forever instead of telling you why. {HINT}"
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
