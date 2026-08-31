//! Autoregressive decoding.
//!
//! Runs the per-window decode loop ([`decode_text`]), one-shot language
//! detection ([`detect_language`]), and the prefill-token assembly that
//! feeds both ([`prefill_tokens`]) — ports `TextDecoder.swift`'s
//! `decodeText`/`detectLanguage`/`prefillDecoderInputs`
//! (`TextDecoder.swift:541-855`, `:420-539`, `:163-216`). Also home to
//! [`filter`], the logits-filter chain the loop runs after every step's
//! raw logits are produced, and [`sampler`], the greedy/seeded-top-k
//! next-token sampler it calls afterward.
//!
//! **Timing fields this sync port actually measures**, inside
//! [`decode_text`]/[`detect_language`]:
//! [`TranscriptionTimings::total_decoding_loops`](crate::audio::whisper::result::TranscriptionTimings::total_decoding_loops),
//! [`TranscriptionTimings::decoding_predictions`](crate::audio::whisper::result::TranscriptionTimings::decoding_predictions)
//! (wall-clock around [`InferenceBackend::decode_step`]),
//! [`TranscriptionTimings::decoding_sampling`](crate::audio::whisper::result::TranscriptionTimings::decoding_sampling)
//! (around [`sampler::GreedyTokenSampler::sample`]), and
//! [`TranscriptionTimings::decoding_filtering`](crate::audio::whisper::result::TranscriptionTimings::decoding_filtering)
//! (around the filter chain; [`decode_text`] only — Swift's
//! `detectLanguage` never times its own filter call either).
//! **Deliberately left untouched** (stay at whatever the caller's shared
//! `TranscriptionTimings` already held going in):
//! `first_token_time` — Swift stamps `CFAbsoluteTimeGetCurrent()`, an
//! absolute wall-clock reading; [`crate::audio::whisper::log`] exposes no clock at all,
//! and a bare [`std::time::Instant`] has no meaningful absolute value to
//! store in its place, so this field is skipped rather than faked;
//! `decoding_kv_caching`/`total_kv_update_runs` — the KV-cache update
//! Swift times as a separate block now happens *inside*
//! [`InferenceBackend::decode_step`], opaque to this module (a
//! deliberate deviation, not an oversight — see [`decode_text`]'s doc);
//! every pipeline-wide field (`pipeline_start`, `model_loading`,
//! `encoding`, `decoding_windowing`, ...) belongs to a higher
//! orchestration layer this module doesn't own.

use std::{
  cell::Cell,
  sync::atomic::{AtomicBool, Ordering},
  time::Instant,
};

use crate::audio::whisper::{
  backend::InferenceBackend,
  constants::{DEFAULT_LANGUAGE_CODE, MAX_TOKEN_CONTEXT, SECONDS_PER_TIME_TOKEN, language_code},
  decode::{
    filter::{
      LanguageLogitsFilter, LogitsFilter, SuppressBlankFilter, SuppressTokensFilter,
      TimestampRulesFilter,
    },
    sampler::GreedyTokenSampler,
  },
  error::DecodeError,
  options::DecodingOptions,
  result::{DecodingResult, TranscriptionProgress, TranscriptionTimings},
  text,
  tokenizer::{SpecialTokens, WhisperTokenizer},
};

pub mod filter;
pub mod sampler;

#[cfg(test)]
mod tests;

/// Numerically stable `max + ln(Σ exp(v - max))` (the log-sum-exp
/// normalizer of `logits`): subtracts the running max before
/// exponentiating so large logits don't overflow `f32::exp`, the same
/// shape as Swift's MLTensor `logSoftmax` normalizer at `f32` precision.
/// Used by [`sampler`]'s zero-temperature log-softmax to score the sampled
/// token. ([`filter`]'s timestamp-mass comparison used to share this, but
/// now replicates BNNS's f16 rounding structure directly in its own
/// `bnns_mass_rule_scalars`.) Returns
/// [`f32::NEG_INFINITY`] when `logits` is empty or every entry is already
/// [`f32::NEG_INFINITY`], rather than the `NaN` that `(-inf) - (-inf)`
/// would otherwise produce.
fn log_sum_exp(logits: &[f32]) -> f32 {
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  if !max.is_finite() {
    return f32::NEG_INFINITY;
  }
  max + logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

// ---------------------------------------------------------------------
// TranscriptionProgressCallback
// ---------------------------------------------------------------------

/// Per-step decode progress sink (Swift `TranscriptionCallback`,
/// `WhisperKit.swift`): called after every non-completed step of
/// [`decode_text`]'s loop — prefill steps included, matching Swift's
/// un-gated dispatch (`TextDecoder.swift:733-741`) — with the tokens/text
/// decoded so far. Returning `Some(false)` requests early stop, honored
/// only once past the prefill steps; `Some(true)`/`None` continue.
///
/// Dispatched **inline**, synchronously, unlike Swift's
/// `Task.detached(priority: .low)` (`TextDecoder.swift:734-741`) — this is
/// a sync library with no background executor to hop to, so the callback
/// simply runs on the calling thread before the loop proceeds to its next
/// iteration.
pub type TranscriptionProgressCallback<'a> =
  &'a (dyn Fn(&TranscriptionProgress) -> Option<bool> + Sync);

// ---------------------------------------------------------------------
// prefill_tokens
// ---------------------------------------------------------------------

/// Assembles the initial prompt token sequence fed to [`decode_text`].
/// Ports the token-assembly half of `prefillDecoderInputs`
/// (`TextDecoder.swift:163-216`) — the KV-cache/mask side effects at
/// `:211-213` are backend-internal state this port has no equivalent
/// field for ([`crate::audio::whisper::backend::InferenceBackend::new_decoder_state`]
/// already starts a fresh state at position 0).
///
/// Shape: `[<|startoftranscript|>]`; if `is_multilingual`, followed by
/// `<|{options.language() or DEFAULT_LANGUAGE_CODE}|>` (falling back to
/// [`SpecialTokens::english_token`] if that string isn't in the
/// vocabulary) and `<|{task}|>` (falling back to
/// [`SpecialTokens::transcribe_token`]); then
/// [`SpecialTokens::no_timestamps_token`] if `options.without_timestamps()`
/// else [`SpecialTokens::time_token_begin`]. If `options.prompt_tokens_slice()`
/// is non-empty, its last `MAX_TOKEN_CONTEXT / 2 - 1` elements (Swift
/// `.suffix`), filtered to ids `< special_token_begin`, are prepended as
/// `[start_of_previous_token] + trimmed + <everything above>`. If
/// `options.prefix_tokens_slice()` is non-empty, its last
/// `MAX_TOKEN_CONTEXT / 2` elements, filtered the same way, are appended.
///
/// Swift's `options == nil` no-prefill branch (`:180-209` all skipped) has
/// no equivalent parameter here — a caller that wants to skip prefill
/// simply passes `&[start_of_transcript_token]` straight to
/// [`decode_text`] instead of calling this function at all, mirroring
/// `TranscribeTask.swift:83+90`.
pub fn prefill_tokens(
  options: &DecodingOptions,
  tokenizer: &WhisperTokenizer,
  is_multilingual: bool,
) -> Vec<u32> {
  let special = tokenizer.special_tokens();
  let mut tokens: Vec<u32> = vec![special.start_of_transcript_token()];

  if is_multilingual {
    let lang = if options.language().is_empty() {
      DEFAULT_LANGUAGE_CODE
    } else {
      options.language()
    };
    let language_token = tokenizer
      .token_to_id(&format!("<|{lang}|>"))
      .unwrap_or_else(|| special.english_token());
    tokens.push(language_token);

    let task_token = tokenizer
      .token_to_id(&format!("<|{}|>", options.task().as_str()))
      .unwrap_or_else(|| special.transcribe_token());
    tokens.push(task_token);
  }

  let timestamps_token = if options.without_timestamps() {
    special.no_timestamps_token()
  } else {
    special.time_token_begin()
  };
  tokens.push(timestamps_token);

  let prompt_tokens = options.prompt_tokens_slice();
  if !prompt_tokens.is_empty() {
    let max_prompt_len = MAX_TOKEN_CONTEXT / 2 - 1;
    let start = prompt_tokens.len().saturating_sub(max_prompt_len);
    let trimmed = prompt_tokens[start..]
      .iter()
      .copied()
      .filter(|&t| t < special.special_token_begin());
    let mut prefixed = Vec::with_capacity(1 + (prompt_tokens.len() - start) + tokens.len());
    prefixed.push(special.start_of_previous_token());
    prefixed.extend(trimmed);
    prefixed.extend(tokens);
    tokens = prefixed;
  }

  let prefix_tokens = options.prefix_tokens_slice();
  if !prefix_tokens.is_empty() {
    let start = prefix_tokens.len().saturating_sub(MAX_TOKEN_CONTEXT / 2);
    tokens.extend(
      prefix_tokens[start..]
        .iter()
        .copied()
        .filter(|&t| t < special.special_token_begin()),
    );
  }

  tokens
}

// ---------------------------------------------------------------------
// create_logits_filters
// ---------------------------------------------------------------------

/// Assembles the per-run logits filter chain, in order. Ports
/// `createLogitsFilters` (`TextDecoder.swift:857-899`); Swift's own
/// injectable `logitsFilters` extension point (`:864`) has no equivalent
/// parameter here — this port has no such extensibility hook at this
/// layer.
///
/// `sample_begin_prefilled` seeds [`SuppressBlankFilter`] (added only when
/// `options.suppress_blank()`); `initial_prompt_len` seeds
/// [`TimestampRulesFilter`] (added unless `options.without_timestamps()`,
/// with `max_initial_timestamp_index` derived from
/// `options.max_initial_timestamp() / SECONDS_PER_TIME_TOKEN`).
/// [`SuppressTokensFilter`] is added when `options.suppress_tokens_slice()`
/// is non-empty, with its ids filtered to `< special.special_token_begin()`.
/// Order matters: each filter mutates `logits` in place and
/// [`TimestampRulesFilter`]'s own mass-comparison rule reads whatever the
/// earlier filters already masked, so it must run last.
pub(crate) fn create_logits_filters(
  options: &DecodingOptions,
  sample_begin_prefilled: usize,
  initial_prompt_len: usize,
  special: &SpecialTokens,
  is_multilingual: bool,
) -> Vec<Box<dyn LogitsFilter>> {
  let mut filters: Vec<Box<dyn LogitsFilter>> = Vec::new();

  if options.suppress_blank() {
    filters.push(Box::new(SuppressBlankFilter::new(
      special,
      sample_begin_prefilled,
    )));
  }

  if !options.suppress_tokens_slice().is_empty() {
    let filtered: Vec<u32> = options
      .suppress_tokens_slice()
      .iter()
      .copied()
      .filter(|&t| t < special.special_token_begin())
      .collect();
    filters.push(Box::new(SuppressTokensFilter::new(filtered)));
  }

  if !options.without_timestamps() {
    let max_initial_timestamp_index = options
      .max_initial_timestamp()
      .map(|seconds| (seconds / SECONDS_PER_TIME_TOKEN) as usize);
    filters.push(Box::new(TimestampRulesFilter::new(
      special,
      initial_prompt_len,
      max_initial_timestamp_index,
      is_multilingual,
    )));
  }

  filters
}

// ---------------------------------------------------------------------
// decode_text
// ---------------------------------------------------------------------

/// Runs the autoregressive decode loop for one 30 s window, from
/// `initial_prompt` up to `options.sample_length()` tokens (capped at
/// `MAX_TOKEN_CONTEXT - 1`). Ports `decodeText`
/// (`TextDecoder.swift:541-855`); every numbered comment in the
/// implementation cites the Swift line range it ports.
///
/// `prefilled_index` — the KV slot the loop starts at — is hardcoded to
/// the constant `0` rather than threaded through as a parameter. Swift
/// reads it from `decoderInputs.cacheLength[0]`
/// (`TextDecoder.swift:555`), but that value is `0` on every path that
/// reaches `decodeText`: a freshly prepared `DecodingInputs` starts at
/// `0`, and `DecodingInputs.reset(maxTokenContext:)` — the only other
/// place `cacheLength[0]` is touched — explicitly zeroes it
/// (`Models.swift:312-313`). `TranscribeTask.swift:83` (fresh state) and
/// `:270-273` (reset before the next window) are the only two call sites
/// that ever reach `decodeText`, and both hold `cacheLength[0] == 0`. A
/// parameter that can never observably vary is not a parameter.
///
/// **Documented deviation — KV cache lives inside `decode_step` now:**
/// Swift's loop, after a non-completing step, explicitly copies the
/// model's new key/value slices into `decoderInputs`' persistent KV
/// tensors and advances two mask arrays (`TextDecoder.swift:688-721`).
/// This port's [`InferenceBackend::decode_step`] contract already folds
/// that KV/mask advance into the backend call itself (see its own doc
/// comment) — there is no separate "commit this step's KV" call for this
/// loop to make, so `TranscriptionTimings::decoding_kv_caching` stays `0.0`
/// (see this module's doc for the full list of timing fields this port does
/// and doesn't populate). The alignment weights are the one exception: they
/// are observed after the loop with no intervening prediction, so a
/// completing step's row must NOT land (whisper #41); `decode_step` only
/// STAGES the row and this loop commits it via
/// [`InferenceBackend::commit_alignment_row`] on the non-completing path
/// (Swift's `:709-717` slot), matching Swift's conditional update.
///
/// `observed_language_token` is an out-parameter for the genuine language
/// OBSERVATION, captured at token-RECOGNITION time (the first `<|lang|>` token
/// sampled into the predicted region, regardless of configured language) and set
/// BEFORE the next
/// fallible step — exactly the way `early_stop` and the sampler's own
/// `drew_from_rng` survive an errored decode. The finalized
/// [`DecodingResult::observed_language`] is then a READ of this already-captured
/// fact, not a re-scan of the finalized tokens; and because the caller owns the
/// cell, a decode that recognizes the language then errors on a LATER step still
/// surfaces the observation (coremlit issue #14, codex round 6
/// post-consolidation F2 — a predicted language used to die with an errored,
/// dropped VAD chunk because its string was only built at successful
/// finalization).
///
/// # Errors
/// [`DecodeError`] if a backend decode step fails, or a progress
/// callback's tokenizer decode fails.
#[allow(clippy::too_many_arguments)] // Mirrors Swift's decodeText argument surface; this
// signature is mandated by this port's own interface contract, and no
// natural subset of these eleven forms a cohesive struct without inventing
// one purely to dodge this lint.
pub fn decode_text<B>(
  backend: &B,
  encoder_output: &B::EncoderOutput,
  state: &mut B::DecoderState,
  initial_prompt: &[u32],
  sampler: &mut GreedyTokenSampler,
  options: &DecodingOptions,
  tokenizer: &WhisperTokenizer,
  timings: &mut TranscriptionTimings,
  early_stop: &AtomicBool,
  observed_language_token: &Cell<Option<u32>>,
  callback: Option<TranscriptionProgressCallback<'_>>,
) -> Result<DecodingResult, DecodeError>
where
  B: InferenceBackend,
{
  let special = *tokenizer.special_tokens();
  let dims = backend.dims();
  let prefilled_index = 0usize; // cacheLength[0] on fresh/reset state — see doc above.
  let initial_prompt_index = initial_prompt.len();
  let mut current_tokens: Vec<u32> = initial_prompt.to_vec();
  let mut log_probs: Vec<f32> = vec![0.0; current_tokens.len()];
  let mut next_token = *initial_prompt
    .last()
    .expect("initial_prompt must contain at least the start-of-transcript token");

  let filters = create_logits_filters(
    options,
    prefilled_index,
    initial_prompt_index,
    &special,
    dims.is_multilingual(),
  );

  // TextDecoder.swift:566 — min(sampleLength, maxTokenContext - 1).
  let loop_count = options.sample_length().min(MAX_TOKEN_CONTEXT - 1);
  let mut logits: Vec<f32> = Vec::with_capacity(dims.vocab());
  let mut is_first_token_log_prob_too_low = false;
  let mut first_token_log_prob = 0.0f32;

  for token_index in prefilled_index..loop_count {
    let is_prefill = token_index + 1 < initial_prompt_index; // :576
    let is_last_prefill_token = token_index + 1 == initial_prompt_index; // :577
    let is_first_token = token_index == prefilled_index; // :578

    if token_index < initial_prompt_index {
      // :581-594 — force prompt tokens, except a model-predicted timestamp
      // may replace a timestamp in the last prefill slot.
      let prompt_is_timestamp = current_tokens[token_index] >= special.time_token_begin();
      let model_predicted_timestamp = next_token >= special.time_token_begin();
      if !(is_last_prefill_token && prompt_is_timestamp && model_predicted_timestamp) {
        next_token = current_tokens[token_index];
      } else {
        current_tokens[token_index] = next_token;
      }
    }

    let step_start = Instant::now();
    backend.decode_step(next_token, token_index, encoder_output, state, &mut logits)?;
    timings.set_decoding_predictions(
      timings.decoding_predictions() + step_start.elapsed().as_secs_f64(),
    );

    let filter_start = Instant::now();
    for filter in &filters {
      filter.filter(&mut logits, &current_tokens); // :640-643
    }
    timings
      .set_decoding_filtering(timings.decoding_filtering() + filter_start.elapsed().as_secs_f64());

    let sample_start = Instant::now();
    let sample = sampler.sample(&logits); // :652
    timings
      .set_decoding_sampling(timings.decoding_sampling() + sample_start.elapsed().as_secs_f64());
    next_token = sample.token();
    let next_token_log_prob = sample.logprob();

    if is_first_token {
      first_token_log_prob = next_token_log_prob;
      if let Some(threshold) = options.first_token_logprob_threshold() {
        is_first_token_log_prob_too_low = next_token_log_prob < threshold; // :662-667
      }
    }
    // Recognize the genuine language OBSERVATION the instant its token lands in
    // the predicted region — the first `<|lang|>` token the model PREDICTS — and
    // stash it in the caller's cell BEFORE any completion or error exit. Running
    // it here, AHEAD of the `is_segment_completed` break below, is what lets a
    // token the model genuinely SAMPLED still register its language even when that
    // same first token trips a completion gate — the low-first-token-logprob
    // threshold or the context cap — instead of being dropped unobserved because
    // the break fired first (codex round 11, M1). An EOT completion cannot carry a
    // `<|lang|>` token, so latching before it is a no-op there; the cell (and so
    // `DecodingResult::observed_language` and the attempt sink's task facts) now
    // sees the sampled language regardless of which gate ends the loop. FIRST
    // wins, matching the finalized scan this replaces and the merge law's
    // first-observation rule; a sampled token at/after `initial_prompt_index` sits
    // in the predicted region, so `!is_prefill` alone is the predicted-region gate
    // (F2, codex round 6 post-consolidation).
    //
    // NOT gated on `options.language().is_empty()`: a PREDICTED `<|lang|>` token
    // is an OBSERVATION distinct from the configured/display language (round 10,
    // F1). A multilingual decode configured `language="en"` with
    // `without_timestamps` forces `[SOT, <|en|>, <|transcribe|>,
    // <|notimestamps|>]`, then the model can still PREDICT `<|es|>` at the first
    // free position: the display stays the Swift-faithful forced `<|en|>` while
    // the observation is the predicted `<|es|>`. Suppressing it under a
    // configured language contradicted the record's own contract
    // (`observed_language` is the outcome, never the configured input) and
    // recorded `None` for a run that plainly detected a language. The forced
    // prefill `<|en|>` is already excluded by `!is_prefill`, not by the config.
    if !is_prefill
      && observed_language_token.get().is_none()
      && tokenizer.all_language_tokens().contains(&next_token)
    {
      observed_language_token.set(Some(next_token));
    }

    let is_segment_completed = sample.completed()
      || current_tokens.len() >= MAX_TOKEN_CONTEXT - 1
      || is_first_token_log_prob_too_low; // :668-671

    if is_segment_completed {
      // :673-678. Swift still counts this iteration in
      // `totalDecodingLoops` before breaking (line 677, ahead of the
      // `break` on 678) — matched here rather than skipping the count.
      timings.set_total_decoding_loops(timings.total_decoding_loops() + 1.0);
      break;
    }

    if !is_prefill {
      current_tokens.push(next_token); // :682-686
      log_probs.push(next_token_log_prob);
    }
    // :709-717 — commit the step's staged alignment row. Reached only when
    // the `is_segment_completed` break above did NOT fire (Swift's `else`
    // branch), for prefill steps too (Swift's update block sits outside the
    // `!isPrefill` append gate); the early-stop break below comes AFTER this,
    // so an early-stopped final step's row is committed exactly as Swift's
    // is. The KV/mask advance still happened inside `decode_step` (see this
    // function's doc); only the alignment write is split out here.
    backend.commit_alignment_row(state);

    if let Some(callback) = callback {
      // :723-741 — dispatched inline; see `TranscriptionProgressCallback`'s
      // doc for why there's no `Task.detached` equivalent here.
      let word_tokens: Vec<u32> = current_tokens
        .iter()
        .copied()
        .filter(|&t| t < special.special_token_begin())
        .collect();
      let text_tokens = if options.skip_special_tokens() {
        &word_tokens
      } else {
        &current_tokens
      };
      let progress = TranscriptionProgress::new(
        timings.clone(),
        tokenizer.decode(text_tokens, false)?,
        current_tokens.clone(),
      )
      .with_avg_logprob(log_probs.iter().sum::<f32>() / log_probs.len() as f32)
      .with_compression_ratio(text::compression_ratio_of_tokens(&current_tokens));
      if callback(&progress) == Some(false) && !is_prefill {
        early_stop.store(true, Ordering::Relaxed);
      }
    }
    timings.set_total_decoding_loops(timings.total_decoding_loops() + 1.0);
    if early_stop.load(Ordering::Relaxed) {
      break; // :753-756
    }
  }

  // `early_stop` is set ONLY by a progress callback's `Some(false)` past the
  // prefill steps (see the loop above) — never by an ordinary EOT / context-cap
  // / low-first-token-logprob termination — so it is the honest witness of a
  // caller-CONTROLLED truncation, carried out on the result for
  // `Provenance::is_reproducible` (coremlit issue #14, codex round 5).
  // The genuine observation is a READ of the cell captured mid-loop, decoded to
  // its ISO code the same way the finalized display language is — never a re-scan
  // of the finalized tokens (F2). On the success path this feeds
  // `DecodingResult::observed_language`; on an errored decode the caller reads the
  // very same cell instead, so the observation survives either way.
  let observed_language = observed_language_token
    .get()
    .map(|token| {
      let decoded = tokenizer.decode(&[token], false)?;
      Ok::<String, DecodeError>(text::trim_special_token_chars(&decoded).to_string())
    })
    .transpose()?;
  Ok(
    finalize_decoding_result(
      current_tokens,
      log_probs,
      first_token_log_prob,
      sampler,
      options,
      tokenizer,
    )?
    .maybe_observed_language(observed_language)
    .maybe_early_stopped(early_stop.load(Ordering::Relaxed)),
  )
}

/// Finalizes a completed (or early-stopped) decode loop into a
/// [`DecodingResult`]. Ports `TextDecoder.swift:776-854`, the tail of
/// `decodeText` that runs after its `for` loop exits.
fn finalize_decoding_result(
  mut current_tokens: Vec<u32>,
  mut log_probs: Vec<f32>,
  first_token_log_prob: f32,
  sampler: &GreedyTokenSampler,
  options: &DecodingOptions,
  tokenizer: &WhisperTokenizer,
) -> Result<DecodingResult, DecodeError> {
  // Read off `tokenizer` rather than taking a redundant parameter — the
  // caller's own `special` is `*tokenizer.special_tokens()`, so threading it
  // in as well would only pad the argument list.
  let special = tokenizer.special_tokens();

  // :776 — appends EOT (+logprob 0.0) unless already present.
  sampler.finalize(&mut current_tokens, &mut log_probs);

  // :780-783 — SOT..=EOT inclusive; `finalize` above guarantees EOT
  // exists, so `end_index`'s fallback (mirroring Swift's own `?? count`,
  // itself unreachable for the same reason) never actually fires.
  let start_index = current_tokens
    .iter()
    .position(|&t| t == special.start_of_transcript_token())
    .unwrap_or(0);
  let end_index = current_tokens
    .iter()
    .position(|&t| t == special.end_token())
    .unwrap_or(current_tokens.len());
  let filtered_tokens = &current_tokens[start_index..=end_index];
  let filtered_log_probs = &log_probs[start_index..=end_index];

  let sum_log_probs: f32 = filtered_log_probs.iter().sum();
  let avg_log_probs = sum_log_probs / filtered_log_probs.len() as f32;

  let token_log_probs: Vec<(u32, f32)> = filtered_tokens
    .iter()
    .copied()
    .zip(filtered_log_probs.iter().copied())
    .collect();

  // :793-794 — compression ratio is computed over the special/timestamp-
  // filtered word tokens, unlike `tokens`/`text` below (the unfiltered
  // SOT..=EOT slice).
  let word_tokens: Vec<u32> = filtered_tokens
    .iter()
    .copied()
    .filter(|&t| t < special.special_token_begin())
    .collect();
  let final_compression_ratio = text::compression_ratio_of_tokens(&word_tokens);

  // :796-800 — `Extensions.rounded(3)`. Same half-away-from-zero formula
  // as `segment::rounded_to_places` (both port the identical Swift
  // extension, just at a different `decimal_places`); deliberately left
  // duplicated here rather than shared. `rounded_to_places` is
  // `pub(crate)`, so calling `crate::audio::whisper::segment::rounded_to_places(value, 3)`
  // from here would compile, but `decode` has no other reason to depend
  // on `segment` — a later pipeline stage built on top of decode's own
  // output — and reaching across that boundary (or adding a new
  // shared-utility module) for one two-line arithmetic formula is more
  // coupling than the duplication it would save.
  let temperature = (sampler.temperature() * 1000.0).round() / 1000.0;

  // :802 — upstream TODO, never actually computed by Swift either.
  let no_speech_prob = 0.0;

  // :804-826 — the DISPLAY language: Swift takes `options.language`, else the
  // FIRST recognized language token in the full SOT..=EOT slice — the forced
  // prefill `<|lang|>` included — else the default. Kept byte-for-byte so
  // `DecodingResult::language` stays Swift-faithful.
  let (language, language_probs) = if !options.language().is_empty() {
    (
      options.language().to_string(),
      vec![(options.language().to_string(), 0.0)],
    )
  } else {
    match filtered_tokens
      .iter()
      .position(|&t| tokenizer.all_language_tokens().contains(&t))
    {
      Some(index) => {
        let decoded = tokenizer.decode(&filtered_tokens[index..=index], false)?;
        let lang = text::trim_special_token_chars(&decoded).to_string();
        let prob = filtered_log_probs[index];
        (lang.clone(), vec![(lang, prob)])
      }
      None => {
        let lang = DEFAULT_LANGUAGE_CODE.to_string();
        (lang.clone(), vec![(lang, 0.0)])
      }
    }
  };

  // The Rust-only detection fact (no Swift equivalent) — the language the model
  // actually PREDICTED, as an ISO code — is no longer rescanned here. It is
  // RECOGNIZED at sampling time inside [`decode_text`]'s loop (the first
  // `<|lang|>` token in the predicted region, regardless of configured
  // language) and stashed in
  // the caller's `observed_language_token` cell, so it survives a decode that
  // errors after recognizing it (F2, codex round 6 post-consolidation);
  // `decode_text` decodes that captured token into `DecodingResult::observed_language`.
  // It carries the predicted STRING, not a boolean, and is scanned OFF the
  // predicted region rather than the display slice: the DISPLAY `language` above
  // follows Swift's first-in-the-WHOLE-slice rule and so reports a forced prefill
  // `<|en|>`, while the model may predict a DIFFERENT language after it (freely,
  // once `without_timestamps` drops the timestamp filter). Reconstructing the
  // observation from the display `language` would misrecord a forced `<|en|>` as
  // the detection when `<|es|>` was predicted (F1, codex round 5). The
  // Swift-compat display above is untouched. The named invariant
  // `language_observed_only_for_a_predicted_language_token` (decode/tests.rs)
  // pins exactly this: a CONFIGURED `"en"` whose model predicts `<|es|>` keeps
  // display `"en"` while `observed_language == Some("es")`.

  // :828 — decoded with `skipSpecialTokens: false` regardless of
  // `options.skipSpecialTokens` (that option only ever gates the live
  // per-step progress text above, never this final transcript).
  let text = tokenizer.decode(filtered_tokens, false)?;

  Ok(
    DecodingResult::new()
      .with_language(language)
      .with_language_probs(language_probs)
      .with_tokens(filtered_tokens.to_vec())
      .with_token_log_probs(token_log_probs)
      .with_text(text)
      .with_avg_logprob(avg_log_probs)
      .with_no_speech_prob(no_speech_prob)
      .with_temperature(temperature)
      .with_compression_ratio(final_compression_ratio)
      .with_first_token_log_prob(first_token_log_prob),
  )
}

// ---------------------------------------------------------------------
// detect_language
// ---------------------------------------------------------------------

/// One-step language-detection probe: decodes a single step from
/// `<|startoftranscript|>` at KV position `0`, restricts the resulting
/// logits to language tokens via [`LanguageLogitsFilter`], and argmaxes.
/// Ports `TextDecoder.detectLanguage` (`TextDecoder.swift:420-539`).
///
/// **Documented deviation:** this step runs through
/// [`InferenceBackend::decode_step`], which advances the backend's KV
/// cache/masks — Swift's probe skips cache updates entirely, setting
/// `inputIds`/`cacheLength` directly and never touching the KV tensors at
/// all (`TextDecoder.swift:456-457`). This function therefore calls
/// [`InferenceBackend::reset_decoder_state`] before returning — on every
/// path, errors included, so a caller that reuses `state` after an `Err`
/// (e.g. to retry) never decodes from the probe's stale row. Equivalence:
/// reset restores the exact initial mask state, and the probe's one stale
/// KV row is dead data — the model ignores cache beyond `cache_length ==
/// 0`, and the first real decode step overwrites position 0 anyway.
///
/// The probe samples through the caller's `sampler` (Swift passes the
/// fallback ladder's own attempt sampler, `TranscribeTask.swift:337-343`,
/// and the probe draws through it, `TextDecoder.swift:500`): at a nonzero
/// temperature the language choice is a top-k draw, not an argmax, and
/// the draw advances the same RNG stream the attempt's text tokens then
/// continue from. One attempt owns one sampler; every sample in the
/// attempt advances the same stream.
///
/// # Errors
/// [`DecodeError`] if the backend step or a tokenizer decode fails.
pub fn detect_language<B>(
  backend: &B,
  encoder_output: &B::EncoderOutput,
  state: &mut B::DecoderState,
  tokenizer: &WhisperTokenizer,
  sampler: &mut GreedyTokenSampler,
  timings: &mut TranscriptionTimings,
) -> Result<DecodingResult, DecodeError>
where
  B: InferenceBackend,
{
  let result = detect_language_probe(backend, encoder_output, state, tokenizer, sampler, timings);
  // Unconditional: the probe may have advanced KV/masks before failing
  // partway (see the deviation note above), so the error paths need the
  // reset as much as the success path does.
  backend.reset_decoder_state(state);
  result
}

/// The fallible body of [`detect_language`]; the public wrapper owns the
/// unconditional state reset.
fn detect_language_probe<B>(
  backend: &B,
  encoder_output: &B::EncoderOutput,
  state: &mut B::DecoderState,
  tokenizer: &WhisperTokenizer,
  sampler: &mut GreedyTokenSampler,
  timings: &mut TranscriptionTimings,
) -> Result<DecodingResult, DecodeError>
where
  B: InferenceBackend,
{
  let special = *tokenizer.special_tokens();
  let filter = LanguageLogitsFilter::new(tokenizer.all_language_tokens(), 0);
  let mut logits: Vec<f32> = Vec::with_capacity(backend.dims().vocab());

  let step_start = Instant::now();
  backend.decode_step(
    special.start_of_transcript_token(),
    0,
    encoder_output,
    state,
    &mut logits,
  )?;
  timings
    .set_decoding_predictions(timings.decoding_predictions() + step_start.elapsed().as_secs_f64());

  let prompt = [special.start_of_transcript_token()];
  filter.filter(&mut logits, &prompt);

  let sample_start = Instant::now();
  let sample = sampler.sample(&logits);
  timings.set_decoding_sampling(timings.decoding_sampling() + sample_start.elapsed().as_secs_f64());

  // :508-514 — Swift iterates `sampleResult.tokens` (`[SOT, sampled]`
  // after `tokenSampler.update` appends the new draw) and keeps only the
  // entries that are language tokens; SOT is never one, so this reduces
  // to just checking the single sampled token.
  let decoded = tokenizer.decode(&[sample.token()], false)?;
  let trimmed = text::trim_special_token_chars(&decoded).to_string();
  let mut language_probs: Vec<(String, f32)> = Vec::new();
  if tokenizer.all_language_tokens().contains(&sample.token()) {
    language_probs.push((trimmed.clone(), sample.logprob()));
  }

  // :516-524 — validated against the known language-code table, else the
  // default.
  let language = if language_code(&trimmed).is_some() {
    trimmed
  } else {
    DEFAULT_LANGUAGE_CODE.to_string()
  };

  // :525-538 — tokens/text stay empty; this is a probe, not a transcript.
  Ok(
    DecodingResult::new()
      .with_language(language)
      .with_language_probs(language_probs),
  )
}
