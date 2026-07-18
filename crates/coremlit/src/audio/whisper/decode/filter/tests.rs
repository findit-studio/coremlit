use super::*;
use crate::tokenizer::SpecialTokens;

const VOCAB: usize = 51865;
const NEG_INF: f32 = f32::NEG_INFINITY;

fn flat_logits() -> Vec<f32> {
  vec![0.0; VOCAB]
}

fn special() -> SpecialTokens {
  SpecialTokens::whisper_defaults()
}

#[test]
fn whisper_defaults_match_swift_fallback_ids() {
  let s = special();
  assert_eq!(s.start_of_transcript_token(), 50258);
  assert_eq!(s.end_token(), 50257);
  assert_eq!(s.special_token_begin(), 50257);
  assert_eq!(s.time_token_begin(), 50364);
  assert_eq!(s.no_timestamps_token(), 50363);
  assert_eq!(s.transcribe_token(), 50359);
  assert_eq!(s.whitespace_token(), 220);
}

#[test]
fn suppress_tokens_masks_exactly_the_listed_ids() {
  let mut logits = flat_logits();
  SuppressTokensFilter::new(vec![3, 5]).filter(&mut logits, &[]);
  assert_eq!(logits[3], NEG_INF);
  assert_eq!(logits[5], NEG_INF);
  assert_eq!(logits[4], 0.0);
}

#[test]
fn suppress_blank_fires_only_at_sample_begin() {
  let s = special();
  let filter = SuppressBlankFilter::new(&s, 2);
  let mut logits = flat_logits();
  filter.filter(&mut logits, &[50258]); // len 1 != 2 -> untouched
  assert_eq!(logits[s.whitespace_token() as usize], 0.0);
  filter.filter(&mut logits, &[50258, 50259]); // len == sample_begin
  assert_eq!(logits[s.whitespace_token() as usize], NEG_INF);
  assert_eq!(logits[s.end_token() as usize], NEG_INF);
}

// --- TimestampRulesFilter: ports LogitsFilter.swift:72-129 rule by rule ---

fn ts(s: &SpecialTokens, index: u32) -> u32 {
  s.time_token_begin() + index
}

#[test]
fn timestamp_rules_no_ops_while_prefilling_multilingual_prompt() {
  // Multilingual sampleBegin: needs a task token in the first 3 tokens
  // (LogitsFilter.swift:131-142); a prompt without one -> early return.
  let s = special();
  let filter = TimestampRulesFilter::new(&s, 3, None, true);
  let mut logits = flat_logits();
  filter.filter(&mut logits, &[50258, 50259]); // no transcribe/translate token yet
  assert!(logits.iter().all(|&v| v == 0.0));
}

#[test]
fn timestamp_rules_always_suppress_no_timestamps_token() {
  let s = special();
  let filter = TimestampRulesFilter::new(&s, 3, None, true);
  let mut logits = flat_logits();
  let prompt = [50258, 50259, s.transcribe_token(), ts(&s, 0)];
  filter.filter(&mut logits, &prompt);
  assert_eq!(logits[s.no_timestamps_token() as usize], NEG_INF);
}

#[test]
fn paired_timestamp_rules_mask_directionally() {
  let s = special();
  let filter = TimestampRulesFilter::new(&s, 1, None, false);
  // Case A: last sampled was a timestamp, penultimate was text ->
  // "cannot be normal text tokens": [0, end_token) masked
  // (LogitsFilter.swift:92-95).
  let mut logits = flat_logits();
  filter.filter(&mut logits, &[50258, 100, ts(&s, 5)]);
  assert_eq!(logits[0], NEG_INF);
  assert_eq!(logits[(s.end_token() - 1) as usize], NEG_INF);
  // NOTE: the paired-timestamp rule alone excludes end_token from its
  // [0, end_token) mask (LogitsFilter.swift:92-95, verified above via
  // end_token - 1). But this flat buffer, after that mask, leaves only
  // 106 unmasked text logits against 1496 unmasked timestamp logits, so
  // the unconditional trailing sum-of-probability rule
  // (LogitsFilter.swift:124-127; independently covered by
  // `timestamp_sum_probability_forces_timestamp_sampling`) also fires —
  // logsumexp over far more equally-likely timestamp slots exceeds the
  // max over far fewer equally-likely text slots — and re-masks the
  // whole text range end to end, end_token included. The brief this test
  // was transcribed from asserted `0.0` ("EOT stays allowed") here,
  // reasoning about the paired rule in isolation; that is wrong for this
  // input under a faithful port of `LogitsFilter.swift`, Swift's BNNS
  // logSoftmax/logSumExp/max included — corrected to match actual
  // end-to-end Swift behavior.
  assert_eq!(logits[s.end_token() as usize], NEG_INF);
  // and timestamps below the last one are forbidden WITHOUT the +1 bump
  // (LogitsFilter.swift:102-108: lastWasTimestamp && !penultimate).
  assert_eq!(logits[ts(&s, 4) as usize], NEG_INF);
  assert_eq!(logits[ts(&s, 5) as usize], 0.0);

  // Case B: last two sampled were both timestamps -> "has to be
  // non-timestamp": [time_token_begin, vocab) masked (LogitsFilter.swift:88-91),
  // and timestamps < last+1 masked (the +1 branch).
  let mut logits = flat_logits();
  filter.filter(&mut logits, &[50258, ts(&s, 5), ts(&s, 5)]);
  assert_eq!(logits[ts(&s, 0) as usize], NEG_INF);
  assert_eq!(logits[VOCAB - 1], NEG_INF);
  assert_eq!(logits[100], 0.0); // text tokens allowed
}

#[test]
fn timestamp_sum_probability_forces_timestamp_sampling() {
  // If logsumexp(logprobs over timestamps) > max(text logprobs), text is
  // masked (LogitsFilter.swift:124-127 + 144-242, BNNS -> f32 math).
  let s = special();
  let filter = TimestampRulesFilter::new(&s, 1, None, false);
  let mut logits = vec![-10.0f32; VOCAB];
  // Many moderately-likely timestamps out-mass one strong text token.
  logits[100] = 2.0;
  for i in 0..1000 {
    logits[(s.time_token_begin() + i) as usize] = 1.8;
  }
  filter.filter(&mut logits, &[50258, 100]); // past sample_begin, last is text
  assert_eq!(
    logits[100], NEG_INF,
    "text masked when timestamp mass dominates"
  );
  assert_eq!(logits[(s.time_token_begin() + 999) as usize], 1.8);

  // And the converse: one dominant text token survives.
  let mut logits = vec![-10.0f32; VOCAB];
  logits[100] = 20.0;
  logits[ts(&s, 3) as usize] = 1.0;
  filter.filter(&mut logits, &[50258, 100]);
  assert_eq!(logits[100], 20.0);
}

#[test]
fn language_filter_keeps_only_language_tokens_after_sample_begin() {
  let language_tokens = [50259u32, 50260, 50261];
  let filter = LanguageLogitsFilter::new(&language_tokens, 1);
  let mut logits = flat_logits();
  filter.filter(&mut logits, &[]); // before sample_begin -> untouched
  assert_eq!(logits[0], 0.0);
  filter.filter(&mut logits, &[50258]);
  assert_eq!(logits[0], NEG_INF);
  assert_eq!(logits[50259], 0.0);
  assert_eq!(logits[50260], 0.0);
  assert_eq!(logits[51864], NEG_INF);
}
