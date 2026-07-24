use half::f16;

use super::*;
use crate::audio::whisper::tokenizer::SpecialTokens;

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

// ---------------------------------------------------------------------
// f16 timestamp-mass rule parity (H1, coremlit issue #41)
//
// The mass comparison replicates BNNS's probed precision structure
// (tests/whisper_swift_probes/probe_massrule2.out, macOS 26.5/M1 Max): a
// stable f32 log-softmax rounded per-element to f16, a naive f32 sum-of-exp
// over the f16 timestamp logprobs rounded to f16, an exact max over the f16
// text logprobs, then compare the two f16 scalars. Every expected hex is a
// committed BNNS output; each test names its probe source.
// ---------------------------------------------------------------------

#[test]
fn mass_rule_scalars_match_bnns_pinned_vector() {
  // Probe DUMP vector V3 (probe_massrule2.out:46-48): 251 f16 values
  // generated purely from bit patterns (no transcendentals in generation),
  // time_begin = 100. BNNS produced (lse, max) = (0xb7ae, 0xc4f2); the
  // sequential-f32 + f16-RNE emulation reproduces both bit-for-bit.
  let mut v3 = vec![0.0f32; 251];
  for (i, slot) in v3.iter_mut().enumerate() {
    let mut bits = 0x2C00u16 + ((i * 37) % 0x1000) as u16;
    if i % 3 == 0 {
      bits |= 0x8000;
    }
    *slot = f16::from_bits(bits).to_f32();
  }
  let (ts, mx) = bnns_mass_rule_scalars(&v3, 100).expect("finite max");
  assert_eq!(ts.to_bits(), 0xb7ae, "timestamp logSumExp scalar (V3)");
  assert_eq!(mx.to_bits(), 0xc4f2, "text max scalar (V3)");
}

#[test]
fn mass_rule_flip_points_match_bnns_scan1() {
  // probe_massrule2.out near-margin scan1 (16-wide): text -6.0 x8 except
  // v[3], timestamps -2.25 x8, time_begin = 8. BNNS flips between 0xb17c and
  // 0xb17d. Red-first note: the OLD f32 rule fired at BOTH (its f32 mass
  // -0.1706 exceeds both f16 neighbors), so it over-fired at 0xb17c.
  let mk = |v3_bits: u16| -> Vec<f32> {
    let mut v = vec![-6.0f32; 16];
    for x in v[8..16].iter_mut() {
      *x = -2.25;
    }
    v[3] = f16::from_bits(v3_bits).to_f32();
    v
  };
  assert!(
    timestamp_mass_exceeds_text(&mk(0xb17d), 8),
    "0xb17d -> fires"
  );
  assert!(
    !timestamp_mass_exceeds_text(&mk(0xb17c), 8),
    "0xb17c -> does not fire"
  );
}

#[test]
fn mass_rule_flip_points_match_bnns_scan2() {
  // probe_massrule2.out scan2 (1541-wide): text -8.0 x40 except v[7],
  // timestamps -9.5 x1501, time_begin = 40. BNNS flips between 0xc05e and
  // 0xc05f. Red-first note: the OLD f32 rule fired at NEITHER (its f32 mass
  // -2.1862 is below both f16 neighbors), so it under-fired at 0xc05f.
  let mk = |v7_bits: u16| -> Vec<f32> {
    let mut v = vec![-8.0f32; 1541];
    for x in v[40..1541].iter_mut() {
      *x = -9.5;
    }
    v[7] = f16::from_bits(v7_bits).to_f32();
    v
  };
  assert!(
    timestamp_mass_exceeds_text(&mk(0xc05f), 40),
    "0xc05f -> fires"
  );
  assert!(
    !timestamp_mass_exceeds_text(&mk(0xc05e), 40),
    "0xc05e -> does not fire"
  );
}

#[test]
fn mass_rule_sub_f16_margin_no_longer_resolved_in_f32() {
  // THE #41 divergence channel, in one test. Two equal top text logits
  // (indices 0,1 = 0.0) and a single timestamp (index 8 = 0.0001) a hair
  // above them: the true f32 timestamp-vs-text margin is +1e-4, so the OLD
  // f32 rule FIRED, but both normalized comparands round to the same f16
  // (0xbc65 ~ -ln 3), so the new rule resolves them EQUAL and does not fire.
  // Margin 1e-4 < half an f16 ulp (2^-11 ~ 4.9e-4) at this magnitude -- the
  // exact sub-f16-margin the old rule resolved in f32 and Swift never could.
  let mut logits = vec![f32::NEG_INFINITY; 12];
  logits[0] = 0.0;
  logits[1] = 0.0;
  logits[8] = 0.0001;
  let (ts, mx) = bnns_mass_rule_scalars(&logits, 8).expect("finite max");
  assert_eq!(ts.to_bits(), mx.to_bits(), "comparands collapse to one f16");
  assert!(
    !timestamp_mass_exceeds_text(&logits, 8),
    "sub-f16-ulp positive margin no longer fires (the old f32 rule did)"
  );
}

#[test]
fn mass_rule_all_masked_never_fires() {
  // Every entry -inf: the None guard yields false (probed LSE(all -inf) = -inf).
  assert!(!timestamp_mass_exceeds_text(&[f32::NEG_INFINITY; 16], 8));
}

#[test]
fn mass_rule_masked_timestamp_region_never_fires() {
  // Finite text, all-(-inf) timestamps -> ts_sum = 0 -> ts = -inf -> false
  // (probed LSE(all -inf) = -inf).
  let mut v = vec![f32::NEG_INFINITY; 16];
  v[0] = 1.0;
  v[3] = 0.5;
  assert!(!timestamp_mass_exceeds_text(&v, 8));
}

#[test]
fn mass_rule_all_masked_text_fires() {
  // All-(-inf) text, finite timestamps -> the finite ts scalar exceeds mx.
  // (Probed BNNS quirk: .max(all -inf) = -65504, not -inf; boolean-immaterial
  // -- this port's mx is -inf here, and the finite ts exceeds both.)
  let mut v = vec![f32::NEG_INFINITY; 16];
  v[10] = 1.0;
  v[12] = 0.5;
  assert!(timestamp_mass_exceeds_text(&v, 8));
}

#[test]
fn mass_rule_nan_poisons_to_non_firing() {
  // NaN anywhere -> the comparison is poisoned to non-firing. Contract: BNNS
  // skips NaN lanes in its normalizer, this port poisons to false; both never
  // fire (the scalars differ), and a model emitting NaN logits is already
  // undefined upstream. Without the NaN this config (favored timestamps)
  // would fire; the NaN is what suppresses it.
  let mut v = vec![0.0f32; 16];
  for x in v[8..16].iter_mut() {
    *x = 1.0;
  }
  v[10] = f32::NAN;
  assert!(!timestamp_mass_exceeds_text(&v, 8));
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
