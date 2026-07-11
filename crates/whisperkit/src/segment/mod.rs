//! Seek-point and segment slicing: turns one decoded window's
//! [`DecodingResult`] into zero or more [`TranscriptionSegment`]s and the
//! sample offset the next window should start decoding from. Ports
//! `SegmentSeeker.findSeekPointAndSegments`
//! (`argmax-oss-swift/Sources/WhisperKit/Core/Text/SegmentSeeker.swift:41-189`).
//!
//! Swift threads `timeToken`/`specialToken`/`sampleRate` in as three
//! separate parameters (`SegmentSeeker.swift:47-49`); this port reads the
//! first two off `tokenizer.special_tokens()` (already a parameter here)
//! and the third off [`crate::constants::SAMPLE_RATE`], collapsing three
//! parameters into the one `tokenizer` this module already needs for
//! decoding slice text.

use crate::{
  constants::{SAMPLE_RATE, SECONDS_PER_TIME_TOKEN},
  error::SegmentError,
  options::DecodingOptions,
  result::{DecodingResult, TranscriptionSegment},
  tokenizer::WhisperTokenizer,
};

/// Turns `decoding` — the just-decoded window starting at `current_seek`
/// samples — into the next seek offset and, unless the window was silent,
/// the [`TranscriptionSegment`]s it contains. `all_segments_count` seeds
/// each new segment's [`TranscriptionSegment::id`] so ids stay unique
/// across every window a caller has already processed; `segment_size` is
/// the window's length in samples (normally
/// [`crate::constants::WINDOW_SAMPLES`], smaller for a final short
/// window).
///
/// Three phases, ported structure-preserving from `findSeekPointAndSegments`:
///
/// 1. **Silence skip** (`SegmentSeeker.swift:57-74`): if
///    `options.no_speech_threshold()` is set and `decoding.no_speech_prob()`
///    exceeds it, the whole window is dropped — seek advances by
///    `segment_size` and `None` is returned — *unless*
///    `options.logprob_threshold()` is also set and `decoding.avg_logprob()`
///    exceeds *that*, which overrides the skip (confident text beats a
///    high no-speech probability).
/// 2. **Consecutive-timestamp slicing** (`:79-148`): otherwise, adjacent
///    timestamp-token pairs in `decoding.tokens_slice()` mark segment
///    boundaries. A lone trailing timestamp (single-timestamp ending) or a
///    trailing run of plain tokens (no-timestamp ending) each contribute
///    one final boundary of their own. Each resulting slice becomes a
///    segment spanning its first-to-last timestamp token; seek advances to
///    the last timestamp found (or by `segment_size` on a no-timestamp
///    ending).
/// 3. **Lump fallback** (`:149-186`): if no consecutive timestamp pair
///    exists at all, the whole window becomes one segment, its end time
///    refined by the last timestamp token above `<|0.00|>` if any exists;
///    seek always advances by `segment_size` on this path.
///
/// # Errors
/// [`SegmentError::Tokenizer`] if decoding a slice's tokens back to text
/// fails.
pub fn find_seek_point_and_segments(
  decoding: &DecodingResult,
  options: &DecodingOptions,
  all_segments_count: usize,
  current_seek: usize,
  segment_size: usize,
  tokenizer: &WhisperTokenizer,
) -> Result<(usize, Option<Vec<TranscriptionSegment>>), SegmentError> {
  let special = tokenizer.special_tokens();
  let time_token = special.time_token_begin();
  let special_token_begin = special.special_token_begin();
  let mut seek = current_seek;
  let time_offset = current_seek as f32 / SAMPLE_RATE as f32;

  // :57-74 — silence skip: no-speech probability above threshold skips the
  // window entirely, unless overridden by high average confidence.
  if let Some(threshold) = options.no_speech_threshold() {
    let mut should_skip = decoding.no_speech_prob() > threshold;
    if let Some(logprob_threshold) = options.logprob_threshold()
      && decoding.avg_logprob() > logprob_threshold
    {
      should_skip = false;
    }
    if should_skip {
      return Ok((seek + segment_size, None));
    }
  }

  let current_tokens = decoding.tokens_slice();
  let current_log_probs = decoding.token_log_probs_slice();
  let is_timestamp_token: Vec<bool> = current_tokens.iter().map(|&t| t >= time_token).collect();

  // :84-86 — the ending shape decides whether/how a trailing boundary is
  // synthesized below. A slice with fewer than 3 tokens can match neither
  // pattern, exactly like Swift's `Array == [Bool]` on a short `suffix`.
  let single_timestamp_ending = matches!(is_timestamp_token.as_slice(), [.., false, true, false]);
  let no_timestamp_ending = matches!(is_timestamp_token.as_slice(), [.., false, false, false]);

  // :88-97 — end index of every consecutive timestamp-token pair.
  let mut slice_indexes: Vec<usize> = Vec::new();
  let mut previous_is_timestamp = false;
  for (index, &is_timestamp) in is_timestamp_token.iter().enumerate() {
    if previous_is_timestamp && is_timestamp {
      slice_indexes.push(index);
    }
    previous_is_timestamp = is_timestamp;
  }

  let mut segments: Vec<TranscriptionSegment> = Vec::new();

  if slice_indexes.is_empty() {
    // :149-186 — no consecutive timestamps anywhere: lump the whole window
    // into one segment.
    let mut duration_seconds = segment_size as f32 / SAMPLE_RATE as f32;
    let timestamp_tokens: Vec<u32> = current_tokens
      .iter()
      .copied()
      .filter(|&t| t > time_token)
      .collect();
    if let Some(&last_timestamp) = timestamp_tokens.last() {
      duration_seconds = (last_timestamp - time_token) as f32 * SECONDS_PER_TIME_TOKEN;
    }

    let word_tokens: Vec<u32> = current_tokens
      .iter()
      .copied()
      .filter(|&t| t < special_token_begin)
      .collect();
    let segment_text_tokens: &[u32] = if options.skip_special_tokens() {
      &word_tokens
    } else {
      current_tokens
    };
    let segment_text = tokenizer.decode(segment_text_tokens, false)?;

    segments.push(
      TranscriptionSegment::new()
        .with_id(all_segments_count + segments.len())
        .with_seek(seek)
        .with_start(time_offset)
        .with_end(time_offset + duration_seconds)
        .with_text(segment_text)
        .with_tokens(current_tokens)
        .with_token_log_probs(current_log_probs)
        .with_temperature(decoding.temperature())
        .with_avg_logprob(decoding.avg_logprob())
        .with_compression_ratio(decoding.compression_ratio())
        .with_no_speech_prob(decoding.no_speech_prob()),
    );

    // Model gave no consecutive-timestamp boundary, so the whole window is
    // consumed regardless of the refined duration above (Swift's own
    // upstream TODO at `:184-185` notes the more accurate
    // `durationSeconds`-based advance is not yet used either).
    seek += segment_size;
  } else {
    // :101-107 — a lone trailing timestamp or trailing run of plain tokens
    // each need one more boundary appended beyond what the main loop above
    // found, to cover the window's tail as a final slice.
    if single_timestamp_ending {
      let single_ending_index = is_timestamp_token
        .iter()
        .rposition(|&t| t)
        .expect("single_timestamp_ending's pattern requires a `true` entry");
      slice_indexes.push(single_ending_index + 1);
    } else if no_timestamp_ending {
      slice_indexes.push(current_tokens.len());
    }

    let mut last_slice_start = 0usize;
    for &current_slice_end in &slice_indexes {
      let sliced_tokens = &current_tokens[last_slice_start..current_slice_end];
      let sliced_log_probs = &current_log_probs[last_slice_start..current_slice_end];

      // Every slice here is bounded by a detected timestamp-pair boundary
      // (this loop's own start) or ends at one (the main loop above only
      // ever records the second index of a `true, true` pair), so it
      // always contains at least one timestamp token — the same invariant
      // Swift trusts with `timestampTokens.first!`/`.last!`.
      let timestamp_tokens: Vec<u32> = sliced_tokens
        .iter()
        .copied()
        .filter(|&t| t >= time_token)
        .collect();
      let start_ts = *timestamp_tokens
        .first()
        .expect("slice bounded by a timestamp pair contains a timestamp token");
      let end_ts = *timestamp_tokens
        .last()
        .expect("slice bounded by a timestamp pair contains a timestamp token");
      let start_seconds = (start_ts - time_token) as f32 * SECONDS_PER_TIME_TOKEN;
      let end_seconds = (end_ts - time_token) as f32 * SECONDS_PER_TIME_TOKEN;

      let word_tokens: Vec<u32> = sliced_tokens
        .iter()
        .copied()
        .filter(|&t| t < special_token_begin)
        .collect();
      let sliced_text_tokens: &[u32] = if options.skip_special_tokens() {
        &word_tokens
      } else {
        sliced_tokens
      };
      let slice_text = tokenizer.decode(sliced_text_tokens, false)?;

      segments.push(
        TranscriptionSegment::new()
          .with_id(all_segments_count + segments.len())
          .with_seek(seek)
          .with_start(time_offset + start_seconds)
          .with_end(time_offset + end_seconds)
          .with_text(slice_text)
          .with_tokens(sliced_tokens)
          .with_token_log_probs(sliced_log_probs)
          .with_temperature(decoding.temperature())
          .with_avg_logprob(decoding.avg_logprob())
          .with_compression_ratio(decoding.compression_ratio())
          .with_no_speech_prob(decoding.no_speech_prob()),
      );

      last_slice_start = current_slice_end;
    }

    // :140-148 — seek to the last timestamp found, unless the tail was an
    // unbounded run of plain tokens (no-timestamp ending), which instead
    // consumes the full window like the lump branch does.
    if no_timestamp_ending {
      seek += segment_size;
    } else {
      let last_index = last_slice_start - usize::from(single_timestamp_ending);
      let last_timestamp_token = current_tokens[last_index] - time_token;
      let last_timestamp_seconds = last_timestamp_token as f32 * SECONDS_PER_TIME_TOKEN;
      seek += (last_timestamp_seconds * SAMPLE_RATE as f32) as usize;
    }
  }

  Ok((seek, Some(segments)))
}

#[cfg(test)]
mod tests;
