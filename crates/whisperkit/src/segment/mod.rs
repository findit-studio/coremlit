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
//!
//! Also home to the word-timestamp core: [`dynamic_time_warping`] (ports
//! `dynamicTimeWarping(withMatrix:)`, `SegmentSeeker.swift:195-278`,
//! tie-breaking in the private `min_cost_and_trace`, `:239-251`),
//! [`find_alignment`] (`:340-408`), and [`merge_punctuations`]
//! (`:280-338`). `addWordTimestamps` (`:410-`) — the orchestration wrapper
//! that re-anchors these against a window's seek offset, clamps against
//! `lastSpeechTimestamp`, and writes results back onto
//! [`TranscriptionSegment`]s — is deferred to a later task; this module
//! ships only the pure alignment math it depends on.

use unicode_categories::UnicodeCategories;

use crate::{
  backend::AlignmentView,
  constants::{SAMPLE_RATE, SECONDS_PER_TIME_TOKEN},
  error::SegmentError,
  options::DecodingOptions,
  result::{DecodingResult, TranscriptionSegment, WordTiming},
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

// ---------------------------------------------------------------------
// Word timestamps: DTW alignment, merge_punctuations, find_alignment
// ---------------------------------------------------------------------

/// One decoded-token/audio-frame alignment path out of
/// [`dynamic_time_warping`]'s cost-matrix backtrace: parallel
/// `text_indices`/`time_indices` sequences of equal length, walking from
/// the matrix's first aligned position to `(rows - 1, cols - 1)`. `isize`
/// mirrors the `-1` entries Swift's border-walk code shape permits
/// (`SegmentSeeker.swift:260-262`); in practice both cursors reach `0`
/// together via the unique `(1, 1) -> (0, 0)` step, so valid inputs never
/// produce one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtwPath {
  text_indices: Vec<isize>,
  time_indices: Vec<isize>,
}

impl DtwPath {
  /// Per-step decoded-token-row index (parallel to
  /// [`Self::time_indices_slice`]).
  #[inline(always)]
  pub fn text_indices_slice(&self) -> &[isize] {
    self.text_indices.as_slice()
  }

  /// Per-step audio-frame-column index (parallel to
  /// [`Self::text_indices_slice`]).
  #[inline(always)]
  pub fn time_indices_slice(&self) -> &[isize] {
    self.time_indices.as_slice()
  }
}

/// The three-way step cost and its winning direction for one DTW cell —
/// `0` diagonal, `1` up, `2` left. Ports `minCostAndTrace`
/// (`SegmentSeeker.swift:239-251`) exactly, including its tie-break order:
/// diagonal wins only by being strictly less than BOTH alternatives, then
/// up wins only by being strictly less than BOTH alternatives, and
/// everything else — including every exact tie — falls to the final
/// `else` and picks left. This `if`/`else if`/`else` shape (not a
/// three-way `min`) is what makes left the tie winner; it must not be
/// reordered or loosened to `<=`.
///
/// Swift computes `c0 = diagonal + value`, `c1 = up + value`, `c2 = left +
/// value` up front and compares those. This compares the bare incoming
/// costs instead and adds `value` to the winner afterward — bit-identical,
/// since `value` is finite and constant across all three, so it cannot
/// flip any `<` result (including when an operand is `f64::INFINITY`:
/// `INFINITY + value == INFINITY`, so the comparison's truth value is
/// unchanged either way).
fn min_cost_and_trace(diagonal: f64, up: f64, left: f64) -> (f64, i8) {
  if diagonal < up && diagonal < left {
    (diagonal, 0)
  } else if up < diagonal && up < left {
    (up, 1)
  } else {
    (left, 2)
  }
}

/// Dynamic time warping over a decoded-token x audio-frame cross-attention
/// alignment matrix. Builds a `(rows + 1) x (cols + 1)` cost/trace
/// matrix — Swift's nested `[[Double]]`/`[[Int]]`, flattened here into
/// row-major `Vec<f64>`/`Vec<i8>` indexed `row * (cols + 1) + col` for the
/// same values without per-row allocation — then backtraces it into a
/// [`DtwPath`]. Ports `dynamicTimeWarping(withMatrix:)`
/// (`SegmentSeeker.swift:195-237`, backtrace `:253-278`); cell
/// tie-breaking is the private `min_cost_and_trace` (`:239-251`).
///
/// # Errors
/// [`SegmentError::InvalidAlignmentShape`] if `matrix` has zero rows or
/// columns. Swift has no equivalent guard: `1...numberOfColumns`/
/// `1...numberOfRows` over a zero dimension is an invalid `ClosedRange`
/// and traps at runtime. This is a deliberate improvement over that
/// crash — a typed, recoverable error on the same malformed input.
pub fn dynamic_time_warping(matrix: &AlignmentView<'_>) -> Result<DtwPath, SegmentError> {
  let (rows, cols) = (matrix.rows(), matrix.cols());
  if rows == 0 || cols == 0 {
    return Err(SegmentError::InvalidAlignmentShape {
      rows,
      cols,
      len: matrix.data().len(),
    });
  }

  let width = cols + 1;
  let mut cost = vec![f64::INFINITY; (rows + 1) * width];
  let mut trace = vec![-1i8; (rows + 1) * width];
  cost[0] = 0.0;
  for cell in &mut trace[1..=cols] {
    *cell = 2; // :208-210 -- top border backtraces LEFT.
  }
  for i in 1..=rows {
    trace[i * width] = 1; // :211-213 -- left border backtraces UP.
  }

  for row in 1..=rows {
    for column in 1..=cols {
      // :217 -- MLMultiArray's flat linear index; equivalent to this
      // AlignmentView's row-major `row(row - 1)[column - 1]`.
      let value = -f64::from(matrix.row(row - 1)[column - 1]);
      let diagonal = cost[(row - 1) * width + column - 1];
      let up = cost[(row - 1) * width + column];
      let left = cost[row * width + column - 1];
      let (best, direction) = min_cost_and_trace(diagonal, up, left);
      cost[row * width + column] = best + value;
      trace[row * width + column] = direction;
    }
  }

  // :253-278 -- backtrace from the bottom-right corner to the origin.
  let (mut i, mut j) = (rows, cols);
  let mut text_indices = Vec::new();
  let mut time_indices = Vec::new();
  while i > 0 || j > 0 {
    text_indices.push(i as isize - 1);
    time_indices.push(j as isize - 1);
    match trace[i * width + j] {
      0 => {
        i -= 1;
        j -= 1;
      }
      1 => i -= 1,
      2 => j -= 1,
      // Unreachable for any (i, j) this loop actually visits: every cell
      // but (0, 0) -- never read, since the loop condition stops there --
      // was written by the border init or the main loop above to 0/1/2.
      // Kept as a defensive exit; Swift's `default: break` only exits the
      // `switch` there, which would spin forever instead if this were
      // ever hit.
      _ => break,
    }
  }
  text_indices.reverse();
  time_indices.reverse();

  Ok(DtwPath {
    text_indices,
    time_indices,
  })
}

/// True where Swift's `String.contains(String)` is: substring search that
/// treats an empty needle as never found (`String`'s/`NSString`'s
/// `range(of: "")` is documented to return no match), unlike
/// `str::contains`, for which an empty pattern matches everywhere.
/// [`merge_punctuations`]'s punctuation-membership checks need Swift's
/// behavior to stay exact for a word that trims to nothing.
fn swift_contains(haystack: &str, needle: &str) -> bool {
  !needle.is_empty() && haystack.contains(needle)
}

/// Trims Swift's `.whitespaces` `CharacterSet` (Unicode general category
/// `Zs` plus U+0009 CHARACTER TABULATION; no newlines) off both ends of
/// `s`. Same predicate as `tokenizer::is_single_punctuation_scalar`'s trim
/// step, duplicated here because that helper is private to its module.
fn trim_swift_whitespaces(s: &str) -> &str {
  s.trim_matches(|c: char| c.is_separator_space() || c == '\u{0009}')
}

/// Merges leading/trailing punctuation-only words in `alignment` onto
/// their neighboring word, then drops the words that end up empty or are
/// themselves bare merged-away punctuation. Ports `mergePunctuations`
/// (`SegmentSeeker.swift:280-338`) in its exact two-pass shape: characters
/// in `prepended` glue onto the FOLLOWING word (`:291-315`), characters in
/// `appended` glue onto the PRECEDING word (`:322-333`), then a final
/// filter drops the leftovers (`:336`).
///
/// Both passes replicate a Swift quirk rather than smoothing it over: each
/// iteration reads its merge neighbor from the *original* pre-merge
/// slice — `alignment[i - 1]` in the prepend pass, `prependedAlignment[i -
/// 1]` in the append pass — never from the tail of the list actually being
/// built. A run of three or more consecutive punctuation-only words
/// therefore does not fully chain together in either this port or
/// upstream Swift: only the immediately preceding original word survives
/// a second merge. Whisper's fixed single-character punctuation
/// vocabularies make three consecutive punctuation-only *words*
/// essentially unreachable in practice, so this port keeps Swift's exact
/// indexing rather than silently changing the observable behavior.
pub fn merge_punctuations(
  alignment: &[WordTiming],
  prepended: &str,
  appended: &str,
) -> Vec<WordTiming> {
  if alignment.is_empty() {
    return Vec::new();
  }

  // :291-315 -- merge PREPEND punctuation onto the following word.
  let mut prepended_alignment: Vec<WordTiming> = Vec::new();
  if !swift_contains(prepended, trim_swift_whitespaces(alignment[0].word())) {
    prepended_alignment.push(alignment[0].clone());
  }
  for pair in alignment.windows(2) {
    let previous = &pair[0];
    let current = &pair[1];
    let previous_starts_with_whitespace = previous
      .word()
      .chars()
      .next()
      .is_some_and(|c| c.is_separator_space() || c == '\u{0009}');
    if previous_starts_with_whitespace
      && swift_contains(prepended, trim_swift_whitespaces(previous.word()))
    {
      let mut word = previous.word().to_string();
      word.push_str(current.word());
      let mut tokens = previous.tokens_slice().to_vec();
      tokens.extend_from_slice(current.tokens_slice());
      let merged = WordTiming::new(
        word,
        tokens,
        current.start(),
        current.end(),
        current.probability(),
      );
      if prepended_alignment.is_empty() {
        prepended_alignment.push(merged);
      } else {
        let last = prepended_alignment.len() - 1;
        prepended_alignment[last] = merged;
      }
    } else {
      prepended_alignment.push(current.clone());
    }
  }

  // :317-333 -- merge APPEND punctuation onto the preceding word.
  let mut appended_alignment: Vec<WordTiming> = Vec::new();
  if let Some(first) = prepended_alignment.first() {
    appended_alignment.push(first.clone());
  }
  for pair in prepended_alignment.windows(2) {
    let previous = &pair[0];
    let current = &pair[1];
    if !previous.word().ends_with(' ')
      && swift_contains(appended, trim_swift_whitespaces(current.word()))
    {
      let mut word = previous.word().to_string();
      word.push_str(current.word());
      let mut tokens = previous.tokens_slice().to_vec();
      tokens.extend_from_slice(current.tokens_slice());
      let merged = WordTiming::new(
        word,
        tokens,
        previous.start(),
        previous.end(),
        previous.probability(),
      );
      let last = appended_alignment.len() - 1;
      appended_alignment[last] = merged;
    } else {
      appended_alignment.push(current.clone());
    }
  }

  // :336 -- drop empties and bare merged-away punctuation words.
  appended_alignment
    .into_iter()
    .filter(|w| {
      !w.word().is_empty()
        && !swift_contains(appended, w.word())
        && !swift_contains(prepended, w.word())
    })
    .collect()
}

/// Word-level timestamps for one window's decoded tokens: runs
/// [`dynamic_time_warping`] over `alignment`, groups `word_token_ids` into
/// words via [`WhisperTokenizer::split_to_word_tokens`], and reads each
/// word's start/end time off the DTW path's per-token-row boundaries plus
/// its mean sampled-token log probability. Ports `findAlignment`
/// (`SegmentSeeker.swift:340-408`); `language_code` is threaded straight
/// into `split_to_word_tokens` in place of Swift's internal
/// `NLLanguageRecognizer` detection (spec §5.3; see
/// [`WhisperTokenizer::split_to_word_tokens`]'s own doc for the same
/// substitution there).
///
/// Returns an empty vec when `split_to_word_tokens` groups `word_token_ids`
/// into one word or fewer (`:351-353`) — DTW timing is meaningless for a
/// single undivided span. DTW itself still runs first regardless (matching
/// Swift's own unconditional call order), so a malformed `alignment`
/// still errors even on that trivial path.
///
/// # Errors
/// [`SegmentError::InvalidAlignmentShape`] if `alignment` has zero rows or
/// columns (from [`dynamic_time_warping`]); [`SegmentError::Tokenizer`] if
/// `split_to_word_tokens` fails to decode `word_token_ids`.
pub fn find_alignment(
  word_token_ids: &[u32],
  alignment: &AlignmentView<'_>,
  token_log_probs: &[f32],
  tokenizer: &WhisperTokenizer,
  language_code: &str,
) -> Result<Vec<WordTiming>, SegmentError> {
  let path = dynamic_time_warping(alignment)?;
  let text_indices = path.text_indices_slice();
  let time_indices = path.time_indices_slice();

  let word_tokens = tokenizer.split_to_word_tokens(word_token_ids, language_code)?;
  if word_tokens.len() <= 1 {
    return Ok(Vec::new());
  }

  // :356-371 -- per-decoded-token-row start/end times: one boundary each
  // time the DTW path's aligned row changes.
  let mut start_times: Vec<f32> = vec![0.0];
  let mut end_times: Vec<f32> = Vec::new();
  let mut current_text_index = text_indices.first().copied().unwrap_or(0);
  for (index, &text_index) in text_indices.iter().enumerate() {
    if text_index != current_text_index {
      current_text_index = text_index;
      let time = time_indices[index] as f32 * SECONDS_PER_TIME_TOKEN;
      start_times.push(time);
      end_times.push(time);
    }
  }
  end_times.push(time_indices.last().copied().unwrap_or(1500) as f32 * SECONDS_PER_TIME_TOKEN);

  // :373-405 -- walk word groups; each consumes `tokens.len()` rows of
  // start_times/end_times/token_log_probs.
  let mut word_timings: Vec<WordTiming> = Vec::with_capacity(word_tokens.len());
  let mut current_token_index = 0usize;
  for (word, tokens) in word_tokens {
    let start_index = current_token_index;
    let word_start_time = start_times[current_token_index];
    current_token_index += tokens.len() - 1;
    let word_end_time = end_times[current_token_index];
    current_token_index += 1;

    let probs = &token_log_probs[start_index..current_token_index];
    let mean_log_prob = probs.iter().sum::<f32>() / probs.len() as f32;

    word_timings.push(WordTiming::new(
      word,
      tokens,
      word_start_time,
      word_end_time,
      mean_log_prob.exp(),
    ));
  }

  Ok(word_timings)
}

#[cfg(test)]
mod tests;
