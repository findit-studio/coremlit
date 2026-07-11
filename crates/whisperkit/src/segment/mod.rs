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
//! [`find_alignment`] (`:340-408`), [`merge_punctuations`] (`:280-338`),
//! [`calculate_word_duration_constraints`] and
//! [`truncate_long_words_at_sentence_boundaries`] (`:498-526`),
//! [`update_segments_with_word_timings`] (`:528-659`) — the final
//! re-anchoring step that walks a word-index cursor shared across every
//! segment, applies the short-word pull-back and pause/boundary
//! heuristics, and writes the results back onto [`TranscriptionSegment`]s
//! — and [`add_word_timestamps`] (`:410-496`), the orchestration wrapper
//! that flattens each segment's tokens/log-probs into the flat index list
//! used to filter the raw CoreML alignment-weights array, then threads
//! that through [`find_alignment`] -> the duration/truncation hack ->
//! [`merge_punctuations`] -> [`update_segments_with_word_timings`] in
//! sequence. Wiring `add_word_timestamps` into the decode loop itself
//! (`TranscribeTask.swift:196-233`) is deferred to a later task; this
//! module ships the orchestration and every pure-math function it
//! depends on.

use unicode_categories::UnicodeCategories;

use crate::{
  backend::{AlignmentMatrix, AlignmentView},
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
/// value` up front and compares THOSE (`SegmentSeeker.swift:239-251`) —
/// and this port does the same, because adding a common finite value is
/// not order-preserving in floating point: a large-magnitude `value` can
/// round distinct incoming costs into exact ties, and exact ties fall to
/// left. Comparing the bare incoming costs picked a different winner on
/// such inputs (phase-gate finding; pinned by
/// `dtw_add_before_compare_matches_swift_rounding_ties`).
fn min_cost_and_trace(diagonal: f64, up: f64, left: f64, value: f64) -> (f64, i8) {
  let c0 = diagonal + value;
  let c1 = up + value;
  let c2 = left + value;
  if c0 < c1 && c0 < c2 {
    (c0, 0)
  } else if c1 < c0 && c1 < c2 {
    (c1, 1)
  } else {
    (c2, 2)
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
      let (best, direction) = min_cost_and_trace(diagonal, up, left, value);
      cost[row * width + column] = best;
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

// ---------------------------------------------------------------------
// Word-duration constraints and sentence-boundary truncation
// ---------------------------------------------------------------------

/// The capped-median/max word-duration pair [`calculate_word_duration_constraints`]
/// computes over one window's word alignment — Swift's anonymous
/// `(median: Float, max: Float)` tuple return (`SegmentSeeker.swift:
/// 498-507`). `Copy`, and deliberately has no constructor: unlike an
/// options type, whose fields a caller assembles piecewise before the
/// fact, both fields here only ever come out of that one function
/// together, and `max` is always exactly `median * 2` — a public
/// constructor would let a caller build a pair that violates that
/// invariant. This type is a computation RESULT, not configuration; that
/// is a deliberate departure from the options pattern, not an oversight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordDurationConstraints {
  median: f32,
  max: f32,
}

impl WordDurationConstraints {
  /// The capped median word duration: `min(0.7, raw median)`, in seconds,
  /// or `0.0` when the source alignment had no positive-duration word
  /// (`SegmentSeeker.swift:502-503`).
  #[inline(always)]
  pub const fn median(&self) -> f32 {
    self.median
  }

  /// The overlong-word threshold: always exactly twice [`Self::median`]
  /// (`SegmentSeeker.swift:504`), consumed by
  /// [`truncate_long_words_at_sentence_boundaries`].
  #[inline(always)]
  pub const fn max_duration(&self) -> f32 {
    self.max
  }
}

/// Computes the capped-median/max word-duration pair used to flag overlong
/// words. Ports `calculateWordDurationConstraints`
/// (`SegmentSeeker.swift:498-507`): every word whose `duration` is not
/// strictly positive is dropped before the median is taken (`:499-500`,
/// `filter { $0 > 0 }`); the median is the sorted list's UPPER middle
/// element — `sorted[count / 2]`, integer division, NOT an average of the
/// two middle values on an even count (`:502`); that raw median is then
/// capped at `0.7` s (`:503`), and `max` is always twice the CAPPED
/// value, never the raw one (`:504`). An empty `alignment`, or one where
/// every word has zero or negative duration, yields `median = max = 0.0`.
pub fn calculate_word_duration_constraints(alignment: &[WordTiming]) -> WordDurationConstraints {
  let mut durations: Vec<f32> = alignment
    .iter()
    .map(WordTiming::duration)
    .filter(|&duration| duration > 0.0)
    .collect();
  durations.sort_by(f32::total_cmp);

  let raw_median = durations.get(durations.len() / 2).copied().unwrap_or(0.0);
  let median = raw_median.min(0.7);
  let max = median * 2.0;

  WordDurationConstraints { median, max }
}

/// Sentence-ending marks [`truncate_long_words_at_sentence_boundaries`]
/// matches a word's text against EXACTLY — no substring or trimmed match
/// (Swift `sentenceEndMarks`, `SegmentSeeker.swift:510`; includes the CJK
/// full-width punctuation forms alongside the ASCII ones).
const SENTENCE_END_MARKS: [&str; 6] = [".", "。", "!", "！", "?", "？"];

/// Clips words whose duration exceeds `max_duration` back down to it, but
/// only where a sentence boundary justifies the clip — guards against a
/// single misaligned DTW timestamp stretching one word far past its real
/// span. Ports `truncateLongWordsAtSentenceBoundaries`
/// (`SegmentSeeker.swift:509-526`) structure-preserving: index `0` is
/// never inspected or modified (the loop runs `1..alignment.len()`,
/// `:514`; Rust's half-open `Range` is simply empty rather than panicking
/// when `alignment` itself is empty, so no separate emptiness guard is
/// needed the way Swift's `1..<0` `ClosedRange` would require). For each
/// later word whose `duration()` exceeds `max_duration`:
/// - if the word's own text EXACTLY matches a mark in the internal
///   `SENTENCE_END_MARKS` table (whole-word — `" ."` with a leading space
///   does not qualify), its `end` is pulled back to `start + max_duration`;
/// - otherwise, if the PRECEDING word's text exactly matches a mark, this
///   word's `start` is pushed forward to `end - max_duration`.
///
/// These are `if`/`else if` branches (`:516-520`), so a word that is
/// itself a mark AND immediately follows another mark only ever takes the
/// first branch: its `end` is adjusted, its `start` never is.
pub fn truncate_long_words_at_sentence_boundaries(
  mut alignment: Vec<WordTiming>,
  max_duration: f32,
) -> Vec<WordTiming> {
  for i in 1..alignment.len() {
    if alignment[i].duration() > max_duration {
      if SENTENCE_END_MARKS.contains(&alignment[i].word()) {
        let start = alignment[i].start();
        alignment[i].set_end(start + max_duration);
      } else if SENTENCE_END_MARKS.contains(&alignment[i - 1].word()) {
        let end = alignment[i].end();
        alignment[i].set_start(end - max_duration);
      }
    }
  }
  alignment
}

/// Re-anchors DTW word-level `merged_alignment` timings onto `segments`,
/// applying Swift's short-word pull-back and pause/boundary heuristics
/// along the way. Ports `updateSegmentsWithWordTimings`
/// (`SegmentSeeker.swift:528-659`) — the final step `addWordTimestamps`
/// (`:410-496`, not yet ported; see this module's own doc) runs after
/// `findAlignment` -> the duration/truncation hack -> `mergePunctuations`.
///
/// `merged_alignment` is walked with a cursor SHARED across every
/// `segments` entry (`:538`, Swift's `wordIndex`): once an alignment entry
/// is consumed by one segment it is never revisited by a later one, even
/// if that segment's own token budget is not fully accounted for (the
/// cursor simply runs out and that segment's word list ends up short).
/// `last_speech_timestamp` is similarly threaded across every segment,
/// seeded by the caller's initial value and updated to each segment's own
/// final `end` once it gets at least one word (`:651`, guarded by the same
/// non-empty check as the pause/boundary hack itself — a wordless segment
/// leaves `last_speech_timestamp` untouched for the next one).
///
/// # Errors
/// [`SegmentError::Tokenizer`] if retokenizing a partially-special-filtered
/// alignment entry's surviving tokens fails (`:556-559`).
pub fn update_segments_with_word_timings(
  segments: &[TranscriptionSegment],
  merged_alignment: &[WordTiming],
  seek: usize,
  last_speech_timestamp: f32,
  constrained_median_duration: f32,
  max_duration: f32,
  tokenizer: &WhisperTokenizer,
) -> Result<Vec<TranscriptionSegment>, SegmentError> {
  // :537 -- this window's seek offset, in seconds.
  let time_offset = seek as f32 / SAMPLE_RATE as f32;
  let special_begin = tokenizer.special_tokens().special_token_begin();
  // :538 -- cursor into `merged_alignment`, shared across every segment
  // below; never reset per segment.
  let mut word_index = 0usize;
  let mut last_speech_timestamp = last_speech_timestamp;
  let mut updated_segments: Vec<TranscriptionSegment> = Vec::with_capacity(segments.len());

  for (segment_index, segment) in segments.iter().enumerate() {
    let mut saved_tokens = 0usize;
    // :544 -- only text tokens count toward this segment's word budget;
    // special/timestamp tokens already in `segment.tokens` never do.
    let text_token_count = segment
      .tokens_slice()
      .iter()
      .filter(|&&token| token < special_begin)
      .count();
    let mut words_in_segment: Vec<WordTiming> = Vec::new();

    // :547's `where savedTokens < textTokens.count` guards each element in
    // Swift's `for timing in mergedAlignment[wordIndex...]`, skipping
    // (not necessarily stopping at) elements while false. `break` here is
    // behaviorally identical: `saved_tokens` and `word_index` both only
    // ever advance inside this loop body, so once the bound trips false it
    // stays false for every later element too, and Swift's `where` never
    // lets the body run again either. `.min(merged_alignment.len())`
    // guards a slice a Swift `mergedAlignment[wordIndex...]` has no
    // equivalent for (it would trap on an out-of-range `wordIndex`); the
    // invariant that `word_index` never exceeds `merged_alignment.len()`
    // holds by construction (each increment consumes one element of the
    // shrinking remaining slice), so this is a zero-cost safety net, not a
    // behavior change.
    for timing in &merged_alignment[word_index.min(merged_alignment.len())..] {
      if saved_tokens >= text_token_count {
        break;
      }
      word_index += 1;

      // :551-554 -- drop special/timestamp tokens from this timing; an
      // all-special entry is consumed from the cursor but emits no word.
      let timing_tokens: Vec<u32> = timing
        .tokens_slice()
        .iter()
        .copied()
        .filter(|&token| token < special_begin)
        .collect();
      if timing_tokens.is_empty() {
        continue;
      }

      // :556-559 -- retokenize only when some (not all) of this timing's
      // tokens were filtered out; otherwise reuse its own decoded word.
      let timing_tokens_len = timing_tokens.len();
      let word = if timing_tokens_len < timing.tokens_slice().len() {
        tokenizer.decode(&timing_tokens, false)?
      } else {
        timing.word().to_string()
      };

      // :561-562.
      let mut start = rounded_to_places(time_offset + timing.start(), 2);
      let end = rounded_to_places(time_offset + timing.end(), 2);

      // :564-596 -- a short-duration word gets its start pulled back into
      // any gap before it: against the previous word in THIS segment if
      // there is one, else (only for a segment's own first word) against
      // the previous segment's already-finalized end.
      if end - start < constrained_median_duration / 4.0 {
        if let Some(previous) = words_in_segment.last() {
          let previous_end = previous.end();
          if start > previous_end {
            let space_available = start - previous_end;
            let desired_duration = space_available.min(constrained_median_duration / 2.0);
            start = rounded_to_places(start - desired_duration, 2);
          }
        } else if segment_index > 0
          && updated_segments.len() > segment_index - 1
          && start > updated_segments[segment_index - 1].end()
        {
          let previous_end = updated_segments[segment_index - 1].end();
          let space_available = start - previous_end;
          let desired_duration = space_available.min(constrained_median_duration / 2.0);
          start = rounded_to_places(start - desired_duration, 2);
        }
      }

      // :598.
      let probability = rounded_to_places(timing.probability(), 2);
      words_in_segment.push(WordTiming::new(
        word,
        timing_tokens,
        start,
        end,
        probability,
      ));
      // :606 -- Swift re-reads `timingTokens.count`, the local filtered
      // vec, not the just-pushed word's own token slice; captured above
      // before `timing_tokens` moved into the `WordTiming`.
      saved_tokens += timing_tokens_len;
    }

    let mut updated_segment = segment.clone();

    // :615-652 -- only a segment that got at least one word runs the
    // pause/boundary hack and advances `last_speech_timestamp`; a wordless
    // segment leaves both `updated_segment`'s bounds and
    // `last_speech_timestamp` untouched.
    if !words_in_segment.is_empty() {
      // :616-620 -- read BEFORE any mutation below, matching Swift's own
      // `firstWord` copy.
      let pause_length = words_in_segment[0].end() - last_speech_timestamp;
      let first_word_too_long = words_in_segment[0].duration() > max_duration;
      let both_words_too_long = words_in_segment.len() > 1
        && words_in_segment[1].end() - words_in_segment[0].start() > max_duration * 2.0;

      // :621-633 -- after an over-long pause, clamp the first word (and,
      // if it is also too long, re-split the 0/1 boundary first) so
      // neither word spans more than `max_duration`.
      if pause_length > constrained_median_duration * 4.0
        && (first_word_too_long || both_words_too_long)
      {
        if words_in_segment.len() > 1 && words_in_segment[1].duration() > max_duration {
          let w1_end = words_in_segment[1].end();
          let boundary = (w1_end / 2.0).max(w1_end - max_duration);
          words_in_segment[0].set_end(boundary);
          words_in_segment[1].set_start(boundary);
        }
        // Reads `words_in_segment[0].end()` LIVE: the boundary re-split
        // just above, if it fired, already changed it.
        let w0_end = words_in_segment[0].end();
        words_in_segment[0].set_start(last_speech_timestamp.max(w0_end - max_duration));
      }

      // :635-640 -- prefer the segment-level start over the (possibly
      // hack-adjusted) first word's start when the word has drifted more
      // than half a second earlier than the segment itself began.
      let w0_start = words_in_segment[0].start();
      let w0_end = words_in_segment[0].end();
      if segment.start() < w0_end && segment.start() - 0.5 > w0_start {
        let clamped = (w0_end - constrained_median_duration)
          .min(segment.start())
          .max(0.0);
        words_in_segment[0].set_start(clamped);
      } else {
        updated_segment.set_start(words_in_segment[0].start());
      }

      // :642-649 -- symmetric preference for the segment-level end over
      // the last word's end. Swift's `wordsInSegment.last` is always
      // non-nil here (guarded by the outer non-empty check already); when
      // there is exactly one word this is the SAME element the
      // start-preference block above just wrote, so `last_start` below
      // can already reflect that mutation.
      let last_index = words_in_segment.len() - 1;
      let last_start = words_in_segment[last_index].start();
      let last_end = words_in_segment[last_index].end();
      if updated_segment.end() > last_start && segment.end() + 0.5 < last_end {
        let clamped = (last_start + constrained_median_duration).max(segment.end());
        words_in_segment[last_index].set_end(clamped);
      } else {
        updated_segment.set_end(last_end);
      }

      // :651.
      last_speech_timestamp = updated_segment.end();
    }

    // :654-655.
    updated_segment.set_words(words_in_segment);
    updated_segments.push(updated_segment);
  }

  Ok(updated_segments)
}

/// Rounds `value` to `decimal_places` decimal digits, half-away-from-zero.
/// Ports `Float.rounded(_:)` (`ArgmaxCore/FoundationExtensions.swift:
/// 9-13`: `(self * divisor).rounded() / divisor`, where Swift's
/// no-argument `.rounded()` defaults to rule `.toNearestOrAwayFromZero`).
/// Rust's [`f32::round`] documents that exact rule (round half-way cases
/// away from `0.0`), so this is a direct, unadjusted port. `pub(crate)`:
/// no caller outside this crate needs it yet — [`update_segments_with_word_timings`]
/// is the first non-test consumer.
pub(crate) fn rounded_to_places(value: f32, decimal_places: i32) -> f32 {
  let divisor = 10f32.powi(decimal_places);
  (value * divisor).round() / divisor
}

// ---------------------------------------------------------------------
// add_word_timestamps: the orchestration wrapper
// ---------------------------------------------------------------------

/// Assembles one window's word-level timestamps end to end. Ports
/// `SegmentSeeker.addWordTimestamps` (`SegmentSeeker.swift:410-496`):
/// flattens `segments`' tokens/log-probs into the flat list
/// [`find_alignment`] needs (`:427-442`), builds a prefix-take,
/// zero-padded [`AlignmentMatrix`] from `alignment` (`:444-461`), then
/// threads that through [`find_alignment`] (`:465-472`) -> the
/// duration-constraint/sentence-boundary truncation hack (`:474-477`) ->
/// [`merge_punctuations`] when non-empty (`:479-482`) ->
/// [`update_segments_with_word_timings`] (`:484-493`).
///
/// `language_code` is threaded straight into `find_alignment` ->
/// `split_to_word_tokens`, the same `NLLanguageRecognizer` replacement
/// documented on [`find_alignment`] and
/// [`WhisperTokenizer::split_to_word_tokens`] (spec §5.3). Swift's
/// `segmentSize`, `options`, and `timings` parameters are unused in the
/// function body (verified against `SegmentSeeker.swift:410-496`) and are
/// dropped here: `timings`' duration/run-count bookkeeping
/// (`TranscribeTask.swift:214-215`) is the caller's responsibility, same
/// as at Swift's own call site.
///
/// # Errors
/// [`SegmentError::InvalidAlignmentShape`] if the prefix-take alignment
/// ends up with zero rows or columns — notably an empty (or
/// all-empty-tokens) `segments` input, which Swift's own unguarded
/// `1...0` range would instead crash on (see [`dynamic_time_warping`]'s
/// doc); [`SegmentError::Tokenizer`] if `split_to_word_tokens` or a
/// partial-special retokenize fails.
#[allow(clippy::too_many_arguments)] // Mirrors Swift's addWordTimestamps argument
// surface (mirroring decode_text's own precedent for this exact lint, per its
// doc comment); no natural subset of these forms a cohesive struct without
// inventing one purely to dodge the lint.
pub fn add_word_timestamps(
  segments: &[TranscriptionSegment],
  alignment: &AlignmentView<'_>,
  tokenizer: &WhisperTokenizer,
  language_code: &str,
  seek: usize,
  prepended: &str,
  appended: &str,
  last_speech_timestamp: f32,
) -> Result<Vec<TranscriptionSegment>, SegmentError> {
  // :427-442 -- flatten every segment's tokens, in order; pair each with
  // its logged log-prob only when Swift's dictionary probe would have
  // found one (`segment.tokenLogProbs[index][token] != nil`): this
  // position's logged token id equals the token actually being gathered.
  // `.get` (rather than a direct index) additionally tolerates a shorter
  // `token_log_probs_slice` -- every `TranscriptionSegment` this crate
  // constructs keeps the two parallel, so that is a defensive no-op here,
  // not an intentional behavior difference from Swift's dictionary array.
  let mut word_token_ids: Vec<u32> = Vec::new();
  let mut filtered_log_probs: Vec<f32> = Vec::new();
  for segment in segments {
    let log_probs = segment.token_log_probs_slice();
    for (index, &token) in segment.tokens_slice().iter().enumerate() {
      word_token_ids.push(token);
      if let Some(&(logged_token, log_prob)) = log_probs.get(index)
        && logged_token == token
      {
        filtered_log_probs.push(log_prob);
      }
    }
  }

  // :444-461 -- Swift's `filteredIndices` are consecutive `0..N` by
  // construction (`:432` unconditionally appends `index + indexOffset`,
  // and `:441` advances `indexOffset` by exactly the previous segment's
  // token count), so "filtering" the alignment weights collapses to a
  // prefix take over rows `0..word_token_ids.len()`. Swift's destination
  // `MLMultiArray` is zero-initialized (`initialValue: FloatType(0)`,
  // `:450`) and only rows `0..alignmentWeights.rows` are ever memcpy'd in
  // (`:454-459`), so rows beyond that read back as zero there too -- this
  // mirrors that exactly, rather than requiring
  // `word_token_ids.len() <= alignment.rows()`.
  let needed = word_token_ids.len();
  let cols = alignment.cols();
  let mut data = vec![0.0f32; needed * cols];
  for (row_index, row) in data
    .chunks_mut(cols)
    .enumerate()
    .take(alignment.rows().min(needed))
  {
    row.copy_from_slice(alignment.row(row_index));
  }
  let filtered = AlignmentMatrix::new(data, needed, cols);

  // :465-472. The construction above guarantees `filtered.rows() ==
  // word_token_ids.len()` always -- the invariant `find_alignment` needs
  // to index `start_times`/`end_times`/`token_log_probs` in lockstep with
  // `word_token_ids` -- regardless of how `alignment.rows()` compares to
  // `word_token_ids.len()`. When `word_token_ids` is empty this makes
  // `filtered.rows() == 0`; `dynamic_time_warping` checks that
  // unconditionally, before `find_alignment`'s own `<= 1 word` early
  // return (see that function's doc), so an empty `segments` input
  // surfaces `SegmentError::InvalidAlignmentShape` here rather than
  // degrading to word-less segments.
  let mut merged = find_alignment(
    &word_token_ids,
    &filtered.view(),
    &filtered_log_probs,
    tokenizer,
    language_code,
  )?;

  // :474-477 -- the upstream "hack" Swift's own comment flags (reference,
  // Swift's own citation at `:474-475`: openai/whisper
  // `whisper/timing.py#L305`, commit `ba3f3cd`): constrain the
  // median/max word duration, then truncate overlong words at sentence
  // boundaries, before merging punctuation.
  let word_durations = calculate_word_duration_constraints(&merged);
  merged = truncate_long_words_at_sentence_boundaries(merged, word_durations.max_duration());

  // :480-482 -- gated on the merged ALIGNMENT being non-empty, not on
  // `prepended`/`appended` (a correction to this task's brief: Swift's
  // `if !alignment.isEmpty` reads the alignment array, not the
  // punctuation-string parameters). `merge_punctuations` is already a
  // no-op on an empty slice (see its own doc), so this gate changes
  // nothing observable -- kept only to mirror Swift's exact shape.
  if !merged.is_empty() {
    merged = merge_punctuations(&merged, prepended, appended);
  }

  // :484-493.
  update_segments_with_word_timings(
    segments,
    &merged,
    seek,
    last_speech_timestamp,
    word_durations.median(),
    word_durations.max_duration(),
    tokenizer,
  )
}

#[cfg(test)]
mod tests;
