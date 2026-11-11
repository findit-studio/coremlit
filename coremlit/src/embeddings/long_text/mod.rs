//! Long-text windowing for the text doors: the options a long-text call takes,
//! the gate that refuses an oversized call before any tokenizer work, and the
//! chunker that cuts a text into the windows a door embeds, one prediction
//! each.
//!
//! One machinery, shared by the doors rather than ported to each. A door brings
//! two things of its own: its model's token ceiling, passed to every function
//! here, and a [`RangeMeasure`], its tokenizer's exact count of a byte range of
//! the text. Each door re-exports the public types at its own path —
//! [`LongTextOptions`] beside its embedder, the refusal payloads in its `error`
//! module — and converts this module's [`Error`] into its own `Error`, variant
//! for variant.

use windit::{
  TailPolicy, WinditError,
  plan::WindowOptions,
  split::{Chunk, ContentAware, MeasureText},
};

/// The default window the text doors share, in tokens: the window
/// [`LongTextOptions::new`] and an omitted `window_options` key carry.
///
/// A default geometry, not a ceiling: each door passes its own ceiling to the
/// functions here, and a door whose default must be its whole model window
/// asserts the equality at compile time.
pub(crate) const DEFAULT_WINDOW: usize = 512;

/// Options for a text door's `embed_long_with` and `embed_windows_with`
/// (rust-options-pattern) — both share this one type: windit's chunk geometry
/// ([`WindowOptions`]) plus the door's pre-tokenization input bound.
///
/// The geometry is reachable whole ([`Self::window_options`]) and, for the tail
/// policy, one field at a time ([`Self::tail_policy`]). Neither requires naming
/// `windit`: the door that re-exports this type re-exports [`WindowOptions`] and
/// [`TailPolicy`] beside it.
///
/// # Wire form
///
/// Serializable under the `serde` feature, which is what coremlit's `serde`
/// turning on `windit?/serde` buys — this type composes windit's
/// [`WindowOptions`], so it could not be derived while that crate carried no
/// impls. The geometry is windit's own document, nested under
/// `window_options`:
///
/// ```json
/// {"window_options":{"window":512,"hop":512,"tail":{"kind":"keep_with_coverage"},
///  "max_windows":null},"max_input_bytes":null}
/// ```
///
/// Both fields carry defaults, so a partial config fills the rest from
/// [`Self::new`] — `{}` deserializes to exactly it, the same convention the
/// doors' construction options and `WindowPlan`s follow.
///
/// UNKNOWN KEYS ARE REFUSED, at this level and (windit's own rule) inside
/// `window_options`. Defaulted fields and a tolerated stray key compose into a
/// silent hole: `{"max_input_byte":4096}` — the plural dropped — would
/// otherwise deserialize to `max_input_bytes: None`, and a door's
/// `embed_long_with` (or, sharing this same options type, its
/// `embed_windows_with`) would then run with NO size gate on its input, which
/// is the one bound a caller sets for untrusted text. The misspelling is a hard
/// error naming the key instead.
///
/// That refusal makes this type UNFLATTENABLE: serde's `deny_unknown_fields`
/// and `flatten` do not compose (a flattened field sees the outer struct's
/// other keys and rejects them), so a config type composing these options must
/// NEST them under a key of its own — `long_text = { … }` — not
/// `#[serde(flatten)]` them into itself.
///
/// # Binary formats
///
/// The document above is a human-readable one; this type ALSO round-trips
/// through a non-self-describing format (postcard and friends), under every
/// tail policy. That is windit's contract, not coremlit's — the nested geometry
/// is windit's own type — and it holds only from windit 0.5: 0.4 tagged
/// `TailPolicy` adjacently through the derive, which writes the tag as a struct
/// FIELD and reads it back through `deserialize_identifier`, so a format
/// carrying no field names refused EVERY variant and a `LongTextOptions`
/// serialized to bytes it could not read back. windit 0.5 asks for the adjacent
/// shape itself (a `deserialize_struct` visitor serving both `visit_map` and
/// `visit_seq`), which leaves the document above byte-identical while making the
/// compact form readable. `long_text_options_round_trip_through_postcard`
/// (granite's tests) pins it, variant by variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LongTextOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_window_options"))]
  window_options: WindowOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  max_input_bytes: Option<usize>,
}

/// The [`LongTextOptions::window_options`] serde default: the same geometry
/// [`LongTextOptions::new`] builds, so an omitted key and an omitted options
/// value agree.
#[cfg(feature = "serde")]
fn default_window_options() -> WindowOptions {
  WindowOptions::new(DEFAULT_WINDOW)
}

impl Default for LongTextOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl From<WindowOptions> for LongTextOptions {
  /// Geometry-only options: the given windit geometry, no input byte limit.
  fn from(window_options: WindowOptions) -> Self {
    Self {
      window_options,
      max_input_bytes: None,
    }
  }
}

impl LongTextOptions {
  /// Options matching a door's `embed_long` and `embed_windows` alike: the
  /// default 512-token geometry (`WindowOptions::new(512)`) and no input byte
  /// limit.
  pub const fn new() -> Self {
    Self {
      window_options: WindowOptions::new(DEFAULT_WINDOW),
      max_input_bytes: None,
    }
  }

  /// The windit chunk geometry (per-chunk token budget, overlap, window cap).
  #[inline]
  pub const fn window_options(&self) -> WindowOptions {
    self.window_options
  }

  /// Builder form of [`Self::set_window_options`].
  #[must_use]
  #[inline]
  pub const fn with_window_options(mut self, window_options: WindowOptions) -> Self {
    self.set_window_options(window_options);
    self
  }

  /// Sets [`Self::window_options`] in place.
  #[inline]
  pub const fn set_window_options(&mut self, window_options: WindowOptions) -> &mut Self {
    self.window_options = window_options;
    self
  }

  /// The [`TailPolicy`] carried by [`Self::window_options`] — what windit does
  /// with a final chunk that does not fill a whole window.
  ///
  /// **It moves a boundary on `embed_long` and `embed_windows` alike; it never
  /// drops text.** The two cases differ, and both are windit 0.5's:
  ///
  /// - [`TailPolicy::PadFull`] is a NAMED GAP in windit's `ContentAware`
  ///   chunker — there is nothing to pad a byte range with, so it keeps the tail
  ///   indistinguishably from [`TailPolicy::KeepWithCoverage`], the default.
  ///   Setting it changes nothing here.
  /// - [`TailPolicy::DropBelowMin`] IS honoured by `ContentAware` from windit
  ///   0.5 (0.4 read every other geometry field and skipped `tail`): windit
  ///   discards a final chunk whose measure is below the minimum unless it fills
  ///   a whole window. What reaches a caller — of `embed_long`, or of a window
  ///   from `embed_windows` — is not that drop, though: the
  ///   separator-reattachment repair behind both exists to cover every byte
  ///   windit leaves out, and the discarded tail is exactly such a gap. It
  ///   comes back as its own chunk, because a dropped tail is by construction
  ///   the content that would not fit its predecessor, so the repair can never
  ///   fuse it left. **Measured across a window × minimum grid: the chunk
  ///   COUNT is unchanged and coverage is unchanged; what moves is the last
  ///   boundary** —
  ///   the separator run between the final two chunks, absorbed rightwards into
  ///   the tail instead of leftwards into its predecessor (e.g. window 4, a
  ///   `\n\n` break: `[(0,8),(8,17),(17,20)]` becomes
  ///   `[(0,8),(8,15),(15,20)]`). The embedding of the last chunk therefore
  ///   changes; the number of CoreML predictions does not. For
  ///   `embed_windows`, that changed last chunk is a moved `byte_range` on the
  ///   last window — geometry a consumer who persists spans alongside an index
  ///   should expect to change under `DropBelowMin`.
  ///
  /// One shape has no chunks to move: when the whole input is a single chunk
  /// below the minimum, windit returns NO chunks at all (its own documented
  /// consequence). `embed_long` and `embed_windows` still embed it — the
  /// non-empty fallback emits the whole input as one chunk, the same escape
  /// whitespace-only text already took — so this option cannot make a non-empty
  /// text embed to nothing.
  #[inline]
  pub const fn tail_policy(&self) -> TailPolicy {
    *self.window_options.tail()
  }

  /// Builder form of [`Self::set_tail_policy`].
  #[must_use]
  #[inline]
  pub const fn with_tail_policy(mut self, tail_policy: TailPolicy) -> Self {
    self.set_tail_policy(tail_policy);
    self
  }

  /// Sets [`Self::tail_policy`] on the carried [`Self::window_options`] in
  /// place, leaving its window, hop and cap alone.
  ///
  /// `DropBelowMin` moves `embed_long`'s and `embed_windows`'s last chunk
  /// boundary (a moved `byte_range` on the last window), `PadFull` does
  /// nothing, and neither drops text — see [`Self::tail_policy`].
  #[inline]
  pub const fn set_tail_policy(&mut self, tail_policy: TailPolicy) -> &mut Self {
    self.window_options = self.window_options.with_tail(tail_policy);
    self
  }

  /// The maximum accepted input length in UTF-8 bytes, if any. `None` (the
  /// default) means unbounded. Enforced before any tokenizer or chunker work —
  /// the limit callers embedding UNTRUSTED text should set.
  #[inline]
  pub const fn max_input_bytes(&self) -> Option<usize> {
    self.max_input_bytes
  }

  /// Builder form of [`Self::set_max_input_bytes`].
  #[must_use]
  #[inline]
  pub const fn with_max_input_bytes(mut self, max_input_bytes: usize) -> Self {
    self.set_max_input_bytes(max_input_bytes);
    self
  }

  /// Sets [`Self::max_input_bytes`] in place (to `Some(max_input_bytes)`).
  #[inline]
  pub const fn set_max_input_bytes(&mut self, max_input_bytes: usize) -> &mut Self {
    self.max_input_bytes = Some(max_input_bytes);
    self
  }
}

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a major bump.**
///
/// `key=value` pairs in declaration order, joined by `,`: [`Self::window_options`]
/// nests [`WindowOptions`]'s own `Display` in parentheses — composed verbatim,
/// not re-derived, so windit 0.5.1 is the one place that nested spelling can
/// change and a drift there fails the same pinning test this one does — and
/// [`Self::max_input_bytes`] prints its `usize` when set or `none` when absent,
/// the same identity fold windit's own `max_windows` uses: `Option` folds to
/// `none`, never to a gap.
///
/// For example, `LongTextOptions::new()` prints
/// `window_options=(window=512,hop=512,tail=keep_with_coverage,max_windows=none),max_input_bytes=none`.
impl core::fmt::Display for LongTextOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "window_options=({}),max_input_bytes=",
      self.window_options
    )?;
    match self.max_input_bytes {
      Some(max_input_bytes) => write!(f, "{max_input_bytes}"),
      None => f.write_str("none"),
    }
  }
}

/// A door's `embed_long_with` was configured with a per-chunk token budget
/// above the model's fixed input window (the door's token ceiling), so every
/// chunk would be silently truncated by the tokenizer. Rejected before any
/// chunking or prediction runs.
///
/// Payload of each text door's `Error::WindowOverBudget`.
#[derive(Debug)]
pub struct WindowOverBudget {
  /// The requested per-chunk token budget (`opts.window()`).
  window: usize,
  /// The model's fixed input window (the door's token ceiling).
  max: usize,
}

impl WindowOverBudget {
  /// Construct from the requested per-chunk token budget and the model's fixed
  /// input window it exceeded.
  #[inline(always)]
  pub const fn new(window: usize, max: usize) -> Self {
    Self { window, max }
  }

  /// The requested per-chunk token budget (`opts.window()`).
  #[inline(always)]
  pub const fn window(&self) -> usize {
    self.window
  }

  /// The model's fixed input window (the door's token ceiling).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// The text handed to a door's `embed_long_with` exceeds the caller-configured
/// input byte limit ([`LongTextOptions::max_input_bytes`]). Enforced BEFORE any
/// tokenizer or chunker work, so the reject path's cost is independent of the
/// input size — the limit to set when embedding untrusted text.
///
/// Payload of each text door's `Error::InputTooLarge`.
#[derive(Debug)]
pub struct InputTooLarge {
  /// The input length, in UTF-8 bytes.
  got: usize,
  /// The configured limit, in UTF-8 bytes.
  max: usize,
}

impl InputTooLarge {
  /// Construct from the input length and the configured byte limit it
  /// exceeded.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// The input length, in UTF-8 bytes.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The configured limit, in UTF-8 bytes.
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A contentless (separator-only) byte run that must be embedded as one whole
/// window — the ENTIRE input when it has no tokenizable content, or a
/// pure-separator gap between packed chunks that neither neighbor can absorb —
/// measures more tokens than the model's fixed window can hold, so embedding
/// it would silently drop its suffix tokens. Refused instead (measured with
/// the non-truncating tokenizer, special tokens included).
///
/// Payload of each text door's `Error::ContentlessInputOverBudget`.
#[derive(Debug)]
pub struct ContentlessInputOverBudget {
  /// Start of the offending run, as a UTF-8 byte offset into the input.
  start: usize,
  /// One past the end of the offending run, as a UTF-8 byte offset.
  end: usize,
  /// The run's untruncated token count (special tokens included).
  tokens: usize,
  /// The fixed window (the door's token ceiling).
  max: usize,
}

impl ContentlessInputOverBudget {
  /// Construct from the offending run's byte span, its untruncated token
  /// count, and the fixed window it exceeded.
  #[inline(always)]
  pub const fn new(start: usize, end: usize, tokens: usize, max: usize) -> Self {
    Self {
      start,
      end,
      tokens,
      max,
    }
  }

  /// Start of the offending run, as a UTF-8 byte offset into the input.
  #[inline(always)]
  pub const fn start(&self) -> usize {
    self.start
  }

  /// One past the end of the offending run, as a UTF-8 byte offset.
  #[inline(always)]
  pub const fn end(&self) -> usize {
    self.end
  }

  /// The run's untruncated token count (special tokens included).
  #[inline(always)]
  pub const fn tokens(&self) -> usize {
    self.tokens
  }

  /// The fixed window (the door's token ceiling).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A refusal raised by the long-text machinery.
///
/// Each door converts it into its own `Error` by `From`, variant for variant,
/// so a door's caller matches, reads and chains that door's variant. Each
/// message here is that variant's message, byte for byte.
#[derive(Debug, thiserror::Error)]
pub enum Error {
  /// The windit engine refused the content-aware chunking: whatever
  /// `ContentAware::chunk` rejects (a zero window, an overlap at or above the
  /// window, its own `max_windows` overrun), or the `max_windows` cap re-bound
  /// on the final chunk list. Carries windit's own typed error unchanged.
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[from] WinditError),

  /// The per-chunk token budget exceeds the door's token ceiling, so every
  /// chunk would be silently truncated. Rejected before any chunking runs.
  #[error(
    "embed_long window budget {} exceeds the model's fixed {}-token input window",
    .0.window(),
    .0.max()
  )]
  WindowOverBudget(WindowOverBudget),

  /// The text exceeds the caller-configured input byte limit
  /// ([`LongTextOptions::max_input_bytes`]), checked before any tokenizer or
  /// chunker work.
  #[error(
    "text input is {} bytes, exceeding the configured {}-byte limit",
    .0.got(),
    .0.max()
  )]
  InputTooLarge(InputTooLarge),

  /// A contentless run that must be embedded whole measures past the door's
  /// token ceiling (see [`ContentlessInputOverBudget`]).
  #[error(
    "contentless text at bytes {}..{} measures {} tokens, exceeding the model's fixed {}-token window",
    .0.start(),
    .0.end(),
    .0.tokens(),
    .0.max()
  )]
  ContentlessInputOverBudget(ContentlessInputOverBudget),
}

/// A door's exact token count of the text being chunked, in the two forms the
/// chunker needs.
///
/// The [`MeasureText`] half is what windit's `ContentAware` measures candidate
/// ranges with. It is infallible: a measure that fails folds to `usize::MAX`
/// ("does not fit"), so windit descends to a smaller range. A measure that
/// keeps the failure ([`take_failure`](Self::take_failure)) has it returned
/// by [`chunk_long`] in place of whatever windit made of those ranges; one
/// that does not leaves a persistent failure to resurface from the door's
/// per-window encode later.
/// [`measure_range`](Self::measure_range) is the fallible exact count the
/// repair runs — gap reattachment and the whole-input fallback — so a failure
/// there surfaces as the door's own error at synthesis rather than as a bogus
/// [`ContentlessInputOverBudget`] of `usize::MAX` tokens. The two are
/// deliberately not unified.
pub(crate) trait RangeMeasure: MeasureText {
  /// The door's error, which every refusal raised here converts into.
  type Error: From<Error>;

  /// The exact token count of `text[start..end]` of the text being chunked,
  /// counted the way the door will encode that range.
  ///
  /// # Errors
  /// The door's error if the count itself fails.
  fn measure_range(&self, start: usize, end: usize) -> Result<usize, Self::Error>;

  /// The first failure the infallible half met and answered "does not fit",
  /// if this measure keeps it. [`chunk_long`] returns it in place of its own
  /// result, whatever windit made of the ranges that did not fit. By default
  /// none is kept.
  fn take_failure(&self) -> Option<Self::Error> {
    None
  }

  /// Room for `additional` more chunks in a list the repair builds. By
  /// default, windit's own refusal: [`Error::Windowing`] carrying
  /// [`WinditError::AllocFailed`].
  ///
  /// # Errors
  /// The door's error if the room cannot be had.
  fn reserve_chunks(&self, chunks: &mut Vec<Chunk>, additional: usize) -> Result<(), Self::Error> {
    reserve_chunks(chunks, additional).map_err(Self::Error::from)
  }
}

/// windit's `ContentAware` takes `&dyn MeasureText`, which a measure behind a
/// `?Sized` parameter cannot coerce to. This forwards BOTH trait methods, so a
/// measure's early-stopping `measure_within` keeps its early stop (a closure
/// over the measure would fall back to the default full measure).
struct Forward<'m, M: ?Sized>(&'m M);

impl<M: MeasureText + ?Sized> MeasureText for Forward<'_, M> {
  fn measure(&self, text: &str) -> usize {
    self.0.measure(text)
  }

  fn measure_within(&self, text: &str, limit: usize) -> Option<usize> {
    self.0.measure_within(text, limit)
  }
}

/// Rejects an oversized or mis-budgeted long-text call before any tokenizer or
/// chunker work. Checked in order: the input byte limit
/// ([`Error::InputTooLarge`]), then the per-chunk budget against the door's
/// token `ceiling` ([`Error::WindowOverBudget`]) — an over-budget window would
/// let the door's encode silently truncate every chunk. Reads only `text.len()`
/// and the options — O(1), no tokenizer access — so the reject path's cost is
/// independent of the input size by construction. Factored out so the check is
/// hermetically testable. `window == 0` and `overlap >= window` are left to
/// windit's own validation (surfacing as [`Error::Windowing`]).
///
/// # Errors
/// [`Error::InputTooLarge`] if `text` exceeds
/// [`LongTextOptions::max_input_bytes`]; [`Error::WindowOverBudget`] if the
/// window exceeds `ceiling`.
pub(crate) fn validate_long_input(
  text: &str,
  opts: &LongTextOptions,
  ceiling: usize,
) -> Result<(), Error> {
  if let Some(max) = opts.max_input_bytes()
    && text.len() > max
  {
    return Err(Error::InputTooLarge(InputTooLarge::new(text.len(), max)));
  }
  let window = opts.window_options().window();
  if window > ceiling {
    return Err(Error::WindowOverBudget(WindowOverBudget::new(
      window, ceiling,
    )));
  }
  Ok(())
}

/// The pure text-splitting stage of a door's `embed_long` and `embed_windows`:
/// token-budgeted, boundary-aware byte ranges over `text`, measured with the
/// door's `measure` — which must measure this same `text`. Model-free, so the
/// chunk geometry is hermetically testable.
///
/// The chunks jointly cover `text`: windit's `ContentAware` extracts tokenized
/// content only, leaving separator bytes (paragraph breaks, whitespace-only
/// interiors, and word-fallback inter-word punctuation) uncovered at chunk
/// boundaries, so [`attach_gaps`] reattaches every such gap — re-measuring the
/// repaired substring against the window — before the chunks are returned. With
/// `overlap == 0` the chunks partition `text` (the first starts at byte 0, each
/// begins where the previous ends, the last ends at `text.len()`); a non-zero
/// overlap covers `text` while preserving its repeats. A
/// [`TailPolicy::DropBelowMin`] tail is a THIRD source of uncovered bytes, from
/// windit 0.5 (see [`LongTextOptions::tail_policy`]): the repair covers it like
/// any other gap, so the policy shifts the last boundary and never drops text.
/// Nonempty text always yields at least one chunk: text that chunks to nothing
/// — no tokenizable content at all (whitespace-only), or a lone chunk the tail
/// policy dropped — becomes a single whole-input chunk, the cost of the
/// whole-input `embed` fallback it is embedded by. Only `""` yields no
/// chunks.
///
/// Every chunk returned has an untruncated measure `<= ceiling`, the door's
/// token ceiling, with a single exception: windit's lone oversized `char` (one
/// `char` encodes to at most a handful of ids, far below a model window, so it
/// can exceed a small `window` but never the model window). Both contentless
/// escapes — a pure-separator gap [`attach_gaps`] would emit as its own chunk,
/// and the whole-input fallback chunk for text with no tokenizable content —
/// are MEASURED and refused past `ceiling` with
/// [`Error::ContentlessInputOverBudget`] rather than left for the door's encode
/// to silently truncate.
///
/// # Errors
/// Every refusal reaches the caller as the door's own error, through
/// `M::Error: From<Error>`: [`Error::Windowing`] carrying whatever windit's
/// `ContentAware::chunk` rejects (a zero window, an overlap at or above the
/// window, or a `max_windows` overrun), or `TooManyWindows` raised here when gap
/// reattachment or the whole-input fallback chunk grows the final list past
/// `opts.max_windows()` — the cap binds the FINAL chunk count, exactly the
/// per-chunk predictions the door dispatches, `got` reporting that full count
/// (windit's own raise aborts at `max + 1`; this one reports the whole
/// overrun); [`Error::ContentlessInputOverBudget`] if a contentless run that
/// must be embedded whole (the whole input, or a pure-separator gap
/// [`attach_gaps`] emits) measures past `ceiling` — measured at synthesis,
/// BEFORE the `max_windows` re-check; the door's own error if
/// [`RangeMeasure::measure_range`] fails on such a run; the failure the
/// measure kept ([`RangeMeasure::take_failure`]) while windit measured, in
/// place of windit's own result; and the measure's refusal of a chunk list's
/// room ([`RangeMeasure::reserve_chunks`]).
pub(crate) fn chunk_long<M: RangeMeasure + ?Sized>(
  measure: &M,
  text: &str,
  opts: &WindowOptions,
  ceiling: usize,
) -> Result<Vec<Chunk>, M::Error> {
  // windit measures through the infallible `MeasureText` half; the repair below
  // calls the fallible `measure_range` (see `RangeMeasure` for why the two stay
  // apart).
  let measure_checked = |a: usize, b: usize| measure.measure_range(a, b);
  let chunked = ContentAware::new(&Forward(measure)).chunk(text, opts);
  if let Some(failure) = measure.take_failure() {
    return Err(failure);
  }
  let chunks = chunked.map_err(|e| M::Error::from(Error::Windowing(e)))?;
  let mut repaired = attach_gaps_with(
    text,
    chunks,
    &measure_checked,
    &|list, additional| measure.reserve_chunks(list, additional),
    opts.window(),
    ceiling,
  )?;
  // Nonempty text can chunk to nothing two ways — no tokenizable content at all
  // (whitespace-only), or, from windit 0.5, a lone chunk below a
  // `TailPolicy::DropBelowMin` minimum, which windit discards as the tail it is
  // — yet the door still embeds it: the whole input through `embed`, one
  // prediction. So the tail policy cannot make a non-empty text embed to
  // nothing. Measure it first: a run measuring past `ceiling` would be silently
  // right-truncated by the door's encode (dropping its suffix tokens), so
  // refuse it with `ContentlessInputOverBudget`; otherwise represent the cost
  // as a single whole-input chunk so the cap below bounds every prediction the
  // result can dispatch. Only `""` stays chunkless (it fails `EmptyText` before
  // any prediction). The measure runs BEFORE the `max_windows` re-check, so
  // contentless over-budget input under `max_windows == 0` yields
  // `ContentlessInputOverBudget`, not `TooManyWindows`.
  if repaired.is_empty() && !text.is_empty() {
    let tokens = measure_checked(0, text.len())?;
    if tokens > ceiling {
      return Err(M::Error::from(Error::ContentlessInputOverBudget(
        ContentlessInputOverBudget::new(0, text.len(), tokens, ceiling),
      )));
    }
    measure.reserve_chunks(&mut repaired, 1)?;
    repaired.push(Chunk::new(0, text.len()));
  }
  // windit enforced `max_windows` on ITS output; each own-chunk the repair
  // inserts — and the whole-input fallback chunk above — grows the count past
  // that check, and every chunk costs one prediction, so the cap re-binds on
  // the final list: it is exactly the number of predictions the door may
  // dispatch. Fail-closed: coverage and the cap cannot both hold here, and
  // silently exceeding the caller's work bound (or silently dropping bytes)
  // would be worse than a typed refusal. `got` is the full final count, not
  // windit's abort count.
  if let Some(max) = opts.max_windows()
    && repaired.len() > max
  {
    return Err(M::Error::from(Error::Windowing(
      WinditError::TooManyWindows {
        got: repaired.len(),
        max,
      },
    )));
  }
  Ok(repaired)
}

/// Reattaches the byte gaps windit leaves between chunks, so [`chunk_long`]'s
/// output covers every byte of `text`. windit's `ContentAware` extracts
/// tokenized content only: paragraph separators (`\n\n` runs), whitespace-only
/// paragraph interiors, and — under its oversized-sentence word fallback — the
/// whitespace and punctuation between words are excluded, so a gap opens wherever
/// such bytes fall on a chunk boundary (including a leading gap before the first
/// chunk and a trailing gap after the last). A tail windit discarded under
/// [`TailPolicy::DropBelowMin`] is a trailing gap of the same kind, and is
/// reattached the same way — always as its own chunk, since the discarded tail
/// is by construction content its predecessor had no room for, so step 1 below
/// cannot take it.
///
/// A single left-to-right sweep closes every positive gap by re-measuring the
/// exact candidate substring against `window` (BPE is not additive — the repaired
/// range is re-measured, never assumed to gain a fixed token count), trying in
/// order:
///
/// 1. append the gap to the left neighbor if the extended range still fits —
///    left-first because terminal punctuation and paragraph breaks belong to the
///    preceding content, and it keeps every chunk starting where content starts;
/// 2. otherwise prepend it to the right neighbor if that range fits;
/// 3. otherwise emit the gap as its own chunk (pure separator bytes), reachable
///    only when both neighbors are already packed to exactly `window`.
///
/// With `overlap == 0` the result partitions `text`: the first chunk starts at
/// byte 0, each starts where the previous ends, the last ends at `text.len()`,
/// and the chunks concatenate back to `text`. With `overlap > 0` the pre-existing
/// overlaps are negative gaps, left untouched, so coverage is completed without
/// disturbing the repeats. The sweep never fuses two input chunks — each maps to
/// exactly one output chunk — so the output count is the input count plus one
/// per own-chunk emitted; [`chunk_long`] re-enforces `max_windows` on that
/// final count.
///
/// Every accepted attachment re-measures within `window`. A pure-separator
/// own-chunk (emitted when neither neighbor can absorb the gap) may still exceed
/// `window` up to `ceiling`, the door's token ceiling — the same tolerance as
/// windit's lone oversized-`char` escape — but its run is MEASURED, and a gap
/// measuring past `ceiling` is refused with
/// [`Error::ContentlessInputOverBudget`] rather than left for the door's encode
/// to silently truncate. Every constructed boundary is a windit cut or
/// `0`/`text.len()`, all on `char` boundaries, so `Chunk::new` never straddles a
/// `char` and `as_str` never returns `None`.
///
/// # Errors
/// [`Error::ContentlessInputOverBudget`] (as `E`) if a pure-separator gap
/// emitted as its own chunk measures more than `ceiling` tokens; whatever
/// `measure` returns if it fails on a candidate substring (surfaced here, one
/// call earlier than the door's per-chunk encode would); [`Error::Windowing`]
/// carrying [`WinditError::AllocFailed`] if the output list cannot be held.
#[cfg(all(test, feature = "granite"))]
pub(crate) fn attach_gaps<E: From<Error>>(
  text: &str,
  chunks: Vec<Chunk>,
  measure: &dyn Fn(usize, usize) -> Result<usize, E>,
  window: usize,
  ceiling: usize,
) -> Result<Vec<Chunk>, E> {
  attach_gaps_with(
    text,
    chunks,
    measure,
    &|list, additional| reserve_chunks(list, additional).map_err(E::from),
    window,
    ceiling,
  )
}

/// [`attach_gaps`], the output list's room had through `reserve`: the
/// measure's own door, where [`chunk_long`] calls it.
///
/// # Errors
/// As [`attach_gaps`], the output list's refusal being `reserve`'s.
fn attach_gaps_with<E: From<Error>>(
  text: &str,
  chunks: Vec<Chunk>,
  measure: &dyn Fn(usize, usize) -> Result<usize, E>,
  reserve: &dyn Fn(&mut Vec<Chunk>, usize) -> Result<(), E>,
  window: usize,
  ceiling: usize,
) -> Result<Vec<Chunk>, E> {
  let Some(&first) = chunks.first() else {
    return Ok(chunks);
  };
  // Each input chunk is pushed once, with at most one own-chunk beside it,
  // and the trailing gap adds at most one: at most `2n + 1` entries, reserved
  // up front so that no push below grows the list.
  let mut out = Vec::new();
  reserve(&mut out, chunks.len().saturating_mul(2).saturating_add(1))?;
  let mut cur = first;
  // Leading gap: extend the first chunk left to byte 0, else emit the gap alone
  // (measured and refused past `ceiling`, never left for the door's encode to
  // silently truncate). `measure(a, b)` is the exact untruncated count of
  // `text[a..b]`.
  if cur.start() > 0 {
    if measure(0, cur.end())? <= window {
      cur = Chunk::new(0, cur.end());
    } else {
      out.push(own_chunk(0, cur.start(), measure, ceiling)?);
    }
  }
  for mut next in chunks.into_iter().skip(1) {
    let (gap_start, gap_end) = (cur.end(), next.start());
    if gap_start < gap_end {
      if measure(cur.start(), gap_end)? <= window {
        cur = Chunk::new(cur.start(), gap_end);
      } else if measure(gap_start, next.end())? <= window {
        next = Chunk::new(gap_start, next.end());
      } else {
        out.push(cur);
        out.push(own_chunk(gap_start, gap_end, measure, ceiling)?);
        cur = next;
        continue;
      }
    }
    out.push(cur);
    cur = next;
  }
  // Trailing gap: extend the last chunk to `text.len()`, else emit the gap alone.
  if cur.end() < text.len() {
    if measure(cur.start(), text.len())? <= window {
      cur = Chunk::new(cur.start(), text.len());
    } else {
      let tail = own_chunk(cur.end(), text.len(), measure, ceiling)?;
      out.push(cur);
      cur = tail;
    }
  }
  out.push(cur);
  Ok(out)
}

/// Room for `additional` more chunks, refused as windit refuses a buffer it
/// cannot hold: [`WinditError::AllocFailed`], carried as [`Error::Windowing`].
fn reserve_chunks(chunks: &mut Vec<Chunk>, additional: usize) -> Result<(), Error> {
  chunks.try_reserve_exact(additional).map_err(|_| {
    Error::Windowing(WinditError::AllocFailed {
      elements: additional,
    })
  })
}

/// Builds the pure-separator own-chunk spanning the `start..end` byte range,
/// measuring its run first: a gap measuring past `ceiling`, the door's token
/// ceiling, would be silently right-truncated by the door's encode (dropping its
/// suffix tokens), so it is refused with [`Error::ContentlessInputOverBudget`]
/// instead. The `(window, ceiling]` tolerance is kept — the same shape as
/// windit's lone oversized `char`.
///
/// # Errors
/// [`Error::ContentlessInputOverBudget`] (as `E`) if the run exceeds
/// `ceiling`; whatever `measure` returns if it fails on the run.
fn own_chunk<E: From<Error>>(
  start: usize,
  end: usize,
  measure: &dyn Fn(usize, usize) -> Result<usize, E>,
  ceiling: usize,
) -> Result<Chunk, E> {
  let tokens = measure(start, end)?;
  if tokens > ceiling {
    return Err(E::from(Error::ContentlessInputOverBudget(
      ContentlessInputOverBudget::new(start, end, tokens, ceiling),
    )));
  }
  Ok(Chunk::new(start, end))
}

#[cfg(test)]
mod tests;
