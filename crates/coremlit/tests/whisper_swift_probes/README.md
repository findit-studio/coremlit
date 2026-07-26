# Whisper Swift oracle probes (H1 f16 mass-rule + H2 argmax tie-break + H6 alignment-gather row pitch)

Documentation-grade reference evidence for the pinned numeric values in the
coremlit issue #41 parity fixes:

- `crates/coremlit/src/audio/whisper/decode/filter/mod.rs` — `bnns_mass_rule_scalars`
  (H1: BNNS f16 timestamp-mass rule replication)
- `crates/coremlit/src/audio/whisper/decode/sampler/mod.rs` — `argmax`
  (H2: first-index, NaN-skipping tie-break)
- `crates/coremlit/src/audio/whisper/segment/mod.rs` — `coreml_f16_row_pitch` /
  `gather_swift_parity_into` / `gather_swift_rows` / `add_word_timestamps`, and
  `options/mod.rs` — `AlignmentGather`
  (H6: the CoreVideo row pitch Swift's alignment gather ignores)

These files are **captured verbatim** from a one-off probe run and are **not
compiled or run in CI** (they are `.swift`/`.out`, never `.rs`, so `cargo` never
picks them up as integration tests). They exist so every "probe-verified" claim
in these functions' doc comments and in their hermetic tests can be traced to
a concrete oracle capture, following the `crates/coremlit/tests/speaker/swift/`
precedent. The hermetic Rust tests in `decode/filter/tests.rs`,
`decode/sampler/tests.rs` and `segment/tests.rs` are the executable, CI-enforced
form of this evidence; these captures are the human-readable provenance behind
their pinned hex and pitches.

## Provenance

- **Host:** macOS 26.5 (25F71), Apple M1 Max, arm64 — near-identical to the
  issue's reference host (macOS 26.5.2 25F84, M1).
- **Toolchain:** Swift 6.3.3 (swiftlang-6.3.3.1.3), SDK MacOSX26.5. Rust
  `half = 2.7.1` (the workspace's pinned version), sequential-f32 `exp`/`ln` =
  system libm.
- **Oracle:** `argmax-oss-swift @ dcf3a00` (WhisperKit), verified clean at the
  issue's pin. `FloatType = Float16` on arm64 (`Sources/ArgmaxCore/FloatType.swift:10`).
- **Build/run:**
  - `swiftc -O -parse-as-library probe_argmax.swift   -o probe_argmax   && ./probe_argmax`
  - `swiftc -O -parse-as-library probe_massrule.swift -o probe_massrule && ./probe_massrule`
  - `swiftc -O -parse-as-library probe_alignment_stride.swift -o probe_alignment_stride && ./probe_alignment_stride`

The probes mirror the oracle's exact API shapes: `BNNSNDArrayDescriptor(...,
scalarType: Float16.self, shape: .vector(n, stride: 1))`, `allocateUninitialized`
outputs, `BNNS.applyActivation(.logSoftmax, batchSize: 1)`,
`BNNS.applyReduction(.logSumExp/.max/.argMax, weights: nil)`
(`LogitsFilter.swift:144-242`, `TokenSampler.swift:86-197`), and
`MLTensor(MLShapedArray<Float16>(mlMultiArray)).cast(to: Float.self).argmax(alongAxis: -1)`
→ `shapedArray(of: Int32.self)` (`TokenSampler.swift:42-83`,
`MLTensorExtensions.swift:14-16`). On macOS 15+ Swift samples via
`sampleWithMLTensor` — the f32-cast MLTensor argmax is the shipping tie-break
path; BNNS f16 `argMax` is the legacy path. Both are probed here.

## Files

| File | Probe | Contents |
| --- | --- | --- |
| `probe_argmax.swift`   | H2 | Source: 16 crafted argmax tie/NaN/signed-zero cases, MLTensor and BNNS paths, 3 repeats each. |
| `probe_argmax.out`     | H2 | Earlier partial run — 11 cases (through `vocab_all_equal`). |
| `probe_argmax2.out`    | H2 | Complete run — all 16 cases, **includes the signed-zero, `nan_at_4`/`nan_at_0`, and `all_nan` cases** that discriminate first-index-IEEE from `total_cmp`. |
| `probe_massrule.swift` | H1 | Source: Q1–Q4 decisive probes (internal-precision, max-subtract, edge semantics, random sweeps, near-margin scans, V3 bit-pinned dump). |
| `probe_massrule.out`   | H1 | Earlier partial run — flip-point scans empty, truncated before the V3 dump. |
| `probe_massrule2.out`  | H1 | **Complete run** — the authoritative capture: flip points `0xb17c` (scan1) / `0xc05e` (scan2), the V3 pins `lse=0xb7ae max=0xc4f2`, and the NaN / all-`-inf` edge semantics. |
| `probe_alignment_stride.swift` | H6 | Source: `MLMultiArray.strides` for ten pixel-buffer-backed Float16 shapes against the plain initializer, plus a write-at-true-stride / read-at-`columnCount` replay of the gather at the shipping 120 × 1500. |
| `probe_alignment_stride.out`   | H6 | Complete run — the pitch table (1500 → **1504**, 100 → 128, 8/9 → 32) and the per-row overrun (`logical row 119 reads 476 element(s) past the copied prefix (kept columns = 1024)`). |

Both `.out` variants are kept for each probe: the base `.out` is an earlier
partial capture, and the `2.out` is the complete, final run. **The pinned values
in the port come from the `2.out` captures** (base `.out` has empty flip-point
scans and no V3 dump). The Q1/Q2/Q3 internal-precision evidence (lines 1–13) is
identical in both.

## What the captures pin

- **H2 (`probe_argmax2.out`):** Swift's argmax is deterministic **first-index**
  on every crafted tie (small/vocab-size, adjacent/distant, many-way, all-equal,
  `-inf`-dominated), signed zeros compare IEEE-equal (`[-0.0@2, +0.0@5]` and the
  reverse both pick 2), and NaN is skipped wherever it sits (`nan_at_4` and
  `nan_at_0` both → 7). The all-NaN case (`all_nan`) diverges by path —
  MLTensor→0, BNNS→last (index 15 at n=16) — unspecified/inconsistent
  upstream; the port pins 0 (the shipping MLTensor path).
- **H1 (`probe_massrule2.out`):** BNNS computes internally in **f32** and rounds
  to f16 only at each operation's output (crafted `[8, -3×1100]`:
  `bnns=0x4802` = f32-sequential, ≠ `0x4800` = pure-f16). Its `.logSumExp`
  reduction is **naive** — no max subtraction (`LSE([-110×1101]) = -inf`, not the
  stable `-103`) — while `.logSoftmax` **is** max-subtracted (`[88×4]` and
  `[100×4]` both give `-1.387`, not `-inf`). A sequential-f32 emulation with
  f16-round-to-nearest-even at each output reproduces BNNS's boolean at the exact
  f16-input flip points (`0xb17c`, `0xc05e`) and on 1500/1500 + 299/300
  adversarially margin-tuned sweeps. Probed edge quirks: `.max(all -inf)` returns
  `-65504` (lowest finite f16), not `-inf` — boolean-immaterial.
- **H6 (`probe_alignment_stride.out`):** WhisperKit's Float16 `MLMultiArray`s
  are **not contiguous**. `MLMultiArray(shape:dataType:initialValue:)`
  (`ArgmaxCore/MLMultiArrayExtensions.swift:11-53`) backs `.float16` with an
  IOSurface `CVPixelBuffer` (`:121-136`), and CoreVideo pads each row out to a
  platform-chosen boundary — 64 bytes, i.e. 32 Float16 elements, **on this
  host** — so `strides[0]` is 1504 for `cols = 1500` (and 128 for 100, 32 for 8
  and for 9), while the plain `MLMultiArray(shape:dataType:)` initializer
  reports the contiguous 1500.
  `SegmentSeeker.addWordTimestamps` binds both stride arrays and then pitches
  its `memcpy` by `columnCount` (`:452-460`), so the gather writes only storage
  `[0, N * cols)` and, because `filteredIndices` are `0..<N`, destination
  storage offset `o` ends up holding SOURCE storage offset `o`;
  `dynamicTimeWarping`'s flat subscript (`:217`) is stride-aware and reads
  logical row `r` at `[r * dst_pitch, r * dst_pitch + cols)`.
  **When the two surfaces share a pitch** the two errors cancel wherever that
  read window still lies inside the copied prefix: row `r` is truncated
  exactly when `r * pitch + cols > N * cols`, and
  only the LAST row can be while `cols >= (N - 2) * (pitch - cols)`. That bound
  is a property of the shipping shape rather than a general one — it holds at
  1500/1504 for any `N <= 224` (the second-to-last row would need `N > 377`)
  and fails at the small `n_audio_ctx` a test backend can set, where a run of
  whole rows goes. At `N = 120` the final row keeps **1024** of its 1500
  columns and reads the other 476 out of storage neither the `memcpy` nor the
  `initialValue:` fill ever wrote. That row is the whole of the
  long-form divergence: it moves the last word's end, hence the segment's end,
  hence the next window's seek. Pinned in `segment/tests.rs` by
  `coreml_f16_row_pitch_answers_with_a_usable_pitch_or_a_typed_refusal`,
  `swift_gather_keeps_only_the_final_rows_prefix` and
  `swift_parity_gather_truncates_final_alignment_row`.

  **When they do NOT share a pitch it is not a truncation at all** — row `r`
  reads a shifted window that splices one logical source row's tail onto the
  next one's head across the source's own padding. Row-count invariance is
  something this host was measured to have, not something CoreVideo promises,
  so the port probes BOTH shapes — the once-allocated **`[kvCacheMaxSequence
  Length, cols]` = `[224, cols]`** source (`TextDecoder.swift:141`) and the
  per-call `[N, cols]` destination (`:450`) — and
  `segment::gather_swift_parity_into`/`gather_swift_rows` reproduce the general
  mapping. `swift_gather_reproduces_the_copy_when_the_two_surfaces_pitch_
  differently` drives it against a storage-level replay at pitches this host
  will never choose.

  **The source height is Swift's 224, not the port's 225.** This port's
  accumulator is `max_token_context + 1` rows because it commits step
  `position`'s row at `position + 1`; Swift has no such headroom. Probing the
  port's height would decode Swift's storage offsets with a pitch Swift's
  array never had wherever `pitch(224, cols) != pitch(225, cols)`, which the
  branch explicitly allows. `swift_parity_probes_swifts_source_height_not_
  this_ports_commit_headroom` injects exactly such a host and asserts which
  height the probe names (round-3 finding #2).

  **`AlignmentGather::Complete` is the DEFAULT; `SwiftParity` is an opt-in
  that fails closed** (owner decision, round 3 of the #41 review). `Complete`
  is the numerically correct gather and has no platform dependency at all — no
  IOSurface allocation, no pitch measurement, no inspection of storage it did
  not write, no `unsafe`. `SwiftParity` reproduces a Swift defect whose
  behavior rests on unspecified CoreVideo layout, and three review rounds each
  uncovered a further unspecified layer beneath the previous one, so it is
  what a caller explicitly asks for having accepted three stated assumptions
  (A1 layout determinism, A2 fresh-allocation zero fill, A3 layout stability
  across `withUnsafeMutableBytes`) enumerated on the variant itself. Nothing
  below claims those assumptions are established.

  **The pitch table above is evidence about this host, not a compiled-in
  rule.** `coreml_f16_row_pitch` MEASURES the running host — it allocates the
  same Float16 IOSurface Swift's gather allocates
  (`MultiArray::f16_surface`) and reads CoreVideo's own strides back — because
  Apple's QA1829 states `CVPixelBuffer` row alignment is hardware-dependent and
  must be queried. An earlier cut of this fix promoted the 32-element quantum
  observed here into a `const fn`, which would have zeroed the WRONG cells (and
  so silently produced non-parity word timings, segment ends and seeks) on a
  host that pads differently. The recorded table survives here and in the
  **`#[ignore]`d** `reference_host_pitch_table` as provenance for the
  hand-computed columns in the model-gated gather fixtures — ignored because
  it compares this machine against one recorded machine, which no production
  path depends on. The live-allocation diagnostics
  (`coreml_f16_row_pitch_reports_this_hosts_live_allocation_layout`) are
  `#[ignore]`d for the same reason: they assert that two independent
  allocations of one shape pitch identically, which is assumption A1 holding
  here rather than anything CoreVideo owes. The portable gate,
  `coreml_f16_row_pitch_answers_with_a_usable_pitch_or_a_typed_refusal`,
  accepts the typed refusals as legitimate outcomes on a valid host (round-3
  finding #5). If a measurement cannot be made at all,
  `AlignmentGather::SwiftParity` fails closed
  (`SegmentError::AlignmentPitchUnavailable`) rather than quietly degrading to
  `AlignmentGather::Complete`.

  **The destination's untouched tail is ASSUMED zero, and that assumption is
  stated rather than disguised (A2).** The storage past `N * cols` is read by
  `dynamicTimeWarping` and written by nobody — the `initialValue:` fill covers
  only the logical `count`, and the `memcpy` covers the same prefix — so it is
  whatever `CVPixelBufferCreate` returned, and no part of CoreVideo promises
  that is zero. An earlier revision claimed to VERIFY it by sampling a freshly
  allocated surface's tail before zeroing it. That was wrong twice over, and
  both errors are now removed:

  - reading those bytes was **undefined behavior** — `read_volatile` prevents
    elision and reordering, not undefinedness, and the safe `f16_surface`
    constructor could execute it under exactly the dirty-tail condition the
    scan existed to detect (round-3 finding #1). `f16_surface` is now
    write-only: it zeroes `CVPixelBufferGetDataSize` bytes and never observes
    the prior contents.
  - one clean allocation proves nothing about a **different** allocation. The
    probe sampled a disposable surface, dropped it, and then synthesized zeros
    for the surface the gather actually described; reuse or a concurrent
    allocation yields a clean probe and a dirty real tail (round-3
    finding #3). No probe replaces it, because none can: the allocation Swift
    gathered into lives in another process.

  **The source padding's zero is likewise an assumption past construction
  (A3).** The `initialValue: FloatType(0)` initializer zeroes the LOGICAL
  element count flat over a pitched buffer, so storage `[0, src_rows * cols)`
  — padding included — *starts* at zero, and `TextDecoder.updateAlignment
  Weights` (`:272-295`), the array's only writer, is stride-aware. That the
  zero SURVIVES is unverified: writer and gather both reach the array through
  `withUnsafeMutableBytes`, whose contract states the closure-provided strides
  may differ from the value before invocation, so no cited contract rules out
  a relayout introducing padding construction never zeroed (round-3
  finding #4). A lifecycle-equivalent native probe at the 224-row height would
  upgrade this to a one-host observation; it would still not be a contract.

## The Swift short-form word-timestamp golden

`crates/coremlit/tests/whisper/fixtures/golden/jfk_tiny_words_golden.json` is a
verbatim capture of the same oracle (`argmax-oss-swift @ dcf3a00`, same host)
running `jfk.wav` on `openai_whisper-tiny` with **word timestamps on**, taken
through an out-of-tree SwiftPM driver that depends on the pinned checkout by
path and writes nothing into it (the `crates/coremlit/tests/speaker/swift/`
precedent). The driver calls `WhisperKit.transcribe(audioArray:decodeOptions:)`
with:

```swift
DecodingOptions(
  verbose: true, task: .transcribe, language: nil,
  temperature: 0.0, temperatureFallbackCount: 0,
  usePrefillPrompt: true, skipSpecialTokens: true, withoutTimestamps: false,
  wordTimestamps: true, concurrentWorkerCount: 1,
  chunkingStrategy: ChunkingStrategy.none)
```

under `ModelComputeOptions(audioEncoderCompute: .cpuAndNeuralEngine,
textDecoderCompute: .cpuAndNeuralEngine)` — the same pinned invocation the #41
long-form evidence used, so short-form and long-form are one option set.
`whisper_parity_jfk`'s
`jfk_tiny_word_timestamps_match_swift_and_this_clip_is_gather_invariant`
mirrors those options exactly and asserts every word (text, start, end,
probability, token ids) with no epsilon **under the shipping default,
`AlignmentGather::Complete`**, plus that the opt-in `AlignmentGather::
SwiftParity` yields the identical list. The first half is what the round-3
default flip had to re-establish — the correct-everywhere gather still
reproduces official Swift on this clip.

The second half establishes gather-invariance **for this clip and nothing
wider**, and the wording matters because an earlier revision generalized it.
The gather runs on every word-timestamp window, long-form or not, and at 30
gathered rows over 1500 columns the measured pitch of 1504 does truncate the
final row under `SwiftParity` (it keeps 1384 of 1500 columns) — so the matrix
DTW consumes here genuinely differs between the modes, and it is only the path
through it that coincides. That coincidence is a property of this clip's
numbers: the port's own fixtures measure the modes producing different
word/segment ends **inside a single window** (0.88 s vs 1.58 s over one
`add_word_timestamps` call in `segment::tests::
swift_parity_gather_truncates_final_alignment_row`; 0.88 s vs 1.18 s in the
first window of `transcribe::tests::
swift_parity_gather_moves_the_first_windows_end_and_the_next_seek`, with no
cascade behind it). Settling the general short-form question needs a corpus of
Swift word goldens across clips, models and window occupancies; this golden is
one point in it. See `AlignmentGather::SwiftParity`'s "Short-form" section.

## Rust transfer check (superseded by hermetic tests)

An exploratory Rust probe (`half 2.7.1` + system libm) confirmed every dumped
value bit-identical to both the Swift emulation and BNNS itself (V3
`lse=0xb7ae max=0xc4f2`; scan flips `0xb17c`/`0xc05e`). That transfer check is
now encoded directly as the CI-run hermetic tests
`mass_rule_scalars_match_bnns_pinned_vector`,
`mass_rule_flip_points_match_bnns_scan1`, and `..._scan2` — the committed tests
ARE the Rust-side proof, so the exploratory cargo probe is not vendored here.
