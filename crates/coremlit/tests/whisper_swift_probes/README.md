# Whisper Swift oracle probes (H1 f16 mass-rule + H2 argmax tie-break)

Documentation-grade reference evidence for the pinned numeric values in the
coremlit issue #41 parity fixes:

- `crates/coremlit/src/audio/whisper/decode/filter/mod.rs` — `bnns_mass_rule_scalars`
  (H1: BNNS f16 timestamp-mass rule replication)
- `crates/coremlit/src/audio/whisper/decode/sampler/mod.rs` — `argmax`
  (H2: first-index, NaN-skipping tie-break)

These files are **captured verbatim** from a one-off probe run and are **not
compiled or run in CI** (they are `.swift`/`.out`, never `.rs`, so `cargo` never
picks them up as integration tests). They exist so every "probe-verified" claim
in the two functions' doc comments and in their hermetic tests can be traced to
a concrete oracle capture, following the `crates/coremlit/tests/speaker/swift/`
precedent. The hermetic Rust tests in `decode/filter/tests.rs` and
`decode/sampler/tests.rs` are the executable, CI-enforced form of this evidence;
these captures are the human-readable provenance behind their pinned hex.

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
| `probe_argmax2.out`    | H2 | Later run — 15 cases, **includes the signed-zero and `nan_at_4` cases** that discriminate first-index-IEEE from `total_cmp`. |
| `probe_massrule.swift` | H1 | Source: Q1–Q4 decisive probes (internal-precision, max-subtract, edge semantics, random sweeps, near-margin scans, V3 bit-pinned dump). |
| `probe_massrule.out`   | H1 | Earlier partial run — flip-point scans empty, truncated before the V3 dump. |
| `probe_massrule2.out`  | H1 | **Complete run** — the authoritative capture: flip points `0xb17c` (scan1) / `0xc05e` (scan2), the V3 pins `lse=0xb7ae max=0xc4f2`, and the NaN / all-`-inf` edge semantics. |

Both `.out` variants are kept for each probe: the base `.out` is an earlier
partial capture, and the `2.out` is the complete, final run. **The pinned values
in the port come from the `2.out` captures** (base `.out` has empty flip-point
scans and no V3 dump). The Q1/Q2/Q3 internal-precision evidence (lines 1–13) is
identical in both.

## What the captures pin

- **H2 (`probe_argmax2.out`):** Swift's argmax is deterministic **first-index**
  on every crafted tie (small/vocab-size, adjacent/distant, many-way, all-equal,
  `-inf`-dominated), signed zeros compare IEEE-equal (`[-0.0@2, +0.0@5]` and the
  reverse both pick 2), and NaN is skipped wherever it sits (`nan_at_4` → 7).
  The all-NaN case (MLTensor→0, BNNS→last) is unspecified upstream; the port
  pins 0 (the shipping MLTensor path).
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

## Rust transfer check (superseded by hermetic tests)

An exploratory Rust probe (`half 2.7.1` + system libm) confirmed every dumped
value bit-identical to both the Swift emulation and BNNS itself (V3
`lse=0xb7ae max=0xc4f2`; scan flips `0xb17c`/`0xc05e`). That transfer check is
now encoded directly as the CI-run hermetic tests
`mass_rule_scalars_match_bnns_pinned_vector`,
`mass_rule_flip_points_match_bnns_scan1`, and `..._scan2` — the committed tests
ARE the Rust-side proof, so the exploratory cargo probe is not vendored here.
