#!/usr/bin/env bash
# F1 gate-inventory check: prove the end-to-end DER gates are actually COMPILED
# and still #[ignore]d — not silently feature-gated out, deleted, or un-ignored.
#
# The DER binaries (`tests/speaker/parity_e2e.rs`, `tests/speaker/parity_shipping_der.rs`,
# `tests/speaker/backend_factorial.rs`, all in the `coremlit-parity` package) are
# `#![cfg(feature = "speaker-oracle")]` — they need dia's own ort inference path as
# the parity oracle. Without `--features speaker-oracle` they compile to nothing, so
# `cargo test -p coremlit --features speaker -- --ignored` reports a green sweep containing
# ZERO DER tests. Every load-bearing gate here is ALSO `#[ignore]`d (each needs
# the gitignored `Models/` tree plus the sibling `diarization` fixtures), and
# the README drives them with `cargo test ... -- --ignored`.
#
# Every cargo invocation below names `--features speaker-oracle` — ONE oracle,
# deliberately, never `--all-features`. The failure this avoids: when a second
# crate in the same build turns on a different `ort` feature, Cargo unifies both
# onto the single `ort`, dia's first Session can then fail to `dlopen` an ONNX
# Runtime dylib that is not installed, and ort's error path re-enters the
# `OnceLock` it is initializing (setup_api -> Error::new_internal -> ort::api()
# -> Once::wait) and blocks forever. That is how `--all-features` used to hang
# these gates on `coremlit`, via `align-oracle`'s `asry` (`ort/load-dynamic`).
# `align-oracle` is not reachable from this package at all now, but the other
# two oracles here (`clap-oracle`, `vad-bundled`) each bring their own `ort`
# consumer, so the single-oracle invocation stays the rule.
#
# Two failure modes this must catch, which a plain `--list` cannot:
#   * a gate DELETED (or renamed) — e.g. dropping `stress_10...`, the central
#     argmax multi-speaker regression — leaves the sweep green with the gate
#     simply gone;
#   * a gate that LOST its `#[ignore]` — a plain `--list` renders every test as
#     `NAME: test` whether ignored or not, so an un-ignored heavy gate still
#     shows up there while the README's `-- --ignored` command silently STOPS
#     running it.
#
# The discriminator is `--list --ignored`, which restricts the listing to
# ignored tests only (libtest's `RunIgnored::Only` filter). Each expected gate
# below must appear in that IGNORED-only list — proving it is BOTH present AND
# still ignored. A deleted gate and an un-ignored gate both drop out of it and
# hard-fail here. The expected-name lists are an explicit, complete manifest of
# every load-bearing DER gate; a gate rename must update them (a deliberate
# act), so a gate cannot silently disappear.
#
# codex r6 F4 — this inventory ALSO EXECUTES each binary's ORDINARY
# (non-`--ignored`) suite. The README drove these binaries with `-- --ignored`
# only, which SKIPS their hermetic ordinary tests: der_calc's DER-math units and
# the mutation-proof pin guards (`assert_pinned_fires_...`,
# `clip09_known_defect_pins_every_field`). Those need no models and no fixtures,
# so dropping an `assert_pinned` clause left every documented command green while
# a pin silently un-breached. Running the ordinary suite here turns that red.
#
# codex r7 F2 — EXECUTING the ordinary suite still only checked that SOME test
# passed (`passed > 0`), which der_calc's seven always-present math units satisfy
# on their own. DELETING a pin/mutation guard therefore stayed green. So each
# binary now also carries an explicit REQUIRED-ORDINARY-name manifest
# (`check_ordinary`), asserted present-and-not-`#[ignore]`d BEFORE the suite runs,
# the ordinary-test analogue of the `#[ignore]`d-gate manifests below.
#
# Run from the workspace root: crates/coremlit-parity/tests/speaker/der_gate_inventory.sh
# Kept a shell script (not a `cargo test`) on purpose: it must shell out to
# `cargo`, which cannot nest inside a `cargo test` run without deadlocking on
# the target-dir lock. Written for bash 3.2 (macOS default) — no associative
# arrays.
#
# Every membership test below matches with a HERESTRING, never
# `printf ... | grep -q`. That idiom is a latent SIGPIPE bug: `grep -q` exits on
# its FIRST match and closes the read end of the pipe, while bash's `printf`
# builtin writes through a 512-byte stdio buffer (measured on macOS bash 3.2),
# so a payload over 512 bytes costs it two or more `write`s and the later one
# can land on a closed pipe — SIGPIPE, 141, which `pipefail` promotes to the
# pipeline's status. Here that corrupts the HEALTHY branch: "present + ignored"
# falls through, and when its `elif` fallback races too the gate reports a
# present, correctly `#[ignore]`d DER gate as "deleted or renamed".
#
# Measured (needle in the first chunk, 40 runs per size): <=512 bytes never
# fails, 513 bytes fails 37/40, 4 KB fails 40/40; move the needle into the last
# chunk and it never fails, because grep must consume everything first. So it
# depends on payload size AND match position.
#
# These payloads straddle the threshold. Measured `--list` sizes here:
#
#     binary                 all      ignored
#     parity_e2e            1104          391
#     parity_shipping_der    751          397
#     backend_factorial      583          135
#
# Every `all` is OVER 512 and every `ignored` is UNDER it, which is the only
# reason this script is not failing today: `check_bin` tests `ignored` first,
# and `check_ordinary`'s racing `all` test happens to fall through to the branch
# that was correct anyway. Neither is a property anyone chose. `ignored` is ~120
# bytes — three gate names — from crossing, and on the far side `check_bin`
# starts reporting present, correctly `#[ignore]`d gates as "deleted or
# renamed", in the `check` job, on every push and PR. Do not reason about sizes;
# remove the writer. A herestring feeds grep from a temp file, so nothing can
# take SIGPIPE.
set -euo pipefail

# Verify one DER binary: $1 = test binary name, $2.. = expected `#[ignore]`d
# gate names. Each must appear in the binary's IGNORED-only test list.
check_bin() {
  bin="$1"
  shift
  echo "== ${bin} =="
  # Full `--list`: one `NAME: test` line per test (ignored or not); stderr
  # (compile noise) dropped. Used only for non-vacuity and to tell a DELETED
  # gate from an UN-IGNORED one. A compile FAILURE still surfaces because
  # `cargo` exits non-zero and the empty list below trips the hard-fail.
  all="$(cargo test -p coremlit-parity --features speaker-oracle --test "speaker_${bin}" -- --list 2>/dev/null || true)"
  count="$(printf '%s\n' "${all}" | grep -c ': test$' || true)"
  if [ "${count}" -eq 0 ]; then
    echo "  FAIL: 0 tests listed for ${bin} — it compiled to nothing."
    echo "        (missing --features speaker-oracle, a broken #![cfg(feature = \"speaker-oracle\")] gate, or a build error)"
    return 1
  fi
  # `--list --ignored`: the SAME `NAME: test` shape, but restricted to ignored
  # tests. This is what distinguishes an ignored gate from an un-ignored one.
  ignored="$(cargo test -p coremlit-parity --features speaker-oracle --test "speaker_${bin}" -- --list --ignored 2>/dev/null || true)"
  ignored_count="$(printf '%s\n' "${ignored}" | grep -c ': test$' || true)"
  echo "  ${count} tests listed (${ignored_count} ignored)"
  rc=0
  for name in "$@"; do
    if grep -q "^${name}: test$" <<<"${ignored}"; then
      echo "  ok:   ${name} (present + ignored)"
    elif grep -q "^${name}: test$" <<<"${all}"; then
      echo "  FAIL: gate '${name}' is present in ${bin} but is NO LONGER #[ignore]d —"
      echo "        the README's \`-- --ignored\` command would silently stop running it."
      rc=1
    else
      echo "  FAIL: expected DER gate '${name}' is not in ${bin}'s test list (deleted or renamed)."
      rc=1
    fi
  done
  return "${rc}"
}

# Run one DER binary's ORDINARY (non-`--ignored`) suite and assert it PASSES with
# at least one test (codex r6 F4). These are the hermetic gates the README's
# `-- --ignored` command silently SKIPS — der_calc's DER-math unit tests plus the
# mutation-proof pin guards — and they need no models and no fixtures, so they run
# here in CI/dev without the gitignored `Models/` tree. `check_bin` above proves
# each gate is COMPILED and still #[ignore]d; this proves the hermetic ordinary
# tests actually RUN and pass, so a dropped `assert_pinned` clause is caught.
run_ordinary() {
  bin="$1"
  echo "== ${bin} (ordinary suite) =="
  out="$(cargo test -p coremlit-parity --features speaker-oracle --test "speaker_${bin}" 2>&1)" || {
    printf '%s\n' "${out}" | tail -25
    echo "  FAIL: ${bin} ordinary suite did not pass — a hermetic gate (der_calc math or a"
    echo "        mutation-proof pin guard) is red."
    return 1
  }
  # Non-vacuity: `cargo test` exits 0 even with zero tests, so require passed > 0.
  passed="$(printf '%s\n' "${out}" | sed -n 's/^test result: ok\. \([0-9][0-9]*\) passed.*/\1/p' | tail -1)"
  if [ -z "${passed}" ] || [ "${passed}" -eq 0 ]; then
    echo "  FAIL: ${bin} ran ZERO ordinary tests — the hermetic gate suite vanished."
    return 1
  fi
  echo "  ok:   ${bin} ordinary suite passed (${passed} hermetic tests)"
  return 0
}

# Assert each REQUIRED ORDINARY (non-`--ignored`) gate is (a) present in the
# binary's full test list AND (b) absent from its ignored-only list — i.e. it is
# a hermetic test `run_ordinary` will actually EXECUTE — BEFORE running the suite
# (codex r7 F2). `run_ordinary` only checks that SOME test passed, so der_calc's
# always-present DER-math units kept the count nonzero even after a pin/mutation
# guard (`assert_pinned_...`, `clip09_known_defect_pins_every_field`) was DELETED
# or accidentally `#[ignore]`d — the deletion the pins exist to prevent slipped
# through green. Naming each falsifiability guard here turns that red. The names
# mirror the ordinary gates the DER binaries compile; a deliberate rename must
# update this manifest, so a guard cannot silently disappear (same contract as
# the `#[ignore]`d-gate manifests in `check_bin`).
check_ordinary() {
  bin="$1"
  shift
  echo "== ${bin} (required ordinary gates) =="
  all="$(cargo test -p coremlit-parity --features speaker-oracle --test "speaker_${bin}" -- --list 2>/dev/null || true)"
  ignored="$(cargo test -p coremlit-parity --features speaker-oracle --test "speaker_${bin}" -- --list --ignored 2>/dev/null || true)"
  rc=0
  for name in "$@"; do
    if ! grep -q "^${name}: test$" <<<"${all}"; then
      echo "  FAIL: required ordinary gate '${name}' is not in ${bin} (deleted or renamed)."
      rc=1
    elif grep -q "^${name}: test$" <<<"${ignored}"; then
      echo "  FAIL: required ordinary gate '${name}' is now #[ignore]d in ${bin} —"
      echo "        run_ordinary would SKIP it while still reporting a green pass count."
      rc=1
    else
      echo "  ok:   ${name} (present + ordinary)"
    fi
  done
  return "${rc}"
}

fail=0

# parity_e2e.rs — the fp32 dia-ort parity gate, the argmax characterization, the
# compute-unit study, and ALL FOUR multi-speaker stress clips. The argmax
# multi-speaker regression lives in `stress_10...`; deleting it (or any stress
# clip) must fail here, not slip through a green `--ignored` sweep.
check_bin parity_e2e \
  fluidaudio_der_parity_vs_dia_ort_and_determinism \
  argmax_source_der_characterization \
  compute_unit_der_study_all_vs_cpuonly \
  stress_10_mrbeast_clean_water_7_speakers \
  stress_06_long_recording_3_speakers \
  stress_12_mrbeast_schools_15_speakers \
  stress_14_mrbeast_strongman_robot_4_speakers || fail=1

# parity_shipping_der.rs — ALL FOUR shipping DER clips (06, 14, 10, 09; the
# fp32 shipping configuration + placement controls since issue #15), plus the
# shipping-default resolver gate, the corpus-selection gate, and the clip-09
# audio-content pin.
check_bin parity_shipping_der \
  shipping_der_06_long_recording_3spk \
  shipping_der_14_mrbeast_strongman_robot_4spk \
  shipping_der_10_mrbeast_clean_water_7spk \
  shipping_der_09_mrbeast_dollar_date_8spk \
  shipping_default_is_the_fp32_embedder \
  shipping_clip_selection_is_the_documented_subset \
  clip09_content_pin_catches_an_audio_swap || fail=1

# backend_factorial.rs — the seg-vs-embed cross-product at the int8-era
# shipping configuration, the precision x placement experiment that
# disambiguates what that cross-product could only implicate as a bundle, and
# the mechanism probe that says WHAT KIND of perturbation each factor applies.
# `model_io.rs`'s recorded attribution (and the issue-#15 embedder retirement)
# cites all three, so none must be deletable without a red build.
check_bin backend_factorial \
  shipping_config_backend_factorial \
  embedding_precision_x_placement \
  quantization_error_structure || fail=1

# ── Require each binary's load-bearing ORDINARY (hermetic) gates by NAME, then
#    execute the ordinary suite (codex r7 F2 + r6 F4). The name manifest runs
#    FIRST so a deleted/`#[ignore]`d pin-falsifiability guard fails even though
#    der_calc's math units would keep run_ordinary's pass count nonzero. ──
check_ordinary parity_e2e \
  assert_pinned_fires_when_a_value_crosses_the_parity_bound \
  equal_delta_der_hides_disjoint_arm_errors \
  stress_gate_roster_is_consistent || fail=1
check_ordinary parity_shipping_der \
  clip09_record_pins_every_field || fail=1
check_ordinary backend_factorial \
  factorial_verdict_pins_every_cell \
  precision_placement_verdict_pins_every_cell \
  mechanism_verdict_pins_every_field || fail=1

run_ordinary parity_e2e || fail=1
run_ordinary parity_shipping_der || fail=1
run_ordinary backend_factorial || fail=1

if [ "${fail}" -ne 0 ]; then
  echo "DER gate inventory FAILED — the gates above are not all compiled, present, and #[ignore]d." >&2
  exit 1
fi
echo "DER gate inventory OK — every expected DER gate is compiled, listed and still ignored, and each binary's ordinary (hermetic) suite passed."
