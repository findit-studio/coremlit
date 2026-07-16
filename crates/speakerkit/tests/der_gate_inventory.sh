#!/usr/bin/env bash
# F1 gate-inventory check: prove the end-to-end DER gates are actually COMPILED,
# not silently feature-gated out.
#
# The DER binaries (`tests/parity_e2e.rs`, `tests/parity_shipping_der.rs`) are
# `#![cfg(feature = "dia")]`. Without `--features dia` they compile to nothing,
# so `cargo test -p speakerkit -- --ignored` reports a green sweep containing
# ZERO DER tests. This script lists each DER binary's tests (via `--list`, which
# compiles but does not run them) and hard-fails if the list is EMPTY or an
# expected gate is missing — making a feature-selection no-op distinguishable
# from a real pass. A gate rename must update the expected-name lists below
# (a deliberate act), so a gate cannot silently disappear.
#
# Run from the workspace root: crates/speakerkit/tests/der_gate_inventory.sh
# Kept a shell script (not a `cargo test`) on purpose: it must shell out to
# `cargo`, which cannot nest inside a `cargo test` run without deadlocking on
# the target-dir lock. Written for bash 3.2 (macOS default) — no associative
# arrays.
set -euo pipefail

# Verify one DER binary: $1 = test binary name, $2.. = expected gate test names.
check_bin() {
  bin="$1"
  shift
  echo "== ${bin} =="
  # `--list` prints one `NAME: test` line per test; stderr (compile noise)
  # dropped. A compile FAILURE still surfaces because `cargo` exits non-zero and
  # `set -e` is relaxed only for the captured substitution — an empty list below
  # then trips the hard-fail regardless.
  list="$(cargo test -p speakerkit --features dia --test "${bin}" -- --list 2>/dev/null || true)"
  count="$(printf '%s\n' "${list}" | grep -c ': test$' || true)"
  if [ "${count}" -eq 0 ]; then
    echo "  FAIL: 0 tests listed for ${bin} — it compiled to nothing."
    echo "        (missing --features dia, a broken #![cfg(feature = \"dia\")] gate, or a build error)"
    return 1
  fi
  echo "  ${count} tests listed"
  rc=0
  for name in "$@"; do
    if printf '%s\n' "${list}" | grep -q "^${name}: test$"; then
      echo "  ok:   ${name}"
    else
      echo "  FAIL: expected DER gate '${name}' is not in ${bin}'s test list"
      rc=1
    fi
  done
  return "${rc}"
}

fail=0

check_bin parity_e2e \
  fluidaudio_der_parity_vs_dia_ort_and_determinism \
  argmax_source_der_characterization \
  compute_unit_der_study_all_vs_cpuonly || fail=1

check_bin parity_shipping_der \
  shipping_int8_der_06_long_recording_3spk \
  shipping_int8_der_10_mrbeast_clean_water_7spk \
  shipping_int8_der_09_mrbeast_dollar_date_8spk_known_defect || fail=1

if [ "${fail}" -ne 0 ]; then
  echo "DER gate inventory FAILED — the gates above are not all compiled/present." >&2
  exit 1
fi
echo "DER gate inventory OK — every expected DER gate is compiled and listed."
