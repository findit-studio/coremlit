#!/usr/bin/env bash
# F1 gate-inventory check: prove the end-to-end DER gates are actually COMPILED
# and still #[ignore]d — not silently feature-gated out, deleted, or un-ignored.
#
# The DER binaries (`tests/parity_e2e.rs`, `tests/parity_shipping_der.rs`) are
# `#![cfg(feature = "dia")]`. Without `--features dia` they compile to nothing,
# so `cargo test -p speakerkit -- --ignored` reports a green sweep containing
# ZERO DER tests. Every load-bearing gate here is ALSO `#[ignore]`d (each needs
# the gitignored `Models/` tree plus the sibling `diarization` fixtures), and
# the README drives them with `cargo test ... -- --ignored`.
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
# Run from the workspace root: crates/speakerkit/tests/der_gate_inventory.sh
# Kept a shell script (not a `cargo test`) on purpose: it must shell out to
# `cargo`, which cannot nest inside a `cargo test` run without deadlocking on
# the target-dir lock. Written for bash 3.2 (macOS default) — no associative
# arrays.
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
  all="$(cargo test -p speakerkit --features dia --test "${bin}" -- --list 2>/dev/null || true)"
  count="$(printf '%s\n' "${all}" | grep -c ': test$' || true)"
  if [ "${count}" -eq 0 ]; then
    echo "  FAIL: 0 tests listed for ${bin} — it compiled to nothing."
    echo "        (missing --features dia, a broken #![cfg(feature = \"dia\")] gate, or a build error)"
    return 1
  fi
  # `--list --ignored`: the SAME `NAME: test` shape, but restricted to ignored
  # tests. This is what distinguishes an ignored gate from an un-ignored one.
  ignored="$(cargo test -p speakerkit --features dia --test "${bin}" -- --list --ignored 2>/dev/null || true)"
  ignored_count="$(printf '%s\n' "${ignored}" | grep -c ': test$' || true)"
  echo "  ${count} tests listed (${ignored_count} ignored)"
  rc=0
  for name in "$@"; do
    if printf '%s\n' "${ignored}" | grep -q "^${name}: test$"; then
      echo "  ok:   ${name} (present + ignored)"
    elif printf '%s\n' "${all}" | grep -q "^${name}: test$"; then
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

# parity_shipping_der.rs — ALL FOUR shipping-int8 DER clips (06, 14, 10, 09),
# plus the shipping-default resolver gate, the corpus-selection gate, and the
# clip-09 audio-content pin. Clip 14 and the resolver/corpus/content-pin gates
# were previously unlisted.
check_bin parity_shipping_der \
  shipping_int8_der_06_long_recording_3spk \
  shipping_int8_der_14_mrbeast_strongman_robot_4spk \
  shipping_int8_der_10_mrbeast_clean_water_7spk \
  shipping_int8_der_09_mrbeast_dollar_date_8spk_known_defect \
  shipping_default_is_the_int8_embedder \
  shipping_clip_selection_is_the_documented_subset \
  clip09_content_pin_catches_an_audio_swap || fail=1

if [ "${fail}" -ne 0 ]; then
  echo "DER gate inventory FAILED — the gates above are not all compiled, present, and #[ignore]d." >&2
  exit 1
fi
echo "DER gate inventory OK — every expected DER gate is compiled, listed, and still ignored."
