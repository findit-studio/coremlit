#!/usr/bin/env bash
# F1 gate-inventory check (codex r6): prove the fp16 graph sweep RUNS in an
# ordinary `cargo test` when the models are present — i.e. it is NOT #[ignore]d
# then, so a plain `cargo test -p coremlit --features whisper` executes it. This is the inverse of
# der_gate_inventory.sh: there every load-bearing gate must be present AND still
# #[ignore]d; here the sweep must be present AND, with models on disk, NOT
# ignored.
#
# The defect this guards: build.rs emits `cfg(models_present)` when `Models/` is
# on disk, which UN-ignores `every_shipped_model_graph_survives_fp16`. But CI's
# model job ran `cargo test -p coremlit --features whisper -- --ignored` — the ignored-ONLY filter
# (libtest `RunIgnored::Only`) — so the sweep was excluded EXACTLY when the
# models were present, while the modelless `check` job skipped it too (ignored
# there). A newly vanishing fp16 guard would merge green. The fix wires the
# model job to run the coremlit sweep binary's ordinary suite; THIS inventory
# proves the libtest selection that makes that wiring correct — that with models
# present the sweep is ordinary, not ignored, so the plain `cargo test` reaches
# it.
#
# REQUIRES `Models/` present (any subtree): build.rs only un-ignores the sweep
# then. Run it on the model job or a dev machine that has the models, NOT the
# modelless `check` job — there the sweep is legitimately #[ignore]d and this
# would (correctly) fail. Kept a shell script, not a `cargo test`, because it
# shells out to `cargo`, which cannot nest inside a `cargo test` run without
# deadlocking on the target-dir lock (same reason as der_gate_inventory.sh).
# Written for bash 3.2 (macOS default).
set -euo pipefail

BIN=fp16_guards
SWEEP=every_shipped_model_graph_survives_fp16

echo "== ${BIN} :: ${SWEEP} =="

# Full `--list`: one `NAME: test` line per test (ignored or not); stderr
# (compile noise) dropped. Non-vacuity + presence. A compile FAILURE still
# surfaces because `cargo` exits non-zero and the empty-list guard below trips.
all="$(cargo test -p coremlit --features whisper --test "${BIN}" -- --list 2>/dev/null || true)"
count="$(printf '%s\n' "${all}" | grep -c ': test$' || true)"
if [ "${count}" -eq 0 ]; then
  echo "  FAIL: 0 tests listed for ${BIN} — it compiled to nothing (a build error?)."
  exit 1
fi

# `--list --ignored`: the SAME `NAME: test` shape, restricted to ignored tests.
ignored="$(cargo test -p coremlit --features whisper --test "${BIN}" -- --list --ignored 2>/dev/null || true)"
ignored_count="$(printf '%s\n' "${ignored}" | grep -c ': test$' || true)"
echo "  ${count} tests listed (${ignored_count} ignored)"

# The sweep must be PRESENT in the full list (not deleted or renamed)...
#
# Matched with a HERESTRING, never `printf ... | grep -q`, which is a SIGPIPE
# BUG that made this gate fail ~4 runs in 5.
#
# `grep -q` exits on its FIRST match and closes the read end of the pipe. Bash's
# `printf` builtin writes through a 512-byte stdio buffer (measured on macOS
# bash 3.2), so any payload over 512 bytes costs it TWO OR MORE `write`s — and
# if `grep` matches in an early chunk and exits before the later one lands,
# `printf` takes SIGPIPE (141). `set -o pipefail` then promotes 141 to the
# pipeline's status, so the MATCHING case — the healthy one — reports failure.
#
# Measured here, needle in the first chunk, 40 runs per size: <=512 bytes gives
# rc=141 zero times; 513 bytes gives it 37 times; 4 KB gives it 40 times. Put
# the needle in the LAST chunk instead and it drops back to zero, because grep
# must then consume everything. So the trigger is payload size AND match
# position, with scheduling deciding the rest — the real `--list` here is
# ~1.3 KB with an early match, which measured 33/40 failures. Note the relevant
# buffer is stdio's 512 bytes, NOT the 64 KB pipe buffer: this fires far below
# the size at which a pipe could ever block.
#
# The ignored-only list below is under the threshold TODAY and would cross it
# as tests are added, so do not reason about sizes — remove the writer. A
# herestring feeds grep from a temp file: nothing can take SIGPIPE, and the
# status is grep's alone. Keep every membership test below in this form.
if ! grep -q "^${SWEEP}: test$" <<<"${all}"; then
  echo "  FAIL: sweep '${SWEEP}' is not in ${BIN}'s test list (deleted or renamed)."
  exit 1
fi

# ...and, with `Models/` present, ABSENT from the ignored-only list — proving a
# plain `cargo test` (no --ignored) runs it. If it appears here, either `Models/`
# is absent (run this only WITH the models) or build.rs no longer emits
# `cfg(models_present)`; both mean the model job's ordinary suite would silently
# stop executing the sweep exactly when there is something to sweep.
if grep -q "^${SWEEP}: test$" <<<"${ignored}"; then
  echo "  FAIL: sweep '${SWEEP}' is #[ignore]d — Models/ is absent, or build.rs no longer"
  echo "        emits cfg(models_present). A plain \`cargo test\` would NOT run the sweep, so"
  echo "        the model job's ordinary suite would skip it exactly when models are present."
  exit 1
fi

echo "  ok:   ${SWEEP} is present and NOT ignored (models present) — a plain \`cargo test -p coremlit --features whisper\` runs it."
echo "fp16 sweep inventory OK — the graph sweep executes in the ordinary suite when Models/ is present."
