#!/usr/bin/env bash
# F1 gate-inventory check (codex r6): prove the fp16 graph sweep RUNS in an
# ordinary `cargo test` when the models are present — i.e. it is NOT #[ignore]d
# then, so a plain `cargo test -p coremlit` executes it. This is the inverse of
# der_gate_inventory.sh: there every load-bearing gate must be present AND still
# #[ignore]d; here the sweep must be present AND, with models on disk, NOT
# ignored.
#
# The defect this guards: build.rs emits `cfg(models_present)` when `Models/` is
# on disk, which UN-ignores `every_shipped_model_graph_survives_fp16`. But CI's
# model job ran `cargo test -p coremlit -- --ignored` — the ignored-ONLY filter
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
all="$(cargo test -p coremlit --test "${BIN}" -- --list 2>/dev/null || true)"
count="$(printf '%s\n' "${all}" | grep -c ': test$' || true)"
if [ "${count}" -eq 0 ]; then
  echo "  FAIL: 0 tests listed for ${BIN} — it compiled to nothing (a build error?)."
  exit 1
fi

# `--list --ignored`: the SAME `NAME: test` shape, restricted to ignored tests.
ignored="$(cargo test -p coremlit --test "${BIN}" -- --list --ignored 2>/dev/null || true)"
ignored_count="$(printf '%s\n' "${ignored}" | grep -c ': test$' || true)"
echo "  ${count} tests listed (${ignored_count} ignored)"

# The sweep must be PRESENT in the full list (not deleted or renamed)...
if ! printf '%s\n' "${all}" | grep -q "^${SWEEP}: test$"; then
  echo "  FAIL: sweep '${SWEEP}' is not in ${BIN}'s test list (deleted or renamed)."
  exit 1
fi

# ...and, with `Models/` present, ABSENT from the ignored-only list — proving a
# plain `cargo test` (no --ignored) runs it. If it appears here, either `Models/`
# is absent (run this only WITH the models) or build.rs no longer emits
# `cfg(models_present)`; both mean the model job's ordinary suite would silently
# stop executing the sweep exactly when there is something to sweep.
if printf '%s\n' "${ignored}" | grep -q "^${SWEEP}: test$"; then
  echo "  FAIL: sweep '${SWEEP}' is #[ignore]d — Models/ is absent, or build.rs no longer"
  echo "        emits cfg(models_present). A plain \`cargo test\` would NOT run the sweep, so"
  echo "        the model job's ordinary suite would skip it exactly when models are present."
  exit 1
fi

echo "  ok:   ${SWEEP} is present and NOT ignored (models present) — a plain \`cargo test -p coremlit\` runs it."
echo "fp16 sweep inventory OK — the graph sweep executes in the ordinary suite when Models/ is present."
