#!/usr/bin/env bash
#
# Prove that ci.yml's `model-tests` shards and coverage.yml's `coverage-models`
# legs RUN THE SAME MODEL GATES — by executing both gate runners and diffing the
# cargo invocations they would issue, not by reading their matrices side by side.
#
# WHY THIS EXISTS. coverage.yml already says the coupling out loud: "the gate
# plans are the same plans — because a coverage leg that measured a DIFFERENT
# set of gates than the job that enforces them would report on code the
# repository does not actually gate." That was asserted in a comment and checked
# by nothing. `plan_parity.sh` next door covers the DOWNLOAD plan; the two files
# were free to disagree about which gates run over what they downloaded, and
# they did: the `speaker` leg went a whole release without the `@lib` group its
# ci.yml twin carries (#61).
#
# HOW, and it is plan_parity.sh's method one level up. Each workflow's gate
# runner is extracted from the file, a stub `cargo` is put first on PATH, and
# the REAL script is executed once per kit with that kit's own `GATES` value.
# The stub canonicalises each invocation and appends it to a log; the logs are
# then diffed per kit. There is no third re-implementation of the
# `features|selectors|filter` grammar to keep in step — the things under test
# are the two runners' own code — so a refactor that changes their syntax while
# preserving the plan stays green, and one that changes the plan cannot.
#
# THE THREE ALLOWED DIFFERENCES, encoded rather than tolerated. The stub knows
# exactly this much about how the two sides may differ, and ANY other token is a
# hard failure that names itself rather than being normalised away:
#
#   1. the subcommand: ci.yml runs `cargo test`, coverage.yml `cargo llvm-cov`;
#   2. `--no-report`, which coverage.yml passes so the per-group runs accumulate
#      into one profile instead of each writing a report;
#   3. `cargo llvm-cov clean` / `cargo llvm-cov report`, coverage-only commands
#      that bracket the gates rather than being gates. They are counted and
#      reported, so they cannot hide a gate by being mistaken for one.
#
# Everything else — features, `--lib`/`--test <name>`/whole-package selection,
# `--no-fail-fast`, and every argument after `--` including `--ignored`, the
# plan's filter, and the `--list` half of the anti-vacuum count — must match
# exactly, in order.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

ci=".github/workflows/ci.yml"
cov=".github/workflows/coverage.yml"

for f in "$ci" "$cov"; do
  if [ ! -f "$f" ]; then
    echo "::error::gate_plan_parity.sh: $f is missing" >&2
    exit 2
  fi
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# ---------------------------------------------------------------------------
# 1. Extract each workflow's gate runner and the matrix it is driven by.
#
#    The runner is found by what it IS — the step whose `env:` binds `GATES`,
#    i.e. the one driven by the matrix plan — rather than by step name or job
#    id, and its matrix is read from the job that CONTAINS it. Renaming either
#    survives; losing either is a diagnosed failure, not a vacuous pass.
#
#    `GATES` rather than the `--list --ignored` text alone, because ci.yml has a
#    SECOND ignored-only counter: the modelless `check` job's VAD block, whose
#    targets are hardcoded because the committed model needs no matrix. It runs
#    gates but is not a gate PLAN, and coverage.yml has no counterpart to it.
# ---------------------------------------------------------------------------
extract() {
  python3 - "$1" "$2" <<'PY'
import re, sys, yaml

path, outdir = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    doc = yaml.safe_load(fh)

runs_gates = re.compile(r"--\s+--list\s+--ignored", re.MULTILINE)

found = []
for job_id, job in (doc.get("jobs") or {}).items():
    for step in job.get("steps") or []:
        run = step.get("run")
        env = step.get("env") or {}
        if isinstance(run, str) and runs_gates.search(run) and "GATES" in env:
            found.append((job_id, job, run))

if len(found) != 1:
    sys.stderr.write(
        f"::error::{path} has {len(found)} plan-driven gate runners (a step that "
        "binds `GATES` and takes an ignored-only `--list` count); this check knows "
        "how to execute exactly one. None means the workflow has lost its gate "
        "runner, and its anti-vacuum guard with it; several have to say which is "
        "the plan.\n"
    )
    sys.exit(4)

job_id, job, script = found[0]

# `${{ }}` is spliced in by the Actions runner before bash sees the script, so a
# script that depends on it cannot be executed faithfully here and any "the
# plans match" verdict would be a lie. Both runners pass their matrix values
# through `env:` for exactly this reason.
if "${{" in script:
    sys.stderr.write(
        f"::error::{path}'s gate runner contains a `${{{{ }}}}` expression, which this "
        "check cannot expand — so it could not run the real script. Move the value "
        "into the step's `env:` block.\n"
    )
    sys.exit(4)

rows = (((job.get("strategy") or {}).get("matrix") or {}).get("include")) or []
plans = {}
for row in rows:
    kit, gates = row.get("kit"), row.get("gates")
    if not kit or not gates:
        sys.stderr.write(
            f"::error::{path}'s {job_id!r} matrix has a row with no `kit` or no "
            f"`gates` ({row.get('kit')!r}); every row must carry a gate plan or this "
            "check cannot compare it.\n"
        )
        sys.exit(4)
    plans[kit] = gates
if not plans:
    sys.stderr.write(f"::error::{path}'s {job_id!r} job has no gate-plan matrix rows\n")
    sys.exit(4)

with open(f"{outdir}/runner.sh", "w", encoding="utf-8") as fh:
    fh.write(script)
with open(f"{outdir}/kits", "w", encoding="utf-8") as fh:
    fh.write("\n".join(sorted(plans)) + "\n")
for kit, gates in plans.items():
    with open(f"{outdir}/gates.{kit}", "w", encoding="utf-8") as fh:
        fh.write(gates)
print(f"{path}: job {job_id!r}, {len(plans)} kit(s)")
PY
}

mkdir -p "$work/ci" "$work/cov"
extract "$ci" "$work/ci"
extract "$cov" "$work/cov"

# ---------------------------------------------------------------------------
# 2. The kit sets must match before the plans can be compared at all: a kit with
#    a shard and no coverage leg is code CI gates and the number never sees, and
#    the reverse is a leg reporting on gates nothing enforces.
# ---------------------------------------------------------------------------
if ! diff -u "$work/ci/kits" "$work/cov/kits" > "$work/kits.diff"; then
  echo "::error::ci.yml's model-gate shards and coverage.yml's legs cover different kits. A kit gated with no leg is code the number never sees; a leg with no shard reports on gates nothing enforces. (-) is ci.yml, (+) is coverage.yml:" >&2
  sed 's/^/    /' "$work/kits.diff" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. The stub `cargo`. It canonicalises one invocation into one log line, so the
#    two runners' argv become directly diffable despite the wrapper.
#
#    It must also keep the REAL script running: both runners pipe the `--list`
#    invocation into `grep -c ': test$'` and abort the whole plan when that
#    count is zero (the anti-vacuum guard). So the stub prints one line in that
#    shape — enough to satisfy the guard, and the count itself is not what this
#    check compares.
# ---------------------------------------------------------------------------
mkdir -p "$work/bin"
cat > "$work/bin/cargo" <<'PY'
#!/usr/bin/env python3
import os, sys

argv = sys.argv[1:]
log = os.environ["GATE_PARITY_LOG"]


def emit(line):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def bail(reason):
    # An unknown token must NOT be normalised away: that is how a parity check
    # gets loosened until it passes. Record it and let the differ fail naming it.
    emit(f"UNKNOWN {reason}: {' '.join(argv)}")
    sys.exit(0)


if not argv:
    bail("no subcommand")

sub, rest = argv[0], argv[1:]

# ALLOWED DIFFERENCE 1: the subcommand.
if sub == "test":
    pass
elif sub == "llvm-cov":
    # ALLOWED DIFFERENCE 3: coverage-only commands that bracket the gates.
    if rest and rest[0] in ("clean", "report"):
        emit(f"NONGATE llvm-cov {rest[0]}")
        sys.exit(0)
    # ALLOWED DIFFERENCE 2: --no-report, so per-group runs accumulate.
    if rest and rest[0] == "--no-report":
        rest = rest[1:]
else:
    bail(f"unknown cargo subcommand {sub!r}")

# Everything from here must be identical between the two runners.
features = None
target = "package"
no_fail_fast = 0
tail = None
i = 0
while i < len(rest):
    tok = rest[i]
    if tok == "--":
        tail = rest[i + 1 :]
        break
    if tok == "-p":
        if i + 1 >= len(rest) or rest[i + 1] != "coremlit":
            bail("gates must run against -p coremlit")
        i += 2
        continue
    if tok == "--features":
        if i + 1 >= len(rest):
            bail("--features with no value")
        features = rest[i + 1]
        i += 2
        continue
    if tok == "--lib":
        target = "lib"
        i += 1
        continue
    if tok == "--test":
        if i + 1 >= len(rest):
            bail("--test with no value")
        target = f"test:{rest[i + 1]}"
        i += 2
        continue
    if tok == "--no-fail-fast":
        no_fail_fast = 1
        i += 1
        continue
    bail(f"unrecognised argument {tok!r}")

if tail is None:
    bail("no `--` separator, so this is not a gate invocation")
if features is None:
    bail("no --features, so the gate's feature set is unknowable")

mode = "list" if "--list" in tail else "run"
emit(
    f"GATE mode={mode} features={features!r} target={target} "
    f"no-fail-fast={no_fail_fast} tail={' '.join(tail)!r}"
)

# Keep the real runner alive past its anti-vacuum guard.
if mode == "list":
    print("stub::gate: test")
PY
chmod +x "$work/bin/cargo"

# ---------------------------------------------------------------------------
# 4. Execute each runner, per kit, and diff.
# ---------------------------------------------------------------------------
run_side() {
  side="$1" kit="$2"
  log="$work/$side.$kit.plan"
  : > "$log"
  # `PROBE` is ci.yml's absent-artifact guard, not part of the gate plan; `.`
  # always exists, so the real loop runs and passes. coverage.yml's runner does
  # not read it. Anything else the scripts need must come through `env:`, which
  # is the same constraint the `${{ }}` refusal above enforces.
  (
    cd "$work"
    PATH="$work/bin:$PATH" \
    GATE_PARITY_LOG="$log" \
    KIT="$kit" \
    PROBE="." \
    GATES="$(cat "$work/$side/gates.$kit")" \
      bash "$work/$side/runner.sh"
  ) > "$work/$side.$kit.out" 2> "$work/$side.$kit.err"
}

# A divergent kit reports ITS verdict and the loop keeps going. Under a global
# early-exit the first mismatch would hide every kit after it — the same masking
# ci.yml's own gate runner is written to avoid, and the reason this job reports
# all six plans rather than the first broken one.
status=0
while read -r kit; do
  [ -n "$kit" ] || continue
  bad=0
  for side in ci cov; do
    if ! run_side "$side" "$kit"; then
      echo "::error::the $side gate runner failed for kit '$kit', so its plan could not be resolved:" >&2
      sed 's/^/    /' "$work/$side.$kit.err" >&2
      bad=1
    fi
  done
  if [ "$bad" -ne 0 ]; then
    status=1
    continue
  fi

  if grep -q '^UNKNOWN ' "$work/ci.$kit.plan" "$work/cov.$kit.plan"; then
    echo "::error::kit '$kit': a gate invocation used an argument this check does not know how to compare. It is refused rather than ignored, because normalising an unknown token away is how a parity check stops checking. Teach .github/scripts/gate_plan_parity.sh the argument, or stop passing it:" >&2
    grep -h '^UNKNOWN ' "$work/ci.$kit.plan" "$work/cov.$kit.plan" | sed 's/^/    /' >&2
    status=1
    continue
  fi

  grep '^GATE ' "$work/ci.$kit.plan" > "$work/ci.$kit.gates" || true
  grep '^GATE ' "$work/cov.$kit.plan" > "$work/cov.$kit.gates" || true
  gates=$(grep -c . "$work/ci.$kit.gates" || true)
  if [ "$gates" -eq 0 ]; then
    echo "::error::kit '$kit': ci.yml's gate runner issued no gate invocation at all, so this comparison would pass over nothing" >&2
    status=1
    continue
  fi

  if diff -u "$work/ci.$kit.gates" "$work/cov.$kit.gates" > "$work/gates.$kit.diff"; then
    nongate=$(grep -c '^NONGATE ' "$work/cov.$kit.plan" || true)
    echo "== $kit: $gates gate invocation(s) identical, $nongate coverage-only command(s) =="
  else
    echo "::error::kit '$kit': ci.yml's model-tests shard and coverage.yml's coverage leg do not run the same model gates. coverage.yml's own comment says these are the same plans; a leg that measures a different set reports on code the repository does not actually gate. (-) is ci.yml, (+) is coverage.yml:" >&2
    sed 's/^/    /' "$work/gates.$kit.diff" >&2
    status=1
  fi
done < "$work/ci/kits"

if [ "$status" -ne 0 ]; then
  exit 1
fi
echo "every kit's gate plan resolves identically in $ci and $cov"
