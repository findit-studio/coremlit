#!/usr/bin/env bash
#
# Prove that ci.yml's INLINE MODELS_LOCK parser and stage.sh's shared one stage
# byte-identical trees — by running both and diffing the plans, not by reading
# them side by side.
#
# WHY THIS EXISTS. `coverage.yml` needs the same per-kit model staging that
# `ci.yml`'s `model-tests` job performs. Copy-pasting that job's inline parser
# into a second workflow would recreate exactly the coupling the kit-tag rework
# removed: two hand-written parsers over one lock file, free to drift, with
# nothing failing when they do. So the parser moved into `stage.sh` and
# `coverage.yml` calls it. ci.yml could not be migrated in the same change (that
# job was under review), so for one release the two DO coexist — and this script
# is what makes that coexistence checkable instead of hopeful.
#
# HOW. It extracts ci.yml's download step's `run:` script, puts a stub `hf` (and
# `pipx`) first on PATH, executes the REAL script once per kit with that kit's
# `KIT` env, and captures the `hf download` argv it would have issued. It then
# runs `stage.sh --mode plan` for the same kit and diffs. There is no third
# re-implementation to keep in step: the thing under test is ci.yml's own code.
#
# SELF-DELETING. Once ci.yml calls the composite action instead of staging models
# inline, there is no `hf download` in it to extract, and this script reports the
# migration complete and asks to be removed along with itself.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

ci=".github/workflows/ci.yml"
lock="MODELS_LOCK"
stage=".github/actions/stage-models/stage.sh"

for f in "$ci" "$lock" "$stage"; do
  if [ ! -f "$f" ]; then
    echo "::error::plan_parity.sh: $f is missing" >&2
    exit 2
  fi
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# ---------------------------------------------------------------------------
# 1. Extract ci.yml's inline download script.
# ---------------------------------------------------------------------------
extracted=0
python3 - "$ci" "$work/ci_download.sh" <<'PY' || extracted=$?
import re, sys, yaml

ci_path, out_path = sys.argv[1], sys.argv[2]
with open(ci_path, encoding="utf-8") as fh:
    doc = yaml.safe_load(fh)

# Anchored to the start of a line so a step only counts when it RUNS
# `hf download`. The loose substring test matched ci.yml's own prose too — the
# comments above the download loop, and the `::error::` text that names the
# command when the speakerkit overlay loses its ordering — and reported two
# staging steps where there is one.
invokes = re.compile(r"^[ \t]*hf[ \t]+download\b", re.MULTILINE)

scripts = []
for job in (doc.get("jobs") or {}).values():
    for step in job.get("steps") or []:
        run = step.get("run")
        if isinstance(run, str) and invokes.search(run):
            scripts.append(run)

if not scripts:
    # No inline staging left anywhere in ci.yml: the migration onto the shared
    # action is complete and this whole script is now dead weight.
    sys.exit(3)
if len(scripts) > 1:
    sys.stderr.write(
        f"::error::{ci_path} has {len(scripts)} steps issuing `hf download`; "
        "this check knows how to run exactly one. Consolidate them, or finish "
        "migrating ci.yml onto .github/actions/stage-models.\n"
    )
    sys.exit(4)

script = scripts[0]
# `${{ ... }}` is spliced in by the Actions runner before bash sees the script,
# so a script that depends on it cannot be executed faithfully here and this
# check would silently compare something ci.yml never runs. ci.yml's download
# step passes its matrix values through `env:` for exactly this reason.
if "${{" in script:
    sys.stderr.write(
        "::error::ci.yml's download script contains a `${{ }}` expression, which "
        "this check cannot expand — so it could not run the real script and any "
        "'plans match' verdict would be a lie. Move the value into the step's "
        "`env:` block.\n"
    )
    sys.exit(4)

with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(script)
PY

if [ "$extracted" -eq 3 ]; then
  echo "ci.yml issues no \`hf download\` of its own: it has been migrated onto"
  echo ".github/actions/stage-models, so there is no second parser left to drift."
  echo "DELETE this script and the coverage.yml job that runs it."
  exit 0
fi
if [ "$extracted" -ne 0 ]; then
  # 4 is a diagnosed refusal (the python side already said why); anything else is
  # a crash in the extractor. Neither may be reported as "the plans match".
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Which kits to compare — taken from the lock, so neither side chooses them.
# ---------------------------------------------------------------------------
kits=$(sed -nE 's/^kit[[:space:]]*=[[:space:]]*"(.*)"[[:space:]]*$/\1/p' "$lock" | sort -u)
if [ -z "$kits" ]; then
  echo "::error::$lock declares no \`kit\` field, so neither parser can select by kit. This branch's coverage workflow requires the kit-tagged lock; rebase onto the change that introduces it." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. Stub `hf` and `pipx`. `hf` echoes its argv in the exact shape stage.sh's
#    `--mode plan` prints, so the two are directly diffable; `pipx` is a no-op
#    because this check must not touch the network.
# ---------------------------------------------------------------------------
mkdir -p "$work/bin"
cat > "$work/bin/hf" <<'SH'
#!/bin/sh
echo "hf $*"
SH
cat > "$work/bin/pipx" <<'SH'
#!/bin/sh
exit 0
SH
chmod +x "$work/bin/hf" "$work/bin/pipx"

status=0
for kit in $kits; do
  # ci.yml's own script, unmodified, with the network stubbed out.
  if ! PATH="$work/bin:$PATH" KIT="$kit" bash "$work/ci_download.sh" \
        > "$work/ci.$kit.out" 2> "$work/ci.$kit.err"; then
    echo "::error::ci.yml's download script failed for kit '$kit':" >&2
    sed 's/^/    /' "$work/ci.$kit.err" >&2
    status=1
    continue
  fi
  grep '^hf download ' "$work/ci.$kit.out" > "$work/ci.$kit.plan" || true

  if ! "$stage" --kit "$kit" --mode plan \
        > "$work/shared.$kit.plan" 2> "$work/shared.$kit.err"; then
    echo "::error::stage.sh failed for kit '$kit':" >&2
    sed 's/^/    /' "$work/shared.$kit.err" >&2
    status=1
    continue
  fi

  if diff -u "$work/ci.$kit.plan" "$work/shared.$kit.plan" > "$work/diff.$kit"; then
    lines=$(grep -c . "$work/shared.$kit.plan" || true)
    echo "== $kit: $lines download(s), ci.yml and stage.sh agree =="
  else
    echo "::error::kit '$kit': ci.yml's inline MODELS_LOCK parser and .github/actions/stage-models/stage.sh disagree on what to download. They stage the same tree today only by accident; fix whichever is wrong, or finish migrating ci.yml onto the action. (-) is ci.yml, (+) is the shared action:" >&2
    sed 's/^/    /' "$work/diff.$kit" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  exit 1
fi
echo "every kit in $lock stages identically through both parsers"
