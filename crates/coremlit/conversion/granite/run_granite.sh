#!/bin/bash
# granite-embedding-97m-multilingual-r2 CoreML conversion pipeline — re-derives the
# text encoder deterministically from the OFFICIAL public checkpoint
# ibm-granite/granite-embedding-97m-multilingual-r2 @ 835ad14087e140460703cf0fae09f97d469d65c2
# (Apache-2.0, ungated). Toolchain: the conv/ phase venv (python 3.11, torch 2.6.0,
# transformers 5.14.0, sentence-transformers 5.6.0, coremltools 9.0, numpy 1.26.4).
# See scripts/_granite_common.py for the pins + the fail-closed SHA verify.
#
# "Re-derives" means re-derivable to the floors in scripts/verify_granite.py, NOT
# bit-reproducible: neither the CoreML compiler nor torch's fp32 reduction order is
# pinned by those versions.
#
# ALL paths come from the environment (no hardcoded absolute paths). Set:
#   GRANITE_CONV        base scratch dir with the .venv + src-model (required)
#   GRANITE_GOLDENS     committed goldens dir (crates/coremlit/tests/granite/fixtures/goldens)
#   GRANITE_MODELS_OUT  gitignored Models/embedkit-granite tree
# Optional: GRANITE_SRC_MODEL, GRANITE_STAGE, and GRANITE_MODEL_CARD — the model
# card to stage beside the bundles. The card belongs to the published artifact set,
# so step 5 FAILS rather than emit a manifest that is short one file; set
# GRANITE_MODEL_CARD unless the card is already staged under the bundle dir.
set -euo pipefail

: "${GRANITE_CONV:?set GRANITE_CONV to the scratch dir holding .venv and src-model}"
: "${GRANITE_GOLDENS:?set GRANITE_GOLDENS to crates/coremlit/tests/granite/fixtures/goldens}"
: "${GRANITE_MODELS_OUT:?set GRANITE_MODELS_OUT to the gitignored Models/embedkit-granite tree}"
# ABSOLUTE before anything uses them. A relative value beginning with `-` is read
# as an OPTION by mkdir/rm/cp — `cp: illegal option -- t` — and every absolute
# path starts with `/`, so none can be. This is a string rewrite, not a `cd`, so
# it works before the directories exist and has no failure mode of its own.
#
# main was protected here only by accident: step 2 ran `cd "$GRANITE_STAGE"`,
# which failed on such a value and stopped the run before step 3. Moving
# compilation into Python removed that guard, and absolutising the compiler's
# operands then let step 2 SUCCEED — so the first failure moved to `cp` in step 3,
# AFTER the artifact root had been cleared. Neither change was destructive alone.
abspath_of() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$PWD/$1" ;; esac; }
GRANITE_CONV="$(abspath_of "$GRANITE_CONV")"
GRANITE_MODELS_OUT="$(abspath_of "$GRANITE_MODELS_OUT")"
export GRANITE_CONV GRANITE_MODELS_OUT
export GRANITE_STAGE="$(abspath_of "${GRANITE_STAGE:-$GRANITE_CONV/granite/staging}")"
export TOKENIZERS_PARALLELISM=false
PY="$GRANITE_CONV/.venv/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$GRANITE_MODELS_OUT/granite-97m-multilingual-r2"
STEM=granite_97m_512

echo "== 1/6 driver faithfulness + trace -> fp16 (shipped) + fp32 (reference) mlpackages =="
"$PY" -u "$HERE/scripts/convert_granite.py"

echo "== 2/6 compile every mlpackage -> mlmodelc + record the compilation (run-bound) =="
"$PY" -u "$HERE/scripts/compile_granite.py"

echo "== 3/6 stage the shipped fp16 mlpackage + mlmodelc into the Models tree =="
# Transactional: copy into scratch beside the artifact root, validate the copies
# against their sources by digest, then promote by rename with rollback. Nothing
# in $ROOT is touched until the new bundles are known good, and the attestations
# move aside BEFORE the bundles, so no interruption leaves CHECKSUMS.sha256 or
# MANIFEST.json describing bytes that are gone. Step 6 regenerates both.
#
# The scope of "no interruption", stated rather than left to be assumed: this is a
# PROCESS-crash property. An exception, ENOSPC, EIO, a Ctrl-C or a SIGKILL at any
# point leaves the previous publication recoverable and never leaves an
# attestation over bytes that are gone, because a rename is atomic and the page
# cache outlives the process. It is NOT power-loss ordering — macOS fsync hands
# bytes to the drive and the drive may still write them out of order — so recovery
# from a power cut is to RE-RUN this pipeline, which regenerable artifacts can
# afford. Nor is it resumability: a run killed mid-promotion leaves its scratch
# beside $ROOT holding the only copy of what it had moved out, and the next run
# REFUSES while that scratch exists rather than sweep the remainder into a second
# one and delete it on success. Reassembly is by hand; the refusal prints where to
# look.
#
# One run at a time. Nothing locks $ROOT, and this pipeline already assumes a
# single run: steps 1 and 2 share $GRANITE_STAGE and would clobber each other long
# before step 3 is reached.
#
# This was four rm/cp lines. A preflight stopped a MISSING source from triggering
# the removals, but could not stop a FAILING copy: the first cp landing and the
# second hitting an unreadable file left the root stripped of both attestations
# and both prior bundles. Python, not shell, because the copy has to be staged,
# digest-validated against its source and rolled back as a unit — which reuses the
# recipe's own digest_tree and has no reasonable shell equivalent.
"$PY" -u "$HERE/scripts/stage_artifact.py"

# The goldens step REWRITES $GRANITE_GOLDENS/corpus.json + driver_crosscheck.json,
# which verify_granite.py then reads as its oracle and write_manifest.py records a
# digest of. It must therefore run BEFORE them: regenerating afterwards would leave
# a manifest whose numbers are labelled "vs the committed goldens" while describing
# comparisons against the previous pair. It runs only when GRANITE_REGEN_GOLDENS=1,
# because a plain artifact replay should not move the committed oracle at all.
# It publishes the crosscheck conversion already computed, so step 1 must precede it.
if [ "${GRANITE_REGEN_GOLDENS:-0}" = "1" ]; then
  echo "== 4/6 regenerate committed goldens (corpus.json + driver_crosscheck.json) =="
  "$PY" -u "$HERE/scripts/generate_goldens.py"
else
  echo "== 4/6 skipped: goldens regeneration (set GRANITE_REGEN_GOLDENS=1 to rewrite \$GRANITE_GOLDENS) =="
fi

echo "== 5/6 fail-closed verify matrix (I/O contract; fp32-vs-goldens >= 0.99999998; fp16 >= 0.99996 and 0 NaN on EVERY unit) =="
"$PY" -u "$HERE/scripts/verify_granite.py"

echo "== 6/6 CHECKSUMS.sha256 + MANIFEST.json (runs AFTER verify so verify_metrics.json is captured) =="
"$PY" -u "$HERE/scripts/write_manifest.py"

echo "granite conversion pipeline complete."
