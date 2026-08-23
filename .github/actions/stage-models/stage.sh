#!/usr/bin/env bash
#
# Stage the MODELS_LOCK tables belonging to ONE model kit.
#
# THIS FILE EXISTS SO THERE IS EXACTLY ONE MODELS_LOCK PARSER, and that is its
# entire justification. `.github/workflows/ci.yml`'s `model-tests` job hand-parses
# the lock inline to build its `hf download` commands; `coverage.yml` needs the
# identical staging for its per-kit legs. Copy-pasting that parser into a second
# workflow would recreate exactly the coupling the kit-tag rework removed — two
# parsers over one lock file, free to drift, with nothing failing when they do.
# So the parser moved HERE, `coverage.yml` calls it through the composite action
# beside this file, and ci.yml's inline copy is migrated onto it as a follow-up
# (it could not be touched in the same change: that job was under review).
#
# Until that migration lands the two DO coexist, and the coexistence is not left
# on trust: `plan_parity.sh` beside this file executes ci.yml's own inline parser
# with `hf` stubbed and diffs its emitted download plan against this script's,
# per kit, and fails on any disagreement. It is wired into coverage.yml as its
# own job. When ci.yml stops staging models inline, that script reports the
# migration complete and asks to be deleted.
#
# ------------------------------------------------------------------------------
# The lock contract this script implements (see MODELS_LOCK's own header):
#
#   ["owner/repo"]        one table per artifact repository
#   kit       = "..."     the model kit whose shard/leg downloads this table
#   include   = "..."     SPACE-SEPARATED glob list -> one `--include` flag each
#     -- or --
#   files     = "..."     space-separated exact filenames, passed positionally
#   revision  = "..."     the pinned commit SHA (or a branch, loudly flagged)
#   local-dir = "..."     where the bytes land
#
# Tables are selected by `kit`, IN LOCK ORDER. Order ACROSS kits is not
# load-bearing. Order WITHIN a kit is, whenever two of its tables share a
# `local-dir`: the last download wins every filename both publish, which for the
# `speaker` kit decides whether the shipping pipeline gets the fp16-guard-repaired
# graphs or the contract-identical pre-repair ones. This script preserves lock
# order for that reason; proving WHICH layer won is the caller's job (ci.yml's
# `Verify staged overlay ordering` step hashes both graphs).
#
# ------------------------------------------------------------------------------
# Modes (`--mode`):
#
#   download  resolve the plan, download it, then verify the staged tree.
#   verify    resolve the plan and verify the staged tree, downloading nothing.
#             This is the CACHE-HIT path: a restored cache means no download ran,
#             and a half-restored or truncated cache entry must fail here rather
#             than an hour later as a wall of "model not found" gate failures.
#   resolve   resolve the plan and emit `local-dirs` only. Runs BEFORE
#             `actions/cache`, so the cache's `path:` list comes from the lock
#             instead of being a second per-kit copy in the workflow matrix.
#   plan      print the `hf download` argv this script WOULD run, one per line,
#             and do nothing else. Consumed by `plan_parity.sh`.
#
# Every mode runs the lock tripwires, so a malformed or mis-tagged lock fails on
# the very first step of a leg rather than after a 748 MB download.

set -euo pipefail

kit=""
lock="MODELS_LOCK"
mode="download"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --kit)  kit="${2-}";  shift 2 ;;
    --lock) lock="${2-}"; shift 2 ;;
    --mode) mode="${2-}"; shift 2 ;;
    *) echo "::error::stage.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

case "$mode" in
  download|verify|resolve|plan) ;;
  *) echo "::error::stage.sh: unknown --mode '$mode' (want download|verify|resolve|plan)" >&2; exit 2 ;;
esac

if [ -z "$kit" ]; then
  echo "::error::stage.sh: --kit is required" >&2
  exit 2
fi
if [ ! -f "$lock" ]; then
  echo "::error::stage.sh: lock file '$lock' does not exist (run from the repository root, or pass --lock)" >&2
  exit 2
fi

# TRIPWIRE 1 — every table must carry a `kit`, or it can never be selected and
# MODELS_LOCK would document a download that nothing performs. Counting headers
# against `kit` fields catches an ADDED table and a table that LOST its tag;
# neither is visible to the per-table guards below, which only inspect tables
# that were actually selected. Zero headers means this parser stopped matching
# the lock's format at all, which is the same defect wearing a different mask.
table_count=$(sed -n 's/^\["\(.*\)"\]$/\1/p' "$lock" | grep -c . || true)
kit_count=$(grep -cE '^kit[[:space:]]*=' "$lock" || true)
if [ "$table_count" -eq 0 ] || [ "$table_count" -ne "$kit_count" ]; then
  echo "::error::$lock has $table_count table(s) but $kit_count 'kit' field(s) — every table must declare the kit whose leg downloads it, or it is silently unreachable. (Zero tables means this parser no longer matches the lock's format at all; a lock predating the kit tags reports 0 'kit' fields and must be rebased onto the kit-tagged one.)" >&2
  exit 1
fi

# One TAB-separated record per table of THIS kit, in lock order:
#   repo <TAB> selector-field <TAB> selector <TAB> revision <TAB> local-dir
#
# Still no TOML crate for a file of this fixed shape, matching ci.yml's inline
# parser and the hermetic `parse_lock` mirror in
# crates/coremlit/tests/whisper/models_lock.rs.
tables=$(awk -v want="$kit" '
  function flush() {
    if (repo != "" && tkit == want) {
      if (files != "") { field = "files"; sel = files }
      else             { field = "include"; sel = include }
      printf "%s\t%s\t%s\t%s\t%s\n", repo, field, sel, revision, localdir
    }
  }
  /^\["/ {
    flush()
    repo = $0; sub(/^\["/, "", repo); sub(/"\]$/, "", repo)
    tkit = ""; include = ""; files = ""; revision = ""; localdir = ""
    next
  }
  $1 == "kit" || $1 == "include" || $1 == "files" || $1 == "revision" || $1 == "local-dir" {
    value = $0
    sub(/^[^=]*=[ \t]*"/, "", value)
    sub(/"[ \t]*$/, "", value)
    if      ($1 == "kit")       tkit = value
    else if ($1 == "include")   include = value
    else if ($1 == "files")     files = value
    else if ($1 == "revision")  revision = value
    else                        localdir = value
  }
  END { flush() }
' "$lock")

# TRIPWIRE 2 — a leg whose kit matches no table would stage nothing and then run
# its gates against a bare checkout, reporting artifact-absent failures instead
# of "the lock and the matrix disagree". Unlike ci.yml's inline copy this fires
# on EVERY mode, so the cache-hit path (which performs no download) is covered
# too. The cross-leg half — a table whose kit NO leg consumes — cannot be seen
# from inside one leg and is pinned hermetically by
# `ci_shards_every_kit_in_the_lock` (crates/coremlit/tests/whisper/models_lock.rs).
if [ -z "$tables" ]; then
  echo "::error::$lock defines no table with kit = \"$kit\", but a \"$kit\" leg asked to stage one — add the table, or drop the leg" >&2
  exit 1
fi

# Build the plan first, validate it whole, and only then act on it. `plan` and
# `resolve` need exactly this and nothing more, and `download` gets its
# fail-before-you-fetch behaviour for free.
plan=()      # one fully-expanded `hf download` argv per table, newline-joined
dirs=()      # this kit's local-dirs, in lock order, deduplicated

while IFS=$'\t' read -r repo field selector revision localdir; do
  # Fail LOUD on any empty extraction: `hf download <repo>` with no selector
  # silently fetches the WHOLE repository, so a lock-format drift must stop here
  # rather than download something other than what the lock intends.
  for v in repo field selector revision localdir; do
    if [ -z "${!v}" ]; then
      echo "::error::$lock's \"$repo\" table (kit $kit) parsed to an empty '$v' — format drift? fix the lock or this parser" >&2
      exit 1
    fi
  done
  read -ra patterns <<< "$selector"
  # The emptiness loop above sees the RAW selector string; this sees what is
  # actually passed to `hf`. A value that is non-empty but expands to no word
  # would otherwise slip through and fetch the whole repository.
  if [ "${#patterns[@]}" -eq 0 ]; then
    echo "::error::$lock's \"$repo\" $field selector ($selector) expanded to no pattern; hf would fetch the whole repository" >&2
    exit 1
  fi
  args=()
  if [ "$field" = "files" ]; then
    args=("${patterns[@]}")
  else
    # `hf` 1.19 accepts `--include` repeatedly (verified against 1.19.0), which
    # is what lets a selector be a pattern LIST. That matters for the speakerkit
    # overlay, whose obvious single glob `*.mlmodelc/*` would drag in a
    # NOT-adopted int8 re-conversion over the base layer's bytes. Nothing pins
    # `hf`'s version, so a future release COULD stop accepting the repeated
    # flag — fail-closed either way: an unknown-option error stops the download,
    # and a last-flag-wins regression stages too little, which the verify pass
    # below and ci.yml's checksum/overlay steps refuse.
    for pattern in "${patterns[@]}"; do args+=(--include "$pattern"); done
  fi
  argv=("$repo" "${args[@]}" --revision "$revision" --local-dir "$localdir")
  # The plan is compared token-by-token against ci.yml's by `plan_parity.sh`,
  # which joins on spaces. An argument CONTAINING whitespace would make that
  # comparison ambiguous and could hide a real disagreement, so refuse it here
  # rather than let the parity check quietly weaken. No selector pattern,
  # revision or path in this lock has ever contained a space, and none can:
  # `include`/`files` values are space-SEPARATED lists by definition.
  for token in "${argv[@]}"; do
    case "$token" in
      *[[:space:]]*)
        echo "::error::$lock's \"$repo\" table (kit $kit) yields the argument '$token', which contains whitespace — hf would take it as one word but the plan-parity diff joins on spaces, so a drift could hide there. Fix the lock." >&2
        exit 1
        ;;
    esac
  done

  echo "MODELS_LOCK[$kit]: $repo@$revision $field=\"$selector\" -> $localdir" >&2
  plan+=("${argv[*]}")

  seen=0
  for d in ${dirs[@]+"${dirs[@]}"}; do
    if [ "$d" = "$localdir" ]; then
      seen=1
      break
    fi
  done
  if [ "$seen" -eq 0 ]; then
    dirs+=("$localdir")
  fi
done <<< "$tables"   # a herestring, never `echo | while`: the loop must run in
                     # THIS shell so a failure inside it exits the script.

if [ "$mode" = "plan" ]; then
  # stdout carries the plan and NOTHING else — the provenance lines above went
  # to stderr precisely so this stays machine-comparable.
  printf 'hf download %s\n' "${plan[@]}"
  exit 0
fi

# `local-dirs` is what the caller feeds to `actions/cache`'s `path:`, so the
# cache list is DERIVED from the lock instead of being a second per-kit copy
# maintained by hand in a workflow matrix.
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "local-dirs<<COREMLIT_STAGE_MODELS_EOF"
    printf '%s\n' "${dirs[@]}"
    echo "COREMLIT_STAGE_MODELS_EOF"
  } >> "$GITHUB_OUTPUT"
fi
printf 'staged directories for kit %s: %s\n' "$kit" "${dirs[*]}"

if [ "$mode" = "resolve" ]; then
  exit 0
fi

if [ "$mode" = "download" ]; then
  # GitHub's macOS runners ship no `hf`; a developer running this script by hand
  # usually has one. Installing only when it is missing keeps both working and
  # keeps the local path offline-clean.
  if ! command -v hf >/dev/null 2>&1; then
    pipx install "huggingface_hub[cli]"
  fi
  for argv in "${plan[@]}"; do
    # Splitting the plan line back into an argv is exact rather than a guess:
    # every token was refused above if it contained whitespace.
    read -ra download_args <<< "$argv"
    hf download "${download_args[@]}" </dev/null
  done
fi

# Verify the staged tree on BOTH the download and the cache-hit path. A
# truncated download and a half-restored cache entry look identical from here
# and must both fail HERE, where the cause is one line, rather than surfacing an
# hour later as a wall of model-load failures inside the gates.
for d in "${dirs[@]}"; do
  if [ ! -d "$d" ]; then
    echo "::error::kit '$kit' stages $d, but that directory does not exist after $mode — the gates would have nothing to run against" >&2
    exit 1
  fi
  # A directory that EXISTS but holds nothing — an empty cache entry, an
  # `hf download` whose selector matched no file — must fail too, rather than
  # pass for having been created. Globbed rather than `find -quit`ed because
  # `-quit` is not in every BSD find, and rather than `ls | head` because that
  # closes a pipe under a writer (the SIGPIPE lesson from
  # crates/coremlit/tests/fp16_sweep_inventory.sh).
  shopt -s nullglob dotglob
  entries=("$d"/*)
  shopt -u nullglob dotglob
  if [ "${#entries[@]}" -eq 0 ]; then
    echo "::error::kit '$kit' stages $d, but that directory is EMPTY after $mode — the download matched no file, or the restored cache entry is empty" >&2
    exit 1
  fi
done
echo "kit '$kit': ${#dirs[@]} staged directory/directories present and non-empty"
