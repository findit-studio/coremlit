#!/usr/bin/env bash
#
# Stage the MODELS_LOCK tables belonging to ONE model kit.
#
# THIS FILE EXISTS SO THERE IS EXACTLY ONE MODELS_LOCK PARSER, and that is its
# entire justification. `.github/workflows/ci.yml`'s `model-tests` shards and
# `coverage.yml`'s `coverage-models` legs need the identical per-kit staging, and
# two hand-written parsers over one lock file would be free to drift with nothing
# failing when they do — exactly the coupling the kit-tag rework removed. So the
# parser lives HERE and BOTH workflows call it through the composite action
# beside this file.
#
# ci.yml used to hand-parse the lock inline and build its own `hf download`
# loop; for one release the two coexisted under a parity check that diffed the
# argv both would issue. That migration is now complete, and completing it closed
# a hole rather than merely removing a copy: the inline loop was the producer of
# every cache entry ci.yml saved, and the per-table checks below run on the
# DOWNLOAD path only — so those trees had been verified table by table by
# nothing, and `verify` mode (all ci.yml then ran, and all coverage.yml runs on a
# cache hit) cannot see the difference. With both workflows downloading through
# this script, a cache entry can only ever be produced by a download that passed
# the per-table pass.
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
#             and do nothing else. Nothing in CI calls it; it is how a person
#             asks what a lock edit would fetch without fetching it.
#
# Every mode runs the lock tripwires, so a malformed or mis-tagged lock fails on
# the very first step of a leg rather than after a 748 MB download.
#
# ------------------------------------------------------------------------------
# Overrides, and why there are only two:
#
#   --lock <path>       the lock to read. It also relocates the manifest
#                       directory, which is `<dirname of the lock>/MODELS_LOCK.d`
#                       and never configured separately, and every `local-dir`
#                       is relative to the working directory — so a caller that
#                       runs this script from another root with its own lock
#                       stages, and verifies, entirely inside that root. That is
#                       all `verify_selftest.sh` beside this file needs; there is
#                       no models-root flag to keep in step.
#   --downloader <cmd>  run <cmd> instead of `hf download`, with the IDENTICAL
#                       argv. Nothing in CI passes it and the default path is
#                       byte-for-byte the one that shipped. It exists because the
#                       per-table manifest pass below runs on the DOWNLOAD path
#                       only — it is precisely the half `verify` cannot perform —
#                       so a self-test that could only run `verify` could not
#                       reach the check it exists to pin.

set -euo pipefail

kit=""
lock="MODELS_LOCK"
mode="download"
downloader=""   # empty is `hf download`; see the overrides box above

while [ "$#" -gt 0 ]; do
  case "$1" in
    --kit)  kit="${2-}";  shift 2 ;;
    --lock) lock="${2-}"; shift 2 ;;
    --mode) mode="${2-}"; shift 2 ;;
    --downloader) downloader="${2-}"; shift 2 ;;
    *) echo "::error::stage.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

if [ -n "$downloader" ] && [ ! -x "$downloader" ]; then
  echo "::error::stage.sh: --downloader '$downloader' is not an executable file" >&2
  exit 2
fi

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
# Still no TOML crate for a file of this fixed shape; the hermetic `parse_lock`
# in coremlit/tests/whisper/models_lock.rs mirrors this pass over the same lock.
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
# of "the lock and the matrix disagree". It fires on EVERY mode, so the cache-hit
# path (which performs no download, and where ci.yml's retired inline copy of
# this guard therefore never ran) is covered too. The cross-leg half — a table
# whose kit NO leg consumes — cannot be seen from inside one leg and is pinned
# hermetically by `ci_shards_every_kit_in_the_lock`
# (coremlit/tests/whisper/models_lock.rs).
if [ -z "$tables" ]; then
  echo "::error::$lock defines no table with kit = \"$kit\", but a \"$kit\" leg asked to stage one — add the table, or drop the leg" >&2
  exit 1
fi

# Build the plan first, validate it whole, and only then act on it. `plan` and
# `resolve` need exactly this and nothing more, and `download` gets its
# fail-before-you-fetch behaviour for free.
plan=()      # one fully-expanded `hf download` argv per table, newline-joined
records=()   # the same tables' lock records, INDEX-ALIGNED with `plan`, so the
             # download loop can verify each table against its own manifest
             # without re-parsing the lock into a second order
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
  # The plan is stored as ONE space-joined string per table and split back into
  # an argv by `read -ra` before the download runs, so an argument CONTAINING
  # whitespace would reach `hf` as two words rather than one. Refused here rather
  # than round-tripped wrongly. No selector pattern, revision or path in this lock
  # has ever contained a space, and none can: `include`/`files` values are
  # space-SEPARATED lists by definition.
  for token in "${argv[@]}"; do
    case "$token" in
      *[[:space:]]*)
        echo "::error::$lock's \"$repo\" table (kit $kit) yields the argument '$token', which contains whitespace — this script joins each table's argv on spaces and splits it back with \`read -ra\` before the download, so hf would receive that one argument as two. Fix the lock." >&2
        exit 1
        ;;
    esac
  done

  echo "MODELS_LOCK[$kit]: $repo@$revision $field=\"$selector\" -> $localdir" >&2
  plan+=("${argv[*]}")
  records+=("$repo"$'\t'"$field"$'\t'"$selector"$'\t'"$revision"$'\t'"$localdir")

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

# The download loop indexes `records` by `plan`'s index. They are appended
# together above, so this can only fire on an edit that breaks that — and a
# silent misalignment would verify one table's download against another table's
# manifest, which is worse than not verifying it.
if [ "${#plan[@]}" -ne "${#records[@]}" ]; then
  echo "::error::stage.sh: ${#plan[@]} plan line(s) but ${#records[@]} table record(s) — the two are built together and must stay index-aligned" >&2
  exit 1
fi

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

# ------------------------------------------------------------------------------
# THE MANIFEST MACHINERY, defined here because the DOWNLOAD needs it too.
#
# `MODELS_LOCK.d/<vendor_dir>@<revision>.sha256` holds one committed file list
# per globbed table. Everything below reads a manifest through
# `manifest_canonical`, which is the one place its grammar is enforced.
manifest_dir="$(dirname "$lock")/MODELS_LOCK.d"
[ "$(dirname "$lock")" = "." ] && manifest_dir="MODELS_LOCK.d"

# `*` crosses `/` in huggingface_hub's `--include`, so the ERE is the pattern
# with `.` escaped and `*` widened — the same semantics `glob_matches` in
# coremlit/tests/model_licences.rs implements, and the reason a `?` or a `[`
# is refused rather than guessed at.
pattern_to_ere() {
  case "$1" in
    *[\?\[\]\^\$\(\)\+\{\}\|\\]*)
      echo "::error::$lock selector pattern '$1' uses a regular-expression metacharacter this verifier does not implement; a silently-wrong match here would be a coverage hole, so it is refused rather than guessed at" >&2
      exit 1
      ;;
  esac
  # `.` is the only regex-special character these selectors contain, and `*`
  # is the only wildcard: escape the first, widen the second. `.*` rather than
  # a class, because `*` crosses `/` in huggingface_hub's matcher.
  printf '%s' "$1" | sed -e 's/\./\\./g' -e 's/\*/.*/g'
}

# ONE anchored alternation per table, into the global `ere`. `|` is refused
# inside a pattern (see pattern_to_ere), so joining on it cannot change what a
# pattern means, and a single ERE keeps this out of awk's `-v` newline
# restriction.
#
# IT SETS A VARIABLE RATHER THAN PRINTING ONE, and the failure of
# `pattern_to_ere` is tested rather than left to `set -e`. Both are the same
# lesson, learned by measuring: written as `ere="$(selector_ere "$selector")"`,
# `pattern_to_ere`'s `exit 1` for a refused metacharacter died in a nested
# command substitution, `set -e` did not carry it out of the assignment, and a
# selector containing `[` verified GREEN while matching nothing — the exact
# coverage hole that refusal exists to prevent. `verify_selftest.sh` pins it.
selector_ere() {
  local pattern part
  ere=""
  for pattern in $1; do
    [ -n "$ere" ] && ere="$ere|"
    if ! part="$(pattern_to_ere "$pattern")"; then
      exit 1
    fi
    ere="$ere$part"
  done
  ere="^($ere)$"
}

# A COMMITTED MANIFEST, VALIDATED AND CANONICALISED, from $1 into $2 (coremlit
# #147, finding 3).
#
# The grammar is 64 lowercase hex digits, two spaces, then a path that names
# ONE file UNDER the table's `local-dir`. At most one leading `./` is stripped:
# upstream's own lists carry one on every line and are committed verbatim.
# Every `/`-separated component that survives must be non-empty and must be
# neither `.` nor `..`.
#
# That rule is not decoration. This script hands a manifest path to `shasum`
# with the staged directory as the working directory, so `../sibling` would
# verify — and the licence register would enumerate as this table's contents —
# a file the table does not stage; `.`, `a/..` and a trailing `/` name a
# DIRECTORY, which has no digest; and `a//b`, `a/./b` and `././a` are further
# spellings of one path, which is how a manifest lists one file twice and reads
# as two entries. A repeated canonical path is refused for that reason even
# when the two digests agree.
#
# The identical rule is `table_relative_path` in
# coremlit/tests/model_licences.rs and in tests/support/models_lock_manifest.rs;
# one falsifier there drives both Rust copies over one case table.
manifest_canonical() {
  awk -v file="$1" '
    function fail(why) {
      printf "::error::%s line %d (%s) %s. Every line of a committed manifest is 64 lowercase hex digits, two spaces, then a path naming ONE file UNDER the table local-dir: this script hashes that path with the staged directory as its working directory, so anything else verifies bytes the table does not stage, or names something that has no digest at all. Regenerate the manifest with shasum -a 256 over the staged tree.\n", file, NR, $0, why > "/dev/stderr"
      failed = 1
      exit 1
    }
    {
      if (length($0) < 67) fail("is shorter than a digest, two spaces and a path")
      if (substr($0, 1, 64) !~ /^[0-9a-f]+$/) fail("does not begin with 64 lowercase hex digits")
      if (substr($0, 65, 2) != "  ") fail("does not put exactly two spaces between the digest and the path")
      path = substr($0, 67)
      sub(/^\.\//, "", path)
      if (path == "") fail("names an empty path")
      n = split(path, part, "/")
      for (i = 1; i <= n; i++) {
        if (part[i] == "")   fail("has an empty path component, from a leading, doubled or trailing slash")
        if (part[i] == ".")  fail("has a . component, which names a directory rather than a file")
        if (part[i] == "..") fail("has a .. component, so it can resolve outside the table local-dir")
      }
      if (path in seen) fail("repeats the path first listed on line " seen[path] "; one path holds one set of bytes, so a second line for it is a generator that ran twice or a merge that went wrong")
      seen[path] = NR
      print substr($0, 1, 64) "  " path
    }
    END {
      if (!failed && NR == 0) {
        printf "::error::%s lists no file. An empty manifest enumerates nothing, so every check that reads it would pass over the table it belongs to without looking at one byte.\n", file > "/dev/stderr"
        exit 1
      }
    }
  ' "$1" > "$2"
}

# EVERY FILE AN EXPLICIT `files` TABLE STAGES IS ON DISK (coremlit #147,
# finding 2).
#
# A `files` list names every file it stages, so the lock IS the enumeration and
# there is nothing for a manifest to add — that has not changed. What the lock
# could not do by itself was say the files ARRIVED, and nothing did: the staged
# tree checks above ask only whether the directory exists and is non-empty, so
# a cache entry holding ONE of Models/tokenizers/whisper-tiny's three files
# restored, verified clean, and handed the gates a tokenizer directory with no
# config.json. Presence is checked here, on both the download and the cache-hit
# path. `-f` is deliberate over `-e`: a directory and a dangling symlink are
# both failures.
#
# $4, when given, collects the listed paths for the reverse reconciliation, so
# a files-staged path that a sibling table's glob happens to match is not then
# reported as staged-but-unlisted.
verify_files_table() {
  local repo="$1" selector="$2" localdir="$3" listed_out="${4-}" path missing="" count=0
  for path in $selector; do
    count=$((count + 1))
    if [ ! -f "$localdir/$path" ]; then
      missing="$missing $path"
    fi
    if [ -n "$listed_out" ]; then
      printf '%s\n' "$path" >> "$listed_out"
    fi
  done
  if [ -n "$missing" ]; then
    echo "::error::$lock stages \"$repo\" into $localdir as an explicit file list, and after $mode$missing is not a regular file there. A files table's lock line IS its enumeration, so a missing entry is not a smaller download: it is a gate that fails an hour later with a model-not-found wall. Re-download the kit, or drop the file from the lock." >&2
    exit 1
  fi
  echo "MANIFEST[$kit]: $repo stages $count explicit file(s) into $localdir, all present"
}

# THE ONE PATH EVERY DIGEST CHECK SKIPS MUST STILL BE ON DISK (coremlit #147,
# round 2 finding 2).
#
# `CHECKSUMS.sha256` is excluded from the snapshots, from the manifest digest
# comparison and from the reverse reconciliation, and it has to be: a digest
# list cannot hold its own digest, and upstream's speakerkit copy proves it by
# listing its own name against the digest of an EMPTY file. The exclusion made
# the file's ABSENCE invisible. A cache entry holding every model file but not
# the checksum list verified GREEN — the byte-identity check below is the only
# thing that reads it, and that check begins with a `find`, so with no such file
# it silently compared nothing at all.
#
# Three of this lock's selectors name that file explicitly, so its absence is a
# short download or a truncated cache entry and never a legitimate tree. The
# paths demanded are exactly the ones that are KNOWABLY selected:
#
#   - a selector pattern containing no `*` names ONE exact path, so a pattern
#     whose basename is CHECKSUMS.sha256 must be a regular file (clapkit's,
#     redimnetkit's and the speakerkit overlay's selectors each carry one);
#   - a line of this table's own manifest that the selector matches and whose
#     basename is CHECKSUMS.sha256 must be a regular file too — the speakerkit
#     overlay's manifest carries such a line, and a wildcard selector could
#     reach one.
#
# WHAT IT STILL CANNOT SAY: a repository that ships a CHECKSUMS.sha256 which
# only a WILDCARD selector reaches and which no committed manifest lists (ced's,
# granite's and siglip's are all of that shape) is not demanded here, because
# nothing in this repository records that upstream publishes one. Those three
# are covered by ci.yml's `Verify staged artifact checksums` step, which reads
# the file directly and fails when it is missing.
#
# `-f` rather than `-e`, so a directory and a dangling symlink are both
# failures, exactly as in `verify_files_table`.
verify_checksums_present() {
  local repo="$1" selector="$2" localdir="$3" canonical="$4" table_ere="$5"
  local wanted pattern path missing="" count=0
  local -a patterns
  wanted="$(mktemp)"
  : > "$wanted"
  # `read -ra`, never an unquoted `for pattern in $selector`: these selectors
  # CONTAIN `*`, so word splitting them bare would also pathname-expand them
  # against whatever the working directory happens to hold.
  read -ra patterns <<< "$selector"
  for pattern in ${patterns[@]+"${patterns[@]}"}; do
    case "$pattern" in
      *'*'*) continue ;;
      CHECKSUMS.sha256|*/CHECKSUMS.sha256) printf '%s\n' "$pattern" >> "$wanted" ;;
    esac
  done
  awk -v ere="$table_ere" '
    {
      path = substr($0, 67)
      if (path ~ ere && (path == "CHECKSUMS.sha256" || path ~ /\/CHECKSUMS\.sha256$/)) print path
    }
  ' "$canonical" >> "$wanted"
  LC_ALL=C sort -u "$wanted" -o "$wanted"
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    count=$((count + 1))
    if [ ! -f "$localdir/$path" ]; then
      missing="$missing $path"
    fi
  done < "$wanted"
  rm -f "$wanted"
  if [ -n "$missing" ]; then
    echo "::error::$lock stages \"$repo\" into $localdir with a selector ($selector) that names$missing, and after $mode that is not a regular file there. EVERY digest check in this script skips a CHECKSUMS.sha256 — a digest list cannot hold its own digest — so its absence is the one staged-file failure nothing else can see: the tree verifies green while the byte-identity check against upstream's own published list, the only thing that reads the file at all, begins with a \`find\` and quietly compares nothing. Re-download the kit." >&2
    exit 1
  fi
  if [ "$count" -gt 0 ]; then
    echo "MANIFEST[$kit]: $repo stages $count selected CHECKSUMS.sha256 into $localdir, present"
  fi
}

# The path+digest of every file under $1 that the anchored ERE $2 matches,
# written to $3 as `<sha>  <path>` lines sorted whole-line (which is what the
# `comm` set differences below require).
#
# Dot-led components are excluded exactly as the reverse reconciliation below
# excludes them: `hf download` writes its own `.cache/` metadata tree under
# `--local-dir`, macOS leaves `._*` and `.DS_Store` beside real files, and `*`
# crossing `/` would otherwise sweep both into a selector's reach.
# `CHECKSUMS.sha256` is excluded for the reason the merged pass documents —
# upstream's speakerkit copy lists its own name against the digest of an EMPTY
# file — which is why its PRESENCE is asserted separately by
# `verify_checksums_present`: excluded from the digests, never from the tree. An
# absent directory is an empty snapshot, which is what a first download sees.
table_digest_snapshot() {
  local dir="$1" ere="$2" out="$3" paths
  : > "$out"
  [ -d "$dir" ] || return 0
  paths="$(mktemp)"
  ( cd "$dir" && find . -type f ) \
    | sed 's|^\./||' \
    | grep -v '\(^\|/\)\.' \
    | grep -E "$ere" \
    | grep -v '\(^\|/\)CHECKSUMS\.sha256$' \
    | LC_ALL=C sort > "$paths" || true
  if [ -s "$paths" ]; then
    ( cd "$dir" && tr '\n' '\0' < "$paths" | xargs -0 shasum -a 256 ) \
      | LC_ALL=C sort > "$out"
  fi
  rm -f "$paths"
}

# ONE TABLE, VERIFIED AGAINST ITS OWN DOWNLOAD, BEFORE THE NEXT CAN OVERLAY IT
# (coremlit #147, finding 1).
#
# The merged pass at the bottom of this file resolves every table's manifest
# last-writer-wins before it compares a single digest, exactly as the download
# order resolves the bytes. That is right for the TREE and blind for the
# MANIFESTS: the speaker kit's overlay replaces `pyannote_segmentation.mlmodelc/*`
# and `wespeaker.mlmodelc/*`, so the BASE table's lines for those ten paths are
# dropped before anything looks at them. It was measured, not reasoned about:
# with all ten of the base manifest's overlaid entries deleted, `--mode verify`
# still reported that the committed manifests agree with the staged tree — and
# the licence register's direction 1, which enumerates the base table from that
# file, would have been enumerating ten fewer files with nothing red.
#
# So the download verifies each table against what THAT table just wrote:
#
#   (i)  every line of this table's manifest that its selector stages is on
#        disk with that digest. This table has just written those bytes, so the
#        check is exact even for a path a later table will replace.
#   (ii) every path+digest pair that is NEW or CHANGED across this download is
#        listed in this table's manifest. A file this download wrote that its
#        own manifest does not list is a failure; a file left INTACT from an
#        earlier table is attributed to that earlier table's manifest and is
#        not demanded of this one.
#
# A `files` table has no manifest and contributes its explicit list to (i) as
# presence checks; a table on `revision = "main"` has no manifest either and is
# skipped here exactly as it is in the merged pass.
verify_table_download() {
  local repo="$1" field="$2" selector="$3" revision="$4" localdir="$5"
  local ere="$6" before="$7" after="$8"
  local vendor manifest canon filtered changed missing unlisted

  if [ "$field" = "files" ]; then
    # Also checked by the merged pass below, which has to do it because `verify`
    # mode never reaches this function. Doing it here as well costs a `[ -f ]`
    # per file and reports a short download against the table that made it,
    # rather than after every other table has downloaded too.
    verify_files_table "$repo" "$selector" "$localdir"
    return 0
  fi
  if [ "$revision" = "main" ]; then
    echo "MANIFEST[$kit]: $repo is on revision \"main\"; there is no immutable file list to check this download against"
    return 0
  fi
  vendor="${localdir#Models/}"
  manifest="$manifest_dir/$vendor@$revision.sha256"
  if [ ! -f "$manifest" ]; then
    echo "::error::$lock stages \"$repo\" by glob at revision $revision and $manifest does not exist. The licence register enumerates a glob's contents from that file (coremlit #139), so without it a bundle this table stages with no licence row is invisible. Regenerate it — the file name carries the revision, so a revision bump needs a new one." >&2
    exit 1
  fi
  canon="$(mktemp)"
  manifest_canonical "$manifest" "$canon"
  # (i.a) The selected CHECKSUMS.sha256, which the digest comparison below
  # excludes and therefore cannot miss. Checked against THIS table's download,
  # where a short fetch is still attributable to the table that made it.
  verify_checksums_present "$repo" "$selector" "$localdir" "$canon" "$ere"

  # This table's manifest, narrowed to what its selector actually stages. The
  # narrowing is load-bearing in both directions: the speakerkit overlay's
  # manifest is upstream's verbatim list and names the `wespeaker_int8` and
  # `.mlpackage` paths its selector deliberately skips, which (i) must not
  # demand, while the base table's manifest covers all nine bundles the base
  # selector really stages.
  filtered="$(mktemp)"
  awk -v ere="$ere" '{ if (substr($0, 67) ~ ere) print }' "$canon" \
    | grep -v '  \(.*/\)\?CHECKSUMS\.sha256$' \
    | LC_ALL=C sort > "$filtered" || true
  if [ ! -s "$filtered" ]; then
    echo "::error::$manifest lists no file that \"$repo\"'s selector ($selector) stages, so this table would be verified against nothing. Either the selector moved without the manifest being regenerated, or the manifest was cut against a different table." >&2
    rm -f "$canon" "$filtered"
    exit 1
  fi

  if ! missing=$(LC_ALL=C comm -23 "$filtered" "$after") || [ -n "$missing" ]; then
    echo "::error::\"$repo\" at $revision has just downloaded into $localdir, and these lines of $manifest are not on disk with the digest they claim:$(echo " $missing" | tr '\n' ' ')— the manifest is what the licence register enumerates this table from and what the model_io gates read their expected digests from, so a line that was never true of the download is a register entry for bytes nobody has. Regenerate the manifest from this download." >&2
    rm -f "$canon" "$filtered"
    exit 1
  fi

  changed="$(mktemp)"
  LC_ALL=C comm -23 "$after" "$before" > "$changed"
  if ! unlisted=$(LC_ALL=C comm -23 "$changed" "$filtered") || [ -n "$unlisted" ]; then
    echo "::error::\"$repo\" at $revision wrote these files into $localdir and $manifest does not list them:$(echo " $unlisted" | cut -c67- | tr '\n' ' ')— a file a table downloads but its own manifest omits is outside the register, and where a LATER table overwrites the same path the merged check below cannot see the omission at all (that is coremlit #147, finding 1). Regenerate the manifest from this download." >&2
    rm -f "$canon" "$filtered" "$changed"
    exit 1
  fi
  echo "MANIFEST[$kit]: $repo at $revision wrote $(grep -c . "$changed" || true) new or changed file(s) into $localdir; $(grep -c . "$filtered") manifest line(s) verified against that download"
  rm -f "$canon" "$filtered" "$changed"
}

if [ "$mode" = "download" ]; then
  # GitHub's macOS runners ship no `hf`; a developer running this script by hand
  # usually has one. Installing only when it is missing keeps both working and
  # keeps the local path offline-clean. A `--downloader` override needs neither.
  if [ -z "$downloader" ] && ! command -v hf >/dev/null 2>&1; then
    pipx install "huggingface_hub[cli]"
  fi
  # ONE TABLE AT A TIME, snapshot-download-snapshot, so each table is verified
  # against its own bytes before the next table can overlay them. See
  # `verify_table_download` for why the merged pass at the bottom of this file
  # cannot do it.
  for i in "${!plan[@]}"; do
    IFS=$'\t' read -r repo field selector revision localdir <<< "${records[$i]}"
    # Splitting the plan line back into an argv is exact rather than a guess:
    # every token was refused above if it contained whitespace.
    read -ra download_args <<< "${plan[$i]}"

    ere=""
    if [ "$field" = "include" ]; then
      selector_ere "$selector"
    fi
    before="$(mktemp)"
    after="$(mktemp)"
    if [ -n "$ere" ]; then
      table_digest_snapshot "$localdir" "$ere" "$before"
    fi

    if [ -n "$downloader" ]; then
      "$downloader" "${download_args[@]}" </dev/null
    else
      hf download "${download_args[@]}" </dev/null
    fi

    if [ -n "$ere" ]; then
      table_digest_snapshot "$localdir" "$ere" "$after"
    fi
    verify_table_download "$repo" "$field" "$selector" "$revision" "$localdir" \
      "$ere" "$before" "$after"
    rm -f "$before" "$after"
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
  # coremlit/tests/fp16_sweep_inventory.sh).
  shopt -s nullglob dotglob
  entries=("$d"/*)
  shopt -u nullglob dotglob
  if [ "${#entries[@]}" -eq 0 ]; then
    echo "::error::kit '$kit' stages $d, but that directory is EMPTY after $mode — the download matched no file, or the restored cache entry is empty" >&2
    exit 1
  fi
done
echo "kit '$kit': ${#dirs[@]} staged directory/directories present and non-empty"

# ------------------------------------------------------------------------------
# THE COMMITTED MANIFESTS (coremlit #139).
#
# `MODELS_LOCK.d/<vendor_dir>@<revision>.sha256` holds one file list per globbed
# table — upstream's own `CHECKSUMS.sha256` where one ships, a `shasum -a 256`
# over the staged tree where none does. `coremlit/tests/model_licences.rs`
# enumerates a glob's contents from it, which is how a bundle staged with no
# licence row became findable at all; the per-kit `model_io` gates read their
# expected digests from it. All of that rests on the file being TRUE of the
# bytes, and only a job that has downloaded them can say so. Hence this pass,
# which runs on the download path AND on the cache-hit path, and asserts four
# things per staged directory:
#
#   1. every globbed table at an immutable revision HAS a committed manifest,
#      and a table on `revision = "main"` has none (there is nothing to pin);
#   2. every file the manifest lists and the selector stages is on disk with
#      that digest — `shasum -c`, which is fail-closed on a missing file, a
#      changed one, and an empty list alike;
#   3. the reverse: every file the selector stages is IN the manifest. Without
#      this an upstream that publishes a file its own checksum list omits would
#      leave that file outside the register with everything green — the exact
#      shape of #139, one level down;
#   4. every file an explicit `files` table lists is on disk. The lock IS the
#      enumeration for such a table and there is still no manifest to add —
#      what changed in #147 is that presence is now checked rather than assumed
#      from the directory being non-empty.
#
# The ONE permitted absence is a table's own `CHECKSUMS.sha256` DIGEST: a digest
# list cannot contain its own digest, and upstream's speakerkit copy proves it
# by listing the digest of an EMPTY file. It is skipped in both directions here
# and carried by `NON_MODEL_FILES` in the licence register, which states the
# same rule. Its digest is exempt; the FILE is not — `verify_checksums_present`
# above demands every selected copy of it, because until #147's round 2 the
# exclusion meant a cache entry missing exactly that file verified green.
#
# Where two tables share a `local-dir` the LAST one staging a path wins it, the
# same rule the download obeys — so the speaker kit's overlay decides the two
# filenames both layers publish, and the base layer's other seven are checked
# against the base manifest.
#
# WHAT THIS PASS CANNOT SAY, AND WHAT CLOSES IT. Resolving last-writer-wins
# before comparing anything means an overlaid table's own manifest lines are
# never compared with the bytes that table downloaded: in `verify` mode those
# bytes are gone, so the base manifest is UNVERIFIABLE here by construction.
# `verify_table_download` above is what checks them, on the download path only.
# The cache key therefore covers the manifests as well as the lock —
# `hashFiles('MODELS_LOCK', 'MODELS_LOCK.d/*.sha256')` in both ci.yml and
# coverage.yml — so an edited manifest can never restore a cache entry built
# from different ones.
#
# THAT GUARANTEE HAS TWO HALVES AND BOTH ARE NOW TRUE. The key is one; the other
# is that every cache entry either workflow saves was produced by THIS script's
# download mode. It was not, for one release: ci.yml hand-parsed the lock and
# ran its own `hf download` loop, then called only `--mode verify`, so every
# tree ci.yml cached — which coverage.yml then restored — had passed no
# per-table check at all. Migrating that job onto the composite action is what
# makes `verify` mode mean: the merged checks below hold, AND this tree was
# built by download mode from these exact manifests, where the per-table checks
# held.
for d in "${dirs[@]}"; do
  effective="$(mktemp)"    # "<sha>  <path>" lines, later tables overwriting earlier
  selected="$(mktemp)"     # the ERE alternation of every selector staging into $d
  files_listed="$(mktemp)" # every path an explicit `files` table stages into $d
  : > "$effective"
  : > "$selected"
  : > "$files_listed"
  vendor_for_d=""
  while IFS=$'\t' read -r repo field selector revision localdir; do
    [ "$localdir" = "$d" ] || continue
    vendor="${localdir#Models/}"
    vendor_for_d="$vendor"
    manifest="$manifest_dir/$vendor@$revision.sha256"
    if [ "$field" = "files" ]; then
      # An explicit `files` list names every file it stages; the lock IS the
      # enumeration and there is nothing for a manifest to add. It is still
      # checked: `verify_files_table` requires each listed path to be a regular
      # file, which is what a half-restored cache entry fails, and hands the
      # paths to the reverse reconciliation so a files-staged path a sibling
      # table's glob happens to match is not reported as unlisted.
      verify_files_table "$repo" "$selector" "$localdir" "$files_listed"
      continue
    fi
    if [ "$revision" = "main" ]; then
      if [ -f "$manifest" ]; then
        echo "::error::$manifest exists, but $lock pins \"$repo\" at revision \"main\" — a moving revision has no immutable file list, so a manifest here claims bytes the next download can change. Delete it, or pin the revision." >&2
        exit 1
      fi
      echo "MANIFEST[$kit]: $repo is on revision \"main\"; nothing to pin (coremlit/tests/model_licences.rs names it as the one table direction 1 cannot enumerate)"
      continue
    fi
    if [ ! -f "$manifest" ]; then
      echo "::error::$lock stages \"$repo\" by glob at revision $revision and $manifest does not exist. The licence register enumerates a glob's contents from that file (coremlit #139), so without it a bundle this table stages with no licence row is invisible. Regenerate it — the file name carries the revision, so a revision bump needs a new one." >&2
      exit 1
    fi
    selector_ere "$selector"
    printf '%s\n' "$ere" >> "$selected"

    # VALIDATED AND CANONICALISED ONCE, and everything below this line reads
    # the result rather than the raw file — the byte-identity comparison, the
    # merged `effective` set, and (through `verify_table_download`) the
    # download path. A manifest path that is not one file under this table's
    # root never reaches `shasum`.
    canonical="$(mktemp)"
    manifest_canonical "$manifest" "$canonical"

    # The selected CHECKSUMS.sha256 is EXCLUDED from every digest comparison
    # below, so nothing there can notice it missing — and the byte-identity
    # check that follows begins with a `find`, which on an absent file compares
    # nothing and says nothing. Demand it first.
    verify_checksums_present "$repo" "$selector" "$d" "$canonical" "$ere"

    # BYTE-IDENTITY, where upstream ships its own digest list. The committed
    # manifest is supposed to BE that file — verbatim where its paths are
    # already relative to `local-dir`, and otherwise the same lines under one
    # constant path prefix (ced's and redimnet's are relative to the `.mlmodelc`
    # root, granite's and siglip's to the tier directory). Checking the digests
    # alone would not catch a "verbatim" copy somebody regenerated by hashing
    # their own tree: it would agree with the bytes and disagree with what
    # upstream published. So every line of the DOWNLOADED file must appear in
    # the committed one, and the prefix is DERIVED from the first line rather
    # than configured, so a copy assembled under two different prefixes fails
    # here rather than reading as normalisation.
    upstream_lists=$( ( cd "$d" && find . -type f -name CHECKSUMS.sha256 ) \
      | sed 's|^\./||' | grep -E "$ere" || true )
    upstream_count=$(printf '%s' "$upstream_lists" | grep -c . || true)
    if [ "$upstream_count" -gt 1 ]; then
      echo "::error::\"$repo\" stages more than one CHECKSUMS.sha256 ($(echo "$upstream_lists" | tr '\n' ' ')); this verifier cannot say which one $manifest is a copy of" >&2
      exit 1
    fi
    if [ "$upstream_count" -eq 1 ]; then
      if ! awk -v manifest="$canonical" '
        function strip(p) { sub(/^\.\//, "", p); return p }
        BEGIN {
          while ((getline line < manifest) > 0) {
            committed[substr(line, 1, 64) "  " strip(substr(line, 67))] = 1
          }
          prefix = "\001"
        }
        {
          sha = substr($0, 1, 64); path = strip(substr($0, 67))
          if (prefix == "\001") {
            for (p in committed) { }
            # Derive the prefix from THIS line: find the committed entry with
            # the same digest whose path ends in this one.
            found = 0
            for (entry in committed) {
              if (substr(entry, 1, 64) != sha) continue
              cpath = substr(entry, 67)
              if (cpath == path) { prefix = ""; found = 1; break }
              if (length(cpath) > length(path) + 1 &&
                  substr(cpath, length(cpath) - length(path)) == "/" path) {
                prefix = substr(cpath, 1, length(cpath) - length(path) - 1) "/"
                found = 1; break
              }
            }
            if (!found) { print "no committed line matches " sha "  " path > "/dev/stderr"; exit 1 }
          }
          if (!((sha "  " prefix path) in committed)) {
            print "committed manifest has no `" sha "  " prefix path "`" > "/dev/stderr"
            exit 1
          }
        }
        END { if (prefix != "") print "  (paths prefixed with `" prefix "`)" }
      ' "$d/$upstream_lists"; then
        echo "::error::$d/$upstream_lists (the digest list \"$repo\" publishes at $revision) is not what $manifest holds. The committed manifest must be that file — verbatim, or every line under one constant path prefix. Regenerate it from the download rather than from a local tree." >&2
        exit 1
      fi
      echo "MANIFEST[$kit]: $manifest matches $d/$upstream_lists line for line"
    fi
    # The manifest filtered through this table's selector, appended AFTER any
    # earlier table's lines so the awk pass below keeps the last writer. The
    # paths are already canonical, which is why nothing is stripped here.
    awk -v ere="$ere" '{ if (substr($0, 67) ~ ere) print }' "$canonical" >> "$effective"
    rm -f "$canonical"
  done <<< "$tables"

  if [ ! -s "$effective" ]; then
    rm -f "$effective" "$selected" "$files_listed"
    echo "MANIFEST[$kit]: $d has no globbed table with a committed manifest; its explicit file lists were checked above and there is nothing further to verify"
    continue
  fi

  # Last writer wins each path, exactly as the download order does.
  resolved="$(mktemp)"
  awk '{ sha[substr($0, 67)] = substr($0, 1, 64) } END { for (p in sha) print sha[p] "  " p }' \
    "$effective" | LC_ALL=C sort -k2 > "$resolved"

  # (2) every listed file present with that digest. `CHECKSUMS.sha256` is
  # excluded because upstream's speakerkit copy lists its own name against the
  # digest of an EMPTY file — the trap the checksum step downstream already
  # documents. The remaining count is pinned so a filter that rots into matching
  # nothing cannot "verify" an empty set.
  checkable="$(mktemp)"
  grep -v '  \(.*/\)\?CHECKSUMS\.sha256$' "$resolved" > "$checkable" || true
  checkable_lines=$(grep -c . "$checkable" || true)
  if [ "$checkable_lines" -eq 0 ]; then
    echo "::error::the committed manifests for $d resolved to no checkable line; the selector filter or the CHECKSUMS.sha256 exclusion has stopped matching, and this step would verify nothing" >&2
    rm -f "$effective" "$selected" "$files_listed" "$resolved" "$checkable"
    exit 1
  fi
  echo "== $d: verifying $checkable_lines file(s) against the committed manifest(s) =="
  ( cd "$d" && shasum -a 256 -c ) < "$checkable"

  # (3) the reverse. Every file the selectors stage must be listed. Paths with a
  # dot-led component are excluded: `hf download` writes its own `.cache/`
  # metadata tree under `--local-dir`, and macOS leaves `._*` / `.DS_Store`
  # beside real files — neither is published by any repository, and `*` crossing
  # `/` would otherwise sweep them into a selector's reach.
  on_disk="$(mktemp)"
  ( cd "$d" && find . -type f ) \
    | sed 's|^\./||' \
    | grep -v '\(^\|/\)\.' \
    | grep -E -f "$selected" \
    | grep -v '\(^\|/\)CHECKSUMS\.sha256$' \
    | LC_ALL=C sort > "$on_disk" || true
  # An explicit `files` table's paths are LISTED too. They have no manifest —
  # the lock is their enumeration — but a sibling table's glob can match one of
  # them (`*` crosses `/`), and reporting a path the lock names explicitly as
  # "staged by nothing" would be a false red pointing at a manifest that is
  # correct.
  listed="$(mktemp)"
  cut -c67- "$checkable" | cat - "$files_listed" | LC_ALL=C sort -u > "$listed"
  if ! unlisted=$(LC_ALL=C comm -23 "$on_disk" "$listed") || [ -n "$unlisted" ]; then
    echo "::error::$d stages files no committed manifest lists: $(echo "$unlisted" | tr '\n' ' ')— the licence register enumerates a table's contents from its manifest, so an unlisted staged file has no licence row and nothing asked for one. Regenerate the manifest for ${vendor_for_d:-that table} at its pinned revision." >&2
    rm -f "$effective" "$selected" "$files_listed" "$resolved" "$checkable" "$on_disk" "$listed"
    exit 1
  fi
  rm -f "$effective" "$selected" "$files_listed" "$resolved" "$checkable" "$on_disk" "$listed"
done
echo "kit '$kit': committed manifests agree with the staged tree"
