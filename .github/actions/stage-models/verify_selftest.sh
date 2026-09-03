#!/usr/bin/env bash
#
# `stage.sh`'s own falsifiers, run against a synthetic lock in a temporary
# directory — no network, no `hf`, no model bytes, about a second.
#
# WHY IT EXISTS. Every check `stage.sh` performs is a check on artifacts that
# only CI downloads, so the script's own failure modes were never exercised by
# anything: a check that silently stopped matching would look exactly like a
# clean run, and one of them HAD (coremlit #147). Each case below is a mutation
# a person could really commit — a manifest line deleted, a digest edited, a
# path that climbs out of the table root, a cache entry short of one file — and
# every one of them fails the script here before it can pass in CI.
#
# THE CASE THAT COST THE MOST. `Models/speakerkit` is staged by two tables, and
# the merged manifest pass resolves them last-writer-wins BEFORE it compares a
# digest, so the base table's lines for the ten paths the overlay replaces were
# dropped unread. Deleting all ten from the committed base manifest left
# `--mode verify` reporting that the manifests agree with the staged tree. The
# fix is per-table verification on the DOWNLOAD path — the only place the base
# bytes still exist — which is why this script drives `--mode download` through
# `stage.sh --downloader`, a stub that copies fixture files instead of calling
# `hf`. Everything else about that path is the real one.
#
# IT IS ALSO THE SHELL-SIDE PROOF OF THE ci.yml MIGRATION. `model-tests` used to
# download through its own inline loop and then call only `--mode verify`, so
# every cache entry it produced had passed no per-table check; that job now
# stages through this script like `coverage.yml` does. Case 2's mutation A —
# `--mode download` over the base+overlay pair with one overlaid entry deleted
# from the base manifest — is exactly the check that path was skipping, and it
# is red here. (The verify leg of the same mutation is green BY DESIGN, and
# stated as such below: that is the difference the migration removes.)
#
# WHAT IT DOES NOT COVER. `hf download`'s own selector semantics: the stub
# copies a fixture tree wholesale, so a table's fixture IS what that repository
# publishes under its selector. That the argv is DERIVED from the lock rather
# than restated is pinned hermetically by
# `ci_workflow_derives_downloads_from_the_lock_instead_of_hardcoding_them`
# (coremlit/tests/whisper/models_lock.rs); `--mode plan` prints it for a person.
#
# COREMLIT_STAGE_SH points this at another copy of stage.sh. It is here so the
# cases can be shown RED against the version before the fix; CI never sets it.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage="${COREMLIT_STAGE_SH:-$here/stage.sh}"
if [ ! -f "$stage" ]; then
  echo "verify_selftest.sh: no stage.sh at $stage" >&2
  exit 2
fi

# `stage.sh` appends its `local-dirs` output to $GITHUB_OUTPUT when the variable
# is set. Under CI it IS set, and thirteen invocations would append thirteen
# blocks to the calling step's real output file. This script needs none of it.
unset GITHUB_OUTPUT

root="$(mktemp -d "${TMPDIR:-/tmp}/coremlit-stage-selftest.XXXXXX")"
trap 'rm -rf "$root"' EXIT

passed=0
failed=0

# ------------------------------------------------------------------------------
# THE SYNTHETIC KIT.
#
# Four tables, shaped like the ones that carry the real hazards:
#
#   fake/base     a glob table staging five files into Models/fakekit
#   fake/overlay  a glob table that REPLACES two of them — the speakerkit shape,
#                 and the reason the base table's manifest needs checking before
#                 this one runs. Its selector also names `CHECKSUMS.sha256`
#                 literally, as clapkit's, redimnetkit's and the real speakerkit
#                 overlay's do: that file is excluded from every digest
#                 comparison, so only a presence check can miss it being gone
#   fake/extras   an explicit `files` table staging into the SAME directory, on
#                 a path the base table's selector also matches — the case the
#                 reverse reconciliation must not report as unlisted
#   fake/tokens   an explicit `files` table with a directory to itself, the
#                 Models/tokenizers/whisper-tiny shape: three files, and a cache
#                 entry holding one of them used to verify clean
template="$root/template"
mkdir -p "$template/MODELS_LOCK.d" "$template/fixtures"

BASE_REV=1111111111111111111111111111111111111111
OVERLAY_REV=2222222222222222222222222222222222222222
EXTRAS_REV=3333333333333333333333333333333333333333
TOKENS_REV=4444444444444444444444444444444444444444

cat > "$template/MODELS_LOCK" <<LOCK
["fake/base"]
kit       = "fake"
include   = "*.mlmodelc/* extra.json"
revision  = "$BASE_REV"
local-dir = "Models/fakekit"

["fake/overlay"]
kit       = "fake"
include   = "b.mlmodelc/* CHECKSUMS.sha256"
revision  = "$OVERLAY_REV"
local-dir = "Models/fakekit"

["fake/extras"]
kit       = "fake"
files     = "extra.json"
revision  = "$EXTRAS_REV"
local-dir = "Models/fakekit"

["fake/tokens"]
kit       = "fake"
files     = "tokenizer.json config.json"
revision  = "$TOKENS_REV"
local-dir = "Models/faketokens"
LOCK

write() { mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; }

write "$template/fixtures/base/a.mlmodelc/coremldata.bin" "a-core"
write "$template/fixtures/base/a.mlmodelc/model.mil"      "a-mil"
write "$template/fixtures/base/b.mlmodelc/coremldata.bin" "b-core-BASE"
write "$template/fixtures/base/b.mlmodelc/model.mil"      "b-mil-BASE"
# Matched by the OVERLAY's selector but not published by it: the overlay leaves
# it intact, so it belongs to the base table's manifest and must not be demanded
# of the overlay's.
write "$template/fixtures/base/b.mlmodelc/base_only.bin"  "b-base-only"

write "$template/fixtures/overlay/b.mlmodelc/coremldata.bin" "b-core-OVERLAY"
write "$template/fixtures/overlay/b.mlmodelc/model.mil"      "b-mil-OVERLAY"
# The digest list this repository PUBLISHES. The redirection creates the file
# before `find` walks the tree, so the list ends up naming ITSELF against the
# sha256 of an empty file — which is not an accident of this fixture but exactly
# what upstream's speakerkit copy does, and the reason no digest check in
# stage.sh may read a CHECKSUMS.sha256 line.
( cd "$template/fixtures/overlay" && find . -type f | LC_ALL=C sort \
  | tr '\n' '\0' | xargs -0 shasum -a 256 ) > "$template/fixtures/overlay/CHECKSUMS.sha256"

write "$template/fixtures/extras/extra.json" "extra"

write "$template/fixtures/tokens/tokenizer.json" "tokenizer"
write "$template/fixtures/tokens/config.json"    "config"

# The base manifest with bare paths, the overlay's with the leading `./` that
# `shasum` writes and that the verbatim upstream copies (clapkit's,
# speakerkit@3db6998's) carry on every line. Both spellings must stay readable,
# which is half of why the path rule strips exactly one `./` and no more.
( cd "$template/fixtures/base" && find . -type f | sed 's|^\./||' | LC_ALL=C sort \
  | tr '\n' '\0' | xargs -0 shasum -a 256 ) > "$template/MODELS_LOCK.d/fakekit@$BASE_REV.sha256"
# The overlay's committed manifest is that repository's OWN published list,
# copied VERBATIM — the shape MODELS_LOCK records for speakerkit, clapkit,
# cedkit, embedkit, siglip and redimnetkit, and what stage.sh's byte-identity
# check compares against. Its self-listing empty-file digest comes along with it.
cp "$template/fixtures/overlay/CHECKSUMS.sha256" \
   "$template/MODELS_LOCK.d/fakekit@$OVERLAY_REV.sha256"

# The `hf download` stand-in. It receives the IDENTICAL argv stage.sh would
# hand `hf`, reads `--local-dir` out of it, and copies that repository's fixture
# tree there — which is what a real download of a table whose selector matches
# exactly its published files does.
cat > "$template/downloader.sh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
repo="$1"; shift
dir=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --local-dir) dir="$2"; shift 2 ;;
    --revision|--include) shift 2 ;;
    *) shift ;;   # a `files` table's positional filenames
  esac
done
[ -n "$dir" ] || { echo "downloader stub: no --local-dir in the argv" >&2; exit 1; }
src="$(cd "$(dirname "$0")" && pwd)/fixtures/${repo##*/}"
[ -d "$src" ] || { echo "downloader stub: no fixture tree at $src" >&2; exit 1; }
mkdir -p "$dir"
cp -R "$src/." "$dir/"
STUB
chmod +x "$template/downloader.sh"

# ------------------------------------------------------------------------------
# The harness.

# `dir` is set rather than printed, and every case calls these as plain
# commands. A command substitution would run them in a SUBSHELL, where the case
# counter and the pass/fail totals are incremented into a copy that is then
# thrown away — which is exactly how the first draft of this file ran thirteen
# cases against one accumulating directory and still reported them.
case_no=0
new_case() { # -> a fresh, un-staged copy of the template, in `dir`
  case_no=$((case_no + 1))
  dir="$root/case-$case_no"
  mkdir -p "$dir"
  cp -R "$template/." "$dir/"
}

run() { # $1 case dir, then stage.sh arguments; sets `run_status` and `run_log`
  local dir="$1"; shift
  run_log="$dir/log-$RANDOM.txt"
  set +e
  ( cd "$dir" && bash "$stage" --kit fake --lock MODELS_LOCK "$@" ) > "$run_log" 2>&1
  run_status=$?
  set -e
}

check() { # $1 label, $2 wanted exit, $3 wanted substring ("" for none)
  local label="$1" want="$2" saying="${3-}"
  if [ "$run_status" -ne "$want" ]; then
    printf 'FAIL  %s\n      wanted exit %s, got %s\n' "$label" "$want" "$run_status"
    sed 's/^/      | /' "$run_log"
    failed=$((failed + 1))
    return
  fi
  if [ -n "$saying" ] && ! grep -qF -- "$saying" "$run_log"; then
    printf 'FAIL  %s\n      exit %s was right but the message never says %s\n' "$label" "$want" "$saying"
    sed 's/^/      | /' "$run_log"
    failed=$((failed + 1))
    return
  fi
  printf 'ok    %s (exit %s)\n' "$label" "$run_status"
  passed=$((passed + 1))
}

staged_case() { # a fresh case with the tree already downloaded green, in `dir`
  new_case
  run "$dir" --mode download --downloader ./downloader.sh
  if [ "$run_status" -ne 0 ]; then
    printf 'FAIL  fixture setup: the baseline download is not green\n'
    sed 's/^/      | /' "$run_log"
    failed=$((failed + 1))
  fi
}

base_manifest() { printf '%s' "$1/MODELS_LOCK.d/fakekit@$BASE_REV.sha256"; }
overlay_manifest() { printf '%s' "$1/MODELS_LOCK.d/fakekit@$OVERLAY_REV.sha256"; }

drop_line() { # $1 manifest, $2 grep pattern to remove
  local kept="$1.kept"
  grep -v -- "$2" "$1" > "$kept"
  if [ "$(wc -l < "$kept")" -eq "$(wc -l < "$1")" ]; then
    echo "verify_selftest.sh: pattern '$2' matched nothing in $1 — the fixture and the mutation have drifted" >&2
    exit 2
  fi
  mv "$kept" "$1"
}

echo "== stage.sh self-test: $stage"

# ------------------------------------------------------------------------------
# 1. The baseline, both modes.
#
# The verify leg is not a formality: `Models/fakekit/extra.json` is staged by an
# explicit `files` table and matched by the BASE table's glob, so before #147 it
# was on disk, inside a selector's reach, and in no manifest — the reverse
# reconciliation reported it as staged-but-unlisted and the whole pass failed.
new_case
run "$dir" --mode download --downloader ./downloader.sh
check "a clean download verifies every table against its own bytes" 0 \
  "wrote 5 new or changed file(s)"
run "$dir" --mode verify
check "verify over the tree that download produced" 0 \
  "committed manifests agree with the staged tree"

# ------------------------------------------------------------------------------
# 2. The overlaid entries — coremlit #147, finding 1.
#
# Mutation A deletes ONE base-manifest entry for a path the overlay replaces.
# In verify mode the base bytes are gone, so the check is not merely absent, it
# is impossible; that is stated as an expectation here rather than left to be
# rediscovered, and the cache key covering MODELS_LOCK.d/ is what stops a tree
# built from other manifests reaching it.
new_case
drop_line "$(base_manifest "$dir")" '  b\.mlmodelc/model\.mil$'
run "$dir" --mode download --downloader ./downloader.sh
check "A: base manifest omits an entry the overlay replaces (download)" 1 \
  "does not list them"

staged_case
drop_line "$(base_manifest "$dir")" '  b\.mlmodelc/model\.mil$'
run "$dir" --mode verify
check "A: the same omission in verify mode, which cannot see it (by design)" 0 \
  "committed manifests agree with the staged tree"

# B: every overlaid entry gone, which is the shape the real experiment used.
new_case
drop_line "$(base_manifest "$dir")" '  b\.mlmodelc/\(coremldata\.bin\|model\.mil\)$'
run "$dir" --mode download --downloader ./downloader.sh
check "B: base manifest omits ALL the overlaid entries (download)" 1 \
  "does not list them"

# The other direction of the same check: the entry is present but WRONG. Nothing
# else would catch it — the merged pass drops that line before comparing.
new_case
manifest="$(base_manifest "$dir")"
sed 's|^[0-9a-f]\{64\}\(  b\.mlmodelc/model\.mil\)$|0000000000000000000000000000000000000000000000000000000000000000\1|' \
  "$manifest" > "$manifest.edited" && mv "$manifest.edited" "$manifest"
run "$dir" --mode download --downloader ./downloader.sh
check "base manifest has the WRONG digest for an overlaid path (download)" 1 \
  "not on disk with the digest they claim"

# And the overlay's own side: a file it downloads that its manifest omits.
new_case
write "$dir/fixtures/overlay/b.mlmodelc/stowaway.bin" "stowaway"
run "$dir" --mode download --downloader ./downloader.sh
check "the overlay writes a file its own manifest does not list (download)" 1 \
  "does not list them"

# C: a NON-overlaid entry. The merged pass CAN see this one, which is what made
# it look as though the merged pass covered the overlaid ones too.
staged_case
drop_line "$(base_manifest "$dir")" '  a\.mlmodelc/model\.mil$'
run "$dir" --mode verify
check "C: base manifest omits a NON-overlaid entry (verify)" 1 \
  "stages files no committed manifest lists"

# ------------------------------------------------------------------------------
# 3. Explicit `files` tables — coremlit #147, finding 2.
#
# A directory staged only by a `files` table used to pass for being non-empty,
# so a cache entry short of one file verified clean and the gates found out an
# hour later.
staged_case
rm -f "$dir/Models/faketokens/config.json"
run "$dir" --mode verify
check "a files-only directory missing one of its files (verify)" 1 \
  "is not a regular file there"

staged_case
rm -f "$dir/Models/fakekit/extra.json"
run "$dir" --mode verify
check "a files table sharing a directory with a glob table, one file gone" 1 \
  "is not a regular file there"

staged_case
rm -f "$dir/Models/faketokens/config.json"
mkdir -p "$dir/Models/faketokens/config.json"
run "$dir" --mode verify
check "a DIRECTORY where a files table names a file (verify)" 1 \
  "is not a regular file there"

# ------------------------------------------------------------------------------
# 3b. The selected `CHECKSUMS.sha256` — coremlit #147, round 2 finding 2.
#
# That one path is excluded from the snapshots, from the manifest digest
# comparison and from the reverse reconciliation, and has to be: a digest list
# cannot hold its own digest, and this fixture's copy — like upstream's
# speakerkit copy — records itself against the sha256 of an EMPTY file. The
# exclusion made its ABSENCE invisible. A cache entry holding every model file
# but not the checksum list verified green, and the byte-identity check against
# upstream's published list, the only thing that reads the file at all, begins
# with a `find` — so with no such file it compared nothing and said nothing.
staged_case
rm -f "$dir/Models/fakekit/CHECKSUMS.sha256"
run "$dir" --mode verify
check "a cache entry missing only the selected CHECKSUMS.sha256 (verify)" 1 \
  "a digest list cannot hold its own digest"

# The download half. The fixture repository stops publishing the file its own
# selector names — what a truncated or wrongly-filtered download looks like from
# here — and it must fail against the table that made it.
new_case
if [ ! -f "$dir/fixtures/overlay/CHECKSUMS.sha256" ]; then
  echo "verify_selftest.sh: the overlay fixture publishes no CHECKSUMS.sha256 — the fixture and this mutation have drifted" >&2
  exit 2
fi
rm -f "$dir/fixtures/overlay/CHECKSUMS.sha256"
run "$dir" --mode download --downloader ./downloader.sh
check "a download that never writes the selected CHECKSUMS.sha256" 1 \
  "a digest list cannot hold its own digest"

# ------------------------------------------------------------------------------
# 4. Manifest paths — coremlit #147, finding 3.
#
# `stage.sh` hashes a manifest path with the staged directory as its working
# directory, so a path that is not one file under the table root either escapes
# that root or names something with no digest.
for bad in '../climbs-out' 'a.mlmodelc/..' '..' '.' './/doubled' 'a/./b' 'a//b' 'trailing/' '/absolute'; do
  staged_case
  printf '%s  %s\n' "0000000000000000000000000000000000000000000000000000000000000000" "$bad" \
    >> "$(base_manifest "$dir")"
  run "$dir" --mode verify
  check "a manifest path of '$bad' (verify)" 1 "Every line of a committed manifest is"
done

staged_case
repeat="$(head -1 "$(base_manifest "$dir")")"
printf '%s\n' "$repeat" >> "$(base_manifest "$dir")"
run "$dir" --mode verify
check "a manifest that lists one path twice, digests agreeing (verify)" 1 \
  "one path holds one set of bytes"

# The positive control. The overlay's manifest already carries `./` on every
# line, so re-spelling the base's the same way must change nothing at all: the
# rule strips exactly one leading `./`, which is what keeps a verbatim upstream
# list readable.
staged_case
manifest="$(base_manifest "$dir")"
sed 's|^\([0-9a-f]\{64\}  \)|\1./|' "$manifest" > "$manifest.dotted" && mv "$manifest.dotted" "$manifest"
run "$dir" --mode verify
check "every base-manifest path re-spelled with a leading ./ (verify)" 0 \
  "committed manifests agree with the staged tree"

# ------------------------------------------------------------------------------
# 5. The selector refusal that this file's own first draft LOST.
#
# `pattern_to_ere` translates a lock selector into an ERE and refuses any
# metacharacter it does not implement, because a silently-wrong match would be
# a coverage hole rather than an error. Factoring the alternation into a
# function that PRINTED its result buried that refusal in a nested command
# substitution: `set -e` did not carry the failure out of the enclosing
# assignment, and a table whose selector contained `[` verified GREEN while
# matching nothing at all. The function sets a variable now. This is what says
# so.
new_case
sed 's|^include   = "\*\.mlmodelc/\* extra\.json"$|include   = "[abc]*.mlmodelc/*"|' \
  "$dir/MODELS_LOCK" > "$dir/MODELS_LOCK.edited" && mv "$dir/MODELS_LOCK.edited" "$dir/MODELS_LOCK"
if ! grep -q '\[abc\]' "$dir/MODELS_LOCK"; then
  echo "verify_selftest.sh: the selector edit matched nothing — the fixture lock has drifted" >&2
  exit 2
fi
run "$dir" --mode download --downloader ./downloader.sh
check "a selector with a metacharacter the verifier does not implement" 1 \
  "regular-expression metacharacter"

# ------------------------------------------------------------------------------
printf '\n%s passed, %s failed\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
