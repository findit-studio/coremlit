#!/usr/bin/env bash
#
# merge_golden_hosts.sh <committed-golden.json> <fresh-golden.json>
#
# Decides, BY MEASUREMENT, what a freshly regenerated whisper golden is allowed
# to claim about the host classes its payload has been reproduced on, and writes
# the resulting golden to STDOUT. It never opens a file for writing, so it
# cannot clobber a committed golden even when invoked with the arguments
# reversed; `regen_goldens.sh` redirects it, and the CI job redirects it outside
# the checkout.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY A SET, AND WHY A TOOL DECIDES IT
#
# `generationHost` used to name ONE host class, and `check_host_class` demanded
# the running machine equal it. That is the right shape for a two-machine world
# and the wrong shape for GitHub's hosted pool: during a `macos-15` image
# rollover some runners report build 24G720 (macOS 15.7.7) and others 24G830
# (15.7.9), a job lands on whichever it lands on, and a single-host golden reds
# roughly half of them while saying nothing whatsoever about the port.
#
# The honest fix is not a wider tolerance and not a pinned runner. It is to say
# what was measured: the three whisper goldens' tokens, segments and word
# timestamps came out byte-identical on BOTH of those images, so the payload
# demonstrably reproduces on both, and the golden should record both.
#
# What it must NOT become is a set that grows by assumption. A class earns its
# place by having reproduced the committed payload exactly, and this script is
# where that is checked:
#
#   payload byte-identical  -> APPEND the fresh host class to the committed set.
#                              (`Match` still means "the oracle produced exactly
#                              these numbers on a machine of this class".)
#   payload differs         -> REPLACE the set with the fresh host class ALONE.
#                              The other classes reproduced the OLD payload;
#                              they have said nothing about this one, and
#                              carrying them over would be a claim nobody
#                              measured. Printed loudly, because changed oracle
#                              output is news.
#   class already recorded  -> unchanged. Re-running on a recorded class is
#                              idempotent, and the `source` already on file is
#                              kept: it is the label of a run that did produce
#                              this payload, and quietly rewriting provenance is
#                              not this tool's job.
#
# The payload is EVERYTHING except the provenance keys `generationHost`,
# `generationHosts` and `source`. `source` is excluded deliberately: the oracle's
# own version label legitimately differs between two images that produce
# identical numbers (the rollover above shipped whisperkit-cli v1.0.0 on 24G720
# and v1.1.0 on 24G830), so folding it into the comparison would force a REPLACE
# on every image bump and defeat the whole point. It is not discarded — each
# recorded host carries the `source` observed there, so the set says which
# oracle build reproduced the payload on which image.
#
# Like `regen_goldens.sh`, this file may never learn to build or run coremlit: a
# golden's entire value is that something other than the code under test
# produced it, and `whisper_golden_provenance` greps both scripts for Rust's
# build tool to keep it that way.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

committed="${1:-}"
fresh="${2:-}"
[[ -n "$committed" && -n "$fresh" ]] ||
  die "usage: merge_golden_hosts.sh <committed-golden.json> <fresh-golden.json>
       Writes the merged golden to stdout."
[[ -s "$fresh" ]] || die "no freshly generated golden at $fresh — nothing to merge"
command -v jq >/dev/null 2>&1 || die "jq is not on PATH (brew install jq)"

# Both documents' recorded host set, normalized to one shape:
#
#   * `generationHosts` (the current schema) is taken as-is;
#   * a legacy single `generationHost` object becomes a one-element set — which
#     is what lets this tool read the goldens committed before the set existed,
#     and the goldens a pre-set `regen_goldens.sh` produced;
#   * an entry with no `source` of its own inherits the document's top-level
#     one, so a legacy golden's provenance survives the promotion instead of
#     being silently dropped.
#
# Entries are rebuilt field by field, so key order is fixed and any unknown key
# is dropped rather than carried into the committed fixture.
normalize='
  def hostentry:
    {osBuild, osProductVersion, chip, arch}
    + (if (.source // null) != null then {source: .source} else {} end);
  def hostkey: {osBuild, osProductVersion, chip, arch};
  def hosts:
    (.source // null) as $top
    | (if (.generationHosts // null) != null then .generationHosts
       elif (.generationHost // null) != null then [.generationHost]
       else [] end)
    | map(if (.source // null) == null and $top != null then . + {source: $top} else . end)
    | map(hostentry);
'

# The payload: the golden minus every provenance key, key-sorted and compact, so
# the comparison is over content and not over formatting or key order.
payload() { jq -S -c 'del(.generationHost, .generationHosts, .source)' "$1"; }

fresh_hosts_json="$(jq -c "$normalize"' hosts' "$fresh")"
[[ "$(printf '%s' "$fresh_hosts_json" | jq 'length')" -ge 1 ]] ||
  die "$fresh records no generation host — regenerate it with regen_goldens.sh, which stamps
       the running machine's host class"

if [[ ! -s "$committed" ]]; then
  # No committed golden to compare against (a brand-new fixture): the fresh
  # host class stands alone, which is exactly the REPLACE shape.
  echo "[merge] NEW: no committed golden at $committed — recording only this run's host class" >&2
  jq "$normalize"'
    (hosts) as $set
    | (del(.generationHost, .generationHosts) | to_entries) as $e
    | ($e | map(.key) | index("source")) as $i
    | (if $i == null then [{key: "generationHosts", value: $set}] + $e
       else $e[0:$i + 1] + [{key: "generationHosts", value: $set}] + $e[$i + 1:] end)
    | from_entries
  ' "$fresh"
  exit 0
fi

if [[ "$(payload "$committed")" == "$(payload "$fresh")" ]]; then
  mode=append
else
  mode=replace
fi

merged="$(jq -n --slurpfile c "$committed" --slurpfile f "$fresh" --arg mode "$mode" \
  "$normalize"'
  ($c[0]) as $C
  | ($f[0]) as $F
  | ($C | hosts) as $ch
  | ($F | hosts) as $fh
  | ($ch | map(hostkey)) as $recorded
  | (if $mode == "append"
     then [$C, $ch + ($fh | map(hostkey as $k | select(($recorded | index($k)) == null)))]
     else [$F, $fh] end) as [$base, $set]
  | $base
  | (del(.generationHost, .generationHosts) | to_entries) as $e
  | ($e | map(.key) | index("source")) as $i
  | (if $i == null then [{key: "generationHosts", value: $set}] + $e
     else $e[0:$i + 1] + [{key: "generationHosts", value: $set}] + $e[$i + 1:] end)
  | from_entries
')"

before="$(jq "$normalize"' hosts | length' "$committed")"
after="$(printf '%s' "$merged" | jq '.generationHosts | length')"
this_host="$(printf '%s' "$fresh_hosts_json" |
  jq -r '.[0] | "macOS \(.osProductVersion) (build \(.osBuild)), \(.chip), \(.arch)"')"

case "$mode" in
  append)
    if [[ "$after" -gt "$before" ]]; then
      echo "[merge] APPEND: payload byte-identical to $(basename "$committed") — recording" \
           "$this_host as a host class this payload reproduces on ($before -> $after recorded)" >&2
    else
      echo "[merge] UNCHANGED: payload byte-identical and $this_host is already one of the" \
           "$before recorded host classes — nothing to record" >&2
    fi
    ;;
  replace)
    {
      echo "[merge] ============================ REPLACE ============================"
      echo "[merge] The oracle produced a DIFFERENT payload on $this_host."
      echo "[merge] $(basename "$committed") is NOT reproduced here, so the $before host"
      echo "[merge] class(es) it recorded no longer say anything about these numbers and"
      echo "[merge] have been dropped. The golden now claims this host class ALONE."
      echo "[merge] READ THE DIFF. Changed oracle output is news — a new macOS build, a new"
      echo "[merge] whisperkit-cli, a new model revision — and it is not routine. Do not"
      echo "[merge] commit it as a host rotation, and do not widen a parity tolerance."
      echo "[merge] ================================================================="
    } >&2
    ;;
esac

printf '%s\n' "$merged"
