#!/usr/bin/env python3
"""Render one or more Cobertura reports as a readable per-module coverage table.

WHY THIS EXISTS. The point of the coverage workflow is that the owner can see
which code the tests actually reach. codecov.io would normally be where that is
read, but this repository is public with NO secrets configured, so there is no
``CODECOV_TOKEN`` and the upload is best-effort at most. The coverage number must
therefore be legible with no token and no third-party service: this script writes
it straight into the run page's job summary.

MERGING. Given several reports it takes, per source file, the UNION of the lines
each report covered. That is the correct merge for this repository's shape and it
is the whole reason the model legs are worth running: a line inside
``src/audio/ced`` is unreachable in the hermetic leg (its gates are ``#[ignore]``d
without a staged artifact) and reached in the ``ced`` leg, and the merged view
must count it once, as covered. Summing hit COUNTS would double-count lines that
both legs execute; unioning hit LINES does not.

Cobertura, not llvm-cov's own JSON, because Cobertura is what ``cargo llvm-cov``
uploads to codecov and what the artifacts hold — so this reads exactly the bytes
that are published, and a malformed report fails here rather than silently
becoming an empty codecov upload.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse(path: str) -> dict[str, dict[int, bool]]:
    """Map ``filename -> {line number: covered}`` for one Cobertura report."""
    files: dict[str, dict[int, bool]] = defaultdict(dict)
    root = ET.parse(path).getroot()
    for cls in root.iter("class"):
        filename = cls.get("filename")
        if not filename:
            continue
        lines = files[filename]
        for line in cls.iter("line"):
            number = line.get("number")
            hits = line.get("hits")
            if number is None or hits is None:
                continue
            n = int(number)
            # `or` not `=`: a file can appear in several <class> elements (one
            # per generic instantiation), and a line covered by any of them is
            # covered.
            lines[n] = lines.get(n, False) or int(hits) > 0
    return files


def module_of(filename: str) -> str:
    """Group a source path into the module a human would ask about.

    ``crates/coremlit/src/audio/ced/prediction/mod.rs`` -> ``audio/ced``;
    ``crates/coremlit/src/lib.rs`` -> ``(crate root)``. Two path components under
    ``src/`` is the level this crate is organised at (``audio::whisper``,
    ``embeddings::granite``), and it is also the level the coverage legs are
    sharded at, so the table lines up with the legs that produced it.
    """
    parts = filename.replace("\\", "/").split("/")
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "(crate root)"
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def bar(pct: float) -> str:
    filled = int(round(pct / 10.0))
    return "█" * filled + "·" * (10 - filled)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--title", required=True)
    ap.add_argument(
        "--note",
        default="",
        help="One line of context printed under the title (what this leg staged).",
    )
    ap.add_argument(
        "--files",
        action="store_true",
        help="Also emit a collapsed per-FILE table, for the merged view.",
    )
    ap.add_argument("reports", nargs="+")
    args = ap.parse_args()

    merged: dict[str, dict[int, bool]] = defaultdict(dict)
    read, skipped = [], []
    for path in args.reports:
        try:
            one = parse(path)
        except (OSError, ET.ParseError) as exc:
            skipped.append(f"{path}: {exc}")
            continue
        read.append(path)
        for filename, lines in one.items():
            target = merged[filename]
            for n, covered in lines.items():
                target[n] = target.get(n, False) or covered

    if not read:
        # Not an error: the merge job runs with `always()` precisely so a leg
        # that failed or was skipped drops its data instead of killing the
        # signal. Say what was missing rather than printing an empty table.
        print(f"## {args.title}\n")
        print("No readable Cobertura report was found. Reports offered:\n")
        for line in skipped or ["(none)"]:
            print(f"- `{line}`")
        return 0

    by_module: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for filename, lines in merged.items():
        stat = by_module[module_of(filename)]
        stat[0] += sum(1 for covered in lines.values() if covered)
        stat[1] += len(lines)

    total_covered = sum(stat[0] for stat in by_module.values())
    total_lines = sum(stat[1] for stat in by_module.values())
    total_pct = 100.0 * total_covered / total_lines if total_lines else 0.0

    out: list[str] = []
    out.append(f"## {args.title}")
    out.append("")
    if args.note:
        out.append(args.note)
        out.append("")
    out.append(
        f"**{total_pct:.2f}%** lines covered — {total_covered} / {total_lines}, "
        f"merged from {len(read)} report(s)."
    )
    out.append("")
    out.append("| module | lines | covered | % | |")
    out.append("|---|---:|---:|---:|---|")
    for module in sorted(by_module, key=lambda m: (-by_module[m][1], m)):
        covered, lines = by_module[module]
        pct = 100.0 * covered / lines if lines else 0.0
        out.append(f"| `{module}` | {lines} | {covered} | {pct:.1f}% | `{bar(pct)}` |")
    out.append("")

    if args.files:
        out.append("<details><summary>Per file</summary>")
        out.append("")
        out.append("| file | lines | covered | % |")
        out.append("|---|---:|---:|---:|")
        for filename in sorted(merged, key=lambda f: (-len(merged[f]), f)):
            lines = merged[filename]
            covered = sum(1 for c in lines.values() if c)
            pct = 100.0 * covered / len(lines) if lines else 0.0
            out.append(f"| `{filename}` | {len(lines)} | {covered} | {pct:.1f}% |")
        out.append("")
        out.append("</details>")
        out.append("")

    if skipped:
        out.append("<details><summary>Reports that could not be read</summary>")
        out.append("")
        for line in skipped:
            out.append(f"- `{line}`")
        out.append("")
        out.append("</details>")
        out.append("")

    text = "\n".join(out)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
