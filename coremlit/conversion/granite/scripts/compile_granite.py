"""Compile the staged mlpackages to ``.mlmodelc`` and RECORD which run built them.

The compiled bundle is what ships and what every Rust gate loads, but
``producer.json`` cannot cover it: compilation happens after that record is
written, and it binds only the two ``.mlpackage``s. That gap is reachable without
any adversary — run A completes, run B converts and then resumes at staging
without recompiling, and A's bundle is copied beside B's packages. Root-versus-
staging equality still holds (they are the same bytes), every floor still clears,
and the manifest stamps run B's identity onto compiled bytes run A produced.

Two things close it, and both live here rather than in the shell loop this
replaced: ``begin_run`` deletes every staged ``.mlmodelc`` so a skipped
compilation fails at the copy instead of succeeding quietly, and this step writes
``compile.json`` — run id, input package digest, output bundle digest, and the
compiler environment — which verification and manifest generation both require.

The compiler environment is recorded because it is the part of this recipe that
is NOT pinned by the Python toolchain: README measures two runs producing
identical ``model.mil`` and ``weight.bin`` while their compiled ``coremldata.bin``
differ, so the bundle's bytes are attributable only alongside the compiler that
emitted them.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _granite_common import (
    COMPILED_PAIRS,
    FP32_REFERENCE,
    RUN_ID_KEY,
    SHIPPED_PACKAGE,
    compiled_digests,
    digest_tree,
    discard_tree,
    read_producer_record,
    stage_dir,
    write_compile_record,
)


def main():
    stage = stage_dir()
    # Bind to the conversion that produced these packages BEFORE compiling: the
    # record this step writes is only meaningful if it names that run, and a
    # stage holding packages from some other run must not be compiled into a
    # bundle this run then claims.
    produced = {
        name: digest_tree(os.path.join(stage, name))
        for name in (SHIPPED_PACKAGE, FP32_REFERENCE)
    }
    run_id = read_producer_record(stage, produced)[RUN_ID_KEY]

    for package, bundle in COMPILED_PAIRS:
        src = os.path.join(stage, package)
        dst = os.path.join(stage, bundle)
        if not os.path.isdir(src):
            raise SystemExit(f"missing {src} — run convert_granite.py first")
        # Never compile ON TOP of an existing bundle: coremlcompiler would leave
        # any file it no longer emits in place, and the result would be a tree
        # that is part this build and part the last one.
        discard_tree(dst)
        # ABSOLUTE at the subprocess boundary. GRANITE_STAGE is caller-supplied
        # and may be relative; a relative stage beginning with `-` makes these
        # operands look like options to coremlcompiler, which has no `--`
        # end-of-options separator to fall back on ("compile command missing
        # model file path"). Every absolute path starts with `/`, so no operand
        # can be misread. This is distinct from quoting: shlex round-trips a
        # leading-dash path perfectly and the receiving program still misparses
        # it — two different failures that both look like "the path broke it".
        try:
            subprocess.run(
                ["xcrun", "coremlcompiler", "compile",
                 os.path.abspath(src), os.path.abspath(stage)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(f"COMPILE FAILED for {src}: {e}") from e
        if not os.path.isdir(dst):
            raise SystemExit(
                f"coremlcompiler reported success for {src} but {dst} does not exist"
            )
        print(f"COMPILED {src} -> {dst}")

    write_compile_record(stage, run_id, compiled_digests(stage))
    print(f"DONE compile (run {run_id[:12]})")


if __name__ == "__main__":
    main()
