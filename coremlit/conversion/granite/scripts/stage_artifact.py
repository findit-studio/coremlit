"""Promote the compiled bundles into the artifact root — transactionally.

Step 3 used to be four shell lines: remove the two attestations, remove the two
prior bundles, then ``cp -R`` the new ones in. A preflight was added so a missing
staging source could not trigger the removals, but that only proves the sources
are directories — it cannot prove the copy will succeed. If the first ``cp``
landed and the second hit ENOSPC, EIO or an unreadable nested file, the artifact
root was already stripped of both attestations and both prior bundles, and
``set -e`` exited leaving a new ``.mlmodelc``, no ``.mlpackage``, and nothing
describing either.

So the copy happens FIRST, into scratch, while the existing publication is still
whole; it is validated against its source by digest; and only then is anything in
the root touched — by renames that can be undone.

Order inside the promotion is load-bearing. It runs as three phases, each flushed
before the next begins:

    1. the two attestations move aside      (fsync root, aside)
    2. the two prior bundles move aside     (fsync root, aside)
    3. the new bundles move into the root   (fsync root, new)

and the scratch holding the copies is flushed before phase 1 touches the root,
since it is about to hold the only copy of everything that leaves.

WHAT THAT BUYS, stated exactly, because a phase-and-fsync structure invites a much
stronger reading than the code supports. The guarantee is against THIS PROCESS
stopping: an exception, ENOSPC, EIO, a Ctrl-C, a SIGKILL, at any point in the
sequence. ``os.replace`` is atomic and the page cache outlives the process, so
whatever the next process reads is one of the states this ordering permits — and
none of them has an attestation in the root without the bundles it describes. That
holds independently of any fsync.

WHAT IT DOES NOT BUY: power-loss ordering. No such claim is made here. The fsyncs
are kept and they do something real — each phase's directory change is pushed out
of host memory, so a kernel panic cannot lose it there — but macOS ``fsync(2)``
states outright that the drive "may not physically write the data to the platters
for quite some time and it may be written in an out-of-order sequence", that
"later writes may be present, while earlier writes are not", and that "This is
not a theoretical edge case." Ordering across a power cut needs
``F_FULLFSYNC``/``F_BARRIERFSYNC``, and even then nothing makes six renames
across three directories recover as a unit. So the recovery story after a power
cut is the honest one: RE-RUN THE CONVERSION. These artifacts are regenerable,
and the gates fail closed on a torn publication rather than certifying it.

Nor is an interrupted promotion resumable. There is no journal here and no
recovery state machine: a run killed mid-promotion leaves its scratch in place,
the next run REFUSES while that scratch exists, and reassembly is by hand. That
refusal is deliberate — see ``assert_no_stale_promotions``.

Rollback runs the phases backwards for the same reason the promotion runs them
forwards: the new bundles come out of the root, the prior bundles go back, and the
attestations go back LAST. It STOPS at the first failure. Continuing past one is
what would put an attestation back over a payload that did not return, which is
the state every ordering rule here exists to prevent.

ONE RUN AT A TIME — a limitation, not a guarantee. Nothing here locks the artifact
root. Two concurrent promotions against the same root can interleave their renames
and lose the previous publication, and the stale-scratch refusal does not prevent
it: that check runs at start-up, so a second run beginning before the first has
minted its scratch passes it and proceeds. This is not introduced by the
transaction, and step 3 is not even where such a pair would first break — steps 1
and 2 already share ``$GRANITE_STAGE``, where ``begin_run`` deletes the other
run's records and both runs write the same package and bundle paths, so a
concurrent pair is incoherent long before it reaches here. The recipe is an
owner-run, single-run pipeline; a lock is deliberately not part of it.

The scratch directory is a sibling of the artifact root, not a child. The
exact-file-set gate walks the ROOT recursively and would reject any temp inside
it, so keeping the scratch outside means a crashed promotion cannot produce a
gate failure that looks like a fault in the artifact. It is also where the ONLY
surviving copy of the previous bundles lives while a promotion is in flight,
which is why it is preserved from the moment the root is first touched until
either the promotion or its rollback has finished.

Peak disk is the one cost of that. The shell step this replaced removed the
prior bundles and then copied, so it never needed much more than max(old, new);
holding the publication whole while the replacement is staged beside it needs
old + new, so the artifact's footprint transiently roughly DOUBLES. Budget for
that on the volume holding $GRANITE_MODELS_OUT — a run that cannot afford it
fails during the copy, before anything in the root is touched.
"""
import os
import shlex
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from _granite_common import (
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    SHIPPED_BUNDLE,
    SHIPPED_PACKAGE,
    digest_tree,
    fsync_dir,
    fsync_tree,
    model_root,
    stage_dir,
)

# Scratch directory for an in-flight promotion, created as a sibling of the
# artifact root. ``tempfile.mkdtemp`` appends its own random suffix.
PROMOTE_PREFIX = ".granite-promote."

# The two attestations, and the two bundles they describe. Each group is one
# promotion phase, and they move in this order — see the module docstring.
ATTESTATIONS = (CHECKSUMS_FILE, MANIFEST_FILE)
BUNDLES = (SHIPPED_BUNDLE, SHIPPED_PACKAGE)


def assert_no_stale_promotions(parent, root):
    """Fail-closed on a scratch left by an earlier interrupted promotion.

    Nothing here deletes it: it may hold the ONLY copy of a previous publication,
    so removing it is not this script's call — the same judgement
    ``assert_no_os_sidecars`` and ``assert_no_stale_publication_temps`` make.

    That is also why the run must STOP rather than proceed past it, which is the
    correction this replaced. Warning and continuing destroys precisely what the
    no-delete rule was protecting: a promotion killed after only
    ``CHECKSUMS.sha256`` moved aside leaves the manifest and both prior bundles
    still in the root, so a RETRY sweeps THOSE into its own scratch and deletes
    that scratch when it succeeds. The first scratch is then holding one checksums
    file, the second is gone, and the previous publication cannot be reconstructed
    from either. Refusing keeps every piece of it somewhere while a human decides
    which pieces to put back.

    A leftover cannot be adopted or collided with — every promotion mints a unique
    name — so this is not a lock. It is a refusal to start a second transaction
    while the first one is still half-applied on disk.

    The remedies are built with ``shlex.join``, absolute, and ``--``-guarded, for
    the reason ``assert_no_stale_publication_temps`` spells out: a path is data and
    a command line is code, and a leading-dash path round-trips through quoting
    perfectly while still being read as an option."""
    stale = sorted(e for e in os.listdir(parent) if e.startswith(PROMOTE_PREFIX))
    if not stale:
        return
    paths = [os.path.abspath(os.path.join(parent, e)) for e in stale]
    raise SystemExit(
        f"INTERRUPTED PROMOTION under {parent} — refusing to start another one:\n"
        + "".join(f"  {p}\n" for p in paths)
        + f"  A previous run was killed mid-promotion. That scratch holds what it had\n"
        f"  already moved out of {root}, and for those objects it is the ONLY copy.\n"
        f"  A second promotion would move the REST of the publication into a new\n"
        f"  scratch and delete that one on success, leaving the previous publication\n"
        f"  unreconstructible. Nothing here removes anything.\n"
        f"  Inspect it, put back whatever belongs in the artifact root, and only then\n"
        f"  remove it:\n"
        f"    {shlex.join(['ls', '-lR', '--', *paths])}\n"
        f"    {shlex.join(['rm', '-rf', '--', *paths])}"
    )


def dangling_component(path):
    """The outermost component of ``path`` that exists as a link but does not
    resolve, or ``None`` when every component resolves.

    ``lexists`` (does not follow the link) against ``exists`` (does) is the test.
    The WALK is what makes it complete: a single ``lexists``/``exists`` pair on
    the whole path answers only for the LEAF, because traversing to the leaf
    THROUGH a dangling ancestor already fails, so ``lexists`` on the leaf is false
    and an ancestor reads exactly like an absent first run.

    Components are tested as written. Normalising first — ``normpath``,
    ``abspath`` — would collapse ``a/link/../b`` to ``a/b`` and hide a dangling
    ``a/link`` that ``realpath`` flattens just the same, since it substitutes the
    link's target and only then applies the ``..``."""
    prefix = os.sep if path.startswith(os.sep) else ""
    for part in path.split(os.sep):
        if not part:
            continue
        prefix = os.path.join(prefix, part) if prefix else part
        if os.path.lexists(prefix) and not os.path.exists(prefix):
            return prefix
    return None


def _restore(done):
    """Undo the completed promotion phases, newest first, STOPPING at the first
    failure.

    The order is the mirror of the promotion and it is the whole guarantee: the
    new bundles come out of the root, then the prior bundles go back, and the
    attestations go back LAST — so no window this process can be killed in has an
    attestation in the root ahead of the bytes it describes. Each group is flushed
    before the next starts, on the best-effort terms ``fsync_dir`` states; the
    ordering guarantee itself is against the PROCESS stopping, not against a power
    cut.

    Which is why this stops. Continuing past a failed undo is how a rollback
    republishes an attestation over a payload that never made it back: the payload
    restore fails, the loop carries on, and ``CHECKSUMS.sha256`` and
    ``MANIFEST.json`` land back in a root whose bundle is still in the scratch.
    That is the invariant this module exists to hold, broken by its own recovery
    path. Whatever is left un-restored stays in the scratch, which the caller then
    preserves.

    ``done`` is the phases that were STARTED, each carrying only the renames that
    actually landed, so a phase interrupted halfway undoes exactly its landed
    half. Returns ``[]`` when every undo and every flush succeeded — the only
    case in which the scratch is safe to delete."""
    for label, applied, flush in reversed(done):
        for src, dst in reversed(applied):
            try:
                os.replace(dst, src)
            except OSError as e:
                return [f"while undoing '{label}': {dst} -> {src}: {e}"]
        for d in flush:
            try:
                fsync_dir(d)
            except OSError as e:
                return [f"while undoing '{label}': flushing {d}: {e}"]
    return []


def main():
    stage = os.path.abspath(stage_dir())
    # ``realpath``, not ``abspath``, and the reason is EXDEV rather than style:
    # every promotion below is a rename, and a rename cannot cross a filesystem.
    # ``abspath`` is purely lexical, so if the root's leaf is a symlink onto
    # another volume its lexical parent is the directory holding the LINK — on
    # the wrong filesystem — and every ``os.replace`` fails with
    # ``OSError(18, 'Cross-device link')``. Resolving first makes the scratch a
    # sibling of the REAL directory, hence always on its filesystem. The ``cp -R``
    # this replaced worked in that layout, so this is the difference between a
    # capability regression and none. (A symlink further up the path was never
    # affected: ``mkdtemp`` opens through it onto the target volume anyway.)
    link = model_root()
    root = os.path.realpath(link)
    parent = os.path.dirname(root)
    # Resolving first has one consequence that must be refused rather than acted
    # on. Two states both read as "the root is not there", and only one of them
    # is ordinary: a root that is ABSENT is a first run and ``makedirs`` must
    # create it, while a root reached THROUGH a link whose target does not exist
    # is a volume that did not mount. Under ``abspath`` the second failed by
    # itself — ``makedirs`` stats through the dangling link and raises. Under
    # ``realpath`` it becomes a plain non-existent path whose PARENTS are
    # creatable, so ``makedirs`` silently materialises the whole resolved chain
    # and the run publishes into it. macOS ``/Volumes`` is writable, so an
    # unmounted external drive turns into a real directory on the BOOT disk that
    # the true volume shadows the instant it mounts: the artifact is stranded on
    # the wrong filesystem, invisible, and every later run republishes into the
    # phantom.
    #
    # ANY component can be that link, not just the leaf, and both leak the same
    # way — measured rather than assumed. A dangling ``$GRANITE_MODELS_OUT``
    # refused itself with ENOENT under ``abspath``; under ``realpath`` it too
    # flattens into a fresh path with creatable parents and ``makedirs`` builds
    # the chain. Pointing a whole Models tree at an external drive is if anything
    # a likelier layout than symlinking one model subdirectory, so the ancestor is
    # not the lesser case. It cannot be caught by any single-path test — see
    # ``dangling_component`` — which is why the check walks every component. This
    # is the recipe's only creation point for the root: verify_granite.py and
    # write_manifest.py read ``model_root()`` and never create it.
    dangling = dangling_component(link)
    if dangling is not None:
        raise SystemExit(
            f"DANGLING ARTIFACT ROOT: {link}\n"
            f"  its component {dangling}\n"
            f"  is a link to {os.readlink(dangling)}, which does not exist.\n"
            f"  Most often the volume holding it is not mounted. Creating the path now\n"
            f"  would publish onto whichever filesystem happens to hold the resolved\n"
            f"  chain — {root} — and the real target would shadow those bytes the\n"
            f"  moment it mounts.\n"
            f"  Mount the volume (or repoint the link), then re-run."
        )
    os.makedirs(root, exist_ok=True)
    assert_no_stale_promotions(parent, root)

    # ---- preflight: the sources must exist and hold files ---------------------
    for sub in BUNDLES:
        src = os.path.join(stage, sub)
        if not os.path.isdir(src):
            raise SystemExit(
                f"MISSING STAGING BUILD: {src}\n"
                f"  Steps 1-2 did not produce it; refusing to touch {root}."
            )
        if not digest_tree(src):
            raise SystemExit(
                f"EMPTY STAGING BUILD: {src} holds no files.\n"
                f"  Refusing to publish it over {root}."
            )

    promote = tempfile.mkdtemp(prefix=PROMOTE_PREFIX, dir=parent)
    # False while the scratch holds nothing the root does not still hold: a
    # failure before phase 1 must NOT leave a scratch behind, or
    # ``assert_no_stale_promotions`` refuses the very retry that would fix it.
    keep_scratch = False
    try:
        new = os.path.join(promote, "new")
        aside = os.path.join(promote, "aside")
        os.makedirs(new)
        os.makedirs(aside)

        # ---- copy into scratch, with the publication still whole --------------
        # ``symlinks=True`` preserves what the `cp -R` this replaced did:
        # copytree's DEFAULT follows symlinks and writes their targets as regular
        # files, which would silently change the shape of the published bundle.
        # `digest_tree` hashes contents, so it would not notice — the difference
        # is structural, not in the bytes.
        for sub in BUNDLES:
            shutil.copytree(os.path.join(stage, sub), os.path.join(new, sub),
                            symlinks=True)

        # ---- validate the copies against their sources -----------------------
        # Full relative-path -> sha256 maps: every file present, same paths, same
        # bytes. This does NOT cover file modes, xattrs, symlinks, mtimes, or
        # empty directories, and it says nothing about whether the SOURCE is
        # correct — verify_granite.py is what gates that, in step 5.
        for sub in BUNDLES:
            want = digest_tree(os.path.join(stage, sub))
            got = digest_tree(os.path.join(new, sub))
            if got != want:
                raise SystemExit(
                    f"COPY DIVERGED for {sub}: the copy under {new} is not the staging build.\n"
                    f"  Nothing in {root} has been touched. Re-run steps 1-2."
                )

        # ---- phase 0: flush the scratch BEFORE the root is touched ------------
        # ``fsync_tree`` covers the copied files and the directories under ``new``;
        # ``promote`` carries the ``new`` and ``aside`` entries and ``parent``
        # carries ``promote``. Best-effort, per ``fsync_dir``: it bounds what a
        # kernel panic can lose from the one place the objects leaving the root are
        # about to live, and it is not a power-loss guarantee.
        fsync_tree(new)
        fsync_dir(promote)
        fsync_dir(parent)
        print(f"[ok] staged copies validated against {stage} ({len(BUNDLES)} bundles)")

        # ---- promote: three phases, each flushed before the next begins -------
        # The last element marks the sources that may legitimately be absent: a
        # first run has no prior publication to move aside. The new bundles were
        # just copied and digest-validated, so one missing THERE is a fault and
        # must raise rather than be skipped into a half-empty publication.
        phases = (
            ("attestations aside",
             [(os.path.join(root, n), os.path.join(aside, n)) for n in ATTESTATIONS],
             (root, aside), True),
            ("prior bundles aside",
             [(os.path.join(root, n), os.path.join(aside, n)) for n in BUNDLES],
             (root, aside), True),
            ("new bundles into the root",
             [(os.path.join(new, n), os.path.join(root, n)) for n in BUNDLES],
             (root, new), False),
        )
        # Set BEFORE the first root mutation, not after the rollback returns. The
        # window between them is where an interrupt — or a raise from inside the
        # rollback itself — reaches the `finally` below, and the flag decides
        # whether that deletes the only copy of the previous publication.
        keep_scratch = True
        done = []
        try:
            for label, moves, flush, optional in phases:
                applied = []
                done.append((label, applied, flush))
                for src, dst in moves:
                    if optional and not os.path.lexists(src):
                        continue
                    os.replace(src, dst)
                    applied.append((src, dst))
                for d in flush:
                    fsync_dir(d)
        except BaseException:
            failed = _restore(done)
            if failed:
                print(f"[FATAL] promotion failed AND rollback could not finish: {failed}\n"
                      f"        It STOPPED there rather than put the attestations back over\n"
                      f"        a payload that did not return. {root} is mid-promotion, and\n"
                      f"        what is missing from it is under {aside}, which has NOT been\n"
                      f"        deleted. Put it back by hand, then remove the scratch — the\n"
                      f"        next run refuses while it is there.", file=sys.stderr)
            else:
                keep_scratch = False
                print("[ok] promotion failed; the previous publication was restored intact",
                      file=sys.stderr)
            raise
        keep_scratch = False

        print(f"[ok] promoted {', '.join(BUNDLES)} into {root}; "
              f"prior attestations invalidated with the bytes they described")
    finally:
        if not keep_scratch:
            shutil.rmtree(promote, ignore_errors=True)
            # The removal is a change to ``parent``. Flushing it keeps a kernel
            # panic from resurrecting a scratch that describes nothing — which the
            # next run would refuse on, unable to tell it from a real one. Same
            # best-effort scope as every other flush here. The error is swallowed
            # because this is a ``finally``: every path that reaches it left the
            # root fully promoted, fully restored, or never touched, so raising
            # here would either fail a completed promotion or replace the
            # exception already on its way out.
            try:
                fsync_dir(parent)
            except OSError:
                pass


if __name__ == "__main__":
    main()
