"""Staging verifies for itself: one process, no stored report.

``stage_manifest.stage_package`` copies the package, attests the COPY
(``_attest.attest``: the checks, token parity on the CPU road and the GPU road
for every window, a digest before and after), digests it again just before
promotion and writes ``CHECKSUMS.sha256`` from the digests the attestation ran
on. Nothing here runs Core ML or torch: the predictors, the reference and the
checks are fakes; the logic under test is the real one.

Run (stdlib ``unittest`` and numpy; no model, no venv):

    python3 -m unittest discover -s coremlit/conversion/embeddinggemma2/tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")
sys.path.insert(0, _SCRIPTS)

import _attest  # noqa: E402
import _eg2_common  # noqa: E402
import stage_manifest  # noqa: E402

FILES = {
    "Manifest.json": b"manifest",
    "Data/com.apple.CoreML/model.mlmodel": b"graph",
    "Data/com.apple.CoreML/weights/weight.bin": b"weights",
}
PACKAGE = _eg2_common.FP32_AUDIO_PACKAGE


def write_package(root, files=FILES):
    package = os.path.join(root, PACKAGE)
    for rel, body in files.items():
        path = os.path.join(package, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
    return package


def windows():
    return [
        {"clip": "jfk", "start": 0, "samples": 160000, "tokens": 4},
        {"clip": "sine440", "start": 0, "samples": 160000, "tokens": 4},
    ]


def tokens(scale=1.0):
    base = np.linspace(0.5, 1.5, 32, dtype=np.float32).reshape(4, 8)
    return base * scale


def reference(window):
    return tokens()


def predictors(bad_road=None, mutate=None):
    """Both roads return the reference's tokens, except ``bad_road``, whose
    first token points the other way. ``mutate(path)`` runs on the first call
    (a swap of bytes while the package is being attested)."""

    def build(path):
        state = {"called": False}

        def predictor(road):
            def predict(window):
                if mutate is not None and not state["called"]:
                    state["called"] = True
                    mutate(path)
                out = tokens()
                if road == bad_road:
                    out = out.copy()
                    out[0] = -out[0]
                return out

            return predict

        return {road: predictor(road) for road in _eg2_common.VERIFY_ROADS}

    return build


def attestation(package_checks=(), **kw):
    build = kw.pop("build", predictors())
    win = kw.pop("windows", windows())

    def run(temp):
        return _attest.attest(
            temp,
            windows=win,
            build_predictors=build,
            reference=reference,
            checks=list(package_checks),
        )

    return run


class Staging(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="eg2-stage-test-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.stage = os.path.join(self.root, "stage")
        self.out = os.path.join(self.root, "out")
        os.makedirs(self.stage)
        self.source = write_package(self.stage)

    def staged_nothing(self):
        """A refusal leaves no destination and no sibling debris."""
        self.assertFalse(os.path.lexists(self.out), "a refused staging published a destination")
        debris = [n for n in os.listdir(self.root) if n.startswith(".")]
        self.assertEqual(debris, [], "a refused staging left debris beside it")

    def refused(self, attest_fn, **kw):
        with self.assertRaises(SystemExit) as caught:
            stage_manifest.stage_package(self.source, self.out, attest_fn, **kw)
        return str(caught.exception)

    def test_a_clean_package_is_staged_with_the_attested_digests(self):
        files = stage_manifest.stage_package(self.source, self.out, attestation())
        with open(os.path.join(self.out, "CHECKSUMS.sha256"), encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, [f"{sha}  ./{PACKAGE}/{rel}" for rel, sha in files])
        self.assertEqual([rel for rel, _ in files], sorted(FILES))

    def test_a_beta_less_layer_norm_is_refused(self):
        original = _eg2_common.package_layer_norms
        _eg2_common.package_layer_norms = lambda path: (2, ["var_160"])
        self.addCleanup(setattr, _eg2_common, "package_layer_norms", original)
        message = self.refused(attestation([("layer_norm_operations", _eg2_common.require_explicit_beta)]))
        self.assertIn("have no beta (var_160)", message)
        self.staged_nothing()

    def test_a_failing_cpu_road_is_refused_by_road(self):
        message = self.refused(attestation(build=predictors(bad_road="CPU_ONLY")))
        self.assertIn("(CPU_ONLY)", message)
        self.assertNotIn("(CPU_AND_GPU)", message)
        self.staged_nothing()

    def test_a_failing_gpu_road_is_refused_by_road(self):
        self.assertIn("(CPU_AND_GPU)", self.refused(attestation(build=predictors(bad_road="CPU_AND_GPU"))))
        self.staged_nothing()

    def test_predictors_for_the_wrong_roads_are_refused(self):
        def one_road(path):
            return {"CPU_AND_GPU": lambda window: tokens()}

        self.assertIn("not ('CPU_ONLY', 'CPU_AND_GPU')", self.refused(attestation(build=one_road)))
        self.staged_nothing()

    def test_no_window_is_refused_as_vacuous(self):
        self.assertIn("vacuously", self.refused(attestation(windows=[])))
        self.staged_nothing()

    def test_a_stale_or_forged_verify_json_changes_nothing(self):
        forged = {"passed": True, "compute_units": ["CPU_ONLY", "CPU_AND_GPU"], "package_digest": "0" * 64}
        # beside the package, where the recipe writes it (inside it would be a
        # stray file, refused for that reason instead)
        for directory in (self.stage, self.root):
            with open(os.path.join(directory, "verify.json"), "w", encoding="utf-8") as f:
                json.dump(forged, f)
        # a forged pass beside a failing package does not rescue it ...
        self.refused(attestation(build=predictors(bad_road="CPU_ONLY")))
        self.staged_nothing()

    def test_staging_never_reads_a_report(self):
        with open(os.path.join(_SCRIPTS, "stage_manifest.py"), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("VERIFY_FILE", text)
        self.assertNotIn("json.load", text.split("def main")[0])
        self.assertNotIn("require_current_verification", text)

    def test_bytes_swapped_while_the_package_is_attested_are_refused(self):
        def swap(path):
            with open(os.path.join(path, "Data/com.apple.CoreML/weights/weight.bin"), "wb") as f:
                f.write(b"swapped")

        message = self.refused(attestation(build=predictors(mutate=swap)))
        self.assertIn("changed while it was being attested", message)
        self.staged_nothing()

    def test_bytes_swapped_between_the_attestation_and_the_write_are_refused(self):
        def swap(temp):
            with open(os.path.join(temp, "Data/com.apple.CoreML/weights/weight.bin"), "wb") as f:
                f.write(b"swapped")

        message = self.refused(attestation(), after_attest=swap)
        self.assertIn("changed after they were attested", message)
        self.staged_nothing()

    def test_a_stray_file_is_refused(self):
        with open(os.path.join(self.source, "stray.bin"), "wb") as f:
            f.write(b"x")
        self.assertIn("not the pinned set", self.refused(attestation()))
        self.staged_nothing()


class PublicationIsOneExclusiveRename(unittest.TestCase):
    """Codex r4 and r5: the destination is published whole by ONE exclusive
    rename and never replaced, so there is no swap, backup, rollback or lock to
    get wrong."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="eg2-publish-test-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.stage = os.path.join(self.root, "stage")
        os.makedirs(self.stage)
        self.source = write_package(self.stage)
        self.out = os.path.join(self.root, "out")

    def debris(self):
        return sorted(name for name in os.listdir(self.root) if name.startswith("."))

    def snapshot(self, directory):
        entries = {}
        for here, _, names in os.walk(directory):
            for name in names:
                path = os.path.join(here, name)
                with open(path, "rb") as f:
                    entries[os.path.relpath(path, directory)] = f.read()
        return entries

    def inject_at_rename(self, make_failure):
        calls = {"reached": 0}
        real = stage_manifest.rename_exclusive

        def rename(source, destination):
            calls["reached"] += 1
            make_failure(real, source, destination)

        stage_manifest.rename_exclusive = rename
        self.addCleanup(setattr, stage_manifest, "rename_exclusive", real)
        return calls

    def test_a_clean_staging_publishes_the_package_and_its_checksums_whole(self):
        files = stage_manifest.stage_package(self.source, self.out, attestation())
        self.assertEqual(sorted(os.listdir(self.out)), ["CHECKSUMS.sha256", PACKAGE])
        self.assertEqual(self.debris(), [])
        with open(os.path.join(self.out, "CHECKSUMS.sha256"), encoding="utf-8") as f:
            self.assertEqual(f.read().splitlines(), [f"{sha}  ./{PACKAGE}/{rel}" for rel, sha in files])

    def test_an_existing_destination_is_refused_by_name_and_left_untouched(self):
        stage_manifest.stage_package(self.source, self.out, attestation())
        before = self.snapshot(self.out)
        for build in (attestation(), attestation(build=predictors(bad_road="CPU_ONLY"))):
            with self.assertRaises(SystemExit) as caught:
                stage_manifest.stage_package(self.source, self.out, build)
            self.assertIn("already exists", str(caught.exception))
            self.assertIn("mv '", str(caught.exception))
        self.assertEqual(self.snapshot(self.out), before)
        self.assertEqual(self.debris(), [])

    def test_an_existing_empty_directory_is_refused_too(self):
        os.makedirs(self.out)
        with self.assertRaises(SystemExit) as caught:
            stage_manifest.stage_package(self.source, self.out, attestation())
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(os.listdir(self.out), [])

    def test_the_kernel_refuses_a_rename_onto_an_existing_empty_directory(self):
        # the property the design rests on: a plain rename would replace it
        src = os.path.join(self.root, "src")
        os.makedirs(src)
        os.makedirs(self.out)
        with self.assertRaises(FileExistsError):
            stage_manifest.rename_exclusive(src, self.out)
        self.assertTrue(os.path.isdir(src))

    def test_an_alias_of_an_existing_destination_is_refused(self):
        stage_manifest.stage_package(self.source, self.out, attestation())
        alias = os.path.join(self.root, "alias")
        os.symlink(self.out, alias)
        with self.assertRaises(SystemExit) as caught:
            stage_manifest.stage_package(self.source, alias, attestation())
        self.assertIn("already exists", str(caught.exception))

    def test_two_concurrent_stagings_publish_exactly_one(self):
        import threading

        barrier = threading.Barrier(2)
        outcomes = []

        def run(label):
            def together(package):
                barrier.wait(timeout=20)  # both attested, both about to publish

            source = write_package(os.path.join(self.root, f"stage-{label}"), {**FILES, "Manifest.json": label.encode()})
            try:
                stage_manifest.stage_package(source, self.out, attestation(), after_attest=together)
                outcomes.append((label, "published"))
            except SystemExit as refusal:
                outcomes.append((label, str(refusal)))

        threads = [threading.Thread(target=run, args=(label,)) for label in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        published = [label for label, result in outcomes if result == "published"]
        refused = [result for _, result in outcomes if result != "published"]
        self.assertEqual(len(published), 1, outcomes)
        self.assertEqual(len(refused), 1, outcomes)
        self.assertIn("already exists", refused[0])
        # whole: the checksums are those of the package that won
        with open(os.path.join(self.out, "CHECKSUMS.sha256"), encoding="utf-8") as f:
            listed = f.read()
        with open(os.path.join(self.out, PACKAGE, "Manifest.json"), "rb") as f:
            self.assertEqual(f.read(), published[0].encode())
        self.assertIn(_eg2_common.sha256_bytes(published[0].encode()), listed)
        self.assertEqual(self.debris(), [])

    def test_a_failure_at_the_rename_leaves_no_destination(self):
        def fail(real, source, destination):
            raise OSError(5, "injected rename failure")

        calls = self.inject_at_rename(fail)
        with self.assertRaises(OSError) as caught:
            stage_manifest.stage_package(self.source, self.out, attestation())
        self.assertEqual(calls["reached"], 1, "the injected failure was never reached")
        self.assertIn("injected rename failure", str(caught.exception))
        self.assertFalse(os.path.lexists(self.out))
        self.assertEqual(self.debris(), [])

    def test_an_interrupt_before_the_rename_leaves_no_destination(self):
        def interrupt(package):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            stage_manifest.stage_package(self.source, self.out, attestation(), after_attest=interrupt)
        self.assertFalse(os.path.lexists(self.out))
        self.assertEqual(self.debris(), [])

    def test_an_interrupt_at_the_rename_leaves_no_destination(self):
        def interrupt(real, source, destination):
            raise KeyboardInterrupt

        calls = self.inject_at_rename(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            stage_manifest.stage_package(self.source, self.out, attestation())
        self.assertEqual(calls["reached"], 1)
        self.assertFalse(os.path.lexists(self.out))
        self.assertEqual(self.debris(), [])

    def test_a_failed_checksum_write_and_a_failed_copy_leave_no_debris(self):
        real_fsync, real_copy = os.fsync, shutil.copytree

        def fsync(descriptor):
            raise OSError("injected fsync failure")

        os.fsync = fsync
        self.addCleanup(setattr, os, "fsync", real_fsync)
        with self.assertRaises(OSError):
            stage_manifest.stage_package(self.source, self.out, attestation())
        os.fsync = real_fsync

        def copytree(source, target, *args, **kwargs):
            os.makedirs(target)
            raise OSError("injected copy failure")

        shutil.copytree = copytree
        self.addCleanup(setattr, shutil, "copytree", real_copy)
        with self.assertRaises(OSError):
            stage_manifest.stage_package(self.source, self.out, attestation())
        self.assertFalse(os.path.lexists(self.out))
        self.assertEqual(self.debris(), [])

    def test_read_only_directories_in_a_failed_generation_are_still_removed(self):
        import stat

        def lock_down(package):
            os.chmod(os.path.join(package, "Data"), stat.S_IRUSR | stat.S_IXUSR)
            raise OSError("fail after making a directory read-only")

        with self.assertRaises(OSError):
            stage_manifest.stage_package(self.source, self.out, attestation(), after_attest=lock_down)
        self.assertEqual(self.debris(), [])


class TheAttestationItself(unittest.TestCase):
    def test_every_window_is_judged_on_both_roads(self):
        with tempfile.TemporaryDirectory() as root:
            package = write_package(root)
            result = _attest.attest(
                package,
                windows=windows(),
                build_predictors=predictors(),
                reference=reference,
                checks=[],
            )
        seen = {(w["clip"], w["start"], w["road"]) for w in result["windows"]}
        self.assertEqual(
            seen, {(c, 0, r) for c in ("jfk", "sine440") for r in ("CPU_ONLY", "CPU_AND_GPU")}
        )


if __name__ == "__main__":
    unittest.main()
